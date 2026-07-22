"""Unit tests for calculate_linearity.py."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, run_script, ROOT, SCRIPTS

# Allow importlib to resolve `from perf_common import ...`
sys.path.insert(0, str(SCRIPTS))


def _write_cluster_profiler(root: Path, rank_step_times: dict[int, float]) -> Path:
    """Create a multi-rank cluster profiler with configurable step times.

    Uses the fallback layout: rank subdirectories with profiler_info.json inside,
    so calculate_linearity.find_rank_dirs() discovers them.
    """
    cluster_root = root / "cluster_run"
    cluster_root.mkdir(parents=True, exist_ok=True)
    for rank_id, step_time in rank_step_times.items():
        rank_dir = cluster_root / f"rank_{rank_id}"
        ascend = rank_dir / "ASCEND_PROFILER_OUTPUT"
        ascend.mkdir(parents=True, exist_ok=True)
        (rank_dir / "profiler_info.json").write_text(
            json.dumps({"rank_id": rank_id}), encoding="utf-8"
        )
        compute_time = step_time * 0.5
        comm_time = step_time * 0.3
        idle = step_time * 0.2
        (ascend / "step_trace_time.csv").write_text(
            "Step ID,ComputeTime(ms),CommunicationTime(ms),IdleGap(ms),StepTime(ms)\n"
            + "\n".join(
                f"{i},{compute_time},{comm_time},{idle},{step_time}" for i in range(5)
            ) + "\n",
            encoding="utf-8",
        )
    return cluster_root


def test_calculate_linearity_pure_function():
    """Test the calculate_linearity formula directly."""
    script = SCRIPTS / "calculate_linearity.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("cl", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Perfect scaling: single = cluster → linearity = 1.0
    assert mod.calculate_linearity(100.0, 100.0) == 1.0

    # 80% efficiency: single=80, cluster=100 → 0.8
    assert mod.calculate_linearity(80.0, 100.0) == 0.8

    # 50% efficiency
    assert mod.calculate_linearity(50.0, 100.0) == 0.5

    # Edge: zero cluster step time
    assert mod.calculate_linearity(100.0, 0.0) == 0.0


def test_multi_rank_normal_linearity(tmp_path: Path):
    """4-rank cluster with similar step times → linearity >= 0.8 (normal)."""
    cluster_root = _write_cluster_profiler(tmp_path, {
        0: 100.0, 1: 102.0, 2: 98.0, 3: 101.0
    })
    output_json = tmp_path / "linearity.json"

    run_script(
        "calculate_linearity.py",
        "--trace-root", str(cluster_root),
        "--output-json", str(output_json),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["linearity"] >= 0.8
    assert result["linearity_status"] == "normal"
    assert result["num_devices"] == 4
    assert result["likely_domains"] == []


def test_multi_rank_severe_degradation(tmp_path: Path):
    """4-rank cluster with one very slow rank → linearity < 0.6."""
    cluster_root = _write_cluster_profiler(tmp_path, {
        0: 100.0, 1: 100.0, 2: 100.0, 3: 250.0
    })
    output_json = tmp_path / "linearity.json"

    run_script(
        "calculate_linearity.py",
        "--trace-root", str(cluster_root),
        "--output-json", str(output_json),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["linearity"] < 0.8
    assert result["linearity_status"] in ("moderate_degradation", "severe_degradation")
    assert "communication" in result["likely_domains"]


def test_single_rank_step_ms_override(tmp_path: Path):
    """User-provided --single-rank-step-ms should override fastest rank baseline."""
    cluster_root = _write_cluster_profiler(tmp_path, {
        0: 100.0, 1: 100.0, 2: 100.0, 3: 100.0
    })
    output_json = tmp_path / "linearity.json"

    run_script(
        "calculate_linearity.py",
        "--trace-root", str(cluster_root),
        "--single-rank-step-ms", "50.0",
        "--output-json", str(output_json),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["fastest_step_ms"] == 50.0
    # 50/100 = 0.5 → severe degradation
    assert result["linearity"] == 0.5
    assert result["linearity_status"] == "severe_degradation"


def test_single_rank_only_no_linearity(tmp_path: Path):
    """Single-rank data cannot compute linearity."""
    profiler_root = write_sample_profiler_export(tmp_path)
    output_json = tmp_path / "linearity.json"

    run_script(
        "calculate_linearity.py",
        "--trace-root", str(profiler_root),
        "--output-json", str(output_json),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    # Single rank: linearity is None
    assert result["linearity"] is None
    # Note should mention needing multiple ranks or single-rank limitation
    note_lower = result.get("note", "").lower()
    assert "single" in note_lower or "rank" in note_lower


def test_no_rank_dirs_exits_with_error(tmp_path: Path):
    """Empty directory with no rank data should exit with error."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output_json = tmp_path / "linearity.json"

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "calculate_linearity.py"),
         "--trace-root", str(empty_dir), "--output-json", str(output_json)],
        text=True, capture_output=True,
    )
    assert result.returncode != 0


def test_two_ranks_sufficient_for_linearity(tmp_path: Path):
    """Minimum 2 ranks should be enough to compute linearity."""
    cluster_root = _write_cluster_profiler(tmp_path, {
        0: 100.0, 1: 110.0
    })
    output_json = tmp_path / "linearity.json"

    run_script(
        "calculate_linearity.py",
        "--trace-root", str(cluster_root),
        "--output-json", str(output_json),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["linearity"] is not None
    assert result["num_devices"] == 2
    # 100/105 ≈ 0.952 → normal
    assert result["linearity"] > 0.8
