"""Unit tests for detect_slow_ranks.py."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, run_script, ROOT, SCRIPTS


def _write_cluster_profiler(root: Path, num_ranks: int = 4, slow_rank: int = None) -> Path:
    cluster_root = root / "cluster_run"
    cluster_root.mkdir(parents=True, exist_ok=True)
    for rank_id in range(num_ranks):
        rank_dir = cluster_root / f"rank_{rank_id}"
        ascend = rank_dir / "ASCEND_PROFILER_OUTPUT"
        ascend.mkdir(parents=True, exist_ok=True)
        (rank_dir / "profiler_info.json").write_text(json.dumps({"rank_id": rank_id}), encoding="utf-8")
        step_time = 250.0 if (slow_rank is not None and rank_id == slow_rank) else 100.0
        compute_time = 160.0 if (slow_rank is not None and rank_id == slow_rank) else 50.0
        (ascend / "step_trace_time.csv").write_text(
            "Step ID,ComputeTime(ms),CommunicationTime(ms),IdleGap(ms),StepTime(ms)\n"
            + "\n".join(f"{i},{compute_time},20,10,{step_time}" for i in range(5)) + "\n",
            encoding="utf-8",
        )
    return cluster_root


def test_single_rank_returns_not_applicable(tmp_path: Path):
    profiler_root = write_sample_profiler_export(tmp_path)
    output_json = tmp_path / "cluster.json"
    run_script("detect_slow_ranks.py", "--trace-root", str(profiler_root), "--output-json", str(output_json))
    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["cluster_analysis_available"] is False


def test_multi_rank_detects_slow_ranks(tmp_path: Path):
    """4-card cluster where one rank is 250ms vs 100ms others.
    Sigma-based detection misses this (outlier inflates std), but the
    median-ratio fallback should catch it (250 > 1.5 * 100)."""
    cluster_root = _write_cluster_profiler(tmp_path, num_ranks=4, slow_rank=2)
    output_json = tmp_path / "cluster.json"
    run_script("detect_slow_ranks.py", "--trace-root", str(cluster_root), "--output-json", str(output_json))
    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["cluster_analysis_available"] is True
    assert result["total_ranks"] == 4
    assert 2 in result["slow_ranks"]
    assert result["analysis"]["bottleneck_type"] is not None


def test_multi_rank_no_outliers(tmp_path: Path):
    cluster_root = _write_cluster_profiler(tmp_path, num_ranks=4, slow_rank=None)
    output_json = tmp_path / "cluster.json"
    run_script("detect_slow_ranks.py", "--trace-root", str(cluster_root), "--output-json", str(output_json))
    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["slow_ranks"] == []
    assert result["fast_ranks"] == []


def test_sigma_inflated_but_median_catches_outlier(tmp_path: Path):
    """When 3 of 4 ranks are slow (majority), median-ratio should NOT flag them.

    This tests the false-positive guard: fewer than half must be flagged.
    """
    cluster_root = tmp_path / "cluster_run"
    cluster_root.mkdir(parents=True, exist_ok=True)
    for rid in range(4):
        rd = cluster_root / f"rank_{rid}"
        asc = rd / "ASCEND_PROFILER_OUTPUT"
        asc.mkdir(parents=True, exist_ok=True)
        (rd / "profiler_info.json").write_text(json.dumps({"rank_id": rid}), encoding="utf-8")
        # 3 ranks at 200, 1 rank at 100. Median=200.
        # The "slow" majority guard prevents false positives.
        step_time = 100.0 if rid == 0 else 200.0
        compute_time = 50.0 if rid == 0 else 120.0
        (asc / "step_trace_time.csv").write_text(
            "Step ID,ComputeTime(ms),CommunicationTime(ms),IdleGap(ms),StepTime(ms)\n"
            + "\n".join(f"{i},{compute_time},20,10,{step_time}" for i in range(5)) + "\n",
            encoding="utf-8",
        )

    output_json = tmp_path / "cluster.json"
    run_script("detect_slow_ranks.py", "--trace-root", str(cluster_root), "--output-json", str(output_json))
    result = json.loads(output_json.read_text(encoding="utf-8"))
    # Guard prevents flagging 3/4 ranks as slow (bimodal, not outliers)
    assert len(result["slow_ranks"]) == 0


def test_no_rank_dirs_exits_with_error(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output_json = tmp_path / "cluster.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "detect_slow_ranks.py"),
         "--trace-root", str(empty_dir), "--output-json", str(output_json)],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
