"""Unit tests for analyze_jitter.py."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, run_script, ROOT, SCRIPTS


def test_jitter_from_stable_step_data(tmp_path: Path):
    profiler_root = write_sample_profiler_export(tmp_path)
    step_json = tmp_path / "step.json"
    run_script("summarize_step_breakdown.py", "--trace-root", str(profiler_root), "--output-json", str(step_json))

    jitter_json = tmp_path / "jitter.json"
    run_script("analyze_jitter.py", "--step-json", str(step_json), "--output-json", str(jitter_json))

    result = json.loads(jitter_json.read_text(encoding="utf-8"))
    assert result["step_time_jitter"]["status"] in ("normal", "warning", "critical")
    assert result["step_time_jitter"]["cv"] is not None


def test_jitter_from_variable_step_data(tmp_path: Path):
    profiler_root = tmp_path / "var_run_ascend_ms"
    ascend = profiler_root / "ASCEND_PROFILER_OUTPUT"
    ascend.mkdir(parents=True)
    (ascend / "step_trace_time.csv").write_text(
        "Step ID,ComputeTime(ms),StepTime(ms)\n"
        + "\n".join(f"{i},{100 + (i % 5) * 80},{120 + (i % 5) * 90}" for i in range(20)) + "\n",
        encoding="utf-8",
    )

    step_json = tmp_path / "step_var.json"
    run_script("summarize_step_breakdown.py", "--trace-root", str(profiler_root), "--output-json", str(step_json))

    jitter_json = tmp_path / "jitter.json"
    run_script("analyze_jitter.py", "--step-json", str(step_json), "--output-json", str(jitter_json))

    result = json.loads(jitter_json.read_text(encoding="utf-8"))
    assert result["step_time_jitter"]["cv"] > 0.25


def test_jitter_with_cross_rank_skew(tmp_path: Path):
    cluster_root = tmp_path / "cluster_run"
    cluster_root.mkdir()
    for rank_id in range(3):
        rank_dir = cluster_root / f"rank_{rank_id}"
        ascend = rank_dir / "ASCEND_PROFILER_OUTPUT"
        ascend.mkdir(parents=True)
        (rank_dir / "profiler_info.json").write_text(json.dumps({"rank_id": rank_id}), encoding="utf-8")
        step_time = 100.0 + rank_id * 15
        (ascend / "step_trace_time.csv").write_text(
            "Step ID,StepTime(ms)\n" + "\n".join(f"{i},{step_time}" for i in range(5)) + "\n",
            encoding="utf-8",
        )

    step_json = tmp_path / "step.json"
    step_json.write_text(json.dumps({
        "steps_analyzed": 5, "average_step_time_ms": 100.0,
        "coefficient_of_variation": 0.05, "stage_totals_ms": {"compute": 60, "communication": 20},
    }), encoding="utf-8")

    jitter_json = tmp_path / "jitter.json"
    run_script("analyze_jitter.py", "--step-json", str(step_json), "--trace-root", str(cluster_root),
               "--output-json", str(jitter_json))

    result = json.loads(jitter_json.read_text(encoding="utf-8"))
    assert result["cross_rank_skew"] is not None
    assert result["cross_rank_skew"]["num_ranks"] == 3
    assert result["cross_rank_skew"]["max_skew_ms"] > 0


def test_jitter_no_step_json_exits_with_error(tmp_path: Path):
    jitter_json = tmp_path / "jitter.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "analyze_jitter.py"), "--output-json", str(jitter_json)],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
