"""Unit tests for analyze_rank_variance.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script


def _write_rank_dir(root: Path, rank_id: int, step_times: list[float]) -> None:
    """Create a rank subdirectory with profiler_info.json and step_trace_time.csv.

    Uses directory name containing rank_id so find_rank_dirs can discover it.
    """
    rank_dir = root / f"rank_{rank_id}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    profiler_dir = rank_dir / "ASCEND_PROFILER_OUTPUT"
    profiler_dir.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(f"{i},{t}" for i, t in enumerate(step_times))
    (profiler_dir / "step_trace_time.csv").write_text(
        f"Step,Step Time (ms)\n{rows}", encoding="utf-8"
    )
    (rank_dir / "profiler_info.json").write_text(
        json.dumps({"rank_id": rank_id}), encoding="utf-8"
    )


def test_single_rank_unavailable(tmp_path: Path):
    """Single rank should return available=False."""
    _write_rank_dir(tmp_path, 0, [100.0, 102.0, 98.0])

    out = tmp_path / "rank_variance.json"
    run_script("analyze_rank_variance.py", "--trace-root", str(tmp_path), "--output-json", str(out))

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["rank_variance_analysis_available"] is False


def test_jittery_rank_detection(tmp_path: Path):
    """Detect jittery rank among stable ranks."""
    # Stable rank 0
    _write_rank_dir(tmp_path, 0, [100.0] * 20)

    # Jittery rank 1: high variance
    values = [100.0] * 10 + [200.0] * 5 + [50.0] * 5
    _write_rank_dir(tmp_path, 1, values)

    out = tmp_path / "rank_variance.json"
    run_script("analyze_rank_variance.py", "--trace-root", str(tmp_path), "--output-json", str(out))

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["rank_variance_analysis_available"] is True
    # Rank 1 should be detected as jittery
    assert result.get("worst_jittery_rank") is not None or result.get("worst_rank_cv", 0) > 0.10


def test_all_stable_ranks(tmp_path: Path):
    """No jittery ranks when all ranks are stable."""
    for rank_id in range(3):
        _write_rank_dir(tmp_path, rank_id, [100.0] * 20)

    out = tmp_path / "rank_variance.json"
    run_script("analyze_rank_variance.py", "--trace-root", str(tmp_path), "--output-json", str(out))

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["rank_variance_analysis_available"] is True
    assert result["jittery_ranks"] == []
