"""Unit tests for correlate_slow_rank_ops.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script


def test_no_rank_data(tmp_path: Path):
    """No rank variance or cluster data should return unavailable."""
    out = tmp_path / "slow_rank_ops.json"
    run_script("correlate_slow_rank_ops.py", "--trace-root", str(tmp_path), "--output-json", str(out))

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["slow_rank_op_analysis_available"] is False


def test_no_jittery_ranks(tmp_path: Path):
    """When rank variance shows no jittery ranks and no cluster slow_ranks."""
    rank_variance = {
        "rank_variance_analysis_available": True,
        "jittery_ranks": [],
        "stable_ranks": [0, 1],
    }
    rv_path = tmp_path / "rank_variance.json"
    rv_path.write_text(json.dumps(rank_variance), encoding="utf-8")

    out = tmp_path / "slow_rank_ops.json"
    run_script(
        "correlate_slow_rank_ops.py",
        "--trace-root", str(tmp_path),
        "--rank-variance-json", str(rv_path),
        "--output-json", str(out),
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["slow_rank_op_analysis_available"] is False


def test_with_cluster_slow_ranks(tmp_path: Path):
    """Use cluster slow_ranks when no jittery ranks available."""
    # Create rank directories
    for rank_id in range(2):
        r = tmp_path / f"rank_{rank_id}"
        r.mkdir()
        (r / "profiler_info.json").write_text(f'{{"rank_id": {rank_id}}}', encoding="utf-8")

    cluster = {"slow_ranks": [1]}
    cluster_path = tmp_path / "cluster.json"
    cluster_path.write_text(json.dumps(cluster), encoding="utf-8")

    out = tmp_path / "slow_rank_ops.json"
    run_script(
        "correlate_slow_rank_ops.py",
        "--trace-root", str(tmp_path),
        "--cluster-json", str(cluster_path),
        "--output-json", str(out),
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    # Will be unavailable because no op_summary CSVs, but should reach the comparison stage
    assert result["slow_rank_op_analysis_available"] is False
    assert "Operator profile" in result.get("reason", "")
