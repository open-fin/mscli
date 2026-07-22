"""Unit tests for attribute_wait_times.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script


def test_no_collective_types(tmp_path: Path):
    """Should return unavailable when no collective types data."""
    out = tmp_path / "wait_times.json"
    run_script("attribute_wait_times.py", "--output-json", str(out))

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["wait_time_attribution_available"] is False


def test_syncbn_attribution(tmp_path: Path):
    """Attribute wait times with SyncBN dominant and jittery ranks."""
    collective_types = {
        "collective_type_analysis_available": True,
        "types": [
            {"type": "SyncBN", "total_time_ms": 500.0, "share_percent": 50.0, "count": 100, "avg_time_ms": 5.0, "avg_size_mb": 0.5, "sample_names": ["SyncBN"]},
            {"type": "GradientAllReduce", "total_time_ms": 400.0, "share_percent": 40.0, "count": 200, "avg_time_ms": 2.0, "avg_size_mb": 64.0, "sample_names": ["AllReduce"]},
        ],
        "dominant_type": "SyncBN",
        "syncbn_dominant": True,
        "syncbn_share_percent": 50.0,
        "total_collective_time_ms": 900.0,
        "total_collective_count": 300,
    }
    ct_path = tmp_path / "collective_types.json"
    ct_path.write_text(json.dumps(collective_types), encoding="utf-8")

    rank_variance = {
        "rank_variance_analysis_available": True,
        "jittery_ranks": [3],
        "worst_jittery_rank": 3,
        "worst_rank_cv": 0.25,
        "drag_effect_ms": 50.0,
        "stable_ranks": [0, 1, 2, 4],
        "per_rank_stats": {"3": {"mean_ms": 120, "cv": 0.25}},
    }
    rv_path = tmp_path / "rank_variance.json"
    rv_path.write_text(json.dumps(rank_variance), encoding="utf-8")

    step = {"average_step_time_ms": 200.0}
    step_path = tmp_path / "step.json"
    step_path.write_text(json.dumps(step), encoding="utf-8")

    out = tmp_path / "wait_times.json"
    run_script(
        "attribute_wait_times.py",
        "--collective-types-json", str(ct_path),
        "--rank-variance-json", str(rv_path),
        "--step-json", str(step_path),
        "--output-json", str(out),
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["wait_time_attribution_available"] is True
    assert result["total_wait_estimated_ms"] == 50.0
    assert result["primary_wait_source"] == "SyncBN"
    assert result["elimination_savings_ms"] == 25.0

    syncbn_attr = next(a for a in result["attributions"] if a["collective_type"] == "SyncBN")
    assert syncbn_attr["is_primary"] is True


def test_no_wait_estimated(tmp_path: Path):
    """When no drag effect or wait ratio, should return unavailable."""
    collective_types = {
        "collective_type_analysis_available": True,
        "types": [
            {"type": "AllReduce", "total_time_ms": 100.0, "share_percent": 100.0, "count": 10, "avg_time_ms": 10.0, "avg_size_mb": None, "sample_names": ["AllReduce"]},
        ],
        "dominant_type": "AllReduce",
        "syncbn_dominant": False,
        "syncbn_share_percent": 0.0,
        "total_collective_time_ms": 100.0,
        "total_collective_count": 10,
    }
    ct_path = tmp_path / "collective_types.json"
    ct_path.write_text(json.dumps(collective_types), encoding="utf-8")

    out = tmp_path / "wait_times.json"
    run_script(
        "attribute_wait_times.py",
        "--collective-types-json", str(ct_path),
        "--output-json", str(out),
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["wait_time_attribution_available"] is False
