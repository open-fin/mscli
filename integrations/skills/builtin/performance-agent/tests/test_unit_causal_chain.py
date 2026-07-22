"""Unit tests for build_causal_chain.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_no_bottlenecks(tmp_path: Path):
    """Empty bottlenecks should return unavailable."""
    bottlenecks = {"ranked_candidates": []}
    b_path = _write_json(tmp_path / "bottlenecks.json", bottlenecks)

    out = tmp_path / "causal_chain.json"
    run_script("build_causal_chain.py", "--bottlenecks-json", str(b_path), "--output-json", str(out))

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["causal_chain_available"] is False


def test_syncbn_jitter_chain(tmp_path: Path):
    """Full SyncBN + jitter causal chain should reach depth 4+."""
    bottlenecks = _write_json(tmp_path / "bottlenecks.json", {
        "primary_candidate": {"name": "communication", "confidence": 0.8},
        "ranked_candidates": [
            {"name": "communication", "confidence": 0.8},
            {"name": "jitter", "confidence": 0.5},
        ],
    })

    _write_json(tmp_path / "collective_types.json", {
        "collective_type_analysis_available": True,
        "syncbn_dominant": True,
        "syncbn_share_percent": 40.0,
        "types": [
            {"type": "SyncBN", "total_time_ms": 500.0, "share_percent": 40.0, "count": 100, "avg_time_ms": 5.0, "avg_size_mb": 0.5, "sample_names": ["SyncBN"]},
        ],
        "total_collective_time_ms": 500.0,
        "total_collective_count": 100,
    })

    _write_json(tmp_path / "rank_variance.json", {
        "rank_variance_analysis_available": True,
        "jittery_ranks": [3],
        "worst_jittery_rank": 3,
        "worst_rank_cv": 0.25,
        "drag_effect_ms": 55.0,
        "stable_ranks": [0, 1, 2, 4],
        "per_rank_stats": {"3": {"mean_ms": 120, "cv": 0.25}},
    })

    _write_json(tmp_path / "slow_rank_ops.json", {
        "slow_rank_op_analysis_available": True,
        "syncbn_divergence_detected": True,
        "primary_divergent_operator": "SyncBNBackwardReduce",
        "top_divergent_operators": [
            {"operator": "SyncBNBackwardReduce", "slowdown_ratio": 4.93, "category": "SyncBN"},
        ],
        "analyzed_rank": 3,
    })

    _write_json(tmp_path / "wait_attribution.json", {
        "wait_time_attribution_available": True,
        "primary_wait_source": "SyncBN",
        "elimination_savings_ms": 38.7,
    })

    _write_json(tmp_path / "communication.json", {
        "communication_pressure": "high",
        "total_time_ms": 1500.0,
    })

    _write_json(tmp_path / "jitter.json", {
        "step_time_jitter": {"cv": 0.20},
        "compute_jitter": {"cv": 0.15},
        "communication_jitter": {"cv": 0.05},
    })

    out = tmp_path / "causal_chain.json"
    run_script(
        "build_causal_chain.py",
        "--bottlenecks-json", str(tmp_path / "bottlenecks.json"),
        "--collective-types-json", str(tmp_path / "collective_types.json"),
        "--rank-variance-json", str(tmp_path / "rank_variance.json"),
        "--slow-rank-ops-json", str(tmp_path / "slow_rank_ops.json"),
        "--wait-attribution-json", str(tmp_path / "wait_attribution.json"),
        "--communication-json", str(tmp_path / "communication.json"),
        "--jitter-json", str(tmp_path / "jitter.json"),
        "--output-json", str(out),
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["causal_chain_available"] is True
    assert result["chains_count"] >= 1
    assert result["max_depth"] >= 4

    # Primary chain should be SyncBN+jitter (deepest)
    primary = result["primary_chain"]
    assert primary["depth"] >= 4
    assert "syncbn" in primary["root_cause"].lower()


def test_comm_overhead_chain(tmp_path: Path):
    """Communication overhead without SyncBN should produce chain."""
    bottlenecks = _write_json(tmp_path / "bottlenecks.json", {
        "primary_candidate": {"name": "communication", "confidence": 0.8},
        "ranked_candidates": [{"name": "communication", "confidence": 0.8}],
    })

    _write_json(tmp_path / "collective_types.json", {
        "collective_type_analysis_available": True,
        "syncbn_dominant": False,
        "syncbn_share_percent": 0.0,
        "types": [
            {"type": "SmallPacketAllReduce", "total_time_ms": 300.0, "share_percent": 30.0, "count": 1000, "avg_time_ms": 0.3, "avg_size_mb": 0.1},
        ],
    })

    out = tmp_path / "causal_chain.json"
    run_script(
        "build_causal_chain.py",
        "--bottlenecks-json", str(tmp_path / "bottlenecks.json"),
        "--collective-types-json", str(tmp_path / "collective_types.json"),
        "--output-json", str(out),
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["causal_chain_available"] is True
    assert any("small_bucket" in c["root_cause"] for c in result["chains"])
