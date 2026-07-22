"""Unit tests for infer_root_cause.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script, write_sample_profiler_export


def _build_bottlenecks(tmp_path: Path) -> Path:
    """Create a sample bottlenecks JSON."""
    bottlenecks_json = tmp_path / "bottlenecks.json"
    bottlenecks_json.write_text(json.dumps({
        "primary_candidate": {
            "name": "communication",
            "confidence": 0.8,
            "evidence": ["comm_ratio=0.45"],
        },
        "ranked_candidates": [
            {"name": "communication", "confidence": 0.8, "evidence": ["comm_ratio=0.45"]},
            {"name": "low_mfu", "confidence": 0.65, "evidence": ["mfu=0.18"]},
        ],
    }), encoding="utf-8")
    return bottlenecks_json


def test_root_cause_inference(tmp_path: Path):
    """Test root cause inference produces causal chains."""
    bottlenecks_json = _build_bottlenecks(tmp_path)

    mfu_json = tmp_path / "mfu.json"
    mfu_json.write_text(json.dumps({"estimated_mfu": 0.18, "mfu_level": "low"}), encoding="utf-8")

    comm_json = tmp_path / "comm.json"
    comm_json.write_text(json.dumps({"total_time_ms": 132, "collective_count": 3}), encoding="utf-8")

    output_json = tmp_path / "root_cause.json"
    run_script("infer_root_cause.py",
               "--bottlenecks-json", str(bottlenecks_json),
               "--mfu-json", str(mfu_json),
               "--communication-json", str(comm_json),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["root_cause_inference_available"] is True
    assert len(result["root_causes"]) >= 1
    assert result["primary_root_cause"] is not None
    assert "causal_chain" in result["root_causes"][0]


def test_root_cause_no_bottlenecks(tmp_path: Path):
    """Test inference with no bottleneck candidates."""
    bottlenecks_json = tmp_path / "bottlenecks.json"
    bottlenecks_json.write_text(json.dumps({
        "primary_candidate": {"name": "inconclusive", "confidence": 0.1, "evidence": []},
        "ranked_candidates": [],
    }), encoding="utf-8")

    output_json = tmp_path / "root_cause.json"
    run_script("infer_root_cause.py",
               "--bottlenecks-json", str(bottlenecks_json),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["root_cause_inference_available"] is False
