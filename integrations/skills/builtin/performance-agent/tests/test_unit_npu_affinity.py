"""Unit tests for analyze_npu_affinity.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script


def test_affinity_with_fusion_opportunity(tmp_path: Path):
    """Test NPU affinity detects fusion opportunity in Step 1."""
    hotspot_json = tmp_path / "hotspot.json"
    hotspot_json.write_text(json.dumps({
        "top_operators": [
            {"operator": "SelfAttention_MatMul", "share_percent": 30.0},
            {"operator": "Softmax_Compute", "share_percent": 15.0},
            {"operator": "Dropout_Gen", "share_percent": 8.0},
        ]
    }), encoding="utf-8")

    output_json = tmp_path / "affinity.json"
    run_script("analyze_npu_affinity.py",
               "--hotspot-json", str(hotspot_json),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["npu_affinity_analysis_available"] is True
    assert len(result["steps"]) == 4
    step1 = result["steps"][0]
    assert step1["name"] == "operator_fusion"
    assert len(step1["findings"]) >= 1
    assert result["overall_affinity_score"] < 1.0


def test_affinity_with_sync_points(tmp_path: Path):
    """Test NPU affinity detects sync issues in Step 2."""
    host_device_json = tmp_path / "host_device.json"
    host_device_json.write_text(json.dumps({
        "host_device_correlation_available": True,
        "sync_points": [
            {"type": "tensor.item", "duration_ms": 5.2, "event_name": "item()"},
            {"type": "torch.isfinite", "duration_ms": 2.1, "event_name": "isfinite()"},
        ],
    }), encoding="utf-8")

    output_json = tmp_path / "affinity.json"
    run_script("analyze_npu_affinity.py",
               "--host-device-json", str(host_device_json),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["npu_affinity_analysis_available"] is True
    step2 = result["steps"][1]
    assert step2["name"] == "stream_sync_elimination"
    assert len(step2["findings"]) >= 1


def test_affinity_no_input(tmp_path: Path):
    """Test affinity with no input data."""
    output_json = tmp_path / "affinity.json"
    run_script("analyze_npu_affinity.py",
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["npu_affinity_analysis_available"] is True
    assert result["overall_affinity_score"] == 1.0


def test_affinity_priority_fix(tmp_path: Path):
    """Test that priority_fix is set when score is low."""
    hotspot_json = tmp_path / "hotspot.json"
    hotspot_json.write_text(json.dumps({
        "top_operators": [
            {"operator": "Attention_Score", "share_percent": 40.0},
            {"operator": "Softmax", "share_percent": 20.0},
        ]
    }), encoding="utf-8")

    output_json = tmp_path / "affinity.json"
    run_script("analyze_npu_affinity.py",
               "--hotspot-json", str(hotspot_json),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result["overall_affinity_score"] < 0.8:
        assert result["priority_fix"] is not None
