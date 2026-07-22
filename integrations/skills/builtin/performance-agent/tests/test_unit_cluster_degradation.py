"""Unit tests for classify_cluster_degradation.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script


def test_scale_up_degradation(tmp_path: Path):
    """Test detection of scale-up degradation pattern."""
    cluster_json = tmp_path / "cluster.json"
    cluster_json.write_text(json.dumps({
        "cluster_analysis_available": True,
        "slow_ranks": [],
        "analysis": {},
    }), encoding="utf-8")

    comm_json = tmp_path / "comm.json"
    comm_json.write_text(json.dumps({
        "communication_pressure": "high",
    }), encoding="utf-8")

    linearity_json = tmp_path / "linearity.json"
    linearity_json.write_text(json.dumps({
        "linearity": 0.65,
    }), encoding="utf-8")

    output_json = tmp_path / "degradation.json"
    run_script("classify_cluster_degradation.py",
               "--cluster-json", str(cluster_json),
               "--communication-json", str(comm_json),
               "--linearity-json", str(linearity_json),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["degradation_classification_available"] is True
    assert "scale_up" in result.get("all_types", [])


def test_slow_node_detection(tmp_path: Path):
    """Test detection of slow node pattern."""
    cluster_json = tmp_path / "cluster.json"
    cluster_json.write_text(json.dumps({
        "cluster_analysis_available": True,
        "slow_ranks": [3],
        "analysis": {"bottleneck_type": "host_dispatch"},
    }), encoding="utf-8")

    output_json = tmp_path / "degradation.json"
    run_script("classify_cluster_degradation.py",
               "--cluster-json", str(cluster_json),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["degradation_classification_available"] is True
    assert "slow_node" in result.get("all_types", [])


def test_no_input_data(tmp_path: Path):
    """Test with no input data."""
    output_json = tmp_path / "degradation.json"
    run_script("classify_cluster_degradation.py",
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["degradation_classification_available"] is False
