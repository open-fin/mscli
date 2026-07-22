"""Unit tests for analyze_collective_types.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script


def _write_comm_json(root: Path, records: list[dict]) -> Path:
    """Write communication.json with given records."""
    comm_path = root / "ASCEND_PROFILER_OUTPUT" / "communication.json"
    comm_path.parent.mkdir(parents=True, exist_ok=True)
    comm_path.write_text(json.dumps(records), encoding="utf-8")
    return comm_path


def test_classify_collective_types(tmp_path: Path):
    """Test collective type classification with mixed records."""
    _write_comm_json(tmp_path, [
        {"name": "SyncBatchNormAllReduce", "time_ms": 500.0, "count": 100, "size_mb": 0.5},
        {"name": "hccl_AllReduce", "time_ms": 800.0, "count": 200, "size_mb": 128.0},
        {"name": "ReduceScatter", "time_ms": 300.0, "count": 100, "size_mb": 64.0},
    ])
    out = tmp_path / "collective_types.json"
    run_script("analyze_collective_types.py", "--trace-root", str(tmp_path), "--output-json", str(out))

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["collective_type_analysis_available"] is True
    assert result["total_collective_time_ms"] == 1600.0
    assert len(result["types"]) == 3

    # SyncBN should be detected
    syncbn = [t for t in result["types"] if t["type"] == "SyncBN"]
    assert len(syncbn) == 1
    assert syncbn[0]["share_percent"] == 31.25


def test_syncbn_dominant(tmp_path: Path):
    """Test SyncBN dominant detection."""
    _write_comm_json(tmp_path, [
        {"name": "SyncBatchNormAllReduce", "time_ms": 600.0, "count": 100, "size_mb": 0.5},
        {"name": "hccl_AllReduce", "time_ms": 200.0, "count": 200, "size_mb": 128.0},
    ])
    out = tmp_path / "collective_types.json"
    run_script("analyze_collective_types.py", "--trace-root", str(tmp_path), "--output-json", str(out))

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["syncbn_dominant"] is True
    assert result["dominant_type"] == "SyncBN"


def test_allreduce_refinement(tmp_path: Path):
    """Test that large AllReduce becomes GradientAllReduce."""
    _write_comm_json(tmp_path, [
        {"name": "hccl_AllReduce", "time_ms": 100.0, "count": 100, "size_mb": 128.0},
    ])
    out = tmp_path / "collective_types.json"
    run_script("analyze_collective_types.py", "--trace-root", str(tmp_path), "--output-json", str(out))

    result = json.loads(out.read_text(encoding="utf-8"))
    types = [t["type"] for t in result["types"]]
    assert "GradientAllReduce" in types


def test_no_communication_file(tmp_path: Path):
    """Test graceful failure when no communication.json exists."""
    out = tmp_path / "collective_types.json"
    run_script("analyze_collective_types.py", "--trace-root", str(tmp_path), "--output-json", str(out))

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["collective_type_analysis_available"] is False
