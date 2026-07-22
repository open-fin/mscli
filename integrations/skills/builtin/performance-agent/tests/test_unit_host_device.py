"""Unit tests for correlate_host_device.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script, write_sample_pta_profiler_export


def test_correlation_with_trace_view(tmp_path: Path):
    """Test correlation analysis with trace_view.json containing host and device events."""
    trace_json = tmp_path / "trace_view.json"
    trace_json.write_text(json.dumps({
        "events": [
            {"name": "host_enqueue_kernel", "ts": 1000, "dur": 50, "cat": "cpu_op"},
            {"name": "host_enqueue_kernel", "ts": 2000, "dur": 30, "cat": "cpu_op"},
            {"name": "MatMul_Kernel", "ts": 1100, "dur": 500, "cat": "npu_op"},
            {"name": "AllReduce_Kernel", "ts": 1700, "dur": 300, "cat": "npu_op"},
            {"name": "Softmax_Kernel", "ts": 2100, "dur": 200, "cat": "npu_op"},
            {"name": "tensor.item()", "ts": 2400, "dur": 10, "cat": "cpu_op"},
            {"name": "torch.isfinite", "ts": 2450, "dur": 5, "cat": "cpu_op"},
        ]
    }), encoding="utf-8")

    output_json = tmp_path / "correlation.json"
    run_script("correlate_host_device.py",
               "--trace-view-json", str(trace_json),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["host_device_correlation_available"] is True
    assert result["sync_points_count"] >= 2
    sync_types = {sp["type"] for sp in result["sync_points"]}
    assert "tensor.item" in sync_types
    assert "torch.isfinite" in sync_types


def test_correlation_no_input(tmp_path: Path):
    """Test correlation with no trace input."""
    output_json = tmp_path / "correlation.json"
    run_script("correlate_host_device.py",
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["host_device_correlation_available"] is False


def test_correlation_empty_events(tmp_path: Path):
    """Test correlation with empty events list."""
    trace_json = tmp_path / "trace_view.json"
    trace_json.write_text(json.dumps({"events": []}), encoding="utf-8")

    output_json = tmp_path / "correlation.json"
    run_script("correlate_host_device.py",
               "--trace-view-json", str(trace_json),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["host_device_correlation_available"] is False


def test_correlation_with_pta_kernel_details(tmp_path: Path):
    """Test correlation with realistic PTA kernel_details.csv (aclnn* kernels)."""
    profiler_root = write_sample_pta_profiler_export(tmp_path)

    output_json = tmp_path / "correlation.json"
    run_script("correlate_host_device.py",
               "--kernel-details-csv", str(profiler_root / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv"),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    # Without trace_view, correlation itself is unavailable
    assert result["host_device_correlation_available"] is False
