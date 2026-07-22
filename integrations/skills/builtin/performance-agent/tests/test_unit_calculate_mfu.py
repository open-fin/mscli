"""Unit tests for calculate_mfu.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, run_script, ROOT, SCRIPTS


def test_mfu_with_model_config(tmp_path: Path):
    profiler_root = write_sample_profiler_export(tmp_path)
    step_json = tmp_path / "step.json"
    run_script("summarize_step_breakdown.py", "--trace-root", str(profiler_root), "--output-json", str(step_json))

    # Use a small model so synthetic step times produce MFU in valid range
    model_config = tmp_path / "model_config.json"
    model_config.write_text(json.dumps({
        "hidden_size": 256, "num_hidden_layers": 4, "seq_length": 128, "batch_size": 2,
    }), encoding="utf-8")

    mfu_json = tmp_path / "mfu.json"
    run_script("calculate_mfu.py", "--step-json", str(step_json), "--model-config", str(model_config),
               "--hardware", "ascend_910b1", "--num-devices", "1", "--output-json", str(mfu_json))

    result = json.loads(mfu_json.read_text(encoding="utf-8"))
    assert result["method"] == "model_config"
    assert result["reliability"] == "high"
    assert result["estimated_mfu"] is not None
    assert result["estimated_mfu"] >= 0, f"MFU should be non-negative, got {result['estimated_mfu']}"
    assert result["mfu_level"] is not None
    assert result["achieved_tflops"] > 0
    assert result["peak_tflops_used"] == 378.88


def test_mfu_with_time_ratio_fallback(tmp_path: Path):
    profiler_root = write_sample_profiler_export(tmp_path)
    step_json = tmp_path / "step.json"
    run_script("summarize_step_breakdown.py", "--trace-root", str(profiler_root), "--output-json", str(step_json))

    mfu_json = tmp_path / "mfu.json"
    run_script("calculate_mfu.py", "--step-json", str(step_json), "--hardware", "ascend_910b2",
               "--output-json", str(mfu_json))

    result = json.loads(mfu_json.read_text(encoding="utf-8"))
    assert result["method"] == "time_ratio"
    assert result["reliability"] == "low"
    assert result["estimated_mfu"] is not None


def test_mfu_multi_device_scales_peak(tmp_path: Path):
    profiler_root = write_sample_profiler_export(tmp_path)
    step_json = tmp_path / "step.json"
    run_script("summarize_step_breakdown.py", "--trace-root", str(profiler_root), "--output-json", str(step_json))

    mfu_json = tmp_path / "mfu.json"
    run_script("calculate_mfu.py", "--step-json", str(step_json), "--hardware", "ascend_910b1",
               "--num-devices", "8", "--output-json", str(mfu_json))

    result = json.loads(mfu_json.read_text(encoding="utf-8"))
    assert result["peak_tflops_used"] == 378.88 * 8


def test_mfu_no_step_time_returns_error(tmp_path: Path):
    mfu_json = tmp_path / "mfu.json"
    run_script("calculate_mfu.py", "--output-json", str(mfu_json))

    result = json.loads(mfu_json.read_text(encoding="utf-8"))
    assert result["estimated_mfu"] is None
    assert result["error"] is not None
