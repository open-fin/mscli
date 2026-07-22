"""Unit tests for validate_profiler_data.py and summarize_aic_metrics.py."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, write_sample_pta_profiler_export, run_script, ROOT, SCRIPTS


def run_validate(*args: str) -> subprocess.CompletedProcess[str]:
    """Run validate_profiler_data.py without check=True since it returns 1 for critical."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_profiler_data.py"), *args],
        check=False, text=True, capture_output=True,
    )


def _write_good_profiler(root: Path) -> Path:
    """Write a profiler export that passes stop_check (valid profiler_info.json)."""
    profiler_root = root / "worker_0_20260325_ascend_ms"
    ascend = profiler_root / "ASCEND_PROFILER_OUTPUT"
    ascend.mkdir(parents=True, exist_ok=True)

    (profiler_root / "profiler_info.json").write_text(
        json.dumps({
            "chip_name": "910b1",
            "status": "completed",
            "profiler_level": "Level2",
            "with_stack": True,
            "with_modules": True,
            "record_shapes": True,
            "profile_memory": False,
            "step_count": 100,
        }),
        encoding="utf-8",
    )
    (profiler_root / "profiler_metadata.json").write_text("{}", encoding="utf-8")
    (profiler_root / "profiler_info_0.json").write_text(
        json.dumps({"chip_name": "910b1", "status": "completed"}), encoding="utf-8"
    )

    (ascend / "step_trace_time.csv").write_text(
        "Step ID,StepTime(ms)\n1,86\n2,83\n3,89\n", encoding="utf-8"
    )
    (ascend / "kernel_details.csv").write_text(
        "Kernel Name,Duration(ms)\nAllReduce,44\n", encoding="utf-8"
    )
    (ascend / "trace_view.json").write_text(
        json.dumps({"events": []}), encoding="utf-8"
    )
    return profiler_root


# -- validate_profiler_data.py --


def test_validate_good_framework_profiler(tmp_path: Path):
    profiler_root = _write_good_profiler(tmp_path)
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["quality_level"] in ("excellent", "good", "fair")
    assert result["stop_check"] is True
    assert result["data_type"] is not None


def test_validate_nonexistent_path_returns_1(tmp_path: Path):
    output_json = tmp_path / "validation.json"
    result = run_validate("--trace-root", str(tmp_path / "nonexistent"), "--output-json", str(output_json))
    assert result.returncode == 1


def test_validate_empty_directory_reports_issues(tmp_path: Path):
    empty_dir = tmp_path / "empty_profiler"
    empty_dir.mkdir()
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(empty_dir), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["quality_level"] in ("critical", "poor")
    assert len(result["issues"]) > 0


def test_validate_reports_deliverables(tmp_path: Path):
    profiler_root = _write_good_profiler(tmp_path)
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    deliverables = result["key_deliverables"]
    assert isinstance(deliverables, dict)
    assert "step_trace_time.csv" in deliverables
    assert deliverables["step_trace_time.csv"] is True


# -- Config parsing --


def test_validate_config_parsing(tmp_path: Path):
    """Config fields should be extracted from profiler_info.json."""
    profiler_root = _write_good_profiler(tmp_path)
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    config = result["profiler_config"]
    assert config is not None
    assert config["config_available"] is True
    assert config["profiler_level"] == "Level2"
    assert config["with_stack"] is True
    assert config["with_modules"] is True
    assert config["record_shapes"] is True
    assert config["profile_memory"] is False
    assert config["step_count"] == 100


def test_validate_config_summary(tmp_path: Path):
    """Config summary should be a human-readable string."""
    profiler_root = _write_good_profiler(tmp_path)
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["config_summary"] is not None
    assert "Level2" in result["config_summary"]
    assert "with_stack" in result["config_summary"]


def test_validate_config_missing_for_msprof(tmp_path: Path):
    """msprof data should have no profiler config."""
    prof_dir = tmp_path / "PROF_12345"
    device_dir = prof_dir / "device_0"
    device_dir.mkdir(parents=True)
    (device_dir / "end_info.0").write_text("done", encoding="utf-8")
    ms_output = prof_dir / "mindstudio_profiler_output"
    ms_output.mkdir()
    (ms_output / "op_summary_0.csv").write_text("Op,Time\nMatMul,10", encoding="utf-8")

    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(prof_dir), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["profiler_config"] is None
    assert result["config_summary"] == "N/A"


# -- Export type detection --


def test_validate_export_type_text(tmp_path: Path):
    """Text mode export should be detected."""
    profiler_root = _write_good_profiler(tmp_path)
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["export_type"] == "text"


def test_validate_export_type_db(tmp_path: Path):
    """DB mode export should be detected."""
    profiler_root = _write_good_profiler(tmp_path)
    # Remove text deliverables to simulate DB-only
    ascend = profiler_root / "ASCEND_PROFILER_OUTPUT"
    (ascend / "trace_view.json").unlink()
    (ascend / "kernel_details.csv").unlink()
    # Add a DB file
    (profiler_root / "worker_profiler_0.db").write_text("fake db", encoding="utf-8")

    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["export_type"] == "db"


def test_validate_export_type_both(tmp_path: Path):
    """Both Text and DB mode should be detected."""
    profiler_root = _write_good_profiler(tmp_path)
    (profiler_root / "worker_profiler_0.db").write_text("fake db", encoding="utf-8")

    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["export_type"] == "both"


# -- msprof deliverables --


def test_validate_msprof_json_deliverable(tmp_path: Path):
    """msprof_{timestamp}.json should be detected as a deliverable."""
    prof_dir = tmp_path / "PROF_12345"
    device_dir = prof_dir / "device_0"
    device_dir.mkdir(parents=True)
    (device_dir / "end_info.0").write_text("done", encoding="utf-8")
    ms_output = prof_dir / "mindstudio_profiler_output"
    ms_output.mkdir()
    (ms_output / "msprof_20260325.json").write_text("{}", encoding="utf-8")
    (ms_output / "op_summary_0.csv").write_text("Op,Time\nMatMul,10", encoding="utf-8")

    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(prof_dir), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["key_deliverables"]["msprof_json"] is True


def test_validate_msprof_no_json_deliverable(tmp_path: Path):
    """Missing msprof JSON should be reported."""
    prof_dir = tmp_path / "PROF_12345"
    device_dir = prof_dir / "device_0"
    device_dir.mkdir(parents=True)
    (device_dir / "end_info.0").write_text("done", encoding="utf-8")
    ms_output = prof_dir / "mindstudio_profiler_output"
    ms_output.mkdir()
    (ms_output / "op_summary_0.csv").write_text("Op,Time\nMatMul,10", encoding="utf-8")

    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(prof_dir), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["key_deliverables"]["msprof_json"] is False


# -- Recommended action --


def test_validate_critical_recommends_recollect(tmp_path: Path):
    """Critical quality should recommend recollection."""
    empty_dir = tmp_path / "empty_profiler"
    empty_dir.mkdir()
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(empty_dir), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result["quality_level"] == "critical":
        assert "recollect" in result["recommended_action"].lower() or "stop" in result["recommended_action"].lower()


def test_validate_schema_version_updated(tmp_path: Path):
    """Schema version should be 0.2."""
    profiler_root = _write_good_profiler(tmp_path)
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["schema_version"] == "performance-agent/0.2"


# -- summarize_aic_metrics.py --


def test_aic_no_data_returns_gracefully(tmp_path: Path):
    profiler_root = write_sample_profiler_export(tmp_path)
    output_json = tmp_path / "aic.json"
    run_script("summarize_aic_metrics.py", "--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["aic_data_available"] is False


def test_aic_with_percentage_values(tmp_path: Path):
    profiler_root = tmp_path / "prof_with_aic"
    device_data = profiler_root / "PROF_0" / "device_0" / "data"
    device_data.mkdir(parents=True)

    (device_data / "aic_metrics_0.csv").write_text(
        "Op Name,Cube Utilization,Duration(us),L2 Hit Rate\n"
        "MatMul_0,5.2%,1000,30.5%\n"
        "MatMul_1,45.0%,500,60.0%\n"
        "MatMul_2,75.0%,200,85.0%\n",
        encoding="utf-8",
    )

    output_json = tmp_path / "aic.json"
    run_script("summarize_aic_metrics.py", "--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["aic_data_available"] is True
    assert result["analyzed_operators"] == 3
    assert result["severity_summary"]["critical"] >= 1
    top = result["top_bottlenecks"][0]
    assert top["severity"] == "critical"
    assert top["cube_utilization"] < 1.0


def test_aic_with_fraction_values(tmp_path: Path):
    profiler_root = tmp_path / "prof_frac"
    device_data = profiler_root / "PROF_0" / "device_0" / "data"
    device_data.mkdir(parents=True)

    (device_data / "aic_metrics_0.csv").write_text(
        "Op Name,cube_utilization,l2_hit_rate\n"
        "MatMul,0.85,0.90\n",
        encoding="utf-8",
    )

    output_json = tmp_path / "aic.json"
    run_script("summarize_aic_metrics.py", "--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["aic_data_available"] is True
    top = result["top_bottlenecks"][0]
    assert top["cube_utilization"] == 0.85
    assert top["l2_hit_rate"] == 0.90


# -- PTA (torch_npu) realistic profiler data tests --


def test_validate_pta_framework_profiler(tmp_path: Path):
    """Validate a PTA profiler export with realistic pro_data-like structure."""
    profiler_root = write_sample_pta_profiler_export(tmp_path)
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["data_type"] == "framework_profiler_pt"
    assert result["stop_check"] is True
    assert result["parse_check"] is True
    assert result["quality_level"] in ("good", "fair")
    # kernel_details.csv should be present
    assert result["key_deliverables"]["kernel_details.csv"] is True
    # step_trace_time.csv and communication.json should be absent
    assert result["key_deliverables"]["step_trace_time.csv"] is False
    assert result["key_deliverables"]["communication.json"] is False


def test_validate_pta_config_parsing(tmp_path: Path):
    """PTA config fields should be extracted from profiler_info.json."""
    profiler_root = write_sample_pta_profiler_export(tmp_path)
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    config = result["profiler_config"]
    assert config["config_available"] is True
    # PTA format has nested config structure
    assert result["config_summary"] is not None


def test_validate_pta_export_type_text(tmp_path: Path):
    """PTA profiler with text export should be detected."""
    profiler_root = write_sample_pta_profiler_export(tmp_path)
    output_json = tmp_path / "validation.json"
    run_validate("--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["export_type"] == "text"


def test_aic_pta_no_data_returns_gracefully(tmp_path: Path):
    """PTA profiler without AIC metrics should report unavailable."""
    profiler_root = write_sample_pta_profiler_export(tmp_path)
    output_json = tmp_path / "aic.json"
    run_script("summarize_aic_metrics.py", "--trace-root", str(profiler_root), "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["aic_data_available"] is False
