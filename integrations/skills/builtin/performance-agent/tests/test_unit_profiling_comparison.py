"""Unit tests for compare_profiling_runs.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script


def _write_run_artifacts(directory: Path, step_time: float, comm_time: float,
                         peak_memory: float, mfu: float) -> None:
    """Write a set of analysis artifacts to a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "step.json").write_text(json.dumps({
        "average_step_time_ms": step_time,
        "stage_totals_ms": {"compute": step_time * 0.5, "communication": comm_time},
    }), encoding="utf-8")
    (directory / "communication.json").write_text(json.dumps({
        "total_time_ms": comm_time,
        "communication_pressure": "high" if comm_time > 40 else "low",
        "collective_count": 3,
    }), encoding="utf-8")
    (directory / "memory.json").write_text(json.dumps({
        "peak_memory_mb": peak_memory,
        "memory_pressure": "high" if peak_memory > 50000 else "low",
    }), encoding="utf-8")
    (directory / "mfu.json").write_text(json.dumps({
        "estimated_mfu": mfu,
    }), encoding="utf-8")


def test_comparison_improved(tmp_path: Path):
    """Test comparison detects improvement."""
    baseline = tmp_path / "baseline"
    comparison = tmp_path / "comparison"
    _write_run_artifacts(baseline, 86.0, 44.0, 45000.0, 0.25)
    _write_run_artifacts(comparison, 70.0, 30.0, 42000.0, 0.35)

    output_json = tmp_path / "comparison_result.json"
    run_script("compare_profiling_runs.py",
               "--baseline-dir", str(baseline),
               "--comparison-dir", str(comparison),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["comparison_available"] is True
    assert result["overall_verdict"] in ("improved", "mixed")
    assert len(result.get("significant_changes", [])) > 0


def test_comparison_regressed(tmp_path: Path):
    """Test comparison detects regression."""
    baseline = tmp_path / "baseline"
    comparison = tmp_path / "comparison"
    _write_run_artifacts(baseline, 60.0, 20.0, 30000.0, 0.45)
    _write_run_artifacts(comparison, 90.0, 50.0, 55000.0, 0.20)

    output_json = tmp_path / "comparison_result.json"
    run_script("compare_profiling_runs.py",
               "--baseline-dir", str(baseline),
               "--comparison-dir", str(comparison),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["comparison_available"] is True
    assert result["overall_verdict"] in ("regressed", "mixed")


def test_comparison_missing_dir(tmp_path: Path):
    """Test comparison with missing directory."""
    output_json = tmp_path / "comparison_result.json"
    run_script("compare_profiling_runs.py",
               "--baseline-dir", str(tmp_path / "nonexistent"),
               "--comparison-dir", str(tmp_path / "also_missing"),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["comparison_available"] is False


def test_comparison_pta_hotspot_change(tmp_path: Path):
    """Test comparison of PTA hotspot changes between two runs."""
    baseline = tmp_path / "baseline"
    comparison = tmp_path / "comparison"
    baseline.mkdir()
    comparison.mkdir()

    # Baseline: MatMul dominates
    (baseline / "hotspot.json").write_text(json.dumps({
        "top_operators": [
            {"operator": "aclnnMatmul_MatMulCommon_MatMulV2", "total_time": 420.5, "share_percent": 35.0, "count": 100},
            {"operator": "aclnnFlashAttentionScoreGrad_GetFlashAttentionSrc", "total_time": 310.4, "share_percent": 25.0, "count": 100},
        ]
    }), encoding="utf-8")

    # Comparison: After optimization, FlashAttention dominates (regression)
    (comparison / "hotspot.json").write_text(json.dumps({
        "top_operators": [
            {"operator": "aclnnFlashAttentionScoreGrad_GetFlashAttentionSrc", "total_time": 520.4, "share_percent": 40.0, "count": 100},
            {"operator": "aclnnMatmul_MatMulCommon_MatMulV2", "total_time": 300.5, "share_percent": 23.0, "count": 100},
        ]
    }), encoding="utf-8")

    output_json = tmp_path / "comparison_result.json"
    run_script("compare_profiling_runs.py",
               "--baseline-dir", str(baseline),
               "--comparison-dir", str(comparison),
               "--output-json", str(output_json))

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["comparison_available"] is True
