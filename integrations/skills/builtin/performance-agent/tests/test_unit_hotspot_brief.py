"""Unit tests for build_hotspot_brief.py."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, run_script, ROOT, SCRIPTS


def _write_hotspot_summary(root: Path, ops: list[dict], cumulative: float = 85.0) -> Path:
    """Write a hotspot_summary.json with given operator data."""
    summary_path = root / "hotspot_summary.json"
    summary_path.write_text(json.dumps({
        "source_file": "op_summary_0.csv",
        "summary": {
            "total_operator_time_ms": 1000.0,
            "top_n_cumulative_share_percent": cumulative,
        },
        "top_operators": ops,
    }), encoding="utf-8")
    return summary_path


def test_build_brief_with_top_ops(tmp_path: Path):
    """Build brief from a hotspot summary with 3 operators."""
    ops = [
        {"operator": "MatMul", "share_percent": 45.0, "category": "computation_or_other"},
        {"operator": "AllReduce", "share_percent": 30.0, "category": "communication"},
        {"operator": "LayerNorm", "share_percent": 10.0, "category": "computation_or_other"},
    ]
    summary_path = _write_hotspot_summary(tmp_path, ops)
    output_json = tmp_path / "brief.json"
    output_md = tmp_path / "brief.md"

    run_script(
        "build_hotspot_brief.py",
        "--input-json", str(summary_path),
        "--output-json", str(output_json),
        "--output-md", str(output_md),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["primary_focus"] == "MatMul"
    assert len(result["priority_queue"]) == 3

    # Verify priority ordering
    assert result["priority_queue"][0]["rank"] == 1
    assert result["priority_queue"][0]["operator"] == "MatMul"
    assert result["priority_queue"][0]["share_percent"] == 45.0

    # Verify category propagation
    assert result["priority_queue"][1]["category"] == "communication"
    assert result["priority_queue"][1]["first_optimization_direction"] != result["priority_queue"][0]["first_optimization_direction"]

    # Verify markdown output
    md_content = output_md.read_text(encoding="utf-8")
    assert "# Hotspot Brief" in md_content
    assert "MatMul" in md_content
    assert "45.0%" in md_content


def test_build_brief_with_top_k_limit(tmp_path: Path):
    """top_k should limit the number of prioritized operators."""
    ops = [
        {"operator": f"Op{i}", "share_percent": float(50 - i * 5), "category": "computation_or_other"}
        for i in range(10)
    ]
    summary_path = _write_hotspot_summary(tmp_path, ops)
    output_json = tmp_path / "brief.json"
    output_md = tmp_path / "brief.md"

    run_script(
        "build_hotspot_brief.py",
        "--input-json", str(summary_path),
        "--output-json", str(output_json),
        "--output-md", str(output_md),
        "--top-k", "3",
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(result["priority_queue"]) == 3
    assert result["priority_queue"][0]["operator"] == "Op0"


def test_build_brief_empty_ops(tmp_path: Path):
    """Empty operator list should produce brief with no primary focus."""
    summary_path = _write_hotspot_summary(tmp_path, [])
    output_json = tmp_path / "brief.json"
    output_md = tmp_path / "brief.md"

    run_script(
        "build_hotspot_brief.py",
        "--input-json", str(summary_path),
        "--output-json", str(output_json),
        "--output-md", str(output_md),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["primary_focus"] is None
    assert len(result["priority_queue"]) == 0


def test_communication_category_has_different_metrics(tmp_path: Path):
    """Communication-category operators should have comm-specific rerun metrics."""
    ops = [
        {"operator": "AllReduce", "share_percent": 60.0, "category": "communication"},
    ]
    summary_path = _write_hotspot_summary(tmp_path, ops)
    output_json = tmp_path / "brief.json"
    output_md = tmp_path / "brief.md"

    run_script(
        "build_hotspot_brief.py",
        "--input-json", str(summary_path),
        "--output-json", str(output_json),
        "--output-md", str(output_md),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    item = result["priority_queue"][0]
    assert "communication time share" in item["rerun_metrics"]
    assert "collective count" in item["rerun_metrics"]


def test_missing_input_exits_with_error(tmp_path: Path):
    """Missing input JSON should exit with error."""
    output_json = tmp_path / "brief.json"
    output_md = tmp_path / "brief.md"

    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "build_hotspot_brief.py"),
         "--input-json", str(tmp_path / "nonexistent.json"),
         "--output-json", str(output_json),
         "--output-md", str(output_md)],
        text=True, capture_output=True,
    )
    assert result.returncode != 0


def test_build_brief_with_pta_realistic_ops(tmp_path: Path):
    """Build brief from PTA hotspot summary with realistic aclnn* operator names."""
    ops = [
        {"operator": "aclnnMatmul_MatMulCommon_MatMulV2", "share_percent": 32.5, "category": "computation_or_other"},
        {"operator": "aclnnFlashAttentionScoreGrad_GetFlashAttentionSrc", "share_percent": 18.2, "category": "computation_or_other"},
        {"operator": "aclnnFlashAttentionScore_GetFlashAttentionSrc", "share_percent": 15.8, "category": "computation_or_other"},
        {"operator": "aclnnMul_MulAiCore_Mul", "share_percent": 8.4, "category": "computation_or_other"},
        {"operator": "aclnnSilu_SiluAiCore_Silu", "share_percent": 5.1, "category": "computation_or_other"},
    ]
    summary_path = _write_hotspot_summary(tmp_path, ops)
    output_json = tmp_path / "brief.json"
    output_md = tmp_path / "brief.md"

    run_script(
        "build_hotspot_brief.py",
        "--input-json", str(summary_path),
        "--output-json", str(output_json),
        "--output-md", str(output_md),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["primary_focus"] == "aclnnMatmul_MatMulCommon_MatMulV2"
    assert len(result["priority_queue"]) == 3  # default top_k=3
    assert result["priority_queue"][0]["share_percent"] == 32.5

    md_content = output_md.read_text(encoding="utf-8")
    assert "aclnnMatmul" in md_content
