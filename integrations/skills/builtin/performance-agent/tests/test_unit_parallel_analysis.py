"""Unit tests for run_parallel_analysis.py."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script, SCRIPTS


def test_dry_run(tmp_path: Path):
    """Test that dry-run produces planned output without executing scripts."""
    output_dir = tmp_path / "out"

    run_script("run_parallel_analysis.py",
               "--working-dir", str(tmp_path),
               "--output-dir", str(output_dir),
               "--dry-run")

    result_json = output_dir / "pipeline_result.json"
    assert result_json.exists()
    result = json.loads(result_json.read_text(encoding="utf-8"))
    assert result["overall_status"] == "dry_run"
    assert len(result["waves"]) >= 7
    for wave in result["waves"]:
        assert wave["status"] == "dry_run"


def test_skip_waves(tmp_path: Path):
    """Test that skip-waves omits specified waves."""
    output_dir = tmp_path / "out"

    run_script("run_parallel_analysis.py",
               "--working-dir", str(tmp_path),
               "--output-dir", str(output_dir),
               "--skip-waves", "3,4,5,6,7,8",
               "--dry-run")

    result_json = output_dir / "pipeline_result.json"
    result = json.loads(result_json.read_text(encoding="utf-8"))
    for wave in result["waves"]:
        if wave["wave"] in (3, 4, 5, 6, 7, 8):
            assert wave["status"] == "skipped"


def test_output_has_schema_version(tmp_path: Path):
    """Test output includes schema_version."""
    output_dir = tmp_path / "out"

    run_script("run_parallel_analysis.py",
               "--working-dir", str(tmp_path),
               "--output-dir", str(output_dir),
               "--dry-run")

    result_json = output_dir / "pipeline_result.json"
    result = json.loads(result_json.read_text(encoding="utf-8"))
    assert result["schema_version"] == "performance-agent/0.1"
