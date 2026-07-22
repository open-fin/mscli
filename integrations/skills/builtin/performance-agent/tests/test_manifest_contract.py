from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = SKILL_ROOT / "skill.yaml"
SKILL = SKILL_ROOT / "SKILL.md"


def _manifest_text() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def test_manifest_contract_fields_present():
    text = _manifest_text()
    assert 'name: "performance-agent"' in text
    assert 'display_name: "Performance Agent"' in text
    assert 'version: "0.4.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'network: "none"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_declares_perf_inputs_and_outputs():
    text = _manifest_text()
    assert 'name: "user_problem"' in text
    assert 'name: "trace_path"' in text
    assert 'name: "metric_focus"' in text
    assert 'name: "env_lock"' in text
    assert 'name: "factory_root"' in text
    assert 'name: "before_metrics_path"' in text
    assert 'name: "after_metrics_path"' in text
    assert 'name: "output_dir"' in text
    assert 'report_schema' in text
    assert 'out_dir_layout' in text


def test_skill_describes_performance_workflow_stages():
    text = SKILL.read_text(encoding="utf-8")
    assert "# Performance Agent" in text
    # The workflow now uses stage numbering (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7)
    assert "data-validator" in text.lower() or "Stage 0.5" in text
    assert "performance-analyzer" in text.lower() or "Stage 1" in text
    assert "bottleneck-validator" in text.lower() or "Stage 2" in text
    assert "snapshot-builder" in text.lower() or "Stage 3" in text
    assert "report-builder" in text.lower() or "Stage 4" in text
    assert "scripts/locate_profiler_output.py" in text
    assert "scripts/summarize_trace_gaps.py" in text
    assert "scripts/build_performance_report.py" in text
