from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SKILL_MD = SKILL_ROOT / "SKILL.md"


def test_skill_markers_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "## Workflow" in text
    assert "## Stage 1: Analyze" in text
    assert "## Stage 2. Consistency Validator" in text
    assert "## Stage 3. Snapshot Builder" in text
    assert "## Stage 4. Report Builder" in text


def test_reference_and_script_files_exist():
    assert (SKILL_ROOT / "references" / "consistency-validation.md").exists()
    assert (SKILL_ROOT / "references" / "debug-script-hygiene.md").exists()
    assert (SKILL_ROOT / "references" / "operator-accuracy-triage.md").exists()
    assert (SKILL_ROOT / "references" / "msprobe-task-router.md").exists()
    assert (SKILL_ROOT / "references" / "msprobe-config-cheatsheet.md").exists()
    assert (SKILL_ROOT / "references" / "msprobe-accuracy-compare.md").exists()
    assert (SKILL_ROOT / "references" / "msprobe-grad-probe.md").exists()
    assert (SKILL_ROOT / "references" / "msprobe-overflow-and-nan.md").exists()
    assert (SKILL_ROOT / "references" / "msprobe-single-op-repro.md").exists()
    assert (SKILL_ROOT / "references" / "msprobe-config-and-ckpt-check.md").exists()
    assert (SKILL_ROOT / "references" / "precision-classification-map.md").exists()
    assert (SKILL_ROOT / "scripts" / "collect_accuracy_context.py").exists()
    assert (SKILL_ROOT / "scripts" / "summarize_metric_diff.py").exists()
