import json
import subprocess
import sys
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SKILL_ROOT.parents[1]
MANIFEST = SKILL_ROOT / "skill.yaml"
SKILL = SKILL_ROOT / "SKILL.md"
PREFLIGHT = REPO_ROOT / "commands" / "preflight.md"
SCRIPTS = SKILL_ROOT / "scripts"
READINESS_VERDICT_REF = Path("meta/readiness-verdict.json")


def _manifest_text() -> str:
    return MANIFEST.read_text(encoding="utf-8")


def test_manifest_contract_fields_present():
    text = _manifest_text()
    assert 'name: "readiness-agent"' in text
    assert 'display_name: "Readiness Agent"' in text
    assert 'version: "0.3.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'network: "optional"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_keeps_core_inputs_and_drops_low_value_ones():
    text = _manifest_text()
    for token in (
        'name: "working_dir"',
        'name: "target"',
        'choices: ["training", "inference", "auto"]',
        'name: "framework_hint"',
        'choices: ["mindspore", "pta", "mixed", "auto"]',
        'name: "cann_path"',
        'name: "mode"',
        'choices: ["check", "fix"]',
        'name: "selected_python"',
        'name: "model_hub_id"',
        'name: "dataset_hub_id"',
        'name: "dataset_split"',
        'name: "task_smoke_cmd"',
        'name: "allow_network"',
    ):
        assert token in text
    for removed in ('name: "fix_scope"', 'name: "factory_root"'):
        assert removed not in text
    assert 'choices: ["check", "fix", "certify"]' not in text
    assert "certify" not in text


def test_skill_describes_streamlined_runtime_smoke_workflow():
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\nname: readiness-agent\ndescription:")
    assert "Use when Codex needs" not in text
    assert "# Readiness Agent" in text
    assert "## Scope" in text
    assert "## Hard Rules" in text
    assert "## Workflow" in text
    assert "## References" in text
    assert "## Scripts" in text
    assert "runtime_smoke" in text
    assert "`scripts/run_readiness_pipeline.py`" in text
    assert "`scripts/readiness_core.py`" not in text
    assert "`scripts/readiness_report.py`" not in text
    assert "`scripts/ascend_compat.py`" not in text
    assert "Write mode-specific artifacts" in text
    assert "AskUserQuestion" in text
    assert "--confirm <field>=<value>" in text
    assert "Do not merely print the JSON question" in text
    assert "`references/structured-confirmation-adapter.md`" in text
    assert (SKILL_ROOT / "references" / "structured-confirmation-adapter.md").exists()
    assert (SKILL_ROOT / "scripts" / "assessment" / "structured_confirmation_adapter.py").exists()
    assert "`references/product-contract.md`" in text
    assert "`references/decision-rules.md`" in text
    assert "`references/env-fix-policy.md`" in text
    assert "`references/ascend-compat.md`" in text


def test_preflight_documents_check_route():
    text = PREFLIGHT.read_text(encoding="utf-8")
    for token in (
        "/preflight",
        "read-only",
    ):
        assert token in text
    for removed in (
        "/preflight certify",
        "/preflight --mode certify",
        "--mode certify",
        "`certify`: `NEEDS_CONFIRMATION`",
    ):
        assert removed not in text


def test_shared_envelope_and_readiness_verdict_validate_against_their_schemas(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "infer.py").write_text("print('hello')\n", encoding="utf-8")
    (workspace / "model").mkdir()

    report_dir = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_readiness_pipeline.py"),
            "--working-dir",
            str(workspace),
            "--output-dir",
            str(report_dir),
            "--target",
            "inference",
            "--model-path",
            "model",
            "--check",
            "--non-interactive",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert completed.stdout
    shared_schema = json.loads((SKILL_ROOT.parent / "_shared" / "contract" / "report.schema.json").read_text(encoding="utf-8"))
    verdict_schema = json.loads((SKILL_ROOT / "contract" / "readiness-verdict.schema.json").read_text(encoding="utf-8"))
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    verdict = json.loads((report_dir / READINESS_VERDICT_REF).read_text(encoding="utf-8"))

    jsonschema.validate(instance=report, schema=shared_schema)
    jsonschema.validate(instance=verdict, schema=verdict_schema)
