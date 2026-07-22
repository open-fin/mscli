from pathlib import Path


SKILL_YAML = Path(__file__).resolve().parents[1] / "skill.yaml"
SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_manifest_contract_fields_present():
    text = SKILL_YAML.read_text(encoding="utf-8")
    assert 'schema_version: "1.1.0"' in text
    assert 'name: "api-helper"' in text
    assert 'display_name: "API Helper"' in text
    assert 'version: "0.1.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'network: "none"' in text
    assert 'filesystem: "workspace-read"' in text


def test_manifest_declares_api_inputs():
    text = SKILL_YAML.read_text(encoding="utf-8")
    assert 'name: "api_name"' in text
    assert 'name: "resolve_backend"' in text
    assert 'name: "working_dir"' in text
    assert 'report_schema' in text
    assert 'out_dir_layout' in text


def test_skill_md_describes_api_workflow():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "api-to-operator" in text
    assert "operator-to-backend" in text
    assert "mint.*" in text
    assert "tensor.*" in text
    assert "forward" in text.lower()
    assert "backward" in text.lower()
