from pathlib import Path


SKILL_YAML = Path(__file__).resolve().parents[1] / "skill.yaml"
SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_manifest_contract_fields_present():
    text = SKILL_YAML.read_text(encoding="utf-8")
    assert 'schema_version: "1.1.0"' in text
    assert 'name: "operator-agent"' in text
    assert 'display_name: "Op Agent"' in text
    assert 'version: "0.3.0"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'network: "none"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_declares_operator_inputs():
    text = SKILL_YAML.read_text(encoding="utf-8")
    assert 'name: "working_dir"' in text
    assert 'name: "framework"' in text
    assert 'name: "operator_name"' in text
    assert 'name: "backend"' in text
    assert 'name: "method_preference"' in text
    assert 'name: "delivery_goal"' in text
    assert 'report_schema' in text
    assert 'out_dir_layout' in text


def test_skill_md_describes_four_stage_workflow():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "operator-analyzer" in text
    assert "method-selector" in text
    assert "implementation-builder" in text
    assert "verification-and-report" in text
    assert "custom-access" in text
    assert "native-framework" in text
    assert "OperatorBuildProfile" in text


def test_skill_md_declares_scope():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "torch" in text
    assert "mindspore" in text
    assert "cpu" in text
    assert "gpu" in text
    assert "npu" in text
