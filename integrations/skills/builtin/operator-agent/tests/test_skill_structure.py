from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_layout_exists():
    assert (ROOT / "SKILL.md").exists()
    assert (ROOT / "skill.yaml").exists()
    assert (ROOT / "references").is_dir()
    assert (ROOT / "scripts").is_dir()
    assert (ROOT / "templates").is_dir()
    assert (ROOT / "verification").is_dir()
    assert (ROOT / "workflows").is_dir()


def test_api_resolution_packs_exist():
    api_dir = ROOT / "references" / "api-resolution"
    assert (api_dir / "api-to-operator.md").exists()
    assert (api_dir / "operator-to-backend.md").exists()
    assert (api_dir / "validation-checklist.md").exists()


def test_verification_packs_exist():
    assert (ROOT / "verification" / "op-info-test.md").exists()
    assert (ROOT / "verification" / "codecheck.md").exists()
    assert (ROOT / "verification" / "op-info-workflow").is_dir()


def test_mindspore_native_workflow_exists():
    wf = ROOT / "workflows" / "mindspore-native"
    assert (wf / "routing.md").exists()
    assert (wf / "reference.md").exists()
    assert (wf / "common").is_dir()
    assert (wf / "backend").is_dir()


def test_mindspore_native_common_steps_exist():
    common = ROOT / "workflows" / "mindspore-native" / "common"
    assert (common / "00-pre-checks.md").exists()
    assert (common / "01-yaml-definition.md").exists()
    assert (common / "02-code-generation.md").exists()
    assert (common / "03-general-infer.md").exists()
    assert (common / "06-bprop.md").exists()
    assert (common / "07-export.md").exists()
    assert (common / "08-unit-tests.md").exists()
    assert (common / "09-docs.md").exists()
    assert (common / "10-build.md").exists()


def test_mindspore_native_backend_aclnn_exists():
    aclnn = ROOT / "workflows" / "mindspore-native" / "backend" / "aclnn"
    assert (aclnn / "aclnn-call-mapping.md").exists()
    assert (aclnn / "aclnn-path-selection.md").exists()


def test_scripts_exist():
    assert (ROOT / "scripts" / "ms_codecheck.py").exists()
    assert (ROOT / "scripts" / "remote_runner_client.py").exists()
    assert (ROOT / "scripts" / "remote_runner_server.py").exists()
