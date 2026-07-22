import json
import sys
from pathlib import Path

import jsonschema

from helpers import CONTRACT, REPO_ROOT, SCRIPTS, fake_framework_python, load_json, make_workspace, run_pipeline, run_pipeline_allow_confirmation


IDENTITY_KEYS = ("schema_version", "producer", "mode", "engine")
EXPECTED_IDENTITY = {
    "schema_version": "readiness.assessment.v1",
    "producer": "readiness-agent",
    "mode": "check",
    "engine": "assessment",
}


def load_assessment_schema() -> dict:
    schema = load_json(CONTRACT / "readiness-assessment-verdict.schema.json")
    schema["properties"]["portable_question"] = load_json(CONTRACT / "portable-question.schema.json")
    return schema


def assert_identity(payload: dict) -> None:
    assert {key: payload[key] for key in IDENTITY_KEYS} == EXPECTED_IDENTITY


def test_check_ready_writes_workspace_lock_and_run_ref(tmp_path: Path):
    workspace = make_workspace(tmp_path, body="import mindspore\nprint('train')\n")
    output_dir = tmp_path / "out"

    fake_python = fake_framework_python(tmp_path)

    completed = run_pipeline(
        "--mode",
        "check",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--confirm",
        "target=training",
        "--confirm",
        "launcher=python",
        "--confirm",
        "framework=mindspore",
        "--confirm",
        f"runtime_environment={fake_python}",
        "--non-interactive",
        cwd=workspace,
    )
    payload = json.loads(completed.stdout)
    latest = output_dir / "latest" / "readiness-agent"
    removed_split_skill_alias = "-".join(("new", "readiness", "agent"))
    workspace_lock = load_json(latest / "workspace-readiness.lock.json")
    run_ref = load_json(latest / "run-ref.json")

    assert (latest / "workspace-readiness.lock.json").exists()
    assert (latest / "run-ref.json").exists()
    assert not (output_dir / "latest" / removed_split_skill_alias).exists()
    assert payload["status"] == "WARN"
    assert_identity(workspace_lock)
    assert_identity(run_ref)
    assert workspace_lock["status"] == "WARN"
    assert workspace_lock["schema_version"] == "readiness.assessment.v1"
    assert workspace_lock["producer"] == "readiness-agent"
    assert workspace_lock["mode"] == "check"
    assert workspace_lock["engine"] == "assessment"
    assert workspace_lock["target"] == {"value": "training", "source": "confirmed"}
    assert workspace_lock["launcher"] == {"type": "python", "command": "python", "source": "confirmed"}
    assert workspace_lock["runtime"]["python"]["executable"] == str(fake_python)
    assert workspace_lock["runtime"]["framework"] == {"name": "mindspore", "source": "confirmed"}
    assert workspace_lock["runtime"]["cann"]["status"] == "unknown"
    assert workspace_lock["runtime"]["cann"]["source"] == "not_found"
    assert workspace_lock["assets"]["model"]["status"] == "present"
    assert workspace_lock["assets"]["model"]["path"] == str(workspace / "model")
    assert workspace_lock["assets"]["model"]["required"] is True
    assert workspace_lock["assets"]["model"]["validation"] == "path-existence-only"
    assert workspace_lock["assets"]["dataset"] == {"status": "unknown", "required": True}
    assert workspace_lock["assets"]["checkpoint"] == {"status": "unknown", "required": False}
    assert run_ref["latest_status"] == "WARN"
    assert run_ref["attempt_id"]


def test_check_artifacts_validate_against_schemas(tmp_path: Path):
    workspace = make_workspace(tmp_path, body="import mindspore\nprint('train')\n")
    output_dir = tmp_path / "out"

    run_pipeline(
        "--mode",
        "check",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--confirm",
        "target=training",
        "--confirm",
        "launcher=python",
        "--confirm",
        "framework=mindspore",
        "--confirm",
        f"runtime_environment={sys.executable}",
        "--non-interactive",
        cwd=workspace,
    )

    workspace_lock = load_json(output_dir / "latest" / "readiness-agent" / "workspace-readiness.lock.json")
    run_ref = load_json(output_dir / "latest" / "readiness-agent" / "run-ref.json")
    workspace_schema = load_json(CONTRACT / "workspace-readiness-lock.schema.json")
    assessment_schema = load_assessment_schema()

    jsonschema.validate(workspace_lock, workspace_schema)
    jsonschema.validate({**run_ref, "status": run_ref["latest_status"]}, assessment_schema)


def test_check_fix_legacy_artifacts_remain_unchanged(tmp_path: Path):
    workspace = make_workspace(tmp_path, script_name="infer.py", body="print('infer')\n")
    output_dir = tmp_path / "out"

    run_pipeline(
        "--mode",
        "check",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--model-path",
        "model",
        "--non-interactive",
        cwd=workspace,
    )

    assert (output_dir / "report.json").exists()
    assert (output_dir / "report.md").exists()
    assert (output_dir / "meta" / "readiness-verdict.json").exists()
    assert (output_dir / "latest" / "readiness-agent").exists()


# --- read-only guarantees ---

def test_assessment_execution_policy_defaults_to_read_only():
    sys.path.insert(0, str(SCRIPTS))
    try:
        from assessment.assessment_core import AssessmentExecutionPolicy
    finally:
        sys.path.remove(str(SCRIPTS))

    policy = AssessmentExecutionPolicy()

    assert policy.allow_install is False
    assert policy.allow_download is False
    assert policy.allow_env_create is False
    assert policy.allow_source_write is False
    assert policy.allow_config_write is False
    assert policy.allow_real_workload is False
    assert policy.allow_task_smoke is False
    assert policy.allow_readiness_artifact_write is True


def test_check_mode_does_not_mutate_workspace(tmp_path: Path):
    workspace = make_workspace(tmp_path, body="from pathlib import Path\nPath('workload-ran').write_text('bad')\n")
    output_dir = tmp_path / "out"

    run_pipeline(
        "--mode",
        "check",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--task-smoke-cmd",
        f"{sys.executable} train.py",
        "--model-hub-id",
        "org/model",
        "--dataset-hub-id",
        "org/dataset",
        "--confirm",
        "target=training",
        "--confirm",
        "launcher=python",
        "--confirm",
        "framework=mindspore",
        "--confirm",
        f"runtime_environment={sys.executable}",
        "--non-interactive",
        cwd=workspace,
    )

    assert not (workspace / ".readiness.env").exists()
    assert not (workspace / ".venv").exists()
    assert not (workspace / "venv").exists()
    assert not (workspace / "env").exists()
    assert not (workspace / "workload-ran").exists()
    lock_path = output_dir / "latest" / "readiness-agent" / "workspace-readiness.lock.json"
    assert lock_path.exists()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert any(check["name"] == "task_smoke_cmd" and check["status"] == "skipped" for check in lock["checks"])


def test_fix_still_creates_readiness_env(tmp_path: Path):
    workspace = make_workspace(tmp_path, script_name="infer.py", body="print('infer')\n")
    output_dir = tmp_path / "out"

    run_pipeline(
        "--mode",
        "fix",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--model-path",
        "model",
        cwd=workspace,
    )

    assert (workspace / ".readiness.env").exists()
    assert (output_dir / "meta" / "readiness-verdict.json").exists()
