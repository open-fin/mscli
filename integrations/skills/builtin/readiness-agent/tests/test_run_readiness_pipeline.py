import json
import os
import subprocess
import sys
from pathlib import Path

from helpers import READINESS_VERDICT_REF, SCRIPTS, fake_framework_python, make_workspace, run_pipeline, run_pipeline_allow_confirmation


def load_report_pair(report_json: Path) -> tuple[dict, dict]:
    envelope = json.loads(report_json.read_text(encoding="utf-8"))
    verdict_json = report_json.parent / READINESS_VERDICT_REF
    verdict = json.loads(verdict_json.read_text(encoding="utf-8"))
    return envelope, verdict


def fake_uv_source() -> str:
    return f"""#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

REAL_PYTHON = r'''{sys.executable}'''

def main() -> int:
    args = sys.argv[1:]
    if not args:
        return 1
    if args[0] == "venv":
        env_root = None
        for item in args[1:]:
            if item.startswith("-"):
                continue
            env_root = Path(item)
            break
        if env_root is None:
            return 2
        target = env_root / ("Scripts" if os.name == "nt" else "bin") / ("python.exe" if os.name == "nt" else "python")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REAL_PYTHON, target)
        return 0
    if len(args) >= 2 and args[0] == "pip" and args[1] == "install":
        return 0
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""


def install_fake_uv(tmp_path: Path, monkeypatch) -> Path:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()

    uv_py = bin_dir / "uv"
    uv_py.write_text(fake_uv_source(), encoding="utf-8")
    uv_py.chmod(uv_py.stat().st_mode | 0o111)

    uv_cmd = bin_dir / "uv.cmd"
    uv_cmd.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0uv" %*\r\n', encoding="utf-8")
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    return bin_dir


def test_check_non_interactive_never_returns_needs_confirmation(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    output_dir = tmp_path / "out"

    completed = run_pipeline(
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
    _, verdict = load_report_pair(output_dir / "report.json")

    assert "NEEDS_CONFIRMATION" not in completed.stdout
    assert verdict["status"] != "NEEDS_CONFIRMATION"


def test_fix_safe_repair_only_in_fix_mode(tmp_path: Path, monkeypatch):
    install_fake_uv(tmp_path, monkeypatch)
    check_workspace = make_workspace(tmp_path / "check")
    fix_workspace = make_workspace(tmp_path / "fix")

    run_pipeline(
        "--mode",
        "check",
        "--working-dir",
        str(check_workspace),
        "--target",
        "inference",
        "--model-path",
        "model",
        "--non-interactive",
        cwd=check_workspace,
    )
    run_pipeline(
        "--mode",
        "fix",
        "--working-dir",
        str(fix_workspace),
        "--target",
        "inference",
        "--model-path",
        "model",
        cwd=fix_workspace,
    )

    assert not (check_workspace / ".venv").exists()
    assert (fix_workspace / ".venv").exists()


def test_run_readiness_pipeline_check_blocks_without_workspace_env(tmp_path: Path):
    workspace = make_workspace(tmp_path)
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--model-path",
        "model",
        "--check",
        "--non-interactive",
        cwd=workspace,
    )

    _, verdict = load_report_pair(output_dir / "report.json")
    fix_applied = verdict["fix_applied"]

    assert verdict["status"] == "BLOCKED"
    assert verdict["can_run"] is False
    assert fix_applied["execute"] is False
    assert fix_applied["planned_actions"] == []
    assert not (workspace / ".readiness.env").exists()


def test_run_readiness_pipeline_warns_on_unknown_cann_and_still_prompts_to_run_model_script(tmp_path: Path, fake_selected_python: Path):
    workspace = make_workspace(
        tmp_path,
        body="import torch\nimport torch_npu\nimport transformers\nprint('infer')\n",
    )
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--framework-hint",
        "pta",
        "--selected-python",
        str(fake_selected_python),
        "--model-path",
        "model",
        "--check",
        "--non-interactive",
        cwd=workspace,
    )

    _, verdict = load_report_pair(output_dir / "report.json")
    assert verdict["status"] == "WARN"
    assert verdict["can_run"] is True
    assert verdict["evidence_level"] == "runtime_smoke"
    assert "Do you want me to run the real model script now?" in verdict["next_action"]


def test_run_readiness_pipeline_blocks_on_missing_explicit_cann_path(tmp_path: Path, fake_selected_python: Path):
    workspace = make_workspace(
        tmp_path,
        body="import torch\nimport torch_npu\nimport transformers\nprint('infer')\n",
    )
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--framework-hint",
        "pta",
        "--selected-python",
        str(fake_selected_python),
        "--model-path",
        "model",
        "--cann-path",
        str(tmp_path / "custom-cann-9.9.9"),
        "--check",
        "--non-interactive",
        cwd=workspace,
    )

    _, verdict = load_report_pair(output_dir / "report.json")
    assert verdict["status"] == "BLOCKED"
    assert verdict["can_run"] is False
    assert "Resolve blockers" in verdict["next_action"]


def test_run_readiness_pipeline_fix_creates_default_env_and_reruns(tmp_path: Path, monkeypatch):
    install_fake_uv(tmp_path, monkeypatch)
    workspace = make_workspace(tmp_path)
    output_dir = tmp_path / "out"

    run_pipeline(
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "inference",
        "--model-path",
        "model",
        "--fix",
        cwd=workspace,
    )

    _, verdict = load_report_pair(output_dir / "report.json")
    env_json = json.loads((output_dir / "meta" / "env.json").read_text(encoding="utf-8"))

    assert (workspace / ".venv").exists()
    assert verdict["status"] == "WARN"
    assert verdict["can_run"] is True
    assert "Do you want me to run the real model script now?" in verdict["next_action"]
    assert env_json["pipeline_passes"] == 2
    assert "create-workspace-env" in verdict["fix_applied"]["executed_actions"]


def test_run_readiness_pipeline_tolerates_missing_and_unknown_cli_args(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_readiness_pipeline.py"),
            "--fix",
            "--verbose",
            "--unknown-flag",
            "mystery",
            "--model-path",
        ],
        cwd=str(workspace),
        check=True,
        text=True,
        capture_output=True,
    )

    summary = json.loads(completed.stdout)
    inputs = json.loads((workspace / "readiness-output" / "meta" / "inputs.json").read_text(encoding="utf-8"))

    assert inputs["ignored_cli_args"] == [
        {"token": "--unknown-flag", "reason": "unknown_flag"},
        {"token": "mystery", "reason": "unknown_flag_value"},
        {"token": "--model-path", "reason": "missing_value"},
    ]


def test_run_readiness_pipeline_rejects_removed_certify_mode(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_readiness_pipeline.py"),
            "--mode",
            "certify",
        ],
        cwd=str(workspace),
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert json.loads(completed.stderr) == {
        "error": "mode certify was removed; use --mode check for guided readiness assessment."
    }


def test_check_needs_confirmation_contract(tmp_path: Path):
    workspace = make_workspace(tmp_path, script_name="run.py", body="print('run')\n")

    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_readiness_pipeline.py"), "--mode", "check", "--non-interactive", "--working-dir", str(workspace)],
        cwd=str(workspace),
        text=True,
        capture_output=True,
        timeout=10,
    )
    # With --non-interactive, check should auto-select and return a final status
    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["status"] in {"READY", "WARN", "BLOCKED"}


def test_check_confirm_replay_advances_state(tmp_path: Path):
    workspace = make_workspace(tmp_path, script_name="run.py", body="print('run')\n")

    fake_python = fake_framework_python(tmp_path)

    completed = run_pipeline(
        "--mode",
        "check",
        "--working-dir",
        str(workspace),
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

    assert payload["status"] == "WARN"
    lock = json.loads((workspace / "readiness-output" / "latest" / "readiness-agent" / "workspace-readiness.lock.json").read_text(encoding="utf-8"))
    assert lock["current_confirmation"] is None
    assert lock["target"] == {"value": "training", "source": "confirmed"}
    assert lock["launcher"] == {"type": "python", "command": "python", "source": "confirmed"}
    assert lock["runtime"]["framework"] == {"name": "mindspore", "source": "confirmed"}


def test_check_ambiguous_mixed_framework_requires_confirmation(tmp_path: Path):
    workspace = make_workspace(
        tmp_path,
        script_name="train.py",
        body="import mindspore\nimport torch\nprint('train')\n",
    )

    completed = run_pipeline_allow_confirmation(
        "--mode",
        "check",
        "--working-dir",
        str(workspace),
        "--confirm",
        "target=training",
        "--confirm",
        "launcher=python",
        cwd=workspace,
    )
    payload = json.loads(completed.stdout)

    assert payload["status"] == "NEEDS_CONFIRMATION"
    assert payload["current_confirmation"]["field"] == "framework"
    values = [option["value"] for option in payload["portable_question"]["options"]]
    assert values[:3] == ["mindspore", "pytorch_npu", "mixed"]


def test_run_readiness_pipeline_rejects_removed_auto_mode(tmp_path: Path):
    workspace = make_workspace(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_readiness_pipeline.py"),
            "--auto",
        ],
        cwd=str(workspace),
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert json.loads(completed.stderr) == {
        "error": "auto mode was removed; use --fix for readiness remediation."
    }


def test_fix_consumes_readiness_lock(tmp_path: Path, monkeypatch):
    install_fake_uv(tmp_path, monkeypatch)
    # Use a neutral script so workspace inference would default to inference;
    # the lock will hold training, proving fix seeds from the lock.
    workspace = make_workspace(tmp_path, script_name="run.py", body="print('run')\n")
    output_dir = tmp_path / "out"

    # Run check with explicit target to force the lock to training
    run_pipeline(
        "--mode",
        "check",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "training",
        "--non-interactive",
        cwd=workspace,
    )
    lock = json.loads((output_dir / "latest" / "readiness-agent" / "workspace-readiness.lock.json").read_text(encoding="utf-8"))
    assert lock["target"]["value"] == "training"

    # Now run fix without explicit --target; it should seed from the lock
    run_pipeline(
        "--mode",
        "fix",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        cwd=workspace,
    )

    _, verdict = load_report_pair(output_dir / "report.json")
    assert verdict["target"] == "training"
    env_json = json.loads((output_dir / "meta" / "env.json").read_text(encoding="utf-8"))
    assert env_json.get("lock_warning") is None


def test_fix_refreshes_lock_after_remediation(tmp_path: Path, monkeypatch):
    install_fake_uv(tmp_path, monkeypatch)
    workspace = make_workspace(tmp_path)
    output_dir = tmp_path / "out"

    # Produce an initial lock
    run_pipeline(
        "--mode",
        "check",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--non-interactive",
        cwd=workspace,
    )
    initial_lock = json.loads((output_dir / "latest" / "readiness-agent" / "workspace-readiness.lock.json").read_text(encoding="utf-8"))

    # Run fix; it should refresh the lock after remediation
    run_pipeline(
        "--mode",
        "fix",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        cwd=workspace,
    )

    refreshed_lock_path = output_dir / "latest" / "readiness-agent" / "workspace-readiness.lock.json"
    assert refreshed_lock_path.exists()
    refreshed_lock = json.loads(refreshed_lock_path.read_text(encoding="utf-8"))
    assert refreshed_lock["producer"] == "readiness-agent"
    assert refreshed_lock["schema_version"] == "readiness.assessment.v1"
    assert refreshed_lock["status"] in {"READY", "WARN", "BLOCKED"}


def test_check_task_smoke_executes_real_workload_command(tmp_path: Path, fake_selected_python: Path):
    workspace = make_workspace(
        tmp_path,
        script_name="train.py",
        body="print('task smoke ok')\n",
    )
    output_dir = tmp_path / "out"

    run_pipeline(
        "--mode",
        "check",
        "--working-dir",
        str(workspace),
        "--output-dir",
        str(output_dir),
        "--target",
        "training",
        "--selected-python",
        str(fake_selected_python),
        "--task-smoke-cmd",
        "python train.py --max_steps 1",
        "--confirm",
        "run_task_smoke=true",
        "--non-interactive",
        cwd=workspace,
    )

    lock = json.loads((output_dir / "latest" / "readiness-agent" / "workspace-readiness.lock.json").read_text(encoding="utf-8"))
    task_smoke = lock.get("task_smoke") or {}
    assert task_smoke.get("status") == "pass"
    assert task_smoke.get("approved_by_user") is True


def test_check_config_path_outside_workspace_is_blocked(tmp_path: Path, fake_selected_python: Path):
    workspace = make_workspace(tmp_path, script_name="run.py", body="print('run')\n")
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
        "--selected-python",
        str(fake_selected_python),
        "--config-path",
        "../../../etc/passwd",
        "--non-interactive",
        cwd=workspace,
    )

    lock = json.loads((output_dir / "latest" / "readiness-agent" / "workspace-readiness.lock.json").read_text(encoding="utf-8"))
    config_check = next((c for c in lock.get("checks", []) if c.get("name") == "config_readability"), None)
    assert config_check is not None
    assert config_check["status"] == "fail"
    assert "outside the workspace root" in config_check.get("stderr", "")
