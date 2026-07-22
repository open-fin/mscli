import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SCRIPTS = ROOT / "scripts"
CONTRACT = ROOT / "contract"
READINESS_VERDICT_REF = Path("meta/readiness-verdict.json")


def run_pipeline(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "run_readiness_pipeline.py"), *args],
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=True,
    )


def run_pipeline_allow_confirmation(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_readiness_pipeline.py"), *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    assert completed.returncode in {0, 20}
    return completed


def make_workspace(tmp_path: Path, script_name: str = "train.py", body: str = "print('train')\n") -> Path:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    (workspace / script_name).write_text(body, encoding="utf-8")
    (workspace / "model").mkdir()
    return workspace


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fake_framework_python(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "fake-framework-python.py"
    script.write_text(
        """
import json
import subprocess
import sys

if len(sys.argv) >= 3 and sys.argv[1] == "-c":
    if "package_versions" in " ".join(sys.argv):
        command = " ".join(sys.argv)
        versions = {}
        errors = {}
        for name in ("mindspore", "torch", "torch_npu"):
            if name in command:
                versions[name] = "1.0.0"
        print(json.dumps({"versions": versions, "errors": errors}))
        raise SystemExit(0)
completed = subprocess.run([sys.executable, *sys.argv[1:]])
raise SystemExit(completed.returncode)
""".lstrip(),
        encoding="utf-8",
    )
    if os.name == "nt":
        launcher = tmp_path / "fake-framework-python.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "%~dp0fake-framework-python.py" %*\r\n', encoding="utf-8")
        return launcher
    script.chmod(script.stat().st_mode | 0o111)
    return script


def assessment_asset_args(**overrides):
    values = {
        "config_path": None,
        "model_path": None,
        "model_hub_id": None,
        "dataset_path": None,
        "dataset_hub_id": None,
        "dataset_split": None,
        "checkpoint_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def runtime_args(**overrides):
    values = {"selected_python": None, "selected_env_root": None, "launch_command": None}
    values.update(overrides)
    return SimpleNamespace(**values)


def args(**values):
    return SimpleNamespace(**values)
