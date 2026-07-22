import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .framework_probes import build_framework_checks
from .runtime_probes import run_safe_probe


SAFE_HELP_LAUNCHERS = {"python", "bash", "make"}


def selected_python_value(confirmed_fields: Dict[str, object]) -> str | None:
    value = confirmed_fields.get("runtime_environment")
    if isinstance(value, dict) and value.get("value"):
        return str(value["value"])
    return None


def selected_launcher_value(confirmed_fields: Dict[str, object]) -> str | None:
    value = confirmed_fields.get("launcher")
    if isinstance(value, dict) and value.get("value"):
        return str(value["value"])
    return None


def build_task_smoke_check(command: str, selected_python: Optional[str], root: Path, timeout_seconds: int) -> dict:
    if not selected_python:
        return {
            "name": "task_smoke_cmd",
            "command": command,
            "status": "skipped",
            "reason": "selected python is unavailable for task smoke",
        }
    parts = shlex.split(command)
    if not parts:
        return {
            "name": "task_smoke_cmd",
            "command": command,
            "status": "fail",
            "reason": "task smoke command is empty after parsing",
        }
    if parts[0] in {"python", "python3"}:
        parts[0] = selected_python
    try:
        completed = subprocess.run(
            parts,
            cwd=str(root),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return {
            "name": "task_smoke_cmd",
            "command": " ".join(parts),
            "status": "fail",
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "timeout_seconds": timeout_seconds,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": "task_smoke_cmd",
            "command": " ".join(parts),
            "status": "timeout",
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timeout_seconds": timeout_seconds,
        }
    return {
        "name": "task_smoke_cmd",
        "command": " ".join(parts),
        "status": "pass" if completed.returncode == 0 else "fail",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timeout_seconds": timeout_seconds,
    }


def build_near_launch_checks(args, root: Path, confirmed_fields: Dict[str, object], run_task_smoke: Optional[str] = None) -> List[dict]:
    checks: List[dict] = []
    selected_python = selected_python_value(confirmed_fields)
    if selected_python:
        checks.append(run_safe_probe([selected_python, "--version"], name="runtime_python", timeout_seconds=getattr(args, "timeout_seconds", 10), cwd=root))
        checks.extend(build_framework_checks(args, root, confirmed_fields, selected_python))
    else:
        checks.append({
            "name": "selected_python",
            "status": "fail",
            "reason": "No usable workspace-local Python environment is selected.",
            "blocker": True,
        })

    launcher = selected_launcher_value(confirmed_fields)
    if launcher and launcher in SAFE_HELP_LAUNCHERS and launcher not in {"python", "bash"}:
        checks.append(run_safe_probe([launcher, "--help"], name="launcher_help", timeout_seconds=getattr(args, "timeout_seconds", 10), cwd=root))

    config_path = getattr(args, "config_path", None)
    if config_path:
        path = Path(str(config_path)).expanduser()
        if not path.is_absolute():
            path = root / path
        try:
            path.resolve().relative_to(root.resolve())
            within_root = True
        except ValueError:
            within_root = False
        if within_root and path.is_file():
            checks.append({
                "name": "config_readability",
                "command": f"read {path}",
                "status": "pass",
                "path": str(path.resolve()),
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "timeout_seconds": 0,
            })
        else:
            checks.append({
                "name": "config_readability",
                "command": f"read {path}",
                "status": "fail",
                "path": str(path.resolve()),
                "returncode": None,
                "stdout": "",
                "stderr": "config path is outside the workspace root" if not within_root else "config file is not readable",
                "timeout_seconds": 0,
            })

    task_smoke_cmd = getattr(args, "task_smoke_cmd", None)
    if task_smoke_cmd:
        if run_task_smoke == "true":
            checks.append(build_task_smoke_check(str(task_smoke_cmd), selected_python, root, getattr(args, "timeout_seconds", 10)))
        elif run_task_smoke == "false":
            checks.append({
                "name": "task_smoke_cmd",
                "command": str(task_smoke_cmd),
                "status": "skipped",
                "reason": "user skipped task smoke execution",
            })
        else:
            checks.append({
                "name": "task_smoke_cmd",
                "command": str(task_smoke_cmd),
                "status": "skipped",
                "reason": "task smoke pending user confirmation",
            })
    else:
        checks.append({
            "name": "task_smoke_cmd",
            "command": None,
            "status": "not_provided",
            "reason": "no task smoke command provided",
        })
    return checks
