import os
import subprocess
from pathlib import Path
from typing import List, Optional

WORKLOAD_SCRIPT_NAMES = {
    "train.py",
    "finetune.py",
    "sft.py",
    "infer.py",
    "inference.py",
    "predict.py",
    "generate.py",
    "chat.py",
}
WORKLOAD_LAUNCHERS = {"torchrun", "msrun", "accelerate", "deepspeed", "llamafactory-cli"}


def is_real_workload_command(command: list[str]) -> bool:
    if not command:
        return False
    lowered = [str(item).lower() for item in command]
    joined = " ".join(lowered)
    if "task_smoke_cmd" in joined:
        return True
    launcher = Path(lowered[0]).name
    if launcher in WORKLOAD_LAUNCHERS:
        return True
    if launcher in {"python", "python3", "python.exe"}:
        return any(Path(token).name in WORKLOAD_SCRIPT_NAMES for token in lowered[1:])
    return Path(launcher).name in WORKLOAD_SCRIPT_NAMES


def ensure_safe_probe_command(command: list[str]) -> None:
    if is_real_workload_command(command):
        raise ValueError(f"assessment refuses to run real workload command: {' '.join(command)}")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def list_workspace_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    skip_dirs = {".git", "__pycache__", "readiness-output", ".venv", "venv", "env", "node_modules"}
    root_depth = len(root.resolve().parts)
    for current_root, dirnames, filenames in os.walk(root):
        current = Path(current_root)
        try:
            depth = len(current.resolve().parts) - root_depth
        except OSError:
            continue
        if depth > 2:
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in skip_dirs and not name.startswith(".")]
        for name in filenames:
            files.append(current / name)
    return files


def run_safe_probe(
    command: list[str],
    *,
    name: str,
    timeout_seconds: int = 10,
    cwd: Optional[Path] = None,
) -> dict:
    ensure_safe_probe_command(command)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return {
            "name": name,
            "command": " ".join(command),
            "status": "fail",
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "timeout_seconds": timeout_seconds,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "command": " ".join(command),
            "status": "timeout",
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timeout_seconds": timeout_seconds,
        }
    return {
        "name": name,
        "command": " ".join(command),
        "status": "pass" if completed.returncode == 0 else "fail",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "timeout_seconds": timeout_seconds,
    }
