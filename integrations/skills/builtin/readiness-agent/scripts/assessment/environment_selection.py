import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .confirmation_flow import option
from .runtime_probes import read_text


KNOWN_LAUNCHERS = ("python", "bash", "torchrun", "accelerate", "deepspeed", "msrun", "llamafactory-cli", "make")


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def python_candidates_for_env_root(env_root: Path) -> List[Path]:
    return [env_root / "bin" / "python", env_root / "Scripts" / "python.exe", env_root / "Scripts" / "python"]


def python_option(path: Path, description: str, *, recommended: bool = False) -> Dict[str, object]:
    return option(str(path), str(path), description, recommended)


def append_unique_candidate(candidates: List[Dict[str, object]], candidate: Dict[str, object]) -> None:
    value = str(candidate.get("value"))
    if value and not any(str(item.get("value")) == value for item in candidates):
        candidates.append(candidate)


def python_from_command(command: Optional[str], root: Path) -> Optional[Path]:
    if not command:
        return None
    try:
        tokens = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        tokens = command.strip().split()
    if not tokens:
        return None
    token = tokens[0]
    token_path = Path(token)
    token_name = token_path.name.lower()
    if token_name not in {"python", "python3", "python.exe"} and not token.lower().endswith(("/python", "/python3", "\\python.exe", "\\python")):
        return None
    if token_path.is_absolute() or token_path.parent != Path("."):
        return resolve_path(token, root)
    resolved = shutil.which(token)
    if resolved:
        return Path(resolved).resolve()
    return resolve_path(token, root)


def env_root_python(env_root_value: str, root: Path) -> Optional[Path]:
    env_root = resolve_path(env_root_value, root)
    for candidate_path in python_candidates_for_env_root(env_root):
        if candidate_path.exists():
            return candidate_path.resolve()
    return None


def active_env_python(root: Path) -> Optional[Path]:
    for env_var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        value = os.environ.get(env_var)
        if not value:
            continue
        candidate = env_root_python(value, root)
        if candidate:
            return candidate
    return None


def infer_target_candidates(args, files: List[Path]) -> List[Dict[str, object]]:
    target_hint = getattr(args, "target_hint", None) or getattr(args, "target", None)
    if target_hint and target_hint != "auto":
        return [option(str(target_hint), str(target_hint).capitalize(), "Provided by explicit target hint.", True)]

    names = [path.name.lower() for path in files]
    has_training = any(token in name for name in names for token in ("train", "finetune", "sft"))
    has_inference = any(token in name for name in names for token in ("infer", "predict", "generate", "chat"))
    if has_training and not has_inference:
        return [option("training", "Training", "Detected training-like entry files.", True)]
    if has_inference and not has_training:
        return [option("inference", "Inference", "Detected inference-like entry files.", True)]
    return [
        option("training", "Training", "Assess a training launch path.", True),
        option("inference", "Inference", "Assess an inference launch path."),
    ]


def infer_launcher_from_command(command: Optional[str]) -> Optional[str]:
    if not command:
        return None
    tokens = command.strip().split()
    if not tokens:
        return None
    first = Path(tokens[0]).name.lower()
    if first in {"python", "python3", "python.exe"}:
        return "python"
    if first in KNOWN_LAUNCHERS:
        return first
    if "llamafactory-cli" in command:
        return "llamafactory-cli"
    return None


def infer_launcher_candidates(args, files: List[Path]) -> List[Dict[str, object]]:
    launcher_hint = getattr(args, "launcher_hint", None)
    if launcher_hint and launcher_hint != "auto":
        return [option(str(launcher_hint), str(launcher_hint), "Provided by explicit launcher hint.", True)]
    command_launcher = infer_launcher_from_command(getattr(args, "launch_command", None))
    if command_launcher:
        return [option(command_launcher, command_launcher, "Detected from launch command.", True)]
    names = {path.name for path in files}
    if "Makefile" in names:
        return [option("make", "make", "Detected workspace Makefile.", True)]
    if any(path.suffix == ".sh" for path in files):
        return [option("bash", "bash", "Detected shell launch scripts.", True), option("python", "python")]
    if any(path.suffix == ".py" for path in files):
        return [option("python", "python", "Detected Python entry files.", True)]
    return [option("python", "python", "Common default launcher.", True), option("bash", "bash"), option("make", "make")]


def infer_framework_candidates(args, files: List[Path]) -> List[Dict[str, object]]:
    hint = getattr(args, "framework_hint", None)
    if hint and hint != "auto":
        value = "pytorch_npu" if hint == "pta" else str(hint)
        return [option(value, value, "Provided by explicit framework hint.", True)]
    text = "\n".join(read_text(path)[:4000] for path in files if path.suffix in {".py", ".txt", ".toml", ".md"})
    lowered = text.lower()
    has_mindspore = "mindspore" in lowered
    has_torch = "torch_npu" in lowered or "import torch" in lowered or "pytorch" in lowered
    if has_mindspore and has_torch:
        return [
            option("mindspore", "mindspore", "Detected MindSpore evidence."),
            option("pytorch_npu", "pytorch_npu", "Detected PyTorch or torch_npu evidence."),
            option("mixed", "mixed", "Intentionally assess a mixed MindSpore and PyTorch/NPU workspace."),
        ]
    if has_mindspore:
        return [option("mindspore", "mindspore", "Detected MindSpore evidence.", True)]
    if has_torch:
        return [option("pytorch_npu", "pytorch_npu", "Detected PyTorch or torch_npu evidence.", True)]
    return [option("mindspore", "mindspore", "Assess MindSpore runtime.", True), option("pytorch_npu", "pytorch_npu", "Assess PyTorch NPU runtime.")]


def current_python_option() -> Dict[str, object]:
    return option(sys.executable, Path(sys.executable).name, f"Current Python: {sys.executable}", True)


def infer_runtime_candidates(args, root: Path) -> List[Dict[str, object]]:
    selected_python = getattr(args, "selected_python", None)
    if selected_python:
        return [python_option(resolve_path(str(selected_python), root), "Provided by explicit selected Python.", recommended=True)]

    selected_env_root = getattr(args, "selected_env_root", None)
    if selected_env_root:
        selected_env_python = env_root_python(str(selected_env_root), root)
        if selected_env_python:
            return [python_option(selected_env_python, "Resolved from explicit environment root.", recommended=True)]
        return [option(str(resolve_path(str(selected_env_root), root)), str(resolve_path(str(selected_env_root), root)), "Provided by explicit environment root.", True)]

    candidates: List[Dict[str, object]] = []
    launch_python = python_from_command(getattr(args, "launch_command", None), root)
    if launch_python:
        append_unique_candidate(candidates, python_option(launch_python, "Detected from explicit launch command.", recommended=True))

    active_python = active_env_python(root)
    if active_python:
        append_unique_candidate(candidates, python_option(active_python, "Detected from active CONDA_PREFIX or VIRTUAL_ENV.", recommended=not candidates))

    for env_name in (".venv", "venv", ".env", "env"):
        env_root = root / env_name
        for candidate_path in python_candidates_for_env_root(env_root):
            if candidate_path.exists():
                append_unique_candidate(candidates, python_option(candidate_path.resolve(), "Detected workspace-local environment.", recommended=not candidates))
                break

    for spec_name in ("environment.yml", "environment.yaml", "conda.yaml", "conda.yml"):
        if (root / spec_name).exists() and not candidates:
            append_unique_candidate(candidates, current_python_option())
            candidates[-1]["description"] = f"Current Python; {spec_name} exists as environment evidence."

    resolved = shutil.which("python") or shutil.which("python3")
    if resolved:
        append_unique_candidate(candidates, python_option(Path(resolved).resolve(), f"System Python resolved on PATH: {resolved}", recommended=not candidates))
    append_unique_candidate(candidates, current_python_option())
    if len(candidates) == 1:
        candidates[0]["confidence"] = "low"
    return candidates
