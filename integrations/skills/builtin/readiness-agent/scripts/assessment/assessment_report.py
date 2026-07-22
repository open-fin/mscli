import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from .schemas import IDENTITY


def canonical_output_dir(args, root: Path) -> Path:
    return Path(getattr(args, "output_dir", None)).resolve() if getattr(args, "output_dir", None) else root / "readiness-output"


def canonical_latest_dir(args, root: Path) -> Path:
    return canonical_output_dir(args, root) / "latest" / "readiness-agent"


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def attempt_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def workspace_lock_from_verdict(verdict: Dict[str, object]) -> Dict[str, object]:
    target = verdict.get("target")
    launcher = verdict.get("launcher") or {}
    runtime = verdict.get("runtime") or {}
    framework = runtime.get("framework") if isinstance(runtime, dict) else None
    python_env = runtime.get("python") if isinstance(runtime, dict) else None

    lock = {
        **IDENTITY,
        "status": verdict.get("status"),
        "target": target,
        "launcher": {
            "type": launcher.get("value") if isinstance(launcher, dict) else None,
            "command": launcher.get("value") if isinstance(launcher, dict) else None,
            "source": launcher.get("source") if isinstance(launcher, dict) else "unknown",
        },
        "runtime": {
            "python": {
                "executable": python_env.get("value") if isinstance(python_env, dict) else None,
                "version": "unknown",
                "source": python_env.get("source") if isinstance(python_env, dict) else "unknown",
            },
            "framework": {
                "name": framework.get("value") if isinstance(framework, dict) else None,
                "source": framework.get("source") if isinstance(framework, dict) else "unknown",
            },
            "cann": runtime.get("cann") if isinstance(runtime.get("cann"), dict) else {
                "status": "unknown",
                "source": "not_checked",
            },
        },
        "assets": verdict.get("assets") or {
            "model": {"status": "unknown"},
            "dataset": {"status": "unknown"},
            "checkpoint": {"status": "unknown"},
        },
        "checks": verdict.get("checks") or [],
        "current_confirmation": verdict.get("current_confirmation"),
    }
    task_smoke = verdict.get("task_smoke")
    if isinstance(task_smoke, dict):
        lock["task_smoke"] = task_smoke
    return lock


def confirmation_latest_from_verdict(verdict: Dict[str, object]) -> Dict[str, object]:
    return {
        **IDENTITY,
        "status": verdict.get("status"),
        "current_confirmation": verdict.get("current_confirmation"),
    }


def run_ref_from_verdict(verdict: Dict[str, object]) -> Dict[str, object]:
    return {
        **IDENTITY,
        "attempt_id": str(verdict.get("attempt_id") or attempt_id()),
        "latest_status": verdict.get("status"),
    }


def attempt_dir(args, root: Path, verdict: Dict[str, object]) -> Path:
    return canonical_output_dir(args, root) / "attempts" / str(verdict.get("attempt_id") or attempt_id())


def write_attempt_artifacts(args, root: Path, verdict: Dict[str, object]) -> None:
    base_dir = attempt_dir(args, root, verdict)
    if verdict.get("status") == "NEEDS_CONFIRMATION":
        write_json(base_dir / "current" / "meta" / "readiness-verdict.json", verdict)
        write_json(base_dir / "current" / "artifacts" / "confirmation-step.json", confirmation_latest_from_verdict(verdict))
        return
    if verdict.get("status") in {"READY", "WARN", "BLOCKED"}:
        write_json(base_dir / "final" / "report.json", verdict)
        write_json(base_dir / "final" / "artifacts" / "workspace-readiness.lock.json", workspace_lock_from_verdict(verdict))


def write_assessment_artifacts(args, root: Path, verdict: Dict[str, object]) -> None:
    if "attempt_id" not in verdict:
        verdict["attempt_id"] = attempt_id()
    latest_dir = canonical_latest_dir(args, root)
    if verdict.get("status") in {"READY", "WARN", "BLOCKED"}:
        write_json(latest_dir / "workspace-readiness.lock.json", workspace_lock_from_verdict(verdict))
    if verdict.get("status") == "NEEDS_CONFIRMATION":
        write_json(latest_dir / "confirmation-latest.json", confirmation_latest_from_verdict(verdict))
    write_json(latest_dir / "run-ref.json", run_ref_from_verdict(verdict))
    write_attempt_artifacts(args, root, verdict)
