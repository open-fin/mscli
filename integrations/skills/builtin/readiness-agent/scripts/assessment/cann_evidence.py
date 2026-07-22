import os
from pathlib import Path
from typing import Dict, Iterable, Optional


ENV_PATHS = (
    "ASCEND_HOME",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
)
COMMON_PATHS = (
    Path("/usr/local/Ascend/ascend-toolkit"),
    Path("/usr/local/Ascend/latest"),
)


def resolve_path(raw_path: str, root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def normalize_cann_path(path: Path) -> Path:
    if path.name == "set_env.sh":
        return path.parent
    return path


def read_version(root: Path) -> Optional[str]:
    for candidate in (root / "version.info", root.parent / "version.info"):
        if candidate.exists() and candidate.is_file():
            text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
            return text.splitlines()[0] if text else "unknown"
    return None


def evidence_for_path(path: Path, *, source: str) -> Dict[str, object]:
    root = normalize_cann_path(path)
    set_env = root / "set_env.sh"
    version = read_version(root)
    evidence = []
    if root.exists():
        evidence.append({"kind": "path", "path": str(root)})
    if set_env.exists():
        evidence.append({"kind": "set_env", "path": str(set_env)})
    if version is not None:
        evidence.append({"kind": "version", "value": version})
    status = "detected" if root.exists() and (set_env.exists() or version is not None) else "incomplete" if root.exists() else "missing"
    payload: Dict[str, object] = {
        "status": status,
        "source": source,
        "path": str(root),
        "candidate_paths": [str(root)],
        "evidence": evidence,
    }
    if set_env.exists():
        payload["set_env"] = str(set_env)
    if version is not None:
        payload["version"] = version
    return payload


def first_detected(paths: Iterable[Path], *, source: str) -> Optional[Dict[str, object]]:
    for path in paths:
        evidence = evidence_for_path(path, source=source)
        if evidence["status"] != "missing":
            return evidence
    return None


def env_candidate_paths() -> list[Path]:
    candidates = []
    for name in ENV_PATHS:
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value).expanduser())
    for item in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if item and "Ascend" in item:
            candidates.append(Path(item).expanduser())
    for item in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if item and "Ascend" in item:
            candidates.append(Path(item).expanduser())
    return candidates


def collect_cann_evidence(args, root: Path) -> Dict[str, object]:
    explicit_path = getattr(args, "cann_path", None)
    if explicit_path:
        return evidence_for_path(resolve_path(str(explicit_path), root), source="explicit")

    env_paths = env_candidate_paths()
    env_evidence = first_detected(env_paths, source="env")
    if env_evidence:
        env_evidence["candidate_paths"] = [str(path) for path in env_paths]
        return env_evidence
    if env_paths:
        return {
            "status": "missing",
            "source": "env",
            "candidate_paths": [str(path) for path in env_paths],
            "evidence": [],
        }

    common_evidence = first_detected(COMMON_PATHS, source="common_path")
    if common_evidence:
        common_evidence["candidate_paths"] = [str(path) for path in COMMON_PATHS]
        return common_evidence

    return {
        "status": "unknown",
        "source": "not_found",
        "candidate_paths": [str(path) for path in COMMON_PATHS],
        "evidence": [],
    }
