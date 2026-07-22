from pathlib import Path
from typing import Dict, List, Optional
import json

from .runtime_probes import run_safe_probe


FRAMEWORK_IMPORTS = {
    "mindspore": ["mindspore"],
    "pytorch_npu": ["torch", "torch_npu"],
    "mixed": ["mindspore", "torch", "torch_npu"],
}
PROBE_SCRIPT = """
import importlib.util
import importlib.metadata
import json
import sys

mode = sys.argv[-2]
payload = json.loads(sys.argv[-1])
packages = payload.get("packages", [])
if mode == "package_versions":
    versions = {}
    errors = {}
    for name in packages:
        if importlib.util.find_spec(name) is None:
            errors[name] = "not importable"
            continue
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    print(json.dumps({"versions": versions, "errors": errors}))
""".strip()
PROBE_CODE = f"exec({PROBE_SCRIPT!r})"


def selected_framework_value(confirmed_fields: Dict[str, object]) -> Optional[str]:
    value = confirmed_fields.get("framework")
    if isinstance(value, dict) and value.get("value"):
        return str(value["value"])
    return None


def framework_import_checks(selected_python: str, framework: str, *, timeout_seconds: int, cwd: Path) -> List[dict]:
    modules = FRAMEWORK_IMPORTS.get(framework, [])
    if not modules:
        return []
    result = run_safe_probe(
        [selected_python, "-c", PROBE_CODE, "package_versions", json.dumps({"packages": modules})],
        name=f"framework_import:{framework}",
        timeout_seconds=timeout_seconds,
        cwd=cwd,
    )
    versions, errors = parse_package_versions(result.get("stdout"))
    checks: List[dict] = []
    for module_name in modules:
        module_result = dict(result)
        module_result["name"] = f"framework_import:{module_name}"
        if module_name in errors:
            module_result["status"] = "fail"
            module_result["stderr"] = str(errors[module_name])
        elif result.get("status") == "pass":
            module_result["version"] = versions.get(module_name, "unknown")
        checks.append(module_result)
    return checks


def parse_package_versions(stdout: object) -> tuple[dict[str, str], dict[str, str]]:
    try:
        payload = json.loads(str(stdout or ""))
    except (ValueError, TypeError):
        return {}, {}
    versions = payload.get("versions") if isinstance(payload, dict) else {}
    errors = payload.get("errors") if isinstance(payload, dict) else {}
    return versions or {}, errors or {}


def build_framework_checks(args, root: Path, confirmed_fields: Dict[str, object], selected_python: Optional[str]) -> List[dict]:
    framework = selected_framework_value(confirmed_fields)
    if not selected_python or not framework:
        return []
    return framework_import_checks(selected_python, framework, timeout_seconds=getattr(args, "timeout_seconds", 10), cwd=root)
