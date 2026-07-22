from typing import Dict, List, Optional


Check = Dict[str, object]
BLOCKING_CHECK_NAMES = {
    "runtime_python",
    "selected_python",
    "framework_import:mindspore",
    "framework_import:torch",
    "framework_import:torch_npu",
    "launcher_help",
    "config_readability",
    "safe_near_launch_probe",
    "task_smoke_cmd",
}
WARNING_ASSET_STATUSES = {"unknown", "remote_declared"}
BLOCKING_ASSET_STATUSES = {"missing"}


def confirmation_is_warning(value: object) -> bool:
    return isinstance(value, dict) and (value.get("source") in {"skipped", "unknown"} or value.get("value") is None)


def asset_is_blocking(asset: Dict[str, object]) -> bool:
    return bool(asset.get("required")) and asset.get("status") in BLOCKING_ASSET_STATUSES


def asset_is_warning(asset: Dict[str, object]) -> bool:
    return bool(asset.get("required")) and asset.get("status") in WARNING_ASSET_STATUSES


def check_is_blocking(check: Check) -> bool:
    if check.get("status") not in {"fail", "failed", "timeout"}:
        return False
    if check.get("blocker") is True:
        return True
    return str(check.get("name")) in BLOCKING_CHECK_NAMES


def check_is_warning(check: Check) -> bool:
    if check.get("name") == "task_smoke_cmd" and check.get("status") == "skipped":
        return False
    if str(check.get("name", "")).startswith("framework_import:") and check.get("status") == "pass" and check.get("version") == "unknown":
        return True
    return check.get("status") in {"warn", "warning", "unknown", "skipped"}


def cann_is_blocking(cann: Optional[Dict[str, object]]) -> bool:
    if not cann:
        return False
    return cann.get("status") == "missing" and cann.get("source") == "explicit"


def cann_is_warning(cann: Optional[Dict[str, object]]) -> bool:
    if not cann or cann_is_blocking(cann):
        return False
    return cann.get("status") in {"unknown", "missing", "incomplete", "warn"}


def decide_assessment_status(
    *,
    confirmed_fields: Dict[str, object],
    assets: Dict[str, Dict[str, object]],
    checks: Optional[List[Check]] = None,
    cann: Optional[Dict[str, object]] = None,
) -> str:
    checks = checks or []
    if any(asset_is_blocking(asset) for asset in assets.values()):
        return "BLOCKED"
    if any(check_is_blocking(check) for check in checks):
        return "BLOCKED"
    if cann_is_blocking(cann):
        return "BLOCKED"
    if any(confirmation_is_warning(value) for value in confirmed_fields.values()):
        return "WARN"
    if any(asset_is_warning(asset) for asset in assets.values()):
        return "WARN"
    if any(check_is_warning(check) for check in checks):
        return "WARN"
    if cann_is_warning(cann):
        return "WARN"
    return "READY"


def build_assessment_verdict(
    *,
    identity: Dict[str, object],
    confirmed_fields: Dict[str, object],
    assets: Dict[str, Dict[str, object]],
    runtime: Dict[str, object],
    checks: Optional[List[Check]] = None,
    cann: Optional[Dict[str, object]] = None,
    task_smoke: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    runtime_payload = dict(runtime)
    if cann is not None:
        runtime_payload["cann"] = cann
    verdict: Dict[str, object] = {
        **identity,
        "status": decide_assessment_status(
            confirmed_fields=confirmed_fields,
            assets=assets,
            checks=checks,
            cann=cann,
        ),
        "phase": "validated",
        "current_confirmation": None,
        "confirmed_fields": confirmed_fields,
        "target": confirmed_fields.get("target"),
        "launcher": confirmed_fields.get("launcher"),
        "runtime": runtime_payload,
        "assets": assets,
        "checks": checks or [],
    }
    if task_smoke is not None:
        verdict["task_smoke"] = task_smoke
    return verdict
