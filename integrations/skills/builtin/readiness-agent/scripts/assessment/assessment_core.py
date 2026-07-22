from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .asset_registry import detect_asset_registry, workspace_lock_asset_view
from .cann_evidence import collect_cann_evidence
from .confirmation_flow import (
    build_resume_command,
    build_task_smoke_resume_command,
    needs_confirmation,
    parse_confirmations,
    portable_question,
    selected_value,
    task_smoke_portable_question,
    validate_confirmation,
)
from .environment_selection import infer_framework_candidates, infer_launcher_candidates, infer_runtime_candidates, infer_target_candidates

@dataclass(frozen=True)
class AssessmentExecutionPolicy:
    allow_install: bool = False
    allow_download: bool = False
    allow_env_create: bool = False
    allow_source_write: bool = False
    allow_config_write: bool = False
    allow_real_workload: bool = False
    allow_task_smoke: bool = False
    allow_readiness_artifact_write: bool = True


from .near_launch_probes import build_near_launch_checks
from .runtime_probes import list_workspace_files
from .schemas import CONFIRMATION_SEQUENCE, IDENTITY, TASK_SMOKE_CONFIRM_FIELD
from .assessment_report import write_assessment_artifacts
from .verdict_builder import build_assessment_verdict


ENABLE_ASSESSMENT_MODE = True
ASSESSMENT_EXECUTION_POLICY = AssessmentExecutionPolicy()


def build_candidates(args, root: Path) -> Dict[str, List[Dict[str, object]]]:
    files = list_workspace_files(root)
    return {
        "target": infer_target_candidates(args, files),
        "launcher": infer_launcher_candidates(args, files),
        "framework": infer_framework_candidates(args, files),
        "runtime_environment": infer_runtime_candidates(args, root),
    }


def run_assessment_pipeline(args):
    if not ENABLE_ASSESSMENT_MODE:
        return {
            **IDENTITY,
            "status": "BLOCKED",
            "reason": "assessment mode is disabled",
        }

    policy = ASSESSMENT_EXECUTION_POLICY
    if not policy.allow_readiness_artifact_write:
        return {
            **IDENTITY,
            "status": "BLOCKED",
            "reason": "assessment readiness artifact writes are disabled",
        }

    root = Path(getattr(args, "working_dir", None) or ".").resolve()
    confirmations = parse_confirmations(getattr(args, "confirm", []))
    candidates = build_candidates(args, root)

    confirmed_fields: Dict[str, object] = {}
    for field in CONFIRMATION_SEQUENCE:
        field_candidates = candidates.get(field, [])
        if field in confirmations and not validate_confirmation(field, confirmations[field], field_candidates):
            question = portable_question(field, field_candidates)
            verdict = {
                **IDENTITY,
                "status": "NEEDS_CONFIRMATION",
                "phase": "awaiting_confirmation",
                "reason": "invalid_confirmation",
                "invalid_confirmation": {
                    "field": field,
                    "value": confirmations[field],
                },
                "current_confirmation": {
                    "field": field,
                    "options": field_candidates,
                    "portable_question": question,
                },
                "portable_question": question,
                "confirmed_fields": confirmed_fields,
                "resume_command": build_resume_command(args, field, field_candidates, {key: value for key, value in confirmations.items() if key != field}),
            }
            write_assessment_artifacts(args, root, verdict)
            return verdict
        if needs_confirmation(args, field, field_candidates, confirmations):
            question = portable_question(field, field_candidates)
            verdict = {
                **IDENTITY,
                "status": "NEEDS_CONFIRMATION",
                "phase": "awaiting_confirmation",
                "current_confirmation": {
                    "field": field,
                    "options": field_candidates,
                    "portable_question": question,
                },
                "portable_question": question,
                "confirmed_fields": confirmed_fields,
                "resume_command": build_resume_command(args, field, field_candidates, confirmations),
            }
            write_assessment_artifacts(args, root, verdict)
            return verdict
        confirmed_fields[field] = selected_value(field, field_candidates, confirmations)

    # Task smoke confirmation gate
    task_smoke_cmd = getattr(args, "task_smoke_cmd", None)
    run_task_smoke = confirmations.get(TASK_SMOKE_CONFIRM_FIELD)
    if task_smoke_cmd and run_task_smoke is None:
        if getattr(args, "non_interactive", False):
            run_task_smoke = "false"
        else:
            question = task_smoke_portable_question(str(task_smoke_cmd))
            verdict = {
                **IDENTITY,
                "status": "NEEDS_CONFIRMATION",
                "phase": "awaiting_task_smoke",
                "portable_question": question,
                "confirmed_fields": confirmed_fields,
                "resume_command": build_task_smoke_resume_command(args, confirmations),
            }
            write_assessment_artifacts(args, root, verdict)
            return verdict

    assets = workspace_lock_asset_view(detect_asset_registry(args, root, confirmed_fields.get("target", {}).get("value")))
    cann = collect_cann_evidence(args, root)
    checks = build_near_launch_checks(args, root, confirmed_fields, run_task_smoke=run_task_smoke)
    verdict = build_assessment_verdict(
        identity=IDENTITY,
        confirmed_fields=confirmed_fields,
        assets=assets,
        runtime={
            "framework": confirmed_fields.get("framework"),
            "python": confirmed_fields.get("runtime_environment"),
        },
        checks=checks,
        cann=cann,
        task_smoke={
            "command": task_smoke_cmd,
            "status": _task_smoke_status_from_checks(checks, run_task_smoke),
            "approved_by_user": run_task_smoke == "true",
        },
    )
    write_assessment_artifacts(args, root, verdict)
    return verdict


def _task_smoke_status_from_checks(checks: List[dict], run_task_smoke: Optional[str]) -> str:
    task_smoke_check = next((c for c in checks if c.get("name") == "task_smoke_cmd"), None)
    if not task_smoke_check:
        return "not_provided"
    status = task_smoke_check.get("status")
    if status == "pass":
        return "pass"
    if status == "fail":
        return "fail"
    if run_task_smoke == "false":
        return "skipped"
    return "not_provided"
