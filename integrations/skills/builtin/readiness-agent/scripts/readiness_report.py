#!/usr/bin/env python3
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple
from uuid import uuid4

from readiness_core import make_check


READY_LEVELS = {"runtime_smoke", "task_smoke"}


def derive_evidence_level(checks: List[dict]) -> str:
    ok_ids = {str(item.get("id")) for item in checks if str(item.get("status") or "").lower() == "ok"}
    if "task-smoke-executed" in ok_ids:
        return "task_smoke"
    if "runtime-smoke" in ok_ids:
        return "runtime_smoke"
    if "framework-importability" in ok_ids:
        return "import"
    return "structural"


def check_by_id(checks: List[dict], check_id: str) -> Optional[dict]:
    for item in checks:
        if str(item.get("id")) == check_id:
            return item
    return None


def interpret_task_smoke_state(target: dict, checks: List[dict]) -> str:
    if not target.get("task_smoke_cmd"):
        return "not_requested"
    task_smoke = check_by_id(checks, "task-smoke-executed")
    if not task_smoke:
        return "missing_result"
    status = str(task_smoke.get("status") or "").lower()
    if status == "ok":
        return "passed"
    if status == "block":
        return "failed"
    if status == "skipped":
        return "skipped"
    return "unknown"


def scopes_for_check(check_id: str) -> Set[str]:
    mapping = {
        "python-selected-env": {"python-environment"},
        "python-selected-python": {"python-environment"},
        "framework-importability": {"framework"},
        "framework-compatibility": {"framework"},
        "runtime-smoke-framework": {"framework", "runtime-smoke"},
        "runtime-smoke-script-parse": {"workspace-assets", "runtime-smoke"},
        "runtime-smoke": {"runtime-smoke"},
        "runtime-dependencies": {"framework", "runtime-smoke"},
        "task-smoke-executed": {"task-smoke", "target"},
        "target-stability": {"target"},
        "framework-selection": {"framework"},
        "workspace-entry-script": {"workspace-assets"},
        "workspace-model-path": {"workspace-assets"},
        "workspace-dataset-path": {"workspace-assets"},
        "workspace-checkpoint-path": {"workspace-assets"},
    }
    return mapping.get(check_id, set())


def derive_revalidation_state(fix_applied: dict, checks: List[dict]) -> Tuple[bool, List[str], List[str]]:
    executed_actions = fix_applied.get("executed_actions") or []
    if not executed_actions:
        return True, [], []
    required_scopes = sorted({scope for scope in fix_applied.get("needs_revalidation") or [] if scope})
    if not required_scopes:
        return True, [], []
    covered_scopes = sorted({scope for item in checks for scope in scopes_for_check(str(item.get("id") or ""))})
    missing = [scope for scope in required_scopes if scope not in covered_scopes]
    return not missing, required_scopes, covered_scopes


def prompt_to_run_model_script(status: str, can_run: bool) -> str:
    if status == "READY":
        return "Readiness looks good. Do you want me to run the real model script now?"
    if can_run:
        return "Readiness finished with warnings. Do you want me to run the real model script now?"
    return "Readiness is not fully certified yet. Do you want me to try running the real model script now, or should I stop at the report?"


def synthesize_user_result(
    target: dict,
    normalized: dict,
    checks: List[dict],
    fix_applied: dict,
) -> Tuple[str, bool, str, str]:
    blockers = normalized.get("blockers_detailed") or []
    warnings = normalized.get("warnings_detailed") or []
    evidence_level = derive_evidence_level(checks)
    target_type = target.get("target_type") or "inference"
    task_smoke_state = interpret_task_smoke_state(target, checks)
    revalidated, _, _ = derive_revalidation_state(fix_applied, checks)

    if blockers:
        if any(str(item.get("id")) == "python-selected-env" for item in blockers):
            return (
                "BLOCKED",
                False,
                f"{target_type.capitalize()} is blocked because no usable workspace-local Python environment is selected.",
                "Create or repair a workspace-local environment, then rerun readiness.",
            )
        return (
            "BLOCKED",
            False,
            f"{target_type.capitalize()} is blocked because required readiness prerequisites remain unresolved.",
            "Resolve blockers and rerun readiness.",
        )

    if task_smoke_state == "failed":
        return (
            "BLOCKED",
            False,
            f"{target_type.capitalize()} is blocked because the explicit task smoke command failed.",
            "Inspect the task smoke failure and rerun readiness.",
        )

    can_run = evidence_level in READY_LEVELS and revalidated
    if warnings:
        summary = (
            f"{target_type.capitalize()} can probably run, but warnings still reduce readiness confidence."
            if can_run
            else f"{target_type.capitalize()} is not fully certified because warnings still limit readiness confidence."
        )
        return "WARN", can_run, summary, prompt_to_run_model_script("WARN", can_run)

    if can_run:
        return (
            "READY",
            True,
            f"Current environment is ready for {target_type}.",
            prompt_to_run_model_script("READY", True),
        )

    return (
        "WARN",
        False,
        f"{target_type.capitalize()} appears plausible, but runtime evidence is not strong enough for READY.",
        prompt_to_run_model_script("WARN", False),
    )


def build_selected_environment_guidance(target: dict, dependency_closure: dict) -> dict:
    python_env = dependency_closure.get("layers", {}).get("python_environment", {})
    selected_env_root = python_env.get("selected_env_root")
    selected_python = python_env.get("probe_python_path")
    working_dir = target.get("working_dir") or dependency_closure.get("working_dir")
    readiness_env_path = str((Path(working_dir).resolve() / ".readiness.env")) if working_dir else None
    guidance = {
        "selected_env_root": selected_env_root,
        "selected_python": selected_python,
        "system_python_allowed": False,
        "readiness_env_path": readiness_env_path,
    }
    if readiness_env_path:
        guidance["source_command"] = f"source {shlex.quote(readiness_env_path)}"
    if selected_python:
        guidance["run_command"] = f"{shlex.quote(selected_python)} <script.py>"
    else:
        guidance["message"] = "Workspace-local Python is unresolved. Do not fall back to system python."
    return guidance


def build_remote_asset_guidance(dependency_closure: dict) -> Optional[dict]:
    remote_assets = dependency_closure.get("layers", {}).get("remote_assets", {})
    assets = remote_assets.get("assets") or {}
    if not assets:
        return None
    cache_layout = remote_assets.get("cache_layout") or {}
    return {
        "mode": "remote-assets",
        "asset_kinds": sorted(assets.keys()),
        "hf_endpoint": remote_assets.get("hf_endpoint"),
        "hf_endpoint_source": remote_assets.get("hf_endpoint_source"),
        "endpoint_reachable": remote_assets.get("endpoint_reachable"),
        "endpoint_error": remote_assets.get("endpoint_error"),
        "cache_source": cache_layout.get("source"),
        "hf_home": cache_layout.get("hf_home"),
        "hub_cache": cache_layout.get("hub_cache"),
        "datasets_cache": cache_layout.get("datasets_cache"),
    }


def build_report(target: dict, normalized: dict, checks: List[dict], dependency_closure: dict, fix_applied: dict) -> dict:
    status, can_run, summary, next_action = synthesize_user_result(target, normalized, checks, fix_applied)
    revalidated, required_scopes, covered_scopes = derive_revalidation_state(fix_applied, checks)
    return {
        "schema_version": "readiness-agent/0.1",
        "skill": "readiness-agent",
        "status": status,
        "can_run": can_run,
        "target": target.get("target_type") or "inference",
        "summary": summary,
        "blockers": normalized.get("blockers") or [],
        "warnings": normalized.get("warnings") or [],
        "next_action": next_action,
        "execution_target": target,
        "evidence_level": derive_evidence_level(checks),
        "task_smoke_state": interpret_task_smoke_state(target, checks),
        "dependency_closure": dependency_closure,
        "checks": checks,
        "blockers_detailed": normalized.get("blockers_detailed") or [],
        "warnings_detailed": normalized.get("warnings_detailed") or [],
        "fix_applied": fix_applied,
        "revalidated": revalidated,
        "revalidation_required_scopes": required_scopes,
        "revalidation_covered_scopes": covered_scopes,
        "selected_environment_guidance": build_selected_environment_guidance(target, dependency_closure),
        "remote_asset_guidance": build_remote_asset_guidance(dependency_closure),
    }


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def artifact_ref(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def map_shared_status(verdict_status: str) -> str:
    if verdict_status == "READY":
        return "success"
    if verdict_status in {"WARN", "BLOCKED"}:
        return "partial"
    return "failed"


def build_shared_envelope(verdict: dict, output_json: Path, output_md: Path, output_verdict_json: Path) -> dict:
    run_root = output_json.parent
    timestamp = now_utc_iso()
    shared_status = map_shared_status(str(verdict.get("status") or ""))
    return {
        "schema_version": "1.0.0",
        "skill": "readiness-agent",
        "run_id": f"readiness-agent-{uuid4().hex[:12]}",
        "status": shared_status,
        "start_time": timestamp,
        "end_time": timestamp,
        "duration_sec": 0,
        "steps": [
            {
                "name": "readiness-assessment",
                "status": shared_status,
                "message": verdict.get("summary", ""),
            }
        ],
        "logs": [],
        "artifacts": [
            artifact_ref(output_md, run_root),
            artifact_ref(output_verdict_json, run_root),
        ],
        "env_ref": "meta/env.json",
        "inputs_ref": "meta/inputs.json",
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# Readiness Report",
        "",
        f"- status: `{report['status']}`",
        f"- can_run: `{str(report['can_run']).lower()}`",
        f"- target: `{report['target']}`",
        f"- summary: {report['summary']}",
        f"- next_action: {report['next_action']}",
        "",
    ]
    if report.get("blockers"):
        lines.extend(["## Blockers", ""])
        for item in report["blockers"]:
            lines.append(f"- {item}")
        lines.append("")
    if report.get("warnings"):
        lines.extend(["## Warnings", ""])
        for item in report["warnings"]:
            lines.append(f"- {item}")
        lines.append("")
    guidance = report.get("selected_environment_guidance") or {}
    lines.extend(["## Environment Guidance", ""])
    lines.append("- system_python_allowed: `false`")
    if guidance.get("readiness_env_path"):
        lines.append(f"- readiness_env_path: `{guidance['readiness_env_path']}`")
    if guidance.get("source_command"):
        lines.append(f"- source_command: `{guidance['source_command']}`")
    if guidance.get("selected_env_root"):
        lines.append(f"- selected_env_root: `{guidance['selected_env_root']}`")
    if guidance.get("selected_python"):
        lines.append(f"- selected_python: `{guidance['selected_python']}`")
    if guidance.get("run_command"):
        lines.append(f"- run_command: `{guidance['run_command']}`")
    if guidance.get("message"):
        lines.append(f"- message: {guidance['message']}")
    lines.append("")
    remote_guidance = report.get("remote_asset_guidance") or {}
    if remote_guidance:
        lines.extend(["## Remote Asset Guidance", ""])
        lines.append(f"- mode: `{remote_guidance.get('mode')}`")
        if remote_guidance.get("asset_kinds"):
            lines.append(f"- asset_kinds: `{', '.join(remote_guidance['asset_kinds'])}`")
        if remote_guidance.get("hf_endpoint"):
            lines.append(f"- hf_endpoint: `{remote_guidance['hf_endpoint']}`")
        if remote_guidance.get("endpoint_error"):
            lines.append(f"- endpoint_error: {remote_guidance['endpoint_error']}")
        if remote_guidance.get("hf_home"):
            lines.append(f"- hf_home: `{remote_guidance['hf_home']}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report_artifacts(output_dir: Path, verdict: dict) -> Tuple[dict, Path]:
    output_json = output_dir / "report.json"
    output_md = output_dir / "report.md"
    verdict_json = output_dir / "meta" / "readiness-verdict.json"

    verdict_json.parent.mkdir(parents=True, exist_ok=True)
    verdict_json.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(verdict), encoding="utf-8")
    envelope = build_shared_envelope(verdict, output_json, output_md, verdict_json)
    output_json.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return envelope, verdict_json


def build_legacy_report_from_assessment(assessment: dict) -> dict:
    status = str(assessment.get("status") or "BLOCKED")
    target_field = assessment.get("target") or {}
    target_type = target_field.get("value") if isinstance(target_field, dict) else None
    target_type = target_type or "inference"

    confirmed_fields = assessment.get("confirmed_fields") or {}
    runtime = assessment.get("runtime") or {}
    python_env = runtime.get("python") or {}
    framework = runtime.get("framework") or {}
    assets = assessment.get("assets") or {}
    checks = assessment.get("checks") or []
    task_smoke = assessment.get("task_smoke") or {}

    # Build legacy checks from assessment checks
    legacy_checks: List[dict] = []
    has_runtime_python = False
    has_framework_import = False
    has_task_smoke_pass = False
    has_task_smoke_fail = False
    has_selected_python_fail = False
    has_config_fail = False

    for check in checks:
        name = str(check.get("name") or "")
        cstatus = str(check.get("status") or "").lower()
        if name == "runtime_python" and cstatus == "pass":
            has_runtime_python = True
        if name.startswith("framework_import:") and cstatus == "pass":
            has_framework_import = True
        if name == "task_smoke_cmd":
            if cstatus == "pass":
                has_task_smoke_pass = True
            elif cstatus in {"fail", "failed"}:
                has_task_smoke_fail = True
        if name == "selected_python" and cstatus in {"fail", "failed"}:
            has_selected_python_fail = True
        if name == "config_readability" and cstatus in {"fail", "failed"}:
            has_config_fail = True

        # Map individual assessment checks to legacy checks
        if name.startswith("framework_import:"):
            legacy_status = "ok" if cstatus == "pass" else "block" if cstatus in {"fail", "failed", "timeout"} else "warn"
            legacy_checks.append(
                make_check("framework-importability", legacy_status, f"Framework import {name.split(':', 1)[1]} {'ok' if legacy_status == 'ok' else 'failed'}.")
            )
        elif name == "selected_python":
            if cstatus in {"fail", "failed"}:
                legacy_checks.append(
                    make_check(
                        "python-selected-env",
                        "block",
                        "No usable workspace-local Python environment is selected.",
                        category_hint="env",
                        remediable=True,
                        remediation_owner="readiness-agent",
                        revalidation_scope=["python-environment", "framework"],
                    )
                )
        elif name == "config_readability":
            legacy_status = "ok" if cstatus == "pass" else "block"
            legacy_checks.append(
                make_check("config-readability", legacy_status, f"Config readability {legacy_status}.")
            )

    # Synthesize runtime-smoke check
    if has_runtime_python and has_framework_import:
        legacy_checks.append(make_check("runtime-smoke", "ok", "Runtime smoke passed."))
    elif has_selected_python_fail:
        legacy_checks.append(
            make_check(
                "runtime-smoke",
                "block",
                "Runtime smoke could not run because entry script parsing prerequisites are unresolved.",
            )
        )
    else:
        legacy_checks.append(make_check("runtime-smoke", "warn", "Runtime smoke did not gather enough evidence to assess this workspace yet."))

    # Synthesize task-smoke-executed check
    if has_task_smoke_pass:
        legacy_checks.append(make_check("task-smoke-executed", "ok", "Explicit task smoke command completed successfully."))
    elif has_task_smoke_fail:
        legacy_checks.append(make_check("task-smoke-executed", "block", "Explicit task smoke command failed."))
    else:
        if task_smoke.get("command"):
            legacy_checks.append(make_check("task-smoke-executed", "skipped", "Task smoke was skipped or not confirmed."))
        else:
            legacy_checks.append(make_check("task-smoke-executed", "skipped", "No explicit task smoke command was requested."))

    # Derive evidence_level
    evidence_level = "structural"
    if has_task_smoke_pass:
        evidence_level = "task_smoke"
    elif has_runtime_python and has_framework_import:
        evidence_level = "runtime_smoke"
    elif has_framework_import:
        evidence_level = "import"

    # Derive can_run
    revalidated = True
    can_run = status == "READY" or (status == "WARN" and evidence_level in READY_LEVELS and revalidated)

    # Build blockers/warnings from checks and confirmed_fields
    blockers_detailed: List[dict] = []
    warnings_detailed: List[dict] = []
    for check in legacy_checks:
        cstatus = str(check.get("status") or "").lower()
        if cstatus == "block":
            blockers_detailed.append(
                {
                    "id": check.get("id"),
                    "summary": check.get("summary"),
                    "evidence": check.get("evidence") or [],
                    "category": "env" if check.get("id") == "python-selected-env" else "framework",
                    "severity": "high",
                    "remediable": check.get("remediable", True),
                    "remediation_owner": check.get("remediation_owner", "readiness-agent"),
                    "revalidation_scope": check.get("revalidation_scope") or [],
                }
            )
        elif cstatus == "warn":
            warnings_detailed.append(
                {
                    "id": check.get("id"),
                    "summary": check.get("summary"),
                    "evidence": check.get("evidence") or [],
                    "category": "framework",
                    "severity": "medium",
                    "remediable": False,
                    "remediation_owner": "readiness-agent",
                    "revalidation_scope": [],
                }
            )

    # Add warnings for skipped/unknown confirmations
    for field, value in confirmed_fields.items():
        if isinstance(value, dict) and value.get("source") in {"skipped", "unknown"}:
            warnings_detailed.append(
                {
                    "id": f"confirmation-{field}",
                    "summary": f"{field} was not confirmed and may be ambiguous.",
                    "evidence": [],
                    "category": "workspace",
                    "severity": "medium",
                    "remediable": False,
                    "remediation_owner": "workspace",
                    "revalidation_scope": [],
                }
            )

    blockers = [b["summary"] for b in blockers_detailed]
    warnings = [w["summary"] for w in warnings_detailed]

    # Determine summary and next_action
    if status == "READY":
        summary = f"Current environment is ready for {target_type}."
        next_action = "Readiness looks good. Do you want me to run the real model script now?"
    elif status == "WARN":
        if can_run:
            summary = f"{target_type.capitalize()} can probably run, but warnings still reduce readiness confidence."
            next_action = "Readiness finished with warnings. Do you want me to run the real model script now?"
        else:
            summary = f"{target_type.capitalize()} appears plausible, but runtime evidence is not strong enough for READY."
            next_action = "Readiness is not fully certified yet. Do you want me to try running the real model script now, or should I stop at the report?"
    else:
        summary = f"{target_type.capitalize()} is blocked because required readiness prerequisites remain unresolved."
        next_action = "Resolve blockers and rerun readiness."

    # Build dependency_closure
    dependency_closure = {
        "working_dir": None,
        "target_type": target_type,
        "layers": {
            "python_environment": {
                "selection_status": "selected" if python_env.get("executable") else "unselected",
                "selected_env_root": None,
                "probe_python_path": python_env.get("executable"),
                "python_version": python_env.get("version"),
            },
            "framework": {
                "framework_path": framework.get("name"),
                "framework_smoke": {"status": "passed" if has_framework_import else "failed"},
            },
            "workspace_assets": {
                "entry_script": {"exists": (assets.get("entry_script") or {}).get("exists", False)},
                "model_path": {"exists": (assets.get("model") or {}).get("exists", False)},
                "dataset_path": {"exists": (assets.get("dataset") or {}).get("exists", False)},
            },
        },
    }

    # Build selected_environment_guidance
    selected_python = python_env.get("executable")
    guidance: dict = {"selected_python": selected_python, "system_python_allowed": False, "readiness_env_path": None}
    if selected_python:
        guidance["run_command"] = f"{selected_python} <script.py>"
    else:
        guidance["message"] = "Workspace-local Python is unresolved. Do not fall back to system python."

    return {
        "schema_version": "readiness-agent/0.1",
        "skill": "readiness-agent",
        "status": status,
        "can_run": can_run,
        "target": target_type,
        "summary": summary,
        "blockers": blockers,
        "warnings": warnings,
        "next_action": next_action,
        "execution_target": {"target_type": target_type, "working_dir": dependency_closure["working_dir"]},
        "evidence_level": evidence_level,
        "task_smoke_state": "passed" if has_task_smoke_pass else "failed" if has_task_smoke_fail else "not_requested" if not task_smoke.get("command") else "skipped",
        "dependency_closure": dependency_closure,
        "checks": legacy_checks,
        "blockers_detailed": blockers_detailed,
        "warnings_detailed": warnings_detailed,
        "fix_applied": {"execute": False, "planned_actions": [], "results": [], "executed_actions": [], "failed_actions": [], "needs_revalidation": []},
        "revalidated": revalidated,
        "revalidation_required_scopes": [],
        "revalidation_covered_scopes": [],
        "selected_environment_guidance": guidance,
        "remote_asset_guidance": None,
    }
