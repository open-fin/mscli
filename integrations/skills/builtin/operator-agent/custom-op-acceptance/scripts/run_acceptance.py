#!/usr/bin/env python3
"""Prepare acceptance assets and optionally run generated pytest cases."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import DEFAULT_PROJECT_ROOT, sanitize_identifier, write_json  # noqa: E402
from evidence import (  # noqa: E402
    analyze_pytest_file,
    collect_static_audits as _collect_static_audits,
    parse_pytest_output,
    parse_pytest_verbose_cases,
)
from generate_cases import generate_acceptance_cases  # noqa: E402
from generate_report import write_report  # noqa: E402
from inspect_operator import collect_operator_meta  # noqa: E402
from scaffold_acceptance_artifacts import scaffold_acceptance_artifacts  # noqa: E402


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _npu_smi_output() -> str:
    completed = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, check=False)
    return completed.stdout + "\n" + completed.stderr


def _device_requirement_missing(meta: dict[str, Any] | None) -> str | None:
    requirements = (meta or {}).get("platform_requirements", {})
    required_devices = requirements.get("required_devices", [])
    if not required_devices:
        return None
    output = _npu_smi_output()
    if any(str(device) in output for device in required_devices):
        return None
    return "device_requires:" + "/".join(str(device) for device in required_devices)


def run_preflight(require_pta: bool, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    missing: list[str] = []
    if not _module_available("mindspore"):
        missing.append("mindspore")
    if require_pta:
        if not _module_available("torch"):
            missing.append("torch")
        if not _module_available("torch_npu"):
            missing.append("torch_npu")
    if shutil.which("npu-smi") is None:
        missing.append("npu-smi")
    else:
        device_missing = _device_requirement_missing(meta)
        if device_missing:
            missing.append(device_missing)
    if not _module_available("pytest"):
        missing.append("pytest")
    return {
        "is_ascend_ready": not missing,
        "missing": missing,
        "python": sys.executable,
    }


def _status_from_pytest(returncode: int, parsed: dict[str, Any]) -> str:
    counts = parsed["counts"]
    if returncode != 0 or counts["failed"] or counts["errors"]:
        return "failed"
    if parsed["clean_pass"]:
        return "passed"
    if counts["skipped"]:
        return "passed_with_skips"
    if counts["passed"] == 0:
        return "no_tests_collected"
    return "coverage_gap"


def _run_command(command: list[str], cwd: Path, log_path: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    completed = subprocess.run(command, cwd=str(cwd), env=run_env, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    combined_output = completed.stdout + "\n" + completed.stderr
    log_path.write_text(
        "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
        encoding="utf-8",
    )
    parsed = parse_pytest_output(combined_output, returncode=completed.returncode)
    case_results = parse_pytest_verbose_cases(combined_output)
    return {
        "status": _status_from_pytest(completed.returncode, parsed),
        "returncode": completed.returncode,
        "pytest": parsed,
        "case_results": case_results,
        "log": str(log_path),
    }


def _run_raw_command(command: list[str], cwd: Path, log_path: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=str(cwd), capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND: " + " ".join(command) + "\n\nSTDOUT:\n" + completed.stdout + "\n\nSTDERR:\n" + completed.stderr,
        encoding="utf-8",
    )
    return {
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "log": str(log_path),
    }


def _read_profile_summary(result_dir: Path) -> dict[str, Any]:
    summary_path = result_dir / "profile_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _read_dtype_support(result_dir: Path) -> dict[str, Any]:
    support_path = result_dir / "dtype_support.json"
    if not support_path.exists():
        return {}
    return json.loads(support_path.read_text(encoding="utf-8"))


def _add_failure_analysis(
    run_summary: dict[str, Any],
    area: str,
    classification: str,
    reason: str,
    evidence: str | None = None,
) -> None:
    entry = {
        "area": area,
        "classification": classification,
        "reason": reason,
    }
    if evidence:
        entry["evidence"] = evidence
    run_summary.setdefault("case_failure_analysis", []).append(entry)


def build_direct_validation_commands(
    operator_name: str,
    run_pta_compare: bool,
    build_pta_custom: bool = False,
    opinfo_status: dict[str, Any] | None = None,
) -> list[str]:
    safe_name = sanitize_identifier(operator_name)
    # Current ms_ops release build refreshes in-place .so and has no PTA-specific compile flag.
    _ = build_pta_custom
    compile_command = "python compile.py --release"
    commands = [compile_command]
    opinfo_status = opinfo_status or {}
    if opinfo_status.get("registered") and opinfo_status.get("test_path") and opinfo_status.get("opinfo_name"):
        commands.append(
            "CUSTOM_OPS_MODE=release MS_DISABLE_KERNEL_BACKOFF=1 python -m pytest "
            f"{opinfo_status['test_path']} -k {opinfo_status['opinfo_name']} -v --tb=line --durations=0 -ra"
        )
    else:
        scaffold = opinfo_status.get("scaffold_command")
        commands.append(str(scaffold or "register operator in tests/ms_adapter/opinfo before opinfo pytest"))
    if run_pta_compare:
        commands.append(
            f"python -m pytest tests/acceptance/profiler/test_{safe_name}_profiler.py "
            "-v --tb=line --durations=0 -ra"
        )
    return commands


def build_compile_command(build_pta_custom: bool = False) -> list[str]:
    # Current ms_ops release build refreshes in-place .so and has no PTA-specific compile flag.
    _ = build_pta_custom
    return ["python", "compile.py", "--release"]


def _generated_test_analysis(generated: dict[str, Path]) -> dict[str, Any]:
    return {
        key: analyze_pytest_file(path)
        for key, path in generated.items()
        if key.endswith("_case") and path.exists()
    }


def _load_acceptance_artifacts(result_dir: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for key, filename in {
        "operator_contract": "operator_contract.json",
        "case_plan": "case_plan.json",
    }.items():
        path = result_dir / filename
        if path.exists():
            artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
    return artifacts


def _finalize_run(meta_path: Path, result_dir: Path, run_summary: dict[str, Any]) -> dict[str, Any]:
    summary_path = result_dir / "run_summary.json"
    write_json(summary_path, run_summary)
    report_paths = write_report(meta_path, summary_path, result_dir)
    run_summary["report_paths"] = {key: str(value) for key, value in report_paths.items()}
    write_json(summary_path, run_summary)
    return run_summary


def run_acceptance(
    operator_name: str,
    project_root: str | Path,
    overwrite: bool = True,
    run_pytest: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    meta = collect_operator_meta(root, operator_name)
    canonical_name = meta["operator_name"]
    result_dir = root / "tests" / "acceptance" / "results" / canonical_name
    result_dir.mkdir(parents=True, exist_ok=True)
    meta_path = result_dir / "operator_meta.json"
    write_json(meta_path, meta)
    pta_compare_enabled = bool(meta.get("has_pta"))
    build_pta_custom = pta_compare_enabled and bool(meta.get("pta_requires_custom_build"))
    operator_contract_path = result_dir / "operator_contract.json"
    case_plan_path = result_dir / "case_plan.json"
    generated = generate_acceptance_cases(
        meta_path,
        root / "tests" / "acceptance",
        overwrite=overwrite,
        operator_contract_path=operator_contract_path if operator_contract_path.exists() else None,
        case_plan_path=case_plan_path if case_plan_path.exists() else None,
    )
    scaffold_acceptance_artifacts(
        meta_path,
        result_dir,
        case_coverage_path=generated["case_coverage"],
        overwrite=False,
    )
    case_coverage = json.loads(generated["case_coverage"].read_text(encoding="utf-8"))
    opinfo_status = case_coverage.get("opinfo_framework", {})
    preflight = run_preflight(require_pta=pta_compare_enabled, meta=meta)
    direct_commands = build_direct_validation_commands(
        canonical_name,
        run_pta_compare=pta_compare_enabled,
        build_pta_custom=build_pta_custom,
        opinfo_status=opinfo_status,
    )
    static_audits = _collect_static_audits(root, meta)
    acceptance_artifacts = _load_acceptance_artifacts(result_dir)
    run_summary: dict[str, Any] = {
        "operator_name": canonical_name,
        "requested_operator_name": operator_name,
        "pta_compare_enabled": pta_compare_enabled,
        "preflight": preflight,
        "generated_files": {key: str(value) for key, value in generated.items()},
        "case_coverage": case_coverage,
        "opinfo_framework": opinfo_status,
        "static_test_analysis": _generated_test_analysis(generated),
        "direct_validation_commands": direct_commands,
        **acceptance_artifacts,
        **static_audits,
    }
    compare_skip_reason = "not_applicable"
    if not preflight["is_ascend_ready"]:
        run_summary.update(
            {
                "pytest": {"status": "not_run", "reason": "environment_blocked"},
                "profiler": {
                    "status": "not_run",
                    "reason": "environment_blocked" if pta_compare_enabled else compare_skip_reason,
                },
                "kernel_consistency": {
                    "status": "not_run",
                    "reason": "environment_blocked" if pta_compare_enabled else compare_skip_reason,
                },
                "pending_commands": direct_commands,
            }
        )
    elif not run_pytest:
        compare_reason = "direct_pytest_required" if pta_compare_enabled else compare_skip_reason
        run_summary.update(
            {
                "pytest": {"status": "not_run", "reason": "direct_pytest_required"},
                "profiler": {"status": "not_run", "reason": compare_reason},
                "kernel_consistency": {"status": "not_run", "reason": compare_reason},
                "pending_commands": direct_commands,
            }
        )
    else:
        if opinfo_status.get("status") != "ready":
            reason = opinfo_status.get("reason") or "ms_adapter opinfo framework is not ready for this operator"
            run_summary.update(
                {
                    "pytest": {"status": "not_run", "reason": "opinfo_framework_not_ready: " + str(reason)},
                    "profiler": {
                        "status": "not_run",
                        "reason": "opinfo_not_clean" if pta_compare_enabled else compare_skip_reason,
                    },
                    "kernel_consistency": {
                        "status": "not_run",
                        "reason": "opinfo_not_clean" if pta_compare_enabled else compare_skip_reason,
                    },
                    "pending_commands": direct_commands,
                }
            )
            _add_failure_analysis(
                run_summary,
                "opinfo",
                "case_design_error",
                "operator must be registered in tests/ms_adapter/opinfo before acceptance pytest can run",
                opinfo_status.get("scaffold_command"),
            )
            return _finalize_run(meta_path, result_dir, run_summary)
        run_summary["build"] = _run_raw_command(
            build_compile_command(build_pta_custom=build_pta_custom),
            cwd=root,
            log_path=result_dir / "build.log",
        )
        if run_summary["build"]["status"] != "passed":
            run_summary.update(
                {
                    "pytest": {"status": "not_run", "reason": "build_failed"},
                    "profiler": {
                        "status": "not_run",
                        "reason": "build_failed" if pta_compare_enabled else compare_skip_reason,
                    },
                    "kernel_consistency": {
                        "status": "not_run",
                        "reason": "build_failed" if pta_compare_enabled else compare_skip_reason,
                    },
                    "pending_commands": direct_commands,
                }
            )
            return _finalize_run(meta_path, result_dir, run_summary)
        opinfo_case = generated["opinfo_case"].relative_to(root)
        run_summary["pytest"] = _run_command(
            [
                sys.executable,
                "-m",
                "pytest",
                str(opinfo_case),
                "-k",
                str(opinfo_status["opinfo_name"]),
                "-v",
                "--tb=line",
                "--durations=0",
                "-ra",
            ],
            cwd=root,
            log_path=result_dir / "opinfo_pytest.log",
            env={"CUSTOM_OPS_MODE": "release", "MS_DISABLE_KERNEL_BACKOFF": "1"},
        )
        run_summary["backoff_validation"] = {
            "status": "passed" if run_summary["pytest"]["status"] == "passed" else "failed",
            "env": {"MS_DISABLE_KERNEL_BACKOFF": "1"},
            "reason": (
                "opinfo pytest passed with kernel backoff disabled"
                if run_summary["pytest"]["status"] == "passed"
                else "opinfo pytest did not pass with kernel backoff disabled"
            ),
        }
        if run_summary["pytest"]["status"] != "passed":
            _add_failure_analysis(
                run_summary,
                "opinfo",
                "needs_triage",
                "opinfo pytest is not clean; classify as case_design_error or operator_bug before final report",
                run_summary["pytest"].get("log"),
            )
            run_summary["profiler"] = {
                "status": "not_run",
                "reason": "opinfo_not_clean" if pta_compare_enabled else compare_skip_reason,
            }
            run_summary["kernel_consistency"] = {
                "status": "not_run",
                "reason": "opinfo_not_clean" if pta_compare_enabled else compare_skip_reason,
            }
            return _finalize_run(meta_path, result_dir, run_summary)
        legacy_tests = run_summary.get("legacy_test_audit", {}).get("matched_tests", [])
        if legacy_tests:
            run_summary["legacy_test_audit"].update(
                _run_command(
                    [sys.executable, "-m", "pytest", *legacy_tests, "-v", "--tb=line", "--durations=0", "-ra"],
                    cwd=root,
                    log_path=result_dir / "legacy_pytest.log",
                    env={"CUSTOM_OPS_MODE": "release"},
                )
            )
            run_summary["legacy_test_audit"]["matched_tests"] = legacy_tests
            if run_summary["legacy_test_audit"]["status"] != "passed":
                _add_failure_analysis(
                    run_summary,
                    "legacy_tests",
                    "needs_triage",
                    "matched legacy pytest is not clean; classify before final review",
                    run_summary["legacy_test_audit"].get("log"),
                )
        if not pta_compare_enabled:
            run_summary["profiler"] = {"status": "not_run", "reason": compare_skip_reason}
            run_summary["kernel_consistency"] = {"status": "not_run", "reason": compare_skip_reason}
            return _finalize_run(meta_path, result_dir, run_summary)
        profiler_case = generated["profiler_case"].relative_to(root)
        run_summary["profiler"] = _run_command(
            [sys.executable, "-m", "pytest", str(profiler_case), "-v", "--tb=line", "--durations=0", "-ra"],
            cwd=root,
            log_path=result_dir / "profiler_pytest.log",
        )
        dtype_support = _read_dtype_support(result_dir)
        if dtype_support:
            run_summary["dtype_support"] = dtype_support
        if run_summary["profiler"]["status"] == "passed":
            profile_summary = _read_profile_summary(result_dir)
            run_summary["profile_summary"] = profile_summary
            profile_status = profile_summary.get("status")
            run_summary["kernel_consistency"] = {
                "status": "passed" if profile_status == "passed" else "coverage_gap",
                "reason": profile_summary.get(
                    "reason",
                    "parsed MS/PTA kernel sequence is required before marking kernel consistency passed",
                ),
                "summary": str(result_dir / "profile_summary.json"),
            }
        else:
            _add_failure_analysis(
                run_summary,
                "profiler",
                "operator_bug",
                "profiler pytest failed after opinfo clean pass; preserve failure evidence for report",
                run_summary["profiler"].get("log"),
            )
            run_summary["kernel_consistency"] = {
                "status": "not_run",
                "reason": "profiler failed",
            }
    return _finalize_run(meta_path, result_dir, run_summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator_name")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--run-pytest", action="store_true", help="Run generated pytest cases after preflight passes")
    args = parser.parse_args(argv)
    try:
        summary = run_acceptance(
            args.operator_name,
            args.project_root,
            overwrite=not args.no_overwrite,
            run_pytest=args.run_pytest,
        )
    except Exception as exc:
        print(f"[custom-op-acceptance] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
