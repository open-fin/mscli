#!/usr/bin/env python3
"""Generate compact intake-prestage algorithm artifacts.

This combined helper keeps intake-prestage file growth small by covering adjacent
intake, code-map, and verification-report generation in one place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


INTAKE_FIELDS = [
    "source_type",
    "paper_title",
    "paper_url",
    "code_url",
    "claimed_contribution",
    "target_task",
    "target_model_family",
    "feature_bucket",
    "likely_integration_surface",
    "dependency_complexity",
    "verification_risk",
    "migration_blockers",
    "qualification_basis",
    "source_status",
    "recommended_next_action",
]

CODE_MAP_FIELDS = [
    "reference_repo",
    "reference_scope",
    "reference_commit_or_tag",
    "target_feature",
    "source_modules",
    "source_configs",
    "source_entrypoints",
    "implementation_delta_summary",
    "target_repo_touchpoints",
    "patch_plan_summary",
    "uncertainties",
]

VERIFY_FIELDS = [
    "feature_name",
    "baseline_model_family",
    "torch_forward",
    "torch_backward",
    "torch_npu_forward",
    "torch_npu_backward",
    "mindspore_npu_forward",
    "mindspore_npu_backward",
    "shape_dtype_checks",
    "feature_toggle_regression",
    "accuracy_drift_status",
    "slot_expected_evidence",
    "allowed_status_values",
    "handoff_needed",
    "handoff_target",
    "notes",
]


def build_intake(args: argparse.Namespace) -> dict:
    return {
        "source_type": args.source_type,
        "paper_title": args.paper_title,
        "paper_url": args.paper_url,
        "code_url": args.code_url,
        "claimed_contribution": args.claimed_contribution,
        "target_task": args.target_task,
        "target_model_family": args.target_model_family,
        "feature_bucket": args.feature_bucket,
        "likely_integration_surface": args.likely_integration_surface,
        "dependency_complexity": args.dependency_complexity,
        "verification_risk": args.verification_risk,
        "migration_blockers": args.migration_blockers,
        "qualification_basis": args.qualification_basis,
        "source_status": args.source_status,
        "recommended_next_action": args.recommended_next_action,
    }


def build_code_map(args: argparse.Namespace) -> dict:
    return {
        "reference_repo": args.reference_repo,
        "reference_scope": args.reference_scope,
        "reference_commit_or_tag": args.reference_commit_or_tag,
        "target_feature": args.target_feature,
        "source_modules": args.source_modules,
        "source_configs": args.source_configs,
        "source_entrypoints": args.source_entrypoints,
        "implementation_delta_summary": args.implementation_delta_summary,
        "target_repo_touchpoints": args.target_repo_touchpoints,
        "patch_plan_summary": args.patch_plan_summary,
        "uncertainties": args.uncertainties,
    }


def build_verify(args: argparse.Namespace) -> dict:
    return {
        "feature_name": args.feature_name,
        "baseline_model_family": args.baseline_model_family,
        "torch_forward": args.torch_forward,
        "torch_backward": args.torch_backward,
        "torch_npu_forward": args.torch_npu_forward,
        "torch_npu_backward": args.torch_npu_backward,
        "mindspore_npu_forward": args.mindspore_npu_forward,
        "mindspore_npu_backward": args.mindspore_npu_backward,
        "shape_dtype_checks": args.shape_dtype_checks,
        "feature_toggle_regression": args.feature_toggle_regression,
        "accuracy_drift_status": args.accuracy_drift_status,
        "slot_expected_evidence": args.slot_expected_evidence,
        "allowed_status_values": args.allowed_status_values,
        "handoff_needed": args.handoff_needed,
        "handoff_target": args.handoff_target,
        "notes": args.notes,
    }


def add_common_output_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", help="Write JSON to this path instead of stdout")


def write_or_print(payload: dict, output: str | None) -> int:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate compact intake-prestage algorithm artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    intake = subparsers.add_parser("intake", help="Generate an intake artifact")
    intake.add_argument("--source-type", default="mixed")
    intake.add_argument("--paper-title", default="")
    intake.add_argument("--paper-url", default="")
    intake.add_argument("--code-url", default="")
    intake.add_argument("--claimed-contribution", default="")
    intake.add_argument("--target-task", default="")
    intake.add_argument("--target-model-family", default="")
    intake.add_argument("--feature-bucket", default="other")
    intake.add_argument("--likely-integration-surface", default="")
    intake.add_argument("--dependency-complexity", default="medium")
    intake.add_argument("--verification-risk", default="medium")
    intake.add_argument("--migration-blockers", default="")
    intake.add_argument("--qualification-basis", default="")
    intake.add_argument("--source-status", default="provisional")
    intake.add_argument("--recommended-next-action", default="watchlist")
    add_common_output_flags(intake)

    code_map = subparsers.add_parser("code-map", help="Generate a code-map artifact")
    code_map.add_argument("--reference-repo", default="")
    code_map.add_argument("--reference-scope", default="")
    code_map.add_argument("--reference-commit-or-tag", default="")
    code_map.add_argument("--target-feature", default="")
    code_map.add_argument("--source-modules", nargs="*", default=[])
    code_map.add_argument("--source-configs", nargs="*", default=[])
    code_map.add_argument("--source-entrypoints", nargs="*", default=[])
    code_map.add_argument("--implementation-delta-summary", default="")
    code_map.add_argument("--target-repo-touchpoints", nargs="*", default=[])
    code_map.add_argument("--patch-plan-summary", default="")
    code_map.add_argument("--uncertainties", default="")
    add_common_output_flags(code_map)

    verify = subparsers.add_parser("verify-report", help="Generate a verification report artifact")
    verify.add_argument("--feature-name", default="")
    verify.add_argument("--baseline-model-family", default="")
    verify.add_argument("--torch-forward", default="pending")
    verify.add_argument("--torch-backward", default="pending")
    verify.add_argument("--torch-npu-forward", default="pending")
    verify.add_argument("--torch-npu-backward", default="pending")
    verify.add_argument("--mindspore-npu-forward", default="pending")
    verify.add_argument("--mindspore-npu-backward", default="pending")
    verify.add_argument("--shape-dtype-checks", default="pending")
    verify.add_argument("--feature-toggle-regression", default="pending")
    verify.add_argument("--accuracy-drift-status", default="unknown")
    verify.add_argument("--slot-expected-evidence", default="")
    verify.add_argument("--allowed-status-values", nargs="*", default=["pass", "fail", "blocked", "not_run", "partial"])
    verify.add_argument("--handoff-needed", default="false")
    verify.add_argument("--handoff-target", default="")
    verify.add_argument("--notes", default="")
    add_common_output_flags(verify)

    args = parser.parse_args()
    if args.command == "intake":
        return write_or_print(build_intake(args), args.output)
    if args.command == "code-map":
        return write_or_print(build_code_map(args), args.output)
    if args.command == "verify-report":
        return write_or_print(build_verify(args), args.output)

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
