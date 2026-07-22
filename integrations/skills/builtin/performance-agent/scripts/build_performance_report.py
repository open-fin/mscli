#!/usr/bin/env python3
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from perf_common import read_json, write_json, write_text


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def relative_to_out(path: Path, out_root: Path) -> str:
    return str(path.resolve().relative_to(out_root.resolve()))


def copy_json_artifact(source: Optional[str], target: Path, fallback):
    if source:
        source_path = Path(source)
        if source_path.exists():
            try:
                payload = read_json(source_path)
            except Exception:
                payload = fallback
        else:
            payload = fallback
    else:
        payload = fallback
    write_json(target, payload)
    return payload


def copy_summary_artifacts(summary_refs: dict, summaries_root: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for name, source in summary_refs.items():
        if not source:
            continue
        source_path = Path(source)
        if not source_path.exists():
            continue
        target = summaries_root / f"{name}.json"
        write_json(target, read_json(source_path))
        copied[name] = str(target)
    return copied


def build_env_payload() -> dict:
    git_commit = None
    # Walk upward from this script to find the git repo root, rather than
    # relying on a fixed number of parent hops that breaks when the file moves.
    candidate = Path(__file__).resolve().parent
    for _ in range(8):
        if (candidate / ".git").exists():
            try:
                git_commit = subprocess.check_output(
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    text=True,
                    timeout=5,
                ).strip()
            except Exception:
                pass
            break
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return {
        "mindspore_version": None,
        "cann_version": None,
        "driver_version": None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit,
    }


def build_inputs_payload(args, run_id: str) -> dict:
    return {
        "skill": "performance-agent",
        "run_id": run_id,
        "parameters": {
            "working_dir": str(Path(args.working_dir).resolve()) if args.working_dir else None,
            "user_problem": args.user_problem,
            "locate_json": args.locate_json,
            "profile_json": args.profile_json,
            "bottlenecks_json": args.bottlenecks_json,
            "validation_json": args.validation_json,
        },
    }


def map_verdict_status(
    validation: Optional[dict],
    primary_name: Optional[str],
    trace_root: Optional[str],
    profile_confidence: Optional[str],
    available_artifacts: Optional[dict] = None,
    data_quality: Optional[str] = None,
) -> tuple[str, str, str]:
    """Determine verdict status based on available evidence.

    Args:
        validation: Validation results if available
        primary_name: Name of primary bottleneck candidate
        trace_root: Path to profiler export root
        profile_confidence: Confidence level of profile
        available_artifacts: Dict of available artifact flags
        data_quality: Data quality level (excellent/good/fair/poor/critical)

    Returns:
        Tuple of (status, summary, next_action)
    """
    if validation:
        overall = validation.get("overall_result")
        if overall == "improved":
            return (
                "VALIDATED_IMPROVEMENT",
                "The selected optimization improved the target validation metrics.",
                "Consider checking whether a new secondary bottleneck now dominates.",
            )
        if overall in {"mixed", "unchanged", "regressed"}:
            return (
                "VALIDATION_PENDING",
                "Validation exists, but the selected optimization is not yet a clear win.",
                "Inspect the comparison and gather stronger or more targeted rerun evidence.",
            )

    # Check critical artifacts availability (these matter more than profiler_info.json alone)
    # Essential: step_summary and hotspot_summary are required for meaningful analysis
    # Advanced: trace_view, communication_summary, memory_summary enhance but aren't blocking
    essential_artifacts = {
        "step_summary": True,  # step_trace_time.csv is essential for step analysis
        "hotspot_summary": True,  # operator_details.csv is essential for hotspot analysis
    }
    advanced_artifacts = {
        "trace_view": True,  # trace_view.json for timeline analysis
        "communication_summary": True,  # communication analysis
        "memory_summary": True,  # memory analysis
        "input_summary": True,  # input pipeline analysis
    }

    has_essential = False
    has_advanced = False
    missing_essential = []
    missing_advanced = []

    if available_artifacts:
        for artifact_name, is_essential in essential_artifacts.items():
            if available_artifacts.get(artifact_name):
                has_essential = True
            else:
                missing_essential.append(artifact_name)
        for artifact_name, is_advanced in advanced_artifacts.items():
            if available_artifacts.get(artifact_name):
                has_advanced = True
            else:
                missing_advanced.append(artifact_name)

    # If we have essential artifacts, we can generate report even without profiler_info.json
    if has_essential or trace_root:
        if primary_name and primary_name != "inconclusive":
            return (
                "BOTTLENECK_IDENTIFIED",
                f"The dominant bottleneck candidate is {primary_name}.",
                "Apply one targeted optimization and compare only the metrics tied to that bottleneck.",
            )
        if profile_confidence in {"strong", "moderate"}:
            return (
                "PROFILE_RECOVERED",
                "Profiler outputs were recovered, and bottleneck analysis is complete.",
                "Apply the recommended optimizations and validate with a new profiler run.",
            )
        if has_essential:
            missing_parts = []
            if missing_essential:
                missing_parts.append(f"essential: {', '.join(missing_essential)}")
            if missing_advanced:
                missing_parts.append(f"advanced: {', '.join(missing_advanced)}")
            missing_str = "; ".join(missing_parts) if missing_parts else "profiler_info.json"
            return (
                "PARTIAL_PROFILE",
                "Analysis completed with available data. Some artifacts are missing but core analysis is valid.",
                f"Recommendation: Ensure profiler.stop() is called properly next run. Missing: {missing_str}.",
            )
        return (
            "TRACE_REQUIRED",
            "Profiler evidence is missing. The run context exists, but bottleneck classification is not yet trustworthy.",
            "Provide a profiler export root or collect the smallest high-signal trace files first.",
        )

    # No critical artifacts and no trace_root - truly insufficient data
    if data_quality == "critical":
        return (
            "INSUFFICIENT_DATA",
            "Critical data quality issues prevent reliable bottleneck analysis.",
            "Check profiler.stop() was called, ensure profiler collection completed normally, and recollect data.",
        )

    return (
        "TRACE_REQUIRED",
        "The current evidence is insufficient for a defensible bottleneck claim.",
        "Collect a profiler export root with step_trace_time.csv, trace_view.json, and operator_details.csv.",
    )


def build_verdict(
    locate: dict, profile: dict, bottlenecks: dict, validation: Optional[dict]
) -> dict:
    primary = bottlenecks.get("primary_candidate", {})
    status, summary, next_action = map_verdict_status(
        validation,
        primary.get("name"),
        profile.get("trace_root"),
        profile.get("confidence"),
        available_artifacts=profile.get("available_artifacts"),
        data_quality=profile.get("data_quality", {}).get("level"),
    )
    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "status": status,
        "summary": summary,
        "working_dir": profile.get("working_dir"),
        "trace_root": profile.get("trace_root"),
        "stack": profile.get("stack") or locate.get("stack"),
        "workload_type": profile.get("workload_type"),
        "metric_focus": profile.get("metric_focus"),
        "primary_symptom": profile.get("primary_symptom"),
        "evidence_level": profile.get("confidence"),
        "available_artifacts": profile.get("available_artifacts", {}),
        "dominant_bottleneck": primary,
        "ranked_bottlenecks": bottlenecks.get("ranked_candidates", []),
        "validation_result": validation,
        "recommended_action": (
            primary.get("optimization_hints", [next_action])[0]
            if primary and primary.get("name") != "inconclusive"
            else None
        ),
        "next_action": next_action,
        "sources": {
            "selected_trace_root": locate.get("selected_root"),
            "selected_files": locate.get("selected_files"),
            "summary_refs": profile.get("summary_refs", {}),
        },
    }


def build_shared_report(
    verdict: dict,
    run_id: str,
    out_root: Path,
    report_md: Path,
    verdict_json: Path,
    profile_json: Path,
    bottlenecks_json: Path,
    validation_json: Optional[Path],
    perf_lock_json: Path,
    extra_artifacts: list[Path],
) -> dict:
    timestamp = now_iso()
    status_map = {
        "VALIDATED_IMPROVEMENT": "success",
        "BOTTLENECK_IDENTIFIED": "success",
        "PARTIAL_PROFILE": "partial",
        "PROFILE_RECOVERED": "partial",
        "VALIDATION_PENDING": "partial",
        "INSUFFICIENT_DATA": "failed",
        "TRACE_REQUIRED": "failed",
    }
    shared_status = status_map.get(verdict["status"], "failed")
    report = {
        "schema_version": "1.0.0",
        "skill": "performance-agent",
        "run_id": run_id,
        "status": shared_status,
        "start_time": timestamp,
        "end_time": timestamp,
        "duration_sec": 0,
        "steps": [
            {
                "name": "performance-diagnosis",
                "status": shared_status,
                "message": verdict["summary"],
            }
        ],
        "logs": [],
        "artifacts": [
            relative_to_out(report_md, out_root),
            relative_to_out(verdict_json, out_root),
            relative_to_out(profile_json, out_root),
            relative_to_out(bottlenecks_json, out_root),
            relative_to_out(perf_lock_json, out_root),
        ],
        "env_ref": "meta/env.json",
        "inputs_ref": "meta/inputs.json",
    }
    if validation_json and validation_json.exists():
        report["artifacts"].append(relative_to_out(validation_json, out_root))
    for artifact in extra_artifacts:
        if artifact.exists():
            report["artifacts"].append(relative_to_out(artifact, out_root))
    if shared_status == "failed":
        report["error"] = {
            "code": "E_VERIFY",
            "message": verdict["summary"],
            "details": {"status": verdict["status"]},
        }
    return report


def render_suggestions_md(suggestions: list[dict]) -> str:
    """Render optimization suggestions as markdown."""
    if not suggestions:
        return ""

    lines = ["## Optimization Suggestions", ""]

    # Group by priority (case-insensitive)
    def get_priority(s: dict) -> str:
        return (s.get("priority") or "").lower()

    high = [s for s in suggestions if get_priority(s) == "high"]
    medium = [s for s in suggestions if get_priority(s) == "medium"]
    low = [s for s in suggestions if get_priority(s) == "low"]

    for label, group in [("HIGH Priority", high), ("MEDIUM Priority", medium), ("LOW Priority", low)]:
        if not group:
            continue
        lines.append(f"### {label}")
        lines.append("")
        for s in group:
            lines.append(f"#### {s.get('id', 'N/A')}: {s['title']}")
            lines.append("")
            lines.append(f"- **Expected Benefit**: {s.get('expected_benefit', 'N/A')}")
            lines.append(f"- **Trigger**: {s.get('trigger_metric', 'N/A')}")
            lines.append("")
            if s.get("actions"):
                lines.append("**Actions**:")
                lines.append("")
                for i, action in enumerate(s["actions"], 1):
                    lines.append(f"{i}. {action}")
                lines.append("")
            if s.get("code_examples"):
                lines.append("**Code Examples**:")
                lines.append("")
                for framework, code in s["code_examples"].items():
                    lines.append(f"*{framework}*:")
                    lines.append("```python")
                    lines.append(code)
                    lines.append("```")
                    lines.append("")
            if s.get("config_examples"):
                lines.append("**Config Examples**:")
                lines.append("")
                for framework, config in s["config_examples"].items():
                    lines.append(f"*{framework}*:")
                    lines.append("```")
                    lines.append(config)
                    lines.append("```")
                    lines.append("")
            lines.append(f"**Validation**: compare {', '.join(s.get('validation_metrics', []))}")
            lines.append("")

    return "\n".join(lines)


def render_markdown(verdict: dict, suggestions: Optional[list[dict]] = None) -> str:
    lines = [
        "# Performance Report",
        "",
        "## Summary",
        "",
        f"- status: `{verdict['status']}`",
        f"- summary: {verdict['summary']}",
        f"- primary_symptom: `{verdict.get('primary_symptom')}`",
        f"- workload_type: `{verdict.get('workload_type')}`",
        f"- metric_focus: `{verdict.get('metric_focus')}`",
        "",
        "## What",
        "",
        f"- dominant_bottleneck: `{verdict['dominant_bottleneck'].get('name') if verdict.get('dominant_bottleneck') else None}`",
        f"- evidence_level: `{verdict.get('evidence_level')}`",
        "",
        "## How",
        "",
    ]
    for item in verdict.get("ranked_bottlenecks", [])[:3]:
        lines.append(f"- `{item['name']}` (confidence={item['confidence']})")
        for evidence in item.get("evidence", []):
            lines.append(f"  evidence: {evidence}")
    lines.extend(["", "## Verify", ""])
    validation = verdict.get("validation_result")
    if validation:
        lines.append(f"- overall_result: `{validation.get('overall_result')}`")
        for metric in validation.get("metrics_compared", []):
            lines.append(f"- {metric['metric']}: {metric['before']} -> {metric['after']} ({metric['outcome']})")
    else:
        lines.append("- validation is pending")

    # Insert optimization suggestions between Verify and Artifacts
    if suggestions:
        lines.append("")
        lines.extend(render_suggestions_md(suggestions).split("\n"))

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- report.json",
            "- report.md",
            "- meta/performance-verdict.json",
            "- meta/performance-profile.json",
            "- meta/bottlenecks.json",
            "- meta/locator.json",
            "- meta/summaries/*.json",
            "- artifacts/perf.lock.json",
            "",
            "## Environment",
            "",
            "- see `meta/env.json`",
            "",
            "## Logs",
            "",
            "- no dedicated logs were generated by the deterministic pipeline",
            "",
            "## Next",
            "",
            f"- {verdict['next_action']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a shared performance report and performance verdict")
    parser.add_argument("--profile-json", required=True, help="performance profile JSON path")
    parser.add_argument("--bottlenecks-json", required=True, help="bottleneck classification JSON path")
    parser.add_argument("--output-json", required=True, help="path to output shared report JSON")
    parser.add_argument("--output-md", required=True, help="path to output report markdown")
    parser.add_argument("--output-verdict-json", help="optional path for performance verdict JSON")
    parser.add_argument("--locate-json", help="optional locator JSON path")
    parser.add_argument("--validation-json", help="optional validation comparison JSON path")
    parser.add_argument("--suggestions-json", help="optional optimization suggestions JSON path")
    parser.add_argument("--working-dir", default=".", help="workspace root")
    parser.add_argument("--user-problem", default="", help="user problem summary")
    args = parser.parse_args()

    profile = read_json(Path(args.profile_json))
    bottlenecks = read_json(Path(args.bottlenecks_json))
    locate = read_json(Path(args.locate_json)) if args.locate_json else {"selected_root": profile.get("trace_root")}
    validation = read_json(Path(args.validation_json)) if args.validation_json else None
    suggestions = read_json(Path(args.suggestions_json)) if args.suggestions_json else None

    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    out_root = output_json.parent.resolve()
    verdict_json = Path(args.output_verdict_json).resolve() if args.output_verdict_json else out_root / "meta" / "performance-verdict.json"
    profile_copy = out_root / "meta" / "performance-profile.json"
    bottlenecks_copy = out_root / "meta" / "bottlenecks.json"
    validation_copy = out_root / "meta" / "validation-comparison.json" if validation else None
    locator_copy = out_root / "meta" / "locator.json"
    summaries_root = out_root / "meta" / "summaries"
    perf_lock_json = out_root / "artifacts" / "perf.lock.json"
    env_json = out_root / "meta" / "env.json"
    inputs_json = out_root / "meta" / "inputs.json"
    run_id = f"performance-agent-{uuid4().hex[:12]}"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    verdict_json.parent.mkdir(parents=True, exist_ok=True)

    copied_profile = copy_json_artifact(args.profile_json, profile_copy, profile)
    copied_bottlenecks = copy_json_artifact(args.bottlenecks_json, bottlenecks_copy, bottlenecks)
    copied_locator = copy_json_artifact(args.locate_json, locator_copy, locate)
    copied_validation = None
    if validation_copy is not None:
        copied_validation = copy_json_artifact(args.validation_json, validation_copy, validation)
    copied_summaries = copy_summary_artifacts(copied_profile.get("summary_refs", {}), summaries_root)

    verdict = build_verdict(copied_locator, copied_profile, copied_bottlenecks, copied_validation)
    verdict["sources"]["locator_ref"] = str(locator_copy)
    verdict["sources"]["summary_artifacts"] = copied_summaries
    if suggestions:
        verdict["optimization_suggestions"] = suggestions.get("suggestions", [])
        verdict["suggestion_summary"] = suggestions.get("suggestion_summary", {})

    extra_artifacts: list[Path] = [locator_copy]
    extra_artifacts.extend(Path(path) for path in copied_summaries.values())
    shared_report = build_shared_report(
        verdict,
        run_id,
        out_root,
        output_md,
        verdict_json,
        profile_copy,
        bottlenecks_copy,
        validation_copy,
        perf_lock_json,
        extra_artifacts,
    )

    write_json(verdict_json, verdict)
    write_json(env_json, build_env_payload())
    write_json(inputs_json, build_inputs_payload(args, run_id))
    write_json(
        perf_lock_json,
        {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "run_id": run_id,
            "trace_root": verdict.get("trace_root"),
            "dominant_bottleneck": verdict.get("dominant_bottleneck"),
            "ranked_bottlenecks": verdict.get("ranked_bottlenecks"),
            "validation_result": verdict.get("validation_result"),
            "sources": verdict.get("sources"),
        },
    )
    write_json(output_json, shared_report)

    # Render markdown with suggestions passed directly to render function
    suggestions_list = verdict.get("optimization_suggestions", [])
    final_md = render_markdown(verdict, suggestions=suggestions_list)

    write_text(output_md, final_md)
    print(json.dumps({"status": verdict["status"], "run_id": run_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
