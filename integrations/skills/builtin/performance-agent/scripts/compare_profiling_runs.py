#!/usr/bin/env python3
"""Compare two profiling analysis runs across multiple dimensions.

Extends beyond basic metric comparison to analyze step breakdown changes,
communication pattern shifts, operator hotspot movements, memory pressure
changes, MFU improvements, and cluster behavior differences.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from perf_common import load_optional_json, parse_number, write_json


def _safe_float(value, default=None) -> Optional[float]:
    """Safely convert to float, returning default if unparseable."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _pct_change(before: float, after: float) -> Optional[float]:
    """Compute percent change, returns None if before is zero."""
    if before == 0:
        return None
    return round((after - before) / abs(before) * 100, 2)


def _classify_change(before: float, after: float, lower_is_better: bool = True) -> str:
    """Classify a change as improved/regressed/unchanged."""
    threshold = 5.0  # 5% minimum to be considered a change
    pct = _pct_change(before, after)
    if pct is None:
        return "unchanged"
    if abs(pct) < threshold:
        return "unchanged"
    if lower_is_better:
        return "improved" if after < before else "regressed"
    return "improved" if after > before else "regressed"


def _load_artifact(directory: Path, *names: str) -> Optional[dict]:
    """Try loading a JSON artifact from directory with multiple name candidates."""
    for name in names:
        path = directory / name
        if path.exists():
            return load_optional_json(str(path))
    return None


def compare_step_breakdown(
    baseline: Optional[dict], comparison: Optional[dict]
) -> dict:
    """Compare step breakdown between two runs."""
    if not baseline or not comparison:
        return {"status": "unavailable", "details": []}

    details = []
    b_stages = baseline.get("stage_totals_ms", {})
    c_stages = comparison.get("stage_totals_ms", {})

    for stage in set(list(b_stages.keys()) + list(c_stages.keys())):
        b_val = _safe_float(b_stages.get(stage))
        c_val = _safe_float(c_stages.get(stage))
        if b_val is None or c_val is None:
            continue
        lower_better = True  # All stage_totals_ms are absolute time in ms — lower is always better
        change = _classify_change(b_val, c_val, lower_is_better=lower_better)
        pct = _pct_change(b_val, c_val)
        if pct is not None and abs(pct) >= 5:
            details.append({
                "metric": f"{stage}_time_ms",
                "baseline": b_val,
                "comparison": c_val,
                "percent_change": pct,
                "status": change,
            })

    # Also compare average step time
    b_avg = _safe_float(baseline.get("average_step_time_ms"))
    c_avg = _safe_float(comparison.get("average_step_time_ms"))
    if b_avg is not None and c_avg is not None:
        change = _classify_change(b_avg, c_avg, lower_is_better=True)
        pct = _pct_change(b_avg, c_avg)
        details.append({
            "metric": "average_step_time_ms",
            "baseline": b_avg,
            "comparison": c_avg,
            "percent_change": pct,
            "status": change,
        })

    overall = "unchanged"
    statuses = {d["status"] for d in details}
    if "improved" in statuses and "regressed" not in statuses:
        overall = "improved"
    elif "regressed" in statuses and "improved" not in statuses:
        overall = "regressed"
    elif statuses - {"unchanged"}:
        overall = "mixed"

    return {"status": overall, "details": details}


def compare_communication(
    baseline: Optional[dict], comparison: Optional[dict]
) -> dict:
    """Compare communication patterns between two runs."""
    if not baseline or not comparison:
        return {"status": "unavailable", "details": []}

    details = []
    b_total = _safe_float(baseline.get("total_time_ms"))
    c_total = _safe_float(comparison.get("total_time_ms"))
    if b_total is not None and c_total is not None:
        details.append({
            "metric": "total_comm_time_ms",
            "baseline": b_total,
            "comparison": c_total,
            "percent_change": _pct_change(b_total, c_total),
            "status": _classify_change(b_total, c_total, lower_is_better=True),
        })

    b_pressure = baseline.get("communication_pressure", "unknown")
    c_pressure = comparison.get("communication_pressure", "unknown")
    if b_pressure != c_pressure:
        details.append({
            "metric": "communication_pressure",
            "baseline": b_pressure,
            "comparison": c_pressure,
            "status": "changed",
        })

    b_count = _safe_float(baseline.get("collective_count"))
    c_count = _safe_float(comparison.get("collective_count"))
    if b_count is not None and c_count is not None:
        details.append({
            "metric": "collective_count",
            "baseline": b_count,
            "comparison": c_count,
            "percent_change": _pct_change(b_count, c_count),
            "status": _classify_change(b_count, c_count, lower_is_better=True),
        })

    overall = "unchanged"
    statuses = {d.get("status") for d in details}
    if "improved" in statuses and "regressed" not in statuses:
        overall = "improved"
    elif "regressed" in statuses and "improved" not in statuses:
        overall = "regressed"
    elif statuses - {"unchanged", "changed", "info"}:
        overall = "mixed"

    return {"status": overall, "details": details}


def compare_hotspots(
    baseline: Optional[dict], comparison: Optional[dict]
) -> dict:
    """Compare operator hotspot rankings between two runs."""
    if not baseline or not comparison:
        return {"status": "unavailable", "details": []}

    b_ops = {op.get("operator", ""): op for op in baseline.get("top_operators", [])}
    c_ops = {op.get("operator", ""): op for op in comparison.get("top_operators", [])}

    details = []
    all_ops = set(b_ops.keys()) | set(c_ops.keys())
    new_hotspots = all_ops - set(b_ops.keys())
    removed_hotspots = all_ops - set(c_ops.keys())

    if new_hotspots:
        details.append({"metric": "new_hotspots", "operators": sorted(new_hotspots), "status": "info"})
    if removed_hotspots:
        details.append({"metric": "removed_hotspots", "operators": sorted(removed_hotspots), "status": "info"})

    for op_name in set(b_ops.keys()) & set(c_ops.keys()):
        b_share = _safe_float(b_ops[op_name].get("share_percent"))
        c_share = _safe_float(c_ops[op_name].get("share_percent"))
        pct = _pct_change(b_share, c_share)
        if pct is not None and abs(pct) >= 10:
            details.append({
                "metric": f"{op_name}_share",
                "baseline": b_share,
                "comparison": c_share,
                "percent_change": pct,
                "status": _classify_change(b_share, c_share, lower_is_better=True),
            })

    overall = "unchanged"
    statuses = {d.get("status") for d in details}
    if "improved" in statuses and "regressed" not in statuses:
        overall = "improved"
    elif "regressed" in statuses:
        overall = "regressed"
    elif new_hotspots or removed_hotspots:
        overall = "changed"

    return {"status": overall, "details": details}


def compare_memory(
    baseline: Optional[dict], comparison: Optional[dict]
) -> dict:
    """Compare memory pressure between two runs."""
    if not baseline or not comparison:
        return {"status": "unavailable", "details": []}

    details = []
    b_peak = _safe_float(baseline.get("peak_memory_mb"))
    c_peak = _safe_float(comparison.get("peak_memory_mb"))
    if b_peak is not None and c_peak is not None:
        details.append({
            "metric": "peak_memory_mb",
            "baseline": b_peak,
            "comparison": c_peak,
            "percent_change": _pct_change(b_peak, c_peak),
            "status": _classify_change(b_peak, c_peak, lower_is_better=True),
        })

    b_pressure = baseline.get("memory_pressure", "unknown")
    c_pressure = comparison.get("memory_pressure", "unknown")
    if b_pressure != c_pressure:
        details.append({
            "metric": "memory_pressure",
            "baseline": b_pressure,
            "comparison": c_pressure,
            "status": "changed",
        })

    overall = "unchanged"
    statuses = {d.get("status") for d in details}
    if "regressed" in statuses:
        overall = "regressed"
    elif "improved" in statuses:
        overall = "improved"
    elif "changed" in statuses:
        overall = "changed"

    return {"status": overall, "details": details}


def compare_mfu(
    baseline: Optional[dict], comparison: Optional[dict]
) -> dict:
    """Compare MFU between two runs."""
    if not baseline or not comparison:
        return {"status": "unavailable", "details": []}

    b_mfu = _safe_float(baseline.get("estimated_mfu"))
    c_mfu = _safe_float(comparison.get("estimated_mfu"))
    if b_mfu is None or c_mfu is None:
        return {"status": "unavailable", "details": []}

    pct = _pct_change(b_mfu, c_mfu)
    return {
        "status": _classify_change(b_mfu, c_mfu, lower_is_better=False),
        "details": [{
            "metric": "estimated_mfu",
            "baseline": round(b_mfu, 4),
            "comparison": round(c_mfu, 4),
            "percent_change": pct,
            "status": _classify_change(b_mfu, c_mfu, lower_is_better=False),
        }],
    }


def compare_profiling_runs(
    baseline_dir: Path,
    comparison_dir: Path,
) -> dict:
    """Compare two profiling analysis output directories."""
    if not baseline_dir.exists() or not comparison_dir.exists():
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "comparison_available": False,
            "reason": "One or both directories not found",
        }

    # Load artifacts from both directories
    b_step = _load_artifact(baseline_dir, "step.json", "step_breakdown.json")
    c_step = _load_artifact(comparison_dir, "step.json", "step_breakdown.json")

    b_comm = _load_artifact(baseline_dir, "communication.json", "comm.json")
    c_comm = _load_artifact(comparison_dir, "communication.json", "comm.json")

    b_hotspot = _load_artifact(baseline_dir, "hotspot.json", "hotspot_summary.json")
    c_hotspot = _load_artifact(comparison_dir, "hotspot.json", "hotspot_summary.json")

    b_memory = _load_artifact(baseline_dir, "memory.json", "memory_pressure.json")
    c_memory = _load_artifact(comparison_dir, "memory.json", "memory_pressure.json")

    b_mfu = _load_artifact(baseline_dir, "mfu.json", "calculate_mfu.json")
    c_mfu = _load_artifact(comparison_dir, "mfu.json", "calculate_mfu.json")

    # Run comparisons
    step_result = compare_step_breakdown(b_step, c_step)
    comm_result = compare_communication(b_comm, c_comm)
    hotspot_result = compare_hotspots(b_hotspot, c_hotspot)
    memory_result = compare_memory(b_memory, c_memory)
    mfu_result = compare_mfu(b_mfu, c_mfu)

    dimensions = {
        "step_breakdown": step_result,
        "communication": comm_result,
        "hotspot": hotspot_result,
        "memory": memory_result,
        "mfu": mfu_result,
    }

    # Overall verdict
    statuses = []
    for dim_result in dimensions.values():
        if dim_result["status"] != "unavailable":
            statuses.append(dim_result["status"])

    improved = sum(1 for s in statuses if s == "improved")
    regressed = sum(1 for s in statuses if s == "regressed")

    if improved and not regressed:
        overall = "improved"
    elif regressed and not improved:
        overall = "regressed"
    elif improved and regressed:
        overall = "mixed"
    elif statuses:
        overall = "unchanged"
    else:
        overall = "insufficient_data"

    # Collect significant changes for summary
    significant = []
    for dim_name, dim_result in dimensions.items():
        for detail in dim_result.get("details", []):
            if detail.get("status") in ("improved", "regressed", "changed"):
                significant.append({
                    "dimension": dim_name,
                    **detail,
                })

    # Build summary
    summary_parts = []
    if step_result["status"] != "unavailable":
        for d in step_result.get("details", []):
            if d.get("percent_change") and abs(d["percent_change"]) >= 5:
                summary_parts.append(f"{d['metric']} changed by {d['percent_change']:+.1f}%")
    if mfu_result["status"] not in ("unavailable", "unchanged"):
        for d in mfu_result.get("details", []):
            summary_parts.append(f"MFU: {d['baseline']:.1%} -> {d['comparison']:.1%}")

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "comparison_available": True,
        "baseline_dir": str(baseline_dir),
        "comparison_dir": str(comparison_dir),
        "dimensions": dimensions,
        "overall_verdict": overall,
        "significant_changes": significant,
        "summary": "; ".join(summary_parts) if summary_parts else "No significant changes detected",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two profiling analysis runs across multiple dimensions"
    )
    parser.add_argument("--baseline-dir", required=True, help="Baseline analysis output directory")
    parser.add_argument("--comparison-dir", required=True, help="Comparison analysis output directory")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    result = compare_profiling_runs(Path(args.baseline_dir), Path(args.comparison_dir))

    write_json(Path(args.output_json), result)
    print(json.dumps({
        "comparison_available": result.get("comparison_available", False),
        "overall_verdict": result.get("overall_verdict", "unknown"),
        "significant_changes": len(result.get("significant_changes", [])),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
