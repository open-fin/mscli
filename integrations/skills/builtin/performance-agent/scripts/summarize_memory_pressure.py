#!/usr/bin/env python3
"""Summarize memory pressure from Ascend profiler CSV exports.

Extends basic peak/operator memory analysis with:
- Memory leak detection via linear regression on time-series data
- Memory fragmentation estimation via coefficient of variation
- OOM risk assessment via multi-factor scoring
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

from perf_common import HARDWARE_SPECS, load_csv_rows, normalize_key, parse_number, write_json


NAME_KEYS = {"op_name", "operator_name", "operator", "name", "kernel_name", "module_name"}
MEMORY_KEYS = {
    "memory_mb",
    "peak_memory_mb",
    "peak_mem_mb",
    "allocated_mb",
    "reserved_mb",
    "memory",
    "peak_memory",
    "peak_mem",
}


def detect_name_and_memory_fields(
    rows: list[dict[str, str]]
) -> tuple[Optional[str], Optional[str]]:
    if not rows:
        return None, None
    normalized = {field: normalize_key(field) for field in rows[0].keys()}
    name_field = next((field for field, key in normalized.items() if key in NAME_KEYS), None)
    memory_field = next((field for field, key in normalized.items() if key in MEMORY_KEYS), None)
    if memory_field is None:
        memory_field = next((field for field, key in normalized.items() if "mem" in key or "memory" in key), None)
    return name_field, memory_field


def summarize_operator_memory(rows: list[dict[str, str]]) -> list[dict]:
    name_field, memory_field = detect_name_and_memory_fields(rows)
    if not name_field or not memory_field:
        return []

    ranked = []
    for row in rows:
        name = str(row.get(name_field) or "").strip()
        value = parse_number(row.get(memory_field))
        if not name or value is None:
            continue
        ranked.append({"name": name, "memory_mb": value})
    ranked.sort(key=lambda item: item["memory_mb"], reverse=True)
    total = sum(item["memory_mb"] for item in ranked) or 1.0
    return [
        {
            "name": item["name"],
            "memory_mb": round(item["memory_mb"], 3),
            "share_percent": round(item["memory_mb"] / total * 100, 2),
        }
        for item in ranked[:5]
    ]


def summarize_peak_memory(
    rows: list[dict[str, str]]
) -> tuple[Optional[float], Optional[str]]:
    if not rows:
        return None, None

    best_field = None
    peak_value = None
    for field in rows[0].keys():
        key = normalize_key(field)
        # Also check for allocated/reserved/active which don't contain "mem" in the key
        if not any(k in key for k in ("mem", "memory", "allocated", "reserved", "active")):
            continue
        for row in rows:
            value = parse_number(row.get(field))
            if value is None:
                continue
            if peak_value is None or value > peak_value:
                peak_value = value
                best_field = field
    return peak_value, best_field


def detect_memory_leak(
    rows: list[dict[str, str]], r_squared_threshold: float = 0.8
) -> Optional[dict]:
    """Detect memory leak via linear regression on time-series memory data.

    Looks for a column with step/index data and a memory column, then fits
    a linear model. If R-squared exceeds the threshold and the slope is
    positive, a leak is flagged.

    Returns leak analysis dict or None if data is insufficient.
    """
    if len(rows) < 10:
        return None

    # Find memory column (prefer allocated/reserved over peak)
    memory_field = None
    for field in rows[0].keys():
        key = normalize_key(field)
        if key in ("allocated_mb", "reserved_mb", "memory_mb"):
            memory_field = field
            break
    if memory_field is None:
        for field in rows[0].keys():
            key = normalize_key(field)
            if "alloc" in key or "used" in key or "memory" in key:
                memory_field = field
                break
    if memory_field is None:
        return None

    # Extract time-series values using row index as time proxy
    values: list[float] = []
    for row in rows:
        val = parse_number(row.get(memory_field))
        if val is not None:
            values.append(val)

    if len(values) < 10:
        return None

    # Linear regression: y = slope * x + intercept
    n = len(values)
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    ss_xy = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))
    ss_yy = sum((values[i] - y_mean) ** 2 for i in range(n))

    if ss_xx == 0 or ss_yy == 0:
        return None

    slope = ss_xy / ss_xx
    r_squared = (ss_xy ** 2) / (ss_xx * ss_yy)

    leak_detected = r_squared > r_squared_threshold and slope > 0

    return {
        "leak_detected": leak_detected,
        "slope_mb_per_step": round(slope, 6),
        "r_squared": round(r_squared, 4),
        "confidence": "high" if r_squared > 0.9 else "moderate" if r_squared > 0.8 else "low",
        "field_analyzed": memory_field,
        "samples": n,
        "total_growth_mb": round(slope * n, 3),
    }


def calculate_fragmentation(
    rows: list[dict[str, str]]
) -> Optional[dict]:
    """Estimate memory fragmentation via coefficient of variation of free memory.

    When free memory varies wildly across steps, it suggests fragmentation
    (interleaved allocation/deallocation patterns).

    Returns fragmentation analysis dict or None.
    """
    if len(rows) < 5:
        return None

    # Find free/available memory column
    free_field = None
    for field in rows[0].keys():
        key = normalize_key(field)
        if key in ("free_mb", "available_mb", "free_memory_mb"):
            free_field = field
            break
    if free_field is None:
        for field in rows[0].keys():
            key = normalize_key(field)
            if "free" in key or "available" in key:
                free_field = field
                break
    if free_field is None:
        return None

    free_values: list[float] = []
    for row in rows:
        val = parse_number(row.get(free_field))
        if val is not None and val >= 0:
            free_values.append(val)

    if len(free_values) < 5:
        return None

    mean_free = sum(free_values) / len(free_values)
    if mean_free <= 0:
        return None

    variance = sum((v - mean_free) ** 2 for v in free_values) / len(free_values)
    std_free = math.sqrt(variance)
    cv = std_free / mean_free

    # CV thresholds for fragmentation severity
    if cv < 0.05:
        level = "low"
    elif cv < 0.15:
        level = "moderate"
    else:
        level = "high"

    return {
        "fragmentation_level": level,
        "free_memory_cv": round(cv, 4),
        "mean_free_mb": round(mean_free, 3),
        "std_free_mb": round(std_free, 3),
        "field_analyzed": free_field,
        "samples": len(free_values),
    }


def assess_oom_risk(
    peak_memory_mb: Optional[float],
    hbm_capacity_gb: Optional[float],
    leak_analysis: Optional[dict],
    fragmentation: Optional[dict],
) -> Optional[dict]:
    """Assess OOM risk using a multi-factor scoring model.

    Factors: memory utilization, leak trend, fragmentation level.
    Returns risk assessment dict or None if insufficient data.
    """
    if peak_memory_mb is None:
        return None

    score = 0.0
    factors: list[dict] = []

    # Factor 1: Memory utilization
    if hbm_capacity_gb and hbm_capacity_gb > 0:
        capacity_mb = hbm_capacity_gb * 1024
        utilization = peak_memory_mb / capacity_mb
        if utilization > 0.95:
            score += 40
        elif utilization > 0.85:
            score += 25
        elif utilization > 0.70:
            score += 10
        factors.append({
            "factor": "utilization",
            "value": round(utilization, 4),
            "description": f"Peak memory utilization: {utilization:.1%}",
        })

    # Factor 2: Memory leak
    if leak_analysis and leak_analysis.get("leak_detected"):
        leak_score = 30 if leak_analysis.get("confidence") == "high" else 15
        score += leak_score
        factors.append({
            "factor": "memory_leak",
            "value": leak_analysis["slope_mb_per_step"],
            "description": f"Leak detected: {leak_analysis['slope_mb_per_step']:.4f} MB/step",
        })

    # Factor 3: Fragmentation
    if fragmentation and fragmentation.get("fragmentation_level") != "low":
        frag_score = 20 if fragmentation["fragmentation_level"] == "high" else 10
        score += frag_score
        factors.append({
            "factor": "fragmentation",
            "value": fragmentation["free_memory_cv"],
            "description": f"Fragmentation CV: {fragmentation['free_memory_cv']:.4f}",
        })

    if not factors:
        return None

    # Classify risk level
    if score >= 60:
        risk_level = "critical"
    elif score >= 35:
        risk_level = "high"
    elif score >= 15:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return {
        "oom_risk_level": risk_level,
        "oom_risk_score": round(score, 1),
        "factors": factors,
    }


def _oom_recommendations(oom_risk: Optional[dict]) -> list[str]:
    """Generate recommendations based on OOM risk assessment."""
    if not oom_risk:
        return []

    recommendations: list[str] = []
    for factor in oom_risk.get("factors", []):
        if factor["factor"] == "utilization" and factor["value"] > 0.85:
            recommendations.append(
                "Memory utilization is high — consider reducing batch_size or enabling gradient checkpointing"
            )
        if factor["factor"] == "memory_leak":
            recommendations.append(
                "Memory leak detected — investigate tensor lifecycle and ensure proper deallocation"
            )
        if factor["factor"] == "fragmentation":
            recommendations.append(
                "Memory fragmentation detected — consider setting max_split_size_mb and enabling expandable_segments"
            )
    return recommendations


def default_paths(
    trace_root: Path,
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    base = trace_root / "ASCEND_PROFILER_OUTPUT"
    files = (
        base / "memory_record.csv",
        base / "operator_memory.csv",
        base / "npu_module_mem.csv",
    )
    return tuple(path if path.exists() else None for path in files)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize memory pressure from Ascend profiler CSV exports")
    parser.add_argument("--trace-root", help="profiler export root")
    parser.add_argument("--memory-record-csv", help="explicit memory_record.csv path")
    parser.add_argument("--operator-memory-csv", help="explicit operator_memory.csv path")
    parser.add_argument("--module-memory-csv", help="explicit npu_module_mem.csv path")
    parser.add_argument("--hardware", help="hardware model key (e.g., ascend_910b2) for HBM capacity lookup")
    parser.add_argument("--output-json", required=True, help="path to write the memory summary JSON")
    args = parser.parse_args()

    memory_record = Path(args.memory_record_csv).resolve() if args.memory_record_csv else None
    operator_memory = Path(args.operator_memory_csv).resolve() if args.operator_memory_csv else None
    module_memory = Path(args.module_memory_csv).resolve() if args.module_memory_csv else None
    if args.trace_root:
        inferred_memory_record, inferred_operator_memory, inferred_module_memory = default_paths(Path(args.trace_root).resolve())
        memory_record = memory_record or inferred_memory_record
        operator_memory = operator_memory or inferred_operator_memory
        module_memory = module_memory or inferred_module_memory

    if not any(path and path.exists() for path in (memory_record, operator_memory, module_memory)):
        print("No memory profiler files were found. Provide explicit memory CSV paths or a trace root.", file=sys.stderr)
        raise SystemExit(1)

    record_rows = load_csv_rows(memory_record) if memory_record and memory_record.exists() else []
    operator_rows = load_csv_rows(operator_memory) if operator_memory and operator_memory.exists() else []
    module_rows = load_csv_rows(module_memory) if module_memory and module_memory.exists() else []

    peak_memory_mb, peak_source_field = summarize_peak_memory(record_rows or module_rows)
    top_operators = summarize_operator_memory(operator_rows)
    top_modules = summarize_operator_memory(module_rows)

    # Enhanced analysis
    leak_analysis = detect_memory_leak(record_rows)
    fragmentation = calculate_fragmentation(record_rows)
    hbm_capacity_gb = None
    if args.hardware:
        spec = HARDWARE_SPECS.get(args.hardware)
        if spec:
            hbm_capacity_gb = spec.get("hbm_capacity_gb")
    oom_risk = assess_oom_risk(peak_memory_mb, hbm_capacity_gb, leak_analysis, fragmentation)

    pressure = "low"
    if oom_risk and oom_risk["oom_risk_level"] in ("critical", "high"):
        pressure = "high"
    elif top_operators and top_operators[0]["share_percent"] >= 35:
        pressure = "high"
    elif peak_memory_mb is not None:
        pressure = "moderate"

    report = {
        "source_files": {
            "memory_record_csv": str(memory_record) if memory_record and memory_record.exists() else None,
            "operator_memory_csv": str(operator_memory) if operator_memory and operator_memory.exists() else None,
            "module_memory_csv": str(module_memory) if module_memory and module_memory.exists() else None,
        },
        "peak_memory_mb": round(peak_memory_mb, 3) if peak_memory_mb is not None else None,
        "peak_source_field": peak_source_field,
        "memory_pressure": pressure,
        "top_operators": top_operators,
        "top_modules": top_modules,
        "memory_leak": leak_analysis,
        "fragmentation": fragmentation,
        "oom_risk": oom_risk,
        "likely_domains": ["memory"] if peak_memory_mb is not None else [],
        "next_action": (
            "Validate peak memory, memory-heavy stage, and batch-size headroom after the first memory-focused change."
            if peak_memory_mb is not None
            else "Collect stronger memory evidence before choosing a memory optimization."
        ),
    }

    oom_recs = _oom_recommendations(oom_risk)
    if oom_recs:
        report["oom_recommendations"] = oom_recs

    write_json(Path(args.output_json), report)
    print(json.dumps({
        "peak_memory_mb": report["peak_memory_mb"],
        "pressure": report["memory_pressure"],
        "oom_risk": oom_risk["oom_risk_level"] if oom_risk else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
