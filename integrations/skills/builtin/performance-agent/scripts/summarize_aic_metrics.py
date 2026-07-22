#!/usr/bin/env python3
"""Summarize AIC (AI Core) microarchitecture metrics from profiling data.

Analyzes Cube/Vector utilization, L2 cache hit rates, and pipeline stall rates
from msprof AIC PMU data. Only runs when AIC metrics data is available
(collected via `msprof op --aic-metrics`).
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Optional

from perf_common import normalize_key as _perf_normalize_key, write_json


# AIC PMU metric column names (may vary by msprof version)
CUBE_UTIL_KEYS = {"cube_util", "cube_utilization", "aiv_cube_ratio", "cube_ratio"}
VECTOR_UTIL_KEYS = {"vector_util", "vector_utilization", "aiv_vector_ratio", "vector_ratio"}
L2_HIT_RATE_KEYS = {"l2_hit_rate", "l2_cache_hit_rate", "l2_hit_ratio", "mic_l2_hit_rate"}
PIPE_UTIL_KEYS = {"pipe_util", "pipe_utilization", "pipeline_utilization", "aiv_pipe_ratio"}
STALL_RATE_KEYS = {"stall_rate", "pipe_stall_rate", "pipeline_stall_rate", "block_rate"}
OP_NAME_KEYS = {"op_name", "operator", "name", "task_name", "kernel_name"}
TIME_KEYS = {"duration_us", "duration", "total_time_us", "task_time_us", "duration_ns"}


def normalize_key(s: str) -> str:
    return _perf_normalize_key(s)


def find_first_match(normalized: str, key_set: set[str]) -> bool:
    return any(k in normalized for k in key_set)


def load_aic_data(profiler_path: Path) -> tuple[list[dict], Optional[Path]]:
    """Find and load AIC metrics data from profiler directory.

    Looks in:
    - PROF_{}/device_{}/data/aic_metrics_*.csv
    - PROF_{}/device_{}/data/aic/*.csv
    - Any CSV with Cube/Vector utilization columns
    """
    candidates = []

    # Known AIC metrics locations
    for pattern in [
        "**/aic_metrics_*.csv",
        "**/aic/*.csv",
        "**/aic_metrics*.csv",
        "**/op_summary_*.csv",  # op_summary may contain AIC columns
    ]:
        candidates.extend(profiler_path.rglob(pattern))

    if not candidates:
        return [], None

    # Try each candidate, looking for AIC PMU columns
    for csv_path in sorted(candidates):
        try:
            rows = []
            with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                # Check if any AIC metric columns exist
                normalized_headers = [normalize_key(h) for h in reader.fieldnames]
                has_aic = any(
                    find_first_match(nh, CUBE_UTIL_KEYS | L2_HIT_RATE_KEYS)
                    for nh in normalized_headers
                )
                if not has_aic:
                    continue
                for row in reader:
                    rows.append(row)
            if rows:
                return rows, csv_path
        except Exception:
            continue

    return [], None


def parse_metrics(rows: list[dict]) -> list[dict]:
    """Parse AIC metrics from CSV rows."""
    results = []
    for row in rows:
        op_name = None
        duration_us = None
        cube_util = None
        vector_util = None
        l2_hit_rate = None
        pipe_util = None
        stall_rate = None

        for header, value in row.items():
            key = normalize_key(header)
            # Parse value
            try:
                num = float(value.replace(",", "").strip().rstrip("%"))
            except (ValueError, AttributeError):
                num = None

            if find_first_match(key, OP_NAME_KEYS):
                op_name = str(value).strip()
            elif find_first_match(key, TIME_KEYS) and num is not None:
                if "ns" in key:
                    duration_us = num / 1000
                elif "ms" in key:
                    duration_us = num * 1000
                elif "us" in key:
                    duration_us = num
                else:
                    # Default assumption: value is in microseconds (most common for profiler output)
                    duration_us = num
            elif find_first_match(key, CUBE_UTIL_KEYS) and num is not None:
                cube_util = num / 100.0 if num > 1 else num
            elif find_first_match(key, VECTOR_UTIL_KEYS) and num is not None:
                vector_util = num / 100.0 if num > 1 else num
            elif find_first_match(key, L2_HIT_RATE_KEYS) and num is not None:
                l2_hit_rate = num / 100.0 if num > 1 else num
            elif find_first_match(key, PIPE_UTIL_KEYS) and num is not None:
                pipe_util = num / 100.0 if num > 1 else num
            elif find_first_match(key, STALL_RATE_KEYS) and num is not None:
                stall_rate = num / 100.0 if num > 1 else num

        if op_name and any(v is not None for v in [cube_util, l2_hit_rate, stall_rate]):
            results.append({
                "operator": op_name,
                "duration_us": duration_us,
                "cube_utilization": cube_util,
                "vector_utilization": vector_util,
                "l2_hit_rate": l2_hit_rate,
                "pipeline_utilization": pipe_util,
                "stall_rate": stall_rate,
            })

    return results


def classify_bottleneck(metrics: dict) -> tuple[str, str]:
    """Classify AIC bottleneck type and severity.

    Returns (bottleneck_type, severity).
    """
    cube = metrics.get("cube_utilization")
    l2 = metrics.get("l2_hit_rate")
    stall = metrics.get("stall_rate")

    # Normalize to 0-100 scale
    def norm100(v):
        return v if v and v > 1 else (v * 100 if v else None)

    cube = norm100(cube)
    l2 = norm100(l2)
    stall = norm100(stall)

    # Severity assessment
    if cube is not None and cube < 10:
        severity = "critical"
    elif stall is not None and stall > 50:
        severity = "critical"
    elif cube is not None and cube < 30:
        severity = "high"
    elif l2 is not None and l2 < 50:
        severity = "high"
    elif cube is not None and cube < 60:
        severity = "medium"
    else:
        severity = "low"

    # Type classification
    types = []
    if cube is not None and cube < 60:
        if cube < 30:
            types.append("compute")
        elif l2 is not None and l2 < 50:
            types.append("memory")
    if l2 is not None and l2 < 50:
        types.append("memory")
    if stall is not None and stall > 30:
        types.append("pipeline")

    if len(types) >= 2:
        return "mixed", severity
    if types:
        return types[0], severity
    return "none", severity


def summarize(parsed_metrics: list[dict]) -> dict:
    """Build summary from parsed AIC metrics."""
    if not parsed_metrics:
        return {
            "aic_data_available": False,
            "note": "No AIC PMU data found. Collect with 'msprof op --aic-metrics'.",
        }

    # Classify each operator
    classified = []
    bottleneck_counts = {"compute": 0, "memory": 0, "pipeline": 0, "mixed": 0, "none": 0}
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for m in parsed_metrics:
        bt, sev = classify_bottleneck(m)
        classified.append({**m, "bottleneck_type": bt, "severity": sev})
        bottleneck_counts[bt] = bottleneck_counts.get(bt, 0) + 1
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    # Sort by severity (critical first), then by lowest cube utilization
    sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    classified.sort(key=lambda x: (sev_order.get(x["severity"], 99), -(x.get("cube_utilization") or 100)))

    # Aggregate recommendations
    recommendations = []
    if severity_counts.get("critical", 0) > 0:
        recommendations.append(
            f"{severity_counts['critical']} operators have critical bottlenecks (Cube util <10% or stall >50%)"
        )
    if bottleneck_counts.get("memory", 0) > len(parsed_metrics) * 0.3:
        recommendations.append("Many operators are memory-bound: consider optimizing data layout and tiling")
    if bottleneck_counts.get("pipeline", 0) > len(parsed_metrics) * 0.2:
        recommendations.append("Pipeline stalls detected in multiple operators: consider instruction scheduling optimization")

    return {
        "aic_data_available": True,
        "analyzed_operators": len(parsed_metrics),
        "bottleneck_summary": bottleneck_counts,
        "severity_summary": severity_counts,
        "top_bottlenecks": [
            {
                "operator": m["operator"],
                "cube_utilization": m.get("cube_utilization"),
                "l2_hit_rate": m.get("l2_hit_rate"),
                "pipeline_stall_rate": m.get("stall_rate"),
                "bottleneck_type": m["bottleneck_type"],
                "severity": m["severity"],
            }
            for m in classified[:10]
        ],
        "recommended_actions": recommendations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize AIC microarchitecture metrics")
    parser.add_argument("--trace-root", required=True, help="Profiler root directory")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    trace_root = Path(args.trace_root).resolve()
    rows, source_file = load_aic_data(trace_root)

    if not rows:
        report = {
            "aic_data_available": False,
            "note": "No AIC PMU data found. Collect with 'msprof op --aic-metrics' for microarchitecture analysis.",
            "source_file": None,
        }
        write_json(Path(args.output_json), report)
        print(json.dumps({"aic_data_available": False}))
        return 0

    parsed = parse_metrics(rows)
    summary = summarize(parsed)
    summary["source_file"] = str(source_file)

    write_json(Path(args.output_json), summary)
    print(json.dumps({
        "aic_data_available": True,
        "analyzed_operators": summary["analyzed_operators"],
        "critical_count": summary.get("severity_summary", {}).get("critical", 0),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
