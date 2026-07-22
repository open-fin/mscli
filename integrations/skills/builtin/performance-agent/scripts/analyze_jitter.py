#!/usr/bin/env python3
"""Analyze jitter (performance variance) from step-level profiling data.

Detects multi-dimensional jitter:
- Step time jitter (overall variance)
- Compute jitter (kernel execution variance)
- Communication jitter (collective operation variance)
- Alignment jitter (cross-step phase alignment)
- Cross-rank alignment skew (cluster-level)
"""
import argparse
import json
import math
from pathlib import Path
from typing import Optional

from perf_common import find_rank_dirs, find_step_trace_csv, avg_step_time_from_csv, load_optional_json, write_json


def compute_cv(values: list[float]) -> Optional[float]:
    """Compute coefficient of variation (CV = std/mean)."""
    if not values or len(values) < 2:
        return None
    mean = sum(values) / len(values)
    if mean == 0:
        return None
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    return std / mean


def compute_percentiles(values: list[float]) -> dict:
    """Compute p50, p95, p99 and outlier count."""
    if not values:
        return {}
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / n)

    def percentile(p: float) -> float:
        idx = math.ceil(n * p / 100) - 1
        return sorted_vals[max(0, min(idx, n - 1))]

    outliers = sum(1 for v in values if abs(v - mean) > 2 * std)

    return {
        "p50_ms": round(percentile(50), 3),
        "p95_ms": round(percentile(95), 3),
        "p99_ms": round(percentile(99), 3),
        "mean_ms": round(mean, 3),
        "std_ms": round(std, 3),
        "outliers": outliers,
    }


def jitter_status(cv: Optional[float], threshold: float) -> str:
    """Classify jitter status from CV."""
    if cv is None:
        return "unknown"
    if cv <= threshold * 0.5:
        return "normal"
    if cv <= threshold:
        return "warning"
    return "critical"


def compute_zscore_outliers(values: list[float], threshold: float = 2.5) -> list[int]:
    """Identify outlier step indices using z-score.

    Returns list of 0-based indices where |z-score| > threshold.
    """
    if len(values) < 3:
        return []
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)
    if std <= 0:
        return []
    return [i for i, v in enumerate(values) if abs(v - mean) / std > threshold]


def _extract_per_step_times(step_json: dict) -> dict[str, list[float]]:
    """Extract per-step timing series from step summary data.

    Returns dict mapping stage names to lists of per-step times in ms.
    """
    per_step: dict[str, list[float]] = {}

    # Try "steps" array format first
    steps = step_json.get("steps", [])
    if isinstance(steps, list) and steps:
        for step in steps:
            if not isinstance(step, dict):
                continue
            for key, value in step.items():
                if isinstance(value, (int, float)):
                    per_step.setdefault(key, []).append(float(value))
        return per_step

    # Try "per_step_breakdown" format
    breakdown = step_json.get("per_step_breakdown", {})
    if isinstance(breakdown, dict):
        for stage, times in breakdown.items():
            if isinstance(times, list):
                per_step[stage] = [float(t) for t in times if isinstance(t, (int, float))]
        return per_step

    return per_step


def analyze_single_step(step_json: dict) -> dict:
    """Analyze multi-dimensional jitter from step summary (single-rank).

    Dimensions:
    - step_time: overall step time variance
    - compute: kernel execution variance
    - communication: collective operation variance
    - alignment: cross-step phase alignment variance
    """
    result: dict = {}

    step_totals = step_json.get("stage_totals_ms", {})
    steps_analyzed = step_json.get("steps_analyzed", 0)
    aggregate_cv = step_json.get("coefficient_of_variation")

    # Extract per-step time series if available
    per_step = _extract_per_step_times(step_json)

    # 1. Step time jitter
    if aggregate_cv is not None:
        result["step_time_jitter"] = {
            "cv": round(aggregate_cv, 4),
            "status": jitter_status(aggregate_cv, 0.15),
            "steps_analyzed": steps_analyzed,
        }
    elif per_step.get("step_total"):
        cv = compute_cv(per_step["step_total"])
        result["step_time_jitter"] = {
            "cv": round(cv, 4) if cv is not None else None,
            "status": jitter_status(cv, 0.15),
            "steps_analyzed": len(per_step["step_total"]),
            "percentiles": compute_percentiles(per_step["step_total"]),
            "outlier_indices": compute_zscore_outliers(per_step["step_total"]),
        }
    else:
        result["step_time_jitter"] = {"cv": None, "status": "unknown"}

    # 2. Compute jitter
    compute_times = per_step.get("compute") or per_step.get("fp_bp_time")
    if compute_times:
        cv = compute_cv(compute_times)
        result["compute_jitter"] = {
            "cv": round(cv, 4) if cv is not None else None,
            "status": jitter_status(cv, 0.10),
            "steps_analyzed": len(compute_times),
            "percentiles": compute_percentiles(compute_times),
            "outlier_indices": compute_zscore_outliers(compute_times),
        }
    else:
        # Estimate from aggregate CV
        compute_time = step_totals.get("compute", 0)
        total_time = sum(v for k, v in step_totals.items() if k != "step_total")
        if total_time > 0 and aggregate_cv is not None:
            COMPUTE_CV_FACTOR = 0.8
            estimated_cv = aggregate_cv * COMPUTE_CV_FACTOR
            result["compute_jitter"] = {
                "cv": round(estimated_cv, 4),
                "status": jitter_status(estimated_cv, 0.10),
                "note": "Estimated from step-time CV",
            }
        else:
            result["compute_jitter"] = {"cv": None, "status": "unknown"}

    # 3. Communication jitter
    comm_times = per_step.get("communication") or per_step.get("comm_time")
    if comm_times:
        cv = compute_cv(comm_times)
        result["communication_jitter"] = {
            "cv": round(cv, 4) if cv is not None else None,
            "status": jitter_status(cv, 0.15),
            "steps_analyzed": len(comm_times),
            "percentiles": compute_percentiles(comm_times),
            "outlier_indices": compute_zscore_outliers(comm_times),
        }
    else:
        result["communication_jitter"] = {"cv": None, "status": "unknown"}

    # 4. Alignment jitter (ratio variance across stages per step)
    step_total_series = per_step.get("step_total", [])
    compute_series = per_step.get("compute") or per_step.get("fp_bp_time", [])
    comm_series = per_step.get("communication") or per_step.get("comm_time", [])

    if len(step_total_series) >= 3 and len(compute_series) >= 3:
        min_len = min(len(step_total_series), len(compute_series))
        compute_ratios = [
            compute_series[i] / step_total_series[i]
            for i in range(min_len)
            if step_total_series[i] > 0
        ]
        if len(compute_ratios) >= 3:
            cv = compute_cv(compute_ratios)
            result["alignment_jitter"] = {
                "cv": round(cv, 4) if cv is not None else None,
                "status": jitter_status(cv, 0.10),
                "description": "Variance in compute-to-total ratio across steps",
            }

    return result


def analyze_cross_rank(trace_root: Path) -> Optional[dict]:
    """Analyze cross-rank alignment skew when cluster data is available."""
    rank_dirs = find_rank_dirs(trace_root)

    if len(rank_dirs) < 2:
        return None

    rank_avg_times: dict[int, float] = {}
    for rank_id, rank_dir in rank_dirs.items():
        step_csv = find_step_trace_csv(rank_dir)
        if not step_csv:
            continue
        try:
            avg_time = avg_step_time_from_csv(step_csv)
            if avg_time is not None:
                rank_avg_times[rank_id] = avg_time
        except Exception:
            continue

    if len(rank_avg_times) < 2:
        return None

    times = list(rank_avg_times.values())
    max_skew = max(times) - min(times)
    mean_time = sum(times) / len(times)

    return {
        "num_ranks": len(rank_avg_times),
        "max_skew_ms": round(max_skew, 3),
        "mean_step_time_ms": round(mean_time, 3),
        "skew_percent": round(max_skew / mean_time * 100, 2) if mean_time > 0 else 0,
        "status": "normal" if max_skew < 5 else ("warning" if max_skew < 20 else "critical"),
    }


def infer_root_causes(jitter: dict, cross_rank: Optional[dict]) -> list[str]:
    """Infer likely root causes for detected jitter across all dimensions."""
    causes: list[str] = []

    step_cv = jitter.get("step_time_jitter", {}).get("cv")
    if step_cv and step_cv > 0.10:
        causes.append("Step time variance detected, possibly from dynamic shapes or GC interference")

    compute_cv = jitter.get("compute_jitter", {}).get("cv")
    if compute_cv and compute_cv > 0.10:
        causes.append("Compute jitter detected, may indicate varying operator execution times")

    comm_cv = jitter.get("communication_jitter", {}).get("cv")
    if comm_cv and comm_cv > 0.15:
        causes.append("Communication jitter detected, may indicate network instability or load imbalance")

    alignment_cv = jitter.get("alignment_jitter", {}).get("cv")
    if alignment_cv and alignment_cv > 0.10:
        causes.append("Alignment jitter detected, phases are shifting relative to each other across steps")

    if cross_rank and cross_rank.get("max_skew_ms", 0) > 5:
        causes.append(
            f"Cross-rank alignment skew of {cross_rank['max_skew_ms']:.1f}ms detected, "
            "likely caused by CPU scheduling imbalance or slow-rank drag"
        )

    return causes


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze performance jitter from profiling data")
    parser.add_argument("--step-json", help="Step summary JSON path")
    parser.add_argument("--trace-root", help="Cluster profiler root for cross-rank analysis")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    step_json = load_optional_json(args.step_json)

    if not step_json:
        raise SystemExit("Step summary JSON is required (--step-json)")

    # 1. Analyze multi-dimensional jitter
    jitter = analyze_single_step(step_json)

    # 2. Cross-rank analysis (if cluster data available)
    cross_rank = None
    if args.trace_root:
        cross_rank = analyze_cross_rank(Path(args.trace_root))

    # 3. Infer root causes
    root_causes = infer_root_causes(jitter, cross_rank)

    # 4. Build recommendations
    recommendations: list[str] = []
    step_status = jitter.get("step_time_jitter", {}).get("status")
    if step_status in ("warning", "critical"):
        recommendations.append("Investigate step time variance: check for dynamic shapes, GC pauses, or OS scheduling interference")

    comm_status = jitter.get("communication_jitter", {}).get("status")
    if comm_status in ("warning", "critical"):
        recommendations.append("Investigate communication jitter: check for network congestion, HCCL buffer sizing, or slow-rank drag")

    compute_status = jitter.get("compute_jitter", {}).get("status")
    if compute_status in ("warning", "critical"):
        recommendations.append("Investigate compute jitter: check for dynamic shapes causing recompilation or varying operator execution times")

    if cross_rank and cross_rank.get("status") in ("warning", "critical"):
        recommendations.append(
            f"Address cross-rank skew ({cross_rank['max_skew_ms']:.1f}ms): "
            "check CPU binding, process priority, and slow-rank identification"
        )

    report = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "step_time_jitter": jitter.get("step_time_jitter"),
        "compute_jitter": jitter.get("compute_jitter"),
        "communication_jitter": jitter.get("communication_jitter"),
        "alignment_jitter": jitter.get("alignment_jitter"),
        "cross_rank_skew": cross_rank,
        "root_causes": root_causes,
        "recommended_actions": recommendations,
    }

    write_json(Path(args.output_json), report)
    print(json.dumps({
        "step_time_jitter_status": jitter.get("step_time_jitter", {}).get("status"),
        "compute_jitter_status": compute_status,
        "communication_jitter_status": comm_status,
        "cross_rank_available": cross_rank is not None,
        "root_causes_count": len(root_causes),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
