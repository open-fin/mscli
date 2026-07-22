#!/usr/bin/env python3
"""Analyze per-rank step time variance to detect jittery ranks.

Computes coefficient of variation (CV) for each rank's step time series,
identifies jittery ranks (high CV), and quantifies the drag effect on
cluster synchronization.
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Optional

from perf_common import find_rank_dirs, find_step_trace_csv, write_json


def load_per_step_times(step_csv: Path) -> list[float]:
    """Load per-step time series from step_trace_time.csv.

    Returns list of step times in ms.
    """
    rows: list[dict] = []
    with step_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        rows = [dict(row) for row in reader]

    if not rows:
        return []

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

    step_times: list[float] = []
    for row in rows:
        for header, value in row.items():
            key = norm(header)
            if "step" in key and ("time" in key or "total" in key or "interval" in key):
                try:
                    num = float(value.replace(",", "").strip())
                    step_times.append(num)
                except (ValueError, AttributeError):
                    continue
                break  # Only take first matching column per row
    return step_times


def compute_stats(values: list[float]) -> Optional[dict]:
    """Compute mean, std, CV, min, max, p95, p99 for a time series."""
    if not values or len(values) < 2:
        return None

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance)
    cv = std / mean if mean > 0 else 0

    sorted_vals = sorted(values)

    def percentile(p: float) -> float:
        idx = math.ceil(n * p / 100) - 1
        return sorted_vals[max(0, min(idx, n - 1))]

    return {
        "mean_ms": round(mean, 3),
        "std_ms": round(std, 3),
        "cv": round(cv, 4),
        "min_ms": round(sorted_vals[0], 3),
        "max_ms": round(sorted_vals[-1], 3),
        "p95_ms": round(percentile(95), 3),
        "p99_ms": round(percentile(99), 3),
        "steps": n,
    }


def analyze_rank_variance(trace_root: Path) -> dict:
    """Analyze per-rank step time variance across all ranks."""
    rank_dirs = find_rank_dirs(trace_root)

    if len(rank_dirs) < 2:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "rank_variance_analysis_available": False,
            "reason": "Need at least 2 ranks with step data",
        }

    per_rank_stats: dict[str, dict] = {}
    for rank_id, rank_dir in rank_dirs.items():
        step_csv = find_step_trace_csv(rank_dir)
        if not step_csv:
            continue
        step_times = load_per_step_times(step_csv)
        stats = compute_stats(step_times)
        if stats is not None:
            per_rank_stats[str(rank_id)] = stats

    if len(per_rank_stats) < 2:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "rank_variance_analysis_available": False,
            "reason": "Fewer than 2 ranks with valid step data",
        }

    # Identify jittery ranks: CV > 0.10 or p95/mean > 1.2
    cv_threshold = 0.10
    p95_ratio_threshold = 1.2

    jittery_ranks = []
    stable_ranks = []
    for rank_id, stats in per_rank_stats.items():
        cv = stats["cv"]
        p95_ratio = stats["p95_ms"] / stats["mean_ms"] if stats["mean_ms"] > 0 else 0
        if cv > cv_threshold or p95_ratio > p95_ratio_threshold:
            jittery_ranks.append(int(rank_id))
        else:
            stable_ranks.append(int(rank_id))

    jittery_ranks.sort()
    stable_ranks.sort()

    # Compute CV spread: max_rank_cv / median_rank_cv
    all_cvs = [s["cv"] for s in per_rank_stats.values() if s["cv"] > 0]
    rank_cv_spread = 0.0
    if len(all_cvs) >= 2:
        sorted_cvs = sorted(all_cvs)
        median_cv = sorted_cvs[len(sorted_cvs) // 2]
        max_cv = sorted_cvs[-1]
        rank_cv_spread = round(max_cv / median_cv, 2) if median_cv > 0 else 0.0

    # Compute drag effect: worst jittery rank's max time - median rank's mean time
    drag_effect_ms = 0.0
    worst_jittery_rank = None
    worst_rank_cv = 0.0
    median_mean = 0.0

    all_means = sorted(s["mean_ms"] for s in per_rank_stats.values())
    median_mean = all_means[len(all_means) // 2]

    if jittery_ranks:
        worst_rank_id = max(
            jittery_ranks,
            key=lambda r: per_rank_stats[str(r)]["cv"],
        )
        worst_stats = per_rank_stats[str(worst_rank_id)]
        worst_rank_cv = worst_stats["cv"]
        drag_effect_ms = round(worst_stats["max_ms"] - median_mean, 3)
        worst_jittery_rank = worst_rank_id

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "rank_variance_analysis_available": True,
        "total_ranks": len(rank_dirs),
        "ranks_with_data": len(per_rank_stats),
        "per_rank_stats": per_rank_stats,
        "jittery_ranks": jittery_ranks,
        "stable_ranks": stable_ranks,
        "rank_cv_spread": rank_cv_spread,
        "drag_effect_ms": drag_effect_ms,
        "median_step_time_ms": round(median_mean, 3),
        "worst_jittery_rank": worst_jittery_rank,
        "worst_rank_cv": worst_rank_cv,
        "cv_threshold": cv_threshold,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze per-rank step time variance to detect jittery ranks"
    )
    parser.add_argument("--trace-root", required=True, help="Cluster profiler root directory")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    trace_root = Path(args.trace_root).resolve()
    result = analyze_rank_variance(trace_root)

    write_json(Path(args.output_json), result)
    print(json.dumps({
        "rank_variance_analysis_available": result.get("rank_variance_analysis_available", False),
        "jittery_ranks": result.get("jittery_ranks", []),
        "worst_jittery_rank": result.get("worst_jittery_rank"),
        "drag_effect_ms": result.get("drag_effect_ms", 0),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
