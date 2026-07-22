#!/usr/bin/env python3
"""Detect slow ranks in a cluster profiling directory.

Analyzes per-rank step_trace_time.csv files to identify outlier ranks
using Dixon Q-test (<=25 ranks) or 3-sigma rule (>25 ranks), then
classifies the bottleneck type (host_dispatch, compute, or communication)
using expert rules from the cluster-fast-slow-rank-detector skill.
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

# Dixon Q-test critical values at 99.5% confidence for sample sizes 3-25.
# Source: standard statistical tables for two-tailed outlier detection.
_DIXON_Q_CRITICAL: dict[int, float] = {
    3: 0.994, 4: 0.926, 5: 0.821, 6: 0.740, 7: 0.680,
    8: 0.634, 9: 0.598, 10: 0.568, 11: 0.542, 12: 0.521,
    13: 0.503, 14: 0.488, 15: 0.475, 16: 0.463, 17: 0.452,
    18: 0.442, 19: 0.433, 20: 0.425, 21: 0.418, 22: 0.411,
    23: 0.404, 24: 0.399, 25: 0.393,
}



def load_step_times(step_csv: Path) -> dict[str, float]:
    """Load step trace and return stage averages in ms."""
    rows: list[dict] = []
    with step_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        rows = [dict(row) for row in reader]

    if not rows:
        return {}

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", s.strip().lower()).strip("_")

    stage_totals: dict[str, float] = {}
    for row in rows:
        for header, value in row.items():
            key = norm(header)
            try:
                num = float(value.replace(",", "").strip())
            except (ValueError, AttributeError):
                continue
            if "step" in key and ("time" in key or "total" in key or "interval" in key):
                stage_totals.setdefault("step_total", []).append(num)
            elif any(t in key for t in ("fp", "forward", "compute", "kernel", "backward", "bp")):
                stage_totals.setdefault("compute", []).append(num)
            elif any(t in key for t in ("comm", "allreduce", "reduce", "gather", "hccl")):
                stage_totals.setdefault("communication", []).append(num)
            elif any(t in key for t in ("idle", "gap", "wait", "tail", "bubble", "free")):
                stage_totals.setdefault("free", []).append(num)

    result = {}
    for stage, values in stage_totals.items():
        if values:
            result[f"{stage}_ms"] = sum(values) / len(values)
    result["steps"] = len(rows)
    return result


def dixon_q_test(values: dict[int, float]) -> tuple[list[int], list[int]]:
    """Detect outliers using Dixon Q-test at 99.5% confidence.

    Best for small sample sizes (3-25 ranks). Uses the r10 statistic:
      Q = |suspect - nearest| / range

    Returns (slow_ranks, fast_ranks).
    """
    n = len(values)
    if n < 3 or n > 25:
        return [], []

    q_crit = _DIXON_Q_CRITICAL.get(n)
    if q_crit is None:
        return [], []

    sorted_pairs = sorted(values.items(), key=lambda item: item[1])
    sorted_vals = [v for _, v in sorted_pairs]
    data_range = sorted_vals[-1] - sorted_vals[0]
    if data_range <= 0:
        return [], []

    slow: list[int] = []
    fast: list[int] = []

    # Check high outlier (slowest rank)
    q_high = (sorted_vals[-1] - sorted_vals[-2]) / data_range
    if q_high > q_crit:
        slow.append(sorted_pairs[-1][0])

    # Check low outlier (fastest rank)
    q_low = (sorted_vals[1] - sorted_vals[0]) / data_range
    if q_low > q_crit:
        fast.append(sorted_pairs[0][0])

    return slow, fast


def sigma_rule(values: dict[int, float], n_sigma: float = 3.0) -> tuple[list[int], list[int]]:
    """Detect outliers using the N-sigma rule.

    Best for larger sample sizes (>25 ranks). Flags values beyond
    mean +/- n_sigma * std as outliers.

    Returns (slow_ranks, fast_ranks).
    """
    if len(values) < 2:
        return [], []

    vals = list(values.values())
    mean = sum(vals) / len(vals)
    variance = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(variance)

    if std <= 0:
        return [], []

    slow = [r for r, v in values.items() if v > mean + n_sigma * std]
    fast = [r for r, v in values.items() if v < mean - n_sigma * std]
    return slow, fast


def detect_outliers(values: dict[int, float]) -> tuple[list[int], list[int]]:
    """Detect outlier ranks using Dixon Q-test or 3-sigma rule.

    Uses Dixon Q-test for <=25 ranks (statistically rigorous for small samples)
    and 3-sigma rule for >25 ranks. Falls back to median-ratio when the primary
    method finds nothing but the spread is large.

    Returns (slow_ranks, fast_ranks).
    """
    if len(values) < 2:
        return [], []

    n = len(values)

    # Primary method selection
    if n <= 25:
        slow, fast = dixon_q_test(values)
        method_used = "dixon_q_test"
    else:
        slow, fast = sigma_rule(values, n_sigma=3.0)
        method_used = "3_sigma"

    # Fallback: median-ratio when primary finds nothing
    if not slow:
        vals = sorted(values.values())
        median = vals[n // 2]
        if median > 0:
            slow = [r for r, v in values.items() if v > median * 1.3]
            fast = [r for r, v in values.items() if v < median * (1.0 / 1.3)]
            if len(slow) >= n / 2:
                slow = []
            if len(fast) >= n / 2:
                fast = []
            if slow or fast:
                method_used = "median_ratio_fallback"

    return slow, fast


def analyze_wait_ratio(rank_metrics: dict[int, dict]) -> Optional[dict]:
    """Analyze wait_ratio across ranks to detect slow-card induced blocking.

    wait_ratio = communication_wait_time / total_op_execution_time
    A delta > 0.2 between max and min wait_ratio indicates a slow card.

    Returns wait_ratio analysis dict or None if data is insufficient.
    """
    wait_ratios: dict[int, float] = {}
    for rank_id, metrics in rank_metrics.items():
        comm_ms = metrics.get("communication_ms", 0)
        total_ms = metrics.get("step_total_ms", 0)
        if total_ms > 0 and comm_ms > 0:
            wait_ratios[rank_id] = comm_ms / total_ms

    if len(wait_ratios) < 2:
        return None

    ratios = list(wait_ratios.values())
    max_ratio = max(ratios)
    min_ratio = min(ratios)
    delta = max_ratio - min_ratio

    slow_card_rank = None
    if delta > 0.2:
        # The card with the shortest comm time is the bottleneck
        # (it makes others wait the longest)
        slow_card_rank = max(wait_ratios, key=lambda r: wait_ratios[r])

    return {
        "max_wait_ratio": round(max_ratio, 4),
        "min_wait_ratio": round(min_ratio, 4),
        "wait_ratio_delta": round(delta, 4),
        "slow_card_detected": delta > 0.2,
        "slow_card_rank": slow_card_rank,
        "threshold": 0.2,
    }


def classify_bottleneck(
    rank_metrics: dict[int, dict],
    slow_ranks: list[int],
    fast_ranks: list[int],
    wait_ratio_analysis: Optional[dict] = None,
) -> dict:
    """Classify the bottleneck type for slow ranks."""
    if not slow_ranks or not rank_metrics:
        return {"bottleneck_type": None, "evidence": {}}

    slow_id = slow_ranks[0]
    slow_m = rank_metrics.get(slow_id, {})

    all_compute_pct: list[float] = []
    all_comm_pct: list[float] = []
    all_free_pct: list[float] = []
    for metrics in rank_metrics.values():
        total = metrics.get("step_total_ms", 0)
        if total <= 0:
            continue
        compute_pct = (metrics.get("compute_ms", 0) / total) * 100
        comm_pct = (metrics.get("communication_ms", 0) / total) * 100
        free_pct = (metrics.get("free_ms", 0) / total) * 100
        all_compute_pct.append(compute_pct)
        all_comm_pct.append(comm_pct)
        all_free_pct.append(free_pct)

    mean_compute_pct = sum(all_compute_pct) / len(all_compute_pct) if all_compute_pct else 0
    mean_free_pct = sum(all_free_pct) / len(all_free_pct) if all_free_pct else 0
    mean_comm_pct = sum(all_comm_pct) / len(all_comm_pct) if all_comm_pct else 0

    slow_total = slow_m.get("step_total_ms", 0)
    slow_free_pct = (slow_m.get("free_ms", 0) / slow_total * 100) if slow_total > 0 else 0
    slow_compute_pct = (slow_m.get("compute_ms", 0) / slow_total * 100) if slow_total > 0 else 0
    slow_comm_pct = (slow_m.get("communication_ms", 0) / slow_total * 100) if slow_total > 0 else 0

    evidence = {
        f"rank_{slow_id}_free_time_percent": round(slow_free_pct, 1),
        f"rank_{slow_id}_compute_time_percent": round(slow_compute_pct, 1),
        f"rank_{slow_id}_comm_time_percent": round(slow_comm_pct, 1),
        "mean_free_percent": round(mean_free_pct, 1),
        "mean_compute_percent": round(mean_compute_pct, 1),
    }

    # Add wait_ratio evidence if available
    if wait_ratio_analysis and wait_ratio_analysis.get("slow_card_detected"):
        evidence["wait_ratio_delta"] = wait_ratio_analysis["wait_ratio_delta"]
        evidence["slow_card_rank"] = wait_ratio_analysis["slow_card_rank"]

    # Rule 1: Host dispatch bottleneck
    if slow_free_pct > 10 and slow_free_pct > mean_free_pct * 2:
        return {
            "bottleneck_type": "host_dispatch",
            "slow_rank_id": slow_id,
            "evidence": evidence,
            "diagnosis": (
                f"Rank {slow_id} is NOT a fast card — it is a slow card causing cluster blocking. "
                f"CPU dispatch is slow, causing NPU starvation (Free Time {slow_free_pct:.1f}% vs mean {mean_free_pct:.1f}%)."
            ),
            "recommended_actions": [
                f"Check CPU affinity and process scheduling for Rank {slow_id}",
                "Check for CPU-bound work on the host serving this rank",
                "Compare API dispatch stats (launch, aclrtSynchronizeDevice) between slow and fast ranks",
            ],
        }

    # Rule 2: Compute bottleneck
    if slow_compute_pct > mean_compute_pct * 1.15:
        return {
            "bottleneck_type": "compute",
            "slow_rank_id": slow_id,
            "evidence": evidence,
            "diagnosis": (
                f"Rank {slow_id} is a compute-bound slow card "
                f"(Compute {slow_compute_pct:.1f}% vs mean {mean_compute_pct:.1f}%)."
            ),
            "recommended_actions": [
                "Compare operator stats between slow and fast ranks to find degraded operators",
                "Check if operator call counts differ (load imbalance) or avg times differ (hardware degradation)",
                "Check for dynamic shapes causing recompilation on this rank",
            ],
        }

    # Rule 3: Communication bottleneck (enhanced with wait_ratio)
    comm_elevated = slow_comm_pct > 30 and slow_comm_pct > mean_comm_pct * 1.2
    wait_ratio_confirms = (
        wait_ratio_analysis
        and wait_ratio_analysis.get("slow_card_detected")
        and wait_ratio_analysis.get("slow_card_rank") == slow_id
    )
    if comm_elevated or wait_ratio_confirms:
        return {
            "bottleneck_type": "communication",
            "slow_rank_id": slow_id,
            "evidence": evidence,
            "diagnosis": (
                f"Rank {slow_id} has high communication overhead "
                f"(Comm {slow_comm_pct:.1f}% vs mean {mean_comm_pct:.1f}%)."
                + (f" wait_ratio delta confirms slow-card blocking ({wait_ratio_analysis['wait_ratio_delta']:.3f})."
                   if wait_ratio_confirms else "")
            ),
            "recommended_actions": [
                "Check for slow links (SDMA bandwidth < 2 GB/s indicates small-packet or alignment issues)",
                "Check if ZeRO3 is causing excessive small-packet communication",
                "Verify HCCS/RDMA link health",
            ],
        }

    return {
        "bottleneck_type": "general",
        "slow_rank_id": slow_id,
        "evidence": evidence,
        "diagnosis": f"Rank {slow_id} is slower than average but the bottleneck type is not clearly classifiable.",
        "recommended_actions": [
            "Compare full step breakdown between slow and fast ranks",
            "Check for OS-level scheduling differences or hardware issues",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect slow ranks in cluster profiling data")
    parser.add_argument("--trace-root", required=True, help="Cluster profiler root directory")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    trace_root = Path(args.trace_root).resolve()

    # 1. Find rank directories
    rank_dirs = find_rank_dirs(trace_root)
    if not rank_dirs:
        step_csv = find_step_trace_csv(trace_root)
        if step_csv:
            report = {
                "schema_version": "performance-agent/0.1",
                "skill": "performance-agent",
                "total_ranks": 1,
                "cluster_analysis_available": False,
                "note": "Single-rank data, cluster analysis not applicable",
            }
            write_json(Path(args.output_json), report)
            print(json.dumps({"total_ranks": 1, "cluster_analysis_available": False}))
            return 0
        else:
            print("No rank directories or step_trace_time.csv found", file=sys.stderr)
            raise SystemExit(1)

    # 2. Load per-rank metrics
    rank_metrics: dict[int, dict] = {}
    rank_step_times: dict[int, float] = {}
    for rank_id, rank_dir in rank_dirs.items():
        step_csv = find_step_trace_csv(rank_dir)
        if not step_csv:
            continue
        metrics = load_step_times(step_csv)
        if metrics:
            rank_metrics[rank_id] = metrics
            rank_step_times[rank_id] = metrics.get("step_total_ms", 0)

    if len(rank_metrics) < 2:
        report = {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "total_ranks": len(rank_dirs),
            "ranks_with_data": len(rank_metrics),
            "cluster_analysis_available": len(rank_metrics) >= 2,
            "note": "Need at least 2 ranks with step data for cluster analysis",
        }
        write_json(Path(args.output_json), report)
        return 0

    # 3. Detect outliers (Dixon Q-test or 3-sigma)
    slow_ranks, fast_ranks = detect_outliers(rank_step_times)

    # 4. Analyze wait_ratio across ranks
    wait_ratio_analysis = analyze_wait_ratio(rank_metrics)

    # 5. Classify bottleneck
    analysis = classify_bottleneck(rank_metrics, slow_ranks, fast_ranks, wait_ratio_analysis)

    n = len(rank_step_times)
    method_used = "dixon_q_test" if n <= 25 else "3_sigma"

    report = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "detection_method": method_used,
        "total_ranks": len(rank_dirs),
        "ranks_with_data": len(rank_metrics),
        "cluster_analysis_available": True,
        "slow_ranks": slow_ranks,
        "fast_ranks": fast_ranks,
        "rank_step_times_ms": {str(k): round(v, 3) for k, v in sorted(rank_step_times.items())},
        "wait_ratio_analysis": wait_ratio_analysis,
        "analysis": analysis,
    }

    write_json(Path(args.output_json), report)
    print(json.dumps({
        "total_ranks": len(rank_dirs),
        "detection_method": method_used,
        "slow_ranks": slow_ranks,
        "fast_ranks": fast_ranks,
        "bottleneck_type": analysis.get("bottleneck_type"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
