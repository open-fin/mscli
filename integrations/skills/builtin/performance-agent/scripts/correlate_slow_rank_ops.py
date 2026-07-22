#!/usr/bin/env python3
"""Correlate operator-level differences between jittery and stable ranks.

Compares operator profiles from op_summary/kernel_details CSV files
between the worst jittery rank and stable reference ranks to identify
which operators diverge the most.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

from perf_common import (
    find_rank_dirs,
    load_csv_rows,
    load_optional_json,
    normalize_key,
    parse_number,
    write_json,
)


# SyncBN-related operator name patterns
SYNCBN_PATTERNS = [
    r"(?i)(sync.?batch.?norm|syncbn|batch.?norm.?sync|bn.?sync|sync_batchnorm)",
]

# Operator name normalization keys
OP_NAME_KEYS = {"name", "op_name", "operator", "operator_name", "kernel_name", "op_name_full"}
OP_TIME_KEYS = {"duration", "duration_ms", "total_time", "total_time_ms", "time", "time_ms", "elapse_time"}
OP_COUNT_KEYS = {"count", "calls", "op_count", "launch_times"}


def _extract_op_name(row: dict) -> str:
    """Extract operator name from a CSV row."""
    for key, value in row.items():
        nk = normalize_key(key)
        if nk in OP_NAME_KEYS:
            return str(value).strip()
    return ""


def _extract_op_time_ms(row: dict) -> Optional[float]:
    """Extract operator total time in ms from a CSV row."""
    for key, value in row.items():
        nk = normalize_key(key)
        if nk in OP_TIME_KEYS:
            parsed = parse_number(value)
            if parsed is not None:
                return parsed
    return None


def _extract_op_count(row: dict) -> int:
    """Extract operator call count from a CSV row."""
    for key, value in row.items():
        nk = normalize_key(key)
        if nk in OP_COUNT_KEYS:
            parsed = parse_number(value)
            if parsed is not None:
                return int(parsed)
    return 1


def _classify_operator_category(name: str) -> str:
    """Classify an operator into a broad category."""
    for pattern in SYNCBN_PATTERNS:
        if re.search(pattern, name):
            return "SyncBN"
    name_lower = name.lower()
    if any(kw in name_lower for kw in ("allreduce", "all_reduce", "reduce_scatter", "allgather")):
        return "Communication"
    if any(kw in name_lower for kw in ("matmul", "gemm", "linear", "dense")):
        return "Matmul"
    if any(kw in name_lower for kw in ("conv", "convolution")):
        return "Conv"
    if any(kw in name_lower for kw in ("attention", "flash", "softmax")):
        return "Attention"
    if any(kw in name_lower for kw in ("norm", "layernorm", "rmsnorm")):
        return "Normalization"
    return "Other"


def _load_rank_operator_profile(rank_dir: Path) -> dict[str, dict]:
    """Load operator profile for a single rank from op_summary or kernel_details."""
    op_profiles: dict[str, dict] = {}

    # Try op_summary CSV first
    op_summary_paths = list(rank_dir.rglob("op_summary_*.csv"))
    if not op_summary_paths:
        op_summary_paths = list(rank_dir.rglob("op_summary.csv"))

    # Fallback to kernel_details
    kernel_paths = []
    if not op_summary_paths:
        kernel_paths = list(rank_dir.rglob("kernel_details.csv"))

    csv_paths = op_summary_paths or kernel_paths
    if not csv_paths:
        return {}

    for csv_path in csv_paths[:1]:  # Use first match
        rows = load_csv_rows(csv_path)
        for row in rows:
            name = _extract_op_name(row)
            if not name:
                continue
            time_ms = _extract_op_time_ms(row)
            count = _extract_op_count(row)
            if time_ms is not None:
                existing = op_profiles.get(name)
                if existing:
                    op_profiles[name] = {
                        "name": name,
                        "time_ms": existing["time_ms"] + time_ms,
                        "count": existing["count"] + count,
                        "category": existing["category"],
                    }
                else:
                    op_profiles[name] = {
                        "name": name,
                        "time_ms": time_ms,
                        "count": count,
                        "category": _classify_operator_category(name),
                    }

    return op_profiles


def _aggregate_reference_profile(
    reference_ranks: list[int],
    rank_dirs: dict[int, Path],
) -> dict[str, dict]:
    """Aggregate operator profiles across stable reference ranks."""
    aggregated: dict[str, dict] = {}

    for rank_id in reference_ranks:
        rank_dir = rank_dirs.get(rank_id)
        if rank_dir is None:
            continue
        profile = _load_rank_operator_profile(rank_dir)
        for name, data in profile.items():
            if name not in aggregated:
                aggregated[name] = {
                    "name": name,
                    "total_time_ms": 0.0,
                    "total_count": 0,
                    "rank_count": 0,
                    "category": data["category"],
                }
            agg = aggregated[name]
            agg["total_time_ms"] += data["time_ms"]
            agg["total_count"] += data["count"]
            agg["rank_count"] += 1

    # Compute mean per reference rank
    for name, agg in aggregated.items():
        rc = max(agg["rank_count"], 1)
        agg["mean_time_ms"] = round(agg["total_time_ms"] / rc, 4)
        agg["mean_count"] = round(agg["total_count"] / rc, 1)

    return aggregated


def correlate_slow_rank_ops(
    trace_root: Path,
    rank_variance: Optional[dict],
    cluster: Optional[dict],
) -> dict:
    """Compare operator profiles between jittery and stable ranks."""
    rank_dirs = find_rank_dirs(trace_root)

    if len(rank_dirs) < 2:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "slow_rank_op_analysis_available": False,
            "reason": "Need at least 2 ranks",
        }

    # Determine target rank and reference ranks
    target_rank = None
    reference_ranks = []

    if rank_variance and rank_variance.get("jittery_ranks"):
        target_rank = rank_variance["worst_jittery_rank"]
        jittery = set(rank_variance["jittery_ranks"])
        reference_ranks = [r for r in rank_dirs if r not in jittery]
    elif cluster and cluster.get("slow_ranks"):
        target_rank = cluster["slow_ranks"][0]
        slow_set = set(cluster["slow_ranks"])
        reference_ranks = [r for r in rank_dirs if r not in slow_set]
    else:
        # Pick the slowest rank by mean step time
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "slow_rank_op_analysis_available": False,
            "reason": "No jittery or slow ranks identified",
        }

    if target_rank is None or not reference_ranks:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "slow_rank_op_analysis_available": False,
            "reason": "Insufficient rank data for comparison",
        }

    # Load profiles
    target_dir = rank_dirs.get(target_rank)
    if target_dir is None:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "slow_rank_op_analysis_available": False,
            "reason": f"Target rank {target_rank} directory not found",
        }

    target_profile = _load_rank_operator_profile(target_dir)
    reference_profile = _aggregate_reference_profile(reference_ranks, rank_dirs)

    if not target_profile or not reference_profile:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "slow_rank_op_analysis_available": False,
            "reason": "Operator profile data unavailable for comparison",
        }

    # Compute divergence
    divergent_ops = []
    for name, target_data in target_profile.items():
        ref_data = reference_profile.get(name)
        if ref_data is None:
            # Operator only in target rank
            if target_data["time_ms"] > 1.0:
                divergent_ops.append({
                    "operator": name,
                    "slow_rank_time_ms": round(target_data["time_ms"], 3),
                    "reference_mean_ms": None,
                    "slowdown_ratio": None,
                    "category": target_data["category"],
                    "note": "Operator not found in reference ranks",
                })
            continue

        ref_mean = ref_data["mean_time_ms"]
        if ref_mean <= 0:
            continue

        ratio = target_data["time_ms"] / ref_mean
        if ratio > 1.5:  # Only report significant divergence
            divergent_ops.append({
                "operator": name,
                "slow_rank_time_ms": round(target_data["time_ms"], 3),
                "reference_mean_ms": round(ref_mean, 3),
                "slowdown_ratio": round(ratio, 2),
                "category": target_data["category"],
            })

    # Sort by slowdown ratio (largest first)
    divergent_ops.sort(key=lambda x: x.get("slowdown_ratio") or 0, reverse=True)

    # Check for SyncBN divergence
    syncbn_divergent = [
        op for op in divergent_ops if op["category"] == "SyncBN"
    ]
    syncbn_divergence_detected = any(
        op.get("slowdown_ratio") is not None and op["slowdown_ratio"] > 2.0
        for op in syncbn_divergent
    )

    primary_divergent = divergent_ops[0]["operator"] if divergent_ops else None

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "slow_rank_op_analysis_available": True,
        "analyzed_rank": target_rank,
        "reference_ranks": reference_ranks[:8],  # Cap at 8
        "total_operators_compared": len(set(target_profile) | set(reference_profile)),
        "divergent_operators_count": len(divergent_ops),
        "top_divergent_operators": divergent_ops[:10],
        "syncbn_divergence_detected": syncbn_divergence_detected,
        "syncbn_divergent_operators": syncbn_divergent[:5],
        "primary_divergent_operator": primary_divergent,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correlate operator differences between jittery and stable ranks"
    )
    parser.add_argument("--trace-root", required=True, help="Profiler root directory")
    parser.add_argument("--rank-variance-json", help="Rank variance analysis JSON")
    parser.add_argument("--cluster-json", help="Cluster analysis JSON")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    rank_variance = load_optional_json(args.rank_variance_json)
    cluster = load_optional_json(args.cluster_json)

    result = correlate_slow_rank_ops(
        Path(args.trace_root).resolve(),
        rank_variance,
        cluster,
    )

    write_json(Path(args.output_json), result)
    print(json.dumps({
        "slow_rank_op_analysis_available": result.get("slow_rank_op_analysis_available", False),
        "analyzed_rank": result.get("analyzed_rank"),
        "divergent_count": result.get("divergent_operators_count", 0),
        "syncbn_divergence": result.get("syncbn_divergence_detected", False),
        "primary_divergent": result.get("primary_divergent_operator"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
