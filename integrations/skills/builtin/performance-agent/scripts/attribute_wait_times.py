#!/usr/bin/env python3
"""Attribute inter-rank wait times to specific collective operation types.

Uses collective type breakdown and rank variance data to estimate how
much synchronization wait time is caused by each collective category
(e.g., SyncBN barrier waits vs normal gradient AllReduce).
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from perf_common import load_optional_json, write_json


def attribute_wait_times(
    collective_types: Optional[dict],
    rank_variance: Optional[dict],
    cluster: Optional[dict],
    step: Optional[dict],
    communication: Optional[dict],
) -> dict:
    """Attribute wait times to specific collective types."""
    if not collective_types or not collective_types.get("collective_type_analysis_available"):
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "wait_time_attribution_available": False,
            "reason": "Collective type analysis not available",
        }

    types = collective_types.get("types", [])
    if not types:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "wait_time_attribution_available": False,
            "reason": "No collective types found",
        }

    # Estimate total wait from rank variance drag effect or wait_ratio
    total_wait_ms = 0.0

    if rank_variance and rank_variance.get("rank_variance_analysis_available"):
        drag = rank_variance.get("drag_effect_ms", 0)
        if drag > 0:
            total_wait_ms = drag

    # Fallback: estimate from cluster wait_ratio_analysis
    if total_wait_ms <= 0 and cluster and cluster.get("wait_ratio_analysis"):
        wait_analysis = cluster["wait_ratio_analysis"]
        delta = wait_analysis.get("wait_ratio_delta", 0)
        rank_times = cluster.get("rank_step_times_ms", {})
        if rank_times and delta > 0:
            numeric_times = []
            for v in rank_times.values():
                try:
                    numeric_times.append(float(v))
                except (ValueError, TypeError):
                    continue
            if numeric_times:
                mean_step = sum(numeric_times) / len(numeric_times)
            else:
                mean_step = 0
            total_wait_ms = mean_step * delta

    # Fallback: estimate from communication time and jitter
    if total_wait_ms <= 0 and communication and rank_variance:
        comm_time = communication.get("total_time_ms", 0)
        worst_cv = rank_variance.get("worst_rank_cv", 0)
        if comm_time > 0 and worst_cv > 0:
            # Rough estimate: wait proportional to variance share of comm
            total_wait_ms = comm_time * min(worst_cv, 1.0)

    if total_wait_ms <= 0:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "wait_time_attribution_available": False,
            "reason": "Cannot estimate total wait time from available data",
        }

    # Attribute wait by collective type share
    total_collective_time = collective_types.get("total_collective_time_ms", 0)
    jittery_ranks = []
    if rank_variance:
        jittery_ranks = rank_variance.get("jittery_ranks", [])

    attributions = []
    for type_info in types:
        type_name = type_info["type"]
        share = type_info["share_percent"]
        type_time = type_info["total_time_ms"]

        if share <= 0:
            continue

        # Proportional attribution: this type's share of total wait
        estimated_wait = total_wait_ms * (share / 100.0)

        # Build mechanism description
        mechanism = ""
        is_primary = False

        if type_name == "SyncBN" and jittery_ranks:
            rank_str = ", ".join(str(r) for r in jittery_ranks[:3])
            mechanism = (
                f"jittery Rank(s) {rank_str} delay SyncBN barrier, "
                f"all other ranks wait {estimated_wait:.1f}ms per step"
            )
            # SyncBN + jittery ranks is a strong primary indicator
            if share > 20:
                is_primary = True
        elif type_name == "SyncBN":
            mechanism = "SyncBN requires cross-rank synchronization at every training step"
            if share > 30:
                is_primary = True
        elif type_name == "GradientAllReduce":
            mechanism = "normal gradient synchronization overhead across ranks"
        elif type_name == "ReduceScatter":
            mechanism = "tensor parallelism ReduceScatter synchronization"
        elif type_name == "AllGather":
            mechanism = "tensor parallelism AllGather synchronization"
        elif type_name == "SmallPacketAllReduce":
            mechanism = "excessive small-packet communication (possible ZeRO-3 or small bucket size)"
        else:
            mechanism = f"{type_name} collective synchronization"

        attributions.append({
            "collective_type": type_name,
            "estimated_wait_ms": round(estimated_wait, 3),
            "share_of_total_wait_percent": round(share, 2),
            "collective_time_ms": round(type_time, 3),
            "is_primary": is_primary,
            "mechanism": mechanism,
        })

    # Sort by estimated wait (largest first)
    attributions.sort(key=lambda a: a["estimated_wait_ms"], reverse=True)

    # Identify primary wait source
    primary_wait_source = None
    primary_attribution = None
    for attr in attributions:
        if attr["is_primary"]:
            primary_wait_source = attr["collective_type"]
            primary_attribution = attr
            break

    # If no primary marked, use the largest
    if primary_wait_source is None and attributions:
        primary_wait_source = attributions[0]["collective_type"]
        primary_attribution = attributions[0]

    # Estimate savings if primary source is eliminated
    elimination_savings_ms = 0.0
    elimination_savings_percent = 0.0
    if primary_attribution:
        elimination_savings_ms = primary_attribution["estimated_wait_ms"]
        avg_step_time = 0
        if step and step.get("average_step_time_ms"):
            avg_step_time = step["average_step_time_ms"]
        elif rank_variance and rank_variance.get("median_step_time_ms"):
            avg_step_time = rank_variance["median_step_time_ms"]
        if avg_step_time > 0:
            elimination_savings_percent = round(
                elimination_savings_ms / avg_step_time * 100, 2
            )

    # Build recommendations
    recommendations = []
    if primary_wait_source == "SyncBN":
        recommendations.extend([
            "Consider replacing SyncBN with GroupNorm or FrozenBN to reduce cross-rank synchronization",
            "If SyncBN is required, reduce synchronization frequency (e.g., sync every N steps)",
            "Check if batch normalization parameters can be computed locally",
        ])
    elif primary_wait_source == "SmallPacketAllReduce":
        recommendations.extend([
            "Increase gradient accumulation bucket size (HCCL_BUFFSIZE)",
            "Consider switching from ZeRO-3 to ZeRO-1/2 to reduce small-packet communication",
        ])
    elif primary_wait_source == "GradientAllReduce":
        recommendations.extend([
            "Enable computation-communication overlap (pipeline parallelism or async AllReduce)",
            "Increase micro batch size to amortize AllReduce overhead",
        ])

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "wait_time_attribution_available": True,
        "total_wait_estimated_ms": round(total_wait_ms, 3),
        "attributions": attributions,
        "primary_wait_source": primary_wait_source,
        "elimination_savings_ms": round(elimination_savings_ms, 3),
        "elimination_savings_percent": elimination_savings_percent,
        "recommendations": recommendations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attribute inter-rank wait times to specific collective types"
    )
    parser.add_argument("--trace-root", help="Profiler root directory (unused, for pipeline compat)")
    parser.add_argument("--collective-types-json", help="Collective type analysis JSON")
    parser.add_argument("--rank-variance-json", help="Rank variance analysis JSON")
    parser.add_argument("--cluster-json", help="Cluster analysis JSON")
    parser.add_argument("--step-json", help="Step breakdown JSON")
    parser.add_argument("--communication-json", help="Communication summary JSON")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    collective_types = load_optional_json(args.collective_types_json)
    rank_variance = load_optional_json(args.rank_variance_json)
    cluster = load_optional_json(args.cluster_json)
    step = load_optional_json(args.step_json)
    communication = load_optional_json(args.communication_json)

    result = attribute_wait_times(
        collective_types, rank_variance, cluster, step, communication,
    )

    write_json(Path(args.output_json), result)
    print(json.dumps({
        "wait_time_attribution_available": result.get("wait_time_attribution_available", False),
        "primary_wait_source": result.get("primary_wait_source"),
        "total_wait_estimated_ms": result.get("total_wait_estimated_ms", 0),
        "attribution_count": len(result.get("attributions", [])),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
