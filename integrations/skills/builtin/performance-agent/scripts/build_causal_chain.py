#!/usr/bin/env python3
"""Build dynamic multi-layer causal chains from all analysis artifacts.

Constructs causal chains of depth 3-5 by combining bottleneck classification,
collective type breakdown, rank variance, operator divergence, and wait time
attribution data. Produces the deepest available root cause analysis.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from perf_common import load_optional_json, write_json


# ---------------------------------------------------------------------------
# Chain builders: each returns a list of chain candidates
# ---------------------------------------------------------------------------

def _build_syncbn_jitter_chain(
    bottlenecks: dict,
    collective_types: Optional[dict],
    rank_variance: Optional[dict],
    slow_rank_ops: Optional[dict],
    wait_attribution: Optional[dict],
    communication: Optional[dict],
) -> list[dict]:
    """Build causal chain: comm_overhead -> SyncBN dominant -> rank jitter -> slow op."""
    chains = []

    # Gate: need collective types showing SyncBN and rank variance showing jitter
    if not collective_types or not collective_types.get("syncbn_dominant"):
        return chains
    if not rank_variance or not rank_variance.get("jittery_ranks"):
        return chains

    syncbn_share = collective_types.get("syncbn_share_percent", 0)
    jittery_ranks = rank_variance["jittery_ranks"]
    worst_rank = rank_variance.get("worst_jittery_rank")
    worst_cv = rank_variance.get("worst_rank_cv", 0)
    drag = rank_variance.get("drag_effect_ms", 0)

    # Layer 1: symptom
    layer1_evidence = []
    if communication:
        pressure = communication.get("communication_pressure", "unknown")
        comm_time = communication.get("total_time_ms", 0)
        layer1_evidence.append(f"comm_pressure={pressure}")
        if comm_time > 0:
            layer1_evidence.append(f"comm_total={comm_time:.1f}ms")

    # Layer 2: SyncBN dominant
    layer2_evidence = [f"SyncBN_share={syncbn_share:.1f}%"]

    # Layer 3: rank jitter
    rank_str = ", ".join(str(r) for r in jittery_ranks[:3])
    layer3_evidence = [
        f"jittery_ranks=[{rank_str}]",
        f"worst_cv={worst_cv:.3f}",
        f"drag_effect={drag:.1f}ms",
    ]

    # Layer 4: specific operator (if available)
    layer4_evidence = []
    root_cause = "rank_compute_jitter_causes_syncbn_barrier_wait"
    fix = "Stabilize compute on jittery rank(s) or reduce SyncBN frequency"

    if slow_rank_ops and slow_rank_ops.get("slow_rank_op_analysis_available"):
        primary_div = slow_rank_ops.get("primary_divergent_operator")
        syncbn_div = slow_rank_ops.get("syncbn_divergence_detected", False)
        top_ops = slow_rank_ops.get("top_divergent_operators", [])

        if syncbn_div:
            syncbn_ops = [op for op in top_ops if op.get("category") == "SyncBN"]
            if syncbn_ops:
                op = syncbn_ops[0]
                ratio = op.get("slowdown_ratio")
                layer4_evidence.append(
                    f"{op['operator']} is {ratio:.1f}x slower on Rank {worst_rank}"
                )
                root_cause = f"{op['operator']}_slow_on_rank_{worst_rank}"
                fix = (
                    f"Operator {op['operator']} on Rank {worst_rank} is {ratio:.1f}x slower. "
                    "Consider replacing SyncBN with GroupNorm, or investigate hardware issue on this rank."
                )
            else:
                layer4_evidence.append(f"syncbn_divergence_detected on Rank {worst_rank}")
        elif primary_div:
            layer4_evidence.append(f"primary_divergent_op={primary_div}")
            root_cause = f"{primary_div}_slow_on_rank_{worst_rank}"

    # Layer 5: wait attribution (if available)
    layer5_evidence = []
    if wait_attribution and wait_attribution.get("wait_time_attribution_available"):
        primary_source = wait_attribution.get("primary_wait_source")
        savings = wait_attribution.get("elimination_savings_ms", 0)
        if primary_source == "SyncBN":
            layer5_evidence.append(f"eliminating_SyncBN_wait_saves_{savings:.1f}ms/step")

    # Build chain
    chain_layers = [
        {"layer": 1, "node": "communication_overhead", "evidence": layer1_evidence[:3]},
        {"layer": 2, "node": "syncbn_sync_dominant", "evidence": layer2_evidence},
        {"layer": 3, "node": f"rank_{worst_rank}_compute_jitter", "evidence": layer3_evidence},
    ]
    if layer4_evidence:
        chain_layers.append({"layer": 4, "node": root_cause, "evidence": layer4_evidence})
    if layer5_evidence:
        chain_layers.append({"layer": 5, "node": "collective_barrier_wait", "evidence": layer5_evidence})

    # Confidence: based on evidence strength
    confidence = 0.70
    if syncbn_share > 30:
        confidence += 0.05
    if worst_cv > 0.20:
        confidence += 0.05
    if slow_rank_ops and slow_rank_ops.get("syncbn_divergence_detected"):
        confidence += 0.07
    if wait_attribution and wait_attribution.get("primary_wait_source") == "SyncBN":
        confidence += 0.05
    confidence = min(confidence, 0.95)

    chains.append({
        "symptom": "communication_overhead",
        "chain": chain_layers,
        "root_cause": root_cause,
        "fix": fix,
        "confidence": round(confidence, 3),
        "depth": len(chain_layers),
    })

    return chains


def _build_comm_overhead_chain(
    bottlenecks: dict,
    collective_types: Optional[dict],
    rank_variance: Optional[dict],
    communication: Optional[dict],
    cluster: Optional[dict],
    mfu: Optional[dict],
) -> list[dict]:
    """Build causal chains for generic communication overhead."""
    chains = []

    # Check if communication is a bottleneck
    comm_candidate = None
    for c in bottlenecks.get("ranked_candidates", []):
        if c.get("name") == "communication":
            comm_candidate = c
            break

    if comm_candidate is None:
        return chains

    comm_confidence = comm_candidate.get("confidence", 0.5)

    # Chain: comm_overhead -> excessive_allreduce -> small_bucket -> increase_buffer
    if collective_types and collective_types.get("collective_type_analysis_available"):
        types = collective_types.get("types", [])
        for type_info in types:
            if type_info["type"] == "SmallPacketAllReduce" and type_info["share_percent"] > 15:
                chains.append({
                    "symptom": "communication_overhead",
                    "chain": [
                        {"layer": 1, "node": "communication_overhead",
                         "evidence": [f"pressure={communication.get('communication_pressure', 'unknown')}" if communication else ""]},
                        {"layer": 2, "node": "excessive_small_packet_allreduce",
                         "evidence": [f"SmallPacketAllReduce_share={type_info['share_percent']:.1f}%"]},
                        {"layer": 3, "node": "small_bucket_size_or_zero3",
                         "evidence": [f"avg_size={type_info.get('avg_size_mb')}MB"]},
                    ],
                    "root_cause": "small_bucket_size_or_zero3",
                    "fix": "Increase gradient bucket size (HCCL_BUFFSIZE) or switch from ZeRO-3 to ZeRO-1/2",
                    "confidence": round(min(comm_confidence * 0.85, 0.90), 3),
                    "depth": 3,
                })
                break

        # Chain: comm_overhead -> allreduce_not_overlapped -> insufficient_compute_for_overlap
        for type_info in types:
            if type_info["type"] in ("GradientAllReduce", "AllReduce") and type_info["share_percent"] > 25:
                chains.append({
                    "symptom": "communication_overhead",
                    "chain": [
                        {"layer": 1, "node": "communication_overhead",
                         "evidence": [f"comm_confidence={comm_confidence:.2f}"]},
                        {"layer": 2, "node": "gradient_allreduce_not_overlapped",
                         "evidence": [f"{type_info['type']}_share={type_info['share_percent']:.1f}%"]},
                        {"layer": 3, "node": "insufficient_compute_for_overlap",
                         "evidence": []},
                    ],
                    "root_cause": "insufficient_compute_for_overlap",
                    "fix": "Increase micro batch size to create more computation for overlap with communication",
                    "confidence": round(min(comm_confidence * 0.80, 0.85), 3),
                    "depth": 3,
                })
                break

    return chains


def _build_rank_imbalance_chain(
    bottlenecks: dict,
    rank_variance: Optional[dict],
    cluster: Optional[dict],
    jitter: Optional[dict],
) -> list[dict]:
    """Build causal chains for rank imbalance."""
    chains = []

    rank_candidate = None
    for c in bottlenecks.get("ranked_candidates", []):
        if c.get("name") == "rank_imbalance":
            rank_candidate = c
            break

    if rank_candidate is None:
        return chains

    if not cluster or not cluster.get("slow_ranks"):
        return chains

    slow_rank = cluster["slow_ranks"][0]
    bt_type = cluster.get("analysis", {}).get("bottleneck_type", "general")

    chain_layers = [
        {"layer": 1, "node": "rank_imbalance",
         "evidence": [f"slow_ranks={cluster['slow_ranks']}", f"bottleneck_type={bt_type}"]},
    ]

    if bt_type == "host_dispatch":
        chain_layers.append({"layer": 2, "node": "host_dispatch_slow",
                             "evidence": [f"rank_{slow_rank}_free_time_high"]})
        chain_layers.append({"layer": 3, "node": "cpu_affinity_or_scheduling",
                             "evidence": []})
        fix = f"Set CPU affinity with numactl/taskset for Rank {slow_rank}"
        root_cause = "cpu_affinity_or_scheduling"
    elif bt_type == "compute":
        chain_layers.append({"layer": 2, "node": "compute_imbalance",
                             "evidence": [f"rank_{slow_rank}_compute_high"]})
        if rank_variance and slow_rank in rank_variance.get("jittery_ranks", []):
            chain_layers.append({"layer": 3, "node": "rank_compute_jitter",
                                 "evidence": [f"rank_{slow_rank}_cv={rank_variance.get('worst_rank_cv', 0):.3f}"]})
            root_cause = "rank_compute_jitter"
        else:
            chain_layers.append({"layer": 3, "node": "hardware_degradation",
                                 "evidence": []})
            root_cause = "hardware_degradation"
        fix = f"Check hardware health on Rank {slow_rank} or investigate compute jitter"
    elif bt_type == "communication":
        chain_layers.append({"layer": 2, "node": "communication_bottleneck",
                             "evidence": [f"rank_{slow_rank}_comm_high"]})
        chain_layers.append({"layer": 3, "node": "link_or_network_issue",
                             "evidence": []})
        root_cause = "link_or_network_issue"
        fix = "Check HCCS/RDMA link health between affected ranks"
    else:
        chain_layers.append({"layer": 2, "node": "general_slow_rank",
                             "evidence": []})
        root_cause = "general_slow_rank"
        fix = f"Investigate Rank {slow_rank} for hardware or scheduling issues"

    chains.append({
        "symptom": "rank_imbalance",
        "chain": chain_layers,
        "root_cause": root_cause,
        "fix": fix,
        "confidence": round(rank_candidate.get("confidence", 0.5) * 0.85, 3),
        "depth": len(chain_layers),
    })

    return chains


def _build_jitter_chain(
    bottlenecks: dict,
    jitter: Optional[dict],
    rank_variance: Optional[dict],
) -> list[dict]:
    """Build causal chains for jitter."""
    chains = []

    jitter_candidate = None
    for c in bottlenecks.get("ranked_candidates", []):
        if c.get("name") == "jitter":
            jitter_candidate = c
            break

    if jitter_candidate is None or not jitter:
        return chains

    step_cv = jitter.get("step_time_jitter", {}).get("cv")
    compute_cv = jitter.get("compute_jitter", {}).get("cv")
    comm_cv = jitter.get("communication_jitter", {}).get("cv")

    if step_cv is None:
        return chains

    chain_layers = [
        {"layer": 1, "node": "step_time_jitter",
         "evidence": [f"cv={step_cv:.4f}"]},
    ]

    root_cause = "unknown_jitter_source"
    fix = "Investigate jitter source"

    if compute_cv is not None and compute_cv > 0.10:
        chain_layers.append({"layer": 2, "node": "compute_jitter",
                             "evidence": [f"compute_cv={compute_cv:.4f}"]})
        # Check if rank-specific
        if rank_variance and rank_variance.get("jittery_ranks"):
            worst = rank_variance.get("worst_jittery_rank")
            chain_layers.append({"layer": 3, "node": f"rank_{worst}_specific_jitter",
                                 "evidence": [f"rank_{worst}_cv={rank_variance.get('worst_rank_cv', 0):.3f}"]})
            root_cause = f"rank_{worst}_specific_jitter"
            fix = f"Investigate compute instability on Rank {worst} (GC, scheduling, dynamic shapes)"
        else:
            chain_layers.append({"layer": 3, "node": "dynamic_shapes_or_gc",
                                 "evidence": []})
            root_cause = "dynamic_shapes_or_gc"
            fix = "Pad variable-length sequences, reduce Python object creation, enable CPU affinity"
    elif comm_cv is not None and comm_cv > 0.15:
        chain_layers.append({"layer": 2, "node": "communication_jitter",
                             "evidence": [f"comm_cv={comm_cv:.4f}"]})
        chain_layers.append({"layer": 3, "node": "network_or_sync_instability",
                             "evidence": []})
        root_cause = "network_or_sync_instability"
        fix = "Check HCCS/RDMA link stability and HCCL buffer configuration"

    chains.append({
        "symptom": "jitter",
        "chain": chain_layers,
        "root_cause": root_cause,
        "fix": fix,
        "confidence": round(jitter_candidate.get("confidence", 0.5) * 0.75, 3),
        "depth": len(chain_layers),
    })

    return chains


def _build_low_mfu_chain(
    bottlenecks: dict,
    mfu: Optional[dict],
    step: Optional[dict],
) -> list[dict]:
    """Build causal chain for low MFU."""
    chains = []

    mfu_candidate = None
    for c in bottlenecks.get("ranked_candidates", []):
        if c.get("name") == "low_mfu":
            mfu_candidate = c
            break

    if mfu_candidate is None or not mfu:
        return chains

    mfu_val = mfu.get("estimated_mfu")
    if mfu_val is None:
        return chains

    chain_layers = [
        {"layer": 1, "node": "low_mfu",
         "evidence": [f"mfu={mfu_val * 100:.1f}%"]},
    ]

    # Infer likely cause
    if step:
        dominant = step.get("dominant_stage", {})
        if dominant.get("name") in ("host_overhead", "idle_gap"):
            chain_layers.append({"layer": 2, "node": "host_overhead_dominant",
                                 "evidence": [f"host_share={dominant.get('share_percent', 0):.1f}%"]})
            chain_layers.append({"layer": 3, "node": "pynative_or_eager_mode",
                                 "evidence": []})
            root_cause = "pynative_or_eager_mode"
            fix = "Switch from PyNative/eager to graph execution mode (GRAPH_MODE or torch.compile)"
        else:
            chain_layers.append({"layer": 2, "node": "compute_underutilization",
                                 "evidence": []})
            root_cause = "compute_underutilization"
            fix = "Enable operator fusion and graph compilation to improve compute utilization"
    else:
        chain_layers.append({"layer": 2, "node": "compute_underutilization",
                             "evidence": []})
        root_cause = "compute_underutilization"
        fix = "Enable graph compilation, check for excessive small operators"

    chains.append({
        "symptom": "low_mfu",
        "chain": chain_layers,
        "root_cause": root_cause,
        "fix": fix,
        "confidence": round(mfu_candidate.get("confidence", 0.5) * 0.80, 3),
        "depth": len(chain_layers),
    })

    return chains


def build_causal_chains(
    bottlenecks: dict,
    profile: Optional[dict] = None,
    collective_types: Optional[dict] = None,
    rank_variance: Optional[dict] = None,
    slow_rank_ops: Optional[dict] = None,
    wait_attribution: Optional[dict] = None,
    step: Optional[dict] = None,
    communication: Optional[dict] = None,
    jitter: Optional[dict] = None,
    cluster: Optional[dict] = None,
    mfu: Optional[dict] = None,
) -> dict:
    """Build all causal chains from analysis artifacts."""
    ranked = bottlenecks.get("ranked_candidates", [])
    if not ranked:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "causal_chain_available": False,
            "reason": "No bottleneck candidates to analyze",
        }

    all_chains: list[dict] = []

    # Priority: SyncBN+jitter chain (deepest), then comm, rank, jitter, MFU
    all_chains.extend(
        _build_syncbn_jitter_chain(
            bottlenecks, collective_types, rank_variance,
            slow_rank_ops, wait_attribution, communication,
        )
    )
    all_chains.extend(
        _build_comm_overhead_chain(
            bottlenecks, collective_types, rank_variance,
            communication, cluster, mfu,
        )
    )
    all_chains.extend(
        _build_rank_imbalance_chain(bottlenecks, rank_variance, cluster, jitter)
    )
    all_chains.extend(
        _build_jitter_chain(bottlenecks, jitter, rank_variance)
    )
    all_chains.extend(
        _build_low_mfu_chain(bottlenecks, mfu, step)
    )

    # Sort by confidence (highest first), then by depth (deepest first)
    all_chains.sort(key=lambda c: (c["confidence"], c["depth"]), reverse=True)

    if not all_chains:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "causal_chain_available": False,
            "reason": "Insufficient data to build causal chains",
        }

    primary = all_chains[0]
    max_depth = max(c["depth"] for c in all_chains)

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "causal_chain_available": True,
        "chains": all_chains,
        "primary_chain": primary,
        "max_depth": max_depth,
        "chains_count": len(all_chains),
        "candidates_analyzed": len(ranked),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build dynamic multi-layer causal chains from analysis artifacts"
    )
    parser.add_argument("--bottlenecks-json", required=True, help="Bottleneck classification JSON")
    parser.add_argument("--profile-json", help="Performance profile JSON")
    parser.add_argument("--collective-types-json", help="Collective type analysis JSON")
    parser.add_argument("--rank-variance-json", help="Rank variance analysis JSON")
    parser.add_argument("--slow-rank-ops-json", help="Slow rank operator analysis JSON")
    parser.add_argument("--wait-attribution-json", help="Wait time attribution JSON")
    parser.add_argument("--step-json", help="Step breakdown JSON")
    parser.add_argument("--communication-json", help="Communication summary JSON")
    parser.add_argument("--jitter-json", help="Jitter analysis JSON")
    parser.add_argument("--cluster-json", help="Cluster analysis JSON")
    parser.add_argument("--mfu-json", help="MFU calculation JSON")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    try:
        bottlenecks = json.loads(Path(args.bottlenecks_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: Cannot read --bottlenecks-json: {exc}", file=sys.stderr)
        return 1

    result = build_causal_chains(
        bottlenecks,
        profile=load_optional_json(args.profile_json),
        collective_types=load_optional_json(args.collective_types_json),
        rank_variance=load_optional_json(args.rank_variance_json),
        slow_rank_ops=load_optional_json(args.slow_rank_ops_json),
        wait_attribution=load_optional_json(args.wait_attribution_json),
        step=load_optional_json(args.step_json),
        communication=load_optional_json(args.communication_json),
        jitter=load_optional_json(args.jitter_json),
        cluster=load_optional_json(args.cluster_json),
        mfu=load_optional_json(args.mfu_json),
    )

    write_json(Path(args.output_json), result)
    print(json.dumps({
        "causal_chain_available": result.get("causal_chain_available", False),
        "chains_count": result.get("chains_count", 0),
        "max_depth": result.get("max_depth", 0),
        "primary_root_cause": result.get("primary_chain", {}).get("root_cause"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
