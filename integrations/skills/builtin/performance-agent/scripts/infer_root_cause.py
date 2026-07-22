#!/usr/bin/env python3
"""Root cause inference engine for performance bottlenecks.

Builds causal chains from classified bottlenecks and profile data,
distinguishing root causes from symptoms and proposing targeted fixes.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from perf_common import load_optional_json, write_json


# Causal chain rules: (symptom_pattern, intermediate_causes, root_cause, fix_hint)
CAUSAL_RULES = [
    {
        "symptom": "low_mfu",
        "chains": [
            {
                "path": ["low_mfu", "excessive_small_ops", "missing_graph_compilation"],
                "evidence_keys": ["estimated_mfu", "operator_count", "graph_mode"],
                "fix": "Enable graph compilation (MindSpore GRAPH_MODE or torch.compile)",
            },
            {
                "path": ["low_mfu", "memory_bound", "poor_data_layout"],
                "evidence_keys": ["estimated_mfu", "cube_utilization", "l2_hit_rate"],
                "fix": "Optimize data layout and check if the model is memory-bound rather than compute-bound",
            },
            {
                "path": ["low_mfu", "kernel_launch_overhead", "pynative_execution"],
                "evidence_keys": ["estimated_mfu", "host_overhead_share"],
                "fix": "Switch from PyNative/eager to graph execution mode",
            },
        ],
    },
    {
        "symptom": "communication",
        "chains": [
            {
                "path": ["communication_overhead", "not_overlapped", "small_micro_batch"],
                "evidence_keys": ["comm_ratio", "overlap_ratio", "micro_batch_size"],
                "fix": "Increase micro batch size to create more computation for overlap with communication",
            },
            {
                "path": ["communication_overhead", "slow_card", "host_dispatch", "cpu_affinity"],
                "evidence_keys": ["comm_ratio", "wait_ratio_delta", "free_time_percent"],
                "fix": "Check CPU affinity (numactl binding) for the affected rank and ensure balanced NUMA allocation",
            },
            {
                "path": ["communication_overhead", "slow_card", "network_issue"],
                "evidence_keys": ["comm_ratio", "inter_node_bandwidth", "all_ranks_affected"],
                "fix": "Check HCCS/RDMA link health and network topology configuration",
            },
            {
                "path": ["communication_overhead", "excessive_collectives", "zero3_small_packets"],
                "evidence_keys": ["comm_ratio", "collective_count", "avg_collective_size"],
                "fix": "Increase bucket size for gradient all-reduce or consider ZeRO-1/2 instead of ZeRO-3",
            },
            {
                "path": ["communication_overhead", "hccl_misconfiguration", "buffer_size_too_small"],
                "evidence_keys": ["comm_ratio", "hccl_buffsize", "communication_bandwidth"],
                "fix": "Calculate and set HCCL_BUFFSIZE per LLM formula: ceil(MBS*S*H*dtype/8MB)",
            },
        ],
    },
    {
        "symptom": "host_framework_overhead",
        "chains": [
            {
                "path": ["host_overhead", "frequent_sync", "tensor_item_calls"],
                "evidence_keys": ["host_overhead_share", "sync_point_count"],
                "fix": "Remove tensor.item() calls from training loop; use NPU-side logic instead",
            },
            {
                "path": ["host_overhead", "python_interpreter_overhead", "pynative_dispatch"],
                "evidence_keys": ["host_overhead_share", "kernel_launch_density"],
                "fix": "Enable graph compilation to reduce per-op Python dispatch overhead",
            },
            {
                "path": ["host_overhead", "gc_pressure", "excessive_object_creation"],
                "evidence_keys": ["host_overhead_share", "gc_pause_detected"],
                "fix": "Reduce Python object creation in the training loop to minimize GC pauses",
            },
        ],
    },
    {
        "symptom": "input_pipeline",
        "chains": [
            {
                "path": ["input_bottleneck", "slow_decode", "cpu_bound_transform"],
                "evidence_keys": ["queue_empty_percent", "input_stage_time"],
                "fix": "Increase num_workers, enable pin_memory and prefetch, cache dataset",
            },
            {
                "path": ["input_bottleneck", "io_bound", "disk_read_slow"],
                "evidence_keys": ["queue_empty_percent", "disk_io_wait"],
                "fix": "Move dataset to SSD or memory, use cached dataset, reduce decode complexity",
            },
        ],
    },
    {
        "symptom": "rank_imbalance",
        "chains": [
            {
                "path": ["rank_imbalance", "slow_rank", "host_dispatch", "cpu_affinity"],
                "evidence_keys": ["slow_ranks", "free_time_percent"],
                "fix": "Set CPU affinity with numactl/taskset for the affected rank process",
            },
            {
                "path": ["rank_imbalance", "slow_rank", "compute_imbalance", "hardware_degradation"],
                "evidence_keys": ["slow_ranks", "compute_time_variance"],
                "fix": "Check hardware health on the affected card; run diagnostic tests",
            },
            {
                "path": ["rank_imbalance", "slow_rank", "communication_bottleneck", "link_degradation"],
                "evidence_keys": ["slow_ranks", "link_bandwidth"],
                "fix": "Check HCCS/RDMA link health between affected ranks",
            },
        ],
    },
    {
        "symptom": "jitter",
        "chains": [
            {
                "path": ["jitter", "dynamic_shapes", "variable_length_input"],
                "evidence_keys": ["step_time_cv", "shape_variance"],
                "fix": "Pad variable-length sequences to fixed sizes to eliminate recompilation",
            },
            {
                "path": ["jitter", "gc_pause", "excessive_object_creation"],
                "evidence_keys": ["step_time_cv", "gc_pause_detected"],
                "fix": "Reduce Python object creation in the training loop to minimize GC pauses",
            },
            {
                "path": ["jitter", "os_scheduling", "no_cpu_affinity"],
                "evidence_keys": ["step_time_cv", "cpu_migration_detected"],
                "fix": "Enable CPU affinity (numactl/taskset) to prevent OS scheduling variance",
            },
        ],
    },
    {
        "symptom": "memory",
        "chains": [
            {
                "path": ["memory_pressure", "activation_memory_high", "no_recomputation"],
                "evidence_keys": ["peak_memory_mb", "activation_memory_share"],
                "fix": "Enable gradient checkpointing / activation recomputation to trade compute for memory",
            },
            {
                "path": ["memory_pressure", "fragmentation", "many_small_allocations"],
                "evidence_keys": ["peak_memory_mb", "fragmentation_ratio"],
                "fix": "Configure PYTORCH_NPU_ALLOC_CONF with max_split_size_mb and expandable_segments",
            },
            {
                "path": ["memory_pressure", "gradient_accumulation_overhead", "large_gradient_buffers"],
                "evidence_keys": ["peak_memory_mb", "gradient_size"],
                "fix": "Reduce gradient accumulation steps or enable ZeRO for gradient sharding",
            },
        ],
    },
    {
        "symptom": "operator_hotspot",
        "chains": [
            {
                "path": ["operator_hotspot", "missing_fusion", "unfused_attention"],
                "evidence_keys": ["top_operator_share", "operator_name"],
                "fix": "Replace attention with npu_fusion_attention (FlashAttention for Ascend)",
            },
            {
                "path": ["operator_hotspot", "suboptimal_kernel", "fallback_implementation"],
                "evidence_keys": ["top_operator_share", "kernel_path"],
                "fix": "Check if backend kernel path is optimal for Ascend; consider custom operator",
            },
        ],
    },
]


def _match_symptom(bottleneck_name: str) -> Optional[dict]:
    """Match a bottleneck name to a causal rule set."""
    name_lower = bottleneck_name.lower()
    mapping = {
        "low_mfu": "low_mfu",
        "communication": "communication",
        "host_framework_overhead": "host_framework_overhead",
        "input_pipeline": "input_pipeline",
        "rank_imbalance": "rank_imbalance",
        "jitter": "jitter",
        "memory": "memory",
        "operator_hotspot": "operator_hotspot",
    }
    for key, symptom in mapping.items():
        if key in name_lower:
            for rule in CAUSAL_RULES:
                if rule["symptom"] == symptom:
                    return rule
    return None


def _collect_evidence(
    chain: dict,
    bottlenecks: dict,
    profile: Optional[dict],
    step: Optional[dict],
    comm: Optional[dict],
    jitter: Optional[dict],
    cluster: Optional[dict],
    mfu: Optional[dict],
) -> list[str]:
    """Collect supporting evidence for a causal chain."""
    evidence = []
    evidence_keys = chain.get("evidence_keys", [])

    # From bottlenecks
    primary = bottlenecks.get("primary_candidate", {})
    for ev in primary.get("evidence", []):
        evidence.append(ev)

    # From MFU
    if mfu and "estimated_mfu" in evidence_keys:
        mfu_val = mfu.get("estimated_mfu")
        if mfu_val is not None:
            evidence.append(f"mfu={mfu_val:.2f}")

    # From communication
    if comm:
        if "comm_ratio" in evidence_keys:
            total_time = comm.get("total_time_ms")
            if total_time is not None:
                evidence.append(f"comm_total={total_time}ms")
        if "collective_count" in evidence_keys:
            count = comm.get("collective_count")
            if count is not None:
                evidence.append(f"collective_count={count}")

    # From jitter
    if jitter and "step_time_cv" in evidence_keys:
        step_jitter = jitter.get("step_time_jitter", {})
        cv = step_jitter.get("cv")
        if cv is not None:
            evidence.append(f"step_time_cv={cv:.3f}")

    # From cluster
    if cluster:
        if "slow_ranks" in evidence_keys:
            slow = cluster.get("slow_ranks", [])
            if slow:
                evidence.append(f"slow_ranks={slow}")

    # From step
    if step and "host_overhead_share" in evidence_keys:
        dominant = step.get("dominant_stage", {})
        if dominant.get("name") in ("host_overhead", "idle_gap"):
            evidence.append(f"host_overhead_share={dominant.get('share_percent', 0):.1f}%")

    return evidence[:5]  # Limit evidence items


def infer_root_causes(
    bottlenecks: dict,
    profile: Optional[dict] = None,
    step: Optional[dict] = None,
    comm: Optional[dict] = None,
    jitter: Optional[dict] = None,
    cluster: Optional[dict] = None,
    mfu: Optional[dict] = None,
) -> dict:
    """Infer root causes from classified bottlenecks."""
    ranked = bottlenecks.get("ranked_candidates", [])
    if not ranked:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "root_cause_inference_available": False,
            "reason": "No bottleneck candidates to analyze",
        }

    root_causes = []

    for candidate in ranked[:5]:  # Top 5 candidates
        name = candidate.get("name", "")
        confidence = candidate.get("confidence", 0)
        rule = _match_symptom(name)

        if not rule:
            continue

        for chain in rule["chains"]:
            evidence = _collect_evidence(chain, bottlenecks, profile, step, comm, jitter, cluster, mfu)
            path = chain["path"]

            # Skip chains with no supporting evidence
            if not evidence:
                continue

            # Scale confidence by evidence strength (more evidence → higher confidence)
            evidence_factor = min(1.0, len(evidence) / 3.0)
            chain_confidence = confidence * 0.6 * (0.5 + 0.5 * evidence_factor)

            root_causes.append({
                "symptom": name,
                "causal_chain": " <- ".join(path),
                "root_cause": path[-1],
                "depth": len(path) - 1,
                "confidence": round(chain_confidence, 3),
                "evidence": evidence,
                "fix": chain["fix"],
            })

    # Sort by confidence
    root_causes.sort(key=lambda r: r["confidence"], reverse=True)

    if not root_causes:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "root_cause_inference_available": False,
        }

    primary = root_causes[0]

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "root_cause_inference_available": True,
        "root_causes": root_causes,
        "primary_root_cause": primary,
        "chain_depth_max": max((r["depth"] for r in root_causes), default=0),
        "candidates_analyzed": min(len(ranked), 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Infer root causes from classified bottleneck analysis"
    )
    parser.add_argument("--bottlenecks-json", required=True, help="Bottleneck classification JSON")
    parser.add_argument("--profile-json", help="Performance profile JSON")
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
    profile = load_optional_json(args.profile_json)
    step = load_optional_json(args.step_json)
    comm = load_optional_json(args.communication_json)
    jitter = load_optional_json(args.jitter_json)
    cluster = load_optional_json(args.cluster_json)
    mfu = load_optional_json(args.mfu_json)

    result = infer_root_causes(bottlenecks, profile, step, comm, jitter, cluster, mfu)

    write_json(Path(args.output_json), result)
    print(json.dumps({
        "inference_available": result.get("root_cause_inference_available", False),
        "root_causes_count": len(result.get("root_causes", [])),
        "primary_root_cause": (
            result["primary_root_cause"]["root_cause"]
            if result.get("primary_root_cause")
            else None
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
