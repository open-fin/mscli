#!/usr/bin/env python3
"""Build actionable optimization suggestions from performance analysis results.

Matches bottleneck candidates and metric thresholds against the optimization
knowledge base to generate prioritized, concrete suggestions with code/config
examples.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from perf_common import load_optional_json, write_json


# ---------------------------------------------------------------------------
# Rule definitions (mirrors optimization-knowledge-base.md)
# ---------------------------------------------------------------------------

def _comm_rules(
    comm_json: Optional[dict],
    bottleneck: Optional[dict],
    step_json: Optional[dict] = None,
) -> list[dict]:
    """Generate communication optimization suggestions."""
    suggestions = []

    if not comm_json:
        return suggestions

    comm_ratio = None
    total_time_ms = comm_json.get("total_time_ms", 0)
    pressure = comm_json.get("communication_pressure", "low")

    # Calculate comm_ratio from total communication time vs sum of all collective shares
    top_collectives = comm_json.get("top_collectives", [])
    if top_collectives:
        total_share = sum(
            float(c.get("share_percent", 0)) for c in top_collectives
        )
        comm_ratio = total_share / 100.0
    elif total_time_ms and total_time_ms > 0 and step_json:
        # Fallback: ratio of comm total_time to step average time
        step_time = step_json.get("average_step_time_ms", 0)
        if step_time and step_time > 0:
            comm_ratio = total_time_ms / step_time

    # COMM-01: Communication overhead too high
    if comm_ratio and comm_ratio > 0.30:
        suggestions.append({
            "id": "COMM-01",
            "title": "Communication overhead is too high (>30%)",
            "priority": "high",
            "category": "communication",
            "expected_benefit": "10-30% training speed improvement",
            "trigger_metric": f"comm_ratio={comm_ratio:.2f}",
            "actions": [
                "Check if TP/DP/PP ratio is balanced for the workload",
                "Enable gradient accumulation to reduce DP communication frequency",
                "Enable communication-computation overlap",
            ],
            "config_examples": {
                "megatron": "--overlap-grad-reduce --overlap-param-gather --gradient-accumulation-steps 4",
                "mindspore": "use_parallel_optimizer: true\noverlap_grad_reduce: true\ngradient_accumulation_steps: 4",
            },
            "validation_metrics": ["comm_ratio", "collective_count", "step_tail_ms"],
        })

    # COMM-03: Excessive small collectives
    collective_count = comm_json.get("collective_count", 0)
    if collective_count > 100:
        suggestions.append({
            "id": "COMM-03",
            "title": "Excessive small collective operations detected",
            "priority": "medium",
            "category": "communication",
            "expected_benefit": "5-15% training speed improvement",
            "trigger_metric": f"collective_count={collective_count}",
            "actions": [
                "Increase bucket size for gradient all-reduce",
                "Check if ZeRO-3 is causing excessive small-packet communication",
                "Consider switching to ZeRO-1 or ZeRO-2 for small models",
            ],
            "validation_metrics": ["collective_count", "avg_collective_size_mb"],
        })

    return suggestions


def _compute_rules(
    step_json: Optional[dict],
    hotspot_json: Optional[dict],
    mfu_json: Optional[dict],
    bottleneck: Optional[dict],
) -> list[dict]:
    """Generate compute optimization suggestions."""
    suggestions = []

    # COMP-01: Compute time ratio too low
    if step_json:
        stage_totals = step_json.get("stage_totals_ms", {})
        total_stage = sum(v for k, v in stage_totals.items() if k != "step_total")
        compute_time = stage_totals.get("compute", 0)
        if total_stage > 0:
            compute_ratio = compute_time / total_stage
            if compute_ratio < 0.50:
                suggestions.append({
                    "id": "COMP-01",
                    "title": "Compute time ratio is too low (<50%)",
                    "priority": "high",
                    "category": "compute",
                    "expected_benefit": "20-40% training speed improvement",
                    "trigger_metric": f"compute_ratio={compute_ratio:.2f}",
                    "actions": [
                        "Identify what is consuming non-compute time (comm, idle, data loading)",
                        "Address the dominant non-compute bottleneck first",
                        "Increase batch size to amortize fixed overhead",
                    ],
                    "validation_metrics": ["compute_ratio", "step_time_ms"],
                })

    # COMP-02: MFU below 20%
    if mfu_json and mfu_json.get("estimated_mfu") is not None:
        mfu = mfu_json["estimated_mfu"]
        if mfu < 0.20:
            suggestions.append({
                "id": "COMP-02",
                "title": f"MFU is very low ({mfu*100:.1f}%), hardware severely underutilized",
                "priority": "high",
                "category": "compute",
                "expected_benefit": "Significant improvement possible",
                "trigger_metric": f"estimated_mfu={mfu:.3f}",
                "actions": [
                    "Enable graph compilation (MindSpore GRAPH_MODE or torch.compile)",
                    "Check for excessive small operators causing launch overhead",
                    "Check if the model is memory-bound rather than compute-bound",
                ],
                "code_examples": {
                    "mindspore": "ms.set_context(mode=ms.GRAPH_MODE)",
                    "pytorch": "model = torch.compile(model)",
                },
                "validation_metrics": ["estimated_mfu", "step_time_ms"],
            })

    # COMP-03: Operator hotspot
    if hotspot_json and hotspot_json.get("top_operators"):
        top_op = hotspot_json["top_operators"][0]
        top_share = float(top_op.get("share_percent", 0))
        if top_share > 35:
            suggestions.append({
                "id": "COMP-03",
                "title": f"Single operator dominates: {top_op['operator']} ({top_share:.1f}% of step time)",
                "priority": "high",
                "category": "compute",
                "expected_benefit": "10-30% step time reduction",
                "trigger_metric": f"top_op_share={top_share:.1f}%",
                "actions": [
                    "Check if a fused variant exists (FlashAttention, FusedLayerNorm, etc.)",
                    "Check if backend kernel path is optimal for Ascend",
                    "Consider custom operator implementation",
                ],
                "validation_metrics": ["top_operator_share", "step_time_ms"],
            })

    return suggestions


def _memory_rules(
    memory_json: Optional[dict],
    bottleneck: Optional[dict],
) -> list[dict]:
    """Generate memory optimization suggestions."""
    suggestions = []

    if not memory_json:
        return suggestions

    pressure = memory_json.get("memory_pressure", "low")
    peak_mb = memory_json.get("peak_memory_mb")

    if pressure in ("high", "moderate") or (peak_mb and peak_mb > 57600):  # >90% of 64GB
        suggestions.append({
            "id": "MEM-01",
            "title": "Memory pressure is high, limiting batch size or causing risk",
            "priority": "high" if pressure == "high" else "medium",
            "category": "memory",
            "expected_benefit": "Enable larger batch sizes, improve throughput",
            "trigger_metric": f"memory_pressure={pressure}, peak_memory={peak_mb}MB",
            "actions": [
                "Enable gradient checkpointing / activation recomputation",
                "Switch from FP32 to BF16 or FP16 if not already done",
                "Check for memory leaks (growing peak memory across steps)",
            ],
            "code_examples": {
                "mindspore": "from mindspore import nn\nnet = nn.Recompute(net)",
                "pytorch": "from torch.utils.checkpoint import checkpoint\noutput = checkpoint(module, *inputs)",
            },
            "validation_metrics": ["peak_memory_mb", "memory_pressure"],
        })

    return suggestions


def _input_rules(
    input_json: Optional[dict],
    bottleneck: Optional[dict],
) -> list[dict]:
    """Generate input pipeline optimization suggestions."""
    suggestions = []

    if not input_json:
        return suggestions

    if input_json.get("bottleneck_detected"):
        queue_empty = input_json.get("queue_empty_percent")
        suggestions.append({
            "id": "INPUT-01",
            "title": "Data loading is a bottleneck" + (f" (queue empty {queue_empty:.0f}%)" if queue_empty else ""),
            "priority": "high",
            "category": "input_pipeline",
            "expected_benefit": "30-80% idle time reduction",
            "trigger_metric": f"bottleneck_detected=true, queue_empty={queue_empty}%",
            "actions": [
                "Increase DataLoader num_workers",
                "Enable pin_memory and prefetch",
                "Cache dataset to memory or SSD",
                "Reduce decode/transform complexity",
            ],
            "code_examples": {
                "mindspore": "dataset = dataset.batch(batch_size, num_parallel_workers=8, drop_remainder=True)",
                "pytorch": "DataLoader(dataset, batch_size=batch_size, num_workers=8, pin_memory=True, prefetch_factor=2)",
            },
            "validation_metrics": ["queue_empty_percent", "pre_compute_idle_ms", "throughput"],
        })

    return suggestions


def _host_rules(
    trace_gaps_json: Optional[dict],
    step_json: Optional[dict],
    bottleneck: Optional[dict],
) -> list[dict]:
    """Generate host/framework overhead optimization suggestions."""
    suggestions = []

    # HOST-01: Excessive host launch overhead
    host_share = 0.0
    if trace_gaps_json and trace_gaps_json.get("dominant_category"):
        dominant = trace_gaps_json["dominant_category"]
        if dominant.get("name") in ("host_overhead", "idle_gap"):
            host_share = dominant.get("share_percent", 0)

    if step_json and step_json.get("dominant_stage"):
        dominant = step_json["dominant_stage"]
        if dominant.get("name") in ("host_overhead", "idle_gap"):
            host_share = max(host_share, dominant.get("share_percent", 0))

    if host_share > 20:
        suggestions.append({
            "id": "HOST-01",
            "title": f"Host/framework overhead is too high ({host_share:.1f}% of step time)",
            "priority": "high",
            "category": "host_framework",
            "expected_benefit": "20-50% latency reduction",
            "trigger_metric": f"host_overhead_share={host_share:.1f}%",
            "actions": [
                "Enable graph compilation (GRAPH_MODE / torch.compile)",
                "Reduce Python-side per-step overhead",
                "Check for unnecessary sync points",
            ],
            "code_examples": {
                "mindspore": "ms.set_context(mode=ms.GRAPH_MODE)",
                "pytorch": "model = torch.compile(model, mode='max-autotune')",
            },
            "validation_metrics": ["host_overhead_share", "kernel_launch_density"],
        })

    # HOST-02: Graph recompilation
    if step_json and step_json.get("dominant_stage"):
        dominant = step_json["dominant_stage"]
        if dominant.get("name") == "graph_compile" and dominant.get("share_percent", 0) > 15:
            suggestions.append({
                "id": "HOST-02",
                "title": "Graph recompilation is consuming significant time",
                "priority": "medium",
                "category": "host_framework",
                "expected_benefit": "Eliminate repeated compile cost",
                "trigger_metric": f"graph_compile_share={dominant['share_percent']:.1f}%",
                "actions": [
                    "Stabilize input shapes (pad to fixed sizes)",
                    "Avoid dynamic control flow in model",
                    "Separate warmup compile cost from steady-state measurement",
                ],
                "validation_metrics": ["compile_count", "compile_time_ms"],
            })

    return suggestions


def _cluster_rules(
    cluster_json: Optional[dict],
    bottleneck: Optional[dict],
) -> list[dict]:
    """Generate cluster optimization suggestions."""
    suggestions = []

    if not cluster_json or not cluster_json.get("cluster_analysis_available"):
        return suggestions

    slow_ranks = cluster_json.get("slow_ranks", [])
    if slow_ranks:
        analysis = cluster_json.get("analysis", {})
        bt_type = analysis.get("bottleneck_type", "general")
        diagnosis = analysis.get("diagnosis", "")

        actions_map = {
            "host_dispatch": [
                "Check CPU affinity (numactl binding) for the affected rank",
                "Check for CPU-bound processes competing for the same NUMA node",
                "Compare API dispatch stats between slow and fast ranks",
            ],
            "compute": [
                "Compare operator stats between slow and fast ranks",
                "Check for dynamic shapes causing recompilation",
                "Check hardware health on the affected card",
            ],
            "communication": [
                "Check HCCS/RDMA link health",
                "Verify network topology and switch configuration",
                "Check for slow inter-node links",
            ],
        }

        suggestions.append({
            "id": "CLUSTER-01",
            "title": f"Slow card detected: Rank {slow_ranks[0]} ({bt_type})",
            "priority": "high",
            "category": "cluster",
            "expected_benefit": "10-30% cluster speedup",
            "trigger_metric": f"slow_ranks={slow_ranks}, bottleneck_type={bt_type}",
            "actions": actions_map.get(bt_type, [
                "Compare full step breakdown between slow and fast ranks",
                "Check for OS-level scheduling differences or hardware issues",
            ]),
            "validation_metrics": ["per_rank_step_times", "rank_imbalance_ratio"],
        })

    return suggestions


def _jitter_rules(
    jitter_json: Optional[dict],
) -> list[dict]:
    """Generate jitter optimization suggestions."""
    suggestions = []

    if not jitter_json:
        return suggestions

    step_jitter = jitter_json.get("step_time_jitter", {})
    cv = step_jitter.get("cv")
    status = step_jitter.get("status")

    if cv and cv > 0.15:
        suggestions.append({
            "id": "JITTER-01",
            "title": f"Step time variance is too high (CV={cv*100:.1f}%)",
            "priority": "medium",
            "category": "jitter",
            "expected_benefit": "More predictable and consistent throughput",
            "trigger_metric": f"step_time_cv={cv:.4f}",
            "actions": [
                "Pad variable-length sequences to fixed sizes",
                "Enable CPU affinity (numactl / taskset)",
                "Reduce Python object creation in training loop to avoid GC pauses",
                "Disable CPU frequency scaling (set to performance governor)",
            ],
            "validation_metrics": ["step_time_cv", "outlier_count"],
        })

    return suggestions


def _fusion_rules(
    fusion_json: Optional[dict],
) -> list[dict]:
    """Generate operator fusion optimization suggestions."""
    suggestions = []

    if not fusion_json or not fusion_json.get("fusion_analysis_available"):
        return suggestions

    for opp in fusion_json.get("opportunities", []):
        fusion_type = opp.get("type", "")
        share = opp.get("combined_share_percent", 0)
        priority = "high" if share >= 20 else "medium"

        code_examples = {}
        if fusion_type == "flash_attention":
            code_examples["pytorch"] = (
                "attn_output = torch_npu.npu_fusion_attention(\n"
                "    q, k, v, head_num=num_heads,\n"
                "    input_layout='BNSD', keep_prob=1.0,\n"
                "    scale=1.0/math.sqrt(head_dim)\n"
                ")"
            )
        elif fusion_type == "matmul_allreduce":
            code_examples["pytorch"] = (
                "output = torch_npu.npu_mm_all_reduce_base(\n"
                "    input, weight, hcomm_info,\n"
                "    reduce_op='sum', comm_turn=0\n"
                ")"
            )
        elif fusion_type == "fused_optimizer":
            fused_name = opp.get("fused_replacement", "NpuFusedAdamW")
            code_examples["pytorch"] = (
                f"optimizer = torch_npu.optim.{fused_name}(model.parameters(), lr=1e-4)"
            )

        suggestions.append({
            "id": opp.get("suggestion_id", "FUSION-01"),
            "title": f"Operator fusion opportunity: {fusion_type} ({share:.1f}% of compute)",
            "priority": priority,
            "category": "fusion",
            "expected_benefit": opp.get("estimated_speedup", "5-15%"),
            "trigger_metric": f"fusion_type={fusion_type}, combined_share={share:.1f}%",
            "actions": [
                f"Replace with {opp.get('replacement_api', 'fused variant')}",
                f"Constraint: {opp.get('constraint', 'check compatibility')}",
            ],
            "code_examples": code_examples if code_examples else None,
            "validation_metrics": ["operator_time_share", "step_time_ms"],
        })

    return suggestions


def _degradation_rules(
    degradation_json: Optional[dict],
) -> list[dict]:
    """Generate cluster degradation-specific suggestions."""
    suggestions = []

    if not degradation_json or not degradation_json.get("degradation_classification_available"):
        return suggestions

    primary_type = degradation_json.get("primary_type")
    sub = degradation_json.get("sub_classification", {})
    confidence = sub.get("confidence", 0.5)
    actions = degradation_json.get("recommended_actions", [])

    if not primary_type:
        return suggestions

    priority = "high" if confidence >= 0.75 else "medium"

    degradation_titles = {
        "scale_up": "Scale-up degradation detected: model sharding strategy likely suboptimal",
        "hardware_change": "Hardware change degradation: component-level regression detected",
        "long_term_training": "Long-term training degradation: possible memory leak or thermal throttling",
        "performance_fluctuation": "Performance fluctuation: intermittent resource contention",
        "slow_node": "Slow node detected: asymmetric performance across nodes",
        "network_problem": "Network problem: all cards affected by communication issues",
    }

    suggestions.append({
        "id": f"DEGRAD-{primary_type.replace('_', '-').upper()}",
        "title": degradation_titles.get(primary_type, f"Cluster degradation: {primary_type}"),
        "priority": priority,
        "category": "cluster_degradation",
        "expected_benefit": "10-30% cluster performance recovery",
        "trigger_metric": f"degradation_type={primary_type}, confidence={confidence:.2f}",
        "actions": actions[:4],
        "validation_metrics": ["step_time_ms", "linearity", "comm_ratio"],
    })

    return suggestions


def _affinity_rules(
    affinity_json: Optional[dict],
) -> list[dict]:
    """Generate NPU affinity optimization suggestions."""
    suggestions = []

    if not affinity_json or not affinity_json.get("npu_affinity_analysis_available"):
        return suggestions

    steps = affinity_json.get("steps", [])
    overall_score = affinity_json.get("overall_affinity_score", 1.0)

    for step in steps:
        step_name = step.get("name", "")
        step_score = step.get("score", 1.0)
        if step_score >= 0.8:
            continue

        step_suggestions = step.get("suggestions", [])
        if not step_suggestions:
            continue

        first_sug = step_suggestions[0]
        priority = "high" if step_score < 0.5 else "medium"

        suggestions.append({
            "id": first_sug.get("id", f"NPU-AFFINITY-{step_name or step.get('step', 0)}"),
            "title": f"NPU affinity gap in {step_name} (score: {step_score:.2f})",
            "priority": priority,
            "category": "npu_affinity",
            "expected_benefit": "10-30% step time reduction",
            "trigger_metric": f"affinity_step={step_name}, score={step_score:.2f}",
            "actions": [
                f"Address {step_name} optimization: {first_sug.get('type', 'unknown')}",
                first_sug.get("code_example", "See optimization-knowledge-base.md"),
            ],
            "validation_metrics": ["overall_affinity_score", "step_time_ms"],
        })

    return suggestions


def _syncbn_rules(
    collective_types_json: Optional[dict],
    rank_variance_json: Optional[dict],
    wait_attribution_json: Optional[dict],
) -> list[dict]:
    """Generate SyncBN-specific optimization suggestions."""
    suggestions = []

    if not collective_types_json or not collective_types_json.get("collective_type_analysis_available"):
        return suggestions

    syncbn_share = collective_types_json.get("syncbn_share_percent", 0)
    syncbn_dominant = collective_types_json.get("syncbn_dominant", False)

    if syncbn_share < 10:
        return suggestions

    # Estimate wait impact
    wait_ms = 0.0
    if wait_attribution_json and wait_attribution_json.get("wait_time_attribution_available"):
        for attr in wait_attribution_json.get("attributions", []):
            if attr.get("collective_type") == "SyncBN":
                wait_ms = attr.get("estimated_wait_ms", 0)
                break

    priority = "high" if syncbn_dominant else "medium"

    evidence = [f"SyncBN collective time share: {syncbn_share:.1f}%"]
    if rank_variance_json and rank_variance_json.get("jittery_ranks"):
        jittery = rank_variance_json["jittery_ranks"]
        rank_str = ", ".join(str(r) for r in jittery[:3])
        evidence.append(f"Jittery ranks [{rank_str}] amplify SyncBN barrier wait")
    if wait_ms > 0:
        evidence.append(f"Estimated SyncBN wait: {wait_ms:.1f}ms/step")

    suggestions.append({
        "id": "SYNCBN-01",
        "title": "Replace SyncBN with GroupNorm or FrozenBN",
        "priority": priority,
        "category": "communication",
        "evidence": evidence,
        "action": (
            "SyncBN requires cross-rank synchronization at every training step, "
            "creating a barrier that amplifies any rank compute jitter. "
            "Replace with GroupNorm (groups=32) or FrozenBN to eliminate synchronization."
        ),
        "config": (
            "# Replace SyncBatchNorm with GroupNorm\n"
            "import torch.nn as nn\n"
            "nn.SyncBatchNorm -> nn.GroupNorm(num_groups=32, num_channels=C)\n\n"
            "# Or use FrozenBN (freeze BN statistics after initial training)\n"
            "model.apply(lambda m: setattr(m, 'track_running_stats', False) "
            "if isinstance(m, nn.BatchNorm2d) else None)"
        ),
    })

    if syncbn_share > 25:
        suggestions.append({
            "id": "SYNCBN-02",
            "title": "Reduce SyncBN synchronization frequency",
            "priority": "medium",
            "category": "communication",
            "evidence": evidence,
            "action": (
                "If SyncBN cannot be replaced, reduce synchronization frequency: "
                "sync every N steps instead of every step, or use delayed synchronization."
            ),
            "config": (
                "# Sync every N steps instead of every step\n"
                "sync_every_n_steps = 10  # adjust based on training stability"
            ),
        })

    return suggestions


def _rank_jitter_rules(
    rank_variance_json: Optional[dict],
    wait_attribution_json: Optional[dict],
) -> list[dict]:
    """Generate rank jitter optimization suggestions."""
    suggestions = []

    if not rank_variance_json or not rank_variance_json.get("rank_variance_analysis_available"):
        return suggestions

    jittery_ranks = rank_variance_json.get("jittery_ranks", [])
    if not jittery_ranks:
        return suggestions

    worst_rank = rank_variance_json.get("worst_jittery_rank")
    worst_cv = rank_variance_json.get("worst_rank_cv", 0)
    drag = rank_variance_json.get("drag_effect_ms", 0)

    priority = "high" if worst_cv > 0.20 else "medium"

    evidence = [
        f"Jittery ranks: {jittery_ranks}",
        f"Rank {worst_rank} CV: {worst_cv:.3f}",
    ]
    if drag > 0:
        evidence.append(f"Drag effect: {drag:.1f}ms per step")

    suggestions.append({
        "id": "JITTER-01",
        "title": f"Stabilize compute on jittery Rank {worst_rank}",
        "priority": priority,
        "category": "jitter",
        "evidence": evidence,
        "action": (
            f"Rank {worst_rank} has high compute variance (CV={worst_cv:.3f}), "
            "which creates barrier wait for all other ranks. "
            "Investigate: dynamic shapes, GC pauses, CPU scheduling, or hardware issues."
        ),
        "config": (
            "# Enable CPU affinity to prevent OS scheduling variance\n"
            "numactl --cpunodebind=0 --membind=0 python train.py\n\n"
            "# Reduce GC pressure in training loop\n"
            "import gc\n"
            "gc.disable()  # in training loop\n"
            "# ... training step ...\n"
            "gc.enable()\n\n"
            "# Pad sequences to fixed sizes to eliminate recompilation"
        ),
    })

    if wait_attribution_json and wait_attribution_json.get("wait_time_attribution_available"):
        primary_source = wait_attribution_json.get("primary_wait_source")
        savings = wait_attribution_json.get("elimination_savings_ms", 0)
        if primary_source and savings > 0:
            suggestions.append({
                "id": "JITTER-02",
                "title": f"Eliminate {primary_source} barrier wait caused by rank jitter",
                "priority": "medium",
                "category": "jitter",
                "evidence": [
                    f"Primary wait source: {primary_source}",
                    f"Estimated savings: {savings:.1f}ms/step",
                ],
                "action": (
                    f"Rank jitter causes {primary_source} barrier waits totaling "
                    f"{savings:.1f}ms per step. Fixing the jitter source will eliminate this wait."
                ),
                "config": "",
            })

    return suggestions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_suggestions(
    profile: dict,
    bottlenecks: dict,
    step_json: Optional[dict],
    comm_json: Optional[dict],
    memory_json: Optional[dict],
    input_json: Optional[dict],
    trace_gaps_json: Optional[dict],
    hotspot_json: Optional[dict],
    mfu_json: Optional[dict],
    cluster_json: Optional[dict],
    jitter_json: Optional[dict],
    fusion_json: Optional[dict] = None,
    degradation_json: Optional[dict] = None,
    affinity_json: Optional[dict] = None,
    collective_types_json: Optional[dict] = None,
    rank_variance_json: Optional[dict] = None,
    wait_attribution_json: Optional[dict] = None,
) -> list[dict]:
    """Build all optimization suggestions."""
    primary = bottlenecks.get("primary_candidate", {})
    all_suggestions = []

    all_suggestions.extend(_comm_rules(comm_json, primary, step_json))
    all_suggestions.extend(_compute_rules(step_json, hotspot_json, mfu_json, primary))
    all_suggestions.extend(_memory_rules(memory_json, primary))
    all_suggestions.extend(_input_rules(input_json, primary))
    all_suggestions.extend(_host_rules(trace_gaps_json, step_json, primary))
    all_suggestions.extend(_cluster_rules(cluster_json, primary))
    all_suggestions.extend(_jitter_rules(jitter_json))
    all_suggestions.extend(_fusion_rules(fusion_json))
    all_suggestions.extend(_degradation_rules(degradation_json))
    all_suggestions.extend(_affinity_rules(affinity_json))
    all_suggestions.extend(_syncbn_rules(collective_types_json, rank_variance_json, wait_attribution_json))
    all_suggestions.extend(_rank_jitter_rules(rank_variance_json, wait_attribution_json))

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    all_suggestions.sort(key=lambda s: priority_order.get(s.get("priority", "low"), 99))

    return all_suggestions


def main() -> int:
    parser = argparse.ArgumentParser(description="Build optimization suggestions from analysis results")
    parser.add_argument("--profile-json", required=True, help="Performance profile JSON")
    parser.add_argument("--bottlenecks-json", required=True, help="Bottleneck classification JSON")
    parser.add_argument("--step-json", help="Step summary JSON")
    parser.add_argument("--communication-json", help="Communication summary JSON")
    parser.add_argument("--memory-json", help="Memory summary JSON")
    parser.add_argument("--input-json", help="Input pipeline summary JSON")
    parser.add_argument("--trace-gaps-json", help="Trace gaps summary JSON")
    parser.add_argument("--hotspot-json", help="Hotspot summary JSON")
    parser.add_argument("--mfu-json", help="MFU calculation JSON")
    parser.add_argument("--cluster-json", help="Cluster analysis JSON")
    parser.add_argument("--jitter-json", help="Jitter analysis JSON")
    parser.add_argument("--fusion-json", help="Operator fusion analysis JSON")
    parser.add_argument("--degradation-json", help="Cluster degradation classification JSON")
    parser.add_argument("--affinity-json", help="NPU affinity analysis JSON")
    parser.add_argument("--collective-types-json", help="Collective type analysis JSON")
    parser.add_argument("--rank-variance-json", help="Rank variance analysis JSON")
    parser.add_argument("--wait-attribution-json", help="Wait time attribution JSON")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    try:
        profile = json.loads(Path(args.profile_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading profile JSON: {e}", file=__import__("sys").stderr)
        return 1
    try:
        bottlenecks = json.loads(Path(args.bottlenecks_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error reading bottlenecks JSON: {e}", file=__import__("sys").stderr)
        return 1

    step = load_optional_json(args.step_json)
    comm = load_optional_json(args.communication_json)
    memory = load_optional_json(args.memory_json)
    input_data = load_optional_json(args.input_json)
    trace_gaps = load_optional_json(args.trace_gaps_json)
    hotspot = load_optional_json(args.hotspot_json)
    mfu = load_optional_json(args.mfu_json)
    cluster = load_optional_json(args.cluster_json)
    jitter = load_optional_json(args.jitter_json)
    fusion = load_optional_json(args.fusion_json)
    degradation = load_optional_json(args.degradation_json)
    affinity = load_optional_json(args.affinity_json)
    collective_types = load_optional_json(args.collective_types_json)
    rank_variance = load_optional_json(args.rank_variance_json)
    wait_attribution = load_optional_json(args.wait_attribution_json)

    suggestions = build_suggestions(
        profile, bottlenecks,
        step, comm, memory, input_data, trace_gaps, hotspot,
        mfu, cluster, jitter, fusion, degradation, affinity,
        collective_types, rank_variance, wait_attribution,
    )

    high_count = sum(1 for s in suggestions if s.get("priority") == "high")
    medium_count = sum(1 for s in suggestions if s.get("priority") == "medium")
    low_count = sum(1 for s in suggestions if s.get("priority") == "low")

    report = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "suggestions": suggestions,
        "suggestion_summary": {
            "total_count": len(suggestions),
            "high_count": high_count,
            "medium_count": medium_count,
            "low_count": low_count,
            "top_suggestion": suggestions[0]["id"] if suggestions else None,
        },
    }

    write_json(Path(args.output_json), report)
    print(json.dumps(report["suggestion_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
