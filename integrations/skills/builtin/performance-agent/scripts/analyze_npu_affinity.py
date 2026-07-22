#!/usr/bin/env python3
"""NPU affinity optimization analysis using the four-step method.

Step 1: Operator Fusion — detect fusion opportunities
Step 2: Stream Sync Elimination — detect unnecessary synchronization
Step 3: Multi-Card Consistency — detect slow card wait patterns
Step 4: CPU Optimization — detect CPU-bound operations
"""
import argparse
import json
import re
from pathlib import Path
from typing import Optional

from perf_common import load_csv_rows, load_optional_json, parse_number, write_json


# Step 1: Fusion opportunity patterns.
FUSION_CANDIDATES = {
    "flash_attention": {
        "patterns": [r"(?i)(attention|self_attn|scaled_dot)", r"(?i)(softmax)"],
        "min_share": 20.0,
        "suggestion_id": "NPU-AFFINITY-03",
        "code_example": (
            "# Before\n"
            "attn_output = F.scaled_dot_product_attention(q, k, v)\n"
            "# After\n"
            "attn_output = torch_npu.npu_fusion_attention(\n"
            "    q, k, v,\n"
            "    head_num=num_heads,\n"
            "    input_layout='BNSD',\n"
            "    keep_prob=1.0,\n"
            "    scale=1.0/math.sqrt(head_dim)\n"
            ")"
        ),
    },
    "matmul_allreduce": {
        "patterns": [r"(?i)(matmul|linear)", r"(?i)(all.?reduce)"],
        "min_share": 15.0,
        "suggestion_id": "NPU-AFFINITY-02",
        "code_example": (
            "# Before\n"
            "output = torch.matmul(input, weight)\n"
            "dist.all_reduce(output, op=ReduceOp.SUM)\n"
            "# After\n"
            "output = torch_npu.npu_mm_all_reduce_base(\n"
            "    input, weight, hcomm_info,\n"
            "    reduce_op='sum', comm_turn=0\n"
            ")"
        ),
    },
    "fused_optimizer": {
        "patterns": [r"(?i)(adam|adamw|sgd|rmsprop)"],
        "min_share": 10.0,
        "suggestion_id": "NPU-AFFINITY-01",
        "code_example": (
            "# Before\n"
            "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)\n"
            "# After\n"
            "optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=1e-4)"
        ),
    },
    "rms_norm": {
        "patterns": [r"(?i)(rms.?norm|layer.?norm)"],
        "min_share": 8.0,
        "suggestion_id": "COMP-03",
        "code_example": (
            "# Check Ascend PT documentation for NPU fused RmsNorm replacement\n"
            "# Applies to: RmsNorm, RmsNormGrad"
        ),
    },
    "swiglu": {
        "patterns": [r"(?i)(swiglu|swi.?glu|silu.*gate)"],
        "min_share": 5.0,
        "suggestion_id": "COMP-03",
        "code_example": (
            "# Check Ascend PT documentation for NPU SwiGlu fused operator"
        ),
    },
}

# Step 2: Sync-inducing operation patterns.
SYNC_OPERATIONS = {
    "tensor.item": {
        "patterns": [r"(?i)(item\(\)|\.item|cpu\(\).*numpy)"],
        "issue": "Triggers h2d synchronization in the training loop",
        "replacement": "Avoid in hot path; use NPU-side logic for conditional checks",
        "suggestion_id": "NPU-AFFINITY-04",
    },
    "tensor.reduce_all": {
        "patterns": [r"(?i)(reduce_all|all_reduce_host)"],
        "issue": "Inserts unnecessary synchronization barrier",
        "replacement": "Use NPU-native alternatives that avoid host roundtrip",
        "suggestion_id": "NPU-AFFINITY-04",
    },
    "torch.isfinite": {
        "patterns": [r"(?i)(isfinite|isnan|isinf)"],
        "issue": "Forces device-to-host transfer for validation",
        "replacement": "Use NPU-side checks or remove from hot loop",
        "suggestion_id": "NPU-AFFINITY-04",
    },
}

# Step 3: Multi-card consistency thresholds.
SLOW_CARD_THRESHOLD_FREE_TIME_PCT = 10.0
SLOW_CARD_THRESHOLD_STEP_TIME_VARIANCE = 0.20

# Step 4: CPU optimization indicators.
CPU_BOUND_INDICATORS = {
    "high_host_overhead": 20.0,  # host_overhead > 20% of step time
    "low_device_utilization": 60.0,  # device utilization < 60%
    "high_dispatch_latency_ms": 1.0,  # dispatch latency > 1ms
}


def _extract_operator_shares(hotspot_json: Optional[dict]) -> dict[str, float]:
    """Extract operator name -> share_percent mapping."""
    result = {}
    if not hotspot_json:
        return result
    for op in hotspot_json.get("top_operators", []):
        name = op.get("operator", "")
        share = parse_number(op.get("share_percent"))
        if name and share is not None:
            result[name] = float(share)
    return result


def _step1_operator_fusion(
    hotspot_json: Optional[dict],
    comm_json: Optional[dict],
) -> dict:
    """Step 1: Detect operator fusion opportunities."""
    op_shares = _extract_operator_shares(hotspot_json)
    if not op_shares:
        return {"findings": [], "suggestions": [], "score": 1.0}

    # Detect if tensor parallelism is in use (from communication data)
    has_tp = False
    if comm_json:
        comm_ops = [op.get("name", "").lower() for op in comm_json.get("operations", [])]
        has_tp = any("allreduce" in op or "reducescatter" in op for op in comm_ops)

    findings = []
    suggestions = []
    total_fusion_share = 0.0

    for fusion_type, config in FUSION_CANDIDATES.items():
        # Skip matmul_allreduce fusion if no TP detected
        if fusion_type == "matmul_allreduce" and not has_tp:
            continue

        matched_ops = []
        matched_share = 0.0
        for op_name, share in op_shares.items():
            for pattern in config["patterns"]:
                if re.search(pattern, op_name):
                    if op_name not in matched_ops:
                        matched_ops.append(op_name)
                        matched_share += share
                    break

        if matched_share >= config["min_share"]:
            findings.append({
                "fusion_type": fusion_type,
                "operators": matched_ops,
                "combined_share": round(matched_share, 1),
                "min_threshold": config["min_share"],
            })
            suggestions.append({
                "id": config["suggestion_id"],
                "type": fusion_type,
                "code_example": config["code_example"],
            })
            total_fusion_share += matched_share

    # Score: 1.0 = fully optimized, lower = more opportunities
    score = max(0.0, 1.0 - total_fusion_share / 100.0)

    return {"findings": findings, "suggestions": suggestions, "score": round(score, 3)}


def _step2_sync_elimination(
    trace_gaps_json: Optional[dict],
    step_json: Optional[dict],
    host_device_json: Optional[dict],
) -> dict:
    """Step 2: Detect unnecessary stream synchronization."""
    findings = []
    suggestions = []

    # Check host-device correlation for sync points
    if host_device_json and host_device_json.get("host_device_correlation_available"):
        sync_points = host_device_json.get("sync_points", [])
        for sp in sync_points:
            sync_type = sp.get("type", "")
            if sync_type in SYNC_OPERATIONS:
                config = SYNC_OPERATIONS[sync_type]
                findings.append({
                    "sync_type": sync_type,
                    "duration_ms": sp.get("duration_ms"),
                    "issue": config["issue"],
                })
                suggestions.append({
                    "id": config["suggestion_id"],
                    "type": sync_type,
                    "code_example": f"# Replace: {sync_type}\n# With: {config['replacement']}",
                })

    # Check host overhead from step/trace
    host_share = 0.0
    if trace_gaps_json and trace_gaps_json.get("dominant_category"):
        dominant = trace_gaps_json["dominant_category"]
        if dominant.get("name") in ("host_overhead", "idle_gap"):
            host_share = float(dominant.get("share_percent", 0))

    if step_json and step_json.get("dominant_stage"):
        dominant = step_json["dominant_stage"]
        if dominant.get("name") in ("host_overhead", "idle_gap"):
            host_share = max(host_share, float(dominant.get("share_percent", 0)))

    if host_share > CPU_BOUND_INDICATORS["high_host_overhead"]:
        findings.append({
            "sync_type": "high_host_overhead",
            "host_share_percent": round(host_share, 1),
            "issue": f"Host overhead is {host_share:.1f}% of step time",
        })

    # Score based on sync findings
    score = max(0.0, 1.0 - len(findings) * 0.15)
    return {"findings": findings, "suggestions": suggestions, "score": round(score, 3)}


def _step3_multi_card_consistency(
    cluster_json: Optional[dict],
    comm_json: Optional[dict],
) -> dict:
    """Step 3: Detect multi-card consistency issues."""
    findings = []
    suggestions = []

    if not cluster_json or not cluster_json.get("cluster_analysis_available"):
        return {"findings": [], "suggestions": [], "score": 1.0}

    slow_ranks = cluster_json.get("slow_ranks", [])
    analysis = cluster_json.get("analysis", {})
    bottleneck_type = analysis.get("bottleneck_type", "general")

    if slow_ranks:
        findings.append({
            "slow_ranks": slow_ranks,
            "bottleneck_type": bottleneck_type,
            "diagnosis": analysis.get("diagnosis", ""),
        })
        suggestions.append({
            "id": "CLUSTER-01",
            "type": "slow_card_wait",
            "code_example": (
                "# Check CPU affinity for the slow rank\n"
                "# numactl --cpunodebind=0 --membind=0 python train.py\n"
                "# Compare operator stats between slow and fast ranks"
            ),
        })

    # Check communication matrix for imbalance
    if comm_json and comm_json.get("matrix_imbalance_ratio") is not None:
        imbalance = float(comm_json["matrix_imbalance_ratio"])
        if imbalance > 0.2:
            findings.append({
                "communication_imbalance": round(imbalance, 3),
                "issue": f"Communication matrix imbalance ratio {imbalance:.3f} exceeds 0.2 threshold",
            })

    score = max(0.0, 1.0 - len(findings) * 0.2)
    return {"findings": findings, "suggestions": suggestions, "score": round(score, 3)}


def _step4_cpu_optimization(
    step_json: Optional[dict],
    trace_gaps_json: Optional[dict],
    bound_type_json: Optional[dict],
) -> dict:
    """Step 4: Detect CPU optimization opportunities."""
    findings = []
    suggestions = []

    # Check bound type
    if bound_type_json and bound_type_json.get("bound_type"):
        bound = bound_type_json["bound_type"]
        if bound == "host":
            findings.append({
                "bound_type": "host",
                "issue": "Workload is Host-Bound: device waits for host dispatch",
                "recommendation": "Use flame graph to analyze CPU bottleneck",
            })
            suggestions.append({
                "id": "HOST-01",
                "type": "cpu_optimization",
                "code_example": (
                    "# Move CPU-bound operations to NPU\n"
                    "# Use multi-process acceleration for CPU-side ops\n"
                    "# Enable graph compilation to reduce host dispatch"
                ),
            })

    # Check host overhead percentage
    host_share = 0.0
    if step_json and step_json.get("dominant_stage"):
        dominant = step_json["dominant_stage"]
        if dominant.get("name") in ("host_overhead", "idle_gap"):
            host_share = float(dominant.get("share_percent", 0))

    if host_share > CPU_BOUND_INDICATORS["high_host_overhead"]:
        findings.append({
            "host_share_percent": round(host_share, 1),
            "issue": f"Host overhead {host_share:.1f}% exceeds {CPU_BOUND_INDICATORS['high_host_overhead']}% threshold",
        })

    score = max(0.0, 1.0 - len(findings) * 0.2)
    return {"findings": findings, "suggestions": suggestions, "score": round(score, 3)}


def analyze_npu_affinity(
    hotspot_json: Optional[dict],
    comm_json: Optional[dict],
    trace_gaps_json: Optional[dict],
    step_json: Optional[dict],
    cluster_json: Optional[dict],
    host_device_json: Optional[dict],
    bound_type_json: Optional[dict],
) -> dict:
    """Run the four-step NPU affinity analysis."""
    step1 = _step1_operator_fusion(hotspot_json, comm_json)
    step2 = _step2_sync_elimination(trace_gaps_json, step_json, host_device_json)
    step3 = _step3_multi_card_consistency(cluster_json, comm_json)
    step4 = _step4_cpu_optimization(step_json, trace_gaps_json, bound_type_json)

    steps = [
        {"step": 1, "name": "operator_fusion", **step1},
        {"step": 2, "name": "stream_sync_elimination", **step2},
        {"step": 3, "name": "multi_card_consistency", **step3},
        {"step": 4, "name": "cpu_optimization", **step4},
    ]

    # Overall affinity score (average of step scores)
    scores = [s["score"] for s in steps]
    overall_score = round(sum(scores) / len(scores), 3) if scores else 1.0

    # Find priority fix from the weakest step
    weakest = min(steps, key=lambda s: s["score"])
    priority_fix = None
    if weakest["score"] < 0.8:
        suggestions = weakest.get("suggestions", [])
        if suggestions:
            first_sug = suggestions[0]
            priority_fix = f"{first_sug.get('type', weakest['name'])} ({first_sug.get('id', '')})"
        else:
            priority_fix = f"Address {weakest['name']} issues (score: {weakest['score']})"

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "npu_affinity_analysis_available": True,
        "steps": steps,
        "overall_affinity_score": overall_score,
        "priority_fix": priority_fix,
        "total_findings": sum(len(s.get("findings", [])) for s in steps),
        "total_suggestions": sum(len(s.get("suggestions", [])) for s in steps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="NPU affinity optimization analysis (four-step method)"
    )
    parser.add_argument("--hotspot-json", help="Hotspot summary JSON path")
    parser.add_argument("--communication-json", help="Communication summary JSON path")
    parser.add_argument("--trace-gaps-json", help="Trace gaps summary JSON path")
    parser.add_argument("--step-json", help="Step breakdown JSON path")
    parser.add_argument("--cluster-json", help="Cluster analysis JSON path")
    parser.add_argument("--host-device-json", help="Host-device correlation JSON path")
    parser.add_argument("--bound-type-json", help="Bound type detection JSON path")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    hotspot = load_optional_json(args.hotspot_json)
    comm = load_optional_json(args.communication_json)
    trace_gaps = load_optional_json(args.trace_gaps_json)
    step = load_optional_json(args.step_json)
    cluster = load_optional_json(args.cluster_json)
    host_device = load_optional_json(args.host_device_json)
    bound_type = load_optional_json(args.bound_type_json)

    result = analyze_npu_affinity(hotspot, comm, trace_gaps, step, cluster, host_device, bound_type)

    write_json(Path(args.output_json), result)
    print(json.dumps({
        "affinity_score": result.get("overall_affinity_score"),
        "priority_fix": result.get("priority_fix"),
        "findings": result.get("total_findings", 0),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
