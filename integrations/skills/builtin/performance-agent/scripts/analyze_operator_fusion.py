#!/usr/bin/env python3
"""Detect operator fusion opportunities from profiling data.

Analyzes operator sequences and time shares to identify candidates for
FlashAttention, MatmulAllReduce, fused optimizers, and other NPU fusion patterns.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Optional

from perf_common import load_csv_rows, load_optional_json, parse_number, write_json


# Fusion pattern definitions: operator name patterns and their fused replacements.
FUSION_PATTERNS = [
    {
        "type": "flash_attention",
        "suggestion_id": "NPU-AFFINITY-03",
        "operator_patterns": [
            r"(?i)(flash.?attn|scaled_dot_product_attention|attention|self_attention|cross_attention)",
            r"(?i)(softmax|scaled.*softmax)",
            r"(?i)(dropout|drop_out)",
        ],
        "min_combined_share": 20.0,
        "estimated_speedup": "20-40%",
        "replacement_api": "torch_npu.npu_fusion_attention",
        "constraint": "Atlas A2 training series, float16/bfloat16",
        "description": "Replace attention + softmax + dropout with npu_fusion_attention",
    },
    {
        "type": "matmul_allreduce",
        "suggestion_id": "NPU-AFFINITY-02",
        "operator_patterns": [
            r"(?i)(matmul|gemm|linear|dense)",
            r"(?i)(all.?reduce|allreduce)",
        ],
        "requires_tp": True,
        "min_combined_share": 15.0,
        "estimated_speedup": "10-20%",
        "replacement_api": "torch_npu.npu_mm_all_reduce_base",
        "constraint": "Atlas A2 training series, TP splitting scenarios",
        "description": "Replace matmul + all_reduce with npu_mm_all_reduce_base",
    },
    {
        "type": "fused_optimizer",
        "suggestion_id": "NPU-AFFINITY-01",
        "operator_patterns": [
            r"(?i)(adam|adamw|sgd|lamb|adadelta|rmsprop)",
            r"(?i)(h2d|host_to_device|memcpy_h2d)",
        ],
        "min_combined_share": 10.0,
        "estimated_speedup": "5-15%",
        "replacement_api": "torch_npu.optim.NpuFused*",
        "constraint": "May increase memory usage",
        "description": "Replace standard optimizer with NPU fused variant",
        "optimizer_map": {
            "adam": "NpuFusedAdam",
            "adamw": "NpuFusedAdamW",
            "sgd": "NpuFusedSGD",
            "lamb": "NpuFusedLamb",
            "adadelta": "NpuFusedAdadelta",
            "rmsprop": "NpuFusedRMSprop",
        },
    },
    {
        "type": "rms_norm",
        "suggestion_id": "COMP-03",
        "operator_patterns": [
            r"(?i)(rms.?norm|rmsnorm|layer.?norm|layernorm)",
        ],
        "min_combined_share": 8.0,
        "estimated_speedup": "5-10%",
        "replacement_api": "NPU fused RmsNorm operator",
        "constraint": "Check Ascend PT documentation for RmsNorm replacement",
        "description": "Replace LayerNorm/RmsNorm with NPU fused variant",
    },
    {
        "type": "rotary_embedding",
        "suggestion_id": "COMP-03",
        "operator_patterns": [
            r"(?i)(rotary|rope|rotary.*mul|position.*embedding)",
        ],
        "min_combined_share": 5.0,
        "estimated_speedup": "5-10%",
        "replacement_api": "NPU RotaryMul fused operator",
        "constraint": "Check Ascend PT documentation for RotaryMul replacement",
        "description": "Replace rotary embedding with NPU RotaryMul fusion",
    },
    {
        "type": "swiglu",
        "suggestion_id": "COMP-03",
        "operator_patterns": [
            r"(?i)(swiglu|swi_glu|silu.*gate|gate.*silu)",
        ],
        "min_combined_share": 5.0,
        "estimated_speedup": "5-10%",
        "replacement_api": "NPU SwiGlu fused operator",
        "constraint": "Check Ascend PT documentation for SwiGlu replacement",
        "description": "Replace SwiGlu/activation with NPU fused variant",
    },
    {
        "type": "scaled_masked_softmax",
        "suggestion_id": "COMP-03",
        "operator_patterns": [
            r"(?i)(scaled.*masked.*softmax|masked.*softmax|causal.*mask)",
        ],
        "min_combined_share": 5.0,
        "estimated_speedup": "5-15%",
        "replacement_api": "NPU ScaledMaskedSoftmax fused operator",
        "constraint": "Check Ascend PT documentation for ScaledMaskedSoftmax replacement",
        "description": "Replace masked softmax with NPU fused variant",
    },
]


def _match_operator(name: str, pattern: str) -> bool:
    """Check if an operator name matches a regex pattern."""
    return bool(re.search(pattern, name))


def _extract_operators(
    hotspot_json: Optional[dict],
    kernel_csv_path: Optional[Path],
) -> list[dict]:
    """Extract operator list from hotspot JSON or kernel_details CSV."""
    operators = []

    if hotspot_json and hotspot_json.get("top_operators"):
        for op in hotspot_json["top_operators"]:
            operators.append({
                "name": op.get("operator", ""),
                "share_percent": parse_number(op.get("share_percent")) or 0,
                "duration_ms": parse_number(op.get("avg_duration_ms")) or 0,
                "count": parse_number(op.get("count")) or 0,
            })

    if kernel_csv_path and kernel_csv_path.exists():
        rows = load_csv_rows(kernel_csv_path)
        for row in rows:
            name = row.get("Name", row.get("Kernel Name", row.get("Op Name", "")))
            duration = parse_number(row.get("Duration(ms)", row.get("Total Time(ms)", "0")))
            if name and duration is not None:
                operators.append({
                    "name": name,
                    "duration_ms": duration,
                })

    return operators


def _is_tp_scenario(comm_json: Optional[dict], profile_json: Optional[dict]) -> bool:
    """Check if the workload uses tensor parallelism."""
    if profile_json:
        parallel = profile_json.get("parallel_config")
        if isinstance(parallel, dict):
            tp = parallel.get("tp_size")
            if tp is not None and tp > 1:
                return True

    if comm_json:
        top_collectives = comm_json.get("top_collectives", [])
        for c in top_collectives:
            name = str(c.get("name", "")).lower()
            if "allreduce" in name or "reduce_scatter" in name:
                return True

    return False


def detect_fusion_opportunities(
    operators: list[dict],
    comm_json: Optional[dict],
    profile_json: Optional[dict],
    step_json: Optional[dict],
) -> list[dict]:
    """Detect fusion opportunities from operator list."""
    opportunities = []
    is_tp = _is_tp_scenario(comm_json, profile_json)

    # Build operator name → share mapping
    op_shares: dict[str, float] = {}
    for op in operators:
        name = op.get("name", "")
        share = op.get("share_percent", 0)
        if name:
            op_shares[name] = max(op_shares.get(name, 0), float(share))

    for pattern_def in FUSION_PATTERNS:
        fusion_type = pattern_def["type"]

        # Skip matmul_allreduce if not TP
        if fusion_type == "matmul_allreduce" and not is_tp:
            continue

        # Find matching operators
        matched_ops = []
        matched_share = 0.0
        for op_name, share in op_shares.items():
            for op_pattern in pattern_def["operator_patterns"]:
                if _match_operator(op_name, op_pattern):
                    if op_name not in matched_ops:
                        matched_ops.append(op_name)
                        matched_share += share
                    break

        if not matched_ops:
            continue

        confidence = min(0.95, matched_share / max(pattern_def["min_combined_share"], 1) * 0.5)

        # Only report if share exceeds threshold
        if matched_share >= pattern_def["min_combined_share"] or len(matched_ops) >= 2:
            opportunity = {
                "type": fusion_type,
                "confidence": round(confidence, 3),
                "affected_operators": matched_ops,
                "combined_share_percent": round(matched_share, 1),
                "estimated_speedup": pattern_def["estimated_speedup"],
                "suggestion_id": pattern_def["suggestion_id"],
                "replacement_api": pattern_def["replacement_api"],
                "constraint": pattern_def["constraint"],
                "description": pattern_def["description"],
            }

            # Add optimizer-specific mapping
            if fusion_type == "fused_optimizer" and "optimizer_map" in pattern_def:
                detected_optimizer = None
                for op_name in matched_ops:
                    name_lower = op_name.lower()
                    for opt_key in pattern_def["optimizer_map"]:
                        if opt_key in name_lower:
                            detected_optimizer = opt_key
                            break
                    if detected_optimizer:
                        break
                if detected_optimizer:
                    opportunity["detected_optimizer"] = detected_optimizer
                    opportunity["fused_replacement"] = pattern_def["optimizer_map"][detected_optimizer]

            opportunities.append(opportunity)

    # Sort by combined share (largest opportunity first)
    opportunities.sort(key=lambda o: o["combined_share_percent"], reverse=True)
    return opportunities


def analyze(
    hotspot_json: Optional[dict],
    kernel_csv_path: Optional[Path],
    comm_json: Optional[dict],
    profile_json: Optional[dict],
    step_json: Optional[dict],
) -> dict:
    """Run operator fusion analysis."""
    operators = _extract_operators(hotspot_json, kernel_csv_path)

    if not operators:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "fusion_analysis_available": False,
        }

    opportunities = detect_fusion_opportunities(operators, comm_json, profile_json, step_json)

    # Estimate total potential speedup
    speedup_ranges = []
    for opp in opportunities:
        speed_range = opp["estimated_speedup"]
        speedup_ranges.append(speed_range)

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "fusion_analysis_available": True,
        "operators_analyzed": len(operators),
        "opportunities": opportunities,
        "total_fusion_candidates": len(opportunities),
        "estimated_total_speedup": (
            f"Combined {len(opportunities)} fusion opportunities detected"
            if opportunities
            else "No significant fusion opportunities detected"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect operator fusion opportunities from profiling data"
    )
    parser.add_argument("--hotspot-json", help="Hotspot summary JSON path")
    parser.add_argument("--kernel-details-csv", help="kernel_details.csv path")
    parser.add_argument("--communication-json", help="Communication summary JSON path")
    parser.add_argument("--profile-json", help="Performance profile JSON path")
    parser.add_argument("--step-json", help="Step breakdown JSON path")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    hotspot = load_optional_json(args.hotspot_json)
    kernel_csv = Path(args.kernel_details_csv) if args.kernel_details_csv else None
    comm_json = load_optional_json(args.communication_json)
    profile_json = load_optional_json(args.profile_json)
    step_json = load_optional_json(args.step_json)

    result = analyze(hotspot, kernel_csv, comm_json, profile_json, step_json)

    write_json(Path(args.output_json), result)
    print(json.dumps({
        "fusion_available": result.get("fusion_analysis_available", False),
        "opportunities": result.get("total_fusion_candidates", 0),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
