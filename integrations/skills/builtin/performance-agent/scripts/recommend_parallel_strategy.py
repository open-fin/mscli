#!/usr/bin/env python3
"""Recommend parallel strategy based on model size, hardware, and profiling data.

Strategy priority: TP → PP → DP → ZeRO/Recompute

Logic:
1. TP (Tensor Parallelism): First choice; limited to NPUs per node
2. PP (Pipeline Parallelism): Between nodes when TP is insufficient
3. DP (Data Parallelism): When resources are abundant
4. ZeRO: Shard optimizer states / gradients / weights
5. Recomputation: Trade time for space; enables larger batch_size
"""
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

from perf_common import (
    get_peak_tflops,
    HARDWARE_SPECS,
    infer_hardware,
    infer_parallel_config,
    read_json,
    write_json,
)


def estimate_model_size_gb(
    num_layers: int,
    hidden_size: int,
    vocab_size: int,
    seq_len: int,
    precision_bytes: int = 2,
    include_embedding: bool = True,
    intermediate_size: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    ffn_type: str = "standard",
) -> float:
    """Estimate model parameter memory in GB.

    Per-layer parameters (transformer):
      Standard FFN:
        Attention: 4 × H² (QKV + output projection)
        FFN: 2 × H × 4H = 8 × H²
        Total per layer: ~12 × H²

      SwiGLU FFN (LLaMA, Mistral, Qwen):
        Attention: 4 × H² (or less with GQA)
        FFN: 3 × H × intermediate_size
        Total per layer: varies

    Supports GQA: when num_kv_heads < num_attention_heads, K/V projections
    are smaller: 2 × H × (H × num_kv_heads / num_attention_heads).

    Args:
        num_layers: Number of transformer layers.
        hidden_size: Hidden dimension size (H).
        vocab_size: Vocabulary size.
        seq_len: Sequence length (reserved for future use).
        precision_bytes: Bytes per parameter (2 for bf16/fp16, 4 for fp32).
        include_embedding: Whether to include embedding parameters.
        intermediate_size: FFN intermediate dimension. Defaults to 4*H for
            standard FFN, or 8*H/3 for SwiGLU if not specified.
        num_attention_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads for GQA. Defaults to
            num_attention_heads (standard MHA).
        ffn_type: "standard" (2 projections) or "swiglu" (3 projections).
    """
    H = hidden_size

    # --- Attention parameters ---
    if num_attention_heads and num_kv_heads and num_kv_heads < num_attention_heads:
        # GQA: smaller K, V projections
        kv_dim = H * num_kv_heads // num_attention_heads
        attention_params = H * H + 2 * H * kv_dim + H * H  # Q + K + V + O
    else:
        attention_params = 4 * H * H  # Q + K + V + O, each H×H

    # --- FFN parameters ---
    if ffn_type == "swiglu":
        # SwiGLU has 3 projections: gate, up, down
        inter = intermediate_size or int(8 * H / 3)  # LLaMA default: 8H/3
        ffn_params = 3 * H * inter
    else:
        # Standard FFN: up + down projections
        inter = intermediate_size or 4 * H
        ffn_params = 2 * H * inter

    # --- LayerNorm / RMSNorm (negligible but included) ---
    norm_params = 2 * H  # 2 norms per layer, each with H params

    params_per_layer = attention_params + ffn_params + norm_params
    total_params = num_layers * params_per_layer

    # --- Embedding ---
    if include_embedding and vocab_size:
        total_params += vocab_size * H

    return total_params * precision_bytes / (1024 ** 3)


def estimate_activation_memory_gb(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_layers: int,
    precision_bytes: int = 2,
) -> float:
    """Rough estimate of activation memory per layer for backward pass."""
    # Activation per layer ≈ 2 × B × S × H (input + output)
    activation_per_layer = 2.0 * batch_size * seq_len * hidden_size * precision_bytes
    return num_layers * activation_per_layer / (1024 ** 3)


def recommend_tp_size(
    model_size_gb: float,
    hbm_per_npu_gb: float,
    npus_per_node: int,
    activation_gb: float = 0.0,
) -> dict:
    """Recommend tensor parallelism size.

    TP reduces per-device memory by sharding model parameters.
    Maximum useful TP = min(npus_per_node, required_by_memory).
    """
    total_memory_needed = model_size_gb * 1.5 + activation_gb  # 1.5x for optimizer + gradients

    if total_memory_needed <= hbm_per_npu_gb:
        return {
            "recommended_tp": 1,
            "reason": "Model fits in single device memory",
            "memory_per_device_gb": round(total_memory_needed, 2),
        }

    # Find minimum TP that fits
    for tp in range(2, npus_per_node + 1):
        mem_per_device = total_memory_needed / tp
        if mem_per_device <= hbm_per_npu_gb:
            return {
                "recommended_tp": tp,
                "reason": f"TP={tp} reduces per-device memory to {mem_per_device:.1f}GB (HBM={hbm_per_npu_gb}GB)",
                "memory_per_device_gb": round(mem_per_device, 2),
            }

    return {
        "recommended_tp": npus_per_node,
        "reason": f"Maximum TP={npus_per_node} still may not fit; consider PP or ZeRO",
        "memory_per_device_gb": round(total_memory_needed / npus_per_node, 2),
        "warning": "Model may not fit even with maximum intra-node TP",
    }


def recommend_pp_size(
    model_size_gb: float,
    hbm_per_npu_gb: float,
    tp_size: int,
    num_nodes: int,
    activation_gb: float = 0.0,
) -> dict:
    """Recommend pipeline parallelism size."""
    total_memory_needed = model_size_gb * 1.5 + activation_gb
    mem_after_tp = total_memory_needed / tp_size

    if mem_after_tp <= hbm_per_npu_gb:
        return {
            "recommended_pp": 1,
            "reason": "TP alone is sufficient for memory fit",
        }

    # Need PP
    max_pp = num_nodes if num_nodes > 1 else 1
    for pp in range(2, max_pp + 1):
        mem_per_device = mem_after_tp / pp
        if mem_per_device <= hbm_per_npu_gb:
            # Calculate pipeline efficiency
            bubble_ratio = 1.0 / pp  # Approximation for 1F1B
            return {
                "recommended_pp": pp,
                "reason": f"PP={pp} with TP={tp_size} reduces memory to {mem_per_device:.1f}GB",
                "memory_per_device_gb": round(mem_per_device, 2),
                "pipeline_bubble_ratio": round(bubble_ratio, 3),
                "note": f"Pipeline bubble waste: ~{bubble_ratio:.1%}. Use m>>p micro-batches to minimize.",
            }

    return {
        "recommended_pp": max_pp,
        "reason": f"Maximum PP={max_pp} with TP={tp_size}; consider ZeRO or recomputation",
        "warning": "Memory may still be tight",
    }


def recommend_zero_stage(
    model_size_gb: float,
    hbm_per_npu_gb: float,
    tp_size: int,
    pp_size: int,
    dp_size: int,
    activation_gb: float = 0.0,
) -> dict:
    """Recommend ZeRO optimization stage."""
    if dp_size < 2:
        return {
            "recommended_stage": 0,
            "reason": "ZeRO requires DP >= 2",
        }

    # Memory breakdown per device (after TP+PP)
    total_memory = model_size_gb * 1.5 + activation_gb
    mem_per_device = total_memory / (tp_size * pp_size)

    optimizer_states_gb = model_size_gb * 0.75 / (tp_size * pp_size)  # ~12 bytes per param for Adam
    gradients_gb = model_size_gb * 0.5 / (tp_size * pp_size)  # fp32 copy

    if mem_per_device > hbm_per_npu_gb:
        # Need aggressive sharding
        if pp_size <= 1:
            return {
                "recommended_stage": 3,
                "reason": f"ZeRO-3 shards weights+grads+optimizer ({mem_per_device:.1f}GB needed, {hbm_per_npu_gb}GB available)",
                "warning": "ZeRO-3 increases communication volume significantly",
            }
        return {
            "recommended_stage": 1,
            "reason": f"ZeRO-1 shards optimizer states only (compatible with PP={pp_size})",
            "memory_saving_gb": round(optimizer_states_gb, 2),
            "note": "ZeRO-2 and ZeRO-3 are incompatible with PP",
        }

    if optimizer_states_gb > hbm_per_npu_gb * 0.3:
        return {
            "recommended_stage": 1,
            "reason": f"ZeRO-1 reduces optimizer memory by {optimizer_states_gb:.1f}GB per device",
            "memory_saving_gb": round(optimizer_states_gb, 2),
        }

    return {
        "recommended_stage": 0,
        "reason": "Memory fits without ZeRO sharding",
    }


def recommend_recomputation(
    activation_gb: float,
    hbm_per_npu_gb: float,
    mem_per_device_gb: float,
) -> dict:
    """Recommend activation recomputation (gradient checkpointing)."""
    if mem_per_device_gb <= hbm_per_npu_gb * 0.8:
        return {
            "recommended": False,
            "reason": "Memory headroom is sufficient without recomputation",
        }

    # Recomputation trades ~33% more compute for ~60% less activation memory
    saving = activation_gb * 0.6
    new_usage = mem_per_device_gb - saving

    if new_usage < 0:
        new_usage = 0.0

    return {
        "recommended": True,
        "reason": f"Recomputation reduces activation memory by ~{saving:.1f}GB ({activation_gb:.1f}GB → {activation_gb - saving:.1f}GB)",
        "activation_saving_gb": round(saving, 2),
        "compute_overhead": "~33% more forward FLOPs",
        "new_estimated_usage_gb": round(new_usage, 2),
        "note": "Enabling recomputation may allow larger batch_size",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recommend parallel strategy based on model and hardware configuration"
    )
    parser.add_argument("--num-layers", type=int, default=None, help="number of transformer layers")
    parser.add_argument("--hidden-size", type=int, default=None, help="hidden dimension size")
    parser.add_argument("--seq-len", type=int, default=None, help="sequence length")
    parser.add_argument("--batch-size", type=int, default=None, help="per-device batch size")
    parser.add_argument("--model-config", help="model config JSON providing num_layers/hidden_size/seq_len/batch_size")
    parser.add_argument("--vocab-size", type=int, default=32000, help="vocabulary size")
    parser.add_argument("--num-nodes", type=int, default=1, help="number of nodes")
    parser.add_argument("--npus-per-node", type=int, default=8, help="NPUs per node")
    parser.add_argument("--hardware", help="hardware model key (e.g., ascend_910b2)")
    parser.add_argument("--trace-root", help="profiler root to infer hardware")
    parser.add_argument("--intermediate-size", type=int, default=None, help="FFN intermediate dimension (default: 4H for standard, 8H/3 for swiglu)")
    parser.add_argument("--num-attention-heads", type=int, default=None, help="number of query attention heads")
    parser.add_argument("--num-kv-heads", type=int, default=None, help="number of KV heads for GQA (default: same as num-attention-heads)")
    parser.add_argument("--ffn-type", default="standard", choices=["standard", "swiglu"], help="FFN type: standard or swiglu")
    parser.add_argument("--output-json", required=True, help="output JSON path")
    args = parser.parse_args()

    # Load model config if provided, filling in missing CLI args
    if args.model_config:
        try:
            mc = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"Cannot read --model-config: {exc}")
        if args.num_layers is None:
            args.num_layers = mc.get("num_layers") or mc.get("num-layer") or mc.get("n_layer")
        if args.hidden_size is None:
            args.hidden_size = mc.get("hidden_size") or mc.get("hidden-size") or mc.get("n_embd")
        if args.seq_len is None:
            args.seq_len = mc.get("seq_len") or mc.get("seq-length") or mc.get("seq_length")
        if args.batch_size is None:
            args.batch_size = mc.get("batch_size") or mc.get("batch-size")

    # Validate required params
    missing = []
    if not args.num_layers:
        missing.append("--num-layers")
    if not args.hidden_size:
        missing.append("--hidden-size")
    if not args.seq_len:
        missing.append("--seq-len")
    if not args.batch_size:
        missing.append("--batch-size")
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)} (or provide --model-config)")

    # Resolve hardware
    hardware = args.hardware
    if not hardware and args.trace_root:
        hardware = infer_hardware(Path(args.trace_root).resolve())

    hbm_per_npu = 64.0  # default
    if hardware:
        spec = HARDWARE_SPECS.get(hardware)
        if spec:
            hbm_per_npu = spec["hbm_capacity_gb"]

    # Estimate model size
    model_size_gb = estimate_model_size_gb(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        ffn_type=args.ffn_type,
    )

    activation_gb = estimate_activation_memory_gb(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )

    # Recommend strategy
    tp_rec = recommend_tp_size(model_size_gb, hbm_per_npu, args.npus_per_node, activation_gb)
    tp_size = tp_rec["recommended_tp"]

    pp_rec = recommend_pp_size(model_size_gb, hbm_per_npu, tp_size, args.num_nodes, activation_gb)
    pp_size = pp_rec["recommended_pp"]

    total_devices = args.npus_per_node * args.num_nodes
    dp_size = max(1, total_devices // (tp_size * pp_size))

    zero_rec = recommend_zero_stage(model_size_gb, hbm_per_npu, tp_size, pp_size, dp_size, activation_gb)

    mem_per_device = (model_size_gb * 1.5 + activation_gb) / (tp_size * pp_size)
    per_device_activation = activation_gb / (tp_size * pp_size) if (tp_size * pp_size) > 0 else activation_gb
    recomp_rec = recommend_recomputation(per_device_activation, hbm_per_npu, mem_per_device)

    report = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "model_estimates": {
            "num_layers": args.num_layers,
            "hidden_size": args.hidden_size,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "vocab_size": args.vocab_size,
            "estimated_model_size_gb": round(model_size_gb, 2),
            "estimated_activation_memory_gb": round(activation_gb, 2),
        },
        "hardware": {
            "model": hardware,
            "hbm_per_npu_gb": hbm_per_npu,
            "npus_per_node": args.npus_per_node,
            "num_nodes": args.num_nodes,
            "total_devices": total_devices,
        },
        "recommendations": {
            "tp": tp_rec,
            "pp": pp_rec,
            "dp": {
                "recommended_dp": dp_size,
                "reason": f"DP={dp_size} from {total_devices} / (TP={tp_size} × PP={pp_size})",
            },
            "zero": zero_rec,
            "recomputation": recomp_rec,
        },
        "strategy_summary": {
            "tp_size": tp_size,
            "pp_size": pp_size,
            "dp_size": dp_size,
            "world_size": tp_size * pp_size * dp_size,
            "zero_stage": zero_rec["recommended_stage"],
            "use_recomputation": recomp_rec["recommended"],
        },
        "parallel_strategy_priority": [
            "1. TP (Tensor Parallelism): first choice; reduces memory, utilizes compute",
            "2. PP (Pipeline Parallelism): between nodes when TP insufficient; minimize PP (more PP = more bubbles)",
            "3. DP (Data Parallelism): when resources are abundant",
            "4. ZeRO: shard optimizer/gradient/weight across DP ranks",
            "5. Recomputation: trade time for space; enables larger batch_size",
        ],
        "additional_notes": [
            "Enable SP (Sequence Parallelism) alongside TP — no extra communication overhead",
            "For 16P Box16: set HCCL_INTRA_ROCE_ENABLE=1 for cross-group communication",
            "HCCL_BUFFSIZE = ceil(MBS × S × H × dtype_size / 8MB) for LLM workloads",
            "Overlap optimization: increase micro batch size OR reduce single AllGather data volume",
        ],
    }

    write_json(Path(args.output_json), report)
    print(json.dumps({
        "strategy": f"TP={tp_size}, PP={pp_size}, DP={dp_size}",
        "world_size": tp_size * pp_size * dp_size,
        "zero_stage": zero_rec["recommended_stage"],
        "recomputation": recomp_rec["recommended"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
