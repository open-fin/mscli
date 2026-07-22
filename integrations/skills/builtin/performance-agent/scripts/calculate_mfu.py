#!/usr/bin/env python3
"""Calculate MFU (Machine FLOP Utilization) from profiling data.

Supports two methods:
1. Model-config-based (precise, requires model_config JSON)
2. Time-ratio-based (rough estimate from compute time ratio)
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from perf_common import (
    load_optional_json,
    read_json,
    get_peak_tflops,
    infer_hardware,
    mfu_level,
    write_json,
)


# ---------------------------------------------------------------------------
# Method 1: Model-config-based (Chinchilla formula)
# ---------------------------------------------------------------------------

def estimate_model_flops(config: dict) -> Optional[float]:
    """Estimate FLOPs per training step from model config.

    Uses the Chinchilla-style formula with support for SwiGLU and GQA:

    Standard Transformer:
      FLOPs = 6 * B * S * L * H^2 * (1 + S / (6 * H))

    SwiGLU Transformer (LLaMA-style):
      FLOPs = 6 * B * S * L * H * (H + intermediate) * (1 + S / (6 * H))
            where intermediate defaults to 8H/3

    Training: forward + backward ≈ 6x matmul FLOPs (3x forward × 2 for backward).

    Expects config keys: hidden_size, num_layers, seq_len, batch_size
    Optional: vocab_size, intermediate_size, ffn_type, num_attention_heads, num_kv_heads
    """
    hidden = config.get("hidden_size") or config.get("hidden")
    num_layers = config.get("num_layers") or config.get("num_hidden_layers") or config.get("n_layer")
    seq_len = config.get("seq_len") or config.get("seq_length") or config.get("max_position_embeddings")
    batch_size = config.get("batch_size") or config.get("global_batch_size") or config.get("micro_batch_size")
    vocab_size = config.get("vocab_size")
    ffn_type = config.get("ffn_type", "standard")
    intermediate_size = config.get("intermediate_size")
    num_attention_heads = config.get("num_attention_heads") or config.get("num_heads")
    num_kv_heads = config.get("num_kv_heads") or config.get("num_key_value_heads")

    if not all([hidden, num_layers, seq_len, batch_size]):
        return None

    hidden = int(hidden)
    num_layers = int(num_layers)
    seq_len = int(seq_len)
    batch_size = int(batch_size)

    # Compute per-layer FLOPs (forward pass)
    # Attention: Q*K^T + attn*V = 2*B*S*H^2 (standard MHA)
    # With GQA: Q*K^T uses H*d_kv, attn*V uses d_kv*H
    if num_attention_heads and num_kv_heads and int(num_kv_heads) < int(num_attention_heads):
        # GQA: total attention FLOPs reduced proportionally
        gqa_ratio = int(num_kv_heads) / int(num_attention_heads)
        attention_flops = 2 * batch_size * seq_len * hidden * hidden * (1 + gqa_ratio)
    else:
        attention_flops = 4 * batch_size * seq_len * hidden * hidden  # Q*K^T + attn*V

    # FFN FLOPs
    if ffn_type == "swiglu":
        inter = int(intermediate_size) if intermediate_size else int(8 * hidden / 3)
        # SwiGLU: gate(H→inter) + up(H→inter) + down(inter→H) = 3 matmuls
        ffn_flops = 2 * batch_size * seq_len * hidden * inter * 3
    else:
        inter = int(intermediate_size) if intermediate_size else 4 * hidden
        # Standard: up(H→inter) + down(inter→H) = 2 matmuls
        ffn_flops = 2 * batch_size * seq_len * hidden * inter * 2

    # Per-layer FLOPs (forward), training = 3x (forward + backward)
    layer_flops_forward = attention_flops + ffn_flops
    total_flops = 3 * num_layers * layer_flops_forward

    # Add embedding FLOPs if vocab_size is provided
    if vocab_size:
        vocab_size = int(vocab_size)
        total_flops += 6 * batch_size * seq_len * vocab_size * hidden

    return total_flops


# ---------------------------------------------------------------------------
# Method 2: Time-ratio-based (rough)
# ---------------------------------------------------------------------------

def estimate_mfu_from_ratio(compute_ratio: float) -> dict:
    """Rough MFU estimate from compute time ratio.

    Assumes compute efficiency of ~80% within the compute-active period.
    """
    estimated_mfu = compute_ratio * 0.8
    return {
        "estimated_mfu": round(estimated_mfu, 4),
        "method": "time_ratio",
        "reliability": "low",
        "note": "Rough estimate from compute time ratio. Use model_config for precise calculation.",
    }


# ---------------------------------------------------------------------------
# Main calculation
# ---------------------------------------------------------------------------

def calculate_mfu(
    step_json: Optional[dict],
    model_config: Optional[dict],
    hardware: Optional[str],
    num_devices: int = 1,
) -> dict:
    """Calculate MFU using the best available method.

    Methods (in priority order):
    1. Model-config-based: uses Chinchilla formula when model config is provided.
    2. Time-ratio-based: rough estimate from compute time ratio in step data.
    """
    # Get step time
    step_time_ms = None
    if step_json:
        step_time_ms = step_json.get("average_step_time_ms")

    if not step_time_ms:
        return {
            "estimated_mfu": None,
            "mfu_level": None,
            "method": None,
            "error": "Step time not available",
        }

    step_time_s = step_time_ms / 1000.0

    # Get peak TFLOPS
    peak_tflops = get_peak_tflops(hardware, "fp16")
    if peak_tflops and num_devices > 1:
        peak_tflops = peak_tflops * num_devices

    # Try Method 1: Model config
    if model_config:
        flops = estimate_model_flops(model_config)
        if flops and step_time_s > 0:
            achieved_tflops = (flops / step_time_s) / 1e12
            if peak_tflops:
                mfu = achieved_tflops / peak_tflops
            else:
                mfu = None
            return {
                "estimated_mfu": round(mfu, 4) if mfu is not None else None,
                "mfu_level": mfu_level(mfu) if mfu is not None else None,
                "method": "model_config",
                "reliability": "high",
                "model_flops_per_step": flops,
                "achieved_tflops": round(achieved_tflops, 2),
                "peak_tflops_used": peak_tflops,
                "step_time_ms": round(step_time_ms, 3),
                "num_devices": num_devices,
                "gap_to_target": {
                    "target_mfu": 0.55,
                    "gap_percent": round((0.55 - mfu) * 100, 1) if mfu is not None else None,
                } if mfu is not None else None,
            }

    # Try Method 2: Time-ratio-based
    if step_json:
        # Compute ratio from stage totals
        stage_totals = step_json.get("stage_totals_ms", {})
        total_stage = sum(v for k, v in stage_totals.items() if k != "step_total")
        compute_time = stage_totals.get("compute", 0)
        if total_stage > 0:
            compute_ratio = compute_time / total_stage
            ratio_result = estimate_mfu_from_ratio(compute_ratio)
            mfu = ratio_result["estimated_mfu"]
            return {
                "estimated_mfu": mfu,
                "mfu_level": mfu_level(mfu),
                "method": "time_ratio",
                "reliability": "low",
                "compute_ratio": round(compute_ratio, 4),
                "step_time_ms": round(step_time_ms, 3),
                "peak_tflops_used": peak_tflops,
                "num_devices": num_devices,
                "note": ratio_result["note"],
            }

    return {
        "estimated_mfu": None,
        "mfu_level": None,
        "method": None,
        "error": "Insufficient data for MFU calculation. Provide model_config or step summaries.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate MFU from profiling data")
    parser.add_argument("--step-json", help="Step summary JSON path")
    parser.add_argument("--model-config", help="Model config JSON path")
    parser.add_argument("--trace-root", help="Profiler root for hardware auto-detection")
    parser.add_argument("--hardware", help="Hardware model name (e.g. ascend_910b2)")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    step = load_optional_json(args.step_json)
    model_config = load_optional_json(args.model_config)

    # Determine hardware
    hardware = args.hardware
    if not hardware and args.trace_root:
        hardware = infer_hardware(Path(args.trace_root))

    result = calculate_mfu(step, model_config, hardware, args.num_devices)
    write_json(Path(args.output_json), result)
    print(json.dumps({
        "estimated_mfu": result.get("estimated_mfu"),
        "mfu_level": result.get("mfu_level"),
        "method": result.get("method"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
