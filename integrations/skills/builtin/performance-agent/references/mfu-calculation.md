# MFU Calculation Reference

Read this file when estimating Machine FLOP Utilization from profiling data.

## Definition

MFU (Machine FLOP Utilization) measures how efficiently the hardware is used
for actual computation:

```
MFU = Achieved TFLOPS / Peak TFLOPS
Achieved TFLOPS = (FLOPs per step) / (step time in seconds) / 1e12
```

## FLOPs Formulas

### Standard Matmul / GEMM

For matrices (M, K) x (K, N):

```
FLOPs = 2 * M * N * K
```

### Batched Matmul

For (B, M, K) x (B, K, N):

```
FLOPs = 2 * B * M * N * K
```

### Linear Layer

Input (B, L, D_in), Weight (D_in, D_out):

```
M = B * L, K = D_in, N = D_out
FLOPs = 2 * B * L * D_in * D_out
```

### Attention QK^T

Q = (B, H, L_q, D_h), K = (B, H, L_k, D_h):

```
B' = B * H, M = L_q, N = L_k, K = D_h
FLOPs = 2 * B * H * L_q * L_k * D_h
```

### FlashAttention (Common Layout)

Full attention (sparse_mode == 0):

```
FLOPs = 2 * B * N * S * S * (D + D)
```

Causal attention (sparse_mode == 2, q_s == k_s):

```
FLOPs = full_attention * 0.5
```

### Transformer Model FLOPs Estimation

For a transformer with:
- hidden_size = H
- num_layers = L
- seq_len = S
- batch_size = B
- vocab_size = V

Forward pass FLOPs (approximation):

```
FLOPs_forward = B * S * (
    L * (12 * H^2 * S + 2 * H * S^2)    # attention + MLP
    + 2 * V * H                           # embedding (optional)
)
```

Total per training step (forward + backward ≈ 3x forward):

```
FLOPs_per_step = 3 * FLOPs_forward
```

Chinchilla-style simplified estimate:

```
FLOPs_per_step = 6 * B * S * L * H^2 * (1 + S / (6 * H))
```

## Peak TFLOPS Reference

> For full hardware specs including HBM bandwidth, capacity, and HCCS bandwidth, see [hardware-specs.md](hardware-specs.md).

| Hardware | FP16/BF16 Peak |
|----------|---------------|
| Ascend 910B1 | 378.88 TFLOPS |
| Ascend 910B2 | 353.89 TFLOPS |
| Ascend 910B3 | 294.91 TFLOPS |
| Ascend 910B4 | 270.00 TFLOPS |
| Atlas A2 (280T) | 280.00 TFLOPS |
| Atlas A2 (313T) | 313.00 TFLOPS |
| Atlas A2 (376T) | 376.00 TFLOPS |

For multi-card, peak TFLOPS scales linearly with the number of devices.

## MFU Levels

| Range | Level | Interpretation |
|-------|-------|----------------|
| < 20% | low | Far from peak; likely memory-bound or launch-overhead dominated |
| 20-40% | below_average | Common for unoptimized workloads |
| 40-60% | medium | Reasonable for most training workloads |
| 60-70% | good | Well-optimized |
| > 70% | excellent | Near device ceiling |

## Calculation Methods

### Method 1: Model-Config-Based (Preferred)

When model configuration (hidden_size, num_layers, seq_len, vocab_size) is
available, use the Chinchilla formula for precise estimation.

### Method 2: Hotspot-Inference-Based (Fallback)

When only profiling summaries are available:
1. Identify matmul/GEMM operators from hotspot summary
2. Estimate total FLOPs from operator shapes and call counts
3. Use step_time to compute achieved TFLOPS

### Method 3: Time-Ratio-Based (Rough Estimate)

When neither model config nor operator details are available:

```
rough_mfu ≈ compute_time_ratio * 0.8  # assumes ~80% compute efficiency
```

This is an approximation and should be flagged as such.

## Per-Layer Transformer FLOPs Breakdown

For a single Transformer layer (forward pass):

```
Attention module:
  QKV projections:     6 × B × S × H²
  Attention (Q×K^T):   2 × B × S² × H
  Attention (Attn×V):  2 × B × S² × H
  Output projection:   2 × B × S × H²
  Attention subtotal:  12 × B × S × H² + 4 × B × S² × H

FFN module:
  Expand + Contract:   16 × B × S × H²

Layer forward total:   24 × B × S × H² + 4 × B × S² × H
```

Total training FLOPs with L layers:

```
Without recomputation: 3 × L × (24×B×S×H² + 4×B×S²×H)
With recomputation:    4 × L × (24×B×S×H² + 4×B×S²×H)
```

**Key insight:** 99.3%+ of FLOPs in GPT-like models come from
fp16 matrix multiplication (GEMM). These should execute on the CUBE unit, not
the VECTOR unit. If VECTOR utilization is high while CUBE is low, the operator
mix may be suboptimal.

## Important Notes

- All FLOPs formulas count multiply-accumulate as 2 operations (1 mul + 1 add).
- For multi-card training, divide step_time by the number of devices for
  per-device MFU, or use total FLOPs across all devices.
- Activation recomputation (gradient checkpointing) adds ~33% forward FLOPs.
- MFU does not account for communication overhead; a low MFU with high
  communication time is expected and should be analyzed separately.

## See Also

- [hardware-specs.md](hardware-specs.md) — Full hardware specs table with TFLOPS, bandwidth, HBM capacity
- [performance-optimization-map.md](performance-optimization-map.md) Section 1.4 — MFU metric overview and threshold quick reference
- [aic-microarch-signatures.md](aic-microarch-signatures.md) — CUBE vs VECTOR utilization analysis
