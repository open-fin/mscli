# Ascend NPU Hardware Specifications

Reference table for MFU calculation, Roofline modeling, and performance
ceiling analysis.

## Supported Hardware

| Model | FP16/BF16 TFLOPS | FP32 TFLOPS | HBM Bandwidth | HBM Capacity | HCCS Bandwidth |
|-------|-----------------|-------------|---------------|-------------|---------------|
| Ascend 910B1 | 378.88 | 189.44 | 1.6 TB/s | 64 GB | 56 GB/s |
| Ascend 910B2 | 353.89 | 176.95 | 1.5 TB/s | 64 GB | 56 GB/s |
| Ascend 910B3 | 294.91 | 147.46 | 1.2 TB/s | 64 GB | 56 GB/s |
| Ascend 910B4 | 270.00 | 135.00 | 1.2 TB/s | 32 GB | 56 GB/s |
| Atlas A2 (280T) | 280.00 | 140.00 | 1.5 TB/s | 64 GB | 56 GB/s |
| Atlas A2 (313T) | 313.00 | 156.50 | 1.8 TB/s | 64 GB | 56 GB/s |
| Atlas A2 (376T) | 376.00 | 188.00 | 2.0 TB/s | 64 GB | 56 GB/s |
| Atlas 300I (310P) | 22.00 | 11.00 | 68 GB/s | - | - |

## Typical NPU Topology

```
Atlas 800 (8x 910B):
  Ring 0: NPU 0 - 1 - 2 - 3  (HCCS 56 GB/s)
  Ring 1: NPU 4 - 5 - 6 - 7  (HCCS 56 GB/s)
  Ring 0 <-> Ring 1: PCIe/Host (~28 GB/s, ~1/2 bandwidth)

Atlas 200T A2 Box16 (16x NPUs):
  Group 0 (8x NPUs): HCCS Full Mesh (56 GB/s)
  Group 1 (8x NPUs): HCCS Full Mesh (56 GB/s)
  Group 0 <-> Group 1: PCIe (~28 GB/s) or RDMA (when HCCL_INTRA_ROCE_ENABLE=1)

Multi-node:
  Intra-node: HCCS (56 GB/s)
  Inter-node: RDMA/RoCE (varies by network config)
```

Cross-ring and cross-node communication is significantly slower than intra-ring.
Ranks straddling ring boundaries (e.g., Rank 3 and Rank 4 in an AllReduce)
will show higher communication latency.

## Roofline Ridge Points

The ridge point (compute intensity at the transition from memory-bound to
compute-bound) can be estimated as:

```
ridge_point = peak_flops / peak_bandwidth
```

| Model | FP16 Ridge Point (FLOP/Byte) |
|-------|------------------------------|
| Ascend 910B1 | ~237 |
| Ascend 910B2 | ~236 |
| Ascend 910B3 | ~246 |
| Ascend 910B4 | ~225 |
| Atlas A2 (280T) | ~187 |
| Atlas A2 (313T) | ~174 |
| Atlas A2 (376T) | ~188 |

## MFU Reference Levels

| MFU Range | Level | Interpretation |
|-----------|-------|----------------|
| < 20% | low | Far from peak, likely memory-bound or launch-overhead dominated |
| 20-40% | below_average | Common for unoptimized workloads |
| 40-60% | medium | Reasonable for most training workloads |
| 60-70% | good | Well-optimized |
| > 70% | excellent | Near device ceiling |

## How To Detect Hardware From Profiler Data

1. Check `profiler_info.json` for `device_info` or `chip_name` fields.
2. Check directory naming patterns: some profiler exports encode the chip
   name in the directory.
3. If neither source reveals the model, ask the user. Do not guess.

## Usage Notes

- All TFLOPS values are theoretical peak for the stated precision.
- Actual achievable TFLOPS depends on operator mix, data layout, and
  kernel implementation.
- HCCS bandwidth applies to intra-node communication. Inter-node uses RDMA
  or RoCE and is typically lower.
- When the exact model is unknown but the chip family is known, use the
  most common variant (e.g., 910B2 for the 910B family).

## Communication Reference

### HCCL Buffer Size Recommendation (LLM)

```
HCCL_BUFFSIZE = ceil(MBS × S × H × dtype_size / (8 × 1024 × 1024))
```

- Default: 200 MB
- Memory-constrained: reduce below default
- Communication-intensive: increase above calculated value

### RDMA/SDMA Bandwidth Expectations

- SDMA (intra-node HCCS): up to 56 GB/s theoretical
- RDMA (inter-node RoCE): varies by network, typically 10-25 GB/s
- PCIe (NPU→CPU): ~28 GB/s

### HCCL Tuning Parameters

| Parameter | Default | Use Case |
|-----------|---------|----------|
| `HCCL_INTRA_ROCE_ENABLE` | 0 | Set to 1 for 16P Box16 setups |
| `HCCL_RDMA_TC` | 132 | Adjust when QoS mismatch with switch (must be multiple of 4) |
| `HCCL_RDMA_SL` | - | RDMA service level configuration |
| `HCCL_BUFFSIZE` | 200 | LLM: calculate per formula above |
