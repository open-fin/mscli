# Performance Optimization Map

> This file serves as the global navigation index for performance diagnosis
> and optimization on Ascend NPU platforms.

---

## Layer 1: Metrics

Performance analysis is grounded in 6 core metrics, each with clear formulas and thresholds.

### 1.1 Throughput

**Definition:** Training samples processed per unit time.

```
Throughput (samples/s) = (BS × N) / step_time
Throughput (tokens/s)  = (BS × N × seq_len) / step_time
```

- BS = batch_size per DP dimension
- N = cluster size (number of DP dimensions)
- NLP/LLM use tokens/s; CV uses samples/s

**Batch Size Optimization:**
1. Use binary search to find the maximum batch_size at memory limit
2. Recommended batch_size is a multiple of 16
3. Batch_size affects recomputation ratio, pipeline stage ratio, and TP sharding ratio

### 1.2 Step Time

**Definition:** Total time to complete one batch iteration (seconds/step).

**Pros:** Intuitive, quantifies operator/scheduling/communication impact.
**Cons:** Not normalized (affected by batch_size, gradient_accumulation), must be read alongside throughput.

**Conversion with throughput:**
```
step_time = (BS × N) / throughput
```

### 1.3 Linearity (Scaling Efficiency)

**Definition:** Ratio of actual speedup to ideal speedup.

```
Linearity = (N-card total throughput) / (N × single-card throughput)
          = (single-node multi-card throughput) / (single-node throughput)
          = actual throughput / theoretical throughput
```

| Threshold | Interpretation |
|-----------|---------------|
| > 0.8 | Normal; communication is NOT the bottleneck |
| < 0.8 | Communication IS the bottleneck (after ruling out IO and CPU factors) |

**Usage:** After ruling out IO and CPU factors, low linearity directly points to distributed communication bottleneck.

### 1.4 MFU (Machine FLOP Utilization)

**Definition:** Ratio of achieved TFLOPS to hardware peak TFLOPS.

```
MFU = FLOPs_per_step / (Peak_TFLOPS × num_devices × step_time)
```

> For per-layer Transformer FLOPs breakdown, Chinchilla formula, recomputation impact,
> and calculation methods, see [mfu-calculation.md](mfu-calculation.md).
> For hardware peak TFLOPS and full specs, see [hardware-specs.md](hardware-specs.md).

**MFU Levels:**

| Range | Level | Interpretation |
|-------|-------|---------------|
| < 20% | low | Severely underutilized; likely memory-bound or launch-overhead dominated |
| 20-40% | below_average | Common for unoptimized workloads |
| 40-60% | medium | Reasonable for most training workloads |
| 60-70% | good | Well-optimized |
| > 70% | excellent | Near device ceiling |

### 1.5 Communication Metrics

#### wait_ratio

```
wait_ratio = wait_time / total_op_execution_time
```

| Threshold | Interpretation |
|-----------|---------------|
| max_wait_ratio - min_wait_ratio > 0.2 | Slow card exists |

**Slow Card Identification:** The card with the shortest communication operator execution time is the bottleneck (because it makes other cards wait the longest).

#### DMA Bandwidth

```
sdma_bandwidth = sdma_data / sdma_transit_time    (intra-node)
rdma_bandwidth = rdma_data / rdma_transit_time    (inter-node)
```

- **SDMA (System DMA):** Intra-node NPU data transfer (via HCCS)
- **RDMA (Remote DMA):** Inter-node NPU data transfer (via RoCE network)

> For slow-card identification using wait_ratio, see [cluster-rank-diagnosis.md](cluster-rank-diagnosis.md).

### 1.6 Pipeline Parallel Efficiency

```
Pipeline Efficiency = (m × p) / (m + p - 1)
```

- m = number of micro-batches
- p = number of pipeline stages

| Schedule Mode | Characteristics |
|--------------|-----------------|
| GPipe | High bubble rate, high peak memory |
| 1F1B (PipeDream) | Steady-state 1-forward-1-backward, earlier memory release |
| Virtual Pipeline | Bubble ratio reduced to 1/v, where v = virtual stages |

**Optimization:** m >> p reduces bubble ratio.

---

## Layer 2: Diagnosis Decision Trees

> For detailed branch-by-branch diagnosis (when to use each branch, primary suspects, recommended sequence),
> see [bottleneck-signatures.md](bottleneck-signatures.md).

### 2.1 Single-Node Diagnosis Flow

```
Start
 │
 ├── Step 1: Collect Profiler Data
 │   └── Use compare_tools for performance decomposition
 │      (Compute / Communication / Scheduling)
 │
 ├── Step 2: Identify Bottleneck Domain
 │   ├── Compute > 50% → Compute bottleneck
 │   ├── Communication > 30% → Communication bottleneck
 │   └── Free/Scheduling > 20% → Scheduling bottleneck
 │
 ├── Step 3: Host-Bound vs Device-Bound Determination
 │   ├── Timeline API lines are vertical → Host-Bound
 │   │   (device idle, waiting for host dispatch)
 │   │   → Use flame graph to analyze CPU bottleneck
 │   │
 │   └── Timeline API lines are slanted → Device-Bound
 │       (host waiting for device completion)
 │       → Analyze compute/communication/memory
 │
 ├── Step 4: Operator Analysis
 │   ├── Check operator time fluctuation
 │   ├── Verify shape consistency across cards per step
 │   └── Locate problematic operators for deep analysis
 │
 ├── Step 5: Communication Analysis (multi-card)
 │   ├── Communication matrix: overall communication status
 │   ├── Specific operations: AllReduce/ReduceScatter bandwidth
 │   └── Communication volume estimation
 │
 ├── Step 6: Communication Duration Analysis
 │   ├── Blue bars = end-to-end time (communication + wait)
 │   ├── Shortest bar = slowest card (others waiting for it)
 │   └── Note: RDMA scenarios — wait time causes bandwidth calculation skew
 │
 ├── Step 7: Timeline Multi-Layer Analysis
 │   ├── Level 1: Python → CANN → Ascend Hardware → AI Core Freq
 │   ├── Level 2: HCCL layer communication analysis
 │   └── Overlap analysis
 │
 ├── Step 8: Fast/Slow Card Analysis (vertical + horizontal search)
 │   ├── Vertical: Ascend Hardware → CANN → Python
 │   └── Horizontal: trace back along time axis to find root cause
 │
 └── Step 9: Summarize → proceed to Layer 3 tuning methods
```

### 2.2 Cluster Diagnosis Flow

> For expert rules on slow-card diagnosis, see [cluster-rank-diagnosis.md](cluster-rank-diagnosis.md).
> For jitter thresholds, see [jitter-detection.md](jitter-detection.md).

```
Cluster Performance Issue
 │
 ├── Type A: Performance Degradation
 │   ├── Scale-up degradation (small → large cluster)
 │   │   → High probability: model sharding strategy issue
 │   │   → Check TP/PP/DP configuration
 │   │
 │   ├── Hardware change degradation
 │   │   → Compare before/after Profiler data
 │   │   → Identify degraded component
 │   │
 │   └── Long-term training degradation (gradual/sudden)
 │       → Check for memory leaks, thermal throttling
 │
 ├── Type B: Performance Fluctuation
 │   ├── Collect fluctuation steps
 │   ├── Compare fluctuation vs normal steps
 │   └── Locate core component (compute/communication/scheduling)
 │
 └── Type C: Slow Node vs Network Problem
     ├── Slow node: asymmetric performance (certain nodes are slow)
     │   → Compare API stats and operator stats
     │
     └── Network problem: all cards affected
         → Check PFC queue anomalies
         → Check network congestion
```

### 2.3 Parallel Strategy Selection Priority

```
Model Requirement Assessment
 │
 ├── Step 1: TP (Tensor Parallelism)
 │   ├── First choice; reduces memory while utilizing compute resources
 │   └── Limit: TP ≤ cards per machine (e.g., 8-card server TP ≤ 8)
 │
 ├── Step 2: PP (Pipeline Parallelism)
 │   ├── Use between machines when TP is insufficient
 │   └── Minimize PP count (more PP = more bubble waste)
 │
 ├── Step 3: DP (Data Parallelism)
 │   └── Enable when resources are abundant, distribute across machines
 │
 ├── Step 4: ZeRO Optimization
 │   ├── ZeRO-1: Shard optimizer states (compatible with PP)
 │   ├── ZeRO-2: Shard gradients (incompatible with PP)
 │   └── ZeRO-3: Shard weights (most complex, most communication)
 │
 └── Step 5: Recomputation
     └── Trade time for space; enables larger batch_size after enabling
```

**SP (Sequence Parallelism) Recommendation:** Enable SP alongside TP — no extra communication overhead.

**Overlap Optimization Key Insight:**
- Problem: Forward compute time < parameter AllGather time → poor overlap
- Solution: Increase micro batch size OR reduce single AG data volume (increase frequency)
- Poor overlap causes severe linearity degradation in large cluster scenarios

---

## Layer 3: Tuning Methods

> For actionable optimization suggestions with thresholds, code examples, and validation methods,
> see [optimization-knowledge-base.md](optimization-knowledge-base.md).

### 3.1 NPU Affinity Adaptation (Four-Step Method)

#### Step 1: Operator Fusion — Reduce redundant computation and submission count

See [Layer 4: Fused Operator Knowledge](#layer-4-fused-operator-knowledge)

#### Step 2: Eliminate Stream Synchronization

```python
# The following operations insert unnecessary synchronization in async streams:
tensor.item()       # Triggers h2d sync
tensor.reduce_all() # Triggers sync
torch.isfinite()    # Triggers sync

# Alternatives:
# - Prefer NPU-side logic, avoid pulling data back to Host
# - Use NPU-native equivalent operations
```

#### Step 3: Multi-Card Performance Consistency

- Problem: Fast cards waiting for slow cards
- Investigation: Compare API dispatch stats, operator stats
- Solution: Check CPU affinity, NUMA binding, ensure balanced load across cards

#### Step 4: CPU Optimization

- Move CPU computations to NPU (e.g., conditional logic, data preprocessing)
- Multi-process acceleration for CPU-side operations

### 3.2 Memory Optimization

#### Environment Variable Tuning

| Parameter | Recommended | Purpose |
|-----------|------------|---------|
| `torch_npu.npu.set_per_process_memory_fraction` | 0.95 | Limit process memory usage ratio (0-1) |
| `PYTORCH_NPU_ALLOC_CONF="garbage_collection_threshold:0.95"` | 0.95 | GC trigger threshold (memory usage percentage) |
| `PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:50"` | 50+ MB | Prevent large blocks from being split into fragments |
| `PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"` | True | Enable memory pool expansion for dynamic shapes |
| `MULTI_STREAM_MEMORY_REUSE=1` | 1 | Multi-stream memory reuse (compute stream reuses comm stream memory) |
| `HCCL_BUFFSIZE` | See formula | Reduce communication memory overhead |

Combined usage example:
```bash
export PYTORCH_NPU_ALLOC_CONF="garbage_collection_threshold:0.95,max_split_size_mb:50,expandable_segments:True"
export MULTI_STREAM_MEMORY_REUSE=1
```

#### Memory Snapshot Analysis

```python
# PyTorch memory snapshot
torch.cuda.memory._record_memory_history()
# ... run training ...
torch.cuda.memory._dump_snapshot("snapshot.pickle")

# Visualization
# python -m torch.cuda.memory._memory_viz viewer snapshot.pickle
```

**CANN-level Analysis:** Memory data under `PROF_XXX/` directory provides finer-grained memory breakdown.

### 3.3 Communication Optimization

#### HCCL Environment Variables

| Parameter | Default | Use Case |
|-----------|---------|----------|
| `HCCL_INTRA_ROCE_ENABLE=1` | 0 | 16P single-server: replace SDMA with RDMA between two 8P groups |
| `HCCL_RDMA_TC=<value>` | 132 | Adjust for QoS mismatch with switch; range 0-255 (must be multiple of 4) |
| `HCCL_RDMA_SL=<value>` | - | RDMA service level configuration |
| `HCCL_BUFFSIZE=<MB>` | 200 | Communication buffer size |

#### HCCL BUFFSIZE Calculation (LLM Scenario)

```
HCCL_BUFFSIZE = ceil(Micro_Batch_Size × Sequence_Length × Hidden_Size × dtype_size / (8 × 1024 × 1024))
```

Example: MBS=4, S=4096, H=4096, BF16 (2 bytes)
```
BUFFSIZE = ceil(4 × 4096 × 4096 × 2 / 8388608) = ceil(16) = 16 MB
```

#### Communication Architecture

```
Single-node 8P (Atlas 800):
  HCCS Full Mesh: Direct NPU interconnect (56 GB/s)
  PCIe: NPU → CPU (~28 GB/s)

Single-node 16P (Atlas 200T A2 Box16):
  Intra-group: HCCS (high speed)
  Inter-group: PCIe or RDMA (use RDMA when HCCL_INTRA_ROCE_ENABLE=1)

Multi-node:
  Intra-node: HCCS
  Inter-node: RDMA/RoCE (Ethernet)
```

### 3.4 Compilation Optimization

#### LTO (Link Time Optimization)

- **ThinLTO:** Recommended; lower runtime overhead and compilation time
- Benefits: Cross-file inlining, specialization, constant propagation, code elimination

#### PGO (Profile-Guided Optimization)

- Benefits: 10-30% performance improvement (data/compute-intensive scenarios)
- Workflow:
  1. Instrumented compilation → 2. Run to collect profile → 3. Merge profile → 4. Optimized compilation

#### Compiler Compatibility Matrix

| Python | PyTorch | torch_npu | Compatible |
|--------|---------|-----------|-----------|
| gcc | gcc | gcc | Yes |
| gcc | BiSheng | BiSheng | Yes |
| BiSheng | gcc | gcc | Yes |
| BiSheng | BiSheng | BiSheng | Yes |
| gcc | gcc | BiSheng | **No** |
| gcc | BiSheng | gcc | **No** |
| BiSheng | gcc | BiSheng | **No** |
| BiSheng | BiSheng | gcc | **No** |

**Key Rule:** Python/PyTorch/torch_npu must all use the same compiler (gcc or BiSheng).

### 3.5 Scheduling Optimization

| Optimization | Description |
|-------------|-------------|
| Python version | Use officially recommended version |
| Pipeline optimization | Reduce unnecessary synchronization points |
| Core binding | Use numactl/taskset to bind CPU cores |

### 3.6 IO Optimization (Data Loading)

| Optimization | Description |
|-------------|-------------|
| num_workers | Increase DataLoader parallel workers |
| pin_memory | Enable pinned memory for faster h2d transfer |
| prefetch | Prefetch data to reduce device wait |
| Caching | Cache dataset to memory or SSD |
| Reduce decode complexity | Optimize transform pipeline |

### 3.7 OS-Level Optimization

| Optimization | Description |
|-------------|-------------|
| High-performance memory allocator | Replace glibc malloc |
| Hugepage memory pool | Reduce page table overhead |
| malloc with hugepages | Direct hugepage allocation |
| tmpfs with hugepages | Shared memory using hugepages |
| glibc dynamic library hugepages | Load dynamic libraries into hugepages |

---

## Layer 4: Fused Operator Knowledge

> For corresponding optimization entries, see [optimization-knowledge-base.md](optimization-knowledge-base.md)
> NPU-AFFINITY-01 to NPU-AFFINITY-04 and COMP-03.

### 4.1 FlashAttentionScore

#### Replacement Paths (4 APIs)

| Original API | Replacement API |
|-------------|-----------------|
| `torch.nn.functional.scaled_dot_product_attention` | `torch_npu.npu_fusion_attention` |
| `flash_attn_func` (flash-attn library) | `torch_npu.npu_fusion_attention` |
| `flash_attn_varlen_func` (variable-length sequences) | `torch_npu.npu_fusion_attention` |
| `xformers.ops.memory_efficient_attention` | `torch_npu.npu_fusion_attention` |

#### Parameter Mapping

| Original Parameter | NPU Parameter |
|-------------------|--------------|
| `dropout` | `keep_prob = 1 - dropout` |
| `softmax_scale` | `scale` |
| `causal` | `atten_mask` (requires triangular mask construction) |
| `cu_seqlens_q/k` | `actual_seq_qlen/kvlen` (convert to host-side list) |

#### Supported Layouts

BNSD, BSND, BSH, SBH, TND

#### Constraints

- Hardware: Atlas A2 training series
- Data types: float16 / bfloat16
- Batch: 1 ~ 2K
- Head num ratio: must be integer
- Sequence length: 1 ~ 512K
- Embedding dimension: 1 ~ 512

### 4.2 MatmulAllReduce

```python
# Original code
output = torch.matmul(input, weight)
dist.all_reduce(output, op=ReduceOp.SUM)

# Optimized code
output = torch_npu.npu_mm_all_reduce_base(
    input, weight, hcomm_info,
    reduce_op="sum", comm_turn=0
)
```

**Applicable Scenarios:** TP (Tensor Parallelism) sharding scenarios.

### 4.3 Fused Optimizers

| Standard Optimizer | NPU Fused Optimizer |
|-------------------|-------------------|
| `torch.optim.SGD` | `torch_npu.optim.NpuFusedSGD` |
| `torch.optim.AdamW` | `torch_npu.optim.NpuFusedAdamW` |
| `torch.optim.Adadelta` | `torch_npu.optim.NpuFusedAdadelta` |
| `Lamb` | `torch_npu.optim.NpuFusedLamb` |
| `AdamP` | `torch_npu.optim.NpuFusedAdamP` |
| `BertAdam` | `torch_npu.optim.NpuFusedBertAdam` |
| `torch.optim.RMSprop` | `torch_npu.optim.NpuFusedRMSprop` |
| `RMSpropTF` | `torch_npu.optim.NpuFusedRMSpropTF` |

**Principle:** Reduces multiple h2d submission operations by unifying host-side computation before single device submission.
**Trade-off:** May use more memory (trades memory for performance).

### 4.4 Other Fused Operators

| Operator | Source |
|----------|--------|
| RotaryMul & RotaryMulGrad | Ascend PT documentation |
| RmsNorm & RmsNormGrad | Ascend PT documentation |
| ScaledMaskedSoftmax & ScaledMaskedSoftmaxGrad | Ascend PT documentation |
| SwiGlu | Ascend PT documentation |
| IndexPut replacement | Ascend PT documentation |
| Nonzero replacement | Ascend PT documentation |
| where replacement | Ascend PT documentation |

### 4.5 Affinity API Replacement

General principle:
- Check if NPU-native API alternatives exist for standard PyTorch operations
- Common optimization: `torch.topk` → NPU-optimized version, etc.


## Threshold Quick Reference

| Metric | Threshold | Interpretation |
|--------|-----------|---------------|
| Linearity | < 0.8 | Communication bottleneck |
| wait_ratio delta | > 0.2 | Slow card exists |
| Pipeline efficiency | m×p/(m+p-1) | PP efficiency ceiling |
| MFU | < 20% | Severely underutilized |
| MFU | > 70% | Excellent / near device ceiling |
| Communication ratio | > 30% | Communication bottleneck |
| Compute ratio | < 50% | Non-compute bottleneck |
| Free Time | > 10% | Host dispatch bottleneck |
| Step Time CV | > 15% | Jitter anomaly |
| Operator variance | > 10% | Anomaly |
| Memory fraction | 0.95 | Recommended value |
| GC threshold | 0.95 | Recommended starting value |
| Batch size | 16× multiple | Recommended default |
| HCCL BUFFSIZE | ceil(MBS×S×H×dtype/8MB) | LLM recommendation |
| Cube utilization | < 30% | Memory-bound |
| L2 hit rate | < 50% | Memory-bound |
| Pipeline stall | > 30% | Pipeline-bound |

---

## See Also

| Reference | Scope |
|-----------|-------|
| [mfu-calculation.md](mfu-calculation.md) | MFU formulas, per-layer FLOPs, calculation methods |
| [hardware-specs.md](hardware-specs.md) | Ascend TFLOPS, HBM, HCCS bandwidth, topology |
| [bottleneck-signatures.md](bottleneck-signatures.md) | Branch-by-branch diagnosis decision trees |
| [cluster-rank-diagnosis.md](cluster-rank-diagnosis.md) | Slow-card expert rules, topology, wait_ratio |
| [jitter-detection.md](jitter-detection.md) | Jitter types, CV thresholds, remediation |
| [aic-microarch-signatures.md](aic-microarch-signatures.md) | CUBE/VECTOR/L2 metrics, compute vs memory bound |
| [optimization-knowledge-base.md](optimization-knowledge-base.md) | Actionable optimizations with code examples |
| [validation-playbook.md](validation-playbook.md) | Before/after comparison methodology |
