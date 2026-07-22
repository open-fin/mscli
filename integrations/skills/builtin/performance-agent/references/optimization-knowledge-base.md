# Optimization Knowledge Base

Actionable optimization suggestions organized by bottleneck category.
Each suggestion includes trigger thresholds, concrete actions, code/config
examples, and expected benefits.

## Related References

- [performance-optimization-map.md](performance-optimization-map.md) — Tuning methods overview (Layer 3) and fused operator knowledge (Layer 4)
- [bottleneck-signatures.md](bottleneck-signatures.md) — Diagnosis decision trees per bottleneck branch
- [validation-playbook.md](validation-playbook.md) — Before/after validation methodology

## Communication

### COMM-01: Communication Overhead Too High (>30%)

- **Threshold:** comm_ratio > 0.30
- **Priority:** HIGH
- **Expected Benefit:** 10-30% training speed improvement
- **Actions:**
  1. Check if TP/DP/PP ratio is balanced for the workload
  2. Enable gradient accumulation to reduce DP communication frequency
  3. Enable communication-computation overlap
- **Config (Megatron):**
  ```bash
  --overlap-grad-reduce --overlap-param-gather
  --gradient-accumulation-steps 4
  ```
- **Config (MindSpore):**
  ```yaml
  use_parallel_optimizer: true
  overlap_grad_reduce: true
  gradient_accumulation_steps: 4
  ```
- **Validation:** Compare comm_ratio, collective count, step tail time after change

### COMM-02: Low Communication Overlap Ratio (<50%)

- **Threshold:** overlap_ratio < 0.50
- **Priority:** HIGH
- **Expected Benefit:** 10-25% training speed improvement
- **Actions:**
  1. Enable gradient reduce overlap
  2. Enable parameter gather overlap
  3. Increase micro batch size to create more computation for overlap
- **Config (Megatron):**
  ```bash
  --overlap-grad-reduce --overlap-param-gather
  ```
- **Config (MindSpore):**
  ```yaml
  use_parallel_optimizer: true
  overlap_grad_reduce: true
  ```
- **Validation:** Compare overlap_ratio and step_time_ms after change

### COMM-03: Excessive Small Collectives

- **Threshold:** collective_count_per_step > 100 AND avg_collective_size < 1MB
- **Priority:** MEDIUM
- **Expected Benefit:** 5-15% training speed improvement
- **Actions:**
  1. Increase bucket size for gradient all-reduce
  2. Check if ZeRO-3 is causing excessive small-packet communication
  3. Consider switching to ZeRO-1 or ZeRO-2 for small models
- **Config (Megatron):**
  ```bash
  --overlap-grad-reduce
  --no-gradient-scaler-bucket-size 256
  ```
- **Validation:** Compare collective count and avg collective size

### COMM-04: Slow Inter-Node Links

- **Threshold:** Inter-node bandwidth utilization < 40% of theoretical
- **Priority:** HIGH
- **Expected Benefit:** 15-40% multi-node speedup
- **Actions:**
  1. Check RDMA/RoCE configuration
  2. Verify network topology and switch configuration
  3. Consider using DP for inter-node (less communication) and TP for intra-node
- **Validation:** Compare inter-node bandwidth utilization

### COMM-05: HCCL Buffer Misconfiguration

- **Threshold:** Communication bandwidth below expected for data volume
- **Priority:** MEDIUM
- **Expected Benefit:** 5-20% communication speed improvement
- **Actions:**
  1. Calculate recommended HCCL_BUFFSIZE for LLM:
     ```
     HCCL_BUFFSIZE = ceil(MBS × S × H × dtype_size / (8 × 1024 × 1024))
     ```
  2. Set environment variable: `export HCCL_BUFFSIZE=<calculated_value>`
  3. For memory-constrained scenarios, reduce below default (200MB)
  4. For communication-intensive scenarios, increase above calculated value
- **Validation:** Compare communication bandwidth after change

### COMM-06: Intra-Node RoCE Not Enabled (16P)

- **Threshold:** 16P single-server deployment with low cross-group bandwidth
- **Priority:** MEDIUM
- **Expected Benefit:** Significant cross-group bandwidth improvement
- **Actions:**
  1. Enable intra-node RoCE: `export HCCL_INTRA_ROCE_ENABLE=1`
  2. Applicable to Atlas 200T A2 Box16 heterogeneous subracks
  3. Replaces SDMA with RDMA links between two 8P groups
- **Validation:** Compare cross-group communication bandwidth

### COMM-07: RDMA Traffic Class Mismatch

- **Threshold:** RDMA communication bandwidth degradation with specific switch
- **Priority:** LOW
- **Expected Benefit:** Restore expected RDMA bandwidth
- **Actions:**
  1. Configure RDMA TC: `export HCCL_RDMA_TC=<value>` (0-255, must be multiple of 4)
  2. Default is 132 (DSCP=33), adjust for QoS mismatch with switch
  3. Optionally configure SL: `export HCCL_RDMA_SL=<value>`
- **Validation:** Compare RDMA bandwidth after TC adjustment

## Compute

### COMP-01: Compute Time Ratio Too Low (<50%)

- **Threshold:** compute_ratio < 0.50
- **Priority:** HIGH
- **Expected Benefit:** 20-40% training speed improvement
- **Actions:**
  1. Identify what is consuming non-compute time (comm, idle, data loading)
  2. Address the dominant non-compute bottleneck first
  3. Increase batch size to amortize fixed overhead
- **Validation:** Compare compute_ratio and step_time_ms

### COMP-02: MFU Below 20%

- **Threshold:** estimated_mfu < 0.20
- **Priority:** HIGH
- **Expected Benefit:** Significant improvement (hardware is severely underutilized)
- **Actions:**
  1. Check if operator fusion is enabled
  2. Check for excessive small operators causing launch overhead
  3. Enable graph compilation (MindSpore GRAPH_MODE, torch.compile)
  4. Check if the model is memory-bound rather than compute-bound
- **Code (MindSpore):**
  ```python
  import mindspore as ms
  ms.set_context(mode=ms.GRAPH_MODE)
  ```
- **Code (PyTorch):**
  ```python
  model = torch.compile(model)
  ```
- **Validation:** Compare MFU after enabling compilation

### COMP-03: Operator Hotspot (Single Op >35% of Step Time)

- **Threshold:** top_operator_share > 35%
- **Priority:** HIGH
- **Expected Benefit:** 10-30% step time reduction
- **Actions:**
  1. Identify the specific operator
  2. Check if a fused variant exists (FlashAttention, FusedLayerNorm, etc.)
  3. Check if backend kernel path is optimal for Ascend
  4. Consider custom operator implementation
- **Validation:** Compare hotspot operator share and step time

## Memory

### MEM-01: Memory Pressure Too High

- **Threshold:** memory_pressure == "high" OR peak_memory > 90% of device capacity
- **Priority:** HIGH
- **Expected Benefit:** Enable larger batch sizes, improve throughput
- **Actions:**
  1. Enable gradient checkpointing / activation recomputation
  2. Switch from FP32 to BF16 or FP16
  3. Reduce batch size or sequence length
  4. Check for memory leaks (growing peak memory across steps)
- **Code (MindSpore):**
  ```python
  from mindspore import nn
  net = nn.Recompute(net)  # Enable gradient checkpointing
  ```
- **Code (PyTorch):**
  ```python
  from torch.utils.checkpoint import checkpoint
  output = checkpoint(module, *inputs)
  ```
- **Validation:** Compare peak memory and batch size headroom

### MEM-02: Memory Fragmentation

- **Threshold:** fragmentation detected OR many small allocations
- **Priority:** MEDIUM
- **Expected Benefit:** 5-10% memory reduction
- **Actions:**
  1. Pre-allocate tensors with known sizes
  2. Use memory pool optimization
  3. Reduce temporary tensor creation
- **Validation:** Compare fragmentation ratio

### MEM-03: NPU Memory Configuration Suboptimal

- **Threshold:** Frequent OOM, GC-induced jitter, or high fragmentation
- **Priority:** MEDIUM
- **Expected Benefit:** 5-15% memory efficiency improvement
- **Actions:**
  1. Set memory fraction to limit process memory:
     ```python
     torch_npu.npu.set_per_process_memory_fraction(0.95)
     ```
  2. Configure GC threshold to reduce jitter:
     ```bash
     export PYTORCH_NPU_ALLOC_CONF="garbage_collection_threshold:0.95"
     ```
  3. Set max split size to prevent fragmentation:
     ```bash
     export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:50"
     ```
  4. Enable expandable segments for dynamic shapes:
     ```bash
     export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
     ```
  5. Enable multi-stream memory reuse:
     ```bash
     export MULTI_STREAM_MEMORY_REUSE=1
     ```
- **Combined Config:**
  ```bash
  export PYTORCH_NPU_ALLOC_CONF="garbage_collection_threshold:0.95,max_split_size_mb:50,expandable_segments:True"
  export MULTI_STREAM_MEMORY_REUSE=1
  ```
- **Validation:** Compare OOM frequency, GC pause time, and fragmentation ratio

## Input Pipeline

### INPUT-01: Data Loading Bottleneck (Queue Empty >20%)

- **Threshold:** queue_empty_percent > 20 OR bottleneck_detected == true
- **Priority:** HIGH
- **Expected Benefit:** 30-80% idle time reduction
- **Actions:**
  1. Increase DataLoader num_workers
  2. Enable pin_memory and prefetch
  3. Cache dataset to memory or SSD
  4. Reduce decode/transform complexity
- **Code (MindSpore):**
  ```python
  dataset = dataset.batch(batch_size,
      num_parallel_workers=8,
      drop_remainder=True)
  ```
- **Code (PyTorch):**
  ```python
  dataloader = DataLoader(dataset,
      batch_size=batch_size,
      num_workers=8,
      pin_memory=True,
      prefetch_factor=2)
  ```
- **Validation:** Compare queue_empty_percent and pre-compute idle time

## Host/Framework Overhead

### HOST-01: Excessive Host Launch Overhead

- **Threshold:** host_overhead share > 20% of step time
- **Priority:** HIGH
- **Expected Benefit:** 20-50% latency reduction
- **Actions:**
  1. Enable graph compilation (GRAPH_MODE / torch.compile)
  2. Reduce Python-side per-step overhead
  3. Check for unnecessary sync points
- **Code (MindSpore):**
  ```python
  ms.set_context(mode=ms.GRAPH_MODE)
  ```
- **Code (PyTorch):**
  ```python
  model = torch.compile(model, mode="max-autotune")
  ```
- **Validation:** Compare host_overhead share and kernel launch density

### HOST-02: Graph Recompilation

- **Threshold:** graph_compile share > 15% of step time (after warmup)
- **Priority:** MEDIUM
- **Expected Benefit:** Eliminate repeated compile cost
- **Actions:**
  1. Stabilize input shapes (pad to fixed sizes)
  2. Avoid dynamic control flow in model
  3. Separate warmup compile cost from steady-state measurement
- **Validation:** Compare compile count and compile time

### HOST-03: Excessive Operator Dispatch Overhead

- **Threshold:** device_util < 50% AND high_freq_small_ops > 25% of CPU calls
  AND avg_dispatch < 40us
- **Priority:** CRITICAL
- **Expected Benefit:** 2-3x throughput improvement
- **Diagnosis:** Run `scripts/analyze_host_bound_root_cause.py` to identify:
  - dispatch density (ops/sec, avg dispatch time)
  - high-frequency small operators (< 50us avg)
  - category breakdown of CPU time
- **Actions (Priority Order):**
  1. **Build fused operators** — identify fusible operator groups:
     | Pattern | Fused Op | Condition |
     |---------|----------|-----------|
     | `silu + mul + matmul(gate) + matmul(up)` | `aclnnSwiGLU` / `MatmulSwiGLU` | SwiGLU MLP |
     | `LayerNorm/RMSNorm + SiLU/GELU` | `aclnnLayerNormSiLU` / `aclnnRMSNormSilu` | Any transformer |
     | `matmul + all_reduce` | `torch_npu.npu_mm_all_reduce_base` | TP scenario |
     | `AdamW per-param mul+add` | `NpuFusedAdamW` | Any optimizer |
     | Multiple `copy_/view/reshape` | Graph mode elimination | Data movement |
     | Multiple `empty_tensor` | Graph mode memory pool | Tensor allocation |
  2. **Enable graph compilation** — most effective single optimization:
     ```python
     model = torch.compile(model, mode="max-autotune")
     # or
     model = torch.compile(model, backend="npu")
     ```
  3. **Reduce per-op Python overhead** — avoid Python-level loops over parameters
- **Validation:** Compare device_util, ops/sec, step_time_ms

### HOST-04: Framework/Autograd Overhead Too High

- **Threshold:** checkpoint + autograd_engine + backward > 35% of CPU time
- **Priority:** HIGH
- **Expected Benefit:** 20-40% CPU time reduction
- **Diagnosis:** Run `scripts/analyze_host_bound_root_cause.py` and check
  category breakdown for `checkpoint`, `autograd_engine`, `backward` shares
- **Actions:**
  1. **Reduce gradient checkpointing scope:**
     - Apply only to compute-heavy layers (not embedding, not output)
     - Use selective checkpointing: only checkpoint layers where activation
       memory savings justify recomputation cost
     ```python
     # Selective: only checkpoint transformer blocks, not embedding
     from torch.utils.checkpoint import checkpoint
     for layer in transformer_layers:
         x = checkpoint(layer, x, use_reentrant=False)
     ```
  2. **Enable graph mode** — fuses autograd graph into compiled backward:
     ```python
     model = torch.compile(model, mode="max-autotune")
     ```
  3. **Check for unnecessary recomputation:**
     - Verify checkpoint segments are not too small (single layer per segment)
     - Consider grouping multiple layers per checkpoint segment
- **Validation:** Compare checkpoint_time_ms, autograd_time_ms, step_time_ms

### HOST-05: Slow IO or Data Operations

- **Threshold:** Any individual IO/data op > 50ms, or io_save+dataloader
  time > 5% of step time during device-idle gaps
- **Priority:** HIGH (if > 500ms) / MEDIUM (if > 50ms)
- **Expected Benefit:** Eliminate sync stalls, reduce idle gaps
- **Diagnosis:** Run `scripts/analyze_host_bound_root_cause.py` and check
  `slow_individual_ops` and `gap_cpu_by_category` for io_save, io_log,
  dataloader entries
- **Actions by type:**

  **DataLoader:**
  ```python
  dataloader = DataLoader(dataset,
      batch_size=batch_size,
      num_workers=8,           # increase from default
      pin_memory=True,         # enable pin memory for NPU
      prefetch_factor=2,       # prefetch batches
      persistent_workers=True) # avoid worker restart overhead
  ```

  **Checkpoint saving:**
  ```python
  # Reduce frequency
  if global_step % 1000 == 0:  # not every step
      torch.save(state_dict, path)

  # Use async save if available
  # torch_npu may support async checkpoint via dedicated thread
  ```

  **Logging/TensorBoard:**
  ```python
  # Reduce sync-inducing scalar logging
  if global_step % 100 == 0:
      writer.add_scalar('loss', loss.item())  # .item() causes sync
  # Use async logging or buffer metrics
  ```

  **Gradient clipping:**
  ```python
  # _foreach_norm causes sync stall (observed up to 522ms)
  # Reduce frequency
  if global_step % 10 == 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

- **Validation:** Compare max_duration of IO/data ops, idle_gap_ms

### HOST-06: Excessive Dtype Cast Overhead

- **Threshold:** dtype_cast total > 3% of CPU time
- **Priority:** MEDIUM
- **Expected Benefit:** 5-10% throughput improvement
- **Diagnosis:** Check `aten::to`, `aten::_to_copy`, `_npu_dtype_cast` call
  counts and total time in host-bound analysis
- **Actions:**
  1. **Unify dtype to BF16:**
     ```python
     # Replace FP16 with BF16 (no need for loss scaler, fewer casts)
     model = model.to(torch.bfloat16)
     # Use autocast instead of manual casts
     with torch.autocast(device_type='npu', dtype=torch.bfloat16):
         output = model(input)
     ```
  2. **Verify AMP config:**
     ```python
     # Wrong: manual mixed precision causes FP16↔FP32 round trips
     # Right: use autocast which minimizes casts
     from torch_npu.npu.amp import autocast
     with autocast():
         output = model(input)
     ```
  3. **Check for unnecessary cast chains:**
     - If using FP16, consider switching to BF16 (Ascend 910B supports native BF16)
     - If `aten::to` appears with high call count, trace source of dtype mismatch
- **Validation:** Compare dtype_cast total time, cast op count

## Cluster / Distributed

### CLUSTER-01: Rank Imbalance (Slow Card Detected)

- **Threshold:** slow_ranks detected by detect_slow_ranks.py
- **Priority:** HIGH
- **Expected Benefit:** 10-30% cluster speedup
- **Actions:**
  1. Identify bottleneck type (host_dispatch / compute / communication)
  2. For host_dispatch: check CPU affinity, NUMA binding
  3. For compute: compare operator stats between slow and fast ranks
  4. For communication: check link health and topology
- **Validation:** Compare per-rank step times after fix

### CLUSTER-02: Pipeline Bubble Too Large

- **Threshold:** PP bubble ratio exceeds theoretical (pp_size-1)/(pp_size-1+micro_batches)
- **Priority:** MEDIUM
- **Expected Benefit:** 5-15% step time reduction
- **Actions:**
  1. Increase number of micro-batches
  2. Enable interleaved pipeline schedule
  3. Balance pipeline stage computation
- **Config (Megatron):**
  ```bash
  --micro-batch-size 1
  --global-batch-size 512
  --virtual-pipeline-model-parallel-size 2
  ```
- **Validation:** Compare bubble ratio and step time

## Jitter

### JITTER-01: Step Time Variance Too High (CV >15%)

- **Threshold:** step_time_cv > 0.15
- **Priority:** MEDIUM
- **Expected Benefit:** More predictable throughput
- **Actions:**
  1. Pad variable-length sequences to fixed sizes
  2. Enable CPU affinity (numactl / taskset)
  3. Check for GC pauses (reduce Python object creation in training loop)
  4. Disable CPU frequency scaling (set to performance governor)
- **Validation:** Compare step time CV after fixes

## NPU Affinity Adaptation

### NPU-AFFINITY-01: Replace Standard Optimizer with Fused Optimizer

- **Threshold:** Optimizer step time > 10% of step time, or many small h2d operations
- **Priority:** MEDIUM
- **Expected Benefit:** 5-15% step time reduction
- **Actions:**
  Replace standard optimizer with NPU fused variant:

  | Standard | NPU Fused |
  |----------|-----------|
  | `torch.optim.SGD` | `torch_npu.optim.NpuFusedSGD` |
  | `torch.optim.AdamW` | `torch_npu.optim.NpuFusedAdamW` |
  | `torch.optim.Adadelta` | `torch_npu.optim.NpuFusedAdadelta` |
  | `Lamb` | `torch_npu.optim.NpuFusedLamb` |
  | `AdamP` | `torch_npu.optim.NpuFusedAdamP` |
  | `BertAdam` | `torch_npu.optim.NpuFusedBertAdam` |
  | `torch.optim.RMSprop` | `torch_npu.optim.NpuFusedRMSprop` |
  | `RMSpropTF` | `torch_npu.optim.NpuFusedRMSpropTF` |

- **Code (PyTorch):**
  ```python
  # Before
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  # After
  optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=1e-4)
  ```
- **Note:** May increase memory usage (trades memory for performance).
- **Validation:** Compare optimizer step time and h2d operation count

### NPU-AFFINITY-02: Replace matmul+all_reduce with MatmulAllReduce

- **Threshold:** TP scenario with separate matmul and all_reduce
- **Priority:** HIGH
- **Expected Benefit:** 10-20% TP communication reduction
- **Actions:**
  ```python
  # Before
  output = torch.matmul(input, weight)
  dist.all_reduce(output, op=ReduceOp.SUM)

  # After
  output = torch_npu.npu_mm_all_reduce_base(
      input, weight, hcomm_info,
      reduce_op="sum", comm_turn=0
  )
  ```
- **Constraint:** Atlas A2 training series only, TP splitting scenarios
- **Validation:** Compare matmul + all_reduce combined time

### NPU-AFFINITY-03: Replace Attention with FlashAttention

- **Threshold:** Attention operators > 20% of compute time
- **Priority:** HIGH
- **Expected Benefit:** 20-40% attention time reduction
- **Actions:**
  Replace with `torch_npu.npu_fusion_attention`:

  | Original API | Parameter Mapping |
  |-------------|-------------------|
  | `F.scaled_dot_product_attention` | Direct replacement |
  | `flash_attn_func` | dropout→keep_prob, softmax_scale→scale |
  | `flash_attn_varlen_func` | cu_seqlens→actual_seq_qlen/kvlen (host list) |
  | `xformers.ops.memory_efficient_attention` | Direct replacement |

- **Constraint:** Atlas A2 training series, float16/bfloat16 only
- **Supported Layouts:** BNSD, BSND, BSH, SBH, TND
- **Validation:** Compare attention operator time

### NPU-AFFINITY-04: Eliminate Unnecessary Stream Synchronization

- **Threshold:** Host-side sync operations causing device idle gaps
- **Priority:** MEDIUM
- **Expected Benefit:** 5-10% latency reduction
- **Actions:**
  Remove or replace sync-inducing operations in training loop:
  - `tensor.item()` → avoid in hot path, use NPU-side logic
  - `tensor.reduce_all()` → use NPU-native alternatives
  - `torch.isfinite()` → use NPU-side checks
- **Validation:** Compare Free Time percentage and kernel launch density

## Compilation

### COMP-04: Enable LTO/PGO Compilation

- **Threshold:** Host-side overhead > 30% AND framework-level optimization exhausted
- **Priority:** LOW
- **Expected Benefit:** 10-30% host-side speedup
- **Actions:**
  1. Enable ThinLTO for PyTorch build:
     ```bash
     export CMAKE_C_FLAGS="-flto=thin -fuse-ld=lld"
     export CMAKE_CXX_FLAGS="-flto=thin -fuse-ld=lld"
     export CC=clang && export CXX=clang++
     ```
  2. For maximum benefit, add PGO (Profile-Guided Optimization):
     - Stage 1: Build with `-fprofile-generate`
     - Stage 2: Run representative workload to collect profile
     - Stage 3: Rebuild with `-fprofile-use`
  3. Ensure compiler consistency: Python/PyTorch/torch_npu must all use
     either gcc or 毕昇 (BiSheng) — never mix.
- **Validation:** Compare host-side overhead and overall step time
