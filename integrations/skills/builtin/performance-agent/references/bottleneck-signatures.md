# Bottleneck Signatures

Read only the branch that matches the dominant pattern in the trace.

## Branch A: Communication Dominates

Use when the trace shows:

- large `AllReduce`, `ReduceScatter`, `AllGather`, or similar collective slices
- long waits near backward end or update
- step tail growing with distributed scale
- linearity < 0.8 (indicates communication bottleneck after ruling out IO/CPU)

Primary suspects:

- communication not overlapped with computation
- too many small collectives
- poor bucketization or fusion
- synchronization inserted in the wrong place
- HCCL buffer misconfiguration (calculate HCCL_BUFFSIZE per LLM formula)

Recommended sequence:

1. Confirm whether the cost is inside backward, update, or step tail.
2. Check whether collectives are serialized or overlapped.
3. Check wait_ratio across ranks: delta > 0.2 indicates slow card.
4. Recommend one communication-focused change first.
5. Validate by comparing collective count, collective time, and step tail.

→ [optimization-knowledge-base.md](optimization-knowledge-base.md) COMM-01 to COMM-07, [analyze_communication_matrix.py](../scripts/analyze_communication_matrix.py)

## Branch B: Host Launch or Idle Gap Dominates

Use when the trace shows:

- large gaps between kernels
- low device utilization with active host-side work
- many small launches with poor packing
- Host-Bound: Timeline API lines are vertical (device idle, waiting for host dispatch)
- Device-Bound: Timeline API lines are slanted (host waiting for device completion)

Primary suspects:

- Python-side per-step overhead
- unnecessary sync points (tensor.item(), tensor.reduce_all(), torch.isfinite())
- too much PyNative-style host dispatch
- fragmented execution instead of graph-heavy execution

Recommended sequence:

1. Identify where the idle gap occurs.
2. Determine Host-Bound vs Device-Bound using timeline API line direction.
3. For Host-Bound: use flame graph to analyze CPU bottleneck.
4. Recommend one host-overhead reduction first.
5. Validate by comparing idle gap and kernel launch density.

→ [optimization-knowledge-base.md](optimization-knowledge-base.md) HOST-01, HOST-02, [detect_bound_type.py](../scripts/detect_bound_type.py)

## Branch C: Input Pipeline Stalls

Use when the trace shows:

- device idle periods before compute starts
- dataloader or input transform time dominating the step gap

Primary suspects:

- insufficient parallel loading
- slow decode or transform path
- missing prefetch or caching

Recommended sequence:

1. Confirm that compute is waiting on input.
2. Keep the first fix in the input path, not in kernels.
3. Validate with shorter pre-compute idle time and better utilization.

→ [optimization-knowledge-base.md](optimization-knowledge-base.md) INPUT-01

## Branch D: Graph Build or Recompilation Dominates

Use when the trace shows:

- large compile sections
- repeated graph build or recompile behavior
- latency dominated by setup rather than steady-state kernels

Primary suspects:

- unstable shapes
- control-flow changes that force recompilation
- repeated graph construction in the serving loop

Recommended sequence:

1. Separate first-run compile cost from steady-state latency.
2. Confirm whether recompilation repeats or only happens once.
3. Recommend one graph-stabilization change first.
4. Validate by comparing compile count and compile time.

→ [optimization-knowledge-base.md](optimization-knowledge-base.md) COMP-04

## Branch E: Compute Operator Hotspot

Use when the trace shows:

- one or a few operators dominating execution time
- little evidence of communication or input stalls

Primary suspects:

- inefficient operator choice
- missing fusion
- backend kernel path not ideal
- redundant computation

Recommended sequence:

1. Name the dominant operator or operator family.
2. Confirm it is truly dominant in time share.
3. Recommend one operator- or fusion-focused change first.
4. Validate by comparing operator time share after rerun.

→ [optimization-knowledge-base.md](optimization-knowledge-base.md) COMP-01 to COMP-03, [hotspot-prioritization.md](hotspot-prioritization.md)

## Branch F: Memory Pressure or Memory-Heavy Operators

Use when the trace shows:

- one or a few operators dominating peak memory
- memory spikes that cap batch size
- allocation pressure causing secondary slowdown

Primary suspects:

- activation-heavy modules
- excessive temporary tensors
- precision choice
- layout or batch-size effects
- missing recomputation or checkpointing where appropriate

Recommended sequence:

1. Name the memory-heavy operator or stage.
2. Distinguish steady high memory from short spikes.
3. Recommend one memory-focused change first.
4. Validate by comparing peak memory and the top memory-heavy stage.

→ [optimization-knowledge-base.md](optimization-knowledge-base.md) MEM-01 to MEM-03

## Branch G: Optimizer or Update Dominates

Use when the trace shows:

- update phase dominating the step
- regression appearing after distributed enablement
- optimizer time inflated by sync or communication

Primary suspects:

- communication surfacing inside update
- too many small reductions
- update-side synchronization
- genuine optimizer compute cost only after communication is ruled out

Recommended sequence:

1. Decide whether update cost is communication-heavy or compute-heavy.
2. Prefer a communication-focused fix first if distributed enablement caused
   the regression.
3. Validate with update-time share, collective time, and step tail.

Common Ascend dual-stack note:

- If the regression appears only after multi-card enablement in MindSpore or
  torch_npu, assume communication-heavy until the trace proves otherwise.

→ [optimization-knowledge-base.md](optimization-knowledge-base.md) COMM-01, CLUSTER-01

## Branch H: Low MFU

Use when the MFU estimate shows:

- MFU < 20%: hardware is severely underutilized
- MFU 20-40%: common for unoptimized workloads but has significant headroom

Primary suspects:

- too many small operators causing kernel launch overhead
- graph mode not enabled (PyNative / eager execution)
- memory-bound workload (not actually compute-limited)
- suboptimal backend kernel path
- CUBE unit underutilized while VECTOR unit overused
  (99.3%+ FLOPs should be on CUBE for GEMM-heavy models)

Recommended sequence:

1. Check if graph compilation is enabled (MindSpore: GRAPH_MODE, PyTorch: torch.compile).
2. Check the operator size distribution for excessive small ops (<10us).
3. Check if the model is memory-bound (high HBM bandwidth utilization, low Cube utilization).
4. Check AIC PMU metrics for Cube vs Vector utilization breakdown (see aic-microarch-signatures.md).
5. Recommend one MFU-focused change first.
6. Validate by comparing MFU and step time.

→ [mfu-calculation.md](mfu-calculation.md), [aic-microarch-signatures.md](aic-microarch-signatures.md)

## Branch I: Cluster Rank Imbalance

Use when cluster analysis shows:

- one or more ranks with significantly higher step times
- rank imbalance detected by detect_slow_ranks.py
- wait_ratio delta > 0.2 across ranks

Primary suspects (see cluster-rank-diagnosis.md for details):

- Host dispatch bottleneck: Free Time >10% on the slow rank
- Compute imbalance: Compute Time significantly above mean
- Communication bottleneck: slow inter-node links
- Slow node vs network: asymmetric performance = slow node, all cards affected = network

Recommended sequence:

1. Identify the bottleneck type from detect_slow_ranks.py output.
2. Classify degradation type: scale-up / hardware change / long-term training.
3. For scale-up degradation: high probability is model sharding strategy issue.
4. For host_dispatch: check CPU affinity and NUMA binding.
5. For compute: compare operator stats between slow and fast ranks.
6. For communication: check HCCS/RDMA link health.
7. Validate by comparing per-rank step times after fix.

→ [cluster-rank-diagnosis.md](cluster-rank-diagnosis.md), [detect_slow_ranks.py](../scripts/detect_slow_ranks.py), [calculate_linearity.py](../scripts/calculate_linearity.py)

## Branch J: Performance Jitter

Use when jitter analysis shows:

- Step time CV >15%: significant variance
- Cross-rank alignment skew >5ms
- Outlier steps detected

Primary suspects:

- dynamic shapes causing recompilation
- Python GC pauses
- OS CPU scheduling without affinity
- variable-length input sequences

Recommended sequence:

1. Check if step time variance is consistent or intermittent.
2. Correlate outliers with specific phases (warmup, data loading, etc.).
3. Recommend one jitter-reduction change first (e.g., pad sequences, set CPU affinity).
4. Validate by comparing step time CV after fix.

→ [jitter-detection.md](jitter-detection.md), [analyze_jitter.py](../scripts/analyze_jitter.py)
