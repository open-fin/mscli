# Cluster Rank Diagnosis

Read this file when analyzing multi-rank (cluster) profiling data to identify
slow cards and diagnose the root cause.

## Expert Rules

These rules are derived from the Huawei Ascend official diagnosis methodology.
Do not rely on single metrics in isolation; always cross-reference.

### Rule 1: Host Dispatch Bottleneck (Pseudo-Fast Card)

**Pattern:**
- One rank shows extremely long `Free Time` (>10% of step time, or significantly
  above the cluster mean)
- That rank's `Compute` and `Communication` times are anomalously short

**Diagnosis:**
This rank is NOT a fast card. It is a slow card causing cluster blocking.
The CPU dispatch is slow, causing NPU starvation (producing large Free Time).
When it finally initiates communication, other ranks have been waiting, so its
communication completes quickly.

**Action:**
- Compare API dispatch stats (launch, aclrtSynchronizeDevice) between slow and fast ranks
- Check CPU affinity (numactl binding) for the affected rank
- Check for CPU-bound processes competing for the same NUMA node
- Compare kernel launch density on the slow rank vs fast ranks

### Rule 2: Pure Compute Slow Card

**Pattern:**
- All ranks have short, uniform Free Time
- One rank's `Compute Time` is significantly larger than the cluster mean

**Diagnosis:**
Compute-bound slow card. Two sub-cases:
1. **Load imbalance**: operator call counts differ → check data partitioning
2. **Hardware degradation**: same call counts but higher avg time → check for
   dynamic shapes, hardware issues, or thermal throttling

**Action:**
- Compare operator stats between slow and fast ranks
- If counts differ: check data sharding and pipeline stage partition
- If avg times differ: check for dynamic shape triggers, operator recompilation,
  or hardware issues on the affected card

### Rule 3: Communication / Slow Link

**Pattern:**
- Communication bandwidth far below theoretical values (e.g., SDMA < 2 GB/s)
- Bandwidth utilization varies significantly across rank pairs

**Diagnosis:**
- Small packet communication (e.g., ZeRO3 sharding too fine-grained)
- SDMA address misalignment
- Hardware link degradation (rare but possible)

**Action:**
- Increase communication bucket size to reduce small-packet overhead
- Check ZeRO stage configuration; ZeRO-3 with small per-rank shards generates
  many small collectives
- Verify HCCS/RDMA link health using link diagnostics

## Detection Method

1. **Collect**: gather step_trace_time.csv from all ranks in the cluster
2. **Compare**: compute per-rank Compute/Communication/Free time percentages
3. **Detect outliers**: use 3-sigma rule or Dixon Q-test
4. **Classify**: apply expert rules above to the outlier rank(s)
5. **Report**: state the bottleneck type, affected rank, and specific actions

## Common Ascend Topology

```
Atlas 800 (8x 910B):
  Ring 0: NPU 0 - 1 - 2 - 3  (HCCS 56 GB/s)
  Ring 1: NPU 4 - 5 - 6 - 7  (HCCS 56 GB/s)
  Ring 0 <-> Ring 1: PCIe/Host (~28 GB/s, ~1/2 bandwidth)
```

Cross-ring communication is significantly slower than intra-ring. Ranks that
straddle ring boundaries (e.g., Rank 3 and Rank 4 in an AllReduce) will show
higher communication latency.

## Multi-Machine Considerations

- Inter-node communication uses RDMA/RoCE, typically much lower bandwidth than
  intra-node HCCS
- Check for slow inter-node links separately from intra-node analysis
- Network congestion on the switching fabric can cause correlated slowdowns
  across multiple ranks simultaneously

## See Also

- [bottleneck-signatures.md](bottleneck-signatures.md) Branch I (Cluster Rank Imbalance) — when to trigger cluster analysis
- [jitter-detection.md](jitter-detection.md) — cross-rank alignment skew and jitter analysis
- [hardware-specs.md](hardware-specs.md) — HCCS topology and bandwidth reference
