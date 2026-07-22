# AIC Microarchitecture Bottleneck Signatures

Read this file when AIC PMU metrics are available (collected via
`msprof op --aic-metrics`) for deeper operator-level bottleneck analysis.

## Background

AI Core (AIC) is the compute unit inside Ascend NPUs. It contains:
- **Cube Unit**: Matrix multiplication engine (primary compute)
- **Vector Unit**: Element-wise operations (activation, normalization)
- **Scalar Unit**: Control flow and scalar operations

AIC PMU (Performance Monitoring Unit) events provide hardware-level metrics
for each operator execution.

## Metric Definitions

| Metric | Description | Healthy Range |
|--------|-------------|---------------|
| Cube Utilization | Percentage of cycles the Cube unit is active | >60% for matmul-heavy ops |
| Vector Utilization | Percentage of cycles the Vector unit is active | Depends on op type |
| L2 Hit Rate | Percentage of L2 cache accesses that hit | >80% for compute-bound |
| Pipeline Utilization | Overall pipeline usage efficiency | >70% |
| Stall Rate | Percentage of cycles stalled due to dependencies | <20% |

## Bottleneck Classification

### Compute Bound

**Signatures:**
- Cube utilization > 70%
- L2 hit rate normal (>80%)
- Stall rate low (<15%)

**Implication:** The operator is using the compute hardware efficiently.
Optimization should focus on reducing the total amount of computation (fusion,
removing redundant ops) rather than improving hardware utilization.

**Recommended Actions:**
- Operator fusion to reduce kernel launch overhead
- Remove redundant computation
- Check if the operator can be replaced by a more efficient variant

### Memory Bound

**Signatures:**
- Cube utilization < 30% (often much lower)
- L2 hit rate < 50%
- High L2 read/write bandwidth utilization

**Implication:** The operator spends more time waiting for data than computing.
Data access patterns are not cache-friendly.

**Recommended Actions:**
- Optimize tiling strategy to improve data locality
- Adjust data layout (e.g., from NCHW to NC1HWC0 for Ascend)
- Increase batch size to amortize memory access overhead
- Use Ascend-optimized data formats (5D formats)

### Pipeline Bound

**Signatures:**
- Stall rate > 30%
- Resource conflict rate high
- Pipeline utilization low despite Cube and memory metrics being reasonable

**Implication:** The operator's instruction schedule has dependencies that cause
pipeline stalls. Instructions cannot be issued in parallel.

**Recommended Actions:**
- Instruction scheduling optimization
- Increase independent operations between dependent instructions
- Consider operator decomposition to reduce dependency chains

### Mixed Bound

**Signatures:**
- Multiple metrics simultaneously in unhealthy ranges
- No single dominant bottleneck

**Implication:** The operator faces multiple constraints simultaneously.

**Recommended Actions:**
- Address the most severe metric first
- Re-evaluate after each fix to see if the bottleneck type shifts

## Severity Levels

| Level | Criteria |
|-------|----------|
| critical | Cube utilization < 10% OR stall rate > 50% |
| high | Cube utilization < 30% OR L2 hit rate < 50% |
| medium | Cube utilization < 60% OR stall rate > 20% |
| low | Minor optimization opportunities |

## Data Collection

To collect AIC metrics:

```bash
# Framework profiler (MindSpore)
profiler = Profiler(aic_metrics="CubeUtilization,PipeUtilization,L2Cache")

# msprof CLI
msprof op --aic-metrics --output=/path/to/output --application="python train.py"
```

Note: AIC metrics collection adds some overhead. Use it for detailed
investigation, not for routine performance monitoring.

## See Also

- [bottleneck-signatures.md](bottleneck-signatures.md) Branch E (Compute Hotspot) and Branch H (Low MFU) — when to trigger AIC-level analysis
- [optimization-knowledge-base.md](optimization-knowledge-base.md) COMP-01 to COMP-03 — compute optimization actions
