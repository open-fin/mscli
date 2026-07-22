# Jitter Detection Reference

Read this file when analyzing step-time variance and performance instability.

## Overview

Jitter refers to per-step performance variance that causes inconsistent
throughput. In distributed training, jitter on one rank can slow the entire
cluster due to synchronization barriers.

## Jitter Types

### Compute Jitter

| Metric | Detection | Threshold |
|--------|-----------|-----------|
| CV (Coefficient of Variation) | std(step_compute_times) / mean(step_compute_times) | >10% = warning, >20% = critical |
| Step time CV | std(step_times) / mean(step_times) | >15% = warning |

**Common Causes:**
- Dynamic shapes causing operator recompilation
- Python GC pauses during training
- OS CPU scheduling jitter (especially without CPU affinity)
- Thermal throttling on NPU
- Variable-length sequences in NLP workloads

**Recommended Actions:**
- Pad sequences to fixed lengths where possible
- Set `torch.cuda.set_device()` or equivalent to pin processes
- Use `numactl` or `taskset` for CPU binding
- Pre-compile graphs when possible (MindSpore GRAPH_MODE, torch.compile)

### Communication Jitter

| Metric | Detection | Threshold |
|--------|-----------|-----------|
| CV | std(step_comm_times) / mean(step_comm_times) | >15% = warning |

**Common Causes:**
- Network congestion on HCCS/RDMA links
- Collective operation size variance (dynamic batch padding)
- Slow-rank drag causing variable wait times at barriers

**Recommended Actions:**
- Use fixed-size collectives (pad to consistent sizes)
- Check HCCS link health
- Identify and fix slow ranks first

### Cross-Rank Alignment Skew

| Metric | Detection | Threshold |
|--------|-----------|-----------|
| Max skew | max(rank_mean_step_times) - min(rank_mean_step_times) | >5ms = warning, >20ms = critical |

**Common Causes:**
- CPU scheduling imbalance across ranks
- One rank running on a different NUMA node
- Data loading imbalance
- Slow-rank drag from a single underperforming card

**Recommended Actions:**
- Ensure all ranks use consistent NUMA binding
- Balance data loading across ranks
- Fix slow ranks (see cluster-rank-diagnosis.md)

## CV Interpretation Guide

| CV Range | Status | Meaning |
|----------|--------|---------|
| 0-5% | normal | Very stable performance |
| 5-10% | normal | Acceptable variance |
| 10-15% | warning | Moderate variance, worth investigating |
| 15-20% | warning | Significant variance, likely impacting throughput |
| >20% | critical | Severe instability, must fix before optimizing |

## Integration with Performance Agent

The jitter analysis script (`scripts/analyze_jitter.py`) integrates with
the main pipeline:

1. Reads step summary from `summarize_step_breakdown.py`
2. Optionally reads cluster data from `detect_slow_ranks.py`
3. Outputs jitter metrics that feed into `build_performance_profile.py`
4. Jitter status contributes to bottleneck classification
