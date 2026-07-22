# Profiler Data Validation Rules

Read this file before Stage 1 to validate profiler data integrity.

## Goal

Verify that profiler data is complete enough for meaningful bottleneck
analysis before investing time in summary generation.

## Data Types

| Type | Identifier | Collection Method |
|------|-----------|-------------------|
| Framework profiler (PyTorch) | `*_ascend_pt` suffix | Code-embedded, process-level |
| Framework profiler (MindSpore) | `*_ascend_ms` suffix | Code-embedded, process-level |
| msprof CLI | `PROF_{}` directory | Command-line, process-level |

## Validation Checks

### 1. Collection Status (Stop Check)

| Data Type | Required File | Failure Action |
|-----------|-------------|----------------|
| Framework profiler | `profiler_info.json` or `profiler_info_{Rank_ID}.json` | **Stop**: collection did not complete normally, advise user to check `profiler.stop()` |
| msprof | `PROF_{}/device_{}/end_info.{}` | **Stop**: collection did not complete normally |

### 2. Parse Status (Parse Check)

| Data Type | Required Directory | Failure Action |
|-----------|-------------------|----------------|
| Framework profiler | `ASCEND_PROFILER_OUTPUT/` | **Pause**: ask user if they want to run offline parser |
| msprof | `PROF_{}/mindstudio_profiler_output/` | **Pause**: ask user if they want to run `msprof --export=on` |

### 3. Key Deliverables Check

| File | Required For | Failure Action |
|------|-------------|----------------|
| `step_trace_time.csv` | Step breakdown analysis | Warning: step analysis unavailable |
| `kernel_details.csv` or `op_summary_*.csv` | Operator hotspot analysis | Warning: hotspot analysis unavailable |
| `trace_view.json` | Timeline gap analysis | Warning: trace gap analysis unavailable |
| `*.db` (profiler DB) | SQL-based analysis | Warning: DB queries unavailable |
| `communication.json` | Communication analysis | Note: comm analysis unavailable |
| `memory_record.csv` / `operator_memory.csv` | Memory analysis | Note: memory analysis unavailable |
| `dataset.csv` / `minddata_pipeline_*.csv` | Input pipeline analysis | Note: pipeline analysis unavailable |

### 4. AIC Metrics (Optional)

| File | Required For | Failure Action |
|------|-------------|----------------|
| AIC PMU data (inside PROF_{}) | Microarchitecture analysis | Note: AIC analysis unavailable (requires `msprof op --aic-metrics`) |

## Quality Levels

| Level | Criteria | Action |
|-------|----------|--------|
| `excellent` | Stop check OK + parse OK + step + kernel + trace all present | Proceed with full analysis |
| `good` | Stop OK + parse OK + at least 2 key files | Proceed, note limitations |
| `fair` | Stop OK + parse OK + only 1 key file | Proceed, warn about limited evidence |
| `poor` | Stop OK but no parse output or no key files | Warn, offer to parse |
| `critical` | Stop check failed or data missing entirely | **Stop**, advise recollection |

## Multi-Card Principle

- Default: spot-check one rank (e.g. Rank 0)
- Only check all ranks when user explicitly requests per-rank validation

## Type Binding Rule

Determine the data type from the top-level path and use only the rules
for that type throughout the entire validation:

- `*_ascend_pt` or `*_ascend_ms` → framework profiler rules only
- `PROF_*` → msprof rules only

Do not mix rules across types.
