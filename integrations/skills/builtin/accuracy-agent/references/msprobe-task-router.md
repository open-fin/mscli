# msprobe Task Router

Use this file when the workspace has `msprobe` and you need to choose one
precision action quickly. Route by symptom first, not by tool name.

## Start With The Smallest Useful Task

- Prefer one focused `msprobe` task over broad multi-task dumping.
- Start with `statistics` before `tensor` unless you need exact values.
- Narrow by `rank`, `step`, and API or module list whenever possible.
- Reuse the same `dump_path` naming convention across baseline and target runs.

## Symptom To Task Mapping

### Step1 Loss Mismatch Or Wrong Final Output

Use:

- `statistics` or `tensor` for capture
- `compare` for full-network or module-level compare

Prefer:

- `statistics` + `compare` when you need the first stable mismatch
- `tensor` + `compare` when exact tensor values are required

Load next:

- `references/msprobe-accuracy-compare.md`
- `references/msprobe-config-cheatsheet.md`

### Cross-Framework Or Cross-Version Mismatch

Use:

- `statistics` or `tensor`
- `compare`

Prefer:

- `L1` API compare for dynamic graph API-level mismatch
- `L0` Cell compare for module mismatch
- `mix` or Layer mapping only after a simpler compare cannot localize enough

Load next:

- `references/msprobe-accuracy-compare.md`

### Later-Stage Drift After A Normal Start

Use:

- `grad_probe`
- `monitor` if the problem may involve optimizer or communication state

Prefer:

- `grad_probe` first when the question is "which step started to drift"
- `monitor` when gradients alone are not enough or communication/optimizer
  anomalies are suspected

Load next:

- `references/msprobe-grad-probe.md`

### NaN Or Inf Appears During Training

Use:

- `overflow_check`
- `statistics` when you need surrounding context

Prefer:

- `overflow_check` to find the first suspicious overflow site
- `statistics` to confirm whether the issue is true overflow or broader invalid
  value propagation

If the workspace later adds a dedicated NaN analysis reference, route there
after overflow localization.

### Suspected Single-Operator Precision Issue

Use:

- `run_ut` when the operator is supported by precheck flow
- generated single-operator script after network dump when broad capture is too
  noisy

Use this only after the workflow has already narrowed the mismatch to one
operator or one very small module.

### No Trusted Baseline

Use:

- `free_benchmark` only for narrow local investigation

Do not start with no-benchmark compare on a full large model.

### Config Or Checkpoint Consistency Suspicion

Use:

- `config_check`
- `ckpt_compare`

Use this when the first gap may come from environment, AMP, optimizer, or
checkpoint lineage rather than operator behavior.

## Decision Hints

- If you need the **first mismatching layer or API**, choose capture plus
  `compare`.
- If you need the **first bad step**, choose `grad_probe`.
- If you need the **first NaN/Inf site**, choose `overflow_check`.
- If you need to prove **environment or checkpoint drift**, choose
  `config_check` or `ckpt_compare`.

## Do Not

- Do not enable `tensor` dump across the whole network by default.
- Do not switch between multiple `msprobe` tasks mid-analysis without a reason.
- Do not use `free_benchmark` as a substitute for a real trusted baseline.
- Do not collect all ranks and all steps when one rank and one step are enough
  to prove the next hypothesis.
