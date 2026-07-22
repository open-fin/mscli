# Precision Classification Map

Use this file first when the user reports an accuracy issue. The goal is to put
the problem into one primary bucket before choosing tools, branches, or known
issue cards.

Choose one **primary** bucket first. Secondary buckets are allowed, but the
workflow should still start from the primary one.

## Bucket 1: Baseline And Compare-Harness Problems

Use when the reported "accuracy issue" may actually be a bad reference,
comparison path, or tolerance policy.

Typical signals:

- testcase name says `float64` but the reference path casts to `float32`
- different framework branches are used inside one compare helper
- scalar or `0D` path behaves differently from normal tensor path
- tolerance is obviously mismatched to dtype or value range

Common examples:

- wrong baseline dtype
- bad compare harness
- wrong tolerance
- fake operator bug created by a reference-side mistake

Preferred next steps:

- `reference-baseline-triage.md`
- compare-harness inspection
- narrow single-case validation before any operator blame

## Bucket 2: Data, Config, And Environment Alignment Problems

Use when the first suspicion is that setup drift, not operator behavior, caused
the mismatch.

Typical signals:

- step1 already mismatches
- AMP or env vars may differ
- two runs use different hyperparameters, libraries, or datasets
- checkpoint lineage is uncertain

Common examples:

- config mismatch
- random-control mismatch
- bad or partial checkpoint conversion
- framework/CANN/env-version drift

Preferred next steps:

- `msprobe-config-and-ckpt-check.md`
- `comparison-scenarios.md`
- consistency checks before deeper operator work

## Bucket 3: Module Or Semantic Mismatch

Use when the first stable gap appears at one model block, one Cell, or one
semantic layer boundary, but not yet one operator.

Typical signals:

- one module output differs while upstream modules are aligned
- API defaults, buffer state, or submodule wiring may differ
- large model compare needs block-level localization first

Common examples:

- wrong padding/mask/tokenizer path
- module parameter or buffer mismatch
- one block diverges but inputs are still aligned

Preferred next steps:

- layer or Cell compare
- module-input verification
- `msprobe-accuracy-compare.md`

## Bucket 4: Single-Operator Numerical Problems

Use only after the workflow has already narrowed the first stable mismatch to
one operator or one tiny local computation path.

Typical signals:

- one operator reproduces the mismatch under aligned inputs
- operator is numerically sensitive
- parameters and module inputs are already aligned

Common examples:

- true operator precision issue
- wrong API default or semantic mismatch at one op
- dtype-path-specific operator behavior

Preferred next steps:

- `operator-accuracy-triage.md`
- `msprobe-single-op-repro.md`

## Bucket 5: Training Drift And Update-Path Problems

Use when step1 is aligned but later steps diverge.

Typical signals:

- first mismatch appears only after several optimizer updates
- gradients or one-step updates differ
- communication, clipping, or loss scale may be involved

Common examples:

- backward mismatch
- optimizer update mismatch
- grad clipping mismatch
- distributed reduction mismatch

Preferred next steps:

- `diagnosis-branches.md` Branch B
- `msprobe-grad-probe.md`
- one-step update checks

## Bucket 6: Invalid-Value, Overflow, And Backend-Evolution Problems

Use when NaN/Inf, overflow mode, or version/backend behavior is central to the
problem.

Typical signals:

- non-fatal NaN/Inf
- overflow-related logs
- backend or version upgrade changed numerical behavior
- shared-kernel or precision-mode questions dominate the diagnosis

Common examples:

- overflow path
- INF/NAN mode mismatch
- backend precision mode change
- version or CANN evolution issue

Preferred next steps:

- `msprobe-overflow-and-nan.md`
- `ascend-precision-notes.md`
- compare old vs new version under aligned setup

## Quick Routing Table

| Symptom | Primary bucket | First move |
| --- | --- | --- |
| `float64` compare fail only in one case | Baseline And Compare-Harness | inspect reference dtype path |
| step1 loss mismatch with uncertain setup | Data, Config, And Environment Alignment | config and determinism checks |
| one block output first diverges | Module Or Semantic Mismatch | verify module inputs, then compare at block level |
| one operator still mismatches under aligned inputs | Single-Operator Numerical | operator triage and single-op repro |
| step1 matches, later loss drifts | Training Drift And Update-Path | grad probe and one-step update checks |
| NaN/Inf or backend version change | Invalid-Value, Overflow, And Backend-Evolution | overflow or backend-path checks |

## Rules

- Do not jump to Bucket 4 before Buckets 1 through 3 are reasonably ruled out.
- Bucket 1 has priority when the compare harness is suspicious.
- Bucket 2 has priority when setup alignment is unproven.
- Bucket 5 has priority when step1 is clean and drift appears later.
- Bucket 6 does not replace the need to find the earliest meaningful
  divergence; it only changes the first validation path.
