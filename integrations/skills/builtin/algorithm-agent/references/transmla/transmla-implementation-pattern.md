# TransMLA Implementation Pattern

Use this reference when the selected `algorithm-agent` case is a TransMLA-style
attention / KV-cache adaptation or conversion-oriented feature patch.

Treat TransMLA as an attention-path and cache-shape adaptation pattern, not as a
generic adapter patch and not as evidence of full runtime completion.

## Route identity

A bounded TransMLA-style port usually touches some combination of:

- config surface for feature gating or proving-scope selection
- attention-path wiring
- checkpoint-remap or load-time compatibility logic
- cache/runtime behavior when the feature interacts with KV layout
- focused tests for baseline-off, semantic slice, and runtime invariants

This maturity pass is about making the route pack easier to apply and verify. It
does not change the bounded proof line that the prior TransMLA work earned.

## Bounded proving progression

A reusable proving sequence is:

1. bounded proving-case patch
2. checkpoint-remap follow-on if load compatibility is required
3. bounded MLA-like semantic slice
4. bounded runtime/cache follow-on when the first runtime contract failure is
   known

Keep each step as one narrow question. Do not merge all of them into one patch.

## Config surface expectations

- Add a default-off gate for the new behavior.
- Expose only the smallest knobs required for the current proving slice.
- Validate impossible or misleading combinations early.
- Preserve baseline-off behavior unless the bounded slice explicitly replaces the
  original path.

## Attention-path touch points

- Keep the baseline path unchanged when the feature is off.
- Add the feature at the smallest stable attention-path seam.
- Preserve existing model input/output contracts unless the bounded scope
  explicitly proves a different contract.
- Treat semantic-slice evidence as separate from runtime/cache evidence.

## Checkpoint-remap boundary

- Treat remap logic as a separate follow-on when compatibility is not already
  implicit in the minimal patch.
- Keep remap claims narrow: supported artifact shapes, supported names, and
  explicit non-claims.
- Do not let remap success imply broader TransMLA maturity or broader checkpoint
  compatibility.

## Runtime/cache boundary

- Treat cache/runtime work as a separate bounded follow-on.
- Diagnose the first concrete runtime contract failure before changing model
  runtime logic.
- Keep paged/runtime-coupled work out of scope unless it becomes the concrete
  blocker for a later task.
- Do not let a non-cached or one-cache-path success line read as broader runtime
  completion.

## Conservative success wording

Use success wording like:

- bounded proving-case success
- bounded checkpoint-remap follow-on
- bounded MLA-like semantic slice
- bounded native cache-path fix
- validated for the intended narrow scope

Avoid wording that implies:

- full TransMLA migration
- fuller MLA semantics
- broader runtime integration
- paged runtime support
- broad checkpoint compatibility
- training/serving completion
- performance claims
- broader route maturity than the bounded evidence supports

## Porting rules

- Preserve baseline-off behavior.
- Prefer one narrow patch per follow-on.
- Keep reference-code -> code-map -> patch-plan bridging explicit.
- Classify blockers before widening scope: environment, sync, code-path,
  runtime contract, semantic mismatch, or accuracy drift.
- If the model family uses generated and source-of-truth files, patch the
  source-of-truth file and regenerate according to the repository workflow.
- Stop when the next step starts requiring paged runtime, broad generation
  integration, production training/serving, or performance work.

## Common failure modes

- semantic slice and runtime/cache work get merged into one unbounded change
- cache fixes are attempted before the concrete cache contract failure is
  diagnosed
- bounded evidence is reported as fuller semantic or runtime completion
- generated/modular file ownership is ignored in model families that require
  source-file patching plus regeneration
- environment blockers are mistaken for model-code failures
- checkpoint-remap success is reported as broader compatibility proof

## Stop / no-go boundary

Stop and classify the step as **not bounded** if it immediately requires:

- paged runtime support
- broader runtime orchestration
- wider generation refactors
- training/serving integration
- performance characterization
- broader semantic completion
- new claims beyond the actually validated slice
