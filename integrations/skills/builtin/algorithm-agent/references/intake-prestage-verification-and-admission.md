# Intake Pre-Stage Verification and Admission

Status: this file describes the current **bounded intake pre-stage** within the `algorithm-agent` workflow. It is used before live patch work for triage and entry decisions; once intake passes, execution returns to the current patch-stage main flow.

Keep verification scaffold guidance and admission guidance compact in this
bounded intake pre-stage unless reuse clearly justifies splitting.

## Verification Scaffold

This bounded intake pre-stage scaffold is a minimum validation recording
contract, not a claim that full execution is already automated or complete.

Each slot should record:
- purpose
- expected evidence
- allowed status

Allowed status values:
- `pass`
- `fail`
- `blocked`
- `not_run`
- `partial`

Minimum repeatable checks:

- `torch` smoke forward
- `torch` backward / training-step sanity when relevant
- `torch_npu` smoke forward
- `torch_npu` backward / training-step sanity when relevant
- MindSpore NPU smoke forward
- MindSpore NPU backward / training-step sanity when relevant
- shape/dtype consistency checks
- feature on/off regression checks
- standard accuracy-drift classification output

### Slot evidence rule

Each slot must have explicit expected evidence before it can be treated as
complete. Unrunnable slots must be recorded as `blocked` or `not_run`; they must
never be implied as `pass`.

### Bounded proving-case result rule

A successful bounded proving-case result may stop once the selected scope has
recorded the expected evidence for that bounded rung. For the current
TransMLA proving case, that bounded success line is: default-off behavior
preserved, no-remap scope preserved, regeneration-backed patch status, passing
import/init, and passing minimal forward on the selected runtime path.

Such a result should be recorded as a successful bounded proving-case outcome,
not as a claim of full migration completeness. Unless separately validated, it
must not be described as checkpoint-remap-compatible, fuller-MLA-complete,
broader-runtime-complete, or MindSpore-native-complete.

## Handoff Rules

Hand off to:

- `readiness-agent` for environment/run readiness blockers
- `accuracy-agent` for wrong-result, mismatch, or drift after successful
  execution

## Combined-helper default

The bounded intake pre-stage should default to one combined helper/scaffold
script covering adjacent:
- intake artifact generation
- code-map artifact generation
- verification report generation

Only split the helper if execution proves the combined shape is no longer
maintainable.

## Runnable vs Scaffold-only Boundary

### Must be runnable in bounded intake
- intake artifact generation
- code-map artifact generation
- verification report generation with expected check slots and statuses
- at least one worked proving-case pass through the scaffold path

### May remain scaffold-only in bounded intake
- full automation of all torch / torch_npu / MindSpore execution steps
- automatic collection of every metric/check without manual assistance
- broad multi-case generation beyond the proving set

## Admission Gate

### Hard blockers
A feature must not enter factory/template if any of these fail:

- minimum verification scaffold is incomplete
- baseline-off behavior is not preserved
- code-map artifact is missing
- verification artifact is missing
- target-family integration touchpoints are not identified
- unresolved correctness/drift issue exists without explicit
  `accuracy-agent` handoff status

### Warnings
A feature may enter with warnings recorded when:

- only one model-family instance has been validated
- performance characterization is incomplete
- some optional robustness checks are pending
- paper/code ambiguity remains but does not block the validated path

### Minimum artifact set before admission
- intake artifact
- code-map artifact
- verification artifact
- one family-specific template or case instance
- admission checklist result with blockers/warnings explicitly recorded
