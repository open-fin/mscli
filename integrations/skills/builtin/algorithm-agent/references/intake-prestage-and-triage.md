# Intake Pre-Stage and Triage

Status: this file describes the current **bounded intake pre-stage** within the `algorithm-agent` workflow. It is used before live patch work for triage and entry decisions; once intake passes, execution returns to the current patch-stage main flow.

Keep bounded intake compact.

This reference combines:
- intake artifact definition
- intake scoring / triage rubric
- default routing behavior
- first proving-case calibration guidance

## Purpose

Use this bounded intake pre-stage to decide whether a candidate should enter the
live patch-focused flow.

Convert early discovery into:
- a structured intake artifact
- a released-code existence check
- a source qualification result
- a go / no-go entry decision
- a recommended next action

Use DeepXiv as the preferred/default paper-intake assistant, but keep intake
source-flexible when other sources improve coverage.

## Intake Artifact

Record at least:

- `source_type`
- `paper_title`
- `paper_url`
- `code_url`
- `claimed_contribution`
- `target_task`
- `target_model_family`
- `feature_bucket`
- `likely_integration_surface`
- `dependency_complexity`
- `verification_risk`
- `migration_blockers`
- `qualification_basis`
- `source_status`
- `recommended_next_action`

## Scoring / Triage Rubric

Score or tag each candidate on:

- `feature_bucket`
- `paper_clarity`
- `code_availability`
- `target_family_fit`
- `integration_surface_clarity`
- `bridge_value`
- `verification_value`
- `intake_tractability`
- `recommended_next_action`

### Hard gates before patch entry

The following must not be low for a bounded intake candidate to enter live patch
work:

- `code_availability`
- `target_family_fit`
- `intake_tractability`

If any of these are low, default to `reject` or `watchlist`.

### Ranking factors

Use these to rank candidates that pass the gates:

- `bridge_value`
- `verification_value`
- `integration_surface_clarity`
- `paper_clarity`

### High / Medium / Low guidance

#### `code_availability`
- high: maintained public repo with reusable core logic
- medium: partial or demo code with recoverable implementation logic
- low: no usable public code path

#### `target_family_fit`
- high: natural fit for Qwen / DeepSeek-like decoder-only models
- medium: adaptable with moderate reshaping
- low: poor or unclear fit

#### `bridge_value`
- high: strongly exercises paper -> code-map -> patch-plan
- medium: only some bridge steps are informative
- low: mostly template reuse with little bridge learning

#### `verification_value`
- high: meaningfully stresses torch / torch_npu / MindSpore scaffold
- medium: validates only part of the scaffold
- low: little new scaffold value

#### `intake_tractability`
- high: realistic bounded entry candidate now
- medium: feasible with constrained scope
- low: too large or ambiguous for bounded intake

## Default routing behavior

- any hard-gate failure -> `reject` or `watchlist`
- passes gates but weak ranking value -> `reference-code extraction`
- passes gates with strong `bridge_value` + `verification_value` + at least
  medium `integration_surface_clarity` -> `proving candidate`
- promising but blocked by timing or ambiguity -> `watchlist`

## Bounded intake calibration

Use `TransMLA` as the first worked example for calibrating:
- intake scoring
- artifact shape
- minimum helper behavior
- minimum file split decisions

Use `mHC` and `AttnRes` as reuse anchors for existing route-pack patterns, not
as reasons to expand file structure early.

