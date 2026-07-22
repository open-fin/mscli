# msprobe Overflow And NaN

Use this file when the symptom is non-fatal NaN or Inf, or when training
continues but suspicious invalid values appear in logs or metrics.

## Primary Goal

Separate:

- the **first true overflow or invalid-value source**
from
- later propagation noise across steps, modules, or ranks

## MindSpore First Choice

For MindSpore, start with `overflow_check`.

Key prerequisites:

- `INF_NAN_MODE_ENABLE=1`
- `MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"`

MindSpore overflow detection depends on INF/NAN mode being enabled and aligned
between the framework and CANN side.

## Level Selection

- Dynamic graph:
  - `L1` for API-level overflow check
  - `L0` for Cell-level overflow check
- Static graph:
  - `L2` for kernel-level overflow check

Do not copy a dynamic-graph `L1` recipe into static-graph analysis.

## Recommended Workflow

1. Confirm the run is truly an Inf/NaN-style problem, not only a large-value
   precision drift.
2. Enable the required INF/NAN overflow mode.
3. Run `overflow_check` with narrow `rank` and `step` scope if possible.
4. Record the first suspicious API, Cell, or kernel.
5. If more context is needed, add `statistics` capture around the same scope.

## Propagation Warning

By the time many nodes show NaN or Inf, the root cause may already be upstream.

Treat the earliest suspicious site as the lead, not the noisiest later node.

## Multi-Rank Note

The dedicated `nan_analyze` tool described in `msprobe` currently targets
PyTorch dump analysis. For MindSpore, keep the workflow centered on
`overflow_check` plus narrowed dump evidence instead of relying on
`nan_analyze`.

## Evidence To Capture

- framework and device overflow mode settings
- task and level used
- rank and step scope
- first suspicious API, Cell, or kernel
- whether invalid values are present in inputs, outputs, or both

## Do Not

- Do not start with all ranks and all steps if one failing rank or one failing
  step is already known.
- Do not confuse large but finite values with true INF/NAN overflow.
- Do not blame a later communication node before ruling out earlier compute
  propagation.
