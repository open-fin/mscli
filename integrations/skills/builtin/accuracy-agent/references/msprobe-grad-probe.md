# msprobe Grad Probe

Use this file when training starts normally and diverges later, or when you
need to answer "which step first became bad".

`msprobe` grad probe is for step localization, not generic first-use dumping.

## When To Use

Use `grad_probe` when:

- step1 is aligned but later loss diverges
- one training run becomes unstable after several optimizer updates
- you need gradient similarity evidence between baseline and target

Prefer `grad_probe` over broad tensor dump when the main question is about
**when drift starts**, not **which layer output is wrong at step1**.

## What It Captures

`grad_probe` can export:

- gradient summary statistics
- optional direction data
- similarity results between two runs

Useful signals include:

- max / min / norm
- interval distribution
- gradient direction similarity

## Minimal Config

```json
{
  "task": "grad_probe",
  "dump_path": "./dump_path",
  "rank": [],
  "step": [],
  "grad_probe": {
    "grad_level": "L1",
    "param_list": [],
    "bounds": [-1, 0, 1]
  }
}
```

## Integration Reminder

- PyTorch: `debugger.monitor(model)`
- MindSpore: `debugger.monitor(optimizer)`

MindSpore integration is optimizer-based. Do not pass a model when the
framework path expects an optimizer.

## Important Semantics

- `step` means optimizer-call count, not always raw training-loop iteration.
- `L1` is the safest default for first use.
- `L2` is more detailed and heavier; use only when needed.
- `param_list` should be narrowed once you know which block is suspicious.

## Similarity Interpretation

The documented heuristic is:

- step 0 similarity below `0.97` is suspicious
- or any step drop larger than `0.03` is suspicious

Treat this as a diagnosis hint, not a mathematical proof. Always combine it
with:

- loss trend
- optimizer behavior
- module-level evidence if available

## Recommended Workflow

1. Capture gradient data for baseline and target under aligned config.
2. Compare the two dump directories.
3. Find the first suspicious step.
4. Use that step to drive the next narrowed investigation:
   - module compare
   - operator repro
   - optimizer/config check

## Good Output For The Main Report

Record:

- compared runs
- rank scope
- suspicious first step
- whether the first issue is present at step 0 or appears later
- which parameters show the earliest or sharpest similarity drop

## Do Not

- Do not use `grad_probe` as the first tool for a pure single-step output
  mismatch.
- Do not confuse training-step count with optimizer-step count.
- Do not escalate to operator blame before proving that the first drift step is
  stable and reproducible.
