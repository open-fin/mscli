# msprobe Config Cheatsheet

Use this file before writing or editing `config.json` for `msprobe`.

`msprobe` uses one entry format:

- `task`
- `dump_path`
- optional `rank`
- optional `step`
- `level`
- one task-specific block such as `statistics`, `tensor`, or `grad_probe`

Start from the minimum config that can answer the current question.

## Universal Rules

- `dump_path` is required.
- `rank: []` and `step: []` mean "all available".
- For single-card training, keep `rank: []` instead of forcing one rank id.
- Prefer narrowing `step` and `rank` early.
- Keep baseline and target configs structurally aligned.

## Fast Task Selection

### `statistics`

Use when:

- you need summary statistics first
- you want cheaper capture before exact tensor dump

Common MindSpore dynamic-graph skeleton:

```json
{
  "task": "statistics",
  "dump_path": "./dump_path",
  "rank": [],
  "step": [],
  "level": "L1",
  "statistics": {
    "scope": [],
    "list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

### `tensor`

Use when:

- summary statistics are not enough
- exact tensor value compare is required

Common MindSpore dynamic-graph skeleton:

```json
{
  "task": "tensor",
  "dump_path": "./dump_path",
  "rank": [],
  "step": [],
  "level": "L1",
  "tensor": {
    "scope": [],
    "list": [],
    "data_mode": ["all"]
  }
}
```

### `grad_probe`

Use when:

- training starts normally but drifts later
- you need the first bad optimizer step

Minimal skeleton:

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

### `overflow_check`

Use when:

- NaN or Inf appears
- you need an overflow-focused first pass

Dynamic-graph skeleton:

```json
{
  "task": "overflow_check",
  "dump_path": "./dump_path",
  "rank": [],
  "step": [],
  "level": "L1",
  "overflow_check": {
    "overflow_nums": 1
  }
}
```

### `free_benchmark`

Use when:

- there is no trusted baseline
- the investigation is already narrow

Do not use as the first full-network action.

## Level Selection

- `L0`
  - module or Cell level
  - useful for broad layer-level localization
- `L1`
  - API level
  - default best starting point for dynamic-graph compare
- `L2`
  - kernel level
  - heavier and usually not the first step
- `mix`
  - combined module plus API data
  - use only when simpler levels cannot localize enough

## Common Safe Defaults

- Start with `statistics` at `L1` for dynamic-graph compare.
- Start with `statistics` or `tensor` at `L2` for static-graph capture.
- Start with `grad_probe` at `L1` for later drift.
- Restrict `list` or `scope` before escalating dump volume.

## Common Pitfalls

- `step` in `grad_probe` tracks optimizer calls, not always raw training loop
  iterations.
- Static-graph and dynamic-graph level choices differ; do not copy a dynamic
  `L1` recipe blindly into a static-graph run.
- `tensor` is expensive. Use it only when `statistics` cannot prove enough.
- Keep baseline and target task/level aligned before trusting compare results.
