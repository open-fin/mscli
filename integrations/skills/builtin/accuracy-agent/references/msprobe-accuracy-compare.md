# msprobe Accuracy Compare

Use this file when the problem is a step1 loss mismatch, wrong final output,
cross-version mismatch, or cross-framework mismatch and `msprobe` is available.

The goal is to use `msprobe` compare to find the first stable mismatch with the
least expensive capture that still answers the question.

## Best Entry Points

### Same Framework, Different Version

Use when:

- MindSpore vs MindSpore version comparison
- same model and input, different framework or CANN stack

Prefer:

- `statistics` or `tensor` capture
- `compare` on aligned dump directories

Good starting points:

- `L1` for dynamic-graph API-level compare
- `L0` for Cell/module compare when the network is easier to reason about by
  block than by API
- `L2` only after higher-level compare cannot explain the gap

### Cross-Framework Compare

Use when:

- PyTorch is the trusted baseline
- MindSpore is the target under diagnosis

Prefer:

- API compare first
- Cell or Layer compare when API names are noisy or mapping is not obvious

`msprobe` supports:

- API mapping
- Cell mapping
- data mapping
- Layer mapping

Do not start with custom mapping unless the default route is clearly
insufficient.

## Recommended Compare Escalation

1. Start with summary capture using `statistics`.
2. Run `compare` and inspect the earliest stable mismatch.
3. If compare is too coarse, narrow by:
   - API list
   - Cell list
   - layer mapping
   - one rank
   - one step
4. Escalate to `tensor` only when exact tensor values are required.
5. Escalate to kernel compare only when API or Cell compare cannot explain the
   gap.

## Compare Modes To Prefer

### API Compare

Best for:

- step1 loss mismatch
- wrong final output
- dynamic-graph cross-framework compare
- operator-level narrowing after a module is already suspected

### Cell Compare

Best for:

- large models where API rows are too noisy
- model-block mismatch
- same-framework module drift

### Layer Mapping

Best for:

- cross-framework compare when you know the corresponding model layers
- reducing noise between many APIs inside one block

Use only after simpler compare modes fail to localize cleanly.

## Minimal Working Flow

1. Capture aligned dump data from baseline and target.
2. Create one compare input file.
3. Run:

```shell
msprobe -f mindspore compare -i ./compare.json -o ./output
```

4. Read the first stable mismatch instead of scanning only the final row.
5. Feed that mismatch back into the main `accuracy-agent` workflow.

## Evidence To Extract

For the report, keep:

- compare mode used
- dump level used
- rank and step scope
- first stable mismatching API, Cell, or Layer
- whether mapping was default or custom
- whether the mismatch persists under narrower scope

## Do Not

- Do not start with full-network `tensor` dump.
- Do not jump to kernel-level compare before higher-level compare narrows the
  scope.
- Do not trust a broad compare result if baseline and target dump config are
  not aligned.
- Do not treat the last mismatching row as the root cause; always prefer the
  earliest stable mismatch.
