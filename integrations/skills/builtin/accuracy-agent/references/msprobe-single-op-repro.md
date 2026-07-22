# msprobe Single-Op Repro

Use this file when the workflow has already narrowed the mismatch to one
operator or one tiny module and `msprobe` dump data exists.

This is the bridge between full-network evidence and operator-level validation.

## When To Use

Use generated single-op repro when:

- `accuracy-agent` has already localized one suspicious operator
- dump data exists at `L1`
- a full-network compare is too noisy to confirm the operator cleanly

Do not use it as the first localization tool.

## What It Generates

`msprobe` can extract one suspicious API from dump data and generate a
standalone script that replays forward or backward behavior.

MindSpore path:

```shell
msprobe -f mindspore op_generate -i ./config_op.json -o ./
```

The generated script can then be run directly with Python.

## Two Data Modes

### `random_data`

Use when:

- you need a fast first confirmation
- exact replay is not yet necessary

Tradeoff:

- faster
- lower fidelity

### `real_data`

Use when:

- the random replay is inconclusive
- the operator is highly data-sensitive
- you need higher-confidence confirmation

Tradeoff:

- slower
- higher fidelity

## Good Default Sequence

1. Try `random_data` first for quick screening.
2. If the result is suspicious or unstable, rerun with `real_data`.
3. Feed the result back into `operator-accuracy-triage.md`.

## Useful Config Fields

- `dump_json_path`
- `api_name`
- `extract_api_path`
- `propagation`
- `data_mode`
- `random_seed`
- `iter_times`

Prefer one operator, one propagation direction, and one minimal repro at a
time.

## What To Report

- operator name
- forward or backward
- random or real data mode
- whether the generated repro still reproduces the mismatch
- key metrics such as cosine, max absolute error, and max relative error

## Do Not

- Do not treat a generated script as a new source of truth; it is still a
  derived validation artifact.
- Do not skip API parameter verification before blaming the operator.
- Do not stop at `random_data` if the operator is known to be highly
  input-sensitive.
