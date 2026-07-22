# msprobe Config And Ckpt Check

Use this file when the first plausible explanation is setup drift rather than a
single bad operator.

Typical triggers:

- two environments are supposed to be aligned but still diverge early
- AMP, env var, optimizer, or dataset settings may differ
- checkpoint lineage may already have drifted before the observed metric gap

## Config Check

`msprobe` config check is for comparing two environments on factors that can
silently affect accuracy.

It can cover:

- environment variables
- third-party library versions
- training hyperparameters
- weights
- dataset-related evidence
- random-related operations

### Good Use Cases

- pre-compare sanity check
- migration validation between two stacks
- proving that a "precision bug" is actually setup inconsistency

### Static Collection

Use CLI collection when shell or yaml config is the main source of truth.

### Dynamic Collection

Use dynamic collection when static scripts are not enough, especially when the
training stack mutates settings at runtime.

## Ckpt Compare

Use checkpoint compare when:

- the runs have already progressed
- you need to prove parameter-level drift
- the question is not "did step1 match" but "how far apart are the states now"

Useful metrics include:

- `l2`
- `cos`
- `numel`
- `shape`

## Recommended Order

1. Use `config_check` when environment or training configuration mismatch is a
   live hypothesis.
2. Use `ckpt_compare` when you need to show that two runs have already diverged
   at parameter level.
3. Only return to operator blame after these broader drift sources are ruled
   out.

## Evidence To Keep

- which collection mode was used
- whether shell or yaml inputs were provided
- summary pass/fail across env vars, versions, hyperparameters, weights, and
  dataset signals
- whether checkpoint similarity is broadly close or already degraded

## Do Not

- Do not skip config comparison when AMP, env vars, or framework versions are
  not yet proven aligned.
- Do not use checkpoint similarity alone to name one operator as the root
  cause.
- Do not treat config or ckpt checks as optional once setup drift is the
  leading hypothesis.
