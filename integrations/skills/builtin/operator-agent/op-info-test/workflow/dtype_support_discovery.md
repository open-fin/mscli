# Operator Dtype Support Discovery

## Goal

Use this workflow for a new or not-yet-registered operator when the task needs
an OpInfo dtype declaration.

The default path is:

1. check whether the API fits the current op_info workflow
2. run full dtype runtime probing against the PTA benchmark surface
3. use docs to cross-check or clarify probe failures
4. use the probe result as the candidate writeback set
5. validate that candidate through the MindSpore OpInfo path before treating it
   as final

## Quick Applicability Gate

Answer these four questions first:

1. Can the API be represented as one operator-level call with representative
   tensor inputs and tensor outputs?
2. Can correctness be judged from one execution plus optional backward, instead
   of multi-step state evolution?
3. Can you build a single-execution reference path for comparison?
4. Is the main behavior tensor computation rather than optimizer state updates,
   distributed coordination, global environment control, or object lifecycle
   management?

If all four answers are **Yes**, continue.

If any answer is **No**:

1. stop this workflow
2. record `not_applicable_to_opinfo`
3. record the blocker reason
4. recommend the alternative validation path

Typical non-applicable cases:

- optimizer state updates across steps
- multi-step sequence behavior that cannot be judged from one run
- distributed or process-group side effects
- global environment control
- object lifecycle management

## Default Path

For most tasks, follow this order:

1. run the runtime probe across the framework dtype set
2. cross-check aclnn, MindSpore, and PTA docs when the probe exposes a gap,
   issue, or disagreement
3. use the runtime-supported forward dtypes as the candidate writeback set
4. validate that candidate through the MindSpore OpInfo path

Default backend rule:

- use the **PTA** interface for runtime probing
- switch to the **MS** interface only when the user explicitly asks for it or
  the PTA path is unavailable

Use the documentation-only path only when:

- the user explicitly narrows the task to doc lookup only, or
- runtime probing is blocked by environment issues

If runtime probing is skipped, mark the result as
`dtype_declaration_source=doc_derived`.

## Outputs

- probe summary JSON and Markdown containing:
  - `forward_supported_dtypes`
  - `backward_supported_dtypes`
  - `backward_not_applicable_dtypes`
  - `forward_unsupported_dtypes`
  - `backward_unsupported_dtypes`
  - `forward_issue_dtypes`
  - `backward_issue_dtypes`
- candidate dtype writeback recommendation for `op_database.py`
- or `not_applicable_to_opinfo` with blocker notes and an alternative path
- if runtime probing is skipped, the same summary schema produced through
  `build_doc_derived_operator_summary(...)`

## Runtime Probe Workflow

Use `scripts/dtype_probe_execution_framework.py`. When helpful, start from
`template/dtype_probe_operator_scaffold.template.py`.

Steps:

1. Start from the framework dtype set unless the user explicitly narrows the
   dtype scope.
2. Build one operator probe driver and fill:
   - representative forward path
   - representative backward path
   - representative probe samples
   - doc-declared forward/backward dtype tuples when available, for comparison
   - probe backend selection
3. For parameter-rich `other` ops such as `conv*`, `linear`, `interpolate`,
   pooling, normalization, and loss/module wrappers, do not rely on a generic
   placeholder sample. Reuse the nearest family pattern already present in the
   repo so the dtype result is based on representative arguments.
4. Run the probe:

```bash
python <skill_root>/scripts/dtype_probe_execution_framework.py \
  --driver ./mint_acos_dtype_probe.py \
  --summary-out ./op_dtype_probe_summary.json \
  --markdown-out ./op_dtype_probe_summary.md
```

5. Review the summary and use the runtime-supported forward dtypes as the
   candidate writeback set.
6. Validate that candidate through the MindSpore OpInfo path. If the OpInfo
   result diverges from the PTA probe result, record and clarify the discrepancy
   instead of silently shrinking the dtype set.

You may pass multiple `--driver` arguments in one run if needed.

## Result Classification

Classify each dtype and direction as follows:

- forward success -> `forward_supported_dtypes`
- backward success -> `backward_supported_dtypes`
- backward not applicable for this dtype/sample path ->
  `backward_not_applicable_dtypes`
- error clearly indicates unsupported dtype -> `unsupported_dtype`
- any other error -> `sample_or_function_issue`

Rules:

- do not convert a generic failure into `unsupported_dtype` unless the error
  really says so
- do not fold "no gradient path" into `backward_supported_dtypes`

## Failure Handling

When the probe fails, use this order:

1. check whether the sample itself is wrong
2. check whether the generic driver logic is wrong
3. cross-check aclnn docs, MindSpore docs, and PTA docs
4. decide whether the issue belongs to:
   - testcase authoring
   - tool/framework logic
   - operator implementation
   - environment setup

If the failure looks like an operator implementation problem, keep the dtype in
`*_issue_dtypes` and record it as a functional blocker.

Do not hide operator bugs by silently removing the dtype from the candidate set.

## Documentation-Only Fallback

Use this path only when the task explicitly asks for doc lookup only or runtime
execution is blocked.

Steps:

1. check aclnn dtype support
2. check the corresponding MindSpore API docs
3. check existing similar OpInfo registrations when helpful
4. build a candidate dtype set and mark it as `doc_derived`
5. when a structured artifact is needed, use
   `build_doc_derived_operator_summary(...)` from
   `scripts/dtype_probe_execution_framework.py`
6. if runtime execution later becomes available, replace the doc-derived result
   with a real probe result

## Writeback And Coverage Rules

After the probe:

1. use the supported forward dtype set as the default candidate source for
   `dtypes_*`
2. for current aclnn OpInfo work, write back only `dtypes_ascend910b` by
   default
3. write `dtypes_backward_ascend910b` only when the backward result is stably
   narrower and the task requires encoding it
4. keep issue dtypes in the summary even when they are not written into
   `op_database.py`
5. do not treat `not_support_dtypes` as the automatic complement of the
   writeback set
6. do not treat the probe result as final until the required OpInfo validation
   matrix passes
7. record whether the final declaration is:
   - `probe_verified`
   - `doc_derived`
   - `probe_vs_doc_conflict`

Coverage notes for final OpInfo authoring:

- static forward dtype coverage should follow the supported forward dtype set
- backward discovery may use the full probe, but final OpInfo authoring does
  not need an extra custom backward-only dtype matrix unless the task explicitly
  asks for it
- dynamic shape/rank does not need a full dtype matrix by default; representative
  dtype coverage is enough
- negative cases do not need a full unsupported-dtype matrix; representative
  unsupported cases are enough
- leave `dtypes_ascend`, `dtypes_cpu`, and `dtypes_gpu` empty unless the task
  explicitly requires those platforms
