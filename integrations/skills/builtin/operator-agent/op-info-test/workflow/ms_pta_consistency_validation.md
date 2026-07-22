# MS/PTA Consistency Validation

<a id="ms-pta-consistency-goal"></a>
## Goal

When the operator has a PTA benchmark interface, validate that the MindSpore interface and the PTA interface produce consistent outputs under the same deterministic inputs.

This workflow intentionally supports:

- one case
- multiple explicitly named cases
- one structured case_spec that can describe either of the above
- PyNative mode
- Graph mode
- both modes in one driver

The primary semantic gate uses the `allclose_nparray` style from the sparse-lightning
reference test in `mindspore/tests/st/custom/ops_custom/ascendc/test_ms_vs_pta_custom_aclnn_sparse_lightning.py`:
`rtol=0`, `atol=0`, `equal_nan=True` by default.

Default to `bitwise_strict` for final pass/fail decisions. Even under strict mode, keep
the `semantic_equal` and `all_equal` results as diagnostic signals so it stays obvious
whether a failure is an actual semantic mismatch or a bit-level mismatch.

Unless the task explicitly narrows the scope, build one validation driver that covers
both PyNative mode and Graph mode by switching `set_context` between runs. Use a
single-mode driver only when the task explicitly requests `pynative` or `graph`.

When generating a new operator-specific driver, prefer the bundled framework and scaffold
instead of rewriting the full flow:

- comparator: `scripts/ms_pta_consistency_output_comparator.py`
- framework: `scripts/ms_pta_consistency_execution_framework.py`
- scaffold: `template/ms_pta_consistency_operator_scaffold.template.py`

This workflow is separate from `workflow/op_info_generation.md`.
Its generated driver, `.npy` artifacts, batch files, and summaries are local validation assets and must not be committed into the operator test repository.

<a id="ms-pta-consistency-assets"></a>
## Asset Roles

- `scripts/ms_pta_consistency_output_comparator.py`
  - Generic comparator for saved `.npy` artifacts
  - Consumes one structured `case_spec` JSON file
  - Owns summary generation and pass/fail decisions

- `scripts/ms_pta_consistency_execution_framework.py`
  - Reusable framework for operator-specific validation drivers
  - Owns runtime setup, seed control, artifact layout, and case_spec assembly
  - Should stay generic and reusable across operators

- `template/ms_pta_consistency_operator_scaffold.template.py`
  - Copy-and-fill scaffold for one concrete operator
  - Owns only operator-specific case construction plus MS/PTA forward or backward execution
  - Should not duplicate the comparator or framework implementation

- `template/ms_pta_consistency_case_spec.template.json`
  - Input example for structured validation input
  - Useful when you need one case, multiple named cases, or want to inspect the expected case_spec schema
  - Reference template only, not a runtime dependency

- `template/ms_pta_consistency_case_summary.template.json`
  - Output example for per-case summaries
  - Useful for understanding which fields the comparator writes and how to report them
  - Reference template only, not a runtime dependency

Keep only one operator scaffold. Put generic execution logic in `scripts/`, and put operator-specific logic in the scaffold.

<a id="ms-pta-consistency-driver-split"></a>
## Why The Driver Stays Split

For this skill, the operator driver is a long-term reusable asset rather than a throwaway local script, so keep the driver split into:

- one reusable framework layer
- one operator-specific scaffold

This split is intentional:

- Changes to runtime setup, seed control, artifact layout, or case_spec generation should be fixed once in the framework and inherited by later drivers.
- The operator scaffold should stay focused on the parts that actually vary by operator: case design, forward path, and backward path.
- A single giant template would duplicate stable framework code into every generated driver and raise long-term maintenance cost.

<a id="ms-pta-consistency-inputs"></a>
## Inputs

- The exact same deterministic inputs for the MS path and the PTA path
- The execution mode for the MS path: `pynative`, `graph`, or `both`
- Whether the operator requires forward-only validation or forward-plus-backward validation
- Saved `.npy` outputs for each side, grouped by output name
- One structured `case_spec` file that contains one or more cases

<a id="ms-pta-consistency-modes"></a>
## Execution Modes

- `pynative`
  - Use when the task explicitly requests PyNative mode only
  - Use a minimal PyNative-mode driver that directly calls the target operator

- `graph`
  - Use when the task explicitly asks for Graph mode only
  - Wrap the target operator in a minimal `nn.Cell` or Graph-mode-compatible driver
  - Reuse the same deterministic source inputs used by the PTA path

- `both`
  - Default recommendation when the task does not specify a mode
  - Reuse one driver and switch `set_context` between PyNative mode and Graph mode
  - Model the two modes as separate case_spec entries inside one run

<a id="ms-pta-consistency-graph-authoring-rules"></a>
## Graph Mode Authoring Rules

- `context.set_context(mode=ms.GRAPH_MODE)` only switches the runtime mode. It does not automatically graph-capture arbitrary Python helper functions.
- Prefer a named `nn.Cell` wrapper for the Graph-mode MS path. In MindSpore source, `Cell.__call__` compiles and runs the cell when the current mode is `GRAPH_MODE`, so this is the most stable default for operator consistency drivers.
- Use `@ms.jit` only when the Graph path genuinely needs a standalone Python function instead of a `Cell`. Treat `jit` as the graph-capture tool for ordinary functions, not as a replacement for the global mode switch.
- Keep Graph-mode callables as named `nn.Cell.construct` methods or named `@ms.jit` functions so source inspection and graph parsing stay stable.
- Run Graph-mode probes from a real `.py` file instead of temporary stdin source so source inspection and graph parsing stay reliable during final validation.
- When one driver covers both modes, keep the mode-specific dispatch shallow:
  - PyNative branch: direct function or op call
  - Graph branch: named `nn.Cell` or named `@ms.jit` callable
  - Avoid mixing Graph-only wrappers into the PyNative branch unless the operator specifically requires it

<a id="ms-pta-consistency-forward-backward"></a>
## Forward / Backward Coverage

- If the operator has a backward path, validate both:
  - forward outputs
  - backward gradients

- If the operator does not expose a backward path, validate forward outputs only.

- Forward and backward parity should reuse the same deterministic source inputs. For backward
  parity, also reuse the same deterministic upstream gradient or sensitivity tensor.

- Prefer stable output names in saved `.npy` artifacts so summaries stay comparable across
  drivers:
  - forward outputs: `forward_<output_name>`
  - backward gradients: `backward_grad_<input_name>`

<a id="ms-pta-consistency-outputs"></a>
## Required Outputs

For a single case:

- `<workdir>/ms_outputs/<output_name>.npy`
- `<workdir>/pta_outputs/<output_name>.npy`
- `<workdir>/ms_pta_consistency_single_case_res_summary.json`

For a one-or-more-case `case_spec` run:

- per-case summaries if requested
- one run summary such as `<workdir>/ms_pta_consistency_all_case_res_summary.json`

<a id="ms-pta-consistency-strategies"></a>
## Comparison Strategies

- `semantic_zero`
  - Optional semantic-only mode
  - Requires `all_equal == true`
  - Uses `allclose_nparray(..., rtol=0, atol=0, equal_nan=True)`

- `bitwise_strict`
  - Default recommendation
  - Requires `all_equal == true` and `all_raw_bytes_equal == true`
  - Raw-byte equality is based on normalized contiguous array bytes, not just `.npy` file hashes
  - Still records `semantic_equal` so failures can be split into semantic mismatches vs bit-level mismatches

Supplemental evidence:

- `all_binary_equal`
  - Equality of saved `.npy` file hashes
  - Useful as an artifact check
  - Do not use it as the primary semantic gate

<a id="ms-pta-consistency-steps"></a>
## Execution Steps

1. Fix the random seed and prepare the same source inputs for both interfaces. Do not let either side generate its own random tensors independently.
2. Choose the MS execution mode:
   - default to `both`
   - use `pynative` or `graph` only when explicitly requested
3. Determine coverage depth:
   - if the operator has backward support, validate both forward and backward
   - otherwise validate forward only
4. Run the MS interface in the selected mode or modes and save every comparable output as `.npy` under `ms_outputs/`.
   - For Graph mode, the driver should call a named `nn.Cell` or named `@ms.jit` function.
   - If the target is naturally modeled as an operator wrapper, prefer `nn.Cell` over `jit`.
5. Run the PTA benchmark interface with the same inputs and save the corresponding outputs as `.npy` under `pta_outputs/`.
6. Build one `case_spec` file.
   - put one case into `cases` if you only need one operator scenario
   - put multiple cases into `cases` if you need multiple scenarios
7. Run the comparator on the generated artifacts.

Single-case `case_spec` example:

```json
{
  "summary_out": "./ms_pta_consistency_all_case_res_summary.json",
  "cases": [
    {
      "case_id": "sparse_lightning_loss",
      "ms_dir": "./ms_outputs",
      "pta_dir": "./pta_outputs",
      "outputs": ["loss", "out0", "out1"],
      "rtol": 0.0,
      "atol": 0.0,
      "strategy": "bitwise_strict",
      "summary_out": "./ms_pta_consistency_single_case_res_summary.json"
    }
  ]
}
```

```bash
python <skill_root>/scripts/ms_pta_consistency_output_comparator.py \
  --case_spec ./ms_pta_consistency_case_spec.json
```

`<skill_root>` means the active `op-info-test` skill directory, regardless of which
editor, assistant runtime, or local integration is loading it.

8. Inspect the generated summary and record:
   - which case or cases were compared
   - which MS execution mode was used
   - whether forward-only or forward-plus-backward parity was validated
   - the strategy used
   - whether `all_equal` passed
   - whether `all_raw_bytes_equal` passed when strict parity is required
   - whether `error_type` is `comparison` or `infra` when the case fails
   - whether `all_binary_equal` passed if artifact hash evidence is useful

<a id="ms-pta-consistency-reporting"></a>
## Reporting Requirements

Record the following when consistency validation is required:

- compared output names
- whether coverage is forward-only or forward-plus-backward
- execution mode: `pynative`, `graph`, or `both`
- strategy: `semantic_zero` or `bitwise_strict`
- generated summary path
- whether `all_equal` passed
- whether `all_raw_bytes_equal` passed
- whether `error_type` is `comparison` or `infra` when validation fails
- whether `all_binary_equal` is useful as supplemental artifact evidence

<a id="ms-pta-consistency-success"></a>
## Success Criteria

- The MS and PTA paths consume the same deterministic inputs
- The selected MS execution mode matches the task requirement. If no mode is specified, `both` is the default.
- Graph-mode paths use Graph-compatible authoring patterns: named `nn.Cell` by default, or named `@jit` function when a standalone function is required.
- If the operator has backward support, backward parity is included in the saved outputs and summary.
- All comparable outputs are saved as `.npy`
- The framework completes and generates the requested summary
- Default `bitwise_strict`: `all_equal == true` and `all_raw_bytes_equal == true`
- Optional `semantic_zero`: `all_equal == true`

<a id="ms-pta-consistency-failure-handling"></a>
## Failure Handling

- If `all_equal == false`, treat it as an accuracy regression and keep the generated summary as evidence.
- If `all_equal == true` but `all_raw_bytes_equal == false`, treat it as a strict binary-consistency mismatch.
- If `error_type == comparison`, repair only the mismatched case logic or artifacts and rerun the failed case.
- If `error_type == infra`, stop changing the operator case and record the environment or file-system blocker first.
- If Graph mode fails during parsing or source inspection, first check whether the driver is using the recommended Graph-friendly form: a named `nn.Cell` or named `@ms.jit` callable executed from a real `.py` file.
- If `.npy` hashes differ while semantic equality holds, record that as artifact-level evidence only; do not automatically treat it as a semantic failure.
- If the framework reports missing files, do not restart the full op-info flow. Regenerate only the missing MS/PTA output artifacts and rerun this workflow.
- If only one case in a case_spec fails, rerun only that case instead of redoing the whole run.
