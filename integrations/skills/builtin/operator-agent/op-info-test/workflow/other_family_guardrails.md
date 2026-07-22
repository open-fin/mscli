# Complex `other`-Type Operator Guardrails

Load this file only when the target operator is a parameter-rich or heavily customized `other`-type API. This is a **family-level structure guide**, not a copy-paste template.

## When To Load This File

Use these guardrails when the operator clearly belongs to a complex `other` family, for example:

- `conv*`
- `linear` / projection
- `interpolate` / resize
- pooling
- normalization
- loss / module-wrapper APIs

For simpler `other` operators, the generic `other` guidance in [op_info_generation.md](op_info_generation.md) is enough.

## Shared Guardrails For Complex `other` Ops

1. **Start from the closest existing family pattern, not from a blank page**  
   Search `op_database.py`, `op_sample_inputs.py`, and `op_wrappers.py` together before writing anything new.
2. **Keep wrappers thin and deterministic**  
   Wrappers should only express the public API call. For module-style operators, set weights/bias/other parameters explicitly instead of relying on random initialization or hidden state.
3. **Separate stable samples from edge samples**
   - `op_basic_reference_inputs_func`: default parameters, one or two stable valid shape/parameter variants, and the main differentiable path if applicable.
   - `op_extra_reference_inputs_func`: discontiguous/layout variants, empty tensors, special values, and other edge behaviors that are meaningful but less stable.
4. **One behavior axis per yielded sample**  
   Each `OpSampleInput` should have a descriptive `sample_name` and should primarily test one thing at a time (shape variation, parameter boundary, empty input, special value, layout, and so on).
5. **Build multi-input coverage in layers**  
   Start with same-dtype and same-shape happy paths. Add broadcasting, mixed dtype, rank mismatch, or parameter-coupled cases only when the operator contract says they matter.
6. **Do not silently omit dynamic or error hooks**  
   If `op_dynamic_inputs_func` or `op_error_inputs_func` is missing, treat that as an explicit coverage gap in the task summary instead of counting the operator as fully validated.
7. **Avoid overfitting to the probe result**  
   The probe tells you which dtypes are candidates. It does not tell you which custom sample matrix is representative. Keep the custom matrix compact and behavior-driven.
8. **Wire the coverage groups deliberately**  
   For `other` ops, check not only `other_op_db`, but also whether the operator should appear in `other_op_kbk_db`, `other_op_error_db`, and `other_op_error_kbk_db`.

## Family-Level Structure Checklists

Use these as structure prompts only. Do **not** reuse another operator's exact shapes, dtype list, or error messages just because it is in the same family.

### `conv*` family

Require explicit coverage for parameter coupling such as:

- `bias`
- `stride`
- `padding`
- `dilation`
- `groups`
- dynamic inputs
- representative constraint errors

Be ready to add device-specific loss overrides when the repo already does so for the same family.

### `linear` / projection family

Require:

- deterministic parameter initialization
- shape-coupling coverage for `input` / `weight` / `bias`
- representative invalid-shape or invalid-bias cases

Prefer a thin wrapper over inline module setup in every sample.

### `interpolate` / resize family

Treat mode-specific behavior as part of the family contract. Sample coverage should distinguish:

- `size` path
- `scale_factor` path
- mode-dependent constraints

Do not fold everything into one generic builder.

### normalization / loss / module-wrapper family

Make these explicit in the sample design:

- reduction mode
- affine / bias options
- target / label contract
- semantic error paths

Keep happy-path samples separate from semantic-error samples.

## Example Pattern: `mint.nn.functional.conv2d`

`mint.nn.functional.conv2d` is a useful family-pattern reference because:

- it uses a thin dedicated wrapper pattern rather than a generic callable shim
- it has separate `basic`, `extra`, `dynamic`, and `error` builders
- its sample matrix covers batched/unbatched inputs, groups, padding, dilation, discontiguous inputs, and representative special-value cases
- it encodes `not_support_dtypes` explicitly instead of assuming a complement set
- it carries dtype-specific loss overrides for Ascend 910B
- it is wired into `other_op_db`, `other_op_kbk_db`, `other_op_error_db`, and `other_op_error_kbk_db`

When the target operator looks like `conv2d` in API shape or parameter complexity, copy that **style of solution**. Do not generate a one-off minimal sample builder and call the task done.
