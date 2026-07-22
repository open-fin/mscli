# Operator ST Test Case Generation

## Goal

Use this workflow to add or update MindSpore Python ST OpInfo coverage for a
target operator, including functionality, accuracy, dynamic shape, dtype, and
representative error-path coverage.

The target is OpInfo-based coverage, not standalone ad hoc test files.

When a benchmark API exists, the tested API should align with the benchmark
behavior. Inputs and behaviors unsupported by the benchmark do not need to be
forced into the target OpInfo coverage.

## Inputs

- API name
- benchmark API when one exists
- API definition, API documentation, and operator documentation when available
- dtype discovery result from
  [dtype_support_discovery.md](dtype_support_discovery.md)

Default scope rule:

- if the task asks for `mint.xxx`, only add the `mint.xxx` OpInfo unless the
  user explicitly asks for additional API forms

## Outputs

- OpInfo registration in
  `tests/st/ops/share/_op_info/op_database.py`
- companion helpers in split-layout branches when needed:
  - `tests/st/ops/share/_op_info/op_sample_inputs.py`
  - `tests/st/ops/share/_op_info/op_wrappers.py`

## Core Route

1. First finish
   [dtype_support_discovery.md](dtype_support_discovery.md).
2. Use the probe result as the candidate dtype writeback set.
3. Choose the right OpInfo authoring path:
   - regular unary/binary/reduction path
   - custom `other`-type path
   - not-applicable path
4. Run the required OpInfo validation matrix.
5. If validation fails, follow the failure-triage order before changing dtype
   declarations.
6. Only treat the candidate writeback as final after the required matrix passes.

Validation defaults to local execution only:

- confirm the current machine is a usable Ascend validation machine before
  running validation
- do not silently switch to remote validation

If you need fast iteration on one newly added operator, you may temporarily use
[patch_out_old_tests.md](patch_out_old_tests.md). Remove that patch before final
full validation and before leaving the repo in a reviewable state.

## Before Editing

- Reuse the current branch layout. In newer layouts, registration stays in
  `op_database.py` while wrappers and sample/error builders live in
  `_op_info/op_wrappers.py` and `_op_info/op_sample_inputs.py`.
- Do not create separate standalone ST test files for an OpInfo task.
- Similar operators are references only. Do not assume their coverage is
  sufficient for the new target.
- Keep documentation lookup supplemental. For new aclnn operators, the runtime
  probe result is the default candidate source of truth.

## Choose The Right Path

### 1. Regular Unary/Binary/Reduction Operators

Use this path when the operator fits the shared OpInfo framework well.

For unary, binary, and reduction operators, prefer the shared OpInfo sample
helpers first. They already cover many common shape variants, broadcasting,
discontiguous tensors, special values, and extreme values once the operator is
registered and wired into the frontend parameterized tests.

Steps:

1. Choose the correct OpInfo class:
   - `UnaryOpInfo`
   - `BinaryOpInfo`
   - `ReductionOpInfo`
   - `OpInfo` for anything else
2. Add the OpInfo registration in `op_database.py`.
3. Write the candidate forward dtype set into `dtypes_ascend910b` by default.
4. If the probe shows a stable narrower backward dtype set and the task
   requires encoding it, set `dtypes_backward_ascend910b`.
5. Add the operator to the corresponding `xxx_op_db`.
6. Add custom input hooks only when the shared path is not enough.
7. Verify that the matching frontend parameterized test file actually includes
   the operator.
8. If the API has no gradient, set `is_differentiable=False` explicitly.

KBK inclusion rule:

Add the operator to the corresponding `xxx_op_kbk_db` only when one of the
following is true:

- dynamic shape inference is non-trivial
- the implementation is composite
- the operator has frontend API overloading

Do not add it when it is a simple passthrough operator or when the same
implementation pattern is already represented in the KBK list.

### 2. Custom `other`-Type Operators

Use this path for operators that need custom wrappers or custom sample logic.

Rules:

- keep registration and db-list wiring in `op_database.py`
- keep wrappers in `_op_info/op_wrappers.py`
- keep basic/extra/dynamic/error sample builders in `_op_info/op_sample_inputs.py`
- if the branch still uses the all-in-one layout, follow that branch's existing
  pattern instead of forcing split layout

Hook completeness matters. Check explicitly:

- `op_basic_reference_inputs_func`
- `op_extra_reference_inputs_func`
- `op_dynamic_inputs_func`
- `op_error_inputs_func`
- any required loss-override fields
- any explicit `not_support_dtypes`
- whether the operator belongs in:
  - `other_op_db`
  - `other_op_kbk_db`
  - `other_op_error_db`
  - `other_op_error_kbk_db`

Quality rules for custom `other` ops:

1. start from the closest existing family pattern
2. keep wrappers thin and deterministic
3. separate stable samples from edge samples
4. keep one main behavior axis per sample
5. add dynamic and error hooks deliberately instead of omitting them silently
6. do not overfit samples to the probe result

When available in the current branch, reuse existing helper patterns such as
`basic_reference_inputs_binary_op_common_func` and
`_generate_binary_op_broadcasting_and_discontiguous_tensor_inputs_func`. In
split-layout branches, the equivalent sample builders may live in
`_op_info/op_sample_inputs.py` instead of the old all-in-one file.

For parameter-rich `other` families such as `conv*`, `linear`, `interpolate`,
pooling, normalization, and loss/module wrappers, load
[other_family_guardrails.md](other_family_guardrails.md) before generating new
helpers.

### 3. APIs Not Suitable For The Current OpInfo Workflow

Do not force every API into OpInfo.

Treat the API as not suitable when:

- the main behavior is state mutation across steps
- correctness depends on multi-step evolution
- the API depends on distributed environment or communication state
- a single operator-level reference comparison is not meaningful

Typical examples:

- optimizers such as `Adam`, `AdamW`, `SGD`
- distributed/process-group APIs
- lifecycle- or environment-driven interfaces

If the API is not suitable:

1. do not create a placeholder registration
2. record the blocker explicitly
3. recommend the correct alternative validation path

## Validation Requirements

### Coverage Requirements

Use this checklist when designing or reviewing OpInfo samples. Mark items that
do not apply to the target operator as uncovered/not-applicable with the reason;
do not silently drop them.

- [ ] `[MUST]` **Default-parameter scenario validation**: call forward and backward with all default parameter values to confirm the basic path works.
- [ ] `[MUST]` **Dynamic shape self-validation**: the frontend test file calls the `test_op_dynamic` method from `OpsFactory`.
- [ ] `[MUST]` **Empty tensor input**: verify whether forward/backward with empty tensors is supported or raises the correct error.
- [ ] `[MUST]` **Full input dtype coverage**: every dtype declared as supported by the operator must have corresponding cases, including exception cases for unsupported types.
- [ ] `[MUST]` **Input dimension coverage**: include both valid dimensions (covering 0D, 8D, and one intermediate-size dimension if supported) and invalid dimensions.
- [ ] `[MUST]` **Input value range validation**: fully cover boundary values, extreme values (very large/very small), and enumerated parameters such as `margin`/`reduction`.
- [ ] `[MUST]` **Cross-input constraint validation**: shape match/mismatch, dtype same/different, and rank same/different.
- [ ] `[MUST]` **Assert exact error messages for exception cases**: exception scenarios must assert the specific message of `TypeError`/`ValueError`/`RuntimeError`.
- [ ] `[MUST]` **Multi-layout coverage**: if the operator supports multiple layouts (such as BSND/TND/PA_BSND), cover forward and backward for every layout combination.
- [ ] `[MUST]` **Discontiguous tensors**: construct discontiguous inputs using `transpose`/`permute` and verify correctness.
- [ ] `[MUST]` **Special-value robustness**: validate `inf`/`-inf`/`nan` scenarios, at minimum ensuring no crash and correct shape/flow behavior.
- [ ] `[SHOULD]` **Variable-length sequences with multiple batches**: if parameters such as `actual_seq_len` are involved, cover multiple batches plus variable-length scenarios.
- [ ] `[MUST]` **bf16 scenarios**: confirm bf16 support. If supported, test accuracy; otherwise include exception cases. Promote to `float32` before comparison.
- [ ] `[MUST]` **Implicit type conversion**: confirm whether automatic promotion is supported when input dtypes differ; if not, include exception cases.
- [ ] `[MUST]` **Broadcasting**: confirm whether shape broadcasting between inputs is supported; if not, include exception cases.
- [ ] `[MUST]` **Inconsistent dtype across multiple Tensor inputs**: confirm whether the operator supports different dtypes across multiple Tensor inputs; if not, include exception cases. This is not required for operators that do not take multiple Tensor inputs.

Sample construction examples:

| Required scenario | How to write it | Example |
| --- | --- | --- |
| Multiple shapes, including scalar, 1D, middle-rank, and high-rank inputs | create multiple `OpSampleInput` entries with different shapes | `make_arg(())`, `make_arg((S,))`, `make_arg((S, M, S))` |
| Empty tensor | include `0` in at least one valid or expected-error shape | `make_arg((0, S))`, `make_arg((S, 0, M))` |
| Discontiguous tensor | build from transpose/permute or use the local helper flag if present | `make_tensor(shape, discontiguous=True)` |
| Boundary parameter values | cover lower, upper, negative-index, and enumerated parameter boundaries | `dim=0`, `dim=-1`, `p=1`, `p=2`, `p=inf` |
| Large tensor | include one reasonably large shape when the operator and environment can support it | `make_arg((LARGE_DIM_SIZE, M))` |

Keep the matrix representative. Do not build full unsupported-dtype or full
dynamic dtype matrices unless the task explicitly requires them.

### Validation Matrix

Run the required matrix for the requested API form:

- backend x mode (`PyNative` / `KBK`) x shape type (`static` / `dynamic`)

Also require:

- `MS_DISABLE_KERNEL_BACKOFF=1`
- explicit local Ascend precheck before validation
- explicit reporting of coverage gaps from:
  - `op_error_inputs_func is not set`
  - `op_dynamic_inputs_func is not set`

A run with either of those coverage gaps cannot be reported as
`fully_validated`.

If the user explicitly asks for stability evidence, or if the new case looks
prone to intermittent failures, rerun the passing command with `--count=50`.

## Failure Triage Order

If the candidate OpInfo writeback fails validation, do this in order:

1. check whether the testcase itself is wrong
2. cross-check aclnn docs, MindSpore docs, and PTA docs
3. decide whether the issue belongs to:
   - testcase authoring
   - tool/framework logic
   - current op_info-path limitations
   - operator implementation
   - environment setup
4. if the current OpInfo path cannot express the case cleanly, record that
   limitation explicitly
5. only rerun after applying the right fix

Do not shrink the candidate dtype declaration just to make a failing sample
disappear.

Useful mental model:

- the dtype probe discovers declaration boundaries
- OpInfo validation checks whether that candidate declaration is actually stable
  in the real ST path

`mint.add`-style failures are the default reminder for this order:

- special-value comparison mismatches often belong to
  `blocked_by_testcase_tool_framework`
- KBK dynamic adapter failures often belong to
  `blocked_by_testcase_tool_framework`
- genuine runtime unsupported-dtype or operator-side failures belong to
  `blocked_by_operator_environment`

## Final Completion Check

Before marking the task complete, confirm:

- the OpInfo registration exists in the expected location
- the operator is added to the correct db list
- the frontend parameterized test path really covers it
- custom helper hooks are present when needed
- `is_differentiable=False` is set when the API has no gradient
- no standalone throwaway ST file was introduced
- the candidate dtype declaration source is clear:
  - `probe_verified`
  - `doc_derived`
  - `probe_vs_doc_conflict`
- any coverage gaps are explicitly counted and reported
- if the API was not suitable for OpInfo, that decision and the alternative
  validation path are explicitly documented

## Success Criteria

- OpInfo registration is present and wired into the frontend parameterized path
- covered scenarios and uncovered scenarios are explicit
- dtype declaration source is explicit
- coverage gaps are explicit
- the final reported result is one of:
  - `fully_validated`
  - `validated_with_coverage_gaps`
  - `blocked_by_testcase_tool_framework`
  - `blocked_by_operator_environment`
