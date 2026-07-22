---
name: op-info-test
description: Generate and locally validate MindSpore Python ST op_info tests. Use when asked to add, repair, or verify op_info-based ST coverage for one or more operators, or run an op_info smoketest on a local Ascend environment.
---

Execute the op_info ST workflow end to end. Prefer direct execution. Ask
questions only when permissions block execution or critical information is
genuinely missing.

Read only the workflow needed for the current step. Load guardrails or
references only when the route below explicitly asks for them.

## Requirement Overview

- Use this skill to add, repair, or verify op_info-based ST coverage.
- Treat the user-requested API form as the default scope.
- Default to local Ascend validation. Do not silently switch to remote
  validation.
- Default to PTA-first dtype probing for new aclnn operators.
- Treat the probe-supported forward dtype set as the candidate writeback set,
  not the final answer.

## Full End-to-End Flow

1. Start with [workflow/dtype_support_discovery.md](workflow/dtype_support_discovery.md) to run the applicability gate and determine dtype support.
2. Continue with [workflow/op_info_generation.md](workflow/op_info_generation.md) to backfill the candidate dtype set, author or update the OpInfo registration, and run the required validation matrix.
3. If the operator has a PTA benchmark interface or the task explicitly requires parity evidence, run [workflow/ms_pta_consistency_validation.md](workflow/ms_pta_consistency_validation.md). If no PTA benchmark interface exists, skip consistency validation and record that blocker in the final summary.
4. If the task explicitly requires stability evidence, or if the new case looks flaky, rerun the passing command with `--count=50` as an optional post-pass stability round.
5. If the required matrix fails, follow the failure-triage order in `op_info_generation.md` before changing dtype declarations.
6. If you need fast iteration on one newly added operator, use [workflow/patch_out_old_tests.md](workflow/patch_out_old_tests.md) as a temporary isolation tactic and remove that patch before final full validation.
7. Write the final summary and finish only after the required matrix and any required consistency step pass, or after the blocking reason is explicitly classified.

## Route Notes

- Load [workflow/other_family_guardrails.md](workflow/other_family_guardrails.md) only for parameter-rich `other` operators such as `conv*`, `linear`, `interpolate`, pooling, normalization, and loss/module-wrapper APIs.
- Use the workflow files for execution details; do not restate their full logic in ad hoc notes or temporary scripts.

## Execution Constraints

- Reuse the bundled workflows, scripts, and templates. Do not rewrite the process from scratch.
- Do not force non-op_info-style APIs into this skill. If the applicability gate fails, record the blocker and use a more appropriate validation path.
- For current aclnn OpInfo tasks, default writeback is `dtypes_ascend910b` and, when needed, `dtypes_backward_ascend910b`. Leave other platform dtype fields empty unless the task explicitly requires them.
- Do not assume `not_support_dtypes` is the complement of `dtypes_ascend910b`.
- Confirm the current machine is a usable Ascend validation machine before running local validation.
- A run with missing `op_error_inputs_func` or `op_dynamic_inputs_func` cannot be reported as `fully_validated`.

## Final Output

- Write `op_info_test_{op_name}_summary.md` in the working directory and do not add it to git. Include covered scenarios, uncovered scenarios, blocking reasons, validation evidence, and MS/PTA consistency evidence when applicable.
- Include error snippets or log paths for failed cases.
- Report coverage gaps separately for `op_error_inputs_func is not set` and `op_dynamic_inputs_func is not set`.
- Include whether dtype support came from `probe_verified`, `doc_derived`, or `probe_vs_doc_conflict`.
- End with one final conclusion category:
  - `fully_validated`
  - `validated_with_coverage_gaps`
  - `blocked_by_testcase_tool_framework`
  - `blocked_by_operator_environment`
