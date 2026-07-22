# Reference Baseline Triage

Use this file when an apparent accuracy bug may actually come from the compare
framework, test harness, or reference implementation rather than from the
MindSpore operator or kernel.

This reference is especially important for:

- dtype-specific failures such as `float64` or `complex128`
- scalar / 0D inputs
- component tests that compare against TensorFlow, PyTorch, or NumPy
- numerically sensitive operators such as `tan`, `exp`, `log`, `div`,
  `reciprocal`, and `sqrt`

## Goal

Avoid false operator bug reports caused by a broken baseline.

Do not assume that the reference framework path is correct just because it is
named `forward_tensorflow_impl`, `forward_pytorch_impl`, `expected`, or
`standard`.

## Step 1: Verify the Compare Really Uses the Intended Dtype

Trace the dtype from the testcase input through every reference branch.

Mandatory checks:

- testcase input dtype
- target-side runtime dtype
- reference-side runtime dtype
- hidden casts inside helper functions
- special-case branches for scalar, complex, or empty-shape inputs

Look for patterns such as:

- `astype(np.float32)` in a generic `else` branch
- framework variables created with hard-coded `float32`
- baseline outputs cast again before comparison
- scalar branches that skip the normal dtype-preserving path

If the testcase is named `float64` but the reference branch runs `float32`, stop
there first. This is a test bug until proven otherwise.

## Step 2: Inspect the Reference Branch, Not Just the Failing Assertion

When the stack shows something like:

- `forward_cmp()`
- `forward_tensorflow_impl()`
- `forward_pytorch_impl()`
- `allclose_nparray()`

follow the real compare path all the way through.

Do not stop at the final assertion error. Open the helper implementation and
verify:

- which framework is treated as source of truth
- whether dtype is preserved end to end
- whether different dtypes share one fallback branch
- whether shape-0 / scalar inputs go through different code

## Step 3: Reconcile Tolerance With Dtype and Operator Sensitivity

Check whether `rtol` and `atol` are consistent with:

- the intended dtype
- the actual dtype after hidden casts
- the operator's numerical sensitivity

Red flags:

- `float64` compare using `float32`-level baseline
- one fixed tolerance for all dtypes
- trigonometric or reciprocal operators near singular or large-value regions
- comparing very large outputs with the same tolerance policy used for bounded
  outputs

Do not call this a MindSpore precision bug until tolerance and dtype alignment
are both clean.

## Step 4: Use a Minimal Recheck to Confirm the Baseline Bug

Build the smallest possible recheck:

1. use the exact failing input value
2. run the reference framework once in the current branch behavior
3. run it again with the intended dtype preserved
4. compare both against the MindSpore result

Interpretation:

- if MindSpore aligns with the dtype-preserved reference and diverges from the
  casted reference, the bug is in the test baseline
- if both references still disagree with MindSpore, continue operator triage

## Canonical Example: Fake `tan` fp64 Precision Bug

Pattern:

- testcase input is `float64`
- compare helper calls TensorFlow baseline
- TensorFlow path preserves complex dtypes but casts all other types to
  `float32`
- assertion reports a `float64` mismatch on a large `tan(x)` value

Root cause:

- the reference implementation silently downgraded `float64` to `float32`
- `tan` is numerically sensitive, so the lowered-precision baseline produced a
  visibly different value
- the issue is a test-harness bug, not a confirmed operator bug

Fix direction:

- preserve `float64` in the reference branch
- then rerun the original testcase before reopening operator triage

## Checklist

- [ ] testcase dtype matches intended scenario
- [ ] target runtime dtype is confirmed
- [ ] reference runtime dtype is confirmed
- [ ] no hidden cast changes the reference precision
- [ ] scalar / 0D path was checked separately if present
- [ ] tolerance policy matches dtype and operator sensitivity
- [ ] minimal recheck distinguishes baseline bug from operator bug

## Do Not

- do not file an operator precision bug before checking the reference branch
- do not trust testcase names such as `float64` without tracing actual runtime
  dtype
- do not ignore scalar / 0D special-case code
- do not assume a numerically sensitive operator is wrong just because the
  output difference looks large in absolute value
