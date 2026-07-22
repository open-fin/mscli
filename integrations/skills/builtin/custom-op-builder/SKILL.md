---
name: custom-op-builder
description: Generate one MindSpore C++ source file that adapts a single Ascend ACLNN operator through ms::pynative::AclnnOpRunner, and when requested generate a self-contained Python test script for that generated op. Use when given an operator name, C++ wrapper signature, ACLNN argument order, tensor input rules, scalar attributes, default values, output shapes, output dtypes, exported function name, and test coverage requirements.
---

# Custom Op Builder

## Goal

Produce exactly one C++ adapter for one ACLNN-backed custom operator. The adapter should allocate MindSpore output tensors, launch the ACLNN op with `ms::pynative::AclnnOpRunner`, and return the outputs.

When the user requests tests, produce one standalone Python script for that op. The test script must contain the one-op interface wrapper and test cases in the same file, and should not depend on local wrapper or helper modules.

Use the local style shown below.

## Examples

Load these examples when a new op resembles one of the patterns:

- `references/examples/mhc_post.cc`: one output with same shape and dtype as an input.
- `references/examples/mhc_post_backward.cc`: multiple gradient outputs matching input shapes.
- `references/examples/mhc_pre_sinkhorn.cc`: many outputs with derived shapes, fixed dtypes, and scalar casts.
- `references/examples/mhc_pre_sinkhorn_backward.cc`: backward adapter with saved forward tensors and multiple parameter gradients.
- `references/examples/dense_lightning_indexer_grad_kl_loss.cc`: optional tensors, optional vector attributes, defaults, and scalar attributes.
- `references/examples/npu_grouped_matmul.cc`: tensor list, variable output count, aclnn path selection

Use the examples as style references; derive the final signature, output allocation, ACLNN argument order, and defaults from the user-provided operator interface.

For tests, read `references/test_generation.md` and use the Python examples under `references/examples/python/` only as raw style and coverage references. Generated tests must inline the one-op wrapper instead of importing a shared wrapper.

## References

Documentation for aclnn interface can be found in `../_shared/aclnn_doc`, for example, `../_shared/aclnn_doc/math/abs/docs/aclnnAbs.md`.

Read the matching ACLNN document before coding. Treat `aclnnXxx.md` as the primary source for:

- The `aclnnXxxGetWorkspaceSize` C signature and `LAUNCH_ACLNN_FUNC` argument order.
- Which arguments are true ACLNN parameters versus higher-level wrapper conveniences.
- Output tensor order, shape rules, dtype rules, and input/output dtype constraints.
- Product-specific support notes and known unsupported dtype/layout combinations.

Read `references/pitfalls.md` for encountered problems and anti-patterns. Don't make the same mistakes.

### Benchmark

#### Torch-NPU

If a torch_npu interface `torch_npu.xxx` is provided as the benchmark interface for the custom op, custom op interface and its behaviour should match the benchmark interface exactly. In this case the `op-plugin` repo should be provided as the source of truth for implementation. The signature and inputs/outputs explanation can be found in `{op-plugin-repo}/codegen/templates/_op_plugin_docs.py`. The c++ interface of custom op doesn't support keyword arguments, this gap should be filled by using a python wrapper function.

The adaptation code for the torch_npu interface can typically be found from path like `{op-plugin-repo}/op-plugin/ops/opapi/*.cpp` (or `torch_npu/third_party/op-plugin/op-plugin/ops/opapi/*.cpp` if repo `torch_npu` is provided). For example, `{op-plugin-repo}/op_plugin/ops/opapi/AddmmKernelNpuOpApi.cpp` contains the adaptation logic to `aclnnAddmm` for the interface `torch.addmm`. Translate torch_npu adaptation code into the custom op adaptation, mirroring its implementation, including path selection for different aclnn interfaces.

> use `mindspore::device::ascend::GetOpApiFunc` as the equivalence of `check_aclnn_kernel_available` in the op-plugin codebase.
> Treat this function as an exception. Don't assume other symbols in `mindspore::device::ascend` are available for use in the custom op adapter.

## Inputs To Establish

Before writing code, identify:

- C++ exported function name, usually `npu_<snake_case_op>`.
- ACLNN launch symbol, for example `aclnnMhcPostBackward`.
- Matching ACLNN doc path under `../_shared/aclnn_doc`.
- Runner display name, usually the ACLNN symbol without the `aclnn` prefix.
- Required tensor inputs and optional tensor inputs.
- Required scalar attributes and optional scalar/list/string attributes.
- Defaults for optional attributes.
- Output tensor names, return order, dtypes, and shapes.
- `PYBOOST_CALLER` output count.

Ask only if output shape or dtype cannot be derived safely from the ACLNN document, wrapper source, or reference implementation.

## Example Structure

```cpp
#include <vector>
#include "ms_extension/all.h"

namespace custom {
namespace {
// Helper functions for defaults and output tensor allocation.
}  // namespace

std::vector<ms::Tensor> npu_example_op(/* inputs and attributes */) {
  auto [out0, out1] = GenResultTensors(/* shape sources */);
  auto runner = std::make_shared<ms::pynative::AclnnOpRunner>("ExampleOp");
  runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnExampleOp, input0, attr0, out0, out1));
  runner->Run({input0}, {out0, out1});
  return {out0, out1};
}
}  // namespace custom

PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_example_op", PYBOOST_CALLER(2, custom::npu_example_op));
}
```

**Important**: don't include other mindspore header file execpt for `ms_extension/all.h`. This is the only public header for custom op building.

## Signature Conventions

Use these forms:

```cpp
const ms::Tensor &x
const std::optional<ms::Tensor> &bias_opt
const std::optional<std::vector<int64_t>> &actual_seq_qlen_opt
const std::optional<std::string> &layout_opt
const std::optional<int64_t> &sparse_mode_opt
double scale_value
int64_t num_iters
bool out_flag
```

Use `const std::optional<T> &` for arguments that may be omitted. Keep required scalars non-optional.

When ACLNN expects `int`, accept `int64_t` and cast locally:

```cpp
int num_iters_value = static_cast<int>(num_iters);
```

## Defaults And Optional Inputs

Apply defaults inside the exported C++ function before `LAUNCH_ACLNN_FUNC`:

```cpp
std::string layout = layout_opt.value_or("BSND");
constexpr int64_t default_max = 9223372036854775807LL;
int64_t sparse_mode = sparse_mode_opt.value_or(3);
int64_t pre_tokens = pre_tokens_opt.value_or(default_max);
int64_t next_tokens = next_tokens_opt.value_or(default_max);
```

For optional tensor inputs, pass the `std::optional<ms::Tensor>` value to `LAUNCH_ACLNN_FUNC` when the ACLNN converter needs to preserve `None` as an optional value. Still materialize an empty tensor for `runner->Run` dependency tracking:

```cpp
auto query_rope = query_rope_opt.value_or(ms::Tensor());
auto key_rope = key_rope_opt.value_or(ms::Tensor());
runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnDenseLightningIndexerGradKLLoss, query, key, query_index, key_index,
                                        weights, softmax_max, softmax_sum, softmax_max_index, softmax_sum_index,
                                        query_rope_opt, key_rope_opt, actual_seq_qlen, actual_seq_klen, scale_value,
                                        layout, sparse_mode, pre_tokens, next_tokens, d_query_index, d_key_index,
                                        d_weights, loss));
runner->Run({query, key, query_index, key_index, weights, softmax_max, softmax_sum, softmax_max_index,
             softmax_sum_index, query_rope, key_rope},
            {d_query_index, d_key_index, d_weights, loss});
```

For optional int64_t array or list attributes, prefer converting to `std::make_pair(value, true)` unless that causes a concrete problem for the target operator. This is required for operator families where the array value may change across sequential calls; passing only a vector can cause segmentation faults. Recognize these cases from the ACLNN interface information and default to the pair form for safety:

```cpp
auto actual_seq_qlen = std::make_pair(actual_seq_qlen_opt.value_or(std::vector<int64_t>{}), true);
auto actual_seq_klen = std::make_pair(actual_seq_klen_opt.value_or(std::vector<int64_t>{}), true);
```

## Output Allocation

Allocate every output before constructing the launch function.

Derive output shape and dtype from the ACLNN document first. If the ACLNN document is ambiguous, use torch_npu op-plugin output allocation and meta registrations as confirmation. Keep notes on the source of nontrivial shape/dtype rules before coding; wrong output metadata can fail at `GetWorkspaceSize` or crash later when the result is synchronized.

Same dtype and shape as an input:

```cpp
auto grad_x = ms::Tensor(x.data_type(), x.shape());
```

Fixed dtype and shape:

```cpp
auto loss = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{1});
```

Derived shape:

```cpp
const auto &x_shape = x.shape();
const int64_t bs = x_shape[0];
const int64_t seq_len = x_shape[1];
auto out = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{bs, seq_len, 1});
```

Multiple outputs:

```cpp
std::tuple<ms::Tensor, ms::Tensor> GenResultTensors(const ms::Tensor &x) {
  auto out0 = ms::Tensor(x.data_type(), x.shape());
  auto out1 = ms::Tensor(ms::TypeId::kNumberTypeFloat32, std::vector<int64_t>{1});
  return std::make_tuple(std::move(out0), std::move(out1));
}
```

## Launch Rules

The argument order in `LAUNCH_ACLNN_FUNC` must match the ACLNN workspace-size interface order, excluding only the trailing workspace-size and executor parameters. Place output tensors exactly where the ACLNN interface expects them.

Do not include torch_npu-only wrapper kwargs in `LAUNCH_ACLNN_FUNC` unless the ACLNN document shows the same parameter in `aclnnXxxGetWorkspaceSize`. For example, a torch_npu API may expose a `dst_dtype` kwarg while the selected ACLNN interface expects only an output tensor whose dtype already encodes the requested output type.

Call `runner->Run(inputs, outputs)` after setting the launch function:

- Include tensor inputs, including materialized empty tensors for optional tensor inputs when used.
- Exclude scalar, string, bool, and list attributes.
- Include every allocated output in return order.

Example:

```cpp
runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMhcPostBackward, grad_y, x, h_res, h_out, h_post, grad_x,
                                        grad_h_res, grad_h_out, grad_h_post));
runner->Run({grad_y, x, h_res, h_out, h_post}, {grad_x, grad_h_res, grad_h_out, grad_h_post});
return {grad_x, grad_h_res, grad_h_out, grad_h_post};
```

## Multiple ACLNN Calls

When one exported custom op needs to call multiple ACLNN interfaces, use one `ms::pynative::AclnnOpRunner` per ACLNN call. Allocate every intermediate tensor explicitly before the call that writes it, then pass that intermediate as a tensor input to the later call and include it in the later `runner->Run` input dependency list.

Example pattern:

```cpp
ms::Tensor npu_mul_add(const ms::Tensor &x, const ms::Tensor &y, const ms::Tensor &z) {
  auto tmp = ms::Tensor(x.data_type(), x.shape());
  auto out = ms::Tensor(x.data_type(), x.shape());

  auto mul_runner = std::make_shared<ms::pynative::AclnnOpRunner>("Mul");
  mul_runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnMul, x, y, tmp));
  mul_runner->Run({x, y}, {tmp});

  auto alpha = mindspore::MakeValue<int64_t>(1);
  auto add_runner = std::make_shared<ms::pynative::AclnnOpRunner>("Add");
  add_runner->SetLaunchFunc(LAUNCH_ACLNN_FUNC(aclnnAdd, tmp, z, alpha, out));
  add_runner->Run({tmp, z}, {out});

  return out;
}
```

Important rules for multi-call adapters:

- Do not reuse one `AclnnOpRunner` for multiple ACLNN APIs; construct a separate runner for each ACLNN launch.
- Every intermediate tensor must have known shape and dtype before its producer ACLNN call.
- Put intermediate tensors in the consumer call's `runner->Run` input list so PyBoost tracks the dependency.
- Return only the outputs that are part of the exported custom op contract; internal intermediates do not need to be returned.
- Check each ACLNN workspace-size signature independently. For example, `aclnnAdd` expects `alpha` as `aclScalar *`, so use `mindspore::MakeValue<int64_t>(1)`. Raw `int64_t` is correct only when the ACLNN signature itself says `int64_t`, such as `pre_tokens` and `next_tokens` in `aclnnDenseLightningIndexerGradKLLoss`.

## Export Block

Use the requested exported function name and exact output count:

```cpp
PYBIND11_MODULE(MS_EXTENSION_NAME, m) {
  m.def("npu_mhc_post_backward", PYBOOST_CALLER(4, custom::npu_mhc_post_backward));
}
```

Use `PYBOOST_CALLER(1, ...)` for a single returned tensor.

## Test Generation

Generate tests only when explicitly requested by the user. Read `references/test_generation.md` before writing a test.

The generated test must be a single Python script for one op. It must inline the interface wrapper using `ms.ops.CustomOpBuilder(...).load()` and must not import any local wrapper module or shared local test helper. Its project-file dependency should be limited to the generated kernel source file or files passed to `CustomOpBuilder`.

Cover dtype and value combinations, not just one happy path:

- Supported dtype matrix such as `fp16`, `bf16`, and `fp32` when valid.
- For enumerable input, cover all possible value in test.
- Random, zero, small, negative, and boundary-ish values when valid.
- Minimum, representative, and larger supported shapes.
- Optional tensor absent/present combinations.
- Default and explicit array/list attributes, including sequential calls with changed arrays.
- Default and non-default scalar attribute combinations.
- Every layout or mode that changes output shape or semantics.

Respect input restrictions from the ACLNN documentation. For example, if `aclnnXxx` only supports 4D inputs, do not test 2D or 3D shapes even if the wrapper accepts them.

When a benchmark is provided, compare the output to the benchmark output with zero tolerance. If no reference API exists, write smoke, shape, dtype, determinism, and no-crash checks that still exercise the supported matrix.

## Checklist

Before finishing, verify:

- Includes are minimal and sufficient.
- Output shapes and dtypes match the ACLNN documentation.
- Defaults match the requested API contract.
- Optional tensors are handled consistently in the launch call and `runner->Run`.
- The ACLNN symbol name and capitalization are exact.
- `runner->Run` contains tensors only.
- `PYBOOST_CALLER` count equals the number of returned tensors.
- The file uses `ms::pynative::AclnnOpRunner` and `LAUNCH_ACLNN_FUNC`.
- Requested tests are a single self-contained Python script with an in-file wrapper and broad dtype/value/attribute coverage.
