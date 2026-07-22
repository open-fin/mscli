# Workflow 3: Aclnn Kernel (Pynative, C++)

Path convention: unless stated otherwise, `reference.md` means `../_shared/reference.md` and `aclnn_doc` means `../_shared/aclnn_doc/`.

## Goal

Implement the ACLNN kernel for PyNative and Graph(KBK) mode.
**The workload of this step differs greatly depending on the integration path:**
- **Path 1 (auto-generated)**: `gen_ops.py` already generated the full call code, so this step only needs **validation**
- **Path 2 (Customize)**: you must handwrite kernel implementation

## Inputs

- **Integration path**: auto / customize
- **YAML definition**: parameter list, `dispatch` configuration
- **PTA source analysis**: ACLNN call details and parameter preprocessing logic
- **(Composite scenarios)** ACLNN call chain: sub-operator list and dependency order

## Outputs

- **Path 1**: validated auto-generated aclnn kernel code
- **Path 2**: handwritten aclnn kernel implementation files

---

## Steps

### Path 1(auto-generated): Validate The Auto-Generated Artifacts

1. **Confirm the generated code exists**
   - For PyBoost, check registration `MS_REG_PYBOOST_OP(Ascend, {PrimitiveName})` in `mindspore/ops/kernel/ascend/aclnn/pyboost_impl/auto_generate/pyboost_ascend_ops_*.cc`
   - For KBK, check registration `MS_ACLNN_COMMON_KERNEL_FACTORY_REG({PrimitiveName}, ...)` or `MS_ACLNN_KERNEL_FACTORY_REG({PrimitiveName}, ...)`.
2. **Confirm the generated PyBoost code follows the post-refactor memory pattern**
   - input Tensor arguments are passed through `TensorToDevice(...)`
   - output Tensor arguments are allocated through `TensorMalloc(...)`
   - ACLNN launch uses `DISPATCH_LAUNCH_ACLNN(...)` or the current generated equivalent

### Path 2(customize kernel): Handwrite Customize Files

#### PyBoost

Commit `42ab8b21a20f1789bf48dc6163c627b2a25fe187` refactored PyBoost customize kernels to decouple customize functions from `OpRunner`. New customize functions should be standalone C++ functions whose parameters are the real op arguments and whose return value is the op output tensor structure.

##### Single-Operator Direct Call Pattern

Standard structure after the PyBoost refactor:
1. declare the customize function without `std::shared_ptr<OpRunner>` in both `.h` and `.cc`
2. infer and create output tensors with `InferOutput<N>(prim::kPrim..., args...)` or `InferDynamicOutput(...)`
3. normalize non-Tensor arguments (`tuple -> vector`, `None` handling, scalar extraction, and so on)
4. call `TensorToDevice(...)` for every input Tensor argument
5. call `TensorMalloc(...)` for output Tensor arguments
6. launch ACLNN with `DISPATCH_ACLNN(...)`
7. return the output tensor, output tuple, or output vector directly

Required includes for the common direct-call pattern:

```cpp
#include "kernel/ascend/aclnn/pyboost_impl/pyboost_aclnn_utils.h"
#include "include/pynative/utils/pyboost/tensor_memory.h"
#include "include/pynative/utils/pyboost/infer_output.h"
#include "include/pynative/utils/pyboost/pyboost_utils.h"
#include "primitive/auto_generate/gen_ops_primitive_x.h"
```

Single-output skeleton:

```cpp
tensor::TensorPtr XxxAscendCustomize(const tensor::TensorPtr &input, const ScalarPtr &alpha) {
  auto output = InferOutput<1>(prim::kPrimXxx, input, alpha);
  auto alpha_imm = GetValue<int64_t>(alpha);

  TensorToDevice(input);
  TensorMalloc(output);
  DISPATCH_ACLNN(aclnnXxx, input, alpha_imm, output);
  return output;
}
```

Fixed multi-output skeleton:

```cpp
std::tuple<tensor::TensorPtr, tensor::TensorPtr> XxxAscendCustomize(const tensor::TensorPtr &input) {
  auto outputs = InferOutput<2>(prim::kPrimXxx, input);
  TensorToDevice(input);
  TensorMalloc(outputs);
  DISPATCH_ACLNN(aclnnXxx, input, std::get<0>(outputs), std::get<1>(outputs));
  return outputs;
}
```

Dynamic-output skeleton:

```cpp
std::vector<tensor::TensorPtr> XxxAscendCustomize(const tensor::TensorPtr &input, const ValueTuplePtr &sections) {
  auto outputs = InferDynamicOutput(prim::kPrimXxx, input, sections);
  auto sections_vec = ConvertValueTupleToVector<int64_t>(sections);
  TensorToDevice(input);
  TensorMalloc(outputs);
  DISPATCH_ACLNN(aclnnXxx, input, sections_vec, outputs);
  return outputs;
}
```

Do not use the pre-refactor pattern for new customize kernels:
- do not add `const std::shared_ptr<OpRunner> &op` to the customize function signature
- do not call `OpRunner::InferOpOutput(op, ...)` when `InferOutput<N>` / `InferDynamicOutput` can represent the output contract
- do not call `PyBoostUtils::PrepareOpInputs`, `PyBoostUtils::PrepareOpOutputs`, `PyBoostUtils::MallocOpInputs`, or `PyBoostUtils::MallocOpOutputs` in new code
- do not manually wrap `LAUNCH_ACLNN` in `PyBoostUtils::DispatchRun`; use `DISPATCH_ACLNN(...)`

Rare special cases may still follow an existing repository example that creates a local `OpRunner` internally, but this is an exception. Prefer the standalone pattern above unless the current codebase has an established special-case implementation for the same output/inference behavior.

##### Tensor Memory Rules

- `TensorToDevice(...)`: allocate device memory for a Tensor and copy it to the device. This is MindSpore's implicit copy mechanism with automatic H2D transfer, which differs from torch and is still retained currently. Usage principle: all input Tensors should call `TensorToDevice`.
- `TensorMalloc(...)`: only allocate device memory for a Tensor. Usually used for output Tensors.
- `TensorToDevice` accepts Tensor inputs, `std::optional<TensorPtr>`, `ValueTuplePtr`, and vectors handled by the helper overloads. Pass all real input Tensor carriers to it, including optional Tensor arguments.
- For outputs returned as `TensorPtr`, `std::tuple<TensorPtr...>`, or `std::vector<TensorPtr>`, call `TensorMalloc(output_or_outputs)` before dispatching ACLNN.

##### Composite Operator Pattern (Composing Small C++ Operator APIs)

When the target operator is implemented as a composition of multiple smaller operators (`reference.md#composite-pyboost-pattern`):
1. include `#include "mindspore/ccsrc/include/pynative/utils/pyboost/functions/auto_generate/functions.h"`
2. call small-operator C++ APIs such as `add()`, `mul()`, `sum_ext()`, and so on to compose the logic directly, **without manually calling `LAUNCH_ACLNN`**
3. set `bprop_expander: False` in YAML so each small operator handles autodiff by itself
4. if the large operator already has its own explicit bprop, use `RequireGradGuard(false)` to prevent the small operators from triggering autodiff twice

##### Input Argument Conversion (`reference.md#pyboost-argument-normalization`)

- `tuple` / `list` -> `std::vector<int64_t>`
- optional `None` inputs -> define the `None` semantics and handle them consistently in PyBoost / Infer / KBK
- scalar parameters -> extract using the project's scalar wrapper pattern
- value scalars passed to ACLNN should be converted before dispatch, for example `GetValue<bool>(flag)`, `GetValue<int64_t>(dim)`, or `static_cast<double>(GetValue<float>(epsilon))`
- if ACLNN expects a null tensor argument, pass `nullptr` directly to `DISPATCH_ACLNN`; do not pass a typed null Tensor variable

##### Inplace Op

For inplace operator, its input tensor is used as the output tensor, so there's no need for calling infer and allocating output tensor.

Check `mindspore/ops/kernel/ascend/aclnn/pyboost_impl/customize/inplace_add_ext.cc` as an example for inplace operator pyboost kernel.

Post-refactor inplace pattern:

```cpp
tensor::TensorPtr InplaceXxxAscendCustomize(const tensor::TensorPtr &input, const tensor::TensorPtr &other,
                                            const ScalarPtr &alpha) {
  TensorToDevice(input, other);
  mindspore::pyboost::CheckMemoryOverlap({input, other}, {input});
  DISPATCH_ACLNN(aclnnInplaceXxx, input, other, alpha);
  return input;
}
```

#### Aclnn Kernelmode(KBK mode)

- The aclnn interface name is passed to base Class(`AclnnKernelMod`)'s constructor.
- `GetWorkSpaceInfo()`: call `GetWorkspaceForResize` for workspace allocation
- `Launch()`: call `RunOp` or the equivalent execution path
- For unneeded outputs, override `GetUseLessOutputIdx()`
- non-Tensor type should be extracted from `KernelTensor` to primitive c++ type using `device::ascend::ConvertKernelTensor<>`
- for inplace op, pass input tensor to aclnn interface as the selfRef tensor.
- registration: `MS_ACLNN_KERNEL_FACTORY_REG`

##### Composite Operator Pattern (Meta DSL, `reference.md#composite-kbk-pattern`)

Meta DSL uses C++ graph construction instead of manual `GetWorkSpaceInfo` / `Launch` / `RunOp`, and the framework then handles type inference and autodiff automatically:
1. create a new `.cc` file under `mindspore/ccsrc/frontend/operator/meta_dsl/func_op/`
2. register the operator with `REGISTER_FUNCTION_OP(OpName)` and optionally pass a validation function
3. inside `BeginFunction(OpName, args...) { ... } EndFunction(OpName)`, compose sub-operators with `Call(Prim(SubOp), ...)`
4. the framework handles multi-platform adaptation automatically, so **no handwritten KBK kernel file is needed**

Code skeletons are available in `reference.md#kbk-skeleton` for single operators and `reference.md#composite-kbk-pattern` for Meta DSL, but **the repository's current code remains the final reference**.

---

## Success Criteria

**Path 1**:
- [ ] Confirm the auto-generated PyBoost and Aclnn Kernelmod registration

**Path 2**:
- [ ] Customize PyBoost call function and Aclnn Kernelmod are implemented
- [ ] PyBoost customize function uses the post-refactor standalone signature without `std::shared_ptr<OpRunner> &op`
- [ ] PyBoost output creation uses `InferOutput<N>` / `InferDynamicOutput` or a justified current-repository special-case pattern
- [ ] All PyBoost input Tensors call `TensorToDevice`; all PyBoost output Tensors call `TensorMalloc`
- [ ] PyBoost ACLNN dispatch uses `DISPATCH_ACLNN` or the current equivalent wrapper
- [ ] Argument conversion is correct (`tuple` / `None` / scalar)
- [ ] In composite scenarios, the calculation logic matches PTA, and intermediate tensors are correctly allocated in PyBoost kernel
---
