# Workflow 2: GeneralInfer (C++ Inference)

Path convention: unless stated otherwise, `reference.md` means `../_shared/reference.md` and `aclnn_doc` means `../_shared/aclnn_doc/`.

## Goal

Implement the operator's output shape and dtype inference with the GeneralInfer flow in C++. Pay special attention to dynamic shape and dynamic rank cases, where dimensions or rank may be unknown at this stage. Add corresponding C++ UT infer test.

## Inputs

- **YAML definition**: parameter list and output structure
- **PTA source analysis**: output shape inference logic

## Outputs

- **Infer implementation files**: `mindspore/ops/infer/ops_func_impl`
- **C++ Infer UT**: `tests/ut/cpp/ops/test_ops_{op_name}.cc`

---

## Steps

### Infer

GeneralInfer mainly performs inference through the `InferInfo` family of interfaces.

Basic interface:
```cpp
class OPS_API XXX : public OpFuncImpl {
 public:
  ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  std::vector<TypeId> InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const override;
  bool GeneralInferRegistered() const override { return true; };
};
```
- Use the `InferInfoPtrList`-based Infer APIs and override `GeneralInferRegistered()` to return `true`, which registers the GeneralInfer flow.
- `InferInfo` is defined in `mindspore/core/include/ops/infer_info/infer_info.h`
- The Infer base class is defined in `mindspore/core/include/ops/ops_func_impl/op_func_impl.h`

#### Step 1: Implement `InferShape`

Responsibility boundary (`reference.md#general-infer-responsibilities`):
- **Only perform inference**, do not validate runtime input legality there; leave runtime checks to the kernel
- Use framework exception macros for errors and include the parameter name, expectation, and actual value

#### Step 2: Handle Dynamic Shape/Rank

Three dynamic categories and their strategies (`reference.md#dynamic-shape-strategy`):

| Type | Infer Strategy |
| --- | --- |
| `InputDynamic` | Set the corresponding output dimension to `kShapeDimAny` |
| `Input Value Depend` | Read with `GetShapeValue()`; fall back when unknown |
| `Compute Depend` | Allocate the largest possible size and call `SyncOutputShape` after execution |

Quick fallback rules (`reference.md#general-infer-dynamic-shape-rank`):
- dynamic rank -> return `kShapeRankAny`
- key parameter unknown -> the affected dimensions fall back to `kShapeDimAny`
- all key parameters known -> return the exact shape

#### Step 3: Implement `InferType`

Usually the output dtype either matches the input dtype or is determined by the operator semantics.

#### Step 4: UT test

If op infer is added/modified, its C++ UT test must be added/modified to validate the infer logic. C++ ut tests are placed in `tests/ut/cpp/ops`.

`EluExt` Example:

```c++
#include "ops/utils/general_infer_utils.h"

namespace mindspore::ops {
namespace {
std::vector<GeneralInferParam> prepare_params() {
  GeneralInferParamGenerator generator;
  generator
    .FeedInputArgs({InferInfoParam{ShapeVector{2, 3}, kNumberTypeFloat32},
                    InferInfoParam{ShapeVector{}, kNumberTypeFloat32, CreateScalar<float>(1.)}})
    .FeedExpectedOutput({{2, 3}}, {kNumberTypeFloat32});
  ...
  return generator.Generate();
}
}  // namespace

INSTANTIATE_TEST_CASE_P(EluExt, GeneralInferTest, testing::ValuesIn(prepare_params()));
}  // namespace mindspore::ops
```

Use `GeneralInferParamGenerator` for infer input/output construction. Construct the gtest case using the macro `INSTANTIATE_TEST_CASE_P({PrimitiveName}, GeneralInferTest, testing::ValuesIn(prepare_params()));`.

Typical construction patterns:
- scalar: `ShapeVector{}` + `CreateScalar<T>(value)`
- tuple: `ShapeArray{{}}` + `ValuePtrList{...}`
- `None`: `kMetaTypeNone` + `kNone`
- unknown: `kValueAny`

Related class and function definition can be found in `tests/ut/cpp/ops/utils`.

Refer to existing C++ UT files for similar operators to confirm the testing macros and parameter structures.

---

## Success Criteria

- [ ] `InferShape` is implemented, covering both exact inference and dynamic fallback
- [ ] `InferType` is implemented
- [ ] C++ UT GeneralInferTest is added.

---
