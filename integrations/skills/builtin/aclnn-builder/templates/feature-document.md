# Feature Document For `{OperatorName}` Operator Development

> **Note**: this document is a **required deliverable** for operator review and test handoff.


<a id="feature-benchmark-api"></a>
## 1. APIs and Benchmarks `[Pre-B Stage]`

- **Reference interface**:  `torch.xxx`
- **Functionality**: {One-sentence description}
- **MindSpore interfaces**:
  - functional: `mindspore.mint.xxx`
  - nn: `mindspore.mint.nn.xxx` (if needed)
  - Tensor: `Tensor.xxx` (if needed)

<a id="feature-functional-spec"></a>
## 3. Functional And API Specification `[Pre-B Stage]`

### Functional Overview

{The operator formula, semantics, and core behavior.}

### Public Interface

```python
mindspore.mint.xxx(
    param1: Tensor,      # [shape], dtype: xxx
    param2: int,         # description
    ...
) -> Tensor | Tuple[Tensor, ...]
```

### Parameter Description

| Parameter | Type | Required/Optional | Default | Description |
| ---- | ---- | -------- | ------ | ---- |
| param1 | Tensor | Required | — | {Description} |
| ... | | | | |

<a id="feature-yaml-definition"></a>
## 4. OP Yaml And API Yaml Definition (Reference) `[After Step 1]`

op_def yaml:
```yaml
# operator xxx
xxx:
    args:
        # {Insert the actual YAML here}
    returns:
        # {Insert the actual YAML here}
    dispatch:
        enable: True
        Ascend: XxxAscend
```

api_def yaml(for tensor method and/or overloaded interface):
```yaml
xxx:
  op_yaml: xxx_op.yaml
  py_method: tensor_xxx
  Ascend: pyboost
  CPU: py_method
  GPU: py_method
  interface: tensor, function
```

<a id="feature-constraints"></a>
## 5. Constraints And Types `[Pre-B Stage]`

- **Input/output dtypes**: {List them}
- **Shapes and ranges**: {List the shape constraints for each input}
<!-- - **Empty Tensor**: {supported / unsupported, with explanation} -->

<a id="feature-execution-modes"></a>
## 6. Execution Modes And Adaptation `[After Step 3]`

### Pynative (PyBoost)
- {Implementation notes}

### Graph (KBK)
- {Implementation notes}

<a id="feature-pta-alignment"></a>
## 7. Differences From PTA And Alignment Status `[Initialized In Pre-B, Completed During Development]`

- **Functional alignment**: {How the implementation aligns with PTA}
- **Numerical accuracy**: {Comparison strategy, such as zero deviation or `rtol/atol`}
- **Differences**: {List the differences from PTA and explain the reasons}

<a id="feature-dynamic-shape"></a>
## 8. Dynamic Shape/Rank Support `[After Step 2]`

- {Dynamic-dimension / dynamic-rank inference strategy}
- {Fallback strategy when compile-time values are unknown}

<a id="feature-validation-and-errors"></a>
## 9. Validation And Error Handling `[After Step 2/3]`

### Inference Phase (Infer)
- {List checked added in inference-time}

### Runtime Phase (ACLNN)
- {List checks added in runtime}

<a id="feature-bprop"></a>
## 10. Backward (BPROP) `[After Step 4]`

- {How BPROP is registered, backward inputs/outputs, and gradient handling}
- If autodiff is used instead, state "no explicit bprop is required"

<a id="feature-test-plan"></a>
## 11. Test Plan `[After Step 2]`

### UT (C++ GeneralInfer)
- {Covered scenarios}

<a id="feature-code-change-summary"></a>
## 12. Code And File Change Summary `[After Development]`

| Category | File Path |
| ---- | -------- |
| Python API | `mindspore/python/mindspore/mint/xxx/yyy.py` |
| YAML | `mindspore/ops/op_def/yaml/xxx_op.yaml`, `mindspore/ops/api_def/xxx.yaml` |
| Infer | `mindspore/ops/infer/ops_func_impl/xxx.cc/.h` |
| PyBoost | `mindspore/ops/kernel/.../customize/xxx.cc/.h` |
| KBK | `mindspore/ops/kernel/.../customize/xxx_aclnn_kernel.cc/.h` |
| BPROP | {Path or "not involved"} |
| Docs (EN) | `mindspore/ops/op_def/yaml/doc/xxx_doc.yaml`, `api_def/function_doc/` |
| Docs (CN) | `docs/api/api_python/ops/mindspore.ops.xxx.rst` |
| Tests (UT) | `tests/ut/cpp/ops/test_ops_xxx.cc` |
