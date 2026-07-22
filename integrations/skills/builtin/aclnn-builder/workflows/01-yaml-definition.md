# Workflow 1: YAML Definition

## Goal

Define the required YAML for the operator (`op_def` + `api_def` + `function_doc`).

## Inputs

- **Feature document**: integration type, parameter list, input/output definitions
- **PTA source review results**: parameter names, types, defaults, and return structure

## Outputs

- **YAML file**: `mindspore/ops/op_def/yaml/{op_name}_op.yaml`
- **Documentation YAML**: `mindspore/ops/op_def/yaml/doc/`, `mindspore/ops/api_def/function_doc/`, `mindspore/ops/api_def/method_doc/`

---

## OP Yaml

**Must** load skill `yaml-helper` for yaml knowledge. Create yaml for new primitive or modify yaml of existing primitive for aclnn dispatch. 

**This is where the path decision lands in YAML**

**Path 1 (auto-generate)** - direct argument passthrough, no Customize needed:
```yaml
dispatch:
  enable: True
  # omit the Ascend field -> build will auto-generate PyBoost/KBK code
```

**Path 2 (Customize)**:
```yaml
dispatch:
  enable: True
  Ascend: {OpNameAscend}    # specify the customize kernel function name
```

## API Yaml

### Tensor Method

For tensor method, i.e. Tensor.xxx, it's interface should be defined in api yaml in `mindspore/ops/api_def`. Add the api yaml if it doens't exist. The `py_method` field requires a python fallback for the interface. Register it in `mindspore/python/mindspore/ops/tensor_method.py`. If no special reason, just call the op of the `op_yaml` field in the python method.

After registering the tensor interface in api yaml, check if there's old tensor method registration in the Tensor class in `mindspore/python/mindspore/common/tensor.py`, delete it.

**Important: do not skip this step just because there's already a handwritten Python interface for the tensor method!!**

### Overloaded interface

For interface that have different signatures, use api yaml to implement the dispatch behaviour.

### API Doc

For newly added `xxx.yaml` in `mindspore/ops/api_def`, corresponding docs are required in `mindspore/ops/api_def/function_doc` and `mindspore/ops/api_def/method_doc`, according to the interface type. Otherwise running `gen_ops.py` throws error. For now, just add placeholder files to pass the check. 

```yaml
xxx:
  description: |
    placeholder
```

Real docs are to be added in later step.

## `gen_ops.py`

After all yaml files are added/modified, running the python script `mindspore/python/mindspore/ops_generate/gen_ops.py` to generate python and C++ code for ops defined by yaml files.
---

## Success Criteria

- [ ] OP YAML files have been created/modified for related ops
- [ ] API YAML files have been created/modified for related tensor method/overloaded interface
- [ ] `mindspore/python/mindspore/ops_generate/gen_ops.py` executed successfully
---
