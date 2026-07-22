# Workflow 0: Pre-Checks (Pre-A / Pre-B / Pre-C)

Path convention: unless stated otherwise, `reference.md` means `../_shared/reference.md` and `aclnn_doc` means `../_shared/aclnn_doc/`.

## Goal

Before writing code, complete the repository inventory check, reference analysis, solution design, and, for composite scenarios, a call-chain inventory.

## Inputs

- **Interface name**: the API name, and optinoally the target ACLNN interface name
- **PTA reference interface**: `torch_npu.npu_xxx` or `torch.xxx`

## Outputs

- **Inventory result**: which parts of this operator already exist or are missing in the MindSpore repository
- **Solution design document**: interface type, integration category, impact analysis, output as a Markdown file
- **ACLNN call-chain inventory** (for composite scenarios): sub-operator coverage status and rollout plan

---

## Pre-A: Inventory Check

Check current codebase for relevant information of api and operators.

### Steps

**Search Python Interface**: for `mint.xxx` interface, search under `mindspore/python/mindspore/mint`. For tensor method, search in `mindspore/python/mindspore/common/tensor.py`.
**Search YAML**: search for the operator name under `mindspore/ops/op_def/yaml/` and `mindspore/ops/api_def/yaml/`. `op_def` defines the underlying c++ primitive, and `api_def` defines the api. 

### Output Template

```text
Operator inventory check: {OpName}

| Component | Status | File Path | Notes |
| ---- | ---- | -------- | ---- |
| Python API | ✅/❌ | ... | |
| YAML (op_def) | ✅/❌ | ... | |
| YAML (api_def) | ✅/❌ | ... | |

```

---

## Pre-B: Solution Design And Reference Analysis

Analyze the differences between the MindSpore, PTA, and ACLNN interfaces, decide the primitive/interface integration strategy, **choose the integration path (Path 1 auto-generated / Path 2 Customize)**, and initialize the Feature document.

### Steps

1. **PTA source review**: review the three key file categories in `op-plugin` repo (see `reference.md#pta-source-review`)
   - `op_plugin_functions.yaml`: function signatures, parameter types/defaults
   - `derivatives.yaml`: backward registration and differentiable inputs
   - `XxxKernelNpuOpApi.cpp`: actual ACLNN call and parameter preprocessing
   - Check whether PTA has **overloads with the same name** but different signatures
2. **ACLNN interface definition**: look up the corresponding ACLNN document in `_shared/aclnn_doc/` (for example `aclnnAbs.md`)
3. **Five-factor interface analysis** (`reference.md#api-analysis-five-factors`)
   - If there's existing interface and operator in MindSpore, check whether functionality, parameter definitions, and dtypes match with pta
   - Decide **whether a new primitive is needed** and **whether to add a new interface or reuse an existing one**
4. **Choose the YAML strategy** (`reference.md#yaml-three-scenarios`)
   - YAML interface definitions are described in `mindspore/ops/op_def/yaml/README.md`
   - Existing YAML + reuse existing primitive -> add a `dispatch` field
   - Existing YAML + new primitive -> create a new YAML with the `_ext` suffix
   - No YAML exists -> create a new one
   - **Do not** edit the signature of existing primitive.
5. **Choose the integration path (core decision)** (`reference.md#dispatch-path-selection`)
   - Determine whether the MindSpore API parameters can be **passed through unchanged** to the ACLNN interface
   - **Path 1 (auto-generated)**: direct passthrough -> PyBoost and ACLNN kernelmod are auto-generated
   - **Path 2 (Customize)**: parameters require preprocessing -> customized PyBoost and ACLNN kernelmod must be generated
   - Common preprocessing cases: scalar extraction, argument reordering, manual output allocation
   - **This decision determines the implementation workload for all later steps and must be finalized in Pre-B**
   - YAML also supports `type_cast` for simple input type conversion(mainly conversion between scalar<->tensor). If the converted parameters then match the ACLNN interface, Path 1 can still be used.
6. **Produce a PTA difference record**: use the `templates/pta-analysis-report.md` template and generate a file such as `{op_name}_pta_analysis.md`

---

## 🔒 Feature Document Initialization (Must Run After Pre-B, Cannot Be Skipped)

> **This is a required review and test-handoff deliverable.** No matter the scenario, forward or backward, single operator or composite, internal or public, you must generate a Feature document. If you skip this step, later review will fail.

### Steps

1. Copy `templates/feature-document.md` and name it `{operator_name}_Feature.md`
2. Fill the following sections based on the Pre-B analysis results:
   - [1. APIs and Benchmarks](../templates/feature-document.md#feature-benchmark-api)
   - [2. Functional And API Specification](../templates/feature-document.md#feature-functional-spec) (interface signature and parameter descriptions)
   - [3. Constraints And Types](../templates/feature-document.md#feature-constraints) (device, dtype, and shape constraints)
   - [4. Differences From PTA And Alignment Status](../templates/feature-document.md#feature-pta-alignment) (initial version)

---

## Pre-C: ACLNN Call-Chain Analysis And Sub-Operator Inventory (Mandatory For Composite Scenarios)

> Execute this only when PTA C++ uses **multiple smaller ACLNN operators chained together**.
> Skip it if PTA directly calls a single `aclnnXxx`.

### Steps

1. **Extract the ACLNN call chain**: extract all forward and backward `EXEC_NPU_CMD` / `aclnnXxx` calls from the PTA C++ code (see `reference.md#aclnn-callchain-extraction`)
2. **Inventory MindSpore coverage**: search one by one to confirm whether each sub-operator has already been integrated (`reference.md#ms-coverage-inventory`)
3. **Produce the coverage inventory** using `templates/aclnn-callchain-analysis.md`
4. **Plan the rollout order**: leaves first, composite later; follow topological order (`reference.md#callchain-rollout-order`)

---

## Success Criteria

**⛔ HARD GATE: before entering Step 1, the following two items must both be completed and delivered to the user:**
1. ✅ PTA source review report (output of Pre-B, using `templates/pta-analysis-report.md`)
2. ✅ Initialized Feature document

**Important**: "delivered to the user" means generating real `.md` files in the workspace and explicitly telling the user their file paths.
