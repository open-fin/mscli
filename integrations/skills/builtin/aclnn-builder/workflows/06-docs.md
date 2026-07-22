# Workflow 6: Documentation

Path convention: unless stated otherwise, `reference.md` means `../_shared/reference.md` and `aclnn_doc` means `../_shared/aclnn_doc/`.

## Goal

Complete the English YAML documentation, Chinese RST documentation, and public interface indexes required by the OP/API YAML added in Step 1.

English and Chinese docs must stay aligned for parameter names, default values, required/optional status, semantics, return values, and examples. They do not need sentence-by-sentence translation, but they must describe the same API contract.

## Inputs

- **OP YAML**: `mindspore/ops/op_def/yaml/{op}_op.yaml`
- **API YAML (optional)**: `mindspore/ops/api_def/{api}.yaml`
- Public export decision from Step 5: `mint`, `ops`, `nn`, `Tensor`, or internal-only

## Deliverable Checklist

Use this table as the source of truth for files that may need to be added or updated. Check only the rows that apply to the operator being adapted.

| Done | Scenario | Add / Update | File Path | Required When |
| --- | --- | --- | --- | --- |
| [ ] | New `op_def` YAML without public `api_def` | Add English op doc YAML | `mindspore/ops/op_def/yaml/doc/{op}_doc.yaml` | A new `mindspore/ops/op_def/yaml/{op}_op.yaml` is added and no separate API YAML doc owns the public interface |
| [ ] | New functional API YAML | Add English function doc YAML | `mindspore/ops/api_def/function_doc/{api}_doc.yaml` | A new functional API is added under `mindspore/ops/api_def/{api}.yaml` |
| [ ] | New Tensor method API YAML | Add English method doc YAML | `mindspore/ops/api_def/method_doc/{method}_doc.yaml` | A new Tensor method is added under `mindspore/ops/api_def/{method}.yaml` |
| [ ] | Public `mindspore.mint` function | Add Chinese RST | `docs/api/api_python/mint/mindspore.mint.func_{api}.rst` | Public `mint` functional interface is exported |
| [ ] | Public `mindspore.ops` function | Add Chinese RST | `docs/api/api_python/ops/mindspore.ops.func_{api}.rst` | Public `ops` functional interface is exported |
| [ ] | Public `mindspore.nn` class/function | Add Chinese RST | `docs/api/api_python/nn/mindspore.nn.{Name}.rst` | Public `nn` interface is exported |
| [ ] | Public Tensor method | Add Chinese Tensor method RST | `docs/api/api_python/mindspore/Tensor/mindspore.Tensor.{method}.rst` | Public `Tensor.{method}` interface is exported |
| [ ] | Public `mint` interface list | Update Chinese index | `docs/api/api_python/mindspore.mint.rst` | Public `mindspore.mint` API is added |
| [ ] | Public `mint` interface list | Update English index | `docs/api/api_python_en/mindspore.mint.rst` | Public `mindspore.mint` API is added |
| [ ] | Public `ops` interface list | Update Chinese index | `docs/api/api_python/mindspore.ops.rst` | Public `mindspore.ops` API is added |
| [ ] | Public `ops` interface list | Update English index | `docs/api/api_python_en/mindspore.ops.rst` | Public `mindspore.ops` API is added |
| [ ] | Public `nn` interface list | Update Chinese index | `docs/api/api_python/mindspore.nn.rst` | Public `mindspore.nn` API is added |
| [ ] | Public `nn` interface list | Update English index | `docs/api/api_python_en/mindspore.nn.rst` | Public `mindspore.nn` API is added |
| [ ] | Public Tensor method list | Update Chinese Tensor index | `docs/api/api_python/mindspore/mindspore.Tensor.rst` | Public Tensor method is added |
| [ ] | Public Tensor method list | Update English Tensor index | `docs/api/api_python_en/mindspore/mindspore.Tensor.rst` | Public Tensor method is added |

---

## Steps

### Step 1: Decide Public Documentation Scope

Classify each interface created or changed by the operator work:

- Public `mint`, `ops`, `nn`, or `Tensor` APIs need Chinese RST and index updates.
- Internal helper interfaces must not be registered in public indexes. Examples include `_ext` interfaces and split helper APIs such as `xxx_tensor` or `xxx_scalar` used only to implement a public wrapper.
- Primitive-only internals may skip Chinese RST only when they are not exported as public APIs. The reason for skipping must be recorded in the final verification.

### Step 2: Add English YAML Documentation

Add exactly one English documentation source for each documented YAML/API surface:

- For a new functional API YAML, add `mindspore/ops/api_def/function_doc/{api}_doc.yaml`.
- For a new Tensor method API YAML, add `mindspore/ops/api_def/method_doc/{method}_doc.yaml`.
- If no API YAML owns the public interface, add `mindspore/ops/op_def/yaml/doc/{op}_doc.yaml` for the new OP YAML.

The YAML doc must contain:
- `desc`: short description of the operator; for public APIs, include principles, formulas, paper references, or other necessary background when appropriate
- `args`: description for each parameter
- `returns`: return-value description
- `examples`: a complete runnable example including imports

### Step 3: Add Chinese RST

For every public API, add or update the matching Chinese RST file from the checklist.

- First inspect similar existing RST files in the same directory and follow their structure.
- Filename, title, and API signature must match the interface exactly. For functional interfaces, the filename usually uses `func_`, while the in-file API name does not.
- The title underline made of `=` must be at least as long as the title.
- Examples must be complete and runnable, including imports.

### Step 4: Update Interface Indexes

Update the matching Chinese and English index files from the checklist for public APIs only.

- Functional APIs are registered under the corresponding `mindspore.{namespace}.rst` file.
- Tensor methods are registered in both `docs/api/api_python/mindspore/mindspore.Tensor.rst` and `docs/api/api_python_en/mindspore/mindspore.Tensor.rst`.
- Keep inserted entries in alphabetical order, matching the local file's existing ordering convention.

### Step 5: Check EN/CN Consistency

Compare English YAML docs and Chinese RST docs against `reference.md#documentation-general-principles`.

| Check Item | Requirement |
| --- | --- |
| Parameter names | Same names in English and Chinese |
| Default values | Same values and formatting intent |
| Required/optional status | Same required/optional behavior |
| Semantics | Same constraints, formulas, dtype/shape rules, and notes |
| Return values | Same structure and meaning |
| Examples | Complete, runnable, and consistent in behavior |

---

## 🔒 Mandatory Check Before Marking Step 6 Complete

```text
Documentation deliverable checklist:

English YAML doc:
  - File path: ___
  - Status: ✅ created / ✅ already existed and updated / ❌ not written (reason: ___)

Chinese RST:
  - File path: ___
  - Status: ✅ newly created / ✅ already exists and covers the new interface / ❌ not written (reason: ___)
  - If skipped: is this an internal operator (not a public API)? yes / no

Interface lists (English + Chinese):
  - Chinese index file: ___
  - English index file: ___
  - Added in alphabetical order? yes / no

EN/CN consistency:
  - Parameter names consistent: yes / no
  - Default values consistent: yes / no
  - Required/optional status consistent: yes / no
  - Semantics and return values consistent: yes / no
  - Examples complete, runnable, and consistent: yes / no
```

> **Public APIs (functional / mint / nn / Tensor) must have Chinese RST.**
> Only **internal operators** that are not exported in `__all__` and do not need public docs may skip it.
> When skipping, you must state the reason explicitly. Silent skipping is not allowed.

## Success Criteria

- [ ] The English YAML doc is complete (`desc` / `args` / `returns` / `examples`)
- [ ] **The Chinese RST file has been created** for public APIs, or the operator is explicitly marked as internal and skippable
- [ ] Parameter names, default values, required/optional status, semantics, return values, and examples are consistent between English and Chinese
- [ ] Examples are complete and runnable, including imports
- [ ] The interface lists have been updated in alphabetical order
- [ ] Filename, title, and interface definition all match

---
