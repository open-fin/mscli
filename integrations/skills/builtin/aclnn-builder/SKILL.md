---
name: aclnn_builder
description: Guides end-to-end ACLNN custom operator development and adaptation in MindSpore (PyBoost/Pynative + KBK/Graph paths), including YAML definitions, code generation, GeneralInfer, kernel registration, bprop wiring, tests (UT), and docs. Use when the user mentions ACLNN adaptation, operator adaptation/operator development, PyBoost, KBK, or Ascend operator adaptation tasks.
---

# ACLNN Operator Development End-to-End Flow (MindSpore Adaptation)

## Goal

Land an ACLNN operator on the Ascend platform in MindSpore **end to end**: forward and backward paths, both mode: PyBoost (Pynative) and KBK (Graph), dynamic shape/rank support, UT, documentation, export, and the required quality checks and validation.

## How To Use This Skill

- When the user says things like "integrate/adapt an ACLNN operator into MindSpore", "add an xxx interface to MindSpore that matches `torch_npu`", "use the skill to add an NPU operator", "add `xxx_op.yaml`", "how should PyBoost/KBK be written", proceed directly through this skill's workflow.

> Shared documents such as `reference.md` and `aclnn_doc` are stored under `../_shared/`.

## Execution Flow

### Workflow Execution Checklist

When using this skill to develop an ACLNN operator, **create a TODOLIST** and execute the following workflows in order.
**Steps marked `🔒 must not be skipped` are mandatory in every scenario.**
**Places marked `⛔ HARD GATE` must be completed before you continue, otherwise stop and wait for user confirmation.**

- [ ] **[Pre](workflows/00-pre-checks.md)** `🔒 must not be skipped`: pre-checks (Pre-A inventory check + Pre-B solution design + Pre-C call-chain analysis)
  - Input: operator name, PTA reference interface
  - **Required outputs**: PTA source review report, initialized Feature document
    > **⛔ HARD GATE**: before entering Step 1, you must confirm that these files have been generated into the workspace
  - After each later step, backfill the corresponding section of the Feature document
- [ ] **[Step 1](workflows/01-yaml-definition.md)**: YAML definition -> backfill Feature(`feature-document.md#feature-yaml-definition`)
  - Input: PTA source review report, Feature document
  - Output: `op_def` for Primitive and  `api_def` for Tensor method or overloaded interface
- [ ] **[Step 2](workflows/02-general-infer.md)**: GeneralInfer + infer UT -> backfill Feature (`feature-document.md#feature-dynamic-shape`) / (`feature-document.md#feature-validation-and-errors`) / (`feature-document.md#feature-test-plan`)
  - Input: YAML, PTA output-shape logic
  - Output: Infer implementation, C++ Infer UT test
- [ ] **[Step 3](workflows/03-aclnn-kernel.md)**: ACLNN kernel (PyBoost + KBK) -> backfill Feature (`feature-document.md#feature-execution-modes`)
  - **Path 1(auto)**: skip handwritten implementation and only validate the auto-generated outputs
  - **Path 2(customize)**: handwrite Customize implementation files
  - Input: YAML, ACLNN invocation details
  - Output: Customize implementation files (Path 2)
- [ ] **[Step 4](workflows/04-bprop.md)**: BPROP implementation -> backfill Feature (`feature-document.md#feature-bprop`)
  - Input: `derivatives.yaml` analysis, backward kernel
  - Output: bprop implementation
- [ ] **[Step 5](workflows/05-export.md)**: export python interface
  - Input: operator primitive, function interface
  - Output: exports under the `mint` namespace; if interface overloads are involved, see (`reference.md#api-overload-adaptation`)
- [ ] **[Step 6](workflows/06-docs.md)**: documentation
  - Input: operator, yaml definition
  - Output: English doc and Chinese doc for newly added operators, plus index updates for public interfaces
- [ ] **[Step 7] Feature document finalization** `🔒 must not be skipped`: complete (`feature-document.md#feature-code-change-summary`)
  - Even if intermediate steps are skipped or deferred, the Feature document must still be completed and delivered to the user.

## Validation Loop (Evidence Required At Every Step) `🔒 must not be skipped`

After every completed step, an execution report **must** be presented to the user using the template below. It may not be omitted, merged away, or deferred.
**This is a mandatory user-facing deliverable, not an internal note.**

```text
━━━ Step X Execution Report ━━━

Execution basis (which skill requirement I followed):
- workflow file: workflows/XX-xxx.md
- corresponding skill requirement: (quote the relevant item from SKILL.md / the workflow)
- success criteria for this step: (copied from the workflow success criteria)

What I did (deliverables):
- ...

Key evidence (code snippets / file paths / search results):
- ...
- Which existing operator implementation I compared against: ...

Validation result:
- ...

Item-by-item success criteria check:
- [ ] Criterion 1: ✅/❌
- [ ] Criterion 2: ✅/❌
- ...

Open issues / risks / next step:
- ...
```

## Review

After all steps are done, review generated code against workflow requirements, with feature document and pta analysis report as reference. Don't do code searching unless potential problems are found. For any problems found, fix them.