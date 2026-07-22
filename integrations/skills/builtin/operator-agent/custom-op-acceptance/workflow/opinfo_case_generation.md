# 用例生成和 OpInfo 验证

本阶段把已确认的接口契约和用例计划对齐到仓库已合入的 `ms_adapter` opinfo 框架。
语义设计在 `case_design.md` 完成，本文件只描述框架注册、覆盖校验和 opinfo 执行。

## 输入

- `tests/acceptance/results/<op>/operator_meta.json`
- `tests/acceptance/results/<op>/operator_contract.json`
- `tests/acceptance/results/<op>/case_plan.json`

当前脚本仍以 `operator_meta.json` 为直接参数；agent 必须用 `operator_contract.json`
和 `case_plan.json` 校验 opinfo 注册和覆盖是否符合计划。

## 生成命令

```bash
python <custom-op-acceptance-skill-dir>/scripts/generate_cases.py \
  --meta tests/acceptance/results/<op>/operator_meta.json \
  --operator-contract tests/acceptance/results/<op>/operator_contract.json \
  --case-plan tests/acceptance/results/<op>/case_plan.json \
  --output-root tests/acceptance \
  --overwrite
```

输出：

| 文件 | 用途 |
|---|---|
| `tests/shared/opinfo/` | 已合入的轻量 opinfo runner |
| `tests/ms_adapter/opinfo/op_database.py` | 算子 OpInfo 注册 |
| `tests/ms_adapter/opinfo/op_sample_inputs.py` | sample input 覆盖 |
| `tests/ms_adapter/opinfo/op_wrappers.py` | MS 调用和 NumPy reference |
| `tests/ms_adapter/opinfo/test_*_ops.py` | generic opinfo pytest driver |
| `profiler/acceptance_profiler.py` | 共享 profiler helper |
| `profiler/test_<op>_profiler.py` | 参数化 profiler 测试 |
| `results/<op>/case_coverage.json` | 生成侧覆盖摘要 |

opinfo 不再生成 `tests/acceptance/opinfo/acceptance_opinfo.py` 或
`test_<op>_opinfo.py`。如果算子尚未注册到 `tests/ms_adapter/opinfo/op_database.py`，
先使用仓库命令生成候选片段，再由 agent 合入到 central database / sample_inputs / wrappers：

```bash
python codegen/ms_adapter/gen_tests.py \
  --ms-adapter-opinfo-scaffold-function <public_func> \
  --output /tmp/ms_adapter_opinfo_<public_func>
```

`case_coverage.json.artifact_inputs` 必须记录是否传入 `operator_contract.json` 和
`case_plan.json`，`plan_alignment` 必须列出计划 case 与覆盖摘要的差异，供 agent 和最终报告复核。

传入已确认 `case_plan.json` 时，`case_coverage.json.shape_design` 必须使用
`case_plan.json.shape_design`；脚本推断的 shape 设计只能作为缺少计划时的草稿，不能覆盖
agent 已确认的用例设计。

## 生成后校验

agent 必须检查：

- `case_coverage.artifact_inputs.operator_contract.provided` 和
  `case_coverage.artifact_inputs.case_plan.provided` 为 `true`，且 `operator_name_matches` 为
  `true`；如果不是，先回到接口确认和用例设计阶段。
- `case_coverage.plan_alignment.missing_generated_case_ids` 为空；否则修正计划或生成逻辑后重跑。
- `case_coverage.shape_design` 与 `case_plan.json.shape_design` 一致。
- `shape_design.analysis_status` 是 `agent_confirmed`、`script_confirmed` 或 `static_confirmed`，且无占位内容。
- 算子已在 `op_database.py` 注册，并出现在 `unary_op_db` / `binary_op_db` / `other_op_db` 之一。
- `op_sample_inputs.py` 的 sample 覆盖 `case_mapping` 里的 shape、dtype、边界和异常语义。
- dtype case 覆盖 PTA 支持 dtype 的全集。
- profiler case 只选择真实模型 shape、关键边界 shape 和性能敏感 shape，不盲目跑所有负例。
- `expect_error`、`require_grad`、empty tensor、broadcast、axis/dim 等标签与 case 语义一致。
- 最终报告会再次校验 `case_plan.json` 中的功能/负例/checklist/shape 映射 case 是否出现在 pytest 通过结果中，
  profiler case 是否出现在 `profile_summary.json` 中。`ms_adapter` opinfo pytest 的 node 粒度是
  `op_name`，逐 shape/dtype 覆盖以 `case_coverage.json` 和 sample input 审视为准。

如果生成结果和 `case_plan.json` 不一致：

1. 判断是计划错误、脚本模板不足，还是算子需要专用生成逻辑。
2. 修正 `case_plan.json`、生成脚本或模板。
3. 重新运行 `generate_cases.py`。
4. 不允许手工改最终报告绕过用例生成问题。

## OpInfo 执行

```bash
CUSTOM_OPS_MODE=release MS_DISABLE_KERNEL_BACKOFF=1 \
  python -m pytest tests/ms_adapter/opinfo/test_<category>_ops.py -k <opinfo_name> -q -ra
```

规则：

- `expect_error` 和 `require_grad` 语义应落在 OpInfo sample builder / runner 中。
- opinfo 必须在 `MS_DISABLE_KERNEL_BACKOFF=1` 下执行，结果写入 `backoff_validation`。
- skipped、xfailed、空测试和占位 skip 都不是 clean pass。

## 失败处理

- case 不符合 `operator_contract.json`：标记 `case_design_error`，修 case 后重跑。
- case 有效但算子失败：标记 `operator_bug`，保留输入、日志和失败摘要。
- 环境问题：标记 `environment_blocked`，保留缺失依赖或设备信息。
- 未归因：标记 `needs_triage`，不得进入可交付报告。
