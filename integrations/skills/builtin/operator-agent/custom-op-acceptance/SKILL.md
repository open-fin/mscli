---
name: custom-op-acceptance
description: >
  对 ms_ops 仓库中的 MindSpore 自定义算子执行验收。用于算子验收、自测、转测、
  生成 opinfo/profiler 用例、在 Ascend 上运行用例、核对 MS/PTA 数值性能、
  补齐证据链并生成可提交审核人的验收报告。
---

# MS 自定义算子验收

## 目标

把一个算子验收到“报告可交给审核人判断是否可交付”。最终交给审核人的主产物是：

- `acceptance_report.md`：验收报告，含审核摘要、流程门禁、阻塞缺口、Checklist 明细、dtype/性能/显存结论及最终验收结论。
- `acceptance_report.json`：与 md 同源的机器可读全量报告，含 `audit_summary`、`blocking_gaps`、`checklist_items`、用例与 profiler 明细等结构化字段，供脚本门禁与程序化读取。

过程门禁产物：

- `agent_evidence_requests.json`：待 Agent 补证的条目清单（`section`/`item`/`reason`/`action`/`agent_evidence_key`）；非空时不能交付审核，需回到对应 workflow 阶段补证。为空时可随机器产物保留，但不是审核人主材料。

## 快速执行

在 `ms_ops` 仓库根目录执行：

```bash
python <custom-op-acceptance-skill-dir>/scripts/run_acceptance.py <op> --project-root . --run-pytest
```

非 Ascend 环境只能生成资产和阻塞态报告；不能把环境阻塞写成算子通过或失败。

## 阶段路由

按当前阶段读取对应 workflow；`agent_evidence_requests.json` 非空时回到对应阶段补证、补跑或归因。

| 阶段 | 产物 | 只在该阶段读取 |
|---|---|---|
| 静态盘点 | `operator_meta.json` | `workflow/operator_inventory.md` |
| 接口确认 | `operator_contract.json` | `workflow/operator_analysis.md` |
| 用例设计 | `case_plan.json` | `workflow/case_design.md` |
| 用例生成 | opinfo/profiler pytest、`case_coverage.json` | `workflow/opinfo_case_generation.md` |
| Ascend 执行 | pytest/profiler 日志 | `workflow/execution.md`、`workflow/profiler_validation.md` |
| 结果收集 | `run_summary.json` | `workflow/result_collection.md` |
| 报告生成 | `acceptance_report.*`、`agent_evidence_requests.json` | `workflow/report_generation.md` |
| AI 审视 | `ai_review` | `workflow/report_review.md` |

## 顶层门禁

- 详细验收条件由脚本和对应 workflow 约束；不要手工改报告结论或用口头判断覆盖脚本状态。
- 最终是否可交付只看 `acceptance_report.json.audit_summary.review_status`。
- 不是 `ready_for_review` 时，按 `blocking_gaps`、`agent_evidence_requests`、`manual_review_items`
  回到对应阶段补证、补跑或归因。

## 边界

- 本 skill 产出审核证据和 `ready_for_review` 判断；是否上线或合入由审核流程决定。
- 发现算子实现或测试资产问题时，给出可复现证据和归因；除非用户要求，不扩展成实现修复任务。
