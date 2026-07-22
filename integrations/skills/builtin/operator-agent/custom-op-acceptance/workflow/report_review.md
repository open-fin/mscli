# AI 报告审视

本阶段是交付前门禁。只有 agent 判断报告已经能让审核人做交付判断，才能写入通过结论。

## 输入

- `acceptance_report.md`
- `acceptance_report.json`
- `agent_evidence_requests.json`
- `operator_contract.json`
- `case_plan.json`
- `run_summary.json`

这些是 agent 自检输入，不等同于审核人主交付物；审核人主材料仍是 `acceptance_report.md` 和
`acceptance_report.json`。

## 审视清单

写入 `ai_review.status == "passed"` 前，agent 必须确认：

- 报告首屏是否给出中文审核摘要、自动化结论、阻塞项、待补证据、人工确认项和非阻塞备注。
- `流程门禁` 是否通过：`operator_contract.json` 和 `case_plan.json` 已存在、顶层
  `analysis_status` 均为 `agent_confirmed`、无占位、生成阶段使用了这两个已确认产物、
  记录的算子名一致、生成 case 与计划一致、shape 设计一致、case 映射与运行结果一致。
- 接口契约是否清楚：输入、输出、属性、shape/dtype 规则、PTA 基线、平台限制。
- 用例覆盖是否有价值：shape 按算子规则覆盖关键维度、axis/dim、broadcast、边界、真实模型和非法 shape。
- dtype 结论是否同时基于 PTA/MS 支持矩阵和非错误功能 case 覆盖；不能只看 case 数量，也不能只看支持矩阵。
- pytest/profiler/kernel/显存证据是否完整且可复核。
- 所有失败是否已归因；用例设计错误是否已修正并重跑。
- `agent_evidence_requests.json` 是否为空。
- 人工确认项是否确实不是 agent 可以继续补证的问题。
- 没有把脚本草稿、占位字段、未分析的 `否` 或 `待确认` 写成通过结论。
- 安全编码负向条目没有把 `是` 当成通过；未发现问题时应显示 `否` 并附扫描证据。

## 路由规则

发现问题时回到对应阶段：

| 问题 | 回退阶段 |
|---|---|
| 接口、shape、dtype、PTA 关系不清 | `operator_analysis.md` |
| 覆盖设计不完整或 case 无效 | `case_design.md` |
| 生成用例和计划不一致 | `opinfo_case_generation.md` |
| pytest/profiler 失败或未跑完 | `execution.md` |
| 缺文档、源码、存量测试、backoff、dtype 证据 | `result_collection.md` |
| 报告展示不清楚或单位/标题不符合规范 | `report_generation.md` |

## 输出

当且仅当报告可交付时，在 `run_summary.json.ai_review` 写入：

```json
{
  "status": "passed",
  "summary": "AI 已审视报告、用例覆盖、静态证据和运行结果，报告可供审核人判断交付状态。",
  "reviewer_value": "报告包含接口语义、覆盖范围、dtype 对齐、kernel/显存证据和非阻塞备注。",
  "remaining_issues": []
}
```

然后重新运行 `generate_report.py` 生成最终报告。
