# 报告生成

本阶段只运行报告脚本并读取输出。不要在本阶段重新推理全部验收规则；脚本门禁是事实来源。

## 输入

- `operator_meta.json`
- `run_summary.json`
- `operator_contract.json`
- `case_plan.json`

`generate_report.py` 会从 `run_summary.json` 或 `--output-dir` 相邻文件读取契约和计划。

## 命令

```bash
python <custom-op-acceptance-skill-dir>/scripts/generate_report.py \
  --meta tests/acceptance/results/<op>/operator_meta.json \
  --run-summary tests/acceptance/results/<op>/run_summary.json \
  --output-dir tests/acceptance/results/<op>
```

脚本输出：

- `acceptance_report.json`
- `acceptance_report.md`
- `agent_evidence_requests.json`

其中 `agent_evidence_requests.json` 是过程门禁产物，用来驱动 agent 补证；不是审核人主材料。

## 读取顺序

1. 先看 `acceptance_report.json.audit_summary`。
2. 再看 `blocking_gaps`、`agent_evidence_requests`、`manual_review_items`。
3. 只有需要核实时才看 case 明细、profile 明细和原始日志。

## 不可绕过的脚本门禁

- 契约和计划必须是 `agent_confirmed`，且无占位内容。
- 用例生成必须使用确认后的契约和计划；算子名、生成 case、shape 设计必须与计划一致。
- pytest/profiler 必须 clean pass；缺失、skip、空测试、环境阻塞都保留 gap。
- dtype、kernel、显存、shape、safety 的通过条件由报告脚本计算，不手工改结论。
- `agent_evidence_requests.json` 非空时报告不能交给审核人；应先按清单补证。为空时可随机器产物保留。
- `ready_for_review` 只接受脚本产出的状态，不由 agent 口头覆盖。

## 回退

| 报告问题 | 回退阶段 |
|---|---|
| 契约或计划缺失/草稿/占位 | `operator_analysis.md` 或 `case_design.md` |
| 生成 case 与计划不一致 | `opinfo_case_generation.md` |
| pytest/profiler 未 clean pass | `execution.md` |
| dtype/kernel/显存证据缺失 | `profiler_validation.md` 或 `result_collection.md` |
| 报告已生成但没有 AI 审视 | `report_review.md` |
