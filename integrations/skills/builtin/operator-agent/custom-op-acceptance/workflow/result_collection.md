# 结果收集

本阶段把固定格式的执行结果和 agent 补证结论合并成 `run_summary.json`。

## 输入

- `operator_meta.json`
- `operator_contract.json`
- `case_plan.json`
- `case_coverage.json`
- pytest/opinfo/profiler 输出
- `profile_summary.json`
- `dtype_support.json`
- 文档、源码、存量测试、backoff 验证和失败归因证据

## 脚本收集

脚本可直接回填：

- `preflight`
- `pytest`
- `profiler`
- `kernel_consistency`
- `profile_summary`
- `dtype_support`
- `case_coverage`
  - `artifact_inputs`：生成阶段是否使用了 `operator_contract.json` 和 `case_plan.json`。
  - `plan_alignment`：`case_plan.json` 中计划 case 与生成 case 的差异。
- `generated_files`
- 原始 profiler 路径
- 结果目录中已有的 `operator_contract.json` 和 `case_plan.json` 会被 `run_acceptance.py`
  读入 `run_summary.json`；缺失时 `run_acceptance.py` 只生成 draft 草稿，仍需 agent 补证确认。
  若分步执行，也可以由 `generate_report.py` 从输出目录相邻文件读取。

## Agent 补证

脚本无法稳定判断的内容由 agent 分析后写入：

- `doc_audit`：中文文档、接口语义、输入输出、平台、样例、测试命令。
- `source_audit`：shape/dtype 推导、动态 shape/rank、反向、安全静态扫描。
- `legacy_test_audit`：存量用例发现、执行命令、结果和不适用原因。
- `backoff_validation`：`MS_DISABLE_KERNEL_BACKOFF=1` 下的真实验证结论。
- `case_failure_analysis`：失败归因。
- `agent_evidence`：对 `agent_evidence_requests` 的补证结论。

`agent_evidence` key 使用 `<section>::<checklist item>`，value 至少包含：

```json
{
  "result": "是",
  "reason": "可复核的判断理由",
  "evidence": ["path/to/file.cpp:42", "pytest case id 或日志摘要"]
}
```

## 输出

`tests/acceptance/results/<op>/run_summary.json`

## 进入报告生成条件

- `operator_contract.json` 和 `case_plan.json` 必须存在，并能通过报告脚本的字段、确认状态、占位内容、case 映射和运行结果校验。
- 静态脚本取不到的信息不能留空；agent 要么补证，要么写入阻塞缺口。
- `agent_evidence_requests.json` 非空时，下一轮先补证或明确阻塞原因。
