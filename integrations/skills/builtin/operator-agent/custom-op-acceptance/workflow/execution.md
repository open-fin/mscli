# Ascend 执行

本阶段只做确定性执行和失败归因入口，不做报告结论包装。

## 前置条件

- `operator_contract.json` 已确认。
- `case_plan.json` 已确认。
- 算子已注册到 `tests/ms_adapter/opinfo`，profiler 测试文件已生成，并与 `case_plan.json` 对齐。
- 当前环境是可运行 MindSpore Ascend 的 release 环境。

## 执行顺序

```bash
python compile.py --release
CUSTOM_OPS_MODE=release MS_DISABLE_KERNEL_BACKOFF=1 \
  python -m pytest tests/ms_adapter/opinfo/test_<category>_ops.py -k <opinfo_name> -q -ra
python -m pytest tests/acceptance/profiler/test_<op>_profiler.py -q -ra
```

当前 `compile.py --release` 默认刷新 in-place `.so` 并构建 wheel；只有需要 wheel-only 定位时才使用
`--no-inplace`，不能作为 acceptance 默认构建命令。
release 构建也默认暂存 AscendC 产物；`--no-ascendc` 只能用于定位构建阻塞，不能作为最终验收命令。

## 停止规则

- opinfo 未注册、未按 `-k <opinfo_name>` 收集到目标算子，或不是 clean pass 时，停止 profiler，先做失败归因。
- pytest return code 为 0 但有 skipped、xfailed、空测试或占位 skip，仍不是 clean pass。
- 环境缺失时记录 `blocked_by_operator_environment`，不要继续伪造运行结果。

## 失败归因

每个失败 case 必须进入 `case_failure_analysis`：

| classification | 含义 |
|---|---|
| `case_design_error` | case 不符合算子契约，修 case 后重跑 |
| `operator_bug` | 有效 case 下算子实现、数值、性能或 kernel 路径失败 |
| `environment_blocked` | 环境、依赖、设备或构建阻塞 |
| `needs_triage` | 自动化发现失败但 agent 尚未完成归因，不得交付 |

## 门禁

- 所有运行命令、返回码、pytest 计数、单 case 结果和关键日志路径都要进入结果收集阶段。
- 失败未归因时不得生成 `ai_review.status == "passed"`。
