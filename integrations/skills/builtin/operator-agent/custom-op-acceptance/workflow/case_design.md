# 用例设计

本阶段由 agent 基于 `operator_contract.json` 设计可覆盖 checklist 的用例计划。脚本生成只能执行这个计划，不能替代语义设计。

## 输入

- `tests/acceptance/results/<op>/operator_contract.json`
- `templates/operator_acceptance_checklist.json`
- 已有存量测试、PTA 行为和模型常见用法。

## Shape 设计

shape 不能机械枚举 1D~8D。先填完整模板，再决定 case：

```text
算子名称：
输入 shape 规则：
输出 shape 规则：
关键维度含义：
需要覆盖的 axis / dim：
需要覆盖的边界 shape：
是否支持 broadcast：
是否支持 empty tensor：
是否支持 0D scalar：
是否有 dtype 相关分支：
是否有动态 shape / 动态 rank：
真实模型中常见 shape：
非法 shape / 报错 shape：
case_mapping：
```

`case_mapping` 必须把实际需要覆盖的项映射到具体 case_id，例如边界 shape、真实模型 shape、非法 shape、0D scalar、empty tensor、broadcast、axis/dim。

## 其他覆盖维度

- dtype：以 PTA 支持 dtype 为全集，每个 PTA 支持 dtype 都必须映射到非错误功能 case；
  `dtype_support.json` 只能证明运行时支持关系，不能替代功能用例覆盖。
- 属性组合：覆盖默认值、关键非默认值、边界值和非法值；最多 3 组非默认组合只是默认上限，不是硬规则。
- 特殊值：按算子语义决定是否覆盖空 Tensor、Inf/NaN；Inf/NaN 不能替代有限取值边界。
- 取值范围：为数值 Tensor 设计有限边界值 case，例如负值、接近 0、小正值和较大正值。
- 反向：只在算子声明 backward 且接口语义需要时生成。
- 异常：只设计符合接口约束的非法输入；不要构造与算子无关的错误。
- 性能：选择真实模型 shape 和关键边界 shape，不用所有功能 case 做 profiler。

## 输出

写入 `tests/acceptance/results/<op>/case_plan.json`：

```json
{
  "operator_name": "...",
  "analysis_status": "agent_confirmed",
  "shape_design": {},
  "dtype_plan": {
    "pta_supported": [],
    "ms_expected": [],
    "cases": []
  },
  "functional_cases": [],
  "profiler_cases": [],
  "profiler_cases_reason": "无 PTA 基线或无性能验收场景时说明原因；否则留空",
  "negative_cases": [],
  "negative_cases_reason": "无非法输入/非法 shape 约束时说明原因；否则留空",
  "checklist_mapping": {
    "验收项": ["case_id"]
  }
}
```

## 门禁

- `analysis_status` 必须是 `agent_confirmed`；scaffold 生成的 `draft_needs_agent_confirmation`
  不能进入最终交付。
- `shape_design` 不能包含 `待确认`、`待补充`、`结合算子场景补充`、`由源码/文档/agent_evidence 确认后回填` 等占位内容。
- `negative_cases` 或 `profiler_cases` 为空时，必须填写非占位的 `*_reason` 说明为什么不需要。
- 每个关键 checklist 项要么映射到 case_id，要么明确说明 `不涉及` 或保留阻塞缺口。
- `输入支持的dtype是否全覆盖` 必须能从功能 case 的 dtype 证明，不能只引用 profiler dtype 探测矩阵。
- `输入取值范围是否有验证` 必须有有限取值边界 case；只有 Inf/NaN 时该项仍为缺口。
- 明显违反算子约束的 case 是用例设计错误，不能作为算子实现 bug 上报。
- 生成脚本产出的 case 必须和 `case_plan.json` 对齐；不对齐时修脚本/模板/计划后重跑。
