# 接口确认

本阶段由 agent 完成语义确认，把脚本候选信息升级成可用于用例设计的算子契约。

## 输入

- `tests/acceptance/results/<op>/operator_meta.json`
- YAML 定义、C++/Python 源码、文档、存量测试、PTA 基线和历史接入方式。

## Agent 动作

可先生成草稿，避免漏掉必填结构：

```bash
python <custom-op-acceptance-skill-dir>/scripts/scaffold_acceptance_artifacts.py \
  --meta tests/acceptance/results/<op>/operator_meta.json \
  --output-dir tests/acceptance/results/<op> \
  --case-coverage tests/acceptance/results/<op>/case_coverage.json
```

草稿的 `analysis_status` 是 `draft_needs_agent_confirmation`，只用于补证起点；不能作为确认结论。

1. 核对 `operator_meta.json` 的候选签名是否和源码实际入口一致。
2. 找到 MS API、ACLNN API、PTA API 的真实对应关系；PTA 是 `torch_npu.*` 时不额外编译 PTA adapter。
3. 分析输入、属性、输出的语义和约束：必选/可选、默认值、list/tuple、layout、axis/dim、动态 shape/rank。
4. 分析 shape/dtype 规则：输出 shape/dtype 如何由输入和属性推导，是否依赖 runtime data。
5. 分析平台和执行限制：Ascend 型号、Graph/PyNative、release/custom build、kernel backoff。
6. 分析反向：是否有反向、是否单算子、是否需要 input grad guard。
7. 记录每个关键结论的证据来源，优先使用可复核路径和行号。

## 输出

写入 `tests/acceptance/results/<op>/operator_contract.json`：

```json
{
  "operator_name": "...",
  "analysis_status": "agent_confirmed",
  "ms_api": "...",
  "aclnn_api": "...",
  "pta_api": "...",
  "pta_requires_custom_build": false,
  "inputs": [],
  "attributes": [],
  "outputs": [],
  "shape_rule": {
    "input_rule": "...",
    "output_rule": "...",
    "key_dimensions": "...",
    "axis_or_dim": "...",
    "supports_broadcast": false,
    "supports_empty_tensor": false,
    "supports_0d_scalar": false,
    "dynamic_shape_or_rank": "..."
  },
  "dtype_rule": {
    "pta_supported": [],
    "ms_expected": [],
    "dtype_branches": "..."
  },
  "platform_requirements": [],
  "backward_rule": "...",
  "evidence": []
}
```

## 门禁

- `analysis_status` 必须是 `agent_confirmed`，不能是 `draft`、`unknown`、`待确认`。
- `draft_needs_agent_confirmation` 只是 scaffold 草稿状态，必须补齐 evidence 后改为 `agent_confirmed`。
- 任何脚本无法确认的信息，agent 必须先查源码/文档/测试/PTA；确实无法确认时写清楚缺口，并阻塞进入最终交付。
- 不允许把 `operator_meta.json` 原样复制成契约。
- 如果发现脚本解析错了，以 agent 证据为准，并在契约中记录差异原因。
