# 静态盘点

本阶段只收集候选元信息，输出不能直接当作最终接口结论。

## 输入

- 算子名：`<op>`，可以是裸名或 `_op` 名。
- `ms_ops` 仓库根目录。

## 脚本动作

运行：

```bash
python <custom-op-acceptance-skill-dir>/scripts/inspect_operator.py <op> \
  --project-root . \
  --output tests/acceptance/results/<op>/operator_meta.json
```

`inspect_operator.py` 从这些 YAML 中提取候选信息：

- `custom_ops/config/function_definitions.yaml`
- `custom_ops/config/aclnn_ops.yaml`
- `custom_ops/config/pyboost_ops.yaml`

搜索顺序是 function_definitions -> aclnn_ops -> pyboost_ops，匹配到第一个即返回。

## 输出

`tests/acceptance/results/<op>/operator_meta.json` 至少包含：

| 字段 | 含义 |
|---|---|
| `operator_name` | canonical 算子名 |
| `kind` | `aclnn` / `function` / `pyboost` |
| `signature` | 候选函数签名 |
| `ms_api` | MS 端 API 路径 |
| `aclnn_api` | ACLNN API，仅 aclnn 类可能存在 |
| `inputs` / `attributes` / `outputs` | 候选输入、属性、输出 |
| `has_backward` | YAML 是否声明反向 |
| `has_pta` / `pta_api` / `pta_baseline_api` | 候选 PTA 基线 |
| `pta_requires_custom_build` | PTA 基线是否依赖本仓库 release 构建产物 |
| `platform_requirements` | 候选平台限制 |

## 门禁

- 文件必须存在且 JSON 可解析。
- 未找到算子时停止，不要凭名字猜测接口。
- 本阶段只产出候选信息；下一阶段必须由 agent 结合源码、文档、存量测试和 PTA 基线确认。
