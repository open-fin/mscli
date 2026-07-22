# 性能验证

运行 profiler 测试，采集 MS/PTA 端到端耗时、kernel 序列、显存占用并做对比。

## 前置条件

- 算子有 PTA 基线（`has_pta: true`），否则跳过。
- 环境已安装 mindspore、torch、torch_npu。

## 执行

```bash
python compile.py --release
python -m pytest tests/acceptance/profiler/test_<op>_profiler.py -v --tb=line --durations=0 -ra
```

`run_acceptance.py --run-pytest` 会自动执行 build → opinfo → profiler。
如果 opinfo 不是 clean pass，`run_acceptance.py` 会停止 profiler，并写入 `case_failure_analysis`
中的 `needs_triage` 记录，要求先判断是用例设计问题还是算子实现问题。

profiler 文件中还包含 dtype 支持矩阵测试：先探测 PTA 支持的 dtype，再验证 MS 是否全部覆盖。
该测试会写出 `results/<op>/dtype_support.json`，用于证明 MS/PTA runtime dtype 支持关系。
最终 dtype 覆盖结论还必须同时检查功能用例中的非错误 dtype 覆盖。

## 测试流程（每个 case）

1. **warm-up** — 分别用 MS 和 PTA 先跑一次，稳定设备状态
2. **MS e2e 计时** — 循环 1000 轮（无 Profiler 开销），`_pynative_executor.sync()` 同步，取平均 us/step
3. **PTA e2e 计时** — 同上，`torch.npu.synchronize()` 同步
4. **MS kernel 提取** — 单独跑一次带 `Profiler`（仅用于 CSV 提取，不计时）
5. **PTA kernel 提取** — 单独跑一次带 `torch_npu.profiler`（仅用于 trace 提取，不计时）
6. **数值校验** — `np.testing.assert_allclose(rtol=1e-3, atol=1e-3)`
7. **kernel 对比** — 比较 MS/PTA kernel 调用序列是否一致
8. **显存对比** — 采集 MS/PTA 峰值显存；ratio 超过 1.10 时写入 `note`，仅备注不阻塞结论

跳过 `expect_error`、`require_grad` 和空 Tensor（shape 含 0）的 case。

## 输出

`results/<op>/profile_summary.json`。原始 e2e 字段保留 `_ms` 单位，报告渲染时统一换算并展示为 `us`：

```json
{
  "operator_name": "...",
  "status": "passed",
  "kernel_status": "passed",
  "memory_status": "passed",
  "cases": [
    {
      "case_id": "shape_2d_fp16",
      "ms_end_to_end_ms": 1.23,
      "pta_end_to_end_ms": 1.45,
      "ms_kernel_total_us": 800.0,
      "pta_kernel_total_us": 820.0,
      "ms_kernel_sequence": ["InterleaveRopeKernel"],
      "pta_kernel_sequence": ["InterleaveRopeKernel"],
      "kernel_consistency": {"status": "passed", ...},
      "ms_peak_memory_mb": 12.5,
      "pta_peak_memory_mb": 13.1,
      "memory_comparison": {"status": "passed", "ratio": 0.954, ...}
    }
  ]
}
```

`results/<op>/dtype_support.json`：

```json
{
  "status": "passed",
  "candidate_dtypes": ["float16", "float32", "bfloat16", "float64", "int32", "int64", "bool"],
  "pta_supported": ["float16", "float32"],
  "ms_supported": ["float16", "float32"],
  "missing_in_ms": [],
  "records": [...]
}
```

`results/<op>/raw_profiler/<case_id>/ms/` 和 `pta/` 下保留原始 profiler 数据。

## 证据有效性规则

- `profile_summary.json` 存在且 `status: passed`，并且每个 case 都有 MS/PTA 显存数值 →
  性能/kernel 可标记通过；显存 ratio 超阈值时只能标记为 `note` / `非阻塞备注`
- `profile_summary.json` 存在但 `status: coverage_gap` → 标注具体差异，不标记通过
- pytest 通过但没有 `profile_summary.json` → 严重缺口，不标记通过
- 显存 API 返回 `not_collected`、空 kernel 序列、全 skipped → 缺口，不标记通过
- `dtype_support.json` 缺失或 `missing_in_ms` 非空 → dtype 支持矩阵不通过
- PTA 支持 dtype 未出现在非错误功能 case 中 → dtype 覆盖不通过
- profiler 中途失败时仍应保留已采集的 `profile_summary.json` 和失败 case 信息
