"""Bilingual (EN/ZH) report strings for performance-agent."""

_STRINGS = {
    # ========= run_analysis.py: report.md =========
    "report_title": {"en": "# Performance Diagnosis Report", "zh": "# 性能诊断报告"},
    "generated": {"en": "Generated:", "zh": "生成时间:"},
    "elapsed": {"en": "Elapsed:", "zh": "耗时:"},
    "data_label": {"en": "Data:", "zh": "数据:"},
    "hardware_label": {"en": "Hardware:", "zh": "硬件:"},
    "verdict_label": {"en": "Verdict:", "zh": "判定:"},
    "exec_summary": {"en": "## Executive Summary", "zh": "## 摘要"},
    "host_bound_summary": {
        "en": "**HOST-BOUND**: NPU utilized only **{dev}%** (kernel: **{kern}%**). Host-device ratio: **{ratio}x**. Enable graph compilation.",
        "zh": "**CPU资源瓶颈**: NPU 利用率仅 **{dev}%**（Kernel: **{kern}%**）。Host-Device 比: **{ratio}x**。建议启用图编译。",
    },
    "model_desc": {
        "en": "Model: ~{layers}-layer Transformer ({scale}), {mlp} MLP, {ga} grad accum.",
        "zh": "模型: ~{layers}层 Transformer（{scale}），{mlp} MLP，{ga} 梯度累积。",
    },
    "timing": {"en": "## Timing", "zh": "## 时间统计"},
    "metric": {"en": "Metric", "zh": "指标"},
    "value": {"en": "Value", "zh": "值"},
    "wall_time": {"en": "Wall time", "zh": "总耗时"},
    "opt_steps": {"en": "Optimizer steps", "zh": "优化器步数"},
    "per_step": {"en": "Per step", "zh": "每步耗时"},
    "dev_util": {"en": "Device utilization", "zh": "Device 利用率"},
    "kern_util": {"en": "Kernel utilization", "zh": "Kernel 利用率"},
    "host_dev_ratio": {"en": "Host-device ratio", "zh": "Host-Device 比"},
    "fwd_bwd_label": {
        "en": "**FW/BW/Autograd:**",
        "zh": "**前向/反向/Autograd:**",
    },
    # MFU section
    "mfu": {"en": "## MFU", "zh": "## MFU（机器算力利用率）"},
    "mfu_metric": {"en": "MFU", "zh": "MFU"},
    "mfu_level": {"en": "Level", "zh": "等级"},
    "mfu_method": {"en": "Method", "zh": "方法"},
    "reliability": {"en": "reliability", "zh": "可靠性"},
    "achieved_tflops": {"en": "Achieved TFLOPS", "zh": "实际 TFLOPS"},
    "peak_tflops": {"en": "Peak TFLOPS", "zh": "峰值 TFLOPS"},
    "step_time": {"en": "Step time", "zh": "步耗时"},
    "gap_to_target": {"en": "Gap to target (55%)", "zh": "与目标 (55%) 的差距"},
    "mfu_rough": {
        "en": "Rough estimate: **{mfu}%** ({method}, low reliability). Provide missing conditions for precise calculation.",
        "zh": "粗略估计: **{mfu}%**（{method}，低可靠性）。请提供缺失条件以进行精确计算。",
    },
    "mfu_missing": {
        "en": "**Missing conditions for precise MFU calculation:**",
        "zh": "**精确 MFU 计算缺失的条件:**",
    },
    "condition": {"en": "Condition", "zh": "条件"},
    "status": {"en": "Status", "zh": "状态"},
    "how_to_provide": {"en": "How to provide", "zh": "提供方式"},
    "available": {"en": "Available", "zh": "已有"},
    "missing_status": {"en": "**Missing**", "zh": "**缺失**"},
    "rerun": {"en": "**Re-run with:**", "zh": "**使用以下命令重新运行:**"},
    "example_config": {
        "en": "**Example model_config.json:**",
        "zh": "**model_config.json 示例:**",
    },
    # Top operators
    "top_ops": {"en": "## Top Operators", "zh": "## Top 算子"},
    "operator": {"en": "Operator", "zh": "算子"},
    "calls": {"en": "Calls", "zh": "调用次数"},
    "total": {"en": "Total", "zh": "总耗时"},
    "share": {"en": "Share", "zh": "占比"},
    "avg": {"en": "Avg", "zh": "平均"},
    "max": {"en": "Max", "zh": "最大"},
    # Top kernels
    "top_kernels": {
        "en": "## Top Device Kernels (from Dequeue ops)",
        "zh": "## Top Device Kernel（来自 Dequeue 算子）",
    },
    "kernel": {"en": "Kernel", "zh": "Kernel"},
    # Bottlenecks
    "bottlenecks": {"en": "## Bottlenecks", "zh": "## 瓶颈分析"},
    "domain": {"en": "Domain", "zh": "领域"},
    "confidence": {"en": "Confidence", "zh": "置信度"},
    "conf_strong": {"en": "strong", "zh": "强"},
    "conf_moderate": {"en": "moderate", "zh": "中"},
    # Suggestions
    "suggestions": {"en": "## Optimization Suggestions", "zh": "## 优化建议"},
    "expected": {"en": "Expected:", "zh": "预期收益:"},
    # Host-bound section in report
    "hb_root_cause": {
        "en": "## Host-Bound Root Cause Analysis",
        "zh": "## CPU资源瓶颈根因分析",
    },
    "primary_cause": {"en": "**Primary cause:**", "zh": "**主要原因:**"},
    "cpu_time_by_cat": {
        "en": "### CPU Time by Category",
        "zh": "### CPU 时间分类",
    },
    "category": {"en": "Category", "zh": "分类"},
    "time_col": {"en": "Time", "zh": "时间"},
    "share_col": {"en": "Share", "zh": "占比"},
    "calls_col": {"en": "Calls", "zh": "调用次数"},
    "dispatch_density": {
        "en": "### Dispatch Density",
        "zh": "### 下发密度",
    },
    "cpu_ops_sec": {"en": "CPU ops/sec:", "zh": "CPU 算子/秒:"},
    "avg_dispatch": {"en": "Avg dispatch:", "zh": "平均下发:"},
    "high_freq_ops": {
        "en": "High-freq small ops:",
        "zh": "高频小算子:",
    },
    "fused_recs": {
        "en": "### Fused Operator Recommendations",
        "zh": "### 融合算子建议",
    },
    "mem_alloc_graph": {
        "en": "**Memory alloc:** `empty_tensor` 2M+ calls → graph mode eliminates",
        "zh": "**内存分配:** `empty_tensor` 200万+次调用 → 图模式可消除",
    },
    "data_move_graph": {
        "en": "**Data movement:** `copy_/view/reshape` → graph mode eliminates",
        "zh": "**数据搬运:** `copy_/view/reshape` → 图模式可消除",
    },
    "full_analysis": {"en": "Full analysis:", "zh": "完整分析:"},
    "non_device_time": {
        "en": "non-device time",
        "zh": "非 Device 时间",
    },
    "of_cpu": {"en": "of CPU", "zh": "的 CPU 时间"},
    "of_all_calls": {"en": "of all calls", "zh": "的总调用"},
    "ops_sec_unit": {"en": "ops/sec", "zh": "算子/秒"},
    "per_op": {"en": "us/op", "zh": "us/算子"},
    "types_count": {"en": "types", "zh": "种"},
    "calls_unit": {"en": "calls", "zh": "次调用"},

    # ========= analyze_host_bound_root_cause.py =========
    "hb_analysis_title": {
        "en": "# Host-Bound Root Cause Analysis",
        "zh": "# CPU资源瓶颈根因分析",
    },
    "hb_wall": {"en": "**Wall:**", "zh": "**总耗时:**"},
    "hb_cpu_ops": {"en": "**CPU ops:**", "zh": "**CPU 算子:**"},
    "hb_device": {"en": "**Device:**", "zh": "**Device:**"},
    "root_cause": {"en": "## Root Cause", "zh": "## 根因"},
    "opt_priority": {
        "en": "Optimization priority:",
        "zh": "优化方向:",
    },
    "cpu_time_category": {
        "en": "## CPU Time by Category",
        "zh": "## CPU 时间分类",
    },
    "dispatch_density_section": {
        "en": "## Dispatch Density",
        "zh": "## 下发密度",
    },
    "high_freq_small": {
        "en": "High-freq small ops (<50us):",
        "zh": "高频小算子（<50us）:",
    },
    "gap_analysis": {
        "en": "## Device-Idle Gap Analysis",
        "zh": "## Device 空闲间隙分析",
    },
    "gap_window": {"en": "Window:", "zh": "窗口:"},
    "gap_coverage": {"en": "device coverage:", "zh": "Device 覆盖率:"},
    "gap_total": {"en": "gaps, total", "zh": "个间隙，总计"},
    "time_in_gaps": {"en": "Time in Gaps", "zh": "间隙内耗时"},
    "top_ops_col": {"en": "Top Ops", "zh": "Top 算子"},
    "slow_ops": {
        "en": "## Slowest CPU Operations",
        "zh": "## 最慢的 CPU 操作",
    },
    "duration": {"en": "Duration", "zh": "耗时"},
    "fusion_recs": {
        "en": "## Fused Operator Recommendations",
        "zh": "## 融合算子建议",
    },
    "involves": {"en": "Involves:", "zh": "涉及:"},
    "recommendation": {"en": "Recommendation:", "zh": "建议:"},
    "graph_mode_title": {
        "en": "### Graph Mode (Most Effective Overall Optimization)",
        "zh": "### 图模式（最有效的整体优化）",
    },
    "recommended_comment": {"en": "Recommended", "zh": "推荐"},
    "npu_graph_comment": {"en": "torch_npu graph mode", "zh": "torch_npu 图模式"},
    # Stack trace suggestion
    "stack_rerun_title": {
        "en": "## Re-run with Stack Trace for Deeper Analysis",
        "zh": "## 重新开启 Stack Trace 进行深度分析",
    },
    "stack_rerun_hint": {
        "en": "Current profiler data does not include Python call stacks (`with_stack=False`). "
              "Re-run profiling with `with_stack=True` to pinpoint the exact code locations "
              "generating these small operators.",
        "zh": "当前 profiler 数据未包含 Python 调用栈（`with_stack=False`）。"
              "请以 `with_stack=True` 重新采集，以定位生成小算子的精确代码位置。",
    },
    "stack_rerun_code_title": {
        "en": "### How to Enable Stack Trace",
        "zh": "### 如何开启 Stack Trace",
    },
    "stack_rerun_overhead_note": {
        "en": "> **Note:** Stack trace collection adds ~10-20% overhead. "
              "Use for targeted debugging, not production benchmarking.",
        "zh": "> **注意:** Stack trace 采集会带来约 10-20% 的性能开销。"
              "仅用于定向调试，不建议在生产 benchmark 中使用。",
    },
    "stack_hotspot_available": {
        "en": "Stack trace data is available. Run `analyze_stack_hotspots.py` to pinpoint Python call sites.",
        "zh": "Stack trace 数据已就绪。运行 `analyze_stack_hotspots.py` 定位 Python 调用栈热点。",
    },
    "stack_note_prefix": {
        "en": "Re-run with `with_stack=True` to identify which Python code generates these ops.",
        "zh": "请以 `with_stack=True` 重新采集，以定位产生这些算子的 Python 代码。",
    },
    # Stack hotspot analysis (analyze_stack_hotspots.py)
    "sh_title": {
        "en": "# Stack Trace Hotspot Analysis",
        "zh": "# Stack Trace 热点分析",
    },
    "sh_ops_count": {
        "en": "**Ops with stack traces:** {count:,}",
        "zh": "**含调用栈的算子数:** {count:,}",
    },
    "sh_top_sites": {
        "en": "## Top Call-Site Hotspots",
        "zh": "## Top 调用栈热点",
    },
    "sh_col_call_site": {"en": "Call Site", "zh": "调用位置"},
    "sh_col_calls": {"en": "Calls", "zh": "调用次数"},
    "sh_col_total": {"en": "Total", "zh": "总耗时"},
    "sh_col_avg": {"en": "Avg", "zh": "平均"},
    "sh_col_top_ops": {"en": "Top Ops", "zh": "Top 算子"},
}


def tr(lang: str, key: str, **fmt) -> str:
    """Get translated string for key in given language."""
    entry = _STRINGS.get(key, {})
    text = entry.get(lang, entry.get("en", key))
    return text.format(**fmt) if fmt else text


# Root cause translations
ROOT_CAUSE_SUMMARIES = {
    "operator_dispatch_overhead": {
        "en": "Excessive operator dispatch overhead: {calls_pct}% of CPU calls are high-frequency small operators "
              "(avg {avg_dispatch}us/op, {cpu_ops_sec:.0f} ops/sec). "
              "{count} small op types, {total_calls:,} calls, total {total_ms:.0f}ms",
        "zh": "算子下发开销过大: {calls_pct}% 的CPU调用是高频小算子 "
              "(avg {avg_dispatch}us/op, {cpu_ops_sec:.0f} ops/sec). "
              "共{count}种小算子, {total_calls:,}次调用, 总耗时{total_ms:.0f}ms",
    },
    "framework_overhead": {
        "en": "Framework/Autograd overhead accounts for {pct:.1f}% of CPU: "
              "Checkpoint={ckpt_ms:.0f}ms, AutogradEngine={ag_ms:.0f}ms, Backward={bwd_ms:.0f}ms",
        "zh": "框架/Autograd开销占CPU {pct:.1f}%: "
              "Checkpoint={ckpt_ms:.0f}ms, AutogradEngine={ag_ms:.0f}ms, Backward={bwd_ms:.0f}ms",
    },
    "gap_checkpoint": {
        "en": "During device idle, checkpoint accounts for {pct}%: {ms:.0f}ms, {calls:,} calls",
        "zh": "Device 空闲期间 checkpoint 占 {pct}%: {ms:.0f}ms, {calls:,}次调用",
    },
    "gap_autograd_engine": {
        "en": "During device idle, autograd_engine accounts for {pct}%: {ms:.0f}ms, {calls:,} calls",
        "zh": "Device 空闲期间 autograd_engine 占 {pct}%: {ms:.0f}ms, {calls:,}次调用",
    },
    "gap_framework_overhead": {
        "en": "During device idle, framework_overhead accounts for {pct}%: {ms:.0f}ms, {calls:,} calls",
        "zh": "Device 空闲期间 framework_overhead 占 {pct}%: {ms:.0f}ms, {calls:,}次调用",
    },
    "slow_io_data": {
        "en": "IO/data operations are slow: {count} ops >50ms, max {max_ms:.0f}ms",
        "zh": "IO/数据操作过慢: {count}个>50ms, 最大{max_ms:.0f}ms",
    },
    "dtype_cast": {
        "en": "Dtype cast overhead: {cast_ms:.0f}ms ({pct:.1f}% of CPU)",
        "zh": "Dtype cast开销: {cast_ms:.0f}ms ({pct:.1f}% of CPU)",
    },
    "general_host_overhead": {
        "en": "CPU overhead is scattered, no single dominant bottleneck",
        "zh": "CPU开销分散，无单一明显瓶颈",
    },
}

ROOT_CAUSE_TITLES = {
    "operator_dispatch_overhead": {
        "en": "Operator Dispatch Overhead",
        "zh": "算子下发开销",
    },
    "framework_overhead": {
        "en": "Framework/Autograd Overhead",
        "zh": "框架/Autograd 开销",
    },
    "slow_io_data": {
        "en": "Slow IO/Data Operations",
        "zh": "IO/数据操作",
    },
    "dtype_cast": {
        "en": "Dtype Cast Overhead",
        "zh": "数据类型转换开销",
    },
    "general_host_overhead": {
        "en": "General Host Overhead",
        "zh": "通用 Host 开销",
    },
}

OPT_PRIORITY = {
    "build_fused_ops": {"en": "build_fused_ops", "zh": "构建融合算子"},
    "reduce_framework_overhead": {"en": "reduce_framework_overhead", "zh": "减少框架开销"},
    "optimize_specific_ops": {"en": "optimize_specific_ops", "zh": "优化特定算子"},
    "optimize_io_data": {"en": "optimize_io_data", "zh": "优化 IO/数据操作"},
    "fix_dtype": {"en": "fix_dtype", "zh": "修复数据类型配置"},
    "enable_graph_mode": {"en": "enable_graph_mode", "zh": "启用图模式"},
}

# Suggestion translations keyed by suggestion ID
SUGGESTION_TITLES = {
    "HOST-01": {"en": "Enable torch.compile (Graph Mode)", "zh": "启用 torch.compile（图模式）"},
    "HOST-03": {"en": "Build Fused Operators to Reduce Dispatch Overhead", "zh": "构建融合算子以减少下发开销"},
    "HOST-04": {"en": "Reduce Framework/Autograd Overhead", "zh": "减少框架/Autograd 开销"},
    "HOST-05": {"en": "Fix Slow IO/Data Operations", "zh": "修复慢速 IO/数据操作"},
    "HOST-06": {"en": "Eliminate Dtype Cast Overhead", "zh": "消除数据类型转换开销"},
    "COMP-02": {"en": "Improve Device Utilization", "zh": "提升 Device 利用率"},
    "NPU-AFFINITY-04": {"en": "Fix Synchronization Stall", "zh": "修复同步阻塞"},
    "HOST-01-SECONDARY": {"en": "Reduce Dtype Cast Overhead", "zh": "减少数据类型转换开销"},
    "COMP-03": {"en": "Optimize Hotspot Operator", "zh": "优化热点算子"},
    "NPU-AFFINITY-01": {"en": "Replace AdamW with NpuFusedAdamW", "zh": "将 AdamW 替换为 NpuFusedAdamW"},
    "HOST-07": {"en": "Enable Stack Trace Profiling to Pinpoint Operator Hotspots", "zh": "开启 Stack Trace 定位算子热点代码"},
}

SUGGESTION_DESCS = {
    "HOST-01": {
        "en": "Host-device ratio > 2x. Graph compilation fuses small ops, eliminates Python overhead.",
        "zh": "Host-Device 比 > 2x。图编译融合小算子，消除 Python 开销。",
    },
    "HOST-03": {
        "en": "Too many small operator launches causing CPU bottleneck. Build fused operators first, then enable graph mode.",
        "zh": "过多小算子下发导致 CPU 瓶颈。优先构建融合算子，然后启用图模式。",
    },
    "HOST-04": {
        "en": "Checkpoint recomputation and autograd engine dominate CPU time. Reduce checkpoint scope and enable graph mode.",
        "zh": "Checkpoint 重计算和 Autograd 引擎占主导 CPU 时间。减少 Checkpoint 范围并启用图模式。",
    },
    "HOST-05": {
        "en": "Specific IO or data loading operations causing sync stalls or idle gaps.",
        "zh": "特定 IO 或数据加载操作导致同步阻塞或空闲间隙。",
    },
    "HOST-06": {
        "en": "Significant FP32/FP16 cast overhead. Unify to BF16 or verify AMP config.",
        "zh": "大量 FP32/FP16 转换开销。统一使用 BF16 或检查 AMP 配置。",
    },
    "COMP-02": {
        "en": "Kernel utilization < 30%. Enable graph compilation (HOST-01) and consider larger batch size.",
        "zh": "Kernel 利用率 < 30%。启用图编译（HOST-01）并考虑增大 batch size。",
    },
    "NPU-AFFINITY-04": {
        "en": "Large sync stalls causing device idle gaps.",
        "zh": "大型同步阻塞导致 Device 空闲间隙。",
    },
    "HOST-01-SECONDARY": {
        "en": "Significant FP32/FP16 cast overhead. Consider BF16 or verify AMP config.",
        "zh": "大量 FP32/FP16 转换开销。考虑使用 BF16 或检查 AMP 配置。",
    },
    "COMP-03": {
        "en": "Single operator dominates compute time. Check for fused variant.",
        "zh": "单个算子主导计算时间。检查是否有融合变体。",
    },
    "NPU-AFFINITY-01": {
        "en": "NPU fused optimizer batches parameter updates.",
        "zh": "NPU 融合优化器批量更新参数。",
    },
    "HOST-07": {
        "en": "with_stack=False in current run. Re-profiling with call stacks reveals which Python functions/lines generate the small operators.",
        "zh": "当前采集未开启调用栈（with_stack=False）。重新采集可揭示哪些 Python 函数/代码行产生了大量小算子。",
    },
}

SUGGESTION_BENEFITS = {
    "HOST-01": {"en": "20-50% throughput improvement", "zh": "20-50% 吞吐量提升"},
    "HOST-03": {"en": "2-3x throughput improvement", "zh": "2-3x 吞吐量提升"},
    "HOST-04": {"en": "20-40% CPU time reduction", "zh": "20-40% CPU 时间减少"},
    "HOST-05": {"en": "Eliminate sync stalls, reduce idle gaps", "zh": "消除同步阻塞，减少空闲间隙"},
    "HOST-06": {"en": "5-10% throughput improvement", "zh": "5-10% 吞吐量提升"},
    "COMP-02": {"en": "2-3x MFU improvement", "zh": "2-3x MFU 提升"},
    "NPU-AFFINITY-04": {"en": "Save hundreds of ms per occurrence", "zh": "每次节省数百毫秒"},
    "HOST-01-SECONDARY": {"en": "5-10% throughput improvement", "zh": "5-10% 吞吐量提升"},
    "COMP-03": {"en": "10-30% step time reduction", "zh": "10-30% 步耗时减少"},
    "NPU-AFFINITY-01": {"en": "5-15% optimizer step reduction", "zh": "5-15% 优化器步耗时减少"},
    "HOST-07": {"en": "Identifies exact code locations for targeted fusion/rewrite", "zh": "定位精确代码位置，便于定向融合/重写"},
}

# Bottleneck title translations keyed by domain
BOTTLENECK_TITLES = {
    "host_framework_overhead": {
        "en": "Host Launch Overhead Dominates",
        "zh": "Host 下发开销占主导",
    },
    "low_mfu": {
        "en": "Device Severely Underutilized",
        "zh": "Device 利用率严重不足",
    },
    "unnecessary_sync": {"en": "Sync Stall", "zh": "同步阻塞"},
    "dtype_cast_overhead": {
        "en": "Dtype Cast Overhead",
        "zh": "数据类型转换开销",
    },
    "operator_hotspot": {"en": "Operator Hotspot", "zh": "算子热点"},
    "operator_dispatch_overhead": {
        "en": "Excessive Operator Dispatch Overhead",
        "zh": "算子下发开销过大",
    },
    "framework_overhead": {
        "en": "Framework/Autograd Overhead",
        "zh": "框架/Autograd 开销",
    },
    "slow_io_data": {
        "en": "Slow IO/Data Operations",
        "zh": "IO/数据操作",
    },
}

# Fusion recommendation translations
FUSION_TITLES = {
    "swiglu": {"en": "SwiGLU Fusion", "zh": "SwiGLU 融合"},
    "norm_activation": {"en": "Norm+Activation Fusion", "zh": "Norm+Activation 融合"},
    "optimizer": {"en": "Optimizer Fusion", "zh": "Optimizer 融合"},
    "dtype_cast": {"en": "Dtype Cast Elimination", "zh": "数据类型转换消除"},
    "copy_view": {"en": "Tensor Copy/View Elimination", "zh": "Tensor Copy/View 消除"},
    "mem_alloc": {"en": "Tensor Memory Allocation Optimization", "zh": "Tensor 内存分配优化"},
    "embedding": {"en": "Embedding Fusion", "zh": "Embedding 融合"},
}

FUSION_DESCS = {
    "swiglu": {
        "en": "Fuse silu+mul+two-path matmul into a single SwiGLU operator",
        "zh": "将 silu+mul+两路 matmul 融合为单个 SwiGLU 算子",
    },
    "norm_activation": {
        "en": "Use fused Norm+Activation operator",
        "zh": "使用融合的 Norm+Activation 算子",
    },
    "optimizer": {
        "en": "Use NpuFusedAdamW for batch parameter updates",
        "zh": "使用 NpuFusedAdamW 批量更新参数",
    },
    "dtype_cast": {
        "en": "Unify to BF16, eliminate FP16/FP32 round-trip conversions",
        "zh": "统一使用 BF16，消除 FP16/FP32 来回转换",
    },
    "copy_view": {
        "en": "Optimize data flow, avoid unnecessary copy/contiguous/view",
        "zh": "优化数据流，避免不必要的 copy/contiguous/view",
    },
    "mem_alloc": {
        "en": "Use memory pool, reduce frequent allocation/deallocation",
        "zh": "使用内存池，减少频繁分配释放",
    },
    "embedding": {
        "en": "Consider Embedding + LayerNorm or Embedding + positional encoding fusion",
        "zh": "考虑 Embedding + LayerNorm 或 Embedding + 位置编码融合",
    },
}

# Gap analysis cause translations
GAP_CAUSE_TITLES = {
    "checkpoint": {"en": "checkpoint", "zh": "checkpoint"},
    "autograd_engine": {"en": "autograd_engine", "zh": "autograd_engine"},
    "framework_overhead": {"en": "framework_overhead", "zh": "framework_overhead"},
}
