#!/usr/bin/env python3
"""Single-command performance analysis pipeline for Ascend profiler data.

Usage:
    python run_analysis.py <profiler_data_dir> [--top-n 30] [--force]

Produces all artifacts under <profiler_data_dir>/out/ in one pass.
Uses a cached summary JSON to avoid re-scanning on repeat runs.

For _ascend_pt format with SQLite DB:
  - One GROUP BY scan for operator hotspots (~60s for 28M rows)
  - One GROUP BY scan for queue/device hotspots
  - Results cached in out/meta/_cache.json for instant re-runs
"""
import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_common import (
    infer_hardware, infer_stack_from_root, get_peak_tflops,
    load_optional_json, write_json, write_text,
)
from calculate_mfu import calculate_mfu
from report_i18n import (
    tr, BOTTLENECK_TITLES, SUGGESTION_TITLES, SUGGESTION_DESCS,
    SUGGESTION_BENEFITS, ROOT_CAUSE_SUMMARIES, ROOT_CAUSE_TITLES, OPT_PRIORITY,
)


# ===========================================================================
# Format detection
# ===========================================================================

def detect_format(root: Path) -> str:
    name = root.name.lower()
    if name.endswith("_ascend_pt") or (root / "ASCEND_PROFILER_OUTPUT").exists():
        if _find_db(root):
            return "ascend_pt_db"
        return "ascend_pt_csv"
    if name.endswith("_ascend_ms"):
        return "ascend_ms"
    return "unknown"


def _check_with_stack(profiler_info: dict) -> bool:
    cfg = profiler_info.get("config", {})
    common = cfg.get("common_config", cfg)
    return bool(common.get("with_stack", False))


def _find_db(root: Path) -> Optional[Path]:
    for p in root.rglob("ascend_pytorch_profiler.db"):
        return p
    for p in root.rglob("*.db"):
        if "profiler" in p.name.lower():
            return p
    return None


# ===========================================================================
# Single-pass DB analysis
# ===========================================================================

def analyze_db(db_path: Path, output_dir: Path, top_n: int = 30) -> dict:
    """Run all analysis in minimum number of table scans."""
    db = sqlite3.connect(str(db_path))
    cur = db.cursor()

    # 1. Load string map and enum map
    cur.execute("SELECT id, value FROM STRING_IDS")
    str_map = {r[0]: r[1] for r in cur.fetchall()}
    cur.execute("SELECT id, name FROM ENUM_API_TYPE")
    enum_map = {r[0]: r[1] for r in cur.fetchall()}

    # 2. Wall time (text MIN/MAX, single scan ~10s)
    print("  [a] Computing wall time...")
    t0 = time.time()
    cur.execute("SELECT MIN(startNs), MAX(endNs) FROM PYTORCH_API")
    r = cur.fetchone()
    wall_ns = int(r[1]) - int(r[0])
    print(f"      Done in {time.time()-t0:.1f}s (wall={wall_ns/1e9:.1f}s)")

    # 3. One GROUP BY for all op-level analysis (single scan ~60s)
    print("  [b] Scanning operator hotspots...")
    t0 = time.time()
    cur.execute(f"""
        SELECT name, type, COUNT(*),
               SUM(CAST(endNs AS INTEGER) - CAST(startNs AS INTEGER)),
               AVG(CAST(endNs AS INTEGER) - CAST(startNs AS INTEGER)),
               MAX(CAST(endNs AS INTEGER) - CAST(startNs AS INTEGER))
        FROM PYTORCH_API
        WHERE endNs > startNs
        GROUP BY name, type
        ORDER BY SUM(CAST(endNs AS INTEGER) - CAST(startNs AS INTEGER)) DESC
    """)
    all_rows = cur.fetchall()
    print(f"      Done in {time.time()-t0:.1f}s ({len(all_rows)} groups)")

    db.close()

    # --- Split into op rows (type=50001) and queue rows (type=50002) ---
    all_op_rows = [r for r in all_rows if r[1] == 50001]
    all_queue_rows = [r for r in all_rows if r[1] == 50002]

    # --- Process results ---
    total_op_ns = sum(int(r[3]) for r in all_op_rows)
    total_queue_ns = sum(int(r[3]) for r in all_queue_rows)

    # Top ops
    top_ops = []
    for r in all_op_rows[:top_n]:
        ns = int(r[3])
        top_ops.append({
            "name": str_map.get(r[0], f"ID:{r[0]}"),
            "type": enum_map.get(r[1], str(r[1])),
            "calls": r[2],
            "total_ns": ns,
            "total_ms": round(ns / 1e6, 1),
            "share_pct": round(ns / total_op_ns * 100, 2),
            "avg_us": round(int(r[4]) / 1e3, 1),
            "max_ms": round(int(r[5]) / 1e6, 2),
        })

    # Queue ops
    top_queue = []
    for r in all_queue_rows[:top_n]:
        ns = int(r[3])
        top_queue.append({
            "name": str_map.get(r[0], f"ID:{r[0]}").replace("Dequeue@", "").replace("Enqueue@", ""),
            "calls": r[2],
            "total_ns": ns,
            "total_ms": round(ns / 1e6, 1),
            "share_pct": round(ns / total_queue_ns * 100, 2),
            "avg_us": round(int(r[4]) / 1e3, 1),
            "max_ms": round(int(r[5]) / 1e6, 2),
        })

    # Sync points (from all_op_rows)
    syncs = []
    for r in all_op_rows:
        max_ms = int(r[5]) / 1e6
        if max_ms > 50:
            syncs.append({
                "name": str_map.get(r[0], f"ID:{r[0]}"),
                "calls": r[2],
                "max_ms": round(max_ms, 1),
                "avg_us": round(int(r[4]) / 1e3, 1),
            })
    syncs.sort(key=lambda x: x["max_ms"], reverse=True)

    # Fwd/Bwd split
    bw_ids = {sid for sid, v in str_map.items() if "Backward" in v}
    ag_ids = {sid for sid, v in str_map.items() if v.startswith("autograd::")}
    fw_ns = sum(int(r[3]) for r in all_op_rows if r[0] not in bw_ids and r[0] not in ag_ids)
    bw_ns = sum(int(r[3]) for r in all_op_rows if r[0] in bw_ids)
    ag_ns = sum(int(r[3]) for r in all_op_rows if r[0] in ag_ids)
    total_fba = fw_ns + bw_ns + ag_ns
    fwd_bwd = {
        "forward_ns": fw_ns, "forward_pct": round(fw_ns / total_fba * 100, 1),
        "backward_ns": bw_ns, "backward_pct": round(bw_ns / total_fba * 100, 1),
        "autograd_ns": ag_ns, "autograd_pct": round(ag_ns / total_fba * 100, 1),
        "total_ns": total_fba,
        "fw_bw_ratio": f"1:{bw_ns / fw_ns:.2f}" if fw_ns else "N/A",
    }

    # Model inference (from all rows including queue ops)
    all_counts = {str_map.get(r[0], f"ID:{r[0]}"): r[2] for r in all_rows}
    opt_steps = sum(v for k, v in all_counts.items() if "Optimizer.step" in k)
    loss_count = all_counts.get("aten::cross_entropy_loss", all_counts.get("aten::nll_loss", 0))
    fa_fwd = all_counts.get("Dequeue@aclnnFlashAttentionScore", 0)
    fa_bwd = all_counts.get("Dequeue@aclnnFlashAttentionScoreGrad", 0)
    silu_bwd = all_counts.get("Dequeue@aclnnSiluBackward", 0)
    linear_count = all_counts.get("aten::linear", 0)
    micro_batches = max(loss_count // max(opt_steps, 1), 1)
    layers = fa_fwd // max(loss_count, 1)
    linear_per_layer = linear_count // max(loss_count, 1) // 2

    model = {
        "optimizer": "AdamW",
        "optimizer_steps": opt_steps,
        "loss_calls": loss_count,
        "micro_batches_per_step": micro_batches,
        "gradient_accumulation": micro_batches,
        "inferred_layers": layers,
        "linear_per_layer_fwd": linear_per_layer,
        "mlp_type": "SwiGLU" if silu_bwd > 0 else "Standard",
        "attention_type": "FlashAttention" if fa_fwd > 0 else "Standard",
        "estimated_scale": _estimate_scale(layers, linear_per_layer),
    }

    # Dtype cast overhead
    cast_total_ns = 0
    cast_ops = []
    for r in all_op_rows:
        name = str_map.get(r[0], "")
        if any(p in name for p in ("_npu_dtype_cast", "npu_dtype_cast", "NpuDtypeCastBackward", "aclnnCast")):
            ns = int(r[3])
            cast_total_ns += ns
            cast_ops.append({"name": name, "calls": r[2], "total_ms": round(ns / 1e6, 1)})
    dtype_cast = {"total_ms": round(cast_total_ns / 1e6, 1), "ops": sorted(cast_ops, key=lambda x: x["total_ms"], reverse=True)}

    # Optimizer step timing
    # We'll derive from loss count and wall time
    per_step_s = (wall_ns / 1e9) / max(opt_steps, 1) if opt_steps else 0

    # Device utilization
    dequeue_ns = sum(r["total_ns"] for r in top_queue if "Dequeue" in str_map.get(
        next((rid for rid, v in str_map.items() if v == r["name"]), 0), ""))
    # Approximate: use all Dequeue time as kernel estimate
    kernel_ns = sum(r["total_ns"] for r in top_queue[:15])  # from queue data

    util = {
        "wall_s": round(wall_ns / 1e9, 1),
        "op_s": round(total_op_ns / 1e9, 1),
        "queue_s": round(total_queue_ns / 1e9, 1),
        "kernel_s": round(total_queue_ns / 1e9, 2),
        "device_util_pct": round(total_queue_ns / wall_ns * 100, 1) if wall_ns else 0,
        "kernel_util_pct": round(total_queue_ns / wall_ns * 100, 1) if wall_ns else 0,
        "host_device_ratio": round(total_op_ns / total_queue_ns, 1) if total_queue_ns else 0,
    }

    return {
        "wall_ns": wall_ns,
        "util": util,
        "top_ops": top_ops,
        "top_queue": top_queue,
        "syncs": syncs,
        "fwd_bwd": fwd_bwd,
        "model": model,
        "dtype_cast": dtype_cast,
        "per_step_s": per_step_s,
        "opt_steps": opt_steps,
        "str_map": {str(k): v for k, v in str_map.items()},
        "raw_cpu_op_rows": [
            {"name_id": r[0], "calls": r[2],
             "total_ns": int(r[3]), "avg_ns": int(r[4]), "max_ns": int(r[5])}
            for r in all_op_rows
        ],
        "total_cpu_ns": int(total_op_ns),
        "total_queue_ns": int(total_queue_ns),
    }


def _estimate_scale(layers: int, linear_per_layer: int) -> str:
    if layers <= 0: return "unknown"
    if layers <= 16: return "~1-3B"
    if layers <= 36: return "~7B"
    if layers <= 48: return "~13B"
    if layers <= 64: return "~30B"
    if layers <= 96: return "~70B"
    return "~100B+"


# ===========================================================================
# Bottleneck classification & suggestions (unchanged logic, compact)
# ===========================================================================

def classify_bottlenecks(util, top_ops, fwd_bwd, syncs, dtype_cast, model):
    bns = []
    dev = util.get("device_util_pct", 100)
    kern = util.get("kernel_util_pct", 100)

    if dev < 50:
        bns.append({"rank": 1, "domain": "host_framework_overhead",
            "severity": "CRITICAL" if dev < 40 else "HIGH",
            "title": f"Host Launch Overhead Dominates ({100-dev:.1f}% non-device time)",
            "confidence": "strong",
            "evidence": [f"Device utilization: {dev}%", f"Kernel utilization: {kern}%",
                         f"Host-device ratio: {util.get('host_device_ratio',0)}x",
                         f"Autograd engine: {fwd_bwd.get('autograd_pct',0)}%"],
            "suggestion_id": "HOST-01"})

    if kern < 30:
        bns.append({"rank": len(bns)+1, "domain": "low_mfu",
            "severity": "HIGH" if kern < 20 else "MEDIUM",
            "title": f"Device Severely Underutilized (kernel: {kern}%)",
            "confidence": "strong",
            "evidence": [f"Kernel time: {util.get('kernel_s',0)}s / wall: {util.get('wall_s',0)}s"],
            "suggestion_id": "COMP-02"})

    big_syncs = [s for s in syncs if s["max_ms"] > 100]
    if big_syncs:
        w = big_syncs[0]
        bns.append({"rank": len(bns)+1, "domain": "unnecessary_sync",
            "severity": "HIGH" if w["max_ms"] > 200 else "MEDIUM",
            "title": f"Sync Stall: {w['name']} (max {w['max_ms']}ms)",
            "confidence": "strong",
            "evidence": [f"{s['name']}: max {s['max_ms']}ms" for s in big_syncs[:3]],
            "suggestion_id": "NPU-AFFINITY-04"})

    cast_ms = dtype_cast.get("total_ms", 0)
    total_ms = fwd_bwd.get("total_ns", 0) / 1e6
    if cast_ms > total_ms * 0.03:
        bns.append({"rank": len(bns)+1, "domain": "dtype_cast_overhead",
            "severity": "MEDIUM",
            "title": f"Dtype Cast Overhead ({cast_ms:.0f}ms, {cast_ms/total_ms*100:.1f}%)",
            "confidence": "moderate",
            "evidence": [f"{op['name']}: {op['total_ms']}ms" for op in dtype_cast.get("ops",[])[:3]],
            "suggestion_id": "HOST-01-SECONDARY"})

    if top_ops and top_ops[0]["share_pct"] > 25:
        op = top_ops[0]
        bns.append({"rank": len(bns)+1, "domain": "operator_hotspot",
            "severity": "HIGH" if op["share_pct"] > 35 else "MEDIUM",
            "title": f"Operator Hotspot: {op['name']} ({op['share_pct']}%)",
            "confidence": "strong",
            "evidence": [f"{op['name']}: {op['total_ms']}ms, {op['calls']} calls"],
            "suggestion_id": "COMP-03"})

    for i, b in enumerate(bns):
        b["rank"] = i + 1
    return bns


SUGGESTIONS_MAP = {
    "HOST-01": ("HIGH", "Enable torch.compile (Graph Mode)",
                "Host-device ratio > 2x. Graph compilation fuses small ops, eliminates Python overhead.",
                "20-50% throughput improvement",
                'model = torch.compile(model, mode="max-autotune")'),
    "HOST-03": ("CRITICAL", "Build Fused Operators to Reduce Dispatch Overhead",
                "Too many small operator launches causing CPU bottleneck. Build fused operators first, then enable graph mode.",
                "2-3x throughput improvement",
                '# Priority: fused ops, then graph mode\nmodel = torch.compile(model, mode="max-autotune")'),
    "HOST-04": ("HIGH", "Reduce Framework/Autograd Overhead",
                "Checkpoint recomputation and autograd engine dominate CPU time. Reduce checkpoint scope and enable graph mode.",
                "20-40% CPU time reduction",
                '# Selective checkpointing\nfor layer in heavy_layers:\n    x = checkpoint(layer, x, use_reentrant=False)'),
    "HOST-05": ("HIGH", "Fix Slow IO/Data Operations",
                "Specific IO or data loading operations causing sync stalls or idle gaps.",
                "Eliminate sync stalls, reduce idle gaps",
                '# Reduce sync-inducing ops\nif global_step % 100 == 0:\n    writer.add_scalar("loss", loss.item())'),
    "HOST-06": ("MEDIUM", "Eliminate Dtype Cast Overhead",
                "Significant FP32/FP16 cast overhead. Unify to BF16 or verify AMP config.",
                "5-10% throughput improvement",
                'model = model.to(torch.bfloat16)\nwith torch.autocast(device_type="npu", dtype=torch.bfloat16):\n    output = model(input)'),
    "COMP-02": ("HIGH", "Improve Device Utilization",
                "Kernel utilization < 30%. Enable graph compilation (HOST-01) and consider larger batch size.",
                "2-3x MFU improvement",
                "print(f'Allocated: {torch_npu.npu.memory_allocated()/1e9:.1f} GB')"),
    "NPU-AFFINITY-04": ("HIGH", "Fix Synchronization Stall",
                "Large sync stalls causing device idle gaps.",
                "Save hundreds of ms per occurrence",
                "if global_step % 100 == 0:\n    norm = clip_grad_norm_(...)"),
    "HOST-01-SECONDARY": ("MEDIUM", "Reduce Dtype Cast Overhead",
                          "Significant FP32/FP16 cast overhead. Consider BF16 or verify AMP config.",
                          "5-10% throughput improvement",
                          "model = model.to(torch.bfloat16)"),
    "COMP-03": ("HIGH", "Optimize Hotspot Operator",
                "Single operator dominates compute time. Check for fused variant.",
                "10-30% step time reduction",
                "# Check for fused variant or custom kernel"),
    "HOST-07": ("HIGH", "Enable Stack Trace Profiling to Pinpoint Operator Hotspots",
                "with_stack=False in current run. Re-profiling with call stacks reveals which Python functions/lines generate the small operators.",
                "Identifies exact code locations for targeted fusion/rewrite",
                'with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],\n     with_stack=True, with_modules=True) as prof:'),
}


def build_suggestions(bottlenecks, model, with_stack_enabled=True):
    suggestions = []
    seen = set()
    for bn in bottlenecks:
        sid = bn.get("suggestion_id", "")
        if sid in SUGGESTIONS_MAP and sid not in seen:
            seen.add(sid)
            pri, title, desc, benefit, code = SUGGESTIONS_MAP[sid]
            suggestions.append({"id": sid, "priority": pri, "title": title,
                                "description": desc, "expected_benefit": benefit, "code_example": code})
    if model.get("optimizer") == "AdamW" and "NPU-AFFINITY-01" not in seen:
        suggestions.append({"id": "NPU-AFFINITY-01", "priority": "MEDIUM",
                            "title": "Replace AdamW with NpuFusedAdamW",
                            "description": "NPU fused optimizer batches parameter updates.",
                            "expected_benefit": "5-15% optimizer step reduction",
                            "code_example": "optimizer = torch_npu.optim.NpuFusedAdamW(model.parameters(), lr=1e-4)"})
    has_dispatch_overhead = any(
        bn.get("domain") == "operator_dispatch_overhead" or bn.get("suggestion_id") == "HOST-03"
        for bn in bottlenecks
    )
    if not with_stack_enabled and has_dispatch_overhead and "HOST-07" not in seen:
        suggestions.append({"id": "HOST-07", "priority": "HIGH",
                            "title": "Enable Stack Trace Profiling to Pinpoint Operator Hotspots",
                            "description": "with_stack=False in current run. Re-profiling with call stacks reveals which Python functions/lines generate the small operators.",
                            "expected_benefit": "Identifies exact code locations for targeted fusion/rewrite",
                            "code_example": 'with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],\n     with_stack=True, with_modules=True) as prof:'})
    return suggestions


# ===========================================================================
# MFU computation
# ===========================================================================

def _compute_mfu(data, hardware, model_config_path, num_devices):
    """Attempt MFU calculation and diagnose missing conditions."""
    model = data.get("model", {})
    step_time_s = data.get("per_step_s", 0)

    # Build step data from analysis results
    step_data = None
    if step_time_s > 0:
        queue_s = data["util"].get("queue_s", 0)
        wall_s = data["util"].get("wall_s", 0)
        step_data = {
            "average_step_time_ms": step_time_s * 1000,
            "stage_totals_ms": {
                "compute": queue_s * 1000,
                "other": max((wall_s - queue_s) * 1000, 0),
                "step_total": wall_s * 1000,
            },
        }

    # Merge user-provided model config with inferred data
    user_config = load_optional_json(model_config_path) if model_config_path else None
    merged = dict(user_config) if user_config else {}
    if not merged.get("num_layers") and model.get("inferred_layers"):
        merged["num_layers"] = model["inferred_layers"]
    if not merged.get("ffn_type") and model.get("mlp_type"):
        merged["ffn_type"] = model["mlp_type"].lower()

    result = calculate_mfu(step_data, merged or None, hardware, num_devices)

    # Diagnose each required condition
    conditions = [
        {
            "field": "step_time",
            "available": step_time_s > 0,
            "value": f"{step_time_s:.2f}s" if step_time_s > 0 else None,
            "hint": "Automatically derived from profiler data",
        },
        {
            "field": "hardware_model",
            "available": hardware is not None,
            "value": hardware,
            "hint": "--hardware (e.g. ascend_910b2)",
        },
        {
            "field": "hidden_size",
            "available": bool(merged.get("hidden_size")),
            "value": merged.get("hidden_size"),
            "hint": "--model-config JSON",
        },
        {
            "field": "num_layers",
            "available": bool(merged.get("num_layers")),
            "value": merged.get("num_layers"),
            "hint": "Inferred from profiler, or --model-config JSON",
        },
        {
            "field": "seq_len",
            "available": bool(merged.get("seq_len")),
            "value": merged.get("seq_len"),
            "hint": "--model-config JSON",
        },
        {
            "field": "batch_size",
            "available": bool(merged.get("batch_size")),
            "value": merged.get("batch_size"),
            "hint": "--model-config JSON (micro batch size per forward)",
        },
    ]
    result["conditions"] = conditions
    return result


# ===========================================================================
# Report rendering
# ===========================================================================

def _translate_bottleneck_title(bn: dict, lang: str) -> str:
    """Translate bottleneck title based on domain and language."""
    domain = bn.get("domain", "")
    base_title = BOTTLENECK_TITLES.get(domain, {})
    translated_base = base_title.get(lang, base_title.get("en", bn.get("title", "")))

    orig = bn.get("title", "")
    # For host-bound injected bottlenecks with bilingual titles
    if "title_zh" in bn:
        return bn.get(f"title_{lang}", bn.get("title", translated_base))
    # Preserve parenthesized details from original title
    if "(" in orig:
        suffix = orig[orig.index("("):]
        return f"{translated_base} {suffix}"
    return translated_base


def _translate_confidence(conf: str, lang: str) -> str:
    if lang == "zh":
        return {"strong": "强", "moderate": "中", "weak": "弱"}.get(conf, conf)
    return conf


def render_report(data, bottlenecks, suggestions, mfu_result, root, hardware, elapsed,
                  host_bound_result=None, lang="en", with_stack_enabled=True) -> str:
    util = data["util"]
    model = data["model"]
    top_ops = data["top_ops"]
    top_queue = data["top_queue"]
    fwd_bwd = data["fwd_bwd"]
    dev = util.get("device_util_pct", 100)
    kern = util.get("kernel_util_pct", 100)
    verdict = "host_bound" if dev < 50 else ("compute_bound" if kern > 70 else "balanced")

    L = [tr(lang, "report_title"),
         f"**{tr(lang, 'generated')}** {time.strftime('%Y-%m-%d %H:%M')} | **{tr(lang, 'elapsed')}** {elapsed:.1f}s",
         f"**{tr(lang, 'data_label')}** `{root.name}` | **{tr(lang, 'hardware_label')}** {hardware or 'Unknown'} | **{tr(lang, 'verdict_label')}** {verdict.upper()}\n",
         f"{tr(lang, 'exec_summary')}\n"]
    if verdict == "host_bound":
        L.append(tr(lang, "host_bound_summary", dev=dev, kern=kern,
                    ratio=util.get('host_device_ratio', 0)))
    if model.get("inferred_layers"):
        L.append(tr(lang, "model_desc", layers=model['inferred_layers'],
                    scale=model.get('estimated_scale', '?'),
                    mlp=model.get('mlp_type', '?'),
                    ga=model.get('gradient_accumulation', '?')))

    L.append(f"{tr(lang, 'timing')}\n")
    L.append(f"| {tr(lang, 'metric')} | {tr(lang, 'value')} |")
    L.append(f"|--------|-------|")
    L.append(f"| {tr(lang, 'wall_time')} | {util.get('wall_s',0)}s |")
    L.append(f"| {tr(lang, 'opt_steps')} | {data.get('opt_steps',0)} |")
    L.append(f"| {tr(lang, 'per_step')} | {data.get('per_step_s',0):.2f}s |")
    L.append(f"| {tr(lang, 'dev_util')} | **{dev}%** |")
    L.append(f"| {tr(lang, 'kern_util')} | **{kern}%** |")
    L.append(f"| {tr(lang, 'host_dev_ratio')} | **{util.get('host_device_ratio',0)}x** |\n")

    if fwd_bwd:
        L.append(f"{tr(lang, 'fwd_bwd_label')} {fwd_bwd['forward_pct']}% / {fwd_bwd['backward_pct']}% / {fwd_bwd['autograd_pct']}% "
                 f"(FW:BW = {fwd_bwd['fw_bw_ratio']})\n")

    # MFU section
    if mfu_result:
        mfu = mfu_result.get("estimated_mfu")
        conditions = mfu_result.get("conditions", [])
        missing = [c for c in conditions if not c["available"]]
        all_met = len(missing) == 0

        L.append(f"{tr(lang, 'mfu')}\n")
        if all_met and mfu is not None:
            L.append(f"| {tr(lang, 'metric')} | {tr(lang, 'value')} |")
            L.append("|--------|-------|")
            L.append(f"| {tr(lang, 'mfu_metric')} | **{mfu * 100:.1f}%** |")
            L.append(f"| {tr(lang, 'mfu_level')} | {mfu_result.get('mfu_level', 'N/A')} |")
            method = mfu_result.get("method", "N/A")
            reliability = mfu_result.get("reliability", "")
            L.append(f"| {tr(lang, 'mfu_method')} | {method} ({reliability} {tr(lang, 'reliability')}) |")
            if mfu_result.get("achieved_tflops") is not None:
                L.append(f"| {tr(lang, 'achieved_tflops')} | {mfu_result['achieved_tflops']} |")
            if mfu_result.get("peak_tflops_used") is not None:
                L.append(f"| {tr(lang, 'peak_tflops')} | {mfu_result['peak_tflops_used']} ({hardware or 'unknown'}) |")
            if mfu_result.get("step_time_ms") is not None:
                L.append(f"| {tr(lang, 'step_time')} | {mfu_result['step_time_ms']:.1f} ms |")
            gap_info = mfu_result.get("gap_to_target") or {}
            if gap_info.get("gap_percent") is not None:
                L.append(f"| {tr(lang, 'gap_to_target')} | {gap_info['gap_percent']:+.1f}% |")
            L.append("")
        else:
            if mfu is not None:
                method = mfu_result.get("method", "")
                L.append(tr(lang, "mfu_rough", mfu=f"{mfu * 100:.1f}", method=method))
            L.append(f"{tr(lang, 'mfu_missing')}\n")
            L.append(f"| {tr(lang, 'condition')} | {tr(lang, 'status')} | {tr(lang, 'value')} | {tr(lang, 'how_to_provide')} |")
            L.append("|-----------|--------|-------|-----------------|")
            for c in conditions:
                status = tr(lang, "available") if c["available"] else tr(lang, "missing_status")
                value = str(c["value"]) if c.get("value") is not None else "-"
                L.append(f"| {c['field']} | {status} | {value} | {c['hint']} |")
            L.append("")
            L.append(tr(lang, "rerun"))
            L.append("```bash")
            L.append("python run_analysis.py <dir> --hardware ascend_910b2 --model-config model_config.json")
            L.append("```\n")
            L.append(tr(lang, "example_config"))
            L.append("```json")
            L.append('{')
            L.append('  "hidden_size": 4096,')
            L.append('  "num_layers": 32,')
            L.append('  "seq_len": 4096,')
            L.append('  "batch_size": 4')
            L.append('}')
            L.append("```\n")

    if top_ops:
        L.append(f"{tr(lang, 'top_ops')}\n")
        L.append(f"| # | {tr(lang, 'operator')} | {tr(lang, 'calls')} | {tr(lang, 'total')} | {tr(lang, 'share')} | {tr(lang, 'avg')} | {tr(lang, 'max')} |")
        L.append("|---|----------|-------|-------|-------|-----|-----|")
        for i, op in enumerate(top_ops[:15], 1):
            tv = f"{op['total_ms']/1000:.2f}s" if op['total_ms'] > 1000 else f"{op['total_ms']:.0f}ms"
            L.append(f"| {i} | `{op['name']}` | {op['calls']:,} | {tv} | {op['share_pct']}% | {op['avg_us']:.0f}us | {op['max_ms']:.0f}ms |")
        L.append("")

    if top_queue:
        L.append(f"{tr(lang, 'top_kernels')}\n")
        L.append(f"| # | {tr(lang, 'kernel')} | {tr(lang, 'calls')} | {tr(lang, 'total')} | {tr(lang, 'share')} |")
        L.append("|---|--------|-------|-------|-------|")
        for i, k in enumerate(top_queue[:10], 1):
            tv = f"{k['total_ms']/1000:.2f}s" if k['total_ms'] > 1000 else f"{k['total_ms']:.0f}ms"
            L.append(f"| {i} | `{k['name'][:50]}` | {k['calls']:,} | {tv} | {k['share_pct']}% |")
        L.append("")

    if bottlenecks:
        L.append(f"{tr(lang, 'bottlenecks')}\n")
        icons = {"CRITICAL": "X", "HIGH": "!", "MEDIUM": "?"}
        for bn in bottlenecks:
            ic = icons.get(bn['severity'], "-")
            title = _translate_bottleneck_title(bn, lang)
            L.append(f"### [{ic}] {bn['rank']}. {title}")
            L.append(f"- **{tr(lang, 'domain')}:** {bn['domain']} | **{tr(lang, 'confidence')}:** {_translate_confidence(bn['confidence'], lang)}")
            for ev in bn['evidence'][:3]:
                L.append(f"  - {ev}")
            L.append("")

    if suggestions:
        L.append(f"{tr(lang, 'suggestions')}\n")
        for i, sg in enumerate(suggestions, 1):
            sid = sg['id']
            s_title = SUGGESTION_TITLES.get(sid, {}).get(lang, sg['title'])
            s_desc = SUGGESTION_DESCS.get(sid, {}).get(lang, sg['description'])
            s_benefit = SUGGESTION_BENEFITS.get(sid, {}).get(lang, sg['expected_benefit'])
            L.append(f"### {i}. [{sid}] {s_title} ({sg['priority']})")
            L.append(f"**{s_desc}** {tr(lang, 'expected')} {s_benefit}")
            L.append(f"```python\n{sg['code_example']}\n```\n")

    # Host-bound deep analysis section
    if host_bound_result and verdict == "host_bound":
        rc = host_bound_result.get("root_cause_diagnosis", {})
        cat_summary = host_bound_result.get("cpu_category_breakdown", [])
        dd = host_bound_result.get("dispatch_density", {})

        L.append(f"{tr(lang, 'hb_root_cause')}\n")
        primary = rc.get('primary_cause', 'unknown')
        L.append(f"{tr(lang, 'primary_cause')} `{primary}` [{rc.get('primary_severity', '?')}]")
        rc_summary = rc.get(f"summary_{lang}", rc.get("summary", ""))
        L.append(f"\n{rc_summary}\n")

        if cat_summary:
            L.append(f"{tr(lang, 'cpu_time_by_cat')}\n")
            L.append(f"| {tr(lang, 'category')} | {tr(lang, 'time_col')} | {tr(lang, 'share_col')} | {tr(lang, 'calls_col')} |")
            L.append("|----------|------|-------|--------|")
            for cat in cat_summary[:10]:
                L.append(f"| {cat['category']} | {cat['total_ms']}ms | {cat['share_pct']}% | {cat['calls']:,} |")
            L.append("")

        hf = dd.get("high_freq_small_ops", {})
        if hf.get("count"):
            L.append(f"{tr(lang, 'dispatch_density')}\n")
            L.append(f"- **{tr(lang, 'cpu_ops_sec')}** {dd.get('cpu_ops_per_sec', 0):.0f}")
            L.append(f"- **{tr(lang, 'avg_dispatch')}** {dd.get('avg_dispatch_us', 0)}{tr(lang, 'per_op')}")
            L.append(f"- **{tr(lang, 'high_freq_ops')}** {hf['count']} {tr(lang, 'types_count')}, "
                     f"{hf.get('total_calls', 0):,} {tr(lang, 'calls_unit')} ({hf.get('calls_pct', 0)}%)\n")

        top_names = {o["name"] for o in hf.get("top_ops", [])}
        if top_names:
            L.append(f"{tr(lang, 'fused_recs')}\n")
            if any("silu" in n for n in top_names) and any("mul" in n for n in top_names):
                L.append("- **SwiGLU:** `silu + mul + matmul(gate) + matmul(up)` → `aclnnSwiGLU`")
            if any("norm" in n.lower() for n in top_names):
                L.append("- **Norm+Activation:** `LayerNorm + SiLU` → `aclnnLayerNormSiLU`")
            if any("empty_tensor" in n for n in top_names):
                L.append(f"- {tr(lang, 'mem_alloc_graph')}")
            if any("copy_" in n or "view" in n for n in top_names):
                L.append(f"- {tr(lang, 'data_move_graph')}")
            L.append("")
            L.append("```python")
            L.append('model = torch.compile(model, mode="max-autotune")')
            L.append("```\n")

        hb_suffix = f"_{lang}" if lang != "en" else ""
        L.append(f"> {tr(lang, 'full_analysis')} `out/host_bound_analysis{hb_suffix}.md`\n")

        # Stack trace rerun suggestion
        rc = host_bound_result.get("root_cause_diagnosis", {})
        has_dispatch = rc.get("primary_cause") == "operator_dispatch_overhead" or any(
            c.get("cause") == "operator_dispatch_overhead" for c in rc.get("all_causes", [])
        )
        if has_dispatch:
            if not with_stack_enabled:
                L.append(f"{tr(lang, 'stack_rerun_title')}\n")
                L.append(f"{tr(lang, 'stack_rerun_hint')}\n")
                L.append(f"{tr(lang, 'stack_rerun_code_title')}\n")
                L.append("```python")
                if lang == "zh":
                    L.append("from torch_npu.profiler import profile, ProfilerActivity")
                else:
                    L.append("from torch_npu.profiler import profile, ProfilerActivity")
                L.append("")
                L.append("with profile(")
                L.append("    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],")
                L.append("    with_stack=True,")
                L.append("    with_modules=True,")
                L.append(") as prof:")
                L.append("    train_one_step()")
                L.append("```\n")
                L.append(f"{tr(lang, 'stack_rerun_overhead_note')}\n")
            else:
                L.append(f"{tr(lang, 'stack_hotspot_available')}\n")

    return "\n".join(L) + "\n"


# ===========================================================================
# Cache management
# ===========================================================================

def _cache_key(db_path: Path) -> str:
    return hashlib.md5(f"{db_path}-{db_path.stat().st_size}".encode()).hexdigest()[:12]


def _cache_path(output_dir: Path) -> Path:
    return output_dir / "meta" / "_cache.json"


def load_cache(output_dir: Path, db_path: Path) -> Optional[dict]:
    cp = _cache_path(output_dir)
    if not cp.exists():
        return None
    try:
        cache = json.loads(cp.read_text(encoding="utf-8"))
        if cache.get("_key") == _cache_key(db_path):
            return cache
    except Exception:
        pass
    return None


def save_cache(output_dir: Path, db_path: Path, data: dict) -> None:
    cp = _cache_path(output_dir)
    data["_key"] = _cache_key(db_path)
    write_json(cp, data)


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Single-command performance analysis")
    parser.add_argument("profiler_dir", help="Path to profiler data directory")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--force", action="store_true", help="Force re-analysis, ignore cache")
    parser.add_argument("--model-config", help="Model config JSON (hidden_size, num_layers, seq_len, batch_size)")
    parser.add_argument("--hardware", help="Hardware model (e.g. ascend_910b2, ascend_910b1)")
    parser.add_argument("--num-devices", type=int, default=1, help="Number of devices for cluster MFU")
    args = parser.parse_args()

    root = Path(args.profiler_dir).resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory", file=sys.stderr)
        return 1

    t0 = time.time()
    fmt = detect_format(root)
    print(f"[1/6] Format: {fmt}")

    out_dir = root / "out"
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    hardware = args.hardware or infer_hardware(root)
    info_file = root / "profiler_info.json"
    profiler_info = json.loads(info_file.read_text(encoding="utf-8", errors="replace")) if info_file.exists() else {}

    # Check cache
    if fmt == "ascend_pt_db" and not args.force:
        db_path = _find_db(root)
        cached = load_cache(out_dir, db_path)
        if cached:
            print(f"[2/6] Using cached results (use --force to re-analyze)")
            data = cached
        else:
            print(f"[2/6] Analyzing DB (first run, will cache results)...")
            data = analyze_db(db_path, out_dir, args.top_n)
            save_cache(out_dir, db_path, data)
    elif fmt == "ascend_pt_db":
        db_path = _find_db(root)
        print(f"[2/6] Analyzing DB (--force)...")
        data = analyze_db(db_path, out_dir, args.top_n)
        save_cache(out_dir, db_path, data)
    else:
        print(f"[2/6] Format {fmt} not supported for automated analysis yet.")
        print(f"      Use the manual pipeline described in SKILL.md.")
        return 1

    print(f"[3/6] Classifying bottlenecks...")
    bottlenecks = classify_bottlenecks(data["util"], data["top_ops"], data["fwd_bwd"],
                                        data["syncs"], data["dtype_cast"], data["model"])

    print(f"[4/6] Generating suggestions...")
    with_stack_enabled = _check_with_stack(profiler_info)
    suggestions = build_suggestions(bottlenecks, data["model"], with_stack_enabled)

    print(f"[4.5/6] Computing MFU...")
    mfu_result = _compute_mfu(data, hardware, args.model_config, args.num_devices)

    print(f"[5/6] Writing artifacts...")
    dev = data["util"].get("device_util_pct", 0)
    kern = data["util"].get("kernel_util_pct", 0)
    verdict = "host_bound" if dev < 50 else ("compute_bound" if kern > 70 else "balanced")

    # Run deep host-bound analysis when verdict is host_bound
    host_bound_result = None
    if verdict == "host_bound":
        # Check if host-bound result is already cached
        host_bound_result = data.get("host_bound_result")
        if host_bound_result:
            print(f"[5.5/6] Using cached host-bound analysis")
        else:
            print(f"[5.5/6] Running deep host-bound root cause analysis...")
            try:
                from analyze_host_bound_root_cause import analyze_host_bound as run_host_bound
                db_path = _find_db(root)
                if db_path:
                    host_bound_result = run_host_bound(
                        db_path, out_dir, args.top_n,
                        cached_str_map=data.get("str_map"),
                        cached_cpu_rows=data.get("raw_cpu_op_rows"),
                        cached_wall_ns=data.get("wall_ns"),
                        cached_total_cpu_ns=data.get("total_cpu_ns"),
                        cached_total_queue_ns=data.get("total_queue_ns"),
                        with_stack_enabled=with_stack_enabled,
                    )
                    # Refine bottlenecks and suggestions with host-bound root cause
                    rc = host_bound_result.get("root_cause_diagnosis", {})
                    for cause in rc.get("all_causes", []):
                        bn_match = None
                        cause_zh = cause.get("summary_zh", cause.get("summary_en", ""))
                        cause_en = cause.get("summary_en", cause.get("summary_zh", ""))
                        cause_evidence = cause.get("evidence", cause.get("evidence_en", []))
                        if cause["cause"] == "operator_dispatch_overhead":
                            bn_match = {"rank": 0, "domain": "operator_dispatch_overhead",
                                "severity": "CRITICAL",
                                "title": cause_zh[:80],
                                "title_zh": cause_zh[:80],
                                "title_en": cause_en[:80],
                                "confidence": "strong",
                                "evidence": cause_evidence,
                                "suggestion_id": "HOST-03"}
                        elif cause["cause"] == "framework_overhead":
                            bn_match = {"rank": 0, "domain": "framework_overhead",
                                "severity": cause["severity"],
                                "title": cause_zh[:80],
                                "title_zh": cause_zh[:80],
                                "title_en": cause_en[:80],
                                "confidence": "strong",
                                "evidence": cause_evidence,
                                "suggestion_id": "HOST-04"}
                        elif cause["cause"] == "slow_io_data":
                            bn_match = {"rank": 0, "domain": "slow_io_data",
                                "severity": cause["severity"],
                                "title": cause_zh[:80],
                                "title_zh": cause_zh[:80],
                                "title_en": cause_en[:80],
                                "confidence": "strong",
                                "evidence": cause_evidence,
                                "suggestion_id": "HOST-05"}
                        elif cause["cause"] == "dtype_cast":
                            bn_match = {"rank": 0, "domain": "dtype_cast_overhead",
                                "severity": cause["severity"],
                                "title": cause_zh[:80],
                                "title_zh": cause_zh[:80],
                                "title_en": cause_en[:80],
                                "confidence": "moderate",
                                "evidence": cause_evidence,
                                "suggestion_id": "HOST-06"}
                        if bn_match:
                            bottlenecks.append(bn_match)
                    # Re-sort by severity
                    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
                    bottlenecks.sort(key=lambda b: sev_order.get(b.get("severity", "LOW"), 4))
                    for i, b in enumerate(bottlenecks):
                        b["rank"] = i + 1
                    # Rebuild suggestions with new bottleneck info
                    suggestions = build_suggestions(bottlenecks, data["model"], with_stack_enabled)
                    print(f"  Root cause: {rc.get('primary_cause', 'unknown')}")
                    # Cache host-bound result for repeat runs
                    data["host_bound_result"] = host_bound_result
                    save_cache(out_dir, db_path, data)
            except Exception as e:
                print(f"  Host-bound analysis skipped: {e}")

    write_json(meta_dir / "performance-profile.json", {
        "schema_version": "performance-agent/0.1",
        "trace_root": str(root), "format": fmt,
        "stack": infer_stack_from_root(root),
        "torch_npu_version": profiler_info.get("torch_npu_version"),
        "cann_version": profiler_info.get("cann_version"),
        "workload_type": "training" if any("Backward" in op["name"] for op in data["top_ops"][:20]) else "inference",
        "primary_symptom": "host launch overhead" if verdict == "host_bound" else "throughput",
        "confidence": "strong",
        "hardware": {"detected_model": hardware, "peak_fp16_tflops": get_peak_tflops(hardware, "fp16")},
        "model_architecture": data["model"],
        "device_utilization": data["util"],
        "forward_backward_ratio": data["fwd_bwd"],
    })
    write_json(meta_dir / "bottlenecks.json", {"bottlenecks": bottlenecks})
    write_json(meta_dir / "optimization-suggestions.json", {"suggestions": suggestions})
    write_json(meta_dir / "performance-verdict.json", {
        "verdict": verdict,
        "dominant_bottleneck": bottlenecks[0]["domain"] if bottlenecks else "unknown",
        "severity": bottlenecks[0]["severity"] if bottlenecks else "UNKNOWN",
        "device_utilization_pct": dev,
        "primary_recommendation": suggestions[0]["title"] if suggestions else "N/A",
    })
    write_json(meta_dir / "mfu.json", mfu_result)

    print(f"[6/6] Generating reports (EN + ZH)...")
    for lang, suffix in [("en", "_en"), ("zh", "_zh")]:
        report = render_report(data, bottlenecks, suggestions, mfu_result, root, hardware,
                               time.time() - t0, host_bound_result, lang=lang,
                               with_stack_enabled=with_stack_enabled)
        write_text(out_dir / f"report{suffix}.md", report)

    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"Done in {elapsed:.1f}s")
    print(f"Reports: {out_dir / 'report_en.md'}, {out_dir / 'report_zh.md'}")
    print(f"Verdict: {verdict.upper()}")
    if bottlenecks:
        print(f"Top bottleneck: {bottlenecks[0]['title']}")
    if suggestions:
        print(f"Top suggestion: {suggestions[0]['title']}")
    print(f"{'='*55}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
