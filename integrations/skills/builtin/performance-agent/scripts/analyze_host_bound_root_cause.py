#!/usr/bin/env python3
"""Deep analysis of HOST_BOUND root cause from profiler SQLite DB.

Efficient approach: uses SQL aggregation to avoid scanning all 17M rows.
"""
import bisect
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

from report_i18n import (
    tr, ROOT_CAUSE_TITLES, OPT_PRIORITY,
    FUSION_TITLES, FUSION_DESCS,
)

CATEGORY_RULES = [
    ("checkpoint", lambda n: "CheckpointFunction" in n),
    ("autograd_engine", lambda n: n.startswith("autograd::engine::evaluate_function")),
    ("autograd_other", lambda n: n.startswith("autograd::")),
    ("backward", lambda n: "Backward" in n and not n.startswith("autograd::")),
    ("dataloader", lambda n: any(k in n for k in ("DataLoader", "Iterator", "Fetch", "prefetch", "collate"))),
    ("io_save", lambda n: any(k in n for k in ("save", "Save", "state_dict", "safetensors"))),
    ("io_log", lambda n: any(k in n for k in ("add_scalar", "histogram", "Scalar", "flush", "print", "Print"))),
    ("dtype_cast", lambda n: any(k in n for k in ("_npu_dtype_cast", "npu_dtype_cast", "_to_copy", "aten::to"))),
    ("optimizer", lambda n: any(k in n for k in ("Optimizer", "optimizer", "_foreach", "clip_grad", "grad_norm"))),
    ("loss", lambda n: any(k in n for k in ("cross_entropy", "nll_loss", "mse_loss", "binary_cross_entropy"))),
    ("attention", lambda n: any(k in n for k in ("scaled_dot_product_attention", "FlashAttention", "sdpa"))),
    ("matmul_linear", lambda n: any(k in n for k in ("aten::linear", "aten::matmul", "aten::mm", "aten::bmm", "aten::addmm"))),
    ("embedding", lambda n: "aten::embedding" in n or "Embedding" in n),
    ("norm", lambda n: any(k in n for k in ("layernorm", "LayerNorm", "rmsnorm", "RMSNorm", "layer_norm", "rms_norm"))),
    ("activation", lambda n: any(k in n for k in ("aten::silu", "aten::gelu", "aten::relu", "aten::sigmoid", "aten::tanh"))),
    ("copy_tensor", lambda n: any(k in n for k in ("aten::copy_", "aten::clone", "aten::contiguous", "aten::expand", "aten::reshape", "aten::view", "aten::permute", "aten::transpose"))),
    ("empty_tensor", lambda n: n == "empty_tensor"),
    ("aten_dispatch", lambda n: n.startswith("aten::")),
    ("framework_overhead", lambda n: True),
]


def classify_op(name: str) -> str:
    for cat, matcher in CATEGORY_RULES:
        if matcher(name):
            return cat
    return "framework_overhead"


def analyze_host_bound(db_path: Path, output_dir: Path, top_n: int = 30,
                       cached_str_map: dict = None,
                       cached_cpu_rows: list = None,
                       cached_wall_ns: int = None,
                       cached_total_cpu_ns: int = None,
                       cached_total_queue_ns: int = None,
                       with_stack_enabled: bool = False):
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    db = sqlite3.connect(str(db_path))
    cur = db.cursor()

    # === Phase 0: Load str_map from cache or DB ===
    if cached_str_map:
        str_map = {int(k): v for k, v in cached_str_map.items()}
    else:
        str_map = dict(cur.execute("SELECT id, value FROM STRING_IDS").fetchall())

    # === Phase 1: Wall time & coverage ===
    if cached_wall_ns is not None and cached_total_cpu_ns is not None and cached_total_queue_ns is not None:
        print("[1/5] Using cached coverage data...")
        wall_ns = cached_wall_ns
        total_cpu_ns = cached_total_cpu_ns
        total_queue_ns = cached_total_queue_ns
        device_pct = round(total_queue_ns / wall_ns * 100, 1)
        print(f"  Wall={wall_ns/1e9:.1f}s CPU={total_cpu_ns/1e9:.1f}s Device={total_queue_ns/1e9:.1f}s ({device_pct}%)")
    else:
        print("[1/5] Computing coverage...")
        wall_start, wall_end = cur.execute(
            "SELECT MIN(startNs), MAX(endNs) FROM PYTORCH_API"
        ).fetchone()
        wall_ns = int(wall_end) - int(wall_start)

        total_cpu_ns = int(cur.execute(
            "SELECT SUM(CAST(endNs AS REAL)-CAST(startNs AS REAL)) FROM PYTORCH_API WHERE type=50001"
        ).fetchone()[0])
        total_queue_ns = int(cur.execute(
            "SELECT SUM(CAST(endNs AS REAL)-CAST(startNs AS REAL)) FROM PYTORCH_API WHERE type=50002"
        ).fetchone()[0])

        device_pct = round(total_queue_ns / wall_ns * 100, 1)
        print(f"  Wall={wall_ns/1e9:.1f}s CPU={total_cpu_ns/1e9:.1f}s Device={total_queue_ns/1e9:.1f}s ({device_pct}%)")

    # === Phase 2: CPU ops grouped by name (use cache if available) ===
    if cached_cpu_rows and cached_str_map:
        print("[2/5] Using cached CPU op hotspots (skip DB scan)...")
        cpu_by_name = {}
        for r in cached_cpu_rows:
            nm = cached_str_map.get(str(r["name_id"]), f"ID:{r['name_id']}")
            cpu_by_name[nm] = {
                "name": nm, "calls": r["calls"],
                "total_ns": r["total_ns"],
                "total_ms": round(r["total_ns"] / 1e6, 1),
                "avg_us": round(r["avg_ns"] / 1e3, 1),
                "max_ms": round(r["max_ns"] / 1e6, 2),
            }
        print(f"  Done from cache, {len(cpu_by_name)} unique ops")
    else:
        print("[2/5] Scanning CPU op hotspots (this takes ~90s)...")
        t1 = time.time()
        cur.execute("""
            SELECT name, COUNT(*),
                   SUM(CAST(endNs AS REAL)-CAST(startNs AS REAL)),
                   AVG(CAST(endNs AS REAL)-CAST(startNs AS REAL)),
                   MAX(CAST(endNs AS REAL)-CAST(startNs AS REAL))
            FROM PYTORCH_API
            WHERE type=50001 AND CAST(endNs AS REAL) > CAST(startNs AS REAL)
            GROUP BY name
            ORDER BY SUM(CAST(endNs AS REAL)-CAST(startNs AS REAL)) DESC
        """)
        cpu_by_name = {}
        for row in cur.fetchall():
            nm = str_map.get(row[0], f"ID:{row[0]}")
            cpu_by_name[nm] = {
                "name": nm, "calls": row[1],
                "total_ns": int(row[2]),
                "total_ms": round(row[2]/1e6, 1),
                "avg_us": round(row[3]/1e3, 1),
                "max_ms": round(row[4]/1e6, 2),
            }
        print(f"  Done in {time.time()-t1:.0f}s, {len(cpu_by_name)} unique ops")

    total_cpu_ns_check = sum(v["total_ns"] for v in cpu_by_name.values())
    total_calls = sum(v["calls"] for v in cpu_by_name.values())

    # === Phase 3: Category breakdown ===
    print("[3/5] Categorizing ops...")
    cat_ns = defaultdict(int)
    cat_calls = defaultdict(int)
    cat_ops = defaultdict(list)
    for info in cpu_by_name.values():
        cat = classify_op(info["name"])
        cat_ns[cat] += info["total_ns"]
        cat_calls[cat] += info["calls"]
        cat_ops[cat].append(info)

    cat_summary = []
    for cat in sorted(cat_ns, key=cat_ns.get, reverse=True):
        ops = sorted(cat_ops[cat], key=lambda x: x["total_ns"], reverse=True)
        cat_summary.append({
            "category": cat,
            "total_ms": round(cat_ns[cat]/1e6, 1),
            "share_pct": round(cat_ns[cat]/total_cpu_ns_check*100, 2),
            "calls": cat_calls[cat],
            "top_ops": [{"name": o["name"], "total_ms": o["total_ms"],
                          "calls": o["calls"], "avg_us": o["avg_us"]} for o in ops[:5]],
        })

    # === Phase 4: Dispatch density ===
    print("[4/5] Analyzing dispatch density...")
    high_freq_ops = [o for o in cpu_by_name.values() if o["calls"] > 1000 and o["avg_us"] < 50]
    high_freq_calls = sum(o["calls"] for o in high_freq_ops)
    high_freq_ms = sum(o["total_ms"] for o in high_freq_ops)

    dispatch_density = {
        "total_cpu_calls": total_calls,
        "cpu_ops_per_sec": round(total_calls / (wall_ns/1e9), 0),
        "avg_dispatch_us": round(total_cpu_ns / total_calls / 1e3, 1),
        "high_freq_small_ops": {
            "count": len(high_freq_ops),
            "total_calls": high_freq_calls,
            "total_ms": round(high_freq_ms, 1),
            "calls_pct": round(high_freq_calls / total_calls * 100, 1),
            "top_ops": sorted(
                [{"name": o["name"], "calls": o["calls"], "total_ms": o["total_ms"], "avg_us": o["avg_us"]}
                 for o in high_freq_ops],
                key=lambda x: x["calls"], reverse=True
            )[:20],
        },
    }

    # === Phase 5: Gap analysis using sampling ===
    print("[5/5] Sampling device-idle gaps...")
    # Use a smaller window: pick one step (20 steps total, pick middle)
    # Find step boundaries
    ckpt_name_id = None
    for sid, sv in str_map.items():
        if sv == "CheckpointFunction":
            ckpt_name_id = sid
            break

    gap_analysis = {}
    if ckpt_name_id is not None:
        # Get a batch of CheckpointFunction start times to define step boundaries
        cur.execute(f"""
            SELECT startNs FROM PYTORCH_API
            WHERE name = {ckpt_name_id}
            ORDER BY startNs LIMIT 5 OFFSET 450
        """)
        step_markers = [int(r[0]) for r in cur.fetchall()]
        if len(step_markers) >= 2:
            # One forward pass = step_markers[0] to step_markers[-1]
            # Use the first layer's forward as proxy for one full step region
            win_start = step_markers[0]
            # Approximate step size: 896 forward ckpt calls per step (56 layers * 16 accum)
            # A step = ~896 ckpt intervals. Get the 896th ckpt start after win_start
            cur.execute(f"""
                SELECT startNs FROM PYTORCH_API
                WHERE name = {ckpt_name_id} AND CAST(startNs AS REAL) >= {win_start}
                ORDER BY startNs LIMIT 1 OFFSET 895
            """)
            step_end_row = cur.fetchone()
            if step_end_row:
                win_end = int(step_end_row[0])
                win_ns = win_end - win_start
                print(f"  Step window: {win_ns/1e6:.0f}ms")

                # Get ALL Dequeue ops in this window
                cur.execute(f"""
                    SELECT CAST(startNs AS REAL), CAST(endNs AS REAL)
                    FROM PYTORCH_API
                    WHERE type=50002
                      AND CAST(startNs AS REAL) >= {win_start}
                      AND CAST(startNs AS REAL) < {win_end}
                    ORDER BY CAST(startNs AS REAL)
                """)
                queue_intervals = cur.fetchall()
                queue_cov_ns = sum(e - s for s, e in queue_intervals)
                print(f"  {len(queue_intervals)} queue ops, coverage {queue_cov_ns/win_ns*100:.1f}%")

                # Merge intervals
                merged = []
                for s, e in queue_intervals:
                    if merged and s <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                    else:
                        merged.append((s, e))

                # Find gaps > 0.1ms
                gaps = []
                for i in range(1, len(merged)):
                    gs = merged[i-1][1]
                    ge = merged[i][0]
                    gns = ge - gs
                    if gns > 100000:
                        gaps.append((gs, ge, gns))

                total_gap_ns = sum(g[2] for g in gaps)
                print(f"  {len(gaps)} gaps, total {total_gap_ns/1e6:.0f}ms ({total_gap_ns/win_ns*100:.1f}%)")

                # Sample TOP 50 largest gaps for CPU op analysis
                sample_gaps = sorted(gaps, key=lambda x: x[2], reverse=True)[:50]
                gap_cpu_ops = defaultdict(lambda: {"calls": 0, "total_ns": 0, "max_ns": 0})

                # Batch: fetch ALL CPU ops in step window in one query, assign to gaps via bisect
                sorted_gaps = sorted(sample_gaps, key=lambda x: x[0])
                gap_starts = [g[0] for g in sorted_gaps]
                gap_ends = [g[1] for g in sorted_gaps]

                print(f"  Fetching CPU ops in step window (single query)...")
                t_gap = time.time()
                cur.execute(f"""
                    SELECT name, CAST(startNs AS REAL), CAST(endNs AS REAL)
                    FROM PYTORCH_API
                    WHERE type=50001
                      AND CAST(startNs AS REAL) >= {win_start}
                      AND CAST(startNs AS REAL) < {win_end}
                      AND CAST(endNs AS REAL) > CAST(startNs AS REAL)
                """)
                cpu_rows = cur.fetchall()
                print(f"  {len(cpu_rows)} CPU ops in step window (fetched in {time.time()-t_gap:.1f}s)")

                for name_id, op_start, op_end in cpu_rows:
                    idx = bisect.bisect_right(gap_starts, op_start) - 1
                    if idx >= 0 and op_start >= gap_starts[idx] and op_start < gap_ends[idx]:
                        op_ns = int(op_end - op_start)
                        nm = str_map.get(name_id, f"ID:{name_id}")
                        gap_cpu_ops[nm]["calls"] += 1
                        gap_cpu_ops[nm]["total_ns"] += op_ns
                        gap_cpu_ops[nm]["max_ns"] = max(gap_cpu_ops[nm]["max_ns"], op_ns)

                gap_by_cat = defaultdict(lambda: {"calls": 0, "total_ns": 0, "ops": []})
                for op_name, data in gap_cpu_ops.items():
                    cat = classify_op(op_name)
                    gap_by_cat[cat]["calls"] += data["calls"]
                    gap_by_cat[cat]["total_ns"] += data["total_ns"]
                    gap_by_cat[cat]["ops"].append({
                        "name": op_name,
                        "calls": data["calls"],
                        "total_ms": round(data["total_ns"]/1e6, 1),
                        "max_ms": round(data["max_ns"]/1e6, 2),
                    })

                for d in gap_by_cat.values():
                    d["ops"].sort(key=lambda x: x["total_ms"], reverse=True)
                    d["total_ms"] = round(d["total_ns"]/1e6, 1)

                gap_analysis = {
                    "window_ms": round(win_ns/1e6, 1),
                    "device_coverage_pct": round(queue_cov_ns/win_ns*100, 1),
                    "num_gaps": len(gaps),
                    "total_gap_ms": round(total_gap_ns/1e6, 1),
                    "gap_pct": round(total_gap_ns/win_ns*100, 1),
                    "sampled_gaps": len(sample_gaps),
                    "gap_cpu_by_category": {
                        cat: {"total_ms": d["total_ms"], "calls": d["calls"],
                              "top_ops": d["ops"][:8]}
                        for cat, d in sorted(gap_by_cat.items(), key=lambda x: x[1]["total_ns"], reverse=True)
                    },
                }

    # === Slow individual CPU ops ===
    print("  Finding slow individual ops...")
    cur.execute("""
        SELECT name, CAST(endNs AS REAL)-CAST(startNs AS REAL) as dur
        FROM PYTORCH_API
        WHERE type=50001 AND CAST(endNs AS REAL) > CAST(startNs AS REAL)
        ORDER BY dur DESC LIMIT 50
    """)
    slow_ops = []
    for row in cur.fetchall():
        nm = str_map.get(row[0], f"ID:{row[0]}")
        slow_ops.append({
            "name": nm,
            "category": classify_op(nm),
            "duration_ms": round(row[1]/1e6, 2),
        })

    db.close()

    # === Root cause determination ===
    root_cause = determine_root_cause(cat_summary, gap_analysis, dispatch_density,
                                       slow_ops, total_cpu_ns, total_queue_ns, wall_ns,
                                       with_stack_enabled=with_stack_enabled)

    result = {
        "wall_time_s": round(wall_ns/1e9, 1),
        "cpu_op_time_s": round(total_cpu_ns/1e9, 1),
        "device_op_time_s": round(total_queue_ns/1e9, 1),
        "device_coverage_pct": device_pct,
        "idle_pct": round((wall_ns-total_queue_ns)/wall_ns*100, 1),
        "cpu_category_breakdown": cat_summary,
        "dispatch_density": dispatch_density,
        "gap_analysis": gap_analysis,
        "slow_individual_ops": slow_ops[:30],
        "root_cause_diagnosis": root_cause,
    }

    # Write both language versions
    for lang, suffix in [("en", "_en"), ("zh", "_zh")]:
        report = render_report(result, lang=lang)
        (output_dir / f"host_bound_analysis{suffix}.md").write_text(report, encoding="utf-8")
    json.dump(result, (meta_dir / "host_bound_analysis.json").open("w", encoding="utf-8"), indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.0f}s")
    print(f"Reports: {output_dir / 'host_bound_analysis_en.md'}, {output_dir / 'host_bound_analysis_zh.md'}")
    print(f"Primary cause: {root_cause['primary_cause']} [{root_cause['primary_severity']}]")
    print(f"  {root_cause.get('summary_zh', root_cause.get('summary_en', ''))}")
    print(f"{'='*60}")
    return result




def determine_root_cause(cat_summary, gap_analysis, dispatch_density, slow_ops,
                          total_cpu_ns, total_queue_ns, wall_ns,
                          with_stack_enabled: bool = False):
    causes = []
    dd = dispatch_density
    hf = dd.get("high_freq_small_ops", {})

    # 1. Operator dispatch overhead
    if hf.get("calls_pct", 0) > 25 and dd.get("avg_dispatch_us", 0) < 40:
        causes.append({
            "cause": "operator_dispatch_overhead",
            "severity": "CRITICAL",
            "summary_en": (f"Excessive operator dispatch overhead: {hf['calls_pct']}% of CPU calls "
                           f"are high-frequency small operators (avg {dd['avg_dispatch_us']}us/op, "
                           f"{dd['cpu_ops_per_sec']:.0f} ops/sec). "
                           f"{hf['count']} small op types, {hf['total_calls']:,} calls, "
                           f"total {hf['total_ms']:.0f}ms"),
            "summary_zh": (f"算子下发开销过大: {hf['calls_pct']}% 的CPU调用是高频小算子 "
                           f"(avg {dd['avg_dispatch_us']}us/op, "
                           f"{dd['cpu_ops_per_sec']:.0f} ops/sec). "
                           f"共{hf['count']}种小算子, {hf['total_calls']:,}次调用, "
                           f"总耗时{hf['total_ms']:.0f}ms"),
            "evidence_en": [
                f"High-freq small ops: {hf['count']} types, {hf['total_calls']:,} calls, {hf['calls_pct']}% of all calls",
                f"CPU dispatch rate: {dd['cpu_ops_per_sec']:.0f} ops/sec",
                f"Avg dispatch: {dd['avg_dispatch_us']}us",
            ],
            "evidence_zh": [
                f"高频小算子: {hf['count']}种, {hf['total_calls']:,}次, {hf['calls_pct']}% of all calls",
                f"CPU下发速率: {dd['cpu_ops_per_sec']:.0f} ops/sec",
                f"平均单次下发: {dd['avg_dispatch_us']}us",
            ],
            "optimization_priority": "build_fused_ops",
        })
    # Add stack note to dispatch overhead cause
    if not with_stack_enabled:
        for cause in causes:
            if cause["cause"] == "operator_dispatch_overhead":
                cause["stack_note_en"] = "Re-run with `with_stack=True` to identify which Python code generates these ops."
                cause["stack_note_zh"] = "请以 `with_stack=True` 重新采集，以定位产生这些算子的 Python 代码。"
                cause["needs_stack_rerun"] = True
                break

    # 2. Framework overhead (autograd, checkpoint)
    fw_cats = {c["category"]: c for c in cat_summary}
    ckpt_ms = fw_cats.get("checkpoint", {}).get("total_ms", 0)
    ag_eng_ms = fw_cats.get("autograd_engine", {}).get("total_ms", 0)
    ag_other_ms = fw_cats.get("autograd_other", {}).get("total_ms", 0)
    bwd_ms = fw_cats.get("backward", {}).get("total_ms", 0)
    framework_total = ckpt_ms + ag_eng_ms + ag_other_ms + bwd_ms
    framework_pct = framework_total / (total_cpu_ns / 1e6) * 100

    if framework_pct > 20:
        causes.append({
            "cause": "framework_overhead",
            "severity": "HIGH" if framework_pct > 35 else "MEDIUM",
            "summary_en": (f"Framework/Autograd overhead accounts for {framework_pct:.1f}% of CPU: "
                           f"Checkpoint={ckpt_ms:.0f}ms, AutogradEngine={ag_eng_ms:.0f}ms, "
                           f"Backward={bwd_ms:.0f}ms"),
            "summary_zh": (f"框架/Autograd开销占CPU {framework_pct:.1f}%: "
                           f"Checkpoint={ckpt_ms:.0f}ms, AutogradEngine={ag_eng_ms:.0f}ms, "
                           f"Backward={bwd_ms:.0f}ms"),
            "evidence_en": [
                f"CheckpointFunction: {ckpt_ms:.0f}ms",
                f"autograd::engine: {ag_eng_ms:.0f}ms",
                f"Backward: {bwd_ms:.0f}ms",
            ],
            "evidence_zh": [
                f"CheckpointFunction: {ckpt_ms:.0f}ms",
                f"autograd::engine: {ag_eng_ms:.0f}ms",
                f"Backward: {bwd_ms:.0f}ms",
            ],
            "optimization_priority": "reduce_framework_overhead",
        })

    # 3. Gap analysis - what runs during device idle
    gap_cat = gap_analysis.get("gap_cpu_by_category", {})
    if gap_cat:
        for cat, data in gap_cat.items():
            if cat in ("checkpoint", "autograd_engine", "framework_overhead"):
                pct_of_gaps = round(data["total_ms"] / gap_analysis["total_gap_ms"] * 100, 1) if gap_analysis["total_gap_ms"] else 0
                if pct_of_gaps > 10:
                    causes.append({
                        "cause": f"gap_{cat}",
                        "severity": "HIGH" if pct_of_gaps > 30 else "MEDIUM",
                        "summary_en": (f"During device idle, {cat} accounts for {pct_of_gaps}%: "
                                       f"{data['total_ms']:.0f}ms, {data['calls']:,} calls"),
                        "summary_zh": (f"Device 空闲期间 {cat} 占 {pct_of_gaps}%: "
                                       f"{data['total_ms']:.0f}ms, {data['calls']:,}次调用"),
                        "evidence_en": [
                            f"{cat} during gaps: {data['total_ms']:.0f}ms ({pct_of_gaps}% of idle)",
                            f"Top: {', '.join(o['name'][:40] for o in data.get('top_ops',[])[:3])}",
                        ],
                        "evidence_zh": [
                            f"{cat} during gaps: {data['total_ms']:.0f}ms ({pct_of_gaps}% of idle)",
                            f"Top: {', '.join(o['name'][:40] for o in data.get('top_ops',[])[:3])}",
                        ],
                        "optimization_priority": "optimize_specific_ops",
                    })

    # 4. Slow IO/data ops
    io_slow = [o for o in slow_ops if o["category"] in ("io_save", "io_log", "dataloader") and o["duration_ms"] > 50]
    if io_slow:
        causes.append({
            "cause": "slow_io_data",
            "severity": "HIGH" if any(o["duration_ms"] > 500 for o in io_slow) else "MEDIUM",
            "summary_en": f"IO/data operations are slow: {len(io_slow)} ops >50ms, max {max(o['duration_ms'] for o in io_slow):.0f}ms",
            "summary_zh": f"IO/数据操作过慢: {len(io_slow)}个>50ms, 最大{max(o['duration_ms'] for o in io_slow):.0f}ms",
            "evidence_en": [f"{o['name'][:50]}: {o['duration_ms']}ms" for o in io_slow[:5]],
            "evidence_zh": [f"{o['name'][:50]}: {o['duration_ms']}ms" for o in io_slow[:5]],
            "optimization_priority": "optimize_io_data",
        })

    # 5. dtype cast
    cast_ms = fw_cats.get("dtype_cast", {}).get("total_ms", 0)
    if cast_ms > (total_cpu_ns/1e6) * 0.03:
        causes.append({
            "cause": "dtype_cast",
            "severity": "MEDIUM",
            "summary_en": f"Dtype cast overhead: {cast_ms:.0f}ms ({cast_ms/(total_cpu_ns/1e6)*100:.1f}% of CPU)",
            "summary_zh": f"Dtype cast开销: {cast_ms:.0f}ms ({cast_ms/(total_cpu_ns/1e6)*100:.1f}% of CPU)",
            "evidence_en": [f"dtype_cast total: {cast_ms:.0f}ms"],
            "evidence_zh": [f"dtype_cast total: {cast_ms:.0f}ms"],
            "optimization_priority": "fix_dtype",
        })

    if not causes:
        causes.append({
            "cause": "general_host_overhead",
            "severity": "MEDIUM",
            "summary_en": "CPU overhead is scattered, no single dominant bottleneck",
            "summary_zh": "CPU开销分散，无单一明显瓶颈",
            "evidence_en": [],
            "evidence_zh": [],
            "optimization_priority": "enable_graph_mode",
        })

    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
    causes.sort(key=lambda x: sev_order.get(x["severity"], 3))

    result = {
        "primary_cause": causes[0]["cause"],
        "primary_severity": causes[0]["severity"],
        "summary_en": causes[0].get("summary_en", causes[0].get("summary", "")),
        "summary_zh": causes[0].get("summary_zh", causes[0].get("summary", "")),
        "summary": causes[0].get("summary_zh", causes[0].get("summary", "")),
        "all_causes": causes,
    }
    return result


def render_report(data, lang="en"):
    L = [tr(lang, "hb_analysis_title")]
    L.append(f"{tr(lang, 'hb_wall')} {data['wall_time_s']}s | "
             f"{tr(lang, 'hb_cpu_ops')} {data['cpu_op_time_s']}s | "
             f"{tr(lang, 'hb_device')} {data['device_op_time_s']}s ({data['device_coverage_pct']}%)\n")

    rc = data["root_cause_diagnosis"]
    L.append(f"{tr(lang, 'root_cause')}\n")
    L.append(f"**{rc.get(f'title_{lang}', rc.get('primary_cause', 'unknown'))}** [{rc['primary_severity']}]")
    rc_summary = rc.get(f"summary_{lang}", rc.get("summary", ""))
    L.append(f"\n{rc_summary}\n")
    for i, c in enumerate(rc["all_causes"], 1):
        cause_title = ROOT_CAUSE_TITLES.get(c["cause"], {}).get(lang, c["cause"])
        L.append(f"### {i}. {cause_title} [{c['severity']}]")
        cause_summary = c.get(f"summary_{lang}", c.get("summary", ""))
        L.append(cause_summary)
        for ev in c.get(f"evidence_{lang}", c.get("evidence", [])):
            L.append(f"- {ev}")
        opt = OPT_PRIORITY.get(c.get("optimization_priority", ""), {})
        opt_text = opt.get(lang, c.get("optimization_priority", ""))
        L.append(f"- **{tr(lang, 'opt_priority')}** `{opt_text}`")
        if c.get("needs_stack_rerun"):
            stack_note = c.get(f"stack_note_{lang}", "")
            if stack_note:
                L.append(f"- {stack_note}")
        L.append("")

    L.append(f"{tr(lang, 'cpu_time_category')}\n")
    L.append(f"| {tr(lang, 'category')} | {tr(lang, 'time_col')} | % | {tr(lang, 'calls_col')} | {tr(lang, 'top_ops_col')} |")
    L.append("|----------|------|---|-------|---------|")
    for cat in data["cpu_category_breakdown"][:15]:
        top = ", ".join(f"{o['name'][:25]}({o['total_ms']}ms)" for o in cat["top_ops"][:2])
        L.append(f"| {cat['category']} | {cat['total_ms']}ms | {cat['share_pct']}% | {cat['calls']:,} | {top} |")
    L.append("")

    dd = data["dispatch_density"]
    L.append(f"{tr(lang, 'dispatch_density_section')}\n")
    L.append(f"- **{tr(lang, 'cpu_ops_sec')}** {dd['cpu_ops_per_sec']:.0f}")
    L.append(f"- **{tr(lang, 'avg_dispatch')}** {dd['avg_dispatch_us']}{tr(lang, 'per_op')}")
    hf = dd.get("high_freq_small_ops", {})
    L.append(f"- **{tr(lang, 'high_freq_small')}** {hf.get('count',0)} {tr(lang, 'types_count')}, "
             f"{hf.get('total_calls',0):,} {tr(lang, 'calls_unit')} ({hf.get('calls_pct',0)}%), "
             f"{hf.get('total_ms',0):.0f}ms\n")
    if hf.get("top_ops"):
        L.append(f"| # | Op | {tr(lang, 'calls')} | {tr(lang, 'time_col')} | {tr(lang, 'avg')} |")
        L.append("|---|----|-------|------|-----|")
        for i, o in enumerate(hf["top_ops"], 1):
            L.append(f"| {i} | `{o['name'][:45]}` | {o['calls']:,} | {o['total_ms']}ms | {o['avg_us']}us |")
        L.append("")

    ga = data.get("gap_analysis", {})
    if ga.get("gap_cpu_by_category"):
        L.append(f"{tr(lang, 'gap_analysis')}\n")
        L.append(f"- {tr(lang, 'gap_window')} {ga['window_ms']:.0f}ms, {tr(lang, 'gap_coverage')} {ga['device_coverage_pct']}%")
        L.append(f"- {ga['num_gaps']} {tr(lang, 'gap_total')} {ga['total_gap_ms']:.0f}ms ({ga['gap_pct']}%)\n")
        L.append(f"| {tr(lang, 'category')} | {tr(lang, 'time_in_gaps')} | {tr(lang, 'calls_col')} | {tr(lang, 'top_ops_col')} |")
        L.append("|----------|-------------|-------|---------|")
        for cat, d in ga["gap_cpu_by_category"].items():
            top = ", ".join(f"{o['name'][:30]}({o['total_ms']}ms)" for o in d.get("top_ops",[])[:2])
            L.append(f"| {cat} | {d['total_ms']}ms | {d['calls']:,} | {top} |")
        L.append("")

    if data.get("slow_individual_ops"):
        L.append(f"{tr(lang, 'slow_ops')}\n")
        L.append(f"| # | Op | {tr(lang, 'category')} | {tr(lang, 'duration')} |")
        L.append("|---|----|----------|----------|")
        for i, o in enumerate(data["slow_individual_ops"][:20], 1):
            L.append(f"| {i} | `{o['name'][:55]}` | {o['category']} | {o['duration_ms']}ms |")
        L.append("")

    L.append(_fusion_recs(data, lang))

    return "\n".join(L) + "\n"


def _fusion_recs(data, lang="en"):
    L = [f"{tr(lang, 'fusion_recs')}\n"]
    hf = data.get("dispatch_density", {}).get("high_freq_small_ops", {})
    top_ops_names = {o["name"] for o in hf.get("top_ops", [])}
    cat_map = {c["category"]: c for c in data.get("cpu_category_breakdown", [])}

    patterns = []

    if any("aten::mul" in n for n in top_ops_names) and any("aten::silu" in n for n in top_ops_names):
        patterns.append(("swiglu", "aten::silu * gate_matmul + up_matmul → aclnnSwiGLU/MatmulSwiGLU"))

    if any("norm" in n.lower() for n in top_ops_names) and any("silu" in n.lower() or "gelu" in n.lower() for n in top_ops_names):
        patterns.append(("norm_activation", "LayerNorm/RMSNorm + SiLU/GELU → aclnnLayerNormSiLU"))

    if cat_map.get("optimizer"):
        patterns.append(("optimizer", "AdamW per-param: mul+add+mul+add → NpuFusedAdamW"))

    cast_ops = [o for o in hf.get("top_ops", []) if "cast" in o["name"].lower() or "to_copy" in o["name"].lower()]
    if cast_ops:
        patterns.append(("dtype_cast", f"{len(cast_ops)} types, {sum(o['calls'] for o in cast_ops):,} calls"))

    copy_ops = [o for o in hf.get("top_ops", []) if any(k in o["name"] for k in ("copy_", "clone", "contiguous", "view", "reshape"))]
    if copy_ops:
        patterns.append(("copy_view", f"{len(copy_ops)} types, {sum(o['calls'] for o in copy_ops):,} calls"))

    empty_ops = [o for o in hf.get("top_ops", []) if o["name"] == "empty_tensor"]
    if empty_ops:
        patterns.append(("mem_alloc", f"empty_tensor: {empty_ops[0]['calls']:,} calls, {empty_ops[0]['total_ms']}ms"))

    emb_ops = [o for o in hf.get("top_ops", []) if "embedding" in o["name"].lower()]
    if emb_ops:
        patterns.append(("embedding", f"aten::embedding: {emb_ops[0]['calls']:,} calls"))

    for key, ops in patterns:
        title = FUSION_TITLES.get(key, {}).get(lang, key)
        desc = FUSION_DESCS.get(key, {}).get(lang, "")
        L.append(f"### {title}")
        L.append(f"- **{tr(lang, 'involves')}** `{ops}`")
        L.append(f"- **{tr(lang, 'recommendation')}** {desc}\n")

    L.append(f"{tr(lang, 'graph_mode_title')}\n")
    L.append("```python")
    L.append(f'# torch.compile ({tr(lang, "recommended_comment")})')
    L.append('model = torch.compile(model, mode="max-autotune")')
    L.append("")
    L.append(f'# {tr(lang, "npu_graph_comment")}')
    L.append('model = torch.compile(model, backend="npu")')
    L.append("```\n")

    return "\n".join(L)




if __name__ == "__main__":
    profiler_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    root = Path(profiler_dir).resolve()
    db_path = None
    for p in root.rglob("ascend_pytorch_profiler.db"):
        db_path = p
        break
    if not db_path:
        print("Error: no DB found", file=sys.stderr)
        sys.exit(1)
    output_dir = root / "out"
    raise SystemExit(analyze_host_bound(db_path, output_dir))
