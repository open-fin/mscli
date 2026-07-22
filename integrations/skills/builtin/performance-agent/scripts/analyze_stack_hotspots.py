#!/usr/bin/env python3
"""Analyze Python call-stack hotspots from profiler DB when with_stack=True.

When callchainId data is available in the profiler DB, this script aggregates
CPU ops by their Python call site, revealing which functions/lines generate
the most small-operator dispatch overhead.
"""
import argparse
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

from perf_common import write_json
from report_i18n import tr


def _parse_stack_frames(stack_str: str) -> list[dict]:
    """Parse a semicolon-separated stack trace string.

    Format: ``func_name (file.py:line);func_name (file.py:line);...``
    """
    frames = []
    if not stack_str:
        return frames
    for part in stack_str.split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(.+?)\s*\((.+?):(\d+)\)", part)
        if m:
            frames.append({
                "function": m.group(1).strip(),
                "file": m.group(2).strip(),
                "line": int(m.group(3)),
            })
    return frames


def analyze_stack_hotspots(db_path: Path, output_dir: Path, top_n: int = 20):
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(str(db_path))
    cur = db.cursor()

    # Check if callchainId column exists and has data
    try:
        row = cur.execute(
            "SELECT COUNT(*) FROM PYTORCH_API WHERE callchainId IS NOT NULL"
        ).fetchone()
    except sqlite3.OperationalError:
        print("No callchainId column found in DB. Stack trace data is not available.")
        print("Re-run profiling with with_stack=True to enable this analysis.")
        db.close()
        return None

    non_null_count = row[0]
    if non_null_count == 0:
        print(f"No stack trace data found ({non_null_count} non-null callchainId).")
        print("Re-run profiling with with_stack=True to enable this analysis.")
        db.close()
        return None

    print(f"Found {non_null_count:,} ops with stack trace data. Analyzing...")

    # Load string map
    str_map = dict(cur.execute("SELECT id, value FROM STRING_IDS").fetchall())

    # Get CPU ops with callchain data
    # callchainId references STRING_IDS for the stack trace string
    cur.execute("""
        SELECT p.name, p.callchainId,
               COUNT(*) as calls,
               SUM(CAST(p.endNs AS REAL) - CAST(p.startNs AS REAL)) as total_ns,
               AVG(CAST(p.endNs AS REAL) - CAST(p.startNs AS REAL)) as avg_ns
        FROM PYTORCH_API p
        WHERE p.type = 50001
          AND p.callchainId IS NOT NULL
          AND CAST(p.endNs AS REAL) > CAST(p.startNs AS REAL)
        GROUP BY p.name, p.callchainId
        ORDER BY total_ns DESC
    """)

    # Group by Python call site (file:line)
    site_ops = defaultdict(lambda: {"calls": 0, "total_ns": 0, "ops": defaultdict(lambda: {"calls": 0, "total_ns": 0})})

    for name_id, callchain_id, calls, total_ns, avg_ns in cur.fetchall():
        op_name = str_map.get(name_id, f"ID:{name_id}")
        stack_str = str_map.get(callchain_id, "")
        frames = _parse_stack_frames(stack_str)

        # Use the top (innermost) frame as the call site
        site_key = "unknown"
        if frames:
            f = frames[0]
            site_key = f"{f['file']}:{f['line']} ({f['function']})"

        site_ops[site_key]["calls"] += calls
        site_ops[site_key]["total_ns"] += int(total_ns)
        site_ops[site_key]["ops"][op_name]["calls"] += calls
        site_ops[site_key]["ops"][op_name]["total_ns"] += int(total_ns)

    db.close()

    # Sort by total time
    sorted_sites = sorted(site_ops.items(), key=lambda x: x[1]["total_ns"], reverse=True)[:top_n]

    total_cpu_ns = sum(s["total_ns"] for _, s in sorted_sites)

    # Build result
    hotspots = []
    for site, data in sorted_sites:
        top_ops = sorted(data["ops"].items(), key=lambda x: x[1]["total_ns"], reverse=True)[:5]
        hotspots.append({
            "call_site": site,
            "calls": data["calls"],
            "total_ms": round(data["total_ns"] / 1e6, 1),
            "avg_us": round(data["total_ns"] / data["calls"] / 1e3, 1),
            "top_ops": [{"name": name, "calls": od["calls"], "total_ms": round(od["total_ns"] / 1e6, 1)}
                        for name, od in top_ops],
        })

    # Render report
    for lang, suffix in [("en", "_en"), ("zh", "_zh")]:
        report = _render_report(hotspots, non_null_count, lang)
        (output_dir / f"stack_hotspots{suffix}.md").write_text(report, encoding="utf-8")

    write_json(meta_dir / "stack_hotspots.json", {"hotspots": hotspots, "stack_ops_count": non_null_count})

    print(f"\nTop {len(hotspots)} call-site hotspots:")
    for i, h in enumerate(hotspots[:5], 1):
        print(f"  {i}. {h['call_site']}: {h['total_ms']}ms, {h['calls']:,} calls")
    print(f"\nReports: {output_dir / 'stack_hotspots_en.md'}, {output_dir / 'stack_hotspots_zh.md'}")
    return {"hotspots": hotspots}


def _render_report(hotspots, stack_ops_count, lang="en"):
    L = [tr(lang, "sh_title"),
         f"{tr(lang, 'sh_ops_count', count=stack_ops_count)}\n"]
    L.append(f"{tr(lang, 'sh_top_sites')}\n")
    L.append(f"| # | {tr(lang, 'sh_col_call_site')} | {tr(lang, 'sh_col_calls')} | {tr(lang, 'sh_col_total')} | {tr(lang, 'sh_col_avg')} | {tr(lang, 'sh_col_top_ops')} |")
    L.append("|---|-----------|-------|-------|-----|---------|")

    for i, h in enumerate(hotspots, 1):
        top_ops_str = ", ".join(f"{o['name'][:30]}({o['total_ms']}ms)" for o in h["top_ops"][:3])
        L.append(f"| {i} | `{h['call_site'][:60]}` | {h['calls']:,} | {h['total_ms']}ms | {h['avg_us']}us | {top_ops_str} |")

    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Python call-stack hotspots from profiler DB")
    parser.add_argument("profiler_dir", help="Profiler data directory")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top hotspots to report")
    args = parser.parse_args()

    root = Path(args.profiler_dir).resolve()
    db_path = None
    for p in root.rglob("ascend_pytorch_profiler.db"):
        db_path = p
        break
    if not db_path:
        print("Error: no DB found", file=sys.stderr)
        sys.exit(1)

    output_dir = root / "out"
    raise SystemExit(analyze_stack_hotspots(db_path, output_dir, args.top_n) is None)
