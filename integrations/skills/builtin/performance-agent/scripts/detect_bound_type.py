#!/usr/bin/env python3
"""Detect Host-Bound vs Device-Bound performance from timeline data.

Host-Bound: Timeline API lines are vertical (device idle, waiting for host dispatch).
  → CPU dispatch is slow, causing NPU starvation.
  → Investigate: flame graph, Python overhead, sync points.

Device-Bound: Timeline API lines are slanted (host waiting for device completion).
  → Device compute/communication is the bottleneck.
  → Investigate: operator hotspots, communication overlap, memory pressure.

This script analyzes trace_view.json or step_trace_time.csv to determine
the dominant bound type using free-time ratio and dispatch patterns.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Optional

from perf_common import load_csv_rows, normalize_key, parse_number, read_json, write_json


# Free time >10% of step total indicates host dispatch bottleneck
_FREE_TIME_THRESHOLD = 0.10

# If free time >20%, it's strongly host-bound
_STRONG_HOST_BOUND = 0.20

# Maximum trace_view.json size to load (500 MB)
_MAX_TRACE_SIZE_BYTES = 500 * 1024 * 1024


def analyze_step_trace_bound(step_rows: list[dict]) -> dict:
    """Analyze bound type from step_trace_time.csv data.

    Uses free/idle time ratio to determine if host dispatch is the bottleneck.
    """
    free_times: list[float] = []
    step_totals: list[float] = []

    for row in step_rows:
        free_val = None
        total_val = None

        for header, value in row.items():
            key = normalize_key(header)
            num = parse_number(value)
            if num is None:
                continue

            if "step" in key and ("time" in key or "total" in key or "interval" in key):
                total_val = num
            elif any(t in key for t in ("idle", "gap", "wait", "tail", "bubble", "free")):
                free_val = num

        if total_val is not None and total_val > 0:
            step_totals.append(total_val)
            free_times.append(free_val or 0.0)

    if not step_totals:
        return {"bound_type": "unknown", "evidence": "No step timing data available"}

    avg_free = sum(free_times) / len(free_times)
    avg_total = sum(step_totals) / len(step_totals)
    free_ratio = avg_free / avg_total if avg_total > 0 else 0.0

    if free_ratio >= _STRONG_HOST_BOUND:
        bound_type = "host_bound"
        severity = "strong"
    elif free_ratio >= _FREE_TIME_THRESHOLD:
        bound_type = "host_bound"
        severity = "moderate"
    elif free_ratio > 0:
        bound_type = "device_bound"
        severity = "normal"
    else:
        bound_type = "device_bound"
        severity = "strong"

    return {
        "bound_type": bound_type,
        "severity": severity,
        "avg_free_time_ms": round(avg_free, 3),
        "avg_step_total_ms": round(avg_total, 3),
        "free_time_ratio": round(free_ratio, 4),
        "steps_analyzed": len(step_totals),
        "evidence": (
            f"Free/idle time is {free_ratio:.1%} of step total "
            f"(avg {avg_free:.1f}ms / {avg_total:.1f}ms over {len(step_totals)} steps)"
        ),
    }


def analyze_trace_view_bound(trace_data: object) -> dict:
    """Analyze bound type from trace_view.json timeline data.

    Looks for API dispatch patterns: vertical lines (host-bound) vs
    slanted lines (device-bound). Falls back to gap analysis.
    """
    events = _extract_events(trace_data)
    if not events:
        return {"bound_type": "unknown", "evidence": "No trace events found"}

    # Classify events by category
    host_dispatch_events: list[dict] = []
    device_compute_events: list[dict] = []
    idle_gap_events: list[dict] = []

    for event in events:
        name = event.get("name", "").lower()
        duration = event.get("duration_ms", 0)

        if any(t in name for t in ("dispatch", "launch", "enqueue", "submit", "host", "cpu")):
            host_dispatch_events.append(event)
        elif any(t in name for t in ("compute", "kernel", "matmul", "conv", "forward", "backward", "fp", "bp")):
            device_compute_events.append(event)
        elif any(t in name for t in ("idle", "gap", "wait", "free", "bubble")):
            idle_gap_events.append(event)

    # Calculate ratios
    host_time = sum(e["duration_ms"] for e in host_dispatch_events)
    device_time = sum(e["duration_ms"] for e in device_compute_events)
    idle_time = sum(e["duration_ms"] for e in idle_gap_events)
    total_time = host_time + device_time + idle_time

    if total_time <= 0:
        return {"bound_type": "unknown", "evidence": "No time data in trace events"}

    host_ratio = host_time / total_time
    device_ratio = device_time / total_time
    idle_ratio = idle_time / total_time

    # Determine bound type
    # Host-bound: high idle/wait ratio (device waiting for host)
    if idle_ratio > 0.20:
        bound_type = "host_bound"
        severity = "strong"
    elif idle_ratio > 0.10:
        bound_type = "host_bound"
        severity = "moderate"
    elif host_ratio > device_ratio:
        bound_type = "host_bound"
        severity = "moderate"
    else:
        bound_type = "device_bound"
        severity = "normal"

    return {
        "bound_type": bound_type,
        "severity": severity,
        "host_dispatch_time_ms": round(host_time, 3),
        "device_compute_time_ms": round(device_time, 3),
        "idle_gap_time_ms": round(idle_time, 3),
        "host_ratio": round(host_ratio, 4),
        "device_ratio": round(device_ratio, 4),
        "idle_ratio": round(idle_ratio, 4),
        "events_analyzed": len(events),
        "evidence": (
            f"Host dispatch: {host_ratio:.1%}, Device compute: {device_ratio:.1%}, "
            f"Idle gaps: {idle_ratio:.1%} ({len(events)} events)"
        ),
    }


_MAX_EXTRACT_DEPTH = 20


def _extract_events(node: object, _depth: int = 0) -> list[dict]:
    """Recursively extract named events with durations from trace data."""
    if _depth > _MAX_EXTRACT_DEPTH:
        return []

    events: list[dict] = []

    if isinstance(node, list):
        for item in node:
            events.extend(_extract_events(item, _depth + 1))
    elif isinstance(node, dict):
        # Check for nested event arrays
        if "traceEvents" in node:
            return _extract_events(node["traceEvents"], _depth + 1)
        if "events" in node:
            return _extract_events(node["events"], _depth + 1)

        name = node.get("name") or node.get("op_name") or node.get("event")
        duration = None
        for key in ("duration_ms", "dur_ms", "time_ms", "elapsed_ms", "duration", "dur"):
            if duration is not None:
                break
            duration = parse_number(node.get(key))

        if name and duration is not None:
            events.append({"name": str(name), "duration_ms": duration})
        else:
            for value in node.values():
                if isinstance(value, (dict, list)):
                    events.extend(_extract_events(value, _depth + 1))

    return events


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect Host-Bound vs Device-Bound performance from profiling data"
    )
    parser.add_argument("--trace-root", help="profiler export root")
    parser.add_argument("--step-csv", help="explicit step_trace_time.csv path")
    parser.add_argument("--trace-json", help="explicit trace_view.json path")
    parser.add_argument("--output-json", required=True, help="output JSON path")
    args = parser.parse_args()

    # Resolve input paths
    step_csv_path = None
    trace_json_path = None

    if args.step_csv:
        step_csv_path = Path(args.step_csv).resolve()
    if args.trace_json:
        trace_json_path = Path(args.trace_json).resolve()

    if args.trace_root:
        root = Path(args.trace_root).resolve()
        ascend_dir = root / "ASCEND_PROFILER_OUTPUT"

        if not step_csv_path:
            candidate = ascend_dir / "step_trace_time.csv"
            if candidate.exists():
                step_csv_path = candidate
            else:
                for match in root.rglob("step_trace_time.csv"):
                    step_csv_path = match
                    break

        if not trace_json_path:
            candidate = ascend_dir / "trace_view.json"
            if candidate.exists():
                trace_json_path = candidate
            else:
                for match in root.rglob("trace_view.json"):
                    trace_json_path = match
                    break

    if not step_csv_path and not trace_json_path:
        print("No step_trace_time.csv or trace_view.json found. Provide --trace-root or explicit paths.", file=sys.stderr)
        raise SystemExit(1)

    # Analyze using available data
    analysis: dict = {"bound_type": "unknown", "evidence": "No data"}

    if trace_json_path and trace_json_path.exists():
        file_size = trace_json_path.stat().st_size
        if file_size > _MAX_TRACE_SIZE_BYTES:
            print(
                f"Warning: trace_view.json is {file_size / (1024**3):.1f} GB, "
                f"exceeds {_MAX_TRACE_SIZE_BYTES / (1024**2):.0f} MB limit. "
                f"Skipping trace view analysis.",
                file=sys.stderr,
            )
        else:
            trace_data = read_json(trace_json_path)
            analysis = analyze_trace_view_bound(trace_data)
            analysis["source"] = f"trace_view: {trace_json_path}"

    # Step CSV provides complementary analysis (free time ratio)
    if step_csv_path and step_csv_path.exists():
        step_rows = load_csv_rows(step_csv_path)
        if step_rows:
            step_analysis = analyze_step_trace_bound(step_rows)

            # If we already have trace analysis, merge
            if analysis["bound_type"] != "unknown":
                analysis["step_trace_analysis"] = step_analysis
                # Disagreement: prefer trace analysis but flag it
                if analysis["bound_type"] != step_analysis["bound_type"]:
                    analysis["note"] = (
                        f"Trace view indicates {analysis['bound_type']} "
                        f"but step trace free-time ratio indicates {step_analysis['bound_type']}"
                    )
            else:
                analysis = step_analysis
                analysis["source"] = f"step_trace: {step_csv_path}"

    # Generate recommendations
    recommendations: list[str] = []
    if analysis["bound_type"] == "host_bound":
        recommendations.extend([
            "Profile CPU-side dispatch: use flame graph to identify Python overhead",
            "Eliminate unnecessary sync points (tensor.item(), tensor.reduce_all(), torch.isfinite())",
            "Move CPU-bound logic to NPU (conditional ops, data preprocessing)",
            "Increase graph mode usage to reduce per-operator host dispatch",
            "Check CPU affinity and NUMA binding for the training process",
        ])
    elif analysis["bound_type"] == "device_bound":
        recommendations.extend([
            "Analyze operator hotspots to find dominant compute kernels",
            "Check communication/compute overlap for distributed training",
            "Verify CUBE unit utilization (should be high for GEMM-heavy models)",
            "Consider operator fusion opportunities (FlashAttention, MatmulAllReduce)",
        ])

    report = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        **analysis,
        "likely_domains": (
            ["host_framework_overhead"] if analysis["bound_type"] == "host_bound"
            else ["compute", "communication"] if analysis["bound_type"] == "device_bound"
            else []
        ),
        "recommended_actions": recommendations,
        "next_action": (
            "Focus on reducing host-side dispatch overhead first."
            if analysis["bound_type"] == "host_bound"
            else "Focus on device-side optimization (compute/communication/memory) first."
            if analysis["bound_type"] == "device_bound"
            else "Collect more profiling data to determine bound type."
        ),
    }

    write_json(Path(args.output_json), report)
    print(json.dumps({
        "bound_type": analysis["bound_type"],
        "severity": analysis.get("severity", "unknown"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
