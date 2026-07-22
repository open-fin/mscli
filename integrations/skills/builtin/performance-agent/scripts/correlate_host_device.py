#!/usr/bin/env python3
"""Correlate host-side and device-side events from trace_view.json.

Identifies host dispatch delays, unnecessary synchronization points,
and links Python-level calls to NPU kernel executions.
"""
import argparse
import json
from pathlib import Path
from typing import Optional

from perf_common import load_csv_rows, load_optional_json, write_json


# Maximum trace_view.json size to load (500 MB)
_MAX_TRACE_SIZE_BYTES = 500 * 1024 * 1024


# Patterns that indicate unnecessary host-device synchronization.
SYNC_INDUCING_PATTERNS = {
    "tensor.item": ["item", "cpu()", "numpy"],
    "tensor.reduce_all": ["reduce_all", "all_reduce_host"],
    "torch.isfinite": ["isfinite", "isnan", "isinf"],
    "h2d_transfer": ["HostToDevice", "h2d", "copy_", "to(npu", "cuda()"],
    "d2h_transfer": ["DeviceToHost", "d2h", "cpu()"],
}

# Host-side event name patterns.
HOST_PATTERNS = [
    "host_", "python", "enqueue", "launch", "dispatch", "schedule",
    "Memcpy", "copy", "Record", "Wait", "Event",
]

# Device-side event name patterns.
DEVICE_PATTERNS = [
    "Kernel", "Compute", "AllReduce", "ReduceScatter", "AllGather",
    "MatMul", "Conv", "BatchNorm", "LayerNorm", "Softmax", "Attention",
    "Optimizer", "Atomic", "aicube", "aivec", "cublas",
]


def _classify_event(event: dict) -> str:
    """Classify an event as host, device, or communication."""
    name = str(event.get("name", "")).lower()
    cat = str(event.get("cat", "")).lower()
    combined = f"{name} {cat}"

    if any(p.lower() in combined for p in DEVICE_PATTERNS):
        return "device"
    if any(p.lower() in combined for p in HOST_PATTERNS):
        return "host"
    if any(kw in combined for kw in ("allreduce", "reduce_scatter", "allgather", "broadcast")):
        return "communication"
    return "unknown"


def _detect_sync_type(event: dict) -> Optional[str]:
    """Detect if an event indicates a sync-inducing operation."""
    name = str(event.get("name", ""))
    name_lower = name.lower()
    for sync_type, patterns in SYNC_INDUCING_PATTERNS.items():
        if any(p.lower() in name_lower for p in patterns):
            return sync_type
    return None


def _get_timestamp(event: dict) -> Optional[float]:
    """Extract timestamp in microseconds from event."""
    for key in ("ts", "timestamp", "start_us", "start_time"):
        val = event.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


def _get_duration(event: dict) -> Optional[float]:
    """Extract duration in microseconds from event."""
    for key in ("dur", "duration", "duration_us"):
        val = event.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    # Fallback: compute from start/end
    ts = _get_timestamp(event)
    end = event.get("end") or event.get("end_us")
    if ts is not None and end is not None:
        try:
            return float(end) - ts
        except (ValueError, TypeError):
            pass
    return None


def parse_trace_events(trace_data: dict) -> list[dict]:
    """Extract events from trace data."""
    if isinstance(trace_data, list):
        return trace_data
    if isinstance(trace_data, dict):
        # Try common keys
        for key in ("events", "traceEvents", "trace_events", "data"):
            events = trace_data.get(key)
            if isinstance(events, list):
                return events
    return []


def classify_events(events: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Separate events into host, device, and communication lists."""
    host_events = []
    device_events = []
    comm_events = []

    for event in events:
        category = _classify_event(event)
        if category == "host":
            host_events.append(event)
        elif category == "device":
            device_events.append(event)
        elif category == "communication":
            comm_events.append(event)

    return host_events, device_events, comm_events


def find_sync_points(events: list[dict]) -> list[dict]:
    """Find synchronization-inducing events."""
    sync_points = []
    for event in events:
        sync_type = _detect_sync_type(event)
        if sync_type:
            duration_us = _get_duration(event)
            sync_points.append({
                "type": sync_type,
                "duration_ms": round(duration_us / 1000, 3) if duration_us is not None else None,
                "event_name": str(event.get("name", "")),
            })
    return sync_points


def compute_dispatch_latency(host_events: list[dict], device_events: list[dict]) -> dict:
    """Compute host-to-device dispatch latency statistics.

    Finds host events that precede device events with no intervening device work.
    """
    host_timestamps = []
    device_timestamps = []

    for ev in host_events:
        ts = _get_timestamp(ev)
        if ts is not None:
            host_timestamps.append(ts)

    for ev in device_events:
        ts = _get_timestamp(ev)
        if ts is not None:
            device_timestamps.append(ts)

    if not host_timestamps or not device_timestamps:
        return {"available": False}

    host_timestamps.sort()
    device_timestamps.sort()

    # Two-pointer: advance through device timestamps for each host timestamp
    gaps = []
    d_idx = 0
    for h_ts in host_timestamps:
        while d_idx < len(device_timestamps) and device_timestamps[d_idx] <= h_ts:
            d_idx += 1
        if d_idx < len(device_timestamps):
            gap = device_timestamps[d_idx] - h_ts
            if gap > 0:
                gaps.append(gap / 1000)  # Convert us to ms
            d_idx += 1

    if not gaps:
        return {"available": False}

    gaps.sort()
    mean_gap = sum(gaps) / len(gaps)
    p95_idx = min(int(len(gaps) * 0.95), len(gaps) - 1)

    return {
        "available": True,
        "mean_ms": round(mean_gap, 3),
        "p95_ms": round(gaps[p95_idx], 3),
        "max_ms": round(gaps[-1], 3),
        "sample_count": len(gaps),
    }


def classify_gaps(
    host_events: list[dict],
    device_events: list[dict],
    sync_points: list[dict],
    dispatch_threshold_us: float = 100.0,
) -> dict[str, int]:
    """Classify timeline gaps by root cause based on timestamp gaps."""
    classification = {
        "host_dispatch_delay": 0,
        "unnecessary_sync": 0,
        "normal_overlap": 0,
    }

    # Count sync-related gaps
    classification["unnecessary_sync"] = len(sync_points)

    # Use timestamp-based classification for host-device dispatch gaps
    host_with_ts = sorted(
        [ev for ev in host_events if _get_timestamp(ev) is not None],
        key=lambda ev: _get_timestamp(ev),
    )
    device_with_ts = sorted(
        [ev for ev in device_events if _get_timestamp(ev) is not None],
        key=lambda ev: _get_timestamp(ev),
    )

    if host_with_ts and device_with_ts:
        # Two-pointer: advance device pointer for each host event
        d_idx = 0
        for h_ev in host_with_ts:
            h_ts = _get_timestamp(h_ev)
            while d_idx < len(device_with_ts) and _get_timestamp(device_with_ts[d_idx]) < h_ts:
                d_idx += 1
            if d_idx < len(device_with_ts):
                gap = _get_timestamp(device_with_ts[d_idx]) - h_ts
                if gap > dispatch_threshold_us:
                    classification["host_dispatch_delay"] += 1
                else:
                    classification["normal_overlap"] += 1
                d_idx += 1
            else:
                classification["host_dispatch_delay"] += 1

    return classification


def generate_recommendations(sync_points: list[dict], gap_classification: dict) -> list[str]:
    """Generate actionable recommendations from correlation analysis."""
    recs = []

    sync_types = {sp["type"] for sp in sync_points}
    if "tensor.item" in sync_types:
        recs.append(
            "Eliminate tensor.item() calls in the training loop hot path. "
            "Use NPU-side logic instead of pulling scalar data back to Host."
        )
    if "tensor.reduce_all" in sync_types:
        recs.append(
            "Replace tensor.reduce_all() with NPU-native alternatives "
            "to avoid unnecessary synchronization."
        )
    if "torch.isfinite" in sync_types:
        recs.append(
            "Replace torch.isfinite() / torch.isnan() checks with NPU-side operations."
        )
    if "h2d_transfer" in sync_types:
        recs.append(
            "Reduce host-to-device transfers by pinning memory and using "
            "prefetch for data loading."
        )
    if "d2h_transfer" in sync_types:
        recs.append(
            "Reduce device-to-host transfers. Avoid printing or logging "
            "tensor values during training steps."
        )

    if gap_classification.get("unnecessary_sync", 0) > 10:
        recs.append(
            f"Found {gap_classification['unnecessary_sync']} synchronization events. "
            "Review the training loop for unnecessary CPU-GPU synchronization."
        )

    if not recs:
        recs.append("No significant host-device synchronization issues detected.")

    return recs


def correlate(
    trace_data: Optional[dict],
    kernel_csv_path: Optional[Path] = None,
) -> dict:
    """Run host-device correlation analysis."""
    if not trace_data:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "host_device_correlation_available": False,
        }

    events = parse_trace_events(trace_data)
    if not events:
        # Try treating trace_data values as event lists
        if isinstance(trace_data, dict):
            for key, value in trace_data.items():
                if isinstance(value, list) and len(value) > 3:
                    events = value
                    break

    if not events:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "host_device_correlation_available": False,
        }

    host_events, device_events, comm_events = classify_events(events)
    sync_points = find_sync_points(events)
    dispatch_latency = compute_dispatch_latency(host_events, device_events)
    gap_classification = classify_gaps(host_events, device_events, sync_points)
    recommendations = generate_recommendations(sync_points, gap_classification)

    # Load kernel details for additional device context
    kernel_details = None
    if kernel_csv_path and kernel_csv_path.exists():
        kernel_details = load_csv_rows(kernel_csv_path)

    correlated_pairs = 0
    if host_events and device_events:
        correlated_pairs = min(len(host_events), len(device_events))

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "host_device_correlation_available": True,
        "total_events": len(events),
        "host_events_count": len(host_events),
        "device_events_count": len(device_events),
        "communication_events_count": len(comm_events),
        "correlated_pairs": correlated_pairs,
        "sync_points": sync_points,
        "sync_points_count": len(sync_points),
        "dispatch_latency_ms": dispatch_latency,
        "gap_classification": gap_classification,
        "kernel_details_available": kernel_details is not None,
        "recommendations": recommendations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correlate host-side and device-side events from trace data"
    )
    parser.add_argument("--trace-root", help="Profiler export root (auto-locates trace_view.json and kernel_details.csv)")
    parser.add_argument("--trace-view-json", help="Path to trace_view.json")
    parser.add_argument("--kernel-details-csv", help="Path to kernel_details.csv (optional)")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    trace_path = None
    kernel_csv_path = None

    if args.trace_root:
        trace_root = Path(args.trace_root).resolve()
        candidate_trace = trace_root / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
        if candidate_trace.exists():
            trace_path = candidate_trace
        kernel_candidates = [
            trace_root / "ASCEND_PROFILER_OUTPUT" / "kernel_details.csv",
            trace_root / "ASCEND_PROFILER_OUTPUT" / "operator_details.csv",
        ]
        for candidate in kernel_candidates:
            if candidate.exists():
                kernel_csv_path = candidate
                break

    if args.trace_view_json:
        trace_path = Path(args.trace_view_json).resolve()
    if args.kernel_details_csv:
        kernel_csv_path = Path(args.kernel_details_csv).resolve()

    trace_data = None
    if trace_path and trace_path.exists():
        file_size = trace_path.stat().st_size
        if file_size > _MAX_TRACE_SIZE_BYTES:
            import sys
            print(
                f"Warning: trace_view.json is {file_size / (1024**3):.1f} GB, "
                f"exceeds {_MAX_TRACE_SIZE_BYTES / (1024**2):.0f} MB limit. "
                f"Skipping trace view analysis.",
                file=sys.stderr,
            )
        else:
            trace_data = load_optional_json(trace_path)

    result = correlate(trace_data, kernel_csv_path)

    write_json(Path(args.output_json), result)
    print(json.dumps({
        "correlation_available": result.get("host_device_correlation_available", False),
        "sync_points": result.get("sync_points_count", 0),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
