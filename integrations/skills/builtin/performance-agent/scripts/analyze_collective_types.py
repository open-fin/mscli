#!/usr/bin/env python3
"""Classify collective communication operations by functional type.

Breaks down communication.json records into categories (SyncBN,
GradientAllReduce, ParameterAllGather, ReduceScatter, Broadcast, Other)
and reports per-type time, count, and share.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

from perf_common import load_optional_json, parse_number, read_json, write_json


# Classification rules: (type_name, list of regex patterns)
COLLECTIVE_TYPE_RULES = [
    ("SyncBN", [
        r"(?i)(sync.?batch.?norm|syncbn|batch.?norm.?sync|bn.?sync|sync_batchnorm)",
    ]),
    ("ReduceScatter", [
        r"(?i)(reduce.?scatter|reduce_scatter)",
    ]),
    ("AllGather", [
        r"(?i)(all.?gather|allgather)",
    ]),
    ("AllReduce", [
        r"(?i)(all.?reduce|allreduce)",
    ]),
    ("Broadcast", [
        r"(?i)(broadcast)",
    ]),
]

# Size threshold (MB) for distinguishing gradient AllReduce from small control AllReduce
GRADIENT_ALLREDUCE_SIZE_MB = 1.0


def classify_collective(name: str, size_mb: Optional[float] = None) -> str:
    """Classify a collective operation name into a functional type."""
    for type_name, patterns in COLLECTIVE_TYPE_RULES:
        for pattern in patterns:
            if re.search(pattern, name):
                return type_name

    # Fallback: if name is empty, return Other
    return "Other"


def refine_allreduce_type(
    base_type: str,
    size_mb: Optional[float],
    count: int,
    total_time_ms: float,
) -> str:
    """Refine AllReduce into GradientAllReduce vs small-packet AllReduce."""
    if base_type != "AllReduce":
        return base_type

    avg_size_mb = (size_mb / count) if count > 0 and size_mb is not None else None
    if avg_size_mb is not None and avg_size_mb >= GRADIENT_ALLREDUCE_SIZE_MB:
        return "GradientAllReduce"
    if avg_size_mb is not None:
        return "SmallPacketAllReduce"
    # No size info: use time heuristic
    avg_time_ms = total_time_ms / count if count > 0 else 0
    if avg_time_ms > 0.5:
        return "GradientAllReduce"
    return "SmallPacketAllReduce"


def load_communication_records(trace_root: Path) -> list[dict]:
    """Load and flatten communication records from trace root."""
    comm_paths = [
        trace_root / "ASCEND_PROFILER_OUTPUT" / "communication.json",
        trace_root / "communication.json",
    ]
    comm_path = None
    for p in comm_paths:
        if p.exists():
            comm_path = p
            break

    if comm_path is None:
        # Search recursively
        for match in trace_root.rglob("communication.json"):
            comm_path = match
            break

    if comm_path is None:
        return []

    raw = read_json(comm_path)
    return _flatten_records(raw)


def _flatten_records(node) -> list[dict]:
    """Flatten nested communication JSON into records with name/time/count/size."""
    if isinstance(node, list):
        records: list[dict] = []
        for item in node:
            records.extend(_flatten_records(item))
        return records
    if isinstance(node, dict):
        name_keys = {"name", "op_name", "operator", "operator_name", "collective", "task_name"}
        time_keys = {"time", "time_ms", "duration", "duration_ms", "total_time", "total_time_ms", "elapse_time"}
        count_keys = {"count", "calls", "op_count"}
        size_keys = {"size_mb", "data_size_mb", "msg_size_mb", "message_size_mb"}

        normalized = {k.strip().lower().replace(" ", "_"): v for k, v in node.items()}

        name_key = next((k for k in normalized if k in name_keys), None)
        time_key = next((k for k in normalized if k in time_keys), None)
        if name_key and time_key:
            name = str(normalized[name_key]).strip()
            time_value = parse_number(normalized[time_key])
            if name and time_value is not None:
                count = None
                size_mb = None
                for k in normalized:
                    if count is None and k in count_keys:
                        count = parse_number(normalized[k])
                    if size_mb is None and k in size_keys:
                        size_mb = parse_number(normalized[k])
                return [{
                    "name": name,
                    "time_ms": time_value,
                    "count": int(count) if count is not None else 1,
                    "size_mb": size_mb,
                }]
        records: list[dict] = []
        for value in node.values():
            records.extend(_flatten_records(value))
        return records
    return []


def analyze_collective_types(records: list[dict]) -> dict:
    """Classify and aggregate collective records by functional type."""
    if not records:
        return {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "collective_type_analysis_available": False,
        }

    total_time = 0.0
    total_count = 0
    type_agg: dict[str, dict] = {}

    for rec in records:
        name = rec.get("name", "")
        time_ms = rec.get("time_ms", 0)
        count = rec.get("count", 1)
        size_mb = rec.get("size_mb")

        base_type = classify_collective(name, size_mb)
        final_type = refine_allreduce_type(base_type, size_mb, count, time_ms)

        total_time += time_ms
        total_count += count

        if final_type not in type_agg:
            type_agg[final_type] = {
                "type": final_type,
                "total_time_ms": 0.0,
                "count": 0,
                "total_size_mb": 0.0,
                "size_samples": 0,
                "sample_names": [],
            }
        agg = type_agg[final_type]
        agg["total_time_ms"] += time_ms
        agg["count"] += count
        if size_mb is not None:
            agg["total_size_mb"] += size_mb
            agg["size_samples"] += 1
        if len(agg["sample_names"]) < 5 and name not in agg["sample_names"]:
            agg["sample_names"].append(name)

    # Build per-type results
    types_result = []
    for type_name in sorted(type_agg, key=lambda t: type_agg[t]["total_time_ms"], reverse=True):
        agg = type_agg[type_name]
        share = (agg["total_time_ms"] / total_time * 100) if total_time > 0 else 0
        avg_size = (agg["total_size_mb"] / agg["size_samples"]) if agg["size_samples"] > 0 else None
        types_result.append({
            "type": type_name,
            "total_time_ms": round(agg["total_time_ms"], 3),
            "count": agg["count"],
            "avg_time_ms": round(agg["total_time_ms"] / agg["count"], 4) if agg["count"] > 0 else 0,
            "share_percent": round(share, 2),
            "avg_size_mb": round(avg_size, 3) if avg_size is not None else None,
            "sample_names": agg["sample_names"],
        })

    dominant = types_result[0] if types_result else None
    syncbn_dominant = dominant is not None and dominant["type"] == "SyncBN"
    syncbn_share = 0.0
    for t in types_result:
        if t["type"] == "SyncBN":
            syncbn_share = t["share_percent"]
            break

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "collective_type_analysis_available": True,
        "types": types_result,
        "dominant_type": dominant["type"] if dominant else None,
        "dominant_type_share_percent": dominant["share_percent"] if dominant else 0,
        "syncbn_dominant": syncbn_dominant,
        "syncbn_share_percent": round(syncbn_share, 2),
        "total_collective_time_ms": round(total_time, 3),
        "total_collective_count": total_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify collective communication operations by functional type"
    )
    parser.add_argument("--trace-root", help="Profiler output root directory")
    parser.add_argument("--step-json", help="Step breakdown JSON (optional, for ratio context)")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    args = parser.parse_args()

    if not args.trace_root:
        print("Error: --trace-root is required", file=sys.stderr)
        return 1

    trace_root = Path(args.trace_root).resolve()
    records = load_communication_records(trace_root)

    if not records:
        result = {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "collective_type_analysis_available": False,
            "reason": "No communication records found",
        }
        write_json(Path(args.output_json), result)
        print(json.dumps({"collective_type_analysis_available": False}))
        return 0

    result = analyze_collective_types(records)
    write_json(Path(args.output_json), result)
    print(json.dumps({
        "collective_type_analysis_available": result.get("collective_type_analysis_available", False),
        "dominant_type": result.get("dominant_type"),
        "syncbn_dominant": result.get("syncbn_dominant", False),
        "types_count": len(result.get("types", [])),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
