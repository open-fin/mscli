#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from perf_common import normalize_key, parse_number, read_json, write_json


NAME_KEYS = {"name", "op_name", "operator", "operator_name", "collective", "task_name"}
TIME_KEYS = {"time", "time_ms", "duration", "duration_ms", "total_time", "total_time_ms", "elapse_time"}
COUNT_KEYS = {"count", "calls", "op_count"}
SIZE_KEYS = {"size_mb", "data_size_mb", "msg_size_mb", "message_size_mb", "transit_size_mb"}
# Collective time keys: actual numeric time fields, NOT container keys like "communication_time_info"
COLLECTIVE_TIME_KEYS = {"elapse_time_ms", "elapsed_time_ms", "time_ms", "duration_ms", "total_time_ms"}
OP_PREFIXES = ("hcom_", "allreduce", "allgather", "broadcast", "reducescatter")
# Keys that indicate a metric container (not an actual op name) - used to skip structural keys
_METRIC_KEY_PREFIXES = ("communication", "bandwidth", "rdma", "hccs", "pcie", "sdma", "sio", "start")


def _op_fraction(normalized: dict) -> float:
    """Fraction of keys that look like collective op names (hcom_*, allreduce*, etc.)."""
    if not normalized:
        return 0.0
    matches = sum(1 for k in normalized if any(k.startswith(p) for p in OP_PREFIXES))
    return matches / len(normalized)


def flatten_records(node) -> list[dict]:
    if isinstance(node, list):
        records: list[dict] = []
        for item in node:
            records.extend(flatten_records(item))
        return records
    if isinstance(node, dict):
        normalized = {normalize_key(key): value for key, value in node.items()}
        name_key = next((key for key in normalized if key in NAME_KEYS), None)
        time_key = next((key for key in normalized if key in TIME_KEYS), None)
        if name_key and time_key:
            name = str(normalized[name_key]).strip()
            time_value = parse_number(normalized[time_key])
            if name and time_value is not None:
                count = None
                size_mb = None
                for key in normalized:
                    if count is None and key in COUNT_KEYS:
                        count = parse_number(normalized[key])
                    if size_mb is None and key in SIZE_KEYS:
                        size_mb = parse_number(normalized[key])
                return [
                    {
                        "name": name,
                        "time_ms": time_value,
                        "count": int(count) if count is not None else 1,
                        "size_mb": size_mb,
                    }
                ]

        # Handle nested collective format: step > collective > {op_name} > {time_info_key, bw_info_key}
        # Also handle when normalized keys are op names (coll_dict level without 'collective' key)
        in_collective_branch = (
            "collective" in normalized
            or any(k in normalized for k in ("communication_time_info", "communication_bandwidth_info"))
            or _op_fraction(normalized) > 0.5
        )
        if in_collective_branch:
            records: list[dict] = []
            for coll_key, coll_value in node.items():
                if not isinstance(coll_value, dict):
                    continue
                coll_name = str(coll_key).strip()
                # Skip metric container keys (Communication Time Info, Bandwidth, RDMA, HCCS, etc.)
                if any(coll_name.lower().startswith(p) for p in _METRIC_KEY_PREFIXES):
                    continue
                ncoll = {normalize_key(k): v for k, v in coll_value.items()}
                # Is coll_value a container of multiple op records (e.g., the collective dict)?
                is_op_container = _op_fraction(ncoll) > 0.5
                # Is coll_value a leaf op data dict (e.g., {Communication Time Info: {...}})?
                is_op_data = (
                    "communication_time_info" in ncoll or "communication_bandwidth_info" in ncoll
                )
                coll_name_normalized = normalize_key(coll_name)
                is_real_op_name = any(coll_name_normalized.startswith(p) for p in OP_PREFIXES)

                time_ms = None
                size_mb = None
                # Check top-level of coll_value if it's neither container nor op_data
                if not is_op_container and not is_op_data:
                    for tk in COLLECTIVE_TIME_KEYS:
                        if tk in ncoll:
                            tv = parse_number(ncoll[tk])
                            if tv is not None:
                                time_ms = tv
                                break
                    for sk in SIZE_KEYS:
                        if sk in ncoll:
                            sv = parse_number(ncoll[sk])
                            if sv is not None:
                                size_mb = sv
                                break
                # Check nested metric categories (RDMA, HCCS, etc.) within op data or container
                if is_op_data or is_op_container:
                    for metric_value in coll_value.values():
                        if isinstance(metric_value, dict):
                            nmetric = {normalize_key(k): v for k, v in metric_value.items()}
                            if time_ms is None:
                                for tk in COLLECTIVE_TIME_KEYS:
                                    if tk in nmetric:
                                        tv = parse_number(nmetric[tk])
                                        if tv is not None:
                                            time_ms = tv
                                            break
                            if size_mb is None:
                                for sk in SIZE_KEYS:
                                    if sk in nmetric:
                                        sv = parse_number(nmetric[sk])
                                        if sv is not None:
                                            size_mb = sv
                                            break
                # Create record when coll_key is a real op name and we found time_ms
                if coll_name and time_ms is not None and not is_op_container and (is_real_op_name or not is_op_data):
                    records.append({
                        "name": coll_name,
                        "time_ms": time_ms,
                        "count": 1,
                        "size_mb": size_mb,
                    })
                # Recurse into container dicts (not op data dicts - they're leaves)
                if is_op_container:
                    records.extend(flatten_records(coll_value))
                if is_op_data:
                    continue
            for value in node.values():
                if isinstance(value, dict):
                    records.extend(flatten_records(value))
                elif isinstance(value, list):
                    records.extend(flatten_records(value))
            return records

    records: list[dict] = []
    for value in node.values():
        if isinstance(value, dict):
            records.extend(flatten_records(value))
        elif isinstance(value, list):
            records.extend(flatten_records(value))
    return records


def matrix_stats(node) -> dict:
    values: list[float] = []
    if isinstance(node, list):
        for item in node:
            stats = matrix_stats(item)
            values.extend(stats.get("values", []))
    elif isinstance(node, dict):
        if "values" in node:
            stats = matrix_stats(node["values"])
            values.extend(stats.get("values", []))
        else:
            for value in node.values():
                stats = matrix_stats(value)
                values.extend(stats.get("values", []))
    else:
        number = parse_number(node)
        if number is not None:
            values.append(number)

    positives = [value for value in values if value > 0]
    if not positives:
        return {"values": [], "imbalance_ratio": None}
    return {
        "values": positives,
        "imbalance_ratio": round(max(positives) / min(positives), 3) if min(positives) > 0 else None,
    }


def summarize_records(records: list[dict]) -> dict:
    totals: dict[str, dict] = {}
    total_time = 0.0
    total_calls = 0
    for record in records:
        total_time += record["time_ms"]
        total_calls += record["count"]
        current = totals.setdefault(
            record["name"],
            {"name": record["name"], "time_ms": 0.0, "count": 0, "size_mb": 0.0, "size_samples": 0},
        )
        current["time_ms"] += record["time_ms"]
        current["count"] += record["count"]
        if record["size_mb"] is not None:
            current["size_mb"] += record["size_mb"]
            current["size_samples"] += 1

    ranked = sorted(totals.values(), key=lambda item: item["time_ms"], reverse=True)
    top_collectives = []
    for item in ranked[:5]:
        share = item["time_ms"] / total_time * 100 if total_time else 0.0
        top_collectives.append(
            {
                "name": item["name"],
                "time_ms": round(item["time_ms"], 3),
                "share_percent": round(share, 2),
                "count": item["count"],
                "avg_size_mb": round(item["size_mb"] / item["size_samples"], 3) if item["size_samples"] else None,
            }
        )

    pressure = "low"
    if top_collectives and top_collectives[0]["share_percent"] >= 40:
        pressure = "high"
    elif top_collectives and top_collectives[0]["share_percent"] >= 20:
        pressure = "moderate"

    return {
        "records_used": len(records),
        "collective_count": total_calls,
        "total_time_ms": round(total_time, 3),
        "communication_pressure": pressure,
        "top_collectives": top_collectives,
        "dominant_collective": top_collectives[0] if top_collectives else None,
    }


def default_comm_paths(trace_root: Path) -> tuple[Optional[Path], Optional[Path]]:
    comm = trace_root / "ASCEND_PROFILER_OUTPUT" / "communication.json"
    matrix = trace_root / "ASCEND_PROFILER_OUTPUT" / "communication_matrix.json"
    return (comm if comm.exists() else None, matrix if matrix.exists() else None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize communication overhead from profiler JSON exports")
    parser.add_argument("--trace-root", help="profiler export root")
    parser.add_argument("--communication-json", help="explicit communication.json path")
    parser.add_argument("--matrix-json", help="explicit communication_matrix.json path")
    parser.add_argument("--output-json", required=True, help="path to write the communication summary JSON")
    args = parser.parse_args()

    comm_path = Path(args.communication_json).resolve() if args.communication_json else None
    matrix_path = Path(args.matrix_json).resolve() if args.matrix_json else None
    if args.trace_root:
        inferred_comm, inferred_matrix = default_comm_paths(Path(args.trace_root).resolve())
        comm_path = comm_path or inferred_comm
        matrix_path = matrix_path or inferred_matrix

    if not comm_path or not comm_path.exists():
        print("communication.json was not found. Provide --communication-json or --trace-root with communication artifacts.", file=sys.stderr)
        raise SystemExit(1)

    comm_records = flatten_records(read_json(comm_path))
    summary = summarize_records(comm_records)
    imbalance_ratio = None
    if matrix_path and matrix_path.exists():
        matrix_payload = read_json(matrix_path)
        matrix_result = matrix_stats(matrix_payload)
        imbalance_ratio = matrix_result["imbalance_ratio"]

    report = {
        "source_files": {
            "communication_json": str(comm_path),
            "matrix_json": str(matrix_path) if matrix_path and matrix_path.exists() else None,
        },
        **summary,
        "matrix_imbalance_ratio": imbalance_ratio,
        "likely_domains": ["communication"] if summary["top_collectives"] else [],
        "next_action": (
            "Validate communication overlap, collective count, and step tail before changing compute kernels."
            if summary["communication_pressure"] in {"moderate", "high"}
            else "Communication does not currently dominate the exported evidence."
        ),
    }

    write_json(Path(args.output_json), report)
    print(json.dumps({"dominant_collective": report["dominant_collective"], "pressure": report["communication_pressure"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
