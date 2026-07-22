#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from perf_common import load_csv_rows, normalize_key, parse_number, read_json, write_json


def infer_indicators_from_csv(rows: list[dict[str, str]]) -> dict:
    queue_empty_percent = None
    wait_time_ms = None
    throughput_gap = None
    for row in rows:
        for field, value in row.items():
            key = normalize_key(field)
            number = parse_number(value)
            if number is None:
                continue
            if queue_empty_percent is None and ("queue_empty" in key or "empty_rate" in key):
                queue_empty_percent = number
            if wait_time_ms is None and ("wait" in key or "idle" in key):
                wait_time_ms = number
            if throughput_gap is None and ("batch_time" in key or "get_next" in key or "dataset_time" in key):
                throughput_gap = number
    return {
        "queue_empty_percent": queue_empty_percent,
        "wait_time_ms": wait_time_ms,
        "dataset_stage_time_ms": throughput_gap,
    }


def infer_indicators_from_json(payload) -> dict:
    queue_empty_percent = None
    wait_time_ms = None
    warning = None
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = normalize_key(key)
            if isinstance(value, (dict, list)):
                nested = infer_indicators_from_json(value)
                queue_empty_percent = queue_empty_percent or nested["queue_empty_percent"]
                wait_time_ms = wait_time_ms or nested["wait_time_ms"]
                warning = warning or nested["warning"]
                continue
            number = parse_number(value)
            if queue_empty_percent is None and normalized in {"queue_empty_percent", "queue_empty_rate"}:
                queue_empty_percent = number
            if wait_time_ms is None and ("wait" in normalized or "idle" in normalized):
                wait_time_ms = number
            if warning is None and "warning" in normalized:
                warning = str(value)
    elif isinstance(payload, list):
        for item in payload:
            nested = infer_indicators_from_json(item)
            queue_empty_percent = queue_empty_percent or nested["queue_empty_percent"]
            wait_time_ms = wait_time_ms or nested["wait_time_ms"]
            warning = warning or nested["warning"]
    return {
        "queue_empty_percent": queue_empty_percent,
        "wait_time_ms": wait_time_ms,
        "warning": warning,
    }


def default_paths(
    trace_root: Path,
) -> tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    base = trace_root / "ASCEND_PROFILER_OUTPUT"
    dataset = base / "dataset.csv"
    pipeline_csv = next(iter(sorted(base.glob("minddata_pipeline_summary_*.csv"))), None)
    pipeline_json = next(iter(sorted(base.glob("minddata_pipeline_summary_*.json"))), None)
    data_preprocess = base / "data_preprocess.csv"
    return (
        dataset if dataset.exists() else None,
        pipeline_csv,
        pipeline_json,
        data_preprocess if data_preprocess.exists() else None,
    )


def summarize_data_preprocess(rows: list[dict[str, str]]) -> dict:
    """Summarize input preprocessing data from data_preprocess.csv.

    Computes per-node totals and overall statistics for dispatch, memcpy,
    and compute time to surface input-pipeline bottlenecks.
    """
    totals: dict[str, dict] = {}
    for row in rows:
        node = row.get("Node", "").strip()
        if not node:
            continue
        dispatch = parse_number(row.get("Dispatch_time(us)", 0)) or 0
        memcpy = parse_number(row.get("Memcpy_time(us)", 0)) or 0
        compute = parse_number(row.get("Compute_time(us)", 0)) or 0
        total = parse_number(row.get("Total_time(us)", 0)) or 0
        entry = totals.setdefault(node, {"dispatch_us": 0.0, "memcpy_us": 0.0, "compute_us": 0.0, "total_us": 0.0, "count": 0})
        entry["dispatch_us"] += dispatch
        entry["memcpy_us"] += memcpy
        entry["compute_us"] += compute
        entry["total_us"] += total
        entry["count"] += 1

    ranked = sorted(totals.values(), key=lambda x: x["total_us"], reverse=True)
    total_dispatch_us = sum(e["dispatch_us"] for e in totals.values())
    total_memcpy_us = sum(e["memcpy_us"] for e in totals.values())
    total_compute_us = sum(e["compute_us"] for e in totals.values())
    total_time_us = sum(e["total_us"] for e in totals.values())

    top_nodes = []
    for name, stats in sorted(totals.items(), key=lambda x: x[1]["total_us"], reverse=True)[:5]:
        top_nodes.append({
            "node": name,
            "total_time_ms": round(stats["total_us"] / 1000.0, 3),
            "dispatch_time_ms": round(stats["dispatch_us"] / 1000.0, 3),
            "memcpy_time_ms": round(stats["memcpy_us"] / 1000.0, 3),
            "count": stats["count"],
        })

    avg_dispatch_ms = total_dispatch_us / len(rows) / 1000.0 if rows else 0

    return {
        "total_nodes": len(totals),
        "total_records": len(rows),
        "total_time_ms": round(total_time_us / 1000.0, 3),
        "total_dispatch_time_ms": round(total_dispatch_us / 1000.0, 3),
        "total_memcpy_time_ms": round(total_memcpy_us / 1000.0, 3),
        "total_compute_time_ms": round(total_compute_us / 1000.0, 3),
        "avg_dispatch_time_ms": round(avg_dispatch_ms, 3),
        "top_nodes": top_nodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize input-pipeline pressure from profiler exports")
    parser.add_argument("--trace-root", help="profiler export root")
    parser.add_argument("--dataset-csv", help="explicit dataset.csv path")
    parser.add_argument("--pipeline-csv", help="explicit minddata pipeline summary CSV path")
    parser.add_argument("--pipeline-json", help="explicit minddata pipeline summary JSON path")
    parser.add_argument("--output-json", required=True, help="path to write the input-pipeline summary JSON")
    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv).resolve() if args.dataset_csv else None
    pipeline_csv = Path(args.pipeline_csv).resolve() if args.pipeline_csv else None
    pipeline_json = Path(args.pipeline_json).resolve() if args.pipeline_json else None
    data_preprocess = None
    if args.trace_root:
        inferred_dataset, inferred_pipeline_csv, inferred_pipeline_json, inferred_data_preprocess = default_paths(Path(args.trace_root).resolve())
        dataset_csv = dataset_csv or inferred_dataset
        pipeline_csv = pipeline_csv or inferred_pipeline_csv
        pipeline_json = pipeline_json or inferred_pipeline_json
        data_preprocess = inferred_data_preprocess

    has_pipeline_files = any(path and path.exists() for path in (dataset_csv, pipeline_csv, pipeline_json))
    has_data_preprocess = data_preprocess and data_preprocess.exists()

    if not has_pipeline_files and not has_data_preprocess:
        report = {
            "source_files": {
                "dataset_csv": None,
                "pipeline_csv": None,
                "pipeline_json": None,
                "data_preprocess_csv": None,
            },
            "pipeline_available": False,
            "queue_empty_percent": None,
            "wait_time_ms": None,
            "dataset_stage_time_ms": None,
            "warning": None,
            "bottleneck_detected": False,
            "likely_domains": [],
            "next_action": "Collect minddata or data_preprocess pipeline evidence before analyzing input-pipeline bottlenecks.",
        }
        write_json(Path(args.output_json), report)
        print(json.dumps({"pipeline_available": False}, indent=2))
        return 0

    csv_indicators = {"queue_empty_percent": None, "wait_time_ms": None, "dataset_stage_time_ms": None}
    if dataset_csv and dataset_csv.exists():
        csv_indicators = infer_indicators_from_csv(load_csv_rows(dataset_csv))
    if pipeline_csv and pipeline_csv.exists():
        pipeline_indicators = infer_indicators_from_csv(load_csv_rows(pipeline_csv))
        for key, value in pipeline_indicators.items():
            csv_indicators[key] = csv_indicators[key] if csv_indicators[key] is not None else value

    json_indicators = {"queue_empty_percent": None, "wait_time_ms": None, "warning": None}
    if pipeline_json and pipeline_json.exists():
        json_indicators = infer_indicators_from_json(read_json(pipeline_json))

    queue_empty_percent = csv_indicators["queue_empty_percent"] or json_indicators["queue_empty_percent"]
    wait_time_ms = csv_indicators["wait_time_ms"] or json_indicators["wait_time_ms"]
    dataset_stage_time_ms = csv_indicators["dataset_stage_time_ms"]
    data_preprocess_summary = None

    bottleneck_detected = False
    if has_data_preprocess and not has_pipeline_files:
        # Use data_preprocess.csv as fallback pipeline evidence
        dp_rows = load_csv_rows(data_preprocess)
        if dp_rows:
            data_preprocess_summary = summarize_data_preprocess(dp_rows)
            # Derive approximate indicators from dispatch time (host overhead)
            if data_preprocess_summary["avg_dispatch_time_ms"] is not None:
                avg_dispatch = data_preprocess_summary["avg_dispatch_time_ms"]
                if wait_time_ms is None and avg_dispatch >= 10:
                    wait_time_ms = avg_dispatch
                    bottleneck_detected = True
    if queue_empty_percent is not None and queue_empty_percent >= 20:
        bottleneck_detected = True
    if wait_time_ms is not None and wait_time_ms >= 10:
        bottleneck_detected = True
    if json_indicators["warning"]:
        bottleneck_detected = True

    report = {
        "source_files": {
            "dataset_csv": str(dataset_csv) if dataset_csv and dataset_csv.exists() else None,
            "pipeline_csv": str(pipeline_csv) if pipeline_csv and pipeline_csv.exists() else None,
            "pipeline_json": str(pipeline_json) if pipeline_json and pipeline_json.exists() else None,
            "data_preprocess_csv": str(data_preprocess) if data_preprocess and data_preprocess.exists() else None,
        },
        "pipeline_available": has_pipeline_files or has_data_preprocess,
        "queue_empty_percent": queue_empty_percent,
        "wait_time_ms": wait_time_ms,
        "dataset_stage_time_ms": dataset_stage_time_ms,
        "warning": json_indicators["warning"],
        "bottleneck_detected": bottleneck_detected,
        "data_preprocess_summary": data_preprocess_summary,
        "likely_domains": ["input_pipeline"] if bottleneck_detected else [],
        "next_action": (
            "Validate queue-empty rate, pre-compute idle time, and end-to-end throughput after the first input-pipeline change."
            if bottleneck_detected
            else "Input-pipeline evidence does not currently dominate the exported summaries."
        ),
    }
    write_json(Path(args.output_json), report)
    print(json.dumps({"bottleneck_detected": report["bottleneck_detected"], "queue_empty_percent": report["queue_empty_percent"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
