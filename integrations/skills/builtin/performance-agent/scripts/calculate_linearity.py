#!/usr/bin/env python3
"""Calculate cluster linearity (scaling efficiency) from profiling data.

Linearity = actual_cluster_throughput / (N × single_device_throughput)

Threshold: <0.8 indicates communication bottleneck after ruling out IO/CPU factors.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from perf_common import find_rank_dirs, find_step_trace_csv, avg_step_time_from_csv, read_json, write_json



def _infer_batch_config(profiler_info: Optional[dict]) -> dict:
    """Extract batch size and parallel config from profiler info."""
    result = {
        "batch_size": None,
        "gradient_accumulation_steps": None,
        "micro_batch_size": None,
        "seq_len": None,
    }
    if not profiler_info:
        return result

    for key in ("batch_size", "global_batch_size", "train_batch_size"):
        val = profiler_info.get(key)
        if val is not None:
            result["batch_size"] = val
            break

    for key in ("gradient_accumulation_steps", "grad_accum_steps", "accumulation_steps"):
        val = profiler_info.get(key)
        if val is not None:
            result["gradient_accumulation_steps"] = val
            break

    for key in ("micro_batch_size", "mbs", "minibatch_size"):
        val = profiler_info.get(key)
        if val is not None:
            result["micro_batch_size"] = val
            break

    for key in ("seq_length", "seq_len", "max_sequence_length"):
        val = profiler_info.get(key)
        if val is not None:
            result["seq_len"] = val
            break

    return result


def calculate_linearity(
    single_rank_step_ms: float,
    cluster_step_ms: float,
) -> float:
    """Calculate linearity = single_device_throughput / cluster_throughput_per_device.

    Since throughput ∝ 1/step_time:
      linearity = single_step_time / (cluster_step_time)
    (assuming same batch size per device).

    More precisely:
      linearity = (actual_cluster_throughput) / (N × single_device_throughput)
                = (BS × N / cluster_step) / (N × BS / single_step)
                = single_step / cluster_step
    """
    if cluster_step_ms <= 0 or single_rank_step_ms <= 0:
        return 0.0
    return single_rank_step_ms / cluster_step_ms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calculate cluster linearity (scaling efficiency)"
    )
    parser.add_argument("--trace-root", required=True, help="cluster profiler root")
    parser.add_argument("--single-rank-step-ms", type=float, help="single-rank avg step time in ms")
    parser.add_argument("--output-json", required=True, help="output JSON path")
    args = parser.parse_args()

    trace_root = Path(args.trace_root).resolve()

    # Find rank directories
    rank_dirs = find_rank_dirs(trace_root)

    # If single-rank data, use it as baseline
    if not rank_dirs:
        step_csv = find_step_trace_csv(trace_root)
        if step_csv:
            single_step = avg_step_time_from_csv(step_csv)
            if single_step:
                report = {
                    "schema_version": "performance-agent/0.1",
                    "skill": "performance-agent",
                    "linearity": None,
                    "note": "Single-rank data: linearity requires multi-rank comparison",
                    "single_rank_avg_step_ms": round(single_step, 3),
                }
                write_json(Path(args.output_json), report)
                print(json.dumps({"linearity": None, "note": "single rank only"}))
                return 0
        print("No rank directories or step data found", file=sys.stderr)
        raise SystemExit(1)

    # Load per-rank step times
    rank_step_times: dict[int, float] = {}
    for rank_id, rank_dir in rank_dirs.items():
        step_csv = find_step_trace_csv(rank_dir)
        if not step_csv:
            continue
        avg = avg_step_time_from_csv(step_csv)
        if avg is not None:
            rank_step_times[rank_id] = avg

    if len(rank_step_times) < 2:
        report = {
            "schema_version": "performance-agent/0.1",
            "skill": "performance-agent",
            "linearity": None,
            "note": "Need at least 2 ranks with step data for linearity calculation",
            "ranks_with_data": len(rank_step_times),
        }
        write_json(Path(args.output_json), report)
        print(json.dumps({"linearity": None}))
        return 0

    # Single-rank baseline: use fastest rank as "ideal single device"
    # (fastest rank is closest to theoretical single-device performance)
    fastest_rank = min(rank_step_times, key=lambda r: rank_step_times[r])
    fastest_step_ms = rank_step_times[fastest_rank]

    # If user provided single-rank baseline, use that instead
    if args.single_rank_step_ms:
        fastest_step_ms = args.single_rank_step_ms

    # Cluster step time: average across all ranks
    cluster_step_ms = sum(rank_step_times.values()) / len(rank_step_times)

    num_devices = len(rank_step_times)
    linearity = calculate_linearity(fastest_step_ms, cluster_step_ms)

    # Classify
    if linearity >= 0.8:
        status = "normal"
        diagnosis = "Communication does NOT appear to be the bottleneck (linearity >= 0.8)"
    elif linearity >= 0.6:
        status = "moderate_degradation"
        diagnosis = "Moderate scaling degradation detected — investigate communication overhead"
    else:
        status = "severe_degradation"
        diagnosis = "Severe scaling degradation — communication IS the bottleneck (after ruling out IO/CPU factors)"

    # Load profiler info if available
    profiler_info = None
    for info_file in sorted(trace_root.glob("profiler_info_*.json"))[:1]:
        try:
            profiler_info = read_json(info_file)
        except Exception:
            break

    batch_config = _infer_batch_config(profiler_info)

    report = {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "linearity": round(linearity, 4),
        "linearity_status": status,
        "num_devices": num_devices,
        "fastest_rank": fastest_rank,
        "fastest_step_ms": round(fastest_step_ms, 3),
        "cluster_avg_step_ms": round(cluster_step_ms, 3),
        "theoretical_speedup": num_devices,
        "actual_speedup": round(linearity * num_devices, 2),
        "efficiency_loss_percent": round((1.0 - linearity) * 100, 1),
        "batch_config": batch_config,
        "rank_step_times_ms": {str(k): round(v, 3) for k, v in sorted(rank_step_times.items())},
        "diagnosis": diagnosis,
        "threshold": 0.8,
        "likely_domains": ["communication"] if linearity < 0.8 else [],
        "recommended_actions": (
            [
                "Check communication matrix for slow links (use analyze_communication_matrix.py)",
                "Verify HCCL buffer sizing: HCCL_BUFFSIZE = ceil(MBS × S × H × dtype / 8MB)",
                "Check if communication is overlapped with computation",
                "Verify TP/PP/DP strategy is appropriate for the model and cluster size",
            ]
            if linearity < 0.8
            else []
        ),
    }

    write_json(Path(args.output_json), report)
    print(json.dumps({
        "linearity": round(linearity, 4),
        "status": status,
        "num_devices": num_devices,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
