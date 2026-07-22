#!/usr/bin/env python3
"""Async pipeline orchestrator for performance-agent analysis scripts.

Runs independent analysis scripts concurrently in dependency waves,
collecting all intermediate JSON artifacts.
"""
import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Optional

from perf_common import write_json


SCRIPT_DIR = Path(__file__).resolve().parent

# Wave definitions: each wave contains scripts that can run in parallel.
# Scripts within a wave are independent; waves run sequentially.
WAVE_DEFS = [
    {
        "wave": 1,
        "scripts": [
            {"name": "validate_profiler_data.py", "required_args": ["--trace-root"]},
            {"name": "find_run_context.py", "required_args": ["--working-dir"]},
            {"name": "locate_profiler_output.py", "required_args": ["--trace-path"]},
        ],
    },
    {
        "wave": 2,
        "scripts": [
            {"name": "summarize_step_breakdown.py", "required_args": ["--trace-root"]},
            {"name": "summarize_communication.py", "required_args": ["--trace-root"]},
            {"name": "summarize_memory_pressure.py", "required_args": ["--trace-root"]},
            {"name": "summarize_input_pipeline.py", "required_args": ["--trace-root"]},
            {"name": "summarize_trace_gaps.py", "required_args": ["--trace-root"]},
            {"name": "detect_bound_type.py", "required_args": ["--trace-root"]},
            {"name": "summarize_msprof_hotspots.py", "required_args": ["--input-dir"]},
            {"name": "summarize_aic_metrics.py", "required_args": ["--trace-root"]},
        ],
    },
    {
        "wave": 3,
        "scripts": [
            {"name": "analyze_collective_types.py", "required_args": ["--trace-root"]},
            {"name": "analyze_rank_variance.py", "required_args": ["--trace-root"]},
        ],
    },
    {
        "wave": 4,
        "scripts": [
            {"name": "detect_slow_ranks.py", "required_args": ["--trace-root"]},
            {"name": "calculate_linearity.py", "required_args": ["--trace-root"]},
            {"name": "analyze_jitter.py", "required_args": ["--step-json"]},
            {"name": "calculate_mfu.py", "required_args": ["--trace-root"]},
            {"name": "recommend_parallel_strategy.py", "required_args": []},
        ],
    },
    {
        "wave": 5,
        "scripts": [
            {"name": "correlate_slow_rank_ops.py", "required_args": ["--trace-root"]},
            {"name": "attribute_wait_times.py", "required_args": []},
        ],
    },
    {
        "wave": 6,
        "scripts": [
            {"name": "correlate_host_device.py", "required_args": []},
            {"name": "analyze_operator_fusion.py", "required_args": []},
            {"name": "classify_cluster_degradation.py", "required_args": []},
        ],
    },
    {
        "wave": 7,
        "scripts": [
            {"name": "analyze_npu_affinity.py", "required_args": []},
            {"name": "build_performance_profile.py", "required_args": ["--working-dir"]},
        ],
    },
    {
        "wave": 8,
        "scripts": [
            {"name": "classify_bottlenecks.py", "required_args": ["--profile-json"]},
        ],
    },
    {
        "wave": 9,
        "scripts": [
            {"name": "build_causal_chain.py", "required_args": ["--bottlenecks-json"]},
            {"name": "build_optimization_suggestions.py", "required_args": ["--profile-json", "--bottlenecks-json"]},
            {"name": "infer_root_cause.py", "required_args": ["--bottlenecks-json"]},
        ],
    },
    {
        "wave": 10,
        "scripts": [
            {"name": "compare_validation_metrics.py", "required_args": ["--before-json", "--after-json"]},
        ],
    },
]


_SCRIPT_ARTIFACT_NAMES: dict[str, str] = {
    "validate_profiler_data.py": "validation",
    "find_run_context.py": "context",
    "locate_profiler_output.py": "profiler_location",
    "summarize_step_breakdown.py": "step",
    "summarize_communication.py": "communication",
    "summarize_memory_pressure.py": "memory",
    "summarize_input_pipeline.py": "input",
    "summarize_trace_gaps.py": "trace_gaps",
    "summarize_msprof_hotspots.py": "hotspot",
    "summarize_aic_metrics.py": "aic_metrics",
    "detect_slow_ranks.py": "slow_ranks",
    "detect_bound_type.py": "bound_type",
    "calculate_linearity.py": "linearity",
    "calculate_mfu.py": "mfu",
    "analyze_jitter.py": "jitter",
    "recommend_parallel_strategy.py": "parallel_strategy",
    "correlate_host_device.py": "host_device",
    "analyze_operator_fusion.py": "fusion",
    "classify_cluster_degradation.py": "degradation",
    "analyze_npu_affinity.py": "affinity",
    "build_performance_profile.py": "profile",
    "classify_bottlenecks.py": "bottlenecks",
    "infer_root_cause.py": "root_cause",
    "build_optimization_suggestions.py": "suggestions",
    "compare_validation_metrics.py": "comparison",
    "compare_profiling_runs.py": "profiling_comparison",
    "build_performance_report.py": "report",
    "build_hotspot_brief.py": "hotspot_brief",
    "analyze_collective_types.py": "collective_types",
    "analyze_rank_variance.py": "rank_variance",
    "correlate_slow_rank_ops.py": "slow_rank_ops",
    "attribute_wait_times.py": "wait_times",
    "build_causal_chain.py": "causal_chain",
}


def _output_name(script_name: str) -> str:
    """Derive an artifact key from the script filename.

    Uses an explicit mapping to avoid fragile prefix-stripping heuristics.
    """
    return _SCRIPT_ARTIFACT_NAMES.get(script_name, script_name.replace(".py", ""))


def _build_args_for_script(
    script: dict,
    trace_root: Optional[str],
    working_dir: str,
    user_problem: str,
    output_dir: Path,
    hardware: Optional[str],
    model_config: Optional[str],
    before_metrics: Optional[str],
    after_metrics: Optional[str],
) -> list[str]:
    """Build the argument list for a script invocation."""
    name = script["name"]
    args: list[str] = []

    # Common output path
    artifact_key = _output_name(name)
    output_json = output_dir / f"{artifact_key}.json"
    args.extend(["--output-json", str(output_json)])

    # Script-specific arguments
    if "--trace-root" in script["required_args"]:
        if trace_root:
            args.extend(["--trace-root", trace_root])
    if "--trace-path" in script["required_args"]:
        if trace_root:
            args.extend(["--trace-path", trace_root])
    if "--working-dir" in script["required_args"]:
        args.extend(["--working-dir", working_dir])
    if "--input-dir" in script["required_args"]:
        if trace_root:
            args.extend(["--input-dir", trace_root])
    if "--step-json" in script["required_args"]:
        step_json = output_dir / "step_breakdown.json"
        args.extend(["--step-json", str(step_json)])

    # --- Helper: map artifact filename → CLI flag for feeding summaries ---
    # artifact_file is the filename in output_dir (from _output_name).
    # Maps to the downstream script's CLI --flag.
    def _feed_artifacts(mappings: list[tuple[str, str]]) -> None:
        """Append --flag <path> for each existing artifact."""
        for artifact_file, flag in mappings:
            candidate = output_dir / f"{artifact_file}.json"
            if candidate.exists():
                args.extend([flag, str(candidate)])

    # profile-json: feed profile path + summaries accepted by most scripts
    if "--profile-json" in script["required_args"]:
        profile_json = output_dir / "performance_profile.json"
        args.extend(["--profile-json", str(profile_json)])
        _feed_artifacts([
            ("step_breakdown", "--step-json"),
            ("communication", "--communication-json"),
            ("memory_pressure", "--memory-json"),
            ("input_pipeline", "--input-json"),
            ("trace_gaps", "--trace-gaps-json"),
            ("msprof_hotspots", "--hotspot-json"),
            ("mfu", "--mfu-json"),
            ("slow_ranks", "--cluster-json"),
            ("jitter", "--jitter-json"),
        ])

    # bottlenecks-json: just the path (summaries come from profile-json block)
    if "--bottlenecks-json" in script["required_args"]:
        bottlenecks_json = output_dir / "bottlenecks.json"
        args.extend(["--bottlenecks-json", str(bottlenecks_json)])

    if "--before-json" in script["required_args"] and before_metrics:
        args.extend(["--before-json", before_metrics])
    if "--after-json" in script["required_args"] and after_metrics:
        args.extend(["--after-json", after_metrics])

    # --- Per-script special args ---

    # Model config for MFU
    if name == "calculate_mfu.py" and model_config:
        args.extend(["--model-config", model_config])
    if name == "calculate_mfu.py" and trace_root:
        if "--trace-root" not in script["required_args"]:
            args.extend(["--trace-root", trace_root])
    if name == "calculate_mfu.py" and hardware:
        args.extend(["--hardware", hardware])

    # Hotspot needs output-md as well
    if name == "summarize_msprof_hotspots.py":
        args.extend(["--output-md", str(output_dir / "hotspot_summary.md")])

    # Recommend parallel strategy needs model config
    if name == "recommend_parallel_strategy.py":
        if model_config:
            args.extend(["--model-config", model_config])
        if trace_root:
            args.extend(["--trace-root", trace_root])

    # build_performance_profile: needs all wave 1-4 summaries
    if name == "build_performance_profile.py":
        args.extend(["--user-problem", user_problem])
        _feed_artifacts([
            ("profiler_output", "--locate-json"),
            ("step_breakdown", "--step-json"),
            ("communication", "--communication-json"),
            ("memory_pressure", "--memory-json"),
            ("input_pipeline", "--input-json"),
            ("trace_gaps", "--trace-gaps-json"),
            ("msprof_hotspots", "--hotspot-json"),
            ("profiler_data", "--validation-json"),
            ("mfu", "--mfu-json"),
            ("slow_ranks", "--cluster-json"),
            ("jitter", "--jitter-json"),
            ("aic_metrics", "--aic-json"),
        ])
        if hardware:
            args.extend(["--hardware", hardware])

    # correlate_host_device: resolve trace_view.json and kernel_details.csv from trace_root
    if name == "correlate_host_device.py":
        if trace_root:
            trace_root_path = Path(trace_root)
            # Check top-level first
            candidate = trace_root_path / "trace_view.json"
            if candidate.exists():
                args.extend(["--trace-view-json", str(candidate)])
            # Also check common subdirectory patterns
            if "--trace-view-json" not in args:
                for subdir in ["ASCEND_PROFILER_OUTPUT", "profiler"]:
                    candidate = trace_root_path / subdir / "trace_view.json"
                    if candidate.exists():
                        args.extend(["--trace-view-json", str(candidate)])
                        break
            kernel_candidate = trace_root_path / "kernel_details.csv"
            if not kernel_candidate.exists():
                for subdir in ["ASCEND_PROFILER_OUTPUT", "profiler"]:
                    kernel_candidate = trace_root_path / subdir / "kernel_details.csv"
                    if kernel_candidate.exists():
                        break
            if kernel_candidate.exists():
                args.extend(["--kernel-details-csv", str(kernel_candidate)])

    # analyze_operator_fusion: needs hotspot, optional kernel_csv, comm, step
    if name == "analyze_operator_fusion.py":
        _feed_artifacts([
            ("msprof_hotspots", "--hotspot-json"),
            ("communication", "--communication-json"),
            ("step_breakdown", "--step-json"),
        ])
        if trace_root:
            for subdir in ["", "ASCEND_PROFILER_OUTPUT", "profiler"]:
                kernel_candidate = Path(trace_root) / subdir / "kernel_details.csv"
                if kernel_candidate.exists():
                    args.extend(["--kernel-details-csv", str(kernel_candidate)])
                    break

    # classify_cluster_degradation: needs slow_ranks, jitter, step, comm, linearity
    if name == "classify_cluster_degradation.py":
        _feed_artifacts([
            ("slow_ranks", "--cluster-json"),
            ("jitter", "--jitter-json"),
            ("step_breakdown", "--step-json"),
            ("communication", "--communication-json"),
            ("linearity", "--linearity-json"),
        ])

    # analyze_npu_affinity: needs hotspot, comm, trace_gaps, step, slow_ranks, host_device, bound_type
    if name == "analyze_npu_affinity.py":
        _feed_artifacts([
            ("msprof_hotspots", "--hotspot-json"),
            ("communication", "--communication-json"),
            ("trace_gaps", "--trace-gaps-json"),
            ("step_breakdown", "--step-json"),
            ("slow_ranks", "--cluster-json"),
            ("correlate_host_device", "--host-device-json"),
            ("bound_type", "--bound-type-json"),
        ])

    # infer_root_cause: needs bottlenecks (from required_args), profile, and summaries
    if name == "infer_root_cause.py":
        _feed_artifacts([
            ("performance_profile", "--profile-json"),
            ("step_breakdown", "--step-json"),
            ("communication", "--communication-json"),
            ("jitter", "--jitter-json"),
            ("slow_ranks", "--cluster-json"),
            ("mfu", "--mfu-json"),
        ])

    # correlate_slow_rank_ops: needs rank_variance and cluster
    if name == "correlate_slow_rank_ops.py":
        _feed_artifacts([
            ("rank_variance", "--rank-variance-json"),
            ("slow_ranks", "--cluster-json"),
        ])

    # attribute_wait_times: needs collective_types, rank_variance, cluster, step, comm
    if name == "attribute_wait_times.py":
        _feed_artifacts([
            ("collective_types", "--collective-types-json"),
            ("rank_variance", "--rank-variance-json"),
            ("slow_ranks", "--cluster-json"),
            ("step_breakdown", "--step-json"),
            ("communication", "--communication-json"),
        ])

    # build_causal_chain: needs bottlenecks + all deep analysis artifacts
    if name == "build_causal_chain.py":
        _feed_artifacts([
            ("performance_profile", "--profile-json"),
            ("collective_types", "--collective-types-json"),
            ("rank_variance", "--rank-variance-json"),
            ("slow_rank_ops", "--slow-rank-ops-json"),
            ("wait_times", "--wait-attribution-json"),
            ("step_breakdown", "--step-json"),
            ("communication", "--communication-json"),
            ("jitter", "--jitter-json"),
            ("slow_ranks", "--cluster-json"),
            ("mfu", "--mfu-json"),
        ])

    # classify_bottlenecks: also pass fusion/degradation/affinity from wave 6
    if name == "classify_bottlenecks.py":
        _feed_artifacts([
            ("operator_fusion", "--fusion-json"),
            ("cluster_degradation", "--degradation-json"),
            ("npu_affinity", "--affinity-json"),
            ("collective_types", "--collective-types-json"),
            ("rank_variance", "--rank-variance-json"),
        ])

    # build_optimization_suggestions: also pass fusion/degradation/affinity + deep analysis
    if name == "build_optimization_suggestions.py":
        _feed_artifacts([
            ("operator_fusion", "--fusion-json"),
            ("cluster_degradation", "--degradation-json"),
            ("npu_affinity", "--affinity-json"),
            ("collective_types", "--collective-types-json"),
            ("rank_variance", "--rank-variance-json"),
            ("wait_times", "--wait-attribution-json"),
        ])

    return args


async def _run_script(
    script_name: str,
    extra_args: list[str],
) -> dict:
    """Run a single script as an async subprocess."""
    script_path = SCRIPT_DIR / script_name
    cmd = ["python", str(script_path)] + extra_args
    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if proc.returncode != 0:
            return {
                "script": script_name,
                "status": "failed",
                "returncode": proc.returncode,
                "stderr": stderr.decode("utf-8", errors="replace")[:500],
                "duration_ms": elapsed_ms,
            }
        return {
            "script": script_name,
            "status": "completed",
            "duration_ms": elapsed_ms,
            "stdout_preview": stdout.decode("utf-8", errors="replace")[:200],
        }
    except asyncio.TimeoutError:
        return {
            "script": script_name,
            "status": "timeout",
            "duration_ms": 120000,
        }
    except Exception as exc:
        return {
            "script": script_name,
            "status": "error",
            "error": str(exc),
        }


async def run_wave(
    wave_def: dict,
    trace_root: Optional[str],
    working_dir: str,
    user_problem: str,
    output_dir: Path,
    hardware: Optional[str],
    model_config: Optional[str],
    before_metrics: Optional[str],
    after_metrics: Optional[str],
    skip_waves: set[int],
    dry_run: bool,
) -> dict:
    """Run all scripts in a wave concurrently."""
    wave_num = wave_def["wave"]
    if wave_num in skip_waves:
        return {"wave": wave_num, "status": "skipped", "scripts": [], "duration_ms": 0}

    tasks = []
    script_names = []
    for script in wave_def["scripts"]:
        args = _build_args_for_script(
            script, trace_root, working_dir, user_problem, output_dir,
            hardware, model_config, before_metrics, after_metrics,
        )
        script_names.append(script["name"])
        if dry_run:
            tasks.append(None)
        else:
            tasks.append(_run_script(script["name"], args))

    if dry_run:
        return {
            "wave": wave_num,
            "status": "dry_run",
            "scripts": [{"script": n, "status": "planned"} for n in script_names],
            "duration_ms": 0,
        }

    start = time.monotonic()
    results = await asyncio.gather(*[t for t in tasks if t is not None])
    elapsed_ms = int((time.monotonic() - start) * 1000)

    wave_status = "completed"
    if any(r["status"] == "failed" for r in results):
        wave_status = "partial_failure"
    if all(r["status"] in ("failed", "error", "timeout") for r in results):
        wave_status = "failed"

    return {
        "wave": wave_num,
        "status": wave_status,
        "scripts": list(results),
        "duration_ms": elapsed_ms,
    }


async def run_pipeline(
    trace_root: Optional[str],
    working_dir: str,
    user_problem: str,
    output_dir: Path,
    hardware: Optional[str],
    model_config: Optional[str],
    before_metrics: Optional[str],
    after_metrics: Optional[str],
    skip_waves: set[int],
    dry_run: bool,
) -> dict:
    """Run the full analysis pipeline across all waves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    wave_results = []
    total_start = time.monotonic()

    for wave_def in WAVE_DEFS:
        result = await run_wave(
            wave_def, trace_root, working_dir, user_problem, output_dir,
            hardware, model_config, before_metrics, after_metrics,
            skip_waves, dry_run,
        )
        wave_results.append(result)
        # Stop pipeline if a critical wave fails
        if result["status"] == "failed" and wave_def["wave"] <= 2:
            break

    total_ms = int((time.monotonic() - total_start) * 1000)

    # Collect artifacts
    artifacts: dict[str, str] = {}
    for json_file in sorted(output_dir.glob("*.json")):
        artifacts[json_file.stem] = str(json_file)

    overall = "completed"
    statuses = {w["status"] for w in wave_results}
    if "dry_run" in statuses:
        overall = "dry_run"
    elif "failed" in statuses or "partial_failure" in statuses:
        overall = "partial_failure"
    elif all(s in ("skipped", "failed") for s in statuses):
        overall = "failed"

    return {
        "schema_version": "performance-agent/0.1",
        "skill": "performance-agent",
        "overall_status": overall,
        "total_duration_ms": total_ms,
        "waves": wave_results,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the performance-agent analysis pipeline with concurrent execution"
    )
    parser.add_argument("--trace-root", help="Profiler output root directory")
    parser.add_argument("--working-dir", default=".", help="Workspace root")
    parser.add_argument("--user-problem", default="", help="Performance symptom description")
    parser.add_argument("--output-dir", required=True, help="Directory for all output artifacts")
    parser.add_argument("--hardware", help="Hardware model override (e.g. ascend_910b1)")
    parser.add_argument("--model-config", help="Model config JSON for MFU calculation")
    parser.add_argument("--before-metrics", help="Before-change metrics JSON path")
    parser.add_argument("--after-metrics", help="After-change metrics JSON path")
    parser.add_argument("--skip-waves", default="", help="Comma-separated wave numbers to skip (e.g. 1,5)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned execution without running")
    args = parser.parse_args()

    skip = set()
    if args.skip_waves:
        skip = {int(w.strip()) for w in args.skip_waves.split(",") if w.strip().isdigit()}

    result = asyncio.run(run_pipeline(
        trace_root=args.trace_root,
        working_dir=args.working_dir,
        user_problem=args.user_problem,
        output_dir=Path(args.output_dir),
        hardware=args.hardware,
        model_config=args.model_config,
        before_metrics=args.before_metrics,
        after_metrics=args.after_metrics,
        skip_waves=skip,
        dry_run=args.dry_run,
    ))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "pipeline_result.json", result)

    print(json.dumps({
        "overall_status": result["overall_status"],
        "total_duration_ms": result["total_duration_ms"],
        "artifacts_count": len(result["artifacts"]),
    }, indent=2))
    return 0 if result["overall_status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
