"""Unit tests for build_optimization_suggestions.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, run_script, ROOT, SCRIPTS


def _run_full_pipeline(tmp_path: Path):
    """Run pipeline up to bottlenecks and return all intermediate paths."""
    profiler_root = write_sample_profiler_export(tmp_path)
    names = ["locate", "step", "comm", "memory", "input", "trace_gaps", "hotspot", "profile", "bottlenecks"]
    paths = {n: tmp_path / f"{n}.json" for n in names}

    run_script("locate_profiler_output.py", "--trace-path", str(profiler_root), "--output-json", str(paths["locate"]))
    run_script("summarize_step_breakdown.py", "--trace-root", str(profiler_root), "--output-json", str(paths["step"]))
    run_script("summarize_communication.py", "--trace-root", str(profiler_root), "--output-json", str(paths["comm"]))
    run_script("summarize_memory_pressure.py", "--trace-root", str(profiler_root), "--output-json", str(paths["memory"]))
    run_script("summarize_input_pipeline.py", "--trace-root", str(profiler_root), "--output-json", str(paths["input"]))
    run_script("summarize_trace_gaps.py", "--trace-root", str(profiler_root), "--output-json", str(paths["trace_gaps"]))
    run_script("summarize_msprof_hotspots.py", "--input-dir", str(profiler_root),
                "--output-md", str(tmp_path / "hotspot.md"), "--output-json", str(paths["hotspot"]))
    run_script("build_performance_profile.py", "--working-dir", str(tmp_path),
                "--user-problem", "training is slow and allreduce dominates",
                "--locate-json", str(paths["locate"]), "--step-json", str(paths["step"]),
                "--communication-json", str(paths["comm"]), "--memory-json", str(paths["memory"]),
                "--input-json", str(paths["input"]), "--trace-gaps-json", str(paths["trace_gaps"]),
                "--hotspot-json", str(paths["hotspot"]), "--output-json", str(paths["profile"]))
    run_script("classify_bottlenecks.py", "--profile-json", str(paths["profile"]),
                "--step-json", str(paths["step"]), "--communication-json", str(paths["comm"]),
                "--memory-json", str(paths["memory"]), "--input-json", str(paths["input"]),
                "--trace-gaps-json", str(paths["trace_gaps"]), "--hotspot-json", str(paths["hotspot"]),
                "--output-json", str(paths["bottlenecks"]))
    return paths


def test_suggestions_generated_for_communication_bottleneck(tmp_path: Path):
    paths = _run_full_pipeline(tmp_path)
    suggestions_json = tmp_path / "suggestions.json"

    run_script("build_optimization_suggestions.py",
               "--profile-json", str(paths["profile"]),
               "--bottlenecks-json", str(paths["bottlenecks"]),
               "--step-json", str(paths["step"]),
               "--communication-json", str(paths["comm"]),
               "--memory-json", str(paths["memory"]),
               "--input-json", str(paths["input"]),
               "--trace-gaps-json", str(paths["trace_gaps"]),
               "--hotspot-json", str(paths["hotspot"]),
               "--output-json", str(suggestions_json))

    result = json.loads(suggestions_json.read_text(encoding="utf-8"))
    assert result["suggestion_summary"]["total_count"] > 0
    ids = [s["id"] for s in result["suggestions"]]
    assert "COMM-01" in ids
    for s in result["suggestions"]:
        assert "id" in s and "title" in s and "priority" in s and "actions" in s


def test_suggestions_sorted_by_priority(tmp_path: Path):
    paths = _run_full_pipeline(tmp_path)
    suggestions_json = tmp_path / "suggestions.json"

    run_script("build_optimization_suggestions.py",
               "--profile-json", str(paths["profile"]),
               "--bottlenecks-json", str(paths["bottlenecks"]),
               "--step-json", str(paths["step"]),
               "--communication-json", str(paths["comm"]),
               "--memory-json", str(paths["memory"]),
               "--input-json", str(paths["input"]),
               "--trace-gaps-json", str(paths["trace_gaps"]),
               "--hotspot-json", str(paths["hotspot"]),
               "--output-json", str(suggestions_json))

    result = json.loads(suggestions_json.read_text(encoding="utf-8"))
    priorities = [s["priority"] for s in result["suggestions"]]
    order = {"high": 0, "medium": 1, "low": 2}
    numeric = [order[p] for p in priorities]
    assert numeric == sorted(numeric)


def test_input_bottleneck_generates_input_suggestion(tmp_path: Path):
    paths = _run_full_pipeline(tmp_path)
    suggestions_json = tmp_path / "suggestions.json"

    run_script("build_optimization_suggestions.py",
               "--profile-json", str(paths["profile"]),
               "--bottlenecks-json", str(paths["bottlenecks"]),
               "--input-json", str(paths["input"]),
               "--output-json", str(suggestions_json))

    result = json.loads(suggestions_json.read_text(encoding="utf-8"))
    ids = [s["id"] for s in result["suggestions"]]
    assert "INPUT-01" in ids
