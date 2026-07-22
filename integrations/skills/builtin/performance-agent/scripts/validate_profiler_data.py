#!/usr/bin/env python3
"""Validate profiler data integrity before performance analysis.

Checks collection status, config, parse status, and key deliverable presence.
Returns exit code: 0=valid, 1=invalid(needs recollection), 2=unparsed(needs parsing).

Supports three profiler types:
- framework_profiler_pt: PyTorch with Ascend NPU ([*]_ascend_pt)
- framework_profiler_ms: MindSpore with Ascend NPU ([*]_ascend_ms)
- msprof: Command-line profiler (PROF_{} directories)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from perf_common import write_json

# Export types for framework profiler deliverable checking
_TEXT_DELIVERABLES = ["trace_view.json", "kernel_details.csv", "step_trace_time.csv"]
_DB_DELIVERABLES = ["profiler_db"]


def detect_data_type(profiler_path: Path, _depth: int = 0) -> Optional[str]:
    """Detect profiler data type from the top-level path.

    Type-binding principle: once detected from the top-level path, only that
    type's rules apply throughout the validation.
    """
    if _depth > 3:
        return None
    name = profiler_path.name.lower()
    if name.endswith("_ascend_pt") or "_ascend_pt_" in name:
        return "framework_profiler_pt"
    if name.endswith("_ascend_ms") or "_ascend_ms_" in name:
        return "framework_profiler_ms"
    # Check for PROF_{} directory pattern (msprof)
    if profiler_path.name.startswith("prof_"):
        return "msprof"
    if (profiler_path / "ASCEND_PROFILER_OUTPUT").exists():
        # Framework profiler output detected by content
        subdirs = [p for p in profiler_path.iterdir() if p.is_dir()]
        for sub in subdirs:
            if sub.name.lower().endswith("_ascend_pt"):
                return "framework_profiler_pt"
            if sub.name.lower().endswith("_ascend_ms"):
                return "framework_profiler_ms"
        # If ASCEND_PROFILER_OUTPUT exists at root, treat as framework profiler
        return "framework_profiler"
    if any(profiler_path.glob("PROF_*")):
        return "msprof"
    # Check if it's a cluster directory containing multiple profiler dirs
    subdirs = [p for p in profiler_path.iterdir() if p.is_dir()]
    types_found = set()
    for sub in subdirs:
        sub_type = detect_data_type(sub, _depth + 1)
        if sub_type:
            types_found.add(sub_type)
    if types_found:
        return f"cluster_{types_found.pop()}"
    return None


def _is_framework(data_type: Optional[str]) -> bool:
    """Check whether a data type is any framework profiler variant."""
    if not data_type:
        return False
    return (
        data_type.startswith("framework_profiler")
        or data_type.startswith("cluster_framework_profiler")
    )


def _is_msprof(data_type: Optional[str]) -> bool:
    """Check whether a data type is msprof (or cluster of msprof)."""
    if not data_type:
        return False
    return data_type == "msprof" or data_type.startswith("cluster_msprof")


# ---------------------------------------------------------------------------
# Step 2: Stop check
# ---------------------------------------------------------------------------

def check_stop_framework(profiler_path: Path) -> tuple[bool, list[str]]:
    """Check if framework profiler collection completed normally.

    Multi-card spot check: only checks the first profiler_info file found
    (typically Rank 0), not every rank.
    """
    issues = []
    info_files = list(profiler_path.glob("profiler_info.json"))
    info_files.extend(profiler_path.glob("profiler_info_*.json"))
    if not info_files:
        issues.append("profiler_info.json not found; collection may not have completed normally")
        return False, issues
    # Spot check: verify only the first info file has content
    for info_file in info_files[:1]:
        try:
            data = json.loads(info_file.read_text(encoding="utf-8"))
            if not data:
                issues.append(f"{info_file.name} is empty")
                return False, issues
        except Exception as e:
            issues.append(f"Failed to read {info_file.name}: {e}")
            return False, issues
    return True, issues


def check_stop_msprof(profiler_path: Path) -> tuple[bool, list[str]]:
    """Check if msprof collection completed normally.

    Multi-card spot check: only checks the first PROF_ directory.
    """
    issues = []
    prof_dirs = [profiler_path]
    if not profiler_path.name.startswith("PROF_"):
        prof_dirs = [p for p in profiler_path.iterdir() if p.is_dir() and p.name.startswith("PROF_")]
        if not prof_dirs:
            issues.append("No PROF_* directory found")
            return False, issues

    # Spot check first PROF_ directory only
    prof_dir = prof_dirs[0]
    device_dirs = [p for p in prof_dir.iterdir() if p.is_dir() and p.name.startswith("device_")]
    if not device_dirs:
        issues.append(f"No device_* directory in {prof_dir.name}")
        return False, issues
    device_dir = device_dirs[0]
    end_info_files = list(device_dir.glob("end_info.*"))
    if not end_info_files:
        issues.append(f"end_info not found in {device_dir.name}; collection may not have completed normally")
        return False, issues

    # Check remaining dirs for partial failures (non-blocking)
    failed_dirs: list[str] = []
    for pd in prof_dirs[1:]:
        dd = [p for p in pd.iterdir() if p.is_dir() and p.name.startswith("device_")]
        if not dd:
            failed_dirs.append(f"No device_* directory in {pd.name}")
            continue
        if not list(dd[0].glob("end_info.*")):
            failed_dirs.append(f"end_info not found in {dd[0].name}")

    if failed_dirs:
        issues.extend([f"Partial: {msg}" for msg in failed_dirs])

    return True, issues


# ---------------------------------------------------------------------------
# Step 3: Config parsing (framework profiler only)
# ---------------------------------------------------------------------------

def parse_profiler_config(profiler_path: Path) -> dict:
    """Parse profiler configuration from profiler_info.json.

    Only applicable for framework profiler data types. Returns config dict
    with: profiler_level, with_stack, with_modules, record_shapes,
    profile_memory, step_count, and any other available fields.
    """
    config: dict = {"config_available": False}

    info_files = list(profiler_path.glob("profiler_info.json"))
    info_files.extend(profiler_path.glob("profiler_info_*.json"))
    if not info_files:
        return config

    try:
        data = json.loads(info_files[0].read_text(encoding="utf-8"))
    except Exception:
        return config

    config["config_available"] = True

    # Extract known configuration fields
    _KEYS = [
        "profiler_level", "with_stack", "with_modules",
        "record_shapes", "profile_memory", "export_type",
        "activities", "schedule_wait", "step_count",
    ]
    for key in _KEYS:
        if key in data:
            config[key] = data[key]

    return config


# ---------------------------------------------------------------------------
# Step 4: Parse check
# ---------------------------------------------------------------------------

def check_parse_framework(profiler_path: Path) -> tuple[bool, list[str]]:
    """Check if framework profiler data has been parsed.

    Looks for ASCEND_PROFILER_OUTPUT directory with content.
    """
    issues = []
    output_dir = profiler_path / "ASCEND_PROFILER_OUTPUT"
    if not output_dir.exists():
        issues.append("ASCEND_PROFILER_OUTPUT directory not found; data may not be parsed")
        return False, issues
    if not any(output_dir.iterdir()):
        issues.append("ASCEND_PROFILER_OUTPUT is empty; data may not be parsed")
        return False, issues
    return True, issues


def check_parse_msprof(profiler_path: Path) -> tuple[bool, list[str]]:
    """Check if msprof data has been parsed/exported.

    Looks for mindstudio_profiler_output directory (result of msprof --export=on).
    """
    issues = []
    prof_dirs = [profiler_path]
    if not profiler_path.name.startswith("PROF_"):
        prof_dirs = [p for p in profiler_path.iterdir() if p.is_dir() and p.name.startswith("PROF_")]
        if not prof_dirs:
            issues.append("No PROF_* directory found")
            return False, issues

    for prof_dir in prof_dirs[:1]:
        ms_output = prof_dir / "mindstudio_profiler_output"
        if not ms_output.exists():
            issues.append(f"mindstudio_profiler_output not found in {prof_dir.name}; run 'msprof --export=on'")
            return False, issues
        if not any(ms_output.iterdir()):
            issues.append(f"mindstudio_profiler_output is empty in {prof_dir.name}")
            return False, issues
    return True, issues


# ---------------------------------------------------------------------------
# Step 5: Key deliverables
# ---------------------------------------------------------------------------

def detect_export_type(profiler_path: Path) -> str:
    """Detect whether framework profiler used Text or DB export mode.

    Returns 'text', 'db', or 'unknown'.
    """
    base = profiler_path / "ASCEND_PROFILER_OUTPUT"
    has_text = (base / "trace_view.json").exists() or (base / "kernel_details.csv").exists()
    db_files = list(profiler_path.glob("*_profiler_*.db"))
    has_db = len(db_files) > 0

    if has_text and has_db:
        return "both"
    if has_text:
        return "text"
    if has_db:
        return "db"
    return "unknown"


def check_key_deliverables(profiler_path: Path, data_type: str) -> dict[str, bool]:
    """Check presence of key analysis files.

    For framework profiler, checks based on export_type:
    - Text mode: trace_view.json, kernel_details.csv, step_trace_time.csv
    - DB mode: *_profiler_*.db files
    For msprof, checks for msprof_*.db or msprof_{timestamp}.json/op_summary.csv.
    """
    files: dict[str, bool] = {}

    if _is_framework(data_type):
        base = profiler_path / "ASCEND_PROFILER_OUTPUT"
        export_type = detect_export_type(profiler_path)
        files["export_type"] = export_type  # type: ignore[assignment]

        # Text mode deliverables
        files["step_trace_time.csv"] = (base / "step_trace_time.csv").exists()
        files["kernel_details.csv"] = (base / "kernel_details.csv").exists()
        files["trace_view.json"] = (base / "trace_view.json").exists()
        files["communication.json"] = (base / "communication.json").exists()
        files["memory_record.csv"] = (base / "memory_record.csv").exists()
        files["operator_memory.csv"] = (base / "operator_memory.csv").exists()
        files["dataset.csv"] = (base / "dataset.csv").exists()

        # DB mode deliverables
        db_files = list(profiler_path.glob("*_profiler_*.db"))
        files["profiler_db"] = len(db_files) > 0

    elif _is_msprof(data_type):
        prof_dirs = [profiler_path]
        if not profiler_path.name.startswith("PROF_"):
            prof_dirs = [p for p in profiler_path.iterdir() if p.is_dir() and p.name.startswith("PROF_")]
        for prof_dir in prof_dirs[:1]:
            # DB deliverables
            db_files = list(prof_dir.glob("msprof_*.db"))
            files["profiler_db"] = len(db_files) > 0

            # Export output deliverables
            ms_output = prof_dir / "mindstudio_profiler_output"
            files["op_summary"] = bool(list(ms_output.glob("op_summary_*.csv")))
            files["task_time"] = bool(list(ms_output.glob("task_time_*.csv")))
            # msprof JSON deliverable (from msprof --export=on)
            files["msprof_json"] = bool(list(ms_output.glob("msprof_*.json")))

    return files


def check_aic_metrics(profiler_path: Path) -> bool:
    """Check if AIC PMU metrics data is available."""
    for pattern in ("**/aic_metrics_*", "**/*aic*pmu*", "**/*pmu*"):
        if list(profiler_path.rglob(pattern)):
            return True
    return False


def determine_quality_level(
    stop_ok: bool,
    parse_ok: bool,
    deliverables: dict[str, bool],
    data_type: Optional[str] = None,
) -> str:
    """Determine overall data quality level.

    Takes export_type into account for framework profiler:
    - Text mode: checks trace_view.json, kernel_details.csv
    - DB mode: checks profiler_db
    """
    if not stop_ok:
        return "critical"
    if not parse_ok:
        return "poor"

    # Filter out non-boolean entries (export_type)
    file_checks = {k: v for k, v in deliverables.items() if isinstance(v, bool)}
    present = sum(1 for v in file_checks.values() if v)

    if _is_framework(data_type):
        export_type = deliverables.get("export_type", "unknown")
        if export_type in ("text", "both"):
            critical_files = ["step_trace_time.csv", "kernel_details.csv", "trace_view.json"]
            critical_present = sum(1 for f in critical_files if file_checks.get(f, False))
            if critical_present >= 2 and present >= 3:
                return "excellent"
            if critical_present >= 1 and present >= 2:
                return "good"
        elif export_type == "db":
            if file_checks.get("profiler_db", False):
                return "excellent" if present >= 2 else "good"
        # Fallback
        if present >= 3:
            return "good"
        if present >= 1:
            return "fair"
    else:
        # msprof or other
        critical_files = ["step_trace_time.csv", "kernel_details.csv", "trace_view.json"]
        critical_present = sum(1 for f in critical_files if file_checks.get(f, False))
        if critical_present >= 2 and present >= 3:
            return "excellent"
        if critical_present >= 1 and present >= 2:
            return "good"
        if present >= 1:
            return "fair"

    return "poor"


def _build_config_summary(config: dict, data_type: Optional[str]) -> str:
    """Build a human-readable config summary line."""
    if not config.get("config_available"):
        return "N/A"
    parts = []
    if "profiler_level" in config:
        parts.append(f"level={config['profiler_level']}")
    if config.get("with_stack"):
        parts.append("with_stack")
    if config.get("with_modules"):
        parts.append("with_modules")
    if config.get("record_shapes"):
        parts.append("record_shapes")
    if config.get("profile_memory"):
        parts.append("profile_memory")
    if config.get("step_count"):
        parts.append(f"steps={config['step_count']}")
    return ", ".join(parts) if parts else "default"


def _recommend(quality: str, parse_ok: bool, data_type: Optional[str] = None) -> str:
    """Generate recommended action based on quality level."""
    if quality == "critical":
        return "Data collection did not complete normally. Check profiler.stop() was called. Recollect profiler data before analysis."
    if quality == "poor" and not parse_ok:
        if _is_framework(data_type):
            return "Data is not parsed. Run offline_parse_pytorch.py or offline_parse_mindspore.py to parse the data first."
        if _is_msprof(data_type):
            return "Data is not parsed. Run 'msprof --export=on --output=<path>' to export analysis results."
        return "Data is not parsed or key files are missing. Parse the profiler data first."
    if quality == "poor":
        return "Key files are missing. Analysis will be limited. Consider recollecting with complete profiler configuration."
    if quality == "fair":
        return "Limited data available. Analysis will be constrained. Consider collecting more complete profiling data."
    return "Data quality is sufficient for performance analysis. Proceed with Stage 1."


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate profiler data integrity")
    parser.add_argument("--trace-root", required=True, help="Profiler data root directory")
    parser.add_argument("--output-json", required=True, help="Path to write validation report JSON")
    args = parser.parse_args()

    trace_root = Path(args.trace_root).resolve()
    if not trace_root.exists():
        print(f"Error: path does not exist: {trace_root}", file=sys.stderr)
        return 1

    # Step 1: Detect data type (type-binding)
    data_type = detect_data_type(trace_root)

    # Step 2: Check collection status
    if _is_framework(data_type):
        stop_ok, stop_issues = check_stop_framework(trace_root)
    elif _is_msprof(data_type):
        stop_ok, stop_issues = check_stop_msprof(trace_root)
    else:
        # Unknown type - try both
        fw_ok, fw_issues = check_stop_framework(trace_root)
        ms_ok, ms_issues = check_stop_msprof(trace_root)
        if fw_ok:
            stop_ok, stop_issues = True, []
            data_type = data_type or "framework_profiler"
        elif ms_ok:
            stop_ok, stop_issues = ms_ok, ms_issues
            data_type = data_type or "msprof"
        else:
            stop_ok = False
            stop_issues = fw_issues + ms_issues
            data_type = data_type or "unknown"

    # Step 3: Parse config (framework profiler only)
    config: dict = {}
    if _is_framework(data_type):
        config = parse_profiler_config(trace_root)

    # Step 4: Check parse status
    if _is_framework(data_type):
        parse_ok, parse_issues = check_parse_framework(trace_root)
    elif _is_msprof(data_type):
        parse_ok, parse_issues = check_parse_msprof(trace_root)
    else:
        parse_ok, parse_issues = False, ["Unknown data type, cannot check parse status"]

    # Step 5: Check key deliverables
    deliverables = check_key_deliverables(trace_root, data_type or "unknown")

    # Optional: AIC metrics
    aic_available = check_aic_metrics(trace_root)

    # Determine quality
    quality = determine_quality_level(stop_ok, parse_ok, deliverables, data_type)

    # Build issues list
    all_issues: list[dict] = []
    if stop_issues:
        all_issues.extend([{"severity": "critical", "message": msg} for msg in stop_issues])
    if parse_issues:
        all_issues.extend([{"severity": "warning", "message": msg} for msg in parse_issues])

    # Missing deliverable warnings — extract export_type without mutating deliverables
    export_type = deliverables.get("export_type")
    deliverable_files = {k: v for k, v in deliverables.items() if isinstance(v, bool)}
    for name, present in deliverable_files.items():
        if not present:
            critical_names = {"step_trace_time.csv", "kernel_details.csv", "trace_view.json", "profiler_db", "op_summary"}
            severity = "warning" if name in critical_names else "note"
            all_issues.append({"severity": severity, "message": f"{name} not found"})

    # Build report
    report = {
        "schema_version": "performance-agent/0.2",
        "skill": "performance-agent",
        "data_type": data_type,
        "quality_level": quality,
        "stop_check": stop_ok,
        "parse_check": parse_ok,
        "export_type": export_type,
        "profiler_config": config if config.get("config_available") else None,
        "config_summary": _build_config_summary(config, data_type),
        "key_deliverables": deliverable_files,
        "aic_metrics_available": aic_available,
        "issues": all_issues,
        "recommended_action": _recommend(quality, parse_ok, data_type),
    }

    write_json(Path(args.output_json), report)
    summary = {
        "quality_level": quality,
        "data_type": data_type,
        "issues": len(all_issues),
    }
    if export_type:
        summary["export_type"] = export_type
    if config.get("config_available"):
        summary["config"] = _build_config_summary(config, data_type)
    print(json.dumps(summary, indent=2))

    if quality == "critical":
        return 1
    if quality == "poor" and not parse_ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
