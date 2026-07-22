#!/usr/bin/env python3
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Optional


TRACE_PATTERNS = {
    "step_trace_time": ["**/ASCEND_PROFILER_OUTPUT/step_trace_time.csv"],
    "kernel_details": ["**/ASCEND_PROFILER_OUTPUT/kernel_details.csv"],
    "trace_view": ["**/ASCEND_PROFILER_OUTPUT/trace_view.json"],
    "communication": ["**/ASCEND_PROFILER_OUTPUT/communication.json"],
    "communication_matrix": ["**/ASCEND_PROFILER_OUTPUT/communication_matrix.json"],
    "memory_record": ["**/ASCEND_PROFILER_OUTPUT/memory_record.csv"],
    "operator_memory": ["**/ASCEND_PROFILER_OUTPUT/operator_memory.csv"],
    "module_memory": ["**/ASCEND_PROFILER_OUTPUT/npu_module_mem.csv"],
    "dataset": ["**/ASCEND_PROFILER_OUTPUT/dataset.csv"],
    "minddata_pipeline_csv": ["**/ASCEND_PROFILER_OUTPUT/minddata_pipeline_summary_*.csv"],
    "minddata_pipeline_json": ["**/ASCEND_PROFILER_OUTPUT/minddata_pipeline_summary_*.json"],
    "op_summary": ["**/mindstudio_profiler_output/op_summary_*.csv"],
    "task_time": ["**/mindstudio_profiler_output/task_time_*.csv"],
    "hotspot_summary_json": ["**/hotspot_summary.json"],
    "hotspot_summary_md": ["**/hotspot_summary.md"],
}


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def parse_number(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return []
        return [dict(row) for row in reader]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


_MAX_WALK_DEPTH = 30


def profiler_root_from_path(path: Path) -> Path:
    current = path.resolve()
    if current.is_file():
        current = current.parent

    if current.name == "ASCEND_PROFILER_OUTPUT":
        return current.parent
    if current.name == "mindstudio_profiler_output":
        return current.parent.parent if current.parent.name.startswith("PROF_") else current.parent

    for _ in range(_MAX_WALK_DEPTH):
        if (current / "ASCEND_PROFILER_OUTPUT").exists():
            return current
        if (current / "profiler_metadata.json").exists():
            return current
        if any(current.glob("profiler_info_*.json")):
            return current
        if current.name.startswith("PROF_") and (current / "mindstudio_profiler_output").exists():
            return current.parent
        parent = current.parent
        if parent == current:
            break
        current = parent

    return path.resolve() if path.resolve().is_dir() else path.resolve().parent


def trace_file_inventory(root: Path) -> dict[str, list[Path]]:
    inventory: dict[str, list[Path]] = {}
    for key, patterns in TRACE_PATTERNS.items():
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(sorted(root.glob(pattern)))
        deduped = []
        seen = set()
        for match in matches:
            resolved = match.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            deduped.append(resolved)
        if deduped:
            inventory[key] = deduped
    return inventory


def score_trace_inventory(inventory: dict[str, list[Path]]) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    weights = {
        "step_trace_time": 30,
        "kernel_details": 25,
        "trace_view": 20,
        "communication": 20,
        "communication_matrix": 10,
        "memory_record": 15,
        "operator_memory": 10,
        "dataset": 10,
        "minddata_pipeline_csv": 10,
        "op_summary": 10,
        "hotspot_summary_json": 12,
    }
    for key, weight in weights.items():
        if inventory.get(key):
            score += weight
            reasons.append(f"found {key}")
    return score, reasons


def confidence_from_score(score: int) -> str:
    if score >= 80:
        return "strong"
    if score >= 45:
        return "moderate"
    if score > 0:
        return "weak"
    return "none"


def infer_stack_from_root(root: Path) -> Optional[str]:
    text = root.name.lower()
    if text.endswith("_ascend_ms") or "_ascend_ms_" in text:
        return "ms"
    if text.endswith("_ascend_pt") or "_ascend_pt_" in text:
        return "pta"
    return None


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def first_file(root: Path, inventory: dict[str, list[Path]], key: str) -> Optional[str]:
    files = inventory.get(key, [])
    if not files:
        return None
    return relpath(files[0], root)


def list_files(root: Path, inventory: dict[str, list[Path]], key: str) -> list[str]:
    return [relpath(path, root) for path in inventory.get(key, [])]


def stage_to_domain(stage_name: str) -> Optional[str]:
    mapping = {
        "communication": "communication",
        "input_pipeline": "input_pipeline",
        "memory_pressure": "memory",
        "host_overhead": "host_framework_overhead",
        "idle_gap": "host_framework_overhead",
        "graph_compile": "graph_compile",
        "compute": "compute",
        "operator_hotspot": "operator_hotspot",
    }
    return mapping.get(stage_name)


def load_optional_json(path_str: Optional[str]):
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return read_json(path)


# ---------------------------------------------------------------------------
# Hardware specs
# ---------------------------------------------------------------------------

HARDWARE_SPECS: dict[str, dict] = {
    "ascend_910b1": {
        "fp16_tflops": 378.88,
        "bf16_tflops": 378.88,
        "fp32_tflops": 189.44,
        "hbm_bandwidth_gb_s": 1600.0,
        "hbm_capacity_gb": 64,
        "hccs_bandwidth_gb_s": 56.0,
    },
    "ascend_910b2": {
        "fp16_tflops": 353.89,
        "bf16_tflops": 353.89,
        "fp32_tflops": 176.95,
        "hbm_bandwidth_gb_s": 1500.0,
        "hbm_capacity_gb": 64,
        "hccs_bandwidth_gb_s": 56.0,
    },
    "ascend_910b3": {
        "fp16_tflops": 294.91,
        "bf16_tflops": 294.91,
        "fp32_tflops": 147.46,
        "hbm_bandwidth_gb_s": 1200.0,
        "hbm_capacity_gb": 64,
        "hccs_bandwidth_gb_s": 56.0,
    },
    "ascend_910b4": {
        "fp16_tflops": 270.0,
        "bf16_tflops": 270.0,
        "fp32_tflops": 135.0,
        "hbm_bandwidth_gb_s": 1200.0,
        "hbm_capacity_gb": 32,
        "hccs_bandwidth_gb_s": 56.0,
    },
    "atlas_a2_280t": {
        "fp16_tflops": 280.0,
        "bf16_tflops": 280.0,
        "fp32_tflops": 140.0,
        "hbm_bandwidth_gb_s": 1500.0,
        "hbm_capacity_gb": 64,
        "hccs_bandwidth_gb_s": 56.0,
    },
    "atlas_a2_313t": {
        "fp16_tflops": 313.0,
        "bf16_tflops": 313.0,
        "fp32_tflops": 156.5,
        "hbm_bandwidth_gb_s": 1800.0,
        "hbm_capacity_gb": 64,
        "hccs_bandwidth_gb_s": 56.0,
    },
    "atlas_a2_376t": {
        "fp16_tflops": 376.0,
        "bf16_tflops": 376.0,
        "fp32_tflops": 188.0,
        "hbm_bandwidth_gb_s": 2000.0,
        "hbm_capacity_gb": 64,
        "hccs_bandwidth_gb_s": 56.0,
    },
    "atlas_300i_310p": {
        "fp16_tflops": 22.0,
        "bf16_tflops": 22.0,
        "fp32_tflops": 11.0,
        "hbm_bandwidth_gb_s": 68.0,
        "hbm_capacity_gb": 0,
        "hccs_bandwidth_gb_s": 0.0,
    },
}

# Mapping from chip name strings found in profiler_info.json to spec keys.
_CHIP_NAME_MAP = {
    "910b1": "ascend_910b1",
    "910b2": "ascend_910b2",
    "910b3": "ascend_910b3",
    "910b4": "ascend_910b4",
    "ascend910b1": "ascend_910b1",
    "ascend910b2": "ascend_910b2",
    "ascend910b3": "ascend_910b3",
    "ascend910b4": "ascend_910b4",
    "280t": "atlas_a2_280t",
    "313t": "atlas_a2_313t",
    "376t": "atlas_a2_376t",
    "310p": "atlas_300i_310p",
    "a2-280t": "atlas_a2_280t",
    "a2-313t": "atlas_a2_313t",
    "a2-376t": "atlas_a2_376t",
}


def infer_hardware(trace_root: Path) -> Optional[str]:
    """Attempt to infer hardware model from profiler_info.json or directory name."""
    root = trace_root.resolve()

    # Try profiler_info.json
    for pattern in ("profiler_info.json", "profiler_info_*.json"):
        for info_file in sorted(root.glob(pattern)):
            try:
                info = read_json(info_file)
            except Exception:
                continue
            for field in ("chip_name", "device_info", "chip_type", "device_type"):
                value = str(info.get(field, "")).lower().replace(" ", "").replace("-", "")
                for chip_key, spec_key in _CHIP_NAME_MAP.items():
                    if chip_key in value:
                        return spec_key
            # Check nested fields
            for nested in ("device_info", "hardware_info"):
                nested_obj = info.get(nested)
                if not isinstance(nested_obj, dict):
                    continue
                for field in ("chip_name", "chip_type", "device_type"):
                    value = str(nested_obj.get(field, "")).lower().replace(" ", "").replace("-", "")
                    for chip_key, spec_key in _CHIP_NAME_MAP.items():
                        if chip_key in value:
                            return spec_key

    # Try directory name
    dir_name = root.name.lower().replace("-", "").replace(" ", "")
    for chip_key, spec_key in _CHIP_NAME_MAP.items():
        if chip_key in dir_name:
            return spec_key

    return None


def get_peak_tflops(hardware: Optional[str], precision: str = "fp16") -> Optional[float]:
    """Return the peak TFLOPS for a hardware model and precision."""
    if not hardware:
        return None
    spec = HARDWARE_SPECS.get(hardware)
    if not spec:
        return None
    key = f"{precision}_tflops"
    value = spec.get(key)
    if value is not None:
        return value
    # Fallback to fp16 only when the requested precision key is absent.
    return spec.get("fp16_tflops")


def infer_parallel_config(trace_root: Path) -> Optional[dict]:
    """Attempt to infer parallel configuration from cluster directory structure.

    Returns dict with tp_size, pp_size, dp_size, world_size or None.
    """
    root = trace_root.resolve()

    # Count rank subdirectories / profiler_info files
    rank_files = list(root.glob("profiler_info_*.json")) + list(root.glob("*/profiler_info.json"))
    subdirs = [p for p in root.iterdir() if p.is_dir() and (p / "profiler_info.json").exists()]
    world_size = max(len(rank_files), len(subdirs))
    if world_size <= 1:
        return None

    # Try to read parallel config from profiler_info.json
    for info_file in sorted(root.glob("profiler_info_*.json"))[:1]:
        try:
            info = read_json(info_file)
            parallel = info.get("parallel_config") or info.get("parallel")
            if isinstance(parallel, dict):
                return {
                    "tp_size": parallel.get("tensor_model_parallel_size") or parallel.get("tp_size"),
                    "pp_size": parallel.get("pipeline_model_parallel_size") or parallel.get("pp_size"),
                    "dp_size": parallel.get("data_parallel_size") or parallel.get("dp_size"),
                    "world_size": world_size,
                    "source": "profiler_info_json",
                }
        except Exception:
            continue

    return {
        "tp_size": None,
        "pp_size": None,
        "dp_size": None,
        "world_size": world_size,
        "source": "inferred_from_directory_structure",
    }


def mfu_level(mfu: float) -> str:
    """Return a human-readable MFU level."""
    if mfu < 0.20:
        return "low"
    if mfu < 0.40:
        return "below_average"
    if mfu < 0.60:
        return "medium"
    if mfu < 0.70:
        return "good"
    return "excellent"


# ---------------------------------------------------------------------------
# Cluster / rank helpers (shared across detect_slow_ranks, calculate_linearity,
# analyze_jitter, profiling_loader)
# ---------------------------------------------------------------------------

def find_rank_dirs(trace_root: Path) -> dict[int, Path]:
    """Find rank directories and their profiler_info files.

    Discovers ranks from:
    1. profiler_info_{RankID}.json files at root level
    2. Subdirectories containing profiler_info.json with rank ID in name
    """
    ranks: dict[int, Path] = {}
    for info_file in sorted(trace_root.glob("profiler_info_*.json")):
        try:
            rank_id = int(info_file.stem.split("_")[-1])
            ranks[rank_id] = info_file.parent
        except ValueError:
            continue
    if not ranks:
        for sub in sorted(trace_root.iterdir()):
            if not sub.is_dir():
                continue
            info_file = sub / "profiler_info.json"
            if not info_file.exists():
                continue
            name = sub.name.lower()
            rank_id = None
            for part in name.split("_"):
                try:
                    rank_id = int(part)
                    break
                except ValueError:
                    continue
            if rank_id is not None:
                ranks[rank_id] = sub
    return ranks


def find_step_trace_csv(rank_dir: Path) -> Optional[Path]:
    """Find step_trace_time.csv in a rank directory."""
    candidates = [
        rank_dir / "ASCEND_PROFILER_OUTPUT" / "step_trace_time.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    for match in rank_dir.rglob("step_trace_time.csv"):
        return match
    return None


def avg_step_time_from_csv(step_csv: Path) -> Optional[float]:
    """Compute average step time in ms from a step_trace_time.csv.

    Normalizes headers and matches columns containing 'step' + ('time'|'total'|'interval').
    """
    total = 0.0
    count = 0
    with step_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for header, value in row.items():
                key = re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")
                if "step" in key and ("time" in key or "total" in key or "interval" in key):
                    try:
                        total += float(value.replace(",", "").strip())
                        count += 1
                    except (ValueError, AttributeError):
                        continue
                    break
    return total / count if count > 0 else None


def check_msprof_available() -> None:
    """Ensure msprof is in PATH; required by CANN profiler analyse.

    Exits with code 1 if msprof is not found.
    """
    if shutil.which("msprof") is None:
        print(
            "Error: msprof command not found. Please source the correct CANN environment, e.g.:\n"
            "  source /usr/local/Ascend/ascend-toolkit/set_env.sh",
            file=sys.stderr,
        )
        sys.exit(1)
