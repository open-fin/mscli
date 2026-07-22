#!/usr/bin/env python3
from __future__ import annotations

"""Unified profiling data loading layer for Ascend NPU profiler exports.

Auto-detects profiler format (mindspore, torch_npu, msprof) and provides
a single interface for loading step trace, communication, memory, operator
hotspot, trace view, and AIC metrics data across all formats.

Usage:
    from profiling_loader import ProfilingLoader

    loader = ProfilingLoader("/path/to/profiler_output")
    step_data = loader.get_step_trace()
    comm_data = loader.get_communication()
"""
import csv
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from perf_common import load_csv_rows, read_json


class ProfilerFormat(Enum):
    MINDSPORE = "mindspore"
    TORCH_NPU = "torch_npu"
    MSPROF = "msprof"
    UNKNOWN = "unknown"


@dataclass
class RankInfo:
    rank_id: int
    path: Path
    has_step_trace: bool = False
    has_communication: bool = False
    has_memory: bool = False


@dataclass
class ProfilingInventory:
    format: ProfilerFormat = ProfilerFormat.UNKNOWN
    root: Optional[Path] = None
    ranks: dict[int, RankInfo] = field(default_factory=dict)
    step_trace_path: Optional[Path] = None
    communication_path: Optional[Path] = None
    communication_matrix_path: Optional[Path] = None
    memory_record_path: Optional[Path] = None
    operator_memory_path: Optional[Path] = None
    module_memory_path: Optional[Path] = None
    trace_view_path: Optional[Path] = None
    aic_metrics_path: Optional[Path] = None
    profiler_info_path: Optional[Path] = None
    dataset_csv_path: Optional[Path] = None


class ProfilingLoader:
    """Unified loader for Ascend NPU profiling data.

    Auto-detects the profiler format, discovers rank structure,
    and provides methods to load each data type.
    """

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._inventory: Optional[ProfilingInventory] = None
        self._cache: dict[str, object] = {}

    @property
    def root(self) -> Path:
        return self._root

    @property
    def format(self) -> ProfilerFormat:
        return self._detect_format()

    @property
    def inventory(self) -> ProfilingInventory:
        if self._inventory is None:
            self._inventory = self._build_inventory()
        return self._inventory

    def _detect_format(self) -> ProfilerFormat:
        """Detect profiler format from directory structure and files."""
        root = self._root

        # Check directory name convention first: *_ascend_pt = torch_npu
        dir_name = root.name.lower()
        if dir_name.endswith("_ascend_pt") or "_ascend_pt_" in dir_name:
            return ProfilerFormat.TORCH_NPU
        if dir_name.endswith("_ascend_ms") or "_ascend_ms_" in dir_name:
            return ProfilerFormat.MINDSPORE

        # Check profiler_info.json for framework indicators
        for info_file in sorted(root.glob("profiler_info*.json")):
            try:
                info = read_json(info_file)
                if "torch_npu_version" in info:
                    return ProfilerFormat.TORCH_NPU
                if "mindspore_version" in info:
                    return ProfilerFormat.MINDSPORE
            except Exception:
                continue

        # MindSpore: ASCEND_PROFILER_OUTPUT directory
        ascend_dir = root / "ASCEND_PROFILER_OUTPUT"
        if ascend_dir.exists():
            return ProfilerFormat.MINDSPORE

        # torch_npu: typically has profiler_info.json at root
        # or rank subdirectories with ASCEND_PROFILER_OUTPUT
        if (root / "profiler_info.json").exists() or list(root.glob("profiler_info_*.json")):
            # Check if subdirectories have ASCEND_PROFILER_OUTPUT
            for sub in root.iterdir():
                if sub.is_dir() and (sub / "ASCEND_PROFILER_OUTPUT").exists():
                    return ProfilerFormat.TORCH_NPU

        # msprof: PROF_ directory with mindstudio_profiler_output
        if root.name.startswith("PROF_") or any(
            p.name.startswith("PROF_") for p in root.iterdir() if p.is_dir()
        ):
            return ProfilerFormat.MSPROF

        # Check for msprof output structure
        if (root / "mindstudio_profiler_output").exists():
            return ProfilerFormat.MSPROF

        # Try subdirectories
        for sub in root.iterdir():
            if not sub.is_dir():
                continue
            if (sub / "ASCEND_PROFILER_OUTPUT").exists():
                return ProfilerFormat.MINDSPORE
            if (sub / "mindstudio_profiler_output").exists():
                return ProfilerFormat.MSPROF

        return ProfilerFormat.UNKNOWN

    def _build_inventory(self) -> ProfilingInventory:
        """Build inventory of available profiling files."""
        inv = ProfilingInventory(
            format=self._detect_format(),
            root=self._root,
        )

        fmt = inv.format

        if fmt == ProfilerFormat.MINDSPORE or fmt == ProfilerFormat.TORCH_NPU:
            self._inventory_ascend(inv)
        elif fmt == ProfilerFormat.MSPROF:
            self._inventory_msprof(inv)

        return inv

    def _inventory_ascend(self, inv: ProfilingInventory) -> None:
        """Build inventory for MindSpore/torch_npu format."""
        root = self._root
        ascend_dir = root / "ASCEND_PROFILER_OUTPUT"

        # If root itself has ASCEND_PROFILER_OUTPUT, single-rank
        if ascend_dir.exists():
            inv.step_trace_path = ascend_dir / "step_trace_time.csv"
            inv.communication_path = ascend_dir / "communication.json"
            inv.communication_matrix_path = ascend_dir / "communication_matrix.json"
            inv.memory_record_path = ascend_dir / "memory_record.csv"
            inv.operator_memory_path = ascend_dir / "operator_memory.csv"
            inv.module_memory_path = ascend_dir / "npu_module_mem.csv"
            inv.trace_view_path = ascend_dir / "trace_view.json"
            inv.dataset_csv_path = ascend_dir / "dataset.csv"

            # AIC metrics
            for aic_path in sorted(ascend_dir.glob("aic_metrics_*.csv")):
                inv.aic_metrics_path = aic_path
                break

            # Profiler info
            for info_path in sorted(root.glob("profiler_info*.json")):
                inv.profiler_info_path = info_path
                break

            # Single rank (rank 0)
            inv.ranks[0] = RankInfo(
                rank_id=0,
                path=root,
                has_step_trace=inv.step_trace_path.exists() if inv.step_trace_path else False,
                has_communication=inv.communication_path.exists() if inv.communication_path else False,
                has_memory=(inv.memory_record_path and inv.memory_record_path.exists())
                or (inv.operator_memory_path and inv.operator_memory_path.exists()),
            )

        # Multi-rank: profiler_info_{RankID}.json at root
        for info_file in sorted(root.glob("profiler_info_*.json")):
            try:
                rank_id = int(info_file.stem.split("_")[-1])
                rank_dir = info_file.parent
                ascend_sub = rank_dir / "ASCEND_PROFILER_OUTPUT"
                inv.ranks[rank_id] = RankInfo(
                    rank_id=rank_id,
                    path=rank_dir,
                    has_step_trace=(ascend_sub / "step_trace_time.csv").exists(),
                    has_communication=(ascend_sub / "communication.json").exists(),
                    has_memory=(ascend_sub / "memory_record.csv").exists()
                    or (ascend_sub / "operator_memory.csv").exists(),
                )
            except ValueError:
                continue

        # Multi-rank: subdirectories with profiler_info.json
        if not inv.ranks:
            for sub in sorted(root.iterdir()):
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
                    ascend_sub = sub / "ASCEND_PROFILER_OUTPUT"
                    inv.ranks[rank_id] = RankInfo(
                        rank_id=rank_id,
                        path=sub,
                        has_step_trace=(ascend_sub / "step_trace_time.csv").exists(),
                        has_communication=(ascend_sub / "communication.json").exists(),
                        has_memory=(ascend_sub / "memory_record.csv").exists()
                        or (ascend_sub / "operator_memory.csv").exists(),
                    )

    def _inventory_msprof(self, inv: ProfilingInventory) -> None:
        """Build inventory for msprof format."""
        root = self._root

        # Find msprof output directory
        msprof_dir = root / "mindstudio_profiler_output"
        if not msprof_dir.exists():
            for sub in root.iterdir():
                if sub.is_dir() and (sub / "mindstudio_profiler_output").exists():
                    msprof_dir = sub / "mindstudio_profiler_output"
                    break

        if msprof_dir.exists():
            # Operator summary
            for path in sorted(msprof_dir.glob("op_summary_*.csv")):
                inv.aic_metrics_path = path
                break

            # Task time as step trace substitute
            for path in sorted(msprof_dir.glob("task_time_*.csv")):
                inv.step_trace_path = path
                break

        # Profiler info
        for info_path in sorted(root.rglob("profiler_info*.json")):
            inv.profiler_info_path = info_path
            break

        # Discover multi-rank msprof data (PROF_* subdirectories)
        prof_dirs = sorted(
            p for p in root.iterdir()
            if p.is_dir() and p.name.startswith("PROF_")
        )
        if prof_dirs:
            for idx, prof_dir in enumerate(prof_dirs):
                ms_out = prof_dir / "mindstudio_profiler_output"
                has_step = bool(list(ms_out.glob("task_time_*.csv"))) if ms_out.exists() else False
                inv.ranks[idx] = RankInfo(
                    rank_id=idx,
                    path=prof_dir,
                    has_step_trace=has_step,
                )
            # Use first PROF_ dir for primary paths if not already set
            if not inv.step_trace_path and prof_dirs:
                first_ms = prof_dirs[0] / "mindstudio_profiler_output"
                if first_ms.exists():
                    for path in sorted(first_ms.glob("task_time_*.csv")):
                        inv.step_trace_path = path
                        break
                    for path in sorted(first_ms.glob("op_summary_*.csv")):
                        inv.aic_metrics_path = path
                        break
        else:
            inv.ranks[0] = RankInfo(
                rank_id=0,
                path=root,
                has_step_trace=inv.step_trace_path is not None and inv.step_trace_path.exists(),
            )

    def _validate_path(self, path: Optional[Path]) -> Optional[Path]:
        """Return path only if it exists."""
        return path if path and path.exists() else None

    def get_step_trace(self, rank_id: Optional[int] = None) -> Optional[list[dict]]:
        """Load step trace CSV data.

        Args:
            rank_id: Specific rank to load (None for single-rank or rank 0).

        Returns:
            List of row dicts from CSV, or None if not available.
        """
        cache_key = f"step_trace_{rank_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        path = self._resolve_rank_file("step_trace_time.csv", rank_id)
        path = path or self._validate_path(self.inventory.step_trace_path)
        if not path:
            return None

        rows = load_csv_rows(path)
        self._cache[cache_key] = rows
        return rows

    def get_communication(self) -> Optional[object]:
        """Load communication.json data."""
        if "communication" in self._cache:
            return self._cache["communication"]

        path = self._validate_path(self.inventory.communication_path)
        if not path:
            return None

        data = read_json(path)
        self._cache["communication"] = data
        return data

    def get_communication_matrix(self) -> Optional[object]:
        """Load communication_matrix.json data."""
        if "communication_matrix" in self._cache:
            return self._cache["communication_matrix"]

        path = self._validate_path(self.inventory.communication_matrix_path)
        if not path:
            return None

        data = read_json(path)
        self._cache["communication_matrix"] = data
        return data

    def get_memory_data(self) -> Optional[dict[str, list[dict]]]:
        """Load all available memory CSV data.

        Returns dict with keys: memory_record, operator_memory, module_memory.
        """
        if "memory" in self._cache:
            return self._cache["memory"]  # type: ignore[return-value]

        result: dict[str, list[dict]] = {}

        for key, path in [
            ("memory_record", self.inventory.memory_record_path),
            ("operator_memory", self.inventory.operator_memory_path),
            ("module_memory", self.inventory.module_memory_path),
        ]:
            validated = self._validate_path(path)
            if validated:
                result[key] = load_csv_rows(validated)

        if result:
            self._cache["memory"] = result
        return result or None

    def get_operator_hotspots(self, rank_id: Optional[int] = None) -> Optional[list[dict]]:
        """Load operator hotspot data.

        For msprof, reads op_summary CSV. For MindSpore, reads kernel_details.
        """
        cache_key = f"hotspots_{rank_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        # Try msprof op_summary
        path = self._validate_path(self.inventory.aic_metrics_path)
        if not path:
            # Try kernel_details for MindSpore
            path = self._resolve_rank_file("kernel_details.csv", rank_id)
            path = self._validate_path(path)

        if not path:
            return None

        rows = load_csv_rows(path)
        self._cache[cache_key] = rows
        return rows

    def get_trace_view(self) -> Optional[object]:
        """Load trace_view.json data."""
        if "trace_view" in self._cache:
            return self._cache["trace_view"]

        path = self._validate_path(self.inventory.trace_view_path)
        if not path:
            return None

        data = read_json(path)
        self._cache["trace_view"] = data
        return data

    def get_aic_metrics(self, rank_id: Optional[int] = None) -> Optional[list[dict]]:
        """Load AIC metrics CSV data."""
        cache_key = f"aic_metrics_{rank_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]  # type: ignore[return-value]

        path = self._resolve_rank_file("aic_metrics.csv", rank_id)
        if not path:
            # Try glob pattern
            ascend_dir = self._root / "ASCEND_PROFILER_OUTPUT"
            for match in sorted(ascend_dir.glob("aic_metrics_*.csv")):
                path = match
                break

        path = self._validate_path(path)
        if not path:
            return None

        rows = load_csv_rows(path)
        self._cache[cache_key] = rows
        return rows

    def get_profiler_info(self) -> Optional[dict]:
        """Load profiler_info.json data."""
        if "profiler_info" in self._cache:
            return self._cache["profiler_info"]  # type: ignore[return-value]

        path = self._validate_path(self.inventory.profiler_info_path)
        if not path:
            return None

        data = read_json(path)
        self._cache["profiler_info"] = data
        return data

    def get_all_rank_step_times(self) -> dict[int, float]:
        """Load average step time for all ranks.

        Returns dict mapping rank_id to average step time in ms.
        """
        if "all_rank_step_times" in self._cache:
            return self._cache["all_rank_step_times"]  # type: ignore[return-value]

        result: dict[int, float] = {}
        for rank_id in self.inventory.ranks:
            rows = self.get_step_trace(rank_id)
            if not rows:
                continue

            total = 0.0
            count = 0
            for row in rows:
                for header, value in row.items():
                    key = re.sub(r"[^a-z0-9]+", "_", header.strip().lower()).strip("_")
                    if "step" in key and ("time" in key or "total" in key or "interval" in key):
                        try:
                            total += float(value.replace(",", "").strip())
                            count += 1
                        except (ValueError, AttributeError):
                            continue
                        break
            if count > 0:
                result[rank_id] = total / count

        self._cache["all_rank_step_times"] = result
        return result

    def _resolve_rank_file(self, filename: str, rank_id: Optional[int] = None) -> Optional[Path]:
        """Resolve a file path for a specific rank."""
        if rank_id is not None and rank_id in self.inventory.ranks:
            rank_info = self.inventory.ranks[rank_id]
            # Check ASCEND_PROFILER_OUTPUT subdirectory
            ascend_dir = rank_info.path / "ASCEND_PROFILER_OUTPUT"
            candidate = ascend_dir / filename
            if candidate.exists():
                return candidate
            # Check rank directory directly
            candidate = rank_info.path / filename
            if candidate.exists():
                return candidate

        # Single rank (no rank_id specified)
        ascend_dir = self._root / "ASCEND_PROFILER_OUTPUT"
        candidate = ascend_dir / filename
        if candidate.exists():
            return candidate

        # Root level
        candidate = self._root / filename
        if candidate.exists():
            return candidate

        return None

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    def summary(self) -> dict:
        """Return a summary of available profiling data."""
        inv = self.inventory
        return {
            "root": str(inv.root),
            "format": inv.format.value,
            "num_ranks": len(inv.ranks),
            "rank_ids": sorted(inv.ranks.keys()),
            "available_data": {
                "step_trace": inv.step_trace_path is not None and inv.step_trace_path.exists(),
                "communication": inv.communication_path is not None and inv.communication_path.exists(),
                "communication_matrix": inv.communication_matrix_path is not None and inv.communication_matrix_path.exists(),
                "memory_record": inv.memory_record_path is not None and inv.memory_record_path.exists(),
                "operator_memory": inv.operator_memory_path is not None and inv.operator_memory_path.exists(),
                "trace_view": inv.trace_view_path is not None and inv.trace_view_path.exists(),
                "aic_metrics": inv.aic_metrics_path is not None and inv.aic_metrics_path.exists(),
                "profiler_info": inv.profiler_info_path is not None and inv.profiler_info_path.exists(),
            },
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect profiler data inventory")
    parser.add_argument("root", help="Profiler output root directory")
    args = parser.parse_args()

    loader = ProfilingLoader(args.root)
    print(json.dumps(loader.summary(), indent=2))
