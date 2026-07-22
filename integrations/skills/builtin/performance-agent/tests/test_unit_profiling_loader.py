"""Unit tests for profiling_loader.py.

Uses subprocess + the CLI interface (__main__) since the module uses
Python 3.10+ syntax (str | Path) that fails under Python 3.9 importlib.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, write_sample_pta_profiler_export, run_script, ROOT, SCRIPTS


def _write_torch_npu_profiler(root: Path) -> Path:
    """Create a torch_npu-style multi-rank profiler directory."""
    profiler_root = root / "pta_run"
    profiler_root.mkdir(parents=True, exist_ok=True)

    for rank_id in range(2):
        rank_dir = profiler_root / f"rank_{rank_id}"
        ascend = rank_dir / "ASCEND_PROFILER_OUTPUT"
        ascend.mkdir(parents=True, exist_ok=True)
        (profiler_root / f"profiler_info_{rank_id}.json").write_text(
            json.dumps({"rank_id": rank_id}), encoding="utf-8"
        )
        (ascend / "step_trace_time.csv").write_text(
            "Step ID,StepTime(ms)\n" + "\n".join(f"{i},{100 + rank_id * 5}" for i in range(3)) + "\n",
            encoding="utf-8",
        )
        (ascend / "communication.json").write_text(
            json.dumps({"communications": [{"op_name": "AllReduce", "time_ms": 50}]}),
            encoding="utf-8",
        )

    return profiler_root


def _write_msprof_profiler(root: Path) -> Path:
    """Create a msprof-style profiler directory."""
    profiler_root = root / "PROF_12345"
    msprof_out = profiler_root / "mindstudio_profiler_output"
    msprof_out.mkdir(parents=True, exist_ok=True)

    (msprof_out / "op_summary_0.csv").write_text(
        "Operator Name,Total Time(ms),Count\nMatMul,100,5\n",
        encoding="utf-8",
    )
    (msprof_out / "task_time_0.csv").write_text(
        "Step,Time(ms)\n1,80\n2,85\n",
        encoding="utf-8",
    )
    (profiler_root / "profiler_info.json").write_text("{}", encoding="utf-8")

    return profiler_root


# ---------------------------------------------------------------------------
# Format detection tests (via CLI summary)
# ---------------------------------------------------------------------------

def test_detect_mindspore_format(tmp_path: Path):
    """MindSpore profiler with ASCEND_PROFILER_OUTPUT → mindspore format."""
    profiler_root = write_sample_profiler_export(tmp_path)
    result = run_script("profiling_loader.py", str(profiler_root))
    data = json.loads(result.stdout)
    assert data["format"] == "mindspore"
    assert data["num_ranks"] >= 1


def test_detect_torch_npu_format(tmp_path: Path):
    """torch_npu multi-rank profiler → torch_npu format."""
    profiler_root = _write_torch_npu_profiler(tmp_path)
    result = run_script("profiling_loader.py", str(profiler_root))
    data = json.loads(result.stdout)
    assert data["format"] == "torch_npu"
    assert data["num_ranks"] == 2


def test_detect_msprof_format(tmp_path: Path):
    """msprof PROF_ directory → msprof format."""
    profiler_root = _write_msprof_profiler(tmp_path)
    result = run_script("profiling_loader.py", str(profiler_root))
    data = json.loads(result.stdout)
    assert data["format"] == "msprof"


def test_detect_unknown_format(tmp_path: Path):
    """Empty directory → unknown format."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = run_script("profiling_loader.py", str(empty_dir))
    data = json.loads(result.stdout)
    assert data["format"] == "unknown"


# ---------------------------------------------------------------------------
# Inventory tests (via CLI summary)
# ---------------------------------------------------------------------------

def test_inventory_mindspore_single_rank(tmp_path: Path):
    """Single-rank MindSpore profiler should report available data."""
    profiler_root = write_sample_profiler_export(tmp_path)
    result = run_script("profiling_loader.py", str(profiler_root))
    data = json.loads(result.stdout)

    assert data["num_ranks"] >= 1
    avail = data["available_data"]
    assert avail["step_trace"] is True
    assert avail["communication"] is True
    assert avail["trace_view"] is True


def test_inventory_torch_npu_multi_rank(tmp_path: Path):
    """Multi-rank torch_npu should discover all rank directories."""
    profiler_root = _write_torch_npu_profiler(tmp_path)
    result = run_script("profiling_loader.py", str(profiler_root))
    data = json.loads(result.stdout)

    assert data["num_ranks"] == 2
    assert sorted(data["rank_ids"]) == [0, 1]


def test_inventory_msprof_has_aic_metrics(tmp_path: Path):
    """msprof profiler should discover op_summary as aic_metrics."""
    profiler_root = _write_msprof_profiler(tmp_path)
    result = run_script("profiling_loader.py", str(profiler_root))
    data = json.loads(result.stdout)

    assert data["available_data"]["aic_metrics"] is True


def test_inventory_reports_missing_data(tmp_path: Path):
    """Profiler with partial data should report what's missing."""
    profiler_root = tmp_path / "partial_run"
    ascend = profiler_root / "ASCEND_PROFILER_OUTPUT"
    ascend.mkdir(parents=True)
    # Only step_trace_time.csv, no communication or memory
    (ascend / "step_trace_time.csv").write_text(
        "Step ID,StepTime(ms)\n1,100\n", encoding="utf-8"
    )

    result = run_script("profiling_loader.py", str(profiler_root))
    data = json.loads(result.stdout)

    avail = data["available_data"]
    assert avail["step_trace"] is True
    assert avail["communication"] is False
    assert avail["memory_record"] is False


# ---------------------------------------------------------------------------
# Non-existent path test
# ---------------------------------------------------------------------------

def test_nonexistent_path_exits_with_error(tmp_path: Path):
    """Non-existent path should exit with error."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "profiling_loader.py"),
         str(tmp_path / "nonexistent")],
        text=True, capture_output=True,
    )
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# PTA realistic data tests (based on real pro_data)
# ---------------------------------------------------------------------------


def test_detect_pta_single_rank_from_directory_name(tmp_path: Path):
    """PTA profiler with *_ascend_pt name → torch_npu format."""
    profiler_root = write_sample_pta_profiler_export(tmp_path)
    result = run_script("profiling_loader.py", str(profiler_root))
    data = json.loads(result.stdout)
    assert data["format"] == "torch_npu"
    assert data["num_ranks"] == 1
    assert data["rank_ids"] == [0]


def test_pta_inventory_kernel_details_available(tmp_path: Path):
    """PTA profiler should discover kernel_details.csv."""
    profiler_root = write_sample_pta_profiler_export(tmp_path)
    result = run_script("profiling_loader.py", str(profiler_root))
    data = json.loads(result.stdout)
    assert data["available_data"]["trace_view"] is True
    assert data["available_data"]["profiler_info"] is True


def test_pta_inventory_reports_missing_step_trace(tmp_path: Path):
    """PTA profiler without step_trace_time.csv should report missing."""
    profiler_root = write_sample_pta_profiler_export(tmp_path)
    result = run_script("profiling_loader.py", str(profiler_root))
    data = json.loads(result.stdout)
    assert data["available_data"]["step_trace"] is False
    assert data["available_data"]["communication"] is False
    assert data["available_data"]["memory_record"] is False
