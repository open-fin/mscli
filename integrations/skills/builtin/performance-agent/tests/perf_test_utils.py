import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_script(script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a script from the scripts/ directory as a subprocess."""
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script_name), *args],
        check=True, text=True, capture_output=True,
    )


def write_sample_profiler_export(root: Path) -> Path:
    profiler_root = root / "worker_0_20260325_ascend_ms"
    ascend = profiler_root / "ASCEND_PROFILER_OUTPUT"
    cann = profiler_root / "PROF_0" / "mindstudio_profiler_output"
    ascend.mkdir(parents=True, exist_ok=True)
    cann.mkdir(parents=True, exist_ok=True)

    (profiler_root / "profiler_metadata.json").write_text("{}", encoding="utf-8")
    (profiler_root / "profiler_info_0.json").write_text("{}", encoding="utf-8")

    (ascend / "step_trace_time.csv").write_text(
        "\n".join(
            [
                "Step ID,DataTime(ms),ComputeTime(ms),CommunicationTime(ms),IdleGap(ms),StepTime(ms)",
                "1,6,28,44,8,86",
                "2,5,27,42,9,83",
                "3,7,29,46,7,89",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (ascend / "kernel_details.csv").write_text(
        "\n".join(
            [
                "Kernel Name,Duration(ms)",
                "AllReduce,44",
                "MatMul,22",
                "LayerNorm,9",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (ascend / "trace_view.json").write_text(
        json.dumps(
            {
                "events": [
                    {"name": "host_idle_gap", "duration_ms": 8},
                    {"name": "launch_overhead", "duration_ms": 12},
                    {"name": "graph_compile", "duration_ms": 6},
                    {"name": "AllReduce", "duration_ms": 44},
                ]
            }
        ),
        encoding="utf-8",
    )
    (ascend / "communication.json").write_text(
        json.dumps(
            {
                "communications": [
                    {"op_name": "AllReduce", "time_ms": 132, "count": 3, "size_mb": 256},
                    {"op_name": "AllGather", "time_ms": 24, "count": 3, "size_mb": 64},
                ]
            }
        ),
        encoding="utf-8",
    )
    (ascend / "communication_matrix.json").write_text(
        json.dumps(
            {
                "matrix": [
                    [120.0, 100.0],
                    [110.0, 95.0],
                ]
            }
        ),
        encoding="utf-8",
    )
    (ascend / "memory_record.csv").write_text(
        "\n".join(
            [
                "Timestamp,Peak Memory(MB)",
                "1,32768",
                "2,38912",
                "3,40960",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (ascend / "operator_memory.csv").write_text(
        "\n".join(
            [
                "Operator Name,Peak Memory(MB)",
                "Attention,16384",
                "Embedding,8192",
                "LayerNorm,2048",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (ascend / "npu_module_mem.csv").write_text(
        "\n".join(
            [
                "Module Name,Peak Memory(MB)",
                "encoder.block0,20480",
                "embedding,8192",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (ascend / "dataset.csv").write_text(
        "\n".join(
            [
                "Queue Empty Percent,Wait Time(ms),Dataset Time(ms)",
                "28,14,18",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (ascend / "minddata_pipeline_summary_0.json").write_text(
        json.dumps({"warning": "queue empty events observed", "queue_empty_percent": 28}),
        encoding="utf-8",
    )
    (cann / "op_summary_0.csv").write_text(
        "\n".join(
            [
                "Operator Name,Total Time(ms),Count",
                "AllReduce,132,3",
                "MatMul,66,3",
                "LayerNorm,27,3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return profiler_root


def write_validation_metrics(root: Path) -> tuple[Path, Path]:
    before = root / "before-metrics.json"
    after = root / "after-metrics.json"
    before.write_text(
        json.dumps(
            {
                "metrics": {
                    "throughput": 100.0,
                    "step_time": 86.0,
                    "communication_time": 44.0,
                    "peak_memory": 40960.0,
                }
            }
        ),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(
            {
                "metrics": {
                    "throughput": 112.0,
                    "step_time": 78.0,
                    "communication_time": 31.0,
                    "peak_memory": 40192.0,
                }
            }
        ),
        encoding="utf-8",
    )
    return before, after


def write_fake_pta_profiler_package(root: Path) -> Path:
    package_root = root / "torch_npu"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "profiler.py").write_text(
        """
from pathlib import Path


class ProfilerActivity:
    CPU = "cpu"
    NPU = "npu"


def schedule(**kwargs):
    return kwargs


def tensorboard_trace_handler(path):
    output = Path(path)

    def handler(_prof):
        output.mkdir(parents=True, exist_ok=True)
        (output / "pta-trace.txt").write_text("trace", encoding="utf-8")

    return handler


class _Profile:
    def __init__(self, on_trace_ready=None, **kwargs):
        self.on_trace_ready = on_trace_ready
        self.kwargs = kwargs
        self.steps = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.on_trace_ready is not None:
            self.on_trace_ready(self)
        return False

    def step(self):
        self.steps += 1


def profile(**kwargs):
    return _Profile(**kwargs)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return package_root


def write_fake_mindspore_profiler_package(root: Path) -> Path:
    package_root = root / "mindspore" / "profiler"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root.parent / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "__init__.py").write_text(
        """
from pathlib import Path


class ProfilerActivity:
    CPU = "cpu"
    NPU = "npu"


def schedule(**kwargs):
    return kwargs


def tensorboard_trace_handler(path):
    output = Path(path)

    def handler(_prof):
        output.mkdir(parents=True, exist_ok=True)
        (output / "ms-trace.txt").write_text("trace", encoding="utf-8")

    return handler


class _Profile:
    def __init__(self, on_trace_ready=None, **kwargs):
        self.on_trace_ready = on_trace_ready
        self.kwargs = kwargs
        self.steps = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.on_trace_ready is not None:
            self.on_trace_ready(self)
        return False

    def step(self):
        self.steps += 1


def profile(**kwargs):
    return _Profile(**kwargs)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return package_root.parent


def write_sample_pta_loop_script(root: Path) -> Path:
    script_path = root / "pta_loop.py"
    script_path.write_text(
        """
def train_one_step(step):
    print("step", step)


def main():
    for step in range(2):
        train_one_step(step)


if __name__ == "__main__":
    main()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return script_path


def write_sample_ms_entry_script(root: Path) -> Path:
    script_path = root / "ms_entry.py"
    script_path.write_text(
        """
def run_inference():
    print("inference")


if __name__ == "__main__":
    run_inference()
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return script_path


def write_sample_pta_profiler_export(root: Path) -> Path:
    """Create a PTA (torch_npu) profiler export matching real pro_data structure.

    Based on real profiling data from Ascend NPU (ascend129_*, torch_npu 2.8.0.post2).
    Uses actual kernel/operator naming conventions from Ascend CANN.
    """
    profiler_root = root / "ascend129_1234567_20260328120500000_ascend_pt"
    ascend = profiler_root / "ASCEND_PROFILER_OUTPUT"
    ascend.mkdir(parents=True, exist_ok=True)

    # profiler_info.json - matches real PTA output structure
    (profiler_root / "profiler_info.json").write_text(
        json.dumps({
            "config": {
                "common_config": {
                    "activities": ["ProfilerActivity.CPU", "ProfilerActivity.NPU"],
                    "schedule": {},
                    "record_shapes": False,
                    "profile_memory": False,
                    "with_stack": False,
                    "with_flops": False,
                    "with_modules": False,
                },
                "experimental_config": {
                    "_profiler_level": "Level0",
                    "_aic_metrics": "ACL_AICORE_NONE",
                    "_l2_cache": False,
                    "_data_simplification": True,
                    "_export_type": ["text"],
                },
            },
            "start_info": {
                "syscnt_enable": True,
                "freq": 100,
                "start_cnt": 363500928237873,
                "start_monotonic": 3634797731454060,
            },
            "end_info": {
                "collectionTimeEnd": 1774670954925463870,
                "MonotonicTimeEnd": 3635024364319500,
            },
            "torch_npu_version": "2.8.0.post2",
            "cann_version": "8.5.0",
        }),
        encoding="utf-8",
    )
    (profiler_root / "profiler_metadata.json").write_text(
        json.dumps({
            "ENV_VARIABLES": {
                "ASCEND_GLOBAL_LOG_LEVEL": "",
                "HCCL_RDMA_TC": "",
                "HCCL_RDMA_SL": "",
            },
        }),
        encoding="utf-8",
    )

    # kernel_details.csv - matches real Ascend kernel naming (aclnn* format)
    (ascend / "kernel_details.csv").write_text(
        "\n".join([
            "Device_id,Name,Type,Accelerator Core,Start Time(us),Duration(us),Wait Time(us),Block Dim",
            "0,aclnnMatmul_MatMulCommon_MatMulV2,N/A,N/A,1774670741777000.0,420.50,0.02,8",
            "0,aclnnMul_MulAiCore_Mul,N/A,N/A,1774670741777420.5,18.32,64.742,0",
            "0,aclnnMul_TransposeAiCore_Transpose,N/A,N/A,1774670741777438.8,13.48,18.98,0",
            "0,aclnnAdd_AddAiCore_Add,N/A,N/A,1774670741777452.3,9.30,0.02,0",
            "0,aclnnCast_CastAiCore_Cast,N/A,N/A,1774670741777461.6,7.80,0.15,0",
            "0,aclnnFlashAttentionScore_GetFlashAttentionSrc,Compute,N/A,1774670741777470.0,280.60,0.01,16",
            "0,aclnnFlashAttentionScoreGrad_GetFlashAttentionSrc,Compute,N/A,1774670741777751.0,310.40,0.02,16",
            "0,aclnnInplaceCopy_InplaceCopyAiCore_InplaceCopy,N/A,N/A,1774670741778062.0,12.50,0.10,0",
            "0,aclnnPowTensorScalar_PowTensorScalarAiCore_PowTensorScalar,N/A,N/A,1774670741778074.5,8.20,0.05,0",
            "0,aclnnReduceSum_ReduceSumAiCore_ReduceSum,N/A,N/A,1774670741778083.0,15.60,0.08,0",
            "0,aclnnSilu_SiluAiCore_Silu,N/A,N/A,1774670741778098.6,5.40,0.03,0",
            "0,aclnnRsqrt_RsqrtAiCore_Rsqrt,N/A,N/A,1774670741778104.0,4.20,0.02,0",
        ])
        + "\n",
        encoding="utf-8",
    )

    # operator_details.csv - matches real PTA operator output
    (ascend / "operator_details.csv").write_text(
        "\n".join([
            "Name,Input Shapes,Call Stack,Host Self Duration(us),Host Total Duration(us),Device Self Duration(us),Device Total Duration(us),Device Self Duration With AICore(us),Device Total Duration With AICore(us)",
            'aten::silu_and_mul,"[(1, 2048, 4096), (1, 2048, 11008)]","",45.2,45.2,580.3,580.3,578.1,578.1',
            'aten::mm,"[(1, 4096, 4096), (4096, 4096)]","",12.3,12.3,420.5,420.5,418.9,418.9',
            'aten::flash_attention,"[(1, 32, 2048, 128), (1, 32, 2048, 128)]","",55.6,55.6,591.0,591.0,589.2,589.2',
            'aten::add,"[(1, 2048, 4096), (1, 2048, 4096)]","",8.1,8.1,9.3,9.3,8.8,8.8',
            'aten::mul,"[(1, 2048, 4096), (1, 2048, 4096)]","",7.5,7.5,18.3,18.3,17.9,17.9',
            'aten::layer_norm,"[(1, 2048, 4096)]","",15.8,15.8,28.4,28.4,27.6,27.6',
            'aten::rsqrt,"[(1, 4096)]","",3.2,3.2,4.2,4.2,3.9,3.9',
            'aten::empty,"","",34.6,34.6,0,0,0,0',
            'aten::to,"","",3.2,3.2,0,0,0,0',
            'aten::detach_,"","",7.8,13.5,0,0,0,0',
        ])
        + "\n",
        encoding="utf-8",
    )

    # trace_view.json - realistic PTA Chrome trace format (compact)
    (ascend / "trace_view.json").write_text(
        json.dumps({
            "traceEvents": [
                {"name": "aten::mm", "cat": "cpu_op", "ts": 1774670741777000, "dur": 433000, "tid": 1, "pid": 1},
                {"name": "aclnnMatmul_MatMulCommon_MatMulV2", "cat": "npu_kernel", "ts": 1774670741777100, "dur": 420500, "tid": 2, "pid": 2},
                {"name": "aten::flash_attention", "cat": "cpu_op", "ts": 1774670741777500, "dur": 647000, "tid": 1, "pid": 1},
                {"name": "aclnnFlashAttentionScore_GetFlashAttentionSrc", "cat": "npu_kernel", "ts": 1774670741777600, "dur": 280600, "tid": 2, "pid": 2},
                {"name": "aclnnFlashAttentionScoreGrad_GetFlashAttentionSrc", "cat": "npu_kernel", "ts": 1774670741777900, "dur": 310400, "tid": 2, "pid": 2},
                {"name": "aten::silu_and_mul", "cat": "cpu_op", "ts": 1774670741778200, "dur": 626000, "tid": 1, "pid": 1},
                {"name": "aclnnMul_MulAiCore_Mul", "cat": "npu_kernel", "ts": 1774670741778250, "dur": 18320, "tid": 2, "pid": 2},
                {"name": "aclnnSilu_SiluAiCore_Silu", "cat": "npu_kernel", "ts": 1774670741778270, "dur": 5400, "tid": 2, "pid": 2},
                {"name": "aten::layer_norm", "cat": "cpu_op", "ts": 1774670741778900, "dur": 44200, "tid": 1, "pid": 1},
                {"name": "aclnnRsqrt_RsqrtAiCore_Rsqrt", "cat": "npu_kernel", "ts": 1774670741778910, "dur": 4200, "tid": 2, "pid": 2},
                {"name": "aten::add", "cat": "cpu_op", "ts": 1774670741778950, "dur": 17400, "tid": 1, "pid": 1},
                {"name": "aclnnAdd_AddAiCore_Add", "cat": "npu_kernel", "ts": 1774670741778960, "dur": 9300, "tid": 2, "pid": 2},
                {"name": "aten::empty", "cat": "cpu_op", "ts": 1774670741779100, "dur": 34600, "tid": 1, "pid": 1},
            ],
        }),
        encoding="utf-8",
    )

    # analysis.db placeholder (real one is SQLite, tests just need the file to exist)
    (ascend / "analysis.db").write_text("", encoding="utf-8")

    return profiler_root
