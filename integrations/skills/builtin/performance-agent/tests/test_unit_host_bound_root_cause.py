"""Unit tests for analyze_host_bound_root_cause.py."""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import ROOT, SCRIPTS

sys.path.insert(0, str(SCRIPTS))

from analyze_host_bound_root_cause import (
    classify_op,
    determine_root_cause,
    render_report,
    CATEGORY_RULES,
)


# ---------------------------------------------------------------------------
# classify_op tests
# ---------------------------------------------------------------------------

class TestClassifyOp:
    def test_checkpoint_forward(self):
        assert classify_op("CheckpointFunction") == "checkpoint"

    def test_checkpoint_backward(self):
        assert classify_op("CheckpointFunctionBackward") == "checkpoint"

    def test_autograd_evaluate(self):
        assert classify_op("autograd::engine::evaluate_function: MulBackward0") == "autograd_engine"

    def test_autograd_other(self):
        assert classify_op("autograd::AccumulateGrad") == "autograd_other"

    def test_backward(self):
        assert classify_op("MatmulBackward0") == "backward"
        assert classify_op("SliceBackward0") == "backward"

    def test_dataloader(self):
        assert classify_op("enumerate(DataLoader)#_MultiProcessingDataLoaderIter") == "dataloader"

    def test_io_save(self):
        assert classify_op("torch.save") == "io_save"
        assert classify_op("state_dict") == "io_save"

    def test_io_log(self):
        assert classify_op("add_scalar") == "io_log"
        assert classify_op("histogram") == "io_log"

    def test_dtype_cast(self):
        assert classify_op("aten::to") == "dtype_cast"
        assert classify_op("aten::_to_copy") == "dtype_cast"
        assert classify_op("_npu_dtype_cast") == "dtype_cast"

    def test_optimizer(self):
        assert classify_op("Optimizer.step#AdamW.step") == "optimizer"
        assert classify_op("aten::_foreach_norm") == "optimizer"
        assert classify_op("clip_grad_norm_") == "optimizer"

    def test_loss(self):
        assert classify_op("aten::cross_entropy_loss") == "loss"
        assert classify_op("aten::nll_loss") == "loss"

    def test_attention(self):
        assert classify_op("aten::scaled_dot_product_attention") == "attention"

    def test_matmul_linear(self):
        assert classify_op("aten::linear") == "matmul_linear"
        assert classify_op("aten::matmul") == "matmul_linear"
        assert classify_op("aten::mm") == "matmul_linear"
        assert classify_op("aten::addmm") == "matmul_linear"

    def test_embedding(self):
        assert classify_op("aten::embedding") == "embedding"
        assert classify_op("aten::embedding_backward") == "embedding"

    def test_norm(self):
        assert classify_op("aten::layer_norm") == "norm"
        assert classify_op("aten::native_layer_norm") == "norm"

    def test_activation(self):
        assert classify_op("aten::silu") == "activation"
        assert classify_op("aten::gelu") == "activation"

    def test_copy_tensor(self):
        assert classify_op("aten::copy_") == "copy_tensor"
        assert classify_op("aten::clone") == "copy_tensor"
        assert classify_op("aten::reshape") == "copy_tensor"
        assert classify_op("aten::view") == "copy_tensor"

    def test_empty_tensor(self):
        assert classify_op("empty_tensor") == "empty_tensor"

    def test_aten_dispatch(self):
        assert classify_op("aten::mul") == "aten_dispatch"
        assert classify_op("aten::pow") == "aten_dispatch"
        assert classify_op("aten::slice_backward") == "aten_dispatch"

    def test_framework_catchall(self):
        assert classify_op("torch::autograd::AccumulateGrad") == "framework_overhead"
        assert classify_op("some_random_op") == "framework_overhead"

    def test_all_categories_have_rules(self):
        """Ensure CATEGORY_RULES covers expected categories."""
        cats = {r[0] for r in CATEGORY_RULES}
        expected = {
            "checkpoint", "autograd_engine", "autograd_other", "backward",
            "dataloader", "io_save", "io_log", "dtype_cast", "optimizer",
            "loss", "attention", "matmul_linear", "embedding", "norm",
            "activation", "copy_tensor", "empty_tensor", "aten_dispatch",
            "framework_overhead",
        }
        assert cats == expected


# ---------------------------------------------------------------------------
# determine_root_cause tests
# ---------------------------------------------------------------------------

def _make_cat_summary(entries):
    """Helper: build cpu_category_breakdown entries."""
    return [{"category": e[0], "total_ms": e[1], "share_pct": e[2],
             "calls": e[3], "top_ops": []} for e in entries]


class TestDetermineRootCause:
    def test_dispatch_overhead_critical(self):
        cat_summary = _make_cat_summary([
            ("aten_dispatch", 60000, 60, 4000000),
            ("checkpoint", 10000, 10, 5000),
        ])
        dd = {
            "total_cpu_calls": 5000000,
            "cpu_ops_per_sec": 120000,
            "avg_dispatch_us": 15,
            "high_freq_small_ops": {
                "count": 50, "total_calls": 4800000, "total_ms": 120000,
                "calls_pct": 96.0, "top_ops": [],
            },
        }
        rc = determine_root_cause(
            cat_summary, {}, dd, [],
            total_cpu_ns=100_000_000_000,
            total_queue_ns=30_000_000_000,
            wall_ns=100_000_000_000,
        )
        assert rc["primary_cause"] == "operator_dispatch_overhead"
        assert rc["primary_severity"] == "CRITICAL"
        assert any(c["cause"] == "operator_dispatch_overhead" for c in rc["all_causes"])

    def test_framework_overhead_high(self):
        cat_summary = _make_cat_summary([
            ("checkpoint", 200000, 48, 25000),
            ("autograd_engine", 50000, 12, 800000),
            ("backward", 30000, 7, 600000),
            ("aten_dispatch", 80000, 19, 3000000),
        ])
        dd = {
            "total_cpu_calls": 5000000,
            "cpu_ops_per_sec": 50000,
            "avg_dispatch_us": 80,
            "high_freq_small_ops": {
                "count": 10, "total_calls": 500000, "total_ms": 10000,
                "calls_pct": 10.0, "top_ops": [],
            },
        }
        rc = determine_root_cause(
            cat_summary, {}, dd, [],
            total_cpu_ns=416_000_000_000,
            total_queue_ns=40_000_000_000,
            wall_ns=125_000_000_000,
        )
        assert rc["primary_cause"] == "framework_overhead"
        assert rc["primary_severity"] == "HIGH"

    def test_slow_io_data(self):
        cat_summary = _make_cat_summary([
            ("aten_dispatch", 30000, 40, 2000000),
            ("dataloader", 5000, 7, 100),
        ])
        slow_ops = [
            {"name": "enumerate(DataLoader)#fetch", "category": "dataloader", "duration_ms": 200},
            {"name": "torch.save", "category": "io_save", "duration_ms": 600},
            {"name": "aten::add", "category": "aten_dispatch", "duration_ms": 1},
        ]
        dd = {
            "total_cpu_calls": 2000000,
            "cpu_ops_per_sec": 20000,
            "avg_dispatch_us": 50,
            "high_freq_small_ops": {
                "count": 5, "total_calls": 500000, "total_ms": 5000,
                "calls_pct": 25.0, "top_ops": [],
            },
        }
        rc = determine_root_cause(
            cat_summary, {}, dd, slow_ops,
            total_cpu_ns=75_000_000_000,
            total_queue_ns=40_000_000_000,
            wall_ns=100_000_000_000,
        )
        assert any(c["cause"] == "slow_io_data" for c in rc["all_causes"])
        io_cause = next(c for c in rc["all_causes"] if c["cause"] == "slow_io_data")
        assert io_cause["severity"] == "HIGH"  # 600ms > 500ms

    def test_dtype_cast_overhead(self):
        cat_summary = _make_cat_summary([
            ("aten_dispatch", 50000, 50, 2000000),
            ("dtype_cast", 8000, 8, 300000),
        ])
        dd = {
            "total_cpu_calls": 3000000,
            "cpu_ops_per_sec": 30000,
            "avg_dispatch_us": 60,
            "high_freq_small_ops": {
                "count": 5, "total_calls": 500000, "total_ms": 5000,
                "calls_pct": 16.7, "top_ops": [],
            },
        }
        rc = determine_root_cause(
            cat_summary, {}, dd, [],
            total_cpu_ns=100_000_000_000,
            total_queue_ns=40_000_000_000,
            wall_ns=100_000_000_000,
        )
        assert any(c["cause"] == "dtype_cast" for c in rc["all_causes"])

    def test_no_clear_bottleneck_falls_to_general(self):
        cat_summary = _make_cat_summary([
            ("aten_dispatch", 20000, 30, 1000000),
            ("copy_tensor", 10000, 15, 500000),
        ])
        dd = {
            "total_cpu_calls": 2000000,
            "cpu_ops_per_sec": 20000,
            "avg_dispatch_us": 60,
            "high_freq_small_ops": {
                "count": 3, "total_calls": 100000, "total_ms": 2000,
                "calls_pct": 5.0, "top_ops": [],
            },
        }
        rc = determine_root_cause(
            cat_summary, {}, dd, [],
            total_cpu_ns=67_000_000_000,
            total_queue_ns=40_000_000_000,
            wall_ns=100_000_000_000,
        )
        assert rc["primary_cause"] == "general_host_overhead"

    def test_causes_sorted_by_severity(self):
        cat_summary = _make_cat_summary([
            ("checkpoint", 200000, 50, 25000),
            ("dtype_cast", 8000, 2, 300000),
            ("aten_dispatch", 80000, 20, 4000000),
        ])
        dd = {
            "total_cpu_calls": 5000000,
            "cpu_ops_per_sec": 120000,
            "avg_dispatch_us": 15,
            "high_freq_small_ops": {
                "count": 50, "total_calls": 4800000, "total_ms": 120000,
                "calls_pct": 96.0, "top_ops": [],
            },
        }
        rc = determine_root_cause(
            cat_summary, {}, dd,
            [{"name": "torch.save", "category": "io_save", "duration_ms": 600}],
            total_cpu_ns=400_000_000_000,
            total_queue_ns=40_000_000_000,
            wall_ns=125_000_000_000,
        )
        sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        severities = [sev_order.get(c["severity"], 99) for c in rc["all_causes"]]
        assert severities == sorted(severities)


# ---------------------------------------------------------------------------
# render_report tests
# ---------------------------------------------------------------------------

class TestRenderReport:
    def test_report_contains_root_cause(self):
        data = {
            "wall_time_s": 120.0,
            "cpu_op_time_s": 400.0,
            "device_op_time_s": 40.0,
            "device_coverage_pct": 33.3,
            "idle_pct": 66.7,
            "cpu_category_breakdown": [
                {"category": "checkpoint", "total_ms": 200000, "share_pct": 50,
                 "calls": 25000, "top_ops": []},
            ],
            "dispatch_density": {
                "total_cpu_calls": 5000000,
                "cpu_ops_per_sec": 100000,
                "avg_dispatch_us": 25,
                "high_freq_small_ops": {
                    "count": 50, "total_calls": 4500000, "total_ms": 100000,
                    "calls_pct": 90.0, "top_ops": [
                        {"name": "empty_tensor", "calls": 2000000, "total_ms": 10000, "avg_us": 5},
                    ],
                },
            },
            "gap_analysis": {},
            "slow_individual_ops": [
                {"name": "aten::to", "category": "dtype_cast", "duration_ms": 500},
            ],
            "root_cause_diagnosis": {
                "primary_cause": "operator_dispatch_overhead",
                "primary_severity": "CRITICAL",
                "summary": "Test summary",
                "all_causes": [
                    {"cause": "operator_dispatch_overhead", "severity": "CRITICAL",
                     "summary": "Test summary", "evidence": ["e1"], "optimization_priority": "build_fused_ops"},
                ],
            },
        }
        report = render_report(data)
        assert "operator_dispatch_overhead" in report
        assert "CRITICAL" in report
        assert "CPU Time by Category" in report
        assert "Dispatch Density" in report
        assert "Fused Operator Recommendations" in report
        assert "empty_tensor" in report

    def test_report_no_gap_analysis(self):
        data = {
            "wall_time_s": 120.0,
            "cpu_op_time_s": 400.0,
            "device_op_time_s": 40.0,
            "device_coverage_pct": 33.3,
            "idle_pct": 66.7,
            "cpu_category_breakdown": [],
            "dispatch_density": {
                "total_cpu_calls": 100,
                "cpu_ops_per_sec": 1000,
                "avg_dispatch_us": 50,
                "high_freq_small_ops": {"count": 0, "total_calls": 0, "total_ms": 0,
                                         "calls_pct": 0.0, "top_ops": []},
            },
            "gap_analysis": {},
            "slow_individual_ops": [],
            "root_cause_diagnosis": {
                "primary_cause": "general_host_overhead",
                "primary_severity": "MEDIUM",
                "summary": "No clear bottleneck",
                "all_causes": [],
            },
        }
        report = render_report(data)
        assert "general_host_overhead" in report
        assert "Device-Idle Gap" not in report  # no gap data


# ---------------------------------------------------------------------------
# Integration test with synthetic SQLite DB
# ---------------------------------------------------------------------------

def _create_test_db(db_path: Path, ops_data: list[dict]):
    """Create a minimal SQLite DB matching the PTA profiler schema.

    ops_data: list of {name, type, start_ns, end_ns, tid}
    """
    db = sqlite3.connect(str(db_path))
    cur = db.cursor()

    cur.execute("""CREATE TABLE STRING_IDS (id INTEGER PRIMARY KEY, value TEXT)""")
    cur.execute("""CREATE TABLE ENUM_API_TYPE (id INTEGER PRIMARY KEY, name TEXT)""")
    cur.execute("""CREATE TABLE PYTORCH_API (
        startNs TEXT, endNs TEXT, globalTid INTEGER,
        connectionId INTEGER, name INTEGER, sequenceNumber INTEGER,
        fwdThreadId INTEGER, inputDtypes TEXT, inputShapes TEXT,
        callchainId INTEGER, type INTEGER
    )""")
    cur.execute("""CREATE TABLE HOST_INFO (hostUid TEXT, hostName TEXT)""")
    cur.execute("""CREATE TABLE META_DATA (name TEXT, value TEXT)""")
    cur.execute("""CREATE TABLE RANK_DEVICE_MAP (rankId INTEGER, deviceId INTEGER)""")
    cur.execute("""CREATE TABLE CONNECTION_IDS (id INTEGER, connectionId INTEGER)""")

    cur.execute("INSERT INTO ENUM_API_TYPE VALUES (50001, 'op')")
    cur.execute("INSERT INTO ENUM_API_TYPE VALUES (50002, 'queue')")

    # Build string map
    str_map = {}
    next_id = 268435456
    for op in ops_data:
        if op["name"] not in str_map:
            str_map[op["name"]] = next_id
            cur.execute("INSERT INTO STRING_IDS VALUES (?, ?)", (next_id, op["name"]))
            next_id += 1

    # Insert ops
    for op in ops_data:
        name_id = str_map[op["name"]]
        cur.execute(
            "INSERT INTO PYTORCH_API (startNs, endNs, globalTid, connectionId, name, sequenceNumber, fwdThreadId, inputDtypes, inputShapes, callchainId, type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(op["start_ns"]), str(op["end_ns"]), op.get("tid", 1), 0,
             name_id, 0, 0, "", "", 0, op["type"]),
        )

    db.commit()
    db.close()
    return str_map


def test_integration_dispatch_overhead(tmp_path: Path):
    """Test full analysis with dispatch-heavy CPU ops → operator_dispatch_overhead."""
    db_path = tmp_path / "test.db"
    base_ns = 1_000_000_000_000  # 1M seconds in ns

    ops = []
    # Many small CPU ops (dispatch pressure)
    for i in range(5000):
        ops.append({"name": "aten::mul", "type": 50001,
                     "start_ns": base_ns + i * 20000, "end_ns": base_ns + i * 20000 + 10000})
    for i in range(3000):
        ops.append({"name": "aten::add", "type": 50001,
                     "start_ns": base_ns + i * 20000, "end_ns": base_ns + i * 20000 + 8000})
    for i in range(10000):
        ops.append({"name": "empty_tensor", "type": 50001,
                     "start_ns": base_ns + i * 10000, "end_ns": base_ns + i * 10000 + 5000})
    # Some device ops (low coverage)
    for i in range(500):
        ops.append({"name": "Dequeue@aclnnMul", "type": 50002,
                     "start_ns": base_ns + i * 200000, "end_ns": base_ns + i * 200000 + 100000})
    # CheckpointFunction for step boundary detection
    for i in range(10):
        ops.append({"name": "CheckpointFunction", "type": 50001,
                     "start_ns": base_ns + i * 100_000_000, "end_ns": base_ns + i * 100_000_000 + 1_000_000})

    _create_test_db(db_path, ops)

    # Create profiler directory structure
    profiler_dir = tmp_path / "test_ascend_pt"
    ascend = profiler_dir / "ASCEND_PROFILER_OUTPUT"
    ascend.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(str(db_path), str(ascend / "ascend_pytorch_profiler.db"))

    out_dir = profiler_dir / "out"
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    from analyze_host_bound_root_cause import analyze_host_bound
    result = analyze_host_bound(ascend / "ascend_pytorch_profiler.db", out_dir)

    # Verify structure
    assert "root_cause_diagnosis" in result
    assert "cpu_category_breakdown" in result
    assert "dispatch_density" in result
    assert "wall_time_s" in result
    assert result["wall_time_s"] > 0
    assert result["device_coverage_pct"] < 50  # host bound

    # Verify category breakdown has expected categories
    cats = {c["category"] for c in result["cpu_category_breakdown"]}
    assert "aten_dispatch" in cats or "empty_tensor" in cats

    # Verify output files
    assert (out_dir / "host_bound_analysis.md").exists()
    assert (meta_dir / "host_bound_analysis.json").exists()

    report = (out_dir / "host_bound_analysis.md").read_text(encoding="utf-8")
    assert "Root Cause" in report
    assert "CPU Time by Category" in report


def test_integration_framework_overhead(tmp_path: Path):
    """Test with checkpoint/autograd heavy profile → framework_overhead."""
    db_path = tmp_path / "test.db"
    base_ns = 1_000_000_000_000

    ops = []
    # Heavy checkpoint + autograd ops
    for i in range(100):
        ops.append({"name": "CheckpointFunctionBackward", "type": 50001,
                     "start_ns": base_ns + i * 10_000_000, "end_ns": base_ns + i * 10_000_000 + 5_000_000})
        ops.append({"name": "autograd::engine::evaluate_function: CheckpointFunctionBackward", "type": 50001,
                     "start_ns": base_ns + i * 10_000_000, "end_ns": base_ns + i * 10_000_000 + 5_000_000})
        ops.append({"name": "CheckpointFunction", "type": 50001,
                     "start_ns": base_ns + i * 10_000_000 + 5_000_000, "end_ns": base_ns + i * 10_000_000 + 6_000_000})
    # Few device ops
    for i in range(50):
        ops.append({"name": "Dequeue@aclnnMatmul", "type": 50002,
                     "start_ns": base_ns + i * 20_000_000, "end_ns": base_ns + i * 20_000_000 + 5_000_000})

    _create_test_db(db_path, ops)

    profiler_dir = tmp_path / "test2_ascend_pt"
    ascend = profiler_dir / "ASCEND_PROFILER_OUTPUT"
    ascend.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(str(db_path), str(ascend / "ascend_pytorch_profiler.db"))

    out_dir = profiler_dir / "out"
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    from analyze_host_bound_root_cause import analyze_host_bound
    result = analyze_host_bound(ascend / "ascend_pytorch_profiler.db", out_dir)

    rc = result["root_cause_diagnosis"]
    # Framework overhead should be detected
    all_causes = [c["cause"] for c in rc["all_causes"]]
    assert "framework_overhead" in all_causes

    cat_map = {c["category"]: c for c in result["cpu_category_breakdown"]}
    assert "checkpoint" in cat_map
    assert cat_map["checkpoint"]["share_pct"] > 30  # checkpoint dominates


def test_integration_output_artifacts(tmp_path: Path):
    """Verify all expected output artifacts are produced."""
    db_path = tmp_path / "test.db"
    base_ns = 1_000_000_000_000

    ops = [
        {"name": "aten::mul", "type": 50001, "start_ns": base_ns, "end_ns": base_ns + 10000},
        {"name": "Dequeue@aclnnMul", "type": 50002, "start_ns": base_ns, "end_ns": base_ns + 50000},
        {"name": "CheckpointFunction", "type": 50001, "start_ns": base_ns, "end_ns": base_ns + 1000},
    ]
    _create_test_db(db_path, ops)

    profiler_dir = tmp_path / "test3_ascend_pt"
    ascend = profiler_dir / "ASCEND_PROFILER_OUTPUT"
    ascend.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(str(db_path), str(ascend / "ascend_pytorch_profiler.db"))

    out_dir = profiler_dir / "out"
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    from analyze_host_bound_root_cause import analyze_host_bound
    result = analyze_host_bound(ascend / "ascend_pytorch_profiler.db", out_dir)

    # Check all artifact files exist
    assert (out_dir / "host_bound_analysis.md").exists()
    assert (meta_dir / "host_bound_analysis.json").exists()

    # JSON is valid
    json_data = json.loads((meta_dir / "host_bound_analysis.json").read_text(encoding="utf-8"))
    assert "root_cause_diagnosis" in json_data
    assert "cpu_category_breakdown" in json_data
    assert "dispatch_density" in json_data
    assert "wall_time_s" in json_data
