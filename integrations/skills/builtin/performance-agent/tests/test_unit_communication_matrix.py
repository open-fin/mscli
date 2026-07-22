"""Unit tests for analyze_communication_matrix.py."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import write_sample_profiler_export, run_script, ROOT, SCRIPTS

# Allow importlib to resolve `from perf_common import ...`
sys.path.insert(0, str(SCRIPTS))


def _write_matrix_json(
    root: Path,
    links: list[dict],
) -> Path:
    """Write a communication_matrix.json with given link records."""
    matrix_path = root / "communication_matrix.json"
    matrix_path.write_text(json.dumps({"matrix": links}), encoding="utf-8")
    return matrix_path


def _make_link_records(num_ranks: int = 8, bandwidth: float = 50.0) -> list[dict]:
    """Generate intra-node HCCS link records for an 8-NPU topology."""
    records = []
    for src in range(num_ranks):
        for dst in range(src + 1, min(src + 2, num_ranks)):
            records.append({
                "src_rank": src,
                "dst_rank": dst,
                "bandwidth_gb_s": bandwidth,
                "transit_time_ms": 10.0,
            })
    return records


def test_classify_link_type_hccs_intra_ring():
    """Ranks within the same ring (0-3 or 4-7) should classify as hccs_intra_ring."""
    script = SCRIPTS / "analyze_communication_matrix.py"
    # Import the function directly for unit testing
    import importlib.util
    spec = importlib.util.spec_from_file_location("acm", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.classify_link_type(0, 1, 8) == "hccs_intra_ring"
    assert mod.classify_link_type(4, 5, 8) == "hccs_intra_ring"


def test_classify_link_type_pcie_cross_ring():
    """Cross-ring (ring 0 <-> ring 1) should classify as pcie_cross_ring."""
    script = SCRIPTS / "analyze_communication_matrix.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("acm", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.classify_link_type(0, 5, 8) == "pcie_cross_ring"
    assert mod.classify_link_type(3, 4, 8) == "pcie_cross_ring"


def test_classify_link_type_rdma_inter_node():
    """Cross-node links should classify as rdma_inter_node."""
    script = SCRIPTS / "analyze_communication_matrix.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("acm", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.classify_link_type(0, 8, 16) == "rdma_inter_node"
    assert mod.classify_link_type(7, 15, 16) == "rdma_inter_node"


def test_extract_link_bandwidths_from_list():
    """Extract links from a flat array format."""
    script = SCRIPTS / "analyze_communication_matrix.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("acm", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    data = [
        {"src_rank": 0, "dst_rank": 1, "bandwidth_gb_s": 50.0},
        {"src_rank": 1, "dst_rank": 2, "bandwidth_gb_s": 48.0},
    ]
    links = mod.extract_link_bandwidths(data)
    assert len(links) == 2
    assert links[0]["src_rank"] == 0
    assert links[0]["bandwidth_gb_s"] == 50.0


def test_extract_link_bandwidths_from_nested_dict():
    """Extract links from a nested dict with 'matrix' key."""
    script = SCRIPTS / "analyze_communication_matrix.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("acm", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    data = {
        "matrix": [
            {"src_rank": 0, "dst_rank": 1, "bandwidth_gb_s": 55.0},
        ]
    }
    links = mod.extract_link_bandwidths(data)
    assert len(links) == 1
    assert links[0]["bandwidth_gb_s"] == 55.0


def test_detect_slow_links_flags_outlier():
    """A single link with much lower bandwidth should be flagged as slow."""
    script = SCRIPTS / "analyze_communication_matrix.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("acm", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 5 normal HCCS links at ~50 GB/s, 1 degraded at 10 GB/s
    links = [
        {"src_rank": 0, "dst_rank": 1, "bandwidth_gb_s": 50.0},
        {"src_rank": 1, "dst_rank": 2, "bandwidth_gb_s": 52.0},
        {"src_rank": 2, "dst_rank": 3, "bandwidth_gb_s": 49.0},
        {"src_rank": 3, "dst_rank": 0, "bandwidth_gb_s": 51.0},
        {"src_rank": 4, "dst_rank": 5, "bandwidth_gb_s": 48.0},
        {"src_rank": 5, "dst_rank": 6, "bandwidth_gb_s": 10.0},  # degraded
    ]
    slow = mod.detect_slow_links(links, world_size=8)
    assert len(slow) == 1
    # The degraded link (rank 5→6) should be flagged
    flagged_pairs = {(s["src_rank"], s["dst_rank"]) for s in slow}
    assert (5, 6) in flagged_pairs


def test_detect_slow_links_no_outliers():
    """Uniform bandwidth should produce no slow links."""
    script = SCRIPTS / "analyze_communication_matrix.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("acm", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    links = _make_link_records(num_ranks=4, bandwidth=50.0)
    slow = mod.detect_slow_links(links, world_size=4)
    assert len(slow) == 0


def test_full_pipeline_with_slow_link(tmp_path: Path):
    """Full CLI pipeline: detect slow link from communication_matrix.json."""
    # Create profiler structure with communication_matrix.json
    profiler_root = write_sample_profiler_export(tmp_path)
    ascend_dir = profiler_root / "ASCEND_PROFILER_OUTPUT"

    # Write a matrix with one degraded link
    matrix_data = {
        "matrix": [
            {"src_rank": 0, "dst_rank": 1, "bandwidth_gb_s": 50.0, "transit_time_ms": 5.0},
            {"src_rank": 1, "dst_rank": 2, "bandwidth_gb_s": 51.0, "transit_time_ms": 5.0},
            {"src_rank": 0, "dst_rank": 2, "bandwidth_gb_s": 8.0, "transit_time_ms": 30.0},  # slow
            {"src_rank": 1, "dst_rank": 3, "bandwidth_gb_s": 49.0, "transit_time_ms": 5.0},
            {"src_rank": 2, "dst_rank": 3, "bandwidth_gb_s": 52.0, "transit_time_ms": 5.0},
            {"src_rank": 0, "dst_rank": 3, "bandwidth_gb_s": 50.0, "transit_time_ms": 5.0},
        ]
    }
    (ascend_dir / "communication_matrix.json").write_text(
        json.dumps(matrix_data), encoding="utf-8"
    )

    output_json = tmp_path / "comm_matrix_result.json"
    run_script(
        "analyze_communication_matrix.py",
        "--trace-root", str(profiler_root),
        "--output-json", str(output_json),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["total_links_analyzed"] > 0
    assert "link_type_summary" in result
    assert "slow_links" in result
    assert "hccl_suggestions" in result
    # The 8 GB/s link should be flagged
    assert result["slow_link_count"] == 1


def test_full_pipeline_no_slow_links(tmp_path: Path):
    """Full CLI pipeline with uniform bandwidth — no slow links detected."""
    profiler_root = write_sample_profiler_export(tmp_path)
    ascend_dir = profiler_root / "ASCEND_PROFILER_OUTPUT"

    matrix_data = {
        "matrix": [
            {"src_rank": 0, "dst_rank": 1, "bandwidth_gb_s": 50.0, "transit_time_ms": 5.0},
            {"src_rank": 1, "dst_rank": 2, "bandwidth_gb_s": 50.5, "transit_time_ms": 5.0},
            {"src_rank": 2, "dst_rank": 3, "bandwidth_gb_s": 49.5, "transit_time_ms": 5.0},
        ]
    }
    (ascend_dir / "communication_matrix.json").write_text(
        json.dumps(matrix_data), encoding="utf-8"
    )

    output_json = tmp_path / "comm_matrix_result.json"
    run_script(
        "analyze_communication_matrix.py",
        "--trace-root", str(profiler_root),
        "--output-json", str(output_json),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["slow_link_count"] == 0


def test_no_matrix_json_exits_with_error(tmp_path: Path):
    """Missing communication_matrix.json should exit with error."""
    profiler_root = write_sample_profiler_export(tmp_path)
    # Remove the default communication_matrix.json
    matrix_file = profiler_root / "ASCEND_PROFILER_OUTPUT" / "communication_matrix.json"
    if matrix_file.exists():
        matrix_file.unlink()

    output_json = tmp_path / "comm_matrix_result.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "analyze_communication_matrix.py"),
         "--trace-root", str(profiler_root), "--output-json", str(output_json)],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
