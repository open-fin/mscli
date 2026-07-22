"""Unit tests for recommend_parallel_strategy.py."""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_test_utils import run_script, ROOT, SCRIPTS

# Allow importlib to resolve `from perf_common import ...`
sys.path.insert(0, str(SCRIPTS))


# ---------------------------------------------------------------------------
# Pure function unit tests (import module directly)
# ---------------------------------------------------------------------------

def _load_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rps", SCRIPTS / "recommend_parallel_strategy.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_estimate_model_size_gb():
    """Verify model size estimation for a standard transformer."""
    mod = _load_module()
    # Standard transformer: 32 layers, 4096 hidden, 32K vocab, bf16
    size = mod.estimate_model_size_gb(
        num_layers=32, hidden_size=4096, vocab_size=32000,
        seq_len=2048, precision_bytes=2,
    )
    # 12 * 4096^2 * 32 + 32000 * 4096 ≈ 6.44B params * 2 bytes ≈ 12.0 GB
    assert 10.0 < size < 15.0, f"Expected ~12GB for standard 7B model, got {size}"


def test_estimate_model_size_swiglu():
    """SwiGLU with explicit intermediate_size should produce known model size."""
    mod = _load_module()
    # LLaMA-7B: intermediate_size=11008 (the actual value, not 8H/3)
    swiglu = mod.estimate_model_size_gb(
        num_layers=32, hidden_size=4096, vocab_size=32000,
        seq_len=2048, precision_bytes=2, ffn_type="swiglu",
        intermediate_size=11008,
    )
    # LLaMA-7B: attention 4*H²=67M, SwiGLU FFN 3*H*11008=135.3M, per layer ≈ 202.3M
    # 32 layers + embedding ≈ 6.74B params * 2 bytes ≈ 12.5 GB
    assert 11.0 < swiglu < 14.0, f"Expected ~12-13GB for LLaMA-7B SwiGLU, got {swiglu}"

    # SwiGLU with intermediate_size=4H should be larger than standard (same 4H)
    standard = mod.estimate_model_size_gb(
        num_layers=4, hidden_size=1024, vocab_size=0,
        seq_len=512, precision_bytes=2, ffn_type="standard",
        intermediate_size=4096,
    )
    swiglu_explicit = mod.estimate_model_size_gb(
        num_layers=4, hidden_size=1024, vocab_size=0,
        seq_len=512, precision_bytes=2, ffn_type="swiglu",
        intermediate_size=4096,
    )
    # SwiGLU: 3 projections vs standard: 2 projections → more params
    assert swiglu_explicit > standard, f"SwiGLU ({swiglu_explicit:.4f}GB) > standard ({standard:.4f}GB)"


def test_estimate_model_size_gqa():
    """GQA should produce smaller model than MHA."""
    mod = _load_module()
    mha = mod.estimate_model_size_gb(
        num_layers=32, hidden_size=4096, vocab_size=32000,
        seq_len=2048, precision_bytes=2,
        num_attention_heads=32, num_kv_heads=32,
    )
    gqa = mod.estimate_model_size_gb(
        num_layers=32, hidden_size=4096, vocab_size=32000,
        seq_len=2048, precision_bytes=2,
        num_attention_heads=32, num_kv_heads=8,
    )
    assert gqa < mha, f"GQA ({gqa:.2f}GB) should be < MHA ({mha:.2f}GB)"


def test_estimate_model_size_no_embedding():
    """Without embedding, size should be smaller."""
    mod = _load_module()
    with_emb = mod.estimate_model_size_gb(
        num_layers=4, hidden_size=1024, vocab_size=32000,
        seq_len=512, include_embedding=True,
    )
    without_emb = mod.estimate_model_size_gb(
        num_layers=4, hidden_size=1024, vocab_size=32000,
        seq_len=512, include_embedding=False,
    )
    assert without_emb < with_emb


def test_recommend_tp_size_fits_single_device():
    """Small model should recommend TP=1."""
    mod = _load_module()
    result = mod.recommend_tp_size(
        model_size_gb=2.0, hbm_per_npu_gb=64.0, npus_per_node=8,
    )
    assert result["recommended_tp"] == 1
    assert "fits" in result["reason"].lower()


def test_recommend_tp_size_needs_sharding():
    """Large model should recommend TP > 1."""
    mod = _load_module()
    result = mod.recommend_tp_size(
        model_size_gb=60.0, hbm_per_npu_gb=64.0, npus_per_node=8,
        activation_gb=10.0,
    )
    assert result["recommended_tp"] > 1
    # 60*1.5 + 10 = 100GB → needs at least TP=2 to fit in 64GB


def test_recommend_pp_size_tp_sufficient():
    """When TP alone is sufficient, PP should be 1."""
    mod = _load_module()
    result = mod.recommend_pp_size(
        model_size_gb=2.0, hbm_per_npu_gb=64.0, tp_size=1,
        num_nodes=2,
    )
    assert result["recommended_pp"] == 1


def test_recommend_zero_stage_no_dp():
    """ZeRO requires DP >= 2, so single device should return stage 0."""
    mod = _load_module()
    result = mod.recommend_zero_stage(
        model_size_gb=10.0, hbm_per_npu_gb=64.0,
        tp_size=1, pp_size=1, dp_size=1,
    )
    assert result["recommended_stage"] == 0


def test_recommend_zero_stage_with_dp():
    """With DP >= 2, ZeRO-1 may be recommended if optimizer states are large."""
    mod = _load_module()
    result = mod.recommend_zero_stage(
        model_size_gb=50.0, hbm_per_npu_gb=64.0,
        tp_size=1, pp_size=1, dp_size=8,
    )
    # Memory is tight (50*1.5=75 > 64), so ZeRO should be recommended
    assert result["recommended_stage"] >= 1


def test_recommend_recomputation_not_needed():
    """When memory headroom is sufficient, recomputation should not be recommended."""
    mod = _load_module()
    result = mod.recommend_recomputation(
        activation_gb=5.0, hbm_per_npu_gb=64.0, mem_per_device_gb=20.0,
    )
    assert result["recommended"] is False


def test_recommend_recomputation_needed():
    """When memory usage exceeds 80% of HBM, recomputation should be recommended."""
    mod = _load_module()
    result = mod.recommend_recomputation(
        activation_gb=20.0, hbm_per_npu_gb=64.0, mem_per_device_gb=60.0,
    )
    assert result["recommended"] is True
    assert result["activation_saving_gb"] > 0


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

def test_full_pipeline_small_model(tmp_path: Path):
    """Small model on single node: TP=1, PP=1, no ZeRO, no recompute."""
    output_json = tmp_path / "strategy.json"
    run_script(
        "recommend_parallel_strategy.py",
        "--num-layers", "4",
        "--hidden-size", "512",
        "--seq-len", "512",
        "--batch-size", "4",
        "--vocab-size", "32000",
        "--hardware", "ascend_910b2",
        "--output-json", str(output_json),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert result["strategy_summary"]["tp_size"] == 1
    assert result["strategy_summary"]["pp_size"] == 1
    assert result["strategy_summary"]["zero_stage"] == 0
    assert result["model_estimates"]["estimated_model_size_gb"] < 1.0


def test_full_pipeline_large_model_multi_node(tmp_path: Path):
    """Large model (70B-class) on multi-node: TP > 1 and/or PP > 1."""
    output_json = tmp_path / "strategy.json"
    run_script(
        "recommend_parallel_strategy.py",
        "--num-layers", "80",
        "--hidden-size", "8192",
        "--seq-len", "4096",
        "--batch-size", "4",
        "--vocab-size", "32000",
        "--num-nodes", "2",
        "--npus-per-node", "8",
        "--hardware", "ascend_910b2",
        "--output-json", str(output_json),
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    # 70B+ model needs TP > 1
    assert result["strategy_summary"]["tp_size"] > 1
    assert result["strategy_summary"]["world_size"] > 1
    assert "tp" in result["recommendations"]
    assert "pp" in result["recommendations"]


def test_full_pipeline_missing_required_arg(tmp_path: Path):
    """Missing required argument should exit with error."""
    output_json = tmp_path / "strategy.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "recommend_parallel_strategy.py"),
         "--output-json", str(output_json)],
        text=True, capture_output=True,
    )
    assert result.returncode != 0
