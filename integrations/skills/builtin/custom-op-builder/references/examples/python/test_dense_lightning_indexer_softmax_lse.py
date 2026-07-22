# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""MindSpore vs PTA (torch_npu) bit-exact comparison tests for
aclnnDenseLightningIndexerSoftmaxLse (PyNative mode via CustomOpBuilder).

Run:
    pytest -sv test_ms_vs_pta_custom_aclnn_dense_lightning_indexer_softmax_lse.py
    pytest -sv test_ms_vs_pta_custom_aclnn_dense_lightning_indexer_softmax_lse.py::test_bsnd_base
"""

import gc
import numpy as np
import pytest
import torch
import torch_npu

import mindspore as ms
from mindspore import Tensor, context
from custom_op import npu_dense_lightning_indexer_softmax_lse

INT64_MAX = 9223372036854775807
CONTEXT_MODES = ["pynative"]


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        f"\ndata_expected_std:{data_expected[greater]}\ndata_me_error:{data_me[greater]}\nloss:{error[greater]}"


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)) or np.any(np.isnan(data_me)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


# ---------------------------------------------------------------------------
# Global device / context setup
# ---------------------------------------------------------------------------
DEVICE_ID = 7
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
ms.context.set_context(device_target="Ascend", device_id=DEVICE_ID)

ms.context.set_context(deterministic="ON", pynative_synchronize=False)
torch.use_deterministic_algorithms(True)


@pytest.fixture(autouse=True)
def _cleanup_npu_memory():
    """Release NPU memory after each test to prevent OOM across cases."""
    yield

    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


# =========================================================================
# Helper: input generation (numpy-based, deterministic)
# =========================================================================
def _gen_inputs_np(
    seed,
    b,
    s1,
    s2,
    nidx1,
    nidx2,
    d_index,
    is_tnd=False,
    seqlens_q=None,
    seqlens_kv=None,
    pta_weights=False,
):
    """Build random float16 numpy inputs for DenseLightningIndexerSoftmaxLse / PTA.

    Returns dict: query_index, key_index, weight, actual_seq_qlen, actual_seq_klen.
    BSND: shapes (B,S1,Nidx1,D), (B,S2,Nidx2,D), (B,S1,Nidx1); seqlens are None.
    TND (is_tnd=True, b==1): reshapes to (S1,Nidx1,D), (S2,Nidx2,D), (S1,Nidx1) and
    passes through seqlens_q / seqlens_kv as actual_seq_qlen / actual_seq_klen.
    pta_weights=True maps raw weights into a tighter range to match PTA-style init.
    """
    np.random.seed(seed)
    query_index = np.random.randn(b, s1, nidx1, d_index).astype(np.float16)
    key_index = np.random.randn(b, s2, nidx2, d_index).astype(np.float16)

    raw_w = np.random.randn(b, s1, nidx1).astype(np.float16)
    if pta_weights:
        a, b_val, kk = -0.05, 0.05, 3.0
        sc = (b_val - a) / (2 * kk)
        sh = (a + b_val) / 2
        weight = (raw_w * sc + sh).astype(np.float16)
    else:
        weight = (raw_w * 0.1 / 6.0).astype(np.float16)

    if is_tnd:
        if b == 1:
            query_index = query_index.reshape(s1, nidx1, d_index)
            key_index = key_index.reshape(s2, nidx2, d_index)
            weight = weight.reshape(s1, nidx1)

        final_seqlens_q = seqlens_q
        final_seqlens_kv = seqlens_kv
    else:
        final_seqlens_q = None
        final_seqlens_kv = None

    return {
        "query_index": query_index,
        "key_index": key_index,
        "weight": weight,
        "actual_seq_qlen": final_seqlens_q,
        "actual_seq_klen": final_seqlens_kv,
    }


# =========================================================================
# Helper: run PTA (torch_npu)
# =========================================================================
def _run_pta(
    inputs,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    use_bf16=False,
):
    """Run torch_npu.npu_dense_lightning_indexer_softmax_lse on NPU; return outputs as numpy."""

    def _to_npu(arr):
        if arr is None:
            return None
        t = torch.from_numpy(arr).npu()
        if use_bf16 and t.dtype == torch.float16:
            t = t.to(torch.bfloat16)
        return t

    results = torch_npu.npu_dense_lightning_indexer_softmax_lse(
        _to_npu(inputs["query_index"]),
        _to_npu(inputs["key_index"]),
        _to_npu(inputs["weight"]),
        actual_seq_qlen=inputs.get("actual_seq_qlen"),
        actual_seq_klen=inputs.get("actual_seq_klen"),
        layout=layout,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
    )

    def _out_np(t):
        if t.dtype == torch.bfloat16:
            return t.float().cpu().numpy()
        return t.cpu().numpy()

    return tuple(_out_np(r) for r in results)


# =========================================================================
# Helper: run MindSpore Graph mode (ops.Custom)
# =========================================================================
def _run_ms_graph(
    inputs,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    use_bf16=False,
):
    """Placeholder for MS ops.Custom Cell in GRAPH_MODE."""
    raise NotImplementedError(
        "Graph mode runner is a placeholder; use pynative for now."
    )




# =========================================================================
# Helper: run MindSpore PyNative mode (CustomOpBuilder)
# =========================================================================
def _run_ms_pynative(
    inputs,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    use_bf16=False,
):
    """Run MS CustomOpBuilder op in PYNATIVE_MODE, return outputs as numpy."""
    context.set_context(mode=ms.PYNATIVE_MODE)

    def _to_ms(arr):
        if arr is None:
            return None
        t = Tensor(arr)
        if use_bf16 and t.dtype == ms.float16:
            t = t.astype(ms.bfloat16)
        return t

    results = npu_dense_lightning_indexer_softmax_lse(
        _to_ms(inputs["query_index"]),
        _to_ms(inputs["key_index"]),
        _to_ms(inputs["weight"]),
        inputs.get("actual_seq_qlen"),
        inputs.get("actual_seq_klen"),
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
    )

    def _out_np(t):
        if t.dtype == ms.bfloat16:
            return t.astype(ms.float32).asnumpy()
        return t.asnumpy()

    return tuple(_out_np(r) for r in results)


# =========================================================================
# Dispatch: select graph or pynative runner by context_mode
# =========================================================================
def _run_ms_dispatch(
    context_mode,
    inputs,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=INT64_MAX,
    next_tokens=INT64_MAX,
    use_bf16=False,
):
    """Route to graph or pynative runner based on context_mode string."""
    runner = _run_ms_pynative if context_mode == "pynative" else _run_ms_graph
    return runner(
        inputs,
        layout=layout,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        use_bf16=use_bf16,
    )


# =========================================================================
# Test: BSND base, fp16 and bf16
# =========================================================================
@pytest.mark.parametrize("context_mode", CONTEXT_MODES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_bsnd_base(dtype, context_mode):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: BSND layout, fp16/bf16 types
    Expectation: Outputs match PTA
    """
    use_bf16 = dtype == "bf16"
    inputs = _gen_inputs_np(
        seed=42, b=1, s1=128, s2=128, nidx1=64, nidx2=1, d_index=128, is_tnd=False
    )

    pta_out = _run_pta(inputs, layout="BSND", use_bf16=use_bf16)
    ms_out = _run_ms_dispatch(context_mode, inputs, layout="BSND", use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)


# =========================================================================
# Test: BSND, different nidx1
# =========================================================================
@pytest.mark.parametrize("context_mode", CONTEXT_MODES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("nidx1", [8, 16, 32, 64])
def test_bsnd_different_nidx1(dtype, nidx1, context_mode):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: BSND layout with various Nidx1 sizes (actual_seq lengths None)
    Expectation: Outputs match PTA
    """
    use_bf16 = dtype == "bf16"
    inputs = _gen_inputs_np(
        seed=300 + nidx1,
        b=1,
        s1=128,
        s2=128,
        nidx1=nidx1,
        nidx2=1,
        d_index=128,
        is_tnd=False,
    )

    pta_out = _run_pta(inputs, layout="BSND", use_bf16=use_bf16)
    ms_out = _run_ms_dispatch(context_mode, inputs, layout="BSND", use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)


# =========================================================================
# Test: TND base, fp16 and bf16
# =========================================================================
@pytest.mark.parametrize("context_mode", CONTEXT_MODES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_tnd_base(dtype, context_mode):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND layout, fp16/bf16 types
    Expectation: Outputs match PTA
    """
    use_bf16 = dtype == "bf16"
    inputs = _gen_inputs_np(
        seed=100,
        b=1,
        s1=128,
        s2=128,
        nidx1=64,
        nidx2=1,
        d_index=128,
        is_tnd=True,
        seqlens_q=[128],
        seqlens_kv=[128],
    )

    pta_out = _run_pta(inputs, layout="TND", use_bf16=use_bf16)
    ms_out = _run_ms_dispatch(context_mode, inputs, layout="TND", use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)


# =========================================================================
# Test: TND, different nidx1
# =========================================================================
@pytest.mark.parametrize("context_mode", CONTEXT_MODES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("nidx1", [8, 16, 32, 64])
def test_tnd_different_nidx1(dtype, nidx1, context_mode):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: TND layout with various Nidx1 sizes
    Expectation: Outputs match PTA
    """
    use_bf16 = dtype == "bf16"
    inputs = _gen_inputs_np(
        seed=200 + nidx1,
        b=1,
        s1=128,
        s2=128,
        nidx1=nidx1,
        nidx2=1,
        d_index=128,
        is_tnd=True,
        seqlens_q=[128],
        seqlens_kv=[128],
    )

    pta_out = _run_pta(inputs, layout="TND", use_bf16=use_bf16)
    ms_out = _run_ms_dispatch(context_mode, inputs, layout="TND", use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)


# =========================================================================
# Test: PTA weights distribution
# =========================================================================
@pytest.mark.parametrize("context_mode", CONTEXT_MODES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["BSND", "TND"])
def test_pta_weights_distribution(dtype, layout, context_mode):
    """
    Feature: MS vs PTA bit-exact comparison
    Description: Test with PTA-style truncated normal weight distribution
    Expectation: Outputs match PTA
    """
    use_bf16 = dtype == "bf16"
    is_tnd = layout == "TND"
    seqlens_q = [128] if is_tnd else None
    seqlens_kv = [128] if is_tnd else None

    inputs = _gen_inputs_np(
        seed=300,
        b=1,
        s1=128,
        s2=128,
        nidx1=64,
        nidx2=1,
        d_index=128,
        is_tnd=is_tnd,
        seqlens_q=seqlens_q,
        seqlens_kv=seqlens_kv,
        pta_weights=True,
    )

    pta_out = _run_pta(inputs, layout=layout, use_bf16=use_bf16)
    ms_out = _run_ms_dispatch(context_mode, inputs, layout=layout, use_bf16=use_bf16)

    allclose_nparray(pta_out[0], ms_out[0], 0, 0)
    allclose_nparray(pta_out[1], ms_out[1], 0, 0)
