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
"""MindSpore vs torch_npu comparison tests for MHC ACLNN adapters."""

import gc

import numpy as np
import pytest
import torch
import torch_npu

import mindspore as ms
from mindspore import Tensor
from custom_op import (
    npu_mhc_post,
    npu_mhc_post_backward,
    npu_mhc_pre_sinkhorn,
    npu_mhc_pre_sinkhorn_backward,
)


DEVICE_ID = 7
torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
ms.context.set_context(device_target="Ascend", device_id=DEVICE_ID)
ms.context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
torch.use_deterministic_algorithms(True)


@pytest.fixture(autouse=True)
def _cleanup_npu_memory():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


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


def _to_pta(arr, use_bf16=False):
    t = torch.from_numpy(arr).npu()
    if use_bf16 and t.dtype == torch.float16:
        t = t.to(torch.bfloat16)
    return t


def _to_ms(arr, use_bf16=False):
    t = Tensor(arr)
    if use_bf16 and t.dtype == ms.float16:
        t = t.astype(ms.bfloat16)
    return t


def _pta_np(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.float().cpu().numpy()
    return tensor.cpu().numpy()


def _ms_np(tensor):
    if tensor.dtype == ms.bfloat16:
        return tensor.astype(ms.float32).asnumpy()
    return tensor.asnumpy()


def _single_output(result):
    if isinstance(result, (tuple, list)):
        assert len(result) == 1
        return result[0]
    return result


def _tuple_outputs(result):
    if isinstance(result, (tuple, list)):
        return tuple(result)
    return (result,)


def _torch_npu_op(*names):
    for name in names:
        if hasattr(torch_npu, name):
            return getattr(torch_npu, name)
    raise AttributeError(f"torch_npu has none of: {', '.join(names)}")


def _gen_mhc_presinkhorn_inputs(seed, dtype):
    rng = np.random.default_rng(seed)
    bs, seq_len, n, c = 1, 1, 4, 1280
    fusion_size = n * n + 2 * n
    use_bf16 = dtype == "bf16"
    return {
        "x": rng.normal(0, 0.2, (bs, seq_len, n, c)).astype(np.float16),
        "phi": rng.normal(0, 0.02, (fusion_size, n * c)).astype(np.float32),
        "alpha": rng.normal(0, 0.2, (3,)).astype(np.float32),
        "bias": rng.normal(0, 0.2, (fusion_size,)).astype(np.float32),
        "grad_h_in": rng.normal(0, 0.2, (bs, seq_len, c)).astype(np.float16),
        "grad_h_post": rng.normal(0, 0.2, (bs, seq_len, n)).astype(np.float32),
        "grad_h_res": rng.normal(0, 0.2, (bs, seq_len, n, n)).astype(np.float32),
        "use_bf16": use_bf16,
    }


def _gen_mhc_post_inputs(seed, shape, use_bf16=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.2, shape).astype(np.float16)
    h_res = rng.normal(0, 0.2, shape[:-1] + (shape[-2],)).astype(np.float32)
    h_out = rng.normal(0, 0.2, shape[:-2] + (shape[-1],)).astype(np.float16)
    h_post = rng.normal(0, 0.2, shape[:-1]).astype(np.float32)
    grad_y = rng.normal(0, 0.2, shape).astype(np.float16)
    return {
        "x": x,
        "h_res": h_res,
        "h_out": h_out,
        "h_post": h_post,
        "grad_y": grad_y,
        "use_bf16": use_bf16,
    }


def _gen_sinkhorn_inputs(seed, shape):
    rng = np.random.default_rng(seed)
    return {
        "x": rng.normal(0, 0.5, shape).astype(np.float32),
        "grad_output": rng.normal(0, 0.2, shape).astype(np.float32),
    }


def _run_pta_mhc_presinkhorn(inputs, hc_mult, num_iters, hc_eps, norm_eps, out_flag):
    use_bf16 = inputs["use_bf16"]
    results = torch_npu.npu_mhc_pre_sinkhorn(
        _to_pta(inputs["x"], use_bf16),
        _to_pta(inputs["phi"]),
        _to_pta(inputs["alpha"]),
        _to_pta(inputs["bias"]),
        hc_mult,
        num_iters,
        hc_eps,
        norm_eps,
        out_flag,
    )
    return tuple(_pta_np(r) for r in _tuple_outputs(results))


def _run_ms_mhc_presinkhorn(inputs, hc_mult, num_iters, hc_eps, norm_eps, out_flag):
    use_bf16 = inputs["use_bf16"]
    results = npu_mhc_pre_sinkhorn(
        _to_ms(inputs["x"], use_bf16),
        _to_ms(inputs["phi"]),
        _to_ms(inputs["alpha"]),
        _to_ms(inputs["bias"]),
        hc_mult=hc_mult,
        num_iters=num_iters,
        hc_eps=hc_eps,
        norm_eps=norm_eps,
        out_flag=out_flag,
    )
    return tuple(_ms_np(r) for r in results)


def _run_pta_mhc_presinkhorn_backward(inputs, saved, hc_eps):
    use_bf16 = inputs["use_bf16"]
    _, _, _, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = saved
    out = torch_npu.npu_mhc_pre_sinkhorn_backward(
        _to_pta(inputs["grad_h_in"], use_bf16),
        _to_pta(inputs["grad_h_post"]),
        _to_pta(inputs["grad_h_res"]),
        _to_pta(inputs["x"], use_bf16),
        _to_pta(inputs["phi"]),
        _to_pta(inputs["alpha"]),
        _to_pta(inputs["bias"]),
        _to_pta(h_pre),
        _to_pta(hc_before_norm),
        _to_pta(inv_rms),
        _to_pta(sum_out),
        _to_pta(norm_out),
        hc_eps,
    )
    return tuple(_pta_np(r) for r in _tuple_outputs(out))


def _run_ms_mhc_presinkhorn_backward(inputs, saved, hc_eps):
    use_bf16 = inputs["use_bf16"]
    _, _, _, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = saved
    results = npu_mhc_pre_sinkhorn_backward(
        _to_ms(inputs["grad_h_in"], use_bf16),
        _to_ms(inputs["grad_h_post"]),
        _to_ms(inputs["grad_h_res"]),
        _to_ms(inputs["x"], use_bf16),
        _to_ms(inputs["phi"]),
        _to_ms(inputs["alpha"]),
        _to_ms(inputs["bias"]),
        _to_ms(h_pre),
        _to_ms(hc_before_norm),
        _to_ms(inv_rms),
        _to_ms(sum_out),
        _to_ms(norm_out),
        hc_eps=hc_eps,
    )
    return tuple(_ms_np(r) for r in results)


def _run_pta_mhc_post(inputs):
    use_bf16 = inputs["use_bf16"]
    out = torch_npu.npu_mhc_post(
        _to_pta(inputs["x"], use_bf16),
        _to_pta(inputs["h_res"]),
        _to_pta(inputs["h_out"], use_bf16),
        _to_pta(inputs["h_post"]),
    )
    return _pta_np(_single_output(out))


def _run_ms_mhc_post(inputs):
    use_bf16 = inputs["use_bf16"]
    out = npu_mhc_post(
        _to_ms(inputs["x"], use_bf16),
        _to_ms(inputs["h_res"]),
        _to_ms(inputs["h_out"], use_bf16),
        _to_ms(inputs["h_post"]),
    )
    return _ms_np(out)


def _run_pta_mhc_post_backward(inputs):
    use_bf16 = inputs["use_bf16"]
    results = torch_npu.npu_mhc_post_backward(
        _to_pta(inputs["grad_y"], use_bf16),
        _to_pta(inputs["x"], use_bf16),
        _to_pta(inputs["h_res"]),
        _to_pta(inputs["h_out"], use_bf16),
        _to_pta(inputs["h_post"]),
    )
    return tuple(_pta_np(r) for r in _tuple_outputs(results))


def _run_ms_mhc_post_backward(inputs):
    use_bf16 = inputs["use_bf16"]
    results = npu_mhc_post_backward(
        _to_ms(inputs["grad_y"], use_bf16),
        _to_ms(inputs["x"], use_bf16),
        _to_ms(inputs["h_res"]),
        _to_ms(inputs["h_out"], use_bf16),
        _to_ms(inputs["h_post"]),
    )
    return tuple(_ms_np(r) for r in results)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_mhc_presinkhorn(dtype):
    hc_mult = 4
    num_iters = 20
    hc_eps = 1e-6
    norm_eps = 1e-6
    out_flag = True
    inputs = _gen_mhc_presinkhorn_inputs(seed=10, dtype=dtype)
    pta_out = _run_pta_mhc_presinkhorn(inputs, hc_mult, num_iters, hc_eps, norm_eps, out_flag)
    ms_out = _run_ms_mhc_presinkhorn(inputs, hc_mult, num_iters, hc_eps, norm_eps, out_flag)
    assert len(pta_out) == len(ms_out)
    for pta_item, ms_item in zip(pta_out, ms_out):
        allclose_nparray(pta_item, ms_item, rtol=0, atol=0)


def test_mhc_presinkhorn_backward():
    hc_mult = 4
    num_iters = 20
    hc_eps = 1e-6
    norm_eps = 1e-6
    out_flag = True
    inputs = _gen_mhc_presinkhorn_inputs(seed=11, dtype="fp16")
    saved = _run_pta_mhc_presinkhorn(inputs, hc_mult, num_iters, hc_eps, norm_eps, out_flag)

    pta_out = _run_pta_mhc_presinkhorn_backward(inputs, saved, hc_eps)
    ms_out = _run_ms_mhc_presinkhorn_backward(inputs, saved, hc_eps)
    assert len(pta_out) == len(ms_out)
    for pta_item, ms_item in zip(pta_out, ms_out):
        allclose_nparray(pta_item, ms_item, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(2, 4, 8), (1, 3, 4, 8)])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_mhc_post(shape, dtype):
    inputs = _gen_mhc_post_inputs(seed=0, shape=shape, use_bf16=(dtype == "bf16"))
    pta_out = _run_pta_mhc_post(inputs)
    ms_out = _run_ms_mhc_post(inputs)
    allclose_nparray(pta_out, ms_out, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_mhc_post_backward(dtype):
    inputs = _gen_mhc_post_inputs(seed=1, shape=(2, 4, 8), use_bf16=(dtype == "bf16"))
    pta_out = _run_pta_mhc_post_backward(inputs)
    ms_out = _run_ms_mhc_post_backward(inputs)
    for pta_item, ms_item in zip(pta_out, ms_out):
        allclose_nparray(pta_item, ms_item, rtol=0, atol=0)
