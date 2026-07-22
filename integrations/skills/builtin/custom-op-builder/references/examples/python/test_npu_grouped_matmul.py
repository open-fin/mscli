import gc
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch_npu

import mindspore as ms
from mindspore import Tensor, context
from mindspore._c_expression import typing


DEVICE_ID = 0
KERNEL_SOURCE = Path(__file__).with_name("npu_grouped_matmul.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
torch.use_deterministic_algorithms(True)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_npu_grouped_matmul_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def _ms_type_id(dtype):
    if dtype is None:
        return None
    mapping = {
        torch.float16: ms.float16,
        torch.float32: ms.float32,
        torch.bfloat16: ms.bfloat16,
        torch.int8: ms.int8,
        torch.int32: ms.int32,
    }
    return typing.type_to_type_id(mapping[dtype])


def npu_grouped_matmul(
    x,
    weight,
    *,
    bias=None,
    scale=None,
    offset=None,
    antiquant_scale=None,
    antiquant_offset=None,
    per_token_scale=None,
    group_list=None,
    activation_input=None,
    activation_quant_scale=None,
    activation_quant_offset=None,
    split_item=0,
    group_type=None,
    group_list_type=0,
    act_type=0,
    tuning_config=None,
    output_dtype=None,
    x_acl_dtype=None,
    weight_acl_dtype=None,
    scale_acl_dtype=None,
    per_token_scale_acl_dtype=None,
):
    group_list_tensor = group_list if isinstance(group_list, Tensor) else None
    group_list_vector = list(group_list) if isinstance(group_list, (list, tuple)) else None
    out = _custom_ops.npu_grouped_matmul(
        x,
        weight,
        bias,
        scale,
        offset,
        antiquant_scale,
        antiquant_offset,
        per_token_scale,
        group_list_tensor,
        group_list_vector,
        activation_input,
        activation_quant_scale,
        activation_quant_offset,
        split_item,
        group_type,
        group_list_type,
        act_type,
        tuning_config,
        _ms_type_id(output_dtype),
        x_acl_dtype,
        weight_acl_dtype,
        scale_acl_dtype,
        per_token_scale_acl_dtype,
    )
    return list(out) if isinstance(out, tuple) else [out]


@pytest.fixture(autouse=True)
def _cleanup_npu_memory():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()


def _to_torch_list(arrays, dtype):
    return [torch.from_numpy(arr).to(dtype).npu() for arr in arrays]


def _to_ms_list(arrays, dtype):
    ms_dtype = {torch.float16: ms.float16, torch.float32: ms.float32, torch.bfloat16: ms.bfloat16}[dtype]
    return [Tensor(arr).astype(ms_dtype) for arr in arrays]


def _torch_np(tensor):
    if tensor.dtype == torch.bfloat16:
        return tensor.float().cpu().numpy()
    return tensor.cpu().numpy()


def _ms_np(tensor):
    if tensor.dtype == ms.bfloat16:
        return tensor.astype(ms.float32).asnumpy()
    return tensor.asnumpy()


def _assert_outputs_close(expected, actual, dtype):
    assert len(expected) == len(actual)
    rtol, atol = (1e-2, 1e-2) if dtype != torch.float32 else (1e-4, 1e-4)
    for exp, act in zip(expected, actual):
        exp_np = _torch_np(exp)
        act_np = _ms_np(act)
        assert exp_np.shape == act_np.shape
        np.testing.assert_allclose(act_np, exp_np, rtol=rtol, atol=atol)


def _run_timed(fn, warmup=3, repeat=10):
    for _ in range(warmup):
        outs = fn()
        for out in outs:
            if hasattr(out, "asnumpy"):
                out.asnumpy()
            else:
                out.cpu()
    start = time.perf_counter()
    for _ in range(repeat):
        outs = fn()
        for out in outs:
            if hasattr(out, "asnumpy"):
                out.asnumpy()
            else:
                out.cpu()
    return (time.perf_counter() - start) / repeat


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_split_item_0_multi_output_matches_torch_npu(dtype):
    rng = np.random.default_rng(0)
    x_np = [
        rng.normal(0.0, 0.1, (16, 32)).astype(np.float32),
        rng.normal(0.0, 0.1, (8, 32)).astype(np.float32),
    ]
    w_np = [
        rng.normal(0.0, 0.1, (32, 24)).astype(np.float32),
        rng.normal(0.0, 0.1, (32, 16)).astype(np.float32),
    ]
    b_np = [
        rng.normal(0.0, 0.1, (24,)).astype(np.float32),
        rng.normal(0.0, 0.1, (16,)).astype(np.float32),
    ]

    torch_x = _to_torch_list(x_np, dtype)
    torch_w = _to_torch_list(w_np, dtype)
    bias_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
    torch_b = _to_torch_list(b_np, bias_dtype)
    ms_x = _to_ms_list(x_np, dtype)
    ms_w = _to_ms_list(w_np, dtype)
    ms_b = _to_ms_list(b_np, bias_dtype)

    expected = torch_npu.npu_grouped_matmul(torch_x, torch_w, bias=torch_b, split_item=0, group_type=-1)
    actual = npu_grouped_matmul(ms_x, ms_w, bias=ms_b, split_item=0, group_type=-1)
    _assert_outputs_close(expected, actual, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_item_3_tensor_group_list_matches_torch_npu(dtype):
    rng = np.random.default_rng(1)
    group_np = np.array([12, 20], dtype=np.int64)
    x_np = [rng.normal(0.0, 0.1, (20, 32)).astype(np.float32)]
    w_np = [rng.normal(0.0, 0.1, (2, 32, 24)).astype(np.float32)]
    b_np = [rng.normal(0.0, 0.1, (2, 24)).astype(np.float32)]

    torch_x = _to_torch_list(x_np, dtype)
    torch_w = _to_torch_list(w_np, dtype)
    bias_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
    torch_b = _to_torch_list(b_np, bias_dtype)
    torch_group = torch.from_numpy(group_np).npu()
    ms_x = _to_ms_list(x_np, dtype)
    ms_w = _to_ms_list(w_np, dtype)
    ms_b = _to_ms_list(b_np, bias_dtype)
    ms_group = Tensor(group_np)

    expected = torch_npu.npu_grouped_matmul(
        torch_x, torch_w, bias=torch_b, group_list=torch_group, split_item=3, group_type=0
    )
    actual = npu_grouped_matmul(ms_x, ms_w, bias=ms_b, group_list=ms_group, split_item=3, group_type=0)
    _assert_outputs_close(expected, actual, dtype)


def test_list_group_list_legacy_path_matches_torch_npu():
    rng = np.random.default_rng(2)
    dtype = torch.float16
    group_list = [5, 11]
    x_np = [rng.normal(0.0, 0.1, (11, 16)).astype(np.float32)]
    w_np = [
        rng.normal(0.0, 0.1, (16, 8)).astype(np.float32),
        rng.normal(0.0, 0.1, (16, 12)).astype(np.float32),
    ]
    ms_x = _to_ms_list(x_np, dtype)
    ms_w = _to_ms_list(w_np, dtype)
    torch_x = _to_torch_list(x_np, dtype)
    torch_w = _to_torch_list(w_np, dtype)

    expected = torch_npu.npu_grouped_matmul(torch_x, torch_w, group_list=group_list, split_item=0)
    actual = npu_grouped_matmul(ms_x, ms_w, group_list=group_list, split_item=0)
    _assert_outputs_close(expected, actual, dtype)


def test_benchmark_against_torch_npu():
    rng = np.random.default_rng(3)
    dtype = torch.float16
    group_np = np.array([64, 128], dtype=np.int64)
    x_np = [rng.normal(0.0, 0.1, (128, 128)).astype(np.float32)]
    w_np = [rng.normal(0.0, 0.1, (2, 128, 128)).astype(np.float32)]

    torch_x = _to_torch_list(x_np, dtype)
    torch_w = _to_torch_list(w_np, dtype)
    torch_group = torch.from_numpy(group_np).npu()
    ms_x = _to_ms_list(x_np, dtype)
    ms_w = _to_ms_list(w_np, dtype)
    ms_group = Tensor(group_np)

    torch_time = _run_timed(
        lambda: torch_npu.npu_grouped_matmul(
            torch_x, torch_w, group_list=torch_group, split_item=3, group_type=0
        )
    )
    custom_time = _run_timed(
        lambda: npu_grouped_matmul(ms_x, ms_w, group_list=ms_group, split_item=3, group_type=0)
    )
    print(f"torch_npu_avg_s={torch_time:.6f} custom_op_avg_s={custom_time:.6f}")
