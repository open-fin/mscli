# Test Generation Reference

Use this reference when the user asks for tests for a generated op adapter.

## Required Shape

Generate one Python script per op. The script must contain:

- Imports.
- Device/context setup.
- One in-file `ms.ops.CustomOpBuilder(...).load()` call for the target kernel source.
- One in-file Python wrapper function for the generated op.
- Input generators.
- Reference runner when a `torch_npu.npu_*` API exists.
- MindSpore runner.
- Comparison helpers.
- Pytest test cases.

Do not import a local wrapper module or local test utility module. The only project file the script should need is the target kernel source file or files passed to `CustomOpBuilder`.

## Skeleton

```python
import gc
from pathlib import Path

import numpy as np
import pytest
import torch
import torch_npu

import mindspore as ms
from mindspore import Tensor, context


DEVICE_ID = 7
KERNEL_SOURCE = Path(__file__).with_name("example_op.cc")

torch.npu.set_device(DEVICE_ID)
torch.npu.set_compile_mode(jit_compile=False)
context.set_context(device_target="Ascend", device_id=DEVICE_ID)
context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
torch.use_deterministic_algorithms(True)

_custom_ops = ms.ops.CustomOpBuilder(
    "custom_ops_example_op_test",
    [str(KERNEL_SOURCE)],
    backend="Ascend",
).load()


def npu_example_op(x, scale=1.0, optional_tensor=None):
    return _custom_ops.npu_example_op(x, scale, optional_tensor)


@pytest.fixture(autouse=True)
def _cleanup_npu_memory():
    yield
    gc.collect()
    torch.npu.empty_cache()
    if hasattr(ms.hal, "empty_cache"):
        ms.hal.empty_cache()
```

Adjust `KERNEL_SOURCE`, builder name, wrapper name, and wrapper arguments for the target op.

## Test Matrix

Cover combinations, not just isolated cases:

- Dtypes: every supported input dtype, commonly `fp16`, `bf16`, and `fp32` where supported.
- Values: random normal values, zeros, small-magnitude values, negative values when valid, boundary-ish scalar values, and safe nonzero denominators for division-like formulas.
- Shapes: minimum supported shape, representative normal shape, and at least one larger or non-square shape when the op supports it.
- Optional tensors: absent and present.
- Optional array/list attributes: default and explicit values; include sequential calls with different arrays when the op accepts changing sequence arrays.
- Scalar attributes: default values and at least one meaningful non-default combination.
- Layout or mode attributes: every supported mode that changes shape or semantics.

Keep cases small enough to avoid OOM and long runtime on Ascend hardware.

## CustomOpBuilder Call Rules

MindSpore `CustomOpBuilder` exports a pybind callable for the generated C++ function, but it does not behave like the torch-npu Python API for all calling conventions.

- Pass all generated C++ parameters positionally in tests, including explicit default values.
- Use `None` positionally for omitted optional tensors and optional scalar attributes.
- Avoid keyword arguments when calling `_custom_ops.npu_xxx(...)`; they can fail even if the torch-npu reference accepts the same keyword.
- Do not assume C++ default arguments are applied by the generated caller. If the exported function has 9 parameters, the test should pass 9 arguments unless the specific wrapper has already been proven to support omitted defaults.

This rule applies to the custom op call, not the torch-npu reference call. The torch-npu side should still use the target public interface naturally when that improves readability.

## HCCL and Multi-Process Tests

For custom ops that require an HCCL communication group, keep the test as one Python script and spawn one child process per rank. The validated minimal pattern on Ascend is `world_size=2` with two NPU devices.

Use `torch.distributed` only to create the HCCL group and obtain the HCCL communication name. Build and call the MindSpore custom op inside each child process after setting that rank's NPU device.

```python
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

import mindspore as ms
from mindspore import Tensor, context


KERNEL_SOURCE = Path(__file__).with_name("example_hccl_op.cc")


def _get_hccl_comm_name(group, rank):
    if torch.__version__ > "2.0.1":
        return group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    return group.get_hccl_comm_name(rank)


def _run_rank(rank, world_size, master_port, source_path, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["HCCL_WHITELIST_DISABLE"] = "1"

    torch_npu.npu.set_device(rank)
    torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group(backend="hccl", world_size=world_size, rank=rank)
    group = dist.distributed_c10d._get_default_group()
    hcom_name = _get_hccl_comm_name(group, rank)

    context.set_context(device_target="Ascend", device_id=rank)
    context.set_context(mode=ms.PYNATIVE_MODE, deterministic="ON", pynative_synchronize=False)
    custom_ops = ms.ops.CustomOpBuilder(
        f"custom_ops_example_hccl_op_ws{world_size}_rank{rank}",
        [source_path],
        backend="Ascend",
    ).load()

    # Create per-rank inputs, run torch_npu reference with hcom_name, then run
    # the custom op with the same hcom_name and compare local rank outputs.
    x = np.ones((16, 16), dtype=np.float16)
    expected = torch_npu.npu_example_hccl_op(torch.from_numpy(x).npu(), hcom_name, world_size)
    actual = custom_ops.npu_example_hccl_op(Tensor(x), hcom_name, world_size)
    np.testing.assert_allclose(expected.cpu().numpy(), actual.asnumpy(), rtol=1e-3, atol=1e-3)

    result_queue.put(rank)
    dist.barrier()
    dist.destroy_process_group()


def test_example_hccl_op_world_size_2():
    if torch.npu.device_count() < 2:
        pytest.skip("HCCL world_size=2 test requires at least two NPU devices")

    world_size = 2
    master_port = int(os.environ.get("MASTER_PORT", str(51000 + os.getpid() % 1000)))
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(world_size)
    processes = []
    for rank in range(world_size):
        process = ctx.Process(target=_run_rank, args=(rank, world_size, master_port, str(KERNEL_SOURCE), result_queue))
        process.start()
        processes.append(process)

    for process in processes:
        process.join(timeout=240)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("HCCL world_size=2 test timed out")
        assert process.exitcode == 0

    ranks = [result_queue.get(timeout=10) for _ in range(world_size)]
    assert sorted(ranks) == [0, 1]
```

Practical rules:

- Set `MASTER_ADDR`, `MASTER_PORT`, and `HCCL_WHITELIST_DISABLE` inside each child before `init_process_group`.
- Call `torch_npu.npu.set_device(rank)` before initializing the HCCL process group.
- Call `context.set_context(device_target="Ascend", device_id=rank)` before building or running the MindSpore custom op.
- Use a unique `CustomOpBuilder` module name per rank, for example include `ws{world_size}_rank{rank}`.
- Pass the HCCL communication name string from torch distributed into both the torch-npu reference call and the custom op call.
- Use a `spawn` multiprocessing context. Do not initialize HCCL or build the custom op in the parent process.
- Keep a timeout on `join()` and fail explicitly so a broken communication setup does not hang pytest indefinitely.
- Use the smallest valid `world_size`, normally 2, unless the operator contract specifically requires 4, 8, or 16.

## Do Not Link Multiple Standalone Pybind Sources Together

Each generated custom op `.cc` file is normally a standalone pybind extension and contains one `PYBIND11_MODULE(MS_EXTENSION_NAME, m)` block. Do not pass two such standalone sources to one `CustomOpBuilder` call:

```python
ms.ops.CustomOpBuilder(
    "custom_ops_bad_combined_test",
    ["npu_group_norm_swish_grad.cc", "npu_group_norm_swish.cc"],
    backend="Ascend",
).load()
```

This compiles both sources into one shared object, so the linker sees two definitions of the same module init symbol and fails.

If a test needs another generated op:

- Prefer using the torch-npu reference op to produce auxiliary/intermediate values when possible, such as forward `mean` and `rstd` inputs for a backward op test.
- Otherwise build each standalone `.cc` file with a separate `CustomOpBuilder` module name.
- Only use one `CustomOpBuilder` call for multiple functions when the source is deliberately written as a combined `.cc` file with exactly one `PYBIND11_MODULE` block that registers all functions.

## Smoke Tests Need Valid CANN Tiling Inputs

Shape/dtype smoke tests still execute CANN tiling. They must satisfy the target kernel's real constraints, not only produce tensors with plausible ranks.

Before finalizing smoke cases, check the ACLNN document, torch-npu op-plugin test cases, meta registrations, and error messages from the validation host for constraints such as:

- exact rank requirements, for example 3D `x` for `FlatQuant`
- cache layout requirements, for example 3D key/value and 4D PA cache tensors for `ScatterPaKvCache`
- large fixed dimensions, for example sparse lightning indexer paths requiring `D=512`, index `D=128`, and rope tensors
- dtype compatibility between paired inputs, for example quantization input and scale dtypes
- mandatory optional tensors on a specific CANN implementation, even when the public API marks them optional

If a reference torch-npu test uses large shapes only for fake tensor/meta coverage, derive the smallest real tiling-valid case where possible. If the kernel is too restrictive or too expensive for practical execution on the validation host, skip with a specific reason and report it as a host/kernel constraint rather than an adapter failure.

## Reference Comparison

Prefer comparing against `torch_npu.npu_*` for the same op when it exists. Convert from the same NumPy arrays into both frameworks.

```python
def _to_pta(arr, use_bf16=False):
    if arr is None:
        return None
    t = torch.from_numpy(arr).npu()
    if use_bf16 and t.dtype == torch.float16:
        t = t.to(torch.bfloat16)
    return t


def _to_ms(arr, use_bf16=False):
    if arr is None:
        return None
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
```

Use bit-exact comparison when the existing op family is deterministic and the reference demonstrates exact equality. Otherwise set tolerances from the dtype and numeric stability of the op.

## Comparison Helper

```python
def allclose_nparray(expected, actual, rtol, atol, equal_nan=True):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    assert expected.shape == actual.shape
    if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=equal_nan):
        err = np.abs(expected - actual)
        bad = np.greater(err, atol + np.abs(actual) * rtol)
        raise AssertionError(
            f"mismatch_count={np.count_nonzero(bad)} total={expected.size}\n"
            f"expected_bad={expected[bad]}\nactual_bad={actual[bad]}\nerror={err[bad]}"
        )
```

## Self-Contained Wrapper Rules

The wrapper in the test script should mirror the generated C++ exported function:

- Use the same function name and argument order.
- Keep defaults in the Python wrapper aligned with the C++ adapter defaults.
- Pass `None` through for optional tensors.
- Pass list attributes directly from the test case.
- Return the raw custom op result; normalize tuple/list handling in runner helpers.

## Existing Examples

Use these raw examples for style and coverage ideas:

- `references/examples/python/custom_op.py`: wrapper style for calling built custom ops.
- `references/examples/python/test_mhc_ops.py`: dtype matrix, forward/backward comparisons, cleanup fixture.
- `references/examples/python/test_dense_lightning_indexer_grad_kl_loss.py`: optional tensors, sequence arrays, layouts, scalar combinations.
- `references/examples/python/test_dense_lightning_indexer_softmax_lse.py`: layout-dependent outputs and exact comparisons.
- `references/examples/python/test_sparse_lightning_indexer_grad_kl_loss.py`: sparse index generation and ragged/TND cases.

The raw examples import shared wrappers; generated tests must inline the one-op wrapper instead.
