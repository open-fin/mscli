#!/usr/bin/env python3
"""Scaffold for operator-specific dtype probing.

Default policy:

- Probe through the PTA interface by default.
- Switch to the MS interface only when the user explicitly asks for MS-based probing
  or the PTA path is unavailable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import mindspore as ms
import mindspore.ops as ops


FRAMEWORK_FILE_NAME = "dtype_probe_execution_framework.py"
PROBE_BACKEND = "pta"  # "pta" by default; switch to "ms" only when explicitly requested.
PTA_DEVICE = "npu:0"


def _iter_framework_candidates(start_file: Path):
    framework_name = FRAMEWORK_FILE_NAME
    seen: set[Path] = set()
    for anchor in [start_file.parent, *start_file.parents]:
        direct_candidates = (
            anchor / framework_name,
            anchor / "scripts" / framework_name,
            anchor / "skills" / "op-info-test" / "scripts" / framework_name,
        )
        for candidate in direct_candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def _load_driver_framework():
    for framework_path in _iter_framework_candidates(Path(__file__).resolve()):
        if not framework_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("dtype_probe_execution_framework", framework_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["dtype_probe_execution_framework"] = module
        spec.loader.exec_module(module)
        return module
    raise RuntimeError(
        "Unable to locate dtype_probe_execution_framework.py. "
        "Copy it alongside this driver or run from a workspace that contains the op-info-test skill."
    )


def _resolve_driver_framework(driver_framework=None):
    if driver_framework is not None:
        return driver_framework

    injected_framework = sys.modules.get("dtype_probe_execution_framework")
    if injected_framework is not None and hasattr(injected_framework, "OperatorProbeSpec"):
        return injected_framework

    main_module = sys.modules.get("__main__")
    if main_module is not None and hasattr(main_module, "OperatorProbeSpec"):
        return main_module

    return _load_driver_framework()


OP_NAME = "example_op"
HAS_BACKWARD = True
DOC_DECLARED_FORWARD_DTYPES_910B = ("float16", "float32", "float64")
DOC_DECLARED_BACKWARD_DTYPES_910B = DOC_DECLARED_FORWARD_DTYPES_910B


def build_samples(dtype_name, ms_dtype, framework=None):
    """Return one or more representative ProbeSample objects for *ms_dtype*.

    Notes:
    - Unary ops can usually reuse `make_unary_sample`.
    - Binary ops can usually reuse `make_binary_same_dtype_sample`.
    - Reduction ops can usually reuse `make_reduction_sample`.
    - If the op takes dtype/index/layout parameters, build samples manually.
    - If the op claims mixed-dtype support, add explicit mixed-dtype samples here.
    """
    framework = _resolve_driver_framework(framework)
    return [
        framework.make_unary_sample(
            ms_dtype,
            shape=(2, 3),
            low=-0.75,
            high=0.75,
            sample_name=f"{dtype_name}_basic",
        )
    ]


def example_ms_forward(x):
    """Replace with the MindSpore forward path, for example: ops.acos(x)."""
    raise NotImplementedError("Replace with the MindSpore forward path.")


def example_pta_forward(x):
    """Replace with the PTA forward path, for example: torch.acos(x)."""
    raise NotImplementedError("Replace with the PTA forward path.")


def _torch_tensor_supports_grad(tensor) -> bool:
    import torch

    return torch.is_floating_point(tensor) or torch.is_complex(tensor)


def _convert_ms_value_to_pta(value, *, requires_grad=False):
    import torch
    import torch_npu  # noqa: F401

    if isinstance(value, ms.Tensor):
        tensor = torch.from_numpy(value.asnumpy().copy()).to(PTA_DEVICE)
        if requires_grad and _torch_tensor_supports_grad(tensor):
            tensor.requires_grad_(True)
        return tensor, [tensor] if tensor.requires_grad else []
    if isinstance(value, list):
        converted = []
        tracked = []
        for item in value:
            converted_item, tracked_item = _convert_ms_value_to_pta(item, requires_grad=requires_grad)
            converted.append(converted_item)
            tracked.extend(tracked_item)
        return converted, tracked
    if isinstance(value, tuple):
        converted = []
        tracked = []
        for item in value:
            converted_item, tracked_item = _convert_ms_value_to_pta(item, requires_grad=requires_grad)
            converted.append(converted_item)
            tracked.extend(tracked_item)
        return tuple(converted), tracked
    if isinstance(value, dict):
        converted = {}
        tracked = []
        for key, item in value.items():
            converted_item, tracked_item = _convert_ms_value_to_pta(item, requires_grad=requires_grad)
            converted[key] = converted_item
            tracked.extend(tracked_item)
        return converted, tracked
    return value, []


def _build_pta_invocation(sample, *, enable_grad):
    grad_positions = sample.grad_position if sample.grad_position is not None else (0,)
    args = []
    tracked_tensors = []
    for index, value in enumerate(sample.positional_args()):
        converted, tracked = _convert_ms_value_to_pta(value, requires_grad=enable_grad and index in grad_positions)
        args.append(converted)
        tracked_tensors.extend(tracked)
    kwargs, _ = _convert_ms_value_to_pta(sample.op_kwargs, requires_grad=False)
    return tuple(args), kwargs, tracked_tensors


def _flatten_torch_outputs(outputs):
    import torch

    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if isinstance(outputs, (list, tuple)):
        flattened = []
        for item in outputs:
            flattened.extend(_flatten_torch_outputs(item))
        return flattened
    raise TypeError("PTA probe template only supports Tensor outputs or tuple/list of Tensor outputs by default.")


def run_ms_forward(sample):
    return example_ms_forward(*sample.positional_args(), **sample.op_kwargs)


def run_pta_forward(sample):
    args, kwargs, _ = _build_pta_invocation(sample, enable_grad=False)
    return example_pta_forward(*args, **kwargs)


def run_pta_backward(sample):
    import torch
    import torch_npu  # noqa: F401

    framework = _resolve_driver_framework()
    args, kwargs, tracked_tensors = _build_pta_invocation(sample, enable_grad=True)
    if not tracked_tensors:
        return framework.BACKWARD_NOT_APPLICABLE

    outputs = example_pta_forward(*args, **kwargs)
    flat_outputs = _flatten_torch_outputs(outputs)
    grad_tensors = [torch.ones_like(output) for output in flat_outputs]
    torch.autograd.backward(flat_outputs, grad_tensors=grad_tensors)
    torch.npu.synchronize()
    return tuple(tensor.grad.detach().cpu() for tensor in tracked_tensors)


def _selected_forward_runner():
    if PROBE_BACKEND == "pta":
        return run_pta_forward
    if PROBE_BACKEND == "ms":
        return run_ms_forward
    raise ValueError(f"Unsupported PROBE_BACKEND: {PROBE_BACKEND}")


def _selected_backward_runner():
    if not HAS_BACKWARD:
        return None
    if PROBE_BACKEND == "pta":
        return run_pta_backward
    if PROBE_BACKEND == "ms":
        framework = _resolve_driver_framework()
        return framework.make_default_backward_runner(example_ms_forward, grad_position=(0,))
    raise ValueError(f"Unsupported PROBE_BACKEND: {PROBE_BACKEND}")


def build_operator_probes(driver_framework=None):
    framework = _resolve_driver_framework(driver_framework)
    return [
        framework.OperatorProbeSpec(
            op_name=OP_NAME,
            build_samples=build_samples,
            forward_runner=_selected_forward_runner(),
            backward_runner=_selected_backward_runner(),
            probe_backend=PROBE_BACKEND,
            candidate_dtypes=framework.DEFAULT_DTYPE_NAMES,
            declared_doc_forward_dtypes=DOC_DECLARED_FORWARD_DTYPES_910B,
            declared_doc_backward_dtypes=DOC_DECLARED_BACKWARD_DTYPES_910B,
            supports_backward=HAS_BACKWARD,
            notes=(
                "Default to PTA runtime probing because new MindSpore interfaces are brought in "
                "against the PTA benchmark surface. Backfill the OpInfo dtype declaration from the "
                "PTA probe first, then use MindSpore opinfo case tests to clarify any divergence. "
                "Switch PROBE_BACKEND to 'ms' only when the task explicitly asks for MindSpore-side probing."
            ),
        )
    ]
