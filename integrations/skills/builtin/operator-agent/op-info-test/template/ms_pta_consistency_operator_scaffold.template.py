#!/usr/bin/env python3
"""Scaffold for operator-specific MS/PTA consistency validation.

Graph-mode note:

- `context.set_context(mode=ms.GRAPH_MODE)` only switches the runtime mode.
- Keep the Graph-mode MS path in a named `nn.Cell` by default.
- Use `@ms.jit` only when a standalone Python function is genuinely the better fit.
- Keep Graph-mode callables as named `Cell.construct` methods or named `@ms.jit`
  functions so source inspection and graph parsing stay stable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_npu  # noqa: F401

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


FRAMEWORK_FILE_NAME = "ms_pta_consistency_execution_framework.py"
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


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
            if candidate not in seen:
                seen.add(candidate)
                yield candidate
        for pattern in (
            f"*/skills/op-info-test/scripts/{framework_name}",
            f"*/*/skills/op-info-test/scripts/{framework_name}",
            f"*/*/*/skills/op-info-test/scripts/{framework_name}",
        ):
            for candidate in anchor.glob(pattern):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved


def _load_driver_framework():
    for framework_path in _iter_framework_candidates(Path(__file__).resolve()):
        if not framework_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("ms_pta_consistency_execution_framework", framework_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise RuntimeError(
        "Unable to locate ms_pta_consistency_execution_framework.py. "
        "Copy it alongside this driver or run from a workspace that contains the op-info-test skill."
    )


framework = _load_driver_framework()


OP_NAME = "example_op"
CASE_PREFIX = OP_NAME.replace(".", "_")
HAS_BACKWARD = True
FORWARD_OUTPUT_NAMES = ["forward_out0"]
BACKWARD_OUTPUT_NAMES = ["backward_grad_x"] if HAS_BACKWARD else []
OUTPUT_NAMES = FORWARD_OUTPUT_NAMES + BACKWARD_OUTPUT_NAMES
DEFAULT_WORKDIR = WORKSPACE_ROOT / "temp_test" / f"{CASE_PREFIX}_consistency_artifacts"
DEFAULT_OUTPUT_COMPARATOR = framework.default_output_comparator_path()


def build_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "basic_fp32",
            "seed": 101,
            "input": np.array([0.0], dtype=np.float32),
        }
    ]


class ExampleCell(nn.Cell):
    """Preferred Graph-mode wrapper. Replace `construct` with the target op call."""

    def construct(self, x):
        raise NotImplementedError("Replace with the Graph-mode forward path")


def example_forward_func(x):
    """PyNative-mode forward path.

    Keep this as the direct operator call used for PyNative mode. If you need a
    Graph-mode function instead of a `Cell`, define a separate named `@ms.jit`
    function.
    """
    raise NotImplementedError("Replace with the PyNative-mode forward path")


class ExampleGradCell(nn.Cell):
    def __init__(self):
        super().__init__()
        self.forward = ExampleCell()
        self.grad = ops.GradOperation(get_all=False, sens_param=True)(self.forward)

    def construct(self, x, sens):
        return self.grad(x, sens)


def run_ms_outputs(case: dict[str, Any], ms_mode: str) -> dict[str, Any]:
    input_tensor = framework.to_ms_tensor(case["input"])
    if ms_mode == "pynative":
        forward = example_forward_func(input_tensor)
    else:
        # In Graph mode, prefer a named `nn.Cell` so `GRAPH_MODE` can compile it
        # directly through `Cell.__call__`.
        forward = ExampleCell()(input_tensor)
    named_outputs = framework.normalize_named_outputs(forward, FORWARD_OUTPUT_NAMES)

    if HAS_BACKWARD:
        sens_source = case.get("sens")
        sens_array = framework.build_linspace_sens(case["input"]) if sens_source is None else sens_source
        sens_tensor = framework.to_ms_tensor(sens_array)
        if ms_mode == "pynative":
            backward = ms.grad(example_forward_func, grad_position=0, sens_param=True)(input_tensor, sens_tensor)
        else:
            # Keep the Graph-mode grad path on top of a named Cell as well.
            backward = ExampleGradCell()(input_tensor, sens_tensor)
        named_outputs.update(framework.normalize_named_outputs(backward, BACKWARD_OUTPUT_NAMES))

    return named_outputs


def run_pta_outputs(case: dict[str, Any], device_id: int) -> dict[str, Any]:
    tensor = framework.to_torch_tensor(case["input"], device_id=device_id, requires_grad=HAS_BACKWARD)
    forward = None  # Replace with the PTA forward path, for example: torch.some_op(tensor)
    if forward is None:
        raise NotImplementedError("Replace with the PTA forward path")

    named_outputs = framework.normalize_named_outputs(forward, FORWARD_OUTPUT_NAMES)

    if HAS_BACKWARD:
        sens_source = case.get("sens")
        sens_array = framework.build_linspace_sens(case["input"]) if sens_source is None else sens_source
        upstream = framework.to_torch_tensor(sens_array, device_id=device_id, requires_grad=False)
        forward.backward(upstream)
        torch.npu.synchronize()
        named_outputs.update(framework.normalize_named_outputs(tensor.grad, BACKWARD_OUTPUT_NAMES))

    return named_outputs


def main() -> int:
    coverage_depth = "forward+backward" if HAS_BACKWARD else "forward"
    args = framework.parse_common_driver_args(
        description=f"Generate representative {OP_NAME} outputs and run MS/PTA consistency validation.",
        default_workdir=DEFAULT_WORKDIR,
        default_output_comparator=DEFAULT_OUTPUT_COMPARATOR,
    )
    return framework.run_driver(
        args=args,
        op_name=OP_NAME,
        build_cases=build_cases,
        run_ms_outputs=run_ms_outputs,
        run_pta_outputs=run_pta_outputs,
        output_names=OUTPUT_NAMES,
        coverage_depth=coverage_depth,
    )


if __name__ == "__main__":
    raise SystemExit(main())
