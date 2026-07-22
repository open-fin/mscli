#!/usr/bin/env python3
"""Reusable execution framework for operator-specific MS/PTA consistency validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable, Mapping


DEFAULT_SINGLE_CASE_RES_SUMMARY_NAME = "ms_pta_consistency_single_case_res_summary.json"
DEFAULT_ALL_CASE_RES_SUMMARY_NAME = "ms_pta_consistency_all_case_res_summary.json"
DEFAULT_CASE_SPEC_NAME = "ms_pta_consistency_case_spec.json"
OUTPUT_COMPARATOR_SCRIPT_NAME = "ms_pta_consistency_output_comparator.py"


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("numpy is required for MS/PTA driver helpers") from exc
    return np


def load_script_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_output_comparator_path() -> Path:
    return Path(__file__).resolve().with_name(OUTPUT_COMPARATOR_SCRIPT_NAME)


def add_common_driver_args(
    parser: argparse.ArgumentParser,
    *,
    default_workdir: Path,
    default_output_comparator: Path,
) -> None:
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=default_workdir,
        help="Directory used to store outputs, case specs, and summaries.",
    )
    parser.add_argument(
        "--output-comparator",
        "--checker-script",
        dest="output_comparator",
        type=Path,
        default=default_output_comparator,
        help="Path to the generic MS/PTA output comparator script.",
    )
    parser.add_argument(
        "--strategy",
        choices=["semantic_zero", "bitwise_strict"],
        default="bitwise_strict",
        help="Consistency strategy passed to the generic output comparator.",
    )
    parser.add_argument(
        "--ms-mode",
        choices=["pynative", "graph", "both"],
        default="both",
        help="MindSpore execution mode selection. Defaults to both PyNative mode and Graph mode.",
    )


def parse_common_driver_args(
    *,
    description: str,
    default_workdir: Path,
    default_output_comparator: Path,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    add_common_driver_args(
        parser,
        default_workdir=default_workdir,
        default_output_comparator=default_output_comparator,
    )
    return parser.parse_args()


def prepare_runtime(device_id: int, ms_mode: str) -> None:
    import mindspore as ms  # type: ignore
    import torch  # type: ignore
    import torch_npu  # noqa: F401  # type: ignore

    torch.npu.set_device(device_id)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.use_deterministic_algorithms(True)

    ms.context.set_context(
        mode=ms.PYNATIVE_MODE if ms_mode == "pynative" else ms.GRAPH_MODE,
        device_target="Ascend",
        device_id=device_id,
        deterministic="ON",
        pynative_synchronize=False,
    )


def reset_seeds(seed: int) -> None:
    import mindspore as ms  # type: ignore
    import torch  # type: ignore

    np = _require_numpy()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    ms.set_seed(seed)


def get_requested_modes(ms_mode: str) -> list[str]:
    if ms_mode == "both":
        return ["pynative", "graph"]
    return [ms_mode]


def as_numpy_array(value: Any, *, dtype: Any | None = None):
    np = _require_numpy()
    if hasattr(value, "asnumpy"):
        value = value.asnumpy()
    elif hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()
    if dtype is not None:
        return np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.ascontiguousarray(np.asarray(value))


def build_linspace_sens(reference: Any, *, start: float = 0.125, end: float = 0.875):
    np = _require_numpy()
    reference_array = as_numpy_array(reference)
    sens = np.linspace(start, end, reference_array.size, dtype=np.float32).reshape(reference_array.shape)
    return as_numpy_array(sens.astype(reference_array.dtype, copy=False))


def build_default_case_payload(
    case: Mapping[str, Any],
    *,
    has_backward: bool,
    input_name: str = "input",
    sens_name: str = "sens",
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = dict(case)
    input_array = as_numpy_array(payload[input_name])
    payload[input_name] = input_array
    named_inputs = {input_name: input_array}

    if has_backward:
        sens_source = payload.get(sens_name)
        sens_array = as_numpy_array(build_linspace_sens(input_array) if sens_source is None else sens_source)
        payload[sens_name] = sens_array
        named_inputs[sens_name] = sens_array

    return payload, named_inputs


def to_ms_tensor(value: Any):
    from mindspore import Tensor  # type: ignore

    return Tensor(as_numpy_array(value))


def to_torch_tensor(value: Any, *, device_id: int, requires_grad: bool = False):
    import torch  # type: ignore
    import torch_npu  # noqa: F401  # type: ignore

    tensor = torch.from_numpy(as_numpy_array(value).copy()).to(f"npu:{device_id}")
    tensor.requires_grad_(requires_grad)
    return tensor


def normalize_named_outputs(raw_outputs: Any, output_names: list[str]) -> dict[str, Any]:
    if isinstance(raw_outputs, Mapping):
        return {name: as_numpy_array(raw_outputs[name]) for name in output_names}
    if isinstance(raw_outputs, (list, tuple)):
        if len(raw_outputs) != len(output_names):
            raise ValueError(
                f"Output count mismatch: expected {len(output_names)} values, got {len(raw_outputs)}"
            )
        return {name: as_numpy_array(value) for name, value in zip(output_names, raw_outputs)}
    if len(output_names) != 1:
        raise ValueError("Single raw output can only be normalized against one output name")
    return {output_names[0]: as_numpy_array(raw_outputs)}


def save_npy(path: Path, value: Any) -> None:
    np = _require_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, as_numpy_array(value), allow_pickle=False)


def ensure_case_inputs(case_root: Path, named_inputs: Mapping[str, Any]) -> None:
    for name, value in named_inputs.items():
        path = case_root / f"{name}.npy"
        if not path.exists():
            save_npy(path, value)


def write_named_outputs(output_dir: Path, named_outputs: Mapping[str, Any]) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in named_outputs.items():
        save_npy(output_dir / f"{name}.npy", value)
    return str(output_dir)


def ensure_cached_outputs(
    *,
    cache: dict[str, str],
    cache_key: str,
    output_dir: Path,
    producer: Callable[[], Mapping[str, Any]],
) -> str:
    if cache_key not in cache:
        cache[cache_key] = write_named_outputs(output_dir, producer())
    return cache[cache_key]


def build_case_record(
    *,
    case_id: str,
    ms_dir: str,
    pta_dir: str,
    output_names: list[str],
    strategy: str,
    ms_mode: str,
    coverage_depth: str,
    summary_out: Path,
    rtol: float = 0.0,
    atol: float = 0.0,
    equal_nan: bool = True,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "ms_dir": ms_dir,
        "pta_dir": pta_dir,
        "outputs": output_names,
        "strategy": strategy,
        "ms_mode": ms_mode,
        "coverage_depth": coverage_depth,
        "rtol": rtol,
        "atol": atol,
        "equal_nan": equal_nan,
        "summary_out": str(summary_out),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_case_spec_validation(
    *,
    comparator: Any,
    workdir: Path,
    case_specs: list[dict[str, Any]],
    case_spec_fields: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    case_spec = {
        "summary_out": str(workdir / DEFAULT_ALL_CASE_RES_SUMMARY_NAME),
        "cases": case_specs,
    }
    if case_spec_fields:
        case_spec.update(dict(case_spec_fields))

    case_spec_path = workdir / DEFAULT_CASE_SPEC_NAME
    write_json(case_spec_path, case_spec)
    batch_payload = comparator.compare_case_spec_file(case_spec_path)
    return case_spec_path, batch_payload


def build_driver_case_specs(
    *,
    args: argparse.Namespace,
    build_cases: Callable[[], list[dict[str, Any]]],
    run_ms_outputs: Callable[[dict[str, Any], str], Mapping[str, Any]],
    run_pta_outputs: Callable[[dict[str, Any], int], Mapping[str, Any]],
    output_names: list[str],
    coverage_depth: str,
    prepare_case_payload: Callable[[Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    case_specs: list[dict[str, Any]] = []
    pta_cache: dict[str, str] = {}
    has_backward = coverage_depth != "forward"

    for ms_mode in get_requested_modes(args.ms_mode):
        for case in build_cases():
            reset_seeds(case["seed"])
            prepare_runtime(args.device_id, ms_mode)

            case_root = args.workdir / case["case_id"]
            case_dir = case_root / ms_mode
            case_dir.mkdir(parents=True, exist_ok=True)

            if prepare_case_payload is None:
                case_payload, named_inputs = build_default_case_payload(case, has_backward=has_backward)
            else:
                case_payload, named_inputs = prepare_case_payload(case)

            ensure_case_inputs(case_root, named_inputs)
            ms_dir = write_named_outputs(case_dir / "ms_outputs", run_ms_outputs(case_payload, ms_mode))
            pta_dir = ensure_cached_outputs(
                cache=pta_cache,
                cache_key=case["case_id"],
                output_dir=case_root / "pta_outputs",
                producer=lambda current_case=case_payload: run_pta_outputs(current_case, args.device_id),
            )

            case_specs.append(
                build_case_record(
                    case_id=f"{case['case_id']}_{ms_mode}",
                    ms_dir=ms_dir,
                    pta_dir=pta_dir,
                    output_names=output_names,
                    strategy=args.strategy,
                    ms_mode=ms_mode,
                    coverage_depth=coverage_depth,
                    summary_out=case_dir / DEFAULT_SINGLE_CASE_RES_SUMMARY_NAME,
                )
            )

    return case_specs


def run_driver(
    *,
    args: argparse.Namespace,
    op_name: str,
    build_cases: Callable[[], list[dict[str, Any]]],
    run_ms_outputs: Callable[[dict[str, Any], str], Mapping[str, Any]],
    run_pta_outputs: Callable[[dict[str, Any], int], Mapping[str, Any]],
    output_names: list[str],
    coverage_depth: str,
    prepare_case_payload: Callable[[Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]] | None = None,
    extra_case_spec_fields: Mapping[str, Any] | None = None,
) -> int:
    comparator = load_script_module("ms_pta_consistency_output_comparator", args.output_comparator)
    args.workdir.mkdir(parents=True, exist_ok=True)
    case_spec_fields = {
        "op_name": op_name,
        "ms_mode": args.ms_mode,
        "coverage_depth": coverage_depth,
    }
    if extra_case_spec_fields:
        case_spec_fields.update(dict(extra_case_spec_fields))

    _, batch_payload = run_case_spec_validation(
        comparator=comparator,
        workdir=args.workdir,
        case_specs=build_driver_case_specs(
            args=args,
            build_cases=build_cases,
            run_ms_outputs=run_ms_outputs,
            run_pta_outputs=run_pta_outputs,
            output_names=output_names,
            coverage_depth=coverage_depth,
            prepare_case_payload=prepare_case_payload,
        ),
        case_spec_fields=case_spec_fields,
    )
    print(json.dumps(batch_payload, ensure_ascii=False, indent=2))
    return 0 if batch_payload["all_passed"] else 1
