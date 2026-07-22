#!/usr/bin/env python3
"""Reusable execution framework for operator dtype probing."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DEFAULT_SUMMARY_NAME = "op_dtype_probe_summary.json"
DEFAULT_MARKDOWN_NAME = "op_dtype_probe_summary.md"

DEFAULT_DTYPE_NAMES = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)

DEFAULT_UNSUPPORTED_ERROR_PATTERNS = (
    r"unsupported\s+(dtype|data type)",
    r"(dtype|data type).*(is\s+)?not supported",
    r"not support.*(dtype|data type)",
    r"only support[s]?.*(dtype|data type)",
    r"(dtype|data type).*(support list)",
    r"implemented for dt_[a-z0-9_]+",
    r"cast.*(dtype|type).*(not supported|unsupported)",
    r"grad.*(dtype|type).*(not supported|unsupported)",
)

DEFAULT_UNSUPPORTED_ERROR_KEYWORDS = (
    "unsupported dtype",
    "unsupported data type",
    "dtype is not supported",
    "data type is not supported",
    "dtype support list",
    "not implemented for dt_",
)


BACKWARD_NOT_APPLICABLE = object()


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("numpy is required for dtype probing") from exc
    return np


def _require_mindspore():
    try:
        import mindspore as ms  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("mindspore is required for dtype probing") from exc
    return ms


def dtype_registry() -> dict[str, Any]:
    ms = _require_mindspore()
    return {
        "bool": ms.bool_,
        "int8": ms.int8,
        "int16": ms.int16,
        "int32": ms.int32,
        "int64": ms.int64,
        "uint8": ms.uint8,
        "uint16": ms.uint16,
        "uint32": ms.uint32,
        "uint64": ms.uint64,
        "float16": ms.float16,
        "bfloat16": ms.bfloat16,
        "float32": ms.float32,
        "float64": ms.float64,
        "complex64": ms.complex64,
        "complex128": ms.complex128,
    }


def normalize_dtype_name(dtype: Any) -> str:
    registry = dtype_registry()
    for name, value in registry.items():
        if dtype == value:
            return name

    text = str(dtype).lower()
    for prefix in ("mindspore.", "mstype.", "ms."):
        text = text.replace(prefix, "")
    text = text.replace("_", "")
    aliases = {
        "bool": "bool",
        "bool_": "bool",
        "float": "float32",
        "double": "float64",
        "half": "float16",
    }
    return aliases.get(text, text)


def resolve_dtype(dtype_name: str) -> Any:
    normalized = normalize_dtype_name(dtype_name)
    registry = dtype_registry()
    if normalized not in registry:
        raise KeyError(f"Unknown dtype name: {dtype_name}")
    return registry[normalized]


def _numpy_dtype_for_name(dtype_name: str):
    np = _require_numpy()
    mapping = {
        "bool": np.bool_,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "float16": np.float16,
        "bfloat16": np.float32,
        "float32": np.float32,
        "float64": np.float64,
        "complex64": np.complex64,
        "complex128": np.complex128,
    }
    return mapping[dtype_name]


def clone_value(value: Any) -> Any:
    ms = _require_mindspore()
    np = _require_numpy()

    if isinstance(value, ms.Tensor):
        return ms.Tensor(value.asnumpy(), dtype=value.dtype)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return [clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: clone_value(item) for key, item in value.items()}
    return value


@dataclass
class ProbeSample:
    """Represents one operator invocation used by the dtype probe."""

    op_input: Any = None
    op_args: tuple[Any, ...] = ()
    op_kwargs: dict[str, Any] = field(default_factory=dict)
    sample_name: str = "basic"
    grad_position: tuple[int, ...] | None = None
    backward_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def positional_args(self) -> tuple[Any, ...]:
        if self.op_input is None:
            return tuple(self.op_args)
        return (self.op_input, *self.op_args)

    def clone(self) -> "ProbeSample":
        return ProbeSample(
            op_input=clone_value(self.op_input),
            op_args=clone_value(self.op_args),
            op_kwargs=clone_value(self.op_kwargs),
            sample_name=self.sample_name,
            grad_position=self.grad_position,
            backward_enabled=self.backward_enabled,
            metadata=clone_value(self.metadata),
        )


@dataclass
class OperatorProbeSpec:
    """Operator-specific probe configuration."""

    op_name: str
    build_samples: Callable[[str, Any], Sequence[ProbeSample]]
    forward_runner: Callable[[ProbeSample], Any]
    backward_runner: Callable[[ProbeSample], Any] | None = None
    probe_backend: str = "pta"
    candidate_dtypes: Sequence[str] = field(default_factory=lambda: DEFAULT_DTYPE_NAMES)
    declared_doc_forward_dtypes: Sequence[str] = field(default_factory=tuple)
    declared_doc_backward_dtypes: Sequence[str] = field(default_factory=tuple)
    supports_backward: bool = True
    notes: str = ""
    unsupported_error_patterns: Sequence[str] = field(default_factory=tuple)
    unsupported_error_keywords: Sequence[str] = field(default_factory=tuple)
    error_classifier: Callable[[BaseException, str, str, ProbeSample | None], str | None] | None = None


def coerce_probe_sample(raw_sample: Any) -> ProbeSample:
    if isinstance(raw_sample, ProbeSample):
        return raw_sample
    required_attrs = ("op_input", "op_args", "op_kwargs", "sample_name", "grad_position", "backward_enabled")
    if all(hasattr(raw_sample, attr) for attr in required_attrs):
        return ProbeSample(
            op_input=getattr(raw_sample, "op_input"),
            op_args=tuple(getattr(raw_sample, "op_args")),
            op_kwargs=dict(getattr(raw_sample, "op_kwargs")),
            sample_name=getattr(raw_sample, "sample_name"),
            grad_position=getattr(raw_sample, "grad_position"),
            backward_enabled=getattr(raw_sample, "backward_enabled"),
            metadata=dict(getattr(raw_sample, "metadata", {})),
        )
    return ProbeSample(op_input=raw_sample)


def coerce_probe_spec(raw_spec: Any) -> OperatorProbeSpec:
    if isinstance(raw_spec, OperatorProbeSpec):
        return raw_spec
    required_attrs = ("op_name", "build_samples", "forward_runner")
    if not all(hasattr(raw_spec, attr) for attr in required_attrs):
        raise TypeError(f"Invalid probe spec {type(raw_spec)!r}; expected OperatorProbeSpec-compatible object.")
    return OperatorProbeSpec(
        op_name=getattr(raw_spec, "op_name"),
        build_samples=getattr(raw_spec, "build_samples"),
        forward_runner=getattr(raw_spec, "forward_runner"),
        backward_runner=getattr(raw_spec, "backward_runner", None),
        probe_backend=str(getattr(raw_spec, "probe_backend", "pta")),
        candidate_dtypes=tuple(getattr(raw_spec, "candidate_dtypes", DEFAULT_DTYPE_NAMES)),
        declared_doc_forward_dtypes=tuple(getattr(raw_spec, "declared_doc_forward_dtypes", tuple())),
        declared_doc_backward_dtypes=tuple(getattr(raw_spec, "declared_doc_backward_dtypes", tuple())),
        supports_backward=bool(getattr(raw_spec, "supports_backward", True)),
        notes=str(getattr(raw_spec, "notes", "")),
        unsupported_error_patterns=tuple(getattr(raw_spec, "unsupported_error_patterns", tuple())),
        unsupported_error_keywords=tuple(getattr(raw_spec, "unsupported_error_keywords", tuple())),
        error_classifier=getattr(raw_spec, "error_classifier", None),
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare_runtime(*, device_target: str, device_id: int) -> dict[str, Any]:
    ms = _require_mindspore()
    os.environ.setdefault("MS_DISABLE_KERNEL_BACKOFF", "1")

    runtime = {
        "device_target": device_target,
        "device_id": device_id,
        "ms_mode": "pynative",
        "ascend_soc": None,
    }

    normalized_target = device_target.lower()
    if normalized_target == "ascend":
        ms.context.set_context(
            mode=ms.PYNATIVE_MODE,
            device_target="Ascend",
            device_id=device_id,
            deterministic="ON",
            pynative_synchronize=False,
        )
        try:
            from mindspore._c_expression import MSContext  # type: ignore

            runtime["ascend_soc"] = MSContext.get_instance().get_ascend_soc_version()
        except Exception:  # pragma: no cover
            runtime["ascend_soc"] = None
    elif normalized_target == "cpu":
        ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    elif normalized_target == "gpu":
        ms.context.set_context(mode=ms.PYNATIVE_MODE, device_target="GPU")
    else:
        raise ValueError(f"Unsupported device target: {device_target}")
    return runtime


def _numel(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def make_tensor(
    shape: Sequence[int] | tuple[int, ...],
    dtype: str | Any,
    *,
    low: float = -3.0,
    high: float = 3.0,
    values: Sequence[Any] | Any | None = None,
) -> Any:
    """Build a deterministic MindSpore tensor for probe samples."""

    ms = _require_mindspore()
    np = _require_numpy()
    ms_dtype = resolve_dtype(dtype) if isinstance(dtype, str) else dtype
    dtype_name = normalize_dtype_name(ms_dtype)
    shape = tuple(shape)
    size = _numel(shape) if shape else 1

    if values is not None:
        array = np.asarray(values)
        array = array.reshape(shape if shape else ())
    elif dtype_name == "bool":
        array = (np.arange(size) % 2 == 0).reshape(shape if shape else ())
    elif dtype_name.startswith("uint"):
        array = (np.arange(size) % 11).reshape(shape if shape else ()).astype(_numpy_dtype_for_name(dtype_name))
    elif dtype_name.startswith("int"):
        base = np.arange(size) - (size // 2)
        array = base.reshape(shape if shape else ()).astype(_numpy_dtype_for_name(dtype_name))
    elif dtype_name in ("float16", "bfloat16", "float32", "float64"):
        array = np.linspace(low, high, num=size, dtype=_numpy_dtype_for_name(dtype_name))
        array = array.reshape(shape if shape else ())
    elif dtype_name in ("complex64", "complex128"):
        base = np.linspace(low, high, num=size, dtype=np.float32 if dtype_name == "complex64" else np.float64)
        imag = np.flip(base)
        array = (base + 1j * imag).reshape(shape if shape else ()).astype(_numpy_dtype_for_name(dtype_name))
    else:
        raise ValueError(f"Unhandled dtype: {dtype_name}")

    if dtype_name == "bfloat16":
        return ms.Tensor(array.astype(np.float32), dtype=ms_dtype)
    return ms.Tensor(array, dtype=ms_dtype)


def make_unary_sample(
    dtype: str | Any,
    *,
    shape: Sequence[int] = (2, 3),
    low: float = -0.75,
    high: float = 0.75,
    sample_name: str = "basic",
    op_args: Sequence[Any] = (),
    op_kwargs: Mapping[str, Any] | None = None,
    grad_position: tuple[int, ...] | None = (0,),
) -> ProbeSample:
    return ProbeSample(
        op_input=make_tensor(shape, dtype, low=low, high=high),
        op_args=tuple(op_args),
        op_kwargs=dict(op_kwargs or {}),
        sample_name=sample_name,
        grad_position=grad_position,
    )


def make_binary_same_dtype_sample(
    dtype: str | Any,
    *,
    input_shape: Sequence[int] = (2, 3),
    other_shape: Sequence[int] = (2, 3),
    input_low: float = -1.0,
    input_high: float = 1.0,
    other_low: float = -0.5,
    other_high: float = 0.5,
    sample_name: str = "basic",
    op_kwargs: Mapping[str, Any] | None = None,
    grad_position: tuple[int, ...] | None = (0,),
) -> ProbeSample:
    return ProbeSample(
        op_input=make_tensor(input_shape, dtype, low=input_low, high=input_high),
        op_args=(make_tensor(other_shape, dtype, low=other_low, high=other_high),),
        op_kwargs=dict(op_kwargs or {}),
        sample_name=sample_name,
        grad_position=grad_position,
    )


def make_reduction_sample(
    dtype: str | Any,
    *,
    shape: Sequence[int] = (2, 3, 4),
    low: float = -1.0,
    high: float = 1.0,
    op_args: Sequence[Any] = (),
    op_kwargs: Mapping[str, Any] | None = None,
    sample_name: str = "basic",
    grad_position: tuple[int, ...] | None = (0,),
) -> ProbeSample:
    return ProbeSample(
        op_input=make_tensor(shape, dtype, low=low, high=high),
        op_args=tuple(op_args),
        op_kwargs=dict(op_kwargs or {}),
        sample_name=sample_name,
        grad_position=grad_position,
    )


def invoke_operator(op: Callable[..., Any], sample: ProbeSample) -> Any:
    positional_args = sample.positional_args()
    return op(*positional_args, **sample.op_kwargs)


def make_default_forward_runner(op: Callable[..., Any]) -> Callable[[ProbeSample], Any]:
    def _runner(sample: ProbeSample) -> Any:
        return invoke_operator(op, sample)

    return _runner


def _contains_differentiable_tensor(value: Any) -> bool:
    ms = _require_mindspore()

    if isinstance(value, ms.Tensor):
        return value.dtype.is_floating_point or value.dtype.is_complex
    if isinstance(value, (list, tuple)):
        return any(_contains_differentiable_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_differentiable_tensor(item) for item in value.values())
    return False


def infer_grad_position(sample: ProbeSample) -> tuple[int, ...]:
    return tuple(
        index
        for index, value in enumerate(sample.positional_args())
        if _contains_differentiable_tensor(value)
    )


def build_default_sens(forward_output: Any) -> Any:
    ms = _require_mindspore()
    np = _require_numpy()

    if isinstance(forward_output, ms.Tensor):
        dtype_name = normalize_dtype_name(forward_output.dtype)
        numpy_dtype = np.float32 if dtype_name == "bfloat16" else _numpy_dtype_for_name(dtype_name)
        ones = np.ones(forward_output.shape if forward_output.shape else (), dtype=numpy_dtype)
        return ms.Tensor(ones, dtype=forward_output.dtype)
    if isinstance(forward_output, tuple):
        return tuple(build_default_sens(item) for item in forward_output)
    if isinstance(forward_output, list):
        return [build_default_sens(item) for item in forward_output]
    raise TypeError(
        "Default sens builder only supports Tensor or tuple/list of Tensor outputs. "
        "Provide a custom backward runner for this operator."
    )


def make_default_backward_runner(
    op: Callable[..., Any],
    *,
    grad_position: tuple[int, ...] | None = (0,),
    sens_builder: Callable[[Any], Any] = build_default_sens,
) -> Callable[[ProbeSample], Any]:
    ms = _require_mindspore()

    def _runner(sample: ProbeSample) -> Any:
        local_sample = sample.clone()
        effective_grad_position = local_sample.grad_position if local_sample.grad_position is not None else grad_position
        if effective_grad_position is None:
            effective_grad_position = infer_grad_position(local_sample)
        if effective_grad_position == ():
            return BACKWARD_NOT_APPLICABLE

        positional_args = local_sample.positional_args()
        op_kwargs = clone_value(local_sample.op_kwargs)
        forward_out = op(*clone_value(positional_args), **clone_value(op_kwargs))
        sens = sens_builder(forward_out)
        grad_fn = ms.grad(
            lambda *call_args: op(*call_args, **clone_value(op_kwargs)),
            grad_position=effective_grad_position,
            sens_param=True,
        )
        return grad_fn(*positional_args, sens)

    return _runner


def format_exception_message(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def classify_failure(
    exc: BaseException,
    *,
    spec: OperatorProbeSpec,
    direction: str,
    dtype_name: str,
    sample: ProbeSample | None,
) -> str:
    if spec.error_classifier is not None:
        classification = spec.error_classifier(exc, direction, dtype_name, sample)
        if classification:
            return classification

    text = format_exception_message(exc).lower()
    patterns = (*DEFAULT_UNSUPPORTED_ERROR_PATTERNS, *spec.unsupported_error_patterns)
    keywords = tuple(keyword.lower() for keyword in (*DEFAULT_UNSUPPORTED_ERROR_KEYWORDS, *spec.unsupported_error_keywords))

    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
        return "unsupported_dtype"
    if any(keyword in text for keyword in keywords):
        return "unsupported_dtype"
    return "sample_or_function_issue"


def _dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = normalize_dtype_name(value)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _build_case_record(
    *,
    op_name: str,
    dtype_name: str,
    direction: str,
    status: str,
    sample_name: str | None = None,
    error_message: str | None = None,
    phase: str = "runtime",
) -> dict[str, Any]:
    record = {
        "op_name": op_name,
        "dtype": dtype_name,
        "direction": direction,
        "status": status,
        "phase": phase,
    }
    if sample_name is not None:
        record["sample_name"] = sample_name
    if error_message:
        record["error_message"] = error_message
    return record


def _aggregate_direction_status(case_records: Sequence[dict[str, Any]], *, supports_backward: bool) -> str:
    if not case_records:
        return "not_applicable" if not supports_backward else "sample_or_function_issue"
    statuses = [record["status"] for record in case_records]
    effective_statuses = [status for status in statuses if status != "not_applicable"]
    if not effective_statuses:
        return "not_applicable"
    if all(status == "supported" for status in effective_statuses):
        return "supported"
    if all(status == "unsupported_dtype" for status in effective_statuses):
        return "unsupported_dtype"
    return "sample_or_function_issue"


def _compare_declared_vs_probe(
    declared: Sequence[str],
    observed_supported: Sequence[str],
) -> dict[str, list[str]] | None:
    if not declared:
        return None
    declared_set = set(_dedupe_preserve_order(declared))
    observed_set = set(_dedupe_preserve_order(observed_supported))
    return {
        "declared_only": sorted(declared_set - observed_set),
        "probe_only": sorted(observed_set - declared_set),
    }


def _has_probe_doc_conflict(diff: Mapping[str, Sequence[str]] | None) -> bool:
    if not diff:
        return False
    return bool(diff.get("declared_only") or diff.get("probe_only"))


def _resolve_dtype_declaration_source(
    *,
    probe_backend: str,
    forward_diff: Mapping[str, Sequence[str]] | None,
    backward_diff: Mapping[str, Sequence[str]] | None,
) -> str:
    if probe_backend == "doc":
        return "doc_derived"
    if _has_probe_doc_conflict(forward_diff) or _has_probe_doc_conflict(backward_diff):
        return "probe_vs_doc_conflict"
    return "probe_verified"


def _build_writeback_guidance(*, dtype_declaration_source: str, probe_backend: str) -> str:
    if dtype_declaration_source == "doc_derived" or probe_backend == "doc":
        return (
            "Documentation-only fallback. Treat these dtypes as candidate writeback evidence "
            "until runtime probing or OpInfo validation confirms them."
        )
    if dtype_declaration_source == "probe_vs_doc_conflict":
        return "Review the declared-vs-probed dtype diff before writing dtypes_*."
    if probe_backend == "pta":
        return (
            "Backfill the OpInfo dtype declaration from the PTA runtime probe first, "
            "then use the MindSpore opinfo case tests to clarify any divergence."
        )
    return "MindSpore runtime probing can be used as the primary writeback evidence for dtypes_*."


def execute_operator_probe(spec: OperatorProbeSpec) -> dict[str, Any]:
    dtype_records: list[dict[str, Any]] = []
    candidate_dtypes = _dedupe_preserve_order(spec.candidate_dtypes)

    for dtype_name in candidate_dtypes:
        ms_dtype = resolve_dtype(dtype_name)
        try:
            raw_samples = list(spec.build_samples(dtype_name, ms_dtype))
        except Exception as exc:
            error_message = format_exception_message(exc)
            dtype_records.append(
                {
                    "dtype": dtype_name,
                    "forward_status": "sample_or_function_issue",
                    "forward_cases": [
                        _build_case_record(
                            op_name=spec.op_name,
                            dtype_name=dtype_name,
                            direction="forward",
                            status="sample_or_function_issue",
                            error_message=error_message,
                            phase="sample_build",
                        )
                    ],
                    "backward_status": "not_applicable" if not spec.supports_backward else "sample_or_function_issue",
                    "backward_cases": [] if not spec.supports_backward else [
                        _build_case_record(
                            op_name=spec.op_name,
                            dtype_name=dtype_name,
                            direction="backward",
                            status="sample_or_function_issue",
                            error_message=error_message,
                            phase="sample_build",
                        )
                    ],
                }
            )
            continue

        forward_case_records: list[dict[str, Any]] = []
        backward_case_records: list[dict[str, Any]] = []
        for raw_sample in raw_samples:
            sample = coerce_probe_sample(raw_sample)

            try:
                spec.forward_runner(sample.clone())
                forward_case_records.append(
                    _build_case_record(
                        op_name=spec.op_name,
                        dtype_name=dtype_name,
                        direction="forward",
                        status="supported",
                        sample_name=sample.sample_name,
                    )
                )
            except Exception as exc:
                status = classify_failure(exc, spec=spec, direction="forward", dtype_name=dtype_name, sample=sample)
                forward_case_records.append(
                    _build_case_record(
                        op_name=spec.op_name,
                        dtype_name=dtype_name,
                        direction="forward",
                        status=status,
                        sample_name=sample.sample_name,
                        error_message=format_exception_message(exc),
                    )
                )

            if not spec.supports_backward:
                backward_case_records.append(
                    _build_case_record(
                        op_name=spec.op_name,
                        dtype_name=dtype_name,
                        direction="backward",
                        status="not_applicable",
                        sample_name=sample.sample_name,
                    )
                )
                continue

            if not sample.backward_enabled:
                backward_case_records.append(
                    _build_case_record(
                        op_name=spec.op_name,
                        dtype_name=dtype_name,
                        direction="backward",
                        status="not_applicable",
                        sample_name=sample.sample_name,
                    )
                )
                continue

            if spec.backward_runner is None:
                backward_case_records.append(
                    _build_case_record(
                        op_name=spec.op_name,
                        dtype_name=dtype_name,
                        direction="backward",
                        status="sample_or_function_issue",
                        sample_name=sample.sample_name,
                        error_message="No backward runner configured for this operator probe.",
                    )
                )
                continue

            try:
                backward_result = spec.backward_runner(sample.clone())
                if backward_result is BACKWARD_NOT_APPLICABLE:
                    backward_case_records.append(
                        _build_case_record(
                            op_name=spec.op_name,
                            dtype_name=dtype_name,
                            direction="backward",
                            status="not_applicable",
                            sample_name=sample.sample_name,
                        )
                    )
                    continue
                backward_case_records.append(
                    _build_case_record(
                        op_name=spec.op_name,
                        dtype_name=dtype_name,
                        direction="backward",
                        status="supported",
                        sample_name=sample.sample_name,
                    )
                )
            except Exception as exc:
                status = classify_failure(exc, spec=spec, direction="backward", dtype_name=dtype_name, sample=sample)
                backward_case_records.append(
                    _build_case_record(
                        op_name=spec.op_name,
                        dtype_name=dtype_name,
                        direction="backward",
                        status=status,
                        sample_name=sample.sample_name,
                        error_message=format_exception_message(exc),
                    )
                )

        dtype_records.append(
            {
                "dtype": dtype_name,
                "forward_status": _aggregate_direction_status(forward_case_records, supports_backward=True),
                "forward_cases": forward_case_records,
                "backward_status": _aggregate_direction_status(
                    backward_case_records,
                    supports_backward=spec.supports_backward,
                ),
                "backward_cases": backward_case_records,
            }
        )

    forward_supported = [record["dtype"] for record in dtype_records if record["forward_status"] == "supported"]
    backward_supported = [record["dtype"] for record in dtype_records if record["backward_status"] == "supported"]
    backward_not_applicable = [record["dtype"] for record in dtype_records if record["backward_status"] == "not_applicable"]
    forward_unsupported = [record["dtype"] for record in dtype_records if record["forward_status"] == "unsupported_dtype"]
    backward_unsupported = [record["dtype"] for record in dtype_records if record["backward_status"] == "unsupported_dtype"]
    forward_issues = [record["dtype"] for record in dtype_records if record["forward_status"] == "sample_or_function_issue"]
    backward_issues = [record["dtype"] for record in dtype_records if record["backward_status"] == "sample_or_function_issue"]
    forward_diff = _compare_declared_vs_probe(spec.declared_doc_forward_dtypes, forward_supported)
    backward_diff = _compare_declared_vs_probe(spec.declared_doc_backward_dtypes, backward_supported)
    dtype_declaration_source = _resolve_dtype_declaration_source(
        probe_backend=spec.probe_backend,
        forward_diff=forward_diff,
        backward_diff=backward_diff,
    )

    return {
        "op_name": spec.op_name,
        "probe_backend": spec.probe_backend,
        "dtype_declaration_source": dtype_declaration_source,
        "writeback_guidance": _build_writeback_guidance(
            dtype_declaration_source=dtype_declaration_source,
            probe_backend=spec.probe_backend,
        ),
        "candidate_dtypes": candidate_dtypes,
        "declared_doc_forward_dtypes": _dedupe_preserve_order(spec.declared_doc_forward_dtypes),
        "declared_doc_backward_dtypes": _dedupe_preserve_order(spec.declared_doc_backward_dtypes),
        "supports_backward": spec.supports_backward,
        "notes": spec.notes,
        "forward_supported_dtypes": forward_supported,
        "backward_supported_dtypes": backward_supported,
        "backward_not_applicable_dtypes": backward_not_applicable,
        "forward_unsupported_dtypes": forward_unsupported,
        "backward_unsupported_dtypes": backward_unsupported,
        "forward_issue_dtypes": forward_issues,
        "backward_issue_dtypes": backward_issues,
        "doc_probe_forward_diff": forward_diff,
        "doc_probe_backward_diff": backward_diff,
        "dtype_records": dtype_records,
    }


def build_doc_derived_operator_summary(
    *,
    op_name: str,
    declared_doc_forward_dtypes: Sequence[str],
    declared_doc_backward_dtypes: Sequence[str] = (),
    candidate_dtypes: Sequence[str] | None = None,
    supports_backward: bool = True,
    notes: str = "",
) -> dict[str, Any]:
    """Build a probe-shaped summary for documentation-only fallback flows."""

    forward_doc = _dedupe_preserve_order(declared_doc_forward_dtypes)
    backward_doc = _dedupe_preserve_order(declared_doc_backward_dtypes)
    candidate = _dedupe_preserve_order(candidate_dtypes or forward_doc or backward_doc)

    return {
        "op_name": op_name,
        "probe_backend": "doc",
        "dtype_declaration_source": _resolve_dtype_declaration_source(
            probe_backend="doc",
            forward_diff=None,
            backward_diff=None,
        ),
        "writeback_guidance": _build_writeback_guidance(
            dtype_declaration_source="doc_derived",
            probe_backend="doc",
        ),
        "candidate_dtypes": candidate,
        "declared_doc_forward_dtypes": forward_doc,
        "declared_doc_backward_dtypes": backward_doc,
        "supports_backward": supports_backward,
        "notes": notes,
        "runtime_probe_skipped": True,
        "forward_supported_dtypes": forward_doc,
        "backward_supported_dtypes": backward_doc if supports_backward else [],
        "backward_not_applicable_dtypes": [] if supports_backward else candidate,
        "forward_unsupported_dtypes": [],
        "backward_unsupported_dtypes": [],
        "forward_issue_dtypes": [],
        "backward_issue_dtypes": [],
        "doc_probe_forward_diff": None,
        "doc_probe_backward_diff": None,
        "dtype_records": [],
    }


def default_markdown_path(summary_out: Path) -> Path:
    return summary_out.with_suffix(".md")


def render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Operator dtype probe summary",
        "",
        f"- device_target: {summary['runtime']['device_target']}",
        f"- device_id: {summary['runtime']['device_id']}",
        f"- ms_mode: {summary['runtime']['ms_mode']}",
    ]
    if summary["runtime"].get("ascend_soc"):
        lines.append(f"- ascend_soc: {summary['runtime']['ascend_soc']}")
    lines.extend([f"- operators: {len(summary['operators'])}", ""])

    for operator in summary["operators"]:
        lines.extend(
            [
                f"## {operator['op_name']}",
                "",
                f"- probe backend: {operator.get('probe_backend', 'pta')}",
                f"- dtype declaration source: {operator.get('dtype_declaration_source', 'probe_verified')}",
                f"- candidate dtypes: {', '.join(operator['candidate_dtypes']) or '(none)'}",
                f"- forward supported: {', '.join(operator['forward_supported_dtypes']) or '(none)'}",
                f"- backward supported: {', '.join(operator['backward_supported_dtypes']) or '(none)'}",
                f"- backward not applicable: {', '.join(operator.get('backward_not_applicable_dtypes', [])) or '(none)'}",
                f"- forward unsupported: {', '.join(operator['forward_unsupported_dtypes']) or '(none)'}",
                f"- backward unsupported: {', '.join(operator['backward_unsupported_dtypes']) or '(none)'}",
                f"- forward issues: {', '.join(operator['forward_issue_dtypes']) or '(none)'}",
                f"- backward issues: {', '.join(operator['backward_issue_dtypes']) or '(none)'}",
            ]
        )
        if operator["declared_doc_forward_dtypes"]:
            lines.append(f"- doc declared forward dtypes: {', '.join(operator['declared_doc_forward_dtypes'])}")
        if operator["declared_doc_backward_dtypes"]:
            lines.append(f"- doc declared backward dtypes: {', '.join(operator['declared_doc_backward_dtypes'])}")
        if operator["doc_probe_forward_diff"]:
            lines.append(
                "- doc/probe forward diff: "
                f"declared_only={operator['doc_probe_forward_diff']['declared_only']}, "
                f"probe_only={operator['doc_probe_forward_diff']['probe_only']}"
            )
        if operator["doc_probe_backward_diff"]:
            lines.append(
                "- doc/probe backward diff: "
                f"declared_only={operator['doc_probe_backward_diff']['declared_only']}, "
                f"probe_only={operator['doc_probe_backward_diff']['probe_only']}"
            )
        if operator["notes"]:
            lines.append(f"- notes: {operator['notes']}")
        if operator.get("writeback_guidance"):
            lines.append(f"- writeback guidance: {operator['writeback_guidance']}")
        if operator.get("runtime_probe_skipped"):
            lines.append("- runtime probe: skipped; summary is doc-derived")

        issue_records = []
        for dtype_record in operator["dtype_records"]:
            for case_record in (*dtype_record["forward_cases"], *dtype_record["backward_cases"]):
                if case_record["status"] == "sample_or_function_issue":
                    issue_records.append(case_record)
        if issue_records:
            lines.append("- issue samples:")
            for issue in issue_records[:10]:
                lines.append(
                    f"  - {issue['direction']} / {issue['dtype']} / "
                    f"{issue.get('sample_name', 'unknown')}: {issue.get('error_message', '')}"
                )
            if len(issue_records) > 10:
                lines.append(f"  - ... {len(issue_records) - 10} more issue records omitted")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_script_module(module_name: str, script_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_path(raw_path: str | Path, *, base_dir: Path) -> Path:
    raw = Path(raw_path)
    return raw if raw.is_absolute() else (base_dir / raw).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch dtype probes for one or more operators.")
    parser.add_argument("--driver", action="append", default=[], help="Path to an operator probe driver script.")
    parser.add_argument("--summary-out", type=Path, default=None, help="Path to the JSON summary output.")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Path to the Markdown summary output.")
    parser.add_argument("--device-target", choices=["Ascend", "CPU", "GPU"], default="Ascend")
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def _load_specs_from_driver(driver_path: Path) -> list[OperatorProbeSpec]:
    module = load_script_module(f"dtype_probe_driver_{driver_path.stem}", driver_path)
    if hasattr(module, "build_operator_probes"):
        builder = getattr(module, "build_operator_probes")
        try:
            raw_specs = builder(sys.modules[__name__])
        except TypeError:
            raw_specs = builder()
    elif hasattr(module, "PROBE_OPERATORS"):
        raw_specs = getattr(module, "PROBE_OPERATORS")
    else:
        raise RuntimeError(
            f"{driver_path} must expose either PROBE_OPERATORS or build_operator_probes()."
        )

    return [coerce_probe_spec(spec) for spec in raw_specs]


def run_batch(
    *,
    driver_paths: Sequence[Path],
    summary_out: Path,
    markdown_out: Path | None,
    device_target: str,
    device_id: int,
) -> dict[str, Any]:
    runtime = prepare_runtime(device_target=device_target, device_id=device_id)

    operators: list[dict[str, Any]] = []
    for driver_path in driver_paths:
        for spec in _load_specs_from_driver(driver_path):
            operator_summary = execute_operator_probe(spec)
            operator_summary["driver_path"] = str(driver_path)
            operators.append(operator_summary)

    payload = {
        "check": "op_dtype_probe",
        "runtime": runtime,
        "drivers": [str(path) for path in driver_paths],
        "operators": operators,
    }
    write_json(summary_out, payload)
    if markdown_out is not None:
        write_text(markdown_out, render_markdown(payload))
    return payload


def main() -> int:
    args = parse_args()

    driver_paths = [_resolve_path(path, base_dir=Path.cwd()) for path in args.driver]
    summary_out = args.summary_out
    markdown_out = args.markdown_out

    if not driver_paths:
        raise ValueError("Provide at least one --driver.")

    if summary_out is None:
        summary_out = Path.cwd() / DEFAULT_SUMMARY_NAME
    if markdown_out is None:
        markdown_out = default_markdown_path(summary_out)

    run_batch(
        driver_paths=driver_paths,
        summary_out=summary_out,
        markdown_out=markdown_out,
        device_target=args.device_target,
        device_id=args.device_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
