#!/usr/bin/env python3
"""Compare saved MindSpore/PTA outputs with semantic and strict options.

This comparator provides a generic MS/PTA consistency comparison capability:

1. load one or more cases from a structured case_spec JSON file
2. use `allclose_nparray` from the sparse-lightning reference style as the
   semantic comparison signal for every output
3. default to strict raw-byte parity when strategy is omitted in case_spec
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STRATEGY_SEMANTIC_ZERO = "semantic_zero"
STRATEGY_BITWISE_STRICT = "bitwise_strict"
SUPPORTED_STRATEGIES = {STRATEGY_SEMANTIC_ZERO, STRATEGY_BITWISE_STRICT}
ERROR_TYPE_COMPARISON = "comparison"
ERROR_TYPE_INFRA = "infra"
FAILURE_REASON_SEMANTIC_MISMATCH = "semantic_mismatch"
FAILURE_REASON_RAW_BYTES_MISMATCH = "raw_bytes_mismatch"


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "numpy is required to compare .npy outputs; install numpy in the comparison environment"
        ) from exc
    return np


def _count_unequal_element(data_expected: Any, data_me: Any, rtol: float, atol: float) -> None:
    np = _require_numpy()
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, (
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
            data_expected[greater], data_me[greater], error[greater]
        )
    )


def allclose_nparray(
    data_expected: Any,
    data_me: Any,
    rtol: float,
    atol: float,
    equal_nan: bool = True,
) -> None:
    np = _require_numpy()
    if np.any(np.isnan(data_expected)) or np.any(np.isnan(data_me)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


def md5_bytes(raw: bytes) -> str:
    hasher = hashlib.md5()
    hasher.update(raw)
    return hasher.hexdigest()


def md5_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    return message or repr(exc)


def normalize_output_names(raw_outputs: Any) -> list[str]:
    if isinstance(raw_outputs, str):
        outputs = [item.strip() for item in raw_outputs.split(",") if item.strip()]
    elif isinstance(raw_outputs, list):
        outputs = [str(item).strip() for item in raw_outputs if str(item).strip()]
    else:
        raise TypeError(f"Unsupported outputs value: {type(raw_outputs)}")
    if not outputs:
        raise ValueError("No output names were provided")
    return outputs


def resolve_path(path_value: Any, *, base_dir: Path | None = None) -> Path:
    path = Path(path_value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def special_value_stats(array: Any) -> dict[str, int]:
    np = _require_numpy()
    normalized = np.asanyarray(array)
    if not np.issubdtype(normalized.dtype, np.inexact):
        return {"nan_count": 0, "posinf_count": 0, "neginf_count": 0}
    return {
        "nan_count": int(np.isnan(normalized).sum()),
        "posinf_count": int(np.isposinf(normalized).sum()),
        "neginf_count": int(np.isneginf(normalized).sum()),
    }


def load_case_spec(case_spec_path: Path) -> dict[str, Any]:
    case_spec_data = json.loads(case_spec_path.read_text(encoding="utf-8"))
    raw_cases = case_spec_data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case_spec must contain a non-empty `cases` list")

    base_dir = case_spec_path.parent
    cases = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise TypeError(f"case_spec entry at index {index} is not an object")
        strategy = raw_case.get("strategy", STRATEGY_BITWISE_STRICT)
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f"Unsupported strategy for case {raw_case!r}: {strategy}")
        cases.append(
            {
                "case_id": raw_case.get("case_id") or raw_case.get("name") or f"case_{index}",
                "ms_dir": resolve_path(raw_case["ms_dir"], base_dir=base_dir),
                "pta_dir": resolve_path(raw_case["pta_dir"], base_dir=base_dir),
                "outputs": normalize_output_names(
                    raw_case.get("outputs") or raw_case.get("output_names")
                ),
                "rtol": float(raw_case.get("rtol", 0.0)),
                "atol": float(raw_case.get("atol", 0.0)),
                "equal_nan": bool(raw_case.get("equal_nan", True)),
                "strategy": strategy,
                "summary_out": resolve_path(raw_case["summary_out"], base_dir=base_dir)
                if raw_case.get("summary_out")
                else None,
            }
        )

    return {
        "case_spec_path": case_spec_path,
        "summary_out": resolve_path(case_spec_data["summary_out"], base_dir=base_dir)
        if case_spec_data.get("summary_out")
        else None,
        "cases": cases,
    }


def compare_output_pair(
    *,
    ms_path: Path,
    pta_path: Path,
    output_name: str,
    rtol: float,
    atol: float,
    equal_nan: bool = True,
) -> dict[str, Any]:
    np = _require_numpy()
    ms_arr = np.load(ms_path, allow_pickle=False)
    pta_arr = np.load(pta_path, allow_pickle=False)
    ms_contiguous = np.ascontiguousarray(ms_arr)
    pta_contiguous = np.ascontiguousarray(pta_arr)

    ms_raw_md5 = md5_bytes(ms_contiguous.tobytes())
    pta_raw_md5 = md5_bytes(pta_contiguous.tobytes())
    ms_npy_md5 = md5_file(ms_path)
    pta_npy_md5 = md5_file(pta_path)
    raw_bytes_equal = ms_raw_md5 == pta_raw_md5
    npy_md5_equal = ms_npy_md5 == pta_npy_md5
    semantic_equal = True
    comparison_error = None
    try:
        allclose_nparray(pta_arr, ms_arr, rtol, atol, equal_nan=equal_nan)
    except (AssertionError, ValueError) as exc:
        semantic_equal = False
        comparison_error = format_exception(exc)

    payload = {
        "name": output_name,
        "equal": semantic_equal,
        "semantic_equal": semantic_equal,
        "shape": list(ms_arr.shape),
        "pta_shape": list(pta_arr.shape),
        "dtype": str(ms_arr.dtype),
        "pta_dtype": str(pta_arr.dtype),
        "rtol": rtol,
        "atol": atol,
        "equal_nan": equal_nan,
        "ms_path": str(ms_path),
        "pta_path": str(pta_path),
        "ms_md5": ms_npy_md5,
        "pta_md5": pta_npy_md5,
        "binary_equal": npy_md5_equal,
        "ms_raw_md5": ms_raw_md5,
        "pta_raw_md5": pta_raw_md5,
        "raw_bytes_equal": raw_bytes_equal,
        "special_values": special_value_stats(ms_arr),
        "pta_special_values": special_value_stats(pta_arr),
    }
    if comparison_error is not None:
        payload["comparison_error"] = comparison_error
    return payload


def compare_saved_outputs(
    *,
    ms_dir: Path,
    pta_dir: Path,
    output_names: list[str],
    rtol: float = 0.0,
    atol: float = 0.0,
    equal_nan: bool = True,
) -> dict[str, Any]:
    outputs: list[dict[str, Any]] = []
    for output_name in output_names:
        ms_path = ms_dir / f"{output_name}.npy"
        pta_path = pta_dir / f"{output_name}.npy"
        if not ms_path.exists():
            raise FileNotFoundError(f"Missing MS output file: {ms_path}")
        if not pta_path.exists():
            raise FileNotFoundError(f"Missing PTA output file: {pta_path}")
        outputs.append(
            compare_output_pair(
                ms_path=ms_path,
                pta_path=pta_path,
                output_name=output_name,
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            )
        )

    return {
        "check": "ms_pta_consistency",
        "all_equal": all(item["equal"] for item in outputs),
        "all_binary_equal": all(item["binary_equal"] for item in outputs),
        "all_raw_bytes_equal": all(item["raw_bytes_equal"] for item in outputs),
        "outputs": outputs,
    }


def evaluate_case_success(payload: dict[str, Any]) -> bool:
    strategy = payload["strategy"]
    if strategy == STRATEGY_SEMANTIC_ZERO:
        return bool(payload["all_equal"])
    if strategy == STRATEGY_BITWISE_STRICT:
        return bool(payload["all_equal"] and payload["all_raw_bytes_equal"])
    raise ValueError(f"Unsupported strategy: {strategy}")


def determine_failure_reason(payload: dict[str, Any]) -> str | None:
    if not payload["all_equal"]:
        return FAILURE_REASON_SEMANTIC_MISMATCH
    if payload["strategy"] == STRATEGY_BITWISE_STRICT and not payload["all_raw_bytes_equal"]:
        return FAILURE_REASON_RAW_BYTES_MISMATCH
    return None


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compare_case_spec(case_spec: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = compare_saved_outputs(
            ms_dir=case_spec["ms_dir"],
            pta_dir=case_spec["pta_dir"],
            output_names=case_spec["outputs"],
            rtol=case_spec["rtol"],
            atol=case_spec["atol"],
            equal_nan=case_spec["equal_nan"],
        )
        payload.update(
            {
                "case_id": case_spec["case_id"],
                "strategy": case_spec["strategy"],
                "passed": False,  # filled below
                "error_type": None,
            }
        )
        payload["passed"] = evaluate_case_success(payload)
        if not payload["passed"]:
            payload["error_type"] = ERROR_TYPE_COMPARISON
            payload["failure_reason"] = determine_failure_reason(payload)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "check": "ms_pta_consistency",
            "case_id": case_spec["case_id"],
            "strategy": case_spec["strategy"],
            "passed": False,
            "error_type": ERROR_TYPE_INFRA,
            "all_equal": False,
            "all_binary_equal": False,
            "all_raw_bytes_equal": False,
            "error": format_exception(exc),
            "outputs": [],
        }

    if case_spec.get("summary_out") is not None:
        write_summary_json(case_spec["summary_out"], payload)
    return payload


def compare_case_spec_file_payload(case_spec_payload: dict[str, Any]) -> dict[str, Any]:
    case_results = [compare_case_spec(case_spec) for case_spec in case_spec_payload["cases"]]
    payload = {
        "check": "ms_pta_consistency_case_spec",
        "case_spec_path": str(case_spec_payload["case_spec_path"]),
        "all_passed": all(case["passed"] for case in case_results),
        "case_count": len(case_results),
        "cases": case_results,
    }
    if case_spec_payload.get("summary_out") is not None:
        write_summary_json(case_spec_payload["summary_out"], payload)
    return payload


def compare_case_spec_file(case_spec_path: Path) -> dict[str, Any]:
    return compare_case_spec_file_payload(load_case_spec(case_spec_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MS and PTA .npy outputs from a structured case_spec JSON file"
    )
    parser.add_argument(
        "--case_spec",
        type=Path,
        required=True,
        help="JSON file describing one or more consistency-validation cases",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = compare_case_spec_file(args.case_spec)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
