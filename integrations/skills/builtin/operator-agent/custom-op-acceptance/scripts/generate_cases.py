#!/usr/bin/env python3
"""Generate checklist-driven acceptance coverage and profiler cases.

Case generation reads the operator signature to produce a matrix covering:
- dtypes: float16, float32 (minimum), plus int32/float64 where the signature allows
- shapes: operator-relevant boundary/model/invalid shapes, not mechanical 1D..8D rank enumeration
- special values: empty tensor, Inf/NaN
- attribute combinations (up to 3 variations when attributes exist)
- error cases: wrong rank and invalid attribute values
- backward: one basic grad case when the operator declares backward support

Operator-specific overrides (attention, MHC, interleave_rope) use valid
domain-specific cases instead of blindly applying every generic shape.
"""
from __future__ import annotations

import argparse
import ast
import itertools
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import case_ids_from_value, read_json, render_template, sanitize_identifier  # noqa: E402

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _required_tensor_names(meta: dict[str, Any]) -> list[str]:
    return [item["name"] for item in meta.get("inputs", []) if str(item.get("dtype", "")).startswith("Tensor")]


def _optional_artifact(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    return read_json(artifact_path)


def _artifact_input_summary(
    path: str | Path | None,
    artifact: dict[str, Any] | None,
    operator_name: str,
) -> dict[str, Any]:
    if artifact is None:
        return {"provided": False}
    actual_name = str(artifact.get("operator_name", ""))
    return {
        "provided": True,
        "path": str(Path(path)) if path is not None else "",
        "analysis_status": str(artifact.get("analysis_status", "")),
        "operator_name": actual_name,
        "operator_name_matches": actual_name == operator_name,
    }


def _project_root_from_output_root(output_root: Path) -> Path:
    if output_root.name == "acceptance" and output_root.parent.name == "tests":
        return output_root.parent.parent
    return output_root


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _opinfo_name_candidates(meta: dict[str, Any]) -> list[str]:
    values = []
    ms_api = str(meta.get("ms_api", ""))
    if ms_api:
        values.append(ms_api.rsplit(".", 1)[-1])
    values.append(str(meta.get("operator_name", "")))

    candidates: list[str] = []
    for value in values:
        if not value:
            continue
        variants = [value]
        if value.endswith("_op"):
            variants.append(value.removesuffix("_op"))
        if value.startswith("npu_"):
            variants.append(value.removeprefix("npu_"))
        for variant in variants:
            if variant and variant not in candidates:
                candidates.append(variant)
    return candidates


def _literal_list_assignment(tree: ast.AST, name: str) -> list[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            return []
        values = []
        for item in node.value.elts:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                values.append(item.value)
        return values
    return []


def _literal_dict_keys(tree: ast.AST, name: str) -> list[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            return []
        keys = []
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
        return keys
    return []


def _ms_adapter_opinfo_status(project_root: Path, meta: dict[str, Any]) -> dict[str, Any]:
    tests_root = project_root / "tests"
    opinfo_root = tests_root / "ms_adapter" / "opinfo"
    required = {
        "shared_runner": tests_root / "shared" / "opinfo" / "runners.py",
        "shared_op_info": tests_root / "shared" / "_op_info" / "op_info.py",
        "database": opinfo_root / "op_database.py",
        "sample_inputs": opinfo_root / "op_sample_inputs.py",
        "wrappers": opinfo_root / "op_wrappers.py",
        "unary_test": opinfo_root / "test_unary_ops.py",
        "binary_test": opinfo_root / "test_binary_ops.py",
        "other_test": opinfo_root / "test_other_ops.py",
    }
    missing = {key: _rel(project_root, path) for key, path in required.items() if not path.exists()}
    candidates = _opinfo_name_candidates(meta)
    status: dict[str, Any] = {
        "framework": "ms_adapter_opinfo",
        "source_commit": "5e4217c",
        "opinfo_name_candidates": candidates,
        "required_files": {key: _rel(project_root, path) for key, path in required.items()},
        "missing_files": missing,
    }
    if missing:
        status.update({
            "status": "missing_framework",
            "registered": False,
            "reason": "ms_adapter opinfo framework files are missing",
        })
        return status

    database = required["database"]
    try:
        tree = ast.parse(database.read_text(encoding="utf-8"), filename=str(database))
    except SyntaxError as exc:
        status.update({
            "status": "invalid_database",
            "registered": False,
            "reason": f"op_database.py is not parseable: {exc}",
        })
        return status

    op_db = set(_literal_dict_keys(tree, "op_db"))
    category_lists = {
        "unary": set(_literal_list_assignment(tree, "unary_op_db")),
        "binary": set(_literal_list_assignment(tree, "binary_op_db")),
        "other": set(_literal_list_assignment(tree, "other_op_db")),
    }
    matched = next((name for name in candidates if name in op_db), "")
    status["registered_names"] = sorted(op_db)
    if not matched:
        scaffold_name = candidates[0] if candidates else str(meta.get("operator_name", ""))
        status.update({
            "status": "not_registered",
            "registered": False,
            "opinfo_name": scaffold_name,
            "reason": "operator is not registered in tests/ms_adapter/opinfo/op_database.py",
            "scaffold_command": (
                "python codegen/ms_adapter/gen_tests.py "
                f"--ms-adapter-opinfo-scaffold-function {scaffold_name} "
                f"--output /tmp/ms_adapter_opinfo_{sanitize_identifier(scaffold_name)}"
            ),
        })
        return status

    category = next((name for name, values in category_lists.items() if matched in values), "")
    test_path = required.get(f"{category}_test") if category else None
    if not category or test_path is None:
        status.update({
            "status": "registered_without_category",
            "registered": True,
            "opinfo_name": matched,
            "reason": "operator is in op_db but not listed in unary_op_db/binary_op_db/other_op_db",
        })
        return status

    status.update({
        "status": "ready",
        "registered": True,
        "opinfo_name": matched,
        "category": category,
        "test_path": _rel(project_root, test_path),
        "pytest_k": matched,
        "reason": "operator is registered in ms_adapter opinfo framework",
    })
    return status


def _plan_alignment(case_plan: dict[str, Any] | None, cases: list[dict[str, Any]]) -> dict[str, Any]:
    generated_case_ids = {str(case.get("case_id", "")) for case in cases if str(case.get("case_id", "")).strip()}
    if not case_plan:
        return {
            "case_plan_provided": False,
            "generated_case_ids": sorted(generated_case_ids),
            "case_plan_case_ids": [],
            "missing_generated_case_ids": [],
            "extra_generated_case_ids": [],
    }
    planned_case_ids: set[str] = set()
    for key in ("functional_cases", "negative_cases", "checklist_mapping"):
        planned_case_ids.update(case_ids_from_value(case_plan.get(key, [])))
    shape_design = case_plan.get("shape_design", {})
    if isinstance(shape_design, dict):
        planned_case_ids.update(case_ids_from_value(shape_design.get("case_mapping", {})))
    return {
        "case_plan_provided": True,
        "generated_case_ids": sorted(generated_case_ids),
        "case_plan_case_ids": sorted(planned_case_ids),
        "missing_generated_case_ids": sorted(planned_case_ids - generated_case_ids),
        "extra_generated_case_ids": sorted(generated_case_ids - planned_case_ids),
    }


def _unique_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in result:
            result.append(text)
    return result


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique_strings(value)
    if isinstance(value, str):
        return _unique_strings([value])
    return []


def _required_dtypes(
    operator_contract: dict[str, Any] | None,
    case_plan: dict[str, Any] | None,
) -> list[str]:
    values: list[Any] = []
    if isinstance(case_plan, dict):
        dtype_plan = case_plan.get("dtype_plan", {})
        if isinstance(dtype_plan, dict):
            values.extend(_string_list(dtype_plan.get("pta_supported", [])))
            values.extend(_string_list(dtype_plan.get("ms_expected", [])))
    if isinstance(operator_contract, dict):
        dtype_rule = operator_contract.get("dtype_rule", {})
        if isinstance(dtype_rule, dict):
            values.extend(_string_list(dtype_rule.get("pta_supported", [])))
            values.extend(_string_list(dtype_rule.get("ms_expected", [])))
    return _unique_strings(values)


def _case_declared_dtypes(case: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    if case.get("dtype"):
        values.append(case["dtype"])
    tensor_dtypes = case.get("tensor_dtypes", {})
    if isinstance(tensor_dtypes, dict):
        values.extend(tensor_dtypes.values())
    return _unique_strings(values)


def _functional_dtypes(cases: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for case in cases:
        if case.get("expect_error"):
            continue
        values.extend(_case_declared_dtypes(case))
    return sorted(set(values))


def _dtype_case_prototype(cases: list[dict[str, Any]]) -> dict[str, Any]:
    for case in cases:
        if not case.get("expect_error") and case.get("shape") == [2, 3]:
            return case
    for case in cases:
        if not case.get("expect_error"):
            return case
    return {}


def _dtype_case_id(dtype: str, prototype: dict[str, Any]) -> str:
    suffix = sanitize_identifier(dtype)
    shape = prototype.get("shape")
    if isinstance(shape, list) and len(shape) == 2:
        return f"dtype_{suffix}_2d"
    return f"dtype_{suffix}_functional"


def _add_required_dtype_cases(
    cases: list[dict[str, Any]],
    meta: dict[str, Any],
    required_dtypes: list[str],
) -> list[dict[str, Any]]:
    covered = set(_functional_dtypes(cases))
    prototype = _dtype_case_prototype(cases)
    tensor_names = prototype.get("tensor_inputs") or _required_tensor_names(meta)
    for dtype in required_dtypes:
        if dtype in covered:
            continue
        case: dict[str, Any] = {
            "case_id": _dtype_case_id(dtype, prototype),
            "dtype": dtype,
            "tensor_inputs": tensor_names,
            "attributes": prototype.get("attributes", _with_attrs(meta, {})),
            "checklist_tags": [f"dtype_{dtype}", "dtype_coverage"],
            "expect_error": False,
        }
        if "input_shapes" in prototype:
            case["input_shapes"] = prototype["input_shapes"]
        else:
            case["shape"] = prototype.get("shape", [2, 3]) or [2, 3]
        cases.append(case)
        covered.add(dtype)
    return cases



def _default_attr_value(attr: dict[str, Any]) -> Any:
    if attr.get("default") is not None:
        return attr["default"]
    dtype = attr.get("dtype", "")
    if dtype.endswith("?"):
        return None
    if dtype == "bool":
        return False
    if dtype in {"float", "double"}:
        return 1.0
    if dtype in {"IntArrayRef", "ShapeVector"}:
        return [0]
    if dtype == "str":
        return "default"
    return 1


def _with_attrs(meta: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    values = {item["name"]: _default_attr_value(item) for item in meta.get("attributes", [])}
    values.update(updates)
    return values


def _alt_attr_value(attr: dict[str, Any]) -> Any:
    """Return a non-default value for a single attribute when possible."""
    dtype = attr.get("dtype", "")
    if dtype == "bool":
        return True
    if dtype in {"float", "double"}:
        return 2.5
    if dtype in {"IntArrayRef", "ShapeVector"}:
        return [2, 4]
    if dtype == "str":
        return "alt"
    if dtype == "int":
        return 3
    return 2


def _attr_combinations(meta: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    """Produce up to *limit* non-default attribute combinations."""
    attrs = meta.get("attributes", [])
    if not attrs:
        return []
    combos: list[dict[str, Any]] = []
    # single-attribute variations first
    for attr in attrs:
        if len(combos) >= limit:
            break
        combo = _with_attrs(meta, {attr["name"]: _alt_attr_value(attr)})
        if combo != _with_attrs(meta, {}):
            combos.append(combo)
    # pairwise variations if budget remains
    for a1, a2 in itertools.combinations(attrs, 2):
        if len(combos) >= limit:
            break
        combo = _with_attrs(meta, {a1["name"]: _alt_attr_value(a1), a2["name"]: _alt_attr_value(a2)})
        if combo not in combos:
            combos.append(combo)
    return combos[:limit]


# ---------------------------------------------------------------------------
# generic case matrix
# ---------------------------------------------------------------------------

def _generic_cases(meta: dict[str, Any]) -> list[dict[str, Any]]:
    tensor_names = _required_tensor_names(meta)
    has_backward = meta.get("has_backward", False)
    multi_input = len(tensor_names) >= 2

    cases: list[dict[str, Any]] = []

    # -- shape coverage: rule-driven boundary/model shapes ------------------
    shape_specs = [
        ([], "0d", ["shape_0d", "scalar_input", "boundary_shape"]),
        ([1], "single_element", ["single_element", "boundary_shape"]),
        ([3, 5], "non_aligned_2d", ["shape_2d", "non_aligned_shape", "boundary_shape"]),
        ([16, 4096], "model_batch_hidden", ["shape_2d", "model_shape"]),
        ([2, 128, 4096], "model_batch_seq_hidden", ["shape_3d", "model_shape"]),
    ]
    for shape, tag, extra_tags in shape_specs:
        cases.append({
            "case_id": f"shape_{tag}_fp16",
            "shape": shape,
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "attributes": _with_attrs(meta, {}),
            "checklist_tags": [*extra_tags, "dtype_float16", "shape_coverage"],
            "expect_error": False,
        })

    # -- dtype coverage ----------------------------------------------------
    # Default to float32 only. Integer dtypes are only added when the
    # operator declares integer-typed attributes, which is a strong signal
    # that the operator handles non-float inputs.
    dtype_specs: list[tuple[str, str]] = [("fp32", "float32")]
    if any(
        attr.get("dtype", "").lower().startswith(("int", "intarrayref"))
        for attr in meta.get("attributes", [])
    ):
        dtype_specs.extend([("int32", "int32"), ("int64", "int64")])
    for tag, dtype in dtype_specs:
        cases.append({
            "case_id": f"dtype_{tag}_2d",
            "shape": [2, 3],
            "dtype": dtype,
            "tensor_inputs": tensor_names,
            "attributes": _with_attrs(meta, {}),
            "checklist_tags": [f"dtype_{dtype}", "dtype_coverage"],
            "expect_error": False,
        })

    # mark the fp32 default-parameters case as the primary smoke
    fp32_entry = next(c for c in cases if c["case_id"] == "dtype_fp32_2d")
    fp32_entry["checklist_tags"].extend(["default_parameters", "forward_smoke"])

    # -- large tensor -------------------------------------------------------
    cases.append({
        "case_id": "large_fp16_2d",
        "shape": [128, 256],
        "dtype": "float16",
        "tensor_inputs": tensor_names,
        "attributes": _with_attrs(meta, {}),
        "checklist_tags": ["large_tensor", "dtype_float16", "shape_range"],
        "expect_error": False,
    })

    # -- empty tensor -------------------------------------------------------
    cases.append({
        "case_id": "empty_fp32",
        "shape": [0],
        "dtype": "float32",
        "tensor_inputs": tensor_names,
        "attributes": _with_attrs(meta, {}),
        "checklist_tags": ["empty_tensor_input"],
        "expect_error": False,
    })

    # -- Inf / NaN ----------------------------------------------------------
    cases.append({
        "case_id": "special_values_fp32",
        "shape": [8],
        "dtype": "float32",
        "tensor_inputs": tensor_names,
        "attributes": _with_attrs(meta, {}),
        "values": [0.0, "inf", "-inf", "nan"],
        "checklist_tags": ["inf_and_nan"],
        "expect_error": False,
    })

    # -- finite value range -------------------------------------------------
    cases.append({
        "case_id": "value_range_fp32",
        "shape": [8],
        "dtype": "float32",
        "tensor_inputs": tensor_names,
        "attributes": _with_attrs(meta, {}),
        "values": [-10.0, -1.0, -0.001, 0.0, 0.001, 1.0, 10.0],
        "checklist_tags": ["value_range"],
        "expect_error": False,
    })

    # -- attribute combinations ---------------------------------------------
    for idx, combo in enumerate(_attr_combinations(meta, limit=3)):
        cases.append({
            "case_id": f"attr_combo_{idx + 1}_fp16",
            "shape": [2, 3],
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "attributes": combo,
            "checklist_tags": ["attribute_combination", "dtype_float16"],
            "expect_error": False,
        })

    # -- broadcast (multi-input only) ---------------------------------------
    if multi_input:
        broadcast_shapes = {
            tensor_names[0]: [3, 1, 4],
            tensor_names[1]: [1, 5, 4],
        }
        for rest in tensor_names[2:]:
            broadcast_shapes[rest] = [3, 5, 4]
        cases.append({
            "case_id": "broadcast_fp16",
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "input_shapes": broadcast_shapes,
            "attributes": _with_attrs(meta, {}),
            "checklist_tags": ["broadcast", "dtype_float16", "input_constraints"],
            "expect_error": False,
        })

        # -- mixed dtype (multi-input only) -----------------------------------
        mixed_dtypes = {}
        for i, name in enumerate(tensor_names):
            mixed_dtypes[name] = ["float16", "float32", "int32"][i % 3]
        cases.append({
            "case_id": "mixed_dtype_fp16_fp32_int32",
            "shape": [2, 3],
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "tensor_dtypes": mixed_dtypes,
            "attributes": _with_attrs(meta, {}),
            "checklist_tags": ["mixed_dtype", "dtype_coverage", "input_constraints"],
            "expect_error": False,
        })

    # -- error: mismatched rank ---------------------------------------------
    if multi_input and len(tensor_names) >= 2:
        error_shapes = {tensor_names[0]: [2, 3], tensor_names[1]: [2, 3, 4]}
        for rest in tensor_names[2:]:
            error_shapes[rest] = [2, 3]
        cases.append({
            "case_id": "error_rank_mismatch",
            "dtype": "float32",
            "tensor_inputs": tensor_names,
            "input_shapes": error_shapes,
            "attributes": _with_attrs(meta, {}),
            "checklist_tags": ["error_cases", "input_constraints"],
            "expect_error": True,
        })

    # -- error: invalid attribute (when applicable) -------------------------
    for attr in meta.get("attributes", []):
        if attr.get("dtype") == "str" and attr.get("default") is not None:
            cases.append({
                "case_id": "error_invalid_attr_str",
                "shape": [2, 3],
                "dtype": "float32",
                "tensor_inputs": tensor_names,
                "attributes": _with_attrs(meta, {attr["name"]: "__invalid_value__"}),
                "checklist_tags": ["error_cases", "attribute_validation"],
                "expect_error": True,
            })
            break

    # -- backward (if supported) --------------------------------------------
    if has_backward:
        cases.append({
            "case_id": "backward_fp32_1d",
            "shape": [4],
            "dtype": "float32",
            "tensor_inputs": tensor_names,
            "attributes": _with_attrs(meta, {}),
            "checklist_tags": ["backward", "dtype_float32"],
            "expect_error": False,
            "require_grad": True,
        })

    return cases


# ---------------------------------------------------------------------------
# operator-specific augmentations
# ---------------------------------------------------------------------------

def _fused_infer_attention_cases(meta: dict[str, Any]) -> list[dict[str, Any]]:
    tensor_names = ["query", "key", "value"]
    specific = [
        {
            "case_id": "bsh_small_fp16_lse",
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "input_shapes": {"query": [1, 2, 64], "key": [1, 2, 64], "value": [1, 2, 64]},
            "attributes": _with_attrs(
                meta,
                {"num_heads": 4, "scale": 0.125, "input_layout": "BSH", "softmax_lse_flag": True},
            ),
            "checklist_tags": ["default_parameters", "layout_bsh", "torch_npu_alignment", "forward_smoke"],
            "expect_error": False,
        },
        {
            "case_id": "bsnd_small_fp16_lse",
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "input_shapes": {"query": [2, 4, 4, 16], "key": [2, 4, 4, 16], "value": [2, 4, 4, 16]},
            "attributes": _with_attrs(
                meta,
                {"num_heads": 4, "scale": 0.25, "input_layout": "BSND", "softmax_lse_flag": True},
            ),
            "checklist_tags": ["layout_bsnd", "rank_shape_variation", "torch_npu_alignment"],
            "expect_error": False,
        },
        {
            "case_id": "bsh_no_lse_fp16",
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "input_shapes": {"query": [1, 4, 64], "key": [1, 4, 64], "value": [1, 4, 64]},
            "attributes": _with_attrs(
                meta,
                {"num_heads": 4, "scale": 0.125, "input_layout": "BSH", "softmax_lse_flag": False},
            ),
            "checklist_tags": ["attribute_combination", "return_softmax_lse_false"],
            "expect_error": False,
        },
        {
            "case_id": "tnd_gqa_fp16",
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "input_shapes": {"query": [6, 8, 32], "key": [12, 2, 32], "value": [12, 2, 32]},
            "attributes": _with_attrs(
                meta,
                {
                    "actual_seq_lengths": [1, 3, 6],
                    "actual_seq_lengths_kv": [3, 7, 12],
                    "num_heads": 8,
                    "num_key_value_heads": 2,
                    "scale": 0.17677669529663687,
                    "input_layout": "TND",
                    "softmax_lse_flag": True,
                },
            ),
            "checklist_tags": ["layout_tnd", "gqa", "dynamic_sequence_lengths", "torch_npu_alignment"],
            "expect_error": False,
        },
        # empty-tensor coverage for attention
        {
            "case_id": "bsh_empty_fp16",
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "input_shapes": {"query": [0, 4, 64], "key": [0, 4, 64], "value": [0, 4, 64]},
            "attributes": _with_attrs(meta, {"num_heads": 4, "scale": 0.125, "input_layout": "BSH"}),
            "checklist_tags": ["empty_tensor_input", "layout_bsh"],
            "expect_error": False,
        },
    ]
    return specific


def _mhc_pre_cases(meta: dict[str, Any]) -> list[dict[str, Any]]:
    tensor_names = ["x", "phi", "alpha", "bias"]
    specific = [
        {
            "case_id": "ascend950_basic_fp16",
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "input_shapes": {"x": [2, 16], "phi": [16], "alpha": [16], "bias": [16]},
            "attributes": _with_attrs(meta, {"out_flag": 0}),
            "checklist_tags": ["ascend950_required", "default_parameters", "forward_smoke"],
            "expect_error": False,
        },
        {
            "case_id": "ascend950_with_gamma_fp16",
            "dtype": "float16",
            "tensor_inputs": tensor_names + ["gamma"],
            "input_shapes": {
                "x": [2, 16], "phi": [16], "alpha": [16], "bias": [16], "gamma": [16],
            },
            "attributes": _with_attrs(meta, {"out_flag": 1}),
            "checklist_tags": ["ascend950_required", "optional_gamma", "attribute_combination"],
            "expect_error": False,
        },
        # empty coverage
        {
            "case_id": "ascend950_empty_fp16",
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "input_shapes": {"x": [0, 16], "phi": [16], "alpha": [16], "bias": [16]},
            "attributes": _with_attrs(meta, {"out_flag": 0}),
            "checklist_tags": ["ascend950_required", "empty_tensor_input"],
            "expect_error": False,
        },
    ]
    return specific


def _interleave_rope_cases(meta: dict[str, Any]) -> list[dict[str, Any]]:
    tensor_names = ["x", "cos", "sin"]
    specific = [
        {
            "case_id": "basic_fp16_4d",
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "input_shapes": {"x": [2, 4, 8, 64], "cos": [2, 1, 8, 64], "sin": [2, 1, 8, 64]},
            "attributes": _with_attrs(meta, {}),
            "checklist_tags": ["valid_shape", "dtype_float16", "forward_smoke"],
            "expect_error": False,
        },
        {
            "case_id": "large_fp16_4d",
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "input_shapes": {"x": [2, 16, 128, 64], "cos": [2, 1, 128, 64], "sin": [2, 1, 128, 64]},
            "attributes": _with_attrs(meta, {}),
            "checklist_tags": ["large_tensor", "dtype_float16", "forward_smoke"],
            "expect_error": False,
        },
        {
            "case_id": "fp32_4d",
            "dtype": "float32",
            "tensor_inputs": tensor_names,
            "input_shapes": {"x": [2, 4, 8, 64], "cos": [2, 1, 8, 64], "sin": [2, 1, 8, 64]},
            "attributes": _with_attrs(meta, {}),
            "checklist_tags": ["dtype_float32", "forward_smoke"],
            "expect_error": False,
        },
    ]
    return specific


def _fast_gelu_cases(meta: dict[str, Any]) -> list[dict[str, Any]]:
    tensor_names = ["self"]
    cases: list[dict[str, Any]] = []
    shape_specs = [
        ([], "0d", ["shape_0d", "scalar_input", "boundary_shape"]),
        ([1], "single_element", ["single_element", "boundary_shape"]),
        ([3, 5], "non_aligned_2d", ["shape_2d", "non_aligned_shape", "boundary_shape"]),
        ([16, 4096], "model_batch_hidden", ["shape_2d", "model_shape"]),
        ([2, 128, 4096], "model_batch_seq_hidden", ["shape_3d", "model_shape"]),
    ]
    for shape, tag, extra_tags in shape_specs:
        cases.append({
            "case_id": f"shape_{tag}_fp16",
            "shape": shape,
            "dtype": "float16",
            "tensor_inputs": tensor_names,
            "attributes": _with_attrs(meta, {}),
            "checklist_tags": [*extra_tags, "dtype_float16", "shape_coverage"],
            "expect_error": False,
        })
    cases.append({
        "case_id": "dtype_fp32_2d",
        "shape": [2, 3],
        "dtype": "float32",
        "tensor_inputs": tensor_names,
        "attributes": _with_attrs(meta, {}),
        "checklist_tags": ["dtype_float32", "dtype_coverage", "default_parameters", "forward_smoke"],
        "expect_error": False,
    })
    cases.append({
        "case_id": "large_fp16_2d",
        "shape": [128, 256],
        "dtype": "float16",
        "tensor_inputs": tensor_names,
        "attributes": _with_attrs(meta, {}),
        "checklist_tags": ["large_tensor", "dtype_float16", "shape_range"],
        "expect_error": False,
    })
    cases.append({
        "case_id": "empty_fp32",
        "shape": [0],
        "dtype": "float32",
        "tensor_inputs": tensor_names,
        "attributes": _with_attrs(meta, {}),
        "checklist_tags": ["empty_tensor_input"],
        "expect_error": False,
    })
    cases.append({
        "case_id": "special_values_fp32",
        "shape": [4],
        "dtype": "float32",
        "tensor_inputs": tensor_names,
        "attributes": _with_attrs(meta, {}),
        "values": [0.0, "inf", "-inf", "nan"],
        "checklist_tags": ["inf_and_nan"],
        "expect_error": False,
    })
    cases.append({
        "case_id": "value_range_fp32",
        "shape": [8],
        "dtype": "float32",
        "tensor_inputs": tensor_names,
        "attributes": _with_attrs(meta, {}),
        "values": [-10.0, -1.0, -0.001, 0.0, 0.001, 1.0, 10.0],
        "checklist_tags": ["value_range"],
        "expect_error": False,
    })
    cases.append({
        "case_id": "error_invalid_dtype_int32",
        "shape": [2, 3],
        "dtype": "int32",
        "tensor_inputs": tensor_names,
        "attributes": _with_attrs(meta, {}),
        "checklist_tags": ["error_cases", "dtype_coverage", "input_constraints"],
        "expect_error": True,
    })
    return cases


def _fast_gelu_backward_cases(meta: dict[str, Any]) -> list[dict[str, Any]]:
    tensor_names = ["grad", "self"]
    cases = _fast_gelu_cases(meta)
    for case in cases:
        case["tensor_inputs"] = tensor_names
        case["case_id"] = case["case_id"] + "_bwd"
    cases.append({
        "case_id": "error_rank_mismatch",
        "dtype": "float32",
        "tensor_inputs": tensor_names,
        "input_shapes": {"grad": [2, 3], "self": [2, 3, 4]},
        "attributes": _with_attrs(meta, {}),
        "checklist_tags": ["error_cases", "input_constraints"],
        "expect_error": True,
    })
    return cases


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def _dispatch_cases(meta: dict[str, Any]) -> list[dict[str, Any]]:
    operator_name = meta.get("operator_name", "")

    if operator_name == "npu_fast_gelu_op":
        return _fast_gelu_cases(meta)
    if operator_name == "npu_fast_gelu_backward_op":
        return _fast_gelu_backward_cases(meta)
    if operator_name == "npu_fused_infer_attention_score_op":
        return _fused_infer_attention_cases(meta)
    if operator_name == "npu_mhc_pre_op":
        return _mhc_pre_cases(meta)
    if operator_name == "npu_interleave_rope_op":
        return _interleave_rope_cases(meta)
    return _generic_cases(meta)


# ---------------------------------------------------------------------------
# coverage summary
# ---------------------------------------------------------------------------

def _shape_case_mapping(cases: list[dict[str, Any]]) -> dict[str, list[str]]:
    mapping = {
        "边界 shape": [],
        "真实模型 shape": [],
        "非法 shape / 报错 shape": [],
        "0D scalar": [],
        "empty tensor": [],
        "broadcast": [],
        "axis/dim": [],
    }
    for case in cases:
        case_id = str(case.get("case_id", ""))
        tags = {str(tag) for tag in case.get("checklist_tags", [])}
        if not case_id:
            continue
        if tags & {"boundary_shape", "single_element", "non_aligned_shape", "scalar_input"}:
            mapping["边界 shape"].append(case_id)
        if "model_shape" in tags:
            mapping["真实模型 shape"].append(case_id)
        if case.get("expect_error"):
            mapping["非法 shape / 报错 shape"].append(case_id)
        if tags & {"shape_0d", "scalar_input"}:
            mapping["0D scalar"].append(case_id)
        if "empty_tensor_input" in tags:
            mapping["empty tensor"].append(case_id)
        if "broadcast" in tags:
            mapping["broadcast"].append(case_id)
        if any(tag in tags for tag in {"axis_first", "axis_last", "axis_negative", "axis_invalid"}):
            mapping["axis/dim"].append(case_id)
    return mapping


def _shape_design(meta: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    operator_name = str(meta.get("operator_name", ""))
    tensor_names = _required_tensor_names(meta)
    tags = {str(tag) for case in cases for tag in case.get("checklist_tags", [])}
    has_broadcast = "broadcast" in tags
    has_empty = "empty_tensor_input" in tags
    has_scalar = "shape_0d" in tags or "scalar_input" in tags
    has_dynamic = bool({"model_shape", "shape_coverage"} & tags)
    dtype_tags = sorted(tag for tag in tags if tag.startswith("dtype_"))
    invalid_cases = [
        str(case.get("case_id", ""))
        for case in cases
        if case.get("expect_error")
    ]
    case_mapping = _shape_case_mapping(cases)

    if "fast_gelu" in operator_name:
        return {
            "analysis_status": "script_confirmed",
            "算子名称": operator_name,
            "输入 shape 规则": "单 Tensor 输入 self，任意 shape，逐元素计算",
            "输出 shape 规则": "输出 shape 与 self 完全一致",
            "关键维度含义": "所有维度均为元素维度，无 axis 语义",
            "需要覆盖的 axis / dim": "不涉及 axis/dim",
            "需要覆盖的边界 shape": "0D scalar、单元素、empty tensor、非对齐 2D、真实模型大 shape",
            "是否支持 broadcast": "不涉及，单 Tensor 输入",
            "是否支持 empty tensor": "是" if has_empty else "待补充",
            "是否支持 0D scalar": "是" if has_scalar else "待补充",
            "是否有 dtype 相关分支": ", ".join(dtype_tags) or "待确认",
            "是否有动态 shape / 动态 rank": "按同 shape 输出规则覆盖动态 shape/rank" if has_dynamic else "待确认",
            "真实模型中常见 shape": "[batch, hidden]、[batch, seq, hidden]",
            "非法 shape / 报错 shape": (
                "无非法 shape 约束；报错用例: " + ", ".join(invalid_cases)
                if invalid_cases else "无非法 shape 约束"
            ),
            "case_mapping": case_mapping,
        }

    return {
        "analysis_status": "draft_needs_agent_confirmation",
        "算子名称": operator_name,
        "输入 shape 规则": (
            "多 Tensor 输入按输入约束分别构造 shape" if len(tensor_names) >= 2 else "单 Tensor 输入按主输入 shape 构造"
        ),
        "输出 shape 规则": "由源码/文档/agent_evidence 确认后回填",
        "关键维度含义": "根据算子接口语义确认关键 batch/channel/seq/axis 维度",
        "需要覆盖的 axis / dim": "有 axis/dim 属性时覆盖首维、末维、负轴和非法轴",
        "需要覆盖的边界 shape": "0D scalar、单元素、empty tensor、非对齐 shape、真实模型 shape",
        "是否支持 broadcast": "是" if has_broadcast else "无 broadcast 证据或不涉及",
        "是否支持 empty tensor": "是" if has_empty else "待补充",
        "是否支持 0D scalar": "是" if has_scalar else "待补充",
        "是否有 dtype 相关分支": ", ".join(dtype_tags) or "待确认",
        "是否有动态 shape / 动态 rank": "已覆盖模型/边界 shape，动态约束需结合源码确认" if has_dynamic else "待确认",
        "真实模型中常见 shape": "结合算子场景补充真实模型 shape",
        "非法 shape / 报错 shape": ", ".join(invalid_cases) or "待补充",
        "case_mapping": case_mapping,
    }


def summarize_case_coverage(cases: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = meta or {}
    dtypes = sorted({str(case.get("dtype", "")) for case in cases if case.get("dtype")})
    functional_dtypes = _functional_dtypes(cases)
    tags = sorted({str(tag) for case in cases for tag in case.get("checklist_tags", [])})
    shapes = []
    for case in cases:
        if "input_shapes" in case:
            shapes.append(case["input_shapes"])
        else:
            shapes.append({"*": case.get("shape", [])})
    error_cases = sum(1 for c in cases if c.get("expect_error"))
    backward_cases = sum(1 for c in cases if c.get("require_grad"))
    return {
        "case_count": len(cases),
        "dtypes": dtypes,
        "functional_dtypes": functional_dtypes,
        "shapes": shapes,
        "checklist_tags": tags,
        "has_empty_tensor": "empty_tensor_input" in tags,
        "has_special_values": "inf_and_nan" in tags,
        "has_value_range": "value_range" in tags,
        "has_multiple_dtypes": len(dtypes) >= 2,
        "has_mixed_dtype": "mixed_dtype" in tags,
        "has_broadcast": "broadcast" in tags,
        "has_error_cases": error_cases > 0,
        "has_backward_cases": backward_cases > 0,
        "error_case_count": error_cases,
        "backward_case_count": backward_cases,
        "shape_design": _shape_design(meta, cases),
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# file I/O
# ---------------------------------------------------------------------------

def _write(path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def generate_acceptance_cases(
    meta_path: str | Path,
    output_root: str | Path,
    overwrite: bool = False,
    operator_contract_path: str | Path | None = None,
    case_plan_path: str | Path | None = None,
) -> dict[str, Path]:
    meta = read_json(Path(meta_path))
    root = Path(output_root)
    project_root = _project_root_from_output_root(root)
    op_name = meta["operator_name"]
    safe_name = sanitize_identifier(op_name)
    operator_contract = _optional_artifact(operator_contract_path)
    case_plan = _optional_artifact(case_plan_path)
    cases = _dispatch_cases(meta)
    cases = _add_required_dtype_cases(cases, meta, _required_dtypes(operator_contract, case_plan))
    coverage = summarize_case_coverage(cases, meta)
    if isinstance(case_plan, dict) and isinstance(case_plan.get("shape_design"), dict):
        coverage["shape_design"] = case_plan["shape_design"]
    coverage["artifact_inputs"] = {
        "operator_contract": _artifact_input_summary(operator_contract_path, operator_contract, op_name),
        "case_plan": _artifact_input_summary(case_plan_path, case_plan, op_name),
    }
    coverage["plan_alignment"] = _plan_alignment(case_plan, cases)
    coverage["opinfo_framework"] = _ms_adapter_opinfo_status(project_root, meta)

    values = {
        "operator_name": op_name,
        "safe_name": safe_name,
        "meta_json": json.dumps(meta, ensure_ascii=False, indent=2),
        "cases_json": json.dumps(cases, ensure_ascii=False, indent=2),
    }

    opinfo_status = coverage["opinfo_framework"]
    opinfo_framework = project_root / "tests" / "shared" / "opinfo"
    opinfo_case = project_root / str(opinfo_status.get("test_path", "tests/ms_adapter/opinfo/test_unary_ops.py"))
    profiler_framework = root / "profiler" / "acceptance_profiler.py"
    profiler_case = root / "profiler" / f"test_{safe_name}_profiler.py"
    results_keep = root / "results" / ".gitkeep"

    _write(profiler_framework, render_template("acceptance_profiler.py.template", values), overwrite)
    _write(profiler_case, render_template("profiler_case.py.template", values), overwrite)
    _write(results_keep, "", overwrite=True)
    _write(root / "results" / safe_name / "case_coverage.json", json.dumps(coverage, indent=2), overwrite=True)

    return {
        "opinfo_framework": opinfo_framework,
        "opinfo_database": project_root / "tests" / "ms_adapter" / "opinfo" / "op_database.py",
        "opinfo_sample_inputs": project_root / "tests" / "ms_adapter" / "opinfo" / "op_sample_inputs.py",
        "opinfo_wrappers": project_root / "tests" / "ms_adapter" / "opinfo" / "op_wrappers.py",
        "profiler_framework": profiler_framework,
        "opinfo_case": opinfo_case,
        "profiler_case": profiler_case,
        "results_keep": results_keep,
        "case_coverage": root / "results" / safe_name / "case_coverage.json",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--operator-contract")
    parser.add_argument("--case-plan")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    try:
        generated = generate_acceptance_cases(
            args.meta,
            args.output_root,
            args.overwrite,
            operator_contract_path=args.operator_contract,
            case_plan_path=args.case_plan,
        )
    except Exception as exc:
        print(f"[custom-op-acceptance] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: str(value) for key, value in generated.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
