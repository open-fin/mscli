"""Tests for the custom-op-acceptance skill helper scripts."""
from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np


PANGU_ROOT = Path(os.environ.get("PANGU_ROOT", r"D:\code\pangu"))
REPO_ROOT = PANGU_ROOT / "ms_ops"
SKILL_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    script_path = SKILL_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with_pta_baseline(meta: dict) -> dict:
    updated = dict(meta)
    updated.update(
        {
            "has_pta": True,
            "pta_api": "torch.ops.custom.npu_interleave_rope",
            "pta_baseline_api": "torch_npu.npu_interleave_rope",
            "pta_requires_custom_build": False,
            "pta_bare_name": "npu_interleave_rope",
            "pta_namespace": "custom",
        }
    )
    return updated


def _shape_rule_tags() -> set[str]:
    return {"shape_0d", "single_element", "boundary_shape", "non_aligned_shape", "model_shape", "shape_coverage"}


def test_common_case_ids_from_value_extracts_nested_ids():
    common = _load_script("common")

    value = {
        "direct": "case_a",
        "case_object": {"case_id": "case_b", "ignored": "not_a_case"},
        "nested": [{"id": "case_c"}, {"name": "case_d"}, {"items": ["case_e", ""]}],
    }

    assert common.case_ids_from_value(value) == {"case_a", "case_b", "case_c", "case_d", "case_e"}


def _shape_design_template(operator_name: str) -> dict[str, object]:
    return {
        "analysis_status": "agent_confirmed",
        "算子名称": operator_name,
        "输入 shape 规则": "任意 rank Tensor，逐元素计算",
        "输出 shape 规则": "输出 shape 与输入 Tensor 完全一致",
        "关键维度含义": "所有维度均为元素维度，无 axis 语义",
        "需要覆盖的 axis / dim": "不涉及 axis",
        "需要覆盖的边界 shape": "0D scalar、单元素、empty tensor、非对齐长度、真实模型大 shape",
        "是否支持 broadcast": "不涉及，单 Tensor 输入",
        "是否支持 empty tensor": "支持并已覆盖",
        "是否支持 0D scalar": "支持并已覆盖",
        "是否有 dtype 相关分支": "float16/float32/bfloat16 对齐 PTA",
        "是否有动态 shape / 动态 rank": "支持动态 shape/rank，按同 shape 输出",
        "真实模型中常见 shape": "[batch, hidden]，例如 [16, 4096]",
        "非法 shape / 报错 shape": "无非法 shape 约束；非法 dtype 报错用例覆盖",
        "case_mapping": {
            "边界 shape": ["shape_0d_fp16", "empty_fp32", "shape_non_aligned_2d_fp16"],
            "真实模型 shape": ["shape_model_batch_hidden_fp16"],
            "非法 shape / 报错 shape": ["error_invalid_dtype_int32"],
            "0D scalar": ["shape_0d_fp16"],
            "empty tensor": ["empty_fp32"],
        },
    }


def _shape_case_results() -> list[dict[str, object]]:
    return [
        {"case_id": "shape_0d_fp16", "outcome": "passed", "duration_s": 0.001},
        {"case_id": "empty_fp32", "outcome": "passed", "duration_s": 0.001},
        {"case_id": "shape_non_aligned_2d_fp16", "outcome": "passed", "duration_s": 0.001},
        {"case_id": "shape_model_batch_hidden_fp16", "outcome": "passed", "duration_s": 0.001},
        {"case_id": "error_invalid_dtype_int32", "outcome": "passed", "duration_s": 0.001},
    ]


def _operator_contract_template(operator_name: str) -> dict[str, object]:
    return {
        "operator_name": operator_name,
        "analysis_status": "agent_confirmed",
        "ms_api": f"custom_ops.ops.{operator_name}",
        "aclnn_api": "",
        "pta_api": "torch_npu.npu_fast_gelu",
        "pta_requires_custom_build": False,
        "inputs": [{"name": "self", "dtype": "Tensor", "required": True}],
        "attributes": [],
        "outputs": [{"name": "out", "dtype": "Tensor"}],
        "shape_rule": {
            "input_rule": "self 支持任意 rank Tensor",
            "output_rule": "输出 shape 与 self 一致",
            "key_dimensions": "所有维度均为元素维度",
            "axis_or_dim": "不涉及 axis/dim",
            "supports_broadcast": False,
            "supports_empty_tensor": True,
            "supports_0d_scalar": True,
            "dynamic_shape_or_rank": "支持动态 shape/rank",
        },
        "dtype_rule": {
            "pta_supported": ["float16", "float32", "bfloat16"],
            "ms_expected": ["float16", "float32", "bfloat16"],
            "dtype_branches": "浮点 dtype 分支",
        },
        "platform_requirements": [],
        "backward_rule": "单算子反向并已检查 input grad guard",
        "evidence": ["custom_ops/ms_adapter/pynative/npu_fast_gelu.cpp:42"],
    }


def _case_plan_template(operator_name: str) -> dict[str, object]:
    return {
        "operator_name": operator_name,
        "analysis_status": "agent_confirmed",
        "shape_design": _shape_design_template(operator_name),
        "dtype_plan": {
            "pta_supported": ["float16", "float32", "bfloat16"],
            "ms_expected": ["float16", "float32", "bfloat16"],
            "cases": ["shape_0d_fp16", "shape_model_batch_hidden_fp16"],
        },
        "functional_cases": ["shape_0d_fp16", "empty_fp32", "shape_model_batch_hidden_fp16"],
        "profiler_cases": ["shape_0d_fp16"],
        "negative_cases": ["error_invalid_dtype_int32"],
        "checklist_mapping": {
            "shape": ["shape_0d_fp16", "shape_model_batch_hidden_fp16", "error_invalid_dtype_int32"],
            "dtype": ["shape_0d_fp16", "shape_model_batch_hidden_fp16"],
        },
    }


def _write_ms_adapter_opinfo_framework(project_root: Path, op_name: str, category: str = "unary") -> None:
    shared = project_root / "tests" / "shared"
    opinfo = project_root / "tests" / "ms_adapter" / "opinfo"
    (shared / "opinfo").mkdir(parents=True, exist_ok=True)
    (shared / "_op_info").mkdir(parents=True, exist_ok=True)
    opinfo.mkdir(parents=True, exist_ok=True)

    (shared / "opinfo" / "runners.py").write_text("# fake runner\n", encoding="utf-8")
    (shared / "_op_info" / "op_info.py").write_text("# fake opinfo model\n", encoding="utf-8")
    lists = {
        "unary": [op_name] if category == "unary" else [],
        "binary": [op_name] if category == "binary" else [],
        "other": [op_name] if category == "other" else [],
    }
    (opinfo / "op_database.py").write_text(
        "\n".join(
            [
                f'op_db = {{"{op_name}": object()}}',
                f"unary_op_db = {lists['unary']!r}",
                f"binary_op_db = {lists['binary']!r}",
                f"other_op_db = {lists['other']!r}",
            ]
        ),
        encoding="utf-8",
    )
    (opinfo / "op_sample_inputs.py").write_text("# fake sample inputs\n", encoding="utf-8")
    (opinfo / "op_wrappers.py").write_text("# fake wrappers\n", encoding="utf-8")
    for test_category in ("unary", "binary", "other"):
        (opinfo / f"test_{test_category}_ops.py").write_text(
            "def test_opinfo_driver():\n    assert True\n",
            encoding="utf-8",
        )


def _artifact_aligned_case_coverage(operator_name: str) -> dict[str, object]:
    generated_case_ids = sorted(
        {
            "shape_0d_fp16",
            "empty_fp32",
            "shape_non_aligned_2d_fp16",
            "shape_model_batch_hidden_fp16",
            "error_invalid_dtype_int32",
        }
    )
    return {
        "artifact_inputs": {
            "operator_contract": {
                "provided": True,
                "analysis_status": "agent_confirmed",
                "operator_name": operator_name,
                "operator_name_matches": True,
            },
            "case_plan": {
                "provided": True,
                "analysis_status": "agent_confirmed",
                "operator_name": operator_name,
                "operator_name_matches": True,
            },
        },
        "plan_alignment": {
            "case_plan_provided": True,
            "generated_case_ids": generated_case_ids,
            "case_plan_case_ids": sorted(
                {
                    "shape_0d_fp16",
                    "empty_fp32",
                    "shape_non_aligned_2d_fp16",
                    "shape_model_batch_hidden_fp16",
                    "error_invalid_dtype_int32",
                }
            ),
            "missing_generated_case_ids": [],
            "extra_generated_case_ids": [],
        },
    }


def _with_acceptance_artifacts(run_summary: dict, operator_name: str) -> dict:
    updated = dict(run_summary)
    updated["operator_contract"] = _operator_contract_template(operator_name)
    updated["case_plan"] = _case_plan_template(operator_name)
    coverage = dict(updated.get("case_coverage", {}))
    coverage.update(_artifact_aligned_case_coverage(operator_name))
    updated["case_coverage"] = coverage
    return updated


def test_scaffold_acceptance_artifacts_creates_draft_contract_and_case_plan(tmp_path):
    scaffold = _load_script("scaffold_acceptance_artifacts")
    meta = {
        "operator_name": "demo_op",
        "ms_api": "custom_ops.ops.demo_op",
        "aclnn_api": "aclnnDemo",
        "pta_api": "torch_npu.demo_op",
        "pta_requires_custom_build": False,
        "inputs": [{"name": "x", "dtype": "Tensor"}],
        "attributes": [],
        "outputs": [{"name": "out", "dtype": "Tensor"}],
        "platform_requirements": [],
    }
    coverage = {"shape_design": _shape_design_template("demo_op")}
    meta_path = tmp_path / "operator_meta.json"
    coverage_path = tmp_path / "case_coverage.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")

    result = scaffold.scaffold_acceptance_artifacts(meta_path, tmp_path, coverage_path)

    contract = json.loads((tmp_path / "operator_contract.json").read_text(encoding="utf-8"))
    case_plan = json.loads((tmp_path / "case_plan.json").read_text(encoding="utf-8"))
    assert result["operator_contract_written"] is True
    assert result["case_plan_written"] is True
    assert contract["analysis_status"] == "draft_needs_agent_confirmation"
    assert contract["evidence"] == []
    assert case_plan["analysis_status"] == "draft_needs_agent_confirmation"
    assert case_plan["shape_design"]["analysis_status"] == "draft_needs_agent_confirmation"
    assert "shape_0d_fp16" in case_plan["functional_cases"]
    assert "error_invalid_dtype_int32" in case_plan["negative_cases"]

    second = scaffold.scaffold_acceptance_artifacts(meta_path, tmp_path, coverage_path)
    assert second["operator_contract_written"] is False
    assert second["case_plan_written"] is False


def test_inspect_operator_reads_aclnn_metadata():
    inspect_operator = _load_script("inspect_operator")

    meta = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_interleave_rope_op")

    assert meta["operator_name"] == "npu_interleave_rope_op"
    assert meta["kind"] == "aclnn"
    assert meta["ms_api"] == "custom_ops.ops.npu_interleave_rope_op"
    assert meta["aclnn_api"] == "aclnnInterleaveRope"
    if meta["has_pta"]:
        assert meta["pta_api"] == "torch.ops.custom.npu_interleave_rope"
        assert meta["pta_baseline_api"] in {
            "torch.ops.custom.npu_interleave_rope",
            "torch_npu.npu_interleave_rope",
        }
    else:
        assert meta["pta_api"] is None
    assert [arg["name"] for arg in meta["inputs"]] == ["x", "cos", "sin"]


def test_inspect_operator_reads_function_backward_metadata():
    inspect_operator = _load_script("inspect_operator")

    meta = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_fast_gelu")

    assert meta["operator_name"] == "npu_fast_gelu"
    assert meta["kind"] == "function"
    assert meta["ms_api"] == "custom_ops.ops.npu_fast_gelu"
    assert meta["has_backward"] is True
    assert meta["has_pta"] is True
    assert meta["pta_baseline_api"] == "torch_npu.npu_fast_gelu"
    assert [arg["name"] for arg in meta["inputs"]] == ["self"]


def test_common_parse_signature_accepts_void_outputs():
    common = _load_script("common")

    parsed = common.parse_signature("npu_prefetch_op(Tensor self) -> ()")

    assert parsed["name"] == "npu_prefetch_op"
    assert [arg["name"] for arg in parsed["inputs"]] == ["self"]
    assert parsed["outputs"] == []


def test_inspect_operator_discovers_external_baseline_and_platform_requirement():
    inspect_operator = _load_script("inspect_operator")

    attention = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_fused_infer_attention_score_op")
    mhc = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_mhc_pre_op")
    fast_gelu = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_fast_gelu_op")
    fast_gelu_backward = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_fast_gelu_backward_op")

    assert attention["pta_baseline_api"] == "torch_npu.npu_fused_infer_attention_score"
    assert attention["has_pta"] is True
    assert fast_gelu["pta_baseline_api"] == "torch_npu.npu_fast_gelu"
    assert fast_gelu["has_pta"] is True
    assert fast_gelu_backward["pta_baseline_api"] == "torch_npu.npu_fast_gelu_backward"
    assert fast_gelu_backward["has_pta"] is True
    assert fast_gelu["platform_requirements"]["required_devices"] == []
    assert "Ascend950" in mhc["platform_requirements"]["required_devices"]
    assert mhc["pta_baseline_api"] == "torch_npu.npu_mhc_pre"
    assert [output["name"] for output in mhc["outputs"]] == [
        "h_in",
        "h_post",
        "h_res",
        "inv_rms",
        "h_mix",
        "h_pre",
    ]


def test_generate_cases_uses_ms_adapter_opinfo_and_creates_profiler_files(tmp_path):
    inspect_operator = _load_script("inspect_operator")
    generate_cases = _load_script("generate_cases")

    meta = _with_pta_baseline(inspect_operator.collect_operator_meta(REPO_ROOT, "npu_interleave_rope_op"))
    project_root = tmp_path / "project"
    _write_ms_adapter_opinfo_framework(project_root, "npu_interleave_rope_op")
    meta_path = tmp_path / "operator_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    output_root = project_root / "tests" / "acceptance"
    generated = generate_cases.generate_acceptance_cases(meta_path, output_root, overwrite=True)

    expected = {
        "opinfo_framework": project_root / "tests" / "shared" / "opinfo",
        "opinfo_case": project_root / "tests" / "ms_adapter" / "opinfo" / "test_unary_ops.py",
        "profiler_case": output_root / "profiler" / "test_npu_interleave_rope_op_profiler.py",
    }
    assert expected.items() <= generated.items()
    for path in expected.values():
        assert path.exists(), path

    profiler_text = expected["profiler_case"].read_text(encoding="utf-8")
    assert "torch.ops.custom.npu_interleave_rope" in profiler_text
    assert "torch_npu.npu_interleave_rope" in profiler_text
    coverage = json.loads(generated["case_coverage"].read_text(encoding="utf-8"))
    assert coverage["opinfo_framework"]["status"] == "ready"
    assert coverage["opinfo_framework"]["opinfo_name"] == "npu_interleave_rope_op"
    cases = coverage["cases"]
    assert cases[0]["dtype"] == "float16"
    assert cases[0]["input_shapes"]["x"] == [2, 4, 8, 64]
    assert cases[0]["input_shapes"]["cos"] == [2, 1, 8, 64]
    assert cases[0]["input_shapes"]["sin"] == [2, 1, 8, 64]
    assert "ms_end_to_end_ms" in profiler_text
    assert "pta_end_to_end_ms" in profiler_text
    assert "ms_kernel_total_us" in profiler_text
    assert "pta_kernel_total_us" in profiler_text
    assert "kernel_consistency" in profiler_text
    assert "_materialize_ms(func(*args, **kwargs))" in profiler_text
    assert "np.testing.assert_allclose" in profiler_text
    assert "dtype_support.json" in profiler_text
    assert "test_npu_interleave_rope_op_dtype_support" in profiler_text
    assert "case_failures" in profiler_text
    assert 'RESULT_DIR / "profile_summary.json"' in profiler_text


def test_generate_cases_records_contract_and_case_plan_inputs(tmp_path):
    generate_cases = _load_script("generate_cases")

    meta = {
        "operator_name": "npu_fast_gelu",
        "ms_api": "custom_ops.ops.npu_fast_gelu",
        "has_pta": True,
        "inputs": [{"name": "self", "dtype": "Tensor"}],
        "attributes": [],
        "outputs": [{"name": "out", "dtype": "Tensor"}],
    }
    meta_path = tmp_path / "operator_meta.json"
    contract_path = tmp_path / "operator_contract.json"
    case_plan_path = tmp_path / "case_plan.json"
    output_root = tmp_path / "generated"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    contract_path.write_text(json.dumps(_operator_contract_template("npu_fast_gelu")), encoding="utf-8")
    case_plan = _case_plan_template("npu_fast_gelu")
    case_plan["functional_cases"] = ["shape_0d_fp16", "missing_planned_case"]
    case_plan_path.write_text(json.dumps(case_plan), encoding="utf-8")

    generated = generate_cases.generate_acceptance_cases(
        meta_path,
        output_root,
        overwrite=True,
        operator_contract_path=contract_path,
        case_plan_path=case_plan_path,
    )
    coverage = json.loads(generated["case_coverage"].read_text(encoding="utf-8"))

    assert coverage["artifact_inputs"]["operator_contract"]["provided"] is True
    assert coverage["artifact_inputs"]["case_plan"]["provided"] is True
    assert "shape_0d_fp16" in coverage["plan_alignment"]["case_plan_case_ids"]
    assert "missing_planned_case" in coverage["plan_alignment"]["case_plan_case_ids"]
    assert "missing_planned_case" in coverage["plan_alignment"]["missing_generated_case_ids"]


def test_generate_cases_uses_confirmed_case_plan_shape_design(tmp_path):
    generate_cases = _load_script("generate_cases")

    meta = {
        "operator_name": "demo_op",
        "ms_api": "custom_ops.ops.demo_op",
        "has_pta": True,
        "inputs": [{"name": "x", "dtype": "Tensor"}],
        "attributes": [],
        "outputs": [{"name": "out", "dtype": "Tensor"}],
    }
    meta_path = tmp_path / "operator_meta.json"
    contract_path = tmp_path / "operator_contract.json"
    case_plan_path = tmp_path / "case_plan.json"
    output_root = tmp_path / "generated"
    case_plan = _case_plan_template("demo_op")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    contract_path.write_text(json.dumps(_operator_contract_template("demo_op")), encoding="utf-8")
    case_plan_path.write_text(json.dumps(case_plan), encoding="utf-8")

    generated = generate_cases.generate_acceptance_cases(
        meta_path,
        output_root,
        overwrite=True,
        operator_contract_path=contract_path,
        case_plan_path=case_plan_path,
    )
    coverage = json.loads(generated["case_coverage"].read_text(encoding="utf-8"))

    assert coverage["shape_design"] == case_plan["shape_design"]


def test_pytest_summary_parser_does_not_treat_skips_as_clean_pass():
    evidence = _load_script("evidence")

    parsed = evidence.parse_pytest_output(
        """
        .s.s...
        SKIPPED [1] tests/operators/functions/test_fast_gelu.py:30: Need baseline data from torch_npu
        SKIPPED [1] tests/operators/functions/test_fast_gelu.py:50: Need baseline data from torch_npu
        5 passed, 2 skipped, 22 warnings in 6.12s
        """
    )

    assert parsed["counts"]["passed"] == 5
    assert parsed["counts"]["skipped"] == 2
    assert parsed["counts"]["warnings"] == 22
    assert not parsed["clean_pass"]
    assert len(parsed["skip_reasons"]) == 2
    assert parsed["skip_reasons"][0]["reason"] == "Need baseline data from torch_npu"


def test_pytest_summary_parser_does_not_treat_xfail_as_clean_pass():
    evidence = _load_script("evidence")

    parsed = evidence.parse_pytest_output("5 passed, 1 xfailed, 22 warnings in 6.12s")

    assert parsed["counts"]["passed"] == 5
    assert parsed["counts"]["xfailed"] == 1
    assert not parsed["clean_pass"]


def test_pytest_summary_parser_accepts_only_zero_skip_zero_failure_pass():
    evidence = _load_script("evidence")

    parsed = evidence.parse_pytest_output("3 passed, 22 warnings in 6.02s")

    assert parsed["counts"]["passed"] == 3
    assert parsed["counts"]["skipped"] == 0
    assert parsed["counts"]["failed"] == 0
    assert parsed["counts"]["errors"] == 0
    assert parsed["clean_pass"]


def test_static_test_analyzer_finds_empty_tests_and_placeholder_skips(tmp_path):
    evidence = _load_script("evidence")
    test_file = tmp_path / "test_placeholder.py"
    test_file.write_text(
        '''
import pytest

def test_empty():
    """Only a docstring."""

def test_placeholder_skip():
    pytest.skip("Need baseline data from torch_npu")

def test_real():
    assert 1 == 1
''',
        encoding="utf-8",
    )

    analysis = evidence.analyze_pytest_file(test_file)

    assert analysis["empty_tests"] == [{"name": "test_empty", "lineno": 4}]
    assert analysis["placeholder_skips"] == [
        {
            "name": "test_placeholder_skip",
            "lineno": 7,
            "reason": "Need baseline data from torch_npu",
        }
    ]


def test_generated_cases_expose_checklist_driven_coverage(tmp_path):
    inspect_operator = _load_script("inspect_operator")
    generate_cases = _load_script("generate_cases")

    meta = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_fast_gelu_op")
    meta_path = tmp_path / "operator_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    generated = generate_cases.generate_acceptance_cases(meta_path, tmp_path / "generated", overwrite=True)
    coverage = json.loads(generated["case_coverage"].read_text(encoding="utf-8"))
    cases = coverage["cases"]

    assert {"float16", "float32"} <= set(coverage["dtypes"])
    assert "empty_tensor_input" in coverage["checklist_tags"]
    assert "inf_and_nan" in coverage["checklist_tags"]
    assert coverage["case_count"] >= 10
    assert "shape_0d" in coverage["checklist_tags"]
    assert "shape_7d" not in coverage["checklist_tags"]
    assert "shape_8d" not in coverage["checklist_tags"]
    assert "dtype_float16" in coverage["checklist_tags"]
    assert "large_tensor" in coverage["checklist_tags"]
    shape_design = coverage["shape_design"]
    assert shape_design["算子名称"] == "npu_fast_gelu_op"
    assert shape_design["输入 shape 规则"]
    assert shape_design["输出 shape 规则"]
    assert shape_design["关键维度含义"]
    assert shape_design["需要覆盖的边界 shape"]
    assert shape_design["真实模型中常见 shape"]
    assert shape_design["非法 shape / 报错 shape"]
    assert shape_design["analysis_status"] == "script_confirmed"
    assert shape_design["case_mapping"]["边界 shape"]
    assert shape_design["case_mapping"]["真实模型 shape"]
    assert shape_design["case_mapping"]["非法 shape / 报错 shape"]
    valid_cases = [case for case in cases if not case.get("expect_error")]
    assert {case["dtype"] for case in valid_cases} <= {"float16", "float32"}
    assert any(
        case["case_id"] == "error_invalid_dtype_int32" and case["expect_error"]
        for case in cases
    )


def test_generate_cases_adds_dtype_cases_required_by_confirmed_artifacts(tmp_path):
    inspect_operator = _load_script("inspect_operator")
    generate_cases = _load_script("generate_cases")

    meta = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_fast_gelu_op")
    meta_path = tmp_path / "operator_meta.json"
    contract_path = tmp_path / "operator_contract.json"
    case_plan_path = tmp_path / "case_plan.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    contract_path.write_text(json.dumps(_operator_contract_template("npu_fast_gelu_op")), encoding="utf-8")
    case_plan_path.write_text(json.dumps(_case_plan_template("npu_fast_gelu_op")), encoding="utf-8")

    generated = generate_cases.generate_acceptance_cases(
        meta_path,
        tmp_path / "generated",
        overwrite=True,
        operator_contract_path=contract_path,
        case_plan_path=case_plan_path,
    )
    coverage = json.loads(generated["case_coverage"].read_text(encoding="utf-8"))

    assert "bfloat16" in coverage["functional_dtypes"]
    bf16_case = next(case for case in coverage["cases"] if case["case_id"] == "dtype_bfloat16_2d")
    assert bf16_case["dtype"] == "bfloat16"
    assert bf16_case["expect_error"] is False
    assert "dtype_bfloat16" in bf16_case["checklist_tags"]


def test_generated_fast_gelu_backward_cases_use_same_shape_inputs_without_grad_case(tmp_path):
    inspect_operator = _load_script("inspect_operator")
    generate_cases = _load_script("generate_cases")

    meta = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_fast_gelu_backward_op")
    meta_path = tmp_path / "operator_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    generated = generate_cases.generate_acceptance_cases(meta_path, tmp_path / "generated", overwrite=True)
    coverage = json.loads(generated["case_coverage"].read_text(encoding="utf-8"))
    cases = coverage["cases"]

    assert not any(case.get("require_grad") for case in cases)
    assert not any(case["case_id"] == "broadcast_fp16" for case in cases)
    for case in cases:
        if case.get("expect_error"):
            continue
        if case.get("input_shapes"):
            assert case["input_shapes"]["grad"] == case["input_shapes"]["self"]


def test_generated_attention_cases_keep_optional_host_arrays_none_by_default(tmp_path):
    inspect_operator = _load_script("inspect_operator")
    generate_cases = _load_script("generate_cases")

    meta = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_fused_infer_attention_score_op")
    meta_path = tmp_path / "operator_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    generated = generate_cases.generate_acceptance_cases(meta_path, tmp_path / "generated", overwrite=True)
    coverage = json.loads(generated["case_coverage"].read_text(encoding="utf-8"))
    cases = coverage["cases"]
    bsh_case = next(case for case in cases if case["case_id"] == "bsh_small_fp16_lse")
    tnd_case = next(case for case in cases if case["case_id"] == "tnd_gqa_fp16")

    assert bsh_case["attributes"]["actual_seq_lengths"] is None
    assert bsh_case["attributes"]["actual_seq_lengths_kv"] is None
    assert tnd_case["attributes"]["actual_seq_lengths"] == [1, 3, 6]


def test_generated_runtime_converts_host_array_attrs_and_pta_kwargs(tmp_path):
    inspect_operator = _load_script("inspect_operator")
    generate_cases = _load_script("generate_cases")

    meta = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_fused_infer_attention_score_op")
    meta_path = tmp_path / "operator_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    generated = generate_cases.generate_acceptance_cases(meta_path, tmp_path / "generated", overwrite=True)

    profiler_text = generated["profiler_case"].read_text(encoding="utf-8")
    assert "func(*args, **kwargs)" in profiler_text


def test_profiler_kernel_parser_reads_csv_and_compares(tmp_path):
    inspect_operator = _load_script("inspect_operator")
    generate_cases = _load_script("generate_cases")

    meta = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_interleave_rope_op")
    meta_path = tmp_path / "operator_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    output_root = tmp_path / "generated"
    generated = generate_cases.generate_acceptance_cases(meta_path, output_root, overwrite=True)
    spec = importlib.util.spec_from_file_location(
        "generated_acceptance_profiler",
        generated["profiler_framework"],
    )
    assert spec is not None and spec.loader is not None
    profiler_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profiler_module)

    ms_dir = tmp_path / "ms"
    pta_dir = tmp_path / "pta"
    ms_dir.mkdir()
    pta_dir.mkdir()
    csv_text = "Name,Duration(us)\nacl/InterleaveRopeKernel,12.5\n"
    (ms_dir / "kernel_details.csv").write_text(csv_text, encoding="utf-8")
    (pta_dir / "kernel_details.csv").write_text(csv_text, encoding="utf-8")

    ms_summary = profiler_module.summarize_kernel_records(
        profiler_module.extract_kernel_records(ms_dir)
    )
    pta_summary = profiler_module.summarize_kernel_records(
        profiler_module.extract_kernel_records(pta_dir)
    )
    comparison = profiler_module.compare_kernel_summaries(ms_summary, pta_summary)

    assert ms_summary["sequence"] == ["InterleaveRopeKernel"]
    assert ms_summary["total_duration_us"] == 12.5
    assert comparison["status"] == "passed"


def test_run_acceptance_exposes_direct_pytest_commands():
    run_acceptance = _load_script("run_acceptance")

    commands = run_acceptance.build_direct_validation_commands(
        "npu_interleave_rope_op",
        run_pta_compare=True,
        build_pta_custom=False,
        opinfo_status={
            "registered": True,
            "test_path": "tests/ms_adapter/opinfo/test_unary_ops.py",
            "opinfo_name": "npu_interleave_rope_op",
        },
    )

    assert commands == [
        "python compile.py --release",
        (
            "CUSTOM_OPS_MODE=release MS_DISABLE_KERNEL_BACKOFF=1 python -m pytest "
            "tests/ms_adapter/opinfo/test_unary_ops.py -k npu_interleave_rope_op "
            "-v --tb=line --durations=0 -ra"
        ),
        (
            "python -m pytest tests/acceptance/profiler/test_npu_interleave_rope_op_profiler.py "
            "-v --tb=line --durations=0 -ra"
        ),
    ]

    custom_pta_commands = run_acceptance.build_direct_validation_commands(
        "some_custom_pta_op",
        run_pta_compare=True,
        build_pta_custom=True,
    )
    assert custom_pta_commands[0] == "python compile.py --release"


def test_run_acceptance_uses_canonical_operator_name_for_outputs_and_commands(tmp_path, monkeypatch):
    run_acceptance = _load_script("run_acceptance")

    project_root = tmp_path / "project"
    config_dir = project_root / "custom_ops" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "aclnn_ops.yaml").write_text(
        """
- op: demo_op(Tensor x) -> Tensor out
  api: aclnnDemo
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        run_acceptance,
        "run_preflight",
        lambda require_pta, meta=None: {"is_ascend_ready": False, "missing": ["mindspore"], "python": "python"},
    )

    summary = run_acceptance.run_acceptance("demo", project_root, overwrite=True, run_pytest=False)

    assert summary["operator_name"] == "demo_op"
    assert "results\\demo_op" in summary["generated_files"]["case_coverage"]
    assert summary["operator_contract"]["analysis_status"] == "draft_needs_agent_confirmation"
    assert summary["case_plan"]["analysis_status"] == "draft_needs_agent_confirmation"
    assert summary["pending_commands"][1] == "register operator in tests/ms_adapter/opinfo before opinfo pytest"


def test_run_acceptance_includes_existing_contract_and_case_plan_in_summary(tmp_path, monkeypatch):
    run_acceptance = _load_script("run_acceptance")

    project_root = tmp_path / "project"
    config_dir = project_root / "custom_ops" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "aclnn_ops.yaml").write_text(
        """
- op: demo_op(Tensor x) -> Tensor out
  api: aclnnDemo
""",
        encoding="utf-8",
    )
    _write_ms_adapter_opinfo_framework(project_root, "demo_op")
    result_dir = project_root / "tests" / "acceptance" / "results" / "demo_op"
    result_dir.mkdir(parents=True)
    (result_dir / "operator_contract.json").write_text(
        json.dumps(_operator_contract_template("demo_op")),
        encoding="utf-8",
    )
    (result_dir / "case_plan.json").write_text(
        json.dumps(_case_plan_template("demo_op")),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        run_acceptance,
        "run_preflight",
        lambda require_pta, meta=None: {"is_ascend_ready": False, "missing": ["mindspore"], "python": "python"},
    )

    summary = run_acceptance.run_acceptance("demo", project_root, overwrite=True, run_pytest=False)

    assert summary["operator_contract"]["operator_name"] == "demo_op"
    assert summary["case_plan"]["operator_name"] == "demo_op"
    assert summary["operator_contract"]["analysis_status"] == "agent_confirmed"
    assert summary["case_plan"]["analysis_status"] == "agent_confirmed"


def test_run_acceptance_runs_build_before_pytest(tmp_path, monkeypatch):
    run_acceptance = _load_script("run_acceptance")

    project_root = tmp_path / "project"
    config_dir = project_root / "custom_ops" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "aclnn_ops.yaml").write_text(
        """
- op: demo_op(Tensor x) -> Tensor out
  api: aclnnDemo
""",
        encoding="utf-8",
    )
    _write_ms_adapter_opinfo_framework(project_root, "demo_op")
    monkeypatch.setattr(
        run_acceptance,
        "run_preflight",
        lambda require_pta, meta=None: {"is_ascend_ready": True, "missing": [], "python": "python"},
    )

    commands = []

    def fake_run(command, cwd, log_path, env=None):
        commands.append(command)
        return {
            "status": "passed",
            "returncode": 0,
            "pytest": {"counts": {"passed": 1, "failed": 0, "skipped": 0, "errors": 0}, "clean_pass": True},
            "log": str(log_path),
        }

    monkeypatch.setattr(run_acceptance, "_run_raw_command", fake_run)
    monkeypatch.setattr(run_acceptance, "_run_command", fake_run)

    summary = run_acceptance.run_acceptance("demo_op", project_root, overwrite=True, run_pytest=True)

    assert commands[0] == ["python", "compile.py", "--release"]
    assert commands[1][:3] == [run_acceptance.sys.executable, "-m", "pytest"]
    assert summary["build"]["status"] == "passed"


def test_run_acceptance_reads_dtype_support_matrix_from_profiler_results(tmp_path, monkeypatch):
    run_acceptance = _load_script("run_acceptance")

    project_root = tmp_path / "project"
    config_dir = project_root / "custom_ops" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "aclnn_ops.yaml").write_text(
        """
- op: demo_op(Tensor x) -> Tensor out
  api: aclnnDemo
  pta_extras:
    bare_name: demo
    namespace: custom
""",
        encoding="utf-8",
    )
    _write_ms_adapter_opinfo_framework(project_root, "demo_op")
    monkeypatch.setattr(
        run_acceptance,
        "run_preflight",
        lambda require_pta, meta=None: {"is_ascend_ready": True, "missing": [], "python": "python"},
    )

    def fake_raw(command, cwd, log_path):
        return {"status": "passed", "returncode": 0, "log": str(log_path)}

    def fake_run(command, cwd, log_path, env=None):
        if "profiler" in str(command):
            dtype_support = {
                "status": "passed",
                "pta_supported": ["float16", "float32"],
                "ms_supported": ["float16", "float32"],
                "missing_in_ms": [],
            }
            (log_path.parent / "dtype_support.json").write_text(
                json.dumps(dtype_support),
                encoding="utf-8",
            )
            profile_summary = {
                "status": "passed",
                "kernel_status": "passed",
                "memory_status": "passed",
                "cases": [
                    {
                        "case_id": "case1",
                        "ms_peak_memory_mb": 1.0,
                        "pta_peak_memory_mb": 1.0,
                        "memory_comparison": {"status": "passed"},
                    }
                ],
            }
            (log_path.parent / "profile_summary.json").write_text(
                json.dumps(profile_summary),
                encoding="utf-8",
            )
        return {
            "status": "passed",
            "returncode": 0,
            "pytest": {"counts": {"passed": 1, "failed": 0, "skipped": 0, "errors": 0}, "clean_pass": True},
            "log": str(log_path),
        }

    monkeypatch.setattr(run_acceptance, "_run_raw_command", fake_raw)
    monkeypatch.setattr(run_acceptance, "_run_command", fake_run)

    summary = run_acceptance.run_acceptance("demo_op", project_root, overwrite=True, run_pytest=True)

    assert summary["dtype_support"]["status"] == "passed"
    assert summary["dtype_support"]["pta_supported"] == ["float16", "float32"]


def test_run_acceptance_records_untriaged_opinfo_failure_and_stops_profiler(tmp_path, monkeypatch):
    run_acceptance = _load_script("run_acceptance")

    project_root = tmp_path / "project"
    config_dir = project_root / "custom_ops" / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "aclnn_ops.yaml").write_text(
        """
- op: demo_op(Tensor x) -> Tensor out
  api: aclnnDemo
  pta_extras:
    bare_name: demo
    namespace: custom
""",
        encoding="utf-8",
    )
    _write_ms_adapter_opinfo_framework(project_root, "demo_op")
    monkeypatch.setattr(
        run_acceptance,
        "run_preflight",
        lambda require_pta, meta=None: {"is_ascend_ready": True, "missing": [], "python": "python"},
    )
    commands = []

    def fake_raw(command, cwd, log_path):
        return {"status": "passed", "returncode": 0, "log": str(log_path)}

    def fake_run(command, cwd, log_path, env=None):
        commands.append(command)
        return {
            "status": "failed",
            "returncode": 1,
            "pytest": {"counts": {"passed": 0, "failed": 1, "skipped": 0, "errors": 0}, "clean_pass": False},
            "log": str(log_path),
        }

    monkeypatch.setattr(run_acceptance, "_run_raw_command", fake_raw)
    monkeypatch.setattr(run_acceptance, "_run_command", fake_run)

    summary = run_acceptance.run_acceptance("demo_op", project_root, overwrite=True, run_pytest=True)

    assert len(commands) == 1
    assert summary["profiler"]["status"] == "not_run"
    assert summary["profiler"]["reason"] == "opinfo_not_clean"
    assert summary["case_failure_analysis"][0]["classification"] == "needs_triage"


def test_run_preflight_blocks_unmatched_required_device(monkeypatch):
    run_acceptance = _load_script("run_acceptance")

    monkeypatch.setattr(run_acceptance, "_module_available", lambda name: True)
    monkeypatch.setattr(run_acceptance.shutil, "which", lambda name: "/usr/bin/npu-smi")
    monkeypatch.setattr(run_acceptance, "_npu_smi_output", lambda: "Name: Ascend910B3")

    preflight = run_acceptance.run_preflight(
        require_pta=True,
        meta={"platform_requirements": {"required_devices": ["Ascend950", "910_95"]}},
    )

    assert not preflight["is_ascend_ready"]
    assert preflight["missing"] == ["device_requires:Ascend950/910_95"]


def test_run_acceptance_collects_static_audit_evidence_for_fast_gelu():
    inspect_operator = _load_script("inspect_operator")
    run_acceptance = _load_script("run_acceptance")

    meta = inspect_operator.collect_operator_meta(REPO_ROOT, "npu_fast_gelu")
    audit = run_acceptance._collect_static_audits(REPO_ROOT, meta)

    assert audit["doc_audit"]["has_operator_doc"] is True
    assert audit["doc_audit"]["has_api_description"] is True
    assert audit["doc_audit"]["has_formula"] is True
    assert audit["doc_audit"]["has_platforms"] is True
    assert audit["doc_audit"]["has_example"] is True
    assert audit["doc_audit"]["has_example_output"] is True
    assert audit["source_audit"]["output_shape_rule"] == "same_shape"
    assert audit["source_audit"]["output_dtype_rule"] == "same_dtype"
    assert audit["source_audit"]["output_shape_depends_on_runtime_data"] is False
    assert audit["source_audit"]["backward_single_op"] is True
    assert audit["source_audit"]["backward_needs_input_grad_guard"] is True
    assert audit["source_audit"]["safety"]["status"] == "passed"
    assert audit["legacy_test_audit"]["status"] in {"not_run", "not_applicable"}
    assert audit["legacy_test_audit"]["matched_tests"]


def test_generate_report_requires_kernel_evidence_for_full_validation():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_interleave_rope_op",
        "ms_api": "custom_ops.ops.npu_interleave_rope_op",
        "pta_api": "torch.ops.custom.npu_interleave_rope",
        "has_pta": True,
        "has_backward": False,
    }
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pytest": {"status": "passed"},
        "profiler": {"status": "passed"},
        "kernel_consistency": {"status": "coverage_gap", "reason": "kernel summary not parsed"},
    }

    report = generate_report.build_acceptance_report(meta, run_summary)

    assert report["final_conclusion"] == "validated_with_coverage_gaps"


def test_generate_report_requires_memory_evidence_for_full_validation():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_interleave_rope_op",
        "ms_api": "custom_ops.ops.npu_interleave_rope_op",
        "pta_api": "torch.ops.custom.npu_interleave_rope",
        "has_pta": True,
        "has_backward": False,
    }
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pta_compare_enabled": True,
        "pytest": {"status": "passed", "pytest": {"counts": {"passed": 4}, "clean_pass": True}},
        "profiler": {"status": "passed", "pytest": {"counts": {"passed": 1}, "clean_pass": True}},
        "kernel_consistency": {"status": "passed"},
        "profile_summary": {
            "status": "coverage_gap",
            "kernel_status": "passed",
            "memory_status": "coverage_gap",
            "cases": [
                {
                    "case_id": "case1",
                    "ms_peak_memory_mb": 15.0,
                    "pta_peak_memory_mb": 10.0,
                    "memory_comparison": {"status": "coverage_gap", "ratio": 1.5},
                }
            ],
        },
        "case_coverage": {
            "case_count": 4,
            "dtypes": ["float16", "float32"],
            "checklist_tags": ["shape_0d", "empty_tensor_input", "inf_and_nan"],
            "has_multiple_dtypes": True,
            "has_special_values": True,
        },
    }

    report = generate_report.build_acceptance_report(meta, run_summary)

    assert report["final_conclusion"] == "validated_with_coverage_gaps"


def test_generate_report_rejects_uncollected_memory_as_full_validation():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_interleave_rope_op",
        "ms_api": "custom_ops.ops.npu_interleave_rope_op",
        "pta_api": "torch.ops.custom.npu_interleave_rope",
        "has_pta": True,
        "has_backward": False,
    }
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pta_compare_enabled": True,
        "pytest": {"status": "passed", "pytest": {"counts": {"passed": 4}, "clean_pass": True}},
        "profiler": {"status": "passed", "pytest": {"counts": {"passed": 1}, "clean_pass": True}},
        "kernel_consistency": {"status": "passed"},
        "profile_summary": {
            "status": "passed",
            "kernel_status": "passed",
            "memory_status": "passed",
            "cases": [
                {
                    "case_id": "case1",
                    "memory_comparison": {"status": "not_collected", "reason": "memory APIs unavailable"},
                }
            ],
        },
        "case_coverage": {
            "case_count": 4,
            "dtypes": ["float16", "float32"],
            "checklist_tags": ["shape_0d", "empty_tensor_input", "inf_and_nan"],
            "has_multiple_dtypes": True,
            "has_special_values": True,
        },
        "dtype_support": {
            "status": "passed",
            "pta_supported": ["float16"],
            "ms_supported": ["float16"],
            "missing_in_ms": [],
        },
    }

    report = generate_report.build_acceptance_report(meta, run_summary)

    assert report["final_conclusion"] == "validated_with_coverage_gaps"
    assert any(gap["area"] == "memory" for gap in report["skip_block_gap_details"])


def test_generate_report_memory_item_requires_every_case_memory_evidence():
    generate_report = _load_script("generate_report")

    result = generate_report._evaluate_item(
        "性能验证",
        {"item": "显存是否持平PTA（请将截图贴于右侧）"},
        {"has_pta": True},
        {
            "pta_compare_enabled": True,
            "profiler": {"status": "passed", "pytest": {"clean_pass": True}},
            "profile_summary": {
                "memory_status": "passed",
                "cases": [
                    {
                        "case_id": "case1",
                        "ms_peak_memory_mb": 10.0,
                        "pta_peak_memory_mb": 10.0,
                        "memory_comparison": {"status": "passed"},
                    },
                    {
                        "case_id": "case2",
                        "ms_peak_memory_mb": None,
                        "pta_peak_memory_mb": 10.0,
                        "memory_comparison": {"status": "not_collected"},
                    },
                ],
            },
        },
    )

    assert result["自测结果"] == "否"
    assert "不是所有 profiler case" in result["备注"]


def test_generate_report_keeps_memory_ratio_notes_non_blocking_but_visible():
    generate_report = _load_script("generate_report")

    result = generate_report._evaluate_item(
        "性能验证",
        {"item": "显存是否持平PTA（请将截图贴于右侧）"},
        {"has_pta": True},
        {
            "pta_compare_enabled": True,
            "profiler": {"status": "passed", "pytest": {"clean_pass": True}},
            "profile_summary": {
                "memory_status": "note",
                "cases": [
                    {
                        "case_id": "case1",
                        "ms_peak_memory_mb": 15.0,
                        "pta_peak_memory_mb": 10.0,
                        "memory_comparison": {
                            "status": "passed",
                            "ratio": 1.5,
                            "threshold": 1.1,
                            "reason": "MS/PTA memory ratio 1.50 exceeds threshold 1.1",
                        },
                    }
                ],
            },
        },
    )

    assert result["自测结果"] == "非阻塞备注"
    assert "仅备注不阻塞" in result["备注"]


def test_generate_report_audit_summary_separates_manual_and_advisory_items():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_fast_gelu",
        "ms_api": "custom_ops.ops.npu_fast_gelu",
        "pta_api": "torch_npu.npu_fast_gelu",
        "pta_baseline_api": "torch_npu.npu_fast_gelu",
        "has_pta": True,
        "has_backward": True,
        "inputs": [{"name": "self", "dtype": "Tensor", "default": None}],
        "attributes": [],
        "outputs": [{"name": "out", "dtype": "Tensor"}],
    }
    shape_tags = _shape_rule_tags()
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pta_compare_enabled": True,
        "pytest": {
            "status": "passed",
            "pytest": {"counts": {"passed": 14}, "clean_pass": True},
            "case_results": _shape_case_results(),
        },
        "profiler": {"status": "passed", "pytest": {"counts": {"passed": 2}, "clean_pass": True}},
        "kernel_consistency": {"status": "passed"},
        "dtype_support": {
            "status": "passed",
            "pta_supported": ["float16", "float32", "bfloat16"],
            "ms_supported": ["float16", "float32", "bfloat16"],
            "missing_in_ms": [],
        },
        "case_coverage": {
            "case_count": 14,
            "dtypes": ["float16", "float32", "bfloat16"],
            "functional_dtypes": ["float16", "float32", "bfloat16"],
            "checklist_tags": sorted(
                shape_tags
                | {
                    "default_parameters",
                    "empty_tensor_input",
                    "error_cases",
                    "inf_and_nan",
                    "value_range",
                    "non_contiguous",
                    "backward",
                }
            ),
            "has_multiple_dtypes": True,
            "has_special_values": True,
            "has_value_range": True,
            "has_error_cases": True,
            "error_case_count": 1,
            "shape_design": _shape_design_template("npu_fast_gelu"),
        },
        "doc_audit": {
            "operator_doc": "docs/operators/npu_fast_gelu.md",
            "has_operator_doc": True,
            "has_chinese_doc": True,
            "has_api_description": True,
            "has_formula": True,
            "has_platforms": True,
            "has_example": True,
            "has_example_output": True,
            "has_test_command": True,
            "input_description_complete": True,
            "output_description_complete": True,
            "raises_description_complete": True,
            "format_ok": True,
        },
        "source_audit": {
            "output_shape_rule": "same_shape",
            "output_dtype_rule": "same_dtype",
            "output_shape_depends_on_runtime_data": False,
            "backward_single_op": True,
            "backward_needs_input_grad_guard": True,
            "set_unused_inputs_required": False,
            "safety": {
                "status": "passed",
                "checked_files": ["npu_fast_gelu.cpp", "aclnn_fast_gelu_op.cpp"],
                "findings": [],
            },
        },
        "legacy_test_audit": {
            "status": "passed",
            "matched_tests": ["tests/legacy/ms_functions/test_fast_gelu.py"],
            "pytest": {"counts": {"passed": 7, "failed": 0, "skipped": 0, "errors": 0}, "clean_pass": True},
        },
        "backoff_validation": {
            "status": "passed",
            "env": {"MS_DISABLE_KERNEL_BACKOFF": "1"},
            "reason": "opinfo pytest passed with kernel backoff disabled",
        },
        "ai_review": {
            "status": "passed",
            "summary": "AI 已审视报告、用例覆盖、静态证据和运行结果，报告可供审核人判断交付状态。",
            "reviewer_value": "报告包含接口语义、覆盖范围、dtype 对齐、kernel/显存证据和非阻塞备注。",
            "remaining_issues": [],
        },
        "profile_summary": {
            "status": "passed",
            "kernel_status": "passed",
            "memory_status": "note",
            "cases": [
                {
                    "case_id": "shape_0d_fp16",
                    "ms_end_to_end_ms": 0.01,
                    "pta_end_to_end_ms": 0.02,
                    "ms_kernel_total_us": 0.9,
                    "pta_kernel_total_us": 1.0,
                    "ms_peak_memory_mb": 0.002,
                    "pta_peak_memory_mb": 0.001,
                    "kernel_consistency": {"status": "passed"},
                    "memory_comparison": {
                        "status": "passed",
                        "ms_peak_memory_mb": 0.002,
                        "pta_peak_memory_mb": 0.001,
                        "ratio": 2.0,
                        "threshold": 1.1,
                        "reason": "MS/PTA memory ratio 2.00 exceeds threshold 1.1",
                    },
                }
            ],
        },
    }
    run_summary = _with_acceptance_artifacts(run_summary, "npu_fast_gelu")

    report = generate_report.build_acceptance_report(meta, run_summary)
    markdown = generate_report.report_to_markdown(report)

    assert report["final_conclusion"] == "fully_validated"
    assert report["audit_summary"]["review_status"] == "ready_for_review"
    assert report["audit_summary"]["blocking_gap_count"] == 0
    assert report["audit_summary"]["manual_review_count"] == 0
    assert any(item["area"] == "memory" for item in report["advisory_items"])
    assert not report["report_lint"]
    manual_items = {(item["section"], item["item"]) for item in report["manual_review_items"]}
    assert ("资料验证", "接口描述是否详细准确:") not in manual_items
    assert ("资料验证", "输出尺寸和input是否一样:") not in manual_items
    assert ("功能验证", "输入是否支持广播") not in manual_items
    assert ("功能验证", "若多Tensor输入，是否支持各Tensor数据类型不一致") not in manual_items
    assert ("功能验证", "反向是否是单算子实现") not in manual_items
    assert ("安全编码检视", "指针是否未判空") not in manual_items
    assert "# npu_fast_gelu 验收测试报告" in markdown
    assert "## 审核摘要" in markdown
    assert "## 流程门禁" in markdown
    assert "## Shape 用例设计" in markdown
    assert "真实模型中常见 shape" in markdown
    assert "## Dtype 支持矩阵" in markdown
    assert "MS 支持 | float16, float32, bfloat16" in markdown
    assert "功能用例覆盖 | float16, float32, bfloat16" in markdown
    assert "Acceptance Report" not in markdown
    assert "Audit Summary" not in markdown
    assert "e2e ms" not in markdown
    assert "Remaining gaps are the rows marked" not in markdown
    assert "| shape_0d_fp16 | 10.00 | 20.00 | 0.90 | 1.00 | 0.0 | 0.0 | passed | note |" in markdown


def test_generate_report_requires_operator_contract_and_case_plan_before_review_ready():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_fast_gelu",
        "ms_api": "custom_ops.ops.npu_fast_gelu",
        "pta_api": "torch_npu.npu_fast_gelu",
        "pta_baseline_api": "torch_npu.npu_fast_gelu",
        "has_pta": True,
        "has_backward": True,
        "inputs": [{"name": "self", "dtype": "Tensor", "default": None}],
        "attributes": [],
        "outputs": [{"name": "out", "dtype": "Tensor"}],
    }
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pta_compare_enabled": True,
        "pytest": {
            "status": "passed",
            "pytest": {"counts": {"passed": 14}, "clean_pass": True},
            "case_results": _shape_case_results(),
        },
        "profiler": {"status": "passed", "pytest": {"counts": {"passed": 2}, "clean_pass": True}},
        "kernel_consistency": {"status": "passed"},
        "dtype_support": {
            "status": "passed",
            "pta_supported": ["float16", "float32", "bfloat16"],
            "ms_supported": ["float16", "float32", "bfloat16"],
            "missing_in_ms": [],
        },
        "case_coverage": {
            "case_count": 14,
            "dtypes": ["float16", "float32"],
            "checklist_tags": sorted(
                _shape_rule_tags() | {"default_parameters", "empty_tensor_input", "inf_and_nan", "value_range"}
            ),
            "has_multiple_dtypes": True,
            "has_special_values": True,
            "has_error_cases": True,
            "error_case_count": 1,
            "shape_design": _shape_design_template("npu_fast_gelu"),
        },
        "doc_audit": {
            "has_operator_doc": True,
            "has_chinese_doc": True,
            "has_api_description": True,
            "has_formula": True,
            "has_platforms": True,
            "has_example": True,
            "has_example_output": True,
            "has_test_command": True,
            "input_description_complete": True,
            "output_description_complete": True,
            "raises_description_complete": True,
            "format_ok": True,
        },
        "source_audit": {
            "output_shape_rule": "same_shape",
            "output_dtype_rule": "same_dtype",
            "output_shape_depends_on_runtime_data": False,
            "backward_single_op": True,
            "backward_needs_input_grad_guard": True,
            "set_unused_inputs_required": False,
            "safety": {"status": "passed", "checked_files": ["npu_fast_gelu.cpp"], "findings": []},
        },
        "legacy_test_audit": {
            "status": "passed",
            "matched_tests": ["tests/legacy/ms_functions/test_fast_gelu.py"],
            "pytest": {"counts": {"passed": 7, "failed": 0, "skipped": 0, "errors": 0}, "clean_pass": True},
        },
        "backoff_validation": {"status": "passed", "env": {"MS_DISABLE_KERNEL_BACKOFF": "1"}},
        "profile_summary": {
            "status": "passed",
            "kernel_status": "passed",
            "memory_status": "passed",
            "cases": [
                {
                    "case_id": "shape_0d_fp16",
                    "ms_end_to_end_ms": 0.01,
                    "pta_end_to_end_ms": 0.02,
                    "ms_kernel_total_us": 0.9,
                    "pta_kernel_total_us": 1.0,
                    "ms_peak_memory_mb": 0.002,
                    "pta_peak_memory_mb": 0.001,
                    "kernel_consistency": {"status": "passed"},
                    "memory_comparison": {"status": "passed", "ratio": 1.0},
                }
            ],
        },
        "ai_review": {
            "status": "passed",
            "summary": "AI 已审视报告、用例覆盖、静态证据和运行结果。",
            "reviewer_value": "报告包含接口语义、覆盖范围、dtype 对齐、kernel/显存证据。",
            "remaining_issues": [],
        },
    }

    report = generate_report.build_acceptance_report(meta, run_summary)

    assert report["final_conclusion"] == "validated_with_coverage_gaps"
    assert report["audit_summary"]["review_status"] == "needs_agent_evidence"
    assert any(item["section"] == "流程门禁" for item in report["agent_evidence_requests"])
    assert any(gap["area"] == "acceptance_artifacts" for gap in report["blocking_gaps"])


def test_generate_report_requires_agent_confirmed_semantic_artifacts():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_fast_gelu",
        "ms_api": "custom_ops.ops.npu_fast_gelu",
        "has_pta": True,
    }
    contract = _operator_contract_template("npu_fast_gelu")
    contract["analysis_status"] = "script_confirmed"
    case_plan = _case_plan_template("npu_fast_gelu")
    case_plan["analysis_status"] = "static_confirmed"
    run_summary = {
        "operator_contract": contract,
        "case_plan": case_plan,
        "pytest": {"case_results": _shape_case_results()},
        "profile_summary": {"cases": [{"case_id": "shape_0d_fp16"}]},
    }

    summary = generate_report._acceptance_artifacts_summary(meta, run_summary)

    assert summary["status"] == "coverage_gap"
    assert any("operator_contract" in issue and "agent_confirmed" in issue for issue in summary["issues"])
    assert any("case_plan" in issue and "agent_confirmed" in issue for issue in summary["issues"])


def test_generate_report_requires_cases_generated_with_confirmed_artifacts():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_fast_gelu",
        "ms_api": "custom_ops.ops.npu_fast_gelu",
        "has_pta": True,
    }
    run_summary = _with_acceptance_artifacts(
        {
            "pytest": {"case_results": _shape_case_results()},
            "profile_summary": {"cases": [{"case_id": "shape_0d_fp16"}]},
            "case_coverage": {
                "artifact_inputs": {
                    "operator_contract": {"provided": False},
                    "case_plan": {"provided": False},
                },
                "plan_alignment": {
                    "case_plan_provided": False,
                    "missing_generated_case_ids": [],
                    "extra_generated_case_ids": [],
                },
            },
        },
        "npu_fast_gelu",
    )
    run_summary["case_coverage"]["artifact_inputs"]["operator_contract"]["provided"] = False
    run_summary["case_coverage"]["artifact_inputs"]["case_plan"]["provided"] = False

    summary = generate_report._acceptance_artifacts_summary(meta, run_summary)

    assert summary["status"] == "coverage_gap"
    assert any("artifact_inputs.operator_contract.provided" in issue for issue in summary["issues"])
    assert any("artifact_inputs.case_plan.provided" in issue for issue in summary["issues"])


def test_generate_report_requires_plan_alignment_without_missing_cases():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_fast_gelu",
        "ms_api": "custom_ops.ops.npu_fast_gelu",
        "has_pta": True,
    }
    run_summary = _with_acceptance_artifacts(
        {
            "pytest": {"case_results": _shape_case_results()},
            "profile_summary": {"cases": [{"case_id": "shape_0d_fp16"}]},
            "case_coverage": {},
        },
        "npu_fast_gelu",
    )
    run_summary["case_coverage"]["plan_alignment"]["missing_generated_case_ids"] = ["missing_planned_case"]

    summary = generate_report._acceptance_artifacts_summary(meta, run_summary)

    assert summary["status"] == "coverage_gap"
    assert any("missing_planned_case" in issue for issue in summary["issues"])


def test_generate_report_rejects_shape_design_mismatch_between_plan_and_coverage():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_fast_gelu",
        "ms_api": "custom_ops.ops.npu_fast_gelu",
        "has_pta": True,
    }
    run_summary = _with_acceptance_artifacts(
        {
            "pytest": {"case_results": _shape_case_results()},
            "profile_summary": {"cases": [{"case_id": "shape_0d_fp16"}]},
            "case_coverage": {
                "case_count": 5,
                "has_multiple_dtypes": True,
                "has_special_values": True,
                "checklist_tags": sorted(_shape_rule_tags()),
                "shape_design": _shape_design_template("npu_fast_gelu"),
            },
        },
        "npu_fast_gelu",
    )
    run_summary["case_coverage"]["shape_design"]["输出 shape 规则"] = "伪造的不一致输出 shape 规则"

    summary = generate_report._acceptance_artifacts_summary(meta, run_summary)

    assert summary["status"] == "coverage_gap"
    assert any("case_coverage.shape_design" in issue for issue in summary["issues"])


def test_generate_report_requires_recorded_artifact_operator_name_match():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_fast_gelu",
        "ms_api": "custom_ops.ops.npu_fast_gelu",
        "has_pta": True,
    }
    run_summary = _with_acceptance_artifacts(
        {
            "pytest": {"case_results": _shape_case_results()},
            "profile_summary": {"cases": [{"case_id": "shape_0d_fp16"}]},
        },
        "npu_fast_gelu",
    )
    artifact = run_summary["case_coverage"]["artifact_inputs"]["operator_contract"]
    artifact.pop("operator_name_matches")
    artifact["operator_name"] = "other_op"

    summary = generate_report._acceptance_artifacts_summary(meta, run_summary)

    assert summary["status"] == "coverage_gap"
    assert any("operator_name_matches" in issue for issue in summary["issues"])
    assert any("operator_name" in issue and "other_op" in issue for issue in summary["issues"])


def test_generate_report_preserves_negative_safety_question_semantics():
    generate_report = _load_script("generate_report")

    result = generate_report._evaluate_item(
        "安全编码检视",
        {"item": "是否存在除零"},
        {},
        {
            "source_audit": {
                "safety": {
                    "status": "passed",
                    "checked_files": ["custom_ops/ms_adapter/pynative/npu_fast_gelu.cpp"],
                    "findings": [],
                }
            }
        },
    )

    assert result["自测结果"] == "否"
    assert "未发现" in result["备注"]
    assert "静态安全扫描通过" in result["备注"]


def test_generate_report_treats_safety_findings_as_unresolved():
    generate_report = _load_script("generate_report")

    result = generate_report._evaluate_item(
        "安全编码检视",
        {"item": "是否存在除零"},
        {},
        {"source_audit": {"safety": {"status": "failed", "findings": ["是否存在除零: denominator unchecked"]}}},
    )
    result["item"] = "是否存在除零"

    requests = generate_report._agent_evidence_requests({"安全编码检视": [result]}, {})

    assert result["自测结果"] == "是"
    assert not generate_report._checklist_item_passed("安全编码检视", result)
    assert requests


def test_write_report_loads_adjacent_operator_contract_and_case_plan(tmp_path):
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_fast_gelu",
        "ms_api": "custom_ops.ops.npu_fast_gelu",
        "pta_api": "torch_npu.npu_fast_gelu",
        "pta_baseline_api": "torch_npu.npu_fast_gelu",
        "has_pta": True,
        "has_backward": True,
        "inputs": [{"name": "self", "dtype": "Tensor", "default": None}],
        "attributes": [],
        "outputs": [{"name": "out", "dtype": "Tensor"}],
    }
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pta_compare_enabled": True,
        "pytest": {
            "status": "passed",
            "pytest": {"counts": {"passed": 14}, "clean_pass": True},
            "case_results": _shape_case_results(),
        },
        "profiler": {"status": "passed", "pytest": {"counts": {"passed": 2}, "clean_pass": True}},
        "kernel_consistency": {"status": "passed"},
        "dtype_support": {
            "status": "passed",
            "pta_supported": ["float16", "float32", "bfloat16"],
            "ms_supported": ["float16", "float32", "bfloat16"],
            "missing_in_ms": [],
        },
        "case_coverage": {
            "case_count": 14,
            "dtypes": ["float16", "float32"],
            "checklist_tags": sorted(
                _shape_rule_tags() | {"default_parameters", "empty_tensor_input", "inf_and_nan", "value_range"}
            ),
            "has_multiple_dtypes": True,
            "has_special_values": True,
            "has_error_cases": True,
            "error_case_count": 1,
            "shape_design": _shape_design_template("npu_fast_gelu"),
        },
        "doc_audit": {
            "has_operator_doc": True,
            "has_chinese_doc": True,
            "has_api_description": True,
            "has_formula": True,
            "has_platforms": True,
            "has_example": True,
            "has_example_output": True,
            "has_test_command": True,
            "input_description_complete": True,
            "output_description_complete": True,
            "raises_description_complete": True,
            "format_ok": True,
        },
        "source_audit": {
            "output_shape_rule": "same_shape",
            "output_dtype_rule": "same_dtype",
            "output_shape_depends_on_runtime_data": False,
            "backward_single_op": True,
            "backward_needs_input_grad_guard": True,
            "set_unused_inputs_required": False,
            "safety": {"status": "passed", "checked_files": ["npu_fast_gelu.cpp"], "findings": []},
        },
        "legacy_test_audit": {
            "status": "passed",
            "matched_tests": ["tests/legacy/ms_functions/test_fast_gelu.py"],
            "pytest": {"counts": {"passed": 7, "failed": 0, "skipped": 0, "errors": 0}, "clean_pass": True},
        },
        "backoff_validation": {"status": "passed", "env": {"MS_DISABLE_KERNEL_BACKOFF": "1"}},
        "profile_summary": {
            "status": "passed",
            "kernel_status": "passed",
            "memory_status": "passed",
            "cases": [
                {
                    "case_id": "shape_0d_fp16",
                    "ms_end_to_end_ms": 0.01,
                    "pta_end_to_end_ms": 0.02,
                    "ms_kernel_total_us": 0.9,
                    "pta_kernel_total_us": 1.0,
                    "ms_peak_memory_mb": 0.002,
                    "pta_peak_memory_mb": 0.001,
                    "kernel_consistency": {"status": "passed"},
                    "memory_comparison": {"status": "passed", "ratio": 1.0},
                }
            ],
        },
        "ai_review": {
            "status": "passed",
            "summary": "AI 已审视报告、用例覆盖、静态证据和运行结果。",
            "reviewer_value": "报告包含接口语义、覆盖范围、dtype 对齐、kernel/显存证据。",
            "remaining_issues": [],
        },
    }
    run_summary["case_coverage"].update(_artifact_aligned_case_coverage("npu_fast_gelu"))
    meta_path = tmp_path / "operator_meta.json"
    summary_path = tmp_path / "run_summary.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    summary_path.write_text(json.dumps(run_summary), encoding="utf-8")
    (tmp_path / "operator_contract.json").write_text(
        json.dumps(_operator_contract_template("npu_fast_gelu")),
        encoding="utf-8",
    )
    (tmp_path / "case_plan.json").write_text(
        json.dumps(_case_plan_template("npu_fast_gelu")),
        encoding="utf-8",
    )

    paths = generate_report.write_report(meta_path, summary_path, tmp_path)
    report = json.loads(paths["json"].read_text(encoding="utf-8"))

    assert report["acceptance_artifacts"]["status"] == "passed"
    assert not any(item["section"] == "流程门禁" for item in report["agent_evidence_requests"])


def test_generate_report_requires_ai_review_before_ready_for_review():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_fast_gelu",
        "ms_api": "custom_ops.ops.npu_fast_gelu",
        "pta_api": "torch_npu.npu_fast_gelu",
        "pta_baseline_api": "torch_npu.npu_fast_gelu",
        "has_pta": True,
        "has_backward": True,
        "inputs": [{"name": "self", "dtype": "Tensor", "default": None}],
        "attributes": [],
        "outputs": [{"name": "out", "dtype": "Tensor"}],
    }
    shape_tags = _shape_rule_tags()
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pta_compare_enabled": True,
        "pytest": {
            "status": "passed",
            "pytest": {"counts": {"passed": 14}, "clean_pass": True},
            "case_results": _shape_case_results(),
        },
        "profiler": {"status": "passed", "pytest": {"counts": {"passed": 2}, "clean_pass": True}},
        "kernel_consistency": {"status": "passed"},
        "dtype_support": {
            "status": "passed",
            "pta_supported": ["float16", "float32", "bfloat16"],
            "ms_supported": ["float16", "float32", "bfloat16"],
            "missing_in_ms": [],
        },
        "case_coverage": {
            "case_count": 14,
            "dtypes": ["float16", "float32", "bfloat16"],
            "functional_dtypes": ["float16", "float32", "bfloat16"],
            "checklist_tags": sorted(
                shape_tags
                | {
                    "default_parameters",
                    "empty_tensor_input",
                    "error_cases",
                    "inf_and_nan",
                    "value_range",
                    "non_contiguous",
                    "backward",
                }
            ),
            "has_multiple_dtypes": True,
            "has_special_values": True,
            "has_value_range": True,
            "has_error_cases": True,
            "error_case_count": 1,
            "shape_design": _shape_design_template("npu_fast_gelu"),
        },
        "doc_audit": {
            "operator_doc": "docs/operators/npu_fast_gelu.md",
            "has_operator_doc": True,
            "has_chinese_doc": True,
            "has_api_description": True,
            "has_formula": True,
            "has_platforms": True,
            "has_example": True,
            "has_example_output": True,
            "has_test_command": True,
            "input_description_complete": True,
            "output_description_complete": True,
            "raises_description_complete": True,
            "format_ok": True,
        },
        "source_audit": {
            "output_shape_rule": "same_shape",
            "output_dtype_rule": "same_dtype",
            "output_shape_depends_on_runtime_data": False,
            "backward_single_op": True,
            "backward_needs_input_grad_guard": True,
            "set_unused_inputs_required": False,
            "safety": {"status": "passed", "checked_files": ["npu_fast_gelu.cpp"], "findings": []},
        },
        "legacy_test_audit": {
            "status": "passed",
            "matched_tests": ["tests/legacy/ms_functions/test_fast_gelu.py"],
            "pytest": {"counts": {"passed": 7, "failed": 0, "skipped": 0, "errors": 0}, "clean_pass": True},
        },
        "backoff_validation": {
            "status": "passed",
            "env": {"MS_DISABLE_KERNEL_BACKOFF": "1"},
            "reason": "opinfo pytest passed with kernel backoff disabled",
        },
        "profile_summary": {
            "status": "passed",
            "kernel_status": "passed",
            "memory_status": "passed",
            "cases": [
                {
                    "case_id": "shape_0d_fp16",
                    "ms_end_to_end_ms": 0.01,
                    "pta_end_to_end_ms": 0.02,
                    "ms_kernel_total_us": 0.9,
                    "pta_kernel_total_us": 1.0,
                    "ms_peak_memory_mb": 0.002,
                    "pta_peak_memory_mb": 0.001,
                    "kernel_consistency": {"status": "passed"},
                    "memory_comparison": {
                        "status": "passed",
                        "ms_peak_memory_mb": 0.002,
                        "pta_peak_memory_mb": 0.001,
                        "ratio": 1.0,
                        "threshold": 1.1,
                    },
                }
            ],
        },
    }
    run_summary = _with_acceptance_artifacts(run_summary, "npu_fast_gelu")

    report = generate_report.build_acceptance_report(meta, run_summary)

    assert report["final_conclusion"] == "fully_validated"
    assert report["audit_summary"]["review_status"] == "needs_ai_review"
    assert report["audit_summary"]["ai_review_status"] == "not_recorded"


def test_generate_report_routes_static_unknowns_to_agent_evidence_requests():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_custom_passthrough",
        "ms_api": "custom_ops.ops.npu_custom_passthrough",
        "has_pta": False,
        "has_backward": False,
        "inputs": [{"name": "x", "dtype": "Tensor", "default": None}],
        "attributes": [],
        "outputs": [{"name": "out", "dtype": "Tensor"}],
    }
    shape_tags = _shape_rule_tags()
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pytest": {"status": "passed", "pytest": {"counts": {"passed": 12}, "clean_pass": True}},
        "case_coverage": {
            "case_count": 12,
            "dtypes": ["float16", "float32"],
            "checklist_tags": sorted(shape_tags | {"default_parameters", "empty_tensor_input", "inf_and_nan"}),
            "has_multiple_dtypes": True,
            "has_special_values": True,
            "shape_design": _shape_design_template("npu_custom_passthrough"),
        },
        "doc_audit": {
            "operator_doc": "docs/operators/npu_custom_passthrough.md",
            "has_operator_doc": True,
            "has_chinese_doc": True,
            "has_api_description": True,
            "has_formula": True,
            "has_platforms": True,
            "has_example": True,
            "has_example_output": True,
            "has_test_command": True,
            "input_description_complete": True,
            "output_description_complete": True,
            "raises_description_complete": True,
            "format_ok": True,
        },
        "source_audit": {
            "source_files": ["custom_ops/ms_adapter/pynative/npu_custom_passthrough.cpp"],
            "safety": {"status": "passed", "checked_files": ["npu_custom_passthrough.cpp"], "findings": []},
        },
        "legacy_test_audit": {"status": "not_applicable", "reason": "no matching legacy tests discovered"},
        "backoff_validation": {
            "status": "passed",
            "env": {"MS_DISABLE_KERNEL_BACKOFF": "1"},
            "reason": "opinfo pytest passed with kernel backoff disabled",
        },
    }

    report = generate_report.build_acceptance_report(meta, run_summary)

    assert report["audit_summary"]["review_status"] == "needs_agent_evidence"
    assert report["audit_summary"]["agent_evidence_request_count"] > 0
    assert any(
        item["section"] == "资料验证"
        and item["item"] == "输出尺寸和input是否一样:"
        and item["action"] == "agent_collect_evidence"
        for item in report["agent_evidence_requests"]
    )


def test_generate_report_uses_agent_evidence_to_resolve_static_unknowns():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_custom_passthrough",
        "ms_api": "custom_ops.ops.npu_custom_passthrough",
        "has_pta": False,
        "has_backward": False,
        "inputs": [{"name": "x", "dtype": "Tensor", "default": None}],
        "attributes": [],
        "outputs": [{"name": "out", "dtype": "Tensor"}],
    }
    shape_tags = _shape_rule_tags()
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pytest": {"status": "passed", "pytest": {"counts": {"passed": 12}, "clean_pass": True}},
        "case_coverage": {
            "case_count": 12,
            "dtypes": ["float16", "float32"],
            "checklist_tags": sorted(
                shape_tags
                | {"default_parameters", "empty_tensor_input", "inf_and_nan", "value_range", "non_contiguous"}
            ),
            "has_multiple_dtypes": True,
            "has_special_values": True,
            "shape_design": _shape_design_template("npu_custom_passthrough"),
        },
        "doc_audit": {
            "operator_doc": "docs/operators/npu_custom_passthrough.md",
            "has_operator_doc": True,
            "has_chinese_doc": True,
            "has_api_description": True,
            "has_formula": True,
            "has_platforms": True,
            "has_example": True,
            "has_example_output": True,
            "has_test_command": True,
            "input_description_complete": True,
            "output_description_complete": True,
            "raises_description_complete": True,
            "format_ok": True,
        },
        "source_audit": {
            "source_files": ["custom_ops/ms_adapter/pynative/npu_custom_passthrough.cpp"],
            "safety": {"status": "passed", "checked_files": ["npu_custom_passthrough.cpp"], "findings": []},
        },
        "legacy_test_audit": {"status": "not_applicable", "reason": "no matching legacy tests discovered"},
        "backoff_validation": {
            "status": "passed",
            "env": {"MS_DISABLE_KERNEL_BACKOFF": "1"},
            "reason": "opinfo pytest passed with kernel backoff disabled",
        },
        "agent_evidence": {
            "资料验证::输出尺寸和input是否一样:": {
                "result": "是",
                "reason": "agent 源码审查确认 GeneralInfer 返回 same_shape(x)",
                "evidence": [
                    "custom_ops/ms_adapter/pynative/npu_custom_passthrough.cpp:42",
                    "custom_ops/op_plugin/config/npu_custom_passthrough.yaml:18",
                ],
            },
            "功能验证::输入取值范围是否有覆盖": {
                "result": "是",
                "reason": "agent 追加边界值 case 并运行通过",
                "evidence": [
                    "tests/ms_adapter/opinfo/test_unary_ops.py::"
                    "test_unary_op_reference_forward[npu_custom_passthrough]"
                ],
            },
        },
    }

    report = generate_report.build_acceptance_report(meta, run_summary)
    markdown = generate_report.report_to_markdown(report)

    shape_item = next(
        item
        for item in report["checklist_items"]["资料验证"]
        if item["item"] == "输出尺寸和input是否一样:"
    )
    assert shape_item["自测结果"] == "是"
    assert "agent 源码审查确认" in shape_item["备注"]
    assert not any(
        item["section"] == "资料验证" and item["item"] == "输出尺寸和input是否一样:"
        for item in report["agent_evidence_requests"]
    )
    assert "## Agent 待补证据" in markdown


def test_generate_report_uses_shape_design_instead_of_rank_enumeration():
    generate_report = _load_script("generate_report")

    result_without_design = generate_report._evaluate_item(
        "功能验证",
        {"item": "输入维度是否按算子 shape 规则覆盖关键维度/axis/broadcast/边界/真实模型/非法 shape"},
        {"has_pta": False},
        {"case_coverage": {"checklist_tags": sorted(_shape_rule_tags())}},
    )

    assert result_without_design["自测结果"] == "否"
    assert "缺少 shape 规则分析字段" in result_without_design["备注"]

    result_with_design = generate_report._evaluate_item(
        "功能验证",
        {"item": "输入维度是否按算子 shape 规则覆盖关键维度/axis/broadcast/边界/真实模型/非法 shape"},
        {"has_pta": False},
        {
            "pytest": {"status": "passed", "pytest": {"clean_pass": True}, "case_results": _shape_case_results()},
            "case_coverage": {
                "checklist_tags": sorted(_shape_rule_tags()),
                "shape_design": _shape_design_template("npu_fast_gelu"),
            }
        },
    )

    assert result_with_design["自测结果"] == "是"
    assert "shape_8d" not in result_with_design["备注"]
    assert "真实模型" in result_with_design["备注"]


def test_generate_report_rejects_placeholder_shape_design():
    generate_report = _load_script("generate_report")

    draft_design = _shape_design_template("npu_generic")
    draft_design.update({
        "analysis_status": "draft_needs_agent_confirmation",
        "输出 shape 规则": "由源码/文档/agent_evidence 确认后回填",
        "真实模型中常见 shape": "结合算子场景补充真实模型 shape",
        "非法 shape / 报错 shape": "待补充",
    })
    result = generate_report._evaluate_item(
        "功能验证",
        {"item": "输入维度是否按算子 shape 规则覆盖关键维度/axis/broadcast/边界/真实模型/非法 shape"},
        {"has_pta": False},
        {
            "pytest": {"status": "passed", "pytest": {"clean_pass": True}, "case_results": _shape_case_results()},
            "case_coverage": {
                "checklist_tags": sorted(_shape_rule_tags()),
                "shape_design": draft_design,
            },
        },
    )

    assert result["自测结果"] == "否"
    assert "shape 规则分析仍有占位内容" in result["备注"]


def test_generate_report_requires_shape_case_mapping_to_passed_cases():
    generate_report = _load_script("generate_report")

    run_summary = {
        "pytest": {
            "status": "passed",
            "pytest": {"clean_pass": True},
            "case_results": [
                {"case_id": "shape_0d_fp16", "outcome": "passed", "duration_s": 0.001},
                {"case_id": "empty_fp32", "outcome": "passed", "duration_s": 0.001},
            ],
        },
        "case_coverage": {
            "checklist_tags": sorted(_shape_rule_tags()),
            "shape_design": _shape_design_template("npu_fast_gelu"),
        },
    }

    result = generate_report._evaluate_item(
        "功能验证",
        {"item": "输入维度是否按算子 shape 规则覆盖关键维度/axis/broadcast/边界/真实模型/非法 shape"},
        {"has_pta": False},
        run_summary,
    )

    assert result["自测结果"] == "否"
    assert "缺少已通过的 shape 映射用例" in result["备注"]


def test_generate_report_dtype_coverage_means_ms_covers_pta_supported_dtypes():
    generate_report = _load_script("generate_report")

    meta = {"has_pta": True}
    run_summary = {
        "pta_compare_enabled": True,
        "dtype_support": {
            "status": "coverage_gap",
            "pta_supported": ["float16", "float32"],
            "ms_supported": ["float16"],
            "missing_in_ms": ["float32"],
        },
    }

    result = generate_report._evaluate_item(
        "功能验证",
        {"item": "输入支持的dtype是否全覆盖，是否支持bfloat16"},
        meta,
        run_summary,
    )

    assert result["自测结果"] == "否"
    assert "PTA 支持但 MS 未覆盖: float32" in result["备注"]


def test_generate_report_dtype_coverage_requires_functional_case_dtypes():
    generate_report = _load_script("generate_report")

    meta = {"has_pta": True}
    run_summary = {
        "pta_compare_enabled": True,
        "dtype_support": {
            "status": "passed",
            "pta_supported": ["float16", "float32", "bfloat16"],
            "ms_supported": ["float16", "float32", "bfloat16"],
            "missing_in_ms": [],
        },
        "case_coverage": {
            "functional_dtypes": ["float16", "float32"],
            "checklist_tags": ["dtype_float16", "dtype_float32", "dtype_coverage"],
        },
    }

    result = generate_report._evaluate_item(
        "功能验证",
        {"item": "输入支持的dtype是否全覆盖，是否支持bfloat16"},
        meta,
        run_summary,
    )

    assert result["自测结果"] == "否"
    assert "功能用例缺少 dtype: bfloat16" in result["备注"]


def test_generate_report_value_range_requires_finite_boundary_values():
    generate_report = _load_script("generate_report")

    result = generate_report._evaluate_item(
        "功能验证",
        {"item": "输入取值范围是否有验证"},
        {},
        {"case_coverage": {"checklist_tags": ["inf_and_nan", "value_range"], "has_special_values": True}},
    )

    assert result["自测结果"] == "否"
    assert "有限取值边界" in result["备注"]
    lint = generate_report._report_lint(
        {"case_coverage": {"checklist_tags": ["inf_and_nan", "value_range"], "has_special_values": True}}
    )
    assert any(item["rule"] == "missing_finite_value_range_cases" for item in lint)


def test_generate_report_keeps_coverage_gaps_out_of_full_validation():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_fused_infer_attention_score_op",
        "ms_api": "custom_ops.ops.npu_fused_infer_attention_score_op",
        "pta_api": "torch.ops.custom.npu_fused_infer_attention_score",
        "has_pta": True,
        "has_backward": False,
    }
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pta_compare_enabled": True,
        "pytest": {"status": "passed", "pytest": {"counts": {"passed": 4}, "clean_pass": True}},
        "profiler": {"status": "passed", "pytest": {"counts": {"passed": 1}, "clean_pass": True}},
        "kernel_consistency": {"status": "passed"},
        "case_coverage": {
            "case_count": 4,
            "dtypes": ["float16"],
            "checklist_tags": ["layout_bsh", "layout_tnd"],
            "has_multiple_dtypes": False,
            "has_special_values": False,
        },
    }

    report = generate_report.build_acceptance_report(meta, run_summary)

    assert report["final_conclusion"] == "validated_with_coverage_gaps"


def test_generate_report_marks_operator_bug_profiler_failure_as_implementation_blocked():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_interleave_rope_op",
        "ms_api": "custom_ops.ops.npu_interleave_rope_op",
        "pta_api": "torch.ops.custom.npu_interleave_rope",
        "has_pta": True,
    }
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pta_compare_enabled": True,
        "pytest": {"status": "passed", "pytest": {"counts": {"passed": 4}, "clean_pass": True}},
        "profiler": {"status": "failed", "reason": "numerical mismatch"},
        "case_failure_analysis": [
            {"case_id": "fp32_4d", "classification": "operator_bug", "reason": "numerical mismatch"}
        ],
    }

    report = generate_report.build_acceptance_report(meta, run_summary)

    assert report["final_conclusion"] == "blocked_by_operator_implementation"
    assert any(
        gap["area"] == "case_failure_analysis" and gap["status"] == "operator_bug"
        for gap in report["skip_block_gap_details"]
    )


def test_generate_report_does_not_count_skipped_pytest_as_function_pass():
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_fast_gelu_op",
        "ms_api": "custom_ops.ops.npu_fast_gelu_op",
        "pta_api": None,
        "has_pta": False,
        "has_backward": False,
    }
    run_summary = {
        "preflight": {"is_ascend_ready": True, "missing": []},
        "pytest": {
            "status": "passed_with_skips",
            "pytest": {
                "counts": {"passed": 5, "failed": 0, "skipped": 2, "errors": 0},
                "clean_pass": False,
                "skip_reasons": [{"reason": "Need baseline data from torch_npu"}],
            },
        },
        "profiler": {"status": "not_run", "reason": "not_applicable"},
    }

    report = generate_report.build_acceptance_report(meta, run_summary)

    function_key = "\u529f\u80fd\u9a8c\u8bc1"
    result_key = "\u81ea\u6d4b\u7ed3\u679c"
    no = "\u5426"

    assert report["final_conclusion"] == "validated_with_coverage_gaps"
    assert report["checklist"][function_key][result_key] == no
    assert report["skip_block_gap_details"][0]["status"] == "passed_with_skips"


def test_ms_adapter_case_rows_do_not_claim_per_case_pass_without_case_results():
    generate_report = _load_script("generate_report")

    run_summary = {
        "opinfo_framework": {
            "framework": "ms_adapter_opinfo",
            "status": "ready",
            "opinfo_name": "demo_op",
        },
        "pytest": {
            "status": "passed",
            "pytest": {
                "counts": {"passed": 1, "failed": 0, "skipped": 0, "errors": 0},
                "clean_pass": True,
            },
        },
        "case_coverage": {
            "cases": [
                {
                    "case_id": "shape_2d_fp16",
                    "shape": [2, 3],
                    "dtype": "float16",
                    "checklist_tags": ["shape_coverage"],
                }
            ]
        },
        "generated_files": {
            "opinfo_case": "tests/ms_adapter/opinfo/test_unary_ops.py",
        },
    }

    rows = generate_report._case_detail_rows(run_summary)

    assert "opinfo_passed" not in rows
    assert "opinfo_driver_passed" in rows
    assert "sample_coverage_review_required" in rows


def test_case_plan_allows_empty_optional_case_lists_when_reason_is_recorded():
    generate_report = _load_script("generate_report")

    meta = {"operator_name": "demo_op"}
    shape_design = _shape_design_template("demo_op")
    shape_design["analysis_status"] = "agent_confirmed"
    case_plan = _case_plan_template("demo_op")
    case_plan["shape_design"] = shape_design
    case_plan["negative_cases"] = []
    case_plan["negative_cases_reason"] = "无非法 shape 约束"
    case_plan["profiler_cases"] = []
    case_plan["profiler_cases_reason"] = "无 PTA 基线，不执行 profiler"
    run_summary = {
        "operator_name": "demo_op",
        "pytest": {"status": "passed", "pytest": {"clean_pass": True}},
        "case_coverage": {
            "shape_design": shape_design,
            "artifact_inputs": {
                "operator_contract": {
                    "provided": True,
                    "analysis_status": "agent_confirmed",
                    "operator_name": "demo_op",
                    "operator_name_matches": True,
                },
                "case_plan": {
                    "provided": True,
                    "analysis_status": "agent_confirmed",
                    "operator_name": "demo_op",
                    "operator_name_matches": True,
                },
            },
            "plan_alignment": {
                "case_plan_provided": True,
                "missing_generated_case_ids": [],
            },
        },
        "opinfo_framework": {
            "framework": "ms_adapter_opinfo",
            "status": "ready",
            "opinfo_name": "demo_op",
        },
        "case_plan": case_plan,
    }

    issues = generate_report._case_plan_issues(meta, run_summary, case_plan)

    assert "case_plan 缺少字段: profiler_cases, negative_cases" not in issues


def test_case_plan_rejects_empty_optional_case_lists_with_placeholder_reason():
    generate_report = _load_script("generate_report")

    meta = {"operator_name": "demo_op"}
    case_plan = _case_plan_template("demo_op")
    case_plan["negative_cases"] = []
    case_plan["negative_cases_reason"] = "待确认：agent 需结合源码确认"
    run_summary = {"pytest": {"status": "passed", "pytest": {"clean_pass": True}}}

    issues = generate_report._case_plan_issues(meta, run_summary, case_plan)

    assert any("negative_cases" in issue for issue in issues)


def test_generate_report_marks_environment_blocked_without_runtime(tmp_path):
    generate_report = _load_script("generate_report")

    meta = {
        "operator_name": "npu_interleave_rope_op",
        "ms_api": "custom_ops.ops.npu_interleave_rope_op",
        "pta_api": "torch.ops.custom.npu_interleave_rope",
        "has_pta": True,
        "has_backward": False,
    }
    run_summary = {
        "preflight": {"is_ascend_ready": False, "missing": ["mindspore", "torch_npu"]},
        "pytest": {"status": "not_run", "reason": "environment_blocked"},
        "profiler": {"status": "not_run", "reason": "environment_blocked"},
    }

    report = generate_report.build_acceptance_report(meta, run_summary)

    function_key = "\u529f\u80fd\u9a8c\u8bc1"
    performance_key = "\u6027\u80fd\u9a8c\u8bc1"
    result_key = "\u81ea\u6d4b\u7ed3\u679c"
    note_key = "\u5907\u6ce8"
    no = "\u5426"

    assert report["final_conclusion"] == "blocked_by_operator_environment"
    # preflight records the missing environment components
    assert "mindspore" in report["preflight"]["missing"]
    assert "torch_npu" in report["preflight"]["missing"]
    # section summaries should mark function and performance as not passed
    assert report["checklist"][function_key][result_key] == no
    assert report["checklist"][performance_key][result_key] == no
    # per-item checklist exists and marks items appropriately
    items = report.get("checklist_items", {})
    assert function_key in items
    assert any(i["item"] for i in items[function_key])
