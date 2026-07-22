#!/usr/bin/env python3
"""Generate checklist-backed acceptance reports.

Each item in operator_acceptance_checklist.json is evaluated against available
evidence (operator metadata, case coverage tags, pytest results, profiler
summary, static audits, and agent-supplied evidence).  Missing evidence is
routed to agent evidence requests before a report can be ready for review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import case_ids_from_value, read_json, render_template, sanitize_identifier, write_json  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

YES = "\u662f"
NO = "\u5426"
NA = "\u4e0d\u6d89\u53ca"
MANUAL = "\u9700\u4eba\u5de5\u786e\u8ba4"
ADVISORY = "\u975e\u963b\u585e\u5907\u6ce8"
RESULT_KEY = "\u81ea\u6d4b\u7ed3\u679c"
NOTE_KEY = "\u5907\u6ce8"
AGENT_EVIDENCE_SOURCE = "agent"

_CHECKLIST_PATH = Path(__file__).resolve().parents[1] / "templates" / "operator_acceptance_checklist.json"


def _load_checklist_template() -> dict[str, Any]:
    return json.loads(_CHECKLIST_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# evidence helpers
# ---------------------------------------------------------------------------

def _case_tags(run_summary: dict[str, Any]) -> set[str]:
    coverage = run_summary.get("case_coverage", {})
    return set(coverage.get("checklist_tags", []))


def _case_declared_dtypes(case: dict[str, Any]) -> list[str]:
    values: list[str] = []
    dtype = str(case.get("dtype", "")).strip()
    if dtype:
        values.append(dtype)
    tensor_dtypes = case.get("tensor_dtypes", {})
    if isinstance(tensor_dtypes, dict):
        values.extend(str(value).strip() for value in tensor_dtypes.values() if str(value).strip())
    return values


def _functional_case_dtypes(run_summary: dict[str, Any]) -> list[str]:
    coverage = run_summary.get("case_coverage", {})
    recorded = coverage.get("functional_dtypes", [])
    if isinstance(recorded, list) and recorded:
        return sorted({str(dtype) for dtype in recorded if str(dtype).strip()})
    cases = coverage.get("cases", [])
    if isinstance(cases, list) and cases:
        values: list[str] = []
        for case in cases:
            if not isinstance(case, dict) or case.get("expect_error"):
                continue
            values.extend(_case_declared_dtypes(case))
        if values:
            return sorted(set(values))
    return sorted({str(dtype) for dtype in coverage.get("dtypes", []) if str(dtype).strip()})


def _pytest_counts(run_summary: dict[str, Any]) -> dict[str, int]:
    pytest_section = run_summary.get("pytest", {})
    detail = pytest_section.get("pytest", {}) if isinstance(pytest_section, dict) else {}
    return detail.get("counts", {})


def _function_passed(run_summary: dict[str, Any]) -> bool:
    section = run_summary.get("pytest", {})
    if not section or section.get("status") not in {"passed", "pass"}:
        return False
    detail = section.get("pytest")
    if isinstance(detail, dict):
        return bool(detail.get("clean_pass"))
    return True


def _profiler_passed(run_summary: dict[str, Any]) -> bool:
    section = run_summary.get("profiler", {})
    if not section or section.get("status") not in {"passed", "pass"}:
        return False
    detail = section.get("pytest")
    if isinstance(detail, dict):
        return bool(detail.get("clean_pass"))
    return True


def _kernel_passed(run_summary: dict[str, Any]) -> bool:
    return run_summary.get("kernel_consistency", {}).get("status") == "passed"


def _pta_on(meta: dict[str, Any], run_summary: dict[str, Any]) -> bool:
    return bool(run_summary.get("pta_compare_enabled", meta.get("has_pta", False)))


def _dtype_support(run_summary: dict[str, Any]) -> dict[str, Any]:
    support = run_summary.get("dtype_support", {})
    return support if isinstance(support, dict) else {}


def _dtype_support_note(support: dict[str, Any]) -> str:
    pta_supported = support.get("pta_supported", [])
    ms_supported = support.get("ms_supported", [])
    missing = support.get("missing_in_ms", [])
    if missing:
        return "PTA 支持但 MS 未覆盖: " + ", ".join(str(item) for item in missing)
    if support.get("status") == "passed":
        return (
            "PTA 支持 dtype: "
            + (", ".join(str(item) for item in pta_supported) or "未记录")
            + "; MS 支持: "
            + (", ".join(str(item) for item in ms_supported) or "未记录")
        )
    reason = support.get("reason") or "dtype 支持矩阵未通过"
    return str(reason)


def _dtype_support_passed(meta: dict[str, Any], run_summary: dict[str, Any]) -> bool:
    if not _pta_on(meta, run_summary):
        return True
    support = _dtype_support(run_summary)
    return bool(
        support
        and support.get("status") == "passed"
        and support.get("pta_supported")
        and not support.get("missing_in_ms")
    )


def _missing_functional_dtypes(meta: dict[str, Any], run_summary: dict[str, Any]) -> list[str]:
    if not _pta_on(meta, run_summary):
        return []
    support = _dtype_support(run_summary)
    pta_supported = [str(dtype) for dtype in support.get("pta_supported", []) if str(dtype).strip()]
    covered = set(_functional_case_dtypes(run_summary))
    return [dtype for dtype in pta_supported if dtype not in covered]


def _dtype_functional_coverage_passed(meta: dict[str, Any], run_summary: dict[str, Any]) -> bool:
    return _dtype_support_passed(meta, run_summary) and not _missing_functional_dtypes(meta, run_summary)


def _dtype_functional_note(meta: dict[str, Any], run_summary: dict[str, Any]) -> str:
    support = _dtype_support(run_summary)
    note = _dtype_support_note(support)
    missing = _missing_functional_dtypes(meta, run_summary)
    if missing:
        return note + "; 功能用例缺少 dtype: " + ", ".join(missing)
    covered = _functional_case_dtypes(run_summary)
    return note + "; 功能用例覆盖 dtype: " + (", ".join(covered) or "未记录")


def _memory_case_status(case: dict[str, Any]) -> str:
    mem = case.get("memory_comparison", {})
    status = mem.get("status") or ""
    has_ms = case.get("ms_peak_memory_mb") is not None or mem.get("ms_peak_memory_mb") is not None
    has_pta = case.get("pta_peak_memory_mb") is not None or mem.get("pta_peak_memory_mb") is not None
    if status == "not_collected" or not (has_ms and has_pta):
        return "not_collected"
    if mem.get("reason"):
        return "note"
    return status or "passed"


def _memory_missing_cases(profile_summary: dict[str, Any]) -> list[str]:
    missing = []
    for case in profile_summary.get("cases", []):
        if _memory_case_status(case) == "not_collected":
            missing.append(str(case.get("case_id", "unknown")))
    return missing


def _memory_note_cases(profile_summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        case
        for case in profile_summary.get("cases", [])
        if _memory_case_status(case) == "note"
    ]


def _memory_evidence_complete(profile_summary: dict[str, Any]) -> bool:
    if profile_summary.get("memory_status") == "coverage_gap":
        return False
    cases = profile_summary.get("cases", [])
    return bool(cases) and not _memory_missing_cases(profile_summary)


def _case_failure_classifications(run_summary: dict[str, Any]) -> set[str]:
    return {
        str(entry.get("classification", ""))
        for entry in run_summary.get("case_failure_analysis", [])
        if isinstance(entry, dict)
    }


def _doc_audit(run_summary: dict[str, Any]) -> dict[str, Any]:
    audit = run_summary.get("doc_audit", {})
    return audit if isinstance(audit, dict) else {}


def _source_audit(run_summary: dict[str, Any]) -> dict[str, Any]:
    audit = run_summary.get("source_audit", {})
    return audit if isinstance(audit, dict) else {}


def _legacy_test_audit(run_summary: dict[str, Any]) -> dict[str, Any]:
    audit = run_summary.get("legacy_test_audit", {})
    return audit if isinstance(audit, dict) else {}


def _backoff_validation(run_summary: dict[str, Any]) -> dict[str, Any]:
    audit = run_summary.get("backoff_validation", {})
    return audit if isinstance(audit, dict) else {}


def _ai_review(run_summary: dict[str, Any]) -> dict[str, Any]:
    review = run_summary.get("ai_review", {})
    return review if isinstance(review, dict) else {}


def _ai_review_passed(review: dict[str, Any]) -> bool:
    if review.get("status") != "passed":
        return False
    if not str(review.get("summary", "")).strip():
        return False
    if not str(review.get("reviewer_value", "")).strip():
        return False
    remaining = review.get("remaining_issues", [])
    return not remaining


def _required_tensor_inputs(meta: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in meta.get("inputs", [])
        if str(item.get("dtype", "")).startswith("Tensor") and str(item.get("dtype", "")) != "Tensor?"
    ]


def _has_scalar_or_attr_conversion_surface(meta: dict[str, Any]) -> bool:
    convertible_prefixes = ("Scalar", "int", "float", "double", "bool")
    for item in meta.get("attributes", []):
        dtype = str(item.get("dtype", ""))
        if dtype.startswith(convertible_prefixes):
            return True
    return len(_required_tensor_inputs(meta)) > 1


def _shape_related_tags(tags: set[str]) -> list[str]:
    shape_prefixes = ("shape_",)
    shape_tags = {
        "shape_coverage",
        "boundary_shape",
        "model_shape",
        "scalar_input",
        "empty_tensor_input",
        "non_aligned_shape",
        "broadcast",
        "dynamic_sequence_lengths",
    }
    return sorted(tag for tag in tags if tag.startswith(shape_prefixes) or tag in shape_tags)


_SHAPE_DESIGN_REQUIRED_KEYS = [
    "analysis_status",
    "算子名称",
    "输入 shape 规则",
    "输出 shape 规则",
    "关键维度含义",
    "需要覆盖的边界 shape",
    "真实模型中常见 shape",
    "非法 shape / 报错 shape",
]

_CONFIRMED_SHAPE_STATUSES = {"agent_confirmed", "script_confirmed", "static_confirmed"}

_SHAPE_PLACEHOLDER_MARKERS = (
    "待确认",
    "待补充",
    "待定",
    "TODO",
    "TBD",
    "由源码/文档/agent_evidence 确认后回填",
    "结合算子场景补充",
    "结合算子场景",
    "无 broadcast 证据",
    "需结合源码确认",
)

_CONFIRMED_SEMANTIC_ARTIFACT_STATUS = "agent_confirmed"

_CONTRACT_REQUIRED_KEYS = [
    "operator_name",
    "analysis_status",
    "ms_api",
    "inputs",
    "outputs",
    "shape_rule",
    "dtype_rule",
    "evidence",
]

_CONTRACT_SHAPE_RULE_KEYS = [
    "input_rule",
    "output_rule",
    "key_dimensions",
    "axis_or_dim",
    "supports_broadcast",
    "supports_empty_tensor",
    "supports_0d_scalar",
    "dynamic_shape_or_rank",
]

_CONTRACT_DTYPE_RULE_KEYS = ["pta_supported", "ms_expected", "dtype_branches"]

_CASE_PLAN_REQUIRED_KEYS = [
    "operator_name",
    "analysis_status",
    "shape_design",
    "dtype_plan",
    "functional_cases",
    "profiler_cases",
    "negative_cases",
    "checklist_mapping",
]


def _shape_design(run_summary: dict[str, Any]) -> dict[str, Any]:
    direct = run_summary.get("shape_design", {})
    if isinstance(direct, dict) and direct:
        return direct
    coverage = run_summary.get("case_coverage", {})
    if isinstance(coverage, dict):
        design = coverage.get("shape_design", {})
        if isinstance(design, dict):
            return design
    return {}


def _shape_design_complete(design: dict[str, Any]) -> bool:
    return not _shape_design_missing_fields(design)


def _shape_design_missing_fields(design: dict[str, Any]) -> list[str]:
    return [key for key in _SHAPE_DESIGN_REQUIRED_KEYS if not str(design.get(key, "")).strip()]


def _shape_design_placeholder_fields(design: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for key, value in design.items():
        if key == "case_mapping":
            continue
        text = str(value)
        if any(marker in text for marker in _SHAPE_PLACEHOLDER_MARKERS):
            fields.append(key)
    return fields


def _shape_case_mapping(design: dict[str, Any]) -> dict[str, list[str]]:
    raw = design.get("case_mapping", {})
    if not isinstance(raw, dict):
        return {}
    mapping: dict[str, list[str]] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, (list, tuple, set)):
            values = [str(item) for item in value if str(item).strip()]
        else:
            values = []
        mapping[str(key)] = values
    return mapping


def _shape_positive_support(value: Any) -> bool:
    text = str(value)
    if not text or any(marker in text for marker in ("不涉及", "不支持", "无 ", "无shape", "无 shape")):
        return False
    return any(marker in text for marker in ("是", "支持", "已覆盖"))


def _shape_not_applicable(value: Any) -> bool:
    text = str(value)
    return not text or any(marker in text for marker in ("不涉及", "不需要", "无", "N/A"))


def _required_shape_mapping_keys(design: dict[str, Any]) -> list[str]:
    required = ["边界 shape", "真实模型 shape"]
    axis_text = str(design.get("需要覆盖的 axis / dim", ""))
    if not _shape_not_applicable(axis_text):
        required.append("axis/dim")
    if _shape_positive_support(design.get("是否支持 broadcast", "")):
        required.append("broadcast")
    if _shape_positive_support(design.get("是否支持 empty tensor", "")):
        required.append("empty tensor")
    if _shape_positive_support(design.get("是否支持 0D scalar", "")):
        required.append("0D scalar")
    illegal_text = str(design.get("非法 shape / 报错 shape", ""))
    if not ("无非法 shape" in illegal_text and "报错" not in illegal_text and "异常" not in illegal_text):
        required.append("非法 shape / 报错 shape")
    return required


def _passed_case_ids(run_summary: dict[str, Any]) -> set[str] | None:
    case_results = run_summary.get("pytest", {}).get("case_results", [])
    if not isinstance(case_results, list) or not case_results:
        return None
    return {
        str(case.get("case_id", ""))
        for case in case_results
        if str(case.get("outcome", "")).lower() in {"passed", "pass"}
    }


def _uses_ms_adapter_opinfo(run_summary: dict[str, Any]) -> bool:
    framework = run_summary.get("opinfo_framework")
    if not isinstance(framework, dict):
        framework = run_summary.get("case_coverage", {}).get("opinfo_framework", {})
    return isinstance(framework, dict) and framework.get("framework") == "ms_adapter_opinfo"


def _ms_adapter_opinfo_passed(run_summary: dict[str, Any]) -> bool:
    framework = run_summary.get("opinfo_framework")
    if not isinstance(framework, dict):
        framework = run_summary.get("case_coverage", {}).get("opinfo_framework", {})
    if not isinstance(framework, dict) or framework.get("status") != "ready":
        return False
    return _function_passed(run_summary)


def _shape_rule_coverage_issues(run_summary: dict[str, Any]) -> list[str]:
    design = _shape_design(run_summary)
    missing_fields = _shape_design_missing_fields(design)
    if missing_fields:
        return ["缺少 shape 规则分析字段: " + ", ".join(missing_fields)]

    status = str(design.get("analysis_status", ""))
    issues: list[str] = []
    if status not in _CONFIRMED_SHAPE_STATUSES:
        issues.append(f"shape 规则分析未确认: {status}")

    placeholder_fields = _shape_design_placeholder_fields(design)
    if placeholder_fields:
        issues.append("shape 规则分析仍有占位内容: " + ", ".join(placeholder_fields))

    tags = _case_tags(run_summary)
    if not _shape_related_tags(tags):
        issues.append("缺少可证明 shape 用例覆盖的 tags")

    mapping = _shape_case_mapping(design)
    required_mapping_keys = _required_shape_mapping_keys(design)
    missing_mapping_keys = [key for key in required_mapping_keys if not mapping.get(key)]
    if missing_mapping_keys:
        issues.append("缺少 shape case 映射: " + ", ".join(missing_mapping_keys))
    else:
        passed_case_ids = _passed_case_ids(run_summary)
        if _ms_adapter_opinfo_passed(run_summary):
            pass
        elif passed_case_ids is None:
            issues.append("缺少 shape 映射用例的 pytest 单用例结果")
        else:
            missing_cases = sorted({
                case_id
                for key in required_mapping_keys
                for case_id in mapping.get(key, [])
                if case_id not in passed_case_ids
            })
            if missing_cases:
                issues.append("缺少已通过的 shape 映射用例: " + ", ".join(missing_cases))

    return issues


def _shape_rule_coverage_passed(run_summary: dict[str, Any]) -> bool:
    return not _shape_rule_coverage_issues(run_summary)


def _has_placeholder_text(value: Any) -> bool:
    return any(marker in str(value) for marker in _SHAPE_PLACEHOLDER_MARKERS)


def _missing_keys(data: dict[str, Any], keys: list[str]) -> list[str]:
    return [key for key in keys if key not in data or data.get(key) in (None, "", [])]


_OPTIONAL_EMPTY_CASE_LIST_REASON_KEYS = {
    "negative_cases": "negative_cases_reason",
    "profiler_cases": "profiler_cases_reason",
}


def _optional_empty_case_list_has_reason(case_plan: dict[str, Any], key: str) -> bool:
    reason_key = _OPTIONAL_EMPTY_CASE_LIST_REASON_KEYS.get(key)
    reason = str(case_plan.get(reason_key or "", "")).strip()
    return (
        reason_key is not None
        and isinstance(case_plan.get(key), list)
        and not case_plan.get(key)
        and bool(reason)
        and not _has_placeholder_text(reason)
    )


def _profiled_case_ids(run_summary: dict[str, Any]) -> set[str] | None:
    cases = run_summary.get("profile_summary", {}).get("cases", [])
    if not isinstance(cases, list) or not cases:
        return None
    return {str(case.get("case_id", "")) for case in cases if str(case.get("case_id", "")).strip()}


def _shape_value_for_compare(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _shape_value_for_compare(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple, set)):
        normalized = [_shape_value_for_compare(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    return str(value)


def _shape_value_matches(left: Any, right: Any) -> bool:
    return _shape_value_for_compare(left) == _shape_value_for_compare(right)


def _operator_contract_issues(meta: dict[str, Any], contract: Any) -> list[str]:
    if not isinstance(contract, dict) or not contract:
        return ["缺少 operator_contract.json 或 run_summary.operator_contract"]

    issues: list[str] = []
    missing = _missing_keys(contract, _CONTRACT_REQUIRED_KEYS)
    if missing:
        issues.append("operator_contract 缺少字段: " + ", ".join(missing))

    status = str(contract.get("analysis_status", ""))
    if status != _CONFIRMED_SEMANTIC_ARTIFACT_STATUS:
        issues.append(
            "operator_contract.analysis_status 必须是 agent_confirmed: "
            f"{status or 'not_recorded'}"
        )

    expected_name = str(meta.get("operator_name", ""))
    actual_name = str(contract.get("operator_name", ""))
    if expected_name and actual_name and actual_name != expected_name:
        issues.append(f"operator_contract 算子名不一致: {actual_name} != {expected_name}")

    shape_rule = contract.get("shape_rule", {})
    if not isinstance(shape_rule, dict):
        issues.append("operator_contract.shape_rule 不是对象")
    else:
        missing_shape = _missing_keys(shape_rule, _CONTRACT_SHAPE_RULE_KEYS)
        if missing_shape:
            issues.append("operator_contract.shape_rule 缺少字段: " + ", ".join(missing_shape))
        placeholder_shape = [
            key for key in _CONTRACT_SHAPE_RULE_KEYS if _has_placeholder_text(shape_rule.get(key, ""))
        ]
        if placeholder_shape:
            issues.append("operator_contract.shape_rule 仍有占位内容: " + ", ".join(placeholder_shape))

    dtype_rule = contract.get("dtype_rule", {})
    if not isinstance(dtype_rule, dict):
        issues.append("operator_contract.dtype_rule 不是对象")
    else:
        missing_dtype = _missing_keys(dtype_rule, _CONTRACT_DTYPE_RULE_KEYS)
        if missing_dtype:
            issues.append("operator_contract.dtype_rule 缺少字段: " + ", ".join(missing_dtype))
        placeholder_dtype = [
            key for key in _CONTRACT_DTYPE_RULE_KEYS if _has_placeholder_text(dtype_rule.get(key, ""))
        ]
        if placeholder_dtype:
            issues.append("operator_contract.dtype_rule 仍有占位内容: " + ", ".join(placeholder_dtype))

    evidence = contract.get("evidence", [])
    if not isinstance(evidence, list) or not evidence:
        issues.append("operator_contract 缺少可复核 evidence")

    return issues


def _case_plan_issues(meta: dict[str, Any], run_summary: dict[str, Any], case_plan: Any) -> list[str]:
    if not isinstance(case_plan, dict) or not case_plan:
        return ["缺少 case_plan.json 或 run_summary.case_plan"]

    issues: list[str] = []
    missing = [
        key for key in _missing_keys(case_plan, _CASE_PLAN_REQUIRED_KEYS)
        if not _optional_empty_case_list_has_reason(case_plan, key)
    ]
    if missing:
        issues.append("case_plan 缺少字段: " + ", ".join(missing))

    status = str(case_plan.get("analysis_status", ""))
    if status != _CONFIRMED_SEMANTIC_ARTIFACT_STATUS:
        issues.append(
            "case_plan.analysis_status 必须是 agent_confirmed: "
            f"{status or 'not_recorded'}"
        )

    expected_name = str(meta.get("operator_name", ""))
    actual_name = str(case_plan.get("operator_name", ""))
    if expected_name and actual_name and actual_name != expected_name:
        issues.append(f"case_plan 算子名不一致: {actual_name} != {expected_name}")

    design = case_plan.get("shape_design", {})
    if not isinstance(design, dict) or not design:
        issues.append("case_plan.shape_design 缺失或不是对象")
    else:
        missing_shape = _shape_design_missing_fields(design)
        if missing_shape:
            issues.append("case_plan.shape_design 缺少字段: " + ", ".join(missing_shape))
        design_status = str(design.get("analysis_status", ""))
        if design_status not in _CONFIRMED_SHAPE_STATUSES:
            issues.append(f"case_plan.shape_design 未确认: {design_status or 'not_recorded'}")
        placeholder_fields = _shape_design_placeholder_fields(design)
        if placeholder_fields:
            issues.append("case_plan.shape_design 仍有占位内容: " + ", ".join(placeholder_fields))
        if not _shape_case_mapping(design):
            issues.append("case_plan.shape_design 缺少 case_mapping")

    dtype_plan = case_plan.get("dtype_plan", {})
    if not isinstance(dtype_plan, dict):
        issues.append("case_plan.dtype_plan 不是对象")
    else:
        if not dtype_plan.get("pta_supported"):
            issues.append("case_plan.dtype_plan 缺少 pta_supported")
        if not dtype_plan.get("ms_expected"):
            issues.append("case_plan.dtype_plan 缺少 ms_expected")

    checklist_mapping = case_plan.get("checklist_mapping", {})
    if not isinstance(checklist_mapping, dict) or not checklist_mapping:
        issues.append("case_plan.checklist_mapping 缺失或为空")

    opinfo_case_ids = set()
    for key in ("functional_cases", "negative_cases", "checklist_mapping"):
        opinfo_case_ids.update(case_ids_from_value(case_plan.get(key, [])))
    if isinstance(design, dict):
        opinfo_case_ids.update(case_ids_from_value(_shape_case_mapping(design)))
    if opinfo_case_ids:
        passed = _passed_case_ids(run_summary)
        if _ms_adapter_opinfo_passed(run_summary):
            pass
        elif passed is None:
            issues.append("case_plan 缺少 pytest 单用例结果用于校验映射 case")
        else:
            missing_passed = sorted(case_id for case_id in opinfo_case_ids if case_id not in passed)
            if missing_passed:
                issues.append("case_plan 映射 case 未通过 pytest: " + ", ".join(missing_passed))

    profiler_case_ids = case_ids_from_value(case_plan.get("profiler_cases", []))
    if profiler_case_ids:
        profiled = _profiled_case_ids(run_summary)
        if profiled is None:
            issues.append("case_plan 缺少 profiler 逐 case 结果用于校验 profiler_cases")
        else:
            missing_profiled = sorted(case_id for case_id in profiler_case_ids if case_id not in profiled)
            if missing_profiled:
                issues.append("case_plan.profiler_cases 未出现在 profiler 结果: " + ", ".join(missing_profiled))

    return issues


def _shape_design_alignment_issues(run_summary: dict[str, Any], case_plan: Any) -> list[str]:
    if not isinstance(case_plan, dict):
        return []
    plan_design = case_plan.get("shape_design", {})
    coverage = run_summary.get("case_coverage", {})
    coverage_design = coverage.get("shape_design", {}) if isinstance(coverage, dict) else {}
    if not isinstance(plan_design, dict) or not isinstance(coverage_design, dict):
        return []

    mismatched_keys = sorted(
        key
        for key in set(plan_design) | set(coverage_design)
        if not _shape_value_matches(plan_design.get(key, ""), coverage_design.get(key, ""))
    )
    if not mismatched_keys:
        return []
    return [
        "case_coverage.shape_design 与 case_plan.shape_design 不一致: "
        + ", ".join(str(key) for key in mismatched_keys)
    ]


def _case_generation_artifact_issues(meta: dict[str, Any], run_summary: dict[str, Any]) -> list[str]:
    coverage = run_summary.get("case_coverage", {})
    if not isinstance(coverage, dict) or not coverage:
        return ["缺少 case_coverage，无法证明用例生成使用已确认契约和计划"]

    issues: list[str] = []
    expected_name = str(meta.get("operator_name") or run_summary.get("operator_name", ""))
    artifact_inputs = coverage.get("artifact_inputs")
    if not isinstance(artifact_inputs, dict):
        issues.append("case_coverage 缺少 artifact_inputs，无法证明生成阶段使用已确认契约和计划")
    else:
        for artifact_name in ("operator_contract", "case_plan"):
            entry = artifact_inputs.get(artifact_name)
            if not isinstance(entry, dict):
                issues.append(f"case_coverage.artifact_inputs.{artifact_name} 缺失")
                continue
            if entry.get("provided") is not True:
                issues.append(f"case_coverage.artifact_inputs.{artifact_name}.provided 必须是 true")
            status = str(entry.get("analysis_status", ""))
            if status != _CONFIRMED_SEMANTIC_ARTIFACT_STATUS:
                issues.append(
                    f"case_coverage.artifact_inputs.{artifact_name}.analysis_status 必须是 agent_confirmed: "
                    f"{status or 'not_recorded'}"
                )
            actual_name = str(entry.get("operator_name", ""))
            if expected_name and actual_name != expected_name:
                issues.append(
                    f"case_coverage.artifact_inputs.{artifact_name}.operator_name 必须是 {expected_name}: "
                    f"{actual_name or 'not_recorded'}"
                )
            if entry.get("operator_name_matches") is not True:
                issues.append(f"case_coverage.artifact_inputs.{artifact_name}.operator_name_matches 必须是 true")

    alignment = coverage.get("plan_alignment")
    if not isinstance(alignment, dict):
        issues.append("case_coverage 缺少 plan_alignment，无法证明生成 case 与 case_plan 对齐")
    else:
        if alignment.get("case_plan_provided") is not True:
            issues.append("case_coverage.plan_alignment.case_plan_provided 必须是 true")
        missing_cases = alignment.get("missing_generated_case_ids", [])
        if missing_cases:
            issues.append(
                "case_coverage.plan_alignment.missing_generated_case_ids 非空: "
                + ", ".join(str(case_id) for case_id in missing_cases)
            )

    return issues


def _acceptance_artifact_issues(meta: dict[str, Any], run_summary: dict[str, Any]) -> list[str]:
    issues = []
    issues.extend(_operator_contract_issues(meta, run_summary.get("operator_contract")))
    issues.extend(_case_plan_issues(meta, run_summary, run_summary.get("case_plan")))
    issues.extend(_shape_design_alignment_issues(run_summary, run_summary.get("case_plan")))
    issues.extend(_case_generation_artifact_issues(meta, run_summary))
    return issues


def _acceptance_artifacts_summary(meta: dict[str, Any], run_summary: dict[str, Any]) -> dict[str, Any]:
    issues = _acceptance_artifact_issues(meta, run_summary)
    return {
        "status": "passed" if not issues else "coverage_gap",
        "issues": issues,
        "has_operator_contract": isinstance(run_summary.get("operator_contract"), dict),
        "has_case_plan": isinstance(run_summary.get("case_plan"), dict),
    }


def _doc_note(audit: dict[str, Any], default: str) -> str:
    path = audit.get("operator_doc")
    if path:
        return f"{default}: {path}"
    return default


def _yes_if_doc(audit: dict[str, Any], key: str, yes_note: str, no_note: str) -> dict[str, Any]:
    if audit.get(key):
        return {RESULT_KEY: YES, NOTE_KEY: _doc_note(audit, yes_note)}
    return {RESULT_KEY: NO, NOTE_KEY: no_note}


def _agent_evidence_key(section_name: str, item_text: str) -> str:
    return f"{section_name}::{item_text}"


def _agent_evidence_map(run_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw = run_summary.get("agent_evidence", {})
    entries: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for key, payload in raw.items():
            if isinstance(payload, dict):
                entries[str(key)] = payload
        return entries
    if isinstance(raw, list):
        for payload in raw:
            if not isinstance(payload, dict):
                continue
            section = payload.get("section")
            item = payload.get("item")
            if section and item:
                entries[_agent_evidence_key(str(section), str(item))] = payload
    return entries


def _agent_evidence_override(
    section_name: str,
    item_text: str,
    run_summary: dict[str, Any],
) -> dict[str, Any] | None:
    payload = _agent_evidence_map(run_summary).get(_agent_evidence_key(section_name, item_text))
    if not payload:
        return None
    result = payload.get("result", payload.get(RESULT_KEY))
    if result not in {YES, NO, NA, ADVISORY, MANUAL}:
        return None
    note = str(payload.get("reason") or payload.get(NOTE_KEY) or "agent evidence supplied")
    evidence = payload.get("evidence", payload.get("sources", []))
    if isinstance(evidence, str):
        evidence = [evidence]
    if isinstance(evidence, list) and evidence:
        note += "; evidence: " + ", ".join(str(item) for item in evidence)
    return {RESULT_KEY: result, NOTE_KEY: note, "evidence_source": AGENT_EVIDENCE_SOURCE}


# ---------------------------------------------------------------------------
# per-item evaluation
# ---------------------------------------------------------------------------

_SAFETY_NEGATIVE_ITEM_KEYWORDS = [
    "指针是否未判空", "指针是否先使用后校验", "数组、指针是否访存越界",
    "是否存在除零", "内存泄露", "异常、错误处理分支",
    "使用new创建对象未声明为nothrow", "未使用安全函数库",
    "数据类型转换导致数值", "冗余代码", "暴露敏感信息",
    "弱随机数生成器", "敏感信息硬编码",
]


def _safety_item_passed(entry: dict[str, Any]) -> bool:
    return entry.get(RESULT_KEY) == NO and "静态安全扫描通过" in str(entry.get(NOTE_KEY, ""))


def _checklist_item_passed(section_name: str, entry: dict[str, Any]) -> bool:
    if section_name == "安全编码检视":
        return _safety_item_passed(entry)
    result = entry.get(RESULT_KEY)
    if result in {YES, NA}:
        return True
    return False


def _evaluate_item(
    section_name: str,
    item: dict[str, Any],
    meta: dict[str, Any],
    run_summary: dict[str, Any],
) -> dict[str, Any]:
    """Return {自测结果, 备注} for a single checklist item."""
    item_text = item["item"]
    agent_result = _agent_evidence_override(section_name, item_text, run_summary)
    if agent_result:
        return agent_result
    tags = _case_tags(run_summary)
    pta = _pta_on(meta, run_summary)
    func_ok = _function_passed(run_summary)
    prof_ok = _profiler_passed(run_summary)
    doc = _doc_audit(run_summary)
    source = _source_audit(run_summary)
    tensor_inputs = _required_tensor_inputs(meta)

    # -- 资料验证 ----------------------------------------------------------
    if "新增接口列表" in item_text:
        return {RESULT_KEY: YES, NOTE_KEY: meta.get("ms_api", "")}

    if "典型场景的ut/st用例" in item_text:
        case_count = run_summary.get("case_coverage", {}).get("case_count", 0)
        if case_count > 0:
            return {RESULT_KEY: YES, NOTE_KEY: f"已生成 {case_count} 个验收用例"}
        return {RESULT_KEY: NO, NOTE_KEY: "未生成验收用例"}

    if "与torch_npu的接口是否一致" in item_text:
        if meta.get("has_pta"):
            baseline = meta.get("pta_baseline_api") or meta.get("torch_npu_reference_api") or meta.get("pta_api")
            return {RESULT_KEY: YES, NOTE_KEY: f"PTA 基线: {baseline}"}
        return {RESULT_KEY: NA, NOTE_KEY: "无 PTA 基线"}

    if "支持的平台都有填写" in item_text:
        platforms = meta.get("platform_requirements", {}).get("required_devices", [])
        if doc.get("has_platforms"):
            return {RESULT_KEY: YES, NOTE_KEY: _doc_note(doc, "文档已声明支持平台")}
        if platforms:
            return {RESULT_KEY: YES, NOTE_KEY: f"已声明: {', '.join(platforms)}"}
        return {RESULT_KEY: NO, NOTE_KEY: "未找到平台声明证据"}

    if "属性描述" in item_text:
        attrs = meta.get("attributes", [])
        if attrs:
            names = ", ".join(a["name"] for a in attrs)
            return {RESULT_KEY: YES, NOTE_KEY: f"属性: {names}"}
        return {RESULT_KEY: NA, NOTE_KEY: "算子无属性"}

    if "input描述" in item_text:
        if doc.get("input_description_complete"):
            return {RESULT_KEY: YES, NOTE_KEY: _doc_note(doc, "文档已覆盖输入描述")}
        inputs = meta.get("inputs", [])
        if inputs:
            names = ", ".join(f"{a['name']}:{a['dtype']}" for a in inputs)
            return {RESULT_KEY: YES, NOTE_KEY: f"输入: {names}"}
        return {RESULT_KEY: NO, NOTE_KEY: "未解析到输入"}

    if "输出描述" in item_text:
        if doc.get("output_description_complete"):
            return {RESULT_KEY: YES, NOTE_KEY: _doc_note(doc, "文档已覆盖输出描述")}
        outputs = meta.get("outputs", [])
        if outputs:
            names = ", ".join(f"{a['name']}:{a['dtype']}" for a in outputs)
            return {RESULT_KEY: YES, NOTE_KEY: f"输出: {names}"}
        return {RESULT_KEY: NO, NOTE_KEY: "未解析到输出"}

    if "输出尺寸和input" in item_text:
        if source.get("output_shape_rule") == "same_shape":
            return {RESULT_KEY: YES, NOTE_KEY: "YAML/C++ 推导为 same_shape，输出 shape 与输入一致"}
        if doc.get("output_shape_same_as_input"):
            return {RESULT_KEY: YES, NOTE_KEY: _doc_note(doc, "文档声明输出 shape 与输入一致")}
        return {RESULT_KEY: NO, NOTE_KEY: "未采集到输出 shape 规则证据"}

    if "Raises项" in item_text:
        return _yes_if_doc(doc, "raises_description_complete", "文档已覆盖异常/约束描述", "文档缺少异常/约束描述")

    if "接口描述是否详细准确" in item_text:
        return _yes_if_doc(doc, "has_api_description", "文档已覆盖接口功能描述", "文档缺少接口功能描述")

    if "文档summary" in item_text:
        return _yes_if_doc(doc, "has_formula", "文档 summary/功能说明已提供公式或等价语义", "文档缺少公式或等价语义说明")

    if "接口是否提供中文rst" in item_text:
        return _yes_if_doc(doc, "has_chinese_doc", "已找到中文/中文内容文档", "未找到中文文档或中文内容")

    if "资料格式" in item_text:
        return _yes_if_doc(doc, "format_ok", "文档标题、章节和表格格式检查通过", "文档格式检查未通过或未采集")

    if "样例是否有提供" in item_text and "打印" not in item_text:
        return _yes_if_doc(doc, "has_example", "文档已提供调用样例", "文档缺少调用样例")

    if "样例是否有打印结果" in item_text:
        return _yes_if_doc(doc, "has_example_output", "样例包含结果打印或已采集样例输出", "样例缺少结果打印证据")

    if "样例执行情况" in item_text:
        example_run = run_summary.get("example_run_audit", {})
        if isinstance(example_run, dict) and example_run.get("status") == "passed":
            return {RESULT_KEY: YES, NOTE_KEY: example_run.get("reason", "样例执行通过")}
        if doc.get("has_test_command") and func_ok:
            return {RESULT_KEY: YES, NOTE_KEY: "验收用例已通过；文档提供可执行测试命令"}
        return {RESULT_KEY: NO, NOTE_KEY: "未采集到样例执行通过证据"}

    # -- 功能验证 ----------------------------------------------------------
    if "默认参数场景是否验证" in item_text:
        if "default_parameters" in tags:
            default_count = sum(1 for t in tags if t.startswith("default_parameters"))
            return {RESULT_KEY: YES, NOTE_KEY: f"已覆盖: {default_count} tags"}
        return {RESULT_KEY: NO, NOTE_KEY: "未生成默认参数用例"}

    if "空Tensor输入" in item_text:
        if "empty_tensor_input" in tags:
            return {RESULT_KEY: YES, NOTE_KEY: "已覆盖空Tensor用例"}
        return {RESULT_KEY: NO, NOTE_KEY: "未生成空Tensor用例"}

    if "inf和nan" in item_text:
        if "inf_and_nan" in tags:
            return {RESULT_KEY: YES, NOTE_KEY: "已覆盖 Inf/NaN 用例"}
        return {RESULT_KEY: NO, NOTE_KEY: "未生成 Inf/NaN 用例"}

    if "算子支持数据类型是否与标杆对齐" in item_text:
        if not pta:
            return {RESULT_KEY: NA, NOTE_KEY: "无 PTA 基线，无法自动对齐 dtype 支持矩阵"}
        support = _dtype_support(run_summary)
        if not support:
            return {RESULT_KEY: NO, NOTE_KEY: "缺少 MS/PTA dtype 支持矩阵"}
        result = YES if _dtype_support_passed(meta, run_summary) else NO
        return {RESULT_KEY: result, NOTE_KEY: _dtype_support_note(support)}

    if "输入取值范围" in item_text:
        if "value_range" in tags and run_summary.get("case_coverage", {}).get("has_value_range") is True:
            return {RESULT_KEY: YES, NOTE_KEY: "已覆盖有限取值边界用例"}
        return {RESULT_KEY: NO, NOTE_KEY: "需要补充有限取值边界用例，Inf/NaN 只能证明特殊值覆盖"}

    if "输入维度是否有覆盖0D-8D" in item_text or "是否按算子 shape 规则覆盖" in item_text:
        design = _shape_design(run_summary)
        shape_tags = _shape_related_tags(tags)
        issues = _shape_rule_coverage_issues(run_summary)
        if not issues:
            boundary = design.get("需要覆盖的边界 shape", "")
            model = design.get("真实模型中常见 shape", "")
            return {
                RESULT_KEY: YES,
                NOTE_KEY: f"已按算子 shape 规则设计用例；边界: {boundary}；真实模型: {model}；tags: {shape_tags}",
            }
        return {RESULT_KEY: NO, NOTE_KEY: "；".join(issues)}

    if "输入支持的dtype是否全覆盖" in item_text:
        if not pta:
            return {RESULT_KEY: NA, NOTE_KEY: "无 PTA 基线，无法自动判定 PTA 支持 dtype"}
        support = _dtype_support(run_summary)
        if not support:
            return {RESULT_KEY: NO, NOTE_KEY: "缺少 MS/PTA dtype 支持矩阵"}
        result = YES if _dtype_functional_coverage_passed(meta, run_summary) else NO
        return {RESULT_KEY: result, NOTE_KEY: _dtype_functional_note(meta, run_summary)}

    if "隐式类型转换" in item_text:
        if not _has_scalar_or_attr_conversion_surface(meta):
            return {RESULT_KEY: NA, NOTE_KEY: "接口无多输入或 scalar/属性隐式转换面"}
        if "implicit_type_conversion" in tags:
            return {RESULT_KEY: YES, NOTE_KEY: "已覆盖隐式类型转换用例"}
        return {RESULT_KEY: NO, NOTE_KEY: "存在隐式转换面但未覆盖隐式类型转换用例"}

    if "输入是否支持广播" in item_text:
        if len(tensor_inputs) < 2:
            return {RESULT_KEY: NA, NOTE_KEY: "单 Tensor 输入算子，无输入间广播场景"}
        if "broadcast" in tags:
            return {RESULT_KEY: YES, NOTE_KEY: "已覆盖广播用例"}
        return {RESULT_KEY: NO, NOTE_KEY: "多 Tensor 输入算子未覆盖广播用例"}

    if "输入之间的约束是否有验证" in item_text:
        if len(tensor_inputs) < 2 and not meta.get("attributes"):
            return {RESULT_KEY: NA, NOTE_KEY: "单 Tensor 且无属性，无输入间约束"}
        if "input_constraints" in tags:
            return {RESULT_KEY: YES, NOTE_KEY: "已覆盖输入约束用例"}
        if "error_cases" in tags:
            return {RESULT_KEY: YES, NOTE_KEY: "已覆盖异常/约束用例"}
        return {RESULT_KEY: NO, NOTE_KEY: "未覆盖输入约束或异常约束用例"}

    if "反向是否支持" in item_text:
        if meta.get("has_backward"):
            return {RESULT_KEY: YES, NOTE_KEY: "算子声明了 backward"}
        return {RESULT_KEY: NA, NOTE_KEY: "算子未声明 backward"}

    if "反向是否是单算子实现" in item_text:
        if not meta.get("has_backward"):
            return {RESULT_KEY: NA, NOTE_KEY: "算子未声明 backward"}
        if source.get("backward_single_op") is True:
            return {RESULT_KEY: YES, NOTE_KEY: "源码/YAML 显示 backward 调用单个反向算子"}
        if source.get("backward_single_op") is False:
            return {RESULT_KEY: NO, NOTE_KEY: "backward 不是单算子实现"}
        return {RESULT_KEY: NO, NOTE_KEY: "未采集到 backward 实现证据"}

    if "正反向的精度用例" in item_text:
        if "backward" in tags:
            return {RESULT_KEY: YES, NOTE_KEY: "已生成反向用例"}
        return {RESULT_KEY: NO, NOTE_KEY: "未生成反向精度用例"}

    if "异常用例是否校验具体报错信息" in item_text:
        error_count = run_summary.get("case_coverage", {}).get("error_case_count", 0)
        if error_count > 0:
            return {RESULT_KEY: YES, NOTE_KEY: f"已生成 {error_count} 个异常用例"}
        return {RESULT_KEY: NO, NOTE_KEY: "未生成异常用例"}

    if "是否提供报错白名单" in item_text:
        if run_summary.get("case_failure_analysis"):
            whitelist = run_summary.get("error_whitelist_audit", {})
            if isinstance(whitelist, dict) and whitelist.get("status") == "passed":
                return {RESULT_KEY: YES, NOTE_KEY: whitelist.get("reason", "已提供报错白名单")}
            return {RESULT_KEY: NO, NOTE_KEY: "存在失败/异常归因但未采集报错白名单"}
        return {RESULT_KEY: NA, NOTE_KEY: "自动化用例无遗留报错，无需报错白名单"}

    if "functional用例" in item_text:
        if run_summary.get("case_coverage", {}).get("case_count", 0) > 0 and func_ok:
            return {RESULT_KEY: YES, NOTE_KEY: "验收 opinfo 用例已调用 public functional API 并通过"}
        return {RESULT_KEY: NO, NOTE_KEY: "functional 验收用例未通过或未运行"}

    if "动态shape/rank/属性" in item_text:
        if "dynamic_sequence_lengths" in tags:
            return {RESULT_KEY: YES, NOTE_KEY: "已覆盖动态序列长度"}
        if _shape_rule_coverage_passed(run_summary) and func_ok:
            attr_note = "无属性" if not meta.get("attributes") else "属性组合见 attribute_combination 用例"
            dynamic_note = _shape_design(run_summary).get("是否有动态 shape / 动态 rank", "shape 规则覆盖通过")
            return {RESULT_KEY: YES, NOTE_KEY: f"{dynamic_note}；{attr_note}"}
        return {RESULT_KEY: NO, NOTE_KEY: "动态/rank/属性覆盖证据不足"}

    if "退避功能验证" in item_text or "MS_DISABLE_KERNEL_BACKOFF" in item_text:
        backoff = _backoff_validation(run_summary)
        if backoff.get("status") == "passed":
            return {RESULT_KEY: YES, NOTE_KEY: backoff.get("reason", "MS_DISABLE_KERNEL_BACKOFF=1 验证通过")}
        return {RESULT_KEY: NO, NOTE_KEY: "未在 MS_DISABLE_KERNEL_BACKOFF=1 下完成验证"}

    if "存量接口用例" in item_text:
        legacy = _legacy_test_audit(run_summary)
        if legacy.get("status") == "passed":
            matched = legacy.get("matched_tests", [])
            return {RESULT_KEY: YES, NOTE_KEY: "存量用例通过: " + (", ".join(matched) or "已记录")}
        if legacy.get("status") == "not_applicable":
            return {RESULT_KEY: NA, NOTE_KEY: legacy.get("reason", "未发现匹配存量用例")}
        return {RESULT_KEY: NO, NOTE_KEY: legacy.get("reason", "存量用例未运行或未通过")}

    if "反向按需求导" in item_text or "bprop" in item_text:
        if len(tensor_inputs) < 2:
            return {RESULT_KEY: NA, NOTE_KEY: "单 Tensor 输入，不涉及多输入反向按需求导"}
        if source.get("backward_needs_input_grad_guard"):
            return {RESULT_KEY: YES, NOTE_KEY: "源码包含 NeedsInputGrad/按需反向保护"}
        return {RESULT_KEY: NO, NOTE_KEY: "未采集到多输入 bprop 按需反向证据"}

    if "算子输出shape是否依赖于算子的计算结果" in item_text:
        if source.get("output_shape_depends_on_runtime_data") is False:
            return {RESULT_KEY: NO, NOTE_KEY: "输出 shape 由静态 shape 推导，不依赖运行时计算结果"}
        if source.get("output_shape_depends_on_runtime_data") is True:
            return {RESULT_KEY: YES, NOTE_KEY: "输出 shape 依赖运行时计算结果，需保留对应处理证据"}
        return {RESULT_KEY: NO, NOTE_KEY: "未采集到输出 shape 依赖计算结果的证据"}

    if "非连续输入" in item_text:
        if "non_contiguous" in tags:
            return {RESULT_KEY: YES, NOTE_KEY: "已覆盖非连续输入用例"}
        return {RESULT_KEY: NO, NOTE_KEY: "未覆盖非连续输入用例"}

    if "torch_npu计算结果0偏差" in item_text:
        if pta and prof_ok:
            return {RESULT_KEY: YES, NOTE_KEY: "profiler 数值对齐通过"}
        if pta:
            return {RESULT_KEY: NO, NOTE_KEY: "profiler 未通过或未运行"}
        return {RESULT_KEY: NA, NOTE_KEY: "无 PTA 基线"}

    if "各Tensor数据类型不一致" in item_text:
        if len(tensor_inputs) < 2:
            return {RESULT_KEY: NA, NOTE_KEY: "单 Tensor 输入算子，无多 Tensor dtype 不一致场景"}
        if "mixed_dtype" in tags:
            return {RESULT_KEY: YES, NOTE_KEY: "已覆盖混合 dtype 用例"}
        return {RESULT_KEY: NO, NOTE_KEY: "多 Tensor 输入算子未覆盖混合 dtype 用例"}

    # -- 性能验证 ----------------------------------------------------------
    if "性能是否验证广播场景" in item_text:
        if len(tensor_inputs) < 2:
            return {RESULT_KEY: NA, NOTE_KEY: "单 Tensor 输入算子，无广播性能场景"}
        if "broadcast" in tags and prof_ok:
            return {RESULT_KEY: YES, NOTE_KEY: "广播场景已进入 profiler 验证"}
        return {RESULT_KEY: NO, NOTE_KEY: "广播性能场景未覆盖或 profiler 未通过"}

    if "反向显存优化" in item_text or "SetUnusedInputs" in item_text:
        if source.get("set_unused_inputs_required") is False:
            return {RESULT_KEY: NA, NOTE_KEY: "源码审查未发现需要 SetUnusedInputs 的未用反向输入"}
        if source.get("set_unused_inputs_required") is True and source.get("set_unused_inputs_present"):
            return {RESULT_KEY: YES, NOTE_KEY: "源码包含 SetUnusedInputs 处理"}
        return {RESULT_KEY: NO, NOTE_KEY: "需要显存优化但未采集到 SetUnusedInputs 证据"}

    if "性能测试是否覆盖不同规格的数据" in item_text:
        perf_cases = run_summary.get("profile_summary", {}).get("cases", [])
        if len(perf_cases) >= 3:
            return {RESULT_KEY: YES, NOTE_KEY: f"已覆盖 {len(perf_cases)} 种性能规格"}
        if len(perf_cases) > 0:
            return {RESULT_KEY: NO, NOTE_KEY: f"仅 {len(perf_cases)} 种规格，需 3+ 种"}
        return {RESULT_KEY: NO, NOTE_KEY: "性能测试未运行"}

    if "显存是否持平PTA" in item_text:
        if pta and prof_ok:
            profile_summary = run_summary.get("profile_summary", {})
            memory_status = profile_summary.get("memory_status", "")
            if memory_status in {"passed", "note"}:
                perf_cases = profile_summary.get("cases", [])
                if perf_cases:
                    missing_cases = _memory_missing_cases(profile_summary)
                    if missing_cases:
                        return {
                            RESULT_KEY: NO,
                            NOTE_KEY: "不是所有 profiler case 都有逐 case MS/PTA 显存数值: "
                            + ", ".join(missing_cases),
                        }
                    note_cases = _memory_note_cases(profile_summary)
                    mem_parts = []
                    for c in perf_cases:
                        ms_mem = c.get("ms_peak_memory_mb")
                        pta_mem = c.get("pta_peak_memory_mb")
                        if ms_mem is not None and pta_mem is not None:
                            mem_parts.append(f"{c['case_id']}: MS={ms_mem:.1f}MB PTA={pta_mem:.1f}MB")
                    if mem_parts:
                        if note_cases:
                            return {
                                RESULT_KEY: ADVISORY,
                                NOTE_KEY: (
                                    f"显存数据已完整采集；{len(note_cases)}/{len(perf_cases)} case "
                                    "超过阈值，按验收规则仅备注不阻塞自动化结论。"
                                ),
                            }
                        return {RESULT_KEY: YES, NOTE_KEY: "; ".join(mem_parts)}
                return {RESULT_KEY: NO, NOTE_KEY: "显存状态为 passed/note，但缺少逐 case MS/PTA 显存数值"}
            if memory_status == "coverage_gap":
                return {RESULT_KEY: NO, NOTE_KEY: "显存数据缺失或未完整采集"}
            return {RESULT_KEY: NO, NOTE_KEY: "profiler 已运行但未收集显存数据"}
        return {RESULT_KEY: NO, NOTE_KEY: "profiler 未运行，无法对比显存"}

    # -- 算子自测 ----------------------------------------------------------
    if "是否自测无遗留问题" in item_text:
        conclusion = _final_conclusion(meta, run_summary)
        if conclusion == "fully_validated":
            return {RESULT_KEY: YES, NOTE_KEY: "自动化关键验收项通过；人工待确认项见审核摘要"}
        return {RESULT_KEY: NO, NOTE_KEY: f"存在缺口: {conclusion}"}

    # -- 安全编码检视 (静态证据优先，缺证据时进入 agent 待补) ----------------
    for kw in _SAFETY_NEGATIVE_ITEM_KEYWORDS:
        if kw in item_text:
            safety = source.get("safety", {})
            if isinstance(safety, dict) and safety.get("status") == "passed":
                checked = safety.get("checked_files", [])
                note = "静态安全扫描通过，未发现该项问题"
                if checked:
                    note += ": " + ", ".join(str(path) for path in checked)
                return {RESULT_KEY: NO, NOTE_KEY: note}
            if isinstance(safety, dict) and safety.get("findings"):
                return {RESULT_KEY: YES, NOTE_KEY: "静态安全扫描发现问题: " + str(safety.get("findings"))}
            return {RESULT_KEY: NO, NOTE_KEY: "未采集到静态安全扫描证据"}

    return {RESULT_KEY: NO, NOTE_KEY: "未自动评估，缺少证据采集规则"}


# ---------------------------------------------------------------------------
# conclusion
# ---------------------------------------------------------------------------

def _missing_note(run_summary: dict[str, Any]) -> str:
    missing = run_summary.get("preflight", {}).get("missing", [])
    if missing:
        return "环境阻塞: " + ", ".join(str(item) for item in missing)
    reason = run_summary.get("pytest", {}).get("reason") or run_summary.get("profiler", {}).get("reason")
    return str(reason or "缺少可用验证证据")


def _has_coverage_gap(run_summary: dict[str, Any]) -> bool:
    coverage = run_summary.get("case_coverage")
    if not isinstance(coverage, dict) or not coverage:
        return True
    if coverage.get("case_count", 0) <= 0:
        return True
    if not _shape_rule_coverage_passed(run_summary):
        return True
    if coverage.get("has_multiple_dtypes") is False:
        return True
    if coverage.get("has_special_values") is False:
        return True
    if coverage.get("has_value_range") is not True:
        return True
    if _missing_functional_dtypes({"has_pta": bool(run_summary.get("pta_compare_enabled"))}, run_summary):
        return True
    return False


def _final_conclusion(meta: dict[str, Any], run_summary: dict[str, Any]) -> str:
    preflight = run_summary.get("preflight", {})
    pta = _pta_on(meta, run_summary)

    if preflight and not preflight.get("is_ascend_ready", False):
        return "blocked_by_operator_environment"
    classifications = _case_failure_classifications(run_summary)
    if "operator_bug" in classifications:
        return "blocked_by_operator_implementation"
    if run_summary.get("pytest", {}).get("status") == "failed":
        if classifications & {"case_design_error", "needs_triage"}:
            return "validated_with_coverage_gaps"
        return "blocked_by_operator_implementation"
    if run_summary.get("profiler", {}).get("status") == "failed":
        if classifications & {"case_design_error", "needs_triage"}:
            return "validated_with_coverage_gaps"
        return "blocked_by_operator_implementation"

    func_ok = _function_passed(run_summary)
    prof_ok = _profiler_passed(run_summary)
    kernel_ok = _kernel_passed(run_summary)

    if func_ok:
        if not pta:
            return "validated_with_coverage_gaps"
        if prof_ok and kernel_ok and _dtype_functional_coverage_passed(meta, run_summary):
            profile_summary = run_summary.get("profile_summary", {})
            if isinstance(profile_summary, dict) and not _memory_evidence_complete(profile_summary):
                return "validated_with_coverage_gaps"
            if _acceptance_artifact_issues(meta, run_summary):
                return "validated_with_coverage_gaps"
            if _has_coverage_gap(run_summary):
                return "validated_with_coverage_gaps"
            return "fully_validated"
    return "validated_with_coverage_gaps"


# ---------------------------------------------------------------------------
# gap analysis
# ---------------------------------------------------------------------------

def _pytest_note(section: dict[str, Any]) -> str:
    detail = section.get("pytest", {})
    counts = detail.get("counts", {}) if isinstance(detail, dict) else {}
    if not counts:
        return section.get("log") or ""
    parts = [f"{key}={counts.get(key, 0)}" for key in ("passed", "failed", "skipped", "errors")]
    return ", ".join(parts) + f"; clean_pass={detail.get('clean_pass', False)}"


def _section_passed(section: dict[str, Any] | None) -> bool:
    if not section or section.get("status") not in {"passed", "pass"}:
        return False
    detail = section.get("pytest")
    if isinstance(detail, dict):
        return bool(detail.get("clean_pass"))
    return True


def _gap_items(meta: dict[str, Any], run_summary: dict[str, Any], pta: bool) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for name in ("pytest", "profiler", "kernel_consistency"):
        section = run_summary.get(name, {})
        status = section.get("status", "not_recorded")
        if name in {"profiler", "kernel_consistency"} and not pta:
            continue
        if status in {"passed", "pass"} and _section_passed(section):
            continue
        gaps.append({"area": name, "status": status, "reason": section.get("reason") or _pytest_note(section)})
        for skip in section.get("pytest", {}).get("skip_reasons", []):
            gaps.append({"area": f"{name}:skip", "status": "skipped", "reason": skip.get("reason")})

    coverage = run_summary.get("case_coverage", {})
    if coverage and not coverage.get("has_multiple_dtypes"):
        gaps.append({"area": "coverage", "status": "gap", "reason": "生成用例覆盖的 dtype 少于两个"})
    if coverage and not coverage.get("has_special_values"):
        gaps.append({"area": "coverage", "status": "gap", "reason": "生成用例缺少 NaN/Inf 取值覆盖"})
    if coverage and coverage.get("has_value_range") is not True:
        gaps.append({"area": "coverage", "status": "gap", "reason": "生成用例缺少有限取值边界覆盖"})
    if coverage:
        shape_issues = _shape_rule_coverage_issues(run_summary)
        if shape_issues:
            gaps.append({"area": "coverage", "status": "gap", "reason": "；".join(shape_issues)})

    artifact_issues = _acceptance_artifact_issues(meta, run_summary)
    if artifact_issues:
        gaps.append({
            "area": "acceptance_artifacts",
            "status": "gap",
            "reason": "；".join(artifact_issues),
        })

    if pta:
        dtype_support = _dtype_support(run_summary)
        if not dtype_support:
            gaps.append({"area": "dtype_support", "status": "not_recorded",
                         "reason": "缺少 MS/PTA dtype 支持矩阵"})
        elif not _dtype_support_passed({"has_pta": pta}, run_summary):
            gaps.append({"area": "dtype_support", "status": dtype_support.get("status", "gap"),
                         "reason": _dtype_support_note(dtype_support)})
        else:
            missing_dtype_cases = _missing_functional_dtypes({"has_pta": pta}, run_summary)
            if missing_dtype_cases:
                gaps.append({
                    "area": "coverage",
                    "status": "gap",
                    "reason": "功能用例缺少 dtype: " + ", ".join(missing_dtype_cases),
                })

        profile_summary = run_summary.get("profile_summary", {})
        if _profiler_passed(run_summary) and isinstance(profile_summary, dict):
            missing_memory_cases = _memory_missing_cases(profile_summary)
            if missing_memory_cases or not profile_summary.get("cases"):
                reason = "缺少逐用例 MS/PTA 显存证据"
                if missing_memory_cases:
                    reason += ": " + ", ".join(missing_memory_cases)
                gaps.append({"area": "memory", "status": "not_collected", "reason": reason})

    for entry in run_summary.get("case_failure_analysis", []):
        gaps.append({
            "area": "case_failure_analysis",
            "status": entry.get("classification", "unknown"),
            "reason": entry.get("reason") or entry.get("case_id") or "",
        })

    for analysis in run_summary.get("static_test_analysis", {}).values():
        for entry in analysis.get("empty_tests", []):
            gaps.append({"area": "static_test_analysis", "status": "empty_test",
                         "reason": f"{analysis.get('path')}: {entry['name']} is empty"})
        for entry in analysis.get("placeholder_skips", []):
            gaps.append({"area": "static_test_analysis", "status": "placeholder_skip",
                         "reason": f"{analysis.get('path')}: {entry['name']} skips with {entry['reason']}"})
    return gaps


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def _load_cases_meta(opinfo_case_path: str | None) -> dict[str, dict[str, Any]]:
    """Parse CASES JSON from a generated opinfo test file, keyed by case_id."""
    if not opinfo_case_path:
        return {}
    case_path = Path(opinfo_case_path)
    if not case_path.exists():
        return {}
    text = case_path.read_text(encoding="utf-8")
    # Extract the CASES = json.loads(r'''...''') block
    start = text.find("CASES = json.loads(r'''")
    if start == -1:
        return {}
    start = text.find("'''", start) + 3
    end = text.find("''')", start)
    if end == -1:
        return {}
    try:
        cases = json.loads(text[start:end])
    except json.JSONDecodeError:
        return {}
    return {c["case_id"]: c for c in cases}


def _format_float(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "—"


def _format_ms_as_us(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 1000:.2f}"
    return "—"


def _format_s_as_us(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 1_000_000:.2f}"
    return "—"


def _case_detail_rows(run_summary: dict[str, Any]) -> str:
    """Render a per-case functional test result table."""
    case_results = run_summary.get("pytest", {}).get("case_results", [])
    cases_meta = _load_cases_meta(
        run_summary.get("generated_files", {}).get("opinfo_case")
    )
    if _uses_ms_adapter_opinfo(run_summary) and not cases_meta and not case_results:
        coverage_cases = run_summary.get("case_coverage", {}).get("cases", [])
        if not isinstance(coverage_cases, list) or not coverage_cases:
            return "| 无 | 未采集单用例结果 |"
        outcome = (
            "opinfo_driver_passed / sample_coverage_review_required"
            if _ms_adapter_opinfo_passed(run_summary)
            else "not_verified"
        )
        rows = []
        for case in coverage_cases:
            cid = str(case.get("case_id", ""))
            shape_val = case.get("shape", case.get("input_shapes", "?"))
            shape = json.dumps(shape_val, ensure_ascii=False)
            dtype = case.get("dtype", "?")
            tags = ", ".join(case.get("checklist_tags", []))
            rows.append(f"| {cid} | {shape} | {dtype} | {outcome} | — | {tags} |")
        header = "| 用例 ID | Shape | Dtype | 结果 | 耗时 | Tags |\n| --- | --- | --- | --- | --- | --- |"
        return header + "\n" + "\n".join(rows)
    if not case_results:
        return "| 无 | 未采集单用例结果 |"
    rows = []
    for cr in case_results:
        cid = cr["case_id"]
        meta = cases_meta.get(cid, {})
        shape_val = meta.get("shape")
        if shape_val is None:
            shape_val = meta.get("input_shapes")
        if shape_val is None:
            shape_val = "?"
        shape = json.dumps(shape_val)
        dtype = meta.get("dtype", "?")
        tags = ", ".join(meta.get("checklist_tags", []))
        dur = f"{cr['duration_s']:.3f}s" if cr.get("duration_s") else "—"
        outcome = cr["outcome"]
        rows.append(f"| {cid} | {shape} | {dtype} | {outcome} | {dur} | {tags} |")
    if not rows:
        return "| 无 | 未采集单用例结果 |"
    header = "| 用例 ID | Shape | Dtype | 结果 | 耗时 | Tags |\n| --- | --- | --- | --- | --- | --- |"
    return header + "\n" + "\n".join(rows)


def _perf_detail_rows(run_summary: dict[str, Any], pta: bool) -> str:
    """Render performance comparison rows. Uses profile_summary when PTA is available,
    or MS-only timing from case results otherwise."""
    if pta:
        # Full MS/PTA comparison from profiler
        cases = run_summary.get("profile_summary", {}).get("cases", [])
        if not cases:
            return (
                "| 无 | — | — | — | — | — | — |\n"
                "profiler 数据未采集"
            )
        rows = []
        for c in cases:
            ms_us = _format_ms_as_us(c.get("ms_end_to_end_ms"))
            pta_us = _format_ms_as_us(c.get("pta_end_to_end_ms"))
            ms_k = _format_float(c.get("ms_kernel_total_us"), 2)
            pta_k = _format_float(c.get("pta_kernel_total_us"), 2)
            ms_mem = _format_float(c.get("ms_peak_memory_mb"), 1)
            pta_mem = _format_float(c.get("pta_peak_memory_mb"), 1)
            k_stat = c.get("kernel_consistency", {}).get("status", "—")
            m_stat = _memory_case_status(c)
            rows.append(
                f"| {c['case_id']} | {ms_us} | {pta_us} | {ms_k} | {pta_k} | "
                f"{ms_mem} | {pta_mem} | {k_stat} | {m_stat} |"
            )
        header = (
            "| 用例 | MS 端到端耗时(us) | PTA 端到端耗时(us) | MS Kernel耗时(us) | PTA Kernel耗时(us) | "
            "MS 显存(MB) | PTA 显存(MB) | Kernel | 显存 |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"
        )
        return header + "\n" + "\n".join(rows)

    # MS-only timing (no PTA baseline)
    case_results = run_summary.get("pytest", {}).get("case_results", [])
    if not case_results:
        return (
            "| 无 | — | — |\n"
            "无 PTA 基线，未采集 MS-only 耗时"
        )
    rows = []
    for cr in case_results:
        dur = _format_s_as_us(cr.get("duration_s"))
        rows.append(f"| {cr['case_id']} | {cr['outcome']} | {dur} |")
    header = "| 用例 ID | 结果 | MS 端到端耗时(us) |\n| --- | --- | ---: |"
    return header + "\n" + "\n".join(rows)


def _memory_detail_rows(profile_summary: dict[str, Any]) -> str:
    """Render per-case memory comparison rows."""
    cases = profile_summary.get("cases", [])
    if not cases:
        return (
            "| 无 | — | — | — |\n"
            "未采集显存数据"
        )
    rows = []
    for c in cases:
        mem = c.get("memory_comparison", {})
        ms_mem = f"{mem['ms_peak_memory_mb']:.4f}" if mem.get("ms_peak_memory_mb") else "—"
        pta_mem = f"{mem['pta_peak_memory_mb']:.4f}" if mem.get("pta_peak_memory_mb") else "—"
        ratio = f"{mem['ratio']:.2f}" if mem.get("ratio") else "—"
        status = _memory_case_status(c)
        note = mem.get("reason", "") if mem.get("reason") else "✓"
        rows.append(f"| {c['case_id']} | {ms_mem} | {pta_mem} | {ratio} | {status} | {note} |")
    header = (
        "| 用例 | MS 显存(MB) | PTA 显存(MB) | 比例 | 状态 | 备注 |\n"
        "| --- | ---: | ---: | ---: | --- | --- |"
    )
    return header + "\n" + "\n".join(rows)


def build_acceptance_report(meta: dict[str, Any], run_summary: dict[str, Any]) -> dict[str, Any]:
    conclusion = _final_conclusion(meta, run_summary)
    pta = _pta_on(meta, run_summary)
    func_ok = _function_passed(run_summary)
    prof_ok = _profiler_passed(run_summary)
    kernel_ok = _kernel_passed(run_summary)
    gaps = _gap_items(meta, run_summary, pta)
    acceptance_artifacts = _acceptance_artifacts_summary(meta, run_summary)

    coverage = run_summary.get("case_coverage", {})
    shape_design = _shape_design(run_summary)
    coverage_matrix = {
        "case_count": coverage.get("case_count", 0),
        "dtypes": coverage.get("dtypes", []),
        "functional_dtypes": coverage.get("functional_dtypes", coverage.get("dtypes", [])),
        "checklist_tags": coverage.get("checklist_tags", []),
        "shapes": coverage.get("shapes", []),
        "has_value_range": coverage.get("has_value_range"),
        "shape_design": shape_design,
    }

    profile_summary = run_summary.get("profile_summary", {})
    performance_cases = profile_summary.get("cases", []) if isinstance(profile_summary, dict) else []

    # Build per-item checklist from the canonical template
    try:
        template = _load_checklist_template()
        sections_data: dict[str, list[dict[str, Any]]] = {}
        for section in template["sections"]:
            section_name = section["name"]
            items = []
            for item in section["items"]:
                evaluated = _evaluate_item(section_name, item, meta, run_summary)
                evaluated["item"] = item["item"]
                items.append(evaluated)
            sections_data[section_name] = items

        # Section-level summaries (backward-compatible with legacy tests)
        section_summaries: dict[str, dict[str, Any]] = {}
        for section_name, items in sections_data.items():
            yes_count = sum(1 for i in items if i[RESULT_KEY] == YES)
            passed_count = sum(1 for i in items if _checklist_item_passed(section_name, i))
            failed_count = sum(
                1
                for i in items
                if i[RESULT_KEY] == NO and not _checklist_item_passed(section_name, i)
            )
            na_count = sum(1 for i in items if i[RESULT_KEY] == NA)
            manual_count = sum(1 for i in items if i[RESULT_KEY] == MANUAL)
            advisory_count = sum(1 for i in items if i[RESULT_KEY] == ADVISORY)
            total = len(items)
            if total == 0:
                section_summaries[section_name] = {RESULT_KEY: NA, NOTE_KEY: "无条目"}
            elif na_count == total:
                section_summaries[section_name] = {RESULT_KEY: NA, NOTE_KEY: "全部不涉及"}
            elif failed_count == 0 and manual_count == 0 and advisory_count == 0 and passed_count == total:
                section_summaries[section_name] = {RESULT_KEY: YES, NOTE_KEY: f"{passed_count}/{total} 通过"}
            elif failed_count == 0:
                section_summaries[section_name] = {
                    RESULT_KEY: ADVISORY if advisory_count else MANUAL,
                    NOTE_KEY: (
                        f"是={yes_count} 需人工确认={manual_count} "
                        f"非阻塞备注={advisory_count} 不涉及={na_count}"
                    ),
                }
            else:
                section_summaries[section_name] = {RESULT_KEY: NO,
                    NOTE_KEY: (
                        f"通过={passed_count} 否={failed_count} 需人工确认={manual_count} "
                        f"非阻塞备注={advisory_count} 不涉及={na_count}"
                    )}
    except Exception:
        # Fallback when checklist template is unavailable
        note = _missing_note(run_summary)
        compare_skip = "无 PTA 基线，跳过 MS/PTA profiler 对比"
        perf_note = _pytest_note(run_summary.get("profiler", {})) or (compare_skip if not pta else note)
        if kernel_ok:
            perf_note += "; kernel 一致性通过"
        elif pta:
            kernel_reason = run_summary.get("kernel_consistency", {}).get("reason", "等待 profiler kernel 摘要")
            perf_note += "; kernel: " + kernel_reason
        section_summaries = {
            "资料验证": {RESULT_KEY: YES,
                NOTE_KEY: f"接口: {meta.get('ms_api')}; 配置: {meta.get('source_config', 'unknown')}"},
            "功能验证": {RESULT_KEY: YES if func_ok else NO,
                NOTE_KEY: (
                    _pytest_note(run_summary.get("pytest", {}))
                    or run_summary.get("pytest", {}).get("log")
                    or note
                )},
            "性能验证": {RESULT_KEY: YES if prof_ok else (NA if not pta else NO),
                NOTE_KEY: perf_note},
            "算子自测": {RESULT_KEY: YES if conclusion == "fully_validated" else NO, NOTE_KEY: conclusion},
            "安全编码检视": {RESULT_KEY: MANUAL, NOTE_KEY: "C++ 安全编码项需人工审查"},
        }
        sections_data = {}

    case_detail_rows = _case_detail_rows(run_summary)
    perf_detail_rows = _perf_detail_rows(run_summary, pta)
    profile_summary_data = run_summary.get("profile_summary", {})
    memory_detail_rows = _memory_detail_rows(profile_summary_data) if isinstance(profile_summary_data, dict) else ""
    dtype_support_data = _dtype_support(run_summary)
    manual_items = _manual_review_items(sections_data)
    advisory_items = _advisory_items(
        sections_data,
        profile_summary_data if isinstance(profile_summary_data, dict) else {},
    )
    agent_requests = _agent_evidence_requests(sections_data, run_summary)
    agent_requests.extend(_artifact_evidence_requests(meta, run_summary))
    ai_review = _ai_review(run_summary)
    audit_summary = _build_audit_summary(
        conclusion,
        gaps,
        manual_items,
        advisory_items,
        agent_requests,
        ai_review,
    )

    report = {
        "operator_name": meta.get("operator_name"),
        "ms_api": meta.get("ms_api"),
        "pta_api": (meta.get("pta_baseline_api") or meta.get("pta_api")) if pta else NA,
        "has_backward": meta.get("has_backward", False),
        "has_pta": meta.get("has_pta", False),
        "pta_compare_enabled": pta,
        "preflight": run_summary.get("preflight", {}),
        "checklist": section_summaries,
        "checklist_items": sections_data,
        "case_coverage": coverage_matrix,
        "shape_design": shape_design,
        "case_detail_rows": case_detail_rows,
        "perf_detail_rows": perf_detail_rows,
        "memory_detail_rows": memory_detail_rows,
        "dtype_support": dtype_support_data,
        "profile_summary": profile_summary_data if isinstance(profile_summary_data, dict) else {},
        "acceptance_artifacts": acceptance_artifacts,
        "audit_summary": audit_summary,
        "ai_review": ai_review,
        "blocking_gaps": gaps,
        "manual_review_items": manual_items,
        "agent_evidence_requests": agent_requests,
        "advisory_items": advisory_items,
        "skip_block_gap_details": gaps,
        "performance_cases": performance_cases,
        "generated_files": run_summary.get("generated_files", {}),
        "final_conclusion": conclusion,
    }
    report["report_lint"] = _report_lint(report)
    return report


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------

def _md_checklist_table(sections_data: dict[str, list[dict[str, Any]]]) -> str:
    if not sections_data:
        return ""
    parts: list[str] = []
    for section_name, items in sections_data.items():
        parts.append(f"### {section_name}")
        parts.append("")
        parts.append(f"| 自测内容 | {RESULT_KEY} | {NOTE_KEY} |")
        parts.append("| --- | --- | --- |")
        for entry in items:
            parts.append(f"| {entry.get('item', '')} | {entry.get(RESULT_KEY, NO)} | {entry.get(NOTE_KEY, '')} |")
        parts.append("")
    return "\n".join(parts)


def _coverage_rows(coverage: dict[str, Any]) -> str:
    functional_dtypes = coverage.get("functional_dtypes", coverage.get("dtypes", []))
    return "\n".join([
        f"| 用例数量 | {coverage.get('case_count', 0)} |",
        f"| dtype 覆盖 | {', '.join(functional_dtypes) or '未覆盖'} |",
        f"| checklist tags | {', '.join(coverage.get('checklist_tags', [])) or '未覆盖'} |",
    ])


def _gap_rows(gaps: list[dict[str, Any]]) -> str:
    if not gaps:
        return "| 无 | passed | 未记录 skip/block/gap |"
    return "\n".join(f"| {g['area']} | {g['status']} | {g.get('reason') or ''} |" for g in gaps)


def _performance_rows(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return (
            "| 无 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 | 未采集 |"
        )
    rows = []
    for c in cases:
        rows.append(
            "| {case_id} | {ms} | {pta} | {ms_kernel} | {pta_kernel} | {ms_mem} | {pta_mem} |".format(
                case_id=c.get("case_id", ""),
                ms=_format_ms_as_us(c.get("ms_end_to_end_ms")),
                pta=_format_ms_as_us(c.get("pta_end_to_end_ms")),
                ms_kernel=_format_float(c.get("ms_kernel_total_us"), 2),
                pta_kernel=_format_float(c.get("pta_kernel_total_us"), 2),
                ms_mem=_format_float(c.get("ms_peak_memory_mb"), 1),
                pta_mem=_format_float(c.get("pta_peak_memory_mb"), 1),
            )
        )
    return "\n".join(rows)


_EVIDENCE_GAP_MARKERS = (
    "缺少",
    "未采集",
    "未生成",
    "未覆盖",
    "未找到",
    "未发现",
    "未运行",
    "未通过或未运行",
    "证据不足",
    "无法自动",
    "not_recorded",
    "not_collected",
)


def _agent_evidence_requests(
    sections_data: dict[str, list[dict[str, Any]]],
    run_summary: dict[str, Any],
) -> list[dict[str, str]]:
    requests: list[dict[str, str]] = []
    for section_name, section_items in sections_data.items():
        for entry in section_items:
            item_text = str(entry.get("item", ""))
            key = _agent_evidence_key(section_name, item_text)
            if entry.get("evidence_source") == AGENT_EVIDENCE_SOURCE:
                continue
            result = entry.get(RESULT_KEY, "")
            note = str(entry.get(NOTE_KEY, ""))
            if section_name == "安全编码检视" and _safety_item_passed(entry):
                continue
            if section_name == "安全编码检视" and result == YES:
                requests.append({
                    "section": section_name,
                    "item": item_text,
                    "status": str(result),
                    "reason": note,
                    "action": "agent_triage_safety_finding",
                    "agent_evidence_key": key,
                })
                continue
            needs_evidence = result == MANUAL or (
                result == NO and any(marker in note for marker in _EVIDENCE_GAP_MARKERS)
            )
            if needs_evidence:
                requests.append({
                    "section": section_name,
                    "item": item_text,
                    "status": str(result),
                    "reason": note,
                    "action": "agent_collect_evidence",
                    "agent_evidence_key": key,
                })
    return requests


def _artifact_evidence_requests(meta: dict[str, Any], run_summary: dict[str, Any]) -> list[dict[str, str]]:
    requests = []
    for issue in _acceptance_artifact_issues(meta, run_summary):
        requests.append({
            "section": "流程门禁",
            "item": "operator_contract.json / case_plan.json",
            "action": "agent_generate_or_fix_artifact",
            "reason": issue,
            "agent_evidence_key": "流程门禁::operator_contract.json / case_plan.json",
        })
    return requests


def _manual_review_items(sections_data: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for section_name, section_items in sections_data.items():
        for entry in section_items:
            result = entry.get(RESULT_KEY, "")
            note = str(entry.get(NOTE_KEY, ""))
            if result == MANUAL:
                items.append({
                    "section": section_name,
                    "item": str(entry.get("item", "")),
                    "status": str(result),
                    "reason": note,
                })
    return items


def _advisory_items(
    sections_data: dict[str, list[dict[str, Any]]],
    profile_summary: dict[str, Any],
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for section_name, section_items in sections_data.items():
        for entry in section_items:
            if entry.get(RESULT_KEY) == ADVISORY:
                items.append({
                    "area": section_name,
                    "status": ADVISORY,
                    "reason": str(entry.get(NOTE_KEY, "")),
                })
    note_cases = _memory_note_cases(profile_summary)
    cases = profile_summary.get("cases", []) if isinstance(profile_summary, dict) else []
    if note_cases:
        items.append({
            "area": "memory",
            "status": "note",
            "reason": (
                f"{len(note_cases)}/{len(cases)} 个 profiler 用例超过显存比例提示阈值"
            ),
        })
    return items


def _review_status(
    conclusion: str,
    blocking_gaps: list[dict[str, Any]],
    manual_items: list[dict[str, str]],
    agent_requests: list[dict[str, str]],
    ai_review: dict[str, Any],
) -> str:
    if conclusion.startswith("blocked_"):
        return "blocked"
    if agent_requests:
        return "needs_agent_evidence"
    if blocking_gaps:
        return "review_blocked"
    if manual_items:
        return "needs_evidence_completion"
    if not _ai_review_passed(ai_review):
        return "needs_ai_review"
    return "ready_for_review"


def _build_audit_summary(
    conclusion: str,
    blocking_gaps: list[dict[str, Any]],
    manual_items: list[dict[str, str]],
    advisory_items: list[dict[str, str]],
    agent_requests: list[dict[str, str]],
    ai_review: dict[str, Any],
) -> dict[str, Any]:
    return {
        "automated_conclusion": conclusion,
        "review_status": _review_status(conclusion, blocking_gaps, manual_items, agent_requests, ai_review),
        "blocking_gap_count": len(blocking_gaps),
        "manual_review_count": len(manual_items),
        "agent_evidence_request_count": len(agent_requests),
        "ai_review_status": ai_review.get("status", "not_recorded"),
        "advisory_count": len(advisory_items),
        "interpretation": (
            "ready_for_review 要求阻塞缺口、Agent 待补证据、人工确认项均为 0，且 AI 报告审视通过；"
            "非阻塞备注会保留给审核人参考，但不作为上线阻塞。"
        ),
    }


def _report_lint(report: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if report.get("final_conclusion") == "fully_validated" and report.get("blocking_gaps"):
        findings.append({
            "level": "error",
            "rule": "fully_validated_has_blocking_gaps",
            "message": "fully_validated must not have blocking gaps",
        })
    if (
        report.get("audit_summary", {}).get("review_status") == "ready_for_review"
        and report.get("manual_review_items")
    ):
        findings.append({
            "level": "error",
            "rule": "ready_for_review_has_manual_items",
            "message": "ready_for_review must not have manual review items",
        })
    if (
        report.get("audit_summary", {}).get("review_status") == "ready_for_review"
        and report.get("agent_evidence_requests")
    ):
        findings.append({
            "level": "error",
            "rule": "ready_for_review_has_agent_evidence_requests",
            "message": "ready_for_review must not have unresolved agent evidence requests",
        })
    if (
        report.get("audit_summary", {}).get("review_status") == "ready_for_review"
        and not _ai_review_passed(report.get("ai_review", {}))
    ):
        findings.append({
            "level": "error",
            "rule": "ready_for_review_missing_ai_review",
            "message": "ready_for_review requires a passed AI review with reviewer-value summary",
        })
    if (
        report.get("audit_summary", {}).get("review_status") == "ready_for_review"
        and report.get("acceptance_artifacts", {}).get("status") != "passed"
    ):
        findings.append({
            "level": "error",
            "rule": "ready_for_review_missing_acceptance_artifacts",
            "message": "ready_for_review requires validated operator_contract and case_plan artifacts",
        })
    support = report.get("dtype_support", {})
    if report.get("pta_compare_enabled") and not support:
        findings.append({
            "level": "error",
            "rule": "missing_dtype_support_matrix",
            "message": "PTA comparison requires dtype_support evidence in the report",
        })
    missing_functional = _missing_functional_dtypes(
        {"has_pta": bool(report.get("pta_compare_enabled"))},
        report,
    )
    if missing_functional:
        findings.append({
            "level": "error",
            "rule": "missing_functional_dtype_cases",
            "message": "Functional cases are missing PTA-supported dtypes: " + ", ".join(missing_functional),
        })
    coverage = report.get("case_coverage", {})
    if isinstance(coverage, dict) and coverage and coverage.get("has_value_range") is not True:
        findings.append({
            "level": "error",
            "rule": "missing_finite_value_range_cases",
            "message": "Finite value-range cases are missing; Inf/NaN only covers special values",
        })
    profile_summary = report.get("profile_summary", {})
    if isinstance(profile_summary, dict) and _memory_note_cases(profile_summary):
        has_memory_advisory = any(item.get("area") == "memory" for item in report.get("advisory_items", []))
        if not has_memory_advisory:
            findings.append({
                "level": "error",
                "rule": "missing_memory_advisory",
                "message": "memory ratio notes must be surfaced in advisory items",
            })
    return findings


def _audit_summary_rows(summary: dict[str, Any]) -> str:
    return "\n".join([
        f"| 自动结论 | `{summary.get('automated_conclusion', '')}` |",
        f"| 审核状态 | `{summary.get('review_status', '')}` |",
        f"| 阻塞缺口 | {summary.get('blocking_gap_count', 0)} |",
        f"| Agent 待补证据 | {summary.get('agent_evidence_request_count', 0)} |",
        f"| 人工确认项 | {summary.get('manual_review_count', 0)} |",
        f"| AI 审视状态 | `{summary.get('ai_review_status', 'not_recorded')}` |",
        f"| 非阻塞备注 | {summary.get('advisory_count', 0)} |",
    ])


def _simple_items_rows(items: list[dict[str, Any]], columns: tuple[str, str, str]) -> str:
    if not items:
        return "| 无 | passed | 无 |"
    left, status, reason = columns
    rows = []
    for item in items:
        rows.append(
            "| {left_value} | {status_value} | {reason_value} |".format(
                left_value=item.get(left, item.get("area", item.get("section", ""))),
                status_value=item.get(status, ""),
                reason_value=item.get(reason, ""),
            )
        )
    return "\n".join(rows)


def _manual_review_rows(items: list[dict[str, Any]]) -> str:
    if not items:
        return "| 无 | 无 | passed | 无 |"
    return "\n".join(
        f"| {item.get('section', '')} | {item.get('item', '')} | {item.get('status', '')} | "
        f"{item.get('reason', '')} |"
        for item in items
    )


def _agent_evidence_request_rows(items: list[dict[str, Any]]) -> str:
    if not items:
        return "| 无 | 无 | passed | 无 | 无 |"
    return "\n".join(
        f"| {item.get('section', '')} | {item.get('item', '')} | {item.get('status', '')} | "
        f"{item.get('reason', '')} | {item.get('agent_evidence_key', '')} |"
        for item in items
    )


def _ai_review_rows(review: dict[str, Any]) -> str:
    if not review:
        return "\n".join([
            "| 状态 | not_recorded |",
            "| 摘要 | AI 尚未审视生成的验收报告 |",
            "| 审核价值 | not_recorded |",
            "| 剩余问题 | not_recorded |",
        ])
    remaining = review.get("remaining_issues", [])
    if isinstance(remaining, list):
        remaining_text = ", ".join(str(item) for item in remaining) or "none"
    else:
        remaining_text = str(remaining)
    return "\n".join([
        f"| 状态 | {review.get('status', 'not_recorded')} |",
        f"| 摘要 | {review.get('summary', '')} |",
        f"| 审核价值 | {review.get('reviewer_value', '')} |",
        f"| 剩余问题 | {remaining_text} |",
    ])


def _dtype_support_rows(support: dict[str, Any], coverage: dict[str, Any] | None = None) -> str:
    coverage = coverage or {}
    functional_dtypes = [
        str(dtype)
        for dtype in coverage.get("functional_dtypes", coverage.get("dtypes", []))
        if str(dtype).strip()
    ]
    if not support:
        return "\n".join([
            "| 状态 | not_recorded |",
            "| PTA 支持 | — |",
            "| MS 支持 | — |",
            "| MS 缺失 | not_recorded |",
            f"| 功能用例覆盖 | {', '.join(functional_dtypes) or '—'} |",
            "| 功能用例缺失 | not_recorded |",
        ])
    missing_functional = [
        str(dtype)
        for dtype in support.get("pta_supported", [])
        if str(dtype).strip() and str(dtype) not in functional_dtypes
    ]
    return "\n".join([
        f"| 状态 | {support.get('status', 'not_recorded')} |",
        f"| PTA 支持 | {', '.join(str(item) for item in support.get('pta_supported', [])) or '—'} |",
        f"| MS 支持 | {', '.join(str(item) for item in support.get('ms_supported', [])) or '—'} |",
        f"| MS 缺失 | {', '.join(str(item) for item in support.get('missing_in_ms', [])) or '无'} |",
        f"| 功能用例覆盖 | {', '.join(functional_dtypes) or '—'} |",
        f"| 功能用例缺失 | {', '.join(missing_functional) or '无'} |",
    ])


def _acceptance_artifact_rows(artifacts: dict[str, Any]) -> str:
    issues = artifacts.get("issues", [])
    return "\n".join([
        f"| 状态 | {artifacts.get('status', 'not_recorded')} |",
        f"| operator_contract.json | {'已校验' if artifacts.get('has_operator_contract') else '缺失'} |",
        f"| case_plan.json | {'已校验' if artifacts.get('has_case_plan') else '缺失'} |",
        f"| 问题 | {'；'.join(str(item) for item in issues) if issues else '无'} |",
    ])


def _shape_coverage_list(coverage: dict[str, Any]) -> str:
    shapes = coverage.get("shapes", [])
    if not shapes:
        return "- 未记录 shape 覆盖"
    return "\n".join(f"- {json.dumps(s, ensure_ascii=False)}" for s in shapes)


def _shape_design_rows(design: dict[str, Any]) -> str:
    keys = [
        "analysis_status",
        "算子名称",
        "输入 shape 规则",
        "输出 shape 规则",
        "关键维度含义",
        "需要覆盖的 axis / dim",
        "需要覆盖的边界 shape",
        "是否支持 broadcast",
        "是否支持 empty tensor",
        "是否支持 0D scalar",
        "是否有 dtype 相关分支",
        "是否有动态 shape / 动态 rank",
        "真实模型中常见 shape",
        "非法 shape / 报错 shape",
    ]
    if not design:
        return "| Shape 设计 | 未采集 |"
    rows = [f"| {key} | {design.get(key, '未采集')} |" for key in keys]
    mapping = _shape_case_mapping(design)
    if mapping:
        mapping_text = "; ".join(
            f"{key}: {', '.join(values) if values else '未映射'}"
            for key, values in mapping.items()
        )
        rows.append(f"| case_mapping | {mapping_text} |")
    return "\n".join(rows)


def report_to_markdown(report: dict[str, Any]) -> str:
    sections_data = report.get("checklist_items", {})
    if sections_data:
        checklist_md = _md_checklist_table(sections_data)
    else:
        rows = []
        for item, payload in report["checklist"].items():
            rows.append(f"| {item} | {payload[RESULT_KEY]} | {payload[NOTE_KEY]} |")
        checklist_md = "\n".join(rows)

    coverage = report.get("case_coverage", {})
    gaps = report.get("skip_block_gap_details", [])

    return render_template(
        "acceptance_report.md.template",
        {
            "operator_name": report.get("operator_name") or "",
            "ms_api": report.get("ms_api") or "",
            "pta_api": report.get("pta_api") or NA,
            "final_conclusion": report.get("final_conclusion") or "",
            "audit_summary_rows": _audit_summary_rows(report.get("audit_summary", {})),
            "audit_interpretation": report.get("audit_summary", {}).get("interpretation", ""),
            "ai_review_rows": _ai_review_rows(report.get("ai_review", {})),
            "acceptance_artifact_rows": _acceptance_artifact_rows(report.get("acceptance_artifacts", {})),
            "blocking_gap_rows": _simple_items_rows(
                report.get("blocking_gaps", []),
                ("area", "status", "reason"),
            ),
            "advisory_rows": _simple_items_rows(
                report.get("advisory_items", []),
                ("area", "status", "reason"),
            ),
            "agent_evidence_request_rows": _agent_evidence_request_rows(
                report.get("agent_evidence_requests", []),
            ),
            "manual_review_rows": _manual_review_rows(report.get("manual_review_items", [])),
            "dtype_support_rows": _dtype_support_rows(report.get("dtype_support", {}), coverage),
            "header_item": "自测内容",
            "header_result": RESULT_KEY,
            "header_note": NOTE_KEY,
            "checklist_rows": checklist_md,
            "coverage_rows": _coverage_rows(coverage),
            "shape_coverage": _shape_coverage_list(coverage),
            "shape_design_rows": _shape_design_rows(report.get("shape_design", {})),
            "gap_rows": _gap_rows(gaps),
            "performance_rows": _performance_rows(report.get("performance_cases", [])),
            "case_detail_rows": report.get("case_detail_rows", "| none | — |"),
            "perf_detail_rows": report.get("perf_detail_rows", "| none | — |"),
            "memory_detail_rows": report.get("memory_detail_rows", "| none | — |"),
        },
    )


# ---------------------------------------------------------------------------
# file I/O
# ---------------------------------------------------------------------------

def _merge_adjacent_acceptance_artifacts(run_summary: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    merged = dict(run_summary)
    for key, filename in {
        "operator_contract": "operator_contract.json",
        "case_plan": "case_plan.json",
    }.items():
        if isinstance(merged.get(key), dict) and merged[key]:
            continue
        path = output_dir / filename
        if path.exists():
            merged[key] = read_json(path)
    return merged


def write_report(meta_path: str | Path, run_summary_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    meta = read_json(Path(meta_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_summary = _merge_adjacent_acceptance_artifacts(read_json(Path(run_summary_path)), out_dir)
    report = build_acceptance_report(meta, run_summary)
    json_path = out_dir / "acceptance_report.json"
    md_path = out_dir / "acceptance_report.md"
    requests_path = out_dir / "agent_evidence_requests.json"
    write_json(json_path, report)
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    write_json(requests_path, report.get("agent_evidence_requests", []))
    return {"json": json_path, "markdown": md_path, "agent_evidence_requests": requests_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--run-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        paths = write_report(args.meta, args.run_summary, args.output_dir)
    except Exception as exc:
        print(f"[custom-op-acceptance] {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
