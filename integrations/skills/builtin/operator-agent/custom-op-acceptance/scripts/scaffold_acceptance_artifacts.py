#!/usr/bin/env python3
"""Create draft operator_contract.json and case_plan.json for agent completion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import case_ids_from_value, read_json, write_json  # noqa: E402


DRAFT_STATUS = "draft_needs_agent_confirmation"
DRAFT_NOTE = "待确认：agent 需结合源码、文档、存量测试和 PTA 基线补证后改为 agent_confirmed"


def _shape_design_from_coverage(meta: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    design = coverage.get("shape_design", {})
    if not isinstance(design, dict) or not design:
        design = {
            "算子名称": meta.get("operator_name", ""),
            "输入 shape 规则": DRAFT_NOTE,
            "输出 shape 规则": DRAFT_NOTE,
            "关键维度含义": DRAFT_NOTE,
            "需要覆盖的 axis / dim": DRAFT_NOTE,
            "需要覆盖的边界 shape": DRAFT_NOTE,
            "是否支持 broadcast": DRAFT_NOTE,
            "是否支持 empty tensor": DRAFT_NOTE,
            "是否支持 0D scalar": DRAFT_NOTE,
            "是否有 dtype 相关分支": DRAFT_NOTE,
            "是否有动态 shape / 动态 rank": DRAFT_NOTE,
            "真实模型中常见 shape": DRAFT_NOTE,
            "非法 shape / 报错 shape": DRAFT_NOTE,
            "case_mapping": {},
        }
    copied = dict(design)
    copied["analysis_status"] = DRAFT_STATUS
    copied.setdefault("case_mapping", {})
    return copied


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _mapping_cases(shape_design: dict[str, Any], *keys: str) -> list[str]:
    mapping = shape_design.get("case_mapping", {})
    if not isinstance(mapping, dict):
        return []
    selected: list[str] = []
    for key in keys:
        selected.extend(sorted(case_ids_from_value(mapping.get(key, []))))
    return _unique(selected)


def build_operator_contract_draft(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator_name": meta.get("operator_name", ""),
        "analysis_status": DRAFT_STATUS,
        "ms_api": meta.get("ms_api", ""),
        "aclnn_api": meta.get("aclnn_api", ""),
        "pta_api": meta.get("pta_baseline_api") or meta.get("pta_api") or "",
        "pta_requires_custom_build": bool(meta.get("pta_requires_custom_build", False)),
        "inputs": meta.get("inputs", []),
        "attributes": meta.get("attributes", []),
        "outputs": meta.get("outputs", []),
        "shape_rule": {
            "input_rule": DRAFT_NOTE,
            "output_rule": DRAFT_NOTE,
            "key_dimensions": DRAFT_NOTE,
            "axis_or_dim": DRAFT_NOTE,
            "supports_broadcast": False,
            "supports_empty_tensor": False,
            "supports_0d_scalar": False,
            "dynamic_shape_or_rank": DRAFT_NOTE,
        },
        "dtype_rule": {
            "pta_supported": [],
            "ms_expected": [],
            "dtype_branches": DRAFT_NOTE,
        },
        "platform_requirements": meta.get("platform_requirements", []),
        "backward_rule": DRAFT_NOTE,
        "evidence": [],
        "scaffold_note": DRAFT_NOTE,
    }


def build_case_plan_draft(meta: dict[str, Any], coverage: dict[str, Any]) -> dict[str, Any]:
    shape_design = _shape_design_from_coverage(meta, coverage)
    all_shape_cases = sorted(case_ids_from_value(shape_design.get("case_mapping", {})))
    profiler_cases = _mapping_cases(shape_design, "真实模型 shape", "边界 shape")
    negative_cases = _mapping_cases(shape_design, "非法 shape / 报错 shape")
    dtype_cases = all_shape_cases[:]
    return {
        "operator_name": meta.get("operator_name", ""),
        "analysis_status": DRAFT_STATUS,
        "shape_design": shape_design,
        "dtype_plan": {
            "pta_supported": [],
            "ms_expected": [],
            "cases": dtype_cases,
            "note": DRAFT_NOTE,
        },
        "functional_cases": all_shape_cases,
        "profiler_cases": profiler_cases,
        "profiler_cases_reason": "" if profiler_cases else DRAFT_NOTE,
        "negative_cases": negative_cases,
        "negative_cases_reason": "" if negative_cases else DRAFT_NOTE,
        "checklist_mapping": {
            "shape": all_shape_cases,
            "dtype": dtype_cases,
        },
        "scaffold_note": DRAFT_NOTE,
    }


def _write_if_needed(path: Path, data: dict[str, Any], overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    write_json(path, data)
    return True


def scaffold_acceptance_artifacts(
    meta_path: str | Path,
    output_dir: str | Path,
    case_coverage_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    meta = read_json(Path(meta_path))
    coverage = {}
    if case_coverage_path and Path(case_coverage_path).exists():
        coverage = read_json(Path(case_coverage_path))
    out_dir = Path(output_dir)
    contract_path = out_dir / "operator_contract.json"
    plan_path = out_dir / "case_plan.json"
    contract_written = _write_if_needed(
        contract_path,
        build_operator_contract_draft(meta),
        overwrite,
    )
    plan_written = _write_if_needed(
        plan_path,
        build_case_plan_draft(meta, coverage),
        overwrite,
    )
    return {
        "operator_contract": str(contract_path),
        "case_plan": str(plan_path),
        "operator_contract_written": contract_written,
        "case_plan_written": plan_written,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-coverage")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = scaffold_acceptance_artifacts(
            args.meta,
            args.output_dir,
            case_coverage_path=args.case_coverage,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"[custom-op-acceptance] {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
