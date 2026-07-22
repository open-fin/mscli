#!/usr/bin/env python3
"""Inspect custom_ops YAML and emit acceptance metadata for one operator."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    DEFAULT_PROJECT_ROOT,
    block_has_key,
    extract_signature_from_block,
    find_block_value,
    iter_yaml_list_blocks,
    normalize_scalar,
    parse_signature,
    write_json,
)


KNOWN_TORCH_NPU_BASELINES = {
    "npu_fast_gelu": "torch_npu.npu_fast_gelu",
    "npu_fast_gelu_backward": "torch_npu.npu_fast_gelu_backward",
    "npu_fused_infer_attention_score": "torch_npu.npu_fused_infer_attention_score",
    "npu_mhc_pre": "torch_npu.npu_mhc_pre",
}


def _candidate_names(operator_name: str) -> set[str]:
    names = {operator_name}
    if operator_name.endswith("_op"):
        names.add(operator_name.removesuffix("_op"))
    else:
        names.add(f"{operator_name}_op")
    return names


def _clean_yaml_value(value: str | None) -> Any:
    if value is None:
        return None
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    return normalize_scalar(value)


def _bare_operator_name(operator_name: str) -> str:
    return operator_name.removesuffix("_op")


def _read_text_if_exists(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _candidate_reference_paths(project_root: Path, bare_name: str, op_name: str) -> list[Path]:
    return [
        project_root / "docs" / "operators" / f"{bare_name}.md",
        project_root / "docs" / "operators" / "supported_operators.md",
        project_root / "tests" / "operators" / "aclnn" / f"test_{op_name}.py",
        project_root / "tests" / "operators" / "aclnn" / f"test_{bare_name}_op.py",
        project_root / "custom_ops" / "ms_adapter" / "python" / "ops.py",
    ]


def _operator_specific_reference_paths(project_root: Path, bare_name: str, op_name: str) -> list[Path]:
    return [
        project_root / "docs" / "operators" / f"{bare_name}.md",
        project_root / "tests" / "operators" / "aclnn" / f"test_{op_name}.py",
        project_root / "tests" / "operators" / "aclnn" / f"test_{bare_name}_op.py",
    ]


def _infer_torch_npu_api(project_root: Path, op_name: str) -> str | None:
    bare_name = _bare_operator_name(op_name)
    pattern = re.compile(rf"\btorch_npu\.{re.escape(bare_name)}\b")
    for path in _candidate_reference_paths(project_root, bare_name, op_name):
        if pattern.search(_read_text_if_exists(path)):
            return f"torch_npu.{bare_name}"
    return KNOWN_TORCH_NPU_BASELINES.get(bare_name)


def _infer_pta(project_root: Path, op_name: str) -> dict[str, Any]:
    torch_npu_api = _infer_torch_npu_api(project_root, op_name)
    if torch_npu_api is None:
        return {
            "has_pta": False,
            "pta_api": None,
            "pta_baseline_api": None,
            "pta_requires_custom_build": False,
            "pta_bare_name": None,
            "pta_namespace": None,
        }
    bare_name = _bare_operator_name(op_name)
    return {
        "has_pta": True,
        "pta_api": f"torch.ops.custom.{bare_name}",
        "pta_baseline_api": torch_npu_api,
        "pta_requires_custom_build": False,
        "pta_bare_name": bare_name,
        "pta_namespace": "custom",
        "torch_npu_reference_api": torch_npu_api,
    }


def _infer_platform_requirements(project_root: Path, op_name: str) -> dict[str, Any]:
    bare_name = _bare_operator_name(op_name)
    texts = [
        _read_text_if_exists(path)
        for path in _operator_specific_reference_paths(project_root, bare_name, op_name)
    ]
    combined = "\n".join(texts)
    if "Ascend950" in combined or "910_95" in combined:
        return {
            "required_devices": ["Ascend950", "910_95"],
            "reason": "operator docs or tests restrict core validation to Ascend950/910_95-class devices",
        }
    return {"required_devices": [], "reason": ""}


def _parse_pta(block: list[str], op_name: str) -> dict[str, Any]:
    has_pta = any(line.strip() == "pta_extras:" for line in block)
    if not has_pta:
        return {
            "has_pta": False,
            "pta_api": None,
            "pta_baseline_api": None,
            "pta_requires_custom_build": False,
            "pta_bare_name": None,
            "pta_namespace": None,
        }
    bare_name = _clean_yaml_value(find_block_value(block, "bare_name")) or op_name.removesuffix("_op")
    namespace = _clean_yaml_value(find_block_value(block, "namespace")) or "custom"
    torch_npu_api = _clean_yaml_value(find_block_value(block, "torch_npu_api"))
    pta_custom_api = f"torch.ops.{namespace}.{bare_name}"
    return {
        "has_pta": True,
        "pta_api": pta_custom_api,
        "pta_baseline_api": torch_npu_api or pta_custom_api,
        "pta_requires_custom_build": torch_npu_api is None,
        "pta_bare_name": bare_name,
        "pta_namespace": namespace,
        "torch_npu_reference_api": torch_npu_api,
    }


def _parse_aclnn_block(project_root: Path, block: list[str]) -> dict[str, Any]:
    parsed = parse_signature(extract_signature_from_block(block, "op"))
    pta = _parse_pta(block, parsed["name"])
    if not pta["has_pta"]:
        pta = _infer_pta(project_root, parsed["name"])
    return {
        "operator_name": parsed["name"],
        "kind": "aclnn",
        "signature": parsed["signature"],
        "source_config": str(project_root / "custom_ops" / "config" / "aclnn_ops.yaml"),
        "ms_api": f"custom_ops.ops.{parsed['name']}",
        "aclnn_api": _clean_yaml_value(find_block_value(block, "api")),
        "inputs": parsed["inputs"],
        "attributes": parsed["attributes"],
        "outputs": parsed["outputs"],
        "has_backward": parsed["name"].endswith("_backward_op") or "grad" in parsed["name"],
        "manual": bool(_clean_yaml_value(find_block_value(block, "manual"))),
        "platform_requirements": _infer_platform_requirements(project_root, parsed["name"]),
        **pta,
    }


def _parse_pyboost_block(project_root: Path, block: list[str]) -> dict[str, Any]:
    parsed = parse_signature(extract_signature_from_block(block, "op"))
    return {
        "operator_name": parsed["name"],
        "kind": "pyboost",
        "signature": parsed["signature"],
        "source_config": str(project_root / "custom_ops" / "config" / "pyboost_ops.yaml"),
        "ms_api": f"custom_ops.ops.{parsed['name']}",
        "aclnn_api": None,
        "pyboost_api": _clean_yaml_value(find_block_value(block, "exec")),
        "inputs": parsed["inputs"],
        "attributes": parsed["attributes"],
        "outputs": parsed["outputs"],
        "has_backward": False,
        "platform_requirements": _infer_platform_requirements(project_root, parsed["name"]),
        "has_pta": False,
        "pta_api": None,
        "pta_baseline_api": None,
        "pta_requires_custom_build": False,
        "pta_bare_name": None,
        "pta_namespace": None,
    }


def _parse_function_block(project_root: Path, block: list[str]) -> dict[str, Any]:
    parsed = parse_signature(extract_signature_from_block(block, "func"))
    pta = _infer_pta(project_root, parsed["name"])
    return {
        "operator_name": parsed["name"],
        "kind": "function",
        "signature": parsed["signature"],
        "source_config": str(project_root / "custom_ops" / "config" / "function_definitions.yaml"),
        "ms_api": f"custom_ops.ops.{parsed['name']}",
        "aclnn_api": None,
        "inputs": parsed["inputs"],
        "attributes": parsed["attributes"],
        "outputs": parsed["outputs"],
        "has_backward": block_has_key(block, "backward"),
        "platform_requirements": _infer_platform_requirements(project_root, parsed["name"]),
        **pta,
    }


def _matches(meta: dict[str, Any], operator_name: str) -> bool:
    names = _candidate_names(operator_name)
    if meta["operator_name"] in names:
        return True
    bare = meta.get("pta_bare_name")
    return isinstance(bare, str) and bare in names


def collect_operator_meta(project_root: str | Path, operator_name: str) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config_dir = root / "custom_ops" / "config"
    metas: list[dict[str, Any]] = []

    for block in iter_yaml_list_blocks(config_dir / "function_definitions.yaml", "func"):
        meta = _parse_function_block(root, block)
        if meta["operator_name"] == operator_name:
            return meta
        metas.append(meta)

    for block in iter_yaml_list_blocks(config_dir / "aclnn_ops.yaml", "op"):
        meta = _parse_aclnn_block(root, block)
        if meta["operator_name"] == operator_name:
            return meta
        metas.append(meta)

    for block in iter_yaml_list_blocks(config_dir / "pyboost_ops.yaml", "op"):
        meta = _parse_pyboost_block(root, block)
        if meta["operator_name"] == operator_name:
            return meta
        metas.append(meta)

    for meta in metas:
        if _matches(meta, operator_name):
            return meta

    raise KeyError(f"operator {operator_name!r} not found under {config_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator_name")
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--output", help="Path for operator_meta.json")
    args = parser.parse_args(argv)

    try:
        meta = collect_operator_meta(args.project_root, args.operator_name)
    except (KeyError, ValueError) as exc:
        print(f"[custom-op-acceptance] {exc}", file=sys.stderr)
        return 2

    if args.output:
        write_json(Path(args.output), meta)
    else:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
