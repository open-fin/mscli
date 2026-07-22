#!/usr/bin/env python3
"""Evidence helpers for pytest output and generated test files."""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


COUNT_KEYS = ("passed", "failed", "skipped", "errors", "warnings", "xfailed", "xpassed")
_SUMMARY_TOKEN_RE = re.compile(
    r"(?P<count>\d+)\s+"
    r"(?P<name>passed|failed|skipped|error|errors|warning|warnings|xfailed|xpassed)\b"
)
_SKIPPED_RE = re.compile(
    r"SKIPPED\s+\[(?P<count>\d+)\]\s+"
    r"(?P<location>.*?):(?P<lineno>\d+):\s+(?P<reason>.*)"
)
# Matches verbose (-v) pytest output lines like:
#   test_file.py::test_func[case_id] PASSED [  6%]
_VERBOSE_CASE_RE = re.compile(
    r"::(?P<test_func>test_\w+)\[(?P<case_id>[^\]]+)\]\s+(?P<outcome>PASSED|FAILED|ERROR|SKIPPED)"
)
# Matches pytest --durations output lines like:
#   0.1234s call     test_file.py::test_func[case_id]
_DURATION_RE = re.compile(
    r"(?P<duration>[\d.]+)s\s+call\s+.*::(?P<test_func>test_\w+)\[(?P<case_id>[^\]]+)\]"
)


def _empty_counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


def _normalize_count_key(name: str) -> str:
    if name == "error":
        return "errors"
    if name == "warning":
        return "warnings"
    return name


def parse_pytest_output(text: str, returncode: int | None = None) -> dict[str, Any]:
    """Parse pytest text output without treating skipped tests as a clean pass."""
    counts = _empty_counts()
    for match in _SUMMARY_TOKEN_RE.finditer(text):
        counts[_normalize_count_key(match.group("name"))] += int(match.group("count"))

    skip_reasons = []
    for line in text.splitlines():
        match = _SKIPPED_RE.search(line.strip())
        if not match:
            continue
        skip_reasons.append(
            {
                "count": int(match.group("count")),
                "location": match.group("location"),
                "lineno": int(match.group("lineno")),
                "reason": match.group("reason").strip(),
            }
        )

    blocking_counts = (
        counts["failed"]
        + counts["errors"]
        + counts["skipped"]
        + counts["xfailed"]
        + counts["xpassed"]
    )
    clean_pass = counts["passed"] > 0 and blocking_counts == 0
    if returncode is not None:
        clean_pass = clean_pass and returncode == 0
    return {
        "counts": counts,
        "clean_pass": clean_pass,
        "skip_reasons": skip_reasons,
    }


def _test_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
        return node.name
    return None


def _body_without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if not body:
        return []
    first = body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        if isinstance(first.value.value, str):
            return body[1:]
    return body


def _is_empty_test(node: ast.AST) -> bool:
    body = _body_without_docstring(getattr(node, "body", []))
    if not body:
        return True
    return all(isinstance(stmt, ast.Pass) for stmt in body)


def _pytest_skip_reason(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return None
    call = node.value
    func = call.func
    is_pytest_skip = (
        isinstance(func, ast.Attribute)
        and func.attr == "skip"
        and isinstance(func.value, ast.Name)
        and func.value.id == "pytest"
    )
    if not is_pytest_skip or not call.args:
        return None
    first_arg = call.args[0]
    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
        return first_arg.value
    return "<dynamic skip reason>"


def parse_pytest_verbose_cases(text: str) -> list[dict[str, Any]]:
    """Extract per-case results from verbose (-v) pytest output.

    Returns a list of dicts with keys: case_id, outcome, duration_s (or None).
    """
    cases: dict[str, dict[str, Any]] = {}
    for line in text.splitlines():
        match = _VERBOSE_CASE_RE.search(line)
        if match:
            case_id = match.group("case_id")
            cases[case_id] = {
                "case_id": case_id,
                "outcome": match.group("outcome").lower(),
                "duration_s": None,
            }
    # Merge durations if available (--durations=0)
    if cases:
        for line in text.splitlines():
            dmatch = _DURATION_RE.search(line)
            if dmatch:
                case_id = dmatch.group("case_id")
                if case_id in cases:
                    cases[case_id]["duration_s"] = float(dmatch.group("duration"))
    return list(cases.values())


def analyze_pytest_file(path: str | Path) -> dict[str, Any]:
    """Find docstring-only tests and direct placeholder pytest.skip calls."""
    test_path = Path(path)
    tree = ast.parse(test_path.read_text(encoding="utf-8"), filename=str(test_path))
    empty_tests = []
    placeholder_skips = []
    for node in ast.walk(tree):
        name = _test_name(node)
        if name is None:
            continue
        if _is_empty_test(node):
            empty_tests.append({"name": name, "lineno": node.lineno})
        for stmt in getattr(node, "body", []):
            reason = _pytest_skip_reason(stmt)
            if reason is not None:
                placeholder_skips.append({"name": name, "lineno": node.lineno, "reason": reason})
    return {
        "path": str(test_path),
        "empty_tests": empty_tests,
        "placeholder_skips": placeholder_skips,
    }


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _bare_name(operator_name: str) -> str:
    return operator_name.removesuffix("_op")


def _short_token(operator_name: str) -> str:
    bare = _bare_name(operator_name)
    if bare.startswith("npu_"):
        bare = bare.removeprefix("npu_")
    return bare


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _operator_doc_path(root: Path, meta: dict[str, Any]) -> Path:
    operator_name = meta.get("operator_name", "")
    bare = _bare_name(operator_name)
    direct = root / "docs" / "operators" / f"{bare}.md"
    if direct.exists():
        return direct
    if bare.endswith("_backward"):
        forward = root / "docs" / "operators" / f"{bare.removesuffix('_backward')}.md"
        if forward.exists():
            return forward
    token = _short_token(operator_name)
    return root / "docs" / "operators" / f"npu_{token}.md"


def _collect_doc_audit(root: Path, meta: dict[str, Any]) -> dict[str, Any]:
    doc_path = _operator_doc_path(root, meta)
    text = _read_text(doc_path)
    text_lower = text.lower()
    input_names = [item.get("name", "") for item in meta.get("inputs", [])]
    output_names = [item.get("name", "") for item in meta.get("outputs", [])]
    has_doc = bool(text)
    return {
        "operator_doc": _rel(root, doc_path) if has_doc else "",
        "has_operator_doc": has_doc,
        "has_chinese_doc": bool(re.search(r"[\u4e00-\u9fff]", text)),
        "has_api_description": has_doc
        and any(marker in text for marker in ("功能说明", "函数原型", "公共 API", "Function")),
        "has_formula": any(marker in text for marker in ("计算公式", "formula", "$$", "FastGelu", "fast_gelu")),
        "has_platforms": any(marker in text for marker in ("产品支持情况", "Atlas", "平台")),
        "has_example": "```python" in text or "调用示例" in text,
        "has_example_output": "print(" in text or "输出" in text,
        "has_test_command": "pytest" in text_lower and "test_" in text_lower,
        "input_description_complete": has_doc and all(name and name in text for name in input_names),
        "output_description_complete": has_doc and (
            all(name and name in text for name in output_names) or "返回值" in text or "output" in text_lower
        ),
        "raises_description_complete": any(
            marker in text for marker in ("约束说明", "Raises", "不能为 None", "cannot be null")
        ),
        "format_ok": has_doc and text.startswith("# ") and "\n## " in text,
        "output_shape_same_as_input": bool(re.search(r"(相同|same).{0,24}(shape|尺寸)", text, re.I)),
    }


def _source_candidates(root: Path, meta: dict[str, Any]) -> list[Path]:
    operator_name = meta.get("operator_name", "")
    token = _short_token(operator_name)
    stems = {
        operator_name,
        _bare_name(operator_name),
        token,
        f"aclnn_{token}",
        f"aclnn_{token}_op",
        f"aclnn_{token}_backward_op",
    }
    base = root / "custom_ops" / "ms_adapter" / "pynative"
    candidates: list[Path] = []
    if base.exists():
        for path in base.rglob("*.cpp"):
            stem = path.stem
            if stem in stems or any(part and part in stem for part in (operator_name, token)):
                candidates.append(path)
    source_config = meta.get("source_config")
    if source_config:
        path = Path(source_config)
        if path.exists():
            candidates.append(path)
    unique: list[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _safety_scan(root: Path, paths: list[Path]) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    for path in paths:
        text = _read_text(path)
        rel = _rel(root, path)
        if re.search(r"\bnew\s+[A-Za-z_]", text) and "nothrow" not in text:
            findings.append({"file": rel, "rule": "new_without_nothrow"})
        if re.search(r"\bmalloc\s*\(", text):
            findings.append({"file": rel, "rule": "malloc_usage"})
        if re.search(r"password\s*=|token\s*=|secret\s*=", text, re.I):
            findings.append({"file": rel, "rule": "sensitive_literal"})
    return {
        "status": "passed" if not findings else "failed",
        "checked_files": [_rel(root, path) for path in paths],
        "findings": findings,
    }


def _collect_source_audit(root: Path, meta: dict[str, Any]) -> dict[str, Any]:
    paths = _source_candidates(root, meta)
    combined = "\n".join(_read_text(path) for path in paths)
    tensor_inputs = [
        item
        for item in meta.get("inputs", [])
        if str(item.get("dtype", "")).startswith("Tensor") and str(item.get("dtype")) != "Tensor?"
    ]
    output_shape_rule = "same_shape" if "same_shape(" in combined or "infer_op::same_shape" in combined else ""
    output_dtype_rule = "same_dtype" if "same_dtype(" in combined or "infer_op::same_dtype" in combined else ""
    backward_single_op = None
    if meta.get("has_backward"):
        backward_single_op = bool(re.search(r"npu_\w+(?:_backward|_grad)_op\s*\(", combined))
    return {
        "source_files": [_rel(root, path) for path in paths],
        "output_shape_rule": output_shape_rule,
        "output_dtype_rule": output_dtype_rule,
        "output_shape_depends_on_runtime_data": False if output_shape_rule == "same_shape" else None,
        "backward_single_op": backward_single_op,
        "backward_needs_input_grad_guard": "NeedsInputGrad" in combined,
        "set_unused_inputs_required": False if len(tensor_inputs) <= 1 else None,
        "safety": _safety_scan(root, paths),
    }


def _discover_legacy_tests(root: Path, meta: dict[str, Any]) -> list[Path]:
    tests_root = root / "tests"
    if not tests_root.exists():
        return []
    operator_name = meta.get("operator_name", "")
    token = _short_token(operator_name)
    bare = _bare_name(operator_name)
    patterns = {token, bare, operator_name}
    matches: list[Path] = []
    for path in tests_root.rglob("test*.py"):
        name = path.name
        if any(pattern and pattern in name for pattern in patterns):
            matches.append(path)
    return sorted(matches)


def collect_static_audits(root: Path, meta: dict[str, Any]) -> dict[str, Any]:
    legacy_tests = _discover_legacy_tests(root, meta)
    return {
        "doc_audit": _collect_doc_audit(root, meta),
        "source_audit": _collect_source_audit(root, meta),
        "legacy_test_audit": {
            "status": "not_run" if legacy_tests else "not_applicable",
            "matched_tests": [_rel(root, path) for path in legacy_tests],
            "reason": (
                "matched legacy tests discovered; run --run-pytest to execute"
                if legacy_tests
                else "no matching legacy tests discovered"
            ),
        },
    }
