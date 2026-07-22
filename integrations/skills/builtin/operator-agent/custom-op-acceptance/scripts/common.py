#!/usr/bin/env python3
"""Shared helpers for the custom-op-acceptance scripts."""
from __future__ import annotations

import json
import re
from pathlib import Path
from string import Template
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
ACCEPTANCE_ROOT = SCRIPT_DIR.parent
DEFAULT_PROJECT_ROOT = Path.cwd()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sanitize_identifier(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")


def normalize_scalar(value: str | None) -> Any:
    if value is None:
        return None
    raw = value.strip().strip('"').strip("'")
    if raw.lower() in {"true", "yes", "on"}:
        return True
    if raw.lower() in {"false", "no", "off"}:
        return False
    if raw.lower() in {"none", "null"}:
        return None
    if raw == "[]":
        return []
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def is_tensor_type(dtype: str) -> bool:
    return dtype.startswith("Tensor")


def split_top_level_csv(value: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    for char in value:
        if quote:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        if char == "," and depth == 0:
            item = "".join(current).strip()
            if item:
                items.append(item)
            current = []
        else:
            current.append(char)
    item = "".join(current).strip()
    if item:
        items.append(item)
    return items


def parse_arg(arg_text: str) -> dict[str, Any]:
    arg_text = arg_text.strip()
    if not arg_text or arg_text == "*":
        raise ValueError("empty argument")
    if "=" in arg_text:
        left, default_text = arg_text.split("=", 1)
        default = normalize_scalar(default_text)
    else:
        left, default = arg_text, None
    parts = left.strip().rsplit(" ", 1)
    if len(parts) != 2:
        raise ValueError(f"cannot parse argument: {arg_text!r}")
    dtype, name = parts
    dtype = dtype.strip()
    return {
        "name": name.strip(),
        "dtype": dtype,
        "default": default,
        "is_tensor": is_tensor_type(dtype),
    }


def parse_signature(signature: str) -> dict[str, Any]:
    normalized = " ".join(signature.split())
    match = re.match(r"(?P<name>[A-Za-z_]\w*)\((?P<args>.*)\)\s*->\s*(?P<returns>.*)$", normalized)
    if not match:
        raise ValueError(f"cannot parse operator signature: {signature!r}")
    args = [parse_arg(item) for item in split_top_level_csv(match.group("args")) if item.strip() != "*"]
    returns = match.group("returns").strip()
    outputs = [] if returns == "()" else [parse_arg(item) for item in split_top_level_csv(returns)]
    return {
        "name": match.group("name"),
        "signature": normalized,
        "args": args,
        "inputs": [arg for arg in args if arg["is_tensor"]],
        "attributes": [arg for arg in args if not arg["is_tensor"]],
        "outputs": outputs,
    }


def iter_yaml_list_blocks(path: Path, item_key: str) -> Iterable[list[str]]:
    if not path.exists():
        return
    pattern = re.compile(rf"^\s*-\s+{re.escape(item_key)}:\s*(.*)$")
    any_item = re.compile(r"^\s*-\s+\w+:\s*")
    current: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if pattern.match(line):
            if current:
                yield current
            current = [line]
            continue
        if current is not None and any_item.match(line):
            yield current
            current = None
        if current is not None:
            current.append(line)
    if current:
        yield current


def extract_signature_from_block(block: list[str], item_key: str) -> str:
    first = re.match(rf"^\s*-\s+{re.escape(item_key)}:\s*(.*)$", block[0])
    if first is None:
        raise ValueError(f"block does not start with {item_key}: {block[0]!r}")
    parts = [first.group(1).strip()]
    for line in block[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r"^[A-Za-z_][\w_]*\s*:", stripped):
            break
        parts.append(stripped)
    return " ".join(parts)


def find_block_value(block: list[str], key: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*(.*)$")
    for line in block:
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def block_has_key(block: list[str], key: str) -> bool:
    return find_block_value(block, key) is not None


def render_template(template_name: str, values: dict[str, Any]) -> str:
    template_path = ACCEPTANCE_ROOT / "templates" / template_name
    return Template(template_path.read_text(encoding="utf-8")).safe_substitute(values)


def case_ids_from_value(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value} if value.strip() else set()
    if isinstance(value, dict):
        for key in ("case_id", "id", "name"):
            if str(value.get(key, "")).strip():
                return {str(value[key])}
        ids: set[str] = set()
        for item in value.values():
            ids.update(case_ids_from_value(item))
        return ids
    if isinstance(value, (list, tuple, set)):
        ids: set[str] = set()
        for item in value:
            ids.update(case_ids_from_value(item))
        return ids
    return set()
