import shlex
from typing import Dict, List, Optional


ALLOWED_CONFIRM_VALUES = {
    "target": {"training", "inference", "__skip__"},
    "framework": {"mindspore", "pytorch_npu", "mixed", "__skip__"},
    "launcher": {"python", "bash", "torchrun", "accelerate", "deepspeed", "msrun", "llamafactory-cli", "make", "__skip__"},
    "run_task_smoke": {"true", "false"},
}


def parse_confirmations(raw_items: Optional[List[str]]) -> Dict[str, str]:
    confirmations: Dict[str, str] = {}
    for item in raw_items or []:
        if "=" not in str(item):
            continue
        field, value = str(item).split("=", 1)
        field = field.strip()
        value = value.strip()
        if field:
            confirmations[field] = value
    return confirmations


def option(value: str, label: str, description: str = "", recommended: bool = False) -> Dict[str, object]:
    payload: Dict[str, object] = {"value": value, "label": label}
    if description:
        payload["description"] = description
    if recommended:
        payload["recommended"] = True
    return payload


def portable_options(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    recommended = [item for item in candidates if item.get("recommended")]
    others = [item for item in candidates if not item.get("recommended")]
    ordered = recommended + others
    visible = ordered[:3]
    visible.append(option("__skip__", "Skip for now", "Continue without confirming this field."))
    return visible[:4]


def portable_question(field: str, candidates: List[Dict[str, object]]) -> Dict[str, object]:
    header = {
        "target": "Target",
        "launcher": "Launcher",
        "framework": "Framework",
        "runtime_environment": "Runtime Env",
        "run_task_smoke": "Task Smoke",
    }.get(field, "Confirm")
    question = {
        "target": "Which target should readiness assess?",
        "launcher": "Which launcher should readiness assess?",
        "framework": "Which framework should readiness assess?",
        "runtime_environment": "Which Python environment should readiness assess?",
        "run_task_smoke": "Run the provided task smoke command to validate readiness?",
    }.get(field, f"Confirm {field}?")
    options = portable_options(candidates)
    values = [str(item.get("value")) for item in candidates if item.get("value") not in {None, ""}]
    if field == "runtime_environment":
        values = values + ["custom"]
    manual_values = "|".join(values) if values else "custom"
    source_index = {str(item.get("value")): index for index, item in enumerate(candidates) if item.get("value") is not None}
    for visible_option in options:
        value = str(visible_option.get("value"))
        if value in source_index:
            visible_option["source_option_index"] = source_index[value]
    return {
        "header": header,
        "question": question,
        "multi_select": False,
        "options": options,
        "selection_strategy": "recommended_first_shortlist",
        "full_option_count": len(candidates),
        "manual_hint": f"Use --confirm {field}=<{manual_values}>",
        "response_binding": {
            "field": field,
            "format": f"--confirm {field}=<value>",
        },
    }


def task_smoke_portable_question(command: str) -> Dict[str, object]:
    return {
        "header": "Task Smoke",
        "question": "Run the provided task smoke command to validate readiness?",
        "multi_select": False,
        "options": [
            option("true", "Run smoke command", "Executes the user-provided smoke command.", recommended=True),
            option("false", "Skip smoke command", "Continue assessment without running smoke."),
        ],
        "selection_strategy": "recommended_first_shortlist",
        "full_option_count": 2,
        "manual_hint": "Use --confirm run_task_smoke=<true|false>",
        "response_binding": {
            "field": "run_task_smoke",
            "format": "--confirm run_task_smoke=<value>",
        },
    }


def append_arg(parts: List[str], name: str, value: object) -> None:
    if value is not None and value != "":
        parts.extend([name, shlex.quote(str(value))])


RESUME_ATTRS = [
    ("working_dir", "--working-dir"),
    ("output_dir", "--output-dir"),
    ("target_hint", "--target-hint"),
    ("launcher_hint", "--launcher-hint"),
    ("framework_hint", "--framework-hint"),
    ("selected_python", "--selected-python"),
    ("selected_env_root", "--selected-env-root"),
    ("model_path", "--model-path"),
    ("dataset_path", "--dataset-path"),
    ("checkpoint_path", "--checkpoint-path"),
    ("launch_command", "--launch-command"),
    ("task_smoke_cmd", "--task-smoke-cmd"),
]


def _build_base_resume_command(args, extra_confirmations: Dict[str, str]) -> List[str]:
    parts = ["python", "scripts/run_readiness_pipeline.py", "--mode", "check"]
    for attr, cli_name in RESUME_ATTRS:
        append_arg(parts, cli_name, getattr(args, attr, None))
    for confirm_field, confirm_value in extra_confirmations.items():
        parts.extend(["--confirm", shlex.quote(f"{confirm_field}={confirm_value}")])
    return parts


def build_resume_command(args, field: str, candidates: List[Dict[str, object]], confirmations: Optional[Dict[str, str]] = None) -> str:
    selected = next((item for item in candidates if item.get("recommended")), candidates[0] if candidates else {"value": "__skip__"})
    merged_confirmations = dict(confirmations or {})
    merged_confirmations[field] = str(selected.get("value"))
    parts = _build_base_resume_command(args, merged_confirmations)
    return " ".join(parts)


def build_task_smoke_resume_command(args, confirmations: Dict[str, str]) -> str:
    merged_confirmations = dict(confirmations or {})
    merged_confirmations["run_task_smoke"] = "true"
    parts = _build_base_resume_command(args, merged_confirmations)
    return " ".join(parts)


def needs_confirmation(args, field: str, candidates: List[Dict[str, object]], confirmations: Dict[str, str]) -> bool:
    if field in confirmations:
        return False
    if getattr(args, "non_interactive", False):
        return False
    if not candidates:
        return True
    if len(candidates) != 1:
        return True
    only = candidates[0]
    if only.get("value") in {None, "", "__unknown__", "__skip__"}:
        return True
    if only.get("confidence") == "low":
        return True
    return False


def validate_confirmation(field: str, value: str, candidates: List[Dict[str, object]]) -> bool:
    candidate_values = {str(item.get("value")) for item in candidates if item.get("value") is not None}
    if value in candidate_values:
        return True
    allowed = ALLOWED_CONFIRM_VALUES.get(field)
    if allowed and value in allowed:
        return True
    if field == "runtime_environment":
        return bool(value)
    return False


def selected_value(field: str, candidates: List[Dict[str, object]], confirmations: Dict[str, str]) -> Dict[str, object]:
    if field in confirmations:
        value = confirmations[field]
        if value == "__skip__":
            return {"value": None, "source": "skipped"}
        if field == "runtime_environment" and not any(str(item.get("value")) == value for item in candidates):
            return {"value": value, "source": "manual"}
        return {"value": value, "source": "confirmed"}
    if candidates:
        return {"value": candidates[0].get("value"), "source": "detected"}
    return {"value": None, "source": "unknown"}
