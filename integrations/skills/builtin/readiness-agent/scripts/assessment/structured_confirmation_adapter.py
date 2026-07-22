from typing import Callable, Dict, List


AskUserQuestionCallback = Callable[[Dict[str, object]], object]
RunAssessmentCallback = Callable[[List[str]], Dict[str, object]]


ALLOWED_MANUAL_FIELDS = {"runtime_environment"}


def confirmation_arg(field: str, value: str) -> str:
    return f"{field}={value}"


def select_value_from_answer(answer: object) -> str:
    if isinstance(answer, dict):
        value = answer.get("value")
    else:
        value = answer
    if value is None or str(value) == "":
        raise ValueError("AskUserQuestion answer must include a non-empty value")
    return str(value)


def ask_payload_from_portable_question(question: Dict[str, object]) -> Dict[str, object]:
    return {
        "header": question["header"],
        "question": question["question"],
        "multiSelect": bool(question.get("multi_select", False)),
        "options": question["options"],
    }


def validate_answer_value(field: str, value: str, question: Dict[str, object]) -> None:
    option_values = {str(option.get("value")) for option in question.get("options", []) if isinstance(option, dict) and option.get("value") is not None}
    if value in option_values:
        return
    if field in ALLOWED_MANUAL_FIELDS:
        return
    raise ValueError(f"AskUserQuestion returned invalid value for {field}: {value}")


def run_assessment_with_structured_confirmation(
    initial_args: List[str],
    run_assessment: RunAssessmentCallback,
    ask_user_question: AskUserQuestionCallback,
    max_rounds: int = 8,
) -> Dict[str, object]:
    args = list(initial_args)
    payload = run_assessment(args)
    seen_fields: List[str] = []

    for _round in range(max_rounds):
        if payload.get("status") != "NEEDS_CONFIRMATION":
            return payload
        question = payload.get("portable_question")
        if not isinstance(question, dict):
            raise ValueError("NEEDS_CONFIRMATION payload is missing portable_question")
        response_binding = question.get("response_binding")
        if not isinstance(response_binding, dict) or not response_binding.get("field"):
            raise ValueError("portable_question is missing response_binding.field")
        field = str(response_binding["field"])
        if field in seen_fields:
            raise RuntimeError(f"Repeated confirmation field without progress: {field}")
        seen_fields.append(field)

        answer = ask_user_question(ask_payload_from_portable_question(question))
        value = select_value_from_answer(answer)
        validate_answer_value(field, value, question)
        args.extend(["--confirm", confirmation_arg(field, value)])
        payload = run_assessment(args)

    raise RuntimeError(f"Exceeded maximum structured confirmation rounds: {max_rounds}")
