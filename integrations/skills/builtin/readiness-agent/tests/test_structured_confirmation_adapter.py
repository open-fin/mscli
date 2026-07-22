import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from assessment.structured_confirmation_adapter import run_assessment_with_structured_confirmation


def question(field: str, options: list[dict]) -> dict:
    return {
        "header": field.capitalize(),
        "question": f"Which {field}?",
        "multi_select": False,
        "options": options,
        "response_binding": {
            "field": field,
            "format": f"--confirm {field}=<value>",
        },
    }


def test_structured_confirmation_adapter_calls_ask_user_question_and_replays_confirm():
    calls = []
    run_args = []
    payloads = [
        {
            "status": "NEEDS_CONFIRMATION",
            "portable_question": question("target", [{"value": "training", "label": "Training"}]),
        },
        {
            "status": "WARN",
            "confirmed_fields": {"target": {"value": "training", "source": "confirmed"}},
        },
    ]

    def run_assessment(args):
        run_args.append(list(args))
        return payloads.pop(0)

    def ask_user_question(payload):
        calls.append(payload)
        return {"value": payload["options"][0]["value"]}

    result = run_assessment_with_structured_confirmation(["--mode", "check", "--working-dir", "workspace"], run_assessment, ask_user_question)

    assert result["status"] == "WARN"
    assert len(calls) == 1
    assert calls[0]["options"][0]["value"] == "training"
    assert "--mode" in run_args[-1]
    assert "check" in run_args[-1]
    assert "--working-dir" in run_args[-1]
    assert "workspace" in run_args[-1]
    assert "--confirm" in run_args[-1]
    assert "target=training" in run_args[-1]


def test_structured_confirmation_adapter_loops_until_final_status():
    calls = []
    run_args = []
    payloads = [
        {
            "status": "NEEDS_CONFIRMATION",
            "portable_question": question("target", [{"value": "training", "label": "Training"}]),
        },
        {
            "status": "NEEDS_CONFIRMATION",
            "portable_question": question("framework", [{"value": "mindspore", "label": "mindspore"}]),
        },
        {"status": "READY"},
    ]

    def run_assessment(args):
        run_args.append(list(args))
        return payloads.pop(0)

    def ask_user_question(payload):
        calls.append(payload)
        return payload["options"][0]["value"]

    result = run_assessment_with_structured_confirmation(["--mode", "check"], run_assessment, ask_user_question)

    assert result["status"] == "READY"
    assert len(calls) == 2
    assert "--mode" in run_args[-1]
    assert "check" in run_args[-1]
    assert run_args[-1].count("--confirm") == 2
    assert "target=training" in run_args[-1]
    assert "framework=mindspore" in run_args[-1]


def test_structured_confirmation_adapter_rejects_plain_payload_without_portable_question():
    def run_assessment(_args):
        return {"status": "NEEDS_CONFIRMATION"}

    def ask_user_question(_payload):
        raise AssertionError("adapter should fail before asking")

    try:
        run_assessment_with_structured_confirmation(["--mode", "check"], run_assessment, ask_user_question)
    except ValueError as exc:
        assert "portable_question" in str(exc)
    else:
        raise AssertionError("missing portable_question was accepted")
