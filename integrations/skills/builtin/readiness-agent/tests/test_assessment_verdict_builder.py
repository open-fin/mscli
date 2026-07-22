import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from assessment.verdict_builder import build_assessment_verdict, decide_assessment_status


IDENTITY = {
    "schema_version": "readiness.assessment.v1",
    "producer": "readiness-agent",
    "mode": "check",
    "engine": "assessment",
}


def confirmed_fields(**overrides):
    fields = {
        "target": {"value": "training", "source": "confirmed"},
        "launcher": {"value": "python", "source": "confirmed"},
        "framework": {"value": "mindspore", "source": "confirmed"},
        "runtime_environment": {"value": sys.executable, "source": "confirmed"},
    }
    fields.update(overrides)
    return fields


def assets(**overrides):
    values = {
        "model": {"status": "present", "required": True},
        "dataset": {"status": "present", "required": True},
        "checkpoint": {"status": "unknown", "required": False},
    }
    values.update(overrides)
    return values


def test_decide_assessment_status_ready_when_required_evidence_is_present():
    assert decide_assessment_status(confirmed_fields=confirmed_fields(), assets=assets()) == "READY"


def test_decide_assessment_status_blocks_missing_required_local_model():
    assert decide_assessment_status(
        confirmed_fields=confirmed_fields(),
        assets=assets(model={"status": "missing", "required": True, "source_type": "local_path"}),
    ) == "BLOCKED"


def test_build_assessment_verdict_carries_status_runtime_checks_and_cann():
    verdict = build_assessment_verdict(
        identity=IDENTITY,
        confirmed_fields=confirmed_fields(),
        assets=assets(),
        runtime={"framework": {"value": "mindspore"}, "python": {"value": sys.executable}},
        checks=[{"name": "python_version", "status": "pass"}],
        cann={"status": "detected"},
    )

    assert verdict["status"] == "READY"
    assert verdict["checks"] == [{"name": "python_version", "status": "pass"}]
    assert verdict["runtime"]["cann"] == {"status": "detected"}
