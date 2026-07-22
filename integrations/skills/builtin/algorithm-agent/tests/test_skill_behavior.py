from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL_MD = ROOT / "SKILL.md"
INTAKE_TRIAGE_MD = ROOT / "references" / "intake-prestage-and-triage.md"
INTAKE_VERIFY_MD = ROOT / "references" / "intake-prestage-verification-and-admission.md"
TRANSMLA_CASE_MD = ROOT / "references" / "transmla" / "transmla-case-study.md"
INTEGRATE_MD = ROOT.parents[1] / "commands" / "integrate.md"


def test_workflow_stages_are_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "1. `feature-analyzer`" in text
    assert "2. `integration-planner`" in text
    assert "3. `patch-builder`" in text
    assert "4. `readiness-handoff-and-report`" in text


def test_route_selection_and_route_packs_are_declared():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Choose exactly one integration route:" in text
    assert "- `generic-feature`" in text
    assert "- `mhc`" in text
    assert "- `attnres`" in text
    assert "- `transmla`" in text
    assert "`integration_route`" in text
    assert "`route_evidence`" in text
    assert "`references/mhc/mhc-implementation-pattern.md`" in text
    assert "`references/mhc/mhc-validation-checklist.md`" in text
    assert "`references/mhc/mhc-qwen3-case-study.md`" in text
    assert "`references/mhc/mindspore-implementation-pattern.md`" in text
    assert "`references/mhc/mindspore-validation-checklist.md`" in text
    assert "`references/mhc/mindspore-qwen3-case-study.md`" in text
    assert "`references/attnres/attnres-implementation-pattern.md`" in text
    assert "`references/attnres/attnres-validation-checklist.md`" in text
    assert "`references/attnres/attnres-qwen3-case-study.md`" in text
    assert "`references/attnres/mindspore-attnres-implementation-pattern.md`" in text
    assert "`references/attnres/mindspore-attnres-validation-checklist.md`" in text
    assert "`references/attnres/mindspore-qwen3-attnres-case-study.md`" in text

def test_mindspore_framework_packs_are_extensions_not_new_routes():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "If the target codebase is MindSpore or `mindone.transformers`" in text
    assert "Select the MindSpore extension pack" in text
    assert "Select the MindSpore AttnRes extension pack" in text
    assert "- `generic-feature`" in text
    assert "- `mhc`" in text
    assert "- `attnres`" in text
    assert "- `transmla`" in text
    assert "- `mhc-mindspore`" not in text
    assert "- `mindspore-mhc`" not in text
    assert "- `attnres-mindspore`" not in text
    assert "- `mindspore-attnres`" not in text

def test_transmla_route_pack_and_conservative_guidance_are_declared():
    text = SKILL_MD.read_text(encoding="utf-8")
    impl_text = (ROOT / "references" / "transmla" / "transmla-implementation-pattern.md").read_text(encoding="utf-8")
    checklist_text = (ROOT / "references" / "transmla" / "transmla-validation-checklist.md").read_text(encoding="utf-8")
    case_text = TRANSMLA_CASE_MD.read_text(encoding="utf-8")

    assert "`references/transmla/transmla-implementation-pattern.md`" in text
    assert "`references/transmla/transmla-validation-checklist.md`" in text
    assert "`references/transmla/transmla-case-study.md`" in text
    assert "checkpoint-remap as a separate follow-on" in text
    assert "semantic-slice work separate from runtime/cache follow-ons" in text
    assert "paged runtime, broader runtime orchestration, and fuller MLA semantics" in text

    assert "## Bounded proving progression" in impl_text
    assert "## Checkpoint-remap boundary" in impl_text
    assert "## Runtime/cache boundary" in impl_text
    assert "Keep each step as one narrow question." in impl_text

    assert "Checkpoint-remap validation only when claimed" in checklist_text
    assert "Cache/runtime smoke only when claimed" in checklist_text
    assert "fuller MLA semantics" in checklist_text

    assert "## What the closed bounded case established" in case_text
    assert "## What stayed separate in the closed result" in case_text
    assert "## Reusable lessons" in case_text
    assert "Do not reopen this closed case" in case_text


def test_intake_prestage_pipeline_rules_are_present():
    triage_text = INTAKE_TRIAGE_MD.read_text(encoding="utf-8")
    verify_text = INTAKE_VERIFY_MD.read_text(encoding="utf-8")
    transmla_case_text = TRANSMLA_CASE_MD.read_text(encoding="utf-8")

    assert "DeepXiv as the preferred/default paper-intake assistant" in triage_text
    assert "intake scoring / triage rubric" in triage_text
    assert "Use `TransMLA` as the first worked example" in triage_text
    assert "`qualification_basis`" in triage_text
    assert "`source_status`" in triage_text

    assert "bounded intake pre-stage should default to one combined helper/scaffold" in verify_text
    assert "### Hard blockers" in verify_text
    assert "Allowed status values:" in verify_text
    assert "- `partial`" in verify_text

    assert "intake -> reference-code map -> bounded patch scope -> focused verification" in transmla_case_text


def test_trending_paper_request_uses_discovery_entry_rule():
    text = SKILL_MD.read_text(encoding="utf-8")
    integrate_text = INTEGRATE_MD.read_text(encoding="utf-8")

    assert "Discovery and intake requests may stop after a bounded shortlist or triage result." in text
    assert "Use that path for requests such as trending papers" in text
    assert "trending paper" in integrate_text
    assert "for discovery or intake requests such as trending papers" in integrate_text


def test_deepxiv_triage_request_stays_discovery_only():
    text = SKILL_MD.read_text(encoding="utf-8")
    integrate_text = INTEGRATE_MD.read_text(encoding="utf-8")

    assert "DeepXiv-assisted triage" in text
    assert "must not imply integration planning, patch generation, or code changes" in text
    assert "DeepXiv" in integrate_text
    assert "bounded shortlist or triage result with recommended next actions" in integrate_text


def test_normal_integration_request_keeps_four_stage_flow():
    text = SKILL_MD.read_text(encoding="utf-8")
    integrate_text = INTEGRATE_MD.read_text(encoding="utf-8")

    assert "Run the integration workflow in this order:" in text
    assert "1. `feature-analyzer`" in text
    assert "2. `integration-planner`" in text
    assert "3. `patch-builder`" in text
    assert "4. `readiness-handoff-and-report`" in text
    assert "for integration requests, present an integration plan with expected changes" in integrate_text
    assert "for integration requests, apply changes incrementally with verification at each step" in integrate_text
    assert "for integration requests, run relevant validation and report results" in integrate_text
