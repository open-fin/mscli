from pathlib import Path


SKILL_YAML = Path(__file__).resolve().parents[1] / "skill.yaml"
SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_manifest_contract_fields_present():
    text = SKILL_YAML.read_text(encoding="utf-8")
    assert 'schema_version: "1.1.0"' in text
    assert 'name: "algorithm-agent"' in text
    assert 'display_name: "Algorithm Agent"' in text
    assert 'version: "0.4.1"' in text
    assert 'type: "manual"' in text
    assert 'path: "SKILL.md"' in text
    assert 'network: "none"' in text
    assert 'filesystem: "workspace-write"' in text


def test_manifest_declares_route_input_and_output_contract():
    text = SKILL_YAML.read_text(encoding="utf-8")
    assert 'name: "working_dir"' in text
    assert 'name: "source_text"' in text
    assert 'name: "reference_code_path"' in text
    assert 'name: "target_framework"' in text
    assert 'choices: ["auto", "mindspore", "pytorch", "huggingface", "unknown"]' in text
    assert 'name: "route_preference"' in text
    assert 'choices: ["generic-feature", "mhc", "attnres", "transmla"]' in text
    assert 'name: "paper_candidates"' in text
    assert 'report_schema' in text
    assert 'out_dir_layout' in text
    assert '"integration_route_guess"' in text
    assert '"qualification_basis"' in text
    assert '"recommended_next_action"' in text
    assert 'optional_fields: ["recency_or_source_status", "feature_bucket"]' in text
    assert '"source_status"' in text
    assert '"reference_scope"' in text
    assert '"reference_commit_or_tag"' in text
    assert '"preliminary_handoff_target"' in text


def test_skill_declares_route_specific_plan_fields():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "`FeatureSpec` that includes" in text
    assert "`route_specific_constraints`" in text
    assert "`route_specific_validations`" in text
    assert "`code_map_summary`" in text
    assert "`target_framework`" in text
