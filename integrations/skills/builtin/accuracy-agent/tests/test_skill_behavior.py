from pathlib import Path


SKILL_MD = Path(__file__).resolve().parents[1] / "SKILL.md"


def test_behavior_rules_require_baseline_and_divergence_reasoning():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Establish a comparable baseline before making root-cause claims." in text
    assert "Find the earliest meaningful divergence before suggesting fixes or naming root" in text
    assert "If there is no trusted baseline, say so explicitly" in text
    assert "Do not claim a fix is confirmed until the user verifies it." in text
    assert "Do not skip workflow stages." in text
    assert "one confirmed accuracy issue per invocation" in text


def test_accuracy_profile_and_consistency_validation_are_present():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Build an `AccuracyProfile`" in text
    assert "Return ranked root-cause candidates with:" in text
    assert "- dtype, precision, API parameter, and device-placement consistency" in text
    assert "- api parameters" in text
    assert "- device placement" in text
    assert "- confidence" in text
    assert "- evidence" in text
    assert "- validation checks" in text
    assert "- fix hints" in text


def test_references_and_scripts_are_declared():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "`references/consistency-validation.md`" in text
    assert "`references/debug-script-hygiene.md`" in text
    assert "`references/operator-accuracy-triage.md`" in text
    assert "`references/reference-baseline-triage.md`" in text
    assert "`references/msprobe-task-router.md`" in text
    assert "`references/msprobe-config-cheatsheet.md`" in text
    assert "`references/msprobe-accuracy-compare.md`" in text
    assert "`references/msprobe-grad-probe.md`" in text
    assert "`references/msprobe-overflow-and-nan.md`" in text
    assert "`references/msprobe-single-op-repro.md`" in text
    assert "`references/msprobe-config-and-ckpt-check.md`" in text
    assert "`references/precision-classification-map.md`" in text
    assert "`scripts/collect_accuracy_context.py`" in text
    assert "`scripts/summarize_metric_diff.py`" in text


def test_debug_script_hygiene_reference_covers_generic_stack_and_determinism():
    text = (SKILL_MD.parent / "references" / "debug-script-hygiene.md").read_text(
        encoding="utf-8"
    )
    assert "backend-specific extension package" in text
    assert "determin" in text
    assert "identity or no-op" in text


def test_operator_triage_reference_tracks_official_api_mapping():
    text = (SKILL_MD.parent / "references" / "operator-accuracy-triage.md").read_text(
        encoding="utf-8"
    )
    assert "https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html" in text
    assert "official PyTorch-to-MindSpore API" in text


def test_skill_requires_baseline_branch_checks_before_operator_blame():
    text = SKILL_MD.read_text(encoding="utf-8")
    assert "Treat the test harness and reference implementation as potential root-cause" in text
    assert "`references/reference-baseline-triage.md`" in text
    assert "the failure appears only in one dtype such as `float64`" in text


def test_reference_baseline_triage_covers_dtype_cast_regression_pattern():
    text = (SKILL_MD.parent / "references" / "reference-baseline-triage.md").read_text(
        encoding="utf-8"
    )
    assert "float64" in text
    assert "astype(np.float32)" in text
    assert "scalar / 0D" in text
    assert "test-harness bug" in text


def test_collect_accuracy_context_tracks_framework_versions():
    text = (SKILL_MD.parent / "scripts" / "collect_accuracy_context.py").read_text(
        encoding="utf-8"
    )
    assert "framework_versions" in text
    assert "torch_npu" in text
    assert "HCCL_DETERMINISTIC" in text


def test_msprobe_router_and_grad_probe_references_cover_core_decisions():
    router = (SKILL_MD.parent / "references" / "msprobe-task-router.md").read_text(
        encoding="utf-8"
    )
    grad = (SKILL_MD.parent / "references" / "msprobe-grad-probe.md").read_text(
        encoding="utf-8"
    )
    assert "Step1 Loss Mismatch" in router
    assert "overflow_check" in router
    assert "config_check" in router
    assert "optimizer-call count" in grad
    assert "0.97" in grad
    assert "0.03" in grad


def test_msprobe_overflow_single_op_and_config_ckpt_references_cover_key_paths():
    overflow = (
        SKILL_MD.parent / "references" / "msprobe-overflow-and-nan.md"
    ).read_text(encoding="utf-8")
    single_op = (
        SKILL_MD.parent / "references" / "msprobe-single-op-repro.md"
    ).read_text(encoding="utf-8")
    config_ckpt = (
        SKILL_MD.parent / "references" / "msprobe-config-and-ckpt-check.md"
    ).read_text(encoding="utf-8")
    assert 'INF_NAN_MODE_ENABLE=1' in overflow
    assert 'MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"' in overflow
    assert "random_data" in single_op
    assert "real_data" in single_op
    assert "config_check" in config_ckpt
    assert "ckpt_compare" in config_ckpt


def test_precision_classification_map_covers_core_buckets():
    text = (
        SKILL_MD.parent / "references" / "precision-classification-map.md"
    ).read_text(encoding="utf-8")
    assert "Baseline And Compare-Harness Problems" in text
    assert "Data, Config, And Environment Alignment Problems" in text
    assert "Module Or Semantic Mismatch" in text
    assert "Single-Operator Numerical Problems" in text
    assert "Training Drift And Update-Path Problems" in text
    assert "Invalid-Value, Overflow, And Backend-Evolution Problems" in text
