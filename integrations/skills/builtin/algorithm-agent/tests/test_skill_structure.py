from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_layout_exists():
    assert (ROOT / "SKILL.md").exists()
    assert (ROOT / "skill.yaml").exists()
    assert (ROOT / "references").is_dir()
    assert (ROOT / "scripts").is_dir()


def test_mhc_route_pack_exists():
    assert (ROOT / "references" / "mhc" / "mhc-implementation-pattern.md").exists()
    assert (ROOT / "references" / "mhc" / "mhc-validation-checklist.md").exists()
    assert (ROOT / "references" / "mhc" / "mhc-qwen3-case-study.md").exists()


def test_mindspore_mhc_extension_pack_exists():
    assert (ROOT / "references" / "mhc" / "mindspore-implementation-pattern.md").exists()
    assert (ROOT / "references" / "mhc" / "mindspore-validation-checklist.md").exists()
    assert (ROOT / "references" / "mhc" / "mindspore-qwen3-case-study.md").exists()


def test_attnres_route_pack_exists():
    assert (ROOT / "references" / "attnres" / "attnres-implementation-pattern.md").exists()
    assert (ROOT / "references" / "attnres" / "attnres-validation-checklist.md").exists()
    assert (ROOT / "references" / "attnres" / "attnres-qwen3-case-study.md").exists()


def test_mindspore_attnres_extension_pack_exists():
    assert (ROOT / "references" / "attnres" / "mindspore-attnres-implementation-pattern.md").exists()
    assert (ROOT / "references" / "attnres" / "mindspore-attnres-validation-checklist.md").exists()
    assert (ROOT / "references" / "attnres" / "mindspore-qwen3-attnres-case-study.md").exists()


def test_transmla_route_pack_exists():
    assert (ROOT / "references" / "transmla" / "transmla-implementation-pattern.md").exists()
    assert (ROOT / "references" / "transmla" / "transmla-validation-checklist.md").exists()
    assert (ROOT / "references" / "transmla" / "transmla-case-study.md").exists()
