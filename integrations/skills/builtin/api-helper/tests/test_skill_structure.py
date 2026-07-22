from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_expected_layout_exists():
    assert (ROOT / "SKILL.md").exists()
    assert (ROOT / "skill.yaml").exists()
    assert (ROOT / "reference").is_dir()


def test_reference_packs_exist():
    assert (ROOT / "reference" / "api-to-operator.md").exists()
    assert (ROOT / "reference" / "operator-to-backend.md").exists()
    assert (ROOT / "reference" / "validation_checklist.md").exists()


def test_tests_dir_exists():
    assert (ROOT / "tests").is_dir()
