from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
import subprocess

import pytest


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import framework

from framework import (
    CaseSpec,
    ClaudeCodeExecutor,
    CodexExecutor,
    ExecutionRequest,
    ExecutionResult,
    ExpectedChangeSet,
    Manifest,
    ManifestError,
    OpenCodeExecutor,
    RepoSpec,
    SessionTaskWindow,
    SkillTestRunner,
    build_argument_parser,
    build_activity_timeline_output_schema,
    find_codex_session_task_window,
    format_elapsed_minutes,
    classify_repo_changes,
    load_manifest,
    normalize_activity_timeline_summary,
    prepare_repos,
    render_prompt,
    resolve_op_plugin_path,
    resolve_skill_path,
    resolve_repo_source,
    skill_install_subdir_for_executor,
    stage_skills_for_case,
    write_task_session_slice,
)


def git(args: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def init_repo(path: Path, files: dict[str, str]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    git(["init"], cwd=path)
    git(["config", "user.email", "test@example.com"], cwd=path)
    git(["config", "user.name", "Test User"], cwd=path)
    for relative_path, content in files.items():
        file_path = path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    git(["add", "."], cwd=path)
    git(["commit", "-m", "initial"], cwd=path)


class FakeExecutor:
    def __init__(self, mutation_map: dict[str, list[tuple[str, str, str | None]]]) -> None:
        self.mutation_map = mutation_map

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        for action, repo_name, relative_path in self.mutation_map[request.case.case_id]:
            repo_root = request.prepared_repos[repo_name].checkout_path
            if action == "add":
                target = repo_root / str(relative_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(f"created by {request.case.case_id}\n", encoding="utf-8")
            elif action == "modify":
                target = repo_root / str(relative_path)
                target.write_text(f"modified by {request.case.case_id}\n", encoding="utf-8")
            elif action == "delete":
                target = repo_root / str(relative_path)
                target.unlink()
            elif action == "artifact":
                target = repo_root / str(relative_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("artifact\n", encoding="utf-8")
            else:
                raise AssertionError(f"Unsupported fake action: {action}")

        stdout_path = request.case_dir / "fake_stdout.txt"
        stderr_path = request.case_dir / "fake_stderr.txt"
        last_message_path = request.case_dir / "fake_last_message.txt"
        stdout_path.write_text("ok\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        last_message_path.write_text(f"finished {request.case.case_id}\n", encoding="utf-8")
        return ExecutionResult(
            command=("fake-executor", request.case.case_id),
            returncode=0,
            started_at=0.0,
            finished_at=1.0,
            timed_out=False,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            last_message_path=last_message_path,
        )


def test_manifest_template_loads():
    manifest = load_manifest(TESTS_DIR / "cases.yaml")
    assert manifest.schema_version == "1.0.0"
    assert manifest.cases[0].case_id == "adaptivemaxpool3d"
    assert manifest.cases[0].enabled is True


def test_render_prompt_replaces_repo_and_artifact_placeholders(tmp_path: Path):
    repo_root = tmp_path / "repo"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(
        case_id="prompt_case",
        prompt="skill={{skill:aclnn_builder}} repo={{repo:mindspore}} artifact={{artifact_dir}} op={{op_plugin_dir}}",
        repos=(repo_spec,),
    )
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        prompt = render_prompt(
            case.prompt,
            prepared_repos=prepared,
            artifact_dir=tmp_path / "artifacts",
            case_dir=tmp_path / "case",
            run_dir=tmp_path / "run",
            op_plugin_dir=op_plugin_root,
        )
        assert str(prepared["mindspore"].checkout_path) in prompt
        assert str(tmp_path / "artifacts") in prompt
        assert str(op_plugin_root) in prompt
        assert "skill=$aclnn_builder" in prompt
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_render_prompt_uses_slash_skill_marker_for_slash_agents(tmp_path: Path):
    repo_root = tmp_path / "repo"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(
        case_id="prompt_case",
        prompt="Use {{skill:aclnn_builder}} and {{skill:other.skill-name}}",
        repos=(repo_spec,),
    )
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        prompt = render_prompt(
            case.prompt,
            prepared_repos=prepared,
            artifact_dir=tmp_path / "artifacts",
            case_dir=tmp_path / "case",
            run_dir=tmp_path / "run",
            op_plugin_dir=op_plugin_root,
            skill_trigger_prefix="/",
        )
        assert "Use /aclnn_builder and /other.skill-name" == prompt
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_classify_repo_changes_separates_artifacts(tmp_path: Path):
    repo_root = tmp_path / "repo"
    init_repo(repo_root, {"tracked.txt": "base\n", "remove_me.txt": "bye\n"})
    repo_spec = RepoSpec(
        name="mindspore",
        source=str(repo_root),
        artifact_globs=("*_Feature.md", "*_pta_analysis.md"),
        expected_changes=ExpectedChangeSet(
            added=("new_file.cc",),
            modified=("tracked.txt",),
            deleted=("remove_me.txt",),
        ),
    )
    case = CaseSpec(case_id="diff_case", prompt="unused", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        checkout = prepared["mindspore"].checkout_path
        (checkout / "tracked.txt").write_text("changed\n", encoding="utf-8")
        (checkout / "new_file.cc").write_text("new\n", encoding="utf-8")
        (checkout / "remove_me.txt").unlink()
        (checkout / "abs_Feature.md").write_text("artifact\n", encoding="utf-8")
        outcome = classify_repo_changes(prepared["mindspore"])
        assert outcome.valid is True
        assert outcome.expected_changes.added == ("new_file.cc",)
        assert outcome.source_changes.added == ("new_file.cc",)
        assert outcome.source_changes.modified == ("tracked.txt",)
        assert outcome.source_changes.deleted == ("remove_me.txt",)
        assert outcome.artifact_changes == ("abs_Feature.md",)
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_extra_changes_are_allowed_if_expected_bucketed_changes_exist(tmp_path: Path):
    repo_root = tmp_path / "repo"
    init_repo(repo_root, {"tracked.txt": "base\n", "remove_me.txt": "bye\n"})
    repo_spec = RepoSpec(
        name="mindspore",
        source=str(repo_root),
        expected_changes=ExpectedChangeSet(
            added=("new_file.cc",),
            modified=("tracked.txt",),
            deleted=("remove_me.txt",),
        ),
    )
    case = CaseSpec(case_id="extra_changes_case", prompt="unused", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        checkout = prepared["mindspore"].checkout_path
        (checkout / "tracked.txt").write_text("changed\n", encoding="utf-8")
        (checkout / "new_file.cc").write_text("new\n", encoding="utf-8")
        (checkout / "remove_me.txt").unlink()
        (checkout / "extra_added.cc").write_text("extra\n", encoding="utf-8")
        (checkout / "extra_modified.txt").write_text("extra tracked\n", encoding="utf-8")
        outcome = classify_repo_changes(prepared["mindspore"])
        assert outcome.valid is True
        assert "new_file.cc" in outcome.source_changes.added
        assert "tracked.txt" in outcome.source_changes.modified
        assert "remove_me.txt" in outcome.source_changes.deleted
        assert "extra_added.cc" in outcome.source_changes.added
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_classify_repo_changes_can_skip_expected_change_comparison(tmp_path: Path):
    repo_root = tmp_path / "repo"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    repo_spec = RepoSpec(
        name="mindspore",
        source=str(repo_root),
        expected_changes=ExpectedChangeSet(added=("missing.cc",)),
    )
    case = CaseSpec(case_id="skip_compare_case", prompt="unused", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        checkout = prepared["mindspore"].checkout_path
        (checkout / "actual.cc").write_text("actual\n", encoding="utf-8")

        checked = classify_repo_changes(prepared["mindspore"])
        skipped = classify_repo_changes(
            prepared["mindspore"],
            compare_expected_changes_enabled=False,
        )

        assert checked.valid is False
        assert "Missing added paths: ['missing.cc']" in checked.problems
        assert skipped.valid is True
        assert skipped.problems == ()
        assert skipped.source_changes.added == ("actual.cc",)
        assert skipped.expected_changes.added == ("missing.cc",)
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_runner_executes_multiple_cases_with_isolation(tmp_path: Path):
    source_repo = tmp_path / "mindspore"
    skill_root = tmp_path / ".codex" / "skills"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(source_repo, {"base.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder").mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder" / "SKILL.md").write_text("skill\n", encoding="utf-8")
    manifest = Manifest(
        schema_version="1.0.0",
        default_artifact_globs=(),
        cases=(
            CaseSpec(
                case_id="case_one",
                prompt="first",
                repos=(
                    RepoSpec(
                        name="mindspore",
                        source=str(source_repo),
                        artifact_globs=("*_Feature.md",),
                        expected_changes=ExpectedChangeSet(
                            added=("alpha.txt",),
                            modified=(),
                            deleted=(),
                        ),
                    ),
                ),
            ),
            CaseSpec(
                case_id="case_two",
                prompt="second",
                repos=(
                    RepoSpec(
                        name="mindspore",
                        source=str(source_repo),
                        artifact_globs=("*_Feature.md",),
                        expected_changes=ExpectedChangeSet(
                            added=("beta.txt",),
                            modified=(),
                            deleted=(),
                        ),
                    ),
                ),
            ),
        ),
    )
    executor = FakeExecutor(
        {
            "case_one": [("add", "mindspore", "alpha.txt"), ("artifact", "mindspore", "alpha_Feature.md")],
            "case_two": [("add", "mindspore", "beta.txt"), ("artifact", "mindspore", "beta_Feature.md")],
        }
    )
    runner = SkillTestRunner(
        manifest=manifest,
        executor=executor,
        ms_root=source_repo,
        path_root=tmp_path,
        skill_path=skill_root,
        op_plugin_dir=op_plugin_root,
        runs_root=tmp_path / "runs",
    )
    run_dir, outcomes = runner.run()
    assert len(outcomes) == 2
    assert all(outcome.valid for outcome in outcomes)
    assert outcomes[0].repo_outcomes[0].source_changes.added == ("alpha.txt",)
    assert outcomes[1].repo_outcomes[0].source_changes.added == ("beta.txt",)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_id"]
    assert isinstance(summary["elapsed_time"], str)
    assert summary["elapsed_time"].endswith("min")
    assert [item["case_id"] for item in summary["case_outcomes"]] == ["case_one", "case_two"]
    assert summary["case_outcomes"][0]["repo_outcomes"][0]["expected_changes"]["added"] == ["alpha.txt"]
    assert summary["case_outcomes"][0]["execution_result"]["elapsed_time"] == format_elapsed_minutes(1.0)
    assert summary["expected_change_comparison_enabled"] is True


def test_runner_can_skip_expected_change_comparison(tmp_path: Path):
    source_repo = tmp_path / "mindspore"
    skill_root = tmp_path / ".codex" / "skills"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(source_repo, {"base.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder").mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder" / "SKILL.md").write_text("skill\n", encoding="utf-8")
    manifest = Manifest(
        schema_version="1.0.0",
        default_artifact_globs=(),
        cases=(
            CaseSpec(
                case_id="case_one",
                prompt="first",
                repos=(
                    RepoSpec(
                        name="mindspore",
                        source=str(source_repo),
                        expected_changes=ExpectedChangeSet(added=("expected.txt",)),
                    ),
                ),
            ),
        ),
    )
    executor = FakeExecutor({"case_one": [("add", "mindspore", "actual.txt")]})
    runner = SkillTestRunner(
        manifest=manifest,
        executor=executor,
        ms_root=source_repo,
        path_root=tmp_path,
        skill_path=skill_root,
        op_plugin_dir=op_plugin_root,
        runs_root=tmp_path / "runs",
        compare_expected_changes_enabled=False,
    )
    run_dir, outcomes = runner.run()

    assert len(outcomes) == 1
    assert outcomes[0].valid is True
    assert outcomes[0].repo_outcomes[0].source_changes.added == ("actual.txt",)
    assert outcomes[0].repo_outcomes[0].expected_changes.added == ("expected.txt",)
    assert outcomes[0].repo_outcomes[0].problems == ()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["expected_change_comparison_enabled"] is False


def test_runner_skip_expected_changes_allows_manifest_without_expected_changes(tmp_path: Path):
    source_repo = tmp_path / "mindspore"
    skill_root = tmp_path / ".codex" / "skills"
    op_plugin_root = tmp_path / "op-plugin"
    manifest_path = tmp_path / "cases.yaml"
    init_repo(source_repo, {"base.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder").mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder" / "SKILL.md").write_text("skill\n", encoding="utf-8")
    manifest_path.write_text(
        f"""
schema_version: "1.0.0"
cases:
  - id: "case_without_expected_changes"
    prompt: "first"
    repos:
      - name: "mindspore"
        source: "{source_repo}"
""".strip(),
        encoding="utf-8",
    )
    manifest = load_manifest(manifest_path)
    executor = FakeExecutor({"case_without_expected_changes": [("add", "mindspore", "actual.txt")]})
    runner = SkillTestRunner(
        manifest=manifest,
        executor=executor,
        ms_root=source_repo,
        path_root=tmp_path,
        skill_path=skill_root,
        op_plugin_dir=op_plugin_root,
        runs_root=tmp_path / "runs",
        compare_expected_changes_enabled=False,
    )
    run_dir, outcomes = runner.run()

    assert len(outcomes) == 1
    assert outcomes[0].valid is True
    assert outcomes[0].repo_outcomes[0].expected_changes.as_dict() == {
        "added": [],
        "modified": [],
        "deleted": [],
    }
    assert outcomes[0].repo_outcomes[0].source_changes.added == ("actual.txt",)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["expected_change_comparison_enabled"] is False


def test_runner_executes_multiple_cases_in_parallel(tmp_path: Path):
    source_repo = tmp_path / "mindspore"
    skill_root = tmp_path / ".codex" / "skills"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(source_repo, {"base.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder").mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder" / "SKILL.md").write_text("skill\n", encoding="utf-8")
    manifest = Manifest(
        schema_version="1.0.0",
        default_artifact_globs=(),
        cases=(
            CaseSpec(
                case_id="case_one",
                prompt="first",
                repos=(
                    RepoSpec(
                        name="mindspore",
                        source=str(source_repo),
                        artifact_globs=("*_Feature.md",),
                        expected_changes=ExpectedChangeSet(added=("alpha.txt",)),
                    ),
                ),
            ),
            CaseSpec(
                case_id="case_two",
                prompt="second",
                repos=(
                    RepoSpec(
                        name="mindspore",
                        source=str(source_repo),
                        artifact_globs=("*_Feature.md",),
                        expected_changes=ExpectedChangeSet(added=("beta.txt",)),
                    ),
                ),
            ),
        ),
    )
    executor = FakeExecutor(
        {
            "case_one": [("add", "mindspore", "alpha.txt"), ("artifact", "mindspore", "alpha_Feature.md")],
            "case_two": [("add", "mindspore", "beta.txt"), ("artifact", "mindspore", "beta_Feature.md")],
        }
    )
    runner = SkillTestRunner(
        manifest=manifest,
        executor=executor,
        ms_root=source_repo,
        path_root=tmp_path,
        skill_path=skill_root,
        op_plugin_dir=op_plugin_root,
        runs_root=tmp_path / "runs",
    )
    _, outcomes = runner.run(workers=2)

    assert len(outcomes) == 2
    assert all(outcome.valid for outcome in outcomes)
    actual_added = {outcome.case_id: outcome.repo_outcomes[0].source_changes.added for outcome in outcomes}
    assert actual_added == {
        "case_one": ("alpha.txt",),
        "case_two": ("beta.txt",),
    }


def test_manifest_rejects_overlapping_expected_paths(tmp_path: Path):
    manifest_path = tmp_path / "bad.yaml"
    manifest_path.write_text(
        """
schema_version: "1.0.0"
cases:
  - id: "bad_case"
    prompt: "test"
    repos:
      - name: "mindspore"
        source: "relative/mindspore"
        expected_changes:
          added: ["same.cc"]
          modified: ["same.cc"]
          deleted: []
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ManifestError):
        load_manifest(manifest_path)


def test_codex_executor_builds_ephemeral_command(tmp_path: Path):
    repo_root = tmp_path / "repo"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="cmd_case", prompt="test", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        executor = CodexExecutor(codex_bin="codex", model="gpt-5.4", ephemeral=True, extra_args=("--search",))
        request = ExecutionRequest(
            case=case,
            prompt="hello",
            prompt_path=tmp_path / "prompt.txt",
            case_dir=tmp_path / "case",
            run_dir=tmp_path / "run",
            artifact_dir=tmp_path / "artifacts",
            op_plugin_dir=op_plugin_root,
            prepared_repos=prepared,
        )
        command = executor.build_command(request)
        assert command[:4] == ["codex", "exec", "--json", "--sandbox"]
        assert "--ephemeral" in command
        assert "--output-last-message" in command
        assert "--search" in command
        assert str(prepared["mindspore"].checkout_path) in command
        assert str(op_plugin_root) in command
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_codex_executor_omits_ephemeral_when_analyze_time_is_enabled(tmp_path: Path):
    repo_root = tmp_path / "repo"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="cmd_case", prompt="test", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        executor = CodexExecutor(codex_bin="codex", analyze_time=True)
        request = ExecutionRequest(
            case=case,
            prompt="hello",
            prompt_path=tmp_path / "prompt.txt",
            case_dir=tmp_path / "case",
            run_dir=tmp_path / "run",
            artifact_dir=tmp_path / "artifacts",
            op_plugin_dir=op_plugin_root,
            prepared_repos=prepared,
        )
        command = executor.build_command(request)
        assert "--ephemeral" not in command
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_cli_analyze_time_flag_defaults_off_and_can_be_enabled():
    parser = build_argument_parser()
    default_args = parser.parse_args([])
    assert default_args.analyze_time is False
    enabled_args = parser.parse_args(["--analyze-time"])
    assert enabled_args.analyze_time is True


def test_cli_sandboxes_default_to_kept_and_cleanup_flag_disables_it():
    parser = build_argument_parser()
    default_args = parser.parse_args([])
    assert default_args.keep_sandboxes is True

    cleanup_args = parser.parse_args(["--cleanup-sandboxes"])
    assert cleanup_args.keep_sandboxes is False

    with pytest.raises(SystemExit):
        parser.parse_args(["--keep-sandboxes"])


def test_cli_workers_defaults_to_one_and_accepts_explicit_value():
    parser = build_argument_parser()
    default_args = parser.parse_args([])
    assert default_args.workers == 1

    custom_args = parser.parse_args(["--workers", "4"])
    assert custom_args.workers == 4

    with pytest.raises(SystemExit):
        parser.parse_args(["--workers", "0"])


def test_opencode_executor_builds_command(tmp_path: Path):
    repo_root = tmp_path / "repo"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="cmd_case", prompt="test", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        executor = OpenCodeExecutor(opencode_bin="opencode", model="provider/model")
        request = ExecutionRequest(
            case=case,
            prompt="hello",
            prompt_path=tmp_path / "prompt.txt",
            case_dir=tmp_path / "case",
            run_dir=tmp_path / "run",
            artifact_dir=tmp_path / "artifacts",
            op_plugin_dir=op_plugin_root,
            prepared_repos=prepared,
        )
        command = executor.build_command(request)
        assert command[:5] == ["opencode", "run", "--dir", str(prepared["mindspore"].checkout_path), "--format"]
        assert "--model" in command
        assert command[-1] == "hello"
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_claudecode_executor_builds_command(tmp_path: Path):
    repo_root = tmp_path / "repo"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="cmd_case", prompt="test", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        executor = ClaudeCodeExecutor(claudecode_bin="claude", model="sonnet")
        request = ExecutionRequest(
            case=case,
            prompt="hello",
            prompt_path=tmp_path / "prompt.txt",
            case_dir=tmp_path / "case",
            run_dir=tmp_path / "run",
            artifact_dir=tmp_path / "artifacts",
            op_plugin_dir=op_plugin_root,
            prepared_repos=prepared,
        )
        command = executor.build_command(request)
        assert command[:5] == ["claude", "-p", "--output-format", "json", "--permission-mode"]
        assert "--model" in command
        assert str(op_plugin_root) in command
        assert "hello" not in command
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_claudecode_executor_passes_prompt_on_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "repo"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="cmd_case", prompt="test", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    captured: dict[str, object] = {}

    class FakePopen:
        def __init__(self, command: list[str], **kwargs) -> None:
            captured["command"] = command
            captured["env"] = kwargs.get("env")
            captured["stdin"] = kwargs.get("stdin")
            self.stdout = kwargs["stdout"]
            self.returncode = 0

        def communicate(self, prompt: str | None = None, timeout: int | None = None) -> tuple[str, str]:
            captured["prompt"] = prompt
            captured["timeout"] = timeout
            self.stdout.write('{"result":"ok"}\n')
            self.stdout.flush()
            return "", ""

    monkeypatch.setattr(framework.subprocess, "Popen", FakePopen)
    monkeypatch.setenv("CODEX_SANDBOX_NETWORK_DISABLED", "1")

    try:
        executor = ClaudeCodeExecutor(claudecode_bin="claude")
        request = ExecutionRequest(
            case=case,
            prompt="hello",
            prompt_path=tmp_path / "prompt.txt",
            case_dir=tmp_path / "case",
            run_dir=tmp_path / "run",
            artifact_dir=tmp_path / "artifacts",
            op_plugin_dir=op_plugin_root,
            prepared_repos=prepared,
        )
        request.case_dir.mkdir(parents=True, exist_ok=True)
        result = executor.run(request)

        assert result.returncode == 0
        assert isinstance(captured["env"], dict)
        assert "CODEX_SANDBOX_NETWORK_DISABLED" not in captured["env"]
        assert captured["stdin"] == subprocess.PIPE
        assert captured["prompt"] == "hello"
        assert captured["timeout"] == case.timeout_sec
        assert "hello" not in captured["command"]
        assert result.last_message_path is not None
        assert result.last_message_path.read_text(encoding="utf-8") == '{"result":"ok"}\n'
    finally:
        monkeypatch.undo()
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_mindspore_repo_defaults_to_ms_root(tmp_path: Path):
    ms_root = tmp_path / "mindspore"
    init_repo(ms_root, {"tracked.txt": "base\n"})
    repo_spec = RepoSpec(name="mindspore")
    assert resolve_repo_source(repo_spec, ms_root=ms_root, path_root=tmp_path) == ms_root.resolve()


def test_stage_skills_copies_into_isolated_mindspore_checkout(tmp_path: Path):
    repo_root = tmp_path / "mindspore"
    skill_root = tmp_path / "skills"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    (skill_root / "aclnn-builder").mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder" / "SKILL.md").write_text("skill\n", encoding="utf-8")
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="skills_case", prompt="test", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        stage_skills_for_case(prepared, skill_root)
        copied = prepared["mindspore"].checkout_path / ".codex" / "skills" / "aclnn-builder" / "SKILL.md"
        assert copied.read_text(encoding="utf-8") == "skill\n"
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_stage_skills_merges_into_existing_target_skill_folder(tmp_path: Path):
    repo_root = tmp_path / "mindspore"
    skill_root = tmp_path / "skills"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    (skill_root / "aclnn-builder").mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder" / "SKILL.md").write_text("skill\n", encoding="utf-8")
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="skills_case", prompt="test", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        existing = prepared["mindspore"].checkout_path / ".codex" / "skills" / "existing-skill" / "SKILL.md"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("existing\n", encoding="utf-8")

        stage_skills_for_case(prepared, skill_root)

        copied = prepared["mindspore"].checkout_path / ".codex" / "skills" / "aclnn-builder" / "SKILL.md"
        assert copied.read_text(encoding="utf-8") == "skill\n"
        assert existing.read_text(encoding="utf-8") == "existing\n"
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_stage_skills_uses_executor_specific_install_directory(tmp_path: Path):
    repo_root = tmp_path / "mindspore"
    skill_root = tmp_path / "skills"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    (skill_root / "aclnn-builder").mkdir(parents=True, exist_ok=True)
    (skill_root / "aclnn-builder" / "SKILL.md").write_text("skill\n", encoding="utf-8")
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="skills_case", prompt="test", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        stage_skills_for_case(prepared, skill_root, skill_install_subdir=(".claude", "skills"))
        copied = prepared["mindspore"].checkout_path / ".claude" / "skills" / "aclnn-builder" / "SKILL.md"
        assert copied.read_text(encoding="utf-8") == "skill\n"
        assert not (prepared["mindspore"].checkout_path / ".codex" / "skills").exists()
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_resolve_skill_path_defaults_to_ms_root_codex_skills(tmp_path: Path):
    ms_root = tmp_path / "mindspore"
    skill_root = ms_root / ".codex" / "skills"
    skill_root.mkdir(parents=True, exist_ok=True)
    assert resolve_skill_path(None, ms_root=ms_root, path_root=tmp_path) == skill_root.resolve()


def test_resolve_skill_path_prefers_working_dir_default(tmp_path: Path):
    path_root = tmp_path / "op-agent"
    ms_root = tmp_path / "mindspore"
    path_root_skill_root = path_root / ".codex" / "skills"
    ms_root_skill_root = ms_root / ".codex" / "skills"
    path_root_skill_root.mkdir(parents=True, exist_ok=True)
    ms_root_skill_root.mkdir(parents=True, exist_ok=True)

    assert resolve_skill_path(None, ms_root=ms_root, path_root=path_root) == path_root_skill_root.resolve()


def test_resolve_skill_path_uses_executor_specific_default(tmp_path: Path):
    ms_root = tmp_path / "mindspore"
    skill_root = ms_root / ".claude" / "skills"
    skill_root.mkdir(parents=True, exist_ok=True)
    assert (
        resolve_skill_path(
            None,
            ms_root=ms_root,
            path_root=tmp_path,
            default_skill_subdir=skill_install_subdir_for_executor("claudecode"),
        )
        == skill_root.resolve()
    )


def test_resolve_skill_path_errors_when_default_missing(tmp_path: Path):
    ms_root = tmp_path / "mindspore"
    ms_root.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ManifestError):
        resolve_skill_path(None, ms_root=ms_root, path_root=tmp_path)


def test_resolve_op_plugin_path_returns_none_when_not_provided():
    assert resolve_op_plugin_path(None, path_root=Path.cwd()) is None


def test_resolve_op_plugin_path_resolves_relative_path(tmp_path: Path):
    op_plugin_root = tmp_path / "op-plugin"
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    assert resolve_op_plugin_path(Path("op-plugin"), path_root=tmp_path) == op_plugin_root.resolve()


def test_render_prompt_requires_op_plugin_when_placeholder_is_used(tmp_path: Path):
    repo_root = tmp_path / "repo"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="prompt_case", prompt="op={{op_plugin_dir}}", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    try:
        with pytest.raises(ManifestError):
            render_prompt(
                case.prompt,
                prepared_repos=prepared,
                artifact_dir=tmp_path / "artifacts",
                case_dir=tmp_path / "case",
                run_dir=tmp_path / "run",
                op_plugin_dir=None,
            )
    finally:
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)


def test_find_codex_session_task_window_matches_by_cwd_and_time(tmp_path: Path):
    sessions_root = tmp_path / "sessions" / "2026" / "04" / "09"
    sessions_root.mkdir(parents=True, exist_ok=True)
    session_path = sessions_root / "rollout-2026-04-09T10-48-17-example.jsonl"
    session_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-04-09T02:48:17.368Z",
                        "type": "session_meta",
                        "payload": {"cwd": "/tmp/workspace/repo"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-09T02:49:25.908Z",
                        "type": "event_msg",
                        "payload": {"type": "task_started", "turn_id": "turn-1"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-09T03:13:01.827Z",
                        "type": "event_msg",
                        "payload": {"type": "task_complete", "turn_id": "turn-1"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    started_at = datetime.fromisoformat("2026-04-09T02:49:25.908+00:00").timestamp()
    finished_at = datetime.fromisoformat("2026-04-09T03:13:01.827+00:00").timestamp()
    window = find_codex_session_task_window(
        started_at=started_at,
        finished_at=finished_at,
        cwd=Path("/tmp/workspace/repo"),
        sessions_root=tmp_path / "sessions",
    )

    assert window is not None
    assert window.session_path == session_path
    assert window.turn_id == "turn-1"


def test_write_task_session_slice_and_normalize_activity_summary(tmp_path: Path):
    session_path = tmp_path / "rollout.jsonl"
    session_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-04-09T02:48:17.368Z",
                        "type": "session_meta",
                        "payload": {"cwd": "/tmp/workspace/repo"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-09T02:49:25.908Z",
                        "type": "event_msg",
                        "payload": {"type": "task_started", "turn_id": "turn-1"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-09T02:50:00.000Z",
                        "type": "event_msg",
                        "payload": {"type": "agent_message", "message": "phase one"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-09T03:13:01.827Z",
                        "type": "event_msg",
                        "payload": {"type": "task_complete", "turn_id": "turn-1"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-09T03:20:00.000Z",
                        "type": "event_msg",
                        "payload": {"type": "agent_message", "message": "outside window"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    window = SessionTaskWindow(
        session_path=session_path,
        turn_id="turn-1",
        started_at="2026-04-09T02:49:25.908Z",
        finished_at="2026-04-09T03:13:01.827Z",
    )
    slice_path = write_task_session_slice(window, tmp_path / "activity_timeline_source.jsonl")
    sliced = slice_path.read_text(encoding="utf-8")
    assert "session_meta" in sliced
    assert "phase one" in sliced
    assert "outside window" not in sliced

    normalized = normalize_activity_timeline_summary(
        {
            "activity_timeline": [
                {
                    "start": "2026-04-09T02:49:25.908Z",
                    "end": "2026-04-09T02:56:42.046Z",
                    "summary": "Pre-checks and strategy selection.",
                    "key_decision": "Use a customize path.",
                }
            ]
        },
        window=window,
        source_path=slice_path,
    )
    assert normalized["source_task_turn_id"] == "turn-1"
    assert normalized["source_window"]["elapsed_time"] == format_elapsed_minutes(1415.919)
    assert normalized["activity_timeline"][0]["elapsed_time"] == format_elapsed_minutes(436.138)
    assert normalized["activity_timeline"][0]["key_decision"] == "Use a customize path."


def test_activity_timeline_schema_requires_nullable_key_decision():
    schema = build_activity_timeline_output_schema()
    items = schema["properties"]["activity_timeline"]["items"]

    assert "key_decision" in items["required"]
    assert items["properties"]["key_decision"]["type"] == ["string", "null"]


def test_normalize_activity_summary_omits_null_key_decision(tmp_path: Path):
    session_path = tmp_path / "rollout.jsonl"
    session_path.write_text("", encoding="utf-8")
    window = SessionTaskWindow(
        session_path=session_path,
        turn_id="turn-1",
        started_at="2026-04-09T02:49:25.908Z",
        finished_at="2026-04-09T03:13:01.827Z",
    )

    normalized = normalize_activity_timeline_summary(
        {
            "activity_timeline": [
                {
                    "start": "2026-04-09T02:49:25.908Z",
                    "end": "2026-04-09T02:56:42.046Z",
                    "summary": "Pre-checks and strategy selection.",
                    "key_decision": None,
                }
            ]
        },
        window=window,
        source_path=tmp_path / "activity_timeline_source.jsonl",
    )

    assert "key_decision" not in normalized["activity_timeline"][0]


def test_activity_timeline_success_cleans_intermediate_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "repo"
    op_plugin_root = tmp_path / "op-plugin"
    init_repo(repo_root, {"tracked.txt": "base\n"})
    op_plugin_root.mkdir(parents=True, exist_ok=True)
    repo_spec = RepoSpec(name="mindspore", source=str(repo_root))
    case = CaseSpec(case_id="cmd_case", prompt="test", repos=(repo_spec,))
    prepared = prepare_repos(case, tmp_path / "sandbox", ms_root=repo_root, path_root=tmp_path)
    session_path = tmp_path / "rollout.jsonl"
    session_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-04-09T02:48:17.368Z",
                        "type": "session_meta",
                        "payload": {"cwd": str(prepared["mindspore"].checkout_path)},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-09T02:49:25.908Z",
                        "type": "event_msg",
                        "payload": {"type": "task_started", "turn_id": "turn-1"},
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-04-09T03:13:01.827Z",
                        "type": "event_msg",
                        "payload": {"type": "task_complete", "turn_id": "turn-1"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    window = SessionTaskWindow(
        session_path=session_path,
        turn_id="turn-1",
        started_at="2026-04-09T02:49:25.908Z",
        finished_at="2026-04-09T03:13:01.827Z",
    )
    monkeypatch.setattr(framework, "find_codex_session_task_window", lambda **_: window)

    class FakePopen:
        def __init__(self, *args, **kwargs) -> None:
            self.returncode = 0

        def communicate(self, prompt: str, timeout: int | None = None) -> tuple[str, str]:
            raw_path = tmp_path / "case" / "activity_timeline_summary_raw.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                json.dumps(
                    {
                        "activity_timeline": [
                            {
                                "start": "2026-04-09T02:49:25.908Z",
                                "end": "2026-04-09T02:56:42.046Z",
                                "summary": "Pre-checks and strategy selection.",
                                "key_decision": None,
                            },
                            {
                                "start": "2026-04-09T02:56:42.046Z",
                                "end": "2026-04-09T03:10:24.000Z",
                                "summary": "Implementation work.",
                                "key_decision": "Use the generated primitive path.",
                            },
                            {
                                "start": "2026-04-09T03:10:24.000Z",
                                "end": "2026-04-09T03:13:01.827Z",
                                "summary": "Verification and close-out.",
                                "key_decision": None,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return "", ""

    monkeypatch.setattr(framework.subprocess, "Popen", FakePopen)

    try:
        executor = CodexExecutor(codex_bin="codex", analyze_time=True)
        request = ExecutionRequest(
            case=case,
            prompt="hello",
            prompt_path=tmp_path / "prompt.txt",
            case_dir=tmp_path / "case",
            run_dir=tmp_path / "run",
            artifact_dir=tmp_path / "artifacts",
            op_plugin_dir=op_plugin_root,
            prepared_repos=prepared,
        )
        session_log_path, turn_id, source_path, summary_path = executor.maybe_generate_activity_timeline_summary(
            request=request,
            started_at=0.0,
            finished_at=1.0,
        )
        assert session_log_path == session_path
        assert turn_id == "turn-1"
        assert source_path is not None and source_path.exists()
        assert summary_path is not None and summary_path.exists()
        assert not (request.case_dir / "activity_timeline_summary_schema.json").exists()
        assert not (request.case_dir / "activity_timeline_summary_raw.json").exists()
        assert not (request.case_dir / "activity_timeline_summary_events.jsonl").exists()
        assert not (request.case_dir / "activity_timeline_summary_stderr.txt").exists()
    finally:
        monkeypatch.undo()
        from framework import cleanup_prepared_repos

        cleanup_prepared_repos(prepared)
