from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


TESTS_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = TESTS_DIR / "cases.yaml"
PROMPT_PLACEHOLDER = re.compile(r"\{\{\s*(repo:([A-Za-z0-9_.-]+)|artifact_dir|case_dir|run_dir)\s*\}\}")
SKILL_PLACEHOLDER = re.compile(r"\{\{\s*skill:([A-Za-z0-9_.-]+)\s*\}\}")
MS_ROOT_TOKENS = {"", "{{ms_root}}", "$MS_ROOT", "${MS_ROOT}"}
PROMPT_OP_PLUGIN_TOKEN = "{{op_plugin_dir}}"
CODEX_SKILL_TRIGGER_PREFIX = "$"
SLASH_SKILL_TRIGGER_PREFIX = "/"
CODEX_SKILLS_SUBDIR = (".codex", "skills")
OPENCODE_SKILLS_SUBDIR = (".opencode", "skills")
CLAUDECODE_SKILLS_SUBDIR = (".claude", "skills")
CODEX_SESSION_MATCH_TOLERANCE_SEC = 300.0
ACTIVITY_TIMELINE_SUMMARY_MIN_PHASES = 3
ACTIVITY_TIMELINE_SUMMARY_MAX_PHASES = 6


class ManifestError(ValueError):
    pass


class Executor(Protocol):
    executor_name: str
    skill_install_subdir: tuple[str, ...]
    skill_trigger_prefix: str

    def run(self, request: "ExecutionRequest") -> "ExecutionResult":
        ...


def log_progress(message: str) -> None:
    print(f"[aclnn-builder-test] {message}", file=sys.stderr, flush=True)


def _parse_iso8601_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _seconds_between(start: str, end: str) -> float:
    return max((_parse_iso8601_timestamp(end) - _parse_iso8601_timestamp(start)).total_seconds(), 0.0)


def format_elapsed_minutes(seconds: float) -> str:
    return f"{max(seconds, 0.0) / 60.0:.5f}min"


def _remove_file_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return


def resolve_codex_sessions_root() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "sessions"
    return Path.home() / ".codex" / "sessions"


def _iter_codex_session_files(
    sessions_root: Path,
    started_at: float,
    finished_at: float,
) -> Iterable[Path]:
    utc_days: set[tuple[int, int, int]] = set()
    for offset in (-1, 0, 1):
        day = datetime.fromtimestamp(started_at, tz=timezone.utc) + timedelta(days=offset)
        utc_days.add((day.year, day.month, day.day))
        day = datetime.fromtimestamp(finished_at, tz=timezone.utc) + timedelta(days=offset)
        utc_days.add((day.year, day.month, day.day))
    for year, month, day in sorted(utc_days, reverse=True):
        day_dir = sessions_root / f"{year:04d}" / f"{month:02d}" / f"{day:02d}"
        if not day_dir.exists():
            continue
        for path in sorted(day_dir.glob("rollout-*.jsonl"), key=lambda item: item.stat().st_mtime, reverse=True):
            yield path


def _paths_match(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left) == str(right)


@dataclass(frozen=True)
class SessionTaskWindow:
    session_path: Path
    turn_id: str
    started_at: str
    finished_at: str


def find_codex_session_task_window(
    started_at: float,
    finished_at: float,
    cwd: Path,
    sessions_root: Path | None = None,
) -> SessionTaskWindow | None:
    root = (sessions_root or resolve_codex_sessions_root()).resolve()
    if not root.exists():
        return None

    best_match: tuple[float, SessionTaskWindow] | None = None
    for session_path in _iter_codex_session_files(root, started_at=started_at, finished_at=finished_at):
        try:
            with session_path.open("r", encoding="utf-8") as handle:
                first_line = handle.readline()
                if not first_line:
                    continue
                session_meta = json.loads(first_line)
                session_cwd = session_meta.get("payload", {}).get("cwd", "")
                if session_cwd and not _paths_match(session_cwd, cwd):
                    continue

                task_started: dict[str, str] = {}
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("type") != "event_msg":
                        continue
                    payload = record.get("payload", {})
                    payload_type = payload.get("type")
                    turn_id = str(payload.get("turn_id", "")).strip()
                    timestamp = str(record.get("timestamp", "")).strip()
                    if not turn_id or not timestamp:
                        continue
                    if payload_type == "task_started":
                        task_started[turn_id] = timestamp
                        continue
                    if payload_type != "task_complete" or turn_id not in task_started:
                        continue
                    task_start = task_started[turn_id]
                    task_start_sec = _parse_iso8601_timestamp(task_start).timestamp()
                    task_end_sec = _parse_iso8601_timestamp(timestamp).timestamp()
                    overlap = min(task_end_sec, finished_at) - max(task_start_sec, started_at)
                    if overlap < -CODEX_SESSION_MATCH_TOLERANCE_SEC:
                        continue
                    score = abs(task_start_sec - started_at) + abs(task_end_sec - finished_at)
                    window = SessionTaskWindow(
                        session_path=session_path,
                        turn_id=turn_id,
                        started_at=task_start,
                        finished_at=timestamp,
                    )
                    if best_match is None or score < best_match[0]:
                        best_match = (score, window)
        except (OSError, json.JSONDecodeError, ValueError):
            continue

    return None if best_match is None else best_match[1]


def write_task_session_slice(window: SessionTaskWindow, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with window.session_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "session_meta":
                dst.write(line)
                continue
            timestamp = str(record.get("timestamp", "")).strip()
            if window.started_at <= timestamp <= window.finished_at:
                dst.write(line)
    return output_path


def build_activity_timeline_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "activity_timeline": {
                "type": "array",
                "minItems": ACTIVITY_TIMELINE_SUMMARY_MIN_PHASES,
                "maxItems": ACTIVITY_TIMELINE_SUMMARY_MAX_PHASES,
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string"},
                        "end": {"type": "string"},
                        "summary": {"type": "string"},
                        "key_decision": {"type": ["string", "null"]},
                    },
                    "required": ["start", "end", "summary", "key_decision"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["activity_timeline"],
        "additionalProperties": False,
    }


def build_activity_timeline_summary_prompt(source_path: Path) -> str:
    return f"""
Read the completed Codex task log in `{source_path.name}` and produce a concise activity timeline.

Requirements:
- Output JSON only, matching the provided schema.
- Produce {ACTIVITY_TIMELINE_SUMMARY_MIN_PHASES} to {ACTIVITY_TIMELINE_SUMMARY_MAX_PHASES} chronological activities.
- Use exact `start` and `end` timestamps copied from the log.
- Each `summary` should state what the agent was doing during that span.
- Set `key_decision` to a short string when the span contains a concrete decision, discovery, or strategy shift. Otherwise set it to `null`.
- Base every statement on the log. Do not invent details and do not mention activity outside this file.
- Prefer boundaries around strategy changes, first code edits, major implementation bursts, and verification/close-out.
""".strip()


def normalize_activity_timeline_summary(
    raw_summary: dict[str, Any],
    window: SessionTaskWindow,
    source_path: Path,
) -> dict[str, Any]:
    normalized_items: list[dict[str, Any]] = []
    for item in raw_summary.get("activity_timeline", ()):
        start = str(item["start"]).strip()
        end = str(item["end"]).strip()
        summary = str(item["summary"]).strip()
        normalized: dict[str, Any] = {
            "start": start,
            "end": end,
            "elapsed_time": format_elapsed_minutes(_seconds_between(start, end)),
            "summary": summary,
        }
        key_decision_value = item.get("key_decision")
        key_decision = ""
        if key_decision_value is not None:
            key_decision = str(key_decision_value).strip()
        if key_decision:
            normalized["key_decision"] = key_decision
        normalized_items.append(normalized)

    return {
        "source_session_jsonl": str(window.session_path),
        "source_task_turn_id": window.turn_id,
        "source_window": {
            "start": window.started_at,
            "end": window.finished_at,
            "elapsed_time": format_elapsed_minutes(_seconds_between(window.started_at, window.finished_at)),
        },
        "activity_timeline_source_path": str(source_path),
        "activity_timeline": normalized_items,
    }


@dataclass(frozen=True)
class ExpectedChangeSet:
    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExpectedChangeSet":
        data = data or {}
        return cls(
            added=tuple(sorted(data.get("added", ()))),
            modified=tuple(sorted(data.get("modified", ()))),
            deleted=tuple(sorted(data.get("deleted", ()))),
        )

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "added": list(self.added),
            "modified": list(self.modified),
            "deleted": list(self.deleted),
        }

    def overlaps(self) -> set[str]:
        buckets = {
            "added": set(self.added),
            "modified": set(self.modified),
            "deleted": set(self.deleted),
        }
        overlap = set()
        names = tuple(buckets.keys())
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap |= buckets[left] & buckets[right]
        return overlap


@dataclass(frozen=True)
class RepoSpec:
    name: str
    source: str = ""
    ref: str = "HEAD"
    isolation: str = "git-worktree"
    artifact_globs: tuple[str, ...] = ()
    expected_changes: ExpectedChangeSet = field(default_factory=ExpectedChangeSet)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        default_artifact_globs: Iterable[str],
    ) -> "RepoSpec":
        return cls(
            name=str(data["name"]),
            source=str(data.get("source", "")),
            ref=str(data.get("ref", "HEAD")),
            isolation=str(data.get("isolation", "git-worktree")),
            artifact_globs=tuple(data.get("artifact_globs", tuple(default_artifact_globs))),
            expected_changes=ExpectedChangeSet.from_dict(data.get("expected_changes")),
        )


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    prompt: str
    description: str = ""
    enabled: bool = True
    timeout_sec: int = 1800
    repos: tuple[RepoSpec, ...] = ()

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        default_artifact_globs: Iterable[str],
    ) -> "CaseSpec":
        return cls(
            case_id=str(data["id"]),
            prompt=str(data["prompt"]).strip(),
            description=str(data.get("description", "")),
            enabled=bool(data.get("enabled", True)),
            timeout_sec=int(data.get("timeout_sec", 1800)),
            repos=tuple(
                RepoSpec.from_dict(repo_data, default_artifact_globs)
                for repo_data in data.get("repos", ())
            ),
        )


@dataclass(frozen=True)
class Manifest:
    schema_version: str
    default_artifact_globs: tuple[str, ...]
    cases: tuple[CaseSpec, ...]

    def enabled_cases(self) -> tuple[CaseSpec, ...]:
        return tuple(case for case in self.cases if case.enabled)

    def by_id(self, case_id: str) -> CaseSpec:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise ManifestError(f"Unknown case id: {case_id}")


@dataclass
class PreparedRepo:
    spec: RepoSpec
    source_path: Path
    checkout_path: Path
    cleanup_kind: str


@dataclass(frozen=True)
class ExecutionRequest:
    case: CaseSpec
    prompt: str
    prompt_path: Path
    case_dir: Path
    run_dir: Path
    artifact_dir: Path
    op_plugin_dir: Path | None
    prepared_repos: dict[str, PreparedRepo]


@dataclass(frozen=True)
class ExecutionResult:
    command: tuple[str, ...]
    returncode: int
    started_at: float
    finished_at: float
    timed_out: bool
    stdout_path: Path
    stderr_path: Path
    last_message_path: Path | None
    session_log_path: Path | None = None
    session_task_turn_id: str | None = None
    activity_timeline_source_path: Path | None = None
    activity_timeline_summary_path: Path | None = None


@dataclass(frozen=True)
class RepoOutcome:
    name: str
    expected_changes: ExpectedChangeSet
    source_changes: ExpectedChangeSet
    artifact_changes: tuple[str, ...]
    valid: bool
    problems: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "expected_changes": self.expected_changes.as_dict(),
            "source_changes": self.source_changes.as_dict(),
            "artifact_changes": list(self.artifact_changes),
            "valid": self.valid,
            "problems": list(self.problems),
        }


@dataclass(frozen=True)
class CaseOutcome:
    case_id: str
    valid: bool
    execution_result: ExecutionResult | None
    repo_outcomes: tuple[RepoOutcome, ...]
    case_dir: Path
    prompt_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "valid": self.valid,
            "execution_result": execution_result_to_dict(self.execution_result),
            "repo_outcomes": [repo.as_dict() for repo in self.repo_outcomes],
            "case_dir": str(self.case_dir),
            "prompt_path": str(self.prompt_path),
        }


def execution_result_to_dict(result: ExecutionResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    payload = asdict(result)
    for key in (
        "stdout_path",
        "stderr_path",
        "last_message_path",
        "session_log_path",
        "activity_timeline_source_path",
        "activity_timeline_summary_path",
    ):
        if payload[key] is not None:
            payload[key] = str(payload[key])
    payload["command"] = list(result.command)
    payload["elapsed_time"] = format_elapsed_minutes(result.finished_at - result.started_at)
    return payload


def load_manifest(path: Path = DEFAULT_MANIFEST) -> Manifest:
    raw = load_yaml_document(path.read_text(encoding="utf-8"), source_path=path) or {}
    schema_version = str(raw.get("schema_version", ""))
    defaults = raw.get("defaults", {}) or {}
    default_artifact_globs = tuple(defaults.get("artifact_globs", ()))
    cases = tuple(
        CaseSpec.from_dict(case_data, default_artifact_globs)
        for case_data in raw.get("cases", ())
    )
    manifest = Manifest(
        schema_version=schema_version,
        default_artifact_globs=default_artifact_globs,
        cases=cases,
    )
    validate_manifest(manifest, path)
    return manifest


def load_yaml_document(text: str, source_path: Path | None = None) -> dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text)

    fallback_python = shutil.which("python")
    if not fallback_python:
        raise ManifestError(
            f"PyYAML is unavailable in the current interpreter and no fallback `python` executable was found for {source_path}"
        )

    helper = subprocess.run(
        [
            fallback_python,
            "-c",
            (
                "import json, sys, yaml; "
                "data = yaml.safe_load(sys.stdin.read()); "
                "json.dump(data, sys.stdout)"
            ),
        ],
        input=text,
        text=True,
        capture_output=True,
    )
    if helper.returncode != 0:
        stderr = helper.stderr.strip() or "unknown error"
        raise ManifestError(f"Failed to parse YAML for {source_path} via fallback python: {stderr}")
    return json.loads(helper.stdout)


def validate_manifest(manifest: Manifest, source_path: Path | None = None) -> None:
    location = f" in {source_path}" if source_path else ""
    if manifest.schema_version != "1.0.0":
        raise ManifestError(f"Unsupported schema_version{location}: {manifest.schema_version!r}")
    case_ids = [case.case_id for case in manifest.cases]
    if len(case_ids) != len(set(case_ids)):
        raise ManifestError(f"Duplicate case ids{location}: {case_ids}")
    for case in manifest.cases:
        if not case.prompt:
            raise ManifestError(f"Case {case.case_id} has an empty prompt{location}")
        if case.timeout_sec <= 0:
            raise ManifestError(f"Case {case.case_id} must use a positive timeout{location}")
        repo_names = [repo.name for repo in case.repos]
        if len(repo_names) != len(set(repo_names)):
            raise ManifestError(f"Case {case.case_id} declares duplicate repo names{location}")
        for repo in case.repos:
            if repo.isolation not in {"git-worktree", "copy"}:
                raise ManifestError(
                    f"Case {case.case_id} repo {repo.name} uses unsupported isolation {repo.isolation!r}{location}"
                )
            overlap = repo.expected_changes.overlaps()
            if overlap:
                raise ManifestError(
                    f"Case {case.case_id} repo {repo.name} repeats paths across status buckets{location}: "
                    f"{sorted(overlap)}"
                )


def render_prompt(
    template: str,
    prepared_repos: dict[str, PreparedRepo],
    artifact_dir: Path,
    case_dir: Path,
    run_dir: Path,
    op_plugin_dir: Path | None,
    skill_trigger_prefix: str = CODEX_SKILL_TRIGGER_PREFIX,
) -> str:
    if PROMPT_OP_PLUGIN_TOKEN in template:
        if op_plugin_dir is None:
            raise ManifestError(
                f"Prompt references {PROMPT_OP_PLUGIN_TOKEN} but no --op-plugin path was provided"
            )
        template = template.replace(PROMPT_OP_PLUGIN_TOKEN, str(op_plugin_dir))

    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        repo_name = match.group(2)
        if repo_name is not None:
            if repo_name not in prepared_repos:
                raise ManifestError(f"Prompt references unknown repo placeholder: {repo_name}")
            return str(prepared_repos[repo_name].checkout_path)
        if token == "artifact_dir":
            return str(artifact_dir)
        if token == "case_dir":
            return str(case_dir)
        if token == "run_dir":
            return str(run_dir)
        raise ManifestError(f"Unsupported prompt placeholder: {token}")

    rendered = PROMPT_PLACEHOLDER.sub(replace, template)
    return render_skill_placeholders(rendered, skill_trigger_prefix=skill_trigger_prefix)


def render_skill_placeholders(template: str, skill_trigger_prefix: str) -> str:
    prefix = skill_trigger_prefix.strip()
    if prefix not in {CODEX_SKILL_TRIGGER_PREFIX, SLASH_SKILL_TRIGGER_PREFIX}:
        raise ManifestError(f"Unsupported skill trigger prefix: {skill_trigger_prefix!r}")

    return SKILL_PLACEHOLDER.sub(lambda match: f"{prefix}{match.group(1)}", template)


def _run_command(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        text=True,
        capture_output=True,
    )


def _external_agent_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CODEX_SANDBOX_NETWORK_DISABLED", None)
    return env


def _is_git_repo(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        _run_command(["git", "rev-parse", "--is-inside-work-tree"], cwd=path)
        return True
    except subprocess.CalledProcessError:
        return False


def _ensure_clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def skill_install_subdir_for_executor(executor_name: str) -> tuple[str, ...]:
    if executor_name == "codex":
        return CODEX_SKILLS_SUBDIR
    if executor_name == "opencode":
        return OPENCODE_SKILLS_SUBDIR
    if executor_name == "claudecode":
        return CLAUDECODE_SKILLS_SUBDIR
    raise ManifestError(f"Unsupported executor: {executor_name}")


def resolve_skill_path(
    skill_path: Path | None,
    ms_root: Path,
    path_root: Path,
    default_skill_subdir: tuple[str, ...] = CODEX_SKILLS_SUBDIR,
) -> Path:
    candidate = skill_path
    if candidate is None:
        default_candidates = (
            path_root.joinpath(*default_skill_subdir),
            ms_root.joinpath(*default_skill_subdir),
        )
        candidate = next((path for path in default_candidates if path.exists()), default_candidates[-1])
        if not candidate.exists():
            searched = ", ".join(str(path) for path in default_candidates)
            raise ManifestError(
                f"Skill path not provided and no default skills directory exists. Searched: {searched}"
            )
    elif not candidate.is_absolute():
        candidate = (path_root / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists():
        raise ManifestError(f"Skill path does not exist: {candidate}")
    if not candidate.is_dir():
        raise ManifestError(f"Skill path is not a directory: {candidate}")
    return candidate


def resolve_op_plugin_path(op_plugin_path: Path | None, path_root: Path) -> Path | None:
    if op_plugin_path is None:
        return None
    candidate = op_plugin_path
    if not candidate.is_absolute():
        candidate = (path_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise ManifestError(f"Op-plugin path does not exist: {candidate}")
    if not candidate.is_dir():
        raise ManifestError(f"Op-plugin path is not a directory: {candidate}")
    return candidate


def resolve_repo_source(repo: RepoSpec, ms_root: Path, path_root: Path) -> Path:
    source = repo.source.strip()
    if source in MS_ROOT_TOKENS:
        if repo.name != "mindspore":
            raise ManifestError(
                f"Repo {repo.name} must declare an explicit source path when it is not the primary `mindspore` repo"
            )
        return ms_root

    source_path = Path(source)
    if source_path.is_absolute():
        return source_path.resolve()
    return (path_root / source_path).resolve()


def prepare_repos(case: CaseSpec, sandbox_root: Path, ms_root: Path, path_root: Path) -> dict[str, PreparedRepo]:
    prepared: dict[str, PreparedRepo] = {}
    repos_root = sandbox_root / "repos"
    repos_root.mkdir(parents=True, exist_ok=True)
    for repo in case.repos:
        source_path = resolve_repo_source(repo, ms_root=ms_root, path_root=path_root)
        if not source_path.exists():
            raise ManifestError(
                f"Case {case.case_id} repo {repo.name} source path does not exist: {source_path}"
            )
        checkout_path = repos_root / repo.name
        if repo.isolation == "git-worktree":
            if not _is_git_repo(source_path):
                raise ManifestError(
                    f"Case {case.case_id} repo {repo.name} source path is not a git repository: {source_path}"
                )
            checkout_path.parent.mkdir(parents=True, exist_ok=True)
            _run_command(
                ["git", "-C", str(source_path), "worktree", "add", "--detach", str(checkout_path), repo.ref],
                cwd=source_path,
            )
            cleanup_kind = "git-worktree"
        else:
            if checkout_path.exists():
                shutil.rmtree(checkout_path)
            shutil.copytree(source_path, checkout_path, symlinks=True)
            cleanup_kind = "copy"
        prepared[repo.name] = PreparedRepo(
            spec=repo,
            source_path=source_path,
            checkout_path=checkout_path,
            cleanup_kind=cleanup_kind,
        )
    return prepared


def cleanup_prepared_repos(prepared: dict[str, PreparedRepo]) -> None:
    for repo in prepared.values():
        if repo.cleanup_kind == "git-worktree":
            subprocess.run(
                ["git", "-C", str(repo.source_path), "worktree", "remove", "--force", str(repo.checkout_path)],
                cwd=repo.source_path,
                check=False,
                text=True,
                capture_output=True,
            )
        elif repo.checkout_path.exists():
            shutil.rmtree(repo.checkout_path, ignore_errors=True)


def stage_skills_for_case(
    prepared: dict[str, PreparedRepo],
    skill_path: Path,
    skill_install_subdir: tuple[str, ...] = CODEX_SKILLS_SUBDIR,
) -> None:
    if "mindspore" not in prepared:
        raise ManifestError("Cannot stage skills because the case does not define a `mindspore` repo")
    invalid_part = any(
        not part or Path(part).is_absolute() or part == ".."
        for part in skill_install_subdir
    )
    if not skill_install_subdir or invalid_part:
        raise ManifestError(f"Invalid skill install subdirectory: {skill_install_subdir!r}")
    target_skill_dir = prepared["mindspore"].checkout_path.joinpath(*skill_install_subdir)
    target_skill_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill_path, target_skill_dir, symlinks=True, dirs_exist_ok=True)


def _collect_git_source_changes(checkout_path: Path) -> list[tuple[str, str]]:
    diff = _run_command(
        ["git", "-C", str(checkout_path), "diff", "--name-status", "--find-renames", "HEAD"],
        cwd=checkout_path,
    ).stdout.splitlines()
    changes: list[tuple[str, str]] = []
    for line in diff:
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) == 3:
            changes.append(("D", parts[1]))
            changes.append(("A", parts[2]))
            continue
        if len(parts) != 2:
            raise ManifestError(f"Unsupported git diff output in {checkout_path}: {line!r}")
        changes.append((status[0], parts[1]))
    untracked = _run_command(
        ["git", "-C", str(checkout_path), "ls-files", "--others", "--exclude-standard"],
        cwd=checkout_path,
    ).stdout.splitlines()
    changes.extend(("A", path) for path in untracked if path)
    return changes


def classify_repo_changes(prepared_repo: PreparedRepo, compare_expected_changes_enabled: bool = True) -> RepoOutcome:
    raw_changes = _collect_git_source_changes(prepared_repo.checkout_path)
    source_buckets = {"A": set(), "M": set(), "D": set()}
    artifact_changes: list[str] = []
    problems: list[str] = []

    for status, path in raw_changes:
        normalized = path.strip()
        is_artifact = any(fnmatch.fnmatch(normalized, pattern) for pattern in prepared_repo.spec.artifact_globs)
        if is_artifact:
            artifact_changes.append(normalized)
            continue
        if status == "A":
            source_buckets["A"].add(normalized)
        elif status == "M":
            source_buckets["M"].add(normalized)
        elif status == "D":
            source_buckets["D"].add(normalized)
        else:
            problems.append(f"Unsupported git change status for {normalized}: {status}")

    actual = ExpectedChangeSet(
        added=tuple(sorted(source_buckets["A"])),
        modified=tuple(sorted(source_buckets["M"])),
        deleted=tuple(sorted(source_buckets["D"])),
    )
    expected = prepared_repo.spec.expected_changes
    if compare_expected_changes_enabled:
        problems.extend(compare_expected_changes(expected, actual))
    return RepoOutcome(
        name=prepared_repo.spec.name,
        expected_changes=expected,
        source_changes=actual,
        artifact_changes=tuple(sorted(artifact_changes)),
        valid=not problems,
        problems=tuple(problems),
    )


def compare_expected_changes(expected: ExpectedChangeSet, actual: ExpectedChangeSet) -> list[str]:
    problems: list[str] = []
    for label in ("added", "modified", "deleted"):
        expected_set = set(getattr(expected, label))
        actual_set = set(getattr(actual, label))
        missing = sorted(expected_set - actual_set)
        if missing:
            problems.append(f"Missing {label} paths: {missing}")
    return problems


def execution_cwd_for_request(request: ExecutionRequest) -> Path:
    if "mindspore" in request.prepared_repos:
        return request.prepared_repos["mindspore"].checkout_path
    if request.prepared_repos:
        return next(iter(request.prepared_repos.values())).checkout_path
    return request.case_dir


class CodexExecutor:
    executor_name = "codex"
    skill_install_subdir = CODEX_SKILLS_SUBDIR
    skill_trigger_prefix = CODEX_SKILL_TRIGGER_PREFIX

    def __init__(
        self,
        codex_bin: str = "codex",
        model: str | None = None,
        sandbox: str = "workspace-write",
        full_auto: bool = True,
        ephemeral: bool = False,
        analyze_time: bool = False,
        extra_args: Iterable[str] = (),
    ) -> None:
        self.codex_bin = codex_bin
        self.model = model
        self.sandbox = sandbox
        self.full_auto = full_auto
        self.ephemeral = ephemeral
        self.analyze_time = analyze_time
        self.extra_args = tuple(extra_args)

    @staticmethod
    def execution_cwd_for(request: ExecutionRequest) -> Path:
        return execution_cwd_for_request(request)

    def build_command(self, request: ExecutionRequest) -> list[str]:
        execution_cwd = self.execution_cwd_for(request)
        command = [self.codex_bin, "exec", "--json", "--sandbox", self.sandbox]
        if self.full_auto:
            command.append("--full-auto")
        # Activity timeline analysis needs the main task session persisted to disk.
        if self.ephemeral and not self.analyze_time:
            command.append("--ephemeral")
        if self.model:
            command.extend(["--model", self.model])
        last_message_path = request.case_dir / "last_message.txt"
        command.extend(["--output-last-message", str(last_message_path)])
        command.extend(["--cd", str(execution_cwd)])
        command.extend(["--add-dir", str(request.case_dir)])
        for repo in request.prepared_repos.values():
            command.extend(["--add-dir", str(repo.checkout_path)])
        if request.op_plugin_dir is not None:
            command.extend(["--add-dir", str(request.op_plugin_dir)])
        command.extend(self.extra_args)
        command.append("-")
        return command

    def build_activity_timeline_summary_command(
        self,
        case_dir: Path,
        schema_path: Path,
        last_message_path: Path,
    ) -> list[str]:
        command = [
            self.codex_bin,
            "exec",
            "--json",
            "--sandbox",
            "read-only",
            "--skip-git-repo-check",
        ]
        if self.full_auto:
            command.append("--full-auto")
        if self.ephemeral:
            command.append("--ephemeral")
        if self.model:
            command.extend(["--model", self.model])
        command.extend(["--output-schema", str(schema_path)])
        command.extend(["--output-last-message", str(last_message_path)])
        command.extend(["--cd", str(case_dir)])
        command.extend(["--add-dir", str(case_dir)])
        command.append("-")
        return command

    def maybe_generate_activity_timeline_summary(
        self,
        request: ExecutionRequest,
        started_at: float,
        finished_at: float,
    ) -> tuple[Path | None, str | None, Path | None, Path | None]:
        execution_cwd = self.execution_cwd_for(request)
        session_window = find_codex_session_task_window(
            started_at=started_at,
            finished_at=finished_at,
            cwd=execution_cwd,
        )
        if session_window is None:
            log_progress(f"{request.case.case_id}: no matching Codex session log found for activity timeline summary")
            return None, None, None, None

        source_path = request.case_dir / "activity_timeline_source.jsonl"
        schema_path = request.case_dir / "activity_timeline_summary_schema.json"
        summary_last_message_path = request.case_dir / "activity_timeline_summary_raw.json"
        summary_events_path = request.case_dir / "activity_timeline_summary_events.jsonl"
        summary_stderr_path = request.case_dir / "activity_timeline_summary_stderr.txt"
        summary_output_path = request.case_dir / "activity_timeline_summary.json"

        write_task_session_slice(session_window, source_path)
        schema_path.write_text(
            json.dumps(build_activity_timeline_output_schema(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        prompt = build_activity_timeline_summary_prompt(source_path)
        command = self.build_activity_timeline_summary_command(
            case_dir=request.case_dir,
            schema_path=schema_path,
            last_message_path=summary_last_message_path,
        )
        timeout_sec = min(max(60, request.case.timeout_sec // 6), 300)
        log_progress(f"{request.case.case_id}: generating activity timeline summary from {session_window.session_path}")

        with summary_events_path.open("w", encoding="utf-8") as stdout_fp, summary_stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_fp:
            process = subprocess.Popen(
                command,
                cwd=request.case_dir,
                stdin=subprocess.PIPE,
                stdout=stdout_fp,
                stderr=stderr_fp,
                text=True,
            )
            try:
                process.communicate(prompt, timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                log_progress(f"{request.case.case_id}: activity timeline summary timed out after {timeout_sec}s")
                process.kill()
                process.communicate()
                return session_window.session_path, session_window.turn_id, source_path, None

        if process.returncode != 0:
            log_progress(
                f"{request.case.case_id}: activity timeline summary failed with return code {process.returncode}"
            )
            return session_window.session_path, session_window.turn_id, source_path, None
        if not summary_last_message_path.exists():
            log_progress(f"{request.case.case_id}: activity timeline summary produced no JSON payload")
            return session_window.session_path, session_window.turn_id, source_path, None

        try:
            raw_summary = json.loads(summary_last_message_path.read_text(encoding="utf-8"))
            normalized = normalize_activity_timeline_summary(
                raw_summary,
                window=session_window,
                source_path=source_path,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log_progress(f"{request.case.case_id}: failed to normalize activity timeline summary: {exc}")
            return session_window.session_path, session_window.turn_id, source_path, None

        summary_output_path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
        for intermediate_path in (
            schema_path,
            summary_last_message_path,
            summary_events_path,
            summary_stderr_path,
        ):
            _remove_file_if_exists(intermediate_path)
        return session_window.session_path, session_window.turn_id, source_path, summary_output_path

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        stdout_path = request.case_dir / "codex_events.jsonl"
        stderr_path = request.case_dir / "codex_stderr.txt"
        last_message_path = request.case_dir / "last_message.txt"
        execution_cwd = self.execution_cwd_for(request)
        command = self.build_command(request)
        log_progress(f"{request.case.case_id}: launching codex in {execution_cwd}")
        started_at = time.time()
        timed_out = False
        returncode = 1

        with stdout_path.open("w", encoding="utf-8") as stdout_fp, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_fp:
            process = subprocess.Popen(
                command,
                cwd=execution_cwd,
                stdin=subprocess.PIPE,
                stdout=stdout_fp,
                stderr=stderr_fp,
                text=True,
            )
            try:
                process.communicate(request.prompt, timeout=request.case.timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                log_progress(f"{request.case.case_id}: codex timed out after {request.case.timeout_sec}s, terminating")
                process.kill()
                process.communicate()
            returncode = process.returncode if process.returncode is not None else 1

        finished_at = time.time()
        log_progress(f"{request.case.case_id}: codex finished with return code {returncode}")
        session_log_path = None
        session_task_turn_id = None
        activity_timeline_source_path = None
        activity_timeline_summary_path = None
        if self.analyze_time:
            session_log_path, session_task_turn_id, activity_timeline_source_path, activity_timeline_summary_path = (
                self.maybe_generate_activity_timeline_summary(
                    request=request,
                    started_at=started_at,
                    finished_at=finished_at,
                )
            )
        return ExecutionResult(
            command=tuple(command),
            returncode=returncode,
            started_at=started_at,
            finished_at=finished_at,
            timed_out=timed_out,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            last_message_path=last_message_path if last_message_path.exists() else None,
            session_log_path=session_log_path,
            session_task_turn_id=session_task_turn_id,
            activity_timeline_source_path=activity_timeline_source_path,
            activity_timeline_summary_path=activity_timeline_summary_path,
        )


class OpenCodeExecutor:
    executor_name = "opencode"
    skill_install_subdir = OPENCODE_SKILLS_SUBDIR
    skill_trigger_prefix = SLASH_SKILL_TRIGGER_PREFIX

    def __init__(
        self,
        opencode_bin: str = "opencode",
        model: str | None = None,
    ) -> None:
        self.opencode_bin = opencode_bin
        self.model = model

    @staticmethod
    def execution_cwd_for(request: ExecutionRequest) -> Path:
        return execution_cwd_for_request(request)

    def build_command(self, request: ExecutionRequest) -> list[str]:
        execution_cwd = self.execution_cwd_for(request)
        command = [
            self.opencode_bin,
            "run",
            "--dir",
            str(execution_cwd),
            "--format",
            "json",
        ]
        if self.model:
            command.extend(["--model", self.model])
        command.append(request.prompt)
        return command

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        stdout_path = request.case_dir / "opencode_output.jsonl"
        stderr_path = request.case_dir / "opencode_stderr.txt"
        last_message_path = request.case_dir / "last_message.txt"
        execution_cwd = self.execution_cwd_for(request)
        command = self.build_command(request)
        log_progress(f"{request.case.case_id}: launching opencode in {execution_cwd}")
        started_at = time.time()
        timed_out = False
        returncode = 1

        with stdout_path.open("w", encoding="utf-8") as stdout_fp, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_fp:
            process = subprocess.Popen(
                command,
                cwd=execution_cwd,
                env=_external_agent_env(),
                stdout=stdout_fp,
                stderr=stderr_fp,
                text=True,
            )
            try:
                process.communicate(timeout=request.case.timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                log_progress(f"{request.case.case_id}: opencode timed out after {request.case.timeout_sec}s, terminating")
                process.kill()
                process.communicate()
            returncode = process.returncode if process.returncode is not None else 1

        if stdout_path.exists():
            last_message_path.write_text(stdout_path.read_text(encoding="utf-8"), encoding="utf-8")
        finished_at = time.time()
        log_progress(f"{request.case.case_id}: opencode finished with return code {returncode}")
        return ExecutionResult(
            command=tuple(command),
            returncode=returncode,
            started_at=started_at,
            finished_at=finished_at,
            timed_out=timed_out,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            last_message_path=last_message_path if last_message_path.exists() else None,
        )


class ClaudeCodeExecutor:
    executor_name = "claudecode"
    skill_install_subdir = CLAUDECODE_SKILLS_SUBDIR
    skill_trigger_prefix = SLASH_SKILL_TRIGGER_PREFIX

    def __init__(
        self,
        claudecode_bin: str = "claude",
        model: str | None = None,
    ) -> None:
        self.claudecode_bin = claudecode_bin
        self.model = model

    @staticmethod
    def execution_cwd_for(request: ExecutionRequest) -> Path:
        return execution_cwd_for_request(request)

    def build_command(self, request: ExecutionRequest) -> list[str]:
        execution_cwd = self.execution_cwd_for(request)
        command = [
            self.claudecode_bin,
            "-p",
            "--output-format",
            "json",
            "--permission-mode",
            "bypassPermissions",
        ]
        if self.model:
            command.extend(["--model", self.model])
        command.extend(["--add-dir", str(request.case_dir)])
        for repo in request.prepared_repos.values():
            command.extend(["--add-dir", str(repo.checkout_path)])
        if request.op_plugin_dir is not None:
            command.extend(["--add-dir", str(request.op_plugin_dir)])
        return command

    def run(self, request: ExecutionRequest) -> ExecutionResult:
        stdout_path = request.case_dir / "claudecode_output.json"
        stderr_path = request.case_dir / "claudecode_stderr.txt"
        last_message_path = request.case_dir / "last_message.txt"
        execution_cwd = self.execution_cwd_for(request)
        command = self.build_command(request)
        log_progress(f"{request.case.case_id}: launching claudecode in {execution_cwd}")
        started_at = time.time()
        timed_out = False
        returncode = 1

        with stdout_path.open("w", encoding="utf-8") as stdout_fp, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_fp:
            process = subprocess.Popen(
                command,
                cwd=execution_cwd,
                env=_external_agent_env(),
                stdin=subprocess.PIPE,
                stdout=stdout_fp,
                stderr=stderr_fp,
                text=True,
            )
            try:
                process.communicate(request.prompt, timeout=request.case.timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                log_progress(
                    f"{request.case.case_id}: claudecode timed out after {request.case.timeout_sec}s, terminating"
                )
                process.kill()
                process.communicate()
            returncode = process.returncode if process.returncode is not None else 1

        if stdout_path.exists():
            last_message_path.write_text(stdout_path.read_text(encoding="utf-8"), encoding="utf-8")
        finished_at = time.time()
        log_progress(f"{request.case.case_id}: claudecode finished with return code {returncode}")
        return ExecutionResult(
            command=tuple(command),
            returncode=returncode,
            started_at=started_at,
            finished_at=finished_at,
            timed_out=timed_out,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            last_message_path=last_message_path if last_message_path.exists() else None,
        )


class SkillTestRunner:
    def __init__(
        self,
        manifest: Manifest,
        executor: Executor,
        ms_root: Path,
        path_root: Path,
        skill_path: Path,
        op_plugin_dir: Path | None,
        runs_root: Path,
        keep_sandboxes: bool = True,
        compare_expected_changes_enabled: bool = True,
    ) -> None:
        self.manifest = manifest
        self.executor = executor
        self.ms_root = ms_root
        self.path_root = path_root
        self.skill_path = skill_path
        self.op_plugin_dir = op_plugin_dir
        self.runs_root = runs_root
        self.keep_sandboxes = keep_sandboxes
        self.compare_expected_changes_enabled = compare_expected_changes_enabled
        self._worktree_locks: dict[Path, threading.Lock] = {}
        self._worktree_locks_guard = threading.Lock()

    def select_cases(self, case_ids: Iterable[str] | None = None, include_disabled: bool = False) -> tuple[CaseSpec, ...]:
        if case_ids:
            selected = tuple(self.manifest.by_id(case_id) for case_id in case_ids)
        else:
            selected = self.manifest.cases
        if include_disabled:
            return selected
        return tuple(case for case in selected if case.enabled)

    def _lock_for_case_repos(self, case: CaseSpec) -> tuple[threading.Lock, ...]:
        source_paths = sorted(
            {
                resolve_repo_source(repo, ms_root=self.ms_root, path_root=self.path_root)
                for repo in case.repos
                if repo.isolation == "git-worktree"
            },
            key=str,
        )
        locks: list[threading.Lock] = []
        with self._worktree_locks_guard:
            for source_path in source_paths:
                locks.append(self._worktree_locks.setdefault(source_path, threading.Lock()))
        return tuple(locks)

    def _run_case(self, case: CaseSpec, run_dir: Path, dry_run: bool) -> CaseOutcome:
        case_dir = run_dir / case.case_id
        sandbox_root = case_dir / "sandbox"
        artifact_dir = case_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        repo_locks = self._lock_for_case_repos(case)
        for lock in repo_locks:
            lock.acquire()
        try:
            log_progress(f"{case.case_id}: preparing isolated repositories")
            prepared = prepare_repos(case, sandbox_root, ms_root=self.ms_root, path_root=self.path_root)
        finally:
            for lock in reversed(repo_locks):
                lock.release()

        try:
            skill_install_subdir = getattr(self.executor, "skill_install_subdir", CODEX_SKILLS_SUBDIR)
            skill_install_path = "/".join(skill_install_subdir)
            log_progress(f"{case.case_id}: staging skills into isolated checkout at {skill_install_path}")
            stage_skills_for_case(prepared, self.skill_path, skill_install_subdir=skill_install_subdir)
            log_progress(f"{case.case_id}: rendering prompt")
            prompt = render_prompt(
                case.prompt,
                prepared_repos=prepared,
                artifact_dir=artifact_dir,
                case_dir=case_dir,
                run_dir=run_dir,
                op_plugin_dir=self.op_plugin_dir,
                skill_trigger_prefix=getattr(self.executor, "skill_trigger_prefix", CODEX_SKILL_TRIGGER_PREFIX),
            )
            prompt_path = case_dir / "prompt.txt"
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt, encoding="utf-8")

            if dry_run:
                executor_name = getattr(self.executor, "executor_name", "executor")
                log_progress(f"{case.case_id}: dry-run mode, skipping {executor_name} execution")
            execution_result = None if dry_run else self.executor.run(
                ExecutionRequest(
                    case=case,
                    prompt=prompt,
                    prompt_path=prompt_path,
                    case_dir=case_dir,
                    run_dir=run_dir,
                    artifact_dir=artifact_dir,
                    op_plugin_dir=self.op_plugin_dir,
                    prepared_repos=prepared,
                )
            )
            if self.compare_expected_changes_enabled:
                log_progress(f"{case.case_id}: validating repository changes")
            else:
                log_progress(f"{case.case_id}: collecting repository changes without expected-change comparison")
            repo_outcomes = tuple(
                classify_repo_changes(
                    repo,
                    compare_expected_changes_enabled=self.compare_expected_changes_enabled,
                )
                for repo in prepared.values()
            )
            case_valid = all(repo.valid for repo in repo_outcomes) and (
                execution_result is None
                or (execution_result.returncode == 0 and not execution_result.timed_out)
            )
            outcome = CaseOutcome(
                case_id=case.case_id,
                valid=case_valid,
                execution_result=execution_result,
                repo_outcomes=repo_outcomes,
                case_dir=case_dir,
                prompt_path=prompt_path,
            )
            summary_path = case_dir / "summary.json"
            summary_path.write_text(json.dumps(outcome.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
            log_progress(f"{case.case_id}: completed with status {'passed' if case_valid else 'failed'}")
            return outcome
        finally:
            if not self.keep_sandboxes:
                cleanup_locks = self._lock_for_case_repos(case)
                for lock in cleanup_locks:
                    lock.acquire()
                try:
                    log_progress(f"{case.case_id}: cleaning up isolated repositories")
                    cleanup_prepared_repos(prepared)
                finally:
                    for lock in reversed(cleanup_locks):
                        lock.release()

    def run(
        self,
        case_ids: Iterable[str] | None = None,
        run_id: str | None = None,
        dry_run: bool = False,
        include_disabled: bool = False,
        workers: int = 1,
    ) -> tuple[Path, tuple[CaseOutcome, ...]]:
        run_started_at = time.time()
        run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:6]}"
        run_dir = self.runs_root / run_id
        _ensure_clean_directory(run_dir)
        selected_cases = self.select_cases(case_ids=case_ids, include_disabled=include_disabled)
        outcomes: list[CaseOutcome] = []
        log_progress(f"starting run {run_id} with {len(selected_cases)} case(s) using {workers} worker(s)")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_case_id = {
                executor.submit(self._run_case, case, run_dir, dry_run): case.case_id for case in selected_cases
            }
            for future in as_completed(future_to_case_id):
                outcomes.append(future.result())

        run_finished_at = time.time()
        run_summary = {
            "elapsed_time": format_elapsed_minutes(run_finished_at - run_started_at),
            "run_id": run_id,
            "manifest_schema_version": self.manifest.schema_version,
            "expected_change_comparison_enabled": self.compare_expected_changes_enabled,
            "run_dir": str(run_dir),
            "case_outcomes": [outcome.as_dict() for outcome in outcomes],
        }
        (run_dir / "summary.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8")
        failed_cases = [outcome.case_id for outcome in outcomes if not outcome.valid]
        log_progress(
            f"run {run_id} finished in {run_summary['elapsed_time']}; "
            f"{len(outcomes) - len(failed_cases)} passed, {len(failed_cases)} failed"
        )
        return run_dir, tuple(outcomes)


def build_argument_parser() -> argparse.ArgumentParser:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("--workers must be a positive integer")
        return parsed

    parser = argparse.ArgumentParser(description="Run isolated aclnn-builder skill test cases.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--executor", choices=("codex", "opencode", "claudecode"), default="codex")
    parser.add_argument("--ms-root", type=Path, default=None)
    parser.add_argument("--skill-path", type=Path, default=None)
    parser.add_argument("--op-plugin", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--workers", type=positive_int, default=1)
    parser.add_argument("--case", dest="cases", action="append", default=[])
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-expected-changes",
        action="store_true",
        help="Collect generated file changes but skip comparing them with manifest expected_changes.",
    )
    parser.set_defaults(keep_sandboxes=True)
    parser.add_argument(
        "--cleanup-sandboxes",
        dest="keep_sandboxes",
        action="store_false",
        help="Delete isolated sandboxes after the run completes.",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--sandbox", default="workspace-write")
    parser.add_argument(
        "--extra-codex-arg",
        action="append",
        default=[],
        help="Repeatable raw argument appended to the codex exec command.",
    )
    parser.add_argument(
        "--analyze-time",
        action="store_true",
        help="After each Codex case completes, derive an activity timeline from the full session log.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    ms_root = (args.ms_root or Path.cwd()).resolve()
    path_root = Path.cwd().resolve()
    try:
        default_skill_subdir = skill_install_subdir_for_executor(args.executor)
        skill_path = resolve_skill_path(
            args.skill_path,
            ms_root=ms_root,
            path_root=path_root,
            default_skill_subdir=default_skill_subdir,
        )
        op_plugin_dir = resolve_op_plugin_path(args.op_plugin, path_root=path_root)
        runs_root = (args.runs_root or (ms_root / "aclnn_skill_test_runs")).resolve()
        if not args.dry_run and op_plugin_dir is None:
            raise ManifestError("--op-plugin is required when running non-dry-run cases")
        manifest = load_manifest(args.manifest)
        if args.executor == "codex":
            executor = CodexExecutor(
                model=args.model or None,
                sandbox=args.sandbox,
                analyze_time=args.analyze_time,
                extra_args=args.extra_codex_arg,
            )
        elif args.executor == "opencode":
            executor = OpenCodeExecutor(
                model=args.model or None,
            )
        else:
            executor = ClaudeCodeExecutor(
                model=args.model or None,
            )
        runner = SkillTestRunner(
            manifest=manifest,
            executor=executor,
            ms_root=ms_root,
            path_root=path_root,
            skill_path=skill_path,
            op_plugin_dir=op_plugin_dir,
            runs_root=runs_root,
            keep_sandboxes=args.keep_sandboxes,
            compare_expected_changes_enabled=not args.skip_expected_changes,
        )
        run_dir, outcomes = runner.run(
            case_ids=args.cases or None,
            run_id=args.run_id or None,
            dry_run=args.dry_run,
            include_disabled=args.include_disabled,
            workers=args.workers,
        )
    except ManifestError as exc:
        print(
            json.dumps(
                {
                    "failed_cases": [],
                    "message": str(exc),
                    "run_dir": "",
                    "total_cases": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    payload = {
        "run_dir": str(run_dir),
        "total_cases": len(outcomes),
        "failed_cases": [outcome.case_id for outcome in outcomes if not outcome.valid],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload["failed_cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
