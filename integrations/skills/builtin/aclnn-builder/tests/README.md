# ACLNN Builder Test Framework

This directory contains the runtime test framework for `skills/aclnn-builder`.

## What It Tests

Each test case defines:

- a prompt that triggers the skill
- one or more source repositories that must be isolated for the run
- the exact source files expected to be `added`, `modified`, or `deleted` when the task finishes
- optional artifact globs for markdown or trace files that are allowed to exist without being treated as source changes

The framework is aimed at end-to-end skill validation:

- a fresh executor process is used for every case
- every repo fixture is isolated per case using `git worktree` by default
- file-change validation is based on the isolated repo diff after the run
- non-committed analysis artifacts are captured separately from source changes

## Execution Model

The runner lives in `framework.py` and the CLI entrypoint is `run_skill_cases.py`.

Per case, the runner:

1. creates a new case directory under `aclnn_skill_test_runs/<run_id>/<case_id>/`
2. prepares isolated checkouts for each declared repo
3. renders the case prompt with placeholders such as `{{repo:mindspore}}`, `{{artifact_dir}}`, and `{{skill:aclnn_builder}}`
4. invokes the selected executor in an isolated session
5. collects the git diff for each isolated repo
6. validates the actual `added/modified/deleted` source files against the case contract
7. writes JSON summaries and raw executor logs into the case directory

This isolation model keeps both workspace state and model session context clean across consecutive runs.
Use `--workers N` to run multiple cases concurrently; result aggregation happens after all workers finish.

## Case Contract

The manifest lives in `cases.yaml`.

Important fields:

- `prompt`: the exact task prompt
- `{{skill:<name>}}`: skill invocation placeholder; renders as `$<name>` for Codex and `/<name>` for OpenCode/Claude Code
- `repos[*].source`: source repo path; for the primary `mindspore` repo this may be omitted and will default to the invocation cwd or `--ms-root`
- `repos[*].expected_changes`: optional source file contract; every listed path must appear in the matching add/modify/delete bucket unless `--skip-expected-changes` is passed, while extra diff entries are allowed
- `artifact_globs`: files that may appear but should not count as source changes
- `enabled: false`: useful for templates or expensive live cases that should not run by default

## Usage

By default, invoke the runner from the MindSpore repository root. If the current working directory is not the MindSpore repo, pass `--ms-root /path/to/mindspore`.
If you do not pass `--runs-root`, outputs go under `<ms_root>/aclnn_skill_test_runs`.
If you do not pass `--skill-path`, the runner searches for skills under the selected executor's default project skill directory in the invocation cwd first, then under `--ms-root`; it fails if neither directory exists.
Before invoking the selected executor, the runner copies the resolved skills tree into the isolated MindSpore checkout under the selected executor's project skill directory.
Use `--op-plugin` to pass the `op-plugin` repository path. In prompts, `{{op_plugin_dir}}` is replaced with that resolved path. `--op-plugin` is required for non-dry-run execution.
Use `--executor` to choose `codex` (default), `opencode`, or `claudecode`.
Use `--workers` to control case-level parallelism; the default is `1`.
A typical run of all test cases:

```bash
python <op_agent_root>/skills/aclnn-builder/tests/run_skill_cases.py --op-plugin=<op_plugin_dir> --workers 4
```

Dry-run prompt rendering and sandbox setup:

```bash
python3 <op_agent_root>/skills/aclnn-builder/tests/run_skill_cases.py --dry-run --include-disabled
```

Run one live case:

```bash
python3 <op_agent_root>/skills/aclnn-builder/tests/run_skill_cases.py --case example_aclnn_abs --include-disabled
```

Run from another working directory:

```bash
python3 <op_agent_root>/skills/aclnn-builder/tests/run_skill_cases.py \
  --executor codex \
  --ms-root /path/to/mindspore \
  --skill-path /path/to/mindspore/.codex/skills \
  --op-plugin /path/to/op-plugin \
  --case example_aclnn_abs \
  --include-disabled
```

By default, isolated worktrees are kept for manual inspection. Clean them up automatically with:

```bash
python3 <op_agent_root>/skills/aclnn-builder/tests/run_skill_cases.py --cleanup-sandboxes
```

## CLI Arguments

The CLI is defined in `framework.py` and exposed through `run_skill_cases.py`.

| Argument | Default | Description |
| --- | --- | --- |
| `--manifest PATH` | `cases.yaml` next to `framework.py` | YAML manifest to load as test input. |
| `--executor {codex,opencode,claudecode}` | `codex` | Agent runner used for each case. Codex uses `$<skill>` triggers; OpenCode and Claude Code use `/<skill>` triggers. |
| `--ms-root PATH` | current working directory | MindSpore repo root. Used as the default source for the `mindspore` repo and as the default parent for outputs and skill lookup. |
| `--skill-path PATH` | executor-specific project skill directory under invocation cwd, then `<ms_root>` | Skills directory copied into each isolated MindSpore checkout before execution. Defaults to `.codex/skills` for Codex, `.opencode/skills` for OpenCode, and `.claude/skills` for Claude Code. Relative paths resolve from the invocation cwd. |
| `--op-plugin PATH` | none | `op-plugin` repo directory. Required for non-dry-run execution and substituted into prompts via `{{op_plugin_dir}}`. Relative paths resolve from the invocation cwd. |
| `--runs-root PATH` | `<ms_root>/aclnn_skill_test_runs` | Directory where run output directories are created. |
| `--run-id ID` | timestamp plus random suffix | Name of the run directory under `--runs-root`. If omitted, a unique id is generated. |
| `--workers N` | `1` | Number of cases to run concurrently. Must be a positive integer. |
| `--case CASE_ID` | all selected cases | Run only the named case. Repeat the argument to run multiple cases. |
| `--include-disabled` | disabled | Include cases whose manifest entry has `enabled: false`. |
| `--dry-run` | disabled | Prepare repos, stage skills, render prompts, and validate setup without invoking an agent. Does not require `--op-plugin` unless the prompt renders `{{op_plugin_dir}}`. |
| `--skip-expected-changes` | disabled | Collect generated file changes but skip comparing them with manifest `expected_changes`. This allows cases to omit `expected_changes`. Executor failures and timeouts still fail the case. |
| `--cleanup-sandboxes` | disabled | Remove isolated repo sandboxes after each case finishes. |
| `--model MODEL` | none | Optional model name passed to the selected executor. |
| `--sandbox MODE` | `workspace-write` | Codex sandbox mode passed to `codex exec`; only used by the Codex executor. |
| `--extra-codex-arg ARG` | none | Raw argument appended to the Codex command. Repeat for multiple Codex-only arguments. |
| `--analyze-time` | disabled | For Codex runs, find the matching Codex session log after each case and write an activity timeline summary into the case directory. |

Executor-specific skill staging:

| Executor | Default `--skill-path` search order | Isolated checkout destination |
| --- | --- | --- |
| `codex` | `<cwd>/.codex/skills`, then `<ms_root>/.codex/skills` | `.codex/skills` |
| `opencode` | `<cwd>/.opencode/skills`, then `<ms_root>/.opencode/skills` | `.opencode/skills` |
| `claudecode` | `<cwd>/.claude/skills`, then `<ms_root>/.claude/skills` | `.claude/skills` |

## Manifest Input

`--manifest` points to the YAML input file consumed by `framework.py`. The default is `skills/aclnn-builder/tests/cases.yaml`.

Top-level fields:

| Field | Required | Description |
| --- | --- | --- |
| `schema_version` | yes | Must be `"1.0.0"`. |
| `defaults.artifact_globs` | no | Default glob patterns for files that may appear in repo diffs but should be reported as artifacts instead of source changes. |
| `cases` | yes | List of case definitions. |

Case fields:

| Field | Required | Description |
| --- | --- | --- |
| `id` | yes | Unique case id. This is the value accepted by `--case` and the name of the case output directory. |
| `prompt` | yes | Prompt template sent to the executor after placeholders are rendered. Empty prompts are rejected. |
| `description` | no | Human-readable notes for maintainers. |
| `enabled` | no | Defaults to `true`. Disabled cases are skipped unless `--include-disabled` is passed. |
| `timeout_sec` | no | Per-case executor timeout in seconds. Defaults to `1800` and must be positive. |
| `repos` | no | Repositories prepared for the case. Each repo gets an isolated checkout under the case sandbox. |

Repo fields:

| Field | Required | Description |
| --- | --- | --- |
| `name` | yes | Repo placeholder name, such as `mindspore`. Prompts can reference it as `{{repo:<name>}}`. |
| `source` | no for `mindspore`, yes for others | Source repo path. For `mindspore`, empty, `{{ms_root}}`, `$MS_ROOT`, or `${MS_ROOT}` resolves to `--ms-root`. Other repos must provide a path. Relative paths resolve from the invocation cwd. |
| `ref` | no | Git ref used when creating a `git-worktree` checkout. Defaults to `HEAD`. |
| `isolation` | no | `git-worktree` or `copy`. Defaults to `git-worktree`. |
| `artifact_globs` | no | Repo-specific artifact globs. If omitted, `defaults.artifact_globs` is used. |
| `expected_changes.added` | no | Source paths expected to be added. |
| `expected_changes.modified` | no | Source paths expected to be modified. |
| `expected_changes.deleted` | no | Source paths expected to be deleted. |

Prompt placeholders:

| Placeholder | Renders To |
| --- | --- |
| `{{repo:<name>}}` | Isolated checkout path for the named repo. |
| `{{artifact_dir}}` | Per-case artifact directory. |
| `{{case_dir}}` | Per-case output directory. |
| `{{run_dir}}` | Current run directory. |
| `{{op_plugin_dir}}` | Resolved `--op-plugin` path. |
| `{{skill:<name>}}` | `$<name>` for Codex or `/<name>` for OpenCode and Claude Code. |

Validation requires every listed expected path to appear in the matching git diff bucket unless `--skip-expected-changes` is passed. Extra source changes are reported in `summary.json` but are not currently treated as validation failures. Paths cannot be repeated across `added`, `modified`, and `deleted` for the same repo.

## Plan

The framework is intentionally split into layers:

- manifest contract: stable, reviewable case definitions in YAML
- isolation layer: fresh worktrees and a fresh executor session per case
- execution layer: pluggable executor, with `CodexExecutor` as the default
- validation layer: exact source diff matching plus separate artifact capture
- unit-test layer: local pytest coverage that exercises the framework without a live model call

This is enough to start adding real `aclnn-builder` cases while keeping the harness deterministic and cheap to validate locally.
