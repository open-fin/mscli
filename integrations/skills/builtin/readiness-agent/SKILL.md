---
name: readiness-agent
description: Check whether a local single-machine workspace is ready to train or run inference, explain what is missing, emit a readiness report, and optionally apply safe user-space fixes. Use for pre-run workspace readiness checks, training or inference preflight, missing-item analysis, or safe environment remediation before execution.
---

# Readiness Agent

You are a readiness diagnosis and repair skill.

Check whether a local single-machine workspace can run training or inference
now, explain what is missing, write a concise readiness report, and in `fix`
mode apply safe user-space repairs.

`readiness-agent` is the canonical public readiness entrypoint. It owns two
modes:

- `readiness-agent --mode check`: guided read-only readiness assessment
- `readiness-agent --mode fix`: explicitly requested safe user-space remediation

## Scope

Use this skill for:

- pre-run training readiness checks
- pre-run inference readiness checks
- workspace missing-item analysis before execution
- safe user-space readiness remediation

Do not use this skill for:

- post-run crashes or tracebacks
- accuracy regressions
- performance tuning
- distributed or multi-node readiness
- system-level driver, firmware, or CANN installation

## Mode Boundaries

### `check`

- Perform guided read-only readiness assessment.
- Discover workspace evidence (target, launcher, framework, Python env, CANN, assets).
- Ask structured confirmation questions when inference is ambiguous.
- Run safe near-launch probes.
- Optionally run `task_smoke_cmd` **only after user confirmation**.
- Produce a readiness verdict: `READY`, `WARN`, or `BLOCKED`.
- Write the canonical readiness lock.
- May return `NEEDS_CONFIRMATION` with a `portable_question`.
- Do not install packages, download assets, create virtual environments, edit
  source files or configs, scaffold files, or write `.readiness.env`.

### `fix`

- Run only when the user explicitly asks for remediation.
- Preserve the existing safe user-space repair semantics.
- May create or reuse workspace-local environments, install user-space packages,
  download explicitly declared assets, and write `.readiness.env` when allowed by
  the existing fix policy.
- Do not return `NEEDS_CONFIRMATION`.
- Reads the latest `workspace-readiness.lock.json` before acting when available.
- Refreshes the lock after successful remediation.

## Structured Confirmation Handling

When `check` returns `status == "NEEDS_CONFIRMATION"` and a
`portable_question` is present, treat that object as the single source of truth
for user interaction.

Hard rules:

- MUST call `AskUserQuestion` when the host supports structured questions.
- MUST NOT rewrite the current confirmation as a plain-text question.
- MUST NOT summarize options and wait for free-form user input.
- MUST ask exactly one field per round.
- MUST preserve the option `value` fields exactly.
- MUST map the selected option to `--confirm <field>=<value>`.
- MUST rerun `readiness-agent --mode check` with previous context preserved.
- MUST NOT continue to final `READY`, `WARN`, or `BLOCKED` until the current
  confirmation field is confirmed or explicitly skipped.
- If the host does not support `AskUserQuestion`, render the portable question
  as a numbered fallback and require a value that can be mapped to
  `--confirm <field>=<value>`.

## Structured Confirmation Adapter

When `scripts/run_readiness_pipeline.py --mode check` returns JSON with:

```json
{
  "status": "NEEDS_CONFIRMATION",
  "portable_question": {
    "header": "...",
    "question": "...",
    "multi_select": false,
    "options": [...],
    "response_binding": {
      "field": "framework",
      "format": "--confirm framework=<value>"
    }
  }
}
```

the host adapter must:

1. Convert `portable_question` into an `AskUserQuestion` call.
2. Preserve option `value` internally and show option `label` / `description` to
   the user.
3. Rerun `scripts/run_readiness_pipeline.py --mode check` with
   `--confirm <field>=<selected value>`.
4. Repeat for the next `NEEDS_CONFIRMATION` response.
5. Only summarize the final `READY`, `WARN`, or `BLOCKED` result after all
   required confirmations are resolved.

Do not merely print the JSON question and continue with a prose answer when
`AskUserQuestion` is available.

## Artifact Identity

Assessment artifacts must include these identity fields:

```json
{
  "schema_version": "readiness.assessment.v1",
  "producer": "readiness-agent",
  "mode": "check",
  "engine": "assessment"
}
```

Status compatibility:

- `check`: `NEEDS_CONFIRMATION`, `READY`, `WARN`, `BLOCKED`
- `fix`: `READY`, `WARN`, `BLOCKED`

## Hard Rules

- Work on the local machine only.
- Treat the current shell path as the default workspace when `working_dir` is
  not provided.
- Treat the selected workspace root as the certification boundary.
- Resolve scripts, configs, assets, and virtual environments from that
  workspace unless the user explicitly points to another path.
- Certify one intended target only: `training` or `inference`.
- Prefer explicit workspace evidence over guesses.
- Keep framework inference conservative. Downgrade to `WARN` when evidence is
  incomplete instead of forcing a confident claim.
- Only use environment variables to resolve external runtime directories such
  as CANN roots, Ascend env scripts, or Hugging Face cache locations.
- Never modify driver, firmware, CANN, or system Python.
- Never silently substitute system `python` or `pip` for a missing
  workspace-local environment.
- Infer the framework only from current workspace evidence, and do not probe an
  unrelated framework path.
- Apply repairs only inside the workspace or user-local tooling.
- Respect existing Hugging Face cache variables when present.
- `runtime_smoke` is the readiness threshold.
- Do not return `READY` when `runtime_smoke` fails.

## Workflow

Run the workflow in this order:

1. Resolve the workspace root and collect explicit inputs.
2. Infer the intended target and framework from high-confidence evidence inside
   that workspace only.
3. Resolve one workspace-local Python from that workspace and use it
   consistently.
4. Run the streamlined readiness checks through
   `scripts/run_readiness_pipeline.py`.
5. In `fix` mode, allow only safe user-space repairs for missing envs,
   packages, example scripts, or explicitly declared remote assets.
6. Re-run affected checks after successful fixes.
7. Write mode-specific artifacts:
   - `check` writes canonical artifacts under
     `readiness-output/latest/readiness-agent/*` and preserves legacy
     readiness artifacts such as `report.json`, `report.md`, and
     `meta/readiness-verdict.json` for compatibility.
   - `fix` preserves legacy readiness artifacts such as `report.json`,
     `report.md`, `meta/readiness-verdict.json`, and `.readiness.env` where
     policy allows, and refreshes the canonical lock.

Do not reconstruct the old multi-script helper pipeline.

The top-level entrypoint is the only public execution path.

## Ready / Warn / Blocked

Return `READY` only when:

- no hard blocker remains
- target confidence is sufficient
- required assets are present or safely recoverable
- `runtime_smoke` passes

Return `WARN` when:

- no hard blocker is proven
- target or framework inference remains weaker than desired
- compatibility cannot be fully confirmed
- `runtime_smoke` passes but warnings remain

Return `BLOCKED` when:

- a required workspace asset is missing and cannot be repaired safely
- no usable workspace-local Python is selected
- framework or runtime dependencies are missing
- explicit task smoke fails
- `runtime_smoke` fails

## Fix Mode

In `fix` mode, allow these repairs:

- install `uv` into the user environment when needed for workspace fixes
- create or reuse a workspace-local virtual environment such as `.venv`
- install missing framework or runtime packages into the selected env
- scaffold a bundled example entry script when a known recipe applies
- download explicitly declared model or dataset assets when `allow_network=true`

Do not:

- edit user model code to invent new behavior
- mutate system packages
- install system-level Ascend components

## Final Interaction

For `check`, state that no real train or inference command was executed unless
user-confirmed task smoke was run.

For `fix`, real model-script execution remains a separate user-approved step
after readiness. The host may use `AskUserQuestion` before or after `check` /
`fix` for intent clarification or next-step choices, such as running `fix` mode,
rerunning `check` with explicit arguments, or stopping at the report.

The user-facing result may ask:

- `Do you want me to run the real model script now?`

## References

Load these references when needed:

- `references/structured-confirmation-adapter.md`
- `references/product-contract.md`
- `references/decision-rules.md`
- `references/env-fix-policy.md`
- `references/ascend-compat.md`

## Scripts

Use these scripts:

- `scripts/run_readiness_pipeline.py` as the only public entrypoint
