# Readiness-Agent Product Contract

`readiness-agent` has one user-visible domain responsibility:

- determine pre-run readiness for a selected local single-machine training or
  inference workspace

It exposes this responsibility through two modes:

- `check`: read-only launch-readiness assessment
- `fix`: safe user-space remediation

Readiness decisions must stay scoped to the selected workspace. Evidence from
other repos, home-directory projects, or bundled skill examples cannot replace
missing workspace-local scripts, assets, or framework signals unless the user
explicitly points to those paths.

The final user-visible result must contain:

- `status`
- `can_run`
- `target`
- `summary`
- `blockers`
- `warnings`
- `next_action`

## Status Rules

`READY`

- no hard blocker remains
- `runtime_smoke` passed
- enough evidence exists to treat the target as runnable now

`WARN`

- no hard blocker is proven
- some readiness evidence remains incomplete or uncertain
- `runtime_smoke` may still have passed, but confidence is reduced

`BLOCKED`

- one or more hard blockers remain
- explicit task smoke failed
- `runtime_smoke` failed

## Internal Result

The internal verdict should continue to preserve:

- `execution_target`
- `evidence_level`
- `task_smoke_state`
- `dependency_closure`
- `checks`
- `blockers_detailed`
- `warnings_detailed`
- `fix_applied`
- `revalidated`

## Artifact Contract

`readiness-output/latest/readiness-agent/*` is the canonical downstream
contract. Integrations should read current assessment state from this location,
including `workspace-readiness.lock.json`, `confirmation-latest.json` when a
confirmation is pending, and `run-ref.json`.

`readiness-output/attempts/<attempt_id>/current|final/*` is retained as a
formal audit/debug trail. Attempt snapshots may include raw verdict payloads so
that a run can be inspected after the fact, but those internals are not a
stable downstream API. Tests and consumers should only depend on the minimal
attempt structure and the `attempt_id` reference from `run-ref.json`.

## Interaction Rule

After `check`, the final user-facing result must say that no real train or
inference command was executed. Point users to `/preflight` for asset validation
or to a separate explicit run command outside check mode for workload
execution.
