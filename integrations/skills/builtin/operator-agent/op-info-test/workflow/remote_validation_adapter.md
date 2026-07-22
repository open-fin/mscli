# Remote Validation Adapter

Use this workflow only when the user explicitly asks for remote validation or explicitly provides a remote validation machine.

The remote machine is an execution adapter. It runs the same validation matrix that was already decided locally. It does not decide:

- which OpInfo cases to add
- which dtype set to declare
- whether the API is suitable for the current op_info workflow
- how failures are classified

## Minimal Flow

1. Prepare the local changes and probe drivers first.
2. Create a remote working directory on the user-provided machine.
3. Run a remote preflight check before the real validation: confirm the remote repo root, key helper imports, and basic MindSpore test-path imports are usable there.
4. Copy the prepared files to that directory.
5. Source the user-provided environment setup on the remote machine.
6. Run the dtype probe on the remote machine first.
7. Backfill the OpInfo dtype declaration from the probe result as a candidate writeback step.
8. Run the MindSpore opinfo case tests on the same remote machine that produced the probe result.
9. If the task explicitly requires stability evidence, or if the newly added case looks prone to intermittent failures, rerun the passing command with `--count=50` as an optional post-pass stability round.
10. Copy the summaries and logs back locally.
11. Continue the normal op_info analysis loop with those results.

## Minimal Command Shape

```bash
ssh <user>@<host> "mkdir -p <remote_workdir>"
scp -r <local_payload> <user>@<host>:<remote_workdir>/
ssh <user>@<host> "source <env_script> && cd <remote_workdir> && <validation_command>"
scp -r <user>@<host>:<remote_workdir>/<result_paths> <local_result_dir>/
```

## Remote Run Notes

- Keep the remote payload small: include only the prepared drivers, helper scripts, and any local summaries needed for context.
- Prefer deterministic commands and explicit output paths.
- Preserve the remote stdout/stderr log alongside structured JSON or Markdown summaries.
- If the remote environment fails before the operator cases run, record it as an environment blocker instead of changing the cases.
- If the remote validation temporarily replaces repository files, back them up first and restore them after the run. Do not leave the remote repo in a modified state unless the user explicitly asks for that.
- If you temporarily isolate one operator for faster remote iteration, keep that override in the temporary remote payload or temporary remote repo state only, then restore it before final full validation.
- Prefer a lightweight import smoke test before the expensive run, for example importing the target `op_database.py` or a minimal shared test helper, so version/layout mismatches are caught as environment issues rather than misclassified as operator failures.
