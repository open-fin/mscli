# Structured Confirmation Adapter Contract

Input:
- JSON payload from `scripts/run_readiness_pipeline.py --mode check`

Trigger:
- `payload.status == "NEEDS_CONFIRMATION"`
- `payload.portable_question` exists

Required behavior:
1. Read `portable_question`.
2. Call host-native `AskUserQuestion` when available.
3. Use the option `value` as the canonical answer.
4. Build `--confirm <field>=<value>`.
5. Rerun assessment with all previous args and previous confirmations preserved.
6. Repeat until `payload.status` is not `NEEDS_CONFIRMATION`.

Reference harness:
- `scripts/assessment/structured_confirmation_adapter.py`

The harness accepts two injected callbacks:

```python
run_assessment(args: list[str]) -> dict
ask_user_question(payload: dict) -> str | dict
```

This makes the adapter behavior testable without importing a host-specific
Claude Code tool. A production host adapter should wire `ask_user_question` to
`AskUserQuestion` and `run_assessment` to `scripts/run_readiness_pipeline.py --mode
check`.

Pseudo-code:

```python
payload = run_assessment(args)

while payload["status"] == "NEEDS_CONFIRMATION":
    q = payload["portable_question"]
    field = q["response_binding"]["field"]

    answer = AskUserQuestion(
        header=q["header"],
        question=q["question"],
        options=q["options"],
        multi_select=q.get("multi_select", False),
    )

    value = answer["value"]
    args = preserve_context_and_add_confirm(args, field, value)
    payload = run_assessment(args)

render_final(payload)
```

Do not rewrite `portable_question` into a plain-text prompt when structured host questions are available.
