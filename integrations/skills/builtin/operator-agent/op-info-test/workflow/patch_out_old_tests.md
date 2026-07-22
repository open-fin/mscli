# Temporary Operator Isolation For Fast Iteration

Use this workflow only when you need to validate **one newly added operator or
a small target operator set** quickly without paying for a full parameterized
run on every iteration.

This is a **temporary debugging tactic**, not the default skill path and not a final repository state.

## What This Workflow Is For

Use it when all of the following are true:

- you are actively editing one new OpInfo registration or a small related set
- the normal frontend parameterized test entry would run many unrelated operators
- you need a fast local iteration loop before the final full validation

Do **not** treat this temporary isolation patch as part of the finished change. Remove it before final validation and before leaving the repo in a reviewable state.

## Minimal Flow

1. Make the real OpInfo change first.
2. Identify every `xxx_op_db` list consumed by the validation entry you will run.
3. Add temporary overrides after the original db definitions:
   - target db list: keep only the target operator names
   - non-target db lists consumed by the same entry: override to `[]`
4. Run the targeted validation loop until the new operator path is stable.
5. Remove the temporary override patch.
6. Rerun the required full validation matrix.
7. Only keep the real OpInfo registration change; do not keep the isolation patch.

## Typical Shape Of The Temporary Patch

Example for `other_op_db`:

```python
other_op_db = [
    ...
]

# Temporary local isolation for fast iteration only.
other_op_db = ["mint.nn.Conv2d"]
```

For every non-target db list consumed by the selected frontend validation
entry, temporarily override it to `[]`.

Example:

```python
other_op_error_db = [
    ...
]

# Temporary local isolation for fast iteration only.
other_op_error_db = []
```

## Rules

- Keep the patch minimal and easy to remove.
- Comment it clearly as temporary isolation.
- Override db lists after their original definitions; do not delete existing
  registrations or rewrite the original list contents.
- Never commit the isolation patch as part of the final clean change unless the user explicitly asks for it.
- After the targeted loop passes, restore the original db wiring and rerun the required full matrix.

## Recommended Use

Prefer this workflow for:

- fast local iteration on one operator or a small related operator set

Do not use it to avoid final full validation. It is a speed tool, not a replacement for the normal completion criteria.
