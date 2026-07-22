# Workflow 5: Export And Placeholder Behavior

## Goal

Ensure the operator is exported correctly through the `mint` package, and that the interface matches PTA.

## Outputs

- **`mint` package exports**: updates to `__init__.py` / `__all__`
- **Interface files**: functional / nn as needed

**Interface alignment constraint**: interface name, parameter names, parameter order and defaults, and input dtype/range constraints must match PTA.

---

## Steps

- Add the operator name in both the **corresponding operator category import block** and **`__all__`** inside the relevant `__init__.py` under `mindspore/python/mindspore/mint/`.
- Ensure the new operator appears in the `__all__` list.

Interfaces named `xxxExt` or `xxx_ext` are internal only. When exporting them publicly, always use `import xxx_ext as xxx` and remove the `ext` suffix. Never expose an `ext` interface directly as the public API.

Fill the `Python API` field in the Feature document.

---

## Success Criteria

- [ ] For public interfaces, exported them in `mindspore.mint`
- [ ] Filled the `Python API` field in the Feature document.

---
