# MindSpore Attention Residuals Validation Checklist

Use this checklist after integrating AttnRes into a MindSpore or
`mindone.transformers` decoder model.

## Minimum validation

1. Local config validation rejects invalid AttnRes values when the target repo
   has config validation logic.
2. The local config can round-trip through `to_dict()` and `from_dict()` with:
   - `attn_residual_variant`
   - `attn_residual_block_size`
   - `attn_residual_eps`
3. `attn_residual_variant="none"` preserves the original forward path.
4. `attn_residual_variant="block"` returns `[batch, seq, hidden]`.
5. `attn_residual_block_size=1` matches `full` exactly or within numerical
   tolerance.
6. AttnRes-specific parameters appear in `parameters_and_names()` and optimizer
   parameter groups. Prefer recognizable names such as
   `attn_residual_mixer.mixer.queries`.
7. Public load works through the local config path:

```python
config = AutoConfig.from_pretrained(model_name, attn_residual_variant="block", attn_residual_block_size=8)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
```

## Strongly recommended validation

1. Run a tiny train or finetune smoke test and confirm loss is finite.
2. Compare first-step loss against baseline. A dramatic jump usually indicates
   wiring, initialization, or registration problems.
3. Run the model in the dtypes that matter, especially `bf16`.
4. If generation or caching is supported, test it with `"none"` and one AttnRes
   variant.
5. If gradient checkpointing is supported, test the AttnRes path with it
   enabled.
6. If attention backend variants exist, confirm AttnRes does not break backend
   selection.

## Example test ideas

```python
config.attn_residual_variant = "block"
config.attn_residual_block_size = 2
model = Model(config)
result = model(input_ids=input_ids)
assert result.last_hidden_state.shape == (batch, seq, hidden)
```

```python
full_config.attn_residual_variant = "full"
block_config.attn_residual_variant = "block"
block_config.attn_residual_block_size = 1

full_model = Model(full_config)
block_model = Model(block_config)
ms.load_param_into_net(block_model, full_model.parameters_dict(), strict_load=False)

np.testing.assert_allclose(
    block_model(input_ids=input_ids).last_hidden_state.asnumpy(),
    full_model(input_ids=input_ids).last_hidden_state.asnumpy(),
    atol=1e-5,
    rtol=1e-5,
)
```

```python
names = [name for name, _ in model.parameters_and_names()]
assert any("attn_residual_mixer" in name and "queries" in name for name in names)
```

## Symptom-to-cause hints

- Config fields print as defaults after loading: the script probably used the
  upstream config or passed fields on the wrong API.
- No AttnRes-specific missing weights: the mixer may not be registered, may be
  registered under an unexpected name, or the variant was not enabled in config.
- `block_size=1` differs from `full`: inspect logical-site indexing and block
  state updates.
- `bf16` crashes in mixer aggregation: ensure softmax weights are cast back to
  source-value dtype before `einsum`.
