# MindSpore Qwen3 Attention Residuals Case Study

Use this reference when the selected `algorithm-agent` route is `attnres` and
the target model resembles Qwen3 in `mindone.transformers`.

## Successful integration shape

The successful MindOne Qwen3 port used the Hugging Face Qwen3 AttnRes retrofit
as the structural reference and adapted tensor operations to MindSpore.

Touched areas:

- local `configuration_qwen3.py`
- local `modeling_qwen3.py`
- package exports such as `models/qwen3/__init__.py` and top-level exports
- Qwen3 model tests or smoke scripts
- train or finetune script using `AutoConfig.from_pretrained`

## Config migration

If `mindone.transformers.models.qwen3` has no local `configuration_qwen3.py`,
copy or migrate the matching upstream `transformers` config first. Then add:

- `attn_residual_variant="none"`
- `attn_residual_block_size=1`
- `attn_residual_eps=1e-6`

Validate variant values and positive block size. Export the local config from
the Qwen3 package and top-level package so AutoConfig can find it before
falling back to upstream `transformers`.

For training scripts, prefer:

```python
config = AutoConfig.from_pretrained(
    model_name,
    num_labels=5,
    pad_token_id=tokenizer.pad_token_id,
    attn_residual_variant="block",
    attn_residual_block_size=8,
    attn_residual_eps=1e-6,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    mindspore_dtype=ms.bfloat16,
)
```

This avoids treating AttnRes fields as stray model-construction kwargs.

## Model patch

Mirror the validated Hugging Face structure where it matters:

- `Qwen3AttentionResidualMixer`
- `Qwen3FullAttentionResidual`
- `Qwen3BlockAttentionResidualState`
- `Qwen3BlockAttentionResidual`
- decoder-layer `attn_output(...)`
- decoder-layer `mlp_output(...)`
- model attribute `attn_residual_mixer`
- model helpers `_forward_full_attnres(...)` and `_forward_block_attnres(...)`

MindSpore-specific adaptations:

- Use `nn.Cell`, `nn.CellList`, `ms.Parameter`, and the target file's existing
  tensor API style, such as `mindspore.mint`.
- Keep logits and depth softmax in `float32` when useful, then cast weights back
  to source dtype before weighted aggregation.
- Use MindSpore loading helpers in tests, for example
  `ms.load_param_into_net(..., strict_load=False)`.

## What must stay fixed

- The original non-AttnRes residual path remains available under
  `attn_residual_variant="none"`.
- Attention and MLP receive ordinary `[B, S, D]` tensors.
- RoPE, cache positions, causal masks, and backend selection remain outside the
  depth mixer.
- The final model output remains `[B, S, D]` before task heads.

## Validation that caught issues

The successful port used these checks:

- syntax/compile check for changed files
- config round-trip for AttnRes fields
- block forward shape smoke
- registered parameter name check for `attn_residual_mixer`
- `block_size=1` vs `full` comparison
- train script smoke using `AutoConfig.from_pretrained(..., attn_residual_variant="block")`

## Common corrections from the port

- Do not rely only on `getattr(config, "attn_residual_variant", "none")`;
  migrate local config if normal public use needs AutoConfig support.
- Do not register the module as a vague name if the reference port expects
  `attn_residual_mixer`; stable names simplify checkpoint and optimizer checks.
- Do not inline all block bookkeeping if a state helper makes logical-site
  accounting easier to inspect.
- Do not change dtype handling blindly to match PyTorch if MindSpore bf16 needs
  safer `float32` depth logits and softmax.
