# MindSpore Attention Residuals Implementation Pattern

Use this reference when the selected `algorithm-agent` route is `attnres` and
the target codebase is MindSpore or `mindone.transformers`.

## Core invariants

- Keep Attention Residuals as a residual-path replacement around attention and
  MLP sites, not as a token-attention kernel.
- Keep hidden states in base shape `[B, S, D]`; do not introduce a stream axis.
- Preserve the original non-AttnRes path behind `attn_residual_variant="none"`.
- Count logical residual sites explicitly. In Qwen-style decoder stacks, one
  decoder layer contributes two sites: attention and MLP.
- Register the mixer on the model during construction. Do not create mixer
  modules inside `construct`.

## Config and load path

MindOne model packages may reuse upstream `transformers` config classes. If the
target model lacks a local `configuration_*.py`, migrate the matching upstream
config into the local package before adding AttnRes fields.

Add these fields to the local config:

- `attn_residual_variant`: `"none" | "full" | "block"`
- `attn_residual_block_size`: positive logical-site count
- `attn_residual_eps`: key-side RMSNorm epsilon

Export the local config from the model package and top-level package when the
repo exposes model classes there. The public loading path should support:

```python
config = AutoConfig.from_pretrained(
    model_name,
    attn_residual_variant="block",
    attn_residual_block_size=8,
    attn_residual_eps=1e-6,
)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config)
```

Use direct `from_pretrained(..., attn_residual_variant=...)` only after the
local AutoConfig path is known to route custom fields correctly.

## MindSpore module sketch

```python
class AttentionResidualMixer(nn.Cell):
    def __init__(self, num_queries, hidden_size, eps):
        self.key_norms = nn.CellList([RMSNorm(hidden_size, eps=eps) for _ in range(num_queries)])
        self.queries = ms.Parameter(mint.zeros((num_queries, hidden_size), dtype=ms.float32), name="queries")

    def mix(self, query_index, source_values):
        sources = mint.stack(source_values, dim=0)  # [L, B, S, D]
        keys = self.key_norms[query_index](sources).to(ms.float32)
        query = self.queries[query_index].to(ms.float32)
        logits = mint.einsum("d,lbsd->lbs", query, keys)
        weights = mint.nn.functional.softmax(logits, dim=0, dtype=ms.float32).to(sources.dtype)
        return mint.einsum("lbs,lbsd->bsd", weights, sources)
```

Keep the source-value aggregation dtype-aligned: compute depth logits and
softmax safely, then cast weights back to `sources.dtype` before aggregation.

## Block state pattern

Use an explicit state helper to match the validated Hugging Face structure and
to make logical-site bookkeeping reviewable:

```python
class BlockAttentionResidualState:
    def __init__(self, embedding, block_size):
        self.completed_blocks = [embedding]
        self.partial_block = None
        self.completed_layers = 0
        self.block_size = block_size

    def current_sources(self):
        return self.completed_blocks if self.partial_block is None else [*self.completed_blocks, self.partial_block]

    def append_layer_output(self, layer_output):
        self.partial_block = layer_output if self.partial_block is None else self.partial_block + layer_output
        self.completed_layers += 1
        if self.completed_layers % self.block_size == 0:
            self.completed_blocks.append(self.partial_block)
            self.partial_block = None
```

Register a model attribute named consistently with the reference port, such as
`attn_residual_mixer`, so parameter names and missing-weight logs are easy to
recognize.

## Model wiring

- Add `attn_output(...)` and `mlp_output(...)` helpers to each decoder layer.
- Keep the layer's original `construct` path equivalent for
  `attn_residual_variant="none"`.
- Drive `full` and `block` from the model loop.
- Prefer model-level helpers such as `_forward_full_attnres` and
  `_forward_block_attnres` when the target file style permits it.
- Keep RoPE, masks, cache metadata, and attention backend selection based on
  ordinary `[B, S, D]` hidden states.

## Common MindSpore failure modes

- Custom config fields are ignored because the local package still falls back to
  upstream `transformers` config.
- `from_pretrained` fails because AttnRes fields were passed as model kwargs
  instead of first building a local config.
- The mixer is registered under a surprising name, making missing-weight and
  optimizer checks harder to compare with the reference port.
- `block_size=1` does not match `full` because block state updates count
  transformer layers instead of logical attention/MLP sites.
- `bf16` training fails because softmax weights and source values are not dtype
  aligned before `einsum`.
