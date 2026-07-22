# MindSpore mHC Validation Checklist

Use this checklist after integrating mHC into a decoder model in MindSpore.

## Minimum validation

1. Config validation rejects invalid values.
2. `use_mhc=False` preserves the original forward path.
3. `use_mhc=True` still returns `[batch, seq, hidden]`.
4. Residual mappings are approximately doubly stochastic:
   - row sums close to `1`
   - column sums close to `1`
5. `from_pretrained(..., use_mhc=True)` or the equivalent public load API
   works.

## Strongly recommended validation

1. Run a forward pass with caching or generation enabled.
2. Run a tiny train or finetune smoke test and confirm loss decreases or
   remains finite.
3. Run the model in the dtypes you care about, especially `bf16`.
4. Exercise any model-specific attention variants such as sliding-window
   attention, grouped-query attention, or custom backends.
5. If the model supports gradient checkpointing, confirm the mHC path works
   with it enabled.
6. Confirm MindSpore operator replacements preserved the expected semantics for
   stream expansion, contraction, and reduction.
7. Confirm initialization produces finite residual mappings before training
   starts.

## Example test ideas

### Shape test

```python
import mindspore as ms

config.use_mhc = True
model = Model(config)
result = model(**inputs)
assert result.last_hidden_state.shape == (batch, seq, hidden)
```

### Doubly-stochastic test

```python
import numpy as np

hidden_states = model._expand_mhc_streams(model.embed_tokens(input_ids))
_, _, residual_mapping = model.layers[0].attn_mhc.compute_mappings(hidden_states)

row_sums = residual_mapping.sum(dim=-1).asnumpy()
col_sums = residual_mapping.sum(dim=-2).asnumpy()

np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-3, rtol=1e-3)
np.testing.assert_allclose(col_sums, np.ones_like(col_sums), atol=1e-3, rtol=1e-3)
```

### Public API smoke test

```python
model = AutoModelForCausalLM.from_pretrained(model_name, use_mhc=True)
```

### Explicit config smoke test

```python
config = Config.from_pretrained(model_name, use_mhc=True)
model = Model.from_pretrained(model_name, config=config)
```

Use the public API smoke test when the local package already routes the mHC
fields through its public config surface. If the public AutoModel path is not
ready yet, the explicit config smoke test is the safer bring-up check for
MindSpore ports.

## Symptom-to-cause hints

- Attention receives 4-D input: stream reduction was skipped before attention
  or MLP.
- LM head receives 4-D input: stream reduction was skipped before final norm or
  task head.
- Rows sum to `1` but columns do not: Sinkhorn implementation is incomplete or
  not iterated enough.
- Loss diverges early: initialization is not near identity, output reduction
  mismatches checkpoint scale, mapping math runs in low precision, or an
  operator replacement changed semantics.
- Generation fails only with cache: cache or position metadata was coupled to
  the expanded residual tensor instead of base token positions.
- Config flags do not take effect: local config wiring is incomplete or the
  public load API is not yet consuming the local config surface.
