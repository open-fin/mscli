# MindSpore Qwen3 mHC Case Study

Use this reference when the selected `algorithm-agent` route is `mhc` and the
target model resembles a Qwen3-style decoder stack in `mindone.transformers`.

## Contents

- Source artifacts
- What the notebook contributes
- What the Qwen3-style port adds
- MindSpore-specific caution

## Source artifacts

- Config patch: local `configuration_qwen3.py`
- Model patch: local `modeling_qwen3.py`
- Tests: local Qwen3 model tests or smoke checks
- Training hook: project-specific train or finetune entrypoint

## What the notebook contributes

The notebook establishes the algorithm shape:

- expand embeddings to `[B, S, N, D]` with `h.unsqueeze(dim=2)` and `repeat`
- compute `H_pre`, `H_post`, and `H_res`
- feed `h_pre` into attention or FFN
- merge with `post_mapping + residual_mapping @ h`
- sum the stream axis at the model output

It also shows the fused RMSNorm formulation and uses `alpha = 0.01`, zero
`beta`, and `max_sk_it = 20`.

## What the Qwen3-style port adds

### 1. Config surface

`configuration_qwen3.py` adds:

- `use_mhc`
- `mhc_expansion_rate`
- `mhc_sinkhorn_iterations`
- `mhc_gating_factor_init`
- `mhc_output_reduce`

It also validates expansion rate, Sinkhorn iteration count, and output
reduction mode. If the local package did not already ship
`configuration_qwen3.py`, copy the matching-version file from `transformers`
first and then extend it.

### 2. Reusable mHC module

`modeling_qwen3.py` adds `Qwen3ManifoldHyperConnection` with:

- `norm + proj` to produce `N^2 + 2N` logits
- `compute_mappings`
- `prepare_layer_input`
- `merge_layer_output`
- internal Sinkhorn-Knopp normalization

Notable engineering choice:

- the port uses `Qwen3RMSNorm -> Linear` instead of the notebook's fused
  RMSNorm path
- mapping logits are promoted to `float32`
- operator replacements must preserve semantics across `mint`, `ops`, or any
  local tensor wrapper used by the file

### 3. Decoder-layer wiring

Each `Qwen3DecoderLayer` gets:

- `self.use_mhc = config.use_mhc`
- `self.attn_mhc`
- `self.mlp_mhc`

The non-mHC path stays unchanged. The mHC path:

- reduces the stream tensor before attention
- merges attention output back into the expanded residual state
- repeats the same pattern around the MLP

### 4. Model-boundary wiring

`Qwen3Model` adds:

- `_expand_mhc_streams`
- `_reduce_mhc_streams`
- `self.use_mhc`
- `self.mhc_expansion_rate`
- `self.mhc_output_reduce`

Important detail:

- RoPE is computed from `inputs_embeds`, not from `hidden_states`, because
  `hidden_states` may become 4-D after stream expansion.

### 5. Initialization

`Qwen3PreTrainedModel._init_weights` calls `reset_mhc_parameters()` for every
mHC module.

The reset method:

- fills `alpha` with `mhc_gating_factor_init`
- sets `pre` bias to `-log(N - 1)` or `10.0` when `N == 1`
- leaves `post` bias at zero
- biases the residual matrix toward identity using diagonal `4.0` and
  off-diagonal `-4.0`

This is stronger than the notebook's zero-`beta` initialization and is worth
preserving when porting to pretrained checkpoints. In MindSpore, the exact
mechanics of this init should follow the local model's existing init style
rather than blindly mirroring PyTorch in-place code.

### 6. Tests

The added Qwen3 checks should cover at least two invariants:

- enabling mHC still returns `[batch, seq, hidden]`
- residual mappings are approximately doubly stochastic along both rows and
  columns

### 7. Training entrypoint

The intended user-facing API is still:

```python
model = AutoModelForCausalLM.from_pretrained(model_name, use_mhc=True)
```

But if the local framework's public config routing is not ready yet, use the
equivalent local config construction path during bring-up and treat Auto
routing separately from the mHC patch itself.

## MindSpore-specific caution

If the target project splits local config and model code differently from
`transformers`, keep the mHC patch focused on:

- local config surface
- model wiring
- operator replacement
- initialization behavior

Do not let unrelated package import or Auto routing problems blur whether the
mHC logic itself is correct.
