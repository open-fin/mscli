# HF Transformers Standalone Model Repos

Use this reference when the source is a Hugging Face-style model repository
rather than a model already present in the upstream `transformers` source tree.

Examples include model repos that ship their own `configuration_*`,
`modeling_*`, `processing_*`, tokenizer, or image/video processor files and
expect `trust_remote_code=True` in Hugging Face.

## Core rule

Do not apply the upstream-transformers rule "do not migrate configuration or
tokenization files" blindly.

For standalone HF-style repos targeting `mindone.transformers`, support for
`trust_remote_code=False` requires every component used by auto loading to exist
locally and be registered locally.

Migrate and register local components when the target repo does not already
provide them:

- `configuration_*` when the model type is not already known to MindOne.
- tokenizer files when processor or auto loading needs a local tokenizer.
- `processing_*`, `image_processing_*`, and `video_processing_*` according to
  the model's actual modalities and target Auto support.

## Processor and tokenizer handling

Before changing a processor class, compare the source implementation and its
source-side auto mappings. A processor that works unchanged in an upstream
transformers tree may depend on auto mappings that MindOne does not yet have.

MindOne may not have a local `AutoTokenizer` registration path equivalent to
Hugging Face. In that case, a source processor with
`tokenizer_class = "AutoTokenizer"` can fall back to external Hugging Face
loading and fail under `trust_remote_code=False`.

Use this order:

1. Prefer adding or reusing a local tokenizer implementation and local tokenizer
   mapping.
2. Preserve source processor attributes when local Auto support can resolve
   them.
3. Only as a documented workaround, point the processor at the concrete local
   tokenizer class or reduce processor attributes to the locally supported
   modalities.

Do not add or change `tokenizer_class`, `attributes`, `image_processor_class`,
or `video_processor_class` by default. Make those changes only when target Auto
capability is missing and the report explains why the source form cannot load.

## Device handling

Do not add PyTorch-style device compatibility to make tests pass.

Remove device-only logic instead:

- no `model.device` or `.device` compatibility property
- no `.to(device)` or `inputs.to(model.device)` in examples or tests
- no `torch.device`, `device=None`, CUDA, MPS, or CPU grouping branches
- no device argument passed through RoPE helpers or image/video processors

MindSpore execution context owns device placement. Tests and examples should
match MindSpore semantics instead of forcing the model to mimic PyTorch.

## Model initialization

Do not load `AutoProcessor` inside model `__init__` just to recover preprocessing
defaults. This can trigger processor/tokenizer loading during model
construction, including remote-code paths that should stay out of
`trust_remote_code=False` execution.

Prefer config values or local image/video processor defaults for preprocessing
metadata needed by the model.

## Verification

For standalone model repos, verification is not complete until these pass with
`trust_remote_code=False`:

- config load through the target local AutoConfig path
- processor/tokenizer load for the intended modalities
- model load through the target local AutoModel class
- a minimal text or multimodal inference path, depending on the model goal
