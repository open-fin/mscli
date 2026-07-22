# HF Transformers Route

Use this route when the source clearly belongs to Hugging Face transformers.

This route migrates Hugging Face transformers models into
`mindone.transformers` with a standard workflow, tool-assisted conversion,
manual MindSpore adaptation, and registration updates.

Top-level `migrate-agent` should keep the route boundary explicit here instead of
forcing the user to choose it up front.

## When to Use

- Porting Hugging Face transformers models such as LLaMA, BERT, GPT, or Qwen
  to MindSpore-oriented targets.
- Migrating a transformers-family source repo into `mindone.transformers` with
  repo-specific rules.
- Adding a new transformer architecture to a MindOne-style transformers tree.

## Repository Assumptions

Source repository:

- Hugging Face `transformers`
- Core path: `transformers/src/transformers`
- Model tests: `transformers/tests/models`
- A standalone HF-style model repository is also valid for this route, but it
  has different configuration, tokenization, and processor rules. Load
  `references/hf-transformers-standalone-model-repo.md` for those cases.

Target repository:

- `mindone`
- Core path: `mindone/mindone/transformers`
- Model tests: `mindone/tests/transformers_tests`

## Route Workflow

### 1. Intake and target confirmation

Collect these inputs before editing files:

1. Identify the source and target repository paths and check that the source is
   genuinely transformers-family.
2. Identify the exact `{model_name}` from the prompt or workspace evidence.
3. Map the default source and target model directories:
   - source: `transformers/src/transformers/models/{model_name}/`
   - target: `mindone/mindone/transformers/models/{model_name}/`

Prioritize `model_name` precision. If the name is ambiguous or incomplete,
pause and confirm the exact model family or version before proceeding.

### 2. Modeling files migration with the auto-convert tool

Copy only the intended files from source to target:

- `modeling_*.py`
- `processing_*.py`
- `image_processing_*.py`
- `video_processing_*.py`

Only migrate processing files when they actually manipulate torch tensors.
Otherwise, prefer reusing the Hugging Face implementation directly.

Run the route-specific single-file in-place conversion before any manual edits:

```bash
python skills/migrate-agent/scripts/hf_transformers_auto_convert.py \
  --src_file path/to/file.py --inplace
```

Install the tool dependency first when needed:

```bash
pip install -r skills/migrate-agent/scripts/hf_transformers_auto_convert.requirements.txt
```

You must run the auto-convert script before manual edits on migrated modeling
files.

### 3. Manual fix checklist

#### 3.1 Structural and API changes

- `torch.nn.Module` -> `mindspore.nn.Cell`
- `forward` -> `construct`
- `torch.nn.Parameter` -> `mindspore.Parameter`
- Replace `torch` and `torch.nn.functional` usages with `mindspore.mint` or
  `mindspore.ops`
- Prefer `mindspore.mint`, then `mindspore.ops`, then `mindspore.nn`
- Treat the auto-converted `mindspore.mint` and `mindspore.mint.nn` forms as
  the default steady-state implementation for this route
- Do not bulk-rewrite auto-converted `mint.*` or `mint.nn.*` calls back into
  `mindspore.nn.*`, `ops.*`, or older MindSpore-style APIs just for style
  consistency
- Only replace a `mint` form when there is a concrete repo-local reason such as
  an existing model-family convention, a missing API, or a verified behavioral
  incompatibility
- Drop unsupported `inplace=True` arguments

#### 3.2 Device handling cleanup

- Remove `.to(device)`, `.cuda()`, `torch.device`, and `mps` branches
- Do not add `model.device`, `.device` compatibility properties, or other
  PyTorch-style device shims
- Remove `inputs.to(model.device)` from tests and examples instead of adapting
  the model to support it
- Check function signatures and remove device-only parameters when they are no
  longer needed
- Do not remove `register_buffer` just because the original code had device
  handling nearby; `register_buffer` remains valid in MindSpore

Example pattern:

Before:

```python
def _dynamic_frequency_update(self, position_ids, device):
    seq_len = mint.max(position_ids) + 1
    if seq_len > self.max_seq_len_cached:
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = seq_len

    if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
        self.original_inv_freq = self.original_inv_freq.to(device)
        self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
        self.max_seq_len_cached = self.original_max_seq_len
```

After:

```python
def _dynamic_frequency_update(self, position_ids):
    seq_len = mint.max(position_ids) + 1
    if seq_len > self.max_seq_len_cached:
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, seq_len=seq_len)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = seq_len

    if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
        self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
        self.max_seq_len_cached = self.original_max_seq_len
```

#### 3.3 Imports and decorators

- Keep config and tokenizer imports from Hugging Face `transformers`
- Use `mindone.transformers.modeling_utils` for modeling utilities
- Remove unused or PyTorch-only imports that are not migrated
- Remove decorators only when they are not defined in `mindone.transformers`
- If a decorator is removed, remove both its import and usage

Typical decorators that may need removal:

- `@deprecate_kwarg`
- `@auto_docstring`
- `@torch.jit.script`
- `@use_kernel_func_from_hub`

#### 3.4 Tensors and shapes

- Use `mindspore.Tensor` in docstrings
- Wrap shape arguments in tuples such as `.view((b, s, h))`

### 4. Registration and exports

Update the target repo registration chain:

- `mindone/mindone/transformers/models/auto/configuration_auto.py`
- `mindone/mindone/transformers/models/auto/modeling_auto.py`
- `mindone/mindone/transformers/models/auto/processing_auto.py` when processor
  files are migrated
- `mindone/mindone/transformers/models/auto/image_processing_auto.py` when
  image processing files are migrated
- `mindone/mindone/transformers/models/auto/video_processing_auto.py` when
  video processing files are migrated
- `mindone/mindone/transformers/models/{model_name}/__init__.py`
- `mindone/mindone/transformers/models/__init__.py`
- `mindone/mindone/transformers/__init__.py`

For `mindone/mindone/transformers/models/{model_name}/__init__.py`:

- Preserve the file header comment exactly
- After the header, keep only `from .<module> import *` lines
- Remove all other non-header lines
- Verify that each referenced module exists locally
- Drop import lines that point to missing modules

Use Hugging Face auto files as a reference for insertion order.

For standalone HF-style model repositories, also verify that every component
needed by `trust_remote_code=False` loading has a local target implementation
and local auto registration.

### 4.1 Processor registration checks

Before changing a processor class, inspect the source processor and source-side
auto mappings for tokenizer, image processor, video processor, and processor
classes.

- Preserve source processor attributes when MindOne has equivalent local Auto
  support.
- If MindOne lacks local `AutoTokenizer` support for the model, add or register
  the tokenizer locally when possible.
- Only use processor-class workarounds, such as pointing `tokenizer_class` at a
  concrete tokenizer class or reducing `attributes`, when local Auto support is
  missing and the migration report documents the reason.

### 4.2 Unit test migration

When the user requests test migration, or when source model tests exist and the
migration goal includes tests, adapt source tests from:

- source: `transformers/tests/models/{model_name}/`
- target: `mindone/tests/transformers_tests/models/{model_name}/`

Use this order:

1. Inspect the upstream test file and classify what it covers:
   - fast model unit tests
   - slow real-weight or generation tests
   - processor/tokenizer tests
   - multimodal input construction
   - attention backend or device-specific tests
2. Inspect existing MindOne tests for the closest local pattern:
   - same model family first
   - same task type second, such as causal LM, vision-to-seq, or multimodal
   - generic named-modules tests, such as `blt`, only when there is no closer pattern
3. Create `mindone/tests/transformers_tests/models/{model_name}/__init__.py` as
   an empty file.
4. Rewrite the fast model unit tests into MindOne's PyTorch-vs-MindSpore
   parity style:
   - add the MindOne provenance header to new test files rewritten from
     upstream, such as `test_modeling_*.py`
   - use `get_modules`, `generalized_parse_args`, and `compute_diffs`
   - use `pytest.mark.parametrize` for dtype and mode expansion
   - use `MODES = [1]`
   - map PyTorch output attributes to MindSpore tuple indices with `outputs_map`
5. Preserve the source test's model intent, but adapt its harness to MindOne:
   - reuse upstream config classes from `transformers` when the model migration
     also reuses upstream config
   - prefer shared testers such as `CausalLMModelTester` when they already
     express the source test inputs
   - set model-required execution fields such as
     `config._attn_implementation = "eager"` when needed
   - do not add version gates such as `if transformers.__version__ >= ...`
   - do not add unrelated config overrides such as `use_cache=False` or
     `sliding_window=None` unless the source behavior or target model requires them
6. Only migrate fast model unit tests. Do not migrate slow real-weight,
   generation, remote asset, flash-attention, or device-specific tests as part
   of this unit test migration step.
7. Use the default transformer parity thresholds unless a measured mismatch
   requires otherwise:
   - `{"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}`

Verification for migrated tests:

- run `python -m py_compile` on new or changed test files
- run the target test file with pytest, for example:
  `pytest -q tests/transformers_tests/models/{model_name}/test_modeling_{model_name}.py -q`
- if a test fails, classify the cause before editing:
  - model migration issue
  - shared MindOne component issue
  - test config/input mismatch
  - precision threshold issue
- treat threshold changes as a last resort and report the reason explicitly

### 4.3 File provenance and license header

Before declaring the migration done, normalize new upstream-adapted target
files to the local MindOne provenance header convention:

```python
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
```

Apply it with these rules:

- Add the two provenance lines to new migrated source files such as
  `modeling_*.py`, `processing_*.py`, `image_processing_*.py`, and
  `video_processing_*.py`.
- Add the same provenance lines to new target-side test files rewritten from
  upstream tests, such as `test_modeling_*.py`.
- If the file already has a copyright header and Apache license block, insert
  the provenance lines after the copyright lines and before the license body.
- If the file starts with an upstream autogenerated notice, preserve that
  notice and place the provenance lines immediately after it.
- Do not add the provenance header to files that are intentionally required to
  stay empty.
- For `mindone/tests/transformers_tests/models/{model_name}/__init__.py`,
  preserve the empty-file convention and do not add a header.

### 5. Done criteria

- The minimal import validation succeeds via a `from transformers import xxx`
  style import path for the migrated target, and the model imports cleanly in
  MindOne
- Auto mappings and exports are updated
- Verification artifacts or next test commands are recorded
- If source model tests were migrated, target tests exist under
  mindone/tests/transformers_tests/models/{model_name}/ and the single-file
  pytest command passes or the remaining failure is reported with cause

Do not mark the migration complete before the `from transformers import xxx`
minimal import validation has passed for the migrated target.

## Route Guardrails

- For upstream `transformers` source-tree migrations, do not migrate
  `configuration_*.py`, `tokenization_*.py`, or `*moduler_*.py` by default.
- For standalone HF-style model repos, migrate and register
  `configuration_*`, tokenizer, and processor components when the target repo
  lacks local implementations required for `trust_remote_code=False`.
- Reuse Hugging Face configuration and tokenization implementations directly
  only when the target repo can load them locally without remote-code fallback.
- Keep changes minimal and aligned with existing MindOne patterns
- Avoid adding custom compatibility wrappers unless they are required
- Use diff-based insertion when updating auto maps

Load the route-specific companion references for environment notes and repo
guardrails:

- `references/hf-transformers-guardrails.md`
- `references/hf-transformers-env.md`
- `references/hf-transformers-standalone-model-repo.md` for standalone
  HF-style model repositories

## Route Output

Report at least:

- files changed and why
- tests run, tests generated, or tests recommended
- whether new migrated files were normalized to the local provenance header
  convention
- remaining TODOs and risks
