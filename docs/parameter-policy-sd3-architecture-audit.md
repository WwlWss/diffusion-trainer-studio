# Phase D1-SD3A: read-only MMDiT architecture and component audit

This is a **diagnostic-only** building block for SD3 / SD3.5 Component-wise LoRA.
It does **not** change `sd3-lora`'s current seven-component policy, create
optimizers, run a trainer, alter checkpoints, or qualify any new `full_bf16`
backend or configuration. The next phases own SD3 scoped Resume and exact-head
CUDA qualification.

## Sources of truth

- Loader architecture: `scripts/dev/library/sd3_utils.py` (`analyze_state_dict_state`,
  `load_mmdit`) and `scripts/dev/library/sd3_models.py` (`MMDiT`, `MMDiTBlock`,
  `SingleDiTBlock`).
- LoRA original-target metadata: `scripts/dev/networks/lora_sd3.py`, preserved by
  `mikazuki/parameter_routing.py` (`AdapterTargetMetadata`).
- The original seven SD3 Component IDs and existing Policy v1 remain unchanged
  in `mikazuki/model_component_profiles.py`.

## Recognized checkpoint structures

| Topology label | Joint blocks | Hidden / heads | Pos grid | QK norm | Image-only extra attention |
| --- | ---: | --- | ---: | --- | --- |
| `sd3-medium` | 24 | 1536 / 24 | 192 | none | none |
| `sd35-medium` | 24 | 1536 / 24 | 384 | RMS | blocks 0-12 |
| `sd35-large-topology` | 38 | 2432 / 38 | 192 | RMS | none |

Block indices are zero-based. Context is `pre_only` in the last block, with no
Context MLP or `context_block.attn.proj`. SD3.5 Medium `x_block.attn2` adds a
second image-stream Attention in the first **13** blocks; it must not be confused
with Joint Attention or Context Attention.

The audit checks contiguous block indices, selected QKV/projection/MLP/modulation
shapes, QK-norm presence, position embedding shape, and the extra attention
subtree. An unrecognized/malformed variant fails closed. The shape index hash is
**not** a weights-file digest and does **not** authenticate a checkpoint. Even a
valid topology does not establish model provenance or a full-BF16 permission.
**Large and Large Turbo can share the same topology**; their exact release and
training objectives must be qualified separately.

## Eleven diagnostic-only groups

| Group | Original target |
| --- | --- |
| `mmdit.x_attention.adapter` | `joint_blocks.*.x_block.attn.{qkv,proj}` |
| `mmdit.context_attention.adapter` | `joint_blocks.*.context_block.attn.{qkv,proj}` |
| `mmdit.x_extra_attention.adapter` | `joint_blocks.{0..12}.x_block.attn2.{qkv,proj}` on SD3.5 Medium |
| `mmdit.x_mlp.adapter` | `joint_blocks.*.x_block.mlp.*` |
| `mmdit.context_mlp.adapter` | `joint_blocks.*.context_block.mlp.*` except final pre-only block |
| `mmdit.x_modulation.adapter` | `joint_blocks.*.x_block.adaLN_modulation.1` |
| `mmdit.context_modulation.adapter` | `joint_blocks.*.context_block.adaLN_modulation.1` |
| `mmdit.input_output.adapter` | explicit supported `x_embedder`, `context_embedder`, `t_embedder`, `y_embedder`, `final_layer` LoRA targets |
| `clip_l.adapter` | CLIP-L Linear LoRA |
| `clip_g.adapter` | CLIP-G Linear LoRA |
| `t5xxl.adapter` | T5XXL Linear LoRA |

These are not new Policy IDs. They are based on original target metadata, **not**
on flattened `lora_unet_...` naming. Unsupported roots, target types, block
indices, attention submodules, or final-block Context MLPs are rejected.
A diagnostic group with zero tensors is normal when a network target is not
created (`emb_dims` unset; frozen/disabled encoder; `train_t5xxl=False`). The
report separates logical component groups from exact per-block tensor counts.

## Usage

With a real checkpoint:

```bash
python tools/audit_sd3_architecture.py \
  --checkpoint /path/to/sd35_model.safetensors \
  --output /path/to/sd3-audit.json
```

The `--checkpoint` path reads only the safetensors tensor **headers/shapes**
through `safe_open(...).get_slice(...).get_shape()`, not tensor values or model
weights into memory. For synthetic tests, the `--shape-index` flag accepts a
JSON dictionary of tensor names to shapes (or `{ "shapes": ... }`). Synthetic
shape input is not proof of a production model.

For optional adapter-target statistics, pass `--adapter-targets targets.json`.
Each list item represents **one LoRA tensor**, for instance `lora_down.weight`
or `lora_up.weight`, and has exactly these fields:

```json
[
  {
    "target_root": "mmdit",
    "target_path": "joint_blocks.12.x_block.attn2.qkv",
    "target_type": "torch.nn.Linear",
    "parameter_shape": [8, 1536]
  }
]
```

For live Step-3 scanner results, a pure API is also available:

```python
from mikazuki.sd3_component_audit import audit_sd3_parameter_descriptors

report = audit_sd3_parameter_descriptors(architecture, scanned_descriptors)
```

This API performs no training/mutation and rejects inconsistent aliases or
repeated physical parameter IDs. Neither the CLI nor this API is integrated into
an existing trainer, Preview, or Start action in PR A.

## Test and release boundaries

```bash
python -m unittest discover -s tests -p 'test_sd3_architecture_audit.py' -v
```

The tests cover the three topologies, pathological shape/size/extra-layer cases,
pre-only semantics, all eleven diagnostic groups, actual per-layer counts,
malformed/missing metadata, negative CLI cases, and a header-only safetensors
reader fake that fails if full tensor contents are requested.

**Not in scope for PR A:** runtime `ModelComponentProfile` migration, UI/Sidecar
v2, training or optimizer routing changes, SD3 Dev NetworkTrainer Resume fixes,
additional T5/FP8/gradient-checkpointing permissions, SD3.5 Medium/Large GPU
qualification, or release of the pending `sd3-lora` full-BF16 gate.