# Parameter Policy / Component-wise Optimization

Parameter Policy is Diffusion Trainer Studio's opt-in component-wise optimizer
and learning-rate system. It lets one training page route different model
components to different optimizer profiles, learning rates, or frozen state.

It is intentionally separate from the historical **Standard** optimizer path.
Standard remains the default.

## Standard vs Component

**Standard** uses the trainer's historical global optimizer and learning-rate
controls. Existing DTS workflows continue to use this path unless
`Optimization Mode` is explicitly switched to `Component`.

**Component** makes Parameter Policy authoritative for optimizer/LR routing.
Each backend exposes a fixed registry-owned component list. A component can be
trained or frozen; trainable components reference an Optimizer Profile and an
independent learning rate.

Switching back to Standard is always allowed. Hidden/stale Component editor
state does not affect Standard compilation and no Parameter Policy sidecar is
written for a Standard request.

## Switching Standard -> Component

When the GUI is switched from Standard to Component, DTS does not bootstrap from
the raw browser object directly. The current form is first resolved through the
Standard Schemastery branch so visible defaults are materialized, then the host
attempts an exact Standard -> Component migration.

The migration is fail-closed:

- representable Standard optimizer/LR semantics are converted to a Component
  policy;
- unsupported or ambiguous semantics return an explicit blocker rather than an
  approximation;
- an already complete Component policy is never overwritten by bootstrap.

If migration fails, the GUI remains in Component mode so the error can be read;
switch back to Standard or edit/reset the configuration before trying again.

## Optimizer Profiles

An Optimizer Profile contains:

- a profile name chosen by the user;
- an optimizer type;
- optimizer-specific arguments.

The editor only allows newly selecting optimizer types whose capability registry
marks them `supported` for Component v1. Restricted/planned optimizer names
remain readable for imported historical policies, but are disabled in the
selector and remain Start-blocked where appropriate.

Optimizer argument values are entered through the legacy GUI as strings. DTS
normalizes literal strings before canonical policy validation. For example,
numeric, boolean, list/dict and null-like values are converted into their native
types when unambiguous, while strings that merely look like literals are
preserved losslessly.

## Components, Train/Freeze, and LR

Component IDs are defined by the backend Model Component Profile registry; the
GUI does not invent arbitrary component names.

For each component:

- `Train = false` freezes the component. Frozen routes discard stale optimizer
  and LR fields during canonicalization.
- `Train = true` requires a primary Optimizer Profile and positive learning
  rate.
- an optional fallback Optimizer Profile and fallback LR may be present when the
  primary optimizer requires parameter eligibility checks.

The exact component list differs by backend. Preview displays the registry
labels/descriptions together with the resulting policy route.

## Muon eligibility and explicit fallback

DTS uses the pinned `pytorch-optimizer==3.10.0` Muon implementation.

Muon is not valid for every parameter. Component v1 only routes
model-profile-approved hidden-layer 2D weights to Muon. Parameters that fail
that eligibility contract must have an explicit fallback route.

Typical pattern:

~~~text
primary profile:   Muon
primary LR:        1e-4
fallback profile:  AdamW
fallback LR:       1e-4
~~~

DTS deliberately does not expose or depend on Muon's internal AdamW fallback
controls. Parameter Policy owns fallback routing explicitly so optimizer
ownership and checkpoint topology remain auditable.

## Preview and runtime readiness

Component Preview is authoritative for whether the current form is ready to
Start.

Any Component edit immediately invalidates the previous green readiness state
before the debounced Preview request is sent. Start stays disabled until the
current Preview returns `runtime_ready=true`.

When Preview reports blockers, they are shown in the normal error area. The
frontend gate is only UX: the backend `/api/run` path repeats the authoritative
validation and remains fail-closed.

The model-free host Preview can show component routes and blockers, but it does
not claim exact parameter counts, tensor counts, Muon-eligible tensor counts, or
device placement. Those facts belong to trainer/runtime diagnostics.

## Curated Component presets

DTS ships a deliberately small example set rather than ten redundant
"same as Standard" presets:

- **Flux Full — Muon + AdamW fallback**: demonstrates eligibility routing across
  the six Flux full components.
- **Anima Full — DiT Muon + AdamW fallback**: demonstrates DiT component routing
  while keeping Qwen3 and the optional LLM Adapter frozen.
- **SDXL Full — U-Net / Text Encoder split LR**: demonstrates independent U-Net
  subcomponent and TE learning rates.
- **SDXL LoRA — selective U-Net**: demonstrates component freeze plus different
  LRs for Attention and FFN adapters.

The numerical learning rates are examples, not universal recommendations.

Component presets merge only the keys listed in the preset. Model checkpoint,
dataset and output paths that the preset does not contain remain untouched.
Unlike historical Standard presets, explicitly listed Component values such as
`false` and `0` are applied literally so a preset can clear an old advanced
setting.

Preview remains authoritative: unrelated stale advanced settings not listed by
a preset may still produce blockers.

## Export and import

Standard configurations that do not require DTS sidecars can be exported as
ordinary TOML.

A Component configuration owns a content-addressed Parameter Policy sidecar, so
Export uses a `.dts.json` training bundle when sidecars are present. The
bundle carries the effective TOML plus owned sidecar contents/hashes.

Import/Rehydrate validates DTS-owned sidecar paths and hashes, canonicalizes the
policy, and converts canonical values back to editor-safe representations. A
complete imported Component policy is preserved; the Standard -> Component
bootstrap does not overwrite it.

## Checkpoint / resume identity

Component runtime records Parameter Policy ownership/topology in the checkpoint
identity/manifest path implemented by the trainer integration.

Resume is fail-closed when the saved optimizer/component ownership topology is
not compatible with the requested run. DTS does not silently reinterpret a
checkpoint as a different Component policy merely to make resume proceed.

This is intentionally stricter than treating a checkpoint as weights-only:
optimizer state must correspond to the same routed ownership contract.

## Supported backend matrix

Component Start is currently qualified for exactly these ten runtime backends:

| Visible page / family | Runtime backend |
| --- | --- |
| SD 1.5 / SD2 LoRA | `sd-lora` |
| SDXL LoRA | `sdxl-lora` |
| SD 1.5 / SD2 DreamBooth | `sd-dreambooth` |
| SDXL Full page (`sdxl-full`) | `sdxl-finetune` |
| SD3 / SD3.5 LoRA | `sd3-lora` |
| Flux LoRA | `flux-lora` |
| Chroma LoRA | `chroma-lora` |
| Flux Full | `flux-finetune` |
| Anima LoRA | `anima-lora` |
| Anima Full | `anima-finetune` |

The page/backend alias is important for SDXL Full: presets are filtered against
the visible page key `sdxl-full`, while the host resolves it to
`sdxl-finetune` for runtime validation.

## Qualification blockers vs semantic blockers

Two blocker classes are intentionally different.

### Qualification blockers

These are combinations that may work in Standard mode but are not yet qualified
with Component ownership/checkpoint semantics. Current examples include:

- full FP16 / full BF16;
- FP8 base / FP8 base U-Net;
- `torch_compile` and Anima per-block compile;
- DeepSpeed;
- fused backward, fused optimizer groups and blockwise fused optimizers;
- CPU/Unsloth checkpoint offload;
- block swap/offload paths;
- explicit multi-GPU selection.

These restrictions are Component-only. They do not remove the corresponding
historical Standard features where the trainer already supports them.

### Semantic blockers

These are features whose optimizer/LR behavior cannot be represented
losslessly by the current Component schema, or whose conditioning/state
semantics conflict with static component ownership. Current examples include:

- preloaded LoRA/network weights combined with Text Encoder output caching on
  affected LoRA backends;
- Anima LoRA base-weight merge combined with Text Encoder output caching;
- `scale_weight_norms` modifying frozen adapter state;
- SDXL Full `block_lr`;
- SD DreamBooth dynamic `stop_text_encoder_training`;
- SD/SDXL LoRA block-LR weighting;
- LoRA+ parameter-level LR groups;
- Flux/Chroma/Anima `network_reg_lrs` regex LR groups;
- an explicitly selected, unreviewed non-canonical LoRA `network_module`.

The exact error text emitted by Preview/Start is the authoritative diagnostic.

## Troubleshooting and returning to Standard

If Component Preview is blocked:

1. Read the Preview blocker text first. It normally names the incompatible
   field or semantic feature directly.
2. Disable the advanced option or adjust the policy as indicated.
3. If the current browser state has accumulated unrelated settings, use Reset
   All and migrate again from a clean Standard configuration.
4. You can switch `Optimization Mode` back to Standard at any time. Hidden
   Component state is ignored by Standard compilation.
5. Imported policies containing restricted optimizer types remain readable for
   compatibility, but that does not make those optimizers runnable in Component
   mode.

For new workflows, a useful pattern is to configure a working Standard run
first, Preview it, then switch to Component and let the exact bootstrap seed the
policy before making component-specific changes.
