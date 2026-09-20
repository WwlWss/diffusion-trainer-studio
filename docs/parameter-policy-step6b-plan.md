# Parameter Policy Step 6B — NetworkTrainer LoRA integration

Step 6B integrates the Step 6A trainer runtime boundary into the concrete LoRA
NetworkTrainer backends used by DTS while deliberately keeping the host Start
allow-list closed.

## Backends

Integrated trainer paths:

- stable SD LoRA: `sd-lora`
- stable SDXL LoRA: `sdxl-lora`
- dev Flux LoRA: `flux-lora`
- dev Chroma LoRA: `chroma-lora`
- dev SD3 LoRA: `sd3-lora`

Anima remains staged separately. Dev SD/SDXL NetworkTrainer variants are not DTS
backend targets and continue to fail closed when a managed policy is supplied.

## Ownership boundary

Standard mode keeps the historical sd-scripts optimizer path. Component mode
never calls `prepare_optimizer_params*`, `get_optimizer()`, or
`prepare_grad_etc()`. Instead it:

1. applies the LoRA network to the legacy target selected by trainer flags;
2. creates one ParameterPolicyTrainerSession over `{"network": network}`;
3. lets routing further freeze, but never expand, the trainer target;
4. uses the session CompositeOptimizer and CompositeLRScheduler;
5. audits ownership/device placement after Accelerate prepare;
6. registers the policy/topology/scheduler checkpoint manifest;
7. clips only session-owned trainable parameters;
8. reasserts the requires-grad contract after epoch-start hooks.

## Text encoder flags and caching

Flux/Chroma and SD3 convert policy component ownership into final per-encoder
training flags after routing. SD3 tracks CLIP-L, CLIP-G, and T5XXL independently.

Legacy cache conflicts are deferred in Component mode until these final routing
flags exist. Flux/Chroma and SD3 create text-encoder caches before the LoRA
network/session exists, so Component mode deliberately stages that pre-session
cache as a full cache with all TE train flags false. Routing then restores the
exact policy-owned flags. Caching is rejected only when the policy actually
trains an adapter whose text-encoder outputs would otherwise be cached.

## ScheduleFree lifecycle

Dev NetworkTrainer reuses its existing sample/validation/save lifecycle, but
Component mode binds those callbacks directly to CompositeOptimizer.train/eval.
Stable NetworkTrainer gains equivalent no-op-in-Standard lifecycle boundaries so
mixed Component optimizer profiles containing ScheduleFree children are safe.

## Logging and metadata

Component LR logs are emitted by component ID from RuntimeSpec and the prepared
CompositeLRScheduler. Legacy `lr_descriptions` and legacy optimizer-type-specific
logging are not authoritative in Component mode.

Saved LoRA metadata identifies the optimizer as `DTSParameterPolicy` and also
records policy hash, runtime topology fingerprint, and profile-to-optimizer
mapping.

## Fail-closed exclusions

`scale_weight_norms` is rejected for LoRA Component mode because the existing
sd-scripts max-norm implementation directly mutates every LoRA state tensor,
including policy-frozen adapters. Supporting max-norm later requires a
policy-owned subset-aware implementation.

A preloaded LoRA (`network_weights`) cannot be combined with Text Encoder
output caching for SDXL, Flux, Chroma, or SD3 Component mode in v1. Those
caches are created before the preloaded adapter weights are applied; a frozen,
nonzero Text Encoder adapter would therefore be omitted from conditioning.
Both in-memory cache and `cache_text_encoder_outputs_to_disk` fail closed for
this combination.

The global host Start allow-list remains empty in Step 6B. GUI Start is opened
only after the wider Step 6 matrix is validated.
