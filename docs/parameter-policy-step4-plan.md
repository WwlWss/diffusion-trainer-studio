# Parameter Policy Step 4 Plan — LoRA Original-Target Metadata

## Scope

Step 4 wires real LoRA adapters to the Step 3 metadata-driven router. It does
not construct optimizers/schedulers, mutate `requires_grad`, integrate
Accelerate, or unblock Component Start.

The architectural invariant is:

```text
base target module
  -> LoRA creation site captures root/path/type
  -> versioned runtime-only marker on the LoRA module
  -> scanner auto-discovers marker
  -> AdapterTargetMetadata
  -> Step 3 component classifier / Muon eligibility
```

No routing decision may be reconstructed from `lora_name`, `lora_down` /
`lora_up` shapes, or checkpoint keys.

## 4A — marker transport + scanner auto-discovery

The trainer-side transport is deliberately dependency-free:

```python
lora._dts_parameter_policy_target_v1 = (
    target_root,
    target_module_path,
    target_module_type,
)
```

The marker contains only three strings. It must not contain a Module, Tensor,
Parameter, class object, weak reference, or any object whose assignment could
alter PyTorch module registration/state.

`mikazuki.parameter_routing` owns conversion into
`AdapterTargetMetadata`.

### Scanner rules

- inspect the already collected `named_modules(remove_duplicate=False)`
  entries; do not perform a second module traversal;
- marker format is exactly `tuple[str, str, str]`;
- malformed markers fail scanning closed;
- explicit `adapter_targets=` remains supported;
- identical explicit + attached metadata is allowed;
- conflicting explicit + attached metadata is a hard error;
- nested attached registrations with conflicting metadata remain a hard error;
- descendants such as split-QKV `lora_down.0.weight` inherit the nearest
  consistent adapter marker through existing ancestor propagation;
- roots with no attached/explicit metadata preserve the existing fast path.

## 4B — native emitters

Add the marker immediately after each real adapter is constructed, while the
original `child_module` and unflattened module path are still available:

- `scripts/stable/networks/lora.py`: SD / SDXL;
- `scripts/dev/networks/lora_flux.py`: Flux / Chroma;
- `scripts/dev/networks/lora_sd3.py`: SD3.

Do not change constructor signatures, `lora_name`, target filters, rank/alpha,
`apply_to`, forward, state_dict, optimizer preparation, save/load, or
checkpoint schema.

## 4C — Anima isolated metadata patch

Do not modify the pinned `sd-scripts` gitlink. Add an exact-source,
fail-closed patch tool for `networks/lora_anima.py` and compose it with the
existing content-addressed Anima runtime staging.

The metadata patch is an independent feature and must compose with Qwen3 joint
training and Multi-Caption patches.

## 4D — end-to-end contract + CI

Validate marker -> scanner -> component classifier -> Muon
primary/fallback routing for SD/SDXL, Flux/Chroma, SD3 and Anima representative
targets. CI must remain PyTorch-free for host contract tests.

## Non-goals

Step 4 does not:

- construct or step optimizers/schedulers;
- mutate `requires_grad`;
- transfer trainer-side LR ownership;
- support LoRA+, regex LR, LyCORIS, DyLoRA or OFT;
- change LoRA checkpoint/state_dict keys;
- unblock Component Start.
