# QG-2-B0 — Direct Trainer / Launcher source architecture review

**Frozen source HEAD:** PR #69, 598a81aee3a1b1181ee333055e6b3268fa24fc8d (Draft, unmerged).
**Current cumulative gate:** B301–B311 GREEN; B312 is the single intentional RED. No production files changed in B0. No CUDA, Checkpoint, qualification evidence or experimental Start permission is added by this review.

## 1. Executive conclusion

The Direct Trainer architecture does not need a replacement. It needs a **separate pure Host Admission preflight** before model/GPU work, followed by the existing model-aware Parameter Policy session and its live runtime checks.

The current trainer integration was deliberately placed after the model and network objects exist because routing requires actual torch.nn.Parameter roots. Moving the entire create_parameter_policy_session() earlier would be incorrect. Conversely, relying on that late session for source/certificate admission is too late: by then the Accelerator, dtype, base model, VAE, Text Encoders, LoRA network and optional network weights may already have been initialized/loaded.

QG-2-B must therefore split:
1. **Early Host Admission:** strict one-invocation CLI intent + canonical Policy + final effective config + first-party ComponentAdmission and existing host blockers; no torch-heavy trainer import, no Accelerator/model/tokenizer cache/Sidecar/Checkpoint write.
2. **Late live session:** after the actual model/network roots exist, revalidate the same intent and same Policy/config identity, then retain all existing semantic, execution, root-routing, optimizer, dtype/device, distributed and Checkpoint guards.

The first stage is not a replacement for the second. Both must fail closed.

## 2. Current source call graph and exact seams

### CLI and config ingress

- scripts/stable/train_network.py:1378-1387 and scripts/dev/train_network.py:2021-2030 call parser.parse_args(), train_util.verify_command_line_training_args(args), train_util.read_config_from_file(args, parser), then trainer.train(args). No typed admission intent is currently constructed or passed.
- Both scripts/stable/library/train_util.py:3075 and scripts/dev/library/train_util.py:3689 add the existing --parameter_policy_config argument through add_training_arguments(). Neither parser defines an experimental acknowledgement flag.
- Both read_config_from_file() implementations flatten every TOML section into ignore_nesting_dict, construct argparse.Namespace(**ignore_nesting_dict), then call parser.parse_args(namespace=config_args). This is a critical trust boundary: merely adding a store_true argparse action does not prevent the same destination from being supplied by TOML/Namespace merge. The reserved acknowledgement key must be explicitly rejected in TOML before the Namespace is built. The output_config path serializes vars(args); the control field must be excluded and positive acknowledgement must never be written to generated TOML.
- verify_training_args(...parameter_policy_runtime_enabled=...) calls _guard_parameter_policy_runtime(): if a policy sidecar is supplied to a trainer whose get_parameter_policy_train_type() returns None, it fails closed instead of silently falling back to the legacy optimizer path. Preserve this.

### Existing early preflight and lazy boundary

- Both trainer bases already detect parameter_policy_requested and determine parameter_policy_train_type near the beginning of train().
- They then lazily import library.dts_parameter_policy_bridge and call preflight_component_runtime_booleans() before verify_training_args, dataset preparation, logging, tokenizer loading, Accelerator creation and model loading.
- scripts/dev/library/dts_parameter_policy_bridge.py and scripts/stable/library/dts_parameter_policy_bridge.py deliberately keep parameter_policy_trainer lazy. The pure early preflight imports only mikazuki.parameter_policy_runtime_bools.
- mikazuki/parameter_policy_trainer.py imports torch at module import time. Do NOT call load_parameter_policy_trainer() or import this module from the new early Admission preflight. Standard mode must remain a true no-op and must not acquire the DTS torch-heavy dependency.
- Historical reason: the October 29 fix/component commit introduced this exact pure-early/lazy-heavy pattern to reject quoted pseudo-booleans without importing the full trainer. QG-2-B should extend that architecture, not remove it.

### Actual train lifecycle

Stable base:
- scripts/stable/train_network.py:145-165: policy request/type and early boolean guard.
- scripts/stable/train_network.py:258: train_util.prepare_accelerator(args).
- scripts/stable/train_network.py:262-266: dtype preparation then load_target_model().
- scripts/stable/train_network.py:362-371: after model/network creation and network.load_weights(), create the Parameter Policy session.

Dev base:
- scripts/dev/train_network.py:512-535: policy request/type and early boolean guard.
- scripts/dev/train_network.py:643-651: Accelerator, dtype and base model loading.
- scripts/dev/train_network.py:758-767: after network application/optional weight load, create the Parameter Policy session.

Both variants call self.assert_extra_args(...) after dataset group preparation and immediately before prepare_accelerator(). Model-family overrides may normalize effective config there. For example, scripts/dev/flux_train_network.py:40-64 sets fp8_base=True when fp8_base_unet is enabled and may promote cache flags; it must be reflected in the final Admission. A robust implementation should run an initial fail-closed Host check before expensive tokenizer/dataset work, then re-evaluate after assert_extra_args() and before Accelerator/model load. The second result is the final pre-model gate.

There are two related side-effect details:
- Stable train_util.load_tokenizer() can write a tokenizer cache directory; the first Admission check should happen before tokenizer loading.
- Flux get_text_encoder_outputs_caching_strategy() calls bridge load_parameter_policy_train_flags() (around scripts/dev/flux_train_network.py:244-270). The current bridge resolves this through the torch-heavy trainer module. Do not use this helper for the early Admission path; the new pure preflight must use the canonical host-side loader directly.

### Late model-aware session and runtime guarantees

- mikazuki/parameter_policy_trainer.py:18 imports torch.
- load_parameter_policy_file():187-206 reads the JSON sidecar, calls the first-party validate_parameter_policy(), canonicalizes it with serialize_parameter_policy() and computes a SHA-256 policy hash.
- create_parameter_policy_session():1618-... currently snapshots effective config, loads the Policy, checks semantic blockers, calls parameter_policy_execution_blockers(...require_assets=True), builds the execution contract, then scans actual model roots, builds routing/runtime/optimizer state and applies the requires-grad contract. It currently does not call assess_component_training_admission() and accepts no experimental_full_bf16_ack intent; this is the exact B312 missing seam.
- This session cannot move before model/network construction: scan_parameter_roots() requires real Parameter objects, alias topology and model root identity. Keep it after network.apply_to() and network.load_weights(), but ensure the pure Host gate has already passed.
- After Accelerator preparation, ParameterPolicyTrainerSession.assert_runtime_contract() checks live root identity, requires-grad ownership, trainable dtype, actual accelerator.num_processes / distributed_type, optimizer parameter identity and device. finalize_after_prepare() runs these checks before registering Checkpoint hooks.
- parameter_policy_execution_environment_blockers() deliberately accepts actual Accelerator facts rather than importing Accelerate or trusting launcher environment variables. Full-BF16 requires single-process and distributed_type=NO. Keep this live post-Accelerator check: a preflight with missing WORLD_SIZE/config hints must never claim that absence proves a single-process runtime.
- Checkpoint manifests contain Policy hash, runtime topology, components, optimizer profiles and execution identity/signature. They contain no acknowledgement. Keep them acknowledgement-free; resume must not restore an opt-in.

## 3. First-party Admission already supports Direct Trainer purpose

mikazuki/parameter_policy_admission.py:202-215 already accepts purpose="direct_trainer" and a strict experimental_ack: bool. It returns:
- inactive for inactive Full-BF16 (no-op);
- legacy_approved for the exact existing source-bound SD1/SDXL D1 compatibility path;
- ack_required when an otherwise audited candidate has no explicit one-job confirmation;
- experimental_host_candidate when an otherwise audited Direct Trainer invocation supplies strict True;
- blocked for real implementation/capability/environment incompatibilities.

A positive acknowledgement cannot turn an unimplemented backend into a candidate. In particular SD3/Flux/Chroma remain blocked by the source-owned readiness producer; no direct CLI flag can promote them.

The current parameter_policy_runtime_blockers() in mikazuki/parameter_policy.py:543-725 is still hardcoded to experimental_ack=False and uses Request-oriented purpose selection. Direct Trainer currently calls the lower-level legacy parameter_policy_execution_blockers() from the late session instead. B312 must introduce a typed direct-trainer path that reuses the same first-party classifier and existing semantic/target/cache/optimizer hard gates. Do not duplicate a weaker readiness classifier in sd-scripts, and do not suppress hard errors by filtering strings such as pending.

## 4. Integrated trainer family matrix

- **Stable network family:** scripts/stable/train_network.py:get_parameter_policy_train_type() returns sd-lora or sdxl-lora according to self.is_sdxl; stable SDXL subclasses this base. This is the current D1 source-audited direct-trainer compatibility path.
- **Dev base:** scripts/dev/train_network.py:get_parameter_policy_train_type() returns None. Dev SDXL/Lumina paths that do not override it remain fail-closed through verify_training_args(parameter_policy_runtime_enabled=False).
- **Dev SD3:** scripts/dev/sd3_train_network.py:get_parameter_policy_train_type() returns sd3-lora.
- **Dev Flux/Chroma:** scripts/dev/flux_train_network.py:get_parameter_policy_train_type() returns flux-lora or chroma-lora.
- SD3/Flux/Chroma have explicit trainer integration but are not source-qualified for Full-BF16 by the current readiness producer. Their CLI acknowledgement must still result in blocked.
- Do not widen the integrated backend matrix or change any model-family get_parameter_policy_train_type() behavior in QG-2-B without a separate reviewed reason.

## 5. Launcher/API transport seam

- mikazuki/training_launcher.py:20-102 run_prepared_train() currently accepts TOML path, trainer file, GPU IDs, CPU threads, page type and run ID. It constructs an Accelerate argument list ending in trainer_file, "--config_file", toml_path, copies os.environ, and creates the task. There is no admission-intent parameter or child CLI acknowledgement flag.
- A child script argument must be appended **after trainer_file** (for example after --config_file toml_path) so Accelerate forwards it to the trainer script rather than consuming it as an Accelerate option.
- mikazuki/app/training_api.py:create_toml_file() currently calls run_prepared_train() without passing prepared.admission_intent. A2 already captures the one-job WebUI intent but SR1 deliberately holds experimental_host_candidate before the launcher. QG-2-B must wire a typed launcher argument and prove the transport in isolation without removing that HOLD.
- Do not transport acknowledgement through environment variables: run_prepared_train() inherits os.environ.copy(), so an environment value can leak into later jobs/ranks. Do not add it to task metadata, TOML, Bundle, Sidecar, checkpoint manifest or resume state.
- Direct CLI and API-launched child processes can use the same one-shot trainer CLI flag; each Accelerate worker receives the argument and independently performs the same Host Admission. The flag is human intent, not a cryptographic credential; OS shell history/process listings may expose the command line, so it must never be treated as a secret.

## 6. Proposed QG-2-B control-plane design

1. Introduce a new immutable DirectTrainerAdmissionIntent (do not widen A1's RequestAdmissionIntent.origin enum). It has a strict bool and is constructed only from the current invocation's explicit CLI flag. The WebUI parent may translate its existing RequestAdmissionIntent(True, "webui_start") to the same child CLI flag through the launcher.
2. Add one canonical strict opt-in CLI flag to the dev and stable train_util.add_training_arguments() parser paths. Extract it immediately after read_config_from_file(), remove it from args before effective-config construction/Trainer logging, and pass the typed intent separately to trainer.train().
3. Explicitly reserve/reject the acknowledgement key in both read_config_from_file() implementations before TOML Namespace merge. Also exclude it from --output_config; reject positive acknowledgement in output-config-only mode. A parser action alone is not a TOML trust boundary.
4. Extend both lazy trainer bridges with a pure early Admission function. It may import parameter_policy.py, parameter_policy_admission.py, optimizer/readiness metadata and JSON/path helpers, but must not import parameter_policy_trainer.py or torch. Prefer extracting the existing canonical sidecar loader/hash into a shared pure module so early preflight and late session cannot drift.
5. At the beginning of train(), after basic argument/deepspeed normalization but before logging, tokenizer cache, dataset work and Accelerator, run the first strict host preflight. After assert_extra_args() (which may normalize FP8/cache/backend flags) and immediately before prepare_accelerator(), re-evaluate against the final effective config. Both use purpose="direct_trainer" and the same per-invocation intent. blocked and ack_required reject; inactive and exact legacy_approved preserve prior behavior. An acknowledged experimental_host_candidate remains an explicit **QG-2-B/final-review HOLD** at this stage.
6. Create a frozen preflight snapshot with Policy canonical hash and an effective-config fingerprint. Pass it and the same intent to create_parameter_policy_session(). The late session re-loads/validates the policy and compares hashes/fingerprint, re-runs first-party Admission before root scanning, then retains existing semantic/execution/target/cache/optimizer checks and the live root/optimizer/device/dtype/environment/Checkpoint contract. This closes a sidecar/config TOCTOU gap without moving root scanning before model construction.
7. Add a typed optional intent to run_prepared_train(); the API passes prepared.admission_intent. The launcher appends the CLI flag only for a valid strict positive Request intent, after the trainer script. No env or task metadata field. The Request-level experimental candidate remains held until the separate final full-chain review authorizes removing it.
8. Do not change parameter_policy_execution_environment_blockers() or its actual Accelerator inputs. Its post-prepare check is a distinct live-runtime guarantee, not replaceable by config or env hints.

## 7. QG-2-B1 Tests-only RED design

Keep the cumulative B312 contract to **exactly one targeted failing test identity** so the inherited QG-1B3/QG-2 workflow continues to report B301–B311 GREEN and only B312 RED. Expand the existing test_b312_direct_trainer_uses_same_admission_and_explicit_cli_intent into one aggregated contract test (no subTest failure fan-out) that checks all of the following missing source seams:
- strict CLI-only flag in both stable/dev parser paths; TOML reserved-key rejection and --output_config non-persistence;
- typed per-invocation intent extracted/removed from args and passed separately to train/session;
- first pure Admission preflight before tokenizer/dataset/Accelerator/model load, plus post-assert_extra_args final-config preflight;
- no torch-heavy trainer import from the early bridge;
- exact first-party assess_component_training_admission(...purpose="direct_trainer") and explicit ack in both early and late gates;
- blocked/ack-required/unsupported backend cannot proceed even with ack; candidate remains HOLD; D1 legacy and inactive remain compatible;
- Policy/config snapshot identity is compared before live root scanning;
- Launcher receives typed Request intent, appends the child CLI flag after trainer_file only for that invocation, and never writes ack to env/task metadata/TOML;
- API passes the prepared Request intent to the launcher;
- existing runtime dtype, live-root identity, actual Accelerator environment, optimizer/device and checkpoint/resume checks remain and checkpoint schema contains no acknowledgement.

Add real passing baselines for Standard no-policy no-op/lazy import, exact D1 without ack, unsupported backend fail-closed, and unchanged runtime/checkpoint invariants. Use AST-extracted/source-isolated functions in minimal Host CI; do not require torch/Accelerate/CUDA to validate control-plane parsing and ordering.

The dedicated QG-2-B1 workflow should run Ubuntu/Windows Python 3.11, validate the exact one B312 RED marker and true baselines, and preserve A1-SR1/A2/B301–B311 GREEN. No manufactured unconditional failures, no skipped tests or xfail. After valid RED, stop before production changes.

## 8. Stop line

QG-2-B0 is architecture review only. No production source, tests or CI have been modified by this review. QG-2-B1 must freeze the contract/RED first. Later GREEN may touch only the justified Direct Trainer/Launcher/config parser seams and tests. Do not alter qualification records, optimizer implementation, Checkpoint schema, CUDA evidence, actual model family allow-list, A2 candidate HOLD or PR merge state. A fresh full-chain Strict Review is required after B312 GREEN.
