# D1-SD3 C4-B — Genuine Checkpoint Save Source Contract / Tests-only RED v1

**STATUS: SOURCE-AUDIT / TESTS-ONLY RED, NOT GREEN.** Baseline PR #65 C4-A-SR1 Host/pinned CPU PASS is 644537bfa6c9a1cfa23873d2027c91394d8cde00. The PR stays Draft/unmerged. This document grants no production Save receipt, CUDA, Resume, worker_complete or merge eligibility.

## 1. Actual production source and call order

1. scripts/dev/train_network.py lines 1003–1056: existing Network pre-save hook removes non-LoRA model weights from Accelerator's staged weights and writes train_state.json (current_step.value+1 and current_epoch). The original SD3 Forward Semantics and Resume Progress pre-hooks are registered via Sd3NetworkTrainer.register_additional_state_hooks. The Parameter Policy Session independently registers its checkpoint manifest and progress pre-hooks.
2. scripts/dev/train_network.py lines 1403–1416 and 1600–1625: after a genuine physical Step, the actual accelerator.unwrap_model(network).save_weights writes the external <output_name>-step0000000N.safetensors using scripts/dev/networks/lora_sd3.py lines 771–796. This happens FIRST.
3. scripts/dev/library/train_util.py lines 5991–6000: the SAME Accelerator instance then performs the genuine accelerator.save_state(<output_name>-step0000000N-state). Its existing production pre-save hooks write progress and policy sidecars.
4. scripts/dev/train_network.py lines 1835–1843 and train_util.py lines 6017–6025: end-of-training <output_name>-state and final Adapter are ancillary LAST saves, not additional Step evidence.
5. tools/sd3_full_bf16_observed_trainer.py lines 140–249: original SD3/Parameter Policy hooks are installed before the C3 Probe and C4-A Step Adapter. A C4-B Save wrapper does not yet exist. C4-A _check_bound_identity rejects unreviewed Accelerator instance-method replacement; GREEN must whitelist only its own identity-pinned Save wrapper without disabling that security check.

## 2. Evidence and authorization

The ONLY source of a Step save candidate is the frozen genuine C3 PhysicalStepProbe physical-Step Anchor accepted by C4-A. Caller metadata, a path, a pre-save hook, or a synthetic JSON file cannot create a verified checkpoint.

**Original call boundary:** install precisely one per-instance Accelerator.save_state wrapper only after original production hooks; pin original call target, original per-instance method-slot presence, and exact prepared Accelerator/Network/Optimizer/Scheduler/Phase Ticket. For step-named Save require the pending, unexpired C4-A real Step Anchor and exact preapproved Phase/Case paths. Snapshot the live prepared Network state_dict including Alpha; actual ordered Child states; Scheduler; Step and RNG fingerprints. No rebaseline or GPU full-model clones. Delegate original save_state exactly once with the same arguments/kwargs. No retry, no other Accelerator.

**Post-return boundary:** only after original returns normally, read the actual external Step Adapter and saved state directory, compare exact paths, regular files (no link/symlink), file SHA256s, typed Tensor keys/shapes/dtypes/bytes, complete 789-tensor production state, independently audited Phase/Step/Child/RNG/progress identity and the true on-disk checkpoint_id. No checkpoint_saved before every check passes. An actual directory, successful prehook or file timestamp alone is not success.

**Nine mandatory production state files:** model.safetensors, optimizer.bin, scheduler.bin, random_states_0.pkl, train_state.json, dts_parameter_policy_manifest.json, dts_parameter_policy_progress.json, dts_sd3_resume_progress.json, dts_sd3_forward_semantics.json. This exact inventory is already defined by tools/sd3_full_bf16_evidence_support.py. Do not introduce receipt files inside the state dir. Accelerate writes ONE optimizer.bin containing the CompositeOptimizer child states (R2 Muon + separate AdamW), and ONE scheduler.bin; NEVER require two separate Child optimizer files. Untrusted Host must not unpickle optimizer.bin or random_states_0.pkl. The independent Host verifier may validate exact file SHA + typed sidecars; runtime must attest genuine serialization sources and state hashes.

**Fixed real SD3.5 Medium production LoRA:** 263 LoRAModules, 526 trained BF16 projection weight Parameters plus 263 separate BF16 Alpha Buffers (scalar 4), 789 serialized Tensor records total. The C3 trained-Parameter digest covers only 526; it must NEVER be substituted for the 789 serialized keys. The real external LoRA Step file and actual Accelerate model.safetensors must match every Tensor descriptor/byte while allowing differing file-level metadata. Production LoRANetwork.save_weights must execute normally, not be replaced by the verifier.

**Production progress:** actual checkpoint_id comes from dts_parameter_policy_progress.json after real Save. Cross-check Step/Phase/Policy/Child/Scheduler and SD3 resume progress (completed_optimizer_steps, current_epoch, batches/accumulation, num_processes) and train_state.json. The source uses current_step.value+1 intentionally; do not rewrite the sidecar to force equality.

**Failure semantics:** original Save failure, prehook exception, missing/extra file, failed 789 inventory, mismatched weight/Alpha bytes, stale Anchor or post-save drift must fail closed. Do not retry or erase partial evidence. Duplicate Save cannot mint a second receipt. No public mark_saved, clear_pending, emit or generic success API. Normal last Save delegates once and remains ancillary. Step-1 pending may become consumed only following authentic full file verification so Fresh Step-2 can proceed. C4-B cannot make C4-C Resume, C4-D phase or Fix D process lineage valid.

## 3. Proposed minimal C4-B internals — intentionally absent in RED

- C4StepAdapter.install() will manage a single same-Accelerator-instance save_state wrapper; original descriptor slot restored exactly on close/exception; other method shadowing remains rejected.
- C4StepAdapter.save_attempt_observation() returns read-only diagnostic result: real original invocation count, whether original returned, qualified status, failure cause; an R1/R2 CPU micro-fixture may prove real original Save but must NEVER assert full production qualification.
- C4StepAdapter.verified_step_save(logical_step=N) returns an immutable typed C4-B Step Save receipt only after genuine full post-return verification; otherwise raises.
- tools/sd3_full_bf16_worker_save_verifier.py provides read-only serialized LoRA file/Tensor validation and snapshot_serialized_tensor_inventory(network, diagnostic_cpu=True). This diagnostic mode describes real 18 tensors from six LoRAModules but can NEVER mint the production 789-tensor/9-file receipt. Production needs exact independent 263-target spec and true serialized file bytes.

These interface names and boundaries are test targets, not existing functionality or a GREEN claim. The implementation must preserve inheritance and authentic hook order and must not change production save/load source.

## 4. Exact pinned CPU RED suite

New tests/test_sd3_full_bf16_worker_c4b_cpu.py uses actual Torch BF16 R1/R2 physical Backwards, actual Muon/AdamW & Scheduler, actual original Accelerator.save_state, actual LoRANetwork.save_weights for a separate tiny real LoRA instance, and Host safetensors byte verification. Exactly 12 methods:

| Test | Operation | Expected old-source outcome |
|---|---|---|
| B01 | R1 physical Step and real four-file CPU Accelerate Save, in-memory BF16 model equality | baseline PASS |
| B02 | R2 real accum2 Muon + AdamW, one actual optimizer.bin/scheduler.bin | baseline PASS |
| B03 | Actual production LoRANetwork.save_weights and BF16 Alpha/Weight triplet | baseline PASS |
| B04 | Host rejects modified genuine safetensors bytes | baseline PASS |
| B05 | Managed per-instance save_state wrapper installed and removed | targeted RED |
| B06 | Post-return attempt observation; actual 4-file CPU state remains unqualified | targeted RED |
| B07 | Real pre-save hook failure cannot publish post-return Save evidence | targeted RED |
| B08 | Real C3 LoRAModule complete serialized inventory includes Alpha; diagnostic 18, not 789 | targeted RED |
| B09 | Unauthorized state path rejected BEFORE real save_state writes | targeted RED |
| B10 | Missing real RNG file cannot yield verified saved receipt | targeted RED |
| B11 | Real filesystem Save failure propagates, never reports checkpoint_saved | baseline PASS |
| B12 | Real ancillary last Save remains a non-authoritative tail | baseline PASS |

The separate actual production 1-module LoRANetwork CPU fixture tests serializer mechanics; the C3 R2 six-LoRAModule CPU diagnostics test physical trained values and all 18 small BF16 tensors. Their combination does NOT itself constitute one completed production 263-target CUDA run. Missing C4-B APIs must generate explicit C4-B CONTRACT ABSENT assertion FAIL, NEVER accidental import/fixture ERROR. RED gate: exactly 12 discovered and run, 0 ERROR/SKIP/xfail/xpass; all FAIL entries must be targeted and pre-existing tests retained. Positive baseline PASS is not RED.

## 5. CI, future GREEN, and hard stop

Add only the C4-B document, C4-B tests, exact-discovery pinned-CPU CI gate and path filters. Preserve F6–F11, C4-A 12, C4-A-SR1 8 and Host both OS. Run real Actions and archive exact commit SHA, red test outcomes, the separate PR/push C1-C3 results and the invalid-vs-valid distinction.

Later minimal C4-B GREEN requires real same-instance post-save wrapper, original called once, clean error propagation, complete state validation and genuine verified pending->consumed transition; rerun 12/12 C4-B, all old C4-A/SR1, Host and regression, then a NEW independent Strict Code Review. Tiny CPU Save is never production full-BF16 approval. Keep C3PhaseEvidence.finalize fail-closed, worker_complete forbidden, PR Draft; C4-C/C4-D/Fix D and physical R1/R2 CUDA remain blocked.
