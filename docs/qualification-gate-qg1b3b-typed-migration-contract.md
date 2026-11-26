# DTS QG-1B3-B — typed certificate-only migration contract

**Parent:** QG-1B3-A scoped GREEN `416ff2b3b84583d0ce336c65041b6ab0c547425a` (PR #69, Draft). No QG-2 GUI/Request/Trainer permission or QG-3 SD3 Full-BF16 implementation is granted here.

## Purpose and source obligations

`parameter_policy_runtime_blockers()` in `mikazuki/parameter_policy.py` still invokes `parameter_policy_execution_blockers()` for all backends, where `_qualification_blocker()` converts a mere missing/pending release record into a launch-fatal string and early-returns before other execution findings. The QG-1B3-A `assess_component_training_admission()` is a typed, first-party, host-only classifier and must be the *sole* authority for deciding if an **audited** Full-BF16 Component configuration has a certificate-only gap.

**Limit scope to already source-audited** `sd-lora` and `sdxl-lora` implementations (with referenced audited optimizers AdamW/Muon). QG-1B3-B must not add `sd3-lora`, Flux, Chroma, Anima, SD2.x, unknown optimizers or unreviewed variants to the allowlist. Their preexisting legacy release diagnostics remain usable for backward compatibility, but *first-party hard capability objections must also block*, including under a forged `qualified` release row.

### Exact behavior

1. `full_bf16` **inactive**: preserve the old execution path, no new Full-BF16 admission/asset probe. Malformed quoted pseudo-booleans must retain the strict old execution blocker.
2. Active `full_bf16`: lazily evaluate the **real first-party** `assess_component_training_admission(canonical, train_type=..., effective_config=..., purpose=..., experimental_ack=False)`, with Preview/Export using `purpose="preview"` and launch-time evaluation using `purpose="start"`. `require_assets` is solely an existing legacy caller intent; it is **not** proof that file/model/dataset assets have been checked by the Host adapter.
3. Audited `sd-lora`/`sdxl-lora`, status `legacy_approved`: no new certificate blocker, preserve original normal D1 admission. Only the current exactly referenced D1 backend/optimizer source IDs establish this exemption. Wrong IDs are not a valid release.
4. Audited `sd-lora`/`sdxl-lora`, status `ack_required`: convert **only** certificate absence to the typed, fail-closed *missing acknowledgement* blocker (no free-text filtering). Never take an acknowledgement value from effective_config/TOML, `ui_custom_params`, the Worker or an exported bundle at this stage. The `experimental_ack=False` argument stays hardcoded until QG-2 adds a genuine per-job control channel. The result is still **not** Start-admissible.
5. **Any** `blocked` status: add its independent hard reasons. Retain legacy execution blockers as additional fail-closed compatibility/hard diagnostics. In particular, an unreviewed backend with a fake release record can never become Start-ready.
6. For backends **outside** the audited SD1/SDXL scope, retain old release diagnostics and add first-party hard reasons. Existing Flux/Anima/SD3 `pending` request regression assertions remain valid. Never suppress actual invalid precision, Source/Policy, semantic, target/TE cache, referenced primary/fallback optimizer, process/DDP, FP8/compile/offload, GPU selection, or Checkpoint guards.
7. Preserve the final existing `parameter_policy_runtime_blockers` return ABI (`list[str]`), deterministic ordered de-duplication, and exact existing Step 6F sequencing. This is a *typed source producer internally*, not yet a structured GUI payload. `training_request.py`, `training_config.py`, `app/training_api.py`, Direct Trainer, release tables, execution identity and sidecar/Checkpoint data MUST remain untouched.

**Avoid two dangerous shortcuts:** dropping messages containing the word `pending` (would discard real errors), and treating `readiness.can_start` or caller `can_start=True` as a launch credential. The branch's four original B309–B312 integration REDs must become exactly **one B309 GREEN + B310–B312 three RED** only after B3-B implementation. The broader PR remains Draft/unmerged.

## Freeze: QG-1B3-B Tests-only RED

New `tests/test_parameter_policy_qg1b3b_runtime_migration.py`: eight new `B3B101–B3B108` source/behavior contracts, all must fail with exact marker `QG-1B3-B TYPED CERTIFICATE MIGRATION ABSENT` while the old owner remains, and four true baseline `B3B901–B3B904` must pass. Additional baseline: previous QG1B 42/42, QG1B3-A 12/12 independent and B301–B308 8/8 all PASS, B309 still RED (without skipping/xfail).

Expected RED: Ubuntu and Windows Python 3.11, eight exact targeted FAIL + four baseline PASS, 0 ERROR/SKIP/xfail/xpass. The true Tests-only RED workflow **must fail its job** after validating exact failure identities and markers, not report green on failing tests.

## Minimum GREEN acceptance and HOLD

After the valid RED, change **only** `mikazuki/parameter_policy.py` plus narrowly necessary tests/workflow/docs. New eight tests + four baselines all PASS and B309 PASS, while B310–B312 remain exactly three intentional RED. Preserve all previous QG1B, QG1B3-A and existing Request/Anima/Parameter Policy regressions on both OS. New CI shall transparently label scoped Host GREEN and validate the three outstanding integration REDs without skipping.

Stop before QG-2 Request/Preview/Export/Start typed output, GUI per-job acknowledgement, trainer CLI flag/launcher transport, CUDA, evidence promotion or checkpoint changes. Independent final strict review and merge decision follow all integration steps, not this limited gate.
