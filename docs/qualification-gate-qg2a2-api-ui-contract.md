# QG-2-A2 — API / WebUI one-job Admission contract and Tests-only RED

Frozen parent: PR #69 Draft at 347176de71d0a76b1892eab7c4cfcc4b4e8efa2a. QG-2-A1-SR1 is scoped Host GREEN; B311 API/UI and B312 Direct Trainer remain intentional RED. **This phase changes only tests, documentation and CI, not production API, UI, Launcher or Trainer.** No new experimental Full-BF16 launch permission, CUDA qualification, checkpoint schema change or PR merge.

## Source findings and design constraints

- mikazuki/training_request.py:43-49 decodes both legacy flat objects and nested {train_type,config} bodies, without separating a one-job control plane. Preserve this decoder for Bootstrap/Rehydrate/legacy requests. The API must parse the explicit Start control separately.
- mikazuki/training_config.py:40-79 already provides frozen RequestAdmissionIntent(ack:strict bool=False, origin default/webui_start) and PreparedTrainingConfig.admission/.admission_intent outside Trainer config. Reuse them without widening the allowed types or persisting them.
- mikazuki/training_request.py:185-377 already reruns the trusted classifier on final post-ui_custom_params effective config; denied or pending Start is rejected independently of Step 6F. The experimental_host_candidate is deliberately HOLD until BOTH API/UI and Direct Trainer transport exist. A2 must not remove this HOLD or Step 6F experimental_ack=False.
- mikazuki/app/training_api.py:84-112 _prepared_payload today serializes TOML, sidecars, portable bundle and runtime_ready/runtime_blockers but omits typed Admission. Keep TOML/bundle distinct from the advisory API admission field.
- training_api.py:179-201 Preview/Export both call prepare_request_config(launch=False) with implicit preview purpose; Export intentionally writes portable sidecars only AFTER validate_prepared_config(False) and _prepared_payload. Wire explicit purpose="preview"/"export" without changing this ordering. Rehydrate materializes only validated content-addressed bundle sidecars.
- training_api.py:236-275 Start compiles with launch=True; Component must validate assets read-only before prepare_runtime_trainer, validate_prepared_trainer, launch-only Anima staging, sidecars, TOML and run_prepared_train. Standard ordering must stay intact. Missing assets and blocked policy must never write files.
- mikazuki/frontend_training_patch.py:39-174 patches pinned JS with __trainingRequest posting JSON.stringify({train_type:t,config:raw}) for Preview/Export/Start. Form is persisted to localStorage. No experimental confirmation UI exists. Never put consent inside form a.value, localStorage, history/preset, ui_custom_params, TOML, rehydrate or bundle. Keep pinned anchor mismatch fail-closed.
- mikazuki/training_launcher.py and parameter_policy_trainer.py are separate Direct Trainer/Accelerate checks; do not modify in A2. B312 must remain independent RED.

## Frozen proposed API boundary

New helper in training_api.py:

    _decode_training_admission_request(body: bytes, *, purpose: Literal["preview","export","start"])
        -> tuple[page_type, raw_config, RequestAdmissionIntent]

For nested JSON {train_type, config}, the **only** allowed new control is an exact top-level property experimental_full_bf16_ack, strict literal true/false. For Start, true creates a fresh RequestAdmissionIntent(True,"webui_start") on the server; absent/false creates default false. Do not accept caller-supplied origin or derive authority from can_start, host_admissible, admission, user-provided readiness/evidence, cookie/header, prior tasks or exported config. Flat legacy JSON remains supported without any acknowledgement, but flat top-level ack must reject; nested config or ui_custom_params control injection remains forbidden. Reject pseudo-bools, null, arrays, objects, bad purpose or malformed nested config before any write.

Preview/Export must neutralize even an explicitly supplied true acknowledgement and never persist/emit it, consistent with A1 behavior. This freeze chooses **ignore+scrub on non-Start** rather than treating the value as permission. Never accept user-provided claim fields as a launch credential.

### API endpoints

1. Preview: explicitly decode with purpose preview, call prepare_request_config(launch=False,admission_purpose="preview",neutral intent), validate_prepared_config(False), _prepared_payload. Model-free; no materialization, assets/Trainer launch or GPU load.
2. Export: decode purpose export, pass neutral intent and admission_purpose export, validate prepared False, serialize payload, THEN deliberately materialize portable sidecars inside existing OSError boundary. No consent in TOML/bundle/sidecars or rehydrate.
3. _prepared_payload: if prepared.admission is a real first-party ComponentAdmission, add payload["admission"] with exactly status,purpose,requires_experimental_ack,host_admissible,evidence_tier,warnings,runtime_blockers as ADVISORY data. For inactive, evidence_tier may be None. No can_start/launch credential and no raw RequestAdmissionIntent. runtime_ready must continue to respect **all** effective runtime blockers, not just host_admissible. Standard response remains compatible. API-only admission never enters effective_config, toml or bundle.
4. Start: decode purpose start and pass fresh typed intent + admission_purpose start into prepare_request_config(launch=True,toml_path=...). Maintain SR1 HOLD for experimental_host_candidate despite true opt-in; Step 6F, GPU, model/assets/Trainer, Sidecar, TOML, run_prepared_train must remain in established order. For qualified D1 legacy and Standard, prior Start remains intact, but no new unreviewed backend can run.
5. Frontend: separate transient __experimentalFullBf16Ack state and explicit visible opt-in control; only a *single Start* includes top-level experimental_full_bf16_ack. Reset on submit/cancel/mode change/unmount; do not persist localStorage/sessionStorage/form/history/preset/export. Do not turn host_admissible into Start-ready; while B312 missing, experimental candidate stays disabled/HOLD. Do not pretend an API source change alone proves working UI: pinned frontend patch needs separate source behavior + Node/browser smoke on initialized submodule.

The control is a human-visible per-request opt-in, NOT a certificate, signature, model SHA or direct Trainer authorization. This A2 phase cannot open real experimental Start without B312 and independent full-chain review.

## Frozen Tests-only RED: twelve targets and four real baselines

New tests/test_parameter_policy_qg2a2_api_red.py, exact marker: QG-2-A2 API/UI CONTRACT ABSENT.

- A201: dedicated typed helper with exact recognized purpose; old decode unchanged.
- A202: nested top-level Start ack true/false/absent, strict bool and server-owned origin.
- A203: flat/nested/config/custom/forged authorization and pseudo-bool cases fail-closed.
- A204: Preview explicit purpose, neutral intent, unchanged model-free/no-write boundary.
- A205: Export explicit export purpose, no consent, deliberate portable Sidecar write order.
- A206: typed immutable first-party Admission status, warnings, evidence tier and blockers appear in response; Standard no forged admission.
- A207: no replay or serialization of opt-in in effective_config/TOML/portable bundle/Sidecar/Rehydrate, full-blocker runtime readiness.
- A208: Start one-job typed envelope before preflight and writes; no caller can_start authorization.
- A209: with true ack an otherwise supported experimental Host candidate is still held before asset checks/filesystem/Trainer/launch until B312.
- A210: denied models, missing assets, multi-GPU, invalid precision/unsupported backend continue hard stop; preserve legacy/Standard sequencing.
- A211: visible frontend-only transient checkbox and single-Start top-level control, no autosave/preset/history/export consent.
- A212: UI readiness uses full blockers and Direct Trainer HOLD, never host_admissible alone; no hidden GUI-only allow.

A901-A904 are genuine old-source baselines: real decode nested/flat; AST-isolated Preview no writes; AST-isolated Export validated payload before materialization; AST-isolated blocked Start no writes/launcher. Each must actually PASS. A201-A212 must each have one true missing-source failure with exact marker, zero ERROR, SKIP, xfail or xpass; no manufactured unconditional failure.

A new .github/workflows/qg2a2-api-host.yml runs Python 3.11 on Ubuntu and Windows. It first preserves QG-2-A1-SR1 8+4, A1 original 12+4, SR1 Fix C 5 and B301-B310+4 baseline GREEN; B311/B312 exactly two intentional RED. Then requires A201-A212 twelve exact FAIL and A901-A904 four actual PASS; it exits FAILURE on valid Tests-only RED (not faux GREEN). Real run logs and HEAD must be recorded and matched per OS.

**Stop after VALID RED:** Future A2 minimum GREEN may change only training_api.py, frontend_training_patch.py and directly affected tests. It must pass source-isolated endpoints, actual frontend control behavior and UI/Node smoke before scoped A2 signoff. B312 Direct Trainer/Launcher remains an intentionally RED independent gate; no merged PR, real CUDA, evidence promotion, optimizer or Checkpoint change.
