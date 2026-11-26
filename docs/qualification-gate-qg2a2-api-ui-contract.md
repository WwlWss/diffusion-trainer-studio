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

## 6. QG-2-A2 minimum GREEN implementation and exact evidence

**Implementation scope:** only `mikazuki/app/training_api.py` and
`mikazuki/frontend_training_patch.py` were changed in production. Existing
Request/Admission/Step6F/asset/Trainer/Launcher/Checkpoint code and release
tables are untouched. Test, CI and documentation files were updated; no
submodule commit was changed.

### API control plane

- `_decode_training_admission_request(body, purpose)` now validates the
  purpose and nested JSON envelope. Only an exact top-level boolean in a
  nested envelope is considered. The server constructs origin
  `webui_start`; caller origin/diagnostics are never trusted. Flat legacy
  JSON remains supported, including the prior non-object flat `config`
  fallback, while a malformed nested `train_type/config` envelope fails.
  Nested `experimental_full_bf16_ack`, `admission_intent`, and `admission`
  injection is rejected.
- Preview and Export explicitly call Request with their own purposes and a
  neutral intent, even if a client sends true. Preview remains no-write.
  Export retains validate → payload → deliberately materialize sidecars
  ordering and the prior OSError boundary.
- Start parses its one-job top-level intent before preparation. It still
  calls SR1 Request/Step6F, and `experimental_host_candidate` is still
  rejected before asset checks, Trainer staging, sidecar/TOML writes or
  launcher invocation. Component asset/Trainer/Anima/Sidecar/TOML ordering
  and Standard's old ordering are unchanged.
- `_prepared_payload` emits a typed, advisory-only `admission` object
  (status, purpose, acknowledgement requirement, host-admissible diagnostic,
  evidence tier, warnings and typed blockers) only from an actual first-party
  `ComponentAdmission`. It is separate from effective Trainer config, TOML,
  portable Bundle and Sidecars; there is no `can_start` token. Public
  `runtime_ready` still uses the full runtime blocker set, not
  `host_admissible` alone.

### Pinned frontend control

- `frontend_training_patch.py` adds a separate transient
  `__experimentalFullBf16Ack` ref and displays a visible checkbox only when
  a Component Full-BF16 Preview reports `requires_experimental_ack`.
- The shared request helper adds top-level
  `experimental_full_bf16_ack=true` only for `/api/run` and only when the
  explicit checkbox is true. Preview, Export, Bootstrap and Rehydrate do not
  send it. It never enters `a.value`, localStorage, sessionStorage,
  history/preset, TOML, Bundle or Sidecars.
- Consent is consumed/reset on Start attempt, reset on form/config changes
  and mode transitions, and reset on unmount. Both checkbox visibility and
  Start submission require the typed Preview status `ack_required` plus the
  explicit checked state; a generic `requires_experimental_ack` flag alone
  cannot enable the control. The Start button can submit an explicit
  candidate attempt, but the server continues to return the SR1 QG-2-A2/B
  HOLD response. No actual experimental training can start.
- The pinned frontend submodule was checked out in a separate CI job. The
  real `layout.96d49288.js` is patched by the production patch function,
  syntax-checked by Node, and the extracted real request/Start handlers are
  behavior-tested. This is a Node behavior smoke, not a claim of full browser
  E2E automation.

### Exact dual-platform acceptance

At exact source HEAD
`b611477337aca616659aa8a6da5810ebcfa71f4b`:

- [QG-2-A2 API/UI + frontend smoke  #42464477769](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42464477769):
  **Ubuntu SUCCESS and Windows SUCCESS**. A201–A212 **12/12 PASS**;
  A901–A904 **4/4 PASS**; inherited B301–B311 plus B901–B904 PASS;
  B312 is the **exact single remaining intentional RED**. Pinned frontend
  smoke: **6/6 PASS on each OS**, including patched JavaScript syntax,
  Start-only envelope and one-shot acknowledgement consumption.
- [QG-1B3-A/B + QG-2-A1 staged Host  #42464477698](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42464477698):
  SUCCESS on both OS; B301–B311 and inherited host baselines pass. The
  standalone B3B workflow remained queued at evidence capture; the
  successful inherited stage executed its 12 B3B tests.
- [QG-2-A1-SR1  #42464477700](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42464477700),
  [QG-2-A1 Request  #42464477746](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42464477746),
  [QG-1A  #42464477749](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42464477749),
  [QG-1B  #42464477696](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42464477696)
  all SUCCESS.
- [Cumulative admission #42464477724](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42464477724)
  intentionally remains FAILURE for exactly B312; its Ubuntu and Windows
  logs identify the original `QG-1B3-QG2 ADMISSION CONTRACT ABSENT` marker.
  [Anima umbrella #42464477760](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42464477760)
  reports 758 tests, exactly the same B312 failure, and 64 optional-dependency
  skips. Neither is a full project GREEN or merge authorization.

**Verdict:** QG-2-A2 scoped API + WebUI GREEN; B311 GREEN; B312 remains
the only integration hold. PR #69 remains Draft/unmerged. No production
Direct Trainer/Launcher changes, no new CUDA evidence, no checkpoint
changes, and no experimental Full-BF16 Start permission.

### Next stage

Proceed to QG-2-B Direct Trainer / Launcher control-plane parity with its
own Tests-only RED. It must independently parse a strict one-job direct-CLI
opt-in, preserve all live dtype/root/optimizer/device/Checkpoint checks, and
match the same first-party Admission decisions. Only after B312 GREEN and a
new independent full-chain Strict Review can the experimental Host candidate
HOLD be reconsidered.
