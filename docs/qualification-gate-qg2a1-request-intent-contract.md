# QG-2-A1 — Request-scoped Admission/experimental intent; implementation plan and Tests-only RED

**Frozen parent:** `PR #69@cb378018117e919255a92e26b368ebf60e4d69d3` (QG-1B3-B scoped Host GREEN, Draft, stacked on #68/#67). **This stage is contract + Tests-only RED, not production integration or authorization.** New source changes to Request/GUI/Trainer/Checkpoint are prohibited until RED is accepted. PR #65 optional formal attestation remains isolated.

## 1. Source seam and failure modes

- `mikazuki/training_request.py:37-43` `decode_training_request(body)` currently returns `(page_type, config)`; for nested requests it copies `payload["config"]`, and for flat ones returns the payload dict. It does **not** distinguish control-plane acknowledgement from arbitrary trainer options. A new GUI/API envelope parser belongs to **QG-2-A2**, not A1. No acknowledgement may be inferred from existing `config` values, `ui_custom_params`, `__ui_custom_overrides`, `bundle`, `runtime_ready` or a claimed certificate.
- `training_request.py:179-239`: (1) raw UI numeric/Policy normalization; (2) content-addressed Policy/Multi-Caption/prompt sidecars are staged in memory; (3) `prepare_training_config`; (4) verify host-owned Policy/Caption paths; (5) Step 6F `parameter_policy_runtime_blockers()` plus GPU blockers; (6) on `launch`, reject before `materialize_sidecars`. This ordering must stay unchanged.
- `training_config.py:165-236` runs `apply_ui_custom_overrides` last, then `_post_override_normalize`; the new admission MUST use `prepared.train_type` and `prepared.config`, not raw GUI fields or pre-override config. `PreparedTrainingConfig` currently has no admission/intent fields. Adding fields with defaults must retain existing positional constructions and Standard-mode behavior.
- `mikazuki/parameter_policy.py:543-725` QG-1B3-B calls the trusted adapter with `experimental_ack=False` and preserves a pending-ack blocker. **Do not update that constant in A1**. QG-2-A2/2-B must perform real dual-entrypoint per-job consent transport; a Request host candidate cannot bypass Step 6F or direct Trainer.
- `app/training_api.py:179-201,237-275` Preview/Export are model-free preparation, Export deliberately writes portable sidecars, Start validates assets before staging/sidecar/TOML/launch. `parameter_policy_trainer.py:create_parameter_policy_session()` still validates loaded sidecar and old source gate before optimizer/root scan. No change to these entrypoints in A1.

## 2. Freeze the **proposed** A1 ABI

### Immutable request-level intent (new `mikazuki.training_config.RequestAdmissionIntent`)

`@dataclass(frozen=True, slots=True)` with:
- `experimental_full_bf16_ack: bool = False`, strict `type(value) is bool`; `"true"`, `"false"`, `1`, `None`, object/mapping are invalid, never normalized.
- `origin: Literal["default","webui_start"] = "default"`, strict exact enum. `ack=True` requires `origin="webui_start"`; this is **user intent metadata**, never a signed privilege, identity proof, device/asset certificate or authorization token. A2 must supply it from a dedicated, visually explicit WebUI control at the API request boundary.
- No mutable global/session persistence or replay from export, TOML, sidecars, rehydrate, `ui_custom_params`, workflow Worker or direct CLI.

### Prepared state (extend `mikazuki.training_config.PreparedTrainingConfig`)

- Add `admission: ComponentAdmission | None = None`, the typed first-party host diagnostic.
- Add `admission_intent: RequestAdmissionIntent | None = None`, request-local immutable intent metadata. For Standard/inactive and Preview/Export, it must not confer or carry reusable positive acknowledgement. It is not a field of `prepared.config`.
- Preserve existing positional ABI `(train_type, trainer_file, config, gpu_ids, warnings, sidecars, runtime_blockers)`, no changes to GPU/optimizer/sidecar schema, no execution-identity or Checkpoint ABI changes. Diagnostic/warnings stay independent.

### Request preflight (`prepare_request_config`)

Keep all existing arguments, append keyword-only:
`admission_intent: RequestAdmissionIntent | None = None`;
`admission_purpose: Literal["preview","export","start"] | None = None`.

- Defaults: `admission_purpose="start"` when `launch=True`, otherwise `"preview"`. Reject unknown purpose or mismatches (e.g. `launch=True` with `"export"`, `launch=False` with `"start"`) **before any side effect**.
- Accept only the exact `RequestAdmissionIntent` type (or `None` = default false). If `ack=True` is supplied for a Standard/inactive feature, do not silently turn it into a future task permission; fail closed. Preview/Export **always** evaluate `experimental_ack=False` even if a transient explicit intent is present; the returned `admission_intent` must be neutral/no-positive and cannot be exported/replayed.
- Reject reserved `experimental_full_bf16_ack` injected into raw Trainer `config`, `ui_custom_params`, `__ui_custom_overrides`, or post-override `prepared.config`. This new control is **not a trainer flag**. A2 later decides how to decode an explicit top-level envelope field into a typed Request intent; A1 must not read it from the existing general-purpose `config` dict.
- Only **after** `prepare_training_config` (post-override) and host-owned path checks, call `assess_component_training_admission(policy, train_type=prepared.train_type, effective_config=prepared.config, purpose=admission_purpose, experimental_ack=(launch and purpose=="start" and typed_intent.ack))`. For active Component, store the typed result in `prepared.admission`. For Standard, do not call the adapter, and preserve prior success path.
- The A1 diagnosis is **non-authoritative**: even with a true explicit request intent and `experimental_host_candidate` diagnosis, the unchanged QG-1B3-B Step 6F `experimental_ack=False` must still block pending-certification Start. This intentional temporary disparity is a safety stop until QG-2-A2/2-B validates both WebUI→launcher and direct Trainer transport. Never filter `runtime_blockers` by string, fabricate a `can_start` credential, or skip `validate_prepared_config(prepared, True)`.
- Still fail before `materialize_sidecars` / autosave TOML / staged Anima Trainer / `run_prepared_train`; no Torch, real model file, CUDA, protected Runner or checkpoint I/O in Host preflight.

### Existing AST-test interop

`tests/test_parameter_policy_request_contract.py` executes only the AST of `prepare_request_config` with a mocked global namespace; do not break its old real Step 6F sequencing through an accidental unresolved module-global helper or eagerly evaluated new type annotation. The implementation may use function-local imports or explicitly adjust the AST fixture in a later reviewed GREEN. The A1 tests will dynamically exercise the extracted function without importing the FastAPI application.

## 3. Frozen 12 +4 contract cases

New `tests/test_parameter_policy_qg2a1_request_red.py` has A101–A112 (all intentionally unmet) and A901–A904 (true existing baselines):
- A101: frozen strict intent ABI, known origin, bad pseudo-booleans/mismatched origin rejected.
- A102: prepared fields exist separately from `config`; preserved positional init, no mutable global intent.
- A103: request signature accepts typed *keyword-only* intent/purpose; consistent `launch`/purpose normalization and strict errors.
- A104: actual first-party admission is evaluated from **post-override** effective config, not raw `full_bf16`; no early pre-override authority.
- A105: preview of audited SD1 with backend certificate `pending` is typed `ack_required`, not an asset or certificate proof, still has Step 6F blocker.
- A106: explicit `webui_start` intent may form a Host experimental candidate, but Start still raises before materialization while QG-2-A2/2-B are missing; absent intent similarly fails.
- A107: forged raw, custom TOML and override-map `experimental_full_bf16_ack` rejected before any launch side effect; fake `can_start` cannot become intent.
- A108: genuine capability failures (unaudited SD3 / invalid bf16 / runtime hard errors) cannot become admissible after intent.
- A109: existing D1 source-qualified SD1 still works without new ack; no spurious experimental block.
- A110: Preview/Export ignore a positive intent and scrub it from prepared metadata and portable config/sidecars.
- A111: separate requests cannot inherit intent; no export/replay into next request or checkpoint identity.
- A112: Standard and inactive paths maintain existing behavior, typed intent not used as generic Trainer param, existing Step 6F/asset/Direct guards untouched.
- A901–A904: unchanged real source baselines: old qualified SD1 + pending SD3, strict pseudo-boolean, Step 6F-before-materialize and Standard launch ordering, and existing Request helper ABI.

Every A101–A112 test fails at the *actual absent A1 interface* with marker `QG-2-A1 REQUEST INTENT CONTRACT ABSENT`, not by a fake deliberate `assert False` or skipped test. After the interface is implemented the tests proceed to real source/behavior assertions. Four baselines pass unchanged. No errors, skips, expected failures, or unrelated failures are accepted.

## 4. CI / stop line

A separate Ubuntu + Windows Python 3.11 workflow checks prior scoped QG-1B3-B stable Host cases and **exactly 12 A1 targeted FAIL + 4 baseline PASS** (zero ERROR/SKIP/xfail/xpass); the workflow ends FAILURE on valid Tests-only RED. Freeze run IDs, exact commit and failure marker after real CI. Concurrent inherited B310–B312 remain intentionally RED; no new blanket rejection of existing D1/Standard.

**Next after valid RED:** minimal A1 GREEN (only `training_config.py`, `training_request.py` and its tests), independent review of input/intent trust and no side effects, then QG-2-A2 GUI/API transport with typed Preview/Export/Start and B311, followed by QG-2-B direct Trainer/launcher B312. No actual experimental Full-BF16 user launch until both entrypoints and physical checks are reviewed; no PR merge before full parity and Final Strict.

## 5. Minimal GREEN implementation following the frozen RED

This section describes the implemented **Request Host** scope after the original
dual-OS Tests-only RED. Sections 1–4 above remain the historical frozen
contract; they do not grant direct Trainer or GUI admission.

- In `mikazuki/training_config.py`, `RequestAdmissionIntent` is frozen and
  slotted, uses strictly typed `bool`, and accepts only `default` /
  `webui_start` origins. `True` requires `webui_start`. Two appended
  `PreparedTrainingConfig` fields carry typed host `admission` and one
  request's `admission_intent` outside the Trainer config/sidecars, preserving
  the existing positional dataclass ABI.
- In `mikazuki/training_request.py`, explicit `admission_intent` and
  `admission_purpose` keyword-only inputs are type checked. The purpose is
  constrained to the actual `launch` mode; raw/custom/nested normalized
  copies of `experimental_full_bf16_ack`, `admission_intent`, or `admission`
  are rejected instead of interpreted as credentials.
- `assess_component_training_admission` runs after
  `prepare_training_config` applies the final `ui_custom_params` overlay
  and after the existing host-owned Policy/Multi-Caption path checks. The
  result is stored separately as `prepared.admission`. Preview/Export scrub
  positive intent and never use it in an admission decision or sidecar.
  New tests additionally exercise the **real** GUI compilation path with
  both `full_bf16: true → false` and `false → true` via custom TOML.
- Importantly the QG-1B3-B `parameter_policy_runtime_blockers` still owns
  the Start hard gate and still passes `experimental_ack=False` internally.
  Even `RequestAdmissionIntent(True, "webui_start")` can create at most a
  *host-side* `experimental_host_candidate`; Start remains blocked for an
  uncertified configuration until both A2 API + B Direct Trainer are
  implemented and reviewed. Validation/sidecar/materialization order and
  current D1-qualified cases remain unchanged.
- Existing AST-isolated `tests/test_parameter_policy_request_contract.py`
  helper namespaces were extended only to inject the new named producer,
  typed intent and existing custom-TOML parser. Mocked Step 6F tests retain
  their historical purposes; real-pipeline tests install the real producer.
- The QG-2-A1 workflow now requires A101–A112 12/12 PASS and A901–A904
  4/4 PASS on both OS, with B301–B310 and prior baselines GREEN. The
  cumulative full-contract workflow remains explicitly **RED** for B311
  (typed GUI/Preview/Export/Start) and B312 (Direct Trainer/CLI/launcher),
  rather than treating a scoped Host pass as complete readiness.

**Stop line:** This is only QG-2-A1 scoped Host GREEN. No production edits
to `app/training_api.py`, `parameter_policy_trainer.py`, launcher transport,
source qualification/release tables, CUDA, optimizer ownership or Checkpoint
are authorized by this gate. Draft PR #69 stays open/unmerged. Next is the
independent Strict Review of Request inputs/one-job lifetime before QG-2-A2.
