# QG-2-B4-A1 — API -> Launcher one-invocation Intent Handoff: source contract and Tests-only RED

**Frozen parent:** `279a14a2ad8769001fe44ee6e70579a9c18f524c` (B4-L2 scoped dual-OS GREEN).  
**Branch/PR:** `feature/qualification-gate-qg1b3-qg2-admission-red` / Draft #69.  
**Scope:** freeze and execute the *actual checked-in* `/run` handler's API -> Launcher handoff; no production changes, no GPU or Full-BF16 admission.

## 1. Verified source graph and sole missing seam

- `mikazuki/app/training_api.py:65-122`: `_decode_training_admission_request()` accepts strict top-level bool only in nested Start envelope, stamps `RequestAdmissionIntent(True,"webui_start")`, neutralizes Preview and Export.
- `mikazuki/training_request.py:291-322`: `prepare_request_config()` initializes a fresh neutral `prepared.admission_intent`, promotes only a first-party `experimental_host_candidate` with explicit Start acknowledgement; the independent Step 6F and GPU blockers retain candidate HOLD.
- `mikazuki/training_config.py:40-80`: exact immutable Request intent class plus `PreparedTrainingConfig.admission_intent`, separately owned from `config`, sidecars and Checkpoint.
- `mikazuki/training_launcher.py:21-85`: B4-L2 accepts `admission_intent` keyword-only, rejects forged intent before GPU/task effects, appends `--experimental_full_bf16_ack` only to child Trainer argv for an exact positive first-party WebUI intent.
- **Actual gap:** `mikazuki/app/training_api.py:create_toml_file():366-373` calls `run_prepared_train(..., page_train_type=..., run_id=...)` but does **not** pass `admission_intent=prepared.admission_intent`. Cumulative B312 already reports precisely `API must pass PreparedTrainingConfig intent to Launcher`.

## 2. Frozen future Minimal GREEN contract (not implemented here)

After the existing `prepare_request_config`, read-only Component asset validation, trainer staging, Anima materialization, sidecar/TOML writes and thread calculation, `/run` MUST pass the **same** `prepared.admission_intent` object to `run_prepared_train` as a keyword argument. A neutral Standard or non-acknowledged Start passes the prepared neutral object; a positive raw request that preparation has downgraded must remain neutral. Do not use raw `admission_intent` / `experimental_full_bf16_ack` from JSON, fabricate an intent, persist it to TOML/sidecars/Task metadata/environment/Checkpoint, alter `run_id`, or move any existing Sidecar/Anima/asset ordering. Leave actual Request, Step 6F, early/final/late Direct Trainer, dtype/optimizer/Checkpoint and CUDA guards unchanged.

The later Minimal GREEN should be a narrow `training_api.py` call-site change only, after independently proving this RED. **A Host-side synthetic successful `prepare_request_config` result in this test is a transport fixture, NOT permission for any real `experimental_host_candidate` to launch.** B4-A1 cannot remove Step 6F HOLD or establish SD3 Full-BF16 GPU qualification.

## 3. Eight genuine target RED identities

`tests/test_parameter_policy_qg2b4a1_api_handoff_red.py` compiles the **actual** checked-in async `create_toml_file` AST and real API decoder AST with inert preparation, staging, serialization and launcher collaborators. It also runs the **actual** Launcher AST from B4-L2 with fake GPU/task/async; no FastAPI process, CUDA, subprocess, Torch, filesystem writes or production launch.

- **B4A101:** one real `/run` Launcher call has exact keyword expression `prepared.admission_intent`, never raw Request.
- **B4A102:** positive first-party prepared Intent forwarded by object identity to Launcher.
- **B4A103:** absent acknowledgement forwards fresh, exact typed neutral Intent explicitly.
- **B4A104:** positive raw acknowledgement *downgraded* by Request preparation forwards only prepared neutral; no raw request escalation.
- **B4A105:** Standard launch forwards typed neutral without changing existing staging order or threads.
- **B4A106:** consecutive positive then neutral Starts are separate objects and never replay.
- **B4A107:** positive Intent crosses only the function argument; existing asset-first/materialization ordering, 8-thread threshold and serialized TOML/sidecars stay unchanged.
- **B4A108:** actual API AST -> actual Launcher AST -> fake task argv sends one child `--experimental_full_bf16_ack` and no flag on the next neutral Start; environment/task metadata remain clean.

Exactly eight target methods must fail because the checked-in API omits the actual keyword, with marker `QG-2-B4-A1 API HANDOFF CONTRACT ABSENT`. There are no fabricated unconditional failures or xfails.

## 4. Six real pre-existing PASS baselines

- **B4A901:** actual dedicated API decoder strict bool, Preview/Export neutralization.
- **B4A902:** Component asset validation before staging and sidecar/TOML/launch.
- **B4A903:** Standard preserved staging-before-validation order and no Component hooks.
- **B4A904:** forged controls rejected before any observable launch/write/prepare.
- **B4A905:** rejected Component asset check prevents staging and writes.
- **B4A906:** actual B4-L2 Launcher already transports positive and neutral typed Intent correctly when invoked directly.

## 5. Exact dual-OS CI, stop rule

New `.github/workflows/qg2b4a1-api-handoff-red.yml`: Ubuntu/Windows Python 3.11; source compilation; inherited 116 Host tests PASS; B4-0 six baselines PASS; B4-L2 8 target plus 6 Standard baselines PASS; cumulative B301-B312 / B901-B904 = 15 PASS + **exactly one B312 FAIL with sole API handoff reason**; new B4A901-906 **6/6 PASS**; new B4A101-108 **exactly 8 real FAIL** with the exact marker, no ERROR/SKIP/xfail/xpass. This specific new workflow intentionally exits 1 after validating genuine RED. All existing historical expected-RED workflows remain untouched.

**Stop after valid RED:** keep PR #69 Draft/unmerged, do not modify `training_api.py` or `training_launcher.py`, do not update B312 or inherited workflow expectations, do not launch Full-BF16/CUDA. Next separately authorized phase is **QG-2-B4-A1 API Minimal GREEN**, then B4-I integrated 16/16 and coordinated inherited workflows.
