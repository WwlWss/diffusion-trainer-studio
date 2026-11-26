# QG-2-A1-SR1: Strict Review repairs (staged, fail-closed)

Frozen parent: PR #69 Draft at 88b3cb255af1dc302621ed30e9b3bbb1bfa0065f. QG-2-A1 scoped Host GREEN is a prior fact; A2 GUI and B Direct Trainer remain intentionally unimplemented (B311/B312 RED). Do not merge, authorize experimental Full-BF16 Start, alter CUDA/Checkpoint or promote formal release evidence.

## Proven source seams / original rationale

- training_request.py prepare_request_config, lines 183–324: QG-2-A1 introduced a typed Host-only classification on the final effective config. The existing Step 6F parameter_policy_runtime_blockers(... experimental_ack=False) was intentionally kept to avoid unilateral GUI permission. This left blocked, ack_required and experimental_host_candidate states dependent on the old gate: independently reject them on Start even if a future change mistakenly returns an empty legacy blocker set.
- ComponentAdmission in parameter_policy_admission.py is a first-party immutable diagnostic, not a signed credential. It must be type checked; an absent/invalid result cannot be silently regarded as eligible. Preview/Export are permitted to report blocked diagnoses without failing the model-free request, but malformed admission must fail closed.
- training_request.py accepts separate launch and materialize flags for old Standard and Export workflows. QG-2-A1 added purpose preview/export/start but did not bind preview to no-materialization. Forbid preview + materialize, forbid non-bool materialize; retain export and start write support.
- A111 currently covers consecutive neutral Preview only. Add Start A (explicit ack) -> Start B (no ack), failed Start -> new Start and export/rehydrate intent non-replay tests. No global or serialized acknowledgement.
- prepared.admission.host_admissible means implementation/Host candidate only; it intentionally precedes GPU selection and other runtime checks. Never derive public runtime_ready or execution authorization from it. Aggregate actual runtime_blockers and retain model/assets/Trainer/GPU checks as separately required.
- training_config.py uses TYPE_CHECKING-only ComponentAdmission to avoid a cold import cycle. get_type_hints(PreparedTrainingConfig) cannot resolve it. Prefer a lazy module attribute resolver or low-dependency type split after proving import order; do not introduce an eager circular import.

## Strict tests-only RED freeze before any production edit

Create tests/test_parameter_policy_qg2a1_sr1.py with eight genuine absent behaviors, SR101–SR108:
- SR101 / SR102: explicit and absent ack with a first-party typed blocked status must reject even with mocked Step6F=[]; no sidecar write.
- SR103: ack_required must independently reject when Step6F=[].
- SR104: experimental_host_candidate remains QG-2-A2/B HOLD, even when Step6F=[].
- SR105: malformed None / wrong-type first-party admission must fail closed, not become 'allow'.
- SR106: preview with materialize=True rejects before any work or write.
- SR107: materialize='false' or 1 is strictly invalid, never truthy-write.
- SR108: get_type_hints(PreparedTrainingConfig)['admission'] resolves to the actual first-party ComponentAdmission optional type.

Four real previously passing baselines SR901–SR904: existing D1 legacy_approved Start; inactive Component Start; preview returns blocked diagnostic without side effects; export and Standard Start deliberate materialization remain allowed. All targeted failures MUST share exact marker QG-2-A1-SR1 REPAIR CONTRACT ABSENT and be real missing source behavior, not dummy asserts/skips. Ubuntu and Windows both need exact 8 FAIL +4 PASS with no ERROR/SKIP/xfail/xpass, CI visibly FAILURE when valid RED.

## Fix A → Fix B → Fix C → Fix D

Fix A (training_request.py only, tests as necessary): after first-party typed admission, validate the actual ComponentAdmission type; preserve Step6F and GPU check call order. Include typed runtime_blockers in the comprehensive diagnosis without relying on free-text filters; independently reject blocked, ack_required and *stage-held* experimental_host_candidate on Start, regardless of legacy blocker list. Preserve inactive / legacy_approved and Standard behavior. Update AST test fixtures to construct true typed inactive ComponentAdmission instead of None, not a production compatibility escape hatch.

Fix B (training_request.py): check type(materialize) is bool before configuration side effects; reject purpose='preview' and materialize=True. Keep purpose='export', launch=False, materialize=True supported, and purpose='start', launch=True, materialize=True supported for qualified configurations. Existing HTTP Export writes after prepare_request_config(...launch=False) and will be made explicit purpose='export' in QG-2-A2; no premature API edit.

Fix C (tests/docs, minimally Request if a genuine regression is found): add new independent lifecycle and GPU consistency tests; no global authorization cache, no exported acknowledgement, preserve runtime_blockers-based public readiness. Do not redefine ComponentAdmission.host_admissible or serialize intent.

Fix D (training_config.py narrowly): make get_type_hints resolve the actual immutable ComponentAdmission class without introducing eager circular imports. Verify fresh-process cold import in both orders: training_config before parameter_policy_admission and vice versa; get_type_hints and existing Step6F API-import behavior on Ubuntu/Windows.

## Acceptance / hold

- SR101–SR108 8/8 PASS, SR901–SR904 4/4 PASS plus Fix C lifecycle tests all PASS on Ubuntu/Windows, no unexpected failures or skips.
- QG2-A1 A101–A112 12/12 PASS, A901–A904 4/4 PASS; QG1A/QG1B/QG1B3-A/B remain GREEN; B301–B310 GREEN; B311/B312 exactly two contract RED.
- Anima umbrella workflow may remain FAILURE for B311/B312 with optional-dependency skips; do not mislabel as full GREEN.
- Only allowed production source changes: mikazuki/training_request.py, mikazuki/training_config.py. Never weaken the original Step6F, asset, real-GPU, dtype, root/optimizer or checkpoint checks.
- A fresh independent Final Strict Review and a separate QG-2-A2 Tests-only RED are required before any GUI/direct Trainer admission implementation.
