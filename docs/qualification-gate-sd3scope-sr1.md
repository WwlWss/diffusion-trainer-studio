# SD3 C2-A Scope SR1: old qualification-row fixture versus independent source audit

Existing valid Tests-only RED: Anima Qwen3 review run 44944330251 on B4-I HEAD and same three failures on earlier B4-A1 run 44942841440. Two R1/R2 subtests and one Preview expectation wrongly assumed that patching a legacy ExecutionFeatureQualification row made all modern readiness blockers disappear.

Source chronology: SD3 C2-A tests created 2026-10-28/29; first-party QG1B2 source-audited readiness introduced 2026-11-06, enhanced 2026-11-07; QG2-A1 Request integration was added 2026-11-07. Current _AUDITED_BACKENDS intentionally includes sd-lora and sdxl-lora only. An SD3 legacy release row is independent of actual implementation audit.

Test-only repair: preserve R1/R2 resolution and optimizer validation; expect the single real capability.backend.audit_required blocker in Preview and preserve the blocked typed admission. Preserve Start missing-model error precedence and all side-effect checks. Add an independent test of production validate_prepared_config and train_utils.validate_model to protect the real absent-model check, ordered before dataset inspection. Do not modify capability registry, allowlist, Request Step 6F, CUDA, optimizer, or checkpoint. Full Anima Qwen3 review must pass before closing this SR1.

Do not describe Host-only test expectations as real SD3 Full-BF16 qualification.
## Full Anima review follow-up: legacy frontend Start callsite

On corrected SD3 scope tests, full Anima workflow run 44952097541 found one previously masked old frontend-test assertion in tests/test_frontend_effective_config_patch.py:20. It still expected the pre-QG2-A2 expression __trainingRequest("/api/run",T()) whereas the actual first-party production patch now passes the transient third argument __ack. Repair keeps all real frontend patch code unchanged, asserts __trainingRequest("/api/run",T(),__ack), ensures acknowledgement is serialized only under Start control plane when strict true, resets one-shot state in finally, and forbids GUI-config persistence. Must prove complete Anima workflow real PASS after updating test.

## CPU runtime follow-up (real old ABI RED)

After full Anima Host validate passed on 140bd1cb9b00d49c54496e1dcebcbc24c2e022e1, its CPU-runtime job 44952209689 executed 77 tests and failed with exactly three TypeErrors in old SD3 Direct Trainer tests: create_parameter_policy_session now requires admission_snapshot and admission_intent, added by B3-A/B/C after the historical Oct-2026 C2-A tests. The two legacy test functions previously omitted both, while their future-qualified fake row also ignored first-party source audit. The fix is test-only: build a real DirectTrainerAdmissionIntent and prepare_direct_trainer_admission snapshot, verify unreviewed SD3 is blocked before parameter scanning, retain R1/R2 geometry and missing model execution asset gate, and preserve the original test IDs used by the pinned CPU workflow. No production admission, GPU, optimizer, or Checkpoint changes. Exact-head complete Anima runtime smoke must pass before declaring full correction.
