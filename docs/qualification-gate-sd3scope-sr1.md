# SD3 C2-A Scope SR1: old qualification-row fixture versus independent source audit

Existing valid Tests-only RED: Anima Qwen3 review run 44944330251 on B4-I HEAD and same three failures on earlier B4-A1 run 44942841440. Two R1/R2 subtests and one Preview expectation wrongly assumed that patching a legacy ExecutionFeatureQualification row made all modern readiness blockers disappear.

Source chronology: SD3 C2-A tests created 2026-10-28/29; first-party QG1B2 source-audited readiness introduced 2026-11-06, enhanced 2026-11-07; QG2-A1 Request integration was added 2026-11-07. Current _AUDITED_BACKENDS intentionally includes sd-lora and sdxl-lora only. An SD3 legacy release row is independent of actual implementation audit.

Test-only repair: preserve R1/R2 resolution and optimizer validation; expect the single real capability.backend.audit_required blocker in Preview and preserve the blocked typed admission. Preserve Start missing-model error precedence and all side-effect checks. Add an independent test of production validate_prepared_config and train_utils.validate_model to protect the real absent-model check, ordered before dataset inspection. Do not modify capability registry, allowlist, Request Step 6F, CUDA, optimizer, or checkpoint. Full Anima Qwen3 review must pass before closing this SR1.

Do not describe Host-only test expectations as real SD3 Full-BF16 qualification.