# QG-2-MR-R2-2-SR1-A - historical GPU harness Admission migration (Tests-only RED)

Frozen parent: a317f39b33f8ff5938132a2c0b9c90a3ad67fb8c.
PR #69 stays Draft, open and unmerged. No production code, existing evidence tooling, CI workflows, GPU training, qualification rows or release metadata may change in this RED.

## Root cause (from actual R2-2 Windows logs)

The original D-drive execution-gpu-regression.json contains 1/8 PASS and 7/8 FAIL. Every failure in the first actual phase is TypeError: create_parameter_policy_session() missing 2 required keyword-only arguments: admission_snapshot and admission_intent. Six optimizer cases never reached the first real step or fresh-process resume; no optimizer defect has yet been established.

GPU execution harness old ABI callsites (frozen parent):
- tools/run_parameter_policy_execution_gpu_matrix.py:2642 (_case_full_bf16_session)
- tools/run_parameter_policy_execution_gpu_matrix.py:2388 (_build_adamw_runtime)
- tools/run_parameter_policy_execution_gpu_matrix.py:1627 (_build_muon_runtime)
- tools/run_parameter_policy_execution_gpu_matrix.py:2107 (_build_c4_runtime)
- tools/run_parameter_policy_gpu_matrix.py:343 (_accelerator_device_audit, not invoked by this run)

The safety contract is intentionally required: production create_parameter_policy_session checks strict first-party admission snapshot identity and typed one-shot intent before touching roots. The Full-BF16 host audit intentionally excludes flux-finetune (only sd-lora and sdxl-lora are audited), so merely supplying two parameters to the four Flux synthetic calls cannot be a valid repair. The old CPU low-level fixture documents CPU-only scope; do not broaden it without review.

## Exactly seven targeted RED obligations

R201: infra synthetic Full-BF16 session must use isolated reviewed GPU evidence construction, not production Late Admission.
R202: AdamW synthetic Full-BF16 runtime must do the same.
R203: Muon synthetic Full-BF16 runtime must do the same.
R204: Muon plus explicit AdamW fallback runtime must do the same.
R205: ordinary GPU device audit must use real first-party prepare_direct_trainer_admission, typed DirectTrainerAdmissionIntent(False) and both required arguments at the production Late Session.
R206: new tools/_parameter_policy_gpu_low_level_fixture.py must enforce the narrowly scoped synthetic Flux Full-BF16 context and construct actual low-level descriptor/routing/optimizer/Session invariants. No fake status, no production skip_admission.
R207: the GPU evidence must disclose session_construction_scope=isolated_gpu_low_level_evidence_only and keep backend/optimizer eligibility false at case level.

All must fail on frozen parent with marker: QG-2-MR-R2-2-SR1 GPU HARNESS MIGRATION ABSENT.

## Exactly six existing safety baselines

R901: mandatory typed Late Session arguments and independent identity/fail-closed checks.
R902: Full-BF16 flux-finetune remains blocked at real Host readiness and Late Session.
R903: CPU-only fixture retains real low-level Parameter Policy invariants.
R904: existing CPU test proves true Flux Full-BF16 rejection before parameter root scanning.
R905: exact clean-head bootstrap, output outside repo, external worker files, fresh subprocess and non-promoting evidence are present.
R906: eight GPU case IDs, fourteen worker phases, schema version 3 are unchanged.

## RED acceptance criteria and STOP

Only tests/test_parameter_policy_qg2_mr_r2_2_sr1_red.py and this document are added. Running this stdlib-only source contract suite against the RED commit must yield:
- 13 tests total;
- exactly 7 FAIL (R201-R207), 6 PASS (R901-R906);
- zero ERROR, SKIP, xfail, xpass;
- nonzero exit code only from the seven genuine missing obligations.

Run with the isolated R2 Python 3.11 environment (no CUDA is launched):
    python -m unittest discover -s tests -p test_parameter_policy_qg2_mr_r2_2_sr1_red.py -v

Keep all run output outside the repository. Preserve the original D-drive execution-gpu-regression.json as an immutable failure record. Do not rerun the CUDA matrix before Minimal GREEN.

Next phase: R2-2-SR1-B Minimal GREEN fixes only tooling and test-only isolation, not production admission, backend qualification, optimizer source, actual trainer or checkpoint implementation. New-head review and fresh GPU regression follow separately.
