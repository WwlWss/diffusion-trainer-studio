# QG-2-MR-R2-2-SR1-B — Minimal GREEN for Historical GPU Admission Harness

Parent RED HEAD: cda138fe435d56bf021077b0fa7815e0e9b8d292.
Source issue: QG2-R2-02, historical GPU evidence harness and baseline GPU device probe used the pre-B3-C create_parameter_policy_session(args, train_type, roots) ABI.
PR #69 must remain Draft/open/unmerged pending an independent strict review and actual Windows CUDA rerun.

## Source migration and boundaries

### Fix A / R206: isolated tool-only GPU constructor
New tools/_parameter_policy_gpu_low_level_fixture.py follows the already-reviewed CPU-only low-level construction pattern without importing its CPU-only module. The new function build_gpu_evidence_full_bf16_session builds the actual production ParameterPolicyTrainerSession, descriptor scan, root reference snapshot, routing, runtime spec, CompositeOptimizer and requires-grad guards. It never invokes or alters the production create_parameter_policy_session() entry.

It accepts only the known _TinyFlux, _TinyMuonFlux and _TinyMuonAdamWFallbackFlux geometry (specific named parameter shapes), transformer root, source-owned AdamW/Muon/fallback profile topology, exact learning-rate routing, true Python full_bf16=True, mixed_precision='bf16', and train_type='flux-finetune'. It rejects non-scaffold variants before root scan and never authorizes a real model. The source full-BF16 Flux Host readiness still blocks real production admission.

### Fix B / R201-R204: four synthetic GPU sites
tools/run_parameter_policy_execution_gpu_matrix.py changes exactly the four Full-BF16 synthetic session-building sites to this isolated constructor: infra session probe, AdamW train-save, Muon train-save and Muon-plus-explicit-AdamW-fallback train-save. The same runtime builders are reused by fresh-process resume. Their existing optimizer, scheduler, tensor-dtype, GPU, save/resume, evidence and checkpoint assertions are otherwise retained.

Keep the local temporary_execution_qualification() evidence lease/restoration mechanism as historical GPU harness metadata only; it does not grant producer-host authorization or alter persistent source qualification rows.

### Fix C / R205: genuine inactive Admission for baseline device audit
tools/run_parameter_policy_gpu_matrix.py uses DirectTrainerAdmissionIntent(False) and real prepare_direct_trainer_admission() for its non-Full-BF16 synthetic _TinyFlux device audit, requires admission.status == 'inactive', and passes both typed values into the unchanged production Late Session.

### Fix D / R207: truthful evidence scope
The execution GPU worker phases and coordinator case rows disclose session_construction_scope='isolated_gpu_low_level_evidence_only' for the seven Full-BF16 synthetic cases; the pure CUDA capability probe is excluded. This is a test-construction-scope label, NOT a qualification grant. Existing backend_qualification_eligible and optimizer_qualification_eligible remain false for those evidence cases. Case identities, phases, bundle schemas and promotion/regression summaries remain unchanged.

## Regression obligations

1. Source-only inherited SR1-A suite: 13 tests, 13 PASS (R201-R207 and R901-R906); prior frozen RED baseline was 7 expected failures, 6 passes on cda138fe.
2. New 7-method CPU dynamic tests in tests/test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.py: AdamW/Muon/C4 real low-level Sessions, non-scaffold rejection, inactive Full-BF16 rejection, real production Flux full-BF16 fail-closed pre-root, and real non-Full-BF16 inactive Admission execution.
3. Existing tests/test_parameter_policy_execution_gpu_harness.py plus historical CPU runtime smoke tests remain unchanged and must pass. No unauthorized xfail/skip.
4. All checks should use clean exact HEAD and the independent Python3.11 R2 venv. Evidence output stays outside repo.
5. Actual 8-case/14-phase execution CUDA matrix and baseline GPU matrix are distinct later steps. Do not label static/CPU GREEN as GPU qualification.

## Explicit STOP

No changes in mikazuki/ production Admission/Readiness/Trainer, Optimizer production, genuine model backend, checkpoint, scripts/stable, scripts/dev, launcher/API, PR merge eligibility or Full-BF16 release matrix. Do not overwrite the original D-drive execution-gpu-regression.json failure evidence. Reassess any unexpected CPU test or CUDA failure before rerunning a GPU matrix or making further source changes.

After the new exact HEAD is pushed and the Windows user verifies the Host/CPU suites, perform an independent source review and then issue a separate real-CUDA evidence command writing to a new file.
