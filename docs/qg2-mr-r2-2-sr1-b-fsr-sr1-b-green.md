# QG-2-MR-R2-2-SR1-B-FSR-SR1-B: Minimal GREEN, scoped evidence hardening

Tests-only RED parent: a0611682143405b79cc90f3950b0141d0d58db51.
PR #69 remains Draft / unmerged; no production admission or real backend qualification changes.

## Fix SR1-01: file ownership and fail-closed pre-import checks

Modify only tests/test_parameter_policy_execution_gpu_harness.py in the existing pre-import untracked-workspace test. Use an exclusively created random phase-c-provenance-untracked-<uuid>.json probe and a test-owned TemporaryDirectory(prefix=..., dir=ROOT.parent) for execution-probe.json and baseline-probe.json. Never unlink fixed ROOT.parent output files. Preserve exact Git status preflight, two real Runner strict rejection checks, pre-import error text, no-produced-output assertion and 25-second child timeout. Clean only paths this test created, including on error.

The new SR101-T4 dynamically mocks both child Runner calls and Git queries within a temporary fake repository with preexisting external evidence JSON sentinels. Existing contents must survive byte-for-byte.

## Fix SR1-02: synthetic-only Full-BF16 profile and model provenance

Modify only tools/_parameter_policy_gpu_low_level_fixture.py. After checking exact train_type, full_bf16/mixed_precision, roots, model kind, parameter names/shapes, verify inspect.getsourcefile(type(model)) resolves to one of these two reviewed source paths:
- tools/run_parameter_policy_execution_gpu_matrix.py
- tests/test_parameter_policy_qg2_mr_r2_2_sr1_gpu_helper.py

Compare the canonical policy's entire optimizer profile objects, including complete argument dictionaries. Match exact AdamW-empty-args, pinned Muon reference argument values (momentum=.95, weight_decay=0, weight_decouple=true, nesterov=true, ns_steps=5, ns_coeffs=original, use_adjusted_lr=false), and C4 explicit AdamW fallback empty args. Preserve original Component routes and exact learning rates. Reject all invalid inputs before scan_parameter_roots. The source allowlist is only a test-scope safeguard, not a cryptographic defense against someone who can rewrite repository code.

Do not move the synthetic model classes, replace real low-level Session/routing/optimizer implementations or edit the production create_parameter_policy_session() interface. Keep Host readiness for flux-finetune Full-BF16 explicitly BLOCKED; no synthetic GPU result is production backend qualification.

## Verification gate

1. Tests-only RED FSR suite: 11/11 GREEN (ten new failure guards, one preserved guard).
2. Previously verified RED->GREEN source suite: 13/13.
3. CPU helper runtime suite: 7/7 (including true production flux-finetune admission block).
4. Existing GPU Harness suite: 25/25 (must not accidentally start the GPU matrix).
5. Original host-side Parameter Policy tests and training runtime smoke as separate review where feasible.
6. Exact clean HEAD before and after. Evidence outside repository, no existing files overwritten. Preserve original failing CUDA JSON.
7. After Independent Final Strict Code Review, run a new 8-case / 14-phase CUDA regression and distinct baseline matrix in isolated evidence directories. Do not imply CUDA GREEN based on CPU tests.

No persistent qualification rows, checkpoint logic, real Trainer, production Admission/Readiness or user model backend may be modified by this fix.
