# QG-2-MR-FSR-PR69-SR2-E2-C2 — R2 Actual CPU Runtime GREEN

Frozen C1 parent: fa0e3d84fcd53503d0cbac14a80453bd548e6b2a
Tree: 29563f3843fd2bfb7f71a032724fe55d485f8dab
C1 run: 46000078202. Both OS: 917 Host identities, 847 PASS + 70 historical skips, 6/6 frontend, exact identity/skip parity, 1 target GREEN + 35 retained RED, 8/8 independent and nested 95/95 baselines, 10/10 C1 behavior.

## Scope

Fix B. Extend the C1 test-only runner with pinned CPU execution. Preserve Host selection, original E2-B contract, C1 contract, both R2 test modules, exact source allowlist, production, GPU runners, Gitlinks and qualification registry. Modify Anima PR paths only; no Anima main push until C3. The new E2 workflow monitors main and both developer branches for its scoped inputs. No other existing workflow changed.

## Real runtime

Python 3.11; torch 2.7.0 CPU wheel from official CPU index; accelerate 1.6.0; numpy 1.26.4; schedulefree 1.4; pytorch-optimizer 3.10.0; safetensors 0.4.5. Both Ubuntu/Windows. The test-only runner checks installed versions, torch module version, CUDA/HIP build absence, CUDA availability false, empty CUDA_VISIBLE_DEVICES and an actual CPU tensor. Each actual job invokes the original seven Helper and eleven FSR test identities in separate processes. Use the original top-level unittest discovery identity, verify exact module files and no duplicate package aliases, record actual startTest identities; fail on missing, duplicate, count mismatch, failure, error, skip, xfail or xpass. No fake CUDA, no new backend qualification.

Full parity uses independent full wildcard legacy discovery in the same pinned CPU environment and fresh Host and CPU diagnostic subprocesses. Require no loader errors, exact union, no overlap/duplicate, and original ordered Host IDs. Discovery parity is not a substitute for executing 18 tests or existing CPU smoke.

## Triggers

Anima already covered original test file names through tests/test_parameter_policy*.py; explicit entries make the frozen E2 input contract auditable. Add missing exact new runner/contract/workflow and GPU fixture/runtime paths; add relevant transitive inputs such as training_gui_args, sd3_full_bf16_scope and sd3_architecture_contract, plus prior helper/smoke and evidence harness dependencies. The E2 workflow also watches the three Gitlink paths and .gitmodules because full Host qualification requires them. Preserve other existing Anima steps and events.

## Stage

Expected C2 scope: E2H101-104 and H106 PASS, H105 and all remaining 30 E2 lifecycle/SR2 targets RED (31 RED total). Eight independent baselines and nested 95 real cases; C1 10/10; C2 6/6 new behavior tests. Retained RED is exit 1, invalid outcomes exit 2. Full Host parity job retained on both OS. CPU job must pass the actual 7+11 and full CPU discovery partition parity. Expected results in this file are not observations. Full integration/stack merge remains HOLD until C3/C4, independent review and E3. Do not rerun CUDA or promote qualification in C2.
