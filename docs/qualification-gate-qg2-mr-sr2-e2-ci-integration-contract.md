# QG-2-MR-FSR-PR69-SR2-E2-B — CI Integration Tests-only RED

## Frozen scope

Parent: 141a0d187fc32085dccf7b20147258c84ffe8fea
Tree: 433c997447c685074a7e0b93bb86cfc74c400038

Main 1dc4f56fb862aade574add48c4f1bea2c54f3e96; PR67 f6b0b757f58546922a2703608b662a3810476eff; PR68 eb4749d8832cf64a55755cb2261104baad2a1efb; PR69 806bd4dda0926677ad40df8413ede336f25249b3. Do not move those refs in E2-B. Only add this document, tests/qg2mr_sr2_e2_contract.py, and .github/workflows/qg2-mr-sr2-e2-ci-integration.yml. No production, previous tests, previous workflows, GPU evidence tools, or submodule changes.

## Expected RED identities

All 36 targets must fail with the contract marker, with eight independent baselines GREEN on each Ubuntu/Windows Python 3.11 no-Torch job. An accepted RED must exit 1; unexpected shapes exit 2. No ERROR/SKIP/xfail/xpass. This document states expectations, not observed results.

| ID | Exact target obligation |
|---|---|
| E2H101 | Run real legacy unittest discovery in a subprocess without Torch. Exactly two real _FailedTest imports from R2 Helper and FSR; no unrelated import errors. Compare future reviewed partition to the actual non-error reference IDs. |
| E2H102 | Executable pinned CPU Helper command, R2G01–07. |
| E2H103 | Executable pinned CPU FSR command, SR101 T1–T4/B1 and SR102 T1–T6. |
| E2H104 | Anima PR paths cover four R2 files and new runner/contract/workflow. |
| E2H105 | New merge-contract requirement: Anima main push covers PR inputs. |
| E2H106 | New long-lived dual-OS pinned R2 CPU job, dependency paths and main/PR triggers. |
| E2L101 | qg1b3-qg2-admission-host.yml: main push while preserving developer branch and dispatch. |
| E2L102 | qg1b3a-admission-adapter-host.yml: main push while preserving developer branch and dispatch. |
| E2L103 | qg1b3b-fsr-runner.yml: main push while preserving developer branch and dispatch. |
| E2L104 | qg1b3b-typed-migration-host.yml: main push while preserving developer branch and dispatch. |
| E2L105 | qg2a1-request-intent-host.yml: main push while preserving developer branch and dispatch. |
| E2L106 | qg2a1-sr1-host.yml: main push while preserving developer branch and dispatch. |
| E2L107 | qg2a2-api-host.yml: main push while preserving developer branch and dispatch. |
| E2L108 | qg2b2-cli-intent-host.yml: main push while preserving developer branch and dispatch. |
| E2L109 | qg2b3a-policy-snapshot-host.yml: main push while preserving developer branch and dispatch. |
| E2L110 | qg2b3b-early-bridge-host.yml: main push while preserving developer branch and dispatch. |
| E2L111 | qg2b3b-sr1-dynamic-host.yml: main push while preserving developer branch and dispatch. |
| E2L112 | qg2b3c-late-session-host.yml: main push while preserving developer branch and dispatch. |
| E2L113 | qg2b3c-sr1-cpu-runtime.yml: main push while preserving developer branch and dispatch. |
| E2L114 | qg2b4-0-b312-contract-red.yml: main push while preserving developer branch and dispatch. |
| E2L115 | qg2b4a1-api-handoff-red.yml: main push while preserving developer branch and dispatch. |
| E2L116 | qg2b4fsr-sr1-ci.yml: main push while preserving developer branch and dispatch. |
| E2L117 | qg2b4l1-launcher-transport-red.yml: main push while preserving developer branch and dispatch. |
| E2L118 | qg2-mr-fsr-pr69-sr2b-ci-deps-red.yml: main push while preserving developer branch and dispatch. |
| E2L119 | Dynamic Host has PR entry. |
| E2L120 | qg1b3a-admission-adapter-host.yml: PR covers frozen push-only paths in PARITY. |
| E2L121 | qg2b3b-early-bridge-host.yml: PR covers frozen push-only paths in PARITY. |
| E2L122 | qg2b3c-late-session-host.yml: PR covers frozen push-only paths in PARITY. |
| E2L123 | qg2b3c-sr1-cpu-runtime.yml: PR covers frozen push-only paths in PARITY. |
| E2L124 | qg2b4a1-api-handoff-red.yml: PR covers frozen push-only paths in PARITY. |
| E2L125 | qg2b4l1-launcher-transport-red.yml: PR covers frozen push-only paths in PARITY. |
| E2L126 | Dynamic Host actual imported/read source dependencies in both events. |
| E2L127 | FSR runner actual typed migration input coverage. |
| E2L128 | B4-A1 actual explicit inherited module coverage, parsed from Python run blocks. |
| E2S101 | Permanent strict SR2 6+5 GREEN instead of development RED/CANDIDATE branch. |
| E2S102 | SR2 watches both producer workflows, producer/import inputs, test, contract and self. |

The complete callable names and source paths are frozen in E2Targets, TARGET_PREFIXES, PARITY, and the explicit source input sets. E2H105 and E2S101 are deliberate lifecycle/qualification requirements, not assertions that historical RED exit 1 was incorrectly green. PR filters with no paths cover all paths; known explicit directory/** filters also cover descendants. Other unrecognized syntax fails closed. Source-contract checks of job wiring do not by themselves prove runtime results.

## Independent baselines

- E2B901: actual frozen 67-file census (all original files retained), two CPU modules, exact model-source path still present.
- E2B902: original R2 source suite 13/13, actual execution.
- E2B903: historical GPU harness 25/25, actual execution, no CUDA.
- E2B904: SR2-C original dependency suite 11/11.
- E2B905: cumulative B312 16/16.
- E2B906: B4-I thirteen inherited workflow contracts.
- E2B907: B4-FSR 15+2.
- E2B908: preserve three unfiltered PR contracts, pinned Anima/B3-C CPU jobs and manual-only GPU workflow.

Each nested suite must actually execute the stated count with no extra outcomes. The 8 outer baseline tests are not a substitute for nested outcomes.

## GREEN interface to implement only after accepted RED

New tests/qg2mr_sr2_e2_runner.py shall partition before import, using the existing unittest discovery module identity convention, exactly excluding the two R2 CPU files from Pure Host discovery. Anima validate step named Parameter Policy host contract must run python -m tests.qg2mr_sr2_e2_runner host. Its diagnostic form host --discover-only must emit one line prefixed DTS_E2_HOST_DISCOVERY_JSON followed by an object containing ids, errors, loaded_cpu; actual IDs must equal independent legacy discovery minus exactly the two known _FailedTest entries. No errors or CPU-only modules in the Host partition. Do not change the two R2 test modules to skip or rename them; do not change the GPU evidence model source allowlist. Existing runtime-dependency skips are not silently counted as execution.

E2 workflow will add job r2-cpu with two unconditional steps: python -m tests.qg2mr_sr2_e2_runner r2-cpu --suite helper and ... --suite fsr. The default r2-cpu mode may run both. Execute exactly 7 and 11 original test methods on both OS with pinned CPU dependencies, actual result identity/count, no skipped/expected/unexpected extra outcomes, no CUDA. Pinned dependencies as E2-A: torch 2.7.0 CPU, accelerate 1.6.0, numpy 1.26.4, schedulefree 1.4, pytorch-optimizer 3.10.0, safetensors 0.4.5. Validate real CPU availability, strict-result mutations, module identity, complete partition ID equality and trigger closure in E2-C/D. A passing source wiring check alone is not the CPU proof.

Preserve all 24 original QG workflows, original stable check IDs and historical execution steps; add required main/PR coverage. Do not alter original GPU manual workflow. E2-B executes only Host contracts. It does not prove CUDA, new backend qualification, or merge authorization. Change new dedicated workflow to green mode only after all corrections; retain historical RED SHA and real logs.

## Expected state

E2-B expected valid RED, not accepted until raw logs on both OS confirm exact IDs, two actual no-Torch discovery failures, 36 target FAIL and 8 actual baseline PASS including nested counts. Any unexpected error, partial RED, baseline failure, or absent run is HOLD. No production or existing workflow change during RED.
