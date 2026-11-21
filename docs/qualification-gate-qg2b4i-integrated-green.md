# QG-2-B4-I: B312 Integrated GREEN and 13-workflow coordinated CI migration

**Parent validated B4-A1 API handoff:** `130d13ec4f8bfcdcb4d5c50d088df38f0fca3f9f`, dual-OS B4-A1 8/8 PASS, baselines 6/6 PASS, inherited 136/136 PASS, cumulative B312 16/16 PASS. PR #69 stays Draft. No real CUDA qualification is conferred.

## 1. Exhaustive workflow inventory

B312 RED was historically asserted in **13** workflow source files; 11 older QG-1B3/QG-2 workflow consumers plus B4-0 and B4-L2. A migration that updates only 11 leaves two failing despite B312 truly passing.

1. `.github/workflows/qg1b3-qg2-admission-host.yml`
2. `.github/workflows/qg1b3a-admission-adapter-host.yml`
3. `.github/workflows/qg1b3b-typed-migration-host.yml`
4. `.github/workflows/qg2a1-request-intent-host.yml`
5. `.github/workflows/qg2a1-sr1-host.yml`
6. `.github/workflows/qg2a2-api-host.yml`
7. `.github/workflows/qg2b2-cli-intent-host.yml`
8. `.github/workflows/qg2b3a-policy-snapshot-host.yml`
9. `.github/workflows/qg2b3b-early-bridge-host.yml`
10. `.github/workflows/qg2b3c-late-session-host.yml`
11. `.github/workflows/qg2b3c-sr1-cpu-runtime.yml`
12. `.github/workflows/qg2b4-0-b312-contract-red.yml`
13. `.github/workflows/qg2b4l1-launcher-transport-red.yml`

Other QG-1A/1B and isolated B3-B-SR1 workflows do not assert B312 and must not be modified.

## 2. Strict scope and test-first plan

- Tests-only RED: introduce `tests/qg2b4i_ci_contract.py`, `tests/test_parameter_policy_qg2b4i_ci_migration.py`, this document and a dual-OS workflow. The **new** 13 source tests must each FAIL on its actual corresponding stale workflow, with exact source failure marker, no unrelated ERROR/skip. Independently execute the already-GREEN cumulative B301-B312 and B901-B904 16/16 Host cases.
- Coordinated CI GREEN: patch all 13 affected workflow files atomically so each **executes** `assert_cumulative_admission_green("stage-id")`; keep its inherited independent tests, cases, Standard/Component barriers, job matrix, compiler and own historical source-stage RED/GREEN tests. Replace B312-only stale failures; all 13 new source tests turn PASS. Update the new B4-I workflow from intentional RED to 13/13 actual PASS.
- The shared helper checks exact identities B301-B312 and B901-B904, 16 run, 0 FAIL/ERROR/SKIP/xfail/xpass and tests B312's post-source-gate runtime classification. No fake xfail, no check based on grep alone, no unconditional exit-0.
- Add `tests/qg2b4i_ci_contract.py`, the cumulative suite, API and Launcher paths to **both** push/pull_request filters for all 13 workflows. Review the complete workflow Git diff for unrelated changes.
- Re-run GitHub Actions both OS, verify 13 workflow identities plus B4-I dedicated matrix. Historical Tests-only RED *in other independent areas* is allowed to stay as historically written but cannot become a reason to fake B312 failure or falsely claim broader approval.

## 3. Safety boundaries

Changing an expected-RED CI status to GREEN documents the one already-closed Host transport seam. **It is not** approval to bypass legacy Step 6F, actual Full-BF16 runtime gate, CUDA R1/R2, optimizer, checkpoint/resume, trust roots, or formal evidence qualification. PR #69 remains Draft and unmerged; B4-FSR independent source/code/CI audit remains required.
