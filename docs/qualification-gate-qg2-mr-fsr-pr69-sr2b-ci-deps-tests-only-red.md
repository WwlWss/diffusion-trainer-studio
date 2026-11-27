# QG-2-MR-FSR-PR69-SR2-B: CI Dependency Tests-only RED

- Frozen parent: `806bd4dda0926677ad40df8413ede336f25249b3` (PR #69 Draft, unmerged).
- Review branch: `feature/qg2-mr-fsr-pr69-sr2b-ci-deps-red`.
- Scope: **tests, a dedicated CI workflow and this review record only**.
- No production source, training scripts, qualification tables, CUDA, checkpoint or existing workflow modifications.

## Source-anchored missing dependencies

The tests parse the **real** `mikazuki/parameter_policy_readiness_producer.py` AST and require every targeted module to remain a direct first-party import. Then each test reads the **real** workflow `on.push.paths` and `on.pull_request.paths`, refusing missing, malformed or duplicate event lists. Both event forms must explicitly watch the source file; a single event is insufficient.

| Test | Workflow | Direct imported file missing at frozen parent |
|---|---|---|
| SRB101 | QG-1B Original | `mikazuki/parameter_policy_evidence_coverage.py` |
| SRB102 | QG-1B Original | `mikazuki/parameter_policy_runtime_bools.py` |
| SRB103 | QG-1B SR1 | `mikazuki/parameter_policy.py` |
| SRB104 | QG-1B SR1 | `mikazuki/parameter_policy_compat.py` |
| SRB105 | QG-1B SR1 | `mikazuki/parameter_policy_matrix.py` |
| SRB106 | QG-1B SR1 | `mikazuki/parameter_policy_runtime_bools.py` |

Five independent, genuine baselines SRB901-905 check the exact direct-import inventory, existing `push/main`, both event filters, Ubuntu/Windows matrix, old explicit test count and zero-skip/xfail/fail-closed CI semantics.

## Exact valid RED/next GREEN acceptance

- Test discovery: **six targeted + five baselines**, one exact identity each.
- Valid RED: **all six targets fail** with `QG2-MR-SR2-B CI DEPENDENCY TRIGGER ABSENT`, **all five baselines pass**, zero unexpected ERROR/SKIP/xfail/xpass.
- **The GitHub Actions job must actually fail** (`exit 1`) on a valid RED. Never use `continue-on-error`, xfail, mocked green or special success status.
- Next phase SR2-C fixes **only** two QG-1B workflow path lists, adding all six targeted path memberships to **both** `push.paths` and `pull_request.paths`. No broad `mikazuki/**` shortcuts or runtime changes.
- Then the same tests must genuinely pass 11/11, exact dual-OS CI becomes GREEN, and an independent review checks that original 16/16 and 42/42 Host tests remain genuine GREEN. This RED is not a production/training qualification.
- Final PR #69 CI and exact SHA CUDA evidence must be handled separately after all source changes settle. Historical evidence at `806bd4d` remains attached to that SHA and must not be silently relabeled.

## Evidence boundary

A local in-container test-shape simulation is preliminary only. The accepted RED must come from GitHub Actions at the actual review-branch HEAD, with explicit Windows and Ubuntu job logs and exact failure identities. An untested or missing GitHub workflow run is **HOLD**, not RED PASS.