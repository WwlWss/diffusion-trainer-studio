# QG-2-MR-FSR-PR69-SR2-E2-C4 — Permanent SR2 & Integrated GREEN

Frozen C3 parent: e930c1397eb65f7be39a9a81970265a2800b0dbf
Tree: 3a16837c25536e723db9158a6f8694d3aab5749a
C3 run 46002994426: both OS 33 target PASS + 3 retained RED, 8 independent and 95 nested baseline PASS; C1 10/10, C2 6/6, C3 7/7; Host 917 identities, 847 PASS + 70 inherited SKIP, Frontend 6/6, CPU 18/18 and 935=917+18 parity.

## Scope and design

Fix D. Preserve original SR2 test file and all 11 actual test identities. Keep historical RED, GREEN candidate and staged contract evidence at original commits; the permanent SR2 workflow has only strict GREEN, no accepted historical RED or candidate classification. Preserve existing job ID qg1b-ci-dependency-red and existing displayed job name to avoid changing the existing check context; the workflow and execution step names describe permanent GREEN. Main plus both original developer branches and dispatch remain. Both filtered events watch exactly 14 explicit paths (8 actual direct producer imports, producer, two producer workflows, original SR2 test, original SR2 document and self). Add exact checkout and no-Torch evidence. Use real unittest starts AND successes, exact method names, exact counts, no FAIL/ERROR/SKIP/xfail/xpass. Each suite produces structured actual identity/outcome records; success only after real 6+5 and exact counts.

E2 workflow transitions from C3 stage (33+3 intentional RED) to permanent strict green entry, which invokes unchanged C1/C2/C3 behaviors, new C4 seven behavior tests and frozen E2-B run_exact("green"). No changes to E2-B assertions, historical stage programs, C1/C2 runner, C1 full Host and C2 CPU jobs, C3 migrated jobs, producer workflows, production, evidence tools or Gitlinks. Add C4 contract/document to both E2 events.

## Qualification

C4 behavior tests exercise the actual embedded SR2 run script (not a duplicate implementation), actual real unittest pass/fail/error/skip/xfail/xpass, identity changes, incomplete execution control flow, event/disabled step/fake command/entry mutations, and actual E2 Host/CPU/strict-green wiring. Main branch event coverage is source-verified only; no actual main push in C4.

Expected dual OS: dedicated SR2 strict 6/6+5/5 actual PASS; E2 integrated 36/36 +8/8 + nested 95/95 baseline + C1 10/10 + C2 6/6 + C3 7/7 + C4 7/7. Re-run both full Host (917, 847 PASS, 70 inherited dependency SKIP, 6 frontend PASS) and pinned CPU 7+11 actual tests and 935 partition parity. All eight jobs must be success, exact HEAD/tree and raw evidence. Historical diagnostic discovery still intentionally observes two old import errors; this is not an active Host execution failure. No new CUDA/backend qualification. Stack merge remains HOLD pending independent E2-D and E2-E/E3-E6. Expected results here are not observations.
