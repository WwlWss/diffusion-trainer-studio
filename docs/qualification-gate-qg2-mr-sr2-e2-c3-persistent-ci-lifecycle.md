# QG-2-MR-FSR-PR69-SR2-E2-C3 — Persistent CI Lifecycle GREEN

Frozen parent: 94228074e9eec37fa0d07472c86724e1c9664645
Tree: 9e7552454ae20b8debccbdfd86b4672595806930
C2 run 46001223874: 5 target GREEN + 31 retained RED; 8 independent and 95 nested baseline PASS, C1 10/10 and C2 6/6. Both OS full Host 917 IDs, 847 PASS + 70 original dependency skips, Frontend 6/6; pinned CPU 7+11 PASS and 935=917+18 discovery parity.

## Scope and decisions

Fix C only. Add main alongside original development branch for 17 QG workflows; preserve dispatch. Anima gains main-only push with the same 158 PR paths. Preserve unfiltered PR for FSR runner, B4-FSR and B4-I. Dynamic Host gains PR. Align six known asymmetric pairs, expand Dynamic/FSR/B4-A1 paths, and add actual explicit compile/run test references found across all 17 migrated workflows. A2 also watches frontend Gitlink and .gitmodules. The existing B4-I broad source guard remains unchanged. No claim of complete transitive dependency closure for every historical job. SR2 is untouched until C4. Preserve original developer branch filters; do not add independent branch to all legacy workflows. Historical execution bodies, names, job IDs, conditions, matrices, permissions remain unchanged.

79 previously missing workflow-to-explicit-input relations were identified across the 17 QG workflows (each input counted once per workflow when absent from both original path lists). The E2 workflow also watches the 24 original QG workflow files and C3 test/document, inherited test contract files and SR2 contract inputs. Added 32 path entries per event to existing C2 triggers; no all-repository wildcard. Future main CI fan-out increases intentionally; no consolidation or automatic GPU execution.

| Workflow | Added push paths | Added PR paths | New explicit input relations |
|---|---:|---:|---:|
| qg1b3-qg2-admission-host.yml | 0 | 0 | 0 |
| qg1b3a-admission-adapter-host.yml | 3 | 5 | 3 |
| qg1b3b-fsr-runner.yml | 15 | 0 | 0 |
| qg1b3b-typed-migration-host.yml | 4 | 4 | 4 |
| qg2a1-request-intent-host.yml | 5 | 5 | 5 |
| qg2a1-sr1-host.yml | 1 | 1 | 1 |
| qg2a2-api-host.yml | 12 | 12 | 9 |
| qg2b2-cli-intent-host.yml | 0 | 0 | 0 |
| qg2b3a-policy-snapshot-host.yml | 0 | 0 | 0 |
| qg2b3b-early-bridge-host.yml | 4 | 6 | 4 |
| qg2b3b-sr1-dynamic-host.yml | 5 | 9 | 0 |
| qg2b3c-late-session-host.yml | 7 | 11 | 7 |
| qg2b3c-sr1-cpu-runtime.yml | 9 | 10 | 9 |
| qg2b4-0-b312-contract-red.yml | 11 | 11 | 11 |
| qg2b4a1-api-handoff-red.yml | 19 | 20 | 14 |
| qg2b4fsr-sr1-ci.yml | 1 | 0 | 0 |
| qg2b4l1-launcher-transport-red.yml | 15 | 16 | 12 |
| anima-qwen3-review.yml | 158 | 0 | 0 |

## Stage acceptance

Original E2 36+8, C1/C2 contracts and test runner remain unchanged. Exact 33 target PASS, E2L118/E2S101/E2S102 marker failures (exit 1), eight baselines including 95 nested cases, C1 10/10, C2 6/6, C3 7/7. C3 validates actual main/PR/dispatch events, direct job compile/run inputs, Dynamic/FSR/B4-A1 inputs, unchanged historical job SHA256, reviewed event/header SHA256, unfiltered PR, unchanged B4-I/GPU, E2 trigger closure, and negative mutations. Header/job hashes are scoped audit snapshots to be reviewed if later intentional changes occur; not permanent restrictions on workflow evolution. SR2 blob guard is stage-only, allowing C4 to run inherited C3 behaviors. Re-run both full Host and pinned CPU jobs unchanged on both OS. The original jobs are preserved, but this is not actual main-push execution of all historical workflows. Final integrated CI and merge proof belong to E3/E4.

No production, existing test source, GPU tools or Gitlink changes. No CUDA or backend qualification; main and PR67/68/69 remain frozen. No merge authorization. Expected results are not observations.
