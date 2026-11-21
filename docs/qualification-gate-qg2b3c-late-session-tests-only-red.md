# QG-2-B3-C — Late Session Admission Identity / First-party Reclassification Tests-only RED

**Frozen parent:** `b25b0a7b8f03761bb10e993e8fe4b255ad59a737` (B3-B-SR1 dual-OS GREEN). PR #69 Draft/open/unmerged.

## 1. Source-level findings

Stable `scripts/stable/train_network.py:440-444` and Dev `scripts/dev/train_network.py:834-838` call `create_parameter_policy_session(args=args, train_type=parameter_policy_train_type, roots={"network":network})` *after* network creation, optional weight loading and post-network effective-config fingerprint validation. Neither passes the in-memory second-gate `admission_snapshot` nor the one-shot `admission_intent`.

`mikazuki/parameter_policy_trainer.py:1618+` still has only `args,train_type,roots,structural_frozen_parameters`. It reads `_effective_config(args)`, loads the sidecar using its **independent** `load_parameter_policy_file()` implementation, then checks v1 semantics and `parameter_policy_execution_blockers(...require_assets=True)` and constructs the execution contract before scanning live root Parameters. It does not compare all four B3-A identity coordinates or re-run first-party `assess_component_training_admission(...purpose="direct_trainer")`.

The early B3-A loader `mikazuki.parameter_policy.load_parameter_policy_sidecar()` rejects duplicate JSON keys and NaN/Infinity. The late `load_parameter_policy_file()` uses plain `json.loads()`, which by default accepts duplicate keys and nonfinite constants. A malicious or changed sidecar can therefore have different parse semantics between early and late. The target fix MUST delegate to the strict shared parser and preserve canonical digest, *without* changing the legacy autosave filename's 24-character ABI.

B3-B-SR1 already guards post-network `args` drift by rehashing `vars(args)`. This does not replace Late Session's independently recomputed identity, because the policy path may be swapped or rewritten between checks. Some derived Dev trainers call `post_process_network` after the SR1 guard, so the Late Session must not trust solely the earlier guard.

## 2. Frozen B3-C minimal GREEN design — not implemented in this RED

1. Extend **only** the real late session `create_parameter_policy_session()` with **required typed** `admission_snapshot: DirectTrainerAdmissionSnapshot` and `admission_intent: DirectTrainerAdmissionIntent` for an active Component path. The existing public default `train(..., admission_intent=None)` concern (SR1-03) remains separately tracked; never create implicit positive acknowledgement. Do not accept bool/dict/status/forged mutable snapshot as a substitute.
2. Both Stable/Dev base trainer call sites pass the **final second-gate in-memory snapshot** and the same typed once-per-invocation intent. Do not put the snapshot into args/config, env, TOML, task metadata, Sidecar, checkpoint or a module-global.
3. Late `load_parameter_policy_file()` delegates to the shared **strict** `load_parameter_policy_sidecar()`, preserving its established exception type and the **full canonical SHA-256**. The 24-character filename ABI is untouched; no torch from Host into Early.
4. Immediately after snapshotting effective args and reloading strict canonical policy, and **before** `scan_parameter_roots`, `build_parameter_routing_plan`, optimizer construction or any requires-grad changes, recompute `fingerprint_effective_training_config(effective_config)` and call `admission_snapshot.assert_matches(policy_hash=...,effective_config_fingerprint=...,train_type=...,experimental_ack=...)`. No omitted fields, permissive fallback, hash truncation, or checkpoint-restored ack.
5. Re-run the **existing first-party** `assess_component_training_admission(policy,train_type=...,effective_config=...,purpose="direct_trainer",experimental_ack=admission_intent.experimental_full_bf16_ack)` at the Late gate, instead of trusting the Early status or inventing a second readiness classifier. Reject blocked/ack_required and keep experimental_host_candidate **HOLD**. Only the unchanged exact D1 legacy-approved/inactive paths may progress into existing semantic/target/cache/execution/runtime hard checks; no new Full-BF16 Start authorization.
6. Preserve `parameter_policy_v1_semantic_blockers`, `parameter_policy_execution_blockers(...require_assets=True)`, `build_parameter_policy_execution_contract`, `scan_parameter_roots`, actual Accelerator distributed/dtype/device/optimizer checks, network identity and checkpoint/resume invariants. New checks must occur before routing/root scans, not simply after optimizer construction.

## 3. Exact RED and passing baselines

New `tests/test_parameter_policy_qg2b3c_late_session_red.py`: B3C301–B3C306 exactly **six** targeted missing-contract failures, plus B3C901–B3C904 exactly **four** real baselines.

- C301: Stable + Dev call sites pass final `admission_snapshot` and separately typed `admission_intent`.
- C302: real Late Session signature accepts required snapshot and intent; no optional bypass.
- C303: Late Session recomputes four-coordinate B3-A `assert_matches` **before** root scan and optimizer.
- C304: Late Session invokes first-party `assess_component_training_admission` with direct_trainer + exact one-shot ack and holds experimental candidate.
- C305: Late Policy Loader delegates to B3-A's strict loader, enforcing duplicate-key/nonfinite rejection.
- C306: independent Late gate fail-closed re-verification of actual policy/config identity, before side effects, not copying or trusting stale early status.
- C901: B3-A immutable five-field Snapshot and pure SHA-256 fingerprint real.
- C902: pure strict Loader fails nonstandard JSON under real file I/O.
- C903: existing late source v1 semantic, strict execution + require_assets=True, execution contract and root scan remain in original sequence.
- C904: both B3-B-SR1 trainers still check post-network fingerprint and retain current Standard lazy path.

Every expected failing identity must carry `QG-2-B3-C LATE SESSION CONTRACT ABSENT`; no manufactured unconditional failures or xfail/skip. The new runner fails intentionally **only** for exactly six real missing-contract identities and four genuine baselines. Do not conflate inherited cumulative B312 (exactly one prior intentional RED) with new B3-C targeted RED.

## 4. Ubuntu/Windows frozen gate

Python 3.11 isolated Host checks; source AST and actual torch-free B3-A loader/fingerprint tests; no GPU, model or Trainer import required. Preserve:
- QG-1B/SR1 42/42 PASS, QG-2-B2 8/8 PASS,
- B3-A 12/12 PASS; B3-B 12/12 PASS,
- B3-B-SR1 dynamic Host 8/8 PASS,
- B301-B311 + B901-B904 15 PASS and exactly one B312 original marker RED;
- exact B3C301–306 six FAIL with marker and B3C901–904 four PASS;
- 0 unexpected ERROR/SKIP/xfail/xpass, double OS.

## 5. Stop line

Tests-only RED: exactly three new files (document, test, workflow), **zero production edits**. No `parameter_policy_trainer.py`, `parameter_policy_admission.py`, bridges, Stable/Dev trainers, derived trainers, Launcher/API, qualification, optimizer, CUDA or Checkpoint/Resume changes. No B312 GREEN, no new Start rights, no PR merge.

Next phase after independently verifying dual-platform exact RED is B3-C *minimal GREEN*, then fresh independent Final Strict Review of the complete early+late identity chain.
