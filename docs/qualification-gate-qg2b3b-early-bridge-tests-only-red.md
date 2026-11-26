# QG-2-B3-B — Stable/Dev Early Admission Bridge source contract and Tests-only RED

**Frozen parent:** QG-2-B3-A scoped Host GREEN at `c6719ed9896aeb4b26b272dcb468a122342330be`.
**PR:** #69 Draft/open/unmerged, stacked on #68/#67.
**Scope:** ONLY this contract, one tests-only suite, and isolated Ubuntu/Windows exact-RED CI. No production modifications. B312 remains exactly one cumulative RED.

## 1. Actual source inspection and rationale

Stable `scripts/stable/train_network.py:150-174` and Dev `scripts/dev/train_network.py:517-541` already accept a separate `admission_intent`, detect optional policy, lazily import their corresponding `library.dts_parameter_policy_bridge` and run `preflight_component_runtime_booleans()`, before `verify_training_args()`, dataset setup, logging, tokenizer/cache, Accelerator, model and network loads. They do NOT call the B3-A snapshot factory.

Stable loads a tokenizer near 187, datasets near 191+, calls `self.assert_extra_args(...)` near 261 and prepares Accelerator near 265. Dev initializes tokenization near 554, dataset near 562+, calls `self.assert_extra_args(...)` near 646 and prepares Accelerator near 650. Some derived trainers normalize FP8/cache/backend flags in `assert_extra_args` (e.g. Flux), so an initial check alone cannot bind final effective config.

The Stable/Dev lazy bridges currently export `preflight_component_runtime_booleans`; their `load_parameter_policy_trainer()` imports the torch-heavy `mikazuki.parameter_policy_trainer`. Early Admission MUST NOT invoke it. B3-A already supplies the pure `prepare_direct_trainer_admission(policy_path, *, train_type, effective_config, admission_intent)` and `DirectTrainerAdmissionSnapshot`.

The late `create_parameter_policy_session` is model/root-aware and must remain after actual network load. It is NOT part of B3-B. No claimed early Host admissibility replaces real root/optimizer/dtype/device/distributed/Checkpoint checks.

## 2. Frozen next-production API and ordering (not implemented by RED)

### 2.1 Both lazy bridges

Add a named public `preflight_direct_trainer_admission(*, policy_path, train_type, effective_config, admission_intent)` returning the exact first-party `DirectTrainerAdmissionSnapshot`, as the sole early bridge to the B3-A Factory. The bridge bootstraps the DTS root in the existing way, imports ONLY the pure `mikazuki.parameter_policy_admission` at call time and delegates to `prepare_direct_trainer_admission`. It must not load the Trainer, torch, Accelerator or live model roots. Keep existing `preflight_component_runtime_booleans()`, `load_parameter_policy_trainer()`, session wrappers and Standard laziness intact.

Do not reimplement Policy parsing, config hashing, readiness, or classifier logic in sd-scripts.

### 2.2 Two gates in both base train() methods

With `parameter_policy_requested`, require exact `DirectTrainerAdmissionIntent` (a plain bool, dict, subclass or copied TOML field cannot be authority). Preserve the already-present Boolean preflight. After existing required argument validation/normalization but **before** setup_logging, tokenizer, strategy, dataset, Accelerator or model loading, call the pure bridge with the actual policy sidecar path, resolved `get_parameter_policy_train_type(args)`, detached effective config and typed one-shot intent.

The first check rejects invalid/missing Policy, unsupported backend, real readiness hard blockers, `ack_required` and `blocked` immediately. A returned `legacy_approved` may proceed only through existing hard checks. `experimental_host_candidate` remains a scoped HOLD: it must not enable previously forbidden real training. Neither `host_admissible` alone nor a caller-supplied status grants permission. `inactive` must not manufacture a new training privilege.

After `self.assert_extra_args()`, regenerate the detached **final effective config** from the now-mutated args and call the same first-party bridge **again**, directly before `train_util.prepare_accelerator()`. Do not silently reuse the first snapshot. The second result is the final pre-model snapshot for future B3-C late-session matching. If either check fails, fail before expensive side effects. Existing model-family setters and source-audited backend allow-list remain unchanged.

The acknowledgement stays out of both effective config views. No sidecar writes, JSON export, task metadata, env persistence or Checkpoint mutation. No new launcher/API transport in B3-B.

### 2.3 Stage boundary

B3-B may store the final snapshot in a **local in-memory variable** during `train()`; it must not pass it into `create_parameter_policy_session()` yet. B3-C separately enforces reloading/identity comparison and re-runs late first-party Admission before root scan. B312 remains one intentional cumulative RED until the full Trainer/Launcher contract is complete.

## 3. Exact Tests-only RED targets

`tests/test_parameter_policy_qg2b3b_early_bridge_red.py` has exactly eight targeted failures (B3B201–B3B208) and four real pre-existing baselines (B3B901–B3B904).

- B3B201: same public pure early function exists in BOTH lazy bridges.
- B3B202: Stable base initial early gate before logging/tokenizer/dataset/Accelerator.
- B3B203: Dev base initial early gate before tokenization/dataset/Accelerator.
- B3B204: bridges delegate to the exact B3-A first-party factory; no heavy trainer import on early path.
- B3B205: first-party result statuses fail-closed; candidate remains HOLD; no bare host-admissible bypass.
- B3B206: both trainers recheck effective config AFTER `assert_extra_args()` and BEFORE Accelerator, not reuse initial snapshot.
- B3B207: typed one-shot intent plus policy path/train-type/config is carried through both checks; no persistence or new source authorization.
- B3B208: snapshot remains in-memory and only pre-model; B3-C late Session/Launcher/API is explicitly NOT implemented in this phase.

Each targeted failure must carry exact `QG-2-B3-B EARLY BRIDGE CONTRACT ABSENT`, with one failing test method per identity, zero ERROR/SKIP/xfail/xpass. No unconditional failing assertion, fake xfail or catch-all skip.

Passing baselines:
- B3B901: B3-A pure factory and snapshot exist; no heavy top-level imports.
- B3B902: existing Stable/Dev Boolean preflight remains before tokenizer/dataset/Accelerator.
- B3B903: preexisting per-invocation CLI intent strict typing and one-shot removal.
- B3B904: both bridges remain lazy for their existing session loader; Standard has no unconditional import.

## 4. Two-OS CI contract

Python 3.11 Ubuntu and Windows:
1. 42/42 prior QG-1B/SR1 host tests PASS.
2. QG-2-B2 eight-case CLI intent PASS.
3. B3-A 8 targets + 4 baselines PASS.
4. Cumulative B301–B311 eleven PASS + B901–B904 four PASS; B312 exactly one intentional RED with original marker.
5. B3-B targeted B3B201–B3B208 exactly eight FAIL with new exact marker; B3B901–B3B904 four real PASS. No unexpected ERROR/SKIP/xfail/xpass.

The CI deliberately exits nonzero ONLY if all expected RED identities and all passing gates match. Any other result is invalid, not accepted.

## 5. Stop line

Do NOT edit `mikazuki/parameter_policy.py`, `mikazuki/parameter_policy_admission.py`, `mikazuki/parameter_policy_trainer.py`, `scripts/{stable,dev}/train_network.py`, either bridge, derived trainer, Launcher/API, qualification table, optimizer, runtime evidence, Checkpoint/Resume, or CUDA in this RED stage. Do NOT remove candidate HOLD, mark B312 GREEN, merge #69 or grant new Full-BF16 execution rights.

Next stage, **only after valid exact Ubuntu/Windows RED**, is scoped B3-B minimal GREEN in the two bridges and two base trainers, followed by independent Strict Review. B3-C late identity verification is a separate stage.
