# QG-2-B3-A — Torch-free Policy Sidecar Loader and Admission Snapshot Tests-only RED

**Frozen parent:** QG-2-B2 scoped GREEN at a3aa22029676c620df8247fe8fe2917a09cbf625.
**PR:** #69, Draft and unmerged.
**Scope:** architecture contract + tests + dual-platform CI only. No production source changes.

## 1. Why this seam exists

The host configuration module mikazuki/parameter_policy.py already owns canonical Policy validation and serialization and imports no torch. However, file loading currently lives in mikazuki/parameter_policy_trainer.py, which imports torch at module import time. The early Direct Trainer Admission cannot import that module before tokenizer/dataset/Accelerator/model setup.

B3-A freezes the minimal extraction:
- a canonical, strict JSON sidecar loader in the existing torch-free parameter_policy.py;
- a canonical effective-config fingerprint function in the host Admission layer;
- a frozen DirectTrainerAdmissionSnapshot that carries only hashes, train type, the strict one-shot acknowledgement bit and immutable first-party ComponentAdmission;
- a prepare_direct_trainer_admission() factory that loads/validates the Policy and invokes the existing first-party classifier with purpose="direct_trainer".

This is a foundation only. B3-A does not wire it into trainer bridges or move any training lifecycle.

## 2. Frozen production API expected by tests

### 2.1 mikazuki.parameter_policy.load_parameter_policy_sidecar(path)

- Accept a nonempty str | os.PathLike[str] path; require a regular file and read UTF-8.
- Parse strict JSON: reject malformed JSON, non-object root, duplicate object keys and non-standard constants (NaN, Infinity, -Infinity).
- Call the existing validate_parameter_policy(); do not create a second Policy schema/normalizer.
- Canonicalize via the existing serialize_parameter_policy() canonical JSON content.
- Return (canonical_policy, policy_hash) where policy_hash is the full 64-character SHA-256 hex digest of canonical serialized content including its existing trailing newline.
- The content-addressed filename continues to use the existing 24-character prefix. That short filename prefix is not the snapshot identity.
- Whitespace/key ordering in the source JSON must not change the canonical hash. A real semantic Policy change must change it.
- No write, no torch, no model/runtime imports. The canonical returned mapping must not alias caller-owned JSON objects.

The current Trainer loader must later delegate to this one shared pure implementation; B3-A itself does not change the Trainer.

### 2.2 fingerprint_effective_training_config(config)

- Accept a Mapping with string keys and JSON-compatible scalar/list/tuple/mapping values.
- Canonicalize mappings by key order and tuples as JSON arrays; preserve scalar types.
- Reject non-string mapping keys, non-finite floats, sets, bytes, paths/custom objects and any nested occurrence of the reserved control key experimental_full_bf16_ack.
- Use deterministic UTF-8 JSON with ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False.
- Return full lowercase 64-character SHA-256.
- Never mutate or retain references to the caller's config.

The acknowledgement is passed as a separate DirectTrainerAdmissionIntent; it must not become part of the effective config fingerprint.

### 2.3 DirectTrainerAdmissionSnapshot

Frozen, slotted dataclass in mikazuki.parameter_policy_admission with exactly the minimal immutable identity:
- policy_hash: full 64-character canonical Policy SHA-256;
- effective_config_fingerprint: full 64-character canonical config SHA-256;
- train_type: normalized nonempty backend identifier;
- experimental_ack: strict real bool from the typed one-shot intent;
- admission: first-party immutable ComponentAdmission with purpose="direct_trainer".

No raw Policy, config mapping, file path, argparse Namespace, model object, Tensor, credential, Checkpoint data or mutable diagnostic dictionary is retained. The ComponentAdmission and its nested ExecutionReadiness are already frozen and normalize issue collections to tuples.

assert_matches(policy_hash=..., effective_config_fingerprint=..., train_type=..., experimental_ack=...) must fail closed on any identity or intent mismatch. It is a pure identity check, not launch authorization.

### 2.4 prepare_direct_trainer_admission(policy_path, *, train_type, effective_config, admission_intent)

- Requires an actual DirectTrainerAdmissionIntent, not a bool, mapping or caller-provided status.
- Uses the shared pure Policy loader.
- Computes the effective config fingerprint from a detached canonical copy.
- Calls the existing assess_component_training_admission(..., purpose="direct_trainer", experimental_ack=intent.experimental_full_bf16_ack).
- Returns the frozen snapshot.
- Never mutates the input config/intent, writes files, imports torch or changes the existing backend allow-list.
- The snapshot result remains diagnostic/host-only. An experimental_host_candidate does not authorize a launch or remove the existing QG-2-B/final-review HOLD.

## 3. Tests-only RED identities

New tests/test_parameter_policy_qg2b3a_snapshot_red.py contains exactly eight target tests and four passing baselines.

Targeted RED:
- B3A201 — public Policy sidecar loader exists in the existing torch-free host module.
- B3A202 — strict file parsing rejects duplicate keys, non-standard constants, invalid JSON, non-object roots, missing files and invalid schema.
- B3A203 — full canonical 64-character Policy hash is stable across JSON whitespace/key order and changes with semantic Policy content; existing 24-character filename prefix is not treated as identity.
- B3A204 — effective config fingerprint is deterministic, strict, detached and rejects non-JSON values/non-string keys/non-finite floats/nested acknowledgement.
- B3A205 — immutable, slotted snapshot has the frozen minimal field contract and validates its typed admission.
- B3A206 — factory requires the typed one-shot intent and invokes the first-party direct_trainer classifier; pending/no-ack and acknowledged candidate remain distinct, unsupported backend remains blocked.
- B3A207 — snapshot identity assertion rejects changed Policy hash, config fingerprint, train type or acknowledgement.
- B3A208 — no mutable aliases or raw config/policy/path are retained; the factory does not mutate inputs or write files and imports remain torch-free.

Passing baselines:
- B3A901 — existing validate_parameter_policy() and serialize_parameter_policy() schema/canonical content/path ABI unchanged.
- B3A902 — existing D1 Direct Trainer compatibility and unsupported backend hard-block behavior unchanged.
- B3A903 — host Policy/Admission/intent modules have no top-level torch/Accelerate/Trainer imports; Standard lazy Trainer boundary remains unchanged.
- B3A904 — existing frozen DirectTrainerAdmissionIntent strict bool and one-shot extraction behavior remains unchanged.

No unconditional failures, skip, xfail, expected-failure decoration or failure fan-out. Each missing source contract produces one exact marker:
QG-2-B3-A POLICY SNAPSHOT CONTRACT ABSENT.

## 4. Dual-platform gate

New .github/workflows/qg2b3a-policy-snapshot-host.yml runs Python 3.11 on Ubuntu and Windows and requires:
- prior QG-1B/SR1 host baseline: 42/42 PASS;
- QG-2-B2 CLI intent: 8/8 PASS;
- cumulative B301-B311: 11 PASS, B901-B904: 4 PASS, exactly one B312 RED with its frozen marker;
- B3-A: exactly eight targeted RED plus four actual baseline PASS;
- zero unexpected ERROR/SKIP/xfail/xpass.

The B3-A workflow exits non-zero only after it verifies all eight exact target failures and all four baseline passes on both operating systems. An unrelated failure or error invalidates the RED.

## 5. Stop line

B3-A is Tests-only RED. Do not modify:
- parameter_policy.py production loader;
- parameter_policy_admission.py snapshot/factory;
- parameter_policy_trainer.py or the two lazy bridges;
- Stable/Dev NetworkTrainer.train() or tokenizer/dataset/Accelerator/model lifecycle;
- Launcher/API, optimizer, dtype/device/root checks, Checkpoint/Resume, CUDA or qualification evidence.

After valid dual-OS RED, stop. The next approved implementation stage is B3-A minimal GREEN for the pure loader, fingerprint and snapshot only. Early bridge integration is a separate B3-B stage; Late Session revalidation is separate B3-C. PR #69 remains Draft/unmerged and no experimental Full-BF16 Start is authorized.
