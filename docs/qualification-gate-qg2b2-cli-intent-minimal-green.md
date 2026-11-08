# QG-2-B2 — Direct Trainer CLI Intent minimal GREEN

**Frozen B1 RED parent:** a7172402550f7a648c12a69635182a295c18d4c4.
**Scope:** typed one-shot CLI intent, Stable/Dev parser registration, TOML rejection and output-config non-persistence only.
**PR #69:** Draft / unmerged.

## 1. Implemented B2 contract

A lightweight module `mikazuki/direct_trainer_admission_intent.py` provides:
- frozen `DirectTrainerAdmissionIntent` with strict `type(value) is bool` validation;
- `add_direct_trainer_admission_argument(parser)` registering `--experimental_full_bf16_ack` as `store_true`, default=False`;
- `extract_direct_trainer_admission_intent(args)` consuming and deleting the raw argparse field, then returning the immutable intent;
- `reject_direct_trainer_ack_toml(values)` rejecting the reserved key from flattened TOML values.

The module intentionally imports no torch, Accelerate, sd-scripts or model-aware Trainer.

## 2. Stable / Dev CLI path

Both `scripts/stable/train_network.py` and `scripts/dev/train_network.py`:
1. register the flag in `setup_parser()`;
2. extract the intent immediately after `parse_args()`;
3. extract before command-line validation and `read_config_from_file()`;
4. pass it separately as `trainer.train(args, admission_intent=admission_intent)`.

Both `NetworkTrainer.train()` signatures accept an optional `admission_intent=None` parameter for ABI-compatible handoff. It is not consumed by the Trainer in B2; the Early/Final Host Admission integration remains B3. Existing `trainer.train(args)` callers remain valid.

## 3. TOML and output-config trust boundary

Both stable and dev `read_config_from_file()` implementations:
- flatten TOML sections using the existing behavior;
- call `reject_direct_trainer_ack_toml(ignore_nesting_dict)` before constructing `argparse.Namespace`;
- defensively remove `experimental_full_bf16_ack` from the `--output_config` serialization key list.

Therefore a TOML root or section key cannot create an intent. A raw flag is extracted before the output-config path and cannot be serialized. The defensive serialization removal also protects callers that invoke `read_config_from_file()` without first using the main entrypoint extractor.

## 4. B2 tests and exact gate

`tests/test_parameter_policy_qg2b2_cli_intent.py` contains exactly eight passing Host tests:
1. frozen intent / strict real boolean;
2. one-shot extraction / raw Namespace field removal;
3. real argparse `store_true` behavior and default false;
4. flattened TOML reserved-key rejection;
5. Stable/Dev parser registration;
6. TOML rejection before Namespace merge;
7. actual Stable/Dev `read_config_from_file()` output-config behavior without serializing acknowledgement;
8. Stable/Dev main extraction ordering and separate Trainer handoff.

`.github/workflows/qg2b2-cli-intent-host.yml` runs on Ubuntu and Windows with Python 3.11. It requires:
- 42/42 inherited QG-1B / SR1 host tests PASS;
- B2 CLI Intent 8/8 PASS;
- cumulative B301-B311 PASS and B901-B904 PASS;
- exactly one B312 intentional RED with the frozen marker;
- 0 ERROR / SKIP / xfail / xpass.

The workflow ends with the expected non-zero exit only after it verifies the exact B312 failure. B312 remains held for the not-yet-implemented Early Admission, immutable Policy/config snapshot, Late Session revalidation and Launcher/API transport.

## 5. Explicitly out of scope

B2 does NOT implement:
- Early Host Admission or final post-override Admission;
- torch-free bridge integration for the Admission classifier;
- Policy/config fingerprint snapshot;
- late model-aware Session revalidation;
- Launcher/API child process intent transport;
- actual Full-BF16 Start authorization;
- optimizer, dtype, device, root, Checkpoint, Resume or CUDA changes.

The Direct Trainer receives the intent as an unused optional parameter only. The experimental candidate remains on HOLD. PR #69 stays Draft and unmerged.
