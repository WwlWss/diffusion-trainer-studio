# DTS D1-SD3 C4-C2-P1-B2-C — protected production Runner issuer: source contract and Tests-only RED

**Stage:** P1-B2-C contract + Tests-only RED. **Frozen reviewed parent HEAD:** `f57207f76667220f043e89ae9af83afebc1f98d5` (P1-B2 Host nonce/closed-Runner mechanism Ubuntu+Windows 12/12 GREEN). This document freezes a **future production authorization boundary**, not a live provisioned signer, physical CUDA parent, or original Load delegation. PR #65 remains **Draft/unmerged**.

## 0. Source audit and exact current blockers

1. `tools/sd3_full_bf16_independent_runner.py` is an executable with `--probe-unprovisioned`, but its `get_pinned_runner_trust_root`, `verify_original_provisional_step1_source` and `issue_provisional_fresh_step1_attestation` deliberately raise. The Python executable has no OS-managed independent signing authority. Its callable interfaces alone provide no privilege or provenance proof.
2. `tools/sd3_full_bf16_runner_trust.py` verifies a correct Ed25519 signature using a **caller-injected TEST** public key and in-process nonce set; it returns only `signature_valid_unqualified`. It is not a production pinned root, protected signer, or parent admission.
3. `tools/sd3_full_bf16_runner_replay.py` uses SQLite to provide durable atomic nonce uniqueness (Host mechanism verified cross-process, Ubuntu and Windows). Its DB path is **caller-selected** and its process does not prove filesystem ownership/ACL separation. It is **not** a production authority or privilege boundary.
4. `tools/sd3_full_bf16_evidence_worker.py` validates tickets and a self-presented resume parent but holds no signing authority. `tools/sd3_full_bf16_worker_step_adapter.py` still denies all original `Accelerator.load_state` before the real prehooks. `C3PhaseEvidence.finalize()` is still fail-closed; `checkpoint_loaded`, `worker_complete` and full CUDA qualification remain blocked.
5. Host generator `tools/prepare_sd3_full_bf16_cases.py` creates manifests/tickets, not a trusted Runner or key provisioner. A Worker manifest path/issuer ID/PID or a file named `physical_binding_valid` cannot supply trust.

## 1. Threat model: genuine OS separation

The Worker is treated as a potentially compromised unprivileged principal with full control of its arguments, inherited environment, Python globals, manifests, Tickets, phase directories, work_root, uploaded/checkpoint bytes, logs, PIDs and all files it can write. The Runner is a separately **provisioned protected service principal**. If the same OS user can read the key or mutate the Runner ledger/verification root, starting a new Python process confers **zero** independent authority.

**Linux target:** dedicated service UID/GID distinct from Worker; kernel-verified peer credentials over protected AF_UNIX transport (e.g. SO_PEERCRED) and protected socket directory; service-owned private key and durable ledger in service-owned paths with effective owner/group/mode and parent-directory checks. Do not infer privilege from self-declared UID/PID or from `chmod 0600` under the *same* UID.

**Windows target:** dedicated service/logon identity with protected token and named-pipe security descriptor; server authenticates the actual connected peer's token/SID using OS APIs and checks service key/container, ledger and directory DACL/reparse conditions. PID, named-pipe name, connection string, registry path and self-reported SID are insufficient; mere file existence or current-user DPAPI does not separate same-user Worker. Reject current-user secrets writable/readable under Worker identity.

**Common:** an independently managed controller provisions private signing material and the verifying public root **outside** the repository, manifests, environment, command line, Worker executable, work_root, test fixtures and CI artifacts. No auto-generation/fallback on first launch, no Worker-supplied key path/issuer label/root, and no in-process private key import into the Worker. Root rotation requires reviewed protected lifecycle; a key ID alone has no trust.

Authenticated IPC is not equivalent to permission to sign. Worker-origin requests remain untrusted until the Runner itself witnesses actual P1-C original Fresh Step-1 physical acceptance. A compromised Worker may request inspection, but cannot supply the accepted source capability, trusted reference identity or issued proof.

## 2. New frozen production source boundary (currently absent)

New module name: `tools.sd3_full_bf16_runner_authority`. The following callable APIs are required for the separately reviewed GREEN design. Their production success signatures remain **unimplemented at RED**; only the specified *negative* interface is pinned now so a hostile Worker cannot choose the principal, key, root, ledger or physical source.

| Name | Forbidden caller-controlled form, must raise ValueError when unprovisioned |
| --- | --- |
| `require_distinct_runner_security_principal` | `worker_identity=...`, `runner_identity=...` claimed UID/SID/PID maps |
| `load_protected_runner_configuration` | any inherited `DTS_RUNNER_*` key/root/issuer/ledger configuration |
| `authenticate_runner_ipc_peer` | `transport=...` dictionary with claimed peer PID/UID/SID or pipe name |
| `open_protected_runner_signer` | `private_key=...`, `key_path=...` from Worker or test code |
| `open_protected_runner_ledger` | `ledger=...` injected Host test object, `work_root=...` or `ledger_path=...` |
| `verify_production_runner_proof` | signed TEST envelope + caller's own public key/nonce set (P1-A API), copied SHA or issuer name |
| `authorize_provisional_fresh_step1_issue` | `worker_candidate=...` self-claimed physical status/Source/Ticket/checkpoint |
| `inspect_runner_credential_acl` | `metadata=...` or `key_path=...` caller-declared ownership/permissions |
| `probe_runner_provisioning_security` | no arguments: **in unprovisioned Host CI**, report exact `{"status":"unprovisioned","may_issue":false,"physical_cuda_qualified":false}` |

These names and negative forms are a **contract proposal pinned by tests**, not proof that a dict or injected callable is a trusted capability. Successful production paths must originate from a kernel-observed, OS-protected service-side capability which cannot be fabricated inside Worker Python. When a design cannot guarantee that provenance, the only permitted result is fail-closed. Production code must not expose a path selecting the test-injected P1-A key or test-injected P1-B ledger as trusted.

Explicit constraints for any future implementation:

- Real security-principal evidence must come from current kernel/token state; reject same UID/SID or cross-user secret readability. Service account startup, credentials and IPC have an independently reviewed provisioning/integration procedure.
- Require protected issuer public root and file permissions with parent path, symlink/junction/reparse and race-safe open/recheck rules. Never trust arbitrary owner/mode dictionaries or a Worker-set environment variable.
- Verify actual IPC connection peer identity using OS-supported APIs. Host unit tests injecting a fake peer are negative only; a positive *protected process* test later must use real credentials.
- The signer does not accept arbitrary Python data, file paths or JSON for signing. A genuine, Runner-owned physical-source capability minted after P1-C must be required. P1-B2-C cannot produce a positive `issue_provisional_fresh_step1_attestation` without that gate.
- Issuance binds P1-A's complete exact signed payload, external pre-pinned root, Fresh Step-1 case/source/ticket/checkpoint/9-file/external LoRA hashes, <=60s validity, audience and atomic one-use nonce. Preserve separate source IDs from signed bytes, and no self-signed envelope key.
- The protected replay ledger must be provisioned under Runner identity and directories that Worker cannot modify or replace. Existing `RunnerNonceLedger(db_path)` proves only a primitive. A Worker-controlled ledger instance is not production admission.
- Do not leak private bytes in stdout/stderr/logs/exceptions/subprocess env/CI. Signer never executes as untrusted Worker's inherited identity or falls back to an ephemeral test key.
- Provisional signature != whole Phase completion != CUDA qualification. The original `Accelerator.load_state`, original hooks, `checkpoint_loaded`, `worker_complete`, C4-D, Fix D and release remain **disabled**.

## 3. Qualification layers: what can turn GREEN and what cannot

**Layer H (normal GitHub Host CI, Ubuntu + Windows):** deterministic negative authorization contract, P1-A/P1-B baselines and unprovisioned Runner. This suite may later become **scoped Host API fail-closed GREEN** when the new source exists. It cannot prove the positive OS security boundary. No privileged CI signing keys, manufactured P1-C parent or fake CUDA artifacts shall be installed to force this green.

**Layer O (future independently administered protected integration executor, NOT run in current ordinary CI):** after provisioning with distinct verified OS principal, test (i) Worker access denied to private key, verifier root and ledger paths, (ii) unauthenticated/same-principal/synthetic IPC refused, (iii) kernel-authenticated correct service peer on both systems, (iv) key rotation and ledger crash/restart/competition boundaries, (v) positive TEST-only signer mechanism without physical parent approval, with real OS metadata and audited service logs. Windows and Linux require **separate** source-specific acceptance. A generic single-account process test cannot satisfy O.

**Layer P (P1-C then C4-C2-P2, still pending):** independent physical original Fresh Step-1 files, 263 LoRA module pairs/789 BF16 tensors, original nine-file inventory, external LoRA writer, physical event chronology/Source/Case/Checkpoint identity, immutable or protected staging. Only then does a Runner-origin provisional proof become a candidate for one original same-instance Load, subsequently requiring post-Load RNG and state verification. No P1-B2-C test can authorize P.

## 4. P1-B2-C Tests-only RED inventory

New file: `tests/test_sd3_full_bf16_worker_c4c_p1b2c_host.py`; **12 test methods** with deterministic discovery and no conditional skips, OS-user creation or privileged fixtures.

- C201–C203: three existing baseline PASS (P1-A test signature remains unqualified; P1-B durable test nonce remains one-use; unprovisioned Runner refuses root/signature).
- C204: untrusted claimed UID/SID/PID cannot constitute protected distinct principal.
- C205: Worker-inherited key/root/issuer/ledger environment cannot provision Runner.
- C206: claimed IPC peer metadata cannot impersonate authenticated OS connection.
- C207: Worker-selected private key/key path cannot open production signer.
- C208: injected Host SQLite ledger or Worker work_root cannot serve as protected ledger.
- C209: ephemeral P1-A TEST signature + caller key cannot become production Runner proof.
- C210: Worker-owned Fresh candidate/`physical_binding_valid` map cannot mint signer capability.
- C211: caller-supplied file ownership/ACL metadata cannot qualify real key protection.
- C212: default Host CI must report **unprovisioned / may_issue=false / physical_cuda_qualified=false**, never production-ready.

At this frozen parent HEAD, all C204–C212 fail on **missing production authority module/entrypoint**, using a dedicated assertion marker `C4-C2-P1-B2-C HOST CONTRACT ABSENT`; C201–C203 pass. Expected **9 targeted FAIL / 3 baseline PASS on each OS**, exactly 12 run, **0 ERROR / SKIP / xfail / xpass**. No TypeError, import crash, missing runtime package, arbitrary test failure or source code modification counts as valid RED.

Dedicated `.github/workflows/sd3-c2b0-p1b2c-red.yml` is a non-CUDA two-OS exact gate with pinned Python 3.11 and existing minimal pure Host requirements. It checks exact failing test identities and the shared marker so unrelated test failures cannot masquerade as intentional RED. The workflow shall be independently read/verified from real Actions logs before freezing.

**Stage discipline:** this round modifies **documentation, new tests and new CI ONLY**. It must not modify the existing Runner, replay ledger, Worker, Step Adapter, training math, production gates or backend qualification. PR #65 stays Draft/unmerged. Later P1-B2-C GREEN must be further separated into Host fail-closed API, independent protected deployment, and actual physical source authorization; independent Final Strict Code Review required before P1-C. 

## P1-B2-C-A: scoped Host fail-closed source implementation

**Source commit:** `ff326cb58893983fdc76a8c9dc7d96c9ac2e87e4`.
The new `tools/sd3_full_bf16_runner_authority.py` exposes all nine
frozen negative APIs. Eight refuse with `ValueError` and a constant
non-secret reason while the production OS authority is unprovisioned;
`probe_runner_provisioning_security()` returns only:

```json
{"status":"unprovisioned","may_issue":false,"physical_cuda_qualified":false}
```

This is an intentional **Host fail-closed API implementation**: it contains
no platform credential APIs, private keys, root/issuer selection, writable
ledger acquisition, test-public-key promotion, signing path or stateful
switch to enable admission. The all-denial implementation prevents accidental
provisioning through Worker dictionaries, fake IPC identity and environment.
These APIs do not yet implement a privileged positive operating-system
signer. The 12 Host tests are sufficient only to accept **Layer H**.

To avoid a false authority claim, **Layer O remains HOLD** until an external
privileged controller provisions a distinct genuine service principal,
kernel-verified IPC peer identity, OS-protected key and pinned root,
service-owned atomic ledger, protected handle/TOCTOU rules, and separately
witnessed true positive/negative integration tests on each platform.
**Layer P remains HOLD** until the authentic nine-file / 789 BF16 tensor
Fresh Step-1, LoRA and physical event checks are independently verified.
Original same-instance `Accelerator.load_state`, `checkpoint_loaded`,
`worker_complete`, CUDA qualification and merge stay blocked.
