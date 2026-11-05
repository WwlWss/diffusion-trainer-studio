# DTS D1-SD3 C4-C2-P1-B2-C — protected production Runner issuer: source contract and staged qualifications

**Current stage:** P1-B2-C-A-SR1 scoped Host fail-closed. The existing `tools.sd3_full_bf16_runner_authority` module exposes only negative authorization gates and an untrusted `unprovisioned` probe. **Historical RED design parent:** `f57207f76667220f043e89ae9af83afebc1f98d5`; **historical P1-B2-C Tests-only RED HEAD:** `74d637ea20db2888d0f77edb2b369988ea9835b1`; **P1-B2-C-A scoped Host GREEN HEAD:** `df7d359f597991d85274a361703e201742333323`. This document freezes a future **OS-protected production authorization boundary**. No protected signer, physical CUDA Fresh parent, original Load delegation or `worker_complete` has been enabled. PR #65 remains **Draft/unmerged**.

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

## 2. Frozen production source boundary (Host fail-closed only)

`tools.sd3_full_bf16_runner_authority` now exists as a **fail-closed Host-only source boundary**. The nine callable APIs below are implemented for the reviewed negative cases; eight deny unconditionally and one reports an untrusted diagnostic. **No positive production implementation, separate protected OS principal, or independently provisioned signer exists.** The historical RED baseline froze these names so a hostile Worker could not choose principal, key, root, ledger or physical source.

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

## Historical P1-B2-C Tests-only RED baseline

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

At the historical Tests-only RED HEAD `74d637ea20db2888d0f77edb2b369988ea9835b1`, C204–C212 failed on **missing production authority module/entrypoint**, using the dedicated assertion marker `C4-C2-P1-B2-C HOST CONTRACT ABSENT`; C201–C203 pass. Expected **9 targeted FAIL / 3 baseline PASS on each OS**, exactly 12 run, **0 ERROR / SKIP / xfail / xpass**. No TypeError, import crash, missing runtime package, arbitrary test failure or source code modification counts as valid RED.

Dedicated `.github/workflows/sd3-c2b0-p1b2c-red.yml` is a non-CUDA two-OS exact gate with pinned Python 3.11 and existing minimal pure Host requirements. It checks exact failing test identities and the shared marker so unrelated test failures cannot masquerade as intentional RED. The workflow shall be independently read/verified from real Actions logs before freezing.

**Historical RED-only stage discipline:** the initial RED commit changed documentation, new tests and CI only. P1-B2-C-A subsequently added the separate fail-closed Host module without enabling issuance; SR1 hardens the tests and diagnostic. The existing independent Runner, replay ledger, Worker, Step Adapter, training math, real Load and CUDA/release gates must not be modified during SR1. PR #65 stays Draft/unmerged. Real protected service deployment and physical source authorization remain separately qualified stages. 

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


## P1-B2-C-B: protected executable and import integrity

**Mandatory HOLD contract for future Layer O; SR1 does not install or run a privileged service.**
A distinct OS account with protected signing-key/ledger ACLs cannot provide
credible signing authority if the privileged process executes code or loads
dependencies from any Worker-writable path. The separately administered Runner
provisioner must audit and lock the *complete* executable/import trust chain.

1. **Protected installation origin:** Run an explicitly pinned, reviewed Python
   interpreter and Runner code package from a service/controller-owned,
   Worker-nonwritable deployment root, outside repository checkout and
   `work_root`. Verify real filesystem ownership/ACL for every component
   and ancestor directory, interpreter binary, entrypoint, modules and
   dependency directory, and reject symlink/junction/reparse path substitution.
   Service updates and rotation require independently authenticated deployment;
   neither Worker nor compromised ordinary service principal may replace the
   installed code as part of normal operation. No auto-update from a
   Worker-selectable Git branch, URL, manifest or plugin path.
2. **Dependency provenance:** Pin reviewed Python runtime, installed wheels,
   native extension modules and dependency digests against independently
   provisioned manifests. Treat a matching package version string alone
   as insufficient. Validate effective executable and imported file paths
   after startup, including `.pth` execution, `sitecustomize`,
   `usercustomize` and native/extension-library loads. The effective module
   origin must not be under a Worker-editable install or a writable site-packages.
3. **Sanitized launch:** The service manager, not a Worker child or its shell,
   constructs the command, effective UID/token, cwd, env and import search
   paths. Reject Worker-controlled `PYTHONPATH`, `PYTHONHOME`,
   `PYTHONUSERBASE`, `PYTHONSTARTUP`, user-site, inherited shell hooks,
   arbitrary `sys.path` or `PATH` additions, and unreviewed plugin loaders.
   Assess Python `-I` / isolated mode as defense in depth, but verify actual
   import resolution and required packages instead of treating the flag as
   independent proof. Worker-selected cwd must never become an import or
   dynamic-loader root.
4. **Native code / cross-platform boundary:** On Linux, independently verify
   actual service UID/GID and directory modes plus dynamic loader variables
   (including `LD_LIBRARY_PATH` and `LD_PRELOAD`) are not Worker-supplied.
   On Windows, separately verify the service token/SID, executable and parent
   DACL, reparse points, safe DLL search order, known DLL/native extension
   paths and protected service configuration. A self-reported ACL dict,
   executable hash in a Worker JSON blob, environment flag or distinct PID
   cannot qualify either platform.
5. **Actual protected integration acceptance (Linux and Windows separately):**
   From the unprivileged Worker account, attempt to replace the service
   entrypoint, dependency, `sitecustomize`, `.pth` payload, a shadow
   `cryptography`/standard library module, cwd and dynamic library.
   Confirm the legitimate service refuses contaminated startup or continues
   to use the independently provisioned immutable originals. Independently
   verify the Worker cannot read the private key, modify the pinned verifier
   root or reset/replace the durable nonce ledger. Verify the real IPC peer
   via OS kernel/token APIs, including negative same-principal and spoofed
   PID/SID cases. Record paths, effective security identities and test
   outcomes without logging private keys or tokens. An ordinary GitHub
   same-account Host job is **not** this qualification.
6. **Signer/source gate remains separate:** Even a perfectly protected
   executable/IPC/key does not permit signing Worker-supplied maps. A
   Runner-origin, independently witnessed and protected original physical
   Fresh Step-1 capability is required by P1-C before provisional issuance.
   Original `Accelerator.load_state` and any qualifying
   `checkpoint_loaded`/`worker_complete` event remain **HOLD**.

**SR1 review resolution:** F1 adds no-arg/default/alternate-arg, untrusted
object, secret-leak, I/O-side-effect and poisoned-environment negative Host
coverage in `tests/test_sd3_full_bf16_worker_c4c_p1b2c_sr1_host.py`.
F3 makes the probe's `"unprovisioned"` value literal and diagnostic-only;
this is not Python module integrity or a signed receipt. The SR1 CI gate
counts exactly eight Host tests and permits only its frozen two-defect
Tests-only RED parent or an all-eight PASS scoped GREEN. Real protected
service/physical CUDA positive acceptance remains unavailable.
