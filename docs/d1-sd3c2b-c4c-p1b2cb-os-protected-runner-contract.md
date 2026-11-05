# DTS C4-C2-P1-B2-C-B: OS-protected independent Runner authority

**Stage:** B0 source contract + B1 Tests-only RED. **Reviewed parent:** `c74bc2a54f4b02750dab43861b5d379de62eb8e7` (P1-B2-C-A-SR1 Scoped Host Final Strict PASS). **Current security:** NOT_PROVISIONED; `may_issue=false`, `physical_cuda_qualified=false`. PR #65 remains Draft/unmerged.

## 1. Trusted principals and non-goals

The Worker controls its arguments, environment, manifests, Tickets, claimed IDs, writable paths, Python modules, staged checkpoint bytes, and all files accessible under its OS identity. Running a second Python subprocess with a different PID under the same identity does **not** create an independent authority. The **independently administered controller** must install and start the Runner under a distinct protected service principal from an executable/import tree the Worker cannot modify. No Worker-supplied key, trust root, peer UID/SID/PID, nonce ledger, signed TEST envelope, or allegedly verified Fresh parent confers service authority.

Key material, pinned public verifier root, IPC endpoint policy and durable nonce store originate from controller-owned provisioning. The public root can be readable; it **must not be writable or replaceable** by the Worker. Provisioning must never silently generate a TEST key or fall back to a user-owned signer.

**This stage cannot authorize arbitrary signing, real original Accelerator.load_state, checkpoint_loaded, worker_complete, CUDA qualification or PR merge.** Even with a real service, provisional signing requires P1-C's independently observed and physically verified Fresh Step-1 capability.

## 2. Per-platform implementation contracts

### Linux: protected service and kernel AF_UNIX peer

1. Service manager starts a dedicated service UID/GID and approved supplementary groups and capabilities, separate from the Worker. Require actual kernel process identity; claimed UID/PID values are not evidence. A Worker-launched subprocess is not a service.
2. Controller installs the pinned interpreter and reviewed code, native extensions, dependencies and signed/hash-pinned deployment inventory under a service/controller-owned, Worker-nonwritable root. Inspect actual ancestors, file ownership, directory access, symlinks and replacement races.
3. Server exposes a protected AF_UNIX stream socket from a controller/service-owned parent. On the **connected** socket it obtains Linux `SO_PEERCRED` using `getsockopt`, validates actual peer credentials and authorized service policy, and rejects fake peer maps, same-principal and unauthorized connections. Linux peer credentials are connect-time snapshots; do not confuse them with provenance of arbitrary payload bytes.
4. Only the service principal can open the signing key and atomically consume its service-owned durable replay ledger. Public root path is pre-pinned and non-Worker-writable. Do not trust Worker-selected key/ledger paths or `chmod` mode strings supplied in a manifest.
5. Launch from a protected work directory with sanitized environment, verified module/import origins and no Worker-controlled `PYTHONPATH`, `PYTHONHOME`, `PYTHONUSERBASE`, user-site, `sitecustomize`, `.pth`, `LD_PRELOAD` or `LD_LIBRARY_PATH`. Python `-I` can help but is not sufficient without actual dependency provenance.

### Windows: protected service and authenticated named pipe

1. Trusted Service Control Manager / controller starts a dedicated service logon identity under a genuinely different `TokenUser` SID. Inspect effective process access token and service configuration. A Worker-origin `subprocess.Popen` with a different PID is not an independent service.
2. Controller protects the interpreter, Runner modules, DLL and `.pyd` dependencies, key, verifier-root integrity, ledger, service install directory and ancestors via effective DACL. Check reparse and DLL search paths; Worker-controlled DPAPI or claimed `icacls` text cannot prove independence.
3. Server creates a named pipe with an explicitly restrictive security descriptor. After connecting and reading an authenticated request, server checks `ImpersonateNamedPipeClient` succeeded, obtains actual client `OpenThreadToken`, inspects `TokenUser` via `GetTokenInformation`, verifies permitted token attributes, and **always** `RevertToSelf` in `finally`. Impersonation failure must ABORT: never process a privileged request under the original server token. Neither pipe name nor client PID nor self-declared SID authenticates the peer.
4. Privileged service uses sanitized cwd/env/import paths and protected DLL search order. It cannot load arbitrary Worker-supplied modules, plugins, `sitecustomize`, `.pth`, scripts, dependencies or service update manifests.
5. A correctly authenticated Worker peer may ask the Runner to inspect evidence but cannot mint a signature by submitting its own purported Fresh candidate.

### Shared source/install integrity and permission transition

- The controller independently pins the interpreter, entrypoint, modules, native extensions and dependencies (digest plus origin), actual effective owner/ACL, protected startup arguments/env and immutable service trust config.
- The service verifies code origin and loaded module locations at startup and refuses Worker-editable parents or import/dynamic-loader paths. No first-run self-provisioned key, test-key fallback, arbitrary trusted object injection or caller-selected verifier root.
- All privileged file access must use reviewed anti-reparse/symlink/hardlink and TOCTOU-resistant handle semantics, including SQLite ancillary WAL/shm/journal paths where relevant. A caller-selected `RunnerNonceLedger(db_path)` proves ONLY the prior Host primitive.
- Positive OS qualification requires **real kernel identities on both ends**, real access-denied probes from the Worker account, actual protected service startup and negative executable-import tampering. Pure Host/same-user CI can only prove rejection behavior.

## 3. Frozen future module interface (B1 RED; no source implementation yet)

Future protected server-side module: `tools.sd3_full_bf16_runner_os_boundary`. The Worker must never import its functions as permission-granting capabilities. This table freezes only negative Host behavior; successful interfaces need a separate reviewed native capability provenance.

| API | Required unprovisioned Host behavior |
| --- | --- |
| `require_protected_service_principal` | Reject claimed Worker/Runner UID/SID/PID and same-account fallback |
| `require_protected_executable_origin` | Reject worker-controlled runtime, code path, module hashes and editable source |
| `require_sanitized_service_startup` | Reject worker env, cwd, `PYTHONPATH`, DLL and dynamic-loader paths |
| `require_authenticated_ipc_service` | Reject claimed pipe/socket names or service PID as proof of server |
| `require_kernel_authenticated_peer` | Reject fake metadata and Linux real same-UID `socketpair`; Windows must require real pipe handle and token |
| `require_service_owned_signing_key` | Reject injected key bytes / Worker key paths |
| `require_protected_verifier_root` | Reject Worker-chosen Ed25519 public key/root path |
| `require_service_owned_replay_ledger` | Reject injected same-account SQLite ledger / Worker work_root |
| `probe_runner_os_protection` | Return ONLY `{"status":"unprovisioned","os_security_qualified":false,"may_issue":false,"physical_cuda_qualified":false}` in ordinary CI |

## 4. Tests and acceptance plan

**B1 Tests-only RED:** file `tests/test_sd3_full_bf16_worker_c4c_p1b2cb_host.py`, exactly 12 tests O301-O312. O301-O303 establish 3 existing baselines PASS; O304-O312 are nine intended tagged `C4-C2-P1-B2-C-B OS BOUNDARY CONTRACT ABSENT` FAIL because the new module is not implemented. No ERROR/SKIP/xfail/xpass. Workflow `.github/workflows/sd3-c2b0-p1b2cb-red.yml` checks exact failing method IDs on Ubuntu and Windows. Do not expose service secrets or fake CUDA files to ordinary CI.

**B2 Host negative source (later):** native service observations and fail-closed startup operations can be implemented in independently reviewed Linux/Windows slices. 12/12 Host PASS is, at most, **scoped unprovisioned Host-negative GREEN**, never protected service authority.

**B3 genuine protected service integration (separate environment):** after out-of-band controller provisioning, verify on BOTH Linux and Windows that: (a) Worker cannot read/replace key, edit public root or reset/replace ledger; (b) Worker cannot alter executable, module, `sitecustomize`, `.pth`, native extension or DLL/so import; (c) actual service security principal differs and IPC peer identity is kernel/token-authenticated, positive authorized client and negative same-principal/spoofed client; (d) process/service restart preserves durable nonce consumption; (e) even authenticated requester-supplied `physical_binding_valid` and TEST signatures cannot cause issuance. Record independent OS ACL/token/UID, hashes, module origins, IPC evidence and separate platform results without printing secrets. Mark **Layer O GREEN only after real protected positives and negatives**. Physical Layer P and original Load remain blocked.

**Narrow stage discipline:** B0/B1 change docs, a new Tests-only RED file, and a dedicated CI workflow ONLY. Do not modify `sd3_full_bf16_runner_authority.py`, `sd3_full_bf16_independent_runner.py`, replay ledger, Worker, Step Adapter, training math or CUDA. Keep PR #65 Draft/unmerged.

## Official platform references

- Linux: https://man7.org/linux/man-pages/man7/unix.7.html (`SO_PEERCRED`).
- Microsoft: https://learn.microsoft.com/en-us/windows/win32/api/namedpipeapi/nf-namedpipeapi-impersonatenamedpipeclient ; https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-openthreadtoken ; https://learn.microsoft.com/en-us/windows/win32/secauthz/access-tokens .
- Python 3.11 isolated mode: https://docs.python.org/3.11/using/cmdline.html#cmdoption-I .
