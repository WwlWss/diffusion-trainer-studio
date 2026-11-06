# DTS C4-C2-P1-B2-C-B2-A: Linux native boundary and read-only service

**Reviewed parent HEAD:** `21ac5b7be3229cbc17e6a736ef2eea1d4497b1b5`; PR #65 stays Draft/unmerged. **Current stage:** Linux A0/A1/A2/A3/A4 implemented; A5 independent Strict Review pending. This is Linux **Scoped Host** native-kernel/negative GREEN, not a completed privileged service. Protected OS Layer O and Fresh CUDA physical Layer P remain HOLD.

## Scope separation (MUST NOT blur)

- A Linux **connected AF_UNIX stream socket** allows `getsockopt(SOL_SOCKET, SO_PEERCRED)` to read the real kernel-issued PID/UID/GID recorded at connection/socketpair time (Linux unix(7)). These are observations, **not** a signature right, immutable peer identity over time, or evidence that a Python process runs under an externally administered service manager.
- A physically different PID, spoofed UID/GID data, `socketpair` under the same OS user, a root-looking `uid=0` argument and module-global trust flags must not grant authority. Even real UID/GID observations require an independently pinned server-side policy and **a real protected OS principal** before they can contribute to positive admission.
- The standalone Linux module `tools.sd3_full_bf16_runner_linux_boundary` is the only new source in A2/A3. **Do not create** the generic `tools.sd3_full_bf16_runner_os_boundary`: the existing B1 cross-platform O304–O312 deliberate nine REDs must survive until Linux+Windows integration.
- The standalone Linux service `tools.sd3_full_bf16_runner_linux_service` is a **read-only denied/unprovisioned skeleton**, never a signing daemon. Its `--probe-unprovisioned` may emit a constant public diagnostic. `--serve` MUST FAIL CLOSED in this stage if no externally administered protected service configuration exists. A test-facing bounded diagnostic handler may consume connected AF_UNIX requests, but every reply must remain `may_issue=false` and `os_security_qualified=false`. No signing, private keys, generated roots, nonce acceptance, original Load, or CUDA evidence.
- The existing `tools/sd3_full_bf16_runner_authority.py`, `tools/sd3_full_bf16_independent_runner.py`, `tools/sd3_full_bf16_runner_replay.py`, Worker and Step Adapter must not change for B2-A. All original Load calls stay rejected before prehooks; no `checkpoint_loaded`/`worker_complete`.
- Host CI `ubuntu-latest` only can prove Linux **native observed** SO_PEERCRED and same-UID denial. It **cannot** establish a protected independent service account. B2-A Host Scoped GREEN does NOT imply Layer O GREEN.

## A1 exact Linux Tests-only RED → A2/A3/A4 GREEN

Dedicated `tests/test_sd3_full_bf16_worker_c4c_p1b2cb_linux_host.py`: **12 unconditional** tests (no skip/xfail) on Ubuntu only.

- L101–L103: 3 existing baselines PASS: B1 missing cross-platform OS integration module, independent authority/Runner fail-closed, and replay primitive unqualified.
- L104: real connected `socket.socketpair(AF_UNIX, SOCK_STREAM)` yields actual kernel PID/effective UID/effective GID in a frozen **non-authorizing** observer.
- L105: forged `peer_uid`/`peer_pid` arguments cannot override real kernel output (observer does not accept identity kwargs).
- L106: wrong socket family, wrong stream type, unconnected stream or non-socket must raise `ValueError`.
- L107: same-current-UID connected AF_UNIX pair cannot become authorized by self-claimed different PID/UID.
- L108: without independent service-manager policy, protected principal authorization refuses even honest kernel observations.
- L109: inherited Worker-controlled Python/dynamic loader variables are rejected before any protected service startup.
- L110: Worker-writable code, symlinks and writable parent directories are rejected; any clean path audit is diagnostic-only and does not prove independent install ownership, ACL or TOCTOU safety.
- L111: Worker-chosen signing key/root/ledger/config paths cannot become trusted service credentials.
- L112: Linux standalone service `--probe-unprovisioned` and public fixed-size socket diagnostic never claim service/CUDA qualification; `--serve` cannot start a listener without controller provisioning.

**Exact A1 RED expectation at parent**: 12 discovered/run = **9 targeted FAIL L104–L112 +3 baseline PASS**, zero ERROR/SKIP/xfail/xpass. Missing Linux module/service contracts cause only tagged `C4-C2-P1-B2-C-B2-A LINUX CONTRACT ABSENT` assertions (not import errors). Scope source acceptance by explicit test identities in independent `.github/workflows/sd3-c2b0-p1b2cb-linux.yml`. After implementing A2/A3/A4 and independent review, exactly **12/12 Host PASS**, never a forged privileged positive.

## A2: Linux kernel credential observation

Implement dataclass `LinuxPeerCredentials(pid,uid,gid)` (frozen) and `read_linux_connected_peer_credentials(connected_socket)`; it must guard `sys.platform.startswith("linux")`, accept only native Python socket objects with `AF_UNIX` and stream socket type, require `getpeername()` to succeed (for socketpair, empty address is valid), fetch exactly 12 bytes via real `SO_PEERCRED`, unpack `struct.Struct("=3i")`, check positive PID, nonnegative UID/GID and refuse `OSError`/short response as `ValueError`. Never accept caller-provided claimed peer identity, kernel credential dict or alternate socket family. This is a diagnostic result, not a protected service capability.

Expose `observe_linux_service_identity()` that obtains actual `os.getresuid/getresgid/getgroups` from kernel (no claimed UID/PID) and returns non-authorizing immutable identity.

`require_linux_authenticated_peer(connected_socket, **worker_claims)`: read true credentials and reject unauthorized current-account peers; until independently provisioned protected server policy exists, **all** positive permissions refuse with `ValueError`.

## A3: executable/startup/source diagnostics

Implement `audit_linux_protected_code_path(path)` that inspects the actual filesystem path components without following symbolic links and rejects group/other-writable files/parents, symlink/reparse aliases and non-regular code files. This **preflight scan is diagnostic-only**; it cannot prove race-free open, ACL semantics or private-key provenance and must never return an authority capability. Do not import from Worker-supplied code paths when checking them.

Implement `require_linux_sanitized_service_startup()` fail-closed if any inherited Python startup/import or dynamic-loader contamination is present, and **also** fail-closed when no protected independently configured service manager/identity and trusted installed executable inventory exists. No Worker-supplied config or test-only environment string may enable positive launch.

Root-owned service installation and protected working directories must be performed by an independent OS controller. The eventual Layer O service needs a separately reviewed OS-supervised install under a distinct UID (with protected code, key/root, ledger and AF_UNIX socket); `python -I` is defense-in-depth only and cannot establish independent authority.

## A4: minimal non-authorizing service shell

Standalone `tools.sd3_full_bf16_runner_linux_service`:
- `python -m ... --probe-unprovisioned`: constant JSON, no filesystem writes, independent of env/claimed credentials.
- `python -m ... --serve`: invokes protected startup validation **before bind/listen or directory creation**; while no protected controller exists, exits nonzero and does not create socket/listen.
- A test-invokable `handle_linux_diagnostic_connection(conn)` accepts only real connected AF_UNIX stream sockets and observes the kernel peer before reading. Read a short bounded single-line request; for exact `{"op":"status"}\n` return a fixed `unprovisioned` response. For `sign`, `load_checkpoint`, `authorize_resume` and all other inputs return a fixed `denied` response with the same `may_issue=false/os_security_qualified=false/physical_cuda_qualified=false` fields. No arbitrary JSON keys, echo, file access, signer call, Worker-supplied trust or positive authenticated status.
- Restrict request length/time and always close the passed connection after response; error paths must not echo raw bytes. This handler is **NOT** a service deployment; Host socketpair exercises only negative semantics.

## A5 and later

Independently inspect exact committed diff, parser limits, genuine `SO_PEERCRED` usage, exception fail-closed, path audit limitations and absence of proxy to signer. Run Linux A2 12/12 Host GREEN; original cross-platform B1 O301–O312 remains 9 deliberate FAIL +3 PASS on both OS, prior A-SR1/B Host GREEN, C1–C3 unchanged, original C4-C2 Host/CPU intentional RED unchanged. Do not treat main RED as regression. Keep PR #65 Draft.

**Layer O real protected integration remains HOLD:** independently administered Linux service manager and service UID, immutably installed code/runtime, actual kernel-authenticated positive and negative cross-user peers, service-owned private key/verifier root/replay paths and directories, tested restart/race behaviors, plus Windows B2-B before cross-platform qualification. Real physical Fresh Step-1 nine-file/789 BF16 provenance remains P1-C HOLD.

Linux official docs: https://man7.org/linux/man-pages/man7/unix.7.html

## B2-A staged source delivery and negative-only Linux deploy example

- A0 contract: `79aec1648d3c580bff6ddd430a3b240affa8773f`.
- A1 tests: `9b739315ab263edb9d2d0b6d8772ad3edad7ef0d`, CI `5338bf8950647ec3d049cae614f151f896a2987c`. [Actual A1 RED #42185295428](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42185295428): 12 discovered/run, exactly L104-L112 nine tagged FAIL plus L101-L103 three PASS; zero ERROR/SKIP/xfail/xpass.
- A2/A3 source: `f67bc35ffa690a5d525101af5ac5a0783226f0c0` (`tools/sd3_full_bf16_runner_linux_boundary.py`), real connected `SO_PEERCRED` and `getresuid/getresgid` observations, conservative root-owned code-shape negative audit, no provisioned service principal or credentials.
- A4 service shell: `461ee9f906569c1bd41f92a678e47cc054252aaf` (`tools/sd3_full_bf16_runner_linux_service.py`); fixed `--probe-unprovisioned` diagnostic, bounded `handle_linux_diagnostic_connection`, `--serve` **deliberately DENIES before listener creation**. [A2–A4 Host #42185600530](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42185600530): Linux 12/12 PASS, zero FAIL/ERROR/SKIP/xfail/xpass. No protected service has been started.

### Controller-owned systemd configuration illustration (NOT DEPLOYABLE YET)

The following illustrates *future* out-of-band service-manager controls and must
not be installed as an already qualified production unit. **The current service
intentionally exits with status 2 on `--serve`, and it never creates a
listening socket.** A real independently owned install, immutable Python
package with native dependencies, access policy, signed trust-root provisioning,
service-private ledger and key protection need separate B3 tests and review.
The `-I` flag removes the repository cwd from Python's usual import path,
so the trusted package **must actually be installed** in the protected venv.

```ini
[Unit]
Description=DTS independent read-only Runner (future protected deployment)
After=local-fs.target

[Service]
Type=exec
User=dts-runner
Group=dts-runner
WorkingDirectory=/opt/dts-runner
ExecStart=/opt/dts-runner/venv/bin/python3 -I -m tools.sd3_full_bf16_runner_linux_service --serve
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
ProtectKernelTunables=yes
ProtectControlGroups=yes
RestrictAddressFamilies=AF_UNIX
RuntimeDirectory=dts-runner
RuntimeDirectoryMode=0700
StateDirectory=dts-runner
StateDirectoryMode=0700
UMask=0077

[Install]
WantedBy=multi-user.target
```

These flags are **defense in depth**, not a proof of principal separation.
An independently managed controller must install service-owned/root-owned
interpreter, code, all ancestors and hash-pinned dependencies in a trusted
location; do not execute from a Worker-writable checkout. The `RuntimeDirectory`
mode above intentionally exposes **no Worker client access**. Later access
requires a separately reviewed authenticated peer policy and socket ACL/mode
design, not merely chmod on a socket after binding.

### Unqualified diagnostics versus protected B3 acceptance

- The code-path audit is a conservative `lstat` chain check: it rejects
  symlinks and writable components. **It is not a TOCTOU-proof handle-based
  open, a complete ACL/mount-namespace verification or a way to qualify a
  privileged service.** Passing this audit must never be used as a signing
  precondition without new OS integration review.
- A `SO_PEERCRED` snapshot is authentic for an actual socket but does
  not grant a service identity or attest the origin of arbitrary request bytes.
  `require_linux_authenticated_peer` still **unconditionally denies**.
- Both `require_linux_protected_service_preflight` and
  `require_linux_protected_credentials` unconditionally refuse.
  `require_linux_sanitized_service_startup` may reject contaminated env
  earlier, but even a clean env does **not** establish service-manager origin.
- The test handler permits only a public fixed unprovisioned/denied response,
  with `may_issue=false` / `os_security_qualified=false`; it does not
  listen, authenticate an OS service or invoke any signer.
- True B3 Layer O acceptance requires an independently created service account,
  controller-protected install, non-Worker-readable private key and non-writable
  pinned public root/ledger; actual distinct-UID positive and negative AF_UNIX
  connections; and separate Windows B2-B. None of these has been witnessed
  by normal GitHub same-user CI. Original `Accelerator.load_state`, P1-C
  genuine nine-file/789-BF16 Fresh Step-1, `checkpoint_loaded`,
  `worker_complete` and full CUDA qualification remain HOLD.
