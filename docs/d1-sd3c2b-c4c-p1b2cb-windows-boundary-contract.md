# DTS C4-C2-P1-B2-C-B2-B: Windows native token / connected named-pipe boundary

**Parent:** `4d0ef4e032c16e051cc8376b0e52ad204df4090c`; PR #65 Draft/unmerged. **Stage:** B2-B0 source contract, B2-B1 Windows Tests-only RED, B2-B2 Win32 process-token / connected named-pipe observation, B2-B3 ACL/startup and service denial, B2-B4 independent native spoof/cleanup Strict Review. Windows **Host native observations and negative tests only**, NEVER protected OS Layer O qualification.

## 1. Roles and source separation

*The untrusted Worker* can supply named-pipe names, claimed SIDs/PIDs, Python objects, environment variables, executable paths, fake DACLs, Test Ed25519 keys, Host sqlite ledgers and claimed physical parent data. None can select a protected service identity or grant signing. A different Windows PID is not a new security principal.

B2-B creates **only** standalone `tools.sd3_full_bf16_runner_windows_boundary` and `tools.sd3_full_bf16_runner_windows_service`. The generic `tools.sd3_full_bf16_runner_os_boundary` stays **ABSENT** until the Linux/Windows integration stage; B1 O304–O312 continue the intentional nine RED/three PASS on each OS. Existing `tools.sd3_full_bf16_runner_authority`, independent Runner, replay, Worker, Step Adapter, original `Accelerator.load_state`, CUDA gates and Phase Finalize MUST NOT CHANGE.

No private signer key, external root, protected ledger, named-pipe listener or legitimate Windows service account is provisioned in this Host slice. A Windows CI test that creates a same-user named pipe is NOT evidence of a protected service. All reported statuses remain `os_security_qualified=false`, `may_issue=false`, `physical_cuda_qualified=false`.

## 2. Native APIs and non-authorizing observations

### Win32 process token

`observe_windows_process_token_user()` obtains a real process token via `OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY)` and determines its SID using `GetTokenInformation(TokenUser)` and `ConvertSidToStringSidW`. Return a frozen `WindowsTokenObservation` containing only a canonical SID text and an explicit unqualified provenance label. Close token and SID-string native allocations in `finally`. No caller-supplied SID/PID/public key accepted.

### Connected named-pipe client token

`observe_windows_named_pipe_client_token(server_pipe_handle: int)` requires an exact native integer handle to a **server-side connected** named-pipe instance on Windows. Reject integers equal to 0/INVALID_HANDLE_VALUE, object spoofers, arbitrary non-pipe handles, stale handles, and unconnected pipe instances.

**Critical ordering:** A server must *first read a bounded client request* from that connected named pipe; then `ImpersonateNamedPipeClient` uses the identity of the **last message read**, not an arbitrary caller's claimed identity. If impersonation returns false, immediately fail closed; do not query the privileged process token or process the request. Upon successful impersonation, use `OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, OpenAsSelf=True)` and `GetTokenInformation(TokenUser)` to derive the client SID. `RevertToSelf` runs in a guaranteed `finally` and token handles are closed. Failure to restore impersonation is a fatal no-continue condition for any future privileged service; do not proceed as a protected authority. A caller-supplied SID, pipe name or `GetNamedPipeClientProcessId` can never substitute for actual token observation.

`require_windows_authenticated_peer(handle, **worker_claims)` is **deny-only** at B2-B: it may call the genuine observer but always raises `ValueError` with or without SID distinction. No positive policy, trusted service capability, signature or Load admission exists.

### Native tests without an actual privileged service

Windows Host CI may create a **test-only same-user local named pipe** via `CreateNamedPipeW` with `PIPE_REJECT_REMOTE_CLIENTS`, connect with `CreateFileW` and client impersonation QoS, write one short request, read it on the server via `ReadFile`, then call the observer and verify the actual client token SID equals the current process token SID. This proves that the observer truly reaches Windows kernel/LSA APIs, NOT that Worker cannot access a protected service.

Test false handles and lack of prior client read must fail safely, and a same-SID genuine pipe may never obtain authorization. Avoid a fabricated positive owner/ACL result or an injected/mock user SID. Test-only pipe handles require reliable `CloseHandle` cleanup; no mock can claim actual positive OS security.

## 3. Startup, file integrity and negative service

`require_windows_protected_service_preflight()` and `require_windows_protected_credentials()` must **always refuse** while there is no independently administered SCM/service identity, immutable deployment, protected key/root and ledger. `require_windows_sanitized_service_startup()` checks known Python/DLL/inherited key variables then still rejects even when clean. Never treat clearing Python variables as proof of SCM ownership.

`audit_windows_protected_code_path(path)` rejects Worker-owned temporary paths, relative aliases, reparse points, and unsafe file shape. It is explicitly an **unqualified diagnostic only**; file existence, ordinary `Path.stat`, a claimed owner or a DACL string do not prove effective ACL, tamper resistance or TOCTOU-safe source installation. Future Layer O needs an independently pinned install, real Windows access checks, effective service-token/TokenUser SID, impersonated peer DACL checks, protected ancestors, native DLL origin and real cross-account negative attempts.

`tools.sd3_full_bf16_runner_windows_service`:
- `--probe-unprovisioned` returns only fixed `{"schema":"dts.sd3c2b.c4c2b2b.windows-service-v1","status":"unprovisioned","os_security_qualified":false,"may_issue":false,"physical_cuda_qualified":false}` without opening a pipe.
- `--serve` immediately calls the fail-closed protected preflight before constructing/creating any named-pipe listener, initializing a key, reading worker-supplied paths, importing signer code or opening a ledger. It exits nonzero and does not listen.
- `deny_windows_diagnostic_request(payload)` is a bounded, test-only, non-listening formatter: exact `{"op":"status"}\n` may return an unqualified status; all other input returns `denied`. Any request longer than 256 bytes or with extra JSON keys is denied. No echo/secret leakage or sign/checkpoint operation.

## 4. Stages and exact RED gate

`tests/test_sd3_full_bf16_worker_c4c_p1b2cb_windows_host.py`: 12 unconditional Windows-only tests; `.github/workflows/sd3-c2b0-p1b2cb-windows.yml` runs **windows-latest** and Python 3.11. No installation of signing keys or protected service accounts.

- W101–W103: 3 existing baselines PASS (generic B1 module absent; existing production/root fail closed; test ledger remains unqualified).
- W104: real `TokenUser` read from current process, stable SID, no authority.
- W105: Worker-provided fake user SID/PID cannot affect native process token observation.
- W106: real connected named pipe, prior `ReadFile`, and `ImpersonateNamedPipeClient/OpenThreadToken/TokenUser` produce genuine same-user client SID.
- W107: same-SID real pipe cannot authorize a Runner peer.
- W108: bad/fabricated/unconnected pipe handle fails closed.
- W109: contaminated and clean inherited env cannot authorize startup.
- W110: Worker-controlled file/reparse source path cannot prove protected DLL/import ACL.
- W111: Worker key/root/ledger paths or fake DACL cannot qualify credentials.
- W112: diagnostic-only CLI, bounded denied replies, `--serve` refuses without creating a listener.

**Frozen B1 RED:** 12 discovered/run, W104–W112 exactly **9 targeted `C4-C2-P1-B2-C-B2-B WINDOWS CONTRACT ABSENT` FAIL**, W101–W103 3 PASS, 0 ERROR/SKIP/xfail/xpass. Failures must stem from missing Windows source module, not a subprocess, import, dependency or Win32 error. Exact named method IDs and marker checked in CI. RED job intentionally exits failure. B2-B2/B3 source must turn them 12/12 Host PASS without changing generic B1 RED. Any native API compatibility failure remains genuine failure, not a SKIP.

**A5 independent Strict Review:** after initial Host GREEN, add separate targeted adversarial tests for stale handles, no-message impersonation, malicious ctypes/Python handle wrappers, `RevertToSelf` on all successful impersonation branches, output leakage, and service listen-before-preflight. Freeze valid targeted RED before fixes, then strict GREEN and regression.

**HOLD:** Windows independent protected service principal, actual SCM startup, trustworthy code/dependency/DLL tree, verified protected DACL/key/root/ledger and true different-SID connected positive/negative tests (Layer O), cross-platform aggregation, P1-C original nine-file/789 BF16 physical Fresh Step-1, original Load/Resume/CUDA R1/R2 and merge. Keep PR #65 Draft.

### Official Win32 contracts

- https://learn.microsoft.com/en-us/windows/win32/api/namedpipeapi/nf-namedpipeapi-impersonatenamedpipeclient
- https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-openthreadtoken
- https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-gettokeninformation
- https://learn.microsoft.com/en-us/windows/win32/api/namedpipeapi/nf-namedpipeapi-createnamedpipew
- https://learn.microsoft.com/en-us/windows/win32/secauthz/impersonation-levels
