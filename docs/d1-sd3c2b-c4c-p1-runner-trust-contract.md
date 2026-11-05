# D1-SD3 C4-C2-P1 — independent Runner trust root and provisional Fresh Step-1

**Stage:** source contract + Tests-only RED. **Frozen parent HEAD:**
\`6db33845b0e637501ab231eb4ca2f920850a0f3c\` (C4-C2 exact 12-CPU +
10-Host valid RED; C4-C1 10/10 GREEN). Parent PR #65 remains Draft and
unmerged. **P1 does not authorize Accelerator.load_state or worker_complete.**

## Precondition: there is no independent Full-BF16 Runner yet

\`tools/prepare_sd3_full_bf16_cases.py\` creates a Host manifest and tickets,
not a Runner. \`tools/sd3_full_bf16_evidence_worker.py:163–212\` binds
self-reported resume parent identity to actual Progress but does not prove
independent Runner acceptance. \`host._verify_resume_fresh_parent()\`
compares physical files to a supplied dict; that alone does not authenticate
the issuer. \`C3PhaseEvidence.finalize()\` deliberately raises, so the
Worker does not emit a completed Fresh phase. The proper intermediate
boundary is an original **Fresh Step-1 provisional physical candidate**
captured after a genuinely observed C4-B Step-1 Save, and rechecked by an
independent Runner; it is NOT an overall Phase pass.

## Source of trust — required separation

1. Runner is a **distinct controller/process** with a private signing key
   provisioned *outside* the Worker executable, inherited environment,
   repository, manifest, Ticket, phase directory or writable work_root.
   No Worker-selectable key path, public-key URI, issuer name or PID can
   replace the **pre-pinned** verifier trust root. Test-only keys are
   generated ephemerally and never qualify a CUDA parent.
2. Choose explicit Ed25519 (pinned mechanism dependency cryptography 44.0.2).
   Signature domain is bytes:
   \`DTS-C4C2-P1-ED25519-V1\x00\` followed by canonical, duplicate-free
   UTF-8 JSON of the **complete exact** signed payload; allow no extra,
   omitted, nonfinite, noncanonical, JSON-surrogate or alternate alias
   representation. An envelope includes:
   \`schema=dts.sd3c2b.c4c2p1.signed-provisional-step1-v1\`,
   \`payload\` and base64 \`signature_ed25519\`; it must not contain an
   issuer-selected public key or Worker-written trust setting.
3. Signed payload schema
   \`dts.sd3c2b.c4c2p1.provisional-step1-v1\`, version=1,
   \`proof_scope=provisional_fresh_step1_only\`,
   \`audience=dts.sd3c2b.c4c2.resume-load\`;
   fixed \`case_id,phase=fresh,logical_step=1,run_id,fresh_phase_id,
   resume_phase_id,source_commit,case_manifest_sha256,input_identity_sha256,
   reference_policy_sha256,reference_target_spec_sha256,
   resume_ticket_sha256,resume_ticket_nonce,checkpoint_id,checkpoint_dir,
   checkpoint_file_sha256\` (exact nine original names), separate
   \`adapter_path,adapter_file_sha256,physical_step_anchor_sha256,
   checkpoint_event_prefix_sha256,issued_at_unix,expires_at_unix,
   one_use_nonce\`. Fixed maximum lifetime ≤60 seconds; both time bounds
   and every identity must be checked.
4. The independent Runner must receive a preliminary Worker candidate and
   then observe/check **real** original Fresh Step-1 physical Save/event
   evidence: exact original nine files, 263 LoRA module pairs,
   526 trainable weights + 263 BF16 Alpha (=789 tensors), external LoRA
   original writer return and hash, genuine current progress/Checkpoint ID,
   original Source/Case/Run/Policy and physical event chronology. A Worker
   candidate, six hashes, an uploaded JSON or \`physical_binding_valid\`
   map **never** counts as this acceptance.
5. Runner must maintain one-use nonce registry/canonical token ID and
   audience/phase/ticket binding outside Worker-writable directories.
   Replay must be rejected across process restarts; a mutable in-process
   \`seen_nonces=set()\` proves only the test **mechanism**, not production
   durable replay protection. Verify before any original Load prehook.
6. To avoid a path-based TOCTOU, parent files must be immutable/protected
   from Worker modification between acceptance and original Load, or
   accessed through separately reviewed protected handles/copy-once
   staging with the **exact production path** contract preserved. Hashing
   before and after alone is not race-free. Revalidate nine-file and
   external LoRA digests at the final pre-Load boundary.
7. Accelerate 1.6.0 may swallow a restored RNG error and return normally.
   C4-C3 must independently compare actual post-load Python/NumPy/Torch/
   CUDA RNG state to the original pre-acceptance source, plus genuine
   BF16 Model/Alpha, Muon/AdamW Child Optimizer and Scheduler. C4-C4
   \`checkpoint_loaded\` and C4-A Resume Step-2 remain blocked until then.
8. Do not turn Worker \`fail\` into \`worker_complete\`, pretend a signed
   test payload is a CUDA parent, or accept a Worker-shared HMAC secret.
   A signature primitive is NOT a physical authority. A Runner-issued
   provisional candidate is NOT an authenticated full phase.

## P1 interface freeze, still absent at RED

A) \`tools.sd3_full_bf16_runner_trust.verify_provisional_step1_signature(
envelope=...,pinned_public_key=...,expected_context=...,now_unix=...,
seen_nonces=...)\`. This **pure Host cryptographic mechanism** MUST return
only \`{"status":"signature_valid_unqualified",
"physical_cuda_qualified":false,...}\` for a valid independently signed
TEST envelope. It rejects bad signature, arbitrary embedded trust root,
wrong exact schema/field set, wrong audience/source/case/step, expired,
premature or overly long window, and nonce replay. The \`pinned_public_key\`
argument is a **mechanism-test injection only**: the production
Runner-verification entrypoint must get its own actual pre-pinned trust
root, not a Worker or envelope input. Signing in a test with an
ephemeral TEST key is explicitly never Runner acceptance.

B) \`tools.sd3_full_bf16_independent_runner\` provides separate Runner-only
\`issue_provisional_fresh_step1_attestation\` (after independent physical
candidate verification), \`verify_original_provisional_step1_source\`
(original files+events+policy), \`get_pinned_runner_trust_root\`
(external trusted provisioning). No Worker import path is allowed to
mint a proof by passing a dict or string. **No production private key
shall be committed to repository or CI test environment.**

C) Later C4-C2-P2 requires independent actual issuer/executor provenance
binding, protected replay ledger and real CUDA **positive** accepted
Fresh Step-1 → original same-instance Load exactly once. The following
18 tests do not establish that positive condition. Do not claim
"C4-C2 full GREEN" merely because a verifier rejects arbitrary input.

## Exact RED test inventory

**Host Ubuntu + Windows:**
\`tests/test_sd3_full_bf16_worker_c4c_p1_host.py\`: 12 methods.
P101–103: genuine ephemeral Ed25519 and existing CPU/Host baseline
controls. P104: cryptographically valid TEST signature may return only
\`signature_valid_unqualified\`; P105–106 altered checkpoint/Signature
reject; P107 attacker-selected key cannot override pinned root;
P108–109 wrong audience, Case, Source, Ticket, lifetime reject;
P110 duplicate nonce rejects within mechanism registry; P111 phase,
Step, scope, nine-file-set reject; P112 extra envelope fields rejected.
No real CUDA qualification or Load authorization.

**Pinned real CPU:**
\`tests/test_sd3_full_bf16_worker_c4c_p1_cpu.py\`: six methods.
R101–103: require a distinct Runner issuer, real physical
provisional-source and externally pinned trust root (all missing at
RED); R104–105 original Worker has no signing secrets and current
C4-C1 Adapter still denies original Load before prehooks; R106 original
R1 CPU four-file genuine new-process Load remains a diagnostic baseline.

**Actions:** add pinned \`cryptography==44.0.2\` solely to minimal Host
test dependencies (Python 3.11; fixed test dependency), both workflow
path filters and precise 12 Host/6 CPU discovery gates. REQUIRE exactly
12+6 actual test executions, 0 ERROR/SKIP/xfail/xpass, all failures
explicit \`C4-C2-P1 HOST CONTRACT ABSENT\` / \`C4-C2-P1 CPU CONTRACT ABSENT\`.
Any unrelated failure, child-process crash or missing crypto dependency
is **INVALID RED**. Prior C4-A/B, FSR/G3, C4-C1 10/10 regressions must
remain PASS before the P1 CPU gate. Existing C4-C2 and C4-C full-stage
RED remain expected, intentionally not green.

## Completion and next phase

This commit is Tests-only RED. The subsequent P1 minimal GREEN must
implement real signature verification with both positive and negative
ephemeral TEST-key controls, and a reviewed distinct Runner source.
Even a P1 Host cryptographic GREEN may not issue genuine CUDA parent
authorization without a credible separate Runner and actual Fresh
Step-1 verification. Keep PR #65 Draft/unmerged, C4-C1 original
\`load_state\` delegation denied, C4-D/Fix D and \`worker_complete\`
fail-closed.


## P1-B0 source audit and P1-B1 dedicated RED freeze (after P1-A)

**Parent:** \`bfdaed52e435005c5c3e7250e735dd727595b3c1\`.
**Stage:** P1-B0 authority/issuer/replay design; P1-B1 **tests-only** contract.
P1-A Ed25519 Host 12/12 each OS stays the reviewed, **unqualified**
mechanism. The original P1 CPU R101/R102/R103 are valid intentional RED,
and C4-C2 Host/C4-C original Load authorization remains RED.

### Reviewed source and principal separation

- The Host generator \`tools/prepare_sd3_full_bf16_cases.py\` signs nothing;
  \`tools/sd3_full_bf16_evidence_worker.py\` has a Worker-owned Ticket and
  no private signing key. Worker Resume logic currently checks a claimed
  Fresh parent but does not validate an independent issuer.
- \`tools/sd3_full_bf16_runner_trust.py\` accepts a TEST-injected public key
  and an **in-memory** test nonce set; it grants only
  \`signature_valid_unqualified\`. Do not promote those parameters into
  the production admission API.
- \`C4StepAdapter._observed_load_state()\` is still a deliberately closed
  interceptor: no original Load delegation or pre-hook execution.
- A separately launched Python process **under the same OS account with
  read access to the key is not an independent signing authority**.
  Production must establish an actual OS security boundary (separate
  service account / equivalent protected signing service, key ACL and
  IPC authentication) and reject a launch lacking that boundary. GitHub
  CI runs both test processes under the same account; their tests prove
  only the mechanism, never the production OS isolation.

### Frozen P1-B interfaces and privileges

1. A standalone executable
   \`python -m tools.sd3_full_bf16_independent_runner\` must support the
   read-only \`--probe-unprovisioned\` command. Without externally
   controlled provisioning it reports JSON schema
   \`dts.sd3c2b.c4c2p1.runner-probe-v1\`,
   \`status=unprovisioned\`, \`may_issue=false\`,
   \`physical_cuda_qualified=false\`. It exposes no private key or
   signing configuration in stdout/stderr. Inherited Worker environment
   variables cannot provision a production signer.
2. The Runner module must expose the previously frozen
   \`issue_provisional_fresh_step1_attestation\`,
   \`verify_original_provisional_step1_source\`, and
   \`get_pinned_runner_trust_root\`. The latter resolves its actual key
   exclusively from an independently configured, OS-protected root;
   Worker-passed public keys, issuer labels, file paths, env overrides and
   self-declared PID cannot select it. Missing provisioning **raises**;
   there is no default test/embedded signer. The issue function MUST
   reject an arbitrary Worker dict; its real successful physical-source
   precondition remains for P1-C and MUST NOT be stubbed into PASS.
3. The non-authorizing pure Host durable replay mechanism is
   \`tools.sd3_full_bf16_runner_replay.RunnerNonceLedger(db_path)\`.
   This is **mechanism-only**; caller-selectable test paths are never a
   production trust root. Its context-managed \`consume_once\` accepts
   only keyword arguments \`issuer_id, audience, ticket_sha256,
   one_use_nonce, token_id, issued_at_unix, expires_at_unix, now_unix\`.
   First valid claim returns
   \`{"status":"nonce_consumed_unqualified",
   "physical_cuda_qualified":false}\`; duplicate globally unique
   nonce (even with different Ticket, audience, token ID or process)
   raises \`ValueError\`. Invalid times, token IDs, corrupted ledger and
   unsafe path aliases fail closed.
4. Replay claims must be a single committed atomic durable write with
   transaction/unique constraint and bounded lock wait; under concurrent
   processes **exactly one** claim succeeds. No in-memory set, appended
   JSON, check-then-insert race, silent sqlite-recreate-on-corruption,
   delete-on-expiry reuse, or best-effort logging may qualify. Persist
   claims across restart and retain until an independently reviewed
   lifecycle cleanup, not per-token expiry.
5. The actual production Runner ledger and pre-pinned public key are
   selected by a protected controller **outside Worker-writable
   work_root**. An injected \`RunnerNonceLedger(db_path)\` proves only a
   durable **test primitive**. Never pass an arbitrary instance of it
   to \`C4StepAdapter\` as Load authority. The production admission path
   must authenticate the issuer, canonical signed token, exact expected
   context and perform the same durable atomic one-use claim before any
   original Load pre-hook.
6. Evidence issuance depends on a Runner-owned, independently verified
   original Fresh Step-1 **physical** capability. P1-B may create the
   protected issuer lifecycle but MUST NOT sign a self-declared Worker
   candidate, and cannot synthesize a CUDA positive acceptance. The
   protected candidate origin, 9-file/789-Tensor/LoRA/event physical
   checks and race-safe staging are explicitly P1-C; genuine accepted
   original Load is later C4-C2-P2. Test-only ephemeral key material is
   never production-provisioned, even if cryptographic verification
   succeeds.

### P1-B1 new exact Host RED and later GREEN requirements

**Dedicated test file:**
\`tests/test_sd3_full_bf16_worker_c4c_p1b_host.py\`, exactly **12**
pure Host cases on **Ubuntu and Windows**:

- B101–B103: three existing-P1-A / Worker boundary baseline PASS.
- B104: independent Runner process probe with no provisioning.
- B105: no Worker-supplied trust root or in-process default root.
- B106: ordinary Worker candidate cannot mint a Runner signature.
- B107: first ledger claim mechanism returns only unqualified and
  immediate replay fails.
- B108: replay is rejected after reopening in a second OS process.
- B109: concurrent process contenders yield exactly one committed claim.
- B110: changing Ticket/audience/token ID does not reset one-use nonce.
- B111: existing corrupted ledger never silently becomes fresh.
- B112: malicious inherited signing-key environment cannot activate
  the unprovisioned Runner or leak the supplied marker.

**Expected Tests-only RED at this parent:** 12 discovered and executed,
**9 targeted** \`C4-C2-P1-B HOST CONTRACT ABSENT\` FAIL, **3 baseline**
PASS on *each* OS, 0 ERROR/SKIP/xfail/xpass. A targeted failure must
come from a missing Runner module or replay-ledger module, **not** a
failed subprocess, ImportError or missing dependency. The stand-alone
\`.github/workflows/sd3-c2b0-p1b-red.yml\` enforces exact discovery and
valid targeted RED; it is not a CUDA/release workflow. P1-B CPU R101–R103
remain intentionally RED until the reviewed Runner sources exist.

**Stage guards:** P1-B0/B1 change only the contract, tests and
test CI. No production code, Worker's invocation, Adapter Load,
checkpoint_loaded, worker_complete, training math, full CUDA eligibility
or release table are altered. After valid RED, proceed to small,
reviewed P1-B GREEN slices: (A) protected Runner launch/root read
fail-closed, (B) atomic durable ledger, (C) independently provisioned
issuer lifecycle with no physical admission, then independent Strict
Review. Full P1-C positive physical candidate acceptance remains HOLD.
