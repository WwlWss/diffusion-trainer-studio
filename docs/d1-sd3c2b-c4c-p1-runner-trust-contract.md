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
