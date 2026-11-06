# DTS QG-1A — pure structured ExecutionReadiness source contract

**Reviewed base:** `main@1dc4f56fb862aade574add48c4f1bea2c54f3e96`. **QG-0 source registry:** [114 gates](https://github.com/WwlWss/diffusion-trainer-studio/blob/docs/qualification-gate-architecture-review/docs/qualification-gate-qg0-inventory-2026-11.md) lives in Draft [PR #66](https://github.com/WwlWss/diffusion-trainer-studio/pull/66), deliberately not merged into this branch. **QG-1A1 valid Tests-only RED has advanced to QG-1A2 scoped pure-module GREEN**; neither stage modifies production train/Start permission or CUDA qualification.

## 1. Responsibility and threat boundary

The new module **`mikazuki.parameter_policy_readiness`** is a *host-only, torch-free, I/O-free, deterministic* reducer from **independently established capability facts** + typed hard/warning findings + **read-only evidence tier** to a diagnostic decision.

- **Capability** answers whether a concrete backend/optimizer/feature is implemented and can be evaluated. No qualification row is permitted to claim an implementation exists.
- **Runtime decision** answers whether validated execution may proceed. Do not confuse this diagnostic with actually creating a task. QG-1A is not wired to `/run` or a trainer.
- **Evidence tier** answers what verification has actually been recorded. It cannot grant runtime capabilities and the reducer **must not** sign, mint, infer an external Runner identity, mutate `FULL_BF16_*_QUALIFICATIONS`, or write a Checkpoint.
- **Actual runtime invariants** remain entirely enforced by existing `parameter_policy_runtime_bools.py`, `parameter_policy_runtime.py`, `parameter_policy_trainer.py`, `parameter_policy_compat.py` and SD3 Resume hooks; QG-1A cannot bypass or replace them.
- **PR #65** remains Draft/held independent formally attested evidence. It is not an import or a required step of normal SD3 LoRA Save/Resume.

No application entrypoint or existing blocking behavior is changed in QG-1A. All existing qualification tables and legacy gates remain untouched for this first RED→GREEN proof of the pure layer. QG-1B is responsible for an independently reviewed transition from legacy `_qualification_blocker` into structured warning/capability evaluation.

## 2. Public Python ABI (freeze)

New module `mikazuki/parameter_policy_readiness.py` must export exactly these primary entrypoints (additional private helpers permitted).

### `ReadinessIssue`

Immutable `@dataclass(frozen=True, slots=True)`:
- `code: str`: stable lowercase dotted identifier, e.g. `runtime.bool.invalid`, `capability.not_implemented`, `qualification.backend.pending`; no empty or arbitrary numeric/whitespace code.
- `message: str`: human-readable nonempty diagnostic. Never use its text as the authorization key.
- `source: str`: stable upstream gate ID such as `E02`, `R03`, `C01`, `G01`, or future explicitly named source. Must be a nonempty identifier.

The reducer takes **separate** hard and warning lists of `ReadinessIssue`. A code cannot occur multiple times or in both categories; conflicting classifications must raise `ValueError` rather than silently preferring a dangerous branch.

### `ExecutionReadiness`

Immutable `@dataclass(frozen=True, slots=True)`:
- `decision: Literal["allow","warn","block"]`
- `capability: Literal["implemented","experimental_candidate","not_implemented","known_incompatible"]`
- `hard_errors: tuple[ReadinessIssue, ...]`
- `warnings: tuple[ReadinessIssue, ...]`
- `evidence_tier: Literal["formally_qualified","observed_real_gpu","host_cpu_checked","unverified","failed"]`
- Read-only property `can_start` evaluates to `True` for allow/warn, `False` for block. This is a **pure recommendation**, not authorization to launch or issue proof.
- Method `to_diagnostics() -> dict` returns only JSON-safe snapshot with the five fields `decision`, `capability`, `hard_errors` (array of {code,message,source}), `warnings` (same) and `evidence_tier`. Return fresh mutable objects each time; never insert these into stable execution/checkpoint identity. **No** signer, `physical_cuda_qualified`, `may_issue`, `worker_complete`, `certificate` or `evidence_case_id` fields exist in this API.

An attempt to directly construct an inconsistent `ExecutionReadiness` (e.g. `decision="allow"` with hard errors, or `can_start=True` when unsupported) must raise `ValueError`. Inputs/outputs have strict types and use defensive tuple copies; no reference to mutable caller lists remains.

### `evaluate_execution_readiness`

Pure function:

```python
def evaluate_execution_readiness(
    *,
    capability: str,
    hard_errors: Sequence[ReadinessIssue] = (),
    warnings: Sequence[ReadinessIssue] = (),
    evidence_tier: str = "unverified",
) -> ExecutionReadiness:
    ...
```

Behavior:
1. Reject unknown capability/evidence values, malformed issue lists, invalid issue objects or duplicate codes with `ValueError`. Do not default unknown capability to supported.
2. `not_implemented` and `known_incompatible` **always BLOCK** even if input says `formally_qualified`. Supply a stable capability-related hard error when caller did not provide one: `capability.not_implemented` or `capability.known_incompatible` (`source="G01"`).
3. Any actual `hard_errors` **always BLOCK**, irrespective of capability and evidence.
4. `experimental_candidate` that has no hard errors returns `WARN` and `can_start=True`; supply stable `capability.experimental` warning (`source="E02"`) if no explicit warning was supplied.
5. `implemented` with explicit warnings returns `WARN` and `can_start=True`. `implemented` with no errors/warnings returns `ALLOW`; **evidence tier alone does not modify decision or capability**.
6. `evidence_tier="failed"` is an evidence observation, not by itself a proof of a runtime defect. If a test demonstrates a true incompatible implementation, caller must explicitly provide `known_incompatible` or a typed hard error. A failed certificate cannot make a known broken feature executable nor automatically mark an independent User job failed.
7. Preserve input diagnostic order and copy mutable input lists. The reducer must remain importable without `torch`, `accelerate`, `safetensors`, model weights, OS APIs, network or file access; no global state, timestamps, PIDs, env probes or I/O.

**Official qualification safety:** `formally_qualified` is a *read-only status field provided by an upstream verified evidence registry*. This pure reducer **cannot establish that provenance**; it must never mint or promote formal qualification, sign a result, or infer it from an untrusted Worker map. An external signature authority is not part of local training.

## 3. Tests-only RED criteria

New `tests/test_parameter_policy_qg1a_readiness.py`:

- **QG101–QG112: exactly 12 independent targeted tests**. Each fails with the exact marker `QG-1A PURE READINESS CONTRACT ABSENT` while new module is missing. Once source is added, the tests exercise genuine semantics (not mere function existence).
- **B901–B904: four existing-code baselines PASS** at frozen `main`: stable BF16 execution identity, invalid precision, existing strict pseudo-boolean rejection and planned optimizer refusal. Their purpose is to show initial RED is only the deliberate absence of the *new* pure module.
- Both Windows and Ubuntu Python 3.11 CI: exactly 16 discovered/executed, 12 target FAIL +4 baseline PASS, zero ERROR/SKIP/xfail/xpass; exact case IDs and marker in all 12 failures. Any other failure shape invalidates the RED. The expected failure is a **real failing CI step** in B1, not an `expectedFailure` or fake GREEN.
- Freeze source-only `RED` before creating `mikazuki/parameter_policy_readiness.py`. QG-1A minimum GREEN is a later, separately reviewed stage.
- No modifications to `mikazuki/parameter_policy_execution.py`, release qualification tables, `parameter_policy.py`, `training_request.py`, `training_api.py`, `parameter_policy_trainer.py`, `full_bf16` flags, legacy tests or PR #65.

### Semantic acceptance obligations for later GREEN

| Case | Input capability / evidence | Expected | Why |
| --- | --- | --- | --- |
| QG103 | implemented / formally_qualified, no issues | allow | Ordinary supported run |
| QG104 | implemented / unverified vs formally_qualified, same execution contract | **identical allow** and immutable checkpoint identity | Certificate is not an implementation prerequisite |
| QG105 | experimental_candidate / unverified | warn; can_start | Explicitly signaled experimental scope |
| QG106 | implemented + hard R03; formal evidence | block | Real runtime failure wins |
| QG107 | not_implemented or known_incompatible; formal evidence | block | Certificate cannot invent runtime support |
| QG108 | implemented + advisory E02; unverified | warn, not block | Show missing qualification as warning only |
| QG109 | failed evidence, without proven runtime fault | runtime result from capability + issues only | Do not equate evidence failure and runtime incompatibility |
| QG110 | malformed enums/issues/duplicated codes | ValueError | Fail safe, no ambiguous bucket |
| QG111 | frozen inputs and snapshots | no mutation | Checkpoint / UI ABI isolation |
| QG112 | pure module import, no credential or signer fields | no side effects | Host/CPU-only decision, never physical CUDA proof |

## 4. Implementation sequence and stop line

- **QG-1A1**: Source contract document + 16-case Tests-only RED and CI, with exact check counts.
- **QG-1A2**: Minimum pure reducer GREEN, no existing Product gate wired and no new CUDA permission.
- **QG-1A3**: Independent Strict Review of the new module and its JSON/immutability/no-proof ABI.
- **QG-1B** only then migrates **E** certificate metadata away from fatal gating; **QG-2** Preview/Start/Direct parity and experimental UX opt-in; **QG-3** actual SD3 Full-BF16 production capability; **QG-4** physical GPU training/Resume. These phases may not be represented as completed by QG-1A RED.

**HOLD:** Running an unqualified training configuration, any user-visible eligibility change, any physical CUDA qualification, replacing PR #65 original independent Runner Load, protected OS B3, merge/promotion. PR remains Draft until actual implementation and review.

## 5. QG-1A2 source and GREEN evidence

**Baseline exact-head RED:** `40178f29fe5a55d5eb64d17a6cc957c482de5bae`,
[PR + push RED logs](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42264884920):
Ubuntu and Windows each **12 targeted FAIL + 4 baseline PASS**, 0 ERROR/SKIP/xfail/xpass.
No source module existed at that commit.

**Pure reducer minimum GREEN implementation:** `65abfab3f78ac808a9492e434d2d9b62eda3e929`,
`mikazuki/parameter_policy_readiness.py` **added** (191 source lines).
The diff from RED HEAD contains **only that one module**; legacy
`parameter_policy_execution.py` / formal evidence registries /
Preview/Export/Start/direct Trainer / Checkpoint code **unchanged**.

[Actual Ubuntu/Windows scoped GREEN #42275463387](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42275463387):
**16/16 PASS per OS**, including all 12 behavioral contracts and 4 genuine
existing-code baselines; 0 FAIL/ERROR/SKIP/xfail/xpass. Tests exercise
frozen issue types, direct-constructor decision consistency, strict
capability/evidence enums, duplicate-code rejection, auto-issued unsupported
hard errors and experimental warnings, source order, defensive copying,
independent diagnostics, immutable BF16 identity, and no signing/CUDA authority.

**Source isolation:** The new module imports only standard-library
`collections.abc`, `dataclasses`, `re`, `typing`.
It does not consult configuration/qualification tables, import torch,
load model files, read environment, create jobs, sign results, issue GPU
certificates, or mutate checkpoint identity. Its `can_start` property is
an **advisory** and does not grant a trainer capability: the caller must
independently establish `implemented` or `experimental_candidate`
from real backend support and hard runtime invariants. A claimed
`formally_qualified` tier is **not authenticated** by this pure reducer.

**Qualification conclusion:** QG-1A2 **SCOPED PURE CONTRACT GREEN**,
**NOT** Full-BF16 execution GREEN, not a CUDA qualification, not a completed
product gate migration and not a production-service entitlement.

**Next independent step:** QG-1A3 Strict Code Review of the pure module,
typed errors, immutable diagnostic ABI and no-proof boundary. Only then
QG-1B Tests-only RED for turning the legacy `_qualification_blocker`
into independent runtime capability and evidence warnings. No normal
user training mode is unlocked by this QG-1A2 implementation.


## 6. QG-1A3-SR1: scoped findings, fixes and QG-1B trust boundary

**Independent review baseline:** `0f0285daf5393b13a21e45931d341bdd3d99641d`; [review #5806131245](https://github.com/WwlWss/diffusion-trainer-studio/pull/67#pullrequestreview-5806131245). **Scope:** repair only (F2) host input-exception contract and (F3) regression CI triggers. **F1** is explicitly a QG-1B integration precondition, not a QG-1A reducer algorithm change. Neither QG-1A3 nor any experimental result may grant production Start, signed evidence, CUDA qualification or Checkpoint original Load.

### F1: trusted classification ownership (QG-1B HOLD)

The QG-1A pure reducer's `hard_errors` and `warnings` are **caller-classified**. `ReadinessIssue.code` and `source` are diagnostic strings, **not authenticated severity claims**. Passing `runtime.dtype.mismatch` (R03) as a warning with `capability="implemented"` can yield `warn/can_start=True`; passing `qualification.backend.pending` (E02) as a hard error can yield `block`. Both are expected consequences of an upstream-owned severity contract, **not evidence of a production bypass** because QG-1A is not integrated into application Start or direct Trainer.

QG-1B MUST create a **first-party trusted classification producer** from concrete implementation, optimizer routing, target/asset, runtime dtype/device/parameter identity, and Checkpoint semantics. It must never trust client/Worker-provided `capability`, `severity`, `decision`, `can_start` or claims of `formally_qualified`. The producer maps actual R03 BF16 dtype mismatch to a hard error regardless of misbucketed client input; maps E02 absent formal certificate to an advisory only **when executable capability has been independently established**. Unknown or structurally unsupported backend/optimizer and currently unsupported multi-process ownership remain blocked. Preview and Start/direct Trainer must recheck genuine invariants. A code prefix (`runtime.*`) is not a severity authority; avoid duplicating a second broad backend classifier inside this pure QG-1A reducer.

Freeze these in later QG-1B Tests-only RED: `QG1B-F1-01` fake warning for real R03 must still block; `QG1B-F1-02` fake hard E02 must not block a proven implemented candidate; `QG1B-F1-03` untrusted `implemented/formally_qualified` cannot convert a planned or unsafe feature into a runnable one. Do not falsely mark these tests as complete in QG-1A3.

### F2: malformed custom Sequence conversion (fix within QG-1A3)

`_issue_tuple()` accepts `collections.abc.Sequence` then invokes `tuple(values)`, which can execute third-party `__len__/__getitem__/__iter__`. For an otherwise valid custom Sequence whose method raises `RuntimeError`, this leaks the exception type despite the frozen contract that malformed issue lists raise `ValueError`.

Preserve the public `Sequence[ReadinessIssue]` ABI, element validation, duplicate-code rules, immutable tuples, issue ordering, and existing decisions. Wrap **only** `result = tuple(values)` in a narrow `try/except`: explicitly re-raise `MemoryError`; convert other ordinary `Exception` failures to a static, non-echoing `ValueError` from `None`. Do not catch `KeyboardInterrupt/SystemExit` or make an invalid sequence into an empty sequence. No change to `_expected_decision` or capability logic.

### F3: CI coverage of imported legacy baselines (fix within QG-1A3)

Existing `.github/workflows/qg1a-readiness-host.yml` watches the new module/test/doc only, but B901–B904 import `parameter_policy_execution.py`, `optimizer_profiles.py` and `parameter_policy_runtime_bools.py`. Add those to **both** push and PR `paths`, and include `parameter_policy.py`, `training_request.py` and `parameter_policy_trainer.py` for the incoming QG-1B training-adapter regressions. Also watch the new `tests/test_parameter_policy_qg1a_sr1.py`, support `push` on `main` and keep `workflow_dispatch`. Preserve existing 16-case discovery/fail identity/0-SKIP gate, especially the old four baselines. If this workflow becomes a required GitHub branch-protection check, separately review the risk of path-filtered required checks remaining pending.

### Staged RED → GREEN acceptance

Dedicated `tests/test_parameter_policy_qg1a_sr1.py` freezes **six exact tests**:

- SR101: custom `Sequence.__len__` throws `RuntimeError`; reducer must raise `ValueError` without reflecting any sensitive exception text.
- SR102: custom `Sequence.__getitem__` throws `RuntimeError`; same required behavior.
- SR103: direct `ExecutionReadiness` construction with malformed sequence must fail with `ValueError` (not silently ALLOW).
- SR104: both `push` and `pull_request` paths include existing B901–B904 dependencies and the SR1 test, while push covers `main` and the dev branch.
- SR105 baseline: ordinary list/tuple inputs preserve decisions, frozen diagnostics and independent copies.
- SR106 baseline: `MemoryError` and `KeyboardInterrupt` are **not** normalized into configuration `ValueError`.

**Tests-only RED must be exact:** 6 discovered/executed, **4 targeted FAIL SR101–SR104 + 2 baseline PASS SR105–SR106**, all targeted failures carry `QG-1A3-SR1 CONTRACT NOT SATISFIED`, 0 ERROR/SKIP/xfail/xpass. The CI gate intentionally exits with failure on valid RED. Original QG-1A 16 tests remain 16/16 GREEN throughout. After minimum F2/F3 change, dedicated SR1 6/6 + existing 16/16 PASS on Ubuntu and Windows, with both PR and push triggers, 0 FAIL/ERROR/SKIP/xfail/xpass. The independent final Strict Review must inspect the exact committed diff, not interpret CI GREEN as actual GPU or training permission.

**QG-1A3-SR1 scope exclusion:** no edits to legacy release qualification registries, `mikazuki/parameter_policy_execution.py`, `parameter_policy.py`, `training_request.py`, `training_api.py`, `parameter_policy_trainer.py`, runtime/optimizer/Checkpoint/SD3 Resume, PR #65, or executable CUDA. QG-1B, QG-2, QG-3, and QG-4 remain distinct later stages.


## 7. QG-1A3-SR1 RED → GREEN evidence and bounded change set

**Scope / parent:** source audited at `0f0285daf5393b13a21e45931d341bdd3d99641d`.
Freeze F1 first-party producer contract in `QG-1B`; repair only F2/F3 in QG-1A3-SR1. No application entrypoint changed.

**Valid Tests-only RED:**
- QG-1A3-SR1 frozen six-case dedicated test/CI HEAD `3bee351f27f95dd0d11a73f2fa65a7f7151a7581`.
- [Actual push RED #42290724528](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42290724528), [PR RED #42290733938](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42290733938): Ubuntu + Windows each **6 run = exactly SR101–SR104 four targeted FAIL + SR105–SR106 two PASS**, with failure marker `QG-1A3-SR1 CONTRACT NOT SATISFIED`, zero ERROR/SKIP/xfail/xpass. Prior frozen QG-1A 16/16 baseline PASS in dedicated CI. The intentional job failure is an accepted RED, not a production regression.
- SR101/SR102 exercise real custom `Sequence.__len__` / `__getitem__` exceptions; SR103 attempts direct `ExecutionReadiness` with malformed input; SR104 reads actual workflow event-specific paths/branches. SR105 preserves list/tuple and diagnostic immutability; SR106 ensures `MemoryError` and `KeyboardInterrupt` are never converted to ordinary config errors.

**Minimal F2 patch:** `e39b37dca44f5cac6dffc423cf0f6c28c863ffea`, in `mikazuki/parameter_policy_readiness.py` only. Guard `tuple(values)` conversion, preserve `MemoryError` (and unhandled `BaseException` control exceptions), normalize any other ordinary `Exception` into a fixed-message `ValueError from None`. Keep `Sequence` API, type validation, duplicate-code detection, issue order, immutable snapshots, capability/decision/evidence and signature unchanged. Never echo raw exception text or Worker data in the new error.

**Minimal F3 patch:** `9dbd692a8cdbaafcce76638dd03b9e86f571803e`, existing `.github/workflows/qg1a-readiness-host.yml` only. Add `push/main` and, in **both** push / pull_request paths, original B901–B904 dependencies `parameter_policy_execution.py`, `optimizer_profiles.py`, `parameter_policy_runtime_bools.py`; plus new SR1 test, adjacent future integration modules `parameter_policy.py`, `training_request.py`, `parameter_policy_trainer.py` and SR1 workflow. Preserve original sixteen-test count, expected failure identity, 0-SKIP enforcement, Windows+Ubuntu matrix and manual dispatch.

**Actual scoped GREEN at F2/F3 code HEAD `9dbd692a8cdbaafcce76638dd03b9e86f571803e`:**
- [Dedicated SR1 push #42290866701](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42290866701) and [PR #42290877764](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42290877764): **Ubuntu and Windows 6/6 PASS** plus prior **16/16 PASS** each, zero FAIL/ERROR/SKIP/xfail/xpass.
- [Original QG-1A push #42290867531](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42290867531) and [PR #42290877382](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42290877382): **Ubuntu 16/16 PASS, Windows 16/16 PASS**, zero FAIL/ERROR/SKIP/xfail/xpass.

**Explicit HOLD:** QG-1B's genuine capability/severity provenance (F1), full legacy fatal blocker migration, Preview/Export/Start/direct Trainer integration, per-job experimental confirmation, SD3 Full-BF16 GPU R1/R2, official evidence signing, PR #65 OS Runner, Checkpoint original independent Load, and merging PR #67. The reducer's `can_start` remains pure advisory, **never** trusted as an authorization signal from the client. No external certificate promotion or production runtime unlock occurred. QG-1A3-SR1 Final Strict Review must be anchored to a fresh exact final HEAD after this evidence record.
