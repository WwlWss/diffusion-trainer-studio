# DTS QG-1A — pure structured ExecutionReadiness source contract

**Reviewed base:** `main@1dc4f56fb862aade574add48c4f1bea2c54f3e96`. **QG-0 source registry:** [114 gates](https://github.com/WwlWss/diffusion-trainer-studio/blob/docs/qualification-gate-architecture-review/docs/qualification-gate-qg0-inventory-2026-11.md) lives in Draft [PR #66](https://github.com/WwlWss/diffusion-trainer-studio/pull/66), deliberately not merged into this branch. **QG-1A is an isolated Tests-only RED stage**, not a production train/Start modification or CUDA qualification.

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
