# DTS QG-1B3 / QG-2 — Production admission boundary, parity and Tests-only RED

**Starting HEAD:** QG-1B2-SR1 Final Strict reviewed `eb4749d8832cf64a55755cb2261104baad2a1efb`, [stacked Draft PR #68](https://github.com/WwlWss/diffusion-trainer-studio/pull/68), itself stacked on [Draft PR #67](https://github.com/WwlWss/diffusion-trainer-studio/pull/67). QG-0's 114-gate inventory remains a distinct evidence source in Draft [PR #66](https://github.com/WwlWss/diffusion-trainer-studio/pull/66).

**This is QG-1B3/QG-2 *contract + Tests-only RED*, not a code migration or an execution unlock.** New source owner, UI control, direct Trainer handshake and certificate gate replacement must be implemented and independently reviewed in later GREEN phases.

## 1. Source audit: actual crossing points and old sequencing

| Path | Current behavior and exact code owner | Nonnegotiable invariant |
| --- | --- | --- |
| Request Decode | `training_request.py:37–43` reads page+config JSON; `prepare_request_config:179–239` canonicalizes Component Policy and compiles **effective** Trainer config after UI custom overrides. | Never trust raw caller `decision`, `capability`, `hard_errors`, `warnings`, `can_start` or purported evidence; never use pre-override raw fields to decide real compatibility. |
| Legacy Runtime Gate | `parameter_policy.py:543–661` calls old `parameter_policy_execution_blockers()` `:584–595`, followed by target/TE cache checks and referenced optimizer restrictions. | Preserve all semantic/target/TE cache, optimizer planned/restricted, GPU selection and actual Tensor/ownership hard failures. Do not filter strings by the word “pending.” |
| Legacy BF16 Evidence Gate | `parameter_policy_execution.py:419–433,479–598` returns early on backend `pending` before reaching other execution checks. | Replace **only** certification-only fatal denial after independently checking capability; keep unknown/unaudited SD3/SD2, bad precision, unsafe optimizer variants, process/DDP and asset/Checkpoint validation blocked. |
| Preview | `app/training_api.py:179–188` calls `prepare_request_config(launch=False)` and `validate_prepared_config(False)`; `_prepared_payload:84–112` presents `runtime_ready` and `runtime_blockers`. | No model-file/Torch/CUDA I/O or launch staging; return typed capability, hard issues, advisories and pending consent separately. |
| Export | `app/training_api.py:191–201` shares Preview preparation, then **deliberately materializes portable sidecars** (`:198`); it is not fully write-free. | Same diagnostic decision as Preview; do not generate a run authorization or store a transient acknowledgement in portable bundle. |
| Start | `app/training_api.py:236–275` invokes `prepare_request_config(launch=True)` before checking assets (`validate_prepared_config(prepared,True)`), staging Anima, sidecars, autosave TOML or launching. | Failed qualification, missing acknowledgement, semantic error, invalid real assets, GPU selection or trainer path must abort **before** the corresponding side effects. No “Preview said yes, so Start is trusted” shortcut. |
| Direct Trainer | `parameter_policy_trainer.py:1618–1665` loads real policy sidecar, rechecks semantic gate and calls legacy `parameter_policy_execution_blockers(require_assets=True)` **before** root scans, optimizer creation and Checkpoint actions. | Independently validate same source-owned qualification *and explicit direct-CLI intent*, then preserve live dtype/root identity, optimizer/device and checkpoint manifest assertions; no WebUI-only override or bypass by a copied diagnostic. |

**Critical discovered dependency:** removing a `pending` message solely in `training_request.py` would still fail in `create_parameter_policy_session`. Dropping Direct Trainer's old gate without a trusted replacement would bypass actual source/ownership checks. Both are prohibited.

## 2. Frozen admission owner and data shape (QG-1B3)

Proposed **new first-party** host adapter, initially absent on the RED branch:

```python
# mikazuki/parameter_policy_admission.py
from typing import Literal

AdmissionPurpose = Literal["preview", "export", "start", "direct_trainer"]
AdmissionStatus = Literal[
    "inactive", "legacy_approved", "ack_required",
    "experimental_host_candidate", "blocked",
]

def assess_component_training_admission(
    canonical_policy,
    *,
    train_type: str,
    effective_config,
    purpose: AdmissionPurpose,
    experimental_ack: bool = False,
) -> ComponentAdmission:
    ...
```

Return immutable `ComponentAdmission` with at least: `status`, `purpose`, `readiness: ExecutionReadiness | None`, `runtime_blockers: tuple[str,...]`, `warnings: tuple[str,...]`, `requires_experimental_ack: bool`, and `host_admissible: bool`. **`host_admissible` authorizes only proceeding to the next genuine runtime/asset check, NOT actual GPU launch or a physical certificate.** Do not add any “signed”/“CUDA verified” shortcut flag. Public API cannot accept a caller-provided `ExecutionReadiness`, `capability`, `can_start`, `evidence_tier`, `hard_errors` or `warnings`; these are computed by the first-party QG-1B producer from validated canonical Policy + post-override effective config.

- **Strict BF16 inactive/Standard:** preserve normal Standard/mixed BF16 behavior and absence of admission-specific side effects. Strict pseudo-boolean `full_bf16="false"` cannot become inactive.
- **Legacy approved route:** existing genuinely qualified SD1/SDXL source-supported Full-BF16 configs (where prior gate succeeds) must remain eligible to proceed to assets/Trainer **without a new consent requirement**, even though QG-1B Host `evidence_tier` is `unverified` for the new job. Do not mistake “per-job model identity unverified” for revoking previously exercised D1 support. An arbitrary fake row/unknown ID must not create genuine “legacy_approved” eligibility.
- **New certification-only gap:** only an implementation-audited `capability=implemented` with no hard issues, in an exact supported backend/optimizer path, may treat a `pending` or missing formal release record as a *warning* (not an execution hard failure). On Preview/Export, show a separate pending acknowledgement condition without doing work. On Start/direct Trainer, require a **per-job explicit boolean user acknowledgement**, strict type `type(...) is bool`; the absence of that acknowledgement yields `ack_required` and **no launch**. A true acknowledgement yields at most `experimental_host_candidate` and still requires assets and live Trainer audits. The old human-readable string “pending” is **not** a capability or severity authority.
- **Unaudited/unsafe** SD3 Full-BF16, SD2.x, unknown backend, planned CAME, unsupported optimizer variants, invalid BF16 precision, invalid booleans, SDXL known gradient-checkpointing ownership risk, DDP/multiprocess, Torch compile/FP8/offload conflicts, invalid target/TE cache and failed resume remain **blocked** even with acknowledgement or forged credential/certificate data.
- **Diagnostic separation:** `runtime_blockers` contains actual hard reasons plus any missing mandatory acknowledgement condition, without merging certificate warnings into hard errors. `warnings` carries evidence outside scope, model identity unverified, real pending release rows and experimental risk. `status` is deterministic. Input/evidence release row edits never change `ParameterPolicyExecutionContract.execution_identity()`, optimizer topology or checkpoint manifest schema.
- **First-party host invariants vs runtime facts:** `classify_component_full_bf16_readiness(... require_assets=False)` is for host diagnostics only; passing `require_assets=True` to it deliberately raises. Physical model/dataset/trainer validation remains owned by `validate_prepared_config(prepared, True)` before side effects, with real tensor/optimizer/Checkpoint invariants still checked in Direct Trainer. A forged `launch_assets_checked=True` value in GUI/Worker input cannot skip this.
- **No overbroad certificate migration:** do not delete `_qualification_blocker` or rewrite `FULL_BF16_*_QUALIFICATIONS`. Preserve it for the original tested paths and use a narrowly reviewed new adapter to classify certificate-only failure without swallowing source hard blockers. There must be a proved stable-code/registry distinction, not a case-insensitive free-text filter.

## 3. The explicit acknowledgement transport is an additional security boundary (QG-2)

The required GUI interaction is a dedicated per-job acknowledgement (`experimental_full_bf16_ack`) shown only for eligible Host candidates; it **must** be opt-in, visible and false by default. It is not equivalent to an arbitrary `can_start`, `runtime_ready` or supplied certificate. Do not accept the acknowledgement from `ui_custom_params` or a copied exported bundle. It must be captured as a real control-plane intent **before** Trainer config normalization and kept separate from trainer optimizer/model fields. Export/rehydrate must not silently persist/replay this intent; Start checks again.

Direct Trainer (headless) must have an equally explicit recognized CLI admission flag with strict parsing and must recompute the first-party producer from the *actually loaded sidecar*. A WebUI-origin run must propagate its single-job acknowledgement through a dedicated **reviewed launcher-to-trainer control channel**; it cannot simply set an arbitrary TOML key that sd-scripts might ignore, or change the execution/checkpoint identity. Until that channel has a true end-to-end test, the integration must fail closed rather than open WebUI launch but block on the child process.

The exact CLI/parser/launcher implementation belongs to a later QG-2 GREEN step; **this RED freezes the requirement rather than pretending the transport already exists**.

## 4. Stage boundary and new Tests-only RED (B301–B312 plus B901–B904)

One new `tests/test_parameter_policy_admission_red.py` runs **16 exact host-side tests**: **12 newly unmet contracts (B301–B312 targeted FAIL)** and **four genuinely passing source baselines (B901–B904 PASS)**. This test suite is intentionally RED and does not create the new adapter module or change any existing function. A dedicated CI runs Python 3.11 on Ubuntu/Windows and accepts exactly 12 tagged target failures +4 baseline successes, **0 ERROR/SKIP/xfail/xpass**; preserve original QG-1B 16/16 and SR1 16+10 before targeted RED. The new tests must become real semantic tests when GREEN is implemented, not be skipped or erased.

| Test | Future passing contract; source owner |
| --- | --- |
| **B301** | New admission adapter exists with narrow explicitly typed signature and immutable result; disallow caller-authority fields. |
| **B302** | Real source-audited, currently qualified SD1 stock LoRA remains host-admissible without novel experimental acknowledgement even when new per-job evidence tier is unverified. |
| **B303** | Same audited backend patched to *pending* certificate produces only advisory and a genuine `ack_required` state in Preview; does **not** issue false physical qualification. |
| **B304** | For that pending-only supported configuration, Start without explicit `True` fails; with `True` becomes *experimental Host candidate only*, with no real launch/asset proof. |
| **B305** | Fake caller `capability/decision/can_start`, fake release row and even explicit acknowledgement cannot admit unreviewed SD3 Full-BF16. |
| **B306** | Explicit acknowledgement cannot override incorrect BF16 precision or quoted boolean `full_bf16`. |
| **B307** | Explicit acknowledgement cannot override SDXL known gradient-checkpoint ownership risk or planned optimizer. |
| **B308** | Standard/inactive and mixed BF16 are unchanged; quoted acknowledgement, unknown purpose and untrusted injected keyword arguments fail closed. |
| **B309** | Parameter Policy legacy runtime-blockers owner uses the trusted adapter for scoped certificate-only migration **without dropping semantic/target/cache/optimizer checks**. |
| **B310** | `prepare_request_config` stores a typed admission/ack state separately from Trainer config, recomputes on effective config and enforces before Start/materialization. No user-crafted `can_start` claim. |
| **B311** | Preview/Export payload exposes typed readiness plus acknowledgement condition, maintains preview model-free behavior and Export's deliberate portable sidecar behavior; actual Start validates assets before side effects. |
| **B312** | Direct Trainer uses the *same first-party adapter* with explicit direct-CLI acknowledgement, loaded-policy/source checks, and all genuine live dtype, optimizer/root and Checkpoint assertions preserved; no WebUI-only bypass. |
| **B901–B904** | Real unchanged baseline: existing D1 qualified SD1 legacy gate, SD3 pending legacy stop, inactive+invalid-boolean behavior, and old Request/Start+Direct Trainer source order. |

**Do not manufacture RED:** the twelve tests must fail for those twelve exact intended missing contracts with marker `QG-1B3-QG2 ADMISSION CONTRACT ABSENT`; the other four must pass using real current source functions. The workflow does not accept ERROR/SKIP/xfail/xpass, unrelated failure, or test-harness syntax/fixture breakages as valid RED. Anima umbrella discovery may report these same twelve intentional failures while this Draft is RED; it must return GREEN before merge.

## 5. Implementation roadmap after valid RED (not in this turn)

1. **QG-1B3-A:** implement the first-party adapter only, tests B301–B308 GREEN while B309–B312 remain intentionally RED; CI must retain honest targeted scope and unchanged old baselines.
2. **QG-1B3-B:** source-level strictly typed certificate-only migration in `parameter_policy_runtime_blockers` without free-text filtering, preserving all independent legacy hard checks; separate exact tests for host readiness and actual release-row variants.
3. **QG-2-A:** dedicated GUI acknowledgement and typed Preview/Export/Start behavior, post-override re-evaluation, fail-closed no-side-effect real Start asset check and no bundle consent replay.
4. **QG-2-B:** explicit headless direct Trainer and launcher-to-trainer opt-in transport; preserve direct loaded sidecar canonicalization, real BF16 Tensor/root/optimizer/device, DDP and Checkpoint Resume checks.
5. **QG-2-C:** independent Final Strict Code Review, full tests/CI, then only after audited branch dependencies consider merge. **QG-3 SD3 true Full-BF16 source ability and QG-4 physical CUDA R1/R2** remain separate later qualification stages. PR #65 protected signer is not a prerequisite for normal local training.

**This contract itself grants no runtime permission.** Old Preview/Start/Direct Trainer behavior remains exactly as checked in the four baseline tests.


## 6. Exact QG-1B3/QG-2 B3 RED evidence and safety interpretation

**Parent:** QG-1B2-SR1 `eb4749d8832cf64a55755cb2261104baad2a1efb`, [Draft PR #68](https://github.com/WwlWss/diffusion-trainer-studio/pull/68). New [Draft stacked PR #69](https://github.com/WwlWss/diffusion-trainer-studio/pull/69). **Test-only B3 RED: no admission implementation, production entrypoint change, GUI/CLI flag, runtime override or CUDA.**

**Test harness correction before acceptance:** Initial commit `630d9de30c1245ce51612280c88dd56fc9466739` executed all original 42 tests GREEN but B306 used two independent failing unittest `subTest` cases when the new source adapter was absent. That generated **13 failure records** instead of 12. [Push #42366028419](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42366028419) / [PR #42366059116](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42366059116) are **INVALID for frozen exact-count RED**; never cite them as valid RED. At `f4bc612f80a79b12d0354e7aa4acc612daafb3e6` B306 was changed **in the test only** to inspect both bad-FP16 and quoted-bool cases under one test method without two failing subtest records. Neither negative assertion was deleted.

**Valid dual-OS RED** at `f4bc612f80a79b12d0354e7aa4acc612daafb3e6`:
- [Push #42366122260](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42366122260): Ubuntu and Windows **42/42 prior PASS**, followed by **16 new cases = precisely B301–B312 twelve tagged FAIL + B901–B904 four real legacy PASS**, 0 ERROR/SKIP/xfail/xpass.
- [PR #42366125598](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42366125598): same exact results on both OS.
- [Anima Qwen3 PR #42366125605](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42366125605) discovered **682 host tests, 12 intentionally failing new B301–B312, 64 optional skips**; the dependent runtime smoke was skipped after upstream failure. This is the intended shared-discovery effect of a real Tests-only RED. This draft cannot be merged while intentionally RED.

**Additional contract strengthening:** `671c3bc921c59020f12251b9533b08e57736430c` added within B302 (without changing the 16 test count) a fake `qualified` release row with a **wrong case ID**. The future first-party adapter may preserve *previously permitted* SD1 Full-BF16 execution behavior, but cannot label a forged case-ID row `legacy_approved`. “Legacy approved” is a compatibility admission category, **NOT a declaration that this new user's model/training geometry was physically D1-certified**; QG-1B evidence tier stays `unverified` unless separately trusted input/file/case identity is established. [Push #42366234427](https://github.com/WwlWss/diffusion-trainer-studio/actions/runs/42366234427) again demonstrated **42/42 prior GREEN and exact 12 targeted FAIL +4 baseline PASS on both Ubuntu and Windows**, with 0 ERROR/SKIP/xfail/xpass.

**Critical interop:** A positive Host diagnosis cannot by itself bypass `validate_prepared_config(prepared, True)`, Trainer's live BF16 Tensor dtype/root/optimizer/device/Checkpoint assertions, or the Headless-specific direct-session preflight. A GUI acknowledgment is not a formal qualification and must remain a user-visible, per-job, opt-in control with a real one-job launcher-to-trainer channel. Neither \`ui_custom_params\` nor replaying an exported bundle may forge it. Until QG-2 closes the transport and parity tests, `experimental_host_candidate` can only mean **proceed to mandatory checks**, never launch permission.

**RED source-scope:** only three new files relative to PR #68's fixed HEAD: this contract, `tests/test_parameter_policy_admission_red.py` (the 16 frozen tests), and `.github/workflows/qg1b3-qg2-admission-host.yml`. The B306/B302 test refinement commits did not modify source. QG-1B3-A and QG-2 GREEN are deliberately not implemented here. Keep PR #69 Draft/unmerged until the new adapter, real Start/Direct wiring, explicit per-job acknowledgement path and independent strict review are all complete.
