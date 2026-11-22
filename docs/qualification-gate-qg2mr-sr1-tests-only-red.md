# QG-2-MR-SR1-A: Overall Pre-Merge Tests-only RED

Frozen parent: 0b728f779aca2eb03db9325acdf4140425ad76f1. PR #69 remains Draft/unmerged; inherited #67 and #68 must not merge based on this RED. No training or CI production files are modified in this stage.

## Source defect inventory and actual RED identities

- MR-01 (13 target identities): six stable/dev real TOML Reader cases (neutral and positive reparse; output_config positive must reject before writing) plus seven Network Trainer main-guard ordering cases. The B0 architecture requested post-TOML extraction; B2 old source assertions inadvertently required pre-TOML extraction.
- MR-02 (9 targets): seven Network Trainer top-level first-party import probes executed in fresh Python -S subprocess without PYTHONPATH, plus Stable/Dev public Full Trainer TOML Reader subprocess probes. The common scripts/library bridge is the already-existing repo-root authority; do not move package-install requirements into Standard mode.
- MR-03 (2 targets): real isolated Stable/Dev train() prefix must reject positive one-shot ACK in Standard mode before training verification/dataset work, while preserving a neutral legacy preflight.
- MR-04 (2 targets): existing QG1B producer and SR1 workflows must watch main push as well as the historical branch, without changing earlier phase RED test meaning.

## Frozen independent passing baselines

Four new real baselines: strict single-use DirectTrainerAdmissionIntent, TOML reserved-key rejection through both real reader bodies, original repository-root lazy bridge cold subprocess load without Torch, and Standard neutral preflight. Eight original B2 tests and the cumulative 16/16 B312 suite remain truly passing. These checks must not be weakened to obtain a valid RED.

## Next controlled transitions

After exact Ubuntu/Windows 26 true source FAIL and baselines PASS, split GREEN into MR-SR1-B Import Bootstrap, MR-SR1-C post-TOML final extraction plus output_config guard, MR-SR1-D Standard positive ACK rejection, MR-SR1-E QG1B main CI and PR metadata, then a new independent FSR. GREEN tests must verify actual clean-subprocess launch/help in an installed CPU environment. Never loosen SD3 Full-BF16 source audit, the Request/Step6F candidate HOLD, Late Session, optimizer, CUDA or checkpoint guards.

## MR-SR1-B Import Bootstrap scoped GREEN changes

Production commit 367adc25dd0a42578610efaa68467c91025d3be5 changes eleven import/reader files only: Stable/Dev bridges use the established root finder to import one first-party stdlib-only CLI Intent module; Stable/Dev base Network Trainers bind the original strict type and functions from that module; five derived trainers reuse their base import; Full Trainer TOML readers reject the reserved ACK directly, before Namespace construction, without importing mikazuki. This does not change main CLI order, output_config behavior, runtime admission, CUDA or checkpoint.

Because historical B2 AST tests required a top-level direct import that is itself the identified defect, the eight B2 test identities are retained and their import clauses updated to require the safe Bridge and an explicit pre-Namespace key guard. Their pre-TOML CLI extraction order assumption remains in place until MR-SR1-C. The nine MR-02 targets must now be actual PASS on both OS, while MR-01/MR-03/MR-04 retain exactly 17 actual source FAIL. Existing four MR baselines, eight B2 and sixteen cumulative B312 tests must stay genuinely passing.

## Staged RED contracts must not enter default unittest discovery

After Import Bootstrap's 9/9 real dual-OS GREEN, the full Anima review automatically discovered the deliberately still-RED MR-01/MR-03/MR-04 tests in a `test_*.py` module and reported precisely 17 expected failures (no production import exception). This is a staging-harness exposure, not an additional production defect.

The exact same 26 target test methods and four baseline methods are preserved verbatim in `tests/qg2mr_sr1_contract.py` (no `test_` filename prefix). The dedicated MR-SR1 workflow explicitly imports and executes 9 current GREEN and 17 still-RED, retaining its own job-level failed RED evidence. The Anima and other generic `unittest discover` suites now exclude only the temporary phase-specific RED suite and continue discovering all established `test_*.py` regression tests. No test body is removed or skipped in the explicit phase CI. As MR-01/MR-03/MR-04 become GREEN, they remain in the exact named contract module until final strict review.
