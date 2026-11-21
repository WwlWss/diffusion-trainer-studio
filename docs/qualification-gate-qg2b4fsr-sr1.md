# QG-2-B4-FSR-SR1: CI coverage and executable B312 step

Frozen parent aaa64316887368a3ab989b654b9e55d5f9efc7e0 and PR #69 Draft.

Tests-only RED: 15 genuine independent source failures. SR101 checks integration source/trigger coverage including main; SR102 ensures an unfiltered pull_request required check. SR201-SR213 each enforce a separate, unconditional B312 step. SR901-902 are real independent parser and negative-mutation baselines. Previous cumulative 16/16 Admission remains PASS.

Minimal GREEN after exact dual-OS RED: update B4-I central push path dependency coverage with broad mikazuki/**, scripts/stable/**, scripts/dev/**, tests/** prefixes; include main branch; make pull_request unfiltered. Preserve its stable job ID. Move the existing B312 assertion of each of 13 workflows into one standalone bash/Python 3.11 Step without erasing other inherited checks. Turn this SR1 workflow from explicit expected-RED to 15/15 real GREEN. Do not edit production training, Step 6F, CUDA or optimizer gates.

Treat older SD3 scope-test mismatch, queued QG1B3-B and required checks as independent merge dependencies. No PR merge until later independent FSR.
## Frozen evidence and staged Green

Tests-only RED accepted at 4e42a9b4ce9d83d35e0dd7c87717043b3064cc66, dual-OS Actions run 44951637251: 15/15 genuine source-contract FAIL, 2/2 mutation baselines PASS and cumulative 16/16 Admission PASS. An earlier GitHub matrix-expression parse error on 22941499 was repaired before counting RED. SR1-Green staged the 13 inherited workflows as one commit, then the central trigger and CI harness as a separate commit; exact HEAD revalidation and Final Strict Review are required.

## SR1 completion evidence

Accepted exact RED at 4e42a9b4ce9d83d35e0dd7c87717043b3064cc66, Actions 44951637251: 15/15 real source-contract FAIL, 2/2 mutation baselines and B312 16/16 real PASS, on Ubuntu/Windows.

Green via two staged commits, 1f3cee1ba8a06bc5b2679bc349d9b58827ca745e and e1d44b01d171eff8771934387a14773927224de2. FSR-SR1 dual-OS success Actions 44951795226, central B4-I dual-OS success Actions 44951795151. Original QG1B3-B runner now also passes on both systems after separate fix bc247daa90208b56c0eeac46610c760bcffc72eb.

Code-bearing HEAD 7b02ee58aa730841db5708de9a8bb3baa095d049 completed 22 same-SHA Actions runs with SUCCESS, including Anima Host and pinned CPU runtime. A following doc-only commit requires exact-head recheck. Main branch protection endpoint is inaccessible through the connected integration (403); repo admin must verify Required Checks, and PRs #67 to #68 to #69 remain Draft/unmerged.
