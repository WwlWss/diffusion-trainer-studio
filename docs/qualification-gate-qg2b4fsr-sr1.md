# QG-2-B4-FSR-SR1: CI coverage and executable B312 step

Frozen parent aaa64316887368a3ab989b654b9e55d5f9efc7e0 and PR #69 Draft.

Tests-only RED: 15 genuine independent source failures. SR101 checks integration source/trigger coverage including main; SR102 ensures an unfiltered pull_request required check. SR201-SR213 each enforce a separate, unconditional B312 step. SR901-902 are real independent parser and negative-mutation baselines. Previous cumulative 16/16 Admission remains PASS.

Minimal GREEN after exact dual-OS RED: update B4-I central push path dependency coverage with broad mikazuki/**, scripts/stable/**, scripts/dev/**, tests/** prefixes; include main branch; make pull_request unfiltered. Preserve its stable job ID. Move the existing B312 assertion of each of 13 workflows into one standalone bash/Python 3.11 Step without erasing other inherited checks. Turn this SR1 workflow from explicit expected-RED to 15/15 real GREEN. Do not edit production training, Step 6F, CUDA or optimizer gates.

Treat older SD3 scope-test mismatch, queued QG1B3-B and required checks as independent merge dependencies. No PR merge until later independent FSR.