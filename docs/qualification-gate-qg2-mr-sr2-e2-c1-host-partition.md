# QG-2-MR-FSR-PR69-SR2-E2-C1 — Host Discovery Partition Minimal GREEN

Frozen accepted RED: eef552a09e2e7da83d331b038535aaed6f9b3928
RED tree: ef274b7b334fde8801cd6ed60f700596dd235e53
RED run: 45998930658, Ubuntu 140505322181, Windows 140505322500.
Both jobs: 36 target FAIL, 8 independent baseline PASS, nested 95/95, 917 non-error discovered test identities and two actual Torch import errors.

## Scope

Fix A only. Add the test-only runner, C1 qualification harness and this document. Change exactly the Anima validate Host run command; extend the new E2 workflow for C1-specific evidence. Keep E2-B contract, all pre-existing tests, production source, tools, submodules, qualification registry, other workflows and main/PR67/68/69 unchanged.

The runner selects every valid current test_parameter_policy*.py file except the two exact R2 CPU-only filenames, before using public unittest discovery per file with the original tests top-level directory. It preserves sorted discovery/module names and supplies the original wildcard pattern to standard load_tests hooks through the public loader method. Unreviewed recursive discovery fails closed. Reject missing frozen CPU files, empty selection, new discoverable packages, invalid filename, loader errors, _FailedTest, duplicate IDs or pre/post discovery loaded CPU aliases. No arbitrary exclusion configuration or runtime install. Host execution retains the historical dependency skips, records actual counts and skip identities, and fails on unexpected errors, failures, expected failure, unexpected success, count mismatch or CPU aliases. CPU execution belongs to E2-C2; C1 alone is not an integrated migration PASS.

## Exact staged evidence

The existing E2-B 36+8 source contract is unchanged. Stage entry runs all tests with E2H101 as the sole intended PASS and all other 35 failures matching the original marker. All eight baselines (including nested 95 actual tests) must PASS. Exit 1 remains required for pending RED; invalid outcomes exit 2. Ten separate behavior tests exercise pre-import exclusion/new files, missing CPU file, empty selection, package guard, actual import error, duplicate IDs via load_tests, correct load_tests, loaded alias guard, _FailedTest and strict outcome/skip accounting.

The independent full Host job uses recursive pinned submodules, Node 20 and Python 3.11 without Torch. The diagnostic legacy reference uses the original unittest discovery only as a comparison, verifies exactly the two real import errors, removes exactly those two _FailedTest entries, and executes all other cases. The actual Host runner never imports then filters. Compare ordered IDs, execution counts, errors/failures, and exact legacy skip identities/reasons. Six real frontend cases must execute, no skips. Expected prior baseline 917 Host identities and 70 existing runtime-dependency skips, subject to actual logs. These counts are evidence for the frozen C1 candidate, not hard-coded caps in the Host runner.

C1 expected: stage job failure (1 GREEN/35 RED), independent full Host job success on both OS. No R2 CPU Job or lifecycle migration yet. Continue to C2 only after raw log review. All new tests and diagnostic commands are test infrastructure only. No CUDA execution or backend qualification. Preserve the RED SHA and original evidence.
