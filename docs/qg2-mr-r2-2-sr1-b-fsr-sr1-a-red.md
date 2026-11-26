# QG-2-MR-R2-2-SR1-B-FSR-SR1-A: Tests-only RED

Frozen parent: 2369f9cd20f9148a5e15225e1b79acdcd2b3bfcf. PR #69 stays Draft / unmerged.

Root cause SR1-01: the strict GPU Harness used fixed ROOT.parent/execution-probe.json and ROOT.parent/baseline-probe.json as outputs and unconditionally unlinked both in finally. Their ownership was never proved. Preserve the previous Git-visible JSON probe, exact dirty-workspace assertions and the 25-second child timeout.

Root cause SR1-02: the synthetic Full-BF16 GPU fixture checks optimizer types but discards their argument dictionaries when comparing profiles, and checks only model class names plus parameter geometry rather than model source. Preserve the original production Admission block for Flux Full-BF16 and the real low-level runtime semantics.

RED contract: 11 tests, ten targets must fail (SR101-T1/T2/T3/T4, SR102-T1 through T6) and one existing safety baseline must pass (SR101-B1). SR101-T4 executes the real Harness test against an isolated fake repository with mocked Git/child processes and pre-existing external JSON sentinels; these files must not be modified or deleted. No GPU process is invoked. The six dynamic SR102 negatives patch scan_parameter_roots so they fail if an unreviewed input reaches the scan.

Minimal GREEN: unique exclusive JSON probe plus private TemporaryDirectory under ROOT.parent, never delete pre-existing external files; canonicalized full AdamW/Muon/C4 optimizer args; allow model definitions only in the GPU runner and previously accepted CPU fixture test module; reject unapproved sources and profiles before scanning. Retain eight GPU cases, fourteen phases, JSON qualification boundaries, real production routing, checkpoints and all production code unchanged. Execute the inherited 45 checks and 10 new checks on the final exact HEAD, followed by Host smoke, independent FSR and later separate CUDA regression.
