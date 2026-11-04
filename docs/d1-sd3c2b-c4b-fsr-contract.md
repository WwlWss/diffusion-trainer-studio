# D1-SD3 C4-B-FSR — independent source-authority remediation contract

**Status at freeze: Tests-only RED candidate**. Parent PR #65 HEAD 1d03c36efd20c21e02b54cad02da7159dfc882f8, PR Draft/unmerged. No production CUDA, authenticated cross-process Resume, worker_complete or Release qualification.

## Frozen source authority and required changes

Real production sequence: NetworkTrainer.save_model invokes LoRANetwork.save_weights to external Step-N safetensors first; train_util.save_and_remove_state_stepwise then calls same-instance Accelerator.save_state (Accelerate 1.6.0). In that version automatic_checkpoint_naming may redirect actual destination; save_state returns its actual save_location. Accelerate checkpointing RNG file includes its **internal** step (not necessarily C3 logical Step), Python, NumPy, Torch CPU and CUDA random states.

**Fix A — destination authority:** freeze and reject automatic checkpoint naming, prohibit pre-existing authoritative Step state directories, compare actual original Save return location to approved State directory, preserve one-call and no-retry semantics. Last is ancillary.

**Fix B — LoRA Writer authority:** pin genuine production LoRANetwork.save_weights class method and observe the actual original method call, exact Step/path/metadata, successful return, real file SHA and Tensor bytes. Prohibit prewritten files and method shadowing. A Step-N Accelerator Save must have an original-writer observation from the same physical Anchor. Do not modify production serialization, math or general Trainer. Tiny real CPU LoRANetwork writer tests cannot claim CUDA.

**Fix C — stored RNG content:** snapshot actual Accelerate step and Python/NumPy/Torch CPU/CUDA RNG before original Save. After original returns, validate real random_states_0.pkl content via pinned, tightly allowlisted weights-only decoding; never use unrestricted pickle or untrusted Host torch.load. Compare exact typed values and same internal Step. If safe decoding is not possible, fail closed.

**Fix D — event provenance:** pin reviewed C3 PhysicalStepProbe._record class callable; reject instance override. After physical Save verification, check that exactly one actual checkpoint_saved event was appended with exact Case, Phase, Source, run, logical Step, checkpoint ID, Anchor and contiguous seq. Freeze event-prefix SHA at append; allow later Step2 events while preserving Step1 prefix.

**Fix E — immutable committed Save provenance:** display-only C4BVerifiedStepSave is not authority. A separately created internal committed record exists only after original Writer and original Accelerator Save return, complete nine-file/789 Tensor/RNG validation and audited event append; it binds original call identity, returned path, physical Anchor, checkpoint ID, event-prefix hash, LoRA file SHA and full state-file SHAs. verified_step_save checks committed provenance, Step identity, event prefix and actual files; a forged row in mutable _verified_step_records must fail. Fresh Step2 validates this real Step1 committed receipt. Never rehash old Step1 weights against current post-Step2 weights.

## Tests-only RED suite

New tests/test_sd3_full_bf16_worker_c4b_fsr_cpu.py, exactly twelve methods; 0 ERROR/SKIP/xfail/xpass; meaningful FAIL only.

- FSR01 forged nine ordinary files and self-calculated hashes cannot mint Step1 receipt
- FSR02 foreign Phase, Source and Checkpoint ID cannot be accepted from forged record
- FSR03 no genuine checkpoint_saved event must not pass append assertion
- FSR04 altered serialized Accelerator RNG internal Step must reject
- FSR05 altered stored Python RNG state must reject
- FSR06 actual original RNG file and actual state snapshot positive control
- FSR07 prewritten LoRA file and shadowed original writer must reject
- FSR08 same LoRA bytes with incorrect ss_steps metadata must reject
- FSR09 genuine LoRANetwork.save_weights must be observed and restored
- FSR10 automatic checkpoint naming rejected before original write
- FSR11 shadowed C3 Probe recorder must reject at a trust boundary
- FSR12 normal last stays ancillary and CPU remains unqualified

Use actual CPU BF16 R1 physical Step, actual original Accelerator.save_state and actual LoRANetwork serializer. A missing implementation yields an explicit C4-B-FSR CONTRACT ABSENT assertion FAIL, not ImportError/AttributeError. No test may spoof CUDA or fake 526 live trainable Parameters to mint production receipt.

GREEN later requires isolated Fix A→B→C→D→E commits; preserved original F6–F11, C4-A 12, SR1 8, C4-B 12, G1/G2 3, new FSR 12, both Host OS and C1–C3 PR/push workflows, plus a *new independent* Final Strict Code Review of the final exact HEAD. C4-C/D, independent Fix D runner, CUDA and worker_complete remain blocked.
