# QG-2-B4-L1 — Launcher per-invocation Request Intent transport: source contract and Tests-only RED

**Frozen parent:** \`a8496bb74f7a976cafec5285782fd48383b18500\` (B4-0 valid dual-OS two-reason B312 RED).
**PR:** #69, \`feature/qualification-gate-qg1b3-qg2-admission-red\`, Draft/Open/Unmerged.
**This phase:** only a contract document, a new executable Host test file, and a dedicated Ubuntu/Windows CI workflow. **Zero production changes**. Never claim that a Full-BF16 candidate may Start.

## 1. Read actual Launcher source

\`mikazuki/training_launcher.py:20-28\` defines \`run_prepared_train(toml_path, trainer_file, gpu_ids=None, cpu_threads=2, *, page_train_type=None, run_id=None)\`. There is no Request intent argument. The existing child argv comprises \`python -m accelerate.commands.launch\`, the Accelerate config / threads / optional multi-GPU arguments, \`trainer_file\`, and the child \`--config_file toml_path\`, with no one-shot acknowledgement flag. The GPU check, required Accelerate GPU YAML existence test, \`tm.create_task(args, env, metadata=...)\`, busy rejection and deferred \`asyncio.to_thread\` scheduling must retain their historical behavior.

\`mikazuki/training_config.py\` already contains the low-dependency immutable \`RequestAdmissionIntent\`, whose exact boolean \`experimental_full_bf16_ack\` and string \`origin in {"default", "webui_start"}\` are checked; positive acknowledgements require \`origin="webui_start"\`. The future Launcher must *also* check \`type(intent) is RequestAdmissionIntent\` and the exact positive fields at the boundary. A caller's dict, bool, structural impostor, forged origin or non-boolean positive must not become an argv acknowledgement.

The B4-0 cumulative B312 RED already names exactly two absent links: Launcher per-job typed input/child CLI flag and API handoff of \`prepared.admission_intent\`. **B4-L1 fixes NEITHER** in production; it freezes only the Launcher contract. B312 must remain the same two-reason RED at this stage.

## 2. Planned production GREEN contract, NOT implemented now

- Add **keyword-only** \`admission_intent: RequestAdmissionIntent | None = None\` to \`run_prepared_train()\`, preserving all existing positional arguments and default Standard argv.
- Treat \`None\` as absent/neutral. If supplied, accept **exact type only** and validate source/boolean strictly *before* CUDA validation, logging, task creation or async scheduling. Invalid or forged intent must fail closed, not silently become a Standard request.
- Only exact \`RequestAdmissionIntent(experimental_full_bf16_ack=True, origin="webui_start")\` may append exactly **one** \`--experimental_full_bf16_ack\` token as a discrete **child trainer argv** element **after** \`trainer_file\` and \`--config_file toml_path\`. The optional flag is confirmation only, not launch authorization; the upstream Request Start Candidate HOLD and downstream Direct Early/Late gates remain unchanged.
- \`RequestAdmissionIntent(False, "default")\`, \`RequestAdmissionIntent(False, "webui_start")\`, or no intent: no flag. Never mutate or consume the frozen input.
- No intent, ack or snapshot in the trainer TOML, Sidecar, environment, persisted task metadata, checkpoint, run ID, frontend session or process-global state. All command arrays must be rebuilt per invocation, so the flag cannot replay on a later neutral Start, busy/retry, or another process.
- Existing \`--multi_gpu\`, \`--num_processes\`, Windows \`--rdzv_backend c10d\` and \`USE_LIBUV=0\` **only** in existing OS-specific situations, must be preserved with proper parent/child argv boundary.
- GPU, missing Accelerator config, busy-task errors and Standard task metadata unchanged. The real API does **not yet** pass the intent; B4-A1 will cover that independently.

## 3. Eight exact executable target RED identities

\`tests/test_parameter_policy_qg2b4l1_launcher_red.py\` compiles the **actual** \`run_prepared_train\` AST from checked-in source (no independent reimplementation) and executes it with observable mock GPU validation, \`tm.create_task\`, \`asyncio.to_thread\` deferred sentinel, fake \`os.environ\` and \`APIResponse\`. This deliberately avoids importing optional CUDA, FastAPI or creating real subprocesses.

- **B4L101**: public Launcher function signature has strict keyword-only \`admission_intent\` with neutral default.
- **B4L102**: real positive typed WebUI Start intent yields exactly one flag in the child argv after \`trainer_file\` and config path.
- **B4L103**: explicit typed false intent is accepted and does not change Standard command/metadata.
- **B4L104**: nontyped bool/dict/impostor/object rejection occurs before GPU validation/task creation/async scheduling.
- **B4L105**: exact-type but forcibly malformed origin/positive boolean cannot append flag and fails before side effects.
- **B4L106**: one positive invocation followed by independent neutral or absent invocation leaves no ack in env/metadata/response and does not replay flag.
- **B4L107**: positive typed intent preserves parent Accelerate multi-GPU and Windows rendezvous arguments; one child flag remains *after* model trainer/config.
- **B4L108**: an already-active/busy task does not schedule an execution or contaminate the next neutral invocation with the earlier positive flag.

Each observed absence must be a **real** Python function-signature incompatibility or real incorrect argv/side effect, explicitly tagged \`QG-2-B4-L1 LAUNCHER TRANSPORT CONTRACT ABSENT\`. No unconditional failures, xfail, or accepted module import errors. On actual GREEN the same eight tests must PASS unchanged.

Six real baseline methods B4L901–906 execute existing Launcher with **no** new keyword and prove original Standard argv, CUDA rejection, missing config, busy behavior, Windows multiGPU existing behavior and genuine immutable first-party Request intent type validation. They must all PASS before implementing production changes.

## 4. Exact Ubuntu/Windows workflow

\`.github/workflows/qg2b4l1-launcher-transport-red.yml\`:
- Python 3.11, no GPU, subprocess, Torch, FastAPI server, or network need.
- B4L901–906 exactly **6 PASS**; B4L101–108 exactly **8 FAIL** with the frozen marker; zero unexpected ERROR, SKIP, xfail, xpass.
- Previous B4-0 six new baselines PASS, inherited QG-1B/SR1 + B2/B3 Host **116 PASS**, cumulative **15 PASS and only B312 RED with exactly two existing Launcher/API reasons**. The new 8 target RED methods are independently scoped and do not count as “another B312”.
- The targeted step exits 1 *only after exact eight real failures and six passing baselines*; any unrelated error or failure identity is INVALID RED.

## 5. Stop line

**Tests-only RED** means exactly three new files (document, tests, workflow) and **zero edits** to \`training_launcher.py\`, \`training_api.py\`, Request Host, any Direct Trainer/CLI, qualifications, GPU/CUDA, optimizer, task metadata, dataset, Checkpoint/Resume, or inherited B312 tests/workflows. PR remains Draft/unmerged; QG-2 Candidate and B312 remain HOLD.

After independent dual-OS valid RED, the next authorized stage is **QG-2-B4-L2 Launcher Minimal GREEN** (production change limited to \`training_launcher.py\`), then a new B4-A1 API handoff RED/GREEN and finally B4-I B312 + inherited workflow coordinated GREEN cutover.
