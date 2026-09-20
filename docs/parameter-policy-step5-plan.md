# Parameter Policy Step 5 Plan - Composite Optimizer + Scheduler Runtime

Step 5 builds runtime objects below the audited RoutingPlan. It does not integrate trainers, mutate requires_grad, or unblock Component Start.

## 5A - deterministic Runtime Spec

parameter_policy_runtime.py stays PyTorch-free.

Runtime hierarchy:

~~~text
ParameterPolicyRuntimeSpec
  -> one OptimizerInstanceSpec per actually used Optimizer Profile
  -> one OptimizerGroupSpec per final learning rate
  -> RuntimeParameterSpec entries sorted by canonical parameter name
~~~

One Profile always means one optimizer instance. Primary and fallback assignments that end at the same Profile and LR share the same group.

Runtime object identity is used only for same-process duplicate ownership checks. Cross-process topology fingerprints use Profile/type/args, final group LR, and each parameter canonical name plus shape. Component labels and primary/fallback labels remain diagnostic metadata and do not enter optimizer-state topology hashes.

## 5B - torch runtime

parameter_policy_torch.py will construct child optimizers from Runtime Spec and expose one real torch.optim.Optimizer facade. Trainer integration remains out of scope.

## 5C - scheduler and ScheduleFree

Implemented contract:

- CompositeLRScheduler is a real PyTorch LRScheduler type but deliberately skips
  LRScheduler.__init__ so child schedulers are not advanced twice.
- Each external-scheduler Profile gets one child scheduler from the injected
  legacy scheduler factory.
- ScheduleFree Profiles are optimizer-managed and receive no external scheduler.
- CompositeOptimizer forwards train()/eval() and starts ScheduleFree children in
  train mode, matching existing dev trainer behavior.
- Composite _step_count propagates Accelerate's direct accumulation adjustment
  to external child scheduler counters.
- Scheduler state is versioned per Profile and validates mode, optimizer
  topology, scheduler class, step count, and epoch before child state mutation.

RAdamScheduleFree, AdamWScheduleFree, and SGDScheduleFree are promoted from
restricted to supported only with this lifecycle/no-external-scheduler contract.

## 5D - Accelerate and resume smoke

Implemented as a dedicated CPU job so the fast host-contract job stays torch-free.

Pinned smoke environment:

- torch 2.7.0 CPU
- accelerate 1.6.0
- schedulefree 1.4
- pytorch-optimizer 3.10.0

Coverage:

- real AdamW + ScheduleFree mixed runtime and an all-ScheduleFree runtime;
- CompositeOptimizer / CompositeLRScheduler child-group identity and stepping;
- two-microbatch Accelerate gradient accumulation with exactly one prepared
  optimizer and one prepared scheduler;
- Accelerate save_state/load_state over nested optimizer/scheduler state;
- manual fresh-runtime resume from a ScheduleFree eval-mode checkpoint;
- fail-closed optimizer topology, child optimizer type, and scheduler class
  mismatches;
- real pinned Muon CPU construction and step with every group forced to
  use_muon=True and Muon momentum state (no Adam fallback state).

## 5E - final review

Merge review findings:

- Standard request/start remains on the historical launch path; Step 5 adds no
  trainer integration and no requires_grad mutation.
- Component Start is still blocked before launch-only staging/materialization.
- Runtime Spec remains torch-free and deterministic; the torch runtime is only
  imported by explicit runtime consumers.
- One physical parameter is owned by one child optimizer, and Composite
  param_groups are live child-group objects rather than copies.
- Optimizer/scheduler checkpoint metadata is validated before child mutation;
  ScheduleFree eval-mode checkpoints resume into the runtime's active lifecycle.
- Real CPU smoke covers all three supported ScheduleFree implementations,
  pinned Muon 3.10.0, mixed external/optimizer-managed scheduling, Accelerate
  accumulation, and Accelerate save/load.

Step 6 handoff constraints that remain intentionally unresolved here:

- Component Start must not be unblocked until each trainer/model family applies
  higher-level target/freeze semantics, scans at the correct lifecycle point,
  and constructs this runtime.
- GPU-only bitsandbytes optimizers still need GPU smoke before release-level
  confidence; Step 5 only validates their constructor mapping structurally.
- GPU AMP/GradScaler behavior is supported by the single-Optimizer facade design
  but is not exercised by the CPU 5D job.
- FSDP/other optimizer-sharding integrations must be explicitly validated or
  blocked in Step 6; the Composite facade owns nested child state rather than a
  conventional flat outer optimizer.state mapping.
- Scheduler class/state is versioned here, but trainer scheduler arguments
  (warmup/total steps/cycles/etc.) are an effective-config concern. Step 6 must
  gate resume on compatible trainer scheduler configuration before
  accelerator.load_state().
- DeepSpeed and existing fused/blockwise optimizer modes remain compatibility
  blockers and are not made supported by Step 5.