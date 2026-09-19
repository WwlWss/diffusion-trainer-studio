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

Add a real LRScheduler facade, external scheduler factory injection, and schedule-free train/eval lifecycle.

## 5D - Accelerate and resume smoke

Add narrow CPU PyTorch/Accelerate coverage for wrapping, step, state round-trip, topology mismatch, real Muon 3.10.0, and ScheduleFree.

## 5E - final review

Standard stays untouched, Component Start stays blocked, and Step 6 integrates only through the public runtime API.