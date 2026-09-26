# Parameter Policy 开发记录、当前能力与后续路线图

> 状态基线：`main@9d9446ad1fd8c5bce45e6322c763043441702add`（PR #46 已合并，2026-09-24）
>
> 本文是 **工程开发文档**，用于说明 Parameter Policy / Component-wise Optimization 从最初设计到当前成品经历了哪些步骤、每一步具体开发了什么代码、形成了什么产物、现在已经能做什么，以及仍有哪些能力尚未完成或尚未完成 release qualification。
>
> 面向使用者的简明说明仍见 [parameter-policy.md](./parameter-policy.md)。历史 Step 3–7 计划文档继续保留，作为阶段性设计记录。

---

## 1. 项目目标

Parameter Policy 的核心目标不是简单地“给不同模块填不同学习率”，而是把 **参数归属、Train/Freeze、Optimizer、Learning Rate、Fallback、Scheduler、checkpoint identity 和 runtime ownership** 统一成一套可审计、可恢复、可 fail-closed 的训练契约。

最终需要同时满足以下要求：

1. **Standard 不被破坏。** 没有显式进入 Component 模式时，继续使用原有 sd-scripts / DTS optimizer、LR 和 trainer 逻辑。
2. **Component 成为唯一 optimizer/LR authority。** 进入 Component 模式后，组件路由决定哪些参数训练、由哪个 Optimizer Profile 管理、使用多少 LR。
3. **模型组件不是字符串猜测。** Component ID 来自 backend-owned Model Component Profile；实际参数通过真实 module/parameter identity 路由。
4. **一个参数只能被一个 optimizer 最终拥有。** shared parameter / alias 必须经过冲突审计。
5. **Muon eligibility 必须显式。** 不是所有参数都能进入 Muon；不符合 eligibility 的参数必须走显式 fallback，而不是依赖 Muon 内部隐藏的 AdamW fallback。
6. **runtime 必须能被 Accelerate 看成一个 optimizer。** 多个 child optimizer 对 trainer 暴露为一个 CompositeOptimizer，并有对应 CompositeLRScheduler。
7. **save/resume 必须绑定 ownership identity。** policy、optimizer topology、scheduler identity 与 checkpoint 必须一致，不能“勉强加载”。
8. **GUI 只能暴露 runtime 真能执行的语义。** Preview/Start 都必须 fail-closed。
9. **高级执行模式不能因为基础训练跑通就自动视为兼容。** full BF16/FP16、FP8、compile、DeepSpeed、fused、offload、multi-GPU 等必须分别 qualification。
10. **所有 backend 保持同一架构，不允许 Anima 成为特殊旁路。**

---

## 2. 到目前为止一共做了多少阶段

从 Parameter Policy 正式进入仓库开始，核心实现由 **7 个主步骤**组成：

| 主步骤 | 对应 PR | 核心目标 | 最终产物 |
| --- | --- | --- | --- |
| Step 1 | #21 | Optimizer capability foundation | optimizer capability registry |
| Step 2 | #22 | Host contract | canonical policy、sidecar、bootstrap/request/rehydrate 契约 |
| Step 3 | #23 | Model routing foundation | Model Component Profile + pure parameter routing |
| Step 4 | #24 | LoRA original-target metadata | adapter 原始目标元数据，解决 LoRA 参数真实归属 |
| Step 5 | #25 | Composite runtime | Runtime Spec、CompositeOptimizer、CompositeLRScheduler |
| Step 6A–6F | #26–#31 | Trainer/runtime integration | 10 backend 的 Component runtime、checkpoint、GPU matrix、Start gate |
| Step 7A–7F | #32–#38 | GUI/release closure | GUI editor、bootstrap、Preview、presets、import/export、regression closure |

随后进行了 **8 个发布后 hardening PR**：

- #39 Component editor / Multi-Caption UX 修复
- #40 legacy Schemastery union 稳定化
- #41 Profile key / entry actions
- #42 Profile name 与 optimizer type UI 分离
- #43 generated frontend asset MIME 修复
- #44 GUI lifecycle gaps 收口
- #45 implicit CUDA device audit 修复
- #46 non-Anima semantics / cache / target / wire-format hardening

因此，从 **PR #21 到 PR #46 共 26 个合并 PR** 构成了当前 Parameter Policy 的完整开发链。

---

# 3. Step 1 — Optimizer Capability Foundation

**PR #21**  
Merge: `795292cf0fa696a81c7fc30b8564f6ca58ffa222`

## 3.1 目标

在任何 Component routing 或 optimizer construction 之前，先建立一个 **optimizer capability registry**，回答：

- 这个 optimizer 是否允许被 Component 模式选择；
- 是否支持 group LR；
- 是否依赖 external scheduler；
- LR semantics 是 normal / adaptive / optimizer-managed；
- 是否需要额外 dependency；
- 是否存在 parameter eligibility 约束；
- 是否属于 supported / restricted / planned。

这是后续所有 GUI、routing、runtime 和 qualification 的共同 source of truth。

## 3.2 代码开发

主要新增/修改：

- `mikazuki/optimizer_profiles.py`
- `tests/test_optimizer_profile_capabilities.py`
- CI wiring

建立 `OptimizerCapability` 数据模型与 registry，并为 Muon 引入：

`eligibility_policy="model_hidden_2d_weight"`

同时把 optimizer 状态分为：

- `supported`
- `restricted`
- `planned`

## 3.3 成品

形成了统一 optimizer capability registry。后续 GUI 不再自己硬编码 optimizer 支持表，router/runtime 也能使用同一个 capability contract。

## 3.4 当时仍未解决

- 没有 Component policy；
- 没有 model component；
- 没有参数 routing；
- 没有 runtime optimizer；
- 没有 trainer integration。

---

# 4. Step 2 — Parameter Policy Host Contract

**PR #22**  
Merge: `a065c2983db90d0d9f5fc2e5324788b4a17307f2`

## 4.1 目标

先建立一个完全 host-side、dependency-light 的 Parameter Policy 契约，让 DTS 可以：

- 表示 optimizer profiles；
- 表示 component Train/Freeze、LR、fallback；
- canonicalize / validate policy；
- 从 Standard 配置 bootstrap Component policy；
- 生成 content-addressed sidecar；
- 让 request / export / import / rehydrate 能携带 policy；
- Standard 模式继续完全忽略隐藏 Component state。

## 4.2 代码开发

主要文件：

- `mikazuki/parameter_policy.py`
- `mikazuki/parameter_policy_bootstrap.py`
- `mikazuki/training_config.py`
- `mikazuki/training_gui_args.py`
- `mikazuki/training_rehydrate.py`
- `mikazuki/training_request.py`
- `mikazuki/app/training_api.py`

## 4.3 成品

形成了 Parameter Policy v1 的 canonical host contract：

```text
Optimizer Profiles
+
Components
  - Train / Freeze
  - primary optimizer profile
  - primary LR
  - optional fallback profile
  - optional fallback LR
```

Policy 可以单独序列化为 sidecar，并进入 Preview / Export / Import / Rehydrate 流程。

## 4.4 关键原则

Step 2 就确立了一个一直保留到现在的原则：

> **Standard 是独立 legacy path，不是“Component 的一个特殊 preset”。**

这使后续 Component runtime 即使复杂化，也不会改变历史 Standard 训练行为。

---

# 5. Step 3 — Model Component Profiles + Pure Parameter Routing

**PR #23**  
Merge: `574cfb2ca80d6da54d8cebc70f19e90c13b77532`

## 5.1 目标

解决真正困难的问题：

> 一个真实 PyTorch Parameter 到底属于哪个模型组件？它是否允许被当前训练 target 使用？是否符合 primary optimizer eligibility？不符合时 fallback 到哪里？

Step 3 明确要求 routing 在 optimizer 构造前完成，而且不允许通过参数名字简单猜测。

## 5.2 代码开发

新增核心模块：

- `mikazuki/model_component_profiles.py`
- `mikazuki/parameter_routing.py`
- `mikazuki/parameter_policy_compat.py`

同时扩展：

- `optimizer_profiles.py`
- `parameter_policy.py`
- `parameter_policy_bootstrap.py`

## 5.3 关键实现

### Model Component Profile

每个 backend 定义固定 component registry。

它回答：

- 参数属于哪个 architectural component；
- 当前更高层 training target 是否允许这个 component；
- 参数结构类别是什么；
- Muon eligibility 是否成立。

### Parameter identity scan

扫描以真实 `id(parameter)` 为中心，不以 name 为中心：

- 保存所有 alias；
- 检查 shared parameter；
- 检查 alias 是否映射到不同 component；
- fail-closed 处理冲突。

### Routing kinds

最终每个参数只能落入：

- `primary`
- `fallback`
- `frozen`
- `unavailable`

## 5.4 成品

到 Step 3，DTS 已经可以在 **不构建 optimizer** 的情况下生成完整 RoutingPlan，并回答：

- 哪些 tensor 要训练；
- 哪些 tensor 被 freeze；
- 哪些参数走 Muon；
- 哪些参数因 eligibility 失败走 fallback；
- 是否有 orphan / duplicate / unavailable / conflict。

---

# 6. Step 4 — LoRA Original-Target Metadata

**PR #24**  
Merge: `5793e178281a2057c7165dd4f96f9cf609081d95`

## 6.1 目标

LoRA 参数的 module path 通常只能看到 adapter 自身，例如：

```text
lora_down.weight
lora_up.weight
```

但 Parameter Policy 真正需要知道的是：

> 这个 LoRA adapter 原本挂在哪个 base-model module 上？

否则无法可靠判断它属于 attention、MLP、text encoder 还是其它组件。

## 6.2 代码开发

修改：

- stable LoRA
- Flux LoRA
- SD3 LoRA
- Anima staged metadata patch

主要文件：

- `scripts/stable/networks/lora.py`
- `scripts/dev/networks/lora_flux.py`
- `scripts/dev/networks/lora_sd3.py`
- `mikazuki/anima_runtime.py`
- `tools/apply_anima_parameter_policy_metadata_patch.py`

## 6.3 成品

LoRA adapter 注册 **original target module identity**，router 可以把 adapter descendant parameters 追溯到真实 base-model component。

这一步使 LoRA Component routing 从“命名约定”升级为“真实 target metadata”。

---

# 7. Step 5 — Composite Optimizer + Scheduler Runtime

**PR #25**  
Merge: `ba2d7495a8fbc8cb143c79add113a8aa63774304`

## 7.1 目标

把已经审计完成的 RoutingPlan 真正编译成可训练 runtime。

需要解决的问题：

- 一个 policy 可以有多个 Optimizer Profile；
- 同一个 Profile 可以有多个 LR group；
- Muon primary + AdamW fallback 必须是两个真正独立 child optimizer；
- trainer / Accelerate 又希望看到一个 optimizer；
- ScheduleFree optimizer 不能再套 external scheduler；
- save/load 必须保存 nested optimizer/scheduler identity。

## 7.2 代码开发

新增：

- `mikazuki/parameter_policy_runtime.py`
- `mikazuki/parameter_policy_torch.py`

核心对象：

```text
RoutingPlan
  ↓
ParameterPolicyRuntimeSpec
  ↓
OptimizerInstanceSpec
  ↓
OptimizerGroupSpec
  ↓
CompositeOptimizer
  ↓
CompositeLRScheduler
```

## 7.3 Runtime Spec

Runtime Spec 仍保持 PyTorch-free：

- 一个实际使用的 Profile = 一个 OptimizerInstanceSpec；
- 同 profile + 同 LR 的参数合并为一个 group；
- topology fingerprint 使用 profile/type/args/LR/parameter canonical identity；
- routing kind 只作为 diagnostics，不改变 checkpoint topology identity。

## 7.4 CompositeOptimizer

对 Accelerate 暴露成一个真实 `torch.optim.Optimizer` facade，但内部维护多个 child optimizer。

它负责：

- `step()`
- `zero_grad()`
- `train()/eval()`
- `param_groups`
- nested `state_dict()`
- topology-safe `load_state_dict()`

## 7.5 CompositeLRScheduler

区分：

- external-scheduler optimizer；
- optimizer-managed ScheduleFree optimizer。

ScheduleFree 不会错误地再获得外部 scheduler。

## 7.6 Muon fallback

Muon 不使用其内部 AdamW fallback。

Parameter Policy 的语义是：

```text
Muon eligible parameter
    → Muon child optimizer

Muon ineligible parameter
    → explicit fallback profile
    → e.g. AdamW child optimizer
```

这样 ownership、LR 和 checkpoint topology 都是显式、可审计的。

## 7.7 当时完成的 runtime smoke

Step 5 CPU runtime 已覆盖：

- AdamW；
- ScheduleFree；
- mixed child optimizers；
- Accelerate accumulation；
- optimizer/scheduler save/load；
- Muon construction / step；
- Muon group 强制 `use_muon=True`，不使用内部 Adam fallback。

---

# 8. Step 6 — Trainer Runtime Integration

Step 6 是把 Step 1–5 的抽象系统真正接进训练器。

---

## 8.1 Step 6A — Shared Trainer Boundary

**PR #26**  
Merge: `bcd0585575b9000cdc856e96a235fe8add2e2d09`

新增核心：

- `mikazuki/parameter_policy_trainer.py`
- stable/dev lazy bridge

形成 `ParameterPolicyTrainerSession`，统一拥有：

1. policy sidecar load；
2. compatibility preflight；
3. parameter scan；
4. routing；
5. Runtime Spec；
6. CompositeOptimizer；
7. requires-grad contract；
8. CompositeLRScheduler；
9. post-prepare device audit；
10. checkpoint manifest / scheduler identity。

并增加第二道防线：

> Component policy 存在时，legacy `get_optimizer()` 不允许偷偷接管。

Standard trainer 不 import bridge，保持物理隔离。

---

## 8.2 Step 6B — LoRA NetworkTrainer Integration

**PR #27**  
Merge: `45e7a259cff95823b47a173c412a405160bbcd0f`

接入：

- SD LoRA
- SDXL LoRA
- Flux LoRA
- Chroma LoRA
- SD3 LoRA

Component mode：

- 不再调用 legacy optimizer-group builder；
- session 从 LoRA network root 扫描参数；
- final target 只能进一步 freeze，不能扩大 trainer target；
- clip 使用 session-owned parameters；
- LR logging 使用 component IDs；
- ScheduleFree lifecycle 与 save/sample 流程对齐。

同时开始处理 Text Encoder cache 与 target ownership 的交互。

---

## 8.3 Step 6C — Full Trainer Integration

**PR #28**  
Merge: `407636fdfa171b6f187769a36906520ede40418e`

接入：

- SD DreamBooth
- SDXL Full
- Flux Full

特殊处理包括：

- DreamBooth dynamic TE stop 在 Component 中 fail-closed；
- SDXL TE structural freeze；
- Flux Full legacy blockwise/fused/offload/DeepSpeed 保持 blocker。

---

## 8.4 Step 6D — Anima Staged Runtime

**PR #29**  
Merge: `e6526511d49b95141b2522305fc15d983533e781`

Anima 不直接修改 submodule，而是把 Parameter Policy runtime 作为 staged feature patch 进入运行时。

接入：

- Anima LoRA
- Anima Full
- DiT
- Qwen3
- optional LLM Adapter

仍由 trainer 保留：

- dtype/device placement；
- model mode；
- Qwen3 checkpointing；
- sampling/save；
- paired sidecars。

Parameter Policy 只接管 optimizer/LR/requires-grad/ownership。

---

## 8.5 Step 6E — Runtime Hardening

**PR #30**  
Merge: `7e4a5714a4c29996f799c8e6b8d9789d0769068c`

统一 runtime lifecycle：

```text
build session
→ Accelerate.prepare
→ finalize_after_prepare
→ resume
→ post_resume contract
→ epoch_start contract
```

### Checkpoint manifest v2

记录：

- policy hash；
- runtime topology fingerprint；
- scheduler signature；
- train type；
- trainable/frozen components；
- tensor/element counts；
- profile → optimizer type。

### Safety audits

- requires-grad contract；
- optimizer ownership；
- trainable device；
- resume identity；
- save/load pre-hooks。

---

## 8.6 Step 6F — GPU Matrix + Component Start

**PR #31**  
Merge: `41fd8d09ccb61caa621c052ecb315efea52f7fbb`

正式打开 baseline Component Start。

### 10 个 runtime backend

- `sd-lora`
- `sdxl-lora`
- `sd-dreambooth`
- `sdxl-finetune`
- `sd3-lora`
- `flux-lora`
- `chroma-lora`
- `flux-finetune`
- `anima-lora`
- `anima-finetune`

### Qualification infrastructure

新增：

- `.github/workflows/parameter-policy-gpu-matrix.yml`
- `tools/run_parameter_policy_gpu_matrix.py`
- `tools/run_parameter_policy_backend_gpu_matrix.py`

区分：

1. backend matrix；
2. optimizer/runtime matrix；
3. real trainer matrix；
4. execution-feature qualification matrix。

这一步开始明确：

> “backend 已支持 Component”并不等于“这个 backend 的所有高级训练选项都已经支持 Component”。

---

# 9. Step 7 — GUI / Presets / Release Closure

---

## 9.1 Step 7A — Editor Host Contract

**PR #32**

新增：

- `parameter_policy_editor.py`
- editor metadata
- bootstrap
- normalize
- preview

GUI metadata 从 registry 生成，不自己复制 component/optimizer 表。

---

## 9.2 Step 7B — Generated Schemastery Editor

**PR #33**

新增：

- `parameter_policy_schema.py`

所有 backend schema 通过 generated fragment 获得：

- Optimization Mode
- Optimizer Profiles
- Components
- Train/Freeze
- primary LR/Profile
- fallback LR/Profile

避免十套页面手工漂移。

---

## 9.3 Step 7C — Frontend Bootstrap + Runtime Readiness

**PR #34**

完成：

```text
Standard form
→ exact-or-fail bootstrap
→ Component editor
→ debounced Preview
→ runtime_ready
→ Start gate
```

任何 Component edit 都先使旧 Preview 失效，避免 stale green state。

Backend `/api/run` 仍会重新验证，因此 frontend gate 不是唯一安全边界。

---

## 9.4 Step 7D — Presets + User Documentation

**PR #35**

新增四个 curated Component preset：

- Flux Full — Muon + AdamW fallback
- Anima Full — DiT Muon + AdamW fallback
- SDXL Full — split LR
- SDXL LoRA — selective U-Net

同时增加：

- user guide；
- README entry；
- preset exact-merge semantics。

---

## 9.5 Step 7E — Full Regression Closure

**PR #36**

建立十 backend release fixture matrix。

验证：

- Standard missing/explicit/stale hidden policy 三态等价；
- Standard → Component bootstrap；
- Preview；
- sidecar；
- rehydrate；
- reprepare；
- canonical policy identity；
- target availability；
- blocker registry closure。

---

## 9.6 Step 7F — Acceptance Closure

**PR #37 + #38**

重点修复 Basic LoRA acceptance / rehydrate round-trip，完成 Step 7 release closure。

---

# 10. 发布后 Hardening

Step 7 结束后，实际 GUI 与真实训练暴露出一批“基础架构正确，但边界条件还不够硬”的问题。PR #39–#46 用于把这些问题彻底收口。

## 10.1 PR #39–#44：GUI / lifecycle hardening

包括：

- Component editor polish；
- Multi-Caption default 修复；
- legacy Schemastery discriminated union 稳定化；
- Profile key / rename / delete action；
- profile name 与 optimizer type 视觉分离；
- generated asset MIME；
- preset / history / reset / bootstrap generation lifecycle；
- stale readiness invalidation；
- GUI hidden-state 清理。

## 10.2 PR #45：CUDA device audit

修复：

```text
Accelerator.device = cuda
parameter.device    = cuda:0
```

在单进程 CUDA 下被错误判定为不同设备的问题。

现在只对 implicit CUDA ordinal 使用 `torch.cuda.current_device()` 做只读解析：

- `cuda` 可解析到当前 logical device；
- 显式 `cuda:0` 与 `cuda:1` 仍严格不等价；
- 不调用 `set_device()`；
- 不移动参数。

## 10.3 PR #46：Non-Anima semantics + cache/runtime hardening

这一轮主要完成：

### Bootstrap representability 与 runtime qualification 分离

Standard → Component bootstrap 只判断“能否无损表示”。

FP8、DeepSpeed、full BF16 等可以先成功进入 Component editor，然后由 Preview/Start 给出 runtime blocker，而不是阻止用户看到 Component state。

### SD3 / Flux / Chroma target semantics

修复：

- SD3 target/T5 wiring；
- Flux/Chroma top-level T5 override；
- Standard partial-cache compatibility；
- `train_t5xxl` exact raw key/value semantics。

### Text Encoder partial cache lifecycle

解决了：

```text
cache 建立早于 policy session
→ tokenization hint 不知道最终 CLIP Train 状态
```

现在通过 validated policy sidecar 做 pre-routing Train hint：

- all TE Frozen → full cache；
- CLIP live → partial cache；
- T5 Train + cache → 提前 fail-closed。

### SD3 CLIP-L / CLIP-G co-residency

因为 SD3 live encoding 会 `torch.cat` CLIP-L/G 输出，所以即使只训练其中一个 CLIP，也不能让另一个留在 CPU。

当前 contract：

- 任一 CLIP Train → L/G co-resident；
- 两个都 Frozen → 一起 offload；
- CPU runtime smoke 真实执行 `Accelerator.prepare()`；
- CUDA GPU matrix 增加 `runtime:sd3-clip-pair-prepare`。

### exact `train_t5xxl` wire semantics

真实 upstream 只认：

```text
train_t5xxl=True
```

以下都保持 inert / false：

```text
train_t5xxl=true
TRAIN_T5XXL=True
train_t5xxl =True
 train_t5xxl=True
train_t5xxl=True 
```

DTS 不再帮用户“纠正”这些值后意外开始训练 T5。

---

# 11. 当前成品是什么

当前 Parameter Policy 已经不是 experimental routing prototype，而是一个完整的 opt-in Component training system。

## 11.1 用户层

用户可以：

- 选择 Standard / Component；
- 从 Standard 自动 bootstrap；
- 建立多个 Optimizer Profile；
- 给不同 component 设置 Train/Freeze；
- 设置独立 LR；
- 设置 fallback optimizer + fallback LR；
- Preview runtime readiness；
- 使用 curated presets；
- Export / Import Component bundle；
- Rehydrate；
- Start；
- Save / Resume。

## 11.2 Host 层

Host 已经拥有：

- canonical policy v1；
- component registry；
- target availability；
- optimizer capability registry；
- bootstrap representability；
- runtime blockers；
- sidecar identity；
- editor metadata；
- Preview / Start validation。

## 11.3 Runtime 层

Runtime 已经拥有：

- real parameter scan；
- alias conflict audit；
- primary/fallback/frozen routing；
- Runtime Spec；
- CompositeOptimizer；
- CompositeLRScheduler；
- requires-grad ownership；
- Accelerate prepare audit；
- component LR logging；
- checkpoint manifest v2；
- save/resume identity；
- ScheduleFree lifecycle；
- Muon eligibility/fallback。

## 11.4 Backend 层

baseline single-process Component Start 已覆盖 10 个 backend。

---

# 12. 当前 Optimizer 状态：不要简单写成“只有 AdamW / Muon”

当前代码实际上已经有多种 optimizer runtime constructor。

## 12.1 Registry 标记为 supported，且 runtime builder 已实现

- AdamW
- AdamW8bit
- PagedAdamW8bit
- PagedAdamW
- PagedAdamW32bit
- Lion
- Lion8bit
- PagedLion8bit
- SGDNesterov
- SGDNesterov8bit
- RAdamScheduleFree
- AdamWScheduleFree
- SGDScheduleFree
- Muon

所以“AdamW 和 Muon 之外完全没有兼容”并不准确。

## 12.2 但它们的成熟度并不完全相同

需要区分：

```text
registry supported
≠
runtime constructor exists
≠
CPU smoke exists
≠
physical CUDA matrix passed
≠
10 backend real-trainer resume matrix passed
≠
长期 release-qualified
```

尤其 bitsandbytes / Lion / SGD / ScheduleFree 等，需要持续维护真实 CUDA evidence 和 dependency-version compatibility。

## 12.3 当前 restricted optimizers

以下已有 capability metadata，但仍明确 restricted：

- DAdaptation
- DAdaptAdamPreprint
- DAdaptAdam
- DAdaptAdaGrad
- DAdaptAdan
- DAdaptAdanIP
- DAdaptLion
- DAdaptSGD
- Prodigy
- AdaFactor
- prodigyplus.ProdigyPlusScheduleFree

主要问题不是“怎么 import 类”，而是：

- adaptive LR semantics；
- profile-owned LR；
- external scheduler interaction；
- checkpoint identity；
- group LR support；
- schedule-free lifecycle。

## 12.4 当前 planned

- `pytorch_optimizer.CAME`
- `Custom`

Custom optimizer 不能直接照搬 Standard 的 arbitrary-import 逻辑，因为 Component runtime 需要先知道 capability、LR semantics、scheduler contract 与 checkpoint topology。

---

# 13. 仍未完成 / 仍被明确阻止的功能

下面是当前真正应该进入后续开发计划的内容。

---

## 13.1 full BF16 / full FP16 Component qualification

这是高优先级缺口。

当前 `full_bf16` / `full_fp16` 仍是 qualification blocker。

原因不是 schema 无法表达，而是尚未完成真实 CUDA 下：

- model/gradient dtype ownership；
- CompositeOptimizer；
- GradScaler / non-GradScaler 分支；
- Accelerate prepare；
- optimizer state dtype；
- save/resume；
- mixed child optimizer；
- backend-specific cast behavior。

### 建议下一步

先做 **full BF16**，因为它不涉及 FP16 GradScaler 的额外复杂度。

Qualification 至少需要：

1. synthetic CompositeOptimizer CUDA case；
2. AdamW；
3. Muon + AdamW fallback；
4. one LoRA backend；
5. one Full backend；
6. Anima；
7. fresh → checkpoint → resume；
8. dtype/device audit；
9. optimizer state round-trip。

full BF16 完成后再单独处理 full FP16 / GradScaler。

---

## 13.2 FP8 base / FP8 base U-Net

当前仍 blocker：

- `fp8_base`
- `fp8_base_unet`

需要验证：

- FP8 base parameter identity 是否稳定；
- trainable adapter / full-model parameters 的实际 dtype；
- CompositeOptimizer 参数归属；
- prepare/device；
- checkpoint save/resume；
- Flux/SD3/Anima backend-specific FP8 path。

---

## 13.3 compile

仍未 qualification：

- Accelerate Dynamo `torch_compile`
- Anima per-block `compile`

需要确认 compile 前后：

- parameter identity；
- optimizer parameter references；
- adapter metadata；
- checkpoint hooks；
- resume topology。

---

## 13.4 DeepSpeed / sharded optimizer

仍然 fail-closed。

DeepSpeed 会改变 optimizer/distributed ownership，不能把 CompositeOptimizer 单纯当作普通 optimizer 塞进去就视为完成。

需要单独设计：

- DeepSpeed ownership model；
- child optimizer state；
- ZeRO stage 行为；
- save/resume；
- manifest identity；
- distributed parameter audit。

---

## 13.5 fused optimizer / fused backward

仍未支持：

- `fused_backward_pass`
- `fused_optimizer_groups`
- `blockwise_fused_optimizers`

这些模式本身会改变 optimizer 创建与 step ownership，因此需要 runtime-level 实现，不只是“解除 blocker”。

---

## 13.6 offload / block swap

仍未 qualification：

- CPU checkpoint offload；
- Unsloth checkpoint offload；
- Flux/Anima block swap；
- double/single block swap。

需要验证训练过程中 parameter residency 动态变化时，device audit 与 optimizer ownership 的正确边界。

---

## 13.7 explicit multi-GPU / DDP

当前显式多 GPU 仍 fail-closed。

需要：

- DDP；
- per-rank policy identity；
- optimizer child state；
- Accelerate prepare；
- checkpoint；
- resume；
- logical CUDA ordinal；
- distributed sampler / accumulation；
- potentially FSDP/ZeRO separate work。

---

# 14. Optimizer 方面下一步真正缺什么

---

## 14.1 “支持更多 optimizer”应拆成三类

### A. 已有 constructor，但需要加强 qualification

例如：

- bitsandbytes AdamW/Lion；
- Lion；
- SGD Nesterov；
- ScheduleFree。

后续工作重点：

- real CUDA artifact；
- dependency pin；
- mixed child combinations；
- fallback combinations；
- checkpoint/resume；
- backend matrix。

### B. restricted optimizer

DAdapt / Prodigy / AdaFactor / ProdigyPlusScheduleFree 需要先定义 **Component semantics**，再谈 runtime。

### C. arbitrary custom optimizer

需要一个 capability declaration contract，而不能继续让 arbitrary import 绕过 Parameter Policy 的 ownership model。

---

## 14.2 Muon + fallback 对 Standard 模式的兼容

这是一个当前 **明确没有完成的产品语义**。

现在：

- Component 模式拥有显式 primary/fallback routing；
- Standard 模式保持历史 sd-scripts 单 optimizer 路径；
- Standard 没有 Parameter Policy 的 per-parameter fallback profile 概念；
- Parameter Policy 也明确禁止使用 Muon 内部隐藏的 AdamW fallback 参数作为 Component ownership 替代品。

因此当前若希望“Standard 也能配置 Muon + AdamW fallback”，需要新增一个明确设计，而不是偷偷复用 Component sidecar。

### 可选方案 A：Standard 专用 Muon compatibility layer

在 Standard optimizer UI 中提供：

```text
Optimizer = Muon
Fallback = AdamW
Muon LR
Fallback LR
Muon args
AdamW args
```

然后由 Standard compiler 生成一个 **legacy-compatible two-route runtime**。

风险：

- Standard 不再是真正的历史单 optimizer path；
- 需要重新定义 save/resume identity；
- 需要处理所有 trainer 的 parameter eligibility。

### 可选方案 B：Standard UI 提供“升级到 Component”快捷入口

Standard 保持原样。

选择 Muon + fallback 时引导用户：

```text
Standard
→ exact bootstrap
→ Component
→ autogenerated Muon + AdamW fallback profiles
```

优点：

- 不复制 routing/runtime；
- 不破坏 Standard；
- checkpoint semantics 已经存在。

从当前架构一致性看，**方案 B 更符合已有设计原则**；如果产品目标必须要求“Standard 内部原生支持 Muon+fallback”，则需要把它作为独立兼容项目，而不能视为简单 UI 功能。

---

# 15. GUI / UX 仍可继续优化

当前 Component editor 已可用，但仍有进一步产品化空间。

## 15.1 legacy optimizer / LR ownership 表达

现有 Component editor 是 intersect 到原 backend schema 上的。

因此后续可以进一步明确：

- Component 模式下 legacy optimizer 字段是否应 disable / hide；
- global legacy LR 是否应显示为 non-authoritative；
- scheduler controls 哪些仍然有效；
- 哪些字段属于 bootstrap source、哪些属于 runtime authority。

目标是减少用户看到“两套 optimizer/LR 控件”时的歧义。

## 15.2 optimizer-specific typed args

目前除 Muon 有更明确的参数约束外，很多 Optimizer Profile args 仍使用通用 dict/literal 编辑。

后续可按 capability 生成 typed UI：

- AdamW betas / eps / weight decay；
- Lion；
- SGD；
- bitsandbytes；
- ScheduleFree；
- restricted optimizer 在解锁后各自专用参数。

## 15.3 fallback capability-aware UI

当前 fallback 语义主要服务于 eligibility optimizer（目前核心是 Muon）。

后续可进一步让 GUI：

- 只在 primary optimizer 需要 eligibility 时显示 fallback；
- 根据 fallback optimizer capability 限制可用字段；
- Profile 类型变化时提供更清楚的 stale fallback migration 提示。

---

# 16. 当前仍属于“语义上不支持”的功能

以下不是简单做 GPU qualification 就能解锁，而是需要扩展 Component schema/ownership semantics：

- LoRA+ parameter-level LR；
- regex-specific LR；
- SD/SDXL LoRA block LR；
- SDXL Full block LR；
- `scale_weight_norms` 对 frozen adapter 的写入；
- DreamBooth dynamic `stop_text_encoder_training`；
- preloaded adapter + TE cache 的 conditioning 时序冲突；
- Anima Qwen-only；
- unreviewed custom `network_module`。

这些应和 full BF16 / FP8 等“只是尚未 qualification”的功能分开管理。

---

# 17. 建议的下一阶段路线图

下面是基于当前架构的建议顺序，不属于已经完成的历史 Step。

## Next-1 — full BF16 Component qualification

目标：

- 先解除 `full_bf16` blocker；
- 不同时做 full FP16；
- AdamW + Muon/fallback；
- real CUDA + backend fresh/resume。

## Next-2 — Optimizer qualification expansion

目标：

把“registry supported”进一步提升为有明确 physical GPU evidence 的 release matrix。

优先顺序建议：

1. AdamW8bit / PagedAdamW；
2. Lion / Lion8bit；
3. SGD Nesterov；
4. ScheduleFree trio；
5. mixed profiles；
6. Muon + non-AdamW fallback。

## Next-3 — Restricted optimizer semantics

逐个处理：

- AdaFactor；
- DAdapt；
- Prodigy；
- ProdigyPlusScheduleFree。

不要一次性开放整个 restricted registry。

## Next-4 — Standard / Component optimizer UX closure

包括：

- Component 模式 legacy optimizer/LR ownership；
- scheduler controls；
- fallback UI；
- Standard → Component Muon/fallback migration；
- 决定是否真的实现 Standard 原生 Muon+fallback。

## Next-5 — advanced execution qualification

按独立 PR：

- full FP16；
- FP8；
- compile；
- block swap/offload；
- explicit multi-GPU；
- DeepSpeed；
- fused runtime。

---

# 18. 后续功能解锁的验收标准

任何现有 blocker 不应因为“本地跑通一次”就删除。

至少要求：

1. **host contract**：配置/Preview/Start 无语义漂移；
2. **runtime contract**：真实 optimizer ownership；
3. **device contract**：prepare 后 device audit；
4. **requires-grad contract**；
5. **fresh training**；
6. **checkpoint save**；
7. **resume**；
8. **topology identity**；
9. **scheduler identity**；
10. **Standard regression**；
11. **real CUDA evidence**；
12. 如果改变 backend-specific lifecycle，再增加对应 backend real-trainer case。

---

# 19. 关键源码地图

## Host / policy

- `mikazuki/optimizer_profiles.py`
- `mikazuki/parameter_policy.py`
- `mikazuki/parameter_policy_bootstrap.py`
- `mikazuki/parameter_policy_compat.py`
- `mikazuki/model_component_profiles.py`
- `mikazuki/parameter_routing.py`

## Runtime

- `mikazuki/parameter_policy_runtime.py`
- `mikazuki/parameter_policy_torch.py`
- `mikazuki/parameter_policy_trainer.py`
- `mikazuki/parameter_policy_matrix.py`

## GUI / request

- `mikazuki/parameter_policy_editor.py`
- `mikazuki/parameter_policy_schema.py`
- `mikazuki/frontend_training_patch.py`
- `mikazuki/training_config.py`
- `mikazuki/training_gui_args.py`
- `mikazuki/training_gui_semantics.py`
- `mikazuki/training_rehydrate.py`
- `mikazuki/training_request.py`

## Trainer integration

- `scripts/stable/train_network.py`
- `scripts/stable/sdxl_train_network.py`
- `scripts/stable/train_db.py`
- `scripts/stable/sdxl_train.py`
- `scripts/dev/train_network.py`
- `scripts/dev/flux_train_network.py`
- `scripts/dev/sd3_train_network.py`
- `scripts/dev/flux_train.py`
- staged Anima runtime patches

## Qualification

- `.github/workflows/anima-qwen3-review.yml`
- `.github/workflows/parameter-policy-gpu-matrix.yml`
- `tools/run_parameter_policy_gpu_matrix.py`
- `tools/run_parameter_policy_backend_gpu_matrix.py`
- `tests/test_parameter_policy*.py`

---

# 20. 当前项目状态总结

截至 `main@9d9446a`：

### 已完成

- 7 个主开发步骤；
- 26 个相关合并 PR；
- Standard / Component 双路径；
- canonical policy + sidecar；
- model component registry；
- real parameter routing；
- Muon eligibility + explicit fallback；
- CompositeOptimizer / CompositeLRScheduler；
- 10 backend trainer integration；
- Accelerate ownership/device audit；
- checkpoint manifest v2；
- save/resume identity；
- generated GUI editor；
- exact Standard → Component bootstrap；
- Preview/Start gate；
- presets；
- export/import/rehydrate；
- cache / target / T5 wire semantics hardening；
- CPU runtime smoke；
- manual real-CUDA qualification harness。

### 仍需开发或 qualification

- full BF16；
- full FP16；
- FP8；
- compile；
- DeepSpeed；
- fused paths；
- offload / block swap；
- explicit multi-GPU/DDP；
- restricted optimizers；
- planned CAME / Custom；
- supported non-AdamW/Muon optimizer 的持续 physical-GPU / backend evidence；
- Standard 模式 Muon + explicit fallback 的产品语义；
- optimizer-specific typed GUI；
- Component 模式 legacy optimizer/LR ownership UX；
- 仍属于 semantic blockers 的高级 LR / dynamic ownership 功能。

---

## 21. 维护原则

后续开发继续遵守：

> **先定义 ownership 语义，再写 runtime；先有 runtime contract，再做 GUI；先有真实 qualification evidence，再解除 blocker。**

尤其不要为了“兼容更多配置”而：

- 自动关闭用户选项；
- 把不支持的 Component 配置偷偷退回 Standard；
- 把 fallback 参数塞回 Muon 内部隐藏 fallback；
- 让两个 optimizer 同时拥有同一参数；
- 用一次成功训练替代 save/resume / device / ownership qualification；
- 为了 GUI 方便而改变 Standard legacy semantics。


---

# 22. 当前参考训练需求：Anima 2.9B 多组件训练工作负载

前面的章节说明了 Parameter Policy 已经实现了什么。本节开始记录它**实际需要服务的参考工作负载**。

这里的目的不是把某一套训练 recipe 固化成 DTS 的唯一用法，而是给后续开发和 qualification 一个足够具体的 acceptance target：如果一个功能声称支持 Component-wise training，它至少不能破坏下面这类真实需求。

## 22.1 当前八阶段数据 / 分辨率计划

当前训练计划使用 8 个阶段：

| 阶段 | 数据规模 | Epoch | 分辨率 |
| --- | ---: | ---: | ---: |
| Stage 1 | 1.2M | 2 | 1024 |
| Stage 2 | 3.0M | 6 | 1024 |
| Stage 3 | 1.5M | 3 | 1024 |
| Stage 4 | 1.0M | 2 | 1536 |
| Stage 5 | 1.2M | 3 | 1536 |
| Stage 6 | 1.5M | 4 | 1536 |
| Stage 7 | 1.0M | 3 | 1536 |
| Stage 8 | 0.5M | 2 | 1536 |

当前计划的全局 batch 目标为 256。

需要特别区分两件事：

- **数据量 / epoch / 分辨率阶段计划已经形成明确工作基线；**
- **只有 Stage 1 的 Component optimizer/LR routing 已经被确认并写成仓库 preset。**

Stage 2–8 曾经讨论过若干 optimizer/LR 数值，但这些数值尚不应作为 DTS 工程 contract 写死。后续如果训练策略再次调整，Parameter Policy 应允许用户通过 policy/preset 表达，而不是要求改 runtime 代码。

## 22.2 已确认的 Stage 1 Component Policy

仓库中的：

`config/presets/component-anima-finetune-muon.toml`

就是当前 Stage 1 的可执行参考配置。

### Optimizer Profiles

`muon`：

- Type: `Muon`
- `momentum = 0.95`
- `weight_decay = 0.01`
- `weight_decouple = true`
- `nesterov = true`
- `ns_steps = 5`
- `ns_coeffs = "original"`
- `use_adjusted_lr = false`

`adamw`：

- Type: `AdamW`
- `betas = [0.9, 0.95]`
- `eps = 1e-8`
- `weight_decay = 0.0`

### Component routing

| Component | Train | Primary Optimizer | Primary LR | Fallback | Fallback LR |
| --- | --- | --- | ---: | --- | ---: |
| `dit.self_attention` | Yes | Muon | 1e-4 | AdamW | 2.5e-5 |
| `dit.cross_attention` | Yes | Muon | 1e-4 | AdamW | 2.5e-5 |
| `dit.mlp` | Yes | Muon | 1e-4 | AdamW | 2.5e-5 |
| `dit.modulation` | Yes | AdamW | 2.5e-5 | — | — |
| `dit.base_other` | Yes | AdamW | 2.5e-5 | — | — |
| `dit.llm_adapter` | Yes | AdamW | 2e-6 | — | — |
| `qwen3` | No | — | — | — | — |

这张表体现了 Parameter Policy 最初真正需要解决的核心问题：

> 同一次 full-model training 中，不同 architectural component 需要不同 optimizer、不同 LR，并且 Muon 只拥有它真正 eligible 的参数；其余参数必须由显式 AdamW fallback 接管。

## 22.3 Stage 1 不是“多个 LR group”这么简单

如果只把需求描述成“分层学习率”，会遗漏至少四个关键语义：

1. Self/Cross Attention 和 MLP 的**主 optimizer 是 Muon**；
2. 同一个 component 内并不是所有参数都满足 Muon eligibility；
3. ineligible 参数必须进入**另一个真实 child optimizer**；
4. Modulation / Base Other / LLM Adapter 从一开始就直接归 AdamW，而不是先进入 Muon 再由 Muon 内部偷偷处理。

因此 Stage 1 的真正 runtime topology 是：

```text
CompositeOptimizer
├─ Muon
│  ├─ dit.self_attention eligible params @ 1e-4
│  ├─ dit.cross_attention eligible params @ 1e-4
│  └─ dit.mlp eligible params @ 1e-4
│
└─ AdamW
   ├─ dit.self_attention fallback params @ 2.5e-5
   ├─ dit.cross_attention fallback params @ 2.5e-5
   ├─ dit.mlp fallback params @ 2.5e-5
   ├─ dit.modulation @ 2.5e-5
   ├─ dit.base_other @ 2.5e-5
   └─ dit.llm_adapter @ 2e-6

qwen3
└─ frozen
```

这也是为什么 Parameter Policy 不能用“一个 optimizer + 参数组”完全替代：这里存在真正的 multi-optimizer ownership。

---

# 23. 从当前训练需求反推的工程硬性要求

下面这些不是可选 UX polish，而是当前训练方案对 runtime 的硬要求。

## 23.1 Component 必须成为 optimizer/LR authority

进入 Component 模式后：

- legacy `optimizer_type` 不能重新接管；
- legacy global LR 不能覆盖 component LR；
- legacy optimizer args 不能偷偷改变 child optimizer；
- scheduler 只能按照 Parameter Policy runtime contract 与 child optimizer capability 构造。

因此 Component 模式下保留 legacy 字段只能有两种意义：

1. Standard → Component bootstrap source；
2. 与 Parameter Policy 无关、且仍由 trainer 拥有的合法公共配置。

它们不能同时成为第二套 optimizer/LR authority。

## 23.2 Qwen3 与 LLM Adapter 必须是独立 ownership domain

Stage 1 明确要求：

- `dit.llm_adapter` Train；
- `qwen3` Freeze。

所以：

```text
LLM Adapter != Qwen3
```

不能因为两者都与语言条件相关，就把它们合并为一个“Text Encoder”开关或共享 LR。

这也是 Anima component profile 中必须保留独立 `dit.llm_adapter` 与 `qwen3` component 的原因。

## 23.3 Muon eligibility 必须发生在真实 parameter routing 层

Muon 的 eligibility 不能通过 component 名称粗略判断。

即使：

```text
component = dit.self_attention
```

其中仍可能存在：

- 非二维参数；
- bias；
- normalization 参数；
- 其它不符合 `model_hidden_2d_weight` 的参数。

所以 eligibility 必须发生在 Parameter identity scan / routing 层，而不是 GUI 或 preset 层。

## 23.4 Fallback 是 ownership，不是 error recovery

Fallback 的语义不是：

```text
Muon.step() 失败
→ 再试 AdamW
```

而是：

```text
routing time
→ 参数 A 属于 Muon
→ 参数 B 不符合 Muon eligibility
→ 参数 B 从一开始就属于 AdamW
```

因此 fallback 必须参与：

- Runtime Spec；
- topology fingerprint；
- checkpoint state；
- LR logging；
- save/resume identity；
- ownership audit。

## 23.5 阶段切换允许改变 policy，但不能伪装成同一 optimizer topology

八阶段训练天然可能改变：

- Train/Freeze；
- optimizer type；
- LR；
- fallback；
- Adapter 是否继续训练。

如果这些变化改变 Runtime Spec，则 checkpoint topology fingerprint 也会改变。

因此必须区分：

### 同阶段 resume

要求：

- policy identity 一致；
- optimizer topology 一致；
- scheduler identity 一致；
- 可以恢复 optimizer/scheduler state。

### 跨阶段 transition

如果新阶段改变 optimizer topology，则不能把旧 optimizer state 当作“同一训练状态”强行加载。

正确语义应是：

```text
previous stage model checkpoint
→ new stage policy
→ new Runtime Spec
→ new optimizer/scheduler state
```

除非未来专门设计并验证 optimizer-state migration contract。

## 23.6 mixed BF16 与 full BF16 必须被当成不同 execution mode

当前 Stage 1 preset 使用：

`anima_precision_mode = "mixed_bf16"`

这与 `full_bf16` 不是一回事。

普通 mixed BF16 可以是：

```text
model parameters: FP32
forward/autocast: BF16
optimizer state: implementation-defined / typically FP32-oriented
```

而 full BF16 会改变参数和梯度本身的 dtype ownership。

因此：

> mixed BF16 跑通不能作为 full BF16 qualification evidence。

这条边界必须持续保持。

---

# 24. 当前 qualification 证据应该怎样理解

当前仓库已经有两类真实 CUDA harness，但它们验证的是不同维度。

## 24.1 Shared optimizer/runtime GPU matrix

`tools/run_parameter_policy_gpu_matrix.py`

当前负责验证：

- registry 中所有 `supported` optimizer 的 synthetic CUDA step；
- optimizer state_dict reload；
- `AdamW + AdamWScheduleFree` mixed child；
- `Muon + AdamW` mixed child；
- AdamW FP16 autocast；
- AdamW BF16 autocast；
- single-process `Accelerator.prepare()` device audit；
- SD3 CLIP-L / CLIP-G co-residency seam。

这是 **shared runtime evidence**。

它能证明：

> CompositeOptimizer / constructor / state reload / 某些 precision wrapper 在真实 CUDA 上基本工作。

但它不能证明：

> 某个真实 backend 的完整 trainer lifecycle 已经支持该 optimizer / feature。

## 24.2 Backend GPU matrix

`tools/run_parameter_policy_backend_gpu_matrix.py`

要求对 10 个已开放 backend：

- fresh Component training；
- 生成 checkpoint manifest v2；
- resume；
- resume 后 manifest identity 不变。

这是 **backend/trainer evidence**。

它覆盖的重点是：

- trainer integration；
- ownership/device contract；
- save/resume；
- checkpoint identity。

## 24.3 两类 matrix 不能互相替代

必须维持：

```text
shared optimizer CUDA evidence
+
backend trainer evidence
+
feature-specific qualification
=
release claim
```

例如：

- Muon synthetic CUDA step 成功，不等于 Anima full BF16 + Muon 已支持；
- Anima baseline fresh/resume 成功，不等于 Lion8bit 在 Anima 上已经完成 release qualification；
- BF16 autocast case 成功，不等于 full BF16 参数训练已支持。

## 24.4 当前 `precision:adamw-bf16` 明确不是 full BF16 test

现有 GPU matrix 的参数创建方式仍是：

```python
torch.nn.Parameter(... dtype=torch.float32)
```

随后只在：

```python
torch.autocast(device_type="cuda", dtype=torch.bfloat16)
```

中执行 forward。

因此该 case 测的是：

> FP32 parameters + BF16 autocast

而不是：

> BF16 parameters + BF16 gradients + Component optimizer state + Accelerate + save/resume

所以它不能用于删除 `full_bf16` blocker。

---

# 25. 当前能力与参考训练需求的对应表

| 需求 | 当前状态 | 说明 |
| --- | --- | --- |
| Anima Full Component Start | 已实现 baseline | `anima-finetune` 在 10-backend matrix 内 |
| Self/Cross/MLP 独立 component | 已实现 | backend-owned component profile |
| Modulation / Base Other 独立 LR | 已实现 | Component policy owns LR |
| LLM Adapter 独立 LR | 已实现 | 与 Qwen3 分离 |
| Qwen3 Freeze | 已实现 | Stage 1 preset 明确 `train=false` |
| Muon primary | 已实现 | supported capability + runtime constructor |
| AdamW explicit fallback | 已实现 | router-owned fallback child |
| Muon + AdamW 同时存在 | 已实现 shared runtime | GPU matrix 有 `mixed:muon-adamw` |
| save/resume topology identity | 已实现 | checkpoint manifest v2 + topology fingerprint |
| ordinary mixed BF16 | 当前 baseline 路径 | Stage 1 preset 使用 mixed BF16 |
| full BF16 | **仍 blocked** | 缺真实 full-BF16 Component qualification |
| full FP16 | **仍 blocked** | 还涉及 GradScaler 等额外 contract |
| FP8 | **仍 blocked** | dtype / identity / backend path 未 qualification |
| explicit multi-GPU/DDP | **仍 blocked** | distributed ownership 未 qualification |
| DeepSpeed | **仍 blocked** | optimizer ownership 会被重新接管 |
| fused optimizer paths | **仍 blocked** | step ownership 语义不同 |
| block swap/offload | **仍 blocked** | parameter residency 动态变化 |
| AdamW/Muon 之外 supported optimizer | constructor + shared CUDA matrix 可覆盖 | 仍需持续 backend/dependency evidence |
| restricted optimizer | **语义未开放** | 不能只靠 GPU smoke 解锁 |
| Standard 原生 Muon + explicit fallback | **产品语义未实现** | 当前应优先通过 Standard → Component 表达 |

---

# 26. Next-1 应细化为 full BF16 qualification 项目

现有路线图把 full BF16 放在第一优先级是正确的，但实现时不能直接“删除 blocker 然后跑一次”。

建议拆成以下独立 gate。

## 26.1 Gate A — synthetic true-BF16 parameter runtime

新增专门 case，参数本身必须：

```python
dtype=torch.bfloat16
```

而不是只用 autocast。

至少覆盖：

- AdamW；
- Muon；
- Muon + AdamW fallback；
- `step()`；
- `zero_grad()`；
- state_dict；
- load_state_dict；
- resume 后再次 step。

需要显式记录：

- parameter dtype；
- gradient dtype；
- child optimizer state dtype；
- loss/output dtype；
- device。

## 26.2 Gate B — Accelerate prepare

使用真正 BF16 parameters 验证：

```text
model
+
CompositeOptimizer
→ Accelerator.prepare()
→ finalize_after_prepare()
```

然后重新执行：

- parameter ownership audit；
- device audit；
- requires-grad audit。

不能假设 mixed-BF16 下的 prepare 行为与 full BF16 相同。

## 26.3 Gate C — one LoRA + one Full backend

至少选择：

- 一个 LoRA backend；
- 一个 Full backend。

原因是二者的 trainable parameter ownership 不同：

- LoRA 主要训练 adapter；
- Full training 直接训练 base-model parameters。

full BF16 对这两类路径的影响不能由同一个 synthetic test 替代。

## 26.4 Gate D — Anima Stage 1 reference workload

必须使用与 Stage 1 相同的 ownership topology：

- Self/Cross/MLP → Muon + AdamW fallback；
- Mod/Base → AdamW；
- LLM Adapter → AdamW；
- Qwen3 frozen。

至少完成：

```text
fresh
→ real forward/backward
→ optimizer step
→ save
→ checkpoint manifest
→ resume
→ another step
```

并验证：

- Qwen3 没有获得梯度；
- fallback 参数确实归 AdamW；
- Muon child 只拥有 eligible 参数；
- LR logging 与 policy 一致；
- resume 后 topology 不漂移。

## 26.5 Gate E — blocker removal policy

当前 `full_bf16` blocker 是全局语义检查。

解除时不能出现：

```text
只验证 Anima
→ 全部 10 backend 一起自动解锁 full_bf16
```

有两种安全做法：

### 方案 1：10 backend 全量 qualification 后再删除全局 blocker

优点：

- 逻辑简单；
- 不新增 feature matrix。

缺点：

- 首次解锁成本高。

### 方案 2：引入 backend × execution-feature qualification registry

例如：

```text
anima-finetune:
  full_bf16: qualified

sdxl-finetune:
  full_bf16: qualified

flux-finetune:
  full_bf16: blocked
```

然后 Preview/Start 根据：

```text
train_type + feature
```

判断，而不是只看全局字段名。

如果后续 FP8、compile、block swap、multi-GPU 都会逐 backend 分批开放，那么长期看方案 2 更可扩展。

---

# 27. Next-2 应细化为 Optimizer qualification expansion

当前文档已经正确区分：

```text
registry supported
!=
release-qualified everywhere
```

后续应把这个差异变成显式工程矩阵。

## 27.1 Shared runtime 层

当前 GPU matrix 已经会枚举所有 `component_support == "supported"` optimizer。

继续要求每个 supported optimizer 至少具备：

- real CUDA construction；
- one step；
- state_dict；
- reload；
- resumed step；
- dependency/version metadata。

## 27.2 Mixed child 层

不能只测单 optimizer。

至少需要有代表性的组合：

- Muon + AdamW；
- Muon + AdamW8bit；
- Muon + Lion；
- AdamW + ScheduleFree；
- 同一 Profile 多 LR groups；
- 多 Profile 同 optimizer type；
- fallback 与 primary 使用不同 dependency family。

这里的目标不是穷举所有笛卡尔积，而是验证 CompositeOptimizer 对不同 child lifecycle 的组合不会失效。

## 27.3 Backend 层

对准备宣称“正式支持”的 optimizer，需要至少在真实 trainer 中覆盖：

- one LoRA backend；
- one Full backend；
- save/resume。

如果 optimizer 有特殊 lifecycle，例如 ScheduleFree，还应额外覆盖：

- `train()` / `eval()`；
- sampling/save 前后状态；
- scheduler absence contract。

## 27.4 Dependency 层

bitsandbytes / lion-pytorch / schedulefree / pytorch-optimizer 的兼容不能只记录“能 import”。

qualification artifact 应记录：

- package version；
- torch version；
- CUDA runtime；
- GPU model；
- git SHA。

现有 GPU matrix 已经记录这些 metadata，应继续沿用。

---

# 28. Standard 模式 Muon + fallback 的决策边界

现阶段不要因为当前参考 workload 需要 Muon + fallback，就把这套 ownership 复制进 Standard。

当前架构最清楚的边界仍然是：

```text
Standard
= legacy single-optimizer semantics

Component
= explicit multi-optimizer / fallback ownership semantics
```

因此当前产品建议继续保持：

> 需要 Muon + explicit fallback 的用户，应通过 Standard → Component bootstrap 进入 Component，而不是在 Standard 内再实现一套 Parallel Parameter Policy。

只有在未来明确提出：

> Standard 必须原生支持 multi-optimizer fallback，同时又不能切换到 Component

时，才应该单独设计 Standard compatibility runtime。

否则会重新引入：

- 两套 ownership compiler；
- 两套 checkpoint topology；
- 两套 fallback semantics；
- 两套 GUI；
- 更难保证 Standard regression。

---

# 29. 后续文档和 release 状态应使用统一术语

为了避免再次出现“支持了到底是什么意思”的争议，后续 issue / PR / 文档建议统一使用以下状态词。

## Implemented

代码路径存在，host/runtime contract 已实现。

不自动意味着真实 CUDA 已通过。

## Shared-CUDA-qualified

在 `run_parameter_policy_gpu_matrix.py` 这类 shared runtime matrix 上有真实 CUDA evidence。

不自动意味着某个具体 backend 已通过。

## Backend-qualified

真实 trainer 完成 fresh + save + resume，并产生可审计 artifact。

## Feature-qualified

某个高级 execution feature，例如 full BF16 / FP8 / compile，针对指定 backend 完成专门 qualification。

## Semantic-unsupported

当前 Parameter Policy schema/ownership 本身无法准确表达。

这类功能不能通过“多跑几个 GPU test”解锁。

## Restricted optimizer

optimizer 已注册，但其 LR/scheduler/ownership contract 尚未定义或尚未验证。

也不能仅凭 constructor 成功就改成 supported。

---

# 30. 当前最直接的开发顺序

从当前 reference workload 出发，后续工作顺序应收敛为：

1. **full BF16 true-parameter qualification**
   - 先补 shared CUDA case；
   - 再做 Accelerate；
   - 再做 LoRA/Full；
   - 最后做 Anima Stage 1 fresh/resume。

2. **决定 full BF16 blocker 的粒度**
   - 全 backend 一次性解锁；
   - 或建立 backend × feature qualification registry。

3. **扩大 supported optimizer 的 release evidence**
   - 不再把 constructor/synthetic smoke 等同于 backend support。

4. **继续 GUI ownership closure**
   - Component 模式下把 legacy optimizer/global LR 明确标成 bootstrap-only / non-authoritative；
   - optimizer-specific typed args；
   - fallback capability-aware UI。

5. **再进入 full FP16 / FP8 / compile / offload / multi-GPU / DeepSpeed / fused runtime**
   - 每个 feature 独立定义 ownership contract；
   - 独立 qualification；
   - 独立解除 blocker。

---

## 31. 当前 reference acceptance target

对于当前 Anima Stage 1，多组件训练达到“release-qualified”的最低闭环应是：

```text
preset load
→ canonical policy
→ Preview runtime_ready
→ parameter scan
→ routing
→ Muon eligibility split
→ explicit AdamW fallback
→ CompositeOptimizer
→ Accelerator.prepare
→ post-prepare audits
→ real forward/backward
→ optimizer step
→ component LR logging
→ save
→ checkpoint manifest v2
→ resume
→ topology/ownership identity check
→ second optimizer step
```

并且同时满足：

- `dit.self_attention` / `dit.cross_attention` / `dit.mlp` 的 eligible 参数只属于 Muon；
- 上述三个 component 的 ineligible 参数只属于 AdamW fallback；
- `dit.modulation` / `dit.base_other` / `dit.llm_adapter` 只属于 AdamW；
- `qwen3` 不属于任何 optimizer；
- 没有 orphan trainable parameter；
- 没有 duplicate ownership；
- 没有 implicit hidden fallback；
- checkpoint topology 与 resume 完全一致；
- Standard path 不受影响。

当前普通 mixed-BF16 Stage 1 已经具备表达和 baseline runtime 基础。

**full BF16 版本仍然必须保持 fail-closed，直到第 26 节的 qualification gate 完成。**
