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

# 26. full BF16 重新审计后的设计结论

本节替代旧版“one LoRA + one Full + Anima 即可解除 blocker”的方案。

重新对照当前 trainer、Parameter Policy runtime、PyTorch 2.7.0、Accelerate 1.6.0 与 pytorch-optimizer 3.10.0 后，确认旧方案虽然方向正确，但仍存在四个会导致错误放行的问题：

1. `full_bf16` 目前是全局 blocker，不能在只验证一个 backend 后全局删除；
2. backend qualification 与 optimizer qualification 不能混为一谈；
3. 当前 runtime contract 只审计 ownership / device / requires-grad，没有 dtype contract；
4. checkpoint manifest v2 没有区分 mixed BF16 与 full BF16 execution identity。

因此 full BF16 必须被定义为：

> **一个独立的 execution feature。**

Anima 2.9B Stage 1 是最高价值 reference workload，但不是全部 release coverage。

## 26.1 “覆盖所有模型”的准确含义

当前 Component release matrix 有 10 个 backend：

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

full BF16 项目完成时，**每个 backend 都必须有显式状态**，但不要求为了追求“10/10”而强行把不安全路径标成支持。

状态只允许：

- `pending`：存在潜在实现路径，但尚未完成 release qualification；
- `qualified`：exact-head 真实 CUDA evidence 已通过，可以开放；
- `unsupported`：当前 trainer contract 明确无法正确表达或代码路径缺失。

当前审计后的初始状态应为：

| Backend | 初始 full BF16 状态 | 原因 |
| --- | --- | --- |
| `sd-lora` | pending | generic NetworkTrainer 有 full-BF16 network cast，但未做 Component qualification |
| `sdxl-lora` | pending | 同上，仍需 SDXL conditioning/cache seam |
| `sd-dreambooth` | unsupported | 当前 `train_db.py` 只有 full-FP16 cast，没有对应 full-BF16 training-model cast |
| `sdxl-finetune` | pending | trainer 有 full-BF16 U-Net / TE cast |
| `sd3-lora` | pending | dev NetworkTrainer 有 full-BF16 path，但 SD3 multi-encoder seam 未验证 |
| `flux-lora` | pending | dev NetworkTrainer 有 full-BF16 path |
| `chroma-lora` | pending | 与 Flux trainer family 共享路径，但不能自动继承 qualification |
| `flux-finetune` | pending | trainer 有明确 full-BF16 Flux cast |
| `anima-lora` | pending | pinned Anima NetworkTrainer 有 full-BF16 path |
| `anima-finetune` | pending | pinned `anima_train.py` 有明确 DiT full-BF16 path |

**新增 backend 时默认没有 full-BF16 资格。**

不能按 trainer family 自动继承 `qualified`，避免未来新增模型因为“共用一个基类”被错误开放。

---

# 27. 不建立笛卡尔积：采用可分离 qualification contract

full BF16 的 release evidence 分成三层：

```text
optimizer/shared-runtime evidence
+
backend/trainer evidence
+
一条真实 cross-axis reference workload
=
full-BF16 release claim
```

不是：

```text
10 backends
× 所有 optimizers
× cache on/off
× TE on/off
× 所有 hyperparameters
```

后者不可维护，也会让 GPU qualification 成为长期开发瓶颈。

## 27.1 为什么当前 full BF16 可以分离 backend 与 optimizer qualification

在目前开放的 Component baseline 路径中：

- trainer 看到的是标准 `torch.optim.Optimizer` facade；
- Parameter Policy 统一暴露 `CompositeOptimizer`；
- normal `optimizer.step()` / `zero_grad()` / scheduler lifecycle 不按 child optimizer 类型改变 trainer 控制流；
- 会改变 step ownership 的 `fused_backward_pass`、blockwise/fused optimizers、DeepSpeed、block swap/offload 仍然独立 fail-closed。

因此 full BF16 可以安全拆成：

### A. Backend qualification

证明：

- 模型/adapter 被正确 cast；
- Accelerator.prepare 不破坏 ownership；
- trainer forward/backward/save/resume 正常；
- feature-specific dtype contract 成立。

### B. Optimizer qualification

证明：

- optimizer 接受真正 BF16 parameters；
- state 初始化与 step 正常；
- state_dict/load_state_dict 正常；
- CompositeOptimizer mixed child 正常。

### C. Cross-axis reference

至少用一个真实 workload 同时覆盖：

- 多 Component；
- Muon primary；
- AdamW explicit fallback；
- full BF16；
- save/resume。

当前 reference 就是 Anima 2.9B Stage 1。

只要 trainer baseline 继续保持 optimizer-agnostic，这种分离就成立。

如果未来某个 feature 让 trainer 根据 optimizer 类型改变 step/control flow，例如：

- fused backward；
- blockwise fused optimizer；
- optimizer-owned offload；
- DeepSpeed optimizer replacement；

则该 feature **不能复用 full-BF16 的可分离假设**，必须做 coupled qualification。

## 27.2 不允许“独立 feature 都 qualified，所以组合自动 qualified”

未来即使：

```text
full_bf16 = qualified
torch_compile = qualified
```

也不能自动推出：

```text
full_bf16 + torch_compile = qualified
```

execution feature 的组合必须显式审查。

当前 full BF16 开放期间，compile / FP8 / swap / offload / DeepSpeed / fused / multi-GPU 仍保持各自 blocker，因此不会产生组合误放行。

后续新增第二个可开放 execution feature 时，必须同时定义其与已有 feature 的组合策略，而不是默认可组合。

---

# 28. 新增独立 execution-feature qualification 层

不要继续把所有高级 runtime feature 都堆进 `parameter_policy_compat.py` 的一串 if。

新增：

`mikazuki/parameter_policy_execution.py`

该模块必须保持：

- host-side；
- torch-free；
- 不加载 model；
- 不构造 optimizer；
- 不依赖 GUI/server；
- 不修改 request/config；
- 纯函数 + 静态 qualification metadata。

## 28.1 最小公共 API

建议只暴露：

```python
parameter_policy_execution_blockers(
    canonical_policy,
    *,
    train_type,
    effective_config,
) -> list[str]

build_parameter_policy_execution_contract(
    *,
    train_type,
    effective_config,
) -> ParameterPolicyExecutionContract | None
```

以及测试需要的只读 qualification table。

不要设计动态 plugin registry，也不要引入 runtime registration。

## 28.2 full BF16 backend qualification table

在该模块中使用显式静态表：

```text
backend
→ pending / qualified / unsupported
→ reason
→ optional evidence case id
```

要求：

- key 集合必须与当前 10-backend Component release matrix 完全一致；
- unknown backend fail closed；
- `qualified` 必须有非空 evidence case id；
- `unsupported` 必须有明确 reason；
- qualification PR 只允许在 exact-head GPU evidence 成功后把 `pending` 改为 `qualified`。

不要把该状态塞进 `PARAMETER_POLICY_BACKEND_MATRIX`。

原因：

- baseline backend integration 与 execution feature qualification 是两层不同事实；
- 否则未来 FP8 / compile / multi-GPU 会把 baseline matrix 变成难以维护的大表。

## 28.3 optimizer full-BF16 gate 不加入 `OptimizerCapability` 字段

不要新增：

```text
supports_full_bf16
supports_fp8
supports_compile
supports_deepspeed
...
```

否则 `OptimizerCapability` 会快速变成 feature flag 垃圾场。

full BF16 的 optimizer qualification 应留在 execution-feature 层。

第一批只开放：

- `AdamW`
- `Muon`
- `Muon + AdamW` mixed children

这正好覆盖当前 Anima Stage 1。

其他 baseline `supported` optimizer：

- bitsandbytes family；
- Lion；
- SGDNesterov；
- ScheduleFree；

继续在普通 Component baseline 中可用，但 **full BF16 下保持 fail-closed**，直到专门完成 full-BF16 optimizer qualification。

这不会降低现有 baseline capability。

## 28.4 AdamW 不能按名字过宽放行

当前非 Muon Optimizer Profile args 是 JSON-safe generic dict。

所以：

```text
type = AdamW
```

不代表只有一种 execution path。

以下参数会改变 kernel / execution semantics：

- `fused`
- `foreach`
- `capturable`
- `differentiable`

第一批 full-BF16 qualification 不应自动开放这些显式 modifier。

允许的 AdamW profile 应至少覆盖当前 reference 所需：

- `betas`
- `eps`
- `weight_decay`

对等于默认值的无害布尔字段可以规范化处理，但任何会选择另一条执行路径的显式 modifier 必须 blocker。

不要为了 full BF16 顺便重写全部 optimizer args schema；只做 feature-specific gate，避免扩大本次改动面。

Muon 已有 typed args contract，可以继续复用。

---

# 29. Execution Contract：dtype 与 topology 必须分离

full BF16 不应该污染现有 optimizer topology fingerprint。

当前 topology fingerprint 的语义保持：

> 哪些 parameter 归哪个 optimizer / group / LR。

precision 是另一条 execution identity。

新增一个小型、torch-free contract：

`ParameterPolicyExecutionContract`

至少表达：

- contract version；
- train type；
- active execution feature identity；
- expected trainable parameter dtype name；
- 是否要求 live-root parameter identity audit。

full BF16 的 contract 至少包含：

```text
feature = full_bf16
mixed_precision = bf16
expected_trainable_parameter_dtype = bfloat16
```

## 29.1 不把 dtype 写入 RuntimeParameterSpec / topology fingerprint

不能在 `compile_parameter_policy_runtime_spec()` 时把 parameter dtype 固化进 Runtime Spec。

原因：

- session 通常在 trainer 执行 full-BF16 cast 之前创建；
- Runtime Spec 记录的是 optimizer ownership topology；
- dtype 是 execution state，不是 ownership topology。

否则会把同一份正确 policy 因 cast 时机不同制造成错误 topology drift。

## 29.2 production dtype audit 只检查 optimizer-owned trainable parameters

full BF16 runtime 要求：

```text
Parameter Policy trainable parameters
→ dtype == torch.bfloat16
```

不要要求：

- 所有 frozen parameters 都是 BF16；
- 所有 activations 都是 BF16；
- 所有 optimizer state 都是 BF16。

这些不是一个统一 contract。

尤其 optimizer state dtype 是 optimizer-specific：

- PyTorch AdamW 的 moment state 跟 parameter dtype；
- pytorch-optimizer 3.10.0 Muon 的 momentum buffer 使用 `zeros_like(parameter)`；
- bitsandbytes / ScheduleFree 可能采用完全不同 representation。

因此 optimizer state dtype 只进入 GPU qualification evidence，不进入生产 runtime 的统一 assert。

## 29.3 production audit 不做 elementwise finite scan

不要在 2.9B 模型上每个 step / epoch 做：

```python
torch.isfinite(parameter).all()
```

这会制造巨大 GPU bandwidth 开销。

生产 runtime 只做：

- tensor identity；
- device；
- requires-grad；
- dtype。

这些都是 O(parameter tensor count)，不扫描全部 parameter elements。

finite / NaN state 检查只在小型 synthetic GPU qualification 中执行。

---

# 30. 防止 optimizer stale reference：增加一次 live-root identity audit

当前大多数 trainer 的顺序是：

```text
create Parameter Policy session
→ build CompositeOptimizer
→ full-BF16 module.to(dtype)
→ Accelerator.prepare
```

对目前允许的 AdamW / Muon：

- optimizer state 都是 lazy initialization；
- `module.to(dtype)` 正常情况下保持 Parameter object identity。

因此不应为了 full BF16 大幅重排 trainer lifecycle。

重排 session / optimizer 构造时机会影响：

- routing；
- scheduler；
- save hooks；
- 10 个 backend 的既有 integration；

回归风险反而更大。

## 30.1 最小防御方案

`ParameterPolicyTrainerSession` 保存创建 session 时传入的 parameter roots 的**浅引用**，不复制模型/tensor。

新增：

```text
assert_live_root_identity_contract()
```

在 full-BF16 active 时，于 `finalize_after_prepare()` 中执行一次：

```text
re-scan live roots
→ canonical parameter name -> physical id
→ compare with session initial scan
```

如果 dtype conversion / wrapper 真的替换了 Parameter object：

```text
fail closed
```

不要尝试偷偷 rebind optimizer。

这样可以封死：

```text
model forward 使用新 Parameter
optimizer 仍持有旧 Parameter
```

这种最危险的 silent training bug。

## 30.2 为什么只对 full BF16 增加该 audit

baseline Component 已经完成 release qualification，不应在同一个 feature PR 中扩大所有 baseline runtime 行为。

先把 root-identity audit 作为 full-BF16 execution contract 的要求。

未来如果长期 evidence 证明该检查对所有 baseline backend 都稳定，再单独考虑把它提升为公共 invariant。

这避免一次 feature 开发顺便改变全部已发布路径。

---

# 31. Checkpoint identity：保持 manifest v2 baseline，不做不必要 schema 大迁移

旧方案考虑直接把 checkpoint manifest 从 v2 升到 v3。

重新审计后，不建议为了 full BF16 立即做全局 manifest migration。

原因：

- 当前 baseline Component checkpoint v2 已经存在；
- full BF16 Component 之前一直被 blocker 阻止，因此不存在合法的旧版 full-BF16 Component checkpoint；
- 为一个尚未开放的 feature 强制迁移全部 baseline resume，会制造不必要兼容风险。

## 31.1 使用 additive nested execution identity

继续保留：

`PARAMETER_POLICY_CHECKPOINT_MANIFEST_VERSION = 2`

baseline checkpoint manifest 内容保持原样。

仅当 execution contract 非 baseline 时，追加：

```json
{
  "execution_identity": {
    "version": 1,
    "train_type": "anima-finetune",
    "features": {
      "full_bf16": {
        "mixed_precision": "bf16",
        "trainable_parameter_dtype": "bfloat16"
      }
    }
  },
  "execution_signature": "..."
}
```

baseline 没有这两个字段。

因此：

- 旧 v2 baseline checkpoint 继续与新 baseline expected manifest 完全相同；
- mixed-BF16 checkpoint 没有 full-BF16 execution identity；
- full-BF16 resume 要求 execution identity/signature 完全一致；
- mixed ↔ full 不可能被当作同一 resume identity。

nested identity 自己有独立 version。

未来 compile / FP8 等 feature 可以扩展 nested execution identity，而不必每增加一个 execution feature 就升级整个 Parameter Policy manifest。

## 31.2 pre-load fail-closed 顺序必须保持

当前 Accelerator load pre-hook 顺序已经正确：

```text
validate_checkpoint_manifest()
→ actual optimizer/model state load
```

full BF16 必须继续利用这个顺序。

precision / execution mismatch 必须在 optimizer state 被 mutation 之前失败。

不要把 execution identity 校验移动到 post-resume。

## 31.3 不把 GPU / package version 写进 resume identity

GPU 型号、CUDA、torch、dependency version 应进入 qualification artifact / startup diagnostics，而不是 checkpoint execution signature。

否则合法地把 checkpoint 移到另一张兼容 GPU 会被无意义地禁止。

依赖版本支持范围由 DTS package pins / release qualification 管理，不由 checkpoint identity 管理。

---

# 32. Host gate 的具体代码落点

## 32.1 `parameter_policy_compat.py`

保留：

- schema 无法表达的 semantic blockers；
- 尚未迁移到 execution layer 的旧 qualification blockers。

把 **full_bf16 的无条件 blocker** 从这里迁出。

`full_fp16`、FP8、compile、DeepSpeed、swap/offload 等本次不顺便迁移，避免 big-bang refactor。

需要更新函数注释，明确：

> `parameter_policy_v1_semantic_blockers()` 不再代表所有 execution-feature qualification；production caller 必须同时运行 execution blockers。

## 32.2 `parameter_policy.py`

`parameter_policy_runtime_blockers()` 继续作为 request-level canonical gate：

```text
validate canonical policy
→ backend baseline gate
→ semantic blockers
→ execution-feature blockers
→ target/cache blockers
→ optimizer baseline support blockers
```

最后统一 deterministic de-duplication。

Preview / Export / Start 因此继续看到完全相同的 blocker set。

## 32.3 `parameter_policy_trainer.py`

trainer 不能只相信 host Preview。

`create_parameter_policy_session()` 在 model load 后再次执行：

- semantic blockers；
- execution-feature blockers；

然后才 scan/routing/build optimizer。

这是 defense-in-depth。

不要让 trainer 直接调用 HTTP/request-layer API。

共享的是纯 host execution helper，不是 training_request。

## 32.4 Anima Standard → Component bootstrap

full BF16 一旦从 unconditional semantic blocker 迁出，Anima bootstrap 可以先生成 canonical Component policy，再由 Preview execution gate 根据：

- backend；
- actual optimizer profiles；
- execution feature；

给出准确 blocker。

这会把原来“切换 Component 就直接拒绝”的行为变成：

```text
可以进入 Component editor
但 runtime_ready=false
```

在所有 backend 仍 pending 时不会开放训练，因此是安全变化。

其余 Anima bootstrap blockers 本次不改。

---

# 33. GPU qualification infrastructure：新增 feature matrix，不破坏 baseline matrix

现有：

- `tools/run_parameter_policy_gpu_matrix.py`
- `tools/run_parameter_policy_backend_gpu_matrix.py`

职责清楚，不要为了 full BF16 改成一个巨大万能 runner。

新增两个明确工具。

## 33.1 Shared execution-feature CUDA matrix

新增：

`tools/run_parameter_policy_execution_gpu_matrix.py`

第一版只支持：

`full_bf16`

单独进程运行，避免 Accelerate global state 在同一 Python process 中先以 `mixed_precision=no` 初始化、之后又切 `bf16` 造成假失败或状态污染。

至少包含：

### Case 1 — AdamW true BF16

- BF16 Parameter；
- backward；
- gradient dtype evidence；
- gradient clipping；
- step；
- zero_grad；
- state dtype summary；
- state_dict；
- load_state_dict；
- resumed second step；
- finite tiny-tensor checks。

### Case 2 — Muon true BF16

同上。

并验证 state 在 first step 前没有被错误地以 pre-cast dtype eager 初始化。

### Case 3 — Muon + AdamW CompositeOptimizer

使用真实 explicit fallback topology，验证：

- child ownership disjoint；
- 两边都 step；
- state reload；
- resumed second step。

### Case 4 — Accelerator BF16 prepare

创建真正：

```text
BF16 model parameters
+
CompositeOptimizer
+
Accelerator(mixed_precision="bf16")
```

验证：

- Accelerator scaler contract；
- prepare 后 device；
- ownership；
- dtype；
- production execution audit。

### Case 5 — gradient accumulation seam

至少验证：

`gradient_accumulation_steps > 1`

不会导致 CompositeOptimizer 被错误 step，避免真实大 batch workload 只在 accumulation=1 时通过。

## 33.2 Real backend feature matrix

新增：

`tools/run_parameter_policy_backend_feature_gpu_matrix.py`

它不能沿用 baseline runner 的：

> 一个 train_type 只能出现一次，且必须正好 10 个。

feature manifest 使用唯一 `case_id`：

```text
case_id
train_type
feature
fresh_command
resume_command
checkpoint_dir
environment
```

同一个 backend 可以有多个 feature case。

runner 至少验证：

- fresh exit = 0；
- manifest 存在；
- execution identity 包含请求的 feature；
- execution signature 存在；
- resume exit = 0；
- resume 后 manifest identity 不漂移；
- artifact 写入 git SHA / package / CUDA / GPU metadata。

## 33.3 不增加 production qualification bypass

禁止：

- `DTS_ALLOW_UNQUALIFIED_FULL_BF16=1`
- hidden CLI；
- test-only runtime flag；
- environment bypass。

qualification PR 的正确流程是：

```text
branch 上把目标 backend pending → qualified
→ exact-head GPU feature matrix
→ PASS 才 merge
```

PR branch 不是 release，不需要在生产代码里留下绕过 blocker 的后门。

---

# 34. Backend qualification 不能只跑最简单路径

full BF16 的 backend evidence 不要求所有配置笛卡尔积，但必须覆盖会改变实际 trainer path 的关键 seam。

## 34.1 LoRA backends

每个准备标成 `qualified` 的 LoRA backend 都需要独立 case。

不能因为共用 NetworkTrainer 就自动继承。

重点检查：

- adapter trainable parameter dtype；
- frozen base model 与 adapter 的 dtype interaction；
- Text Encoder adapter 路径；
- 已存在的 cache contract；
- save/resume。

SDXL / SD3 / Flux 等已有 model-specific conditioning/cache seam，至少要保留一个会触达该 seam 的 case。

## 34.2 Full backends

`sdxl-finetune`、`flux-finetune`、`anima-finetune` 必须分别跑。

重点检查：

- base-model trainable parameters 真正 BF16；
- trainable Text Encoder（backend 支持时）；
- gradient checkpointing；
- ordinary cache path；
- save/resume。

## 34.3 Anima reference workload

Anima 2.9B Stage 1 是 cross-axis release reference：

```text
Self/Cross/MLP
→ Muon

Muon-ineligible params
→ AdamW fallback

Mod/Base/LLM Adapter
→ AdamW

Qwen3
→ frozen
```

full BF16 reference 至少完成：

```text
preset load
→ canonical policy
→ Preview runtime_ready
→ true BF16 cast
→ live-root identity audit
→ Accelerator.prepare
→ ownership/device/requires-grad/dtype audit
→ forward/backward
→ gradient clip
→ Muon + AdamW step
→ save
→ manifest v2 + execution identity
→ resume
→ execution/topology/scheduler identity validation
→ second step
```

建议同时使用当前真实常用：

- gradient checkpointing；
- latent cache；
- text encoder output cache（Qwen3 frozen 时）。

这样 Anima case 不只是一个“能跑起来”的最小 toy configuration。

## 34.4 SD DreamBooth 单独处理

当前 `scripts/stable/train_db.py` 没有 full-BF16 training-model cast。

不要只加一行：

```python
elif args.full_bf16:
    ...
```

然后就标 qualified。

如果产品上确实需要 SD DreamBooth full BF16，应单开实现 PR：

1. 明确 U-Net / Text Encoder 哪些对象必须 BF16；
2. 检查 SD1/2 上游原本“不支持 full_bf16”的历史原因；
3. 加 Standard-mode regression，防止 Component patch 改坏 legacy；
4. Component dtype/ownership qualification；
5. fresh/save/resume；
6. 成功后再从 `unsupported` 改为 `qualified`。

如果没有足够证据，保持 unsupported 是正确结果。

---

# 35. 防止性能问题与测试矩阵爆炸

## 35.1 production hot path 不增加重操作

允许增加：

- parameter tensor count 级别的 dtype audit；
- 一次 live-root name/id re-scan；
- compact execution signature。

禁止增加：

- 每 step 全模型 parameter finite scan；
- 每 step optimizer state 遍历；
- full-model state dtype dump；
- 为 qualification 保留 debug tensor copy。

## 35.2 GPU evidence 不保存巨大 state dump

artifact 只记录 summary，例如：

```text
parameter_dtypes:
  bfloat16: N

gradient_dtypes:
  bfloat16: N

optimizer_state_tensor_dtypes:
  bfloat16: N
  float32: M
```

不要把真实 optimizer state tensor 写进 JSON。

## 35.3 不跑 optimizer × backend 全组合

release coverage 使用：

```text
每个 backend 的 AdamW reference feature case
+
shared AdamW/Muon feature matrix
+
Anima Muon+AdamW cross-axis reference
```

而不是给所有 backend 都跑所有 optimizer。

前提是：

- 해당 backend baseline Component path 不按 optimizer type 分支；
- 会改变 optimizer lifecycle 的高级 feature 仍被 blocker 阻止。

一旦该前提变化，就为那个新 feature 增加 coupled qualification，而不是无限扩大全局 matrix。

## 35.4 full BF16 runtime qualification 不等于训练质量认证

通过本项目只证明：

- execution contract 正确；
- optimizer/runtime 不崩；
- dtype/ownership/save/resume 正确。

它不证明：

- BF16 optimizer state 与 FP32 state 有同等收敛质量；
- 某个 LR 对某个模型最优；
- 长期训练不会出现数值质量差异。

训练质量属于 recipe / experiment validation，不应混入基础 runtime release gate。

---

# 36. 文件级实施清单

## Phase A — execution feature host infrastructure

新增：

- `mikazuki/parameter_policy_execution.py`
- `tests/test_parameter_policy_execution.py`

生产代码仅修改：

- `mikazuki/parameter_policy_compat.py`
- `mikazuki/parameter_policy.py`
- `mikazuki/parameter_policy_trainer.py`（只增加 host-side preflight，不改变 torch/runtime lifecycle）

测试修改：

- `tests/test_parameter_policy_compat.py`
- `tests/test_parameter_policy_request_contract.py`
- `tests/test_parameter_policy_trainer_runtime_smoke.py`

文档/注释 closeout：

- `mikazuki/parameter_policy_matrix.py`
- `docs/parameter-policy-step6f-plan.md`
- 本文档

Phase A 的唯一行为迁移：

```text
full_bf16 unconditional compatibility blocker
→ canonical-policy-aware execution qualification blocker
```

不得顺便迁移 `full_fp16`、FP8、compile、DeepSpeed、fused、swap/offload 或 multi-GPU。

硬性 invariant：

- Standard 没有 `parameter_policy_config` 时不进入 execution subsystem；
- Component `full_bf16` 未启用时，execution helper 必须在访问 backend/optimizer qualification metadata 之前立即返回空 blocker；
- inactive `full_bf16` 不改变现有 blocker 内容和顺序；
- inactive `full_bf16` 不改变 canonical policy、routing、optimizer topology、scheduler、checkpoint 或 trainer lifecycle；
- direct trainer 与 Preview/Start 使用同一个 execution gate，且 active-but-unqualified full BF16 必须在 parameter scan / optimizer construction 前 fail closed；
- 10 backend 初始全部为 `pending` / `unsupported`，没有任何 `qualified`；
- 所有 baseline-supported optimizer 初始均为 `pending`，没有任何 `qualified`；
- `sd-dreambooth` 明确为 `unsupported`，直到单独实现并验证真正 full-BF16 trainer contract；
- missing qualification metadata 与 `qualified` 但缺 evidence id 都必须 fail closed；
- 不新增 hidden CLI / environment bypass；
- 不修改 frontend、checkpoint schema、具体 model trainer BF16 cast、`parameter_policy_torch.py` 或训练 hot path。

Phase A 按三个 closeout 单元执行：

1. **A1 dormant infrastructure**：先加入静态 qualification metadata / pure host helpers / tests，没有 caller，零产品行为变化；
2. **A2 ownership transfer**：只迁 `full_bf16` blocker，并把 request gate 与 direct trainer preflight 接到同一 helper；
3. **A3 closeout**：只更新矩阵说明与开发文档，不再改变生产逻辑。

此阶段**不开放任何 Component full BF16 backend 或 optimizer**。
## Phase B — execution runtime contract

Phase B 分为四个独立 closeout 单元，避免把 identity、physical audit 与 checkpoint migration 混成一次大改：

### B1 — deterministic execution contract / identity ABI

只修改：

- `mikazuki/parameter_policy_execution.py`；
- `tests/test_parameter_policy_execution.py`；
- 本开发文档。

职责：

- 定义 immutable `ParameterPolicyExecutionContract`；
- full BF16 contract 只表达 `train_type`、active feature、`mixed_precision=bf16`、expected trainable dtype 与 runtime-audit requirement；
- serialized execution identity 只包含 checkpoint/resume-relevant execution semantics；
- `require_live_root_identity` 等 enforcement implementation detail 明确不进入 identity/signature；
- identity 使用固定 schema/version、canonical JSON 与 SHA-256；
- `build_parameter_policy_execution_contract()` 只从 effective config 编译 execution semantics，不读取 backend/optimizer qualification metadata；
- inactive execution feature 返回 `None`，且发生在 train_type / mixed_precision / qualification metadata 访问之前；
- future unknown feature combination fail closed；
- 不接 trainer、checkpoint、model metadata 或 runtime audit caller。

B1 merge 后仍不开放任何 full-BF16 backend/optimizer；其作用是冻结后续 B2/B3 共用的 execution identity ABI。

### B2 — physical runtime contract

实现范围：

- `mikazuki/parameter_policy_trainer.py`；
- `tests/test_parameter_policy_trainer_runtime_smoke.py`；
- 本开发文档。

已实现：

- qualification 通过后才构建 B1 `ParameterPolicyExecutionContract`；
- inactive execution feature 保持 `execution_contract=None`、`execution_root_refs=()`，不增加 root snapshot / weakref / live scan；
- active execution feature 对 semantic roots 做一次 shallow snapshot，并仅保存 `weakref`，避免 session 额外强引用完整 module tree；
- `post_prepare` 与 `post_resume` 两个结构变化 seam 执行 live-root rescan；
- live-root audit 比较完整 alias topology、physical Parameter identity、shape 与 numel，但不比较 dtype / requires_grad；
- live-root audit 必须先于 requires-grad、dtype、optimizer ownership/device audit；
- 发现 Parameter replacement / alias topology drift / dead root 时 fail closed，不自动 rebind optimizer；
- `epoch_start`、checkpoint save/load 等普通 runtime assertion 不重复扫描 module tree；
- execution dtype audit 只检查 RuntimeSpec 实际 optimizer-owned trainable Parameters；
- frozen Parameters 可以保持其它 dtype；
- optimizer state、gradients、activations、finite tensor scan 不进入 B2 generic production invariant；
- host contract dtype name 使用显式 `str -> torch.dtype` 映射，不做动态 `getattr(torch, ...)`；
- B2 不修改 checkpoint manifest/model metadata/startup diagnostics，B3 仍负责 execution identity 的持久化。

性能边界：

- Standard 路径不进入 Parameter Policy trainer runtime；
- ordinary Component / `full_bf16 != true` 仅多一次 B1 contract builder 的 O(1) inactive return；
- active full-BF16 仅在 startup/post-resume 做 metadata-only root scan，不按 tensor element 扫描，也不在每 epoch 扫描；
- dtype audit 为 O(trainable tensor count) metadata read，不触发 tensor copy / finite scan / CUDA elementwise work。

B2 完成后仍不把任何 backend/optimizer 从 `pending` 改成 `qualified`。
### B3 — checkpoint / metadata execution identity

实现范围：

- `mikazuki/parameter_policy_trainer.py`；
- `tests/test_parameter_policy_trainer_runtime_smoke.py`；
- 本开发文档。

已实现：

- B1 `ParameterPolicyExecutionContract` 是 execution identity/signature 的唯一权威来源；B3 不从 args/effective config 重新推导 execution semantics，也不缓存第二套 execution state；
- manifest version 保持 v2；active execution contract 时 additive 写入 `execution_identity` / `execution_signature`，inactive/baseline 时两个 key 完全不存在；
- checkpoint validation 保持既有 policy → topology → scheduler 错误优先级，在 generic exact-manifest equality 前增加 execution-specific field-pair comparison；
- execution comparison按 key presence + value整体比较，可捕获 mixed↔full 双向 mismatch、missing/partial pair、identity/signature drift；
- mismatch 发生在 Accelerator load pre-hook，先于 model / optimizer / scheduler state mutation；
- model metadata仅 active execution时追加 canonical JSON execution identity与 signature；
- startup diagnostics仅 active execution时追加 execution identity/signature；human log只追加 feature、expected trainable dtype、signature 三行摘要；
- checkpoint内容不会反向修改当前 execution contract，也不会自动开启 `full_bf16`；
- B3 不检查 optimizer state/gradient/activation dtype，不引入 GPU qualification，也不改变任何 backend/optimizer qualification状态。

Baseline compatibility hardening：

- baseline v2 manifest完整 dict shape被冻结；
- baseline manifest serialized JSON text被冻结；
- baseline model metadata key set被冻结；
- baseline startup diagnostics key set被冻结；
- baseline startup log完整文本被冻结；
- active same-identity manifest可正向验证；
- mixed-BF16 checkpoint → full-BF16 run 与 full-BF16 checkpoint → mixed-BF16 run 都在 state mutation前 fail closed；
- malformed execution identity/signature partial pair得到 execution-specific error。

性能边界：

- B3仅处理小型 execution identity payload与 SHA-256 fingerprint，复杂度与模型 parameter数量无关；
- 不增加 root scan、tensor-content scan、GPU copy或 hot-path elementwise工作；
- 不为减少 hash开销缓存第二份 identity/signature，优先避免 state drift。

B3 完成后仍不把任何 backend/optimizer 从 `pending` 改成 `qualified`；真实 CUDA full-BF16 evidence 仍由 Phase C/D 负责。
### B4 — Phase B closeout

Phase B closeout 只做组合审计、精确回归与文档收口，不开放任何新的 execution capability。

组合审计结论：

- request-level Preview/Start runtime blocker、direct trainer execution gate、B1 contract builder、B2 physical runtime audit、B3 checkpoint persistence 按单向 owner 链路组合，没有第二套 execution semantics；
- `full_bf16` inactive 仍在 qualification metadata / contract / physical audit 之前 no-op；ordinary Component 不持有 execution contract/root refs，也不增加 live-root scan；
- B1 execution identity ABI/version/signature reference 保持冻结；B2/B3 只消费该 contract，不从 args/checkpoint 重新推导；
- B2 `post_prepare` / `post_resume` physical identity audit、trainable dtype audit、optimizer ownership/device audit 的顺序保持 fail-closed；
- B3 manifest version 仍为 v2；baseline manifest/metadata/diagnostics/log exact regression 保持不变；active execution identity/signature 只做 additive persistence；
- mixed-BF16 ↔ full-BF16 checkpoint mismatch 使用真实 production lifecycle (`prepare -> finalize_after_prepare -> load_state`) 在 model/optimizer/scheduler state mutation 前 fail closed；
- external scheduler 与 optimizer-managed scheduler 均覆盖 active execution persistence；
- B3 / PR #50 merge 后 main tree 已验证与其 exact-head PR tree 无源码差异。

Phase B closeout qualification state：

- backend：`sd-dreambooth=unsupported`；其余 release-matrix backend 全部 `pending`；
- optimizer：当前 baseline `component_support="supported"` 的所有 optimizer 全部 `pending`；
- `qualified` 记录数保持 0；Phase B 没有使用测试 mock 以外的 capability promotion。

性能与兼容性边界：

- Standard 路径不进入 Parameter Policy trainer runtime；
- ordinary Component 仅承担 inactive execution feature detection/contract no-op，新增成本为 O(1) host metadata；
- active full-BF16 只在 `post_prepare` / `post_resume` 执行 metadata-only live-root scan，并进行 O(trainable tensor count) dtype metadata audit；不做 tensor-content scan；
- B3 persistence/hash 与模型大小无关；
- 不支持/未验证的 DeepSpeed、compile、FP8、full FP16、offload/swap、fused optimizer 等语义仍由已有 compatibility blockers fail closed。

Phase C 入口条件：

1. Phase B exact-head host/runtime CI 全绿；
2. qualification table 仍保持上述 pending/unsupported 状态；
3. B1 execution identity ABI 不再在 Phase C 内修改；
4. B2/B3 只作为 evidence target 使用，Phase C 不通过绕过/放宽这些 runtime/checkpoint invariant 来换取测试通过；
5. 只有 shared true-BF16 CUDA matrix 在 exact head 上通过的 optimizer variant，才允许后续单独 promotion；
6. backend qualification 仍必须等 Phase D backend-specific fresh forward/backward/step/save/resume/second-step evidence。

因此 Phase B 的最终产物是：**execution semantics 已可定义、审计、持久化和恢复验证，但 production capability 仍保持关闭。**
## Phase C — shared true-BF16 CUDA qualification

### C0 — reachable execution state

Phase C 首先冻结 shared full-BF16 qualification 的运行环境边界，不把单卡 CUDA evidence 外推到未经验证的 runtime modifier。

已实现：

- `parameter_policy_execution_environment_blockers()` 保持 torch-free / accelerate-free，只消费已经建立的 execution contract 与规范化 `num_processes` / `distributed_type`；
- active `full_bf16` 当前只允许 `num_processes == 1` 且 `distributed_type == NO`；multi-process/DDP/FSDP 等 distributed runtime 在独立 exact-head CUDA qualification 前 fail closed；
- ordinary Component (`execution_contract=None`) 不受该限制，现有 multi-process Component 路径不被 C0 改写；
- environment gate 在既有 trainable dtype audit 后、optimizer ownership/device audit 前执行，不重排 B2 live-root/requires-grad/dtype 语义；
- compile / torch_compile / FP8 / full FP16 / fused optimizer / DeepSpeed / offload / block swap 等已有 runtime blocker 继续保持 fail closed；
- `gradient_accumulation_steps > 1` 不在 C0 被禁止，留给 Phase C shared CUDA evidence。

### C1 — exact-head execution CUDA evidence harness

新增：

- `tools/run_parameter_policy_execution_gpu_matrix.py`；
- host-side harness contract test；
- GitHub host CI 对 runner 的 path/py_compile 与 source contract 检查。

C1 runner contract：

- parent coordinator 与 worker subprocess 都要求 `--expected-commit`，并在 import torch / Accelerate / DTS runtime 之前验证实际 `git rev-parse HEAD` 完全一致；
- qualification workspace 必须完全 clean，`git status --porcelain=v1 --untracked-files=all` 不能有任何输出；tracked/staged/untracked 污染都会 fail closed；evidence JSON 因此必须写到 repository 外；
- CUDA 不可用、无 visible GPU、或 `torch.cuda.is_bf16_supported()` 不成立时直接 FAIL，不允许 CPU fallback/skip；
- 每个 case 使用 fresh `sys.executable` subprocess，`shell=False`，为 C2/C3 的 train/save 与 fresh resume phase 保留同一 protocol；
- temporary qualification lease 只存在于 worker 进程内：`pending` 可临时 leased，正式 `qualified + evidence_case_id` 直接使用，malformed qualified / unsupported / invalid status 全部 fail closed；退出时必须恢复原 qualification row；该状态机有 CPU 行为测试覆盖正常退出与异常退出恢复；
- 不增加 production bypass 环境变量，不修改 production qualification table。

C1 infrastructure cases：

1. `infra:cuda-bf16-capability:v1`：记录真实 CUDA/BF16 hardware/runtime/package evidence，包括 GPU、compute capability、VRAM、CUDA runtime、NVIDIA driver 与 pinned package versions；
2. `infra:full-bf16-session:v1`：使用 tiny Flux-shaped scaffold 走 `create session -> model.to(bfloat16) -> Accelerator(mixed_precision='bf16') -> prepare -> finalize_after_prepare -> B2 audit -> B3 execution manifest`。

两个 case 都明确：

- `scope = infrastructure`；
- `promotion_eligible = false`；
- backend scaffold 不具备 backend qualification 资格；
- AdamW scaffold 不具备 optimizer qualification 资格；
- C1 不执行 optimizer step，不产生 C2 optimizer evidence。

CUDA evidence execution model：

- GitHub Actions 只负责 host/source/CPU/runtime contract CI；真实 CUDA qualification 不依赖 GitHub self-hosted workflow；
- 现有 `.github/workflows/parameter-policy-gpu-matrix.yml` 保持原样，不调用新的 execution qualification runner，也不作为 C1/C2 的 authoritative evidence gate；
- 在已知可用的 CUDA 环境（本地 GPU 或云 GPU）直接 checkout PR exact SHA，并运行 standalone runner；
- 旧 `tools/run_parameter_policy_gpu_matrix.py` 保持历史宽松调用兼容，但 Phase C qualification 必须显式传 `--expected-commit` 进入 strict mode；strict mode 同样在 import torch 前验证 exact HEAD + 完整 clean workspace；
- 新 execution runner 强制 `--expected-commit`，两个 standalone runner 的 authoritative qualification evidence 因此使用同等级 provenance，不依赖 CI provider 注入的环境变量；
- standalone evidence JSON 由运行者保留并提交 review；GitHub workflow 是否能成功调度不影响 qualification 结论。

推荐 exact-head CUDA evidence 命令：

```text
python tools/run_parameter_policy_gpu_matrix.py \
  --expected-commit <PR_EXACT_SHA> \
  --output ../parameter-policy-gpu-matrix.json

python tools/run_parameter_policy_execution_gpu_matrix.py \
  --expected-commit <PR_EXACT_SHA> \
  --output ../parameter-policy-execution-gpu-matrix.json
```

C2 source-development 入口条件：

1. GitHub host/source/CPU/runtime CI exact-head 全绿；
2. C0/C1 focused source review 无 P1/P2 blocker；
3. backend/optimizer production qualification 状态仍保持 B4 closeout 的 pending/unsupported 状态；
4. B1 identity、B2 physical audit、B3 checkpoint schema 不为后续实现放宽；
5. C2 只能在现有 evidence schema / subprocess protocol 上新增 AdamW train/save/fresh-resume/second-step cases，不重写 harness architecture；
6. C2/C3/Phase D 可以继续完成源码、host contract、CPU/runtime smoke 与 UI/Preview/Start wiring；source-development 不要求此时 clone CUDA 环境或生成真实 GPU evidence。

因此，C1 standalone CUDA runner 是后续 final qualification 的 evidence infrastructure，不是继续开发 C2/C3/Phase D 源码的前置硬门槛。这样可以避免在完整 UI positive path 尚未形成时重复 clone / 配置 GPU 环境、反复生成很快会因源码继续变化而失效的中间态 evidence。

Final qualification / promotion gate：

1. 完整 UI positive path 已具备，能够从 UI 配置 full-BF16 / Component，经过 Preview 与 Start 进入真实 trainer lifecycle；
2. 必须选择最终 candidate exact head，在已知可用 CUDA 环境 checkout 该 SHA；
3. 旧 shared GPU matrix 以 `--expected-commit` strict mode PASS；
4. C1 execution infrastructure matrix PASS；
5. C2/C3/C4 对应的 AdamW、Muon、Muon + AdamW fallback 真实 step / accumulation / save / fresh-resume / second-step evidence PASS；
6. Phase D 对目标 backend 的真实 forward/backward/step/save/resume evidence PASS；
7. 完整 UI -> Preview -> Start -> train -> checkpoint -> resume -> second-step positive path 在相同 qualification candidate 上 PASS；
8. 两类 standalone evidence JSON 均写在 repository 外，并完成 commit/provenance/runtime/case review；
9. 在上述 evidence 完整之前，backend/optimizer production qualification 继续保持 pending/unsupported，不允许仅凭源码完成或 CPU CI 将 capability 切为 qualified。

为了测试最终 UI positive path，允许后续设计一个显式、测试专用的 qualification session/lease，但必须保持以下边界：只用于 qualification 环境、进程内临时生效、退出自动恢复、普通 UI 用户不可触发、不能通过环境变量全局绕过、不能修改 production qualification table。现有 C1 temporary qualification lease 只属于 evidence tooling，不能直接当成 production UI bypass。

### C2 — AdamW full-BF16 lifecycle evidence

C2 source-development 目标是把 AdamW 从 C1 的 infrastructure scaffold推进到完整 lifecycle evidence，但仍不修改 production qualification table。

C2 将 execution GPU evidence schema 从 v1 升为 v2，因为 `cases[]` 从单个 `phase/status/details` row扩展为 `case -> phases[]`。C1 的两个 infrastructure case仍保持单 `probe` phase，但输出统一采用 v2 phased shape；旧 v1 evidence不应与 C2 v2 evidence混用。

新增稳定 evidence bundle：

- `phase-c:adamw-full-bf16:v1`
- `optimizer:adamw:full-bf16:accum1:v1`
- `optimizer:adamw:full-bf16:accum2:v1`

两个 AdamW case 都固定为：

```text
train_save
→ fresh Python subprocess
→ resume_second_step
```

若 `train_save` 失败，后续 phase 必须记录 `not_run / dependency_failed`，不得继续运行并形成部分假绿。只有 accum1 与 accum2 两个 case 的全部 phase 都 PASS 时，coordinator 才能把 evidence bundle标为 PASS。若只选择其中一个 AdamW case，bundle必须标为 `incomplete` 且 runner 返回非零；partial case仅用于诊断，不能被外层脚本误判为 qualification成功。

C2 lifecycle contract：

1. temporary qualification lease 仅在 worker进程内把 Flux scaffold与 AdamW 暂时视为可执行；backend始终 `backend_qualification_eligible=false`；
2. model trainable parameters在 prepare前后都必须保持真实 `torch.bfloat16`；
3. 使用 production `create_parameter_policy_session()`、`make_legacy_scheduler_factory()`、`Accelerator.prepare()`、`finalize_after_prepare()` 与 checkpoint hooks，不另造训练 runtime；
4. accum1 一个 microstep完成一个 logical step；
5. accum2 必须通过 `accelerator.accumulate(model)` 证明第一 microstep不更新参数、不建立/推进 optimizer state、不推进 scheduler，第二 microstep才真正 step；
6. `train_save` 完成第一个 logical step后用 `accelerator.save_state()` 保存 checkpoint与 handoff evidence；
7. `resume_second_step` 必须在 fresh subprocess中重新创建 model/session/optimizer/scheduler/Accelerator，先证明 optimizer state为空，再 `load_state()`；
8. resume后执行 `post_resume` runtime audit，并验证 model、optimizer、scheduler、execution identity/signature与 handoff一致；
9. resume后再执行第二个 logical step，证明参数、AdamW state和scheduler继续前进；
10. optimizer state tensor dtype不预设必须BF16；C2记录真实 state schema/dtype/fingerprint，并验证 finite与save/resume一致性。

新增 tooling-only runtime evidence helper负责 tensor/model/optimizer/scheduler/gradient fingerprint；production trainer/request不得引用该模块。

C2 host CI同时增加 CPU accumulation smoke，使用真实 `Accelerator(cpu=True, gradient_accumulation_steps=2)` 验证 CompositeOptimizer/CompositeLRScheduler在第一 microstep被 Accelerate正确抑制。CPU smoke只验证 integration semantics，不替代最终 CUDA qualification。

C2 source closeout后仍必须保持：

```text
AdamW full-BF16 production qualification = pending
all backend full-BF16 qualification       = pending/unsupported
```

真实 CUDA execution与完整 UI positive path继续延后到 final qualification / promotion gate。

后续 C3/C4 将分别加入 Muon、Muon + AdamW fallback 的对应 lifecycle evidence；在 final qualification / promotion gate 通过之前，不改变任何 optimizer qualification。

### C3 — Muon full-BF16 lifecycle evidence

C3 在 C2 phased evidence protocol 上新增 pure-Muon lifecycle，不修改 production optimizer/trainer/runtime semantics，也不提前实现 Muon + AdamW fallback。稳定 evidence IDs：

- `phase-c:muon-full-bf16:v1`
- `optimizer:muon:full-bf16:accum1:v1`
- `optimizer:muon:full-bf16:accum2:v1`

两个 Muon case 与 C2 一样固定为：

```text
train_save
→ fresh Python subprocess
→ resume_second_step
```

C3 使用独立 Flux-shaped scaffold：`Linear(8, 12, bias=False) -> Linear(12, 8, bias=False)`。它只产生 `transformer.double_stream` 下两个 approved hidden 2D matrix weights，shape 分别为 `[12, 8]` 与 `[8, 12]`。这样同一 pure-Muon lifecycle 同时覆盖 pinned Muon Newton-Schulz 的 `rows > cols -> transpose=True` 与 `rows < cols -> transpose=False` 两条矩形执行路径，同时仍不引入 bias/norm 等 Muon-ineligible 参数；显式 AdamW fallback 保留给 C4。

C3 在通用 C2 lifecycle assertions 之外增加 Muon-specific evidence：

1. evidence worker要求 `pytorch-optimizer==3.10.0`，并记录真实 provider class；
2. production routing 必须恰好产生两个 Muon profile 的 `primary` matrix assignments，shape 为 `[12, 8]` 与 `[8, 12]`，`fallback_count == 0`，并显式记录 transpose true/false coverage；
3. CompositeOptimizer 必须只有一个 `Muon` child，所有 provider param group 都必须 `use_muon=True`；
4. fresh optimizer state 必须为空，Muon group step从 0 开始；
5. accumulation>1 的非同步 microstep不得创建 momentum state、推进 group step、修改参数或推进 scheduler；
6. synchronized Muon step后两个 routed parameters 都必须建立 `momentum_buffer`，state parameter count 必须等于 routed parameter count，不得出现 provider内部 AdamW path的 `exp_avg` / `exp_avg_sq`；
7. Muon step counter读取 provider真实的 `param_groups[*]["step"]`，而不是沿用 AdamW per-parameter step semantics；
8. optimizer state仍记录真实 tensor dtype/schema/fingerprint，不把 momentum state dtype写成 production ABI；
9. train/save handoff记录 provider、routing、Muon qualification-family contract、Muon path、execution identity/signature与 model/optimizer/scheduler evidence；
10. fresh resume必须先证明新 Muon state为空，再 load checkpoint；load后 provider/routing/qualification-family/path/state fingerprint必须与 handoff一致，group step恢复到 1，随后第二 logical step推进到 2。

C3 v1 将当前 DTS production `MUON_ARGUMENTS` 视为同一个 pure-Muon execution/state family，而不是把 reference run 的一组具体数值误解为唯一允许配置。该 family 当前包括 `momentum`、`weight_decay`、`weight_decouple`、`nesterov`、`ns_steps`、`ns_coeffs` 与 `use_adjusted_lr`：这些参数可以改变算法数值行为，但当前不会改变 provider class、`use_muon=True` ownership、`momentum_buffer` state family、external scheduler ownership、checkpoint topology model或 distributed/fallback execution family。

tooling 中的 `MUON_FULL_BF16_ARGUMENT_FAMILY` 必须通过 host test 与 production `MUON_ARGUMENTS` exact equality。C3 reference policy仍显式提供当前 family 的全部参数，并从真实 `session.runtime_spec.optimizers[0].optimizer_arguments` 生成 `dts.parameter-policy.muon-full-bf16-qualification-family` evidence，记录 provider、argument family 与 canonical reference arguments。任何未来 production Muon argument surface 变化都必须先使 host CI fail closed，再显式判断该参数是否仍属于已有 C3 execution/state family；如果新参数会改变 provider、state topology、fallback、distributed/fused/foreach 等 execution path，则必须增加新的 blocker/evidence，不能静默继承旧 qualification。

C3 host/runtime CI同时增加：

- real pinned Muon + CompositeOptimizer/CompositeLRScheduler + Accelerate accumulation=2 CPU smoke，并使用 `[12, 8]` + `[8, 12]` 两个矩形参数覆盖两种 Newton-Schulz orientation；
- fresh two-parameter Muon optimizer state round-trip，验证 group step、两个 `momentum_buffer`、normalized `ns_coeffs` 与 parameter-state mapping；
- tooling two-parameter state fingerprint round-trip，防止 provider state中出现跨进程不稳定表示或 multi-parameter load ordering漂移；
- C3 Muon argument-family 与 production `MUON_ARGUMENTS` exact-parity host contract；
- Muon bundle的 absent/incomplete/fail/pass contract；
- AdamW 与 Muon evidence bundles additive coexistence source contract。

C3 source closeout仍保持：

```text
Muon full-BF16 production qualification  = pending
AdamW full-BF16 production qualification = pending
all backend full-BF16 qualification       = pending/unsupported
```

真实 BF16/CUDA Muon evidence仍与 C1/C2 一样留到 final qualification exact-head candidate执行；CPU smoke只证明 integration/state semantics。C4 再单独验证 Muon eligible parameters与 Muon-ineligible parameters同时存在时，显式 Muon + AdamW fallback 的 multi-child ownership、scheduler与checkpoint lifecycle。

### C4 — Muon + AdamW explicit fallback lifecycle evidence

C4 在 C2 AdamW 与 C3 pure-Muon shared optimizer evidence 之上增加 multi-child composition evidence；不新增 production optimizer-topology qualification table，也不修改 Parameter Policy fallback semantics。稳定 evidence IDs：

- `phase-c:muon-adamw-explicit-fallback-full-bf16:v1`
- `optimizer:muon-adamw-explicit-fallback:full-bf16:accum1:v1`
- `optimizer:muon-adamw-explicit-fallback:full-bf16:accum2:v1`

两个 case继续固定：

```text
train_save
→ fresh Python subprocess
→ resume_second_step
```

C4 使用独立 Flux-shaped scaffold：`Linear(8, 12, bias=True) -> Linear(12, 8, bias=True)`。同一 `transformer.double_stream` component 内：

- `[12, 8]` 与 `[8, 12]` matrix weights 必须通过 production Muon eligibility，并路由为 `primary -> muon`；
- `[12]` 与 `[8]` biases 必须因 Muon eligibility 不成立而路由为 `fallback -> adamw_fallback`；
- primary LR固定为 `2e-2`，explicit fallback LR固定为 `1e-2`，证明 Runtime Spec / child scheduler 保持独立 LR ownership；
- Muon child必须继续全部 `use_muon=True`，禁止使用 provider internal AdamW fallback。

C4 authoritative evidence新增以下 invariants：

1. RoutingPlan必须恰好包含 2 primary matrix weights + 2 fallback biases，且 canonical name/shape/class/profile精确匹配 reference topology；
2. actual Muon child parameter IDs必须与 Muon primary routing IDs完全相等，actual AdamW child IDs必须与 explicit fallback routing IDs完全相等；两 child physical ownership交集为空，并且并集等于全部 trainable parameter IDs；
3. evidence handoff只持久化 canonical ownership names，不持久化跨 fresh Python process不稳定的 `id(parameter)`；
4. route/profile-specific parameter fingerprints分别记录 `muon` 与 `adamw_fallback` 参数，synchronized logical step后四个 trainable parameter都必须真实变化，不能只靠 aggregate model hash判断两个 child都执行；
5. 每个 trainable parameter都必须产生 finite/nonzero gradient；每次 optimizer/scheduler step后还必须递归审计 routed parameter、Composite optimizer state与scheduler state中的所有 tensor evidence仍为 finite，防止 full-BF16 step产生 NaN/Inf却仅凭 hash/counter变化假通过；
6. Muon child actual provider class必须与 pinned `pytorch-optimizer==3.10.0` resolver记录的 provider class完全一致；同步 step后两个 weights均产生 `momentum_buffer`，不得出现 `exp_avg` / `exp_avg_sq`；
7. Muon group step与 explicit AdamW per-parameter step必须严格解析为 non-negative integral counters，拒绝 bool、fractional或负值；AdamW child同步 step后两个 biases均产生 `step/exp_avg/exp_avg_sq`，step counters分别从 1推进到 fresh-resume后的 2；
8. accumulation=2的第一 physical microstep必须同时 suppress两个 child：所有 routed parameters、两个 optimizer states、两个 external child schedulers与 Composite scheduler均不得推进；
9. C4 是第一次 shared full-BF16 evidence同时运行两个 external child schedulers；必须验证每个 scheduler attached到对应 child optimizer、两个 scheduler step/epoch保持一致，并保留 profile LR `muon=2e-2` / `adamw_fallback=1e-2`；
10. train/save handoff记录 routing、canonical ownership、qualification-family、execution identity/signature、scheduler identity/signature、profile parameter evidence、Composite optimizer/scheduler fingerprints、Muon/AdamW child state与per-profile scheduler evidence；
11. fresh resume先证明两个 child state都为空，再通过 production `Accelerator.load_state()` 与 checkpoint hooks恢复；恢复后上述 evidence必须与 handoff精确一致，再完成第二 logical step。

C4 bundle使用 `scope=optimizer_topology`，topology为 `muon_primary_adamw_explicit_fallback`，并记录 C2 AdamW与C3 Muon bundles为 promotion prerequisites；但 C4 本身不会创建 `FULL_BF16_OPTIMIZER_TOPOLOGY_QUALIFICATIONS` 或其它 production组合表。production release gate仍按单 optimizer qualification + final C2/C3/C4 evidence共同审查，避免随着 optimizer组合增加形成 topology qualification笛卡尔积。

C4 host/runtime CI新增：

- C4 bundle absent/incomplete/fail/pass contract；
- Muon + AdamW dual pending temporary lease与 exact restoration；
- real pinned Muon + torch AdamW + CompositeOptimizer + two external schedulers + Accelerate accumulation=2 CPU smoke；
- dual-child optimizer/scheduler save/load round-trip与 second-step continuation；
- source contract保证 C4不修改 production qualification table，也不新增 topology qualification table。

C4 source closeout仍保持：

```text
Muon full-BF16 production qualification  = pending
AdamW full-BF16 production qualification = pending
all backend full-BF16 qualification       = pending/unsupported
```

真实 BF16/CUDA C4 lifecycle evidence继续留到 final qualification exact-head candidate执行。C4只证明 shared explicit-fallback composition；实际 Anima/Flux/SDXL 等 backend中 Mod/Base/Adapter/Qwen3 等复杂 parameter distribution与trainer-specific lifecycle继续属于 Phase D backend qualification。

## Phase D — backend feature qualification

Phase D压缩为三个主要开发阶段：

```text
D0 — qualification infrastructure + shared AdamW/Muon CUDA promotion
D1 — all non-Anima backend qualification
D2 — Anima LoRA/Full + final UI closure
```

### D0 — qualification infrastructure + shared AdamW/Muon promotion

D0 candidate将 `AdamW` 与 `Muon` 的 production full-BF16 qualification row切为 `qualified`，二者共同引用：

- `phase-d0:shared-adamw-muon-full-bf16:v1`

这不是单独复用某个 C2/C3 bundle，而是要求同一 exact head上的完整 C1-C4 matrix全部 PASS：

- `infra:cuda-bf16-capability:v1`
- `infra:full-bf16-session:v1`
- AdamW accum1 / accum2
- Muon accum1 / accum2
- Muon + AdamW explicit fallback accum1 / accum2

`summarize_shared_full_bf16_promotion()` 同时验证：

- 8个 required cases全部 PASS；
- AdamW/Muon source rows均为 `qualified` 且 evidence id完全一致；
- 其它 optimizer没有被 D0意外 promotion；
- 所有 backend仍保持 `pending/unsupported`，D0不开放真实 backend训练能力。

Phase C原 bundle继续保持 `promotion_eligible=false`；只有 D0 shared promotion summary可以得到 `promotion_eligible=true`。

D0 将 strict C1-C4 execution runner正式接入 `.github/workflows/parameter-policy-gpu-matrix.yml`。所有 GPU evidence统一写入 `$RUNNER_TEMP/dts-parameter-policy-evidence`，避免旧 runner输出污染 repo clean-tree contract。运行顺序固定为：

```text
checkout exact head
→ install pinned runtime
→ strict C1-C4 execution qualification
→ legacy synthetic CUDA regression
→ optional legacy all-backend regression
→ optional Phase D backend-feature runner
→ upload external evidence directory
```

D0同时新增：

- `tools/parameter_policy_backend_feature_gpu_support.py`
- `tools/run_parameter_policy_backend_feature_gpu_matrix.py`

backend-feature runner在 torch/runtime import之前必须验证：

- `--expected-commit` 与 `HEAD` 完全相等；
- repo tracked/untracked workspace完全 clean；
- manifest与output均位于 repo之外。

manifest v1允许按 case选择，不再强制一次覆盖全部10个 backend。每个 case至少包含：

```text
case_id
train_type
feature=full_bf16
fresh_command
resume_command
cwd/environment
fresh_checkpoint_dir
resume_checkpoint_dir
```

两个 checkpoint目录必须不同且位于 repo之外。runner要求 fresh命令真正生成 checkpoint-1，resume命令从fresh state继续并生成 checkpoint-2；两个 checkpoint manifest都必须是 production manifest v2，并包含完全匹配的：

- policy hash；
- runtime topology fingerprint；
- trainable/frozen components；
- optimizer topology；
- scheduler identity/signature；
- full-BF16 execution identity/signature。

D0 backend-feature runner只建立 D1 execution infrastructure，所有 case仍输出：

```text
backend_qualification_eligible=false
production_qualification_mutated=false
```

不会临时 lease、修改或绕过 production backend qualification。

D0 merge gate：

```text
host CI PASS
→ strict source review
→ same exact SHA real-CUDA C1-C4 PASS
→ shared_optimizer_promotion.status=pass
→ verification review
→ PASS才merge
```

D0完成后预期 production状态：

```text
AdamW full-BF16 = qualified
Muon  full-BF16 = qualified

all backends = pending / unsupported
```

### D1 — non-Anima backend qualification

D1使用 D0 backend-feature runner一次处理非Anima backend，各 backend保持独立 evidence bundle/status；某一 backend失败时只保持该 row为 pending，不阻塞其它已通过 backend。目标包括：

- `sdxl-finetune`
- `flux-finetune`
- `sdxl-lora`
- `flux-lora`
- `chroma-lora`
- `sd3-lora`
- `sd-lora`

`sd-dreambooth`继续保持 `unsupported`，除非后续单独实现等价 full-BF16 cast contract。

### D2 — Anima + final UI closure

D2处理：

- `anima-lora`
- `anima-finetune`
- Anima Full Stage-1 Muon + AdamW explicit fallback reference topology；
- final GUI → Preview → Start → checkpoint → fresh resume → second step positive path。

Anima Stage-1 reference默认保持 Qwen3 frozen；未单独 qualification的 Qwen3 training variant继续 fail-closed。

## Phase E — optimizer expansion

在 full-BF16 AdamW/Muon 基线稳定后，再逐个开放：

- bitsandbytes；
- Lion；
- SGD；
- ScheduleFree。

每个 optimizer family 必须先通过 shared full-BF16 CUDA evidence。

只有存在 optimizer-specific trainer coupling 时才增加 backend cross case。

不要一次性把所有当前 `component_support="supported"` optimizer 全部标成 full-BF16 supported。

---

# 37. 必须新增的回归与 contract tests

至少包含以下 CPU/source tests。

## 37.1 qualification table

- backend key set == 10-backend release matrix；
- unknown backend blocked；
- pending blocked；
- unsupported blocked + reason；
- qualified 必须有 evidence id；
- 新增 backend 若未加入 feature table，测试失败。

## 37.2 optimizer full-BF16 rules

- AdamW reference args accepted；
- Muon typed args accepted；
- Muon + AdamW referenced profiles accepted；
- Lion / bitsandbytes / ScheduleFree 初始 blocked；
- AdamW explicit `fused/foreach/capturable/differentiable` execution variant blocked；
- full_bf16=false 时不增加这些 blocker。

## 37.3 bootstrap / Preview / Start parity

- Anima full-BF16 可以进入 Component editor；
- pending 时 `runtime_ready=false`；
- Start 同一 blocker；
- backend qualified + optimizer unqualified 仍然 blocked；
- backend qualified + optimizer qualified 才 runtime_ready。

## 37.4 runtime contract

- BF16 trainable params pass；
- 任一 optimizer-owned FP32 parameter fail；
- frozen parameter dtype 不影响该 contract；
- physical parameter replacement 被 live-root identity audit 捕获；
- ordinary dtype cast 不替换 Parameter 时通过；
- baseline session 不执行 full-BF16-specific root audit。

## 37.5 checkpoint identity

- baseline v2 manifest byte/structure semantics保持不变；
- full-BF16 manifest 包含 execution identity/signature；
- mixed-BF16 checkpoint 不能 resume full-BF16；
- full-BF16 checkpoint 不能 resume mixed-BF16；
- topology mismatch 仍按原逻辑失败；
- scheduler mismatch 仍按原逻辑失败；
- execution mismatch 在 optimizer state load 前失败。

## 37.6 Standard regression

必须证明：

- Standard 不 import / construct Parameter Policy execution runtime；
- Standard optimizer selection 不变；
- Standard full_bf16 原有 backend 行为不被 Component gate 改写；
- Component feature metadata 不泄漏进 Standard training config。

---

# 38. 明确不做的方案

为了避免继续堆技术债，本项目明确不采用以下做法。

## 38.1 不删除全局 blocker 后默认全部 backend 支持

错误。

## 38.2 不把 Anima 一次成功训练当作全模型 qualification

错误。

## 38.3 不给 `OptimizerCapability` 不断增加 feature bool

会形成 feature-flag 垃圾场。

## 38.4 不建立全量 backend × optimizer × feature 笛卡尔矩阵

GPU 成本和维护成本不可接受。

## 38.5 不把 dtype 混进 optimizer topology fingerprint

会混淆 ownership topology 与 execution state。

## 38.6 不为 qualification 增加隐藏 bypass

会留下生产后门。

## 38.7 不为了 full BF16 重排全部 trainer lifecycle

现有 optimizer-before-cast 对首批 AdamW/Muon 的 lazy state contract 可验证；大范围重排风险更高。

## 38.8 不在 production hot path 做全模型 finite scan

性能代价过高。

## 38.9 不在本次 full BF16 项目顺便开放 FP8 / compile / DeepSpeed / swap / fused / multi-GPU

每个 feature 都需要独立 ownership contract。

---

# 39. 新的 full BF16 release acceptance target

一个 backend 从 `pending` 变为 `qualified`，至少要求：

```text
canonical policy
→ execution feature gate PASS
→ trainer semantic gate PASS
→ parameter scan/routing
→ CompositeOptimizer
→ trainer full-BF16 cast
→ live-root parameter identity audit
→ Accelerator.prepare
→ ownership/device/requires-grad/dtype audit
→ real forward/backward
→ optimizer step
→ save
→ manifest v2 + execution identity/signature
→ resume pre-hook identity validation
→ post-resume runtime audit
→ second optimizer step
```

同时：

- 其它未 qualified execution feature 仍然 blocker；
- 未 qualified optimizer execution variant 仍然 blocker；
- Standard path 不受影响；
- baseline Component mixed precision 不受影响；
- no hidden fallback；
- no duplicate ownership；
- no orphan trainable parameter；
- no stale optimizer parameter reference。

对于 Anima 2.9B Stage 1，还额外要求：

- Muon 只拥有 eligible hidden 2D weights；
- AdamW fallback 只拥有 Muon-ineligible routed parameters；
- Mod/Base/LLM Adapter 归 AdamW；
- Qwen3 frozen；
- component LR log 与 policy 一致；
- gradient checkpointing/cache reference path 可运行；
- fresh/resume topology、scheduler、execution identity 全部一致。

---

# 40. 后续 feature 的扩展原则

full BF16 不是为了制造一套只服务 BF16 的特殊架构。

本次新增的 execution layer 应满足：

```text
baseline Parameter Policy
    |
    +-- ownership topology
    |
    +-- execution feature qualification
            |
            +-- full_bf16
            +-- future full_fp16
            +-- future fp8
            +-- future compile
            +-- future distributed/offload
```

但未来每新增一个 feature，只在它真正需要时增加：

- feature-specific backend qualification；
- feature-specific optimizer qualification；
- feature-specific execution identity；
- feature-specific runtime audit。

不要提前抽象未出现的共性。

尤其：

> **先保持 API 形状可扩展，再让真实第二个 feature 验证抽象是否成立。**

这是避免为了“未来可扩展”过度工程、最终反而形成屎山的核心原则。

当前最安全的开发起点因此是：

> **Phase A：execution feature host infrastructure，全部 backend 仍 pending/unsupported，不开放任何新训练能力。**

只有 Phase A/B 的 source-level contract 稳定以后，才进入真实 full-BF16 CUDA qualification。
