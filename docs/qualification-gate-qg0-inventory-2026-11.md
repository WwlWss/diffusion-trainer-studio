# QG-0 — Parameter Policy Qualification Gate 全量分类登记（源码级）

**基准**：`main@1dc4f56fb862aade574add48c4f1bea2c54f3e96`，2026-11-06。**隔离范围**：`PR #66` 独立文档审查；`PR #65@10331057` 仅作为可选外部证据链的边界参考，不从其未合并分支继承训练行为。**结论**：QG-0 分类完成；生产源码 **0 处改动**；本登记不是 QG-1 实现或 CUDA 资格晋升。GitHub 仓库树 517 项（非截断）已索引，以下登记覆盖 Component-wise 配置→Preview/Export→Start→直接 Trainer→优化器/运行时→Resume 的可达资格/能力阻断；不宣称逐行穷举所有训练器内部一般异常或 UI Schema 字段验证。

## 0. 判断口径（不得混淆）

| 代号 | 决策 | QG-1 行为 | 证据要求 |
| --- | --- | --- | --- |
| **H** | 真实硬错误 / 尚无可执行实现 | 继续 BLOCK | 语义不成立、运行实现缺失、依赖不可用、数据/参数/Checkpoint 不变量失败 |
| **A** | 高风险能力边界，尚不能证明实现可正确执行 | 现阶段继续 BLOCK，单独源码/实机审查后决定 | 不是把「没有证书」视为错误；是尚不能建立有用的参数身份、状态/进程所有权或运行语义 |
| **W\*** | 参考样例外的候选「实验性允许」 | 具体验证实现可达且保留运行时审计后 WARN；**不得现在直接放行** | 真实产线配置消费、参数路由、dtype/状态与资源预算必须核查 |
| **E** | 纯资格/证明覆盖限制 | 不再独立构成 BLOCK；转成证据等级及 WARN | 保留正式证据签发的拒绝，禁止「无资格」伪装成「已通过」 |
| **N** | 现有正常功能或产品约束 | 不作为 QG 解锁对象 | Standard 模式、正常 mixed BF16、单任务启动等独立约束 |

`A` 与 `W*` 的区别至关重要：**未测试**并不能推导出**必然错误**，但也无法证明**已实现可执行**。两类都不能在 QG-0 阶段偷偷移除旧 blocker；QG-1 先把诊断分类结构化，QG-3 再逐项真正解锁。

代码位置以固定 `main` 为准。路径省略规则：`E`=`mikazuki/parameter_policy_execution.py`；`P`=`mikazuki/parameter_policy.py`；`C`=`mikazuki/parameter_policy_compat.py`；`M`=`mikazuki/parameter_policy_matrix.py`；`S`=`mikazuki/sd3_full_bf16_scope.py`；`O`=`mikazuki/optimizer_profiles.py`；`T`=`mikazuki/parameter_policy_trainer.py`；`R`=`mikazuki/parameter_policy_runtime.py`；`K`=`mikazuki/parameter_policy_torch.py`。所有路径位于仓库根。

## 1. 完整调用链与作用范围

`/training/preview`：`mikazuki/app/training_api.py:179–188` → `mikazuki/training_request.py:179–236` → `P:543–661` → `C:203–414` + `E:479–598` + Component/target 检查；`parameter_policy_editor.py:446–486` 将所有 blocker 混为 `runtime_ready=false`。Preview 不进行模型文件资产检查，但仍报告资格限制。

`/training/export`：`training_api.py:191–201` 与 Preview 使用同一组逻辑；导出可包含 blocker，但必须保持安全/结构检查（尤其托管 sidecar）。

`/run`：`training_api.py:236–268` → `prepare_request_config(..., launch=True)`（`training_request.py:223–236` 有任何 blocker 即抛出）→ 实际资产/Trainer 读校验（`training_validation.py:11–112`）→ materialize sidecars → `training_launcher.py:20–101` 启动。**Start 不可仅跳过 Python 前置列表，否则直接 Trainer 缺口仍在。**

直接 Trainer：`scripts/dev/library/dts_parameter_policy_bridge.py:32–43` / stable 同名桥先执行严格 bool 检查 → `T:1618–1653` 再调用同一个 `E:479–598`；创建 Session → `R:490–629`、`K:157–260` → `T:950–1108,1297–1333,1395–1575` 的真实参数/状态检查。

旧资格点：`E:409–433` 单字符串 `_qualification_blocker` 将证书缺失/`pending`/`unsupported` 统统转换为运行阻断；`E:533–541` 首先拒绝 `sd3-lora`，因此当前 `S:282–386` 的 R1/R2 细粒度验证被提前短路。`E:318–353` 的执行身份构造**本来就不依赖资格表**，是迁移的关键现成基础。

**隔离**：`Standard` 不进入 Component 前置门槛；`full_bf16=false` 时 `E:491–499` 不再检查证书；原 `PR #60/#61` 的 SD3 normal/mixed BF16 真实 Fresh→Resume 不依赖 `PR #65` 的独立 Runner 签发服务。任何 QG 修改都不能借检查从属证据签名而阻断该正常生产路径。

## 2. G：入口、解析与用户可见阻断（15 项）

| ID | 已有条件 / 源码 | 分类 | QG-1 处置 |
| --- | --- | --- | --- |
| G01 | Component 后端必须在 `M:20–67` 10 种已接入 trainer 的矩阵内；`P:563–572` | **H** | 未映射或未知后端继续 BLOCK；「已注册」不等于所有训练模式已认证 |
| G02 | `P:559` 与 `P:164–395` 结构化 Policy、合法 Profile/Component/引用/数值 | **H** | 无效 Sidecar、引用孤儿、错误组件/无 trainable 保持 BLOCK |
| G03 | `parameter_policy_editor.py:54–96` 仅把 `component_support=supported` 展示为可选 optimizer | **H/A** | 已无实现的 planned 不加入可运行菜单；能力标签与 Full-BF16 证书须分开呈现 |
| G04 | `training_request.py:208–218` 禁止覆盖 host-owned Parameter Policy/Multi-Caption sidecar | **H** | 禁止伪造/替换 sidecar，保留原始配置语义 |
| G05 | `training_request.py:223–236` 任意字符串 blocker 阻断 Start | **E+H 混合** | 使用结构化 hard_errors 与 warnings；只用 hard_errors 决定真实 BLOCK |
| G06 | `parameter_policy_editor.py:479–486` `runtime_ready=not bool(blockers)` | **E+H 混合** | `runtime_ready` 仅由 hard errors 决定；增补兼容式 `runtime_warnings/evidence_tier`，不可重定义为已认证 |
| G07 | `training_api.py:179–201` Preview/Export 只做 model-free 检查 | **N** | 保持无本地模型 I/O；允许报告「正式资格未验证」 |
| G08 | `training_api.py:236–257` Start 在 materialize/创建任务前验证 | **H** | 保留只读 preflight 和失败零副作用 |
| G09 | `training_validation.py:59–110` 启动脚本/模型/外置编码器/数据集/Prompt 真实路径、类型可用性 | **H** | 无文件/不兼容格式继续 BLOCK；不能以 experimental 越过 |
| G10 | `training_validation.py:29–41` Multi-Caption + Text Encoder 输出缓存/自定义 dataset_class | **H** | 现有动态字幕与缓存语义冲突，保持 |
| G11 | `training_request.py:189–199` 原始 GUI → sidecar → effective config 先后次序 | **N/H** | 用户 opt-in 不得通过改变这条路径、静默写配置或改变训练身份 |
| G12 | `gpu_runtime.py:8–54` CUDA 不可用、设备 ID 无效或重复，实际 GPU 列表 | **H** | 不能因为缺少 CUDA 资格就自动回退 CPU，仍保持错误 |
| G13 | `training_launcher.py:71–83` 已有训练任务时拒绝重入 | **N** | 任务互斥独立于资格评估，不随 QG 改动 |
| G14 | `scripts/dev/library/dts_parameter_policy_bridge.py:32–43`、stable 桥同级严格布尔前检 | **H** | 直接 Trainer 与 Start/Preview 对参数类型一致，不得绕过 |
| G15 | `training_config.py:89–110` Anima Full Low RAM 未实现、lowram/highvram 冲突 | **H** | 明确未实现与互斥配置继续 BLOCK |

## 3. E：纯证据状态、已验证范围与执行模式（15 项）

| ID | 已有条件 / 源码 | 分类 | QG-1 处置 |
| --- | --- | --- | --- |
| E01 | `E:169–213` 10 后端 Full-BF16 release rows：SD1/SDXL LoRA qualified；SD DreamBooth unsupported；SD3/Flux/Chroma/SDXL full/Anima pending | **E**（证书）+ **H/A**（真正实现能力） | 拆开资格表与能力表；`pending` **本身**不能决定 Start；`unsupported` 必须另查能力原因 |
| E02 | `E:409–433` missing / pending / qualified-but-no-evidence-ID 转致命字符串 | **E** | 仅阻断**签发正式资格**；运行许可由能力与不变量决定 |
| E03 | `E:533–541` backend pending 过早返回，掩盖 SD3 具体问题 | **E** | 分别计算 hard/warnings，不能早退吞没独立的真实问题 |
| E04 | `E:216–275` optimizer 资格表：AdamW/Muon qualified；BNB、Lion、SGD、ScheduleFree 等 pending | **E** + **A** | Full-BF16 证书外提；另按 K/O 真实实现与保存/恢复能力审查 |
| E05 | `E:562–598` **实际引用**的 optimizer primary 和 fallback 各自需证书；unused Profile 不参与 | **E** | primary/fallback 继续审查真实路由；证书仅影响 evidence_tier |
| E06 | `E:460–476,591–596` AdamW args 只允许 betas/eps/weight_decay | **E/W*** | 限定的是 D0 参考 case，其他 `torch.optim.AdamW` 认可参数应做语义/状态检查后降为 WARN，不凭全局字符串拦截 |
| E07 | `E:507–516` SD1.x verified、SD2.x (`v2=true`) pending | **E/A** | SD2 Full-BF16 标 unverified，先核查真正的 trainable Tensor/Trainer 支持，再决定是否实验性可执行 |
| E08 | `E:518–531` SDXL Full-BF16 + gradient checkpointing 的 Text Encoder embedding ownership 风险 | **H/A** | 源码已有具体漏管风险；保留 BLOCK 直到修复/独立 ownership 测试 |
| E09 | `E:304–315,318–353` Full-BF16 必须 mixed_precision=bf16；执行身份不读取资格状态 | **H** + **N** | 不改变 BF16 dtype/精度契约；资格变更不得影响 Checkpoint identity |
| E10 | `E:356–405` num_processes!=1 或 distributed_type!=NO 以证书理由 BLOCK | **A** | 多进程改变 CompositeOptimizer/状态 ownership；先保持 BLOCK，后用实现/跨进程证据开能力 |
| E11 | `M:109–129` Component 选择多 GPU 阻断 | **A** | 不因证书分离立刻开启 DDP；与 E10 共享统一实际能力判断 |
| E12 | `E:291–301,479–499` `full_bf16` 不启用就不查 qualification | **N** | 保留 Standard/正常 mixed-BF16 无侵入；严格输入类型仍检查 |
| E13 | `FULL_BF16_*_QUALIFICATIONS` 的 `evidence_case_id` 存在/缺失 | **E** | 缺正式 ID 不允许资格提升；绝不改变 optimizer/runtime topology |
| E14 | PR #65 `tools/sd3_full_bf16_worker_step_adapter.py:755–796` 的原始 Load 委托在独立父证明未取得时拒绝 | **E（隔离轨道）** | 仅外部认证 Worker 的拒绝；不得引入普通 SD3 production Resume 作为全局前置条件 |
| E15 | 固定测试样例的 CUDA `exact-head/clean-tree`、Runner 外部签名、跨账户服务证据 | **E（隔离轨道）** | 只约束正式远程证据/Release 认证；用户普通训练不以操作系统信任根签名作为依赖 |

## 4. C：Component 表达能力与运行语义（25 项）

| ID | 条件 / 源码 | 分类 | 处置与依据 |
| --- | --- | --- | --- |
| C01 | `C:217–229` 真实 Boolean 类型检查，禁止 TOML `"false"` 等字符串 | **H** | 运行时 Python truthiness 会错误启用开关 |
| C02 | `C:234–245` Anima `compile` | **A** | 改变 DiT 模块 identity，需要参数/Checkpoint 审计 |
| C03 | `C:246–256` `torch_compile` / Accelerate Dynamo | **A** | 包装会改变参数身份可观察性；不能立即降级 |
| C04 | `C:258–269` `full_fp16`、GradScaler 及状态 | **A** | BF16 路径不等于 Full-FP16 安全性 |
| C05 | `C:271–280` `fp8_base` / `fp8_base_unet` | **A** | 参数 dtype/device/forward ownership 风险，部分 stock FP8 dtype 明确不可用 |
| C06 | `C:282–291` `fused_backward_pass` | **H** | 尚无 Composite 等价 step owner |
| C07 | `C:293–302` `fused_optimizer_groups` | **H** | 新的优化器分组/step 路径缺等价 Component 实现 |
| C08 | `C:304–313` `blockwise_fused_optimizers` | **H** | block 级构造/step 状态无法无损映射 |
| C09 | `C:315–321` `deepspeed` | **A** | 可能托管 Optimizer/distributed ownership，需另实现与状态验证 |
| C10 | `C:323–330` CPU / Unsloth checkpoint offload | **A** | 参数/梯度驻留和反向图改变；非普通证书警告 |
| C11 | `C:332–339` swap blocks 三类数量字段 | **A** | 优化器参数设备审计需要适配迁移时序 |
| C12 | `C:341–366` 预载 LoRA network_weights + Text Encoder 输出缓存 | **H** | 缓存先于 adapter 权重加载，conditioning 与实际权重不一致 |
| C13 | `C:368–390` Anima base_weights + TE cache | **H** | Qwen/DiT merge 时序与缓存不一致 |
| C14 | `C:392–402` scale_weight_norms 修改 frozen LoRA 权重 | **H** | Component policy 的 Train=false 将被破坏 |
| C15 | `C:449–458` SDXL Full block_lr=23 个 U-Net 分组 | **H** | 现有 Component schema 无损表示不了 block-level LR |
| C16 | `C:460–495` SD DreamBooth 动态停止 TE 训练 | **H** | 静态 Train=true/false 无法表达冻结时刻 |
| C17 | `C:497–505` SD/SDXL block LR Network Args | **H** | U-Net 子组 LR 语义超出 Component 分组 |
| C18 | `C:507–513` LoRA+ lora_up 独立 LR | **H** | 需要参数级不同 LR；非单纯未验收 |
| C19 | `C:515–521` regex network_reg_lrs | **H** | regex 子组 LR 无损路由缺失 |
| C20 | `C:523–534` 非 canonical network_module | **A** | 不能假设自定义 LoRA 的参数名/optimizer group 与默认相同 |
| C21 | `P:421–488` Component 集合不匹配/有效目标不可用/无 Trainable | **H** | 实际模型 Target/参数路由不成立 |
| C22 | `P:491–540` trainable TE 与 TE cache（SDXL/SD3/Flux/Chroma） | **H** | 实际 Trainer 已对 T5XXL Cache 建立限制；CLIP 只在支持的部分缓存模式下保留 |
| C23 | `P:612–634` Anima Qwen3 permission/cache/Qwen-only | **H** | 当前 target 能力、缓存和无 DiT 路由缺口；需独立改实现 |
| C24 | `training_gui_semantics.py:350–368` 噪声互斥、latent cache 与 color_aug/random_crop、TE cache 与 shuffle_caption、OFT 后端限制 | **H** | 已有正常训练器约束，不属于 Full-BF16 资格耦合 |
| C25 | `training_validation.py:29–41` Dynamic Multi-Caption 与 TE output cache / custom dataset_class | **H** | 已知数据语义不兼容，继续作为全局实质错误 |

> **重要**：多个 `C` 项即使错误文本写着「尚未验证」，也不能简单判成证书不足；必须查看是否已有参数/优化器/状态的等价执行实现。`C06–C08` 属无等价执行实现，`C02–C05/C09–C11/C20` 是需独立实施审查的高风险边界。

## 5. S：SD3 R1/R2 资格样例硬编码（28 项）

`S:25–100,146–214,229–385` 是测试候选范围审查，不是 SD3 一般训练器能力声明。必须区分**完整 Parameter Policy 结构**与**参考样例的唯一取值**：

| ID | 条件 / 源码 | 分类 | 合理的下一步 |
| --- | --- | --- | --- |
| S01 | `S:153–155` 必须包含完整注册的七种 SD3 Component IDs | **H** | 保持 schema/路由完整性 |
| S02 | `S:164–168` Train 集合必须恰好 MMDiT Attention/MLP/Modulation，其他四组冻结 | **W*/A** | 非参考组件组合若路由支持可实验；训练 TE、MMDiT Other 等需独立参数所有权和缓存审查 |
| S03 | `S:171–214` 只接收 R1 AdamW 或 R2 Muon+AdamW 两种 Profile 形状 | **W*/A** | 已实现的其它 Profile 可以单独走实验能力验证；真未实现的一律 BLOCK |
| S04 | `S:173–176,191–197` AdamW/Muon 参数对象必须与样例完全相等 | **W*** | 依据实际 optimizer 构造支持参数、调度器和 Checkpoint 状态判定 |
| S05 | `S:180–188,201–209` 每路学习率及 fallback LR 固定 `1e-4` | **W*** | 正数/范围/优化器路由才是硬限制，固定基准值仅为实验覆盖 |
| S06 | `S:199–208` R2 每组显式 Muon+AdamW Fallback | **H/A** | Muon 不合格 tensor 必须有真实 fallback；是否每组都需 fallback 用物理类别判断 |
| S07 | `S:303–315` `network_module=networks.lora_sd3` | **H/A** | 非默认网络模块不能靠警告直接执行，除非独立参数路由适配 |
| S08 | `S:303–315` 全局 `optimizer_type=AdamW` | **W*/A** | 区分 legacy argparse 占位和 Composite Optimizer owner；不可让其覆盖 Policy |
| S09 | `S:303–315` `lr_scheduler=constant` | **W*/A** | 若 CompositeLRScheduler 已支持其它 scheduler，需要保存/恢复一致性测试 |
| S10 | `S:303–315` `save_model_as=safetensors` | **H/A** | 当前证据/加载契约针对 safetensors；其它格式需真实保存/加载支持才放行 |
| S11 | `S:308–315` `weighting_scheme=uniform` | **W*/A** | 其它噪声权重策略需检查 Forward/Grad 语义、不能仅看证书 |
| S12 | `S:309–315` `loss_type=l2` | **W*/A** | 其它 loss 要确认实际 SD3 Trainer 与 dtype 正确性 |
| S13 | `S:317–321` `network_train_unet_only=true` | **W*/A** | TE adapter 训练需要真路由/TE 缓存及 dtype 适配 |
| S14 | `S:349–357` `network_dim=4` | **W*** | 维度/rank 由训练器真实支持决定，超出证明样例标记未验收 |
| S15 | `S:349–357` `network_alpha=4` | **W*** | 与 rank 一起检测合法范围及 adapter tensor shape |
| S16 | `S:349–357` `train_batch_size=1` | **W*** | 有效正整数与 VRAM/OOM 是实际执行约束 |
| S17 | `S:349–357` `gradient_accumulation_steps` R1=1、R2=2 | **W*/A** | 一般累积可实验，但 R2 child/scheduler 真实 step 需要验收 |
| S18 | `S:115–139,358–359` 固定分辨率 `512×512`，禁止 bucket | **W*** | 一般 SD3 Trainer 已支持 bucket；其它分辨率仍受模型/VRAM/位置编码实际约束 |
| S19 | `S:349–357` `max_data_loader_n_workers=0` | **W*/A** | Worker>0 已有 DataLoader，但恢复可重复性不保证，需 source check |
| S20 | `S:60–67,341–348` dropout、随机 positional crop、training_shift、T5 长度固定 | **W*/A** | 参数有效性检查仍保留；数值变化属于样例外或改变条件化语义 |
| S21 | `S:75,95` `enable_bucket` / `persistent_data_loader_workers` 必须关闭 | **W*/A** | 单独确认 DataLoader 批次顺序、Resume 几何；非自动解锁 |
| S22 | `S:76–80,83–89` gradient checkpointing、FP8/full_fp16、DeepSpeed、compile、fused/Offload | **A/H** | 与 `C02–C11` 是同一能力门槛，**不可**因移除 R1/R2 形状检查而意外开启 |
| S23 | `S:81–82,90–92` `no_half_vae` / TE/cache-to-disk/offload | **W*/A** | VAE fp32 路径、Cache 时序需看实际依赖；不是 BF16 parameter dtype 的代名词 |
| S24 | `S:93–100` TE-only、Attention masks、masked loss、HuggingFace Resume | **A/H** | 改变模型/target/数据/恢复权属，需分别建立兼容语义 |
| S25 | `S:335–340` `network_args`、`network_weights`、`base_weights`、`optimizer_args` 非空统统阻断 | **A** | 部分可能仅资格参考范围，但加载预置 LoRA 与最终 optimizer owner 需真审计 |
| S26 | `S:44–50,335–340` `dataset_config`、`dataset_class`、`in_json`、reg_data_dir、Multi-Caption、conditioning_data_dir、t5xxl_device/dtype、min/max timestep | **W*/A** | 按训练器对每一种源/override 是否消费和是否改变优化器/数据流逐项拆分；禁止一键全解 |
| S27 | `S:229–279` 绝对路径、.safetensors header、SD35 Medium shape/loader、缺外置 CLIP-L/G/T5/VAE | **H + E/A** | 缺文件/无法加载的 checkpoint/必要依赖仍 BLOCK；「仅 SD3.5 Medium」是当前证据范围，其他 model 实现需另查；相对路径是否可加载应与真实 loader 一致 |
| S28 | `S:282–385` `sd3_full_bf16_scope_blockers` 被官方 release gate 后置并作为 Start/Direct 强制拦截 | **E + H/A/W*** | 拆成「实际 capability/asset/invariant」与「R1/R2 资格样例匹配」；前者 BLOCK，后者仅标记证据范围 |

## 6. O：Optimizer 实现能力与生命周期（12 项）

| ID | 源码 | 分类 | 处置 |
| --- | --- | --- | --- |
| O01 | `O:43,197–208` AdamW/Muon 定义 supported；`K:162–166,245–256` 有真实构造路径 | **N/H** | 保持真实 Profile 和 Muon 参数 eligibility，已在 D0 CUDA 已验证参考 |
| O02 | `O:44–47` AdamW8bit/PagedAdamW8bit/PagedAdamW/PagedAdamW32bit；`K:120–154` | **E+A** | 构造路径存在，但缺 Full-BF16 save/resume 实机证据；优先选一个 BNB 变体做专用实验性资格 |
| O03 | `O:48–50` Lion、Lion8bit、PagedLion8bit；`K:189–214` | **E+A** | 实现可达；需要 verify 实际 dtype/state/依赖与 scheduler |
| O04 | `O:51–52` SGDNesterov/8bit；`K:100–117,168–187` | **E+A** | 已有构造与 Nesterov 必需值；真实 GPU/resume/8bit 需复核 |
| O05 | `O:53–79` 三种 ScheduleFree；`K:216–243,381–398` | **E+A** | train/eval 和 optimizer-managed scheduler 已实现；resume/config 仍需独立验证 |
| O06 | `O:80–178` DAdaptation、Prodigy、AdaFactor、ProdigyPlusScheduleFree restricted | **H** | adaptive LR/schedulefree Profile 专用 owner 未全落实，不因无证书直接解锁 |
| O07 | `O:179–195` Custom/CAME planned | **H** | 无通用受控构造器/能力语义，不是单纯缺 GPU 证书 |
| O08 | `O:237–273` require_unrestricted_component_optimizer；`P:649–660` | **H + E** | 能力限制独立于证书；保留 restricted/planned 阻断，剥离重复资格消息 |
| O09 | `R:225–425` 同一物理 Parameter 不能重复 route；policy/profile/LR 一致 | **H** | 参数/拓扑真实完整性 |
| O10 | `R:490–629` 无 trainable、缺 Profile、group LR 违反实现能力 | **H** | 真实运行约束 |
| O11 | `K:63–73,120–154,157–260` 第三方依赖/优化器类缺失/构造失败 | **H** | 缺依赖时明确错误；不得静默替换 optimizer |
| O12 | `K:313–500,540–721` CompositeOptimizer/CompositeLRScheduler 子状态/调度器一一匹配 | **H** | 执行和 Resume 不变量，保留严格校验 |

> 「Optimizer Profile supported」说明 **Component v1 代码里有相应构造/语义路径**；「Full-BF16 qualified」说明 **某个固定真实 CUDA case 已被验收**。两者不应同义，也不应互相推断。

## 7. R：运行时参数审计和 Checkpoint/Resume（19 项）

| ID | 源码 | 分类 | 处置 |
| --- | --- | --- | --- |
| R01 | `T:1627–1653` Direct Trainer policy + semantic + execution preflight | **E/H 混合** | 直接 Trainer 使用相同结构化能力评估；不得仅放开 GUI |
| R02 | `T:623–751` `requires_grad`、frozen 与 live-root 参数别名/身份 | **H** | 防止训练对象被替换或误训冻结参数 |
| R03 | `T:950–978` optimizer-owned Full-BF16 trainable 参数必须真正 torch.bfloat16 | **H** | 没有 BF16 tensor 就不能声称/执行 Full-BF16 契约 |
| R04 | `T:981–1007` runtime processes/distributed type | **A** | 一体化重新设计进程/优化器 ownership，先保持硬阻断 |
| R05 | `T:1009–1062` optimizer topology、重复/泄漏参数、集合一致 | **H** | 核心所有权不变量 |
| R06 | `T:1064–1110` Accelerate device 与 trainable Tensor device 一致 | **H** | 防止 CPU/CUDA、卡号错误/隐式转换 |
| R07 | `T:1310–1333` checkpoint/save/post_resume 都执行 runtime contract | **H** | 动态 invariant 无法仅靠 Preview 证明 |
| R08 | `T:1395–1445` checkpoint manifest policy_hash、optimizer topology、scheduler、execution identity exact match | **H** | 继续作为真正 resume 的完整性前置 |
| R09 | `T:1450–1553` 进度 sidecar/完整 child optimizer/scheduler state | **H** | 不得因为未认证而缺失必要完整状态 |
| R10 | `T:1169–1295` Load progress schema、版本、Checkpoint ID/parent、实际 optimizer step 与预期一致 | **H** | 真实恢复校验，不应变成可点击忽略的 warning |
| R11 | `T:1555–1575` Accelerate save/load prehook 在状态修改前验证 | **H** | 保持 prehook 顺序，防止先恢复再拒绝 |
| R12 | `scripts/dev/library/sd3_forward_contract.py:87–139` SD3 extra-attention forward semantics sidecar | **H** | Forward 协议不同会造成错误 resume；加载为新 stage 可以是另一个受支持操作 |
| R13 | `scripts/dev/library/sd3_resume_cursor.py:41–97,120–167` SD3 Save 的 Epoch/Step/accum/process/data geometry 和 sidecar | **H** | 恢复游标和真实 DataLoader 状态是实际正确性条件 |
| R14 | `scripts/dev/library/sd3_resume_cursor.py:85–91,153–166` Epoch 中途禁止 Resume | **H（当前实现）** | 没有等价 sampler 顺序/RNG 连续性算法；**暂不能降为 Warning**，未来改实现再开放 |
| R15 | `scripts/dev/library/sd3_resume_cursor.py:66–68,155–158` SD3 Resume num_processes!=1 | **A/H** | 真实多进程游标/数据分片缺完整实现；保留阻断 |
| R16 | `mikazuki/sd3_resume_ui.py:48–84,159–184` GUI 对原始 Sidecar/总 Steps/累积一致性 | **H** | 不允许残缺/冲突恢复；原 Checkpoint 不存总步数只能 WARN 用户核对 |
| R17 | `mikazuki/sd3_resume_ui.py:87–150` 新 GUI Epoch/Steps 与旧生命游标冲突 | **H** | 不能让两个位置同时控制恢复进度 |
| R18 | `mikazuki/sd3_resume_ui.py:188–192` Resume 多 GPU 选择阻断 | **A/H** | 与 R15/E10 和真实 Cursor 功能相绑定，不是单纯正式证书 |
| R19 | `scripts/dev/sd3_train_network.py:61–118,214–290` T5XXL train+cache、swap/offload 互斥、CLIP 配对及真实训练 flags | **H/A** | 标准训练器已经有自己的真实约束；新的 QG 不可跳过其保护 |

## 8. 程序路径边界：三种「可继续」不能相互替代

1. **Normal/mixed-BF16 production path**：PR #60/#61/#62 真实训练/保存/新进程恢复已在限定范围通过。保持正常 ALLOW，不插入 PR #65 的外部身份凭据。
2. **Full-BF16 experimental production path**：需要先建立 `BF16 实际参数 dtype、optimizer owner、grad/step、Save→Load` 可用的具体配置。仅把证书降为 WARN 不代表现有 `S/R/O` 的硬约束自动全部消失；这是 QG-3 源码+GPU 的主体。
3. **Independent attested physical qualification path**：PR #65 的 Ed25519/Runner/OS/nonce、九个原生 Checkpoint 文件和重建父证据只决定能否签发**外部可信资格**，不能变成用户个人本地恢复的依赖，也不能因实验性训练成功而自动改成 GREEN。

## 9. 测试所有权和 QG-1 必须显式迁移的旧断言

| 测试文件 | 当前保护/断言 | 迁移要求 |
| --- | --- | --- |
| `tests/test_parameter_policy_execution.py` | `test_pending_and_unsupported_backends_fail_closed`、`test_unknown_backend_fails_closed`、`test_qualified_without_evidence_still_fails_closed`、`test_d1_sd_lora_full_bf16_is_scoped_to_sd1`、`test_adamw_first_release_variant_is_narrow` | 分开「正式资格不得通过」与「真实实现能力/实验性运行判断」；未知后端/无实现继续拒绝；已验证证据表不得变 |
| `tests/test_sd3_full_bf16_scope.py` | 28 条围绕唯一 R1/R2 shape、固定 512/rank/LR/bucket/禁用项的现有严格断言 | **保留原样例作为正式资格选择器**；新建独立生产能力测试，而不是直接删除 negative tests |
| `tests/test_parameter_policy_request_contract.py` | pending Full-BF16 拒绝 Start、Preview 暴露 blocker、直接 Trainer/多卡等 | 新增 warnings 与用户选择后的 Start 正向；真实语义 BLOCK 仍在无 materialize 阶段拒绝 |
| `tests/test_parameter_policy_trainer_runtime_smoke.py` | 40 条真实 dtype、参数身份、Optimizer step、Checkpoint Manifest/Resume | 保持全部关键 Runtime Invariant，实验性权限不能绕过 |
| `tests/test_parameter_policy_compat.py` | 编译/FP8/融合/未实现 LR 语义不兼容 | 仅迁移确有真实可执行实现的项，其余维持 BLOCK |
| `tests/test_parameter_policy_runtime_bools.py` | 字符串 pseudo-bools 在 GUI/Trainer 前置拒绝 | **必须保持全 GREEN**，包括通过 `ui_custom_params` 注入的情况 |
| `tests/test_parameter_policy_runtime.py` + `tests/test_optimizer_profile_capabilities.py` | 参数路由拓扑、restricted/planned、dependency、group LR | 实际语义错误不得被证书拆分影响 |
| `tests/test_sd3_resume_cursor.py` + `tests/test_sd3_ui_resume.py` | 中途/多进程不支持、旧 sidecar、损坏/错配和步骤游标 | 保持硬拒绝，除非先实现等价采样器/分布式恢复机制 |
| `tests/test_parameter_policy_step6f_contract.py` + `tests/test_parameter_policy_step7_contract.py` | freeze release & blocker/semantic 字段、Start sequencing | 对新结构化字段更新**明确**契约，不用隐式改变旧 runtime_ready 含义 |
| `tests/test_training_semantic_contract.py` + `tests/test_training_api_overlay_contract.py` | Standard 和基础后端 UI/Trainer 真实语义、请求/Sidecar | 不能出现新模式污染 Standard、旧 TOML/Bundle 导入语义 |
| PR #65 自己的 C4-C2、B1、Linux/Windows Host Tests-only RED | 外部证据安全服务尚未安装/载入原始 Checkpoint | **继续原预期 RED**；不得通过普通用户 WARN/opt-in 偷偷给 Worker 正式签发权限 |

**QG-1 新增 Tests-only RED 起点**（不要在 QG-0 提前写生产代码）：
- `QG101` 同一合法 Full-BF16 执行语义分别搭配 `pending` 与 `qualified` metadata：**execution identity/topology/checkpoint schema 完全一致**，只是 evidence tier / warning 不同。
- `QG102` 真正未接入后端、planned/未实现优化器、伪 Boolean、错误混合精度、重复 optimizer 参数和已知 TE cache 冲突：全部 BLOCK。
- `QG103` R1/R2 示例外的合法 rank/alpha/LR/resolution：应能进入**能力审查通路**并显示「超出已认证样例」，但在证明 production 支持之前不可强行通过所有前置条件。
- `QG104` Preview/Export 同一配置无写入；Start 实际只有 hard_errors 才禁止创建任务；Direct Trainer 再次执行相同硬检查，不能绕过。
- `QG105` 实验性任务有显式确认/Headless 声明，记录在**独立审计元数据**，不进 stable execution identity，不更改 `FULL_BF16_*_QUALIFICATIONS` 或 PR #65 的可信签发事实。
- `QG106` SD3 常规 mixed BF16、SD1/SDXL 已认证场景、GPU 选择、真实 Save/Load/epoch-boundary Resume、ABI 与旧 TOML 导入维持原状。

## 10. QG-0 判定与开发顺序（冻结）

**结论 1：最明确的过度阻断**是 `E02/E03/E04/E05/E06`：缺少证书就阻止已有真实代码路径。优先从证书读取端解耦，不能通过临时把 `status` 改成 `qualified` 或强行发一个虚构 evidence_case_id 修复。

**结论 2：最明确的「测试样例误当产品上限」**是 `S03–S05/S14–S19`：精确 optimizer args、固定学习率、rank、alpha、batch、resolution、累积和 worker 数。后续升级为 QG-3 的真正生产运行检查，不直接删除资格选择器。

**结论 3：绝不降级的安全/正确性约束**是 `C01/C06–C08/C12–C19/C21–C25`、`O06/O07/O09–O12` 和 `R02/R03/R05–R14/R16/R17`；未经等价新实现不能把它们伪装成实验性支持。

**结论 4：需要单独深度审查的风险边界**是 multi-GPU/DDP、DeepSpeed、FP8/full_fp16、torch.compile、block swap/offload、SD3 可训练 Text Encoder、批次中途 Resume、各 BNB/ScheduleFree Full-BF16 变体。这些当前仍 HOLD/BLOCK，是基于实现不确定和所有权/恢复风险，**不是因为我们崇拜证书**。

**建议开发顺序**：
1. QG-1A：引入纯 `ExecutionReadiness` 数据模型及稳定 issue code（`hard_errors` / `warnings` / `evidence_tier` / `capability_status`），先 Tests-only RED，再最小 GREEN。不得改训练/Loader。
2. QG-1B：把 E 证书状态从 blocking logic 分离；保留原资格表不变，保证硬错误照旧，回归旧动态约束。
3. QG-2：Preview/Export/Start/Direct parity 和 GUI/headless 实验性确认（仅具体 `implemented/experimental_candidate`），只改运行时许可、不伪造 Checkpoint identity。
4. QG-3：SD3 Full-BF16 生产训练器支持矩阵 & 可用配置实验性解锁，优先 R1 AdamW 和 R2 Muon+explicit AdamW fallback；每条变体都保留真实 dtype/optimizer routing/Checkpoint/Forward 保护。
5. QG-4：真实 GPU 短训、Step-1 Save、跨进程 Resume Step-2 及优化器/梯度/LoRA/调度器证据。通过后才考虑扩大可选 rank/resolution/optimizer 家族。
6. QG-5：独立 Strict Review；PR #65 外部独立签发 B3/P1-C 暂缓，不作为本地训练的发布前置条件。

**QG-0 完成条件**：本审查登记源代码位置、作用域、真实原因、分类、现有测试归属和 QG-1 迁移方向；验证 PR #66 diff 只增文档；任何具体实验性组合的正向开关都等待新的 Tests-only RED→GREEN 和实机审查。本阶段不执行 GPU、不宣称新的训练功能已通过。
