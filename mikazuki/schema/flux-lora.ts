Schema.intersect([
    Schema.object({
        model_type: Schema.union(["flux", "chroma", "anima"]).default("flux").description("模型架构：FLUX / Chroma / Anima"),
        pretrained_model_name_or_path: Schema.string().role('filepicker', { type: "model-file" }).default("./sd-models/model.safetensors").description("底模路径；Anima 请选择 DiT safetensors"),
        resume: Schema.string().role('filepicker', { type: "folder" }).description("从某个 save_state 保存的中断状态继续训练"),
    }).description("训练用模型"),

    Schema.union([
        Schema.object({
            model_type: Schema.union(["flux", "chroma"]).required(),
            ae: Schema.string().role('filepicker', { type: "model-file" }).description("AE 模型文件路径"),
            clip_l: Schema.string().role('filepicker', { type: "model-file" }).description("CLIP-L 模型文件路径"),
            t5xxl: Schema.string().role('filepicker', { type: "model-file" }).description("T5-XXL 模型文件路径"),
        }),
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_model_variant: Schema.union(["base", "2.9b"]).default("base").description("Anima 模型版本：base = 标准 28 blocks；2.9b = 扩展 40 blocks（约 2.9B 参数）"),
            // Do not use an implicit default here. The legacy prebuilt frontend can
            // evaluate sibling unions before a default has been materialized into
            // the form model. Requiring an explicit value makes all mode-dependent
            // controls deterministic. Included presets already set this field.
            anima_training_mode: Schema.union(["lora", "finetune"]).description("Anima 训练方式：请选择 LoRA 或全参微调；显式选择可确保旧版 GUI 正确切换条件设置"),
            qwen3: Schema.string().role('filepicker', { type: "model-file" }).description("Anima 文本编码器：Qwen3-0.6B safetensors；也可手工填写本地 HuggingFace 模型目录，启动训练时后端会校验必填"),
            vae: Schema.string().role('filepicker', { type: "model-file" }).description("Anima VAE：Qwen-Image VAE safetensors / pth；启动训练时后端会校验必填"),
            t5_tokenizer_path: Schema.string().role('filepicker', { type: "folder" }).description("可选：T5 tokenizer 目录；留空使用 sd-scripts 内置配置"),
        }),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.object({
            model_type: Schema.union(["flux", "chroma"]).required(),
            model_train_type: Schema.string().default("flux-lora").disabled().description("实际训练种类"),
        }),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.object({
            model_type: Schema.union(["flux", "chroma"]).required(),
            timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift"]).default("sigmoid").description("时间步采样"),
            sigmoid_scale: Schema.number().step(0.001).default(1.0).description("sigmoid 缩放"),
            model_prediction_type: Schema.union(["raw", "additive", "sigma_scaled"]).default("raw").description("模型预测类型"),
            discrete_flow_shift: Schema.number().step(0.001).default(1.0).description("Euler 调度器离散流位移"),
            loss_type: Schema.union(["l1", "l2", "huber", "smooth_l1"]).default("l2").description("损失函数类型"),
            guidance_scale: Schema.number().step(0.01).default(1.0).description("CFG 引导缩放"),
            t5xxl_max_token_length: Schema.number().step(1).description("T5XXL 最大 token 长度（不填写使用自动）"),
            train_t5xxl: Schema.boolean().default(false).description("训练 T5XXL（不推荐）"),
            apply_t5_attn_mask: Schema.boolean().default(true).description("对 T5-XXL 编码器和 FLUX double block 应用注意力掩码"),
        }).description("Flux / Chroma 专用参数"),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift", "flux_shift"]).default("sigmoid").description("决定训练抽取哪些噪声强度。sigmoid 是通用默认；sigma 用于 logit_normal/mode；shift/flux_shift 适合明确要改变噪声分布的实验。无明确目的建议保持 sigmoid。"),
                sigmoid_scale: Schema.number().step(0.001).default(1.0).description("仅 sigmoid/shift/flux_shift 生效；1.0 为默认。>1 会让采样更分散、更多落在低/高噪声两端，<1 更集中在中间；通常保持 1.0。"),
                discrete_flow_shift: Schema.number().step(0.001).default(1.0).description("仅 sigma/shift 生效；1.0 为基线，增大通常把采样推向更高噪声区。建议先用 Timestep 分布诊断观察后再改。"),
                attn_mode: Schema.union(["torch", "xformers", "flash"]).default("torch").description("Attention 实现。torch 最稳妥；xformers/flash 在环境支持时可能省显存或提速。更换后建议先跑短 smoke test。"),
                blocks_to_swap: Schema.number().min(0).max(38).step(1).default(0).description("把部分 DiT block 交换到 CPU 以降低 VRAM；0=关闭。值越大越省显存，但会增加主机 RAM/PCIe 开销并变慢。仅显存不足时逐步增加；base 最多 26，2.9B 最多 38。"),
            }).description("Anima 核心训练参数"),
            Schema.object({
                ip_noise_gamma: Schema.string().description("Input Perturbation 正则化；留空/0=关闭。可在过拟合或想增强鲁棒性时尝试，常见起点约 0.05~0.1；过大可能增加训练噪声、降低收敛速度。"),
                ip_noise_gamma_random_strength: Schema.boolean().default(false).description("开启后每个 batch 在 0~ip_noise_gamma 间随机强度；用于让正则化更柔和。通常先关闭，确认固定 gamma 有益后再试。"),
                qwen3_max_token_length: Schema.number().min(1).step(1).default(512).description("Qwen3 最大 token 长度；512 为默认。只有 caption 经常超过截断长度时才提高；更长会增加文本编码与缓存开销。"),
                t5_max_token_length: Schema.number().min(1).step(1).default(512).description("T5 tokenizer 最大长度；512 为默认。通常无需修改，只有使用自定义 tokenizer/超长文本时才调整。"),
                split_attn: Schema.boolean().default(false).description("拆分 attention 计算以降低峰值显存，代价是速度下降；xformers 模式后端会自动开启。显存充足时通常关闭。"),
                vae_chunk_size: Schema.number().min(2).step(2).default(64).description("Qwen-Image VAE 空间分块大小；64 为默认。3D VAE 缓存阶段 OOM 时可尝试 32/16，越小越省显存但越慢；2D VAE 下影响较小。"),
                vae_disable_cache: Schema.boolean().default(true).description("关闭 3D VAE 内部 cache；单图训练通常保持 true 以避免额外显存占用。使用 2D VAE 时该项无效果。"),
                qwen_image_vae_2d: Schema.boolean().default(true).description("图像训练推荐保持 true：使用 image-only 2D Qwen-Image VAE，通常更快且更省显存。仅为兼容 3D VAE 工作流时关闭。"),
            }).description("Anima 高级性能与正则化（通常保持默认）").collapse(),
            Schema.object({
                show_timesteps: Schema.union(["off", "console", "image"]).default("off").description("只诊断当前 timestep/loss-weight 分布然后退出，不会训练。console 打印直方图，image 打开 matplotlib；调采样策略前建议先用 console。"),
                show_timesteps_resolution: Schema.string().default("1024").description("诊断时假定的图像分辨率；1024 为默认，可写 1024,768。主要影响 flux_shift 这类分辨率相关采样。"),
                show_timesteps_offset: Schema.number().default(0).description("仅用于预览 subset timestep_sampling offset；默认 0，仅 sigmoid/shift/flux_shift 有效。普通训练不需要修改。"),
            }).description("Timestep 分布诊断（仅调参/排查时使用）").collapse(),
        ]),
        Schema.object({}),
    ]),

    Schema.object(
        UpdateSchema(SHARED_SCHEMAS.RAW.DATASET_SETTINGS, {
            resolution: Schema.string().default("1024,1024").description("训练图片分辨率，宽x高。Anima 推荐从 1024,1024 起按数据集调整。"),
            enable_bucket: Schema.boolean().default(true).description("启用 arb 桶以允许非固定宽高比的图片"),
            min_bucket_reso: Schema.number().default(256).description("arb 桶最小分辨率"),
            max_bucket_reso: Schema.number().default(2048).description("arb 桶最大分辨率"),
            bucket_reso_steps: Schema.number().default(64).description("arb 桶分辨率划分单位；Anima 要求可被 16 整除"),
        })
    ).description("数据集设置"),

    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            dataset_repeats: Schema.number().min(1).step(1).default(1).description("仅 caption/metadata 数据集使用；1=不额外重复。小数据集想增加每轮采样次数时可设 2~10，但它不会增加信息量，过高更容易过拟合。"),
            cache_info: Schema.boolean().default(false).description("仅 DreamBooth 目录数据集有意义；缓存 caption/尺寸元信息以缩短下次启动时间。数据目录经常变化时建议关闭。"),
            debug_dataset: Schema.boolean().default(false).description("数据集检查模式：可视化/打印实际训练样本后退出，不训练。排查 caption、bucket、裁剪、mask 时临时开启。"),
            dataset_class: Schema.string().description("自定义 Python Dataset class（package.module.Class）。只有自己实现了 sd-scripts 兼容数据集时才填写；填写后普通 train_data_dir/dataset_config 不再作为数据源。"),
        }).description("高级数据集控制（通常不需要）").collapse(),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("finetune").required(),
            dataset_config: Schema.string().role('filepicker', { type: "file" }).description("多数据集/多 subset 的 sd-scripts dataset TOML/JSON。只有需要精细控制 repeats、bucket、subset 属性时才用；填写后普通 train_data_dir/in_json 会被忽略。"),
            in_json: Schema.string().role('filepicker', { type: "file" }).description("metadata JSON 数据集模式；适合已有统一图片目录+metadata 的数据。普通文件夹+同名 txt caption 训练时留空。"),
            skip_image_resolution: Schema.string().description("跳过尺寸不超过阈值的图片，例如 512 或 512,768。用于清理过小素材；没有明确数据清洗需求时留空。"),
            resize_interpolation: Schema.union(["lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area"]).description("图像 resize 插值算法。通常留空使用 trainer 默认；照片/插画常见可选 lanczos/bicubic，只有明确比较插值影响时修改。"),
        }).description("全参微调高级数据集来源（通常使用普通训练目录即可）").collapse(),
        Schema.object({}),
    ]),

    // Keep the existing save UI for LoRA. Only full finetune is fixed to the
    // Anima safetensors saver and therefore gets a format-specific save block.
    Schema.union([
        Schema.intersect([
            Schema.object({ model_type: Schema.union(["flux", "chroma"]).required() }),
            SHARED_SCHEMAS.SAVE_SETTINGS,
        ]),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("lora").required(),
            }),
            SHARED_SCHEMAS.SAVE_SETTINGS,
        ]),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("finetune").required(),
                output_name: Schema.string().default("aki").description("输出模型文件名（不含扩展名）。建议用能区分实验的名称，例如角色/数据版本+日期。"),
                output_dir: Schema.string().role('filepicker', { type: "folder" }).default("./output").description("checkpoint 输出目录。训练前确认磁盘空间；全参模型的周期 checkpoint 会占用较多空间。"),
                save_precision: Schema.union(["fp16", "float", "bf16"]).default("bf16").description("保存权重精度。BF16 训练通常保存 bf16；float 体积更大，除非有明确兼容需求不必使用。"),
                save_every_n_epochs: Schema.number().min(1).default(1).description("每 N 个 epoch 保存一次；默认 1。长训练/大模型磁盘压力高时可设 2~5，短测试一般保持 1。"),
            }).description("Anima 全参微调保存设置（固定 safetensors）"),
            Schema.object({
                save_n_epoch_ratio: Schema.number().min(1).description("按总 epoch 自动决定保存频率；例如 5 约等于整个训练保留 5 个阶段性节点。已明确设置 save_every_n_epochs 时通常留空。"),
                save_last_n_epochs: Schema.number().min(1).description("只保留最近 N 个 epoch checkpoint，避免长期训练占满磁盘；例如 2~5。留空=不自动删除。"),
                save_last_n_steps: Schema.number().min(1).description("按 step 保存时保留最近 N step 范围；只有使用 save_every_n_steps 时才有意义。"),
                save_state: Schema.boolean().default(false).description("额外保存 optimizer/scheduler/Accelerate state，以便精确断点续训。会显著增大文件量；只需要模型权重时保持关闭。"),
                save_state_on_train_end: Schema.boolean().default(false).description("训练结束时额外保存最终 state。只有准备继续同一 run 时有用。"),
                save_last_n_epochs_state: Schema.number().min(1).description("最多保留最近 N 个 epoch state；用于控制 resume state 磁盘占用。"),
                save_last_n_steps_state: Schema.number().min(1).description("按 step 保存 state 时只保留最近 N step 范围；通常与 step 保存策略配套。"),
            }).description("保存轮换与 Resume State（按需使用）").collapse(),
        ]),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.object({
            model_type: Schema.union(["flux", "chroma"]).required(),
            max_train_epochs: Schema.number().min(1).default(20).description("最大训练 epoch（轮数）"),
            train_batch_size: Schema.number().min(1).default(1).description("批量大小, 越高显存占用越高"),
            gradient_checkpointing: Schema.boolean().default(true).description("梯度检查点"),
            gradient_accumulation_steps: Schema.number().min(1).default(1).description("梯度累加步数"),
        }).description("训练相关参数"),
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("lora").required(),
            max_train_steps: Schema.number().min(1).description("可选：最大优化器步数；填写后优先按 step 控制训练"),
            max_train_epochs: Schema.number().min(1).default(1).description("最大 epoch；填写 max_train_steps 时后端优先 step"),
            save_every_n_steps: Schema.number().min(1).description("可选：每 N step 保存一次模型"),
            train_batch_size: Schema.number().min(1).default(1).description("批量大小"),
            gradient_accumulation_steps: Schema.number().min(1).default(1).description("梯度累加步数"),
        }).description("Anima LoRA 训练相关参数"),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("finetune").required(),
                max_train_steps: Schema.number().min(1).description("按 optimizer step 限制训练长度；填写后优先于 epoch。做 smoke test 常用 2~20，正式训练按数据量/收敛曲线决定。"),
                max_train_epochs: Schema.number().min(1).default(1).description("按完整遍历数据集的轮数控制训练；默认 1。未设置 max_train_steps 时生效。小数据集通常需更多 epoch，但更易过拟合。"),
                save_every_n_steps: Schema.number().min(1).description("每 N optimizer step 保存一次；只在需要细粒度中间 checkpoint 时填写，否则按 epoch 保存更简单。"),
                train_batch_size: Schema.number().min(1).default(1).description("每个优化 step 实际同时处理的样本数。显存允许时优先提高真实 batch；全参 2.9B 常从 1 开始。"),
                gradient_accumulation_steps: Schema.number().min(1).default(1).description("累积 N 个 mini-batch 后再更新一次参数；用于提高有效 batch 而不增加单步显存。会按比例降低 optimizer 更新频率。"),
                anima_checkpoint_mode: Schema.union(["off", "standard", "cpu", "unsloth"]).default("standard").description("Activation checkpointing。standard 是常用省显存方案；cpu/unsloth 进一步省 VRAM 但更依赖 RAM/传输，只有 OOM 时再尝试；off 最快但最吃显存。"),
            }).description("Anima 全参微调训练参数"),
            Schema.object({
                max_grad_norm: Schema.number().min(0).step(0.1).default(1.0).description("梯度裁剪阈值；1.0 是稳妥默认，0=关闭。训练出现偶发梯度尖峰/不稳定时可保留 1.0 或更低。"),
                max_data_loader_n_workers: Schema.number().min(0).step(1).default(8).description("数据加载 worker 数；8 为常见起点。CPU/RAM 紧张可降到 2~4，NVMe+多核 CPU 可适当提高；0 会禁用 persistent workers。"),
                fused_backward_pass: Schema.boolean().default(false).description("AdaFactor 专用的 fused backward/step 省显存路径。仅 AdaFactor + gradient_accumulation_steps=1 且不训练 Qwen3 时使用；其他情况保持关闭。"),
            }).description("训练执行高级参数（通常保持默认）").collapse(),
        ]),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("lora").required(),
            initial_epoch: Schema.number().min(0).step(1).description("只修改训练计数起点，不恢复 optimizer/scheduler。用于手工接着编号或迁移训练记录；真正断点续训优先用 resume state。"),
            initial_step: Schema.number().min(0).step(1).description("只设置 global step 起点，不恢复优化器状态。通常留空；需要与旧 run 的步数编号衔接时才用。"),
            skip_until_initial_step: Schema.boolean().default(false).description("同时跳过前 initial_step 对应的数据，使续跑尽量保持原数据顺序。只有明确要复现旧数据位置时开启。"),
            validation_split: Schema.number().min(0).max(1).step(0.01).description("从训练集划出验证集；0/留空=不验证。常见起点 0.05~0.1，小数据集慎用以免训练样本进一步减少。"),
            validation_seed: Schema.number().description("验证集划分随机种子；需要可复现划分时固定，例如 42/1337，否则可留空继承训练 seed。"),
            validate_every_n_steps: Schema.number().min(1).step(1).description("每 N optimizer step 验证。验证较耗时，长训练可用数百~数千 step；若已按 epoch 验证通常无需同时设置。"),
            validate_every_n_epochs: Schema.number().min(1).step(1).description("每 N epoch 验证；1 表示每轮。数据集很大或验证昂贵时可提高到 2~5。"),
            max_validation_steps: Schema.number().min(1).step(1).description("限制每次验证处理的 batch 数；留空=完整验证集。只想快速看趋势时可设 10~100。"),
            training_comment: Schema.string().description("写入 LoRA metadata 的备注，不影响训练结果。可记录数据版本、实验目的或关键参数。"),
            no_metadata: Schema.boolean().default(false).description("不保存训练 metadata；几乎总是保持关闭，除非明确需要最小化/清理输出元数据。"),
        }).description("LoRA 续训与验证（按需使用）").collapse(),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.intersect([
            Schema.object({ model_type: Schema.union(["flux", "chroma"]).required() }),
            SHARED_SCHEMAS.LR_OPTIMIZER,
        ]),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("lora").required(),
                learning_rate: Schema.string().default("5e-5").description("LoRA 主学习率。5e-5 是稳妥起点；角色/风格 LoRA 常在约 1e-5~1e-4 内调，数据少或过拟合快时向下调。"),
                lr_scheduler: Schema.union(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "cosine_with_min_lr", "warmup_stable_decay", "piecewise_constant", "custom"]).default("constant").description("学习率调度。短 LoRA 常用 constant；较长训练可尝试 cosine/linear。没有明确需求时保持 constant。"),
                loss_type: Schema.union(["l1", "l2", "huber", "smooth_l1"]).default("l2").description("损失函数。l2 为通用默认；Huber/smooth_l1 对异常样本更稳健，但会改变梯度形态，通常不需要改。"),
                weighting_scheme: Schema.union(["uniform", "sigma_sqrt", "cosmap", "logit_normal", "mode"]).default("uniform").description("不同噪声区间的采样/损失权重。uniform 为默认；sigma_sqrt/cosmap 重加权 loss；logit_normal/mode 仅配合 timestep_sampling=sigma。"),
                optimizer_type: Schema.union(["AdamW", "AdamW8bit", "PagedAdamW8bit", "RAdamScheduleFree", "Lion", "Lion8bit", "PagedLion8bit", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD", "AdaFactor", "Prodigy", "prodigyplus.ProdigyPlusScheduleFree", "pytorch_optimizer.CAME", "Custom"]).default("AdamW8bit").description("优化器。AdamW8bit 是默认且省显存；换 Prodigy/D-Adaptation/AdaFactor 等时需按对应算法重新评估 LR，不建议只换名字不改训练策略。"),
            }).description("Anima LoRA 学习率与优化器"),
            Schema.object({
                lr_warmup_steps: Schema.number().default(0).description("Warmup；0=关闭。长训练/较大学习率时可尝试总步数约 1%~5%，短 LoRA 多数保持 0。"),
                lr_decay_steps: Schema.number().min(0).description("Decay 段长度；仅特定 scheduler 使用。普通 constant/cosine 留空。"),
                lr_scheduler_args: Schema.array(String).role('table').description("额外 scheduler key=value；只在 custom/piecewise 等特殊调度器中需要。"),
                optimizer_args_custom: Schema.array(String).role('table').description("额外 optimizer 参数。通常留空；只有 AdaFactor/自定义 optimizer 等需要专用参数时填写。"),
            }).description("优化器与调度器高级参数（通常留空）").collapse(),
            Schema.union([
                Schema.object({
                    optimizer_type: Schema.const("Custom").required(),
                    anima_lora_custom_optimizer_type: Schema.string().description("完整 optimizer type/class"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    lr_scheduler: Schema.const("custom").required(),
                    anima_lora_custom_lr_scheduler_type: Schema.string().description("完整 scheduler class；不含点时从 torch.optim.lr_scheduler 查找"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    lr_scheduler: Schema.const("cosine_with_restarts").required(),
                    lr_scheduler_num_cycles: Schema.number().default(1).description("重启次数"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    weighting_scheme: Schema.const("logit_normal").required(),
                    logit_mean: Schema.string().default("0.0").description("logit-normal timestep 分布均值"),
                    logit_std: Schema.string().default("1.0").description("logit-normal timestep 分布标准差"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    weighting_scheme: Schema.const("mode").required(),
                    mode_scale: Schema.string().default("1.29").description("mode timestep 分布缩放"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    optimizer_type: Schema.const("Prodigy").required(),
                    prodigy_d0: Schema.string(),
                    prodigy_d_coef: Schema.string().default("2.0"),
                }),
                Schema.object({}),
            ]),
        ]),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("finetune").required(),
                anima_finetune_learning_rate: Schema.string().default("1e-5").description("DiT 主学习率。全参微调建议从 1e-5 量级起做短跑；数据很小或容易过拟合时可降到 1e-6~5e-6。分组件 LR 留空时都继承它。"),
                lr_scheduler: Schema.union(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "cosine_with_min_lr", "warmup_stable_decay", "piecewise_constant", "custom"]).default("constant").description("学习率调度策略。短跑/继续训练常用 constant；长训练常用 cosine/linear。没有明确调度需求时保持 constant。"),
                loss_type: Schema.union(["l1", "l2", "huber", "smooth_l1"]).default("l2").description("训练损失。l2 是默认且最常用；Huber/smooth_l1 对异常样本更稳健，但会改变梯度形态，建议只在明确需要抗异常值时使用。"),
                weighting_scheme: Schema.union(["uniform", "sigma_sqrt", "cosmap", "logit_normal", "mode"]).default("uniform").description("控制不同噪声区间的采样/损失权重。uniform 为稳妥默认；sigma_sqrt/cosmap 会重加权 loss；logit_normal/mode 只在 timestep_sampling=sigma 时改变采样分布。"),
                optimizer_type: Schema.union(["AdamW", "AdamW8bit", "PagedAdamW", "PagedAdamW8bit", "PagedAdamW32bit", "RAdamScheduleFree", "Lion", "Lion8bit", "PagedLion8bit", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptAdan", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD", "AdaFactor", "Prodigy", "prodigyplus.ProdigyPlusScheduleFree", "pytorch_optimizer.CAME", "Custom"]).default("AdamW8bit").description("优化器。AdamW8bit 是通用省显存默认；AdaFactor 更省 optimizer state，常用于大模型/显存紧张场景；自适应类优化器需按其规则重新评估 LR。"),
            }).description("Anima 全参微调学习率与优化器"),
            Schema.object({
                self_attn_lr: Schema.string().description("Self-Attention 独立 LR。通常留空继承主 LR；只在想让注意力层比其他模块学得更快/更慢时设置。0 会冻结该组件。"),
                cross_attn_lr: Schema.string().description("Cross-Attention 独立 LR。通常留空；做文本条件对齐/提示词响应专项调整时才常用。0=冻结。"),
                mlp_lr: Schema.string().description("MLP 独立 LR。通常留空；想单独控制容量/特征变换层更新强度时使用。0=冻结。"),
                mod_lr: Schema.string().description("AdaLN modulation 独立 LR。通常留空；只在明确研究调制层更新时设置。0=冻结。"),
                llm_adapter_lr: Schema.string().description("DiT 内嵌 LLM Adapter 独立 LR。通常留空继承主 LR；担心文本适配过快时可设为主 LR 的约 0.25~0.5 倍；0=冻结 Adapter。"),
            }).description("分组件学习率（高级；大多数训练留空）").collapse(),
            Schema.object({
                lr_warmup_steps: Schema.number().default(0).description("Warmup 长度；0=关闭。长训练或较大学习率时可用总步数约 1%~5% 作为起点；也可填小于 1 的比例值。"),
                lr_decay_steps: Schema.number().min(0).description("Decay 段长度，仅部分 scheduler 使用。没有使用 warmup_stable_decay 等调度时通常留空。"),
                lr_scheduler_args: Schema.array(String).role('table').description("传给 scheduler 的额外 key=value；仅自定义/特殊 scheduler 需要，普通 constant/cosine 通常留空。"),
                optimizer_args_custom: Schema.array(String).role('table').description("传给 optimizer 的额外 key=value。一般留空；只有使用 AdaFactor/自定义 optimizer 或按官方参数调优时填写。"),
            }).description("优化器与调度器高级参数（通常留空）").collapse(),
            Schema.union([
                Schema.object({
                    weighting_scheme: Schema.const("logit_normal").required(),
                    logit_mean: Schema.string().default("0.0").description("logit-normal timestep 分布均值"),
                    logit_std: Schema.string().default("1.0").description("logit-normal timestep 分布标准差"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    weighting_scheme: Schema.const("mode").required(),
                    mode_scale: Schema.string().default("1.29").description("mode timestep 分布缩放"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    optimizer_type: Schema.const("Custom").required(),
                    anima_custom_optimizer_type: Schema.string().description("完整 optimizer type/class，例如 bitsandbytes.optim.PagedAdEMAMix8bit"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    lr_scheduler: Schema.const("custom").required(),
                    anima_custom_lr_scheduler_type: Schema.string().description("完整 scheduler class；不含点时从 torch.optim.lr_scheduler 查找"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    lr_scheduler: Schema.const("cosine_with_restarts").required(),
                    lr_scheduler_num_cycles: Schema.number().default(1).description("重启次数"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    lr_scheduler: Schema.const("cosine_with_min_lr").required(),
                    lr_scheduler_num_cycles: Schema.number().default(1).description("Cosine 周期数"),
                    lr_scheduler_min_lr_ratio: Schema.number().min(0).max(1).step(0.01).default(0).description("最小 LR / 初始 LR 比例"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    lr_scheduler: Schema.const("inverse_sqrt").required(),
                    lr_scheduler_timescale: Schema.number().min(1).step(1).description("Inverse-sqrt timescale；留空时底层使用 warmup steps"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    lr_scheduler: Schema.const("warmup_stable_decay").required(),
                    lr_scheduler_num_cycles: Schema.number().default(1).description("Decay cosine cycles"),
                    lr_scheduler_min_lr_ratio: Schema.number().min(0).max(1).step(0.01).default(0).description("衰减后的最小 LR 比例"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    lr_scheduler: Schema.const("polynomial").required(),
                    lr_scheduler_power: Schema.number().step(0.1).default(1.0).description("Polynomial scheduler power"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    loss_type: Schema.const("huber").required(),
                    huber_schedule: Schema.union(["constant", "exponential", "snr"]).default("snr").description("Huber 参数调度方式"),
                    huber_c: Schema.number().step(0.01).default(0.1).description("Huber decay 参数"),
                    huber_scale: Schema.number().step(0.1).default(1.0).description("Huber scale 参数"),
                }),
                Schema.object({
                    loss_type: Schema.const("smooth_l1").required(),
                    huber_schedule: Schema.union(["constant", "exponential", "snr"]).default("snr").description("Huber 参数调度方式"),
                    huber_c: Schema.number().step(0.01).default(0.1).description("Huber decay 参数"),
                    huber_scale: Schema.number().step(0.1).default(1.0).description("Huber scale 参数"),
                }),
                Schema.object({}),
            ]),
            Schema.union([
                Schema.object({
                    optimizer_type: Schema.const("Prodigy").required(),
                    prodigy_d0: Schema.string(),
                    prodigy_d_coef: Schema.string().default("2.0"),
                }),
                Schema.object({}),
            ]),
        ]),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("finetune").required(),
            train_qwen3_text_encoder: Schema.boolean().default(false).description("训练 Qwen3-0.6B 文本编码器（本次训练全程）。关闭时保持 sd-scripts 原有冻结 Qwen3 的行为"),
        }).description("Anima 文本编码器微调"),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("finetune").required(),
            train_qwen3_text_encoder: Schema.const(true).required(),
            qwen3_lr: Schema.string().default("5e-7").description("Qwen3 文本编码器学习率；必须大于 0。建议从 5e-7 起，显著低于 DiT 学习率"),
            qwen3_gradient_checkpointing: Schema.boolean().default(true).description("Qwen3 梯度检查点：降低 activation 显存占用，但会增加重计算时间"),
            qwen3_output_dir: Schema.string().role('filepicker', { type: "folder" }).description("可选：Qwen3 sidecar checkpoint 输出目录；留空时与主 Anima checkpoint 保存在同一目录，文件名自动配对"),
        }).description("⚠ Qwen3 联合训练要求 Qwen3 输出缓存=Off；cache_latents 不受影响。第一版不支持 DeepSpeed、fused_backward_pass、D-Adaptation、Prodigy 或 Adafactor。每个主模型 checkpoint 会保存匹配的完整 Qwen3 sidecar。"),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.intersect([
            Schema.object({
                model_type: Schema.union(["flux", "chroma"]).required(),
                network_module: Schema.union(["networks.lora_flux", "networks.oft_flux", "lycoris.kohya"]).default("networks.lora_flux").description("训练网络模块"),
                network_weights: Schema.string().role('filepicker').description("从已有的 LoRA 模型上继续训练，填写路径"),
                network_dim: Schema.number().min(1).default(2).description("网络维度，常用 4~128"),
                network_alpha: Schema.number().min(1).default(16).description("LoRA alpha"),
                network_dropout: Schema.number().step(0.01).default(0).description("dropout 概率"),
                scale_weight_norms: Schema.number().step(0.01).min(0).description("最大范数正则化。如果使用，推荐为 1"),
                network_args_custom: Schema.array(String).role('table').description("自定义 network_args，一行一个"),
                enable_base_weight: Schema.boolean().default(false).description("启用基础权重（差异炼丹）"),
            }).description("网络设置"),
            SHARED_SCHEMAS.LYCORIS_MAIN,
            SHARED_SCHEMAS.LYCORIS_LOKR,
            SHARED_SCHEMAS.NETWORK_OPTION_BASEWEIGHT,
        ]),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("lora").required(),
                network_module: Schema.string().default("networks.lora_anima").disabled().description("固定使用 Anima 专用 LoRA 模块；通常无需关注。"),
                network_weights: Schema.string().role('filepicker').description("已有 LoRA 权重路径；用于在已有 LoRA 上继续训练。全新 LoRA 留空。"),
                network_dim: Schema.number().min(1).default(8).description("LoRA rank/容量。8 是官方/轻量起点；角色/简单风格常用 8~16，复杂风格或高容量需求可试 32/64，但参数量、显存和过拟合风险都会上升。"),
                network_alpha: Schema.number().min(1).default(1).description("LoRA 缩放系数。Anima 官方示例为 1；改变 alpha 会改变有效更新幅度，应与 LR 一起评估。没有经验时保持 1。"),
                network_dropout: Schema.number().min(0).max(1).step(0.01).default(0).description("LoRA neuron dropout；0=关闭。小数据集过拟合时可尝试 0.05~0.1，过高会削弱学习能力。"),
                anima_lora_train_llm_adapter: Schema.boolean().default(false).description("同时给 DiT 内嵌 LLM Adapter 建 LoRA。需要强化文本条件适配时开启；普通视觉风格/角色 LoRA 通常先关闭，避免增加可训练模块。"),
                network_train_unet_only: Schema.boolean().default(true).disabled().description("固定为仅训练 Anima DiT LoRA；字段名沿用 sd-scripts 历史 U-Net 命名"),
            }).description("Anima LoRA 网络设置"),
            Schema.object({
                anima_lora_rank_dropout: Schema.string().description("按 rank 随机丢弃 LoRA 维度；留空=关闭。可作为正则化尝试，常见 0.05~0.2；小数据集有过拟合迹象时才考虑。"),
                anima_lora_module_dropout: Schema.string().description("随机跳过整个 LoRA module；留空=关闭。正则化更强，通常比 rank dropout 更激进，建议从 0.05 左右小幅尝试。"),
                scale_weight_norms: Schema.number().step(0.01).min(0).description("限制 LoRA 权重范数；仅在权重增长过快/训练不稳定时使用，常见参考值 1。"),
                anima_lora_include_patterns: Schema.string().role('textarea').description("只对匹配 regex 的模块建立 LoRA，一行一个。用于精确限定训练层；普通训练留空，避免误漏模块。"),
                anima_lora_exclude_patterns: Schema.string().role('textarea').description("额外排除匹配 regex 的模块。sd-scripts 已默认排除 modulation/norm/embedder/final_layer；只有明确要进一步缩小范围时填写。"),
                anima_lora_network_reg_dims: Schema.string().description("按 regex 为不同模块指定不同 rank，例如 blocks\\.0=16,blocks\\.1=8。实验性容量分配，普通 LoRA 留空。"),
                anima_lora_network_reg_lrs: Schema.string().description("按 regex 为不同模块指定 LR，例如 blocks\\.0=1e-4,blocks\\.1=5e-5。只有做分层学习率实验时填写。"),
                anima_lora_loraplus_lr_ratio: Schema.string().description("LoRA+ 全局 B/A 学习率倍率；留空=关闭。需要 LoRA+ 时常从 4~16 倍做实验，并重新评估基础 LR。"),
                anima_lora_loraplus_unet_lr_ratio: Schema.string().description("仅 DiT/U-Net LoRA+ 倍率；通常留空，让全局 ratio 或普通 LR 生效。"),
                anima_lora_loraplus_text_encoder_lr_ratio: Schema.string().description("文本编码器 LoRA+ 倍率；仅训练 Qwen3 LoRA 时有意义，通常留空。"),
                anima_lora_network_verbose: Schema.boolean().default(false).description("打印模块匹配/创建详情，排查 include/exclude/regex 时开启；正常训练关闭以减少日志噪声。"),
                network_args_custom: Schema.array(String).role('table').description("专家级 network_args；同名 key 会覆盖上方 GUI 生成值。仅用于尚未显式支持的有效 lora_anima 参数。"),
                dim_from_weights: Schema.boolean().default(false).description("从已有 network_weights 自动读取 rank/alpha；继续训练结构未知的 LoRA 时有用，全新训练保持关闭。"),
                enable_base_weight: Schema.boolean().default(false).description("训练前先把一个或多个基础 LoRA 合并进底模。只用于差异炼丹/增量方案，普通 LoRA 训练保持关闭。"),
                base_weights: Schema.string().role('textarea').description("基础 LoRA 路径，一行一个；仅 enable_base_weight=true 时使用。"),
                base_weights_multiplier: Schema.string().role('textarea').description("基础 LoRA 合并倍率；可填一个值应用全部，或与路径数量一致。通常从 1.0 开始。"),
            }).description("LoRA 网络高级选项（大多数训练保持默认/留空）").collapse(),
        ]),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.intersect([
            Schema.object({ model_type: Schema.union(["flux", "chroma"]).required() }),
            SHARED_SCHEMAS.PREVIEW_IMAGE,
        ]),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                enable_preview: Schema.boolean().default(false).description("启用 Anima 训练预览图"),
            }).description("Anima 训练预览"),
            Schema.union([
                Schema.object({
                    enable_preview: Schema.const(true).required(),
                    randomly_choice_prompt: Schema.boolean().default(false).description("随机选择预览图 Prompt；dataset_config 模式建议改用 Prompt 文件"),
                    prompt_file: Schema.string().role('textarea').description("Prompt 文件路径；填写后下方 Prompt 文本设置失效"),
                    positive_prompts: Schema.string().role('textarea').default('masterpiece, best quality, 1girl, solo').description("Prompt"),
                    negative_prompts: Schema.string().role('textarea').default('lowres, bad anatomy, bad hands, text, error, worst quality, low quality').description("Negative Prompt"),
                    sample_width: Schema.number().min(16).step(16).default(1024).description("预览宽度"),
                    sample_height: Schema.number().min(16).step(16).default(1024).description("预览高度"),
                    sample_cfg: Schema.number().min(1).default(7).description("CFG Scale"),
                    sample_seed: Schema.number().default(2333).description("预览种子"),
                    sample_steps: Schema.number().min(1).max(1000).default(30).description("Rectified Flow Euler 步数"),
                    sample_flow_shift: Schema.number().step(0.1).default(3.0).description("仅用于预览采样的 Flow Shift；与训练 timestep 的 discrete_flow_shift 不同"),
                    anima_preview_cadence: Schema.union(["epoch", "step"]).default("epoch").description("预览频率按 epoch 或 optimizer step 计算；后端只会生成一种 sd-scripts cadence 参数"),
                    anima_preview_interval: Schema.number().min(1).step(1).default(1).description("每 N 个 epoch/step 生成一次预览"),
                    sample_at_first: Schema.boolean().default(false).description("训练开始前先生成一次预览"),
                }),
                Schema.object({}),
            ]),
        ]),
        Schema.object({}),
    ]),

    SHARED_SCHEMAS.LOG_SETTINGS,

    Schema.union([
        Schema.intersect([
            Schema.object({ model_type: Schema.union(["flux", "chroma"]).required() }),
            Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS, {}, ["max_token_length"])).description("caption（Tag）选项"),
        ]),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                caption_extension: Schema.string().default(".txt").description("Caption 文件扩展名；默认 .txt。只有数据集使用其他后缀（如 .caption）时修改。"),
                shuffle_caption: Schema.boolean().default(false).description("随机打乱逗号分隔的 tag 顺序，降低模型对固定顺序的依赖。只有 tag-list caption 才常用；缓存 Qwen3 输出时必须关闭。"),
                keep_tokens: Schema.number().min(0).max(255).step(1).default(0).description("shuffle 时固定保留最前 N 个 token/tag。常用于保留角色名/触发词；0=不保留。"),
            }).description("Anima caption（Tag）选项"),
            Schema.object({
                caption_separator: Schema.string().default(",").description("Tag 分隔符；Danbooru/Pixiv tag 数据通常保持逗号。自然语言 caption 通常无需修改。"),
                secondary_separator: Schema.string().description("第二分隔符；shuffle/dropout 前会解析并替换成主分隔符。只有数据源混用两种分隔符时使用。"),
                enable_wildcard: Schema.boolean().default(false).description("启用 {a|b|c} 随机 wildcard；用于 caption 随机变体。普通固定 caption 保持关闭。"),
                caption_prefix: Schema.string().description("给每条 caption 自动加固定前缀；例如统一质量词/风格提示。大多数数据集应在标注阶段处理，训练时通常留空。"),
                caption_suffix: Schema.string().description("给每条 caption 自动加固定后缀；用途同 prefix，普通训练通常留空。"),
                keep_tokens_separator: Schema.string().description("keep_tokens 专用分隔符；只有你的保留段与普通 tag 使用不同分隔规则时才需要。"),
                token_warmup_min: Schema.number().min(0).step(1).default(1).description("Token warmup 开始时至少使用的 tag 数；1 为默认。只在想让训练早期逐步增加 caption 信息时使用。"),
                token_warmup_step: Schema.number().min(0).step(0.01).default(0).description("Token warmup 完成位置；0=关闭，小于 1 按总步数比例解释。实验性策略，通常保持 0。"),
                caption_dropout_rate: Schema.number().min(0).max(1).step(0.01).description("随机丢弃整条 caption 的概率，用于降低对文本条件过拟合。小数据集可尝试 0.05~0.1；缓存 Qwen3 输出时需确认兼容。"),
                caption_dropout_every_n_epochs: Schema.number().min(0).max(100).step(1).description("每 N 个 epoch 整轮丢弃 caption；非常少用，普通训练留空/0。"),
                caption_tag_dropout_rate: Schema.number().min(0).max(1).step(0.01).description("按 tag 随机丢弃，适合 tag-list 正则化；常见可从 0.05~0.1 试起。缓存 Qwen3 输出时不能开启。"),
            }).description("Caption 高级增强（通常保持默认/留空）").collapse(),
        ]),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.intersect([
            Schema.object({ model_type: Schema.union(["flux", "chroma"]).required() }),
            SHARED_SCHEMAS.NOISE_SETTINGS,
        ]),
        Schema.object({ model_type: Schema.const("anima").required() }).description("Anima 使用 Rectified Flow；不显示旧 SD noise_offset / multires noise 选项"),
        Schema.object({}),
    ]),

    SHARED_SCHEMAS.DATA_ENCHANCEMENT,

    Schema.union([
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("finetune").required(),
                alpha_mask: Schema.boolean().default(false).description("使用图像 alpha 通道作为 loss mask；数据集图像必须实际包含 alpha"),
                face_crop_aug_range: Schema.string().description("可选：人脸中心裁剪范围，例如 2.0,4.0"),
                skip_cache_check: Schema.boolean().default(false).description("跳过已有 latent / text-encoder cache 内容有效性检查；仅建议在确认缓存与当前设置匹配时启用"),
                masked_loss: Schema.boolean().default(false).description("启用 conditioning mask loss；需要 conditioning_data_dir"),
            }).description("Anima 全参微调高级数据/Mask 设置"),
            Schema.union([
                Schema.object({
                    masked_loss: Schema.const(true).required(),
                    conditioning_data_dir: Schema.string().role('filepicker', { type: "folder" }).description("与训练图像对应的 conditioning / mask 数据目录"),
                }),
                Schema.object({}),
            ]),
        ]),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            metadata_title: Schema.string().description("写入模型文件的标题；不影响训练。发布模型时建议填写。"),
            metadata_author: Schema.string().description("作者/组织信息；不影响训练。"),
            metadata_description: Schema.string().role('textarea').description("模型说明；可记录用途、数据范围和推荐设置，不影响训练。"),
            metadata_license: Schema.string().description("许可证标识；发布时按实际授权填写。"),
            metadata_tags: Schema.string().description("模型标签；便于模型管理/分发平台索引。"),
            metadata_usage_hint: Schema.string().description("简短使用提示，例如推荐 prompt/strength；不影响训练。"),
            metadata_thumbnail: Schema.string().role('filepicker', { type: "file" }).description("嵌入 metadata 的缩略图；发布需要时再填。"),
            metadata_merged_from: Schema.string().description("记录来源/合并自哪些模型；只有确有来源关系时填写。"),
            metadata_trigger_phrase: Schema.string().description("触发词说明；角色/风格 LoRA 发布时有用，全参模型通常可留空。"),
            metadata_preprocessor: Schema.string().description("记录预处理器信息；一般留空，只有工作流需要追踪时填写。"),
            metadata_is_negative_embedding: Schema.boolean().default(false).description("仅负向 embedding 使用；Anima 模型/LoRA 几乎始终保持 false。"),
        }).description("模型 Metadata（发布时再填写）").collapse(),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            huggingface_repo_id: Schema.string().description("训练时自动上传的目标 repo，例如 user/model-name。平时本地训练留空；认证使用 hf auth/HF_TOKEN，GUI 不保存 token。"),
            huggingface_repo_type: Schema.union(["model", "dataset"]).default("model").description("输出 checkpoint 通常选 model；只有明确把训练产物当 dataset 仓库管理时才选 dataset。"),
            huggingface_path_in_repo: Schema.string().description("Repo 内子目录；默认根目录即可，多实验共用一个 repo 时再填写。"),
            huggingface_repo_visibility: Schema.union(["public", "private"]).default("private").description("创建 repo 时的可见性；实验阶段建议 private，确认可公开后再切 public。"),
            async_upload: Schema.boolean().default(false).description("后台上传 checkpoint，减少训练等待但可能增加网络/磁盘并发；网络稳定且 checkpoint 较大时可开启。"),
            save_state_to_huggingface: Schema.boolean().default(false).description("同时上传 optimizer/scheduler 等 resume state；只在需要跨机器精确续训时开启，文件体积会明显增加。"),
            resume_from_huggingface: Schema.boolean().default(false).description("从 HF 上的 state 精确续训；必须同时填写 resume。普通从本地 checkpoint 继续训练时保持关闭。"),
        }).description("Hugging Face 保存/恢复（不用云端训练时保持折叠）").collapse(),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.intersect([
            Schema.object({ model_type: Schema.union(["flux", "chroma"]).required() }),
            SHARED_SCHEMAS.OTHER,
        ]),
        Schema.object({
            model_type: Schema.const("anima").required(),
            seed: Schema.number().default(1337).description("训练随机种子；固定后有助于复现实验。通常保持一个固定值即可，只有做多 seed 对照实验时更换。"),
            ui_custom_params: Schema.string().role('textarea').description("专家级 TOML 覆盖，优先级高于 GUI。仅用于尚未做成控件的有效 sd-scripts 参数；写错可能改变/覆盖现有设置，普通训练不要填写。"),
        }).description("其他高级设置").collapse(),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.intersect([
            Schema.object({ model_type: Schema.union(["flux", "chroma"]).required() }),
            Schema.object(
                UpdateSchema(SHARED_SCHEMAS.RAW.PRECISION_CACHE_BATCH, {
                    fp8_base: Schema.boolean().default(true).description("对基础模型使用 FP8 精度"),
                    fp8_base_unet: Schema.boolean().description("仅对 U-Net 使用 FP8 精度（CLIP-L不使用）"),
                    sdpa: Schema.boolean().default(true).description("启用 sdpa"),
                    cache_text_encoder_outputs: Schema.boolean().default(true).description("缓存文本编码器输出；使用时需要关闭 shuffle_caption"),
                    cache_text_encoder_outputs_to_disk: Schema.boolean().default(true).description("缓存文本编码器输出到磁盘"),
                }, ["xformers"])
            ).description("Flux / Chroma 速度优化选项"),
        ]),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("lora").required(),
                mixed_precision: Schema.union(["no", "fp16", "bf16"]).default("bf16").description("训练计算精度。RTX 30/40/50 系通常优先 bf16；fp16 兼容更广但更容易数值溢出；no 主要用于调试。"),
                cache_latents: Schema.boolean().default(true).description("预先缓存 VAE latent，训练时不再重复跑 VAE，通常能明显省显存/提速；图像增强若会改变像素内容则可能不兼容。"),
                cache_latents_to_disk: Schema.boolean().default(true).description("把 latent cache 写到磁盘而不是长期占 RAM；大数据集通常推荐开启，代价是磁盘空间和首次缓存时间。"),
                cache_text_encoder_outputs: Schema.boolean().default(true).description("缓存 Qwen3 文本输出以省显存/加速；启用后 caption 不能在训练中动态 shuffle/dropout。需要训练/动态增强文本时关闭。"),
                cache_text_encoder_outputs_to_disk: Schema.boolean().default(true).description("将 Qwen3 输出缓存到磁盘；大数据集推荐，代价是额外磁盘空间。"),
                anima_lora_checkpoint_mode: Schema.union(["off", "standard", "cpu", "unsloth"]).default("standard").description("Activation checkpointing。standard 是通用省显存默认；cpu/unsloth 在 OOM 时进一步省 VRAM，但更慢且更吃 RAM/PCIe；off 最快但显存最高。"),
            }).description("Anima LoRA 精度、缓存与显存"),
            Schema.object({
                full_fp16: Schema.boolean().default(false).description("连模型权重/梯度也用 FP16；比普通 mixed fp16 更激进，可能更不稳定。除非明确测试过，否则保持关闭。"),
                full_bf16: Schema.boolean().default(false).description("连模型权重/梯度也用 BF16；一般 LoRA 不需要开启，除非有明确显存/吞吐需求并做过 smoke test。"),
                no_half_vae: Schema.boolean().default(false).description("强制 VAE 用 FP32；只有半精度 VAE 出现 NaN/色彩异常/不兼容时开启，显存占用会上升。"),
                lowram: Schema.boolean().default(false).description("低主机内存模式"),
                persistent_data_loader_workers: Schema.boolean().default(true).description("跨 epoch 保留 DataLoader worker，减少重启开销但增加 RAM 常驻；内存紧张或偶发 worker 问题时关闭。"),
                vae_batch_size: Schema.number().min(1).default(1).description("VAE 缓存阶段 batch size；1 最稳妥。显存充足时可逐步提高以加快首次缓存。"),
                text_encoder_batch_size: Schema.number().min(1).description("Qwen3 文本缓存 batch；留空跟随数据集 batch。只影响缓存阶段，显存充足时可提高以加速预缓存。"),
                skip_cache_check: Schema.boolean().default(false).description("跳过已有 cache 与当前配置是否匹配的检查。只有你完全确认分辨率/模型/tokenizer 等未变时才开，否则可能复用错误缓存。"),
                anima_lora_compile_mode: Schema.union(["off", "accelerate", "per_block"]).default("off").description("torch.compile 实验优化。off 最稳；accelerate 编译整体训练图；per_block 逐 block 编译。首次编译慢，收益依硬件/shape 而异，建议先短跑 benchmark。"),
                dynamo_backend: Schema.string().default("inductor").description("Accelerate compile backend；通常保持 inductor，只有调试/兼容问题时更换。"),
                compile_backend: Schema.string().default("inductor").description("Per-block compile backend；通常保持 inductor。"),
                compile_mode: Schema.union(["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]).default("default").description("Per-block compile 模式。default 最稳；reduce-overhead 偏低开销；max-autotune 编译更久，只有长训练才可能回本。"),
                compile_dynamic: Schema.union(["auto", "true", "false"]).default("auto").description("动态 shape 编译；auto 为默认。bucket 尺寸多时 dynamic 可能减少重编译，但 Windows dynamic=true 需要可用的 VS2022 C++ toolchain。"),
                compile_fullgraph: Schema.boolean().default(false).description("要求整块 graph 可编译；可能提高优化空间但更容易因 graph break 失败。通常关闭，且不能与 split_attn 同时启用。"),
                compile_cache_size_limit: Schema.number().min(1).step(1).description("Dynamo graph cache 上限；通常留空。只有 bucket/动态 shape 导致频繁 cache limit 警告时才提高。"),
                cuda_allow_tf32: Schema.boolean().default(true).description("Ampere+ 允许 TF32 矩阵计算，通常能提高速度且对训练质量影响很小；一般保持开启。"),
                cuda_cudnn_benchmark: Schema.boolean().default(false).description("让 cuDNN 为固定 shape 搜索更快 kernel；固定分辨率可能受益，多 bucket/频繁变 shape 时可能反而增加开销。"),
            }).description("LoRA 性能与编译高级选项（通常保持默认）").collapse(),
        ]),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("finetune").required(),
                anima_precision_mode: Schema.union(["fp32", "mixed_fp16", "full_fp16", "mixed_bf16", "full_bf16"]).default("full_bf16").description("训练精度模式。4090/新卡通常优先 BF16；full_bf16 更省显存但更依赖模型/优化器兼容，mixed_bf16 更保守；FP32 主要用于排查数值问题。"),
                anima_latent_cache_mode: Schema.union(["off", "memory", "disk"]).default("disk").description("VAE latent 缓存。Disk 是大多数全参训练的推荐默认；Memory 更快但吃 RAM；Off 允许像素级随机增强但每步都要跑 VAE。"),
                anima_text_encoder_cache_mode: Schema.union(["off", "memory", "disk"]).default("disk").description("Qwen3 输出缓存。冻结 Qwen3 时通常选 Disk；训练 Qwen3 时必须 Off，因为文本编码器需要参与反向传播。"),
            }).description("Anima 全参微调精度与缓存"),
            Schema.object({
                persistent_data_loader_workers: Schema.boolean().default(true).description("跨 epoch 保留 DataLoader worker；通常开启可减少停顿，RAM 紧张或 worker 不稳定时关闭。"),
                vae_batch_size: Schema.number().min(1).default(1).description("VAE 预缓存 batch；1 最稳。显存有余量时可提高到 2/4 加速首次缓存。"),
                text_encoder_batch_size: Schema.number().min(1).description("Qwen3 预缓存 batch；留空跟随数据集 batch。只影响缓存阶段，可在显存充足时提高。"),
                highvram: Schema.boolean().default(false).description("启用 sd-scripts High VRAM 模式，减少缓存阶段频繁清理 CUDA cache；仅在显存余量充足时建议开启"),
                torch_compile: Schema.boolean().default(false).description("启用 Accelerate torch.compile。可能提高长训练吞吐，但首次编译慢且兼容性依赖环境；默认关闭，开启后先做短 smoke test。"),
            }).description("全参性能高级选项（通常保持默认）").collapse(),
            Schema.union([
                Schema.object({
                    torch_compile: Schema.const(true).required(),
                    dynamo_backend: Schema.union(["eager", "aot_eager", "inductor", "aot_ts_nvfuser", "nvprims_nvfuser", "cudagraphs", "ofi", "fx2trt", "onnxrt", "tensort", "ipex", "tvm"]).default("inductor").description("Accelerate dynamo backend"),
                }),
                Schema.object({}),
            ]),
        ]),
        Schema.object({}),
    ]),

    Schema.union([
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("finetune").required(),
                deepspeed: Schema.boolean().default(false).description("启用 DeepSpeed；Qwen3 联合训练第一版不支持。启用后 sd-scripts 会将 DataLoader workers 固定为 1"),
            }).description("DeepSpeed（多卡/显存优化专家选项；单卡通常不用）").collapse(),
            Schema.union([
                Schema.object({
                    deepspeed: Schema.const(true).required(),
                    zero_stage: Schema.union([0, 1, 2, 3]).default(2).description("DeepSpeed ZeRO stage"),
                    offload_optimizer_device: Schema.union(["cpu", "nvme"]).description("Optimizer offload；仅 ZeRO-2/3 可用"),
                    offload_optimizer_nvme_path: Schema.string().description("Optimizer NVMe offload 路径；仅在 optimizer offload=nvme 时填写"),
                    offload_param_device: Schema.union(["cpu", "nvme"]).description("Parameter offload；仅 ZeRO-3 可用"),
                    offload_param_nvme_path: Schema.string().description("Parameter NVMe offload 路径；仅在 parameter offload=nvme 时填写"),
                    zero3_init_flag: Schema.boolean().default(false).description("ZeRO-3 zero.Init；仅 stage 3 可用"),
                    zero3_save_16bit_model: Schema.boolean().default(false).description("ZeRO-3 保存 16-bit model；仅 stage 3 可用"),
                    fp16_master_weights_and_gradients: Schema.boolean().default(false).description("仅 ZeRO-2 + optimizer CPU offload + FP16 模式有效"),
                }),
                Schema.object({}),
            ]),
        ]),
        Schema.object({}),
    ]),

    SHARED_SCHEMAS.DISTRIBUTED_TRAINING,

    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("finetune").required(),
            ddp_static_graph: Schema.boolean().default(false).description("DDP static_graph；只有模型图在各 iteration 稳定时才建议启用"),
        }).description("DDP 高级选项（多卡训练才需要）").collapse(),
        Schema.object({}),
    ])
]);