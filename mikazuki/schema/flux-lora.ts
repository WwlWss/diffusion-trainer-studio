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
        Schema.object({
            model_type: Schema.const("anima").required(),
            timestep_sampling: Schema.union(["sigma", "uniform", "sigmoid", "shift", "flux_shift"]).default("sigmoid").description("Anima Rectified Flow 时间步采样"),
            sigmoid_scale: Schema.number().step(0.001).default(1.0).description("sigmoid / shift / flux_shift 缩放"),
            discrete_flow_shift: Schema.number().step(0.001).default(1.0).description("Rectified Flow 离散流位移；主要用于 shift 采样"),
            qwen3_max_token_length: Schema.number().min(1).step(1).default(512).description("Qwen3 最大 token 长度"),
            t5_max_token_length: Schema.number().min(1).step(1).default(512).description("T5 tokenizer 最大 token 长度"),
            attn_mode: Schema.union(["torch", "xformers", "flash"]).default("torch").description("Attention 实现；选择 xformers 时后端会自动启用 split_attn"),
            split_attn: Schema.boolean().default(false).description("拆分 attention 计算以降低显存；xformers 模式会被自动开启"),
            vae_chunk_size: Schema.number().min(2).step(2).default(64).description("Qwen-Image VAE 空间分块大小；3D VAE 下越小越省显存；2D VAE 下影响较小"),
            vae_disable_cache: Schema.boolean().default(true).description("关闭 3D Qwen-Image VAE 内部缓存；启用 2D VAE 时此项无效果"),
            qwen_image_vae_2d: Schema.boolean().default(true).description("使用图像专用 2D Qwen-Image VAE；单图训练推荐，速度更快且显存更低"),
            blocks_to_swap: Schema.number().min(0).max(38).step(1).default(0).description("将 DiT block 交换到 CPU；0 为关闭。标准 Anima 最多 26，Anima 2.9B 最多 38；不能与 CPU/Unsloth checkpoint offload 同时使用"),
        }).description("Anima 专用参数"),
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
            anima_training_mode: Schema.const("finetune").required(),
            dataset_config: Schema.string().role('filepicker', { type: "file" }).description("可选：sd-scripts dataset TOML/JSON 配置；填写后由 trainer 使用该配置组织多 dataset/subset，并忽略普通 train_data_dir 参数"),
            in_json: Schema.string().role('filepicker', { type: "file" }).description("可选：metadata JSON；不使用 dataset_config 时可配合 train_data_dir 使用 metadata finetune 数据集"),
            skip_image_resolution: Schema.string().description("可选：跳过不大于指定分辨率的图片，例如 512 或 512,768"),
            resize_interpolation: Schema.union(["lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area"]).description("需要 resize 时使用的插值算法；留空使用 trainer 默认"),
        }).description("Anima 全参微调高级数据集来源"),
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
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("finetune").required(),
            output_name: Schema.string().default("aki").description("模型保存名称；Anima full finetune 固定保存为 safetensors"),
            output_dir: Schema.string().role('filepicker', { type: "folder" }).default("./output").description("模型保存文件夹"),
            save_precision: Schema.union(["fp16", "float", "bf16"]).default("bf16").description("checkpoint 保存精度；全参 BF16 训练建议保存 bf16"),
            save_every_n_epochs: Schema.number().min(1).default(1).description("每 N epoch 保存一次模型"),
            save_n_epoch_ratio: Schema.number().min(1).description("按总 epoch 比例自动计算保存间隔；例如 5 表示训练过程中至少保存约 5 个 epoch checkpoint"),
            save_last_n_epochs: Schema.number().min(1).description("最多保留最近 N 个 epoch checkpoint；留空不自动轮换"),
            save_last_n_steps: Schema.number().min(1).description("按 step 保存时仅保留最近 N step 范围内的 checkpoint"),
            save_state: Schema.boolean().default(false).description("保存 optimizer / scheduler 等训练状态，用于精确 resume"),
            save_state_on_train_end: Schema.boolean().default(false).description("训练结束时额外保存最后的训练状态"),
            save_last_n_epochs_state: Schema.number().min(1).description("最多保留最近 N 个 epoch state"),
            save_last_n_steps_state: Schema.number().min(1).description("按 step 保存时仅保留最近 N step 范围内的 state"),
        }).description("Anima 全参微调保存设置（固定 safetensors）"),
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
            gradient_checkpointing: Schema.boolean().default(true).description("梯度检查点"),
            unsloth_offload_checkpointing: Schema.boolean().default(false).description("异步将 checkpoint activation 卸载到 CPU；不能与 blocks_to_swap 同时使用"),
            gradient_accumulation_steps: Schema.number().min(1).default(1).description("梯度累加步数"),
        }).description("Anima LoRA 训练相关参数"),
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("finetune").required(),
            max_train_steps: Schema.number().min(1).description("可选：最大优化器步数；填写后优先按 step 控制训练"),
            max_train_epochs: Schema.number().min(1).default(1).description("最大 epoch；填写 max_train_steps 时后端优先 step"),
            save_every_n_steps: Schema.number().min(1).description("可选：每 N step 保存一次模型"),
            train_batch_size: Schema.number().min(1).default(1).description("批量大小；优先提高真实 batch，再考虑梯度累积"),
            gradient_accumulation_steps: Schema.number().min(1).default(1).description("梯度累加步数"),
            anima_checkpoint_mode: Schema.union(["off", "standard", "cpu", "unsloth"]).default("standard").description("Gradient Checkpointing：off=关闭；standard=普通；cpu=activation CPU offload；unsloth=异步 CPU offload。CPU/Unsloth 与 blocks_to_swap 不能同时使用"),
            max_grad_norm: Schema.number().min(0).step(0.1).default(1.0).description("梯度裁剪最大范数；0 表示关闭 clipping"),
            max_data_loader_n_workers: Schema.number().min(0).step(1).default(8).description("DataLoader 最大 worker 数；DeepSpeed 会由后端按 sd-scripts 语义固定为 1"),
            fused_backward_pass: Schema.boolean().default(false).description("融合 backward 与 optimizer step 以降低显存；当前 sd-scripts 仅支持 AdaFactor 且 gradient_accumulation_steps 必须为 1；Qwen3 联合训练不支持"),
        }).description("Anima 全参微调训练参数"),
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
                learning_rate: Schema.string().default("5e-5").description("Anima LoRA 的 DiT 学习率"),
                lr_scheduler: Schema.union(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "cosine_with_min_lr", "warmup_stable_decay", "piecewise_constant", "custom"]).default("constant").description("学习率调度器；Custom 会映射到 sd-scripts 的 lr_scheduler_type"),
                lr_warmup_steps: Schema.number().default(0).description("学习率预热步数"),
                lr_decay_steps: Schema.number().min(0).description("学习率衰减步数"),
                lr_scheduler_args: Schema.array(String).role('table').description("自定义 scheduler 参数，一行一个 key=value"),
                loss_type: Schema.union(["l1", "l2", "huber", "smooth_l1"]).default("l2").description("损失函数类型"),
                weighting_scheme: Schema.union(["uniform", "sigma_sqrt", "cosmap", "logit_normal", "mode"]).default("uniform").description("Loss/timestep weighting；logit_normal/mode 仅在 timestep_sampling=sigma 时改变真实采样分布"),
                optimizer_type: Schema.union(["AdamW", "AdamW8bit", "PagedAdamW8bit", "RAdamScheduleFree", "Lion", "Lion8bit", "PagedLion8bit", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD", "AdaFactor", "Prodigy", "prodigyplus.ProdigyPlusScheduleFree", "pytorch_optimizer.CAME", "Custom"]).default("AdamW8bit").description("优化器设置；Custom 可填写 sd-scripts 支持的完整 optimizer type/class"),
                optimizer_args_custom: Schema.array(String).role('table').description("自定义 optimizer_args，一行一个"),
            }).description("Anima LoRA 学习率与优化器"),
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
                anima_finetune_learning_rate: Schema.string().default("1e-5").description("Anima 全参 DiT 总学习率；必须大于 0。分组件学习率留空时继承此值"),
                self_attn_lr: Schema.string().description("Self-Attention 学习率；留空=总学习率，0=冻结该组件"),
                cross_attn_lr: Schema.string().description("Cross-Attention 学习率；留空=总学习率，0=冻结该组件"),
                mlp_lr: Schema.string().description("MLP 学习率；留空=总学习率，0=冻结该组件"),
                mod_lr: Schema.string().description("AdaLN modulation 学习率；留空=总学习率，0=冻结该组件"),
                llm_adapter_lr: Schema.string().description("内嵌 LLM Adapter 学习率；留空=总学习率，0=冻结 Adapter"),
                lr_scheduler: Schema.union(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "cosine_with_min_lr", "warmup_stable_decay", "piecewise_constant", "custom"]).default("constant").description("学习率调度器；Custom 会映射到 sd-scripts 的 lr_scheduler_type"),
                lr_warmup_steps: Schema.number().default(0).description("学习率预热步数；可填整数 step，底层 parser 也支持小于 1 的比例值"),
                lr_decay_steps: Schema.number().min(0).description("学习率衰减步数；warmup_stable_decay 等 scheduler 使用"),
                lr_scheduler_args: Schema.array(String).role('table').description("自定义 scheduler 参数，一行一个 key=value；piecewise/custom scheduler 可用"),
                loss_type: Schema.union(["l1", "l2", "huber", "smooth_l1"]).default("l2").description("损失函数类型"),
                weighting_scheme: Schema.union(["uniform", "sigma_sqrt", "cosmap", "logit_normal", "mode"]).default("uniform").description("Loss/timestep weighting；logit_normal/mode 仅在 timestep_sampling=sigma 时改变真实采样分布"),
                optimizer_type: Schema.union(["AdamW", "AdamW8bit", "PagedAdamW", "PagedAdamW8bit", "PagedAdamW32bit", "RAdamScheduleFree", "Lion", "Lion8bit", "PagedLion8bit", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptAdan", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD", "AdaFactor", "Prodigy", "prodigyplus.ProdigyPlusScheduleFree", "pytorch_optimizer.CAME", "Custom"]).default("AdamW8bit").description("优化器设置；Custom 可填写 sd-scripts 支持的完整 optimizer class；训练 Qwen3 时仍受独立参数组 LR 白名单限制"),
                optimizer_args_custom: Schema.array(String).role('table').description("自定义 optimizer_args，一行一个"),
            }).description("Anima 全参微调学习率与优化器"),
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
                    loss_type: Schema.union(["huber", "smooth_l1"]).required(),
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
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("lora").required(),
            network_module: Schema.string().default("networks.lora_anima").disabled().description("Anima LoRA 网络模块"),
            network_weights: Schema.string().role('filepicker').description("从已有的 Anima LoRA 上继续训练"),
            network_dim: Schema.number().min(1).default(8).description("LoRA rank；官方示例为 8，风格 LoRA 可按容量需要提高到 16/32"),
            network_alpha: Schema.number().min(1).default(1).description("LoRA alpha；官方示例为 1。提高 alpha 时应重新评估学习率"),
            network_dropout: Schema.number().min(0).max(1).step(0.01).default(0).description("LoRA dropout"),
            scale_weight_norms: Schema.number().step(0.01).min(0).description("最大范数正则化"),
            network_args_custom: Schema.array(String).role('table').description("Anima network_args；可填写 train_llm_adapter=True、network_reg_dims=...、network_reg_lrs=... 等，一行一个"),
            network_train_unet_only: Schema.boolean().default(true).disabled().description("固定为仅训练 Anima DiT LoRA；字段名沿用 sd-scripts 历史 U-Net 命名"),
        }).description("Anima LoRA 网络设置"),
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
        Schema.object({
            model_type: Schema.const("anima").required(),
            caption_extension: Schema.string().default(".txt").description("Tag 文件扩展名"),
            caption_separator: Schema.string().default(",").description("caption tag 分隔符"),
            secondary_separator: Schema.string().description("可选第二分隔符；drop/shuffle 后会替换成主 caption_separator"),
            enable_wildcard: Schema.boolean().default(false).description("启用 {a|b|c} wildcard caption"),
            caption_prefix: Schema.string().description("可选：在每条 caption 前添加固定前缀"),
            caption_suffix: Schema.string().description("可选：在每条 caption 后添加固定后缀"),
            shuffle_caption: Schema.boolean().default(false).description("训练时随机打乱 tokens；缓存 Qwen3 输出时必须关闭"),
            keep_tokens: Schema.number().min(0).max(255).step(1).default(0).description("在随机打乱 tokens 时，保留前 N 个不变"),
            keep_tokens_separator: Schema.string().description("保留 tokens 时使用的分隔符"),
            token_warmup_min: Schema.number().min(0).step(1).default(1).description("token warmup 起始 tag 数"),
            token_warmup_step: Schema.number().min(0).step(0.01).default(0).description("token warmup 完成 step；小于 1 时按 max_train_steps 比例解释"),
            caption_dropout_rate: Schema.number().min(0).max(1).step(0.01).description("丢弃全部标签的概率；Anima 缓存文本输出仍支持此项"),
            caption_dropout_every_n_epochs: Schema.number().min(0).max(100).step(1).description("每 N 个 epoch 丢弃全部标签"),
            caption_tag_dropout_rate: Schema.number().min(0).max(1).step(0.01).description("按 tag 随机丢弃；缓存 Qwen3 输出时不能启用"),
        }).description("Anima caption（Tag）选项"),
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
        Schema.intersect([
            Schema.object({ model_type: Schema.union(["flux", "chroma"]).required() }),
            SHARED_SCHEMAS.OTHER,
        ]),
        Schema.object({
            model_type: Schema.const("anima").required(),
            seed: Schema.number().default(1337).description("随机种子"),
            ui_custom_params: Schema.string().role('textarea').description("危险：自定义 TOML 参数会覆盖界面参数；Anima 后端会拒绝与语义模式冲突或 full trainer 尚未实现的 no-op 参数"),
        }).description("Anima 其他设置"),
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
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("lora").required(),
            mixed_precision: Schema.union(["no", "fp16", "bf16"]).default("bf16").description("训练混合精度"),
            full_fp16: Schema.boolean().default(false).description("完全使用 FP16"),
            full_bf16: Schema.boolean().default(false).description("完全使用 BF16"),
            no_half_vae: Schema.boolean().default(false).description("VAE 不使用半精度"),
            lowram: Schema.boolean().default(false).description("低主机内存模式"),
            cache_latents: Schema.boolean().default(true).description("缓存 Qwen-Image VAE latent"),
            cache_latents_to_disk: Schema.boolean().default(true).description("将 latent 缓存到磁盘"),
            cache_text_encoder_outputs: Schema.boolean().default(true).description("缓存 Qwen3 输出"),
            cache_text_encoder_outputs_to_disk: Schema.boolean().default(true).description("将 Qwen3 输出缓存到磁盘"),
            persistent_data_loader_workers: Schema.boolean().default(true).description("保留 DataLoader workers"),
            vae_batch_size: Schema.number().min(1).default(1).description("VAE 编码批量大小"),
            cuda_allow_tf32: Schema.boolean().default(true).description("允许 Ampere 及更新 GPU 使用 TF32；Anima LoRA trainer 会实际应用该设置"),
            cuda_cudnn_benchmark: Schema.boolean().default(false).description("启用 cuDNN benchmark；输入形状变化很大时未必更快"),
        }).description("Anima LoRA 速度与缓存选项"),
        Schema.intersect([
            Schema.object({
                model_type: Schema.const("anima").required(),
                anima_training_mode: Schema.const("finetune").required(),
                anima_precision_mode: Schema.union(["fp32", "mixed_fp16", "full_fp16", "mixed_bf16", "full_bf16"]).default("full_bf16").description("训练精度；单一选项映射到 mixed_precision/full_fp16/full_bf16，避免非法组合"),
                anima_latent_cache_mode: Schema.union(["off", "memory", "disk"]).default("disk").description("Latent cache：Off / RAM / Disk。Disk 会显式启用 cache_latents"),
                anima_text_encoder_cache_mode: Schema.union(["off", "memory", "disk"]).default("disk").description("Qwen3 输出 cache：Off / RAM / Disk。训练 Qwen3 时必须选择 Off"),
                persistent_data_loader_workers: Schema.boolean().default(true).description("保留 DataLoader workers，减少 epoch 间停顿但增加主机内存占用"),
                vae_batch_size: Schema.number().min(1).default(1).description("VAE latent 缓存批量大小"),
                text_encoder_batch_size: Schema.number().min(1).description("Qwen3 输出缓存批量大小；留空使用数据集 batch size"),
                highvram: Schema.boolean().default(false).description("启用 sd-scripts High VRAM 模式，减少缓存阶段频繁清理 CUDA cache；仅在显存余量充足时建议开启"),
                torch_compile: Schema.boolean().default(false).description("启用 Accelerate/torch.compile；与 Anima per-block compile 是不同机制。当前与 blocks_to_swap 的组合会被后端拒绝，直到完成 GPU 验证"),
            }).description("Anima 全参微调精度、缓存与性能"),
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
            }).description("Anima 全参微调 DeepSpeed"),
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
        }).description("Anima 全参微调 DDP 高级选项"),
        Schema.object({}),
    ])
]);