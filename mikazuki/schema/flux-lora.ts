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
            // the form model. Requiring an explicit value makes mode-dependent
            // Qwen3 controls deterministic. Included presets already set this field.
            anima_training_mode: Schema.union(["lora", "finetune"]).description("Anima 训练方式：请选择 LoRA 或全参微调；显式选择可确保旧版 GUI 正确切换条件设置"),
            qwen3: Schema.string().role('filepicker', { type: "model-file" }).description("Anima 文本编码器：Qwen3-0.6B safetensors 或本地 HuggingFace 模型目录；启动训练时后端会校验必填"),
            vae: Schema.string().role('filepicker', { type: "model-file" }).description("Anima VAE：Qwen-Image VAE safetensors / pth；启动训练时后端会校验必填"),
            llm_adapter_path: Schema.string().role('filepicker', { type: "model-file" }).description("可选：独立 LLM Adapter 权重；留空时从 DiT 中读取"),
            t5_tokenizer_path: Schema.string().role('filepicker', { type: "folder" }).description("可选：T5 tokenizer 目录；留空使用 sd-scripts 内置配置"),
        }),
        Schema.object({}),
    ]),

    // Legacy frontend caches disabled default fields when a union branch changes.
    // Therefore model_train_type is exposed only for Flux/Chroma. For Anima the
    // backend derives anima-lora/anima-finetune exclusively from anima_training_mode.
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
            unsloth_offload_checkpointing: Schema.boolean().default(false).description("将 checkpoint activation 异步卸载到 CPU；不能和 blocks_to_swap / cpu_offload_checkpointing 同时使用"),
            cuda_allow_tf32: Schema.boolean().default(true).description("Ampere 及更新显卡允许 TF32；旧显卡保持开启也不会获得 TF32 加速"),
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

    SHARED_SCHEMAS.SAVE_SETTINGS,

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
            max_train_steps: Schema.number().min(1).description("可选：最大优化器步数；填写后优先按 step 控制训练"),
            max_train_epochs: Schema.number().min(1).default(1).description("最大 epoch；默认按 Epoch 控制训练；填写 max_train_steps 时后端优先 step"),
            save_every_n_steps: Schema.number().min(1).description("可选：每 N step 保存一次模型；仅在需要按 step 保存时填写"),
            train_batch_size: Schema.number().min(1).default(1).description("批量大小；优先在显存允许时提高真实 batch，再考虑梯度累积"),
            gradient_checkpointing: Schema.boolean().default(true).description("梯度检查点；显著降低显存占用"),
            gradient_accumulation_steps: Schema.number().min(1).default(1).description("梯度累加步数；显存不足以提高真实 batch 时使用"),
        }).description("Anima 训练相关参数"),
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
                learning_rate: Schema.string().default("5e-5").description("Anima 总学习率。LoRA 模式下用于 DiT LoRA；不再使用旧 U-Net / 文本编码器学习率"),
                lr_scheduler: Schema.union(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]).default("constant").description("学习率调度器；风格 LoRA 首轮测试推荐 constant"),
                lr_warmup_steps: Schema.number().default(0).description("学习率预热步数"),
                loss_type: Schema.union(["l1", "l2", "huber", "smooth_l1"]).default("l2").description("损失函数类型"),
                optimizer_type: Schema.union(["AdamW", "AdamW8bit", "PagedAdamW8bit", "RAdamScheduleFree", "Lion", "Lion8bit", "PagedLion8bit", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD", "AdaFactor", "Prodigy", "prodigyplus.ProdigyPlusScheduleFree", "pytorch_optimizer.CAME"]).default("AdamW8bit").description("优化器设置"),
                optimizer_args_custom: Schema.array(String).role('table').description("自定义 optimizer_args，一行一个"),
            }).description("Anima 学习率与优化器设置"),
            Schema.union([
                Schema.object({
                    lr_scheduler: Schema.const("cosine_with_restarts").required(),
                    lr_scheduler_num_cycles: Schema.number().default(1).description("重启次数"),
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

    // Keep the existing finetune component-LR controls attached to Anima itself.
    // The backend removes these fields automatically for LoRA; changing this older
    // block would broaden the regression surface of the Qwen3 feature.
    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            self_attn_lr: Schema.string().description("Self-Attention 学习率；仅 finetune 使用；留空=总学习率，0=冻结该组件"),
            cross_attn_lr: Schema.string().description("Cross-Attention 学习率；仅 finetune 使用；留空=总学习率，0=冻结该组件"),
            mlp_lr: Schema.string().description("MLP 学习率；仅 finetune 使用；留空=总学习率，0=冻结该组件"),
            mod_lr: Schema.string().description("AdaLN modulation 学习率；仅 finetune 使用；留空=总学习率，0=冻结该组件"),
            llm_adapter_lr: Schema.string().description("LLM Adapter 学习率；仅 finetune 使用；留空=总学习率，0=冻结 Adapter"),
            cpu_offload_checkpointing: Schema.boolean().description("仅 finetune 使用；将 gradient-checkpoint activation 卸载到 CPU；不能与 blocks_to_swap / unsloth_offload_checkpointing 同时使用"),
        }).description("Anima 全参微调分组件学习率（LoRA 模式下后端自动忽略）"),
        Schema.object({}),
    ]),

    // Qwen3 joint finetuning is the only new mode-dependent block. It relies on
    // anima_training_mode being an explicit form value (not an implicit default).
    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("finetune").required(),
            train_qwen3_text_encoder: Schema.boolean().default(false).description("训练 Qwen3-0.6B 文本编码器（本次训练全程）。关闭时保持 sd-scripts 原有冻结 Qwen3 的行为"),
        }).description("Anima 文本编码器微调"),
        Schema.object({}),
    ]),

    // The detailed controls are expanded only after the user explicitly enables
    // Qwen3 training. The backend independently validates every incompatibility.
    Schema.union([
        Schema.object({
            model_type: Schema.const("anima").required(),
            anima_training_mode: Schema.const("finetune").required(),
            train_qwen3_text_encoder: Schema.const(true).required(),
            qwen3_lr: Schema.string().default("5e-7").description("Qwen3 文本编码器学习率；必须大于 0。建议从 5e-7 起，显著低于 DiT 学习率"),
            qwen3_gradient_checkpointing: Schema.boolean().default(true).description("Qwen3 梯度检查点：降低 activation 显存占用，但会增加重计算时间"),
            qwen3_output_dir: Schema.string().role('filepicker', { type: "folder" }).description("可选：Qwen3 sidecar checkpoint 输出目录；留空时与主 Anima checkpoint 保存在同一目录，文件名自动配对"),
        }).description("⚠ Qwen3 联合训练与 cache_text_encoder_outputs / cache_text_encoder_outputs_to_disk 不兼容；请关闭两项 Qwen3 输出缓存。cache_latents 不受影响。第一版也不支持 DeepSpeed、fused_backward_pass、D-Adaptation、Prodigy 或 Adafactor。每个主模型 checkpoint 会保存匹配的完整 Qwen3 sidecar，因此会增加磁盘占用和保存停顿。"),
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
        // Do not gate these fields on anima_training_mode. The legacy frontend may
        // resolve this segment before the mode default is materialized and would then
        // permanently choose the empty branch. Finetune strips network_* in backend.
        Schema.object({
            model_type: Schema.const("anima").required(),
            network_module: Schema.string().default("networks.lora_anima").disabled().description("Anima LoRA 网络模块；finetune 模式下后端自动忽略"),
            network_weights: Schema.string().role('filepicker').description("从已有的 Anima LoRA 上继续训练；仅 LoRA 使用"),
            network_dim: Schema.number().min(1).default(8).description("LoRA rank；仅 LoRA 使用；官方示例为 8，风格 LoRA 可按容量需要提高到 16/32"),
            network_alpha: Schema.number().min(1).default(1).description("LoRA alpha；仅 LoRA 使用；官方示例为 1。提高 alpha 时应重新评估学习率"),
            network_dropout: Schema.number().min(0).max(1).step(0.01).default(0).description("LoRA dropout；仅 LoRA 使用"),
            scale_weight_norms: Schema.number().step(0.01).min(0).description("最大范数正则化；仅 LoRA 使用"),
            network_args_custom: Schema.array(String).role('table').description("Anima network_args；仅 LoRA 使用；可填写 train_llm_adapter=True、network_reg_dims=...、network_reg_lrs=... 等，一行一个"),
            network_train_unet_only: Schema.boolean().default(true).disabled().description("固定为仅训练 Anima DiT LoRA；字段名沿用 sd-scripts 历史 U-Net 命名；finetune 时后端自动忽略"),
        }).description("Anima LoRA 网络设置（finetune 模式下后端自动忽略）"),
        Schema.object({}),
    ]),

    SHARED_SCHEMAS.PREVIEW_IMAGE,
    SHARED_SCHEMAS.LOG_SETTINGS,

    Schema.union([
        Schema.intersect([
            Schema.object({ model_type: Schema.union(["flux", "chroma"]).required() }),
            Schema.object(UpdateSchema(SHARED_SCHEMAS.RAW.CAPTION_SETTINGS, {}, ["max_token_length"])).description("caption（Tag）选项"),
        ]),
        Schema.object({
            model_type: Schema.const("anima").required(),
            caption_extension: Schema.string().default(".txt").description("Tag 文件扩展名"),
            shuffle_caption: Schema.boolean().default(false).description("训练时随机打乱 tokens；缓存 Qwen3 输出时必须关闭"),
            keep_tokens: Schema.number().min(0).max(255).step(1).default(0).description("在随机打乱 tokens 时，保留前 N 个不变"),
            keep_tokens_separator: Schema.string().description("保留 tokens 时使用的分隔符"),
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
            Schema.object({ model_type: Schema.union(["flux", "chroma"]).required() }),
            SHARED_SCHEMAS.OTHER,
        ]),
        Schema.object({
            model_type: Schema.const("anima").required(),
            seed: Schema.number().default(1337).description("随机种子"),
            ui_custom_params: Schema.string().role('textarea').description("危险：自定义 TOML 参数会覆盖界面参数。仅用于 sd-scripts 已支持但 GUI 尚未暴露的参数"),
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
            mixed_precision: Schema.union(["no", "fp16", "bf16"]).default("bf16").description("训练混合精度；Ampere 及更新显卡推荐 bf16"),
            full_fp16: Schema.boolean().default(false).description("完全使用 FP16；通常不需要"),
            full_bf16: Schema.boolean().default(false).description("完全使用 BF16；仅在明确需要时开启"),
            no_half_vae: Schema.boolean().default(false).description("VAE 不使用半精度"),
            lowram: Schema.boolean().default(false).description("低主机内存模式；会改变模型加载/迁移策略，不等同于省显存模式"),
            cache_latents: Schema.boolean().default(true).description("缓存 Qwen-Image VAE latent"),
            cache_latents_to_disk: Schema.boolean().default(true).description("将 latent 缓存到磁盘"),
            cache_text_encoder_outputs: Schema.boolean().default(true).description("缓存 Qwen3 输出以释放显存；启用后必须关闭 shuffle_caption 和 caption_tag_dropout_rate"),
            cache_text_encoder_outputs_to_disk: Schema.boolean().default(true).description("将 Qwen3 输出缓存到磁盘"),
            persistent_data_loader_workers: Schema.boolean().default(true).description("保留加载训练集的 worker，减少 epoch 之间的停顿"),
            vae_batch_size: Schema.number().min(1).default(1).description("VAE 编码批量大小"),
        }).description("Anima 速度与缓存选项"),
        Schema.object({}),
    ]),

    SHARED_SCHEMAS.DISTRIBUTED_TRAINING
]);