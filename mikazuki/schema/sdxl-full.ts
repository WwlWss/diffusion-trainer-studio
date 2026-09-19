Schema.intersect([
    Schema.object({
        model_train_type: Schema.string().default("sdxl-finetune").disabled().description("固定训练后端：scripts/stable/sdxl_train.py"),
        pretrained_model_name_or_path: Schema.string().role("filepicker", { type: "model-file" }).default("./sd-models/model.safetensors").description("SDXL 底模路径"),
        vae: Schema.string().role("filepicker", { type: "model-file" }).description("可选：外置 SDXL VAE"),
        resume: Schema.string().role("filepicker", { type: "folder" }).description("从 save_state 恢复训练"),
    }).description("SDXL 全参微调模型"),

    Schema.object({
        train_data_dir: Schema.string().role("filepicker", { type: "folder" }).default("./train/aki").description("普通目录数据集；dataset_config 存在时由其覆盖"),
        reg_data_dir: Schema.string().role("filepicker", { type: "folder" }).description("可选正则化数据集"),
        dataset_config: Schema.string().role("filepicker", { type: "file" }).description("可选：kohya dataset config TOML/JSON；设置后忽略 train_data_dir 与 in_json"),
        in_json: Schema.string().role("filepicker", { type: "file" }).description("可选：fine-tuning metadata JSON；仍需 train_data_dir；dataset_config 存在时会被忽略"),
        prior_loss_weight: Schema.number().step(0.1).default(1.0).description("正则化 prior loss 权重"),
        resolution: Schema.string().default("1024,1024").description("训练分辨率"),
        enable_bucket: Schema.boolean().default(true).description("启用 aspect-ratio bucket"),
        min_bucket_reso: Schema.number().default(256).description("最小 bucket 分辨率"),
        max_bucket_reso: Schema.number().default(2048).description("最大 bucket 分辨率"),
        bucket_reso_steps: Schema.number().default(32).description("SDXL bucket 步长；trainer 要求可被 32 整除"),
        bucket_no_upscale: Schema.boolean().default(true).description("bucket 不放大图片"),
        flip_aug: Schema.boolean().default(false).description("水平翻转；latent cache 可用"),
        color_aug: Schema.boolean().default(false).description("颜色增强；与 latent cache 冲突"),
        random_crop: Schema.boolean().default(false).description("随机裁剪；与 latent cache 冲突"),
    }).description("数据集设置"),

    Schema.object({
        max_token_length: Schema.number().min(75).step(75).default(225).description("SDXL tokenizer 最大 token 长度；Standard 与 Multi 共用"),
    }).description("Caption 全局编码"),

    SHARED_SCHEMAS.CAPTION_MODE_SHARED(
        Schema.object({
            caption_extension: Schema.string().default(".txt").description("caption 扩展名"),
            shuffle_caption: Schema.boolean().default(false).description("随机打乱 caption token；TE output cache 时必须关闭"),
            keep_tokens: Schema.number().min(0).step(1).default(0).description("shuffle 时固定前 N token"),
            keep_tokens_separator: Schema.string().description("keep_tokens 分隔符"),
            caption_dropout_rate: Schema.number().min(0).max(1).step(0.01).description("整条 caption dropout；TE output cache 时必须关闭"),
            caption_dropout_every_n_epochs: Schema.number().min(0).step(1).description("每 N epoch 丢弃 caption"),
            caption_tag_dropout_rate: Schema.number().min(0).max(1).step(0.01).description("tag dropout；TE output cache 时必须关闭"),
        }).description("Standard Caption")
    ),

    Schema.object({
        output_name: Schema.string().default("sdxl-finetune").description("输出模型名"),
        output_dir: Schema.string().role("filepicker", { type: "folder" }).default("./output").description("输出目录"),
        save_model_as: Schema.union(["safetensors", "ckpt", "diffusers", "diffusers_safetensors"]).default("safetensors").description("SDXL full trainer 支持的输出格式"),
        save_precision: Schema.union(["fp16", "float", "bf16"]).default("bf16").description("保存精度"),
        save_every_n_epochs: Schema.number().min(1).default(1).description("每 N epoch 保存"),
        save_every_n_steps: Schema.number().min(1).description("可选：每 N step 保存"),
        save_n_epoch_ratio: Schema.number().min(1).description("整个训练划分为 N 个 epoch 保存区间；设置后覆盖 save_every_n_epochs"),
        save_last_n_epochs: Schema.number().min(1).description("保留最近 N epoch checkpoint"),
        save_last_n_steps: Schema.number().min(1).description("保留最近 N step checkpoint"),
        save_state: Schema.boolean().default(false).description("保存 optimizer/scheduler state"),
        save_state_on_train_end: Schema.boolean().default(false).description("训练结束保存 state"),
        save_last_n_epochs_state: Schema.number().min(1).description("保留最近 N epoch state"),
        save_last_n_steps_state: Schema.number().min(1).description("保留最近 N step state"),
    }).description("保存设置"),

    Schema.object({
        max_train_steps: Schema.number().min(1).description("最大 optimizer steps；填写后后端会移除默认 max_train_epochs"),
        max_train_epochs: Schema.number().min(1).default(1).description("最大 epoch；未填写 max_train_steps 时使用"),
        train_batch_size: Schema.number().min(1).default(1).description("batch size"),
        gradient_accumulation_steps: Schema.number().min(1).default(1).description("梯度累积；fused 模式要求 1"),
        gradient_checkpointing: Schema.boolean().default(true).description("梯度检查点"),
        max_grad_norm: Schema.number().min(0).step(0.1).default(1.0).description("梯度裁剪；0 关闭"),
        max_data_loader_n_workers: Schema.number().min(0).step(1).default(8).description("DataLoader workers"),
        persistent_data_loader_workers: Schema.boolean().default(true).description("跨 epoch 保留 workers"),
    }).description("训练控制"),

    Schema.intersect([
        Schema.object({
            learning_rate: Schema.string().default("1e-6").description("SDXL U-Net 总学习率；0 可冻结 U-Net"),
            train_text_encoder: Schema.boolean().default(false).description("训练两个 SDXL 文本编码器；启用后必须保持 TE output cache 关闭"),
            block_lr: Schema.string().description("可选：必须恰好 23 个逗号分隔 U-Net block 学习率；替代统一 U-Net LR"),
            optimizer_type: Schema.union(["AdamW", "AdamW8bit", "PagedAdamW8bit", "RAdamScheduleFree", "Lion", "Lion8bit", "PagedLion8bit", "SGDNesterov", "SGDNesterov8bit", "AdaFactor", "Prodigy", "DAdaptation", "DAdaptAdam", "DAdaptAdaGrad", "DAdaptAdanIP", "DAdaptLion", "DAdaptSGD"]).default("AdamW8bit").description("优化器"),
            optimizer_args_custom: Schema.array(String).role("table").description("自定义 optimizer_args，一行一个"),
            lr_scheduler: Schema.union(["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]).default("constant").description("LR scheduler"),
            lr_warmup_steps: Schema.number().min(0).default(0).description("warmup steps"),
            lr_decay_steps: Schema.number().min(0).description("decay steps"),
            lr_scheduler_num_cycles: Schema.number().min(1).default(1).description("cosine restart cycles"),
            lr_scheduler_power: Schema.number().step(0.1).default(1.0).description("polynomial power"),
            fused_backward_pass: Schema.boolean().default(false).description("AdaFactor fused backward；要求 accumulation=1，不能与 DeepSpeed/fused_optimizer_groups 同时使用"),
            fused_optimizer_groups: Schema.number().min(1).step(1).description("多个 optimizer group 的 backward-hook step；要求 accumulation=1，不能与 DeepSpeed/fused_backward_pass 同时使用"),
        }),
        Schema.union([
            Schema.object({
                train_text_encoder: Schema.const(true).required(),
                learning_rate_te1: Schema.string().default("5e-7").description("Text Encoder 1 (ViT-L) LR；0 冻结"),
                learning_rate_te2: Schema.string().default("5e-7").description("Text Encoder 2 (BiG-G) LR；0 冻结"),
            }),
            Schema.object({}),
        ]),
    ]).description("优化器与学习率"),

    Schema.object({
        loss_type: Schema.union(["l1", "l2", "huber", "smooth_l1"]).default("l2").description("loss 类型"),
        huber_schedule: Schema.union(["constant", "exponential", "snr"]).default("snr").description("Huber schedule"),
        huber_c: Schema.number().step(0.01).default(0.1).description("Huber c"),
        huber_scale: Schema.number().step(0.1).default(1.0).description("Huber scale"),
        min_snr_gamma: Schema.number().step(0.1).description("Min-SNR gamma；常用 5"),
        noise_offset: Schema.number().step(0.01).description("noise offset"),
        multires_noise_iterations: Schema.number().min(1).step(1).description("multi-resolution noise iterations；不要与 noise_offset 同时使用"),
        multires_noise_discount: Schema.number().step(0.1).description("multi-resolution noise discount"),
        v_parameterization: Schema.boolean().default(false).description("v-parameterization training"),
        scale_v_pred_loss_like_noise_pred: Schema.boolean().default(false).description("缩放 v-pred loss"),
        debiased_estimation_loss: Schema.boolean().default(false).description("Debiased Estimation loss"),
        masked_loss: Schema.boolean().default(false).description("masked loss"),
        conditioning_data_dir: Schema.string().role("filepicker", { type: "folder" }).description("mask/conditioning 图目录"),
    }).description("损失与噪声"),

    Schema.object({
        mixed_precision: Schema.union(["no", "fp16", "bf16"]).default("bf16").description("训练精度"),
        full_fp16: Schema.boolean().default(false).description("full FP16；要求 mixed_precision=fp16"),
        full_bf16: Schema.boolean().default(false).description("full BF16；要求 mixed_precision=bf16"),
        no_half_vae: Schema.boolean().default(false).description("VAE 固定 FP32"),
        xformers: Schema.boolean().default(true).description("U-Net xformers attention；与 sdpa/diffusers_xformers 二选一"),
        sdpa: Schema.boolean().default(false).description("U-Net PyTorch SDPA；与 xformers/diffusers_xformers 二选一"),
        diffusers_xformers: Schema.boolean().default(false).description("走 Diffusers VAE xformers 独立分支；启用时关闭 U-Net xformers/sdpa"),
        cache_latents: Schema.boolean().default(true).description("缓存 VAE latents"),
        cache_latents_to_disk: Schema.boolean().default(true).description("latents 写盘；启用会隐含 cache_latents"),
        vae_batch_size: Schema.number().min(1).default(1).description("VAE cache batch"),
        cache_text_encoder_outputs: Schema.boolean().default(false).description("可选：缓存两个 SDXL 文本编码器输出；训练 TE 时必须关闭"),
        cache_text_encoder_outputs_to_disk: Schema.boolean().default(false).description("TE outputs 写盘；启用会隐含 TE output cache"),
        highvram: Schema.boolean().default(false).description("高显存加载路径"),
    }).description("精度、Attention 与缓存"),

    Schema.intersect([
        Schema.object({
            enable_preview: Schema.boolean().default(false).description("启用训练预览"),
        }),
        Schema.union([
            Schema.object({
                enable_preview: Schema.const(true).required(),
                sample_prompts: Schema.string().role("textarea").default("masterpiece, best quality, 1girl, solo --w 1024 --h 1024 --l 7 --s 24 --d 1337").description("sample prompt 参数"),
                sample_sampler: Schema.union(["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]).default("euler_a").description("采样器"),
                sample_every_n_epochs: Schema.number().min(1).default(1).description("每 N epoch 预览；若填写 step cadence，后端会移除此默认值"),
                sample_every_n_steps: Schema.number().min(1).description("每 N step 预览；填写后优先于 epoch cadence"),
                sample_at_first: Schema.boolean().default(false).description("训练前先预览"),
            }),
            Schema.object({}),
        ]),
    ]).description("训练预览"),

    SHARED_SCHEMAS.LOG_SETTINGS,

    Schema.intersect([
        Schema.object({
            deepspeed: Schema.boolean().default(false).description("启用 DeepSpeed；不能与 fused backward 模式组合"),
            ddp_timeout: Schema.number().min(0).description("DDP timeout"),
            ddp_gradient_as_bucket_view: Schema.boolean().default(false).description("DDP gradient_as_bucket_view"),
            ddp_static_graph: Schema.boolean().default(false).description("DDP static graph"),
            seed: Schema.number().default(1337).description("随机种子"),
            ui_custom_params: Schema.string().role("textarea").description("高级自定义 TOML；仅用于 sdxl_train.py 已支持参数"),
        }),
        Schema.union([
            Schema.object({
                deepspeed: Schema.const(true).required(),
                zero_stage: Schema.union([0, 1, 2, 3]).default(2).description("ZeRO stage"),
                offload_optimizer_device: Schema.union(["cpu", "nvme"]).description("optimizer offload"),
                offload_optimizer_nvme_path: Schema.string().description("optimizer NVMe path"),
                offload_param_device: Schema.union(["cpu", "nvme"]).description("parameter offload；仅 ZeRO-3"),
                offload_param_nvme_path: Schema.string().description("parameter NVMe path"),
                zero3_init_flag: Schema.boolean().default(false).description("ZeRO-3 init"),
                zero3_save_16bit_model: Schema.boolean().default(false).description("ZeRO-3 保存 16-bit model"),
                fp16_master_weights_and_gradients: Schema.boolean().default(false).description("特殊 ZeRO-2 + CPU offload + FP16 模式"),
            }),
            Schema.object({}),
        ]),
    ]).description("分布式与高级设置")
]);
