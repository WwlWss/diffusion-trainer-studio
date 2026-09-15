"""Apply the optional Anima Qwen3 joint-finetune patch to the pinned sd-scripts tree.

This is a staging tool until the project can point the submodule at a WwlWss-owned
sd-scripts fork. It is pinned to the exact current submodule commit and fails
closed if any expected source block has drifted.

Usage:
    python tools/apply_anima_qwen3_sd_scripts_patch.py --check
    python tools/apply_anima_qwen3_sd_scripts_patch.py --write
"""

from __future__ import annotations

import argparse
import ast
import subprocess
from pathlib import Path


EXPECTED_SD_SCRIPTS_HEAD = "45dddfccb704b6b0591f65d98f0d695c997b3115"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def patch_anima_train_utils(text: str) -> str:
    text = replace_once(
        text,
        "from library import anima_models, anima_utils, checkpoint_io, sampling, qwen_image_autoencoder_kl\n",
        "from library import anima_models, anima_utils, checkpoint_io, sampling, qwen_image_autoencoder_kl, huggingface_util\n",
        "anima_train_utils import",
    )

    qwen_arg = '''    parser.add_argument(\n        "--qwen3",\n        type=str,\n        default=None,\n        help="Path to Qwen3-0.6B model (safetensors file or directory)",\n    )\n'''
    text = replace_once(
        text,
        qwen_arg,
        qwen_arg
        + '''    parser.add_argument(\n        "--train_qwen3_text_encoder",\n        action="store_true",\n        help="Jointly train the Qwen3-0.6B text encoder during Anima full finetuning.",\n    )\n    parser.add_argument(\n        "--qwen3_lr",\n        type=float,\n        default=5e-7,\n        help="Learning rate for trainable Qwen3 text encoder (default: 5e-7).",\n    )\n    parser.add_argument(\n        "--qwen3_gradient_checkpointing",\n        action="store_true",\n        help="Enable gradient checkpointing in Qwen3 when it is trainable.",\n    )\n    parser.add_argument(\n        "--qwen3_output_dir",\n        type=str,\n        default=None,\n        help="Optional directory for Qwen3 sidecar checkpoints. Defaults to the Anima checkpoint directory.",\n    )\n''',
        "Anima Qwen3 CLI arguments",
    )

    text = replace_once(
        text,
        '''    llm_adapter_lr: Optional[float] = None,\n):\n''',
        '''    llm_adapter_lr: Optional[float] = None,\n    return_names: bool = False,\n):\n''',
        "get_anima_param_groups signature",
    )
    text = replace_once(
        text,
        '''    param_groups = []\n    for lr, params, name in [\n''',
        '''    param_groups = []\n    group_names = []\n    for lr, params, name in [\n''',
        "Anima param group names init",
    )
    text = replace_once(
        text,
        '''        elif len(params) > 0:\n            param_groups.append({"params": params, "lr": lr})\n\n    total_trainable = sum(p.numel() for group in param_groups for p in group["params"] if p.requires_grad)\n''',
        '''        elif len(params) > 0:\n            param_groups.append({"params": params, "lr": lr})\n            group_names.append(name)\n\n    total_trainable = sum(p.numel() for group in param_groups for p in group["params"] if p.requires_grad)\n''',
        "Anima param group names append",
    )
    text = replace_once(
        text,
        '''    return param_groups\n\n\n# Save functions\n''',
        '''    if return_names:\n        return param_groups, group_names\n    return param_groups\n\n\ndef get_qwen3_sidecar_path(args: argparse.Namespace, main_ckpt_file: str, create_dir: bool = True) -> str:\n    output_dir = args.qwen3_output_dir or os.path.dirname(main_ckpt_file) or "."\n    if create_dir:\n        os.makedirs(output_dir, exist_ok=True)\n    basename = os.path.splitext(os.path.basename(main_ckpt_file))[0]\n    return os.path.join(output_dir, basename + "_qwen3.safetensors")\n\n\ndef save_qwen3_sidecar(\n    args: argparse.Namespace,\n    main_ckpt_file: str,\n    qwen3_text_encoder,\n    save_dtype: torch.dtype,\n    force_sync_upload: bool = False,\n):\n    if qwen3_text_encoder is None:\n        return\n\n    qwen3_file = get_qwen3_sidecar_path(args, main_ckpt_file)\n    temp_file = qwen3_file + ".tmp"\n    try:\n        anima_utils.save_qwen3_text_encoder(temp_file, qwen3_text_encoder, save_dtype)\n        os.replace(temp_file, qwen3_file)\n    finally:\n        if os.path.exists(temp_file):\n            os.remove(temp_file)\n\n    if args.huggingface_repo_id is not None:\n        huggingface_util.upload(\n            args, qwen3_file, "/" + os.path.basename(qwen3_file), force_sync_upload=force_sync_upload\n        )\n\n\ndef remove_old_qwen3_sidecar(\n    args: argparse.Namespace,\n    on_epoch_end: bool,\n    epoch: int,\n    num_train_epochs: int,\n    global_step: int,\n):\n    if on_epoch_end:\n        epoch_no = epoch + 1\n        if args.save_every_n_epochs is None or epoch_no % args.save_every_n_epochs != 0 or epoch_no >= num_train_epochs:\n            return\n        remove_no = checkpoint_io.get_remove_epoch_no(args, epoch_no)\n        if remove_no is None:\n            return\n        old_name = checkpoint_io.get_epoch_ckpt_name(args, ".safetensors", remove_no)\n    else:\n        remove_no = checkpoint_io.get_remove_step_no(args, global_step)\n        if remove_no is None:\n            return\n        old_name = checkpoint_io.get_step_ckpt_name(args, ".safetensors", remove_no)\n\n    old_main = os.path.join(args.output_dir, old_name)\n    old_qwen3 = get_qwen3_sidecar_path(args, old_main, create_dir=False)\n    if os.path.exists(old_qwen3):\n        logger.info(f"removing old Qwen3 sidecar: {old_qwen3}")\n        os.remove(old_qwen3)\n\n\n# Save functions\n''',
        "Qwen3 sidecar helpers",
    )

    text = replace_once(
        text,
        '''def save_anima_model_on_train_end(\n    args: argparse.Namespace,\n    save_dtype: torch.dtype,\n    epoch: int,\n    global_step: int,\n    dit: anima_models.Anima,\n):\n''',
        '''def save_anima_model_on_train_end(\n    args: argparse.Namespace,\n    save_dtype: torch.dtype,\n    epoch: int,\n    global_step: int,\n    dit: anima_models.Anima,\n    qwen3_text_encoder=None,\n):\n''',
        "final saver signature",
    )
    text = replace_once(
        text,
        '''        # Save with 'net.' prefix for ComfyUI compatibility\n        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)\n\n    checkpoint_io.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)\n''',
        '''        # Save Qwen3 first: if the sidecar write fails, do not publish a new\n        # DiT checkpoint that has no matching text encoder.\n        save_qwen3_sidecar(args, ckpt_file, qwen3_text_encoder, save_dtype, force_sync_upload=True)\n        # Save with 'net.' prefix for ComfyUI compatibility\n        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)\n\n    checkpoint_io.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)\n''',
        "final Qwen3 sidecar save",
    )

    text = replace_once(
        text,
        '''def save_anima_model_on_epoch_end_or_stepwise(\n    args: argparse.Namespace,\n    on_epoch_end: bool,\n    accelerator: Accelerator,\n    save_dtype: torch.dtype,\n    epoch: int,\n    num_train_epochs: int,\n    global_step: int,\n    dit: anima_models.Anima,\n):\n''',
        '''def save_anima_model_on_epoch_end_or_stepwise(\n    args: argparse.Namespace,\n    on_epoch_end: bool,\n    accelerator: Accelerator,\n    save_dtype: torch.dtype,\n    epoch: int,\n    num_train_epochs: int,\n    global_step: int,\n    dit: anima_models.Anima,\n    qwen3_text_encoder=None,\n):\n''',
        "step saver signature",
    )
    text = replace_once(
        text,
        '''        dit_sd = dit.state_dict()\n        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)\n\n    checkpoint_io.save_sd_model_on_epoch_end_or_stepwise_common(\n''',
        '''        dit_sd = dit.state_dict()\n        save_qwen3_sidecar(args, ckpt_file, qwen3_text_encoder, save_dtype)\n        anima_utils.save_anima_model(ckpt_file, dit_sd, sai_metadata, save_dtype)\n\n    checkpoint_io.save_sd_model_on_epoch_end_or_stepwise_common(\n''',
        "step Qwen3 sidecar save",
    )
    text = replace_once(
        text,
        '''        sd_saver,\n        None,\n    )\n\n\n# Sampling (Euler discrete for rectified flow)\n''',
        '''        sd_saver,\n        None,\n    )\n    if qwen3_text_encoder is not None:\n        remove_old_qwen3_sidecar(args, on_epoch_end, epoch, num_train_epochs, global_step)\n\n\n# Sampling (Euler discrete for rectified flow)\n''',
        "Qwen3 sidecar rotation",
    )
    return text


def patch_anima_utils(text: str) -> str:
    old = '''    logger.info(f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")\n    return model, tokenizer\n\n\ndef load_t5_tokenizer'''
    new = '''    logger.info(f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")\n    return model, tokenizer\n\n\ndef save_qwen3_text_encoder(save_path: str, text_encoder, dtype: Optional[torch.dtype] = None):\n    """Save the trainable Qwen3 base model as a standalone safetensors sidecar."""\n    state_dict = {}\n    for key, value in text_encoder.state_dict().items():\n        target_dtype = dtype if dtype is not None else value.dtype\n        # Copy directly to CPU. Avoid clone().to("cpu"), which first creates an\n        # unnecessary GPU clone and can add a large checkpoint-time VRAM spike.\n        value = value.detach().to(device="cpu", dtype=target_dtype, copy=True).contiguous()\n        state_dict[key] = value\n\n    output_dir = os.path.dirname(save_path)\n    if output_dir:\n        os.makedirs(output_dir, exist_ok=True)\n    save_file(state_dict, save_path, metadata={"format": "pt", "model": "qwen3-0.6b-anima-text-encoder"})\n    logger.info(f"Saved Qwen3 text encoder to {save_path}")\n\n\ndef load_t5_tokenizer'''
    return replace_once(text, old, new, "Qwen3 saver")


def patch_strategy_anima(text: str) -> str:
    text = replace_once(
        text,
        '''        encoder_device = qwen3_text_encoder.device\n\n        qwen3_input_ids = qwen3_input_ids.to(encoder_device)\n''',
        '''        encoder_device = getattr(qwen3_text_encoder, "device", None)\n        if encoder_device is None:\n            encoder_device = next(qwen3_text_encoder.parameters()).device\n\n        qwen3_input_ids = qwen3_input_ids.to(encoder_device)\n''',
        "Qwen3 DDP device lookup",
    )
    text = replace_once(
        text,
        '''        prompt_embeds = outputs.last_hidden_state\n        prompt_embeds[~qwen3_attn_mask.bool()] = 0\n\n        return [prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask]\n''',
        '''        prompt_embeds = outputs.last_hidden_state\n        if prompt_embeds.requires_grad:\n            prompt_embeds = prompt_embeds.masked_fill(~qwen3_attn_mask.bool().unsqueeze(-1), 0)\n        else:\n            # Preserve the original zero-allocation path when Qwen3 is frozen.\n            prompt_embeds[~qwen3_attn_mask.bool()] = 0\n\n        return [prompt_embeds, qwen3_attn_mask, t5_input_ids, t5_attn_mask]\n''',
        "Qwen3 autograd-safe padding mask",
    )
    return text


def patch_anima_train(text: str) -> str:
    text = replace_once(text, "import gc\nimport math\n", "import gc\nimport json\nimport math\n", "anima_train json import")

    text = replace_once(
        text,
        '''    setup_logging(args, reset=True)\n\n    flux_train_utils.log_timestep_sampling_info(args)\n\n    # backward compatibility\n''',
        '''    setup_logging(args, reset=True)\n\n    train_qwen3 = bool(getattr(args, "train_qwen3_text_encoder", False))\n    if train_qwen3:\n        if args.cache_text_encoder_outputs or args.cache_text_encoder_outputs_to_disk:\n            raise ValueError(\n                "Qwen3 training is incompatible with cache_text_encoder_outputs and "\n                "cache_text_encoder_outputs_to_disk."\n            )\n        if args.deepspeed:\n            raise ValueError("Qwen3 joint finetuning does not support DeepSpeed in the first implementation.")\n        if args.fused_backward_pass:\n            raise ValueError("Qwen3 joint finetuning does not support fused_backward_pass in the first implementation.")\n        if args.learning_rate is None or args.learning_rate <= 0:\n            raise ValueError("Qwen3 joint finetuning currently requires learning_rate > 0 for the Anima DiT.")\n        if args.qwen3_lr is None or args.qwen3_lr <= 0:\n            raise ValueError("qwen3_lr must be greater than zero when training Qwen3.")\n        supported_qwen_optimizers = {\n            "adamw",\n            "adamw8bit",\n            "pagedadamw8bit",\n            "lion",\n            "lion8bit",\n            "pagedlion8bit",\n            "sgdnesterov",\n            "sgdnesterov8bit",\n        }\n        optimizer_name = str(args.optimizer_type or "AdamW").lower()\n        if optimizer_name not in supported_qwen_optimizers:\n            raise ValueError(\n                "Qwen3 joint finetuning requires an optimizer with reliable independent parameter-group LRs. "\n                f"Unsupported optimizer: {args.optimizer_type}"\n            )\n\n    flux_train_utils.log_timestep_sampling_info(args)\n\n    # backward compatibility\n''',
        "Qwen3 trainer guards",
    )

    text = replace_once(
        text,
        '''    # Prepare text encoder (always frozen for Anima)\n    qwen3_text_encoder.to(weight_dtype)\n    qwen3_text_encoder.requires_grad_(False)\n\n    # Cache text encoder outputs\n''',
        '''    # Prepare text encoder. Preserve the original frozen path unless the\n    # optional joint-finetune flag is explicitly enabled.\n    qwen3_text_encoder.to(weight_dtype)\n    qwen3_text_encoder.requires_grad_(train_qwen3)\n    if train_qwen3 and args.qwen3_gradient_checkpointing:\n        if not hasattr(qwen3_text_encoder, "gradient_checkpointing_enable"):\n            raise ValueError("The loaded Qwen3 text encoder does not support gradient checkpointing.")\n        qwen3_text_encoder.gradient_checkpointing_enable()\n\n    # Cache text encoder outputs\n''',
        "Qwen3 trainability setup",
    )

    text = replace_once(
        text,
        '''    # Setup optimizer with parameter groups\n    if train_dit:\n        param_groups = anima_train_utils.get_anima_param_groups(\n            dit,\n            base_lr=args.learning_rate,\n            self_attn_lr=args.self_attn_lr,\n            cross_attn_lr=args.cross_attn_lr,\n            mlp_lr=args.mlp_lr,\n            mod_lr=args.mod_lr,\n            llm_adapter_lr=args.llm_adapter_lr,\n        )\n    else:\n        param_groups = []\n\n    training_models = []\n    if train_dit:\n        training_models.append(dit)\n''',
        '''    # Setup optimizer with parameter groups\n    if train_dit:\n        param_groups, lr_names = anima_train_utils.get_anima_param_groups(\n            dit,\n            base_lr=args.learning_rate,\n            self_attn_lr=args.self_attn_lr,\n            cross_attn_lr=args.cross_attn_lr,\n            mlp_lr=args.mlp_lr,\n            mod_lr=args.mod_lr,\n            llm_adapter_lr=args.llm_adapter_lr,\n            return_names=True,\n        )\n    else:\n        param_groups = []\n        lr_names = []\n\n    if train_qwen3:\n        qwen3_params = [p for p in qwen3_text_encoder.parameters() if p.requires_grad]\n        if not qwen3_params:\n            raise RuntimeError("Qwen3 training was enabled but no trainable Qwen3 parameters were found.")\n        param_groups.append({"params": qwen3_params, "lr": args.qwen3_lr})\n        lr_names.append("qwen3")\n\n    training_models = []\n    if train_dit:\n        training_models.append(dit)\n    if train_qwen3:\n        training_models.append(qwen3_text_encoder)\n''',
        "Qwen3 optimizer group",
    )

    text = replace_once(
        text,
        '''    else:\n        if train_dit:\n            dit = accelerator.prepare(dit, device_placement=[not is_swapping_blocks])\n            if is_swapping_blocks:\n                accelerator.unwrap_model(dit).move_to_device_except_swap_blocks(accelerator.device)\n        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)\n\n    # Move non-training models back to GPU\n    if not args.cache_text_encoder_outputs and qwen3_text_encoder is not None:\n        qwen3_text_encoder.to(accelerator.device)\n''',
        '''    else:\n        if train_dit:\n            dit = accelerator.prepare(dit, device_placement=[not is_swapping_blocks])\n            if is_swapping_blocks:\n                accelerator.unwrap_model(dit).move_to_device_except_swap_blocks(accelerator.device)\n        if train_qwen3:\n            qwen3_text_encoder = accelerator.prepare(qwen3_text_encoder)\n        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)\n\n        # Use only prepared/wrapped model references for accumulation and clipping.\n        training_models = []\n        if train_dit:\n            training_models.append(dit)\n        if train_qwen3:\n            training_models.append(qwen3_text_encoder)\n\n    # Move non-training models back to GPU\n    if not args.cache_text_encoder_outputs and qwen3_text_encoder is not None and not train_qwen3:\n        qwen3_text_encoder.to(accelerator.device)\n''',
        "Qwen3 accelerator preparation",
    )

    text = replace_once(
        text,
        '''    # resume\n    args_util.resume_from_local_or_hf_if_specified(accelerator, args)\n\n    if args.fused_backward_pass:\n''',
        '''    # Save-state compatibility marker. Frozen-Qwen jobs keep their old\n    # state layout; Qwen-training states are explicitly marked so ON/OFF resumes\n    # cannot be mixed accidentally.\n    if train_qwen3:\n        def save_qwen3_mode_hook(models, weights, output_dir):\n            if accelerator.is_main_process:\n                with open(os.path.join(output_dir, "anima_qwen3_training.json"), "w", encoding="utf-8") as f:\n                    json.dump({"train_qwen3_text_encoder": True}, f)\n\n        accelerator.register_save_state_pre_hook(save_qwen3_mode_hook)\n\n    def load_qwen3_mode_hook(models, input_dir):\n        marker = os.path.join(input_dir, "anima_qwen3_training.json")\n        if not os.path.exists(marker):\n            if train_qwen3:\n                raise ValueError("Cannot resume Qwen3 joint finetuning from a state without Qwen3 training metadata.")\n            return\n        with open(marker, "r", encoding="utf-8") as f:\n            saved_mode = bool(json.load(f).get("train_qwen3_text_encoder"))\n        if saved_mode != train_qwen3:\n            raise ValueError("Cannot resume across different Qwen3 text-encoder training modes.")\n\n    accelerator.register_load_state_pre_hook(load_qwen3_mode_hook)\n\n    # resume\n    args_util.resume_from_local_or_hf_if_specified(accelerator, args)\n\n    if args.fused_backward_pass:\n''',
        "Qwen3 save-state mode guard",
    )

    text = replace_once(
        text,
        '''    # For --sample_at_first\n    optimizer_eval_fn()\n    anima_train_utils.sample_images(\n        accelerator,\n        args,\n        0,\n        global_step,\n        dit,\n        vae,\n        qwen3_text_encoder,\n        tokenize_strategy,\n        text_encoding_strategy,\n        sample_prompts_te_outputs,\n    )\n    optimizer_train_fn()\n''',
        '''    def sample_with_current_qwen_state(epoch_value, step_value):\n        qwen3_was_training = qwen3_text_encoder is not None and qwen3_text_encoder.training\n        if train_qwen3 and qwen3_text_encoder is not None:\n            qwen3_text_encoder.eval()\n        try:\n            anima_train_utils.sample_images(\n                accelerator,\n                args,\n                epoch_value,\n                step_value,\n                dit,\n                vae,\n                qwen3_text_encoder,\n                tokenize_strategy,\n                text_encoding_strategy,\n                sample_prompts_te_outputs,\n            )\n        finally:\n            if train_qwen3 and qwen3_text_encoder is not None and qwen3_was_training:\n                qwen3_text_encoder.train()\n\n    # For --sample_at_first\n    optimizer_eval_fn()\n    sample_with_current_qwen_state(0, global_step)\n    optimizer_train_fn()\n''',
        "Qwen3 sampling train/eval wrapper",
    )

    text = replace_once(
        text,
        '''    if qwen3_text_encoder is not None:\n        logger.info(f"qwen3 device: {qwen3_text_encoder.device}")\n''',
        '''    if qwen3_text_encoder is not None:\n        unwrapped_qwen3 = accelerator.unwrap_model(qwen3_text_encoder)\n        logger.info(f"qwen3 device: {next(unwrapped_qwen3.parameters()).device}")\n''',
        "Qwen3 wrapped device logging",
    )

    text = replace_once(
        text,
        '''                    with torch.no_grad():\n                        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoding_strategy.encode_tokens(\n                            tokenize_strategy, [qwen3_text_encoder], input_ids_list\n                        )\n''',
        '''                    with torch.set_grad_enabled(train_qwen3):\n                        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoding_strategy.encode_tokens(\n                            tokenize_strategy, [qwen3_text_encoder], input_ids_list\n                        )\n''',
        "Qwen3 gradient-enabled text encoding",
    )

    text = replace_once(
        text,
        '''                optimizer_eval_fn()\n                anima_train_utils.sample_images(\n                    accelerator,\n                    args,\n                    None,\n                    global_step,\n                    dit,\n                    vae,\n                    qwen3_text_encoder,\n                    tokenize_strategy,\n                    text_encoding_strategy,\n                    sample_prompts_te_outputs,\n                )\n''',
        '''                optimizer_eval_fn()\n                sample_with_current_qwen_state(None, global_step)\n''',
        "step sampling wrapper use",
    )

    text = replace_once(
        text,
        '''                            global_step,\n                            accelerator.unwrap_model(dit) if train_dit else None,\n                        )\n''',
        '''                            global_step,\n                            accelerator.unwrap_model(dit) if train_dit else None,\n                            accelerator.unwrap_model(qwen3_text_encoder) if train_qwen3 else None,\n                        )\n''',
        "step checkpoint Qwen3 sidecar argument",
    )

    text = replace_once(
        text,
        '''                    ["base", "self_attn", "cross_attn", "mlp", "mod", "llm_adapter"] if train_dit else [],\n''',
        '''                    lr_names,\n''',
        "dynamic Anima LR names",
    )

    text = replace_once(
        text,
        '''                    global_step,\n                    accelerator.unwrap_model(dit) if train_dit else None,\n                )\n\n        anima_train_utils.sample_images(\n            accelerator,\n            args,\n            epoch + 1,\n            global_step,\n            dit,\n            vae,\n            qwen3_text_encoder,\n            tokenize_strategy,\n            text_encoding_strategy,\n            sample_prompts_te_outputs,\n        )\n''',
        '''                    global_step,\n                    accelerator.unwrap_model(dit) if train_dit else None,\n                    accelerator.unwrap_model(qwen3_text_encoder) if train_qwen3 else None,\n                )\n\n        sample_with_current_qwen_state(epoch + 1, global_step)\n''',
        "epoch checkpoint and sampling Qwen3 integration",
    )

    text = replace_once(
        text,
        '''    # End training\n    is_main_process = accelerator.is_main_process\n    dit = accelerator.unwrap_model(dit)\n\n    accelerator.end_training()\n''',
        '''    # End training\n    is_main_process = accelerator.is_main_process\n    dit = accelerator.unwrap_model(dit)\n    qwen3_to_save = accelerator.unwrap_model(qwen3_text_encoder) if train_qwen3 else None\n\n    accelerator.end_training()\n''',
        "final Qwen3 unwrap",
    )

    text = replace_once(
        text,
        '''            global_step,\n            dit,\n        )\n''',
        '''            global_step,\n            dit,\n            qwen3_to_save,\n        )\n''',
        "final Qwen3 sidecar argument",
    )
    return text


def patch_files(sd_scripts_dir: Path) -> dict[Path, str]:
    patchers = {
        sd_scripts_dir / "anima_train.py": patch_anima_train,
        sd_scripts_dir / "library/anima_train_utils.py": patch_anima_train_utils,
        sd_scripts_dir / "library/anima_utils.py": patch_anima_utils,
        sd_scripts_dir / "library/strategy_anima.py": patch_strategy_anima,
    }
    result = {}
    for path, patcher in patchers.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        original = path.read_text(encoding="utf-8-sig")
        result[path] = patcher(original)
    return result


def verify_head(sd_scripts_dir: Path) -> None:
    completed = subprocess.run(
        ["git", "-C", str(sd_scripts_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    head = completed.stdout.strip()
    if head != EXPECTED_SD_SCRIPTS_HEAD:
        raise RuntimeError(
            f"sd-scripts HEAD is {head}, expected pinned {EXPECTED_SD_SCRIPTS_HEAD}. "
            "Re-review the patch against the new upstream before applying it."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd-scripts-dir", default="sd-scripts")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Validate all source anchors and patched Python syntax without writing.")
    mode.add_argument("--write", action="store_true", help="Apply the patch to the pinned sd-scripts working tree.")
    args = parser.parse_args()

    sd_scripts_dir = Path(args.sd_scripts_dir).resolve()
    verify_head(sd_scripts_dir)

    capability_file = sd_scripts_dir / "library/anima_train_utils.py"
    current_capability = capability_file.read_text(encoding="utf-8-sig") if capability_file.is_file() else ""
    if "--train_qwen3_text_encoder" in current_capability:
        raise RuntimeError("The sd-scripts tree already appears to contain Qwen3 joint-finetune support; refusing to patch twice.")

    patched = patch_files(sd_scripts_dir)
    for path, source in patched.items():
        ast.parse(source, filename=str(path))

    if args.check:
        print("Anima Qwen3 sd-scripts patch check passed; no files were changed.")
        return

    for path, source in patched.items():
        path.write_text(source, encoding="utf-8")

    subprocess.run(["git", "-C", str(sd_scripts_dir), "diff", "--check"], check=True)
    print("Anima Qwen3 sd-scripts patch applied. Review and commit the sd-scripts diff in a WwlWss-owned fork.")


if __name__ == "__main__":
    main()
