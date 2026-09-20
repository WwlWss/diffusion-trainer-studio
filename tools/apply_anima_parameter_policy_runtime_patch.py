"""Stage DTS Parameter Policy runtime into the pinned Anima sd-scripts tree.

The patch is intentionally source-anchored and applies only to the reviewed
sd-scripts commit.  It never mutates the submodule in normal runtime staging.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tools.apply_anima_qwen3_sd_scripts_patch import EXPECTED_SD_SCRIPTS_HEAD


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def patch_train_network(text: str) -> str:
    text = replace_once(
        text,
        """        mean_combined_norm=None,\n    ):\n        logs = {\"loss/current\": current_loss, \"loss/average\": avr_loss}\n""",
        """        mean_combined_norm=None,\n        parameter_policy_session=None,\n    ):\n        logs = {\"loss/current\": current_loss, \"loss/average\": avr_loss}\n\n        if parameter_policy_session is not None:\n            logs.update(parameter_policy_session.component_lr_logs(lr_scheduler))\n            return logs\n""",
        "NetworkTrainer component LR logging",
    )
    text = replace_once(
        text,
        """    def is_train_text_encoder(self, args):\n        return not args.network_train_unet_only\n\n    def cache_text_encoder_outputs_if_needed""",
        """    def is_train_text_encoder(self, args):\n        policy_value = getattr(self, \"_parameter_policy_train_text_encoder\", None)\n        if policy_value is not None:\n            return policy_value\n        return not args.network_train_unet_only\n\n    def get_parameter_policy_train_type(self, args):\n        return None\n\n    def configure_parameter_policy_training(self, args, session, text_encoders):\n        raise NotImplementedError(\n            f\"{type(self).__name__} has not integrated DTS Parameter Policy runtime.\"\n        )\n\n    def cache_text_encoder_outputs_if_needed""",
        "NetworkTrainer Parameter Policy hooks",
    )
    text = replace_once(
        text,
        """        session_id = random.randint(0, 2**32)\n        training_started_at = time.time()\n        args_util.verify_training_args(args)\n""",
        """        session_id = random.randint(0, 2**32)\n        training_started_at = time.time()\n        parameter_policy_requested = bool(\n            str(getattr(args, \"parameter_policy_config\", \"\") or \"\").strip()\n        )\n        parameter_policy_train_type = (\n            self.get_parameter_policy_train_type(args) if parameter_policy_requested else None\n        )\n        if parameter_policy_requested and parameter_policy_train_type is None:\n            raise ValueError(\n                \"--parameter_policy_config was provided, but this trainer has not enabled DTS Parameter Policy runtime.\"\n            )\n        args_util.verify_training_args(args)\n""",
        "NetworkTrainer Parameter Policy request gate",
    )
    text = replace_once(
        text,
        """        if args.network_weights is not None:\n            # FIXME consider alpha of weights: this assumes that the alpha is not changed\n            info = network.load_weights(args.network_weights)\n            accelerator.print(f\"load network weights from {args.network_weights}: {info}\")\n\n        if args.gradient_checkpointing:\n""",
        """        if args.network_weights is not None:\n            # FIXME consider alpha of weights: this assumes that the alpha is not changed\n            info = network.load_weights(args.network_weights)\n            accelerator.print(f\"load network weights from {args.network_weights}: {info}\")\n\n        parameter_policy_session = None\n        parameter_policy_bridge = None\n        if parameter_policy_requested:\n            from library import dts_parameter_policy_bridge as parameter_policy_bridge\n\n            parameter_policy_session = parameter_policy_bridge.create_parameter_policy_session(\n                args=args,\n                train_type=parameter_policy_train_type,\n                roots={\"network\": network},\n            )\n            train_unet, train_text_encoder = self.configure_parameter_policy_training(\n                args, parameter_policy_session, text_encoders\n            )\n\n        if args.gradient_checkpointing:\n""",
        "NetworkTrainer session creation",
    )
    text = replace_once(
        text,
        """        # make backward compatibility for text_encoder_lr\n        support_multiple_lrs = hasattr(network, \"prepare_optimizer_params_with_multiple_te_lrs\")\n""",
        """        if parameter_policy_session is None:\n            # make backward compatibility for text_encoder_lr\n            support_multiple_lrs = hasattr(network, \"prepare_optimizer_params_with_multiple_te_lrs\")\n""",
        "NetworkTrainer legacy optimizer branch start",
    )
    # Indent the legacy optimizer block through get_optimizer.
    legacy = """        if parameter_policy_session is None:\n            # make backward compatibility for text_encoder_lr\n            support_multiple_lrs = hasattr(network, \"prepare_optimizer_params_with_multiple_te_lrs\")\n"""
    start = text.index(legacy)
    end_marker = """        # prepare dataloader\n"""
    end = text.index(end_marker, start)
    block = text[start:end]
    lines = block.splitlines(True)
    # first two lines already have the correct nesting; indent the remaining legacy body
    rebuilt = "".join(lines[:3]) + "".join("    " + line if line.strip() else line for line in lines[3:])
    if rebuilt == block:
        raise RuntimeError("NetworkTrainer legacy optimizer indentation did not change")
    rebuilt += """        else:\n            # Legacy text_encoder_lr is metadata-only in Component mode.\n            text_encoder_lr = None\n            lr_descriptions = None\n            optimizer_name = \"DTSParameterPolicy\"\n            optimizer_args = \"\"\n            optimizer = parameter_policy_session.optimizer\n            optimizer_train_fn = optimizer.train\n            optimizer_eval_fn = optimizer.eval\n\n"""
    text = text[:start] + rebuilt + text[end:]

    text = replace_once(
        text,
        """        # lr schedulerを用意する\n        lr_scheduler = optimizer_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)\n""",
        """        # lr schedulerを用意する\n        if parameter_policy_session is None:\n            lr_scheduler = optimizer_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)\n        else:\n            scheduler_factory = parameter_policy_bridge.make_legacy_scheduler_factory(\n                args=args,\n                get_scheduler_fix=optimizer_util.get_scheduler_fix,\n                num_processes=accelerator.num_processes,\n            )\n            lr_scheduler = parameter_policy_session.build_scheduler(scheduler_factory)\n""",
        "NetworkTrainer component scheduler",
    )
    text = replace_once(
        text,
        """        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)\n\n        if not cache_latents:""",
        """        if parameter_policy_session is None:\n            accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)\n        else:\n            parameter_policy_session.finalize_after_prepare(\n                accelerator=accelerator, optimizer=optimizer, scheduler=lr_scheduler\n            )\n\n        if not cache_latents:""",
        "NetworkTrainer post-prepare audit",
    )
    text = replace_once(
        text,
        """                        if args.max_grad_norm != 0.0:\n                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()\n                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)\n""",
        """                        if args.max_grad_norm != 0.0:\n                            params_to_clip = (\n                                parameter_policy_session.trainable_parameters\n                                if parameter_policy_session is not None\n                                else accelerator.unwrap_model(network).get_trainable_params()\n                            )\n                            accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)\n""",
        "NetworkTrainer component clipping",
    )
    text = replace_once(
        text,
        """        self._build_metadata(\n            args,\n            session_id=session_id,\n            training_started_at=training_started_at,\n            model_version=model_version,\n            text_encoder_lr=text_encoder_lr,\n            optimizer_name=optimizer_name,\n            optimizer_args=optimizer_args,\n            num_train_epochs=num_train_epochs,\n            num_batches_per_epoch=len(train_dataloader),\n            net_kwargs=net_kwargs,\n            total_batch_size=total_batch_size,\n            train_dataset_group=train_dataset_group,\n            val_dataset_group=val_dataset_group,\n            use_user_config=use_user_config,\n            use_dreambooth_method=use_dreambooth_method,\n        )\n""",
        """        self._build_metadata(\n            args,\n            session_id=session_id,\n            training_started_at=training_started_at,\n            model_version=model_version,\n            text_encoder_lr=text_encoder_lr,\n            optimizer_name=optimizer_name,\n            optimizer_args=optimizer_args,\n            num_train_epochs=num_train_epochs,\n            num_batches_per_epoch=len(train_dataloader),\n            net_kwargs=net_kwargs,\n            total_batch_size=total_batch_size,\n            train_dataset_group=train_dataset_group,\n            val_dataset_group=val_dataset_group,\n            use_user_config=use_user_config,\n            use_dreambooth_method=use_dreambooth_method,\n        )\n        if parameter_policy_session is not None:\n            self._metadata.update(parameter_policy_session.model_metadata())\n""",
        "NetworkTrainer unified Parameter Policy metadata",
    )

    text = replace_once(
        text,
        """                        mean_grad_norm,\n                        mean_combined_norm,\n                    )\n""",
        """                        mean_grad_norm,\n                        mean_combined_norm,\n                        parameter_policy_session,\n                    )\n""",
        "NetworkTrainer component log call",
    )
    text = replace_once(
        text,
        """            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)  # network.train() is called here\n\n            # TRAINING\n""",
        """            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)  # network.train() is called here\n            if parameter_policy_session is not None:\n                parameter_policy_session.assert_runtime_contract(\n                    phase=\"epoch_start\", accelerator=accelerator, optimizer=optimizer\n                )\n\n            # TRAINING\n""",
        "NetworkTrainer epoch ownership assertion",
    )
    text = replace_once(
        text,
        """        # resumeする\n        train_util.resume_from_local_or_hf_if_specified(accelerator, args)\n""",
        """        # resumeする\n        train_util.resume_from_local_or_hf_if_specified(accelerator, args)\n        if parameter_policy_session is not None:\n            parameter_policy_session.assert_runtime_contract(\n                phase=\"post_resume\", accelerator=accelerator, optimizer=optimizer\n            )\n""",
        "NetworkTrainer post-resume runtime contract",
    )
    return text


def patch_anima_train_network(text: str) -> str:
    text = replace_once(
        text,
        """    def assert_extra_args(\n""",
        """    def get_parameter_policy_train_type(self, args):\n        return \"anima-lora\"\n\n    def configure_parameter_policy_training(self, args, session, text_encoders):\n        train_dit = session.trains_prefix(\"dit.\") or session.trains_component(\"llm_adapter.adapter\")\n        train_qwen3 = session.trains_component(\"qwen3.adapter\")\n        self._parameter_policy_train_text_encoder = train_qwen3\n        return train_dit, train_qwen3\n\n    def assert_extra_args(\n""",
        "Anima LoRA Parameter Policy hooks",
    )
    text = replace_once(
        text,
        """        assert (\n            args.network_train_unet_only or not args.cache_text_encoder_outputs\n        ), \"network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません\"\n""",
        """        parameter_policy_requested = bool(\n            str(getattr(args, \"parameter_policy_config\", \"\") or \"\").strip()\n        )\n        if parameter_policy_requested:\n            from library import dts_parameter_policy_bridge\n            policy, _policy_hash = dts_parameter_policy_bridge.load_parameter_policy_file(\n                args.parameter_policy_config\n            )\n            qwen_route = policy[\"components\"].get(\"qwen3.adapter\", {\"train\": False})\n            if args.cache_text_encoder_outputs and bool(qwen_route.get(\"train\")):\n                raise ValueError(\n                    \"Anima Component LoRA cannot train qwen3.adapter while caching Text Encoder outputs.\"\n                )\n        else:\n            assert (\n                args.network_train_unet_only or not args.cache_text_encoder_outputs\n            ), \"network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません\"\n""",
        "Anima LoRA policy-aware cache validation",
    )
    text = replace_once(
        text,
        """def setup_parser() -> argparse.ArgumentParser:\n    parser = train_network.setup_parser()\n""",
        """def setup_parser() -> argparse.ArgumentParser:\n    parser = train_network.setup_parser()\n    parser.add_argument(\"--parameter_policy_config\", type=str, default=None)\n""",
        "Anima LoRA policy CLI argument",
    )
    return text


def patch_anima_train(text: str) -> str:
    has_qwen_patch = "train_qwen3 = bool(getattr(args, \"train_qwen3_text_encoder\", False))" in text

    if has_qwen_patch:
        text = replace_once(
            text,
            """    train_qwen3 = bool(getattr(args, \"train_qwen3_text_encoder\", False))\n    if train_qwen3:\n""",
            """    parameter_policy_requested = bool(\n        str(getattr(args, \"parameter_policy_config\", \"\") or \"\").strip()\n    )\n    parameter_policy_bridge = None\n    policy_qwen3_train = False\n    train_qwen3 = False\n    if parameter_policy_requested:\n        from library import dts_parameter_policy_bridge as parameter_policy_bridge\n        policy, _policy_hash = parameter_policy_bridge.load_parameter_policy_file(\n            args.parameter_policy_config\n        )\n        policy_qwen3_train = bool(policy[\"components\"].get(\"qwen3\", {\"train\": False}).get(\"train\"))\n\n    target_qwen3 = bool(getattr(args, \"train_qwen3_text_encoder\", False))\n    train_qwen3 = policy_qwen3_train if parameter_policy_requested else target_qwen3\n    if train_qwen3:\n""",
            "Anima Full policy/Qwen preflight",
        )
        text = text.replace(
            """        if args.learning_rate is None or args.learning_rate <= 0:\n            raise ValueError(\"Qwen3 joint finetuning currently requires learning_rate > 0 for the Anima DiT.\")\n        if args.qwen3_lr is None or args.qwen3_lr <= 0:\n            raise ValueError(\"qwen3_lr must be greater than zero when training Qwen3.\")\n""",
            """        if not parameter_policy_requested:\n            if args.learning_rate is None or args.learning_rate <= 0:\n                raise ValueError(\"Qwen3 joint finetuning currently requires learning_rate > 0 for the Anima DiT.\")\n            if args.qwen3_lr is None or args.qwen3_lr <= 0:\n                raise ValueError(\"qwen3_lr must be greater than zero when training Qwen3.\")\n""",
            1,
        )
        text = replace_once(
            text,
            """        if optimizer_name not in supported_qwen_optimizers:\n""",
            """        if not parameter_policy_requested and optimizer_name not in supported_qwen_optimizers:\n""",
            "Anima Full Qwen legacy optimizer guard",
        )
        text = replace_once(
            text,
            """    qwen3_text_encoder.to(weight_dtype)\n    qwen3_text_encoder.requires_grad_(train_qwen3)\n""",
            """    qwen3_text_encoder.to(weight_dtype)\n    if not parameter_policy_requested:\n        qwen3_text_encoder.requires_grad_(train_qwen3)\n    else:\n        qwen3_text_encoder.requires_grad_(False)\n""",
            "Anima Full Qwen pre-session grad ownership",
        )
    else:
        text = replace_once(
            text,
            """    setup_logging(args, reset=True)\n\n    flux_train_utils.log_timestep_sampling_info(args)\n""",
            """    setup_logging(args, reset=True)\n\n    parameter_policy_requested = bool(\n        str(getattr(args, \"parameter_policy_config\", \"\") or \"\").strip()\n    )\n    parameter_policy_bridge = None\n    policy_qwen3_train = False\n    if parameter_policy_requested:\n        from library import dts_parameter_policy_bridge as parameter_policy_bridge\n        policy, _policy_hash = parameter_policy_bridge.load_parameter_policy_file(\n            args.parameter_policy_config\n        )\n        policy_qwen3_train = bool(policy[\"components\"].get(\"qwen3\", {\"train\": False}).get(\"train\"))\n        if policy_qwen3_train:\n            raise ValueError(\n                \"Parameter Policy requests qwen3 training but train_qwen3_text_encoder target permission is disabled.\"\n            )\n\n    flux_train_utils.log_timestep_sampling_info(args)\n""",
            "Anima Full base policy preflight",
        )

    text = replace_once(
        text,
        """    # Load DiT (MiniTrainDIT + optional LLM Adapter)\n    logger.info(\"Loading Anima DiT...\")\n    dit = anima_utils.load_anima_model(\n        \"cpu\", args.pretrained_model_name_or_path, args.attn_mode, args.split_attn, \"cpu\", dit_weight_dtype=None\n    )\n\n    if args.gradient_checkpointing:\n""",
        """    # Load DiT (MiniTrainDIT + optional LLM Adapter)\n    logger.info(\"Loading Anima DiT...\")\n    dit = anima_utils.load_anima_model(\n        \"cpu\", args.pretrained_model_name_or_path, args.attn_mode, args.split_attn, \"cpu\", dit_weight_dtype=None\n    )\n\n    parameter_policy_session = None\n    if parameter_policy_requested:\n        roots = {\"dit\": dit}\n        if bool(getattr(args, \"train_qwen3_text_encoder\", False)) and qwen3_text_encoder is not None:\n            roots[\"qwen3\"] = qwen3_text_encoder\n        parameter_policy_session = parameter_policy_bridge.create_parameter_policy_session(\n            args=args, train_type=\"anima-finetune\", roots=roots\n        )\n        train_dit = parameter_policy_session.trains_prefix(\"dit.\")\n        train_qwen3 = parameter_policy_session.trains_component(\"qwen3\")\n        if train_qwen3 != policy_qwen3_train:\n            raise RuntimeError(\"Anima Qwen3 policy preflight disagrees with final runtime routing.\")\n        if train_qwen3 and not bool(getattr(args, \"train_qwen3_text_encoder\", False)):\n            raise ValueError(\"qwen3 Train requires train_qwen3_text_encoder target permission.\")\n        if train_qwen3 and not train_dit:\n            raise ValueError(\"Component-wise Anima v1 does not support Qwen3-only training.\")\n    else:\n        train_dit = args.learning_rate != 0\n\n    if args.gradient_checkpointing:\n""",
        "Anima Full session creation",
    )
    text = replace_once(
        text,
        """    train_dit = args.learning_rate != 0\n    dit.requires_grad_(train_dit)\n    if not train_dit:\n""",
        """    if parameter_policy_session is None:\n        dit.requires_grad_(train_dit)\n    if not train_dit:\n""",
        "Anima Full DiT grad ownership",
    )

    # Replace legacy parameter-group/optimizer setup with a policy branch.
    start_marker = """    # Setup optimizer with parameter groups\n"""
    start = text.index(start_marker)
    end = text.index("""    # prepare dataloader\n""", start)
    legacy = text[start:end]
    policy_block = """    # Setup optimizer with parameter groups\n    if parameter_policy_session is not None:\n        param_groups = []\n        lr_names = []\n        training_models = []\n        if train_dit:\n            training_models.append(dit)\n        if train_qwen3 and qwen3_text_encoder is not None:\n            training_models.append(qwen3_text_encoder)\n        n_params = sum(p.numel() for p in parameter_policy_session.trainable_parameters)\n        accelerator.print(f\"train dit: {train_dit}\")\n        accelerator.print(f\"train qwen3: {train_qwen3}\")\n        accelerator.print(f\"number of training models: {len(training_models)}\")\n        accelerator.print(f\"number of trainable parameters: {n_params:,}\")\n        accelerator.print(\"prepare optimizer, data loader etc.\")\n        optimizer = parameter_policy_session.optimizer\n        optimizer_train_fn = optimizer.train\n        optimizer_eval_fn = optimizer.eval\n    else:\n"""
    indented = "".join("    " + line if line.strip() else line for line in legacy.splitlines(True))
    text = text[:start] + policy_block + indented + text[end:]

    text = replace_once(
        text,
        """    # lr scheduler\n    lr_scheduler = optimizer_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)\n""",
        """    # lr scheduler\n    if parameter_policy_session is None:\n        lr_scheduler = optimizer_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)\n    else:\n        scheduler_factory = parameter_policy_bridge.make_legacy_scheduler_factory(\n            args=args,\n            get_scheduler_fix=optimizer_util.get_scheduler_fix,\n            num_processes=accelerator.num_processes,\n        )\n        lr_scheduler = parameter_policy_session.build_scheduler(scheduler_factory)\n""",
        "Anima Full component scheduler",
    )

    # Qwen-patched source has its own prepared-Qwen branch.
    if has_qwen_patch:
        prep_anchor = """        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)\n\n        # Use only prepared/wrapped model references for accumulation and clipping.\n"""
        if prep_anchor not in text:
            raise RuntimeError("Anima Full Qwen prepared branch anchor missing")
    else:
        text = replace_once(
            text,
            """        if train_dit:\n            dit = accelerator.prepare(dit, device_placement=[not is_swapping_blocks])\n            if is_swapping_blocks:\n                accelerator.unwrap_model(dit).move_to_device_except_swap_blocks(accelerator.device)\n        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)\n""",
            """        if train_dit:\n            dit = accelerator.prepare(dit, device_placement=[not is_swapping_blocks])\n            if is_swapping_blocks:\n                accelerator.unwrap_model(dit).move_to_device_except_swap_blocks(accelerator.device)\n        if train_qwen3 and qwen3_text_encoder is not None:\n            qwen3_text_encoder = accelerator.prepare(qwen3_text_encoder)\n        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)\n        if parameter_policy_session is not None:\n            training_models = [m for m in (dit if train_dit else None, qwen3_text_encoder if train_qwen3 else None) if m is not None]\n""",
            "Anima Full base selective prepare",
        )

    base_resume_anchor = """    # resume\n    args_util.resume_from_local_or_hf_if_specified(accelerator, args)\n"""
    if has_qwen_patch:
        qwen_marker_anchor = """    # Save-state compatibility marker. Frozen-Qwen jobs keep their old\n"""
        insert = """    if parameter_policy_session is not None:\n        parameter_policy_session.finalize_after_prepare(\n            accelerator=accelerator, optimizer=optimizer, scheduler=lr_scheduler\n        )\n\n"""
        if qwen_marker_anchor not in text:
            raise RuntimeError("Anima Full Qwen resume marker missing")
        text = text.replace(qwen_marker_anchor, insert + qwen_marker_anchor, 1)
    else:
        text = replace_once(
            text,
            base_resume_anchor,
            """    if parameter_policy_session is not None:\n        parameter_policy_session.finalize_after_prepare(\n            accelerator=accelerator, optimizer=optimizer, scheduler=lr_scheduler\n        )\n\n""" + base_resume_anchor,
            "Anima Full manifest before resume",
        )

    text = replace_once(
        text,
        base_resume_anchor,
        base_resume_anchor
        + """    if parameter_policy_session is not None:\n        parameter_policy_session.assert_runtime_contract(\n            phase=\"post_resume\", accelerator=accelerator, optimizer=optimizer\n        )\n""",
        "Anima Full post-resume runtime contract",
    )

    text = replace_once(
        text,
        """                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:\n                        params_to_clip = []\n                        for m in training_models:\n                            params_to_clip.extend(m.parameters())\n                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)\n""",
        """                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:\n                        if parameter_policy_session is not None:\n                            params_to_clip = parameter_policy_session.trainable_parameters\n                        else:\n                            params_to_clip = []\n                            for m in training_models:\n                                params_to_clip.extend(m.parameters())\n                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)\n""",
        "Anima Full component clipping",
    )

    # Logging differs after the Qwen patch only by lr_names; replace the entire call.
    old_log = """                optimizer_util.append_lr_to_logs_with_names(\n                    logs,\n                    lr_scheduler,\n                    args.optimizer_type,\n"""
    if old_log not in text:
        raise RuntimeError("Anima Full LR logging anchor missing")
    idx = text.index(old_log)
    call_end = text.index("""                accelerator.log(logs, step=global_step)\n""", idx)
    old_call = text[idx:call_end]
    new_call = """                if parameter_policy_session is not None:\n                    logs.update(parameter_policy_session.component_lr_logs(lr_scheduler))\n                else:\n""" + "".join("    " + line if line.strip() else line for line in old_call.splitlines(True))
    text = text[:idx] + new_call + text[call_end:]

    text = replace_once(
        text,
        """        for m in training_models:\n            m.train()\n\n        for step, batch in enumerate(train_dataloader):\n""",
        """        for m in training_models:\n            m.train()\n        if parameter_policy_session is not None:\n            parameter_policy_session.assert_runtime_contract(\n                phase=\"epoch_start\", accelerator=accelerator, optimizer=optimizer\n            )\n\n        for step, batch in enumerate(train_dataloader):\n""",
        "Anima Full epoch ownership assertion",
    )

    text = replace_once(
        text,
        """\n    # End training\n""",
        """\n        if parameter_policy_session is not None:\n            optimizer_train_fn()\n\n    # End training\n""",
        "Anima Full component epoch optimizer lifecycle restoration",
    )

    text = replace_once(
        text,
        """    anima_train_utils.add_anima_training_arguments(parser)\n""",
        """    anima_train_utils.add_anima_training_arguments(parser)\n    parser.add_argument(\"--parameter_policy_config\", type=str, default=None)\n""",
        "Anima Full policy CLI argument",
    )
    return text


def patch_files(sd_scripts_dir: Path) -> dict[Path, str]:
    targets = {
        sd_scripts_dir / "train_network.py": patch_train_network,
        sd_scripts_dir / "anima_train_network.py": patch_anima_train_network,
        sd_scripts_dir / "anima_train.py": patch_anima_train,
    }
    patched: dict[Path, str] = {}
    for path, patcher in targets.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        source = path.read_text(encoding="utf-8-sig")
        result = patcher(source)
        ast.parse(result, filename=str(path))
        patched[path] = result

    bridge_source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "dev"
        / "library"
        / "dts_parameter_policy_bridge.py"
    ).read_text(encoding="utf-8")
    bridge_path = sd_scripts_dir / "library" / "dts_parameter_policy_bridge.py"
    ast.parse(bridge_source, filename=str(bridge_path))
    patched[bridge_path] = bridge_source
    return patched


def validate_patch(sd_scripts_dir: Path) -> None:
    patched = patch_files(sd_scripts_dir)
    required = {
        sd_scripts_dir / "train_network.py": (
            "parameter_policy_session",
            "component_lr_logs",
            "register_checkpoint_manifest",
        ),
        sd_scripts_dir / "anima_train_network.py": (
            'return "anima-lora"',
            "qwen3.adapter",
            "--parameter_policy_config",
        ),
        sd_scripts_dir / "anima_train.py": (
            '"anima-finetune"',
            "parameter_policy_session",
            "--parameter_policy_config",
        ),
        sd_scripts_dir / "library" / "dts_parameter_policy_bridge.py": (
            "load_parameter_policy_file",
            "create_parameter_policy_session",
        ),
    }
    for path, markers in required.items():
        source = patched[path]
        for marker in markers:
            if marker not in source:
                raise RuntimeError(f"Patched {path} is missing required marker: {marker}")


__all__ = [
    "EXPECTED_SD_SCRIPTS_HEAD",
    "patch_files",
    "validate_patch",
]
