from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import unittest

from mikazuki.model_component_profiles import (
    get_model_component_profile,
    resolve_training_target_profile,
)
from mikazuki.multi_caption_config import build_multi_caption_sidecar
from mikazuki.optimizer_profiles import list_optimizer_capabilities
from mikazuki.parameter_policy import (
    build_parameter_policy_sidecar,
    parameter_policy_runtime_blockers,
    validate_parameter_policy,
)
from mikazuki.parameter_policy_compat import parameter_policy_v1_semantic_blockers
from mikazuki.parameter_policy_editor import (
    bootstrap_parameter_policy_editor,
    normalize_parameter_policy_editor_state,
    parameter_policy_editor_metadata,
    parameter_policy_editor_preview,
)
from mikazuki.parameter_policy_matrix import (
    PARAMETER_POLICY_QUALIFICATION_BLOCKER_FIELDS,
    PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
    PARAMETER_POLICY_SEMANTIC_BLOCKER_FEATURES,
    parameter_policy_gpu_selection_blockers,
)
from mikazuki.training_config import PAGE_BACKEND_MAP, prepare_training_config
from mikazuki.training_rehydrate import rehydrate_trainer_config
from mikazuki.training_validation import validate_prepared_config
from mikazuki.utils import train_utils


@dataclass(frozen=True)
class ReleaseCase:
    page_type: str
    backend: str
    standard_raw: Mapping[str, object]


def _frozen(values: dict[str, object]) -> Mapping[str, object]:
    return MappingProxyType(values)


RELEASE_CASES = (
    ReleaseCase(
        "lora-master",
        "sd-lora",
        _frozen(
            {
                "optimizer_type": "AdamW",
                "learning_rate": "1e-4",
                "lora_target": "unet_text_encoder",
                "memory_mode": "auto",
            }
        ),
    ),
    ReleaseCase(
        "sdxl-lora",
        "sdxl-lora",
        _frozen(
            {
                "optimizer_type": "AdamW",
                "learning_rate": "1e-4",
                "lora_target": "unet_text_encoder",
                "memory_mode": "auto",
            }
        ),
    ),
    ReleaseCase(
        "dreambooth",
        "sd-dreambooth",
        _frozen(
            {
                "optimizer_type": "AdamW",
                "learning_rate": "1e-6",
                "memory_mode": "auto",
            }
        ),
    ),
    ReleaseCase(
        "sdxl-full",
        "sdxl-finetune",
        _frozen(
            {
                "optimizer_type": "AdamW",
                "learning_rate": "1e-6",
                "train_text_encoder": True,
                "mixed_precision": "bf16",
                "memory_mode": "auto",
            }
        ),
    ),
    ReleaseCase(
        "sd3-lora",
        "sd3-lora",
        _frozen(
            {
                "optimizer_type": "AdamW",
                "learning_rate": "1e-4",
                "lowram": False,
            }
        ),
    ),
    ReleaseCase(
        "flux-lora",
        "flux-lora",
        _frozen(
            {
                "optimizer_type": "AdamW",
                "learning_rate": "1e-4",
                "flux_lora_target": "dit",
                "memory_mode": "auto",
            }
        ),
    ),
    ReleaseCase(
        "chroma-lora",
        "chroma-lora",
        _frozen(
            {
                "optimizer_type": "AdamW",
                "learning_rate": "1e-4",
                "flux_lora_target": "dit",
                "memory_mode": "auto",
            }
        ),
    ),
    ReleaseCase(
        "flux-finetune",
        "flux-finetune",
        _frozen(
            {
                "optimizer_type": "AdamW",
                "learning_rate": "1e-6",
                "mixed_precision": "bf16",
                "blocks_to_swap": 0,
                "memory_mode": "auto",
            }
        ),
    ),
    ReleaseCase(
        "anima-lora",
        "anima-lora",
        _frozen(
            {
                "anima_model_variant": "base",
                "network_module": "networks.lora_anima",
                "optimizer_type": "AdamW",
                "learning_rate": "1e-4",
                "anima_lora_target": "dit",
                "mixed_precision": "bf16",
                "cache_latents": True,
                "cache_latents_to_disk": True,
                "cache_text_encoder_outputs": True,
                "cache_text_encoder_outputs_to_disk": True,
                "anima_lora_checkpoint_mode": "standard",
                "anima_lora_compile_mode": "off",
                "blocks_to_swap": 0,
                "memory_mode": "auto",
            }
        ),
    ),
    ReleaseCase(
        "anima-finetune",
        "anima-finetune",
        _frozen(
            {
                "anima_model_variant": "base",
                "optimizer_type": "AdamW",
                "anima_finetune_learning_rate": "1e-5",
                "lr_scheduler": "constant",
                "anima_precision_mode": "mixed_bf16",
                "anima_latent_cache_mode": "disk",
                "anima_text_encoder_cache_mode": "disk",
                "anima_checkpoint_mode": "standard",
                "blocks_to_swap": 0,
                "memory_mode": "auto",
                "train_qwen3_text_encoder": False,
            }
        ),
    ),
)

def _resolve_backend(config: dict, requested: str):
    del config
    return requested, f"./{requested}.py"


def _prepare_policy_request(raw: Mapping[str, object], page_type: str):
    config = deepcopy(dict(raw))
    train_utils.fix_config_types(config)
    normalize_parameter_policy_editor_state(config)
    policy_path, sidecars, policy = build_parameter_policy_sidecar(
        config,
        page_type,
    )
    multi_path, multi_sidecars, multi_policy = build_multi_caption_sidecar(
        config,
        page_type,
    )
    if multi_path is not None or multi_policy is not None:
        raise AssertionError("Step 7 release fixtures must remain caption_mode=standard.")
    prepared = prepare_training_config(
        config,
        page_train_type=page_type,
        resolve_backend=_resolve_backend,
        launch=False,
    )
    prepared.sidecars.update(sidecars)
    prepared.sidecars.update(multi_sidecars)

    effective_policy_path = prepared.config.get("parameter_policy_config")
    if policy_path is None:
        if effective_policy_path not in (None, ""):
            raise AssertionError(
                "Standard request unexpectedly retained parameter_policy_config."
            )
    elif effective_policy_path != policy_path:
        raise AssertionError(
            "Prepared Component request changed the host-owned policy path."
        )

    if policy is not None:
        prepared.runtime_blockers.extend(
            parameter_policy_runtime_blockers(
                policy,
                train_type=prepared.train_type,
                effective_config=prepared.config,
                integrated_train_types=PARAMETER_POLICY_RUNTIME_TRAIN_TYPES,
            )
        )
        prepared.runtime_blockers.extend(
            parameter_policy_gpu_selection_blockers(prepared.gpu_ids)
        )

    validate_prepared_config(prepared, False)
    return prepared, policy


def _project_rehydrated_gui_for_page(
    gui: Mapping[str, object],
    page_type: str,
) -> dict:
    """Mirror the one legacy browser schema projection not modeled in Python."""

    projected = deepcopy(dict(gui))
    if page_type == "sd3-lora":
        mode = projected.pop("memory_mode", "auto")
        if mode != "auto":
            raise AssertionError(
                f"SD3 rehydrate unexpectedly produced memory_mode={mode!r}."
            )
        projected["lowram"] = False
    return projected


def _stale_standard_state(base: Mapping[str, object]) -> dict:
    return {
        **deepcopy(dict(base)),
        "optimization_mode": "standard",
        "parameter_policy_profiles": {
            "stale": {
                "type": "NotARealOptimizer",
                "args": {"broken": "["},
            }
        },
        "parameter_policy_components": {
            "stale.component": {
                "train": True,
                "optimizer_profile": "missing",
                "learning_rate": "not-a-number",
            }
        },
    }


class ParameterPolicyStep7ReleaseMatrixTests(unittest.TestCase):
    def test_release_case_matrix_is_exact_and_page_aliases_resolve(self):
        backends = {case.backend for case in RELEASE_CASES}
        self.assertEqual(backends, PARAMETER_POLICY_RUNTIME_TRAIN_TYPES)
        self.assertEqual(len(RELEASE_CASES), len(PARAMETER_POLICY_RUNTIME_TRAIN_TYPES))
        self.assertEqual(len({case.page_type for case in RELEASE_CASES}), len(RELEASE_CASES))
        for case in RELEASE_CASES:
            with self.subTest(page_type=case.page_type):
                self.assertEqual(
                    PAGE_BACKEND_MAP.get(case.page_type, case.page_type),
                    case.backend,
                )

    def test_standard_missing_explicit_and_stale_component_states_are_identical(self):
        for case in RELEASE_CASES:
            with self.subTest(backend=case.backend):
                raw_missing = deepcopy(dict(case.standard_raw))
                raw_explicit = {
                    **deepcopy(dict(case.standard_raw)),
                    "optimization_mode": "standard",
                }
                raw_stale = _stale_standard_state(case.standard_raw)

                prepared_missing, policy_missing = _prepare_policy_request(
                    raw_missing,
                    case.page_type,
                )
                prepared_explicit, policy_explicit = _prepare_policy_request(
                    raw_explicit,
                    case.page_type,
                )
                prepared_stale, policy_stale = _prepare_policy_request(
                    raw_stale,
                    case.page_type,
                )

                self.assertIsNone(policy_missing)
                self.assertIsNone(policy_explicit)
                self.assertIsNone(policy_stale)

                for prepared in (
                    prepared_missing,
                    prepared_explicit,
                    prepared_stale,
                ):
                    self.assertEqual(prepared.train_type, case.backend)
                    self.assertEqual(prepared.trainer_file, f"./{case.backend}.py")
                    self.assertEqual(prepared.sidecars, {})
                    self.assertEqual(prepared.runtime_blockers, [])
                    self.assertNotIn("parameter_policy_config", prepared.config)

                self.assertEqual(prepared_missing.config, prepared_explicit.config)
                self.assertEqual(prepared_missing.config, prepared_stale.config)
                self.assertEqual(prepared_missing.warnings, prepared_explicit.warnings)
                self.assertEqual(prepared_missing.warnings, prepared_stale.warnings)
                self.assertEqual(prepared_missing.gpu_ids, prepared_explicit.gpu_ids)
                self.assertEqual(prepared_missing.gpu_ids, prepared_stale.gpu_ids)

    def test_component_bootstrap_preview_rehydrate_round_trip_is_stable(self):
        for case in RELEASE_CASES:
            with self.subTest(backend=case.backend):
                source = deepcopy(dict(case.standard_raw))
                before = deepcopy(source)
                editor_state = bootstrap_parameter_policy_editor(
                    source,
                    case.page_type,
                    resolve_backend=_resolve_backend,
                )
                self.assertEqual(source, before)

                component_gui = deepcopy(source)
                component_gui.update(editor_state)
                prepared_a, policy_a = _prepare_policy_request(
                    component_gui,
                    case.page_type,
                )

                self.assertIsNotNone(policy_a)
                canonical_a = validate_parameter_policy(policy_a)
                self.assertEqual(prepared_a.train_type, case.backend)
                self.assertEqual(prepared_a.runtime_blockers, [])
                self.assertEqual(len(prepared_a.sidecars), 1)
                policy_path_a = prepared_a.config.get("parameter_policy_config")
                self.assertIsInstance(policy_path_a, str)
                self.assertIn(policy_path_a, prepared_a.sidecars)

                target_a = resolve_training_target_profile(
                    prepared_a.train_type,
                    prepared_a.config,
                )
                for component_id, route in canonical_a["components"].items():
                    if route.get("train"):
                        self.assertTrue(
                            target_a.is_available(component_id),
                            (
                                f"{case.backend}: {component_id} is Train=true "
                                "but unavailable before rehydrate"
                            ),
                        )

                preview_a = parameter_policy_editor_preview(
                    canonical_a,
                    case.page_type,
                    prepared_a.runtime_blockers,
                )
                self.assertTrue(preview_a["runtime_ready"])

                rehydrated = rehydrate_trainer_config(
                    deepcopy(prepared_a.config),
                    case.page_type,
                    sidecars=deepcopy(prepared_a.sidecars),
                )
                rehydrated = _project_rehydrated_gui_for_page(
                    rehydrated,
                    case.page_type,
                )
                self.assertEqual(rehydrated["optimization_mode"], "component")
                self.assertTrue(rehydrated["parameter_policy_profiles"])
                self.assertTrue(rehydrated["parameter_policy_components"])

                prepared_b, policy_b = _prepare_policy_request(
                    rehydrated,
                    case.page_type,
                )
                self.assertIsNotNone(policy_b)
                canonical_b = validate_parameter_policy(policy_b)

                policy_path_b = prepared_b.config.get("parameter_policy_config")
                self.assertEqual(canonical_a, canonical_b)
                self.assertEqual(policy_path_a, policy_path_b)
                self.assertEqual(
                    prepared_a.sidecars[policy_path_a],
                    prepared_b.sidecars[policy_path_b],
                )
                self.assertEqual(prepared_a.config, prepared_b.config)
                self.assertEqual(prepared_a.warnings, prepared_b.warnings)
                self.assertEqual(prepared_a.runtime_blockers, prepared_b.runtime_blockers)

                target_b = resolve_training_target_profile(
                    prepared_b.train_type,
                    prepared_b.config,
                )
                for component_id, route in canonical_b["components"].items():
                    if route.get("train"):
                        self.assertTrue(
                            target_b.is_available(component_id),
                            (
                                f"{case.backend}: {component_id} is Train=true "
                                "but unavailable after rehydrate"
                            ),
                        )
                self.assertEqual(
                    target_a.available_components,
                    target_b.available_components,
                )
                self.assertEqual(
                    dict(target_a.unavailable_reasons),
                    dict(target_b.unavailable_reasons),
                )

                preview_b = parameter_policy_editor_preview(
                    canonical_b,
                    case.page_type,
                    prepared_b.runtime_blockers,
                )
                self.assertEqual(preview_a, preview_b)


class ParameterPolicyStep7RegistryClosureTests(unittest.TestCase):
    def test_editor_metadata_closes_over_component_and_optimizer_registries(self):
        supported_optimizers = {
            capability.name
            for capability in list_optimizer_capabilities()
            if capability.component_support == "supported"
        }
        for case in RELEASE_CASES:
            with self.subTest(backend=case.backend):
                metadata = parameter_policy_editor_metadata(case.page_type)
                self.assertEqual(metadata["train_type"], case.backend)
                self.assertEqual(
                    {row["id"] for row in metadata["components"]},
                    set(get_model_component_profile(case.backend).components),
                )
                self.assertEqual(
                    {row["type"] for row in metadata["optimizer_types"]},
                    supported_optimizers,
                )

    def test_step6f_qualification_blocker_field_set_is_frozen(self):
        expected = frozenset(
            {
                "torch_compile",
                "compile",
                "deepspeed",
                "fused_backward_pass",
                "fused_optimizer_groups",
                "blockwise_fused_optimizers",
                "cpu_offload_checkpointing",
                "unsloth_offload_checkpointing",
                "blocks_to_swap",
                "double_blocks_to_swap",
                "single_blocks_to_swap",
                "full_fp16",
                "full_bf16",
                "fp8_base",
                "fp8_base_unet",
            }
        )
        self.assertEqual(PARAMETER_POLICY_QUALIFICATION_BLOCKER_FIELDS, expected)

        active_values = {
            "torch_compile": True,
            "compile": True,
            "deepspeed": True,
            "fused_backward_pass": True,
            "fused_optimizer_groups": 2,
            "blockwise_fused_optimizers": True,
            "cpu_offload_checkpointing": True,
            "unsloth_offload_checkpointing": True,
            "blocks_to_swap": 1,
            "double_blocks_to_swap": 1,
            "single_blocks_to_swap": 1,
            "full_fp16": True,
            "full_bf16": True,
            "fp8_base": True,
            "fp8_base_unet": True,
        }
        for field in sorted(expected):
            with self.subTest(field=field):
                train_type = "anima-finetune" if field == "compile" else "flux-finetune"
                blockers = parameter_policy_v1_semantic_blockers(
                    {field: active_values[field]},
                    train_type,
                )
                self.assertTrue(
                    blockers,
                    f"{field} stopped producing a Step 6F qualification blocker",
                )

    def test_semantic_blocker_taxonomy_is_frozen(self):
        self.assertEqual(
            PARAMETER_POLICY_SEMANTIC_BLOCKER_FEATURES,
            frozenset(
                {
                    "lora_plus",
                    "regex_lr",
                    "sd_lora_block_lr",
                    "sdxl_full_block_lr",
                    "scale_weight_norms",
                    "dreambooth_dynamic_text_encoder_stop",
                    "preloaded_adapter_text_encoder_cache",
                    "anima_qwen_only",
                    "unreviewed_network_module",
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
