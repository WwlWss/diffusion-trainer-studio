from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.parameter_policy_backend_feature_gpu_support import (
    BACKEND_FEATURE_EVIDENCE_SCHEMA,
    BACKEND_FEATURE_EVIDENCE_VERSION,
    BACKEND_FEATURE_MANIFEST_SCHEMA,
    CHECKPOINT_PROGRESS_SCHEMA,
    SD_LORA_FULL_BF16_CASE_IDS,
    SD_LORA_FULL_BF16_EVIDENCE_ID,
    SDXL_LORA_FULL_BF16_CASE_IDS,
    SDXL_LORA_FULL_BF16_EVIDENCE_ID,
    SHARED_FULL_BF16_OPTIMIZER_EVIDENCE_ID,
    BackendFeatureGpuMatrixError,
    backend_feature_qualification_snapshot,
    compare_backend_checkpoint_contracts,
    compare_checkpoint_progress,
    load_backend_feature_manifest,
    summarize_sd_lora_full_bf16_promotion,
    summarize_sdxl_lora_full_bf16_promotion,
    validate_case_input_contract,
    validate_checkpoint_progress,
    validate_full_bf16_checkpoint_manifest,
    validate_shared_full_bf16_regression_evidence,
)
from tools.parameter_policy_execution_gpu_support import (
    EXECUTION_GPU_CASE_PHASES,
    EXECUTION_GPU_EVIDENCE_SCHEMA,
    EXECUTION_GPU_EVIDENCE_VERSION,
    EXECUTION_INFRA_CASE_IDS,
    SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID,
)



ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools" / "run_parameter_policy_backend_feature_gpu_matrix.py"
WORKFLOW = ROOT / ".github" / "workflows" / "parameter-policy-gpu-matrix.yml"


def _case(root: Path) -> dict:
    return {
        "case_id": "backend:flux-finetune:full-bf16:reference:v1",
        "train_type": "flux-finetune",
        "feature": "full_bf16",
        "fresh_command": [
            
            "scripts/dev/flux_train.py",
            "--max_train_steps",
            "2",
            "--save_every_n_steps",
            "1",
            "--save_state",
        ],
        "resume_command": [
            
            "scripts/dev/flux_train.py",
            "--resume",
            str(root / "checkpoint-1"),
            "--max_train_steps",
            "2",
            "--save_every_n_steps",
            "1",
            "--save_state",
        ],
        "cwd": str(ROOT),
        "environment": {"DTS_TEST": "1"},
        "fresh_checkpoint_dir": str(root / "checkpoint-1"),
        "resume_checkpoint_dir": str(root / "checkpoint-2"),
    }


def _sd_lora_policy(kind: str) -> dict:
    if kind == "adamw":
        profiles = {
            "main": {"type": "AdamW", "args": {}},
        }
        train_route = {
            "train": True,
            "optimizer_profile": "main",
            "learning_rate": 1e-4,
        }
        components = {
            "unet.attention.adapter": dict(train_route),
            "unet.feed_forward.adapter": dict(train_route),
            "unet.conv.adapter": dict(train_route),
            "unet.other.adapter": {"train": False},
            "text_encoder.adapter": dict(train_route),
        }
    elif kind == "muon_adamw_fallback":
        profiles = {
            "muon": {"type": "Muon", "args": {}},
            "fallback": {"type": "AdamW", "args": {}},
        }
        train_route = {
            "train": True,
            "optimizer_profile": "muon",
            "learning_rate": 1e-3,
            "fallback_optimizer_profile": "fallback",
            "fallback_learning_rate": 1e-4,
        }
        components = {
            "unet.attention.adapter": dict(train_route),
            "unet.feed_forward.adapter": dict(train_route),
            "unet.conv.adapter": dict(train_route),
            "unet.other.adapter": {"train": False},
            "text_encoder.adapter": dict(train_route),
        }
    else:
        raise AssertionError(f"unknown SD LoRA D1 policy kind: {kind}")
    return {
        "version": 1,
        "optimizer_profiles": profiles,
        "components": components,
    }


def _sd_lora_case(root: Path, case_id: str) -> dict:
    kind = (
        "adamw"
        if case_id == SD_LORA_FULL_BF16_CASE_IDS[0]
        else "muon_adamw_fallback"
    )
    policy_path = root / f"{kind}-policy.json"
    policy_path.write_text(
        json.dumps(_sd_lora_policy(kind)),
        encoding="utf-8",
    )
    base_model = root / "sd1-base.safetensors"
    if not base_model.exists():
        base_model.write_bytes(b"sd1-base-model")
    fresh_output = root / f"{kind}-fresh-output"
    resume_output = root / f"{kind}-resume-output"
    fresh_checkpoint = fresh_output / "at-step00000001-state"
    resume_checkpoint = resume_output / "at-step00000002-state"
    accumulation = (
        "1"
        if case_id == SD_LORA_FULL_BF16_CASE_IDS[0]
        else "2"
    )
    common = [
        "scripts/stable/train_network.py",
        "--parameter_policy_config",
        str(policy_path),
        "--pretrained_model_name_or_path",
        str(base_model),
        "--network_module",
        "networks.lora",
        "--gradient_accumulation_steps",
        accumulation,
        "--mixed_precision",
        "bf16",
        "--full_bf16",
        *(
            ["--network_args", "conv_dim=4"]
            if case_id == SD_LORA_FULL_BF16_CASE_IDS[1]
            else []
        ),
        "--max_train_steps",
        "2",
        "--save_every_n_steps",
        "1",
        "--save_state",
    ]
    return {
        "case_id": case_id,
        "train_type": "sd-lora",
        "feature": "full_bf16",
        "fresh_command": [
            *common,
            "--output_dir",
            str(fresh_output),
        ],
        "resume_command": [
            "scripts/stable/train_network.py",
            "--parameter_policy_config",
            str(policy_path),
            "--pretrained_model_name_or_path",
            str(base_model),
            "--network_module",
            "networks.lora",
            "--gradient_accumulation_steps",
            accumulation,
            "--mixed_precision",
            "bf16",
            "--full_bf16",
            *(
                ["--network_args", "conv_dim=4"]
                if case_id == SD_LORA_FULL_BF16_CASE_IDS[1]
                else []
            ),
            "--resume",
            str(fresh_checkpoint),
            "--max_train_steps",
            "2",
            "--save_every_n_steps",
            "1",
            "--save_state",
            "--output_dir",
            str(resume_output),
        ],
        "cwd": str(ROOT),
        "environment": {},
        "fresh_checkpoint_dir": str(fresh_checkpoint),
        "resume_checkpoint_dir": str(resume_checkpoint),
    }


def _sdxl_lora_case(root: Path, case_id: str) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    source_case = (
        SD_LORA_FULL_BF16_CASE_IDS[0]
        if case_id == SDXL_LORA_FULL_BF16_CASE_IDS[0]
        else SD_LORA_FULL_BF16_CASE_IDS[1]
    )
    case = _sd_lora_case(root, source_case)
    case["case_id"] = case_id
    case["train_type"] = "sdxl-lora"
    base_model = root / "sdxl-base.safetensors"
    base_model.write_bytes(b"sdxl-base-model-fixture")
    for key in ("fresh_command", "resume_command"):
        command = case[key]
        command[0] = "scripts/stable/sdxl_train_network.py"
        model_idx = command.index("--pretrained_model_name_or_path") + 1
        command[model_idx] = str(base_model)
    policy_path = Path(
        case["fresh_command"][
            case["fresh_command"].index("--parameter_policy_config") + 1
        ]
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    old = policy["components"].pop("text_encoder.adapter")
    policy["components"]["text_encoder_1.adapter"] = dict(old)
    policy["components"]["text_encoder_2.adapter"] = dict(old)
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return case


def _sdxl_d1_case_row(case_id: str) -> dict:
    source_case = (
        SD_LORA_FULL_BF16_CASE_IDS[0]
        if case_id == SDXL_LORA_FULL_BF16_CASE_IDS[0]
        else SD_LORA_FULL_BF16_CASE_IDS[1]
    )
    row = _d1_case_row(source_case, train_type="sdxl-lora")
    row["case_id"] = case_id
    row["policy_contract"]["model_family"] = "sdxl-base"
    row["policy_contract"]["covers_text_encoder_adapter"] = None
    row["policy_contract"]["covered_text_components"] = [
        "text_encoder_1.adapter",
        "text_encoder_2.adapter",
    ]
    row["input_contract"]["model_family"] = "sdxl-base"
    row["component_update_evidence"] = {
        "model_family": "sdxl-base",
        "weight_keys_match": True,
        "changed_tensor_counts": {
            "unet": 10,
            "te1": 4,
            "te2": 8,
            "conv3x3": 2 if case_id == SDXL_LORA_FULL_BF16_CASE_IDS[1] else 0,
        },
    }
    return row


def _signature(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _progress(
    *,
    train_type: str = "flux-finetune",
    policy_hash: str = "policy",
    topology: str = "topology",
    execution_signature: str,
    optimizer_step: int,
    scheduler_step_count: int,
    scheduler_last_epoch: int,
    checkpoint_id: str = "checkpoint",
    resume_source_checkpoint_id: str | None = None,
) -> dict:
    return {
        "schema": CHECKPOINT_PROGRESS_SCHEMA,
        "version": 1,
        "train_type": train_type,
        "policy_hash": policy_hash,
        "runtime_topology_fingerprint": topology,
        "execution_signature": execution_signature,
        "checkpoint_id": checkpoint_id,
        "resume_source_checkpoint_id": resume_source_checkpoint_id,
        "optimizer_profiles": {
            "main": {
                "optimizer_type": "AdamW",
                "step_values": [optimizer_step],
            }
        },
        "scheduler": {
            "step_count": scheduler_step_count,
            "last_epoch": scheduler_last_epoch,
        },
    }


def _d1_case_row(
    case_id: str,
    *,
    status: str = "pass",
    train_type: str = "sd-lora",
    feature: str = "full_bf16",
    optimizer_types: tuple[str, ...] | None = None,
) -> dict:
    if optimizer_types is None:
        if case_id == SD_LORA_FULL_BF16_CASE_IDS[0]:
            optimizer_types = ("AdamW",)
        elif case_id == SD_LORA_FULL_BF16_CASE_IDS[1]:
            optimizer_types = ("Muon", "AdamW")
        else:
            optimizer_types = ("AdamW",)

    profiles = {
        f"profile_{index}": optimizer_type
        for index, optimizer_type in enumerate(optimizer_types)
    }
    manifest = {
        "policy_hash": "policy",
        "optimizer_profiles": profiles,
        "optimizers": [
            {
                "profile_name": profile_name,
                "optimizer_type": optimizer_type,
            }
            for profile_name, optimizer_type in profiles.items()
        ],
    }
    policy_kind = (
        "adamw"
        if case_id == SD_LORA_FULL_BF16_CASE_IDS[0]
        else "muon_adamw_fallback"
    )
    return {
        "case_id": case_id,
        "train_type": train_type,
        "feature": feature,
        "status": status,
        "policy_contract": {
            "policy_hash": "policy",
            "model_family": "sd1",
            "v2": False,
            "kind": policy_kind,
            "gradient_accumulation_steps": (
                1
                if case_id == SD_LORA_FULL_BF16_CASE_IDS[0]
                else 2
            ),
            "mixed_precision": "bf16",
            "full_bf16": True,
            "network_module": "networks.lora",
            "covers_text_encoder_adapter": True,
            "covered_text_components": ["text_encoder.adapter"],
            "conv_dim": (
                None
                if case_id == SD_LORA_FULL_BF16_CASE_IDS[0]
                else 4
            ),
            "fallback_component": (
                None
                if policy_kind == "adamw"
                else "unet.conv.adapter"
            ),
        },
        "input_contract": {
            "model_family": "sd1",
            "v2": False,
            "signature": "input-signature",
            "output_contract": {
                "fresh_output_dir": "fresh",
                "resume_output_dir": "resume",
            },
            "base_model": {"sha256": "base-model"},
            "parameter_policy": {"sha256": "policy-file"},
        },
        "pre_fresh_input_contract_valid": True,
        "pre_resume_input_contract_valid": True,
        "fresh_checkpoint_manifest": copy.deepcopy(manifest),
        "resume_checkpoint_manifest": copy.deepcopy(manifest),
    }


def _qualification_contract() -> dict:
    pinned = {
        "accelerate": "1.6.0",
        "diffusers": "0.32.1",
        "pytorch-optimizer": "3.10.0",
        "transformers": "4.54.1",
    }
    return {
        "schema": "dts.parameter-policy.gpu-qualification-environment",
        "version": 2,
        "status": "pass",
        "python_major_minor": "3.11",
        "torch_base_version": "2.7.0",
        "torch_full_version": "2.7.0+cu128",
        "torchvision_version": "0.22.0",
        "torchvision_full_version": "0.22.0+cu128",
        "pytorch_optimizer_version": "3.10.0",
        "requirements_sha256": "requirements",
        "requirements_exact_pins": dict(pinned),
        "installed_requirement_versions": dict(pinned),
        "cuda_available": True,
        "bf16_supported": True,
    }


def _shared_regression_contract(commit: str = "exact-head") -> dict:
    return {
        "id": SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID,
        "status": "pass",
        "commit": commit,
        "qualification_contract": _qualification_contract(),
        "required_cases": list(EXECUTION_GPU_CASE_PHASES),
    }


def _shared_regression_evidence(
    *,
    commit: str = "exact-head",
    qualification_contract: dict | None = None,
) -> dict:
    contract = qualification_contract or _qualification_contract()
    cases = [
        {
            "case_id": case_id,
            "status": "pass",
            "phases": [
                {"phase": phase, "status": "pass"}
                for phase in EXECUTION_GPU_CASE_PHASES[case_id]
            ],
        }
        for case_id in EXECUTION_GPU_CASE_PHASES
    ]
    return {
        "schema": EXECUTION_GPU_EVIDENCE_SCHEMA,
        "version": EXECUTION_GPU_EVIDENCE_VERSION,
        "commit": commit,
        "expected_commit": commit,
        "final_provenance_commit": commit,
        "qualification_mode": "regression",
        "qualification_contract": contract,
        "cases": cases,
        "shared_optimizer_regression": {
            "id": SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID,
            "scope": "shared_optimizer_regression",
            "status": "pass",
            "target_rows_match": True,
            "infra_complete": all(
                row["status"] == "pass"
                for row in cases
                if row["case_id"] in EXECUTION_INFRA_CASE_IDS
            ),
            "required_cases": list(EXECUTION_GPU_CASE_PHASES),
            "missing_cases": [],
            "failed_cases": [],
        },
    }


def _checkpoint(train_type: str = "flux-finetune") -> dict:
    scheduler_identity = {
        "schema": "dts.parameter-policy.scheduler-identity",
        "version": 1,
        "provider": "sd-scripts.get_scheduler_fix",
    }
    execution_identity = {
        "schema": "dts.parameter-policy.execution-identity",
        "version": 1,
        "train_type": train_type,
        "features": {
            "full_bf16": {
                "mixed_precision": "bf16",
                "trainable_parameter_dtype": "bfloat16",
            }
        },
    }
    return {
        "version": 2,
        "train_type": train_type,
        "policy_hash": "policy",
        "runtime_topology_fingerprint": "topology",
        "trainable_components": ["transformer.double_stream"],
        "frozen_components": ["transformer.single_stream"],
        "trainable_parameter_tensors": 2,
        "trainable_parameter_elements": 96,
        "optimizer_profiles": {"main": "AdamW"},
        "optimizers": [
            {
                "profile_name": "main",
                "optimizer_type": "AdamW",
                "topology_fingerprint": "optimizer",
            }
        ],
        "scheduler_identity": scheduler_identity,
        "scheduler_signature": _signature(scheduler_identity),
        "schedulers": [
            {
                "profile_name": "main",
                "mode": "external",
                "scheduler_type": "constant",
                "topology_fingerprint": "scheduler-topology",
            }
        ],
        "execution_identity": execution_identity,
        "execution_signature": _signature(execution_identity),
    }


class BackendFeatureManifestTests(unittest.TestCase):
    def _write(self, temp: Path, cases: list[dict], **overrides) -> Path:
        payload = {
            "schema": BACKEND_FEATURE_MANIFEST_SCHEMA,
            "version": 1,
            "cases": cases,
        }
        payload.update(overrides)
        path = temp / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_valid_partial_manifest_is_allowed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cases = load_backend_feature_manifest(
                self._write(temp, [_case(temp)]),
                repo_root=ROOT,
            )
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["train_type"], "flux-finetune")
        self.assertEqual(cases[0]["feature"], "full_bf16")
        self.assertEqual(cases[0]["entrypoint"], "scripts/dev/flux_train.py")
        self.assertEqual(cases[0]["fresh_command"][0], "scripts/dev/flux_train.py")

    def test_d1_sd_lora_manifest_validates_explicit_policy_contracts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cases = [
                _sd_lora_case(temp, case_id)
                for case_id in SD_LORA_FULL_BF16_CASE_IDS
            ]
            loaded = load_backend_feature_manifest(
                self._write(temp, cases),
                repo_root=ROOT,
            )
        self.assertEqual(
            [case["policy_contract"]["kind"] for case in loaded],
            ["adamw", "muon_adamw_fallback"],
        )
        self.assertIsNone(loaded[0]["policy_contract"]["fallback_component"])
        self.assertEqual(
            loaded[1]["policy_contract"]["fallback_component"],
            "unet.conv.adapter",
        )
        self.assertTrue(loaded[0]["policy_contract"]["policy_hash"])
        self.assertTrue(loaded[1]["policy_contract"]["policy_hash"])
        self.assertEqual(
            [case["policy_contract"]["gradient_accumulation_steps"] for case in loaded],
            [1, 2],
        )
        self.assertTrue(
            all(case["policy_contract"]["full_bf16"] for case in loaded)
        )
        self.assertTrue(
            all(
                case["policy_contract"]["covers_text_encoder_adapter"]
                for case in loaded
            )
        )
        self.assertEqual(loaded[1]["policy_contract"]["conv_dim"], 4)
        self.assertTrue(
            all(case["policy_contract"]["model_family"] == "sd1" for case in loaded)
        )
        self.assertTrue(
            all(case["policy_contract"]["v2"] is False for case in loaded)
        )
        self.assertTrue(
            all(case["input_contract"]["signature"] for case in loaded)
        )
        self.assertTrue(
            all(case["input_contract"]["base_model"]["sha256"] for case in loaded)
        )

    def test_d1_fresh_resume_training_inputs_match_and_output_dirs_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            case["resume_command"].extend(["--caption_dropout_rate", "0.1"])
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "training-input argv must match exactly",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            fresh_output = case["fresh_command"][
                case["fresh_command"].index("--output_dir") + 1
            ]
            resume_output_index = case["resume_command"].index("--output_dir")
            case["resume_command"][resume_output_index + 1] = fresh_output
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "--output_dir must be distinct",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_d1_requires_explicit_local_base_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            for phase_key in ("fresh_command", "resume_command"):
                option_index = case[phase_key].index(
                    "--pretrained_model_name_or_path"
                )
                del case[phase_key][option_index : option_index + 2]
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "--pretrained_model_name_or_path exactly once",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_d1_input_files_are_revalidated_before_each_phase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            loaded = load_backend_feature_manifest(
                self._write(temp, [case]),
                repo_root=ROOT,
            )[0]
            validate_case_input_contract(loaded)

            model_path = Path(loaded["input_contract"]["base_model"]["path"])
            model_path.write_bytes(b"changed-model")
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "base_model.*changed",
            ):
                validate_case_input_contract(loaded)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            loaded = load_backend_feature_manifest(
                self._write(temp, [case]),
                repo_root=ROOT,
            )[0]
            policy_path = Path(loaded["input_contract"]["parameter_policy"]["path"])
            policy_path.write_text(
                json.dumps(_sd_lora_policy("adamw"), indent=2),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "parameter_policy.*changed",
            ):
                validate_case_input_contract(loaded)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            trainer_config = temp / "trainer.toml"
            trainer_config.write_text(
                "caption_dropout_rate = 0.0\n",
                encoding="utf-8",
            )
            for phase_key in ("fresh_command", "resume_command"):
                case[phase_key].extend(["--config_file", str(trainer_config)])
            loaded = load_backend_feature_manifest(
                self._write(temp, [case]),
                repo_root=ROOT,
            )[0]
            trainer_config.write_text(
                "caption_dropout_rate = 0.5\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "trainer config.*changed",
            ):
                validate_case_input_contract(loaded)

    def test_d1_rejects_sd2_v2_from_cli_or_trainer_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            for phase_key in ("fresh_command", "resume_command"):
                case[phase_key].append("--v2")
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "outside the D1 qualification scope",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            trainer_config = temp / "trainer-v2.toml"
            trainer_config.write_text("v2 = true\n", encoding="utf-8")
            for phase_key in ("fresh_command", "resume_command"):
                case[phase_key].extend(["--config_file", str(trainer_config)])
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "enables SD2.x v2 mode",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_d1_config_v2_false_remains_sd1(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            trainer_config = temp / "trainer-sd1.toml"
            trainer_config.write_text("v2 = false\n", encoding="utf-8")
            for phase_key in ("fresh_command", "resume_command"):
                case[phase_key].extend(["--config_file", str(trainer_config)])
            loaded = load_backend_feature_manifest(
                self._write(temp, [case]),
                repo_root=ROOT,
            )
        self.assertEqual(loaded[0]["policy_contract"]["model_family"], "sd1")

    def test_d1_cases_require_stock_network_module_and_real_conv_lora(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            for phase_key in ("fresh_command", "resume_command"):
                module_index = case[phase_key].index("--network_module")
                case[phase_key][module_index + 1] = "networks.other"
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "--network_module exactly once to 'networks.lora'",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[1])
            for phase_key in ("fresh_command", "resume_command"):
                args_index = case[phase_key].index("--network_args")
                del case[phase_key][args_index : args_index + 2]
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "requires integer conv_dim>0",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_d1_cases_lock_accumulation_and_full_bf16_cli(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[1])
            for phase_key in ("fresh_command", "resume_command"):
                accum_index = case[phase_key].index("--gradient_accumulation_steps")
                case[phase_key][accum_index + 1] = "1"
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "--gradient_accumulation_steps exactly once to '2'",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            for phase_key in ("fresh_command", "resume_command"):
                case[phase_key].remove("--full_bf16")
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "--full_bf16 exactly once",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_d1_authority_fields_cannot_be_hidden_in_trainer_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            trainer_config = temp / "trainer-authority.toml"
            trainer_config.write_text(
                'gradient_accumulation_steps = 1\n',
                encoding="utf-8",
            )
            for phase_key in ("fresh_command", "resume_command"):
                case[phase_key].extend(["--config_file", str(trainer_config)])
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "may not hide D1 qualification authority fields",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_d1_sd_lora_requires_explicit_same_policy_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            for phase_key in ("fresh_command", "resume_command"):
                option_index = case[phase_key].index("--parameter_policy_config")
                del case[phase_key][option_index : option_index + 2]
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "--parameter_policy_config exactly once",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            second = temp / "other-policy.json"
            second.write_text(
                json.dumps(_sd_lora_policy("adamw")),
                encoding="utf-8",
            )
            option_index = case["resume_command"].index("--parameter_policy_config")
            case["resume_command"][option_index + 1] = str(second)
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "same Parameter Policy sidecar",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_d1_muon_case_requires_real_conv_fallback_route(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[1])
            policy_path = Path(
                case["fresh_command"][
                    case["fresh_command"].index("--parameter_policy_config") + 1
                ]
            )
            policy = json.loads(policy_path.read_text(encoding="utf-8"))
            policy["components"]["unet.conv.adapter"] = {"train": False}
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "must train D1 coverage components",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_d1_policy_cannot_be_hidden_in_trainer_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _sd_lora_case(temp, SD_LORA_FULL_BF16_CASE_IDS[0])
            policy_path = case["fresh_command"][
                case["fresh_command"].index("--parameter_policy_config") + 1
            ]
            trainer_config = temp / "trainer.toml"
            trainer_config.write_text(
                f'parameter_policy_config = "{policy_path.replace(chr(92), chr(47))}"\n',
                encoding="utf-8",
            )
            for phase_key in ("fresh_command", "resume_command"):
                case[phase_key].extend(["--config_file", str(trainer_config)])
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "may not hide D1 qualification authority fields",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_manifest_must_live_outside_repository(self):
        internal = ROOT / "backend-feature-manifest-test.json"
        try:
            internal.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "outside the repository",
            ):
                load_backend_feature_manifest(internal, repo_root=ROOT)
        finally:
            internal.unlink(missing_ok=True)

    def test_duplicate_case_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _case(temp)
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "Duplicate",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case, dict(case)]),
                    repo_root=ROOT,
                )

    def test_unknown_backend_and_feature_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            bad_backend = _case(temp)
            bad_backend["train_type"] = "future-backend"
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "unknown train_type",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [bad_backend]),
                    repo_root=ROOT,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            bad_feature = _case(temp)
            bad_feature["feature"] = "future_feature"
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "feature='full_bf16'",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [bad_feature]),
                    repo_root=ROOT,
                )

    def test_d0_d1_scope_rejects_dreambooth_and_anima(self):
        for train_type in ("sd-dreambooth", "anima-lora", "anima-finetune"):
            with self.subTest(train_type=train_type), tempfile.TemporaryDirectory() as temp_dir:
                temp = Path(temp_dir)
                case = _case(temp)
                case["train_type"] = train_type
                with self.assertRaisesRegex(
                    BackendFeatureGpuMatrixError,
                    "outside the D0/D1 backend qualification scope",
                ):
                    load_backend_feature_manifest(
                        self._write(temp, [case]),
                        repo_root=ROOT,
                    )

    def test_command_must_start_with_canonical_production_trainer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _case(temp)
            case["fresh_command"][0] = "scripts/dev/flux_train_network.py"
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "canonical production trainer",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_resume_command_must_reference_fresh_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _case(temp)
            case["resume_command"] = [
                
                "scripts/dev/flux_train.py",
                "--max_train_steps",
                "2",
                "--save_every_n_steps",
                "1",
                "--save_state",
            ]
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "--resume exactly once",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_resume_command_accepts_equals_form_checkpoint_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _case(temp)
            case["resume_command"] = [
                
                "scripts/dev/flux_train.py",
                f"--resume={case['fresh_checkpoint_dir']}",
                "--max_train_steps=2",
                "--save_every_n_steps=1",
                "--save_state",
            ]
            loaded = load_backend_feature_manifest(
                self._write(temp, [case]),
                repo_root=ROOT,
            )
        self.assertEqual(len(loaded), 1)

    def test_qualification_commands_keep_two_step_scheduler_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _case(temp)
            case["fresh_command"] = [
                
                "scripts/dev/flux_train.py",
                "--max_train_steps",
                "1",
                "--save_every_n_steps",
                "1",
                "--save_state",
            ]
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "--max_train_steps exactly once to '2'",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _case(temp)
            case["resume_command"] = [
                
                "scripts/dev/flux_train.py",
                "--resume",
                case["fresh_checkpoint_dir"],
                "--max_train_steps",
                "1",
                "--save_every_n_steps",
                "1",
                "--save_state",
            ]
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "--max_train_steps exactly once to '2'",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_trainer_config_cannot_hide_lifecycle_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            trainer_config = temp / "trainer.toml"
            trainer_config.write_text("max_train_epochs = 1\n", encoding="utf-8")
            case = _case(temp)
            case["fresh_command"].extend(
                ["--config_file", str(trainer_config)]
            )
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "trainer config may not override qualification lifecycle fields",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_trainer_config_nested_lifecycle_override_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            trainer_config = temp / "trainer-nested.toml"
            trainer_config.write_text(
                "[training]\ninitial_step = 1\n",
                encoding="utf-8",
            )
            case = _case(temp)
            case["resume_command"].extend(
                ["--config_file", str(trainer_config)]
            )
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "trainer config may not override qualification lifecycle fields",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_qualification_commands_require_state_save_each_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _case(temp)
            case["fresh_command"].remove("--save_state")
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "--save_state",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_commands_must_use_exact_repo_cwd_and_entrypoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            case = _case(temp)
            case["cwd"] = str(temp)
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "cwd must be the exact qualification repository root",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            external_script = temp / "external.py"
            external_script.write_text("pass\n", encoding="utf-8")
            case = _case(temp)
            case["fresh_command"][0] = str(external_script)
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "canonical production trainer",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [case]),
                    repo_root=ROOT,
                )

    def test_checkpoint_directories_must_be_distinct_and_external(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            same = _case(temp)
            same["resume_checkpoint_dir"] = same["fresh_checkpoint_dir"]
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "distinct fresh/resume",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [same]),
                    repo_root=ROOT,
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            internal = _case(temp)
            internal["fresh_checkpoint_dir"] = str(ROOT / "checkpoint-1")
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "outside the repository",
            ):
                load_backend_feature_manifest(
                    self._write(temp, [internal]),
                    repo_root=ROOT,
                )


class SharedRegressionEvidenceTests(unittest.TestCase):
    def test_exact_head_regression_evidence_passes(self):
        contract = _qualification_contract()
        validated = validate_shared_full_bf16_regression_evidence(
            _shared_regression_evidence(
                commit="exact-head",
                qualification_contract=contract,
            ),
            expected_commit="exact-head",
            qualification_contract=contract,
        )
        self.assertEqual(validated["status"], "pass")
        self.assertEqual(
            validated["id"],
            SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID,
        )
        self.assertEqual(validated["commit"], "exact-head")

    def test_wrong_commit_or_mode_fails_closed(self):
        contract = _qualification_contract()

        wrong_commit = _shared_regression_evidence(
            commit="other-head",
            qualification_contract=contract,
        )
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "does not match the D1 exact head",
        ):
            validate_shared_full_bf16_regression_evidence(
                wrong_commit,
                expected_commit="exact-head",
                qualification_contract=contract,
            )

        wrong_mode = _shared_regression_evidence(
            commit="exact-head",
            qualification_contract=contract,
        )
        wrong_mode["qualification_mode"] = "d0-promotion"
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "qualification_mode='regression'",
        ):
            validate_shared_full_bf16_regression_evidence(
                wrong_mode,
                expected_commit="exact-head",
                qualification_contract=contract,
            )

    def test_failed_or_incomplete_c1_c4_matrix_fails_closed(self):
        contract = _qualification_contract()
        failed = _shared_regression_evidence(
            commit="exact-head",
            qualification_contract=contract,
        )
        failed["cases"][-1]["status"] = "fail"
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "complete PASS C1-C4 matrix",
        ):
            validate_shared_full_bf16_regression_evidence(
                failed,
                expected_commit="exact-head",
                qualification_contract=contract,
            )

        incomplete = _shared_regression_evidence(
            commit="exact-head",
            qualification_contract=contract,
        )
        incomplete["cases"] = incomplete["cases"][:-1]
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "complete PASS C1-C4 matrix",
        ):
            validate_shared_full_bf16_regression_evidence(
                incomplete,
                expected_commit="exact-head",
                qualification_contract=contract,
            )

    def test_wrong_summary_id_status_or_environment_fails_closed(self):
        contract = _qualification_contract()

        wrong_id = _shared_regression_evidence(
            commit="exact-head",
            qualification_contract=contract,
        )
        wrong_id["shared_optimizer_regression"]["id"] = "wrong"
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "unexpected summary id",
        ):
            validate_shared_full_bf16_regression_evidence(
                wrong_id,
                expected_commit="exact-head",
                qualification_contract=contract,
            )

        failed_summary = _shared_regression_evidence(
            commit="exact-head",
            qualification_contract=contract,
        )
        failed_summary["shared_optimizer_regression"]["status"] = "fail"
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "summary did not pass",
        ):
            validate_shared_full_bf16_regression_evidence(
                failed_summary,
                expected_commit="exact-head",
                qualification_contract=contract,
            )

        other_contract = copy.deepcopy(contract)
        other_contract["torch_base_version"] = "2.8.0"
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "qualification environment does not match",
        ):
            validate_shared_full_bf16_regression_evidence(
                _shared_regression_evidence(
                    commit="exact-head",
                    qualification_contract=other_contract,
                ),
                expected_commit="exact-head",
                qualification_contract=contract,
            )


class BackendFeatureCheckpointTests(unittest.TestCase):
    def test_full_bf16_manifest_contract_passes(self):
        payload = _checkpoint()
        self.assertIs(
            validate_full_bf16_checkpoint_manifest(
                payload,
                train_type="flux-finetune",
                manifest_version=2,
            ),
            payload,
        )

    def test_wrong_manifest_version_or_execution_identity_fails(self):
        with self.assertRaisesRegex(BackendFeatureGpuMatrixError, "version"):
            validate_full_bf16_checkpoint_manifest(
                _checkpoint(),
                train_type="flux-finetune",
                manifest_version=3,
            )

        payload = _checkpoint()
        payload["execution_identity"]["features"]["full_bf16"][
            "trainable_parameter_dtype"
        ] = "float32"
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "full-BF16 contract",
        ):
            validate_full_bf16_checkpoint_manifest(
                payload,
                train_type="flux-finetune",
                manifest_version=2,
            )

    def test_identity_signatures_must_match_their_payloads(self):
        payload = _checkpoint()
        payload["execution_signature"] = "wrong"
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "execution_signature does not match",
        ):
            validate_full_bf16_checkpoint_manifest(
                payload,
                train_type="flux-finetune",
                manifest_version=2,
            )

        payload = _checkpoint()
        payload["scheduler_signature"] = "wrong"
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "scheduler_signature does not match",
        ):
            validate_full_bf16_checkpoint_manifest(
                payload,
                train_type="flux-finetune",
                manifest_version=2,
            )

    def test_missing_scheduler_or_execution_identity_fails_closed(self):
        for key in (
            "scheduler_identity",
            "scheduler_signature",
            "execution_identity",
            "execution_signature",
        ):
            with self.subTest(key=key):
                payload = _checkpoint()
                del payload[key]
                with self.assertRaisesRegex(
                    BackendFeatureGpuMatrixError,
                    "missing required",
                ):
                    validate_full_bf16_checkpoint_manifest(
                        payload,
                        train_type="flux-finetune",
                        manifest_version=2,
                    )

    def test_checkpoint_progress_requires_lineage_ids(self):
        manifest = _checkpoint()
        payload = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=1,
            scheduler_step_count=2,
            scheduler_last_epoch=1,
            checkpoint_id="",
        )
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "non-empty checkpoint_id",
        ):
            validate_checkpoint_progress(
                payload,
                checkpoint_manifest=manifest,
            )

    def test_checkpoint_progress_proves_logical_step_one_to_two(self):
        manifest = _checkpoint()
        fresh = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=1,
            scheduler_step_count=2,
            scheduler_last_epoch=1,
            checkpoint_id="fresh-id",
        )
        resumed = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=2,
            scheduler_step_count=3,
            scheduler_last_epoch=2,
            checkpoint_id="resume-id",
            resume_source_checkpoint_id="fresh-id",
        )
        self.assertIs(
            validate_checkpoint_progress(
                fresh,
                checkpoint_manifest=manifest,
            ),
            fresh,
        )
        validate_checkpoint_progress(
            resumed,
            checkpoint_manifest=manifest,
        )
        compare_checkpoint_progress(fresh, resumed)

    def test_checkpoint_progress_rejects_independent_fresh_or_wrong_parent(self):
        manifest = _checkpoint()
        fresh = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=1,
            scheduler_step_count=2,
            scheduler_last_epoch=1,
            checkpoint_id="fresh-id",
        )
        independent = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=2,
            scheduler_step_count=3,
            scheduler_last_epoch=2,
            checkpoint_id="independent-id",
        )
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "lineage does not reference",
        ):
            compare_checkpoint_progress(fresh, independent)

        wrong_parent = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=2,
            scheduler_step_count=3,
            scheduler_last_epoch=2,
            checkpoint_id="resume-id",
            resume_source_checkpoint_id="other-id",
        )
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "lineage does not reference",
        ):
            compare_checkpoint_progress(fresh, wrong_parent)

    def test_checkpoint_progress_rejects_reused_id_or_skipped_step(self):
        manifest = _checkpoint()
        fresh = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=1,
            scheduler_step_count=2,
            scheduler_last_epoch=1,
            checkpoint_id="fresh-id",
        )
        reused = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=2,
            scheduler_step_count=3,
            scheduler_last_epoch=2,
            checkpoint_id="fresh-id",
            resume_source_checkpoint_id="fresh-id",
        )
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "distinct checkpoint_id",
        ):
            compare_checkpoint_progress(fresh, reused)

        skipped = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=3,
            scheduler_step_count=4,
            scheduler_last_epoch=3,
            checkpoint_id="resume-id",
            resume_source_checkpoint_id="fresh-id",
        )
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "logical optimizer step 2",
        ):
            compare_checkpoint_progress(fresh, skipped)


    def test_fresh_resume_trainable_counts_must_match(self):
        fresh = _checkpoint()
        resumed = _checkpoint()
        resumed["trainable_parameter_elements"] += 1
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "trainable_parameter_elements",
        ):
            compare_backend_checkpoint_contracts(fresh, resumed)

    def test_fresh_resume_contract_must_match(self):
        fresh = _checkpoint()
        resumed = _checkpoint()
        compare_backend_checkpoint_contracts(fresh, resumed)

        resumed = _checkpoint()
        resumed["runtime_topology_fingerprint"] = "different"
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "runtime_topology_fingerprint",
        ):
            compare_backend_checkpoint_contracts(fresh, resumed)


class SdLoraFullBf16PromotionTests(unittest.TestCase):
    def _snapshot(self):
        return copy.deepcopy(backend_feature_qualification_snapshot())

    def _passing_rows(self):
        return [_d1_case_row(case_id) for case_id in SD_LORA_FULL_BF16_CASE_IDS]

    def _summary(self, rows=None, snapshot=None, regression=None):
        return summarize_sd_lora_full_bf16_promotion(
            self._passing_rows() if rows is None else rows,
            self._snapshot() if snapshot is None else snapshot,
            _shared_regression_contract() if regression is None else regression,
        )

    def test_complete_d1_matrix_is_promotion_eligible(self):
        summary = self._summary()
        self.assertEqual(summary["id"], SD_LORA_FULL_BF16_EVIDENCE_ID)
        self.assertEqual(
            SD_LORA_FULL_BF16_EVIDENCE_ID,
            "phase-d1:backend:sd-lora-sd1:full-bf16:v2",
        )
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["promotion_eligible"])
        self.assertTrue(summary["backend_qualification_eligible"])
        self.assertTrue(summary["backend_row_match"])
        self.assertTrue(summary["shared_optimizer_authority_match"])
        self.assertTrue(summary["shared_regression_match"])
        self.assertEqual(
            summary["shared_regression_id"],
            SHARED_FULL_BF16_REGRESSION_EVIDENCE_ID,
        )
        self.assertEqual(summary["case_contract_errors"], [])
        self.assertEqual(summary["unexpected_backend_promotions"], [])
        self.assertEqual(summary["unexpected_optimizer_promotions"], [])
        self.assertFalse(summary["production_qualification_mutated"])

    def test_missing_or_failed_case_blocks_promotion(self):
        rows = self._passing_rows()
        summary = self._summary(rows=rows[:-1])
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["missing_cases"], [SD_LORA_FULL_BF16_CASE_IDS[1]])

        failed = self._passing_rows()
        failed[1]["status"] = "fail"
        summary = self._summary(rows=failed)
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["failed_cases"], [SD_LORA_FULL_BF16_CASE_IDS[1]])

    def test_case_identity_optimizer_topology_and_input_identity_are_verified(self):
        wrong_identity = self._passing_rows()
        wrong_identity[0]["train_type"] = "flux-lora"
        summary = self._summary(rows=wrong_identity)
        self.assertEqual(summary["status"], "fail")
        self.assertTrue(summary["case_contract_errors"])

        wrong_topology = self._passing_rows()
        wrong_topology[1] = _d1_case_row(
            SD_LORA_FULL_BF16_CASE_IDS[1],
            optimizer_types=("Muon",),
        )
        summary = self._summary(rows=wrong_topology)
        self.assertEqual(summary["status"], "fail")
        self.assertTrue(summary["case_contract_errors"])

        wrong_policy = self._passing_rows()
        wrong_policy[1]["policy_contract"]["kind"] = "adamw"
        summary = self._summary(rows=wrong_policy)
        self.assertEqual(summary["status"], "fail")
        self.assertTrue(summary["case_contract_errors"])

        wrong_hash = self._passing_rows()
        wrong_hash[0]["fresh_checkpoint_manifest"]["policy_hash"] = "other"
        summary = self._summary(rows=wrong_hash)
        self.assertEqual(summary["status"], "fail")
        self.assertTrue(summary["case_contract_errors"])

        stale_input = self._passing_rows()
        stale_input[0]["pre_resume_input_contract_valid"] = False
        summary = self._summary(rows=stale_input)
        self.assertEqual(summary["status"], "fail")
        self.assertTrue(summary["case_contract_errors"])

    def test_source_backend_authority_must_match_exact_d1_evidence(self):
        for status, evidence in (
            ("pending", None),
            ("qualified", "wrong:evidence"),
        ):
            with self.subTest(status=status, evidence=evidence):
                snapshot = self._snapshot()
                snapshot["backends"]["sd-lora"]["status"] = status
                snapshot["backends"]["sd-lora"]["evidence_case_id"] = evidence
                summary = self._summary(snapshot=snapshot)
                self.assertEqual(summary["status"], "fail")
                self.assertFalse(summary["backend_row_match"])

    def test_d0_optimizer_authority_is_required_for_both_shared_optimizers(self):
        self.assertEqual(
            SHARED_FULL_BF16_OPTIMIZER_EVIDENCE_ID,
            "phase-d0:shared-adamw-muon-full-bf16:v1",
        )
        for optimizer_name in ("AdamW", "Muon"):
            with self.subTest(optimizer=optimizer_name):
                snapshot = self._snapshot()
                snapshot["optimizers"][optimizer_name]["evidence_case_id"] = "wrong"
                summary = self._summary(snapshot=snapshot)
                self.assertEqual(summary["status"], "fail")
                self.assertFalse(summary["shared_optimizer_authority_match"])

    def test_shared_regression_contract_is_mandatory(self):
        summary = self._summary(regression={})
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(summary["shared_regression_match"])

        wrong = _shared_regression_contract()
        wrong["id"] = "wrong:regression"
        summary = self._summary(regression=wrong)
        self.assertEqual(summary["status"], "fail")
        self.assertFalse(summary["shared_regression_match"])

    def test_unexpected_backend_or_optimizer_promotion_blocks_d1(self):
        snapshot = self._snapshot()
        snapshot["backends"]["sdxl-lora"] = {
            "status": "qualified",
            "reason": "unexpected",
            "evidence_case_id": "unexpected:backend",
        }
        snapshot["optimizers"]["Lion"] = {
            "status": "qualified",
            "reason": "unexpected",
            "evidence_case_id": "unexpected:optimizer",
        }
        summary = self._summary(snapshot=snapshot)
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["unexpected_backend_promotions"], ["sdxl-lora"])
        self.assertEqual(summary["unexpected_optimizer_promotions"], ["Lion"])


class SdXlLoraFullBf16QualificationTests(unittest.TestCase):
    def test_two_sdxl_cases_accept_only_canonical_real_input_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            cases = [
                _sdxl_lora_case(temp / "adamw", SDXL_LORA_FULL_BF16_CASE_IDS[0]),
                _sdxl_lora_case(temp / "muon", SDXL_LORA_FULL_BF16_CASE_IDS[1]),
            ]
            loaded = load_backend_feature_manifest(
                SdXlLoraFullBf16QualificationTests._manifest(temp, cases),
                repo_root=ROOT,
            )
        self.assertEqual([row["case_id"] for row in loaded], list(SDXL_LORA_FULL_BF16_CASE_IDS))
        for row in loaded:
            with self.subTest(case=row["case_id"]):
                self.assertEqual(row["train_type"], "sdxl-lora")
                self.assertEqual(row["input_contract"]["model_family"], "sdxl-base")
                self.assertEqual(
                    row["policy_contract"]["covered_text_components"],
                    ["text_encoder_1.adapter", "text_encoder_2.adapter"],
                )
                self.assertIsNotNone(row["policy_contract"]["policy_hash"])

    @staticmethod
    def _manifest(temp: Path, cases: list[dict]) -> Path:
        path = temp / "sdxl-manifest.json"
        path.write_text(
            json.dumps({
                "schema": BACKEND_FEATURE_MANIFEST_SCHEMA,
                "version": 1,
                "cases": cases,
            }),
            encoding="utf-8",
        )
        return path

    def test_missing_text_encoder_or_wrong_model_family_fails_closed(self):
        for mutation in ("remove_te2", "wrong_trainer", "wrong_base_model", "unet_only"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                temp = Path(td)
                case = _sdxl_lora_case(temp, SDXL_LORA_FULL_BF16_CASE_IDS[0])
                if mutation == "remove_te2":
                    policy_path = Path(
                        case["fresh_command"][
                            case["fresh_command"].index("--parameter_policy_config") + 1
                        ]
                    )
                    policy = json.loads(policy_path.read_text(encoding="utf-8"))
                    policy["components"]["text_encoder_2.adapter"]["train"] = False
                    policy_path.write_text(json.dumps(policy), encoding="utf-8")
                elif mutation == "wrong_trainer":
                    case["fresh_command"][0] = "scripts/stable/train_network.py"
                elif mutation == "wrong_base_model":
                    index = case["resume_command"].index("--pretrained_model_name_or_path") + 1
                    other_model = temp / "different.safetensors"
                    other_model.write_bytes(b"different-model")
                    case["resume_command"][index] = str(other_model)
                else:
                    case["fresh_command"].append("--network_train_unet_only")
                    case["resume_command"].append("--network_train_unet_only")
                with self.assertRaises(BackendFeatureGpuMatrixError):
                    load_backend_feature_manifest(
                        self._manifest(temp, [case]), repo_root=ROOT
                    )

    def test_uncertified_training_variants_fail_closed(self):
        for flag in ("--cache_text_encoder_outputs", "--cache_text_encoder_outputs_to_disk",
                     "--gradient_checkpointing", "--v2"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as td:
                temp = Path(td)
                case = _sdxl_lora_case(temp, SDXL_LORA_FULL_BF16_CASE_IDS[0])
                case["fresh_command"].append(flag)
                case["resume_command"].append(flag)
                with self.assertRaises(BackendFeatureGpuMatrixError):
                    load_backend_feature_manifest(
                        self._manifest(temp, [case]), repo_root=ROOT
                    )

    def test_conv_fallback_is_required_in_muon_case(self):
        with tempfile.TemporaryDirectory() as td:
            temp = Path(td)
            case = _sdxl_lora_case(temp, SDXL_LORA_FULL_BF16_CASE_IDS[1])
            for phase in ("fresh_command", "resume_command"):
                cmd = case[phase]
                idx = cmd.index("--network_args")
                del cmd[idx:idx + 2]
            with self.assertRaisesRegex(BackendFeatureGpuMatrixError, "conv_dim"):
                load_backend_feature_manifest(
                    self._manifest(temp, [case]), repo_root=ROOT
                )

    def test_independent_sdxl_promotion_requires_both_full_cases(self):
        rows = [_sdxl_d1_case_row(case_id) for case_id in SDXL_LORA_FULL_BF16_CASE_IDS]
        snapshot = copy.deepcopy(backend_feature_qualification_snapshot())
        regression = _shared_regression_contract()
        summary = summarize_sdxl_lora_full_bf16_promotion(
            rows, snapshot, regression
        )
        self.assertEqual(summary["id"], SDXL_LORA_FULL_BF16_EVIDENCE_ID)
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["promotion_eligible"])
        self.assertEqual(summary["unexpected_backend_promotions"], [])
        self.assertEqual(
            summarize_sdxl_lora_full_bf16_promotion(
                rows[:1], snapshot, regression
            )["status"],
            "fail",
        )

        missing_updates = copy.deepcopy(rows)
        missing_updates[0]["component_update_evidence"]["changed_tensor_counts"]["te2"] = 0
        self.assertEqual(
            summarize_sdxl_lora_full_bf16_promotion(
                missing_updates, snapshot, regression
            )["status"],
            "fail",
        )
        missing_conv = copy.deepcopy(rows)
        missing_conv[1]["component_update_evidence"]["changed_tensor_counts"]["conv3x3"] = 0
        self.assertEqual(
            summarize_sdxl_lora_full_bf16_promotion(
                missing_conv, snapshot, regression
            )["status"],
            "fail",
        )

    def test_existing_sd1_promotion_remains_valid_with_exact_sdxl_authority(self):
        rows = [_d1_case_row(case_id) for case_id in SD_LORA_FULL_BF16_CASE_IDS]
        snapshot = copy.deepcopy(backend_feature_qualification_snapshot())
        summary = summarize_sd_lora_full_bf16_promotion(
            rows, snapshot, _shared_regression_contract()
        )
        self.assertEqual(summary["status"], "pass")
        snapshot["backends"]["sdxl-lora"]["evidence_case_id"] = "wrong-id"
        summary = summarize_sd_lora_full_bf16_promotion(
            rows, snapshot, _shared_regression_contract()
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["unexpected_backend_promotions"], ["sdxl-lora"])


class BackendFeatureRunnerSourceTests(unittest.TestCase):
    def test_runner_checks_exact_clean_head_before_torch_import(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"--expected-commit"', source)
        self.assertIn('"status", "--porcelain=v1", "--untracked-files=all"', source)
        self.assertLess(
            source.index("_BOOTSTRAP_COMMIT = _assert_clean_head"),
            source.index("import torch"),
        )

    def test_runner_rechecks_exact_head_after_fresh_resume_and_final(self):
        source = RUNNER.read_text(encoding="utf-8")
        tree = ast.parse(source)

        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        run_case = functions["_run_case"]
        main = functions["main"]

        run_case_calls = [
            node
            for node in ast.walk(run_case)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_assert_clean_head"
        ]
        self.assertGreaterEqual(len(run_case_calls), 4)

        assigned_keys = {
            target.slice.value
            for node in ast.walk(run_case)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "row"
            and isinstance(target.slice, ast.Constant)
            and isinstance(target.slice.value, str)
        }
        self.assertTrue(
            {
                "pre_fresh_commit",
                "post_fresh_commit",
                "pre_resume_commit",
                "post_resume_commit",
            }.issubset(assigned_keys)
        )

        final_assignments = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "evidence"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "final_provenance_commit"
                for target in node.targets
            )
        ]
        self.assertEqual(len(final_assignments), 1)
        final_value = final_assignments[0].value
        self.assertIsInstance(final_value, ast.Call)
        self.assertIsInstance(final_value.func, ast.Name)
        self.assertEqual(final_value.func.id, "_assert_clean_head")

    def test_runner_rejects_stale_checkpoint_directories(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("def _assert_checkpoint_dir_absent(", source)
        self.assertIn("stale qualification", source)
        self.assertIn("evidence is forbidden", source)
        self.assertGreaterEqual(
            source.count("_assert_checkpoint_dir_absent("),
            4,
        )

    def test_manifest_support_binds_commands_to_repo_entrypoint(self):
        support = (
            ROOT / "tools" / "parameter_policy_backend_feature_gpu_support.py"
        ).read_text(encoding="utf-8")
        self.assertIn("def _canonical_backend_entrypoint(", support)
        self.assertIn("PARAMETER_POLICY_BACKEND_MATRIX", support)
        self.assertIn("canonical production trainer", support)
        self.assertIn("cwd must be the exact", support)

    def test_runner_owns_python_and_accelerate_launcher(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("def _qualification_command(", source)
        self.assertIn("sys.executable", source)
        self.assertIn('"accelerate.commands.launch"', source)
        self.assertIn('"config" / "accelerate-gpu.yaml"', source)

    def test_runner_records_production_dependency_provenance(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"requirements_sha256"', source)
        for package in (
            "transformers",
            "diffusers",
            "safetensors",
            "huggingface-hub",
            "toml",
            "numpy",
            "opencv-python",
            "imagesize",
            "torchvision",
        ):
            self.assertIn(f'"{package}"', source)

    def test_runner_records_redacted_command_contract(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("def _redacted_argv(", source)
        self.assertIn("_SECRET_OPTION_MARKERS", source)
        self.assertIn('"<redacted>"', source)
        self.assertIn('"command_contract": _command_contract(case)', source)
        self.assertIn('"environment_keys": sorted(case["environment"])', source)
        self.assertIn('"policy_contract": case.get("policy_contract")', source)
        self.assertIn('"input_contract": case.get("input_contract")', source)
        self.assertIn("validate_case_input_contract(case)", source)

    def test_runner_requires_external_manifest_output_and_two_checkpoints(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('field="Backend feature --manifest"', source)
        self.assertIn('field="Backend feature --shared-regression-evidence"', source)
        self.assertIn('field="Backend feature --output"', source)
        self.assertIn('"fresh_checkpoint_dir"', source)
        self.assertIn('"resume_checkpoint_dir"', source)
        self.assertIn("compare_backend_checkpoint_contracts", source)
        self.assertIn("PARAMETER_POLICY_CHECKPOINT_PROGRESS", source)
        self.assertIn("compare_checkpoint_progress", source)
        support = (
            ROOT / "tools" / "parameter_policy_backend_feature_gpu_support.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"resume_source_checkpoint_id"', support)
        self.assertIn('"checkpoint_id"', support)

    def test_runner_never_promotes_or_bypasses_production_qualification(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("FULL_BF16_BACKEND_QUALIFICATIONS", source)
        self.assertNotIn("FULL_BF16_OPTIMIZER_QUALIFICATIONS", source)
        self.assertNotIn("temporary_execution_qualification", source)
        self.assertNotIn("DTS_FULL_BF16_BYPASS", source)
        self.assertIn('"backend_qualification_eligible": False', source)
        self.assertIn('"production_qualification_mutated": False', source)

    def test_runner_emits_d1_sd_lora_promotion_summary(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("backend_feature_qualification_snapshot", source)
        self.assertIn("summarize_sd_lora_full_bf16_promotion", source)
        self.assertIn('"backend_promotion"', source)
        self.assertIn("SD_LORA_FULL_BF16_CASE_IDS", source)
        self.assertIn("validate_shared_full_bf16_regression_evidence", source)
        self.assertIn('"shared_regression_contract"', source)
        self.assertIn("validate_qualification_environment", source)

    def test_workflow_uses_external_evidence_directory(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("DTS_EVIDENCE_DIR", workflow)
        self.assertIn("runner.temp", workflow)
        self.assertIn('rm -rf "$evidence_dir"', workflow)
        self.assertIn("Remove-Item -Recurse -Force", workflow)
        self.assertIn("run_parameter_policy_execution_gpu_matrix.py", workflow)
        self.assertIn("run_parameter_policy_backend_feature_gpu_matrix.py", workflow)
        self.assertIn("parameter-policy-execution-gpu-matrix.json", workflow)
        self.assertIn("parameter-policy-backend-feature-gpu-matrix.json", workflow)
        self.assertIn("--shared-regression-evidence", workflow)
        self.assertIn("execution_gate_mode=regression", workflow)

    def test_evidence_schema_is_stable(self):
        self.assertEqual(
            BACKEND_FEATURE_EVIDENCE_SCHEMA,
            "dts.parameter-policy.backend-feature-gpu-matrix",
        )
        self.assertEqual(BACKEND_FEATURE_EVIDENCE_VERSION, 2)


if __name__ == "__main__":
    unittest.main()
