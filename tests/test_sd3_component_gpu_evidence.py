"""Host-only regression tests for D1-SD3C1 Component mixed-BF16 R0 evidence.

These tests use no CUDA device and do not claim Full-BF16 qualification.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mikazuki.model_component_profiles import get_model_component_profile
from tools import run_sd3_component_gpu_evidence as r0


class SD3ComponentBaselineTests(unittest.TestCase):
    def _policy(self):
        components = {}
        for name in get_model_component_profile("sd3-lora").components:
            if name in {"mmdit.attention.adapter", "mmdit.mlp.adapter"}:
                components[name] = {
                    "train": True, "optimizer_profile": "baseline",
                    "learning_rate": 0.0001,
                }
            else:
                components[name] = {"train": False}
        return {
            "version": 1,
            "optimizer_profiles": {"baseline": {"type": "AdamW", "args": {}}},
            "components": components,
        }

    def _fixture(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        model = root / "sd3.5_medium.safetensors"
        model.write_bytes(b"host-only fake checkpoint")
        dataset = root / "dataset"
        dataset.mkdir()
        policy = root / "component.json"
        policy.write_text(json.dumps(self._policy()), encoding="utf-8")
        fresh, resumed = root / "fresh", root / "resumed"
        common = [
            "scripts/dev/sd3_train_network.py",
            "--pretrained_model_name_or_path", str(model),
            "--train_data_dir", str(dataset),
            "--parameter_policy_config", str(policy),
            "--max_train_steps", "2",
            "--save_every_n_steps", "1",
            "--save_state",
            "--mixed_precision", "bf16",
            "--optimizer_type", "AdamW",
            "--network_module", "networks.lora_sd3",
            "--network_train_unet_only",
            "--train_batch_size", "1",
            "--gradient_accumulation_steps", "1",
            "--resolution", "512",
            "--network_dim", "4",
            "--network_alpha", "4",
            "--save_model_as", "safetensors",
        ]
        return {
            "schema": r0.SCHEMA,
            "version": r0.VERSION,
            "fresh_command": [*common, "--output_dir", str(fresh)],
            "resume_command": [
                *common, "--resume", str(fresh / "r0-step00000001-state"),
                "--output_dir", str(resumed),
            ],
            "fresh_checkpoint_dir": str(fresh / "r0-step00000001-state"),
            "resume_checkpoint_dir": str(resumed / "r0-step00000002-state"),
            "fresh_adapter_path": str(fresh / "r0-step00000001.safetensors"),
            "resume_adapter_path": str(resumed / "r0-step00000002.safetensors"),
        }

    def test_valid_canonical_seven_component_adamw_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            data = self._fixture(Path(td))
            validated = r0.validate_manifest(data)
            self.assertEqual(validated["fresh"]["--gradient_accumulation_steps"], "1")
            self.assertEqual(validated["policy"]["optimizer_profiles"]["baseline"]["type"], "AdamW")
            self.assertEqual(len(validated["policy"]["components"]), 7)
            self.assertEqual(len(validated["policy_hash"]), 64)

    def test_negative_execution_features_and_unsafe_override_options(self):
        with tempfile.TemporaryDirectory() as td:
            data = self._fixture(Path(td))
            for option, value in (
                ("--full_bf16", None),
                ("--fp8_base", None),
                ("--gradient_checkpointing", None),
                ("--deepspeed", None),
                ("--network_train_text_encoder_only", None),
                ("--network_args", "train_t5xxl=true"),
                ("--network_weights", str(Path(td) / "existing.safetensors")),
                ("--fused_optimizer_groups", "2"),
            ):
                with self.subTest(option=option):
                    candidate = copy.deepcopy(data)
                    for command in ("fresh_command", "resume_command"):
                        candidate[command].append(option)
                        if value is not None:
                            candidate[command].append(value)
                    with self.assertRaises((r0.SD3ComponentBaselineError, r0.sd3b.SD3RuntimeEvidenceError)):
                        r0.validate_manifest(candidate)

    def test_fails_closed_on_unqualified_policy_components_and_optimizer(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = self._fixture(root)
            path = root / "component.json"
            for mutation in ("text_encoder", "missing_mlp", "muon", "missing_component"):
                with self.subTest(mutation=mutation):
                    policy = self._policy()
                    if mutation == "text_encoder":
                        policy["components"]["clip_l.adapter"] = {
                            "train": True, "optimizer_profile": "baseline",
                            "learning_rate": 0.0001,
                        }
                    elif mutation == "missing_mlp":
                        policy["components"]["mmdit.mlp.adapter"] = {"train": False}
                    elif mutation == "muon":
                        policy["optimizer_profiles"]["baseline"]["type"] = "Muon"
                    elif mutation == "missing_component":
                        del policy["components"]["t5xxl.adapter"]
                    path.write_text(json.dumps(policy), encoding="utf-8")
                    with self.assertRaises(r0.SD3ComponentBaselineError):
                        r0.validate_manifest(data)

    def test_rejects_modified_inputs_and_missing_policy(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = self._fixture(root)
            validated = r0.validate_manifest(data)
            (root / "component.json").write_text(json.dumps(self._policy()) + " ", encoding="utf-8")
            with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "changed"):
                r0._assert_unchanged_inputs(validated)
            (root / "component.json").unlink()
            with self.assertRaises(r0.SD3ComponentBaselineError):
                r0.validate_manifest(data)

    def test_manifest_only_claims_mixed_component_ownership(self):
        policy = self._policy()
        trained = sorted(k for k, v in policy["components"].items() if v["train"])
        frozen = sorted(k for k, v in policy["components"].items() if not v["train"])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "policy.json"
            path.write_text(json.dumps(policy), encoding="utf-8")
            canonical, digest = r0._read_policy(path)

        manifest = {
            "version": 2,
            "train_type": "sd3-lora",
            "policy_hash": digest,
            "runtime_topology_fingerprint": "topology-id",
            "trainable_components": trained,
            "frozen_components": frozen,
            "trainable_parameter_tensors": 12,
            "optimizer_profiles": {"baseline": "AdamW"},
            "optimizers": [{
                "profile_name": "baseline",
                "optimizer_type": "AdamW",
                "topology_fingerprint": "child-topology",
            }],
            "scheduler_identity": {"scheduler": "constant"},
            "scheduler_signature": None,
            "schedulers": [{"profile_name": "baseline", "topology_fingerprint": "sched-topology"}],
        }
        manifest["scheduler_signature"] = hashlib.sha256(
            json.dumps(manifest["scheduler_identity"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        evidence = {
            "parameter_policy_manifest": manifest,
            "progress": {
                "schema": "dts.sd3.resume-progress", "version": 1,
                "current_epoch": 1,
                "completed_optimizer_steps": 1, "gradient_accumulation_steps": 1,
                "num_processes": 1, "dataloader_batches": 1,
            },
            "forward_semantics": {"attention_contract": "extra-attention-head-dim-v1"},
            "optimizer_state_sha256": {"optimizer.bin": "step-1"},
        }
        self.assertEqual(r0.verify_component_checkpoint(evidence, policy=canonical, policy_hash=digest, expected_step=1), manifest)
        wrong = copy.deepcopy(evidence)
        wrong["parameter_policy_manifest"]["policy_hash"] = "incorrect"
        with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "hash mismatch"):
            r0.verify_component_checkpoint(wrong, policy=canonical, policy_hash=digest, expected_step=1)

        wrong = copy.deepcopy(evidence)
        wrong["parameter_policy_manifest"]["execution_identity"] = {"full_bf16": True}
        with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "full-BF16"):
            r0.verify_component_checkpoint(wrong, policy=canonical, policy_hash=digest, expected_step=1)

        wrong = copy.deepcopy(evidence)
        wrong["parameter_policy_manifest"]["trainable_components"] = frozen
        with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "trainable"):
            r0.verify_component_checkpoint(wrong, policy=canonical, policy_hash=digest, expected_step=1)

        wrong = copy.deepcopy(evidence)
        wrong["parameter_policy_manifest"]["scheduler_signature"] = "forged"
        with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "signature"):
            r0.verify_component_checkpoint(wrong, policy=canonical, policy_hash=digest, expected_step=1)

        wrong = copy.deepcopy(evidence)
        wrong["progress"]["schema"] = "bad"
        with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "Epoch boundary"):
            r0.verify_component_checkpoint(wrong, policy=canonical, policy_hash=digest, expected_step=1)

        wrong = copy.deepcopy(evidence)
        wrong["progress"]["num_processes"] = 2
        with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "Epoch boundary"):
            r0.verify_component_checkpoint(wrong, policy=canonical, policy_hash=digest, expected_step=1)

        second = copy.deepcopy(evidence)
        second["progress"]["completed_optimizer_steps"] = 2
        second["progress"]["current_epoch"] = 2
        second["optimizer_state_sha256"]["optimizer.bin"] = "step-2"
        r0.compare_component_checkpoints(evidence, second)
        second["parameter_policy_manifest"]["optimizer_profiles"] = {"foreign": "AdamW"}
        with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "changed"):
            r0.compare_component_checkpoints(evidence, second)

    def test_source_never_promotes_full_bf16(self):
        source = (Path(__file__).resolve().parents[1] / "tools/run_sd3_component_gpu_evidence.py").read_text(encoding="utf-8")
        self.assertIn('"qualifies_full_bf16": False', source)
        self.assertIn('"backend_qualification_eligible": False', source)
        self.assertIn('"production_qualification_mutated": False', source)
        self.assertNotIn("FULL_BF16_BACKEND_QUALIFICATIONS[", source)


class SD3ComponentInputGeneratorTests(unittest.TestCase):
    def test_generates_external_policy_and_replayable_manifest(self):
        from tools.prepare_sd3_component_baseline import generate

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            model = base / "sd35_medium.safetensors"
            model.write_bytes(b"host fixture; no CUDA")
            dataset = base / "dataset"
            dataset.mkdir()
            work = base / "r0-evidence"

            artifacts = generate(
                model=model, dataset=dataset, work_dir=work,
                external_models={}, cache_latents=True,
                cache_text_encoder_outputs=True,
            )
            manifest = json.loads(Path(artifacts["manifest"]).read_text(encoding="utf-8"))
            validated = r0.validate_manifest(manifest)
            self.assertEqual(validated["policy_hash"], r0._read_policy(Path(artifacts["policy"]))[1])
            self.assertTrue(Path(artifacts["policy"]).is_file())
            self.assertNotIn("--full_bf16", manifest["fresh_command"])
            self.assertIn("--cache_latents", manifest["fresh_command"])
            self.assertIn("--cache_text_encoder_outputs", manifest["fresh_command"])
            self.assertTrue(validated["policy"]["components"]["mmdit.modulation_norm.adapter"]["train"])
            with self.assertRaises(ValueError):
                generate(
                    model=model, dataset=dataset, work_dir=work,
                    external_models={}, cache_latents=True,
                    cache_text_encoder_outputs=True,
                )

    def test_generator_does_not_touch_production_qualification(self):
        source = (Path(__file__).resolve().parents[1] / "tools/prepare_sd3_component_baseline.py").read_text(encoding="utf-8")
        self.assertNotIn("FULL_BF16_BACKEND_QUALIFICATIONS", source)
        self.assertNotIn("full_bf16", source)



class SD3ComponentAdapterEvidenceTests(unittest.TestCase):
    def test_each_trained_group_requires_real_weight_advance(self):
        names = {
            "attention": "lora_unet_joint_blocks_0_x_block_attn_qkv.lora_up.weight",
            "mlp": "lora_unet_joint_blocks_0_x_block_mlp_fc1.lora_up.weight",
            "modulation": "lora_unet_joint_blocks_0_x_block_adaLN_modulation_1.lora_up.weight",
        }
        policy = SD3ComponentBaselineTests()._policy()
        policy["components"]["mmdit.modulation_norm.adapter"] = {
            "train": True, "optimizer_profile": "baseline", "learning_rate": 0.0001,
        }
        flags = {name: True for name in names.values()}
        summary = r0.summarize_component_weight_updates(flags, policy=policy)
        self.assertEqual(summary["changed_by_group"], {
            "attention": 1, "mlp": 1, "modulation": 1, "other": 0,
        })

        flags[names["mlp"]] = False
        with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "did not all advance"):
            r0.summarize_component_weight_updates(flags, policy=policy)

        flags[names["mlp"]] = True
        flags["lora_te1_block_0.lora_up.weight"] = True
        with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "non-MMDiT"):
            r0.summarize_component_weight_updates(flags, policy=policy)

        del flags["lora_te1_block_0.lora_up.weight"]
        flags[names["attention"]] = True
        flags["lora_unet_x_embedder_proj.lora_up.weight"] = True
        with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "frozen MMDiT Other"):
            r0.summarize_component_weight_updates(flags, policy=policy)
        del flags["lora_unet_x_embedder_proj.lora_up.weight"]

        flags[names["attention"]] = 1
        with self.assertRaisesRegex(r0.SD3ComponentBaselineError, "invalid weight"):
            r0.summarize_component_weight_updates(flags, policy=policy)


if __name__ == "__main__":
    unittest.main()
