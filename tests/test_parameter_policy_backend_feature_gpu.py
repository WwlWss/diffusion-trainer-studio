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
    BACKEND_FEATURE_MANIFEST_SCHEMA,
    CHECKPOINT_PROGRESS_SCHEMA,
    SD_LORA_FULL_BF16_CASE_IDS,
    SD_LORA_FULL_BF16_EVIDENCE_ID,
    SHARED_FULL_BF16_OPTIMIZER_EVIDENCE_ID,
    BackendFeatureGpuMatrixError,
    backend_feature_qualification_snapshot,
    compare_backend_checkpoint_contracts,
    compare_checkpoint_progress,
    load_backend_feature_manifest,
    summarize_sd_lora_full_bf16_promotion,
    validate_checkpoint_progress,
    validate_full_bf16_checkpoint_manifest,
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
        "optimizer_profiles": profiles,
        "optimizers": [
            {
                "profile_name": profile_name,
                "optimizer_type": optimizer_type,
            }
            for profile_name, optimizer_type in profiles.items()
        ],
    }
    return {
        "case_id": case_id,
        "train_type": train_type,
        "feature": feature,
        "status": status,
        "fresh_checkpoint_manifest": copy.deepcopy(manifest),
        "resume_checkpoint_manifest": copy.deepcopy(manifest),
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

    def test_complete_d1_matrix_is_promotion_eligible(self):
        summary = summarize_sd_lora_full_bf16_promotion(
            self._passing_rows(),
            self._snapshot(),
        )
        self.assertEqual(summary["id"], SD_LORA_FULL_BF16_EVIDENCE_ID)
        self.assertEqual(
            SD_LORA_FULL_BF16_EVIDENCE_ID,
            "phase-d1:backend:sd-lora:full-bf16:v1",
        )
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["promotion_eligible"])
        self.assertTrue(summary["backend_qualification_eligible"])
        self.assertTrue(summary["backend_row_match"])
        self.assertTrue(summary["shared_optimizer_authority_match"])
        self.assertEqual(summary["case_contract_errors"], [])
        self.assertEqual(summary["unexpected_backend_promotions"], [])
        self.assertEqual(summary["unexpected_optimizer_promotions"], [])
        self.assertFalse(summary["production_qualification_mutated"])

    def test_missing_or_failed_case_blocks_promotion(self):
        rows = self._passing_rows()
        summary = summarize_sd_lora_full_bf16_promotion(
            rows[:-1],
            self._snapshot(),
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["missing_cases"], [SD_LORA_FULL_BF16_CASE_IDS[1]])

        failed = self._passing_rows()
        failed[1]["status"] = "fail"
        summary = summarize_sd_lora_full_bf16_promotion(
            failed,
            self._snapshot(),
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["failed_cases"], [SD_LORA_FULL_BF16_CASE_IDS[1]])

    def test_case_identity_and_optimizer_topology_are_verified(self):
        wrong_identity = self._passing_rows()
        wrong_identity[0]["train_type"] = "flux-lora"
        summary = summarize_sd_lora_full_bf16_promotion(
            wrong_identity,
            self._snapshot(),
        )
        self.assertEqual(summary["status"], "fail")
        self.assertTrue(summary["case_contract_errors"])

        wrong_topology = self._passing_rows()
        wrong_topology[1] = _d1_case_row(
            SD_LORA_FULL_BF16_CASE_IDS[1],
            optimizer_types=("Muon",),
        )
        summary = summarize_sd_lora_full_bf16_promotion(
            wrong_topology,
            self._snapshot(),
        )
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
                summary = summarize_sd_lora_full_bf16_promotion(
                    self._passing_rows(),
                    snapshot,
                )
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
                summary = summarize_sd_lora_full_bf16_promotion(
                    self._passing_rows(),
                    snapshot,
                )
                self.assertEqual(summary["status"], "fail")
                self.assertFalse(summary["shared_optimizer_authority_match"])

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
        summary = summarize_sd_lora_full_bf16_promotion(
            self._passing_rows(),
            snapshot,
        )
        self.assertEqual(summary["status"], "fail")
        self.assertEqual(summary["unexpected_backend_promotions"], ["sdxl-lora"])
        self.assertEqual(summary["unexpected_optimizer_promotions"], ["Lion"])


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
        ):
            self.assertIn(f'"{package}"', source)

    def test_runner_records_redacted_command_contract(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("def _redacted_argv(", source)
        self.assertIn("_SECRET_OPTION_MARKERS", source)
        self.assertIn('"<redacted>"', source)
        self.assertIn('"command_contract": _command_contract(case)', source)
        self.assertIn('"environment_keys": sorted(case["environment"])', source)

    def test_runner_requires_external_manifest_output_and_two_checkpoints(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('field="Backend feature --manifest"', source)
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

    def test_evidence_schema_is_stable(self):
        self.assertEqual(
            BACKEND_FEATURE_EVIDENCE_SCHEMA,
            "dts.parameter-policy.backend-feature-gpu-matrix",
        )


if __name__ == "__main__":
    unittest.main()
