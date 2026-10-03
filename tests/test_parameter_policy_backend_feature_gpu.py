from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.parameter_policy_backend_feature_gpu_support import (
    BACKEND_FEATURE_EVIDENCE_SCHEMA,
    BACKEND_FEATURE_MANIFEST_SCHEMA,
    CHECKPOINT_PROGRESS_SCHEMA,
    BackendFeatureGpuMatrixError,
    compare_backend_checkpoint_contracts,
    compare_checkpoint_progress,
    load_backend_feature_manifest,
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
            "python",
            "scripts/dev/flux_train.py",
            "--max_train_steps",
            "2",
            "--save_every_n_steps",
            "1",
            "--save_state",
        ],
        "resume_command": [
            "python",
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
) -> dict:
    return {
        "schema": CHECKPOINT_PROGRESS_SCHEMA,
        "version": 1,
        "train_type": train_type,
        "policy_hash": policy_hash,
        "runtime_topology_fingerprint": topology,
        "execution_signature": execution_signature,
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
                "python",
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
                "python",
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
                "python",
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
                "python",
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
            case["fresh_command"][1] = str(external_script)
            with self.assertRaisesRegex(
                BackendFeatureGpuMatrixError,
                "existing Python entrypoint from this repository",
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

    def test_checkpoint_progress_proves_logical_step_one_to_two(self):
        manifest = _checkpoint()
        fresh = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=1,
            scheduler_step_count=2,
            scheduler_last_epoch=1,
        )
        resumed = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=2,
            scheduler_step_count=3,
            scheduler_last_epoch=2,
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

    def test_checkpoint_progress_rejects_fake_or_skipped_resume(self):
        manifest = _checkpoint()
        fresh = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=1,
            scheduler_step_count=2,
            scheduler_last_epoch=1,
        )
        fake = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=1,
            scheduler_step_count=2,
            scheduler_last_epoch=1,
        )
        with self.assertRaisesRegex(
            BackendFeatureGpuMatrixError,
            "logical optimizer step 2",
        ):
            compare_checkpoint_progress(fresh, fake)

        skipped = _progress(
            execution_signature=manifest["execution_signature"],
            optimizer_step=3,
            scheduler_step_count=4,
            scheduler_last_epoch=3,
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


class BackendFeatureRunnerSourceTests(unittest.TestCase):
    def test_runner_checks_exact_clean_head_before_torch_import(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('"--expected-commit"', source)
        self.assertIn('"status", "--porcelain=v1", "--untracked-files=all"', source)
        self.assertLess(
            source.index("_BOOTSTRAP_COMMIT = _assert_clean_head"),
            source.index("import torch"),
        )

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

    def test_runner_never_promotes_or_bypasses_production_qualification(self):
        source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("FULL_BF16_BACKEND_QUALIFICATIONS", source)
        self.assertNotIn("FULL_BF16_OPTIMIZER_QUALIFICATIONS", source)
        self.assertNotIn("temporary_execution_qualification", source)
        self.assertNotIn("DTS_FULL_BF16_BYPASS", source)
        self.assertIn('"backend_qualification_eligible": False', source)
        self.assertIn('"production_qualification_mutated": False', source)

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
