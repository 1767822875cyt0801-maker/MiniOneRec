import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s4_sasrec_sid_valid_pipeline as s4


class PatchedS4Constants:
    def __enter__(self):
        self.old = {
            "EXPECTED_TRAIN_ROWS": s4.EXPECTED_TRAIN_ROWS,
            "EXPECTED_VALID_ROWS": s4.EXPECTED_VALID_ROWS,
            "EXPECTED_NUM_ITEMS": s4.EXPECTED_NUM_ITEMS,
            "EXPECTED_CATEGORY": s4.EXPECTED_CATEGORY,
        }
        s4.EXPECTED_TRAIN_ROWS = 2
        s4.EXPECTED_VALID_ROWS = 1
        s4.EXPECTED_NUM_ITEMS = 3
        s4.EXPECTED_CATEGORY = "Tiny"
        return self

    def __exit__(self, exc_type, exc, tb):
        for key, value in self.old.items():
            setattr(s4, key, value)


class TestS4SASRecSIDValidPipeline(unittest.TestCase):
    def write_json(self, path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_csv(self, path: Path, rows: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "user_id",
            "history_item_title",
            "item_title",
            "history_item_id",
            "item_id",
            "history_item_sid",
            "item_sid",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(rows):
                writer.writerow(
                    {
                        "user_id": f"u{idx}",
                        "history_item_title": "['h']",
                        "item_title": "target",
                        "history_item_id": "['1']",
                        "item_id": "2",
                        "history_item_sid": "['<a_0><b_0><c_0><d_0>']",
                        "item_sid": "<a_0><b_0><c_0><d_1>",
                    }
                )

    def make_sid_version(self, sid_root: Path, version: str, category: str, permute: bool = False):
        root = sid_root / version / category
        sids = [
            "<a_0><b_0><c_0><d_0>",
            "<a_0><b_0><c_0><d_1>",
            "<a_0><b_1><c_0><d_2>",
        ]
        items = ["1", "2", "3"]
        assigned = [sids[1], sids[2], sids[0]] if permute else sids
        item2sid = dict(zip(items, assigned))
        sid2items = {sid: [item] for item, sid in item2sid.items()}
        index = {item: s4.parse_sid_tokens(sid) for item, sid in item2sid.items()}
        self.write_csv(root / "train.csv", 2)
        self.write_csv(root / "valid.csv", 1)
        self.write_json(root / "item2sid.json", item2sid)
        self.write_json(root / "sid2items.json", sid2items)
        self.write_json(root / "valid_sid_set.json", sorted(sid2items))
        self.write_json(root / "index.json", index)
        (root / "info.txt").write_text(
            "".join(f"{sid}\ttitle {item}\t{item}\n" for item, sid in item2sid.items()),
            encoding="utf-8",
        )
        return root

    def make_formal_size_sid_version(self, sid_root: Path, version: str, category: str, permute: bool = False):
        root = sid_root / version / category
        item_count = 3686
        item_ids = [str(idx + 1) for idx in range(item_count)]
        sid_values = [
            f"<a_{idx % 512}><b_{(idx // 512) % 512}><c_{idx % 257}><d_{idx}>"
            for idx in range(item_count)
        ]
        if permute:
            assigned = sid_values[1:] + sid_values[:1]
        else:
            assigned = sid_values
        item2sid = dict(zip(item_ids, assigned))
        sid2items = {sid: [item] for item, sid in item2sid.items()}
        index = {item: s4.parse_sid_tokens(sid) for item, sid in item2sid.items()}
        self.write_csv(root / "train.csv", 36259)
        self.write_csv(root / "valid.csv", 4532)
        self.write_json(root / "item2sid.json", item2sid)
        self.write_json(root / "sid2items.json", sid2items)
        self.write_json(root / "valid_sid_set.json", sorted(sid2items))
        self.write_json(root / "index.json", index)
        (root / "info.txt").write_text(
            "".join(f"{sid}\ttitle {item}\t{item}\n" for item, sid in item2sid.items()),
            encoding="utf-8",
        )
        return root

    def make_config(self, tmp_path: Path, *, allow_missing_base_model: bool = False):
        category = "Tiny"
        sid_root = tmp_path / "data" / "Amazon" / "sid_versions"
        self.make_sid_version(sid_root, "cf_k512_dedup", category)
        self.make_sid_version(sid_root, "sasrec_v3_k512_dedup", category, permute=True)
        base_model = tmp_path / "base_model"
        base_model.mkdir()
        args = type(
            "Args",
            (),
            {
                "category": category,
                "sid_root": sid_root,
                "baseline_version": "cf_k512_dedup",
                "treatment_version": "sasrec_v3_k512_dedup",
                "base_model": base_model,
                "results_root": tmp_path / "results" / "s4_sasrec_sid_valid",
                "outputs_root": tmp_path / "outputs" / "s4_sasrec_sid_valid",
                "seed": 42,
                "config_mode": "smoke",
                "num_gpus": 1,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 4,
                "learning_rate": "2e-5",
                "cutoff_len": 128,
                "num_train_epochs": "1",
                "bf16": "False",
                "sample": 8,
                "gradient_checkpointing": "False",
                "num_beams": 4,
                "max_new_tokens": 6,
                "length_penalty": "0.0",
                "max_pred_sids": 4,
                "max_candidates": 10,
                "action": "dry-run",
                "allow_missing_base_model": allow_missing_base_model,
            },
        )()
        return s4.make_config(args)

    def make_args(self, tmp_path: Path, **overrides):
        category = overrides.pop("category", "Tiny")
        sid_root = tmp_path / "data" / "Amazon" / "sid_versions"
        if not (sid_root / "cf_k512_dedup" / category).exists():
            self.make_sid_version(sid_root, "cf_k512_dedup", category)
            self.make_sid_version(sid_root, "sasrec_v3_k512_dedup", category, permute=True)
        base_model = tmp_path / "base_model"
        base_model.mkdir(exist_ok=True)
        data = {
            "category": category,
            "sid_root": sid_root,
            "baseline_version": "cf_k512_dedup",
            "treatment_version": "sasrec_v3_k512_dedup",
            "base_model": base_model,
            "results_root": tmp_path / "results" / "s4_sasrec_sid_valid",
            "outputs_root": tmp_path / "outputs" / "s4_sasrec_sid_valid",
            "seed": 42,
            "config_mode": "smoke",
            "num_gpus": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": "2e-5",
            "cutoff_len": 128,
            "num_train_epochs": "1",
            "bf16": "False",
            "sample": None,
            "gradient_checkpointing": "False",
            "num_beams": 4,
            "max_new_tokens": None,
            "length_penalty": "0.0",
            "max_pred_sids": 4,
            "max_candidates": 10,
            "action": "dry-run",
            "allow_missing_base_model": False,
        }
        data.update(overrides)
        return type("Args", (), data)()

    def selected_dry_run_command(self, stdout: str, key: str) -> str:
        marker = f"DRY_RUN: would execute {key}:"
        lines = stdout.splitlines()
        for index, line in enumerate(lines):
            if line == marker and index + 1 < len(lines):
                return lines[index + 1]
        self.fail(f"Missing dry-run command marker: {marker}\n{stdout}")

    def test_parity_manifest_records_same_base_and_token_sets(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            manifest = s4.run_audit(config, write_manifest=True)
            self.assertEqual(manifest["base_model"]["path"], config.base_model.as_posix())
            self.assertTrue(manifest["special_tokens"]["token_string_sets_equal"])
            self.assertTrue(manifest["special_tokens"]["d_tokens_preserved"])
            self.assertEqual(manifest["contracts"]["test_read"], False)
            self.assertIn("runtime_environment", manifest)
            self.assertIn("preflight", manifest["runtime_environment"])
            self.assertEqual(manifest["training_contract"]["effective_train_sample"], 8)
            self.assertNotIn("lacks valid.csv", " ".join(manifest["unconfirmed_local_evidence"]))
            self.assertTrue((config.results_dir / "s4_parity_manifest.json").exists())

    def test_mode_defaults_make_smoke_bounded_and_generation_tokens_positive(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(Path(tmp))
            config = s4.make_config(args)
            self.assertEqual(config.sample, s4.DEFAULT_SMOKE_SAMPLE)
            self.assertEqual(config.max_new_tokens, s4.DEFAULT_MAX_NEW_TOKENS)
            plan = s4.build_command_plan(config)
            self.assertIn(str(s4.DEFAULT_SMOKE_SAMPLE), plan["baseline_train"])
            self.assertIn(str(s4.DEFAULT_SMOKE_SAMPLE), plan["treatment_train"])
            generation_cmd = plan["treatment_valid_generation"]
            self.assertGreater(int(generation_cmd[generation_cmd.index("--max_new_tokens") + 1]), 0)

    def test_smoke_rejects_unbounded_sample(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(Path(tmp), sample=-1)
            with self.assertRaises(ValueError):
                s4.make_config(args)

    def test_formal_default_uses_predefined_30k_budget(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            args = self.make_args(Path(tmp), config_mode="formal")
            config = s4.make_config(args)
            self.assertEqual(config.sample, s4.DEFAULT_FORMAL_SAMPLE)

    def test_dry_run_does_not_write_checkpoint_or_candidate_artifacts(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            s4.run_audit(config, write_manifest=False)
            self.assertFalse(config.output_dir.exists())
            self.assertFalse(config.prediction_file.exists())
            self.assertFalse(config.candidates_jsonl.exists())

    def test_missing_base_model_rejected_unless_explicitly_allowed(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            missing_config = replace(config, base_model=Path(tmp) / "missing")
            with self.assertRaises(FileNotFoundError):
                s4.run_audit(missing_config, write_manifest=False)
            allowed = replace(missing_config, allow_missing_base_model=True)
            manifest = s4.run_audit(allowed, write_manifest=False)
            self.assertFalse(manifest["base_model"]["exists"])

    def test_inaccessible_base_model_fingerprint_is_recorded(self):
        with mock.patch.object(Path, "exists", side_effect=PermissionError("blocked")):
            info = s4.path_fingerprint(Path("/root/autodl-tmp/models/Qwen2.5-0.5B"))
        self.assertFalse(info["exists"])
        self.assertIn("PermissionError", info["error"])

    def test_test_path_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                s4.require_valid_path(Path(tmp) / "test.csv", "eval")
            with self.assertRaises(ValueError):
                s4.reject_test_path(Path(tmp) / "data" / "test" / "x.csv", "train")

    def test_row_count_sid_uniqueness_and_d_token_integrity(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            report = s4.validate_sid_version(config.treatment, 2, 1)
            self.assertEqual(report["train_rows"], 2)
            self.assertEqual(report["valid_rows"], 1)
            self.assertEqual(report["unique_full_sids"], 3)
            self.assertGreater(report["d_token_count"], 0)

    def test_output_isolation_rejects_old_cf_namespace(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            bad = replace(config, results_dir=Path("results/cf_k512_dedup/valid"))
            with self.assertRaises(ValueError):
                s4.validate_output_isolation(bad)

    def test_command_plan_resume_overwrite_disabled_and_valid_only(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            plan = s4.build_command_plan(config)
            baseline_cmd = plan["baseline_train"]
            train_cmd = plan["treatment_train"]
            ignored_options = {"--train-csv", "--valid-csv", "--sid-index-path", "--output-dir"}
            for option in [
                "--base-model",
                "--num-gpus",
                "--per-device-train-batch-size",
                "--gradient-accumulation-steps",
                "--learning-rate",
                "--cutoff-len",
                "--num-train-epochs",
                "--bf16",
                "--sample",
                "--gradient-checkpointing",
            ]:
                self.assertNotIn(option, ignored_options)
                self.assertEqual(
                    baseline_cmd[baseline_cmd.index(option) + 1],
                    train_cmd[train_cmd.index(option) + 1],
                )
            self.assertIn("cf_k512_dedup", baseline_cmd[baseline_cmd.index("--output-dir") + 1])
            self.assertIn("sasrec_v3_k512_dedup", train_cmd[train_cmd.index("--output-dir") + 1])
            self.assertNotEqual(
                baseline_cmd[baseline_cmd.index("--output-dir") + 1],
                train_cmd[train_cmd.index("--output-dir") + 1],
            )
            self.assertNotIn("--resume-from-checkpoint", train_cmd)
            self.assertIn("--valid-csv", train_cmd)
            eval_cmd = plan["treatment_exact_candidate_eval"]
            self.assertIn("--eval-split", eval_cmd)
            self.assertIn("valid", eval_cmd)
            self.assertNotIn("test", " ".join(eval_cmd).lower())
            generation_cmd = plan["treatment_valid_generation"]
            self.assertGreater(int(generation_cmd[generation_cmd.index("--max_new_tokens") + 1]), 0)

    def test_preflight_failure_raises_before_torchrun(self):
        report = {
            "ok": False,
            "failures": [
                "symbol import failed: transformers.AutoModelForCausalLM: RecursionError: dtype repr",
                "symbol import failed: transformers.generation.utils.GenerationMixin: ImportError",
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "before torchrun"):
            s4.assert_runtime_environment_ok(report)

    def test_valid_candidate_schema_and_invalid_sid_stats(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self.make_config(tmp_path)
            artifacts = s4.stream_artifacts(config, "treatment")
            prediction_file = artifacts.prediction_file
            prediction_file.parent.mkdir(parents=True)
            prediction_file.write_text(
                json.dumps(
                    [
                        {
                            "output": "<a_0><b_1><c_0><d_2>",
                            "predict": ["<a_0><b_1><c_0><d_2>", "<bad_0>"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            cmd = s4.build_candidate_eval_command(config, artifacts)
            subprocess.run(cmd, cwd=ROOT, check=True)
            report = json.loads(artifacts.candidate_report.read_text(encoding="utf-8"))
            self.assertEqual(report["inputs"]["eval_split"], "valid")
            self.assertEqual(report["num_samples"], 1)
            self.assertEqual(report["sid_validity"]["invalid_sid_count"], 1)
            row = json.loads(artifacts.candidates_jsonl.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn("candidate_item_ids", row)
            self.assertIn("candidate_hit_rank_0_based", row)

    def test_summarize_candidate_report_adds_mrr(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self.make_config(tmp_path)
            config.candidates_jsonl.parent.mkdir(parents=True)
            config.candidates_jsonl.write_text(
                json.dumps({"candidate_hit_rank_0_based": 1}) + "\n"
                + json.dumps({"candidate_hit_rank_0_based": None}) + "\n",
                encoding="utf-8",
            )
            self.write_json(
                config.candidate_report,
                {
                    "output_jsonl": config.candidates_jsonl.as_posix(),
                    "inputs": {"eval_split": "valid"},
                    "num_samples": 2,
                    "item_level_before_rerank": {},
                },
            )
            summary = s4.summarize_candidate_report(config.candidate_report)
            self.assertEqual(summary["split"], "valid")
            self.assertEqual(summary["mrr"], 0.25)

    def test_stream_artifacts_isolate_baseline_and_treatment_outputs(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            baseline = s4.stream_artifacts(config, "baseline")
            treatment = s4.stream_artifacts(config, "treatment")
            self.assertNotEqual(baseline.results_dir, treatment.results_dir)
            self.assertIn("cf_k512_dedup", baseline.prediction_file.as_posix())
            self.assertIn("sasrec_v3_k512_dedup", treatment.prediction_file.as_posix())
            s4.validate_stream_output_isolation(baseline)
            s4.validate_stream_output_isolation(treatment)

    def test_prepare_candidates_creates_dirs_refuses_overwrite_and_limits_rows(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self.make_config(tmp_path)
            prep = s4.prepare_candidate_outputs(config, "baseline", overwrite=False, candidate_row_limit=1)
            artifacts = s4.stream_artifacts(
                config,
                "baseline",
                eval_csv=Path(prep["eval_csv"]),
                candidate_row_limit=1,
            )
            self.assertTrue(artifacts.generation_dir.is_dir())
            self.assertTrue(artifacts.candidates_dir.is_dir())
            self.assertTrue(artifacts.summary_dir.is_dir())
            self.assertTrue(artifacts.artifact_manifest.is_file())
            self.assertEqual(prep["expected_prediction_rows"], 1)
            self.assertIn("cf_k512_dedup", prep["commands"]["generation"])
            self.assertNotIn("sasrec_v3_k512_dedup", prep["commands"]["generation"])
            self.assertFalse(artifacts.prediction_file.exists())
            self.assertFalse(artifacts.candidates_jsonl.exists())
            self.assertFalse(artifacts.candidate_report.exists())
            artifacts.prediction_file.write_text("[]\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                s4.prepare_candidate_outputs(config, "baseline", overwrite=False, candidate_row_limit=1)
            s4.prepare_candidate_outputs(config, "baseline", overwrite=True, candidate_row_limit=1)

    def test_candidate_scope_isolates_bounded_and_full_valid_paths(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            bounded = s4.stream_artifacts(config, "baseline", candidate_row_limit=8)
            full = s4.stream_artifacts(config, "baseline", candidate_row_limit=0)
            self.assertIn("preflight_rows8", bounded.prediction_file.as_posix())
            self.assertIn("full_valid", full.prediction_file.as_posix())
            self.assertNotEqual(bounded.prediction_file, full.prediction_file)
            self.assertEqual(s4.candidate_scope(8), "preflight_rows8")
            self.assertEqual(s4.candidate_scope(0), "full_valid")

    def test_audit_classifies_empty_malformed_bounded_full_and_wrong_stream(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self.make_config(tmp_path)

            bounded_prep = s4.prepare_candidate_outputs(config, "baseline", overwrite=False, candidate_row_limit=1)
            bounded = s4.stream_artifacts(
                config,
                "baseline",
                eval_csv=Path(bounded_prep["eval_csv"]),
                candidate_row_limit=1,
            )
            bounded.prediction_file.write_text(json.dumps([{"predict": []}]), encoding="utf-8")
            bounded.candidates_jsonl.write_text(json.dumps({"candidate_hit_rank_0_based": 0}) + "\n", encoding="utf-8")
            self.write_json(
                bounded.candidate_report,
                {"output_jsonl": bounded.candidates_jsonl.as_posix(), "inputs": {"eval_split": "valid"}, "num_samples": 1},
            )
            s4.write_candidate_artifact_manifest(config, bounded, 1, complete=True, actual_prediction_rows=1)
            self.assertEqual(s4.audit_candidate_artifacts(config, bounded, 1)["classification"], "bounded")

            full = s4.stream_artifacts(config, "baseline", candidate_row_limit=0)
            full.generation_dir.mkdir(parents=True)
            full.candidates_dir.mkdir(parents=True)
            full.summary_dir.mkdir(parents=True)
            full.prediction_file.write_text(json.dumps([{"predict": []}]), encoding="utf-8")
            full.candidates_jsonl.write_text(json.dumps({"candidate_hit_rank_0_based": 0}) + "\n", encoding="utf-8")
            self.write_json(
                full.candidate_report,
                {"output_jsonl": full.candidates_jsonl.as_posix(), "inputs": {"eval_split": "valid"}, "num_samples": 1},
            )
            s4.write_candidate_artifact_manifest(config, full, 0, complete=True, actual_prediction_rows=1)
            self.assertEqual(s4.audit_candidate_artifacts(config, full, 0)["classification"], "full_valid")

            empty = s4.stream_artifacts(config, "treatment", candidate_row_limit=1)
            empty.generation_dir.mkdir(parents=True)
            empty.prediction_file.touch()
            self.assertEqual(s4.audit_candidate_artifacts(config, empty, 1)["classification"], "empty")

            malformed = s4.stream_artifacts(config, "treatment", candidate_row_limit=2)
            malformed.generation_dir.mkdir(parents=True)
            malformed.candidates_dir.mkdir(parents=True)
            malformed.summary_dir.mkdir(parents=True)
            malformed.prediction_file.write_text("{bad json", encoding="utf-8")
            malformed.candidates_jsonl.write_text("{}\n", encoding="utf-8")
            malformed.candidate_report.write_text("{}", encoding="utf-8")
            s4.write_candidate_artifact_manifest(config, malformed, 2, complete=True, actual_prediction_rows=2)
            self.assertEqual(s4.audit_candidate_artifacts(config, malformed, 2)["classification"], "malformed")

            wrong = s4.stream_artifacts(config, "treatment", candidate_row_limit=3)
            wrong.generation_dir.mkdir(parents=True)
            wrong.candidates_dir.mkdir(parents=True)
            wrong.summary_dir.mkdir(parents=True)
            wrong.prediction_file.write_text(json.dumps([{"predict": []}]), encoding="utf-8")
            wrong.candidates_jsonl.write_text("{}\n", encoding="utf-8")
            wrong.candidate_report.write_text("{}", encoding="utf-8")
            manifest = s4.candidate_manifest_payload(config, wrong, 3, complete=True, actual_prediction_rows=1)
            manifest["stream"] = "baseline"
            self.write_json(wrong.artifact_manifest, manifest)
            self.assertEqual(s4.audit_candidate_artifacts(config, wrong, 3)["classification"], "wrong_stream")

    def test_bounded_artifact_does_not_block_full_valid_prepare(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            s4.prepare_candidate_outputs(config, "baseline", overwrite=False, candidate_row_limit=1)
            full_prep = s4.prepare_candidate_outputs(config, "baseline", overwrite=False, candidate_row_limit=0)
            self.assertIn("full_valid", full_prep["prediction_file"])

    def test_validate_predictions_requires_file_and_row_alignment(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = self.make_config(tmp_path)
            prep = s4.prepare_candidate_outputs(config, "treatment", overwrite=False, candidate_row_limit=1)
            prediction_file = Path(prep["prediction_file"])
            eval_csv = Path(prep["eval_csv"])
            with self.assertRaises(FileNotFoundError):
                s4.validate_predictions(prediction_file, eval_csv)
            prediction_file.write_text(json.dumps([{"predict": []}]), encoding="utf-8")
            result = s4.validate_predictions(prediction_file, eval_csv)
            self.assertTrue(result["row_alignment_ok"])
            prediction_file.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaises(ValueError):
                s4.validate_predictions(prediction_file, eval_csv)

    def test_recommended_token_budget_covers_sid_newline_and_eos(self):
        with PatchedS4Constants(), tempfile.TemporaryDirectory() as tmp:
            config = self.make_config(Path(tmp))
            info = s4.recommended_max_new_tokens_from_sid_lengths([
                config.baseline.info_txt,
                config.treatment.info_txt,
            ])
            self.assertEqual(info["max_sid_tokens"], 4)
            self.assertEqual(info["newline_tokens"], 1)
            self.assertEqual(info["eos_tokens"], 1)
            self.assertEqual(info["recommended_max_new_tokens"], 6)

    def test_constrained_processor_has_terminal_eos_defense(self):
        source = (ROOT / "LogitProcessor.py").read_text(encoding="utf-8")
        self.assertIn("self.eos_token_id is not None and self.eos_token_id in hash_key", source)
        self.assertIn("mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = 0", source)

    def test_shell_dry_run_prints_valid_commands_without_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            category = "Industrial_and_Scientific"
            sid_root = tmp_path / "data" / "Amazon" / "sid_versions"
            self.make_formal_size_sid_version(sid_root, "cf_k512_dedup", category)
            self.make_formal_size_sid_version(sid_root, "sasrec_v3_k512_dedup", category, permute=True)
            base_model = tmp_path / "base_model"
            base_model.mkdir()
            results_root = tmp_path / "results" / "s4_sasrec_sid_valid"
            outputs_root = tmp_path / "outputs" / "s4_sasrec_sid_valid"
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON": sys.executable,
                    "STAGE": "parity",
                    "DRY_RUN": "1",
                    "CATEGORY": category,
                    "BASE_MODEL": base_model.as_posix(),
                    "SID_ROOT": sid_root.as_posix(),
                    "RESULTS_ROOT": results_root.as_posix(),
                    "OUTPUTS_ROOT": outputs_root.as_posix(),
                    "CONFIG_MODE": "smoke",
                    "PER_DEVICE_TRAIN_BATCH_SIZE": "2",
                    "GRADIENT_ACCUMULATION_STEPS": "4",
                    "CUTOFF_LEN": "128",
                    "NUM_BEAMS": "4",
                    "MAX_PRED_SIDS": "4",
                    "MAX_CANDIDATES": "10",
                }
            )
            proc = subprocess.run(
                ["bash", "scripts/run_s4_sasrec_sid_valid.sh"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("DRY_RUN", proc.stdout)
            self.assertIn("treatment_train", proc.stdout)
            self.assertIn("--sample 2000", proc.stdout)
            self.assertIn("--max_new_tokens 6", proc.stdout)
            self.assertFalse(outputs_root.exists())

    def test_shell_train_stream_baseline_dry_run_selects_baseline_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            category = "Industrial_and_Scientific"
            sid_root = tmp_path / "data" / "Amazon" / "sid_versions"
            self.make_formal_size_sid_version(sid_root, "cf_k512_dedup", category)
            self.make_formal_size_sid_version(sid_root, "sasrec_v3_k512_dedup", category, permute=True)
            base_model = tmp_path / "base_model"
            base_model.mkdir()
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON": sys.executable,
                    "STAGE": "train",
                    "TRAIN_STREAM": "baseline",
                    "DRY_RUN": "1",
                    "CATEGORY": category,
                    "BASE_MODEL": base_model.as_posix(),
                    "SID_ROOT": sid_root.as_posix(),
                    "RESULTS_ROOT": (tmp_path / "results" / "s4_sasrec_sid_valid").as_posix(),
                    "OUTPUTS_ROOT": (tmp_path / "outputs" / "s4_sasrec_sid_valid").as_posix(),
                    "CONFIG_MODE": "smoke",
                    "PER_DEVICE_TRAIN_BATCH_SIZE": "2",
                    "GRADIENT_ACCUMULATION_STEPS": "4",
                    "CUTOFF_LEN": "128",
                    "SAMPLE": "8",
                    "NUM_BEAMS": "4",
                    "MAX_NEW_TOKENS": "6",
                    "MAX_PRED_SIDS": "4",
                    "MAX_CANDIDATES": "10",
                }
            )
            proc = subprocess.run(
                ["bash", "scripts/run_s4_sasrec_sid_valid.sh"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("DRY_RUN: would execute baseline_train", proc.stdout)
            self.assertIn("outputs/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup", proc.stdout)

    def test_shell_candidate_baseline_dry_run_selects_only_cf_generation_and_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            category = "Industrial_and_Scientific"
            sid_root = tmp_path / "data" / "Amazon" / "sid_versions"
            self.make_formal_size_sid_version(sid_root, "cf_k512_dedup", category)
            self.make_formal_size_sid_version(sid_root, "sasrec_v3_k512_dedup", category, permute=True)
            base_model = tmp_path / "base_model"
            base_model.mkdir()
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON": sys.executable,
                    "STAGE": "candidates",
                    "TRAIN_STREAM": "baseline",
                    "DRY_RUN": "1",
                    "CATEGORY": category,
                    "BASE_MODEL": base_model.as_posix(),
                    "SID_ROOT": sid_root.as_posix(),
                    "RESULTS_ROOT": (tmp_path / "results" / "s4_sasrec_sid_valid").as_posix(),
                    "OUTPUTS_ROOT": (tmp_path / "outputs" / "s4_sasrec_sid_valid").as_posix(),
                    "CONFIG_MODE": "smoke",
                    "PER_DEVICE_TRAIN_BATCH_SIZE": "2",
                    "GRADIENT_ACCUMULATION_STEPS": "4",
                    "CUTOFF_LEN": "128",
                    "NUM_BEAMS": "4",
                    "MAX_PRED_SIDS": "4",
                    "MAX_CANDIDATES": "10",
                }
            )
            proc = subprocess.run(
                ["bash", "scripts/run_s4_sasrec_sid_valid.sh"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            gen_cmd = self.selected_dry_run_command(proc.stdout, "baseline_valid_generation")
            eval_cmd = self.selected_dry_run_command(proc.stdout, "baseline_exact_candidate_eval")
            self.assertIn("cf_k512_dedup", gen_cmd)
            self.assertIn("cf_k512_dedup", eval_cmd)
            self.assertNotIn("sasrec_v3_k512_dedup", gen_cmd)
            self.assertNotIn("sasrec_v3_k512_dedup", eval_cmd)
            self.assertNotIn("DRY_RUN: would execute treatment_valid_generation", proc.stdout)

    def test_shell_candidate_treatment_dry_run_selects_only_sasrec_generation_and_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            category = "Industrial_and_Scientific"
            sid_root = tmp_path / "data" / "Amazon" / "sid_versions"
            self.make_formal_size_sid_version(sid_root, "cf_k512_dedup", category)
            self.make_formal_size_sid_version(sid_root, "sasrec_v3_k512_dedup", category, permute=True)
            base_model = tmp_path / "base_model"
            base_model.mkdir()
            env = os.environ.copy()
            env.update(
                {
                    "PYTHON": sys.executable,
                    "STAGE": "candidates",
                    "TRAIN_STREAM": "treatment",
                    "DRY_RUN": "1",
                    "CATEGORY": category,
                    "BASE_MODEL": base_model.as_posix(),
                    "SID_ROOT": sid_root.as_posix(),
                    "RESULTS_ROOT": (tmp_path / "results" / "s4_sasrec_sid_valid").as_posix(),
                    "OUTPUTS_ROOT": (tmp_path / "outputs" / "s4_sasrec_sid_valid").as_posix(),
                    "CONFIG_MODE": "smoke",
                    "PER_DEVICE_TRAIN_BATCH_SIZE": "2",
                    "GRADIENT_ACCUMULATION_STEPS": "4",
                    "CUTOFF_LEN": "128",
                    "NUM_BEAMS": "4",
                    "MAX_PRED_SIDS": "4",
                    "MAX_CANDIDATES": "10",
                }
            )
            proc = subprocess.run(
                ["bash", "scripts/run_s4_sasrec_sid_valid.sh"],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            gen_cmd = self.selected_dry_run_command(proc.stdout, "treatment_valid_generation")
            eval_cmd = self.selected_dry_run_command(proc.stdout, "treatment_exact_candidate_eval")
            self.assertIn("sasrec_v3_k512_dedup", gen_cmd)
            self.assertIn("sasrec_v3_k512_dedup", eval_cmd)
            self.assertNotIn("cf_k512_dedup", gen_cmd)
            self.assertNotIn("cf_k512_dedup", eval_cmd)
            self.assertNotIn("DRY_RUN: would execute baseline_valid_generation", proc.stdout)


if __name__ == "__main__":
    unittest.main()
