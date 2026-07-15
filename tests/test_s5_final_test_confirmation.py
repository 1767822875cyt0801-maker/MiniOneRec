import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s5_auxiliary_fusion_conversion as s5  # noqa: E402
import s5_final_test_confirmation as s5final  # noqa: E402


def strings_in(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings_in(item)
    elif isinstance(value, str):
        yield value


def row(row_index, target, items):
    return {
        "row_index": str(row_index),
        "target_item_id": str(target),
        "history_item_id": ["10", "11"],
        "candidate_item_ids": [str(item) for item in items],
        "candidate_details": [
            {"item_id": str(item), "source_type": "exact", "expansion_level": 3, "bucket_size": 1}
            for item in items
        ],
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_test_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["item_id", "history_item_id", "item_sid", "history_item_sid"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx in range(rows):
            writer.writerow({
                "item_id": str(idx + 1),
                "history_item_id": f"[{idx}]",
                "item_sid": f"<a_{idx}><b_0><c_0>",
                "history_item_sid": f"['<a_{idx}><b_0><c_0>']",
            })


def tiny_resume_config(output_root="out"):
    return {
        "selected_candidate_policy": "source_aware_rrf_lam0.75_bonus0.01",
        "policy": "source_aware_rrf",
        "lambda_sasrec": 0.75,
        "source_bonus": 0.01,
        "ranker_compatibility_mode": "source_independent_projection",
        "candidate_mode": "exact_full_sid_only",
        "num_beams": 50,
        "max_new_tokens": 6,
        "max_pred_sids": 50,
        "max_candidates": 1000,
        "seed": 42,
        "selected_config_hash": "selected",
        "projection_contract_hash": "projection",
        "ranker": {
            "model_path": "ranker/model.json",
            "model_sha256": "ranker",
            "feature_schema_hash": "schema",
            "normalization_metadata_hash": "norm",
        },
        "baseline_stream": {"sid_version": "cf_k512_dedup"},
        "auxiliary_stream": {"sid_version": "sasrec_v3_k512_dedup"},
        "output_root": output_root,
    }


def make_partial_tree(root, cfg, rows=2):
    cf_paths = s5final.stream_test_paths(cfg, "baseline_stream")
    sas_paths = s5final.stream_test_paths(cfg, "auxiliary_stream")
    for paths in [cf_paths, sas_paths]:
        for key in ["train_csv", "eval_csv"]:
            write_test_csv(root / paths[key], rows)
        for key in ["info_file", "item2sid", "sid2items", "valid_sid_set"]:
            target = root / paths[key]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}\n", encoding="utf-8")
        (root / paths["checkpoint"]).mkdir(parents=True, exist_ok=True)
    write_json(root / cf_paths["predictions"], [{"predict": [], "output": "x"} for _ in range(rows)])
    write_jsonl(root / cf_paths["candidates"], [
        {"row_index": idx, "target_item_id": str(idx), "candidate_item_ids": [str(idx)]}
        for idx in range(rows)
    ])
    write_json(root / cf_paths["candidate_report"], {
        "num_samples": rows,
        "candidate_count": {"min": 1, "max": 1, "mean": 1.0},
        "sid_validity": {"invalid_sid_count": 0, "total_pred_sid_count": rows},
    })
    return cf_paths, sas_paths


def make_resume_manifest(root, cf_paths, bad_hash=False):
    hashes = {
        "cf_predictions": s5final.file_sha256(root / cf_paths["predictions"]),
        "cf_candidates": s5final.file_sha256(root / cf_paths["candidates"]),
        "cf_candidate_report": s5final.file_sha256(root / cf_paths["candidate_report"]),
    }
    if bad_hash:
        hashes["cf_predictions"] = "0" * 64
    path = root / "resume_manifest.json"
    write_json(path, {"expected_cf_artifact_sha256": hashes})
    return Path("resume_manifest.json")


class S5FinalTestConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = s5final.load_config(s5final.CONFIG_PATH_REL)

    def test_selected_config_is_immutable(self):
        s5final.verify_immutable_config(self.config)
        bad = copy.deepcopy(self.config)
        bad["lambda_sasrec"] = 0.5
        with self.assertRaises(ValueError):
            s5final.verify_immutable_config(bad)

    def test_feature_order_and_ranker_dimensions_match(self):
        model = s5final.load_model()
        contract = s5final.feature_contract(model)
        self.assertEqual(contract["ordered_feature_names"], model["feature_names"])
        self.assertEqual(contract["dimension_count"], len(model["feature_names"]))
        self.assertEqual(len(model["feature_stats"]["mean"]), contract["dimension_count"])
        self.assertEqual(len(model["feature_stats"]["std"]), contract["dimension_count"])

    def test_source_specific_normalized_features_zero_and_others_unchanged(self):
        model = s5final.load_model()
        raw = {name: float(idx + 1) for idx, name in enumerate(model["feature_names"])}
        projected = s5final.project_normalized_features(raw, model)
        means = model["feature_stats"]["mean"]
        stds = model["feature_stats"]["std"]
        for idx, name in enumerate(model["feature_names"]):
            if name in s5final.SOURCE_SPECIFIC_FEATURES_ZEROED:
                self.assertEqual(projected[idx], 0.0)
            else:
                self.assertEqual(projected[idx], (raw[name] - means[idx]) / stds[idx])

    def test_relative_path_resolution_for_distinct_project_roots(self):
        rel = Path("nested/artifact.txt")
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            left_root = Path(left)
            right_root = Path(right)
            (left_root / rel.parent).mkdir(parents=True)
            (right_root / rel.parent).mkdir(parents=True)
            (left_root / rel).write_text("same-content", encoding="utf-8")
            (right_root / rel).write_text("same-content", encoding="utf-8")
            self.assertEqual(s5final.resolve_repo_path(rel, left_root), left_root / rel)
            self.assertEqual(s5final.resolve_repo_path(rel, right_root), right_root / rel)
            self.assertEqual(s5final.file_sha256_rel(rel, left_root), s5final.file_sha256_rel(rel, right_root))

    def test_path_traversal_and_absolute_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                s5final.resolve_repo_path("../escape.txt", root)
            with self.assertRaises(ValueError):
                s5final.resolve_repo_path(Path("/tmp/escape.txt"), root)

    def test_command_plan_has_no_local_absolute_paths(self):
        plan = s5final.command_plan(self.config)
        text = json.dumps(plan, sort_keys=True)
        self.assertNotIn("/" + "home/", text)
        self.assertNotIn(str(ROOT), text)
        self.assertNotIn("/root/autodl-tmp", text)
        self.assertIn("${PROJECT_ROOT:?set PROJECT_ROOT}", text)
        self.assertIn("${BASE_MODEL:?set BASE_MODEL}", text)

    def test_frozen_config_contains_no_machine_absolute_paths(self):
        for value in strings_in(self.config):
            self.assertNotIn("/" + "home/", value)
            self.assertNotIn("/root/autodl-tmp", value)
            if value.startswith("/"):
                self.fail(f"config contains absolute path: {value}")

    def test_hash_mismatch_and_missing_file_fail_fast(self):
        bad_hash = copy.deepcopy(self.config)
        bad_hash["ranker"]["model_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            s5final.verify_release_hashes(bad_hash)

        missing = copy.deepcopy(self.config)
        missing["ranker"]["model_path"] = "results/missing/model.json"
        with self.assertRaises(FileNotFoundError):
            s5final.verify_release_hashes(missing)

    def test_dry_run_guards_final_test_execution(self):
        with mock.patch.dict("os.environ", {"DRY_RUN": "1", "CONFIRM_FINAL_TEST": "1"}, clear=False):
            with self.assertRaises(SystemExit):
                s5final.require_final_confirmation()
        with mock.patch.dict("os.environ", {"DRY_RUN": "0", "CONFIRM_FINAL_TEST": "0"}, clear=False):
            with self.assertRaises(SystemExit):
                s5final.require_final_confirmation()

    def test_overwrite_protection_uses_project_root_resolution(self):
        cfg = copy.deepcopy(self.config)
        cfg["output_root"] = "out"
        cfg["baseline_stream"]["sid_version"] = "cf"
        cfg["auxiliary_stream"]["sid_version"] = "sas"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "out/cf/generation/predictions.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("[]", encoding="utf-8")
            with mock.patch.object(s5final, "PROJECT_ROOT", root):
                with self.assertRaises(FileExistsError):
                    s5final.protect_final_outputs(cfg, allow_overwrite=False)

    def test_prediction_row_alignment_uses_resolved_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pred = Path("predictions.json")
            eval_csv = Path("eval.csv")
            (root / pred).write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
            with open(root / eval_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["item_id"])
                writer.writeheader()
                writer.writerow({"item_id": "1"})
                writer.writerow({"item_id": "2"})
            with mock.patch.object(s5final, "PROJECT_ROOT", root):
                self.assertEqual(s5final.validate_prediction_rows(pred, eval_csv)["actual"], 2)
                (root / pred).write_text(json.dumps([{"a": 1}]), encoding="utf-8")
                with self.assertRaises(ValueError):
                    s5final.validate_prediction_rows(pred, eval_csv)

    def test_candidate_provenance_and_item_dedup_are_deterministic(self):
        cf_rows = [row(0, "2", ["1", "2", "3"])]
        sas_rows = [row(0, "2", ["3", "2", "4"])]
        first = s5.provenance_rows(s5.build_samples(cf_rows, sas_rows, {"0": "valid_select"}))
        second = s5.provenance_rows(s5.build_samples(cf_rows, sas_rows, {"0": "valid_select"}))
        self.assertEqual(first, second)
        self.assertEqual(len([item for item in first if item["item_id"] == "3"]), 1)

    def test_missing_sasrec_test_csv_fails_before_resume_commands(self):
        cfg = tiny_resume_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cf_paths, sas_paths = make_partial_tree(root, cfg)
            (root / sas_paths["eval_csv"]).unlink()
            with (
                mock.patch.object(s5final, "PROJECT_ROOT", root),
                mock.patch.object(s5final, "verify_release_hashes"),
                mock.patch.object(s5final, "file_sha256_rel", return_value="hash"),
            ):
                with self.assertRaises(FileNotFoundError):
                    s5final.input_inventory_preflight(cfg, expected_rows=2)

    def test_baseline_treatment_test_row_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cf = Path("cf/test.csv")
            sas = Path("sas/test.csv")
            write_test_csv(root / cf, 2)
            write_test_csv(root / sas, 2)
            with mock.patch.object(s5final, "PROJECT_ROOT", root):
                self.assertTrue(s5final.validate_test_csv_pair(cf, sas, expected_rows=2)["item_id_aligned"])
                with open(root / sas, encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                rows[1]["item_id"] = "changed"
                with open(root / sas, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    s5final.validate_test_csv_pair(cf, sas, expected_rows=2)

    def test_cf_complete_sasrec_missing_partial_state_accepted(self):
        cfg = tiny_resume_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cf_paths, _ = make_partial_tree(root, cfg)
            manifest = make_resume_manifest(root, cf_paths)
            with mock.patch.object(s5final, "PROJECT_ROOT", root), mock.patch.object(s5final, "verify_release_hashes"):
                audit = s5final.partial_state_audit(cfg, manifest, expected_rows=2, require_expected_hashes=True)
                self.assertEqual(audit["state"], "cf_complete_sasrec_missing")

    def test_cf_artifact_missing_rejected(self):
        cfg = tiny_resume_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cf_paths, _ = make_partial_tree(root, cfg)
            manifest = make_resume_manifest(root, cf_paths)
            (root / cf_paths["candidates"]).unlink()
            with mock.patch.object(s5final, "PROJECT_ROOT", root), mock.patch.object(s5final, "verify_release_hashes"):
                with self.assertRaises(FileNotFoundError):
                    s5final.partial_state_audit(cfg, manifest, expected_rows=2, require_expected_hashes=True)

    def test_cf_artifact_hash_mismatch_rejected(self):
        cfg = tiny_resume_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cf_paths, _ = make_partial_tree(root, cfg)
            manifest = make_resume_manifest(root, cf_paths, bad_hash=True)
            with mock.patch.object(s5final, "PROJECT_ROOT", root), mock.patch.object(s5final, "verify_release_hashes"):
                with self.assertRaises(ValueError):
                    s5final.partial_state_audit(cfg, manifest, expected_rows=2, require_expected_hashes=True)

    def test_sasrec_partial_artifact_present_rejected(self):
        cfg = tiny_resume_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cf_paths, sas_paths = make_partial_tree(root, cfg)
            manifest = make_resume_manifest(root, cf_paths)
            write_json(root / sas_paths["predictions"], [{"predict": []}])
            with mock.patch.object(s5final, "PROJECT_ROOT", root), mock.patch.object(s5final, "verify_release_hashes"):
                with self.assertRaises(FileExistsError):
                    s5final.partial_state_audit(cfg, manifest, expected_rows=2, require_expected_hashes=True)

    def test_final_metrics_already_present_rejected_and_second_resume_blocked(self):
        cfg = tiny_resume_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cf_paths, _ = make_partial_tree(root, cfg)
            manifest = make_resume_manifest(root, cf_paths)
            write_json(root / "out/final_test_metrics.json", {"complete": True})
            with mock.patch.object(s5final, "PROJECT_ROOT", root), mock.patch.object(s5final, "verify_release_hashes"):
                with self.assertRaises(FileExistsError):
                    s5final.partial_state_audit(cfg, manifest, expected_rows=2, require_expected_hashes=True)

    def test_resume_command_plan_excludes_cf_generation_and_starts_at_sasrec(self):
        plan = s5final.resume_command_plan(self.config)
        text = json.dumps(plan, sort_keys=True)
        self.assertNotIn("cf_test_generation", plan["commands"])
        self.assertNotIn("cf_test_candidate_eval", plan["commands"])
        self.assertIn("sasrec_test_generation", plan["commands"])
        self.assertLess(plan["sequence"].index("sasrec_test_generation"), plan["sequence"].index("sasrec_test_candidate_eval"))
        self.assertIn("sasrec_test_generation", text)

    def test_resume_confirmation_required(self):
        with mock.patch.dict("os.environ", {"DRY_RUN": "1", "CONFIRM_FINAL_TEST_RESUME": "1"}, clear=False):
            with self.assertRaises(SystemExit):
                s5final.require_final_resume_confirmation()
        with mock.patch.dict("os.environ", {"DRY_RUN": "0", "CONFIRM_FINAL_TEST_RESUME": "0"}, clear=False):
            with self.assertRaises(SystemExit):
                s5final.require_final_resume_confirmation()

    def test_overwrite_cannot_bypass_final_or_resume_contract(self):
        cfg = tiny_resume_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_partial_tree(root, cfg)
            with mock.patch.object(s5final, "PROJECT_ROOT", root):
                with self.assertRaises(FileExistsError):
                    s5final.protect_final_outputs(cfg, allow_overwrite=True)
                with mock.patch.dict("os.environ", {"DRY_RUN": "0", "CONFIRM_FINAL_TEST_RESUME": "1"}, clear=False):
                    with self.assertRaises(FileExistsError):
                        s5final.resume_final_test_after_cf(cfg, Path("missing.json"), allow_overwrite=True)

    def test_resume_template_does_not_select_parameters_from_test_metrics(self):
        template = s5final.make_partial_resume_manifest_template(self.config)
        self.assertTrue(template["no_parameter_change"])
        self.assertEqual(template["frozen_parameters"]["lambda_sasrec"], 0.75)
        self.assertEqual(template["frozen_parameters"]["source_bonus"], 0.01)
        self.assertNotIn("test_metric_selection", json.dumps(template))


if __name__ == "__main__":
    unittest.main()
