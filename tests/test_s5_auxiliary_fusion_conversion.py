import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s5_auxiliary_fusion_conversion as s5  # noqa: E402


def row(row_index, target, items, history=None):
    return {
        "row_index": str(row_index),
        "target_item_id": str(target),
        "history_item_id": history or ["10", "11"],
        "candidate_item_ids": [str(item) for item in items],
        "candidate_details": [
            {
                "item_id": str(item),
                "source_type": "exact",
                "expansion_level": 3,
                "bucket_size": 1,
            }
            for item in items
        ],
    }


class S5AuxiliaryFusionConversionTests(unittest.TestCase):
    def setUp(self):
        self.cf_rows = [
            row(0, "2", ["1", "2", "3"]),
            row(1, "5", ["4", "6", "7"]),
        ]
        self.sasrec_rows = [
            row(0, "2", ["3", "8", "2"]),
            row(1, "5", ["5", "6", "9"]),
        ]
        self.split_map = {"0": "valid_select", "1": "valid_fit"}
        self.samples = s5.build_samples(self.cf_rows, self.sasrec_rows, self.split_map)

    def test_candidate_row_alignment_and_target_mismatch_rejected(self):
        self.assertEqual([sample.sample_id for sample in self.samples], ["0", "1"])
        bad_sasrec = [dict(self.sasrec_rows[0], target_item_id="999"), self.sasrec_rows[1]]
        with self.assertRaises(ValueError):
            s5.build_samples(self.cf_rows, bad_sasrec, self.split_map)

    def test_item_dedup_and_provenance_preservation(self):
        sample = self.samples[0]
        self.assertEqual(set(sample.candidates), {"1", "2", "3", "8"})
        duplicate = sample.candidates["3"]
        self.assertTrue(duplicate.source_cf)
        self.assertTrue(duplicate.source_sasrec)
        self.assertTrue(duplicate.duplicate_across_sources)
        target = sample.candidates["2"]
        self.assertEqual(target.cf_rank, 1)
        self.assertEqual(target.sasrec_rank, 2)
        rows = s5.provenance_rows(self.samples)
        target_rows = [r for r in rows if r["sample_id"] == "0" and r["item_id"] == "2"]
        self.assertEqual(len(target_rows), 1)
        self.assertTrue(target_rows[0]["is_ground_truth"])
        self.assertEqual(target_rows[0]["source_count"], 2)

    def test_deterministic_output(self):
        first = s5.provenance_rows(self.samples)
        second = s5.provenance_rows(s5.build_samples(self.cf_rows, self.sasrec_rows, self.split_map))
        self.assertEqual(first, second)

    def test_valid_select_report_disjointness(self):
        select = s5.split_filter(self.samples, s5.SPLIT_SELECT)
        report = s5.split_filter(self.samples, s5.SPLIT_REPORT)
        self.assertEqual({s.sample_id for s in select}, {"0"})
        self.assertEqual({s.sample_id for s in report}, {"1"})
        self.assertTrue({s.sample_id for s in select}.isdisjoint({s.sample_id for s in report}))

    def test_no_test_path_rejection(self):
        self.assertFalse(s5.no_test_path(Path("data/Amazon/test/file.csv")))
        with self.assertRaises(ValueError):
            s5.assert_no_test_paths([Path("results/test/output.json")])

    def test_selected_config_freeze_shape(self):
        configs = s5.fixed_policy_configs()
        select_rows = [s5.evaluate_policy(self.samples, cfg, s5.SPLIT_SELECT, [1, 5, 10, 20, 50]) for cfg in configs]
        selected = max(select_rows, key=s5.selection_key)
        frozen = {"config_id": selected["config_id"], "policy": selected["policy"], "frozen": True}
        self.assertTrue(frozen["frozen"])
        self.assertIn("config_id", frozen)
        self.assertIn("policy", frozen)

    def test_ranker_schema_compatibility_matrix_flags_source_alias(self):
        model = {
            "feature_names": [
                "text_present",
                "cf_present",
                "text_rank_filled",
                "cf_rank_filled",
            ]
        }
        matrix = s5.compatibility_matrix(model, Path("stage7"))
        direct = matrix[0]
        self.assertFalse(direct["compatible_directly_with_cf_sasrec"])
        self.assertTrue(direct["source_fields_hardcoded"])
        self.assertIn("explicit alias", direct["allowed_use_in_s5"])

    def test_metric_recomputation_consistency(self):
        cfg = {"config_id": "cf_only_top50", "policy": "cf_only_top50"}
        metrics = s5.evaluate_policy(self.samples, cfg, s5.SPLIT_SELECT, [1, 5, 10, 20, 50])
        self.assertEqual(metrics["num_samples"], 1)
        self.assertEqual(metrics["target_in_pool_count"], 1)
        self.assertEqual(metrics["hr@1"], 0.0)
        self.assertEqual(metrics["hr@5"], 1.0)

    def test_output_paths_are_writable_without_touching_formal_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "out.json"
            s5.write_json(path, {"ok": True})
            self.assertEqual(json.loads(path.read_text()), {"ok": True})


if __name__ == "__main__":
    unittest.main()
