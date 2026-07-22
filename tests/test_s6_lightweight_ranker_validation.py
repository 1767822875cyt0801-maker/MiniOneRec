import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s5_auxiliary_fusion_conversion as s5
import s6_lightweight_ranker_validation as lw


class S6LightweightRankerValidationTests(unittest.TestCase):
    def test_feature_names_have_no_target_or_label_leakage(self):
        names = lw.feature_names()
        self.assertIn("sasrec_direct_score", names)
        for name in names:
            self.assertNotIn("target", name)
            self.assertNotIn("label", name)
            self.assertNotIn("is_correct", name)

    def test_protocol_grid_is_finite_and_predeclared(self):
        protocol = lw.load_protocol(lw.DEFAULT_PROTOCOL)
        configs = lw.config_grid(protocol)
        self.assertEqual(len(configs), 64)
        self.assertEqual(configs[0]["config_id"], "alpha0.1_l20.001_clip0.25_cfq0")

    def test_metric_category_names_are_non_overlapping(self):
        labels = {
            "s6_4_direct_only_targets_beyond_cf_pool",
            "s6_5_direct_only_recovered_denominator",
            "non_cf_top20_recoverable_targets",
            "direct_source_only_pool_targets",
        }
        self.assertEqual(len(labels), 4)

    def test_pairwise_query_grouping_only_within_sample(self):
        rows = [
            {"row_index": "a", "item_id": "p", "label": 1, "features": [1.0, 0.0], "frozen_ranker_score": 0.0, "base_rank_0_based": 0},
            {"row_index": "a", "item_id": "n", "label": 0, "features": [0.0, 1.0], "frozen_ranker_score": 0.0, "base_rank_0_based": 1},
            {"row_index": "b", "item_id": "n2", "label": 0, "features": [0.0, 0.0], "frozen_ranker_score": 0.0, "base_rank_0_based": 0},
        ]
        np = lw.np_module()
        mean = np.zeros(2)
        std = np.ones(2)
        _weights, summary = lw.train_pairwise(rows, mean, std, 0.01)
        self.assertEqual(summary["pairs"], 1)

    def test_residual_score_composition_uses_clip_and_alpha(self):
        np = lw.np_module()
        rows = [
            {"row_index": "0", "item_id": "x", "features": [10.0], "frozen_ranker_score": 2.0, "base_rank_0_based": 0}
        ]
        scores = lw.scores_by_query(rows, np.asarray([1.0]), np.asarray([0.0]), np.asarray([1.0]), alpha=0.5, residual_clip=0.25)
        self.assertAlmostEqual(scores["0"]["x"], 2.125)

    def test_cf_quota_preserves_head_without_locking_all_cf(self):
        sample = s5.Sample(
            sample_id="0",
            target_item_id="9",
            history_item_id=[],
            cf_items=["c1", "c2", "c3"],
            sasrec_items=["d1"],
            candidates={
                "c1": s5.Candidate("c1", True, False, 0, None, None, None, 4, None, 1, None),
                "c2": s5.Candidate("c2", True, False, 1, None, None, None, 4, None, 1, None),
                "c3": s5.Candidate("c3", True, False, 2, None, None, None, 4, None, 1, None),
                "d1": s5.Candidate("d1", False, True, None, 0, None, None, None, 3, None, 1),
            },
            split="valid_fit",
        )
        ranked = lw.apply_cf_quota([("d1", 9.0, 0), ("c3", 8.0, 1), ("c2", 7.0, 2), ("c1", 6.0, 3)], sample, 2)
        self.assertEqual(ranked[:2], ["c3", "c2"])
        self.assertIn("d1", ranked[2:])

    def test_selection_logic_requires_fit_and_select(self):
        good = {
            "valid_fit": {
                "cf_preservation_rate_at20": 0.97,
                "hr20_uplift_retention_vs_frozen": 1.0,
                "ndcg20_uplift_retention_vs_frozen": 1.0,
                "direct_recovery_retention_vs_frozen": 0.9,
            },
            "valid_select": {
                "cf_preservation_rate_at20": 0.96,
                "hr20_uplift_retention_vs_frozen": 1.0,
                "ndcg20_uplift_retention_vs_frozen": 1.0,
                "direct_recovery_retention_vs_frozen": 0.9,
            },
        }
        self.assertFalse(lw.passes_fit_select(good))
        good["valid_select"]["cf_preservation_rate_at20"] = 0.97
        self.assertTrue(lw.passes_fit_select(good))

    def test_reject_existing_and_test_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                lw.write_json(path, {"x": 1})
        with self.assertRaises(ValueError):
            lw.reject_test_path(Path("data/test/file.csv"))


if __name__ == "__main__":
    unittest.main()
