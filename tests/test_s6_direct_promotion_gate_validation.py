import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s5_auxiliary_fusion_conversion as s5
import s6_direct_promotion_gate_validation as promo


def make_sample() -> s5.Sample:
    candidates = {}
    cf_items = [f"c{i}" for i in range(1, 22)]
    direct_items = ["d_target", "d_noise", "c5"]
    for idx, item in enumerate(cf_items):
        candidates[item] = s5.Candidate(item, True, item == "c5", idx, 2 if item == "c5" else None, None, None, 4, None, 1, None)
    candidates["d_target"] = s5.Candidate("d_target", False, True, None, 0, None, 9.0, None, 3, None, 1)
    candidates["d_noise"] = s5.Candidate("d_noise", False, True, None, 1, None, 8.0, None, 3, None, 1)
    return s5.Sample(
        sample_id="0",
        target_item_id="d_target",
        history_item_id=["h1", "h2"],
        cf_items=cf_items,
        sasrec_items=direct_items,
        candidates=candidates,
        split="valid_fit",
    )


class S6DirectPromotionGateValidationTests(unittest.TestCase):
    def test_feature_names_have_no_leakage_terms(self):
        names = promo.feature_names()
        self.assertIn("direct_score", names)
        for name in names:
            self.assertNotIn("target", name)
            self.assertNotIn("label", name)
            self.assertNotIn("post_ranking", name)

    def test_metric_reconciliation_can_use_frozen_aggregate_without_gate_rows(self):
        fake_report = {
            "aggregate": {
                "cf_only": {
                    "target_in_pool_count": 629,
                    "union_target_in_pool_count": 865,
                    "direct_only_targets_available_at20": 268,
                }
            }
        }
        reconciliation = promo.metric_definition_reconciliation({}, fake_report)
        self.assertTrue(reconciliation["reconciles"])
        self.assertEqual(reconciliation["frozen_full_validation_aggregate"]["cf_candidate_pool_exclusive_target"], 236)
        self.assertEqual(reconciliation["frozen_full_validation_aggregate"]["non_cf_top20_target_available_in_direct_top20"], 268)

    def test_protocol_grid_is_bounded_after_fit_thresholds(self):
        protocol = promo.load_protocol(promo.DEFAULT_PROTOCOL)
        grid = promo.policy_grid(protocol, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(len(grid), 12)
        self.assertEqual(grid[0]["config_id"], "mp1_thr0.100000")

    def test_direct_exclusive_candidate_filtering(self):
        sample = make_sample()
        self.assertEqual(promo.direct_exclusive_items(sample), ["d_target", "d_noise"])

    def test_boundary_candidate_construction_uses_cf_tail(self):
        sample = make_sample()
        protocol = promo.load_protocol(promo.DEFAULT_PROTOCOL)
        boundaries = promo.boundary_items(sample, protocol)
        self.assertEqual(boundaries[0], (10, "c11"))
        self.assertEqual(boundaries[-1], (19, "c20"))

    def test_pair_features_include_candidate_boundary_differences(self):
        direct = {
            "direct_rank": 1.0,
            "direct_reciprocal_rank": 1.0,
            "direct_score": 9.0,
            "direct_zscore": 2.0,
            "direct_top1_margin": 0.0,
            "direct_score_minus_topk_mean": 4.0,
            "frozen_projected_score": 0.8,
            "history_cosine": 0.1,
            "recent_cosine": 0.2,
            "popularity_score": 0.3,
            "bucket_penalty": -1.0,
            "cf_rank": 99.0,
            "reciprocal_cf_rank": 0.0,
        }
        boundary = {
            "direct_rank": 99.0,
            "direct_reciprocal_rank": 0.0,
            "direct_score": 0.0,
            "direct_zscore": 0.0,
            "direct_top1_margin": 0.0,
            "direct_score_minus_topk_mean": 0.0,
            "frozen_projected_score": 0.2,
            "history_cosine": 0.4,
            "recent_cosine": 0.5,
            "popularity_score": 0.6,
            "bucket_penalty": -2.0,
            "cf_rank": 20.0,
            "reciprocal_cf_rank": 0.05,
        }
        features = promo.pair_features(direct, boundary, 19)
        self.assertEqual(len(features), len(promo.feature_names()))
        self.assertAlmostEqual(features[-3], 0.6)
        self.assertAlmostEqual(features[-2], 0.95)

    def test_apply_promotion_preserves_retained_cf_internal_order(self):
        sample = make_sample()
        protocol = promo.load_protocol(promo.DEFAULT_PROTOCOL)
        side = {}
        for item in list(sample.candidates):
            direct = sample.candidates[item].sasrec_rank
            cf = sample.candidates[item].cf_rank
            side[item] = {
                "direct_rank": 99.0 if direct is None else float(direct + 1),
                "direct_reciprocal_rank": 0.0 if direct is None else 1.0 / float(direct + 1),
                "direct_score": 10.0 if item == "d_target" else 0.0,
                "direct_zscore": 10.0 if item == "d_target" else 0.0,
                "direct_top1_margin": 0.0,
                "direct_score_minus_topk_mean": 10.0 if item == "d_target" else 0.0,
                "frozen_projected_score": 1.0 if item == "d_target" else 0.0,
                "history_cosine": 0.0,
                "recent_cosine": 0.0,
                "popularity_score": 0.0,
                "bucket_penalty": 0.0,
                "cf_rank": 99.0 if cf is None else float(cf + 1),
                "reciprocal_cf_rank": 0.0 if cf is None else 1.0 / float(cf + 1),
            }
        np = promo.np_module()
        weights = np.zeros(len(promo.feature_names()))
        weights[2] = 1.0
        mean = np.zeros(len(promo.feature_names()))
        std = np.ones(len(promo.feature_names()))
        orders, audit, _rows = promo.apply_promotion_policy(
            [sample],
            {"0": {"side_features": side}},
            protocol,
            weights,
            0.0,
            mean,
            std,
            {"max_promotions": 1, "promotion_threshold": 0.5, "margin_threshold": 0.0},
        )
        self.assertEqual(orders["0"][:19], sample.cf_items[:19])
        self.assertEqual(orders["0"][19], "d_target")
        self.assertEqual(audit["true_promotions"], 1)

    def test_max_promotion_bound_is_enforced(self):
        sample = make_sample()
        protocol = promo.load_protocol(promo.DEFAULT_PROTOCOL)
        side = {}
        for item in list(sample.candidates):
            direct = sample.candidates[item].sasrec_rank
            cf = sample.candidates[item].cf_rank
            side[item] = {
                "direct_rank": 99.0 if direct is None else float(direct + 1),
                "direct_reciprocal_rank": 0.0 if direct is None else 1.0 / float(direct + 1),
                "direct_score": 10.0,
                "direct_zscore": 10.0,
                "direct_top1_margin": 0.0,
                "direct_score_minus_topk_mean": 10.0,
                "frozen_projected_score": 1.0,
                "history_cosine": 0.0,
                "recent_cosine": 0.0,
                "popularity_score": 0.0,
                "bucket_penalty": 0.0,
                "cf_rank": 99.0 if cf is None else float(cf + 1),
                "reciprocal_cf_rank": 0.0 if cf is None else 1.0 / float(cf + 1),
            }
        np = promo.np_module()
        weights = np.zeros(len(promo.feature_names()))
        weights[2] = 1.0
        orders, audit, _rows = promo.apply_promotion_policy(
            [sample],
            {"0": {"side_features": side}},
            protocol,
            weights,
            0.0,
            np.zeros(len(promo.feature_names())),
            np.ones(len(promo.feature_names())),
            {"max_promotions": 1, "promotion_threshold": 0.5, "margin_threshold": 0.0},
        )
        self.assertEqual(audit["queries_with_one_promotion"], 1)
        self.assertEqual(len([item for item in orders["0"][:20] if item.startswith("d_")]), 1)

    def test_fit_only_threshold_calibration_uses_positive_rows(self):
        np = promo.np_module()
        rows = [
            {"features": [1.0], "label": 1},
            {"features": [2.0], "label": 1},
            {"features": [-1.0], "label": 0},
        ]
        thresholds = promo.calibrate_thresholds(rows, np.asarray([1.0]), 0.0, np.asarray([0.0]), np.asarray([1.0]), promo.load_protocol(promo.DEFAULT_PROTOCOL))
        self.assertTrue(1 <= len(thresholds) <= 4)
        self.assertEqual(thresholds, sorted(thresholds))

    def test_gate_is_not_opened_when_no_policy_passes(self):
        bad = {
            "valid_fit": {
                "cf_preservation_rate_at20": 0.97,
                "invalid_candidate_count": 0,
                "hr20_uplift_retention_vs_frozen": 1.0,
                "ndcg20_uplift_retention_vs_frozen": 1.0,
                "direct_recovery_retention_vs_frozen": 0.9,
            },
            "valid_select": {
                "cf_preservation_rate_at20": 0.96,
                "invalid_candidate_count": 0,
                "hr20_uplift_retention_vs_frozen": 1.0,
                "ndcg20_uplift_retention_vs_frozen": 1.0,
                "direct_recovery_retention_vs_frozen": 0.9,
            },
        }
        self.assertFalse(promo.passes_fit_select(bad))
        bad["valid_select"]["cf_preservation_rate_at20"] = 0.97
        self.assertTrue(promo.passes_fit_select(bad))

    def test_reject_existing_and_test_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                promo.write_json(path, {"x": 1})
        with self.assertRaises(ValueError):
            promo.reject_test_path(Path("results/final_test/candidates.jsonl"))


if __name__ == "__main__":
    unittest.main()
