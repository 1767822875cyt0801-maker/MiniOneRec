import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s5_auxiliary_fusion_conversion as s5
import s5_final_test_confirmation as s5final
import s6_frozen_ranker_validation as fr


def cf_row(row_index, target, history, candidates):
    return {
        "row_index": str(row_index),
        "target_item_id": str(target),
        "history_item_id": [str(item) for item in history],
        "candidate_item_ids": [str(item) for item in candidates],
        "candidate_details": [
            {"item_id": str(item), "source_type": "exact", "sid_rank_0_based": rank, "expansion_level": 4, "bucket_size": 1}
            for rank, item in enumerate(candidates)
        ],
    }


def union_row(row_index, target, history, details):
    return {
        "row_index": str(row_index),
        "target_item_id": str(target),
        "history_item_id": [str(item) for item in history],
        "candidate_item_ids": [str(d["item_id"]) for d in details],
        "candidate_details": details,
    }


class S6FrozenRankerValidationTests(unittest.TestCase):
    def test_frozen_model_hash_and_feature_order(self):
        config = fr.load_frozen_config(fr.DEFAULT_CONFIG)
        model, contract = fr.load_model_and_contract(config)
        self.assertEqual(config["ranker"]["model_sha256"], fr.EXPECTED_MODEL_SHA256)
        self.assertEqual(fr.file_sha256(ROOT / config["ranker"]["model_path"]), fr.EXPECTED_MODEL_SHA256)
        self.assertEqual(model["feature_names"], contract["ordered_feature_names"])
        self.assertEqual(model["feature_names"], config["projection_contract"]["ordered_feature_names"])

    def test_projection_zeroes_source_specific_features(self):
        model = s5final.load_model()
        raw = {name: float(idx + 1) for idx, name in enumerate(model["feature_names"])}
        projected = s5final.project_normalized_features(raw, model)
        for idx, name in enumerate(model["feature_names"]):
            if name in s5final.SOURCE_SPECIFIC_FEATURES_ZEROED:
                self.assertEqual(projected[idx], 0.0)

    def test_direct_rows_from_union_preserves_direct_order_and_score_sidecar(self):
        rows = [
            union_row(
                0,
                "9",
                ["1"],
                [
                    {"item_id": "2", "from_sasrec_direct": True, "sasrec_direct_rank_0_based": 1, "sasrec_direct_detail": {"item_id": "2", "sasrec_direct_score": 1.0}},
                    {"item_id": "3", "from_sasrec_direct": False},
                    {"item_id": "4", "from_sasrec_direct": True, "sasrec_direct_rank_0_based": 0, "sasrec_direct_detail": {"item_id": "4", "sasrec_direct_score": 2.0}},
                ],
            )
        ]
        direct = fr.direct_rows_from_union(rows)
        self.assertEqual(direct[0]["candidate_item_ids"], ["4", "2"])
        self.assertEqual(direct[0]["candidate_details"][0]["sasrec_direct_score"], 2.0)

    def test_forbidden_direct_score_features_do_not_enter_raw_feature_surface(self):
        sample = s5.Sample(
            sample_id="0",
            target_item_id="2",
            history_item_id=["1"],
            cf_items=["1"],
            sasrec_items=["2"],
            candidates={
                "1": s5.Candidate("1", True, False, 0, None, None, None, 4, None, 1, None),
                "2": s5.Candidate("2", False, True, None, 0, None, 9.9, None, None, None, None),
            },
            split="valid_fit",
        )
        class Matrix:
            def __getitem__(self, _idx):
                return self
            def dot(self, _other):
                return 0.0
        raw = s5.raw_features(sample, sample.candidates["2"], 2, 2, Matrix(), {"1": 0, "2": 1}, {}, 1.0, "source_independent_projection")
        self.assertTrue(fr.FORBIDDEN_DIRECT_FEATURES.isdisjoint(raw))

    def test_metrics_decomposition_counts_direct_only_recovery(self):
        sample = s5.Sample(
            sample_id="0",
            target_item_id="2",
            history_item_id=["1"],
            cf_items=["1"],
            sasrec_items=["2"],
            candidates={
                "1": s5.Candidate("1", True, False, 0, None, None, None, 4, None, 1, None),
                "2": s5.Candidate("2", False, True, None, 0, None, None, None, None, None, None),
            },
            split="valid_fit",
        )
        metrics = fr.metrics_for_orders([sample], {"0": ["2", "1"]})
        self.assertEqual(metrics["direct_only_targets_available_at20"], 1)
        self.assertEqual(metrics["direct_only_targets_recovered_at20"], 1)
        self.assertEqual(metrics["cf_hits_lost_at20"], 0)

    def test_reject_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                fr.write_json(path, {"new": True})

    def test_reject_test_paths(self):
        with self.assertRaises(ValueError):
            fr.reject_test_path(Path("data/Amazon/test/file.csv"))


if __name__ == "__main__":
    unittest.main()
