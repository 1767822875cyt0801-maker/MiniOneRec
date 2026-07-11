import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_stage7_p3_valid_fusion_sweep as p3_sweep
import train_stage7_p2_history_ranker as history_ranker


class TestStage7P3FusionSweep(unittest.TestCase):
    def test_select_best_uses_ndcg_then_hr_then_fewer_candidates(self):
        rows = [
            {"config_id": "a", "history_ranker_ndcg20": 0.2, "history_ranker_hr20": 0.4, "mean_candidates": 40, "lambda_text": 0.5, "source_weight": 4.0},
            {"config_id": "b", "history_ranker_ndcg20": 0.3, "history_ranker_hr20": 0.1, "mean_candidates": 10, "lambda_text": 0.6, "source_weight": 4.0},
            {"config_id": "c", "history_ranker_ndcg20": 0.3, "history_ranker_hr20": 0.2, "mean_candidates": 50, "lambda_text": 0.4, "source_weight": 2.0},
            {"config_id": "d", "history_ranker_ndcg20": 0.3, "history_ranker_hr20": 0.2, "mean_candidates": 30, "lambda_text": 0.5, "source_weight": 4.0},
        ]
        selected = p3_sweep.select_best(rows)
        self.assertEqual(selected["selected_config_id"], "d")

    def test_apply_frozen_ranker_does_not_change_model_and_reranks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_csv = tmp_path / "train.csv"
            with open(train_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["user_id", "history_item_id", "item_id"])
                writer.writeheader()
                writer.writerow({"user_id": "u1", "history_item_id": "['1']", "item_id": "2"})
            emb = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            emb_path = tmp_path / "emb.npy"
            np.save(emb_path, emb)
            row_index = tmp_path / "row_index.json"
            row_index.write_text(json.dumps({"1": 0, "2": 1, "3": 2}), encoding="utf-8")

            candidates = tmp_path / "valid_candidates.jsonl"
            details = [
                {
                    "item_id": "3",
                    "text_rank": "",
                    "behavior_rank": 1,
                    "from_text": False,
                    "from_behavior": True,
                    "sid_rank_0_based": 0,
                    "source_type": "exact",
                    "expansion_level": 3,
                    "bucket_size": 1,
                    "fusion_score": 1.0,
                },
                {
                    "item_id": "2",
                    "text_rank": 2,
                    "behavior_rank": "",
                    "from_text": True,
                    "from_behavior": False,
                    "sid_rank_0_based": 1,
                    "source_type": "exact",
                    "expansion_level": 3,
                    "bucket_size": 1,
                    "fusion_score": 0.5,
                },
            ]
            record = {
                "row_index": 0,
                "target_item_id": "2",
                "history_item_id": ["1"],
                "candidate_item_ids": ["3", "2"],
                "candidate_details": details,
            }
            candidates.write_text(json.dumps(record) + "\n", encoding="utf-8")

            weights = [0.0 for _ in history_ranker.FEATURE_NAMES]
            weights[history_ranker.FEATURE_NAMES.index("history_cosine")] = 10.0
            model = {
                "model_type": "history_aware_linear_pairwise_logistic",
                "feature_names": history_ranker.FEATURE_NAMES,
                "weights": weights,
                "feature_stats": {
                    "feature_names": history_ranker.FEATURE_NAMES,
                    "mean": [0.0 for _ in history_ranker.FEATURE_NAMES],
                    "std": [1.0 for _ in history_ranker.FEATURE_NAMES],
                    "count": 1,
                },
                "selected_config_id": "tiny",
                "heuristic_component_weights": {
                    "sid_rank_weight": 1.0,
                    "source_weight": 4.0,
                    "popularity_weight": 0.2,
                    "history_cosine_weight": 0.8,
                    "recent_cosine_weight": 0.4,
                    "bucket_penalty_weight": 0.05,
                },
            }
            model_path = tmp_path / "model.json"
            model_path.write_text(json.dumps(model), encoding="utf-8")
            before = hashlib.sha256(model_path.read_bytes()).hexdigest()
            out_dir = tmp_path / "out"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "apply_stage7_p2_history_ranker.py"),
                "--candidate-jsonl",
                str(candidates),
                "--frozen-ranker",
                str(model_path),
                "--train-csv",
                str(train_csv),
                "--item-emb",
                str(emb_path),
                "--row-index",
                str(row_index),
                "--output-dir",
                str(out_dir),
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)
            after = hashlib.sha256(model_path.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            report = json.loads((out_dir / "ranker_apply_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["frozen_ranker_integrity"]["unchanged"])
            self.assertEqual(report["history_aware_rerank"]["hr@20"], 1.0)
            row = json.loads((out_dir / "history_reranked_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["history_aware_reranked_item_ids"][0], "2")

    def test_p3_runner_writes_summary_row_per_parameter_combo(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            eval_csv = tmp_path / "valid.csv"
            train_csv = tmp_path / "train.csv"
            for path in [eval_csv, train_csv]:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["user_id", "history_item_id", "item_id"])
                    writer.writeheader()
                    writer.writerow({"user_id": "u1", "history_item_id": "['1']", "item_id": "2"})

            text_candidates = tmp_path / "text_candidates.jsonl"
            behavior_candidates = tmp_path / "behavior_candidates.jsonl"
            text_candidates.write_text(
                json.dumps(
                    {
                        "row_index": 0,
                        "target_item_id": "2",
                        "history_item_id": ["1"],
                        "candidate_item_ids": ["2"],
                        "candidate_details": [
                            {"item_id": "2", "source_type": "exact", "sid_rank_0_based": 0, "bucket_size": 1}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            behavior_candidates.write_text(
                json.dumps(
                    {
                        "row_index": 0,
                        "target_item_id": "2",
                        "history_item_id": ["1"],
                        "candidate_item_ids": ["3"],
                        "candidate_details": [
                            {"item_id": "3", "source_type": "exact", "sid_rank_0_based": 0, "bucket_size": 1}
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            emb_path = tmp_path / "emb.npy"
            np.save(emb_path, np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            row_index = tmp_path / "row_index.json"
            row_index.write_text(json.dumps({"1": 0, "2": 1, "3": 2}), encoding="utf-8")

            weights = [0.0 for _ in history_ranker.FEATURE_NAMES]
            model = {
                "model_type": "history_aware_linear_pairwise_logistic",
                "feature_names": history_ranker.FEATURE_NAMES,
                "weights": weights,
                "feature_stats": {
                    "feature_names": history_ranker.FEATURE_NAMES,
                    "mean": [0.0 for _ in history_ranker.FEATURE_NAMES],
                    "std": [1.0 for _ in history_ranker.FEATURE_NAMES],
                    "count": 1,
                },
                "selected_config_id": "tiny",
                "heuristic_component_weights": {
                    "sid_rank_weight": 1.0,
                    "source_weight": 4.0,
                    "popularity_weight": 0.2,
                    "history_cosine_weight": 0.8,
                    "recent_cosine_weight": 0.4,
                    "bucket_penalty_weight": 0.05,
                },
            }
            model_path = tmp_path / "model.json"
            model_path.write_text(json.dumps(model), encoding="utf-8")
            out_dir = tmp_path / "p3_sweep"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "run_stage7_p3_valid_fusion_sweep.py"),
                "--category",
                "Tiny",
                "--text-candidates",
                str(text_candidates),
                "--behavior-candidates",
                str(behavior_candidates),
                "--eval-csv",
                str(eval_csv),
                "--train-csv",
                str(train_csv),
                "--item-emb",
                str(emb_path),
                "--row-index",
                str(row_index),
                "--frozen-ranker",
                str(model_path),
                "--out-dir",
                str(out_dir),
                "--lambdas",
                "0.4",
                "0.5",
                "--source-weights",
                "4.0",
            ]
            subprocess.run(cmd, cwd=ROOT, check=True)
            with open(out_dir / "fusion_sweep_valid.csv", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["config_dir"] for row in rows}), 2)
            self.assertTrue((out_dir / "lambda0p4_source4" / "config.json").exists())
            selected = json.loads((out_dir / "selected_config.json").read_text(encoding="utf-8"))
            self.assertIn("valid NDCG@20", selected["selection_rule"])
            self.assertNotIn("test", json.dumps(selected).lower())


if __name__ == "__main__":
    unittest.main()
