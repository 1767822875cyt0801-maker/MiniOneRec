import json
import sys
import tempfile
import unittest
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_sasrec_embeddings as sasrec_builder


class TestSASRecEmbeddingBuilder(unittest.TestCase):
    def _args(self):
        return argparse.Namespace(topk=[1, 10, 20], max_seq_len=4, seed=7)

    def test_shifted_target_padding_and_truncation(self):
        short = sasrec_builder.shifted_example([5, 6], max_seq_len=4)
        self.assertEqual(short.input_ids, [5, 0, 0, 0])
        self.assertEqual(short.target_ids, [6, 0, 0, 0])

        truncated = sasrec_builder.shifted_example([1, 2, 3, 4, 5], max_seq_len=3)
        self.assertEqual(truncated.input_ids, [2, 3, 4])
        self.assertEqual(truncated.target_ids, [3, 4, 5])

    def test_causal_mask(self):
        mask = sasrec_builder.causal_attention_mask(4)
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[0, 1])
        self.assertFalse(mask[3, 0])
        self.assertEqual(mask.shape, (4, 4))

    def test_valid_input_is_independent_of_target(self):
        row_index = {"10": 0, "11": 1, "12": 2, "13": 3}
        base = {"history_item_id": "[10, 11]", "item_id": "12"}
        changed = {"history_item_id": "[10, 11]", "item_id": "13"}

        base_input = sasrec_builder.valid_input(
            sasrec_builder.row_sequence(base, row_index, include_target=False),
            max_seq_len=4,
        )
        changed_input = sasrec_builder.valid_input(
            sasrec_builder.row_sequence(changed, row_index, include_target=False),
            max_seq_len=4,
        )

        self.assertEqual(base_input, changed_input)
        self.assertEqual(base_input, [1, 2, 0, 0])

    def test_valid_model_input_does_not_append_target(self):
        row_index = {"10": 0, "11": 1, "12": 2}
        row = {"history_item_id": "[10, 11]", "item_id": "12"}

        history_only = sasrec_builder.row_sequence(row, row_index, include_target=False)
        with_target = sasrec_builder.row_sequence(row, row_index, include_target=True)
        model_input = sasrec_builder.valid_input(history_only, max_seq_len=4)

        self.assertEqual(history_only, [1, 2])
        self.assertEqual(with_target, [1, 2, 3])
        self.assertEqual(model_input, [1, 2, 0, 0])
        self.assertNotIn(3, model_input)

    def test_safe_rank_does_not_count_all_ties_or_nan_as_hits(self):
        rank, candidate_count, top10 = sasrec_builder.rank_from_scores([0.0, 0.0, 0.0, 0.0], target_index=1)
        self.assertEqual(candidate_count, 4)
        self.assertEqual(rank, 3)
        self.assertEqual(top10, [0, 1, 2, 3])

        rank, candidate_count, top10 = sasrec_builder.rank_from_scores([1.0, float("nan"), 0.5], target_index=1)
        self.assertIsNone(rank)
        self.assertEqual(candidate_count, 3)
        self.assertEqual(top10, [])

    def test_full_candidate_metrics_and_baselines(self):
        item_order = ["10", "11", "12", "13"]
        row_index = {item: idx for idx, item in enumerate(item_order)}
        valid_rows = [
            {"history_item_id": "[10, 11]", "item_id": "12"},
            {"history_item_id": "[10]", "item_id": "13"},
        ]
        args = self._args()

        metrics = sasrec_builder.evaluate_score_rows(
            [[0.1, 0.2, 0.9, 0.0], [0.3, 0.2, 0.1, 0.8]],
            valid_rows,
            row_index,
            item_order,
            args,
            debug_limit=2,
        )
        self.assertEqual(metrics["candidate_count_expected"], 4)
        self.assertEqual(metrics["candidate_count_min"], 4)
        self.assertEqual(metrics["candidate_count_max"], 4)
        self.assertTrue(metrics["candidate_count_all_full"])
        self.assertEqual(metrics["hr@1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(len(metrics["debug_valid_samples"]), 2)
        self.assertFalse(metrics["debug_valid_samples"][0]["target_in_input"])
        self.assertEqual(metrics["finite_score_samples"], 2)
        self.assertEqual(metrics["nonfinite_score_samples"], 0)
        self.assertEqual(metrics["unranked_samples"], 0)

        counts = {"10": 5, "11": 4, "12": 3, "13": 2}
        popularity = sasrec_builder.popularity_baseline_metrics(valid_rows, row_index, item_order, counts, args)
        random = sasrec_builder.random_baseline_metrics(valid_rows, row_index, item_order, args)
        for baseline in [popularity, random]:
            self.assertIn("hr@1", baseline)
            self.assertIn("hr@10", baseline)
            self.assertIn("hr@20", baseline)
            self.assertIn("ndcg@10", baseline)
            self.assertIn("ndcg@20", baseline)
            self.assertIn("mrr", baseline)
            self.assertTrue(baseline["candidate_count_all_full"])

    def test_canonical_mapping_alignment_and_export_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            item_order = ["10", "11", "12"]
            row_index = {"10": 0, "11": 1, "12": 2}
            item_order_path = tmp_path / "item_order.json"
            row_index_path = tmp_path / "row_index.json"
            item_order_path.write_text(json.dumps(item_order), encoding="utf-8")
            row_index_path.write_text(json.dumps(row_index), encoding="utf-8")
            loaded_order, loaded_index = sasrec_builder.load_canonical_mapping(item_order_path, row_index_path)
            self.assertEqual(loaded_order, item_order)
            self.assertEqual(loaded_index, row_index)

            raw = np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 2.0],
                ],
                dtype=np.float32,
            )
            counts = {"10": 3, "11": 0, "12": 1}
            matrix, cold = sasrec_builder.prepare_export_matrix(raw, item_order, counts, normalization="l2")
            self.assertTrue(np.isfinite(matrix).all())
            self.assertEqual(matrix.shape, (3, 2))
            self.assertEqual(cold["cold_item_ids"], ["11"])
            self.assertGreater(float(np.linalg.norm(matrix[1])), 0.0)

            paths = sasrec_builder.write_artifacts(
                output_dir=tmp_path / "export",
                category="Tiny",
                matrix=matrix,
                item_order=item_order,
                row_index=row_index,
                train_counts=counts,
                cold_report=cold,
                config={"normalization": "l2"},
                metrics={"valid": {"hr@20": 0.0}},
                checkpoint_path=None,
            )
            for key in [
                "embedding",
                "row_index",
                "item_order",
                "train_item_count",
                "cold_item_report",
                "config",
                "metrics_summary",
                "artifact_manifest",
            ]:
                self.assertTrue(Path(paths[key]).exists(), key)
            manifest = json.loads(Path(paths["artifact_manifest"]).read_text(encoding="utf-8"))
            self.assertTrue(manifest["finite"])
            self.assertFalse(manifest["train_only_policy"]["test_read"])

    @unittest.skipUnless(sasrec_builder.torch_available()[0], "PyTorch is not installed in this local environment")
    def test_tiny_torch_forward(self):
        import torch

        args = argparse.Namespace(
            embedding_dim=8,
            max_seq_len=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        model = sasrec_builder.build_torch_model(num_items=5, args=args)
        logits = model(torch.tensor([[1, 2, 3, 0]], dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (1, 4, 5))

    @unittest.skipUnless(sasrec_builder.torch_available()[0], "PyTorch is not installed in this local environment")
    def test_two_layer_right_padding_is_finite_for_short_histories(self):
        import torch
        import torch.nn.functional as F

        args = argparse.Namespace(
            embedding_dim=16,
            max_seq_len=10,
            num_heads=2,
            num_layers=2,
            dropout=0.0,
            topk=[1, 10, 20, 50],
            seed=123,
        )
        item_count = 3686
        histories = [
            [1],
            [2, 3],
            [4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
        ]
        targets = [100, 101, 102, 103, 104]
        input_ids = [sasrec_builder.valid_input(history, args.max_seq_len) for history in histories]
        for row, target in zip(input_ids, targets):
            self.assertNotIn(target, row)
        inputs = torch.tensor(input_ids, dtype=torch.long)
        model = sasrec_builder.build_torch_model(num_items=item_count, args=args)
        model.train()

        full_logits = model(inputs, check_finite=True, finite_context=sasrec_builder.finite_context(inputs, epoch=1, batch_index=1))
        self.assertTrue(torch.isfinite(full_logits).all())
        last_logits = model.last_valid_logits(
            inputs,
            check_finite=True,
            finite_context=sasrec_builder.finite_context(inputs, epoch=1, batch_index=1),
        )
        self.assertEqual(tuple(last_logits.shape), (len(histories), item_count))
        self.assertTrue(torch.isfinite(last_logits).all())
        loss = F.cross_entropy(last_logits, torch.tensor([target - 1 for target in targets], dtype=torch.long))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        sasrec_builder.assert_model_gradients_finite(model, "test_backward", sasrec_builder.finite_context(inputs, epoch=1, batch_index=1))

        item_order = [str(idx + 1) for idx in range(item_count)]
        row_index = {item: idx for idx, item in enumerate(item_order)}
        valid_rows = [
            {"history_item_id": str(history), "item_id": str(target)}
            for history, target in zip(histories, targets)
        ]
        metrics = sasrec_builder.evaluate_score_rows(
            last_logits.detach().numpy(),
            valid_rows,
            row_index,
            item_order,
            args,
        )
        self.assertEqual(metrics["candidate_count_expected"], 3686)
        self.assertEqual(metrics["candidate_count_min"], 3686)
        self.assertEqual(metrics["candidate_count_max"], 3686)
        self.assertEqual(metrics["nonfinite_score_samples"], 0)
        self.assertEqual(metrics["unranked_samples"], 0)


if __name__ == "__main__":
    unittest.main()
