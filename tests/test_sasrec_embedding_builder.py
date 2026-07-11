import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import build_sasrec_embeddings as sasrec_builder


class TestSASRecEmbeddingBuilder(unittest.TestCase):
    def test_shifted_target_padding_and_truncation(self):
        short = sasrec_builder.shifted_example([5, 6], max_seq_len=4)
        self.assertEqual(short.input_ids, [0, 0, 0, 5])
        self.assertEqual(short.target_ids, [0, 0, 0, 6])

        truncated = sasrec_builder.shifted_example([1, 2, 3, 4, 5], max_seq_len=3)
        self.assertEqual(truncated.input_ids, [2, 3, 4])
        self.assertEqual(truncated.target_ids, [3, 4, 5])

    def test_causal_mask(self):
        mask = sasrec_builder.causal_attention_mask(4)
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[0, 1])
        self.assertFalse(mask[3, 0])
        self.assertEqual(mask.shape, (4, 4))

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
        import argparse
        import torch

        args = argparse.Namespace(
            embedding_dim=8,
            max_seq_len=4,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
        )
        model = sasrec_builder.build_torch_model(num_items=5, args=args)
        logits = model(torch.tensor([[0, 1, 2, 3]], dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (1, 4, 5))


if __name__ == "__main__":
    unittest.main()
