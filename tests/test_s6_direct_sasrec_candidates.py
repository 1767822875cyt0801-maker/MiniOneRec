import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import export_sasrec_direct_candidates as direct  # noqa: E402
import merge_direct_sasrec_candidates as merge  # noqa: E402


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class S6DirectSASRecCandidateTests(unittest.TestCase):
    def test_stable_topk_ties_break_by_item_order(self):
        self.assertEqual(direct.stable_topk([0.2, 0.5, 0.5, 0.1], 3), [1, 2, 0])

    def test_score_features_are_finite_and_ranked(self):
        selected = direct.stable_topk([1.0, 3.0, 2.0], 3)
        features = direct.score_features([1.0, 3.0, 2.0], selected)
        self.assertEqual(features[1]["sasrec_direct_rank"], 1.0)
        for item_features in features.values():
            for value in item_features.values():
                self.assertTrue(value == value)
                self.assertNotEqual(value, float("inf"))

    def test_only_frozen_k_values_are_allowed_by_parser(self):
        self.assertEqual(direct.ALLOWED_K, {20, 50, 100})

    def test_checkpoint_report_must_be_go(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "config.json"
            report = root / "report.json"
            cfg.write_text(json.dumps({"seed": 42}), encoding="utf-8")
            report.write_text(json.dumps({"verdict": "NO_GO_CHECKPOINT_COMPATIBILITY"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                direct.load_checked_config(cfg, report)

    def test_union_preserves_overlap_provenance(self):
        cf = {
            "row_index": "0",
            "target_item_id": "3",
            "history_item_id": ["1"],
            "candidate_item_ids": ["2", "3"],
            "candidate_details": [{"item_id": "2"}, {"item_id": "3"}],
        }
        sas = {
            "row_index": "0",
            "target_item_id": "3",
            "history_item_id": ["1"],
            "candidate_item_ids": ["3", "4"],
            "candidate_details": [
                {"item_id": "3", "sasrec_direct_score": 9.0},
                {"item_id": "4", "sasrec_direct_score": 8.0},
            ],
        }
        out = merge.merge_rows(cf, sas)
        self.assertEqual(out["candidate_item_ids"], ["2", "3", "4"])
        detail_by_item = {row["item_id"]: row for row in out["candidate_details"]}
        self.assertTrue(detail_by_item["3"]["from_cf"])
        self.assertTrue(detail_by_item["3"]["from_sasrec_direct"])
        self.assertTrue(detail_by_item["3"]["overlap"])
        self.assertEqual(out["candidate_hit_rank_0_based"], 1)

    def test_merge_rejects_row_misalignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cf = root / "cf.jsonl"
            sas = root / "sas.jsonl"
            write_jsonl(cf, [{"row_index": "0", "target_item_id": "1", "candidate_item_ids": ["1"]}])
            write_jsonl(sas, [{"row_index": "1", "target_item_id": "1", "candidate_item_ids": ["1"]}])
            with self.assertRaises(ValueError):
                merge.merge_files(cf, sas, root / "out.jsonl", root / "report.json")

    def test_merge_rejects_existing_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cf = root / "cf.jsonl"
            sas = root / "sas.jsonl"
            out = root / "out.jsonl"
            write_jsonl(cf, [{"row_index": "0", "target_item_id": "1", "candidate_item_ids": ["1"]}])
            write_jsonl(sas, [{"row_index": "0", "target_item_id": "1", "candidate_item_ids": ["1"]}])
            out.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                merge.merge_files(cf, sas, out, root / "report.json")

    def test_prefix_views_are_exact_prefixes(self):
        scored = [{
            "row": {"source_row_index": "0", "user_id": "u"},
            "target": "3",
            "target_index": 2,
            "history_item_id": ["1"],
            "top_indices": [2, 0, 1, 3],
            "features": {
                2: {"sasrec_direct_score": 4.0},
                0: {"sasrec_direct_score": 3.0},
                1: {"sasrec_direct_score": 2.0},
                3: {"sasrec_direct_score": 1.0},
            },
        }]
        item_order = ["1", "2", "3", "4"]
        k2_rows, _, k2_ranks = direct.make_candidate_rows(scored, item_order, 2)
        k4_rows, _, k4_ranks = direct.make_candidate_rows(scored, item_order, 4)
        self.assertEqual(k2_rows[0]["candidate_item_ids"], k4_rows[0]["candidate_item_ids"][:2])
        self.assertEqual(k2_ranks, [0])
        self.assertEqual(k4_ranks, [0])

    def test_reject_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.jsonl"
            path.write_text("already here\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                direct.reject_existing({"candidates": path})

    def test_output_root_and_run_id_isolate_paths(self):
        args = type("Args", (), {
            "output_dir": None,
            "output_root": Path("/tmp/s6"),
            "run_id": "rerun1",
        })()
        path = direct.output_root_for(args, "valid_fit", 20, False)
        self.assertEqual(path, Path("/tmp/s6/valid_fit/rerun1/k20"))


if __name__ == "__main__":
    unittest.main()
