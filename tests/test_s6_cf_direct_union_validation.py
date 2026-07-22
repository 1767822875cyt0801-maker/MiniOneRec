import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s6_cf_direct_union_validation as union


def cf_row(row_index, target, history, candidates):
    return {
        "row_index": str(row_index),
        "target_item_id": str(target),
        "history_item_id": [str(item) for item in history],
        "candidate_item_ids": [str(item) for item in candidates],
        "candidate_details": [
            {
                "item_id": str(item),
                "source_type": "exact",
                "sid_rank_0_based": rank,
                "source_sid": f"<sid_{item}>",
            }
            for rank, item in enumerate(candidates)
        ],
    }


def direct_row(row_index, target, history, candidates):
    return {
        "row_index": str(row_index),
        "target_item_id": str(target),
        "history_item_id": [str(item) for item in history],
        "candidate_item_ids": [str(item) for item in candidates],
        "candidate_details": [
            {
                "item_id": str(item),
                "source_type": "sasrec_direct",
                "sid_rank_0_based": rank,
                "sasrec_direct_rank_0_based": rank,
                "sasrec_direct_rank": float(rank + 1),
                "sasrec_direct_reciprocal_rank": 1.0 / float(rank + 1),
                "sasrec_direct_score": float(10 - rank),
            }
            for rank, item in enumerate(candidates)
        ],
    }


class S6CFDirectUnionValidationTest(unittest.TestCase):
    def test_filter_cf_split_preserves_split_order_and_hashes(self):
        rows = [
            cf_row(0, "t0", ["h0"], ["a"]),
            cf_row(1, "t1", ["h1"], ["b"]),
        ]
        expected = [
            {"source_row_index": "1", "item_id": "t1", "history_item_id": "[h1]"},
            {"source_row_index": "0", "item_id": "t0", "history_item_id": "[h0]"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidates.jsonl"
            report = Path(tmp) / "candidate_report.json"
            result = union.filter_cf_split(rows, expected, out, report, source_cf_hash="abc", run_id="unit")
            written = union.read_jsonl(out)
            self.assertEqual(["1", "0"], [row["row_index"] for row in written])
            self.assertEqual(result["split_row_count"], 2)
            self.assertEqual(result["output_sha256"], union.file_sha256(out))

    def test_filter_cf_split_rejects_existing_output(self):
        rows = [cf_row(0, "t0", ["h0"], ["a"])]
        expected = [{"source_row_index": "0", "item_id": "t0", "history_item_id": "[h0]"}]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "candidates.jsonl"
            report = Path(tmp) / "candidate_report.json"
            out.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                union.filter_cf_split(rows, expected, out, report, source_cf_hash="abc", run_id="unit")

    def test_filter_cf_split_rejects_misaligned_target(self):
        rows = [cf_row(0, "wrong", ["h0"], ["a"])]
        expected = [{"source_row_index": "0", "item_id": "t0", "history_item_id": "[h0]"}]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                union.filter_cf_split(
                    rows,
                    expected,
                    Path(tmp) / "candidates.jsonl",
                    Path(tmp) / "candidate_report.json",
                    source_cf_hash="abc",
                    run_id="unit",
                )

    def test_union_preserves_provenance_deduplicates_and_accounts_targets(self):
        cf = [cf_row(0, "t", ["h"], ["a", "b"])]
        direct = [direct_row(0, "t", ["h"], ["b", "t", "c"])]
        merged = union.merge_union_rows(cf, direct)
        self.assertEqual(merged[0]["candidate_item_ids"], ["a", "b", "t", "c"])
        details = {row["item_id"]: row for row in merged[0]["candidate_details"]}
        self.assertTrue(details["b"]["overlap"])
        self.assertEqual(details["b"]["cf_rank_0_based"], 1)
        self.assertEqual(details["b"]["sasrec_direct_rank_0_based"], 0)
        self.assertIsNotNone(details["t"]["sasrec_direct_detail"])
        metrics = union.union_metrics(cf, direct, merged)
        self.assertEqual(metrics["cf_target_count"], 0)
        self.assertEqual(metrics["direct_target_count"], 1)
        self.assertEqual(metrics["union_target_count"], 1)
        self.assertEqual(metrics["direct_only_target_count"], 1)
        self.assertEqual(metrics["candidate_overlap_count"], 1)
        self.assertEqual(metrics["added_candidate_count"], 2)
        self.assertEqual(metrics["duplicate_candidate_count_after_union"], 0)
        self.assertEqual(metrics["candidate_provenance_failures"], 0)

    def test_cf_target_preservation(self):
        cf = [cf_row(0, "t", ["h"], ["t", "a"])]
        direct = [direct_row(0, "t", ["h"], ["b", "c"])]
        merged = union.merge_union_rows(cf, direct)
        metrics = union.union_metrics(cf, direct, merged)
        self.assertEqual(metrics["cf_target_count"], 1)
        self.assertEqual(metrics["cf_target_preservation_rate"], 1.0)

    def test_select_k_smallest_passing_fit_and_select(self):
        metrics = {
            "valid_fit": {
                20: {"uplift_retention": 0.8},
                50: {"uplift_retention": 0.91},
                100: {"uplift_retention": 1.0},
            },
            "valid_select": {
                20: {"uplift_retention": 0.95},
                50: {"uplift_retention": 0.92},
                100: {"uplift_retention": 1.0},
            },
            "valid_gate": {
                20: {"uplift_retention": 0.1},
                50: {"uplift_retention": 0.1},
                100: {"uplift_retention": 0.1},
            },
        }
        selected = union.select_k(metrics)
        self.assertEqual(selected["selected_k"], 50)

    def test_reject_test_paths(self):
        with self.assertRaises(ValueError):
            union.reject_test_path(Path("/tmp/project/data/test/file.csv"))


if __name__ == "__main__":
    unittest.main()
