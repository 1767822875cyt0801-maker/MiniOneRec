import csv
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import audit_amazon_feedback_semantics as audit


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestAuditAmazonFeedbackSemantics(unittest.TestCase):
    def _processed(self, tmp_path: Path, with_rating: bool = False, bad_history: bool = False):
        fields = ["user_id", "history_item_id", "item_id"]
        if with_rating:
            fields.append("rating")
        train = tmp_path / "train.csv"
        valid = tmp_path / "valid.csv"
        test = tmp_path / "heldout.csv"
        train_row = {"user_id": "u1", "history_item_id": "['i1']", "item_id": "i2"}
        valid_row = {"user_id": "u2", "history_item_id": "not_a_list" if bad_history else "['i2']", "item_id": "i3"}
        test_row = {"user_id": "u3", "history_item_id": "['i3']", "item_id": "i4"}
        if with_rating:
            train_row["rating"] = "5"
            valid_row["rating"] = ""
            test_row["rating"] = "1"
        write_csv(train, fields, [train_row])
        write_csv(valid, fields, [valid_row])
        write_csv(test, fields, [test_row])
        return train, valid, test

    def test_raw_rating_present_processed_rating_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train, valid, test = self._processed(tmp_path)
            raw = tmp_path / "raw.jsonl"
            raw.write_text(
                "\n".join(
                    [
                        json.dumps({"reviewerID": "u1", "asin": "i2", "overall": 5.0, "unixReviewTime": 1}),
                        json.dumps({"reviewerID": "u2", "asin": "i3", "overall": 2.0, "unixReviewTime": 2}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            out = tmp_path / "out"
            result = audit.run_audit(
                Namespace(
                    category="Tiny",
                    train_csv=train,
                    valid_csv=valid,
                    test_csv=test,
                    raw_interactions=raw,
                    output_dir=out,
                    max_raw_rows=0,
                )
            )
            semantics = result["interaction_semantics"]
            self.assertEqual(semantics["raw_rating_status"], "raw_contains_rating")
            self.assertEqual(semantics["processed_rating_status"], "processed_csv_missing_rating")
            rating_csv = (out / "rating_distribution.csv").read_text(encoding="utf-8")
            self.assertIn("processed_target_train", rating_csv)
            self.assertIn("raw,5.0,1", rating_csv)

    def test_raw_missing_is_not_interpreted_as_no_rating(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train, valid, test = self._processed(tmp_path)
            out = tmp_path / "out"
            result = audit.run_audit(
                Namespace(
                    category="Tiny",
                    train_csv=train,
                    valid_csv=valid,
                    test_csv=test,
                    raw_interactions=tmp_path / "missing.jsonl",
                    output_dir=out,
                    max_raw_rows=0,
                )
            )
            semantics = result["interaction_semantics"]
            self.assertEqual(semantics["raw_rating_status"], "raw_unavailable_cannot_verify_rating")
            self.assertFalse(semantics["raw_field_can_verify_rating"])

    def test_aliases_bad_history_and_split_misalignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train, valid, test = self._processed(tmp_path, with_rating=True, bad_history=True)
            raw = tmp_path / "raw.csv"
            write_csv(
                raw,
                ["uid", "iid", "score", "event"],
                [
                    {"uid": "u1", "iid": "i2", "score": "4", "event": "rating"},
                    {"uid": "u9", "iid": "i9", "score": "", "event": "rating"},
                ],
            )
            out = tmp_path / "out"
            result = audit.run_audit(
                Namespace(
                    category="Tiny",
                    train_csv=train,
                    valid_csv=valid,
                    test_csv=test,
                    raw_interactions=raw,
                    output_dir=out,
                    max_raw_rows=0,
                )
            )
            raw_inventory = result["raw_field_inventory"]
            processed = result["processed_field_inventory"]
            split = result["split_consistency"]
            self.assertEqual(raw_inventory["field_aliases"]["rating"], "score")
            self.assertTrue(processed["train"]["has_rating_field"])
            self.assertEqual(processed["valid"]["history_parse_errors"], 1)
            self.assertEqual(split["valid_users_not_in_train_count"], 1)
            self.assertEqual(result["interaction_semantics"]["processed_rating_status"], "processed_csv_contains_rating")


if __name__ == "__main__":
    unittest.main()
