import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_s6_validation_split as split  # noqa: E402


def write_valid_csv(path: Path) -> None:
    rows = [
        {"user_id": "u1", "history_item_id": "[1]", "item_id": "2"},
        {"user_id": "u2", "history_item_id": "[3]", "item_id": "4"},
        {"user_id": "u1", "history_item_id": "[2]", "item_id": "5"},
        {"user_id": "u3", "history_item_id": "[6]", "item_id": "7"},
        {"user_id": "u4", "history_item_id": "[8]", "item_id": "9"},
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "history_item_id", "item_id"])
        writer.writeheader()
        writer.writerows(rows)


class S6ValidationSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_user_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid = Path(tmp) / "valid.csv"
            write_valid_csv(valid)
            first = split.build_manifest(valid, "Tiny", 42, "user_id")
            second = split.build_manifest(valid, "Tiny", 42, "user_id")
        self.assertEqual(first, second)
        self.assertEqual(first["source_valid_rows"], 5)
        self.assertEqual(first["total_split_rows"], 5)
        self.assertEqual(first["test_read"], False)
        self.assertTrue(all(value == 0 for value in first["overlap_checks"].values()))
        self.assertEqual(sum(item["row_count"] for item in first["splits"].values()), 5)

    def test_split_hash_changes_with_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid = Path(tmp) / "valid.csv"
            write_valid_csv(valid)
            first = split.build_manifest(valid, "Tiny", 42, "user_id")
            second = split.build_manifest(valid, "Tiny", 43, "user_id")
        self.assertNotEqual(
            json.dumps(first["splits"], sort_keys=True),
            json.dumps(second["splits"], sort_keys=True),
        )

    def test_test_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            test_csv = Path(tmp) / "test.csv"
            write_valid_csv(test_csv)
            with self.assertRaises(ValueError):
                split.build_manifest(test_csv, "Tiny", 42, "user_id")

    def test_missing_split_key_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid = Path(tmp) / "valid.csv"
            write_valid_csv(valid)
            with self.assertRaises(ValueError):
                split.build_manifest(valid, "Tiny", 42, "missing_user")


if __name__ == "__main__":
    unittest.main()
