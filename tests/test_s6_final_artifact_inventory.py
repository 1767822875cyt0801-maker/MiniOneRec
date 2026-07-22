import tempfile
import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s6_final_artifact_inventory as inv


class S6FinalArtifactInventoryTests(unittest.TestCase):
    def test_allowlist_is_bounded_and_ordered_deterministically(self):
        paths = [spec.path for spec in inv.ALLOWLIST]
        self.assertIn("docs/s6_final", paths)
        self.assertLess(len(paths), 100)
        inventory = inv.build_inventory(full_hash=False)
        emitted = [entry["relative_path"] for entry in inventory["entries"]]
        self.assertEqual(emitted, sorted(emitted))

    def test_path_normalization_rejects_absolute_traversal_and_test(self):
        self.assertEqual(inv.normalize_rel_path("docs/s6_final").as_posix(), "docs/s6_final")
        with self.assertRaises(ValueError):
            inv.normalize_rel_path("/tmp/x")
        with self.assertRaises(ValueError):
            inv.normalize_rel_path("../x")
        with self.assertRaises(ValueError):
            inv.normalize_rel_path("results/final_test/x.json")

    def test_canonical_ablation_and_negative_classifications_exist(self):
        statuses = {spec.status for spec in inv.ALLOWLIST}
        self.assertIn("canonical", statuses)
        self.assertIn("ablation", statuses)
        self.assertIn("negative", statuses)

    def test_small_file_hashing_and_full_hash_opt_in(self):
        file_entry = inv.inventory_entry(
            inv.ArtifactSpec("docs/s6_stage_closeout.md", "S6-7", "canonical"),
            full_hash=False,
        )
        self.assertEqual(file_entry["type"], "file")
        self.assertIsNotNone(file_entry["sha256"])
        dir_entry = inv.inventory_entry(
            inv.ArtifactSpec("docs", "test", "canonical", "directory"),
            full_hash=False,
        )
        self.assertEqual(dir_entry["hash_policy"], "directory_not_recursed")
        full_dir_entry = inv.inventory_entry(
            inv.ArtifactSpec("docs/s6_final", "test", "canonical", "directory"),
            full_hash=True,
        )
        self.assertEqual(full_dir_entry["hash_policy"], "full_hash_directory_tree")

    def test_missing_artifact_reporting(self):
        entry = inv.inventory_entry(
            inv.ArtifactSpec("docs/s6_final/does_not_exist.md", "missing", "canonical"),
            full_hash=False,
        )
        self.assertFalse(entry["exists"])
        self.assertEqual(entry["type"], "missing")

    def test_overwrite_rejection(self):
        inventory = {"full_hash": False, "entry_count": 0, "missing_count": 0, "entries": []}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            inv.write_outputs(out, inventory, overwrite=False)
            with self.assertRaises(FileExistsError):
                inv.write_outputs(out, inventory, overwrite=False)
            inv.write_outputs(out, inventory, overwrite=True)


if __name__ == "__main__":
    unittest.main()
