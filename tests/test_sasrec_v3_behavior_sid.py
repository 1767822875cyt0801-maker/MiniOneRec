import argparse
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_sasrec_v3_behavior_sid as s3_builder


def sha256(path: Path) -> str:
    return s3_builder.sha256(path)


class TestSASRecV3BehaviorSID(unittest.TestCase):
    def make_fixture(self, tmp_path: Path, dirname: str = "formal_v3_finite_maskfix_seed42", bad_embedding=None):
        category = "Tiny"
        sasrec_dir = tmp_path / "behavior_embeddings" / "sasrec" / category / dirname
        sasrec_dir.mkdir(parents=True)
        item_order = ["10", "11", "12", "13", "14", "15"]
        row_index = {item: idx for idx, item in enumerate(item_order)}
        embedding = np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        if bad_embedding is not None:
            embedding = bad_embedding
        np.save(sasrec_dir / f"{category}.sasrec_item_emb.npy", embedding)
        self.write_json(sasrec_dir / f"{category}.item_order.json", item_order)
        self.write_json(sasrec_dir / f"{category}.row_index.json", row_index)
        self.write_json(sasrec_dir / f"{category}.train_item_count.json", {"10": 2, "11": 1, "12": 1, "13": 1, "14": 0, "15": 0})
        config = {
            "category": category,
            "normalization": "l2",
            "evaluation_contract": {"candidate_count": len(item_order), "split": "valid"},
        }
        metrics = {"final_valid": {"nonfinite_score_samples": 0, "unranked_samples": 0, "candidate_count_min": len(item_order), "candidate_count_max": len(item_order)}}
        self.write_json(sasrec_dir / "config.json", config)
        self.write_json(sasrec_dir / "metrics_summary.json", metrics)
        manifest = {
            "category": category,
            "embedding_source": "train_only_sasrec",
            "dimension": 4 if embedding.ndim == 2 else 0,
            "num_items": int(embedding.shape[0]) if embedding.ndim else 0,
            "normalization": "l2",
            "train_only_policy": {"test_read": False, "train_used_for_fitting": True, "valid_used_for_checkpoint_selection": True, "cold_fallback_uses_valid_or_test": False},
            "hashes": {
                "embedding": sha256(sasrec_dir / f"{category}.sasrec_item_emb.npy"),
                "item_order": sha256(sasrec_dir / f"{category}.item_order.json"),
                "row_index": sha256(sasrec_dir / f"{category}.row_index.json"),
                "train_item_count": sha256(sasrec_dir / f"{category}.train_item_count.json"),
                "config": sha256(sasrec_dir / "config.json"),
                "metrics_summary": sha256(sasrec_dir / "metrics_summary.json"),
            },
        }
        self.write_json(sasrec_dir / "artifact_manifest.json", manifest)

        data_root = tmp_path / "data"
        train_csv = data_root / "train" / f"{category}_5_2016-10-2018-11.csv"
        valid_csv = data_root / "valid" / f"{category}_5_2016-10-2018-11.csv"
        self.write_csv(train_csv)
        self.write_csv(valid_csv)
        item_json = data_root / "index" / f"{category}.item.json"
        self.write_json(item_json, {item: {"title": f"item {item}"} for item in item_order})
        return {
            "category": category,
            "sasrec_dir": sasrec_dir,
            "train_csv": train_csv,
            "valid_csv": valid_csv,
            "item_json": item_json,
            "output_root": tmp_path / "sid_versions",
            "item_order": item_order,
        }

    def write_json(self, path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["user_id", "history_item_title", "item_title", "history_item_id", "item_id", "history_item_sid", "item_sid"])
            writer.writeheader()
            writer.writerow({"user_id": "u1", "history_item_title": "[]", "item_title": "t", "history_item_id": "['10', '11']", "item_id": "12", "history_item_sid": "['<a_0>','<a_1>']", "item_sid": "<a_2>"})
            writer.writerow({"user_id": "u2", "history_item_title": "[]", "item_title": "t", "history_item_id": "['14']", "item_id": "15", "history_item_sid": "['<a_4>']", "item_sid": "<a_5>"})

    def args(self, fixture, **overrides):
        data = {
            "category": fixture["category"],
            "config_mode": "smoke",
            "sasrec_dir": fixture["sasrec_dir"],
            "sid_version": "sasrec_v3_tiny_k4_dedup",
            "output_root": fixture["output_root"],
            "train_csv": fixture["train_csv"],
            "valid_csv": fixture["valid_csv"],
            "item_json": fixture["item_json"],
            "num_levels": 3,
            "codebook_size": 4,
            "max_iter": 2,
            "seed": 7,
            "batch_size": None,
            "dedup_mode": "append",
            "expected_num_items": 6,
            "expected_dim": 4,
            "dry_run": False,
            "overwrite": False,
        }
        data.update(overrides)
        return argparse.Namespace(**data)

    def test_validate_input_contract_and_cold_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            paths = s3_builder.resolve_paths(self.args(fixture))
            report = s3_builder.validate_sasrec_input(paths, expected_num_items=6, expected_dim=4)
            self.assertEqual(report["embedding_shape"], [6, 4])
            self.assertEqual(report["cold_items"], ["14", "15"])

    def test_bad_shape_nan_v2_and_test_paths_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = self.make_fixture(tmp_path, bad_embedding=np.ones((5, 4), dtype=np.float32))
            with self.assertRaises(ValueError):
                s3_builder.validate_sasrec_input(s3_builder.resolve_paths(self.args(fixture)), 6, 4)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            emb = np.ones((6, 4), dtype=np.float32)
            emb[0, 0] = np.nan
            fixture = self.make_fixture(tmp_path, bad_embedding=emb)
            with self.assertRaises(ValueError):
                s3_builder.validate_sasrec_input(s3_builder.resolve_paths(self.args(fixture)), 6, 4)

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp), dirname="formal_v2_no_leakage_seed42")
            with self.assertRaises(ValueError):
                s3_builder.validate_sasrec_input(s3_builder.resolve_paths(self.args(fixture)), 6, 4)

        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            bad_args = self.args(fixture, valid_csv=Path(tmp) / "data" / "test" / "Tiny.csv")
            with self.assertRaises(ValueError):
                s3_builder.validate_sasrec_input(s3_builder.resolve_paths(bad_args), 6, 4)

    def test_dry_run_does_not_write_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            args = self.args(fixture, dry_run=True)
            result = s3_builder.build(args)
            self.assertTrue(result["dry_run"])
            self.assertFalse((fixture["output_root"] / args.sid_version / fixture["category"]).exists())

    def test_smoke_build_rewrites_train_valid_and_reports_cold(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            args = self.args(fixture)
            result = s3_builder.build(args)
            out = fixture["output_root"] / args.sid_version / fixture["category"]
            self.assertTrue((out / "item2sid.json").exists())
            self.assertTrue((out / "train.csv").exists())
            self.assertTrue((out / "valid.csv").exists())
            self.assertFalse((out / "test.csv").exists())
            report = result["static_report"]
            self.assertEqual(report["num_items"], 6)
            self.assertEqual(report["cold_item_analysis"]["cold_item_count"], 2)
            self.assertEqual(report["collision_stats"]["full_unique"], 6)
            self.assertEqual(result["csv_validation"]["train"]["rows"], 2)
            with self.assertRaises(FileExistsError):
                s3_builder.refuse_existing_formal_outputs(out, overwrite=False)

    def test_reproducible_item2sid_for_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            args1 = self.args(fixture, output_root=Path(tmp) / "out1")
            args2 = self.args(fixture, output_root=Path(tmp) / "out2")
            s3_builder.build(args1)
            s3_builder.build(args2)
            p1 = Path(tmp) / "out1" / args1.sid_version / fixture["category"] / "item2sid.json"
            p2 = Path(tmp) / "out2" / args2.sid_version / fixture["category"] / "item2sid.json"
            self.assertEqual(p1.read_text(encoding="utf-8"), p2.read_text(encoding="utf-8"))

    def test_shell_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = self.make_fixture(Path(tmp))
            env = {
                **dict(),
                "CATEGORY": fixture["category"],
                "CONFIG_MODE": "smoke",
                "DRY_RUN": "1",
                "SID_VERSION": "sasrec_v3_tiny_k4_dedup",
                "SASREC_DIR": fixture["sasrec_dir"].as_posix(),
                "OUTPUT_ROOT": fixture["output_root"].as_posix(),
                "TRAIN_CSV": fixture["train_csv"].as_posix(),
                "VALID_CSV": fixture["valid_csv"].as_posix(),
                "ITEM_JSON": fixture["item_json"].as_posix(),
                "CODEBOOK_SIZE": "4",
                "MAX_ITER": "2",
                "EXPECTED_NUM_ITEMS": "6",
                "EXPECTED_DIM": "4",
                "PYTHON": sys.executable,
            }
            full_env = __import__("os").environ.copy()
            full_env.update(env)
            proc = subprocess.run(
                ["bash", "scripts/run_sasrec_v3_behavior_sid.sh"],
                cwd=ROOT,
                env=full_env,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("S3 SASRec Behavior-SID DRY_RUN", proc.stdout)


if __name__ == "__main__":
    unittest.main()
