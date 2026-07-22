import json
import unittest
from types import SimpleNamespace
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s6_sasrec_checkpoint_audit as audit  # noqa: E402


class S6CheckpointAuditTests(unittest.TestCase):
    def test_checkpoint_report_is_go_and_embedding_exact(self):
        report = audit.build_report(SimpleNamespace(
            checkpoint=audit.DEFAULT_CHECKPOINT,
            sha256_file=audit.DEFAULT_SHA256,
            config=audit.DEFAULT_CONFIG,
            manifest=audit.DEFAULT_MANIFEST,
            exported_embedding=audit.DEFAULT_EXPORTED,
            output=audit.DEFAULT_OUTPUT,
        ))
        self.assertEqual(report["verdict"], "GO_CHECKPOINT_COMPATIBLE")
        self.assertEqual(report["checkpoint_sha256"], audit.EXPECTED_SHA256)
        self.assertEqual(report["state_dict"]["tensor_count"], 28)
        self.assertTrue(report["structure_audit"]["complete_sequence_model"])
        self.assertEqual(report["embedding_export_comparison"]["max_absolute_error"], 0.0)
        self.assertEqual(report["embedding_export_comparison"]["cold_item_count"], 39)

    def test_report_file_verdict_matches_expected(self):
        path = ROOT / "results/s6_cost_aware_aux/Industrial_and_Scientific/s6_checkpoint_compatibility_report.json"
        self.assertTrue(path.is_file())
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(report["verdict"], "GO_CHECKPOINT_COMPATIBLE")


if __name__ == "__main__":
    unittest.main()
