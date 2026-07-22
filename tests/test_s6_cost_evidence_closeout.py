import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s6_cost_evidence_closeout as cost


class S6CostEvidenceCloseoutTests(unittest.TestCase):
    def test_elapsed_seconds_from_gnu_time(self):
        text = "\tElapsed (wall clock) time (h:mm:ss or m:ss): 11:02.68\n"
        self.assertAlmostEqual(cost.elapsed_seconds_from_gnu_time(text), 662.68)

    def test_bundle_audit_verdict_and_population(self):
        if not cost.DEFAULT_BUNDLE.exists():
            self.skipTest("cost evidence bundle unavailable")
        report = cost.audit_bundle(cost.DEFAULT_BUNDLE, cost.DEFAULT_SHA)
        self.assertEqual(report["verdict"], "S6_DIRECT_RETRIEVAL_GO_RANKING_PARTIAL_CLOSEOUT")
        self.assertTrue(report["population"]["all_4532_valid_rows_completed"])
        self.assertFalse(report["population"]["test_read"])

    def test_wall_ratio_uses_direct_wall_over_qwen_wall(self):
        if not cost.DEFAULT_BUNDLE.exists():
            self.skipTest("cost evidence bundle unavailable")
        report = cost.audit_bundle(cost.DEFAULT_BUNDLE, cost.DEFAULT_SHA)
        expected = cost.DIRECT_WALL_SECONDS / report["qwen_runtime"]["pipeline_wall_seconds"]
        self.assertAlmostEqual(report["ratios"]["direct_to_qwen_wall_time_ratio"], expected)
        self.assertIsNone(report["ratios"]["model_time_ratio"])


if __name__ == "__main__":
    unittest.main()
