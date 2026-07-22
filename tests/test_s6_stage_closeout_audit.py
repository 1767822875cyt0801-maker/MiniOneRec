import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import s6_stage_closeout_audit as closeout


class S6StageCloseoutAuditTests(unittest.TestCase):
    def test_oracle_from_uses_union_pool_as_candidate_recall(self):
        metrics = {
            "num_samples": 10,
            "union_target_in_pool_count": 4,
            "cf_hits_at20": 2,
            "direct_only_targets_available_at20": 3,
        }
        oracle = closeout.oracle_from(metrics)
        self.assertEqual(oracle["hits@20"], 4)
        self.assertAlmostEqual(oracle["hr@20"], 0.4)
        self.assertIsNone(oracle["ndcg@20"])
        self.assertEqual(oracle["direct_only_targets_recovered_at20"], 3)

    def test_compact_metrics_keeps_core_fields(self):
        metrics = {
            "hr@20": 0.1,
            "ndcg@20": 0.2,
            "hits@20": 3,
            "irrelevant": "drop",
        }
        compact = closeout.compact_metrics(metrics)
        self.assertEqual(compact["hr@20"], 0.1)
        self.assertNotIn("irrelevant", compact)

    def test_test_path_rejected(self):
        with self.assertRaises(ValueError):
            closeout.reject_test_path(Path("results/final_test/metrics.json"))

    def test_cost_status_is_pending_without_timing_fields(self):
        status = closeout.comparable_qwen_cost_status()
        self.assertEqual(status["status"], "COST_EVIDENCE_PENDING")
        self.assertFalse(status["comparable_evidence_found"])
        self.assertIn("autodl_profiling_command", status)

    def test_build_report_verdict_and_no_test(self):
        report = closeout.build_report(closeout.DEFAULT_OUTPUT_ROOT)
        self.assertEqual(report["overall_verdict"], "S6_DIRECT_RETRIEVAL_GO_RANKING_PARTIAL_CLOSEOUT")
        self.assertFalse(report["test_read"])
        self.assertEqual(report["negative_evidence"]["s6_6r"]["valid_gate_opened"], False)


if __name__ == "__main__":
    unittest.main()
