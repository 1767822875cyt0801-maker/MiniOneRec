import unittest
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_s6_direct_sasrec_smoke as audit  # noqa: E402


class S6SmokeAuditTests(unittest.TestCase):
    def test_smoke_bundle_audit_is_go_when_bundle_exists(self):
        if not audit.DEFAULT_BUNDLE.is_file():
            self.skipTest("S6 smoke bundle is not available locally")
        report = audit.audit_bundle(audit.DEFAULT_BUNDLE, audit.DEFAULT_BUNDLE_SHA)
        self.assertEqual(report["verdict"], "GO_SMOKE_AUDIT")
        self.assertEqual(report["row_count"], 2660)
        self.assertEqual(report["invalid_item_count_recomputed"], 0)
        self.assertEqual(report["duplicate_candidate_rows"], 0)
        self.assertEqual(report["metrics_recomputed"]["target_in_pool_count"], 358)


if __name__ == "__main__":
    unittest.main()
