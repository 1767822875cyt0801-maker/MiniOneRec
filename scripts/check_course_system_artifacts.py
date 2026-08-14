#!/usr/bin/env python3
"""Read-only artifact checker for course-system runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minionerec_system.checker import check_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a course-system run directory without repairing it.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    report = check_run(parse_args().run_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
