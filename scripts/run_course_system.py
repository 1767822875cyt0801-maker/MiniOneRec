#!/usr/bin/env python3
"""Unified entry point for the frozen-prediction course system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minionerec_system.config import ConfigError, load_course_config
from minionerec_system.pipeline import CoursePipeline, build_command_plan, command_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen-prediction downstream course system.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cardinality-audit", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--formal-preflight", action="store_true")
    parser.add_argument("--expected-course-commit", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if sum([args.dry_run, args.cardinality_audit, args.plan_only, args.formal_preflight]) > 1:
        parser.error("--dry-run, --cardinality-audit, --plan-only, and --formal-preflight are mutually exclusive")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        config = load_course_config(args.config, ROOT)
        if config.is_formal and args.max_samples is not None and not args.dry_run:
            raise ConfigError("formal configs may use --max-samples only with --dry-run")
        pipeline = CoursePipeline(config)
        if args.plan_only:
            print(json.dumps(build_command_plan(config, "full_pipeline"), indent=2, ensure_ascii=False))
            return 0
        if args.dry_run:
            print(json.dumps(pipeline.dry_run(args.max_samples), indent=2, ensure_ascii=False))
            return 0
        if args.formal_preflight:
            if not args.expected_course_commit:
                raise ConfigError("--formal-preflight requires --expected-course-commit")
            if not args.run_id:
                raise ConfigError("--formal-preflight requires --run-id for output collision checking")
            try:
                report = pipeline.formal_preflight(args.expected_course_commit, args.run_id)
            except (ConfigError, ValueError, FileNotFoundError) as exc:
                report = {
                    "schema": "course_formal_preflight.v1",
                    "status": "FAIL",
                    "formal_experiment_run": False,
                    "checks": [
                        {
                            "name": "formal preflight",
                            "status": "FAIL",
                            "expected": "all formal guards PASS",
                            "actual": f"{type(exc).__name__}: {exc}",
                        }
                    ],
                }
                print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
                return 2
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0
        run_mode = "cardinality_audit" if args.cardinality_audit else "full_pipeline"
        run_dir = pipeline.run(
            run_mode=run_mode,
            run_id=args.run_id,
            max_samples=args.max_samples,
            command=command_text(sys.argv),
            expected_course_commit=args.expected_course_commit,
        )
        print(json.dumps({"status": "COMPLETED", "run_mode": run_mode, "run_dir": str(run_dir)}, indent=2))
        return 0
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"COURSE_SYSTEM_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
