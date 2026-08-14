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
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if sum([args.dry_run, args.cardinality_audit, args.plan_only]) > 1:
        parser.error("--dry-run, --cardinality-audit, and --plan-only are mutually exclusive")
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
        run_mode = "cardinality_audit" if args.cardinality_audit else "full_pipeline"
        run_dir = pipeline.run(
            run_mode=run_mode,
            run_id=args.run_id,
            max_samples=args.max_samples,
            command=command_text(sys.argv),
        )
        print(json.dumps({"status": "COMPLETED", "run_mode": run_mode, "run_dir": str(run_dir)}, indent=2))
        return 0
    except (ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"COURSE_SYSTEM_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
