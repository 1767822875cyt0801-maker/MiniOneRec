#!/usr/bin/env python3
"""Check row/query alignment across SID-version CSV splits."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = ["row_index", "user_id", "history_item_id", "item_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check split alignment for Text/Behavior/optional CS CSVs.")
    parser.add_argument("--text-csv", type=Path, required=True)
    parser.add_argument("--behavior-csv", type=Path, required=True)
    parser.add_argument("--cs-csv", type=Path, default=None)
    parser.add_argument("--category", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    errors: list[str] = []
    if not path.exists():
        return [], [f"missing file: {path}"]
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    missing_required = [field for field in ["user_id", "history_item_id", "item_id"] if field not in columns]
    if missing_required:
        errors.append(f"missing required columns: {missing_required}")
    return rows, errors


def normalize_listish(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    if isinstance(parsed, (list, tuple)):
        return json.dumps([str(item) for item in parsed], ensure_ascii=False, separators=(",", ":"))
    return str(parsed)


def field_value(row: dict[str, str], idx: int, field: str) -> str:
    if field == "row_index":
        value = row.get("row_index")
        return str(idx) if value in (None, "") else str(value).strip()
    value = row.get(field, "")
    if field == "history_item_id":
        return normalize_listish(value)
    return str(value).strip()


def compare_rows(
    base_name: str,
    base_rows: list[dict[str, str]],
    other_name: str,
    other_rows: list[dict[str, str]],
    max_samples: int,
) -> dict[str, Any]:
    row_count_match = len(base_rows) == len(other_rows)
    checks: dict[str, Any] = {
        "row_count": {
            "ok": row_count_match,
            "base_rows": len(base_rows),
            "other_rows": len(other_rows),
        }
    }
    compare_len = min(len(base_rows), len(other_rows))
    for field in IDENTITY_FIELDS:
        mismatches: list[dict[str, Any]] = []
        for idx in range(compare_len):
            base_value = field_value(base_rows[idx], idx, field)
            other_value = field_value(other_rows[idx], idx, field)
            if base_value != other_value:
                mismatches.append({
                    "row_index": idx,
                    base_name: base_value,
                    other_name: other_value,
                })
                if len(mismatches) >= max_samples:
                    break
        checks[field] = {
            "ok": not mismatches,
            "mismatch_count_sampled": len(mismatches),
            "mismatch_samples": mismatches,
        }
    checks["overall_ok"] = row_count_match and all(check["ok"] for key, check in checks.items() if key != "overall_ok")
    return checks


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_md(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Split Alignment Report",
        "",
        f"- Category: `{report.get('category') or 'not_provided'}`",
        f"- Split: `{report.get('split') or 'not_provided'}`",
        f"- Overall OK: `{report['overall_ok']}`",
        "",
        "## Inputs",
        "",
    ]
    for name, value in report["inputs"].items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(["", "## Checks", ""])
    for name, checks in report["comparisons"].items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- overall_ok: `{checks['overall_ok']}`")
        for field, check in checks.items():
            if field == "overall_ok":
                continue
            lines.append(f"- {field}: `{check['ok']}`")
        lines.append("")
    if report["errors"]:
        lines.extend(["## Errors", ""])
        for name, errors in report["errors"].items():
            for error in errors:
                lines.append(f"- {name}: {error}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    inputs = {
        "text_csv": args.text_csv.as_posix(),
        "behavior_csv": args.behavior_csv.as_posix(),
    }
    if args.cs_csv is not None:
        inputs["cs_csv"] = args.cs_csv.as_posix()

    text_rows, text_errors = read_csv_rows(args.text_csv)
    behavior_rows, behavior_errors = read_csv_rows(args.behavior_csv)
    errors = {"text": text_errors, "behavior": behavior_errors}
    comparisons = {
        "text_vs_behavior": compare_rows(
            "text",
            text_rows,
            "behavior",
            behavior_rows,
            args.max_samples,
        )
    }
    if args.cs_csv is not None:
        cs_rows, cs_errors = read_csv_rows(args.cs_csv)
        errors["cs"] = cs_errors
        comparisons["text_vs_cs"] = compare_rows("text", text_rows, "cs", cs_rows, args.max_samples)

    overall_ok = not any(errors.values()) and all(item["overall_ok"] for item in comparisons.values())
    report = {
        "category": args.category,
        "split": args.split,
        "overall_ok": overall_ok,
        "inputs": inputs,
        "errors": errors,
        "comparisons": comparisons,
    }
    write_json(args.output_json, report)
    write_md(args.output_md, report)
    print(f"overall_ok={overall_ok}")
    print(f"wrote_json={args.output_json}")
    print(f"wrote_md={args.output_md}")


if __name__ == "__main__":
    main()
