#!/usr/bin/env python3
"""Rewrite MiniOneRec CSV SID columns with a new item_id -> SID mapping."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path
from typing import Any

from utils_sid import load_json, normalize_sid, parse_sid_tokens


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value_str = str(value).strip().lower()
    if value_str in {"1", "true", "yes", "y", "on"}:
        return True
    if value_str in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite history_item_sid/item_sid columns using item2sid mapping.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--item2sid", type=Path, required=True)
    parser.add_argument("--history-item-col", default="history_item_id")
    parser.add_argument("--target-item-col", default="item_id")
    parser.add_argument("--history-sid-col", default="history_item_sid")
    parser.add_argument("--target-sid-col", default="item_sid")
    parser.add_argument("--keep-old-sid", type=str2bool, default=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_item_id(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def parse_item_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [normalize_item_id(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_item_id(item) for item in value]

    value_str = "" if value is None else str(value).strip()
    if not value_str:
        return []

    try:
        parsed = ast.literal_eval(value_str)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, (list, tuple)):
        return [normalize_item_id(item) for item in parsed]
    if parsed is not None and not isinstance(parsed, (dict, set)):
        return [normalize_item_id(parsed)]

    parts = [part for part in re.split(r"[,\s;]+", value_str) if part]
    return [normalize_item_id(part) for part in parts]


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def rate(num: int | float, den: int | float) -> float:
    if den == 0:
        return 1.0
    return float(num) / float(den)


def load_item2sid(path: Path) -> dict[str, str]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected item2sid JSON object: {path}")
    mapping = {normalize_item_id(item_id): normalize_sid(str(sid)) for item_id, sid in raw.items()}
    bad = {item_id: sid for item_id, sid in mapping.items() if not parse_sid_tokens(sid)}
    if bad:
        sample = list(bad.items())[:20]
        raise ValueError(f"item2sid contains invalid SID values, sample={sample}")
    return mapping


def main() -> None:
    args = parse_args()
    if args.input_csv.resolve() == args.output_csv.resolve():
        raise ValueError("output-csv must not be the same file as input-csv")
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output CSV already exists: {args.output_csv}. Use --overwrite to replace it.")

    item2sid = load_item2sid(args.item2sid)
    missing_items: dict[str, int] = {}
    output_rows: list[dict[str, Any]] = []

    target_success = 0
    target_missing = 0
    history_total = 0
    history_success = 0
    history_missing = 0
    history_length_mismatch = 0
    sid_parse_ok = 0
    sid_parse_total = 0
    old_new_same_target = 0

    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        required = {
            args.history_item_col,
            args.target_item_col,
            args.history_sid_col,
            args.target_sid_col,
        }
        missing_cols = sorted(required - set(fieldnames))
        if missing_cols:
            raise ValueError(f"{args.input_csv} missing columns: {missing_cols}")

        old_history_col = f"{args.history_sid_col}_old"
        old_target_col = f"{args.target_sid_col}_old"
        output_fieldnames = list(fieldnames)
        if args.keep_old_sid:
            if old_history_col not in output_fieldnames:
                output_fieldnames.append(old_history_col)
            if old_target_col not in output_fieldnames:
                output_fieldnames.append(old_target_col)

        for row_no, row in enumerate(reader, start=2):
            row_out = dict(row)
            old_history_sid = row.get(args.history_sid_col, "")
            old_target_sid = normalize_sid(row.get(args.target_sid_col, ""))
            if args.keep_old_sid:
                row_out[old_history_col] = old_history_sid
                row_out[old_target_col] = row.get(args.target_sid_col, "")

            target_item_id = normalize_item_id(row.get(args.target_item_col, ""))
            target_sid = item2sid.get(target_item_id)
            sid_parse_total += 1
            if target_sid is None:
                target_missing += 1
                missing_items[target_item_id] = missing_items.get(target_item_id, 0) + 1
            else:
                target_success += 1
                row_out[args.target_sid_col] = target_sid
                old_new_same_target += int(old_target_sid == target_sid)
                sid_parse_ok += int(bool(parse_sid_tokens(target_sid)))

            history_item_ids = parse_item_list(row.get(args.history_item_col, ""))
            try:
                old_history_sids = parse_item_list(old_history_sid)
            except Exception:
                old_history_sids = []
            if len(old_history_sids) != len(history_item_ids):
                history_length_mismatch += 1

            new_history_sids: list[str] = []
            for item_id in history_item_ids:
                history_total += 1
                sid_parse_total += 1
                sid = item2sid.get(item_id)
                if sid is None:
                    history_missing += 1
                    missing_items[item_id] = missing_items.get(item_id, 0) + 1
                    continue
                history_success += 1
                new_history_sids.append(sid)
                sid_parse_ok += int(bool(parse_sid_tokens(sid)))
            row_out[args.history_sid_col] = repr(new_history_sids)
            output_rows.append(row_out)

    report = {
        "input_csv": args.input_csv.as_posix(),
        "output_csv": args.output_csv.as_posix(),
        "item2sid": args.item2sid.as_posix(),
        "num_rows": len(output_rows),
        "target_rewrite_success": target_success,
        "target_missing_count": target_missing,
        "history_total_items": history_total,
        "history_rewrite_success": history_success,
        "history_missing_count": history_missing,
        "history_length_mismatch_count": history_length_mismatch,
        "sid_parse_success_rate": rate(sid_parse_ok, sid_parse_total),
        "old_new_same_target_sid_rate": rate(old_new_same_target, target_success),
        "missing_items_sample": [
            {"item_id": item_id, "count": count}
            for item_id, count in sorted(missing_items.items(), key=lambda item: (-item[1], item[0]))[:50]
        ],
        "keep_old_sid": args.keep_old_sid,
        "overwrite": bool(args.overwrite),
    }

    if missing_items:
        write_report(args.report_path, report)
        raise KeyError(
            f"Missing {len(missing_items)} unique item ids in item2sid. "
            f"Report written to {args.report_path}."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    write_report(args.report_path, report)
    print(f"Wrote rewritten CSV: {args.output_csv}")
    print(f"Wrote rewrite report: {args.report_path}")
    print(
        "Rewrite summary: "
        f"rows={report['num_rows']} "
        f"target_missing={target_missing} "
        f"history_missing={history_missing} "
        f"sid_parse_success_rate={report['sid_parse_success_rate']:.6f}"
    )


if __name__ == "__main__":
    main()
