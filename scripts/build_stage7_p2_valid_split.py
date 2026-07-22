#!/usr/bin/env python3
"""Build the fixed P2 valid_fit/valid_select manifest.

This is a protocol step, not a model step. It reads validated Stage 7 artifacts
and the P2-0 headroom report, then assigns validation rows to a deterministic
valid_fit/valid_select split using only seed + row_index.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("results/stage7_validation_protocol/valid/Industrial_and_Scientific")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed P2 valid_fit/valid_select manifest.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--split", choices=["valid"], default="valid")
    parser.add_argument("--candidate-mode", choices=["exact"], default="exact")
    parser.add_argument("--fit-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object rows in {path}")
                rows.append(value)
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{value}\n")


def item_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def normalize_id(value: Any, fallback: int) -> str:
    if value is None or value == "":
        return str(fallback)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        row_id = normalize_id(row.get("row_index", row.get("sample_id")), idx)
        if row_id in indexed:
            raise ValueError(f"Duplicate row id in {label}: {row_id}")
        indexed[row_id] = row
    return indexed


def list_field(row: dict[str, Any], field: str) -> list[str]:
    values = row.get(field, [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values]


def rank_of(target_item_id: str, items: list[str]) -> int | None:
    target = str(target_item_id)
    for idx, item_id in enumerate(items):
        if str(item_id) == target:
            return idx
    return None


def hit_at(rank: int | None, k: int) -> bool:
    return rank is not None and rank < k


def stable_hash(seed: int, row_index: str) -> tuple[str, int]:
    digest = hashlib.sha256(f"{seed}:{row_index}".encode("utf-8")).hexdigest()
    return digest, int(digest, 16)


def rate(count: int | float, total: int) -> float:
    return 0.0 if total == 0 else float(count) / float(total)


def bool_int(value: bool) -> int:
    return 1 if value else 0


def source_group(record: dict[str, Any]) -> str:
    text_hit = hit_at(record["text_rank_0_based"], 20)
    cf_hit = hit_at(record["cf_rank_0_based"], 20)
    if text_hit and cf_hit:
        return "both_top20"
    if text_hit:
        return "text_only_top20"
    if cf_hit:
        return "cf_only_top20"
    if record["fusion_rank_0_based"] is not None:
        return "union_rank_gt20"
    return "not_in_union"


def one_based(rank: int | None) -> int | None:
    return None if rank is None else rank + 1


def no_test_path(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    return "test" not in lowered_parts and path.name.lower() != "test.csv"


def is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def build_records(root: Path, seed: int) -> list[dict[str, Any]]:
    text_path = root / "text/candidates/candidates.jsonl"
    cf_path = root / "cf/candidates/candidates.jsonl"
    fusion_path = root / "fusion/dual_fused_candidates.jsonl"
    rerank_path = root / "rerank/reranked_candidates.jsonl"
    for path in [text_path, cf_path, fusion_path, rerank_path]:
        assert_file(path)

    text_rows = index_rows(read_jsonl(text_path), "text candidates")
    cf_rows = index_rows(read_jsonl(cf_path), "cf candidates")
    fusion_rows = index_rows(read_jsonl(fusion_path), "fusion candidates")
    rerank_rows = index_rows(read_jsonl(rerank_path), "rerank candidates")
    id_sets = [set(text_rows), set(cf_rows), set(fusion_rows), set(rerank_rows)]
    if len({frozenset(values) for values in id_sets}) != 1:
        raise ValueError(
            "Stage 7 row ids are not aligned: "
            f"text={len(text_rows)} cf={len(cf_rows)} fusion={len(fusion_rows)} rerank={len(rerank_rows)}"
        )

    records: list[dict[str, Any]] = []
    for row_id in sorted(text_rows, key=item_sort_key):
        rows = {
            "text": text_rows[row_id],
            "cf": cf_rows[row_id],
            "fusion": fusion_rows[row_id],
            "rerank": rerank_rows[row_id],
        }
        targets = {
            str(row.get("target_item_id", "")).strip()
            for row in rows.values()
            if str(row.get("target_item_id", "")).strip()
        }
        if len(targets) != 1:
            raise ValueError(f"Target mismatch for row_index={row_id}: {sorted(targets)}")
        target_item_id = next(iter(targets))
        text_items = list_field(rows["text"], "candidate_item_ids")
        cf_items = list_field(rows["cf"], "candidate_item_ids")
        fusion_items = list_field(rows["fusion"], "candidate_item_ids")
        rerank_items = list_field(rows["rerank"], "reranked_item_ids")

        record = {
            "row_index": row_id,
            "target_item_id": target_item_id,
            "text_rank_0_based": rank_of(target_item_id, text_items),
            "cf_rank_0_based": rank_of(target_item_id, cf_items),
            "fusion_rank_0_based": rank_of(target_item_id, fusion_items),
            "heuristic_rank_0_based": rank_of(target_item_id, rerank_items),
            "num_text_candidates": len(text_items),
            "num_cf_candidates": len(cf_items),
            "num_union_candidates": len(fusion_items),
            "num_reranked_candidates": len(rerank_items),
        }
        digest, score = stable_hash(seed, row_id)
        record["split_hash"] = digest
        record["split_score"] = score
        record["source_group"] = source_group(record)
        record["text_hit20"] = bool_int(hit_at(record["text_rank_0_based"], 20))
        record["cf_hit20"] = bool_int(hit_at(record["cf_rank_0_based"], 20))
        record["oracle_union_hit20"] = bool_int(record["text_hit20"] == 1 or record["cf_hit20"] == 1)
        record["fusion_hit20"] = bool_int(hit_at(record["fusion_rank_0_based"], 20))
        record["heuristic_hit20"] = bool_int(hit_at(record["heuristic_rank_0_based"], 20))
        record["recoverable_missed20"] = bool_int(record["oracle_union_hit20"] == 1 and record["heuristic_hit20"] == 0)
        records.append(record)
    return records


def assign_splits(records: list[dict[str, Any]], fit_count: int) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: (row["split_score"], item_sort_key(row["row_index"])))
    fit_ids = {row["row_index"] for row in ordered[:fit_count]}
    out: list[dict[str, Any]] = []
    split_ordinals = defaultdict(int)
    for row in sorted(records, key=lambda value: item_sort_key(value["row_index"])):
        split_name = "valid_fit" if row["row_index"] in fit_ids else "valid_select"
        split_ordinals[split_name] += 1
        record = dict(row)
        record["p2_split"] = split_name
        record["split_ordinal"] = split_ordinals[split_name] - 1
        for field in ["text_rank", "cf_rank", "fusion_rank", "heuristic_rank"]:
            rank0 = record.pop(f"{field}_0_based")
            record[f"{field}_1_based"] = one_based(rank0)
        out.append(record)
    return out


def split_balance_rows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_by_split = Counter(row["p2_split"] for row in manifest_rows)

    group_counts: dict[tuple[str, str, str], int] = Counter()
    for row in manifest_rows:
        split_name = row["p2_split"]
        group_counts[(split_name, "source_group", row["source_group"])] += 1
        for field in [
            "text_hit20",
            "cf_hit20",
            "oracle_union_hit20",
            "fusion_hit20",
            "heuristic_hit20",
            "recoverable_missed20",
        ]:
            group_counts[(split_name, field, str(row[field]))] += 1

    for (split_name, metric, value), count in sorted(group_counts.items()):
        rows.append(
            {
                "p2_split": split_name,
                "metric": metric,
                "value": value,
                "count": count,
                "rate": rate(count, total_by_split[split_name]),
                "split_size": total_by_split[split_name],
            }
        )
    return rows


def make_summary(
    args: argparse.Namespace,
    root: Path,
    out_dir: Path,
    manifest_rows: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    headroom: dict[str, Any],
) -> dict[str, Any]:
    split_counts = Counter(row["p2_split"] for row in manifest_rows)
    balance = split_balance_rows(manifest_rows)
    rates: dict[str, dict[str, float]] = defaultdict(dict)
    for row in balance:
        rates[row["p2_split"]][f"{row['metric']}={row['value']}"] = row["rate"]

    source_group_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for row in balance:
        if row["metric"] == "source_group":
            source_group_counts[row["p2_split"]][row["value"]] = int(row["count"])

    recommendation = headroom.get("recommendation", {})
    return {
        "split": args.split,
        "category": args.category,
        "candidate_mode": args.candidate_mode,
        "input_root": root.as_posix(),
        "output_dir": out_dir.as_posix(),
        "num_samples": len(manifest_rows),
        "fit_ratio": args.fit_ratio,
        "seed": args.seed,
        "split_policy": "sort by sha256(seed:row_index); first valid_fit_count rows are valid_fit",
        "valid_fit_count": split_counts["valid_fit"],
        "valid_select_count": split_counts["valid_select"],
        "source_group_counts": source_group_counts,
        "balance_rates": rates,
        "headroom_decision": recommendation.get("decision"),
        "headroom_recoverable_gap@20": headroom.get("topk_metrics", {}).get("recoverable_gap_vs_heuristic@20"),
        "checks": checks,
        "overall_ok": all(check["ok"] for check in checks),
        "outputs": {
            "manifest_jsonl": (out_dir / "valid_split_manifest.jsonl").as_posix(),
            "manifest_csv": (out_dir / "valid_split_manifest.csv").as_posix(),
            "summary_json": (out_dir / "valid_split_summary.json").as_posix(),
            "summary_md": (out_dir / "valid_split_summary.md").as_posix(),
            "balance_csv": (out_dir / "split_balance.csv").as_posix(),
            "valid_fit_row_indices": (out_dir / "valid_fit_row_indices.txt").as_posix(),
            "valid_select_row_indices": (out_dir / "valid_select_row_indices.txt").as_posix(),
        },
    }


def make_markdown(summary: dict[str, Any]) -> str:
    source_counts = summary["source_group_counts"]
    groups = sorted(set(source_counts.get("valid_fit", {})) | set(source_counts.get("valid_select", {})))
    lines = [
        "# P2-1 Valid Split Manifest",
        "",
        f"- split: `{summary['split']}`",
        f"- category: `{summary['category']}`",
        f"- candidate_mode: `{summary['candidate_mode']}`",
        f"- num_samples: `{summary['num_samples']}`",
        f"- valid_fit_count: `{summary['valid_fit_count']}`",
        f"- valid_select_count: `{summary['valid_select_count']}`",
        f"- seed: `{summary['seed']}`",
        f"- overall_ok: `{summary['overall_ok']}`",
        "",
        "## Headroom Link",
        "",
        f"- headroom_decision: `{summary['headroom_decision']}`",
        f"- recoverable_gap@20: `{summary['headroom_recoverable_gap@20']}`",
        "",
        "## Source Group Counts",
        "",
        "| Source group | valid_fit | valid_select |",
        "|---|---:|---:|",
    ]
    for group in groups:
        lines.append(
            f"| {group} | {source_counts.get('valid_fit', {}).get(group, 0)} | "
            f"{source_counts.get('valid_select', {}).get(group, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- `valid_fit` is the only split for fitting learned reranker parameters.",
            "- `valid_select` is the only split for selecting reranker variants and hyperparameters.",
            "- `test` remains untouched until the final frozen protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root
    out_dir = args.out_dir or (root / "p2_valid_split")
    if not (0.0 < args.fit_ratio < 1.0):
        raise ValueError(f"--fit-ratio must be in (0, 1), got {args.fit_ratio}")

    headroom_path = root / "p2_headroom/headroom_summary.json"
    consistency_path = root / "summary/consistency_report.json"
    metrics_path = root / "summary/metrics_summary.json"
    for path in [headroom_path, consistency_path, metrics_path]:
        assert_file(path)
    headroom = read_json(headroom_path)
    consistency = read_json(consistency_path)
    metrics = read_json(metrics_path)

    records = build_records(root, args.seed)
    fit_count = int(round(len(records) * args.fit_ratio))
    if not (0 < fit_count < len(records)):
        raise ValueError(f"Invalid valid_fit_count={fit_count} for {len(records)} records")
    manifest_rows = assign_splits(records, fit_count)

    input_paths = [headroom_path, consistency_path, metrics_path]
    output_paths = [
        out_dir / "valid_split_manifest.jsonl",
        out_dir / "valid_split_manifest.csv",
        out_dir / "valid_split_summary.json",
        out_dir / "valid_split_summary.md",
        out_dir / "split_balance.csv",
        out_dir / "valid_fit_row_indices.txt",
        out_dir / "valid_select_row_indices.txt",
    ]
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for path in input_paths:
        add_check(f"input under root: {path.as_posix()}", is_under(path, root), path.as_posix())
        add_check(f"input has no test path: {path.as_posix()}", no_test_path(path), path.as_posix())
    for path in output_paths:
        add_check(f"output under root: {path.as_posix()}", is_under(path, root), path.as_posix())
        add_check(f"output has no test path: {path.as_posix()}", no_test_path(path), path.as_posix())
    add_check("P1 consistency overall_ok", consistency.get("overall_ok") is True, consistency.get("overall_ok"))
    add_check("P2-0 headroom overall_ok", headroom.get("overall_ok") is True, headroom.get("overall_ok"))
    add_check("P2-0 decision continue_to_p2_1", headroom.get("recommendation", {}).get("decision") == "continue_to_p2_1", headroom.get("recommendation", {}).get("decision"))
    add_check("metrics split valid", metrics.get("split") == args.split, metrics.get("split"))
    add_check("metrics candidate mode exact", metrics.get("candidate_mode") == args.candidate_mode, metrics.get("candidate_mode"))
    add_check("row count matches metrics", len(manifest_rows) == int(metrics.get("num_valid_rows", -1)), len(manifest_rows))
    add_check("valid_fit count matches headroom recommendation", fit_count == int(headroom.get("recommendation", {}).get("valid_fit_count", fit_count)), fit_count)
    add_check("valid_select count matches headroom recommendation", len(manifest_rows) - fit_count == int(headroom.get("recommendation", {}).get("valid_select_count", len(manifest_rows) - fit_count)), len(manifest_rows) - fit_count)
    add_check("no overlapping split assignments", len({row["row_index"] for row in manifest_rows}) == len(manifest_rows), len(manifest_rows))

    summary = make_summary(args, root, out_dir, manifest_rows, checks, headroom)

    manifest_fields = [
        "row_index",
        "p2_split",
        "split_ordinal",
        "target_item_id",
        "source_group",
        "text_hit20",
        "cf_hit20",
        "oracle_union_hit20",
        "fusion_hit20",
        "heuristic_hit20",
        "recoverable_missed20",
        "text_rank_1_based",
        "cf_rank_1_based",
        "fusion_rank_1_based",
        "heuristic_rank_1_based",
        "num_text_candidates",
        "num_cf_candidates",
        "num_union_candidates",
        "num_reranked_candidates",
        "split_hash",
        "split_score",
    ]
    write_jsonl(out_dir / "valid_split_manifest.jsonl", manifest_rows)
    write_csv(out_dir / "valid_split_manifest.csv", manifest_fields, manifest_rows)
    write_csv(out_dir / "split_balance.csv", ["p2_split", "metric", "value", "count", "rate", "split_size"], split_balance_rows(manifest_rows))
    write_lines(out_dir / "valid_fit_row_indices.txt", [row["row_index"] for row in manifest_rows if row["p2_split"] == "valid_fit"])
    write_lines(out_dir / "valid_select_row_indices.txt", [row["row_index"] for row in manifest_rows if row["p2_split"] == "valid_select"])
    write_json(out_dir / "valid_split_summary.json", summary)
    with open(out_dir / "valid_split_summary.md", "w", encoding="utf-8") as f:
        f.write(make_markdown(summary))

    if not summary["overall_ok"]:
        failed = [check for check in summary["checks"] if not check["ok"]]
        raise SystemExit(f"P2-1 split manifest written but checks failed: {failed}")

    print(f"Wrote P2 valid split manifest: {out_dir / 'valid_split_manifest.jsonl'}")
    print(
        "P2-1 summary: "
        f"valid_fit={summary['valid_fit_count']} "
        f"valid_select={summary['valid_select_count']} "
        f"seed={summary['seed']}"
    )


if __name__ == "__main__":
    main()
