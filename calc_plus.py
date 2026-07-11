#!/usr/bin/env python3
"""Enhanced evaluation for existing MiniOneRec SID prediction JSON files.

The script is intentionally read-only for model/data artifacts. It consumes the
legacy evaluate.py output format and writes additional reports under output_dir.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from utils_sid import load_json, load_json_set, normalize_sid, parse_sid_tokens


NOT_AVAILABLE = "not_available"
RESERVED_METRICS = {
    "item_level_hr_ndcg": NOT_AVAILABLE,
    "candidate_bucket_expansion_item_level_eval": NOT_AVAILABLE,
    "score_aware_rerank_comparison": NOT_AVAILABLE,
    "latency": NOT_AVAILABLE,
    "strict_beam_diversity": NOT_AVAILABLE,
    "candidate_item_ids": NOT_AVAILABLE,
    "reranked_candidate_item_ids": NOT_AVAILABLE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced SID-level evaluation for MiniOneRec predictions.")
    parser.add_argument("--prediction-file", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, default=None, help="Evaluation CSV for the selected split.")
    parser.add_argument("--test-csv", type=Path, default=None, help="Backward-compatible alias for --eval-csv.")
    parser.add_argument("--eval-split", choices=["valid", "test"], default=None)
    parser.add_argument("--item2sid", type=Path, default=None)
    parser.add_argument("--sid2items", type=Path, default=None)
    parser.add_argument("--valid-sid-set", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50])
    parser.add_argument("--head-ratio", type=float, default=0.2)
    parser.add_argument("--tail-ratio", type=float, default=0.2)
    parser.add_argument("--history-bucket", choices=["median"], default="median")
    parser.add_argument("--short-history-max-len", type=int, default=None)
    parser.add_argument("--sid-version", default="text")
    parser.add_argument("--prefix-levels", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if args.eval_csv is None:
        args.eval_csv = args.test_csv
    elif args.test_csv is not None and args.test_csv != args.eval_csv:
        parser.error("--eval-csv and --test-csv were both provided with different paths")
    if args.eval_split is None and args.eval_csv is not None:
        args.eval_split = infer_eval_split(args.eval_csv)
    if args.eval_split is None:
        args.eval_split = "test"
    return args


def infer_eval_split(path: Path) -> str:
    path_text = path.as_posix().lower()
    if "/valid" in path_text or "valid.csv" in path_text or "validation" in path_text:
        return "valid"
    return "test"


def item_sort_key(value: Any) -> tuple[int, int | str]:
    value_str = str(value)
    try:
        return (0, int(value_str))
    except ValueError:
        return (1, value_str)


def rate(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def safe_mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def median(values: list[int | float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return (float(values[mid - 1]) + float(values[mid])) / 2.0


def parse_list(value: Any) -> tuple[list[Any], str | None]:
    if isinstance(value, list):
        return value, None
    value_str = "" if value is None else str(value).strip()
    if not value_str:
        return [], None
    try:
        parsed = ast.literal_eval(value_str)
    except (SyntaxError, ValueError) as exc:
        return [], str(exc)
    if isinstance(parsed, (list, tuple)):
        return list(parsed), None
    return [parsed], None


def clean_sid(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return normalize_sid(str(value).strip(" \n\""))


def sid_prefix(sid: str, level: int) -> str | None:
    tokens = parse_sid_tokens(sid)
    if len(tokens) < level:
        return None
    return "".join(tokens[:level])


def sid_length_distribution(sids: list[str] | set[str]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for sid in sids:
        length = len(parse_sid_tokens(sid))
        key = str(length)
        distribution[key] = distribution.get(key, 0) + 1
    return dict(sorted(distribution.items(), key=lambda item: int(item[0])))


def infer_prefix_levels(
    valid_sid_set: set[str],
    records: list[dict[str, Any]],
    explicit_levels: list[int] | None,
) -> list[int]:
    if explicit_levels:
        return sorted(set(level for level in explicit_levels if level > 0))

    lengths = [len(parse_sid_tokens(sid)) for sid in valid_sid_set]
    lengths.extend(len(parse_sid_tokens(record["target_sid"])) for record in records)
    for record in records:
        lengths.extend(len(parse_sid_tokens(sid)) for sid in record["pred_sids"])
    max_level = max(lengths) if lengths else 0
    return list(range(1, max_level + 1))


def read_csv_rows(path: Path | None) -> tuple[list[dict[str, str]], list[str]]:
    if path is None:
        return [], [NOT_AVAILABLE]
    if not path.exists():
        return [], [f"missing file: {path}"]
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f)), []


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def json_for_csv(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def normalize_item2sid(path: Path | None) -> tuple[dict[str, str], list[str]]:
    if path is None:
        return {}, [NOT_AVAILABLE]
    if not path.exists():
        return {}, [f"missing file: {path}"]
    raw = load_json(path)
    return {str(item_id): normalize_sid(str(sid)) for item_id, sid in raw.items()}, []


def normalize_sid2items(path: Path | None) -> tuple[dict[str, list[str]], list[str]]:
    if path is None:
        return {}, [NOT_AVAILABLE]
    if not path.exists():
        return {}, [f"missing file: {path}"]
    raw = load_json(path)
    out: dict[str, list[str]] = {}
    for sid, items in raw.items():
        if isinstance(items, list):
            out[normalize_sid(str(sid))] = [str(item_id) for item_id in items]
        else:
            out[normalize_sid(str(sid))] = [str(items)]
    return out, []


def load_predictions(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Prediction file must be a JSON list: {path}")

    normalized: list[dict[str, Any]] = []
    for idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            sample = {"output": "", "predict": []}
        raw_predict = sample.get("predict", [])
        if isinstance(raw_predict, list):
            pred_sids = [clean_sid(pred) for pred in raw_predict]
        elif raw_predict is None:
            pred_sids = []
        else:
            pred_sids = [clean_sid(raw_predict)]

        normalized.append({
            "row_index": idx,
            "target_sid": clean_sid(sample.get("output", "")),
            "pred_sids": pred_sids,
            "raw": sample,
        })
    return normalized


def first_hit_rank(target_sid: str, pred_sids: list[str]) -> int | None:
    for idx, pred_sid in enumerate(pred_sids):
        if pred_sid == target_sid:
            return idx
    return None


def first_prefix_hit_rank(target_sid: str, pred_sids: list[str], level: int) -> int | None:
    target_prefix = sid_prefix(target_sid, level)
    if target_prefix is None:
        return None
    for idx, pred_sid in enumerate(pred_sids):
        if sid_prefix(pred_sid, level) == target_prefix:
            return idx
    return None


def aggregate_rank_metrics(
    records: list[dict[str, Any]],
    rank_key: str,
    topk: list[int],
    ndcg: bool = True,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"num_samples": len(records)}
    for k in topk:
        hits = 0
        ndcg_sum = 0.0
        for record in records:
            rank = record.get(rank_key)
            if rank is not None and rank < k:
                hits += 1
                if ndcg:
                    ndcg_sum += 1.0 / math.log2(rank + 2)
        metrics[f"hr@{k}"] = rate(hits, len(records))
        if ndcg:
            metrics[f"ndcg@{k}"] = rate(ndcg_sum, len(records))
    return metrics


def aggregate_prefix_metrics(records: list[dict[str, Any]], level: int, topk: list[int]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"num_samples": len(records)}
    rank_key = f"prefix_{level}_hit_rank_0_based"
    for k in topk:
        hits = sum(1 for record in records if record.get(rank_key) is not None and record[rank_key] < k)
        metrics[f"prefix@{level}_hit@{k}"] = rate(hits, len(records))
    return metrics


def build_train_frequency(train_rows: list[dict[str, str]]) -> tuple[Counter[str], Counter[str], list[dict[str, str]]]:
    target_counts: Counter[str] = Counter()
    history_counts: Counter[str] = Counter()
    parse_errors: list[dict[str, str]] = []
    for row_idx, row in enumerate(train_rows, start=2):
        item_id = str(row.get("item_id", "")).strip()
        if item_id:
            target_counts[item_id] += 1

        history_ids, error = parse_list(row.get("history_item_id", "[]"))
        if error:
            if len(parse_errors) < 20:
                parse_errors.append({"row": str(row_idx), "history_item_id_error": error})
            continue
        for history_id in history_ids:
            history_id = str(history_id).strip()
            if history_id:
                history_counts[history_id] += 1
    return target_counts, history_counts, parse_errors


def assign_popularity_groups(
    item_universe: set[str],
    target_counts: Counter[str],
    history_counts: Counter[str],
    head_ratio: float,
    tail_ratio: float,
) -> dict[str, str]:
    def total_count(item_id: str) -> int:
        return int(target_counts.get(item_id, 0) + history_counts.get(item_id, 0))

    sorted_items = sorted(item_universe, key=lambda item_id: (-total_count(item_id), item_sort_key(item_id)))
    n_items = len(sorted_items)
    head_n = int(n_items * head_ratio)
    tail_n = int(n_items * tail_ratio)
    if head_ratio > 0 and n_items:
        head_n = max(1, head_n)
    if tail_ratio > 0 and n_items:
        tail_n = max(1, tail_n)
    if head_n + tail_n > n_items:
        tail_n = max(0, n_items - head_n)

    head_items = set(sorted_items[:head_n])
    tail_items = set(sorted_items[n_items - tail_n:]) if tail_n else set()
    groups: dict[str, str] = {}
    for item_id in item_universe:
        if item_id in head_items:
            groups[item_id] = "head"
        elif item_id in tail_items:
            groups[item_id] = "tail"
        else:
            groups[item_id] = "mid"
    return groups


def enrich_with_eval_alignment(
    prediction_records: list[dict[str, Any]],
    eval_rows: list[dict[str, str]],
    train_rows: list[dict[str, str]],
    item2sid: dict[str, str],
    topk: list[int],
    head_ratio: float,
    tail_ratio: float,
    short_history_max_len: int | None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not eval_rows:
        return {"available": False, "reason": "eval_csv not provided or empty"}, []
    if len(eval_rows) != len(prediction_records):
        return {
            "available": False,
            "reason": f"prediction/eval_csv length mismatch: {len(prediction_records)} != {len(eval_rows)}",
        }, []
    if not train_rows:
        return {"available": False, "reason": "train_csv not provided or empty"}, []

    target_counts, history_counts, train_parse_errors = build_train_frequency(train_rows)
    item_universe = set(item2sid) if item2sid else set(target_counts) | set(history_counts)
    for row in eval_rows:
        item_id = str(row.get("item_id", "")).strip()
        if item_id:
            item_universe.add(item_id)
    popularity_groups = assign_popularity_groups(item_universe, target_counts, history_counts, head_ratio, tail_ratio)

    history_lengths: list[int] = []
    eval_parse_errors: list[dict[str, str]] = []
    aligned_meta: list[dict[str, Any]] = []
    target_sid_mismatch_count = 0
    for idx, (record, row) in enumerate(zip(prediction_records, eval_rows)):
        item_id = str(row.get("item_id", "")).strip()
        csv_target_sid = clean_sid(row.get("item_sid", ""))
        if csv_target_sid and csv_target_sid != record["target_sid"]:
            target_sid_mismatch_count += 1

        history_ids, error = parse_list(row.get("history_item_id", "[]"))
        if error:
            history_ids = []
            if len(eval_parse_errors) < 20:
                eval_parse_errors.append({"row_index": str(idx), "history_item_id_error": error})
        history_len = len(history_ids)
        history_lengths.append(history_len)
        total_train_count = int(target_counts.get(item_id, 0) + history_counts.get(item_id, 0))
        aligned_meta.append({
            "target_item_id": item_id,
            "csv_target_sid": csv_target_sid,
            "history_len": history_len,
            "train_target_count": int(target_counts.get(item_id, 0)),
            "train_history_count": int(history_counts.get(item_id, 0)),
            "train_total_interaction_count": total_train_count,
            "popularity_group": popularity_groups.get(item_id, "tail" if total_train_count == 0 else "mid"),
            "cold_warm_group": "cold" if total_train_count == 0 else "warm",
        })

    if short_history_max_len is None:
        threshold = median(history_lengths)
        threshold_source = "median"
    else:
        threshold = float(short_history_max_len)
        threshold_source = "short_history_max_len"

    for record, meta in zip(prediction_records, aligned_meta):
        meta["history_length_group"] = "short" if meta["history_len"] <= threshold else "long"
        record.update(meta)

    group_metrics = {
        "available": True,
        "target_sid_mismatch_count": target_sid_mismatch_count,
        "history_length_threshold": threshold,
        "history_length_threshold_source": threshold_source,
        "train_parse_error_samples": train_parse_errors,
        "eval_parse_error_samples": eval_parse_errors,
        "popularity": compute_group_metrics(prediction_records, "popularity_group", ["head", "mid", "tail"], topk),
        "cold_warm": compute_group_metrics(prediction_records, "cold_warm_group", ["cold", "warm"], topk),
        "history_length": compute_group_metrics(prediction_records, "history_length_group", ["short", "long"], topk),
    }
    return group_metrics, aligned_meta


def compute_group_metrics(
    records: list[dict[str, Any]],
    group_key: str,
    group_names: list[str],
    topk: list[int],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group_name in group_names:
        group_records = [record for record in records if record.get(group_key) == group_name]
        metrics = aggregate_rank_metrics(group_records, "sid_hit_rank_0_based", topk)
        metrics["num_samples"] = len(group_records)
        out[group_name] = metrics
    return out


def evaluate_predictions(
    prediction_records: list[dict[str, Any]],
    valid_sid_set: set[str],
    sid2items: dict[str, list[str]],
    topk: list[int],
    prefix_levels: list[int],
) -> dict[str, Any]:
    total_predictions = 0
    invalid_predictions = 0
    duplicate_predictions = 0
    samples_with_duplicates = 0

    for record in prediction_records:
        pred_sids = record["pred_sids"]
        valid_flags = [pred_sid in valid_sid_set for pred_sid in pred_sids]
        total_predictions += len(pred_sids)
        invalid_predictions += sum(1 for flag in valid_flags if not flag)
        duplicate_count = len(pred_sids) - len(set(pred_sids))
        duplicate_predictions += duplicate_count
        samples_with_duplicates += int(duplicate_count > 0)

        record["pred_valid_flags"] = valid_flags
        record["duplicate_count"] = duplicate_count
        record["sid_hit_rank_0_based"] = first_hit_rank(record["target_sid"], pred_sids)
        record["sid_hit_rank_1_based"] = (
            record["sid_hit_rank_0_based"] + 1 if record["sid_hit_rank_0_based"] is not None else None
        )
        for level in prefix_levels:
            rank = first_prefix_hit_rank(record["target_sid"], pred_sids, level)
            record[f"prefix_{level}_hit_rank_0_based"] = rank
            record[f"prefix_{level}_hit_rank_1_based"] = rank + 1 if rank is not None else None
        record["target_sid_bucket_size"] = len(sid2items.get(record["target_sid"], [])) if sid2items else NOT_AVAILABLE

    n_beams = [len(record["pred_sids"]) for record in prediction_records]
    sid_level = aggregate_rank_metrics(prediction_records, "sid_hit_rank_0_based", topk)
    prefix_level = {
        f"prefix@{level}": aggregate_prefix_metrics(prediction_records, level, topk)
        for level in prefix_levels
    }
    validity = {
        "invalid_sid_count": invalid_predictions,
        "total_predicted_sids": total_predictions,
        "invalid_sid_rate": rate(invalid_predictions, total_predictions),
    }
    duplication = {
        "duplicate_prediction_count": duplicate_predictions,
        "total_predicted_sids": total_predictions,
        "duplicate_generation_rate": rate(duplicate_predictions, total_predictions),
        "samples_with_duplicates": samples_with_duplicates,
        "samples_with_duplicates_rate": rate(samples_with_duplicates, len(prediction_records)),
    }
    prediction_shape = {
        "num_samples": len(prediction_records),
        "min_predictions_per_sample": min(n_beams) if n_beams else 0,
        "max_predictions_per_sample": max(n_beams) if n_beams else 0,
        "mean_predictions_per_sample": safe_mean(n_beams),
    }
    return {
        "prediction_shape": prediction_shape,
        "sid_level": sid_level,
        "prefix_level": prefix_level,
        "validity": validity,
        "duplication": duplication,
    }


def build_per_sample_rows(records: list[dict[str, Any]], prefix_levels: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {
            "row_index": record["row_index"],
            "target_sid": record["target_sid"],
            "target_item_id": record.get("target_item_id", ""),
            "history_len": record.get("history_len", ""),
            "popularity_group": record.get("popularity_group", NOT_AVAILABLE),
            "cold_warm_group": record.get("cold_warm_group", NOT_AVAILABLE),
            "history_length_group": record.get("history_length_group", NOT_AVAILABLE),
            "pred_sids": json_for_csv(record["pred_sids"]),
            "pred_valid_flags": json_for_csv(record.get("pred_valid_flags", [])),
            "duplicate_count": record.get("duplicate_count", 0),
            "hit_rank_0_based": "" if record.get("sid_hit_rank_0_based") is None else record["sid_hit_rank_0_based"],
            "hit_rank_1_based": "" if record.get("sid_hit_rank_1_based") is None else record["sid_hit_rank_1_based"],
            "target_sid_bucket_size": record.get("target_sid_bucket_size", NOT_AVAILABLE),
        }
        for level in prefix_levels:
            key = f"prefix_{level}_hit_rank_0_based"
            row[f"prefix{level}_hit_rank_0_based"] = "" if record.get(key) is None else record[key]
        rows.append(row)
    return rows


def flatten_metrics(data: Any, prefix: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_metrics(value, child_prefix))
    elif isinstance(data, list):
        rows.append({"metric": prefix, "value": json_for_csv(data)})
    else:
        value = "" if data is None else str(data)
        rows.append({"metric": prefix, "value": value})
    return rows


def main() -> None:
    args = parse_args()
    topk = sorted(set(k for k in args.topk if k > 0))
    valid_sid_set = {normalize_sid(sid) for sid in load_json_set(args.valid_sid_set)}
    item2sid, item2sid_errors = normalize_item2sid(args.item2sid)
    sid2items, sid2items_errors = normalize_sid2items(args.sid2items)
    prediction_records = load_predictions(args.prediction_file)
    prefix_levels = infer_prefix_levels(valid_sid_set, prediction_records, args.prefix_levels)

    eval_rows, eval_errors = read_csv_rows(args.eval_csv)
    train_rows, train_errors = read_csv_rows(args.train_csv)
    core_metrics = evaluate_predictions(prediction_records, valid_sid_set, sid2items, topk, prefix_levels)
    group_metrics, _ = enrich_with_eval_alignment(
        prediction_records,
        eval_rows,
        train_rows,
        item2sid,
        topk,
        args.head_ratio,
        args.tail_ratio,
        args.short_history_max_len,
    )

    report = {
        "sid_version": args.sid_version,
        "eval_split": args.eval_split,
        "inputs": {
            "prediction_file": args.prediction_file.as_posix(),
            "eval_csv": args.eval_csv.as_posix() if args.eval_csv else NOT_AVAILABLE,
            "eval_split": args.eval_split,
            "item2sid": args.item2sid.as_posix() if args.item2sid else NOT_AVAILABLE,
            "sid2items": args.sid2items.as_posix() if args.sid2items else NOT_AVAILABLE,
            "valid_sid_set": args.valid_sid_set.as_posix(),
            "train_csv": args.train_csv.as_posix() if args.train_csv else NOT_AVAILABLE,
        },
        "topk": topk,
        "prefix_levels": prefix_levels,
        "errors": {
            "eval_csv": eval_errors,
            "train_csv": train_errors,
            "item2sid": item2sid_errors,
            "sid2items": sid2items_errors,
        },
        "prediction_shape": core_metrics["prediction_shape"],
        "sid_level_hr_ndcg": core_metrics["sid_level"],
        "prefix_hit": core_metrics["prefix_level"],
        "validity": core_metrics["validity"],
        "duplicate_generation": core_metrics["duplication"],
        "sid_length_distribution": {
            "valid_sid_set": sid_length_distribution(valid_sid_set),
            "targets": sid_length_distribution([record["target_sid"] for record in prediction_records]),
            "predictions": sid_length_distribution([
                sid for record in prediction_records for sid in record["pred_sids"]
            ]),
        },
        "dedup_sid_note": (
            "For dedup SID versions, levels 1-3 are semantic/KMeans prefixes; "
            "level 4 is a disambiguation suffix and should not be interpreted as a semantic level."
        ),
        "group_metrics": group_metrics,
        "reserved_metrics": RESERVED_METRICS,
    }

    output_dir = args.output_dir
    report_json = output_dir / f"eval_report_{args.sid_version}.json"
    report_csv = output_dir / f"eval_report_{args.sid_version}.csv"
    per_sample_csv = output_dir / "per_sample_eval.csv"
    write_json(report_json, report)
    write_csv(report_csv, ["metric", "value"], flatten_metrics(report))
    per_sample_fieldnames = [
            "row_index",
            "target_sid",
            "target_item_id",
            "history_len",
            "popularity_group",
            "cold_warm_group",
            "history_length_group",
            "pred_sids",
            "pred_valid_flags",
            "duplicate_count",
            "hit_rank_0_based",
            "hit_rank_1_based",
        ]
    per_sample_fieldnames.extend([f"prefix{level}_hit_rank_0_based" for level in prefix_levels])
    per_sample_fieldnames.append("target_sid_bucket_size")
    write_csv(per_sample_csv, per_sample_fieldnames, build_per_sample_rows(prediction_records, prefix_levels))

    print(f"Wrote report JSON: {report_json}")
    print(f"Wrote report CSV: {report_csv}")
    print(f"Wrote per-sample CSV: {per_sample_csv}")
    print(
        "SID-level summary: "
        f"num_samples={core_metrics['prediction_shape']['num_samples']} "
        f"invalid_sid_rate={core_metrics['validity']['invalid_sid_rate']:.6f} "
        f"duplicate_generation_rate={core_metrics['duplication']['duplicate_generation_rate']:.6f}"
    )
    for k in topk:
        key = f"hr@{k}"
        if key in core_metrics["sid_level"]:
            print(
                f"  SID HR@{k}={core_metrics['sid_level'][key]:.6f} "
                f"NDCG@{k}={core_metrics['sid_level'][f'ndcg@{k}']:.6f}"
            )


if __name__ == "__main__":
    main()
