#!/usr/bin/env python3
"""Compare SFT and RL per-sample evaluation outputs.

The expected inputs are per_sample_eval.csv files produced by calc_plus.py, but
the column detection is intentionally permissive so older result files can also
be compared.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ALIGN_CANDIDATES = [
    "sample_id",
    "idx",
    "row_id",
    "row_index",
    "index",
    "id",
]

EXACT_RANK_0_BASED = [
    "hit_rank_0_based",
    "sid_hit_rank_0_based",
    "exact_hit_rank_0_based",
    "rank_0_based",
]

EXACT_RANK_1_BASED = [
    "hit_rank_1_based",
    "sid_hit_rank_1_based",
    "exact_hit_rank_1_based",
    "rank_1_based",
]

GROUP_CANDIDATES = {
    "popularity": [
        "popularity_group",
        "popularity_bucket",
        "popularity",
        "pop_group",
    ],
    "cold_warm": [
        "cold_warm_group",
        "cold_warm_bucket",
        "cold_warm",
    ],
    "history_length": [
        "history_length_group",
        "history_length_bucket",
        "history_len_group",
        "history_bucket",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SFT and RL per-sample eval CSV files.")
    parser.add_argument("--sft-csv", type=Path, required=True)
    parser.add_argument("--rl-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument(
        "--focus-k",
        type=int,
        default=20,
        help="K used for focused sft_only/rl_only diagnostics.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=200,
        help="Maximum focused transition rows to write to focus_hitK_examples.csv.",
    )
    return parser.parse_args()


def read_eval_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def choose_alignment_key(sft: pd.DataFrame, rl: pd.DataFrame) -> tuple[str | None, str, list[str]]:
    warnings: list[str] = []
    for col in ALIGN_CANDIDATES:
        if col not in sft.columns or col not in rl.columns:
            continue
        sft_key = sft[col].astype(str)
        rl_key = rl[col].astype(str)
        if not sft_key.is_unique or not rl_key.is_unique:
            warnings.append(f"Alignment candidate {col!r} skipped because values are not unique.")
            continue
        overlap = len(set(sft_key) & set(rl_key))
        min_rows = min(len(sft), len(rl))
        if min_rows == 0 or overlap == 0:
            warnings.append(f"Alignment candidate {col!r} skipped because overlap is {overlap}.")
            continue
        if overlap < min_rows:
            warnings.append(
                f"Alignment candidate {col!r} has partial overlap: {overlap}/{min_rows}; "
                "using the overlapping rows only."
            )
        return col, f"column:{col}", warnings

    warnings.append("No unique sample id column found; falling back to row-order alignment.")
    return None, "row_order", warnings


def add_alignment_key(df: pd.DataFrame, key: str | None) -> pd.DataFrame:
    out = df.copy()
    out["_source_row_number"] = range(len(out))
    if key is None:
        out["_align_key"] = out["_source_row_number"].astype(str)
    else:
        out["_align_key"] = out[key].astype(str)
    return out


def prefix_columns(df: pd.DataFrame, side: str) -> pd.DataFrame:
    rename = {
        col: col if col == "_align_key" else f"{side}__{col}"
        for col in df.columns
    }
    return df.rename(columns=rename)


def align_frames(sft: pd.DataFrame, rl: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    key, method, warnings = choose_alignment_key(sft, rl)
    sft_aligned = prefix_columns(add_alignment_key(sft, key), "sft")
    rl_aligned = prefix_columns(add_alignment_key(rl, key), "rl")
    merged = sft_aligned.merge(rl_aligned, on="_align_key", how="inner", validate="one_to_one")
    info = {
        "method": method,
        "key": key if key is not None else "row_number",
        "sft_rows": int(len(sft)),
        "rl_rows": int(len(rl)),
        "aligned_rows": int(len(merged)),
        "sft_unaligned_rows": int(len(sft) - len(merged)),
        "rl_unaligned_rows": int(len(rl) - len(merged)),
    }
    if len(merged) != len(sft) or len(merged) != len(rl):
        warnings.append(
            f"Aligned {len(merged)} rows from SFT={len(sft)} and RL={len(rl)}; "
            "summary metrics use aligned rows only."
        )
    return merged, info, warnings


def prefixed(side: str, col: str) -> str:
    return f"{side}__{col}"


def first_existing(columns: list[str], candidates: list[str], side: str) -> str | None:
    for col in candidates:
        name = prefixed(side, col)
        if name in columns:
            return name
    return None


def rank0_from_columns(df: pd.DataFrame, zero_col: str | None, one_col: str | None) -> pd.Series:
    if zero_col:
        return pd.to_numeric(df[zero_col], errors="coerce")
    if one_col:
        return pd.to_numeric(df[one_col], errors="coerce") - 1
    return pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")


def exact_rank(df: pd.DataFrame, side: str) -> tuple[pd.Series, dict[str, str | None]]:
    cols = list(df.columns)
    zero_col = first_existing(cols, EXACT_RANK_0_BASED, side)
    one_col = first_existing(cols, EXACT_RANK_1_BASED, side)
    return rank0_from_columns(df, zero_col, one_col), {
        "rank_0_based_column": zero_col,
        "rank_1_based_column": one_col,
    }


def prefix_rank_candidates(level: int) -> tuple[list[str], list[str]]:
    zero = [
        f"prefix{level}_hit_rank_0_based",
        f"prefix_{level}_hit_rank_0_based",
        f"prefix@{level}_hit_rank_0_based",
        f"prefix{level}_rank_0_based",
        f"prefix_{level}_rank_0_based",
    ]
    one = [
        f"prefix{level}_hit_rank_1_based",
        f"prefix_{level}_hit_rank_1_based",
        f"prefix@{level}_hit_rank_1_based",
        f"prefix{level}_rank_1_based",
        f"prefix_{level}_rank_1_based",
    ]
    return zero, one


def prefix_rank(df: pd.DataFrame, side: str, level: int) -> tuple[pd.Series, dict[str, str | None]]:
    zero_candidates, one_candidates = prefix_rank_candidates(level)
    cols = list(df.columns)
    zero_col = first_existing(cols, zero_candidates, side)
    one_col = first_existing(cols, one_candidates, side)
    return rank0_from_columns(df, zero_col, one_col), {
        "rank_0_based_column": zero_col,
        "rank_1_based_column": one_col,
    }


def hit_from_rank(rank: pd.Series, k: int) -> pd.Series:
    numeric = pd.to_numeric(rank, errors="coerce")
    return numeric.notna() & (numeric >= 0) & (numeric <= k - 1)


def clipped_rank(rank: pd.Series, k: int) -> pd.Series:
    numeric = pd.to_numeric(rank, errors="coerce")
    return numeric.where(numeric.notna() & (numeric >= 0) & (numeric <= k - 1), math.inf)


def rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else float(count) / float(total)


def hit_transition_stats(sft_hit: pd.Series, rl_hit: pd.Series) -> dict[str, Any]:
    total = int(len(sft_hit))
    both_hit = int((sft_hit & rl_hit).sum())
    sft_only = int((sft_hit & ~rl_hit).sum())
    rl_only = int((~sft_hit & rl_hit).sum())
    both_miss = int((~sft_hit & ~rl_hit).sum())
    return {
        "samples": total,
        "sft_hit_count": int(sft_hit.sum()),
        "rl_hit_count": int(rl_hit.sum()),
        "sft_hit_rate": rate(int(sft_hit.sum()), total),
        "rl_hit_rate": rate(int(rl_hit.sum()), total),
        "hit_rate_delta_rl_minus_sft": rate(int(rl_hit.sum()), total) - rate(int(sft_hit.sum()), total),
        "both_hit": both_hit,
        "sft_only_hit": sft_only,
        "rl_only_hit": rl_only,
        "both_miss": both_miss,
    }


def rank_change_stats(sft_rank: pd.Series, rl_rank: pd.Series, k: int) -> dict[str, int]:
    sft_clipped = clipped_rank(sft_rank, k)
    rl_clipped = clipped_rank(rl_rank, k)
    return {
        "rl_rank_better": int((rl_clipped < sft_clipped).sum()),
        "rl_rank_worse": int((rl_clipped > sft_clipped).sum()),
        "rl_rank_same": int((rl_clipped == sft_clipped).sum()),
    }


def exact_summary_for_k(sft_rank: pd.Series, rl_rank: pd.Series, k: int) -> dict[str, Any]:
    sft_hit = hit_from_rank(sft_rank, k)
    rl_hit = hit_from_rank(rl_rank, k)
    out = hit_transition_stats(sft_hit, rl_hit)
    out.update(rank_change_stats(sft_rank, rl_rank, k))
    return out


def detect_group_fields(df: pd.DataFrame) -> tuple[dict[str, str], list[str]]:
    warnings: list[str] = []
    detected: dict[str, str] = {}
    for group_name, candidates in GROUP_CANDIDATES.items():
        for col in candidates:
            sft_col = prefixed("sft", col)
            rl_col = prefixed("rl", col)
            if sft_col in df.columns or rl_col in df.columns:
                chosen = sft_col if sft_col in df.columns else rl_col
                detected[group_name] = chosen
                if sft_col in df.columns and rl_col in df.columns:
                    mismatch = int((df[sft_col].astype(str) != df[rl_col].astype(str)).sum())
                    if mismatch:
                        warnings.append(
                            f"Group field {col!r} differs between SFT and RL on {mismatch} rows; "
                            "using SFT values for grouping."
                        )
                break
    return detected, warnings


def add_compare_columns(
    df: pd.DataFrame,
    topk: list[int],
    sft_rank: pd.Series,
    rl_rank: pd.Series,
    prefix_ranks: dict[int, tuple[pd.Series, pd.Series]],
) -> pd.DataFrame:
    out = df.copy()
    out["sft_hit_rank_0_based"] = sft_rank
    out["rl_hit_rank_0_based"] = rl_rank
    out["rank_delta_sft_minus_rl"] = sft_rank.fillna(math.inf) - rl_rank.fillna(math.inf)

    for k in topk:
        sft_hit = hit_from_rank(sft_rank, k)
        rl_hit = hit_from_rank(rl_rank, k)
        out[f"sft_hit@{k}"] = sft_hit
        out[f"rl_hit@{k}"] = rl_hit
        out[f"hit_change@{k}"] = [
            classify_hit_change(bool(s), bool(r))
            for s, r in zip(sft_hit.tolist(), rl_hit.tolist())
        ]
        sft_clipped = clipped_rank(sft_rank, k)
        rl_clipped = clipped_rank(rl_rank, k)
        out[f"rl_rank_better@{k}"] = rl_clipped < sft_clipped
        out[f"rl_rank_worse@{k}"] = rl_clipped > sft_clipped

    for level, (sft_prefix_rank, rl_prefix_rank) in prefix_ranks.items():
        out[f"sft_prefix{level}_rank_0_based"] = sft_prefix_rank
        out[f"rl_prefix{level}_rank_0_based"] = rl_prefix_rank
        for k in topk:
            sft_prefix_hit = hit_from_rank(sft_prefix_rank, k)
            rl_prefix_hit = hit_from_rank(rl_prefix_rank, k)
            out[f"sft_prefix{level}_hit@{k}"] = sft_prefix_hit
            out[f"rl_prefix{level}_hit@{k}"] = rl_prefix_hit
            out[f"prefix{level}_hit_change@{k}"] = [
                classify_hit_change(bool(s), bool(r))
                for s, r in zip(sft_prefix_hit.tolist(), rl_prefix_hit.tolist())
            ]
            prefix_better = clipped_rank(rl_prefix_rank, k) < clipped_rank(sft_prefix_rank, k)
            exact_worse = clipped_rank(rl_rank, k) > clipped_rank(sft_rank, k)
            out[f"rl_prefix{level}_better_exact_worse@{k}"] = prefix_better & exact_worse

    return out


def classify_hit_change(sft_hit: bool, rl_hit: bool) -> str:
    if sft_hit and rl_hit:
        return "both_hit"
    if sft_hit and not rl_hit:
        return "sft_only_hit"
    if not sft_hit and rl_hit:
        return "rl_only_hit"
    return "both_miss"


def value_counts_rows(
    df: pd.DataFrame,
    transition_name: str,
    field_name: str,
    col: str,
) -> list[dict[str, Any]]:
    total = int(len(df))
    if col not in df.columns:
        return []
    counts = df[col].astype(str).replace({"": "missing"}).value_counts(dropna=False)
    return [
        {
            "transition": transition_name,
            "analysis": "value_counts",
            "field": field_name,
            "value": str(value),
            "count": int(count),
            "rate_within_transition": rate(int(count), total),
        }
        for value, count in counts.items()
    ]


def rank_bucket(rank: Any) -> str:
    numeric = pd.to_numeric(pd.Series([rank]), errors="coerce").iloc[0]
    if pd.isna(numeric) or numeric < 0:
        return "missing_or_not_hit"
    numeric_int = int(numeric)
    if numeric_int <= 4:
        return "rank_1_5"
    if numeric_int <= 9:
        return "rank_6_10"
    if numeric_int <= 14:
        return "rank_11_15"
    if numeric_int <= 19:
        return "rank_16_20"
    return "rank_over_20"


def add_focus_transition_columns(df: pd.DataFrame, focus_k: int) -> pd.DataFrame:
    out = df.copy()
    sft_col = f"sft_hit@{focus_k}"
    rl_col = f"rl_hit@{focus_k}"
    if sft_col not in out.columns or rl_col not in out.columns:
        raise ValueError(
            f"Focused diagnostics require {sft_col!r} and {rl_col!r}; "
            f"available columns include: {list(out.columns)[:30]}..."
        )
    out[f"transition@{focus_k}"] = [
        classify_hit_change(bool(s), bool(r))
        for s, r in zip(out[sft_col].tolist(), out[rl_col].tolist())
    ]
    out[f"sft_rank_bucket@{focus_k}"] = out["sft_hit_rank_0_based"].map(rank_bucket)
    out[f"rl_rank_bucket@{focus_k}"] = out["rl_hit_rank_0_based"].map(rank_bucket)
    return out


def transition_subsets(df: pd.DataFrame, focus_k: int) -> dict[str, pd.DataFrame]:
    transition_col = f"transition@{focus_k}"
    return {
        name: df[df[transition_col] == name].copy()
        for name in ["both_hit", "sft_only_hit", "rl_only_hit", "both_miss"]
    }


def summarize_focus_transitions(
    df: pd.DataFrame,
    group_fields: dict[str, str],
    prefix_levels: list[int],
    focus_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    focused = add_focus_transition_columns(df, focus_k)
    subsets = transition_subsets(focused, focus_k)
    total = int(len(focused))

    counts = {name: int(len(subset)) for name, subset in subsets.items()}
    summary: dict[str, Any] = {
        "focus_k": focus_k,
        "samples": total,
        "transition_counts": counts,
        "transition_rates": {name: rate(count, total) for name, count in counts.items()},
        "diagnostics": {},
    }
    rows: list[dict[str, Any]] = []

    for transition_name, subset in subsets.items():
        rows.append({
            "transition": transition_name,
            "analysis": "transition_count",
            "field": "all",
            "value": transition_name,
            "count": int(len(subset)),
            "rate_within_transition": 1.0 if len(subset) else 0.0,
        })
        for group_name, col in group_fields.items():
            rows.extend(value_counts_rows(subset, transition_name, group_name, col))

        rows.extend(
            value_counts_rows(
                subset,
                transition_name,
                f"sft_rank_bucket@{focus_k}",
                f"sft_rank_bucket@{focus_k}",
            )
        )
        rows.extend(
            value_counts_rows(
                subset,
                transition_name,
                f"rl_rank_bucket@{focus_k}",
                f"rl_rank_bucket@{focus_k}",
            )
        )

    sft_only = subsets["sft_only_hit"]
    rl_only = subsets["rl_only_hit"]
    sft_only_diag: dict[str, Any] = {
        "samples": int(len(sft_only)),
        "rank_loss_distribution": {},
        "rl_prefix_preserved_while_exact_lost": {},
        "rl_prefix_better_exact_worse": {},
    }
    rl_only_diag: dict[str, Any] = {
        "samples": int(len(rl_only)),
        "new_hit_rank_distribution": {},
        "sft_prefix_present_before_rl_exact_gain": {},
    }

    if len(sft_only):
        sft_rank_counts = sft_only[f"sft_rank_bucket@{focus_k}"].value_counts(dropna=False)
        sft_only_diag["rank_loss_distribution"] = {
            str(bucket): int(count)
            for bucket, count in sft_rank_counts.items()
        }
    if len(rl_only):
        rl_rank_counts = rl_only[f"rl_rank_bucket@{focus_k}"].value_counts(dropna=False)
        rl_only_diag["new_hit_rank_distribution"] = {
            str(bucket): int(count)
            for bucket, count in rl_rank_counts.items()
        }

    for level in prefix_levels:
        rl_prefix_col = f"rl_prefix{level}_hit@{focus_k}"
        sft_prefix_col = f"sft_prefix{level}_hit@{focus_k}"
        better_exact_worse_col = f"rl_prefix{level}_better_exact_worse@{focus_k}"

        if rl_prefix_col in sft_only.columns:
            count = int(sft_only[rl_prefix_col].fillna(False).astype(bool).sum())
            sft_only_diag["rl_prefix_preserved_while_exact_lost"][f"prefix{level}"] = {
                "count": count,
                "rate": rate(count, len(sft_only)),
            }
            rows.append({
                "transition": "sft_only_hit",
                "analysis": "rl_prefix_preserved_while_exact_lost",
                "field": f"prefix{level}",
                "value": f"rl_prefix{level}_hit@{focus_k}",
                "count": count,
                "rate_within_transition": rate(count, len(sft_only)),
            })

        if better_exact_worse_col in sft_only.columns:
            count = int(sft_only[better_exact_worse_col].fillna(False).astype(bool).sum())
            sft_only_diag["rl_prefix_better_exact_worse"][f"prefix{level}"] = {
                "count": count,
                "rate": rate(count, len(sft_only)),
            }
            rows.append({
                "transition": "sft_only_hit",
                "analysis": "rl_prefix_better_exact_worse",
                "field": f"prefix{level}",
                "value": better_exact_worse_col,
                "count": count,
                "rate_within_transition": rate(count, len(sft_only)),
            })

        if sft_prefix_col in rl_only.columns:
            count = int(rl_only[sft_prefix_col].fillna(False).astype(bool).sum())
            rl_only_diag["sft_prefix_present_before_rl_exact_gain"][f"prefix{level}"] = {
                "count": count,
                "rate": rate(count, len(rl_only)),
            }
            rows.append({
                "transition": "rl_only_hit",
                "analysis": "sft_prefix_present_before_rl_exact_gain",
                "field": f"prefix{level}",
                "value": f"sft_prefix{level}_hit@{focus_k}",
                "count": count,
                "rate_within_transition": rate(count, len(rl_only)),
            })

    summary["diagnostics"]["sft_only_hit"] = sft_only_diag
    summary["diagnostics"]["rl_only_hit"] = rl_only_diag

    example_transitions = ["sft_only_hit", "rl_only_hit"]
    examples = focused[focused[f"transition@{focus_k}"].isin(example_transitions)].copy()
    sort_cols = [f"transition@{focus_k}", "sft_hit_rank_0_based", "rl_hit_rank_0_based"]
    sort_cols = [col for col in sort_cols if col in examples.columns]
    if sort_cols:
        examples = examples.sort_values(sort_cols, na_position="last")
    return summary, rows, examples, focused


def summarize_prefix(
    topk: list[int],
    sft_rank: pd.Series,
    rl_rank: pd.Series,
    exact_sft_rank: pd.Series,
    exact_rl_rank: pd.Series,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in topk:
        stats = exact_summary_for_k(sft_rank, rl_rank, k)
        prefix_better = clipped_rank(rl_rank, k) < clipped_rank(sft_rank, k)
        exact_worse = clipped_rank(exact_rl_rank, k) > clipped_rank(exact_sft_rank, k)
        stats["rl_prefix_better_exact_worse"] = int((prefix_better & exact_worse).sum())
        out[f"hit@{k}"] = stats
    return out


def summarize_groups(
    df: pd.DataFrame,
    group_fields: dict[str, str],
    topk: list[int],
    sft_rank_col: str,
    rl_rank_col: str,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for group_name, col in group_fields.items():
        group_summary: dict[str, Any] = {}
        values = sorted([v for v in df[col].astype(str).unique().tolist() if v != "" and v.lower() != "nan"])
        for value in values:
            subset = df[df[col].astype(str) == value]
            sft_rank = pd.to_numeric(subset[sft_rank_col], errors="coerce")
            rl_rank = pd.to_numeric(subset[rl_rank_col], errors="coerce")
            group_summary[value] = {
                f"hit@{k}": exact_summary_for_k(sft_rank, rl_rank, k)
                for k in topk
            }
            group_summary[value]["samples"] = int(len(subset))
        summaries[group_name] = {
            "source_column": col,
            "groups": group_summary,
        }
    return summaries


def flatten_summary(data: Any, prefix: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_summary(value, child))
    elif isinstance(data, list):
        rows.append({"metric": prefix, "value": json.dumps(data, ensure_ascii=False)})
    else:
        rows.append({"metric": prefix, "value": "" if data is None else str(data)})
    return rows


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def compact_per_sample(df: pd.DataFrame, group_fields: dict[str, str], topk: list[int], prefix_levels: list[int]) -> pd.DataFrame:
    preferred = [
        "_align_key",
        "sft___source_row_number",
        "rl___source_row_number",
        "sft__row_index",
        "rl__row_index",
        "sft__target_item_id",
        "rl__target_item_id",
        "sft__target_sid",
        "rl__target_sid",
    ]
    preferred.extend(group_fields.values())
    preferred.extend([
        "sft_hit_rank_0_based",
        "rl_hit_rank_0_based",
        "rank_delta_sft_minus_rl",
    ])
    preferred.extend([
        col
        for col in sorted(df.columns)
        if col.startswith("transition@") or "rank_bucket@" in col
    ])
    for k in topk:
        preferred.extend([
            f"sft_hit@{k}",
            f"rl_hit@{k}",
            f"hit_change@{k}",
            f"rl_rank_better@{k}",
            f"rl_rank_worse@{k}",
        ])
    for level in prefix_levels:
        preferred.extend([
            f"sft_prefix{level}_rank_0_based",
            f"rl_prefix{level}_rank_0_based",
        ])
        for k in topk:
            preferred.extend([
                f"sft_prefix{level}_hit@{k}",
                f"rl_prefix{level}_hit@{k}",
                f"prefix{level}_hit_change@{k}",
                f"rl_prefix{level}_better_exact_worse@{k}",
            ])
    cols = [col for col in preferred if col in df.columns]
    return df[cols]


def main() -> None:
    args = parse_args()
    topk = sorted({k for k in args.topk if k > 0})
    if args.focus_k <= 0:
        raise ValueError("--focus-k must be positive")
    if args.focus_k not in topk:
        topk = sorted([*topk, args.focus_k])
    if not topk:
        raise ValueError("--topk must contain at least one positive integer")

    sft = read_eval_csv(args.sft_csv)
    rl = read_eval_csv(args.rl_csv)
    aligned, alignment_info, warnings = align_frames(sft, rl)

    sft_rank, sft_rank_info = exact_rank(aligned, "sft")
    rl_rank, rl_rank_info = exact_rank(aligned, "rl")
    aligned["sft_hit_rank_0_based"] = sft_rank
    aligned["rl_hit_rank_0_based"] = rl_rank

    if sft_rank_info["rank_0_based_column"] is None and sft_rank_info["rank_1_based_column"] is None:
        warnings.append("No SFT exact hit-rank column found; exact hit metrics will be empty.")
    if rl_rank_info["rank_0_based_column"] is None and rl_rank_info["rank_1_based_column"] is None:
        warnings.append("No RL exact hit-rank column found; exact hit metrics will be empty.")

    prefix_ranks: dict[int, tuple[pd.Series, pd.Series]] = {}
    prefix_columns_used: dict[str, Any] = {}
    for level in [1, 2, 3]:
        sft_prefix_rank, sft_prefix_info = prefix_rank(aligned, "sft", level)
        rl_prefix_rank, rl_prefix_info = prefix_rank(aligned, "rl", level)
        sft_found = sft_prefix_info["rank_0_based_column"] or sft_prefix_info["rank_1_based_column"]
        rl_found = rl_prefix_info["rank_0_based_column"] or rl_prefix_info["rank_1_based_column"]
        if sft_found and rl_found:
            prefix_ranks[level] = (sft_prefix_rank, rl_prefix_rank)
            prefix_columns_used[f"prefix{level}"] = {
                "sft": sft_prefix_info,
                "rl": rl_prefix_info,
            }

    group_fields, group_warnings = detect_group_fields(aligned)
    warnings.extend(group_warnings)

    compared = add_compare_columns(aligned, topk, sft_rank, rl_rank, prefix_ranks)

    exact_by_k = {
        f"hit@{k}": exact_summary_for_k(sft_rank, rl_rank, k)
        for k in topk
    }
    prefix_summary = {
        f"prefix{level}": summarize_prefix(topk, ranks[0], ranks[1], sft_rank, rl_rank)
        for level, ranks in prefix_ranks.items()
    }
    group_summary = summarize_groups(
        compared,
        group_fields,
        topk,
        "sft_hit_rank_0_based",
        "rl_hit_rank_0_based",
    )
    focus_summary, focus_rows, focus_examples, focused_compared = summarize_focus_transitions(
        compared,
        group_fields,
        sorted(prefix_ranks),
        args.focus_k,
    )

    exact_rank_change_overall = {
        "rl_rank_better": int((rl_rank.fillna(math.inf) < sft_rank.fillna(math.inf)).sum()),
        "rl_rank_worse": int((rl_rank.fillna(math.inf) > sft_rank.fillna(math.inf)).sum()),
        "rl_rank_same": int((rl_rank.fillna(math.inf) == sft_rank.fillna(math.inf)).sum()),
    }

    summary = {
        "inputs": {
            "sft_csv": args.sft_csv.as_posix(),
            "rl_csv": args.rl_csv.as_posix(),
        },
        "topk": topk,
        "alignment": alignment_info,
        "columns_detected": {
            "exact_rank": {
                "sft": sft_rank_info,
                "rl": rl_rank_info,
            },
            "prefix_rank": prefix_columns_used,
            "group_fields": group_fields,
        },
        "overall": {
            "samples": int(len(compared)),
            "exact_rank_change_overall": exact_rank_change_overall,
            "exact_hit": exact_by_k,
            "prefix_hit": prefix_summary,
            "groups": group_summary,
            f"focus_hit@{args.focus_k}": focus_summary,
        },
        "warnings": warnings,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.output_dir / "summary.json"
    summary_csv = args.output_dir / "summary.csv"
    per_sample_csv = args.output_dir / "per_sample_compare.csv"
    focus_json = args.output_dir / f"focus_hit{args.focus_k}_summary.json"
    focus_csv = args.output_dir / f"focus_hit{args.focus_k}_breakdown.csv"
    focus_examples_csv = args.output_dir / f"focus_hit{args.focus_k}_examples.csv"

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    with open(focus_json, "w", encoding="utf-8") as f:
        json.dump(json_safe(focus_summary), f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    pd.DataFrame(flatten_summary(json_safe(summary))).to_csv(summary_csv, index=False)
    pd.DataFrame(focus_rows).to_csv(focus_csv, index=False)
    focus_examples.head(args.example_limit).to_csv(focus_examples_csv, index=False)
    compact_per_sample(focused_compared, group_fields, topk, sorted(prefix_ranks)).to_csv(per_sample_csv, index=False)

    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote per-sample compare CSV: {per_sample_csv}")
    print(f"Wrote focus summary JSON: {focus_json}")
    print(f"Wrote focus breakdown CSV: {focus_csv}")
    print(f"Wrote focus examples CSV: {focus_examples_csv}")
    print(
        "Aligned rows: "
        f"{alignment_info['aligned_rows']} "
        f"(method={alignment_info['method']})"
    )
    for k in topk:
        stats = exact_by_k[f"hit@{k}"]
        print(
            f"Hit@{k}: "
            f"SFT={stats['sft_hit_rate']:.6f} "
            f"RL={stats['rl_hit_rate']:.6f} "
            f"delta={stats['hit_rate_delta_rl_minus_sft']:.6f} "
            f"sft_only={stats['sft_only_hit']} "
            f"rl_only={stats['rl_only_hit']}"
        )
    focus_counts = focus_summary["transition_counts"]
    print(
        f"Focus Hit@{args.focus_k}: "
        f"sft_only={focus_counts.get('sft_only_hit', 0)} "
        f"rl_only={focus_counts.get('rl_only_hit', 0)} "
        f"both_hit={focus_counts.get('both_hit', 0)} "
        f"both_miss={focus_counts.get('both_miss', 0)}"
    )


if __name__ == "__main__":
    main()
