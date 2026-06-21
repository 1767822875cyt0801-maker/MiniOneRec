#!/usr/bin/env python3
"""Build Collaborative-Semantic embeddings for MiniOneRec.

The script reads text embeddings and train interactions, then writes new CF/CS
embedding artifacts under a separate output directory. It never reads valid/test
splits and never overwrites the original text embeddings.
"""

from __future__ import annotations

import argparse
import ast
import csv
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Any

from utils_sid import load_json, save_json


def import_required_dependencies() -> tuple[Any, Any, Any]:
    missing: list[str] = []
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
        missing.append("numpy")

    try:
        from scipy import sparse  # type: ignore
    except Exception:
        sparse = None
        missing.append("scipy")

    try:
        from sklearn.decomposition import TruncatedSVD  # type: ignore
    except Exception:
        TruncatedSVD = None
        missing.append("scikit-learn")

    if missing:
        raise SystemExit(
            "Missing required dependencies for build_cs_embeddings.py: "
            + ", ".join(missing)
            + ". Install them first, for example: pip install numpy scipy scikit-learn. "
            + "The script will not fall back to dense matrix computation."
        )

    return np, sparse, TruncatedSVD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build collaborative-semantic embeddings.")
    parser.add_argument("--manifest", type=Path, default=Path("data/Amazon/sid_maps/experiment_manifest.json"))
    parser.add_argument("--sid-version", default="text")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--text-emb-root", type=Path, default=Path("data/Amazon/index"))
    parser.add_argument("--text-emb-suffix", default="emb-qwen-td.npy")
    parser.add_argument("--output-dir", type=Path, default=Path("data/Amazon/cs_embeddings"))
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.7, 0.5])
    parser.add_argument("--cf-dim", default="auto")
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--min-cooccur", type=int, default=1)
    parser.add_argument("--normalize", choices=["l2"], default="l2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    return parser.parse_args()


def item_sort_key(value: Any) -> tuple[int, int | str]:
    value_str = str(value)
    try:
        return (0, int(value_str))
    except ValueError:
        return (1, value_str)


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    value_str = "" if value is None else str(value).strip()
    if not value_str:
        return []
    parsed = ast.literal_eval(value_str)
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def text_embedding_path(root: Path, category: str, suffix: str) -> Path:
    return root / f"{category}.{suffix.lstrip('.')}"


def alpha_tag(alpha: float) -> str:
    return f"alpha{alpha:g}"


def row_norms(np: Any, matrix: Any) -> Any:
    return np.linalg.norm(matrix, axis=1)


def norm_stats(np: Any, values: Any) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def l2_normalize(np: Any, matrix: Any, keep_zero: bool = True) -> tuple[Any, Any]:
    norms = row_norms(np, matrix).astype(np.float32)
    denom = norms.copy()
    zero_mask = denom <= 1e-12
    denom[zero_mask] = 1.0
    normalized = matrix / denom[:, None]
    if keep_zero and zero_mask.any():
        normalized[zero_mask] = 0.0
    return normalized.astype(np.float32, copy=False), norms


def load_train_interactions(
    train_csv: Path,
    row_index: dict[str, int],
    window_size: int,
) -> tuple[Counter[tuple[int, int]], Any, int, int, list[str], int]:
    pair_counts: Counter[tuple[int, int]] = Counter()
    train_counts = [0] * len(row_index)
    train_rows = 0
    skipped_items = 0
    warnings: list[str] = []

    with open(train_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"history_item_id", "item_id"}
        missing_cols = sorted(required - set(reader.fieldnames or []))
        if missing_cols:
            raise ValueError(f"{train_csv} missing required columns: {missing_cols}")

        for row_no, row in enumerate(reader, start=2):
            train_rows += 1
            target_id = str(row.get("item_id", "")).strip()
            try:
                history_ids = [str(item).strip() for item in parse_list(row.get("history_item_id", "[]"))]
            except Exception as exc:
                if len(warnings) < 20:
                    warnings.append(f"Failed to parse history_item_id at {train_csv}:{row_no}: {exc}")
                history_ids = []

            if window_size > 0:
                history_ids = history_ids[-window_size:]

            raw_context = history_ids + ([target_id] if target_id else [])
            context_rows: set[int] = set()
            for item_id in raw_context:
                if item_id not in row_index:
                    skipped_items += 1
                    continue
                item_row = row_index[item_id]
                context_rows.add(item_row)
                train_counts[item_row] += 1

            if len(context_rows) < 2:
                continue

            ordered_rows = sorted(context_rows)
            for left, right in itertools.combinations(ordered_rows, 2):
                pair_counts[(left, right)] += 1
                pair_counts[(right, left)] += 1

    return pair_counts, train_counts, train_rows, skipped_items, warnings, sum(1 for count in train_counts if count > 0)


def build_cooccurrence_matrix(
    np: Any,
    sparse: Any,
    pair_counts: Counter[tuple[int, int]],
    num_items: int,
    min_cooccur: int,
) -> tuple[Any, int, int]:
    nnz_before_filter = len(pair_counts)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (row, col), count in pair_counts.items():
        if count >= min_cooccur:
            rows.append(row)
            cols.append(col)
            data.append(float(count))

    matrix = sparse.coo_matrix(
        (
            np.asarray(data, dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(num_items, num_items),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix, nnz_before_filter, matrix.nnz


def cooccurrence_to_ppmi(np: Any, sparse: Any, cooc: Any) -> Any:
    coo = cooc.tocoo()
    if coo.nnz == 0:
        return cooc.copy()

    row_sum = np.asarray(cooc.sum(axis=1)).ravel().astype(np.float64)
    col_sum = np.asarray(cooc.sum(axis=0)).ravel().astype(np.float64)
    total = float(coo.data.sum())
    denom = row_sum[coo.row] * col_sum[coo.col]
    valid = denom > 0
    values = np.zeros_like(coo.data, dtype=np.float64)
    values[valid] = np.log((coo.data[valid].astype(np.float64) * total) / denom[valid])
    keep = np.isfinite(values) & (values > 0)
    ppmi = sparse.coo_matrix(
        (values[keep].astype(np.float32), (coo.row[keep], coo.col[keep])),
        shape=cooc.shape,
        dtype=np.float32,
    ).tocsr()
    ppmi.sum_duplicates()
    return ppmi


def run_svd(np: Any, TruncatedSVD: Any, ppmi: Any, cf_dim: int, seed: int) -> tuple[Any, float | None]:
    if ppmi.nnz == 0:
        return np.zeros((ppmi.shape[0], cf_dim), dtype=np.float32), None

    svd = TruncatedSVD(n_components=cf_dim, random_state=seed)
    cf_emb = svd.fit_transform(ppmi).astype(np.float32, copy=False)
    evr = getattr(svd, "explained_variance_ratio_", None)
    evr_sum = float(evr.sum()) if evr is not None else None
    return cf_emb, evr_sum


def resolve_cf_dim(cf_dim_arg: str, text_dim: int, num_items: int) -> int:
    if cf_dim_arg == "auto":
        cf_dim = text_dim
    else:
        try:
            cf_dim = int(cf_dim_arg)
        except ValueError as exc:
            raise ValueError(f"--cf-dim must be 'auto' or an integer, got {cf_dim_arg!r}") from exc
        if cf_dim != text_dim:
            raise ValueError(
                f"V1 weighted-sum fusion requires cf_dim == text_dim. "
                f"Got cf_dim={cf_dim}, text_dim={text_dim}. "
                f"Use --cf-dim auto or specify {text_dim}."
            )
    if cf_dim <= 0:
        raise ValueError(f"cf_dim must be positive, got {cf_dim}")
    if cf_dim > num_items:
        raise ValueError(
            f"cf_dim={cf_dim} exceeds min(matrix shape)={num_items}; "
            f"TruncatedSVD cannot produce that many components for this item matrix."
        )
    return cf_dim


def build_item_order(num_items: int) -> tuple[list[str], dict[str, int]]:
    item_order = [str(idx) for idx in range(num_items)]
    row_index = {item_id: idx for idx, item_id in enumerate(item_order)}
    return item_order, row_index


def build_category_embeddings(
    np: Any,
    sparse: Any,
    TruncatedSVD: Any,
    category: str,
    entry: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    warnings: list[str] = []
    train_csv = Path(entry["train_csv"])
    item2sid_path = Path(entry["item2sid"])
    item_mapping_path = Path(entry["item_mapping"])
    text_path = text_embedding_path(args.text_emb_root, category, args.text_emb_suffix)
    out_dir = args.output_dir / category
    out_dir.mkdir(parents=True, exist_ok=True)

    text_emb_raw = np.load(text_path)
    text_emb = text_emb_raw.astype(np.float32, copy=False)
    if text_emb.ndim != 2:
        raise ValueError(f"Text embedding must be a 2D array, got shape={text_emb.shape} at {text_path}")

    num_items, text_dim = int(text_emb.shape[0]), int(text_emb.shape[1])
    item2sid = {str(item_id): sid for item_id, sid in load_json(item2sid_path).items()}
    item_mapping = load_json(item_mapping_path)
    item_order, row_index = build_item_order(num_items)
    item_order_ok = len(item_order) == num_items and all(item_order[idx] == str(idx) for idx in range(num_items))
    item2sid_coverage_ok = set(item_order) == set(item2sid.keys())
    item_mapping_coverage_ok = set(item_order) == set(str(key) for key in item_mapping.keys())
    if not item2sid_coverage_ok:
        warnings.append("item_order does not exactly match item2sid keys.")
    if not item_mapping_coverage_ok:
        warnings.append("item_order does not exactly match item_mapping keys.")

    save_json(item_order, out_dir / f"{category}.item_order.json")
    save_json(row_index, out_dir / f"{category}.row_index.json")

    pair_counts, train_counts, train_rows, skipped_items, parse_warnings, cf_covered_items = load_train_interactions(
        train_csv=train_csv,
        row_index=row_index,
        window_size=args.window_size,
    )
    warnings.extend(parse_warnings)
    if skipped_items:
        warnings.append(f"Skipped {skipped_items} train item references that were absent from row_index.")

    cooc, nnz_before, nnz_after = build_cooccurrence_matrix(
        np=np,
        sparse=sparse,
        pair_counts=pair_counts,
        num_items=num_items,
        min_cooccur=args.min_cooccur,
    )
    ppmi = cooccurrence_to_ppmi(np, sparse, cooc)
    sparsity = 1.0 - (float(nnz_after) / float(num_items * num_items))
    cf_dim = resolve_cf_dim(args.cf_dim, text_dim, num_items)
    cf_emb, evr_sum = run_svd(np, TruncatedSVD, ppmi, cf_dim, args.seed)

    train_counts_arr = np.asarray(train_counts, dtype=np.int64)
    cold_mask = train_counts_arr == 0
    if cold_mask.any():
        cf_emb[cold_mask] = 0.0
    raw_cf_norms = row_norms(np, cf_emb)
    warm_zero_cf_mask = (~cold_mask) & (raw_cf_norms <= 1e-12)
    if warm_zero_cf_mask.any():
        warnings.append(
            f"{int(warm_zero_cf_mask.sum())} warm items have zero CF norm after filtering/SVD; "
            f"they will fall back to text embeddings during fusion."
        )

    cf_emb_norm, cf_norms = l2_normalize(np, cf_emb, keep_zero=True)
    text_emb_norm, text_norms = l2_normalize(np, text_emb, keep_zero=True)
    if (text_norms <= 1e-12).any():
        warnings.append(f"{int((text_norms <= 1e-12).sum())} text embedding rows have zero norm.")

    cf_available_mask_arr = (~cold_mask) & (cf_norms > 1e-12)
    cf_available_mask = {
        item_id: bool(cf_available_mask_arr[row_index[item_id]])
        for item_id in item_order
    }
    save_json(cf_available_mask, out_dir / f"{category}.cf_available_mask.json")
    np.save(out_dir / f"{category}.cf_emb.npy", cf_emb.astype(args.dtype, copy=False))

    cs_shapes: dict[str, list[int]] = {}
    cs_norm_stats: dict[str, dict[str, float]] = {}
    output_dtype = np.float32 if args.dtype == "float32" else np.float16
    for alpha in args.alpha:
        fused = alpha * text_emb_norm + (1.0 - alpha) * cf_emb_norm
        fallback_mask = ~cf_available_mask_arr
        if fallback_mask.any():
            fused[fallback_mask] = text_emb_norm[fallback_mask]
        fused_norm, _ = l2_normalize(np, fused.astype(np.float32, copy=False), keep_zero=True)
        tag = alpha_tag(alpha)
        out_path = out_dir / f"{category}.cs_emb_{tag}.npy"
        np.save(out_path, fused_norm.astype(output_dtype, copy=False))
        cs_shapes[tag] = [int(dim) for dim in fused_norm.shape]
        cs_norm_stats[tag] = norm_stats(np, row_norms(np, fused_norm))

    report = {
        "category": category,
        "text_emb_path": text_path.as_posix(),
        "text_emb_shape": [num_items, text_dim],
        "text_emb_dtype": str(text_emb_raw.dtype),
        "cf_emb_shape": [int(dim) for dim in cf_emb.shape],
        "cs_emb_shapes": cs_shapes,
        "num_items": num_items,
        "train_rows": train_rows,
        "cooccurrence_nnz_before_filter": int(nnz_before),
        "cooccurrence_nnz_after_filter": int(nnz_after),
        "ppmi_nnz": int(ppmi.nnz),
        "sparsity": sparsity,
        "cf_dim": cf_dim,
        "svd_explained_variance_ratio_sum": evr_sum,
        "cf_covered_items": int(cf_covered_items),
        "cold_fallback_items": int(cold_mask.sum()),
        "cold_fallback_rate": float(cold_mask.mean()) if num_items else 0.0,
        "warm_zero_cf_items": int(warm_zero_cf_mask.sum()),
        "item_order_ok": bool(item_order_ok),
        "item2sid_coverage_ok": bool(item2sid_coverage_ok),
        "item_mapping_coverage_ok": bool(item_mapping_coverage_ok),
        "no_valid_test_used": True,
        "text_norm_min": norm_stats(np, text_norms)["min"],
        "text_norm_mean": norm_stats(np, text_norms)["mean"],
        "text_norm_max": norm_stats(np, text_norms)["max"],
        "cf_norm_min": norm_stats(np, cf_norms[cf_available_mask_arr])["min"],
        "cf_norm_mean": norm_stats(np, cf_norms[cf_available_mask_arr])["mean"],
        "cf_norm_max": norm_stats(np, cf_norms[cf_available_mask_arr])["max"],
        "cf_norm_including_cold": norm_stats(np, cf_norms),
        "cs_norm_min_mean_max": cs_norm_stats,
        "window_size": args.window_size,
        "min_cooccur": args.min_cooccur,
        "alpha": list(args.alpha),
        "dtype": args.dtype,
        "warnings": warnings,
        "outputs": {
            "cf_emb": (out_dir / f"{category}.cf_emb.npy").as_posix(),
            "cf_available_mask": (out_dir / f"{category}.cf_available_mask.json").as_posix(),
            "item_order": (out_dir / f"{category}.item_order.json").as_posix(),
            "row_index": (out_dir / f"{category}.row_index.json").as_posix(),
            "cs_embeddings": {
                alpha_tag(alpha): (out_dir / f"{category}.cs_emb_{alpha_tag(alpha)}.npy").as_posix()
                for alpha in args.alpha
            },
            "report": (out_dir / f"{category}.cs_embedding_report.json").as_posix(),
        },
    }
    save_json(report, out_dir / f"{category}.cs_embedding_report.json")
    return report


def print_report_summary(report: dict[str, Any]) -> None:
    print(f"[{report['category']}]")
    print(f"  text_emb_shape={report['text_emb_shape']}")
    print(f"  cf_emb_shape={report['cf_emb_shape']}")
    print(f"  cs_emb_shapes={report['cs_emb_shapes']}")
    print(
        f"  cooccurrence_nnz_before_filter={report['cooccurrence_nnz_before_filter']} "
        f"cooccurrence_nnz_after_filter={report['cooccurrence_nnz_after_filter']} "
        f"ppmi_nnz={report['ppmi_nnz']}"
    )
    print(
        f"  cf_covered_items={report['cf_covered_items']} "
        f"cold_fallback_items={report['cold_fallback_items']} "
        f"cold_fallback_rate={report['cold_fallback_rate']:.6f}"
    )
    print(
        f"  item_order_ok={report['item_order_ok']} "
        f"item2sid_coverage_ok={report['item2sid_coverage_ok']}"
    )
    print(f"  cs_norm_min_mean_max={report['cs_norm_min_mean_max']}")


def main() -> None:
    args = parse_args()
    np, sparse, TruncatedSVD = import_required_dependencies()
    manifest = load_json(args.manifest)
    sid_versions = manifest.get("sid_versions", {})
    if args.sid_version not in sid_versions:
        raise KeyError(f"sid_version={args.sid_version!r} not found in {args.manifest}")

    category_entries = sid_versions[args.sid_version]
    categories = args.categories or sorted(category_entries)
    for category in categories:
        if category not in category_entries:
            raise KeyError(f"category={category!r} not found under sid_version={args.sid_version!r}")
        report = build_category_embeddings(np, sparse, TruncatedSVD, category, category_entries[category], args)
        print_report_summary(report)


if __name__ == "__main__":
    main()
