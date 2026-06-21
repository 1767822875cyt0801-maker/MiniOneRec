#!/usr/bin/env python3
"""Generate MiniOneRec SID files from an embedding matrix.

This wrapper is intentionally separate from the original rq scripts. It reads a
specified embedding file and item order, runs residual MiniBatchKMeans, and
writes generated SID artifacts under a versioned output directory.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from utils_sid import build_sid2items, parse_sid_tokens, save_json


def import_required_dependencies() -> tuple[Any, Any]:
    missing: list[str] = []
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
        missing.append("numpy")

    try:
        from sklearn.cluster import MiniBatchKMeans  # type: ignore
    except Exception:
        MiniBatchKMeans = None
        missing.append("scikit-learn")

    if missing:
        raise SystemExit(
            "Missing required dependencies for run_rqkmeans_with_emb.py: "
            + ", ".join(missing)
            + ". Install them first, for example: pip install numpy scikit-learn."
        )
    return np, MiniBatchKMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SID artifacts from an embedding matrix.")
    parser.add_argument("--category", required=True)
    parser.add_argument("--sid-version", required=True)
    parser.add_argument("--emb-path", type=Path, required=True)
    parser.add_argument("--item-order", type=Path, required=True)
    parser.add_argument("--row-index", type=Path, required=True)
    parser.add_argument("--item-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/Amazon/sid_maps/generated"))
    parser.add_argument("--num-levels", type=int, default=3)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--generated-manifest",
        type=Path,
        default=Path("data/Amazon/sid_maps/generated/generated_sid_manifest.json"),
    )
    parser.add_argument("--normalize", choices=["l2", "none"], default="l2")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_preserve_order(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def clean_tsv_field(value: Any) -> str:
    return str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


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


def load_item_order(path: Path) -> list[str]:
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"Expected list item_order JSON: {path}")
    return [str(item_id) for item_id in data]


def load_row_index(path: Path) -> dict[str, int]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict row_index JSON: {path}")
    out: dict[str, int] = {}
    for item_id, row in data.items():
        try:
            out[str(item_id)] = int(row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid row index for item {item_id!r}: {row!r}") from exc
    return out


def validate_order(item_order: list[str], row_index: dict[str, int], num_rows: int) -> tuple[bool, bool, list[str]]:
    warnings: list[str] = []
    item_order_ok = len(item_order) == num_rows and len(set(item_order)) == len(item_order)
    if not item_order_ok:
        warnings.append(
            f"item_order length/uniqueness check failed: len(item_order)={len(item_order)}, num_rows={num_rows}"
        )

    row_index_ok = set(row_index.keys()) == set(item_order)
    for row, item_id in enumerate(item_order):
        if row_index.get(item_id) != row:
            row_index_ok = False
            if len(warnings) < 20:
                warnings.append(f"row_index mismatch for item_id={item_id!r}: expected {row}, got {row_index.get(item_id)}")

    if not row_index_ok:
        warnings.append("row_index keys or positions do not exactly match item_order.")
    return item_order_ok, row_index_ok, warnings


def l2_normalize(np: Any, embeddings: Any) -> tuple[Any, int]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).astype(np.float32)
    zero_mask = norms[:, 0] <= 1e-12
    safe_norms = norms.copy()
    safe_norms[zero_mask] = 1.0
    normalized = embeddings / safe_norms
    if zero_mask.any():
        normalized[zero_mask] = 0.0
    return normalized.astype(np.float32, copy=False), int(zero_mask.sum())


def norm_stats(np: Any, matrix: Any) -> dict[str, float]:
    norms = np.linalg.norm(matrix, axis=1)
    if norms.size == 0:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(norms.min()),
        "mean": float(norms.mean()),
        "max": float(norms.max()),
    }


def run_residual_kmeans(
    np: Any,
    MiniBatchKMeans: Any,
    embeddings: Any,
    num_levels: int,
    codebook_size: int,
    max_iter: int,
    seed: int,
    batch_size: int | None,
) -> tuple[Any, list[Any], list[int], list[int], list[dict[str, float | int]]]:
    num_items = int(embeddings.shape[0])
    n_clusters = min(codebook_size, num_items)
    residual = embeddings.astype(np.float32, copy=True)
    reconstruction = np.zeros_like(residual, dtype=np.float32)
    codes = np.empty((num_items, num_levels), dtype=np.int32)
    codebooks: list[Any] = []
    actual_clusters_by_level: list[int] = []
    level_unique_codes: list[int] = []
    residual_norm_by_level: list[dict[str, float | int]] = [
        {"level": 0, **norm_stats(np, residual)}
    ]

    for level in range(num_levels):
        kwargs: dict[str, Any] = {
            "n_clusters": n_clusters,
            "random_state": seed + level,
            "max_iter": max_iter,
            "n_init": 10,
        }
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        model = MiniBatchKMeans(**kwargs)
        labels = model.fit_predict(residual).astype(np.int32, copy=False)
        centroids = model.cluster_centers_.astype(np.float32, copy=False)
        assigned = centroids[labels]

        codes[:, level] = labels
        codebooks.append(centroids)
        actual_clusters_by_level.append(int(centroids.shape[0]))
        level_unique_codes.append(int(len(np.unique(labels))))

        reconstruction += assigned
        residual -= assigned
        residual_norm_by_level.append({"level": level + 1, **norm_stats(np, residual)})

    reconstruction_mse = float(np.mean((embeddings - reconstruction) ** 2))
    residual_norm_by_level[-1]["reconstruction_mse"] = reconstruction_mse
    return codes, codebooks, actual_clusters_by_level, level_unique_codes, residual_norm_by_level


def codes_to_index(codes: Any, item_order: list[str]) -> dict[str, list[str]]:
    if codes.shape[1] > 26:
        raise ValueError("SID token format supports at most 26 levels with a-z prefixes.")
    index: dict[str, list[str]] = {}
    for row, item_id in enumerate(item_order):
        tokens = []
        for level, code in enumerate(codes[row].tolist()):
            tokens.append(f"<{chr(97 + level)}_{int(code)}>")
        index[item_id] = tokens
    return index


def build_item2sid(index: dict[str, list[str]]) -> dict[str, str]:
    return {item_id: "".join(tokens) for item_id, tokens in index.items()}


def build_item_mapping(
    item_order: list[str],
    item2sid: dict[str, str],
    item_titles: dict[str, str],
) -> dict[str, dict[str, Any]]:
    return {
        item_id: {
            "sid": item2sid[item_id],
            "title": item_titles.get(item_id, ""),
            "sid_tokens": parse_sid_tokens(item2sid[item_id]),
        }
        for item_id in item_order
    }


def load_item_titles(item_json_path: Path, item_order: list[str]) -> tuple[dict[str, str], list[str]]:
    raw = load_json(item_json_path)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected dict item JSON: {item_json_path}")
    titles: dict[str, str] = {}
    missing: list[str] = []
    for item_id in item_order:
        item_data = raw.get(item_id, {})
        if isinstance(item_data, dict):
            title = item_data.get("title", "")
        else:
            title = ""
        if title == "":
            missing.append(item_id)
        titles[item_id] = clean_tsv_field(title)
    return titles, missing


def write_info(path: Path, item_order: list[str], item2sid: dict[str, str], item_titles: dict[str, str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for item_id in item_order:
            f.write(f"{item2sid[item_id]}\t{item_titles.get(item_id, '')}\t{item_id}\n")
            count += 1
    return count


def bucket_metrics(item2sid: dict[str, str], sid2items: dict[str, list[str]]) -> dict[str, float | int]:
    num_items = len(item2sid)
    num_unique_sid = len(sid2items)
    bucket_sizes = [len(items) for items in sid2items.values()]
    collided_items = sum(size for size in bucket_sizes if size > 1)
    return {
        "num_unique_sid": num_unique_sid,
        "collision_rate": 1.0 - rate(num_unique_sid, num_items),
        "collided_item_rate": rate(collided_items, num_items),
        "max_bucket_size": max(bucket_sizes) if bucket_sizes else 0,
    }


def update_generated_manifest(path: Path, sid_version: str, category: str, entry: dict[str, Any]) -> None:
    if path.exists():
        manifest = load_json(path)
        if not isinstance(manifest, dict):
            raise TypeError(f"Expected dict generated manifest: {path}")
    else:
        manifest = {"sid_versions": {}}
    manifest.setdefault("sid_versions", {}).setdefault(sid_version, {})[category] = entry
    save_json(manifest, path)


def output_paths(output_dir: Path, category: str, sid_version: str) -> dict[str, Path]:
    return {
        "index": output_dir / f"{category}.index.json",
        "info": output_dir / f"{category}.info.txt",
        "item2sid": output_dir / f"item2sid_{sid_version}.json",
        "sid2items": output_dir / f"sid2items_{sid_version}.json",
        "valid_sid_set": output_dir / f"valid_sid_set_{sid_version}.json",
        "item_mapping": output_dir / f"item_mapping_{sid_version}.json",
        "codes": output_dir / f"{category}.codes.npy",
        "codebooks": output_dir / f"{category}.codebooks.npz",
        "generation_report": output_dir / "generation_report.json",
    }


def as_posix_map(paths: dict[str, Path]) -> dict[str, str]:
    return {key: path.as_posix() for key, path in paths.items()}


def print_summary(report: dict[str, Any]) -> None:
    print(f"[{report['category']}]")
    print(f"  emb_shape={report['emb_shape']}")
    print(f"  level_unique_codes={report['level_unique_codes']}")
    print(f"  num_unique_sid={report['num_unique_sid']}")
    print(f"  collision_rate={report['collision_rate']:.6f}")
    print(f"  collided_item_rate={report['collided_item_rate']:.6f}")
    print(f"  max_bucket_size={report['max_bucket_size']}")
    print(f"  reconstruction_mse={report['reconstruction_mse']:.8f}")
    print(f"  item_order_ok={report['item_order_ok']}")
    print(f"  row_index_ok={report['row_index_ok']}")
    print("  output_paths:")
    for key, value in report["output_paths"].items():
        print(f"    {key}: {value}")


def main() -> None:
    args = parse_args()
    if args.num_levels <= 0:
        raise ValueError("--num-levels must be positive")
    if args.num_levels > 26:
        raise ValueError("--num-levels must be <= 26")
    if args.codebook_size <= 0:
        raise ValueError("--codebook-size must be positive")
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive when provided")

    np, MiniBatchKMeans = import_required_dependencies()
    warnings: list[str] = []

    embeddings_raw = np.load(args.emb_path)
    if embeddings_raw.ndim != 2:
        raise ValueError(f"Embedding must be a 2D array, got shape={embeddings_raw.shape} at {args.emb_path}")
    embeddings = embeddings_raw.astype(np.float32, copy=False)
    num_items = int(embeddings.shape[0])
    if num_items == 0:
        raise ValueError("Embedding array has zero rows")

    item_order = load_item_order(args.item_order)
    row_index = load_row_index(args.row_index)
    item_order_ok, row_index_ok, order_warnings = validate_order(item_order, row_index, num_items)
    warnings.extend(order_warnings)
    if not item_order_ok or not row_index_ok:
        raise ValueError("item_order/row_index validation failed; refusing to generate SID artifacts")

    if args.codebook_size > num_items:
        warnings.append(
            f"codebook_size={args.codebook_size} exceeds num_items={num_items}; "
            f"using n_clusters={num_items}."
        )

    zero_norm_rows = 0
    if args.normalize == "l2":
        embeddings_for_kmeans, zero_norm_rows = l2_normalize(np, embeddings)
        if zero_norm_rows:
            warnings.append(f"{zero_norm_rows} embedding rows have zero norm before L2 normalization.")
    else:
        embeddings_for_kmeans = embeddings

    codes, codebooks, actual_clusters, unique_codes, residual_norms = run_residual_kmeans(
        np=np,
        MiniBatchKMeans=MiniBatchKMeans,
        embeddings=embeddings_for_kmeans,
        num_levels=args.num_levels,
        codebook_size=args.codebook_size,
        max_iter=args.max_iter,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    reconstruction_mse = float(residual_norms[-1].get("reconstruction_mse", math.nan))

    index = codes_to_index(codes, item_order)
    item2sid = build_item2sid(index)
    sid2items = build_sid2items(item2sid)
    item_titles, missing_title_items = load_item_titles(args.item_json, item_order)
    if missing_title_items:
        warnings.append(f"{len(missing_title_items)} items are missing title in item_json.")
    item_mapping = build_item_mapping(item_order, item2sid, item_titles)
    metrics = bucket_metrics(item2sid, sid2items)

    output_dir = args.output_root / args.sid_version / args.category
    paths = output_paths(output_dir, args.category, args.sid_version)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(paths["codes"], codes)
    np.savez_compressed(paths["codebooks"], **{f"codebook_{idx}": cb for idx, cb in enumerate(codebooks)})
    write_json_preserve_order(index, paths["index"])
    info_item_count = write_info(paths["info"], item_order, item2sid, item_titles)
    save_json(item2sid, paths["item2sid"])
    save_json(sid2items, paths["sid2items"])
    save_json(sorted(sid2items.keys()), paths["valid_sid_set"])
    save_json(item_mapping, paths["item_mapping"])

    path_strings = as_posix_map(paths)
    generated_manifest_entry = {
        "category": args.category,
        "sid_version": args.sid_version,
        "emb_path": args.emb_path.as_posix(),
        "item_order": args.item_order.as_posix(),
        "row_index": args.row_index.as_posix(),
        "item_json": args.item_json.as_posix(),
        "index": path_strings["index"],
        "info": path_strings["info"],
        "item2sid": path_strings["item2sid"],
        "sid2items": path_strings["sid2items"],
        "valid_sid_set": path_strings["valid_sid_set"],
        "item_mapping": path_strings["item_mapping"],
        "codes": path_strings["codes"],
        "codebooks": path_strings["codebooks"],
        "generation_report": path_strings["generation_report"],
    }
    update_generated_manifest(args.generated_manifest, args.sid_version, args.category, generated_manifest_entry)

    report = {
        "category": args.category,
        "sid_version": args.sid_version,
        "method": "residual_minibatch_kmeans",
        "emb_path": args.emb_path.as_posix(),
        "emb_shape": [int(dim) for dim in embeddings_raw.shape],
        "emb_dtype": str(embeddings_raw.dtype),
        "embedding_normalization": args.normalize,
        "zero_norm_rows": zero_norm_rows,
        "num_items": num_items,
        "num_levels": args.num_levels,
        "codebook_size": args.codebook_size,
        "actual_clusters_by_level": actual_clusters,
        "level_unique_codes": unique_codes,
        "num_unique_sid": metrics["num_unique_sid"],
        "collision_rate": metrics["collision_rate"],
        "collided_item_rate": metrics["collided_item_rate"],
        "max_bucket_size": metrics["max_bucket_size"],
        "reconstruction_mse": reconstruction_mse,
        "residual_norm_by_level": residual_norms,
        "item_order_ok": item_order_ok,
        "row_index_ok": row_index_ok,
        "index_item_count": len(index),
        "info_item_count": info_item_count,
        "item_order_first_last": [item_order[0], item_order[-1]],
        "row_index_first_last": [row_index[item_order[0]], row_index[item_order[-1]]],
        "kmeans": {
            "implementation": "sklearn.cluster.MiniBatchKMeans",
            "max_iter": args.max_iter,
            "seed": args.seed,
            "batch_size": args.batch_size,
        },
        "generated_manifest": args.generated_manifest.as_posix(),
        "output_paths": path_strings,
        "warnings": warnings,
    }
    save_json(report, paths["generation_report"])
    print_summary(report)


if __name__ == "__main__":
    main()
