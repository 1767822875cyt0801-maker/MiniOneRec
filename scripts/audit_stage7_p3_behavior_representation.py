#!/usr/bin/env python3
"""Stage 7 P3-0 behavior representation read-only audit.

The script inspects existing code and artifacts, then writes an audit report.
It does not train, rebuild embeddings/SIDs, run inference, or evaluate test.
The test CSV is read only for split-provenance overlap counts.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


FROZEN_P2_RANKER_ID = (
    "p2_history_ranker__Industrial_and_Scientific__valid__exact__"
    "hist_lr0.003_l20.01_ep18_neg80"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit current CF baseline and SASRec readiness for P3-0."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--category", default="Industrial_and_Scientific")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--candidate-mode", default="exact")
    parser.add_argument("--cf-version", default="cf_k512_dedup")
    parser.add_argument("--text-version", default="text_mbk_k512_dedup")
    parser.add_argument(
        "--stage7-root",
        type=Path,
        default=Path("results/stage7_validation_protocol/valid/Industrial_and_Scientific"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--write", action="store_true", default=True)
    return parser.parse_args()


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def file_status(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"path": path.as_posix(), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        out.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime_epoch": float(stat.st_mtime),
            }
        )
    return out


def parse_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    value_str = "" if value is None else str(value).strip()
    if not value_str:
        return []
    parsed = ast.literal_eval(value_str)
    if isinstance(parsed, (list, tuple)):
        return list(parsed)
    return [parsed]


def csv_item_stats(path: Path) -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    rows = 0
    history_refs = 0
    parse_errors = 0
    all_items: set[str] = set()
    target_items: set[str] = set()
    history_items: set[str] = set()
    columns: list[str] = []
    first_row: dict[str, str] | None = None

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for row in reader:
            if first_row is None:
                first_row = dict(row)
            rows += 1
            target_id = str(row.get("item_id", "")).strip()
            if target_id:
                target_items.add(target_id)
                all_items.add(target_id)
            try:
                history = parse_list(row.get("history_item_id", "[]"))
            except Exception:
                parse_errors += 1
                history = []
            for item in history:
                item_id = str(item).strip()
                if item_id:
                    history_items.add(item_id)
                    all_items.add(item_id)
                    history_refs += 1

    stats = {
        "path": path.as_posix(),
        "exists": True,
        "columns": columns,
        "rows": rows,
        "unique_all_items": len(all_items),
        "unique_target_items": len(target_items),
        "unique_history_items": len(history_items),
        "history_item_refs": history_refs,
        "history_parse_errors": parse_errors,
        "first_row_has_user_id": bool(first_row and first_row.get("user_id")),
    }
    return stats, all_items, target_items, history_items


def safe_sample(values: set[str] | list[str], n: int = 20) -> list[str]:
    return sorted(str(v) for v in values)[:n]


def load_numpy_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": path.as_posix(), "exists": False}
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        return {
            "path": path.as_posix(),
            "exists": True,
            "numpy_available": False,
            "error": repr(exc),
        }

    try:
        arr = np.load(path, mmap_mode="r")
        meta: dict[str, Any] = {
            "path": path.as_posix(),
            "exists": True,
            "numpy_available": True,
            "shape": [int(dim) for dim in arr.shape],
            "dtype": str(arr.dtype),
        }
        if arr.ndim == 2:
            arr32 = np.asarray(arr, dtype=np.float32)
            norms = np.linalg.norm(arr32, axis=1)
            meta["row_norm"] = {
                "min": float(norms.min()) if norms.size else 0.0,
                "mean": float(norms.mean()) if norms.size else 0.0,
                "max": float(norms.max()) if norms.size else 0.0,
                "zero_rows": int((norms <= 1e-12).sum()),
            }
        return meta
    except Exception as exc:
        return {"path": path.as_posix(), "exists": True, "error": repr(exc)}


def load_numpy_row_norms(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    try:
        import numpy as np  # type: ignore

        arr = np.load(path, mmap_mode="r")
        if arr.ndim != 2:
            return None
        norms = np.linalg.norm(np.asarray(arr, dtype=np.float32), axis=1)
        return [float(x) for x in norms.tolist()]
    except Exception:
        return None


def list_existing(paths: list[Path]) -> list[dict[str, Any]]:
    return [file_status(path) for path in paths]


def code_contains(path: Path, needles: list[str]) -> dict[str, bool]:
    if not path.exists():
        return {needle: False for needle in needles}
    text = read_text(path)
    return {needle: needle in text for needle in needles}


def collect_sasrec_inventory(repo: Path) -> dict[str, Any]:
    ignored = {".git", "results", "outputs", "__pycache__", ".venv", "venv"}
    primary_files = {"sasrec.py", "SASRecModules_ori.py"}
    excluded_files = {"audit_stage7_p3_behavior_representation.py"}
    candidate_files: list[Path] = []
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(repo).parts)
        if parts & ignored:
            continue
        if path.name in excluded_files:
            continue
        if path.suffix != ".py" and path.name not in primary_files:
            continue
        try:
            text = read_text(path)
        except Exception:
            continue
        is_primary = path.name in primary_files
        is_runtime_usage = (
            "from sasrec import SASRec" in text
            or "SASRec(" in text
            or 'reward_type == "sasrec"' in text
            or "reward_type == 'sasrec'" in text
        )
        if is_primary or is_runtime_usage:
            candidate_files.append(path)

    inventory_files: list[dict[str, Any]] = []
    for path in sorted(candidate_files):
        text = read_text(path)
        inventory_files.append(
            {
                "path": rel(path, repo),
                "lines": text.count("\n") + 1,
                "has_class_SASRec": "class SASRec" in text,
                "has_main": "if __name__ == '__main__'" in text
                or 'if __name__ == "__main__"' in text,
                "has_item_embeddings": "item_embeddings" in text
                or "item_emb" in text,
                "has_s_fc": "s_fc" in text,
                "has_export_item_embedding": "sasrec_item_emb" in text
                or "export" in text.lower(),
                "reads_test_dir": "data_directory_test" in text
                or "./data/Amazon/test" in text,
                "uses_BCE": "BCEWithLogitsLoss" in text,
                "uses_CE": "CrossEntropyLoss" in text,
                "uses_negative_sampling": "np.random.randint(item_num)" in text,
                "saves_state_dict": "state_dict" in text and "torch.save" in text,
            }
        )

    sasrec_py = repo / "sasrec.py"
    sasrec_facts = code_contains(
        sasrec_py,
        [
            "class SASRec",
            "self.item_embeddings",
            "self.s_fc",
            "BCEWithLogitsLoss",
            "CrossEntropyLoss",
            "np.random.randint(item_num)",
            "data_directory_train = './data/Amazon/train/'",
            "data_directory_valid = './data/Amazon/valid/'",
            "data_directory_test = './data/Amazon/test/'",
            "seq_size = 10",
            "item_num = len(data_info)",
            "torch.save(best_model.state_dict()",
            "save_logits",
        ],
    )
    modules_facts = code_contains(
        repo / "SASRecModules_ori.py",
        ["class MultiHeadAttention", "class PositionwiseFeedForward", "torch.tril"],
    )

    return {
        "files": inventory_files,
        "summary": {
            "has_model_class": sasrec_facts.get("class SASRec", False),
            "has_standalone_trainer": sasrec_facts.get(
                "data_directory_train = './data/Amazon/train/'", False
            )
            and sasrec_facts.get("torch.save(best_model.state_dict()", False),
            "has_evaluator": sasrec_facts.get("data_directory_test = './data/Amazon/test/'", False),
            "has_embedding_exporter": False,
            "directly_reusable_for_p3": False,
            "reason_not_directly_reusable": (
                "Legacy sasrec.py mixes model/training/evaluation, reads test during training, "
                "uses pad id=item_num, uses an untied s_fc output head, and has no "
                "sasrec_item_emb.npy/item_id_order.json exporter."
            ),
        },
        "sasrec_py_facts": sasrec_facts,
        "modules_facts": modules_facts,
    }


def probe_checkpoint(path: Path) -> dict[str, Any]:
    parent = path.parent
    probe_paths = [
        path / "config.json",
        path / "adapter_config.json",
        path / "tokenizer_config.json",
        path / "special_tokens_map.json",
        path / "generation_config.json",
        path / "pytorch_model.bin",
        path / "model.safetensors",
        parent / "trainer_state.json",
        parent / "training_args.bin",
        parent / "README.md",
    ]
    trainer_state = read_json(parent / "trainer_state.json", default=None)
    return {
        "checkpoint_dir": path.as_posix(),
        "exists": path.exists() and path.is_dir(),
        "files": list_existing(probe_paths),
        "trainer_state_available": trainer_state is not None,
        "trainer_state_best_model_checkpoint": (
            trainer_state.get("best_model_checkpoint")
            if isinstance(trainer_state, dict)
            else None
        ),
        "trainer_state_global_step": (
            trainer_state.get("global_step") if isinstance(trainer_state, dict) else None
        ),
    }


def build_contract(category: str) -> dict[str, Any]:
    base = f"data/Amazon/behavior_embeddings/sasrec/{category}"
    return {
        "sequence_schema": {
            "source": f"data/Amazon/train/{category}_*.csv",
            "columns": ["user_id", "history_item_id", "item_id"],
            "sample": "one next-item sample per train row: chronological history_item_id -> item_id",
            "history_order": "preserve the order already stored in history_item_id",
            "valid_usage": "valid.csv may be used only for early stopping/model selection",
            "test_usage": "test.csv is not read by training or model selection",
        },
        "temporal_boundary": {
            "positive_interactions": "train split only",
            "valid_future_interactions": "forbidden for training/export",
            "test_future_interactions": "forbidden for training/export",
            "future_target_labels": "forbidden outside the chosen split target in each supervised train row",
        },
        "item_vocabulary": {
            "source": f"data/Amazon/cs_embeddings/{category}/{category}.item_order.json",
            "coverage": "all canonical original item_id rows, including train-cold rows",
            "internal_index_rule": "internal_id = row_index[item_id] + 1",
            "padding_id": 0,
            "unknown_item": "fail closed by default; do not silently map to padding",
            "mask_id": "not used for causal SASRec P3-1",
        },
        "model": {
            "minimal_hidden_dim": 128,
            "max_seq_len": "configurable; start with 50 for P3-1 smoke",
            "loss": "sampled softmax or full softmax over internal item ids 1..N",
            "negative_sampling": "train-only positives; negatives sampled from vocabulary without using valid/test labels",
            "output_head_contract": "tie scores to item_emb.weight so target item embeddings are directly trained",
        },
        "export": {
            "tensor_source": "model.item_emb.weight[1:num_items+1]",
            "use_item_emb_weight": True,
            "postprocess": "float32 L2 row-normalization after zeroing train-unseen positive rows if configured",
            "row_alignment": "sasrec_item_emb.npy row i corresponds exactly to item_id_order.json[i]",
            "required_outputs": [
                f"{base}/sasrec_item_emb.npy",
                f"{base}/item_id_order.json",
                f"{base}/row_index.json",
                f"{base}/sasrec_config.json",
                f"{base}/checkpoint/",
                f"{base}/checkpoint_provenance.json",
                f"{base}/embedding_manifest.json",
            ],
        },
        "alignment_checks": [
            "len(item_id_order) == sasrec_item_emb.shape[0]",
            "row_index[item_id_order[i]] == i for every row",
            "set(item_id_order) == set(cf_k512_dedup item2sid keys)",
            "manifest records train_rows, train_positive_items, train_cold_items",
            "valid/test-only item ids are identified and never used as positives during training",
        ],
    }


def build_fair_protocol(category: str, split: str, candidate_mode: str) -> dict[str, Any]:
    root = f"results/stage7_validation_protocol/{split}/{category}"
    return {
        "research_question": (
            "Does replacing PPMI/SVD with SASRec behavioral representation improve "
            "behavioral candidate retrieval and final recommendation quality?"
        ),
        "fixed_controls": {
            "dataset": category,
            "development_split": split,
            "candidate_mode": candidate_mode,
            "text_branch": "frozen",
            "text_checkpoint": (
                "outputs/sft_sidonly_Industrial_and_Scientific_text_mbk_k512_dedup_"
                "sample30000_ep1_noearly/final_checkpoint"
            ),
            "candidate_budgets": "frozen from Stage 7 valid exact protocol",
            "fusion_contract": "frozen dual SID exact candidate schema",
            "ranker": FROZEN_P2_RANKER_ID,
            "valid_fit_valid_select": "frozen P2 protocol",
            "test": "untouched",
        },
        "baseline_chain": [
            "PPMI/SVD CF embedding",
            "cf_k512_dedup SID",
            "existing CF generator checkpoint",
            "exact candidates",
            "same fusion contract",
            "frozen P2 ranker",
        ],
        "new_chain": [
            "SASRec train-only item embedding",
            "new SASRec CF-SID, e.g. sasrec_k512_dedup",
            "retrained CF generator only for SASRec-SID",
            "exact candidates with same budget",
            "same fusion contract",
            "same frozen P2 ranker",
        ],
        "metric_groups": {
            "representation_sid": [
                "embedding shape/dtype/norm/zero rows",
                "pre-dedup collision",
                "post-dedup collision",
                "codebook utilization",
                "bucket size distribution",
            ],
            "candidate_side": [
                "CF GT coverage",
                "Text-only coverage",
                "CF-only coverage",
                "Both",
                "Neither",
                "Text union CF coverage",
                "source overlap",
            ],
            "final_ranking": [
                "HR@10",
                "HR@20",
                "HR@50",
                "NDCG@10",
                "NDCG@20",
                "NDCG@50",
            ],
            "decision_focus": [
                "CF-only coverage gain",
                "union coverage gain",
                "frozen-ranker HR@20 gain",
                "frozen-ranker NDCG@20 gain",
            ],
        },
        "expected_artifact_root": root,
    }


def build_artifact_tree(category: str, split: str) -> list[str]:
    return [
        f"results/stage7_validation_protocol/{split}/{category}/p3_behavior_audit/",
        "  p3_behavior_audit_report.json",
        "  p3_behavior_audit_report.md",
        "  p3_behavior_audit_checks.csv",
        f"data/Amazon/behavior_embeddings/sasrec/{category}/",
        "  checkpoint/",
        "  sasrec_item_emb.npy",
        "  item_id_order.json",
        "  row_index.json",
        "  sasrec_config.json",
        "  checkpoint_provenance.json",
        "  embedding_manifest.json",
        f"data/Amazon/sid_versions/sasrec_k512_dedup/{category}/",
        "  index.json",
        "  item2sid.json",
        "  sid2items.json",
        "  valid_sid_set.json",
        "  train.csv",
        "  valid.csv",
        "  test.csv (rewrite only in later stage; no test evaluation)",
    ]


def build_p3_1_plan(category: str) -> list[str]:
    return [
        "Add a new train-only SASRec builder/exporter; do not reuse sasrec.py main.",
        "Build canonical item vocabulary from existing item_order/row_index and use pad id 0.",
        "Train on raw train CSV only; use valid CSV only for early stopping and model selection.",
        "Export L2-normalized sasrec_item_emb.npy plus item_id_order.json, row_index.json, config, checkpoint provenance, and manifest.",
        "Run SID generation with run_rqkmeans_with_emb.py using codebook_size=512, num_levels=3, dedup_mode=append.",
        "Rewrite train/valid CSV for SASRec-SID; defer any test rewrite/evaluation until the frozen protocol allows it.",
        "Retrain only the SASRec-SID CF generator under the same SFT budget as cf_k512_dedup.",
        "Run valid exact candidates, frozen fusion, and the frozen P2 ranker for comparison.",
    ]


def split_leakage_classification(checks: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if checks.get("cf_report_no_valid_test_used") is False:
        return "CONFIRMED_LEAKAGE", ["cs_embedding_report explicitly says no_valid_test_used is false."]
    if checks.get("validtest_only_nonzero_cf_rows", 0) > 0:
        return (
            "POSSIBLE_LEAKAGE",
            [
                "Items absent from train but present in valid/test have non-zero CF vectors.",
                "This conflicts with a strict train-only PPMI/SVD interaction signal.",
            ],
        )
    clean_signals = [
        checks.get("source_builder_uses_entry_train_csv"),
        checks.get("source_builder_has_no_valid_test_cli_inputs"),
        checks.get("cf_report_no_valid_test_used"),
        checks.get("cf_report_train_rows_match_raw_train"),
        checks.get("validtest_only_nonzero_cf_rows", 0) == 0,
    ]
    if all(clean_signals):
        reasons.extend(
            [
                "build_cs_embeddings.py derives interactions from entry['train_csv'].",
                "The builder has no valid_csv/test_csv CLI input path.",
                "cs_embedding_report records no_valid_test_used=true and train_rows matching raw train rows.",
                "Items seen only outside train have zero CF rows.",
            ]
        )
        return "CLEAN_TRAIN_ONLY", reasons
    if any(signal is None for signal in clean_signals):
        return "UNKNOWN", ["One or more required provenance checks could not be evaluated."]
    return "POSSIBLE_LEAKAGE", ["Train-only provenance is incomplete or internally inconsistent."]


def markdown_table(rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(str(row[idx])) for row in rows) for idx in range(len(rows[0]))]
    lines: list[str] = []
    for row_idx, row in enumerate(rows):
        line = "| " + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"
        lines.append(line)
        if row_idx == 0:
            lines.append("| " + " | ".join("-" * widths[idx] for idx in range(len(row))) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Stage 7 P3-0 Behavior Representation Upgrade Audit")
    lines.append("")
    lines.append(f"- category: `{report['category']}`")
    lines.append(f"- split: `{report['split']}`")
    lines.append(f"- candidate_mode: `{report['candidate_mode']}`")
    lines.append(f"- frozen_p2_ranker: `{FROZEN_P2_RANKER_ID}`")
    lines.append(f"- leakage_class: `{report['split_leakage_audit']['classification']}`")
    lines.append("")

    lines.append("## 1. Current CF Baseline Provenance")
    cf = report["current_cf_baseline_provenance"]
    lines.extend(
        markdown_table(
            [
                ["component", "actual path / value"],
                ["raw train", cf["raw_interaction_files"]["train"]["path"]],
                ["raw valid", cf["raw_interaction_files"]["valid"]["path"]],
                ["raw test", cf["raw_interaction_files"]["test"]["path"]],
                ["cf embedding", cf["embedding_artifacts"]["cf_emb"]["path"]],
                ["row index", cf["embedding_artifacts"]["row_index"]["path"]],
                ["item order", cf["embedding_artifacts"]["item_order"]["path"]],
                ["SID version dir", cf["sid_artifacts"]["version_dir"]],
                ["generation report", cf["sid_artifacts"]["generation_report"]["path"]],
                ["CF checkpoint", cf["generator_checkpoint"]["checkpoint_dir"]],
            ]
        )
    )
    lines.append("")
    lines.append("Key facts:")
    for fact in cf["key_facts"]:
        lines.append(f"- {fact}")
    lines.append("")

    lines.append("## 2. Split Leakage Audit")
    leak = report["split_leakage_audit"]
    for reason in leak["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.extend(
        markdown_table(
            [
                ["check", "value"],
                ["classification", leak["classification"]],
                ["train rows", leak["checks"].get("raw_train_rows")],
                ["report train rows", leak["checks"].get("cf_report_train_rows")],
                ["valid/test-only items", leak["checks"].get("validtest_only_items")],
                ["valid/test-only nonzero CF rows", leak["checks"].get("validtest_only_nonzero_cf_rows")],
                ["universe-not-train items", leak["checks"].get("universe_not_train_items")],
                ["universe-not-train nonzero CF rows", leak["checks"].get("universe_not_train_nonzero_cf_rows")],
            ]
        )
    )
    lines.append("")

    lines.append("## 3. SASRec Code Inventory")
    inv = report["sasrec_code_inventory"]
    for key, value in inv["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.extend(
        markdown_table(
            [["file", "class", "trainer", "test read", "export"]]
            + [
                [
                    item["path"],
                    item["has_class_SASRec"],
                    item["has_main"],
                    item["reads_test_dir"],
                    item["has_export_item_embedding"],
                ]
                for item in inv["files"]
            ]
        )
    )
    lines.append("")

    lines.append("## 4. Reusable Components")
    for item in report["reusable_components"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## 5. Missing Components")
    for item in report["missing_components"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 6. Proposed SASRec Builder Contract")
    contract = report["proposed_sasrec_builder_contract"]
    lines.append(f"- sequence source: `{contract['sequence_schema']['source']}`")
    lines.append(f"- internal index: `{contract['item_vocabulary']['internal_index_rule']}`")
    lines.append(f"- padding id: `{contract['item_vocabulary']['padding_id']}`")
    lines.append(f"- tensor source: `{contract['export']['tensor_source']}`")
    lines.append(f"- normalization: `{contract['export']['postprocess']}`")
    lines.append("")

    lines.append("## 7. Item-ID Alignment Contract")
    for item in contract["alignment_checks"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 8. Embedding Export Contract")
    for item in contract["export"]["required_outputs"]:
        lines.append(f"- `{item}`")
    lines.append("")

    lines.append("## 9. P3 Artifact Tree")
    for item in report["p3_artifact_tree"]:
        if item.startswith("  "):
            lines.append(f"  - `{item.strip()}`")
        else:
            lines.append(f"- `{item}`")
    lines.append("")

    lines.append("## 10. Fair Comparison Protocol")
    proto = report["fair_comparison_protocol"]
    lines.append(f"- question: {proto['research_question']}")
    lines.append("- fixed controls:")
    for key, value in proto["fixed_controls"].items():
        lines.append(f"  - {key}: `{value}`")
    lines.append("- baseline chain: " + " -> ".join(proto["baseline_chain"]))
    lines.append("- new chain: " + " -> ".join(proto["new_chain"]))
    lines.append("")

    lines.append("## 11. P3-1 Minimal Implementation Plan")
    for idx, item in enumerate(report["p3_1_minimal_implementation_plan"], start=1):
        lines.append(f"{idx}. {item}")
    lines.append("")
    return "\n".join(lines)


def write_checks_csv(path: Path, checks: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["check", "value"])
        for key in sorted(checks):
            writer.writerow([key, checks[key]])


def main() -> None:
    args = parse_args()
    repo = args.repo_root.resolve()
    category = args.category
    stage7_root = args.stage7_root
    out_dir = args.out_dir or stage7_root / "p3_behavior_audit"

    raw_paths = {
        "train": repo / "data/Amazon/train" / f"{category}_5_2016-10-2018-11.csv",
        "valid": repo / "data/Amazon/valid" / f"{category}_5_2016-10-2018-11.csv",
        "test": repo / "data/Amazon/test" / f"{category}_5_2016-10-2018-11.csv",
    }
    raw_stats: dict[str, Any] = {}
    item_sets: dict[str, set[str]] = {}
    target_sets: dict[str, set[str]] = {}
    history_sets: dict[str, set[str]] = {}
    for split, path in raw_paths.items():
        if path.exists():
            stats, all_items, target_items, history_items = csv_item_stats(path)
        else:
            stats, all_items, target_items, history_items = (
                {"path": path.as_posix(), "exists": False},
                set(),
                set(),
                set(),
            )
        raw_stats[split] = stats
        item_sets[split] = all_items
        target_sets[split] = target_items
        history_sets[split] = history_items

    cs_root = repo / "data/Amazon/cs_embeddings" / category
    cf_emb = cs_root / f"{category}.cf_emb.npy"
    cf_mask_path = cs_root / f"{category}.cf_available_mask.json"
    item_order_path = cs_root / f"{category}.item_order.json"
    row_index_path = cs_root / f"{category}.row_index.json"
    cs_report_path = cs_root / f"{category}.cs_embedding_report.json"
    cs_report = read_json(cs_report_path, default={}) or {}
    item_order = read_json(item_order_path, default=[]) or []
    row_index = read_json(row_index_path, default={}) or {}
    cf_mask = read_json(cf_mask_path, default={}) or {}

    item_order_set = set(str(item) for item in item_order)
    row_index_set = set(str(item) for item in row_index)
    row_index_exact = bool(item_order) and all(
        str(item_order[idx]) in row_index and int(row_index[str(item_order[idx])]) == idx
        for idx in range(len(item_order))
    )

    sid_dir = repo / "data/Amazon/sid_versions" / args.cf_version / category
    gen_report_path = sid_dir / "reports/generation_report.json"
    gen_report = read_json(gen_report_path, default={}) or {}
    rewrite_reports = {
        split: read_json(
            sid_dir / "reports" / f"rewrite_{split}_report.json",
            default={},
        )
        or {}
        for split in ["train", "valid", "test"]
    }
    item2sid_path = sid_dir / "item2sid.json"
    item2sid = read_json(item2sid_path, default={}) or {}

    cf_meta = load_numpy_metadata(cf_emb)
    cf_norms = load_numpy_row_norms(cf_emb)
    norm_by_item: dict[str, float] = {}
    if cf_norms is not None and item_order and len(cf_norms) == len(item_order):
        norm_by_item = {str(item_id): cf_norms[idx] for idx, item_id in enumerate(item_order)}

    validtest_items = item_sets["valid"] | item_sets["test"]
    validtest_only = validtest_items - item_sets["train"]
    universe_not_train = item_order_set - item_sets["train"]
    validtest_only_nonzero = {
        item for item in validtest_only if norm_by_item.get(item, 0.0) > 1e-12
    }
    universe_not_train_nonzero = {
        item for item in universe_not_train if norm_by_item.get(item, 0.0) > 1e-12
    }

    build_cs_path = repo / "build_cs_embeddings.py"
    build_cs_text = read_text(build_cs_path) if build_cs_path.exists() else ""
    checks: dict[str, Any] = {
        "raw_train_rows": raw_stats["train"].get("rows"),
        "raw_valid_rows": raw_stats["valid"].get("rows"),
        "raw_test_rows": raw_stats["test"].get("rows"),
        "cf_report_train_rows": cs_report.get("train_rows"),
        "cf_report_train_rows_match_raw_train": cs_report.get("train_rows")
        == raw_stats["train"].get("rows"),
        "cf_report_no_valid_test_used": cs_report.get("no_valid_test_used"),
        "source_builder_uses_entry_train_csv": 'entry["train_csv"]' in build_cs_text,
        "source_builder_has_no_valid_test_cli_inputs": "valid_csv" not in build_cs_text
        and "test_csv" not in build_cs_text,
        "item_order_len": len(item_order) if isinstance(item_order, list) else None,
        "row_index_len": len(row_index) if isinstance(row_index, dict) else None,
        "item2sid_len": len(item2sid) if isinstance(item2sid, dict) else None,
        "row_index_exact_item_order": row_index_exact,
        "item_order_row_index_key_match": item_order_set == row_index_set,
        "item_order_item2sid_key_match": item_order_set == set(str(k) for k in item2sid),
        "train_unique_items": len(item_sets["train"]),
        "valid_unique_items": len(item_sets["valid"]),
        "test_unique_items": len(item_sets["test"]),
        "valid_not_train_items": len(item_sets["valid"] - item_sets["train"]),
        "test_not_train_items": len(item_sets["test"] - item_sets["train"]),
        "validtest_only_items": len(validtest_only),
        "validtest_only_nonzero_cf_rows": len(validtest_only_nonzero),
        "universe_not_train_items": len(universe_not_train),
        "universe_not_train_nonzero_cf_rows": len(universe_not_train_nonzero),
        "cf_available_mask_len": len(cf_mask) if isinstance(cf_mask, dict) else None,
        "cf_available_true": sum(1 for v in cf_mask.values() if v) if isinstance(cf_mask, dict) else None,
        "cf_available_false": sum(1 for v in cf_mask.values() if not v) if isinstance(cf_mask, dict) else None,
    }
    classification, reasons = split_leakage_classification(checks)

    text_ckpt = (
        repo
        / "outputs/sft_sidonly_Industrial_and_Scientific_text_mbk_k512_dedup_sample30000_ep1_noearly/final_checkpoint"
    )
    cf_ckpt = (
        repo
        / "outputs/sft_sidonly_Industrial_and_Scientific_cf_k512_dedup_sample30000_ep1_noearly/final_checkpoint"
    )

    current_cf = {
        "version": args.cf_version,
        "raw_interaction_files": raw_stats,
        "user_sequence_construction": {
            "script": "build_cs_embeddings.py",
            "function": "load_train_interactions",
            "source_columns": ["history_item_id", "item_id"],
            "window_size": cs_report.get("window_size"),
            "rule": "context = parsed history_item_id plus target item_id, deduplicated per row as set of row ids",
            "pairs": "all unordered item pairs in each row context are counted symmetrically",
        },
        "cooccurrence_ppmi_svd": {
            "script": "build_cs_embeddings.py",
            "cooccurrence": {
                "min_cooccur": cs_report.get("min_cooccur"),
                "nnz_before_filter": cs_report.get("cooccurrence_nnz_before_filter"),
                "nnz_after_filter": cs_report.get("cooccurrence_nnz_after_filter"),
            },
            "ppmi": {
                "formula": "max(log(count * total / (row_sum * col_sum)), 0)",
                "ppmi_nnz": cs_report.get("ppmi_nnz"),
            },
            "svd": {
                "implementation": "sklearn.decomposition.TruncatedSVD",
                "cf_dim": cs_report.get("cf_dim"),
                "seed": 42,
                "explained_variance_ratio_sum": cs_report.get(
                    "svd_explained_variance_ratio_sum"
                ),
            },
        },
        "embedding_artifacts": {
            "cf_emb": cf_meta,
            "cf_available_mask": file_status(cf_mask_path),
            "item_order": file_status(item_order_path),
            "row_index": file_status(row_index_path),
            "cs_embedding_report": file_status(cs_report_path),
            "alignment": {
                "row_index_exact_item_order": row_index_exact,
                "item_order_row_index_key_match": item_order_set == row_index_set,
                "item_order_item2sid_key_match": item_order_set
                == set(str(k) for k in item2sid),
                "item_order_first_last": (
                    [str(item_order[0]), str(item_order[-1])] if item_order else []
                ),
            },
            "normalization": {
                "cf_emb_saved_raw_svd": "not L2-normalized before np.save in build_cs_embeddings.py",
                "cf_emb_for_cs_fusion": "L2-normalized in memory before weighted text+CF fusion",
                "rqkmeans_input": "L2-normalized inside run_rqkmeans_with_emb.py",
            },
        },
        "sid_artifacts": {
            "version_dir": sid_dir.as_posix(),
            "generation_report": file_status(gen_report_path),
            "method": gen_report.get("method"),
            "emb_path": gen_report.get("emb_path"),
            "emb_shape": gen_report.get("emb_shape"),
            "emb_dtype": gen_report.get("emb_dtype"),
            "num_levels": gen_report.get("num_levels"),
            "codebook_size": gen_report.get("codebook_size"),
            "kmeans": gen_report.get("kmeans"),
            "dedup_mode": gen_report.get("dedup_mode"),
            "pre_dedup": gen_report.get("pre_dedup"),
            "post_dedup": gen_report.get("post_dedup"),
            "dedup": gen_report.get("dedup"),
            "level_unique_codes": gen_report.get("level_unique_codes"),
            "outputs": gen_report.get("output_paths"),
        },
        "csv_rewrite": {
            split: {
                "report": file_status(sid_dir / "reports" / f"rewrite_{split}_report.json"),
                "summary": rewrite_reports.get(split, {}),
                "output_csv": (sid_dir / f"{split}.csv").as_posix(),
            }
            for split in ["train", "valid", "test"]
        },
        "generator_checkpoint": probe_checkpoint(cf_ckpt),
        "generator_training": {
            "stage7_expected_checkpoint": cf_ckpt.as_posix(),
            "entrypoint_family": [
                "scripts/run_sid_30k_validation.sh",
                "scripts/tmp_run_sidonly_10k_compare.sh",
                "scripts/run_sft_smoke_sid_version.sh",
                "sft.py",
            ],
            "stage7_fixed_constants": {
                "sample_size": 30000,
                "num_epochs": 1,
                "run_label": "noearly",
                "behavior_sid_version": args.cf_version,
            },
            "noearly_contract_from_scripts": {
                "EARLY_STOPPING_PATIENCE": 0,
                "LOAD_BEST_MODEL_AT_END": False,
                "SAVE_DURING_TRAINING_default": False,
                "source": "scripts/run_sid_30k_validation.sh lines 95-106 and scripts/tmp_run_sidonly_10k_compare.sh lines 151-161",
            },
            "run_sft_smoke_defaults": {
                "base_model_default": "/root/autodl-tmp/models/Qwen2.5-0.5B",
                "cutoff_len": 512,
                "batch_size": 64,
                "micro_batch_size": 4,
                "learning_rate": 0.0003,
                "bf16": True,
                "sft_task_mode": "sid_only",
            },
            "sft_py_dataset": {
                "train_dataset": "SidSFTDataset(train_file=..., sample=..., seed=42)",
                "eval_dataset": "SidSFTDataset(eval_file=..., sample=..., seed=42)",
                "input_columns": ["history_item_sid", "item_sid"],
                "save": "final_checkpoint via save_pretrained plus tokenizer.save_pretrained",
            },
            "local_note": (
                "The local WSL copy may not include outputs/. AutoDL should show the actual "
                "final_checkpoint files and optional trainer_state.json in generator_checkpoint."
            ),
        },
        "text_checkpoint_for_fixed_branch": probe_checkpoint(text_ckpt),
        "key_facts": [
            "The current CF embedding is PPMI/SVD over train CSV row contexts, not SASRec/LightGCN.",
            "item_order is the canonical text embedding row order: item_id strings 0..N-1.",
            "cf_k512_dedup is generated from cf_emb.npy with 3 residual MiniBatchKMeans levels, codebook_size=512, seed=42.",
            "The _dedup suffix appends <d_i> only for collided 3-token SID buckets.",
            "train/valid/test CSV files are rewritten by item_id -> SID lookup; rewriting is not model evaluation.",
            "The CF generator checkpoint is expected under outputs/sft_sidonly_Industrial_and_Scientific_cf_k512_dedup_sample30000_ep1_noearly/final_checkpoint.",
            "The CF generator training family is run_sft_smoke_sid_version.sh -> sft.py with SidSFTDataset, sid_only, sample=30000, epochs=1, noearly.",
        ],
    }

    sasrec_inventory = collect_sasrec_inventory(repo)
    reusable = [
        "SASRec model skeleton in sasrec.py.",
        "MultiHeadAttention and PositionwiseFeedForward in SASRecModules_ori.py.",
        "Existing raw train/valid CSV sequence columns history_item_id and item_id.",
        "Existing canonical item_order/row_index from cs_embeddings.",
        "Existing run_rqkmeans_with_emb.py can consume any 2D item embedding matrix.",
        "Existing rewrite_sid_csv.py can rewrite CSVs once item2sid.json exists.",
        "Existing Stage 7 valid exact candidate/fusion/ranker protocol can be reused after a new CF generator exists.",
    ]
    missing = [
        "A P3-compliant train-only SASRec dataset builder with explicit split boundary.",
        "A clean SASRec trainer that never reads test during training/model selection.",
        "A stable internal item id mapping with pad id 0 and original item_id provenance.",
        "A tied item embedding/output scoring contract suitable for exporting item_emb.weight.",
        "An exporter for sasrec_item_emb.npy, item_id_order.json, row_index.json, config, checkpoint provenance, and embedding manifest.",
        "Embedding diagnostics and alignment checks for SASRec rows vs original item_id.",
        "A SASRec-SID generation runner and valid-only comparison runner that references the frozen P2 ranker.",
    ]

    report: dict[str, Any] = {
        "audit_type": "P3-0 Behavior Representation Upgrade Audit",
        "category": category,
        "split": args.split,
        "candidate_mode": args.candidate_mode,
        "read_only_boundaries": {
            "no_sasrec_training": True,
            "no_cf_embedding_rebuild": True,
            "no_sid_regeneration": True,
            "no_gpu_inference": True,
            "no_test_evaluation": True,
            "test_csv_read_only_for_provenance_overlap": True,
            "does_not_modify_p2_frozen_manifest": True,
        },
        "current_cf_baseline_provenance": current_cf,
        "split_leakage_audit": {
            "classification": classification,
            "reasons": reasons,
            "checks": checks,
            "samples": {
                "valid_not_train": safe_sample(item_sets["valid"] - item_sets["train"]),
                "test_not_train": safe_sample(item_sets["test"] - item_sets["train"]),
                "validtest_only_nonzero_cf_rows": safe_sample(validtest_only_nonzero),
                "universe_not_train_nonzero_cf_rows": safe_sample(universe_not_train_nonzero),
            },
            "if_leakage_then_minimal_clean_rebuild_plan": [
                "Rebuild CF embedding from raw train CSV only with build_cs_embeddings.py.",
                "Write a manifest that records exact train_csv path, row counts, git revision, and input hashes.",
                "Regenerate cf_k512_dedup with run_rqkmeans_with_emb.py using the rebuilt train-only cf_emb.npy.",
                "Rewrite train/valid CSVs for validation protocol; do not evaluate test.",
            ]
            if classification != "CLEAN_TRAIN_ONLY"
            else [],
        },
        "sasrec_code_inventory": sasrec_inventory,
        "reusable_components": reusable,
        "missing_components": missing,
        "proposed_sasrec_builder_contract": build_contract(category),
        "item_id_alignment_contract": build_contract(category)["item_vocabulary"],
        "embedding_export_contract": build_contract(category)["export"],
        "p3_artifact_tree": build_artifact_tree(category, args.split),
        "fair_comparison_protocol": build_fair_protocol(
            category, args.split, args.candidate_mode
        ),
        "p3_1_minimal_implementation_plan": build_p3_1_plan(category),
    }

    md = render_markdown(report)
    if args.write:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "p3_behavior_audit_report.json", report)
        (out_dir / "p3_behavior_audit_report.md").write_text(md, encoding="utf-8")
        write_checks_csv(out_dir / "p3_behavior_audit_checks.csv", checks)

    print("P3-0 behavior representation audit completed.")
    print(f"classification={classification}")
    print(f"report_json={(out_dir / 'p3_behavior_audit_report.json').as_posix()}")
    print(f"report_md={(out_dir / 'p3_behavior_audit_report.md').as_posix()}")


if __name__ == "__main__":
    main()
