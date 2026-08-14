"""Read-only checker for course-system run artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .artifacts import COMMON_ARTIFACTS, FULL_ARTIFACTS
from .config import FROZEN_FORMAL_BUDGETS, FROZEN_FORMAL_SAMPLE_COUNT
from .profiling import PROFILING_SCOPE


FORBIDDEN_LATENCY_TERMS = (
    "full llm inference latency",
    "complete online generative inference latency",
)
UNRESOLVED_RE = re.compile(r"(__UNRESOLVED__|\$\{|<resolved|<unresolved|\bTODO\b)", re.IGNORECASE)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _contains_unresolved(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_unresolved(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_unresolved(item) for item in value)
    return isinstance(value, str) and bool(UNRESOLVED_RE.search(value))


def check_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(condition: bool, name: str, detail: Any = None) -> None:
        checks.append({"name": name, "ok": bool(condition), "detail": detail})
        if not condition:
            errors.append(f"{name}: {detail}")

    check(root.is_dir(), "run directory exists", str(root))
    if not root.is_dir():
        return {"status": "FAIL", "run_dir": str(root), "checks": checks, "errors": errors}
    manifest_path = root / "manifest.json"
    status_path = root / "status.json"
    check(manifest_path.is_file(), "manifest exists", str(manifest_path))
    check(status_path.is_file(), "status exists", str(status_path))
    if not manifest_path.is_file() or not status_path.is_file():
        return {"status": "FAIL", "run_dir": str(root), "checks": checks, "errors": errors}

    manifest = _read_json(manifest_path)
    status = _read_json(status_path)
    run_mode = manifest.get("run_mode")
    required = FULL_ARTIFACTS if run_mode == "full_pipeline" else COMMON_ARTIFACTS
    for name in required:
        path = root / name
        check(path.is_file() and path.stat().st_size > 0, f"required artifact non-empty: {name}", str(path))
    check(status.get("status") == "completed", "status is completed", status.get("status"))
    check(manifest.get("completion_status") == "completed", "manifest completion status", manifest.get("completion_status"))
    check(manifest.get("profiling_scope") == PROFILING_SCOPE, "profiling scope is downstream-only", manifest.get("profiling_scope"))
    split = manifest.get("split")
    check(split in {"smoke", "valid", "test"}, "split is legal", split)
    if split == "test":
        guard = manifest.get("test_guard") or {}
        check(bool(guard.get("confirm_frozen_course_config")), "test frozen config guard", guard)
        check(bool(guard.get("test_result_must_not_change_parameters")), "test no-tuning guard", guard)
    config = _read_json(root / "config.resolved.json") if (root / "config.resolved.json").is_file() else {}
    if manifest.get("formal"):
        check(not _contains_unresolved(config), "formal config has no placeholder", None)
        check(
            manifest.get("budgets") == FROZEN_FORMAL_BUDGETS,
            "formal budget grid is frozen including K90",
            manifest.get("budgets"),
        )
        check(
            int(manifest.get("sample_count", -1)) == FROZEN_FORMAL_SAMPLE_COUNT,
            "formal sample count is 1360",
            manifest.get("sample_count"),
        )
        formal_guard = manifest.get("formal_execution_guard") or {}
        check(formal_guard.get("status") == "PASS", "formal execution guard PASS", formal_guard.get("status"))
    serialized = json.dumps({"manifest": manifest, "config": config}, ensure_ascii=False).lower()
    for term in FORBIDDEN_LATENCY_TERMS:
        check(term not in serialized, f"forbidden latency term absent: {term}", None)

    cardinality_summary_path = root / "cardinality_summary.json"
    cardinality_rows_path = root / "cardinality_per_sample.csv"
    if cardinality_summary_path.is_file() and cardinality_rows_path.is_file():
        card_summary = _read_json(cardinality_summary_path)
        card_rows = _csv_rows(cardinality_rows_path)
        check(card_summary.get("gate", {}).get("status") == "PASS", "cardinality gate PASS", card_summary.get("gate"))
        check(int(card_summary.get("sample_count", -1)) == len(card_rows), "cardinality sample count", len(card_rows))
        check(int(manifest.get("sample_count", -1)) == len(card_rows), "manifest/cardinality sample count", manifest.get("sample_count"))
        for budget in manifest.get("budgets", []):
            label = "all" if str(budget) == "all" else f"k{budget}"
            check(
                all(f"effective_count_at_{label}" in row for row in card_rows),
                f"cardinality column exists: effective_count_at_{label}",
                len(card_rows),
            )
            if str(budget) != "all":
                check(
                    all(f"truncated_at_{label}" in row for row in card_rows),
                    f"cardinality column exists: truncated_at_{label}",
                    len(card_rows),
                )
        card_ids = {row["sample_id"] for row in card_rows}
    else:
        card_ids = set()

    if run_mode == "full_pipeline" and all((root / name).is_file() for name in ["candidates.jsonl", "rankings.jsonl", "per_sample_quality.csv"]):
        candidates = _jsonl_rows(root / "candidates.jsonl")
        rankings = _jsonl_rows(root / "rankings.jsonl")
        quality = _csv_rows(root / "per_sample_quality.csv")
        candidate_ids = {str(row.get("sample_id")) for row in candidates}
        ranking_ids = {str(row.get("sample_id")) for row in rankings}
        quality_ids = {str(row.get("sample_id")) for row in quality}
        check(candidate_ids == card_ids, "candidate/cardinality sample IDs", sorted(candidate_ids ^ card_ids)[:10])
        check(ranking_ids == card_ids, "ranking/cardinality sample IDs", sorted(ranking_ids ^ card_ids)[:10])
        check(quality_ids == card_ids, "quality/cardinality sample IDs", sorted(quality_ids ^ card_ids)[:10])
        budget_values = {str(value) for value in manifest.get("budgets", [])}
        expected_pairs = {(sample_id, budget) for sample_id in card_ids for budget in budget_values}
        candidate_pairs = {(str(row.get("sample_id")), str(row.get("budget"))) for row in candidates}
        ranking_pairs = {(str(row.get("sample_id")), str(row.get("budget"))) for row in rankings}
        quality_pairs = {(str(row.get("sample_id")), str(row.get("budget"))) for row in quality}
        check(candidate_pairs == expected_pairs, "candidate sample×budget coverage", sorted(candidate_pairs ^ expected_pairs)[:10])
        check(ranking_pairs == expected_pairs, "ranking sample×budget coverage", sorted(ranking_pairs ^ expected_pairs)[:10])
        check(quality_pairs == expected_pairs, "quality sample×budget coverage", sorted(quality_pairs ^ expected_pairs)[:10])
        card_by_id = {row["sample_id"]: row for row in card_rows}
        top_n = int(manifest.get("top_n", 20))
        for row in candidates:
            budget = row.get("budget")
            count = int(row.get("candidate_count", -1))
            items = row.get("candidate_item_ids", [])
            check(count == len(items), "candidate count matches items", row.get("sample_id"))
            label = "all" if str(budget) == "all" else f"k{budget}"
            expected_count = int(card_by_id[str(row.get("sample_id"))][f"effective_count_at_{label}"])
            check(count == expected_count, "candidate/cardinality effective count", {"sample_id": row.get("sample_id"), "budget": budget})
            if str(budget) != "all":
                check(count <= int(budget), "post-budget count <= K", {"sample_id": row.get("sample_id"), "budget": budget})
            check(row.get("budget_position") == "post_fusion_pre_rerank", "budget position", row.get("sample_id"))
        for row in rankings:
            ranked = row.get("ranked_item_ids", [])
            top = row.get("top20_item_ids", [])
            candidate_count = int(row.get("candidate_count", -1))
            check(len(ranked) == candidate_count, "ranking/candidate count", row.get("sample_id"))
            check(len(top) == min(top_n, candidate_count), "Top-20 length legal", row.get("sample_id"))
            check(top == ranked[:top_n], "Top-20 is ranking prefix", row.get("sample_id"))

    return {
        "schema": "course_artifact_check.v1",
        "status": "PASS" if not errors else "FAIL",
        "run_dir": str(root),
        "run_mode": run_mode,
        "checks": checks,
        "errors": errors,
    }
