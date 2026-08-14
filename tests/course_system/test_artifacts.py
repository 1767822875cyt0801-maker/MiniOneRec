from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

from minionerec_system.artifacts import ArtifactRun
from minionerec_system.checker import check_run
from minionerec_system.config import PROJECT_ROOT, ResolvedConfig, load_course_config
from minionerec_system.pipeline import build_command_plan


def _tmp_config(tmp_path: Path) -> ResolvedConfig:
    base = load_course_config(PROJECT_ROOT / "configs/course_system/smoke_cf_sasrec_exact.yaml", PROJECT_ROOT)
    data = copy.deepcopy(base.data)
    data["experiment"]["output_root"] = str(tmp_path)
    return ResolvedConfig(base.source_path, base.project_root, data)


def test_manifest_status_started_then_completed(tmp_path: Path) -> None:
    config = _tmp_config(tmp_path)
    run = ArtifactRun(config, "lifecycle", "cardinality_audit", "synthetic command", build_command_plan(config, "cardinality_audit"))
    assert json.loads(run.path("status.json").read_text())["status"] == "started"
    run.write_csv(
        "cardinality_per_sample.csv",
        [
            {
                "sample_id": "0",
                "effective_count_at_k20": 20,
                "truncated_at_k20": True,
                "effective_count_at_k50": 50,
                "truncated_at_k50": True,
                "effective_count_at_k75": 75,
                "truncated_at_k75": True,
                "effective_count_at_k90": 90,
                "truncated_at_k90": True,
                "effective_count_at_all": 95,
            }
        ],
    )
    run.write_json("cardinality_summary.json", {"sample_count": 1, "gate": {"status": "PASS"}})
    run.complete(1, {"cardinality_gate": "PASS"})
    assert json.loads(run.path("status.json").read_text())["status"] == "completed"
    assert json.loads(run.path("manifest.json").read_text())["completion_status"] == "completed"
    report = check_run(run.run_dir)
    assert report["status"] == "PASS"
    assert any(check["name"] == "cardinality column exists: effective_count_at_k90" for check in report["checks"])


def test_checker_reports_missing_files_without_repair(tmp_path: Path) -> None:
    run_dir = tmp_path / "missing"
    run_dir.mkdir()
    before = set(run_dir.iterdir())
    report = check_run(run_dir)
    assert report["status"] == "FAIL"
    assert set(run_dir.iterdir()) == before


def test_checker_reports_complete_formal_guard_state(tmp_path: Path) -> None:
    config = _tmp_config(tmp_path)
    data = copy.deepcopy(config.data)
    data["execution"]["formal"] = True
    data["experiment"]["split"] = "valid"
    formal = ResolvedConfig(config.source_path, config.project_root, data)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=config.project_root, text=True
    ).strip()
    guard = {
        "schema": "course_formal_execution_guard.v2",
        "status": "PASS",
        "expected_course_commit": head,
        "git": {
            "head": head,
            "tracked_worktree_clean": True,
            "index_clean": True,
            "course_paths_tracked_by_head": True,
            "course_untracked_count": 0,
            "unrelated_untracked_count": 2,
            "unrelated_untracked_paths": ["incoming/prediction.json", "results/run.json"],
        },
    }
    run = ArtifactRun(
        formal,
        "formal-guard-checker",
        "cardinality_audit",
        "synthetic command",
        build_command_plan(formal, "cardinality_audit"),
        formal_guard=guard,
    )
    rows = [
        {
            "sample_id": str(idx),
            "effective_count_at_k20": 20,
            "truncated_at_k20": True,
            "effective_count_at_k50": 50,
            "truncated_at_k50": True,
            "effective_count_at_k75": 75,
            "truncated_at_k75": True,
            "effective_count_at_k90": 90,
            "truncated_at_k90": True,
            "effective_count_at_all": 95,
        }
        for idx in range(1360)
    ]
    run.write_csv("cardinality_per_sample.csv", rows)
    run.write_json("cardinality_summary.json", {"sample_count": 1360, "gate": {"status": "PASS"}})
    run.complete(1360, {"cardinality_gate": "PASS"})
    report = check_run(run.run_dir)
    assert report["status"] == "PASS"
    names = {check["name"]: check for check in report["checks"]}
    assert names["formal execution guard PASS"]["ok"] is True
    assert names["formal guard exact HEAD"]["ok"] is True
    assert names["formal guard unrelated untracked provenance count"]["ok"] is True

    manifest_path = run.path("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_execution_guard"]["status"] = "FAIL"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    failed = check_run(run.run_dir)
    assert failed["status"] == "FAIL"
    assert any("formal execution guard PASS" in error for error in failed["errors"])
