"""Non-overwriting run directories and auditable artifact lifecycle."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .config import ResolvedConfig


COMMON_ARTIFACTS = (
    "config.resolved.json",
    "command.txt",
    "command_plan.json",
    "environment.json",
    "manifest.json",
    "status.json",
    "run.log",
    "cardinality_per_sample.csv",
    "cardinality_summary.json",
)
FULL_ARTIFACTS = COMMON_ARTIFACTS + (
    "candidates.jsonl",
    "rankings.jsonl",
    "per_sample_quality.csv",
    "per_sample_latency.csv",
    "quality_summary.json",
    "latency_summary.json",
    "resource_summary.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in name).strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe}-{stamp}"


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    _atomic_json(path, payload)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    fields = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=project_root, check=True, text=True, capture_output=True
        ).stdout.strip()

    porcelain = run("status", "--porcelain")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(porcelain),
        "status_porcelain": porcelain.splitlines(),
    }


def environment_payload() -> dict[str, Any]:
    try:
        import torch

        torch_version: str | None = torch.__version__
        cuda_available: bool | str = torch.cuda.is_available()
        cuda_version: str | None = torch.version.cuda
    except Exception:
        torch_version = None
        cuda_available = "not_applicable"
        cuda_version = None
    return {
        "captured_at_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_version,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
    }


class ArtifactRun:
    def __init__(
        self,
        config: ResolvedConfig,
        run_id: str | None,
        run_mode: str,
        command: str,
        command_plan: dict[str, Any],
        formal_guard: dict[str, Any] | None = None,
    ) -> None:
        if run_id and (Path(run_id).name != run_id or run_id in {".", ".."}):
            raise ValueError("run_id must be a single safe path component")
        self.config = config
        self.run_mode = run_mode
        self.run_id = run_id or default_run_id(config.data["experiment"]["name"])
        experiment = config.data["experiment"]
        self.run_dir = config.output_root / experiment["split"] / experiment["dataset"] / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.started_at = utc_now()
        self.completed_files: list[str] = []
        self.command = command
        self.plan = command_plan
        self.formal_guard = formal_guard
        self._write_initial()

    def path(self, name: str) -> Path:
        return self.run_dir / name

    def _remember(self, name: str) -> None:
        if name not in self.completed_files:
            self.completed_files.append(name)

    def _write_initial(self) -> None:
        write_json(self.path("config.resolved.json"), self.config.data)
        self._remember("config.resolved.json")
        self.path("command.txt").write_text(self.command.rstrip() + "\n", encoding="utf-8")
        self._remember("command.txt")
        write_json(self.path("command_plan.json"), self.plan)
        self._remember("command_plan.json")
        write_json(self.path("environment.json"), environment_payload())
        self._remember("environment.json")
        self.path("run.log").write_text(f"{self.started_at} started {self.run_mode}\n", encoding="utf-8")
        self._remember("run.log")
        self.write_status("started")
        self.write_manifest("started", sample_count=None)

    def append_log(self, message: str) -> None:
        with self.path("run.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now()} {message}\n")

    def write_status(
        self,
        status: str,
        *,
        failed_stage: str | None = None,
        error: BaseException | None = None,
        resumable: bool = False,
    ) -> None:
        payload = {
            "schema": "course_run_status.v1",
            "run_id": self.run_id,
            "status": status,
            "started_at_utc": self.started_at,
            "updated_at_utc": utc_now(),
            "failed_stage": failed_stage,
            "error_type": type(error).__name__ if error else None,
            "error_message": str(error) if error else None,
            "completed_artifacts": sorted(self.completed_files),
            "resume_possible": resumable,
        }
        write_json(self.path("status.json"), payload)
        self._remember("status.json")

    def write_manifest(self, status: str, sample_count: int | None, extra: dict[str, Any] | None = None) -> None:
        data = self.config.data
        inputs = data["inputs"]
        fusion = data["fusion"]
        rerank = data["rerank"]
        state = git_state(self.config.project_root)
        file_metadata: dict[str, Any] = {}
        for key in ["cf_prediction", "sasrec_prediction", "valid_split_manifest"]:
            path = Path(inputs[key])
            if path.is_file():
                stat = path.stat()
                file_metadata[key] = {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        payload = {
            "schema": "course_system_manifest.v1",
            "run_id": self.run_id,
            "run_mode": self.run_mode,
            "formal": bool(data.get("execution", {}).get("formal", False)),
            "branch": state["branch"],
            "head": state["head"],
            "dirty_state": state["dirty"],
            "dataset": data["experiment"]["dataset"],
            "split": data["experiment"]["split"],
            "quality_split": data["evaluation"]["quality_split"],
            "sample_count": sample_count,
            "valid_split_manifest": inputs["valid_split_manifest"],
            "inputs": file_metadata,
            "retrieval_streams": inputs["streams"],
            "frozen_fusion_config": {
                "method": fusion["method"],
                "lambda_sasrec": fusion["lambda_sasrec"],
                "source_bonus": fusion["source_bonus"],
            },
            "frozen_ranker_config": rerank["frozen_config"],
            "frozen_ranker_model": rerank["frozen_model"],
            "candidate_mode": data["candidate"]["mode"],
            "candidate_budget_position": data["budget"]["position"],
            "budgets": data["budget"]["values"],
            "top_n": data["budget"]["top_n"],
            "profiling_scope": data["profiling"]["scope"],
            "output_files": sorted(self.completed_files),
            "completion_status": status,
            "test_guard": data.get("test_guard"),
            "formal_execution_guard": self.formal_guard,
        }
        if extra:
            payload.update(extra)
        write_json(self.path("manifest.json"), payload)
        self._remember("manifest.json")

    def complete(self, sample_count: int, extra_manifest: dict[str, Any] | None = None) -> None:
        self.append_log("completed")
        self.write_manifest("completed", sample_count, extra_manifest)
        self.write_status("completed")

    def fail(self, stage: str, error: BaseException, sample_count: int | None = None) -> None:
        self.append_log(f"failed stage={stage} type={type(error).__name__} message={error}")
        self.write_manifest("failed", sample_count, {"failed_stage": stage})
        self.write_status("failed", failed_stage=stage, error=error, resumable=False)

    def write_json(self, name: str, payload: Any) -> None:
        write_json(self.path(name), payload)
        self._remember(name)

    def write_jsonl(self, name: str, rows: Iterable[dict[str, Any]]) -> None:
        write_jsonl(self.path(name), rows)
        self._remember(name)

    def write_csv(self, name: str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
        write_csv(self.path(name), rows, fieldnames)
        self._remember(name)
