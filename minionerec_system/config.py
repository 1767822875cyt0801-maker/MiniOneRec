"""Configuration loading and frozen course-contract validation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schemas import BudgetValue


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_DATASET = "Industrial_and_Scientific"
FROZEN_STREAMS = ["cf_sid", "sasrec_sid"]
FROZEN_LAMBDA_SASREC = 0.75
FROZEN_SOURCE_BONUS = 0.01
FROZEN_TOP_N = 20
FROZEN_FORMAL_BUDGETS: list[BudgetValue] = [20, 50, 75, 90, "all"]
FROZEN_BUDGET_POSITION = "post_fusion_pre_rerank"
FROZEN_PROFILING_SCOPE = "frozen_prediction_downstream"
FROZEN_FORMAL_BRANCH = "course-system-bounded-rerank-v1"
FROZEN_FORMAL_SAMPLE_COUNT = 1360
FROZEN_COURSE_PATHS = (
    "minionerec_system",
    "configs/course_system",
    "scripts/run_course_system.py",
    "scripts/check_course_system_artifacts.py",
    "tests/course_system",
    "docs/优化五/course_system_skeleton_implementation_report.md",
    "docs/优化五/course_system_cardinality_closeout_v1.md",
)
FORMAL_IMMUTABLE_FIELDS = {
    "cf_prediction": ("inputs", "cf_prediction"),
    "sasrec_prediction": ("inputs", "sasrec_prediction"),
    "cf_sid_mapping": ("inputs", "cf_sid_mapping"),
    "sasrec_sid_mapping": ("inputs", "sasrec_sid_mapping"),
    "valid_data": ("inputs", "valid_data"),
    "frozen_config": ("rerank", "frozen_config"),
    "frozen_model": ("rerank", "frozen_model"),
}
FROZEN_IMMUTABLE_IDENTITIES = {
    "cf_prediction": {
        "path": "incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036/results/s4_sasrec_sid_valid/Industrial_and_Scientific/cf_k512_dedup/seed42/formal/valid/full_valid/generation/predictions.json",
        "size_bytes": 10295761,
        "sha256": "40f9bd734da5e555cb3c5788bc3b462b08fa2413b61fdc82a57052afa334a903",
    },
    "sasrec_prediction": {
        "path": "incoming/s4_formal_full_valid/s4_formal_full_valid_closeout_20260714_181036/results/s4_sasrec_sid_valid/Industrial_and_Scientific/sasrec_v3_k512_dedup/seed42/formal/valid/full_valid/generation/predictions.json",
        "size_bytes": 9785754,
        "sha256": "1db9fbefa70a8b6064544c10bbdacf4aa102757036500410af3cbb6274493210",
    },
    "cf_sid_mapping": {
        "path": "data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/sid2items.json",
        "size_bytes": 167919,
        "sha256": "7f4da9544bb8da8b7df38c19fe7e201f7186c308c28bf9db40ef4bac7dbeaa79",
    },
    "sasrec_sid_mapping": {
        "path": "data/Amazon/sid_versions/sasrec_v3_k512_dedup/Industrial_and_Scientific/sid2items.json",
        "size_bytes": 162696,
        "sha256": "81ca537d6a838d762d0647066d617c1540c0b5fbf67cc7f361c152d3db8fa392",
    },
    "valid_data": {
        "path": "data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/valid.csv",
        "size_bytes": 3578448,
        "sha256": "5234a104809d20585aa9ba91dfdf98d348d9fe6429a131a737a89d77c5780e9e",
    },
    "frozen_config": {
        "path": "configs/s5_auxiliary_fusion/frozen_release_config.json",
        "size_bytes": 8566,
        "sha256": "fe53b9ae6b5d12d98a3ccd6f0a6526e3a00bed680f595f72c5521f0e928a6e6a",
    },
    "frozen_model": {
        "path": "results/stage7_validation_protocol/valid/Industrial_and_Scientific/p2_history_ranker/model.json",
        "size_bytes": 4579,
        "sha256": "6a3bca7cc714d295572ce4a3953d60719a1c72757895835554a83d747dca65ce",
    },
}
REQUIRED_METRICS = {"candidate_recall", "hr@10", "hr@20", "ndcg@10", "ndcg@20"}
UNRESOLVED_RE = re.compile(r"(__UNRESOLVED__|\$\{|<resolved|<unresolved|\bTODO\b)", re.IGNORECASE)


class ConfigError(ValueError):
    """Raised when a configuration violates the frozen course contract."""


@dataclass(frozen=True)
class ResolvedConfig:
    source_path: Path
    project_root: Path
    data: dict[str, Any]

    @property
    def budgets(self) -> list[BudgetValue]:
        return list(self.data["budget"]["values"])

    @property
    def is_formal(self) -> bool:
        return bool(self.data.get("execution", {}).get("formal", False))

    @property
    def output_root(self) -> Path:
        return Path(self.data["experiment"]["output_root"])


PATH_FIELDS = (
    ("inputs", "cf_prediction"),
    ("inputs", "sasrec_prediction"),
    ("inputs", "cf_sid_mapping"),
    ("inputs", "sasrec_sid_mapping"),
    ("inputs", "valid_data"),
    ("inputs", "valid_split_manifest"),
    ("rerank", "frozen_config"),
    ("rerank", "frozen_model"),
    ("rerank", "train_data"),
    ("rerank", "item_embedding"),
    ("rerank", "row_index"),
)


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ConfigError(f"Course config must contain a mapping: {path}")
    return payload


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"Missing required config field: {context}.{key}")
    return mapping[key]


def _resolve_path(value: Any, project_root: Path, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a non-empty path string")
    path = Path(value)
    return str(path.resolve() if path.is_absolute() else (project_root / path).resolve())


def _contains_unresolved(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_unresolved(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_unresolved(item) for item in value)
    return isinstance(value, str) and bool(UNRESOLVED_RE.search(value))


def _path_uses_test_data(value: str) -> bool:
    tokens = [token for part in Path(value).parts for token in re.split(r"[._-]+", part.lower())]
    return "test" in tokens or "tests" in tokens


def _parse_budgets(values: Any, top_n: int) -> list[BudgetValue]:
    if not isinstance(values, list) or not values:
        raise ConfigError("budget.values must be a non-empty list")
    parsed: list[BudgetValue] = []
    for value in values:
        if isinstance(value, str) and value.lower() == "all":
            parsed.append("all")
        elif isinstance(value, int) and not isinstance(value, bool):
            if value < top_n:
                raise ConfigError(f"Candidate budget K={value} is smaller than Top-{top_n}")
            parsed.append(value)
        else:
            raise ConfigError(f"Unsupported candidate budget: {value!r}")
    if len(set(parsed)) != len(parsed):
        raise ConfigError("budget.values contains duplicates")
    if "all" not in parsed:
        raise ConfigError("budget.values must include the unbounded 'all' baseline")
    return parsed


def _validate_frozen_references(config: dict[str, Any]) -> None:
    frozen_path = Path(config["rerank"]["frozen_config"])
    model_path = Path(config["rerank"]["frozen_model"])
    if not frozen_path.is_file():
        raise ConfigError(f"Frozen S5 config is missing: {frozen_path}")
    if not model_path.is_file():
        raise ConfigError(f"Frozen P2 model is missing: {model_path}")
    with frozen_path.open("r", encoding="utf-8") as handle:
        frozen = json.load(handle)
    if frozen.get("policy") != "source_aware_rrf":
        raise ConfigError("Frozen S5 config does not use source_aware_rrf")
    if not math.isclose(float(frozen.get("lambda_sasrec", -1)), FROZEN_LAMBDA_SASREC):
        raise ConfigError("Frozen S5 lambda_sasrec is not 0.75")
    if not math.isclose(float(frozen.get("source_bonus", -1)), FROZEN_SOURCE_BONUS):
        raise ConfigError("Frozen S5 source_bonus is not 0.01")
    if frozen.get("ranker_compatibility_mode") != "source_independent_projection":
        raise ConfigError("Frozen S5 ranker compatibility mode drifted")
    # plan-only validates the frozen reference without reading model weights.
    # The bounded dry-run schema check and execution load the P2 JSON later.


def validate_course_config(config: dict[str, Any]) -> None:
    experiment = _require(config, "experiment", "config")
    inputs = _require(config, "inputs", "config")
    candidate = _require(config, "candidate", "config")
    fusion = _require(config, "fusion", "config")
    budget = _require(config, "budget", "config")
    rerank = _require(config, "rerank", "config")
    evaluation = _require(config, "evaluation", "config")
    profiling = _require(config, "profiling", "config")
    artifacts = _require(config, "artifacts", "config")

    if experiment.get("dataset") != FROZEN_DATASET:
        raise ConfigError(f"dataset must be {FROZEN_DATASET}")
    if experiment.get("overwrite") is not False:
        raise ConfigError("experiment.overwrite must be false")
    split = str(experiment.get("split", ""))
    if split not in {"smoke", "valid", "test"}:
        raise ConfigError("experiment.split must be smoke, valid, or guarded test")
    if inputs.get("mode") != "frozen_predictions":
        raise ConfigError("inputs.mode must be frozen_predictions")
    if inputs.get("streams") != FROZEN_STREAMS:
        raise ConfigError("retrieval streams must be exactly CF-SID + SASRec-SID")
    if candidate.get("mode") != "exact":
        raise ConfigError("candidate.mode must be exact")
    if candidate.get("stable_dedup") is not True:
        raise ConfigError("candidate.stable_dedup must be true")
    if fusion.get("method") != "source_aware_rrf":
        raise ConfigError("fusion.method must be source_aware_rrf")
    if not math.isclose(float(fusion.get("lambda_sasrec", -1)), FROZEN_LAMBDA_SASREC):
        raise ConfigError("fusion.lambda_sasrec drifted from frozen value 0.75")
    if not math.isclose(float(fusion.get("source_bonus", -1)), FROZEN_SOURCE_BONUS):
        raise ConfigError("fusion.source_bonus drifted from frozen value 0.01")
    if int(budget.get("top_n", -1)) != FROZEN_TOP_N or int(rerank.get("top_n", -1)) != FROZEN_TOP_N:
        raise ConfigError("budget.top_n and rerank.top_n must both equal 20")
    if budget.get("position") != FROZEN_BUDGET_POSITION:
        raise ConfigError(f"budget.position must be {FROZEN_BUDGET_POSITION}")
    budget["values"] = _parse_budgets(budget.get("values"), FROZEN_TOP_N)
    if rerank.get("method") != "p2_frozen_history_aware":
        raise ConfigError("rerank.method must be p2_frozen_history_aware")
    if rerank.get("compatibility_mode") != "source_independent_projection":
        raise ConfigError("rerank.compatibility_mode must be source_independent_projection")
    if evaluation.get("quality_split") != "valid_select":
        raise ConfigError("evaluation.quality_split must be valid_select")
    if set(evaluation.get("metrics", [])) != REQUIRED_METRICS:
        raise ConfigError(f"evaluation.metrics must be exactly {sorted(REQUIRED_METRICS)}")
    if profiling.get("scope") != FROZEN_PROFILING_SCOPE:
        raise ConfigError(f"profiling.scope must be {FROZEN_PROFILING_SCOPE}")
    if profiling.get("synchronize_cuda_if_used") is not True:
        raise ConfigError("profiling.synchronize_cuda_if_used must be true")
    if int(profiling.get("warmup_samples", -1)) < 0 or int(profiling.get("repeats", 0)) < 1:
        raise ConfigError("profiling warmup_samples must be >=0 and repeats must be >=1")
    required_artifacts = {
        "save_resolved_config",
        "save_command_plan",
        "save_manifest",
        "save_cardinality_report",
        "save_per_sample_quality",
        "save_per_sample_latency",
    }
    if not all(artifacts.get(field) is True for field in required_artifacts):
        raise ConfigError("all required artifact switches must be true")
    for section, field in PATH_FIELDS:
        _require(config[section], field, section)
    if _contains_unresolved(config):
        raise ConfigError("resolved config contains an unresolved placeholder")
    if split == "test":
        guard = config.get("test_guard", {})
        if not guard.get("confirm_frozen_course_config") or not guard.get("test_result_must_not_change_parameters"):
            raise ConfigError("test split requires frozen course config and explicit one-shot test guard")
        if not guard.get("frozen_course_config"):
            raise ConfigError("test split requires test_guard.frozen_course_config")
    if bool(config.get("execution", {}).get("formal")):
        if split != "valid":
            raise ConfigError("formal course execution is valid-only; test is forbidden")
        if not inputs.get("valid_split_manifest"):
            raise ConfigError("formal valid execution requires inputs.valid_split_manifest")
        if budget["values"] != FROZEN_FORMAL_BUDGETS:
            raise ConfigError(f"formal budget.values must be exactly {FROZEN_FORMAL_BUDGETS}")
        input_path_fields = [field for section, field in PATH_FIELDS if section == "inputs"]
        if any(_path_uses_test_data(str(inputs[field])) for field in input_path_fields):
            raise ConfigError("formal course execution may not reference test inputs")
        guard = config.get("formal_guard")
        if not isinstance(guard, dict):
            raise ConfigError("formal execution requires formal_guard")
        if guard.get("required_branch") != FROZEN_FORMAL_BRANCH:
            raise ConfigError(f"formal_guard.required_branch must be {FROZEN_FORMAL_BRANCH}")
        if guard.get("required_python_major_minor") != "3.11":
            raise ConfigError("formal_guard.required_python_major_minor must be 3.11")
        if int(guard.get("expected_sample_count", -1)) != FROZEN_FORMAL_SAMPLE_COUNT:
            raise ConfigError(f"formal_guard.expected_sample_count must be {FROZEN_FORMAL_SAMPLE_COUNT}")
        if guard.get("expected_quality_split") != "valid_select":
            raise ConfigError("formal_guard.expected_quality_split must be valid_select")
        if guard.get("expected_budget_grid") != FROZEN_FORMAL_BUDGETS:
            raise ConfigError(f"formal_guard.expected_budget_grid must be exactly {FROZEN_FORMAL_BUDGETS}")
        if tuple(guard.get("course_paths", [])) != FROZEN_COURSE_PATHS:
            raise ConfigError("formal_guard.course_paths does not cover the frozen course-system scope")
        if guard.get("tracked_worktree_must_be_clean") is not True:
            raise ConfigError("formal guard must require a clean tracked worktree")
        if guard.get("index_must_be_clean") is not True:
            raise ConfigError("formal guard must require a clean index")
        immutable = guard.get("immutable_files")
        if not isinstance(immutable, dict) or set(immutable) != set(FORMAL_IMMUTABLE_FIELDS):
            raise ConfigError(f"formal_guard.immutable_files must contain exactly {sorted(FORMAL_IMMUTABLE_FIELDS)}")
        for label, spec in immutable.items():
            if not isinstance(spec, dict) or not isinstance(spec.get("path"), str):
                raise ConfigError(f"formal_guard.immutable_files.{label}.path is required")
            if not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256", ""))):
                raise ConfigError(f"formal_guard.immutable_files.{label}.sha256 must be lowercase SHA-256")
            if not isinstance(spec.get("size_bytes"), int) or spec["size_bytes"] <= 0:
                raise ConfigError(f"formal_guard.immutable_files.{label}.size_bytes must be positive")
            if spec != FROZEN_IMMUTABLE_IDENTITIES[label]:
                raise ConfigError(f"formal_guard.immutable_files.{label} drifted from the frozen identity")
    _validate_frozen_references(config)


def load_course_config(path: str | Path, project_root: Path = PROJECT_ROOT) -> ResolvedConfig:
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = (project_root / source_path).resolve()
    if not source_path.is_file():
        raise ConfigError(f"Config file does not exist: {source_path}")
    config = copy.deepcopy(_read_config(source_path))
    for section, field in PATH_FIELDS:
        if section not in config or field not in config[section]:
            raise ConfigError(f"Missing required config field: {section}.{field}")
        config[section][field] = _resolve_path(config[section][field], project_root, f"{section}.{field}")
    output_value = _require(config.get("experiment", {}), "output_root", "experiment")
    config["experiment"]["output_root"] = _resolve_path(output_value, project_root, "experiment.output_root")
    validate_course_config(config)
    return ResolvedConfig(source_path=source_path, project_root=project_root.resolve(), data=config)


def _git(project_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=project_root, check=check, text=True, capture_output=True
    )


def _under_course_scope(path: str, course_paths: list[str]) -> bool:
    normalized = Path(path).as_posix()
    return any(normalized == root or normalized.startswith(f"{root.rstrip('/')}/") for root in course_paths)


def verify_formal_git_state(
    project_root: Path,
    course_paths: list[str],
    required_branch: str | None,
) -> dict[str, Any]:
    """Require tracked/index cleanliness and HEAD coverage without rejecting unrelated untracked files."""

    root = project_root.resolve()
    branch = _git(root, "branch", "--show-current").stdout.strip()
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    errors: list[str] = []
    if required_branch and branch != required_branch:
        errors.append(f"current branch is {branch!r}, expected {required_branch!r}")
    tracked = _git(root, "diff", "--quiet", check=False)
    if tracked.returncode not in {0, 1}:
        raise ConfigError(tracked.stderr.strip() or "git diff --quiet failed")
    if tracked.returncode == 1:
        errors.append("tracked worktree is not clean")
    index = _git(root, "diff", "--cached", "--quiet", check=False)
    if index.returncode not in {0, 1}:
        raise ConfigError(index.stderr.strip() or "git diff --cached --quiet failed")
    if index.returncode == 1:
        errors.append("index is not clean")

    missing_from_head: list[str] = []
    for path in course_paths:
        probe = _git(root, "cat-file", "-e", f"HEAD:{path}", check=False)
        if probe.returncode != 0:
            missing_from_head.append(path)
    if missing_from_head:
        errors.append(f"course-system paths are not tracked by HEAD: {missing_from_head}")

    untracked = _git(root, "ls-files", "--others", "--exclude-standard").stdout.splitlines()
    course_untracked = sorted(path for path in untracked if _under_course_scope(path, course_paths))
    unrelated_untracked = sorted(path for path in untracked if not _under_course_scope(path, course_paths))
    if course_untracked:
        errors.append(f"untracked course-system paths are not covered by HEAD: {course_untracked}")
    if errors:
        raise ConfigError("formal Git guard failed: " + "; ".join(errors))
    warnings = []
    if unrelated_untracked:
        warnings.append(
            f"allowed {len(unrelated_untracked)} unrelated untracked paths; use a clean checkout for formal evidence"
        )
    return {
        "branch": branch,
        "head": head,
        "tracked_worktree_clean": True,
        "index_clean": True,
        "course_paths_tracked_by_head": True,
        "course_untracked_count": 0,
        "unrelated_untracked_count": len(unrelated_untracked),
        "warnings": warnings,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_formal_execution_guard(config: ResolvedConfig) -> dict[str, Any]:
    """Run immediately before a real formal audit/matrix execution, never during plan-only."""

    if not config.is_formal:
        raise ConfigError("formal execution guard was requested for a non-formal config")
    guard = config.data["formal_guard"]
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if version != str(guard.get("required_python_major_minor")):
        raise ConfigError(
            f"formal Python guard failed: running {version}, expected {guard.get('required_python_major_minor')}"
        )
    git_report = verify_formal_git_state(
        config.project_root,
        list(guard["course_paths"]),
        str(guard["required_branch"]),
    )
    immutable_report: dict[str, Any] = {}
    for label, (section, field) in FORMAL_IMMUTABLE_FIELDS.items():
        spec = guard["immutable_files"][label]
        expected_path = (config.project_root / spec["path"]).resolve()
        configured_path = Path(config.data[section][field]).resolve()
        if configured_path != expected_path:
            raise ConfigError(
                f"formal immutable path drift for {label}: {configured_path} != {expected_path}"
            )
        if not configured_path.is_file():
            raise ConfigError(f"formal immutable input is missing: {configured_path}")
        size = configured_path.stat().st_size
        if size != int(spec["size_bytes"]):
            raise ConfigError(f"formal immutable size drift for {label}: {size} != {spec['size_bytes']}")
        digest = _sha256(configured_path)
        if digest != spec["sha256"]:
            raise ConfigError(f"formal immutable SHA-256 drift for {label}")
        immutable_report[label] = {
            "path": str(configured_path),
            "size_bytes": size,
            "sha256": digest,
        }
    return {
        "schema": "course_formal_execution_guard.v1",
        "status": "PASS",
        "expected_sample_count": FROZEN_FORMAL_SAMPLE_COUNT,
        "expected_quality_split": "valid_select",
        "expected_budget_grid": list(FROZEN_FORMAL_BUDGETS),
        "python_major_minor": version,
        "git": git_report,
        "immutable_files": immutable_report,
    }
