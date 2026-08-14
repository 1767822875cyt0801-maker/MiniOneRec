from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from minionerec_system.config import ConfigError, PROJECT_ROOT, load_course_config, verify_formal_git_state


BASE = PROJECT_ROOT / "configs/course_system/smoke_cf_sasrec_exact.yaml"
FORMAL = PROJECT_ROOT / "configs/course_system/valid_select_cf_sasrec_exact_kmatrix_v1.yaml"


def _payload() -> dict:
    return yaml.safe_load(BASE.read_text(encoding="utf-8"))


def _load_mutated(tmp_path: Path, mutate) -> None:
    payload = copy.deepcopy(_payload())
    mutate(payload)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    load_course_config(path, PROJECT_ROOT)


def test_config_parses_frozen_smoke_contract() -> None:
    config = load_course_config(BASE, PROJECT_ROOT)
    assert config.data["inputs"]["streams"] == ["cf_sid", "sasrec_sid"]
    assert config.budgets == [20, 50, 75, 90, "all"]
    assert config.data["profiling"]["scope"] == "frozen_prediction_downstream"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: cfg["experiment"].__setitem__("dataset", "Office_Products"),
        lambda cfg: cfg["candidate"].__setitem__("mode", "prefix"),
        lambda cfg: cfg["budget"].__setitem__("values", [10, 50, 75, "all"]),
        lambda cfg: cfg["budget"].__setitem__("position", "pre_fusion"),
        lambda cfg: cfg["fusion"].__setitem__("lambda_sasrec", 0.5),
    ],
)
def test_invalid_frozen_contract_is_rejected(tmp_path: Path, mutate) -> None:
    with pytest.raises(ConfigError):
        _load_mutated(tmp_path, mutate)


def test_formal_valid_without_split_manifest_is_rejected(tmp_path: Path) -> None:
    def mutate(payload: dict) -> None:
        payload["experiment"]["split"] = "valid"
        payload["execution"]["formal"] = True
        payload["inputs"]["valid_split_manifest"] = ""

    with pytest.raises(ConfigError, match="valid_split_manifest"):
        _load_mutated(tmp_path, mutate)


def test_formal_config_has_complete_frozen_budget_order() -> None:
    config = load_course_config(FORMAL, PROJECT_ROOT)
    assert config.budgets == [20, 50, 75, 90, "all"]
    assert config.data["evaluation"]["quality_split"] == "valid_select"


def test_formal_config_rejects_test_even_with_test_guard(tmp_path: Path) -> None:
    payload = yaml.safe_load(FORMAL.read_text(encoding="utf-8"))
    payload["experiment"]["split"] = "test"
    payload["test_guard"] = {
        "confirm_frozen_course_config": True,
        "test_result_must_not_change_parameters": True,
        "frozen_course_config": "not-used.json",
    }
    path = tmp_path / "formal-test.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError, match="valid-only; test is forbidden"):
        load_course_config(path, PROJECT_ROOT)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("candidate", "mode", "prefix"),
        ("fusion", "lambda_sasrec", 0.5),
        ("rerank", "compatibility_mode", "source_specific"),
        ("budget", "values", [20, 50, 75, "all"]),
    ],
)
def test_formal_config_rejects_parameter_drift(tmp_path: Path, section: str, field: str, value) -> None:
    payload = yaml.safe_load(FORMAL.read_text(encoding="utf-8"))
    payload[section][field] = value
    path = tmp_path / "drift.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_course_config(path, PROJECT_ROOT)


def test_formal_git_guard_rejects_untracked_course_code(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "anchor.txt").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "anchor.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Course Test", "-c", "user.email=course@example.invalid", "commit", "-qm", "anchor"],
        cwd=tmp_path,
        check=True,
    )
    (tmp_path / "course.py").write_text("VALUE = 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not tracked by HEAD"):
        verify_formal_git_state(tmp_path, ["course.py"], required_branch=None)


def test_repo_root_relative_paths_are_portable(tmp_path: Path) -> None:
    payload = _payload()
    payload["rerank"]["frozen_config"] = "frozen/config.json"
    payload["rerank"]["frozen_model"] = "frozen/model.json"
    frozen_dir = tmp_path / "frozen"
    frozen_dir.mkdir()
    (frozen_dir / "config.json").write_text(
        json.dumps(
            {
                "policy": "source_aware_rrf",
                "lambda_sasrec": 0.75,
                "source_bonus": 0.01,
                "ranker_compatibility_mode": "source_independent_projection",
            }
        ),
        encoding="utf-8",
    )
    (frozen_dir / "model.json").write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "portable.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = load_course_config(config_path, tmp_path)
    assert config.data["inputs"]["cf_prediction"] == str(
        (tmp_path / "tests/course_system/fixtures/cf_predictions.json").resolve()
    )
    assert config.data["rerank"]["frozen_config"] == str((tmp_path / "frozen/config.json").resolve())
