from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from minionerec_system.config import ConfigError, verify_formal_git_state


PROTECTED_PATHS = [
    "minionerec_system",
    "configs/course_system",
    "scripts/run_course_system.py",
    "scripts/check_course_system_artifacts.py",
]


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    _git(tmp_path, "init", "-q")
    files = {
        "minionerec_system/config.py": "VALUE = 1\n",
        "configs/course_system/formal.yaml": "formal: true\n",
        "scripts/run_course_system.py": "print('run')\n",
        "scripts/check_course_system_artifacts.py": "print('check')\n",
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(tmp_path, "add", *files)
    _git(
        tmp_path,
        "-c",
        "user.name=Course Test",
        "-c",
        "user.email=course@example.invalid",
        "commit",
        "-qm",
        "course anchor",
    )
    return tmp_path, _git(tmp_path, "rev-parse", "HEAD")


def _verify(root: Path, head: str) -> dict:
    return verify_formal_git_state(root, PROTECTED_PATHS, required_branch=None, expected_head=head)


def _worktree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


def test_formal_guard_passes_without_untracked_files(tmp_path: Path) -> None:
    root, head = _repository(tmp_path)
    report = _verify(root, head)
    assert report["head"] == head
    assert report["unrelated_untracked_count"] == 0
    assert report["unrelated_untracked_paths"] == []


@pytest.mark.parametrize(
    "relative",
    ["results/course/run.json", "incoming/frozen/predictions.json"],
)
def test_formal_guard_allows_untracked_data_artifacts(tmp_path: Path, relative: str) -> None:
    root, head = _repository(tmp_path)
    path = root / relative
    path.parent.mkdir(parents=True)
    path.write_text("not course code\n", encoding="utf-8")
    report = _verify(root, head)
    assert report["unrelated_untracked_paths"] == [relative]
    assert report["warnings"]


def test_formal_guard_allows_phase4_untracked_and_records_warning(tmp_path: Path) -> None:
    root, head = _repository(tmp_path)
    path = root / "joint_ranking_probe.py"
    path.write_text("PHASE4 = True\n", encoding="utf-8")
    report = _verify(root, head)
    assert report["unrelated_untracked_count"] == 1
    assert report["unrelated_untracked_paths"] == ["joint_ranking_probe.py"]
    assert "allowed 1 unrelated untracked paths" in report["warnings"][0]


@pytest.mark.parametrize(
    "relative",
    ["minionerec_system/untracked.py", "configs/course_system/untracked.yaml"],
)
def test_formal_guard_rejects_untracked_course_paths(tmp_path: Path, relative: str) -> None:
    root, head = _repository(tmp_path)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("untracked\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="untracked course-system paths"):
        _verify(root, head)


def test_formal_guard_rejects_tracked_runner_modification(tmp_path: Path) -> None:
    root, head = _repository(tmp_path)
    (root / "scripts/run_course_system.py").write_text("print('modified')\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="tracked worktree is not clean"):
        _verify(root, head)


def test_formal_guard_rejects_nonempty_index(tmp_path: Path) -> None:
    root, head = _repository(tmp_path)
    (root / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(root, "add", "staged.txt")
    with pytest.raises(ConfigError, match="index is not clean"):
        _verify(root, head)


def test_formal_guard_rejects_head_mismatch(tmp_path: Path) -> None:
    root, _ = _repository(tmp_path)
    with pytest.raises(ConfigError, match="expected course commit"):
        _verify(root, "0" * 40)


def test_formal_guard_never_deletes_moves_or_ignores_files(tmp_path: Path) -> None:
    root, head = _repository(tmp_path)
    unrelated = root / "results/course/run.json"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text("artifact\n", encoding="utf-8")
    protected = root / "minionerec_system/untracked.py"
    protected.write_text("course\n", encoding="utf-8")
    before = _worktree_snapshot(root)
    with pytest.raises(ConfigError):
        _verify(root, head)
    assert _worktree_snapshot(root) == before
    assert not (root / ".gitignore").exists()
