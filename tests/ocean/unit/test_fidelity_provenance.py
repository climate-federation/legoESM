"""Fail-closed producer-revision stamp (legoesm.ocean.fidelity.provenance)."""

from __future__ import annotations

import subprocess

import pytest

from legoesm.ocean.fidelity.provenance import git_sha


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.txt").write_text("a\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_tree_stamps_head(repo):
    assert git_sha(repo=repo) == _git(repo, "rev-parse", "HEAD")


def test_untracked_file_is_not_dirt(repo):
    (repo / "scratch.txt").write_text("x\n")
    assert git_sha(repo=repo) == _git(repo, "rev-parse", "HEAD")


def test_tracked_edit_fails_closed_unless_allowed(repo):
    (repo / "a.txt").write_text("b\n")
    with pytest.raises(RuntimeError, match="a.txt"):
        git_sha(repo=repo)
    assert git_sha(repo=repo, allow_dirty=True) == _git(repo, "rev-parse", "HEAD") + "-dirty"


def test_off_git_fails_closed(tmp_path):
    with pytest.raises(RuntimeError, match="cannot stamp"):
        git_sha(repo=tmp_path)


def test_default_tree_is_the_importing_checkout():
    # The stamp must identify the code that executes (this checkout), not cwd.
    import legoesm.ocean.fidelity.provenance as module
    from pathlib import Path
    expected = subprocess.check_output(
        ["git", "-C", str(Path(module.__file__).resolve().parent), "rev-parse", "HEAD"],
        text=True).strip()
    assert git_sha(allow_dirty=True).split("-")[0] == expected
