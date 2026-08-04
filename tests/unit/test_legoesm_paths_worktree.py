"""#1389: the path helpers must work from a git worktree.

``legoesm.__path__`` reports the EDITABLE INSTALL's roots, which point at the
canonical checkout. A test running from a worktree therefore got roots outside
its own tree, and ``Path.relative_to`` raised a bare ``ValueError`` naming a
foreign repo path — which is what made two ratchets "error" in #1389 and got
misread as a harness quirk rather than a real fragility.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.legoesm_paths import legoesm_root_paths, legoesm_source_path

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_roots_are_repo_relative_when_repo_root_given():
    roots = legoesm_root_paths(REPO_ROOT)
    assert roots
    for r in roots:
        # The property the ratchets depend on: relative_to must not raise.
        r.relative_to(REPO_ROOT)


def test_foreign_root_is_remapped_into_the_given_checkout(tmp_path, monkeypatch):
    """Simulate the worktree case: install roots live in ANOTHER tree."""
    other = tmp_path / "canonical"
    (other / "src" / "legoesm").mkdir(parents=True)
    (other / "packages" / "core" / "legoesm").mkdir(parents=True)
    here = tmp_path / "worktree"
    (here / "src" / "legoesm").mkdir(parents=True)
    (here / "packages" / "core" / "legoesm").mkdir(parents=True)

    import legoesm
    monkeypatch.setattr(
        legoesm, "__path__",
        [str(other / "src" / "legoesm"),
         str(other / "packages" / "core" / "legoesm")])

    roots = legoesm_root_paths(here)
    assert roots == [here / "src" / "legoesm",
                     here / "packages" / "core" / "legoesm"], roots
    for r in roots:
        r.relative_to(here)  # must not raise


def test_unmappable_root_is_returned_unchanged(tmp_path, monkeypatch):
    """No counterpart in this checkout -> return the original, do not invent."""
    import legoesm
    foreign = tmp_path / "elsewhere" / "src" / "legoesm"
    foreign.mkdir(parents=True)
    monkeypatch.setattr(legoesm, "__path__", [str(foreign)])
    assert legoesm_root_paths(tmp_path / "empty") == [foreign]


def test_source_path_threads_repo_root():
    p = legoesm_source_path("constants.py", REPO_ROOT)
    assert p.is_file()
    assert p.is_relative_to(REPO_ROOT)
    with pytest.raises(FileNotFoundError):
        legoesm_source_path("definitely_not_a_module_1389.py", REPO_ROOT)
