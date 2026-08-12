"""Provenance must describe the IMPORTED package, never the CWD.

Regression tests for the 2026-08-10 incident: a run launched ``cd``-ed into a
git worktree pinned at one commit imported the editable install of another
tree, and every provenance stamp (manifest ``legoESM.commit``, ``git_dirty``)
certified the pin while the other tree's code ran.

These tests build real throwaway git repositories: one plays the imported
package's repo (via a monkeypatched anchor / module ``__file__``), another
plays the launch directory.  The load-bearing assertions are of the form
"stamps THAT repo, not ``os.getcwd()``" — they fail against the old
CWD-derived implementation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from legoesm.io.git_provenance import (
    ALLOW_MISMATCH_ENV,
    check_cwd_import_consistency,
    detect_cwd_import_mismatch,
    git_provenance,
    is_legoesm_checkout,
    repo_root_of,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
         *args],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _make_repo(path: Path, *, legoesm_like: bool = False,
               marker: str = "a") -> str:
    """Create a one-commit git repo at *path*; return its HEAD SHA."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    (path / f"{marker}.txt").write_text(marker)
    if legoesm_like:
        # Federation-layout signature is_legoesm_checkout keys on.
        (path / "src" / "legoesm").mkdir(parents=True)
        (path / "src" / "legoesm" / "constants.py").write_text("# marker\n")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", marker)
    return _git(path, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# git_provenance / repo_root_of
# ---------------------------------------------------------------------------

def test_git_provenance_reports_anchor_repo_not_cwd(tmp_path, monkeypatch):
    head_a = _make_repo(tmp_path / "repo_a", marker="a")
    head_b = _make_repo(tmp_path / "repo_b", marker="b")
    assert head_a != head_b
    anchor = tmp_path / "repo_a" / "a.txt"

    # Sit in repo_b: the old implementation (bare `git rev-parse HEAD`) would
    # report head_b here; the anchor-based one must report repo_a.
    monkeypatch.chdir(tmp_path / "repo_b")
    prov = git_provenance(anchor)
    assert prov.commit == head_a
    assert prov.commit != head_b
    assert Path(prov.root).resolve() == (tmp_path / "repo_a").resolve()
    assert prov.anchor == str(anchor.resolve())
    assert prov.dirty is False

    # Dirty flag also tracks the ANCHOR repo, not the CWD repo.
    (tmp_path / "repo_a" / "scratch.txt").write_text("x")
    assert git_provenance(anchor).dirty is True
    (tmp_path / "repo_b" / "scratch.txt").write_text("x")  # CWD repo dirty
    (tmp_path / "repo_a" / "scratch.txt").unlink()
    assert git_provenance(anchor).dirty is False


def test_git_provenance_outside_any_repo(tmp_path):
    d = tmp_path / "norepo"
    d.mkdir()
    (d / "f.py").write_text("")
    prov = git_provenance(d / "f.py")
    assert prov.root == ""
    assert prov.commit == ""
    assert prov.ref == ""
    assert prov.dirty is False
    assert repo_root_of(d) == ""


def test_git_provenance_branch_ref(tmp_path):
    _make_repo(tmp_path / "r", marker="a")
    branch = _git(tmp_path / "r", "rev-parse", "--abbrev-ref", "HEAD")
    assert git_provenance(tmp_path / "r" / "a.txt").ref == branch


# ---------------------------------------------------------------------------
# is_legoesm_checkout
# ---------------------------------------------------------------------------

def test_is_legoesm_checkout_signatures(tmp_path):
    src_style = tmp_path / "s"
    (src_style / "src" / "legoesm").mkdir(parents=True)
    assert is_legoesm_checkout(src_style)

    pkg_style = tmp_path / "p"
    (pkg_style / "packages" / "core" / "legoesm").mkdir(parents=True)
    assert is_legoesm_checkout(pkg_style)

    plain = tmp_path / "q"
    plain.mkdir()
    assert not is_legoesm_checkout(plain)


# ---------------------------------------------------------------------------
# mismatch detection / fail-loud check
# ---------------------------------------------------------------------------

def test_mismatch_detected_for_legoesm_cwd_at_other_head(tmp_path):
    head_import = _make_repo(tmp_path / "imp", legoesm_like=True, marker="i")
    head_cwd = _make_repo(tmp_path / "pin", legoesm_like=True, marker="p")
    anchor = tmp_path / "imp" / "i.txt"

    mm = detect_cwd_import_mismatch(anchor, cwd=tmp_path / "pin")
    assert mm is not None
    assert mm["cwd_commit"] == head_cwd
    assert mm["import_commit"] == head_import
    assert Path(mm["cwd_root"]).resolve() == (tmp_path / "pin").resolve()


def test_no_mismatch_for_unrelated_repo_same_repo_or_no_repo(tmp_path):
    _make_repo(tmp_path / "imp", legoesm_like=True, marker="i")
    anchor = tmp_path / "imp" / "i.txt"

    # Unrelated (non-legoesm) repo as CWD: a normal driving-repo workflow.
    _make_repo(tmp_path / "other", marker="o")
    assert detect_cwd_import_mismatch(anchor, cwd=tmp_path / "other") is None

    # CWD inside the imported repo itself.
    assert detect_cwd_import_mismatch(anchor, cwd=tmp_path / "imp") is None

    # CWD not a git repo at all.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert detect_cwd_import_mismatch(anchor, cwd=plain) is None


def test_check_raises_with_both_shas_unless_overridden(tmp_path):
    head_import = _make_repo(tmp_path / "imp", legoesm_like=True, marker="i")
    head_cwd = _make_repo(tmp_path / "pin", legoesm_like=True, marker="p")
    anchor = tmp_path / "imp" / "i.txt"

    with pytest.raises(RuntimeError) as exc:
        check_cwd_import_consistency(anchor, cwd=tmp_path / "pin", environ={})
    msg = str(exc.value)
    assert head_import in msg and head_cwd in msg  # both sides named

    mm = check_cwd_import_consistency(
        anchor, cwd=tmp_path / "pin", environ={ALLOW_MISMATCH_ENV: "1"}
    )
    assert mm is not None and mm["cwd_commit"] == head_cwd

    # Consistent case returns None regardless of the override.
    assert check_cwd_import_consistency(
        anchor, cwd=tmp_path / "imp", environ={}
    ) is None


# ---------------------------------------------------------------------------
# state_checkpoint stamps the imported package's repo too
# ---------------------------------------------------------------------------

def test_state_checkpoint_git_hash_uses_module_repo(tmp_path, monkeypatch):
    import legoesm.io.state_checkpoint as sc

    head_a = _make_repo(tmp_path / "repo_a", marker="a")
    head_b = _make_repo(tmp_path / "repo_b", marker="b")
    monkeypatch.setattr(sc, "__file__", str(tmp_path / "repo_a" / "a.txt"))
    monkeypatch.chdir(tmp_path / "repo_b")
    assert sc._get_git_hash() == head_a
    assert sc._get_git_hash() != head_b
