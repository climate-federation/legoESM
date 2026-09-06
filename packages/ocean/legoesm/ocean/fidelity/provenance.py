"""Producer-revision stamp for oracle-fidelity gate artifacts.

A gate JSON that names a git SHA claims "this revision produced these
numbers".  ``git rev-parse HEAD`` on a tree with uncommitted TRACKED edits
does not identify the producing code -- the OVERFLOW 19-frame gate once
stamped ``5104de94`` on a JSON that only the not-yet-committed ``3a68e433``
could have written.  So the stamp FAILS CLOSED on tracked dirt; a caller that
knowingly runs uncommitted code passes ``allow_dirty=True`` and gets an
explicit ``<sha>-dirty`` stamp that no receipt may cite as a revision.

Untracked files are ignored (``--untracked-files=no``): they cannot change a
number, and counting them made an earlier stamp read "dirty" permanently.

Harness glue, not model code (same home as ``precision_gate``).
"""
from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
from pathlib import Path

ALLOW_DIRTY_ENV = "LEGOESM_GATE_ALLOW_DIRTY"
# A gate that already owns an --allow-dirty flag must be able to reach the
# shared stamp with it; otherwise wiring the stamp in silently kills the flag
# and the gate raises AFTER its model run.
_ALLOW_DIRTY = [False]


def allow_dirty_stamps(enable: bool = True) -> None:
    """Let a caller's own ``--allow-dirty`` flag reach ``worktree_stamp``."""
    _ALLOW_DIRTY[0] = bool(enable)

__all__ = ["git_sha", "worktree_stamp"]


def git_sha(*, allow_dirty: bool = False, repo: str | Path | None = None) -> str:
    """Full HEAD SHA of the tree this module was imported from.

    Raises ``RuntimeError`` off Git or on tracked dirt unless ``allow_dirty``,
    in which case the stamp is ``"<sha>-dirty"``.  ``repo`` overrides the
    tree (tests); the default is the checkout that owns this file, i.e. the
    code actually executing, not the caller's working directory.
    """
    tree = Path(repo) if repo is not None else Path(__file__).resolve().parent

    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(tree), *args], text=True, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(f"cannot stamp git SHA from {tree}: {error}") from error

    sha = _git("rev-parse", "HEAD").strip()
    # No strip() on the porcelain output: " M path" keeps its leading space.
    dirty = [line for line in _git("status", "--porcelain", "--untracked-files=no").splitlines()
             if line.strip()]
    if not dirty:
        return sha
    if allow_dirty:
        return f"{sha}-dirty"
    raise RuntimeError(
        f"refusing to stamp {sha[:12]}: {len(dirty)} tracked file(s) modified "
        f"({', '.join(line[3:] for line in dirty[:5])}); commit first or pass "
        "allow_dirty=True to stamp '<sha>-dirty'"
    )


def worktree_stamp(*, repo: str | Path | None = None) -> dict:
    """The full identity of the tree a gate report was produced from.

    ``git_sha`` above answers "which revision", and it fails closed on tracked
    dirt for exactly the right reason.  It does not answer "was ``HEAD`` even
    describing this code" --- a worktree left DETACHED by another session
    reports a perfectly valid SHA that names different source, and every gate
    run in it emits a report that silently disagrees with its own stamp.  That
    happened for most of a day during round 30.

    So this returns the SHA plus the three things that distinguish the two
    situations: the branch (or ``DETACHED``), whether the tree is clean, and,
    when it is not, the SHA-256 of the full ``git diff HEAD`` so the exact
    content that produced the numbers is pinned rather than described.
    Detached is RECORDED, never refused --- the cross-card probe lanes are
    deliberately detached worktrees, and a detached tree whose status is clean
    has no ``HEAD``/code disagreement to hide.

    Dirt refuses, through ``git_sha``'s own guard, unless
    ``LEGOESM_GATE_ALLOW_DIRTY=1`` names the escape in the environment; the
    escape suppresses nothing, it records ``clean: false`` and the diff hash.
    """
    tree = Path(repo) if repo is not None else Path(__file__).resolve().parent

    def _git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(tree), *args], text=True,
            stderr=subprocess.DEVNULL)

    try:
        top = _git("rev-parse", "--show-toplevel").strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot stamp a worktree from {tree}: {error}") from error
    # FIRST, before anything is measured or refused for another reason: a stamp
    # that names a DIFFERENT tree than the caller's source is the round-30
    # failure wearing a stamp.  Run a gate by absolute path without PYTHONPATH
    # and the INSTALLED package answers for a checkout the gate never came
    # from, and every number in the report is then attributed to source that
    # did not produce it.  Refuse rather than describe the wrong tree.
    if repo is None:
        caller_file = Path(inspect.stack()[1].filename).resolve()
        if caller_file.exists():
            try:
                caller_top = subprocess.check_output(
                    ["git", "-C", str(caller_file.parent), "rev-parse",
                     "--show-toplevel"],
                    text=True, stderr=subprocess.DEVNULL).strip()
            except (OSError, subprocess.CalledProcessError):
                caller_top = None
            if caller_top is not None and Path(caller_top) != Path(top):
                raise RuntimeError(
                    f"worktree_stamp: this module is in {top} but its caller "
                    f"{caller_file} is in {caller_top}, so the report would "
                    "name a tree the gate's own source did not come from -- "
                    "set PYTHONPATH to the checkout you mean to measure.")
    allow_dirty = (os.environ.get(ALLOW_DIRTY_ENV) == "1") or _ALLOW_DIRTY[0]
    sha = git_sha(allow_dirty=allow_dirty, repo=tree)
    clean = not sha.endswith("-dirty")
    commit = sha[:-len("-dirty")] if not clean else sha
    branch = _git("rev-parse", "--abbrev-ref", "HEAD").strip()
    dirty = sorted(
        line[3:] for line in
        _git("status", "--porcelain", "--untracked-files=no").splitlines()
        if line.strip())
    # ls-files is path-limited to its cwd where status is not, so ask from
    # the worktree root or the count silently describes one subdirectory.
    untracked = [line for line in subprocess.check_output(
        ["git", "-C", top, "ls-files", "--others", "--exclude-standard"],
        text=True, stderr=subprocess.DEVNULL).splitlines() if line.strip()]
    return {
        "root": str(tree),
        "commit": commit,
        "branch": "DETACHED" if branch == "HEAD" else branch,
        "detached": branch == "HEAD",
        "clean": clean,
        "dirty_paths": dirty,
        "diff_sha256": (None if clean else
                        hashlib.sha256(_git("diff", "HEAD").encode()).hexdigest()),
        "untracked_count": len(untracked),
        "allow_dirty_escape_used": (not clean) and allow_dirty,
    }
