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

import functools
import hashlib
import inspect
import os
import subprocess
from pathlib import Path

ALLOW_DIRTY_ENV = "LEGOESM_GATE_ALLOW_DIRTY"
# A gate that already owns an --allow-dirty flag must be able to reach the
# shared stamp with it; otherwise wiring the stamp in silently kills the flag
# and the gate raises AFTER its model run.
#
# THIS LATCH ONCE LEAKED, AND THAT WAS A PROVENANCE DEFECT, not a test quirk.
# It is process-global, and nothing reset it: a driver ``main`` that armed it
# left every LATER ``worktree_stamp`` in the same process accepting a dirty
# tree SILENTLY -- so a harness running two gates back to back would stamp the
# second one clean while it was not.  It was found because a stamp test passed
# alone and failed inside the suite, which is the same leak seen from the other
# side.  The escape is now SCOPED: ``allow_dirty_stamps`` restores the previous
# value when its context exits, and every driver ``main`` that arms it wears
# ``@scoped_allow_dirty`` so the escape cannot outlive the call.
_ALLOW_DIRTY = [False]


class _AllowDirtyScope:
    """The armed escape, as a context manager that always restores."""

    def __init__(self, previous: bool) -> None:
        self._previous = previous

    def __enter__(self) -> "_AllowDirtyScope":
        return self

    def __exit__(self, *_exc) -> bool:
        _ALLOW_DIRTY[0] = self._previous
        return False


def allow_dirty_stamps(enable: bool = True) -> _AllowDirtyScope:
    """Let a caller's own ``--allow-dirty`` flag reach ``worktree_stamp``.

    Arms immediately, so an existing bare call behaves exactly as before, and
    RETURNS a context manager that restores the previous value.  Prefer
    ``with allow_dirty_stamps(flag):`` in new code, or ``@scoped_allow_dirty``
    on a driver ``main`` that arms it somewhere inside.
    """
    previous = _ALLOW_DIRTY[0]
    _ALLOW_DIRTY[0] = bool(enable)
    return _AllowDirtyScope(previous)


def dirty_stamps_allowed() -> bool:
    """Whether the process-global escape is currently armed (for tests)."""
    return _ALLOW_DIRTY[0]


def scoped_allow_dirty(function):
    """Wrap a driver ``main`` so its allow-dirty escape cannot outlive it.

    The wrapped call restores the latch on every exit path, including an
    exception and an early ``return``.  A driver that arms the escape and is
    invoked IN-PROCESS -- which every end-to-end test of a gate does -- would
    otherwise disarm the fail-closed stamp for the rest of the process.
    """
    @functools.wraps(function)
    def _scoped(*args, **kwargs):
        previous = _ALLOW_DIRTY[0]
        try:
            return function(*args, **kwargs)
        finally:
            _ALLOW_DIRTY[0] = previous
    return _scoped


__all__ = ["git_sha", "worktree_stamp", "allow_dirty_stamps",
           "dirty_stamps_allowed", "scoped_allow_dirty"]


def _git_output(tree, *args: str) -> str:
    """``git -C <tree> <args>`` stdout.  Raises on a missing git or a failure.

    One owner for the subprocess call; both stamps below wrap it with their
    own error text.
    """
    return subprocess.check_output(
        ["git", "-C", str(tree), *args], text=True, stderr=subprocess.DEVNULL)


def git_sha(*, allow_dirty: bool = False, repo: str | Path | None = None) -> str:
    """Full HEAD SHA of the tree this module was imported from.

    Raises ``RuntimeError`` off Git or on tracked dirt unless ``allow_dirty``,
    in which case the stamp is ``"<sha>-dirty"``.  ``repo`` overrides the
    tree (tests); the default is the checkout that owns this file, i.e. the
    code actually executing, not the caller's working directory.
    """
    tree = Path(repo) if repo is not None else Path(__file__).resolve().parent

    try:
        sha = _git_output(tree, "rev-parse", "HEAD").strip()
        # No strip() on porcelain output: " M path" keeps its leading space.
        status = _git_output(
            tree, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"cannot stamp git SHA from {tree}: {error}") from error
    dirty = [line for line in status.splitlines() if line.strip()]
    if not dirty:
        return sha
    if allow_dirty:
        return f"{sha}-dirty"
    raise RuntimeError(
        f"refusing to stamp {sha[:12]}: {len(dirty)} tracked file(s) modified "
        f"({', '.join(line[3:] for line in dirty[:5])}); commit first or pass "
        "allow_dirty=True to stamp '<sha>-dirty'"
    )


def worktree_stamp(*, repo: str | Path | None = None,
                   allow_dirty: bool = False) -> dict:
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
        return _git_output(tree, *args)

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
    # Explicit argument first: a caller that says what it wants needs no
    # module state, which is the form new code should use.
    allow_dirty = (allow_dirty
                   or os.environ.get(ALLOW_DIRTY_ENV) == "1"
                   or _ALLOW_DIRTY[0])
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
