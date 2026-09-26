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

import subprocess
from pathlib import Path

__all__ = ["git_sha"]


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
