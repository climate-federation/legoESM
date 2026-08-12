"""Git provenance of the IMPORTED legoesm package — never of the CWD.

Why this module exists (2026-08-10 incident, twice in three days): a run was
launched ``cd``-ed into a git worktree pinned at a specific commit, but with no
``PYTHONPATH`` set ``import legoesm`` resolved to the pip editable install of a
*different* tree.  Every provenance stamp (log line, ``run_manifest.json``
``legoESM.commit`` / ``reproducibility.git_dirty``) was derived from the CWD's
repository via bare ``git rev-parse HEAD`` — so the run *self-certified* the
pinned commit while executing branch-HEAD code, silently invalidating the
controlled comparison the pin existed for.

The rule encoded here: provenance is a property of the code that RUNS, i.e. the
imported package.  Every helper takes an *anchor* — the ``__file__`` of the
module doing the stamping — and resolves git facts with ``git -C`` against the
repository containing that file.  The CWD appears only in the consistency
check, whose job is to abort loudly when the CWD is a legoesm checkout whose
HEAD differs from the imported package's (a launcher cd-ed into a pinned tree
that silently runs other code is never what anyone meant).

Override for the rare legitimate case (e.g. deliberately importing tree A while
archiving into tree B's directory): set ``LEGOESM_ALLOW_IMPORT_MISMATCH=1``;
the mismatch is then *recorded* in the manifest instead of fatal.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import NamedTuple

#: Environment variable that downgrades the CWD-vs-import mismatch from fatal
#: to recorded-in-manifest.  Values "1"/"true"/"yes" (case-insensitive) enable.
ALLOW_MISMATCH_ENV = "LEGOESM_ALLOW_IMPORT_MISMATCH"

_GIT_TIMEOUT_S = 5


class GitProvenance(NamedTuple):
    """Git facts about the repository containing *anchor* (all "" / False if
    the anchor is not inside a git repository, e.g. a wheel install)."""

    anchor: str   # resolved path of the module file provenance was derived from
    root: str     # repository top-level ("" if not a git repo)
    commit: str   # full HEAD SHA ("" if unknown)
    ref: str      # branch name ("" if detached or unknown)
    dirty: bool   # uncommitted changes anywhere in the repo (False if unknown)


def _git(args: list[str], cwd: str | Path) -> str:
    """Run a git command rooted at *cwd*; return stripped stdout, "" on any
    failure (missing git binary, not a repo, timeout).  Never raises."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


def repo_root_of(path: str | Path) -> str:
    """Top-level directory of the git repository containing *path* ("" if none).

    Accepts a file or directory; a file anchors at its parent directory.
    Worktrees resolve to the worktree root (what ``git -C`` operates on), not
    the primary checkout.
    """
    p = Path(path)
    d = p if p.is_dir() else p.parent
    return _git(["rev-parse", "--show-toplevel"], d)


def git_provenance(anchor: str | Path) -> GitProvenance:
    """Provenance of the repository containing *anchor* (a module ``__file__``).

    This is the ONLY sanctioned way to stamp a git SHA / dirty flag into a
    reproducibility record: it answers "what code is imported", not "where was
    the process launched from".
    """
    anchor_path = Path(anchor)
    try:
        anchor_resolved = str(anchor_path.resolve())
    except OSError:
        anchor_resolved = str(anchor_path)
    root = repo_root_of(anchor_path)
    if not root:
        return GitProvenance(anchor=anchor_resolved, root="", commit="",
                             ref="", dirty=False)
    commit = _git(["rev-parse", "HEAD"], root)
    ref = _git(["rev-parse", "--abbrev-ref", "HEAD"], root)
    if ref == "HEAD":  # detached
        ref = ""
    dirty = bool(_git(["status", "--porcelain"], root))
    return GitProvenance(anchor=anchor_resolved, root=root, commit=commit,
                         ref=ref, dirty=dirty)


def is_legoesm_checkout(root: str | Path) -> bool:
    """True if *root* looks like a legoesm source tree (federation layout).

    Used to scope the CWD consistency check: launching from an UNRELATED git
    repository (e.g. a calibration campaign repo driving legoesm runs) is a
    normal workflow and must not trip the mismatch abort — only a legoesm
    checkout whose HEAD differs from the imported package's is the trap.
    """
    r = Path(root)
    if (r / "src" / "legoesm").is_dir():
        return True
    packages = r / "packages"
    if packages.is_dir():
        try:
            return any(p.is_dir() for p in packages.glob("*/legoesm"))
        except OSError:
            return False
    return False


def detect_cwd_import_mismatch(
    anchor: str | Path, cwd: str | Path | None = None
) -> dict | None:
    """Return mismatch facts if the CWD is a legoesm checkout at a DIFFERENT
    HEAD than the imported package's repository; ``None`` when consistent.

    ``None`` (consistent) covers: CWD not a git repo, CWD in the same repo as
    the import, CWD in a non-legoesm repo, identical HEADs, or the imported
    package not being in a git repo at all (wheel install — nothing to compare).
    """
    prov = git_provenance(anchor)
    if not prov.root or not prov.commit:
        return None
    cwd_root = repo_root_of(Path(cwd) if cwd is not None else Path(os.getcwd()))
    if not cwd_root:
        return None
    try:
        if Path(cwd_root).resolve() == Path(prov.root).resolve():
            return None
    except OSError:
        pass
    if not is_legoesm_checkout(cwd_root):
        return None
    cwd_commit = _git(["rev-parse", "HEAD"], cwd_root)
    if not cwd_commit or cwd_commit == prov.commit:
        return None
    return {
        "cwd_root": cwd_root,
        "cwd_commit": cwd_commit,
        "import_root": prov.root,
        "import_commit": prov.commit,
        "import_anchor": prov.anchor,
    }


def _mismatch_allowed(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(ALLOW_MISMATCH_ENV, "")).strip().lower() in (
        "1", "true", "yes",
    )


def check_cwd_import_consistency(
    anchor: str | Path, cwd: str | Path | None = None, environ=None
) -> dict | None:
    """Fail loudly if the launch directory pins one legoesm tree while the
    import runs another.

    Returns ``None`` when consistent.  On mismatch: raises ``RuntimeError``
    naming both repositories and HEADs, unless ``LEGOESM_ALLOW_IMPORT_MISMATCH``
    is set — then the mismatch dict is returned so the caller can RECORD it in
    the provenance document instead.
    """
    mismatch = detect_cwd_import_mismatch(anchor, cwd=cwd)
    if mismatch is None:
        return None
    if _mismatch_allowed(environ):
        return mismatch
    raise RuntimeError(
        "legoesm import/CWD provenance mismatch: the current directory is a "
        "legoesm checkout at a different commit than the IMPORTED legoesm "
        "package.  The code that runs is the imported package, not the CWD — "
        "a launcher cd-ed into a pinned tree that silently executes another "
        "tree invalidates the pin.\n"
        f"  CWD repo:      {mismatch['cwd_root']} @ {mismatch['cwd_commit']}\n"
        f"  imported repo: {mismatch['import_root']} @ {mismatch['import_commit']}\n"
        f"  imported from: {mismatch['import_anchor']}\n"
        "Fix: point PYTHONPATH at the pinned tree (packages/* + src), or run "
        "from elsewhere.  To proceed deliberately, set "
        f"{ALLOW_MISMATCH_ENV}=1 — the mismatch is then recorded in "
        "run_manifest.json instead of fatal."
    )
