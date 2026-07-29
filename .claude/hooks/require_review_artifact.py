#!/usr/bin/env python3
"""Stop hook: remind (once per session) to run the mandatory Codex adversarial
review when the working tree has a major numerics/physics change and no recent
review artifact.

CLAUDE.md makes a Codex adversarial review MANDATORY after any substantial
numerics/physics/parallel/dycore/ocean/coupler change. This hook is the
enforcement nudge: on Stop, if such a change is uncommitted and no review patch
was produced recently, it injects a one-time reminder so "done" isn't declared
without the review.

Safety (it must never trap a session):
  * Fires at most ONCE per (cwd, session) — a hashed sentinel under a private
    0700 temp subdir, plus the ``stop_hook_active`` guard.
  * Fail-open: any error (no git, parse failure, sentinel write failure, …) →
    allow stop. It only EVER blocks when it has both detected a major change AND
    successfully recorded the one-shot sentinel.
  * Advisory: a single ``decision: block`` with a reason; the next Stop is allowed.

Runs under whatever ``python3`` the harness invokes it with, including the
Python 3.6 shipped as the system interpreter on some HPC login nodes — so it
avoids 3.7+-only syntax and stdlib kwargs (``from __future__ import
annotations``, ``subprocess.run(capture_output=, text=)``).
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

# Path fragments (matched against a leading-slash-normalised path) marking a
# change as "major / review-worthy".
_SIGNIFICANT = (
    "/dynamics/", "/physics/", "/parallel/", "/ocean/", "/coupler/",
    "/timestepping/", "/conservation", "/ice/", "/land/",
)
_REVIEW_RECENCY_S = 2 * 60 * 60  # a review patch within 2h counts as "reviewed"


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-C", cwd] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True, timeout=15,
    ).stdout


def _is_significant_py(path: str) -> bool:
    norm = "/" + path.replace("\\", "/").lstrip("/")
    return norm.endswith(".py") and any(s in norm for s in _SIGNIFICANT)


def _has_recent_review(cwd: str) -> bool:
    reviews = pathlib.Path(cwd) / ".claude" / "reviews"
    if not reviews.is_dir():
        return False
    now = time.time()
    return any(now - p.stat().st_mtime < _REVIEW_RECENCY_S for p in reviews.glob("*.patch"))


def _sentinel(cwd: str, sid: str) -> pathlib.Path:
    key = hashlib.sha256(f"{cwd}:{sid}".encode()).hexdigest()[:16]
    base = pathlib.Path(tempfile.gettempdir()) / "legoesm_harness"
    base.mkdir(mode=0o700, exist_ok=True)
    return base / f"stop_reviewed_{key}"


def main() -> None:
    try:
        data = json.load(sys.stdin)
        if not isinstance(data, dict):
            sys.exit(0)
        if data.get("stop_hook_active"):  # already continuing from a stop hook
            sys.exit(0)
        cwd = data.get("cwd") or os.getcwd()
        sid = str(data.get("session_id", "nosession"))
        sentinel = _sentinel(cwd, sid)
        if sentinel.exists():
            sys.exit(0)

        # Only a change to a numerics/physics path is review-worthy here — a large
        # edit to CLI/docs/test/build Python must NOT trigger a *physics* reminder.
        touched: set[str] = set()
        for line in _git(cwd, "diff", "--numstat", "HEAD").splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and _is_significant_py(parts[2]):
                touched.add(parts[2])
        # Untracked (new) files are invisible to ``diff``; catch new schemes.
        for f in _git(cwd, "ls-files", "--others", "--exclude-standard").splitlines():
            if _is_significant_py(f):
                touched.add(f)

        if not touched or _has_recent_review(cwd):
            sys.exit(0)

        # One-shot: only block if we can durably record that we reminded.
        try:
            sentinel.write_text("reminded")
        except Exception:
            sys.exit(0)  # cannot guarantee one-shot -> fail open (never loop)
    except Exception:
        sys.exit(0)

    listed = ", ".join(sorted(touched)[:5]) + (" …" if len(touched) > 5 else "")
    reason = (
        "Harness reminder (require_review_artifact): a numerics/physics change is "
        "uncommitted (touched " + listed + ") and no Codex review "
        "patch was produced in the last 2h. CLAUDE.md requires the iterate-with-"
        "codex loop (/codex:adversarial-review --wait -> fix -> /codex:review --wait) "
        "before declaring done. This advisory fires once per session."
    )
    # Advisory, NOT a block: ``decision: block`` made every session open
    # with a "Stop hook blocking error" banner (2026-06-12 user request).
    # The reminder still surfaces via systemMessage; the stop proceeds.
    print(json.dumps({"systemMessage": reason}))
    sys.exit(0)


if __name__ == "__main__":
    main()
