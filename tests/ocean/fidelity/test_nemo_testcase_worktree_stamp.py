"""Every NEMO-testcase gate report must say which tree produced it.

Round 30's worktree was left DETACHED by an earlier session, so for most of a
day every gate run in it emitted a report whose ``HEAD`` named different source
than the code that produced the numbers -- and none of the round-30 artifacts
carried a revision stamp at all, so nothing in them said so.

``legoesm.ocean.fidelity.provenance.worktree_stamp`` closes that, and the
ratchet below refuses a NEW report emitter that does not carry it.  The
non-vacuity arms are the point: each of them is a synthetic violation that the
gate must reject, so none of these tests can pass by inspecting nothing.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest
from legoesm.ocean.fidelity.provenance import ALLOW_DIRTY_ENV, worktree_stamp

GATES = (Path(__file__).resolve().parents[3]
         / "scripts/validate/ocean_fidelity/testcases")


def _report_dicts(source: str) -> list[ast.Dict]:
    """Every dict literal carrying a ``"format"`` key -- i.e. a gate report."""
    return [node for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Dict)
            and any(isinstance(k, ast.Constant) and k.value == "format"
                    for k in node.keys)]


def _unstamped(source: str) -> int:
    return sum(
        1 for d in _report_dicts(source)
        if not any(isinstance(k, ast.Constant) and k.value == "worktree"
                   for k in d.keys))


def test_every_report_emitter_stamps_the_worktree():
    """Grow-only: a new report dict without a worktree stamp goes red."""
    offenders = {}
    total = 0
    for path in sorted(GATES.glob("*.py")):
        source = path.read_text()
        dicts = _report_dicts(source)
        total += len(dicts)
        bad = _unstamped(source)
        if bad:
            offenders[path.name] = bad
    assert total >= 57, f"the scan found only {total} report dicts; it broke"
    assert offenders == {}, (
        "these gates emit a report that cannot say which tree produced it: "
        f"{offenders}")


def test_the_ratchet_can_fail():
    """Non-vacuity: the scan rejects a report dict with the stamp removed."""
    stamped = '{"worktree": worktree_stamp(), "format": "x", "rows": []}'
    assert _unstamped(f"r = {stamped}") == 0
    assert _unstamped('r = {"format": "x", "rows": []}') == 1
    # and it must not be fooled by a "worktree" key in a NON-report dict
    assert _unstamped('r = {"worktree": 1}\ns = {"format": "x"}') == 1


def _repo(tmp_path: Path) -> Path:
    def git(*args):
        subprocess.run(("git", "-C", str(tmp_path), *args),
                       check=True, capture_output=True, text=True)
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    git("config", "user.email", "t@t"); git("config", "user.name", "t")
    (tmp_path / "f.txt").write_text("one\n")
    git("add", "f.txt"); git("commit", "-qm", "one")
    return tmp_path


def test_a_clean_tree_stamps_its_branch_and_commit(tmp_path):
    stamp = worktree_stamp(repo=_repo(tmp_path))
    assert stamp["clean"] is True
    assert stamp["detached"] is False
    assert stamp["branch"] not in ("DETACHED", "HEAD")
    assert len(stamp["commit"]) == 40
    assert stamp["diff_sha256"] is None


def test_a_dirty_tree_refuses(tmp_path, monkeypatch):
    """The state a HEAD/code disagreement produces must not emit a report."""
    repo = _repo(tmp_path)
    (repo / "f.txt").write_text("two\n")
    monkeypatch.delenv(ALLOW_DIRTY_ENV, raising=False)
    with pytest.raises(RuntimeError, match="refusing to stamp"):
        worktree_stamp(repo=repo)


def test_the_dirty_escape_records_rather_than_suppresses(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "f.txt").write_text("two\n")
    monkeypatch.setenv(ALLOW_DIRTY_ENV, "1")
    stamp = worktree_stamp(repo=repo)
    assert stamp["clean"] is False
    assert stamp["allow_dirty_escape_used"] is True
    assert stamp["dirty_paths"] == ["f.txt"]
    assert stamp["diff_sha256"] and len(stamp["diff_sha256"]) == 64


def test_a_detached_tree_is_recorded_not_refused(tmp_path):
    """The cross-card probe lanes are deliberately detached worktrees."""
    repo = _repo(tmp_path)
    head = subprocess.run(("git", "-C", str(repo), "rev-parse", "HEAD"),
                          check=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(("git", "-C", str(repo), "checkout", "-q", "--detach", head),
                   check=True)
    stamp = worktree_stamp(repo=repo)
    assert stamp["detached"] is True
    assert stamp["branch"] == "DETACHED"
    assert stamp["clean"] is True
    assert stamp["commit"] == head


def test_a_tree_git_cannot_describe_refuses(tmp_path):
    with pytest.raises(RuntimeError):
        worktree_stamp(repo=tmp_path / "not-a-repo")


def test_untracked_files_are_counted_not_refused(tmp_path):
    repo = _repo(tmp_path)
    (repo / "scratch.py").write_text("# not committed\n")
    stamp = worktree_stamp(repo=repo)
    assert stamp["clean"] is True
    assert stamp["untracked_count"] == 1
