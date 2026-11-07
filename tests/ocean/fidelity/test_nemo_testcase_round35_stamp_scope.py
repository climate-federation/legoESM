"""The allow-dirty stamp escape must not outlive the driver that armed it.

WHAT THIS EXISTS FOR.  ``worktree_stamp`` fails closed on a tree with
uncommitted TRACKED edits, which is what makes a gate report citable as
"revision X produced these numbers".  A driver that knowingly runs uncommitted
code arms an escape.  That escape was a PROCESS-GLOBAL LATCH THAT NOTHING
RESET: once any driver ``main`` armed it, every later ``worktree_stamp`` in the
same process accepted a dirty tree SILENTLY.  A harness running two gates back
to back would stamp the second one clean while it was not.

It surfaced as a stamp test that passed alone and failed inside the full suite,
because two end-to-end tests call a gate's ``main`` in-process and left the
latch armed for everything collected after them.  That is the same leak seen
from the other side, and the fix is at the source rather than in the test.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from legoesm.ocean.fidelity.provenance import (
    allow_dirty_stamps,
    dirty_stamps_allowed,
    scoped_allow_dirty,
    worktree_stamp,
)

REPO = Path(__file__).resolve().parents[3]
# The ratchet's scope is the DRIVERS *and* the shared module they all call.
# It saw only the scripts directory, so the six comparison reports written by
# ``ulp_move_gate`` -- the module every one of those drivers routes through --
# were outside it and went unstamped.
DRIVERS = sorted(
    (REPO / "scripts/validate/ocean_fidelity/testcases").glob("*.py"))
SHARED_FIDELITY = sorted(
    (REPO / "packages/ocean/legoesm/ocean/fidelity").glob("*_gate.py"))


@pytest.fixture(autouse=True)
def _restore_latch():
    previous = dirty_stamps_allowed()
    yield
    allow_dirty_stamps(previous)


def _dirty_repo(tmp_path: Path) -> Path:
    tree = tmp_path / "tree"
    tree.mkdir()
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", "-C", str(tree), *a], check=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run("init", "-q")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "t")
    (tree / "a.txt").write_text("one\n")
    run("add", "a.txt")
    run("commit", "-q", "-m", "one")
    (tree / "a.txt").write_text("two\n")          # tracked dirt
    return tree


def test_two_stamps_in_one_process_the_second_refuses(tmp_path):
    """The regression itself: arm, stamp, then stamp again WITHOUT arming."""
    tree = _dirty_repo(tmp_path)

    @scoped_allow_dirty
    def first_driver_main():
        allow_dirty_stamps(True)
        return worktree_stamp(repo=tree)

    stamp = first_driver_main()
    assert stamp["clean"] is False
    assert stamp["allow_dirty_escape_used"] is True

    assert dirty_stamps_allowed() is False, (
        "the first driver's escape outlived its main; every later stamp in "
        "this process would accept a dirty tree silently")
    with pytest.raises(RuntimeError):
        worktree_stamp(repo=tree)


def test_the_leak_is_what_the_decorator_removes(tmp_path):
    """Non-vacuity: the SAME body without the decorator leaks, and is caught."""
    tree = _dirty_repo(tmp_path)

    def undecorated_driver_main():                # the pre-fix shape
        allow_dirty_stamps(True)
        return worktree_stamp(repo=tree)

    undecorated_driver_main()
    assert dirty_stamps_allowed() is True         # the defect, reproduced
    assert worktree_stamp(repo=tree)["clean"] is False   # accepted silently
    allow_dirty_stamps(False)


def test_context_manager_restores_on_exception(tmp_path):
    tree = _dirty_repo(tmp_path)
    assert dirty_stamps_allowed() is False
    with pytest.raises(ZeroDivisionError):
        with allow_dirty_stamps(True):
            assert worktree_stamp(repo=tree)["clean"] is False
            1 / 0
    assert dirty_stamps_allowed() is False


def test_explicit_argument_needs_no_module_state(tmp_path):
    tree = _dirty_repo(tmp_path)
    stamp = worktree_stamp(repo=tree, allow_dirty=True)
    assert stamp["allow_dirty_escape_used"] is True
    assert dirty_stamps_allowed() is False
    with pytest.raises(RuntimeError):
        worktree_stamp(repo=tree)


def test_every_driver_that_arms_the_escape_scopes_it():
    """The GATE, not the prose: a new driver cannot reintroduce the leak.

    Rule 2 of this repo's memory: a rule that no mechanism checks is not
    enforced.  Any driver that calls ``allow_dirty_stamps`` must also carry
    ``@scoped_allow_dirty`` on its ``main``.
    """
    offenders = []
    for path in [*DRIVERS, *SHARED_FIDELITY]:
        text = path.read_text()
        if "allow_dirty_stamps" not in text:
            continue
        decorated = re.search(r"^@scoped_allow_dirty\s*\ndef main\(", text, re.M)
        if not decorated:
            offenders.append(path.name)
    assert offenders == [], (
        "these drivers arm the process-global allow-dirty escape without "
        f"scoping it to their main: {offenders}")


def test_the_shared_module_is_inside_the_ratchet():
    """The scope, pinned: ulp_move_gate is the module every driver calls."""
    assert any(path.name == "ulp_move_gate.py" for path in SHARED_FIDELITY), (
        "the comparison module the drivers share is outside the audit's "
        "scope; six comparison reports went unstamped inside it")


def test_a_comparison_report_carries_its_provenance():
    """A comparison is a MEASUREMENT and is stamped like the reports it reads.

    Red on the parent commit: ``run_ulp_comparison`` returned a dict with no
    ``worktree`` key at all.
    """
    import inspect

    from legoesm.ocean.fidelity import ulp_move_gate

    source = inspect.getsource(ulp_move_gate.run_ulp_comparison)
    assert 'result["worktree"] = worktree_stamp()' in source, (
        "run_ulp_comparison does not stamp its result; a comparison taken on "
        "a dirty tree would be indistinguishable from one taken at a commit")


def test_the_ratchet_can_fail(tmp_path):
    """Synthetic violation: the audit above must reject an undecorated driver."""
    fake = tmp_path / "nemo_testcase_fake_driver.py"
    fake.write_text("from x import allow_dirty_stamps\ndef main():\n    pass\n")
    text = fake.read_text()
    assert "allow_dirty_stamps" in text
    assert not re.search(r"^@scoped_allow_dirty\s*\ndef main\(", text, re.M)
