"""Direct test for the CLUBB TOC regenerator.

Every new `.py` gets a test that imports and exercises the leaf module. What is
worth gating here is that the checker can SEE drift: a `--check` that always
reports "exact" would leave the TOC free to rot while looking guarded, which is
the failure mode the generator exists to end.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC = _ROOT / "scripts" / "validate" / "regen_clubb_toc.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("regen_clubb_toc", _SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_real_file_is_exact(mod):
    """The committed TOC agrees with the committed headers."""
    lines = pathlib.Path(mod._DEFAULT).read_text().split("\n")
    _, drift = mod.retarget(lines)
    assert not drift, drift


def test_planted_drift_is_found_and_repaired(mod):
    """Non-vacuity, both directions: a wrong number must be SEEN and FIXED."""
    lines = [
        '"""Module.',
        "",
        "Table of contents (sections, in order; flag reference table at line 99)",
        "",
        "  1. [line  7] First",
        "  2. [line  8] Second",
        '"""',
        "# 1. First",
        "# 2. Second",
        "# CAM-default CLUBB model-flag values (reference table)",
    ]
    # As written the TOC is a lie: the headers are at 8 and 9, the banner at 10.
    out, drift = mod.retarget(lines)
    assert len(drift) == 3, drift
    assert "  1. [line  8] First" in out
    assert "  2. [line  9] Second" in out
    assert "flag reference table at line 10" in out[2]
    # ...and the repaired text is then clean, so the fix is a fixed point.
    _, again = mod.retarget(out)
    assert not again, again


def test_a_later_mention_does_not_steal_a_row(mod):
    """The FIRST `# N. ` wins: prose citing a section number must not retarget."""
    lines = [
        '"""M.', "", "  1. [line  5] First", '"""', "# 1. First",
        "x = 1", "# 1. see above",
    ]
    out, drift = mod.retarget(lines)
    assert not drift, drift
    assert "  1. [line  5] First" in out   # NOT line 7, the later mention


def test_toc_in_sync(mod):
    """PYTEST is the invoker.

    `--check` exits 1 on drift, but nothing runs it: this repo has no CI, so
    the only thing that reliably executes is the suite. This is the door that
    is actually open.
    """
    assert mod.check_toc() == []


def test_the_checker_refuses_to_validate_nothing(mod, tmp_path):
    """The silent pass: a TOC whose format drifted matches zero rows, and a
    checker that then reports 'in sync' is worse than one reporting drift."""
    f = tmp_path / "fake.py"
    f.write_text('"""No TOC rows here at all."""\n# 1. First\n')
    out = mod.check_toc(f)
    assert out and "validating nothing" in out[0], out


def _full_toc(mod, pointer=True, drop_header=None):
    """A synthetic file with all EXPECTED_TOC_ROWS rows, correctly numbered."""
    n = mod.EXPECTED_TOC_ROWS
    head = ['"""M.']
    if pointer:
        head.append("Table of contents (flag reference table at line 0)")
    head += [f"  {k}. [line  0] S{k}" for k in range(1, n + 1)] + ['"""']
    body = [f"# {k}. S{k}" for k in range(1, n + 1) if k != drop_header]
    return head + body + ["# CAM-default CLUBB model-flag values (reference table)"]


def test_a_missing_pointer_is_not_exact(mod):
    """All rows right but the flag-table pointer gone must not read as exact."""
    lines, _ = mod.retarget(_full_toc(mod, pointer=False))
    out = mod.check_lines(lines)
    assert out and "pointers" in out[0], out
    # Control: the same file WITH its pointer is exact once renumbered.
    fixed, _ = mod.retarget(_full_toc(mod))
    assert mod.check_lines(fixed) == []


def test_an_unrepairable_toc_is_not_rewritten(mod, tmp_path):
    """A row whose header is gone cannot be renumbered; refuse, do not write."""
    f = tmp_path / "fake.py"
    text = "\n".join(_full_toc(mod, drop_header=3))
    f.write_text(text)
    assert mod.main(["--path", str(f)]) == 1
    assert f.read_text() == text
    # Control: pure renumbering drift IS repaired and written.
    f.write_text("\n".join(_full_toc(mod)))
    assert mod.main(["--path", str(f)]) == 0
    assert mod.check_toc(f) == []
