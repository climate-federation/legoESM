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
