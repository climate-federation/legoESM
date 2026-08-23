"""The one edit that makes a generated oracle deck moist."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SRC = (pathlib.Path(__file__).resolve().parents[2]
        / "scripts/validate/fv3_native/flip_adiabatic_flag.py")
_spec = importlib.util.spec_from_file_location("_flip", _SRC)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


def test_the_flip_happens_and_nothing_else_moves():
    src = " &fv_core_nml\n   npz = 5\n   adiabatic = .true.\n /\n"
    out = M.flip(src)
    assert "adiabatic = .false." in out
    assert out.replace(".false.", ".true.") == src   # sole edit


@pytest.mark.parametrize("bad,why", [
    ("adiabatic = .false.\n", "already false -> not the intended source"),
    ("npz = 5\n", "absent -> nothing to flip"),
    ("adiabatic = .true.\nadiabatic = .true.\n", "duplicate -> ambiguous"),
    ("adiabatic = .true.\nadiabatic = .false.\n", "mixed -> ambiguous"),
])
def test_anything_ambiguous_is_refused(bad, why):
    """A silent no-op here publishes a DRY deck under a moist name."""
    with pytest.raises(SystemExit):
        M.flip(bad)


def test_a_commented_assignment_does_not_count():
    """`!` starts a comment; a commented .true. beside a live one would
    make the count look like two and must not silently pick either."""
    with pytest.raises(SystemExit):
        M.flip("!adiabatic = .true.\n adiabatic = .false.\n")
