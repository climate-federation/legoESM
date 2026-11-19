"""Direct test of scripts/validate/oracle/spread_f4/gen_intercept.perturb_text, the
token rewriter of the F4 oracle round-off spread instrument."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPREAD = Path("/burg/glab/users/pg2328/oracle_audit/spread/perturb.py")
GEN = Path(__file__).resolve().parents[2] / "scripts/validate/oracle/spread_f4/gen_intercept.py"
pytestmark = pytest.mark.skipif(not SPREAD.exists(), reason="shared perturb.py lives on Ginsburg")


@pytest.fixture(scope="module")
def gi():
    spec = importlib.util.spec_from_file_location("gen_intercept", GEN)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TEXT = "15 32\n3.3370000000000000e+05 9.75 -2.5e-3 0.0 -0.0 1.0d0\n7 1e5 .5 x1.5 nan\n"


def test_base_mode_is_byte_identical(gi):
    t, v = gi.perturb_text(TEXT, None)
    assert t == TEXT
    np.testing.assert_array_equal(v, [3.337e5, 9.75, -2.5e-3, 0.0, -0.0, 1.0, 1e5, 0.5])


def test_seed_moves_every_nonzero_float_one_adjacent_step(gi):
    _, v0 = gi.perturb_text(TEXT, None)
    t, v = gi.perturb_text(TEXT, 3)
    nz = v0 != 0
    step = np.nextafter(v0[nz], np.where(v[nz] > v0[nz], np.inf, -np.inf))
    assert step.tobytes() == v[nz].tobytes()
    assert np.all(v[~nz] == 0)
    toks = t.split()
    assert toks[:2] == ["15", "32"] and toks[8] == "7" and "x1.5" in toks and "nan" in toks
    _, w = gi.perturb_text(t, None)            # the text carries exactly the perturbed values
    assert w.tobytes() == v.tobytes()


def test_per_call_seed_differs_and_is_deterministic(gi):
    a = gi.perturb_text(TEXT, 3, 0)[1]
    assert gi.perturb_text(TEXT, 3, 0)[1].tobytes() == a.tobytes()
    many = np.array([float(i) + 0.5 for i in range(64)])
    txt = " ".join(repr(x) for x in many)
    assert gi.perturb_text(txt, 3, 0)[1].tobytes() != gi.perturb_text(txt, 3, 1)[1].tobytes()
