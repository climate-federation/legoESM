"""The #1029 vertical-coordinate exponent probe must change only what it says.

``standard_hybrid_levels(transition_exponent=...)`` and the matrix runner's
``LEGOESM_MATRIX_TRANSITION_EXPONENT`` override exist to run the
``held_suarez_topo`` case at ``B = eta**2`` against ``eta**3`` with everything
else held fixed.  A probe that silently changes nothing, or that also moves the
reference level pressures, would make that comparison meaningless.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import jax
import numpy as np
import pytest

from legoesm.grids.vertical import standard_hybrid_levels

jax.config.update("jax_enable_x64", True)

MATRIX = (pathlib.Path(__file__).resolve().parents[2]
          / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")


def _coeffs(c):
    return np.asarray(c.A_half), np.asarray(c.B_half)


def test_default_is_unchanged_and_explicit_tier_value_matches():
    a0, b0 = _coeffs(standard_hybrid_levels(40))
    a3, b3 = _coeffs(standard_hybrid_levels(40, transition_exponent=3))
    np.testing.assert_array_equal(a0, a3)
    np.testing.assert_array_equal(b0, b3)


def test_exponent_two_moves_B_but_not_the_reference_pressures():
    a3, b3 = _coeffs(standard_hybrid_levels(40))
    a2, b2 = _coeffs(standard_hybrid_levels(40, transition_exponent=2))
    # the probe is live: terrain-following character changes ...
    assert np.max(np.abs(b2 - b3)) > 0.05
    # ... while A + B (level pressures at p_s = p_ref) agrees to the float32
    # storage rounding of the coefficients (~3e-8, a few mPa)
    np.testing.assert_allclose(a2 + b2, a3 + b3, rtol=0, atol=1e-7)
    # and exponent 2 has no negative-layer-mass threshold above 500 hPa
    p_s = 5.0e4
    dp = np.diff(a2 * 1.0e5 + b2 * p_s)
    assert np.all(dp > 0)


@pytest.fixture(scope="module")
def matrix():
    spec = importlib.util.spec_from_file_location("_atm_matrix", MATRIX)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod      # dataclasses resolve their module here
    spec.loader.exec_module(mod)
    return mod


def test_matrix_override_unset_is_the_standard_tier(matrix, monkeypatch):
    monkeypatch.delenv("LEGOESM_MATRIX_TRANSITION_EXPONENT", raising=False)
    np.testing.assert_array_equal(
        np.asarray(matrix._create_vertical(40, "hybrid").B_half),
        np.asarray(standard_hybrid_levels(40).B_half))


def test_matrix_override_selects_exponent_two(matrix, monkeypatch):
    monkeypatch.setenv("LEGOESM_MATRIX_TRANSITION_EXPONENT", "2")
    np.testing.assert_array_equal(
        np.asarray(matrix._create_vertical(40, "hybrid").B_half),
        np.asarray(standard_hybrid_levels(40, transition_exponent=2).B_half))


@pytest.mark.parametrize("value,coord", [("4", "hybrid"), ("x", "hybrid"),
                                         ("2", "sigma")])
def test_matrix_override_refuses_bad_or_inert_values(matrix, monkeypatch,
                                                     value, coord):
    monkeypatch.setenv("LEGOESM_MATRIX_TRANSITION_EXPONENT", value)
    with pytest.raises(ValueError, match="LEGOESM_MATRIX_TRANSITION_EXPONENT"):
        matrix._create_vertical(40, coord)


# ----------------------------------------------------------------------
# the survival reader that applies the pre-registered rule
# ----------------------------------------------------------------------

READER = MATRIX.parents[1] / "validate" / "issue_1029_exponent_survival.py"


@pytest.fixture(scope="module")
def reader():
    spec = importlib.util.spec_from_file_location("_i1029_reader", READER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arm(root, e, s, notes, *, status="FAIL", override=True):
    tag = f"e{e}_s{s}"
    d = root / tag / "hydrostatic" / "held_suarez_topo"
    d.mkdir(parents=True)
    (d / "results.txt").write_text(
        f"days: 200\ndt: 200.0\nstatus: {status}\nnotes: {notes}\n")
    (root / f"{tag}.log").write_text(
        f"  #1029 LEGOESM_MATRIX_TRANSITION_EXPONENT override: B = eta**{e}\n"
        if override else "no override line\n")
    return root / tag


def test_reader_classifies_every_arm_outcome(reader, tmp_path):
    died = _arm(tmp_path, 3, 1, "BLOWUP at step 23600 (day 54.63), reason: x")
    lived = _arm(tmp_path, 2, 1, "jet ok", status="PASS")
    failed = _arm(tmp_path, 2, 2, "jet too weak")             # FAIL, no BLOWUP
    inert = _arm(tmp_path, 2, 3, "jet ok", status="PASS", override=False)
    (tmp_path / "e3_s9").mkdir()                              # crashed
    assert reader.read_arm(died, 3) == (54.63, False)
    assert reader.read_arm(lived, 2) == (200.0, True)
    assert isinstance(reader.read_arm(failed, 2), str)
    assert isinstance(reader.read_arm(inert, 2), str)
    assert isinstance(reader.read_arm(tmp_path / "e3_s9", 3), str)


def test_reader_verdicts_follow_the_preregistered_rule(reader):
    base = {(3, s): (10.0 + s, False) for s in range(10)}
    cured = {**base, **{(2, s): (200.0, True) for s in range(10)}}
    assert "CONFIRMED" in reader.verdict(cured)
    delayed = {**base, **{(2, s): (30.0 + s, False) for s in range(10)}}
    assert "NOT CURED" in reader.verdict(delayed)
    same = {**base, **{(2, s): base[(3, s)] for s in range(10)}}
    assert "REFUTED" in reader.verdict(same)
    worse = {**base, **{(2, s): (5.0 + 0.1 * s, False) for s in range(10)}}
    assert "WORSE" in reader.verdict(worse)
    few = {k: v for k, v in cured.items() if k[1] < 5}
    assert "fewer than 10" in reader.verdict(few)


def test_main_requires_every_preregistered_arm(reader, tmp_path):
    for s in reader.SEEDS:
        _arm(tmp_path, 3, s, f"BLOWUP at step 100 (day {10 + s % 7}.00), reason: x")
        _arm(tmp_path, 2, s, "ok", status="PASS")
    dup = tmp_path / "e3_s42dup" / "hydrostatic" / "held_suarez_topo"
    dup.mkdir(parents=True)
    (dup / "results.txt").write_text(
        (tmp_path / "e3_s42" / "hydrostatic" / "held_suarez_topo"
         / "results.txt").read_text())
    (tmp_path / "e3_s42dup.log").write_text((tmp_path / "e3_s42.log").read_text())
    assert reader.main(tmp_path) == 0
    import shutil
    shutil.rmtree(tmp_path / "e2_s7")
    assert reader.main(tmp_path) == 1
