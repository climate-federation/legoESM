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


def test_matrix_hybrid_unset_is_cam6_l32(matrix, monkeypatch):
    """No probe knob: every grid's hybrid rows run the cube duo's CAM6 L32."""
    from legoesm.grids.vertical import make_cam6_l32_levels
    monkeypatch.delenv("LEGOESM_MATRIX_TRANSITION_EXPONENT", raising=False)
    monkeypatch.delenv("LEGOESM_MATRIX_NLEV", raising=False)
    got, ref = matrix._create_vertical(40, "hybrid"), make_cam6_l32_levels()
    assert got.n_levels == 32
    np.testing.assert_array_equal(np.asarray(got.A_half), np.asarray(ref.A_half))
    np.testing.assert_array_equal(np.asarray(got.B_half), np.asarray(ref.B_half))
    assert matrix._vertical_table_id(
        matrix._vertical_table_label("hybrid", got)) == "cam6_l32"
    # sigma rows are untouched by the switch
    assert matrix._create_vertical(40, "sigma").n_levels == 40


def test_matrix_nlev_override_selects_the_analytic_tier(matrix, monkeypatch):
    monkeypatch.delenv("LEGOESM_MATRIX_TRANSITION_EXPONENT", raising=False)
    monkeypatch.setenv("LEGOESM_MATRIX_NLEV", "40")
    got = matrix._create_vertical(40, "hybrid")
    np.testing.assert_array_equal(np.asarray(got.B_half),
                                  np.asarray(standard_hybrid_levels(40).B_half))
    assert matrix._vertical_table_id(matrix._vertical_table_label(
        "hybrid", got)) == "analytic_hybrid_L40_exp3"


def test_cross_grid_rms_refuses_different_tables(matrix):
    """Equal level counts are not enough: the tables must be the same."""
    arr = np.ones((3, 4, 8, 32))           # (time, lat, lon, lev)
    snap = {"T_3d": arr, "lat": np.linspace(-90, 90, 4)}

    class _Snap(dict):
        files = property(lambda self: list(self))

    def row(table):
        return {"snapshots": _Snap(snap),
                "metadata": {"notes": f"x; vertical table: {table} (y)"}}

    same = {"cubed_sphere": row("cam6_l32"), "latlon": row("cam6_l32")}
    assert matrix._compute_cross_grid_rms_agreement(same, "T_3d") is not None
    diff = {"cubed_sphere": row("cam6_l32"),
            "latlon": row("analytic_hybrid_L32_exp3")}
    assert matrix._compute_cross_grid_rms_agreement(diff, "T_3d") is None
    assert matrix._vertical_table_id("a; vertical table: cam6_l32-v2 (x)") is None
    assert matrix._vertical_table_id("a; vertical table: cam6_l32") == "cam6_l32"
    untagged = {"cubed_sphere": row("cam6_l32"),
                "latlon": {"snapshots": _Snap(snap), "metadata": {"notes": "x"}}}
    assert matrix._compute_cross_grid_rms_agreement(untagged, "T_3d") is None
    legacy = {g: {"snapshots": _Snap(snap), "metadata": {"notes": "x"}}
              for g in ("cubed_sphere", "latlon")}
    assert matrix._compute_cross_grid_rms_agreement(legacy, "T_3d") is not None
    # ... but an untagged HYBRID tree is refused: equal counts hid different
    # tables there (analytic L32 off-cube vs CAM6 L32 on the cube)
    for r in legacy.values():
        r["vertical_coord"] = "hybrid"
    assert matrix._compute_cross_grid_rms_agreement(legacy, "T_3d") is None
    # ... and so is an untagged SIGMA tree containing an fv3_duo row (the
    # duo runs CAM6 L32 under any label), while untagged sigma rows with no
    # duo (DCMIP transport: one shared table) keep the count fallback
    for r in legacy.values():
        r["vertical_coord"] = "sigma"
    assert matrix._compute_cross_grid_rms_agreement(legacy, "T_3d") is not None
    legacy["cubed_sphere"]["metadata"]["notes"] = (
        "mass drift=0; fv3_duo FV3DuoDynamicsModel km=32 eta=cam6_l32")
    assert matrix._compute_cross_grid_rms_agreement(legacy, "T_3d") is None


@pytest.mark.parametrize("nlev_env,exp_env,want", [
    ("40", None, "analytic_hybrid_L40_exp3"),   # tier default resolved
    ("20", None, "analytic_hybrid_L20_exp2"),   # a different tier
    (None, "2", "analytic_hybrid_L40_exp2"),    # exponent alone -> L40
    ("32", "2", "analytic_hybrid_L32_exp2"),
])
def test_analytic_tag_records_effective_nlev_and_exp(matrix, monkeypatch,
                                                     nlev_env, exp_env, want):
    """Either probe knob opts into the analytic table; the row tag and the
    namelist both carry its effective level count and B = eta**exp power."""
    for k, v in (("LEGOESM_MATRIX_NLEV", nlev_env),
                 ("LEGOESM_MATRIX_TRANSITION_EXPONENT", exp_env)):
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    nlev = int(nlev_env) if nlev_env else 40
    monkeypatch.setattr(matrix, "DEFAULT_NLEV", nlev)
    sig = matrix._create_vertical(nlev, "hybrid")
    label = matrix._vertical_table_label("hybrid", sig)
    assert matrix._vertical_table_id(label) == want
    n, e = want.split("_L")[1].split("_exp")
    assert f"analytic (nlev={n}, exp={e})" in label
    nl = matrix._namelist_vertical_table(matrix.TestCase(
        "hydrostatic", "held_suarez", "latlon", "72x144", "hybrid", 1, 1))
    assert nl.split()[0] == want and f"analytic (nlev={n}, exp={e})" in nl


def test_namelist_vertical_table_default_and_scope(matrix, monkeypatch):
    monkeypatch.delenv("LEGOESM_MATRIX_NLEV", raising=False)
    monkeypatch.delenv("LEGOESM_MATRIX_TRANSITION_EXPONENT", raising=False)
    mk = lambda g, v, eq="hydrostatic": matrix.TestCase(
        eq, "held_suarez", g, "x", v, 1, 1)
    assert matrix._namelist_vertical_table(
        mk("latlon", "hybrid")).split()[0] == "cam6_l32"
    assert matrix._namelist_vertical_table(
        mk("cubed_sphere", "hybrid")).split()[0] == "cam6_l32"
    assert matrix._namelist_vertical_table(mk("latlon", "sigma")) is None
    assert matrix._namelist_vertical_table(
        matrix.TestCase("hydrostatic", "amip", "cubed_sphere", "C36",
                        "hybrid", 1, 1)) is None   # skipped row
    monkeypatch.setenv("LEGOESM_MATRIX_TRANSITION_EXPONENT", "+2")
    assert matrix._namelist_vertical_table(
        mk("latlon", "hybrid")).startswith("invalid")   # no crash
    monkeypatch.delenv("LEGOESM_MATRIX_TRANSITION_EXPONENT")
    assert matrix._namelist_vertical_table(
        mk("latlon", "hybrid", "nonhydrostatic")) is None
    # the cube ignores the probe knobs: it always runs CAM6 L32
    monkeypatch.setenv("LEGOESM_MATRIX_NLEV", "40")
    assert matrix._namelist_vertical_table(
        mk("cubed_sphere", "hybrid")).split()[0] == "cam6_l32"


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
