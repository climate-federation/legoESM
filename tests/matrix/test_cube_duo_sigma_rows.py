"""Cube (FV3 duo) matrix rows labelled "sigma" run on sigma levels.

The other grids' sigma rows (Held-Suarez, baroclinic, baroclinic_steady)
run ``create_sigma_coordinate(nlev)``; the cube duo used to run its CAM6 L32
hybrid table for them whatever the label.  These tests pin: the FV3 table
built from that SAME sigma coordinate (``fv3_sigma_eta_table``: ak = 0 /
bk = sigma below a fixed lid ptop = sigma_top * p_ref), the duo IC
pressures on it (so the Held-Suarez sigma = p/p_s sees the sigma levels),
and the matrix dispatch selecting it for sigma rows only.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402

A = pytest.importorskip("scripts.matrix._cube_duo_adapters")

NLEV = 40


def _sigma_half(km):
    from legoesm.grids.vertical import create_sigma_coordinate
    return np.asarray(create_sigma_coordinate(km, dtype=jnp.float64).sigma_half)


def test_fv3_sigma_table_is_pure_sigma_below_a_fixed_lid():
    from legoesm.grids.vertical import fv3_sigma_eta_table
    from legoesm.core.fv3_native_eta import interface_pressure
    s = _sigma_half(NLEV)
    ak, bk, ptop = fv3_sigma_eta_table(s)
    assert ptop == s[0] * constants.p_ref
    assert ak[0] == ptop and bk[0] == 0.0
    assert np.all(ak[1:] == 0.0) and np.array_equal(bk[1:], s[1:])
    for ps in (constants.p_ref, 0.93 * constants.p_ref, 1.04 * constants.p_ref):
        pe = interface_pressure(ak, bk, ps)
        assert pe[0] == ptop
        np.testing.assert_array_equal(pe[1:], s[1:] * ps)
    # at p_ref every interface is the sigma coordinate's (the lid too)
    np.testing.assert_allclose(interface_pressure(ak, bk, constants.p_ref),
                               s * constants.p_ref, rtol=1e-15, atol=0.0)
    with pytest.raises(ValueError, match="float64"):
        fv3_sigma_eta_table(s.astype(np.float32))
    with pytest.raises(ValueError, match="increase"):
        fv3_sigma_eta_table(s[::-1].copy())
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and > 0"):
            fv3_sigma_eta_table(s, p_ref=bad)


def test_duo_sigma_table_is_the_other_grids_sigma_coordinate():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import duo_eta_table
    from legoesm.grids.vertical import create_sigma_coordinate
    ak, bk, ptop = duo_eta_table("sigma", NLEV)
    assert ak.shape == bk.shape == (NLEV + 1,)
    # the matrix's own (default-policy) sigma rows, to their float32 rounding
    other = np.asarray(create_sigma_coordinate(NLEV).sigma_half, np.float64)
    np.testing.assert_allclose(bk[1:], other[1:], rtol=0, atol=1e-7)
    np.testing.assert_allclose(ptop / constants.p_ref, other[0], rtol=0,
                               atol=1e-9)
    with pytest.raises(ValueError, match="km >= 2"):
        duo_eta_table("sigma", 1)


def _pe_ps(bundle, duo):
    pe = np.asarray(bundle["press"]["pe"])
    # (6, n+2, km+1, n+2) [face, i, k, j] with a one-cell ring -> (6,n,n,k)
    return np.moveaxis(pe[:, 1:-1, :, 1:-1], 2, -1), duo.ps(bundle)


def test_duo_held_suarez_sigma_ic_and_step_sit_on_the_sigma_levels():
    """The built duo (C36, deck dt) on eta='sigma': the IC's p_var
    interfaces are sigma * p_s below the fixed lid, and stay so after one
    real step (dynamics + FV3 Lagrangian-to-Eulerian remap onto ak + bk ps,
    with p_s now varying) + Held-Suarez forcing -- so the forcing's
    sigma = p / p_s (hswf: layer-mean pl / ps) sees the sigma levels."""
    duo = A.build_duo_hydro(36, held_suarez=True, km=NLEV, eta=A.DUO_SIGMA_ETA)
    s = _sigma_half(NLEV)
    assert duo.km == NLEV and duo.eta == "sigma"
    np.testing.assert_allclose(duo.level_values, 0.5 * (s[:-1] + s[1:]),
                               rtol=1e-14)
    ptop = s[0] * constants.p_ref
    pe, ps = _pe_ps(duo.state, duo)
    np.testing.assert_allclose(pe[..., 0], ptop, rtol=1e-14)
    np.testing.assert_allclose(pe[..., 1:], s[1:] * ps[..., None], rtol=1e-13)
    b1 = duo.step(duo.state, duo.dt)
    pe, ps = _pe_ps(b1, duo)
    assert np.all(np.isfinite(pe))
    assert np.ptp(ps) > 1.0          # p_s moved: the lid/sigma split is tested
    np.testing.assert_allclose(pe[..., 0], ptop, rtol=1e-14)
    np.testing.assert_allclose(pe[..., 1:], s[1:] * ps[..., None], rtol=1e-12)
    # the forcing reads THIS press (hs_fn replaces only the state): its
    # hswf sigma = pl / ps, pl = delp / dln(pe), lies inside each sigma
    # layer below the lid layer.  (That the forcing runs, plain HS94, is
    # test_cube_duo_shared_ic.test_matrix_cube_held_suarez_is_plain_hs94.)
    pl = np.diff(pe, axis=-1) / np.diff(np.log(pe), axis=-1)
    sigl = pl[..., 1:] / ps[..., None]
    assert np.all(sigl > s[1:-1]) and np.all(sigl < s[2:])


def test_sigma_table_refused_for_cam6_only_builders():
    with pytest.raises(ValueError, match="CAM6 L32 only"):
        A.build_duo_hydro(36, rotated_alpha=0.5, km=NLEV, eta=A.DUO_SIGMA_ETA)
    with pytest.raises(ValueError, match="CAM6 L32 only"):
        A.build_duo_hydro(36, shared_case="rest_state_topo", km=NLEV,
                          eta=A.DUO_SIGMA_ETA)


# ---------------------------------------------------------------------------
# matrix dispatch: sigma rows select the sigma table, hybrid rows do not
# ---------------------------------------------------------------------------

class _Stop(Exception):
    pass


@pytest.fixture(scope="module")
def matrix_mod():
    import importlib.util
    import pathlib
    import sys
    path = (pathlib.Path(__file__).resolve().parents[2]
            / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    spec = importlib.util.spec_from_file_location("_atmmatrix_duo_sigma", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("case, runner", [
    ("held_suarez", "run_held_suarez"),
    ("baroclinic", "run_baroclinic"),
    ("baroclinic_steady", "run_baroclinic"),
])
@pytest.mark.parametrize("vert", ["sigma", "hybrid"])
def test_cube_branch_selects_the_rows_vertical_table(matrix_mod, monkeypatch,
                                                     tmp_path, case, runner,
                                                     vert):
    seen = {}

    def fake(n, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(A, "build_duo_hydro", fake)
    kw = {"perturbed": False} if case == "baroclinic_steady" else {}
    tc = matrix_mod.TestCase("hydrostatic", case, "cubed_sphere", "C36",
                             vert, 1.0, 0.25, kw)
    with pytest.raises(_Stop):
        getattr(matrix_mod, runner)(tc, tmp_path, 0.25)
    if vert == "sigma":
        assert seen.get("eta") == A.DUO_SIGMA_ETA, seen
        assert seen.get("km") == matrix_mod.DEFAULT_NLEV, seen
    else:
        assert "eta" not in seen and "km" not in seen, seen


@pytest.mark.parametrize("case, runner, kw, match", [
    ("held_suarez_topo", "run_held_suarez", {}, "CAM6 L32 only"),
    ("rest_state_topo", "run_baroclinic", {}, "CAM6 L32 only"),
    ("rotated_baroclinic", "run_baroclinic", {}, "CAM6 L32 only"),
    ("inertio_gravity_3_2", "run_baroclinic", {}, "CAM6 L32 only"),
    ("gravity_wave_3_1", "run_baroclinic", {}, "own DCMIP 2012 L10"),
])
def test_sigma_label_on_a_cam6_only_cube_builder_is_refused(
        matrix_mod, monkeypatch, tmp_path, case, runner, kw, match):
    """No silent L32 substitution: a sigma-labelled row whose cube builder
    is CAM6-L32-only (e.g. via the LEGOESM_MATRIX_VCOORD probe) raises."""
    for v in ("LEGOESM_TOPO_H0", "LEGOESM_TOPO_SEED"):
        monkeypatch.delenv(v, raising=False)
    tc = matrix_mod.TestCase("hydrostatic", case, "cubed_sphere", "C36",
                             "sigma", 1.0, 0.25, kw)
    with pytest.raises(ValueError, match=match):
        getattr(matrix_mod, runner)(tc, tmp_path, 0.25)


def test_sigma_note_carries_the_other_grids_table_id(matrix_mod):
    from types import SimpleNamespace
    duo = SimpleNamespace(note="n", eta=A.DUO_SIGMA_ETA, km=NLEV)
    note = matrix_mod._duo_vertical_note("sigma", duo)
    assert matrix_mod._vertical_table_id(note) == f"sigma_L{NLEV}"
    assert "NOT honoured" not in note
