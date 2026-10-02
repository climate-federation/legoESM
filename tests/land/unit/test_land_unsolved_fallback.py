"""An unsolved but finite canopy column is accepted with energy-closed fallback fluxes.

Defect (2026-10-02, coupled AMIP): a column whose two-leaf canopy solve failed was
HELD -- state reverted, zero turbulent flux, skin frozen at its start-of-step top
soil temperature.  It restarted from the identical state, failed again, and was
measured frozen for 40 consecutive land steps (20 h) at a 342 K midday skin, with
its net radiation discarded every step.  CLM CanopyFluxes instead proceeds past its
iteration cap with the last iterate and carries the leaf energy imbalance into
sensible heat (eflx_sh_veg = ... + err).

Now: the fallback fluxes (cold state) are closed into sensible heat so that
Rn_ext = SH + LE + G exactly, and the column advances.  Carbon is held and CO2
flux is zero on such a step; the warm-start cache keeps its previous root.  Guards
revert it as before if the step would move the top soil > 20 K or report a
turbulent flux > 1000 W/m2 (user decision 2026-10-02).  Non-finite columns are
still reverted.  float64.
"""
from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.land import multilayer_land as ml
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.multilayer_land import (
    FALLBACK_MAX_FLUX_W_M2,
    FALLBACK_MAX_TOP_SOIL_CHANGE_K,
    _hold_unsolved_columns,
)
from legoesm.land.richards import psi_dry_floor
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.soil_hydraulics import theta_from_psi
from legoesm.land.state import MultiLayerLandState
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.surface_scheme.base import SurfaceFluxOutput
import legoesm.land.surface_scheme.two_leaf_canopy as tl

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="float64 energy gates; run with JAX_ENABLE_X64=1")


# --------------------------------------------------------------------------- #
# the containment decision (unit)                                              #
# --------------------------------------------------------------------------- #

_N = 4


def _state(T0, cache=None):
    return MultiLayerLandState(
        T_soil=jnp.broadcast_to(jnp.asarray(T0)[:, None], (_N, 4)) * 1.0,
        psi_soil=jnp.full((_N, 4), -1.0), theta_soil=jnp.full((_N, 4), 0.3),
        runoff_surface=jnp.zeros(_N), runoff_subsurface=jnp.zeros(_N),
        snow_depth=jnp.zeros(_N), snow_age=jnp.zeros(_N),
        surface_water=jnp.zeros(_N), canopy_x=cache)


def _response(shflx, lhflx):
    base = {f: jnp.zeros(_N) for f in TileResponse._fields}
    base.update(T_sfc=jnp.full(_N, 300.0), T_rad=jnp.full(_N, 300.0),
                albedo=jnp.full(_N, 0.25), emissivity=jnp.full(_N, 0.97),
                z0=jnp.full(_N, 0.1), q_surface=jnp.full(_N, 0.008),
                lw_up=jnp.full(_N, 450.0), co2_flux=jnp.full(_N, 1e-7),
                shflx=jnp.asarray(shflx, float), lhflx=jnp.asarray(lhflx, float))
    return TileResponse(**base)


def _forcing():
    o = jnp.ones(_N)
    return AtmToSurface(
        T_lowest=300.0 * o, q_lowest=0.005 * o, u_lowest=2.0 * o, v_lowest=0 * o,
        p_lowest=95000.0 * o, p_surface=97000.0 * o, rho_lowest=1.15 * o,
        sw_down=300.0 * o, lw_down=320.0 * o, cos_zenith=0.5 * o,
        precip_total=0 * o, precip_snow=0 * o, co2_ppmv=420.0 * o,
        has_radiation=o, has_precipitation=o)


def _sfc(converged):
    z = jnp.zeros(_N)
    return SurfaceFluxOutput(
        shflx=z, lhflx=z, tau_x=z, tau_y=z, sw_net=z, lw_net=z, lw_up=z,
        G_soil=z, T_surface=z + 300.0, q_surface=z, albedo=z + 0.2,
        emissivity=z + 0.97, z0=z + 0.1, converged=jnp.asarray(converged))


_CFG = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=4, total_depth=2.0))


def _decide(new_T, shflx, lhflx, converged, new_cache=None, fallback_ok=True):
    old_cache = jnp.tile(jnp.arange(6.0)[None, :], (_N, 1))
    old = _state([300.0] * _N, cache=old_cache)
    new = _state(new_T, cache=(jnp.full((_N, 6), jnp.nan) if new_cache is None
                               else new_cache))
    c_old = {"C_fol": jnp.full(_N, 100.0)}
    c_new = {"C_fol": jnp.full(_N, 101.0)}
    out = _hold_unsolved_columns(
        old, new, _response(shflx, lhflx), _sfc(converged), _forcing(), _CFG, _N,
        carbon_old=c_old, carbon_new=c_new, fallback_ok=fallback_ok)
    return old, new, out


def test_a_finite_unsolved_column_is_accepted_and_the_guards_revert_the_rest():
    # col 0 converged; col 1 unsolved, moderate -> ACCEPTED;
    # col 2 unsolved, finite but SH/LH -2040/-3231 W/m2 (the old guard fixture)
    # -> flux guard reverts it; col 3 unsolved, top soil +25 K -> T guard reverts.
    new_T = [301.0, 290.0, 299.0, 300.0 + FALLBACK_MAX_TOP_SOIL_CHANGE_K + 5.0]
    old, new, (st, resp, carbon, held, n_held, fb, n_fb, rej, n_rej) = _decide(
        new_T, [40.0, -150.0, -2040.0, 30.0], [60.0, 20.0, -3231.0, 10.0],
        [True, False, False, False])
    assert [bool(v) for v in fb] == [False, True, False, False]
    assert [bool(v) for v in held] == [False, False, True, True]
    assert (int(n_fb), int(n_held), int(n_rej)) == (1, 2, 2)
    assert [bool(v) for v in rej] == [False, False, True, True]
    # accepted column advanced, kept its fluxes
    assert float(st.T_soil[1, 0]) == 290.0
    assert float(resp.shflx[1]) == -150.0 and float(resp.lhflx[1]) == 20.0
    # reverted columns: old state, no exchange
    for c in (2, 3):
        assert float(st.T_soil[c, 0]) == 300.0
        assert float(resp.shflx[c]) == 0.0 and float(resp.lhflx[c]) == 0.0
    # converged column untouched
    assert float(st.T_soil[0, 0]) == 301.0 and float(resp.shflx[0]) == 40.0
    assert abs(FALLBACK_MAX_FLUX_W_M2 - 1000.0) < 1e-12


def test_fallback_holds_carbon_zeroes_co2_and_keeps_the_previous_root():
    # Contract: on the old code (column fully held) these values also hold, so
    # this pins the D3 / cache contract of the new path rather than the fix.
    _old, _new, (st, resp, carbon, *_rest) = _decide(
        [301.0, 295.0, 301.0, 301.0], [40.0] * 4, [60.0] * 4,
        [True, False, True, True])
    assert float(carbon["C_fol"][1]) == 100.0
    assert float(carbon["C_fol"][0]) == 101.0
    assert float(resp.co2_flux[1]) == 0.0 and float(resp.co2_flux[0]) == 1e-7
    np.testing.assert_array_equal(np.asarray(st.canopy_x[1]), np.arange(6.0))


def test_a_nan_cache_is_the_cold_start_sentinel_but_inf_is_a_failure():
    cache = jnp.full((_N, 6), jnp.nan).at[2, 0].set(jnp.inf).at[0].set(1.0)
    _o, _n, (_st, _r, _c, held, _nh, fb, _nfb, _rj, _nr) = _decide(
        [301.0, 295.0, 296.0, 301.0], [40.0] * 4, [60.0] * 4,
        [True, False, False, True], new_cache=cache)
    assert [bool(v) for v in fb] == [False, True, False, False]
    assert [bool(v) for v in held] == [False, False, True, False]


def test_without_fallback_ok_every_unsolved_column_is_still_reverted():
    _o, _n, (st, _r, _c, held, n_held, fb, n_fb, _rj, n_rej) = _decide(
        [301.0, 295.0, 301.0, 301.0], [40.0] * 4, [60.0] * 4,
        [True, False, True, True], fallback_ok=False)
    assert int(n_held) == 1 and bool(held[1]) and int(n_fb) == 0 and int(n_rej) == 0
    assert float(st.T_soil[1, 0]) == 300.0


def test_a_carbon_structure_change_reverts_the_fallback_column():
    # Carrier rebuilt this step: the pools cannot be kept, so the column cannot
    # be accepted under D3 and is reverted (and counted as guard-rejected).
    old = _state([300.0] * _N)
    new = _state([301.0, 295.0, 301.0, 301.0])
    out = _hold_unsolved_columns(
        old, new, _response([40.0] * 4, [60.0] * 4),
        _sfc([True, False, True, True]), _forcing(), _CFG, _N,
        carbon_old={"C_fol": jnp.full(_N, 100.0)},
        carbon_new={"C_fol": jnp.full(_N, 101.0), "C_root": jnp.full(_N, 5.0)},
        fallback_ok=True)
    st, _r, _c, held, n_held, fb, n_fb, rej, n_rej = out
    assert int(n_fb) == 0 and bool(held[1]) and bool(rej[1]) and int(n_rej) == 1
    assert float(st.T_soil[1, 0]) == 300.0


# --------------------------------------------------------------------------- #
# the integrated land step                                                     #
# --------------------------------------------------------------------------- #

_TA_SPLIT = 300.5     # [K] columns whose air is warmer than this are forced unsolved


@pytest.fixture
def forced_unsolved(monkeypatch):
    """Column 1 (air 301 K) reports converged=False; column 0 (299 K) is the real
    solve.  The two-leaf scheme then evaluates column 1 at its cold state."""
    real = tl.solve_canopy_closure_diag

    def diag(x0, bun, cc):
        x, n, conv, *rest = real(x0, bun, cc)
        return (x, n, conv & (bun.Ta < _TA_SPLIT), *rest)

    monkeypatch.setattr(tl, "solve_canopy_closure_diag", diag)


def _forcing2(sw, lw=380.0):
    o = jnp.ones(2)
    T = jnp.array([299.0, 301.0])
    ps = 1e5 * o
    return AtmToSurface(
        sw_down=sw * o, lw_down=lw * o, precip_total=0 * o, precip_snow=0 * o,
        T_lowest=T, q_lowest=0.002 * o, u_lowest=5.0 * o, v_lowest=0 * o,
        p_lowest=0.99 * ps, p_surface=ps, rho_lowest=ps / (constants.R_d * T),
        cos_zenith=(0.8 if sw > 0 else 0.0) * o, co2_ppmv=412 * o,
        has_radiation=o, has_precipitation=o)


def _setup(T_init, n_layers=4, depth=2.0):
    cfg = MultiLayerLandConfig(
        soil_grid=SoilGridConfig(n_layers=n_layers, total_depth=depth),
        surface_scheme=TwoLeafCanopyConfig())
    tfl = float(jnp.max(theta_from_psi(psi_dry_floor(cfg.hydraulics), cfg.hydraulics)))
    st = ml.init_multilayer_land_state(2, cfg, T_init=T_init, theta_init=tfl + 0.05)
    return cfg, st


def test_a_clipped_fallback_is_energy_closed_and_the_column_advances(
        forced_unsolved, monkeypatch):
    """Hot soil (340 K) under 301 K air at night: the cold-state ground flux is
    clipped to -500 W/m2, so the fallback fluxes do not close by themselves.
    The fold must put the clipped remainder into sensible heat so Rn_ext =
    SH + LE + G for the column (measured: the unclipped cold-state SH is ~4.2e3
    W/m2 and the fold, ~-3.9e3, brings the EXPORTED SH to ~270 W/m2, inside the
    1000 W/m2 guard), and the column must ADVANCE over four steps.  Reverted
    (old hold): T_soil frozen, n_held = 1 per step."""
    cfg, st = _setup(340.0)
    f = _forcing2(sw=0.0)
    calls = []
    real = ml.solve_soil_thermal

    sig = inspect.signature(real)

    def spy(*a, **k):
        calls.append(sig.bind(*a, **k).arguments["G_surface"])
        return real(*a, **k)

    monkeypatch.setattr(ml, "solve_soil_thermal", spy)

    def body(s, _):
        calls.clear()
        new, r, _c, so = ml.step_multilayer_land_with_diagnostics(
            s, f, cfg, 1.0, 1800.0, lat=jnp.full(2, 0.3))
        X = so.lhflx - r.lhflx
        fold = r.shflx - so.shflx - X
        E = so.Rn_ext - r.shflx - r.lhflx - calls[-1]
        return new, (new.T_soil[:, 0], so.n_held, so.n_fallback, fold, E,
                     so.converged)

    _, (T_top, n_held, n_fb, fold, E, conv) = jax.jit(
        lambda s: jax.lax.scan(body, s, None, length=4))(st)
    assert not bool(jnp.any(conv[:, 1])) and bool(jnp.all(conv[:, 0]))
    assert int(jnp.sum(n_held)) == 0, "the unsolved column was reverted"
    assert int(jnp.sum(n_fb)) == 4
    assert float(jnp.abs(fold[0, 1])) > 5.0, fold        # non-vacuity: clip active
    assert float(jnp.max(jnp.abs(E[:, 1]))) < 1e-9, E
    T1 = np.concatenate([[340.0], np.asarray(T_top[:, 1])])
    assert np.all(np.abs(np.diff(T1)) > 1e-3), T1          # it advances every step
    assert np.all(np.abs(np.diff(T1)) < FALLBACK_MAX_TOP_SOIL_CHANGE_K)


def test_the_fallback_flux_carries_a_gradient(forced_unsolved):
    """A held column exports zero flux, so its gradient is exactly zero; an
    accepted fallback column's sensible heat responds to the forcing."""
    cfg, st = _setup(305.0, n_layers=8, depth=3.0)

    def sh(lw):
        _, r, _c, so = ml.step_multilayer_land_with_diagnostics(
            st, _forcing2(sw=600.0, lw=lw), cfg, 1.0, 1800.0, lat=jnp.full(2, 0.3))
        return r.shflx[1], so.n_fallback

    (v, nfb), g = jax.jit(jax.value_and_grad(sh, has_aux=True))(380.0)
    assert int(nfb) == 1
    assert np.isfinite(float(v)) and np.isfinite(float(g))
    assert abs(float(g)) > 1e-3, g
