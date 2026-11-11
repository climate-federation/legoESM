"""Multilayer canopy (CLM-ML) dew limit: no canopy-air layer above saturation.

Before the limit, a humid calm night held canopy-air layers at up to 1.41x
saturation for the whole night (measured 2026-10-11).  The implicit air solve
now condenses the excess of every supersaturated layer (heat into that layer's
air, water onto its leaves), and the per-layer heat and water balances of the
solve (``ErrorCheck02``, run here) stay closed with the condensation as a source.

Run under ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLFluxProfileSolutionMod as fps
import legoesm.land.canopy.clm_ml_interface as ifc
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLWaterVaporMod import SatVap
from legoesm.land.canopy.clm_ml_interface import compute_clm_ml_canopy_fluxes
from legoesm.land.canopy.config import CLMMLCanopyConfig
from legoesm.land.config import MultiLayerLandConfig
from legoesm.thermo import saturation_specific_humidity

jax.config.update("jax_enable_x64", True)

_P = 95000.0
_NIGHT_TA = np.linspace(291.0, 286.0, 6)   # cooling humid night [K]


def _forcing(Ta, u, sw=0.0, cz=0.0):
    q = 0.97 * float(saturation_specific_humidity(jnp.asarray(Ta), jnp.asarray(_P)))
    f = lambda v: jnp.full(1, v)  # noqa: E731
    return AtmToSurface(
        sw_down=f(sw), lw_down=f(320.0), precip_total=f(0.0), precip_snow=f(0.0),
        T_lowest=f(Ta), q_lowest=f(q), u_lowest=f(u), v_lowest=f(0.0), p_lowest=f(_P),
        p_surface=f(100000.0), rho_lowest=f(1.15), cos_zenith=f(cz), co2_ppmv=f(400.0),
        has_radiation=f(1.0), has_precipitation=f(1.0))


def _night(u, eps, monkeypatch):
    """Warm-up day step, then six steps of a cooling humid night.  Returns the
    worst canopy-air e/e_sat, the summed last-sub-step condensation and its
    minimum over the night."""
    monkeypatch.setattr(fps, "DEBUG_FPS_CHECKS", True)   # per-layer balances
    ifc._last_topology_key = None
    cfg = CLMMLCanopyConfig(dew_limit_smoothing_mol=eps)
    lc = MultiLayerLandConfig(surface_scheme=cfg)
    Ts, psi, th = (jnp.full((1, 8), v) for v in (290.0, -0.5, 0.30))
    kw = dict(canopy_config=cfg, land_config=lc, land_params=None, dt=1800.0,
              T_soil=Ts, psi_soil=psi, theta_soil=th, lat=jnp.zeros(1), doy=180.0)
    _, st = compute_clm_ml_canopy_fluxes(
        T_soil_top=Ts[:, 0], forcing=_forcing(295.0, u, 400.0, 0.7),
        canopy_state=None, **kw)
    worst, cond_min, cond_sum = 0.0, 0.0, 0.0
    for Ta in _NIGHT_TA:
        out, st = compute_clm_ml_canopy_fluxes(
            T_soil_top=Ts[:, 0], forcing=_forcing(float(Ta), u), canopy_state=st, **kw)
        m = st.mlcanopy
        p = int(np.argmax(np.asarray(m.ncan_canopy)))
        n = int(np.asarray(m.ncan_canopy)[p])
        T = np.asarray(m.tair_profile)[p, 1:n + 1]
        E = np.asarray(m.eair_profile)[p, 1:n + 1]
        worst = max(worst, float(np.max(E / np.asarray(SatVap(jnp.asarray(T))[0]))))
        cond = np.asarray(m.cond_profile)[p, 1:n + 1]      # last sub-step
        cond_min, cond_sum = min(cond_min, float(cond.min())), cond_sum + float(cond.sum())
        assert np.isfinite(float(out.lhflx[0]))
    return worst, cond_sum, cond_min


@pytest.mark.parametrize("u", [0.5, 3.0])
def test_canopy_air_layers_never_supersaturate_at_the_hard_limit(u, monkeypatch):
    worst, cond_sum, cond_min = _night(u, 0.0, monkeypatch)
    # Saturation is linearised at the solve's own temperature: second order.
    assert worst <= 1.0 + 1e-6, worst          # 1.41 (u=0.5) / 1.06 before
    assert cond_sum > 0.0                      # the limit actually condensed
    assert cond_min >= -1e-15, cond_min        # never evaporates air water (round-off)


# The smoothed default (dew_limit_smoothing_mol = 1e-6 mol/mol): worst e/e_sat
# and summed condensation [mol m-2 s-1] over the night, measured 2026-11-11.
# A change to the smoothing or its default width re-measures these.
GAP = {0.5: dict(worst=9.9999996218e-01, cond_sum=2.7497314819e-03),   # hard: 1.0, 2.7496951122e-03
       3.0: dict(worst=9.9999982101e-01, cond_sum=1.6221905582e-02)}   # hard: 1.0, 1.6221406365e-02


@pytest.mark.parametrize("u", [0.5, 3.0])
def test_default_smoothing_gap_is_pinned(u, monkeypatch):
    worst, cond_sum, cond_min = _night(u, CLMMLCanopyConfig().dew_limit_smoothing_mol,
                                       monkeypatch)
    assert cond_min >= 0.0, cond_min           # the smooth form never evaporates air water
    for k, v in dict(worst=worst, cond_sum=cond_sum).items():
        assert np.isclose(v, GAP[u][k], rtol=1e-6, atol=1e-12), (k, v, GAP[u][k])


# ----------------------------------------------------- the factored solve
def _synthetic():
    """Three coupled layers: supersaturated, at saturation, subsaturated."""
    t_base = jnp.asarray([290.0, 289.0, 288.0])
    es, _ = SatVap(t_base)
    pref = 95000.0
    q_base = es / pref * jnp.asarray([1.02, 1.0, 0.97])
    q_unit = -jnp.asarray([[2.0, 0.3, 0.1], [0.3, 2.0, 0.3], [0.1, 0.3, 2.0]])
    t_unit = jnp.asarray([[50.0, 5.0, 1.0], [5.0, 50.0, 5.0], [1.0, 5.0, 50.0]])
    return t_base, q_base, t_unit, q_unit, pref


def _old_active_set(t_base, q_base, t_unit, q_unit, pref):
    """Frozen copy of the dew limit as it was before the smoothing (2026-11-11)."""
    n = t_base.shape[0]
    _eye = jnp.eye(n)
    cond = jnp.zeros(n)
    flag = jnp.zeros(n, dtype=bool)
    t_cur, q_cur = t_base, q_base
    for _ in range(n + 1):
        es_c, des_c = SatVap(t_cur)
        qs_c, dqs_c = es_c / pref, des_c / pref
        flag = (flag & (cond > 0.0)) | (q_cur > qs_c)
        m_rows = q_unit.T - dqs_c[:, None] * t_unit.T
        r_rows = (qs_c - dqs_c * t_cur) - (q_base - dqs_c * t_base)
        cond = jnp.linalg.solve(jnp.where(flag[:, None], m_rows, _eye),
                                jnp.where(flag, r_rows, 0.0))
        t_cur = t_base + t_unit.T @ cond
        q_cur = q_base + q_unit.T @ cond
    return cond, t_cur, q_cur


def test_hard_limit_is_the_old_active_set_bit_for_bit():
    args = _synthetic()
    new = fps.dew_limit_solve(*args, 0.0)
    old = _old_active_set(*args)
    for a, b in zip(new, old):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert float(new[0][0]) > 0.0 and float(new[0][2]) == 0.0   # non-vacuous


def test_default_smoothing_converges_and_is_differentiable():
    from jax.test_util import check_grads
    eps = CLMMLCanopyConfig().dew_limit_smoothing_mol
    t_base, q_base, t_unit, q_unit, pref = _synthetic()
    cond, t, q = fps.dew_limit_solve(t_base, q_base, t_unit, q_unit, pref, eps)
    # The fixed Newton passes reach the smoothed complementarity: a * s = eps^2.
    es, des = SatVap(t)
    s = es / pref - q
    m_ii = -jnp.diagonal(q_unit.T - (des / pref)[:, None] * t_unit.T)
    np.testing.assert_allclose(np.asarray(m_ii * cond * s), eps ** 2, rtol=1e-6)
    assert bool(jnp.all(cond > 0.0)) and bool(jnp.all(s > 0.0))
    f = lambda qb: fps.dew_limit_solve(t_base, qb, t_unit, q_unit, pref, eps)[0]  # noqa: E731
    check_grads(f, (q_base,), order=2, modes=["fwd", "rev"], eps=1e-3 * eps, rtol=1e-4)
