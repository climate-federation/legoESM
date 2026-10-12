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


@pytest.mark.parametrize("u", [0.5, 3.0])
def test_canopy_air_layers_never_supersaturate(u, monkeypatch):
    """Warm-up day step, then six steps of a cooling humid night."""
    monkeypatch.setattr(fps, "DEBUG_FPS_CHECKS", True)   # per-layer balances
    ifc._last_topology_key = None
    cfg = CLMMLCanopyConfig()
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
    # Saturation is linearised at the solve's own temperature: second order.
    assert worst <= 1.0 + 1e-6, worst          # 1.41 (u=0.5) / 1.06 before
    assert cond_sum > 0.0                      # the limit actually condensed
    assert cond_min >= -1e-15, cond_min        # never evaporates air water (round-off)
