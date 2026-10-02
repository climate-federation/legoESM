"""Unmet evaporation energy leaves as sensible heat, not into the top soil layer.

Defect (2026-10-02, coupled AMIP deserts): latent energy the surface solve spent
on evaporation the reservoirs could not supply was added to the ground heat flux
of the ~3 mm top soil layer after the skin temperature was solved -- 85-160 W/m2
at midday over NW Australia, top soil 336-351 K, and those columns then failed
their canopy solve and were held.  CLM/CTSM (SoilFluxesMod, "conserve total
energy flux") moves that energy to ground sensible heat instead; the soil heat
flux is unchanged.  Three sources, each tested separately (each forces X > 10 W/m2):

* Richards dry-floor refill (two-leaf canopy, dry column);
* SimpleSEB bare-soil resistance (dry column, no snow);
* snow-pack sublimation cap (thin pack, cold dry air).

Each fails with the fix reverted: the ground flux handed to the final soil solve
then carries X and the exported sensible heat does not.  float64.
"""
from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land import multilayer_land as ml
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.richards import psi_dry_floor
from legoesm.land.soil_grid import SoilGridConfig
from legoesm.land.soil_hydraulics import theta_from_psi
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig
from legoesm.thermo import latent_heat_sublimation, latent_heat_vaporization

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="float64 energy gates; run with JAX_ENABLE_X64=1")

_X_MIN = 10.0     # [W m-2] non-vacuity: the step must actually have unmet demand


def _forcing(n, T=310.0, q=0.002, sw=600.0):
    o = jnp.ones(n)
    ps = 1e5 * o
    return AtmToSurface(
        sw_down=sw * o, lw_down=380.0 * o, precip_total=0 * o, precip_snow=0 * o,
        T_lowest=T * o, q_lowest=q * o, u_lowest=5.0 * o, v_lowest=0 * o,
        p_lowest=0.99 * ps, p_surface=ps, rho_lowest=ps / (constants.R_d * T),
        cos_zenith=(0.8 if sw > 0 else 0.0) * o, co2_ppmv=412 * o, has_radiation=o,
        has_precipitation=o)


def _step(monkeypatch, scheme, top, T_init=305.0, snow=None, forcing=None):
    """One jitted land step; returns (state0, new, response, surface_out, G_final)
    where G_final is the ground flux handed to the FINAL soil-thermal solve."""
    cfg = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0),
                               surface_scheme=scheme)
    tfl = float(jnp.max(theta_from_psi(psi_dry_floor(cfg.hydraulics), cfg.hydraulics)))
    st = ml.init_multilayer_land_state(2, cfg, T_init=T_init, theta_init=tfl + top)
    if snow is not None:
        st = st._replace(snow_depth=jnp.full(2, snow))
    f = _forcing(2) if forcing is None else forcing
    calls = []
    real = ml.solve_soil_thermal

    sig = inspect.signature(real)

    def spy(*a, **k):
        calls.append(sig.bind(*a, **k).arguments["G_surface"])
        return real(*a, **k)

    monkeypatch.setattr(ml, "solve_soil_thermal", spy)

    def step(s):
        calls.clear()
        new, r, _c, so = ml.step_multilayer_land_with_diagnostics(
            s, f, cfg, 1.0, 1800.0, lat=jnp.full(2, 0.3))
        return new, r, so, calls[-1]

    new, r, so, G = jax.jit(step)(st)
    return st, new, r, so, G


def _check(st, new, r, so, G):
    X = np.asarray(so.lhflx - r.lhflx)              # unmet latent energy [W m-2]
    assert np.all(X > _X_MIN), X                     # non-vacuity
    assert int(so.n_held) == 0
    # The soil receives the scheme's own ground flux -- no X in it.
    np.testing.assert_allclose(np.asarray(G), np.asarray(so.G_soil), rtol=0, atol=1e-9)
    # X leaves as sensible heat to the air.
    np.testing.assert_allclose(np.asarray(r.shflx - so.shflx), X, rtol=0, atol=1e-9)
    assert np.all(np.isfinite(np.asarray(new.T_soil)))


def test_dry_soil_refill_energy_goes_to_sensible_heat(monkeypatch):
    """Two-leaf canopy on a column 1e-3 above its dry floor: the Richards floor
    refills part of the draw (measured X = 19 W/m2 in this step)."""
    st, new, r, so, G = _step(monkeypatch, TwoLeafCanopyConfig(), top=1.0e-3)
    assert bool(jnp.all(so.converged))
    _check(st, new, r, so, G)
    # land boundary, two-leaf: Rn_ext = SH + LE + G to the scheme's own residual
    E = so.Rn_ext - r.shflx - r.lhflx - G
    np.testing.assert_allclose(np.asarray(E), np.asarray(so.residual_ext),
                               rtol=0, atol=1e-9)
    # latent heat = L_v(T_surface) x delivered mass (start-of-step top-soil T)
    L_v = latent_heat_vaporization(st.T_soil[:, 0])
    np.testing.assert_allclose(np.asarray(r.surface_mass_flux * L_v),
                               np.asarray(r.lhflx), rtol=0, atol=1e-9)


def test_bare_soil_resistance_energy_goes_to_sensible_heat(monkeypatch):
    """SimpleSEB, no snow: the bare-soil resistance cuts the delivered
    evaporation below the scheme's demand (measured X = 107 W/m2)."""
    st, new, r, so, G = _step(monkeypatch, SimpleSEBConfig(), top=5.0e-3)
    assert float(jnp.max(st.snow_depth)) == 0.0
    _check(st, new, r, so, G)


def test_snow_sublimation_cap_energy_goes_to_sensible_heat(monkeypatch):
    """SimpleSEB over a 1e-3 kg/m2 pack in cold dry air: sublimation demand far
    exceeds the pack (measured X = 77 W/m2); the whole pack leaves, the excess
    leaves as sensible heat.  No melt (sub-freezing), so G is the scheme's."""
    st, new, r, so, G = _step(
        monkeypatch, SimpleSEBConfig(), top=0.1, T_init=265.0, snow=1.0e-3,
        forcing=_forcing(2, T=268.0, q=0.0005, sw=300.0))
    _check(st, new, r, so, G)
    # the cap bound: the pack is gone and exactly it left as vapour
    assert float(jnp.max(new.snow_depth)) < 1e-12
    np.testing.assert_allclose(np.asarray(r.surface_mass_flux * 1800.0),
                               1.0e-3, rtol=1e-9)
    np.testing.assert_allclose(
        np.asarray(r.lhflx),
        np.asarray(1.0e-3 / 1800.0 * latent_heat_sublimation(st.T_soil[:, 0])),
        rtol=1e-9)
