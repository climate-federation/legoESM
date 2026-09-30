"""Evaporation / root uptake reported to the atmosphere = water the soil actually gave.

Defect (2026-09-29, coupled AMIP deserts): a draw the dry column could not supply
ended at the Richards psi dry floor, which refilled it, while the full demand was
still reported as evaporation (+0.46 mm/d of created water over the Sahara).
``solve_richards`` now returns that refill (``water_created``) and the land step
reports the demand minus it.  Budget gates are float64 (JAX_ENABLE_X64=1); each
multi-step case runs as ONE jitted scan (eager land steps exhaust LLVM memory).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig, RichardsConfig
from legoesm.land.multilayer_land import init_multilayer_land_state, step_multilayer_land
from legoesm.land.richards import psi_dry_floor, solve_richards
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta, theta_from_psi
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig
from legoesm.land.canopy.interception import InterceptionConfig

_RHO = constants.rho_water
_DAY = 86400.0

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="float64 budget gates; run with JAX_ENABLE_X64=1")


def _theta_floor(hyd):
    return float(jnp.max(theta_from_psi(psi_dry_floor(hyd), hyd)))


@pytest.mark.parametrize("where", ["top", "roots"])
def test_richards_returns_the_floor_refill(where):
    """A 2 mm/day draw on a column 5e-4 above its dry floor: much of it cannot be
    supplied; the solver returns that refill, and the column loses exactly the
    imposed draw minus the refill (what the caller may report)."""
    hyd = SoilHydraulicsConfig()
    grid = make_soil_grid(SoilGridConfig(n_layers=8, total_depth=3.0))
    dz = jnp.asarray(grid.dz)
    theta = jnp.full((1, 8), _theta_floor(hyd) + 5.0e-4)
    psi = psi_from_theta(theta, hyd)
    dt, e = 1800.0, 2.0e-3 / _DAY                       # [s], [m/s]
    flux = jnp.array([-e if where == "top" else 0.0])
    sink = (jnp.zeros((1, 8)) if where == "top"
            else jnp.full((1, 8), e / 3.0).at[:, 3:].set(0.0) / dz[None, :])
    out = solve_richards(psi, theta, grid, hyd, RichardsConfig(bottom_bc="zero_flux"),
                         flux, sink, dt, surface_water=jnp.zeros(1))
    imposed = float(flux[0] - jnp.sum(sink[0] * dz)) * dt      # [m], < 0
    dW = float(jnp.sum((out.theta_new[0] - theta[0]) * dz) + out.surface_water[0])
    refill = float(out.refill[0])
    assert refill > 0.1 * (-imposed), (refill, imposed)        # the floor refilled
    assert refill <= -imposed * (1 + 1e-12)                    # never beyond the draw
    assert abs(dW - (imposed + refill)) < 1e-15
    assert abs(float(out.water_created[0])) < 1e-15


@pytest.mark.parametrize("curve", ["van_genuchten", "pdi", "lu"])
@pytest.mark.parametrize("theta0,flux,pond", [
    (0.43, 1.0e-4, 0.0),      # heavy rain on a saturated column
    (0.43, -1.0e-7, 0.02),    # ponded, evaporating
    (0.30, 5.0e-6, 0.0),      # wetting front
    (None, 2.0e-9, 0.0),      # drizzle on a floor-pinned dry column (debit path)
])
def test_returned_psi_theta_consistent_on_wet_columns(theta0, flux, pond, curve):
    """The non-draw debit touches only unsaturated layers, so saturated / ponded
    columns keep (psi, theta) on the retention curve (incl. positive pressure),
    and the debit conserves water on every curve.  On the debit path (psi from
    theta) consistency is exact only where the inverse is (van Genuchten); PDI /
    Lu invert approximately, a pre-existing limitation."""
    hyd = SoilHydraulicsConfig(retention_curve=curve)
    grid = make_soil_grid(SoilGridConfig(n_layers=8, total_depth=3.0))
    theta = jnp.full((1, 8), _theta_floor(hyd) + 1.0e-4 if theta0 is None
                     else min(theta0, float(jnp.max(hyd.theta_sat))))
    psi = psi_from_theta(theta, hyd)
    out = solve_richards(psi, theta, grid, hyd, RichardsConfig(), jnp.array([flux]),
                         jnp.zeros((1, 8)), 1800.0, surface_water=jnp.full(1, pond))
    # Debit path conserves (pre-fix reset of theta from psi lost 2.9e-4 m for PDI).
    # Lu excluded: its state/floor sit off its own approximate inverse, so the
    # solve itself does not close there (pre-existing, not the debit).
    if theta0 is None and curve != "lu":
        assert abs(float(out.water_created[0])) < 1e-15, float(out.water_created[0])
    if theta0 is not None or curve == "van_genuchten":
        np.testing.assert_allclose(np.asarray(theta_from_psi(out.psi_new, hyd)),
                                   np.asarray(out.theta_new), rtol=0, atol=1e-10)

    def total_psi(f):
        o = solve_richards(psi, theta, grid, hyd, RichardsConfig(), jnp.array([f]),
                           jnp.zeros((1, 8)), 1800.0, surface_water=jnp.full(1, pond))
        return jnp.sum(o.psi_new) + jnp.sum(o.theta_new)

    assert np.isfinite(float(jax.grad(total_psi)(flux)))


def _forcing(n, T=310.0, q=0.002, P=0.0, sw=600.0):
    o = jnp.ones(n)
    ps = 1e5 * o
    return AtmToSurface(
        sw_down=sw * o, lw_down=380.0 * o, precip_total=P * o, precip_snow=0 * o,
        T_lowest=T * o, q_lowest=q * o, u_lowest=5.0 * o, v_lowest=0 * o,
        p_lowest=0.99 * ps, p_surface=ps, rho_lowest=ps / (constants.R_d * T),
        cos_zenith=(0.8 if sw > 0 else 0.0) * o, co2_ppmv=412 * o, has_radiation=o,
        has_precipitation=o)


def _day(scheme, top, deep, n_steps=48, dt=1800.0, precip=0.0, state_dtype=None,
         sw=600.0, bottom_bc="zero_flux", **cfg_kw):
    """One jitted scan of land steps; per-step residual [kg/m2] of
    dStorage(soil + pond + canopy store) - (P - E_reported - runoff), total E
    [kg/m2], max |lhflx - L_v * mass|."""
    cfg = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0),
                               richards=RichardsConfig(bottom_bc=bottom_bc),
                               surface_scheme=scheme, **cfg_kw)
    dz = jnp.asarray(make_soil_grid(cfg.soil_grid).dz)
    tfl = _theta_floor(cfg.hydraulics)
    st = init_multilayer_land_state(2, cfg, T_init=305.0, theta_init=tfl + deep)
    th = st.theta_soil.at[:, 0].set(tfl + top)
    st = st._replace(theta_soil=th, psi_soil=psi_from_theta(th, cfg.hydraulics))
    if cfg.interception is not None and st.W_canopy is None:
        st = st._replace(W_canopy=jnp.zeros(2))
    if state_dtype is not None:   # coupled lane: float32 land state, x64 solves
        st = jax.tree.map(lambda x: x.astype(state_dtype)
                          if getattr(x, "dtype", None) == jnp.float64 else x, st)
    f, lat = _forcing(2, P=precip, sw=sw), jnp.full(2, 0.3)

    def W(s):
        wc = 0.0 if s.W_canopy is None else s.W_canopy[0]
        return (jnp.sum(dz * s.theta_soil[0]) + s.surface_water[0]) * _RHO + wc

    def body(s, _):
        s2, r, _ = step_multilayer_land(s, f, cfg, 1.0, dt, lat=lat)
        e = r.surface_mass_flux[0] * dt
        ro = (s2.runoff_surface[0] + s2.runoff_subsurface[0]) * dt
        return s2, (W(s2) - W(s) - (precip * dt - e - ro), e,
                    r.lhflx[0] - r.surface_mass_flux[0] * constants.L_v)

    _, (res, e, gap) = jax.jit(lambda s: jax.lax.scan(body, s, None, length=n_steps))(st)
    return np.asarray(res), float(jnp.sum(e)), float(jnp.max(jnp.abs(gap)))


@pytest.mark.parametrize("scheme,cfg_kw", [
    (TwoLeafCanopyConfig(), {}),
    (SimpleSEBConfig(), {"soil_evap_resistance_exp": 0.0}),
], ids=["two_leaf", "simple_seb_no_crust"])
def test_dry_column_reports_only_water_it_loses(scheme, cfg_kw):
    """A hot dry day on a column 5e-3 above its dry floor: every step, the water
    the land reports to the atmosphere is exactly what the column lost, and the
    reported latent heat is L_v times it.  (Before the fix the two-leaf column
    reported ~0.04 kg/m2/day it never held.)"""
    res, et, gap = _day(scheme, top=5.0e-3, deep=5.0e-3, **cfg_kw)
    assert et > 0.0
    assert np.abs(res).max() < 1e-10, np.abs(res).max()
    assert gap < 1e-9


@pytest.mark.parametrize("precip,cfg_kw", [
    (5.0e-5, {}),
    (2.0e-6, {"interception": InterceptionConfig()}),
    (2.0e-6, {"sw": 0.0}),
    (5.0e-5, {"bottom_bc": "free_drainage"}),
], ids=["rain", "drizzle_interception", "drizzle_at_night", "rain_free_drainage"])
def test_rain_on_a_dry_column_creates_no_water(precip, cfg_kw):
    """Rain / drizzle on a column 3e-3 above its dry floor.  Before the fix the
    drizzle case created 0.63 kg/m2/day: the refill of the evaporative draws AND
    Picard creation on the floor-pinned top layer (water the solve put INTO the
    soil, now taken back out of it)."""
    res, et, gap = _day(TwoLeafCanopyConfig(), top=3.0e-3, deep=3.0e-3,
                        precip=precip, **cfg_kw)
    assert np.isfinite(et)
    assert np.abs(res).max() < 1e-10, np.abs(res).max()
    assert gap < 1e-9


def test_float32_state_dry_column():
    """Coupled-lane precision (float32 land state, float64 solves): the dry column
    closes to float32 storage roundoff (pre-fix: 0.038 kg/m2/day created)."""
    res, et, _ = _day(TwoLeafCanopyConfig(), top=5.0e-3, deep=5.0e-3,
                      state_dtype=jnp.float32)
    assert et > 0.0
    assert abs(res.sum()) < 1e-4, res.sum()
    assert np.abs(res).max() < 1e-5, np.abs(res).max()


def test_moist_subsoil_keeps_capillary_supply():
    """A dry top layer over a moist subsoil: the column CAN supply the demand from
    below, nothing is refilled, and the day's ET is unchanged (the correction must
    not cut legitimate upward supply).  Regression guard."""
    res, et, _ = _day(TwoLeafCanopyConfig(), top=2.0e-4, deep=0.12)
    # Pinned to the pre-fix model (origin/main aade0a444), which created no water
    # in this regime: the fix must leave it unchanged.  A start-of-step cap on
    # the top layer (rejected design) cut it by ~4%.
    assert abs(et - 0.9565067983140837) < 1e-6 * et, et
    assert np.abs(res).max() < 1e-10


def test_jit_and_grad_finite_on_a_dry_column():
    cfg = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0),
                               surface_scheme=TwoLeafCanopyConfig())
    tfl = _theta_floor(cfg.hydraulics)
    st0 = init_multilayer_land_state(2, cfg, T_init=305.0, theta_init=tfl + 5.0e-3)
    f = _forcing(2)

    def et_of(dtop):
        th = st0.theta_soil.at[:, 0].set(tfl + dtop)
        st = st0._replace(theta_soil=th, psi_soil=psi_from_theta(th, cfg.hydraulics))
        _, r, _ = step_multilayer_land(st, f, cfg, 1.0, 1800.0, lat=jnp.full(2, 0.3))
        return jnp.sum(r.surface_mass_flux)

    v, g = jax.jit(jax.value_and_grad(et_of))(5.0e-3)
    assert np.isfinite(float(v)) and np.isfinite(float(g))
