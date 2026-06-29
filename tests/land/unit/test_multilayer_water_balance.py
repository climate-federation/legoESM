"""Water-conservation gate for the multilayer (Richards) land column + a check on
the surface soil resistance.

The conservation gate was MISSING — its absence let a budget-diagnostic scare go
unverified.  ``solve_richards`` uses the mass-conservative mixed (Celia 1990) form;
this pins that: an isolated Richards step and a full multilayer step both close the
column water budget (dStorage == infiltration - transpiration - drainage, and
dStorage == precip - ET - surface_runoff - drainage) to machine tolerance, in both
a wetting and a drying regime, for stiff (clay) and mild (loam) soils.

The surface-resistance check pins the user-requested physics: bare-soil evaporation
is throttled by the TOP-layer moisture (a drying crust), so a dry surface
evaporates strictly less than with no resistance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig, RichardsConfig
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import SoilHydraulicsConfig, psi_from_theta
from legoesm.land.richards import solve_richards
from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state

_RHO = constants.rho_water


def _grid(n=8, depth=3.0):
    return make_soil_grid(SoilGridConfig(n_layers=n, total_depth=depth))


def _col(v):
    return jnp.full((1, 1), v)


# ── isolated Richards conservation (the core invariant) ─────────────────────
def _richards_residual(hyd, flux_top, sink_rate, max_iter=10):
    grid = _grid()
    dz = jnp.asarray(grid.dz)
    theta0 = jnp.full((1, 8), 0.30)
    psi0 = psi_from_theta(theta0, hyd)
    sink = jnp.full((1, 8), sink_rate) / dz[None, :]          # [1/s]
    out = solve_richards(psi0, theta0, grid, hyd, RichardsConfig(max_iter=max_iter),
                         jnp.array([flux_top]), sink, 3600.0)
    dW = float(jnp.sum(dz * (out.theta_new[0] - theta0[0])) * _RHO)
    infil = float(flux_top) * _RHO - float(out.runoff_surface[0])
    sink_tot = float(jnp.sum(sink[0] * dz)) * _RHO
    drain = float(out.runoff_subsurface[0])
    return dW - (infil - sink_tot - drain) * 3600.0


def test_richards_step_conserves_loam():
    """The mixed-form (Celia 1990) Richards step conserves for a mild loam, wetting
    and draining — the common case + the conservation gate.  Tolerance 1e-3 kg/m2 is
    float32-safe (the residual is machine-zero in float64); a real flux/sink leak is
    O(0.1-1) kg/m2/step, so it is still caught."""
    loam = SoilHydraulicsConfig()
    assert abs(_richards_residual(loam, 5.0e-6, 1.0e-7)) < 1.0e-3   # wetting
    assert abs(_richards_residual(loam, 0.0, 5.0e-7)) < 1.0e-3      # drying


def test_stiff_clay_conserves_after_specific_storage_switch():
    """A very stiff clay (n_vg~1.09) — whose drying step formerly leaked ~0.7 kg/m2
    at the default iteration count — now CONSERVES at the same max_iter=10, wetting
    and draining.  Fix: the ParFlow/CliMA specific-storage variable switch makes
    theta(psi) consistent with the capacity above saturation, so the mixed form
    conserves without the (non-conservative) theta clip that previously masked the
    inconsistency.  (Originally misdiagnosed as Picard non-convergence — it was the
    theta/capacity mismatch.)"""
    clay = SoilHydraulicsConfig(theta_r=_col(0.068), theta_sat=_col(0.38),
                                alpha_vg=_col(1.0), n_vg=_col(1.09), K_sat=_col(5.56e-8))
    assert abs(_richards_residual(clay, 0.0, 5.0e-7)) < 1.0e-3      # drying
    assert abs(_richards_residual(clay, 5.0e-6, 1.0e-7)) < 1.0e-3   # wetting


# ── full multilayer step conservation ───────────────────────────────────────
def _forcing(ncol, *, T_air, q_air, precip):
    o = jnp.ones(ncol)
    p_s = 1.0e5 * o
    return AtmToSurface(
        sw_down=250.0 * o, lw_down=330.0 * o, precip_total=precip * o,
        precip_snow=0.0 * o, T_lowest=T_air * o, q_lowest=q_air * o,
        u_lowest=4.0 * o, v_lowest=0.0 * o, p_lowest=0.99 * p_s, p_surface=p_s,
        rho_lowest=p_s / (constants.R_d * T_air), cos_zenith=0.5 * o,
        co2_ppmv=412.0 * o, has_radiation=o, has_precipitation=o)


def _full_step_budget(cfg, forcing, n_steps, dt, theta_init):
    ncol = forcing.T_lowest.shape[0]
    dz = jnp.asarray(make_soil_grid(cfg.soil_grid).dz)
    st = init_multilayer_land_state(ncol, cfg, T_init=290.0, theta_init=theta_init)

    def body(s, _):
        s2, r, _ = step_multilayer_land(s, forcing, cfg, 1.0, dt, lat=jnp.full(ncol, 0.3))
        return s2, (r.surface_mass_flux[0], s2.runoff_surface[0],
                    s2.runoff_subsurface[0], jnp.sum(dz * s2.theta_soil[0]) * _RHO,
                    jnp.min(s2.theta_soil))

    W0 = float(jnp.sum(dz * st.theta_soil[0]) * _RHO)
    _, (ET, RS, RD, W, th_min) = jax.lax.scan(body, st, None, length=n_steps)
    P = float(forcing.precip_total[0]) * dt * n_steps
    dW = float(W[-1]) - W0
    resid = dW - (P - float(jnp.sum(ET)) * dt - float(jnp.sum(RS)) * dt - float(jnp.sum(RD)) * dt)
    return resid, P, float(jnp.sum(ET)) * dt, float(jnp.min(th_min))


def test_full_step_conserves_water_drying_and_wetting():
    """dStorage == precip - ET - surface_runoff - drainage over a multi-step run,
    and no layer is driven below residual."""
    cfg = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0))
    tr = float(cfg.hydraulics.theta_r)
    for f, th0 in [(_forcing(4, T_air=305.0, q_air=0.002, precip=2.0e-5), 0.30),   # drying
                   (_forcing(4, T_air=295.0, q_air=0.012, precip=1.0e-4), 0.20)]:  # wetting
        resid, P, _ET, th_min = _full_step_budget(cfg, f, 300, 3600.0, th0)
        assert abs(resid) < 1.0e-2 * max(P, 1.0) + 1.0e-3, resid
        assert th_min >= tr - 1e-6, th_min


# ── surface soil resistance (the user-requested physics) ────────────────────
def test_surface_resistance_throttles_dry_soil_evaporation():
    """A drying surface forms a crust: bare-soil evaporation with the resistance
    (exp>0) is strictly less than with none (exp=0) from a dry top layer."""
    base = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=8, total_depth=3.0))
    f = _forcing(4, T_air=305.0, q_air=0.002, precip=0.0)
    dry = 0.10
    _, _, et_resist, _ = _full_step_budget(base._replace(soil_evap_resistance_exp=2.0),
                                           f, 60, 3600.0, dry)
    _, _, et_none, _ = _full_step_budget(base._replace(soil_evap_resistance_exp=0.0),
                                         f, 60, 3600.0, dry)
    assert et_resist < et_none, (et_resist, et_none)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
