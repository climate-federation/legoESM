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
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig, RichardsConfig
from legoesm.land.soil_grid import SoilGridConfig, make_soil_grid
from legoesm.land.soil_hydraulics import (
    SoilHydraulicsConfig, psi_from_theta, theta_from_psi, hydraulic_conductivity)
from legoesm.land.richards import solve_richards
from legoesm.land.multilayer_land import step_multilayer_land, init_multilayer_land_state

_RHO = constants.rho_water


def _grid(n=8, depth=3.0):
    return make_soil_grid(SoilGridConfig(n_layers=n, total_depth=depth))


def _col(v):
    return jnp.full((1, 1), v)


# ── isolated Richards conservation (the core invariant) ─────────────────────
def _richards_residual(hyd, flux_top, sink_rate, max_iter=10, pond0_val=0.0,
                       theta_init=0.30):
    grid = _grid()
    dz = jnp.asarray(grid.dz)
    theta0 = jnp.full((1, 8), theta_init)
    psi0 = psi_from_theta(theta0, hyd)
    sink = jnp.full((1, 8), sink_rate) / dz[None, :]          # [1/s]
    pond0 = jnp.full((1,), pond0_val)
    out = solve_richards(psi0, theta0, grid, hyd, RichardsConfig(max_iter=max_iter),
                         jnp.array([flux_top]), sink, 3600.0, surface_water=pond0)
    # total storage = soil + surface pond
    dW = float((jnp.sum(dz * (out.theta_new[0] - theta0[0]))
                + (out.surface_water[0] - pond0[0])) * _RHO)
    sink_tot = float(jnp.sum(sink[0] * dz)) * _RHO
    src = (float(flux_top) * _RHO - sink_tot
           - float(out.runoff_subsurface[0]) - float(out.runoff_surface[0]))
    return dW - src * 3600.0


def test_richards_step_conserves_loam():
    """The mixed-form (Celia 1990) Richards step conserves for a mild loam, wetting
    and draining — the common case + the conservation gate.  Tolerance 1e-3 kg/m2 is
    float32-safe (the residual is machine-zero in float64); a real flux/sink leak is
    O(0.1-1) kg/m2/step, so it is still caught."""
    loam = SoilHydraulicsConfig()
    assert abs(_richards_residual(loam, 5.0e-6, 1.0e-7)) < 1.0e-3   # wetting
    assert abs(_richards_residual(loam, 0.0, 5.0e-7)) < 1.0e-3      # drying


def test_surface_cell_handles_ksat_shapes():
    """The surface cell's top-layer K_sat extraction must handle every config shape
    (scalar, per-column (ncol,1), full (ncol,nlayers)) without mis-mapping layer
    variation onto columns (codex) — and still conserve.  Uses ncol != nlayers so a
    wrong reshape/broadcast would raise or leak."""
    grid = _grid()                       # 8 layers
    dz = jnp.asarray(grid.dz)
    ncol, nlayers = 3, 8
    base = SoilHydraulicsConfig()
    theta0 = jnp.full((ncol, nlayers), 0.30)
    shapes = {
        "scalar": base,
        "per_column": base._replace(K_sat=jnp.full((ncol, 1), float(base.K_sat))),
        "full_field": base._replace(K_sat=jnp.full((ncol, nlayers), float(base.K_sat))),
        # layer-varying (nlayers,): exercises BOTH the top cell and the bottom
        # free-drainage K (codex's exact mis-broadcast scenario)
        "per_layer": base._replace(K_sat=jnp.full((nlayers,), float(base.K_sat))),
    }
    for name, hyd in shapes.items():
        psi0 = psi_from_theta(theta0, hyd)
        pond0 = jnp.zeros(ncol)
        out = solve_richards(psi0, theta0, grid, hyd, RichardsConfig(),
                             jnp.full(ncol, 5.0e-6), jnp.zeros((ncol, nlayers)),
                             3600.0, surface_water=pond0)
        dW = (jnp.sum(dz[None, :] * (out.theta_new - theta0), axis=-1)
              + (out.surface_water - pond0)) * _RHO
        src = (5.0e-6 * _RHO - out.runoff_subsurface - out.runoff_surface) * 3600.0
        assert jnp.max(jnp.abs(dW - src)) < 1.0e-3, (name, float(jnp.max(jnp.abs(dW - src))))


def test_evaporation_conserves_with_and_without_pond():
    """Net surface evaporation (flux_top < 0) conserves whether or not a pond exists.

    With no pond the available-supply cap is negative, so the surface<->soil flux q01
    goes negative: the soil supplies the bare-soil evaporative demand and the coupled
    cell reduces to the old Neumann flux_top BC.  With a standing pond the pond drains
    to the atmosphere first.  Both close the soil+pond budget to machine tolerance (the
    codex check on the _avail_rate < 0 branch).

    Precondition: flux_top must be physically realizable — an evaporative demand that
    exceeds the top layer's available water would drive it to theta_r, where the van-
    Genuchten asymptote floors theta and the (fixed Neumann) evap flux leaks O(0.03)
    kg/m2 (a PRE-EXISTING solver property the old flux_infiltrated BC shared, not the
    coupled cell).  The solver returns that refill as ``water_created`` and the land
    step reports only the water the soil gave (test_land_evap_supply_limit.py).  So
    this isolated-solver gate uses a demand the column can supply."""
    loam = SoilHydraulicsConfig()
    evap = -5.0e-7  # net upward surface water flux [m/s], within column supply
    assert abs(_richards_residual(loam, evap, 0.0)) < 1.0e-3                 # dry surface
    assert abs(_richards_residual(loam, evap, 0.0, pond0_val=0.02)) < 1.0e-3  # standing pond


def test_richards_drainage_reported_at_solver_debited_K():
    """Reported subsurface drainage must be the K the last Picard iteration's rhs
    actually DEBITED (the free-drainage bottom BC is explicit, evaluated at the
    carry entering that iteration) — not K re-evaluated at psi_final.  The two
    differ by O(last dpsi), and that mismatch showed up 1:1 as a spurious column
    budget residual: on this draining wet column, drainage-at-psi_final leaves
    ~6.1e-6 kg/m2/step while the solve-consistent report leaves ~4.6e-8 kg/m2
    (only the last iteration's O(dpsi^2) linearization error remains).  The soil
    STATE is untouched by the fix (bit-identical psi/theta/pond/runoff_surface,
    probe-verified in x64 and float32); only the diagnostic moved.

    The 5e-7 budget gate is an fp64 tolerance (the Picard residual floor is well
    below fp32 roundoff on this column), so this node runs under x64; the fp32
    carry-dtype path is exercised separately by
    ``test_richards_free_drainage_runs_under_fp32_policy`` in the x64-off lane."""
    if not jax.config.read("jax_enable_x64"):
        pytest.skip("budget-closure gate needs fp64; run this node with "
                    "JAX_ENABLE_X64=1")
    loam = SoilHydraulicsConfig()
    theta_wet = float(loam.theta_sat) - 0.005

    # (a) budget gate: draining wet column closes far below the pre-fix mismatch.
    resid = _richards_residual(loam, 0.0, 0.0, theta_init=theta_wet)
    assert abs(resid) < 5.0e-7, resid   # pre-fix (K at psi_final): ~6.1e-6 kg/m2

    # (b) semantics pin: the reported drainage equals K at the carry ENTERING the
    # last iteration — recoverable as the raw psi of a (max_iter-1) run, since the
    # Picard body is deterministic (the layer-0 pond-deficit debit only touches the
    # RETURNED theta, so recompute theta from psi).  And it must NOT be K(psi_final)
    # (the old evaluation) on this still-draining column.
    grid = _grid()
    theta0 = jnp.full((1, 8), theta_wet)
    psi0 = psi_from_theta(theta0, loam)
    flux = jnp.zeros(1)
    sink = jnp.zeros((1, 8))
    pond0 = jnp.zeros(1)
    out = solve_richards(psi0, theta0, grid, loam, RichardsConfig(max_iter=10),
                         flux, sink, 3600.0, surface_water=pond0)
    out_m1 = solve_richards(psi0, theta0, grid, loam, RichardsConfig(max_iter=9),
                            flux, sink, 3600.0, surface_water=pond0)
    K_carry = hydraulic_conductivity(
        out_m1.psi_new, theta_from_psi(out_m1.psi_new, loam), loam)[:, -1]
    K_final = hydraulic_conductivity(out.psi_new, out.theta_new, loam)[:, -1]
    reported = np.asarray(out.runoff_subsurface) / _RHO   # [m/s]
    np.testing.assert_allclose(reported, np.asarray(K_carry), rtol=1e-12, atol=0.0)
    assert float(jnp.max(jnp.abs(out.runoff_subsurface / _RHO - K_final))) > 0.0, (
        "vacuous pin: K(psi_final) coincides with the debited K on the "
        "draining column; pick a wetter/faster-draining scenario")


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
        W = (jnp.sum(dz * s2.theta_soil[0]) + s2.surface_water[0]) * _RHO  # soil + pond
        return s2, (r.surface_mass_flux[0], s2.runoff_surface[0],
                    s2.runoff_subsurface[0], W, jnp.min(s2.theta_soil))

    W0 = float((jnp.sum(dz * st.theta_soil[0]) + st.surface_water[0]) * _RHO)
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


# ── float32-only Richards free-drainage carry (the _K_bot0 seed fix) ────────
def test_richards_free_drainage_runs_under_fp32_policy():
    """solve_richards seeds a 4th fori_loop carry slot (_K_bot0) cast to the
    working dtype so the scan carry input/output dtypes AGREE under a pure
    float32 run — otherwise the free-drainage K_bot diagnostic seed (float32)
    would mismatch the loop-body output and fail at compile ("scan body carry
    input and output must have equal types").  Only x64 was exercised; this
    pins the fp32-only path.

    MUST run with x64 DISABLED (do NOT set JAX_ENABLE_X64=1) so the default
    dtype — and hence the solver working dtype — is genuinely float32.  If the
    suite is run under x64 everything upcasts to float64 and there is no fp32
    carry to test, so we skip rather than give a false green."""
    if jax.config.read("jax_enable_x64"):
        pytest.skip(
            "fp32 carry path needs x64 OFF; run this node without "
            "JAX_ENABLE_X64=1")

    from legoesm.core.precision import get_policy, set_policy, PrecisionPolicy

    saved_policy = get_policy()
    set_policy(PrecisionPolicy.fp32())
    try:
        grid = _grid()                      # 8-layer soil grid (float32 under x64-off)
        assert grid.dz.dtype == jnp.float32, grid.dz.dtype
        hyd = SoilHydraulicsConfig()
        cfg = RichardsConfig()              # default bottom_bc == "free_drainage"
        assert cfg.bottom_bc == "free_drainage"

        theta0 = jnp.full((1, 8), 0.30, dtype=jnp.float32)
        psi0 = psi_from_theta(theta0, hyd).astype(jnp.float32)
        flux_top = jnp.asarray([5.0e-6], dtype=jnp.float32)   # net infiltration [m/s]
        sink = jnp.zeros((1, 8), dtype=jnp.float32)
        pond0 = jnp.zeros((1,), dtype=jnp.float32)

        # A float32-mismatched carry would raise at trace/compile time here.
        out = solve_richards(psi0, theta0, grid, hyd, cfg,
                             flux_top, sink, 3600.0, surface_water=pond0)

        # The fp32 path is genuinely exercised (no silent float64 widening).
        assert out.runoff_subsurface.dtype == jnp.float32, out.runoff_subsurface.dtype
        assert bool(jnp.all(jnp.isfinite(out.runoff_subsurface)))
        assert bool(jnp.all(jnp.isfinite(out.theta_new)))
        assert bool(jnp.all(jnp.isfinite(out.psi_new)))
        # Free drainage on a wet column drains DOWN: subsurface runoff > 0.
        assert float(jnp.sum(out.runoff_subsurface)) > 0.0
    finally:
        set_policy(saved_policy)


# ── surface soil resistance (the user-requested physics) ────────────────────
def test_surface_resistance_throttles_dry_soil_evaporation():
    """A drying surface forms a crust: bare-soil evaporation with the resistance
    (exp>0) is strictly less than with none (exp=0) from a dry top layer.

    Was xfail'd: the extreme 60-step dry+hot scenario also tripped a SimpleSEB
    surface-energy <-> soil-T thermal runaway (the ``S_top**exp`` resistance buries the
    un-evaporated energy in ``G_surface``, which the EXPLICIT surface coupling amplified
    to NaN).  The semi-implicit surface conductance ported from origin/main
    (``compute_simple_seb_fluxes`` -> ``solve_soil_thermal(surface_conductance=...)``, a
    Robin BC) damps that feedback, so the run now stays finite and the resistance-
    throttling physics is directly testable — this doubles as the regression guard."""
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
