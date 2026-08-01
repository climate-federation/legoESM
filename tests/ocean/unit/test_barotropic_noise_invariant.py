"""Crit 3 CI invariant test for the barotropic mode-noise issue.

Tests that the implicit Crank-Nicolson barotropic solver
(``barotropic_solver = 'implicit_cn'``) keeps the time-mean V_baro
clean of grid-scale chequerboard noise.

Setup: small flat-bottom lat-lon C-grid basin, polar caps, weak cosine-
latitude zonal wind stress, started from rest.  Spin up for 30 days;
time-average over the last 10 days; assert the Crit 1 invariants on the
time-mean V_baro.

Crit 1 invariants (from docs/dev-notes/issues/barotropic_mode_noise.md):
  1.  var(∇·U_baro) / var(U_baro) < 0.05
  2.  max |⟨V_baro⟩| outside polar caps < 5e-3 m/s
  3.  σ(grid-scale V_baro after 3-pt meridional Laplacian) < 1e-2 m/s

The test runs only the implicit solver and asserts Crit 1.  A separate
test compares the implicit and explicit solvers and asserts that the
implicit solver produces a *significantly* cleaner V_baro under the
same forcing — that is the regression we are protecting against (a
future change must not silently revert the implicit-CN benefit).

Run with:
    JAX_ENABLE_X64=1 python -m pytest tests/ocean/unit/test_barotropic_noise_invariant.py -v
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")
jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
from legoesm.ocean.physics.combined import OceanPhysicsConfig
from legoesm.ocean.physics.surface_forcing.config import (
    PrescribedForcingConfig,
    SurfaceForcingConfig,
)
from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig


# ----- Crit 1 metrics (mirror scripts/validate/eval_barotropic_noise_invariants.py) -----


def _depth_mean_to_face(field_3d, dz):
    H = float(dz.sum())
    return np.sum(field_3d * dz[None, None, :], axis=-1) / H


def _crit1_metrics(state_means, dz, grid, mask, u_mask, v_mask,
                   polar_lat_deg=70.0):
    """Compute the three Crit 1 metrics from time-mean state arrays."""
    u_mean = state_means["u"]   # (n_lat, n_lon+1, nlev)
    v_mean = state_means["v"]   # (n_lat+1, n_lon, nlev)
    U_baro = _depth_mean_to_face(u_mean, dz) * u_mask
    V_baro = _depth_mean_to_face(v_mean, dz) * v_mask

    # Crit 1.1
    div = np.asarray(
        divergence_cgrid(jnp.asarray(U_baro), jnp.asarray(V_baro), grid)
    ) * mask
    wet_cells = mask > 0.5
    wet_u = u_mask > 0.5
    var_div = float(np.var(div[wet_cells]))
    var_U = float(np.var(U_baro[wet_u]))
    ratio = var_div / max(var_U, 1.0e-30)

    # Polar mask along v-face latitudes
    lat_deg = np.asarray(grid.lat) * 180.0 / np.pi
    lat_v_deg = np.concatenate([
        [lat_deg[0] - 0.5 * float(grid.dlat) * 180.0 / np.pi],
        0.5 * (lat_deg[:-1] + lat_deg[1:]),
        [lat_deg[-1] + 0.5 * float(grid.dlat) * 180.0 / np.pi],
    ])
    polar = np.abs(lat_v_deg) > polar_lat_deg

    # Crit 1.2: max |V_baro| off polar caps
    interior_v = (v_mask > 0.5).copy()
    interior_v[polar, :] = False
    max_V = float(np.max(np.abs(V_baro)[interior_v])) if interior_v.any() else 0.0

    # Crit 1.3: sigma(3-pt meridional Laplacian V_baro)
    Vlap = np.zeros_like(V_baro)
    Vlap[1:-1] = V_baro[2:] - 2.0 * V_baro[1:-1] + V_baro[:-2]
    interior_lap = (v_mask > 0.5).copy()
    interior_lap[0, :] = False
    interior_lap[-1, :] = False
    interior_lap[polar, :] = False
    sigma_lap = (
        float(np.std(Vlap[interior_lap])) if interior_lap.any() else 0.0
    )

    return {
        "var_div_U_over_var_U": ratio,
        "max_abs_V_baro_off_polar": max_V,
        "sigma_3pt_laplacian_V_baro": sigma_lap,
    }


# ----- Test setup ---------------------------------------------------------


def _make_physics(tau_max: float = 0.05):
    return OceanPhysicsConfig(
        surface_forcing=SurfaceForcingConfig(
            scheme="prescribed",
            prescribed=PrescribedForcingConfig(
                wind_profile="cosine_latitude", tau_max=tau_max,
            ),
        ),
        lateral_mixing=LateralMixingConfig(scheme="none"),
    )


def _build_model(solver: str, physics):
    return LatLonCGridOceanModel(
        create_latlon_grid(n_lat=18, n_lon=36),
        create_ocean_z_star(n_levels=5, H_max=4000.0),
        LatLonCGridOceanConfig.from_flat(
            A_h=1.0e4, K_h=0.0, A_v=1.0e-3, K_v=1.0e-5,
            bottom_drag_r=1.1e-3,
            n_barotropic_substeps=30,
            physics=physics,
            barotropic_solver=solver,
        ),
    )


def _spin_up_and_average(model, state0, n_steps, dt, average_last_fraction=1.0/3.0):
    n_skip = int(n_steps * (1.0 - average_last_fraction))
    n_avg = n_steps - n_skip
    u_sum = np.zeros_like(np.asarray(state0.u.data), dtype=np.float64)
    v_sum = np.zeros_like(np.asarray(state0.v.data), dtype=np.float64)
    eta_sum = np.zeros_like(np.asarray(state0.eta.data), dtype=np.float64)
    state = state0
    for i in range(n_steps):
        state = model.step(state, dt=dt)
        if i >= n_skip:
            u_sum += np.asarray(state.u.data)
            v_sum += np.asarray(state.v.data)
            eta_sum += np.asarray(state.eta.data)
    return state, {
        "u": u_sum / n_avg,
        "v": v_sum / n_avg,
        "eta": eta_sum / n_avg,
    }


# ----- Tests --------------------------------------------------------------


def test_implicit_solver_meets_crit1_in_rest_plus_wind_spinup():
    """The implicit CN solver delivers a smooth time-mean V_baro after a
    30-day rest + cosine-latitude weak-wind spinup, satisfying Crit 1."""
    grid = create_latlon_grid(n_lat=18, n_lon=36)
    z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)

    physics = _make_physics(tau_max=0.05)
    model = _build_model("implicit_cn", physics)
    state0 = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0, land_lat_threshold=80.0,
    )

    dt = 1800.0
    n_steps = int(30.0 * 86400.0 / dt)
    final_state, means = _spin_up_and_average(
        model, state0, n_steps, dt, average_last_fraction=1.0 / 3.0,
    )

    assert bool(jnp.all(jnp.isfinite(final_state.eta.data))), \
        "implicit_cn produced NaN in eta after 30-day spinup"
    assert bool(jnp.all(jnp.isfinite(final_state.u.data))), \
        "implicit_cn produced NaN in u after 30-day spinup"

    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    mask = np.asarray(state0.land_mask.data, dtype=np.float64)
    u_mask = np.asarray(state0.u_mask.data, dtype=np.float64)
    v_mask = np.asarray(state0.v_mask.data, dtype=np.float64)
    metrics = _crit1_metrics(
        means, dz, grid, mask, u_mask, v_mask, polar_lat_deg=70.0,
    )

    assert metrics["var_div_U_over_var_U"] < 0.05, (
        f"Crit 1.1 failed: var(div U)/var(U) = "
        f"{metrics['var_div_U_over_var_U']:.4e}")
    assert metrics["max_abs_V_baro_off_polar"] < 5.0e-3, (
        f"Crit 1.2 failed: max|V_baro| off polar = "
        f"{metrics['max_abs_V_baro_off_polar']:.4e} m/s")
    assert metrics["sigma_3pt_laplacian_V_baro"] < 1.0e-2, (
        f"Crit 1.3 failed: sigma(3-pt Lap V_baro) = "
        f"{metrics['sigma_3pt_laplacian_V_baro']:.4e} m/s")


def test_implicit_solver_cleaner_than_explicit_substep():
    """Same spinup, both solvers — assert implicit gives ≥3× cleaner
    grid-scale V_baro than the explicit substep.  Catches regressions
    that would silently revert the implicit-CN benefit."""
    grid = create_latlon_grid(n_lat=18, n_lon=36)
    z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)
    physics = _make_physics(tau_max=0.05)
    state0 = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0, land_lat_threshold=80.0,
    )
    dt = 1800.0
    n_steps = int(30.0 * 86400.0 / dt)

    dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
    mask = np.asarray(state0.land_mask.data, dtype=np.float64)
    u_mask = np.asarray(state0.u_mask.data, dtype=np.float64)
    v_mask = np.asarray(state0.v_mask.data, dtype=np.float64)

    metrics = {}
    for solver in ("implicit_cn", "explicit_substep"):
        model = _build_model(solver, physics)
        _, means = _spin_up_and_average(
            model, state0, n_steps, dt, average_last_fraction=1.0 / 3.0,
        )
        metrics[solver] = _crit1_metrics(
            means, dz, grid, mask, u_mask, v_mask, polar_lat_deg=70.0,
        )

    # Implicit solver must give at least 3× lower σ(3-pt Lap V_baro).
    sigma_imp = metrics["implicit_cn"]["sigma_3pt_laplacian_V_baro"]
    sigma_exp = metrics["explicit_substep"]["sigma_3pt_laplacian_V_baro"]
    assert sigma_imp < sigma_exp / 3.0, (
        f"implicit_cn σ(3-pt Lap V) = {sigma_imp:.3e} should be < "
        f"σ_explicit/3 = {sigma_exp/3.0:.3e} "
        f"(explicit σ = {sigma_exp:.3e})"
    )


@pytest.fixture
def _fp64():
    """#1388: the conservation gate below reasons in fp64 but ran in fp32.

    `JAX_ENABLE_X64=1` does NOT change legoESM's precision policy — state
    constructors read `get_policy().storage`, which defaults to fp32 — so the
    mass gate was measuring float32 round-off, not the solver. MEASURED on the
    same 100 steps: fp32 drift 1.161e-06 (over the 5e-7 ceiling), fp64 drift
    4.351e-16. Ten orders of magnitude: the implicit solver conserves mass to
    machine precision, and the "failure" was the precision policy.
    """
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def test_implicit_solver_conserves_mass(_fp64):
    """Per-step mass conservation: ``Σ η·area`` drift over 100 steps
    should be at PCG-tolerance level (relative to the running ``Σ |η|·area``).
    """
    grid = create_latlon_grid(n_lat=18, n_lon=36)
    z_coord = create_ocean_z_star(n_levels=5, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e4, A_v=1.0e-3, K_v=1.0e-5,
        bottom_drag_r=1.1e-3,
        physics=_make_physics(tau_max=0.05),
        barotropic_solver="implicit_cn",
        barotropic_implicit_pcg_tol=1.0e-12,
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0, land_lat_threshold=80.0,
    )

    area = np.asarray(grid.area)
    mask = np.asarray(state.land_mask.data)
    V0 = float(np.sum(np.asarray(state.eta.data) * area * mask))
    abs_eta_sum = float(np.sum(np.abs(np.asarray(state.eta.data)) * area * mask))

    for _ in range(100):
        state = model.step(state, dt=600.0)
        abs_eta_sum = max(abs_eta_sum,
                          float(np.sum(np.abs(np.asarray(state.eta.data)) *
                                       area * mask)))

    V1 = float(np.sum(np.asarray(state.eta.data) * area * mask))
    # Reference scale: largest |Σ |η|·area| seen during the run.
    rel_drift = abs(V1 - V0) / max(abs_eta_sum, 1.0e-30)
    # PCG residual with tol=1e-12 contributes at most ~1e-10 relative
    # mass error per step; over 100 steps and float64 cell-summation noise
    # ~ sqrt(N_cells) · tol, the cumulative drift should remain << 1e-6.
    # That reasoning is fp64 reasoning, which is why this test now REQUIRES the
    # fp64 policy (see _fp64): under the default fp32 storage the same run
    # drifts 1.161e-06 -- round-off, not the solver (#1388).
    # The 5e-7 ceiling absorbs fp-ordering variation across momentum-
    # advection schemes (vector-invariant vs WENO vs PV-flux Sadourny);
    # the docstring-stated bound is "<< 1e-6", which is what's tested.
    assert rel_drift < 5.0e-7, (
        f"Mass drift after 100 steps: {rel_drift:.3e} (relative to "
        f"Σ |η|·area = {abs_eta_sum:.3e})")


def test_implicit_solver_grad_smoke():
    """jax.grad through the implicit CN solver returns finite values."""
    grid = create_latlon_grid(n_lat=18, n_lon=36)
    z_coord = create_ocean_z_star(n_levels=4, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_h=1.0e4, A_v=1.0e-3, K_v=1.0e-5,
        bottom_drag_r=1.1e-3,
        barotropic_solver="implicit_cn",
        barotropic_implicit_pcg_maxiter=100,
    )
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=10.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0, land_lat_threshold=80.0,
    )

    rng = np.random.default_rng(0)
    eta0 = jnp.asarray(0.01 * rng.standard_normal(state0.eta.data.shape),
                       dtype=jnp.float64)

    def loss(eta):
        s = state0._replace(eta=state0.eta.replace(data=eta))
        s_new = model.step(s, dt=600.0)
        return jnp.sum(s_new.eta.data ** 2)

    g = jax.grad(loss)(eta0)
    assert bool(jnp.all(jnp.isfinite(g))), "jax.grad produced NaN/Inf"
    assert float(jnp.linalg.norm(g)) > 0.0, \
        "jax.grad returned exactly zero gradient"


def test_implicit_solver_default_is_explicit_substep():
    """Default config uses the legacy explicit substep so existing
    experiments are bit-stable."""
    cfg = LatLonCGridOceanConfig.from_flat()
    assert cfg.barotropic.barotropic_solver == "explicit_substep"


def test_invalid_barotropic_solver_rejected():
    """Config validation rejects unknown solver names."""
    grid = create_latlon_grid(n_lat=8, n_lon=16)
    z_coord = create_ocean_z_star(n_levels=2, H_max=1000.0)
    cfg = LatLonCGridOceanConfig.from_flat(barotropic_solver="bogus")
    with pytest.raises(ValueError, match="barotropic_solver"):
        LatLonCGridOceanModel(grid, z_coord, cfg)
