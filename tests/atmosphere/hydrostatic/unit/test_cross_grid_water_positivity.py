"""Cross-grid tracer-positivity wiring (#1354/#1515).

The spectral and lat-lon C-grid lanes previously had NO tracer positivity
floor, so non-monotone transport left negative water.  Both now route through
the shared ``apply_water_positivity`` (default: column-conserving borrow).
These tests prove the stage actually FIRES on each lane — negatives removed and
the borrow conserves the dp-weighted column integral — so the wiring is not
silently inert.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.field import Field
from legoesm.grids.vertical import create_sigma_coordinate


# ---------------------------------------------------------------------------
# Spectral lane
# ---------------------------------------------------------------------------

def test_spectral_positivity_fires_and_conserves():
    from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPEConfig, SpectralPrimitiveEquationModel,
        isothermal_rest_state_spectral,
    )
    grid = create_gaussian_grid(n_max=21)
    sigma = create_sigma_coordinate(8)
    model = SpectralPrimitiveEquationModel(grid, sigma, SpectralPEConfig())
    state = isothermal_rest_state_spectral(
        grid, sigma, perturbation_amplitude=0.0)

    shp = (grid.n_lat, grid.n_lon, sigma.n_levels)
    qv = jnp.full(shp, 5.0e-3).at[:, :, 0].set(-2.0e-3)  # negative surface lobe
    state = state._replace(tracers={
        "q_v": Field(data=qv, name="q_v",
                     dims=("lat", "lon", "level"), units="kg/kg")})

    out = model._apply_tracer_positivity(state)
    qv_out = out.tracers["q_v"].data
    assert float(jnp.min(qv_out)) >= -1e-30            # negatives removed

    # Borrow conserves the dp-weighted global integral (T is spectral, untouched)
    p_s = jnp.exp(sh_synthesis(grid, state.lnps_hat.data))
    dp = p_s[..., None] * sigma.dsigma
    pre = float(jnp.sum(qv * dp))
    post = float(jnp.sum(qv_out * dp))
    assert abs(post - pre) <= 1e-9 * abs(pre)

    # Non-vacuity: with the borrow OFF a plain floor would ADD water.
    model_off = SpectralPrimitiveEquationModel(
        grid, sigma, SpectralPEConfig(conservative_tracer_clamp=False))
    qv_plain = model_off._apply_tracer_positivity(state).tracers["q_v"].data
    assert float(jnp.min(qv_plain)) >= -1e-30
    assert float(jnp.sum(qv_plain * dp)) > pre + 1e-6 * abs(pre)  # created mass


# ---------------------------------------------------------------------------
# Lat-lon C-grid lane
# ---------------------------------------------------------------------------

def test_latlon_positivity_fires_and_conserves():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
        CGridLatLonHydrostaticState,
    )
    grid = create_latlon_grid(
        n_lat=32, radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(8)
    # fix_mass off so the positivity stage is isolated from the mass-fix rescale.
    model = CGridLatLonPrimitiveEquationModel(
        grid, sigma, CGridLatLonPrimitiveEquationConfig(fix_mass=False))

    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
    qv = jnp.full((n_lat, n_lon, nlev), 5.0e-3).at[:, :, 0].set(-2.0e-3)
    state = CGridLatLonHydrostaticState(
        u=jnp.zeros((n_lat, n_lon + 1, nlev)),
        v=jnp.zeros((n_lat + 1, n_lon, nlev)),
        T=jnp.full((n_lat, n_lon, nlev), 300.0),
        p_s=jnp.full((n_lat, n_lon), 1.0e5),
        phis=jnp.zeros((n_lat, n_lon)),
        tracers={"q_v": qv},
    )

    out = model._apply_safety_rails(state)
    qv_out = out.tracers["q_v"]
    assert float(jnp.min(qv_out)) >= -1e-30            # negatives removed

    dp = state.p_s[..., None] * sigma.dsigma
    pre = float(jnp.sum(qv * dp))
    post = float(jnp.sum(qv_out * dp))
    # rtol 1e-6: this lane's tracer storage is fp32, so the borrow's residual
    # redistribution conserves to fp32-class roundoff (measured 1.7e-8 here),
    # not the fp64 1e-9 the spectral lane hits.
    assert abs(post - pre) <= 1e-6 * abs(pre)          # borrow conserved
    assert jnp.allclose(out.T, state.T)                # borrow leaves T alone


# ---------------------------------------------------------------------------
# Area weighting (user directive 2026-10-02): the stage weights by cell
# MASS (dp * area), so a net-negative column's invented mass is handed back
# per unit of mass -- the dp-only weight mis-conserved on cos(lat) grids.
# ---------------------------------------------------------------------------

def test_spectral_positivity_global_residual_is_area_weighted():
    """A wholly negative column at the smallest polar cell: the area-
    weighted global mass is kept (the model uses dp*area), the dp-only
    integral is NOT (the control)."""
    from legoesm.grids.gaussian import create_gaussian_grid, sh_synthesis
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPEConfig, SpectralPrimitiveEquationModel,
        isothermal_rest_state_spectral,
    )
    from legoesm.core.conservation import apply_water_positivity
    grid = create_gaussian_grid(n_max=21)
    sigma = create_sigma_coordinate(8)
    model = SpectralPrimitiveEquationModel(grid, sigma, SpectralPEConfig())
    state = isothermal_rest_state_spectral(
        grid, sigma, perturbation_amplitude=0.0)
    shp = (grid.n_lat, grid.n_lon, sigma.n_levels)
    qv = jnp.full(shp, 5.0e-3).at[0, 3, :].set(-4.0e-3)     # polar row column
    state = state._replace(tracers={
        "q_v": Field(data=qv, name="q_v",
                     dims=("lat", "lon", "level"), units="kg/kg")})
    qv_out = model._apply_tracer_positivity(state).tracers["q_v"].data
    assert float(jnp.min(qv_out)) >= -1e-30
    p_s = jnp.exp(sh_synthesis(grid, state.lnps_hat.data))
    dp = p_s[..., None] * sigma.dsigma
    area = jnp.asarray(grid.grid_area)[..., None]
    assert float(area.max() / area.min()) > 5.0            # the grid discriminates
    m = lambda q: float(jnp.sum(q * dp * area))             # noqa: E731
    assert abs(m(qv_out) - m(qv)) <= 1e-9 * abs(m(qv))
    ctrl, _ = apply_water_positivity({"q_v": qv}, None, dp,
                                     conservative=True, energy_consistent=False)
    assert abs(m(ctrl["q_v"]) - m(qv)) > 1e-6 * abs(m(qv))   # dp-only mis-conserves


def test_latlon_positivity_global_residual_is_area_weighted():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel, CGridLatLonPrimitiveEquationConfig,
        CGridLatLonHydrostaticState,
    )
    from legoesm.core.conservation import apply_water_positivity
    grid = create_latlon_grid(
        n_lat=32, radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(8)
    model = CGridLatLonPrimitiveEquationModel(
        grid, sigma, CGridLatLonPrimitiveEquationConfig(fix_mass=False))
    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
    qv = jnp.full((n_lat, n_lon, nlev), 5.0e-3,
                  dtype=jnp.float64).at[0, 3, :].set(-4.0e-3)
    state = CGridLatLonHydrostaticState(
        u=jnp.zeros((n_lat, n_lon + 1, nlev)),
        v=jnp.zeros((n_lat + 1, n_lon, nlev)),
        T=jnp.full((n_lat, n_lon, nlev), 300.0),
        p_s=jnp.full((n_lat, n_lon), 1.0e5),
        phis=jnp.zeros((n_lat, n_lon)),
        tracers={"q_v": qv},
    )
    qv_out = model._apply_safety_rails(state).tracers["q_v"]
    assert float(jnp.min(qv_out)) >= -1e-30
    # The instrument is fp64 end to end: ``dsigma`` and ``grid.area`` are
    # float32 on this grid, and a weak-typed ``qv`` times either sums the
    # 16k-cell budget in fp32 (measured 1e-5 of noise, read as a defect).
    dp = state.p_s.astype(jnp.float64)[..., None] * jnp.asarray(
        sigma.dsigma, dtype=jnp.float64)
    area = jnp.asarray(grid.grid_area, dtype=jnp.float64)[..., None]
    assert float(area.max() / area.min()) > 5.0
    m = lambda q: float(jnp.sum(q.astype(jnp.float64) * dp * area))  # noqa: E731
    assert abs(m(qv_out) - m(qv)) <= 1e-9 * abs(m(qv))
    ctrl, _ = apply_water_positivity({"q_v": qv}, None, dp,
                                     conservative=True, energy_consistent=False)
    assert abs(m(ctrl["q_v"]) - m(qv)) > 1e-5 * abs(m(qv))


def _area_budget_check(q_in, q_out, dp, area, ctrl_dp_only):
    """Area-weighted tracer mass kept by the lane; the dp-only weight is the
    control that mis-conserves (so a lane silently back on dp alone fails)."""
    from legoesm.core.conservation import apply_water_positivity
    area = jnp.asarray(area, dtype=jnp.float64)[..., None]
    assert float(area.max() / area.min()) > 1.15           # res2 1.20x, C8 1.28x
    m = lambda q: float(jnp.sum(q.astype(jnp.float64) * dp * area))  # noqa: E731
    assert float(jnp.min(q_out)) >= -1e-30
    assert abs(m(q_out) - m(q_in)) <= 1e-9 * abs(m(q_in))
    ctrl, _ = apply_water_positivity({"q_v": q_in}, None, ctrl_dp_only,
                                     conservative=True, energy_consistent=False)
    assert abs(m(ctrl["q_v"]) - m(q_in)) > 1e-6 * abs(m(q_in))


def test_mpas_step_positivity_global_residual_is_area_weighted():
    """codex 2026-10-02 P2: the MPAS lane's own step (positivity lives inside
    ``_step_jit``).  Isothermal rest, uniform p_s, fixer off: the only tracer
    change across the step is the positivity stage.  The wholly negative
    column sits on the SMALLEST cell so dp-only vs dp*area differ most."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig, MPASPrimitiveEquationModel)
    from legoesm.core.state import MPASHydrostaticState
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
    sigma = create_sigma_coordinate(5)
    model = MPASPrimitiveEquationModel(
        mesh, sigma, MPASPrimitiveEquationConfig(fix_mass=False))
    nc, nlev = mesh.nCells, sigma.n_levels
    c_min = int(jnp.argmin(jnp.asarray(mesh.areaCell)))
    qv = jnp.full((nc, nlev), 5.0e-3, dtype=jnp.float64).at[c_min, :].set(-4.0e-3)
    state = MPASHydrostaticState(
        u=Field(jnp.zeros((mesh.nEdges, nlev)), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        T=Field(jnp.full((nc, nlev), 300.0), name="T",
                dims=("nCells", "nlev"), units="K"),
        p_s=Field(jnp.full((nc,), 1.0e5), name="p_s", dims=("nCells",),
                  units="Pa"),
        phis=Field(jnp.zeros((nc,)), name="phis", dims=("nCells",),
                   units="m^2/s^2"),
        tracers={"q_v": Field(qv, name="q_v", dims=("nCells", "nlev"),
                              units="kg/kg")},
    )
    out = model.step(state, 60.0)
    dp = jnp.full((nc,), 1.0e5, dtype=jnp.float64)[:, None] * jnp.asarray(
        sigma.dsigma, dtype=jnp.float64)
    _area_budget_check(qv, out.tracers["q_v"].data, dp, mesh.areaCell, dp)


def test_cube_step_positivity_global_residual_is_area_weighted():
    """codex 2026-10-02 P2: the cube (C-D grid) lane's own step."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig, CDGridPrimitiveEquationModel)
    from legoesm.core.state import FV3HydrostaticState
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    n, nlev = 8, 5
    base = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    model = CDGridPrimitiveEquationModel(
        base, sigma, CDGridPrimitiveEquationConfig(fix_mass=False))
    area = jnp.asarray(model.grid.grid_area)
    f, i, j = (int(v) for v in jnp.unravel_index(jnp.argmin(area), area.shape))
    qv = jnp.full((6, n, n, nlev), 5.0e-3,
                  dtype=jnp.float64).at[f, i, j, :].set(-4.0e-3)
    state = FV3HydrostaticState(
        u_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="u_d"),
        v_d=Field(jnp.zeros((6, n + 1, n + 1, nlev)), name="v_d"),
        T=Field(jnp.full((6, n, n, nlev), 300.0), name="T"),
        p_s=Field(jnp.full((6, n, n), 1.0e5), name="p_s"),
        phis=Field(jnp.zeros((6, n, n)), name="phis"),
        tracers={"q_v": Field(qv, name="q_v")},
    )
    out = model.step(state, 60.0)
    dp = jnp.full((6, n, n), 1.0e5, dtype=jnp.float64)[..., None] * jnp.asarray(
        sigma.dsigma, dtype=jnp.float64)
    _area_budget_check(qv, out.tracers["q_v"].data, dp, area, dp)
