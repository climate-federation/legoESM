"""Cross-grid regression test for the iter-1..14 anchored mass fixers.

Runs a short ``Held-Suarez``-style integration on each of the four
atmosphere hydrostatic dycores and asserts mass drift stays at fp64
floor (≤ 1e-10).  The test_dycores branch ``test_dycores.md``
log documents the iter-by-iter reductions that brought every path
into machine precision; a regression here means one of the fixers
(``anchor_mass_to_initial`` plumbing, ``_conservation_accumulator``
casts, dropped ``correction.astype`` calls) has been undone.

Each grid block is small (T21 / C12 / ico4 / 36x72) and runs only
~5 steps with ``dt=600s``; the full test completes in well under a
minute on CPU with ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import math

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

# Matrix-runner-grade tolerance: ``mass_drift`` should sit at the
# fp64-accumulation floor (~1e-15) after iter-1..14.  Allow a generous
# 1e-10 ceiling so per-step trace differences across JAX versions don't
# flap the test, while still flagging a regression by orders of
# magnitude (pre-iter-1 baselines were 1e-3 to 1e-5).
# iter-17: tighten from 1e-10 → 1e-12 and extend per-test integration
# from 5 → 20 steps.  Direct measurement on cube C12 PE shows
# 50-step mass drift ~4e-15, so a 20-step run sits comfortably below
# the 1e-12 ceiling while staying sub-second per grid.
DRIFT_TOL = 1e-12
N_STEPS = 20


def _rel_drift(m0: float, m1: float) -> float:
    return abs(m1 - m0) / max(abs(m0), 1.0)


def test_mass_conservation_cubed_sphere_pe():
    """Cubed-sphere PE: anchor_mass_to_initial + use_conservation_fixer."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init

    grid = create_cubed_sphere(12)
    sigma = create_sigma_coordinate(10)
    cfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True,
        fix_mass=True,
        anchor_mass_to_initial=True,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    state = held_suarez_init(grid, sigma)

    def _mass(s):
        return float(jnp.sum(
            s.p_s.data.astype(jnp.float64)
            * grid.area.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(N_STEPS):
        state = model.step(state, 600.0)
    assert _rel_drift(m0, _mass(state)) < DRIFT_TOL


def test_mass_conservation_latlon_pe():
    """C-grid lat-lon PE: anchor_mass_to_initial + fp64 budget acc."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init_latlon

    grid = create_latlon_grid(36, 72)
    sigma = create_sigma_coordinate(10)
    # CFL-safe dt for the coarse lat-lon grid (pole cells).
    dx_pole = float(grid.radius) * grid.dlon * math.cos(
        math.pi / 2 - grid.dlat / 2)
    dt = min(200.0, 0.5 * dx_pole / 300.0)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=dt)
    state = hydrostatic_to_cgrid(
        held_suarez_init_latlon(grid, sigma), grid,
    )

    def _mass(s):
        return float(jnp.sum(
            s.p_s.astype(jnp.float64) * grid.area.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(N_STEPS):
        state = model.step(state, dt)
    assert _rel_drift(m0, _mass(state)) < DRIFT_TOL


def test_mass_conservation_mpas_pe():
    """MPAS PE: anchor_mass_to_initial + fp64 fixer."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
        MPASPrimitiveEquationModel,
        MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init_mpas

    mesh = create_voronoi_mesh(4)
    sigma = create_sigma_coordinate(10)
    cfg = MPASPrimitiveEquationConfig(
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
    state = held_suarez_init_mpas(mesh, sigma)

    def _mass(s):
        return float(jnp.sum(
            s.p_s.data.astype(jnp.float64)
            * mesh.areaCell.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(N_STEPS):
        state = model.step(state, 200.0)
    assert _rel_drift(m0, _mass(state)) < DRIFT_TOL


def test_mass_conservation_spectral_pe():
    """Spectral PE: anchored ``lnps_hat[0]`` rescale."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPrimitiveEquationModel,
        SpectralPEConfig,
        isothermal_rest_state_spectral,
        spectral_pe_to_grid,
    )

    grid = create_gaussian_grid(21)
    sigma = create_sigma_coordinate(10)
    cfg = SpectralPEConfig(
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = SpectralPrimitiveEquationModel(
        grid, sigma, cfg, allow_unsupported_backend=True,
    )
    state = isothermal_rest_state_spectral(grid, sigma)

    def _mass(s):
        fields = spectral_pe_to_grid(s, grid, sigma)
        return float(jnp.sum(
            fields["p_s"].astype(jnp.float64)
            * grid.grid_area.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(N_STEPS):
        state = model.step(state, 600.0)
    assert _rel_drift(m0, _mass(state)) < DRIFT_TOL


# ---------------------------------------------------------------------------
# iter-33: long-run drift check.  Ensures the anchor doesn't accumulate
# error across 5x the standard N_STEPS — catches any per-step drift the
# 20-step gate would miss (e.g. a slow O(N_steps) bias rather than the
# bounded O(ULP) random walk the anchor is supposed to enforce).
# ---------------------------------------------------------------------------

def test_long_run_mass_conservation_cubed_sphere_pe():
    """100-step cube PE: anchor must NOT random-walk over long runs."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init

    grid = create_cubed_sphere(12)
    sigma = create_sigma_coordinate(10)
    cfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True,
        fix_mass=True,
        anchor_mass_to_initial=True,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    state = held_suarez_init(grid, sigma)

    def _mass(s):
        return float(jnp.sum(
            s.p_s.data.astype(jnp.float64)
            * grid.area.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(100):
        state = model.step(state, 600.0)
    drift = _rel_drift(m0, _mass(state))
    # 100 steps × ULP(p_s) random walk ~ sqrt(100)·1e-15 ~ 1e-14.
    # Direct measurement on cube C12 gives 4.18e-15.  Allow 1e-12 (a
    # factor of ~200 above the observed floor) for resilience to
    # JAX-version / platform jitter, while still catching a true O(1e-9)
    # regression by 3 orders of magnitude.
    assert drift < 1e-12, (
        f"cube PE 100-step drift {drift:.2e} exceeds 1e-12 — possible "
        f"per-step accumulation bug in the anchored fixer"
    )


def test_long_run_mass_conservation_spectral_pe():
    """iter-61: 100-step spectral PE long-run guard.

    Parallel to iter-33 cube PE.  Validates the iter-3 spectral PE
    anchored fixer (`lnps_hat[0] += log(target/now)·sqrt(4π)`) holds
    over 100 steps.  Direct measurement: drift = 8.03e-16 on T21
    spectral PE with isothermal rest state.
    """
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.spectral_pe import (
        SpectralPrimitiveEquationModel, SpectralPEConfig,
        isothermal_rest_state_spectral, spectral_pe_to_grid,
    )

    grid = create_gaussian_grid(21)
    sigma = create_sigma_coordinate(10)
    cfg = SpectralPEConfig(
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = SpectralPrimitiveEquationModel(
        grid, sigma, cfg, allow_unsupported_backend=True,
    )
    state = isothermal_rest_state_spectral(grid, sigma)

    def _mass(s):
        fields = spectral_pe_to_grid(s, grid, sigma)
        return float(jnp.sum(
            fields["p_s"].astype(jnp.float64)
            * grid.grid_area.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(100):
        state = model.step(state, 600.0)
    drift = _rel_drift(m0, _mass(state))
    assert drift < 1e-12, (
        f"spectral PE 100-step drift {drift:.2e} exceeds 1e-12"
    )


def test_long_run_mass_conservation_latlon_pe():
    """iter-62: 100-step lat-lon PE long-run guard.

    Parallel to iter-33 cube PE / iter-61 spectral PE.  Exercises the
    iter-2/12 lat-lon ``_apply_safety_rails`` (T-floor + p_s-floor +
    mass fixer + tracer rescale) over 100 steps.  Direct measurement:
    drift = 1.61e-16 on 36x72 lat-lon, dt=40.4s.
    """
    import math
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init_latlon

    grid = create_latlon_grid(36, 72)
    sigma = create_sigma_coordinate(10)
    dx_pole = float(grid.radius) * grid.dlon * math.cos(
        math.pi / 2 - grid.dlat / 2)
    dt = min(200.0, 0.5 * dx_pole / 300.0)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg, dt=dt)
    state = hydrostatic_to_cgrid(
        held_suarez_init_latlon(grid, sigma), grid,
    )

    def _mass(s):
        return float(jnp.sum(
            s.p_s.astype(jnp.float64) * grid.area.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(100):
        state = model.step(state, dt)
    drift = _rel_drift(m0, _mass(state))
    assert drift < 1e-12, (
        f"lat-lon PE 100-step drift {drift:.2e} exceeds 1e-12"
    )
