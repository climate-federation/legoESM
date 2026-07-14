"""Cross-grid regression test for the iter-1..14 anchored mass fixers.

Runs a short ``Held-Suarez``-style integration on each of the four
atmosphere hydrostatic dycores and asserts mass drift stays at fp64
floor (≤ 1e-10).  The test_dycores branch ``test_dycores.md``
log documents the iter-by-iter reductions that brought every path
into machine precision; a regression here means one of the fixers
(``anchor_mass_to_initial`` plumbing, ``conservation_accumulator``
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
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon

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
    """MPAS PE: anchor_mass_to_initial + fp64 fixer.

    Runs under an explicit ``PrecisionPolicy.fp64()`` (the documented
    "fp64 floor" intent of this module). The MPAS PE dycore integrates
    with ``ssp_rk54``, whose fixed-dtype scan carry rejects an fp64
    tendency against an fp32 state (the cube/lat-lon PE paths tolerate
    the mixed dtype, which is why only the MPAS PE tests need the policy
    set explicitly). Without it the default fp32 storage policy makes the
    anchored fixer emit an fp64 ``p_s`` against an fp32 state and the
    integrator raises a dtype-mismatch TypeError. The 1.6e-16 drift below
    was originally measured under fp64.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel,
        MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
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
    finally:
        set_policy(saved)


def test_mass_conservation_spectral_pe():
    """Spectral PE: anchored ``lnps_hat[0]`` rescale."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
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
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

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
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
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
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
        hydrostatic_to_cgrid,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_latlon

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


def test_long_run_mass_conservation_mpas_pe():
    """iter-63: 100-step MPAS PE long-run guard.

    Completes the PE long-run matrix (cube iter-33, spectral iter-61,
    lat-lon iter-62, MPAS iter-63).  Exercises iter-11's anchored
    `_fix_mass_mpas_hydro` on a Voronoi mesh over the 100-step
    horizon.  Direct measurement: drift = 1.61e-16 on level-4
    Voronoi + 10 sigma levels.

    Runs under an explicit ``PrecisionPolicy.fp64()`` for the same reason
    as ``test_mass_conservation_mpas_pe``: the MPAS PE ``ssp_rk54``
    integrator rejects an fp64 fixer tendency against an fp32 state, and
    the 1.61e-16 floor above was measured under fp64.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy, get_policy
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas

    saved = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64())
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
        for _ in range(100):
            state = model.step(state, 200.0)
        drift = _rel_drift(m0, _mass(state))
        assert drift < 1e-12, (
            f"MPAS PE 100-step drift {drift:.2e} exceeds 1e-12"
        )
    finally:
        set_policy(saved)
