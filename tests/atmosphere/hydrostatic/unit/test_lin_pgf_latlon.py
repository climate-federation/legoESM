"""Unit tests for the Lin (1997) cross-product PGF on the lat-lon C-grid.

#1029: the legacy two-term PGF (−∇(Φ+KE) − R_d T (B p_s/p) ∇ln p_s) is a
linearly unstable discretisation over steep ridged terrain-following
coordinates on this grid (DCMIP 2-0-0 rest state grows ~2.6 e-folds/day at
72x144 L40, dt-independent, h_0 threshold between 1000 and 2000 m).  The
``pgf_scheme="lin1997"`` path computes the full hydrostatic PGF as one
well-conditioned cross-product at the faces (shared operator with the
FV3-faithful cube port in ``_fv3_lin_pgf``).

These are CI tripwires (operator properties + short-integration bound);
the full instability verdict runs at matrix scale (held_suarez_topo).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import (
    create_sigma_coordinate,
    standard_hybrid_levels,
)

from legoesm.atmosphere.dynamics.gcm._fv3_lin_pgf import (
    lin1997_pgf_latlon_cgrid,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
    cgrid_latlon_hydrostatic_tendencies,
    hydrostatic_to_cgrid,
)
from tests.test_cases.dcmip2012.rest_state_topography import (
    rest_state_topography_init_latlon,
)

N_LAT, N_LON, NLEV = 24, 48, 8


@pytest.fixture(scope="module")
def grid():
    return create_latlon_grid(N_LAT, N_LON)


def _pressures(coord, p_s, hybrid: bool):
    if hybrid:
        from legoesm.grids.vertical import pressure_from_hybrid
        return (pressure_from_hybrid(coord, p_s, full=False),
                pressure_from_hybrid(coord, p_s, full=True))
    from legoesm.grids.vertical import pressure_from_sigma
    return (pressure_from_sigma(coord.sigma_half, p_s),
            pressure_from_sigma(coord.sigma_full, p_s))


@pytest.mark.parametrize("coord_name", ["hybrid", "sigma"])
def test_uniform_columns_give_exactly_zero_pgf(grid, coord_name):
    """Identical neighbour columns → cross-product is zero by construction."""
    hybrid = coord_name == "hybrid"
    coord = standard_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    T = jnp.full((N_LAT, N_LON, NLEV), 287.5)
    p_s = jnp.full((N_LAT, N_LON), constants.p_ref)
    phis = jnp.zeros((N_LAT, N_LON))
    p_half, p_full = _pressures(coord, p_s, hybrid)
    pgf_x, pgf_y = lin1997_pgf_latlon_cgrid(T, p_s, phis, p_half, p_full, grid)
    assert float(jnp.max(jnp.abs(pgf_x))) == 0.0
    assert float(jnp.max(jnp.abs(pgf_y))) == 0.0
    assert pgf_x.shape == (N_LAT, N_LON + 1, NLEV)
    assert pgf_y.shape == (N_LAT + 1, N_LON, NLEV)


def test_pgf_sign_pressure_ridge(grid):
    """p_s increasing eastward (flat topo, uniform T) → westward force.

    Mirrors the sign-convention note in ``fv3_lin1997_pgf_3d_cgrid``: the
    returned tendency is ADDED to du/dt, so a positive zonal pressure
    gradient must yield a NEGATIVE pgf_x.
    """
    coord = standard_hybrid_levels(NLEV)
    T = jnp.full((N_LAT, N_LON, NLEV), 287.5)
    phis = jnp.zeros((N_LAT, N_LON))
    lon = jnp.linspace(0.0, 2.0 * jnp.pi, N_LON, endpoint=False)
    # Smooth zonal wavenumber-1 perturbation: dp_s/dx ∝ cos(lon).
    p_s = constants.p_ref + 1000.0 * jnp.sin(lon)[None, :] * jnp.ones((N_LAT, 1))
    p_half, p_full = _pressures(coord, p_s, hybrid=True)
    pgf_x, _ = lin1997_pgf_latlon_cgrid(T, p_s, phis, p_half, p_full, grid)
    # Interior faces in the first quarter (cos(lon) > 0 → p_s increasing
    # east) must be accelerated WESTWARD at every level.
    mid = N_LAT // 2
    sample = pgf_x[mid, 2:N_LON // 4 - 2, :]
    assert float(jnp.max(sample)) < 0.0
    # And symmetric: third quarter (p_s decreasing east) → eastward.
    sample2 = pgf_x[mid, N_LON // 2 + 2:3 * N_LON // 4 - 2, :]
    assert float(jnp.min(sample2)) > 0.0


@pytest.mark.parametrize("coord_name", ["hybrid", "sigma"])
def test_pole_faces_zeroed(grid, coord_name):
    """pgf_y carries the wall BC (v=0) at the physical pole faces."""
    hybrid = coord_name == "hybrid"
    coord = standard_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    state = hydrostatic_to_cgrid(
        rest_state_topography_init_latlon(grid, coord, h_0=2000.0), grid)
    p_half, p_full = _pressures(coord, state.p_s, hybrid)
    _, pgf_y = lin1997_pgf_latlon_cgrid(
        state.T, state.p_s, state.phis, p_half, p_full, grid)
    assert float(jnp.max(jnp.abs(pgf_y[0]))) == 0.0
    assert float(jnp.max(jnp.abs(pgf_y[-1]))) == 0.0


@pytest.mark.parametrize("coord_name", ["hybrid", "sigma"])
def test_rest_state_short_integration_bounded(grid, coord_name):
    """DCMIP 2-0-0 rest state stays near rest over 100 steps (lin1997).

    Not the full instability verdict (that needs 72x144 L40, matrix
    scale) — a tripwire that the balance is sane and nothing NaNs.
    """
    hybrid = coord_name == "hybrid"
    coord = standard_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    config = CGridLatLonPrimitiveEquationConfig(
        A_h=0.0, use_polar_filter=True, pgf_scheme="lin1997")
    dt = 300.0
    model = CGridLatLonPrimitiveEquationModel(grid, coord, config, dt=dt)
    state = hydrostatic_to_cgrid(
        rest_state_topography_init_latlon(grid, coord, h_0=2000.0), grid)
    for _ in range(100):
        state = model.step(state, dt)
    assert bool(jnp.all(jnp.isfinite(state.T)))
    # The bounded truncation imbalance excites an O(0.1-1) m/s
    # gravity-wave adjustment at this coarse resolution (DCMIP reference
    # models show the same order); the DEFECT signature is exponential
    # growth, so trip on growth, not the adjustment amplitude.
    # (sigma at this deliberately under-resolved 7.5° grid peaks ~3.8 m/s
    # in the first hours, then decays monotonically — measured step-600
    # value ~2.0.)
    max_100 = float(jnp.max(jnp.abs(state.u)))
    assert max_100 < 5.0
    for _ in range(100):
        state = model.step(state, dt)
    max_200 = float(jnp.max(jnp.abs(state.u)))
    assert bool(jnp.all(jnp.isfinite(state.T)))
    # two_term at matrix scale grows at 2.6 e-folds/day ⇒ ×2.4 over these
    # 100 extra steps (0.35 d).  Saturated adjustment must stay well under
    # that: allow mild transient reshuffling only.
    assert max_200 < 1.6 * max_100 + 1e-6, (max_100, max_200)


def test_unknown_pgf_scheme_raises(grid):
    coord = standard_hybrid_levels(NLEV)
    with pytest.raises(ValueError, match="pgf_scheme"):
        CGridLatLonPrimitiveEquationModel(
            grid, coord,
            CGridLatLonPrimitiveEquationConfig(pgf_scheme="lin97"), dt=600.0)
    state = hydrostatic_to_cgrid(
        rest_state_topography_init_latlon(grid, coord, h_0=2000.0), grid)
    with pytest.raises(ValueError, match="pgf_scheme"):
        cgrid_latlon_hydrostatic_tendencies(
            state, grid, coord,
            CGridLatLonPrimitiveEquationConfig(pgf_scheme="lin97"))


def test_default_scheme_is_two_term_legacy(grid):
    """Default config runs the legacy path (bit-compat guard for every
    existing latlon result until the default flip is decided)."""
    assert CGridLatLonPrimitiveEquationConfig().pgf_scheme == "two_term"


@pytest.mark.parametrize("coord_name", ["hybrid", "sigma"])
def test_lin_pgf_tendencies_differentiable(grid, coord_name):
    """jax.grad flows through the lin1997 momentum tendency (no nondiff
    ops, denominator floor keeps 0/0 out of the VJP)."""
    hybrid = coord_name == "hybrid"
    coord = standard_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    state = hydrostatic_to_cgrid(
        rest_state_topography_init_latlon(grid, coord, h_0=2000.0), grid)
    config = CGridLatLonPrimitiveEquationConfig(A_h=0.0, pgf_scheme="lin1997")

    def loss(T):
        du, dv, _, _, _ = cgrid_latlon_hydrostatic_tendencies(
            state._replace(T=T), grid, coord, config)
        return jnp.sum(du ** 2) + jnp.sum(dv ** 2)

    g = jax.grad(loss)(state.T)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0
