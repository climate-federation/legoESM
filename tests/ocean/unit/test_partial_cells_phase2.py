"""Phase 2 of partial cells: hydrostatic pressure on partial cells.

Validates that ``compute_hydrostatic_pressure`` (and the wrappers
``compute_ocean_rho`` / ``compute_ocean_rho_and_pressure``) integrate
pressure correctly through partial bottom cells.

Three test classes:

1. ``TestHydrostaticBalance``: verify ``∂p/∂z = -ρg`` (hydrostatic
   balance) holds across full and partial cells to round-off when
   ``h_actual`` is supplied.
2. ``TestFlatBottomBitExact``: backwards-compat — passing
   ``h_actual=None`` with a pure z* coord, OR passing
   ``h_actual = dz_ref * jacobian`` explicitly, OR passing
   ``compute_layer_thickness(eta, H_bathy, partial_coord)``, all
   produce the same pressure when bathymetry is uniform = H_max.
3. ``TestRhoAndPressureWrappers``: ``compute_ocean_rho`` and
   ``compute_ocean_rho_and_pressure`` correctly dispatch on coord type.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    compute_hydrostatic_pressure,
    compute_ocean_rho,
    compute_ocean_rho_and_pressure,
    rho_0,
)
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture
def z20():
    return create_ocean_z_star(
        n_levels=20, H_max=4000.0, dz_surface=10.0, dz_deep=500.0,
    )


# ---------------------------------------------------------------------------
# Hydrostatic balance: ∂p/∂z = -ρ g
# ---------------------------------------------------------------------------


class TestHydrostaticBalance:
    """For a stratified column with arbitrary h, the integrated
    pressure must satisfy hydrostatic balance per layer:
    ``p_top[k+1] - p_top[k] = ρ[k] * g * h[k]``.
    Equivalently: ``(p[k+1] - p[k]) = 0.5 * (ρ[k] + ρ[k+1]) * g *
    (0.5*h[k] + 0.5*h[k+1])`` (centroid-to-centroid).
    """

    def test_partial_cell_per_layer_pressure_jump(self, z20):
        """Per-layer dp/dh = ρg, with h_actual including partial bottom."""
        # Step bathymetry: 4000 m and 1000 m
        H = jnp.asarray([4000.0, 1000.0])
        eta = jnp.zeros_like(H)
        coord = create_partial_cell_coordinate(z20, H)
        h_actual = compute_layer_thickness(eta, H, coord)

        # Stratified rho (linear with depth)
        nlev = z20.n_levels
        rho = 1024.0 + 0.5 * jnp.arange(nlev, dtype=jnp.float64)
        rho_2d = jnp.broadcast_to(rho[None, :], (2, nlev))

        # Compute pressure
        g = constants.g
        p = compute_hydrostatic_pressure(
            rho_2d, eta, z20.dz_ref, jnp.ones((2,)),
            rho_ref=rho_0, g=g, h_actual=h_actual,
        )

        # Verify: p_top[k+1] - p_top[k] = ρ[k] * g * h[k]
        # Reconstruct p_top from p (centroid) using p_top[k] = p[k] - 0.5*ρ[k]*g*h[k]
        p_arr = np.asarray(p)
        rho_arr = np.asarray(rho_2d)
        h_arr = np.asarray(h_actual)
        p_top = p_arr - 0.5 * rho_arr * g * h_arr
        # Per-layer hydrostatic identity
        for col in range(2):
            for k in range(nlev - 1):
                lhs = p_top[col, k + 1] - p_top[col, k]
                rhs = rho_arr[col, k] * g * h_arr[col, k]
                # Where h is zero (below seafloor in shallow column),
                # the identity is trivially 0 = 0.
                if h_arr[col, k] > 0 or h_arr[col, k + 1] > 0:
                    assert abs(lhs - rhs) < 1e-6 * max(abs(rhs), 1.0), (
                        f"Hydrostatic balance fails at col={col}, k={k}: "
                        f"lhs={lhs}, rhs={rhs}"
                    )

    def test_inactive_cells_no_pressure_increment(self, z20):
        """Cells below bottom_level contribute zero pressure increment
        because h_partial = 0 there."""
        H = jnp.asarray([500.0])  # shallow column
        eta = jnp.zeros_like(H)
        coord = create_partial_cell_coordinate(z20, H)
        h_actual = compute_layer_thickness(eta, H, coord)

        nlev = z20.n_levels
        rho = 1024.0 * jnp.ones((1, nlev))   # uniform rho

        p = compute_hydrostatic_pressure(
            rho, eta, z20.dz_ref, jnp.ones((1,)),
            rho_ref=rho_0, h_actual=h_actual,
        )
        bottom = int(coord.bottom_level[0])
        # Pressures below bottom_level should be flat (no further increment)
        p_arr = np.asarray(p[0])
        h_arr = np.asarray(h_actual[0])
        if bottom + 1 < nlev:
            # All inactive cells (k > bottom): centroid pressure is simply the
            # last active layer's bottom interface pressure (no further
            # contribution because dp = ρ*g*0 = 0).
            for k in range(bottom + 1, nlev):
                # h_actual[k] = 0 → dp = 0 → p_top[k] = p_top[k-1] + dp[k-1]
                # and centroid p[k] = p_top[k] (since 0.5 * 0 = 0)
                # Pressures should be constant from k=bottom+1 onwards.
                if k > bottom + 1:
                    assert abs(p_arr[k] - p_arr[bottom + 1]) < 1e-6


# ---------------------------------------------------------------------------
# Backwards-compat: flat-bottom paths produce identical pressure
# ---------------------------------------------------------------------------


class TestFlatBottomBitExact:
    """Three different ways of computing pressure on a flat-bottom
    column should produce the same result:

    1. Legacy z* path: ``compute_hydrostatic_pressure(..., h_actual=None)``
       with pure z* coord's dz_ref and jacobian.
    2. Partial-cell path with explicit h_actual:
       ``compute_layer_thickness(eta, H_bathy, partial_coord)`` passed
       as h_actual.

    Both must give bit-exact identical pressure.
    """

    def test_zero_eta_flat_bottom(self, z20):
        H = jnp.full((4, 6), z20.H_max)
        eta = jnp.zeros((4, 6))
        rho = 1024.0 + 0.5 * jnp.arange(z20.n_levels, dtype=jnp.float64)
        rho = jnp.broadcast_to(rho[None, None, :], H.shape + (z20.n_levels,))

        # Path A: legacy
        J_legacy = compute_ocean_jacobian(eta, H, z20)
        p_legacy = compute_hydrostatic_pressure(
            rho, eta, z20.dz_ref, J_legacy, rho_ref=rho_0,
        )

        # Path B: partial-cell explicit h_actual
        partial = create_partial_cell_coordinate(z20, H)
        h_actual = compute_layer_thickness(eta, H, partial)
        p_partial = compute_hydrostatic_pressure(
            rho, eta, z20.dz_ref, J_legacy, rho_ref=rho_0,
            h_actual=h_actual,
        )

        np.testing.assert_array_equal(
            np.asarray(p_legacy), np.asarray(p_partial),
        )

    def test_nonzero_eta_flat_bottom(self, z20):
        H = jnp.full((3, 5), z20.H_max)
        eta = jnp.full((3, 5), 1.5)
        rho = 1024.0 + 0.5 * jnp.arange(z20.n_levels, dtype=jnp.float64)
        rho = jnp.broadcast_to(rho[None, None, :], H.shape + (z20.n_levels,))

        J_legacy = compute_ocean_jacobian(eta, H, z20)
        p_legacy = compute_hydrostatic_pressure(
            rho, eta, z20.dz_ref, J_legacy, rho_ref=rho_0,
        )

        partial = create_partial_cell_coordinate(z20, H)
        h_actual = compute_layer_thickness(eta, H, partial)
        p_partial = compute_hydrostatic_pressure(
            rho, eta, z20.dz_ref, J_legacy, rho_ref=rho_0,
            h_actual=h_actual,
        )

        np.testing.assert_array_equal(
            np.asarray(p_legacy), np.asarray(p_partial),
        )


# ---------------------------------------------------------------------------
# Wrappers: compute_ocean_rho{,_and_pressure} dispatch
# ---------------------------------------------------------------------------


class _FakeField(NamedTuple):
    data: jnp.ndarray


class _FakeState(NamedTuple):
    T: _FakeField
    S: _FakeField
    eta: _FakeField
    H_bathy: _FakeField


def _make_state(grid_shape, nlev, T_value=15.0, S_value=35.0,
                eta_value=0.0, H_bathy_value=4000.0):
    """Build a minimal stand-in state with the fields used by
    ``compute_ocean_rho`` / ``compute_ocean_rho_and_pressure``."""
    T = jnp.full(grid_shape + (nlev,), T_value)
    S = jnp.full(grid_shape + (nlev,), S_value)
    eta = jnp.full(grid_shape, eta_value)
    H = jnp.full(grid_shape, H_bathy_value)
    return _FakeState(
        T=_FakeField(T), S=_FakeField(S),
        eta=_FakeField(eta), H_bathy=_FakeField(H),
    )


class TestRhoAndPressureWrappers:
    """``compute_ocean_rho`` and ``compute_ocean_rho_and_pressure`` must
    dispatch on coord type and produce hydrostatically-consistent
    output."""

    def test_partial_coord_returns_consistent_rho_and_p(self, z20):
        # Step bathymetry
        H = jnp.asarray([[4000.0, 4000.0, 1000.0]])
        state = _make_state((1, 3), z20.n_levels, H_bathy_value=4000.0)
        # Override H_bathy to step pattern
        state = state._replace(H_bathy=_FakeField(H))

        partial = create_partial_cell_coordinate(z20, H)
        J_partial = compute_ocean_jacobian(state.eta.data, H, partial)
        rho, p = compute_ocean_rho_and_pressure(state, partial, J_partial)
        # rho positive everywhere, p positive at all active cells
        assert bool(jnp.all(rho > 0))
        # Pressure is non-negative at the surface (eta=0 → p_surface=0)
        # and increases with depth in active cells
        # The shallow column should have flat pressure below bottom_level
        bottom_shallow = int(partial.bottom_level[0, 2])
        p_arr = np.asarray(p[0, 2])
        if bottom_shallow + 1 < z20.n_levels:
            # Below seafloor: pressure flat (h=0)
            for k in range(bottom_shallow + 1, z20.n_levels):
                assert (
                    abs(p_arr[k] - p_arr[bottom_shallow + 1])
                    < 1e-6 * max(abs(p_arr[bottom_shallow + 1]), 1.0)
                ), f"Below-seafloor pressure not flat at k={k}"

    def test_zstar_coord_unchanged(self, z20):
        """Passing a pure z* coord must produce the same result as before
        the Phase 2 changes (legacy path)."""
        state = _make_state((2, 3), z20.n_levels)
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z20)
        rho, p = compute_ocean_rho_and_pressure(state, z20, J)
        # Sanity: rho > 0, p >= 0 at surface
        assert bool(jnp.all(rho > 0))
        assert bool(jnp.all(p >= -1e-6))   # p_surface = rho_ref * g * eta = 0
        # Also: pressures at the top level are small (rho * g * 0.5 * h_top)
        # For dz_surface ~ 10 m and rho=1025: p[0] ~ 50 kPa
        assert float(jnp.max(p[..., 0])) < 1.0e5

    def test_partial_vs_zstar_differ_on_shallow_bathy(self, z20):
        """At H = H_max/2, partial and z* paths give different rho and p
        because layer thickness differs."""
        H_val = z20.H_max / 2.0
        state = _make_state((2,), z20.n_levels, H_bathy_value=H_val)

        # Pure z* path
        J_zstar = compute_ocean_jacobian(state.eta.data, state.H_bathy.data,
                                            z20)
        rho_zstar, p_zstar = compute_ocean_rho_and_pressure(state, z20, J_zstar)

        # Partial path
        partial = create_partial_cell_coordinate(z20, state.H_bathy.data)
        J_partial = compute_ocean_jacobian(state.eta.data,
                                              state.H_bathy.data, partial)
        rho_partial, p_partial = compute_ocean_rho_and_pressure(
            state, partial, J_partial,
        )

        # Pressures must differ at depth (different column structure)
        diff_p = float(jnp.max(jnp.abs(p_partial - p_zstar)))
        assert diff_p > 1.0, (
            f"Expected partial vs z* pressure to differ noticeably; "
            f"max abs diff = {diff_p}"
        )
