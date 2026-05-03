"""Phase 1 of partial cells: layer-thickness-aware operators.

Validates that ``compute_layer_thickness`` and
``compute_ocean_jacobian`` correctly dispatch on coord type:

- Pure ``OceanZStarCoordinate``: legacy formula
  ``h_k = dz_ref[k] * (eta + H_bathy) / H_max`` unchanged.
- ``OceanPartialCellCoordinate``: new formula
  ``h_k = h_partial[..., k] * (eta + H_bathy) / H_bathy``.

Critical regression check: when ``H_bathy = H_max`` everywhere (flat
bottom, no partial cells), both formulas give bit-exact identical
results.  This is the foundational guarantee for the Phase 6
backwards-compat gate.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
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
# Partial-cell layer thickness identity tests
# ---------------------------------------------------------------------------


class TestPartialCellLayerThickness:
    """``compute_layer_thickness`` on a partial-cell coord obeys
    ``h_k = h_partial[..., k] * (eta + H_bathy) / H_bathy``.
    """

    def test_zero_eta_returns_h_partial(self, z20):
        """When eta=0, layer thickness equals h_partial exactly."""
        H = jnp.asarray(np.linspace(500.0, 3500.0, 8).reshape(2, 4))
        coord = create_partial_cell_coordinate(z20, H)
        eta = jnp.zeros_like(H)
        h = compute_layer_thickness(eta, H, coord)
        np.testing.assert_allclose(
            np.asarray(h), np.asarray(coord.h_partial),
            rtol=1e-12,
        )

    def test_column_thickness_scales_with_eta(self, z20):
        """Sum h_k over k = (eta + H_bathy) per cell."""
        rng = np.random.default_rng(seed=7)
        H = jnp.asarray(rng.uniform(500.0, 3800.0, size=(3, 5)))
        eta = jnp.asarray(rng.uniform(-2.0, 2.0, size=(3, 5)))
        coord = create_partial_cell_coordinate(z20, H)
        h = compute_layer_thickness(eta, H, coord)
        col_sum = np.asarray(jnp.sum(h, axis=-1))
        expected = np.asarray(eta + H)
        np.testing.assert_allclose(col_sum, expected, rtol=1e-6)

    def test_uniform_compression_across_active_levels(self, z20):
        """All active layers scale by the same ratio under varying eta."""
        H = jnp.full((4,), 2000.0)
        coord = create_partial_cell_coordinate(z20, H)
        h0 = compute_layer_thickness(jnp.zeros_like(H), H, coord)
        h1 = compute_layer_thickness(jnp.full_like(H, 200.0), H, coord)
        # Ratio h1[active]/h0[active] should be (eta + H)/H = 2200/2000 = 1.1.
        is_active = np.asarray(coord.is_active)
        h0_arr = np.asarray(h0)
        h1_arr = np.asarray(h1)
        # Sample only active cells
        h0_active = h0_arr[is_active]
        h1_active = h1_arr[is_active]
        ratio = h1_active / h0_active
        # rtol=1e-6 tolerates float32 precision in h_partial.
        np.testing.assert_allclose(ratio, 1.1, rtol=1e-6)

    def test_inactive_cells_stay_zero(self, z20):
        """Cells below bottom_level have zero thickness regardless of eta."""
        H = jnp.full((4,), 500.0)
        coord = create_partial_cell_coordinate(z20, H)
        for eta_val in [0.0, 5.0, -2.0, 100.0]:
            eta = jnp.full_like(H, eta_val)
            h = compute_layer_thickness(eta, H, coord)
            inactive = ~np.asarray(coord.is_active)
            assert float(jnp.max(jnp.abs(h * inactive))) == 0.0

    def test_dry_column_zero_thickness(self, z20):
        """A column with H_bathy <= 0 has zero thickness everywhere."""
        H = jnp.asarray([0.0, -10.0, 100.0])  # mix dry / wet
        coord = create_partial_cell_coordinate(z20, H)
        eta = jnp.asarray([0.0, 0.0, 0.0])
        h = compute_layer_thickness(eta, H, coord)
        # Dry columns: total thickness = 0
        assert float(jnp.sum(h[0])) == 0.0
        assert float(jnp.sum(h[1])) == 0.0
        # Wet column: total thickness = H_bathy (~100)
        assert abs(float(jnp.sum(h[2])) - 100.0) < 1e-6

    def test_min_water_column_clip(self, z20):
        """``min_water_column_m`` clips ``eta + H_bathy`` from below."""
        H = jnp.full((3,), 100.0)
        eta = jnp.full((3,), -200.0)  # would give negative water column
        coord = create_partial_cell_coordinate(z20, H)
        h = compute_layer_thickness(eta, H, coord, min_water_column_m=10.0)
        # Effective water column floored to 10 m → sum h_k = 10
        col_sum = np.asarray(jnp.sum(h, axis=-1))
        np.testing.assert_allclose(col_sum, 10.0, rtol=1e-6)


class TestPartialCellJacobian:
    """``compute_ocean_jacobian`` on partial-cell coord = (eta+H)/H_bathy."""

    def test_zero_eta_jacobian_is_one(self, z20):
        H = jnp.asarray([500.0, 1500.0, 3000.0])
        coord = create_partial_cell_coordinate(z20, H)
        J = compute_ocean_jacobian(jnp.zeros_like(H), H, coord)
        np.testing.assert_allclose(np.asarray(J), 1.0, rtol=1e-12)

    def test_jacobian_scales_with_eta(self, z20):
        H = jnp.full((3,), 2000.0)
        coord = create_partial_cell_coordinate(z20, H)
        J = compute_ocean_jacobian(jnp.full_like(H, 200.0), H, coord)
        np.testing.assert_allclose(np.asarray(J), 1.1, rtol=1e-12)

    def test_partial_jacobian_differs_from_zstar_at_shallow_H(self, z20):
        """At H = H_max/2, partial Jacobian = (1+eta/H) but z\\* Jacobian
        = (H/2 + eta)/H_max = 0.5 + eta/H_max — they differ in the
        shallow-water case (this is the whole point of partial cells)."""
        H = jnp.full((3,), 2000.0)  # H_max = 4000, so H = H_max/2
        eta = jnp.full((3,), 0.0)
        partial_coord = create_partial_cell_coordinate(z20, H)
        J_partial = compute_ocean_jacobian(eta, H, partial_coord)
        J_zstar = compute_ocean_jacobian(eta, H, z20)
        # J_partial = 1.0, J_zstar = 0.5
        assert abs(float(J_partial[0]) - 1.0) < 1e-12
        assert abs(float(J_zstar[0]) - 0.5) < 1e-12


# ---------------------------------------------------------------------------
# Backwards-compat regression: flat bottom is bit-exact
# ---------------------------------------------------------------------------


class TestFlatBottomBitExact:
    """When H_bathy = H_max everywhere, ``compute_layer_thickness`` on
    a partial-cell coord must give the SAME result as on a pure-z\\* coord.
    This is the Phase 6 backwards-compat foundation: existing experiments
    using the legacy z\\* path produce identical numbers under the new code.
    """

    def test_flat_bottom_zero_eta(self, z20):
        H = jnp.full((4, 6), z20.H_max)
        eta = jnp.zeros((4, 6))
        partial_coord = create_partial_cell_coordinate(z20, H)
        h_partial_path = compute_layer_thickness(eta, H, partial_coord)
        h_zstar_path = compute_layer_thickness(eta, H, z20)
        np.testing.assert_array_equal(
            np.asarray(h_partial_path), np.asarray(h_zstar_path),
        )

    def test_flat_bottom_nonzero_eta(self, z20):
        H = jnp.full((4, 6), z20.H_max)
        eta = jnp.full((4, 6), 2.5)   # 2.5 m surface displacement
        partial_coord = create_partial_cell_coordinate(z20, H)
        h_partial_path = compute_layer_thickness(eta, H, partial_coord)
        h_zstar_path = compute_layer_thickness(eta, H, z20)
        np.testing.assert_array_equal(
            np.asarray(h_partial_path), np.asarray(h_zstar_path),
        )

    def test_flat_bottom_jacobian_bit_exact(self, z20):
        H = jnp.full((4, 6), z20.H_max)
        eta = jnp.full((4, 6), 1.0)
        partial_coord = create_partial_cell_coordinate(z20, H)
        J_partial = compute_ocean_jacobian(eta, H, partial_coord)
        J_zstar = compute_ocean_jacobian(eta, H, z20)
        np.testing.assert_array_equal(
            np.asarray(J_partial), np.asarray(J_zstar),
        )


# ---------------------------------------------------------------------------
# Pure z* path unchanged
# ---------------------------------------------------------------------------


class TestPureZStarPathUnchanged:
    """Calling compute_layer_thickness with an OceanZStarCoordinate
    must produce the same output as before this commit (pure z\\*
    formula)."""

    def test_pure_zstar_formula(self, z20):
        H = jnp.asarray(np.linspace(500.0, 4000.0, 12).reshape(3, 4))
        eta = jnp.asarray(np.linspace(-1.0, 1.0, 12).reshape(3, 4))
        h = compute_layer_thickness(eta, H, z20)
        # Reference formula: dz_ref * (eta + H) / H_max
        J = (eta + H) / z20.H_max
        expected = z20.dz_ref[None, None, :] * J[..., None]
        np.testing.assert_allclose(
            np.asarray(h), np.asarray(expected), rtol=1e-12,
        )

    def test_pure_zstar_jacobian(self, z20):
        H = jnp.full((4,), 1000.0)
        eta = jnp.full((4,), 0.5)
        J = compute_ocean_jacobian(eta, H, z20)
        expected = (eta + H) / z20.H_max
        np.testing.assert_allclose(
            np.asarray(J), np.asarray(expected), rtol=1e-12,
        )
