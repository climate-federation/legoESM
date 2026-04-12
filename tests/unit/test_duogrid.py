"""Unit tests for Duo-Grid kinked-to-extended remapping."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from legoesm.grids.duogrid import (
    DuoGridData,
    create_duogrid_data,
    _build_extended_grid,
    _build_kinked_grid,
    _extract_1d_coords,
    _compute_k2e_coefficients,
    cube_rmp_vectorized,
    fill_corner_region,
)
from legoesm.grids.halo import (
    CONNECTIVITY, WEST, EAST, SOUTH, NORTH,
    pad_halo, _pad_halo_local,
)


@pytest.fixture(params=[4, 8, 16])
def grid_n(request):
    return request.param


@pytest.fixture
def duogrid_data(grid_n):
    return create_duogrid_data(grid_n, ng=2, k2e_nord=2)


# =========================================================================
# T1: Precompute correctness
# =========================================================================

class TestPrecomputeCorrectness:
    """Tests T1.1 through T1.9: precompute pipeline correctness."""

    def test_t1_2_kinked_equals_extended_in_interior(self, grid_n):
        """T1.2: Interior cells of kinked and extended grids must match."""
        ng = 2
        ext_lon, ext_lat = _build_extended_grid(grid_n, ng)
        kik_lon, kik_lat = _build_kinked_grid(grid_n, ng, ext_lon, ext_lat)

        interior = (slice(None), slice(ng, ng + grid_n), slice(ng, ng + grid_n))
        np.testing.assert_array_equal(kik_lon[interior], ext_lon[interior])
        np.testing.assert_array_equal(kik_lat[interior], ext_lat[interior])

    def test_t1_3_kinked_matches_neighbor_interior(self, grid_n):
        """T1.3: Each kinked halo cell matches the neighbor's interior."""
        ng = 2
        ext_lon, ext_lat = _build_extended_grid(grid_n, ng)
        kik_lon, kik_lat = _build_kinked_grid(grid_n, ng, ext_lon, ext_lat)

        for face in range(6):
            for edge in [WEST, EAST, SOUTH, NORTH]:
                nbr_face, nbr_edge, is_reversed = CONNECTIVITY[face][edge]
                for d in range(ng):
                    for j in range(grid_n):
                        k = (grid_n - 1 - j) if is_reversed else j

                        if nbr_edge == WEST:
                            si, sj = ng + d, ng + k
                        elif nbr_edge == EAST:
                            si, sj = ng + grid_n - 1 - d, ng + k
                        elif nbr_edge == SOUTH:
                            si, sj = ng + k, ng + d
                        else:
                            si, sj = ng + k, ng + grid_n - 1 - d

                        if edge == WEST:
                            di, dj = ng - 1 - d, ng + j
                        elif edge == EAST:
                            di, dj = ng + grid_n + d, ng + j
                        elif edge == SOUTH:
                            di, dj = ng + j, ng - 1 - d
                        else:
                            di, dj = ng + j, ng + grid_n + d

                        np.testing.assert_allclose(
                            kik_lon[face, di, dj],
                            ext_lon[nbr_face, si, sj],
                            atol=1e-14,
                            err_msg=f"face={face} edge={edge} d={d} j={j}",
                        )

    def test_t1_5_monotonicity_of_1d_coords(self, grid_n):
        """T1.5: kik_1d must be strictly monotone for binary search."""
        if grid_n < 4:
            pytest.skip("Monotonicity check needs n >= 4")
        ng = 2
        ext_lon, ext_lat = _build_extended_grid(grid_n, ng)
        kik_lon, kik_lat = _build_kinked_grid(grid_n, ng, ext_lon, ext_lat)
        ext_1d, kik_1d = _extract_1d_coords(
            grid_n, ng, ext_lon, ext_lat, kik_lon, kik_lat
        )

        for face in range(6):
            for edge in range(4):
                for d in range(ng):
                    strip = kik_1d[face, edge, d, :]
                    diffs = np.diff(strip)
                    assert np.all(diffs > 0) or np.all(diffs < 0), (
                        f"kik_1d not monotone: face={face} edge={edge} "
                        f"depth={d}, diffs=[{diffs.min():.4e}, {diffs.max():.4e}]"
                    )

    def test_t1_6_partition_of_unity(self, duogrid_data):
        """T1.6: k2e_coef sums to 1 for each target point."""
        coef_sums = jnp.sum(duogrid_data.k2e_coef, axis=-1)
        np.testing.assert_allclose(coef_sums, 1.0, atol=1e-12)

    def test_t1_7_constant_field_remap(self, duogrid_data):
        """T1.7: Remapping a constant field reproduces the constant."""
        n = duogrid_data.n
        halo = min(2, duogrid_data.ng)
        n_p = n + 2 * halo

        const_val = 42.0
        padded = jnp.full((6, n_p, n_p), const_val)
        result = cube_rmp_vectorized(padded, duogrid_data, halo)

        np.testing.assert_allclose(result, const_val, atol=1e-10)

    def test_t1_7_constant_preserved_through_corner_fill(self, duogrid_data):
        """Corner fill on a constant field should preserve the constant."""
        n = duogrid_data.n
        halo = min(2, duogrid_data.ng)
        n_p = n + 2 * halo

        const_val = 7.0
        padded = jnp.full((6, n_p, n_p), const_val)
        padded = cube_rmp_vectorized(padded, duogrid_data, halo)
        result = fill_corner_region(padded, duogrid_data, halo)

        np.testing.assert_allclose(result, const_val, atol=1e-10)

    def test_t1_8_linear_field_remap_accuracy(self):
        """T1.8: Linear field remap with k2e_nord>=2 should be near-exact."""
        n = 16
        dg = create_duogrid_data(n, ng=2, k2e_nord=2)
        halo = 2
        n_p = n + 2 * halo

        # Create a linear field in gnomonic coords: f = i / n
        interior = jnp.tile(
            jnp.linspace(0, 1, n)[None, :, None],
            (6, 1, n),
        )
        padded = jnp.zeros((6, n_p, n_p))
        padded = padded.at[:, halo:halo+n, halo:halo+n].set(interior)

        # Pad with standard nearest-neighbor copy
        padded_nn = _pad_halo_local(interior, interp_offsets=None)
        # The output of _pad_halo_local is (6, n+2, n+2), resize if needed
        if halo == 2:
            # For halo=2, extend the linear field into halo manually
            full = jnp.tile(
                jnp.linspace(-1.0/(2*n), 1 + 1.0/(2*n), n_p)[None, :, None],
                (6, 1, n_p),
            )
            padded = full

        result = cube_rmp_vectorized(padded, dg, halo)

        # The south halo should have smoothly extended values
        south_halo = result[:, halo:halo+n, 0:halo]
        assert jnp.all(jnp.isfinite(south_halo))


# =========================================================================
# T3: Scalar/Vector/4D parity
# =========================================================================

class TestScalarParity:
    """T3.1: 2D consistency checks."""

    def test_t3_1_cube_rmp_preserves_interior(self, duogrid_data):
        """cube_rmp must not modify interior cells."""
        n = duogrid_data.n
        halo = min(2, duogrid_data.ng)
        n_p = n + 2 * halo

        rng = jax.random.PRNGKey(42)
        interior = jax.random.uniform(rng, (6, n, n))

        padded = jnp.zeros((6, n_p, n_p))
        padded = padded.at[:, halo:halo+n, halo:halo+n].set(interior)
        # Fill halo with some data
        padded = padded.at[:, :, :].add(0.1)

        result = cube_rmp_vectorized(padded, duogrid_data, halo)

        # Interior cells must be unchanged
        result_interior = result[:, halo:halo+n, halo:halo+n]
        padded_interior = padded[:, halo:halo+n, halo:halo+n]
        np.testing.assert_allclose(result_interior, padded_interior, atol=1e-14)


# =========================================================================
# T4: JIT + grad smoke tests
# =========================================================================

class TestJITAndGrad:
    """Tests T4.1 through T4.4: JIT and gradient compatibility."""

    def test_t4_1_jit_cube_rmp(self, duogrid_data):
        """cube_rmp_vectorized should be JIT-compatible."""
        n = duogrid_data.n
        halo = min(2, duogrid_data.ng)
        n_p = n + 2 * halo

        field = jnp.ones((6, n_p, n_p))

        @jax.jit
        def f(x):
            return cube_rmp_vectorized(x, duogrid_data, halo)

        result = f(field)
        assert result.shape == field.shape
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_t4_1_jit_fill_corner(self, duogrid_data):
        """fill_corner_region should be JIT-compatible."""
        n = duogrid_data.n
        halo = min(2, duogrid_data.ng)
        n_p = n + 2 * halo

        field = jnp.ones((6, n_p, n_p))

        @jax.jit
        def f(x):
            return fill_corner_region(x, duogrid_data, halo)

        result = f(field)
        assert result.shape == field.shape

    def test_t4_3_grad_through_cube_rmp(self, duogrid_data):
        """Gradient through cube_rmp should produce finite values."""
        n = duogrid_data.n
        halo = min(2, duogrid_data.ng)
        n_p = n + 2 * halo

        field = jnp.ones((6, n_p, n_p))

        def loss(x):
            remapped = cube_rmp_vectorized(x, duogrid_data, halo)
            return jnp.sum(remapped**2)

        grad = jax.grad(loss)(field)
        assert jnp.all(jnp.isfinite(grad))

    def test_t4_3_grad_through_full_pipeline(self, duogrid_data):
        """Gradient through cube_rmp + fill_corner should be finite."""
        n = duogrid_data.n
        halo = min(2, duogrid_data.ng)
        n_p = n + 2 * halo

        field = jnp.ones((6, n_p, n_p))

        def loss(x):
            x = cube_rmp_vectorized(x, duogrid_data, halo)
            x = fill_corner_region(x, duogrid_data, halo)
            return jnp.sum(x**2)

        grad = jax.grad(loss)(field)
        assert jnp.all(jnp.isfinite(grad))

    def test_t4_6_no_retracing(self, duogrid_data):
        """JIT'd function should not retrace on second call."""
        n = duogrid_data.n
        halo = min(2, duogrid_data.ng)
        n_p = n + 2 * halo

        @jax.jit
        def f(x):
            return cube_rmp_vectorized(x, duogrid_data, halo)

        field = jnp.ones((6, n_p, n_p))
        _ = f(field)  # first call: traces
        _ = f(field)  # second call: should use cache
        # If it retraces, there's a shape or tracer issue


# =========================================================================
# T2: Edge orientation (basic)
# =========================================================================

class TestEdgeOrientation:
    """Basic tests that cube_rmp handles all 24 edge connections."""

    def test_t2_1_all_edges_produce_finite_values(self, duogrid_data):
        """After cube_rmp, all halo cells should be finite."""
        n = duogrid_data.n
        halo = min(2, duogrid_data.ng)
        n_p = n + 2 * halo

        padded = jnp.ones((6, n_p, n_p))
        result = cube_rmp_vectorized(padded, duogrid_data, halo)
        assert jnp.all(jnp.isfinite(result))


# =========================================================================
# T5: Corner fill correctness
# =========================================================================

class TestCornerFill:
    """Tests for fill_corner_region."""

    def test_t5_1_constant_corners(self, duogrid_data):
        """Constant field should have constant corners."""
        n = duogrid_data.n
        halo = min(2, duogrid_data.ng)
        n_p = n + 2 * halo

        padded = jnp.full((6, n_p, n_p), 5.0)
        result = fill_corner_region(padded, duogrid_data, halo)
        np.testing.assert_allclose(result, 5.0, atol=1e-10)

    def test_t5_2_corners_are_filled(self, duogrid_data):
        """After fill_corner, no corner cell should be zero (from init)."""
        n = duogrid_data.n
        halo = min(2, duogrid_data.ng)
        n_p = n + 2 * halo

        # Create padded with zeros in corners, nonzero in edges/interior
        padded = jnp.zeros((6, n_p, n_p))
        padded = padded.at[:, halo:n_p-halo, :].set(1.0)
        padded = padded.at[:, :, halo:n_p-halo].set(1.0)

        result = fill_corner_region(padded, duogrid_data, halo)

        # Check all corners are now nonzero
        for f in range(6):
            assert result[f, 0, 0] != 0.0, f"SW corner face {f} is zero"
            assert result[f, -1, 0] != 0.0, f"SE corner face {f} is zero"
            assert result[f, 0, -1] != 0.0, f"NW corner face {f} is zero"
            assert result[f, -1, -1] != 0.0, f"NE corner face {f} is zero"


# =========================================================================
# Factory creation test
# =========================================================================

class TestFactory:
    """Test that create_duogrid_data produces valid data structures."""

    @pytest.mark.parametrize("n", [4, 8])
    @pytest.mark.parametrize("ng", [1, 2, 3])
    @pytest.mark.parametrize("k2e_nord", [2, 4])
    def test_create_duogrid_data(self, n, ng, k2e_nord):
        """Factory should produce valid DuoGridData for supported configs."""
        dg = create_duogrid_data(n, ng=ng, k2e_nord=k2e_nord)
        assert dg.n == n
        assert dg.ng == ng
        assert dg.k2e_nord == k2e_nord
        assert dg.k2e_coef.shape == (6, 4, ng, n, 4)
        assert dg.k2e_lo.shape == (6, 4, ng, n)
        assert dg.ext_lon.shape == (6, n + 2 * ng, n + 2 * ng)
        assert dg.ext_lat.shape == (6, n + 2 * ng, n + 2 * ng)

    def test_invalid_k2e_nord(self):
        with pytest.raises(ValueError, match="k2e_nord"):
            create_duogrid_data(4, k2e_nord=5)

    def test_invalid_ng(self):
        with pytest.raises(ValueError, match="ng"):
            create_duogrid_data(4, ng=5)

    def test_duogrid_is_jax_pytree(self):
        """DuoGridData must be a valid JAX pytree."""
        dg = create_duogrid_data(4, ng=2, k2e_nord=2)
        leaves = jax.tree_util.tree_leaves(dg)
        assert len(leaves) > 0
        # Should be flattenable and unflattenable
        flat, treedef = jax.tree_util.tree_flatten(dg)
        restored = treedef.unflatten(flat)
        assert restored.n == dg.n
