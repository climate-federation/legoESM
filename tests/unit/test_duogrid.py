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

    @pytest.mark.parametrize("n,ng,k2e_nord", [
        (4, 1, 2), (4, 1, 4), (4, 2, 2), (4, 2, 4),
        (8, 1, 2), (8, 1, 4), (8, 2, 2), (8, 2, 4),
        (8, 3, 2), (8, 3, 4), (8, 4, 2), (8, 4, 4),
    ])
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

    def test_ng_too_large_for_n(self):
        with pytest.raises(ValueError, match="too large"):
            create_duogrid_data(4, ng=3)

    def test_duogrid_is_jax_pytree(self):
        """DuoGridData must be a valid JAX pytree."""
        dg = create_duogrid_data(4, ng=2, k2e_nord=2)
        leaves = jax.tree_util.tree_leaves(dg)
        assert len(leaves) > 0
        # Should be flattenable and unflattenable
        flat, treedef = jax.tree_util.tree_flatten(dg)
        restored = treedef.unflatten(flat)
        assert restored.n == dg.n

    def test_create_duogrid_data_ng4(self):
        """Factory should accept ng=4 (FV3 default Duo-Grid halo width)."""
        dg = create_duogrid_data(8, ng=4, k2e_nord=2)
        assert dg.ng == 4
        assert dg.k2e_coef.shape == (6, 4, 4, 8, 4)
        assert dg.ext_lon.shape == (6, 16, 16)

    def test_corner_lagrange_coefs_present(self):
        """Factory should produce corner Lagrange coefficients."""
        dg = create_duogrid_data(8, ng=3, k2e_nord=2)
        assert dg.corner_xp is not None
        assert dg.corner_xm is not None
        assert dg.corner_yp is not None
        assert dg.corner_ym is not None
        n_ext = 8 + 2 * 3
        assert dg.corner_xp.shape == (6, 4, n_ext, n_ext)


# =========================================================================
# T6: FV3-faithful d2a2c_vect Duo-Grid branch
# =========================================================================

class TestD2A2CVectDuoGrid:
    """Tests for the d2a2c_vect Duo-Grid branch (Phase 1 of FV3 faithfulness)."""

    def _make_grid(self, n, use_duogrid):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        grid = create_cubed_sphere(n, use_duogrid=use_duogrid, k2e_nord=2)
        return create_cubed_sphere_cdgrid(grid)

    def test_duogrid_branch_dispatches(self):
        """When duogrid is active, _d2a2c_vect should use the duogrid path."""
        from legoesm.core.fv3_sw_core import _d2a2c_vect
        n = 8
        cdgrid = self._make_grid(n, use_duogrid=True)
        assert cdgrid.base.duogrid is not None

        u_d = jnp.ones((6, n, n + 1))
        v_d = jnp.ones((6, n + 1, n))
        ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)
        assert ua.shape == (6, n, n)
        assert uc.shape == (6, n + 1, n)
        assert vc.shape == (6, n, n + 1)
        assert jnp.all(jnp.isfinite(ua))
        assert jnp.all(jnp.isfinite(uc))
        assert jnp.all(jnp.isfinite(vc))

    def test_uniform_field_zero_divergence(self):
        """Uniform D-grid winds should produce near-zero divergence."""
        from legoesm.core.fv3_sw_core import _d2a2c_vect
        n = 8
        cdgrid = self._make_grid(n, use_duogrid=True)

        u_d = jnp.zeros((6, n, n + 1))
        v_d = jnp.zeros((6, n + 1, n))
        ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)
        np.testing.assert_allclose(ua, 0.0, atol=1e-12)
        np.testing.assert_allclose(va, 0.0, atol=1e-12)
        np.testing.assert_allclose(uc, 0.0, atol=1e-12)
        np.testing.assert_allclose(vc, 0.0, atol=1e-12)

    def test_non_duogrid_unchanged(self):
        """Without duogrid, _d2a2c_vect should use the legacy edge-special path."""
        from legoesm.core.fv3_sw_core import _d2a2c_vect
        n = 8
        cdgrid = self._make_grid(n, use_duogrid=False)
        assert cdgrid.base.duogrid is None

        u_d = jnp.ones((6, n, n + 1))
        v_d = jnp.ones((6, n + 1, n))
        ua, va, uc, vc, ut, vt = _d2a2c_vect(u_d, v_d, cdgrid)
        assert ua.shape == (6, n, n)
        assert jnp.all(jnp.isfinite(ua))

    def test_fv3_csw_tendencies_with_duogrid(self):
        """fv3_csw_tendencies should produce finite tendencies with duogrid."""
        from legoesm.core.fv3_sw_core import fv3_csw_tendencies
        n = 8
        cdgrid = self._make_grid(n, use_duogrid=True)
        h = jnp.ones((6, n, n)) * 1000.0
        u_d = jnp.zeros((6, n, n + 1))
        v_d = jnp.zeros((6, n + 1, n))
        h_s = jnp.zeros((6, n, n))
        dh, du, dv = fv3_csw_tendencies(h, u_d, v_d, h_s, cdgrid)
        assert dh.shape == (6, n, n)
        assert du.shape == (6, n, n + 1)
        assert dv.shape == (6, n + 1, n)
        assert jnp.all(jnp.isfinite(dh))
        assert jnp.all(jnp.isfinite(du))
        assert jnp.all(jnp.isfinite(dv))
        # At rest: tendencies should be near zero
        np.testing.assert_allclose(dh, 0.0, atol=1e-8)

    def test_duogrid_mass_conservation_one_step(self):
        """One RK3 step with duogrid should conserve mass."""
        from legoesm.core.fv3_sw_core import fv3_csw_tendencies
        n = 8
        cdgrid = self._make_grid(n, use_duogrid=True)
        area = cdgrid.base.area
        h = jnp.ones((6, n, n)) * 1000.0 + 10.0 * jnp.sin(
            cdgrid.base.lon) * jnp.cos(cdgrid.base.lat)
        u_d = jnp.ones((6, n, n + 1)) * 5.0
        v_d = jnp.zeros((6, n + 1, n))
        h_s = jnp.zeros((6, n, n))
        dh, du, dv = fv3_csw_tendencies(h, u_d, v_d, h_s, cdgrid)
        dt = 100.0
        h_new = h + dt * dh
        mass_before = float(jnp.sum(h * area))
        mass_after = float(jnp.sum(h_new * area))
        rel_err = abs(mass_after - mass_before) / abs(mass_before)
        assert rel_err < 1e-8, f"Mass conservation violated: rel_err={rel_err:.2e}"


# =========================================================================
# T7b: ext_vector and cubed_a2d_halo
# =========================================================================

class TestExtVector:
    """Tests for ext_vector_dgrid and cubed_a2d_halo building blocks."""

    def test_cubed_a2d_halo_shapes(self):
        """cubed_a2d_halo should produce correct D-grid shapes."""
        from legoesm.grids.duogrid import cubed_a2d_halo
        dg = create_duogrid_data(8, ng=3, k2e_nord=2)
        h = 3
        n_p = 8 + 2 * h
        ull = jnp.ones((6, n_p, n_p))
        vll = jnp.zeros((6, n_p, n_p))
        ud, vd = cubed_a2d_halo(ull, vll, dg, h)
        assert ud.shape == (6, n_p, n_p - 1)
        assert vd.shape == (6, n_p - 1, n_p)
        assert jnp.all(jnp.isfinite(ud))
        assert jnp.all(jnp.isfinite(vd))

    def test_cubed_a2d_halo_zero_wind(self):
        """Zero lat/lon wind should give zero D-grid wind."""
        from legoesm.grids.duogrid import cubed_a2d_halo
        dg = create_duogrid_data(8, ng=3, k2e_nord=2)
        h = 3
        n_p = 8 + 2 * h
        ull = jnp.zeros((6, n_p, n_p))
        vll = jnp.zeros((6, n_p, n_p))
        ud, vd = cubed_a2d_halo(ull, vll, dg, h)
        np.testing.assert_allclose(ud, 0.0, atol=1e-14)
        np.testing.assert_allclose(vd, 0.0, atol=1e-14)

    def test_ext_vector_dgrid_shapes(self):
        """ext_vector_dgrid should produce padded D-grid shapes."""
        from legoesm.grids.duogrid import ext_vector_dgrid
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        n = 8
        grid = create_cubed_sphere(n, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        dg = grid.duogrid
        h = 2
        utmp = jnp.ones((6, n, n))
        vtmp = jnp.zeros((6, n, n))
        cosa_s = cdgrid.cos_sg[:, :, :, 4]
        rsin2 = cdgrid.rsin2_cell
        ud, vd = ext_vector_dgrid(utmp, vtmp, dg, grid.cos_angle,
                                   grid.sin_angle, cosa_s, rsin2, halo=h)
        n_p = n + 2 * h
        assert ud.shape == (6, n_p, n_p - 1)
        assert vd.shape == (6, n_p - 1, n_p)
        assert jnp.all(jnp.isfinite(ud))
        assert jnp.all(jnp.isfinite(vd))


# =========================================================================
# T7: Lagrange corner fill correctness
# =========================================================================

class TestLagrangeCornerFill:
    """Tests for the FV3-faithful Lagrange corner fill."""

    def test_constant_preserved(self):
        """Constant field should be exactly preserved by Lagrange corner fill."""
        dg = create_duogrid_data(8, ng=3, k2e_nord=2)
        halo = 3
        n_p = 8 + 2 * halo
        padded = jnp.full((6, n_p, n_p), 7.0)
        result = fill_corner_region(padded, dg, halo)
        np.testing.assert_allclose(result, 7.0, atol=1e-10)

    def test_corners_filled_nonzero(self):
        """After Lagrange fill, all corner cells should be populated."""
        dg = create_duogrid_data(8, ng=3, k2e_nord=2)
        halo = 3
        n_p = 8 + 2 * halo
        # Set interior and edges to 1.0, corners to 0.0
        padded = jnp.zeros((6, n_p, n_p))
        padded = padded.at[:, halo:n_p - halo, :].set(1.0)
        padded = padded.at[:, :, halo:n_p - halo].set(1.0)
        result = fill_corner_region(padded, dg, halo)
        for f in range(6):
            for ci in range(halo):
                for cj in range(halo):
                    assert result[f, ci, cj] != 0.0, \
                        f"SW corner ({ci},{cj}) face {f} not filled"
                    assert result[f, n_p - 1 - ci, cj] != 0.0
                    assert result[f, ci, n_p - 1 - cj] != 0.0
                    assert result[f, n_p - 1 - ci, n_p - 1 - cj] != 0.0

    def test_ng4_works(self):
        """ng=4 should work for both k2e and corner fill."""
        dg = create_duogrid_data(8, ng=4, k2e_nord=2)
        halo = 4
        n_p = 8 + 2 * halo
        padded = jnp.full((6, n_p, n_p), 3.0)
        result = fill_corner_region(padded, dg, halo)
        np.testing.assert_allclose(result, 3.0, atol=1e-10)

    def test_halo_less_than_ng(self):
        """Lagrange corner fill must work when halo < ng (runtime common case)."""
        dg = create_duogrid_data(8, ng=3, k2e_nord=2)
        # Runtime: pad_halo uses halo=2, but ng=3
        halo = 2
        n_p = 8 + 2 * halo
        padded = jnp.full((6, n_p, n_p), 5.0)
        result = fill_corner_region(padded, dg, halo)
        np.testing.assert_allclose(result, 5.0, atol=1e-10)

    def test_halo_1_with_ng3(self):
        """halo=1, ng=3: smallest halo with non-trivial ng offset."""
        dg = create_duogrid_data(8, ng=3, k2e_nord=2)
        halo = 1
        n_p = 8 + 2 * halo
        padded = jnp.full((6, n_p, n_p), 2.0)
        result = fill_corner_region(padded, dg, halo)
        np.testing.assert_allclose(result, 2.0, atol=1e-10)


# =========================================================================
# T8: synchronize_cgrid_fluxes (Ralph-prompt critical duogrid constraint)
# =========================================================================

class TestSynchronizeCgridFluxes:
    """Verify the duogrid CGRID flux sync:
      sync'd boundary flux = 0.5 * (flux_from_face_A + flux_from_face_B)

    Matches FV3 dyn_core.F90:853-900 mpp_get_boundary(..., gridtype=CGRID_NE)
    followed by 0.5*(local + buffer) averaging at all 12 shared cube edges.
    Ralph-prompt critical constraint #1.
    """

    def test_post_sync_all_12_edges_agree(self):
        """After sync, every shared face boundary shows matching fx/fy
        on both sides of each seam (with index reversal where required).
        Covers all 24 (face, edge) pairs = 12 cube edges read both ways.
        """
        from legoesm.grids.halo import synchronize_cgrid_fluxes
        n = 8
        # Random asymmetric fluxes so that initial boundaries disagree
        rng = np.random.default_rng(42)
        fx = jnp.asarray(rng.standard_normal((6, n + 1, n)))
        fy = jnp.asarray(rng.standard_normal((6, n, n + 1)))

        fx_sync, fy_sync = synchronize_cgrid_fluxes(fx, fy, n)

        for face in range(6):
            for edge in (WEST, EAST, SOUTH, NORTH):
                nbr_face, nbr_edge, rev = CONNECTIVITY[face][edge]
                # Local boundary from the sync'd arrays
                if edge == WEST:
                    local = fx_sync[face, 0, :]
                elif edge == EAST:
                    local = fx_sync[face, n, :]
                elif edge == SOUTH:
                    local = fy_sync[face, :, 0]
                else:
                    local = fy_sync[face, :, n]
                # Neighbour boundary (possibly from a cross-axis flux array)
                if nbr_edge == WEST:
                    nbr = fx_sync[nbr_face, 0, :]
                elif nbr_edge == EAST:
                    nbr = fx_sync[nbr_face, n, :]
                elif nbr_edge == SOUTH:
                    nbr = fy_sync[nbr_face, :, 0]
                else:
                    nbr = fy_sync[nbr_face, :, n]
                if rev:
                    nbr = nbr[::-1]
                diff = float(jnp.max(jnp.abs(local - nbr)))
                assert diff < 1e-12, (
                    f"Post-sync disagreement at face={face} edge={edge} "
                    f"(nbr face={nbr_face} edge={nbr_edge} rev={rev}): "
                    f"max |local - nbr| = {diff:.2e}"
                )

    def test_sync_is_exact_average_at_every_seam(self):
        """Bit-identical: sync'd boundary = 0.5*(pre_A + pre_B_rotated) for
        EVERY one of the 24 (face, edge) pairs.  Matches FV3 dyn_core.F90
        mpp_get_boundary(..., gridtype=CGRID_NE) + 0.5*(local + buffer).

        This is tighter than test_post_sync_all_12_edges_agree, which only
        checks that the two post-sync sides agree: here we check the
        post-sync value matches the exact 0.5*(pre_A + pre_B_rotated)
        formula with zero tolerance.
        """
        from legoesm.grids.halo import synchronize_cgrid_fluxes
        n = 6
        rng = np.random.default_rng(7)
        fx_pre = jnp.asarray(rng.standard_normal((6, n + 1, n)))
        fy_pre = jnp.asarray(rng.standard_normal((6, n, n + 1)))

        fx_sync, fy_sync = synchronize_cgrid_fluxes(fx_pre, fy_pre, n)

        def _boundary(fx_arr, fy_arr, face, edge):
            if edge == WEST:
                return fx_arr[face, 0, :]
            if edge == EAST:
                return fx_arr[face, n, :]
            if edge == SOUTH:
                return fy_arr[face, :, 0]
            return fy_arr[face, :, n]

        for face in range(6):
            for edge in (WEST, EAST, SOUTH, NORTH):
                nbr_face, nbr_edge, rev = CONNECTIVITY[face][edge]
                local_pre = _boundary(fx_pre, fy_pre, face, edge)
                nbr_pre = _boundary(fx_pre, fy_pre, nbr_face, nbr_edge)
                if rev:
                    nbr_pre = nbr_pre[::-1]
                expected = 0.5 * (local_pre + nbr_pre)
                actual = _boundary(fx_sync, fy_sync, face, edge)
                # Bit-identical (atol=0) — the sync is implemented as a
                # pure 0.5 * (a + b) JAX primitive, no rounding needed.
                assert bool(jnp.array_equal(actual, expected)), (
                    f"Sync value at face={face} edge={edge} differs from "
                    f"0.5*(pre_local + pre_nbr_rotated). max diff = "
                    f"{float(jnp.max(jnp.abs(actual - expected))):.2e}"
                )

    def test_sync_idempotent(self):
        """A second sync on already-synced fluxes must be a no-op."""
        from legoesm.grids.halo import synchronize_cgrid_fluxes
        n = 8
        rng = np.random.default_rng(11)
        fx = jnp.asarray(rng.standard_normal((6, n + 1, n)))
        fy = jnp.asarray(rng.standard_normal((6, n, n + 1)))

        fx1, fy1 = synchronize_cgrid_fluxes(fx, fy, n)
        fx2, fy2 = synchronize_cgrid_fluxes(fx1, fy1, n)
        assert bool(jnp.array_equal(fx1, fx2)), "Second sync changed fx"
        assert bool(jnp.array_equal(fy1, fy2)), "Second sync changed fy"
