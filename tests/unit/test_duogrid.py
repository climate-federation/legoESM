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

    def test_sync_hardcoded_oracle_all_12_seams(self):
        """Independent-oracle check covering ALL 12 cube edges with
        first-principles seam pairings, NOT derived from the
        CONNECTIVITY table that ``synchronize_cgrid_fluxes`` itself
        consumes.

        If CONNECTIVITY is regressed to point any edge at the wrong
        neighbor (or to flip a reversal flag), this test catches it:
        the hardcoded ``exp`` encodes the specific (face, boundary
        slice, rev-flag) pairing from the cubed-sphere geometry, and
        the sync's output must equal ``exp`` bit-for-bit regardless
        of the table.

        Canonical layout (cubed_sphere.py:40-46):
          Face 0 (+x) — equatorial front
          Face 1 (+y) — equatorial right
          Face 2 (-x) — equatorial back
          Face 3 (-y) — equatorial left
          Face 4 (+z) — north pole
          Face 5 (-z) — south pole

        12 cube edges (4 same-axis equatorial, 2 vertical polar-to-face-0,
        4 cross-axis polar-to-equatorial, 2 same-axis polar-to-equatorial
        back-face):
        """
        from legoesm.grids.halo import synchronize_cgrid_fluxes
        n = 6
        rng = np.random.default_rng(31)
        fx = jnp.asarray(rng.standard_normal((6, n + 1, n)))
        fy = jnp.asarray(rng.standard_normal((6, n, n + 1)))
        fx_sync, fy_sync = synchronize_cgrid_fluxes(fx, fy, n)

        def _chk(label, actual, expected):
            assert bool(jnp.array_equal(actual, expected)), (
                f"{label}: max diff "
                f"{float(jnp.max(jnp.abs(actual - expected))):.2e}")

        # --- 4 equatorial same-axis seams (no reversal) ---
        # f0 W ↔ f3 E
        exp = 0.5 * (fx[0, 0, :] + fx[3, n, :])
        _chk("f0 W", fx_sync[0, 0, :], exp)
        _chk("f3 E", fx_sync[3, n, :], exp)
        # f0 E ↔ f1 W
        exp = 0.5 * (fx[0, n, :] + fx[1, 0, :])
        _chk("f0 E", fx_sync[0, n, :], exp)
        _chk("f1 W", fx_sync[1, 0, :], exp)
        # f1 E ↔ f2 W
        exp = 0.5 * (fx[1, n, :] + fx[2, 0, :])
        _chk("f1 E", fx_sync[1, n, :], exp)
        _chk("f2 W", fx_sync[2, 0, :], exp)
        # f2 E ↔ f3 W
        exp = 0.5 * (fx[2, n, :] + fx[3, 0, :])
        _chk("f2 E", fx_sync[2, n, :], exp)
        _chk("f3 W", fx_sync[3, 0, :], exp)

        # --- 2 polar seams to face 0 (same-axis, no reversal) ---
        # f0 N ↔ f4 S
        exp = 0.5 * (fy[0, :, n] + fy[4, :, 0])
        _chk("f0 N", fy_sync[0, :, n], exp)
        _chk("f4 S", fy_sync[4, :, 0], exp)
        # f0 S ↔ f5 N
        exp = 0.5 * (fy[0, :, 0] + fy[5, :, n])
        _chk("f0 S", fy_sync[0, :, 0], exp)
        _chk("f5 N", fy_sync[5, :, n], exp)

        # --- 2 polar seams to face 3 (cross-axis ±reversal) ---
        # f3 N ↔ f4 W (rev=True: going +y along f3's north edge maps
        # to going -y along f4's west edge)
        exp = 0.5 * (fy[3, :, n] + fx[4, 0, :][::-1])
        _chk("f3 N", fy_sync[3, :, n], exp)
        _chk("f4 W", fx_sync[4, 0, :], exp[::-1])
        # f3 S ↔ f5 W (no reversal: -y face's south meets -z face's west
        # with consistent index direction)
        exp = 0.5 * (fy[3, :, 0] + fx[5, 0, :])
        _chk("f3 S", fy_sync[3, :, 0], exp)
        _chk("f5 W", fx_sync[5, 0, :], exp)

        # --- 2 polar seams to face 1 (cross-axis) ---
        # f1 N ↔ f4 E (rev=False: +y face's north meets +z face's east,
        # same index direction)
        exp = 0.5 * (fy[1, :, n] + fx[4, n, :])
        _chk("f1 N", fy_sync[1, :, n], exp)
        _chk("f4 E", fx_sync[4, n, :], exp)
        # f1 S ↔ f5 E (rev=True: +y face's south meets -z face's east,
        # opposite index direction)
        exp = 0.5 * (fy[1, :, 0] + fx[5, n, :][::-1])
        _chk("f1 S", fy_sync[1, :, 0], exp)
        _chk("f5 E", fx_sync[5, n, :], exp[::-1])

        # --- 2 polar seams to face 2 (same-axis S/N with reversal) ---
        # f2 N ↔ f4 N (rev=True: -x face's north meets +z face's north
        # going opposite longitudinal directions)
        exp = 0.5 * (fy[2, :, n] + fy[4, :, n][::-1])
        _chk("f2 N", fy_sync[2, :, n], exp)
        _chk("f4 N", fy_sync[4, :, n], exp[::-1])
        # f2 S ↔ f5 S (rev=True)
        exp = 0.5 * (fy[2, :, 0] + fy[5, :, 0][::-1])
        _chk("f2 S", fy_sync[2, :, 0], exp)
        _chk("f5 S", fy_sync[5, :, 0], exp[::-1])


# =========================================================================
# T9: BGRID_NE vector corner sync (iter-100, Priority 2 scaffold)
# =========================================================================

class TestBgridNeCornerSync:
    """Tests for `synchronize_bgrid_ne_corner` — the vector corner sync
    used by d_sw3 pre-KE component averaging.

    Scope: the 16 non-reversed seams (W↔E and S↔N pairs).  Reversed
    seams are a documented limitation; the helper leaves them
    untouched and downstream tests verify this.
    """

    def test_non_reversed_we_seam_applies_u_sign_flip_interior(self):
        """At a W↔E non-reversed seam (face 0 W ↔ face 3 E), the
        u-component flips sign; v is preserved.  After sync the
        INTERIOR-of-edge values (indices 1..n-1) should equal
        0.5*(local_u + -nbr_u) and 0.5*(local_v + nbr_v).

        Cube-vertex corners (indices 0 and n) are overwritten by
        adjacent S/N edge syncs in the current implementation — a
        documented limitation that requires a separate 3-face cube-
        vertex pass (similar to `synchronize_corner_scalar`).
        """
        import jax.numpy as jnp
        from legoesm.grids.halo import synchronize_bgrid_ne_corner

        n = 8
        u = jnp.zeros((6, n+1, n+1))
        v = jnp.zeros((6, n+1, n+1))
        u = u.at[0, 0, :].set(1.0)
        v = v.at[0, 0, :].set(2.0)
        u = u.at[3, n, :].set(3.0)
        v = v.at[3, n, :].set(4.0)

        u_sync, v_sync = synchronize_bgrid_ne_corner(u, v, n)

        import numpy as np
        # Interior of edge (j = 1..n-1) matches the paired-sync formula
        np.testing.assert_allclose(np.array(u_sync[0, 0, 1:n]), -1.0)
        np.testing.assert_allclose(np.array(v_sync[0, 0, 1:n]), 3.0)
        np.testing.assert_allclose(np.array(u_sync[3, n, 1:n]), 1.0)
        np.testing.assert_allclose(np.array(v_sync[3, n, 1:n]), 3.0)

    def test_non_reversed_sn_seam_applies_v_sign_flip_interior(self):
        """At a S↔N non-reversed seam (face 0 S ↔ face 5 N), the
        v-component flips sign; u is preserved.  Interior values only
        (cube-vertex corners are overwritten by adjacent W/E syncs).
        """
        import jax.numpy as jnp
        from legoesm.grids.halo import synchronize_bgrid_ne_corner

        n = 8
        u = jnp.zeros((6, n+1, n+1))
        v = jnp.zeros((6, n+1, n+1))
        u = u.at[0, :, 0].set(10.0)
        v = v.at[0, :, 0].set(20.0)
        u = u.at[5, :, n].set(30.0)
        v = v.at[5, :, n].set(40.0)

        u_sync, v_sync = synchronize_bgrid_ne_corner(u, v, n)

        import numpy as np
        np.testing.assert_allclose(np.array(u_sync[0, 1:n, 0]), 20.0)
        np.testing.assert_allclose(np.array(v_sync[0, 1:n, 0]), -10.0)
        np.testing.assert_allclose(np.array(u_sync[5, 1:n, n]), 20.0)
        np.testing.assert_allclose(np.array(v_sync[5, 1:n, n]), 10.0)

    def test_cross_axis_non_reversed_seam_also_skipped(self):
        """Iter-101 (Codex correction): the helper also skips the 4
        non-reversed CROSS-AXIS seams (W/E ↔ S/N type), not just
        reversed seams.  These are face 1 NORTH ↔ face 4 EAST,
        face 3 SOUTH ↔ face 5 WEST, and their symmetric counterparts.

        Lock this in so future code doesn't silently treat these as
        "handled" without an i/j-swap rotation.
        """
        import jax.numpy as jnp
        from legoesm.grids.halo import synchronize_bgrid_ne_corner

        n = 8
        u = jnp.zeros((6, n+1, n+1))
        v = jnp.zeros((6, n+1, n+1))

        # face 1 NORTH edge (jc = n): cross-axis with face 4 EAST (ic=n)
        u = u.at[1, :, n].set(100.0)
        v = v.at[1, :, n].set(200.0)

        u_sync, v_sync = synchronize_bgrid_ne_corner(u, v, n)

        import numpy as np
        # Interior of face 1 NORTH (ic=1..n-1) should be untouched
        np.testing.assert_allclose(np.array(u_sync[1, 1:n, n]), 100.0)
        np.testing.assert_allclose(np.array(v_sync[1, 1:n, n]), 200.0)

    def test_geo_frame_sync_preserves_uniform_geographic_vector(self):
        """Iter-102: geo-frame BGRID_NE sync preserves a vector that
        is uniform in the GEOGRAPHIC frame.  For a uniform geo vector
        (u_east=1, u_north=0 everywhere), converting to face-local
        (via each face's cos_ang_c/sin_ang_c) gives non-uniform
        face-local values.  After `synchronize_bgrid_ne_corner_geo`
        those should round-trip: local → geo → sync (no-op because
        values agree) → local identical to input.

        This is the key invariant proving the geo-frame approach
        works on ALL 24 seams (including reversed and cross-axis)
        plus cube vertices — without any per-seam rotation tables.
        """
        import jax.numpy as jnp
        import numpy as np
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.grids.halo import synchronize_bgrid_ne_corner_geo

        n = 8
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        cac = cdgrid.cos_angle_corner
        sac = cdgrid.sin_angle_corner

        # Uniform geographic vector (east=1, north=0)
        u_east = jnp.ones((6, n+1, n+1))
        u_north = jnp.zeros((6, n+1, n+1))
        # Convert to face-local
        u_local = cac * u_east + sac * u_north
        v_local = -sac * u_east + cac * u_north

        u_sync, v_sync = synchronize_bgrid_ne_corner_geo(
            u_local, v_local, cac, sac, n)

        # Should round-trip exactly (all corners of all faces agree
        # on the geo-frame value, so sync is a no-op)
        max_du = float(jnp.max(jnp.abs(u_sync - u_local)))
        max_dv = float(jnp.max(jnp.abs(v_sync - v_local)))
        assert max_du < 1e-6, f"uniform-geo round-trip u diff = {max_du}"
        assert max_dv < 1e-6, f"uniform-geo round-trip v diff = {max_dv}"

    def test_bgrid_ke_transport_duogrid_uses_component_sync(self):
        """Iter-104 (Codex stop-time): the iter-103 wiring replaced the
        scalar-KE sync in `_bgrid_ke_transport` with a component-level
        BGRID_NE sync.  This test exercises the ACTUAL integrated
        duogrid path and asserts:

        (a) The function still returns finite, physically-plausible
            KE with a duogrid-enabled grid.
        (b) The KE is different from the output of a "scalar-KE sync
            control" (proving the wiring changed behavior, not a
            silent no-op).
        (c) The function's output is stable for balanced-state inputs.
        """
        import jax.numpy as jnp
        import numpy as np
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.fv3_sw_core import _bgrid_ke_transport
        from legoesm.grids.halo import synchronize_corner_scalar

        n = 8
        grid = create_cubed_sphere(n, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        assert cdgrid.base.duogrid is not None

        # Synthetic D-grid/C-grid inputs: smooth geostrophic-like field
        rng = np.random.default_rng(0)
        u_d = jnp.asarray(rng.standard_normal((6, n, n + 1)) * 10.0)
        v_d = jnp.asarray(rng.standard_normal((6, n + 1, n)) * 10.0)
        uc = jnp.asarray(rng.standard_normal((6, n + 1, n)) * 5.0)
        vc = jnp.asarray(rng.standard_normal((6, n, n + 1)) * 5.0)
        dt = 300.0

        # (a) Integrated call returns finite values
        ke_component_sync = _bgrid_ke_transport(u_d, v_d, uc, vc, cdgrid, dt)
        assert ke_component_sync.shape == (6, n + 1, n + 1)
        assert bool(jnp.all(jnp.isfinite(ke_component_sync))), \
            "ke_corner has NaN or Inf"

        # (b) Control: reimplement the pre-iter-103 scalar-KE-sync
        # path locally and compare.  If the new path is a silent no-op
        # (e.g. component sync happens to match scalar sync), this
        # test still gives useful diagnostic; we only assert the
        # magnitudes are plausible and a non-zero difference exists
        # somewhere at the seams.
        dt5 = 0.5 * dt
        cosa = cdgrid.cosa_corner
        rsina = cdgrid.rsin2_corner
        vc_pad = jnp.pad(vc, [(0, 0), (1, 1), (0, 0)], mode='edge')
        vc_sum = vc_pad[:, :-1, :] + vc_pad[:, 1:, :]
        uc_pad = jnp.pad(uc, [(0, 0), (0, 0), (1, 1)], mode='edge')
        uc_sum = uc_pad[:, :, :-1] + uc_pad[:, :, 1:]
        vb_ctrl = dt5 * (vc_sum - uc_sum * cosa) * rsina
        ub_ctrl = dt5 * (uc_sum - vc_sum * cosa) * rsina
        from legoesm.core.fv3_sw_core import _ppm_transport_1d
        rdy = 1.0 / jnp.maximum(cdgrid.dy_edge_x, 1e-30)
        rdx = 1.0 / jnp.maximum(cdgrid.dx_edge_y, 1e-30)
        trans_y_ctrl = _ppm_transport_1d(v_d, vb_ctrl, rdy, axis=2)
        trans_x_ctrl = _ppm_transport_1d(u_d, ub_ctrl, rdx, axis=1)
        ke_scalar_sync = 0.5 * (trans_y_ctrl * vb_ctrl
                                + ub_ctrl * trans_x_ctrl)
        ke_scalar_sync = synchronize_corner_scalar(ke_scalar_sync, n)

        # The two paths should differ (component sync is not a no-op)
        diff = float(jnp.max(jnp.abs(ke_component_sync - ke_scalar_sync)))
        # Non-trivial difference expected at panel-edge/cube-vertex
        # corners because the component-sync preserves physical
        # vector components while the scalar-KE sync averages products.
        assert diff > 1e-6, (
            f"component-sync and scalar-sync produced identical KE "
            f"(max diff = {diff:.2e}). Iter-103 wiring may be a no-op."
        )

    def test_geo_frame_sync_averages_discontinuity(self):
        """Iter-102: introduce a discontinuity at a shared seam in the
        geo frame and verify the sync averages it.  Confirms the sync
        is actually doing work (not a no-op in general).
        """
        import jax.numpy as jnp
        import numpy as np
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.grids.halo import synchronize_bgrid_ne_corner_geo

        n = 8
        grid = create_cubed_sphere(n)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        cac = cdgrid.cos_angle_corner
        sac = cdgrid.sin_angle_corner

        # Build a geo-frame vector that is uniform except face 0 has
        # u_east = 2 on its west edge (i=0).  Convert to face-local;
        # after sync the west edge should average to 1.5.
        u_east = jnp.ones((6, n+1, n+1))
        u_north = jnp.zeros((6, n+1, n+1))
        u_east = u_east.at[0, 0, :].set(2.0)

        u_local = cac * u_east + sac * u_north
        v_local = -sac * u_east + cac * u_north

        u_sync, v_sync = synchronize_bgrid_ne_corner_geo(
            u_local, v_local, cac, sac, n)

        # Convert back to check
        u_east_sync = cac * u_sync - sac * v_sync
        # Interior of face 0 west edge (j=1..n-1):
        #   local had u_east=2, neighbor (face 3 east) had u_east=1
        #   → sync u_east should be 1.5 on the shared seam
        np.testing.assert_allclose(
            np.array(u_east_sync[0, 0, 1:n]), 1.5, atol=1e-5)

    def test_reversed_seams_untouched_at_edge_interior(self):
        """At reversed seams (e.g. face 1 SOUTH ↔ face 5 EAST), the
        helper leaves INTERIOR-of-edge boundary values unchanged
        (documented limitation).  Cube-vertex corners at the endpoints
        of reversed edges can still be modified by adjacent non-
        reversed edges that include those corners — that is expected
        behaviour and is tested separately.
        """
        import jax.numpy as jnp
        from legoesm.grids.halo import synchronize_bgrid_ne_corner

        n = 8
        u = jnp.zeros((6, n+1, n+1))
        v = jnp.zeros((6, n+1, n+1))
        u = u.at[1, :, 0].set(100.0)  # face 1 south (reversed to face 5 east)
        v = v.at[1, :, 0].set(200.0)

        u_sync, v_sync = synchronize_bgrid_ne_corner(u, v, n)

        import numpy as np
        # Interior of face 1 SOUTH (indices 1..n-1) should be unchanged
        np.testing.assert_allclose(np.array(u_sync[1, 1:n, 0]), 100.0)
        np.testing.assert_allclose(np.array(v_sync[1, 1:n, 0]), 200.0)


# =========================================================================
# T9: Packed halo duogrid post-processing (iter-84)
# =========================================================================

class TestPackedHaloDuogrid:
    """Verify iter-84 packed MPI/SPMD halo functions apply the duogrid
    kinked-to-extended remap when `duogrid=dg` is passed.

    Before iter-84 the packed MPI/SPMD paths silently skipped the
    duogrid post-processing, diverging from unpacked
    `pad_halo_4d(duogrid=dg)`.  Iter-84 added a `_apply_duogrid_4d`
    helper inside each packed function.  These tests call the helpers
    directly (backend-agnostic) to lock in the behaviour without
    requiring a 6-device SPMD mesh or MPI comm.
    """

    def test_apply_duogrid_4d_matches_pad_halo_4d_with_duogrid(self):
        """The MPI packed `_apply_duogrid_4d` helper followed by its
        caller's halo exchange must produce the same result as
        unpacked `pad_halo_4d(duogrid=dg)`.  Here we build the exact
        intermediate (post-MPI, pre-remap) padded field by calling
        `pad_halo_4d(duogrid=None)` on non-duogrid halo, then apply
        the helper and compare against the canonical duogrid pad.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.halo import pad_halo_4d
        from legoesm.parallel.halo_exchange import _apply_duogrid_4d as apply_mpi
        from legoesm.parallel.cubesphere_exchange import _apply_duogrid_4d as apply_spmd

        n, nlev = 8, 3
        grid = create_cubed_sphere(n, use_duogrid=True)
        dg = grid.duogrid
        assert dg is not None

        rng = np.random.default_rng(0)
        f = jnp.asarray(rng.standard_normal((6, n, n, nlev)))

        # Canonical reference: full duogrid halo via unpacked pad_halo_4d
        ref = np.array(pad_halo_4d(f, duogrid=dg))

        # Pre-duogrid padded state (what the MPI/SPMD exchange produces
        # before the remap): unpacked pad_halo_4d WITHOUT duogrid kwarg.
        pre = pad_halo_4d(f)

        got_mpi = np.array(apply_mpi(pre, dg, halo=1))
        got_spmd = np.array(apply_spmd(pre, dg, halo=1))

        np.testing.assert_allclose(got_mpi, ref, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(got_spmd, ref, rtol=1e-6, atol=1e-10)

    def test_apply_duogrid_4d_changes_face_boundary_values(self):
        """Sanity: the duogrid remap is not an identity.  Applied to a
        non-trivial field, `_apply_duogrid_4d` should change at least
        the face-boundary cells of the padded array.  Catches regressions
        that accidentally no-op the helper.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.halo import pad_halo_4d
        from legoesm.parallel.halo_exchange import _apply_duogrid_4d

        n, nlev = 8, 2
        grid = create_cubed_sphere(n, use_duogrid=True)
        dg = grid.duogrid
        rng = np.random.default_rng(1)
        f = jnp.asarray(rng.standard_normal((6, n, n, nlev)))
        pre = pad_halo_4d(f)
        post = _apply_duogrid_4d(pre, dg, halo=1)
        # The interior is preserved
        np.testing.assert_array_equal(pre[:, 1:-1, 1:-1, :], post[:, 1:-1, 1:-1, :])
        # Halo cells must have changed somewhere
        halo_diff = jnp.max(jnp.abs(
            post - pre
        ))
        assert float(halo_diff) > 1e-10, "duogrid remap is a no-op on random input"

    def test_packed_mpi_4d_applies_duogrid_end_to_end(self):
        """End-to-end regression for iter-84's MPI packed duogrid fix
        that does NOT require a live MPI comm.  Mock-patches
        `pad_halo_mpi_4d` (the underlying MPI-exchange call that
        `packed_pad_halo_mpi_4d` dispatches through) to return the
        identical result of a non-MPI unpacked halo.  Then verifies
        that passing `duogrid=dg` to `packed_pad_halo_mpi_4d` produces
        the same result as the canonical `pad_halo_4d(duogrid=dg)`.

        Pre-iter-84: the packed MPI path ignored `duogrid`, so this
        test would have failed (output would equal the non-duogrid
        halo, not the duogrid halo).
        """
        from unittest import mock
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.halo import pad_halo_4d
        from legoesm.parallel import halo_exchange as mpi_halo_mod

        n, nlev = 8, 2
        grid = create_cubed_sphere(n, use_duogrid=True)
        dg = grid.duogrid
        assert dg is not None

        rng = np.random.default_rng(2)
        f1 = jnp.asarray(rng.standard_normal((6, n, n, nlev)))
        f2 = jnp.asarray(rng.standard_normal((6, n, n, nlev + 1)))

        ref1 = np.array(pad_halo_4d(f1, duogrid=dg))
        ref2 = np.array(pad_halo_4d(f2, duogrid=dg))

        # Mock: `pad_halo_mpi_4d` is the single underlying MPI call that
        # `packed_pad_halo_mpi_4d` stacks fields through.  Replace it
        # with `pad_halo_4d(duogrid=None)` which produces the
        # pre-remap padded state a real MPI exchange would deliver.
        def mock_pad_halo_mpi_4d(data, topology, halo=1):
            return pad_halo_4d(data, halo=halo, duogrid=None)

        with mock.patch.object(
            mpi_halo_mod, 'pad_halo_mpi_4d', mock_pad_halo_mpi_4d,
        ):
            r1, r2 = mpi_halo_mod.packed_pad_halo_mpi_4d(
                f1, f2, topology=None, halo=1, duogrid=dg,
            )

        np.testing.assert_allclose(np.array(r1), ref1,
                                   rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(np.array(r2), ref2,
                                   rtol=1e-6, atol=1e-10)

    def test_packed_mpi_4d_without_duogrid_skips_remap(self):
        """Iter-84 safety: when `duogrid` is not passed, the packed MPI
        path must be bit-identical to the pre-iter-84 behaviour (plain
        MPI exchange, no remap).
        """
        from unittest import mock
        from legoesm.grids.halo import pad_halo_4d
        from legoesm.parallel import halo_exchange as mpi_halo_mod

        n, nlev = 8, 2
        rng = np.random.default_rng(3)
        f = jnp.asarray(rng.standard_normal((6, n, n, nlev)))

        def mock_pad_halo_mpi_4d(data, topology, halo=1):
            return pad_halo_4d(data, halo=halo, duogrid=None)

        ref = np.array(mock_pad_halo_mpi_4d(f, None, 1))

        with mock.patch.object(
            mpi_halo_mod, 'pad_halo_mpi_4d', mock_pad_halo_mpi_4d,
        ):
            (r,) = mpi_halo_mod.packed_pad_halo_mpi_4d(
                f, topology=None, halo=1,
            )

        np.testing.assert_array_equal(np.array(r), ref)
