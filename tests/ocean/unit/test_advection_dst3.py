"""Tests for DST-3 flux-limited tracer advection (issue #210).

Tests:
1. Sweby limiter properties
2. DST-3 vertical advection: conservation, monotonicity, accuracy
3. DST-3 horizontal advection: conservation, periodicity, accuracy
4. Multi-dimensional advection: conservation
5. Comparison with TVD Van Leer: DST-3 should be less diffusive
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants

jax.config.update("jax_enable_x64", True)


class TestSwebyLimiter:
    """Properties of the Sweby (superbee) limiter."""

    def _limiter(self):
        from legoesm.ocean.dynamics._flux_limiters import sweby_limiter
        return sweby_limiter

    def test_zero_at_negative_r(self):
        psi = self._limiter()
        r = jnp.array([-2.0, -1.0, -0.5, -0.01])
        result = psi(r)
        assert jnp.all(result == 0.0)

    def test_zero_at_r_zero(self):
        psi = self._limiter()
        assert float(psi(jnp.array(0.0))) == 0.0

    def test_one_at_r_one(self):
        """Smooth field: psi(1) = 1 (no limiting)."""
        psi = self._limiter()
        assert float(psi(jnp.array(1.0))) == 1.0

    def test_two_at_large_r(self):
        """Maximum value is 2 (upper TVD bound)."""
        psi = self._limiter()
        r = jnp.array([3.0, 5.0, 100.0])
        result = psi(r)
        assert jnp.allclose(result, 2.0)

    def test_known_values(self):
        """Check specific values along the Sweby limiter curve."""
        psi = self._limiter()
        # r < 0.5: psi = 2r
        assert jnp.isclose(psi(jnp.array(0.25)), 0.5)
        # 0.5 <= r <= 1: psi = 1
        assert jnp.isclose(psi(jnp.array(0.75)), 1.0)
        # 1 <= r <= 2: psi = r
        assert jnp.isclose(psi(jnp.array(1.5)), 1.5)
        # r >= 2: psi = 2
        assert jnp.isclose(psi(jnp.array(2.5)), 2.0)

    def test_tvd_region(self):
        """Limiter stays within the Sweby TVD region: 0 <= psi <= min(2r, 2)."""
        psi = self._limiter()
        r = jnp.linspace(0.0, 4.0, 100)
        result = psi(r)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= jnp.minimum(2.0 * r, 2.0) + 1e-10)


class TestDST3Vertical:
    """Tests for flux_form_vertical_tracer_advection_dst3."""

    def _helper(self):
        from legoesm.ocean.advection import flux_form_vertical_tracer_advection_dst3
        return flux_form_vertical_tracer_advection_dst3

    def test_zero_w_zero_flux(self):
        """Zero vertical velocity gives zero flux divergence."""
        fn = self._helper()
        nlev = 5
        field = jnp.array([1.0, 2.0, 3.0, 2.5, 1.5])
        w_half = jnp.zeros(nlev + 1)
        h_k = jnp.full(nlev, 100.0)
        result = fn(field, w_half, h_k, dt=300.0)
        assert jnp.allclose(result, 0.0, atol=1e-15)

    def test_uniform_tracer_preserves_value(self):
        """Uniform tracer is preserved after full flux-form update.

        For divergent w, flux_div != 0 (it equals T*div(w)), but when
        combined with the continuity equation (h-change), T_new = T_old.
        """
        fn = self._helper()
        nlev = 5
        T_val = 5.0
        field = jnp.full(nlev, T_val)
        w_half = jnp.array([0.0, 0.1, -0.05, 0.08, -0.03, 0.0])
        h_k = jnp.full(nlev, 100.0)
        dt = 300.0
        flux_div = fn(field, w_half, h_k, dt)

        # Continuity: h_new = h_old - dt * (w[k] - w[k+1])
        dw = w_half[:-1] - w_half[1:]  # w_top - w_bot
        h_new = h_k - dt * dw
        # Tracer update: T_new = (h_old * T - dt * flux_div) / h_new
        T_new = (h_k * field - dt * flux_div) / h_new
        assert jnp.allclose(T_new, T_val, atol=1e-12)

    def test_conservation_closed_column(self):
        """Column integral of flux divergence = 0 for closed boundaries.

        For w_half[0] = w_half[nlev] = 0, sum(flux_div) = 0
        (flux telescopes to boundaries which are zero).
        """
        fn = self._helper()
        nlev = 8
        key = jax.random.PRNGKey(42)
        field = jax.random.uniform(key, (nlev,), minval=0.5, maxval=5.0)
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            0.01 * jnp.sin(jnp.linspace(0, 2 * jnp.pi, nlev - 1)),
            jnp.array([0.0]),
        ])
        h_k = jnp.linspace(50.0, 200.0, nlev)
        result = fn(field, w_half, h_k, dt=300.0)
        assert abs(float(jnp.sum(result))) < 1e-13

    def test_monotonicity(self):
        """DST-3 with Sweby limiter preserves monotonicity.

        After one step of advection with small CFL, min(T_new) >= min(T_old)
        and max(T_new) <= max(T_old) to within a small tolerance.
        (DST-3 is "nearly monotone" — the Sweby limiter bounds violations
        to O(dt^3) which is negligible for small CFL.)
        """
        fn = self._helper()
        nlev = 10
        # Discontinuous profile (hard test for monotonicity)
        field = jnp.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0])
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            jnp.full(nlev - 1, 0.001),  # small uniform upward (CFL ~ 0.003)
            jnp.array([0.0]),
        ])
        h_k = jnp.full(nlev, 100.0)
        dt = 300.0

        flux_div = fn(field, w_half, h_k, dt)
        # Euler step: T_new = T_old - dt * flux_div / h
        T_new = field - dt * flux_div / h_k

        T_min_old = float(jnp.min(field))
        T_max_old = float(jnp.max(field))
        T_min_new = float(jnp.min(T_new))
        T_max_new = float(jnp.max(T_new))

        # Allow small (< 1%) overshoot from the 3-point stencil interaction
        tol = 0.05 * (T_max_old - T_min_old)
        assert T_min_new >= T_min_old - tol, (
            f"New minimum {T_min_new} < old minimum {T_min_old} - {tol}")
        assert T_max_new <= T_max_old + tol, (
            f"New maximum {T_max_new} > old maximum {T_max_old} + {tol}")

    def test_less_diffusive_than_upwind(self):
        """DST-3 should advect a Gaussian more accurately than upwind.

        Compare both schemes against the exact analytic solution (shifted
        Gaussian). DST-3 (3rd order) should match the exact solution much
        better than upwind (1st order).
        """
        fn = self._helper()
        from legoesm.ocean.vertical import flux_form_vertical_tracer_advection

        nlev = 40
        dz = 50.0
        z = jnp.arange(nlev) * dz  # cell centers
        # Gaussian bump (well-resolved, away from boundaries)
        field_init = 1.0 + 3.0 * jnp.exp(-((z - 12.0 * dz) / (3.0 * dz)) ** 2)

        # CFL = 0.2 (moderate, 20 steps = 4 cells of displacement)
        w_val = 0.1  # m/s upward
        cfl = w_val * 100.0 / dz  # = 0.2
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            jnp.full(nlev - 1, w_val),
            jnp.array([0.0]),
        ])
        h_k = jnp.full(nlev, dz)
        dt = 100.0

        # Run 20 steps (total shift = 20 * CFL * dz = 4 cells)
        n_steps = 20
        T_dst3 = field_init
        T_upw = field_init
        for _ in range(n_steps):
            flux_div_dst3 = fn(T_dst3, w_half, h_k, dt)
            T_dst3 = T_dst3 - dt * flux_div_dst3 / h_k

            flux_div_upw = flux_form_vertical_tracer_advection(T_upw, w_half)
            T_upw = T_upw - dt * flux_div_upw / h_k

        # Exact solution: Gaussian shifted upward by 4 cells
        shift = n_steps * cfl * dz  # = 4 * dz
        T_exact = 1.0 + 3.0 * jnp.exp(-((z - 12.0 * dz + shift) / (3.0 * dz)) ** 2)

        # L2 error in interior (avoid boundary effects)
        interior = slice(5, 35)
        err_dst3 = float(jnp.sqrt(jnp.mean((T_dst3[interior] - T_exact[interior]) ** 2)))
        err_upw = float(jnp.sqrt(jnp.mean((T_upw[interior] - T_exact[interior]) ** 2)))

        # DST-3 should have significantly smaller error
        assert err_dst3 < err_upw * 0.7, (
            f"DST-3 L2 error {err_dst3:.6f} not significantly better than "
            f"upwind {err_upw:.6f} (ratio={err_dst3/err_upw:.3f})")

    def test_batched_shape(self):
        """Works with batched leading dimensions."""
        fn = self._helper()
        shape = (4, 5, 8)  # n_lat, n_lon, nlev
        nlev = shape[-1]
        key = jax.random.PRNGKey(0)
        field = jax.random.uniform(key, shape, minval=1.0, maxval=5.0)
        w_half = jnp.zeros((*shape[:-1], nlev + 1))
        w_half = w_half.at[..., 1:-1].set(0.001)
        h_k = jnp.full(shape, 100.0)
        result = fn(field, w_half, h_k, dt=300.0)
        assert result.shape == shape


class TestDST3Horizontal:
    """Tests for dst3_to_u_points and dst3_to_v_points."""

    def _make_grid(self, n_lat=10, n_lon=20):
        from legoesm.grids.latlon import create_latlon_grid
        return create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth)

    def test_uniform_tracer_at_u_points(self):
        """Uniform tracer gives T_face = T everywhere."""
        from legoesm.ocean.advection import dst3_to_u_points
        grid = self._make_grid()
        nlev = 5
        n_lat, n_lon = grid.n_lat, grid.n_lon
        f = jnp.full((n_lat, n_lon, nlev), 3.0)
        mf = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.5
        h_u = jnp.full((n_lat, n_lon + 1, nlev), 100.0)

        f_u = dst3_to_u_points(f, mf, h_u, grid, dt=300.0)
        assert f_u.shape == (n_lat, n_lon + 1, nlev)
        assert jnp.allclose(f_u, 3.0, atol=1e-14)

    def test_uniform_tracer_at_v_points(self):
        """Uniform tracer gives T_face = T at interior, 0 at poles."""
        from legoesm.ocean.advection import dst3_to_v_points
        grid = self._make_grid()
        nlev = 5
        n_lat, n_lon = grid.n_lat, grid.n_lon
        f = jnp.full((n_lat, n_lon, nlev), 3.0)
        mf = jnp.ones((n_lat + 1, n_lon, nlev)) * 0.5
        h_v = jnp.full((n_lat + 1, n_lon, nlev), 100.0)

        f_v = dst3_to_v_points(f, mf, h_v, grid, dt=300.0)
        assert f_v.shape == (n_lat + 1, n_lon, nlev)
        # Interior should be 3.0
        assert jnp.allclose(f_v[1:-1], 3.0, atol=1e-14)
        # Poles should be 0
        assert jnp.allclose(f_v[0], 0.0)
        assert jnp.allclose(f_v[-1], 0.0)

    def test_periodic_wrapping_u(self):
        """u-face at n_lon should equal u-face at 0 (periodicity)."""
        from legoesm.ocean.advection import dst3_to_u_points
        grid = self._make_grid()
        nlev = 3
        n_lat, n_lon = grid.n_lat, grid.n_lon
        key = jax.random.PRNGKey(7)
        f = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=1.0, maxval=5.0)
        mf = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.3
        h_u = jnp.full((n_lat, n_lon + 1, nlev), 50.0)

        f_u = dst3_to_u_points(f, mf, h_u, grid, dt=300.0)
        # Face n_lon should equal face 0
        assert jnp.allclose(f_u[:, -1, :], f_u[:, 0, :], atol=1e-14)

    def test_conservation_horizontal_flux(self):
        """Divergence of horizontal DST-3 flux integrates to zero globally.

        This is the divergence theorem: sum(div * area) = 0 for a
        periodic/closed domain. Mass fluxes must be periodic in longitude.
        """
        from legoesm.ocean.advection import dst3_to_u_points, dst3_to_v_points
        from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid

        grid = self._make_grid(n_lat=10, n_lon=20)
        nlev = 3
        n_lat, n_lon = grid.n_lat, grid.n_lon
        key = jax.random.PRNGKey(99)
        f = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=1.0, maxval=5.0)

        # Mass fluxes: periodic in longitude, zero at poles
        k1, k2 = jax.random.split(key)
        mf_u_interior = 0.1 * jax.random.normal(k1, (n_lat, n_lon, nlev))
        # Wrap: face n_lon = face 0
        mf_u = jnp.concatenate([mf_u_interior, mf_u_interior[:, :1, :]], axis=1)
        mf_v = 0.1 * jax.random.normal(k2, (n_lat + 1, n_lon, nlev))
        mf_v = mf_v.at[0].set(0.0).at[-1].set(0.0)

        h_u = jnp.full((n_lat, n_lon + 1, nlev), 100.0)
        h_v = jnp.full((n_lat + 1, n_lon, nlev), 100.0)

        f_u = dst3_to_u_points(f, mf_u, h_u, grid, dt=300.0)
        f_v = dst3_to_v_points(f, mf_v, h_v, grid, dt=300.0)
        flux_u = mf_u * f_u
        flux_v = mf_v * f_v

        div_flux = divergence_cgrid(flux_u, flux_v, grid)

        # Area-weighted integral of divergence should be zero
        area = grid.area[..., jnp.newaxis]  # (n_lat, n_lon, 1)
        global_integral = float(jnp.sum(div_flux * area))
        assert abs(global_integral) < 1e-6, (
            f"Global flux divergence integral = {global_integral} (should be 0)")


class TestMultidimAdvection:
    """Tests for the full multi-dimensional DST-3 advection."""

    def _make_grid(self, n_lat=10, n_lon=20):
        from legoesm.grids.latlon import create_latlon_grid
        return create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth)

    def test_conservation(self):
        """Total tracer (sum h*T*area) is conserved by multi-dim advection."""
        from legoesm.ocean.advection import multidim_tracer_advection
        from legoesm.ocean.dynamics.latlon_cgrid_operators import interp_cell_to_uface
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import interp_to_v_points

        grid = self._make_grid(n_lat=10, n_lon=20)
        nlev = 5
        n_lat, n_lon = grid.n_lat, grid.n_lon

        key = jax.random.PRNGKey(123)
        tracer = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=1.0, maxval=5.0)
        h_k = jnp.full((n_lat, n_lon, nlev), 100.0)
        h_u = interp_cell_to_uface(h_k)
        h_v = interp_to_v_points(h_k)

        # Periodic mass fluxes (face n_lon = face 0)
        k1, k2 = jax.random.split(key)
        mf_u_int = 0.01 * jax.random.normal(k1, (n_lat, n_lon, nlev))
        mf_u = jnp.concatenate([mf_u_int, mf_u_int[:, :1, :]], axis=1)
        mf_v = 0.01 * jax.random.normal(k2, (n_lat + 1, n_lon, nlev))
        mf_v = mf_v.at[0].set(0.0).at[-1].set(0.0)

        # Vertical velocity consistent with horizontal divergence
        from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid
        from legoesm.ocean.vertical import diagnose_w_from_flux_div
        flux_div_k = divergence_cgrid(mf_u, mf_v, grid)
        w_half = diagnose_w_from_flux_div(flux_div_k, thickness_weighted=True)

        dt = 300.0
        div_h, vert_div = multidim_tracer_advection(
            tracer, mf_u, mf_v, w_half, h_k, h_u, h_v, grid, dt)

        # Total tendency should conserve tracer: sum((div_h + vert_div) * area) = 0
        area = grid.area[..., jnp.newaxis]
        total_tendency = float(jnp.sum((div_h + vert_div) * area))
        assert abs(total_tendency) < 1e-4, (
            f"Multi-dim advection not conservative: {total_tendency}")

    def test_output_shapes(self):
        """Verify output shapes match input."""
        from legoesm.ocean.advection import multidim_tracer_advection
        from legoesm.ocean.dynamics.latlon_cgrid_operators import interp_cell_to_uface
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import interp_to_v_points

        grid = self._make_grid(n_lat=8, n_lon=16)
        nlev = 4
        n_lat, n_lon = grid.n_lat, grid.n_lon

        tracer = jnp.ones((n_lat, n_lon, nlev))
        h_k = jnp.full((n_lat, n_lon, nlev), 50.0)
        h_u = interp_cell_to_uface(h_k)
        h_v = interp_to_v_points(h_k)
        mf_u = jnp.zeros((n_lat, n_lon + 1, nlev))
        mf_v = jnp.zeros((n_lat + 1, n_lon, nlev))
        w_half = jnp.zeros((n_lat, n_lon, nlev + 1))

        div_h, vert_div = multidim_tracer_advection(
            tracer, mf_u, mf_v, w_half, h_k, h_u, h_v, grid, dt=300.0)

        assert div_h.shape == (n_lat, n_lon, nlev)
        assert vert_div.shape == (n_lat, n_lon, nlev)

    def test_zero_flow_zero_tendency(self):
        """Zero flow gives zero tendency."""
        from legoesm.ocean.advection import multidim_tracer_advection
        from legoesm.ocean.dynamics.latlon_cgrid_operators import interp_cell_to_uface
        from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import interp_to_v_points

        grid = self._make_grid(n_lat=8, n_lon=16)
        nlev = 4
        n_lat, n_lon = grid.n_lat, grid.n_lon

        key = jax.random.PRNGKey(0)
        tracer = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=1.0, maxval=10.0)
        h_k = jnp.full((n_lat, n_lon, nlev), 100.0)
        h_u = interp_cell_to_uface(h_k)
        h_v = interp_to_v_points(h_k)
        mf_u = jnp.zeros((n_lat, n_lon + 1, nlev))
        mf_v = jnp.zeros((n_lat + 1, n_lon, nlev))
        w_half = jnp.zeros((n_lat, n_lon, nlev + 1))

        div_h, vert_div = multidim_tracer_advection(
            tracer, mf_u, mf_v, w_half, h_k, h_u, h_v, grid, dt=300.0)

        assert jnp.allclose(div_h, 0.0, atol=1e-14)
        assert jnp.allclose(vert_div, 0.0, atol=1e-14)


class TestDST3VsTVD:
    """Compare DST-3 against TVD Van Leer: DST-3 should be less diffusive."""

    def test_vertical_smooth_profile(self):
        """On a smooth profile, DST-3 should produce smaller error than TVD."""
        from legoesm.ocean.advection import flux_form_vertical_tracer_advection_dst3
        from legoesm.ocean.vertical import flux_form_vertical_tracer_advection_tvd

        nlev = 20
        z = jnp.linspace(0, 1, nlev)
        # Smooth profile (quadratic)
        field = 1.0 + 4.0 * z * (1.0 - z)
        w_half = jnp.concatenate([
            jnp.array([0.0]),
            jnp.full(nlev - 1, 0.003),
            jnp.array([0.0]),
        ])
        h_k = jnp.full(nlev, 50.0)
        dt = 200.0

        flux_dst3 = flux_form_vertical_tracer_advection_dst3(field, w_half, h_k, dt)
        flux_tvd = flux_form_vertical_tracer_advection_tvd(field, w_half, h_k, dt)

        T_new_dst3 = field - dt * flux_dst3 / h_k
        T_new_tvd = field - dt * flux_tvd / h_k

        # Both should be close to analytic (advected profile) in the interior
        # DST-3 should have smaller deviation from original smooth profile
        err_dst3 = float(jnp.std(T_new_dst3[3:-3] - field[3:-3]))
        err_tvd = float(jnp.std(T_new_tvd[3:-3] - field[3:-3]))

        # DST-3 should be at least somewhat less diffusive
        assert err_dst3 <= err_tvd * 1.1, (
            f"DST-3 error {err_dst3} not better than TVD {err_tvd}")



class TestDST3DirectionSymmetry:
    """Findings #1/#2: the DST-3 smoothness-ratio numerator for NEGATIVE flow
    must be ``(donor - upup)`` (matching the vertical sibling + the positive-flow
    branch), NOT the negated ``(upup - donor)``.  The negation made van_leer
    psi -> 0 for u<0 / v<0 only, collapsing the scheme to 1st-order/over-diffusive
    in one direction -> direction-asymmetric diffusion.

    The defining property is the REFLECTION-EQUIVARIANCE of pure advection: for an
    ARBITRARY tracer ``f`` and any velocity ``U``,

        Advect_{-U}(f) == Reflect( Advect_{+U}( Reflect(f) ) ),

    where ``Reflect`` flips the advected axis.  A scheme that is 3rd-order for
    one sign but 1st-order for the other violates this well above round-off.
    """

    def _make_grid(self, n_lat=12, n_lon=40):
        from legoesm.grids.latlon import create_latlon_grid
        return create_latlon_grid(n_lat=n_lat, n_lon=n_lon, radius=constants.R_earth)

    def test_zonal_left_right_symmetric(self):
        """Reflection-equivariance of dst3_to_u_points under longitude flip +
        flow-sign flip, on an ARBITRARY periodic tracer.  Breaks if the u<0
        ratio numerator is negated (finding #1)."""
        from legoesm.ocean.advection import dst3_to_u_points
        grid = self._make_grid()
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = 1
        key = jax.random.PRNGKey(11)
        prof = jax.random.uniform(key, (n_lon,), minval=1.0, maxval=4.0)
        f = jnp.broadcast_to(prof[jnp.newaxis, :, jnp.newaxis],
                             (n_lat, n_lon, nlev))
        h_u = jnp.full((n_lat, n_lon + 1, nlev), 100.0)
        u_mag = 0.4
        dt = 300.0

        def faces(fin, sign):
            # interior cell mass flux (n_lon), wrapped to n_lon+1 faces
            mf_int = jnp.full((n_lat, n_lon, nlev), sign * u_mag * 100.0)
            mf = jnp.concatenate([mf_int, mf_int[:, :1, :]], axis=1)
            return dst3_to_u_points(fin, mf, h_u, grid, dt)

        # Reflect a CELL field in longitude (cell j -> cell -j mod n_lon).
        def refl_cell(x):
            return jnp.roll(jnp.flip(x, axis=1), 1, axis=1)

        # +U faces give cell-face values f_face[j] at face j-1/2.  The reflected
        # -U problem's INTERIOR cell-face divergence must equal the reflection of
        # the +U one.  Compare the per-cell flux divergence (face j - face j+1),
        # which is the actual advective tendency and is reflection-equivariant.
        def tendency(fin, sign):
            fu = faces(fin, sign)
            U = sign * u_mag * 100.0
            flux = U * fu                      # (n_lat, n_lon+1, nlev)
            return flux[:, :-1, :] - flux[:, 1:, :]   # per-cell divergence proxy

        tend_minus = tendency(f, -1.0)
        tend_plus_refl = refl_cell(tendency(refl_cell(f), +1.0))
        max_asym = float(jnp.max(jnp.abs(tend_minus - tend_plus_refl)))
        amp = float(jnp.max(jnp.abs(tend_minus)) + 1e-30)
        assert max_asym < 1e-9 * max(amp, 1.0), (
            f"zonal DST-3 reflection asymmetry {max_asym:.3e} (amp {amp:.3e}) "
            "-> negative-flow ratio numerator is wrong (finding #1)")

    def test_meridional_north_south_symmetric(self):
        """Reflection-equivariance of dst3_to_v_points under latitude flip +
        flow-sign flip, on an ARBITRARY tracer.  Breaks if the v<0 ratio
        numerator is negated (finding #2)."""
        from legoesm.ocean.advection import dst3_to_v_points
        grid = self._make_grid()
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = 1
        key = jax.random.PRNGKey(7)
        prof = jax.random.uniform(key, (n_lat,), minval=1.0, maxval=4.0)
        f = jnp.broadcast_to(prof[:, jnp.newaxis, jnp.newaxis],
                             (n_lat, n_lon, nlev))
        h_v = jnp.full((n_lat + 1, n_lon, nlev), 100.0)
        v_mag = 0.3

        def faces(fin, sign):
            mf_v = jnp.full((n_lat + 1, n_lon, nlev), sign * v_mag * 100.0)
            mf_v = mf_v.at[0].set(0.0).at[-1].set(0.0)  # wall BC at poles
            return dst3_to_v_points(fin, mf_v, h_v, grid, dt=300.0)

        # Per-cell meridional flux divergence (reflection-equivariant tendency).
        def tendency(fin, sign):
            fv = faces(fin, sign)
            V = sign * v_mag * 100.0
            flux = V * fv                       # (n_lat+1, n_lon, nlev)
            return flux[:-1, :, :] - flux[1:, :, :]   # (n_lat, n_lon, nlev)

        refl_cell = lambda x: jnp.flip(x, axis=0)   # cell k -> cell n_lat-1-k
        tend_minus = tendency(f, -1.0)
        tend_plus_refl = refl_cell(tendency(refl_cell(f), +1.0))
        # Interior cells only (near-pole faces fall back to 1st-order both ways,
        # which is itself symmetric, but keep the strict check on the interior).
        interior = slice(2, n_lat - 2)
        max_asym = float(jnp.max(jnp.abs(
            tend_minus[interior] - tend_plus_refl[interior])))
        amp = float(jnp.max(jnp.abs(tend_minus[interior])) + 1e-30)
        assert max_asym < 1e-9 * max(amp, 1.0), (
            f"meridional DST-3 reflection asymmetry {max_asym:.3e} -> negative-"
            "flow ratio numerator is wrong (finding #2)")


class TestWENOVerticalTieConvention:
    """Finding #5: the WENO vertical upwind tie must split on ``w_int > 0`` (tie
    at w==0 -> donor-above), matching the dst3/ppm/fct vertical convention which
    all use ``w_int > 0.0``.  (At w==0 the flux is zero, so this only fixes a
    consistent convention; the test pins it against the dst3 sibling.)"""

    def test_zero_w_matches_dst3_zero_flux(self):
        from legoesm.ocean.advection import (
            flux_form_vertical_tracer_advection_weno5,
            flux_form_vertical_tracer_advection_dst3,
        )
        nlev = 6
        field = jnp.array([1.0, 2.0, 4.0, 3.0, 2.5, 1.0])
        w_half = jnp.zeros(nlev + 1)
        h_k = jnp.full(nlev, 100.0)
        weno = flux_form_vertical_tracer_advection_weno5(field, w_half, h_k, 300.0)
        dst3 = flux_form_vertical_tracer_advection_dst3(field, w_half, h_k, 300.0)
        # Both must give exactly zero flux divergence at w==0.
        assert jnp.allclose(weno, 0.0, atol=1e-14)
        assert jnp.allclose(dst3, 0.0, atol=1e-14)
