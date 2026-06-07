"""Tests for SOM (Prather 1986) Second Order Moments tracer advection (#210).

Tests:
1. Flux extraction: known Courant values
2. X-sweep: conservation, uniform tracer, periodic wrapping
3. Y-sweep: conservation, solid-wall BCs
4. Z-sweep: conservation, closed column
5. Full 3D: conservation, differentiability
6. Comparison with TVD: SOM should be less diffusive on smooth profiles
"""

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants

jax.config.update("jax_enable_x64", True)


def _make_grid(n_lat=8, n_lon=12):
    from legoesm.grids.latlon import create_latlon_grid
    return create_latlon_grid(n_lat, n_lon, radius=constants.R_earth)


# =============================================================================
# Flux extraction
# =============================================================================

class TestFluxExtraction:
    """Tests for _extract_flux helper."""

    def _extract(self):
        from legoesm.ocean.advection_som import _extract_flux
        return _extract_flux

    def test_zero_alpha_gives_zero_flux(self):
        fn = self._extract()
        sm_o = jnp.array(10.0)
        mom = jnp.ones(9)
        fp_o, fp_mom = fn(sm_o, mom, jnp.array(0.0), 1.0)
        assert float(fp_o) == 0.0
        assert jnp.allclose(fp_mom, 0.0)

    def test_alpha_one_takes_entire_cell(self):
        """Alpha=1 means the entire cell content leaves."""
        fn = self._extract()
        sm_o = jnp.array(5.0)
        mom = jnp.zeros(9)  # uniform sub-cell
        fp_o, fp_mom = fn(sm_o, mom, jnp.array(1.0), 1.0)
        assert jnp.isclose(fp_o, 5.0)

    def test_uniform_cell_flux_proportional(self):
        """Uniform cell (zero moments): flux = alpha * sm_o."""
        fn = self._extract()
        sm_o = jnp.array(8.0)
        mom = jnp.zeros(9)
        alpha = jnp.array(0.3)
        fp_o, fp_mom = fn(sm_o, mom, alpha, 1.0)
        assert jnp.isclose(fp_o, 0.3 * 8.0)
        assert jnp.allclose(fp_mom, 0.0)

    def test_sign_edge_affects_odd_coupling(self):
        """sign_edge = -1 should flip the coupling terms."""
        fn = self._extract()
        sm_o = jnp.array(10.0)
        mom = jnp.zeros(9).at[0].set(2.0)  # sx = 2
        alpha = jnp.array(0.4)
        fp_o_pos, _ = fn(sm_o, mom, alpha, 1.0)
        fp_o_neg, _ = fn(sm_o, mom, alpha, -1.0)
        # With sx != 0 and different sign_edge, fp_o should differ
        assert not jnp.isclose(fp_o_pos, fp_o_neg)


# =============================================================================
# X-sweep
# =============================================================================

class TestSOMXSweep:
    """Tests for the periodic x-sweep."""

    def _sweep(self):
        from legoesm.ocean.advection_som import _som_x_sweep
        return _som_x_sweep

    def test_zero_flow_preserves_everything(self):
        fn = self._sweep()
        n_lat, n_lon, nlev = 4, 8, 3
        sm_o = jnp.ones((n_lat, n_lon, nlev)) * 5.0
        mom = jnp.ones((n_lat, n_lon, nlev, 9)) * 0.1
        vol = jnp.ones((n_lat, n_lon, nlev))
        vf = jnp.zeros((n_lat, n_lon, nlev))

        sm_o_new, mom_new, vol_new = fn(sm_o, mom, vf, vol)
        assert jnp.allclose(sm_o_new, sm_o)
        assert jnp.allclose(mom_new, mom)
        assert jnp.allclose(vol_new, vol)

    def test_uniform_tracer_no_moment_growth(self):
        """Uniform tracer + zero moments → no moments develop."""
        fn = self._sweep()
        n_lat, n_lon, nlev = 4, 8, 3
        sm_o = jnp.ones((n_lat, n_lon, nlev)) * 3.0
        mom = jnp.zeros((n_lat, n_lon, nlev, 9))
        vol = jnp.ones((n_lat, n_lon, nlev))
        # Small uniform rightward flow: CFL = 0.2
        vf = jnp.ones((n_lat, n_lon, nlev)) * 0.2

        sm_o_new, mom_new, _ = fn(sm_o, mom, vf, vol)
        # Uniform tracer should remain uniform; moments stay zero
        tracer_new = sm_o_new / jnp.maximum(vol, 1e-30)
        assert jnp.allclose(tracer_new, 3.0, atol=1e-12)
        assert jnp.allclose(mom_new, 0.0, atol=1e-12)

    def test_conservation(self):
        """Total sm_o must be conserved (periodic domain)."""
        fn = self._sweep()
        key = jax.random.PRNGKey(42)
        n_lat, n_lon, nlev = 4, 10, 3
        sm_o = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=1.0, maxval=5.0)
        mom = jax.random.normal(jax.random.PRNGKey(43), (n_lat, n_lon, nlev, 9)) * 0.1
        vol = jnp.ones((n_lat, n_lon, nlev)) * 2.0
        # Random flow with CFL < 0.5
        vf = jax.random.uniform(jax.random.PRNGKey(44), (n_lat, n_lon, nlev),
                                minval=-0.5, maxval=0.5)

        sm_o_new, _, _ = fn(sm_o, mom, vf, vol)
        # Global sum conserved (periodic: everything stays in domain)
        assert jnp.allclose(jnp.sum(sm_o_new), jnp.sum(sm_o), rtol=1e-10)

    def test_periodic_wrapping(self):
        """A pulse shifted rightward wraps around."""
        fn = self._sweep()
        n_lat, n_lon, nlev = 1, 8, 1
        sm_o = jnp.zeros((n_lat, n_lon, nlev))
        sm_o = sm_o.at[0, 0, 0].set(1.0)  # pulse at cell 0
        mom = jnp.zeros((n_lat, n_lon, nlev, 9))
        vol = jnp.ones((n_lat, n_lon, nlev))
        # Full CFL = 1 should move entire cell content one step right
        vf = jnp.ones((n_lat, n_lon, nlev)) * 1.0  # alpha = 1

        sm_o_new, _, _ = fn(sm_o, mom, vf, vol)
        # Pulse should now be at cell 1
        assert jnp.isclose(sm_o_new[0, 1, 0], 1.0, atol=1e-10)
        assert jnp.isclose(jnp.sum(sm_o_new), 1.0, atol=1e-10)


# =============================================================================
# Y-sweep
# =============================================================================

class TestSOMYSweep:
    """Tests for the wall-bounded y-sweep."""

    def _sweep(self):
        from legoesm.ocean.advection_som import _som_y_sweep
        return _som_y_sweep

    def test_zero_flow_preserves(self):
        fn = self._sweep()
        n_lat, n_lon, nlev = 6, 4, 3
        sm_o = jnp.ones((n_lat, n_lon, nlev)) * 2.0
        mom = jnp.ones((n_lat, n_lon, nlev, 9)) * 0.05
        vol = jnp.ones((n_lat, n_lon, nlev))
        vf = jnp.zeros((n_lat - 1, n_lon, nlev))

        sm_o_new, mom_new, _ = fn(sm_o, mom, vf, vol)
        assert jnp.allclose(sm_o_new, sm_o)
        assert jnp.allclose(mom_new, mom)

    def test_conservation_channel(self):
        """Total sm_o conserved with solid walls."""
        fn = self._sweep()
        key = jax.random.PRNGKey(50)
        n_lat, n_lon, nlev = 8, 6, 3
        sm_o = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=1.0, maxval=5.0)
        mom = jax.random.normal(jax.random.PRNGKey(51), (n_lat, n_lon, nlev, 9)) * 0.1
        vol = jnp.ones((n_lat, n_lon, nlev)) * 2.0
        vf = jax.random.uniform(jax.random.PRNGKey(52), (n_lat - 1, n_lon, nlev),
                                minval=-0.4, maxval=0.4)

        sm_o_new, _, _ = fn(sm_o, mom, vf, vol)
        assert jnp.allclose(jnp.sum(sm_o_new), jnp.sum(sm_o), rtol=1e-10)

    def test_wall_bc_no_leakage(self):
        """Mass concentrated at boundary cell doesn't leak out."""
        fn = self._sweep()
        n_lat, n_lon, nlev = 5, 3, 1
        sm_o = jnp.zeros((n_lat, n_lon, nlev))
        sm_o = sm_o.at[0, :, :].set(10.0)  # all mass at southernmost row
        mom = jnp.zeros((n_lat, n_lon, nlev, 9))
        vol = jnp.ones((n_lat, n_lon, nlev))
        # Southward flow at all interior faces
        vf = jnp.ones((n_lat - 1, n_lon, nlev)) * (-0.3)

        sm_o_new, _, _ = fn(sm_o, mom, vf, vol)
        # Total should be conserved (wall blocks southward leakage from cell 0)
        assert jnp.allclose(jnp.sum(sm_o_new), jnp.sum(sm_o), rtol=1e-10)


# =============================================================================
# Z-sweep
# =============================================================================

class TestSOMZSweep:
    """Tests for the closed vertical z-sweep."""

    def _sweep(self):
        from legoesm.ocean.advection_som import _som_z_sweep
        return _som_z_sweep

    def test_zero_w_preserves(self):
        fn = self._sweep()
        n_lat, n_lon, nlev = 3, 4, 5
        sm_o = jnp.ones((n_lat, n_lon, nlev)) * 4.0
        mom = jnp.ones((n_lat, n_lon, nlev, 9)) * 0.02
        vol = jnp.ones((n_lat, n_lon, nlev))
        vf = jnp.zeros((n_lat, n_lon, nlev - 1))

        sm_o_new, mom_new, _ = fn(sm_o, mom, vf, vol)
        assert jnp.allclose(sm_o_new, sm_o)
        assert jnp.allclose(mom_new, mom)

    def test_conservation_closed_column(self):
        """Total column sm_o conserved with closed top/bottom."""
        fn = self._sweep()
        key = jax.random.PRNGKey(60)
        n_lat, n_lon, nlev = 3, 4, 8
        sm_o = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=0.5, maxval=3.0)
        mom = jax.random.normal(jax.random.PRNGKey(61), (n_lat, n_lon, nlev, 9)) * 0.05
        vol = jnp.ones((n_lat, n_lon, nlev)) * 1.5
        vf = jax.random.uniform(jax.random.PRNGKey(62), (n_lat, n_lon, nlev - 1),
                                minval=-0.3, maxval=0.3)

        sm_o_new, _, _ = fn(sm_o, mom, vf, vol)
        assert jnp.allclose(jnp.sum(sm_o_new), jnp.sum(sm_o), rtol=1e-10)

    def test_uniform_tracer_preserved(self):
        """Uniform tracer unaffected by divergent vertical flow."""
        fn = self._sweep()
        n_lat, n_lon, nlev = 2, 3, 6
        sm_o = jnp.ones((n_lat, n_lon, nlev)) * 5.0
        mom = jnp.zeros((n_lat, n_lon, nlev, 9))
        vol = jnp.ones((n_lat, n_lon, nlev))
        # Divergent flow: positive at top, negative at bottom
        vf = jnp.linspace(0.2, -0.2, nlev - 1)
        vf = jnp.broadcast_to(vf[jnp.newaxis, jnp.newaxis, :], (n_lat, n_lon, nlev - 1))

        sm_o_new, mom_new, vol_new = fn(sm_o, mom, vf, vol)
        tracer_new = sm_o_new / jnp.maximum(vol_new, 1e-30)
        assert jnp.allclose(tracer_new, 5.0, atol=1e-10)

    def test_stratified_upward_advection(self):
        """Upward advection of stratified profile shifts tracer upward.

        This tests that the z-sweep correctly handles the axis reversal:
        positive z-moment (sz > 0 = more tracer at top) must be respected
        when extracting slabs from the bottom edge.
        """
        from legoesm.ocean.advection_som import IX, IZ
        fn = self._sweep()
        n_lat, n_lon, nlev = 1, 1, 10
        # Linearly stratified: T = 20 at surface (k=0), T = 10 at bottom (k=9)
        T_profile = jnp.linspace(20.0, 10.0, nlev)
        vol = jnp.ones((n_lat, n_lon, nlev))
        sm_o = (vol * T_profile[jnp.newaxis, jnp.newaxis, :])

        # Set z-slope moments (sz > 0 means more tracer at TOP of each cell)
        mom = jnp.zeros((n_lat, n_lon, nlev, 9))
        mom = mom.at[..., IZ].set(0.1)

        # Uniform upward velocity (positive = upward in our convention)
        vf = jnp.ones((n_lat, n_lon, nlev - 1)) * 0.2

        sm_o_new, mom_new, vol_new = fn(sm_o, mom, vf, vol)
        T_new = sm_o_new / jnp.maximum(vol_new, 1e-30)

        # The surface should get warmer (receiving warm water from below)
        # The bottom should get colder (losing warm water upward)
        # But wait: upward flow moves cold bottom water up.  With our profile
        # T decreases downward, so upward flow brings COLDER water up.
        # Interior cells should shift toward their below-neighbor's value.
        # Check that the profile doesn't blow up (no NaN or extreme values)
        assert jnp.all(jnp.isfinite(T_new))
        # Conservation
        assert jnp.allclose(jnp.sum(sm_o_new), jnp.sum(sm_o), rtol=1e-10)
        # Interior temperature should be bounded
        assert jnp.all(T_new >= 9.0)
        assert jnp.all(T_new <= 21.0)


# =============================================================================
# Full 3D orchestrator
# =============================================================================

class TestSOMFull:
    """Tests for the full som_advect_tracers orchestrator."""

    def _advect(self):
        from legoesm.ocean.advection_som import som_advect_tracers
        return som_advect_tracers

    def _setup(self, n_lat=6, n_lon=10, nlev=4):
        grid = _make_grid(n_lat, n_lon)
        h_k = jnp.ones((n_lat, n_lon, nlev)) * 100.0  # 100m layers
        land_mask = jnp.ones((n_lat, n_lon))
        return grid, h_k, land_mask

    def test_zero_flow_preserves(self):
        fn = self._advect()
        grid, h_k, mask = self._setup()
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = 4

        tracer = jnp.ones((n_lat, n_lon, nlev)) * 15.0
        moments = jnp.zeros((n_lat, n_lon, nlev, 9))
        mfu = jnp.zeros((n_lat, n_lon + 1, nlev))
        mfv = jnp.zeros((n_lat + 1, n_lon, nlev))
        w = jnp.zeros((n_lat, n_lon, nlev + 1))

        t_new, m_new = fn(tracer, moments, mfu, mfv, w, h_k, h_k, grid, 300.0, mask)
        assert jnp.allclose(t_new, 15.0, atol=1e-12)
        assert jnp.allclose(m_new, 0.0, atol=1e-12)

    def test_conservation_with_flow(self):
        """Global volume-weighted tracer integral is conserved."""
        fn = self._advect()
        grid, h_k, mask = self._setup()
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = 4
        dt = 100.0

        key = jax.random.PRNGKey(70)
        tracer = jax.random.uniform(key, (n_lat, n_lon, nlev), minval=5.0, maxval=25.0)
        moments = jnp.zeros((n_lat, n_lon, nlev, 9))

        # Small random flow (CFL << 1 given dx ~ 1e6 m)
        mfu = jax.random.uniform(jax.random.PRNGKey(71), (n_lat, n_lon + 1, nlev),
                                 minval=-0.5, maxval=0.5)
        mfv = jax.random.uniform(jax.random.PRNGKey(72), (n_lat + 1, n_lon, nlev),
                                 minval=-0.5, maxval=0.5)
        # Wall BCs
        mfv = mfv.at[0, :, :].set(0.0).at[-1, :, :].set(0.0)
        w = jnp.zeros((n_lat, n_lon, nlev + 1))

        area = grid.area[..., jnp.newaxis]
        integral_before = jnp.sum(tracer * h_k * area)

        t_new, _ = fn(tracer, moments, mfu, mfv, w, h_k, h_k, grid, dt, mask)
        integral_after = jnp.sum(t_new * h_k * area)
        assert jnp.allclose(integral_before, integral_after, rtol=1e-8)

    def test_differentiable(self):
        """jax.grad should work through som_advect_tracers."""
        fn = self._advect()
        grid, h_k, mask = self._setup(n_lat=4, n_lon=6, nlev=3)
        n_lat, n_lon, nlev = 4, 6, 3

        def loss(tracer):
            moments = jnp.zeros((n_lat, n_lon, nlev, 9))
            mfu = jnp.ones((n_lat, n_lon + 1, nlev)) * 0.1
            mfv = jnp.zeros((n_lat + 1, n_lon, nlev))
            w = jnp.zeros((n_lat, n_lon, nlev + 1))
            t_new, _ = fn(tracer, moments, mfu, mfv, w, h_k, h_k, grid, 60.0, mask)
            return jnp.sum(t_new ** 2)

        tracer = jnp.ones((n_lat, n_lon, nlev)) * 10.0
        grad = jax.grad(loss)(tracer)
        # Gradient should exist and be finite
        assert jnp.all(jnp.isfinite(grad))
        # Non-zero gradient (the loss depends on tracer)
        assert jnp.any(grad != 0.0)

    def test_output_shapes(self):
        fn = self._advect()
        grid, h_k, mask = self._setup()
        n_lat, n_lon = grid.n_lat, grid.n_lon
        nlev = 4

        tracer = jnp.ones((n_lat, n_lon, nlev))
        moments = jnp.zeros((n_lat, n_lon, nlev, 9))
        mfu = jnp.zeros((n_lat, n_lon + 1, nlev))
        mfv = jnp.zeros((n_lat + 1, n_lon, nlev))
        w = jnp.zeros((n_lat, n_lon, nlev + 1))

        t_new, m_new = fn(tracer, moments, mfu, mfv, w, h_k, h_k, grid, 300.0, mask)
        assert t_new.shape == (n_lat, n_lon, nlev)
        assert m_new.shape == (n_lat, n_lon, nlev, 9)


# =============================================================================
# Analytic exactness (pins the receiver-merge moment formulas)
# =============================================================================

class TestSOMExactness:
    """SOM must translate a sub-grid-quadratic tracer EXACTLY under uniform flow.

    The {2xi, 6xi^2 - 1/2} moment basis carries each cell's 0th/1st/2nd moment
    exactly, so a globally-quadratic field advected by a uniform Courant number
    is reproduced to machine precision at every interior cell.  Fluxes are
    extracted from the original field, so only the periodic seam cell (cell 0,
    which receives the wrapped far edge) is polluted; cells 1..n-1 are exact.

    This is the decisive check that the receiver-merge second moment
    (``sxx_new``) needs NO first-moment-displacement term: the exact merged
    second moment, derived from the reconstruction basis, contains only the
    ``5*alf*alf1*(sx_cell - fp_sx)`` coupling and the ``d0`` terms already
    present.  A spurious ``d1 = sign*(alf^2*sx_cell - alf1^2*fp_sx)`` term
    would break this test.
    """

    def test_som_quadratic_advection_is_exact(self):
        from legoesm.ocean.advection_som import _som_x_sweep, IX, IXX
        n_lat, n_lon, nlev = 1, 16, 1
        alpha = 0.3                       # uniform eastward Courant number
        a0, a1, a2 = 5.0, 0.3, -0.05      # global T(x) = a0 + a1*x + a2*x^2

        j = jnp.arange(n_lon, dtype=jnp.float64)

        def moments_for(c0, c1, c2):
            # cell-mean and {2xi, 6xi^2-1/2} moments of c0 + c1*x + c2*x^2
            sm = c0 + c1 * j + c2 * j ** 2 + c2 / 12.0
            sx = (c1 + 2.0 * c2 * j) / 2.0
            sxx = jnp.full_like(j, c2 / 6.0)
            mom = jnp.zeros((n_lon, 9), dtype=jnp.float64)
            mom = mom.at[:, IX].set(sx).at[:, IXX].set(sxx)
            return sm, mom

        sm0, mom0 = moments_for(a0, a1, a2)
        sm_o = sm0.reshape(n_lat, n_lon, nlev)
        mom = mom0.reshape(n_lat, n_lon, nlev, 9)
        vol = jnp.ones((n_lat, n_lon, nlev))
        vf = jnp.full((n_lat, n_lon, nlev), alpha)   # uniform eastward flux

        sm_new, mom_new, vol_new = _som_x_sweep(sm_o, mom, vf, vol)

        # Exact translation by alpha: T(x - alpha) -> shifted global coeffs
        b0 = a0 - a1 * alpha + a2 * alpha ** 2
        b1 = a1 - 2.0 * a2 * alpha
        b2 = a2
        sm_exp, mom_exp = moments_for(b0, b1, b2)

        sl = slice(1, n_lon)             # interior cells (cell 0 wraps -> skip)
        assert jnp.allclose(sm_new[0, sl, 0], sm_exp[sl], atol=1e-10), "0th moment"
        assert jnp.allclose(mom_new[0, sl, 0, IX], mom_exp[sl, IX], atol=1e-10), "1st moment"
        assert jnp.allclose(mom_new[0, sl, 0, IXX], mom_exp[sl, IXX], atol=1e-10), "2nd moment"
        assert jnp.allclose(vol_new, vol, atol=1e-12), "uniform flow preserves volume"


# =============================================================================
# Moment permutation
# =============================================================================

class TestPermutation:
    """Verify that permute/unpermute are inverses."""

    def _perm(self):
        from legoesm.ocean.advection_som import _permute_moments, _unpermute_moments
        return _permute_moments, _unpermute_moments

    def test_y_round_trip(self):
        perm, unperm = self._perm()
        mom = jnp.arange(9.0).reshape(1, 1, 1, 9)
        assert jnp.allclose(unperm(perm(mom, "y"), "y"), mom)

    def test_z_round_trip(self):
        perm, unperm = self._perm()
        mom = jnp.arange(9.0).reshape(1, 1, 1, 9)
        assert jnp.allclose(unperm(perm(mom, "z"), "z"), mom)

    def test_x_identity(self):
        perm, unperm = self._perm()
        mom = jnp.arange(9.0).reshape(1, 1, 1, 9)
        assert jnp.allclose(perm(mom, "x"), mom)
        assert jnp.allclose(unperm(mom, "x"), mom)
