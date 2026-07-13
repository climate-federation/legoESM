"""Unit tests for the spectral shallow water solver."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.gaussian import (
    create_gaussian_grid,
    sh_analysis,
    sh_synthesis,
    sh_analysis_oc2,
    sh_analysis_dmu,
    uv_from_vordiv,
    spectral_laplacian,
    spectral_hyperdiffusion,
)
from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
    SpectralSWConfig,
    SpectralShallowWaterModel,
    spectral_sw_tendencies,
    williamson_test2_spectral,
    williamson_test5_spectral,
    spectral_to_grid,
    compute_spectral_diagnostics,
)
from legoesm import constants


# Enable float64
jax.config.update("jax_enable_x64", True)


def _proper_hyperdiff(grid):
    """Compute resolution-appropriate hyperdiffusion (4-hour damping at truncation)."""
    a = grid.radius
    eig_max = grid.n_max * (grid.n_max + 1) / (a * a)
    return 1.0 / (4.0 * 3600.0 * eig_max**2)


@pytest.fixture(scope="module")
def grid_t21():
    """T21 Gaussian grid (small, for fast tests)."""
    return create_gaussian_grid(n_max=21)


@pytest.fixture(scope="module")
def grid_t42():
    """T42 Gaussian grid (standard resolution)."""
    return create_gaussian_grid(n_max=42)


# =============================================================================
# Grid construction tests
# =============================================================================

class TestGaussianGrid:
    """Tests for Gaussian grid construction."""

    def test_no_runtime_warnings(self):
        """Grid creation should not emit any RuntimeWarning (e.g., divide by zero)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            create_gaussian_grid(n_max=21)

    def test_grid_dimensions(self, grid_t21):
        """Check grid dimensions are correct."""
        g = grid_t21
        assert g.n_max == 21
        assert g.n_lat >= 33  # >= 3*(21+1)/2
        assert g.n_lon == 2 * g.n_lat
        assert g.n_sh == (21 + 1) * (21 + 2) // 2  # 253

    def test_gaussian_latitudes_symmetric(self, grid_t21):
        """Gaussian latitudes should be symmetric about equator."""
        lat = np.array(grid_t21.lat)
        n = len(lat)
        for i in range(n // 2):
            np.testing.assert_allclose(lat[i], -lat[n - 1 - i], atol=1e-14)

    def test_gaussian_weights_sum(self, grid_t21):
        """Gaussian weights should sum to 2 (integral of 1 over [-1,1])."""
        w_sum = float(jnp.sum(grid_t21.weights))
        np.testing.assert_allclose(w_sum, 2.0, atol=1e-14)

    def test_legendre_shape(self, grid_t21):
        """Check Legendre matrix shapes."""
        g = grid_t21
        assert g.Pnm.shape == (g.n_lat, g.n_sh)
        assert g.Hnm.shape == (g.n_lat, g.n_sh)

    def test_spectral_index_arrays(self, grid_t21):
        """Check ls and ms arrays have correct ranges."""
        g = grid_t21
        assert g.ls.shape == (g.n_sh,)
        assert g.ms.shape == (g.n_sh,)
        assert int(jnp.max(g.ls)) == g.n_max
        assert int(jnp.max(g.ms)) == g.n_max
        # m <= n for all entries
        assert bool(jnp.all(g.ms <= g.ls))


# =============================================================================
# SH transform tests
# =============================================================================

class TestSHTransforms:
    """Tests for spherical harmonic transforms."""

    def test_roundtrip_constant(self, grid_t21):
        """Constant field survives roundtrip."""
        f = jnp.ones((grid_t21.n_lat, grid_t21.n_lon), dtype=jnp.float64) * 7.0
        coeffs = sh_analysis(grid_t21, f)
        f_rec = sh_synthesis(grid_t21, coeffs)
        np.testing.assert_allclose(f_rec, 7.0, atol=1e-10)

    def test_roundtrip_y10(self, grid_t21):
        """Y_1^0 ~ sin(lat) ~ cos(colatitude) roundtrip."""
        # Y_1^0 is proportional to sin(lat)
        f = jnp.sin(grid_t21.lat2d)  # = sin(lat), shape (n_lat, n_lon)
        coeffs = sh_analysis(grid_t21, f)
        f_rec = sh_synthesis(grid_t21, coeffs)
        np.testing.assert_allclose(f_rec, f, atol=1e-10)

    def test_roundtrip_y11(self, grid_t21):
        """Y_1^1 ~ cos(lat)*cos(lon) roundtrip."""
        f = jnp.cos(grid_t21.lat2d) * jnp.cos(grid_t21.lon2d)
        coeffs = sh_analysis(grid_t21, f)
        f_rec = sh_synthesis(grid_t21, coeffs)
        np.testing.assert_allclose(f_rec, f, atol=1e-10)

    def test_roundtrip_y22(self, grid_t21):
        """Higher order harmonic Y_2^2 roundtrip."""
        lat = grid_t21.lat2d
        lon = grid_t21.lon2d
        # Y_2^2 is proportional to cos(lat)^2 * cos(2*lon)
        f = jnp.cos(lat)**2 * jnp.cos(2 * lon)
        coeffs = sh_analysis(grid_t21, f)
        f_rec = sh_synthesis(grid_t21, coeffs)
        np.testing.assert_allclose(f_rec, f, atol=1e-10)

    def test_roundtrip_complex_pattern(self, grid_t21):
        """A combination of several harmonics survives roundtrip."""
        lat = grid_t21.lat2d
        lon = grid_t21.lon2d
        f = (3.0 * jnp.sin(lat) +
             2.0 * jnp.cos(lat) * jnp.sin(lon) +
             1.5 * jnp.cos(lat)**2 * jnp.cos(2 * lon))
        coeffs = sh_analysis(grid_t21, f)
        f_rec = sh_synthesis(grid_t21, coeffs)
        np.testing.assert_allclose(f_rec, f, atol=1e-10)

    def test_parseval(self, grid_t21):
        """Parseval's theorem: L2 norms should match between grid and spectral."""
        g = grid_t21
        f = jnp.sin(g.lat2d) + 0.5 * jnp.cos(g.lat2d) * jnp.cos(g.lon2d)

        # Grid-space L2 norm (with Gaussian quadrature)
        dlon = 2.0 * jnp.pi / g.n_lon
        grid_norm_sq = jnp.sum(f**2 * g.weights[:, None] * dlon)

        # Spectral L2 norm: sum |c_nm|^2 for all (n,m).
        # Only m>=0 stored; for m>0, count twice (conjugate pair).
        coeffs = sh_analysis(g, f)
        ms = g.ms
        weight = jnp.where(ms == 0, 1.0, 2.0)
        spec_norm_sq = jnp.sum(weight * jnp.abs(coeffs)**2)

        np.testing.assert_allclose(float(grid_norm_sq), float(spec_norm_sq),
                                   rtol=1e-10)


# =============================================================================
# Spectral operator tests
# =============================================================================

class TestSpectralOperators:
    """Tests for spectral Laplacian and related operators."""

    def test_laplacian_eigenvalue(self, grid_t21):
        """Laplacian of Y_n^m should give -n(n+1)/a^2 * Y_n^m."""
        g = grid_t21
        a = g.radius

        # Test with Y_2^0 ~ (3*sin^2(lat) - 1)/2
        f = 0.5 * (3.0 * jnp.sin(g.lat2d)**2 - 1.0)
        coeffs = sh_analysis(g, f)

        lap_coeffs = spectral_laplacian(g, coeffs)
        expected = -2.0 * 3.0 / (a * a) * coeffs  # n=2: -n(n+1) = -6

        # Most coefficients should be near zero except the n=2 ones
        # Compare the full arrays
        f_lap = sh_synthesis(g, lap_coeffs)
        f_expected = sh_synthesis(g, expected)
        np.testing.assert_allclose(f_lap, f_expected, atol=1e-10)

    def test_spectral_divergence(self, grid_t21):
        """Verify spectral div/curl operators with known vector fields."""
        g = grid_t21
        a = g.radius
        im_over_a = 1j * g.ms.astype(jnp.float64) / a
        one_over_a = 1.0 / a

        # div(u=cosφ*cosλ, v=0) = -sinλ/a
        U = jnp.cos(g.lat2d) * jnp.cos(g.lon2d) * g.cos_lat[:, None]
        V = jnp.zeros_like(U)

        div_hat = im_over_a * sh_analysis_oc2(g, U) - one_over_a * sh_analysis_dmu(g, V)
        div_grid = sh_synthesis(g, div_hat)
        expected_div = -jnp.sin(g.lon2d) / a
        np.testing.assert_allclose(div_grid, expected_div, atol=1e-6)

    def test_hyperdiffusion_is_dissipative_for_order1(self, grid_t21):
        """order=1 should still be dissipative (negative damping)."""
        g = grid_t21
        coeffs = jnp.ones((g.n_sh,), dtype=jnp.complex128)
        diff = spectral_hyperdiffusion(g, coeffs, nu=1.0, order=1)
        # n=1, m=0 mode (index 1) should be damped.
        assert float(jnp.real(diff[1])) < 0.0

    def test_hyperdiffusion_invalid_order_raises(self, grid_t21):
        """Non-positive hyperdiffusion order should raise."""
        coeffs = jnp.ones((grid_t21.n_sh,), dtype=jnp.complex128)
        with pytest.raises(ValueError, match="order must be >= 1"):
            spectral_hyperdiffusion(grid_t21, coeffs, nu=1.0, order=0)

    def test_hyperdiffusion_negative_nu_raises(self, grid_t21):
        """Negative hyperdiffusion coefficient should be rejected."""
        coeffs = jnp.ones((grid_t21.n_sh,), dtype=jnp.complex128)
        with pytest.raises(ValueError, match="nu must be >= 0"):
            spectral_hyperdiffusion(grid_t21, coeffs, nu=-1.0, order=2)

    def test_hyperdiffusion_zero_nu_returns_zero(self, grid_t21):
        """Zero hyperdiffusion coefficient should return identically zero tendency."""
        coeffs = jnp.ones((grid_t21.n_sh,), dtype=jnp.complex128)
        diff = spectral_hyperdiffusion(grid_t21, coeffs, nu=0.0, order=2)
        assert jnp.all(diff == 0.0)


# =============================================================================
# Velocity reconstruction tests
# =============================================================================

class TestVelocityReconstruction:
    """Tests for velocity reconstruction from vorticity and divergence."""

    def test_solid_body_rotation(self, grid_t21):
        """Solid body rotation: vor = 2*u0/R * sin(lat), div=0 -> u=u0*cos(lat), v=0."""
        g = grid_t21
        u_0 = 20.0
        R = g.radius

        # Vorticity for solid body rotation
        vor = 2.0 * u_0 / R * jnp.sin(g.lat2d)
        div_field = jnp.zeros_like(vor)

        vor_hat = sh_analysis(g, vor)
        div_hat = sh_analysis(g, div_field)

        u_cos, v_cos = uv_from_vordiv(g, vor_hat, div_hat)
        u = u_cos / g.cos_lat[:, None]
        v = v_cos / g.cos_lat[:, None]

        u_expected = u_0 * jnp.cos(g.lat2d)
        v_expected = jnp.zeros_like(u)

        # Exclude near-poles (cos_lat ~ 0 causes division issues)
        mask = g.cos_lat[:, None] > 0.1
        np.testing.assert_allclose(
            u * mask, u_expected * mask, atol=0.5,
            err_msg="u reconstruction failed for solid body rotation"
        )
        np.testing.assert_allclose(
            v * mask, v_expected * mask, atol=0.5,
            err_msg="v should be zero for solid body rotation"
        )


# =============================================================================
# Spectral shallow water model tests
# =============================================================================

class TestSpectralSW:
    """Tests for the spectral shallow water model."""

    def test_test2_initialization(self, grid_t21):
        """Test Case 2 initialization produces finite spectral coefficients."""
        state = williamson_test2_spectral(grid_t21)
        assert jnp.all(jnp.isfinite(state.vor_hat.data))
        assert jnp.all(jnp.isfinite(state.div_hat.data))
        assert jnp.all(jnp.isfinite(state.phi_hat.data))
        assert state.vor_hat.data.shape == (grid_t21.n_sh,)

    def test_test5_initialization(self, grid_t21):
        """Test Case 5 initialization produces finite spectral coefficients."""
        state = williamson_test5_spectral(grid_t21)
        assert jnp.all(jnp.isfinite(state.vor_hat.data))
        assert jnp.all(jnp.isfinite(state.phi_hat.data))
        # Surface geopotential should be nonzero (mountain)
        assert float(jnp.max(jnp.abs(state.phis_hat.data))) > 0

    def test_test2_small_tendencies(self, grid_t21):
        """Test Case 2 (geostrophic balance) should have small tendencies."""
        state = williamson_test2_spectral(grid_t21)
        # Background depth (gh0 = 2.94e4 m^2/s^2) is carried by the TC2 initial
        # phi_hat from williamson_test2_spectral, not by any config knob.
        config = SpectralSWConfig(hyperdiff_coeff=0.0)
        tend = spectral_sw_tendencies(state, grid_t21, config)

        # Relative to the fields themselves
        vor_scale = float(jnp.max(jnp.abs(state.vor_hat.data)))
        phi_scale = float(jnp.max(jnp.abs(state.phi_hat.data)))

        vor_tend_max = float(jnp.max(jnp.abs(tend.vor_hat.data)))
        phi_tend_max = float(jnp.max(jnp.abs(tend.phi_hat.data)))

        # Tendencies should be small relative to field values
        assert vor_tend_max / vor_scale < 1e-4, \
            f"Vorticity tendency too large: {vor_tend_max/vor_scale:.2e}"
        assert phi_tend_max / phi_scale < 1e-4, \
            f"Geopotential tendency too large: {phi_tend_max/phi_scale:.2e}"

    def test_single_step_finite(self, grid_t21):
        """One time step should produce all-finite state."""
        state = williamson_test5_spectral(grid_t21)
        nu = _proper_hyperdiff(grid_t21)
        config = SpectralSWConfig(hyperdiff_coeff=nu)
        model = SpectralShallowWaterModel(grid_t21, config)
        state_new = model.step(state, 200.0)
        assert jnp.all(jnp.isfinite(state_new.vor_hat.data))
        assert jnp.all(jnp.isfinite(state_new.div_hat.data))
        assert jnp.all(jnp.isfinite(state_new.phi_hat.data))

    def test_multi_step_stability(self, grid_t21):
        """500 steps without blowup (dt=200s, within CFL for T21)."""
        state = williamson_test5_spectral(grid_t21)
        nu = _proper_hyperdiff(grid_t21)
        config = SpectralSWConfig(hyperdiff_coeff=nu)
        model = SpectralShallowWaterModel(grid_t21, config)
        for _ in range(500):
            state = model.step(state, 200.0)
        assert jnp.all(jnp.isfinite(state.vor_hat.data))
        assert jnp.all(jnp.isfinite(state.phi_hat.data))

    def test_spectral_to_grid(self, grid_t21):
        """spectral_to_grid returns all expected keys with correct shapes."""
        state = williamson_test5_spectral(grid_t21)
        fields = spectral_to_grid(state, grid_t21)
        expected_keys = {'h', 'u', 'v', 'vor', 'div', 'phi', 'phis', 'h_s'}
        assert set(fields.keys()) == expected_keys
        for k, val in fields.items():
            assert val.shape == (grid_t21.n_lat, grid_t21.n_lon), \
                f"Wrong shape for {k}: {val.shape}"

    def test_diagnostics(self, grid_t21):
        """Conservation diagnostics return finite values."""
        state = williamson_test5_spectral(grid_t21)
        diag = compute_spectral_diagnostics(state, grid_t21)
        assert 'mass' in diag
        assert 'energy' in diag
        assert 'enstrophy' in diag
        assert np.isfinite(diag['mass'])
        assert np.isfinite(diag['energy'])

    def test_williamson5_energy_includes_topography_PE(self, grid_t21):
        """The total mechanical energy on Williamson Test 5 must include
        the topography PE term ``g·h·h_s``.

        Why non-vacuous: under the prior bug, the energy diagnostic was
        ``0.5·h·|v|² + 0.5·g·h²`` only, ignoring the rest energy of the
        fluid column above non-zero bottom topography.  Williamson 5 has
        a tall isolated mountain (h_s peak ~2000 m) — including it
        increases the diagnostic energy by ``g·∫h·h_s dA`` ≈ a measurable
        ~0.3% of the total energy (mountain footprint × mean h ≈ 5 km ×
        2000 m = 1e7 m² × 9.8 × 5e3 ≈ 5e11 J/m, vs total ~1.5e14 J/m).

        This test asserts the energy diagnostic on a Williamson-5 state
        is HIGHER than what the buggy formula would give, by at least
        the topography PE ``g · ∫h·h_s dA``.
        """
        state = williamson_test5_spectral(grid_t21)
        diag = compute_spectral_diagnostics(state, grid_t21)
        energy_with_topo = diag['energy']

        # Recompute the buggy value (without h·h_s) for comparison.
        from legoesm.atmosphere.dynamics.gcm.spectral_sw import spectral_to_grid
        fields = spectral_to_grid(state, grid_t21)
        h, u, v, h_s = fields['h'], fields['u'], fields['v'], fields['h_s']
        w = grid_t21.weights[:, None]
        dlon = 2.0 * jnp.pi / grid_t21.n_lon
        a2 = grid_t21.radius * grid_t21.radius
        dA = w * dlon * a2
        g = constants.g

        energy_buggy = float(jnp.sum(
            (0.5 * h * (u**2 + v**2) + 0.5 * g * h**2) * dA
        ))
        energy_topo_term = float(jnp.sum(g * h * h_s * dA))

        # The diagnostic must equal the buggy value PLUS the topo term
        # (within float roundoff).
        assert abs(energy_with_topo - (energy_buggy + energy_topo_term)) < 1e-3 * abs(energy_topo_term), (
            f"Energy diagnostic mismatch: with-topo={energy_with_topo:.6e}, "
            f"buggy+topo_term={energy_buggy + energy_topo_term:.6e}, "
            f"buggy={energy_buggy:.6e}, topo_term={energy_topo_term:.6e}"
        )
        # Topography PE must be non-trivial on TC5 (mountain present).
        assert energy_topo_term > 1e10, (
            f"Topography PE on Williamson 5 was {energy_topo_term:.3e}; "
            f"expected > 1e10 J/m for a 2000 m isolated mountain."
        )

    def test_differentiability(self, grid_t21):
        """jax.grad through a single step should produce finite gradients."""
        state = williamson_test2_spectral(grid_t21)
        nu = _proper_hyperdiff(grid_t21)
        # Background depth is carried by the TC2 initial phi_hat (gh0 = 2.94e4),
        # not by any config knob.
        config = SpectralSWConfig(hyperdiff_coeff=nu)

        def loss_fn(vor_hat_data):
            s = state._replace(
                vor_hat=state.vor_hat.replace(data=vor_hat_data)
            )
            tend = spectral_sw_tendencies(s, grid_t21, config)
            return jnp.sum(jnp.abs(tend.vor_hat.data)**2).real

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(state.vor_hat.data)
        assert jnp.all(jnp.isfinite(g))
