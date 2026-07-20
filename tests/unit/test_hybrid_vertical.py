"""Unit tests for the hybrid sigma-pressure vertical coordinate.

Tests cover:
- Factory functions and coordinate construction
- Pressure computation
- Geopotential via hybrid Simmons-Burridge
- Mass flux (eta-dot analog)
- Vertical advection
- Pressure velocity (omega)
- Semi-implicit Gamma matrix
- Equivalence with pure sigma when A=0, B=sigma
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    create_sigma_coordinate,
    create_hybrid_coordinate,
    make_hybrid_levels,
    standard_hybrid_levels,
    hybrid_from_sigma,
    pressure_from_sigma,
    pressure_from_hybrid,
    dp_from_hybrid,
    compute_geopotential,
    compute_geopotential_hybrid,
    compute_sigma_dot,
    compute_mass_flux_hybrid,
    compute_mass_flux_from_cumsum,
    vertical_advection,
    vertical_advection_hybrid,
    compute_pressure_velocity,
    compute_omega_hybrid,
)

NLEV = 20
P_REF = 1e5


class TestHybridCoordinateConstruction:
    """Test factory functions and coordinate properties."""

    def test_make_hybrid_levels_shape(self):
        """make_hybrid_levels returns correct shapes."""
        coord = make_hybrid_levels(NLEV, p_top_Pa=200.0)
        assert coord.n_levels == NLEV
        assert coord.A_half.shape == (NLEV + 1,)
        assert coord.B_half.shape == (NLEV + 1,)
        assert coord.A_full.shape == (NLEV,)
        assert coord.B_full.shape == (NLEV,)
        assert coord.dA.shape == (NLEV,)
        assert coord.dB.shape == (NLEV,)
        assert coord.ln_ratio_ref.shape == (NLEV,)
        assert coord.alpha_ref.shape == (NLEV,)
        assert coord.dsigma_eff.shape == (NLEV,)

    def test_boundary_conditions(self):
        """A(0)=p_top/p_ref, B(0)=0, A(-1)=0, B(-1)=1."""
        p_top = 200.0
        coord = make_hybrid_levels(NLEV, p_top_Pa=p_top)
        assert float(coord.B_half[0]) == pytest.approx(0.0, abs=1e-7)
        assert float(coord.B_half[-1]) == pytest.approx(1.0, abs=1e-7)
        assert float(coord.A_half[-1]) == pytest.approx(0.0, abs=1e-7)
        assert float(coord.A_half[0]) == pytest.approx(p_top / P_REF, rel=1e-5)

    def test_reference_pressure_uniform(self):
        """At p_s=p_ref, interface pressures are uniformly spaced."""
        p_top = 200.0
        coord = make_hybrid_levels(NLEV, p_top_Pa=p_top)
        p_half_ref = (coord.A_half + coord.B_half) * P_REF
        dp_ref = jnp.diff(p_half_ref)
        # Should be uniform
        np.testing.assert_allclose(dp_ref, dp_ref[0], rtol=1e-5)

    def test_B_range(self):
        """B_range = B_half[-1] - B_half[0] = 1.0."""
        coord = make_hybrid_levels(NLEV)
        assert coord.B_range == pytest.approx(1.0, abs=1e-7)

    def test_dsigma_eff_sums_to_one(self):
        """dsigma_eff = dA + dB should sum to A[-1]+B[-1] - A[0]-B[0]."""
        coord = make_hybrid_levels(NLEV, p_top_Pa=200.0)
        total = float(jnp.sum(coord.dsigma_eff))
        expected = float(
            (coord.A_half[-1] + coord.B_half[-1])
            - (coord.A_half[0] + coord.B_half[0])
        )
        assert total == pytest.approx(expected, rel=1e-5)

    def test_transition_exponent_1_is_sigma_like(self):
        """With exponent=1, B=eta (linear) and A contains only p_top contribution."""
        coord = make_hybrid_levels(NLEV, p_top_Pa=200.0, transition_exponent=1)
        eta = np.linspace(0, 1, NLEV + 1)
        np.testing.assert_allclose(coord.B_half, eta, atol=1e-6)

    def test_hybrid_from_sigma_roundtrip(self):
        """Converting sigma to hybrid should preserve B=sigma, A=0."""
        sigma = create_sigma_coordinate(NLEV)
        hybrid = hybrid_from_sigma(sigma)
        np.testing.assert_allclose(hybrid.A_half, 0.0, atol=1e-10)
        np.testing.assert_allclose(hybrid.B_half, sigma.sigma_half, atol=1e-6)

    def test_sigma_compatibility_views(self):
        """Hybrid coordinate exposes sigma-style compatibility properties."""
        coord = make_hybrid_levels(NLEV, p_top_Pa=200.0)
        np.testing.assert_allclose(coord.sigma_half, coord.A_half + coord.B_half, atol=1e-7)
        np.testing.assert_allclose(coord.sigma_full, coord.A_full + coord.B_full, atol=1e-7)
        np.testing.assert_allclose(coord.dsigma, coord.dA + coord.dB, atol=1e-7)
        np.testing.assert_allclose(coord.dsigma_full, np.diff(coord.sigma_full), atol=1e-7)


class TestHybridPressure:
    """Test pressure computation from hybrid coordinates."""

    def test_pressure_full_shape(self):
        """pressure_from_hybrid returns correct shape."""
        coord = make_hybrid_levels(NLEV)
        p_s = jnp.full((6, 4, 4), P_REF)
        p_full = pressure_from_hybrid(coord, p_s)
        assert p_full.shape == (6, 4, 4, NLEV)

    def test_pressure_half_shape(self):
        """Half-level pressures have nlev+1 levels."""
        coord = make_hybrid_levels(NLEV)
        p_s = jnp.full((6, 4, 4), P_REF)
        p_half = pressure_from_hybrid(coord, p_s, full=False)
        assert p_half.shape == (6, 4, 4, NLEV + 1)

    def test_pressure_surface_equals_ps(self):
        """Pressure at bottom interface should equal p_s."""
        coord = make_hybrid_levels(NLEV)
        p_s = jnp.full((6, 4, 4), 1.013e5)
        p_half = pressure_from_hybrid(coord, p_s, full=False)
        np.testing.assert_allclose(p_half[..., -1], p_s, rtol=1e-5)

    def test_pressure_top(self):
        """Pressure at top interface should be p_top."""
        p_top = 200.0
        coord = make_hybrid_levels(NLEV, p_top_Pa=p_top)
        p_s = jnp.full((6, 4, 4), P_REF)
        p_half = pressure_from_hybrid(coord, p_s, full=False)
        np.testing.assert_allclose(p_half[..., 0], p_top, rtol=1e-3)

    def test_dp_positive(self):
        """Layer pressure thickness should be positive everywhere."""
        coord = make_hybrid_levels(NLEV)
        p_s = jnp.full((6, 4, 4), P_REF)
        dp = dp_from_hybrid(coord, p_s)
        assert jnp.all(dp > 0)

    def test_dp_sums_to_ps_minus_ptop(self):
        """Sum of dp should equal p_s - p_top."""
        p_top = 200.0
        coord = make_hybrid_levels(NLEV, p_top_Pa=p_top)
        p_s = jnp.full((6, 4, 4), P_REF)
        dp = dp_from_hybrid(coord, p_s)
        dp_total = jnp.sum(dp, axis=-1)
        np.testing.assert_allclose(dp_total, P_REF - p_top, rtol=1e-4)

    def test_sigma_hybrid_pressure_equivalence(self):
        """Pure-sigma hybrid should give same pressures as SigmaCoordinate."""
        sigma = create_sigma_coordinate(NLEV)
        hybrid = hybrid_from_sigma(sigma)
        p_s = jnp.full((6, 4, 4), P_REF)

        p_full_sigma = pressure_from_sigma(sigma.sigma_full, p_s)
        p_full_hybrid = pressure_from_hybrid(hybrid, p_s)

        np.testing.assert_allclose(p_full_hybrid, p_full_sigma, rtol=1e-4)


class TestHybridGeopotential:
    """Test Simmons-Burridge geopotential with hybrid coordinates."""

    def test_geopotential_shape(self):
        """Geopotential should have correct shape."""
        coord = make_hybrid_levels(NLEV)
        T = jnp.full((6, 4, 4, NLEV), 280.0)
        p_s = jnp.full((6, 4, 4), P_REF)
        phis = jnp.zeros((6, 4, 4))
        Phi = compute_geopotential_hybrid(T, p_s, coord, phis)
        assert Phi.shape == (6, 4, 4, NLEV)

    def test_geopotential_increases_with_height(self):
        """Geopotential should increase with height (decrease with k)."""
        coord = make_hybrid_levels(NLEV)
        T = jnp.full((6, 4, 4, NLEV), 280.0)
        p_s = jnp.full((6, 4, 4), P_REF)
        phis = jnp.zeros((6, 4, 4))
        Phi = compute_geopotential_hybrid(T, p_s, coord, phis)
        # Phi[..., 0] (top) > Phi[..., -1] (bottom)
        assert float(Phi[0, 0, 0, 0]) > float(Phi[0, 0, 0, -1])

    def test_geopotential_bottom_near_surface(self):
        """Bottom-level geopotential should be close to surface geopotential."""
        coord = make_hybrid_levels(NLEV)
        T = jnp.full((6, 4, 4, NLEV), 280.0)
        p_s = jnp.full((6, 4, 4), P_REF)
        phis = jnp.full((6, 4, 4), 1000.0)  # some surface geopotential
        Phi = compute_geopotential_hybrid(T, p_s, coord, phis)
        # Bottom level should be close to phis (within one layer's worth)
        assert float(jnp.abs(Phi[0, 0, 0, -1] - phis[0, 0, 0])) < 5000.0

    def test_sigma_hybrid_geopotential_equivalence(self):
        """Pure-sigma hybrid should give same geopotential as SigmaCoordinate."""
        sigma = create_sigma_coordinate(NLEV)
        hybrid = hybrid_from_sigma(sigma)
        T = jnp.full((6, 4, 4, NLEV), 280.0)
        p_s = jnp.full((6, 4, 4), P_REF)
        phis = jnp.zeros((6, 4, 4))

        Phi_sigma = compute_geopotential(T, p_s, sigma, phis)
        Phi_hybrid = compute_geopotential_hybrid(T, p_s, hybrid, phis)

        np.testing.assert_allclose(Phi_hybrid, Phi_sigma, rtol=1e-3)


class TestHybridMassFlux:
    """Test vertical mass flux computation."""

    def test_mass_flux_shape(self):
        """Mass flux should have nlev+1 levels."""
        coord = make_hybrid_levels(NLEV)
        div = jnp.ones((6, 4, 4, NLEV)) * 1e-5
        p_s = jnp.full((6, 4, 4), P_REF)
        F, _ = compute_mass_flux_hybrid(div, p_s, coord)
        assert F.shape == (6, 4, 4, NLEV + 1)

    def test_mass_flux_boundary_conditions(self):
        """Mass flux should be zero at top and surface."""
        coord = make_hybrid_levels(NLEV)
        div = jnp.ones((6, 4, 4, NLEV)) * 1e-5
        p_s = jnp.full((6, 4, 4), P_REF)
        F, _ = compute_mass_flux_hybrid(div, p_s, coord)
        np.testing.assert_allclose(F[..., 0], 0.0, atol=1e-20)
        np.testing.assert_allclose(F[..., -1], 0.0, atol=1e-10)

    def test_zero_divergence_gives_zero_flux(self):
        """Zero divergence should give zero mass flux."""
        coord = make_hybrid_levels(NLEV)
        div = jnp.zeros((6, 4, 4, NLEV))
        p_s = jnp.full((6, 4, 4), P_REF)
        F, _ = compute_mass_flux_hybrid(div, p_s, coord)
        np.testing.assert_allclose(F, 0.0, atol=1e-20)

    def test_sigma_hybrid_mass_flux_equivalence(self):
        """Pure-sigma hybrid mass flux should match p_s * sigma_dot."""
        sigma = create_sigma_coordinate(NLEV)
        hybrid = hybrid_from_sigma(sigma)

        key = jax.random.PRNGKey(42)
        div = jax.random.normal(key, (6, 4, 4, NLEV)) * 1e-5
        p_s = jnp.full((6, 4, 4), P_REF)

        sigma_dot = compute_sigma_dot(div, sigma)
        mass_flux, _ = compute_mass_flux_hybrid(div, p_s, hybrid)

        # mass_flux = p_s * sigma_dot for pure sigma
        expected = p_s[..., None] * sigma_dot
        np.testing.assert_allclose(mass_flux, expected, rtol=1e-3, atol=1e-5)


class TestMassFluxFromCumsum:
    """Direct tests for the factored-out hybrid mass-flux boundary closure."""

    def test_matches_compute_mass_flux_hybrid_bit_for_bit(self):
        """compute_mass_flux_hybrid == manual cumsum + the shared closure (the dedup)."""
        coord = make_hybrid_levels(NLEV)
        key = jax.random.PRNGKey(7)
        div = jax.random.normal(key, (6, 4, 4, NLEV)) * 1e-5
        p_s = jnp.full((6, 4, 4), P_REF)
        F_ref, _ = compute_mass_flux_hybrid(div, p_s, coord)
        # Rebuild via the public helper from the SAME advective div_dp.
        dp = dp_from_hybrid(coord, p_s)
        cumsum_div = jnp.cumsum(div * dp, axis=-1)
        F_helper = compute_mass_flux_from_cumsum(
            cumsum_div, cumsum_div[..., -1:], coord)
        # The refactor must be byte-identical, not merely close.
        np.testing.assert_array_equal(np.asarray(F_helper), np.asarray(F_ref))

    def test_boundary_conditions(self):
        """F = 0 at top + surface for any cumsum; shape is nlev+1."""
        coord = make_hybrid_levels(NLEV)
        cumsum_div = jnp.cumsum(jnp.ones((6, 4, 4, NLEV)) * 1e-5, axis=-1)
        F = compute_mass_flux_from_cumsum(cumsum_div, cumsum_div[..., -1:], coord)
        assert F.shape == (6, 4, 4, NLEV + 1)
        np.testing.assert_allclose(F[..., 0], 0.0, atol=1e-20)
        np.testing.assert_allclose(F[..., -1], 0.0, atol=1e-10)

    def test_caller_chooses_div_convention(self):
        """The closure is pure over its cumsum: a flux-form cumsum that differs
        from the advective one (grad(p_s) != 0) yields a different, valid flux."""
        coord = make_hybrid_levels(NLEV)
        key = jax.random.PRNGKey(11)
        cumsum_adv = jnp.cumsum(jax.random.normal(key, (5, NLEV)) * 1e-5, axis=-1)
        # A distinct "flux-form" cumsum (the v*grad(dp) term shifts it).
        cumsum_flux = cumsum_adv + 1e-6 * jnp.arange(NLEV)
        F_adv = compute_mass_flux_from_cumsum(cumsum_adv, cumsum_adv[..., -1:], coord)
        F_flux = compute_mass_flux_from_cumsum(cumsum_flux, cumsum_flux[..., -1:], coord)
        assert not np.allclose(np.asarray(F_adv), np.asarray(F_flux))
        for F in (F_adv, F_flux):  # both still satisfy the boundaries
            np.testing.assert_allclose(np.asarray(F)[..., 0], 0.0, atol=1e-20)
            np.testing.assert_allclose(np.asarray(F)[..., -1], 0.0, atol=1e-10)

    def test_differentiable_and_vmappable(self):
        """Pure-JAX: jit+grad through the closure and vmap over leading dims."""
        coord = make_hybrid_levels(NLEV)
        cumsum_div = jnp.cumsum(jnp.ones((3, NLEV)) * 1e-5, axis=-1)

        def _scalar(c):
            F = compute_mass_flux_from_cumsum(c, c[..., -1:], coord)
            return jnp.sum(F ** 2)

        g = jax.jit(jax.grad(_scalar))(cumsum_div)
        assert g.shape == cumsum_div.shape
        assert np.all(np.isfinite(np.asarray(g)))
        F_vmap = jax.vmap(
            lambda c: compute_mass_flux_from_cumsum(c, c[..., -1:], coord)
        )(cumsum_div)
        assert F_vmap.shape == (3, NLEV + 1)


class TestHybridVerticalAdvection:
    """Test vertical advection in hybrid coordinates."""

    def test_advection_shape(self):
        """Vertical advection tendency should have correct shape."""
        coord = make_hybrid_levels(NLEV)
        field = jnp.ones((6, 4, 4, NLEV)) * 280.0
        mass_flux = jnp.zeros((6, 4, 4, NLEV + 1))
        p_s = jnp.full((6, 4, 4), P_REF)
        tend = vertical_advection_hybrid(field, mass_flux, p_s, coord)
        assert tend.shape == (6, 4, 4, NLEV)

    def test_uniform_field_zero_advection(self):
        """Uniform field should have zero vertical advection tendency."""
        coord = make_hybrid_levels(NLEV)
        field = jnp.ones((6, 4, 4, NLEV)) * 280.0
        # Non-zero mass flux
        mass_flux = jnp.zeros((6, 4, 4, NLEV + 1)).at[..., 5].set(100.0)
        p_s = jnp.full((6, 4, 4), P_REF)
        tend = vertical_advection_hybrid(field, mass_flux, p_s, coord)
        np.testing.assert_allclose(tend, 0.0, atol=1e-8)

    def test_zero_flux_zero_advection(self):
        """Zero mass flux should give zero advection regardless of field."""
        coord = make_hybrid_levels(NLEV)
        key = jax.random.PRNGKey(7)
        field = jax.random.normal(key, (6, 4, 4, NLEV))
        mass_flux = jnp.zeros((6, 4, 4, NLEV + 1))
        p_s = jnp.full((6, 4, 4), P_REF)
        tend = vertical_advection_hybrid(field, mass_flux, p_s, coord)
        np.testing.assert_allclose(tend, 0.0, atol=1e-20)


class TestHybridOmega:
    """Test pressure velocity computation."""

    def test_omega_shape(self):
        """Omega should have correct shape."""
        coord = make_hybrid_levels(NLEV)
        mass_flux = jnp.zeros((6, 4, 4, NLEV + 1))
        p_s = jnp.full((6, 4, 4), P_REF)
        dp_s_dt = jnp.zeros((6, 4, 4))
        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt, coord)
        assert omega.shape == (6, 4, 4, NLEV)

    def test_omega_zero_for_rest_state(self):
        """Zero mass flux and zero dp_s/dt gives zero omega."""
        coord = make_hybrid_levels(NLEV)
        mass_flux = jnp.zeros((6, 4, 4, NLEV + 1))
        p_s = jnp.full((6, 4, 4), P_REF)
        dp_s_dt = jnp.zeros((6, 4, 4))
        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt, coord)
        np.testing.assert_allclose(omega, 0.0, atol=1e-20)


class TestHybridSemiImplicit:
    """Test semi-implicit Gamma matrix with hybrid coordinates."""

    def test_gamma_shape(self):
        """Gamma matrix should be (nlev, nlev)."""
        from legoesm.timestepping.semi_implicit import compute_Gamma_matrix
        coord = make_hybrid_levels(NLEV)
        Gamma = compute_Gamma_matrix(coord, T_ref=300.0)
        assert Gamma.shape == (NLEV, NLEV)

    def test_gamma_positive_diagonal(self):
        """Diagonal of Gamma should be positive (self-coupling)."""
        from legoesm.timestepping.semi_implicit import compute_Gamma_matrix
        coord = make_hybrid_levels(NLEV)
        Gamma = compute_Gamma_matrix(coord, T_ref=300.0)
        diag = jnp.diag(Gamma)
        assert jnp.all(diag > 0)

    def test_sigma_hybrid_gamma_equivalence(self):
        """Pure-sigma hybrid should give same Gamma as SigmaCoordinate."""
        from legoesm.timestepping.semi_implicit import compute_Gamma_matrix
        sigma = create_sigma_coordinate(NLEV)
        hybrid = hybrid_from_sigma(sigma)

        Gamma_sigma = compute_Gamma_matrix(sigma, T_ref=300.0)
        Gamma_hybrid = compute_Gamma_matrix(hybrid, T_ref=300.0)

        np.testing.assert_allclose(Gamma_hybrid, Gamma_sigma, rtol=1e-4)

    def test_precompute_si_matrices(self):
        """precompute_si_matrices should work with hybrid coordinates."""
        from legoesm.timestepping.semi_implicit import precompute_si_matrices
        from legoesm.grids.gaussian import create_gaussian_grid

        grid = create_gaussian_grid(21)
        coord = make_hybrid_levels(NLEV)
        si_data = precompute_si_matrices(grid, coord, T_ref=300.0, dt=600.0)

        assert si_data.Gamma.shape == (NLEV, NLEV)
        assert si_data.si_matrices.shape[1:] == (NLEV, NLEV)


class TestSemiImplicitReferenceMatrixWellPosed:
    """Issue #960: the semi-implicit ``Gamma`` reference matrix must be a
    well-posed vertical structure operator — diagonalizable with REAL,
    POSITIVE eigenvalues (the gravity-wave "equivalent depths" H = lambda/g).

    Complex eigenvalues mean ``Gamma`` is non-normal, which amplifies the
    resolved modes independently of the RHS/increment formulation and is the
    2nd contributor to the T85 NaN.  The fix is a CONSISTENT reference
    thermodynamic coupling ``tau`` (the linearization of the model's own
    adiabatic term, ``dT'/dt = -tau @ D``) instead of the diagonal
    ``tau = T_ref*I`` approximation.
    """

    @staticmethod
    def _eig(Gamma):
        w = np.linalg.eigvals(np.asarray(Gamma, dtype=np.float64))
        rel_imag = np.max(np.abs(w.imag)) / (np.max(np.abs(w.real)) + 1e-300)
        return w, rel_imag

    @pytest.mark.parametrize("nlev", [4, 8, 20, 32, 40])
    def test_sigma_gamma_eigenvalues_real_positive(self, nlev):
        from legoesm.timestepping.semi_implicit import compute_Gamma_matrix
        coord = create_sigma_coordinate(nlev, sigma_top=0.01)
        w, rel_imag = self._eig(compute_Gamma_matrix(coord, T_ref=300.0))
        assert rel_imag < 1e-8, f"sigma L{nlev}: complex eigenvalues rel|Im|={rel_imag:.2e}"
        assert w.real.min() > 0.0, f"sigma L{nlev}: non-positive eigenvalue {w.real.min():.3e}"

    @pytest.mark.parametrize("nlev", [4, 8, 20, 32])
    def test_hybrid_gamma_eigenvalues_real_positive(self, nlev):
        from legoesm.timestepping.semi_implicit import compute_Gamma_matrix
        coord = standard_hybrid_levels(nlev)
        w, rel_imag = self._eig(compute_Gamma_matrix(coord, T_ref=300.0))
        assert rel_imag < 1e-8, f"hybrid L{nlev}: complex eigenvalues rel|Im|={rel_imag:.2e}"
        assert w.real.min() > 0.0, f"hybrid L{nlev}: non-positive eigenvalue {w.real.min():.3e}"

    def test_external_mode_equivalent_depth_physical(self):
        """Largest eigenvalue = external (Lamb) mode; H = lambda/g ~ 10 km."""
        from legoesm import constants
        from legoesm.timestepping.semi_implicit import compute_Gamma_matrix
        coord = standard_hybrid_levels(32)
        w, _ = self._eig(compute_Gamma_matrix(coord, T_ref=300.0))
        H_max = w.real.max() / constants.g
        assert 8_000.0 < H_max < 15_000.0, f"external-mode equiv depth {H_max:.0f} m unphysical"

    @pytest.mark.parametrize("nlev", [4, 8, 20, 40])
    def test_diagonal_tau_is_non_normal_selftest(self, nlev):
        """NON-VACUOUS GUARD: feeding the *old* diagonal reference
        ``tau = T_ref*I`` reproduces the buggy ``Gamma = R_d*T_ref*(S + 1*b^T)``,
        which IS non-normal with complex eigenvalues at nlev >= 4.  This proves
        the real-eigenvalue gate above actually discriminates the fix from the
        regression (it fails on the old form)."""
        from legoesm.timestepping.semi_implicit import compute_Gamma_matrix
        coord = create_sigma_coordinate(nlev, sigma_top=0.01)
        tau_diag = jnp.asarray(300.0 * np.eye(nlev), dtype=jnp.float64)
        _, rel_imag = self._eig(compute_Gamma_matrix(coord, T_ref=300.0, tau=tau_diag))
        assert rel_imag > 1e-3, (
            f"diagonal-tau Gamma should be non-normal at nlev={nlev}, "
            f"got rel|Im|={rel_imag:.2e}"
        )

    def test_stored_tau_matches_gamma(self):
        """The temperature correction and the implicit reference must share the
        SAME linearization: si_data.Gamma == R_d*(S @ si_data.tau + T_ref*1*b^T)."""
        from legoesm.grids.gaussian import create_gaussian_grid
        from legoesm.timestepping.semi_implicit import (
            precompute_si_matrices, compute_Gamma_matrix,
        )
        grid = create_gaussian_grid(21)
        coord = standard_hybrid_levels(20)
        si = precompute_si_matrices(grid, coord, T_ref=300.0, dt=1800.0)
        assert si.tau.shape == (20, 20)
        G = compute_Gamma_matrix(coord, T_ref=300.0, tau=si.tau)
        np.testing.assert_allclose(
            np.asarray(si.Gamma), np.asarray(G), rtol=1e-12, atol=0.0,
        )


class TestHybridDifferentiability:
    """Test JAX differentiability of hybrid coordinate functions."""

    def test_geopotential_differentiable(self):
        """compute_geopotential_hybrid should be differentiable w.r.t. T."""
        coord = make_hybrid_levels(NLEV)
        p_s = jnp.full((2, 2, 2), P_REF)
        phis = jnp.zeros((2, 2, 2))

        def f(T):
            return jnp.sum(compute_geopotential_hybrid(T, p_s, coord, phis))

        T = jnp.full((2, 2, 2, NLEV), 280.0)
        grad = jax.grad(f)(T)
        assert grad.shape == T.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_mass_flux_differentiable(self):
        """compute_mass_flux_hybrid should be differentiable."""
        coord = make_hybrid_levels(NLEV)
        p_s = jnp.full((2, 2, 2), P_REF)

        def f(div):
            mf, _ = compute_mass_flux_hybrid(div, p_s, coord)
            return jnp.sum(mf ** 2)

        div = jnp.ones((2, 2, 2, NLEV)) * 1e-5
        grad = jax.grad(f)(div)
        assert grad.shape == div.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_pressure_differentiable_wrt_ps(self):
        """Pressure should be differentiable w.r.t. p_s."""
        coord = make_hybrid_levels(NLEV)

        def f(p_s):
            return jnp.sum(pressure_from_hybrid(coord, p_s))

        p_s = jnp.full((2, 2, 2), P_REF)
        grad = jax.grad(f)(p_s)
        assert grad.shape == p_s.shape
        assert jnp.all(jnp.isfinite(grad))


# ==============================================================================
# Task 5: Enhanced vertical resolution and level design
# ==============================================================================

class TestStretching:
    """Test sinh-based stretching for boundary layer resolution."""

    def test_stretching_concentrates_near_surface(self):
        """With stretching > 0, layers should be thinner near the surface."""
        coord_uniform = make_hybrid_levels(40, stretching=0.0)
        coord_stretched = make_hybrid_levels(40, stretching=2.0)

        # Compute dp at reference for the bottom 5 levels
        dp_uniform = jnp.array(coord_uniform.dA + coord_uniform.dB) * P_REF
        dp_stretched = jnp.array(coord_stretched.dA + coord_stretched.dB) * P_REF

        # Bottom levels should be thinner with stretching
        bottom_mean_uniform = float(jnp.mean(dp_uniform[-5:]))
        bottom_mean_stretched = float(jnp.mean(dp_stretched[-5:]))
        assert bottom_mean_stretched < bottom_mean_uniform

    def test_stretching_preserves_pressure_range(self):
        """Total pressure range should be the same regardless of stretching."""
        for s in [0.0, 1.0, 2.0, 3.0]:
            coord = make_hybrid_levels(40, p_top_Pa=200.0, stretching=s)
            p_half_ref = (coord.A_half + coord.B_half) * P_REF
            assert float(p_half_ref[0]) == pytest.approx(200.0, rel=1e-3)
            assert float(p_half_ref[-1]) == pytest.approx(P_REF, rel=1e-5)

    def test_stretching_zero_is_uniform(self):
        """stretching=0 should give uniform pressure spacing at reference."""
        coord = make_hybrid_levels(40, p_top_Pa=200.0, stretching=0.0)
        p_half_ref = (coord.A_half + coord.B_half) * P_REF
        dp_ref = jnp.diff(p_half_ref)
        np.testing.assert_allclose(dp_ref, dp_ref[0], rtol=1e-4)

    def test_stretching_positive_dp(self):
        """All layers should have positive pressure thickness with stretching."""
        for s in [1.0, 2.0, 3.0, 4.0]:
            coord = make_hybrid_levels(40, stretching=s)
            p_s = jnp.full((2, 2, 2), P_REF)
            dp = dp_from_hybrid(coord, p_s)
            assert jnp.all(dp > 0), f"stretching={s} gave non-positive dp"

    def test_stretching_monotonic_pressure(self):
        """Pressure should increase monotonically from top to surface."""
        coord = make_hybrid_levels(60, p_top_Pa=10.0, stretching=2.5)
        p_s = jnp.full((2, 2, 2), P_REF)
        p_half = pressure_from_hybrid(coord, p_s, full=False)
        dp = jnp.diff(p_half, axis=-1)
        assert jnp.all(dp > 0), "Pressure not monotonically increasing"


class TestStandardLevels:
    """Test standard_hybrid_levels presets."""

    @pytest.mark.parametrize("nlev", [20, 40, 60])
    def test_standard_creates_valid_coordinate(self, nlev):
        """standard_hybrid_levels should create valid coordinates."""
        coord = standard_hybrid_levels(nlev)
        assert coord.n_levels == nlev
        assert coord.A_half.shape == (nlev + 1,)
        assert float(coord.B_half[0]) == pytest.approx(0.0, abs=1e-7)
        assert float(coord.B_half[-1]) == pytest.approx(1.0, abs=1e-7)

    @pytest.mark.parametrize("nlev", [20, 40, 60])
    def test_standard_positive_dp(self, nlev):
        """All standard level sets should have positive dp."""
        coord = standard_hybrid_levels(nlev)
        p_s = jnp.full((2, 2), P_REF)
        dp = dp_from_hybrid(coord, p_s)
        assert jnp.all(dp > 0)

    @pytest.mark.parametrize("nlev", [20, 40, 60])
    def test_standard_monotonic_pressure(self, nlev):
        """All standard level sets should have monotonic pressure."""
        coord = standard_hybrid_levels(nlev)
        p_half_ref = (coord.A_half + coord.B_half) * P_REF
        dp = jnp.diff(p_half_ref)
        assert jnp.all(dp > 0)

    def test_l40_has_bl_resolution(self):
        """L40 should have enhanced BL resolution (thinner bottom layers)."""
        coord = standard_hybrid_levels(40)
        p_half_ref = (coord.A_half + coord.B_half) * P_REF
        dp_ref = jnp.diff(p_half_ref)

        # Bottom layer should be thinner than mean
        mean_dp = float(jnp.mean(dp_ref))
        bottom_dp = float(dp_ref[-1])
        assert bottom_dp < mean_dp * 0.5, (
            f"Bottom layer {bottom_dp:.0f} Pa not much thinner than mean {mean_dp:.0f} Pa"
        )

    def test_l40_ptop(self):
        """L40 should have p_top around 200 Pa."""
        coord = standard_hybrid_levels(40)
        p_half_ref = (coord.A_half + coord.B_half) * P_REF
        p_top = float(p_half_ref[0])
        assert 100 < p_top < 500, f"L40 p_top={p_top:.0f} Pa outside expected range"

    def test_l60_ptop(self):
        """L60 should have p_top around 10 Pa (0.1 hPa)."""
        coord = standard_hybrid_levels(60)
        p_half_ref = (coord.A_half + coord.B_half) * P_REF
        p_top = float(p_half_ref[0])
        assert 1 < p_top < 50, f"L60 p_top={p_top:.0f} Pa outside expected range"

    def test_l40_geopotential_finite(self):
        """L40 geopotential should be finite and well-behaved."""
        coord = standard_hybrid_levels(40)
        T = jnp.full((2, 2, 2, 40), 250.0)
        p_s = jnp.full((2, 2, 2), P_REF)
        phis = jnp.zeros((2, 2, 2))
        Phi = compute_geopotential_hybrid(T, p_s, coord, phis)
        assert jnp.all(jnp.isfinite(Phi))
        assert float(Phi[0, 0, 0, 0]) > float(Phi[0, 0, 0, -1])

    @pytest.mark.parametrize("nlev", [15, 25, 35, 50, 80])
    def test_arbitrary_nlev(self, nlev):
        """standard_hybrid_levels should work for arbitrary nlev values."""
        coord = standard_hybrid_levels(nlev)
        assert coord.n_levels == nlev
        p_s = jnp.full((2, 2), P_REF)
        dp = dp_from_hybrid(coord, p_s)
        assert jnp.all(dp > 0)


class TestSB81FullLevelLnP:
    """#1029: SB81 full-level log-pressure — the discrete pair of Phi."""

    @pytest.fixture(autouse=True)
    def _fp64(self):
        # The machine-precision invariants below (1e-12 relative) require
        # fp64; make the class self-contained rather than depending on the
        # runner's JAX_ENABLE_X64 environment.
        from legoesm.core.precision import (
            set_policy, get_policy, PrecisionPolicy)
        prev = get_policy()
        set_policy(PrecisionPolicy.fp64())
        try:
            yield
        finally:
            set_policy(prev)

    def _coord(self, nlev=20):
        from legoesm.grids.vertical import standard_hybrid_levels
        return standard_hybrid_levels(nlev)

    def test_shared_alpha_bit_identical_with_geopotential(self):
        """sb81_halflevel_construction is THE construction Phi integrates."""
        from legoesm.grids.vertical import (
            sb81_halflevel_construction, compute_geopotential_hybrid)
        from legoesm import constants
        coord = self._coord()
        p_s = jnp.asarray([[9.3e4, 1.01e5], [6.5e4, 1.03e5]], dtype=jnp.float64)
        p_half_safe, ln_ratio, alpha = sb81_halflevel_construction(coord, p_s)
        # Reconstruct Phi from the triple exactly as compute_geopotential_hybrid
        T = jnp.full(p_s.shape + (coord.n_levels,), 260.0, dtype=jnp.float64)
        phis = jnp.zeros_like(p_s)
        dPhi = constants.R_d * T * ln_ratio
        cs = jnp.cumsum(dPhi[..., ::-1], axis=-1)[..., ::-1]
        Phi_above = phis[..., None] + cs
        Phi_below = jnp.concatenate([Phi_above[..., 1:], phis[..., None]], axis=-1)
        Phi_rec = Phi_below + alpha * constants.R_d * T
        Phi = compute_geopotential_hybrid(T, p_s, coord, phis)
        np.testing.assert_array_equal(np.asarray(Phi_rec), np.asarray(Phi))

    def test_full_level_ln_p_inside_layer(self):
        """exp(ln p_k) must lie strictly inside (p_{k-1/2}, p_{k+1/2})."""
        from legoesm.grids.vertical import (
            sb81_full_level_ln_p, sb81_halflevel_construction)
        coord = self._coord()
        p_s = jnp.asarray([8.0e4, 1.0e5], dtype=jnp.float64)
        lnp = sb81_full_level_ln_p(coord, p_s)
        p_half_safe, _, _ = sb81_halflevel_construction(coord, p_s)
        p_sb = np.exp(np.asarray(lnp))
        assert np.all(p_sb < np.asarray(p_half_safe[..., 1:]))
        # top layer p_{k-1/2} can be tiny; allow equality only there
        assert np.all(p_sb[..., 1:] > np.asarray(p_half_safe[..., 1:-1]))

    def test_isothermal_rest_invariant(self):
        """Phi_k + R_d T0 ln p_k^SB is COLUMN-INDEPENDENT at isothermal rest.

        With phis = -R_d T0 ln(p_s/p0), the sum telescopes to R_d T0 ln p0 at
        every level, for ANY p_s — so any discrete horizontal gradient of
        (Phi + R_d T ln p^SB) vanishes, which is exactly the #1029
        rest-over-terrain balance the momentum PGF needs.
        """
        from legoesm.grids.vertical import (
            sb81_full_level_ln_p, compute_geopotential_hybrid)
        from legoesm import constants
        coord = self._coord()
        T0 = 300.0
        p0 = 1.0e5
        p_s = jnp.asarray(
            [6.0e4, 7.5e4, 9.0e4, 1.0e5, 1.05e5], dtype=jnp.float64)
        phis = -constants.R_d * T0 * jnp.log(p_s / p0)
        T = jnp.full(p_s.shape + (coord.n_levels,), T0, dtype=jnp.float64)
        Phi = compute_geopotential_hybrid(T, p_s, coord, phis)
        lnp = sb81_full_level_ln_p(coord, p_s)
        inv = np.asarray(Phi + constants.R_d * T0 * lnp)  # (5, nlev)
        ref = constants.R_d * T0 * np.log(p0)
        # column-to-column spread per level must vanish (FP cumsum noise only)
        spread = np.abs(inv - ref).max()
        assert spread < 1e-12 * abs(ref), (
            f"isothermal rest invariant violated: spread {spread:.3e} "
            f"vs |ref| {abs(ref):.3e}")

    def test_a0_reduces_to_sigma_form(self):
        """A=0: ln p^SB - ln p_s is column-independent per level (to ~1e-12).

        So grad(ln p^SB) matches grad(ln p_s) — the sigma-path correction —
        up to the top-layer zero-clip artifact documented below; NOT an exact
        identity in the clipped top layer.
        """
        from legoesm.grids.vertical import (
            sb81_full_level_ln_p, create_hybrid_coordinate,
            standard_hybrid_levels)
        std = standard_hybrid_levels(20)
        coord = create_hybrid_coordinate(
            20, jnp.zeros_like(std.A_half), std.B_half)
        p_s = jnp.asarray([7.0e4, 8.5e4, 1.0e5], dtype=jnp.float64)
        lnp = np.asarray(sb81_full_level_ln_p(coord, p_s))
        # ln p^SB(ps) - ln(ps) must be the same constant for every column.
        # Tol 1e-11, not machine-eps: at A=0 the top interface is p=0 clipped
        # to 1e-10 Pa, which makes alpha_0 column-dependent at ~1e-12 — a clip
        # artifact confined to the top layer, physically nil.
        resid = lnp - np.log(np.asarray(p_s))[..., None]
        assert np.abs(resid - resid[0]).max() < 1e-11
