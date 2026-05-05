"""Tests for ocean diagnostics: deformation radius and energy/enstrophy spectra.

Run with:

    JAX_ENABLE_X64=1 python3 -m pytest tests/ocean/unit/test_ocean_diagnostics.py -v
"""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.diagnostics import (
    first_baroclinic_deformation_radius,
    isotropic_energy_spectrum,
    isotropic_enstrophy_spectrum,
)
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm import constants


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


@pytest.fixture(scope="module")
def z_coord():
    """Simple uniform-thickness vertical coordinate for testing."""
    return create_ocean_z_star(
        n_levels=10, H_max=1000.0, dz_surface=100.0, dz_deep=100.0,
    )


# ============================================================================
# 1. Deformation radius tests
# ============================================================================


class TestDeformationRadius:
    """Tests for ``first_baroclinic_deformation_radius``."""

    def _uniform_rho(self, z_coord, N_target, rho_ref=1025.0):
        """Build a density profile with uniform N^2 = N_target^2.

        From N^2 = -(g/rho_ref) * drho/dz, with dz > 0 downward and
        rho increasing with depth:
            drho/dz = -(rho_ref / g) * N^2
        So rho[k] = rho_ref - (rho_ref/g)*N^2 * z_full[k], where z < 0.
        """
        g = constants.g
        N2 = N_target ** 2
        z = z_coord.z_full_ref  # (nlev,) negative values
        rho = rho_ref - (rho_ref / g) * N2 * z
        return rho

    def test_shape(self, z_coord):
        """Output shape matches horizontal shape of inputs."""
        n_lat, n_lon = 4, 8
        rho = self._uniform_rho(z_coord, 1e-2)
        rho_3d = jnp.broadcast_to(rho, (n_lat, n_lon, z_coord.n_levels))
        jacobian = jnp.ones((n_lat, n_lon))
        f = 1e-4 * jnp.ones((n_lat, n_lon))

        L_d = first_baroclinic_deformation_radius(
            rho_3d, z_coord, jacobian, f)
        assert L_d.shape == (n_lat, n_lon)

    def test_non_negative(self, z_coord):
        """Deformation radius is non-negative for stable stratification."""
        rho = self._uniform_rho(z_coord, 1e-2)
        rho_3d = rho[jnp.newaxis, jnp.newaxis, :]
        jacobian = jnp.ones((1, 1))
        f = 1e-4 * jnp.ones((1, 1))

        L_d = first_baroclinic_deformation_radius(
            rho_3d, z_coord, jacobian, f)
        assert float(L_d.squeeze()) >= 0.0

    def test_uniform_stratification_analytical(self):
        """For uniform N, L_d = N*H / (pi*|f|).

        The discrete integral sum(N * dz_half) covers (nlev-1)/nlev of H
        because N^2 is defined at interior interfaces only.  Use many
        levels to keep the boundary truncation small.
        """
        z50 = create_ocean_z_star(
            n_levels=50, H_max=1000.0, dz_surface=20.0, dz_deep=20.0)
        N_target = 1e-2  # 0.01 s^-1
        H = z50.H_max
        f_val = 1e-4

        rho = self._uniform_rho(z50, N_target)
        rho_3d = rho[jnp.newaxis, :]
        jacobian = jnp.ones((1,))
        f = f_val * jnp.ones((1,))

        L_d = first_baroclinic_deformation_radius(
            rho_3d, z50, jacobian, f)

        expected = N_target * H / (jnp.pi * f_val)
        # 50 levels → boundary truncation ~2%, allow 5% total
        assert jnp.allclose(L_d, expected, rtol=0.05), (
            f"L_d={float(L_d.squeeze()):.1f} vs expected={float(expected):.1f}")

    def test_inversely_proportional_to_f(self, z_coord):
        """Doubling |f| should halve L_d."""
        N_target = 1e-2
        rho = self._uniform_rho(z_coord, N_target)
        rho_3d = rho[jnp.newaxis, :]
        jacobian = jnp.ones((1,))

        f1 = 1e-4 * jnp.ones((1,))
        f2 = 2e-4 * jnp.ones((1,))

        L_d1 = first_baroclinic_deformation_radius(
            rho_3d, z_coord, jacobian, f1)
        L_d2 = first_baroclinic_deformation_radius(
            rho_3d, z_coord, jacobian, f2)

        ratio = float((L_d1 / L_d2).squeeze())
        assert jnp.allclose(ratio, 2.0, rtol=1e-10)

    def test_zero_stratification(self, z_coord):
        """Uniform density (N=0) gives L_d ≈ 0."""
        rho_ref = 1025.0
        rho_3d = jnp.full((1, z_coord.n_levels), rho_ref)
        jacobian = jnp.ones((1,))
        f = 1e-4 * jnp.ones((1,))

        L_d = first_baroclinic_deformation_radius(
            rho_3d, z_coord, jacobian, f)
        assert float(L_d.squeeze()) < 1.0  # essentially zero [m]

    def test_equatorial_f_min_floor(self, z_coord):
        """Near-equatorial f=0 is floored by f_min, preventing blow-up."""
        N_target = 1e-2
        rho = self._uniform_rho(z_coord, N_target)
        rho_3d = rho[jnp.newaxis, :]
        jacobian = jnp.ones((1,))
        f = jnp.zeros((1,))

        f_min = 1e-8
        L_d = first_baroclinic_deformation_radius(
            rho_3d, z_coord, jacobian, f, f_min=f_min)
        assert jnp.isfinite(L_d).all()
        # Should use f_min, giving a very large but finite radius
        expected_max = N_target * z_coord.H_max / (jnp.pi * f_min)
        assert float(L_d.squeeze()) <= float(expected_max) * 1.05

    def test_grad_finite(self, z_coord):
        """jax.grad through deformation radius is finite."""
        N_target = 1e-2
        rho = self._uniform_rho(z_coord, N_target)
        rho_3d = rho[jnp.newaxis, :]
        jacobian = jnp.ones((1,))
        f = 1e-4 * jnp.ones((1,))

        def loss(rho_in):
            return jnp.sum(first_baroclinic_deformation_radius(
                rho_in, z_coord, jacobian, f) ** 2)

        g = jax.grad(loss)(rho_3d)
        assert jnp.all(jnp.isfinite(g))


# ============================================================================
# 2. Energy spectrum tests
# ============================================================================


class TestEnergySpectrum:
    """Tests for ``isotropic_energy_spectrum``."""

    def test_single_mode_peak(self):
        """A single Fourier mode should produce a peak at the right wavenumber."""
        nx, ny = 64, 64
        dx = 1000.0  # 1 km grid spacing

        # Mode with wavenumber k = 4 * dk in x-direction
        mode_n = 4
        k_expected = mode_n / (nx * dx)  # [1/m]
        x = jnp.arange(nx) * dx
        u = jnp.sin(2 * jnp.pi * mode_n * x / (nx * dx))
        u = jnp.broadcast_to(u[jnp.newaxis, :], (ny, nx))
        v = jnp.zeros_like(u)

        k, spec = isotropic_energy_spectrum(u, v, dx, detrend=False)

        # Peak should be near k_expected
        peak_idx = jnp.argmax(spec)
        k_peak = float(k[peak_idx])
        assert abs(k_peak - k_expected) < 2 * float(k[1] - k[0]), (
            f"Peak at k={k_peak:.2e} but expected near {k_expected:.2e}")

    def test_parseval(self):
        """Total spectral energy should equal spatial variance (Parseval)."""
        nx, ny = 32, 32
        dx = 1000.0

        rng = np.random.RandomState(123)
        u = jnp.array(rng.randn(ny, nx) * 0.1, dtype=jnp.float64)
        v = jnp.array(rng.randn(ny, nx) * 0.1, dtype=jnp.float64)

        k, spec = isotropic_energy_spectrum(u, v, dx, detrend=True)

        # Spatial KE variance (per unit area)
        u_anom = u - jnp.mean(u)
        v_anom = v - jnp.mean(v)
        spatial_ke = 0.5 * jnp.mean(u_anom ** 2 + v_anom ** 2)

        # Integrate spectrum: sum(spec * dk)
        dk = float(k[1] - k[0]) if len(k) > 1 else 1.0
        spectral_ke = jnp.sum(spec) * dk

        # Should agree to within ~10% (binning introduces small errors)
        assert jnp.allclose(spectral_ke, spatial_ke, rtol=0.15), (
            f"Spectral KE={float(spectral_ke):.6e} vs spatial KE={float(spatial_ke):.6e}")

    def test_zero_field(self):
        """Zero velocity field gives zero spectrum."""
        nx, ny = 16, 16
        dx = 1000.0
        u = jnp.zeros((ny, nx))
        v = jnp.zeros((ny, nx))

        k, spec = isotropic_energy_spectrum(u, v, dx)
        assert jnp.allclose(spec, 0.0, atol=1e-30)

    def test_output_shapes(self):
        """Output arrays have consistent shapes."""
        nx, ny = 32, 64
        dx = 1000.0
        u = jnp.ones((ny, nx)) * 0.1
        v = jnp.zeros((ny, nx))

        k, spec = isotropic_energy_spectrum(u, v, dx)
        assert k.shape == spec.shape
        n_bins = max(nx, ny) // 2
        assert k.shape == (n_bins,)

    def test_spectrum_non_negative(self):
        """Power spectrum is non-negative by construction."""
        rng = np.random.RandomState(99)
        nx, ny = 32, 32
        dx = 500.0
        u = jnp.array(rng.randn(ny, nx) * 0.5, dtype=jnp.float64)
        v = jnp.array(rng.randn(ny, nx) * 0.5, dtype=jnp.float64)

        _, spec = isotropic_energy_spectrum(u, v, dx)
        assert jnp.all(spec >= 0.0)


# ============================================================================
# 3. Enstrophy spectrum tests
# ============================================================================


class TestEnstrophySpectrum:
    """Tests for ``isotropic_enstrophy_spectrum``."""

    def test_single_mode_peak(self):
        """Sinusoidal vorticity has a spectral peak at the right wavenumber."""
        nx, ny = 64, 64
        dx = 1000.0
        mode_n = 6
        k_expected = mode_n / (nx * dx)

        x = jnp.arange(nx) * dx
        zeta = jnp.sin(2 * jnp.pi * mode_n * x / (nx * dx))
        zeta = jnp.broadcast_to(zeta[jnp.newaxis, :], (ny, nx))

        k, spec = isotropic_enstrophy_spectrum(zeta, dx, detrend=False)

        peak_idx = jnp.argmax(spec)
        k_peak = float(k[peak_idx])
        assert abs(k_peak - k_expected) < 2 * float(k[1] - k[0])

    def test_zero_vorticity(self):
        """Zero vorticity gives zero enstrophy spectrum."""
        zeta = jnp.zeros((16, 16))
        _, spec = isotropic_enstrophy_spectrum(zeta, 1000.0)
        assert jnp.allclose(spec, 0.0, atol=1e-30)

    def test_parseval_enstrophy(self):
        """Total spectral enstrophy ≈ spatial half-variance of zeta."""
        nx, ny = 32, 32
        dx = 1000.0
        rng = np.random.RandomState(77)
        zeta = jnp.array(rng.randn(ny, nx) * 1e-5, dtype=jnp.float64)

        k, spec = isotropic_enstrophy_spectrum(zeta, dx, detrend=True)

        zeta_anom = zeta - jnp.mean(zeta)
        spatial_enstrophy = 0.5 * jnp.mean(zeta_anom ** 2)

        dk = float(k[1] - k[0]) if len(k) > 1 else 1.0
        spectral_enstrophy = jnp.sum(spec) * dk

        assert jnp.allclose(spectral_enstrophy, spatial_enstrophy, rtol=0.15), (
            f"Spectral={float(spectral_enstrophy):.6e} vs spatial={float(spatial_enstrophy):.6e}")

    def test_spectrum_non_negative(self):
        """Enstrophy spectrum is non-negative."""
        rng = np.random.RandomState(55)
        zeta = jnp.array(rng.randn(32, 32) * 1e-5, dtype=jnp.float64)
        _, spec = isotropic_enstrophy_spectrum(zeta, 500.0)
        assert jnp.all(spec >= 0.0)
