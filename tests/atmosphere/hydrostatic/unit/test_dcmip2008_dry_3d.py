"""Sanity tests for the DCMIP 2008 dry-3D Hughes-tutorial cases.

Covers four IC modules added in M1.b:

- ``gravity_wave_3_1``  — gravity wave on (notionally) non-rotating Earth
- ``inertio_gravity_3_2`` — inertio-gravity wave on rotating planet
- ``mountain_rossby_5_0`` — mountain-induced Rossby wave (Gaussian peak)
- ``rossby_haurwitz_6_0`` — small-amplitude 3D Rossby-Haurwitz wave-4

Each test asserts shape, finite values, and physical sanity bounds
appropriate to the case (T near 300 K, max |u| reasonable, surface
pressure above the freezing-isotherm exponential floor, etc.).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import standard_hybrid_levels
from legoesm.grids.voronoi import create_voronoi_mesh


@pytest.fixture(scope="module")
def sigma8():
    return standard_hybrid_levels(8)


# ---------------------------------------------------------------------------
# Shared assertions for any DCMIP 2008 dry-3D state
# ---------------------------------------------------------------------------


def _assert_dry3d_state_grid_space(state, *, T_min=200.0, T_max=320.0,
                                   p_s_max=constants.p_ref):
    """Common asserts for grid-space (cubed-sphere / lat-lon) states."""
    assert bool(jnp.all(jnp.isfinite(state.T.data)))
    assert bool(jnp.all(jnp.isfinite(state.u.data)))
    assert bool(jnp.all(jnp.isfinite(state.v.data)))
    assert bool(jnp.all(jnp.isfinite(state.p_s.data)))

    # Temperature near isothermal background ± Δθ perturbation
    assert T_min <= float(jnp.min(state.T.data))
    assert float(jnp.max(state.T.data)) <= T_max

    # Surface pressure positive and bounded
    assert float(jnp.min(state.p_s.data)) > 0.0
    assert float(jnp.max(state.p_s.data)) <= p_s_max + 1e-6


def _assert_dry3d_state_mpas(state, *, T_min=200.0, T_max=320.0):
    assert bool(jnp.all(jnp.isfinite(state.T.data)))
    assert bool(jnp.all(jnp.isfinite(state.u.data)))
    assert bool(jnp.all(jnp.isfinite(state.p_s.data)))
    assert T_min <= float(jnp.min(state.T.data))
    assert float(jnp.max(state.T.data)) <= T_max
    assert float(jnp.min(state.p_s.data)) > 0.0


def _assert_dry3d_state_spectral(state):
    assert bool(jnp.all(jnp.isfinite(state.vor_hat.data.real)))
    assert bool(jnp.all(jnp.isfinite(state.div_hat.data.real)))
    assert bool(jnp.all(jnp.isfinite(state.T_hat.data.real)))
    assert bool(jnp.all(jnp.isfinite(state.lnps_hat.data.real)))


# ---------------------------------------------------------------------------
# DCMIP 2008 §3-1 — gravity wave on non-rotating Earth
# ---------------------------------------------------------------------------


class TestGravityWave31:
    def test_cubed_sphere(self, sigma8):
        from tests.test_cases.dcmip2008.gravity_wave_3_1 import (
            gravity_wave_init,
        )
        grid = create_cubed_sphere(8)
        state = gravity_wave_init(grid, sigma8)
        _assert_dry3d_state_grid_space(state)
        # Solid-body u_0 ≈ 20 m/s with cos(lat) ≤ 1 so |u| ≤ ~20 m/s
        assert float(jnp.max(jnp.abs(state.u.data))) < 25.0
        # Theta perturbation is small (δθ = 1 K) so T variation < 5 K
        T_dev = float(jnp.max(jnp.abs(state.T.data - 300.0)))
        assert T_dev < 5.0

    def test_latlon(self, sigma8):
        from tests.test_cases.dcmip2008.gravity_wave_3_1 import (
            gravity_wave_init_latlon,
        )
        grid = create_latlon_grid(24, 48)
        state = gravity_wave_init_latlon(grid, sigma8)
        _assert_dry3d_state_grid_space(state)

    def test_mpas(self, sigma8):
        from tests.test_cases.dcmip2008.gravity_wave_3_1 import (
            gravity_wave_init_mpas,
        )
        mesh = create_voronoi_mesh(4)
        state = gravity_wave_init_mpas(mesh, sigma8)
        _assert_dry3d_state_mpas(state)

    def test_spectral(self, sigma8):
        from tests.test_cases.dcmip2008.gravity_wave_3_1 import (
            gravity_wave_init_spectral,
        )
        grid = create_gaussian_grid(21)
        state = gravity_wave_init_spectral(grid, sigma8)
        _assert_dry3d_state_spectral(state)


# ---------------------------------------------------------------------------
# DCMIP 2008 §3-2 — inertio-gravity wave on rotating planet
# ---------------------------------------------------------------------------


class TestInertioGravity32:
    def test_cubed_sphere(self, sigma8):
        from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
            inertio_gravity_init,
        )
        grid = create_cubed_sphere(8)
        state = inertio_gravity_init(grid, sigma8)
        _assert_dry3d_state_grid_space(state)
        # §3-2 uses smaller δθ = 0.5 K so T variation < 3 K
        T_dev = float(jnp.max(jnp.abs(state.T.data - 300.0)))
        assert T_dev < 3.0

    def test_latlon(self, sigma8):
        from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
            inertio_gravity_init_latlon,
        )
        grid = create_latlon_grid(24, 48)
        state = inertio_gravity_init_latlon(grid, sigma8)
        _assert_dry3d_state_grid_space(state)

    def test_mpas(self, sigma8):
        from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
            inertio_gravity_init_mpas,
        )
        mesh = create_voronoi_mesh(4)
        state = inertio_gravity_init_mpas(mesh, sigma8)
        _assert_dry3d_state_mpas(state)

    def test_spectral(self, sigma8):
        from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
            inertio_gravity_init_spectral,
        )
        grid = create_gaussian_grid(21)
        state = inertio_gravity_init_spectral(grid, sigma8)
        _assert_dry3d_state_spectral(state)


# ---------------------------------------------------------------------------
# DCMIP 2008 §5-0 — mountain-induced Rossby wave
# ---------------------------------------------------------------------------


class TestMountainRossby50:
    def test_cubed_sphere(self, sigma8):
        from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
            mountain_rossby_init,
        )
        grid = create_cubed_sphere(8)
        state = mountain_rossby_init(grid, sigma8)
        _assert_dry3d_state_grid_space(state, T_min=280.0, T_max=295.0)
        # Mountain peak ≈ 2000 m → φ_s up to ≈ 19620 m²/s²
        phis_max = float(jnp.max(state.phis.data))
        assert 0.0 < phis_max <= constants.g * 2000.0 + 1.0
        # Surface pressure must drop over the mountain
        p_min = float(jnp.min(state.p_s.data))
        assert p_min < constants.p_ref

    def test_latlon(self, sigma8):
        from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
            mountain_rossby_init_latlon,
        )
        grid = create_latlon_grid(24, 48)
        state = mountain_rossby_init_latlon(grid, sigma8)
        _assert_dry3d_state_grid_space(state, T_min=280.0, T_max=295.0)
        assert float(jnp.max(state.phis.data)) > 0.0

    def test_mpas(self, sigma8):
        from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
            mountain_rossby_init_mpas,
        )
        mesh = create_voronoi_mesh(4)
        state = mountain_rossby_init_mpas(mesh, sigma8)
        _assert_dry3d_state_mpas(state, T_min=280.0, T_max=295.0)
        assert float(jnp.max(state.phis.data)) > 0.0

    def test_spectral(self, sigma8):
        from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
            mountain_rossby_init_spectral,
        )
        grid = create_gaussian_grid(21)
        state = mountain_rossby_init_spectral(grid, sigma8)
        _assert_dry3d_state_spectral(state)
        # phis should have non-zero spectral power
        assert float(jnp.max(jnp.abs(state.phis_hat.data))) > 0.0


# ---------------------------------------------------------------------------
# DCMIP 2008 3D Rossby-Haurwitz wave (reference test 4-0-0, Sect. 1.4)
# ---------------------------------------------------------------------------

# Lapse-rate columns reach ~100 K at the top of a ~2-4 hPa model lid
# (T0 (p/p_ref)^(Gamma R_d/g), Eq. 66) and p_s rises to ~1030 hPa (Eq. 70).
_RH_BOUNDS = dict(T_min=90.0, T_max=295.0, p_s_max=103500.0)


def _assert_rh_eq70_eq66(state, lon, lat, radius, coord):
    """p_s from Eq. (70) and T from Eq. (66) at the builder's own points,
    written out here from the reference (T0 = 288 K, Gamma = 6.5 K/km,
    p_ref = 955 hPa, M = K = 50/(4 a)); T at the coordinate's full levels
    p = A coord.p_ref + B p_s (Eq. 68)."""
    import numpy as np
    from legoesm.core.williamson_sw_analytic import (
        rossby_haurwitz_4_geopotential)
    gamma, t0, p_ref = 0.0065, 288.0, 95500.0
    mk = 50.0 / (4.0 * float(radius))
    phi = rossby_haurwitz_4_geopotential(
        np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64),
        radius=float(radius), omega=constants.Omega, gh0=0.0,
        omega_wave=mk, k_wave=mk)
    ps_ref = p_ref * (1.0 + gamma * phi / (constants.g * t0)) ** (
        constants.g / (gamma * constants.R_d))
    p_full = (np.asarray(coord.A_full) * coord.p_ref
              + np.asarray(coord.B_full) * ps_ref[..., None])
    T_ref = t0 * (p_full / p_ref) ** (constants.R_d * gamma / constants.g)
    # rtol 1e-6: the builders may store at the fp32 policy dtype
    np.testing.assert_allclose(np.asarray(state.p_s.data), ps_ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(state.T.data), T_ref, rtol=1e-6)
    assert ps_ref.max() - ps_ref.min() > 5000.0      # not uniform


class TestRossbyHaurwitz60:
    def test_cubed_sphere(self, sigma8):
        from tests.test_cases.dcmip2008.rossby_haurwitz_6_0 import (
            rossby_haurwitz_init,
        )
        grid = create_cubed_sphere(8)
        state = rossby_haurwitz_init(grid, sigma8)
        _assert_dry3d_state_grid_space(state, **_RH_BOUNDS)
        # u0 = 50 m/s, M = K = u0/(n a): analytic peak |V| = 25 m/s
        # (equator, lambda = pi/4); C8 cell centres sample 23.6.  The Williamson
        # case-6 amplitude (K = 7.848e-6) would peak at ~100 m/s.
        max_speed = float(jnp.max(jnp.sqrt(
            state.u.data ** 2 + state.v.data ** 2)))
        assert 20.0 < max_speed < 25.0 + 1e-6

    def test_latlon(self, sigma8):
        from tests.test_cases.dcmip2008.rossby_haurwitz_6_0 import (
            rossby_haurwitz_init_latlon,
        )
        grid = create_latlon_grid(24, 48)
        state = rossby_haurwitz_init_latlon(grid, sigma8)
        _assert_dry3d_state_grid_space(state, **_RH_BOUNDS)
        _assert_rh_eq70_eq66(state, grid.lon2d, grid.lat2d, grid.radius,
                             sigma8)
        # a hybrid table whose own p_ref is neither 1000 nor 955 hPa: the
        # level pressures use the table's p_ref, T uses the reference's
        coord = standard_hybrid_levels(8, p_ref=90000.0)
        _assert_rh_eq70_eq66(rossby_haurwitz_init_latlon(grid, coord),
                             grid.lon2d, grid.lat2d, grid.radius, coord)

    def test_mpas(self, sigma8):
        from tests.test_cases.dcmip2008.rossby_haurwitz_6_0 import (
            rossby_haurwitz_init_mpas,
        )
        mesh = create_voronoi_mesh(4)
        state = rossby_haurwitz_init_mpas(mesh, sigma8)
        _assert_dry3d_state_mpas(state, T_min=_RH_BOUNDS["T_min"],
                                 T_max=_RH_BOUNDS["T_max"])
        _assert_rh_eq70_eq66(state, mesh.lonCell, mesh.latCell, mesh.radius,
                             sigma8)

    def test_spectral(self, sigma8):
        from tests.test_cases.dcmip2008.rossby_haurwitz_6_0 import (
            rossby_haurwitz_init_spectral,
        )
        grid = create_gaussian_grid(21)
        state = rossby_haurwitz_init_spectral(grid, sigma8)
        _assert_dry3d_state_spectral(state)

    def test_surface_pressure_matches_reference(self):
        """Eq. (70): p_s = p_ref at both poles, and the area mean the
        reference quotes (~1000.377 hPa, text below Eq. 74).  The old IC
        (uniform 1000 hPa) misses the mean by 0.377 hPa."""
        import numpy as np
        from tests.test_cases.dcmip2008.rossby_haurwitz_6_0 import (
            RH_P_REF, rh_balanced_columns,
        )
        x, w = np.polynomial.legendre.leggauss(200)
        lat = np.arcsin(x)
        lon = np.linspace(0.0, 2.0 * np.pi, 800, endpoint=False)
        lon2, lat2 = np.meshgrid(lon, lat)
        p_s, _ = rh_balanced_columns(jnp.asarray(lon2), jnp.asarray(lat2),
                                     constants.R_earth, standard_hybrid_levels(8))
        p_s = np.asarray(p_s)
        mean_hpa = float(p_s.mean(axis=1) @ w) / 2.0 / 100.0
        # legoesm R_d/Omega give 1000.3748 (reference Appendix G constants
        # give 1000.3771): 5e-3 hPa covers that constants difference only.
        assert abs(mean_hpa - 1000.377) < 5e-3, mean_hpa
        p_pole, _ = rh_balanced_columns(
            jnp.asarray([0.3, 1.7]), jnp.asarray([np.pi / 2, -np.pi / 2]),
            constants.R_earth, standard_hybrid_levels(8))
        np.testing.assert_allclose(np.asarray(p_pole), RH_P_REF, rtol=1e-12)
        # Spatial phase, independent of the shared geopotential helper:
        # Eqs. (72)-(74) written out at (lambda, phi) = (0, 45 deg) and
        # (pi/4, 45 deg), where cos(n lambda) = +1 / -1, cos(2 n lambda) = 1.
        n, a, om = 4.0, constants.R_earth, constants.Omega
        mk = 50.0 / (n * a)
        c = np.cos(np.pi / 4.0)
        A = (mk * (2 * om + mk) / 2 * c ** 2
             + mk ** 2 / 4 * c ** (2 * n) * ((n + 1) * c ** 2 + 2 * n ** 2 - n - 2)
             - n ** 2 * mk ** 2 / 2 * c ** (2 * (n - 1)))
        B = (2 * (om + mk) * mk / ((n + 1) * (n + 2)) * c ** n
             * ((n ** 2 + 2 * n + 2) - (n + 1) ** 2 * c ** 2))
        C = mk ** 2 / 4 * c ** (2 * n) * ((n + 1) * c ** 2 - (n + 2))
        phi_prime = a ** 2 * np.array([A + B + C, A - B + C])
        expo = constants.g / (0.0065 * constants.R_d)
        ps_hand = 95500.0 * (1 + 0.0065 * phi_prime / (constants.g * 288.0)) ** expo
        p_pts, _ = rh_balanced_columns(
            jnp.asarray([0.0, np.pi / 4.0]), jnp.asarray([np.pi / 4.0] * 2),
            constants.R_earth, standard_hybrid_levels(8))
        np.testing.assert_allclose(np.asarray(p_pts), ps_hand, rtol=1e-12)
        assert ps_hand[0] - ps_hand[1] > 1000.0      # wave-4 ridge at lambda=0

    def test_initial_divergence_tendency_is_balanced(self):
        """The reference's p_s/T balance the winds: the spectral PE's initial
        divergence tendency is small next to its vorticity tendency (the
        wave's propagation).  On T21 / CAM6 L32: MEASURED rms ratio 6.6e-3
        for this IC and 10.1 for the old one (uniform p_s, isothermal)."""
        import numpy as np
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPEConfig, spectral_pe_tendencies)
        from legoesm.grids.gaussian import sh_synthesis_3d
        from legoesm.grids.vertical import make_cam6_l32_levels
        from tests.test_cases.dcmip2008.rossby_haurwitz_6_0 import (
            rossby_haurwitz_init_spectral,
        )
        grid = create_gaussian_grid(21)
        coord = make_cam6_l32_levels()
        state = rossby_haurwitz_init_spectral(grid, coord)
        tend = spectral_pe_tendencies(state, grid, coord, SpectralPEConfig())

        def rms(hat):
            return float(np.sqrt(np.mean(
                np.asarray(sh_synthesis_3d(grid, hat)) ** 2)))
        ratio = rms(tend.div_hat.data) / rms(tend.vor_hat.data)
        assert ratio < 0.05, ratio


# ---------------------------------------------------------------------------
# Small-planet helpers
# ---------------------------------------------------------------------------


class TestSmallPlanet:
    def test_scaled_radius_and_omega(self):
        from legoesm.atmosphere.idealized.small_planet import (
            scaled_omega,
            scaled_radius,
        )
        R = scaled_radius(125)
        Omega_eff = scaled_omega(125)
        # Small-planet R · Ω product equals Earth's: 125· Ω · R/125 = Ω·R
        assert R * Omega_eff == pytest.approx(
            constants.R_earth * constants.Omega, rel=1e-12)

    def test_make_small_planet_cubed_sphere(self):
        from legoesm.atmosphere.idealized.small_planet import (
            make_small_planet_cubed_sphere,
        )
        grid = make_small_planet_cubed_sphere(8, factor=125)
        assert float(grid.radius) == pytest.approx(
            constants.R_earth / 125.0, rel=1e-9)

    def test_make_small_planet_gaussian_scales_omega(self):
        """Spectral small-planet now also rescales Ω via the new
        ``omega`` parameter on ``create_gaussian_grid``.  The Coriolis
        baked into ``grid.f`` should reflect the scaled rotation rate
        ``Ω · factor``."""
        from legoesm.atmosphere.idealized.small_planet import (
            make_small_planet_gaussian,
        )
        grid = make_small_planet_gaussian(21, factor=125)
        assert float(grid.radius) == pytest.approx(
            constants.R_earth / 125.0, rel=1e-9)
        # f = 2·Ω·sin(lat); at the northern-most Gauss point sin(lat) ≈
        # 0.992 and Ω_eff = Ω·125, so f_max ≈ 2·125·Ω·0.992.
        f_max = float(jnp.max(grid.f))
        f_max_expected = 2.0 * 125 * constants.Omega * float(
            jnp.max(jnp.abs(jnp.sin(grid.lat))))
        assert f_max == pytest.approx(f_max_expected, rel=1e-6)


# ---------------------------------------------------------------------------
# DCMIP §3-1 must be runnable on a truly non-rotating sphere
# ---------------------------------------------------------------------------


class TestNonRotatingGravityWave:
    """``create_gaussian_grid(omega=0)`` produces an Ω=0 grid with f=0,
    so DCMIP §3-1 ships canonical rather than as an approximation."""

    def test_gaussian_omega_zero_zeroes_coriolis(self):
        from legoesm.grids.gaussian import create_gaussian_grid
        grid = create_gaussian_grid(21, omega=0.0)
        assert float(jnp.max(jnp.abs(grid.f))) == 0.0

    def test_cubed_sphere_omega_zero_zeroes_coriolis(self):
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        grid = create_cubed_sphere(8, omega=0.0)
        # CubedSphereGrid stores f = 2·Ω·sin(lat); under Ω=0 it must
        # be zero everywhere.
        assert float(jnp.max(jnp.abs(grid.f))) == 0.0

    def test_cubed_sphere_cdgrid_inherits_omega_zero(self):
        """The CD-grid corner Coriolis must follow the base grid's Ω.

        Regression for an earlier bug where ``create_cubed_sphere_cdgrid``
        defaulted to ``omega=constants.Omega`` and silently overrode a
        non-rotating base grid — making §3-1 still feel Earth's Coriolis
        at the corners even after we passed ``omega=0`` to the base
        factory.
        """
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import (
            create_cubed_sphere_cdgrid,
        )
        base = create_cubed_sphere(8, omega=0.0)
        cdgrid = create_cubed_sphere_cdgrid(base)
        assert float(jnp.max(jnp.abs(cdgrid.f_corner))) == 0.0

    def test_cubed_sphere_cdgrid_default_inherits_earth(self):
        """Earth-default base grid → Earth-default cdgrid (no regression
        for the canonical AMIP / Held-Suarez path)."""
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import (
            create_cubed_sphere_cdgrid,
        )
        base = create_cubed_sphere(8)  # default omega = Earth's
        cdgrid = create_cubed_sphere_cdgrid(base)
        # f_corner should be of order 2·Ω·O(1) ≈ 1e-4, NOT zero.
        assert float(jnp.max(jnp.abs(cdgrid.f_corner))) > 1e-5

    def test_latlon_omega_zero_zeroes_coriolis(self):
        from legoesm.grids.latlon import create_latlon_grid
        grid = create_latlon_grid(24, 48, omega=0.0)
        assert float(jnp.max(jnp.abs(grid.f))) == 0.0

    def test_voronoi_omega_zero_zeroes_coriolis(self):
        from legoesm.grids.voronoi import create_voronoi_mesh
        mesh = create_voronoi_mesh(4, omega=0.0)
        # MPAS stores Coriolis as fEdge/fVertex; both must be zero.
        if hasattr(mesh, "fEdge"):
            assert float(jnp.max(jnp.abs(mesh.fEdge))) == 0.0
        if hasattr(mesh, "fVertex"):
            assert float(jnp.max(jnp.abs(mesh.fVertex))) == 0.0
