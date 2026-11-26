"""Sanity tests for the DCMIP 2008 dry-3D Hughes-tutorial cases.

Covers three IC modules added in M1.b (the gravity wave is DCMIP 2012
test 3-1: ``test_dcmip2012_gravity_wave_3_1.py``):

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
# DCMIP 2008 §3-2 — inertio-gravity wave on rotating planet
# ---------------------------------------------------------------------------


def _assert_ig_630(state, lon, lat, radius, coord):
    """DCMIP 2008 [6-3-0] written out from the reference at the builder's
    points: u = v = 0, p_s = 1000 hPa, T = 300 K + 10 K s sin(2 pi z/20 km)
    (p/p0)^kappa (Eqs. 94-98), s the cosine bell of radius a/3 about
    (pi, pi/4), z = (R_d T0/g) ln(p_s/p) (Eq. 92)."""
    import numpy as np
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    a, t0, p0 = float(radius), 300.0, constants.p_ref
    r = a * np.arccos(np.clip(np.sin(np.pi / 4) * np.sin(lat)
                              + np.cos(np.pi / 4) * np.cos(lat)
                              * np.cos(lon - np.pi), -1, 1))
    s = np.where(r < a / 3, 0.5 * (1 + np.cos(np.pi * r / (a / 3))), 0.0)
    p = (np.asarray(coord.A_full) * coord.p_ref
         + np.asarray(coord.B_full) * p0)
    z = constants.R_d * t0 / constants.g * np.log(p0 / p)
    T_ref = t0 + (10.0 * s[..., None] * np.sin(2 * np.pi * z / 20000.0)
                  * (p / p0) ** constants.kappa)
    np.testing.assert_allclose(np.asarray(state.p_s.data), p0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(state.u.data), 0.0, atol=0.0)
    # rtol 1e-6: the builders may store at the fp32 policy dtype
    np.testing.assert_allclose(np.asarray(state.T.data), T_ref, rtol=1e-6)


class TestInertioGravity32:
    """DCMIP 2008 test 6-3-0 (Sect. 1.6), row name kept."""

    def test_cubed_sphere(self, sigma8):
        from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
            inertio_gravity_init,
        )
        grid = create_cubed_sphere(8)
        state = inertio_gravity_init(grid, sigma8)
        _assert_dry3d_state_grid_space(state, T_min=290.0, T_max=310.0)
        _assert_ig_630(state, grid.lon, grid.lat, grid.radius, sigma8)
        assert float(jnp.max(jnp.abs(state.v.data))) == 0.0

    def test_latlon(self, sigma8):
        from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
            inertio_gravity_init_latlon,
        )
        grid = create_latlon_grid(24, 48)
        state = inertio_gravity_init_latlon(grid, sigma8)
        _assert_dry3d_state_grid_space(state, T_min=290.0, T_max=310.0)
        _assert_ig_630(state, grid.lon2d, grid.lat2d, grid.radius, sigma8)

    def test_mpas(self, sigma8):
        from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
            inertio_gravity_init_mpas,
        )
        mesh = create_voronoi_mesh(4)
        state = inertio_gravity_init_mpas(mesh, sigma8)
        _assert_dry3d_state_mpas(state, T_min=290.0, T_max=310.0)
        _assert_ig_630(state, mesh.lonCell, mesh.latCell, mesh.radius,
                       sigma8)

    def test_spectral(self, sigma8):
        import numpy as np
        from legoesm.grids.gaussian import sh_analysis, sh_analysis_3d
        from tests.test_cases.dcmip2008.inertio_gravity_3_2 import (
            inertio_gravity_init_spectral,
        )
        grid = create_gaussian_grid(21)
        state = inertio_gravity_init_spectral(grid, sigma8)
        _assert_dry3d_state_spectral(state)
        # fp64 precondition for the tight projection tolerance below
        assert grid.lat.dtype == jnp.float64
        assert state.T_hat.data.dtype == jnp.complex128
        # p_s = 1000 hPa (Eq. 87 with u0 = 0): ln p_s projects onto the
        # mean only
        lnps_ref = np.asarray(sh_analysis(
            grid, jnp.full((grid.n_lat, grid.n_lon), np.log(constants.p_ref))))
        np.testing.assert_allclose(np.asarray(state.lnps_hat.data), lnps_ref,
                                   rtol=0, atol=1e-12 * np.abs(lnps_ref).max())
        # at rest: no vorticity / divergence
        assert float(jnp.max(jnp.abs(state.vor_hat.data))) == 0.0
        assert float(jnp.max(jnp.abs(state.div_hat.data))) == 0.0
        # T_hat = projection of the [6-3-0] T written out independently
        lat2 = np.broadcast_to(np.asarray(grid.lat)[:, None],
                               (grid.n_lat, grid.n_lon))
        lon2 = np.broadcast_to(np.asarray(grid.lon2d),
                               (grid.n_lat, grid.n_lon))
        a = float(grid.radius)
        r = a * np.arccos(np.clip(np.sin(np.pi / 4) * np.sin(lat2)
                                  + np.cos(np.pi / 4) * np.cos(lat2)
                                  * np.cos(lon2 - np.pi), -1, 1))
        s = np.where(r < a / 3, 0.5 * (1 + np.cos(3 * np.pi * r / a)), 0.0)
        p = (np.asarray(sigma8.A_full) * sigma8.p_ref
             + np.asarray(sigma8.B_full) * constants.p_ref)
        z = constants.R_d * 300.0 / constants.g * np.log(constants.p_ref / p)
        T = 300.0 + (10.0 * s[..., None] * np.sin(2 * np.pi * z / 20000.0)
                     * (p / constants.p_ref) ** constants.kappa)
        ref = np.asarray(sh_analysis_3d(grid, jnp.asarray(T)))
        np.testing.assert_allclose(np.asarray(state.T_hat.data), ref, rtol=0,
                                   atol=1e-10 * np.abs(ref).max())


# ---------------------------------------------------------------------------
# DCMIP 2008 §5-0 — mountain-induced Rossby wave
# ---------------------------------------------------------------------------


def _mr_eq83_fields(lon, lat, radius, omega=constants.Omega):
    """``(Phi_s, p_s)`` written out from the reference:
    p_s = p_sp exp[-(a N^2 u0/(2 g^2 kappa))(u0/a + 2 Omega)(sin^2 phi - 1)
    - N^2 Phi_s/(g^2 kappa)], N^2 = g^2/(c_p T0) (Eq. 80), Phi_s from
    Eqs. (81)-(82) with h0 = 2000 m, d = 1500 km, centre (pi/2, pi/6)."""
    import numpy as np
    g, a, t0, u0 = constants.g, float(radius), 288.0, 20.0
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    r = a * np.arccos(np.clip(
        np.sin(np.pi / 6) * np.sin(lat)
        + np.cos(np.pi / 6) * np.cos(lat) * np.cos(lon - np.pi / 2), -1, 1))
    phis = g * 2000.0 * np.exp(-(r / 1.5e6) ** 2)
    n2 = g ** 2 / (constants.c_pd * t0)
    k = constants.kappa
    ps = 93000.0 * np.exp(
        -(a * n2 * u0 / (2 * g ** 2 * k)) * (u0 / a + 2 * omega)
        * (np.sin(lat) ** 2 - 1) - n2 / (g ** 2 * k) * phis)
    return phis, ps


def _assert_mr_eq83(state, lon, lat, radius, omega=constants.Omega):
    """Eq. (83) and Eqs. (81)-(82) at the builder's own points, T = 288 K."""
    import numpy as np
    phis, ps = _mr_eq83_fields(lon, lat, radius, omega)
    # the builders may run at the fp32 policy dtype: fp32 great-circle
    # distances put MEASURED rel errors of 3.3e-6 on Phi_s (cube C8, MPAS)
    np.testing.assert_allclose(np.asarray(state.phis.data), phis,
                               rtol=1e-5, atol=1e-2)
    np.testing.assert_allclose(np.asarray(state.p_s.data), ps, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(state.T.data), 288.0, rtol=1e-6)


# Eq. (83): p_s up to ~1043 hPa at the equator, ~800 hPa on the peak.
_MR_BOUNDS = dict(T_min=280.0, T_max=295.0, p_s_max=104500.0)


class TestMountainRossby50:
    def test_cubed_sphere(self, sigma8):
        from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
            mountain_rossby_init,
        )
        grid = create_cubed_sphere(8)
        state = mountain_rossby_init(grid, sigma8)
        _assert_dry3d_state_grid_space(state, **_MR_BOUNDS)
        _assert_mr_eq83(state, grid.lon, grid.lat, grid.radius)

    def test_latlon(self, sigma8):
        import numpy as np
        from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
            mountain_rossby_init_latlon,
        )
        grid = create_latlon_grid(24, 48)
        state = mountain_rossby_init_latlon(grid, sigma8)
        _assert_dry3d_state_grid_space(state, **_MR_BOUNDS)
        _assert_mr_eq83(state, grid.lon2d, grid.lat2d, grid.radius)
        # the balance uses the grid's own rotation rate
        fast = create_latlon_grid(24, 48, omega=2.0 * constants.Omega)
        _assert_mr_eq83(mountain_rossby_init_latlon(fast, sigma8),
                        fast.lon2d, fast.lat2d, fast.radius,
                        omega=2.0 * constants.Omega)
        # the peak sits at 90E / 30N (7.5 deg cells)
        i = np.unravel_index(np.argmax(np.asarray(state.phis.data)),
                             state.phis.data.shape)
        assert abs(np.degrees(float(grid.lon2d[i])) - 90.0) < 7.5
        assert abs(np.degrees(float(grid.lat2d[i])) - 30.0) < 7.5

    def test_mpas(self, sigma8):
        from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
            mountain_rossby_init_mpas,
        )
        mesh = create_voronoi_mesh(4)
        state = mountain_rossby_init_mpas(mesh, sigma8)
        _assert_dry3d_state_mpas(state, T_min=280.0, T_max=295.0)
        _assert_mr_eq83(state, mesh.lonCell, mesh.latCell, mesh.radius)

    def test_spectral(self, sigma8):
        from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
            mountain_rossby_init_spectral,
        )
        grid = create_gaussian_grid(21)
        state = mountain_rossby_init_spectral(grid, sigma8)
        _assert_dry3d_state_spectral(state)
        # spectral projections of Eqs. (81)-(83) evaluated independently on
        # the Gaussian grid (fp64 lane): pins the centre and the zonal term
        import numpy as np
        from legoesm.grids.gaussian import sh_analysis
        lat2 = jnp.broadcast_to(grid.lat[:, None], (grid.n_lat, grid.n_lon))
        lon2 = jnp.broadcast_to(grid.lon2d, (grid.n_lat, grid.n_lon))
        phis, ps = _mr_eq83_fields(lon2, lat2, grid.radius)
        for got, ref in ((state.phis_hat.data, sh_analysis(grid, jnp.asarray(phis))),
                         (state.lnps_hat.data, sh_analysis(grid, jnp.log(jnp.asarray(ps))))):
            got, ref = np.asarray(got), np.asarray(ref)
            np.testing.assert_allclose(got, ref, rtol=0,
                                       atol=1e-10 * np.abs(ref).max())

    def test_surface_pressure_matches_reference(self):
        """Eq. (83): p_s = p_sp = 930 hPa at the South Pole, and the area
        mean the reference quotes (~1001.456 hPa, text below Eq. 83).  The
        old IC (1000 hPa exp(-Phi_s/(R_d T0))) has mean 996.93 hPa."""
        import numpy as np
        from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
            P_SP, _phis, eq83_zonal_surface_pressure,
        )
        x, w = np.polynomial.legendre.leggauss(400)
        lat = jnp.asarray(np.arcsin(x))[:, None]
        lon = jnp.asarray(np.linspace(0.0, 2.0 * np.pi, 1600,
                                      endpoint=False))[None, :]
        p_s = np.asarray(
            eq83_zonal_surface_pressure(lat, constants.R_earth,
                                        omega=constants.Omega)
            * jnp.exp(-_phis(lon, lat, constants.R_earth)
                      / (constants.R_d * 288.0)))
        mean_hpa = float(p_s.mean(axis=1) @ w) / 2.0 / 100.0
        # legoesm constants give 1001.4518 (reference Appendix G constants
        # 1001.4555): 5e-3 hPa covers that constants difference only.
        assert abs(mean_hpa - 1001.456) < 5e-3, mean_hpa
        # full Eq. (83) at the South Pole: the latitude factor is 1 there,
        # and Phi_s ~ g h0 exp(-79) makes exp(-Phi_s/(R_d T0)) round to 1
        sp_lat, sp_lon = jnp.asarray(-np.pi / 2), jnp.asarray(0.0)
        sp = (eq83_zonal_surface_pressure(sp_lat, constants.R_earth,
                                          omega=constants.Omega)
              * jnp.exp(-_phis(sp_lon, sp_lat, constants.R_earth)
                        / (constants.R_d * 288.0)))
        np.testing.assert_allclose(float(sp), 93000.0, rtol=1e-12)
        assert P_SP == 93000.0

    def test_initial_divergence_tendency_is_balanced(self):
        """Eq. (83) puts the initial state in horizontal momentum balance
        (the mountain flow itself still has a nonzero p_s tendency): the
        spectral PE's initial divergence tendency vanishes to round-off.  On T21 / CAM6 L32:
        MEASURED rms 2.4e-19 1/s^2 for this IC, 5.4e-10 for the old one
        (1000 hPa, no zonal-flow balance term)."""
        import numpy as np
        from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
            SpectralPEConfig, spectral_pe_tendencies)
        from legoesm.grids.gaussian import sh_synthesis_3d
        from legoesm.grids.vertical import make_cam6_l32_levels
        from tests.test_cases.dcmip2008.mountain_rossby_5_0 import (
            mountain_rossby_init_spectral,
        )
        grid = create_gaussian_grid(21)
        coord = make_cam6_l32_levels()
        state = mountain_rossby_init_spectral(grid, coord)
        tend = spectral_pe_tendencies(state, grid, coord, SpectralPEConfig())
        ddiv = np.asarray(sh_synthesis_3d(grid, tend.div_hat.data))
        assert float(np.sqrt(np.mean(ddiv ** 2))) < 1e-15


# ---------------------------------------------------------------------------
# DCMIP 2008 3D Rossby-Haurwitz wave (reference test 4-0-0, Sect. 1.4)
# ---------------------------------------------------------------------------

# Lapse-rate columns reach ~100 K at the top of a ~2-4 hPa model lid
# (T0 (p/p_ref)^(Gamma R_d/g), Eq. 66) and p_s rises to ~1030 hPa (Eq. 70).
_RH_BOUNDS = dict(T_min=90.0, T_max=295.0, p_s_max=103500.0)


def _assert_rh_eq70_eq66(state, lon, lat, radius, coord,
                         omega=constants.Omega):
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
        radius=float(radius), omega=omega, gh0=0.0,
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
        # Eqs. (72)-(73) use the grid's own rotation rate
        fast = create_latlon_grid(24, 48, omega=2.0 * constants.Omega)
        _assert_rh_eq70_eq66(rossby_haurwitz_init_latlon(fast, sigma8),
                             fast.lon2d, fast.lat2d, fast.radius, sigma8,
                             omega=2.0 * constants.Omega)

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
                                     constants.R_earth, standard_hybrid_levels(8),
            omega=constants.Omega)
        p_s = np.asarray(p_s)
        mean_hpa = float(p_s.mean(axis=1) @ w) / 2.0 / 100.0
        # legoesm R_d/Omega give 1000.3748 (reference Appendix G constants
        # give 1000.3771): 5e-3 hPa covers that constants difference only.
        assert abs(mean_hpa - 1000.377) < 5e-3, mean_hpa
        p_pole, _ = rh_balanced_columns(
            jnp.asarray([0.3, 1.7]), jnp.asarray([np.pi / 2, -np.pi / 2]),
            constants.R_earth, standard_hybrid_levels(8),
            omega=constants.Omega)
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
            constants.R_earth, standard_hybrid_levels(8),
            omega=constants.Omega)
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
