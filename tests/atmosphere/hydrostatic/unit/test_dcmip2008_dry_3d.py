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


def _assert_dry3d_state_grid_space(state, *, T_min=200.0, T_max=320.0):
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
    assert float(jnp.max(state.p_s.data)) <= constants.p_ref + 1e-6


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
# DCMIP 2008 §6-0 — small-amplitude 3D Rossby-Haurwitz wave
# ---------------------------------------------------------------------------


class TestRossbyHaurwitz60:
    def test_cubed_sphere(self, sigma8):
        from tests.test_cases.dcmip2008.rossby_haurwitz_6_0 import (
            rossby_haurwitz_init,
        )
        grid = create_cubed_sphere(8)
        state = rossby_haurwitz_init(grid, sigma8)
        _assert_dry3d_state_grid_space(state)
        # R-H wind magnitudes scale as R · K · cos(lat)^(R-1)... with the
        # standard parameters (K = 7.848e-6, wave-4) the surface |u|
        # peaks around 100 m/s at the equator.
        max_speed = float(jnp.max(jnp.sqrt(
            state.u.data ** 2 + state.v.data ** 2)))
        assert 50.0 < max_speed < 200.0

    def test_latlon(self, sigma8):
        from tests.test_cases.dcmip2008.rossby_haurwitz_6_0 import (
            rossby_haurwitz_init_latlon,
        )
        grid = create_latlon_grid(24, 48)
        state = rossby_haurwitz_init_latlon(grid, sigma8)
        _assert_dry3d_state_grid_space(state)

    def test_mpas(self, sigma8):
        from tests.test_cases.dcmip2008.rossby_haurwitz_6_0 import (
            rossby_haurwitz_init_mpas,
        )
        mesh = create_voronoi_mesh(4)
        state = rossby_haurwitz_init_mpas(mesh, sigma8)
        _assert_dry3d_state_mpas(state)

    def test_spectral(self, sigma8):
        from tests.test_cases.dcmip2008.rossby_haurwitz_6_0 import (
            rossby_haurwitz_init_spectral,
        )
        grid = create_gaussian_grid(21)
        state = rossby_haurwitz_init_spectral(grid, sigma8)
        _assert_dry3d_state_spectral(state)


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
