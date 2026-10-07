"""Sanity tests for ``tests.test_cases._terrain_helpers``."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm import constants

from tests.test_cases._terrain_helpers import (
    dcmip_2_0_0_mountain,
    gaussian_mountain,
    great_circle_distance,
    surface_geopotential,
    williamson5_cone,
)


def test_great_circle_distance_zero_at_centre():
    lon_c, lat_c = 1.5, 0.5
    d = great_circle_distance(
        jnp.array([lon_c]), jnp.array([lat_c]),
        lon_c, lat_c, constants.R_earth,
    )
    assert float(d[0]) < 1.0  # less than 1 m


def test_great_circle_distance_antipodal():
    """Antipodal points: distance = π · radius."""
    R = constants.R_earth
    d = great_circle_distance(
        jnp.array([0.0]), jnp.array([0.0]),
        jnp.pi, 0.0, R,
    )
    assert float(d[0]) == pytest.approx(jnp.pi * R, rel=1e-6)


def test_williamson5_cone_peak_height():
    z = williamson5_cone(jnp.array([3.0 * jnp.pi / 2.0]),
                         jnp.array([jnp.pi / 6.0]),
                         h_0=2000.0)
    assert float(z[0]) == pytest.approx(2000.0, abs=1e-6)


def test_williamson5_cone_zero_outside_radius():
    """Far from the centre, the cone is zero."""
    z = williamson5_cone(jnp.array([0.0]), jnp.array([-jnp.pi / 4.0]),
                         h_0=2000.0)
    # The clipped lon/lat radius caps at R_m, giving z=0 at the boundary
    assert float(z[0]) == pytest.approx(0.0, abs=1e-6)


def test_dcmip_2_0_0_mountain_peak():
    """At the DCMIP centre (3π/2, 0), the mountain envelope is at maximum."""
    R = constants.R_earth
    z = dcmip_2_0_0_mountain(
        jnp.array([3.0 * jnp.pi / 2.0]),
        jnp.array([0.0]),
        R, h_0=2000.0,
    )
    # At d=0: envelope = h_0/2 * (1 + cos(0)) = h_0; ridges = cos(0)^2 = 1
    assert float(z[0]) == pytest.approx(2000.0, abs=1.0)


def test_dcmip_2_0_0_mountain_ridge_period_is_pi_over_16():
    """Ridges repeat every π/16 rad of arc (DCMIP 2012): a trough at ζ/2."""
    R = constants.R_earth
    z = dcmip_2_0_0_mountain(
        jnp.array([3.0 * jnp.pi / 2.0 + jnp.pi / 32.0, 3.0 * jnp.pi / 2.0 + jnp.pi / 16.0]),
        jnp.array([0.0, 0.0]),
        R, h_0=2000.0,
    )
    assert float(z[0]) == pytest.approx(0.0, abs=1e-6)
    envelope = 1000.0 * (1.0 + float(jnp.cos(jnp.pi * (jnp.pi / 16.0) / (3.0 * jnp.pi / 4.0))))
    assert float(z[1]) == pytest.approx(envelope, rel=1e-6)


def test_dcmip_2_0_0_mountain_envelope_reaches_3pi_over_4():
    """The envelope reaches 3π/4 rad: the ridge crest at π/4 (= 4ζ) stands at 1500 m."""
    R = constants.R_earth
    z = dcmip_2_0_0_mountain(
        jnp.array([3.0 * jnp.pi / 2.0 + jnp.pi / 4.0]),
        jnp.array([0.0]),
        R, h_0=2000.0,
    )
    assert float(z[0]) == pytest.approx(1500.0, rel=1e-6)


def test_dcmip_2_0_0_mountain_decays():
    """Away from the centre, height should be smaller than the peak."""
    R = constants.R_earth
    z = dcmip_2_0_0_mountain(
        jnp.array([0.0]), jnp.array([-jnp.pi / 3.0]),
        R, h_0=2000.0,
    )
    assert float(z[0]) < 2000.0


def test_gaussian_mountain_finite_and_in_range():
    R = constants.R_earth
    lons = jnp.linspace(0.0, 2.0 * jnp.pi, 16, endpoint=False)
    lats = jnp.linspace(-jnp.pi / 2.0, jnp.pi / 2.0, 16)
    lon2, lat2 = jnp.meshgrid(lons, lats, indexing="ij")
    z = gaussian_mountain(lon2, lat2, R, h_0=2000.0)
    assert bool(jnp.all(jnp.isfinite(z)))
    assert float(jnp.min(z)) >= 0.0
    assert float(jnp.max(z)) <= 2000.0 + 1e-6


def test_surface_geopotential_uses_constants():
    z = jnp.array([0.0, 1000.0, 2000.0])
    phi = surface_geopotential(z)
    assert float(phi[0]) == 0.0
    assert float(phi[2]) == pytest.approx(constants.g * 2000.0, rel=1e-12)
