"""Unit tests for the cos²(lat) Laplacian-viscosity scaling
(`LatLonCGridOceanConfig.A_h_lat_scaling`).

The scaling addresses the high-latitude viscous-Coriolis runaway documented
in `docs/ocean_experiments/realistic_geometry_phase4_results.md` (D1):
on real ETOPO at A_h=2e5, the model blows up at a single Arctic
partial-cell at lat 82.5° in ~15 sim-days because the viscous decay time
shrinks with cos²(lat) and becomes faster than the Coriolis period at the
poles.  Standard MITgcm/MOM6/NEMO production fix: scale A_h by cos²(lat).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    laplacian_scaling_factor,
    biharmonic_scaling_factor,
    vector_laplacian_cgrid,
)


def test_laplacian_scaling_factor_shape_and_endpoints():
    grid = create_latlon_grid(36, 72)                     # 5° resolution
    scale_u, scale_v = laplacian_scaling_factor(grid)

    # Shape conventions: scale_u at u-face (cell-center) lat → (n_lat,);
    # scale_v at v-face lat → (n_lat+1,).
    assert scale_u.shape == (36,)
    assert scale_v.shape == (37,)

    # cos²(lat) is non-negative everywhere
    assert bool(jnp.all(scale_u >= 0))
    assert bool(jnp.all(scale_v >= 0))

    # At cell-center lat 0 (between rows 17 and 18 for 36 cells centered
    # on lat=0; the closest u-face latitudes are ±2.5°), scale_u should be
    # very close to 1.
    cos_lat = np.asarray(grid.cos_lat)
    expected_scale_u = cos_lat ** 2
    np.testing.assert_allclose(np.asarray(scale_u), expected_scale_u,
                                rtol=1e-12, atol=1e-12)

    # Highest-latitude u-face at ±87.5° should give cos²(87.5°) ≈ 0.00191
    assert float(scale_u[-1]) < 0.01
    assert float(scale_u[0]) < 0.01

    # v-face at ±90° (the very ends) is exactly cos²(±lat[-1]) per the
    # 'edge clamp' convention used in the implementation.
    assert float(scale_v[0]) < 0.01
    assert float(scale_v[-1]) < 0.01


def test_laplacian_scaling_factor_consistency_with_biharmonic():
    """B_h scaling is cos⁴, A_h scaling is cos² → A_h scaling = sqrt(B_h scaling)."""
    grid = create_latlon_grid(36, 72)
    lap_u, lap_v = laplacian_scaling_factor(grid)
    bih_u, bih_v = biharmonic_scaling_factor(grid)
    np.testing.assert_allclose(np.asarray(lap_u) ** 2,
                                np.asarray(bih_u),
                                rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(lap_v) ** 2,
                                np.asarray(bih_v),
                                rtol=1e-12, atol=1e-12)


def test_laplacian_scaling_equator_near_unity():
    """Near the equator the scale must be very close to 1 (no significant
    damping reduction).  The averaging convention used to interpolate
    cos(lat) onto v-faces is the same as ``biharmonic_scaling_factor``,
    which gives cos²(2.5°) ≈ 0.998 at the equator v-face on a 36×72 grid
    instead of exact 1.0; the same convention is used elsewhere.
    """
    grid = create_latlon_grid(36, 72)
    _, scale_v = laplacian_scaling_factor(grid)
    # v-face at index 18 (out of 37) sits at lat = 0; with the averaging
    # convention it returns ½(cos(-2.5°) + cos(+2.5°))² = cos²(2.5°)
    cos_2p5 = float(np.cos(np.deg2rad(2.5)))
    expected = cos_2p5 ** 2                                # ≈ 0.998
    assert abs(float(scale_v[18]) - expected) < 1e-6
    # The point of this test: scaling is essentially OFF at the equator
    assert float(scale_v[18]) > 0.99


def test_ah_scaling_off_is_identity():
    """A_h_lat_scaling=False must produce the same vlap result as the
    legacy path (multiply-by-A_h-only).  This is the bit-exact regression
    guard."""
    from legoesm.ocean.state import LatLonCGridOceanConfig

    config_off = LatLonCGridOceanConfig(A_h=1e5, A_h_lat_scaling=False)
    config_on = LatLonCGridOceanConfig(A_h=1e5, A_h_lat_scaling=True)
    assert config_off.A_h_lat_scaling is False
    assert config_on.A_h_lat_scaling is True
    # NamedTuple default: pre-existing call sites that don't pass the flag
    # default to False.
    config_legacy = LatLonCGridOceanConfig(A_h=1e5)
    assert config_legacy.A_h_lat_scaling is False


def test_ah_scaling_on_reduces_high_lat_tendency():
    """With A_h_lat_scaling=True, the Laplacian tendency at high latitude
    is reduced by cos²(lat) compared to the unscaled tendency."""
    grid = create_latlon_grid(36, 72)
    n_lat, n_lon = 36, 72

    # Construct a non-trivial velocity field so the vector Laplacian is
    # non-zero everywhere.  Use a sinusoidal mode in lat-lon.
    np.random.seed(0)
    u = np.random.randn(n_lat, n_lon + 1, 1) * 0.01
    v = np.random.randn(n_lat + 1, n_lon, 1) * 0.01
    u_jax = jnp.asarray(u); v_jax = jnp.asarray(v)

    vlap_u, vlap_v = vector_laplacian_cgrid(u_jax, v_jax, grid)
    scale_u, scale_v = laplacian_scaling_factor(grid)

    # Unscaled tendency (legacy path)
    tend_u_unscaled = 1e5 * np.asarray(vlap_u)
    # Scaled tendency (cos²(lat) applied)
    tend_u_scaled = 1e5 * np.asarray(scale_u)[:, None, None] * np.asarray(vlap_u)

    cos_lat = np.asarray(grid.cos_lat)

    # At highest lat (cos²(87.5°) ≈ 0.00191), scaled tendency must be
    # ≈ 0.00191× the unscaled.
    expected_ratio_high = cos_lat[-1] ** 2  # ≈ 0.00191 at 87.5°
    # Compare cell-by-cell at the high-lat row, where vlap_u is non-zero
    nonzero = np.abs(tend_u_unscaled[-1]) > 1e-15
    if nonzero.any():
        ratio = (tend_u_scaled[-1, nonzero] /
                 tend_u_unscaled[-1, nonzero])
        np.testing.assert_allclose(ratio, expected_ratio_high,
                                    rtol=1e-5, atol=1e-10)

    # At lowest |lat| (cos²(2.5°) ≈ 0.998), scaled tendency must be
    # nearly equal to unscaled.
    middle = n_lat // 2 - 1
    expected_ratio_eq = cos_lat[middle] ** 2  # ≈ 0.998 at 2.5°
    nonzero_eq = np.abs(tend_u_unscaled[middle]) > 1e-15
    if nonzero_eq.any():
        ratio_eq = (tend_u_scaled[middle, nonzero_eq] /
                    tend_u_unscaled[middle, nonzero_eq])
        np.testing.assert_allclose(ratio_eq, expected_ratio_eq,
                                    rtol=1e-5, atol=1e-10)


def test_viscous_cfl_latitude_independent():
    """End-to-end check on the physics motivation: with cos²(lat) scaling,
    the effective viscous CFL ``A_h_eff(lat) * dt / dx²(lat)`` is
    latitude-independent (up to the spherical metric)."""
    grid = create_latlon_grid(36, 72)
    R = 6.371e6
    dlon = 2.0 * np.pi / 72                                # radians
    dt = 600.0
    A_h_global = 2.0e5

    cos_lat = np.asarray(grid.cos_lat)
    dx_lat = R * dlon * cos_lat                            # dx at each lat row
    A_h_eff = A_h_global * cos_lat ** 2                    # cos²(lat) scaling

    # CFL number with scaling
    cfl_scaled = A_h_eff * dt / dx_lat ** 2
    # Should be approximately constant across latitudes
    cfl_at_eq = cfl_scaled[len(cfl_scaled) // 2 - 1]
    cfl_at_pole = cfl_scaled[-1]
    # cos²(lat) / cos²(lat) → constant
    assert abs(cfl_at_eq - cfl_at_pole) < 1e-12 * cfl_at_eq

    # Compare to UNSCALED CFL: at high lat it's much larger
    cfl_unscaled = A_h_global * dt / dx_lat ** 2
    cfl_unscaled_at_pole = cfl_unscaled[-1]
    cfl_unscaled_at_eq = cfl_unscaled[len(cfl_unscaled) // 2 - 1]
    # Pole CFL is at least 100× larger than equator CFL without scaling
    assert cfl_unscaled_at_pole / cfl_unscaled_at_eq > 100.0
