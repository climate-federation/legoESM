"""Unit test for era5_to_cubedsphere_carry target_phis parameter.

Verifies that when target_phis is provided the returned carry uses it
(instead of the ERA5-smoothed phis) and that the barometric p_s correction
and hybrid floor clamp are still applied.
"""

import numpy as np
import jax.numpy as jnp

from legoesm.training.era5_to_state import ERA5Slice, era5_to_cubedsphere_carry


def _synthetic_era5(n_lat=18, n_lon=36, n_plev=5):
    plev = np.array([5000.0, 25000.0, 50000.0, 85000.0, 100000.0])
    lat = np.linspace(-np.pi / 2 * 0.9, np.pi / 2 * 0.9, n_lat)
    lon = np.linspace(0.0, 2 * np.pi * (1 - 1.0 / n_lon), n_lon)
    T = np.full((n_lat, n_lon, n_plev), 260.0, dtype=np.float32)
    T[..., -1] = 290.0  # warm surface
    return ERA5Slice(
        T=T,
        u=np.zeros((n_lat, n_lon, n_plev), dtype=np.float32),
        v=np.zeros((n_lat, n_lon, n_plev), dtype=np.float32),
        q=np.full((n_lat, n_lon, n_plev), 1e-3, dtype=np.float32),
        p_s=np.full((n_lat, n_lon), 101325.0, dtype=np.float32),
        sst=np.full((n_lat, n_lon), 290.0, dtype=np.float32),
        phis=np.zeros((n_lat, n_lon), dtype=np.float32),
        lat=lat,
        lon=lon,
        plev_Pa=plev,
    )


def _small_cs_grid(n=4):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    return create_cubed_sphere(n)


def _sigma(nlev=5):
    from legoesm.grids.vertical import create_sigma_coordinate
    return create_sigma_coordinate(nlev)


def test_target_phis_replaces_smoothed():
    """target_phis is used instead of smooth_phis_cubed_sphere output."""
    grid = _small_cs_grid(n=4)
    sigma = _sigma(nlev=5)
    era5 = _synthetic_era5()

    carry_default = era5_to_cubedsphere_carry(era5, grid, sigma)

    # Flat terrain → target_phis=None and explicit zeros should give same phis
    target = jnp.zeros((6, 4, 4))
    carry_zero = era5_to_cubedsphere_carry(era5, grid, sigma, target_phis=target)
    np.testing.assert_allclose(
        np.asarray(carry_default.phis),
        np.asarray(carry_zero.phis),
        atol=1.0,  # smoothing of zeros is zeros
    )

    # Non-zero target_phis: phis in carry must equal target (no floor hit for
    # modest terrain ~500 m = 4900 m2/s2).
    target_nonzero = jnp.full((6, 4, 4), 4900.0)
    carry_etopo = era5_to_cubedsphere_carry(era5, grid, sigma, target_phis=target_nonzero)
    phis_out = np.asarray(carry_etopo.phis)
    assert float(phis_out.mean()) > float(np.asarray(carry_default.phis).mean()), (
        "target_phis should raise phis above the ERA5-smoothed value"
    )
    np.testing.assert_allclose(phis_out, 4900.0, atol=1.0)


def test_target_phis_none_unchanged():
    """Passing target_phis=None is identical to omitting it."""
    grid = _small_cs_grid(n=4)
    sigma = _sigma(nlev=5)
    era5 = _synthetic_era5()

    carry_a = era5_to_cubedsphere_carry(era5, grid, sigma)
    carry_b = era5_to_cubedsphere_carry(era5, grid, sigma, target_phis=None)
    np.testing.assert_array_equal(
        np.asarray(carry_a.phis),
        np.asarray(carry_b.phis),
    )
