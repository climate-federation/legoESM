"""Unit test for era5_to_cubedsphere_carry ``target_phis`` parameter.

Decision C (2026-10-02): ``target_phis`` is the grid's terrain product and
the dynamics start on it.  Earlier contract (no-op parameter):
``target_phis`` was accepted for caller
compatibility (the driver passes the model's ETOPO surface geopotential here)
but is **intentionally not applied** — the IC dynamics are initialised on the
*smoothed ERA5* orography, while the CMOR ``orog`` field separately reports the
ETOPO mountain mask (see ``era5_to_cubedsphere_carry`` docstring and
``model_driver._setup_diagnostics``).  Therefore the returned carry must be
*identical* whether ``target_phis`` is omitted, ``None``, or a non-zero ETOPO
field.  Placing the dynamics on ``target_phis`` is a deliberate,
revalidation-gated change that is NOT made here; if it is ever made, this test
must be updated together with the IC physics so the contract change is explicit.
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
        phis=np.full((n_lat, n_lon), 100.0, dtype=np.float32),  # not zero-filled
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


def test_target_phis_is_the_dynamics_terrain_and_moves_ps():
    """Decision C (2026-10-02): the carry's phis IS ``target_phis`` (the grid's
    terrain product) bit for bit, p_s moved barometrically from the raw ERA5
    phis to it (raw phis 100 m2/s2 here, so a 4900 m2/s2 target LOWERS p_s
    by exp(-4800/(R_d T_1000)) and a flat target RAISES it), and the
    target is required."""
    import pytest
    from legoesm import constants
    grid = _small_cs_grid(n=4)
    sigma = _sigma(nlev=5)
    era5 = _synthetic_era5()
    target = jnp.full((6, 4, 4), 4900.0)
    carry = era5_to_cubedsphere_carry(era5, grid, sigma, target_phis=target)
    np.testing.assert_array_equal(np.asarray(carry.phis), np.asarray(target))
    expect = 101325.0 * np.exp(-4800.0 / (constants.R_d * 290.0))
    np.testing.assert_allclose(np.asarray(carry.p_s), expect, rtol=1e-5)
    flat = era5_to_cubedsphere_carry(era5, grid, sigma, target_phis=jnp.zeros((6, 4, 4)))
    assert float(np.abs(np.asarray(flat.phis)).max()) == 0.0
    np.testing.assert_allclose(np.asarray(flat.p_s),
                               101325.0 * np.exp(100.0 / (constants.R_d * 290.0)), rtol=1e-5)
    with pytest.raises(TypeError):
        era5_to_cubedsphere_carry(era5, grid, sigma)
