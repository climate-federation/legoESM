"""Unit test for era5_to_cubedsphere_carry ``target_phis`` parameter.

Locks the VALIDATED no-op contract: ``target_phis`` is accepted for caller
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


def test_target_phis_is_a_noop():
    """A non-zero ``target_phis`` must NOT change the returned carry.

    The validated IC uses smoothed-ERA5 phis for the dynamics; ``target_phis``
    (the ETOPO field) is only carried for caller compatibility and reported via
    CMOR ``orog`` elsewhere.  A non-zero ETOPO field passed here must leave the
    carry bit-for-bit identical to the default.
    """
    grid = _small_cs_grid(n=4)
    sigma = _sigma(nlev=5)
    era5 = _synthetic_era5()

    carry_default = era5_to_cubedsphere_carry(era5, grid, sigma)

    # A realistic ~500 m mountain mask (≈4900 m2/s2) must be ignored.
    target_nonzero = jnp.full((6, 4, 4), 4900.0)
    carry_etopo = era5_to_cubedsphere_carry(
        era5, grid, sigma, target_phis=target_nonzero
    )

    # phis is the field the ETOPO override would have touched — it must be
    # unchanged (dynamics stay on the smoothed-ERA5 orography).
    np.testing.assert_array_equal(
        np.asarray(carry_default.phis),
        np.asarray(carry_etopo.phis),
    )
    # And the smoothed-ERA5 phis (flat synthetic terrain) stays near zero,
    # i.e. the 4900 m2/s2 target did not leak into the dynamics state.
    assert float(np.abs(np.asarray(carry_etopo.phis)).max()) < 1.0


def test_target_phis_none_unchanged():
    """Passing ``target_phis=None`` is identical to omitting it."""
    grid = _small_cs_grid(n=4)
    sigma = _sigma(nlev=5)
    era5 = _synthetic_era5()

    carry_a = era5_to_cubedsphere_carry(era5, grid, sigma)
    carry_b = era5_to_cubedsphere_carry(era5, grid, sigma, target_phis=None)
    np.testing.assert_array_equal(
        np.asarray(carry_a.phis),
        np.asarray(carry_b.phis),
    )
