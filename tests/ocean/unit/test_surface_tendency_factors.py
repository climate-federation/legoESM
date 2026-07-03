"""Unit test for the shared surface flux→tendency helper (#518 §3).

Pins `surface_tendency_factors` to the inline formula the four surface-forcing
schemes used, and checks the wet/land masking + the explicit reference-density
source (so flux_feedback's cfg-pinned rho_0/c_sw and the others' eos constants
both flow through one kernel).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm.ocean.physics.surface_forcing._shared import (
    surface_tendency_factors,
    _DZ_FLOOR_M,
)


def _reference(is_ocean, dz_0, rho_0, c_sw, dz_floor):
    dz_safe = jnp.maximum(dz_0, dz_floor)
    inv_rho_dz = jnp.where(is_ocean, 1.0 / (rho_0 * dz_safe), 0.0)
    inv_rho_csw_dz = jnp.where(is_ocean, 1.0 / (rho_0 * c_sw * dz_safe), 0.0)
    return inv_rho_dz, inv_rho_csw_dz


def test_byte_identical_to_inline_formula():
    rng = np.random.default_rng(0)
    dz_0 = jnp.asarray(rng.random((6, 8)) * 50.0, dtype=jnp.float64)
    # mix wet + land cells
    is_ocean = dz_0 > 1.0
    for rho_0, c_sw in [(1025.0, 3994.0), (1035.0, 3850.0)]:  # eos vs cfg-pinned
        got = surface_tendency_factors(is_ocean, dz_0, rho_0, c_sw)
        ref = _reference(is_ocean, dz_0, rho_0, c_sw, _DZ_FLOOR_M)
        for g_, r_ in zip(got, ref):
            assert np.array_equal(np.asarray(g_), np.asarray(r_))


def test_land_cells_zeroed():
    dz_0 = jnp.array([[0.0, 5.0, 0.0, 10.0]], dtype=jnp.float64)
    is_ocean = dz_0 > 1.0
    inv_rho_dz, inv_rho_csw_dz = surface_tendency_factors(is_ocean, dz_0, 1025.0, 3994.0)
    land = ~np.asarray(is_ocean)
    assert np.all(np.asarray(inv_rho_dz)[land] == 0.0)
    assert np.all(np.asarray(inv_rho_csw_dz)[land] == 0.0)
    wet = np.asarray(is_ocean)
    assert np.all(np.asarray(inv_rho_dz)[wet] > 0.0)


def test_heat_factor_is_momentum_over_csw():
    """inv_rho_csw_dz == inv_rho_dz / c_sw on wet cells (consistency)."""
    dz_0 = jnp.array([[3.0, 12.0, 40.0]], dtype=jnp.float64)
    is_ocean = dz_0 > 1.0
    c_sw = 3994.0
    inv_rho_dz, inv_rho_csw_dz = surface_tendency_factors(is_ocean, dz_0, 1025.0, c_sw)
    assert np.allclose(np.asarray(inv_rho_csw_dz),
                       np.asarray(inv_rho_dz) / c_sw)
