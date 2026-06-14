"""Unit tests for create_z_star_from_thicknesses (NEMO-vertical matching).

Builds a z* coordinate from EXPLICIT reference layer thicknesses (e.g. NEMO
``e3t_1d``) so legoESM can reproduce another model's vertical grid exactly.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.ocean.vertical import (  # noqa: E402
    create_ocean_z_star,
    create_z_star_from_thicknesses,
)


def test_reproduces_explicit_thicknesses():
    dz = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
    z = create_z_star_from_thicknesses(dz)
    assert z.n_levels == dz.size
    np.testing.assert_allclose(z.H_max, dz.sum(), rtol=1e-12)
    # Recovered thicknesses match the input (to the bottom-snap precision).
    np.testing.assert_allclose(np.asarray(z.dz_ref), dz, rtol=1e-9, atol=1e-7)
    # Interfaces: surface 0, monotonically deepening, bottom = -H_max.
    zh = np.asarray(z.z_half_ref)
    assert zh[0] == 0.0
    np.testing.assert_allclose(zh[-1], -dz.sum(), rtol=1e-12)
    assert np.all(np.diff(zh) < 0)                       # strictly deepening
    # Centres are midway between interfaces.
    zc = np.asarray(z.z_full_ref)
    np.testing.assert_allclose(zc, 0.5 * (zh[:-1] + zh[1:]), atol=1e-9)


def test_column_sum_identity_holds():
    # sum(dz_ref) == H_max to bit precision (Hallberg-Adcroft partial-cell req).
    dz = np.linspace(1.0, 204.0, 75)                     # NEMO-like growth
    z = create_z_star_from_thicknesses(dz)
    np.testing.assert_allclose(float(np.sum(np.asarray(z.dz_ref))), z.H_max,
                               rtol=0, atol=1e-9)


def test_centres_match_cumulative_depth():
    # Cell centre depth == half-thickness below the cell's top interface.
    dz = np.array([10.0, 30.0, 60.0, 100.0])
    z = create_z_star_from_thicknesses(dz)
    top_iface = np.concatenate([[0.0], np.cumsum(dz)[:-1]])
    expected_centre_depth = top_iface + 0.5 * dz
    np.testing.assert_allclose(-np.asarray(z.z_full_ref), expected_centre_depth,
                               atol=1e-9)


def test_finer_surface_than_default_tanh():
    # The whole point: NEMO-like thicknesses give a MUCH finer surface layer
    # than the default 20-level tanh stretch over the same depth.
    dz_nemo = np.linspace(1.0, 204.0, 75)
    z_nemo = create_z_star_from_thicknesses(dz_nemo)
    z_def = create_ocean_z_star(n_levels=20, H_max=float(dz_nemo.sum()))
    assert float(np.asarray(z_nemo.dz_ref)[0]) < float(np.asarray(z_def.dz_ref)[0])
    assert z_nemo.n_levels > z_def.n_levels


@pytest.mark.parametrize("bad", [
    np.array([5.0]),                       # < 2 levels
    np.array([10.0, -5.0, 20.0]),          # negative thickness
    np.zeros((3, 2)),                      # not 1-D (zeros)
    np.ones((4, 3)) * 10.0,                # not 1-D but all POSITIVE: the ndim
                                           # check must reject it BEFORE ravel
                                           # (else it silently flattens). codex.
])
def test_rejects_bad_thicknesses(bad):
    with pytest.raises(ValueError):
        create_z_star_from_thicknesses(bad)
