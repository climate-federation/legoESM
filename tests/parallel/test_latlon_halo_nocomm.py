"""The lat-lon SPMD no-comm timing knob: it must break the answer, loudly.

``LEGOESM_LATLON_HALO_NOCOMM=1`` drops every lat-band halo collective so the
step's communication term can be measured as ``full - nocomm`` against an
otherwise identical program (the profiler on this stack does not record the
halo collectives).  It is a MEASUREMENT knob: it deliberately produces wrong
ghost rows, so the test asserts exactly that -- the padded field must CHANGE
when the knob is on.  If the knob were silently a no-op (e.g. the exchange
helper reverted to a plain ``ppermute``), this test fails.

Runs on host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from legoesm.parallel.latlon_spmd import (
    _resolve_halo_nocomm,
    pad_halo_latlon_band_spmd,
)

N_DEV = 4
N_LAT = 16
N_LON = 8
HALO = 1


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def _pad(field, mesh):
    return np.asarray(pad_halo_latlon_band_spmd(mesh, halo=HALO)(field))


def test_resolver_off_on_and_typo():
    assert _resolve_halo_nocomm("") is False
    assert _resolve_halo_nocomm("0") is False
    assert _resolve_halo_nocomm("1") is True
    for bad in ("true", "yes", "2", "01", " 1"):
        with pytest.raises(ValueError, match="LEGOESM_LATLON_HALO_NOCOMM"):
            _resolve_halo_nocomm(bad)


def test_nocomm_changes_the_interior_cut_ghost_rows(monkeypatch):
    """Off: band b's north ghost is band b+1's first row.  On: it is band b's
    OWN first row (the collective is gone), so the two arms must differ."""
    mesh = _mesh()
    rng = np.random.default_rng(4242)
    # Distinct per-row values so a wrong ghost row cannot coincide with a
    # right one.
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))

    monkeypatch.delenv("LEGOESM_LATLON_HALO_NOCOMM", raising=False)
    off = _pad(field, mesh)

    monkeypatch.setenv("LEGOESM_LATLON_HALO_NOCOMM", "1")
    on = _pad(field, mesh)

    assert off.shape == on.shape
    assert not np.allclose(off, on), (
        "no-comm arm is bit-identical to the real exchange -- the knob is "
        "inert, so any budget split measured with it is meaningless")

    # Name the exact rows: band 0 occupies rows [0, nl+2h) of the gathered
    # output; its north ghost is the last row of that block.
    nl = N_LAT // N_DEV
    block = nl + 2 * HALO
    band0_north_ghost_on = on[block - 1, HALO:HALO + N_LON]
    band0_own_south_edge = np.asarray(field)[0, :]
    band1_south_edge = np.asarray(field)[nl, :]
    np.testing.assert_allclose(band0_north_ghost_on, band0_own_south_edge,
                               rtol=0, atol=0)
    off_ghost = off[block - 1, HALO:HALO + N_LON]
    np.testing.assert_allclose(off_ghost, band1_south_edge, rtol=0, atol=0)


def test_default_environment_leaves_the_knob_off():
    assert os.environ.get("LEGOESM_LATLON_HALO_NOCOMM", "") in ("", "0")
