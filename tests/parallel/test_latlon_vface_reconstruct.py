"""Stage 1 of the atm latlon SPMD step: the v-face round-trip array helpers.

``reconstruct_vface_lower`` rebuilds the (n_lat+1) staggered v-faces from the
(n_lat)-row ``v_lower`` inside a shard_map over ``"lat"`` (each band's north
boundary face = the next band's ``v_lower[0]``, lifted via ppermute; the
north-most band gets the pole-wall zero). These are the SHARED array cores
(factored from the ocean step's closures) used by BOTH the ocean and atm
lat-band SPMD steps, so the v-stagger numerics live in one place. Runs on host
CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm.parallel.latlon_spmd import (
    latlon_band_perms, reconstruct_vface_lower, to_vface_lower)
from legoesm.parallel.shard_map_compat import shard_map

N_DEV = 4
N_LAT = 16
NL = N_LAT // N_DEV
N_LON = 8


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    from jax.sharding import Mesh
    return jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


@pytest.mark.parametrize("nlev", [None, 5])
def test_reconstruct_vface_matches_serial(nlev):
    mesh = _mesh()
    perm_north, _ = latlon_band_perms(N_DEV)
    rng = np.random.default_rng(7 + (0 if nlev is None else 1))
    shp = (N_LAT + 1, N_LON) if nlev is None else (N_LAT + 1, N_LON, nlev)
    v_full = rng.standard_normal(shp)
    v_full[0] = 0.0       # south pole wall (v == 0 at the pole face)
    v_full[-1] = 0.0      # north pole wall
    v_lower = jnp.asarray(v_full[:N_LAT])  # n_lat rows, divisible by N_DEV

    isp = P("lat", *((None,) * (v_lower.ndim - 1)))
    vl_sh = jax.device_put(v_lower, NamedSharding(mesh, isp))

    @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp, check_vma=False)
    def f(vl):
        return reconstruct_vface_lower(vl, "lat", perm_north)

    out = np.asarray(f(vl_sh))            # (N_DEV*(NL+1), N_LON[,nlev])
    blk = NL + 1
    assert out.shape[0] == N_DEV * blk
    worst = 0.0
    for b in range(N_DEV):
        band = out[b * blk:(b + 1) * blk]
        # band b's full faces are global v[b*NL : b*NL + NL + 1]; the north-most
        # band's top face is the pole wall (0 == v_full[-1]).
        ref = v_full[b * NL: b * NL + NL + 1]
        worst = max(worst, float(np.max(np.abs(band - ref))))
    assert worst < 1e-12, f"reconstruct vs serial (nlev={nlev}): {worst:.3e}"


@pytest.mark.parametrize("nlev", [None, 5])
def test_vface_round_trip_identity(nlev):
    """to_vface_lower(reconstruct(v_lower)) == v_lower per band (pole-walled v)."""
    mesh = _mesh()
    perm_north, _ = latlon_band_perms(N_DEV)
    rng = np.random.default_rng(42 + (0 if nlev is None else 1))
    shp = (N_LAT + 1, N_LON) if nlev is None else (N_LAT + 1, N_LON, nlev)
    v_full = rng.standard_normal(shp)
    v_full[0] = 0.0
    v_full[-1] = 0.0
    v_lower = jnp.asarray(v_full[:N_LAT])
    isp = P("lat", *((None,) * (v_lower.ndim - 1)))
    vl_sh = jax.device_put(v_lower, NamedSharding(mesh, isp))

    @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp, check_vma=False)
    def rt(vl):
        return to_vface_lower(reconstruct_vface_lower(vl, "lat", perm_north))

    out = np.asarray(rt(vl_sh))
    assert out.shape == tuple(v_lower.shape)
    assert float(np.max(np.abs(out - np.asarray(v_lower)))) < 1e-12
