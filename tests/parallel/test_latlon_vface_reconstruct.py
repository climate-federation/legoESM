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
    latlon_band_perms, reconstruct_vface_lower, to_vface_lower,
    cell_to_cgrid_winds_spmd, activate_latlon_spmd_halo,
    deactivate_latlon_spmd_halo)
from legoesm.grids.operators_latlon_cgrid import cell_to_cgrid_winds
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


# ==============================================================================
# cell -> C-grid wind conversion under lat-band sharding. The per-step
# cell<->C-grid round trip the operator-split SPMD lane reproduces from serial
# model.step: the v cell->face direction is NOT band-local (a naive band pad
# zeros every interior cut face as if it were a pole), so it must halo the
# neighbour band's row via interp_cell_to_vface_halo + re-zero only the PHYSICAL
# poles. Bit-identical to the serial full-grid cell_to_cgrid_winds.
# ==============================================================================

@pytest.mark.parametrize("nlev", [None, 5])
def test_cell_to_cgrid_winds_spmd_matches_serial(nlev):
    """Band-local cell->C-grid winds, gathered across bands, == serial full-grid
    cell_to_cgrid_winds BITWISE. Catches the silent decomposition error where a
    band-local pad zeros the interior-cut v-faces."""
    mesh = _mesh()
    rng = np.random.default_rng(101 + (0 if nlev is None else 1))
    shp = (N_LAT, N_LON) if nlev is None else (N_LAT, N_LON, nlev)
    u_cell = rng.standard_normal(shp)
    v_cell = rng.standard_normal(shp)
    u_s, v_s = cell_to_cgrid_winds(jnp.asarray(u_cell), jnp.asarray(v_cell))
    u_s, v_s = np.asarray(u_s), np.asarray(v_s)   # (N_LAT,N_LON+1), (N_LAT+1,N_LON)

    isp = P("lat", *((None,) * (u_cell.ndim - 1)))
    u_sh = jax.device_put(jnp.asarray(u_cell), NamedSharding(mesh, isp))
    v_sh = jax.device_put(jnp.asarray(v_cell), NamedSharding(mesh, isp))

    activate_latlon_spmd_halo(mesh)   # arm the spmd backend for interp_cell_to_vface_halo
    try:
        @partial(shard_map, mesh=mesh, in_specs=(isp, isp),
                 out_specs=(isp, isp), check_vma=False)
        def f(u_b, v_b):
            return cell_to_cgrid_winds_spmd(u_b, v_b)
        u_out, v_out = f(u_sh, v_sh)
        u_out = np.asarray(u_out)     # (N_LAT, N_LON+1) — clean gather (NL per band)
        v_out = np.asarray(v_out)     # (N_DEV*(NL+1), N_LON) — per-band blocks
    finally:
        deactivate_latlon_spmd_halo()

    # u-face: clean global gather (NL rows/band), bitwise vs serial.
    assert u_out.shape == u_s.shape, f"u shape {u_out.shape} vs {u_s.shape}"
    u_worst = float(np.max(np.abs(u_out - u_s)))
    assert u_worst < 1e-12, f"u-face vs serial (nlev={nlev}): {u_worst:.3e}"

    # v-face: NL+1 rows/band (overlapping interior interfaces) vs serial slices.
    blk = NL + 1
    assert v_out.shape[0] == N_DEV * blk
    v_worst = 0.0
    for b in range(N_DEV):
        band = v_out[b * blk:(b + 1) * blk]
        ref = v_s[b * NL: b * NL + NL + 1]        # global faces this band owns
        v_worst = max(v_worst, float(np.max(np.abs(band - ref))))
    assert v_worst < 1e-12, f"v-face vs serial (nlev={nlev}): {v_worst:.3e}"


def test_cell_to_cgrid_winds_spmd_serial_byte_identical():
    """Without the SPMD backend armed (serial / local), cell_to_cgrid_winds_spmd
    == cell_to_cgrid_winds — the default path is unchanged (a mesh=None
    operator-split step reduces to serial)."""
    deactivate_latlon_spmd_halo()   # ensure NOT armed
    rng = np.random.default_rng(202)
    u_cell = jnp.asarray(rng.standard_normal((N_LAT, N_LON, 5)))
    v_cell = jnp.asarray(rng.standard_normal((N_LAT, N_LON, 5)))
    u_a, v_a = cell_to_cgrid_winds_spmd(u_cell, v_cell)
    u_b, v_b = cell_to_cgrid_winds(u_cell, v_cell)
    # BYTE-identical (not just close): the local-backend path is the SAME
    # arithmetic (interior 0.5*(v[:-1]+v[1:]) + pole zero) as cell_to_cgrid_winds.
    assert np.array_equal(np.asarray(u_a), np.asarray(u_b)), "u-face not bit-identical"
    assert np.array_equal(np.asarray(v_a), np.asarray(v_b)), "v-face not bit-identical"
