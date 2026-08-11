"""Mixed-boundary fused band pad (the lat-lon collective-collapse slice).

``make_latlon_band_mixed_pad_body`` exchanges each field's edge rows
ONCE and applies per-output boundary handling (wall constants vs pole
fold) — so the PPM transport's fold pad rides the entry exchange
instead of running its own (2 band exchanges/stage -> 1). These tests
pin BIT-identity against the legacy single-purpose pads and the
collective count.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

N_DEV, N_LAT, N_LON, NLEV = 4, 16, 12, 3


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def _fields():
    rng = np.random.default_rng(7)
    T = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
    u = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
    dp = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
    return T, u, dp


SPECS = (("wall", 0, 1, 0.0, 0.0), ("wall", 1, 1, 0.0, 0.0),
         ("wall", 2, 1, 0.0, 0.0), ("fold", 0, 2, False))


def _run_sharded(mesh, fields):
    from legoesm.parallel.latlon_spmd import make_latlon_band_mixed_pad_body
    body = make_latlon_band_mixed_pad_body(mesh, halo=2, outputs=SPECS)
    fn = jax.jit(jax.shard_map(
        body, mesh=mesh,
        in_specs=(P("lat"), P("lat"), P("lat")),
        out_specs=(P("lat"), P("lat"), P("lat"), P("lat")),
        check_vma=False))
    return [np.asarray(o) for o in fn(*fields)]


def test_mixed_pad_bit_identical_to_legacy(monkeypatch):
    """Per-band local pads == the corresponding rows of the serial
    single-purpose pads (wall == pad_with_pole_bc_lat, fold ==
    pad_halo_latlon_3d) — BIT-identical: the exchange is a copy.

    shard_map's out_specs CONCATENATE the per-band padded tiles, so
    the assembled axis 0 is (n_dev * (tile + 2*oh)); each band's block
    must equal serial_pad[b*tile : b*tile + tile + 2*oh]."""
    monkeypatch.setenv("LEGOESM_LATLON_SPMD_FUSED_HALO", "1")
    mesh = _mesh()
    T, u, dp = _fields()
    got = _run_sharded(mesh, (T, u, dp))

    from legoesm.grids.halo_latlon import (
        pad_halo_latlon_3d_local, pad_with_pole_bc_lat,
    )
    tile = N_LAT // N_DEV
    serial = [np.asarray(pad_with_pole_bc_lat(f, halo=1,
                                              south_value=0.0,
                                              north_value=0.0))
              for f in (T, u, dp)]
    serial.append(np.asarray(pad_halo_latlon_3d_local(T, halo=2)))
    halos = (1, 1, 1, 2)
    for k, (g, w, oh) in enumerate(zip(got, serial, halos)):
        per_band = g.reshape((N_DEV, tile + 2 * oh) + g.shape[1:])
        for b in range(N_DEV):
            np.testing.assert_array_equal(
                per_band[b], w[b * tile: b * tile + tile + 2 * oh],
                err_msg=f"output {k} band {b}")


def test_one_exchange_pair_total(monkeypatch):
    """The compiled mixed pad holds exactly TWO collective-permutes
    (north + south, one dtype group) — the fold output adds NONE."""
    mesh = _mesh()
    T, u, dp = _fields()
    from legoesm.parallel.latlon_spmd import make_latlon_band_mixed_pad_body
    body = make_latlon_band_mixed_pad_body(mesh, halo=2, outputs=SPECS)
    fn = jax.jit(jax.shard_map(
        body, mesh=mesh,
        in_specs=(P("lat"),) * 3, out_specs=(P("lat"),) * 4,
        check_vma=False))
    txt = fn.lower(T, u, dp).compile().as_text()
    n_cp = (txt.count("collective-permute-start(")
            + txt.count(" collective-permute("))
    assert n_cp == 2, f"expected 2 CPs (one pair), got {n_cp}"
