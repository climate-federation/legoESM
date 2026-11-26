"""Pole-row v estimate (``latlon_pole_cell_v``) is decomposition-independent.

A ONE-row pole band lacks the second real v-face locally and must fetch it
(and its latitude) through the lat halo; the result must equal serial.
4 lat bands x 1 row on a 4-row global grid: every band is one row, both pole
bands exercise the halo fetch.  Needs
``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="fp64: run with JAX_ENABLE_X64=1")


@pytest.mark.parametrize("n_lat,n_dev", [(4, 4), (8, 4), (2, 2)])  # 1/2/1 rows per band; (2,2): 2-row global grid
def test_thin_pole_band_matches_serial(n_lat, n_dev):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        latlon_pole_cell_v)
    from legoesm.parallel.latlon_spmd import (
        activate_latlon_spmd_halo, deactivate_latlon_spmd_halo)
    from legoesm.parallel.shard_map_compat import shard_map

    if len(jax.devices()) < n_dev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={n_dev}")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:n_dev]), ("lat",))
    g = create_latlon_grid(n_lat, 8)
    rng = np.random.default_rng(3)
    v = rng.standard_normal((n_lat + 1, 8, 2))
    v[0] = v[-1] = 0.0                                # pole walls
    v = jnp.asarray(v)
    serial = np.asarray(latlon_pole_cell_v(0.5 * (v[:-1] + v[1:]), v, g))

    n = n_lat // n_dev
    # per-band stacks: v-face arrays hold both shared end faces
    vb = jnp.stack([v[b * n:b * n + n + 1] for b in range(n_dev)])
    latb = jnp.stack([g.lat[b * n:b * n + n] for b in range(n_dev)])
    lvb = jnp.stack([g.lat_v[b * n:b * n + n + 1] for b in range(n_dev)])
    cvb = jnp.stack([g.cos_lat_v[b * n:b * n + n + 1] for b in range(n_dev)])

    def body(vl, la, lv, cv):
        gl = g._replace(n_lat=n, lat=la[0], lat_v=lv[0], cos_lat_v=cv[0])
        vv = vl[0]
        return latlon_pole_cell_v(0.5 * (vv[:-1] + vv[1:]), vv, gl)[None]

    spec = P("lat")
    fn = jax.jit(shard_map(body, mesh=mesh, in_specs=(spec,) * 4,
                           out_specs=spec, check_vma=False))
    put = lambda a: jax.device_put(a, NamedSharding(mesh, spec))
    activate_latlon_spmd_halo(mesh)
    try:
        out = np.asarray(fn(put(vb), put(latb), put(lvb), put(cvb)))
    finally:
        deactivate_latlon_spmd_halo()
    np.testing.assert_allclose(out.reshape(n_lat, 8, 2), serial,
                               rtol=1e-13, atol=1e-13)
    # reverse-mode THROUGH the halo fetch: on a one-row south pole band the
    # pole-row value is v1 + (v1 - v2) * w with v2 the neighbour band's second
    # real face, so d out[band 0, row 0] / d vb[band 1, row 1] = -w (exact).
    if n == 1 and n_dev > 2:   # (n_dev=2: second face is the other pole wall)
        sm = shard_map(body, mesh=mesh, in_specs=(spec,) * 4, out_specs=spec,
                       check_vma=False)
        gfn = jax.jit(jax.grad(lambda a: sm(a, put(latb), put(lvb), put(cvb))[0, 0, 3, 1]))
        activate_latlon_spmd_halo(mesh)
        try:
            gr = np.asarray(gfn(put(vb)))
        finally:
            deactivate_latlon_spmd_halo()
        w = float((g.lat_v[1] - g.lat[0]) / (g.lat_v[2] - g.lat_v[1]))
        assert np.all(np.isfinite(gr))
        np.testing.assert_allclose(gr[1, 1, 3, 1], -w, rtol=1e-6)  # fp32 grid lat
    # non-vacuous: the pole rows really differ from the plain mean
    plain = np.asarray(0.5 * (v[:-1] + v[1:]))
    assert np.abs(serial[0] - plain[0]).max() > 1e-3
