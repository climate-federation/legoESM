"""SPMD equivalence gate for the lat-lon C-grid ocean step (multi-node OMIP).

This is the CORRECTNESS GATE for ``make_sharded_ocean_step`` (lat-band SPMD over
the ``"lat"`` axis): N steps under ``shard_map`` on 4 (CPU) devices must match
the single-device step to tolerance. It is the eORCA025 ¼° enabler (the ¼° grid
OOMs on one 32 GiB GPU; lat-band sharding fits it at N>=5).

KNOWN-INCOMPLETE (the crux, see memory ``omip-multinode-spmd-scope``): the C-grid
staggers fields — u/eta/T are ``n_lat`` rows but v is ``n_lat+1`` (coprime), so v
CANNOT share u's lat-band ``NamedSharding`` (49 % 4 != 0). The wrapper needs a
band-local v representation (each device owns ``n_lat/N`` cells + reconstructs its
bounding v-faces via ``make_latlon_band_pad_body``), not a uniform shard of the
raw ``(n_lat+1)`` array. Until that lands, this test SKIPS with a clear reason
(it is the spec the focused build targets, not a passing test yet).

Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      JAX_ENABLE_X64=1 pytest tests/parallel/test_latlon_ocean_spmd_step.py``
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel


def _perturbed_state(grid, z_coord):
    """Rest state + small u/v/T/eta perturbations so the step exercises every
    term (advection/Coriolis/PGF), not the trivial rest fixed point."""
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(0)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    nlev = z_coord.n_levels
    u = 0.02 * rng.standard_normal((n_lat, n_lon + 1, nlev))
    v = 0.02 * rng.standard_normal((n_lat + 1, n_lon, nlev))
    eta = 0.005 * rng.standard_normal((n_lat, n_lon))
    T = (5.0 + 15.0 * np.exp(np.linspace(0, -4, nlev))[None, None, :]
         + 0.05 * rng.standard_normal((n_lat, n_lon, nlev)))
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
        T=state.T.replace(data=jnp.asarray(T)))


def _have_sharded_step():
    try:
        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
            make_sharded_ocean_step,
        )
        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(jax.device_count() < 4,
                    reason="needs >=4 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
@pytest.mark.xfail(reason="make_sharded_ocean_step WIP: staggered v-field "
                          "(n_lat+1) band decomposition not yet implemented — "
                          "this is the focused-sprint GATE, see "
                          "omip-multinode-spmd-scope", strict=False)
def test_latlon_ocean_spmd_matches_single_device():
    from jax.sharding import NamedSharding, PartitionSpec as P
    from legoesm.parallel.mesh import create_latlon_mesh
    from legoesm.ocean.dynamics.sharded_ocean_step import make_sharded_ocean_step

    n_lat, n_lon, nlev = 48, 96, 10
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    cfg = LatLonCGridOceanConfig()
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = _perturbed_state(grid, z_coord)
    dt, n_steps = 600.0, 3

    # single-device reference
    s = state0
    for _ in range(n_steps):
        s = model.step(s, dt)

    # lat-band SPMD on 4 devices
    dev = create_latlon_mesh(n_devices=4)
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = jax.tree.map(
        lambda x: jax.device_put(
            x, NamedSharding(dev.mesh,
                             P("lat", *((None,) * (np.ndim(x) - 1))))),
        state0)
    for _ in range(n_steps):
        ss = step(ss, dt)

    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(s, nm).data)
        b = np.asarray(getattr(ss, nm).data)
        np.testing.assert_allclose(b, a, atol=1e-8, rtol=1e-6,
                                   err_msg=f"SPMD {nm} mismatch")
