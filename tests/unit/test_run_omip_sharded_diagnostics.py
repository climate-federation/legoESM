"""Diagnostic scalars must come off the SHARDED state, without a global gather.

At 1/12 degree the state is 7.4 GiB per 3-D field, so gathering it onto one
device every diagnostic sample is not a convenience cost: one arm died with
``RESOURCE_EXHAUSTED`` on exactly that allocation and another hung for its whole
wall-clock in the first block.  These tests pin that the scalars computed from a
lat-band sharded state equal the ones computed from the replicated state, and
that computing them never materialises a global field on one device.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star

_ROOT = Path(__file__).resolve().parents[2]
_N_LAT, _N_LON, _NLEV = 16, 32, 4


def _load_run_omip():
    spec = importlib.util.spec_from_file_location(
        "run_omip_diag_mod", _ROOT / "scripts" / "run" / "run_omip.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = ["run_omip.py"]
    spec.loader.exec_module(mod)
    return mod


def _state():
    """Rest state with a non-trivial velocity and land mask, so every scalar
    the diagnostics report (mean temperature, salinity, sea level, maximum
    speed, its location, barotropic kinetic-energy fraction) is non-degenerate.
    """
    import jax.numpy as jnp
    grid = create_latlon_grid(n_lat=_N_LAT, n_lon=_N_LON)
    z_coord = create_ocean_z_star(n_levels=_NLEV, H_max=4000.0)
    land = np.ones((_N_LAT, _N_LON))
    land[:2, :4] = 0.0
    st = rest_state_latlon_cgrid_ocean(grid, z_coord,
                                       land_mask_override=jnp.asarray(land))
    rng = np.random.default_rng(0)
    return st._replace(
        T=st.T.replace(data=jnp.asarray(
            rng.normal(10.0, 2.0, st.T.data.shape))),
        S=st.S.replace(data=jnp.asarray(
            rng.normal(35.0, 0.5, st.S.data.shape))),
        eta=st.eta.replace(data=jnp.asarray(
            rng.normal(0.0, 0.01, st.eta.data.shape))),
        u=st.u.replace(data=jnp.asarray(
            rng.normal(0.0, 0.2, st.u.data.shape))),
        # NON-zero data on the dead top v-face row (the sharded state drops
        # that row): the serial branch must mask it off, or the two layouts
        # disagree there.  Only the MASK is checked when a state is sharded.
        v=st.v.replace(data=jnp.asarray(
            rng.normal(0.0, 0.2, st.v.data.shape))),
    ), grid, z_coord


def _sharded(state):
    from legoesm.ocean.dynamics.sharded_ocean_step import shard_state_latlon
    from legoesm.parallel.mesh import create_latlon_mesh
    return shard_state_latlon(state, create_latlon_mesh(n_devices=4).mesh)


def test_scalars_match_between_sharded_and_replicated():
    run_omip = _load_run_omip()
    state, grid, z_coord = _state()
    ref = run_omip._extract_scalars(state, "tripole", grid, z_coord)
    got = run_omip._extract_scalars(_sharded(state), "tripole", grid, z_coord)
    assert set(ref) == set(got)
    for k in ref:
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-10, atol=1e-10,
                                   err_msg=f"scalar {k}")
    assert ref["max_speed"] > 0.0 and ref["P_bt"] > 0.0
    assert ref["j_maxu"] >= 0


def test_scalars_never_pull_a_state_field_to_the_host(monkeypatch):
    """Pulling a sharded field to the host is what the gather did, and under
    one process per GPU it is also what fails: the remote bands are not
    addressable.  Any host conversion of a state-sized array is the defect."""
    run_omip = _load_run_omip()
    state, grid, z_coord = _state()
    ss = _sharded(state)
    size_3d = int(np.prod(state.T.data.shape))
    pulled = []
    real_asarray = np.asarray

    def guard(a, *args, **kwargs):
        if isinstance(a, jax.Array) and a.ndim >= 2 and a.size >= size_3d:
            pulled.append(a.shape)
        return real_asarray(a, *args, **kwargs)

    monkeypatch.setattr(run_omip.np, "asarray", guard)
    run_omip._extract_scalars(ss, "tripole", grid, z_coord)
    assert not pulled, f"state-sized fields pulled to the host: {pulled}"
