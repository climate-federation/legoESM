"""Stage 2 of the atm latlon SPMD step: the 6-field C-grid state shard/gather.

``shard_state_atm_latlon`` lays the C-grid hydrostatic state onto the lat-band
mesh (cell/u leaves ``P("lat")``; the staggered ``v`` carried as
``v_lower=v[:n_lat]``); ``gather_state_atm_latlon`` is its inverse (re-append the
zero pole-wall v-face). Round-trip identity holds for a pole-walled v
(``v[-1]==0``), the state the model produces. Host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
)
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
    shard_state_atm_latlon, gather_state_atm_latlon)

N_DEV = 4
N_LAT = 16
N_LON = 8
NLEV = 4


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def _rand_state(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N_LAT + 1, N_LON, NLEV))
    v[0] = 0.0    # south pole wall
    v[-1] = 0.0   # north pole wall (the face dropped + re-appended)
    return CGridLatLonHydrostaticState(
        u=jnp.asarray(rng.standard_normal((N_LAT, N_LON + 1, NLEV))),
        v=jnp.asarray(v),
        T=jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)) + 250.0),
        p_s=jnp.asarray(rng.standard_normal((N_LAT, N_LON)) + 1.0e5),
        phis=jnp.asarray(rng.standard_normal((N_LAT, N_LON))),
        tracers={"q": jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))},
    )


def test_shard_gather_round_trip_identity():
    mesh = _mesh()
    s = _rand_state()
    rt = gather_state_atm_latlon(shard_state_atm_latlon(s, mesh), mesh)
    for fld in ("u", "T", "p_s", "phis"):
        a, b = np.asarray(getattr(rt, fld)), np.asarray(getattr(s, fld))
        assert a.shape == b.shape, fld
        assert np.array_equal(a, b), f"{fld} not bit-identical round-trip"
    # v rebuilt to (n_lat+1, ...); identical because the top face is the 0 wall.
    assert np.asarray(rt.v).shape == (N_LAT + 1, N_LON, NLEV)
    assert np.array_equal(np.asarray(rt.v), np.asarray(s.v)), "v round-trip"
    assert np.array_equal(np.asarray(rt.tracers["q"]),
                          np.asarray(s.tracers["q"])), "tracer round-trip"


def test_shard_drops_v_top_row():
    """shard carries v as v_lower=v[:n_lat] (n_lat rows, divisible by N_DEV)."""
    mesh = _mesh()
    s = _rand_state(seed=1)
    sh = shard_state_atm_latlon(s, mesh)
    assert sh.v.shape == (N_LAT, N_LON, NLEV)
    assert np.array_equal(np.asarray(sh.v), np.asarray(s.v)[:N_LAT])
    # cell/u leaves keep their shapes (only their sharding changed).
    assert sh.u.shape == s.u.shape
    assert sh.T.shape == s.T.shape
