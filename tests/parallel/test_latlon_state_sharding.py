"""shard_state must shard a lat-lon state on the LAT axis, not fall
through to replicated under the cubed-sphere default (codex P1,
2026-06-13: the driver called shard_state without grid_type, so a
single-node multi-GPU lat-lon run replicated the state — wrong layout
vs the grid-aware tracer shard_pytree).

Needs >1 host device: XLA_FLAGS=--xla_force_host_platform_device_count=4.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from legoesm.parallel.mesh import create_latlon_mesh
from legoesm.parallel.sharded_dynamics import shard_state


def _latlon_cfg():
    if len(jax.devices()) < 2:
        pytest.skip("needs --xla_force_host_platform_device_count>=2")
    return create_latlon_mesh(n_devices=len(jax.devices()))


class _S:  # minimal state-like pytree (leaves are jax arrays)
    def __init__(self, n_lat, n_lon, nlev):
        self.T = jnp.ones((n_lat, n_lon, nlev))
        self.eta = jnp.ones((n_lat, n_lon))

    def tree_flatten(self):
        return (self.T, self.eta), None

    @classmethod
    def tree_unflatten(cls, aux, kids):
        o = object.__new__(cls)
        o.T, o.eta = kids
        return o


jax.tree_util.register_pytree_node(
    _S, _S.tree_flatten, _S.tree_unflatten)


def test_latlon_state_shards_on_lat_axis():
    cfg = _latlon_cfg()
    state = _S(16, 32, 4)
    sharded = shard_state(state, cfg, grid_type="latlon")
    # 2D+ leaves shard on the lat (axis-0) mesh dim.
    for leaf in (sharded.T, sharded.eta):
        spec = leaf.sharding.spec
        assert spec[0] == "lat", (
            f"lat-lon leaf not lat-sharded: spec={spec}")


def test_cubed_default_would_replicate_latlon_state():
    # The bug the fix avoids: the cubed-sphere default sees no leading
    # dim of 6, so it replicates the lat-lon state instead of sharding.
    cfg = _latlon_cfg()
    state = _S(16, 32, 4)
    bugged = shard_state(state, cfg)  # grid_type defaults to cubed_sphere
    spec = bugged.T.sharding.spec
    assert all(ax is None for ax in spec), (
        f"expected replicated (the bug); got {spec} — if this now "
        f"shards, the default changed and the driver fix is moot")
