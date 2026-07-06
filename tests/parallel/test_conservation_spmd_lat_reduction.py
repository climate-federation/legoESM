"""Stage 0 of the atm latlon SPMD step: the shared lat-band SPMD area-sum.

Under a single-process ``shard_map`` (``is_distributed()`` is False) each band
holds only a PARTIAL area-weighted sum. ``batch_global_area_sums`` must combine
them across the ``"lat"`` axis via ``jax.lax.psum`` (the new
``_spmd_lat_psum_or_none`` dispatch) to recover the GLOBAL sum — otherwise the
mass fixer divides a band-local numerator by the global area and produces an
O(N) wrong correction. This is the load-bearing conservation fix shared by the
ocean and atm lat-band SPMD steps. Host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm.core.conservation import (
    batch_global_area_sums, global_area_sum, _spmd_lat_psum_or_none)
from legoesm.parallel.latlon_spmd import (
    activate_latlon_spmd_halo, deactivate_latlon_spmd_halo)
from legoesm.parallel.shard_map_compat import shard_map

N_DEV = 4
N_LAT = 16
N_LON = 8


class _StubGrid:
    """Minimal grid: batch_global_area_sums reads only ``.area``."""
    def __init__(self, area):
        self.area = area


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def test_spmd_lat_psum_or_none_is_none_when_not_spmd():
    """Serial / MPI backends fall through (None) — the spmd branch is inert."""
    deactivate_latlon_spmd_halo()  # ensure not armed
    assert _spmd_lat_psum_or_none([jnp.ones(3)]) is None


def test_batch_global_area_sums_spmd_lat_matches_serial():
    mesh = _mesh()
    rng = np.random.default_rng(11)
    area = rng.uniform(0.5, 1.5, (N_LAT, N_LON))
    f1 = rng.standard_normal((N_LAT, N_LON))
    f2 = rng.standard_normal((N_LAT, N_LON))
    # Serial reference: the true global area-weighted sums over the full grid.
    ref1 = float(np.sum(f1 * area))
    ref2 = float(np.sum(f2 * area))

    isp = P("lat", None)
    a_sh = jax.device_put(jnp.asarray(area), NamedSharding(mesh, isp))
    f1_sh = jax.device_put(jnp.asarray(f1), NamedSharding(mesh, isp))
    f2_sh = jax.device_put(jnp.asarray(f2), NamedSharding(mesh, isp))

    activate_latlon_spmd_halo(mesh)  # backend -> 'spmd', mesh -> this 'lat' mesh
    try:
        @partial(shard_map, mesh=mesh,
                 in_specs=(isp, isp, isp), out_specs=P(), check_vma=False)
        def f(a_b, f1_b, f2_b):
            # Each band sees only its rows; batch_global_area_sums must psum.
            sums = batch_global_area_sums([f1_b, f2_b], _StubGrid(a_b))
            return jnp.stack([sums[0], sums[1]])

        out = np.asarray(f(a_sh, f1_sh, f2_sh))
    finally:
        deactivate_latlon_spmd_halo()

    assert abs(float(out[0]) - ref1) < 1e-9 * max(abs(ref1), 1.0), \
        f"global sum 1: {out[0]} vs {ref1}"
    assert abs(float(out[1]) - ref2) < 1e-9 * max(abs(ref2), 1.0), \
        f"global sum 2: {out[1]} vs {ref2}"


def test_batch_global_area_sums_serial_unchanged():
    """Without SPMD armed, the full-grid path is the plain local sum."""
    deactivate_latlon_spmd_halo()
    rng = np.random.default_rng(3)
    area = jnp.asarray(rng.uniform(0.5, 1.5, (N_LAT, N_LON)))
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    got = batch_global_area_sums([field], _StubGrid(area))[0]
    ref = float(np.sum(np.asarray(field) * np.asarray(area)))
    assert abs(float(got) - ref) < 1e-9 * max(abs(ref), 1.0)


def test_global_area_sum_spmd_lat_matches_serial():
    """The SINGLE-array global_area_sum must psum band-local partials across the
    'lat' axis too (the operator-split mass/moisture fixers reduce through it,
    not the batched variant). Without this the fixer numerator is a band-local
    partial while the denominator is the global area -> O(N) wrong correction."""
    mesh = _mesh()
    rng = np.random.default_rng(17)
    area = rng.uniform(0.5, 1.5, (N_LAT, N_LON))
    field = rng.standard_normal((N_LAT, N_LON))
    ref = float(np.sum(field * area))          # true global area-weighted sum

    isp = P("lat", None)
    a_sh = jax.device_put(jnp.asarray(area), NamedSharding(mesh, isp))
    f_sh = jax.device_put(jnp.asarray(field), NamedSharding(mesh, isp))

    activate_latlon_spmd_halo(mesh)
    try:
        @partial(shard_map, mesh=mesh,
                 in_specs=(isp, isp), out_specs=P(), check_vma=False)
        def f(a_b, f_b):
            # Each band sees only its rows; global_area_sum must psum "lat".
            return global_area_sum(f_b, _StubGrid(a_b))

        got = float(np.asarray(f(a_sh, f_sh)))
    finally:
        deactivate_latlon_spmd_halo()

    assert abs(got - ref) < 1e-9 * max(abs(ref), 1.0), \
        f"single-array global sum: {got} vs {ref}"


def test_global_area_sum_serial_unchanged():
    """Without SPMD armed, global_area_sum is the plain local area-weighted sum
    (default path byte-unchanged: no armed backend -> _spmd_lat_psum returns
    None)."""
    deactivate_latlon_spmd_halo()
    rng = np.random.default_rng(5)
    area = jnp.asarray(rng.uniform(0.5, 1.5, (N_LAT, N_LON)))
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    got = global_area_sum(field, _StubGrid(area))
    ref = float(np.sum(np.asarray(field) * np.asarray(area)))
    assert abs(float(got) - ref) < 1e-9 * max(abs(ref), 1.0)
