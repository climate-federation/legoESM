"""Lat-band SPMD tripolar NORTH FOLD — bit-identity vs serial.

The tripole north fold (``pad_ns_scalar`` / ``pad_ns_vector_v`` /
``pad_ns_vector_u``) overwrites the north (last) lat-row with the fold-partner
row (``i``-reversal ``perm_T``/``perm_v`` + vector sign).  In serial / MPI this
is a Python ``if fold_is_local(grid)`` overwrite; under the lat-band SPMD
backend ONE ``shard_map`` trace runs on every band, so ``fold_is_local`` is
uniformly False (the slicer sets ``fold_j=-1`` on every band) and the seam is
selected DATA-dependently on the north band (``axis_index("lat")==N-1``) via
``north_fold_mask`` + ``apply_north_fold``.  This must be bit-identical to the
serial fold per band — the same proven methodology as
``test_latlon_spmd_halo.py``.

Runs on 4 host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``); the production target
is N GPUs but the shard_map / axis_index logic is device-agnostic.
"""
from __future__ import annotations

import types
from functools import partial

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

try:
    from jax import shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm.grids.latlon import FoldDescriptor
from legoesm.grids.operators_latlon_cgrid import pad_ns_scalar, pad_ns_vector_v
from legoesm.ocean.dynamics.latlon_cgrid_operators import pad_ns_vector_u
from legoesm.parallel.latlon_spmd import (
    activate_latlon_spmd_halo, deactivate_latlon_spmd_halo,
)

N_DEV = 4
N_LAT = 16
NL = N_LAT // N_DEV
N_LON = 8


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    from jax.sharding import Mesh
    return Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def _fold(fold_j):
    """Synthetic tripolar fold: i-reversal perms, vector sign -1.  ``fold_j>=0``
    = serial north-pole-touching rank (fold local); ``fold_j=-1`` = the
    band-localized descriptor every SPMD band carries (perms preserved)."""
    rev = jnp.asarray(np.arange(N_LON)[::-1].copy(), dtype=jnp.int32)
    return FoldDescriptor(
        is_active=True, fold_j=fold_j, cap_j=fold_j,
        perm_T=rev, perm_v=rev, vector_sign_u=-1.0, vector_sign_v=-1.0)


# pad-fn -> the serial sign/perm it applies to row[-1], for a no-fold sanity ref.
_PADS = {
    "scalar": (pad_ns_scalar, "perm_T", 1.0),
    "vector_v": (pad_ns_vector_v, "perm_v", -1.0),
    "vector_u": (pad_ns_vector_u, "perm_T", -1.0),
}


@pytest.mark.parametrize("kind", list(_PADS))
@pytest.mark.parametrize("nlev", [None, 5])
def test_north_fold_spmd_matches_serial(kind, nlev):
    mesh = _mesh()
    pad_fn, _perm, _sign = _PADS[kind]

    rng = np.random.default_rng(31 + len(kind) + (0 if nlev is None else 7))
    shp = (N_LAT, N_LON) if nlev is None else (N_LAT, N_LON, nlev)
    field = jnp.asarray(rng.standard_normal(shp))

    grid_serial = types.SimpleNamespace(fold=_fold(0))     # fold local (fold_j>=0)
    grid_band = types.SimpleNamespace(fold=_fold(-1))      # SPMD band (fold_j=-1)
    ref = np.asarray(pad_fn(field, grid_serial))           # serial north fold

    isp = P("lat", *((None,) * (field.ndim - 1)))
    field_sh = jax.device_put(field, NamedSharding(mesh, isp))
    activate_latlon_spmd_halo(mesh)
    try:
        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
                 check_vma=False)
        def _ex(tile):
            return pad_fn(tile, grid_band)
        out = np.asarray(_ex(field_sh))
    finally:
        deactivate_latlon_spmd_halo()

    blk = NL + 2
    assert out.shape[0] == N_DEV * blk
    worst = max(
        float(np.max(np.abs(out[b * blk:(b + 1) * blk]
                            - ref[b * NL: b * NL + NL + 2])))
        for b in range(N_DEV))
    assert worst < 1e-12, (
        f"north-fold SPMD vs serial ({kind}, nlev={nlev}): {worst:.3e}")

    # Non-vacuity: the north row IS the permuted fold partner, not a wall/0 or
    # the un-permuted row (perm reverses lon -> differs for this random field).
    fold_top = ref[-1]                       # serial north (fold) row
    wall_like = np.zeros_like(fold_top)
    assert np.max(np.abs(fold_top - wall_like)) > 1e-6, "fold row is all-zero wall"
    unpermuted = _sign * np.asarray(field)[-1]
    assert np.max(np.abs(fold_top - unpermuted)) > 1e-6, (
        "fold row equals the un-permuted row (perm not applied)")
