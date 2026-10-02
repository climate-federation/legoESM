"""Bitwise parity of the row-index-select pole clamps with the
``.at[row].set`` / concatenate-then-where originals they replaced (the
originals cost a full copy pass each on the Derecho CPU profile of
2026-09-26; the selects fuse into the producer)."""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.parallel import latlon_spmd


def _apply(*args, **kwargs):
    # resolved per call so a monkeypatch of the module is seen
    return latlon_spmd.apply_pole_end_masks(*args, **kwargs)


def _field(shape, dtype, seed=0):
    x = jax.random.normal(jax.random.PRNGKey(seed), shape, jnp.float32)
    # -0.0, inf and nan must survive a select bit-for-bit
    x = x.at[0, 1].set(-0.0).at[1, 2].set(jnp.inf).at[2, 3].set(jnp.nan)
    return x.astype(dtype)


def _bits_equal(a, b):
    return np.array_equal(np.asarray(a).view(np.uint8), np.asarray(b).view(np.uint8))


@pytest.mark.parametrize("south,north", itertools.product([False, True], repeat=2))
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("shape,dtype", [((7, 8), jnp.float32), ((7, 8, 3), jnp.float64)])
def test_apply_pole_end_masks_matches_row_set(south, north, offset, shape, dtype):
    f = _field(shape, dtype)
    n = shape[0]
    ref = f
    if south:
        ref = ref.at[offset].set(jnp.zeros_like(ref[offset]))
    if north:
        ref = ref.at[n - 1 - offset].set(jnp.zeros_like(ref[n - 1 - offset]))
    out = _apply(f, (jnp.bool_(south), jnp.bool_(north)), offset=offset)
    assert out.dtype == f.dtype
    assert _bits_equal(out, ref)


@pytest.mark.parametrize("n", [1, 2])
def test_apply_pole_end_masks_tiny_rows_both_poles(n):
    f = _field((n, 8, 3), jnp.float32)
    out = _apply(f, (jnp.bool_(True), jnp.bool_(True)), offset=0)
    assert _bits_equal(out, jnp.zeros_like(f))


def test_apply_pole_end_masks_rejects_offset_outside_rows():
    f = _field((3, 8), jnp.float32)
    with pytest.raises(ValueError, match="offset 3 outside"):
        _apply(f, (jnp.bool_(True), jnp.bool_(False)), offset=3)


N_DEV, N_LAT, N_LON = 4, 16, 8
NL = N_LAT // N_DEV


@pytest.mark.parametrize("nlev", [None, 5])
def test_interp_cell_to_vface_halo_spmd_pole_restore_bitwise(nlev):
    """Band outputs of the SPMD v-face interp, INCLUDING the restored pole
    edge-copy rows, equal the padded-copy reference bit-for-bit."""
    from functools import partial
    from jax.sharding import NamedSharding, PartitionSpec as P
    from legoesm.grids.operators_latlon_cgrid import interp_cell_to_vface_halo
    from legoesm.parallel.latlon_spmd import (
        activate_latlon_spmd_halo, deactivate_latlon_spmd_halo)
    from legoesm.parallel.shard_map_compat import shard_map
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))
    shp = (N_LAT, N_LON) if nlev is None else (N_LAT, N_LON, nlev)
    f = jax.random.normal(jax.random.PRNGKey(3), shp, jnp.float32)
    isp = P("lat", *((None,) * (f.ndim - 1)))
    f_sh = jax.device_put(f, NamedSharding(mesh, isp))
    activate_latlon_spmd_halo(mesh)
    try:
        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp, check_vma=False)
        def g(fb):
            return interp_cell_to_vface_halo(fb)
        out = np.asarray(g(f_sh))  # (N_DEV*(NL+1), N_LON[, nlev]) per-band blocks
    finally:
        deactivate_latlon_spmd_halo()
    zero = jnp.zeros_like(f[0:1])
    for b in range(N_DEV):
        fb = f[b * NL:(b + 1) * NL]
        south = f[b * NL - 1:b * NL] if b > 0 else zero
        north = f[(b + 1) * NL:(b + 1) * NL + 1] if b < N_DEV - 1 else zero
        fp = jnp.concatenate([south, fb, north], axis=0)
        ref = 0.5 * (fp[:-1] + fp[1:])
        if b == 0:
            ref = jnp.concatenate([fb[0:1], ref[1:]], axis=0)
        if b == N_DEV - 1:
            ref = jnp.concatenate([ref[:-1], fb[-1:]], axis=0)
        blk = out[b * (NL + 1):(b + 1) * (NL + 1)]
        assert _bits_equal(blk, ref), f"band {b} differs"


@pytest.mark.parametrize("south,north", itertools.product([False, True], repeat=2))
def test_apply_pole_end_masks_traced_masks_under_jit(south, north):
    f = _field((7, 8, 3), jnp.float32)
    fn = jax.jit(lambda x, s, n: _apply(x, (s, n), offset=0))
    out = fn(f, jnp.bool_(south), jnp.bool_(north))
    ref = f
    if south:
        ref = ref.at[0].set(jnp.zeros_like(ref[0]))
    if north:
        ref = ref.at[-1].set(jnp.zeros_like(ref[-1]))
    assert _bits_equal(out, ref)


def test_apply_pole_end_masks_rejects_non_scalar_masks():
    f = _field((3, 8), jnp.float32)
    with pytest.raises(ValueError, match="scalar booleans"):
        _apply(f, (jnp.ones((3,), bool), jnp.bool_(False)), offset=0)


@pytest.mark.parametrize("south,north", itertools.product([False, True], repeat=2))
def test_apply_pole_end_masks_gradient_matches_row_set(south, north):
    f = jax.random.normal(jax.random.PRNGKey(5), (7, 8, 3), jnp.float32)
    w = jax.random.normal(jax.random.PRNGKey(6), (7, 8, 3), jnp.float32)
    masks = (jnp.bool_(south), jnp.bool_(north))

    def ref_fn(x):
        out = x
        if south:
            out = out.at[0].set(jnp.zeros_like(out[0]))
        if north:
            out = out.at[-1].set(jnp.zeros_like(out[-1]))
        return jnp.sum(out * w)

    g_new = jax.grad(lambda x: jnp.sum(_apply(x, masks, offset=0) * w))(f)
    g_ref = jax.grad(ref_fn)(f)
    assert _bits_equal(g_new, g_ref)
