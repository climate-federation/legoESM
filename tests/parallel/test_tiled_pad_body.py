"""P4 phase-1b: the unwrapped tiled pad body works inside an outer
shard_map and matches the wrapped exchange bit-for-bit.

``make_tiled_pad_body`` returns the tiled halo pad WITHOUT its own
shard_map wrapper, so the tiled FV3 tendency stage can call it from
inside its own ``shard_map(face,tile_i,tile_j)`` (a nested shard_map —
what the operators' SPMD pad does — would be illegal).  This pins:

1. wrapped ``_make_exchange_ppermute_tiled`` still equals the serial
   pad after the _build_tiled_pad refactor (refactor-safety);
2. the unwrapped body, invoked inside a SEPARATE outer shard_map over
   the same axes, produces the identical global padded result.

Runs on 24 host CPU devices (kt=2) — no MPI:
``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
try:
    from jax import shard_map  # JAX >= 0.8 top-level
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map

from legoesm.grids.halo import pad_halo_local, compute_halo_interp_offsets
from legoesm.parallel.cubesphere_exchange import (
    _make_exchange_ppermute_tiled,
    make_tiled_pad_body,
    make_tiled_pad_vector_body,
)

KT = 2
NL = 24
N = KT * NL


@pytest.fixture(scope="module")
def mesh():
    if len(jax.devices()) < 24:
        pytest.skip("needs 24 host devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=24)")
    dev = np.array(jax.devices()[:24]).reshape(6, KT, KT)
    return Mesh(dev, axis_names=("face", "tile_i", "tile_j"))


def test_wrapped_tiled_exchange_matches_serial(mesh):
    rng = np.random.default_rng(1)
    ref = jnp.asarray(rng.standard_normal((6, N, N)))
    offs = compute_halo_interp_offsets(N)
    serial = np.asarray(pad_halo_local(ref, interp_offsets=offs))

    sharding = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
    ref_sh = jax.device_put(ref, sharding)
    fn = _make_exchange_ppermute_tiled(mesh, ndim=3, halo=1,
                                       with_offsets=True)
    out = fn(ref_sh, jnp.asarray(offs))
    blk = NL + 2
    for shard in out.addressable_shards:
        f = shard.index[0].start or 0
        ti = (shard.index[1].start or 0) // blk
        tj = (shard.index[2].start or 0) // blk
        got = np.asarray(shard.data)[0]
        want = serial[f, ti * NL: ti * NL + blk, tj * NL: tj * NL + blk]
        np.testing.assert_array_equal(got, want)


def test_unwrapped_body_inside_outer_shardmap_matches_wrapped(mesh):
    """The whole point: the body runs inside a DIFFERENT shard_map and
    gives the same padded blocks as the wrapped exchange."""
    rng = np.random.default_rng(2)
    ref = jnp.asarray(rng.standard_normal((6, N, N)))
    offs = jnp.asarray(compute_halo_interp_offsets(N))
    sharding = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
    ref_sh = jax.device_put(ref, sharding)

    wrapped = _make_exchange_ppermute_tiled(
        mesh, ndim=3, halo=1, with_offsets=True)
    out_wrapped = wrapped(ref_sh, offs)

    body = make_tiled_pad_body(mesh, ndim=3, halo=1, with_offsets=True)

    # Outer shard_map (separate from the body's own) that calls the
    # unwrapped body on each device's local tile — exactly how the
    # tiled tendency stage will invoke it.
    out_blk = NL + 2
    in_sp = (P("face", "tile_i", "tile_j"), P())
    out_sp = P("face", "tile_i", "tile_j")

    @partial(shard_map, mesh=mesh, in_specs=in_sp, out_specs=out_sp,
             check_vma=False)
    def _stage(local_shard, o):
        return body(local_shard[0], o)[None]

    out_body = _stage(ref_sh, offs)

    for sw, sb in zip(out_wrapped.addressable_shards,
                      out_body.addressable_shards):
        np.testing.assert_array_equal(
            np.asarray(sw.data), np.asarray(sb.data),
            err_msg="unwrapped body != wrapped exchange")
    assert out_body.shape == (6, KT * out_blk, KT * out_blk)


def test_ndim4_body_matches_ndim3_per_level(mesh):
    """The 4D in-stage SCALAR halo (ndim=4) — used by the tiled
    ``fv3_hydrostatic_tendencies`` MOMENTUM stage to exchange the cc
    intermediates {zeta, B, 1/T, ln_ps, hf} — must equal the proven ndim=3
    horizontal halo applied INDEPENDENTLY per vertical level (the halo is
    purely horizontal, so it broadcasts over the trailing nlev axis).  This
    pins the never-before-exercised ndim=4 ``make_tiled_pad_body`` (incl. the
    offset cross-face Lagrange interp broadcast over the trailing axis) BEFORE
    the momentum capstone relies on it.  C=3 (>1) catches a trailing-axis bug a
    C=1 lane would hide; the momentum stage uses BOTH C=nlev and C=1 (ln_ps)."""
    C = 3
    rng = np.random.default_rng(3)
    ref4 = jnp.asarray(rng.standard_normal((6, N, N, C)))
    offs = jnp.asarray(compute_halo_interp_offsets(N))
    sh4 = NamedSharding(mesh, P("face", "tile_i", "tile_j", None))
    sh3 = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
    ref4_sh = jax.device_put(ref4, sh4)

    body4 = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)
    body3 = make_tiled_pad_body(mesh, ndim=3, halo=1, with_offsets=True)
    out_blk = NL + 2

    @partial(shard_map, mesh=mesh,
             in_specs=(P("face", "tile_i", "tile_j", None), P()),
             out_specs=P("face", "tile_i", "tile_j", None), check_vma=False)
    def _stage4(local_shard, o):
        return body4(local_shard[0], o)[None]

    @partial(shard_map, mesh=mesh,
             in_specs=(P("face", "tile_i", "tile_j"), P()),
             out_specs=P("face", "tile_i", "tile_j"), check_vma=False)
    def _stage3(local_shard, o):
        return body3(local_shard[0], o)[None]

    out4 = _stage4(ref4_sh, offs)              # (6, KT*out_blk, KT*out_blk, C)
    assert out4.shape == (6, KT * out_blk, KT * out_blk, C)
    for k in range(C):
        ref3_sh = jax.device_put(ref4[..., k], sh3)
        out3 = _stage3(ref3_sh, offs)
        np.testing.assert_array_equal(
            np.asarray(out4)[..., k], np.asarray(out3),
            err_msg=f"ndim=4 halo level {k} != ndim=3 halo of that level")


def test_ndim4_vector_body_matches_ndim3_per_level(mesh):
    """The 4D in-stage ROTATING VECTOR halo (ndim=4) — needed by the tiled
    ``center_to_dgrid_vector`` lift (the section-12c ``_vert_adv_uv_d``
    contribution to the hydrostatic momentum capstone) — must equal the proven
    ndim=3 vector halo applied INDEPENDENTLY per vertical level (the grid->geo
    rotation is purely horizontal, so the cos/sin angle metrics broadcast over
    the trailing nlev axis).  Pins the never-before-exercised ndim=4 path of
    ``make_tiled_pad_vector_body`` (the 3D-PE momentum stage used only SCALAR
    halos, so the angle-broadcast over the channel axis was untested).  C=3 (>1)
    catches a trailing-axis broadcast bug a C=1 lane would hide."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")
    g = create_cubed_sphere_cdgrid(create_cubed_sphere(N)).base
    assert g.duogrid is None, "base cut is non-duogrid"
    C = 3
    rng = np.random.default_rng(4)
    u4 = jnp.asarray(rng.standard_normal((6, N, N, C)))
    v4 = jnp.asarray(rng.standard_normal((6, N, N, C)))
    offs = jnp.asarray(g.halo_interp_offsets)
    ca, sa = g.cos_angle, g.sin_angle
    cap, sap = g.cos_angle_padded, g.sin_angle_padded   # (6, N+2, N+2)

    vbody4 = make_tiled_pad_vector_body(mesh, ndim=4, halo=1, with_offsets=True)
    vbody3 = make_tiled_pad_vector_body(mesh, ndim=3, halo=1, with_offsets=True)

    fw4 = P("face", "tile_i", "tile_j", None)
    fw3 = P("face", "tile_i", "tile_j")
    fo = P("face", None, None)                          # angles face-repl, sliced per tile

    def _tile_slice(arr, a_i, a_j, si, sj):
        arr = jax.lax.dynamic_slice_in_dim(arr, a_i, si, axis=1)
        return jax.lax.dynamic_slice_in_dim(arr, a_j, sj, axis=2)

    @partial(shard_map, mesh=mesh, in_specs=(fw4, fw4, fo, fo, fo, fo, P()),
             out_specs=(fw4, fw4), check_vma=False)
    def _stage4(u, v, ca_, sa_, cap_, sap_, o):
        a_i = jax.lax.axis_index("tile_i") * NL
        a_j = jax.lax.axis_index("tile_j") * NL
        ca_t = _tile_slice(ca_, a_i, a_j, NL, NL)[0]
        sa_t = _tile_slice(sa_, a_i, a_j, NL, NL)[0]
        cap_t = _tile_slice(cap_, a_i, a_j, NL + 2, NL + 2)[0]
        sap_t = _tile_slice(sap_, a_i, a_j, NL + 2, NL + 2)[0]
        up, vp = vbody4(u[0], v[0], ca_t, sa_t, cap_t, sap_t, o)
        return up[None], vp[None]

    @partial(shard_map, mesh=mesh, in_specs=(fw3, fw3, fo, fo, fo, fo, P()),
             out_specs=(fw3, fw3), check_vma=False)
    def _stage3(u, v, ca_, sa_, cap_, sap_, o):
        a_i = jax.lax.axis_index("tile_i") * NL
        a_j = jax.lax.axis_index("tile_j") * NL
        ca_t = _tile_slice(ca_, a_i, a_j, NL, NL)[0]
        sa_t = _tile_slice(sa_, a_i, a_j, NL, NL)[0]
        cap_t = _tile_slice(cap_, a_i, a_j, NL + 2, NL + 2)[0]
        sap_t = _tile_slice(sap_, a_i, a_j, NL + 2, NL + 2)[0]
        up, vp = vbody3(u[0], v[0], ca_t, sa_t, cap_t, sap_t, o)
        return up[None], vp[None]

    sh4 = NamedSharding(mesh, fw4)
    sh3 = NamedSharding(mesh, fw3)
    sho = NamedSharding(mesh, fo)
    ca_d, sa_d = jax.device_put(ca, sho), jax.device_put(sa, sho)
    cap_d, sap_d = jax.device_put(cap, sho), jax.device_put(sap, sho)
    out4u, out4v = _stage4(jax.device_put(u4, sh4), jax.device_put(v4, sh4),
                           ca_d, sa_d, cap_d, sap_d, offs)
    out_blk = NL + 2
    assert np.asarray(out4u).shape == (6, KT * out_blk, KT * out_blk, C)
    for k in range(C):
        o3u, o3v = _stage3(jax.device_put(u4[..., k], sh3),
                           jax.device_put(v4[..., k], sh3),
                           ca_d, sa_d, cap_d, sap_d, offs)
        np.testing.assert_array_equal(
            np.asarray(out4u)[..., k], np.asarray(o3u),
            err_msg=f"ndim=4 vector u halo level {k} != ndim=3")
        np.testing.assert_array_equal(
            np.asarray(out4v)[..., k], np.asarray(o3v),
            err_msg=f"ndim=4 vector v halo level {k} != ndim=3")


def test_multi_body_matches_per_field_pads(mesh):
    """make_tiled_pad_multi_body (the fused wave-A exchange) must be
    bit-identical per field to individual scalar-body pads, with mixed
    dtypes forced into separate groups and a C=1 field included (the
    ln_ps_3d shape)."""
    from legoesm.parallel.cubesphere_exchange import (
        make_tiled_pad_multi_body,
    )

    rng = np.random.default_rng(7)
    nlev = 3
    fields = (
        jnp.asarray(rng.standard_normal((6, N, N, nlev)), dtype=jnp.float32),
        jnp.asarray(rng.standard_normal((6, N, N, 1)), dtype=jnp.float64),
        jnp.asarray(rng.standard_normal((6, N, N, nlev)), dtype=jnp.float32),
        jnp.asarray(rng.standard_normal((6, N, N, 2)), dtype=jnp.float64),
    )
    offs = jnp.asarray(compute_halo_interp_offsets(N))
    sharding = NamedSharding(mesh, P("face", "tile_i", "tile_j"))
    fields_sh = tuple(jax.device_put(f, sharding) for f in fields)

    body = make_tiled_pad_body(mesh, ndim=4, halo=1, with_offsets=True)
    multi = make_tiled_pad_multi_body(body)

    in_sp = tuple(P("face", "tile_i", "tile_j") for _ in fields) + (P(),)
    out_sp = tuple(P("face", "tile_i", "tile_j") for _ in fields)

    @partial(shard_map, mesh=mesh, in_specs=in_sp, out_specs=out_sp,
             check_vma=False)
    def _fused(*args):
        tiles, o = args[:-1], args[-1]
        return tuple(p[None] for p in multi(tuple(t[0] for t in tiles), o))

    @partial(shard_map, mesh=mesh, in_specs=in_sp, out_specs=out_sp,
             check_vma=False)
    def _per_field(*args):
        tiles, o = args[:-1], args[-1]
        return tuple(body(t[0], o)[None] for t in tiles)

    got = _fused(*fields_sh, offs)
    want = _per_field(*fields_sh, offs)
    for i, (g, w) in enumerate(zip(got, want)):
        assert g.dtype == fields[i].dtype, i
        np.testing.assert_array_equal(np.asarray(g), np.asarray(w),
                                      err_msg=f"field {i} differs")
