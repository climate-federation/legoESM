"""SPMD fused multi-field halo (message aggregation, audit item 7).

Gates the three contracts of ``make_latlon_band_wall_multi_pad_body`` /
the ``LEGOESM_LATLON_SPMD_FUSED_HALO`` opt-in:

1. BIT-identity vs the per-field wall pads (mixed trailing shapes, mixed
   boundary constants, mixed dtypes) — the fusion is packing, never
   arithmetic.
2. End-to-end bit-identity of the FULL sharded ocean step with the flag
   on vs off.
3. The point, mechanically: compiled ``collective-permute`` count drops
   (one pair per direction per dtype group instead of one per field).

Also gates the two M2-leftover message-aggregation levers (M4 quick wins):

4. ``reconstruct_vface_lower_multi`` — the fused v-carrier boundary-row
   reconstruction (v + v_mask in ONE ppermute per dtype group inside the
   sharded ocean step): bit-identity vs the per-field
   ``reconstruct_vface_lower`` + the compiled collective-permute drop.
5. ``eta_floor._global_sum_pair`` — the batched SPMD psum pair (ONE packed
   ``psum`` via ``batch_psum_spmd`` instead of two): bit-identity vs the
   separate psums + exactly one compiled all-reduce.

Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=4 \
      pytest tests/parallel/test_latlon_spmd_fused_halo.py``
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from legoesm.parallel.latlon_spmd import (
    make_latlon_band_wall_multi_pad_body,
    make_latlon_band_wall_pad_body,
)
from legoesm.parallel.shard_map_compat import shard_map

pytestmark = pytest.mark.skipif(
    jax.device_count() < 4,
    reason="needs >=4 devices (XLA_FLAGS host device count)")

N_LAT, N_LON, NLEV = 32, 16, 3


def _mesh():
    return Mesh(np.array(jax.devices()[:4]), axis_names=("lat",))


def _fields():
    rng = np.random.default_rng(11)
    a = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))            # T-like
    b = jnp.asarray(rng.standard_normal((N_LAT, N_LON + 1)))        # u-like
    c = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))      # 3-D
    d = jnp.asarray(
        rng.standard_normal((N_LAT, N_LON)).astype(np.float32))     # f32
    return a, b, c, d


def test_fused_bit_identical_to_per_field():
    mesh = _mesh()
    fields = _fields()
    sv = (0.0, 1.5, 0.0, -2.0)
    nv = (0.0, 0.0, 3.0, 0.5)

    fused_body = make_latlon_band_wall_multi_pad_body(
        mesh, halo=1, south_values=sv, north_values=nv,
        n_fields=len(fields))
    per_bodies = [
        make_latlon_band_wall_pad_body(
            mesh, halo=1, south_value=sv[i], north_value=nv[i])
        for i in range(len(fields))
    ]

    specs = tuple(P("lat", *((None,) * (f.ndim - 1))) for f in fields)
    out_specs = specs

    fused = shard_map(
        lambda *fs: fused_body(*fs), mesh=mesh,
        in_specs=specs, out_specs=out_specs, check_vma=False)(*fields)
    per = shard_map(
        lambda *fs: tuple(per_bodies[i](fs[i]) for i in range(len(fs))),
        mesh=mesh, in_specs=specs, out_specs=out_specs,
        check_vma=False)(*fields)

    for i, (f_out, p_out) in enumerate(zip(fused, per)):
        np.testing.assert_array_equal(
            np.asarray(f_out), np.asarray(p_out),
            err_msg=f"fused pad diverged from per-field pad on field {i}")
        assert f_out.dtype == fields[i].dtype


@pytest.mark.parametrize("halo", [2, 3])
def test_fused_bit_identical_wide_halo(halo):
    """halo>1 parity: the wide-halo helpers route their W / W+1 exchanges
    through the same dispatch (codex) — the packing must hold at any
    width."""
    mesh = _mesh()
    fields = _fields()[:3]
    fused_body = make_latlon_band_wall_multi_pad_body(
        mesh, halo=halo, n_fields=len(fields))
    per_bodies = [make_latlon_band_wall_pad_body(mesh, halo=halo)
                  for _ in fields]
    specs = tuple(P("lat", *((None,) * (f.ndim - 1))) for f in fields)
    fused = shard_map(lambda *fs: fused_body(*fs), mesh=mesh,
                      in_specs=specs, out_specs=specs,
                      check_vma=False)(*fields)
    per = shard_map(
        lambda *fs: tuple(per_bodies[i](fs[i]) for i in range(len(fs))),
        mesh=mesh, in_specs=specs, out_specs=specs,
        check_vma=False)(*fields)
    for i, (f_out, p_out) in enumerate(zip(fused, per)):
        np.testing.assert_array_equal(np.asarray(f_out), np.asarray(p_out),
                                      err_msg=f"halo={halo} field {i}")


def test_fused_wrong_arity_raises():
    mesh = _mesh()
    body = make_latlon_band_wall_multi_pad_body(mesh, n_fields=2)
    with pytest.raises(ValueError, match="built for 2 fields"):
        shard_map(lambda a: body(a), mesh=mesh, in_specs=(P("lat"),),
                  out_specs=P("lat"), check_vma=False)(
            jnp.zeros((N_LAT, N_LON)))


def _count_ppermutes(hlo_text: str) -> int:
    return sum(1 for line in hlo_text.splitlines()
               if "collective-permute" in line and "done" not in line)


def test_fused_cuts_collective_count():
    """5 same-dtype fields: per-field = 10 ppermutes (2/field), fused = 2."""
    mesh = _mesh()
    rng = np.random.default_rng(3)
    fields = tuple(jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
                   for _ in range(5))
    specs = tuple(P("lat", None) for _ in fields)

    fused_body = make_latlon_band_wall_multi_pad_body(
        mesh, n_fields=5)
    per_bodies = [make_latlon_band_wall_pad_body(mesh) for _ in fields]

    fused_fn = jax.jit(shard_map(
        lambda *fs: fused_body(*fs), mesh=mesh, in_specs=specs,
        out_specs=specs, check_vma=False))
    per_fn = jax.jit(shard_map(
        lambda *fs: tuple(per_bodies[i](fs[i]) for i in range(len(fs))),
        mesh=mesh, in_specs=specs, out_specs=specs, check_vma=False))

    n_fused = _count_ppermutes(
        fused_fn.lower(*fields).compile().as_text())
    n_per = _count_ppermutes(
        per_fn.lower(*fields).compile().as_text())
    assert n_per >= 10, n_per
    assert n_fused <= 2, n_fused


def _count_allreduces(hlo_text: str) -> int:
    return sum(1 for line in hlo_text.splitlines()
               if ("all-reduce(" in line or "all-reduce-start(" in line))


# ---------------------------------------------------------------------------
# M2 leftover (M4 quick win): fused v-carrier boundary-row reconstruction —
# the sharded ocean step's v + v_mask staggered carriers ride ONE ppermute
# per dtype group instead of one each (reconstruct_vface_lower_multi).
# ---------------------------------------------------------------------------

def _vcarrier_fields():
    """v-like carriers: 3-D f64 (v), 2-D f64 (v_mask), 3-D f32 (dtype group)."""
    rng = np.random.default_rng(7)
    v = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
    vm = jnp.asarray((rng.random((N_LAT, N_LON)) > 0.3).astype(np.float64))
    v32 = jnp.asarray(
        rng.standard_normal((N_LAT, N_LON, NLEV)).astype(np.float32))
    return v, vm, v32


def test_vface_multi_reconstruct_bit_identical():
    """Fused v-carrier reconstruction == per-field reconstruct_vface_lower,
    bit-for-bit (mixed trailing shapes + a second dtype group), including the
    north band's ppermute non-target zero row."""
    from legoesm.parallel.latlon_spmd import (
        latlon_band_perms,
        reconstruct_vface_lower,
        reconstruct_vface_lower_multi,
    )

    mesh = _mesh()
    fields = _vcarrier_fields()
    axis = "lat"
    n_dev = mesh.devices.size
    perm_north, _ = latlon_band_perms(n_dev)
    specs = tuple(P("lat", *((None,) * (f.ndim - 1))) for f in fields)
    # Shard the (nl+1)-row per-band outputs back on "lat" (the tiled
    # methodology): the global result stacks EVERY band's block — band b's
    # window is [b*(nl+1) : (b+1)*(nl+1)] — so the parity covers all bands
    # including the north band's ppermute non-target zero row.
    out_specs = specs

    def _fused(*fs):
        return reconstruct_vface_lower_multi(fs, axis, perm_north)

    def _per_field(*fs):
        return tuple(reconstruct_vface_lower(f, axis, perm_north) for f in fs)

    fused = shard_map(_fused, mesh=mesh, in_specs=specs,
                      out_specs=out_specs, check_vma=False)(*fields)
    per = shard_map(_per_field, mesh=mesh, in_specs=specs,
                    out_specs=out_specs, check_vma=False)(*fields)

    for i, (f_out, p_out) in enumerate(zip(fused, per)):
        assert f_out.shape[0] == N_LAT + n_dev   # n_dev stacked (nl+1) blocks
        np.testing.assert_array_equal(
            np.asarray(f_out), np.asarray(p_out),
            err_msg=f"fused v-carrier reconstruction diverged on field {i}")
        assert f_out.dtype == fields[i].dtype
        # north band's boundary row is the ppermute non-target zero
        assert not np.asarray(f_out)[-1].any()


def test_vface_multi_reconstruct_cuts_collective_count():
    """2 same-dtype carriers (the production v + v_mask pair): per-field = 2
    collective-permutes, fused = 1."""
    from legoesm.parallel.latlon_spmd import (
        latlon_band_perms,
        reconstruct_vface_lower,
        reconstruct_vface_lower_multi,
    )

    mesh = _mesh()
    v, vm, _ = _vcarrier_fields()
    axis = "lat"
    perm_north, _ = latlon_band_perms(mesh.devices.size)
    specs = (P("lat", None, None), P("lat", None))
    out_specs = specs                       # stacked per-band (nl+1) blocks

    fused_fn = jax.jit(shard_map(
        lambda a, b: reconstruct_vface_lower_multi((a, b), axis, perm_north),
        mesh=mesh, in_specs=specs, out_specs=out_specs, check_vma=False))
    per_fn = jax.jit(shard_map(
        lambda a, b: (reconstruct_vface_lower(a, axis, perm_north),
                      reconstruct_vface_lower(b, axis, perm_north)),
        mesh=mesh, in_specs=specs, out_specs=out_specs, check_vma=False))

    n_fused = _count_ppermutes(fused_fn.lower(v, vm).compile().as_text())
    n_per = _count_ppermutes(per_fn.lower(v, vm).compile().as_text())
    assert n_per >= 2, n_per
    assert n_fused <= 1, n_fused


# ---------------------------------------------------------------------------
# M2 leftover (M4 quick win): eta_floor._global_sum_pair batches its two
# scalars through batch_psum_spmd — ONE packed psum, bit-identical values.
# ---------------------------------------------------------------------------

def test_global_sum_pair_spmd_batched_bit_identical_single_collective():
    from legoesm.ocean.dynamics.eta_floor import _global_sum_pair
    from legoesm.parallel.latlon_spmd import (
        activate_latlon_spmd_halo,
        deactivate_latlon_spmd_halo,
    )

    mesh = _mesh()
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    y = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    specs = (P("lat", None), P("lat", None))

    def _pair_body(xl, yl):
        # the clamp_and_redistribute pattern: band-local partial sums
        return _global_sum_pair(jnp.sum(xl), jnp.sum(yl))

    def _ref_body(xl, yl):
        # the pre-batching semantics: two separate psums
        return (jax.lax.psum(jnp.sum(xl), "lat"),
                jax.lax.psum(jnp.sum(yl), "lat"))

    # _global_sum_pair dispatches on the ARMED spmd backend at trace time.
    activate_latlon_spmd_halo(mesh)
    try:
        pair_fn = jax.jit(shard_map(
            _pair_body, mesh=mesh, in_specs=specs, out_specs=(P(), P()),
            check_vma=False))
        ref_fn = jax.jit(shard_map(
            _ref_body, mesh=mesh, in_specs=specs, out_specs=(P(), P()),
            check_vma=False))
        a_b, b_b = pair_fn(x, y)
        a_r, b_r = ref_fn(x, y)
        hlo_pair = pair_fn.lower(x, y).compile().as_text()
    finally:
        deactivate_latlon_spmd_halo()

    # BIT-identical to the separate psums (packing only, no arithmetic).
    np.testing.assert_array_equal(np.asarray(a_b), np.asarray(a_r))
    np.testing.assert_array_equal(np.asarray(b_b), np.asarray(b_r))
    # ...and the pair rides exactly ONE compiled all-reduce.
    assert _count_allreduces(hlo_pair) == 1, hlo_pair.count("all-reduce")


def _env_flag(monkeypatch, value):
    monkeypatch.setenv("LEGOESM_LATLON_SPMD_FUSED_HALO", value)


@pytest.mark.parametrize("n_steps", [2])
def test_ocean_step_bit_identical_with_fusion(monkeypatch, n_steps):
    """The FULL sharded ocean step: fusion on vs off must be BIT-identical
    (packing only, no arithmetic) — and the compiled step must carry fewer
    collective-permutes with fusion on (the aggregation actually fires on
    the production pad_multi sites)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon,
        make_sharded_ocean_step,
        shard_state_latlon,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.parallel.mesh import create_latlon_mesh

    grid = create_latlon_grid(n_lat=48, n_lon=96)
    z_coord = create_ocean_z_star(n_levels=6, H_max=4000.0)
    model = LatLonCGridOceanModel(grid, z_coord,
                                  LatLonCGridOceanConfig.from_flat())
    state0 = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0)
    rng = np.random.default_rng(0)
    eta = 0.005 * rng.standard_normal((48, 96))
    state0 = state0._replace(eta=state0.eta.replace(data=jnp.asarray(eta)))
    model._ensure_vertex_mask(state0)
    dev = create_latlon_mesh(n_devices=4)

    outs = {}
    hlo_counts = {}
    for flag in ("0", "1"):
        _env_flag(monkeypatch, flag)
        step = make_sharded_ocean_step(model, dev.mesh)
        ss = shard_state_latlon(state0, dev.mesh)
        hlo_counts[flag] = _count_ppermutes(
            jax.jit(step).lower(ss, 600.0).compile().as_text())
        for _ in range(n_steps):
            ss = step(ss, 600.0)
        outs[flag] = gather_state_latlon(ss, dev.mesh)

    for nm in ("eta", "u", "v", "T", "S"):
        np.testing.assert_array_equal(
            np.asarray(getattr(outs["1"], nm).data),
            np.asarray(getattr(outs["0"], nm).data),
            err_msg=f"fused-halo step diverged on {nm}")
    assert hlo_counts["1"] < hlo_counts["0"], hlo_counts

    # Flag flip on a REUSED step object must rebuild the shard_map (the
    # switch is trace-time; the step wrapper's per-CALL cache key carries
    # it — codex).  Fresh outer lambdas per lowering: jax.jit caches on
    # the callable identity, so re-jitting the SAME step object would
    # freeze the wrapper's python body and never re-evaluate the key
    # (an outer-jit artifact, not the production call pattern — the
    # driver calls step() directly each step).
    _env_flag(monkeypatch, "0")
    step = make_sharded_ocean_step(model, dev.mesh)
    ss = shard_state_latlon(state0, dev.mesh)
    n_off = _count_ppermutes(
        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
        .compile().as_text())
    _env_flag(monkeypatch, "1")
    n_on = _count_ppermutes(
        jax.jit(lambda s, d: step(s, d)).lower(ss, 600.0)
        .compile().as_text())
    assert n_on < n_off, (n_off, n_on)
