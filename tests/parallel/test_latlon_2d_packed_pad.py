"""Packed 2-D tile halo: identical values, far fewer messages.

The tiled decomposition moves 5.3x fewer halo bytes than latitude bands and
still measured 27% slower at 64 GPUs, because the packing that gives the
band lane 13 messages per step refuses any non-1-D mesh — so the tiled lane
issued one message PER FIELD, 108 of them, at about 26 microseconds each.

These tests pin the two things that make the packed body worth having: it
must agree with the per-field body to the bit, and it must actually collapse
the message count. A packed body that is merely correct fixes nothing.

Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=8``).
"""
from __future__ import annotations

import re
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from legoesm.parallel.latlon_spmd import (
    make_latlon_2d_packed_pad_body,
    make_latlon_2d_pad_body,
)
from legoesm.parallel.shard_map_compat import shard_map

N_LAT, N_LON, HALO = 16, 32, 1


def _mesh(p_lat, p_lon):
    need = p_lat * p_lon
    if len(jax.devices()) < need:
        pytest.skip(f"needs --xla_force_host_platform_device_count={need}")
    return Mesh(np.array(jax.devices()[:need]).reshape(p_lat, p_lon),
                axis_names=("lat", "lon"))


def _fields(rng, nlev_list):
    out = []
    for nlev in nlev_list:
        shape = (N_LAT, N_LON) if nlev is None else (N_LAT, N_LON, nlev)
        out.append(jnp.asarray(rng.standard_normal(shape), dtype=jnp.float32))
    return out


def _run_packed(mesh, specs, fields):
    body = make_latlon_2d_packed_pad_body(mesh, specs)
    isp = [P("lat", "lon", *((None,) * (f.ndim - 2))) for f in fields]

    @partial(shard_map, mesh=mesh, in_specs=tuple(isp), out_specs=tuple(isp),
             check_vma=False)
    def _ex(*fs):
        return body(*fs)

    return _ex(*fields)


def _run_per_field(mesh, specs, fields):
    outs = []
    for sp, f in zip(specs, fields):
        body = make_latlon_2d_pad_body(mesh, halo=sp[1], negate=sp[2])
        isp = P("lat", "lon", *((None,) * (f.ndim - 2)))

        @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
                 check_vma=False)
        def _ex(x, _body=body):
            return _body(x)

        outs.append(_ex(f))
    return outs


# (1, 1) and (4, 1) exercise the single-lon-tile WRAP, where a tile's east
# ghost is its own west columns rather than a message. It has its own branch
# in the packed body and the direction is easy to get backwards, so it is
# covered rather than assumed. (1, 2) is a single latitude band, where both
# pole folds land on the same tile.
@pytest.mark.parametrize("p_lat,p_lon",
                         [(2, 2), (1, 2), (2, 4), (1, 1), (4, 1), (2, 1)])
@pytest.mark.parametrize("halo", [1, 2])
def test_packed_matches_per_field_bit_for_bit(p_lat, p_lon, halo):
    mesh = _mesh(p_lat, p_lon)
    if N_LON // p_lon < 4 * halo:
        pytest.skip(f"tile width {N_LON // p_lon} under 4x halo {halo}")
    rng = np.random.default_rng(11 + p_lat * 10 + p_lon + 100 * halo)
    fields = _fields(rng, [None, 3, 5])
    specs = (("fold", halo, False), ("fold", halo, True),
             ("fold", halo, False))

    got = _run_packed(mesh, specs, fields)
    ref = _run_per_field(mesh, specs, fields)
    for i, (a, b) in enumerate(zip(ref, got)):
        assert a.shape == b.shape, f"field {i}: {b.shape} != {a.shape}"
        assert a.dtype == b.dtype, f"field {i}: {b.dtype} != {a.dtype}"
        np.testing.assert_array_equal(
            np.asarray(jax.device_get(a)), np.asarray(jax.device_get(b)),
            err_msg=f"field {i}: packed differs from per-field")


def _count_collectives(fn, *args):
    text = jax.jit(fn).lower(*args).compile().as_text()
    return len(re.findall(r"collective-permute(?:-start)?\(", text))


def test_packed_collapses_the_message_count():
    """The whole point. Six fields through the per-field body cost six times
    the messages; through the packed body they cost the same as one."""
    mesh = _mesh(2, 4)
    rng = np.random.default_rng(99)
    fields = _fields(rng, [None, 3, 5, 4, None, 6])
    specs = tuple(("fold", HALO, i % 2 == 1) for i in range(len(fields)))

    packed = _count_collectives(
        lambda *fs: _run_packed(mesh, specs, fs), *fields)
    per_field = _count_collectives(
        lambda *fs: _run_per_field(mesh, specs, fs), *fields)

    # The claim is that the count is INDEPENDENT of the field count, so
    # compare against one field packed. A "3x fewer" bar would pass a
    # regression to two packed batches.
    one = _count_collectives(
        lambda f: _run_packed(mesh, specs[:1], (f,)), fields[0])
    assert packed > 0, "no collectives lowered — the test proves nothing"
    assert packed == one, (
        f"packed body issues {packed} messages for six fields against "
        f"{one} for one; it was supposed to be independent of the field "
        f"count (per-field path: {per_field})")


def test_refuses_what_it_cannot_pack():
    mesh = _mesh(2, 2)
    with pytest.raises(ValueError, match="unknown spec kind"):
        make_latlon_2d_packed_pad_body(mesh, (("bucket", 1, 0.0),))
    with pytest.raises(ValueError, match="needs a 2-D"):
        band = Mesh(np.array(jax.devices()[:2]), axis_names=("lat",))
        make_latlon_2d_packed_pad_body(band, (("fold", 1, False),))
    with pytest.raises(ValueError, match="fold spec must be"):
        make_latlon_2d_packed_pad_body(mesh, (("fold", 1),))
    with pytest.raises(ValueError, match="wall spec must be"):
        make_latlon_2d_packed_pad_body(mesh, (("wall", 1, 0.0),))
    with pytest.raises(ValueError, match="sources must be plain integers"):
        make_latlon_2d_packed_pad_body(
            mesh, (("fold", 1, False),), sources=(True,))


def test_takes_what_the_band_twin_takes():
    """Non-vacuity for the refusals above, and the contract this body was
    widened to meet: wall specs and mixed halo depths are ACCEPTED now, so
    the dycore's stage epoch -- two fold depths and several walls -- can ride
    one exchange on a tile the way it already does on a band."""
    mesh = _mesh(2, 2)
    body = make_latlon_2d_packed_pad_body(
        mesh, (("fold", 1, False), ("fold", 2, False),
               ("wall", 1, 0.0, 0.0)))
    assert callable(body)


def test_refuses_mixed_dtypes_rather_than_promoting_them():
    """A packed buffer promotes bfloat16 with float32 to float32 and nothing
    casts back, so a mixed group would silently change field dtypes. The
    per-field path preserves them, so this must raise rather than differ."""
    mesh = _mesh(2, 2)
    rng = np.random.default_rng(5)
    a = jnp.asarray(rng.standard_normal((N_LAT, N_LON, 3)), dtype=jnp.float32)
    b = jnp.asarray(rng.standard_normal((N_LAT, N_LON, 3)), dtype=jnp.bfloat16)
    specs = (("fold", HALO, False), ("fold", HALO, False))
    with pytest.raises(ValueError, match="share one dtype"):
        _run_packed(mesh, specs, [a, b])


def test_mixed_depths_match_the_per_field_body():
    """The dycore's stage epoch asks for one field at halo 1 and another at
    halo 2 in the same exchange. Packing them together must give exactly what
    padding each separately gives -- the exchange is a bit-copy and the pack
    is a reshape, so anything else is an indexing mistake in the unpacking."""
    mesh = _mesh(2, 2)
    rng = np.random.default_rng(11)
    fields = _fields(rng, [3, 3])
    specs = (("fold", 1, False), ("fold", 2, True))
    packed = _run_packed(mesh, specs, fields)
    per_field = _run_per_field(mesh, specs, fields)
    for i, (a, b) in enumerate(zip(packed, per_field)):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"field {i} at halo {specs[i][1]} differs from the "
                    f"per-field pad")


def test_a_wall_field_matches_the_wall_body():
    """A wall field's pole ghost is a constant, not the atmospheric fold, and
    its pad is latitude-only. Riding it in the packed exchange must give what
    the wall body gives on the same tile mesh; applying fold semantics to it
    would be silent, since the shapes would still line up in latitude."""
    from legoesm.parallel.latlon_spmd import make_latlon_band_wall_pad_body

    mesh = _mesh(2, 2)
    rng = np.random.default_rng(12)
    fields = _fields(rng, [3, 3])
    specs = (("fold", 1, False), ("wall", 1, -2.5, 7.25))
    packed = _run_packed(mesh, specs, fields)

    wall_body = make_latlon_band_wall_pad_body(
        mesh, halo=1, south_value=-2.5, north_value=7.25)
    isp = P("lat", "lon", None)

    @partial(shard_map, mesh=mesh, in_specs=isp, out_specs=isp,
             check_vma=False)
    def _ex(x):
        return wall_body(x)

    np.testing.assert_array_equal(
        np.asarray(packed[1]), np.asarray(_ex(fields[1])),
        err_msg="the packed wall field differs from the wall body")
    # Non-vacuity: the wall constants must actually appear, or this would
    # pass on two bodies that both quietly dropped them.
    edge = np.asarray(packed[1])
    assert np.isclose(edge[0], -2.5).any() or np.isclose(edge[-1], 7.25).any()


def test_sources_let_one_array_ride_twice():
    """The stage epoch asks for temperature at two depths with two different
    semantics from ONE array. The sharing cannot be seen inside the traced
    body, so the caller declares it; both outputs must still match what
    padding that array separately gives."""
    mesh = _mesh(2, 2)
    rng = np.random.default_rng(13)
    field = _fields(rng, [3])[0]
    specs = (("fold", 1, False), ("fold", 2, False))
    body = make_latlon_2d_packed_pad_body(mesh, specs, sources=(0, 0))
    isp = P("lat", "lon", None)

    @partial(shard_map, mesh=mesh, in_specs=(isp,), out_specs=(isp, isp),
             check_vma=False)
    def _ex(x):
        return body(x)

    shared = _ex(field)
    separate = _run_per_field(mesh, specs, [field, field])
    for i, (a, b) in enumerate(zip(shared, separate)):
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(b),
            err_msg=f"shared-input output {i} differs from the per-field pad")
