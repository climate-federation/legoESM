"""Level-leading serialisation of the lat-band halo's edge payloads.

``LEGOESM_LATLON_HALO_LEVEL_LEADING=1`` reorders the elements INSIDE each
three-dimensional edge payload of the packed band exchange and of the fused
v-carrier reconstruction, from C-order of ``(h, lon, lev)`` to C-order of
``(lev, h, lon)``.  Nothing else moves: same widths, same offsets, same dtype
grouping, same field order, same number of collectives, same payload bytes.

Why it exists: the compiled step holds 28 transposes, none of them free,
converting between ``(level, lat, lon)`` and ``(lat, lon, level)`` — the state
is stored lat-leading and XLA:GPU pipelines level-leading.  Twenty-six of the
28 act on halo edge slices of one or two latitude rows, whose shape does not
follow the shard, which is why the transpose class is 87 % device-count
independent while every other class scales.  Whether asking for the other wire
order actually removes them is a GPU question this file does not answer; what
it answers is that asking cannot change the model's numbers.

Gates:
1. ``test_packed_body_bit_identical_off_vs_on`` — padded outputs of the packed
   band body are bit-identical with the flag off and on, on 2 and 4 devices,
   over mixed halos, mixed semantics, mixed dtypes, an odd trailing dimension
   and a two-dimensional field that must pass through untouched.
2. ``test_vface_multi_bit_identical_off_vs_on`` — the same for the fused
   v-carrier boundary row, which is the second packing site.
3. ``test_edge_serialisation_roundtrip`` — pack then unpack is the identity
   for both settings, on 2-D, 3-D and 1-D payloads.
4. ``test_wrong_inverse_is_detected`` — NON-VACUITY.  With the derived inverse
   replaced by the forward permutation (the classic wrong-inverse bug), the
   round-trip stops being the identity.  This is the executable form of the
   check: it fails if ``_unpack_edge_row`` ever stops inverting the pack.
5. ``test_env_gate_dispatch_hardening`` — an unknown value raises rather than
   silently running the default order.

Runs on host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4`` or more).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm.parallel import latlon_spmd
from legoesm.parallel.latlon_spmd import (
    latlon_band_perms,
    make_latlon_band_packed_pad_body,
    reconstruct_vface_lower_multi,
)
from legoesm.parallel.shard_map_compat import shard_map

ENV = "LEGOESM_LATLON_HALO_LEVEL_LEADING"
N_LAT = 16
N_LON = 16
NLEV = 5          # odd, so a level-leading permutation cannot be a no-op


def _mesh(n_dev):
    if len(jax.devices()) < n_dev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={n_dev}")
    return jax.sharding.Mesh(
        np.array(jax.devices()[:n_dev]), axis_names=("lat",))


def _fields():
    """Mixed halos, semantics, dtypes, an odd trailing dim and a 2-D field."""
    rng = np.random.default_rng(20260821)
    f_a = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
    f_b = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)),
                      dtype=jnp.float32)
    f_c = jnp.asarray(rng.standard_normal((N_LAT, N_LON + 1, 3)))
    f_d = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))   # 2-D: untouched
    specs = (("fold", 1, False), ("fold", 2, True),
             ("wall", 1, 1.5, -2.5), ("wall", 1, 0.0, 0.0))
    return (f_a, f_b, f_c, f_d), specs


def _run_packed(monkeypatch, n_dev, value):
    monkeypatch.delenv(ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    mesh = _mesh(n_dev)
    fields, specs = _fields()
    packed = make_latlon_band_packed_pad_body(mesh, specs)
    in_specs = tuple(P("lat", *((None,) * (f.ndim - 1))) for f in fields)
    fn = jax.jit(shard_map(packed, mesh=mesh, in_specs=in_specs,
                           out_specs=in_specs, check_vma=False))
    args = tuple(jax.device_put(f, NamedSharding(mesh, sp))
                 for f, sp in zip(fields, in_specs))
    return fn(*args)


@pytest.mark.parametrize("n_dev", [2, 4])
def test_packed_body_bit_identical_off_vs_on(monkeypatch, n_dev):
    base = _run_packed(monkeypatch, n_dev, None)   # unset is the default
    off = _run_packed(monkeypatch, n_dev, "0")
    on = _run_packed(monkeypatch, n_dev, "1")
    assert len(base) == 4
    for i, (b, o, n) in enumerate(zip(base, off, on)):
        b, o, n = (np.asarray(x) for x in (b, o, n))
        assert b.dtype == o.dtype == n.dtype, i
        assert b.shape == o.shape == n.shape, i
        # tobytes, not allclose: this is a permutation and its inverse, so
        # anything short of bit-identical is a defect, and a signed-zero
        # difference would slip past ``==``.
        assert b.tobytes() == o.tobytes(), f"'0' differs from unset on field {i}"
        assert b.tobytes() == n.tobytes(), f"level-leading moved field {i}"
    # Non-vacuity of the harness: the halo really was applied.
    assert np.asarray(base[0]).shape[0] == N_LAT + 2 * n_dev
    assert np.asarray(base[1]).shape[0] == N_LAT + 4 * n_dev


def test_vface_multi_bit_identical_off_vs_on(monkeypatch):
    n_dev = 4
    mesh = _mesh(n_dev)
    rng = np.random.default_rng(7)
    fields = (
        jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV))),
        jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)),
                    dtype=jnp.float32),
        jnp.asarray(rng.standard_normal((N_LAT, N_LON))),      # 2-D
    )
    perm_north, _ = latlon_band_perms(n_dev)
    specs = tuple(P("lat", *((None,) * (f.ndim - 1))) for f in fields)

    def run(value):
        monkeypatch.delenv(ENV, raising=False)
        if value is not None:
            monkeypatch.setenv(ENV, value)
        return shard_map(
            lambda *fs: reconstruct_vface_lower_multi(fs, "lat", perm_north),
            mesh=mesh, in_specs=specs, out_specs=specs,
            check_vma=False)(*fields)

    base, on = run(None), run("1")
    for i, (b, n) in enumerate(zip(base, on)):
        b, n = np.asarray(b), np.asarray(n)
        assert b.dtype == n.dtype and b.shape == n.shape, i
        assert b.tobytes() == n.tobytes(), f"level-leading moved v-carrier {i}"
    assert np.asarray(base[0]).shape[0] == N_LAT + n_dev


@pytest.mark.parametrize("shape", [(1, N_LON, NLEV), (2, N_LON + 1, 3),
                                   (1, N_LON), (7,)])
@pytest.mark.parametrize("level_leading", [False, True])
def test_edge_serialisation_roundtrip(shape, level_leading):
    x = jnp.asarray(np.random.default_rng(1).standard_normal(shape))
    row, w = latlon_spmd._pack_edge_slice(x, level_leading)
    assert row.shape == (1, int(np.prod(shape)))
    assert w == int(np.prod(shape))
    y = latlon_spmd._unpack_edge_row(row, shape, level_leading)
    assert np.asarray(y).shape == shape
    assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


def test_wrong_inverse_is_detected(monkeypatch):
    """NON-VACUITY: with the inverse replaced by the forward permutation,
    the round-trip is no longer the identity and gate 3 would fail."""
    shape = (2, N_LON + 1, 3)
    x = jnp.asarray(np.random.default_rng(2).standard_normal(shape))
    monkeypatch.setattr(latlon_spmd, "_inverse_perm", lambda perm: perm)
    row, _ = latlon_spmd._pack_edge_slice(x, True)
    y = latlon_spmd._unpack_edge_row(row, shape, True)
    assert np.asarray(x).tobytes() != np.asarray(y).tobytes()
    # And the correct inverse still round-trips, so the mutation is what
    # broke it rather than the payload being degenerate.
    monkeypatch.undo()
    y2 = latlon_spmd._unpack_edge_row(row, shape, True)
    assert np.asarray(x).tobytes() == np.asarray(y2).tobytes()


@pytest.mark.parametrize("bad", ["2", "true", "on", " 1", "yes", "-1"])
def test_env_gate_dispatch_hardening(bad):
    with pytest.raises(ValueError, match=ENV):
        latlon_spmd._resolve_halo_level_leading(bad)


def test_flag_actually_changes_the_compiled_body(monkeypatch):
    """NON-VACUITY OF THE FLAG ITSELF.

    Every other gate here asserts that something does NOT change, and all of
    them would stay green if the switch were never read. This one asserts the
    switch DOES reach the compiled program: with it on, the lowered body must
    contain transpose instructions that the default body does not.
    """
    n_dev = 2
    mesh = _mesh(n_dev)
    fields, specs = _fields()
    in_specs = tuple(P("lat", *((None,) * (f.ndim - 1))) for f in fields)

    def lowered(value):
        monkeypatch.delenv(ENV, raising=False)
        if value is not None:
            monkeypatch.setenv(ENV, value)
        packed = make_latlon_band_packed_pad_body(mesh, specs)
        fn = jax.jit(shard_map(packed, mesh=mesh, in_specs=in_specs,
                               out_specs=in_specs, check_vma=False))
        return fn.lower(*fields).as_text()

    n_off = lowered(None).count("transpose")
    n_on = lowered("1").count("transpose")
    assert n_on > n_off, (
        f"the flag changed nothing in the lowered body "
        f"({n_off} transposes off, {n_on} on) -- every bit-identity gate in "
        f"this file would then be vacuous")


def test_cache_key_carries_the_flag():
    """A traced-body cache keyed without the halo environment hands back a
    body built under the previous setting while the receipt reports the new
    one. The signature is the thing that stops it, so pin that it moves."""
    import os
    from legoesm.parallel.latlon_spmd import HALO_ENV_FLAGS, halo_env_signature
    assert ENV in HALO_ENV_FLAGS
    before = halo_env_signature()
    old = os.environ.get(ENV)
    try:
        os.environ[ENV] = "1"
        assert halo_env_signature() != before
    finally:
        if old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = old
    assert halo_env_signature() == before


def test_env_gate_accepts_the_two_documented_spellings():
    assert latlon_spmd._resolve_halo_level_leading("") is False
    assert latlon_spmd._resolve_halo_level_leading("0") is False
    assert latlon_spmd._resolve_halo_level_leading("1") is True
