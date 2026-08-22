"""Packing one array once when a caller hands it over twice.

The dycore's packed exchange epoch calls the band packer with the SAME
temperature array at two positions — a two-row fold pad and a one-row wall
pad — and the dycore says so in a comment: the duplicate buys uniform
per-field unpacking. A four-point payload curve at 128 GPUs measured 1.66 ms
per copy of the halo payload with a fixed term of zero, so bytes are the only
thing on that lane that costs anything and the second copy is pure waste.

``LEGOESM_LATLON_HALO_DEDUP=1`` packs each distinct array once at the deepest
halo any of its specs asks for and slices the shallower consumers out of the
received ghost. The rows are the SAME rows — this is a copy, not a
re-derivation, which is what separates it from the exchange-merging work that
a one-unit-in-the-last-place mismatch blocked.

Gates:
1. ``test_duplicate_field_bit_identical`` — the real shape of the dycore's
   call, one array at two depths with two different semantics, padded outputs
   bit-identical off and on, at 1, 2 and 4 devices so polar bands and interior
   cuts are both covered.
2. ``test_no_duplicate_still_works`` — the ordinary path, every field
   distinct, unchanged.
3. ``test_wire_payload_actually_shrinks`` — NON-VACUITY. Every other gate
   asserts something does NOT change and would stay green if the switch were
   never read. This one lowers both programs and requires the collective's
   operand to be strictly narrower with the switch on.
4. ``test_env_gate_dispatch_hardening`` — an unknown value raises.
5. ``test_band_too_thin_for_the_deepest_halo_raises`` — the nesting the shared
   pack relies on is stated as an error rather than left to a reshape.

MUTATION CHECKS RUN, both directions, because a symmetric copy-paste bug lives
in exactly the half nobody tested: flipping the SOUTH slice from last-rows to
first-rows, and flipping the NORTH slice from first to last, each fail
``test_duplicate_field_bit_identical`` at 2 and 4 devices and pass at 1 — the
right signature, since a single band has no neighbour to disagree with.

Runs on host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4`` or more).
"""
from __future__ import annotations

import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm.parallel import latlon_spmd
from legoesm.parallel.latlon_spmd import make_latlon_band_packed_pad_body
from legoesm.parallel.shard_map_compat import shard_map

ENV = "LEGOESM_LATLON_HALO_DEDUP"
N_LAT = 16
N_LON = 16
NLEV = 5          # odd, so nothing about the packing can be a coincidence


def _mesh(n_dev):
    if len(jax.devices()) < n_dev:
        pytest.skip(f"needs --xla_force_host_platform_device_count={n_dev}")
    return jax.sharding.Mesh(
        np.array(jax.devices()[:n_dev]), axis_names=("lat",))


def _dycore_shaped_call():
    """The dycore's own argument list: Bln, T, T, u, dp — T twice.

    The two temperature entries differ in BOTH depth and semantics, which is
    the case the deduplication has to get right; a version that only handled
    equal depths would pass a weaker test.
    """
    rng = np.random.default_rng(20260822)
    bln = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV + 1)))
    t = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
    u = jnp.asarray(rng.standard_normal((N_LAT, N_LON + 1, NLEV)))
    dp = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
    # Four INPUTS, five OUTPUTS: temperature feeds two of them. The sharing
    # cannot be found inside the traced body -- one array passed twice yields
    # two distinct tracers -- so the caller declares it.
    fields = (bln, t, u, dp)
    specs = (("fold", 1, False), ("fold", 2, False),
             ("wall", 1, 0.0, 0.0), ("wall", 1, 0.0, 0.0),
             ("wall", 1, 0.0, 0.0))
    sources = (0, 1, 1, 2, 3)
    return fields, specs, sources


def _run(monkeypatch, n_dev, value, fields, specs, sources=None):
    monkeypatch.delenv(ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(ENV, value)
    mesh = _mesh(n_dev)
    packed = make_latlon_band_packed_pad_body(mesh, specs, sources)
    in_specs = tuple(P("lat", *((None,) * (f.ndim - 1))) for f in fields)
    out_specs = tuple(P("lat", *((None,) * (fields[s].ndim - 1)))
                      for s in (sources if sources is not None
                                else range(len(fields))))
    fn = jax.jit(shard_map(packed, mesh=mesh, in_specs=in_specs,
                           out_specs=out_specs, check_vma=False))
    args = tuple(jax.device_put(f, NamedSharding(mesh, sp))
                 for f, sp in zip(fields, in_specs))
    return fn(*args)


@pytest.mark.parametrize("n_dev", [1, 2, 4])
def test_duplicate_field_bit_identical(monkeypatch, n_dev):
    fields, specs, sources = _dycore_shaped_call()
    base = _run(monkeypatch, n_dev, None, fields, specs, sources)
    off = _run(monkeypatch, n_dev, "0", fields, specs, sources)
    on = _run(monkeypatch, n_dev, "1", fields, specs, sources)
    assert len(base) == len(specs)
    for i, (b, o, n) in enumerate(zip(base, off, on)):
        b, o, n = (np.asarray(x) for x in (b, o, n))
        assert b.dtype == o.dtype == n.dtype, i
        assert b.shape == o.shape == n.shape, i
        # tobytes, not allclose: the deduplicated rows are the SAME rows, so
        # anything short of bit-identical means the slice took the wrong end.
        assert b.tobytes() == o.tobytes(), f"'0' differs from unset, field {i}"
        assert b.tobytes() == n.tobytes(), f"dedup moved field {i}"
    # Non-vacuity of the harness: the two temperature outputs really do have
    # different depths and different semantics, so they are not trivially equal.
    t_fold, t_wall = np.asarray(base[1]), np.asarray(base[2])
    assert t_fold.shape != t_wall.shape


@pytest.mark.parametrize("n_dev", [2, 4])
def test_no_duplicate_still_works(monkeypatch, n_dev):
    rng = np.random.default_rng(5)
    fields = (jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV))),
              jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV))),
              jnp.asarray(rng.standard_normal((N_LAT, N_LON))))
    specs = (("fold", 1, False), ("wall", 2, 1.5, -2.5), ("wall", 1, 0.0, 0.0))
    base = _run(monkeypatch, n_dev, None, fields, specs)
    on = _run(monkeypatch, n_dev, "1", fields, specs)
    for i, (b, n) in enumerate(zip(base, on)):
        assert np.asarray(b).tobytes() == np.asarray(n).tobytes(), i


def _collective_operand_width(text: str) -> int:
    """Total elements carried by the module's collective-permutes.

    The packed exchange sends one flat ``(1, w)`` row per dtype group per
    direction, so the widths in the lowered text ARE the wire payload.
    """
    widths = []
    for line in text.splitlines():
        if "collective_permute" not in line:
            continue
        # `... -> tensor<1x2160xf64>`: the packed row is (1, w), so w is the
        # second extent of the collective's own result type.
        found = re.findall(r"tensor<1x(\d+)x", line)
        if found:
            widths.append(int(found[-1]))
    return sum(widths)


def test_wire_payload_actually_shrinks(monkeypatch):
    """NON-VACUITY: the switch must reach the wire, not just compile.

    Everything else here asserts that a number does NOT move. If the switch
    were never read those gates would all pass while the change did nothing.
    """
    n_dev = 4
    fields, specs, sources = _dycore_shaped_call()
    mesh = _mesh(n_dev)
    in_specs = tuple(P("lat", *((None,) * (f.ndim - 1))) for f in fields)
    out_specs = tuple(P("lat", *((None,) * (fields[s].ndim - 1)))
                      for s in sources)

    def lowered(value):
        monkeypatch.delenv(ENV, raising=False)
        if value is not None:
            monkeypatch.setenv(ENV, value)
        packed = make_latlon_band_packed_pad_body(mesh, specs, sources)
        fn = jax.jit(shard_map(packed, mesh=mesh, in_specs=in_specs,
                               out_specs=out_specs, check_vma=False))
        return fn.lower(*fields).as_text()

    text_off, text_on = lowered(None), lowered("1")
    w_off, w_on = (_collective_operand_width(t) for t in (text_off, text_on))
    assert w_off > 0, ("found no collective operand widths to compare — the "
                       "pattern no longer matches this dialect, so this gate "
                       "is measuring nothing")
    assert w_on < w_off, (
        f"the wire did not shrink: {w_off} elements off, {w_on} on. Either "
        f"the switch is not read or the duplicate is still being packed")
    # The duplicate is one row of (n_lon, nlev) PER DIRECTION, and the count
    # above sums the north and south collectives, so the saving is twice that.
    assert w_off - w_on == 2 * N_LON * NLEV, (w_off, w_on)
    # And it is a real fraction of the epoch, not a rounding: the temperature
    # wall row is a sixth of what this epoch ships.
    assert 0.14 < (w_off - w_on) / w_off < 0.18, (w_off, w_on)


def test_band_too_thin_for_the_deepest_halo_raises(monkeypatch):
    """The nesting needs each band to own at least the deepest halo. Below
    that the exchange is no longer one hop and the shallower rows stop being
    a suffix of the deeper ones — which would be silently wrong, so it is an
    error instead."""
    n_dev = 4
    rng = np.random.default_rng(11)
    # 4 rows over 4 devices is 1 row each, against a depth-2 fold.
    fields = (jnp.asarray(rng.standard_normal((n_dev, N_LON, NLEV))),)
    specs = (("fold", 2, False),)
    with pytest.raises(ValueError, match="deepest halo"):
        _run(monkeypatch, n_dev, "1", fields, specs, (0,))


@pytest.mark.parametrize("bad", ["2", "true", "on", " 1", "yes", "-1"])
def test_env_gate_dispatch_hardening(bad):
    resolver = getattr(latlon_spmd, "_resolve_halo_dedup", None)
    if resolver is None:
        pytest.fail("no _resolve_halo_dedup in latlon_spmd")
    with pytest.raises(ValueError, match=ENV):
        resolver(bad)


def test_flag_is_in_the_signature():
    """A switch that changes the compiled program must reach the cache key."""
    assert ENV in latlon_spmd.HALO_ENV_FLAGS
