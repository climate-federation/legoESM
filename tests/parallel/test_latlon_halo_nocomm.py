"""The lat-lon SPMD no-comm timing knob: it must break the answer, loudly.

``LEGOESM_LATLON_HALO_NOCOMM=1`` drops every lat-band halo collective so the
step's communication term can be measured as ``full - nocomm`` against an
otherwise identical program (the profiler on this stack does not record the
halo collectives).  It is a MEASUREMENT knob: it deliberately produces wrong
ghost rows, so the test asserts exactly that -- the padded field must CHANGE
when the knob is on.  If the knob were silently a no-op (e.g. the exchange
helper reverted to a plain ``ppermute``), this test fails.

Runs on host CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from functools import partial

from legoesm.parallel.latlon_spmd import (
    _resolve_halo_nocomm,
    make_latlon_band_packed_pad_body,
    pad_halo_latlon_band_spmd,
)
from legoesm.parallel.shard_map_compat import shard_map

N_DEV = 4
N_LAT = 16
N_LON = 8
HALO = 1


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def _pad(field, mesh):
    return np.asarray(pad_halo_latlon_band_spmd(mesh, halo=HALO)(field))


def test_resolver_off_on_and_typo():
    assert _resolve_halo_nocomm("") == "off"
    assert _resolve_halo_nocomm("0") == "off"
    assert _resolve_halo_nocomm("1") == "1"
    assert _resolve_halo_nocomm("wire") == "wire"
    for bad in ("true", "yes", "2", "01", " 1", "WIRE"):
        with pytest.raises(ValueError, match="LEGOESM_LATLON_HALO_NOCOMM"):
            _resolve_halo_nocomm(bad)


def test_nocomm_changes_the_interior_cut_ghost_rows(monkeypatch):
    """Off: band b's north ghost is band b+1's first row.  On: it is band b's
    OWN first row (the collective is gone), so the two arms must differ."""
    mesh = _mesh()
    rng = np.random.default_rng(4242)
    # Distinct per-row values so a wrong ghost row cannot coincide with a
    # right one.
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))

    monkeypatch.delenv("LEGOESM_LATLON_HALO_NOCOMM", raising=False)
    off = _pad(field, mesh)

    monkeypatch.setenv("LEGOESM_LATLON_HALO_NOCOMM", "1")
    on = _pad(field, mesh)

    assert off.shape == on.shape
    assert not np.allclose(off, on), (
        "no-comm arm is bit-identical to the real exchange -- the knob is "
        "inert, so any budget split measured with it is meaningless")

    # Name the exact rows: band 0 occupies rows [0, nl+2h) of the gathered
    # output; its north ghost is the last row of that block.
    nl = N_LAT // N_DEV
    block = nl + 2 * HALO
    band0_north_ghost_on = on[block - 1, HALO:HALO + N_LON]
    band0_own_south_edge = np.asarray(field)[0, :]
    band1_south_edge = np.asarray(field)[nl, :]
    np.testing.assert_allclose(band0_north_ghost_on, band0_own_south_edge,
                               rtol=0, atol=0)
    off_ghost = off[block - 1, HALO:HALO + N_LON]
    np.testing.assert_allclose(off_ghost, band1_south_edge, rtol=0, atol=0)


def test_default_environment_leaves_the_knob_off():
    assert os.environ.get("LEGOESM_LATLON_HALO_NOCOMM", "") in ("", "0")


def test_nocomm_reaches_the_packed_body_the_measurement_actually_runs():
    """The budget job runs the PACKED stage-entry exchange, not the basic
    band pad.  A knob that only reached the basic path would leave the
    measured path armed and the receipt meaningless, so assert the packed
    body responds too.
    """
    mesh = _mesh()
    from jax.sharding import PartitionSpec as P

    rng = np.random.default_rng(77)
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))
    specs = (("fold", HALO, False),)

    def run():
        body = make_latlon_band_packed_pad_body(mesh, specs)
        spec = P("lat", None)

        @partial(shard_map, mesh=mesh, in_specs=spec, out_specs=spec,
                 check_vma=False)
        def _ex(x):
            return body(x)[0]

        return np.asarray(_ex(field))

    os.environ.pop("LEGOESM_LATLON_HALO_NOCOMM", None)
    off = run()
    os.environ["LEGOESM_LATLON_HALO_NOCOMM"] = "1"
    try:
        on = run()
    finally:
        os.environ.pop("LEGOESM_LATLON_HALO_NOCOMM", None)

    assert off.shape == on.shape
    assert not np.allclose(off, on), (
        "the packed exchange is unaffected by the no-comm knob, so the "
        "budget receipt measured a path the knob never touched")


def test_ballast_resolver_and_bit_identity(monkeypatch):
    """The payload knob must scale bytes without touching the answer.

    Unlike the no-comm knob this one is a CONTROL: if it changed any value
    the payload slope it measures would be confounded by different
    arithmetic.
    """
    from legoesm.parallel.latlon_spmd import _resolve_halo_ballast

    assert _resolve_halo_ballast("") == 1
    assert _resolve_halo_ballast("1") == 1
    assert _resolve_halo_ballast("4") == 4
    for bad in ("0", "9", "01", "two", " 2", "-1"):
        with pytest.raises(ValueError, match="LEGOESM_LATLON_HALO_BALLAST"):
            _resolve_halo_ballast(bad)

    mesh = _mesh()
    rng = np.random.default_rng(2024)
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)))

    monkeypatch.delenv("LEGOESM_LATLON_HALO_BALLAST", raising=False)
    off = _pad(field, mesh)
    monkeypatch.setenv("LEGOESM_LATLON_HALO_BALLAST", "3")
    on = _pad(field, mesh)
    np.testing.assert_allclose(on, off, rtol=0, atol=0)


def _collective_bytes(mesh, field, specs):
    """Bytes moved by collective-permutes in the LOWERED packed body.

    A value-only test cannot see whether the payload multiplier actually
    reached the wire: XLA may rewrite slice(ppermute(concat(x, x))) back to
    ppermute(x), keeping every value identical while shipping the original
    bytes. Count the operand bytes instead.
    """
    import re
    from jax.sharding import PartitionSpec as P

    body = make_latlon_band_packed_pad_body(mesh, specs)
    spec = P("lat", None)

    @partial(shard_map, mesh=mesh, in_specs=spec, out_specs=spec,
             check_vma=False)
    def _ex(x):
        return body(x)[0]

    text = jax.jit(_ex).lower(field).compile().as_text()
    dt_bytes = {"f32": 4, "f64": 8, "s32": 4, "bf16": 2, "pred": 1}
    shape_re = re.compile(r"(f32|f64|s32|bf16|pred)\[([0-9,]*)\]")
    total = 0
    for line in text.splitlines():
        if "collective-permute" not in line or "=" not in line:
            continue
        m = shape_re.match(line.split("=", 1)[1].strip())
        if not m:
            continue
        elems = 1
        for tok in m.group(2).split(","):
            if tok.strip():
                elems *= int(tok)
        total += elems * dt_bytes[m.group(1)]
    return total


def test_ballast_actually_doubles_the_bytes_on_the_wire():
    mesh = _mesh()
    rng = np.random.default_rng(31337)
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)), dtype=jnp.float32)
    specs = (("fold", HALO, False),)

    os.environ.pop("LEGOESM_LATLON_HALO_BALLAST", None)
    one = _collective_bytes(mesh, field, specs)
    os.environ["LEGOESM_LATLON_HALO_BALLAST"] = "2"
    try:
        two = _collective_bytes(mesh, field, specs)
    finally:
        os.environ.pop("LEGOESM_LATLON_HALO_BALLAST", None)

    assert one > 0, "no collective-permute found in the lowered body"
    assert two == 2 * one, (
        f"payload multiplier did not reach the wire: {one} bytes at 1x, "
        f"{two} at 2x (expected {2 * one}). The compiler folded the extra "
        f"copy away, so every payload measurement taken with it is void")


def test_ballast_and_nocomm_compose_into_a_packing_only_arm():
    """Both knobs together must give an arm with the extra packing and NO
    collective -- the control that separates packing cost from wire time."""
    mesh = _mesh()
    rng = np.random.default_rng(4242)
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)), dtype=jnp.float32)
    specs = (("fold", HALO, False),)

    os.environ["LEGOESM_LATLON_HALO_BALLAST"] = "2"
    os.environ["LEGOESM_LATLON_HALO_NOCOMM"] = "1"
    try:
        both = _collective_bytes(mesh, field, specs)
    finally:
        os.environ.pop("LEGOESM_LATLON_HALO_BALLAST", None)
        os.environ.pop("LEGOESM_LATLON_HALO_NOCOMM", None)
    assert both == 0, (
        f"the packing-only control still moves {both} bytes; it cannot "
        f"isolate packing cost")


def test_wire_arm_keeps_the_collective_and_the_broken_trajectory():
    """The control that prices communication without a trajectory difference.

    Subtracting the no-communication arm from the full run does not isolate
    wire time: the two arms carry different fields from the second step on, so
    their local work can differ too, and a contamination that is constant is
    invisible to any drift check.  The ``wire`` arm runs the collective and
    then keeps the local rows anyway, so it answers exactly as the ``1`` arm
    does while paying the full wire cost -- the difference between the two is
    the collective and nothing else.

    Both halves are asserted, because either one alone is satisfiable by a
    broken implementation: a compiler that deletes the unused collective gives
    matching answers and no bytes, and an arm that forgot to discard the
    result gives the right bytes and the wrong (correct) answer.
    """
    from jax.sharding import PartitionSpec as P

    mesh = _mesh()
    rng = np.random.default_rng(20260821)
    field = jnp.asarray(rng.standard_normal((N_LAT, N_LON)), dtype=jnp.float32)
    specs = (("fold", HALO, False),)

    def values():
        body = make_latlon_band_packed_pad_body(mesh, specs)
        spec = P("lat", None)

        @partial(shard_map, mesh=mesh, in_specs=spec, out_specs=spec,
                 check_vma=False)
        def _ex(x):
            return body(x)[0]

        return np.asarray(_ex(field))

    def _with(setting, fn):
        if setting is None:
            os.environ.pop("LEGOESM_LATLON_HALO_NOCOMM", None)
        else:
            os.environ["LEGOESM_LATLON_HALO_NOCOMM"] = setting
        try:
            return fn()
        finally:
            os.environ.pop("LEGOESM_LATLON_HALO_NOCOMM", None)

    full_bytes = _with(None, lambda: _collective_bytes(mesh, field, specs))
    none_bytes = _with("1", lambda: _collective_bytes(mesh, field, specs))
    wire_bytes = _with("wire", lambda: _collective_bytes(mesh, field, specs))

    assert full_bytes > 0
    assert none_bytes == 0
    assert wire_bytes == full_bytes, (
        f"the wire arm moves {wire_bytes} bytes, the full run moves "
        f"{full_bytes}; it is not paying the communication it exists to "
        f"price, so the difference it anchors is not wire time")

    full_vals = _with(None, values)
    none_vals = _with("1", values)
    wire_vals = _with("wire", values)

    assert not np.allclose(full_vals, wire_vals), (
        "the wire arm answers like the full run, so it did not discard the "
        "exchange and shares no trajectory with the no-communication arm")
    np.testing.assert_array_equal(wire_vals, none_vals)
