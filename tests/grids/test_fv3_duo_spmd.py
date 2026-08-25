"""Duo SPMD depth census: decode self-test, the measured census, the gate.

The census is the SPMD ring design's load-bearing number (every cross-face
read must land inside the gathered ring), so this file pins it and proves
the instrument can fail.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids.fv3_duo_spmd import (
    _self_test,
    assert_ring_width_covers,
    census_cross_face_depth,
)

N, NG = 12, 3


@pytest.fixture(scope="module")
def tab():
    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.grids.fv3_duo_halos import build_jax_duo_halo_tables
    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    return build_jax_duo_halo_tables(ctx["ectx"], ctx["gs6"], nq=1)


def test_census_matches_the_measured_depth(tab):
    """C48 measured 7 (jobs 9493433-38); the builders' strip arithmetic is
    n-independent (codex-read: neighbor maps are affine, halo widths come
    from ng/ngp), so N=12 at the same ng=3/ngp=4 must census the same.
    A different number here is a FINDING about n-dependence, not a
    tolerance to relax."""
    assert census_cross_face_depth(tab) == 7


def test_gate_refuses_a_narrow_ring(tab):
    with pytest.raises(ValueError, match="does not cover"):
        assert_ring_width_covers(tab, 7)   # == depth: no margin, refused
    assert assert_ring_width_covers(tab, 8) == 7


def test_decode_self_test_can_fail(tab):
    """Synthetic violation (non-vacuity): a layout whose idx LIES must be
    caught by the self-test -- otherwise a broken decode could bless a
    broken ring width."""
    class _Lying:
        shapes = tab.lay_a.shapes
        bases = tab.lay_a.bases
        total = tab.lay_a.total

        @staticmethod
        def idx(k, face, i0, j0):
            return tab.lay_a.idx(k, face, i0, j0) + 1   # off by one

    with pytest.raises(AssertionError, match="self-test"):
        _self_test(_Lying())


def test_census_refuses_an_unknown_op_family(tab):
    """A table op without 'dst' must be a hard error, never a skip."""
    from legoesm.grids.fv3_duo_spmd import _max_cross_depth

    class _Alien:
        pass

    with pytest.raises(TypeError, match="fail closed"):
        _max_cross_depth(_Alien(), tab.lay_a)


# ---- ring exchange: bitwise parity vs the certified exchange + poison ----
#
# Needs >= 6 devices: the sbatch wrapper sets
# XLA_FLAGS=--xla_force_host_platform_device_count=6 BEFORE jax imports.

def _devices_or_skip(n):
    import jax
    if len(jax.devices()) < n:
        pytest.skip(f"needs {n} devices (set "
                    f"xla_force_host_platform_device_count)")
    return jax.devices()[:n]


@pytest.mark.parametrize("n_dev", [2, 3, 6])
@pytest.mark.parametrize("stag", ["A", "B"])
def test_ring_exchange_matches_certified_bitwise(tab, n_dev, stag):
    """The ring exchange copies the same VALUES the certified exchange
    reads (borders are copies, own faces are originals), then runs the
    SAME table program -- own-face outputs must be BITWISE equal.  A
    tolerance here would hide an index shift."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface
    from legoesm.grids.fv3_duo_spmd import make_ring_ext_scalar_sixface

    devs = _devices_or_skip(n_dev)
    m = tab.n + 2 * tab.ng + (1 if stag == "B" else 0)
    rng = np.random.default_rng(3)
    f6 = jnp.asarray(rng.standard_normal((6, m, m)))
    ref = np.asarray(ext_scalar_sixface(f6, tab, stag))

    mesh = Mesh(np.array(devs), ("face",))
    fn, depth = make_ring_ext_scalar_sixface(tab, stag, mesh)
    assert depth == 7
    f6s = jax.device_put(f6, NamedSharding(mesh, P("face")))
    got = np.asarray(fn(f6s))
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), (
        f"ring exchange diverged: |d|max="
        f"{np.abs(got - ref).max():.3e}")


@pytest.mark.parametrize("stag", ["A", "B"])
def test_ring_exchange_poison_proves_no_out_of_ring_read(tab, stag):
    """poison=True fills every non-border, non-own cell with NaN; the
    output must STILL match the certified exchange bitwise.  One NaN in
    the output = a read outside the census's ring -- the design's failure
    mode, made loud (GLM: zeros are indistinguishable from real data)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface
    from legoesm.grids.fv3_duo_spmd import make_ring_ext_scalar_sixface

    devs = _devices_or_skip(6)
    m = tab.n + 2 * tab.ng + (1 if stag == "B" else 0)
    rng = np.random.default_rng(4)
    f6 = jnp.asarray(rng.standard_normal((6, m, m)))
    ref = np.asarray(ext_scalar_sixface(f6, tab, stag))

    mesh = Mesh(np.array(devs), ("face",))
    fn, _ = make_ring_ext_scalar_sixface(tab, stag, mesh, poison=True)
    f6s = jax.device_put(f6, NamedSharding(mesh, P("face")))
    got = np.asarray(fn(f6s))
    assert np.isfinite(got).all(), \
        "NaN reached the output: the tables read outside the ring"
    assert np.array_equal(got, ref)


# ---- vector ring exchange: bitwise parity + poison, D and C flows ----

def _vec_fixture(tab, grid, seed):
    ma = tab.n + 2 * tab.ng
    mb = ma + 1
    rng = np.random.default_rng(seed)
    if grid == "D":
        return (rng.standard_normal((6, ma, mb)),
                rng.standard_normal((6, mb, ma)))
    return (rng.standard_normal((6, mb, ma)),
            rng.standard_normal((6, ma, mb)))


@pytest.mark.parametrize("seed", [11, 23, 47])
@pytest.mark.parametrize("n_dev", [2, 3, 6])
@pytest.mark.parametrize("grid", ["D", "C"])
def test_ring_vector_matches_certified_bitwise(tab, n_dev, grid, seed):
    """Full 7-step vector flow under the ring exchange must equal the
    certified single-device flow BITWISE on own faces: other faces' ring
    c2l values are recomputed from their gathered u/v rings (same inputs,
    same op).  NaN slots (c2l band edges, pack_p1) must match too --
    array_equal with equal_nan pins the whole layout."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.fv3_duo_halos import (
        ext_vector_cgrid_sixface, ext_vector_dgrid_sixface)
    from legoesm.grids.fv3_duo_spmd import make_ring_ext_vector_sixface

    devs = _devices_or_skip(n_dev)
    u6, v6 = map(jnp.asarray, _vec_fixture(tab, grid, seed))
    flow = (ext_vector_dgrid_sixface if grid == "D"
            else ext_vector_cgrid_sixface)
    ru, rv = flow(u6, v6, tab)
    ru, rv = np.asarray(ru), np.asarray(rv)

    mesh = Mesh(np.array(devs), ("face",))
    fn, depth = make_ring_ext_vector_sixface(tab, grid, mesh)
    assert depth == 7
    sh = NamedSharding(mesh, P("face"))
    gu, gv = fn(jax.device_put(u6, sh), jax.device_put(v6, sh))
    gu, gv = np.asarray(gu), np.asarray(gv)
    assert np.array_equal(gu, ru, equal_nan=True), (
        f"u diverged: |d|max={np.nanmax(np.abs(gu - ru)):.3e}")
    assert np.array_equal(gv, rv, equal_nan=True), (
        f"v diverged: |d|max={np.nanmax(np.abs(gv - rv)):.3e}")


@pytest.mark.parametrize("grid", ["D", "C"])
def test_ring_vector_poison_proves_ring_sufficiency(tab, grid):
    """poison=True: non-ring, non-own cells are NaN.  The composed flow's
    EFFECTIVE cross-face footprint (strips + c2l-feeding-geo, GLM's
    compounding-depth hole) must fit inside width 8 -- a NaN escaping
    into a certified-finite own-face cell fails array_equal loudly."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    from legoesm.grids.fv3_duo_halos import (
        ext_vector_cgrid_sixface, ext_vector_dgrid_sixface)
    from legoesm.grids.fv3_duo_spmd import make_ring_ext_vector_sixface

    devs = _devices_or_skip(6)
    u6, v6 = map(jnp.asarray, _vec_fixture(tab, grid, 12))
    flow = (ext_vector_dgrid_sixface if grid == "D"
            else ext_vector_cgrid_sixface)
    ru, rv = flow(u6, v6, tab)

    mesh = Mesh(np.array(devs), ("face",))
    # Observability (codex): the comparison can only catch a wrong NaN
    # where the certified reference is FINITE -- pin that the reference
    # halo band genuinely is (its NaN slots are the structural c2l-band
    # ones only, a bounded count), so equal_nan cannot hide a defect.
    n_nan_u = int(np.isnan(np.asarray(ru)).sum())
    n_nan_v = int(np.isnan(np.asarray(rv)).sum())
    assert n_nan_u < np.asarray(ru).size // 4, "reference u mostly NaN"
    assert n_nan_v < np.asarray(rv).size // 4, "reference v mostly NaN"
    fn, _ = make_ring_ext_vector_sixface(tab, grid, mesh, poison=True)
    sh = NamedSharding(mesh, P("face"))
    gu, gv = fn(jax.device_put(u6, sh), jax.device_put(v6, sh))
    assert np.array_equal(np.asarray(gu), np.asarray(ru), equal_nan=True)
    assert np.array_equal(np.asarray(gv), np.asarray(rv), equal_nan=True)


def test_poison_instrument_can_fire(tab, monkeypatch):
    """NEGATIVE CONTROL (codex): with the border NARROWER than the tables
    read (mask width 4, gate bypassed by patching the mask builder), the
    poisoned exchange MUST diverge from the certified one -- proving the
    poison instrument can actually fail.  Without this, every poison
    pass above could be vacuous."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    import legoesm.grids.fv3_duo_spmd as spmd
    from legoesm.grids.fv3_duo_halos import ext_scalar_sixface

    devs = _devices_or_skip(6)
    m = tab.n + 2 * tab.ng
    rng = np.random.default_rng(5)
    f6 = jnp.asarray(rng.standard_normal((6, m, m)))
    ref = np.asarray(ext_scalar_sixface(f6, tab, "A"))

    real_mask = spmd._border_mask
    monkeypatch.setattr(spmd, "_border_mask",
                        lambda mm, w: real_mask(mm, 4))   # too narrow
    mesh = Mesh(np.array(devs), ("face",))
    fn, _ = spmd.make_ring_ext_scalar_sixface(tab, "A", mesh, poison=True)
    got = np.asarray(fn(jax.device_put(
        f6, NamedSharding(mesh, P("face")))))
    assert not np.array_equal(got, ref, equal_nan=True), (
        "narrow-border poison run still matched: the poison instrument "
        "cannot fire and every poison pass is vacuous")
