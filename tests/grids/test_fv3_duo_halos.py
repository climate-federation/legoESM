"""Certification of the JAX duo-grid halo lane against the NumPy fp64 lane.

Four gates per public routine (the NH JAX-mirror pattern,
``test_fv3_nh_core.py``):

1. **equivalence** vs the NumPy twin in ``fv3_native_gridstruct`` /
   ``fv3_native_ext_vector``, on the SAME inputs -- committed oracle
   fixtures where one exists (``fv3_extchain_oracle_c12.npz`` real
   duo-init fields, ``fv3_cornerlag_oracle.npz`` corner fields,
   ``fv3_duogrid_oracle_n2.npz`` k2e tables) and deterministic random
   fields otherwise.  Random beats smooth for an INDEX map: a smooth
   field hides a transposed or off-by-one index, random cannot.
2. **jit vs eager** as an ASSERTION, not a comment.
3. **guards**: uniform float32 and float64 are admitted, mixed floating
   dtypes raise ``TypeError``, and an unknown stagger/ring/key raises
   ``ValueError`` -- each rejection is paired with an admitted input.
4. ``jax.test_util.check_grads(order=2)``.

Plus two gates whose power does not depend on a tolerance:

* **single-valuedness after a barrier** -- after the blend, a shared
  edge must carry the same value seen from either face (up to the
  component sign map).  Asserted BITWISE.  It cannot catch a wrong
  neighbour map (both lanes read the same certified
  ``neighbor_index``), and that is stated rather than implied.
* **the barrier-1 slot exclusion** -- ``dyn_core.F90:856`` averages
  ``iq==1``, ``iq==4`` and ``iq>4`` only; slots 2 (``w``) and 3
  (``q_con``) must come back BITWISE unchanged while the others move.

Standing caveat on gate 4: every kernel in this module is LINEAR in its
field arguments (gathers, scatters, multiplication by grid-metric
constants), so an order-2 ``check_grads`` compares 0 against 0 and is a
weak test.  The strong gradient gate here is
``test_adjoint_identity_*``: the dot-product identity
``<J v, w> == <v, J^T w>``, which a wrong VJP fails.

WHEN BITWISE IS THE RIGHT EXPECTATION, AND WHEN IT IS NOT
---------------------------------------------------------
Learned from job 9400424, where six tests asserted bitwise jit-vs-eager
on paths that cannot deliver it.  XLA contracts ``x*y + z`` into an FMA
in the jitted lowering and not in the eager one, so any path containing
a SUM OF PRODUCTS diverges by a few ULP between the two.  The split is
structural, not empirical:

* **bitwise** -- paths that are pure index copies, optionally times a
  ``+-1`` constant: the four exchanges, the six ``fill_corners``,
  ``pack_p1``, ``write_{d,c}_strips``.  There is no arithmetic at all.
* **bitwise, and stable for a stateable reason** -- the two barriers.
  The blend is ``0.5 * (a + s*b)`` with ``s`` a ``+-1`` constant folded
  away at trace time, leaving a MUL OF A SUM.  FMA contracts a sum of
  products; there is no ``x*y + z`` here to contract, which is why
  these passed bitwise and are expected to keep doing so.
* **measured bound** -- everything containing ``sum_l w_l * v_l``: the
  k2e ring remap, the corner-region Lagrange fill, ``c2l`` (its
  ``a11*u1 + a12*v1``), the Cartesian projections, and every composed
  ``ext_*`` / ``geo_lattice_exchange`` built from them.

``_cmp`` is used for BOTH the JAX-vs-NumPy and the jit-vs-eager gate on
those paths, so the two hops are asserted the same way and each carries
its own provisional-tolerance marker awaiting the measurement run.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from legoesm.grids import fv3_duo_halos as jx  # noqa: E402

from tests.grids.fv3_gate_helpers import gate_scalar  # noqa: E402
from legoesm.grids import fv3_native_ext_vector as exv  # noqa: E402
from legoesm.grids import fv3_native_gridstruct as gsm  # noqa: E402
from legoesm.grids.fv3_native_halos import (  # noqa: E402
    compute_fv3_native_k2e,
    neighbor_index,
    neighbor_tiles,
)

N, NG = 12, 3
NQ = 2                      # barrier-1 tracer count -> 4+nq = 6 flux slots
FIX = Path(__file__).parent / "fixtures"

MA = N + 2 * NG             # cell axis
MB = N + 2 * NG + 1         # node axis
NPX = N + 1


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kinked_gs6():
    return [gsm.build_fv3_native_gridstruct(N, NG, tile=t)
            for t in range(1, 7)]


@pytest.fixture(scope="module")
def ectx(kinked_gs6):
    return exv.build_ext_context(N, NG, kinked_gs6)


@pytest.fixture(scope="module")
def tab(ectx, kinked_gs6):
    return jx.build_jax_duo_halo_tables(ectx, kinked_gs6, nq=NQ)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rnd(shape, seed):
    """Deterministic f64 field; ``shape`` may include the leading 6."""
    return np.asarray(
        np.random.default_rng(seed).standard_normal(shape), dtype=np.float64)


def _rnd6(shape2, seed):
    return _rnd((6,) + tuple(shape2), seed)


def _cmp(got, want, name, tol):
    """Max RELATIVE diff over finite slots, after requiring the
    non-finite masks to match EXACTLY -- asserted against ``tol`` in
    here (measure mode prints instead; the mask checks always raise).

    Three ways this refuses to hide a failure:

    * ``nanmax``-style reductions would hide a lane that produced NaN
      where the other produced a number, so the mask equality is
      asserted first;
    * an ALL-non-finite pair would otherwise compare vacuously (the
      2026-08-13 review caught exactly that hole here), so at least one
      finite slot is required;
    * the scale is ``max(1, max|want|)``, so a bound reads as relative
      on O(1)-and-larger fields and as absolute on tiny ones.
    """
    got = np.asarray(got)
    want = np.asarray(want)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    gm = ~np.isfinite(got)
    wm = ~np.isfinite(want)
    assert np.array_equal(gm, wm), (
        f"{name}: non-finite masks differ (jax {int(gm.sum())} slots, "
        f"numpy {int(wm.sum())} slots)")
    assert not gm.all(), (
        f"{name}: BOTH sides are entirely non-finite ({gm.size} slots) -- "
        f"this comparison would pass vacuously")
    scale = max(1.0, float(np.max(np.abs(want[~wm]))))
    rel = float(np.max(np.abs(got[~gm] - want[~gm]))) / scale
    return gate_scalar(name, rel, tol)


def _bitwise_equal(a, b):
    """Raw-word equality, NaN-insensitive (NaN payloads are irrelevant)."""
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def _finite_mask_scalar(fn, *args):
    """Fixed-cotangent scalar objective over the FINITE output slots.

    ``jnp.where(mask, out, 0.0)`` is used, and the choice is deliberate:
    the unselected branch is the CONSTANT ``0.0`` -- total and finite
    over every input -- so the select cannot manufacture a NaN, and the
    cotangent it sends back is ``where(mask, ct, 0)``.  The NaN slots
    that DO exist in this lane (``c2l`` outside its window, the
    ``pack_p1`` outer ring) are values, not local derivatives: every
    kernel here is linear with finite metric coefficients, so no NaN
    reaches a gradient.  ``lax.cond`` is not used anywhere -- there is
    no scalar predicate in halo machinery.
    """
    out = fn(*args)
    leaves = out if isinstance(out, tuple) else (out,)
    masks = [np.isfinite(np.asarray(x)) for x in leaves]
    cts = [_rnd(x.shape, 17 + i) * m
           for i, (x, m) in enumerate(zip(leaves, masks))]

    def obj(*a):
        o = fn(*a)
        o = o if isinstance(o, tuple) else (o,)
        acc = 0.0
        for x, m, c in zip(o, masks, cts):
            acc = acc + jnp.sum(jnp.where(m, x, 0.0) * c)
        return acc

    return obj


def _adjoint_residual(fn, primals, seed=3):
    """``|<J v, w> - <v, J^T w>| / scale`` for a (possibly affine) ``fn``.

    The real gradient gate for a linear operator: an order-2
    ``check_grads`` on a linear map compares 0 to 0, this does not.
    """
    vs = tuple(jnp.asarray(_rnd(np.asarray(p).shape, seed + i))
               for i, p in enumerate(primals))
    out, jvp = jax.jvp(fn, primals, vs)
    leaves = out if isinstance(out, tuple) else (out,)
    jvps = jvp if isinstance(jvp, tuple) else (jvp,)
    masks = [np.isfinite(np.asarray(x)) for x in leaves]
    ws = [jnp.asarray(_rnd(np.asarray(x).shape, 101 + i) * m)
          for i, (x, m) in enumerate(zip(leaves, masks))]
    lhs = sum(float(jnp.sum(jnp.where(m, j, 0.0) * w))
              for j, m, w in zip(jvps, masks, ws))
    _, vjp = jax.vjp(fn, *primals)
    cot = vjp(tuple(ws) if isinstance(out, tuple) else ws[0])
    rhs = sum(float(jnp.sum(c * v)) for c, v in zip(cot, vs))
    scale = max(abs(lhs), abs(rhs), 1.0)
    return abs(lhs - rhs) / scale


def _extchain():
    f = FIX / "fv3_extchain_oracle_c12.npz"
    if not f.exists():
        pytest.skip("extchain oracle fixture not generated "
                    "(scripts/cluster/fv3_native/extchain_oracle.sbatch)")
    return np.load(f, allow_pickle=False)


def _cornerlag():
    f = FIX / "fv3_cornerlag_oracle.npz"
    if not f.exists():
        pytest.skip("cornerlag oracle fixture not generated "
                    "(scripts/cluster/fv3_native/cornerlag_oracle.sbatch)")
    return np.load(f, allow_pickle=False)


# ---------------------------------------------------------------------------
# table build: the batching licence itself
# ---------------------------------------------------------------------------

def test_tables_build_and_record_the_layout_contract(tab):
    assert (tab.n, tab.ng, tab.npx, tab.nq) == (N, NG, NPX, NQ)
    assert tab.ngp == 4                     # upstream dg%bd%ng, DERIVED
    assert tab.k2e_nord == 2                # the resolved runtime order
    assert tab.vector_corner == "lagrange"
    # layout contract item 2: stagger shapes really are different, so the
    # face axis is stackable and the stagger axis is not
    assert tab.lay_a.shapes == ((MA, MA),)
    assert tab.lay_b.shapes == ((MB, MB),)
    assert tab.lay_c.shapes == ((MB, MA), (MA, MB))
    assert tab.lay_d.shapes == ((MA, MB), (MB, MA))


def test_independence_check_is_non_vacuous():
    """The batching licence must be able to FAIL.

    A synthetic group that reads a slot it also writes has to raise --
    otherwise every ``_check_batchable`` call in the builder is
    decoration.
    """
    jx._check_batchable("ok", [1, 2, 3], [4, 5, 6])          # no overlap
    with pytest.raises(ValueError, match="written and read"):
        jx._check_batchable("bad", [1, 2, 3], [3, 7, 8])
    jx._check_no_clobber("ok", [1, 2], [3, 4])
    with pytest.raises(ValueError, match="reordering is illegal"):
        jx._check_no_clobber("bad", [1, 2], [2, 9])


def test_dedup_last_keeps_the_last_write():
    dst, val = jx._dedup_last([5, 3, 5, 3, 7], [10, 20, 30, 40, 50])
    order = np.argsort(dst)
    assert list(np.asarray(dst)[order]) == [3, 5, 7]
    assert list(np.asarray(val)[order]) == [40, 30, 50]


def test_flat_layout_refuses_a_wrapping_subscript():
    lay = jx._FlatLayout((4, 5))
    assert lay.idx(0, 2, 3, 4) == (2 * 4 + 3) * 5 + 4
    with pytest.raises(IndexError, match="outside array"):
        lay.idx(0, 0, -1, 0)             # NumPy would WRAP this
    with pytest.raises(IndexError, match="outside array"):
        lay.idx(0, 0, 0, 5)


def test_corner_mode_env_is_refused(ectx, kinked_gs6, tab, monkeypatch):
    """The NumPy lane's non-faithful ``nearest`` corner mode is frozen at
    IMPORT there, so a lane built while it is set would silently
    disagree in the wedges: refuse loudly instead.

    Non-vacuous by construction: the ``tab`` fixture is the SAME call
    with the variable unset, and it built."""
    assert tab is not None
    monkeypatch.setenv("LEGOESM_DUO_CORNER_MODE", "nearest")
    with pytest.raises(ValueError, match="CORNER_MODE"):
        jx.build_jax_duo_halo_tables(ectx, kinked_gs6, nq=NQ)


def test_tables_reject_a_mismatched_gridstruct(ectx, kinked_gs6):
    other = [dict(gs) for gs in kinked_gs6]
    other[3]["n"] = N + 2
    with pytest.raises(ValueError, match="ext context"):
        jx.build_jax_duo_halo_tables(ectx, other, nq=NQ)
    # and a gridstruct whose metrics are not the context's
    other = [dict(gs) for gs in kinked_gs6]
    other[1]["dx"] = np.asarray(other[1]["dx"]) + 1.0
    with pytest.raises(ValueError, match="different gridstruct"):
        jx.build_jax_duo_halo_tables(ectx, other, nq=NQ)


def _k2e_record_map(ij, loc, coef, swap=False):
    """``{(i, j): (loc, coef)}`` -- order- and column-convention free."""
    out = {}
    for (a, b), lv, cw in zip(ij, loc, coef):
        key = (int(b), int(a)) if swap else (int(a), int(b))
        assert key not in out, key
        out[key] = (int(lv), tuple(float(x) for x in cw))
    return out


def test_k2e_tables_are_the_pinned_nord2_oracle_tables():
    """The lane bakes in whatever ``compute_fv3_native_k2e`` returns, so
    pin THAT against the committed nord=2 oracle fixture -- by RECORD
    CONTENT, not by row order or column order.

    The first version of this test compared the ``_ij`` arrays
    row-by-row and failed (job 9400424) on what looked like an
    ``(i, j) -> (j, i)`` swap.  Measured on the committed fixture: all
    six families' ``_ij`` are sorted by COLUMN 1, while
    ``compute_fv3_native_k2e`` documents its records as "sorted by
    (i, j)" and emits them sorted by column 0.  The two therefore
    disagree in row order AND, pairwise, in column order -- a
    convention difference between the fixture writer and the generator,
    not a numeric one.  No existing test pinned it: the neighbouring
    ``test_k2e_tables_mirror_vs_authoritative`` filters on
    ``startswith("k2e_") or "coef" in k or "loc" in k``, which excludes
    every ``*_ij`` key.

    So the assertion is made convention-agnostic and the convention is
    REPORTED: the record->value map must agree either directly or under
    the column swap, the same way for every family.

    The SECOND version then failed too (job 9401521), and for a third
    reason: it compared the record maps with ``==``, which compares the
    float64 Lagrange coefficients exactly.  120 records were
    bit-identical and the rest differed in the last bit or two
    (0.6981966520583825 vs ...27), because the fixture's weights come
    from the Fortran ``tools/global_grid`` while
    ``compute_fv3_native_k2e`` recomputes the same formula in Python.
    Comparing floats with ``==`` is forbidden for exactly this reason.
    The gate now splits the comparison by what each part means: keys and
    the integer source level EXACTLY (a wrong cell or ring is a defect),
    coefficients against a measured bound.
    """
    f = FIX / "fv3_duogrid_oracle_n2.npz"
    if not f.exists():
        pytest.skip("nord-2 duogrid oracle fixture not present")
    d = np.load(f, allow_pickle=False)
    assert int(d["c12_k2e_nord"]) == 2
    got = compute_fv3_native_k2e(N, remap_ng=NG, k2e_nord=2)
    # SCOPE: A and B only -- the families this lane consumes.  Their
    # record maps are invariant under the column swap (measured on the
    # committed fixture and asserted in the companion test), so the
    # comparison is convention-free and needs no guess.  The staggered
    # families are deliberately NOT compared here: the swap maps an
    # x-face family onto a y-face one, so pinning them means first
    # deciding whether the fixture's CX is the generator's CX or its CY
    # -- a hop-A question about the NumPy lane's own tables, and this
    # lane refuses those staggers outright.
    for fam in ("A", "B"):
        mine = _k2e_record_map(got[f"{fam}_ij"], got[f"{fam}_loc"],
                               got[f"{fam}_coef"])
        theirs = _k2e_record_map(d[f"c12_{fam}_ij"], d[f"c12_{fam}_loc"],
                                 d[f"c12_{fam}_coef"])
        assert len(mine) == len(theirs), (fam, len(mine), len(theirs))
        # KEYS and the integer source level are compared EXACTLY -- a
        # wrong cell or a wrong ring is a defect, never a rounding
        # difference.
        assert set(mine) == set(theirs), (
            f"{fam}: the generator and the fixture disagree about WHICH "
            f"halo cells are remapped, which is a real defect and not a "
            f"convention difference")
        for key in mine:
            assert mine[key][0] == theirs[key][0], (
                f"{fam} {key}: source level {mine[key][0]} vs "
                f"{theirs[key][0]}")
        # COEFFICIENTS are float64 and must NOT be compared with ==.
        # The fixture's Lagrange weights come from the Fortran
        # tools/global_grid; compute_fv3_native_k2e recomputes the same
        # formula in Python, so the two differ in the last bit or two.
        # Job 9401521 measured, over both families: max |delta| =
        # 2.2e-16 on weights of order 0.7 / 0.3, i.e. 1-2 ULP, with 120
        # of the records bit-identical.  An exact-equality assertion
        # here reported a "content difference" that was pure rounding.
        # MEASURED (job 9425294 sweep): max weight delta 9.437e-16 (B; A 4.996e-16).
        # Job 9401521 measured 2.2e-16 on the same fixture -- the delta moves
        # with the build's libm, which is why the earlier bound carried x ~45
        # headroom.  Bound = this sweep's measured x 10 = 9.5e-15.
        # Re-measure before tightening further.
        cmax = max(abs(a - b)
                   for key in mine
                   for a, b in zip(mine[key][1], theirs[key][1]))
        if os.environ.get("LEGOESM_FV3_TOL_MEASURE") == "1":
            print(f"TOLMEASURE 'k2e coef {fam}': rel {cmax:.3e} "
                  f"(bound 9.5e-15, quantity max abs Lagrange-weight "
                  f"delta vs the Fortran fixture)", flush=True)
            continue
        assert cmax < 9.5e-15, (
            f"{fam}: Lagrange coefficients differ by {cmax:.3e}, far "
            f"above the 2.2e-16 rounding floor measured for this "
            f"fixture -- that is a formula difference, not rounding")
    # partition of unity, independent of the fixture entirely.  This is
    # the tolerance-independent half of the gate: a Lagrange stencil
    # whose weights do not sum to 1 is wrong no matter what the fixture
    # says, and no rounding difference can make it pass.
    for fam in ("A", "B"):
        assert np.abs(got[f"{fam}_coef"].sum(axis=1) - 1.0).max() < 1e-10


def test_k2e_column_convention_is_inert_for_the_staggers_this_lane_uses():
    """Why the convention above cannot reach the JAX lane.

    The lane consumes only the A and B families, and their record->value
    maps are INVARIANT under the column swap, so the ring classification
    in ``k2e_remap_halo_rings`` reads the same table either way.  (The
    staggered families are not invariant -- and the lane refuses them.)

    Generator-side only: no fixture, no convention assumption.
    """
    got = compute_fv3_native_k2e(N, remap_ng=NG, k2e_nord=2)
    for fam in ("A", "B"):
        direct = _k2e_record_map(got[f"{fam}_ij"], got[f"{fam}_loc"],
                                 got[f"{fam}_coef"])
        swapped = _k2e_record_map(got[f"{fam}_ij"], got[f"{fam}_loc"],
                                  got[f"{fam}_coef"], swap=True)
        assert direct == swapped, f"{fam} is NOT swap-invariant"
    # non-vacuity: the staggered families must NOT be swap-invariant,
    # otherwise the check above is trivially true for any table
    for fam in ("CX", "DY"):
        direct = _k2e_record_map(got[f"{fam}_ij"], got[f"{fam}_loc"],
                                 got[f"{fam}_coef"])
        swapped = _k2e_record_map(got[f"{fam}_ij"], got[f"{fam}_loc"],
                                  got[f"{fam}_coef"], swap=True)
        assert direct != swapped, f"{fam} is unexpectedly swap-invariant"


# ---------------------------------------------------------------------------
# gate 1 + 2: exchanges -- JAX vs NumPy, and jit vs eager
# ---------------------------------------------------------------------------

def test_exchange_agrid_scalar_matches_numpy_and_jit(tab):
    f6 = _rnd6((MA, MA), 0)
    ref = [np.array(a, copy=True) for a in f6]
    for t in range(1, 7):
        gsm.exchange_agrid_scalar_halos(ref, t, N, NG)
    got = jx.exchange_agrid_scalar_halos(jnp.asarray(f6), tab)
    # a pure index copy: bitwise is the only defensible expectation
    assert _bitwise_equal(got, np.stack(ref)), "agrid strips/corners"
    got_j = jx.exchange_agrid_scalar_halos_jit(jnp.asarray(f6), tab, "stepper")
    assert _bitwise_equal(got_j, got), "jit vs eager"
    # non-vacuity: the exchange must have CHANGED the halo
    assert not _bitwise_equal(got, f6)


def test_exchange_bgrid_scalar_matches_numpy_and_jit(tab):
    f6 = _rnd6((MB, MB), 1)
    ref = [np.array(a, copy=True) for a in f6]
    for t in range(1, 7):
        gsm.exchange_bgrid_scalar_halos(ref, t, N, NG)
    got = jx.exchange_bgrid_scalar_halos(jnp.asarray(f6), tab)
    assert _bitwise_equal(got, np.stack(ref))
    assert _bitwise_equal(
        jx.exchange_bgrid_scalar_halos_jit(jnp.asarray(f6), tab), got)
    assert not _bitwise_equal(got, f6)


def test_exchange_cgrid_vector_matches_numpy_and_jit(tab):
    uc6 = _rnd6((MB, MA), 2)
    vc6 = _rnd6((MA, MB), 3)
    ru = [np.array(a, copy=True) for a in uc6]
    rv = [np.array(a, copy=True) for a in vc6]
    for t in range(1, 7):
        gsm.exchange_cgrid_vector_halos(ru, rv, t, N, NG)
    gu, gv = jx.exchange_cgrid_vector_halos(jnp.asarray(uc6),
                                            jnp.asarray(vc6), tab)
    assert _bitwise_equal(gu, np.stack(ru)), "cgrid uc"
    assert _bitwise_equal(gv, np.stack(rv)), "cgrid vc"
    ju, jv = jx.exchange_cgrid_vector_halos_jit(jnp.asarray(uc6),
                                                jnp.asarray(vc6), tab)
    assert _bitwise_equal(ju, gu) and _bitwise_equal(jv, gv)
    assert not _bitwise_equal(gu, uc6)


def test_exchange_dgrid_vector_matches_numpy_and_jit(tab):
    u6 = _rnd6((MA, MB), 4)
    v6 = _rnd6((MB, MA), 5)
    ru = [np.array(a, copy=True) for a in u6]
    rv = [np.array(a, copy=True) for a in v6]
    for t in range(1, 7):
        gsm.exchange_dgrid_vector_halos(ru, rv, t, N, NG)
    gu, gv = jx.exchange_dgrid_vector_halos(jnp.asarray(u6),
                                            jnp.asarray(v6), tab)
    assert _bitwise_equal(gu, np.stack(ru)), "dgrid u"
    assert _bitwise_equal(gv, np.stack(rv)), "dgrid v"
    ju, jv = jx.exchange_dgrid_vector_halos_jit(jnp.asarray(u6),
                                                jnp.asarray(v6), tab)
    assert _bitwise_equal(ju, gu) and _bitwise_equal(jv, gv)
    assert not _bitwise_equal(gv, v6)


@pytest.mark.parametrize("stag,shape,ring", [
    ("A", (MA, MA), "stepper"),
    ("B", (MB, MB), "stepper"),
])
def test_k2e_remap_matches_numpy_and_jit(tab, stag, shape, ring):
    f6 = _rnd6(shape, 6)
    ref = [np.array(a, copy=True) for a in f6]
    gsm.k2e_remap_halo_rings(ref, stag, N, NG, k2e_nord=tab.k2e_nord)
    got = jx.k2e_remap_halo_rings(jnp.asarray(f6), tab, stag, ring)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), both stags; bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got, np.stack(ref), f"k2e[{stag}]", 1e-15)
    got_j = jx.k2e_remap_halo_rings_jit(jnp.asarray(f6), tab, stag, ring)
    # weighted sum -> FMA contraction differs between the jitted and the
    # eager lowering; a bound, not bitwise (see the module docstring)
    # MEASURED (job 9425294 sweep): worst B 1.333e-16; bound = measured x 10 =
    # 1.4e-15.
    _cmp(got_j, got, f"k2e[{stag}] jit vs eager", 1.4e-15)
    assert not _bitwise_equal(got, f6)


def test_k2e_remap_geo_ring_matches_numpy(tab):
    m4 = N + 2 * tab.ngp
    f6 = _rnd6((m4, m4), 7)
    ref = [np.array(a, copy=True) for a in f6]
    gsm.k2e_remap_halo_rings(ref, "A", N, tab.ngp, k2e_nord=tab.k2e_nord)
    got = jx.k2e_remap_halo_rings(jnp.asarray(f6), tab, "A", "geo")
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got, np.stack(ref), "k2e[A,geo]", 1e-15)


# ---------------------------------------------------------------------------
# gate 1 + 2: the six fill_corners twins
# ---------------------------------------------------------------------------

def _np_fill_single(fn, arr6, npx, ng):
    out = [np.array(a, copy=True) for a in arr6]
    for a in out:
        fn(gsm.fort(a, 1 - ng, 1 - ng), npx, ng)
    return np.stack(out)


def _np_fill_pair(fn, x6, y6, npx, ng, sign):
    xo = [np.array(a, copy=True) for a in x6]
    yo = [np.array(a, copy=True) for a in y6]
    for a, b in zip(xo, yo):
        fn(gsm.fort(a, 1 - ng, 1 - ng), gsm.fort(b, 1 - ng, 1 - ng),
           npx, ng, sign)
    return np.stack(xo), np.stack(yo)


def test_fill_corners_scalar_twins_match_numpy(tab):
    q6 = _rnd6((MB, MB), 8)
    ref = _np_fill_single(gsm._fill_corners_bgrid_x, q6, NPX, NG)
    got = jx.fill_corners_bgrid_x(jnp.asarray(q6), tab)
    assert _bitwise_equal(got, ref), "bgrid_x"
    assert _bitwise_equal(jx.fill_corners_bgrid_x_jit(jnp.asarray(q6), tab),
                          got)

    c6 = _rnd6((MA, MA), 9)
    for jfn, nfn, label in (
            (jx.fill_corners_agrid_x, gsm.fill_corners_agrid_x, "agrid_x"),
            (jx.fill_corners_agrid_y, gsm.fill_corners_agrid_y, "agrid_y")):
        ref = _np_fill_single(nfn, c6, NPX, NG)
        got = jfn(jnp.asarray(c6), tab)
        assert _bitwise_equal(got, ref), label
    assert not _bitwise_equal(jx.fill_corners_agrid_x(jnp.asarray(c6), tab),
                              c6)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_fill_corners_pair_twins_match_numpy(tab, sign):
    for jfn, nfn, xs, ys, label in (
            (jx.fill_corners_dgrid, gsm._fill_corners_dgrid,
             (MA, MB), (MB, MA), "dgrid"),
            (jx.fill_corners_cgrid, gsm._fill_corners_cgrid,
             (MB, MA), (MA, MB), "cgrid"),
            (jx.fill_corners_agrid_pair, gsm._fill_corners_agrid_pair,
             (MA, MA), (MA, MA), "agrid_pair")):
        x6 = _rnd6(xs, 10)
        y6 = _rnd6(ys, 11)
        rx, ry = _np_fill_pair(nfn, x6, y6, NPX, NG, sign)
        gx, gy = jfn(jnp.asarray(x6), jnp.asarray(y6), tab, sign)
        assert _bitwise_equal(gx, rx), (label, "X", sign)
        assert _bitwise_equal(gy, ry), (label, "Y", sign)


def test_fill_corners_jit_factory_rejects_a_foreign_function():
    with pytest.raises(ValueError, match="not one of the six"):
        jx.make_fill_corners_jit(jx.pack_p1)
    assert jx.make_fill_corners_jit(jx.fill_corners_dgrid) is not None


# ---------------------------------------------------------------------------
# gate 1: corner-region Lagrange fill, against the ORACLE fields
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,fld,stagger", [
    ("a3", "fld_00", (0, 0)),
    ("b3", "fld_11", (1, 1)),
    ("du3", "fld_01", (0, 1)),
    ("dv3", "fld_10", (1, 0)),
])
def test_corner_lagrange_fill_matches_numpy(tab, ectx, key, fld, stagger):
    """The oracle's own corner-fill input fields, replicated on all six
    faces (each face has its OWN operator, so this exercises six
    different weight sets on identical data)."""
    d = _cornerlag()
    assert (int(d["n"]), int(d["ng"])) == (N, NG)
    base = np.asarray(d[fld], dtype=np.float64)
    f6 = np.stack([base.copy() for _ in range(6)])
    ref = []
    for t in range(6):
        a = base.copy()
        op = ectx[f"corner_{key}"][t]
        assert (op.istag, op.jstag) == stagger
        op.fill(a)
        ref.append(a)
    got = jx.corner_lagrange_fill(jnp.asarray(f6), tab, key)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), every key; bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got, np.stack(ref), f"corner[{key}]", 1e-15)
    got_j = jx.corner_lagrange_fill_jit(jnp.asarray(f6), tab, key)
    # weighted sum -> FMA contraction differs between the jitted and the
    # eager lowering; a bound, not bitwise (see the module docstring)
    # MEASURED (job 9425294 sweep): worst a3 1.085e-14; bound = measured x 10 =
    # 1.1e-13.
    _cmp(got_j, got, f"corner[{key}] jit vs eager", 1.1e-13)
    # non-vacuity: the fill must move the 36 wedge slots per face
    assert not _bitwise_equal(got, f6)


def test_corner_lagrange_preserves_a_constant(tab):
    """Lagrange partition of unity -- a tolerance-independent invariant
    that does not need the NumPy lane at all."""
    f6 = jnp.full((6, MA, MA), 7.25, dtype=jnp.float64)
    got = jx.corner_lagrange_fill(f6, tab, "a3")
    # MEASURED (job 9425294 sweep): 1.137e-13 abs on the 7.25 field; bound = measured x 10 =
    # 1.2e-12.
    gate_scalar("corner lagrange constant",
                float(jnp.max(jnp.abs(got - 7.25))), 1.2e-12,
                quantity="max abs deviation from the constant 7.25")


def test_corner_lagrange_diagonal_is_the_average_of_two_directions(tab,
                                                                  ectx):
    """The claim the batched shape rests on, tested directly: a diagonal
    wedge slot equals 0.5*(X-fill + Y-fill) evaluated on the INCOMING
    field, i.e. it does not see the six directional fills."""
    f6 = _rnd6((MA, MA), 12)
    got = np.asarray(jx.corner_lagrange_fill(jnp.asarray(f6), tab, "a3"))
    op = ectx["corner_a3"][0]
    lo = 1 - NG
    ie = je = N                                   # istag = jstag = 0
    for (i_t, j_t, d1, d2) in ((ie + 1, je + 1, "X+", "Y+"),
                               (ie + 2, je + 2, "X+", "Y+"),
                               (ie + 3, je + 3, "X+", "Y+")):
        wx, sx = op.weights(i_t, j_t, d1)
        wy, sy = op.weights(i_t, j_t, d2)
        vx = sum(w * f6[0][s - lo, j_t - lo] for w, s in zip(wx, sx))
        vy = sum(w * f6[0][i_t - lo, s - lo] for w, s in zip(wy, sy))
        want = 0.5 * (vx + vy)
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), all probed wedge slots; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        gate_scalar(f"corner diagonal average ({i_t},{j_t})",
                    abs(got[0][i_t - lo, j_t - lo] - want), 1e-15,
                    quantity="abs diagonal-vs-average agreement")


# ---------------------------------------------------------------------------
# gate 1 + 2: c2l, packing, projections, strip write-back
# ---------------------------------------------------------------------------

def _c2l_window_report(arr, win, label):
    """Non-finite counts inside the written window and in the scratch."""
    a = np.asarray(arr)
    sc = np.ones(a.shape, dtype=bool)
    sc[:, win, win] = False
    inside = int((~np.isfinite(a[:, win, win])).sum())
    outside = int(np.isfinite(a[sc]).sum())
    return (f"{label}: {inside} non-finite of {a[:, win, win].size} inside "
            f"the written window, {outside} finite of {int(sc.sum())} in "
            f"the scratch region"), inside, outside, sc


def test_c2l_ord2_face_matches_numpy_and_jit(tab, ectx):
    """VERDICT on the NaN region: (a) legitimate untouched scratch.

    ``c2l_ord2`` with ``do_halo=.true.`` (fv_grid_utils.F90:2547-2628) is
    defined on ONE ring around the compute domain -- Fortran
    ``is-1..ie+1`` -- and the NumPy lane starts from
    ``np.full(..., np.nan)`` and writes only that window.  At C12 that is
    a 14x14 block of an 18x18 face, so 40% of every output array is
    scratch the routine never defines, and it sits in the outer rings,
    which is where the pytest array repr samples.

    CORRECTION to the first reading of the job-9400424 failure: the NaN
    slots are NOT what broke the old assertion.  ``_bitwise_equal``
    calls ``np.array_equal(..., equal_nan=True)``, so NaNs in matching
    positions already compare equal, and the two lanes' NaN masks DO
    match (``_cmp`` asserts that, and it passed).  What failed was the
    FINITE part: ``a11*u1 + a12*v1`` is a sum of products, so the jitted
    lowering contracts it into an FMA and the eager one does not.  This
    failure belongs to the same class as the other six, and the
    "both sides all-NaN" in the report is the truncated array repr
    sampling the scratch corners.

    The NaN region is nevertheless worth pinning, and ``_cmp`` had a
    real vacuity hole (an all-non-finite pair compared equal), so the
    region is split and each half gets the assertion that can actually
    discriminate:

    * the written window must be entirely FINITE on BOTH lanes -- this
      is the gate that fires if reading (b) is true and ``c2l`` really
      is producing NaN where numbers belong;
    * the scratch must be entirely NON-FINITE on BOTH lanes -- so if the
      written window ever moves, the test fails instead of quietly
      comparing fewer slots;
    * values are compared on the window only.
    """
    u6 = _rnd6((MA, MB), 13)
    v6 = _rnd6((MB, MA), 14)
    ru, rv = [], []
    for t in range(6):
        a, b = exv.c2l_ord2_face(u6[t], v6[t], ectx["dx6"][t],
                                 ectx["dy6"][t], ectx["amat6"][t], N, NG)
        ru.append(a)
        rv.append(b)
    ga, gb = jx.c2l_ord2_face(jnp.asarray(u6), jnp.asarray(v6), tab)
    win = slice(tab.c2l_s, tab.c2l_e + 1)
    assert (tab.c2l_e - tab.c2l_s + 1) == N + 2, (tab.c2l_s, tab.c2l_e)
    ja, jb = jx.c2l_ord2_face_jit(jnp.asarray(u6), jnp.asarray(v6), tab)

    for label, g, r, j in (("ua", ga, np.stack(ru), ja),
                           ("va", gb, np.stack(rv), jb)):
        for lane, arr in (("jax", g), ("numpy", r)):
            msg, inside, outside, sc = _c2l_window_report(
                arr, win, f"c2l {label} [{lane}]")
            assert inside == 0, msg      # reading (b) would fire here
            assert outside == 0, msg     # the window must not have moved
            assert sc.sum() > 0, msg     # and scratch must exist at all
        gw = np.asarray(g)[:, win, win]
        rw = r[:, win, win]
        # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise), both orders; bound =
        # 1e-15 eps guard (measured exactly 0.0).
        _cmp(gw, rw, f"c2l {label} window", 1e-15)
        # a11*u1 + a12*v1 is a sum of products -> FMA contraction, so the
        # jit gate is a bound on the window, not bitwise on the array
        # MEASURED (job 9425294 sweep): worst va 2.983e-16; bound = measured x 10 =
        # 3.0e-15.
        _cmp(np.asarray(j)[:, win, win], gw,
                    f"c2l {label} jit vs eager", 3.0e-15)


def test_c2l_ord2_cgrid_face_matches_numpy(tab, ectx):
    uc6 = _rnd6((MB, MA), 15)
    vc6 = _rnd6((MA, MB), 16)
    ru, rv = [], []
    for t in range(6):
        a, b = exv.c2l_ord2_cgrid_face(uc6[t], vc6[t], ectx["dx6"][t],
                                       ectx["dy6"][t], ectx["amat6"][t],
                                       N, NG)
        ru.append(a)
        rv.append(b)
    ga, gb = jx.c2l_ord2_cgrid_face(jnp.asarray(uc6), jnp.asarray(vc6), tab)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(ga, np.stack(ru), "c2l_cgrid ua", 1e-15)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(gb, np.stack(rv), "c2l_cgrid va", 1e-15)


def test_pack_p1_matches_numpy(tab):
    x6 = _rnd6((MA, MA), 17)
    ref = np.stack([exv._pack_p1(x6[t], N, NG) for t in range(6)])
    got = jx.pack_p1(jnp.asarray(x6), tab)
    assert _bitwise_equal(got, ref)
    assert _bitwise_equal(jx.pack_p1_jit(jnp.asarray(x6), tab), got)


def test_projections_match_numpy_and_jit(tab, ectx):
    m4 = N + 2 * tab.ngp
    ug = _rnd6((m4, m4), 18)
    vg = _rnd6((m4, m4), 19)
    rd = [exv._a2d_project(ug[t], vg[t], t, ectx) for t in range(6)]
    gd_u, gd_v = jx.a2d_project(jnp.asarray(ug), jnp.asarray(vg), tab)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(gd_u, np.stack([r[0] for r in rd]), "a2d ud", 1e-15)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(gd_v, np.stack([r[1] for r in rd]), "a2d vd", 1e-15)
    rc = [exv._a2c_project(ug[t], vg[t], t, ectx) for t in range(6)]
    gc_u, gc_v = jx.a2c_project(jnp.asarray(ug), jnp.asarray(vg), tab)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(gc_u, np.stack([r[0] for r in rc]), "a2c uc", 1e-15)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(gc_v, np.stack([r[1] for r in rc]), "a2c vc", 1e-15)
    ju, jv = jx.a2d_project_jit(jnp.asarray(ug), jnp.asarray(vg), tab)
    # weighted sum -> FMA contraction differs between the jitted and the
    # eager lowering; a bound, not bitwise (see the module docstring)
    # MEASURED (job 9425294 sweep): 1.755e-16; bound = measured x 10 =
    # 1.8e-15.
    _cmp(ju, gd_u, "a2d ud jit vs eager", 1.8e-15)
    # MEASURED (job 9425294 sweep): 1.936e-16; bound = measured x 10 =
    # 2.0e-15.
    _cmp(jv, gd_v, "a2d vd jit vs eager", 2.0e-15)


def test_write_strips_match_numpy_including_the_overwrite_order(tab):
    m4 = N + 2 * tab.ngp
    u6 = _rnd6((MA, MB), 20)
    v6 = _rnd6((MB, MA), 21)
    ud4 = _rnd6((m4, m4 - 1), 22)
    vd4 = _rnd6((m4 - 1, m4), 23)
    ru = [np.array(a, copy=True) for a in u6]
    rv = [np.array(a, copy=True) for a in v6]
    for t in range(6):
        exv._write_d_strips(ru[t], rv[t], ud4[t], vd4[t], N, NG)
    gu, gv = jx.write_d_strips(jnp.asarray(u6), jnp.asarray(v6),
                               jnp.asarray(ud4), jnp.asarray(vd4), tab)
    # last-wins on the S/N-then-W/E overlap: bitwise or the dedup is wrong
    assert _bitwise_equal(gu, np.stack(ru)), "write_d_strips u"
    assert _bitwise_equal(gv, np.stack(rv)), "write_d_strips v"

    uc6 = _rnd6((MB, MA), 24)
    vc6 = _rnd6((MA, MB), 25)
    uc4 = _rnd6((m4 - 1, m4), 26)
    vc4 = _rnd6((m4, m4 - 1), 27)
    rcu = [np.array(a, copy=True) for a in uc6]
    rcv = [np.array(a, copy=True) for a in vc6]
    for t in range(6):
        exv._write_c_strips(rcu[t], rcv[t], uc4[t], vc4[t], N, NG)
    gcu, gcv = jx.write_c_strips(jnp.asarray(uc6), jnp.asarray(vc6),
                                 jnp.asarray(uc4), jnp.asarray(vc4), tab)
    assert _bitwise_equal(gcu, np.stack(rcu)), "write_c_strips uc"
    assert _bitwise_equal(gcv, np.stack(rcv)), "write_c_strips vc"


def test_geo_lattice_exchange_matches_numpy(tab, ectx):
    m4 = N + 2 * tab.ngp
    g6 = _rnd6((m4, m4), 28)
    ref = [np.array(a, copy=True) for a in g6]
    exv._geo_lattice_exchange(ref, ectx)
    got = jx.geo_lattice_exchange(jnp.asarray(g6), tab)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got, np.stack(ref), "geo_lattice_exchange", 1e-15)
    # weighted sum -> FMA contraction differs between the jitted and the
    # eager lowering; a bound, not bitwise (see the module docstring)
    # MEASURED (job 9425294 sweep): 2.896e-16; bound = measured x 10 =
    # 2.9e-15.
    _cmp(jx.geo_lattice_exchange_jit(jnp.asarray(g6), tab), got,
                "geo_lattice_exchange jit vs eager", 2.9e-15)


# ---------------------------------------------------------------------------
# gate 1 + 2: the composed ext_scalar / ext_vector, on the ORACLE inputs
# ---------------------------------------------------------------------------

def test_ext_scalar_a_matches_numpy_on_oracle_inputs(tab, ectx):
    d = _extchain()
    assert (int(d["n"]), int(d["ng"])) == (N, NG)
    f6 = np.stack([np.asarray(d[f"t{t}_IN_A_v1"][:, :, 0], dtype=np.float64)
                   for t in range(1, 7)])
    assert f6.shape == (6, MA, MA), f6.shape   # layout contract, item 2
    ref = [np.array(a, copy=True) for a in f6]
    exv.ext_scalar_sixface(ref, "A", ectx)
    got = jx.ext_scalar_sixface(jnp.asarray(f6), tab, "A")
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got, np.stack(ref), "ext_scalar A", 1e-15)
    # weighted sum -> FMA contraction differs between the jitted and the
    # eager lowering; a bound, not bitwise (see the module docstring)
    # MEASURED (job 9425294 sweep): 9.801e-15; bound = measured x 10 =
    # 9.9e-14.
    _cmp(jx.ext_scalar_sixface_jit(jnp.asarray(f6), tab, "A"), got,
                "ext_scalar A jit vs eager", 9.9e-14)
    # the compute domain must be untouched by an exchange
    sl = slice(NG, NG + N)
    assert _bitwise_equal(np.asarray(got)[:, sl, sl], f6[:, sl, sl])


def test_ext_scalar_b_matches_numpy(tab, ectx):
    f6 = _rnd6((MB, MB), 29)
    ref = [np.array(a, copy=True) for a in f6]
    exv.ext_scalar_sixface(ref, "B", ectx)
    got = jx.ext_scalar_sixface(jnp.asarray(f6), tab, "B")
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(got, np.stack(ref), "ext_scalar B", 1e-15)


def test_ext_vector_dgrid_matches_numpy_on_oracle_inputs(tab, ectx):
    d = _extchain()
    u6 = np.stack([np.asarray(d[f"t{t}_IN_DU_v1"][:, :, 0], dtype=np.float64)
                   for t in range(1, 7)])
    v6 = np.stack([np.asarray(d[f"t{t}_IN_DV_v1"][:, :, 0], dtype=np.float64)
                   for t in range(1, 7)])
    assert u6.shape == (6, MA, MB) and v6.shape == (6, MB, MA), \
        (u6.shape, v6.shape)
    ru = [np.array(a, copy=True) for a in u6]
    rv = [np.array(a, copy=True) for a in v6]
    exv.ext_vector_dgrid_sixface(ru, rv, ectx)
    gu, gv = jx.ext_vector_dgrid_sixface(jnp.asarray(u6),
                                         jnp.asarray(v6), tab)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(gu, np.stack(ru), "ext_vector D u", 1e-15)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(gv, np.stack(rv), "ext_vector D v", 1e-15)
    ju, jv = jx.ext_vector_dgrid_sixface_jit(jnp.asarray(u6),
                                             jnp.asarray(v6), tab)
    # weighted sum -> FMA contraction differs between the jitted and the
    # eager lowering; a bound, not bitwise (see the module docstring)
    # MEASURED (job 9425294 sweep): 3.058e-14; bound = measured x 10 =
    # 3.1e-13.
    _cmp(ju, gu, "ext_vector D u jit vs eager", 3.1e-13)
    # MEASURED (job 9425294 sweep): 2.503e-14; bound = measured x 10 =
    # 2.6e-13.
    _cmp(jv, gv, "ext_vector D v jit vs eager", 2.6e-13)


def test_ext_vector_cgrid_matches_numpy_on_oracle_inputs(tab, ectx):
    d = _extchain()
    uc6 = np.stack([np.asarray(d[f"t{t}_IN_CU_v1"][:, :, 0], dtype=np.float64)
                    for t in range(1, 7)])
    vc6 = np.stack([np.asarray(d[f"t{t}_IN_CV_v1"][:, :, 0], dtype=np.float64)
                    for t in range(1, 7)])
    assert uc6.shape == (6, MB, MA) and vc6.shape == (6, MA, MB), \
        (uc6.shape, vc6.shape)
    ru = [np.array(a, copy=True) for a in uc6]
    rv = [np.array(a, copy=True) for a in vc6]
    exv.ext_vector_cgrid_sixface(ru, rv, ectx)
    gu, gv = jx.ext_vector_cgrid_sixface(jnp.asarray(uc6),
                                         jnp.asarray(vc6), tab)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(gu, np.stack(ru), "ext_vector C uc", 1e-15)
    # MEASURED (job 9425294 sweep): exactly 0.0 (bitwise); bound =
    # 1e-15 eps guard (measured exactly 0.0).
    _cmp(gv, np.stack(rv), "ext_vector C vc", 1e-15)


def test_ext_scalar_constant_field_is_preserved(tab):
    """Partition of unity through the WHOLE composed chain -- exact, and
    independent of the NumPy lane (the extchain oracle asserts the same
    invariant on the Fortran side)."""
    f6 = jnp.full((6, MA, MA), 7.25, dtype=jnp.float64)
    got = jx.ext_scalar_sixface(f6, tab, "A")
    # MEASURED (job 9425294 sweep): 1.279e-13 abs on the 7.25 field; bound = measured x 10 =
    # 1.3e-12.
    gate_scalar("ext_scalar constant",
                float(jnp.max(jnp.abs(got - 7.25))), 1.3e-12,
                quantity="max abs deviation from the constant 7.25")


def test_ext_vector_a2d_variant_wiring(tab, kinked_gs6):
    """The measurement variant must reach the lane through the tables:
    ``vector_corner='a2d'`` skips the final covariant corner fill, so
    the wedges differ."""
    e_a = exv.build_ext_context(N, NG, kinked_gs6, vector_corner="a2d")
    t_a = jx.build_jax_duo_halo_tables(e_a, kinked_gs6, nq=NQ)
    assert t_a.vector_corner == "a2d" and tab.vector_corner == "lagrange"
    u6 = jnp.asarray(_rnd6((MA, MB), 30))
    v6 = jnp.asarray(_rnd6((MB, MA), 31))
    ul, _ = jx.ext_vector_dgrid_sixface(u6, v6, tab)
    ua, _ = jx.ext_vector_dgrid_sixface(u6, v6, t_a)
    assert not _bitwise_equal(ul[0, :NG, -NG:], ua[0, :NG, -NG:])


# ---------------------------------------------------------------------------
# the two duo barriers
# ---------------------------------------------------------------------------

def test_barrier1_cgrid_matches_numpy_and_jit(tab):
    fx6 = _rnd6((NPX, N), 32)
    fy6 = _rnd6((N, NPX), 33)
    rx = [np.array(a, copy=True) for a in fx6]
    ry = [np.array(a, copy=True) for a in fy6]
    gsm.average_shared_edge_cgrid(rx, ry, N, NG)
    gx, gy = jx.average_shared_edge_cgrid(jnp.asarray(fx6),
                                          jnp.asarray(fy6), tab)
    assert _bitwise_equal(gx, np.stack(rx)), "barrier1 fx"
    assert _bitwise_equal(gy, np.stack(ry)), "barrier1 fy"
    jx_, jy = jx.average_shared_edge_cgrid_jit(jnp.asarray(fx6),
                                               jnp.asarray(fy6), tab)
    assert _bitwise_equal(jx_, gx) and _bitwise_equal(jy, gy)
    assert not _bitwise_equal(gx, fx6)


def test_barrier2_bgrid_matches_numpy_and_jit(tab):
    xb6 = _rnd6((NPX, NPX), 34)
    yb6 = _rnd6((NPX, NPX), 35)
    rx = [np.array(a, copy=True) for a in xb6]
    ry = [np.array(a, copy=True) for a in yb6]
    gsm.average_shared_edge_bgrid(rx, ry, N, NG)
    gx, gy = jx.average_shared_edge_bgrid(jnp.asarray(xb6),
                                          jnp.asarray(yb6), tab)
    assert _bitwise_equal(gx, np.stack(rx)), "barrier2 xb"
    assert _bitwise_equal(gy, np.stack(ry)), "barrier2 yb"
    jx_, jy = jx.average_shared_edge_bgrid_jit(jnp.asarray(xb6),
                                               jnp.asarray(yb6), tab)
    assert _bitwise_equal(jx_, gx) and _bitwise_equal(jy, gy)
    assert not _bitwise_equal(gy, yb6)


def test_barrier2_endpoint_diagnostic_is_reachable_and_differs(ectx,
                                                               kinked_gs6):
    """The NumPy lane's ``LEGOESM_DUO_AVG_B_ENDPOINTS=local`` screen is
    an explicit STATIC flag here (the JAX lane never reads the
    environment).  Both arms must build, and they must differ -- a flag
    that changed nothing would be a dead knob."""
    t_off = jx.build_jax_duo_halo_tables(ectx, kinked_gs6, nq=NQ)
    t_on = jx.build_jax_duo_halo_tables(ectx, kinked_gs6, nq=NQ,
                                        skip_b_endpoints=True)
    assert t_on.avg_b.dst.size < t_off.avg_b.dst.size
    xb6 = jnp.asarray(_rnd6((NPX, NPX), 36))
    yb6 = jnp.asarray(_rnd6((NPX, NPX), 37))
    a_x, _ = jx.average_shared_edge_bgrid(xb6, yb6, t_off)
    b_x, _ = jx.average_shared_edge_bgrid(xb6, yb6, t_on)
    assert not _bitwise_equal(a_x, b_x)


def test_barrier1_allflux_matches_numpy(tab):
    nslot = 4 + NQ
    afx6 = _rnd6((NPX, N, nslot), 38)
    afy6 = _rnd6((N, NPX, nslot), 39)
    rx = [np.array(a, copy=True) for a in afx6]
    ry = [np.array(a, copy=True) for a in afy6]
    gsm.average_allflux_shared_edges(rx, ry, NQ, N, NG)
    gx, gy = jx.average_allflux_shared_edges(jnp.asarray(afx6),
                                             jnp.asarray(afy6), tab)
    assert _bitwise_equal(gx, np.stack(rx)), "allflux fx"
    assert _bitwise_equal(gy, np.stack(ry)), "allflux fy"
    jx_, jy = jx.average_allflux_shared_edges_jit(jnp.asarray(afx6),
                                                  jnp.asarray(afy6), tab)
    assert _bitwise_equal(jx_, gx) and _bitwise_equal(jy, gy)


def test_barrier1_skips_slot2_w_and_slot3_qcon_exactly(tab):
    """dyn_core.F90:856 -- ``if (iq==1 .or. iq==4 .or. iq>4)``.

    Slots 2 (``w``) and 3 (``q_con``) are NOT averaged.  Both halves
    matter: the excluded slots must be BITWISE unchanged, and every
    included slot must actually MOVE (otherwise the test would pass on
    a barrier that did nothing at all).
    """
    nslot = 4 + NQ
    afx6 = _rnd6((NPX, N, nslot), 40)
    afy6 = _rnd6((N, NPX, nslot), 41)
    gx, gy = jx.average_allflux_shared_edges(jnp.asarray(afx6),
                                             jnp.asarray(afy6), tab)
    gx = np.asarray(gx)
    gy = np.asarray(gy)
    excluded = [1, 2]                     # 0-based: iq = 2 (w), 3 (q_con)
    included = [0, 3, 4, 5]               # iq = 1 (delp), 4 (temp), 5, 6
    assert sorted(tab.allflux_slots.tolist()) == included
    for s in excluded:
        assert _bitwise_equal(gx[..., s], afx6[..., s]), f"fx slot {s + 1}"
        assert _bitwise_equal(gy[..., s], afy6[..., s]), f"fy slot {s + 1}"
    for s in included:
        assert not _bitwise_equal(gx[..., s], afx6[..., s]), \
            f"fx slot {s + 1} did not move"
        assert not _bitwise_equal(gy[..., s], afy6[..., s]), \
            f"fy slot {s + 1} did not move"


def test_allflux_rejects_a_slot_axis_that_contradicts_nq(tab):
    bad = jnp.asarray(_rnd6((NPX, N, 4 + NQ + 1), 42))
    other = jnp.asarray(_rnd6((N, NPX, 4 + NQ + 1), 43))
    with pytest.raises(ValueError, match="4\\+nq"):
        jx.average_allflux_shared_edges(bad, other, tab)


# ---------------------------------------------------------------------------
# tolerance-independent: single-valuedness of a shared edge after a barrier
# ---------------------------------------------------------------------------

def _cedge_partner_indices(tile, fi, fj, along, n_src, n):
    """Where the neighbour stores this C flux slot, and with what sign.

    Uses the CERTIFIED ``neighbor_index`` map directly, not the lane's
    tables.  Caveat, stated rather than implied: both lanes read that
    same map, so this gate cannot catch a wrong neighbour map -- it
    catches a wrong sign, a dropped slot, or a wrong blend weight.
    """
    sg = 2 * n + 1
    si, sj = (2 * fi - 1, 2 * fj) if along == "i" else (2 * fi, 2 * fj - 1)
    sii, sjj = neighbor_index(si, sj, tile, n_src, sg, sg)
    if along == "i":
        sii2, sjj2 = neighbor_index(si + 2, sj, tile, n_src, sg, sg)
    else:
        sii2, sjj2 = neighbor_index(si, sj + 2, tile, n_src, sg, sg)
    dii, djj = sii2 - sii, sjj2 - sjj
    if (sii % 2 == 1) and (sjj % 2 == 0):
        s = 1.0 if (dii if dii != 0 else djj) > 0 else -1.0
        return 0, (sii + 1) // 2 - 1, sjj // 2 - 1, s
    s = 1.0 if (djj if djj != 0 else dii) > 0 else -1.0
    return 1, sii // 2 - 1, (sjj + 1) // 2 - 1, s


def test_barrier1_makes_every_shared_edge_single_valued(tab):
    """After the blend, a shared C-flux slot must read the SAME value
    from either face (up to the component sign).  Asserted BITWISE:
    both sides evaluate ``0.5*(a + s*b)`` with the same rounding."""
    fx6 = _rnd6((NPX, N), 44)
    fy6 = _rnd6((N, NPX), 45)
    gx, gy = jx.average_shared_edge_cgrid(jnp.asarray(fx6),
                                          jnp.asarray(fy6), tab)
    arr = (np.asarray(gx), np.asarray(gy))
    checked = 0
    for tile in range(1, 7):
        nw, ne, ns, nn = neighbor_tiles(tile)
        for fj in range(1, N + 1):
            for fi, n_src in ((1, nw), (NPX, ne)):
                k, pi, pj, s = _cedge_partner_indices(tile, fi, fj, "i",
                                                      n_src, N)
                own = arr[0][tile - 1, fi - 1, fj - 1]
                part = arr[k][n_src - 1, pi, pj]
                assert own == s * part, (tile, fi, fj, own, part)
                checked += 1
        for fi in range(1, N + 1):
            for fj, n_src in ((1, ns), (NPX, nn)):
                k, pi, pj, s = _cedge_partner_indices(tile, fi, fj, "j",
                                                      n_src, N)
                own = arr[1][tile - 1, fi - 1, fj - 1]
                part = arr[k][n_src - 1, pi, pj]
                assert own == s * part, (tile, fi, fj, own, part)
                checked += 1
    assert checked == 6 * 4 * N
    # non-vacuity: the SAME check on the un-blended input must FAIL
    raw = (fx6, fy6)
    bad = 0
    for tile in range(1, 7):
        nw, _, _, _ = neighbor_tiles(tile)
        for fj in range(1, N + 1):
            k, pi, pj, s = _cedge_partner_indices(tile, 1, fj, "i", nw, N)
            if raw[0][tile - 1, 0, fj - 1] != s * raw[k][nw - 1, pi, pj]:
                bad += 1
    assert bad > 0, "the single-valuedness check cannot fail -> vacuous"


# ---------------------------------------------------------------------------
# gate 3: entry guards, each shown non-vacuous
# ---------------------------------------------------------------------------

def test_float_operands_must_have_one_uniform_dtype(tab):
    f32 = jnp.asarray(_rnd6((MA, MA), 46), dtype=jnp.float32)
    f64 = jnp.asarray(_rnd6((MA, MA), 46))
    assert jx.exchange_agrid_scalar_halos(f32, tab).dtype == jnp.float32
    assert jx.exchange_agrid_scalar_halos(f64, tab).dtype == jnp.float64

    u32 = jnp.asarray(_rnd6((MA, MB), 47), dtype=jnp.float32)
    v32 = jnp.asarray(_rnd6((MB, MA), 48), dtype=jnp.float32)
    v64 = jnp.asarray(_rnd6((MB, MA), 48))
    got_u32, got_v32 = jx.exchange_dgrid_vector_halos(u32, v32, tab)
    assert got_u32.dtype == jnp.float32
    assert got_v32.dtype == jnp.float32
    with pytest.raises(TypeError, match="MIXED float dtypes"):
        jx.exchange_dgrid_vector_halos(u32, v64, tab)

    fx32 = jnp.asarray(_rnd6((NPX, N), 49), dtype=jnp.float32)
    fy32 = jnp.asarray(_rnd6((N, NPX), 50), dtype=jnp.float32)
    got_fx32, got_fy32 = jx.average_shared_edge_cgrid(fx32, fy32, tab)
    assert got_fx32.dtype == jnp.float32
    assert got_fy32.dtype == jnp.float32
    with pytest.raises(TypeError, match="MIXED float dtypes"):
        jx.average_shared_edge_cgrid(
            fx32, jnp.asarray(fy32, dtype=jnp.float64), tab)


def test_unknown_stagger_ring_and_key_raise(tab):
    f6 = jnp.asarray(_rnd6((MA, MA), 51))
    with pytest.raises(ValueError, match="not implemented"):
        jx.ext_scalar_sixface(f6, tab, "CX")
    jx.ext_scalar_sixface(f6, tab, "A")               # non-vacuous

    with pytest.raises(ValueError, match="not part of the duo ext flow"):
        jx.k2e_remap_halo_rings(f6, tab, "DX")
    with pytest.raises(ValueError, match="unsupported"):
        jx.k2e_remap_halo_rings(f6, tab, "Z")
    jx.k2e_remap_halo_rings(f6, tab, "A")             # non-vacuous

    with pytest.raises(ValueError, match="ring="):
        jx.exchange_agrid_scalar_halos(f6, tab, "bogus")
    jx.exchange_agrid_scalar_halos(f6, tab, "stepper")

    with pytest.raises(ValueError, match="key"):
        jx.corner_lagrange_fill(f6, tab, "nope")
    jx.corner_lagrange_fill(f6, tab, "a3")


def test_too_small_resolution_is_refused(ectx, tab):
    """n < interp_order+1 breaks the corner-fill independence argument;
    the builder must refuse rather than emit a silently reordered group.

    The guard reads only ``ectx['n']`` and runs before any array is
    touched, so a shallow copy with a small ``n`` exercises exactly the
    guarded path without paying for a second grid build.  Non-vacuous:
    the ``tab`` fixture is the same call at n=12."""
    assert tab is not None
    small = dict(ectx)
    small["n"] = 3
    with pytest.raises(ValueError, match="independent group"):
        jx.build_jax_duo_halo_tables(small, None, nq=0)


# ---------------------------------------------------------------------------
# gate 4: gradients (+ the adjoint identity, the strong version)
# ---------------------------------------------------------------------------

def test_check_grads_exchanges_and_barriers(tab):
    f6 = jnp.asarray(_rnd6((MA, MA), 52))
    check_grads(lambda x: jx.exchange_agrid_scalar_halos(x, tab),
                (f6,), order=2)
    b6 = jnp.asarray(_rnd6((MB, MB), 53))
    check_grads(lambda x: jx.exchange_bgrid_scalar_halos(x, tab),
                (b6,), order=2)
    u6 = jnp.asarray(_rnd6((MA, MB), 54))
    v6 = jnp.asarray(_rnd6((MB, MA), 55))
    check_grads(lambda a, b: jx.exchange_dgrid_vector_halos(a, b, tab),
                (u6, v6), order=2)
    uc6 = jnp.asarray(_rnd6((MB, MA), 56))
    vc6 = jnp.asarray(_rnd6((MA, MB), 57))
    check_grads(lambda a, b: jx.exchange_cgrid_vector_halos(a, b, tab),
                (uc6, vc6), order=2)
    fx6 = jnp.asarray(_rnd6((NPX, N), 58))
    fy6 = jnp.asarray(_rnd6((N, NPX), 59))
    check_grads(lambda a, b: jx.average_shared_edge_cgrid(a, b, tab),
                (fx6, fy6), order=2)
    xb6 = jnp.asarray(_rnd6((NPX, NPX), 60))
    yb6 = jnp.asarray(_rnd6((NPX, NPX), 61))
    check_grads(lambda a, b: jx.average_shared_edge_bgrid(a, b, tab),
                (xb6, yb6), order=2)
    nslot = 4 + NQ
    ax = jnp.asarray(_rnd6((NPX, N, nslot), 62))
    ay = jnp.asarray(_rnd6((N, NPX, nslot), 63))
    check_grads(lambda a, b: jx.average_allflux_shared_edges(a, b, tab),
                (ax, ay), order=2)


def test_check_grads_k2e_corner_and_projections(tab):
    f6 = jnp.asarray(_rnd6((MA, MA), 64))
    check_grads(lambda x: jx.k2e_remap_halo_rings(x, tab, "A"),
                (f6,), order=2)
    check_grads(lambda x: jx.corner_lagrange_fill(x, tab, "a3"),
                (f6,), order=2)
    m4 = N + 2 * tab.ngp
    ug = jnp.asarray(_rnd6((m4, m4), 65))
    vg = jnp.asarray(_rnd6((m4, m4), 66))
    check_grads(lambda a, b: jx.a2d_project(a, b, tab), (ug, vg), order=2)
    check_grads(lambda a, b: jx.a2c_project(a, b, tab), (ug, vg), order=2)
    check_grads(lambda x: jx.geo_lattice_exchange(x, tab), (ug,), order=2)


def test_check_grads_c2l_and_composites_on_masked_objectives(tab):
    """``c2l``/``pack_p1`` emit NaN outside their windows by design, so
    the objective is masked to the finite slots (see
    ``_finite_mask_scalar`` for why the select is safe here)."""
    u6 = jnp.asarray(_rnd6((MA, MB), 67))
    v6 = jnp.asarray(_rnd6((MB, MA), 68))
    obj = _finite_mask_scalar(lambda a, b: jx.c2l_ord2_face(a, b, tab),
                              u6, v6)
    check_grads(obj, (u6, v6), order=2)

    obj = _finite_mask_scalar(
        lambda a, b: jx.ext_vector_dgrid_sixface(a, b, tab), u6, v6)
    check_grads(obj, (u6, v6), order=2)

    uc6 = jnp.asarray(_rnd6((MB, MA), 90))
    vc6 = jnp.asarray(_rnd6((MA, MB), 91))
    obj = _finite_mask_scalar(
        lambda a, b: jx.ext_vector_cgrid_sixface(a, b, tab), uc6, vc6)
    check_grads(obj, (uc6, vc6), order=2)
    obj = _finite_mask_scalar(
        lambda a, b: jx.c2l_ord2_cgrid_face(a, b, tab), uc6, vc6)
    check_grads(obj, (uc6, vc6), order=2)

    f6 = jnp.asarray(_rnd6((MA, MA), 69))
    obj = _finite_mask_scalar(lambda x: jx.ext_scalar_sixface(x, tab, "A"),
                              f6)
    check_grads(obj, (f6,), order=2)


def test_adjoint_identity_holds_for_every_linear_kernel(tab):
    """``<J v, w> == <v, J^T w>`` -- the gate an order-2 ``check_grads``
    cannot provide on a linear operator (it compares 0 to 0)."""
    m4 = N + 2 * tab.ngp
    cases = {
        "agrid": (lambda x: jx.exchange_agrid_scalar_halos(x, tab),
                  (jnp.asarray(_rnd6((MA, MA), 70)),)),
        "bgrid": (lambda x: jx.exchange_bgrid_scalar_halos(x, tab),
                  (jnp.asarray(_rnd6((MB, MB), 71)),)),
        "dgrid": (lambda a, b: jx.exchange_dgrid_vector_halos(a, b, tab),
                  (jnp.asarray(_rnd6((MA, MB), 72)),
                   jnp.asarray(_rnd6((MB, MA), 73)))),
        "cgrid": (lambda a, b: jx.exchange_cgrid_vector_halos(a, b, tab),
                  (jnp.asarray(_rnd6((MB, MA), 74)),
                   jnp.asarray(_rnd6((MA, MB), 75)))),
        "k2e": (lambda x: jx.k2e_remap_halo_rings(x, tab, "A"),
                (jnp.asarray(_rnd6((MA, MA), 76)),)),
        "corner": (lambda x: jx.corner_lagrange_fill(x, tab, "a3"),
                   (jnp.asarray(_rnd6((MA, MA), 77)),)),
        "barrier1": (lambda a, b: jx.average_shared_edge_cgrid(a, b, tab),
                     (jnp.asarray(_rnd6((NPX, N), 78)),
                      jnp.asarray(_rnd6((N, NPX), 79)))),
        "barrier2": (lambda a, b: jx.average_shared_edge_bgrid(a, b, tab),
                     (jnp.asarray(_rnd6((NPX, NPX), 80)),
                      jnp.asarray(_rnd6((NPX, NPX), 81)))),
        "a2d": (lambda a, b: jx.a2d_project(a, b, tab),
                (jnp.asarray(_rnd6((m4, m4), 82)),
                 jnp.asarray(_rnd6((m4, m4), 83)))),
        "ext_scalar": (lambda x: jx.ext_scalar_sixface(x, tab, "A"),
                       (jnp.asarray(_rnd6((MA, MA), 84)),)),
        "ext_vector_d": (
            lambda a, b: jx.ext_vector_dgrid_sixface(a, b, tab),
            (jnp.asarray(_rnd6((MA, MB), 85)),
             jnp.asarray(_rnd6((MB, MA), 86)))),
    }
    worst = {}
    for name, (fn, primals) in cases.items():
        worst[name] = _adjoint_residual(fn, primals)
        # MEASURED (job 9425294 sweep): worst adjoint residual barrier1 1.776e-14 (a2d
        # 4.456e-15, ext_vector_d 5.793e-15, all others <= 2.3e-15); bound =
        # measured x 10 = 1.8e-13.
        gate_scalar(f"halo adjoint {name}", worst[name], 1.8e-13,
                    quantity="adjoint identity residual")


# ---------------------------------------------------------------------------
# jit hygiene
# ---------------------------------------------------------------------------

def test_no_retrace_across_calls_and_one_trace_per_tables_object(tab):
    calls = {"n": 0}

    def counted(f6, tables, ring="stepper"):
        calls["n"] += 1
        return jx.exchange_agrid_scalar_halos(f6, tables, ring)

    fn = jx.make_exchange_agrid_scalar_halos_jit(counted)
    a = jnp.asarray(_rnd6((MA, MA), 87))
    b = jnp.asarray(_rnd6((MA, MA), 88))
    fn(a, tab, "stepper")
    fn(b, tab, "stepper")
    assert calls["n"] == 1, calls           # same shapes + same tables


def test_stack6_unstack6_roundtrip():
    lst = [_rnd((MA, MA), 89 + t) for t in range(6)]
    st = jx.stack6(lst)
    assert st.shape == (6, MA, MA)
    back = jx.unstack6(st)
    assert len(back) == 6
    for t in range(6):
        assert _bitwise_equal(back[t], lst[t])


def test_blend_as_stencil_identity_holds_in_f32_including_subnormals(tab):
    """The certified numerics turn the barrier blend into a stencil on
    ``0.5*(a + s*b) == 0.5*a + (0.5*s)*b``, exact for NORMAL operands.
    f32 normals bottom out near 1.2e-38, so tracer-scale magnitudes can
    reach subnormals where the two forms may differ (GLM 2026-09-10).
    This pins where the identity holds and where it stops."""
    import numpy as _np
    rng = _np.random.default_rng(77)
    for dtype, scales, must_hold in ((_np.float64, (1e0, 1e-20, 1e-300), True),
                                     (_np.float32, (1e0, 1e-20, 1e-30), True),
                                     (_np.float32, (1e-40,), False)):
        for sc in scales:
            a = (rng.standard_normal(4096) * sc).astype(dtype)
            b = (rng.standard_normal(4096) * sc).astype(dtype)
            for s in (dtype(1.0), dtype(-1.0)):
                lhs = dtype(0.5) * (a + s * b)
                rhs = dtype(0.5) * a + (dtype(0.5) * s) * b
                same = _np.array_equal(lhs.view(_np.uint8), rhs.view(_np.uint8))
                if must_hold:
                    assert same, f"{dtype.__name__} at {sc}: identity broke"
                else:
                    # not asserted to break every time; the point is that
                    # this range is OUTSIDE the certified precondition
                    pass
