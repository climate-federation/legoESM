"""FV3 tp_core / sw_core transport kernels -- JAX lane (xppm, yppm,
pert_ppm, fv_tp_2d, copy_corners, deln_flux, xtp_u, ytp_v).

Functional, jit-compatible mirror of the certified NumPy fp64 lane
(``fv3_native_d_sw.py``, itself a loop-faithful port of the pinned
oracle).  The NumPy lane is the SPECIFICATION for this module (hop B of
the authority chain in ``docs/atmosphere/fv3_duo_jax_lane_strategy.md``);
these kernels are certified AGAINST the NumPy lane, never against the
Fortran directly (one authority per hop).

Verified line numbers in the PINNED oracle tree
``/burg-archive/glab/users/pg2328/fv3_oracle_pinned/
atmos_cubed_sphere-symmetryclean`` (every number below was read with
``sed``/``grep`` on that tree while writing this module -- R7):

* ``model/tp_core.F90``  -- module constants :35-39 / :51-55 / :57-58 /
  :63-65 / :69-70, ``fv_tp_2d`` :80-225, ``copy_corners`` :229-306,
  ``xppm`` :308-681, ``yppm`` :684-1102, ``pert_ppm`` :1156-1214,
  ``deln_flux`` :1217-1365.
* ``model/sw_core.F90`` -- module constants :36-39 / :48-49 / :57-59,
  ``xtp_u`` :2540-2894 (duo-unclamped ``is3``/``ie3`` at :2566,
  ``pert_ppm(1, u(2,j), ...)`` at :2839 / :2862), ``ytp_v`` :2897-3353
  (duo-unclamped at :2923, ``pert_ppm(ie-is+2, v(is,j), ...)`` at :3277
  / :3315).

Every numeric constant used here is IMPORTED from the NumPy twin's
module (``fv3_native_d_sw``) rather than restated, so a constant cannot
drift between the two lanes (R2).

Two OTHER JAX implementations of this material already exist and are a
DIFFERENT LANE -- do not confuse them with this one, and do not merge
them:

* ``legoesm.core.fv_tp_2d.fv_tp_2d`` (:1122) and
* ``legoesm.core.fv3_sw_core.ppm_transport_1d`` (:2927).

Those are cdgrid-signature research solvers: they take a ``cdgrid``
object, do their own ``pad_halo`` cross-face exchanges, are written for
the six-face batched (6, N, M) layout, and were tuned against our own
shallow-water suite.  They are NOT loop-faithful mirrors of the duo
NumPy lane -- their index windows, their halo model and their branch
coverage all differ.  This module exists because the duo campaign needs
a twin whose only permitted difference from ``fv3_native_d_sw`` is
mechanical; reusing a differently-shaped solver would forfeit exactly
the property (any divergence is a port bug) that makes the port
debuggable.

Mirror doctrine (R1-R5 of the strategy doc):

* **Functional**: the NumPy lane mutates ``flux``/``fx``/``fy``/``q``
  in place; every twin here RETURNS its outputs.  ``fv_tp_2d`` returns
  ``(q, fx, fy)`` -- ``q`` because ``copy_corners`` mutates its corner
  ghosts, which the caller must carry forward (tp_core.F90:138-140 and
  :159-161).
* **Explicit x64**: every float operand must arrive float64 (the oracle
  build is ``-fdefault-real-8``); a float32 operand raises ``TypeError``
  at entry.  The check reads only static dtypes, so it is jit-safe.
* **No ``donate_argnums``** anywhere in this lane (grad-path doctrine).
* **Static index bounds**: ``is_``/``ie``/``isd``/... /``npx``/``npy``
  and every scheme selector are STATIC Python values; ``_rng``/``_idx``
  turn each Fortran index window into a python slice and RAISE if the
  window escapes the declared bounds (a negative python start would
  otherwise WRAP silently, which is how a halo-loop-bound bug hides).
* **Fill value**: locals are allocated ``NaN`` at their exact Fortran
  declared bounds, exactly like the NumPy lane's ``_fl``/``_fl1``, so a
  cell the oracle never writes stays a tripwire instead of becoming a
  plausible number.  ONE divergence follows from this and is deliberate:
  the ``iord < 0`` clamp is python ``max(0., al)`` in the NumPy lane
  (which returns ``0.0`` for a NaN operand) and ``jnp.maximum`` here
  (which returns NaN).  It can only differ on cells the oracle never
  wrote; on the whole-face cubed-sphere window ``al`` is fully written
  (main loop ``is1..ie3`` plus the two edge blocks), so the two lanes
  agree there.

Loop vectorisation is NOT automatic (strategy correction 1).  An
``i``/``j`` loop is vectorised here only where no iteration reads a
location another iteration writes; the dependence argument is stated in
a comment at every vectorised loop.  Where a real read-after-write
recurrence exists it is kept ORDERED:

* ``deln_flux``'s ``do n=1,nord`` pass loop (fv3_native_d_sw.py:1150,
  tp_core.F90:1290) is a RECURRENCE -- pass ``n`` reads the ``fx2``/
  ``fy2`` that pass ``n-1`` wrote.  It stays a python ``for`` over a
  STATIC trip count, i.e. an unrolled ordered chain; it is never
  vectorised and never turned into a scan (the window ``nt = nord - n``
  shrinks each pass, so the trip bodies are not even the same shape).
* the four corner blocks of ``copy_corners`` are applied as an ORDERED
  chain of ``.at[].set`` (the second ``dir`` call reads cells the first
  wrote -- tp_core.F90:138-140 then :159-161 in ``fv_tp_2d``).

``jnp.where`` is a SELECT, not lazy control flow (strategy correction
2): both arms are evaluated, so a division by zero in the DEAD arm
still poisons the reverse-mode gradient of the live one.  Every
data-dependent branch in this module was checked; exactly one family is
non-total, the PPM positive-definite test

    ``fmin = a0 + 0.25/a4*da1**2 + a4*r12``   (a4 = -3*(al+ar))

which appears in ``pert_ppm`` (iv=0), ``xppm``/``yppm`` at
``iord/jord == -5`` and at ``iord/jord in (7, 12)``.  Its guard
``abs(da1) < -a4`` already implies ``a4 < 0``, so the standard
double-``where`` is exact here: the denominator is replaced by ``-1.0``
wherever the guard is false, which is precisely where the value is
discarded.  Every other branch (``c > 0`` upwind selects, the
``smt5``/``smt6``/``hi5``/``hi6`` limiter flags, the ``min``/``max``
PPM clamps, ``copysign``) has two total, finite arms and uses a plain
``jnp.where``.

Differentiability.  Away from switching surfaces every kernel here is
piecewise polynomial/rational in its operands and order-2
``check_grads`` holds.  The NON-SMOOTH SITES, each only C^0 (or
discontinuous) across the named surface, are:

1. ``jnp.copysign(x, s)``     -- kinked at ``x = 0``; the SIGN factor
   flips at ``s = 0`` (a jump unless ``x = 0``).  Sites: ``dm`` in the
   ``iord >= 7`` arms of xppm/yppm and the ``iord >= 8`` arms of
   xtp_u/ytp_v; ``bl``/``br`` at ``iord == 8`` and ``iord == 11``;
   ``fx0`` at ``iord == 3`` in xtp_u/ytp_v.
2. ``jnp.minimum``/``jnp.maximum`` ties -- the three-way ``dm`` clamp,
   the ``min(max(0,pmp,lac), max(b, min(0,pmp,lac)))`` PPM clamps at
   ``iord`` 9/10, the ``xt`` monotonicity clamps in the edge blocks,
   and the ``iord < 0`` ``max(0., al)`` clamp.
3. the upwind select ``c(i,j) > 0`` -- the flux VALUE is continuous
   there (both arms reduce to ``al(i)`` at ``c = 0``) but the
   derivative w.r.t. ``c`` is not.
4. the limiter flags ``smt5 = bl*br < 0``, ``smt5 = |lim_fac*b0| <
   |bl-br|``, ``smt6 = 3|b0| < |bl-br|``, ``hi5``/``hi6`` -- crossing
   any of these ADDS or DROPS the whole ``fx1``/``fx0`` term, so the
   flux is DISCONTINUOUS across them, not merely kinked.
5. ``pert_ppm``: ``al*ar = 0`` (iv != 0), ``a6da = -da2`` and
   ``a6da = da2`` (iv != 0), ``a0 = 0`` and ``fmin = 0`` (iv = 0).
6. the ``0.25/a4`` family above is UNDEFINED at ``a4 = 0``; the
   double-``where`` makes the returned VALUE right there, but no
   gradient claim is made at ``a4 = 0`` exactly.

Because of 4, a rounding-level difference between the two lanes near a
limiter surface can flip a branch and produce a discrepancy FAR above
1e-15.  Tolerances for these kernels must therefore be quoted per
fixture with a statement of whether the fixture sits in a smooth region
or crosses a limiter surface; a single "pointwise ~1e-15" bound is not
meaningful for xppm/yppm.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np  # STATIC trace-time index arrays only, never traced
from legoesm.core.fv3_native_d_sw import (
    NEAR_ZERO_SW,
    NEAR_ZERO_TP,
    PPM_FAC,
    R3,
    R12,
    SW_C1,
    SW_C2,
    SW_C3,
    SW_P1,
    SW_P2,
    SW_S11,
    SW_S14,
    SW_S15,
    TP_C1,
    TP_C2,
    TP_C3,
    TP_P1,
    TP_P2,
    TP_S11,
    TP_S14,
    TP_S15,
)
from legoesm.core.fv3_phase3d_common import require_uniform_float_jax

# ---------------------------------------------------------------------
# Supported scheme selectors.  Dispatch-hardening (CLAUDE.md): an
# unknown order must RAISE on the STATIC value at function entry, never
# fall through to a neighbouring branch.  The sets below are exactly the
# values the NumPy lane BRANCHES ON; anything else reaches a bare
# ``else`` there and would silently run a different scheme.
#
# xppm/yppm (fv3_native_d_sw.py:289-653 / :680-1088):
#   iord <  7 -> mord = abs(iord) selects 1, 2, 3, 4, else {5, 6};
#                iord < 0 additionally clamps al, and iord == -5 has its
#                own positive-definite arm.  mord 0 or mord >= 7 with
#                iord < 7 hits the {5,6} else -> rejected.
#   iord >= 7 -> 8, 10, 11, {7, 12}, else {9, 13} (+ pert_ppm for 9/13).
#                14 would take the else and SKIP pert_ppm -> rejected,
#                even though fv_arrays.F90:349 documents ">12" as
#                positive-definite.
_PPM_ORDS = (-6, -5, -4, -3, -2, -1,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

# xtp_u/ytp_v (fv3_native_d_sw.py:2002-2310 / :2349-2721): the staggered
# routines branch on the RAW iord (no abs()), so negatives are NOT a
# scheme here -- they would fall into the {5,6,7} else.
#   iord <  8 -> 1, 2, 3, 4, else {5, 6, 7}
#   iord >= 8 -> 8, 9, 10, else {11}
_SW_ORDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


def _rng(lo: int, hi: int, org: int, n: int, what: str) -> slice:
    """Python slice for the INCLUSIVE Fortran window ``lo..hi``.

    ``org`` is the Fortran index of element 0 of the axis and ``n`` its
    length.  ``hi < lo`` is Fortran's zero-trip ``do lo,hi`` and returns
    an empty slice.  Anything outside the declared bounds RAISES: a
    negative python start silently wraps to the far end of the axis,
    which is exactly the "halo loop-bound bug" failure mode (strategy
    doc section 8).
    """
    if hi < lo:
        return slice(0, 0)
    if lo < org or hi > org + n - 1:
        raise ValueError(
            f"{what}: Fortran window {lo}..{hi} escapes the declared "
            f"bounds {org}..{org + n - 1} (a negative python start would "
            f"WRAP silently)")
    return slice(lo - org, hi - org + 1)


def _idx(i: int, org: int, n: int, what: str) -> int:
    """Python index for a single Fortran index (bounds-checked)."""
    if i < org or i > org + n - 1:
        raise ValueError(
            f"{what}: Fortran index {i} escapes the declared bounds "
            f"{org}..{org + n - 1}")
    return i - org


def _nan(shape, dtype=jnp.float64):
    """Local temp at its exact Fortran declared bounds, NaN filled."""
    return jnp.full(shape, jnp.nan, dtype)


def _validate_ord(fname: str, argname: str, value, allowed) -> None:
    """Dispatch-hardening guard on a STATIC scheme selector."""
    if value not in allowed:
        raise ValueError(
            f"{fname}: {argname}={value!r} is not a supported scheme. "
            f"Supported: {sorted(allowed)}. An unsupported order would "
            f"silently fall through to a neighbouring branch in the "
            f"NumPy lane and run different numerics.")


# =====================================================================
# tp_core_mod
# =====================================================================

def pert_ppm(a0, al, ar, iv: int):
    """JAX twin of ``fv3_native_d_sw.pert_ppm`` (tp_core.F90:1156-1214).

    Functional: the NumPy lane mutates ``al``/``ar`` in place over
    0-based length-``im`` views; this twin takes the same three operands
    and RETURNS ``(al, ar)``.  ``im`` is dropped -- the Fortran loop
    ``do i=1,im`` is elementwise, so the twin applies to whatever shape
    the caller hands it (a 1-D run, a 2-D (i, j) block, or a single
    column).  ``iv`` is STATIC (a python branch, never traced).

    Dependence (correction 1): the loop writes ``al(i)``/``ar(i)`` and
    reads only ``a0(i)``/``al(i)``/``ar(i)`` -- iteration ``i`` touches
    no other iteration's cells, so the whole loop vectorises.

    Non-total branch (correction 2): ``fmin`` divides by ``a4``.  The
    guard ``abs(da1) < -a4`` implies ``a4 < 0``, so the denominator is
    replaced by ``-1.0`` wherever the guard is false (double-``where``);
    that is exactly where ``fmin`` is discarded, so the returned value
    is unchanged and no Inf/NaN reaches the reverse-mode gradient.

    Switching surfaces (non-smooth): ``a0 = 0`` and ``fmin = 0`` for
    ``iv == 0``; ``al*ar = 0``, ``a6da = -da2`` and ``a6da = da2`` for
    ``iv != 0``.
    """
    require_uniform_float_jax("pert_ppm", {"a0": a0, "al": al, "ar": ar})
    a0 = jnp.asarray(a0)
    al = jnp.asarray(al)
    ar = jnp.asarray(ar)

    if iv == 0:
        # :1168-1191  Positive definite constraint
        a4 = -3.0 * (ar + al)
        da1 = ar - al
        pos = a0 > 0.0                       # else-arm of `a0 <= 0.`
        inner = jnp.abs(da1) < -a4           # implies a4 < 0
        # double-where: a4 is only used where `inner` holds (a4 < 0).
        a4_safe = jnp.where(inner, a4, -1.0)
        fmin = a0 + 0.25 / a4_safe * da1 ** 2 + a4 * R12
        hit = pos & inner & (fmin < 0.0)
        both = hit & (ar > 0.0) & (al > 0.0)
        pos_da = hit & (~both) & (da1 > 0.0)
        neg_da = hit & (~both) & (~(da1 > 0.0))
        # `ar = -2.*al` then `al = -2.*ar` are exclusive arms, so each
        # reads the PRE-update partner exactly as the Fortran does.
        ar_new = jnp.where(both, 0.0, jnp.where(pos_da, -2.0 * al, ar))
        al_new = jnp.where(both, 0.0, jnp.where(neg_da, -2.0 * ar, al))
        # `a0 <= 0.` zeroes both (this arm wins over everything above).
        al_out = jnp.where(pos, al_new, 0.0)
        ar_out = jnp.where(pos, ar_new, 0.0)
        return al_out, ar_out

    # :1193-1212  Standard PPM constraint
    cross = al * ar < 0.0
    da1 = al - ar
    da2 = da1 ** 2
    a6da = 3.0 * (al + ar) * da1
    lo = cross & (a6da < -da2)
    hi = cross & (a6da > da2)
    ar_out = jnp.where(cross, jnp.where(lo, -2.0 * al, ar), 0.0)
    al_out = jnp.where(cross, jnp.where(hi, -2.0 * ar, al), 0.0)
    return al_out, ar_out


def _corner_block(q, ilo: int, jlo: int, i0: int, i1: int, j0: int,
                  j1: int, src_i, src_j, what: str):
    """One ``copy_corners`` corner block as a functional gather+scatter.

    ``i0..i1`` / ``j0..j1`` are the INCLUSIVE Fortran target window (the
    two ``do`` bounds); ``src_i``/``src_j`` map the target Fortran
    ``(i, j)`` to the source Fortran indices -- the literal index
    expressions from the loop body.  Building the index arrays at trace
    time (static python ints -> ``np.meshgrid``) keeps the translation
    mechanical: the same expression appears here as in the loop.

    Dependence (correction 1): within one corner block the source and
    target windows are disjoint in at least one axis for every case in
    ``copy_corners`` (checked per case at the call sites below), so the
    block is a pure gather and the ``for j: for i:`` nest vectorises.
    ACROSS blocks nothing is assumed -- the four blocks are applied as
    an ordered chain, so a later block sees the earlier ones' writes
    exactly as the in-place loop would.
    """
    ni, nj = q.shape[0], q.shape[1]
    ii = np.arange(i0, i1 + 1)
    jj = np.arange(j0, j1 + 1)
    if ii.size == 0 or jj.size == 0:
        return q
    ti, tj = np.meshgrid(ii, jj, indexing="ij")
    si = src_i(ti, tj)
    sj = src_j(ti, tj)
    for name, arr, org, n in (("target i", ti, ilo, ni),
                              ("target j", tj, jlo, nj),
                              ("source i", si, ilo, ni),
                              ("source j", sj, jlo, nj)):
        if arr.min() < org or arr.max() > org + n - 1:
            raise ValueError(
                f"{what}: {name} range {int(arr.min())}..{int(arr.max())} "
                f"escapes the declared bounds {org}..{org + n - 1}")
    return q.at[ti - ilo, tj - jlo].set(q[si - ilo, sj - jlo])


def copy_corners(q, npx: int, npy: int, dir_: int, bounded_domain: bool,
                 bd, sw_corner: bool, se_corner: bool, nw_corner: bool,
                 ne_corner: bool, *, duogrid: bool = False):
    """JAX twin of ``fv3_native_d_sw.copy_corners``
    (tp_core.F90:229-306).

    Functional: the NumPy lane rotates ``q``'s corner ghosts in place;
    this twin RETURNS the new ``q``.  ``q`` carries Fortran bounds
    ``(bd.isd:bd.ied, bd.jsd:bd.jed)``.  ``bd`` is the NumPy lane's
    ``Bounds`` NamedTuple -- hashable BY VALUE, so a fresh-but-equal
    ``Bounds`` shares one jit cache entry (same doctrine as
    ``fv3_nh_core``'s static bounds tuple).

    DUO (symmetryclean tp_core.F90:239): ``if (bounded_domain .or.
    duogrid) return`` -- the duo halos carry real cross-face corner
    data, so the rotation is skipped entirely.

    Every corner block is a pure gather (see :func:`_corner_block`); the
    four blocks run in the oracle's source order.
    """
    require_uniform_float_jax("copy_corners", {"q": q})
    q = jnp.asarray(q)
    ng = bd.ng
    ilo, jlo = bd.isd, bd.jsd

    if bounded_domain or duogrid:
        return q

    if dir_ == 1:
        # :242-266 XDir.  Every case: the source j window is disjoint
        # from the target j window (targets sit in the j halo, sources
        # in the j interior, and vice versa at the north corners), so
        # each block is a pure gather.
        if sw_corner:                                    # q(i,j)=q(j,1-i)
            q = _corner_block(q, ilo, jlo, 1 - ng, 0, 1 - ng, 0,
                              lambda i, j: j, lambda i, j: 1 - i,
                              "copy_corners dir1 sw")
        if se_corner:                       # q(i,j)=q(npy-j,i-npx+1)
            q = _corner_block(q, ilo, jlo, npx, npx + ng - 1, 1 - ng, 0,
                              lambda i, j: npy - j,
                              lambda i, j: i - npx + 1,
                              "copy_corners dir1 se")
        if ne_corner:                    # q(i,j)=q(j,2*npx-1-i)
            q = _corner_block(q, ilo, jlo, npx, npx + ng - 1, npy,
                              npy + ng - 1,
                              lambda i, j: j,
                              lambda i, j: 2 * npx - 1 - i,
                              "copy_corners dir1 ne")
        if nw_corner:                    # q(i,j)=q(npy-j,i-1+npx)
            q = _corner_block(q, ilo, jlo, 1 - ng, 0, npy, npy + ng - 1,
                              lambda i, j: npy - j,
                              lambda i, j: i - 1 + npx,
                              "copy_corners dir1 nw")
        return q

    if dir_ == 2:
        # :268-292 YDir (same disjointness argument, axes swapped).
        if sw_corner:                                    # q(i,j)=q(1-j,i)
            q = _corner_block(q, ilo, jlo, 1 - ng, 0, 1 - ng, 0,
                              lambda i, j: 1 - j, lambda i, j: i,
                              "copy_corners dir2 sw")
        if se_corner:                        # q(i,j)=q(npy+j-1,npx-i)
            q = _corner_block(q, ilo, jlo, npx, npx + ng - 1, 1 - ng, 0,
                              lambda i, j: npy + j - 1,
                              lambda i, j: npx - i,
                              "copy_corners dir2 se")
        if ne_corner:                    # q(i,j)=q(2*npy-1-j,i)
            q = _corner_block(q, ilo, jlo, npx, npx + ng - 1, npy,
                              npy + ng - 1,
                              lambda i, j: 2 * npy - 1 - j,
                              lambda i, j: i,
                              "copy_corners dir2 ne")
        if nw_corner:                    # q(i,j)=q(j+1-npx,npy-i)
            q = _corner_block(q, ilo, jlo, 1 - ng, 0, npy, npy + ng - 1,
                              lambda i, j: j + 1 - npx,
                              lambda i, j: npy - i,
                              "copy_corners dir2 nw")
        return q

    raise ValueError(
        f"copy_corners: dir={dir_!r} is not a supported direction "
        f"(tp_core.F90:241/:267 branch on 1 == XDir and 2 == YDir only; "
        f"a third value would silently return q unrotated)")


def xppm(q, c, iord: int, is_: int, ie: int, isd: int, ied: int,
         jfirst: int, jlast: int, jsd: int, jed: int, npx: int, npy: int,
         dxa, bounded_domain: bool, grid_type: int, lim_fac: float, *,
         duogrid: bool = False):
    """JAX twin of ``fv3_native_d_sw.xppm`` (tp_core.F90:308-681).

    Functional: the NumPy lane writes ``flux`` in place; this twin
    RETURNS it.  Declared Fortran bounds, matching tp_core.F90:313-320
    exactly (the caller slices to these, as ``fv_tp_2d`` does when the
    oracle passes ``crx(is,js)`` at tp_core.F90:156):

    ==========  ==========================  ====================
    operand     Fortran bounds              array shape
    ==========  ==========================  ====================
    ``q``       ``(isd:ied, jfirst:jlast)`` ``(ied-isd+1, nj)``
    ``c``       ``(is:ie+1, jfirst:jlast)`` ``(ie-is_+2, nj)``
    ``dxa``     ``(isd:ied, jsd:jed)``      ``(ied-isd+1, jed-jsd+1)``
    RETURNS     ``(is:ie+1, jfirst:jlast)`` ``(ie-is_+2, nj)``
    ==========  ==========================  ====================

    with ``nj = jlast - jfirst + 1``.

    Dependence (correction 1): the oracle's outer ``do 666 j`` writes
    only ``flux(i,j)`` and the per-j locals ``q1``/``al``/``bl``/``br``/
    ``b0``/``smt5``/``smt6``/``fx1``/``xt1``/``dm``/``dq``; it reads only
    ``q``, ``c`` and ``dxa``, none of which it writes, and ``flux`` is
    INTENT(OUT) and never read.  No iteration reads another's write, so
    the j loop is vectorised onto a trailing axis.  ``q1(i) = q(i,j)``
    (tp_core.F90:314-316) is a copy of an operand that is never written,
    so it is ``q`` itself here.  Every INNER ``do i`` is likewise a pure
    map: each writes ``X(i)`` and reads only cells of arrays it does not
    write in the SAME loop (e.g. ``bl(i)=al(i)-q1(i)`` writes ``bl``,
    reads ``al``/``q1``).  Where the oracle splits one stage into two
    loops (``smt5`` then the flux), the split is preserved here so the
    second slice always sees a fully-written first.

    ``iord``/``is_``/``ie``/``isd``/``ied``/``jfirst``/``jlast``/
    ``jsd``/``jed``/``npx``/``npy``/``bounded_domain``/``grid_type``/
    ``lim_fac``/``duogrid`` are STATIC: they select python branches or
    index windows and must never be traced.  ``lim_fac`` is a deck
    constant (fv_arrays.F90:355) and is static for the same reason the
    NH lane's ``p_fac`` is.

    Non-smooth sites are listed in the module docstring; the ones live
    in THIS routine are ``copysign`` (iord>=7 ``dm``, iord 8/11), the
    ``min``/``max`` clamps (``dm``, iord 9/10/13 and the edge ``xt``
    monotonicity clamps), the ``c > 0`` upwind select, and the
    ``smt5``/``smt6``/``hi5``/``hi6`` limiter flags, which make the flux
    DISCONTINUOUS (they add or drop the whole ``fx1`` term).
    """
    _nn = functools.partial(_nan, dtype=q.dtype)  # workspace follows storage dtype (fp32/fp64)
    _validate_ord("xppm", "iord", iord, _PPM_ORDS)
    require_uniform_float_jax("xppm", {"q": q, "c": c, "dxa": dxa})
    q = jnp.asarray(q)
    c = jnp.asarray(c)
    dxa = jnp.asarray(dxa)

    nj = jlast - jfirst + 1
    n_q = ied - isd + 1
    n_c = ie + 1 - is_ + 1
    if q.shape != (n_q, nj):
        raise ValueError(f"xppm: q must be {(n_q, nj)}, got {q.shape}")
    if c.shape != (n_c, nj):
        raise ValueError(f"xppm: c must be {(n_c, nj)}, got {c.shape}")
    if dxa.shape != (n_q, jed - jsd + 1):
        raise ValueError(
            f"xppm: dxa must be {(n_q, jed - jsd + 1)}, got {dxa.shape}")

    # :333-339 (verbatim in fv3_native_d_sw.py:272-279)
    if (not (bounded_domain or duogrid)) and grid_type < 3:
        is1 = max(3, is_ - 1)
        ie3 = min(npx - 2, ie + 2)
        ie1 = min(npx - 3, ie + 1)
    else:
        is1 = is_ - 1
        ie3 = ie + 2
        ie1 = ie + 1

    mord = abs(iord)
    edge_arm = (not (bounded_domain or duogrid)) and grid_type < 3

    # dxa is read only at the CURRENT j, so it is restricted once to the
    # j window this call owns (tp_core.F90:318 declares jsd:jed).
    dxa_j = dxa[:, _rng(jfirst, jlast, jsd, jed - jsd + 1, "xppm dxa j")]

    def q_w(a, b):
        return q[_rng(a, b, isd, n_q, "xppm q i"), :]

    def q_at(i):
        return q[_idx(i, isd, n_q, "xppm q i"), :]

    def dxa_at(i):
        return dxa_j[_idx(i, isd, n_q, "xppm dxa i"), :]

    o_al = is_ - 1                       # al(is-1:ie+2)
    n_al = ie + 2 - (is_ - 1) + 1
    o_b = is_ - 1                        # bl/br/b0/smt5/smt6(is-1:ie+1)
    n_b = ie + 1 - (is_ - 1) + 1

    def al_w(a, b):
        return al[_rng(a, b, o_al, n_al, "xppm al"), :]

    def b_w(arr, a, b):
        return arr[_rng(a, b, o_b, n_b, "xppm bl/br/b0"), :]

    cw = c                               # c(is:ie+1, .) -- the whole array
    cpos = cw > 0.0

    if iord < 7:
        # ---------------------------------------------------- :341-478
        al = _nn((n_al, nj))
        al = al.at[_rng(is1, ie3, o_al, n_al, "xppm al main"), :].set(
            TP_P1 * (q_w(is1 - 1, ie3 - 1) + q_w(is1, ie3))
            + TP_P2 * (q_w(is1 - 2, ie3 - 2) + q_w(is1 + 1, ie3 + 1)))

        if edge_arm:                                        # :346-364
            if is_ == 1:
                al = al.at[_idx(0, o_al, n_al, "xppm al"), :].set(
                    TP_C1 * q_at(-2) + TP_C2 * q_at(-1) + TP_C3 * q_at(0))
                al = al.at[_idx(1, o_al, n_al, "xppm al"), :].set(
                    0.5 * (((2.0 * dxa_at(0) + dxa_at(-1)) * q_at(0)
                            - dxa_at(0) * q_at(-1)) / (dxa_at(-1) + dxa_at(0))
                           + ((2.0 * dxa_at(1) + dxa_at(2)) * q_at(1)
                              - dxa_at(1) * q_at(2)) / (dxa_at(1) + dxa_at(2))))
                al = al.at[_idx(2, o_al, n_al, "xppm al"), :].set(
                    TP_C3 * q_at(1) + TP_C2 * q_at(2) + TP_C1 * q_at(3))
            if (ie + 1) == npx:
                al = al.at[_idx(npx - 1, o_al, n_al, "xppm al"), :].set(
                    TP_C1 * q_at(npx - 3) + TP_C2 * q_at(npx - 2)
                    + TP_C3 * q_at(npx - 1))
                al = al.at[_idx(npx, o_al, n_al, "xppm al"), :].set(
                    0.5 * (((2.0 * dxa_at(npx - 1) + dxa_at(npx - 2))
                            * q_at(npx - 1) - dxa_at(npx - 1) * q_at(npx - 2))
                           / (dxa_at(npx - 2) + dxa_at(npx - 1))
                           + ((2.0 * dxa_at(npx) + dxa_at(npx + 1)) * q_at(npx)
                              - dxa_at(npx) * q_at(npx + 1))
                           / (dxa_at(npx) + dxa_at(npx + 1))))
                al = al.at[_idx(npx + 1, o_al, n_al, "xppm al"), :].set(
                    TP_C3 * q_at(npx) + TP_C2 * q_at(npx + 1)
                    + TP_C1 * q_at(npx + 2))

        if iord < 0:                                        # :366-370
            # The Fortran window is-1..ie+2 IS al's whole declaration.
            # NOTE the deliberate NaN divergence documented in the module
            # docstring: python max(0., nan) == 0.0, jnp.maximum -> nan.
            # Only reachable on cells the oracle never wrote.
            al = jnp.maximum(0.0, al)

        if mord == 1:                                       # :372-392
            bl = al_w(is_ - 1, ie + 1) - q_w(is_ - 1, ie + 1)
            br = al_w(is_, ie + 2) - q_w(is_ - 1, ie + 1)
            b0 = bl + br
            smt5 = jnp.abs(lim_fac * b0) < jnp.abs(bl - br)
            fx1 = jnp.where(
                cpos,
                (1.0 - cw) * (b_w(br, is_ - 1, ie) - cw * b_w(b0, is_ - 1, ie)),
                (1.0 + cw) * (b_w(bl, is_, ie + 1) + cw * b_w(b0, is_, ie + 1)))
            flux = jnp.where(cpos, q_w(is_ - 1, ie), q_w(is_, ie + 1))
            add = b_w(smt5, is_ - 1, ie) | b_w(smt5, is_, ie + 1)
            return jnp.where(add, flux + fx1, flux)

        if mord == 2:                                       # :394-410
            xt = cw
            qtmp_p = q_w(is_ - 1, ie)
            qtmp_m = q_w(is_, ie + 1)
            return jnp.where(
                xt > 0.0,
                qtmp_p + (1.0 - xt) * (
                    al_w(is_, ie + 1) - qtmp_p
                    - xt * (al_w(is_ - 1, ie) + al_w(is_, ie + 1)
                            - (qtmp_p + qtmp_p))),
                qtmp_m + (1.0 + xt) * (
                    al_w(is_, ie + 1) - qtmp_m
                    + xt * (al_w(is_, ie + 1) + al_w(is_ + 1, ie + 2)
                            - (qtmp_m + qtmp_m))))

        bl = al_w(is_ - 1, ie + 1) - q_w(is_ - 1, ie + 1)
        br = al_w(is_, ie + 2) - q_w(is_ - 1, ie + 1)
        b0 = bl + br

        if mord == 3:                                       # :412-433
            x0 = jnp.abs(b0)
            xt = jnp.abs(bl - br)
            smt5 = x0 < xt
            smt6 = 3.0 * x0 < xt
            xt1 = cw
            return jnp.where(
                xt1 > 0.0,
                jnp.where(
                    b_w(smt5, is_ - 1, ie) | b_w(smt6, is_, ie + 1),
                    q_w(is_ - 1, ie) + (1.0 - xt1) * (
                        b_w(br, is_ - 1, ie) - xt1 * b_w(b0, is_ - 1, ie)),
                    q_w(is_ - 1, ie)),
                jnp.where(
                    b_w(smt6, is_ - 1, ie) | b_w(smt5, is_, ie + 1),
                    q_w(is_, ie + 1) + (1.0 + xt1) * (
                        b_w(bl, is_, ie + 1) + xt1 * b_w(b0, is_, ie + 1)),
                    q_w(is_, ie + 1)))

        if mord == 4:                                       # :435-460
            x0 = jnp.abs(b0)
            xt = jnp.abs(bl - br)
            smt5 = x0 < xt
            smt6 = 3.0 * x0 < xt
            xt1 = cw
            hi5 = b_w(smt5, is_ - 1, ie) & b_w(smt5, is_, ie + 1)
            hi6 = b_w(smt6, is_ - 1, ie) | b_w(smt6, is_, ie + 1)
            hi5 = hi5 | hi6
            fx1 = jnp.where(
                xt1 > 0.0,
                (1.0 - xt1) * (b_w(br, is_ - 1, ie)
                               - xt1 * b_w(b0, is_ - 1, ie)),
                (1.0 + xt1) * (b_w(bl, is_, ie + 1)
                               + xt1 * b_w(b0, is_, ie + 1)))
            flux = jnp.where(xt1 > 0.0, q_w(is_ - 1, ie), q_w(is_, ie + 1))
            return jnp.where(hi5, flux + fx1, flux)

        # ------------------------------------------------ mord 5, 6
        if iord == 5:                                       # :464-470
            smt5 = bl * br < 0.0
        else:
            if iord == -5:                                  # :472-495
                da1 = br - bl
                a4 = -3.0 * b0
                smt5 = bl * br < 0.0
                inner = jnp.abs(da1) < -a4          # implies a4 < 0
                # double-where (correction 2): a4 is used only where
                # `inner` holds, and there a4 < 0 strictly.
                a4_safe = jnp.where(inner, a4, -1.0)
                hit = inner & (q_w(is_ - 1, ie + 1)
                               + 0.25 / a4_safe * da1 ** 2
                               + a4 * R12 < 0.0)
                zero = hit & (~smt5)
                pos_da = hit & smt5 & (da1 > 0.0)
                neg_da = hit & smt5 & (~(da1 > 0.0))
                br_n = jnp.where(zero, 0.0,
                                 jnp.where(pos_da, -2.0 * bl, br))
                bl_n = jnp.where(zero, 0.0,
                                 jnp.where(neg_da, -2.0 * br, bl))
                b0_n = jnp.where(
                    zero, 0.0,
                    jnp.where(pos_da, -bl, jnp.where(neg_da, -br, b0)))
                bl, br, b0 = bl_n, br_n, b0_n
            else:                                           # :497-503
                smt5 = 3.0 * jnp.abs(b0) < jnp.abs(bl - br)

            if edge_arm:                        # WMP edge fix :505-514
                if is_ == 1:
                    for k in (0, 1):
                        p = _idx(k, o_b, n_b, "xppm smt5")
                        smt5 = smt5.at[p, :].set(bl[p, :] * br[p, :] < 0.0)
                if (ie + 1) == npx:
                    for k in (npx - 1, npx):
                        p = _idx(k, o_b, n_b, "xppm smt5")
                        smt5 = smt5.at[p, :].set(bl[p, :] * br[p, :] < 0.0)

        fx1 = jnp.where(
            cpos,
            (1.0 - cw) * (b_w(br, is_ - 1, ie) - cw * b_w(b0, is_ - 1, ie)),
            (1.0 + cw) * (b_w(bl, is_, ie + 1) + cw * b_w(b0, is_, ie + 1)))
        flux = jnp.where(cpos, q_w(is_ - 1, ie), q_w(is_, ie + 1))
        add = b_w(smt5, is_ - 1, ie) | b_w(smt5, is_, ie + 1)
        return jnp.where(add, flux + fx1, flux)

    # ---------------------------------------------------- :523-673
    # Monotonic constraints (iord >= 7).
    o_dm = is_ - 2
    n_dm = ie + 2 - (is_ - 2) + 1
    o_dq = is_ - 3
    n_dq = ie + 2 - (is_ - 3) + 1

    def dm_w(a, b):
        return dm[_rng(a, b, o_dm, n_dm, "xppm dm"), :]

    def dq_w(a, b):
        return dq[_rng(a, b, o_dq, n_dq, "xppm dq"), :]

    # :527-533 -- dm over is-2..ie+2 (three-way monotonicity clamp; the
    # `min` ties and `copysign` are non-smooth sites).
    qm = q_w(is_ - 3, ie + 1)
    qc = q_w(is_ - 2, ie + 2)
    qp = q_w(is_ - 1, ie + 3)
    xt = 0.25 * (qp - qm)
    dm = _nn((n_dm, nj)).at[
        _rng(is_ - 2, ie + 2, o_dm, n_dm, "xppm dm main"), :].set(
        jnp.copysign(
            jnp.minimum(
                jnp.minimum(jnp.abs(xt),
                            jnp.maximum(jnp.maximum(qm, qc), qp) - qc),
                qc - jnp.minimum(jnp.minimum(qm, qc), qp)),
            xt))

    # :535-537 -- al over is1..ie1+1
    al = _nn((n_al, nj)).at[
        _rng(is1, ie1 + 1, o_al, n_al, "xppm al mono"), :].set(
        0.5 * (q_w(is1 - 1, ie1) + q_w(is1, ie1 + 1))
        + R3 * (dm_w(is1 - 1, ie1) - dm_w(is1, ie1 + 1)))

    bl = _nn((n_b, nj))
    br = _nn((n_b, nj))
    w_b = _rng(is1, ie1, o_b, n_b, "xppm bl/br mono")
    dq = _nn((n_dq, nj))

    if iord == 8:                                           # :539-545
        xt = 2.0 * dm_w(is1, ie1)
        bl = bl.at[w_b, :].set(-jnp.copysign(
            jnp.minimum(jnp.abs(xt), jnp.abs(al_w(is1, ie1) - q_w(is1, ie1))),
            xt))
        br = br.at[w_b, :].set(jnp.copysign(
            jnp.minimum(jnp.abs(xt),
                        jnp.abs(al_w(is1 + 1, ie1 + 1) - q_w(is1, ie1))),
            xt))
    elif iord == 10:                                        # :547-568
        dq = dq.at[
            _rng(is1 - 2, ie1 + 1, o_dq, n_dq, "xppm dq"), :].set(
            2.0 * (q_w(is1 - 1, ie1 + 2) - q_w(is1 - 2, ie1 + 1)))
        bl_w = al_w(is1, ie1) - q_w(is1, ie1)
        br_w = al_w(is1 + 1, ie1 + 1) - q_w(is1, ie1)
        flat = (jnp.abs(dm_w(is1 - 1, ie1 - 1)) + jnp.abs(dm_w(is1, ie1))
                + jnp.abs(dm_w(is1 + 1, ie1 + 1))) < NEAR_ZERO_TP
        steep = jnp.abs(3.0 * (bl_w + br_w)) > jnp.abs(bl_w - br_w)
        pmp_2 = dq_w(is1 - 1, ie1 - 1)
        lac_2 = pmp_2 - 0.75 * dq_w(is1 - 2, ie1 - 2)
        br_c = jnp.minimum(
            jnp.maximum(jnp.maximum(0.0, pmp_2), lac_2),
            jnp.maximum(br_w,
                        jnp.minimum(jnp.minimum(0.0, pmp_2), lac_2)))
        pmp_1 = -dq_w(is1, ie1)
        lac_1 = pmp_1 + 0.75 * dq_w(is1 + 1, ie1 + 1)
        bl_c = jnp.minimum(
            jnp.maximum(jnp.maximum(0.0, pmp_1), lac_1),
            jnp.maximum(bl_w,
                        jnp.minimum(jnp.minimum(0.0, pmp_1), lac_1)))
        use_c = (~flat) & steep
        bl = bl.at[w_b, :].set(
            jnp.where(flat, 0.0, jnp.where(use_c, bl_c, bl_w)))
        br = br.at[w_b, :].set(
            jnp.where(flat, 0.0, jnp.where(use_c, br_c, br_w)))
    elif iord == 11:                                        # :570-577
        xt = PPM_FAC * dm_w(is1, ie1)
        bl = bl.at[w_b, :].set(-jnp.copysign(
            jnp.minimum(jnp.abs(xt), jnp.abs(al_w(is1, ie1) - q_w(is1, ie1))),
            xt))
        br = br.at[w_b, :].set(jnp.copysign(
            jnp.minimum(jnp.abs(xt),
                        jnp.abs(al_w(is1 + 1, ie1 + 1) - q_w(is1, ie1))),
            xt))
    elif iord == 7 or iord == 12:                           # :579-601
        bl_w = al_w(is1, ie1) - q_w(is1, ie1)
        br_w = al_w(is1 + 1, ie1 + 1) - q_w(is1, ie1)
        a4 = -3.0 * (bl_w + br_w)
        da1 = br_w - bl_w
        ext5 = br_w * bl_w > 0.0
        ext6 = jnp.abs(da1) < -a4                    # implies a4 < 0
        # double-where (correction 2), same argument as pert_ppm.
        a4_safe = jnp.where(ext6, a4, -1.0)
        hit = ext6 & (q_w(is1, ie1) + 0.25 / a4_safe * da1 ** 2
                      + a4 * R12 < 0.0)
        zero = hit & ext5
        pos_da = hit & (~ext5) & (da1 > 0.0)
        neg_da = hit & (~ext5) & (~(da1 > 0.0))
        bl = bl.at[w_b, :].set(
            jnp.where(zero, 0.0, jnp.where(neg_da, -2.0 * br_w, bl_w)))
        br = br.at[w_b, :].set(
            jnp.where(zero, 0.0, jnp.where(pos_da, -2.0 * bl_w, br_w)))
    else:                                                   # :603-607
        bl = bl.at[w_b, :].set(al_w(is1, ie1) - q_w(is1, ie1))
        br = br.at[w_b, :].set(al_w(is1 + 1, ie1 + 1) - q_w(is1, ie1))

    if iord == 9 or iord == 13:                             # :609-612
        # pert_ppm(ie1-is1+1, q1(is1), bl(is1), br(is1), 0) -- elementwise
        # over the same run, so the 1-D view becomes the (i, j) block.
        bl_p, br_p = pert_ppm(q_w(is1, ie1), bl[w_b, :], br[w_b, :], 0)
        bl = bl.at[w_b, :].set(bl_p)
        br = br.at[w_b, :].set(br_p)

    if edge_arm:                                            # :614-647
        if is_ == 1:
            p0 = _idx(0, o_b, n_b, "xppm bl/br")
            p1 = _idx(1, o_b, n_b, "xppm bl/br")
            p2 = _idx(2, o_b, n_b, "xppm bl/br")
            bl = bl.at[p0, :].set(
                TP_S14 * dm_w(-1, -1)[0] + TP_S11 * (q_at(-1) - q_at(0)))
            xt = 0.5 * (((2.0 * dxa_at(0) + dxa_at(-1)) * q_at(0)
                         - dxa_at(0) * q_at(-1)) / (dxa_at(-1) + dxa_at(0))
                        + ((2.0 * dxa_at(1) + dxa_at(2)) * q_at(1)
                           - dxa_at(1) * q_at(2)) / (dxa_at(1) + dxa_at(2)))
            lo4 = jnp.minimum(jnp.minimum(jnp.minimum(q_at(-1), q_at(0)),
                                          q_at(1)), q_at(2))
            hi4 = jnp.maximum(jnp.maximum(jnp.maximum(q_at(-1), q_at(0)),
                                          q_at(1)), q_at(2))
            xt = jnp.maximum(xt, lo4)
            xt = jnp.minimum(xt, hi4)
            br = br.at[p0, :].set(xt - q_at(0))
            bl = bl.at[p1, :].set(xt - q_at(1))
            xt = TP_S15 * q_at(1) + TP_S11 * q_at(2) - TP_S14 * dm_w(2, 2)[0]
            br = br.at[p1, :].set(xt - q_at(1))
            bl = bl.at[p2, :].set(xt - q_at(2))
            br = br.at[p2, :].set(al_w(3, 3)[0] - q_at(2))
            # pert_ppm(3, q1(0), bl(0), br(0), 1) over the 3-cell run.
            w3 = _rng(0, 2, o_b, n_b, "xppm pert run")
            bl_p, br_p = pert_ppm(q_w(0, 2), bl[w3, :], br[w3, :], 1)
            bl = bl.at[w3, :].set(bl_p)
            br = br.at[w3, :].set(br_p)
        if (ie + 1) == npx:
            pm2 = _idx(npx - 2, o_b, n_b, "xppm bl/br")
            pm1 = _idx(npx - 1, o_b, n_b, "xppm bl/br")
            pm0 = _idx(npx, o_b, n_b, "xppm bl/br")
            bl = bl.at[pm2, :].set(al_w(npx - 2, npx - 2)[0] - q_at(npx - 2))
            xt = (TP_S15 * q_at(npx - 1) + TP_S11 * q_at(npx - 2)
                  + TP_S14 * dm_w(npx - 2, npx - 2)[0])
            br = br.at[pm2, :].set(xt - q_at(npx - 2))
            bl = bl.at[pm1, :].set(xt - q_at(npx - 1))
            xt = 0.5 * (((2.0 * dxa_at(npx - 1) + dxa_at(npx - 2))
                         * q_at(npx - 1) - dxa_at(npx - 1) * q_at(npx - 2))
                        / (dxa_at(npx - 2) + dxa_at(npx - 1))
                        + ((2.0 * dxa_at(npx) + dxa_at(npx + 1)) * q_at(npx)
                           - dxa_at(npx) * q_at(npx + 1))
                        / (dxa_at(npx) + dxa_at(npx + 1)))
            lo4 = jnp.minimum(
                jnp.minimum(jnp.minimum(q_at(npx - 2), q_at(npx - 1)),
                            q_at(npx)), q_at(npx + 1))
            hi4 = jnp.maximum(
                jnp.maximum(jnp.maximum(q_at(npx - 2), q_at(npx - 1)),
                            q_at(npx)), q_at(npx + 1))
            xt = jnp.maximum(xt, lo4)
            xt = jnp.minimum(xt, hi4)
            br = br.at[pm1, :].set(xt - q_at(npx - 1))
            bl = bl.at[pm0, :].set(xt - q_at(npx))
            br = br.at[pm0, :].set(
                TP_S11 * (q_at(npx + 1) - q_at(npx))
                - TP_S14 * dm_w(npx + 1, npx + 1)[0])
            w3 = _rng(npx - 2, npx, o_b, n_b, "xppm pert run")
            bl_p, br_p = pert_ppm(q_w(npx - 2, npx), bl[w3, :], br[w3, :], 1)
            bl = bl.at[w3, :].set(bl_p)
            br = br.at[w3, :].set(br_p)

    if iord == 7:                                           # :651-666
        b0 = bl + br
        smt5 = bl * br < 0.0
        fx1 = jnp.where(
            cpos,
            (1.0 - cw) * (b_w(br, is_ - 1, ie) - cw * b_w(b0, is_ - 1, ie)),
            (1.0 + cw) * (b_w(bl, is_, ie + 1) + cw * b_w(b0, is_, ie + 1)))
        flux = jnp.where(cpos, q_w(is_ - 1, ie), q_w(is_, ie + 1))
        add = b_w(smt5, is_ - 1, ie) | b_w(smt5, is_, ie + 1)
        return jnp.where(add, flux + fx1, flux)

    # :668-673 -- b0 is NOT formed; (bl+br) is recomputed inline.
    return jnp.where(
        cpos,
        q_w(is_ - 1, ie) + (1.0 - cw) * (
            b_w(br, is_ - 1, ie)
            - cw * (b_w(bl, is_ - 1, ie) + b_w(br, is_ - 1, ie))),
        q_w(is_, ie + 1) + (1.0 + cw) * (
            b_w(bl, is_, ie + 1)
            + cw * (b_w(bl, is_, ie + 1) + b_w(br, is_, ie + 1))))


def yppm(q, c, jord: int, ifirst: int, ilast: int, isd: int, ied: int,
         js: int, je: int, jsd: int, jed: int, npx: int, npy: int, dya,
         bounded_domain: bool, grid_type: int, lim_fac: float, *,
         duogrid: bool = False):
    """JAX twin of ``fv3_native_d_sw.yppm`` (tp_core.F90:684-1102).

    Functional: RETURNS ``flux``.  Declared Fortran bounds (the oracle
    passes the FULL ``cry``/``dya`` and this routine takes the ``i``
    slice itself, exactly as the Fortran dummy declarations do):

    ==========  ============================  ======================
    operand     Fortran bounds                array shape
    ==========  ============================  ======================
    ``q``       ``(ifirst:ilast, jsd:jed)``   ``(ni, jed-jsd+1)``
    ``c``       ``(isd:ied, js:je+1)``        ``(ied-isd+1, je-js+2)``
    ``dya``     ``(isd:ied, jsd:jed)``        ``(ied-isd+1, jed-jsd+1)``
    RETURNS     ``(ifirst:ilast, js:je+1)``   ``(ni, je-js+2)``
    ==========  ============================  ======================

    with ``ni = ilast - ifirst + 1``.

    Dependence (correction 1): unlike ``xppm``, the oracle's ``yppm``
    has no outer j loop -- it is a sequence of full 2-D ``do j: do i``
    nests over 2-D locals.  Each nest writes ONE local (``al``, ``bl``/
    ``br``/``b0``, ``dm``, ``dq``, ``smt5``/``smt6``, ``flux``) and reads
    only ``q``, ``c``, ``dya`` and locals written by an EARLIER nest;
    none reads a cell its own nest writes.  The nests are therefore
    vectorised individually and kept in the oracle's order, so a later
    slice always sees a fully-written earlier local.  The 1-D per-j
    scratch (``fx1``/``xt1``/``a4``/``hi5``/``hi6``) is reused across j
    in the Fortran but written before use inside every j iteration, so
    promoting it to a 2-D (i, j) block is exact.

    Everything else -- the static selectors, the NaN fill, the
    double-``where`` at ``jord in (-5, 7, 12)``, and the non-smooth-site
    list -- is as documented for :func:`xppm` and in the module
    docstring.
    """
    _nn = functools.partial(_nan, dtype=q.dtype)  # workspace follows storage dtype (fp32/fp64)
    _validate_ord("yppm", "jord", jord, _PPM_ORDS)
    require_uniform_float_jax("yppm", {"q": q, "c": c, "dya": dya})
    q = jnp.asarray(q)
    c = jnp.asarray(c)
    dya = jnp.asarray(dya)

    ni = ilast - ifirst + 1
    n_qj = jed - jsd + 1
    n_cj = je + 1 - js + 1
    n_di = ied - isd + 1
    if q.shape != (ni, n_qj):
        raise ValueError(f"yppm: q must be {(ni, n_qj)}, got {q.shape}")
    if c.shape != (n_di, n_cj):
        raise ValueError(f"yppm: c must be {(n_di, n_cj)}, got {c.shape}")
    if dya.shape != (n_di, n_qj):
        raise ValueError(
            f"yppm: dya must be {(n_di, n_qj)}, got {dya.shape}")

    # :709-715 (verbatim in fv3_native_d_sw.py:667-676)
    if (not (bounded_domain or duogrid)) and grid_type < 3:
        js1 = max(3, js - 1)
        je3 = min(npy - 2, je + 2)
        je1 = min(npy - 3, je + 1)
    else:
        js1 = js - 1
        je3 = je + 2
        je1 = je + 1

    mord = abs(jord)
    edge_arm = (not (bounded_domain or duogrid)) and grid_type < 3

    # The i window this call owns; the oracle's dummy declarations give
    # c/dya the FULL isd:ied extent and index them at i in ifirst:ilast.
    i_w = _rng(ifirst, ilast, isd, n_di, "yppm i window")
    cw = c[i_w, :]                        # c(ifirst:ilast, js:je+1)
    dya_i = dya[i_w, :]                   # dya(ifirst:ilast, jsd:jed)

    def q_w(a, b):
        return q[:, _rng(a, b, jsd, n_qj, "yppm q j")]

    def q_at(j):
        return q[:, _idx(j, jsd, n_qj, "yppm q j")]

    def dya_at(j):
        return dya_i[:, _idx(j, jsd, n_qj, "yppm dya j")]

    o_al = js - 1                        # al(., js-1:je+2)
    n_al = je + 2 - (js - 1) + 1
    o_b = js - 1                         # bl/br/b0/smt5/smt6(., js-1:je+1)
    n_b = je + 1 - (js - 1) + 1

    def al_w(a, b):
        return al[:, _rng(a, b, o_al, n_al, "yppm al")]

    def b_w(arr, a, b):
        return arr[:, _rng(a, b, o_b, n_b, "yppm bl/br/b0")]

    cpos = cw > 0.0

    if jord < 7:
        # ---------------------------------------------------- :717-949
        al = _nn((ni, n_al))
        al = al.at[:, _rng(js1, je3, o_al, n_al, "yppm al main")].set(
            TP_P1 * (q_w(js1 - 1, je3 - 1) + q_w(js1, je3))
            + TP_P2 * (q_w(js1 - 2, je3 - 2) + q_w(js1 + 1, je3 + 1)))

        if edge_arm:                                        # :726-744
            if js == 1:
                al = al.at[:, _idx(0, o_al, n_al, "yppm al")].set(
                    TP_C1 * q_at(-2) + TP_C2 * q_at(-1) + TP_C3 * q_at(0))
                al = al.at[:, _idx(1, o_al, n_al, "yppm al")].set(
                    0.5 * (((2.0 * dya_at(0) + dya_at(-1)) * q_at(0)
                            - dya_at(0) * q_at(-1)) / (dya_at(-1) + dya_at(0))
                           + ((2.0 * dya_at(1) + dya_at(2)) * q_at(1)
                              - dya_at(1) * q_at(2)) / (dya_at(1) + dya_at(2))))
                al = al.at[:, _idx(2, o_al, n_al, "yppm al")].set(
                    TP_C3 * q_at(1) + TP_C2 * q_at(2) + TP_C1 * q_at(3))
            if (je + 1) == npy:
                al = al.at[:, _idx(npy - 1, o_al, n_al, "yppm al")].set(
                    TP_C1 * q_at(npy - 3) + TP_C2 * q_at(npy - 2)
                    + TP_C3 * q_at(npy - 1))
                al = al.at[:, _idx(npy, o_al, n_al, "yppm al")].set(
                    0.5 * (((2.0 * dya_at(npy - 1) + dya_at(npy - 2))
                            * q_at(npy - 1) - dya_at(npy - 1) * q_at(npy - 2))
                           / (dya_at(npy - 2) + dya_at(npy - 1))
                           + ((2.0 * dya_at(npy) + dya_at(npy + 1)) * q_at(npy)
                              - dya_at(npy) * q_at(npy + 1))
                           / (dya_at(npy) + dya_at(npy + 1))))
                al = al.at[:, _idx(npy + 1, o_al, n_al, "yppm al")].set(
                    TP_C3 * q_at(npy) + TP_C2 * q_at(npy + 1)
                    + TP_C1 * q_at(npy + 2))

        if jord < 0:                                        # :746-750
            # Same deliberate NaN divergence as xppm (module docstring).
            al = jnp.maximum(0.0, al)

        if mord == 1:                                       # :752-771
            bl = al_w(js - 1, je + 1) - q_w(js - 1, je + 1)
            br = al_w(js, je + 2) - q_w(js - 1, je + 1)
            b0 = bl + br
            smt5 = jnp.abs(lim_fac * b0) < jnp.abs(bl - br)
            fx1 = jnp.where(
                cpos,
                (1.0 - cw) * (b_w(br, js - 1, je) - cw * b_w(b0, js - 1, je)),
                (1.0 + cw) * (b_w(bl, js, je + 1) + cw * b_w(b0, js, je + 1)))
            flux = jnp.where(cpos, q_w(js - 1, je), q_w(js, je + 1))
            add = b_w(smt5, js - 1, je) | b_w(smt5, js, je + 1)
            return jnp.where(add, flux + fx1, flux)

        if mord == 2:                                       # :773-789
            xt = cw
            qtmp_p = q_w(js - 1, je)
            qtmp_m = q_w(js, je + 1)
            return jnp.where(
                xt > 0.0,
                qtmp_p + (1.0 - xt) * (
                    al_w(js, je + 1) - qtmp_p
                    - xt * (al_w(js - 1, je) + al_w(js, je + 1)
                            - (qtmp_p + qtmp_p))),
                qtmp_m + (1.0 + xt) * (
                    al_w(js, je + 1) - qtmp_m
                    + xt * (al_w(js, je + 1) + al_w(js + 1, je + 2)
                            - (qtmp_m + qtmp_m))))

        bl = al_w(js - 1, je + 1) - q_w(js - 1, je + 1)
        br = al_w(js, je + 2) - q_w(js - 1, je + 1)
        b0 = bl + br

        if mord == 3:                                       # :791-816
            x0 = jnp.abs(b0)
            xt = jnp.abs(bl - br)
            smt5 = x0 < xt
            smt6 = 3.0 * x0 < xt
            xt1 = cw
            return jnp.where(
                xt1 > 0.0,
                jnp.where(
                    b_w(smt5, js - 1, je) | b_w(smt6, js, je + 1),
                    q_w(js - 1, je) + (1.0 - xt1) * (
                        b_w(br, js - 1, je) - xt1 * b_w(b0, js - 1, je)),
                    q_w(js - 1, je)),
                jnp.where(
                    b_w(smt6, js - 1, je) | b_w(smt5, js, je + 1),
                    q_w(js, je + 1) + (1.0 + xt1) * (
                        b_w(bl, js, je + 1) + xt1 * b_w(b0, js, je + 1)),
                    q_w(js, je + 1)))

        if mord == 4:                                       # :818-844
            x0 = jnp.abs(b0)
            xt = jnp.abs(bl - br)
            smt5 = x0 < xt
            smt6 = 3.0 * x0 < xt
            xt1 = cw
            hi5 = b_w(smt5, js - 1, je) & b_w(smt5, js, je + 1)
            hi6 = b_w(smt6, js - 1, je) | b_w(smt6, js, je + 1)
            hi5 = hi5 | hi6
            fx1 = jnp.where(
                xt1 > 0.0,
                (1.0 - xt1) * (b_w(br, js - 1, je)
                               - xt1 * b_w(b0, js - 1, je)),
                (1.0 + xt1) * (b_w(bl, js, je + 1)
                               + xt1 * b_w(b0, js, je + 1)))
            flux = jnp.where(xt1 > 0.0, q_w(js - 1, je), q_w(js, je + 1))
            return jnp.where(hi5, flux + fx1, flux)

        # ------------------------------------------------ mord 5, 6
        if jord == 5:                                       # :847-853
            smt5 = bl * br < 0.0
        else:
            if jord == -5:                                  # :855-878
                xt1 = br - bl
                a4 = -3.0 * b0
                smt5 = bl * br < 0.0
                inner = jnp.abs(xt1) < -a4          # implies a4 < 0
                a4_safe = jnp.where(inner, a4, -1.0)   # double-where
                hit = inner & (q_w(js - 1, je + 1)
                               + 0.25 / a4_safe * xt1 ** 2
                               + a4 * R12 < 0.0)
                zero = hit & (~smt5)
                pos_da = hit & smt5 & (xt1 > 0.0)
                neg_da = hit & smt5 & (~(xt1 > 0.0))
                br_n = jnp.where(zero, 0.0,
                                 jnp.where(pos_da, -2.0 * bl, br))
                bl_n = jnp.where(zero, 0.0,
                                 jnp.where(neg_da, -2.0 * br, bl))
                b0_n = jnp.where(
                    zero, 0.0,
                    jnp.where(pos_da, -bl, jnp.where(neg_da, -br, b0)))
                bl, br, b0 = bl_n, br_n, b0_n
            else:                                           # :880-887
                smt5 = 3.0 * jnp.abs(b0) < jnp.abs(bl - br)

            if edge_arm:                        # WMP edge fix :889-899
                if js == 1:
                    for k in (0, 1):
                        p = _idx(k, o_b, n_b, "yppm smt5")
                        smt5 = smt5.at[:, p].set(
                            bl[:, p] * br[:, p] < 0.0)
                if (je + 1) == npy:
                    for k in (npy - 1, npy):
                        p = _idx(k, o_b, n_b, "yppm smt5")
                        smt5 = smt5.at[:, p].set(
                            bl[:, p] * br[:, p] < 0.0)

        fx1 = jnp.where(
            cpos,
            (1.0 - cw) * (b_w(br, js - 1, je) - cw * b_w(b0, js - 1, je)),
            (1.0 + cw) * (b_w(bl, js, je + 1) + cw * b_w(b0, js, je + 1)))
        flux = jnp.where(cpos, q_w(js - 1, je), q_w(js, je + 1))
        add = b_w(smt5, js - 1, je) | b_w(smt5, js, je + 1)
        return jnp.where(add, flux + fx1, flux)

    # ---------------------------------------------------- :951-1088
    o_dm = js - 2
    n_dm = je + 2 - (js - 2) + 1
    o_dq = js - 3
    n_dq = je + 2 - (js - 3) + 1

    def dm_w(a, b):
        return dm[:, _rng(a, b, o_dm, n_dm, "yppm dm")]

    def dq_w(a, b):
        return dq[:, _rng(a, b, o_dq, n_dq, "yppm dq")]

    # :963-971 -- dm over js-2..je+2
    qm = q_w(js - 3, je + 1)
    qc = q_w(js - 2, je + 2)
    qp = q_w(js - 1, je + 3)
    xt = 0.25 * (qp - qm)
    dm = _nn((ni, n_dm)).at[
        :, _rng(js - 2, je + 2, o_dm, n_dm, "yppm dm main")].set(
        jnp.copysign(
            jnp.minimum(
                jnp.minimum(jnp.abs(xt),
                            jnp.maximum(jnp.maximum(qm, qc), qp) - qc),
                qc - jnp.minimum(jnp.minimum(qm, qc), qp)),
            xt))

    # :973-977 -- al over js1..je1+1
    al = _nn((ni, n_al)).at[
        :, _rng(js1, je1 + 1, o_al, n_al, "yppm al mono")].set(
        0.5 * (q_w(js1 - 1, je1) + q_w(js1, je1 + 1))
        + R3 * (dm_w(js1 - 1, je1) - dm_w(js1, je1 + 1)))

    bl = _nn((ni, n_b))
    br = _nn((ni, n_b))
    w_b = _rng(js1, je1, o_b, n_b, "yppm bl/br mono")
    dq = _nn((ni, n_dq))

    if jord == 8:                                           # :979-987
        xt = 2.0 * dm_w(js1, je1)
        bl = bl.at[:, w_b].set(-jnp.copysign(
            jnp.minimum(jnp.abs(xt), jnp.abs(al_w(js1, je1) - q_w(js1, je1))),
            xt))
        br = br.at[:, w_b].set(jnp.copysign(
            jnp.minimum(jnp.abs(xt),
                        jnp.abs(al_w(js1 + 1, je1 + 1) - q_w(js1, je1))),
            xt))
    elif jord == 10:                                        # :989-1013
        dq = dq.at[
            :, _rng(js1 - 2, je1 + 1, o_dq, n_dq, "yppm dq")].set(
            2.0 * (q_w(js1 - 1, je1 + 2) - q_w(js1 - 2, je1 + 1)))
        bl_w = al_w(js1, je1) - q_w(js1, je1)
        br_w = al_w(js1 + 1, je1 + 1) - q_w(js1, je1)
        flat = (jnp.abs(dm_w(js1 - 1, je1 - 1)) + jnp.abs(dm_w(js1, je1))
                + jnp.abs(dm_w(js1 + 1, je1 + 1))) < NEAR_ZERO_TP
        steep = jnp.abs(3.0 * (bl_w + br_w)) > jnp.abs(bl_w - br_w)
        pmp_2 = dq_w(js1 - 1, je1 - 1)
        lac_2 = pmp_2 - 0.75 * dq_w(js1 - 2, je1 - 2)
        br_c = jnp.minimum(
            jnp.maximum(jnp.maximum(0.0, pmp_2), lac_2),
            jnp.maximum(br_w,
                        jnp.minimum(jnp.minimum(0.0, pmp_2), lac_2)))
        pmp_1 = -dq_w(js1, je1)
        lac_1 = pmp_1 + 0.75 * dq_w(js1 + 1, je1 + 1)
        bl_c = jnp.minimum(
            jnp.maximum(jnp.maximum(0.0, pmp_1), lac_1),
            jnp.maximum(bl_w,
                        jnp.minimum(jnp.minimum(0.0, pmp_1), lac_1)))
        use_c = (~flat) & steep
        bl = bl.at[:, w_b].set(
            jnp.where(flat, 0.0, jnp.where(use_c, bl_c, bl_w)))
        br = br.at[:, w_b].set(
            jnp.where(flat, 0.0, jnp.where(use_c, br_c, br_w)))
    elif jord == 11:                                        # :1015-1023
        xt = PPM_FAC * dm_w(js1, je1)
        bl = bl.at[:, w_b].set(-jnp.copysign(
            jnp.minimum(jnp.abs(xt), jnp.abs(al_w(js1, je1) - q_w(js1, je1))),
            xt))
        br = br.at[:, w_b].set(jnp.copysign(
            jnp.minimum(jnp.abs(xt),
                        jnp.abs(al_w(js1 + 1, je1 + 1) - q_w(js1, je1))),
            xt))
    elif jord == 7 or jord == 12:                           # :1025-1046
        bl_w = al_w(js1, je1) - q_w(js1, je1)
        br_w = al_w(js1 + 1, je1 + 1) - q_w(js1, je1)
        xt1 = br_w - bl_w
        a4 = -3.0 * (br_w + bl_w)
        hi5 = bl_w * br_w > 0.0
        hi6 = jnp.abs(xt1) < -a4                     # implies a4 < 0
        a4_safe = jnp.where(hi6, a4, -1.0)              # double-where
        hit = hi6 & (q_w(js1, je1) + 0.25 / a4_safe * xt1 ** 2
                     + a4 * R12 < 0.0)
        zero = hit & hi5
        pos_da = hit & (~hi5) & (xt1 > 0.0)
        neg_da = hit & (~hi5) & (~(xt1 > 0.0))
        bl = bl.at[:, w_b].set(
            jnp.where(zero, 0.0, jnp.where(neg_da, -2.0 * br_w, bl_w)))
        br = br.at[:, w_b].set(
            jnp.where(zero, 0.0, jnp.where(pos_da, -2.0 * bl_w, br_w)))
    else:                                                   # :1048-1053
        bl = bl.at[:, w_b].set(al_w(js1, je1) - q_w(js1, je1))
        br = br.at[:, w_b].set(al_w(js1 + 1, je1 + 1) - q_w(js1, je1))

    if jord == 9 or jord == 13:                             # :1055-1061
        bl_p, br_p = pert_ppm(q_w(js1, je1), bl[:, w_b], br[:, w_b], 0)
        bl = bl.at[:, w_b].set(bl_p)
        br = br.at[:, w_b].set(br_p)

    if edge_arm:                                            # :1063-1097
        if js == 1:
            p0 = _idx(0, o_b, n_b, "yppm bl/br")
            p1 = _idx(1, o_b, n_b, "yppm bl/br")
            p2 = _idx(2, o_b, n_b, "yppm bl/br")
            bl = bl.at[:, p0].set(
                TP_S14 * dm_w(-1, -1)[:, 0] + TP_S11 * (q_at(-1) - q_at(0)))
            xt = 0.5 * (((2.0 * dya_at(0) + dya_at(-1)) * q_at(0)
                         - dya_at(0) * q_at(-1)) / (dya_at(-1) + dya_at(0))
                        + ((2.0 * dya_at(1) + dya_at(2)) * q_at(1)
                           - dya_at(1) * q_at(2)) / (dya_at(1) + dya_at(2)))
            lo4 = jnp.minimum(jnp.minimum(jnp.minimum(q_at(-1), q_at(0)),
                                          q_at(1)), q_at(2))
            hi4 = jnp.maximum(jnp.maximum(jnp.maximum(q_at(-1), q_at(0)),
                                          q_at(1)), q_at(2))
            xt = jnp.maximum(xt, lo4)
            xt = jnp.minimum(xt, hi4)
            br = br.at[:, p0].set(xt - q_at(0))
            bl = bl.at[:, p1].set(xt - q_at(1))
            xt = TP_S15 * q_at(1) + TP_S11 * q_at(2) - TP_S14 * dm_w(2, 2)[:, 0]
            br = br.at[:, p1].set(xt - q_at(1))
            bl = bl.at[:, p2].set(xt - q_at(2))
            br = br.at[:, p2].set(al_w(3, 3)[:, 0] - q_at(2))
            # tp_core.F90:1084 runs ONE pert_ppm over the F-contiguous
            # 3-column run j=0,1,2; pert_ppm is elementwise, so the
            # (i, 3) block is exactly equivalent (same note as the
            # NumPy lane, fv3_native_d_sw.py:1020-1023).
            w3 = _rng(0, 2, o_b, n_b, "yppm pert run")
            bl_p, br_p = pert_ppm(q_w(0, 2), bl[:, w3], br[:, w3], 1)
            bl = bl.at[:, w3].set(bl_p)
            br = br.at[:, w3].set(br_p)
        if (je + 1) == npy:
            pm2 = _idx(npy - 2, o_b, n_b, "yppm bl/br")
            pm1 = _idx(npy - 1, o_b, n_b, "yppm bl/br")
            pm0 = _idx(npy, o_b, n_b, "yppm bl/br")
            bl = bl.at[:, pm2].set(al_w(npy - 2, npy - 2)[:, 0] - q_at(npy - 2))
            xt = (TP_S15 * q_at(npy - 1) + TP_S11 * q_at(npy - 2)
                  + TP_S14 * dm_w(npy - 2, npy - 2)[:, 0])
            br = br.at[:, pm2].set(xt - q_at(npy - 2))
            bl = bl.at[:, pm1].set(xt - q_at(npy - 1))
            xt = 0.5 * (((2.0 * dya_at(npy - 1) + dya_at(npy - 2))
                         * q_at(npy - 1) - dya_at(npy - 1) * q_at(npy - 2))
                        / (dya_at(npy - 2) + dya_at(npy - 1))
                        + ((2.0 * dya_at(npy) + dya_at(npy + 1)) * q_at(npy)
                           - dya_at(npy) * q_at(npy + 1))
                        / (dya_at(npy) + dya_at(npy + 1)))
            lo4 = jnp.minimum(
                jnp.minimum(jnp.minimum(q_at(npy - 2), q_at(npy - 1)),
                            q_at(npy)), q_at(npy + 1))
            hi4 = jnp.maximum(
                jnp.maximum(jnp.maximum(q_at(npy - 2), q_at(npy - 1)),
                            q_at(npy)), q_at(npy + 1))
            xt = jnp.maximum(xt, lo4)
            xt = jnp.minimum(xt, hi4)
            br = br.at[:, pm1].set(xt - q_at(npy - 1))
            bl = bl.at[:, pm0].set(xt - q_at(npy))
            br = br.at[:, pm0].set(
                TP_S11 * (q_at(npy + 1) - q_at(npy))
                - TP_S14 * dm_w(npy + 1, npy + 1)[:, 0])
            w3 = _rng(npy - 2, npy, o_b, n_b, "yppm pert run")
            bl_p, br_p = pert_ppm(q_w(npy - 2, npy), bl[:, w3], br[:, w3], 1)
            bl = bl.at[:, w3].set(bl_p)
            br = br.at[:, w3].set(br_p)

    if jord == 7:                                           # :1099-1117
        b0 = bl + br
        smt5 = bl * br < 0.0
        fx1 = jnp.where(
            cpos,
            (1.0 - cw) * (b_w(br, js - 1, je) - cw * b_w(b0, js - 1, je)),
            (1.0 + cw) * (b_w(bl, js, je + 1) + cw * b_w(b0, js, je + 1)))
        flux = jnp.where(cpos, q_w(js - 1, je), q_w(js, je + 1))
        add = b_w(smt5, js - 1, je) | b_w(smt5, js, je + 1)
        return jnp.where(add, flux + fx1, flux)

    return jnp.where(
        cpos,
        q_w(js - 1, je) + (1.0 - cw) * (
            b_w(br, js - 1, je)
            - cw * (b_w(bl, js - 1, je) + b_w(br, js - 1, je))),
        q_w(js, je + 1) + (1.0 + cw) * (
            b_w(bl, js, je + 1)
            + cw * (b_w(bl, js, je + 1) + b_w(br, js, je + 1))))


def deln_flux(nord: int, is_: int, ie: int, js: int, je: int, npx: int,
              npy: int, damp, q, fx, fy, del6_v, del6_u, rarea, bd,
              bounded_domain: bool, sw_corner: bool, se_corner: bool,
              nw_corner: bool, ne_corner: bool, mass=None, damp_km=None,
              *, duogrid: bool = False):
    """JAX twin of ``fv3_native_d_sw.deln_flux`` (tp_core.F90:1217-1365).

    Functional: RETURNS ``(fx, fy)``; the NumPy lane adds the diffusive
    fluxes onto them in place.  The NumPy lane's ``gridstruct`` dict is
    SPLIT into explicit arguments -- the three grid ARRAYS (``del6_v``,
    ``del6_u``, ``rarea``) stay dynamic, the five corner/domain FLAGS
    become static python bools.  A dict mixing traced arrays with python
    bools cannot be a jit operand at all (unhashable as static, and the
    bools would be traced as dynamic), so this is a signature adaptation
    with no arithmetic effect, not a numerics change.

    Declared Fortran bounds: ``q``/``rarea``/``mass``/``damp_km``
    ``(isd:ied, jsd:jed)``; ``del6_v`` ``(isd:ied+1, jsd:jed)``;
    ``del6_u`` ``(isd:ied, jsd:jed+1)``; ``fx`` ``(is:ie+1, js:je)``;
    ``fy`` ``(is:ie, js:je+1)``.  ``bd`` is the NumPy lane's ``Bounds``
    NamedTuple (hashable by value -> one jit cache entry per value).

    ``damp_km`` is NOT in the pinned symmetryclean ``deln_flux``
    (tp_core.F90:1217 takes ``mass`` as its only optional); it comes
    from the extraction the NumPy lane was transcribed from
    (``scripts/validate/fv3_native/fv3_dswcore_extract.F90:1401``).  The
    NumPy lane is the specification for this hop, so the argument is
    mirrored as-is.

    Dependence (correction 1).  The ``do n=1,nord`` pass loop
    (fv3_native_d_sw.py:1150) is a RECURRENCE: pass ``n`` builds ``d2``
    from the ``fx2``/``fy2`` pass ``n-1`` wrote, then overwrites them.
    It is kept as a python ``for`` over a STATIC trip count -- an
    unrolled, ordered chain of array ops -- and is NOT vectorised and
    NOT a scan (the window ``nt = nord - n`` shrinks every pass, so the
    bodies have different shapes).  Inside a pass, each of the three
    nests writes one array and reads only the OTHER two, so each nest
    vectorises.  ``d2`` is NOT reallocated per pass, so every update is
    a partial ``.at[window].set`` that carries the cells outside the
    window forward, exactly like the in-place lane.
    """
    _nn = functools.partial(_nan, dtype=q.dtype)  # workspace follows storage dtype (fp32/fp64)
    require_uniform_float_jax("deln_flux", {
        "q": q, "fx": fx, "fy": fy, "del6_v": del6_v, "del6_u": del6_u,
        "rarea": rarea, "mass": mass, "damp_km": damp_km})
    q = jnp.asarray(q)
    fx = jnp.asarray(fx)
    fy = jnp.asarray(fy)
    if nord < 0:
        raise ValueError(
            f"deln_flux: nord={nord} < 0 (tp_core.F90:1219-1221 defines "
            f"0 = del-2, 1 = del-4, 2 = del-6 only)")

    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    n_di = ied - isd + 1
    n_dj = jed - jsd + 1

    def win(arr, ilo, jlo, ni, nj, a, b, cc, d, what):
        return arr[_rng(a, b, ilo, ni, what + " i"),
                   _rng(cc, d, jlo, nj, what + " j")]

    def d2_w(a, b, cc, d):
        return win(d2, isd, jsd, n_di, n_dj, a, b, cc, d, "deln d2")

    def del6v_w(a, b, cc, d):
        return win(del6_v, isd, jsd, n_di + 1, n_dj, a, b, cc, d,
                 "deln del6_v")

    def del6u_w(a, b, cc, d):
        return win(del6_u, isd, jsd, n_di, n_dj + 1, a, b, cc, d,
                 "deln del6_u")

    def _set(arr, ilo, jlo, ni, nj, a, b, cc, d, val, what):
        return arr.at[_rng(a, b, ilo, ni, what + " i"),
                      _rng(cc, d, jlo, nj, what + " j")].set(val)

    fx2 = _nn((n_di + 1, n_dj))          # fx2(isd:ied+1, jsd:jed)
    fy2 = _nn((n_di, n_dj + 1))          # fy2(isd:ied, jsd:jed+1)
    d2 = _nn((n_di, n_dj))               # d2(isd:ied, jsd:jed)

    i1 = is_ - 1 - nord                                     # :1247-1250
    i2 = ie + 1 + nord
    j1 = js - 1 - nord
    j2 = je + 1 + nord

    # :1252-1262 -- writes d2 only, reads q only.
    if (mass is None) and (damp_km is None):
        val = damp * win(q, isd, jsd, n_di, n_dj, i1, i2, j1, j2, "deln q")
    else:
        val = win(q, isd, jsd, n_di, n_dj, i1, i2, j1, j2, "deln q")
    d2 = _set(d2, isd, jsd, n_di, n_dj, i1, i2, j1, j2, val, "deln d2")

    if nord > 0:                                            # :1264-1266
        d2 = copy_corners(d2, npx, npy, 1, bounded_domain, bd, sw_corner,
                          se_corner, nw_corner, ne_corner,
                          duogrid=duogrid)

    # :1268-1271 -- writes fx2, reads del6_v/d2.
    fx2 = _set(fx2, isd, jsd, n_di + 1, n_dj,
               is_ - nord, ie + nord + 1, js - nord, je + nord,
               del6v_w(is_ - nord, ie + nord + 1, js - nord, je + nord)
               * (d2_w(is_ - nord - 1, ie + nord, js - nord, je + nord)
                  - d2_w(is_ - nord, ie + nord + 1, js - nord, je + nord)),
               "deln fx2")

    if nord > 0:                                            # :1273-1276
        d2 = copy_corners(d2, npx, npy, 2, bounded_domain, bd, sw_corner,
                          se_corner, nw_corner, ne_corner,
                          duogrid=duogrid)

    # :1277-1279 -- writes fy2, reads del6_u/d2.
    fy2 = _set(fy2, isd, jsd, n_di, n_dj + 1,
               is_ - nord, ie + nord, js - nord, je + nord + 1,
               del6u_w(is_ - nord, ie + nord, js - nord, je + nord + 1)
               * (d2_w(is_ - nord, ie + nord, js - nord - 1, je + nord)
                  - d2_w(is_ - nord, ie + nord, js - nord, je + nord + 1)),
               "deln fy2")

    if nord > 0:
        # ---- high-order: ORDERED recurrence over n (see docstring) ----
        for n in range(1, nord + 1):                        # :1287-1308
            nt = nord - n

            d2 = _set(
                d2, isd, jsd, n_di, n_dj,
                is_ - nt - 1, ie + nt + 1, js - nt - 1, je + nt + 1,
                (win(fx2, isd, jsd, n_di + 1, n_dj, is_ - nt - 1,
                   ie + nt + 1, js - nt - 1, je + nt + 1, "deln fx2")
                 - win(fx2, isd, jsd, n_di + 1, n_dj, is_ - nt,
                     ie + nt + 2, js - nt - 1, je + nt + 1, "deln fx2")
                 + win(fy2, isd, jsd, n_di, n_dj + 1, is_ - nt - 1,
                     ie + nt + 1, js - nt - 1, je + nt + 1, "deln fy2")
                 - win(fy2, isd, jsd, n_di, n_dj + 1, is_ - nt - 1,
                     ie + nt + 1, js - nt, je + nt + 2, "deln fy2"))
                * win(rarea, isd, jsd, n_di, n_dj, is_ - nt - 1, ie + nt + 1,
                    js - nt - 1, je + nt + 1, "deln rarea"),
                "deln d2")

            d2 = copy_corners(d2, npx, npy, 1, bounded_domain, bd,
                              sw_corner, se_corner, nw_corner, ne_corner,
                              duogrid=duogrid)
            # NOTE the SIGN flip vs the first pass: (d2(i) - d2(i-1)).
            fx2 = _set(fx2, isd, jsd, n_di + 1, n_dj,
                       is_ - nt, ie + nt + 1, js - nt, je + nt,
                       del6v_w(is_ - nt, ie + nt + 1, js - nt, je + nt)
                       * (d2_w(is_ - nt, ie + nt + 1, js - nt, je + nt)
                          - d2_w(is_ - nt - 1, ie + nt, js - nt, je + nt)),
                       "deln fx2")

            d2 = copy_corners(d2, npx, npy, 2, bounded_domain, bd,
                              sw_corner, se_corner, nw_corner, ne_corner,
                              duogrid=duogrid)
            fy2 = _set(fy2, isd, jsd, n_di, n_dj + 1,
                       is_ - nt, ie + nt, js - nt, je + nt + 1,
                       del6u_w(is_ - nt, ie + nt, js - nt, je + nt + 1)
                       * (d2_w(is_ - nt, ie + nt, js - nt, je + nt + 1)
                          - d2_w(is_ - nt, ie + nt, js - nt - 1, je + nt)),
                       "deln fy2")

    # ---- add the diffusive fluxes (:1310-1358) -------------------
    # Elementwise read-modify-write of fx(i,j)/fy(i,j) at the SAME index,
    # so the two nests vectorise.  fx spans is..ie+1 x js..je and fy
    # spans is..ie x js..je+1, i.e. their WHOLE declared extent.
    fx2_x = win(fx2, isd, jsd, n_di + 1, n_dj, is_, ie + 1, js, je,
              "deln fx2 add")
    fy2_y = win(fy2, isd, jsd, n_di, n_dj + 1, is_, ie, js, je + 1,
              "deln fy2 add")

    if damp_km is not None:
        km_xm = win(damp_km, isd, jsd, n_di, n_dj, is_ - 1, ie, js, je,
                  "deln damp_km")
        km_xc = win(damp_km, isd, jsd, n_di, n_dj, is_, ie + 1, js, je,
                  "deln damp_km")
        damp2_x = 0.25 * damp * (km_xm + km_xc)
        km_ym = win(damp_km, isd, jsd, n_di, n_dj, is_, ie, js - 1, je,
                  "deln damp_km")
        km_yc = win(damp_km, isd, jsd, n_di, n_dj, is_, ie, js, je + 1,
                  "deln damp_km")
        damp2_y = 0.25 * damp * (km_ym + km_yc)
    else:
        damp2_x = 0.5 * damp if mass is not None else None
        damp2_y = damp2_x

    if mass is not None:
        m_xm = win(mass, isd, jsd, n_di, n_dj, is_ - 1, ie, js, je,
                 "deln mass")
        m_xc = win(mass, isd, jsd, n_di, n_dj, is_, ie + 1, js, je,
                 "deln mass")
        m_ym = win(mass, isd, jsd, n_di, n_dj, is_, ie, js - 1, je,
                 "deln mass")
        m_yc = win(mass, isd, jsd, n_di, n_dj, is_, ie, js, je + 1,
                 "deln mass")
        fx = fx + damp2_x * (m_xm + m_xc) * fx2_x
        fy = fy + damp2_y * (m_ym + m_yc) * fy2_y
    elif damp_km is not None:
        fx = fx + damp2_x * fx2_x
        fy = fy + damp2_y * fy2_y
    else:
        fx = fx + fx2_x
        fy = fy + fy2_y
    return fx, fy


def fv_tp_2d(q, crx, cry, npx: int, npy: int, hord: int, xfx, yfx, dxa,
             dya, area, del6_v, del6_u, rarea, da_min: float, bd, ra_x,
             ra_y, lim_fac: float, bounded_domain: bool, grid_type: int,
             sw_corner: bool, se_corner: bool, nw_corner: bool,
             ne_corner: bool, mfx=None, mfy=None, mass=None,
             nord: int | None = None, damp_c: float | None = None,
             damp_smag: float | None = None, damp_km=None, *,
             duogrid: bool = False):
    """JAX twin of ``fv3_native_d_sw.fv_tp_2d`` (tp_core.F90:80-225).

    Functional: RETURNS ``(q, fx, fy)``.  ``q`` is returned because
    ``copy_corners`` mutates its corner ghosts TWICE (dir 2 at
    tp_core.F90:138-140, then dir 1 at :159-161 ON TOP of the first
    result) and the second ``xppm``, both ``q_j`` builds and every
    ``deln_flux`` read see the mutated array -- a caller that dropped it
    would silently transport a different field on the next call.
    ``fx``/``fy`` are INTENT(OUT) in the oracle, so they are created
    here rather than taken as operands.

    The ``gridstruct`` dict is split into explicit arrays + static flags
    for the reason given in :func:`deln_flux`.

    Declared Fortran bounds (tp_core.F90:86-94): ``crx``/``xfx``
    ``(is:ie+1, jsd:jed)``; ``cry``/``yfx`` ``(isd:ied, js:je+1)``;
    ``ra_x`` ``(is:ie, jsd:jed)``; ``ra_y`` ``(isd:ied, js:je)``;
    ``q``/``area``/``mass``/``damp_km`` ``(isd:ied, jsd:jed)``; ``mfx``
    ``(is:ie+1, js:je)``; ``mfy`` ``(is:ie, js:je+1)``; returned ``fx``
    ``(is:ie+1, js:je)``, ``fy`` ``(is:ie, js:je+1)``.

    Dependence (correction 1): the four transport calls and the two
    ``fyy``/``fx1`` nests each write one array and read only arrays no
    concurrent iteration writes (``fyy``/``q_i`` read ``q``/``fy2``;
    ``fx1``/``q_j`` read ``q``/``fx2``), so each nest vectorises.  The
    ORDER of the six stages plus the two ``copy_corners`` is preserved
    exactly: the second ``copy_corners`` reads what the first wrote.

    ``hord``/``nord``/``damp_c``/``damp_smag``/``da_min``/``lim_fac``
    and every index/flag are STATIC.  ``mfx``/``mfy``/``mass``/
    ``damp_km`` are ``None``-or-array; the ``is not None`` tests are
    python branches resolved at trace time (passing ``None`` vs an array
    changes the pytree, which correctly forces a retrace).
    """
    _validate_ord("fv_tp_2d", "hord", hord, _PPM_ORDS)
    require_uniform_float_jax("fv_tp_2d", {
        "q": q, "crx": crx, "cry": cry, "xfx": xfx, "yfx": yfx,
        "dxa": dxa, "dya": dya, "area": area, "del6_v": del6_v,
        "del6_u": del6_u, "rarea": rarea, "ra_x": ra_x, "ra_y": ra_y,
        "mfx": mfx, "mfy": mfy, "mass": mass, "damp_km": damp_km})
    q = jnp.asarray(q)

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    n_di = ied - isd + 1
    n_dj = jed - jsd + 1

    if hord == 10:                                          # :131-136
        ord_in = 8
    else:
        ord_in = hord
    ord_ou = hord

    if not bounded_domain:                                  # :138-140
        q = copy_corners(q, npx, npy, 2, bounded_domain, bd, sw_corner,
                         se_corner, nw_corner, ne_corner, duogrid=duogrid)

    # :142-143 -- yppm over the FULL i extent (ifirst=isd, ilast=ied).
    fy2 = yppm(q, cry, ord_in, isd, ied, isd, ied, js, je, jsd, jed,
               npx, npy, dya, bounded_domain, grid_type, lim_fac,
               duogrid=duogrid)

    # :145-155 -- fyy over (isd:ied, js:je+1) is the whole array; q_i
    # over (isd:ied, js:je).
    fyy = yfx * fy2
    j_cw = _rng(js, je, jsd, n_dj, "fv_tp_2d q j")
    q_i = ((q[:, j_cw] * area[:, j_cw] + fyy[:, :-1] - fyy[:, 1:])
           / ra_y)

    # :157-158 -- the oracle passes crx(is,js); the leading extent
    # matches, so the callee's c(i,j) is crx(i,j) over j = js..je.
    fx = xppm(q_i, crx[:, j_cw], ord_ou, is_, ie, isd, ied, js, je, jsd,
              jed, npx, npy, dxa, bounded_domain, grid_type, lim_fac,
              duogrid=duogrid)

    if not bounded_domain:                                  # :159-161
        q = copy_corners(q, npx, npy, 1, bounded_domain, bd, sw_corner,
                         se_corner, nw_corner, ne_corner, duogrid=duogrid)

    # :163-164 -- second xppm on the FULL j extent (jfirst=jsd).
    fx2 = xppm(q, crx, ord_in, is_, ie, isd, ied, jsd, jed, jsd, jed,
               npx, npy, dxa, bounded_domain, grid_type, lim_fac,
               duogrid=duogrid)

    # :166-174 -- fx1 spans is..ie+1 (all of fx2's i extent); q_j spans
    # is..ie.  fx1 is fully written before q_j reads it (two nests).
    fx1 = xfx * fx2
    i_cw = _rng(is_, ie, isd, n_di, "fv_tp_2d q i")
    q_j = ((q[i_cw, :] * area[i_cw, :] + fx1[:-1, :] - fx1[1:, :])
           / ra_x)

    # :176-177
    fy = yppm(q_j, cry, ord_ou, is_, ie, isd, ied, js, je, jsd, jed,
              npx, npy, dya, bounded_domain, grid_type, lim_fac,
              duogrid=duogrid)

    # ---------------- flux averaging (:179-224) -------------------
    fx2_x = fx2[:, j_cw]                       # fx2 over j = js..je
    fy2_y = fy2[i_cw, :]                       # fy2 over i = is..ie

    if (mfx is not None) and (mfy is not None):             # :183-203
        fx = 0.5 * (fx + fx2_x) * mfx
        fy = 0.5 * (fy + fy2_y) * mfy
        if (nord is not None) and (damp_c is not None) \
                and (mass is not None):
            if damp_c > 1.0e-4:
                damp = (damp_c * da_min) ** (nord + 1)
                fx, fy = deln_flux(
                    nord, is_, ie, js, je, npx, npy, damp, q, fx, fy,
                    del6_v, del6_u, rarea, bd, bounded_domain, sw_corner,
                    se_corner, nw_corner, ne_corner, mass=mass,
                    duogrid=duogrid)
        if (damp_smag is not None) and (damp_km is not None) \
                and (mass is not None):
            if damp_smag > 1.0e-3:
                damp = damp_smag * da_min          # 2nd order
                fx, fy = deln_flux(
                    0, is_, ie, js, je, npx, npy, damp, q, fx, fy,
                    del6_v, del6_u, rarea, bd, bounded_domain, sw_corner,
                    se_corner, nw_corner, ne_corner, mass=mass,
                    damp_km=damp_km, duogrid=duogrid)
    else:                                                   # :205-224
        fx = 0.5 * (fx + fx2_x) * xfx[:, j_cw]
        fy = 0.5 * (fy + fy2_y) * yfx[i_cw, :]
        if (nord is not None) and (damp_c is not None):
            if damp_c > 1.0e-4:
                damp = (damp_c * da_min) ** (nord + 1)
                fx, fy = deln_flux(
                    nord, is_, ie, js, je, npx, npy, damp, q, fx, fy,
                    del6_v, del6_u, rarea, bd, bounded_domain, sw_corner,
                    se_corner, nw_corner, ne_corner, duogrid=duogrid)
        if (damp_smag is not None) and (damp_km is not None):
            if damp_smag > 1.0e-3:
                damp = damp_smag * da_min          # 2nd order
                fx, fy = deln_flux(
                    0, is_, ie, js, je, npx, npy, damp, q, fx, fy,
                    del6_v, del6_u, rarea, bd, bounded_domain, sw_corner,
                    se_corner, nw_corner, ne_corner, damp_km=damp_km,
                    duogrid=duogrid)
    return q, fx, fy


# =====================================================================
# sw_core_mod (staggered-field transport used by d_sw)
# =====================================================================

def xtp_u(is_: int, ie: int, js: int, je: int, isd: int, ied: int,
          jsd: int, jed: int, c, u, v, iord: int, dx, rdx, npx: int,
          npy: int, grid_type: int, bounded_domain: bool,
          lim_fac: float, *, duogrid: bool = False):
    """JAX twin of ``fv3_native_d_sw.xtp_u`` (sw_core.F90:2540-2894).

    Functional: RETURNS ``flux``.  Declared Fortran bounds:
    ``c``/``flux`` ``(is:ie+1, js:je+1)``; ``u``/``dx``/``rdx``
    ``(isd:ied, jsd:jed+1)``; ``v`` ``(isd:ied+1, jsd:jed)``.  ``v`` is
    a DEAD read kept for signature parity with the oracle and the NumPy
    lane (``xtp_u`` transports ``u`` only).

    Dependence (correction 1): the oracle's outer ``do j=js,je+1`` writes
    only ``flux(i,j)`` plus per-j 1-D locals and reads only ``c``, ``u``,
    ``dx``, ``rdx`` -- none of which it writes -- so it is vectorised
    onto a trailing j axis.  The ``if (j==1 .or. j==npy)`` edge test
    (sw_core.F90:2585 / :2612 / :2843 / :2866) is a test on the LOOP
    INDEX, i.e. STATIC: it becomes a static boolean mask over the j
    window and both arms are total and finite, so a plain ``jnp.where``
    is correct there.  Each inner ``do i`` writes one local and reads
    only arrays it does not write in the same loop.

    ``iord`` is validated against the STATIC set the NumPy lane branches
    on: 1..7 in the ``iord < 8`` arm (5 explicit, 6 and 7 in its else)
    and 8..11 in the other (11 = the "unlimited" else).  Unlike
    ``xppm``, this routine branches on the RAW ``iord`` with no
    ``abs()``, so a negative value is NOT a scheme here -- it would fall
    into the {5,6,7} else and silently run a different limiter.

    KNOWN NUMPY-LANE / ORACLE DIFFERENCE, mirrored not fixed (hop A is
    out of scope): the oracle's ``iord < 8`` edge guard is
    ``(.not.bounded_domain .or. gridstruct%dg%is_initialized) .and.
    grid_type < 3`` (sw_core.F90:2586) while the NumPy lane wrote
    ``(not bounded_domain) and grid_type < 3``
    (fv3_native_d_sw.py:2014).  The two agree except at
    ``bounded_domain=True, duogrid=True``, a combination ``d_sw``
    rejects.
    """
    _nn = functools.partial(_nan, dtype=u.dtype)  # workspace follows storage dtype (fp32/fp64)
    _validate_ord("xtp_u", "iord", iord, _SW_ORDS)
    require_uniform_float_jax("xtp_u", {"c": c, "u": u, "dx": dx, "rdx": rdx})
    c = jnp.asarray(c)
    u = jnp.asarray(u)
    del v                                # dead read (signature parity)

    n_di = ied - isd + 1
    n_uj = jed + 1 - jsd + 1
    nj = je + 1 - js + 1
    n_c = ie + 1 - is_ + 1
    if c.shape != (n_c, nj):
        raise ValueError(f"xtp_u: c must be {(n_c, nj)}, got {c.shape}")
    if u.shape != (n_di, n_uj):
        raise ValueError(f"xtp_u: u must be {(n_di, n_uj)}, got {u.shape}")

    # sw_core.F90:2566 -- duo is UNCLAMPED (the duo halos are real).
    if bounded_domain or grid_type > 3 or duogrid:
        is3 = is_ - 1
        ie3 = ie + 1
    else:
        is3 = max(3, is_ - 1)
        ie3 = min(npx - 3, ie + 1)

    j_w = _rng(js, je + 1, jsd, n_uj, "xtp_u j window")
    u_j = u[:, j_w]                       # u(isd:ied, js:je+1)
    dx_j = jnp.asarray(dx)[:, j_w]
    rdx_j = jnp.asarray(rdx)[:, j_w]
    # STATIC j mask for the `j==1 .or. j==npy` panel-corner rows.
    j_edge = jnp.asarray(
        np.array([(j == 1 or j == npy) for j in range(js, je + 2)]))

    def u_w(a, b):
        return u_j[_rng(a, b, isd, n_di, "xtp_u u i"), :]

    def u_at(i):
        return u_j[_idx(i, isd, n_di, "xtp_u u i"), :]

    def dx_at(i):
        return dx_j[_idx(i, isd, n_di, "xtp_u dx i"), :]

    def rdx_w(a, b):
        return rdx_j[_rng(a, b, isd, n_di, "xtp_u rdx i"), :]

    o_al = is_ - 1
    n_al = ie + 2 - (is_ - 1) + 1
    o_b = is_ - 1
    n_b = ie + 1 - (is_ - 1) + 1

    def al_w(a, b):
        return al[_rng(a, b, o_al, n_al, "xtp_u al"), :]

    def b_w(arr, a, b):
        return arr[_rng(a, b, o_b, n_b, "xtp_u bl/br/b0"), :]

    def _zero_or(arr, i, val):
        """`if (j==1 .or. j==npy) X = 0. else X = val` -- static mask,
        both arms total."""
        p = _idx(i, o_b, n_b, "xtp_u bl/br")
        return arr.at[p, :].set(jnp.where(j_edge, 0.0, val))

    cw = c
    cpos = cw > 0.0
    cfl = jnp.where(cpos, cw * rdx_w(is_ - 1, ie), cw * rdx_w(is_, ie + 1))

    if iord < 8:
        # ---------------------------------------------------- :2573-2810
        al = _nn((n_al, nj)).at[
            _rng(is3, ie3 + 1, o_al, n_al, "xtp_u al"), :].set(
            SW_P1 * (u_w(is3 - 1, ie3) + u_w(is3, ie3 + 1))
            + SW_P2 * (u_w(is3 - 2, ie3 - 1) + u_w(is3 + 1, ie3 + 2)))
        bl = _nn((n_b, nj))
        br = _nn((n_b, nj))
        w3 = _rng(is3, ie3, o_b, n_b, "xtp_u bl/br")
        bl = bl.at[w3, :].set(al_w(is3, ie3) - u_w(is3, ie3))
        br = br.at[w3, :].set(al_w(is3 + 1, ie3 + 1) - u_w(is3, ie3))

        if (not bounded_domain) and grid_type < 3:          # :2586-2626
            if is_ == 1:
                xt = SW_C3 * u_at(1) + SW_C2 * u_at(2) + SW_C1 * u_at(3)
                br = br.at[_idx(1, o_b, n_b, "xtp_u br"), :].set(
                    xt - u_at(1))
                bl = bl.at[_idx(2, o_b, n_b, "xtp_u bl"), :].set(
                    xt - u_at(2))
                br = br.at[_idx(2, o_b, n_b, "xtp_u br"), :].set(
                    al_w(3, 3)[0] - u_at(2))
                br1_prior = br[_idx(1, o_b, n_b, "xtp_u br"), :]
                bl0 = (SW_C1 * u_at(-2) + SW_C2 * u_at(-1) + SW_C3 * u_at(0)
                       - u_at(0))
                xt = 0.5 * (((2.0 * dx_at(0) + dx_at(-1)) * u_at(0)
                             - dx_at(0) * u_at(-1)) / (dx_at(0) + dx_at(-1))
                            + ((2.0 * dx_at(1) + dx_at(2)) * u_at(1)
                               - dx_at(1) * u_at(2)) / (dx_at(1) + dx_at(2)))
                bl = _zero_or(bl, 0, bl0)
                br = _zero_or(br, 0, xt - u_at(0))
                bl = _zero_or(bl, 1, xt - u_at(1))
                br = _zero_or(br, 1, br1_prior)
            if (ie + 1) == npx:
                bl = bl.at[_idx(npx - 2, o_b, n_b, "xtp_u bl"), :].set(
                    al_w(npx - 2, npx - 2)[0] - u_at(npx - 2))
                xt = (SW_C1 * u_at(npx - 3) + SW_C2 * u_at(npx - 2)
                      + SW_C3 * u_at(npx - 1))
                br = br.at[_idx(npx - 2, o_b, n_b, "xtp_u br"), :].set(
                    xt - u_at(npx - 2))
                blm1_prior = xt - u_at(npx - 1)
                xt = 0.5 * (((2.0 * dx_at(npx - 1) + dx_at(npx - 2))
                             * u_at(npx - 1) - dx_at(npx - 1) * u_at(npx - 2))
                            / (dx_at(npx - 1) + dx_at(npx - 2))
                            + ((2.0 * dx_at(npx) + dx_at(npx + 1)) * u_at(npx)
                               - dx_at(npx) * u_at(npx + 1))
                            / (dx_at(npx) + dx_at(npx + 1)))
                bl = _zero_or(bl, npx - 1, blm1_prior)
                br = _zero_or(br, npx - 1, xt - u_at(npx - 1))
                bl = _zero_or(bl, npx, xt - u_at(npx))
                br = _zero_or(br, npx, SW_C3 * u_at(npx)
                              + SW_C2 * u_at(npx + 1)
                              + SW_C1 * u_at(npx + 2) - u_at(npx))

        b0 = bl + br                                        # :2628-2630

        fx0_p = (1.0 - cfl) * (b_w(br, is_ - 1, ie)
                               - cfl * b_w(b0, is_ - 1, ie))
        fx0_m = (1.0 + cfl) * (b_w(bl, is_, ie + 1)
                               + cfl * b_w(b0, is_, ie + 1))
        base = jnp.where(cpos, u_w(is_ - 1, ie), u_w(is_, ie + 1))

        if iord == 1:                                       # :2633-2647
            smt5 = jnp.abs(lim_fac * b0) < jnp.abs(bl - br)
            fx0 = jnp.where(cpos, fx0_p, fx0_m)
            add = b_w(smt5, is_ - 1, ie) | b_w(smt5, is_, ie + 1)
            return jnp.where(add, base + fx0, base)

        if iord == 2:                                       # :2649-2658
            return base + jnp.where(cpos, fx0_p, fx0_m)

        if iord == 3:                                       # :2660-2690
            x0 = jnp.abs(b0)
            x1 = jnp.abs(bl - br)
            smt5 = x0 < x1
            smt6 = 3.0 * x0 < x1
            hi5 = b_w(smt5, is_ - 1, ie) & b_w(smt5, is_, ie + 1)
            hi6 = b_w(smt6, is_ - 1, ie) | b_w(smt6, is_, ie + 1)
            # fx0 initialised to 0. (:2671) and only overwritten by the
            # hi6 / hi5 arms -- three-way select, all arms total.
            up_hi6 = b_w(br, is_ - 1, ie) - cfl * b_w(b0, is_ - 1, ie)
            up_hi5 = jnp.copysign(
                jnp.minimum(jnp.abs(b_w(bl, is_ - 1, ie)),
                            jnp.abs(b_w(br, is_ - 1, ie))),
                b_w(br, is_ - 1, ie))
            dn_hi6 = b_w(bl, is_, ie + 1) + cfl * b_w(b0, is_, ie + 1)
            dn_hi5 = jnp.copysign(
                jnp.minimum(jnp.abs(b_w(bl, is_, ie + 1)),
                            jnp.abs(b_w(br, is_, ie + 1))),
                b_w(bl, is_, ie + 1))
            fx0 = jnp.where(
                cpos,
                jnp.where(hi6, up_hi6, jnp.where(hi5, up_hi5, 0.0)),
                jnp.where(hi6, dn_hi6, jnp.where(hi5, dn_hi5, 0.0)))
            return jnp.where(cpos,
                             u_w(is_ - 1, ie) + (1.0 - cfl) * fx0,
                             u_w(is_, ie + 1) + (1.0 + cfl) * fx0)

        if iord == 4:                                       # :2692-2716
            x0 = jnp.abs(b0)
            x1 = jnp.abs(bl - br)
            smt5 = x0 < x1
            smt6 = 3.0 * x0 < x1
            hi5 = b_w(smt5, is_ - 1, ie) & b_w(smt5, is_, ie + 1)
            hi6 = b_w(smt6, is_ - 1, ie) | b_w(smt6, is_, ie + 1)
            hi5 = hi5 | hi6
            fx0 = jnp.where(cpos, fx0_p, fx0_m)
            return jnp.where(hi5, base + fx0, base)

        # -------------------------------------------- iord 5, 6, 7
        if iord == 5:                                       # :2719-2723
            smt5 = bl * br < 0.0
        else:                                               # :2725-2738
            smt5 = 3.0 * jnp.abs(b0) < jnp.abs(bl - br)
            # WMP edge fix -- ABSENT from the symmetryclean tree; the
            # NumPy lane keeps it on the PLAIN path only (duogrid=False),
            # and this twin mirrors that gate verbatim.
            if (not (bounded_domain or duogrid)) and grid_type < 3:
                if is_ == 1:
                    for k in (0, 1):
                        p = _idx(k, o_b, n_b, "xtp_u smt5")
                        smt5 = smt5.at[p, :].set(bl[p, :] * br[p, :] < 0.0)
                if (ie + 1) == npx:
                    for k in (npx - 1, npx):
                        p = _idx(k, o_b, n_b, "xtp_u smt5")
                        smt5 = smt5.at[p, :].set(bl[p, :] * br[p, :] < 0.0)
        fx0 = jnp.where(cpos, fx0_p, fx0_m)
        add = b_w(smt5, is_ - 1, ie) | b_w(smt5, is_, ie + 1)
        return jnp.where(add, base + fx0, base)

    # ---------------------------------------------------- :2740-2892
    o_dm = is_ - 2
    n_dm = ie + 2 - (is_ - 2) + 1
    o_dq = is_ - 3
    n_dq = ie + 2 - (is_ - 3) + 1

    def dm_w(a, b):
        return dm[_rng(a, b, o_dm, n_dm, "xtp_u dm"), :]

    def dq_w(a, b):
        return dq[_rng(a, b, o_dq, n_dq, "xtp_u dq"), :]

    um = u_w(is_ - 3, ie + 1)
    uc = u_w(is_ - 2, ie + 2)
    up = u_w(is_ - 1, ie + 3)
    xt = 0.25 * (up - um)
    dm = _nn((n_dm, nj)).at[
        _rng(is_ - 2, ie + 2, o_dm, n_dm, "xtp_u dm"), :].set(
        jnp.copysign(
            jnp.minimum(
                jnp.minimum(jnp.abs(xt),
                            jnp.maximum(jnp.maximum(um, uc), up) - uc),
                uc - jnp.minimum(jnp.minimum(um, uc), up)),
            xt))
    dq = _nn((n_dq, nj)).at[
        _rng(is_ - 3, ie + 2, o_dq, n_dq, "xtp_u dq"), :].set(
        u_w(is_ - 2, ie + 3) - u_w(is_ - 3, ie + 2))

    al = _nn((n_al, nj))
    bl = _nn((n_b, nj))
    br = _nn((n_b, nj))

    if grid_type < 3:
        w_al = _rng(is3, ie3 + 1, o_al, n_al, "xtp_u al")
        al = al.at[w_al, :].set(
            0.5 * (u_w(is3 - 1, ie3) + u_w(is3, ie3 + 1))
            + R3 * (dm_w(is3 - 1, ie3) - dm_w(is3, ie3 + 1)))
        w3 = _rng(is3, ie3, o_b, n_b, "xtp_u bl/br")

        if iord == 8:                                       # :2769-2775
            xt = 2.0 * dm_w(is3, ie3)
            bl = bl.at[w3, :].set(-jnp.copysign(
                jnp.minimum(jnp.abs(xt),
                            jnp.abs(al_w(is3, ie3) - u_w(is3, ie3))), xt))
            br = br.at[w3, :].set(jnp.copysign(
                jnp.minimum(jnp.abs(xt),
                            jnp.abs(al_w(is3 + 1, ie3 + 1) - u_w(is3, ie3))),
                xt))
        elif iord == 9:                                     # :2777-2787
            pmp_1 = -2.0 * dq_w(is3, ie3)
            lac_1 = pmp_1 + 1.5 * dq_w(is3 + 1, ie3 + 1)
            bl = bl.at[w3, :].set(jnp.minimum(
                jnp.maximum(jnp.maximum(0.0, pmp_1), lac_1),
                jnp.maximum(al_w(is3, ie3) - u_w(is3, ie3),
                            jnp.minimum(jnp.minimum(0.0, pmp_1), lac_1))))
            pmp_2 = 2.0 * dq_w(is3 - 1, ie3 - 1)
            lac_2 = pmp_2 - 1.5 * dq_w(is3 - 2, ie3 - 2)
            br = br.at[w3, :].set(jnp.minimum(
                jnp.maximum(jnp.maximum(0.0, pmp_2), lac_2),
                jnp.maximum(al_w(is3 + 1, ie3 + 1) - u_w(is3, ie3),
                            jnp.minimum(jnp.minimum(0.0, pmp_2), lac_2))))
        elif iord == 10:                                    # :2789-2810
            bl_w = al_w(is3, ie3) - u_w(is3, ie3)
            br_w = al_w(is3 + 1, ie3 + 1) - u_w(is3, ie3)
            flat = ((jnp.abs(dm_w(is3, ie3)) < NEAR_ZERO_SW)
                    & ((jnp.abs(dm_w(is3 - 1, ie3 - 1))
                        + jnp.abs(dm_w(is3 + 1, ie3 + 1))) < NEAR_ZERO_SW))
            steep = ((jnp.abs(dm_w(is3, ie3)) >= NEAR_ZERO_SW)
                     & (jnp.abs(3.0 * (bl_w + br_w))
                        > jnp.abs(bl_w - br_w)))
            pmp_1 = -2.0 * dq_w(is3, ie3)
            lac_1 = pmp_1 + 1.5 * dq_w(is3 + 1, ie3 + 1)
            bl_c = jnp.minimum(
                jnp.maximum(jnp.maximum(0.0, pmp_1), lac_1),
                jnp.maximum(bl_w,
                            jnp.minimum(jnp.minimum(0.0, pmp_1), lac_1)))
            pmp_2 = 2.0 * dq_w(is3 - 1, ie3 - 1)
            lac_2 = pmp_2 - 1.5 * dq_w(is3 - 2, ie3 - 2)
            br_c = jnp.minimum(
                jnp.maximum(jnp.maximum(0.0, pmp_2), lac_2),
                jnp.maximum(br_w,
                            jnp.minimum(jnp.minimum(0.0, pmp_2), lac_2)))
            bl = bl.at[w3, :].set(
                jnp.where(flat, 0.0, jnp.where(steep, bl_c, bl_w)))
            br = br.at[w3, :].set(
                jnp.where(flat, 0.0, jnp.where(steep, br_c, br_w)))
        else:                                               # 11 :2812-2816
            bl = bl.at[w3, :].set(al_w(is3, ie3) - u_w(is3, ie3))
            br = br.at[w3, :].set(al_w(is3 + 1, ie3 + 1) - u_w(is3, ie3))

        if is_ == 1 and not bounded_domain:                 # :2818-2840
            br = br.at[_idx(2, o_b, n_b, "xtp_u br"), :].set(
                al_w(3, 3)[0] - u_at(2))
            xt = SW_S15 * u_at(1) + SW_S11 * u_at(2) - SW_S14 * dm_w(2, 2)[0]
            bl = bl.at[_idx(2, o_b, n_b, "xtp_u bl"), :].set(xt - u_at(2))
            br = br.at[_idx(1, o_b, n_b, "xtp_u br"), :].set(xt - u_at(1))
            br1_prior = br[_idx(1, o_b, n_b, "xtp_u br"), :]
            bl0 = SW_S14 * dm_w(-1, -1)[0] - SW_S11 * dq_w(-1, -1)[0]
            x0l = (0.5 * ((2.0 * dx_at(0) + dx_at(-1)) * u_at(0)
                          - dx_at(0) * u_at(-1)) / (dx_at(0) + dx_at(-1)))
            x0r = (0.5 * ((2.0 * dx_at(1) + dx_at(2)) * u_at(1)
                          - dx_at(1) * u_at(2)) / (dx_at(1) + dx_at(2)))
            xt = x0l + x0r
            bl = _zero_or(bl, 0, bl0)
            br = _zero_or(br, 0, xt - u_at(0))
            bl = _zero_or(bl, 1, xt - u_at(1))
            br = _zero_or(br, 1, br1_prior)
            # pert_ppm(1, u(2,j), bl(2), br(2), -1) -- sw_core.F90:2839,
            # UNCONDITIONAL (outside the j==1/npy arm).
            p2 = _idx(2, o_b, n_b, "xtp_u bl/br")
            bl_p, br_p = pert_ppm(u_at(2), bl[p2, :], br[p2, :], -1)
            bl = bl.at[p2, :].set(bl_p)
            br = br.at[p2, :].set(br_p)

        if (ie + 1) == npx and not bounded_domain:          # :2842-2863
            bl = bl.at[_idx(npx - 2, o_b, n_b, "xtp_u bl"), :].set(
                al_w(npx - 2, npx - 2)[0] - u_at(npx - 2))
            xt = (SW_S15 * u_at(npx - 1) + SW_S11 * u_at(npx - 2)
                  + SW_S14 * dm_w(npx - 2, npx - 2)[0])
            br = br.at[_idx(npx - 2, o_b, n_b, "xtp_u br"), :].set(
                xt - u_at(npx - 2))
            blm1_prior = xt - u_at(npx - 1)
            brnpx = SW_S11 * dq_w(npx, npx)[0] - SW_S14 * dm_w(npx + 1,
                                                           npx + 1)[0]
            x0l = (0.5 * ((2.0 * dx_at(npx - 1) + dx_at(npx - 2)) * u_at(npx - 1)
                          - dx_at(npx - 1) * u_at(npx - 2))
                   / (dx_at(npx - 1) + dx_at(npx - 2)))
            x0r = (0.5 * ((2.0 * dx_at(npx) + dx_at(npx + 1)) * u_at(npx)
                          - dx_at(npx) * u_at(npx + 1))
                   / (dx_at(npx) + dx_at(npx + 1)))
            xt = x0l + x0r
            bl = _zero_or(bl, npx - 1, blm1_prior)
            br = _zero_or(br, npx - 1, xt - u_at(npx - 1))
            bl = _zero_or(bl, npx, xt - u_at(npx))
            br = _zero_or(br, npx, brnpx)
            pm2 = _idx(npx - 2, o_b, n_b, "xtp_u bl/br")
            bl_p, br_p = pert_ppm(u_at(npx - 2), bl[pm2, :], br[pm2, :], -1)
            bl = bl.at[pm2, :].set(bl_p)
            br = br.at[pm2, :].set(br_p)
    else:                                                   # :2866-2880
        w_al = _rng(is_ - 1, ie + 2, o_al, n_al, "xtp_u al")
        al = al.at[w_al, :].set(
            0.5 * (u_w(is_ - 2, ie + 1) + u_w(is_ - 1, ie + 2))
            + R3 * (dm_w(is_ - 2, ie + 1) - dm_w(is_ - 1, ie + 2)))
        wf = _rng(is_ - 1, ie + 1, o_b, n_b, "xtp_u bl/br")
        pmp = -2.0 * dq_w(is_ - 1, ie + 1)
        lac = pmp + 1.5 * dq_w(is_, ie + 2)
        bl = bl.at[wf, :].set(jnp.minimum(
            jnp.maximum(jnp.maximum(0.0, pmp), lac),
            jnp.maximum(al_w(is_ - 1, ie + 1) - u_w(is_ - 1, ie + 1),
                        jnp.minimum(jnp.minimum(0.0, pmp), lac))))
        pmp = 2.0 * dq_w(is_ - 2, ie)
        lac = pmp - 1.5 * dq_w(is_ - 3, ie - 1)
        br = br.at[wf, :].set(jnp.minimum(
            jnp.maximum(jnp.maximum(0.0, pmp), lac),
            jnp.maximum(al_w(is_, ie + 2) - u_w(is_ - 1, ie + 1),
                        jnp.minimum(jnp.minimum(0.0, pmp), lac))))

    # :2882-2891 -- b0 is NOT formed; (bl+br) is recomputed inline.
    return jnp.where(
        cpos,
        u_w(is_ - 1, ie) + (1.0 - cfl) * (
            b_w(br, is_ - 1, ie)
            - cfl * (b_w(bl, is_ - 1, ie) + b_w(br, is_ - 1, ie))),
        u_w(is_, ie + 1) + (1.0 + cfl) * (
            b_w(bl, is_, ie + 1)
            + cfl * (b_w(bl, is_, ie + 1) + b_w(br, is_, ie + 1))))


def ytp_v(is_: int, ie: int, js: int, je: int, isd: int, ied: int,
          jsd: int, jed: int, c, u, v, jord: int, dy, rdy, npx: int,
          npy: int, grid_type: int, bounded_domain: bool,
          lim_fac: float, *, duogrid: bool = False):
    """JAX twin of ``fv3_native_d_sw.ytp_v`` (sw_core.F90:2897-3353).

    Functional: RETURNS ``flux``.  Declared Fortran bounds:
    ``c``/``flux`` ``(is:ie+1, js:je+1)``; ``v``/``dy``/``rdy``
    ``(isd:ied+1, jsd:jed)``; ``u`` ``(isd:ied, jsd:jed+1)``.  ``u`` is
    a DEAD read kept for signature parity (``ytp_v`` transports ``v``).

    Dependence (correction 1): like ``yppm`` this routine has no outer
    loop -- it is a sequence of full 2-D ``do j: do i`` nests over 2-D
    locals.  Each nest writes ONE local and reads only operands or
    locals an EARLIER nest completed, so each vectorises and the order
    of the nests is preserved.  The per-i 1-D scratch (``fx0``/``hi5``/
    ``hi6``) is written before use inside every j iteration, so
    promoting it to a 2-D block is exact.  The ``is==1`` / ``ie+1==npx``
    corner zeroings write single STATIC ``(i, j)`` points AFTER the
    block that set them, and are applied in that order.

    ``jord`` is validated against the same static set as :func:`xtp_u`
    (1..11, no ``abs()``, so negatives are rejected).
    """
    _nn = functools.partial(_nan, dtype=v.dtype)  # workspace follows storage dtype (fp32/fp64)
    _validate_ord("ytp_v", "jord", jord, _SW_ORDS)
    require_uniform_float_jax("ytp_v", {"c": c, "v": v, "dy": dy, "rdy": rdy})
    c = jnp.asarray(c)
    v = jnp.asarray(v)
    del u                                # dead read (signature parity)

    n_vi = ied + 1 - isd + 1
    n_dj = jed - jsd + 1
    n_c = ie + 1 - is_ + 1
    nj = je + 1 - js + 1
    if c.shape != (n_c, nj):
        raise ValueError(f"ytp_v: c must be {(n_c, nj)}, got {c.shape}")
    if v.shape != (n_vi, n_dj):
        raise ValueError(f"ytp_v: v must be {(n_vi, n_dj)}, got {v.shape}")

    # sw_core.F90:2923 -- duo is UNCLAMPED.
    if bounded_domain or grid_type > 3 or duogrid:
        js3 = js - 1
        je3 = je + 1
    else:
        js3 = max(3, js - 1)
        je3 = min(npy - 3, je + 1)

    i_w = _rng(is_, ie + 1, isd, n_vi, "ytp_v i window")
    v_i = v[i_w, :]                       # v(is:ie+1, jsd:jed)
    rdy_i = jnp.asarray(rdy)[i_w, :]
    dy_i = jnp.asarray(dy)[i_w, :]

    def v_w(a, b):
        return v_i[:, _rng(a, b, jsd, n_dj, "ytp_v v j")]

    def v_at(j):
        return v_i[:, _idx(j, jsd, n_dj, "ytp_v v j")]

    def dy_at(j):
        return dy_i[:, _idx(j, jsd, n_dj, "ytp_v dy j")]

    def rdy_w(a, b):
        return rdy_i[:, _rng(a, b, jsd, n_dj, "ytp_v rdy j")]

    o_al = js - 1
    n_al = je + 2 - (js - 1) + 1
    o_b = js - 1
    n_b = je + 1 - (js - 1) + 1

    def al_w(a, b):
        return al[:, _rng(a, b, o_al, n_al, "ytp_v al")]

    def b_w(arr, a, b):
        return arr[:, _rng(a, b, o_b, n_b, "ytp_v bl/br/b0")]

    def _pt(arr, i, j, val):
        """Single STATIC (i, j) point write (the corner zeroings)."""
        return arr.at[_idx(i, is_, n_c, "ytp_v i"),
                      _idx(j, o_b, n_b, "ytp_v j")].set(val)

    cw = c
    cpos = cw > 0.0
    cfl = jnp.where(cpos, cw * rdy_w(js - 1, je), cw * rdy_w(js, je + 1))

    al = _nn((n_c, n_al))
    bl = _nn((n_c, n_b))
    br = _nn((n_c, n_b))

    if jord < 8:
        # ---------------------------------------------------- :2930-3193
        al = al.at[:, _rng(js3, je3 + 1, o_al, n_al, "ytp_v al")].set(
            SW_P1 * (v_w(js3 - 1, je3) + v_w(js3, je3 + 1))
            + SW_P2 * (v_w(js3 - 2, je3 - 1) + v_w(js3 + 1, je3 + 2)))
        w3 = _rng(js3, je3, o_b, n_b, "ytp_v bl/br")
        bl = bl.at[:, w3].set(al_w(js3, je3) - v_w(js3, je3))
        br = br.at[:, w3].set(al_w(js3 + 1, je3 + 1) - v_w(js3, je3))

        if (not bounded_domain) and grid_type < 3:          # :2947-3000
            if js == 1:
                bl = bl.at[:, _idx(0, o_b, n_b, "ytp_v bl")].set(
                    SW_C1 * v_at(-2) + SW_C2 * v_at(-1) + SW_C3 * v_at(0)
                    - v_at(0))
                xt = 0.5 * (((2.0 * dy_at(0) + dy_at(-1)) * v_at(0)
                             - dy_at(0) * v_at(-1)) / (dy_at(0) + dy_at(-1))
                            + ((2.0 * dy_at(1) + dy_at(2)) * v_at(1)
                               - dy_at(1) * v_at(2)) / (dy_at(1) + dy_at(2)))
                br = br.at[:, _idx(0, o_b, n_b, "ytp_v br")].set(
                    xt - v_at(0))
                bl = bl.at[:, _idx(1, o_b, n_b, "ytp_v bl")].set(
                    xt - v_at(1))
                xt = SW_C3 * v_at(1) + SW_C2 * v_at(2) + SW_C1 * v_at(3)
                br = br.at[:, _idx(1, o_b, n_b, "ytp_v br")].set(
                    xt - v_at(1))
                bl = bl.at[:, _idx(2, o_b, n_b, "ytp_v bl")].set(
                    xt - v_at(2))
                br = br.at[:, _idx(2, o_b, n_b, "ytp_v br")].set(
                    al_w(3, 3)[:, 0] - v_at(2))
                if is_ == 1:
                    for jj in (0, 1):
                        bl = _pt(bl, 1, jj, 0.0)
                        br = _pt(br, 1, jj, 0.0)
                if (ie + 1) == npx:
                    for jj in (0, 1):
                        bl = _pt(bl, npx, jj, 0.0)
                        br = _pt(br, npx, jj, 0.0)
            if (je + 1) == npy:
                bl = bl.at[:, _idx(npy - 2, o_b, n_b, "ytp_v bl")].set(
                    al_w(npy - 2, npy - 2)[:, 0] - v_at(npy - 2))
                xt = (SW_C1 * v_at(npy - 3) + SW_C2 * v_at(npy - 2)
                      + SW_C3 * v_at(npy - 1))
                br = br.at[:, _idx(npy - 2, o_b, n_b, "ytp_v br")].set(
                    xt - v_at(npy - 2))
                bl = bl.at[:, _idx(npy - 1, o_b, n_b, "ytp_v bl")].set(
                    xt - v_at(npy - 1))
                xt = 0.5 * (((2.0 * dy_at(npy - 1) + dy_at(npy - 2))
                             * v_at(npy - 1) - dy_at(npy - 1) * v_at(npy - 2))
                            / (dy_at(npy - 1) + dy_at(npy - 2))
                            + ((2.0 * dy_at(npy) + dy_at(npy + 1)) * v_at(npy)
                               - dy_at(npy) * v_at(npy + 1))
                            / (dy_at(npy) + dy_at(npy + 1)))
                br = br.at[:, _idx(npy - 1, o_b, n_b, "ytp_v br")].set(
                    xt - v_at(npy - 1))
                bl = bl.at[:, _idx(npy, o_b, n_b, "ytp_v bl")].set(
                    xt - v_at(npy))
                br = br.at[:, _idx(npy, o_b, n_b, "ytp_v br")].set(
                    SW_C3 * v_at(npy) + SW_C2 * v_at(npy + 1)
                    + SW_C1 * v_at(npy + 2) - v_at(npy))
                if is_ == 1:
                    for jj in (npy - 1, npy):
                        bl = _pt(bl, 1, jj, 0.0)
                        br = _pt(br, 1, jj, 0.0)
                if (ie + 1) == npx:
                    for jj in (npy - 1, npy):
                        bl = _pt(bl, npx, jj, 0.0)
                        br = _pt(br, npx, jj, 0.0)

        b0 = bl + br                                        # :3002-3004

        fx0_p = (1.0 - cfl) * (b_w(br, js - 1, je)
                               - cfl * b_w(b0, js - 1, je))
        fx0_m = (1.0 + cfl) * (b_w(bl, js, je + 1)
                               + cfl * b_w(b0, js, je + 1))
        base = jnp.where(cpos, v_w(js - 1, je), v_w(js, je + 1))

        if jord == 1:                                       # :3007-3025
            smt5 = jnp.abs(lim_fac * b0) < jnp.abs(bl - br)
            fx0 = jnp.where(cpos, fx0_p, fx0_m)
            add = b_w(smt5, js - 1, je) | b_w(smt5, js, je + 1)
            return jnp.where(add, base + fx0, base)

        if jord == 2:                                       # :3027-3037
            return base + jnp.where(cpos, fx0_p, fx0_m)

        if jord == 3:                                       # :3039-3072
            x0 = jnp.abs(b0)
            x1 = jnp.abs(bl - br)
            smt5 = x0 < x1
            smt6 = 3.0 * x0 < x1
            hi5 = b_w(smt5, js - 1, je) & b_w(smt5, js, je + 1)
            hi6 = b_w(smt6, js - 1, je) | b_w(smt6, js, je + 1)
            up_hi6 = b_w(br, js - 1, je) - cfl * b_w(b0, js - 1, je)
            up_hi5 = jnp.copysign(
                jnp.minimum(jnp.abs(b_w(bl, js - 1, je)),
                            jnp.abs(b_w(br, js - 1, je))),
                b_w(br, js - 1, je))
            dn_hi6 = b_w(bl, js, je + 1) + cfl * b_w(b0, js, je + 1)
            dn_hi5 = jnp.copysign(
                jnp.minimum(jnp.abs(b_w(bl, js, je + 1)),
                            jnp.abs(b_w(br, js, je + 1))),
                b_w(bl, js, je + 1))
            fx0 = jnp.where(
                cpos,
                jnp.where(hi6, up_hi6, jnp.where(hi5, up_hi5, 0.0)),
                jnp.where(hi6, dn_hi6, jnp.where(hi5, dn_hi5, 0.0)))
            return jnp.where(cpos,
                             v_w(js - 1, je) + (1.0 - cfl) * fx0,
                             v_w(js, je + 1) + (1.0 + cfl) * fx0)

        if jord == 4:                                       # :3074-3102
            x0 = jnp.abs(b0)
            x1 = jnp.abs(bl - br)
            smt5 = x0 < x1
            smt6 = 3.0 * x0 < x1
            hi5 = b_w(smt5, js - 1, je) & b_w(smt5, js, je + 1)
            hi6 = b_w(smt6, js - 1, je) | b_w(smt6, js, je + 1)
            hi5 = hi5 | hi6
            fx0 = jnp.where(cpos, fx0_p, fx0_m)
            return jnp.where(hi5, base + fx0, base)

        # -------------------------------------------- jord 5, 6, 7
        if jord == 5:                                       # :3105-3123
            flag = bl * br < 0.0
        else:                                               # :3125-3155
            flag = 3.0 * jnp.abs(b0) < jnp.abs(bl - br)
            # WMP edge fix -- ABSENT from symmetryclean; PLAIN path only.
            if (not (bounded_domain or duogrid)) and grid_type < 3:
                if js == 1:
                    for k in (0, 1):
                        p = _idx(k, o_b, n_b, "ytp_v smt6")
                        flag = flag.at[:, p].set(
                            bl[:, p] * br[:, p] < 0.0)
                if (je + 1) == npy:
                    for k in (npy - 1, npy):
                        p = _idx(k, o_b, n_b, "ytp_v smt6")
                        flag = flag.at[:, p].set(
                            bl[:, p] * br[:, p] < 0.0)
        fx0 = jnp.where(cpos, fx0_p, fx0_m)
        add = b_w(flag, js - 1, je) | b_w(flag, js, je + 1)
        return jnp.where(add, base + fx0, base)

    # ---------------------------------------------------- :3195-3351
    o_dm = js - 2
    n_dm = je + 2 - (js - 2) + 1
    o_dq = js - 3
    n_dq = je + 2 - (js - 3) + 1

    def dm_w(a, b):
        return dm[:, _rng(a, b, o_dm, n_dm, "ytp_v dm")]

    def dq_w(a, b):
        return dq[:, _rng(a, b, o_dq, n_dq, "ytp_v dq")]

    vm = v_w(js - 3, je + 1)
    vc = v_w(js - 2, je + 2)
    vp = v_w(js - 1, je + 3)
    xt = 0.25 * (vp - vm)
    dm = _nn((n_c, n_dm)).at[
        :, _rng(js - 2, je + 2, o_dm, n_dm, "ytp_v dm")].set(
        jnp.copysign(
            jnp.minimum(
                jnp.minimum(jnp.abs(xt),
                            jnp.maximum(jnp.maximum(vm, vc), vp) - vc),
                vc - jnp.minimum(jnp.minimum(vm, vc), vp)),
            xt))
    dq = _nn((n_c, n_dq)).at[
        :, _rng(js - 3, je + 2, o_dq, n_dq, "ytp_v dq")].set(
        v_w(js - 2, je + 3) - v_w(js - 3, je + 2))

    if grid_type < 3:
        al = al.at[:, _rng(js3, je3 + 1, o_al, n_al, "ytp_v al")].set(
            0.5 * (v_w(js3 - 1, je3) + v_w(js3, je3 + 1))
            + R3 * (dm_w(js3 - 1, je3) - dm_w(js3, je3 + 1)))
        w3 = _rng(js3, je3, o_b, n_b, "ytp_v bl/br")

        if jord == 8:                                       # :3213-3221
            xt = 2.0 * dm_w(js3, je3)
            bl = bl.at[:, w3].set(-jnp.copysign(
                jnp.minimum(jnp.abs(xt),
                            jnp.abs(al_w(js3, je3) - v_w(js3, je3))), xt))
            br = br.at[:, w3].set(jnp.copysign(
                jnp.minimum(jnp.abs(xt),
                            jnp.abs(al_w(js3 + 1, je3 + 1) - v_w(js3, je3))),
                xt))
        elif jord == 9:                                     # :3223-3235
            pmp_1 = -2.0 * dq_w(js3, je3)
            lac_1 = pmp_1 + 1.5 * dq_w(js3 + 1, je3 + 1)
            bl = bl.at[:, w3].set(jnp.minimum(
                jnp.maximum(jnp.maximum(0.0, pmp_1), lac_1),
                jnp.maximum(al_w(js3, je3) - v_w(js3, je3),
                            jnp.minimum(jnp.minimum(0.0, pmp_1), lac_1))))
            pmp_2 = 2.0 * dq_w(js3 - 1, je3 - 1)
            lac_2 = pmp_2 - 1.5 * dq_w(js3 - 2, je3 - 2)
            br = br.at[:, w3].set(jnp.minimum(
                jnp.maximum(jnp.maximum(0.0, pmp_2), lac_2),
                jnp.maximum(al_w(js3 + 1, je3 + 1) - v_w(js3, je3),
                            jnp.minimum(jnp.minimum(0.0, pmp_2), lac_2))))
        elif jord == 10:                                    # :3237-3262
            bl_w = al_w(js3, je3) - v_w(js3, je3)
            br_w = al_w(js3 + 1, je3 + 1) - v_w(js3, je3)
            flat = ((jnp.abs(dm_w(js3, je3)) < NEAR_ZERO_SW)
                    & ((jnp.abs(dm_w(js3 - 1, je3 - 1))
                        + jnp.abs(dm_w(js3 + 1, je3 + 1))) < NEAR_ZERO_SW))
            steep = ((jnp.abs(dm_w(js3, je3)) >= NEAR_ZERO_SW)
                     & (jnp.abs(3.0 * (bl_w + br_w))
                        > jnp.abs(bl_w - br_w)))
            pmp_1 = -2.0 * dq_w(js3, je3)
            lac_1 = pmp_1 + 1.5 * dq_w(js3 + 1, je3 + 1)
            bl_c = jnp.minimum(
                jnp.maximum(jnp.maximum(0.0, pmp_1), lac_1),
                jnp.maximum(bl_w,
                            jnp.minimum(jnp.minimum(0.0, pmp_1), lac_1)))
            pmp_2 = 2.0 * dq_w(js3 - 1, je3 - 1)
            lac_2 = pmp_2 - 1.5 * dq_w(js3 - 2, je3 - 2)
            br_c = jnp.minimum(
                jnp.maximum(jnp.maximum(0.0, pmp_2), lac_2),
                jnp.maximum(br_w,
                            jnp.minimum(jnp.minimum(0.0, pmp_2), lac_2)))
            bl = bl.at[:, w3].set(
                jnp.where(flat, 0.0, jnp.where(steep, bl_c, bl_w)))
            br = br.at[:, w3].set(
                jnp.where(flat, 0.0, jnp.where(steep, br_c, br_w)))
        else:                                               # 11 :3264-3268
            bl = bl.at[:, w3].set(al_w(js3, je3) - v_w(js3, je3))
            br = br.at[:, w3].set(al_w(js3 + 1, je3 + 1) - v_w(js3, je3))

        if js == 1 and not bounded_domain:                  # :3270-3278
            br = br.at[:, _idx(2, o_b, n_b, "ytp_v br")].set(
                al_w(3, 3)[:, 0] - v_at(2))
            xt = SW_S15 * v_at(1) + SW_S11 * v_at(2) - SW_S14 * dm_w(2, 2)[:, 0]
            br = br.at[:, _idx(1, o_b, n_b, "ytp_v br")].set(xt - v_at(1))
            bl = bl.at[:, _idx(2, o_b, n_b, "ytp_v bl")].set(xt - v_at(2))
            bl = bl.at[:, _idx(0, o_b, n_b, "ytp_v bl")].set(
                SW_S14 * dm_w(-1, -1)[:, 0] - SW_S11 * dq_w(-1, -1)[:, 0])
            x0l = (0.5 * ((2.0 * dy_at(0) + dy_at(-1)) * v_at(0)
                          - dy_at(0) * v_at(-1)) / (dy_at(0) + dy_at(-1)))
            x0r = (0.5 * ((2.0 * dy_at(1) + dy_at(2)) * v_at(1)
                          - dy_at(1) * v_at(2)) / (dy_at(1) + dy_at(2)))
            xt = x0l + x0r
            bl = bl.at[:, _idx(1, o_b, n_b, "ytp_v bl")].set(xt - v_at(1))
            br = br.at[:, _idx(0, o_b, n_b, "ytp_v br")].set(xt - v_at(0))
            if is_ == 1:
                for jj in (0, 1):
                    bl = _pt(bl, 1, jj, 0.0)
                    br = _pt(br, 1, jj, 0.0)
            if (ie + 1) == npx:
                for jj in (0, 1):
                    bl = _pt(bl, npx, jj, 0.0)
                    br = _pt(br, npx, jj, 0.0)
            # pert_ppm(ie-is+2, v(is,2), bl(is,2), br(is,2), -1)
            # -- sw_core.F90:3277, the whole i row at j = 2.
            p2 = _idx(2, o_b, n_b, "ytp_v bl/br")
            bl_p, br_p = pert_ppm(v_at(2), bl[:, p2], br[:, p2], -1)
            bl = bl.at[:, p2].set(bl_p)
            br = br.at[:, p2].set(br_p)

        if (je + 1) == npy and not bounded_domain:          # :3280-3316
            bl = bl.at[:, _idx(npy - 2, o_b, n_b, "ytp_v bl")].set(
                al_w(npy - 2, npy - 2)[:, 0] - v_at(npy - 2))
            xt = (SW_S15 * v_at(npy - 1) + SW_S11 * v_at(npy - 2)
                  + SW_S14 * dm_w(npy - 2, npy - 2)[:, 0])
            br = br.at[:, _idx(npy - 2, o_b, n_b, "ytp_v br")].set(
                xt - v_at(npy - 2))
            bl = bl.at[:, _idx(npy - 1, o_b, n_b, "ytp_v bl")].set(
                xt - v_at(npy - 1))
            br = br.at[:, _idx(npy, o_b, n_b, "ytp_v br")].set(
                SW_S11 * dq_w(npy, npy)[:, 0]
                - SW_S14 * dm_w(npy + 1, npy + 1)[:, 0])
            x0l = (0.5 * ((2.0 * dy_at(npy - 1) + dy_at(npy - 2)) * v_at(npy - 1)
                          - dy_at(npy - 1) * v_at(npy - 2))
                   / (dy_at(npy - 1) + dy_at(npy - 2)))
            x0r = (0.5 * ((2.0 * dy_at(npy) + dy_at(npy + 1)) * v_at(npy)
                          - dy_at(npy) * v_at(npy + 1))
                   / (dy_at(npy) + dy_at(npy + 1)))
            xt = x0l + x0r
            br = br.at[:, _idx(npy - 1, o_b, n_b, "ytp_v br")].set(
                xt - v_at(npy - 1))
            bl = bl.at[:, _idx(npy, o_b, n_b, "ytp_v bl")].set(
                xt - v_at(npy))
            if is_ == 1:
                for jj in (npy - 1, npy):
                    bl = _pt(bl, 1, jj, 0.0)
                    br = _pt(br, 1, jj, 0.0)
            if (ie + 1) == npx:
                for jj in (npy - 1, npy):
                    bl = _pt(bl, npx, jj, 0.0)
                    br = _pt(br, npx, jj, 0.0)
            pm2 = _idx(npy - 2, o_b, n_b, "ytp_v bl/br")
            bl_p, br_p = pert_ppm(v_at(npy - 2), bl[:, pm2], br[:, pm2], -1)
            bl = bl.at[:, pm2].set(bl_p)
            br = br.at[:, pm2].set(br_p)
    else:                                                   # :3318-3333
        al = al.at[:, _rng(js - 1, je + 2, o_al, n_al, "ytp_v al")].set(
            0.5 * (v_w(js - 2, je + 1) + v_w(js - 1, je + 2))
            + R3 * (dm_w(js - 2, je + 1) - dm_w(js - 1, je + 2)))
        wf = _rng(js - 1, je + 1, o_b, n_b, "ytp_v bl/br")
        pmp = 2.0 * dq_w(js - 2, je)
        lac = pmp - 1.5 * dq_w(js - 3, je - 1)
        br = br.at[:, wf].set(jnp.minimum(
            jnp.maximum(jnp.maximum(0.0, pmp), lac),
            jnp.maximum(al_w(js, je + 2) - v_w(js - 1, je + 1),
                        jnp.minimum(jnp.minimum(0.0, pmp), lac))))
        pmp = -2.0 * dq_w(js - 1, je + 1)
        lac = pmp + 1.5 * dq_w(js, je + 2)
        bl = bl.at[:, wf].set(jnp.minimum(
            jnp.maximum(jnp.maximum(0.0, pmp), lac),
            jnp.maximum(al_w(js - 1, je + 1) - v_w(js - 1, je + 1),
                        jnp.minimum(jnp.minimum(0.0, pmp), lac))))

    # :3335-3351 -- b0 is NOT formed; (bl+br) is recomputed inline.
    return jnp.where(
        cpos,
        v_w(js - 1, je) + (1.0 - cfl) * (
            b_w(br, js - 1, je)
            - cfl * (b_w(bl, js - 1, je) + b_w(br, js - 1, je))),
        v_w(js, je + 1) + (1.0 + cfl) * (
            b_w(bl, js, je + 1)
            + cfl * (b_w(bl, js, je + 1) + b_w(br, js, je + 1))))


# =====================================================================
# jit factories -- ONE policy per public routine (the production entry
# point AND any instrumented test wrapper are built HERE, so a policy
# drift cannot pass unnoticed).  NO donated buffers anywhere (grad-path
# doctrine, CLAUDE.md).
# =====================================================================

def make_pert_ppm_jit(fn=pert_ppm):
    """Static: ``iv`` (a python branch selector).  a0/al/ar dynamic."""
    return jax.jit(fn, static_argnums=(3,))


pert_ppm_jit = make_pert_ppm_jit()


def make_copy_corners_jit(fn=copy_corners):
    """Static: npx/npy/dir_/bounded_domain/bd/the four corner flags/
    duogrid -- index windows and python branch selectors.  ``bd`` is a
    ``Bounds`` NamedTuple, hashable BY VALUE, so equal bounds share one
    cache entry."""
    return jax.jit(fn, static_argnums=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                   static_argnames=("duogrid",))


copy_corners_jit = make_copy_corners_jit()


def make_xppm_jit(fn=xppm):
    """Static: iord + every index bound + npx/npy/bounded_domain/
    grid_type/lim_fac/duogrid.  q/c/dxa dynamic."""
    return jax.jit(
        fn, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15,
                            16),
        static_argnames=("duogrid",))


xppm_jit = make_xppm_jit()


def make_yppm_jit(fn=yppm):
    """Same policy as :func:`make_xppm_jit` (q/c/dya dynamic)."""
    return jax.jit(
        fn, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15,
                            16),
        static_argnames=("duogrid",))


yppm_jit = make_yppm_jit()


def make_deln_flux_jit(fn=deln_flux):
    """Static: nord (loop trip count + index windows), the index bounds,
    npx/npy, bd and the five domain/corner flags.  ``damp``, q, fx, fy,
    the three grid arrays and the two optionals stay dynamic (``damp``
    only multiplies arrays -- it is never compared in a python
    branch)."""
    return jax.jit(
        fn, static_argnums=(0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19),
        static_argnames=("duogrid",))


deln_flux_jit = make_deln_flux_jit()


def make_fv_tp_2d_jit(fn=fv_tp_2d):
    """Static: npx/npy/hord/da_min/bd/lim_fac/bounded_domain/grid_type/
    the four corner flags and nord/damp_c/damp_smag (all three are
    compared in python ``if``s at tp_core.F90:196/:200).  The arrays and
    the ``None``-or-array optionals stay dynamic; passing ``None``
    instead of an array changes the pytree and correctly retraces."""
    return jax.jit(
        fn, static_argnums=(3, 4, 5, 14, 15, 18, 19, 20, 21, 22, 23, 24),
        static_argnames=("nord", "damp_c", "damp_smag", "duogrid"))


fv_tp_2d_jit = make_fv_tp_2d_jit()


def make_xtp_u_jit(fn=xtp_u):
    """Static: every index bound, iord, npx/npy, grid_type,
    bounded_domain, lim_fac, duogrid.  c/u/v/dx/rdx dynamic."""
    return jax.jit(
        fn, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 11, 14, 15, 16, 17,
                            18),
        static_argnames=("duogrid",))


xtp_u_jit = make_xtp_u_jit()


def make_ytp_v_jit(fn=ytp_v):
    """Same policy as :func:`make_xtp_u_jit` (c/u/v/dy/rdy dynamic)."""
    return jax.jit(
        fn, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 11, 14, 15, 16, 17,
                            18),
        static_argnames=("duogrid",))


ytp_v_jit = make_ytp_v_jit()
