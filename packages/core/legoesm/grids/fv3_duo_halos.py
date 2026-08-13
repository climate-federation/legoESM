"""FV3 duo-grid halo machinery -- JAX lane (exchanges, k2e ring remap,
corner fills, the two duo flux-averaging barriers, and the composed
``ext_scalar`` / ``ext_vector``).

Functional, jit-compatible mirror of the certified NumPy fp64 lane
(``fv3_native_gridstruct.py`` halo/barrier half + ``fv3_native_ext_vector.py``),
itself a loop-faithful port of the pinned oracle
(``/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean``,
``dyn_core.F90`` md5 ``e5a5fab9…``, 3128 lines):

* ``tools/fv_duogrid.F90:456-502`` ``ext_scalar_2d`` / ``:505-569``
  ``ext_scalar_3d``, ``:626-975`` ``ext_vector``, ``:977-1137``
  ``cube_rmp``, ``:1719-1903`` ``fill_corner_region_2d``,
  ``:2159-2272`` ``compute_lagrange_coeff``, ``:2590-2674``
  ``cubed_a2c_halo``, ``:2676-2763`` ``cubed_a2d_halo``, ``:2765-2844``
  ``c2l_ord2_cgrid``;
* ``tools/fv_mp_mod.F90:1032`` ``fill_corners_2d_r8``, ``:1257``
  ``fill_corners_dgrid_r8``, ``:1369`` ``fill_corners_cgrid_r8``,
  ``:1433`` ``fill_corners_agrid_r8``;
* **the two duo barriers** -- ``model/dyn_core.F90:853-901`` (BARRIER 1,
  ``mpp_get_boundary`` at ``:872`` with ``gridtype=CGRID_NE`` at
  ``:874``, slot selection ``if (iq==1 .or. iq==4 .or. iq>4)`` at
  ``:856``) and ``:969-1011`` (BARRIER 2, ``mpp_get_boundary`` at
  ``:984`` with ``gridtype=BGRID_NE`` at ``:986``).  Every line number
  in this docstring was verified with ``grep -n`` / ``sed -n`` against
  the pinned tree, not recalled.  Three further ``mpp_get_boundary``
  calls (``:1037``, ``:1141``, ``:1189``) are commented out, so exactly
  two barriers are live.

The NumPy lane stays the oracle-parity reference; this module is the
production/JAX twin and is certified AGAINST the NumPy lane, never
against the Fortran directly (one authority per hop, strategy doc §2).

Doctrine, following ``legoesm.core.fv3_nh_core``:

* **Functional**: the NumPy lane mutates its six-face lists in place;
  every twin here RETURNS new arrays.  No aliasing, no output
  parameters, no ``donate_argnums`` (grad-path doctrine, CLAUDE.md).
* **Explicit x64**: every field operand must arrive float64 (the oracle
  build is ``-fdefault-real-8``); a float32 operand raises
  ``TypeError`` at entry exactly like the NumPy lane's ``_require_f64``
  (the check reads only static dtypes, so it is jit-safe).
* **Setup stays NumPy.**  ``build_ext_context`` /
  ``build_fv3_native_gridstruct`` / ``compute_fv3_native_k2e`` /
  ``_CornerLagrange.__init__`` are NOT ported; they run once and their
  output is converted here by :func:`build_jax_duo_halo_tables` into
  static numpy index/weight tables carried by :class:`DuoHaloTables`,
  which is passed to every kernel as a **static** (identity-hashable)
  jit argument.  Index tables therefore index under jit as compile-time
  constants and are never differentiated.

STATE LAYOUT CONTRACT (binding on the rest of the port)
--------------------------------------------------------
This module is where the JAX lane's six-face state layout is decided,
so it is stated here as a contract rather than left implicit.  Every
other module of this port follows it.

1. **Face axis, not a list.**  Six-face state of ONE stagger is a
   single array with a LEADING face axis of length 6 --
   ``(6, m0, m1)`` -- not a 6-element list/tuple PyTree.  Rationale:
   every operation in this module is a halo scatter, and the leading
   axis is what turns six per-face scatters into one; a list of six
   traced arrays would also work under jit but forfeits exactly the
   batching this module exists to provide.  ``stack6`` / ``unstack6``
   convert to and from the NumPy lane's list container at the lane
   boundary; nothing inside the JAX lane uses lists.

2. **The face axis is stackable; the STAGGER axis is not.**  A, B, C
   and D fields have different per-face shapes -- with ``ma = n+2*ng``
   and ``mb = n+2*ng+1``: A ``(6, ma, ma)``, B ``(6, mb, mb)``,
   C ``uc (6, mb, ma)`` + ``vc (6, ma, mb)``, D ``u (6, ma, mb)`` +
   ``v (6, mb, ma)``.  Within one stagger all six faces share a shape,
   so the stack always exists; ACROSS staggers it does not, which is
   why a vector pair is carried as TWO arrays and the internal batched
   buffers are a CONCATENATION OF FLATTENINGS (:class:`_FlatLayout`),
   never a stack.  A future module must not "unify" u and v into one
   array: their shapes differ by construction.

3. **Face ordering** is FV3 reference tile numbering: array face
   ``t`` is Fortran tile ``t+1``, matching ``neighbor_tiles`` /
   ``neighbor_index`` (1-based) as used by the NumPy lane.  There is
   no permutation or relabelling anywhere in this module.

4. **Halo width and array origin.**  Stepper fields carry ``ng``
   rings (the deck's 3) and put Fortran index ``1-ng`` at numpy 0 on
   BOTH axes, exactly like the NumPy lane, so a cell axis spans
   Fortran ``1-ng .. n+ng`` and a node axis ``1-ng .. n+1+ng``.  The
   internal geographic lattice of the vector flow carries ``ngp`` = 4
   rings (upstream ``dg%bd%ng``, ``set_bd_ext_duo``), DERIVED from the
   context's own bases rather than re-declared.  The two barriers are
   the one exception and are flagged in their docstrings: their
   operands are COMPUTE-RING slabs with Fortran ``(is, js) = (1, 1)``
   at numpy ``(0, 0)`` and no halo at all -- fx ``(6, npx, n)``,
   fy ``(6, n, npx)``, B arrays ``(6, npx, npx)``.

5. **Static vs traced leaves.**  STATIC (never traced, never
   differentiated, hashed by identity into the jit cache):
   :class:`DuoHaloTables` in its entirety -- integer index tables, the
   cube topology they encode, Lagrange/k2e weights, grid metrics
   (``amat``/``dx``/``dy``/``vlon4``/``vlat4``/``ew4``/``es4``),
   ``n``/``ng``/``ngp``/``nq``, the stagger and ring selector strings,
   and the ``mySign`` of the corner fills.  TRACED: field data only.
   Consequence to respect downstream: differentiating a kernel means
   differentiating w.r.t. its field arguments, and grid metrics are
   NOT differentiable parameters in this lane.

6. **No traced ``jnp.where``.**  ``jnp.where`` is a SELECT: both
   branches are evaluated, so a NaN/Inf produced in the unselected
   branch contaminates the reverse-mode gradient of the selected one.
   This module has ZERO ``jnp.where`` sites on traced data (there is
   no data-dependent branch in halo machinery at all).  The single
   ``where`` in the file is a ``numpy`` one over the STATIC sign-power
   table inside :func:`_apply_scatter`, evaluated at trace time on
   constants and folded away before any tracer exists.  Any future
   addition here must state which of the three shapes it uses -- a
   total-and-finite ``jnp.where``, a sanitize-then-select, or a
   ``lax.cond`` on a scalar predicate.

Write shape (the design question this module exists to answer)
--------------------------------------------------------------
The NumPy lane expresses all of this as scalar-indexed writes into
``fort`` (1-based adapter) arrays.  Under jit each such write would be
one ``.at[i, j].set(v)``; a C48 A-grid exchange has ~2300 of them per
face.  Every group below is therefore emitted as ONE batched
``.at[idx].set(val)`` over a STATIC index table -- but only after the
group is proved order-independent, and the proof is MECHANICAL, not
prose: :func:`_check_batchable` recomputes each group's destination and
source index sets at table-build time and raises if they intersect.
A group whose writes really do feed each other is kept sequential (it
would fail that check loudly rather than silently reassociate).

The per-group verdicts, with the argument each rests on:

* **strip exchanges** (A/B scalar, C/D vector): every source read is
  guarded to the neighbour's COMPUTE domain (``if 1 <= ci <= n`` in
  ``exchange_agrid_scalar_halos``; the analogous window in the vector
  exchanges), and every destination is a halo slot.  Compute slots are
  never written, so all six faces' strips are mutually independent and
  the NumPy loop ``for t in 1..6`` may be flattened into one scatter.
* **corner fills after the strips**: their sources are side-strip slots
  (halo x compute) and their destinations are corner-diagonal slots
  (halo x halo) -- disjoint, so the ng^2 writes of one corner are
  mutually independent.  They must still run AFTER the strip phase, and
  moving all six faces' corner fills past the other faces' strip writes
  is legal because no corner destination is a strip source (checked by
  :func:`_check_no_clobber`).
* **k2e ring remap**: the NumPy lane takes an explicit snapshot
  (``src = f.copy()``) and every write reads only the snapshot, so the
  group is independent BY CONSTRUCTION.
* **corner-region Lagrange fill**: the NumPy ``fill`` looks sequential
  (six directional fills, then three averaged diagonal slots per
  corner), but reading ``_CornerLagrange._apply_dir``/``diag`` shows
  ``diag`` copies ``f``, applies ONE directional fill to the copy and
  reads back only the target slot -- i.e. ``diag`` is
  ``0.5*(wsum_dir1 + wsum_dir2)`` evaluated on the CURRENT field.  All
  36 targets read abscissa columns/rows inside the compute domain
  (``X+``: ``i in [ie-3, ie]``; ``X-``: ``i in [1, 4]``; likewise for
  ``Y``) while every target is a wedge slot, so the whole fill is one
  independent group -- PROVIDED the two sets do not collide, which
  needs ``n >= 4`` (at ``n = 3`` the ``X-`` source column 4 IS the
  ``X+`` wedge).  The check enforces it; it is not assumed.
  The corner sequence is NOT composed into a precomputed sparse matrix
  (that would be a mathematical re-derivation, forbidden by R1); it
  stays two scatters -- directional targets, then diagonal targets --
  in the NumPy lane's own order.
* **barriers**: the NumPy lane is already explicitly two-phase (gather
  every partner value, then apply), mirroring ``mpp_get_boundary``
  buffering.  ``flat.at[dst].set(0.5 * (flat[dst] + sign*flat[src]))``
  reads both operands from the pre-update buffer, which is the same
  thing.
* **strip write-back** (``rmp_s/n/w/e``): sources are a DIFFERENT array
  (the projected ng=4 lattice), so the group is independent; the S/N
  then W/E passes DO write some slots twice, and last-wins is preserved
  by deduplicating destinations keeping the LAST record
  (:func:`_dedup_last`) -- required, because ``.at[].set()`` with
  duplicate indices has no defined order in JAX.

Reduction association
---------------------
``(w * vals).sum()`` in the NumPy lane is replaced by an explicit
left-to-right unrolled accumulation over the STATIC stencil length
(:func:`_dot_static`), which is the loop order the Fortran writes and
the order NumPy's pairwise summation degenerates to below its 8-element
blocking threshold.  This is asserted by the parity tests, not assumed:
if the two differ it is a one-ulp residual in the "accumulating"
tolerance class of the strategy doc §4, and the test reports the
measured value.

Differentiability
-----------------
Every kernel here is LINEAR (affine, where a NaN-filled constant
background is involved) in its field arguments: gathers, scatters,
multiplications by grid-metric constants and sums.  There is no
limiter, no ``jnp.maximum``, no data-dependent branch -- so there are
no switching points, gradients are exact, and second derivatives are
identically zero.  ``check_grads(order=2)`` therefore passes
everywhere but is a WEAK gate on a linear operator; the adjoint
(dot-product) identity test in the test module is the strong one.
NaN values living in never-consumed slots (``c2l`` outside its window,
``_pack_p1``'s outer ring) do NOT poison gradients precisely because
the local derivatives are the finite metric constants, not the values.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.grids.fv3_native_halos import (
    compute_fv3_native_k2e,
    neighbor_index,
    neighbor_tiles,
)

# Wedge depth of fill_corner_region: upstream writes exactly three rings
# of the corner-diagonal region (fv_duogrid.F90:1743-1901 -- the literal
# ie+1/ie+2/ie+3 offsets), independently of the field's own halo width.
# Mirrors the literal offsets in _CornerLagrange.fill.
_WEDGE = 3

# fv_duogrid.F90:80 interporder; the Lagrange stencil is order+1 wide.
# Only used here to state the corner-fill independence condition.
_INTERP_ORDER = 3


def _require_f64_jax(fname: str, arrays: dict) -> None:
    """Static-dtype gate mirroring the NumPy lane's ``_require_f64``.

    Reads only ``.dtype`` (static under jit): a float32 operand would
    otherwise be silently upcast -- or, with x64 disabled, the whole
    exchange would silently run in float32 -- and the oracle build is
    ``-fdefault-real-8``.

    DUPLICATION, deliberate and temporary: this is character-identical
    to ``fv3_nh_core._require_f64_jax``, which is private and therefore
    not importable across modules (the empty-allowlist ratchet
    ``tests/test_no_private_cross_imports.py``).  It is not promoted
    there in this change because that file is under concurrent edit;
    FOLLOW-UP: promote the ``fv3_nh_core`` definition to a public name
    and delete this copy.
    """
    for name, a in arrays.items():
        if jnp.asarray(a).dtype != jnp.float64:
            raise TypeError(
                f"{fname}: {name} must be float64 (got "
                f"{jnp.asarray(a).dtype}); enable jax_enable_x64 and pass "
                f"f64 operands (oracle build is -fdefault-real-8)")


# ---------------------------------------------------------------------------
# static (setup-time) record containers
# ---------------------------------------------------------------------------

class _Scatter:
    """``out[dst] = base * sign**sgn_pow * in[src]`` over a flat buffer.

    ``dst``/``src`` are int32 flat indices into the layout the table was
    built for.  Two independent sign channels, because the NumPy lane
    has two: ``base`` is the record's own orientation sign (the mpp
    NE-vector map derivative in the C/D strip exchanges, +1 everywhere
    else), and ``sgn_pow`` in {0, 1} selects whether the CALLER's static
    ``mySign`` multiplies that record -- the ``fill_corners_*`` pair
    routines sign only SOME of their four statements per corner.
    """

    __slots__ = ("dst", "src", "base", "sgn_pow", "name")

    def __init__(self, dst, src, base, sgn_pow, name):
        self.dst = np.asarray(dst, dtype=np.int32)
        self.src = np.asarray(src, dtype=np.int32)
        self.base = np.asarray(base, dtype=np.float64)
        self.sgn_pow = np.asarray(sgn_pow, dtype=np.int8)
        self.name = name


class _Blend:
    """``out[dst] = 0.5 * (in[dst] + sign * in[src])`` (a duo barrier)."""

    __slots__ = ("dst", "src", "sign", "name")

    def __init__(self, dst, src, sign, name):
        self.dst = np.asarray(dst, dtype=np.int32)
        self.src = np.asarray(src, dtype=np.int32)
        self.sign = np.asarray(sign, dtype=np.float64)
        self.name = name


class _Stencil:
    """``out[dst] = sum_l w[:, l] * in[src[:, l]]`` (k2e ring / Lagrange)."""

    __slots__ = ("dst", "src", "w", "name")

    def __init__(self, dst, src, w, name):
        self.dst = np.asarray(dst, dtype=np.int32)
        self.src = np.asarray(src, dtype=np.int32)
        self.w = np.asarray(w, dtype=np.float64)
        self.name = name


class _StencilPair:
    """``out[dst] = 0.5 * (dot(wx, in[srcx]) + dot(wy, in[srcy]))``.

    The three averaged diagonal slots per corner of
    ``fill_corner_region_2d`` (fv_duogrid.F90:1719-1903).
    """

    __slots__ = ("dst", "srcx", "wx", "srcy", "wy", "name")

    def __init__(self, dst, srcx, wx, srcy, wy, name):
        self.dst = np.asarray(dst, dtype=np.int32)
        self.srcx = np.asarray(srcx, dtype=np.int32)
        self.wx = np.asarray(wx, dtype=np.float64)
        self.srcy = np.asarray(srcy, dtype=np.int32)
        self.wy = np.asarray(wy, dtype=np.float64)
        self.name = name


class _FlatLayout:
    """Flat index arithmetic over one or two stacked six-face arrays.

    Each entry is ``(m0, m1)``; the buffer is the concatenation of the
    ``(6, m0, m1)`` arrays' row-major flattenings.  ``idx`` RAISES on an
    out-of-range subscript: the NumPy lane's negative ``i - lo`` would
    silently WRAP (a plausible-looking value, the worst failure mode),
    so the table builder refuses to encode one.
    """

    __slots__ = ("shapes", "sizes", "bases", "total")

    def __init__(self, *shapes):
        self.shapes = tuple(tuple(int(x) for x in s) for s in shapes)
        self.sizes = [6 * s[0] * s[1] for s in self.shapes]
        self.bases = [int(b) for b in np.cumsum([0] + self.sizes)[:-1]]
        self.total = int(sum(self.sizes))

    def idx(self, k: int, face: int, i0: int, j0: int) -> int:
        m0, m1 = self.shapes[k]
        if not (0 <= i0 < m0 and 0 <= j0 < m1):
            raise IndexError(
                f"duo halo table: subscript ({i0}, {j0}) outside array "
                f"{k} of shape (6, {m0}, {m1}) -- the NumPy lane would "
                f"wrap a negative index and return a plausible wrong "
                f"number")
        if not (0 <= face < 6):
            raise IndexError(f"duo halo table: face {face} outside 0..5")
        return self.bases[k] + (face * m0 + i0) * m1 + j0


# ---------------------------------------------------------------------------
# build-time independence checks (the batching licence, machine-enforced)
# ---------------------------------------------------------------------------

def _check_batchable(name: str, dst, src) -> None:
    """A grouped scatter is order-identical iff it reads nothing it writes.

    ``dst`` may contain duplicates (resolved last-wins by
    :func:`_dedup_last`); the fatal condition is an intersection with
    the source set, which would make the batched result depend on an
    execution order JAX does not define.
    """
    d = np.unique(np.asarray(dst, dtype=np.int64))
    s = np.unique(np.asarray(src, dtype=np.int64).ravel())
    bad = np.intersect1d(d, s, assume_unique=True)
    if bad.size:
        raise ValueError(
            f"{name}: {bad.size} slot(s) are both written and read by the "
            f"same batched group (first flat index {int(bad[0])}); the "
            f"NumPy lane's sequential order is load-bearing there and "
            f"this group may not be batched")


def _check_no_clobber(name: str, later_dst, earlier_src) -> None:
    """Legality of hoisting an earlier phase's writes ahead of a later one.

    The NumPy lane interleaves (strips_1, corners_1, strips_2, ...); the
    JAX lane runs (all strips, all corners).  Some reads of phase 1 that
    originally happened AFTER a phase-2 write now happen before it, so
    phase 2 must not write anything phase 1 reads.
    """
    d = np.unique(np.asarray(later_dst, dtype=np.int64))
    s = np.unique(np.asarray(earlier_src, dtype=np.int64).ravel())
    bad = np.intersect1d(d, s, assume_unique=True)
    if bad.size:
        raise ValueError(
            f"{name}: phase reordering is illegal -- {bad.size} slot(s) "
            f"written by the later phase are read by the earlier one "
            f"(first flat index {int(bad[0])})")


def _dedup_last(dst, *cols):
    """Keep the LAST record per destination, preserving NumPy semantics.

    ``.at[idx].set()`` with repeated indices has no defined winner in
    JAX; the NumPy lane's repeated writes (the ``rmp_s/n/w/e`` strip
    passes) are plain last-wins.  Safe only together with
    :func:`_check_batchable`, which guarantees the duplicates do not
    feed each other.
    """
    dst = np.asarray(dst, dtype=np.int64)
    if dst.size == 0:
        return (dst.astype(np.int32),) + tuple(np.asarray(c) for c in cols)
    order = np.arange(dst.size)
    # stable sort by (dst, order) then take the last per dst
    key = np.lexsort((order, dst))
    ds = dst[key]
    keep_sorted = np.ones(ds.size, dtype=bool)
    keep_sorted[:-1] = ds[1:] != ds[:-1]
    keep = np.sort(key[keep_sorted])
    return (dst[keep].astype(np.int32),) + tuple(
        np.asarray(c)[keep] for c in cols)


# ---------------------------------------------------------------------------
# apply helpers (the only places that touch traced data)
# ---------------------------------------------------------------------------

def _dot_static(w: np.ndarray, v):
    """``sum_l w[..., l] * v[..., l]`` evaluated LEFT TO RIGHT.

    ``w`` is a static numpy weight table, ``v`` the gathered traced
    values; the trailing length is static and small (2 or 4 for k2e,
    ``_INTERP_ORDER + 1`` for the corner Lagrange, 3 for the Cartesian
    projections).  An explicit unroll is the mechanical image of the
    Fortran/NumPy accumulation order -- ``jnp.sum`` would let XLA
    reassociate.
    """
    acc = w[..., 0] * v[..., 0]
    for lev in range(1, w.shape[-1]):
        acc = acc + w[..., lev] * v[..., lev]
    return acc


def _pack(*arrs):
    return jnp.concatenate([jnp.asarray(a).reshape(-1) for a in arrs])


def _unpack(layout: _FlatLayout, flat):
    out = []
    for k, s in enumerate(layout.shapes):
        b = layout.bases[k]
        out.append(flat[b:b + layout.sizes[k]].reshape((6,) + s))
    return tuple(out)


def _apply_scatter(flat, sc: _Scatter, sign: float = 1.0):
    # STATIC numpy where over the sign-power table, NOT jnp.where: both
    # operands are compile-time constants, so this folds away before any
    # tracer exists and cannot contaminate a gradient (see the module
    # docstring's layout contract, item 6).
    eff = sc.base * np.where(sc.sgn_pow == 1, float(sign), 1.0)
    return flat.at[sc.dst].set(flat[sc.src] * eff)


def _apply_blend(flat, bl: _Blend):
    return flat.at[bl.dst].set(0.5 * (flat[bl.dst] + bl.sign * flat[bl.src]))


def _apply_stencil(out_flat, src_flat, st: _Stencil):
    return out_flat.at[st.dst].set(_dot_static(st.w, src_flat[st.src]))


def _apply_stencil_pair(out_flat, src_flat, sp: _StencilPair):
    vx = _dot_static(sp.wx, src_flat[sp.srcx])
    vy = _dot_static(sp.wy, src_flat[sp.srcy])
    return out_flat.at[sp.dst].set(0.5 * (vx + vy))


def stack6(f6) -> jnp.ndarray:
    """NumPy lane's list of six face arrays -> one ``(6, …)`` array."""
    return jnp.asarray(np.stack([np.asarray(a) for a in f6], axis=0))


def unstack6(x) -> tuple:
    """``(6, …)`` array -> a six-tuple, the NumPy lane's container."""
    return tuple(x[t] for t in range(6))


# ---------------------------------------------------------------------------
# table builders -- one per NumPy routine, replaying ITS index arithmetic
# ---------------------------------------------------------------------------

def _agrid_strip_records(lay, k, n, ng):
    """fv3_native_gridstruct.exchange_agrid_scalar_halos:1304-1318."""
    sg_npx = 2 * n + 1
    lo = 1 - ng
    dst, src = [], []
    for tile in range(1, 7):
        nw, ne, ns, nn = neighbor_tiles(tile)
        strips = (
            (nw, range(1 - ng, 0 + 1), range(1, n + 1)),
            (ne, range(n + 1, n + ng + 1), range(1, n + 1)),
            (ns, range(1, n + 1), range(1 - ng, 0 + 1)),
            (nn, range(1, n + 1), range(n + 1, n + ng + 1)),
        )
        for n_src, fi_range, fj_range in strips:
            for fi in fi_range:
                for fj in fj_range:
                    si, sj = 2 * fi, 2 * fj
                    ii, jj = neighbor_index(si, sj, tile, n_src,
                                            sg_npx, sg_npx)
                    ci, cj = ii // 2, jj // 2
                    if 1 <= ci <= n and 1 <= cj <= n:
                        dst.append(lay.idx(k, tile - 1, fi - lo, fj - lo))
                        src.append(lay.idx(k, n_src - 1, ci - lo, cj - lo))
    return dst, src


def _bgrid_strip_records(lay, k, n, ng):
    """fv3_native_gridstruct.exchange_bgrid_scalar_halos:1182-1196."""
    sg_npx = 2 * n + 1
    npx = n + 1
    lo = 1 - ng
    dst, src = [], []
    for tile in range(1, 7):
        nw, ne, ns, nn = neighbor_tiles(tile)
        strips = (
            (nw, range(1 - ng, 0 + 1), range(1, npx + 1)),
            (ne, range(npx + 1, npx + ng + 1), range(1, npx + 1)),
            (ns, range(1, npx + 1), range(1 - ng, 0 + 1)),
            (nn, range(1, npx + 1), range(npx + 1, npx + ng + 1)),
        )
        for n_src, fi_range, fj_range in strips:
            for fi in fi_range:
                for fj in fj_range:
                    si, sj = 2 * fi - 1, 2 * fj - 1
                    ii, jj = neighbor_index(si, sj, tile, n_src,
                                            sg_npx, sg_npx)
                    bi, bj = (ii + 1) // 2, (jj + 1) // 2
                    if 1 <= bi <= npx and 1 <= bj <= npx:
                        dst.append(lay.idx(k, tile - 1, fi - lo, fj - lo))
                        src.append(lay.idx(k, n_src - 1, bi - lo, bj - lo))
    return dst, src


def _cgrid_strip_records(lay, n, ng):
    """fv3_native_gridstruct.exchange_cgrid_vector_halos:1225-1284.

    Buffer 0 = uc (x-faces), buffer 1 = vc (y-faces).  ``src_value`` /
    ``src_value_y`` pick the source COMPONENT and orientation sign from
    the discrete rotation's map derivative (mpp's NE-vector convention);
    both are static index arithmetic and are replayed verbatim.
    """
    sg_npx = 2 * n + 1
    npx = n + 1
    lo = 1 - ng
    dst, src, sgn = [], [], []

    def src_value(si, sj, tile, n_src):
        sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
        sii2, sjj2 = neighbor_index(si + 2, sj, tile, n_src, sg_npx, sg_npx)
        dii, djj = sii2 - sii, sjj2 - sjj
        if dii != 0:                      # axes aligned: uc <- uc
            s = 1.0 if dii > 0 else -1.0
            fi, fj = (sii + 1) // 2, sjj // 2
            return lay.idx(0, n_src - 1, fi - lo, fj - lo), s
        s = 1.0 if djj > 0 else -1.0      # axes swapped: uc <- vc
        fi, fj = sii // 2, (sjj + 1) // 2
        return lay.idx(1, n_src - 1, fi - lo, fj - lo), s

    def src_value_y(si, sj, tile, n_src):
        sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
        sii2, sjj2 = neighbor_index(si, sj + 2, tile, n_src, sg_npx, sg_npx)
        dii, djj = sii2 - sii, sjj2 - sjj
        if djj != 0:                      # axes aligned: vc <- vc
            s = 1.0 if djj > 0 else -1.0
            fi, fj = sii // 2, (sjj + 1) // 2
            return lay.idx(1, n_src - 1, fi - lo, fj - lo), s
        s = 1.0 if dii > 0 else -1.0      # axes swapped: vc <- uc
        fi, fj = (sii + 1) // 2, sjj // 2
        return lay.idx(0, n_src - 1, fi - lo, fj - lo), s

    for tile in range(1, 7):
        nw, ne, ns, nn = neighbor_tiles(tile)
        strips = (
            (nw, range(1 - ng, 0 + 1), "i"),
            (ne, range(npx + 1, npx + ng + 1), "i"),
            (ns, range(1 - ng, 0 + 1), "j"),
            (nn, range(npx + 1, npx + ng + 1), "j"),
        )
        for n_src, rng, axis in strips:
            if axis == "i":
                for fi in rng:
                    for fj in range(1, n + 1):
                        s_idx, s = src_value(2 * fi - 1, 2 * fj, tile, n_src)
                        dst.append(lay.idx(0, tile - 1, fi - lo, fj - lo))
                        src.append(s_idx)
                        sgn.append(s)
                vc_cols = (range(1 - ng, 0 + 1) if rng.start < 1
                           else range(npx, npx + ng))
                for fi in vc_cols:
                    for fj in range(1, n + 2):
                        s_idx, s = src_value_y(2 * fi, 2 * fj - 1,
                                               tile, n_src)
                        dst.append(lay.idx(1, tile - 1, fi - lo, fj - lo))
                        src.append(s_idx)
                        sgn.append(s)
            else:
                for fj in rng:
                    for fi in range(1, n + 1):
                        s_idx, s = src_value_y(2 * fi, 2 * fj - 1,
                                               tile, n_src)
                        dst.append(lay.idx(1, tile - 1, fi - lo, fj - lo))
                        src.append(s_idx)
                        sgn.append(s)
                uc_rows = (range(1 - ng, 0 + 1) if rng.start < 1
                           else range(npx, npx + ng))
                for fj in uc_rows:
                    for fi in range(1, n + 2):
                        s_idx, s = src_value(2 * fi - 1, 2 * fj, tile, n_src)
                        dst.append(lay.idx(0, tile - 1, fi - lo, fj - lo))
                        src.append(s_idx)
                        sgn.append(s)
    return dst, src, sgn


def _dgrid_strip_records(lay, n, ng):
    """fv3_native_gridstruct.exchange_dgrid_vector_halos:1345-1395.

    Buffer 0 = u (x-component on y-faces), buffer 1 = v.
    """
    sg_npx = 2 * n + 1
    npx = n + 1
    lo = 1 - ng
    dst, src, sgn = [], [], []

    def src_u(si, sj, tile, n_src):
        sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
        sii2, sjj2 = neighbor_index(si + 2, sj, tile, n_src, sg_npx, sg_npx)
        dii, djj = sii2 - sii, sjj2 - sjj
        if dii != 0:                      # aligned: u <- u
            s = 1.0 if dii > 0 else -1.0
            fi, fj = sii // 2, (sjj + 1) // 2
            return lay.idx(0, n_src - 1, fi - lo, fj - lo), s
        s = 1.0 if djj > 0 else -1.0      # swapped: u <- v
        fi, fj = (sii + 1) // 2, sjj // 2
        return lay.idx(1, n_src - 1, fi - lo, fj - lo), s

    def src_v(si, sj, tile, n_src):
        sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
        sii2, sjj2 = neighbor_index(si, sj + 2, tile, n_src, sg_npx, sg_npx)
        dii, djj = sii2 - sii, sjj2 - sjj
        if djj != 0:                      # aligned: v <- v
            s = 1.0 if djj > 0 else -1.0
            fi, fj = (sii + 1) // 2, sjj // 2
            return lay.idx(1, n_src - 1, fi - lo, fj - lo), s
        s = 1.0 if dii > 0 else -1.0      # swapped: v <- u
        fi, fj = sii // 2, (sjj + 1) // 2
        return lay.idx(0, n_src - 1, fi - lo, fj - lo), s

    for tile in range(1, 7):
        nw, ne, ns, nn = neighbor_tiles(tile)
        strips = (
            (nw, range(1 - ng, 0 + 1), "i"),
            (ne, range(n + 1, n + ng + 1), "i"),
            (ns, range(1 - ng, 0 + 1), "j"),
            (nn, range(npx + 1, npx + ng + 1), "j"),
        )
        for n_src, rng, axis in strips:
            if axis == "i":
                for fi in rng:
                    for fj in range(1, npx + 1):
                        s_idx, s = src_u(2 * fi, 2 * fj - 1, tile, n_src)
                        dst.append(lay.idx(0, tile - 1, fi - lo, fj - lo))
                        src.append(s_idx)
                        sgn.append(s)
                v_cols = (range(1 - ng, 0 + 1) if rng.start < 1
                          else range(npx + 1, npx + ng + 1))
                for fi in v_cols:
                    for fj in range(1, n + 1):
                        s_idx, s = src_v(2 * fi - 1, 2 * fj, tile, n_src)
                        dst.append(lay.idx(1, tile - 1, fi - lo, fj - lo))
                        src.append(s_idx)
                        sgn.append(s)
            else:
                for fj in rng:
                    for fi in range(1, n + 1):
                        s_idx, s = src_u(2 * fi, 2 * fj - 1, tile, n_src)
                        dst.append(lay.idx(0, tile - 1, fi - lo, fj - lo))
                        src.append(s_idx)
                        sgn.append(s)
                vj = (range(1 - ng, 0 + 1) if rng.start <= 0
                      else range(n + 1, n + ng + 1))
                for fj in vj:
                    for fi in range(1, npx + 1):
                        s_idx, s = src_v(2 * fi - 1, 2 * fj, tile, n_src)
                        dst.append(lay.idx(1, tile - 1, fi - lo, fj - lo))
                        src.append(s_idx)
                        sgn.append(s)
    return dst, src, sgn


def _corner_records(kind: str, ka: int, kb: int, npx: int, ng: int, lo: int):
    """The six ``_fill_corners_*`` index maps (fv3_native_gridstruct:450-528).

    ``kind`` selects the map; ``ka``/``kb`` are the buffer slots of the
    ``X``/``Y`` arrays (equal for the single-array scalar maps).
    Returns ``(dst, src, sgn_pow)`` as ``(buffer, i0, j0)`` triples in
    LOCAL (tile-0, numpy-origin) coordinates, in the NumPy statement
    order.  Oracle bodies: ``fv_mp_mod.F90:1032`` (scalar X/Y dir),
    ``:1257`` dgrid, ``:1369`` cgrid, ``:1433`` agrid pair.
    """
    npy = npx
    dst, src, pw = [], [], []

    def rec(kd, di, dj, ks, si, sj, signed):
        dst.append((kd, di - lo, dj - lo))
        src.append((ks, si - lo, sj - lo))
        pw.append(1 if signed else 0)

    rng = range(1, ng + 1)
    if kind == "bgrid_x":
        for j in rng:
            for i in rng:
                rec(ka, 1 - i, 1 - j, ka, 1 - j, i + 1, False)
                rec(ka, 1 - i, npy + j, ka, 1 - j, npy - i, False)
                rec(ka, npx + i, 1 - j, ka, npx + j, i + 1, False)
                rec(ka, npx + i, npy + j, ka, npx + j, npy - i, False)
    elif kind == "agrid_x":
        for j in rng:
            for i in rng:
                rec(ka, 1 - i, 1 - j, ka, 1 - j, i, False)
                rec(ka, 1 - i, npy - 1 + j, ka, 1 - j, npy - i, False)
                rec(ka, npx - 1 + i, 1 - j, ka, npx - 1 + j, i, False)
                rec(ka, npx - 1 + i, npy - 1 + j, ka, npx - 1 + j,
                    npy - i, False)
    elif kind == "agrid_y":
        for j in rng:
            for i in rng:
                rec(ka, 1 - j, 1 - i, ka, i, 1 - j, False)
                rec(ka, 1 - j, npy - 1 + i, ka, i, npy - 1 + j, False)
                rec(ka, npx - 1 + j, 1 - i, ka, npx - i, 1 - j, False)
                rec(ka, npx - 1 + j, npy - 1 + i, ka, npx - i,
                    npy - 1 + j, False)
    elif kind == "dgrid":
        for j in rng:
            for i in rng:
                rec(ka, 1 - i, 1 - j, kb, 1 - j, i, True)
                rec(ka, 1 - i, npy + j, kb, 1 - j, npy - i, False)
                rec(ka, npx - 1 + i, 1 - j, kb, npx + j, i, False)
                rec(ka, npx - 1 + i, npy + j, kb, npx + j, npy - i, True)
        for j in rng:
            for i in rng:
                rec(kb, 1 - i, 1 - j, ka, j, 1 - i, True)
                rec(kb, 1 - i, npy - 1 + j, ka, j, npy + i, False)
                rec(kb, npx + i, 1 - j, ka, npx - j, 1 - i, False)
                rec(kb, npx + i, npy - 1 + j, ka, npx - j, npy + i, True)
    elif kind == "cgrid":
        for j in rng:
            for i in rng:
                rec(ka, 1 - i, 1 - j, kb, j, 1 - i, False)
                rec(ka, 1 - i, npy - 1 + j, kb, j, npy + i, True)
                rec(ka, npx + i, 1 - j, kb, npx - j, 1 - i, True)
                rec(ka, npx + i, npy - 1 + j, kb, npx - j, npy + i, False)
        for j in rng:
            for i in rng:
                rec(kb, 1 - i, 1 - j, ka, 1 - j, i, False)
                rec(kb, 1 - i, npy + j, ka, 1 - j, npy - i, True)
                rec(kb, npx - 1 + i, 1 - j, ka, npx + j, i, True)
                rec(kb, npx - 1 + i, npy + j, ka, npx + j, npy - i, False)
    elif kind == "agrid_pair":
        for j in rng:
            for i in rng:
                rec(ka, 1 - i, 1 - j, kb, 1 - j, i, True)
                rec(ka, 1 - i, npy - 1 + j, kb, 1 - j, npy - i, False)
                rec(ka, npx - 1 + i, 1 - j, kb, npx - 1 + j, i, False)
                rec(ka, npx - 1 + i, npy - 1 + j, kb, npx - 1 + j,
                    npy - i, True)
        for j in rng:
            for i in rng:
                rec(kb, 1 - j, 1 - i, ka, i, 1 - j, True)
                rec(kb, 1 - j, npy - 1 + i, ka, i, npy - 1 + j, False)
                rec(kb, npx - 1 + j, 1 - i, ka, npx - i, 1 - j, False)
                rec(kb, npx - 1 + j, npy - 1 + i, ka, npx - i,
                    npy - 1 + j, True)
    else:  # pragma: no cover - guarded by the public dispatchers
        raise ValueError(f"unknown corner-fill kind {kind!r}")
    return dst, src, pw


def _tile_corner_scatter(kind, lay, ka, kb, npx, ng, lo, name):
    """Replicate a single-face corner map onto all six faces.

    INDEPENDENCE (the batching licence for this group): every source is
    a SIDE-STRIP slot (halo x compute) and every destination is a
    CORNER-DIAGONAL slot (halo x halo), so the ng^2 writes of one corner
    never feed each other -- verified below by :func:`_check_batchable`,
    not asserted.  Across faces the map is face-local (each face reads
    and writes only itself) and face-independent (``fv_mp_mod.F90``
    operates on one tile's array with a tile-independent index map), so
    the six copies are trivially independent too.
    """
    d0, s0, p0 = _corner_records(kind, ka, kb, npx, ng, lo)
    dst, src, base, pw = [], [], [], []
    for face in range(6):
        for (kd, di, dj), (ks, si, sj), p in zip(d0, s0, p0):
            dst.append(lay.idx(kd, face, di, dj))
            src.append(lay.idx(ks, face, si, sj))
            base.append(1.0)
            pw.append(p)
    _check_batchable(name, dst, src)          # on the RAW record set
    dd, ss, bb, pp = _dedup_last(dst, src, base, pw)
    return _Scatter(dd, ss, bb, pp, name)


def _plain_scatter(name, d, s):
    """Unsigned batched copy; independence checked on the RAW record set.

    The check runs BEFORE :func:`_dedup_last` on purpose: dedup drops
    whole records, and checking afterwards would silently exempt the
    sources of the dropped ones.
    """
    _check_batchable(name, d, s)
    dd, ss = _dedup_last(d, s)
    return _Scatter(dd, ss, np.ones(dd.size), np.zeros(dd.size), name)


def _signed_scatter(name, d, s, g):
    """Batched copy carrying the record's own orientation sign."""
    _check_batchable(name, d, s)
    dd, ss, gg = _dedup_last(d, s, g)
    return _Scatter(dd, ss, gg, np.zeros(dd.size), name)


def _k2e_stencil(lay, k, stag, n, ng, k2e_nord, name):
    """fv3_native_gridstruct.k2e_remap_halo_rings:1412-1472 (cube_rmp).

    Every write reads the pre-remap snapshot ``src = f.copy()``, so the
    group is independent by construction; the record set, the ring
    classification and the two bounds guards are replayed verbatim.
    """
    tab = compute_fv3_native_k2e(n, remap_ng=ng, k2e_nord=k2e_nord)
    ij = tab[f"{stag}_ij"]
    loc = tab[f"{stag}_loc"]
    coef = tab[f"{stag}_coef"]
    npd = coef.shape[1] // 2 - 1
    lo_off = 1 - ng
    a0 = 1 - lo_off
    i_hi = n + 1 if stag in ("B", "CX", "DX") else n
    j_hi = n + 1 if stag in ("B", "CY", "DY") else n
    shp = lay.shapes[k]
    dst, src, w = [], [], []
    for (fi, fj), lv, cw in zip(ij, loc, coef):
        start = a0 + (int(lv) - npd - 1)
        i_ring = fi < 1 or fi > i_hi
        j_ring = fj < 1 or fj > j_hi
        if i_ring == j_ring:
            continue                     # corner-diagonal record classes
        col = int(fi) - lo_off
        row = int(fj) - lo_off
        if not (0 <= col < shp[0] and 0 <= row < shp[1]):
            continue
        if i_ring:                       # W/E ring: along-edge = j
            if not (0 <= start and start + len(cw) <= shp[1]):
                continue
            idx = [(col, start + t) for t in range(len(cw))]
        else:                            # S/N ring: along-edge = i
            if not (0 <= start and start + len(cw) <= shp[0]):
                continue
            idx = [(start + t, row) for t in range(len(cw))]
        dst.append((col, row))
        src.append(idx)
        w.append(np.asarray(cw, dtype=np.float64))
    nrec = len(dst)
    flat_d = np.empty(6 * nrec, dtype=np.int64)
    flat_s = np.empty((6 * nrec, coef.shape[1]), dtype=np.int64)
    flat_w = np.empty((6 * nrec, coef.shape[1]), dtype=np.float64)
    for face in range(6):
        for r in range(nrec):
            flat_d[face * nrec + r] = lay.idx(k, face, *dst[r])
            flat_w[face * nrec + r] = w[r]
            for t, (a, b) in enumerate(src[r]):
                flat_s[face * nrec + r, t] = lay.idx(k, face, a, b)
    st = _Stencil(flat_d, flat_s, flat_w, name)
    # destinations must be unique: `.at[].set()` has no defined winner
    if np.unique(st.dst).size != st.dst.size:
        raise ValueError(f"{name}: duplicate k2e destination records")
    return st


# corner-region Lagrange fill sequence, transcribed verbatim from
# _CornerLagrange.fill (fv3_native_ext_vector.py:345-395), which mirrors
# fv_duogrid.F90:1743-1901.  ``ie``/``je``/``is_``/``js_`` are supplied
# per stagger by the caller.
def _corner_lagrange_sequence(ie, je, is_, js_):
    w = _WEDGE
    dirs, diags = [], []
    # NE
    dirs += [(ie + 1, je + 2, "X+"), (ie + 1, je + 3, "X+"),
             (ie + 2, je + 3, "X+"), (ie + 2, je + 1, "Y+"),
             (ie + 3, je + 1, "Y+"), (ie + 3, je + 2, "Y+")]
    diags += [(ie + 1, je + 1, "X+", "Y+"), (ie + 3, je + 3, "X+", "Y+"),
              (ie + 2, je + 2, "X+", "Y+")]
    # NW
    dirs += [(is_ - 1, je + 2, "X-"), (is_ - 1, je + 3, "X-"),
             (is_ - 2, je + 3, "X-"), (is_ - 2, je + 1, "Y+"),
             (is_ - 3, je + 1, "Y+"), (is_ - 3, je + 2, "Y+")]
    diags += [(is_ - 1, je + 1, "X-", "Y+"), (is_ - 3, je + 3, "X-", "Y+"),
              (is_ - 2, je + 2, "X-", "Y+")]
    # SE
    dirs += [(ie + 1, js_ - 2, "X+"), (ie + 1, js_ - 3, "X+"),
             (ie + 2, js_ - 3, "X+"), (ie + 2, js_ - 1, "Y-"),
             (ie + 3, js_ - 1, "Y-"), (ie + 3, js_ - 2, "Y-")]
    diags += [(ie + 1, js_ - 1, "X+", "Y-"), (ie + 3, js_ - 3, "X+", "Y-"),
              (ie + 2, js_ - 2, "X+", "Y-")]
    # SW
    dirs += [(is_ - 1, js_ - 2, "X-"), (is_ - 1, js_ - 3, "X-"),
             (is_ - 2, js_ - 3, "X-"), (is_ - 2, js_ - 1, "Y-"),
             (is_ - 3, js_ - 1, "Y-"), (is_ - 3, js_ - 2, "Y-")]
    diags += [(is_ - 1, js_ - 1, "X-", "Y-"), (is_ - 3, js_ - 3, "X-", "Y-"),
              (is_ - 2, js_ - 2, "X-", "Y-")]
    assert len(dirs) == 4 * 2 * w and len(diags) == 4 * w
    return dirs, diags


def _corner_lagrange_tables(ops, lay, k, n, ng, name):
    """Weight/index tables for one ``_CornerLagrange`` operator family.

    ``ops`` is the per-face list of NumPy operators from
    ``build_ext_context``; the Lagrange weights are READ from them
    (``CornerLagrange.weights``) rather than re-derived -- re-deriving
    the coefficients would be a mathematical port of
    ``compute_lagrange_coeff`` (fv_duogrid.F90:2159-2272) and is
    forbidden by R1.
    """
    istag, jstag = ops[0].istag, ops[0].jstag
    ie, je = n + istag, n + jstag
    dirs, diags = _corner_lagrange_sequence(ie, je, 1, 1)
    lo = 1 - ng
    ndir, ndia = len(dirs), len(diags)
    order = _INTERP_ORDER + 1
    d_dst = np.empty(6 * ndir, dtype=np.int64)
    d_src = np.empty((6 * ndir, order), dtype=np.int64)
    d_w = np.empty((6 * ndir, order), dtype=np.float64)
    g_dst = np.empty(6 * ndia, dtype=np.int64)
    g_sx = np.empty((6 * ndia, order), dtype=np.int64)
    g_wx = np.empty((6 * ndia, order), dtype=np.float64)
    g_sy = np.empty((6 * ndia, order), dtype=np.int64)
    g_wy = np.empty((6 * ndia, order), dtype=np.float64)

    def one(op, face, i_t, j_t, direction):
        w, src_f = op.weights(i_t, j_t, direction)
        if len(w) != order:
            raise ValueError(
                f"{name}: Lagrange stencil width {len(w)} != {order}")
        if direction.startswith("X"):
            idx = [lay.idx(k, face, s - lo, j_t - lo) for s in src_f]
        else:
            idx = [lay.idx(k, face, i_t - lo, s - lo) for s in src_f]
        return np.asarray(w, dtype=np.float64), np.asarray(idx)

    for face in range(6):
        op = ops[face]
        if (op.istag, op.jstag) != (istag, jstag):
            raise ValueError(f"{name}: mixed staggers across faces")
        for r, (i_t, j_t, direction) in enumerate(dirs):
            w, idx = one(op, face, i_t, j_t, direction)
            row = face * ndir + r
            d_dst[row] = lay.idx(k, face, i_t - lo, j_t - lo)
            d_src[row] = idx
            d_w[row] = w
        for r, (i_t, j_t, d1, d2) in enumerate(diags):
            wx, ix = one(op, face, i_t, j_t, d1)
            wy, iy = one(op, face, i_t, j_t, d2)
            row = face * ndia + r
            g_dst[row] = lay.idx(k, face, i_t - lo, j_t - lo)
            g_sx[row], g_wx[row] = ix, wx
            g_sy[row], g_wy[row] = iy, wy

    st_dir = _Stencil(d_dst, d_src, d_w, f"{name}.directional")
    st_dia = _StencilPair(g_dst, g_sx, g_wx, g_sy, g_wy, f"{name}.diagonal")
    # Independence: every target is a wedge slot, every abscissa is a
    # compute-domain column/row.  Needs n >= _INTERP_ORDER + 1; the
    # check is mechanical, so a resolution that breaks it is loud.
    all_dst = np.concatenate([st_dir.dst, st_dia.dst])
    all_src = np.concatenate([st_dir.src.ravel(), st_dia.srcx.ravel(),
                              st_dia.srcy.ravel()])
    _check_batchable(name, all_dst, all_src)
    if np.unique(all_dst).size != all_dst.size:
        raise ValueError(f"{name}: duplicate corner-fill destination")
    return st_dir, st_dia


def _cgrid_edge_blend(lay, n, ng):
    """fv3_native_gridstruct.average_shared_edge_cgrid:1501-1535.

    BARRIER 1 (dyn_core.F90:853-901; ``mpp_get_boundary`` :872,
    ``gridtype=CGRID_NE`` :874).  Compute-ring slabs, origin (1, 1):
    fx (npx, n) x-faces, fy (n, npx) y-faces.
    """
    sg_npx = 2 * n + 1
    npx = n + 1
    dst, src, sgn = [], [], []

    def partner(tile, si, sj, along, n_src):
        sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
        if along == "i":
            sii2, sjj2 = neighbor_index(si + 2, sj, tile, n_src,
                                        sg_npx, sg_npx)
        else:
            sii2, sjj2 = neighbor_index(si, sj + 2, tile, n_src,
                                        sg_npx, sg_npx)
        dii, djj = sii2 - sii, sjj2 - sjj
        if (sii % 2 == 1) and (sjj % 2 == 0):        # x-face slot
            s = 1.0 if (dii if dii != 0 else djj) > 0 else -1.0
            fi, fj = (sii + 1) // 2, sjj // 2
            return lay.idx(0, n_src - 1, fi - 1, fj - 1), s
        s = 1.0 if (djj if djj != 0 else dii) > 0 else -1.0
        fi, fj = sii // 2, (sjj + 1) // 2
        return lay.idx(1, n_src - 1, fi - 1, fj - 1), s

    for tile in range(1, 7):
        nw, ne, ns, nn = neighbor_tiles(tile)
        for fj in range(1, n + 1):
            for fi, n_src in ((1, nw), (npx, ne)):
                p, s = partner(tile, 2 * fi - 1, 2 * fj, "i", n_src)
                dst.append(lay.idx(0, tile - 1, fi - 1, fj - 1))
                src.append(p)
                sgn.append(s)
        for fi in range(1, n + 1):
            for fj, n_src in ((1, ns), (npx, nn)):
                p, s = partner(tile, 2 * fi, 2 * fj - 1, "j", n_src)
                dst.append(lay.idx(1, tile - 1, fi - 1, fj - 1))
                src.append(p)
                sgn.append(s)
    bl = _Blend(dst, src, sgn, "average_shared_edge_cgrid")
    if np.unique(bl.dst).size != bl.dst.size:
        raise ValueError("average_shared_edge_cgrid: duplicate destination")
    return bl


def _bgrid_edge_blend(lay, n, ng, skip_endpoints):
    """fv3_native_gridstruct.average_shared_edge_bgrid:1580-1623.

    BARRIER 2 (dyn_core.F90:969-1011; ``mpp_get_boundary`` :984,
    ``gridtype=BGRID_NE`` :986; ``tempfx1 <- ubb``, ``tempfy1 <-
    vbbtemp``).  Compute-ring B arrays (npx, npx), origin (1, 1).
    """
    sg_npx = 2 * n + 1
    npx = n + 1
    dst, src, sgn = [], [], []

    def partner(tile, si, sj, along, n_src):
        sii, sjj = neighbor_index(si, sj, tile, n_src, sg_npx, sg_npx)
        if along == "i":
            sii2, sjj2 = neighbor_index(si + 2, sj, tile, n_src,
                                        sg_npx, sg_npx)
        else:
            sii2, sjj2 = neighbor_index(si, sj + 2, tile, n_src,
                                        sg_npx, sg_npx)
        dii, djj = sii2 - sii, sjj2 - sjj
        bi, bj = (sii + 1) // 2, (sjj + 1) // 2
        aligned = (dii != 0) if along == "i" else (djj != 0)
        if along == "i":
            k = 0 if aligned else 1
            d = dii if aligned else djj
        else:
            k = 1 if aligned else 0
            d = djj if aligned else dii
        s = 1.0 if d > 0 else -1.0
        return lay.idx(k, n_src - 1, bi - 1, bj - 1), s

    for tile in range(1, 7):
        nw, ne, ns, nn = neighbor_tiles(tile)
        for fi in range(1, npx + 1):
            if skip_endpoints and fi in (1, npx):
                continue
            for fj, n_src in ((1, ns), (npx, nn)):
                p, s = partner(tile, 2 * fi - 1, 2 * fj - 1, "j", n_src)
                dst.append(lay.idx(1, tile - 1, fi - 1, fj - 1))
                src.append(p)
                sgn.append(s)
        for fj in range(1, npx + 1):
            if skip_endpoints and fj in (1, npx):
                continue
            for fi, n_src in ((1, nw), (npx, ne)):
                p, s = partner(tile, 2 * fi - 1, 2 * fj - 1, "i", n_src)
                dst.append(lay.idx(0, tile - 1, fi - 1, fj - 1))
                src.append(p)
                sgn.append(s)
    bl = _Blend(dst, src, sgn, "average_shared_edge_bgrid")
    if np.unique(bl.dst).size != bl.dst.size:
        raise ValueError("average_shared_edge_bgrid: duplicate destination")
    return bl


def _d_strip_records(lay, n, ng, ngp):
    """fv3_native_ext_vector._write_d_strips:598-636 (rmp_s/n/w/e).

    Source buffers 2/3 = ud4/vd4 on the ng=4 projection lattice;
    destination buffers 0/1 = the stepper u/v.  Distinct buffers, so
    the group is independent; S/N then W/E write some slots twice and
    last-wins is preserved by :func:`_dedup_last`.
    """
    npx = n + 1
    lo = 1 - ng
    dst, src = [], []

    def u4(face, i_f, j_f):
        return lay.idx(2, face, i_f - 1 + ngp, j_f - 2 + ngp)

    def v4(face, i_f, j_f):
        return lay.idx(3, face, i_f - 2 + ngp, j_f - 1 + ngp)

    for face in range(6):
        for j_f in list(range(1 - ng, 0 + 1)):
            for i_f in range(1 - ng, n + ng + 1):
                dst.append(lay.idx(0, face, i_f - lo, j_f - lo))
                src.append(u4(face, i_f, j_f))
        for j_f in list(range(npx + 1, npx + ng + 1)):
            for i_f in range(1 - ng, n + ng + 1):
                dst.append(lay.idx(0, face, i_f - lo, j_f - lo))
                src.append(u4(face, i_f, j_f))
        for j_f in (list(range(1 - ng, 0 + 1))
                    + list(range(n + 1, n + ng + 1))):
            for i_f in range(1 - ng, npx + ng + 1):
                dst.append(lay.idx(1, face, i_f - lo, j_f - lo))
                src.append(v4(face, i_f, j_f))
        for i_f in (list(range(1 - ng, 0 + 1))
                    + list(range(n + 1, n + ng + 1))):
            for j_f in range(1 - ng, npx + ng + 1):
                dst.append(lay.idx(0, face, i_f - lo, j_f - lo))
                src.append(u4(face, i_f, j_f))
        for i_f in (list(range(1 - ng, 0 + 1))
                    + list(range(npx + 1, npx + ng + 1))):
            for j_f in range(1 - ng, n + ng + 1):
                dst.append(lay.idx(1, face, i_f - lo, j_f - lo))
                src.append(v4(face, i_f, j_f))
    return dst, src


def _c_strip_records(lay, n, ng, ngp):
    """fv3_native_ext_vector._write_c_strips:702-726."""
    npx = n + 1
    lo = 1 - ng
    dst, src = [], []

    def u4(face, i_f, j_f):
        return lay.idx(2, face, i_f - 2 + ngp, j_f - 1 + ngp)

    def v4(face, i_f, j_f):
        return lay.idx(3, face, i_f - 1 + ngp, j_f - 2 + ngp)

    for face in range(6):
        for j_f in (list(range(1 - ng, 0 + 1))
                    + list(range(n + 1, n + ng + 1))):
            for i_f in range(1 - ng, npx + ng + 1):
                dst.append(lay.idx(0, face, i_f - lo, j_f - lo))
                src.append(u4(face, i_f, j_f))
        for j_f in (list(range(1 - ng, 0 + 1))
                    + list(range(npx + 1, npx + ng + 1))):
            for i_f in range(1 - ng, n + ng + 1):
                dst.append(lay.idx(1, face, i_f - lo, j_f - lo))
                src.append(v4(face, i_f, j_f))
        for i_f in (list(range(1 - ng, 0 + 1))
                    + list(range(npx + 1, npx + ng + 1))):
            for j_f in range(1 - ng, n + ng + 1):
                dst.append(lay.idx(0, face, i_f - lo, j_f - lo))
                src.append(u4(face, i_f, j_f))
        for i_f in (list(range(1 - ng, 0 + 1))
                    + list(range(n + 1, n + ng + 1))):
            for j_f in range(1 - ng, npx + ng + 1):
                dst.append(lay.idx(1, face, i_f - lo, j_f - lo))
                src.append(v4(face, i_f, j_f))
    return dst, src


# ---------------------------------------------------------------------------
# the table bundle
# ---------------------------------------------------------------------------

class DuoHaloTables:
    """Static index/weight tables + grid metrics for the JAX duo lane.

    Identity-hashable ON PURPOSE: it is passed as a **static** jit
    argument so its numpy index tables become jaxpr constants and can
    index under jit (the task's "keep integer index tables STATIC"
    requirement).  Consequence: two structurally identical bundles
    compile twice, and the metric arrays are baked into each executable.
    Build one per resolution and reuse it.

    Nothing in here is differentiated: grid metrics and index tables are
    constants, never traced arguments.
    """

    __slots__ = (
        "n", "ng", "ngp", "npx", "nq", "k2e_nord", "vector_corner",
        "skip_b_endpoints", "allflux_slots",
        "lay_a", "lay_b", "lay_c", "lay_d", "lay_a4", "lay_geo",
        "lay_fx", "lay_bb", "lay_wd", "lay_wc", "lay_ap",
        "ex_a_strip", "ex_a_corner", "ex_a4_strip", "ex_a4_corner",
        "ex_b_strip", "ex_c_strip", "ex_c_corner",
        "ex_d_strip", "ex_d_corner",
        "k2e_a", "k2e_a4", "k2e_b",
        "corner", "avg_c", "avg_b", "wr_d", "wr_c",
        "fc_bgrid_x", "fc_agrid_x", "fc_agrid_y",
        "fc_dgrid", "fc_cgrid", "fc_agrid_pair",
        "amat", "dx", "dy", "vlon4", "vlat4", "ew4", "es4",
        "c2l_s", "c2l_e",
    )

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __repr__(self):                              # pragma: no cover
        return (f"DuoHaloTables(n={self.n}, ng={self.ng}, ngp={self.ngp}, "
                f"nq={self.nq}, k2e_nord={self.k2e_nord}, "
                f"vector_corner={self.vector_corner!r})")


def build_jax_duo_halo_tables(ectx: dict, gs6: list | None = None, *,
                              nq: int = 0,
                              skip_b_endpoints: bool = False
                              ) -> DuoHaloTables:
    """Convert the NumPy setup (``build_ext_context``) into JAX tables.

    ``ectx`` is the dict from
    ``legoesm.grids.fv3_native_ext_vector.build_ext_context``; ``gs6``,
    when given, is the matching list of kinked gridstructs and is used
    ONLY to cross-check that the context describes the same grid (a
    context built at another resolution would otherwise produce a
    silently wrong lane).  ``nq`` is the tracer count of BARRIER 1's
    ``do iq=1,4+nq`` loop (dyn_core.F90:855).

    ``skip_b_endpoints`` mirrors the NumPy lane's
    ``LEGOESM_DUO_AVG_B_ENDPOINTS=local`` diagnostic.  The JAX lane does
    NOT read that environment variable (a production lane whose numerics
    depend on the environment is a silent-divergence trap); pass the
    flag explicitly when reproducing a diagnostic NumPy run.
    """
    if os.environ.get("LEGOESM_DUO_CORNER_MODE", "") == "nearest":
        raise ValueError(
            "LEGOESM_DUO_CORNER_MODE=nearest is set: the NumPy lane's "
            "_CornerLagrange.fill then applies a NON-FAITHFUL "
            "nearest-corner overwrite that this JAX lane does not "
            "implement.  Unset it, or the two lanes silently disagree in "
            "the corner wedges.")
    n = int(ectx["n"])
    ng = int(ectx["ng"])
    # the geographic lattice width is DERIVED from the context's own
    # bases (upstream set_bd_ext_duo pins dg%bd%ng = 4,
    # fv_duogrid.F90:146) rather than re-declared here
    ngp = (np.asarray(ectx["vlon4"]).shape[1] - n) // 2
    npx = n + 1
    if n < _INTERP_ORDER + 1:
        raise ValueError(
            f"n={n}: the corner-region Lagrange fill reads abscissae "
            f"i in [1, {_INTERP_ORDER + 1}] and i in [ie-{_INTERP_ORDER}, "
            f"ie]; below n={_INTERP_ORDER + 1} those windows overlap the "
            f"opposite wedge and the fill is no longer a single "
            f"independent group")
    if gs6 is not None:
        for t, gs in enumerate(gs6):
            if int(gs["n"]) != n or int(gs["ng"]) != ng:
                raise ValueError(
                    f"gs6[{t}] is (n={gs['n']}, ng={gs['ng']}) but the ext "
                    f"context is (n={n}, ng={ng})")
            if not np.array_equal(np.asarray(gs["dx"]),
                                  np.asarray(ectx["dx6"][t])):
                raise ValueError(
                    f"gs6[{t}]['dx'] differs from the context's dx6 -- the "
                    f"context was built from a different gridstruct")

    ma = n + 2 * ng               # cell axis
    mb = n + 2 * ng + 1           # node axis
    m4 = n + 2 * ngp              # geographic-lattice cell axis
    tab = DuoHaloTables()
    tab.n, tab.ng, tab.ngp, tab.npx, tab.nq = n, ng, ngp, npx, int(nq)
    tab.k2e_nord = int(ectx.get("k2e_nord", 4))
    tab.vector_corner = ectx.get("vector_corner", "lagrange")
    tab.skip_b_endpoints = bool(skip_b_endpoints)

    # --- layouts ---------------------------------------------------------
    tab.lay_a = _FlatLayout((ma, ma))                       # A scalar
    tab.lay_b = _FlatLayout((mb, mb))                       # B scalar
    tab.lay_c = _FlatLayout((mb, ma), (ma, mb))             # uc, vc
    tab.lay_d = _FlatLayout((ma, mb), (mb, ma))             # u, v
    tab.lay_a4 = _FlatLayout((m4, m4))                      # geo lattice
    tab.lay_fx = _FlatLayout((npx, n), (n, npx))            # barrier-1 slabs
    tab.lay_bb = _FlatLayout((npx, npx), (npx, npx))        # barrier-2 slabs
    tab.lay_wd = _FlatLayout((ma, mb), (mb, ma),
                             (m4, m4 - 1), (m4 - 1, m4))    # u,v,ud4,vd4
    tab.lay_wc = _FlatLayout((mb, ma), (ma, mb),
                             (m4 - 1, m4), (m4, m4 - 1))    # uc,vc,uc4,vc4
    tab.lay_geo = tab.lay_a4

    # --- A/B scalar exchanges (strips, then corner fill) -----------------
    # INDEPENDENCE (strips, all four exchanges): every source read is
    # guarded to the neighbour's COMPUTE domain and every destination is
    # a halo slot, so no strip write feeds another -- which is what makes
    # the NumPy lane's `for tile in 1..6` loop flattenable into ONE
    # scatter.  _check_batchable recomputes the two index sets and
    # raises if they intersect; it is the licence, not the comment.
    # PHASE ORDER: the corner fill must still run AFTER the strips (it
    # reads them), and hoisting all six faces' strips ahead of all six
    # corner fills is legal only because no corner destination is a
    # strip source -- that is _check_no_clobber, one per exchange.
    d, s = _agrid_strip_records(tab.lay_a, 0, n, ng)
    tab.ex_a_strip = _plain_scatter("exchange_agrid_scalar_halos.strips",
                                    d, s)
    tab.ex_a_corner = _tile_corner_scatter(
        "agrid_x", tab.lay_a, 0, 0, npx, ng, 1 - ng,
        "exchange_agrid_scalar_halos.corners")
    _check_no_clobber("exchange_agrid_scalar_halos",
                      tab.ex_a_corner.dst, tab.ex_a_strip.src)

    d4, s4 = _agrid_strip_records(tab.lay_a4, 0, n, ngp)
    tab.ex_a4_strip = _plain_scatter("geo_lattice_exchange.strips", d4, s4)
    tab.ex_a4_corner = _tile_corner_scatter(
        "agrid_x", tab.lay_a4, 0, 0, npx, ngp, 1 - ngp,
        "geo_lattice_exchange.corners")
    _check_no_clobber("geo_lattice_exchange",
                      tab.ex_a4_corner.dst, tab.ex_a4_strip.src)

    # B scalar has NO corner fill (the NumPy lane leaves corner-diagonal
    # regions untouched), so there is no phase-order question here.
    d, s = _bgrid_strip_records(tab.lay_b, 0, n, ng)
    tab.ex_b_strip = _plain_scatter("exchange_bgrid_scalar_halos", d, s)

    # --- C/D vector exchanges (strips, then the VECTOR corner fill) ------
    # Same independence argument, over the CONCATENATED [uc|vc] / [u|v]
    # buffer: a strip slot of one component may be sourced from the other
    # component of a neighbouring face (the mpp NE-vector component
    # swap), so the check has to span both arrays at once -- which is
    # exactly why the buffer is one flat layout rather than two.
    d, s, g = _cgrid_strip_records(tab.lay_c, n, ng)
    tab.ex_c_strip = _signed_scatter("exchange_cgrid_vector_halos.strips",
                                     d, s, g)
    tab.ex_c_corner = _tile_corner_scatter(
        "cgrid", tab.lay_c, 0, 1, npx, ng, 1 - ng,
        "exchange_cgrid_vector_halos.corners")
    _check_no_clobber("exchange_cgrid_vector_halos",
                      tab.ex_c_corner.dst, tab.ex_c_strip.src)

    d, s, g = _dgrid_strip_records(tab.lay_d, n, ng)
    tab.ex_d_strip = _signed_scatter("exchange_dgrid_vector_halos.strips",
                                     d, s, g)
    tab.ex_d_corner = _tile_corner_scatter(
        "dgrid", tab.lay_d, 0, 1, npx, ng, 1 - ng,
        "exchange_dgrid_vector_halos.corners")
    _check_no_clobber("exchange_dgrid_vector_halos",
                      tab.ex_d_corner.dst, tab.ex_d_strip.src)

    # --- k2e ring remaps --------------------------------------------------
    # INDEPENDENCE: the NumPy lane takes an explicit snapshot
    # (`src = f.copy()`) and every ring write reads only the snapshot, so
    # this group is independent BY CONSTRUCTION -- the JAX twin keeps the
    # same shape (scatter into a fresh array, gather from the incoming
    # one).  Destination uniqueness is asserted inside _k2e_stencil.
    tab.k2e_a = _k2e_stencil(tab.lay_a, 0, "A", n, ng, tab.k2e_nord,
                             "k2e_remap_halo_rings[A]")
    tab.k2e_a4 = _k2e_stencil(tab.lay_a4, 0, "A", n, ngp, tab.k2e_nord,
                              "k2e_remap_halo_rings[A,geo]")
    tab.k2e_b = _k2e_stencil(tab.lay_b, 0, "B", n, ng, tab.k2e_nord,
                             "k2e_remap_halo_rings[B]")

    # --- corner-region Lagrange operators ---------------------------------
    # INDEPENDENCE: `diag` in the NumPy lane copies f, applies ONE
    # directional fill to the copy and reads back only the target slot,
    # so it is 0.5*(wsum_dir1 + wsum_dir2) on the CURRENT field -- the
    # nine slots per corner do NOT feed each other.  All 36 targets are
    # wedge slots and all abscissae are compute-domain columns/rows, so
    # the whole fill is one group; that needs n >= _INTERP_ORDER+1 and is
    # checked (twice: the n guard above and _check_batchable inside).
    # The sequence is NOT composed into one sparse matrix -- that would
    # be a mathematical re-derivation, forbidden by R1.
    # Each operator family is addressed as a SINGLE-array layout at its
    # own stagger shape: a3/a4 cells, b3 nodes, du3 the u-shaped (0,1)
    # array, dv3 the v-shaped (1,0) array (the fill is per component,
    # fv3_native_ext_vector.py:693-695).
    tab.corner = {}
    for key, shp, ngk in (("a4", (m4, m4), ngp),
                          ("a3", (ma, ma), ng),
                          ("b3", (mb, mb), ng),
                          ("du3", (ma, mb), ng),
                          ("dv3", (mb, ma), ng)):
        tab.corner[key] = _corner_lagrange_tables(
            ectx[f"corner_{key}"], _FlatLayout(shp), 0, n, ngk,
            f"corner_lagrange[{key}]")

    # --- the two duo barriers ---------------------------------------------
    # INDEPENDENCE: the NumPy lane is already explicitly two-phase
    # (gather every partner value, then apply), mirroring
    # mpp_get_boundary's buffering.  `flat.at[dst].set(0.5*(flat[dst] +
    # sign*flat[src]))` reads BOTH operands from the pre-update buffer,
    # which is the same thing; destination uniqueness is asserted in the
    # builders.
    tab.avg_c = _cgrid_edge_blend(tab.lay_fx, n, ng)
    tab.avg_b = _bgrid_edge_blend(tab.lay_bb, n, ng, skip_b_endpoints)
    # dyn_core.F90:856 -- `if (iq==1 .or. iq==4 .or. iq>4)`.  Slots 2 (w)
    # and 3 (q_con) are deliberately NOT averaged; the exclusion is
    # STRUCTURAL here (they never enter the selected list) rather than a
    # runtime guard that could be edited away silently.
    tab.allflux_slots = np.asarray(
        [iq - 1 for iq in range(1, 4 + int(nq) + 1)
         if iq == 1 or iq >= 4], dtype=np.int32)

    # --- ext strip write-back ---------------------------------------------
    # INDEPENDENCE: sources live in the ng=4 PROJECTION buffers (slots
    # 2/3 of the layout) and destinations in the stepper wind arrays
    # (slots 0/1) -- different arrays, so nothing feeds anything.  The
    # S/N pass and the W/E pass DO write some slots twice; last-wins is
    # preserved by _dedup_last, which is required because `.at[].set()`
    # with duplicate indices has no defined winner in JAX.
    d, s = _d_strip_records(tab.lay_wd, n, ng, ngp)
    tab.wr_d = _plain_scatter("write_d_strips", d, s)
    d, s = _c_strip_records(tab.lay_wc, n, ng, ngp)
    tab.wr_c = _plain_scatter("write_c_strips", d, s)

    # --- standalone fill_corners twins ------------------------------------
    tab.fc_bgrid_x = _tile_corner_scatter(
        "bgrid_x", tab.lay_b, 0, 0, npx, ng, 1 - ng, "fill_corners_bgrid_x")
    tab.fc_agrid_x = _tile_corner_scatter(
        "agrid_x", tab.lay_a, 0, 0, npx, ng, 1 - ng, "fill_corners_agrid_x")
    tab.fc_agrid_y = _tile_corner_scatter(
        "agrid_y", tab.lay_a, 0, 0, npx, ng, 1 - ng, "fill_corners_agrid_y")
    tab.fc_dgrid = _tile_corner_scatter(
        "dgrid", tab.lay_d, 0, 1, npx, ng, 1 - ng, "fill_corners_dgrid")
    tab.fc_cgrid = _tile_corner_scatter(
        "cgrid", tab.lay_c, 0, 1, npx, ng, 1 - ng, "fill_corners_cgrid")
    tab.lay_ap = _FlatLayout((ma, ma), (ma, ma))
    tab.fc_agrid_pair = _tile_corner_scatter(
        "agrid_pair", tab.lay_ap, 0, 1, npx, ng, 1 - ng,
        "fill_corners_agrid_pair")

    # --- grid metrics (constants; never traced, never differentiated) -----
    tab.amat = np.stack([np.stack(a, axis=0) for a in ectx["amat6"]], axis=0)
    tab.dx = np.stack([np.asarray(a) for a in ectx["dx6"]], axis=0)
    tab.dy = np.stack([np.asarray(a) for a in ectx["dy6"]], axis=0)
    tab.vlon4 = np.asarray(ectx["vlon4"])
    tab.vlat4 = np.asarray(ectx["vlat4"])
    tab.ew4 = np.asarray(ectx["ew4"])
    tab.es4 = np.asarray(ectx["es4"])
    # c2l window: Fortran is-1 .. ie+1 in numpy coordinates
    tab.c2l_s = (1 - 1) - (1 - ng)
    tab.c2l_e = (n + 1) - (1 - ng)
    return tab


# ---------------------------------------------------------------------------
# public kernels -- functional twins
# ---------------------------------------------------------------------------

def _gate(fname, **arrays):
    _require_f64_jax(fname, arrays)


def exchange_agrid_scalar_halos(f6, tab: DuoHaloTables, ring: str = "stepper"):
    """JAX twin of ``fv3_native_gridstruct.exchange_agrid_scalar_halos``.

    ``f6``: ``(6, m, m)`` A-grid (cell-centre) fields, Fortran ``1-ng``
    at numpy 0.  ``ring`` selects the halo width: ``"stepper"`` (the
    field's own ``ng``) or ``"geo"`` (the ng=4 duo lattice,
    ``dg%bd%ng``).  Returns the new ``(6, m, m)`` stack; strips first,
    then the AGRID corner fill, exactly as the NumPy lane's per-tile
    call does.
    """
    _gate("exchange_agrid_scalar_halos", f6=f6)
    if ring == "stepper":
        lay, strip, corner = tab.lay_a, tab.ex_a_strip, tab.ex_a_corner
    elif ring == "geo":
        lay, strip, corner = tab.lay_a4, tab.ex_a4_strip, tab.ex_a4_corner
    else:
        raise ValueError(
            f"exchange_agrid_scalar_halos: ring={ring!r} (expected "
            f"'stepper' or 'geo')")
    flat = jnp.asarray(f6).reshape(-1)
    flat = _apply_scatter(flat, strip)
    flat = _apply_scatter(flat, corner)
    return _unpack(lay, flat)[0]


def exchange_bgrid_scalar_halos(f6, tab: DuoHaloTables):
    """JAX twin of ``exchange_bgrid_scalar_halos`` (CORNER scalar).

    ``f6``: ``(6, m+1, m+1)`` B-node fields.  Corner-diagonal regions
    are left untouched, like the NumPy lane.  Returns the new stack.
    """
    _gate("exchange_bgrid_scalar_halos", f6=f6)
    flat = _apply_scatter(jnp.asarray(f6).reshape(-1), tab.ex_b_strip)
    return _unpack(tab.lay_b, flat)[0]


def exchange_cgrid_vector_halos(uc6, vc6, tab: DuoHaloTables):
    """JAX twin of ``exchange_cgrid_vector_halos`` (CGRID_NE).

    ``uc6`` ``(6, m+1, m)`` x-faces, ``vc6`` ``(6, m, m+1)`` y-faces.
    Returns ``(uc6, vc6)`` -- strips (with the mpp component/orientation
    signs) then the VECTOR corner fill with ``mySign = -1``.
    """
    _gate("exchange_cgrid_vector_halos", uc6=uc6, vc6=vc6)
    flat = _apply_scatter(_pack(uc6, vc6), tab.ex_c_strip)
    flat = _apply_scatter(flat, tab.ex_c_corner, sign=-1.0)
    return _unpack(tab.lay_c, flat)


def exchange_dgrid_vector_halos(u6, v6, tab: DuoHaloTables):
    """JAX twin of ``exchange_dgrid_vector_halos`` (DGRID_NE).

    ``u6`` ``(6, m, m+1)``, ``v6`` ``(6, m+1, m)``.  Returns ``(u6, v6)``.
    """
    _gate("exchange_dgrid_vector_halos", u6=u6, v6=v6)
    flat = _apply_scatter(_pack(u6, v6), tab.ex_d_strip)
    flat = _apply_scatter(flat, tab.ex_d_corner, sign=-1.0)
    return _unpack(tab.lay_d, flat)


def k2e_remap_halo_rings(f6, tab: DuoHaloTables, stag: str,
                         ring: str = "stepper"):
    """JAX twin of ``k2e_remap_halo_rings`` (``cube_rmp`` semantics).

    ``stag`` is ``"A"`` or ``"B"``; the C/D table families exist in the
    NumPy lane for the REJECTED per-stagger position-only remap and are
    not part of the duo ext flow, so they raise here rather than
    silently selecting a different interpolation.  Corner-diagonal cells
    are untouched.  Returns the new stack.
    """
    _gate("k2e_remap_halo_rings", f6=f6)
    if stag == "A" and ring == "stepper":
        lay, st = tab.lay_a, tab.k2e_a
    elif stag == "A" and ring == "geo":
        lay, st = tab.lay_a4, tab.k2e_a4
    elif stag == "B" and ring == "stepper":
        lay, st = tab.lay_b, tab.k2e_b
    elif stag in ("CX", "CY", "DX", "DY"):
        raise ValueError(
            f"k2e_remap_halo_rings: stagger {stag!r} is not part of the "
            f"duo ext flow (upstream ext_scalar supports (0,0) and (1,1) "
            f"only; the C/D table families belong to the rejected "
            f"position-only vector remap)")
    else:
        raise ValueError(
            f"k2e_remap_halo_rings: stagger {stag!r} / ring {ring!r} "
            f"unsupported")
    flat = jnp.asarray(f6).reshape(-1)
    out = _apply_stencil(flat, flat, st)     # reads the pre-remap snapshot
    return _unpack(lay, out)[0]


def corner_lagrange_fill(f6, tab: DuoHaloTables, key: str):
    """JAX twin of ``_CornerLagrange.fill`` (fv_duogrid.F90:1719-1903).

    ``key``: ``"a4"`` (geo lattice), ``"a3"``/``"b3"`` (scalar A/B at
    the stepper halo), ``"du3"``/``"dv3"`` (the (0,1)/(1,0) staggered
    covariant components).  Two scatters -- the 24 directional targets
    then the 12 averaged diagonal targets -- both reading the SAME
    incoming field, which is what the NumPy lane computes (``diag``
    applies its directional fills to COPIES and reads back only the
    target slot).  Returns the new stack.
    """
    _gate("corner_lagrange_fill", f6=f6)
    if key not in tab.corner:
        raise ValueError(
            f"corner_lagrange_fill: key {key!r} (expected one of "
            f"{sorted(tab.corner)})")
    st_dir, st_dia = tab.corner[key]
    f6 = jnp.asarray(f6)
    shape = f6.shape
    flat = f6.reshape(-1)
    out = _apply_stencil(flat, flat, st_dir)
    out = _apply_stencil_pair(out, flat, st_dia)
    return out.reshape(shape)


def fill_corners_bgrid_x(q6, tab: DuoHaloTables):
    """JAX twin of ``_fill_corners_bgrid_x`` (fv_mp_mod.F90:1032, XDir/BGRID)."""
    _gate("fill_corners_bgrid_x", q6=q6)
    flat = _apply_scatter(jnp.asarray(q6).reshape(-1), tab.fc_bgrid_x)
    return flat.reshape(q6.shape)


def fill_corners_agrid_x(q6, tab: DuoHaloTables):
    """JAX twin of ``_fill_corners_agrid_x`` (fv_mp_mod.F90:1032, XDir/AGRID)."""
    _gate("fill_corners_agrid_x", q6=q6)
    flat = _apply_scatter(jnp.asarray(q6).reshape(-1), tab.fc_agrid_x)
    return flat.reshape(q6.shape)


def fill_corners_agrid_y(q6, tab: DuoHaloTables):
    """JAX twin of ``_fill_corners_agrid_y`` (fv_mp_mod.F90:1032, YDir/AGRID)."""
    _gate("fill_corners_agrid_y", q6=q6)
    flat = _apply_scatter(jnp.asarray(q6).reshape(-1), tab.fc_agrid_y)
    return flat.reshape(q6.shape)


def fill_corners_dgrid(x6, y6, tab: DuoHaloTables, sign: float = 1.0):
    """JAX twin of ``_fill_corners_dgrid`` (fv_mp_mod.F90:1257, mySign).

    ``sign`` is a STATIC python float (the caller's ``mySign``); the
    table records which of the four statements per corner carry it.
    Returns ``(x6, y6)``.
    """
    _gate("fill_corners_dgrid", x6=x6, y6=y6)
    flat = _apply_scatter(_pack(x6, y6), tab.fc_dgrid, sign=sign)
    return _unpack(tab.lay_d, flat)


def fill_corners_cgrid(x6, y6, tab: DuoHaloTables, sign: float = 1.0):
    """JAX twin of ``_fill_corners_cgrid`` (fv_mp_mod.F90:1369). -> (x6, y6)."""
    _gate("fill_corners_cgrid", x6=x6, y6=y6)
    flat = _apply_scatter(_pack(x6, y6), tab.fc_cgrid, sign=sign)
    return _unpack(tab.lay_c, flat)


def fill_corners_agrid_pair(x6, y6, tab: DuoHaloTables, sign: float = 1.0):
    """JAX twin of ``_fill_corners_agrid_pair`` (fv_mp_mod.F90:1433).

    -> ``(x6, y6)``; both ``(6, m, m)`` cell arrays.
    """
    _gate("fill_corners_agrid_pair", x6=x6, y6=y6)
    flat = _apply_scatter(_pack(x6, y6), tab.fc_agrid_pair, sign=sign)
    return _unpack(tab.lay_ap, flat)


# ---------------------------------------------------------------------------
# the two duo barriers
# ---------------------------------------------------------------------------

def average_shared_edge_cgrid(fx6, fy6, tab: DuoHaloTables):
    """BARRIER 1 -- dyn_core.F90:853-901, ``mpp_get_boundary`` :872 with
    ``gridtype=CGRID_NE`` :874, then the ``0.5*(mine + neighbour)`` blend.

    ``fx6`` ``(6, npx, n)`` x-face fluxes, ``fy6`` ``(6, n, npx)``
    y-face fluxes, compute ring only, origin (is, js) = (1, 1).  Both
    operands of the blend are read from the pre-update buffer, which is
    what ``mpp_get_boundary``'s buffering (and the NumPy lane's
    gather-then-apply) does.  Returns ``(fx6, fy6)``.
    """
    _gate("average_shared_edge_cgrid", fx6=fx6, fy6=fy6)
    flat = _apply_blend(_pack(fx6, fy6), tab.avg_c)
    return _unpack(tab.lay_fx, flat)


def average_allflux_shared_edges(afx6, afy6, tab: DuoHaloTables):
    """BARRIER 1 over the allflux stacks -- dyn_core.F90:855-856.

    ``afx6`` ``(6, npx, n, 4+nq)``, ``afy6`` ``(6, n, npx, 4+nq)``.
    Fortran averages ONLY ``iq==1`` (delp), ``iq==4`` (temp) and
    ``iq>4`` (tracers): slots 2 (``w``) and 3 (``q_con``) are NOT
    averaged and come back byte-identical.  Returns ``(afx6, afy6)``.
    """
    _gate("average_allflux_shared_edges", afx6=afx6, afy6=afy6)
    afx6 = jnp.asarray(afx6)
    afy6 = jnp.asarray(afy6)
    nslot = 4 + tab.nq
    if afx6.shape[-1] != nslot or afy6.shape[-1] != nslot:
        raise ValueError(
            f"average_allflux_shared_edges: slot axis "
            f"{afx6.shape[-1]}/{afy6.shape[-1]} != 4+nq = {nslot} (the "
            f"table was built with nq={tab.nq})")
    sel = tab.allflux_slots
    # slot axis to the front, then one blend broadcast over the selected
    # slots -- the per-slot 2-D layout is exactly the barrier's own
    x = jnp.moveaxis(afx6[..., sel], -1, 0).reshape(sel.size, -1)
    y = jnp.moveaxis(afy6[..., sel], -1, 0).reshape(sel.size, -1)
    flat = jnp.concatenate([x, y], axis=1)
    bl = tab.avg_c
    flat = flat.at[:, bl.dst].set(
        0.5 * (flat[:, bl.dst] + bl.sign * flat[:, bl.src]))
    nx = x.shape[1]
    xs = jnp.moveaxis(flat[:, :nx].reshape((sel.size,) + afx6.shape[:-1]),
                      0, -1)
    ys = jnp.moveaxis(flat[:, nx:].reshape((sel.size,) + afy6.shape[:-1]),
                      0, -1)
    return afx6.at[..., sel].set(xs), afy6.at[..., sel].set(ys)


def average_shared_edge_bgrid(xb6, yb6, tab: DuoHaloTables):
    """BARRIER 2 -- dyn_core.F90:969-1011, ``mpp_get_boundary`` :984 with
    ``gridtype=BGRID_NE`` :986.

    ``xb6``/``yb6`` ``(6, npx, npx)`` compute-ring B arrays: the oracle
    loads ``tempfx1 <- ubb`` and ``tempfy1 <- vbbtemp`` (:971-981), so
    ``xb6`` is the x-like member and ``yb6`` the y-like one.  Returns
    ``(xb6, yb6)``.
    """
    _gate("average_shared_edge_bgrid", xb6=xb6, yb6=yb6)
    flat = _apply_blend(_pack(xb6, yb6), tab.avg_b)
    return _unpack(tab.lay_bb, flat)


# ---------------------------------------------------------------------------
# c2l / projections / packing
# ---------------------------------------------------------------------------

def c2l_ord2_face(u6, v6, tab: DuoHaloTables):
    """JAX twin of ``fv3_native_ext_vector.c2l_ord2_face``.

    fv_grid_utils.F90:2547-2628 with ``do_halo=.true.`` (one ring).
    ``u6`` ``(6, m_a, m_b)`` D-grid x-wind on y-faces, ``v6``
    ``(6, m_b, m_a)``.  Returns geographic ``(ua6, va6)``
    ``(6, m_a, m_a)``, valid on Fortran ``is-1..ie+1``; every other slot
    is NaN, exactly like the NumPy lane (a plausible-looking 0.0 there
    would pass every downstream finite check).
    """
    _gate("c2l_ord2_face", u6=u6, v6=v6)
    s, e = tab.c2l_s, tab.c2l_e
    isl = slice(s, e + 1)
    dx, dy = tab.dx, tab.dy
    a11, a12, a21, a22 = (tab.amat[:, 0], tab.amat[:, 1],
                          tab.amat[:, 2], tab.amat[:, 3])
    wu_lo = u6[:, isl, s:e + 1] * dx[:, isl, s:e + 1]
    wu_hi = u6[:, isl, s + 1:e + 2] * dx[:, isl, s + 1:e + 2]
    u1 = 2.0 * (wu_lo + wu_hi) / (dx[:, isl, s:e + 1]
                                  + dx[:, isl, s + 1:e + 2])
    wv_lo = v6[:, s:e + 1, isl] * dy[:, s:e + 1, isl]
    wv_hi = v6[:, s + 1:e + 2, isl] * dy[:, s + 1:e + 2, isl]
    v1 = 2.0 * (wv_lo + wv_hi) / (dy[:, s:e + 1, isl]
                                  + dy[:, s + 1:e + 2, isl])
    m = tab.n + 2 * tab.ng
    nan = jnp.full((6, m, m), jnp.nan, dtype=u6.dtype)
    ua = nan.at[:, isl, isl].set(a11[:, isl, isl] * u1 + a12[:, isl, isl] * v1)
    va = nan.at[:, isl, isl].set(a21[:, isl, isl] * u1 + a22[:, isl, isl] * v1)
    return ua, va


def c2l_ord2_cgrid_face(uc6, vc6, tab: DuoHaloTables):
    """JAX twin of ``c2l_ord2_cgrid_face`` (fv_duogrid.F90:2765-2844).

    ``uc6`` ``(6, m_b, m_a)`` C-grid x-wind on x-faces, ``vc6``
    ``(6, m_a, m_b)``.  Note the metric swap vs the D variant (``dy``
    weights the x-faces).  Returns ``(ua6, va6)``.
    """
    _gate("c2l_ord2_cgrid_face", uc6=uc6, vc6=vc6)
    s, e = tab.c2l_s, tab.c2l_e
    isl = slice(s, e + 1)
    dx, dy = tab.dx, tab.dy
    a11, a12, a21, a22 = (tab.amat[:, 0], tab.amat[:, 1],
                          tab.amat[:, 2], tab.amat[:, 3])
    wu_lo = uc6[:, s:e + 1, isl] * dy[:, s:e + 1, isl]
    wu_hi = uc6[:, s + 1:e + 2, isl] * dy[:, s + 1:e + 2, isl]
    u1 = 2.0 * (wu_lo + wu_hi) / (dy[:, s:e + 1, isl]
                                  + dy[:, s + 1:e + 2, isl])
    wv_lo = vc6[:, isl, s:e + 1] * dx[:, isl, s:e + 1]
    wv_hi = vc6[:, isl, s + 1:e + 2] * dx[:, isl, s + 1:e + 2]
    v1 = 2.0 * (wv_lo + wv_hi) / (dx[:, isl, s:e + 1]
                                  + dx[:, isl, s + 1:e + 2])
    m = tab.n + 2 * tab.ng
    nan = jnp.full((6, m, m), jnp.nan, dtype=uc6.dtype)
    ua = nan.at[:, isl, isl].set(a11[:, isl, isl] * u1 + a12[:, isl, isl] * v1)
    va = nan.at[:, isl, isl].set(a21[:, isl, isl] * u1 + a22[:, isl, isl] * v1)
    return ua, va


def pack_p1(x6, tab: DuoHaloTables):
    """JAX twin of ``_pack_p1``: embed the stepper lattice in the ng=4 one.

    Slots outside the embedded block stay NaN (the NumPy lane's
    ``np.full(..., np.nan)``).  Returns ``(6, m4, m4)``.
    """
    _gate("pack_p1", x6=x6)
    m4 = tab.n + 2 * tab.ngp
    ma = tab.n + 2 * tab.ng
    d = tab.ngp - tab.ng
    out = jnp.full((6, m4, m4), jnp.nan, dtype=x6.dtype)
    return out.at[:, d:d + ma, d:d + ma].set(x6)


def a2d_project(ug6, vg6, tab: DuoHaloTables):
    """JAX twin of ``_a2d_project`` (``cubed_a2d_halo``, fv_duogrid:2676-2763).

    Geographic A -> 3-D Cartesian -> 2-point edge average -> projection
    onto the extended-lattice edge bases.  Returns ``(ud4, vd4)`` of
    shapes ``(6, m4, m4-1)`` and ``(6, m4-1, m4)``.
    """
    _gate("a2d_project", ug6=ug6, vg6=vg6)
    v3 = ug6[..., None] * tab.vlon4 + vg6[..., None] * tab.vlat4
    ue = 0.5 * (v3[:, :, :-1] + v3[:, :, 1:])       # D-u slots (i, j-1/2)
    ve = 0.5 * (v3[:, :-1, :] + v3[:, 1:, :])       # D-v slots (i-1/2, j)
    ud = _dot_static(tab.es4[:, :, 1:-1, :, 0], ue)
    vd = _dot_static(tab.ew4[:, 1:-1, :, :, 1], ve)
    return ud, vd


def a2c_project(ug6, vg6, tab: DuoHaloTables):
    """JAX twin of ``_a2c_project`` (``cubed_a2c_halo``, fv_duogrid:2590-2674).

    Returns ``(uc4, vc4)`` of shapes ``(6, m4-1, m4)`` and
    ``(6, m4, m4-1)``.
    """
    _gate("a2c_project", ug6=ug6, vg6=vg6)
    v3 = ug6[..., None] * tab.vlon4 + vg6[..., None] * tab.vlat4
    ue = 0.5 * (v3[:, :-1, :] + v3[:, 1:, :])       # C-u slots (i-1/2, j)
    ve = 0.5 * (v3[:, :, :-1] + v3[:, :, 1:])       # C-v slots (i, j-1/2)
    uc = _dot_static(tab.ew4[:, 1:-1, :, :, 0], ue)
    vc = _dot_static(tab.es4[:, :, 1:-1, :, 1], ve)
    return uc, vc


def write_d_strips(u6, v6, ud4, vd4, tab: DuoHaloTables):
    """JAX twin of ``_write_d_strips`` (ext_vector rmp_s/n/w/e,
    fv_duogrid.F90:890-955).  Returns ``(u6, v6)``.

    S/N strips span the full i data range and W/E then overwrite; the
    duplicated slots keep the LAST (W/E) record, as in the NumPy loop.
    """
    _gate("write_d_strips", u6=u6, v6=v6, ud4=ud4, vd4=vd4)
    flat = _pack(u6, v6, ud4, vd4)
    flat = _apply_scatter(flat, tab.wr_d)
    out = _unpack(tab.lay_wd, flat)
    return out[0], out[1]


def write_c_strips(uc6, vc6, uc4, vc4, tab: DuoHaloTables):
    """JAX twin of ``_write_c_strips``.  Returns ``(uc6, vc6)``."""
    _gate("write_c_strips", uc6=uc6, vc6=vc6, uc4=uc4, vc4=vc4)
    flat = _pack(uc6, vc6, uc4, vc4)
    flat = _apply_scatter(flat, tab.wr_c)
    out = _unpack(tab.lay_wc, flat)
    return out[0], out[1]


def geo_lattice_exchange(g6, tab: DuoHaloTables):
    """JAX twin of ``_geo_lattice_exchange`` (steps 3-4 of the vector flow).

    Neighbour exchange + k2e ring remap + Lagrange corner regions on the
    ng=4 geographic lattice.  The NumPy lane's optional ``stage_dump``
    hook is NOT mirrored (a host callback inside a jitted kernel); it is
    a debugging instrument of the NumPy lane, and dropping it is
    byte-identical to running it with ``dump=None``.  Returns the new
    ``(6, m4, m4)`` stack.
    """
    g6 = exchange_agrid_scalar_halos(g6, tab, ring="geo")
    g6 = k2e_remap_halo_rings(g6, tab, "A", ring="geo")
    return corner_lagrange_fill(g6, tab, "a4")


# ---------------------------------------------------------------------------
# ext_scalar / ext_vector
# ---------------------------------------------------------------------------

def ext_scalar_sixface(f6, tab: DuoHaloTables, stag: str):
    """JAX twin of ``ext_scalar_sixface`` (fv_duogrid.F90:456-502/505-569).

    mpp exchange + cube_rmp rings + Lagrange corner regions at the
    field's own staggering.  ``stag``: ``"A"`` (0,0) or ``"B"`` (1,1) --
    upstream ``ext_scalar`` supports those two only.  Returns the new
    stack.
    """
    if stag == "A":
        f6 = exchange_agrid_scalar_halos(f6, tab, ring="stepper")
        f6 = k2e_remap_halo_rings(f6, tab, "A", ring="stepper")
        return corner_lagrange_fill(f6, tab, "a3")
    if stag == "B":
        f6 = exchange_bgrid_scalar_halos(f6, tab)
        f6 = k2e_remap_halo_rings(f6, tab, "B", ring="stepper")
        return corner_lagrange_fill(f6, tab, "b3")
    raise ValueError(
        f"ext_scalar_sixface: stagger {stag!r} not implemented (upstream "
        f"ext_scalar supports (0,0) and (1,1) only)")


def ext_vector_dgrid_sixface(u6, v6, tab: DuoHaloTables):
    """JAX twin of ``ext_vector(u, v, …, 0,1,1,0)`` -- D-grid covariant winds.

    fv_duogrid.F90:626-975.  Returns ``(u6, v6)``.  The seven-step
    upstream flow is preserved: mpp DGRID_NE exchange, ``c2l_ord2`` to
    geographic winds at cell centres, embedding in the ng=4 lattice,
    scalar exchange + k2e + corner Lagrange of BOTH components on that
    lattice, ``cubed_a2d_halo`` projection, halo side-strip write-back,
    then ``fill_corner_region`` per component at its own staggering.
    ``tab.vector_corner == "a2d"`` skips that last step (a MEASUREMENT
    variant of the NumPy lane, not a production mode).
    """
    u6, v6 = exchange_dgrid_vector_halos(u6, v6, tab)
    ua, va = c2l_ord2_face(u6, v6, tab)
    ug6 = geo_lattice_exchange(pack_p1(ua, tab), tab)
    vg6 = geo_lattice_exchange(pack_p1(va, tab), tab)
    ud4, vd4 = a2d_project(ug6, vg6, tab)
    u6, v6 = write_d_strips(u6, v6, ud4, vd4, tab)
    if tab.vector_corner == "lagrange":
        u6 = corner_lagrange_fill(u6, tab, "du3")
        v6 = corner_lagrange_fill(v6, tab, "dv3")
    return u6, v6


def ext_vector_cgrid_sixface(uc6, vc6, tab: DuoHaloTables):
    """JAX twin of ``ext_vector(uc, vc, …, 1,0,0,1)`` -- C-grid winds.

    Returns ``(uc6, vc6)``.  The C-u component has stagger (1,0) and
    therefore takes the ``dv3`` corner operator, the C-v component
    (0,1) takes ``du3`` -- as in the NumPy lane.
    """
    uc6, vc6 = exchange_cgrid_vector_halos(uc6, vc6, tab)
    ua, va = c2l_ord2_cgrid_face(uc6, vc6, tab)
    ug6 = geo_lattice_exchange(pack_p1(ua, tab), tab)
    vg6 = geo_lattice_exchange(pack_p1(va, tab), tab)
    uc4, vc4 = a2c_project(ug6, vg6, tab)
    uc6, vc6 = write_c_strips(uc6, vc6, uc4, vc4, tab)
    if tab.vector_corner == "lagrange":
        uc6 = corner_lagrange_fill(uc6, tab, "dv3")   # C-u stagger = (1,0)
        vc6 = corner_lagrange_fill(vc6, tab, "du3")   # C-v stagger = (0,1)
    return uc6, vc6


# ---------------------------------------------------------------------------
# jit policies -- the ONE place each entry point's staticness is decided
# ---------------------------------------------------------------------------

def make_exchange_agrid_scalar_halos_jit(fn=exchange_agrid_scalar_halos):
    """Static: the tables bundle (identity-hashable, carries the index
    constants) and the ``ring`` selector (a Python branch).  No donated
    buffers (grad-path doctrine)."""
    return jax.jit(fn, static_argnums=(1, 2))


def make_exchange_bgrid_scalar_halos_jit(fn=exchange_bgrid_scalar_halos):
    return jax.jit(fn, static_argnums=(1,))


def make_exchange_cgrid_vector_halos_jit(fn=exchange_cgrid_vector_halos):
    return jax.jit(fn, static_argnums=(2,))


def make_exchange_dgrid_vector_halos_jit(fn=exchange_dgrid_vector_halos):
    return jax.jit(fn, static_argnums=(2,))


def make_k2e_remap_halo_rings_jit(fn=k2e_remap_halo_rings):
    return jax.jit(fn, static_argnums=(1, 2, 3))


def make_corner_lagrange_fill_jit(fn=corner_lagrange_fill):
    return jax.jit(fn, static_argnums=(1, 2))


def make_fill_corners_jit(fn):
    """Shared policy for the six ``fill_corners_*`` twins.

    Static: the tables bundle, plus the ``mySign`` float on the three
    PAIR routines (it multiplies a compile-time constant table; a traced
    sign would defeat that folding).  The three single-array scalar
    fills take no sign.  No donated buffers (grad-path doctrine).
    """
    single = (fill_corners_bgrid_x, fill_corners_agrid_x,
              fill_corners_agrid_y)
    pair = (fill_corners_dgrid, fill_corners_cgrid, fill_corners_agrid_pair)
    if fn in single:
        return jax.jit(fn, static_argnums=(1,))
    if fn in pair:
        return jax.jit(fn, static_argnums=(2, 3))
    raise ValueError(
        f"make_fill_corners_jit: {getattr(fn, '__name__', fn)!r} is not one "
        f"of the six fill_corners twins")


def make_average_shared_edge_cgrid_jit(fn=average_shared_edge_cgrid):
    return jax.jit(fn, static_argnums=(2,))


def make_average_allflux_shared_edges_jit(fn=average_allflux_shared_edges):
    return jax.jit(fn, static_argnums=(2,))


def make_average_shared_edge_bgrid_jit(fn=average_shared_edge_bgrid):
    return jax.jit(fn, static_argnums=(2,))


def make_c2l_ord2_face_jit(fn=c2l_ord2_face):
    return jax.jit(fn, static_argnums=(2,))


def make_c2l_ord2_cgrid_face_jit(fn=c2l_ord2_cgrid_face):
    return jax.jit(fn, static_argnums=(2,))


def make_pack_p1_jit(fn=pack_p1):
    return jax.jit(fn, static_argnums=(1,))


def make_a2d_project_jit(fn=a2d_project):
    return jax.jit(fn, static_argnums=(2,))


def make_a2c_project_jit(fn=a2c_project):
    return jax.jit(fn, static_argnums=(2,))


def make_write_d_strips_jit(fn=write_d_strips):
    return jax.jit(fn, static_argnums=(4,))


def make_write_c_strips_jit(fn=write_c_strips):
    return jax.jit(fn, static_argnums=(4,))


def make_geo_lattice_exchange_jit(fn=geo_lattice_exchange):
    return jax.jit(fn, static_argnums=(1,))


def make_ext_scalar_sixface_jit(fn=ext_scalar_sixface):
    return jax.jit(fn, static_argnums=(1, 2))


def make_ext_vector_dgrid_sixface_jit(fn=ext_vector_dgrid_sixface):
    return jax.jit(fn, static_argnums=(2,))


def make_ext_vector_cgrid_sixface_jit(fn=ext_vector_cgrid_sixface):
    return jax.jit(fn, static_argnums=(2,))


exchange_agrid_scalar_halos_jit = make_exchange_agrid_scalar_halos_jit()
exchange_bgrid_scalar_halos_jit = make_exchange_bgrid_scalar_halos_jit()
exchange_cgrid_vector_halos_jit = make_exchange_cgrid_vector_halos_jit()
exchange_dgrid_vector_halos_jit = make_exchange_dgrid_vector_halos_jit()
k2e_remap_halo_rings_jit = make_k2e_remap_halo_rings_jit()
corner_lagrange_fill_jit = make_corner_lagrange_fill_jit()
fill_corners_bgrid_x_jit = make_fill_corners_jit(fill_corners_bgrid_x)
fill_corners_agrid_x_jit = make_fill_corners_jit(fill_corners_agrid_x)
fill_corners_agrid_y_jit = make_fill_corners_jit(fill_corners_agrid_y)
fill_corners_dgrid_jit = make_fill_corners_jit(fill_corners_dgrid)
fill_corners_cgrid_jit = make_fill_corners_jit(fill_corners_cgrid)
fill_corners_agrid_pair_jit = make_fill_corners_jit(fill_corners_agrid_pair)
average_shared_edge_cgrid_jit = make_average_shared_edge_cgrid_jit()
average_allflux_shared_edges_jit = make_average_allflux_shared_edges_jit()
average_shared_edge_bgrid_jit = make_average_shared_edge_bgrid_jit()
c2l_ord2_face_jit = make_c2l_ord2_face_jit()
c2l_ord2_cgrid_face_jit = make_c2l_ord2_cgrid_face_jit()
pack_p1_jit = make_pack_p1_jit()
a2d_project_jit = make_a2d_project_jit()
a2c_project_jit = make_a2c_project_jit()
write_d_strips_jit = make_write_d_strips_jit()
write_c_strips_jit = make_write_c_strips_jit()
geo_lattice_exchange_jit = make_geo_lattice_exchange_jit()
ext_scalar_sixface_jit = make_ext_scalar_sixface_jit()
ext_vector_dgrid_sixface_jit = make_ext_vector_dgrid_sixface_jit()
ext_vector_cgrid_sixface_jit = make_ext_vector_cgrid_sixface_jit()
