"""FV3 hydrostatic/non-hydrostatic pressure-gradient chain -- JAX lane
(geopk, p_grad_c, one_grad_p, nh_p_grad, pk3_halo, pln_halo, pe_halo).

Functional, jit-compatible mirror of the certified NumPy fp64 lane
``fv3_native_pgrad.py``, which is itself a loop-faithful port of the
PINNED oracle tree
``/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean``
(``model/dyn_core.F90``, md5 ``e5a5fab9...``, 3128 lines).  Every line
number quoted in this module was re-derived against THAT tree with
``grep``/``sed``; the instrumented ``fv3_recon/duo_model`` copy (+102
lines) is NOT cited anywhere here.  Subroutine spans, verified:

- ``geopk``      -- dyn_core.F90:2660-2790
- ``p_grad_c``   -- dyn_core.F90:2073-2132
- ``nh_p_grad``  -- dyn_core.F90:2135-2230
- ``one_grad_p`` -- dyn_core.F90:2347-2480
- ``pk3_halo``   -- dyn_core.F90:1832-1884
- ``pln_halo``   -- dyn_core.F90:1886-1933   (the NumPy lane's docstring
  says ``1886-1931``; ``end subroutine pln_halo`` is on :1933)
- ``pe_halo``    -- dyn_core.F90:1935-1963   (the NumPy lane's docstring
  says ``1933-1963``; :1933 is ``end subroutine pln_halo`` and
  ``subroutine pe_halo`` starts on :1935)

The NumPy lane stays the oracle-parity reference; this module is the
production/JAX twin and is certified AGAINST the NumPy lane, never
against the Fortran directly (one authority per hop).

Mirror doctrine (inherited verbatim from ``fv3_nh_core.py``, the
pattern-setter for this campaign)
---------------------------------------------------------------------
* **Functional**: the NumPy lane mutates ``uc``/``vc``/``u``/``v``/
  ``pk``/``gz``/``pp``/``pk3``/``pe`` in place (Fortran ``intent(inout)``);
  every twin here takes the same operands and RETURNS them.  The returned
  tuple order is documented in each docstring.  No aliasing, no output
  parameters.
* **k-recurrences via ``lax.scan``** -- NOT ``associative_scan``, NOT
  ``jnp.cumsum``.  ``geopk``'s top-down ``p1d`` accumulator, its bottom-up
  ``gz`` integral and the three halo-ring column integrals are strictly
  sequential sums whose ASSOCIATION ORDER is part of the parity contract
  (``fv3_native_pgrad.py`` docstring: "the sum ORDER is part of the
  bit-exact contract"); ``jnp.cumsum`` may lower to a log-depth
  associative scan and re-associate them.
* **A loop is vectorised ONLY after the dependence is PROVED, and the
  proof is written at the site.**  Vectorising is legal exactly when no
  iteration reads a location an earlier iteration writes -- including
  through two ``fort`` views aliased onto one buffer, and through
  halo/corner storage.  Every vectorised loop in this module carries a
  one-line dependence argument in a comment; the four that are NOT
  trivially independent are called out in "Dependence proofs" below.
* **Explicit x64**: every operand must arrive float64 (the oracle build is
  ``-fdefault-real-8``); a float32 operand raises ``TypeError`` at entry
  exactly like the NumPy lane's ``_require_f64`` (the check reads only
  static dtypes, so it is jit-safe).
* **No ``donate_argnums``** anywhere in this lane (grad-path doctrine,
  CLAUDE.md).
* **Static Python ``if`` stays a Python ``if``.**  ``cg``, ``duogrid``,
  ``computehalo``, ``sw_dynamics``, ``hydrostatic``, ``use_logp``,
  ``d_ext > 0``, the four corner flags and ``grid_type`` all select
  Fortran ``#ifdef``/branch arms and are STATIC.

``jnp.where`` / ``lax.cond`` sites: NONE
---------------------------------------------------------------------
``jnp.where`` is a SELECT, not lazy control flow: both operands are
traced and evaluated, so a division, ``log``, ``sqrt`` or out-of-range
gather in the UNSELECTED operand still produces NaN/Inf, and under
reverse mode ``0 * NaN = NaN`` contaminates the gradient of the operand
that WAS selected.  This chain is full of ``log`` and of divisions by
interface-pressure differences, so the rule here is simply: **there is
not one data-dependent branch in the whole chain, and this module
contains zero ``jnp.where`` and zero ``lax.cond``.**  Every branch listed
above is a STATIC Python ``if`` on a deck constant, resolved at trace
time, so the untaken arm is never traced at all -- no operand of it is
evaluated and no NaN can be manufactured by a dead branch.

The same hazard *does* exist here in a second form, and is contracted
and tested rather than assumed: the NaN-filled local scratch.  The NumPy
twin allocates ``qx``/``qy``/``qxx``/``qyy``/``q1``/``q2`` (a2b) and
``wk``/``wk1`` (one_grad_p / nh_p_grad) as NaN so that a read of a slot
the Fortran never wrote is LOUD instead of plausible, and this lane
keeps that.  A NaN in a never-read slot is harmless in the primal AND in
the VJP -- the VJP of ``.at[win].set`` w.r.t. its base zeroes the written
window and the base here is a constant, and every arithmetic op the NaN
slots feed is ``constant * array``, whose VJP never multiplies a
cotangent by the NaN primal.  What would break that is a WINDOW SLIP: one
index off and a NaN enters a live expression.  ``tests/grids/
test_fv3_pgrad.py`` therefore poisons, for every routine, the operand
region the routine must never read, and asserts that BOTH the primal and
the reverse-mode gradient stay finite.

Dependence proofs (the four non-trivial vectorised loops)
---------------------------------------------------------------------
1. **The ``do k`` nests of p_grad_c (:2100), one_grad_p (:2447) and
   nh_p_grad (:2190)** write ``uc``/``vc``/``u``/``v`` at level k and
   read ``pkc``/``gz``/``pk``/``pp`` at the bracketing interfaces k and
   k+1 of arrays that this loop never writes.  No iteration reads a
   location any iteration writes -> vectorise over k.
2. **``a2b_ord4``'s per-level calls** (:2399/:2409/:2456, :2182-:2191)
   become one ``jax.vmap`` over the level axis: a2b reads only the plane
   it is handed (no level index appears in any of its expressions), and
   its ``qout`` scratch is never read before it is written WITHIN a call
   (proved window by window in ``_a2b_ord4``'s docstring), so nothing
   carries from level k to level k+1.  vmap is a batching transform, not
   a reassociation -- each level executes the identical op sequence.
3. **``replace=.true.`` is a read-after-write ACROSS the call boundary**:
   a2b overwrites its own input ``pk``/``gz``/``pp``/``pk3`` on the B box
   and the ``do k`` momentum loop that follows reads those OVERWRITTEN
   values (B-grid corner values), not the A-grid originals.  The
   functional rewrite THREADS the returned array explicitly
   (``pk = pk.at[...].set(_a2b_replace(pk[...]))``) and every later read
   goes to the rebound name; recomputing from the original operand would
   silently use A-grid values in a B-grid formula.
4. **``one_grad_p``'s ``wk`` is ALIASED in the NumPy twin** -- the same
   buffer is a2b's ``qout`` scratch and the k-loop hydrostatic weight.
   The alias is dead: the k loop rewrites the ENTIRE B box
   ``[is,ie+1] x [js,je+1]`` before any read, and its four reads
   (``ui``/``uip1`` x ``uj``, ``vi`` x ``vj``/``vjp1``) all lie inside
   that box, so no a2b leftover is ever read.  This lane allocates the
   two separately, which is equivalent for exactly that reason.
   ``nh_p_grad``'s ``wk1`` is NOT in this class -- it IS a2b's output and
   is read as such.

Why ``a2b_ord4`` is mirrored PRIVATELY here
---------------------------------------------------------------------
``one_grad_p`` (:2399/:2409/:2456) and ``nh_p_grad`` (:2182/:2183/:2185/
:2191) call ``a2b_ord4``.  A JAX cell->corner interpolator already exists
as ``operators_cdgrid.interp_center_to_corner_a2b_ord4`` -- it is NOT
reused, deliberately:

* it is a cdgrid-signature REIMPLEMENTATION (``(face, i, j)`` layout,
  its own halo/edge conventions), not a loop-faithful mirror of the
  NumPy lane's ``fv3_native_d_sw.a2b_ord4``;
* under R1 (literal translation) the NumPy lane is THE specification for
  this lane, and "obviously equivalent" substitutions are exactly the
  class of change the Koldunov literal-translation rule forbids -- a
  divergence introduced by swapping in a differently-derived operator is
  no longer a port bug by definition, and the debugging search space
  stops collapsing;
* the two disagree structurally on the duo arm: the NumPy lane's
  ``a2b_ord4`` widens its three ``bounded_domain`` gates to
  ``bounded .or. dg%is_initialized`` (a2b_edge.F90 gates 98/185/241), so
  duo takes the interior 4th-order arm EVERYWHERE and reads none of
  ``dxa``/``dya``/``grid``/``agrid``/``edge_*``.

``_a2b_ord4`` below therefore mirrors ``fv3_native_d_sw.a2b_ord4``
statement for statement -- ALL arms: duo/bounded, the plain
corner+edge arm (3-way corner extrapolation, one-sided west/east/south/
north reconstructions, the ``c1``/``c2`` edge blends) and the
``grid_type>=3`` doubly-periodic arm.  It is private (leading underscore)
because it is a pgrad-lane transcription, not a new public operator; when
the d_sw JAX lane lands it will need the same mirror and the two should
then be factored, not duplicated.

Deliberate deviations from the NumPy twin (each one declared)
---------------------------------------------------------------------
1. **``gs`` carries ARRAYS only; the a2b FLAGS are explicit static
   keyword arguments** (``bounded_domain``, ``grid_type``,
   ``sw_corner``/``se_corner``/``ne_corner``/``nw_corner``).  A Python
   ``bool`` inside a traced pytree becomes a tracer and
   ``if gridstruct["sw_corner"]`` would raise; the same split is what
   ``fv3_nh_core.update_dz_c`` already does for its corner flags.
2. **``a2b_gridstruct_view`` returns raw arrays, not ``fort`` views.**
   ``fort`` is a NumPy-lane storage device; the JAX twin carries the
   Fortran origins explicitly (``ilo = is - ng``, exactly the a2b dummy
   declaration ``qin(is-ng:ie+ng, js-ng:je+ng)``).  ``one_grad_p``/
   ``nh_p_grad`` raise if ``bd.isd != is - ng`` -- the Fortran REBASES
   the array at that dummy declaration, so a mismatch is a silent
   shifted-data read on both lanes.
3. **``one_grad_p``'s ``bvertex_mean2`` screen is NOT ported.**  The
   NumPy lane documents it as "a NON-FAITHFUL diagnostic screen (codex
   vertex-kill C3) ... Default OFF = faithful"; the JAX lane ports the
   faithful path only and the parameter does not exist here (it cannot
   be switched on by accident).
4. **Strict-``bool`` guards on ``hydrostatic``** (p_grad_c, one_grad_p).
   The NumPy lane branches on truthiness; a ``np.bool_``, a string or a
   tracer would silently take an arm.  Added guard, no numerics change.
5. **``ng >= 2`` guard in ``_a2b_ord4``** and the shape guards in
   ``geopk``.  The NumPy lane's ``fort`` views index ``i - ilo``, so a
   too-small halo WRAPS to the far edge and returns numbers; here it is
   a loud error (same class as ``fv3_nh_core.update_dz_c``'s ng guard).
6. **``_require_f64_jax`` is REPLICATED, not imported.**  The pattern-
   setter's copy is ``fv3_nh_core._require_f64_jax`` -- a private symbol,
   and ``tests/test_no_private_cross_imports.py`` forbids cross-module
   private imports with a permanently EMPTY allowlist.  Promoting it
   would edit the pattern-setter, which is outside this unit's scope.

Differentiability
---------------------------------------------------------------------
The chain is smooth algebra (``log``, ``exp``, ``**``, +-*/) with no
limiter, no ``max``/``min`` and no data-dependent select, so
``check_grads(order=2)`` is expected to hold generically.  The non-smooth
/ singular sites, named rather than silently avoided:

* ``1 / (wk(i,j) + wk(i+1,j))`` in p_grad_c (:2118), one_grad_p (:2466)
  and nh_p_grad (:2201) -- singular where two adjacent interface-pressure
  differences (or the B-grid ``delp``) sum to zero, i.e. a zero-thickness
  layer pair.  Never in a physical column.
* ``log(p1d)`` / ``exp(akap*log(p))`` -- singular at ``p <= 0``.
* ``ptop ** akap`` at ``ptop = 0`` (the ``-DSW_DYNAMICS`` convention):
  ``d/dptop`` is infinite there.  ``ptop`` is STATIC in this lane, so no
  gradient is taken through it.
* ``_great_circle_dist`` (plain a2b arm only) -- ``arcsin(sqrt(x))`` is
  non-differentiable at ``x = 0`` (coincident points) and its derivative
  diverges at ``x = 1`` (antipodal).  Grid corners are neither.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np  # STATIC trace-time scalars only (peln1), never traced
from jax import lax

# a2b_edge.F90 module constants, imported from the NumPy twin so they can
# never drift from it (R2: constants are imported or carry file:line).
from legoesm.core.fv3_native_d_sw import (
    A2B_A1,
    A2B_A2,
    A2B_B1,
    A2B_B2,
    A2B_R3,
)
from legoesm.core.fv3_native_sw_core import BIG_NUMBER

__all__ = [
    "a2b_gridstruct_view",
    "geopk",
    "geopk_jit",
    "make_geopk_jit",
    "make_nh_p_grad_jit",
    "make_one_grad_p_jit",
    "make_p_grad_c_jit",
    "make_pe_halo_jit",
    "make_pk3_halo_jit",
    "make_pln_halo_jit",
    "nh_p_grad",
    "nh_p_grad_jit",
    "one_grad_p",
    "one_grad_p_jit",
    "p_grad_c",
    "p_grad_c_jit",
    "pe_halo",
    "pe_halo_jit",
    "pk3_halo",
    "pk3_halo_jit",
    "pln_halo",
    "pln_halo_jit",
]

# The ten geometry arrays a2b_ord4 associates (a2b_gridstruct_view).
_A2B_GEOM_KEYS = ("grid_lon", "grid_lat", "agrid_lon", "agrid_lat",
                  "dxa", "dya", "edge_w", "edge_e", "edge_s", "edge_n")


def _require_f64_jax(fname: str, arrays: dict) -> None:
    """Static-dtype gate mirroring the NumPy lane's ``_require_f64``.

    Replicated rather than imported from ``fv3_nh_core`` -- see deviation
    (6) in the module docstring.  Reads only ``.dtype`` (static under
    jit): a float32 operand would otherwise be silently upcast -- or
    worse, with x64 disabled the whole chain would silently run in
    float32 -- and the oracle build is ``-fdefault-real-8``.
    """
    for name, a in arrays.items():
        if jnp.asarray(a).dtype != jnp.float64:
            raise TypeError(
                f"{fname}: {name} must be float64 (got "
                f"{jnp.asarray(a).dtype}); enable jax_enable_x64 and pass "
                f"f64 operands (oracle build is -fdefault-real-8)")


def _w(lo: int, ia: int, ib: int) -> slice:
    """Slice for the Fortran window ``ia..ib`` on an axis whose python
    index 0 holds Fortran index ``lo`` (verbatim from the NumPy lane).

    Every argument is a STATIC Python int, so the slice is a trace-time
    constant and no gather is emitted.
    """
    if ib < ia:
        return slice(0, 0)
    return slice(ia - lo, ib - lo + 1)


def _require_bool(fname: str, name: str, v) -> None:
    """Strict-``bool`` dispatch guard (declared deviation 4).

    ``np.bool_``/``jnp`` scalars/strings are all truthy-or-falsy and would
    select a branch silently; a traced value would raise deep inside the
    trace instead of at the entry point.
    """
    if v is not True and v is not False:
        raise ValueError(
            f"{fname}: {name}={v!r} is not a Python bool.  This flag "
            f"selects a Fortran branch (a different set of formulas) and "
            f"is STATIC in this lane; pass True or False")


# =====================================================================
# a2b_edge_mod -- private mirror of fv3_native_d_sw.a2b_ord4
# =====================================================================

def a2b_gridstruct_view(gs: dict, bd) -> dict:
    """The geometry ``_a2b_ord4`` consumes, as float64 jax arrays.

    JAX twin of ``fv3_native_pgrad.a2b_gridstruct_view``.  Two declared
    differences (module docstring, deviations 1-2): the arrays are NOT
    wrapped in ``fort`` views (``_a2b_ord4`` carries the Fortran origins
    itself) and the corner/domain FLAGS are not carried in the dict --
    they are explicit static keyword arguments, because a Python bool
    inside a traced pytree becomes a tracer.

    NOTE (lane fact, carried from the NumPy twin and verified against
    ``a2b_edge.F90`` gates 98/185/241): on the DUO branch a2b_ord4 takes
    the interior-everywhere arms and reads NONE of ``dxa``/``dya``/
    ``grid``/``agrid``/``edge_*``.  They are staged because the Fortran
    unconditionally associates the pointers (and the plain lane does read
    them) -- not because the duo lane consumes them.

    ``bd`` is accepted for signature parity with the NumPy twin (which
    needs it to build the ``fort`` origins) and is unused here.
    """
    del bd
    missing = [k for k in _A2B_GEOM_KEYS if k not in gs]
    if missing:
        raise KeyError(
            f"a2b_gridstruct_view: gridstruct is missing {missing}; "
            f"a2b_ord4 associates all of {list(_A2B_GEOM_KEYS)}")
    out = {k: jnp.asarray(gs[k]) for k in _A2B_GEOM_KEYS}
    _require_f64_jax("a2b_gridstruct_view", out)
    return out


def _great_circle_dist(q1, q2, radius: float | None = None):
    """``fv_grid_utils.F90 great_circle_dist`` (mirrors the NumPy twin).

    ``q1``/``q2`` are (lon, lat) pairs of 0-d arrays.  a2b_ord4 calls
    ``extrap_corner`` -> here WITHOUT ``radius`` (the angle form); the
    parameter is kept for signature parity with the twin.
    """
    beta = jnp.arcsin(jnp.sqrt(
        jnp.sin((q1[1] - q2[1]) / 2.0) ** 2
        + jnp.cos(q1[1]) * jnp.cos(q2[1])
        * jnp.sin((q1[0] - q2[0]) / 2.0) ** 2)) * 2.0

    if radius is not None:
        return radius * beta
    else:
        return beta   # Returns the angle


def _extrap_corner(p0, p1, p2, q1, q2):
    """``a2b_edge.F90 extrap_corner`` (mirrors the NumPy twin)."""
    x1 = _great_circle_dist(p1, p0)
    x2 = _great_circle_dist(p2, p0)

    return q1 + x1 / (x2 - x1) * (q1 - q2)


def _a2b_ord4(qin, qout, geom: dict, npx: int, npy: int, is_: int, ie: int,
              js: int, je: int, ng: int, replace: bool | None = None,
              duogrid: bool = False, *, bounded_domain: bool = False,
              grid_type: int = 0, sw_corner: bool = True,
              se_corner: bool = True, ne_corner: bool = True,
              nw_corner: bool = True):
    """Functional JAX mirror of ``fv3_native_d_sw.a2b_ord4`` (all arms).

    ``qin``/``qout`` are 2-D arrays with Fortran origin
    ``(is-ng, js-ng)`` -- the a2b dummy declaration
    ``qin(is-ng:ie+ng, js-ng:je+ng)``, which is what the Fortran REBASES
    the caller's ``qin(isd,jsd,k)`` to.  The per-level loop lives in the
    caller (``_a2b_ord4_k`` vmaps this over a trailing k axis).

    Returns ``(qin, qout)``: ``qin`` is only changed when ``replace`` is
    true (the ``.true.`` actual argument at dyn_core.F90:2399/:2409/
    :2182/:2183/:2185), in which case the B box ``[is,ie+1] x [js,je+1]``
    holds the corner values; ``qout`` holds the B-grid result in that box
    and the caller's incoming values everywhere else.

    ``qout``'s incoming values are never READ (checked window by window
    against the twin: every ``qout`` read in the plain arm -- ``qxx[i,2]``
    <- ``qout[i,1]``, ``qyy[2,j]`` <- ``qout[1,j]``, and the two
    ``npy``/``npx`` siblings -- is guarded by the same ``js==1`` /
    ``is_==1`` / ``je+1==npy`` / ``ie+1==npx`` condition that wrote it, on
    the identical i/j window), so a NaN-filled scratch stays out of the
    result and out of the gradient.
    """
    if ng < 2:
        raise ValueError(
            f"_a2b_ord4: ng={ng} < 2 unsupported (the interior stencil "
            f"reads i = is-2 and j = js-2; the NumPy lane's fort views "
            f"would silently WRAP to the far edge instead of raising)")
    _require_bool("_a2b_ord4", "duogrid", duogrid)
    _require_bool("_a2b_ord4", "bounded_domain", bounded_domain)

    qin = jnp.asarray(qin)
    qout = jnp.asarray(qout)
    if qin.ndim != 2 or qout.ndim != 2:
        raise ValueError(
            f"_a2b_ord4: qin/qout must be 2-D (the Fortran dummy is 2-D; "
            f"the k loop lives in the caller), got {qin.shape}/"
            f"{qout.shape}")
    dtype = qin.dtype

    # local: compact 4-pt cubic (FUNCTION-LOCAL c1/c2 -- these SHADOW the
    # sw_core module c1/c2/c3; a classic transcription trap, called out
    # in the NumPy twin)
    c1 = 2.0 / 3.0
    c2 = -1.0 / 6.0

    grid_lon = geom["grid_lon"]
    grid_lat = geom["grid_lat"]
    agrid_lon = geom["agrid_lon"]
    agrid_lat = geom["agrid_lat"]
    dxa = geom["dxa"]
    dya = geom["dya"]
    # edge_{w,e,s,n} are 1-D Fortran arrays edge(1..npx); the gs dict
    # holds them 0-based, so Fortran edge(j) is edge[j-1] -- indexed here
    # through _w(1, ...) so the off-by-one cannot creep back in.
    edge_w = geom["edge_w"]
    edge_e = geom["edge_e"]
    edge_s = geom["edge_s"]
    edge_n = geom["edge_n"]

    ilo, jlo = is_ - ng, js - ng            # qin/qout/dxa/dya/grid origin

    def _p(i, j):
        """qin(i, j) -- scalar Fortran read."""
        return qin[i - ilo, j - jlo]

    def _pw(ia, ib, ja, jb):
        """qin(ia:ib, ja:jb) -- Fortran window."""
        return qin[_w(ilo, ia, ib), _w(jlo, ja, jb)]

    def _grid(i, j):
        return (grid_lon[i - ilo, j - jlo], grid_lat[i - ilo, j - jlo])

    def _agrid(i, j):
        return (agrid_lon[i - ilo, j - jlo], agrid_lat[i - ilo, j - jlo])

    # Local Fortran-bounded scratch (NaN fill, exactly like the twin's
    # _fl locals: an unwritten slot that is later read must be LOUD).
    qx_i, qx_j = is_, js - ng
    qy_i, qy_j = is_ - ng, js
    qq_i, qq_j = is_ - ng, js - ng
    qx = jnp.full((ie + 1 - qx_i + 1, je + ng - qx_j + 1), jnp.nan, dtype)
    qy = jnp.full((ie + ng - qy_i + 1, je + 1 - qy_j + 1), jnp.nan, dtype)
    qxx = jnp.full((ie + ng - qq_i + 1, je + ng - qq_j + 1), jnp.nan,
                   dtype)
    qyy = jnp.full((ie + ng - qq_i + 1, je + ng - qq_j + 1), jnp.nan,
                   dtype)
    q1 = jnp.full((ie + 1 - (is_ - 1) + 1,), jnp.nan, dtype)
    q2 = jnp.full((je + 1 - (js - 1) + 1,), jnp.nan, dtype)

    if grid_type < 3:

        is1 = max(1, is_ - 1)
        js1 = max(1, js - 1)
        is2 = max(2, is_)
        js2 = max(2, js)

        ie1 = min(npx - 1, ie + 1)
        je1 = min(npy - 1, je + 1)

        # Corners:
        # 3-way extrapolation
        if bounded_domain or duogrid:

            # j = js-2..je+2, i = is..ie+1.  DEPENDENCE: reads qin only
            # (qin is untouched until the `replace` copy at the end), so
            # no iteration reads a location any iteration writes.
            qx = qx.at[_w(qx_i, is_, ie + 1),
                       _w(qx_j, js - 2, je + 2)].set(
                A2B_B2 * (_pw(is_ - 2, ie - 1, js - 2, je + 2)
                          + _pw(is_ + 1, ie + 2, js - 2, je + 2))
                + A2B_B1 * (_pw(is_ - 1, ie, js - 2, je + 2)
                            + _pw(is_, ie + 1, js - 2, je + 2)))

        else:

            if sw_corner:
                p0 = _grid(1, 1)
                qout = qout.at[1 - ilo, 1 - jlo].set(
                    (_extrap_corner(p0, _agrid(1, 1), _agrid(2, 2),
                                    _p(1, 1), _p(2, 2))
                     + _extrap_corner(p0, _agrid(0, 1), _agrid(-1, 2),
                                      _p(0, 1), _p(-1, 2))
                     + _extrap_corner(p0, _agrid(1, 0), _agrid(2, -1),
                                      _p(1, 0), _p(2, -1))) * A2B_R3)
            if se_corner:
                p0 = _grid(npx, 1)
                qout = qout.at[npx - ilo, 1 - jlo].set(
                    (_extrap_corner(p0, _agrid(npx - 1, 1),
                                    _agrid(npx - 2, 2),
                                    _p(npx - 1, 1), _p(npx - 2, 2))
                     + _extrap_corner(p0, _agrid(npx - 1, 0),
                                      _agrid(npx - 2, -1),
                                      _p(npx - 1, 0), _p(npx - 2, -1))
                     + _extrap_corner(p0, _agrid(npx, 1),
                                      _agrid(npx + 1, 2),
                                      _p(npx, 1), _p(npx + 1, 2)))
                    * A2B_R3)
            if ne_corner:
                p0 = _grid(npx, npy)
                qout = qout.at[npx - ilo, npy - jlo].set(
                    (_extrap_corner(p0, _agrid(npx - 1, npy - 1),
                                    _agrid(npx - 2, npy - 2),
                                    _p(npx - 1, npy - 1),
                                    _p(npx - 2, npy - 2))
                     + _extrap_corner(p0, _agrid(npx, npy - 1),
                                      _agrid(npx + 1, npy - 2),
                                      _p(npx, npy - 1),
                                      _p(npx + 1, npy - 2))
                     + _extrap_corner(p0, _agrid(npx - 1, npy),
                                      _agrid(npx - 2, npy + 1),
                                      _p(npx - 1, npy),
                                      _p(npx - 2, npy + 1))) * A2B_R3)
            if nw_corner:
                p0 = _grid(1, npy)
                qout = qout.at[1 - ilo, npy - jlo].set(
                    (_extrap_corner(p0, _agrid(1, npy - 1),
                                    _agrid(2, npy - 2),
                                    _p(1, npy - 1), _p(2, npy - 2))
                     + _extrap_corner(p0, _agrid(0, npy - 1),
                                      _agrid(-1, npy - 2),
                                      _p(0, npy - 1), _p(-1, npy - 2))
                     + _extrap_corner(p0, _agrid(1, npy),
                                      _agrid(2, npy + 1),
                                      _p(1, npy), _p(2, npy + 1)))
                    * A2B_R3)

            # ------------
            # X-Interior:
            # ------------
            # DEPENDENCE: writes qx, reads qin only.
            ja, jb = max(1, js - 2), min(npy - 1, je + 2)
            ia, ib = max(3, is_), min(npx - 2, ie + 1)
            qx = qx.at[_w(qx_i, ia, ib), _w(qx_j, ja, jb)].set(
                A2B_B2 * (_pw(ia - 2, ib - 2, ja, jb)
                          + _pw(ia + 1, ib + 1, ja, jb))
                + A2B_B1 * (_pw(ia - 1, ib - 1, ja, jb)
                            + _pw(ia, ib, ja, jb)))

            # *** West Edges:
            # DEPENDENCE (the one non-trivial block on this arm): the
            # Fortran writes qx(1,j) and qx(2,j) in ONE `do j` body and
            # qx(2,j) reads qx(1,j) -- but only at the SAME j, and no
            # iteration writes qx(1,j') for j' != j.  Splitting into two
            # full-window passes (qx[1,:] then qx[2,:], on the REBOUND
            # qx) is therefore identical.  qx(3,j) and q2 come from
            # earlier, completed loops.
            if is_ == 1:
                jw = _w(jlo, js1, je1)
                q2 = q2.at[_w(js - 1, js1, je1)].set(
                    (qin[0 - ilo, jw] * dxa[1 - ilo, jw]
                     + qin[1 - ilo, jw] * dxa[0 - ilo, jw])
                    / (dxa[0 - ilo, jw] + dxa[1 - ilo, jw]))
                ew = edge_w[_w(1, js2, je1)]
                qout = qout.at[1 - ilo, _w(jlo, js2, je1)].set(
                    ew * q2[_w(js - 1, js2 - 1, je1 - 1)]
                    + (1.0 - ew) * q2[_w(js - 1, js2, je1)])
                ja, jb = max(1, js - 2), min(npy - 1, je + 2)
                jc, jq = _w(jlo, ja, jb), _w(qx_j, ja, jb)
                g_in = dxa[2 - ilo, jc] / dxa[1 - ilo, jc]
                g_ou = dxa[-1 - ilo, jc] / dxa[0 - ilo, jc]
                qx = qx.at[1 - qx_i, jq].set(
                    0.5 * (((2.0 + g_in) * qin[1 - ilo, jc]
                            - qin[2 - ilo, jc]) / (1.0 + g_in)
                           + ((2.0 + g_ou) * qin[0 - ilo, jc]
                              - qin[-1 - ilo, jc]) / (1.0 + g_ou)))
                qx = qx.at[2 - qx_i, jq].set(
                    (3.0 * (g_in * qin[1 - ilo, jc] + qin[2 - ilo, jc])
                     - (g_in * qx[1 - qx_i, jq] + qx[3 - qx_i, jq]))
                    / (2.0 + 2.0 * g_in))

            # East Edges:
            # DEPENDENCE: same shape as the West block (qx(npx-1,j) reads
            # qx(npx,j) at the SAME j, and qx(npx-2,j) from the completed
            # X-Interior loop).  The West block runs FIRST, exactly as in
            # the source, so a degenerate grid where npx-2 <= 2 makes the
            # two blocks collide identically on both lanes.
            if (ie + 1) == npx:
                jw = _w(jlo, js1, je1)
                q2 = q2.at[_w(js - 1, js1, je1)].set(
                    (qin[npx - 1 - ilo, jw] * dxa[npx - ilo, jw]
                     + qin[npx - ilo, jw] * dxa[npx - 1 - ilo, jw])
                    / (dxa[npx - 1 - ilo, jw] + dxa[npx - ilo, jw]))
                ee = edge_e[_w(1, js2, je1)]
                qout = qout.at[npx - ilo, _w(jlo, js2, je1)].set(
                    ee * q2[_w(js - 1, js2 - 1, je1 - 1)]
                    + (1.0 - ee) * q2[_w(js - 1, js2, je1)])
                ja, jb = max(1, js - 2), min(npy - 1, je + 2)
                jc, jq = _w(jlo, ja, jb), _w(qx_j, ja, jb)
                g_in = dxa[npx - 2 - ilo, jc] / dxa[npx - 1 - ilo, jc]
                g_ou = dxa[npx + 1 - ilo, jc] / dxa[npx - ilo, jc]
                qx = qx.at[npx - qx_i, jq].set(
                    0.5 * (((2.0 + g_in) * qin[npx - 1 - ilo, jc]
                            - qin[npx - 2 - ilo, jc]) / (1.0 + g_in)
                           + ((2.0 + g_ou) * qin[npx - ilo, jc]
                              - qin[npx + 1 - ilo, jc]) / (1.0 + g_ou)))
                qx = qx.at[npx - 1 - qx_i, jq].set(
                    (3.0 * (qin[npx - 2 - ilo, jc]
                            + g_in * qin[npx - 1 - ilo, jc])
                     - (g_in * qx[npx - qx_i, jq]
                        + qx[npx - 2 - qx_i, jq]))
                    / (2.0 + 2.0 * g_in))

        # ------------
        # Y-Interior:
        # ------------

        if bounded_domain or duogrid:

            # j = js..je+1, i = is-2..ie+2.  DEPENDENCE: reads qin only.
            qy = qy.at[_w(qy_i, is_ - 2, ie + 2),
                       _w(qy_j, js, je + 1)].set(
                A2B_B2 * (_pw(is_ - 2, ie + 2, js - 2, je - 1)
                          + _pw(is_ - 2, ie + 2, js + 1, je + 2))
                + A2B_B1 * (_pw(is_ - 2, ie + 2, js - 1, je)
                            + _pw(is_ - 2, ie + 2, js, je + 1)))

        else:

            ja, jb = max(3, js), min(npy - 2, je + 1)
            ia, ib = max(1, is_ - 2), min(npx - 1, ie + 2)
            qy = qy.at[_w(qy_i, ia, ib), _w(qy_j, ja, jb)].set(
                A2B_B2 * (_pw(ia, ib, ja - 2, jb - 2)
                          + _pw(ia, ib, ja + 1, jb + 1))
                + A2B_B1 * (_pw(ia, ib, ja - 1, jb - 1)
                            + _pw(ia, ib, ja, jb)))

            # South Edges:
            # DEPENDENCE: the qy(i,1)/qy(i,2) pair is the transpose of the
            # West block's argument -- qy(i,2) reads qy(i,1) at the SAME
            # i, and qy(i,3) from the completed Y-Interior loop.
            if js == 1:
                iw = _w(ilo, is1, ie1)
                q1 = q1.at[_w(is_ - 1, is1, ie1)].set(
                    (qin[iw, 0 - jlo] * dya[iw, 1 - jlo]
                     + qin[iw, 1 - jlo] * dya[iw, 0 - jlo])
                    / (dya[iw, 0 - jlo] + dya[iw, 1 - jlo]))
                es = edge_s[_w(1, is2, ie1)]
                qout = qout.at[_w(ilo, is2, ie1), 1 - jlo].set(
                    es * q1[_w(is_ - 1, is2 - 1, ie1 - 1)]
                    + (1.0 - es) * q1[_w(is_ - 1, is2, ie1)])
                ia, ib = max(1, is_ - 2), min(npx - 1, ie + 2)
                ic, iq = _w(ilo, ia, ib), _w(qy_i, ia, ib)
                g_in = dya[ic, 2 - jlo] / dya[ic, 1 - jlo]
                g_ou = dya[ic, -1 - jlo] / dya[ic, 0 - jlo]
                qy = qy.at[iq, 1 - qy_j].set(
                    0.5 * (((2.0 + g_in) * qin[ic, 1 - jlo]
                            - qin[ic, 2 - jlo]) / (1.0 + g_in)
                           + ((2.0 + g_ou) * qin[ic, 0 - jlo]
                              - qin[ic, -1 - jlo]) / (1.0 + g_ou)))
                qy = qy.at[iq, 2 - qy_j].set(
                    (3.0 * (g_in * qin[ic, 1 - jlo] + qin[ic, 2 - jlo])
                     - (g_in * qy[iq, 1 - qy_j] + qy[iq, 3 - qy_j]))
                    / (2.0 + 2.0 * g_in))

            # North Edges:
            # DEPENDENCE: as South; the South block runs FIRST, as in the
            # source, so any degenerate-grid collision resolves the same
            # way on both lanes.
            if (je + 1) == npy:
                iw = _w(ilo, is1, ie1)
                q1 = q1.at[_w(is_ - 1, is1, ie1)].set(
                    (qin[iw, npy - 1 - jlo] * dya[iw, npy - jlo]
                     + qin[iw, npy - jlo] * dya[iw, npy - 1 - jlo])
                    / (dya[iw, npy - 1 - jlo] + dya[iw, npy - jlo]))
                en = edge_n[_w(1, is2, ie1)]
                qout = qout.at[_w(ilo, is2, ie1), npy - jlo].set(
                    en * q1[_w(is_ - 1, is2 - 1, ie1 - 1)]
                    + (1.0 - en) * q1[_w(is_ - 1, is2, ie1)])
                ia, ib = max(1, is_ - 2), min(npx - 1, ie + 2)
                ic, iq = _w(ilo, ia, ib), _w(qy_i, ia, ib)
                g_in = dya[ic, npy - 2 - jlo] / dya[ic, npy - 1 - jlo]
                g_ou = dya[ic, npy + 1 - jlo] / dya[ic, npy - jlo]
                qy = qy.at[iq, npy - qy_j].set(
                    0.5 * (((2.0 + g_in) * qin[ic, npy - 1 - jlo]
                            - qin[ic, npy - 2 - jlo]) / (1.0 + g_in)
                           + ((2.0 + g_ou) * qin[ic, npy - jlo]
                              - qin[ic, npy + 1 - jlo]) / (1.0 + g_ou)))
                qy = qy.at[iq, npy - 1 - qy_j].set(
                    (3.0 * (qin[ic, npy - 2 - jlo]
                            + g_in * qin[ic, npy - 1 - jlo])
                     - (g_in * qy[iq, npy - qy_j]
                        + qy[iq, npy - 2 - qy_j]))
                    / (2.0 + 2.0 * g_in))

        # --------------------------------------

        if bounded_domain or duogrid:

            # DEPENDENCE: qxx reads only qx and qyy only qy (both
            # complete); the Fortran writes qyy(i,j) and qout(i,j) in one
            # j body, but qout reads qyy at the SAME (i,j) -- no
            # neighbour read -- so the three full-window passes below are
            # identical to the interleaved loop.
            bi, bj = _w(qq_i, is_, ie + 1), _w(qq_j, js, je + 1)
            xi = _w(qx_i, is_, ie + 1)
            qxx = qxx.at[bi, bj].set(
                A2B_A2 * (qx[xi, _w(qx_j, js - 2, je - 1)]
                          + qx[xi, _w(qx_j, js + 1, je + 2)])
                + A2B_A1 * (qx[xi, _w(qx_j, js - 1, je)]
                            + qx[xi, _w(qx_j, js, je + 1)]))

            yj = _w(qy_j, js, je + 1)
            qyy = qyy.at[bi, bj].set(
                A2B_A2 * (qy[_w(qy_i, is_ - 2, ie - 1), yj]
                          + qy[_w(qy_i, is_ + 1, ie + 2), yj])
                + A2B_A1 * (qy[_w(qy_i, is_ - 1, ie), yj]
                            + qy[_w(qy_i, is_, ie + 1), yj]))

            qout = qout.at[_w(ilo, is_, ie + 1),
                           _w(jlo, js, je + 1)].set(
                0.5 * (qxx[bi, bj] + qyy[bi, bj]))     # averaging

        else:

            # DEPENDENCE: the qxx interior reads qx only.  The two
            # one-sided specials below read qxx(i,3) / qxx(i,npy-2) --
            # written by THIS completed interior loop, at the same i --
            # and qout(i,1) / qout(i,npy), written by the South/North
            # edge blocks on the identical i window (is2..ie1).  Source
            # order (interior, then js==1, then je+1==npy) is preserved,
            # so a degenerate npy where the two specials touch the same
            # j resolves identically on both lanes.
            ja, jb = max(3, js), min(npy - 2, je + 1)
            ia, ib = max(2, is_), min(npx - 1, ie + 1)
            qxx = qxx.at[_w(qq_i, ia, ib), _w(qq_j, ja, jb)].set(
                A2B_A2 * (qx[_w(qx_i, ia, ib), _w(qx_j, ja - 2, jb - 2)]
                          + qx[_w(qx_i, ia, ib),
                               _w(qx_j, ja + 1, jb + 1)])
                + A2B_A1 * (qx[_w(qx_i, ia, ib),
                               _w(qx_j, ja - 1, jb - 1)]
                            + qx[_w(qx_i, ia, ib), _w(qx_j, ja, jb)]))

            ia, ib = max(2, is_), min(npx - 1, ie + 1)
            if js == 1:
                qxx = qxx.at[_w(qq_i, ia, ib), 2 - qq_j].set(
                    c1 * (qx[_w(qx_i, ia, ib), 1 - qx_j]
                          + qx[_w(qx_i, ia, ib), 2 - qx_j])
                    + c2 * (qout[_w(ilo, ia, ib), 1 - jlo]
                            + qxx[_w(qq_i, ia, ib), 3 - qq_j]))
            if (je + 1) == npy:
                qxx = qxx.at[_w(qq_i, ia, ib), npy - 1 - qq_j].set(
                    c1 * (qx[_w(qx_i, ia, ib), npy - 2 - qx_j]
                          + qx[_w(qx_i, ia, ib), npy - 1 - qx_j])
                    + c2 * (qout[_w(ilo, ia, ib), npy - jlo]
                            + qxx[_w(qq_i, ia, ib), npy - 2 - qq_j]))

            # j = max(2,js)..min(npy-1,je+1): the qyy interior, the two
            # one-sided i specials and the qout average all live in ONE
            # Fortran j loop, and this loop BOTH reads and writes qout.
            # DEPENDENCE (the proof that lets it vectorise over j): the
            # only qout READS here are qout(1,j) and qout(npx,j), and the
            # qout WRITE window is i = max(2,is)..min(npx-1,ie+1) --
            # which contains neither i=1 nor i=npx.  So no j iteration
            # can read a qout slot another j iteration wrote; the reads
            # come from the completed West/East edge blocks.  Every other
            # cross-statement read (qyy(3,j) by the i=2 special, qyy(i,j)
            # by qout) is at the SAME j.  Source order is preserved, so
            # the specials still overwrite the interior write at i=2 /
            # i=npx-1 before qout averages.
            ja, jb = max(2, js), min(npy - 1, je + 1)
            ia, ib = max(3, is_), min(npx - 2, ie + 1)
            jq, jy = _w(qq_j, ja, jb), _w(qy_j, ja, jb)
            qyy = qyy.at[_w(qq_i, ia, ib), jq].set(
                A2B_A2 * (qy[_w(qy_i, ia - 2, ib - 2), jy]
                          + qy[_w(qy_i, ia + 1, ib + 1), jy])
                + A2B_A1 * (qy[_w(qy_i, ia - 1, ib - 1), jy]
                            + qy[_w(qy_i, ia, ib), jy]))
            if is_ == 1:
                qyy = qyy.at[2 - qq_i, jq].set(
                    c1 * (qy[1 - qy_i, jy] + qy[2 - qy_i, jy])
                    + c2 * (qout[1 - ilo, _w(jlo, ja, jb)]
                            + qyy[3 - qq_i, jq]))
            if (ie + 1) == npx:
                qyy = qyy.at[npx - 1 - qq_i, jq].set(
                    c1 * (qy[npx - 2 - qy_i, jy] + qy[npx - 1 - qy_i, jy])
                    + c2 * (qout[npx - ilo, _w(jlo, ja, jb)]
                            + qyy[npx - 2 - qq_i, jq]))

            ia, ib = max(2, is_), min(npx - 1, ie + 1)
            qout = qout.at[_w(ilo, ia, ib), _w(jlo, ja, jb)].set(
                0.5 * (qxx[_w(qq_i, ia, ib), jq]
                       + qyy[_w(qq_i, ia, ib), jq]))      # averaging

    else:  # grid_type>=3
        # ------------------------
        # Doubly periodic domain:
        # ------------------------
        # X-sweep: PPM   (NOTE the B1-then-B2 term order -- it differs
        # from the grid_type<3 arms above and the sum order is part of
        # the parity contract).  DEPENDENCE: qx/qy read qin only; qout
        # reads the two completed sweeps.
        qx = qx.at[_w(qx_i, is_, ie + 1), _w(qx_j, js - 2, je + 2)].set(
            A2B_B1 * (_pw(is_ - 1, ie, js - 2, je + 2)
                      + _pw(is_, ie + 1, js - 2, je + 2))
            + A2B_B2 * (_pw(is_ - 2, ie - 1, js - 2, je + 2)
                        + _pw(is_ + 1, ie + 2, js - 2, je + 2)))
        # Y-sweep: PPM
        qy = qy.at[_w(qy_i, is_ - 2, ie + 2), _w(qy_j, js, je + 1)].set(
            A2B_B1 * (_pw(is_ - 2, ie + 2, js - 1, je)
                      + _pw(is_ - 2, ie + 2, js, je + 1))
            + A2B_B2 * (_pw(is_ - 2, ie + 2, js - 2, je - 1)
                        + _pw(is_ - 2, ie + 2, js + 1, je + 2)))

        xi = _w(qx_i, is_, ie + 1)
        yj = _w(qy_j, js, je + 1)
        qout = qout.at[_w(ilo, is_, ie + 1), _w(jlo, js, je + 1)].set(
            0.5 * (
                A2B_A1 * (qx[xi, _w(qx_j, js - 1, je)]
                          + qx[xi, _w(qx_j, js, je + 1)]
                          + qy[_w(qy_i, is_ - 1, ie), yj]
                          + qy[_w(qy_i, is_, ie + 1), yj])
                + A2B_A2 * (qx[xi, _w(qx_j, js - 2, je - 1)]
                            + qx[xi, _w(qx_j, js + 1, je + 2)]
                            + qy[_w(qy_i, is_ - 2, ie - 1), yj]
                            + qy[_w(qy_i, is_ + 1, ie + 2), yj])))

    # DEPENDENCE: elementwise qin(i,j) <- qout(i,j) on one box, and qin
    # is not read again in this call.  `replace=False` and `replace=None`
    # are BOTH no-ops -- the Fortran guard is
    # `if (present(replace)) then; if (replace) then`.
    if replace is not None:
        if replace:
            bi, bj = _w(ilo, is_, ie + 1), _w(jlo, js, je + 1)
            qin = qin.at[bi, bj].set(qout[bi, bj])

    return qin, qout


def _a2b_ord4_k(qin3, qout3, geom: dict, npx: int, npy: int, is_: int,
                ie: int, js: int, je: int, ng: int,
                replace: bool | None = None, duogrid: bool = False,
                **flags):
    """``_a2b_ord4`` over a trailing k axis (the caller's ``do k`` loop).

    The Fortran calls ``a2b_ord4`` once per level with a 2-D slice
    ``q(isd,jsd,k)`` (dyn_core.F90:2399/:2409/:2456 and :2182-:2191).
    DEPENDENCE PROOF for turning that loop into a batch axis: (a) no
    expression inside ``a2b_ord4`` carries a level index -- it reads only
    the plane it is handed; (b) its writes land in that same plane
    (``replace``) and in ``qout``; (c) ``qout`` is never read before it
    is written WITHIN a call (established window by window in
    ``_a2b_ord4``'s docstring), so nothing survives from level k to level
    k+1 through the shared scratch -- which is what makes the Fortran's
    OMP ``private(wk1)`` legal in the first place.  ``jax.vmap`` maps the
    identical operation sequence over the axis: a batching transform, no
    reassociation, no cross-level mixing.

    Returns ``(qin3, qout3)`` with the k axis restored to position 2.
    """

    def _one(a, b):
        return _a2b_ord4(a, b, geom, npx, npy, is_, ie, js, je, ng,
                         replace=replace, duogrid=duogrid, **flags)

    return jax.vmap(_one, in_axes=(2, 2), out_axes=(2, 2))(qin3, qout3)


# =====================================================================
# dyn_core_mod
# =====================================================================

def geopk(delp, pt, hs, bd, *, km: int, ptop: float, akap: float,
          cp_air: float, cg: bool, duogrid: bool, computehalo: bool,
          npx: int, npy: int, a2b_ord: int, bounded_domain: bool = False,
          sw_dynamics: bool = False, q_con=None, use_cond: bool = False,
          unwritten_fill: float = BIG_NUMBER) -> dict:
    """JAX twin of ``fv3_native_pgrad.geopk`` (dyn_core.F90:2660-2790).

    Functional by construction: the Fortran declares ``pk``/``gz``/``pe``/
    ``peln``/``pkz`` ``intent(OUT)``, so this twin ALLOCATES them and
    returns ``{"pk", "gz", "pe", "peln", "pkz"}`` -- the same dict the
    NumPy twin returns, with the same shapes, origins and
    ``unwritten_fill`` sentinel in every slot the Fortran never writes.

    Two k loops, both strict recurrences, both ``lax.scan``:
    the top-down ``p1d`` accumulator (:2742-2764 -- ``p1d`` is carried
    across k, so summing bottom-up or re-associating changes the last
    bit) and the bottom-up ``gz`` integral (:2766-2779).  The i/j loops
    are vectorised over the identical Fortran windows.

    ``duogrid`` is geopk's own ``duogrid`` dummy: dyn_core feeds it from
    ``gridstruct%dg%is_initialized`` at the C-grid site (:534) and from
    ``flagstruct%duogrid`` at the D-grid site (:1402) -- TWO DIFFERENT
    structure members, not unified here (the NumPy twin's UNCERTAIN U5).

    Every keyword is STATIC (deck constant or ``#ifdef``/branch
    selector); ``delp``, ``pt``, ``hs`` are the traced operands.
    """
    if use_cond:
        raise ValueError(
            "geopk: use_cond=True selects the -DUSE_COND peg/pkg branch "
            "(dyn_core.F90:2721-2724, 2747-2750, 2772-2773) which is NOT "
            "ported and NOT certified by any fixture; refusing to run a "
            "condensate-loaded column through the dry formulas")
    if a2b_ord not in (2, 4):
        raise ValueError(f"geopk: unknown a2b_ord={a2b_ord!r} (expected 2 or 4)")
    if km < 1:
        raise ValueError(f"geopk: km must be >= 1, got {km!r}")
    _require_bool("geopk", "cg", cg)
    _require_bool("geopk", "sw_dynamics", sw_dynamics)
    _require_f64_jax("geopk", {"delp": delp, "pt": pt, "hs": hs})
    del q_con  # unreferenced without -DUSE_COND; kept for interface fidelity

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    m_i = ied - isd + 1
    m_j = jed - jsd + 1
    n_i = ie - is_ + 1
    n_j = je - js + 1

    delp = jnp.asarray(delp)
    pt = jnp.asarray(pt)
    hs = jnp.asarray(hs)

    # --- :2697-2703 range predicate.  A cg=True call ALWAYS takes the
    # 1-halo width (both disjuncts carry `.not. CG`).
    if ((not cg) and a2b_ord == 4) or ((bounded_domain or duogrid) and not cg):
        ifirst, ilast = is_ - 2, ie + 2
        jfirst, jlast = js - 2, je + 2
    else:
        ifirst, ilast = is_ - 1, ie + 1
        jfirst, jlast = js - 1, je + 1

    # --- :2705-2710 computehalo extension (bounded/duo only)
    if (bounded_domain or duogrid) and computehalo:
        if is_ == 1:
            ifirst = isd
        if ie == npx - 1:
            ilast = ied
        if js == 1:
            jfirst = jsd
        if je == npy - 1:
            jlast = jed

    box_i = _w(isd, ifirst, ilast)
    box_j = _w(jsd, jfirst, jlast)
    n_bi = ilast - ifirst + 1
    n_bj = jlast - jfirst + 1

    pk = jnp.full((m_i, m_j, km + 1), unwritten_fill, dtype=jnp.float64)
    gz = jnp.full((m_i, m_j, km + 1), unwritten_fill, dtype=jnp.float64)
    pe = jnp.full((n_i + 2, km + 1, n_j + 2), unwritten_fill,
                  dtype=jnp.float64)
    peln = jnp.full((n_i, km + 1, n_j), unwritten_fill, dtype=jnp.float64)
    pkz = jnp.full((n_i, n_j, km), unwritten_fill, dtype=jnp.float64)

    # --- seeds :2717-2739.  ptk uses the `**` OPERATOR (dyn_core.F90:248,
    # `ptk = ptop ** akap`) while the k loop below uses exp(akap*log(p))
    # (:2747) -- DIFFERENT operations that may differ in the last bit.
    # Mirror each at its site.  ptop/akap are STATIC, so this is a
    # trace-time Python float, computed exactly as the NumPy twin does.
    ptk = ptop ** akap
    pk = pk.at[box_i, box_j, 0].set(ptk)
    hs_box = hs[box_i, box_j]
    gz = gz.at[box_i, box_j, km].set(hs_box)

    # peln seed: `#ifndef SW_DYNAMICS` (:2727-2733), j in js..je, i in is..ie
    j_pl0, j_pl1 = max(jfirst, js), min(jlast, je)
    if not sw_dynamics:
        if ifirst > is_ or ilast < ie:
            raise ValueError(
                "geopk: the peln window (is..ie) escapes the ifirst..ilast "
                "box — upstream would read undefined logp; refusing")
        # dyn_core.F90:246 `peln1 = log(ptop)`; a STATIC trace-time scalar
        # (float(np.log(...)) so the constant is bit-identical to the
        # NumPy twin's).
        peln = peln.at[:, 0, _w(js, j_pl0, j_pl1)].set(float(np.log(ptop)))

    # pe seed (:2735-2739): j in (js-2, je+2) EXCLUSIVE, i clipped to the box
    i_pe0, i_pe1 = max(ifirst, is_ - 1), min(ilast, ie + 1)
    j_pe0, j_pe1 = max(jfirst, js - 1), min(jlast, je + 1)
    pe_i = _w(is_ - 1, i_pe0, i_pe1)
    pe_j = _w(js - 1, j_pe0, j_pe1)
    pe = pe.at[pe_i, 0, pe_j].set(ptop)

    # box-relative sub-windows of p1d/logp for the pe/peln writes
    src_pe_i = _w(ifirst, i_pe0, i_pe1)
    src_pe_j = _w(jfirst, j_pe0, j_pe1)
    src_pl_i = _w(ifirst, is_, ie)
    src_pl_j = _w(jfirst, j_pl0, j_pl1)
    pl_j = _w(js, j_pl0, j_pl1)

    # --- top-down recursion :2742-2764.  p1d is a RUNNING accumulator
    # carried across k (strictly top to bottom); lax.scan, never cumsum.
    def _down(p1d, dp_k):
        p1d = p1d + dp_k                             # :2745
        logp = jnp.log(p1d)                          # :2746
        return p1d, (p1d, logp, jnp.exp(akap * logp))   # :2747

    p1d0 = jnp.full((n_bi, n_bj), ptop, dtype=jnp.float64)
    _, (p1d_k, logp_k, pk_k) = lax.scan(
        _down, p1d0, jnp.moveaxis(delp[box_i, box_j, 0:km], 2, 0))

    pk = pk.at[box_i, box_j, 1:km + 1].set(jnp.moveaxis(pk_k, 0, 2))
    pe = pe.at[pe_i, 1:km + 1, pe_j].set(
        jnp.moveaxis(p1d_k[:, src_pe_i, src_pe_j], 0, 1))
    if not sw_dynamics:
        peln = peln.at[:, 1:km + 1, pl_j].set(
            jnp.moveaxis(logp_k[:, src_pl_i, src_pl_j], 0, 1))

    # --- bottom-up recursion :2766-2779 over the FULL box (wider than
    # the pe/peln window).  `cp_air` is DROPPED under -DSW_DYNAMICS (:2770).
    # DEPENDENCE: this is a read-after-write across loops -- gz(.,.,k)
    # reads pk written by the top-down loop above -- so it reads the
    # REBOUND pk, never the sentinel-filled allocation.  In k it is a
    # true recurrence (gz(k) <- gz(k+1)); in (i, j) every statement is
    # elementwise.
    pk_box = pk[box_i, box_j, :]
    dpk = jnp.moveaxis(pk_box[:, :, 1:km + 1] - pk_box[:, :, 0:km], 2, 0)
    pt_box = jnp.moveaxis(pt[box_i, box_j, 0:km], 2, 0)

    def _up(gz_kp1, x):
        pt_k, dpk_k = x
        if sw_dynamics:
            gz_k = gz_kp1 + pt_k * dpk_k
        else:
            gz_k = gz_kp1 + cp_air * pt_k * dpk_k
        return gz_k, gz_k

    _, gz_k = lax.scan(_up, hs_box, (pt_box, dpk), reverse=True)
    gz = gz.at[box_i, box_j, 0:km].set(jnp.moveaxis(gz_k, 0, 2))

    # --- pkz :2781-2787: only when `.not. CG`, only on [is,ie]x[js,je],
    # and it CONSUMES peln — so it is invalid wherever peln was not
    # written.  Under -DSW_DYNAMICS peln is never written at all
    # (:2727-2733), so upstream's pkz would consume undefined memory;
    # the port leaves the sentinel instead of manufacturing a value.
    # DOCUMENTED DEVIATION, SW lane only (carried from the NumPy twin).
    if (not cg) and not sw_dynamics:
        ki = _w(isd, is_, ie)
        kj = _w(jsd, js, je)
        pkz = pkz.at[:, :, :].set(
            (pk[ki, kj, 1:km + 1] - pk[ki, kj, 0:km])
            / (akap * jnp.transpose(peln[:, 1:km + 1, :] - peln[:, 0:km, :],
                                    (0, 2, 1))))

    return {"pk": pk, "gz": gz, "pe": pe, "peln": peln, "pkz": pkz}


def p_grad_c(dt2: float, delpc, pkc, gz, uc, vc, gs: dict, bd, *,
             npz: int, hydrostatic: bool = True):
    """JAX twin of ``fv3_native_pgrad.p_grad_c`` (dyn_core.F90:2073-2132).

    Functional: the NumPy twin mutates ``uc``/``vc`` in place; this twin
    RETURNS ``(uc, vc)`` with the two Fortran write windows
    (uc: i=is..ie+1, j=js..je; vc: i=is..ie, j=js..je+1) rewritten and
    every other slot carried through unchanged.

    ``delpc`` is UNREAD on the hydrostatic branch (:2109-2113) -- it is
    kept in the signature for interface fidelity and may be ``None``
    there.  The oracle proves it is never read by re-running the Fortran
    with ``delpc = -9.e9`` and requiring a BITWISE identical result.

    NON-HYDROSTATIC branch: the ONLY difference is the denominator
    weight -- ``wk = delpc(:,:,k)`` instead of the ``pkc`` interface
    difference; the two momentum expressions are shared verbatim.  On
    that branch ``pkc`` is FULL interface pressure (the header comment at
    :2079-2081 and ``Riem_Solver_c``'s ``pef = pe2 + pem``), not
    ``pe**cappa`` -- the CALLER owns handing the right quantity.

    There is NO inter-k coupling (:2100): the only vertical reads are the
    bracketing interfaces ``k`` and ``k+1``, so the ``do k=1,npz`` nest is
    a trailing-axis slice here.  The operand GROUPING is copied exactly
    (``dt2*rdxc/(wk+wk)*(...)``): regrouping to ``dt2*(rdxc/(wk+wk))*(...)``
    changes the last bit.

    ``gs`` supplies ``rdxc``/``rdyc`` (the Fortran passes them as explicit
    dummies; the NumPy twin packs them in the gridstruct dict).
    """
    _require_bool("p_grad_c", "hydrostatic", hydrostatic)
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd

    gate = {"pkc": pkc, "gz": gz, "uc": uc, "vc": vc,
            "rdxc": gs["rdxc"], "rdyc": gs["rdyc"]}
    if hydrostatic:
        del delpc  # hydrostatic branch never reads it (see docstring)
    else:
        gate["delpc"] = delpc
    _require_f64_jax("p_grad_c", gate)

    pkc = jnp.asarray(pkc)
    gz = jnp.asarray(gz)
    uc = jnp.asarray(uc)
    vc = jnp.asarray(vc)
    rdxc = jnp.asarray(gs["rdxc"])
    rdyc = jnp.asarray(gs["rdyc"])

    # wk window :2103-2113 == the CG geopk write box
    wlo_i, wlo_j = is_ - 1, js - 1
    wk_i = _w(isd, is_ - 1, ie + 1)
    wk_j = _w(jsd, js - 1, je + 1)

    # uc: i in is..ie+1, j in js..je   (:2116-2122)
    u_i = _w(isd, is_, ie + 1)
    u_im1 = _w(isd, is_ - 1, ie)
    u_j = _w(jsd, js, je)
    uw_i = _w(wlo_i, is_, ie + 1)
    uw_im1 = _w(wlo_i, is_ - 1, ie)
    uw_j = _w(wlo_j, js, je)

    # vc: i in is..ie, j in js..je+1   (:2123-2129)
    v_i = _w(isd, is_, ie)
    v_j = _w(jsd, js, je + 1)
    v_jm1 = _w(jsd, js - 1, je)
    vw_i = _w(wlo_i, is_, ie)
    vw_j = _w(wlo_j, js, je + 1)
    vw_jm1 = _w(wlo_j, js - 1, je)

    # DEPENDENCE (the `do k=1,npz` nest, :2100): wk is rebuilt from pkc
    # (or delpc) at each k and both are READ-ONLY here; uc/vc are written
    # at level k and read at level k only.  No iteration reads a location
    # any iteration writes -> the whole nest vectorises over k.
    if hydrostatic:
        wk = pkc[wk_i, wk_j, 1:npz + 1] - pkc[wk_i, wk_j, 0:npz]  # :2104-2108
    else:
        wk = jnp.asarray(delpc)[wk_i, wk_j, 0:npz]                # :2109-2113

    # Grouping copied EXACTLY (:2118-2120).
    uc = uc.at[u_i, u_j, 0:npz].set(
        uc[u_i, u_j, 0:npz]
        + dt2 * rdxc[u_i, u_j][:, :, None]
        / (wk[uw_im1, uw_j, :] + wk[uw_i, uw_j, :])
        * ((gz[u_im1, u_j, 1:npz + 1] - gz[u_i, u_j, 0:npz])
           * (pkc[u_i, u_j, 1:npz + 1] - pkc[u_im1, u_j, 0:npz])
           + (gz[u_im1, u_j, 0:npz] - gz[u_i, u_j, 1:npz + 1])
           * (pkc[u_im1, u_j, 1:npz + 1] - pkc[u_i, u_j, 0:npz])))
    # :2125-2127 — note the asymmetric second gz term, verbatim
    vc = vc.at[v_i, v_j, 0:npz].set(
        vc[v_i, v_j, 0:npz]
        + dt2 * rdyc[v_i, v_j][:, :, None]
        / (wk[vw_i, vw_jm1, :] + wk[vw_i, vw_j, :])
        * ((gz[v_i, v_jm1, 1:npz + 1] - gz[v_i, v_j, 0:npz])
           * (pkc[v_i, v_j, 1:npz + 1] - pkc[v_i, v_jm1, 0:npz])
           + (gz[v_i, v_jm1, 0:npz] - gz[v_i, v_j, 1:npz + 1])
           * (pkc[v_i, v_jm1, 1:npz + 1] - pkc[v_i, v_j, 0:npz])))
    return uc, vc


def _check_a2b_origin(fname: str, bd, ng: int) -> None:
    """The a2b dummy declaration REBASES qin to ``(is-ng, js-ng)``.

    ``a2b_ord4(pk(isd,jsd,k), ...)`` passes the address of ``pk(isd,jsd)``
    to a dummy declared ``qin(is-ng:ie+ng, js-ng:je+ng)``, so the Fortran
    itself assumes ``isd == is-ng``.  If a caller's bounds/ng disagree,
    BOTH lanes read shifted data and return plausible numbers; here it is
    a loud error.
    """
    if bd.isd != bd.is_ - ng or bd.jsd != bd.js - ng:
        raise ValueError(
            f"{fname}: a2b_ord4's dummy is qin(is-ng:ie+ng, js-ng:je+ng), "
            f"so isd must equal is-ng (got isd={bd.isd}, is={bd.is_}, "
            f"jsd={bd.jsd}, js={bd.js}, ng={ng}); a mismatch reads "
            f"SHIFTED data on both lanes")


def one_grad_p(u, v, pk, gz, divg2, delp, gs: dict, bd, *, npx: int,
               npy: int, npz: int, dt: float, ptop: float, akap: float,
               hydrostatic: bool = True, a2b_ord: int = 4,
               d_ext: float = 0.0, ng: int | None = None,
               duogrid: bool = True, bounded_domain: bool = False,
               grid_type: int = 0, sw_corner: bool = True,
               se_corner: bool = True, nw_corner: bool = True,
               ne_corner: bool = True):
    """JAX twin of ``fv3_native_pgrad.one_grad_p`` (dyn_core.F90:2347-2480).

    Functional: the NumPy twin mutates ``u``, ``v``, AND ``pk``, ``gz``
    in place (``a2b_ord4`` is called with ``replace=.true.`` at :2399/
    :2409).  This twin RETURNS ``(u, v, pk, gz)`` in that order.  On
    return ``pk`` and ``gz`` hold B-GRID CORNER values on
    ``[is,ie+1] x [js,je+1]`` and A-grid values everywhere else.
    ``delp`` is NOT modified (:2456-2461 omits ``replace``) and, on the
    hydrostatic branch, is not even read -- the oracle proves that with a
    ``delp = -9.e9`` re-run, so ``None`` is accepted there.

    Storage convention (kept deliberately): ``pk``/``gz`` stay
    ``(m_a, m_a, npz+1)`` CELL-shaped planes reused as B-node storage
    after the replace.  That is valid because a2b writes only
    ``is..ie+1`` / ``js..je+1`` and ``ng >= 1`` leaves the slots.

    The a2b k-bounds DIFFER between the two fields and that is
    LOAD-BEARING: ``pk`` over k=2..npz+1 (:2397; k=1 is already the
    seeded ``top_value``), ``gz`` over k=1..npz+1 (:2407).  Both are
    issued as ONE vmapped call over the level axis (see
    :func:`_a2b_ord4_k`).

    ``duogrid`` selects a2b_ord4's duo (interior-everywhere) branch; it
    is an ARGUMENT here because this repo's ``a2b_ord4`` takes it as one,
    while upstream reads ``gridstruct%dg%is_initialized`` inside a2b.
    The NumPy twin's non-faithful ``bvertex_mean2`` screen is NOT ported
    (module docstring, deviation 3).
    """
    _require_bool("one_grad_p", "hydrostatic", hydrostatic)
    # Checked HERE, not only inside _a2b_ord4: a raise from inside the
    # vmap trace is harder to attribute, and these two select the
    # interior-everywhere arm vs the corner/edge arm -- a different set
    # of formulas, not a different number.
    _require_bool("one_grad_p", "duogrid", duogrid)
    _require_bool("one_grad_p", "bounded_domain", bounded_domain)
    if a2b_ord != 4:
        raise NotImplementedError(
            f"one_grad_p: a2b_ord={a2b_ord!r}; only the a2b_ord==4 arm is "
            "ported.  The a2b_ord2 else-arms (dyn_core.F90:2401/2411/2460) "
            "are linked in the Fortran extract purely so they resolve and "
            "are DEAD on this lane — there is no python a2b_ord2 and no "
            "fixture that would certify one")
    if not hydrostatic:
        raise NotImplementedError(
            "one_grad_p: the non-hydrostatic branch needs top_value=ptop "
            "(:2385) and an a2b_ord4(delp) wk (:2456-2461, no `replace`); "
            "neither is certified by any fixture")

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd, ied, jed = bd.isd, bd.jsd, bd.ied, bd.jed
    if ng is None:
        ng = bd.ng
    _check_a2b_origin("one_grad_p", bd, ng)

    _require_f64_jax("one_grad_p", {
        "u": u, "v": v, "pk": pk, "gz": gz, "divg2": divg2,
        "rdx": gs["rdx"], "rdy": gs["rdy"]})
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    pk = jnp.asarray(pk)
    gz = jnp.asarray(gz)
    divg2 = jnp.asarray(divg2)
    del delp  # hydrostatic branch never reads it (see docstring)

    # :2380-2386 — `ptk` (the `**` operator, :248/:2382), NOT exp(akap*log)
    top_value = ptop ** akap

    b_i = _w(isd, is_, ie + 1)
    b_j = _w(jsd, js, je + 1)

    # :2388-2393 — B box seed at k=1 ONLY
    pk = pk.at[b_i, b_j, 0].set(top_value)

    geom = a2b_gridstruct_view(gs, bd)
    flags = dict(bounded_domain=bounded_domain, grid_type=grid_type,
                 sw_corner=sw_corner, se_corner=se_corner,
                 ne_corner=ne_corner, nw_corner=nw_corner)

    # a2b's `qout` scratch: ONE local reused across every call upstream.
    # NaN outside the B box so an out-of-window read is LOUD (the duo
    # branch writes exactly [is,ie+1]x[js,je+1]; the plain branch's
    # corner/edge writes land inside that same box).
    def _a2b_replace(planes):
        scratch = jnp.full(
            (ied - isd + 1, jed - jsd + 1, planes.shape[2]), jnp.nan,
            dtype=jnp.float64)
        qin, _ = _a2b_ord4_k(planes, scratch, geom, npx, npy, is_, ie, js,
                             je, ng, replace=True, duogrid=duogrid,
                             **flags)
        return qin

    # READ-AFTER-WRITE ACROSS THE CALL BOUNDARY (docstring proof 3):
    # replace=.true. overwrites pk/gz on the B box, and every read below
    # -- the wk weight and the u/v brackets -- must see those OVERWRITTEN
    # B-grid corner values.  The rebound names carry them; nothing here
    # recomputes from the A-grid originals.
    pk = pk.at[:, :, 1:npz + 1].set(_a2b_replace(pk[:, :, 1:npz + 1]))
    gz = gz.at[:, :, 0:npz + 1].set(_a2b_replace(gz[:, :, 0:npz + 1]))

    # :2415-2443 — external-mode filter increments; both zero at d_ext<=0.
    # The windows are taken off is_/js so any bounds are representable.
    if d_ext > 0.0:
        wk2 = divg2[_w(is_, is_, ie), _w(js, js, je + 1)] \
            - divg2[_w(is_, is_ + 1, ie + 1), _w(js, js, je + 1)]
        wk1 = divg2[_w(is_, is_, ie + 1), _w(js, js, je)] \
            - divg2[_w(is_, is_, ie + 1), _w(js, js + 1, je + 1)]
    else:
        wk2 = jnp.zeros((ie - is_ + 1, je - js + 2), dtype=jnp.float64)
        wk1 = jnp.zeros((ie - is_ + 2, je - js + 1), dtype=jnp.float64)

    rdx = jnp.asarray(gs["rdx"])
    rdy = jnp.asarray(gs["rdy"])

    # u: i in is..ie, j in js..je+1   (:2464-2470)
    ui = _w(isd, is_, ie)
    uip1 = _w(isd, is_ + 1, ie + 1)
    uj = _w(jsd, js, je + 1)
    # v: i in is..ie+1, j in js..je   (:2471-2477)
    vi = _w(isd, is_, ie + 1)
    vj = _w(jsd, js, je)
    vjp1 = _w(jsd, js + 1, je + 1)

    # :2450-2455 — wk recomputed on the B box inside the Fortran k loop.
    # ALIASING (docstring proof 4): upstream and the NumPy twin reuse the
    # SAME buffer for a2b's qout scratch and for this weight.  The alias
    # is dead -- the whole B box is rewritten here before any read, and
    # the four reads below (ui/uip1 x uj, vi x vj/vjp1) all lie inside
    # it -- so allocating separately is equivalent.  NaN outside the box
    # keeps a window slip loud.
    # DEPENDENCE (the k nest): u/v are written at level k from pk/gz/wk
    # at levels k and k+1 of arrays this nest never writes.
    wk = jnp.full((ied - isd + 1, jed - jsd + 1, npz), jnp.nan,
                  dtype=jnp.float64)
    wk = wk.at[b_i, b_j, :].set(pk[b_i, b_j, 1:npz + 1]
                                - pk[b_i, b_j, 0:npz])

    # ASSIGNMENT (not an increment, unlike p_grad_c): the whole bracket
    # is multiplied by rdx/rdy.
    u = u.at[ui, uj, 0:npz].set(
        rdx[ui, uj][:, :, None] * (
            wk2[:, :, None] + u[ui, uj, 0:npz]
            + dt / (wk[ui, uj, :] + wk[uip1, uj, :]) * (
                (gz[ui, uj, 1:npz + 1] - gz[uip1, uj, 0:npz])
                * (pk[uip1, uj, 1:npz + 1] - pk[ui, uj, 0:npz])
                + (gz[ui, uj, 0:npz] - gz[uip1, uj, 1:npz + 1])
                * (pk[ui, uj, 1:npz + 1] - pk[uip1, uj, 0:npz]))))
    v = v.at[vi, vj, 0:npz].set(
        rdy[vi, vj][:, :, None] * (
            wk1[:, :, None] + v[vi, vj, 0:npz]
            + dt / (wk[vi, vj, :] + wk[vi, vjp1, :]) * (
                (gz[vi, vj, 1:npz + 1] - gz[vi, vjp1, 0:npz])
                * (pk[vi, vjp1, 1:npz + 1] - pk[vi, vj, 0:npz])
                + (gz[vi, vj, 0:npz] - gz[vi, vjp1, 1:npz + 1])
                * (pk[vi, vj, 1:npz + 1] - pk[vi, vjp1, 0:npz]))))
    return u, v, pk, gz


def nh_p_grad(u, v, pp, gz, delp, pk3, gs: dict, bd, *, npx: int,
              npy: int, npz: int, dt: float, ptop: float, akap: float,
              use_logp: bool = False, ng: int | None = None,
              duogrid: bool = True, bounded_domain: bool = False,
              grid_type: int = 0, sw_corner: bool = True,
              se_corner: bool = True, nw_corner: bool = True,
              ne_corner: bool = True):
    """JAX twin of ``fv3_native_pgrad.nh_p_grad`` (dyn_core.F90:2135-2230).

    The NH D-stage pressure update that replaces ``one_grad_p`` at
    ``beta = 0`` (the dispatch is ``beta < -0.1`` for one_grad_p, so 0
    lands HERE).

    Functional: the NumPy twin mutates ``u``, ``v`` and -- via
    ``a2b_ord4(replace=.true.)`` -- ``pp``, ``pk3``, ``gz`` in place.
    This twin RETURNS ``(u, v, pp, pk3, gz)`` in that order.  On return
    those three fields hold B-GRID CORNER values on
    ``[is,ie+1] x [js,je+1]`` (``pk3``/``pkc`` is perturbation-carrying
    A-grid input INTO the call and B-grid scratch AFTER it).  ``delp`` is
    read through a NO-replace a2b into a scratch and left unmodified.

    Windows (:2172-2181): the k=1 seed writes ``pp = 0`` and
    ``pk3 = top_value`` over the B box only; ``top_value`` is
    ``peln1 = log(ptop)`` under ``use_logp`` else ``ptk = ptop**akap``
    (the ``**`` operator, :246-248, matching one_grad_p).

    Per level (:2190-2230): ``wk`` = B-grid ``pk3`` interface difference
    (the hydrostatic weight), ``wk1`` = B-grid ``delp`` (the NH weight);
    u adds ``du1`` (hydrostatic form) plus the NH term in ``pp``, then
    multiplies by ``rdx``; v likewise with ``rdy``.  The grouping is
    copied exactly -- see the p_grad_c note on why regrouping changes the
    last bit.

    Cadence note: the Fortran issues ``a2b(pp_k), a2b(pk3_k), a2b(gz_k)``
    inside one ``do k`` loop.  Each call touches a DIFFERENT plane and
    the ``wk1`` scratch is overwritten per call, so the three batched
    calls here are level-for-level identical.
    """
    _require_bool("nh_p_grad", "use_logp", use_logp)
    _require_bool("nh_p_grad", "duogrid", duogrid)
    _require_bool("nh_p_grad", "bounded_domain", bounded_domain)
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd, ied, jed = bd.isd, bd.jsd, bd.ied, bd.jed
    if ng is None:
        ng = bd.ng
    _check_a2b_origin("nh_p_grad", bd, ng)

    _require_f64_jax("nh_p_grad", {
        "u": u, "v": v, "pp": pp, "gz": gz, "delp": delp, "pk3": pk3,
        "rdx": gs["rdx"], "rdy": gs["rdy"]})
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    pp = jnp.asarray(pp)
    gz = jnp.asarray(gz)
    delp = jnp.asarray(delp)
    pk3 = jnp.asarray(pk3)

    top_value = float(np.log(ptop)) if use_logp else ptop ** akap

    b_i = _w(isd, is_, ie + 1)
    b_j = _w(jsd, js, je + 1)

    geom = a2b_gridstruct_view(gs, bd)
    flags = dict(bounded_domain=bounded_domain, grid_type=grid_type,
                 sw_corner=sw_corner, se_corner=se_corner,
                 ne_corner=ne_corner, nw_corner=nw_corner)

    def _scratch(nk):
        return jnp.full((ied - isd + 1, jed - jsd + 1, nk), jnp.nan,
                        dtype=jnp.float64)

    def _a2b_replace(planes):
        qin, _ = _a2b_ord4_k(planes, _scratch(planes.shape[2]), geom, npx,
                             npy, is_, ie, js, je, ng, replace=True,
                             duogrid=duogrid, **flags)
        return qin

    # :2172-2185 — k=1 B-box seed for pp/pk3; a2b for k>=2; gz EVERY k.
    # READ-AFTER-WRITE ACROSS THE CALL BOUNDARY (docstring proof 3): the
    # replace overwrites pp/pk3/gz on the B box and the momentum loop
    # below reads THOSE values (B-grid), so every read goes to the
    # rebound name.  The three fields are independent arrays and a2b
    # touches only the plane it is handed, so issuing them as three
    # batched calls instead of interleaved-per-k changes nothing.
    pp = pp.at[b_i, b_j, 0].set(0.0)
    pk3 = pk3.at[b_i, b_j, 0].set(top_value)
    pp = pp.at[:, :, 1:npz + 1].set(_a2b_replace(pp[:, :, 1:npz + 1]))
    pk3 = pk3.at[:, :, 1:npz + 1].set(_a2b_replace(pk3[:, :, 1:npz + 1]))
    gz = gz.at[:, :, 0:npz + 1].set(_a2b_replace(gz[:, :, 0:npz + 1]))

    rdx = jnp.asarray(gs["rdx"])
    rdy = jnp.asarray(gs["rdy"])

    # u: i in is..ie, j in js..je+1 (:2196-2211)
    ui = _w(isd, is_, ie)
    uip1 = _w(isd, is_ + 1, ie + 1)
    uj = _w(jsd, js, je + 1)
    # v: i in is..ie+1, j in js..je (:2213-2228)
    vi = _w(isd, is_, ie + 1)
    vj = _w(jsd, js, je)
    vjp1 = _w(jsd, js + 1, je + 1)

    # DEPENDENCE (the k nest, :2190): u/v are written at level k from
    # pp/pk3/gz/wk/wk1 at levels k and k+1 of arrays this nest never
    # writes -> vectorises over k.  wk1 here IS a2b's output and is read
    # as such (unlike one_grad_p's wk, docstring proof 4).
    # :2191 — B-grid delp into wk1 (NO replace; delp unmodified)
    _, wk1 = _a2b_ord4_k(delp[:, :, 0:npz], _scratch(npz), geom, npx, npy,
                         is_, ie, js, je, ng, replace=None,
                         duogrid=duogrid, **flags)
    # :2192-2195 — hydrostatic weight from pk3 interface differences
    wk = _scratch(npz).at[b_i, b_j, :].set(
        pk3[b_i, b_j, 1:npz + 1] - pk3[b_i, b_j, 0:npz])

    # :2196-2211 — u: INCREMENT-then-scale, hydrostatic du1 + NH pp
    du1 = dt / (wk[ui, uj, :] + wk[uip1, uj, :]) * (
        (gz[ui, uj, 1:npz + 1] - gz[uip1, uj, 0:npz])
        * (pk3[uip1, uj, 1:npz + 1] - pk3[ui, uj, 0:npz])
        + (gz[ui, uj, 0:npz] - gz[uip1, uj, 1:npz + 1])
        * (pk3[ui, uj, 1:npz + 1] - pk3[uip1, uj, 0:npz]))
    u = u.at[ui, uj, 0:npz].set(
        (u[ui, uj, 0:npz] + du1
         + dt / (wk1[ui, uj, :] + wk1[uip1, uj, :]) * (
             (gz[ui, uj, 1:npz + 1] - gz[uip1, uj, 0:npz])
             * (pp[uip1, uj, 1:npz + 1] - pp[ui, uj, 0:npz])
             + (gz[ui, uj, 0:npz] - gz[uip1, uj, 1:npz + 1])
             * (pp[ui, uj, 1:npz + 1] - pp[uip1, uj, 0:npz]))
         ) * rdx[ui, uj][:, :, None])

    # :2213-2228 — v
    dv1 = dt / (wk[vi, vj, :] + wk[vi, vjp1, :]) * (
        (gz[vi, vj, 1:npz + 1] - gz[vi, vjp1, 0:npz])
        * (pk3[vi, vjp1, 1:npz + 1] - pk3[vi, vj, 0:npz])
        + (gz[vi, vj, 0:npz] - gz[vi, vjp1, 1:npz + 1])
        * (pk3[vi, vj, 1:npz + 1] - pk3[vi, vjp1, 0:npz]))
    v = v.at[vi, vj, 0:npz].set(
        (v[vi, vj, 0:npz] + dv1
         + dt / (wk1[vi, vj, :] + wk1[vi, vjp1, :]) * (
             (gz[vi, vj, 1:npz + 1] - gz[vi, vjp1, 0:npz])
             * (pp[vi, vjp1, 1:npz + 1] - pp[vi, vj, 0:npz])
             + (gz[vi, vj, 0:npz] - gz[vi, vjp1, 1:npz + 1])
             * (pp[vi, vj, 1:npz + 1] - pp[vi, vjp1, 0:npz]))
         ) * rdy[vi, vj][:, :, None])
    return u, v, pp, pk3, gz


def _require_column_depth(fname: str, delp, npz: int) -> None:
    """``delp`` must carry at least ``npz`` levels.

    A shorter array would be SILENTLY CLIPPED by ``delp[..., 0:npz]`` and
    the column integral would run over fewer levels than asked -- a
    plausible number, not an error.  (The write would then fail on a
    shape mismatch, but far from the cause.)
    """
    n = jnp.asarray(delp).shape[2]
    if n < npz:
        raise ValueError(
            f"{fname}: delp has {n} levels but npz={npz}; the column "
            f"integral would be silently clipped")


def _hydro_col(delp_blk, ptop: float):
    """Sequential top-down column integral ``pe(k+1) = pe(k) + delp(k)``.

    ``delp_blk`` is ``(..., npz)``; returns ``(..., npz)`` holding the
    running pressure AFTER each add (the Fortran's ``pei``/``pet`` scalar
    at the end of iteration k).  ``lax.scan``, never ``cumsum``: the
    accumulator order is part of the parity contract.
    """
    def _step(pei, dp_k):
        pei = pei + dp_k
        return pei, pei

    init = jnp.full(delp_blk.shape[:-1], ptop, dtype=delp_blk.dtype)
    _, out = lax.scan(_step, init, jnp.moveaxis(delp_blk, -1, 0))
    return jnp.moveaxis(out, 0, -1)


def pk3_halo(pk3, delp, bd, *, npz: int, ptop: float, akap: float):
    """JAX twin of ``fv3_native_pgrad.pk3_halo`` (dyn_core.F90:1832-1884).

    Functional: RETURNS the updated ``pk3``.  Locally rebuilds the TWO
    x-rings (i in {is-2, is-1, ie+1, ie+2}, j = js..je) and TWO y-rings
    (j in {js-2, js-1, je+1, je+2}, i = is-2..ie+2) from halo ``delp``.
    This is a local recomputation, NOT a halo exchange -- replacing it
    with an exchange changes both the arithmetic and the corner coverage.
    Level 1 (the top interface) is NEVER written here.

    DEPENDENCE: the Fortran interleaves two columns per k loop
    (``pei(is-2)`` and ``pei(is-1)`` advance together), but each column
    carries its OWN accumulator and reads only its own ``delp`` -- the
    columns are independent, so a ring integrates as one batched
    ``lax.scan`` whose only sequential axis is k.  The x-ring set
    (j = js..je) and the y-ring set (j in {js-2, js-1, je+1, je+2}) are
    DISJOINT in j, so no column is written twice and the write order
    cannot matter.
    """
    _require_f64_jax("pk3_halo", {"pk3": pk3, "delp": delp})
    _require_column_depth("pk3_halo", delp, npz)
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd
    pk3 = jnp.asarray(pk3)
    delp = jnp.asarray(delp)

    jw = _w(jsd, js, je)
    for i in (is_ - 2, is_ - 1, ie + 1, ie + 2):
        pei = _hydro_col(delp[i - isd, jw, 0:npz], ptop)
        pk3 = pk3.at[i - isd, jw, 1:npz + 1].set(
            jnp.exp(akap * jnp.log(pei)))

    iw = _w(isd, is_ - 2, ie + 2)
    for j in (js - 2, js - 1, je + 1, je + 2):
        pej = _hydro_col(delp[iw, j - jsd, 0:npz], ptop)
        pk3 = pk3.at[iw, j - jsd, 1:npz + 1].set(
            jnp.exp(akap * jnp.log(pej)))
    return pk3


def pln_halo(pk3, delp, bd, *, npz: int, ptop: float):
    """JAX twin of ``fv3_native_pgrad.pln_halo`` (dyn_core.F90:1886-1933).

    The ``use_logp`` sibling of :func:`pk3_halo` (``log(p)`` rings
    instead of ``p**kappa``); same rings, same "never writes level 1"
    contract.  Functional: RETURNS the updated ``pk3``.  Dead on the
    pinned deck (USE_LOGP=F) but ten lines away.
    """
    _require_f64_jax("pln_halo", {"pk3": pk3, "delp": delp})
    _require_column_depth("pln_halo", delp, npz)
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd
    pk3 = jnp.asarray(pk3)
    delp = jnp.asarray(delp)

    jw = _w(jsd, js, je)
    for i in (is_ - 2, is_ - 1, ie + 1, ie + 2):
        pet = _hydro_col(delp[i - isd, jw, 0:npz], ptop)
        pk3 = pk3.at[i - isd, jw, 1:npz + 1].set(jnp.log(pet))

    iw = _w(isd, is_ - 2, ie + 2)
    for j in (js - 2, js - 1, je + 1, je + 2):
        pet = _hydro_col(delp[iw, j - jsd, 0:npz], ptop)
        pk3 = pk3.at[iw, j - jsd, 1:npz + 1].set(jnp.log(pet))
    return pk3


def pe_halo(pe, delp, bd, *, npz: int, ptop: float):
    """JAX twin of ``fv3_native_pgrad.pe_halo`` (dyn_core.F90:1935-1963).

    Functional: RETURNS the updated ``pe``.  Fills the ONE-ring edges --
    i in {is-1, ie+1} for j = js..je, then j in {js-1, je+1} for
    i = is-1..ie+1 -- by local hydrostatic integration of halo ``delp``,
    seeding ``pe(.,1,.) = ptop``.  ``pe`` is the oracle's
    ``(is-1:ie+1, npz+1, js-1:je+1)`` (i, k, j) array; ``delp`` is the
    padded (i, j, k) A-grid field.  Runs only on remap substeps
    (:1441-1442).

    DEPENDENCE: the two ring sets are DISJOINT -- the first loop's j
    window is js..je and the second's j values are js-1/je+1 -- so no
    column is written twice and the write order cannot matter (source
    order is preserved regardless).  Each column carries its own ``pe``
    accumulator, so the columns are independent and only the k sum is
    sequential (``lax.scan``).
    """
    _require_f64_jax("pe_halo", {"pe": pe, "delp": delp})
    _require_column_depth("pe_halo", delp, npz)
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd
    pe = jnp.asarray(pe)
    delp = jnp.asarray(delp)

    def _col(dp_blk):
        """(..., npz) -> (..., npz+1) with ptop in slot 0."""
        head = jnp.full(dp_blk.shape[:-1] + (1,), ptop,
                        dtype=dp_blk.dtype)
        return jnp.concatenate([head, _hydro_col(dp_blk, ptop)], axis=-1)

    jw_src = _w(jsd, js, je)
    jw_dst = _w(js - 1, js, je)
    for i in (is_ - 1, ie + 1):
        col = _col(delp[i - isd, jw_src, 0:npz])           # (nj, npz+1)
        pe = pe.at[i - (is_ - 1), :, jw_dst].set(col.T)

    iw_src = _w(isd, is_ - 1, ie + 1)
    iw_dst = _w(is_ - 1, is_ - 1, ie + 1)
    for j in (js - 1, je + 1):
        col = _col(delp[iw_src, j - jsd, 0:npz])           # (ni, npz+1)
        pe = pe.at[iw_dst, :, j - (js - 1)].set(col)
    return pe


# =====================================================================
# jit policies -- ONE per public routine (the production entry point and
# any instrumented test wrapper are built HERE, so a policy drift cannot
# pass unnoticed).  No donated buffers anywhere (grad-path doctrine).
# `bd` is a Bounds NamedTuple: hashable BY VALUE, so a fresh-but-equal
# bounds object shares one cache entry instead of retracing.
# =====================================================================

def make_geopk_jit(fn=geopk):
    """Static: bd + every keyword (deck constants and #ifdef/branch
    selectors).  delp/pt/hs/q_con dynamic."""
    return jax.jit(
        fn, static_argnums=(3,),
        static_argnames=("km", "ptop", "akap", "cp_air", "cg", "duogrid",
                         "computehalo", "npx", "npy", "a2b_ord",
                         "bounded_domain", "sw_dynamics", "use_cond",
                         "unwritten_fill"))


geopk_jit = make_geopk_jit()


def make_p_grad_c_jit(fn=p_grad_c):
    """Static: dt2 (deck constant -- a different dt intentionally
    recompiles), bd, npz, hydrostatic.  delpc/pkc/gz/uc/vc/gs dynamic."""
    return jax.jit(fn, static_argnums=(0, 7),
                   static_argnames=("npz", "hydrostatic"))


p_grad_c_jit = make_p_grad_c_jit()


def make_one_grad_p_jit(fn=one_grad_p):
    """Static: bd plus every keyword -- deck constants (npx/npy/npz/dt/
    ptop/akap/d_ext/ng) and branch selectors (hydrostatic/a2b_ord/
    duogrid/bounded_domain/grid_type/the four corner flags).
    u/v/pk/gz/divg2/delp/gs dynamic."""
    return jax.jit(
        fn, static_argnums=(7,),
        static_argnames=("npx", "npy", "npz", "dt", "ptop", "akap",
                         "hydrostatic", "a2b_ord", "d_ext", "ng",
                         "duogrid", "bounded_domain", "grid_type",
                         "sw_corner", "se_corner", "nw_corner",
                         "ne_corner"))


one_grad_p_jit = make_one_grad_p_jit()


def make_nh_p_grad_jit(fn=nh_p_grad):
    """Same policy as :func:`make_one_grad_p_jit`, with ``use_logp`` in
    place of ``hydrostatic``/``a2b_ord``/``d_ext``."""
    return jax.jit(
        fn, static_argnums=(7,),
        static_argnames=("npx", "npy", "npz", "dt", "ptop", "akap",
                         "use_logp", "ng", "duogrid", "bounded_domain",
                         "grid_type", "sw_corner", "se_corner",
                         "nw_corner", "ne_corner"))


nh_p_grad_jit = make_nh_p_grad_jit()


def make_pk3_halo_jit(fn=pk3_halo):
    """Static: bd, npz, ptop, akap.  pk3/delp dynamic."""
    return jax.jit(fn, static_argnums=(2,),
                   static_argnames=("npz", "ptop", "akap"))


pk3_halo_jit = make_pk3_halo_jit()


def make_pln_halo_jit(fn=pln_halo):
    """Static: bd, npz, ptop.  pk3/delp dynamic."""
    return jax.jit(fn, static_argnums=(2,),
                   static_argnames=("npz", "ptop"))


pln_halo_jit = make_pln_halo_jit()


def make_pe_halo_jit(fn=pe_halo):
    """Static: bd, npz, ptop.  pe/delp dynamic."""
    return jax.jit(fn, static_argnums=(2,),
                   static_argnames=("npz", "ptop"))


pe_halo_jit = make_pe_halo_jit()
