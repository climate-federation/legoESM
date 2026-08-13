"""FV3 ``sw_core`` shallow-water core -- JAX lane (``c_sw`` + the six
``d_sw`` stages, duo branch, plus the plain leaves ``c_sw`` needs).

Functional, jit-compatible, differentiable mirror of the certified NumPy
fp64 lane (``fv3_native_sw_core.py`` + ``fv3_native_duo_sw_core.py``,
themselves loop-faithful ports of the pinned oracle).  **The NumPy lane
is the SPECIFICATION for this module** (hop B of the authority chain in
``docs/atmosphere/fv3_duo_jax_lane_strategy.md``); every kernel here is
certified AGAINST the NumPy twin, never against the Fortran directly.

Verified line numbers in the PINNED oracle tree
``/burg-archive/glab/users/pg2328/fv3_oracle_pinned/
atmos_cubed_sphere-symmetryclean`` -- every number below was read with
``grep``/``sed`` on that tree while writing this module (R7).  The
INSTRUMENTED copy under ``fv3_recon/duo_model/`` is never cited; its
``dyn_core.F90`` is +102 lines (``sw_core.F90`` happens to be identical
across the copies, but the rule is one tree):

* ``model/sw_core.F90`` -- ``public :: c_sw, d_sw1, d_sw2, d_sw3, d_sw4,
  d_sw5, d_sw6, fill_4corners, del6_vt_flux, divergence_corner,
  divergence_corner_nest`` at :74; module constants ``near_zero`` :39,
  ``big_number = 1.E30`` (non-``OVERLOAD_R4``) :43, ``p1``/``p2``
  :48-49, ``a1 = 0.5625`` :53, ``a2 = -0.0625`` :54, ``c1 = -2./14.``
  :57, ``c2 = 11./14.`` :58, ``c3 = 5./14.`` :59;
  ``c_sw`` :79-494 (``dt4 = 0.5*dt2`` :367), ``d_sw1`` :500-998
  (duo ut/vt interior :621-634, the ``.not.bounded .or. .not.duogrid``
  panel-edge guard :656, the corner ``1./(1.-0.0625*cosa_u*cosa_v)``
  systems :740-811, ``xfx_adv = dt*ut`` :830, ``ra_x`` :875-884, the
  ``delp`` transport :886-887, the ``pt`` transport with
  ``nord=nord_v, damp_c=damp_v`` :959-961), ``d_sw2`` :1000-1199,
  ``d_sw3`` :1201-1388 (``dt5 = 0.5*dt`` :1257), ``d_sw4`` :1390-1472
  (``dt6 = dt/6.`` :1442), ``d_sw5`` :1474-1869
  (``dd8 = (da_min_c*d4_bg)**n2`` :1811, ``damp2 = da_min_c*max(d2_bg,
  min(0.20, dddmp*vort))`` :1816, ``dddmp<1.E-5`` :1790, ``d_con >
  1.e-5`` :1824), ``d_sw6`` :1872-2006 (``damp_v>1.E-5`` :1948 / :1989,
  ``d_con > 1.e-5`` :1953), ``del6_vt_flux`` :2008-2121,
  ``divergence_corner`` :2124-2229, ``divergence_corner_duo``
  :2345-2447, ``xtp_u`` :2540-2894, ``ytp_v`` :2897-3353,
  ``d2a2c_vect`` :3361-3706, ``fill2_4corners`` :3794-3854,
  ``fill_4corners`` :3856-3915.
* ``model/fv_arrays.F90`` -- ``grid_type`` values :324-330 and the
  "2, 3, 6 and 7 are not supported" note :292-297; the duo implication
  ``bounded_domain = regional .or. nested .or. duogrid`` :1512.

Every numeric constant used here is IMPORTED from the NumPy twin's
module (``fv3_native_sw_core``) or carries the ``file.F90:line`` + the
literal value read there, so a constant cannot drift between lanes (R2).

**R3 -- the six-stage decomposition is load-bearing.**  The oracle keeps
``d_sw`` as six public routines and splits dyn_core's k-loop into four
nests (``dyn_core.F90:744, 914, 1066, 1218``) precisely so the two duo
flux-averaging barriers have somewhere to live: ``dyn_core.F90:872``
(``CGRID_NE``, after ``d_sw1``) and ``:984`` (``BGRID_NE``, on the
B-grid corner velocities).  A monolithic ``d_sw`` cannot express those
barriers.  The monolithic JAX ``_d_sw_native`` (``fv3_sw_core.py:3359``)
is a different, cdgrid-signature research solver: it is NOT the target
of this port, it is not reused here, and it is not touched.

Mirror doctrine (R1-R5):

* **Functional (R4)**: the NumPy lane's ``c_sw``/``d_sw*_duo`` already
  copy every input at entry and return dicts, so R4 costs nothing there
  -- confirmed per routine rather than assumed (``fv3_native_sw_core.py``
  :520-522 and ``fv3_native_duo_sw_core.py`` :273-281, :642-643,
  :750-753, :830-834, :926-936, :1118-1125; ``d_sw2_duo`` copies
  ``delp``/``pt``/``w`` and slices ``allflux_*`` into fresh arrays).
  The leaves ``d2a2c_vect``/``divergence_corner``/``fill*_4corners`` are
  the exception: the two ``fill*`` routines MUTATE their operands in the
  NumPy lane, so their JAX twins RETURN them and ``c_sw`` threads the
  result.
* **Explicit x64**: every float operand must arrive float64; a float32
  operand raises ``TypeError`` at entry.  The check reads only static
  dtypes, so it is jit-safe.
* **No ``donate_argnums``** anywhere in this lane (grad-path doctrine).
* **Static index bounds**: ``bd``/``npx``/``npy``/every scheme selector
  is a STATIC python value; ``_fw``/``_fs`` turn each Fortran window
  into a python slice and RAISE if it escapes the declared bounds (a
  negative python start would otherwise WRAP silently -- the "halo
  loop-bound bug" of strategy section 8).
* **Fill values mirror the twin exactly**, because they are part of the
  contract: NaN where the twin uses NaN (a tripwire), ``0.0`` where the
  plain ``d2a2c_vect`` zero-initialises, ``big_number`` where the duo
  ``d2a2c_vect``/``d_sw1`` workspace does, and ``1.0e25`` for
  ``divergence_corner_duo``'s ``divg_d``.  ``d_sw1``'s ut/vt sentinel is
  NOT decoration: the plain-conventions panel-edge/corner blocks READ
  ut/vt cells the duo interior never writes, and both lanes initialise
  them to ``workspace_sentinel`` so those cells are deterministic and
  bit-comparable (see ``d_sw1_duo``'s twin docstring).

**R1a -- vectorisation requires a proven-independent loop.**  An i/j
loop is sliced here only where no iteration reads a location another
iteration writes (including through the NumPy lane's ``fort`` 1-based
views onto the same buffer); the dependence argument is stated in a
comment at every vectorised site.  Where a real read-after-write
recurrence exists it stays an ORDERED python chain over a STATIC trip
count:

* ``d_sw5_duo``'s ``do n=1,nord`` divergence-damping pass loop
  (sw_core.F90:1738-1788) is a RECURRENCE -- pass ``n`` reads the
  ``divg_d`` pass ``n-1`` wrote.  It is an unrolled ordered chain, never
  vectorised and never a ``scan`` (the window ``nt = nord - n`` shrinks
  each pass, so the bodies are not the same shape).
* ``del6_vt_flux``'s ``do n=1,nord`` loop is the same shape of
  recurrence and is treated identically.
* ``d_sw1_duo``'s four panel-edge blocks and four corner 2x2 systems
  are an ORDERED chain: the South ``ut`` block reads ``vt`` cells the
  South ``vt`` block just wrote, and the corner systems read ``ut``/
  ``vt`` cells the edge blocks wrote.
* ``d2a2c_vect``'s corner overrides run BEFORE the edge blocks that read
  them (``utmp``'s corner writes feed ``vtmp``'s, and ``UA``'s feed the
  ``edge_interpolate4`` columns), so the blocks are an ordered chain.

**R1b -- ``jnp.where`` is a select, not lazy control flow.**  Both arms
are evaluated, so a division in the DEAD arm poisons the reverse-mode
gradient of the live one (``NaN * 0 = NaN`` through the ``where`` VJP).
The rule applied here:

1. **A branch on a LOOP INDEX is STATIC and never becomes a ``where``.**
   Every ``if (i==1 .or. i==npx)`` / ``if (j==1 .or. j==npy)`` /
   ``if (j==1) ... elseif ...`` chain in ``c_sw``, ``divergence_corner``
   and ``d2a2c_vect`` is resolved at trace time into disjoint index runs
   (``_runs``) and each formula is applied on its own run.  This is both
   more literal AND removes the hazard: the ``dt2*(v - uc*cosa_u)/
   sina_u`` form (sw_core.F90:420-434) is never evaluated at the panel
   edges where the oracle replaces it, so a degenerate ``sina_u`` there
   can neither produce a NaN nor reach a gradient.
2. **A branch on DATA becomes a ``jnp.where`` only when both arms are
   total and finite on every admitted input.**  The sites, all upwind or
   sign selections whose two arms are plain gathers or products:
   ``c_sw``'s ut/vt scaling and its delp/pt/w upwind fluxes; the KE and
   vorticity upwind selects; the ``fy1>0``/``fx1>0`` vorticity-flux
   selects; ``d2a2c_vect``'s four ``ut/vt > 0`` ``sin_sg`` selects;
   ``d_sw1_duo``'s ``crx``/``cry`` upwind pair and the four
   ``uc*dt > 0`` / ``vc*dt > 0`` panel-edge selects.
   The two panel-edge families are the only ``where``s whose arms
   DIVIDE: ``uc(1,j)/sin_sg(0,j,3)`` vs ``uc(1,j)/sin_sg(1,j,1)``
   (sw_core.F90:658-663).  Both arms are total on an admitted
   gridstruct, where ``sin_sg`` is the sine of the angle between grid
   lines and is strictly positive; this is a stated PRECONDITION of the
   operand, not an assumption about the selected branch, and it cannot
   be checked statically because ``sin_sg`` is traced.  Note also that
   the two arms divide by DIFFERENT metric entries, so masking one would
   change the live answer, not just the dead one.

Non-smooth sites (each only C^0, or discontinuous, across the named
surface) -- the gradient gates target these explicitly:

1. every upwind select above, at its ``> 0`` surface: the VALUE is
   generally NOT continuous there (the two arms read different cells),
   so the flux itself jumps -- these are discontinuities, not kinks;
2. ``d_sw5_duo``'s ``sqrt(delpc**2 + vort**2)`` (sw_core.F90:1798) is
   non-differentiable at ``delpc = vort = 0``;
3. ``d_sw5_duo``'s ``max(d2_bg, min(0.20, dddmp*vort))``
   (sw_core.F90:1816) kinks at both clamp ties;
4. everything ``fv_tp_2d``/``xtp_u``/``ytp_v`` inherit from the PPM
   limiters (``fv3_tp_core``'s docstring enumerates them) -- those
   surfaces are DISCONTINUOUS, so a rounding-level lane difference near
   one produces a discrepancy far above 1e-15 and the tolerance for any
   gate that crosses one must be quoted separately.

**Gridstruct convention.**  The NumPy lane hands every kernel one
``gridstruct`` dict mixing traced ARRAYS with python bools and floats.
That dict cannot be a jit operand at all (unhashable as static; the
bools would be traced as dynamic), so it is SPLIT exactly as
``fv3_tp_core`` splits it:

* ``gs`` -- a dict of float64 arrays only, a normal traced pytree;
* ``flags`` -- a :class:`GridFlags` NamedTuple carrying ``da_min``,
  ``da_min_c``, ``bounded_domain``, ``grid_type`` and the four corner
  flags.  It is hashable BY VALUE, so a fresh-but-equal instance shares
  one jit cache entry (same doctrine as ``Bounds``).

``GridFlags.from_gs`` reconstructs it from the NumPy lane's dict, so a
parity test is one line.  This is a signature adaptation with no
arithmetic effect.

**What is NOT re-implemented here.**  ``fv_tp_2d``, ``xtp_u``, ``ytp_v``
and ``copy_corners`` are imported from ``legoesm.core.fv3_tp_core``;
``deln_flux`` and ``pert_ppm`` live there too and are reached THROUGH
``fv_tp_2d``/``xppm``/``yppm`` rather than called directly from this
module.  ``a2b_ord4`` is imported from ``legoesm.core.fv3_pgrad``.  None
of them is duplicated.

**``del6_vt_flux`` ownership.**  It is a ``sw_core`` routine, so its
natural home is this module and it is PUBLIC here.  A private copy
currently lives at ``fv3_nh_core._del6_vt_flux``; that copy should be
deleted in favour of this one (a cross-module private import is banned
by the CI ratchet, which is why it was duplicated).  This module does
not edit ``fv3_nh_core.py``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.core.fv3_native_sw_core import (
    A1,
    A2,
    BIG_NUMBER,
    C1,
    C2,
    C3,
)
from legoesm.core.fv3_pgrad import a2b_ord4
from legoesm.core.fv3_tp_core import copy_corners, fv_tp_2d, xtp_u, ytp_v

__all__ = [
    "GridFlags",
    "edge_interpolate4",
    "fill_4corners",
    "fill2_4corners",
    "d2a2c_vect",
    "d2a2c_vect_duo",
    "divergence_corner",
    "divergence_corner_duo",
    "del6_vt_flux",
    "c_sw",
    "d_sw1_duo",
    "d_sw2_duo",
    "d_sw3_duo",
    "d_sw4_duo",
    "d_sw5_duo",
    "d_sw6_duo",
    "make_edge_interpolate4_jit",
    "make_fill_4corners_jit",
    "make_fill2_4corners_jit",
    "make_d2a2c_vect_jit",
    "make_d2a2c_vect_duo_jit",
    "make_divergence_corner_jit",
    "make_divergence_corner_duo_jit",
    "make_del6_vt_flux_jit",
    "make_c_sw_jit",
    "make_d_sw1_duo_jit",
    "make_d_sw2_duo_jit",
    "make_d_sw3_duo_jit",
    "make_d_sw4_duo_jit",
    "make_d_sw5_duo_jit",
    "make_d_sw6_duo_jit",
]

# ---------------------------------------------------------------------
# Oracle literals that are NOT already exported by the NumPy twin.
# (A1/A2/C1/C2/C3/BIG_NUMBER are imported above -- sw_core.F90:53-54,
# :57-59 and :43 -- so they cannot drift between the two lanes.)
# ---------------------------------------------------------------------

# sw_core.F90:740 -- `damp = 1. / (1.-0.0625*cosa_u(2,0)*cosa_v(1,0))`,
# the d_sw1 cube-corner 2x2 systems.  |cosa| <= 1 so the denominator is
# bounded below by 1 - 0.0625 > 0 and the division is always total.
_CORNER_DAMP_COEF = 0.0625

# sw_core.F90:1816 -- `min(0.20, dddmp*vort(i,j))` in d_sw5's del-2
# damping coefficient.
_DDDMP_CAP = 0.20

# sw_core.F90:1790 / :1824 / :1948 / :1953 / :1989 -- the four "is this
# knob switched on" thresholds, read verbatim at those lines.
_DDDMP_OFF = 1.0e-5      # `if ( dddmp<1.E-5 )`
_D_CON_ON = 1.0e-5       # `if ( d_con > 1.e-5 )`
_DAMP_V_ON = 1.0e-5      # `if ( damp_v>1.E-5 )`
_DAMP_W_ON = 1.0e-5      # d_sw2's `if ( damp_w>1.E-5 )`

# fv_arrays.F90:324-330 documents grid_type -1 (read from file), 0 (ED
# gnomonic), 1, 2, 3 (lat-lon, "to be implemented"), 4 (doubly-periodic
# cartesian) and 5 (user-defined orthogonal); :292-297 adds that 2, 3, 6
# and 7 "are not supported and will likely not run".  Dispatch-hardening
# (CLAUDE.md): a value outside this set must RAISE at entry rather than
# silently take the `grid_type < 3` arm -- e.g. a numpy float `3.0`
# compares `< 3` as False and would run the doubly-periodic branch on a
# cubed sphere.
_GRID_TYPES = (-1, 0, 1, 2, 3, 4, 5)

# Scheme selectors.  These sets are RESTATED (not imported) because
# ``fv3_tp_core._PPM_ORDS`` / ``._SW_ORDS`` are private and CLAUDE.md's
# ratchet forbids a cross-module private import; the test file asserts
# they are equal to tp_core's, so a drift goes red rather than silent.
# Derivation is tp_core's: the NumPy lane's xppm/yppm branch on
# ``abs(iord)`` below 7 and on the raw value at/above it, while
# xtp_u/ytp_v branch on the RAW iord (no ``abs``), so negatives are not
# a scheme there.
_PPM_ORDS = (-6, -5, -4, -3, -2, -1,
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
_SW_ORDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)


class GridFlags(NamedTuple):
    """Static (non-traced) half of the NumPy lane's ``gridstruct``.

    Hashable BY VALUE, so an equal-but-fresh instance shares one jit
    cache entry -- the same doctrine ``Bounds`` follows.  Every field is
    read in a PYTHON branch or a python power somewhere in this module
    (``da_min``/``da_min_c`` feed ``(damp_c*da_min)**(nord+1)`` and
    ``(da_min_c*d4_bg)**(nord+1)``; the rest select formulas), which is
    exactly why they cannot be traced.
    """

    da_min: float = 0.0
    da_min_c: float = 0.0
    bounded_domain: bool = False
    grid_type: int = 0
    sw_corner: bool = True
    se_corner: bool = True
    nw_corner: bool = True
    ne_corner: bool = True

    @classmethod
    def from_gs(cls, gs: dict) -> GridFlags:
        """Build from the NumPy lane's mixed ``gridstruct`` dict.

        Defaults match the NumPy twins' own ``gs.get(..., default)``
        calls so a dict that predates a flag behaves identically in both
        lanes.
        """
        return cls(
            da_min=float(gs.get("da_min", 0.0)),
            da_min_c=float(gs.get("da_min_c", 0.0)),
            bounded_domain=bool(gs.get("bounded_domain", False)),
            grid_type=int(gs.get("grid_type", 0)),
            sw_corner=bool(gs.get("sw_corner", True)),
            se_corner=bool(gs.get("se_corner", True)),
            nw_corner=bool(gs.get("nw_corner", True)),
            ne_corner=bool(gs.get("ne_corner", True)),
        )


# ---------------------------------------------------------------------
# entry gates
# ---------------------------------------------------------------------

def _require_f64_jax(fname: str, arrays: dict) -> None:
    """Static-dtype gate mirroring the NumPy lane's ``_require_f64``.

    Reads only ``.dtype`` (static under jit): a float32 operand would
    otherwise be silently upcast -- or worse, with ``jax_enable_x64``
    disabled the whole core would silently run in float32 -- and the
    oracle build is ``-fdefault-real-8``.

    (Byte-identical in intent to ``fv3_tp_core._require_f64_jax`` and
    ``fv3_nh_core._require_f64_jax``.  It is re-stated rather than
    imported because CLAUDE.md forbids importing a private symbol across
    modules; promoting one shared public helper is an edit to another
    lane's file and is flagged as follow-up debt, not done here.)
    """
    for name, a in arrays.items():
        if a is None:
            continue
        if jnp.asarray(a).dtype != jnp.float64:
            raise TypeError(
                f"{fname}: {name} must be float64 (got "
                f"{jnp.asarray(a).dtype}); enable jax_enable_x64 and pass "
                f"f64 operands (oracle build is -fdefault-real-8)")


def _validate_ord(fname: str, argname: str, value, allowed) -> None:
    """Dispatch-hardening guard on a STATIC scheme selector."""
    if value not in allowed:
        raise ValueError(
            f"{fname}: {argname}={value!r} is not a supported scheme. "
            f"Supported: {sorted(allowed)}. An unsupported order would "
            f"silently fall through to a neighbouring branch in the "
            f"NumPy lane and run different numerics.")


def _validate_grid_type(fname: str, grid_type) -> None:
    """Dispatch-hardening guard on ``grid_type`` (fv_arrays.F90:324-330).

    Every consumer here branches on ``grid_type < 3`` or ``> 3``; an
    out-of-set value (or a float) would take one of those arms silently
    and run a DIFFERENT grid's formulas, so it raises instead.
    """
    if not isinstance(grid_type, int) or isinstance(grid_type, bool) \
            or grid_type not in _GRID_TYPES:
        raise ValueError(
            f"{fname}: grid_type={grid_type!r} is not a supported grid "
            f"(fv_arrays.F90:324-330 defines {list(_GRID_TYPES)}). Every "
            f"branch in this module tests `grid_type < 3` or `> 3`, so an "
            f"unknown value would silently select a different grid's "
            f"formulas.")


def _validate_nord(fname: str, argname: str, nord) -> None:
    """``nord`` is a LOOP TRIP COUNT and a set of index windows."""
    if not isinstance(nord, int) or isinstance(nord, bool) or nord < 0:
        raise ValueError(
            f"{fname}: {argname}={nord!r} must be a non-negative python "
            f"int (tp_core.F90:1219-1221 / sw_core.F90:2010-2014 define "
            f"0 = del-2, 1 = del-4, 2 = del-6; it is a loop trip count "
            f"and a set of index windows, never a traced value).")


def _require_bool(fname: str, name: str, v) -> None:
    if not isinstance(v, bool):
        raise ValueError(
            f"{fname}: {name}={v!r} is not a python bool. This flag "
            f"selects a Fortran branch (a different set of formulas) and "
            f"is STATIC in this lane; pass True or False.")


def _geom(fname: str, gs: dict, keys) -> dict:
    """Extract + f64-check the geometry arrays a routine associates.

    Mirrors ``fv3_pgrad.a2b_gridstruct_view``: a MISSING key raises with
    the full list rather than letting a ``.get`` default a metric to
    something plausible.
    """
    missing = [k for k in keys if k not in gs]
    if missing:
        raise KeyError(
            f"{fname}: gridstruct is missing {missing}; this routine "
            f"associates all of {list(keys)}")
    out = {k: jnp.asarray(gs[k]) for k in keys}
    _require_f64_jax(fname, out)
    return out


def _check_shape(fname: str, name: str, a, want) -> None:
    """Tier-0 gate: an operand at the wrong Fortran bounds would slice a
    valid-looking window out of the wrong array."""
    got = tuple(jnp.asarray(a).shape)
    if got != tuple(want):
        raise ValueError(
            f"{fname}: {name} must have shape {tuple(want)} (its Fortran "
            f"declared bounds), got {got}")


# ---------------------------------------------------------------------
# Fortran-window helpers (the read/write halves of the NumPy lane's
# ``fort`` view, made functional and bound-checked)
# ---------------------------------------------------------------------

def _fw(a, ilo, jlo, i0, i1, j0, j1, k=None):
    """Read ``a(i0:i1, j0:j1)`` -- INCLUSIVE, as a Fortran ``do``.

    ``ilo``/``jlo`` are the Fortran indices of storage element ``[0,0]``.
    Unlike ``fort.__getitem__`` (which evaluates ``a[i - ilo]`` and so
    silently WRAPS on a negative result) an out-of-storage window RAISES.
    Every bound is a STATIC python int, so the check runs once at trace
    time.  ``i1 < i0`` is a legal EMPTY window (Fortran's zero-trip
    ``do``).  ``k`` selects a trailing slot of a 3-D metric
    (``sin_sg``/``cos_sg``), 0-based.
    """
    lo_i, hi_i = i0 - ilo, i1 - ilo
    lo_j, hi_j = j0 - jlo, j1 - jlo
    if i0 <= i1 and (lo_i < 0 or hi_i >= a.shape[0]):
        raise IndexError(
            f"_fw: i window ({i0}:{i1}) escapes an array holding Fortran "
            f"i = {ilo}..{ilo + a.shape[0] - 1}")
    if j0 <= j1 and (lo_j < 0 or hi_j >= a.shape[1]):
        raise IndexError(
            f"_fw: j window ({j0}:{j1}) escapes an array holding Fortran "
            f"j = {jlo}..{jlo + a.shape[1] - 1}")
    w = a[lo_i:hi_i + 1, lo_j:hi_j + 1]
    return w if k is None else w[:, :, k]


def _fs(a, ilo, jlo, i0, i1, j0, j1, v):
    """Functional write ``a(i0:i1, j0:j1) = v`` (INCLUSIVE bounds).

    Returns a NEW array with the same static bound checks as
    :func:`_fw`; the array is NOT reallocated, so cells outside the
    window carry forward exactly as the in-place lane leaves them.
    """
    lo_i, hi_i = i0 - ilo, i1 - ilo
    lo_j, hi_j = j0 - jlo, j1 - jlo
    if i0 <= i1 and (lo_i < 0 or hi_i >= a.shape[0]):
        raise IndexError(
            f"_fs: i window ({i0}:{i1}) escapes an array holding Fortran "
            f"i = {ilo}..{ilo + a.shape[0] - 1}")
    if j0 <= j1 and (lo_j < 0 or hi_j >= a.shape[1]):
        raise IndexError(
            f"_fs: j window ({j0}:{j1}) escapes an array holding Fortran "
            f"j = {jlo}..{jlo + a.shape[1] - 1}")
    if i0 > i1 or j0 > j1:
        return a
    return a.at[lo_i:hi_i + 1, lo_j:hi_j + 1].set(v)


def _new(ilo, ihi, jlo, jhi, fill=jnp.nan):
    """A local at its exact Fortran declared bounds.

    The FILL is part of the contract, not decoration: the NumPy lane
    uses NaN as a tripwire for a cell the oracle never writes, ``0.0``
    where the plain ``d2a2c_vect`` zero-initialises its outputs,
    ``big_number`` for the duo ``d2a2c``/``d_sw1`` workspaces and
    ``1.0e25`` for ``divergence_corner_duo``'s ``divg_d`` -- and the
    per-stage certificates compare those cells.
    """
    return jnp.full((ihi - ilo + 1, jhi - jlo + 1), fill, jnp.float64)


def _runs(lo: int, hi: int, key):
    """Maximal runs of a STATIC per-index class over ``lo..hi``.

    Used wherever the Fortran branches on a LOOP INDEX
    (``if (i==1 .or. i==npx)``, the ``d2a2c_vect`` Ydir if/elseif
    chain, ...).  Resolving those at trace time into disjoint index runs
    is BOTH the literal translation (the Fortran really does run
    different code at different i) AND the R1b-safe one: the formula the
    oracle replaces at an edge is never evaluated there, so it can
    neither produce a NaN nor reach a gradient.  Returns
    ``[(cls, first, last), ...]`` in ascending order.
    """
    out: list[list] = []
    for x in range(lo, hi + 1):
        c = key(x)
        if out and out[-1][0] == c and out[-1][2] == x - 1:
            out[-1][2] = x
        else:
            out.append([c, x, x])
    return [(c, a, b) for c, a, b in out]


# =====================================================================
# sw_core_mod -- scalar corner fills and the edge interpolant
# =====================================================================

def edge_interpolate4(ua4, dxa4):
    """JAX twin of ``fv3_native_sw_core.edge_interpolate4``
    (sw_core.F90 ``edge_interpolate4``, verbatim).

    ``ua4``/``dxa4`` are 4-element SEQUENCES of arrays (the NumPy lane
    builds python lists of ``fort`` reads and indexes ``[0..3]``); the
    body is elementwise, so the twin applies to whatever shape the
    caller hands it -- a scalar, a j-column, or a 2-D block.

    ``t1``/``t2`` are sums of adjacent cell widths, so both divisions are
    on the LIVE path (not a dead branch) and are total on any admitted
    gridstruct, where ``dxa > 0``.
    """
    ops = {}
    for i in range(4):
        ops[f"ua4[{i}]"] = ua4[i]
        ops[f"dxa4[{i}]"] = dxa4[i]
    _require_f64_jax("edge_interpolate4", ops)
    t1 = dxa4[0] + dxa4[1]
    t2 = dxa4[2] + dxa4[3]
    return 0.5 * (((t1 + dxa4[1]) * ua4[1] - dxa4[1] * ua4[0]) / t1
                  + ((t2 + dxa4[2]) * ua4[2] - dxa4[2] * ua4[3]) / t2)


def _corner_pairs(direction: int, npx: int, npy: int, fname: str):
    """The ``(target_i, target_j, source_i, source_j)`` list of one
    ``fill_4corners`` direction (sw_core.F90:3856-3915, verbatim index
    mapping, mirrored from ``fv3_native_sw_core.fill_4corners``).

    Returned as a per-corner dict so ``fill2_4corners`` can apply the
    IDENTICAL mapping to both operands (the Fortran's ``fill2`` is
    literally ``fill`` twice, sw_core.F90:3794-3854).
    """
    if direction == 1:
        return {
            "sw": [(-1, 0, 0, 2), (0, 0, 0, 1)],
            "se": [(npx + 1, 0, npx, 2), (npx, 0, npx, 1)],
            "nw": [(0, npy, 0, npy - 1), (-1, npy, 0, npy - 2)],
            "ne": [(npx, npy, npx, npy - 1),
                   (npx + 1, npy, npx, npy - 2)],
        }
    if direction == 2:
        return {
            "sw": [(0, 0, 1, 0), (0, -1, 2, 0)],
            "se": [(npx, 0, npx - 1, 0), (npx, -1, npx - 2, 0)],
            "nw": [(0, npy, 1, npy), (0, npy + 1, 2, npy)],
            "ne": [(npx, npy, npx - 1, npy),
                   (npx, npy + 1, npx - 2, npy)],
        }
    raise ValueError(
        f"{fname}: dir={direction!r} is not a supported direction "
        f"(sw_core.F90:3858/:3887 branch on 1 == XDir and 2 == YDir "
        f"only; a third value would silently leave the corners "
        f"UNFILLED, which is a wrong answer rather than an error)")


def _apply_corner_fill(q, ilo, jlo, pairs, order, flags_on, fname):
    """Apply one direction's corner writes as an ORDERED chain.

    Dependence (R1a): within a corner the SECOND write's source is never
    the FIRST write's target (checked per case against the verbatim
    mapping -- e.g. dir 1 sw writes ``(-1,0)`` then ``(0,0)`` while
    reading ``(0,2)`` and ``(0,1)``), so each pair is a pure gather; the
    chain order is preserved anyway so that a future mapping change
    cannot silently reorder them.
    """
    for corner in order:
        if not flags_on[corner]:
            continue
        for (ti, tj, si, sj) in pairs[corner]:
            q = _fs(q, ilo, jlo, ti, ti, tj, tj,
                    _fw(q, ilo, jlo, si, si, sj, sj))
    return q


def fill_4corners(q, direction: int, npx: int, npy: int, bd, *,
                  sw: bool = True, se: bool = True, ne: bool = True,
                  nw: bool = True):
    """JAX twin of ``fv3_native_sw_core.fill_4corners``
    (sw_core.F90:3856-3915).

    Functional: the NumPy lane fills ``q``'s corner ghosts IN PLACE;
    this twin RETURNS the new ``q``.  ``q`` carries Fortran bounds
    ``(bd.isd:bd.ied, bd.jsd:bd.jed)``.
    """
    _require_f64_jax("fill_4corners", {"q": q})
    q = jnp.asarray(q)
    pairs = _corner_pairs(direction, npx, npy, "fill_4corners")
    return _apply_corner_fill(
        q, bd.isd, bd.jsd, pairs, ("sw", "se", "nw", "ne"),
        {"sw": sw, "se": se, "nw": nw, "ne": ne}, "fill_4corners")


def fill2_4corners(q1, q2, direction: int, npx: int, npy: int, bd, *,
                   sw: bool = True, se: bool = True, ne: bool = True,
                   nw: bool = True):
    """JAX twin of ``fv3_native_sw_core.fill2_4corners``
    (sw_core.F90:3794-3854).  Functional: RETURNS ``(q1, q2)``.

    The Fortran interleaves the two operands' writes inside each corner
    block; because the two arrays are DISJOINT the interleaving is
    unobservable, and applying the same ordered chain to each separately
    is exact.
    """
    _require_f64_jax("fill2_4corners", {"q1": q1, "q2": q2})
    q1 = jnp.asarray(q1)
    q2 = jnp.asarray(q2)
    pairs = _corner_pairs(direction, npx, npy, "fill2_4corners")
    on = {"sw": sw, "se": se, "nw": nw, "ne": ne}
    order = ("sw", "se", "nw", "ne")
    q1 = _apply_corner_fill(q1, bd.isd, bd.jsd, pairs, order, on,
                            "fill2_4corners")
    q2 = _apply_corner_fill(q2, bd.isd, bd.jsd, pairs, order, on,
                            "fill2_4corners")
    return q1, q2


# =====================================================================
# divergence_corner (plain) and divergence_corner_duo
# =====================================================================

_DIVG_KEYS = ("sin_sg", "cos_sg", "dxc", "dyc", "rarea_c")


def divergence_corner(u, v, ua, va, gs: dict, bd, npx: int, npy: int, *,
                      grid_type: int = 0):
    """JAX twin of ``fv3_native_sw_core.divergence_corner``
    (sw_core.F90:2124-2229).  Returns ``divg_d`` (B-node array,
    ``(isd:ied+1, jsd:jed+1)``).

    Never-written halo slots stay NaN, exactly like the twin (the oracle
    driver holds a sentinel there and the reconciliation test compares
    written slots only).

    Dependence (R1a): the ``uf`` nest writes ``uf`` and reads ``u``/
    ``va``/the metrics; the ``vf`` nest writes ``vf`` and reads
    ``v``/``ua``; the ``divg`` nest reads both completed work arrays.
    Each nest therefore vectorises, and the four corner-term removals
    plus the ``rarea_c`` scaling are applied as an ORDERED chain after
    it (the scaling reads what the removal wrote).

    R1b: the ``if (j==1 .or. j==npy)`` test (sw_core.F90 ``uf`` nest) and
    the ``is==1`` / ``ie+1==npx`` ``vf`` overrides are tests on the LOOP
    INDEX, i.e. STATIC -- they are resolved into disjoint index runs, not
    into a ``jnp.where``, so the ``va``-corrected form is never evaluated
    on the panel-edge rows where the oracle drops it.
    """
    _validate_grid_type("divergence_corner", grid_type)
    _require_f64_jax("divergence_corner",
                     {"u": u, "v": v, "ua": ua, "va": va})
    g = _geom("divergence_corner", gs, _DIVG_KEYS)
    u, v, ua, va = (jnp.asarray(x) for x in (u, v, ua, va))

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    ni, nj = ied - isd + 1, jed - jsd + 1
    _check_shape("divergence_corner", "u", u, (ni, nj + 1))
    _check_shape("divergence_corner", "v", v, (ni + 1, nj))
    _check_shape("divergence_corner", "ua", ua, (ni, nj))
    _check_shape("divergence_corner", "va", va, (ni, nj))

    if grid_type > 3:   # mirrors the twin's own refusal
        raise NotImplementedError(
            "divergence_corner: grid_type > 3 not ported (the NumPy "
            "authority raises the same way; sw_core.F90:2124 branches on "
            "grid_type>3 into a doubly-periodic form)")

    sg, cg = g["sin_sg"], g["cos_sg"]
    dxc, dyc, rarea_c = g["dxc"], g["dyc"], g["rarea_c"]

    def rd(a, i0, i1, j0, j1, k=None):        # data-domain origin read
        return _fw(a, isd, jsd, i0, i1, j0, j1, k)

    divg = _new(isd, ied + 1, jsd, jed + 1)
    uf = _new(is_ - 2, ie + 2, js - 1, je + 2)
    vf = _new(is_ - 1, ie + 2, js - 2, je + 2)

    is2 = max(2, is_)
    ie1 = min(npx - 1, ie + 1)

    # ---- uf (:2160-2181) ----------------------------------------
    # STATIC j classes: the panel-edge rows drop the va correction.
    def _uf_cls(j):
        return "edge" if (j == 1 or j == npy) else "interior"

    for cls, ja, jb in _runs(js, je + 1, _uf_cls):
        i0, i1 = is_ - 1, ie + 1
        sgs = rd(sg, i0, i1, ja - 1, jb - 1, 3) + rd(sg, i0, i1, ja, jb, 1)
        base = rd(u, i0, i1, ja, jb)
        if cls == "interior":
            base = base - 0.25 * (rd(va, i0, i1, ja - 1, jb - 1)
                                  + rd(va, i0, i1, ja, jb)) \
                * (rd(cg, i0, i1, ja - 1, jb - 1, 3)
                   + rd(cg, i0, i1, ja, jb, 1))
        uf = _fs(uf, is_ - 2, js - 1, i0, i1, ja, jb,
                 base * rd(dyc, i0, i1, ja, jb) * 0.5 * sgs)

    # ---- vf (:2184-2203) ----------------------------------------
    j0, j1 = js - 1, je + 1
    if is2 <= ie1:
        vf = _fs(vf, is_ - 1, js - 2, is2, ie1, j0, j1,
                 (rd(v, is2, ie1, j0, j1)
                  - 0.25 * (rd(ua, is2 - 1, ie1 - 1, j0, j1)
                            + rd(ua, is2, ie1, j0, j1))
                  * (rd(cg, is2 - 1, ie1 - 1, j0, j1, 2)
                     + rd(cg, is2, ie1, j0, j1, 0)))
                 * rd(dxc, is2, ie1, j0, j1) * 0.5
                 * (rd(sg, is2 - 1, ie1 - 1, j0, j1, 2)
                    + rd(sg, is2, ie1, j0, j1, 0)))
    if is_ == 1:
        vf = _fs(vf, is_ - 1, js - 2, 1, 1, j0, j1,
                 rd(v, 1, 1, j0, j1) * rd(dxc, 1, 1, j0, j1) * 0.5
                 * (rd(sg, 0, 0, j0, j1, 2) + rd(sg, 1, 1, j0, j1, 0)))
    if (ie + 1) == npx:
        vf = _fs(vf, is_ - 1, js - 2, npx, npx, j0, j1,
                 rd(v, npx, npx, j0, j1) * rd(dxc, npx, npx, j0, j1) * 0.5
                 * (rd(sg, npx - 1, npx - 1, j0, j1, 2)
                    + rd(sg, npx, npx, j0, j1, 0)))

    # ---- divg (:2206-2225) --------------------------------------
    def uf_at(i0, i1, ja, jb):
        return _fw(uf, is_ - 2, js - 1, i0, i1, ja, jb)

    def vf_at(i0, i1, ja, jb):
        return _fw(vf, is_ - 1, js - 2, i0, i1, ja, jb)

    divg = _fs(divg, isd, jsd, is_, ie + 1, js, je + 1,
               vf_at(is_, ie + 1, js - 1, je) - vf_at(is_, ie + 1, js, je + 1)
               + uf_at(is_ - 1, ie, js, je + 1)
               - uf_at(is_, ie + 1, js, je + 1))

    # remove the extra term at the corners (ORDERED after the nest)
    for (i, j, sgn) in ((1, 1, -1.0), (npx, 1, -1.0),
                        (npx, npy, +1.0), (1, npy, +1.0)):
        jj = 0 if j == 1 else npy
        divg = _fs(divg, isd, jsd, i, i, j, j,
                   _fw(divg, isd, jsd, i, i, j, j)
                   + sgn * vf_at(i, i, jj, jj))

    divg = _fs(divg, isd, jsd, is_, ie + 1, js, je + 1,
               rd(rarea_c, is_, ie + 1, js, je + 1)
               * _fw(divg, isd, jsd, is_, ie + 1, js, je + 1))
    return divg


def divergence_corner_duo(u, v, ua, va, gs: dict, bd, npx: int, npy: int,
                          *, grid_type: int = 0):
    """JAX twin of ``fv3_native_duo_sw_core.divergence_corner_duo``
    (sw_core.F90:2345-2447).  Returns ``divg_d``
    ``(isd:ied+1, jsd:jed+1)``, initialised to ``1.0e25`` exactly like
    the twin (``divg_d = 1.e25`` in the oracle).

    The interior formula runs over the whole data domain (no ``is2``/
    ``ie1`` clamp), then the duo panel-edge ZEROING and next-to-seam
    QUARTERING remove the cube-seam divergence the duo halos would
    otherwise double-count.

    VERBATIM quirk preserved from the twin: the upstream j-tests compare
    against ``npx``, not ``npy`` (harmless on the square single tile).

    Dependence (R1a): ``uf`` and ``vf`` are written by two independent
    nests reading only inputs; the ``divg`` nest reads both.  The
    zeroing/quartering are per-cell modifications of the value just
    computed at the SAME (i,j) -- no cross-cell reads -- so they are
    applied as ordered whole-row/column writes in source order.
    """
    _validate_grid_type("divergence_corner_duo", grid_type)
    _require_f64_jax("divergence_corner_duo",
                     {"u": u, "v": v, "ua": ua, "va": va})
    g = _geom("divergence_corner_duo", gs, _DIVG_KEYS)
    u, v, ua, va = (jnp.asarray(x) for x in (u, v, ua, va))

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    ni, nj = ied - isd + 1, jed - jsd + 1
    _check_shape("divergence_corner_duo", "u", u, (ni, nj + 1))
    _check_shape("divergence_corner_duo", "v", v, (ni + 1, nj))
    _check_shape("divergence_corner_duo", "ua", ua, (ni, nj))
    _check_shape("divergence_corner_duo", "va", va, (ni, nj))

    if grid_type > 3:
        raise NotImplementedError(
            "divergence_corner_duo: grid_type > 3 not ported (the NumPy "
            "authority raises the same way)")

    sg, cg = g["sin_sg"], g["cos_sg"]
    dxc, dyc, rarea_c = g["dxc"], g["dyc"], g["rarea_c"]

    def rd(a, i0, i1, j0, j1, k=None):
        return _fw(a, isd, jsd, i0, i1, j0, j1, k)

    divg = _new(isd, ied + 1, jsd, jed + 1, 1.0e25)
    uf = _new(isd, ied, jsd, jed + 1)
    vf = _new(isd, ied + 1, jsd, jed)

    uf = _fs(uf, isd, jsd, isd, ied, jsd + 1, jed,
             (rd(u, isd, ied, jsd + 1, jed)
              - 0.25 * (rd(va, isd, ied, jsd, jed - 1)
                        + rd(va, isd, ied, jsd + 1, jed))
              * (rd(cg, isd, ied, jsd, jed - 1, 3)
                 + rd(cg, isd, ied, jsd + 1, jed, 1)))
             * rd(dyc, isd, ied, jsd + 1, jed) * 0.5
             * (rd(sg, isd, ied, jsd, jed - 1, 3)
                + rd(sg, isd, ied, jsd + 1, jed, 1)))

    vf = _fs(vf, isd, jsd, isd + 1, ied, jsd, jed,
             (rd(v, isd + 1, ied, jsd, jed)
              - 0.25 * (rd(ua, isd, ied - 1, jsd, jed)
                        + rd(ua, isd + 1, ied, jsd, jed))
              * (rd(cg, isd, ied - 1, jsd, jed, 2)
                 + rd(cg, isd + 1, ied, jsd, jed, 0)))
             * rd(dxc, isd + 1, ied, jsd, jed) * 0.5
             * (rd(sg, isd, ied - 1, jsd, jed, 2)
                + rd(sg, isd + 1, ied, jsd, jed, 0)))

    i0, i1 = isd + 1, ied
    j0, j1 = jsd + 1, jed
    divg = _fs(divg, isd, jsd, i0, i1, j0, j1,
               (_fw(vf, isd, jsd, i0, i1, j0 - 1, j1 - 1)
                - _fw(vf, isd, jsd, i0, i1, j0, j1)
                + _fw(uf, isd, jsd, i0 - 1, i1 - 1, j0, j1)
                - _fw(uf, isd, jsd, i0, i1, j0, j1))
               * rd(rarea_c, i0, i1, j0, j1))

    def _zero_i(i):
        return _fs(divg, isd, jsd, i, i, j0, j1,
                   jnp.zeros((1, j1 - j0 + 1), jnp.float64))

    def _zero_j(j):
        return _fs(divg, isd, jsd, i0, i1, j, j,
                   jnp.zeros((i1 - i0 + 1, 1), jnp.float64))

    def _quarter_i(i):
        return _fs(divg, isd, jsd, i, i, j0, j1,
                   0.25 * _fw(divg, isd, jsd, i, i, j0, j1))

    def _quarter_j(j):
        return _fs(divg, isd, jsd, i0, i1, j, j,
                   0.25 * _fw(divg, isd, jsd, i0, i1, j, j))

    # duo panel-edge zeroing (the seam B-nodes), source order
    if is_ == 1 and i0 <= is_ <= i1:
        divg = _zero_i(is_)
    if (ie + 1) == npx and i0 <= ie + 1 <= i1:
        divg = _zero_i(ie + 1)
    if js == 1 and j0 <= 1 <= j1:
        divg = _zero_j(1)
    if je + 1 == npx and j0 <= je + 1 <= j1:
        divg = _zero_j(je + 1)
    # next-to-seam quartering (verbatim; the j-tests use npx upstream)
    if is_ == 1 and i0 <= is_ + 1 <= i1:
        divg = _quarter_i(is_ + 1)
    if (ie + 1) == npx and i0 <= ie <= i1:
        divg = _quarter_i(ie)
    if js == 1 and j0 <= 2 <= j1:
        divg = _quarter_j(2)
    if je + 1 == npx and j0 <= je <= j1:
        divg = _quarter_j(je)
    return divg


# =====================================================================
# del6_vt_flux  (sw_core.F90:2008-2121)
# =====================================================================

def del6_vt_flux(nord: int, npx: int, npy: int, damp, q, bd, del6_u,
                 del6_v, rarea, bounded_domain: bool, sw_corner: bool,
                 se_corner: bool, nw_corner: bool, ne_corner: bool,
                 duogrid: bool, damp_km=None, fx2=None, fy2=None):
    """JAX twin of ``fv3_native_d_sw.del6_vt_flux``
    (sw_core.F90:2008-2121, non-``USE_SG`` branch).

    Del-``nord`` damping of the relative vorticity: the same operator as
    ``tp_core``'s ``deln_flux`` except that it does NOT add the
    diffusive fluxes into the regular fluxes.  Functional: RETURNS
    ``(fx2, fy2)``; the NumPy lane writes them into CALLER work arrays
    (``d_sw6`` passes ``ut``/``vt``!).

    ``fx2``/``fy2`` may be supplied as the caller's incoming work arrays
    so the cells OUTSIDE the oracle's write windows carry forward
    exactly as the in-place lane leaves them -- that is what makes
    ``d_sw6_duo``'s returned ``ut``/``vt`` match the twin.  When omitted
    they are NaN-filled tripwires.

    Declared Fortran bounds: ``q``/``rarea`` ``(isd:ied, jsd:jed)``;
    ``del6_v`` ``(isd:ied+1, jsd:jed)``; ``del6_u``
    ``(isd:ied, jsd:jed+1)``; ``fx2`` ``(isd:ied+1, jsd:jed)``; ``fy2``
    ``(isd:ied, jsd:jed+1)``.

    ``nord`` is a LOOP-ITERATION COUNT, so it is a static python int by
    construction (CLAUDE.md: loop counts are never traced), and it also
    sets every index window; ``damp`` rides as a dynamic multiplier (it
    is never compared).

    ``damp_km`` is NOT in the pinned oracle -- ``sw_core.F90:2008``
    declares ``(nord, npx, npy, damp, q, d2, fx2, fy2, gridstruct, bd)``
    and has no such argument.  It comes from the extraction the NumPy
    lane was transcribed from, and since the NumPy lane is the
    specification for this hop it is mirrored as-is.  No caller in this
    module passes it.

    Dependence (R1a): the ``do n=1,nord`` pass loop is a RECURRENCE --
    pass ``n`` builds ``d2`` from the ``fx2``/``fy2`` pass ``n-1`` wrote,
    then overwrites them.  It stays a python ``for`` over a STATIC trip
    count (an unrolled, ordered chain) and is never vectorised or
    scanned; the window ``nt = nord - n`` shrinks each pass, so the
    bodies are not even the same shape.  Inside a pass each nest writes
    ONE array and reads only the others, so each nest vectorises, and
    ``d2`` is never reallocated so cells outside a window carry forward.

    A note on the ``copy_corners`` guard: the oracle's condition is
    ``.not. bounded_domain .or. .not. dg%is_initialized``
    (sw_core.F90:2060 / :2072 / :2093 / :2105) while the NumPy lane
    writes ``not bounded_domain``.  The two agree for every combination
    because ``copy_corners`` itself early-returns on
    ``bounded_domain .or. duogrid`` (tp_core.F90:239) -- the extra call
    the oracle makes at ``bounded .and. .not.duo`` is a no-op.  The
    NumPy lane is the authority for this hop, so its form is mirrored.
    """
    _validate_nord("del6_vt_flux", "nord", nord)
    for nm, v in (("bounded_domain", bounded_domain),
                  ("sw_corner", sw_corner), ("se_corner", se_corner),
                  ("nw_corner", nw_corner), ("ne_corner", ne_corner),
                  ("duogrid", duogrid)):
        _require_bool("del6_vt_flux", nm, v)
    _require_f64_jax("del6_vt_flux", {
        "q": q, "del6_u": del6_u, "del6_v": del6_v, "rarea": rarea,
        "damp_km": damp_km, "fx2": fx2, "fy2": fy2})
    q = jnp.asarray(q)
    del6_u = jnp.asarray(del6_u)
    del6_v = jnp.asarray(del6_v)
    rarea = jnp.asarray(rarea)

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    nid, njd = ied - isd + 1, jed - jsd + 1
    _check_shape("del6_vt_flux", "q", q, (nid, njd))
    _check_shape("del6_vt_flux", "rarea", rarea, (nid, njd))
    _check_shape("del6_vt_flux", "del6_v", del6_v, (nid + 1, njd))
    _check_shape("del6_vt_flux", "del6_u", del6_u, (nid, njd + 1))

    d2 = _new(isd, ied, jsd, jed)
    fx2 = _new(isd, ied + 1, jsd, jed) if fx2 is None else jnp.asarray(fx2)
    fy2 = _new(isd, ied, jsd, jed + 1) if fy2 is None else jnp.asarray(fy2)
    _check_shape("del6_vt_flux", "fx2", fx2, (nid + 1, njd))
    _check_shape("del6_vt_flux", "fy2", fy2, (nid, njd + 1))

    def d2_at(i0, i1, j0, j1):
        return _fw(d2, isd, jsd, i0, i1, j0, j1)

    def cc(a, dir_):
        return copy_corners(a, npx, npy, dir_, bounded_domain, bd,
                            sw_corner, se_corner, nw_corner, ne_corner,
                            duogrid=duogrid)

    i1, i2 = is_ - 1 - nord, ie + 1 + nord                 # :2051-2052
    j1, j2 = js - 1 - nord, je + 1 + nord
    d2 = _fs(d2, isd, jsd, i1, i2, j1, j2,                 # :2054-2058
             damp * _fw(q, isd, jsd, i1, i2, j1, j2))

    if nord > 0 and (not bounded_domain):                  # :2060-2061
        d2 = cc(d2, 1)
    fx2 = _fs(fx2, isd, jsd, is_ - nord, ie + nord + 1,    # :2062-2070
              js - nord, je + nord,
              _fw(del6_v, isd, jsd, is_ - nord, ie + nord + 1,
                  js - nord, je + nord)
              * (d2_at(is_ - nord - 1, ie + nord, js - nord, je + nord)
                 - d2_at(is_ - nord, ie + nord + 1, js - nord, je + nord)))

    if nord > 0 and (not bounded_domain):                  # :2072-2073
        d2 = cc(d2, 2)
    fy2 = _fs(fy2, isd, jsd, is_ - nord, ie + nord,        # :2074-2082
              js - nord, je + nord + 1,
              _fw(del6_u, isd, jsd, is_ - nord, ie + nord,
                  js - nord, je + nord + 1)
              * (d2_at(is_ - nord, ie + nord, js - nord - 1, je + nord)
                 - d2_at(is_ - nord, ie + nord, js - nord, je + nord + 1)))

    if nord > 0:                                           # :2084-2119
        # ORDERED recurrence over n -- see the docstring.
        for n in range(1, nord + 1):
            nt = nord - n
            d2 = _fs(d2, isd, jsd, is_ - nt - 1, ie + nt + 1,
                     js - nt - 1, je + nt + 1,
                     (_fw(fx2, isd, jsd, is_ - nt - 1, ie + nt + 1,
                          js - nt - 1, je + nt + 1)
                      - _fw(fx2, isd, jsd, is_ - nt, ie + nt + 2,
                            js - nt - 1, je + nt + 1)
                      + _fw(fy2, isd, jsd, is_ - nt - 1, ie + nt + 1,
                            js - nt - 1, je + nt + 1)
                      - _fw(fy2, isd, jsd, is_ - nt - 1, ie + nt + 1,
                            js - nt, je + nt + 2))
                     * _fw(rarea, isd, jsd, is_ - nt - 1, ie + nt + 1,
                           js - nt - 1, je + nt + 1))

            if not bounded_domain:
                d2 = cc(d2, 1)
            fx2 = _fs(fx2, isd, jsd, is_ - nt, ie + nt + 1,
                      js - nt, je + nt,
                      _fw(del6_v, isd, jsd, is_ - nt, ie + nt + 1,
                          js - nt, je + nt)
                      * (d2_at(is_ - nt, ie + nt + 1, js - nt, je + nt)
                         - d2_at(is_ - nt - 1, ie + nt, js - nt, je + nt)))

            if not bounded_domain:
                d2 = cc(d2, 2)
            fy2 = _fs(fy2, isd, jsd, is_ - nt, ie + nt,
                      js - nt, je + nt + 1,
                      _fw(del6_u, isd, jsd, is_ - nt, ie + nt,
                          js - nt, je + nt + 1)
                      * (d2_at(is_ - nt, ie + nt, js - nt, je + nt + 1)
                         - d2_at(is_ - nt, ie + nt, js - nt - 1, je + nt)))

    if damp_km is not None:   # coefficient multiplied in earlier
        damp_km = jnp.asarray(damp_km)
        _check_shape("del6_vt_flux", "damp_km", damp_km, (nid, njd))
        fx2 = _fs(fx2, isd, jsd, is_, ie + 1, js, je,
                  _fw(fx2, isd, jsd, is_, ie + 1, js, je) * 0.5
                  * _fw(damp_km, isd, jsd, is_, ie + 1, js, je))
        fy2 = _fs(fy2, isd, jsd, is_, ie, js, je + 1,
                  _fw(fy2, isd, jsd, is_, ie, js, je + 1) * 0.5
                  * _fw(damp_km, isd, jsd, is_, ie, js, je + 1))
    return fx2, fy2


# =====================================================================
# d2a2c_vect -- the duo branch, then the plain branch
# =====================================================================

_D2A2C_DUO_KEYS = ("cosa_s", "rsin2", "cosa_u", "rsin_u", "cosa_v",
                   "rsin_v")
_D2A2C_KEYS = _D2A2C_DUO_KEYS + ("sin_sg", "dxa", "dya")


def d2a2c_vect_duo(u, v, gs: dict, bd, npx: int, npy: int, *,
                   dord4: bool = True, grid_type: int = 0):
    """JAX twin of ``fv3_native_duo_sw_core.d2a2c_vect_duo``
    (``d2a2c_vect``'s ``gridstruct%dg%is_initialized`` branch,
    sw_core.F90:3361-3706 -- the duo arms at :3421-3454, :3558-3563 and
    :3690-3693).

    D-grid ``(u, v)`` -> A-grid ``(ua, va)`` -> C-grid ``(uc, vc)`` plus
    the contravariant ``(ut, vt)``.  The duo branch runs the INTERIOR
    4th-order formulas over the full data domain and SKIPS every
    panel-edge / cube-corner special case (the duo halos carry real
    cross-face winds).  Returns ``{ua, va, uc, vc, ut, vt}``.

    ``dord4``/``id`` is UNUSED on the duo branch (its interior covers the
    full domain id-independently); it is kept for signature parity with
    the twin.  Every output is ``big_number``-filled where the oracle
    never writes, exactly like the twin.

    Dependence (R1a): ``utmp``'s interior nest and its two boundary rows
    write DISJOINT j slabs (``jsd+1..jed-1`` vs ``jsd``/``jed``) and read
    only ``u``; same for ``vtmp`` in i.  ``ua``/``va`` read both
    completed temporaries; ``uc``/``ut`` read ``utmp``/``v``;
    ``vc``/``vt`` read ``vtmp``/``u``.  No nest reads a location another
    iteration of the same nest writes, so every one vectorises.  There is
    no data-dependent branch at all on this lane, hence no ``jnp.where``.
    """
    _validate_grid_type("d2a2c_vect_duo", grid_type)
    _require_bool("d2a2c_vect_duo", "dord4", dord4)
    _require_f64_jax("d2a2c_vect_duo", {"u": u, "v": v})
    g = _geom("d2a2c_vect_duo", gs, _D2A2C_DUO_KEYS)
    u, v = jnp.asarray(u), jnp.asarray(v)

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    ni, nj = ied - isd + 1, jed - jsd + 1
    _check_shape("d2a2c_vect_duo", "u", u, (ni, nj + 1))
    _check_shape("d2a2c_vect_duo", "v", v, (ni + 1, nj))

    if grid_type >= 3:
        raise NotImplementedError(
            "d2a2c_vect_duo: grid_type >= 3 not ported (the NumPy "
            "authority raises the same way; the duo lane is the "
            "cubed-sphere oracle only)")

    a1, a2 = A1, A2
    cosa_s, rsin2 = g["cosa_s"], g["rsin2"]
    cosa_u, rsin_u = g["cosa_u"], g["rsin_u"]
    cosa_v, rsin_v = g["cosa_v"], g["rsin_v"]

    def rd(a, i0, i1, j0, j1):
        return _fw(a, isd, jsd, i0, i1, j0, j1)

    utmp = _new(isd, ied, jsd, jed, BIG_NUMBER)
    vtmp = _new(isd, ied, jsd, jed, BIG_NUMBER)
    ua = _new(isd, ied, jsd, jed, BIG_NUMBER)
    va = _new(isd, ied, jsd, jed, BIG_NUMBER)
    uc = _new(isd, ied + 1, jsd, jed, BIG_NUMBER)
    vc = _new(isd, ied, jsd, jed + 1, BIG_NUMBER)
    ut = _new(isd, ied, jsd, jed, BIG_NUMBER)
    vt = _new(isd, ied, jsd, jed, BIG_NUMBER)

    # ---- D -> A (duo interior; :3421-3454) ----------------------
    utmp = _fs(utmp, isd, jsd, isd, ied, jsd + 1, jed - 1,
               a2 * (rd(u, isd, ied, jsd, jed - 2)
                     + rd(u, isd, ied, jsd + 3, jed + 1))
               + a1 * (rd(u, isd, ied, jsd + 1, jed - 1)
                       + rd(u, isd, ied, jsd + 2, jed)))
    # 0.5*(u(j+1)+u(j+1)) -- the twin writes u(jsd+1)/u(jed+1) verbatim
    utmp = _fs(utmp, isd, jsd, isd, ied, jsd, jsd,
               rd(u, isd, ied, jsd + 1, jsd + 1))
    utmp = _fs(utmp, isd, jsd, isd, ied, jed, jed,
               rd(u, isd, ied, jed + 1, jed + 1))

    vtmp = _fs(vtmp, isd, jsd, isd + 1, ied - 1, jsd, jed,
               a2 * (rd(v, isd, ied - 2, jsd, jed)
                     + rd(v, isd + 3, ied + 1, jsd, jed))
               + a1 * (rd(v, isd + 1, ied - 1, jsd, jed)
                       + rd(v, isd + 2, ied, jsd, jed)))
    vtmp = _fs(vtmp, isd, jsd, isd, isd, jsd, jed,
               rd(v, isd + 1, isd + 1, jsd, jed))
    vtmp = _fs(vtmp, isd, jsd, ied, ied, jsd, jed,
               rd(v, ied + 1, ied + 1, jsd, jed))

    utf = _fw(utmp, isd, jsd, isd, ied, jsd, jed)
    vtf = _fw(vtmp, isd, jsd, isd, ied, jsd, jed)
    cs = rd(cosa_s, isd, ied, jsd, jed)
    r2 = rd(rsin2, isd, ied, jsd, jed)
    ua = _fs(ua, isd, jsd, isd, ied, jsd, jed, (utf - vtf * cs) * r2)
    va = _fs(va, isd, jsd, isd, ied, jsd, jed, (vtf - utf * cs) * r2)

    # ---- A -> C  X-dir (duo ifirst=is-1, ilast=ie+2; :3558-3563) --
    ia, ib, ja, jb = is_ - 1, ie + 2, js - 1, je + 1
    uc_w = (a2 * (_fw(utmp, isd, jsd, ia - 2, ib - 2, ja, jb)
                  + _fw(utmp, isd, jsd, ia + 1, ib + 1, ja, jb))
            + a1 * (_fw(utmp, isd, jsd, ia - 1, ib - 1, ja, jb)
                    + _fw(utmp, isd, jsd, ia, ib, ja, jb)))
    uc = _fs(uc, isd, jsd, ia, ib, ja, jb, uc_w)
    ut = _fs(ut, isd, jsd, ia, ib, ja, jb,
             (uc_w - rd(v, ia, ib, ja, jb) * rd(cosa_u, ia, ib, ja, jb))
             * rd(rsin_u, ia, ib, ja, jb))

    # ---- A -> C  Y-dir (duo interior for every j; :3690-3693) ----
    ia, ib, ja, jb = is_ - 1, ie + 1, js - 1, je + 2
    vc_w = (a2 * (_fw(vtmp, isd, jsd, ia, ib, ja - 2, jb - 2)
                  + _fw(vtmp, isd, jsd, ia, ib, ja + 1, jb + 1))
            + a1 * (_fw(vtmp, isd, jsd, ia, ib, ja - 1, jb - 1)
                    + _fw(vtmp, isd, jsd, ia, ib, ja, jb)))
    vc = _fs(vc, isd, jsd, ia, ib, ja, jb, vc_w)
    vt = _fs(vt, isd, jsd, ia, ib, ja, jb,
             (vc_w - rd(u, ia, ib, ja, jb) * rd(cosa_v, ia, ib, ja, jb))
             * rd(rsin_v, ia, ib, ja, jb))

    return {"ua": ua, "va": va, "uc": uc, "vc": vc, "ut": ut, "vt": vt}


def d2a2c_vect(u, v, gs: dict, bd, npx: int, npy: int, *,
               dord4: bool = True, grid_type: int = 0,
               bounded_domain: bool = False):
    """JAX twin of ``fv3_native_sw_core.d2a2c_vect``
    (sw_core.F90:3361-3706, the NON-duo branch).

    Returns the tuple ``(ua, va, uc, vc, ut, vt)`` -- the twin's own
    return shape.  Outputs are ZERO-initialised (matching the oracle
    driver), so slots the Fortran never writes hold ``0.0``; the
    ``utmp``/``vtmp`` temporaries are ``big_number``-filled.

    Dependence and branch analysis (R1a / R1b), stage by stage, in the
    oracle's own order -- the ORDER is load-bearing and is preserved as
    an ordered chain:

    1. the interior ``utmp``/``vtmp`` nests write one array and read only
       ``u``/``v``  ->  vectorised;
    2. the four panel-edge blocks overwrite sub-slabs of the same
       temporaries reading only ``u``/``v``  ->  vectorised, applied in
       source order so a later block wins exactly as in place;
    3. ``ua``/``va`` read the PRE-corner ``utmp``/``vtmp`` -- which is
       why step 4 must come after this one;
    4. the ``utmp`` cube-corner overrides read ``vtmp`` (a DIFFERENT
       array), so each is a pure gather; they are emitted as literal
       per-element writes because the source index runs BACKWARDS
       (``utmp(i,0) = -vtmp(0,1-i)``) and a reversed slice is exactly the
       kind of silent transposition this port cannot afford;
    5. the 4th-order ``uc``/``ut`` nest reads the POST-corner ``utmp``;
       ``ut`` reads ``uc`` at the SAME ``(i,j)``, so computing both from
       one expression is exact;
    6. the ``is==1`` / ``ie+1==npx`` edge columns are STATIC index
       branches, so they are separate writes, not a ``where``.  Inside
       each, ``ut(1,j)`` is written by ``edge_interpolate4`` and then
       READ by ``uc(1,j)``; vectorising over ``j`` preserves that because
       no ``j`` reads another ``j``.  The ``ut(1,j) > 0`` select is on
       DATA and both arms are plain products  ->  ``jnp.where``;
    7. the ``vtmp`` corner overrides read the ALREADY-MODIFIED ``utmp``
       (step 4), and the ``va`` corner overrides read the
       already-modified ``ua``  ->  ordered chain, pure gathers;
    8. the Ydir ``vc``/``vt`` if/elseif chain is a branch on the LOOP
       INDEX ``j``: it is resolved into disjoint index runs by
       :func:`_runs`, evaluated in the oracle's own test order, so the
       ``edge_interpolate4`` form is never evaluated on an interior row
       and vice versa.

    ``bounded_domain=True`` raises, exactly as the twin does (the oracle
    corpus for this routine is cubed-sphere only).
    """
    _validate_grid_type("d2a2c_vect", grid_type)
    _require_bool("d2a2c_vect", "dord4", dord4)
    _require_bool("d2a2c_vect", "bounded_domain", bounded_domain)
    _require_f64_jax("d2a2c_vect", {"u": u, "v": v})
    g = _geom("d2a2c_vect", gs, _D2A2C_KEYS)
    u, v = jnp.asarray(u), jnp.asarray(v)

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    ni, nj = ied - isd + 1, jed - jsd + 1
    _check_shape("d2a2c_vect", "u", u, (ni, nj + 1))
    _check_shape("d2a2c_vect", "v", v, (ni + 1, nj))

    if bounded_domain:
        raise NotImplementedError(
            "d2a2c_vect: bounded_domain not ported (the NumPy authority "
            "raises the same way; the bounded lane is reached through "
            "d2a2c_vect_duo)")

    sg = g["sin_sg"]
    cosa_s, rsin2 = g["cosa_s"], g["rsin2"]
    cosa_u, rsin_u = g["cosa_u"], g["rsin_u"]
    cosa_v, rsin_v = g["cosa_v"], g["rsin_v"]
    dxa, dya = g["dxa"], g["dya"]

    def rd(a, i0, i1, j0, j1, k=None):
        return _fw(a, isd, jsd, i0, i1, j0, j1, k)

    ua = _new(isd, ied, jsd, jed, 0.0)
    va = _new(isd, ied, jsd, jed, 0.0)
    ut = _new(isd, ied, jsd, jed, 0.0)
    vt = _new(isd, ied, jsd, jed, 0.0)
    uc = _new(isd, ied + 1, jsd, jed, 0.0)
    vc = _new(isd, ied, jsd, jed + 1, 0.0)
    utmp = _new(isd, ied, jsd, jed, BIG_NUMBER)
    vtmp = _new(isd, ied, jsd, jed, BIG_NUMBER)

    id_ = 1 if dord4 else 0
    npt = 4 if (grid_type < 3 and not bounded_domain) else -2

    # ---- 1. interior ------------------------------------------------
    ja, jb = max(npt, js - 1), min(npy - npt, je + 1)
    ia, ib = max(npt, isd), min(npx - npt, ied)
    if ja <= jb and ia <= ib:
        utmp = _fs(utmp, isd, jsd, ia, ib, ja, jb,
                   A2 * (rd(u, ia, ib, ja - 1, jb - 1)
                         + rd(u, ia, ib, ja + 2, jb + 2))
                   + A1 * (rd(u, ia, ib, ja, jb)
                           + rd(u, ia, ib, ja + 1, jb + 1)))
    ja, jb = max(npt, jsd), min(npy - npt, jed)
    ia, ib = max(npt, is_ - 1), min(npx - npt, ie + 1)
    if ja <= jb and ia <= ib:
        vtmp = _fs(vtmp, isd, jsd, ia, ib, ja, jb,
                   A2 * (rd(v, ia - 1, ib - 1, ja, jb)
                         + rd(v, ia + 2, ib + 2, ja, jb))
                   + A1 * (rd(v, ia, ib, ja, jb)
                           + rd(v, ia + 1, ib + 1, ja, jb)))

    # ---- 2. panel edges (ordered; later block wins) -----------------
    def _edge_block(utmp, vtmp, i0, i1, j0, j1):
        if i0 > i1 or j0 > j1:
            return utmp, vtmp
        utmp = _fs(utmp, isd, jsd, i0, i1, j0, j1,
                   0.5 * (rd(u, i0, i1, j0, j1)
                          + rd(u, i0, i1, j0 + 1, j1 + 1)))
        vtmp = _fs(vtmp, isd, jsd, i0, i1, j0, j1,
                   0.5 * (rd(v, i0, i1, j0, j1)
                          + rd(v, i0 + 1, i1 + 1, j0, j1)))
        return utmp, vtmp

    if grid_type < 3:
        jlo, jhi = max(npt, jsd), min(npy - npt, jed)
        if js == 1 or jsd < npt:
            utmp, vtmp = _edge_block(utmp, vtmp, isd, ied, jsd, npt - 1)
        if (je + 1) == npy or jed >= (npy - npt):
            utmp, vtmp = _edge_block(utmp, vtmp, isd, ied,
                                     npy - npt + 1, jed)
        if is_ == 1 or isd < npt:
            utmp, vtmp = _edge_block(utmp, vtmp, isd, npt - 1, jlo, jhi)
        if (ie + 1) == npx or ied >= (npx - npt):
            utmp, vtmp = _edge_block(utmp, vtmp, npx - npt + 1, ied,
                                     jlo, jhi)

    # ---- 3. contra-variant components at cell centre ----------------
    ja, jb = js - 1 - id_, je + 1 + id_
    ia, ib = is_ - 1 - id_, ie + 1 + id_
    utf = _fw(utmp, isd, jsd, ia, ib, ja, jb)
    vtf = _fw(vtmp, isd, jsd, ia, ib, ja, jb)
    cs = rd(cosa_s, ia, ib, ja, jb)
    r2 = rd(rsin2, ia, ib, ja, jb)
    ua = _fs(ua, isd, jsd, ia, ib, ja, jb, (utf - vtf * cs) * r2)
    va = _fs(va, isd, jsd, ia, ib, ja, jb, (vtf - utf * cs) * r2)

    # ---- 4. A -> C: fix the edges, Xdir (literal per-element) -------
    def _pt(dst, ti, tj, val):
        return _fs(dst, isd, jsd, ti, ti, tj, tj, val)

    def _vt_at(i, j):
        return _fw(vtmp, isd, jsd, i, i, j, j)

    def _ut_at(i, j):
        return _fw(utmp, isd, jsd, i, i, j, j)

    for i in range(-2, 0 + 1):                        # sw corner
        utmp = _pt(utmp, i, 0, -_vt_at(0, 1 - i))
    for i in range(0, 2 + 1):                         # se corner
        utmp = _pt(utmp, npx + i, 0, _vt_at(npx, i + 1))
    for i in range(0, 2 + 1):                         # ne corner
        utmp = _pt(utmp, npx + i, npy, -_vt_at(npx, je - i))
    for i in range(-2, 0 + 1):                        # nw corner
        utmp = _pt(utmp, i, npy, _vt_at(0, je + i))

    # ---- 5. 4th-order interpolation for interior points -------------
    if grid_type < 3 and not bounded_domain:
        ifirst, ilast = max(3, is_ - 1), min(npx - 2, ie + 2)
    else:
        ifirst, ilast = is_ - 1, ie + 2
    ja, jb = js - 1, je + 1
    if ifirst <= ilast:
        uc_w = (A2 * (_fw(utmp, isd, jsd, ifirst - 2, ilast - 2, ja, jb)
                      + _fw(utmp, isd, jsd, ifirst + 1, ilast + 1,
                            ja, jb))
                + A1 * (_fw(utmp, isd, jsd, ifirst - 1, ilast - 1,
                            ja, jb)
                        + _fw(utmp, isd, jsd, ifirst, ilast, ja, jb)))
        uc = _fs(uc, isd, jsd, ifirst, ilast, ja, jb, uc_w)
        ut = _fs(ut, isd, jsd, ifirst, ilast, ja, jb,
                 (uc_w - rd(v, ifirst, ilast, ja, jb)
                  * rd(cosa_u, ifirst, ilast, ja, jb))
                 * rd(rsin_u, ifirst, ilast, ja, jb))

    if grid_type < 3:
        # ---- 6a. Xdir ua corner overrides (pure gathers from va) ----
        for (ti, tj, si, sj, s) in (
                (-1, 0, 0, 2, -1.0), (0, 0, 0, 1, -1.0),          # sw
                (npx, 0, npx, 1, +1.0), (npx + 1, 0, npx, 2, +1.0),  # se
                (npx, npy, npx, npy - 1, -1.0),                    # ne
                (npx + 1, npy, npx, npy - 2, -1.0),
                (-1, npy, 0, npy - 2, +1.0),                       # nw
                (0, npy, 0, npy - 1, +1.0)):
            ua = _pt(ua, ti, tj, s * _fw(va, isd, jsd, si, si, sj, sj))

        def _utmp_col(i):
            return _fw(utmp, isd, jsd, i, i, ja, jb)

        def _ua_col(i):
            return _fw(ua, isd, jsd, i, i, ja, jb)

        def _dxa_col(i):
            return _fw(dxa, isd, jsd, i, i, ja, jb)

        def _col(a, i):
            return _fw(a, isd, jsd, i, i, ja, jb)

        # ---- 6b. west edge (sw_core.F90 `if (is==1 ...)`) ----------
        if is_ == 1 and not bounded_domain:
            uc0 = (C1 * _utmp_col(-2) + C2 * _utmp_col(-1)
                   + C3 * _utmp_col(0))
            ut1 = edge_interpolate4(
                [_ua_col(-1), _ua_col(0), _ua_col(1), _ua_col(2)],
                [_dxa_col(-1), _dxa_col(0), _dxa_col(1), _dxa_col(2)])
            # DATA branch; both arms are products of written metrics.
            uc1 = jnp.where(ut1 > 0.0,
                            ut1 * _fw(sg, isd, jsd, 0, 0, ja, jb, 2),
                            ut1 * _fw(sg, isd, jsd, 1, 1, ja, jb, 0))
            uc2 = (C1 * _utmp_col(3) + C2 * _utmp_col(2)
                   + C3 * _utmp_col(1))
            uc = _fs(uc, isd, jsd, 0, 0, ja, jb, uc0)
            ut = _fs(ut, isd, jsd, 1, 1, ja, jb, ut1)
            uc = _fs(uc, isd, jsd, 1, 1, ja, jb, uc1)
            uc = _fs(uc, isd, jsd, 2, 2, ja, jb, uc2)
            ut = _fs(ut, isd, jsd, 0, 0, ja, jb,
                     (uc0 - _col(v, 0) * _col(cosa_u, 0))
                     * _col(rsin_u, 0))
            ut = _fs(ut, isd, jsd, 2, 2, ja, jb,
                     (uc2 - _col(v, 2) * _col(cosa_u, 2))
                     * _col(rsin_u, 2))

        # ---- 6c. east edge ----------------------------------------
        if (ie + 1) == npx and not bounded_domain:
            ucm = (C1 * _utmp_col(npx - 3) + C2 * _utmp_col(npx - 2)
                   + C3 * _utmp_col(npx - 1))
            utn = edge_interpolate4(
                [_ua_col(npx - 2), _ua_col(npx - 1), _ua_col(npx),
                 _ua_col(npx + 1)],
                [_dxa_col(npx - 2), _dxa_col(npx - 1), _dxa_col(npx),
                 _dxa_col(npx + 1)])
            ucn = jnp.where(
                utn > 0.0,
                utn * _fw(sg, isd, jsd, npx - 1, npx - 1, ja, jb, 2),
                utn * _fw(sg, isd, jsd, npx, npx, ja, jb, 0))
            ucp = (C3 * _utmp_col(npx) + C2 * _utmp_col(npx + 1)
                   + C1 * _utmp_col(npx + 2))
            uc = _fs(uc, isd, jsd, npx - 1, npx - 1, ja, jb, ucm)
            ut = _fs(ut, isd, jsd, npx, npx, ja, jb, utn)
            uc = _fs(uc, isd, jsd, npx, npx, ja, jb, ucn)
            uc = _fs(uc, isd, jsd, npx + 1, npx + 1, ja, jb, ucp)
            ut = _fs(ut, isd, jsd, npx - 1, npx - 1, ja, jb,
                     (ucm - _col(v, npx - 1) * _col(cosa_u, npx - 1))
                     * _col(rsin_u, npx - 1))
            ut = _fs(ut, isd, jsd, npx + 1, npx + 1, ja, jb,
                     (ucp - _col(v, npx + 1) * _col(cosa_u, npx + 1))
                     * _col(rsin_u, npx + 1))

    # ---- 7. Ydir corner overrides (read the MODIFIED utmp/ua) -------
    for j in range(-2, 0 + 1):                        # sw corner
        vtmp = _pt(vtmp, 0, j, -_ut_at(1 - j, 0))
    for j in range(0, 2 + 1):                         # nw corner
        vtmp = _pt(vtmp, 0, npy + j, _ut_at(j + 1, npy))
    for j in range(-2, 0 + 1):                        # se corner
        vtmp = _pt(vtmp, npx, j, _ut_at(ie + j, 0))
    for j in range(0, 2 + 1):                         # ne corner
        vtmp = _pt(vtmp, npx, npy + j, -_ut_at(ie - j, npy))

    for (ti, tj, si, sj, s) in (
            (0, -1, 2, 0, -1.0), (0, 0, 1, 0, -1.0),               # sw
            (npx, 0, npx - 1, 0, +1.0), (npx, -1, npx - 2, 0, +1.0),
            (npx, npy, npx - 1, npy, -1.0),                        # ne
            (npx, npy + 1, npx - 2, npy, -1.0),
            (0, npy, 1, npy, +1.0), (0, npy + 1, 2, npy, +1.0)):   # nw
        va = _fs(va, isd, jsd, ti, ti, tj, tj,
                 s * _fw(ua, isd, jsd, si, si, sj, sj))

    # ---- 8. Ydir vc/vt ---------------------------------------------
    ia, ib = is_ - 1, ie + 1

    def wv(a, off, ja2, jb2, k=None):
        """``a(is-1:ie+1, j+off)`` over the run ``ja2..jb2``."""
        return _fw(a, isd, jsd, ia, ib, ja2 + off, jb2 + off, k)

    def _vc_cls(j):
        # the oracle's own if/elseif ORDER (sw_core.F90 Ydir chain)
        if j == 1 and not bounded_domain:
            return "edge"
        if j == 0 or (j == (npy - 1) and not bounded_domain):
            return "back"
        if j == 2 or (j == (npy + 1) and not bounded_domain):
            return "fwd"
        if j == npy and not bounded_domain:
            return "edge"
        return "interior"

    if grid_type < 3:
        for cls, ja2, jb2 in _runs(js - 1, je + 2, _vc_cls):
            if cls == "edge":
                vt_w = edge_interpolate4(
                    [wv(va, -2, ja2, jb2), wv(va, -1, ja2, jb2),
                     wv(va, 0, ja2, jb2), wv(va, +1, ja2, jb2)],
                    [wv(dya, -2, ja2, jb2), wv(dya, -1, ja2, jb2),
                     wv(dya, 0, ja2, jb2), wv(dya, +1, ja2, jb2)])
                vc_w = jnp.where(vt_w > 0.0,
                                 vt_w * wv(sg, -1, ja2, jb2, 3),
                                 vt_w * wv(sg, 0, ja2, jb2, 1))
                vc = _fs(vc, isd, jsd, ia, ib, ja2, jb2, vc_w)
                vt = _fs(vt, isd, jsd, ia, ib, ja2, jb2, vt_w)
                continue
            if cls == "back":
                vc_w = (C1 * wv(vtmp, -2, ja2, jb2)
                        + C2 * wv(vtmp, -1, ja2, jb2)
                        + C3 * wv(vtmp, 0, ja2, jb2))
            elif cls == "fwd":
                vc_w = (C1 * wv(vtmp, +1, ja2, jb2)
                        + C2 * wv(vtmp, 0, ja2, jb2)
                        + C3 * wv(vtmp, -1, ja2, jb2))
            else:
                vc_w = (A2 * (wv(vtmp, -2, ja2, jb2)
                              + wv(vtmp, +1, ja2, jb2))
                        + A1 * (wv(vtmp, -1, ja2, jb2)
                                + wv(vtmp, 0, ja2, jb2)))
            vc = _fs(vc, isd, jsd, ia, ib, ja2, jb2, vc_w)
            vt = _fs(vt, isd, jsd, ia, ib, ja2, jb2,
                     (vc_w - wv(u, 0, ja2, jb2) * wv(cosa_v, 0, ja2, jb2))
                     * wv(rsin_v, 0, ja2, jb2))
    else:
        ja2, jb2 = js - 1, je + 2
        vc_w = (A2 * (_fw(vtmp, isd, jsd, ia, ib, ja2 - 2, jb2 - 2)
                      + _fw(vtmp, isd, jsd, ia, ib, ja2 + 1, jb2 + 1))
                + A1 * (_fw(vtmp, isd, jsd, ia, ib, ja2 - 1, jb2 - 1)
                        + _fw(vtmp, isd, jsd, ia, ib, ja2, jb2)))
        vc = _fs(vc, isd, jsd, ia, ib, ja2, jb2, vc_w)
        vt = _fs(vt, isd, jsd, ia, ib, ja2, jb2, vc_w)

    return ua, va, uc, vc, ut, vt


# =====================================================================
# c_sw  (sw_core.F90:79-494)
# =====================================================================

_CSW_KEYS = ("sin_sg", "cos_sg", "cosa_u", "cosa_v", "sina_u", "sina_v",
             "dx", "dy", "dxc", "dyc", "rdxc", "rdyc", "rarea",
             "rarea_c", "fC")


def c_sw(delp, pt, w, u, v, gs: dict, bd, npx: int, npy: int, dt2, *,
         nord: int = 1, hydrostatic: bool = True, dord4: bool = True,
         grid_type: int = 0, duogrid: bool = False,
         bounded_domain: bool = False):
    """JAX twin of ``fv3_native_sw_core.c_sw`` (sw_core.F90:79-494) --
    one C-grid forward half step.

    Returns a dict with ``delpc, ptc, wc, uc, vc, ua, va, ut, vt,
    divg_d`` -- the twin's own keys.  The ``delp``/``pt``/``w`` corner
    ghosts are filled INTERNALLY (``fill2_4corners``/``fill_4corners``)
    on the plain lane exactly as upstream; those fills are functional
    here and the mutated fields are threaded forward, but (like the
    twin) they are NOT returned, because ``c_sw`` does not export them.

    ``duogrid`` (STATIC feature gate) selects the symmetryclean DUO
    branch the production solver runs: ``d2a2c_vect_duo`` +
    ``divergence_corner_duo``, the simple upwind KE/vorticity (no
    ``sin_sg`` panel-edge special case) and the interior-everywhere
    vorticity transport, SKIPPING the corner fills and the
    corner-term removal (the duo halos carry real cross-face data).
    ``duogrid=False`` is the plain branch.

    PERMANENT lane guard, mirrored from the twin: ``fv_arrays.F90:1512``
    reads ``bounded_domain = regional .or. nested .or. duogrid``, so
    ``duogrid=True`` on unbounded metrics is a combination no upstream
    run can reach and it RAISES.  The implication is ONE-WAY -- bounded
    WITHOUT duo is the legitimate regional/nested category, refused
    separately at the KE branch as unported.

    Dependence and branch analysis (R1a / R1b):

    * the ut/vt scaling, the delp/pt/w upwind fluxes, the KE and
      vorticity upwind selects and the vorticity-flux selects are all
      ELEMENTWISE at one ``(i, j)`` -- each writes the cell it reads and
      no iteration reads another's write -> vectorised.  Every one of
      those data branches has two TOTAL arms (a gather or a product of
      written metrics), so a plain ``jnp.where`` is correct;
    * the plain KE/vorticity panel-edge cases and the plain
      vorticity-transport ``if (i==1 .or. i==npx)`` /
      ``if (j==1 .or. j==npy)`` cases are branches on the LOOP INDEX.
      They are resolved into STATIC index writes/runs, NOT a ``where``.
      This matters for more than literalness: the interior form
      ``dt2*(v - uc*cosa_u)/sina_u`` (sw_core.F90:420-434) is exactly
      the one the oracle REPLACES at the panel edges, so evaluating it
      there under a ``where`` would put a possible division blow-up into
      a dead arm and poison the reverse-mode gradient of the live one;
    * ``fx``/``fx1``/``fy``/``fy1`` are REUSED by the vorticity-transport
      stage, which partially overwrites them.  They are kept as the same
      arrays with partial writes (never reallocated), so any cell the
      oracle carries forward is carried forward here too;
    * the ``fxc``/``fyc`` nests complete before ``vortc`` reads them; the
      four corner-term removals and the ``fC + rarea_c*`` scaling are an
      ORDERED chain after it.
    """
    _validate_grid_type("c_sw", grid_type)
    _validate_nord("c_sw", "nord", nord)
    for nm, vv in (("hydrostatic", hydrostatic), ("dord4", dord4),
                   ("duogrid", duogrid),
                   ("bounded_domain", bounded_domain)):
        _require_bool("c_sw", nm, vv)
    _require_f64_jax("c_sw", {"delp": delp, "pt": pt, "w": w, "u": u,
                              "v": v, "dt2": jnp.asarray(dt2)})
    g = _geom("c_sw", gs, _CSW_KEYS)

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    iep1, jep1 = ie + 1, je + 1
    ni, nj = ied - isd + 1, jed - jsd + 1

    # fv_arrays.F90:1512 -- duogrid forces bounded_domain
    if duogrid and not bounded_domain:
        raise ValueError(
            "c_sw: duogrid=True requires bounded_domain=True "
            "(fv_arrays.F90:1512: bounded_domain = regional .or. nested "
            ".or. duogrid); duogrid on unbounded metrics is an "
            "upstream-impossible combination")

    delp = jnp.asarray(delp)
    pt = jnp.asarray(pt)
    w = jnp.asarray(w)
    u = jnp.asarray(u)
    v = jnp.asarray(v)
    _check_shape("c_sw", "delp", delp, (ni, nj))
    _check_shape("c_sw", "pt", pt, (ni, nj))
    _check_shape("c_sw", "w", w, (ni, nj))
    _check_shape("c_sw", "u", u, (ni, nj + 1))
    _check_shape("c_sw", "v", v, (ni + 1, nj))

    sg, cg = g["sin_sg"], g["cos_sg"]
    cosa_u, cosa_v = g["cosa_u"], g["cosa_v"]
    sina_u, sina_v = g["sina_u"], g["sina_v"]
    dx, dy, dxc, dyc = g["dx"], g["dy"], g["dxc"], g["dyc"]
    rdxc, rdyc = g["rdxc"], g["rdyc"]
    rarea, rarea_c, f_cor = g["rarea"], g["rarea_c"], g["fC"]

    def rd(a, i0, i1, j0, j1, k=None):
        return _fw(a, isd, jsd, i0, i1, j0, j1, k)

    # ---- D -> A -> C winds -----------------------------------------
    if duogrid:
        d2a = d2a2c_vect_duo(u, v, gs, bd, npx, npy, dord4=dord4,
                             grid_type=grid_type)
        ua, va = d2a["ua"], d2a["va"]
        uc, vc = d2a["uc"], d2a["vc"]
        ut, vt = d2a["ut"], d2a["vt"]
    else:
        ua, va, uc, vc, ut, vt = d2a2c_vect(
            u, v, gs, bd, npx, npy, dord4=dord4, grid_type=grid_type,
            bounded_domain=bounded_domain)

    if nord > 0:
        if duogrid:
            divg_d = divergence_corner_duo(u, v, ua, va, gs, bd, npx,
                                           npy, grid_type=grid_type)
        else:
            divg_d = divergence_corner(u, v, ua, va, gs, bd, npx, npy,
                                       grid_type=grid_type)
    else:
        divg_d = _new(isd, ied + 1, jsd, jed + 1)

    # ---- scale the transport winds (elementwise; both arms total) ---
    i0, i1, j0, j1 = is_ - 1, iep1 + 1, js - 1, jep1
    utw = _fw(ut, isd, jsd, i0, i1, j0, j1)
    dyw = rd(dy, i0, i1, j0, j1)
    ut = _fs(ut, isd, jsd, i0, i1, j0, j1,
             jnp.where(utw > 0.0,
                       dt2 * utw * dyw * rd(sg, i0 - 1, i1 - 1, j0, j1, 2),
                       dt2 * utw * dyw * rd(sg, i0, i1, j0, j1, 0)))
    i0, i1, j0, j1 = is_ - 1, iep1, js - 1, je + 2
    vtw = _fw(vt, isd, jsd, i0, i1, j0, j1)
    dxw = rd(dx, i0, i1, j0, j1)
    vt = _fs(vt, isd, jsd, i0, i1, j0, j1,
             jnp.where(vtw > 0.0,
                       dt2 * vtw * dxw * rd(sg, i0, i1, j0 - 1, j1 - 1, 3),
                       dt2 * vtw * dxw * rd(sg, i0, i1, j0, j1, 1)))

    # ---- transport delp (+ pt, and w if non-hydrostatic) -----------
    fx = _new(is_ - 1, ie + 2, js - 1, je + 1)
    fx1 = _new(is_ - 1, ie + 2, js - 1, je + 1)
    fx2 = _new(is_ - 1, ie + 2, js - 1, je + 1)
    fy = _new(is_ - 1, ie + 1, js - 1, je + 2)
    fy1 = _new(is_ - 1, ie + 1, js - 1, je + 2)
    fy2 = _new(is_ - 1, ie + 1, js - 1, je + 2)
    delpc = _new(isd, ied, jsd, jed)
    ptc = _new(isd, ied, jsd, jed)
    wc = _new(isd, ied, jsd, jed)

    plain_fills = (grid_type < 3) and (not bounded_domain) \
        and (not duogrid)

    # Xdir
    if plain_fills:
        delp, pt = fill2_4corners(delp, pt, 1, npx, npy, bd)
    if (not hydrostatic) and grid_type < 3 and (not duogrid):
        w = fill_4corners(w, 1, npx, npy, bd)
    i0, i1, j0, j1 = is_ - 1, ie + 2, js - 1, jep1
    utw = _fw(ut, isd, jsd, i0, i1, j0, j1)
    sel = utw > 0.0
    fx1v = utw * jnp.where(sel, rd(delp, i0 - 1, i1 - 1, j0, j1),
                           rd(delp, i0, i1, j0, j1))
    fxv = fx1v * jnp.where(sel, rd(pt, i0 - 1, i1 - 1, j0, j1),
                           rd(pt, i0, i1, j0, j1))
    fx1 = _fs(fx1, is_ - 1, js - 1, i0, i1, j0, j1, fx1v)
    fx = _fs(fx, is_ - 1, js - 1, i0, i1, j0, j1, fxv)
    if not hydrostatic:
        fx2 = _fs(fx2, is_ - 1, js - 1, i0, i1, j0, j1,
                  fx1v * jnp.where(sel, rd(w, i0 - 1, i1 - 1, j0, j1),
                                   rd(w, i0, i1, j0, j1)))

    # Ydir
    if plain_fills:
        delp, pt = fill2_4corners(delp, pt, 2, npx, npy, bd)
    if (not hydrostatic) and grid_type < 3 and (not duogrid):
        w = fill_4corners(w, 2, npx, npy, bd)
    i0, i1, j0, j1 = is_ - 1, iep1, js - 1, je + 2
    vtw = _fw(vt, isd, jsd, i0, i1, j0, j1)
    sel = vtw > 0.0
    fy1v = vtw * jnp.where(sel, rd(delp, i0, i1, j0 - 1, j1 - 1),
                           rd(delp, i0, i1, j0, j1))
    fyv = fy1v * jnp.where(sel, rd(pt, i0, i1, j0 - 1, j1 - 1),
                           rd(pt, i0, i1, j0, j1))
    fy1 = _fs(fy1, is_ - 1, js - 1, i0, i1, j0, j1, fy1v)
    fy = _fs(fy, is_ - 1, js - 1, i0, i1, j0, j1, fyv)
    if not hydrostatic:
        fy2 = _fs(fy2, is_ - 1, js - 1, i0, i1, j0, j1,
                  fy1v * jnp.where(sel, rd(w, i0, i1, j0 - 1, j1 - 1),
                                   rd(w, i0, i1, j0, j1)))

    def flx(a, i_a, i_b, j_a, j_b):
        return _fw(a, is_ - 1, js - 1, i_a, i_b, j_a, j_b)

    i0, i1, j0, j1 = is_ - 1, iep1, js - 1, jep1
    ra = rd(rarea, i0, i1, j0, j1)
    dp0 = rd(delp, i0, i1, j0, j1)
    pt0 = rd(pt, i0, i1, j0, j1)
    dpc = dp0 + (flx(fx1, i0, i1, j0, j1) - flx(fx1, i0 + 1, i1 + 1, j0, j1)
                 + flx(fy1, i0, i1, j0, j1)
                 - flx(fy1, i0, i1, j0 + 1, j1 + 1)) * ra
    delpc = _fs(delpc, isd, jsd, i0, i1, j0, j1, dpc)
    ptc = _fs(ptc, isd, jsd, i0, i1, j0, j1,
              (pt0 * dp0
               + (flx(fx, i0, i1, j0, j1) - flx(fx, i0 + 1, i1 + 1, j0, j1)
                  + flx(fy, i0, i1, j0, j1)
                  - flx(fy, i0, i1, j0 + 1, j1 + 1)) * ra) / dpc)
    if not hydrostatic:
        w0 = rd(w, i0, i1, j0, j1)
        wc = _fs(wc, isd, jsd, i0, i1, j0, j1,
                 (w0 * dp0
                  + (flx(fx2, i0, i1, j0, j1)
                     - flx(fx2, i0 + 1, i1 + 1, j0, j1)
                     + flx(fy2, i0, i1, j0, j1)
                     - flx(fy2, i0, i1, j0 + 1, j1 + 1)) * ra) / dpc)

    # ---- KE (cubed-sphere branch) ----------------------------------
    # Upstream's predicate is `bounded .or. grid_type>=3 .or. duogrid`
    # -> ONE simple-upwind branch; the duo lane below IS that branch, so
    # only a bounded / grid_type>=3 request WITHOUT duo is unported.
    if (bounded_domain or grid_type >= 3) and not duogrid:
        raise NotImplementedError(
            "c_sw: bounded/grid_type>=3 KE branch only ported via the "
            "duo lane (duogrid=True)")

    i0, i1, j0, j1 = is_ - 1, iep1, js - 1, jep1
    uaw = rd(ua, i0, i1, j0, j1)
    vaw = rd(va, i0, i1, j0, j1)
    ke_pos = _fw(uc, isd, jsd, i0, i1, j0, j1)
    ke_neg = _fw(uc, isd, jsd, i0 + 1, i1 + 1, j0, j1)
    vo_pos = _fw(vc, isd, jsd, i0, i1, j0, j1)
    vo_neg = _fw(vc, isd, jsd, i0, i1, j0 + 1, j1 + 1)

    if not duogrid:
        # STATIC index overrides of the generic value (the panel-edge
        # `sin_sg`/`cos_sg` form), applied at exactly the i / j the
        # oracle names -- no `where` on a loop index.
        def _seti(blk, i, val):
            return blk.at[i - i0, :].set(val[0, :])

        def _setj(blk, j, val):
            return blk.at[:, j - j0].set(val[:, 0])

        for i in (1, npx):
            if i0 <= i <= i1:
                ke_pos = _seti(
                    ke_pos, i,
                    _fw(uc, isd, jsd, i, i, j0, j1)
                    * rd(sg, i, i, j0, j1, 0)
                    + rd(v, i, i, j0, j1) * rd(cg, i, i, j0, j1, 0))
        for (i, src) in ((0, 1), (npx - 1, npx)):
            if i0 <= i <= i1:
                ke_neg = _seti(
                    ke_neg, i,
                    _fw(uc, isd, jsd, src, src, j0, j1)
                    * rd(sg, i, i, j0, j1, 2)
                    + rd(v, src, src, j0, j1) * rd(cg, i, i, j0, j1, 2))
        for j in (1, npy):
            if j0 <= j <= j1:
                vo_pos = _setj(
                    vo_pos, j,
                    _fw(vc, isd, jsd, i0, i1, j, j)
                    * rd(sg, i0, i1, j, j, 1)
                    + rd(u, i0, i1, j, j) * rd(cg, i0, i1, j, j, 1))
        for (j, src) in ((0, 1), (npy - 1, npy)):
            if j0 <= j <= j1:
                vo_neg = _setj(
                    vo_neg, j,
                    _fw(vc, isd, jsd, i0, i1, src, src)
                    * rd(sg, i0, i1, j, j, 3)
                    + rd(u, i0, i1, src, src) * rd(cg, i0, i1, j, j, 3))

    ke_w = jnp.where(uaw > 0.0, ke_pos, ke_neg)
    vort_w = jnp.where(vaw > 0.0, vo_pos, vo_neg)
    dt4 = 0.5 * dt2                                    # sw_core.F90:367
    ke_w = dt4 * (uaw * ke_w + vaw * vort_w)

    # ke_w carries Fortran bounds (is-1:ie+1, js-1:je+1)
    ke_i0, ke_j0 = is_ - 1, js - 1

    def ke_at(i_a, i_b, j_a, j_b):
        return ke_w[i_a - ke_i0:i_b - ke_i0 + 1,
                    j_a - ke_j0:j_b - ke_j0 + 1]

    # ---- circulation on the C grid ---------------------------------
    fxc = _new(is_ - 1, ie + 2, js - 1, je + 1)
    fyc = _new(is_ - 1, ie + 1, js - 1, je + 2)
    fxc = _fs(fxc, is_ - 1, js - 1, is_, ie + 1, js - 1, je + 1,
              _fw(uc, isd, jsd, is_, ie + 1, js - 1, je + 1)
              * rd(dxc, is_, ie + 1, js - 1, je + 1))
    fyc = _fs(fyc, is_ - 1, js - 1, is_ - 1, ie + 1, js, je + 1,
              _fw(vc, isd, jsd, is_ - 1, ie + 1, js, je + 1)
              * rd(dyc, is_ - 1, ie + 1, js, je + 1))

    def fxc_at(i_a, i_b, j_a, j_b):
        return _fw(fxc, is_ - 1, js - 1, i_a, i_b, j_a, j_b)

    def fyc_at(i_a, i_b, j_a, j_b):
        return _fw(fyc, is_ - 1, js - 1, i_a, i_b, j_a, j_b)

    vortc = _new(is_, ie + 1, js, je + 1)
    vortc = _fs(vortc, is_, js, is_, ie + 1, js, je + 1,
                fxc_at(is_, ie + 1, js - 1, je) - fxc_at(is_, ie + 1, js, je + 1)
                - fyc_at(is_ - 1, ie, js, je + 1)
                + fyc_at(is_, ie + 1, js, je + 1))

    if not duogrid:
        # sw_core.F90:395-401 (.not.duogrid -- the duo halos carry real
        # cross-face circulation)
        for (i, j, si, sgn) in ((1, 1, 0, +1.0), (npx, 1, npx, -1.0),
                                (npx, npy, npx, -1.0),
                                (1, npy, 0, +1.0)):
            vortc = _fs(vortc, is_, js, i, i, j, j,
                        _fw(vortc, is_, js, i, i, j, j)
                        + sgn * fyc_at(si, si, j, j))

    vortc = _fs(vortc, is_, js, is_, ie + 1, js, je + 1,
                rd(f_cor, is_, ie + 1, js, je + 1)
                + rd(rarea_c, is_, ie + 1, js, je + 1)
                * _fw(vortc, is_, js, is_, ie + 1, js, je + 1))

    def vortc_at(i_a, i_b, j_a, j_b):
        return _fw(vortc, is_, js, i_a, i_b, j_a, j_b)

    # ---- transport absolute vorticity ------------------------------
    # fy1/fy on (is:ie+1, js:je); fx1/fx on (is:ie, js:je+1).
    def _fy1_cls(i):
        return "edge" if (i == 1 or i == npx) else "interior"

    def _fx1_cls(j):
        return "edge" if (j == 1 or j == npy) else "interior"

    if js <= je:
        for cls, ia2, ib2 in ([("interior", is_, iep1)] if duogrid
                              else _runs(is_, iep1, _fy1_cls)):
            vv = rd(v, ia2, ib2, js, je)
            if cls == "edge":
                f1 = dt2 * vv
            else:
                f1 = dt2 * (vv - _fw(uc, isd, jsd, ia2, ib2, js, je)
                            * rd(cosa_u, ia2, ib2, js, je)) \
                    / rd(sina_u, ia2, ib2, js, je)
            fy1 = _fs(fy1, is_ - 1, js - 1, ia2, ib2, js, je, f1)
            fy = _fs(fy, is_ - 1, js - 1, ia2, ib2, js, je,
                     jnp.where(f1 > 0.0, vortc_at(ia2, ib2, js, je),
                               vortc_at(ia2, ib2, js + 1, je + 1)))

    if is_ <= ie:
        for cls, ja2, jb2 in ([("interior", js, jep1)] if duogrid
                              else _runs(js, jep1, _fx1_cls)):
            uu = rd(u, is_, ie, ja2, jb2)
            if cls == "edge":
                f1 = dt2 * uu
            else:
                f1 = dt2 * (uu - _fw(vc, isd, jsd, is_, ie, ja2, jb2)
                            * rd(cosa_v, is_, ie, ja2, jb2)) \
                    / rd(sina_v, is_, ie, ja2, jb2)
            fx1 = _fs(fx1, is_ - 1, js - 1, is_, ie, ja2, jb2, f1)
            fx = _fs(fx, is_ - 1, js - 1, is_, ie, ja2, jb2,
                     jnp.where(f1 > 0.0, vortc_at(is_, ie, ja2, jb2),
                               vortc_at(is_ + 1, ie + 1, ja2, jb2)))

    # ---- update the time-centred C-grid winds ----------------------
    uc = _fs(uc, isd, jsd, is_, iep1, js, je,
             _fw(uc, isd, jsd, is_, iep1, js, je)
             + flx(fy1, is_, iep1, js, je) * flx(fy, is_, iep1, js, je)
             + rd(rdxc, is_, iep1, js, je)
             * (ke_at(is_ - 1, ie, js, je) - ke_at(is_, iep1, js, je)))
    vc = _fs(vc, isd, jsd, is_, ie, js, jep1,
             _fw(vc, isd, jsd, is_, ie, js, jep1)
             - flx(fx1, is_, ie, js, jep1) * flx(fx, is_, ie, js, jep1)
             + rd(rdyc, is_, ie, js, jep1)
             * (ke_at(is_, ie, js - 1, je) - ke_at(is_, ie, js, jep1)))

    return {"delpc": delpc, "ptc": ptc, "wc": wc, "uc": uc, "vc": vc,
            "ua": ua, "va": va, "ut": ut, "vt": vt, "divg_d": divg_d}


# =====================================================================
# d_sw1  (sw_core.F90:500-998) -- the D-grid TRANSPORT stage
# =====================================================================

_DSW1_KEYS = ("area", "sin_sg", "cosa_u", "cosa_v", "rsin_u", "rsin_v",
              "dx", "dy", "rdxa", "rdya", "dxa", "dya", "rarea",
              "del6_v", "del6_u")


def d_sw1_duo(delp, pt, w, uc, vc, xflux, yflux, cx, cy, gs: dict,
              flags: GridFlags, bd, npx: int, npy: int, *, dt,
              hord_tr: int = 8, hord_vt: int = 6, hord_tm: int = 6,
              hord_dp: int = 6, nord_v: int = 1, nord_t: int = 0,
              damp_v: float = 0.2, damp_t: float = 0.0,
              hydrostatic: bool = True, inline_q: bool = False,
              lim_fac: float = 1.0, duogrid: bool = True,
              workspace_sentinel: float = 1.0e30):
    """JAX twin of ``fv3_native_duo_sw_core.d_sw1_duo``
    (sw_core.F90:500-998, DUO branch).

    ``d_sw1`` computes FLUXES ONLY -- the delp/pt UPDATES happen in
    ``d_sw2`` AFTER dyn_core's inter-panel flux averaging
    (``dyn_core.F90:872``, barrier 1).  Returns a dict with
    ``crx_adv/cry_adv/xfx_adv/yfx_adv``, ``ra_x/ra_y``, ``ut/vt``,
    ``allflux_x/allflux_y`` (k=1 slab, ``4+nq = 5`` slots; slot 2 is
    written only on the NH lane, slots 3/5 stay NaN), the (corner-ghost
    mutated on the plain-fill lanes) ``delp/pt/w``, and the accumulated
    ``cx/cy/xflux/yflux`` capacitors.

    WORKSPACE CONTRACT (mirrored verbatim from the twin, and the reason
    ``ut``/``vt`` are ``workspace_sentinel``-filled rather than NaN):
    the panel-edge and corner blocks fire on the plain-conventions lane
    and READ ut/vt cells the duo interior never writes -- in dyn_core
    those are uninitialised stack memory.  The oracle extract therefore
    certifies a DEFINED contract in which both sides initialise ut/vt to
    the same sentinel, making those cells deterministic and
    bit-comparable.  A NaN fill here would break that certificate.

    Dependence (R1a), stage by stage:

    * the duo ut/vt interior nests write one array and read ``uc``/
      ``vc``/metrics -> vectorised;
    * the four panel-edge blocks are an ORDERED chain and each has an
      internal order that matters: West/East write ``ut`` at i = 1 / npx
      and the following ``vt`` writes READ those cells; South/North
      write ``vt`` at j = 1 / npy and the following ``ut`` writes read
      them.  Within one block, the two written rows/columns never read
      each other, so each vectorises over the free index;
    * the four corner 2x2 systems are emitted as an ORDERED chain of
      single-point writes -- literal, and immune to any re-ordering;
    * ``crx``/``xfx`` and ``cry``/``yfx`` read the OLD ``xfx_adv`` at the
      same ``(i, j)`` and then overwrite it; computing both from the old
      value in one vectorised step is exact;
    * the three ``fv_tp_2d`` calls are threaded in source ORDER, and the
      ``pt`` transport receives the ``delp`` RETURNED by the first call
      (upstream passes the same array object, whose corner ghosts the
      first call may have rotated).

    R1b: every data branch here (the ``crx``/``cry`` upwind pair and the
    four ``uc*dt > 0`` / ``vc*dt > 0`` panel-edge selects) uses
    ``jnp.where``.  The four panel-edge selects are the only ones whose
    arms DIVIDE -- ``uc(1,j)/sin_sg(0,j,3)`` versus
    ``uc(1,j)/sin_sg(1,j,1)`` (sw_core.F90:658-663).  Both arms are total
    on an admitted gridstruct, where ``sin_sg`` (the sine of the angle
    between grid lines) is strictly positive; that is a stated
    PRECONDITION of the operand, and it cannot be checked statically
    because ``sin_sg`` is traced.  Masking one arm is not an option: the
    two arms divide by DIFFERENT metric entries, so a mask would change
    the live answer.
    """
    for nm, vv in (("hord_tr", hord_tr), ("hord_vt", hord_vt),
                   ("hord_tm", hord_tm), ("hord_dp", hord_dp)):
        _validate_ord("d_sw1_duo", nm, vv, _PPM_ORDS)
    _validate_nord("d_sw1_duo", "nord_v", nord_v)
    _validate_nord("d_sw1_duo", "nord_t", nord_t)
    _validate_grid_type("d_sw1_duo", flags.grid_type)
    for nm, vv in (("hydrostatic", hydrostatic), ("inline_q", inline_q),
                   ("duogrid", duogrid)):
        _require_bool("d_sw1_duo", nm, vv)
    if not duogrid:
        raise NotImplementedError(
            "d_sw1_duo is the DUO-stage port; the plain path is the "
            "certified monolithic d_sw (phase-4b)")
    if inline_q:
        raise NotImplementedError(
            "d_sw1_duo: inline_q=False lane only (matches the oracle "
            "driver; q_con/tracer transports not exercised)")
    _require_f64_jax("d_sw1_duo", {
        "delp": delp, "pt": pt, "w": w, "uc": uc, "vc": vc,
        "xflux": xflux, "yflux": yflux, "cx": cx, "cy": cy,
        "dt": jnp.asarray(dt)})
    g = _geom("d_sw1_duo", gs, _DSW1_KEYS)

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    nid, njd = ied - isd + 1, jed - jsd + 1
    nci, ncj = ie - is_ + 1, je - js + 1

    delp, pt, w = jnp.asarray(delp), jnp.asarray(pt), jnp.asarray(w)
    uc, vc = jnp.asarray(uc), jnp.asarray(vc)
    xflux, yflux = jnp.asarray(xflux), jnp.asarray(yflux)
    cx, cy = jnp.asarray(cx), jnp.asarray(cy)
    _check_shape("d_sw1_duo", "delp", delp, (nid, njd))
    _check_shape("d_sw1_duo", "pt", pt, (nid, njd))
    _check_shape("d_sw1_duo", "w", w, (nid, njd))
    _check_shape("d_sw1_duo", "uc", uc, (nid + 1, njd))
    _check_shape("d_sw1_duo", "vc", vc, (nid, njd + 1))
    _check_shape("d_sw1_duo", "xflux", xflux, (nci + 1, ncj))
    _check_shape("d_sw1_duo", "yflux", yflux, (nci, ncj + 1))
    _check_shape("d_sw1_duo", "cx", cx, (nci + 1, njd))
    _check_shape("d_sw1_duo", "cy", cy, (nid, ncj + 1))

    area, sg = g["area"], g["sin_sg"]
    cosa_u, cosa_v = g["cosa_u"], g["cosa_v"]
    rsin_u, rsin_v = g["rsin_u"], g["rsin_v"]
    dx, dy, rdxa, rdya = g["dx"], g["dy"], g["rdxa"], g["rdya"]

    def rd(a, i0, i1, j0, j1, k=None):
        return _fw(a, isd, jsd, i0, i1, j0, j1, k)

    ut = _new(isd, ied + 1, jsd, jed, workspace_sentinel)
    vt = _new(isd, ied, jsd, jed + 1, workspace_sentinel)
    crx_adv = _new(is_, ie + 1, jsd, jed)
    xfx_adv = _new(is_, ie + 1, jsd, jed)
    cry_adv = _new(isd, ied, js, je + 1)
    yfx_adv = _new(isd, ied, js, je + 1)
    ra_x = _new(is_, ie, jsd, jed)
    ra_y = _new(isd, ied, js, je)
    nq = 1
    allflux_x = jnp.full((nci + 1, ncj, 4 + nq), jnp.nan, jnp.float64)
    allflux_y = jnp.full((nci, ncj + 1, 4 + nq), jnp.nan, jnp.float64)

    def ut_at(i0, i1, j0, j1):
        return _fw(ut, isd, jsd, i0, i1, j0, j1)

    def vt_at(i0, i1, j0, j1):
        return _fw(vt, isd, jsd, i0, i1, j0, j1)

    def set_ut(i0, i1, j0, j1, val):
        return _fs(ut, isd, jsd, i0, i1, j0, j1, val)

    def set_vt(i0, i1, j0, j1, val):
        return _fs(vt, isd, jsd, i0, i1, j0, j1, val)

    # ---- ut/vt DUO interior (sw_core.F90:621-634) -------------------
    ut = set_ut(is_, ie + 1, jsd, jed,
               (_fw(uc, isd, jsd, is_, ie + 1, jsd, jed)
                - 0.25 * rd(cosa_u, is_, ie + 1, jsd, jed)
                * (_fw(vc, isd, jsd, is_ - 1, ie, jsd, jed)
                   + _fw(vc, isd, jsd, is_, ie + 1, jsd, jed)
                   + _fw(vc, isd, jsd, is_ - 1, ie, jsd + 1, jed + 1)
                   + _fw(vc, isd, jsd, is_, ie + 1, jsd + 1, jed + 1)))
               * rd(rsin_u, is_, ie + 1, jsd, jed))
    vt = set_vt(isd, ied, js, je + 1,
               (_fw(vc, isd, jsd, isd, ied, js, je + 1)
                - 0.25 * rd(cosa_v, isd, ied, js, je + 1)
                * (_fw(uc, isd, jsd, isd, ied, js - 1, je)
                   + _fw(uc, isd, jsd, isd + 1, ied + 1, js - 1, je)
                   + _fw(uc, isd, jsd, isd, ied, js, je + 1)
                   + _fw(uc, isd, jsd, isd + 1, ied + 1, js, je + 1)))
               * rd(rsin_v, isd, ied, js, je + 1))

    # ---- panel edges (sw_core.F90:656-726) -------------------------
    # Guard `.not.bounded .or. .not.duogrid` (:656): fires on the
    # plain-conventions lane (bounded=F) and is SKIPPED in the real duo
    # runs (bounded=T), where the interior formula stands at the panel
    # edges and reads the bounded gridstruct's real edge rsin_u/rsin_v.
    plain_edges = not flags.bounded_domain
    if plain_edges and is_ == 1:                       # West edge
        ucc = _fw(uc, isd, jsd, 1, 1, jsd, jed)
        ut = set_ut(1, 1, jsd, jed,
                   jnp.where(ucc * dt > 0.0,
                             ucc / rd(sg, 0, 0, jsd, jed, 2),
                             ucc / rd(sg, 1, 1, jsd, jed, 0)))
        ja, jb = max(3, js), min(npy - 2, je + 1)
        if ja <= jb:
            for i in (0, 1):
                vt = set_vt(i, i, ja, jb,
                           _fw(vc, isd, jsd, i, i, ja, jb)
                           - 0.25 * rd(cosa_v, i, i, ja, jb)
                           * (ut_at(i, i, ja - 1, jb - 1)
                              + ut_at(i + 1, i + 1, ja - 1, jb - 1)
                              + ut_at(i, i, ja, jb)
                              + ut_at(i + 1, i + 1, ja, jb)))
    if plain_edges and (ie + 1) == npx:                # East edge
        ucc = _fw(uc, isd, jsd, npx, npx, jsd, jed)
        ut = set_ut(npx, npx, jsd, jed,
                   jnp.where(ucc * dt > 0.0,
                             ucc / rd(sg, npx - 1, npx - 1, jsd, jed, 2),
                             ucc / rd(sg, npx, npx, jsd, jed, 0)))
        ja, jb = max(3, js), min(npy - 2, je + 1)
        if ja <= jb:
            for i in (npx - 1, npx):
                vt = set_vt(i, i, ja, jb,
                           _fw(vc, isd, jsd, i, i, ja, jb)
                           - 0.25 * rd(cosa_v, i, i, ja, jb)
                           * (ut_at(i, i, ja - 1, jb - 1)
                              + ut_at(i + 1, i + 1, ja - 1, jb - 1)
                              + ut_at(i, i, ja, jb)
                              + ut_at(i + 1, i + 1, ja, jb)))
    if plain_edges and js == 1:                        # South edge
        vcc = _fw(vc, isd, jsd, isd, ied, 1, 1)
        vt = set_vt(isd, ied, 1, 1,
                   jnp.where(vcc * dt > 0.0,
                             vcc / rd(sg, isd, ied, 0, 0, 3),
                             vcc / rd(sg, isd, ied, 1, 1, 1)))
        ia, ib = max(3, is_), min(npx - 2, ie + 1)
        if ia <= ib:
            for j in (0, 1):
                ut = set_ut(ia, ib, j, j,
                           _fw(uc, isd, jsd, ia, ib, j, j)
                           - 0.25 * rd(cosa_u, ia, ib, j, j)
                           * (vt_at(ia - 1, ib - 1, j, j)
                              + vt_at(ia, ib, j, j)
                              + vt_at(ia - 1, ib - 1, j + 1, j + 1)
                              + vt_at(ia, ib, j + 1, j + 1)))
    if plain_edges and (je + 1) == npy:                # North edge
        vcc = _fw(vc, isd, jsd, isd, ied, npy, npy)
        vt = set_vt(isd, ied, npy, npy,
                   jnp.where(vcc * dt > 0.0,
                             vcc / rd(sg, isd, ied, npy - 1, npy - 1, 3),
                             vcc / rd(sg, isd, ied, npy, npy, 1)))
        ia, ib = max(3, is_), min(npx - 2, ie + 1)
        if ia <= ib:
            for j in (npy - 1, npy):
                ut = set_ut(ia, ib, j, j,
                           _fw(uc, isd, jsd, ia, ib, j, j)
                           - 0.25 * rd(cosa_u, ia, ib, j, j)
                           * (vt_at(ia - 1, ib - 1, j, j)
                              + vt_at(ia, ib, j, j)
                              + vt_at(ia - 1, ib - 1, j + 1, j + 1)
                              + vt_at(ia, ib, j + 1, j + 1)))

    # ---- corner 2x2 systems (sw_core.F90:739-811), ORDERED ----------
    def cu(i, j):
        return rd(cosa_u, i, i, j, j)

    def cv(i, j):
        return rd(cosa_v, i, i, j, j)

    def ucp(i, j):
        return _fw(uc, isd, jsd, i, i, j, j)

    def vcp(i, j):
        return _fw(vc, isd, jsd, i, i, j, j)

    if flags.sw_corner:
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(2, 0) * cv(1, 0))
        ut = set_ut(2, 2, 0, 0,
                   (ucp(2, 0) - 0.25 * cu(2, 0)
                    * (vt_at(1, 1, 1, 1) + vt_at(2, 2, 1, 1) + vt_at(2, 2, 0, 0)
                       + vcp(1, 0)
                       - 0.25 * cv(1, 0) * (ut_at(1, 1, 0, 0)
                                            + ut_at(1, 1, -1, -1)
                                            + ut_at(2, 2, -1, -1)))) * damp)
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(0, 1) * cv(0, 2))
        vt = set_vt(0, 0, 2, 2,
                   (vcp(0, 2) - 0.25 * cv(0, 2)
                    * (ut_at(1, 1, 1, 1) + ut_at(1, 1, 2, 2) + ut_at(0, 0, 2, 2)
                       + ucp(0, 1)
                       - 0.25 * cu(0, 1) * (vt_at(0, 0, 1, 1)
                                            + vt_at(-1, -1, 1, 1)
                                            + vt_at(-1, -1, 2, 2)))) * damp)
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(2, 1) * cv(1, 2))
        ut = set_ut(2, 2, 1, 1,
                   (ucp(2, 1) - 0.25 * cu(2, 1)
                    * (vt_at(1, 1, 1, 1) + vt_at(2, 2, 1, 1) + vt_at(2, 2, 2, 2)
                       + vcp(1, 2)
                       - 0.25 * cv(1, 2) * (ut_at(1, 1, 1, 1)
                                            + ut_at(1, 1, 2, 2)
                                            + ut_at(2, 2, 2, 2)))) * damp)
        vt = set_vt(1, 1, 2, 2,
                   (vcp(1, 2) - 0.25 * cv(1, 2)
                    * (ut_at(1, 1, 1, 1) + ut_at(1, 1, 2, 2) + ut_at(2, 2, 2, 2)
                       + ucp(2, 1)
                       - 0.25 * cu(2, 1) * (vt_at(1, 1, 1, 1)
                                            + vt_at(2, 2, 1, 1)
                                            + vt_at(2, 2, 2, 2)))) * damp)
    if flags.se_corner:
        p = npx
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(p - 1, 0) * cv(p - 1, 0))
        ut = set_ut(p - 1, p - 1, 0, 0,
                   (ucp(p - 1, 0) - 0.25 * cu(p - 1, 0)
                    * (vt_at(p - 1, p - 1, 1, 1) + vt_at(p - 2, p - 2, 1, 1)
                       + vt_at(p - 2, p - 2, 0, 0) + vcp(p - 1, 0)
                       - 0.25 * cv(p - 1, 0)
                       * (ut_at(p, p, 0, 0) + ut_at(p, p, -1, -1)
                          + ut_at(p - 1, p - 1, -1, -1)))) * damp)
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(p + 1, 1) * cv(p, 2))
        vt = set_vt(p, p, 2, 2,
                   (vcp(p, 2) - 0.25 * cv(p, 2)
                    * (ut_at(p, p, 1, 1) + ut_at(p, p, 2, 2)
                       + ut_at(p + 1, p + 1, 2, 2) + ucp(p + 1, 1)
                       - 0.25 * cu(p + 1, 1)
                       * (vt_at(p, p, 1, 1) + vt_at(p + 1, p + 1, 1, 1)
                          + vt_at(p + 1, p + 1, 2, 2)))) * damp)
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(p - 1, 1) * cv(p - 1, 2))
        ut = set_ut(p - 1, p - 1, 1, 1,
                   (ucp(p - 1, 1) - 0.25 * cu(p - 1, 1)
                    * (vt_at(p - 1, p - 1, 1, 1) + vt_at(p - 2, p - 2, 1, 1)
                       + vt_at(p - 2, p - 2, 2, 2) + vcp(p - 1, 2)
                       - 0.25 * cv(p - 1, 2)
                       * (ut_at(p, p, 1, 1) + ut_at(p, p, 2, 2)
                          + ut_at(p - 1, p - 1, 2, 2)))) * damp)
        vt = set_vt(p - 1, p - 1, 2, 2,
                   (vcp(p - 1, 2) - 0.25 * cv(p - 1, 2)
                    * (ut_at(p, p, 1, 1) + ut_at(p, p, 2, 2)
                       + ut_at(p - 1, p - 1, 2, 2) + ucp(p - 1, 1)
                       - 0.25 * cu(p - 1, 1)
                       * (vt_at(p - 1, p - 1, 1, 1) + vt_at(p - 2, p - 2, 1, 1)
                          + vt_at(p - 2, p - 2, 2, 2)))) * damp)
    if flags.ne_corner:
        p, q = npx, npy
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(p - 1, q) * cv(p - 1, q + 1))
        ut = set_ut(p - 1, p - 1, q, q,
                   (ucp(p - 1, q) - 0.25 * cu(p - 1, q)
                    * (vt_at(p - 1, p - 1, q, q) + vt_at(p - 2, p - 2, q, q)
                       + vt_at(p - 2, p - 2, q + 1, q + 1) + vcp(p - 1, q + 1)
                       - 0.25 * cv(p - 1, q + 1)
                       * (ut_at(p, p, q, q) + ut_at(p, p, q + 1, q + 1)
                          + ut_at(p - 1, p - 1, q + 1, q + 1)))) * damp)
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(p + 1, q - 1) * cv(p, q - 1))
        vt = set_vt(p, p, q - 1, q - 1,
                   (vcp(p, q - 1) - 0.25 * cv(p, q - 1)
                    * (ut_at(p, p, q - 1, q - 1) + ut_at(p, p, q - 2, q - 2)
                       + ut_at(p + 1, p + 1, q - 2, q - 2) + ucp(p + 1, q - 1)
                       - 0.25 * cu(p + 1, q - 1)
                       * (vt_at(p, p, q, q) + vt_at(p + 1, p + 1, q, q)
                          + vt_at(p + 1, p + 1, q - 1, q - 1)))) * damp)
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(p - 1, q - 1)
                      * cv(p - 1, q - 1))
        ut = set_ut(p - 1, p - 1, q - 1, q - 1,
                   (ucp(p - 1, q - 1) - 0.25 * cu(p - 1, q - 1)
                    * (vt_at(p - 1, p - 1, q, q) + vt_at(p - 2, p - 2, q, q)
                       + vt_at(p - 2, p - 2, q - 1, q - 1) + vcp(p - 1, q - 1)
                       - 0.25 * cv(p - 1, q - 1)
                       * (ut_at(p, p, q - 1, q - 1) + ut_at(p, p, q - 2, q - 2)
                          + ut_at(p - 1, p - 1, q - 2, q - 2)))) * damp)
        vt = set_vt(p - 1, p - 1, q - 1, q - 1,
                   (vcp(p - 1, q - 1) - 0.25 * cv(p - 1, q - 1)
                    * (ut_at(p, p, q - 1, q - 1) + ut_at(p, p, q - 2, q - 2)
                       + ut_at(p - 1, p - 1, q - 2, q - 2) + ucp(p - 1, q - 1)
                       - 0.25 * cu(p - 1, q - 1)
                       * (vt_at(p - 1, p - 1, q, q) + vt_at(p - 2, p - 2, q, q)
                          + vt_at(p - 2, p - 2, q - 1, q - 1)))) * damp)
    if flags.nw_corner:
        q = npy
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(2, q) * cv(1, q + 1))
        ut = set_ut(2, 2, q, q,
                   (ucp(2, q) - 0.25 * cu(2, q)
                    * (vt_at(1, 1, q, q) + vt_at(2, 2, q, q)
                       + vt_at(2, 2, q + 1, q + 1) + vcp(1, q + 1)
                       - 0.25 * cv(1, q + 1)
                       * (ut_at(1, 1, q, q) + ut_at(1, 1, q + 1, q + 1)
                          + ut_at(2, 2, q + 1, q + 1)))) * damp)
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(0, q - 1) * cv(0, q - 1))
        vt = set_vt(0, 0, q - 1, q - 1,
                   (vcp(0, q - 1) - 0.25 * cv(0, q - 1)
                    * (ut_at(1, 1, q - 1, q - 1) + ut_at(1, 1, q - 2, q - 2)
                       + ut_at(0, 0, q - 2, q - 2) + ucp(0, q - 1)
                       - 0.25 * cu(0, q - 1)
                       * (vt_at(0, 0, q, q) + vt_at(-1, -1, q, q)
                          + vt_at(-1, -1, q - 1, q - 1)))) * damp)
        damp = 1.0 / (1.0 - _CORNER_DAMP_COEF * cu(2, q - 1) * cv(1, q - 1))
        ut = set_ut(2, 2, q - 1, q - 1,
                   (ucp(2, q - 1) - 0.25 * cu(2, q - 1)
                    * (vt_at(1, 1, q, q) + vt_at(2, 2, q, q)
                       + vt_at(2, 2, q - 1, q - 1) + vcp(1, q - 1)
                       - 0.25 * cv(1, q - 1)
                       * (ut_at(1, 1, q - 1, q - 1) + ut_at(1, 1, q - 2, q - 2)
                          + ut_at(2, 2, q - 2, q - 2)))) * damp)
        vt = set_vt(1, 1, q - 1, q - 1,
                   (vcp(1, q - 1) - 0.25 * cv(1, q - 1)
                    * (ut_at(1, 1, q - 1, q - 1) + ut_at(1, 1, q - 2, q - 2)
                       + ut_at(2, 2, q - 2, q - 2) + ucp(2, q - 1)
                       - 0.25 * cu(2, q - 1)
                       * (vt_at(1, 1, q, q) + vt_at(2, 2, q, q)
                          + vt_at(2, 2, q - 1, q - 1)))) * damp)

    # ---- xfx/crx, yfx/cry (sw_core.F90:830-869) --------------------
    # Elementwise: crx and the NEW xfx are both built from the OLD xfx
    # at the SAME (i,j), so the in-place overwrite vectorises exactly.
    xf = dt * ut_at(is_, ie + 1, jsd, jed)
    selx = xf > 0.0
    crx_adv = _fs(crx_adv, is_, jsd, is_, ie + 1, jsd, jed,
                  jnp.where(selx, xf * rd(rdxa, is_ - 1, ie, jsd, jed),
                            xf * rd(rdxa, is_, ie + 1, jsd, jed)))
    dyw = rd(dy, is_, ie + 1, jsd, jed)
    xfx_adv = _fs(xfx_adv, is_, jsd, is_, ie + 1, jsd, jed,
                  jnp.where(selx,
                            dyw * xf * rd(sg, is_ - 1, ie, jsd, jed, 2),
                            dyw * xf * rd(sg, is_, ie + 1, jsd, jed, 0)))
    yf = dt * vt_at(isd, ied, js, je + 1)
    sely = yf > 0.0
    cry_adv = _fs(cry_adv, isd, js, isd, ied, js, je + 1,
                  jnp.where(sely, yf * rd(rdya, isd, ied, js - 1, je),
                            yf * rd(rdya, isd, ied, js, je + 1)))
    dxw = rd(dx, isd, ied, js, je + 1)
    yfx_adv = _fs(yfx_adv, isd, js, isd, ied, js, je + 1,
                  jnp.where(sely,
                            dxw * yf * rd(sg, isd, ied, js - 1, je, 3),
                            dxw * yf * rd(sg, isd, ied, js, je + 1, 1)))

    # ---- ra_x/ra_y (sw_core.F90:875-884) ---------------------------
    ra_x = _fs(ra_x, is_, jsd, is_, ie, jsd, jed,
               rd(area, is_, ie, jsd, jed)
               + _fw(xfx_adv, is_, jsd, is_, ie, jsd, jed)
               - _fw(xfx_adv, is_, jsd, is_ + 1, ie + 1, jsd, jed))
    ra_y = _fs(ra_y, isd, js, isd, ied, js, je,
               rd(area, isd, ied, js, je)
               + _fw(yfx_adv, isd, js, isd, ied, js, je)
               - _fw(yfx_adv, isd, js, isd, ied, js + 1, je + 1))

    tp_args = (npx, npy)
    tp_geom = (g["dxa"], g["dya"], area, g["del6_v"], g["del6_u"],
               g["rarea"], flags.da_min, bd, ra_x, ra_y, lim_fac,
               flags.bounded_domain, flags.grid_type, flags.sw_corner,
               flags.se_corner, flags.nw_corner, flags.ne_corner)

    # ---- delp fluxes (sw_core.F90:886-887) + slot 1 + capacitors ----
    delp, fx, fy = fv_tp_2d(delp, crx_adv, cry_adv, *tp_args, hord_dp,
                            xfx_adv, yfx_adv, *tp_geom, nord=nord_v,
                            damp_c=damp_v, duogrid=duogrid)
    allflux_x = allflux_x.at[:, :, 0].set(fx)
    allflux_y = allflux_y.at[:, :, 0].set(fy)
    cx = cx + crx_adv
    xflux = xflux + fx
    cy = cy + cry_adv
    yflux = yflux + fy

    # ---- NH w fluxes (sw_core.F90:923-939) -> slot 2 ---------------
    # BEFORE the pt transport, exactly as upstream -- gx/gy are then
    # REUSED for pt, so the order is load-bearing.
    if not hydrostatic:
        w, gx, gy = fv_tp_2d(w, crx_adv, cry_adv, *tp_args, hord_vt,
                             xfx_adv, yfx_adv, *tp_geom, mfx=fx, mfy=fy,
                             duogrid=duogrid)
        allflux_x = allflux_x.at[:, :, 1].set(gx)
        allflux_y = allflux_y.at[:, :, 1].set(gy)

    # ---- pt fluxes (sw_core.F90:959-961: nord=nord_v, damp_c=damp_v)
    pt, gx, gy = fv_tp_2d(pt, crx_adv, cry_adv, *tp_args, hord_tm,
                          xfx_adv, yfx_adv, *tp_geom, mfx=fx, mfy=fy,
                          mass=delp, nord=nord_v, damp_c=damp_v,
                          duogrid=duogrid)
    allflux_x = allflux_x.at[:, :, 3].set(gx)
    allflux_y = allflux_y.at[:, :, 3].set(gy)

    return {"crx_adv": crx_adv, "cry_adv": cry_adv,
            "xfx_adv": xfx_adv, "yfx_adv": yfx_adv,
            "ra_x": ra_x, "ra_y": ra_y, "ut": ut, "vt": vt,
            "allflux_x": allflux_x, "allflux_y": allflux_y,
            "delp": delp, "pt": pt, "w": w,
            "cx": cx, "cy": cy, "xflux": xflux, "yflux": yflux}


# =====================================================================
# d_sw2  (sw_core.F90:1000-1199) -- the post-averaging UPDATE stage
# =====================================================================

def d_sw2_duo(delp, pt, allflux_x, allflux_y, gs: dict,
              flags: GridFlags, bd, *, w=None, npx: int | None = None,
              npy: int | None = None, dt=0.0, kgb=0.0, nord_w: int = 2,
              damp_w: float = 0.0, hydrostatic: bool = True,
              inline_q: bool = False,
              workspace_sentinel: float = 1.0e30):
    """JAX twin of ``fv3_native_duo_sw_core.d_sw2_duo``
    (sw_core.F90:1000-1199) -- the delp/pt UPDATE that runs AFTER
    dyn_core's inter-panel flux averaging (barrier 1,
    ``dyn_core.F90:872``), on the oracle lane (``inline_q=F``, no
    ``SW_DYNAMICS``/``USE_COND``).

    ``d_sw2`` has NO duo/edge branches -- it is a pure cell update given
    the fluxes -- so there is no ``jnp.where`` in it at all.

    ``allflux_x``/``allflux_y`` are the ``d_sw1_duo`` output stacks
    (slot axis last); slot 1 carries the delp flux and slot 4 the pt
    flux, slot 2 the NH w flux.  ``ptc`` (upstream ``intent(OUT)``,
    shimmed to inout in the extract so the sentinel round-trip is
    standard-defined) is never written on this lane, and ``dw`` is
    written only under ``damp_w > 1e-5``; both return
    ``workspace_sentinel`` fills, mirroring the oracle driver's 1e30
    init.  Returns ``dict(delp, pt, w, heat_source, ptc, dw)``.

    Dependence (R1a): the update body is elementwise at one ``(i, j)``.
    ``pt`` is mass-weighted, ``delp`` is updated, then ``pt`` is divided
    by the NEW ``delp`` -- all three at the same cell, reading only
    fluxes, so evaluating them in that order on whole slabs is exact.
    The NH ``w`` block runs BEFORE the ``delp`` update so its mass
    weighting uses the OLD ``delp``, exactly as upstream.
    """
    _require_bool("d_sw2_duo", "hydrostatic", hydrostatic)
    _require_bool("d_sw2_duo", "inline_q", inline_q)
    _validate_nord("d_sw2_duo", "nord_w", nord_w)
    if inline_q:
        raise NotImplementedError(
            "d_sw2_duo: inline_q=False lane only (matches the oracle "
            "driver; q_con/tracer updates not exercised)")
    if not hydrostatic and (w is None or npx is None or npy is None):
        raise ValueError(
            "d_sw2_duo: the NH arm needs w, npx and npy "
            "(sw_core.F90:1077-1109)")
    _require_f64_jax("d_sw2_duo", {
        "delp": delp, "pt": pt, "allflux_x": allflux_x,
        "allflux_y": allflux_y, "w": w, "dt": jnp.asarray(dt),
        "kgb": jnp.asarray(kgb)})
    keys = ("rarea",) if hydrostatic or damp_w <= _DAMP_W_ON else \
        ("rarea", "del6_v", "del6_u")
    g = _geom("d_sw2_duo", gs, keys)

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    nid, njd = ied - isd + 1, jed - jsd + 1
    nci, ncj = ie - is_ + 1, je - js + 1

    delp, pt = jnp.asarray(delp), jnp.asarray(pt)
    allflux_x = jnp.asarray(allflux_x)
    allflux_y = jnp.asarray(allflux_y)
    _check_shape("d_sw2_duo", "delp", delp, (nid, njd))
    _check_shape("d_sw2_duo", "pt", pt, (nid, njd))
    rarea = g["rarea"]

    # allflux compute rings -> Fortran origin (is, js), matching the
    # dummy bounds allflux_x(is:ie+1, js:je, k, slot)
    fx = allflux_x[:, :, 0]
    gx = allflux_x[:, :, 3]
    fy = allflux_y[:, :, 0]
    gy = allflux_y[:, :, 3]

    def cw(a, i0, i1, j0, j1):        # compute-ring origin (is, js)
        return _fw(a, is_, js, i0, i1, j0, j1)

    def rd(a, i0, i1, j0, j1):
        return _fw(a, isd, jsd, i0, i1, j0, j1)

    # heat_source zeroing (sw_core.F90:1074-1078, #ifndef SW_DYNAMICS)
    heat_source = jnp.zeros((nci, ncj), jnp.float64)
    dw = jnp.full((nci, ncj), workspace_sentinel, jnp.float64)

    ra = rd(rarea, is_, ie, js, je)
    w_out = None
    if not hydrostatic:
        wf = jnp.asarray(w)
        _check_shape("d_sw2_duo", "w", wf, (nid, njd))
        if damp_w > _DAMP_W_ON:
            dd8 = kgb * jnp.abs(dt)
            damp4 = (damp_w * flags.da_min_c) ** (nord_w + 1)
            fx2, fy2 = del6_vt_flux(
                nord_w, npx, npy, damp4, wf, bd, g["del6_u"],
                g["del6_v"], rarea, flags.bounded_domain,
                flags.sw_corner, flags.se_corner, flags.nw_corner,
                flags.ne_corner, True)
            dwv = (_fw(fx2, isd, jsd, is_, ie, js, je)
                   - _fw(fx2, isd, jsd, is_ + 1, ie + 1, js, je)
                   + _fw(fy2, isd, jsd, is_, ie, js, je)
                   - _fw(fy2, isd, jsd, is_, ie, js + 1, je + 1)) * ra
            dw = dwv
            # verbatim sw_core.F90:1087 (the commented-out -d_con form
            # sits directly above it): the dd8 term is included even
            # though ke_bg = 0 on the pinned deck.
            heat_source = dd8 - dwv * (rd(wf, is_, ie, js, je)
                                       + 0.5 * dwv)
        gx2 = allflux_x[:, :, 1]
        gy2 = allflux_y[:, :, 1]
        wf = _fs(wf, isd, jsd, is_, ie, js, je,
                 rd(delp, is_, ie, js, je) * rd(wf, is_, ie, js, je)
                 + (cw(gx2, is_, ie, js, je)
                    - cw(gx2, is_ + 1, ie + 1, js, je)
                    + cw(gy2, is_, ie, js, je)
                    - cw(gy2, is_, ie, js + 1, je + 1)) * ra)
        w_out = wf

    # else-arm update (sw_core.F90:1181-1196; inline_q=F)
    dp0 = rd(delp, is_, ie, js, je)
    ptm = rd(pt, is_, ie, js, je) * dp0 \
        + (cw(gx, is_, ie, js, je) - cw(gx, is_ + 1, ie + 1, js, je)
           + cw(gy, is_, ie, js, je)
           - cw(gy, is_, ie, js + 1, je + 1)) * ra
    dpn = dp0 + (cw(fx, is_, ie, js, je) - cw(fx, is_ + 1, ie + 1, js, je)
                 + cw(fy, is_, ie, js, je)
                 - cw(fy, is_, ie, js + 1, je + 1)) * ra
    delp = _fs(delp, isd, jsd, is_, ie, js, je, dpn)
    pt = _fs(pt, isd, jsd, is_, ie, js, je, ptm / dpn)

    ptc = jnp.full_like(delp, workspace_sentinel)
    return {"delp": delp, "pt": pt, "w": w_out,
            "heat_source": heat_source, "ptc": ptc, "dw": dw}


# =====================================================================
# d_sw3  (sw_core.F90:1201-1388) -- the KE-flux stage
# =====================================================================

_DSW3_KEYS = ("cosa", "rsina", "dx", "rdx", "dy", "rdy")


def d_sw3_duo(u, v, uc, vc, gs: dict, flags: GridFlags, bd, npx: int,
              npy: int, *, dt, hord_mt: int = 6, duogrid: bool = True):
    """JAX twin of ``fv3_native_duo_sw_core.d_sw3_duo``
    (sw_core.F90:1201-1388, DUO branch).

    The B-grid contravariant ``vb``/``ub`` run over the FULL unclamped
    ranges (``bounded .or. duogrid`` -> ``is2=is, ie1=ie+1, js2=js,
    je1=je+1``) with the INTERIOR formula everywhere; the vt/ut-based
    edge and corner extrapolations live in the SKIPPED non-duo else, so
    ``ut``/``vt`` are never read on this lane and no workspace sentinel
    is needed.  Returns the four fields dyn_core carries to ``d_sw5``'s
    KE assembly -- ``ubbtemp``/``vbbtemp`` (post-``ytp_v`` ``ub`` and
    pre-``xtp_u`` ``vb``) and ``ubb``/``vbb`` (final) -- which is where
    barrier 2 (``dyn_core.F90:984``, ``BGRID_NE``) acts.

    VERBATIM quirk preserved from the twin: ``d_sw3`` passes
    ``bounded_domain=.false.`` to ``ytp_v``/``xtp_u`` as a LITERAL
    (the commented-out originals passed the real flag); the duo
    behaviour inside them comes from the symmetryclean
    ``gridstruct%dg%is_initialized`` gates, carried here as
    ``duogrid=True``.

    Dependence (R1a): each of the four stages writes one array and reads
    only arrays an earlier stage completed (``vb`` <- metrics/uc/vc;
    ``ub`` <- ``ytp_v(vb)``; the ``ubbtemp``/``vbbtemp`` snapshot; then
    ``ub`` recomputed and ``vb`` <- ``xtp_u(ub)``).  The two snapshots
    are whole-array copies, which is why the recompute of ``ub`` cannot
    disturb them.  No data branch, hence no ``jnp.where``.
    """
    _validate_ord("d_sw3_duo", "hord_mt", hord_mt, _SW_ORDS)
    _require_bool("d_sw3_duo", "duogrid", duogrid)
    if not duogrid:
        raise NotImplementedError(
            "d_sw3_duo is the DUO-stage port; the plain path is the "
            "certified monolithic d_sw (phase-4b)")
    _require_f64_jax("d_sw3_duo", {"u": u, "v": v, "uc": uc, "vc": vc,
                                   "dt": jnp.asarray(dt)})
    g = _geom("d_sw3_duo", gs, _DSW3_KEYS)
    del flags   # d_sw3 reads no static gridstruct flag on this lane

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    nid, njd = ied - isd + 1, jed - jsd + 1

    u, v = jnp.asarray(u), jnp.asarray(v)
    uc, vc = jnp.asarray(uc), jnp.asarray(vc)
    _check_shape("d_sw3_duo", "u", u, (nid, njd + 1))
    _check_shape("d_sw3_duo", "v", v, (nid + 1, njd))
    _check_shape("d_sw3_duo", "uc", uc, (nid + 1, njd))
    _check_shape("d_sw3_duo", "vc", vc, (nid, njd + 1))

    cosa, rsina = g["cosa"], g["rsina"]
    dx, rdx, dy, rdy = g["dx"], g["rdx"], g["dy"], g["rdy"]

    def rd(a, i0, i1, j0, j1):
        return _fw(a, isd, jsd, i0, i1, j0, j1)

    dt5 = 0.5 * dt                                   # sw_core.F90:1257
    # duo ranges (sw_core.F90:1260-1263)
    is2, ie1 = is_, ie + 1
    js2, je1 = js, je + 1

    vb = _new(is_, ie + 1, js, je + 1)

    # vb: duo interior formula everywhere (sw_core.F90:1271-1275)
    vb = _fs(vb, is_, js, is2, ie1, js2, je1,
             dt5 * (rd(vc, is2 - 1, ie1 - 1, js2, je1)
                    + rd(vc, is2, ie1, js2, je1)
                    - (rd(uc, is2, ie1, js2 - 1, je1 - 1)
                       + rd(uc, is2, ie1, js2, je1))
                    * rd(cosa, is2, ie1, js2, je1))
             * rd(rsina, is2, ie1, js2, je1))

    # sw_core.F90:1318 -- bounded_domain literal .false.; duo via the dg
    # gates, carried as duogrid=True.
    ub = ytp_v(is_, ie, js, je, isd, ied, jsd, jed, vb, u, v, hord_mt,
               dy, rdy, npx, npy, 0, False, 1.0, duogrid=True)

    ubbtemp = ub
    vbbtemp = vb

    # ub: duo interior formula everywhere (sw_core.F90:1330-1338)
    ub = _fs(ub, is_, js, is2, ie1, js, je + 1,
             dt5 * (rd(uc, is2, ie1, js - 1, je)
                    + rd(uc, is2, ie1, js, je + 1)
                    - (rd(vc, is2 - 1, ie1 - 1, js, je + 1)
                       + rd(vc, is2, ie1, js, je + 1))
                    * rd(cosa, is2, ie1, js, je + 1))
             * rd(rsina, is2, ie1, js, je + 1))

    # sw_core.F90:1378 -- bounded_domain literal .false.
    vb = xtp_u(is_, ie, js, je, isd, ied, jsd, jed, ub, u, v, hord_mt,
               dx, rdx, npx, npy, 0, False, 1.0, duogrid=True)

    return {"ubbtemp": ubbtemp, "vbbtemp": vbbtemp, "ubb": ub,
            "vbb": vb}


# =====================================================================
# d_sw4  (sw_core.F90:1390-1472) -- the 4-corner KE fix
# =====================================================================

def d_sw4_duo(u, v, ut, vt, ke, flags: GridFlags, bd, npx: int,
              npy: int, *, dt):
    """JAX twin of ``fv3_native_duo_sw_core.d_sw4_duo``
    (sw_core.F90:1390-1472).

    Its guard ``.not.bounded .or. .not.duogrid`` (sw_core.F90:1440)
    fires on the PLAIN-conventions lane (bounded=F) and is SKIPPED in the
    bounded duo runs; the corner flags gate each block exactly as
    upstream (they are FALSE under bounded anyway -- the explicit
    ``not bounded`` conjunction makes a raw gridstruct with default-true
    flags behave like source).

    ``ke`` is INTENT(INOUT) upstream (dyn_core hands it the inline KE
    assembly ``0.5*(ubbtemp*vbbtemp + ubb*vbb)``); the stage writes ONLY
    the four corner B-nodes, and everything else round-trips.  Returns
    ``dict(ke)``.

    On the plain lane the corner formulas read ``u``/``v`` and the
    ``d_sw1`` ``ut``/``vt`` workspace at cells the duo interior DOES
    write (``ut`` over is:ie+1 x jsd:jed and ``vt`` over isd:ied x
    js:je+1 cover every corner read), so no input is sentinel-dependent
    here.  Four single-point writes; no loop, no branch on data, no
    ``jnp.where``.

    This routine reads NO gridstruct ARRAY -- only the static flags --
    so it takes ``flags`` and no ``gs``.  That is the dict->NamedTuple
    adaptation, not a dropped operand.
    """
    _require_f64_jax("d_sw4_duo", {"u": u, "v": v, "ut": ut, "vt": vt,
                                   "ke": ke, "dt": jnp.asarray(dt)})
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    nid, njd = ied - isd + 1, jed - jsd + 1
    u, v = jnp.asarray(u), jnp.asarray(v)
    ut, vt, ke = jnp.asarray(ut), jnp.asarray(vt), jnp.asarray(ke)
    _check_shape("d_sw4_duo", "u", u, (nid, njd + 1))
    _check_shape("d_sw4_duo", "v", v, (nid + 1, njd))
    _check_shape("d_sw4_duo", "ut", ut, (nid + 1, njd))
    _check_shape("d_sw4_duo", "vt", vt, (nid, njd + 1))
    _check_shape("d_sw4_duo", "ke", ke, (nid + 1, njd + 1))

    def pnt(a, i, j):
        return _fw(a, isd, jsd, i, i, j, j)

    def set_ke(i, j, val):
        return _fs(ke, isd, jsd, i, i, j, j, val)

    bounded = flags.bounded_domain
    sw_c = (not bounded) and flags.sw_corner
    se_c = (not bounded) and flags.se_corner
    ne_c = (not bounded) and flags.ne_corner
    nw_c = (not bounded) and flags.nw_corner

    dt6 = dt / 6.0                                   # sw_core.F90:1442
    if sw_c:
        ke = set_ke(1, 1,
                   dt6 * ((pnt(ut, 1, 1) + pnt(ut, 1, 0)) * pnt(u, 1, 1)
                          + (pnt(vt, 1, 1) + pnt(vt, 0, 1)) * pnt(v, 1, 1)
                          + (pnt(ut, 1, 1) + pnt(vt, 1, 1)) * pnt(u, 0, 1)))
    if se_c:
        i = npx
        ke = set_ke(i, 1,
                   dt6 * ((pnt(ut, i, 1) + pnt(ut, i, 0)) * pnt(u, i - 1, 1)
                          + (pnt(vt, i, 1) + pnt(vt, i - 1, 1)) * pnt(v, i, 1)
                          + (pnt(ut, i, 1) - pnt(vt, i - 1, 1))
                          * pnt(u, i, 1)))
    if ne_c:
        i, j = npx, npy
        ke = set_ke(i, j,
                   dt6 * ((pnt(ut, i, j) + pnt(ut, i, j - 1))
                          * pnt(u, i - 1, j)
                          + (pnt(vt, i, j) + pnt(vt, i - 1, j))
                          * pnt(v, i, j - 1)
                          + (pnt(ut, i, j - 1) + pnt(vt, i - 1, j))
                          * pnt(u, i, j)))
    if nw_c:
        j = npy
        ke = set_ke(1, j,
                   dt6 * ((pnt(ut, 1, j) + pnt(ut, 1, j - 1)) * pnt(u, 1, j)
                          + (pnt(vt, 1, j) + pnt(vt, 0, j))
                          * pnt(v, 1, j - 1)
                          + (pnt(ut, 1, j - 1) - pnt(vt, 1, j))
                          * pnt(u, 0, j)))
    return {"ke": ke}


# =====================================================================
# d_sw5  (sw_core.F90:1474-1869) -- vorticity, divergence damping,
#        Smagorinsky vort, and the vorticity-flux transport
# =====================================================================

_DSW5_KEYS = ("rarea", "rarea_c", "divg_u", "divg_v", "dx", "dy", "f0",
              "dxa", "dya", "area", "del6_v", "del6_u", "grid_lon",
              "grid_lat", "agrid_lon", "agrid_lat", "edge_w", "edge_e",
              "edge_s", "edge_n")
_A2B_KEYS = ("grid_lon", "grid_lat", "agrid_lon", "agrid_lat", "dxa",
             "dya", "edge_w", "edge_e", "edge_s", "edge_n")


def d_sw5_duo(delp, u, v, uc, vc, ua, va, divg_d, crx_adv, cry_adv,
              xfx_adv, yfx_adv, ra_x, ra_y, ke, gs: dict,
              flags: GridFlags, bd, npx: int, npy: int, *, dt,
              hord_vt: int = 6, nord: int = 1, dddmp: float = 0.2,
              d2_bg: float = 0.0, d4_bg: float = 0.12,
              d_con: float = 0.0, hydrostatic: bool = True,
              lim_fac: float = 1.0, w=None, dw=None,
              damp_w: float = 0.0, do_f3d: bool = False,
              workspace_sentinel: float = 1.0e30):
    """JAX twin of ``fv3_native_duo_sw_core.d_sw5_duo``
    (sw_core.F90:1474-1869, DUO branch, oracle lane).

    Stages, in source order: the vorticity prep (``vt = u*dx``,
    ``ut = v*dy`` over the FULL data domain -- ``d_sw5`` OVERWRITES the
    ``d_sw1`` ut/vt workspace, so this twin builds FRESH arrays and takes
    no ut/vt input), ``wk`` = volume-mean relative vorticity, the
    optional NH ``w`` finalisation, the ``nord`` higher-order
    divergence-damping loop (``delpc`` = the saved ``divg_d``; ``uc``/
    ``vc`` are CLOBBERED as gradient workspaces -- oracle semantics,
    returned as outputs; duo SKIPS the corner-term removal and
    ``fill_c`` is false), ``a2b_ord4`` + Smagorinsky ``vort``, the
    ``ke`` damping increment over the B compute ring, and the
    vorticity-flux transport ``fv_tp_2d(wk + f0)``.

    ``crx``/``cry``/``xfx``/``yfx``/``ra_x``/``ra_y`` are the ``d_sw1``
    outputs (upstream declares the first four ``intent(OUT)`` yet only
    reads them; the extract shims them to inout).  ``ptc`` is unwritten
    on the ``nord>0`` branch and ``ub``/``vb`` are untouched at
    ``d_con=0`` -- all three are returned as ``workspace_sentinel``
    fills, mirroring the driver.

    **Dependence (R1a) -- the pass loop is a RECURRENCE.**
    ``do n=1,nord`` (sw_core.F90:1738-1788) reads the ``divg_d`` the
    previous pass wrote, so it stays an ORDERED python ``for`` over a
    STATIC trip count: unrolled, never vectorised, never a ``scan`` (the
    window ``nt = nord - n`` shrinks each pass, so the bodies are not
    the same shape).  INSIDE a pass, ``vc`` and ``uc`` each read only
    the OLD ``divg_d``, then ``divg_d`` reads the NEW ``uc``/``vc``, then
    the ``rarea_c`` scaling is elementwise on ``divg_d`` -- so each of
    the four nests vectorises, and none of ``uc``/``vc``/``divg_d`` is
    reallocated, so cells outside a window carry forward exactly as
    in place.

    R1b: no data-dependent branch reaches a ``jnp.where`` here.  The
    only non-smooth operators are the oracle's own and they are on the
    LIVE path: ``sqrt(delpc**2 + vort**2)`` (sw_core.F90:1798, not
    differentiable where both are zero) and
    ``max(d2_bg, min(0.20, dddmp*vort))`` (:1816, kinked at both clamp
    ties).  ``dddmp < 1e-5`` and ``d_con > 1e-5`` are tests on STATIC
    deck constants, so they are python ``if``s.
    """
    _validate_ord("d_sw5_duo", "hord_vt", hord_vt, _PPM_ORDS)
    _validate_nord("d_sw5_duo", "nord", nord)
    _validate_grid_type("d_sw5_duo", flags.grid_type)
    _require_bool("d_sw5_duo", "hydrostatic", hydrostatic)
    _require_bool("d_sw5_duo", "do_f3d", do_f3d)
    if d_con > _D_CON_ON or nord not in (1, 2):
        raise NotImplementedError(
            "d_sw5_duo: oracle lane only (d_con=0, nord in {1, 2}; "
            "nord <= ng-1)")
    if not hydrostatic and do_f3d:
        raise NotImplementedError(
            "d_sw5_duo: do_f3d needs the ROT3 build define, which the "
            "pinned oracle build does not set (sw_core.F90:1601-1611)")
    if not hydrostatic and w is None:
        raise ValueError(
            "d_sw5_duo: the NH arm needs w (sw_core.F90:1600-1627)")
    if not hydrostatic and damp_w > _DAMP_W_ON and dw is None:
        raise ValueError(
            "d_sw5_duo: damp_w > 1e-5 needs the d_sw2 dw increment")
    _require_f64_jax("d_sw5_duo", {
        "delp": delp, "u": u, "v": v, "uc": uc, "vc": vc, "ua": ua,
        "va": va, "divg_d": divg_d, "crx_adv": crx_adv,
        "cry_adv": cry_adv, "xfx_adv": xfx_adv, "yfx_adv": yfx_adv,
        "ra_x": ra_x, "ra_y": ra_y, "ke": ke, "w": w, "dw": dw,
        "dt": jnp.asarray(dt)})
    g = _geom("d_sw5_duo", gs, _DSW5_KEYS)

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    ng = bd.ng
    nid, njd = ied - isd + 1, jed - jsd + 1
    nci, ncj = ie - is_ + 1, je - js + 1

    u, v = jnp.asarray(u), jnp.asarray(v)
    uc, vc = jnp.asarray(uc), jnp.asarray(vc)
    divg_d = jnp.asarray(divg_d)
    ke = jnp.asarray(ke)
    crx, cry = jnp.asarray(crx_adv), jnp.asarray(cry_adv)
    xfx, yfx = jnp.asarray(xfx_adv), jnp.asarray(yfx_adv)
    ra_x, ra_y = jnp.asarray(ra_x), jnp.asarray(ra_y)
    _check_shape("d_sw5_duo", "u", u, (nid, njd + 1))
    _check_shape("d_sw5_duo", "v", v, (nid + 1, njd))
    _check_shape("d_sw5_duo", "uc", uc, (nid + 1, njd))
    _check_shape("d_sw5_duo", "vc", vc, (nid, njd + 1))
    _check_shape("d_sw5_duo", "divg_d", divg_d, (nid + 1, njd + 1))
    _check_shape("d_sw5_duo", "ke", ke, (nid + 1, njd + 1))

    rarea, rarea_c = g["rarea"], g["rarea_c"]
    divg_u, divg_v = g["divg_u"], g["divg_v"]
    dx, dy, f0 = g["dx"], g["dy"], g["f0"]
    da_min_c = flags.da_min_c

    def rd(a, i0, i1, j0, j1):
        return _fw(a, isd, jsd, i0, i1, j0, j1)

    # ---- vorticity prep (sw_core.F90:1582-1598) --------------------
    ut = _new(isd, ied + 1, jsd, jed)
    vt = _new(isd, ied, jsd, jed + 1)
    vt = _fs(vt, isd, jsd, isd, ied, jsd, jed + 1,
             rd(u, isd, ied, jsd, jed + 1) * rd(dx, isd, ied, jsd, jed + 1))
    ut = _fs(ut, isd, jsd, isd, ied + 1, jsd, jed,
             rd(v, isd, ied + 1, jsd, jed) * rd(dy, isd, ied + 1, jsd, jed))
    wk = _new(isd, ied, jsd, jed)
    wk = _fs(wk, isd, jsd, isd, ied, jsd, jed,
             rd(rarea, isd, ied, jsd, jed)
             * (_fw(vt, isd, jsd, isd, ied, jsd, jed)
                - _fw(vt, isd, jsd, isd, ied, jsd + 1, jed + 1)
                - _fw(ut, isd, jsd, isd, ied, jsd, jed)
                + _fw(ut, isd, jsd, isd + 1, ied + 1, jsd, jed)))

    # ---- NH w finalisation (sw_core.F90:1600-1627) -----------------
    # d_sw2 left w MASS-WEIGHTED (delp*w + fluxdiv); divide by the
    # UPDATED delp, then add the d_sw2 del6 damping increment.
    w_out = None
    if not hydrostatic:
        delpf = jnp.asarray(delp)
        _check_shape("d_sw5_duo", "delp", delpf, (nid, njd))
        wf = jnp.asarray(w)
        _check_shape("d_sw5_duo", "w", wf, (nid, njd))
        wf = _fs(wf, isd, jsd, is_, ie, js, je,
                 rd(wf, is_, ie, js, je) / rd(delpf, is_, ie, js, je))
        if damp_w > _DAMP_W_ON:
            dwf = jnp.asarray(dw)
            _check_shape("d_sw5_duo", "dw", dwf, (nci, ncj))
            wf = _fs(wf, isd, jsd, is_, ie, js, je,
                     rd(wf, is_, ie, js, je)
                     + _fw(dwf, is_, js, is_, ie, js, je))
        w_out = wf

    # ---- higher-order divergence damping (sw_core.F90:1731-1824) ---
    delpc = _new(isd, ied, jsd, jed, workspace_sentinel)
    delpc = _fs(delpc, isd, jsd, is_, ie + 1, js, je + 1,
                _fw(divg_d, isd, jsd, is_, ie + 1, js, je + 1))

    # ORDERED recurrence (see the docstring): nt = nord - n shrinks.
    # fill_c is FALSE on the duo lane (guard `.not.(bounded .or.
    # duogrid)`), the corner-term removal is duo-SKIPPED (:1771), and
    # the rarea_c scaling runs unconditionally (not stretched).
    for n_it in range(1, nord + 1):
        nt = nord - n_it
        vc = _fs(vc, isd, jsd, is_ - 1 - nt, ie + 1 + nt,
                 js - nt, je + 1 + nt,
                 (_fw(divg_d, isd, jsd, is_ - nt, ie + 2 + nt,
                      js - nt, je + 1 + nt)
                  - _fw(divg_d, isd, jsd, is_ - 1 - nt, ie + 1 + nt,
                        js - nt, je + 1 + nt))
                 * rd(divg_u, is_ - 1 - nt, ie + 1 + nt,
                     js - nt, je + 1 + nt))
        uc = _fs(uc, isd, jsd, is_ - nt, ie + 1 + nt,
                 js - 1 - nt, je + 1 + nt,
                 (_fw(divg_d, isd, jsd, is_ - nt, ie + 1 + nt,
                      js - nt, je + 2 + nt)
                  - _fw(divg_d, isd, jsd, is_ - nt, ie + 1 + nt,
                        js - 1 - nt, je + 1 + nt))
                 * rd(divg_v, is_ - nt, ie + 1 + nt,
                     js - 1 - nt, je + 1 + nt))
        divg_d = _fs(divg_d, isd, jsd, is_ - nt, ie + 1 + nt,
                     js - nt, je + 1 + nt,
                     _fw(uc, isd, jsd, is_ - nt, ie + 1 + nt,
                         js - 1 - nt, je + nt)
                     - _fw(uc, isd, jsd, is_ - nt, ie + 1 + nt,
                           js - nt, je + 1 + nt)
                     + _fw(vc, isd, jsd, is_ - 1 - nt, ie + nt,
                           js - nt, je + 1 + nt)
                     - _fw(vc, isd, jsd, is_ - nt, ie + 1 + nt,
                           js - nt, je + 1 + nt))
        divg_d = _fs(divg_d, isd, jsd, is_ - nt, ie + 1 + nt,
                     js - nt, je + 1 + nt,
                     _fw(divg_d, isd, jsd, is_ - nt, ie + 1 + nt,
                         js - nt, je + 1 + nt)
                     * rd(rarea_c, is_ - nt, ie + 1 + nt,
                         js - nt, je + 1 + nt))

    # ---- Smagorinsky vort (sw_core.F90:1790-1806) ------------------
    vort = _new(isd, ied, jsd, jed)
    if dddmp < _DDDMP_OFF:
        vort = jnp.zeros_like(vort)
    else:
        wk, vort = a2b_ord4(
            wk, vort, {k: g[k] for k in _A2B_KEYS}, npx, npy, is_, ie,
            js, je, ng, False, True,
            bounded_domain=flags.bounded_domain, grid_type=0,
            sw_corner=flags.sw_corner, se_corner=flags.se_corner,
            ne_corner=flags.ne_corner, nw_corner=flags.nw_corner)
        vort = _fs(vort, isd, jsd, is_, ie + 1, js, je + 1,
                   jnp.abs(dt) * jnp.sqrt(
                       _fw(delpc, isd, jsd, is_, ie + 1, js, je + 1) ** 2
                       + _fw(vort, isd, jsd, is_, ie + 1,
                             js, je + 1) ** 2))

    dd8 = (da_min_c * d4_bg) ** (nord + 1)         # sw_core.F90:1811
    vw = _fw(vort, isd, jsd, is_, ie + 1, js, je + 1)
    damp2 = da_min_c * jnp.maximum(                # sw_core.F90:1816
        d2_bg, jnp.minimum(_DDDMP_CAP, dddmp * vw))
    vw = (damp2 * _fw(delpc, isd, jsd, is_, ie + 1, js, je + 1)
          + dd8 * _fw(divg_d, isd, jsd, is_, ie + 1, js, je + 1))
    vort = _fs(vort, isd, jsd, is_, ie + 1, js, je + 1, vw)
    ke = _fs(ke, isd, jsd, is_, ie + 1, js, je + 1,
             _fw(ke, isd, jsd, is_, ie + 1, js, je + 1) + vw)

    # d_con = 0: the ub/vb dissipation strips are untouched.

    # ---- vorticity transport (sw_core.F90:1838-1862) ---------------
    vort = _fs(vort, isd, jsd, isd, ied, jsd, jed,
               _fw(wk, isd, jsd, isd, ied, jsd, jed)
               + rd(f0, isd, ied, jsd, jed))
    _vort, vortfluxx, vortfluxy = fv_tp_2d(
        vort, crx, cry, npx, npy, hord_vt, xfx, yfx, g["dxa"], g["dya"],
        g["area"], g["del6_v"], g["del6_u"], rarea, flags.da_min, bd,
        ra_x, ra_y, lim_fac, flags.bounded_domain, 0, flags.sw_corner,
        flags.se_corner, flags.nw_corner, flags.ne_corner, duogrid=True)
    del _vort   # copy_corners is a no-op on the duo lane; not returned

    ptc = jnp.full((nid, njd), workspace_sentinel, jnp.float64)
    ub = jnp.full((nci + 1, ncj + 1), workspace_sentinel, jnp.float64)
    vb = jnp.full((nci + 1, ncj + 1), workspace_sentinel, jnp.float64)
    return {"delpc": delpc, "divg_d": divg_d, "wk": wk, "ke": ke,
            "vortfluxx": vortfluxx, "vortfluxy": vortfluxy,
            "uc": uc, "vc": vc, "ut": ut, "vt": vt,
            "ptc": ptc, "ub": ub, "vb": vb, "w": w_out}


# =====================================================================
# d_sw6  (sw_core.F90:1872-2006) -- the final wind update
# =====================================================================

_DSW6_KEYS = ("del6_v", "del6_u", "rarea")


def d_sw6_duo(u, v, ut, vt, ke, wk, vortfluxx, vortfluxy, gs: dict,
              flags: GridFlags, bd, npx: int, npy: int, *,
              nord_v: int = 1, damp_v: float = 0.2, d_con: float = 0.0,
              duogrid: bool = True,
              workspace_sentinel: float = 1.0e30):
    """JAX twin of ``fv3_native_duo_sw_core.d_sw6_duo``
    (sw_core.F90:1872-2006, oracle lane: ``damp_v = 0.2``,
    ``d_con = 0``).

    The circulation-form wind update (sw_core.F90:1934-1944)::

        u = vt + ke - ke(i+1,j) + fy      (fy = vortfluxy)
        v = ut + ke - ke(i,j+1) - fx      (fx = vortfluxx)

    then the ``damp_v`` del-6 vorticity damping -- ``del6_vt_flux``
    CLOBBERS ``ut``/``vt`` as its flux outputs, which is why the
    incoming arrays are handed to it as the ``fx2``/``fy2`` seeds so
    every cell outside its write windows carries forward exactly as the
    in-place lane leaves it -- and the diffusive-flux add
    ``u += vt``, ``v -= ut`` (:1989-2000).

    ``d_con = 0`` skips the heating block (:1953-1986), so ``ub``/``vb``
    and ``heat_source`` are untouched sentinel round-trips (``d_sw2``,
    which zeroes ``heat_source`` in the real pipeline, is not part of
    this chain).  Returns ``dict(u, v, ut, vt, ub, vb, heat_source)``.

    Dependence (R1a): the two wind nests write ``u``/``v`` and read
    ``ut``/``vt``/``ke``/the vorticity fluxes -- none of which they
    write -- so both vectorise; the ``del6_vt_flux`` call then replaces
    ``ut``/``vt`` wholesale before the diffusive add reads them, which
    is why the add must come AFTER it.  No data-dependent branch, hence
    no ``jnp.where``; ``damp_v > 1e-5`` and ``d_con > 1e-5`` are tests
    on STATIC deck constants.
    """
    _validate_nord("d_sw6_duo", "nord_v", nord_v)
    _require_bool("d_sw6_duo", "duogrid", duogrid)
    if d_con > _D_CON_ON:
        raise NotImplementedError("d_sw6_duo: d_con=0 oracle lane only")
    if not duogrid:
        raise NotImplementedError(
            "d_sw6_duo is the DUO-stage port; the plain path is the "
            "certified monolithic d_sw (phase-4b)")
    _require_f64_jax("d_sw6_duo", {
        "u": u, "v": v, "ut": ut, "vt": vt, "ke": ke, "wk": wk,
        "vortfluxx": vortfluxx, "vortfluxy": vortfluxy})
    g = _geom("d_sw6_duo", gs, _DSW6_KEYS)

    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    nid, njd = ied - isd + 1, jed - jsd + 1
    nci, ncj = ie - is_ + 1, je - js + 1

    u, v = jnp.asarray(u), jnp.asarray(v)
    ut, vt = jnp.asarray(ut), jnp.asarray(vt)
    ke, wk = jnp.asarray(ke), jnp.asarray(wk)
    fx = jnp.asarray(vortfluxx)
    fy = jnp.asarray(vortfluxy)
    _check_shape("d_sw6_duo", "u", u, (nid, njd + 1))
    _check_shape("d_sw6_duo", "v", v, (nid + 1, njd))
    _check_shape("d_sw6_duo", "ut", ut, (nid + 1, njd))
    _check_shape("d_sw6_duo", "vt", vt, (nid, njd + 1))
    _check_shape("d_sw6_duo", "ke", ke, (nid + 1, njd + 1))
    _check_shape("d_sw6_duo", "wk", wk, (nid, njd))
    _check_shape("d_sw6_duo", "vortfluxx", fx, (nci + 1, ncj))
    _check_shape("d_sw6_duo", "vortfluxy", fy, (nci, ncj + 1))

    def rd(a, i0, i1, j0, j1):
        return _fw(a, isd, jsd, i0, i1, j0, j1)

    def cw(a, i0, i1, j0, j1):
        return _fw(a, is_, js, i0, i1, j0, j1)

    # ---- final circulation-form wind update (:1934-1944) -----------
    u = _fs(u, isd, jsd, is_, ie, js, je + 1,
            rd(vt, is_, ie, js, je + 1) + rd(ke, is_, ie, js, je + 1)
            - rd(ke, is_ + 1, ie + 1, js, je + 1)
            + cw(fy, is_, ie, js, je + 1))
    v = _fs(v, isd, jsd, is_, ie + 1, js, je,
            rd(ut, is_, ie + 1, js, je) + rd(ke, is_, ie + 1, js, je)
            - rd(ke, is_, ie + 1, js + 1, je + 1)
            - cw(fx, is_, ie + 1, js, je))

    # ---- damp_v del6 vorticity damping (:1948-1951) ----------------
    if damp_v > _DAMP_V_ON:
        damp4 = (damp_v * flags.da_min_c) ** (nord_v + 1)
        ut, vt = del6_vt_flux(
            nord_v, npx, npy, damp4, wk, bd, g["del6_u"], g["del6_v"],
            g["rarea"], flags.bounded_domain, flags.sw_corner,
            flags.se_corner, flags.nw_corner, flags.ne_corner, True,
            fx2=ut, fy2=vt)

        # ---- add the diffusive fluxes (:1989-2000) -----------------
        u = _fs(u, isd, jsd, is_, ie, js, je + 1,
                rd(u, is_, ie, js, je + 1) + rd(vt, is_, ie, js, je + 1))
        v = _fs(v, isd, jsd, is_, ie + 1, js, je,
                rd(v, is_, ie + 1, js, je) - rd(ut, is_, ie + 1, js, je))

    ub = jnp.full((nci + 1, ncj + 1), workspace_sentinel, jnp.float64)
    vb = jnp.full((nci + 1, ncj + 1), workspace_sentinel, jnp.float64)
    heat_source = jnp.full((nci, ncj), workspace_sentinel, jnp.float64)
    return {"u": u, "v": v, "ut": ut, "vt": vt, "ub": ub, "vb": vb,
            "heat_source": heat_source}


# =====================================================================
# jit factories
#
# Policy, uniform across the module: index bounds (``bd``, ``npx``,
# ``npy``), every scheme selector, every python-branched flag and the
# ``GridFlags`` tuple are STATIC; the field arrays, the ``gs`` dict of
# arrays and the time step stay DYNAMIC.  ``bd`` and ``GridFlags`` are
# NamedTuples, hashable BY VALUE, so a fresh-but-equal instance shares
# one cache entry.  ``dt``/``dt2``/``kgb`` are dynamic on purpose: they
# only multiply arrays or feed a ``> 0`` test on DATA, never a python
# branch, so a new time step must NOT retrace.  No ``donate_argnums``
# anywhere (it conflicts with reverse-mode AD, which is the point of
# this lane).
# =====================================================================

def make_edge_interpolate4_jit(fn=edge_interpolate4):
    """Both operands are 4-element pytrees of arrays -- all dynamic."""
    return jax.jit(fn)


def make_fill_4corners_jit(fn=fill_4corners):
    """Static: direction/npx/npy/bd and the four corner flags."""
    return jax.jit(fn, static_argnums=(1, 2, 3, 4),
                   static_argnames=("sw", "se", "ne", "nw"))


def make_fill2_4corners_jit(fn=fill2_4corners):
    """Same policy as :func:`make_fill_4corners_jit`."""
    return jax.jit(fn, static_argnums=(2, 3, 4, 5),
                   static_argnames=("sw", "se", "ne", "nw"))


def make_d2a2c_vect_jit(fn=d2a2c_vect):
    """Static: bd/npx/npy + dord4/grid_type/bounded_domain."""
    return jax.jit(fn, static_argnums=(3, 4, 5),
                   static_argnames=("dord4", "grid_type",
                                    "bounded_domain"))


def make_d2a2c_vect_duo_jit(fn=d2a2c_vect_duo):
    """Static: bd/npx/npy + dord4/grid_type."""
    return jax.jit(fn, static_argnums=(3, 4, 5),
                   static_argnames=("dord4", "grid_type"))


def make_divergence_corner_jit(fn=divergence_corner):
    """Static: bd/npx/npy + grid_type."""
    return jax.jit(fn, static_argnums=(5, 6, 7),
                   static_argnames=("grid_type",))


def make_divergence_corner_duo_jit(fn=divergence_corner_duo):
    """Same policy as :func:`make_divergence_corner_jit`."""
    return jax.jit(fn, static_argnums=(5, 6, 7),
                   static_argnames=("grid_type",))


def make_del6_vt_flux_jit(fn=del6_vt_flux):
    """Static: nord (a trip count AND every index window), npx/npy, bd
    and the six domain/corner/duo flags.  ``damp`` stays dynamic -- it
    only multiplies ``q`` and is never compared."""
    return jax.jit(fn, static_argnums=(0, 1, 2, 5, 9, 10, 11, 12, 13,
                                       14))


def make_c_sw_jit(fn=c_sw):
    """Static: bd/npx/npy + nord/hydrostatic/dord4/grid_type/duogrid/
    bounded_domain.  ``dt2`` is DYNAMIC (it only scales arrays)."""
    return jax.jit(fn, static_argnums=(6, 7, 8),
                   static_argnames=("nord", "hydrostatic", "dord4",
                                    "grid_type", "duogrid",
                                    "bounded_domain"))


def make_d_sw1_duo_jit(fn=d_sw1_duo):
    """Static: flags/bd/npx/npy + every scheme selector and knob that a
    python ``if`` reads (``damp_v``/``damp_t`` feed ``damp_c > 1e-4``
    inside ``fv_tp_2d``; ``lim_fac``/``workspace_sentinel`` are
    trace-time constants).  ``dt`` stays DYNAMIC."""
    return jax.jit(
        fn, static_argnums=(10, 11, 12, 13),
        static_argnames=("hord_tr", "hord_vt", "hord_tm", "hord_dp",
                         "nord_v", "nord_t", "damp_v", "damp_t",
                         "hydrostatic", "inline_q", "lim_fac",
                         "duogrid", "workspace_sentinel"))


def make_d_sw2_duo_jit(fn=d_sw2_duo):
    """Static: flags/bd + npx/npy/nord_w/damp_w/hydrostatic/inline_q/
    workspace_sentinel.  ``w``/``dt``/``kgb`` stay dynamic (passing
    ``w=None`` changes the pytree and correctly retraces)."""
    return jax.jit(
        fn, static_argnums=(5, 6),
        static_argnames=("npx", "npy", "nord_w", "damp_w",
                         "hydrostatic", "inline_q",
                         "workspace_sentinel"))


def make_d_sw3_duo_jit(fn=d_sw3_duo):
    """Static: flags/bd/npx/npy + hord_mt/duogrid."""
    return jax.jit(fn, static_argnums=(5, 6, 7, 8),
                   static_argnames=("hord_mt", "duogrid"))


def make_d_sw4_duo_jit(fn=d_sw4_duo):
    """Static: flags/bd/npx/npy.  ``dt`` stays dynamic."""
    return jax.jit(fn, static_argnums=(5, 6, 7, 8))


def make_d_sw5_duo_jit(fn=d_sw5_duo):
    """Static: flags/bd/npx/npy + every deck knob a python ``if`` or a
    python power reads (``dddmp``, ``d_con``, ``damp_w``, ``d2_bg``,
    ``d4_bg``, ``nord``, ``hord_vt``, ``lim_fac``, ``do_f3d``,
    ``hydrostatic``, ``workspace_sentinel``).  ``dt``/``w``/``dw`` stay
    dynamic."""
    return jax.jit(
        fn, static_argnums=(16, 17, 18, 19),
        static_argnames=("hord_vt", "nord", "dddmp", "d2_bg", "d4_bg",
                         "d_con", "hydrostatic", "lim_fac", "damp_w",
                         "do_f3d", "workspace_sentinel"))


def make_d_sw6_duo_jit(fn=d_sw6_duo):
    """Static: flags/bd/npx/npy + nord_v/damp_v/d_con/duogrid/
    workspace_sentinel."""
    return jax.jit(
        fn, static_argnums=(9, 10, 11, 12),
        static_argnames=("nord_v", "damp_v", "d_con", "duogrid",
                         "workspace_sentinel"))






