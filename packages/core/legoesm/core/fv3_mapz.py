"""FV3 Lagrangian-to-Eulerian vertical remap (``fv_mapz``) -- JAX lane.

Functional, jit-compatible, reverse-differentiable mirror of the
certified NumPy fp64 lane (``fv3_native_mapz.py``, itself a loop-faithful
port of the pinned oracle
``/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-
symmetryclean/model/fv_mapz.F90``).  Oracle spans re-verified against
that tree while writing this module (``grep -n '^ *subroutine'``):
``Lagrangian_to_Eulerian`` :62-1080, ``map_scalar`` :1361-1453,
``map1_ppm`` :1456-1547, ``map1_q2`` :1666-1756, ``scalar_profile``
:1855-2262, ``cs_profile`` :2265-2690, ``cs_limiters`` :2693-2768, the
rezone :1412-1451, the ``kord > 7`` dispatch :1406/:1500/:1708, and the
module parameters :43-45/:51-52.

**The NumPy lane is the specification.**  This module is certified
AGAINST it, never against the Fortran directly (one authority per hop).
Every constant is IMPORTED from the NumPy twin rather than retyped.

WHAT IS PORTED, AND WHAT IS NOT
-------------------------------
The same scope as the NumPy lane, and no more.  ``ppm_profile`` /
``ppm_limiters`` / ``steepz`` (``kord <= 7``), ``mapn_tracer``
(``nq > 5``), ``remap_z`` / ``rst_remap`` / ``mappm``,
``compute_total_energy`` / ``pkez`` are NOT ported here either, and every
guard that refuses them raises on the SAME condition with the same
exception type as the NumPy lane: :func:`ppm_profile_is_unported` is
IMPORTED from it (not re-derived), and ``_refuse_iv_m3`` /
``_refuse_unported_lane`` are re-stated locally only because the NumPy
lane keeps them private and this repo forbids cross-module private
imports.

TRANSLATION RULES APPLIED (and where each one bit)
--------------------------------------------------
* **k-RECURRENCES ARE ``lax.scan``** -- never ``associative_scan``,
  never ``cumsum``.  The recurrences here are the two tridiagonal edge
  solves (:func:`_edge_solve_iv_m2`, :func:`_edge_solve_default`), the
  rezone's ``k0`` search hint, the ``w_limiter``'s two spill passes, and
  the omega interval search.
* **A k-LOOP IS VECTORISED ONLY AFTER PROVING NO ITERATION READS WHAT AN
  EARLIER ITERATION WROTE.**  Proved per loop in the docstring of each
  helper.  The loops that FAIL that test -- the rezone target loop
  (carries ``k0``), the ``w_limiter`` passes (each level reads the
  level the previous iteration just clipped), the omega loop (carries
  ``k_next``), both tridiagonals -- stay sequential ``lax.scan``.
* **``jnp.where`` IS A SELECT: BOTH ARMS ARE EVALUATED.**  It is used
  only where both arms are TOTAL and FINITE on every admitted input, so
  the untaken arm can never manufacture a NaN/Inf that contaminates the
  reverse-mode gradient of the taken one.  Every division that could be
  partial is either (a) sanitised before the select with a
  provably value-neutral guard (``cs_limiters`` iv=0, see there), or
  (b) shown to divide by a layer thickness that is strictly positive on
  every input the NumPy lane also accepts.  No site here admits
  ``lax.cond``: every predicate is elementwise over the column batch, and
  ``lax.cond`` needs a scalar predicate.
* **Static Python ``if`` for static values** (``iv``, ``kord``,
  ``qmin is None``, ``hydrostatic``, ``last_step``, ...); traced ``if``
  never appears.
* **FUNCTIONAL**: the NumPy lane mutates ``a4``/``q``/``pt``/``delp``/...
  in place; every routine here RETURNS the updated arrays.  Return order
  is documented per routine; the driver returns a NamedTuple so the order
  is carried by the type.
* **f64 entry gate** on every public routine, and **no
  ``donate_argnums``** anywhere (grad-path doctrine).

THE 1-BASED CONVENTION IS INHERITED
-----------------------------------
Exactly as in the NumPy lane: vertical indices are FORTRAN's, arrays
carry a dummy slot 0, ``a4``'s first axis is addressed 1..4.
:func:`pad1` / :func:`unpad1` convert at the boundary.

NON-SMOOTH SITES (the gradient is sub-differential AT these points)
------------------------------------------------------------------
1. :func:`cs_limiters` -- the flat / clamp selections (iv=0 positivity
   test, iv=1 extremum test, the ``a6da`` vs ``+-da2`` tests).  C^0 at
   each switch; after a clamp fires the state sits EXACTLY on
   ``a6da == -da2``, which is what the test-side margin check exploits.
2. :func:`_large_scale_constraints` -- ``jnp.minimum`` / ``jnp.maximum``
   clamps on the interface values.
3. :func:`_huynh_edges` -- the nested min/max of Huynh's 2nd constraint.
4. :func:`_interior_block` -- the ``extm`` / ``ext5`` / ``ext6`` boolean
   selections and ``jnp.abs``.
5. :func:`_top_two_layers` / :func:`_bottom_two_layers` -- the iv=0
   ``max(0, .)`` floor and the iv=-1 sign rule.
6. The rezone's interval search: the SELECTED source layer is an integer
   and is piecewise constant in the coordinates, so gradients w.r.t.
   ``pe1``/``pe2`` are sub-differential wherever a target edge sits ON a
   source interface -- which the driver's own construction FORCES at the
   two copied endpoints (``fv_mapz.F90:297-300``).  Gradients w.r.t. the
   FIELD are smooth there.  The contained-cell / spanning-cell select is
   continuous in value across its switch (both arms integrate the same
   parabola, and the spanning arm's first ``m`` term vanishes as the
   target edge approaches the source interface).
7. The ``w_limiter`` clamps and the top escape valve.
8. The omega interval search -- same class as (6).

Gradients are checked (``check_grads(order=2)``) at states proven away
from 1, 2, 3, 4, 6 and 7 by measured margins; the proofs live next to the
tests that use them.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from legoesm.core.fv3_native_mapz import (
    CONSV_MIN,
    R3,
    R12,
    R23,
    T_MIN,
    W_MAX_MAPZ,
    W_MIN_MAPZ,
    ppm_profile_is_unported,
)

__all__ = [
    "CONSV_MIN", "T_MIN", "W_MAX_MAPZ", "W_MIN_MAPZ",
    "LagrangianToEulerianOut",
    "cs_limiters", "cs_profile", "lagrangian_to_eulerian", "map1_ppm",
    "map1_q2", "map_scalar", "pad1", "ppm_profile_is_unported",
    "scalar_profile", "unpad1",
    "make_cs_limiters_jit", "make_cs_profile_jit",
    "make_lagrangian_to_eulerian_jit", "make_map1_ppm_jit",
    "make_map1_q2_jit", "make_map_scalar_jit", "make_pad1_jit",
    "make_scalar_profile_jit", "make_unpad1_jit",
    "cs_limiters_jit", "cs_profile_jit", "lagrangian_to_eulerian_jit",
    "map1_ppm_jit", "map1_q2_jit", "map_scalar_jit", "pad1_jit",
    "scalar_profile_jit", "unpad1_jit",
]

# Index dtype for every search/hint integer.  Pinned to int32 EXPLICITLY:
# a lax.scan carry must keep one dtype across steps, and jnp.arange's
# default width flips with jax_enable_x64.
_IDX = jnp.int32


def _require_f64_jax(fname: str, arrays: dict) -> None:
    """Static-dtype gate, the same one the JAX NH lane applies.

    Reads only ``.dtype`` (static under jit).  A float32 operand would
    otherwise be silently upcast -- or, with x64 disabled, the WHOLE
    remap would run in float32 -- and the oracle build is
    ``-fdefault-real-8``.  Stated locally rather than imported because
    the NH lane's copy is private and this repo forbids importing
    private symbols across modules.
    """
    for name, a in arrays.items():
        if a is None:
            continue
        if jnp.asarray(a).dtype != jnp.float64:
            raise TypeError(
                f"{fname}: {name} must be float64 (got "
                f"{jnp.asarray(a).dtype}); enable jax_enable_x64 and pass "
                f"f64 operands (oracle build is -fdefault-real-8)")


def pad1(a):
    """0-based ``(im, nk)`` -> 1-based ``(im, nk+1)`` with a dummy k=0.

    Twin of ``fv3_native_mapz.pad1``.  Functional (the NumPy lane already
    is here).
    """
    a = jnp.asarray(a)
    _require_f64_jax("pad1", {"a": a})
    return jnp.concatenate([jnp.zeros((a.shape[0], 1), a.dtype), a], axis=1)


def unpad1(a):
    """1-based ``(im, nk+1)`` -> 0-based ``(im, nk)``.

    Twin of ``fv3_native_mapz.unpad1``.  The NumPy lane's
    ``np.ascontiguousarray`` has no JAX counterpart and no meaning here
    (JAX arrays are immutable, so there is nothing to alias).
    """
    a = jnp.asarray(a)
    _require_f64_jax("unpad1", {"a": a})
    return a[:, 1:]


def _zeros_like_col(x, n: int = 1):
    """``(im, n)`` zeros matching ``x``'s batch length and dtype."""
    return jnp.zeros((x.shape[0], n), x.dtype)


def _a6_3x2(q1, a2, a3):
    """``3.*(2.*qbar - (AL+AR))`` -- the grouping used almost everywhere.

    Twin of the NumPy lane's ``_a6_3x2`` (private there, so re-stated).
    """
    return 3.0 * (2.0 * q1 - (a2 + a3))


def _a6_6m3(q1, a2, a3):
    """``6.*qbar - 3.*(AL+AR)`` -- cs_profile kord 9, and kord 12 in both.

    Algebraically identical to :func:`_a6_3x2`, and NOT identical in
    floating point.  Which one a branch uses is copied from the NumPy
    lane line by line; do not "simplify" one into the other.
    """
    return 6.0 * q1 - 3.0 * (a2 + a3)


def _gather_k(a, idx):
    """``a[i, idx[i]]`` for a 1-based ``(im, nk)`` array.

    ``idx`` is clamped into ``[0, nk-1]`` so the gather is always in
    range; a clamped read is only ever CONSUMED behind a ``found``
    select (see :func:`_rezone`), so the clamp cannot leak a plausible
    number into an output.
    """
    n = a.shape[1]
    j = jnp.minimum(jnp.maximum(idx, 0), n - 1)
    return a[jnp.arange(a.shape[0]), j]


def _first_bracket_scan(x, lo_t, hi_t, start, nidx: int):
    """First ``idx`` in ``[start, nidx]`` with ``lo[idx] <= x <= hi[idx]``.

    THE interval-location primitive of this module: the rezone's source
    search (``fv_mapz.F90:1415-1417``) and the omega search (``:1080-1082``
    in the NumPy lane) are the same predicate, so they share one helper
    rather than two copies.

    Traversal is a SEQUENTIAL ``lax.scan`` over ``idx = 1 .. nidx`` in
    ascending order with an explicit FIRST-MATCH rule (``~found`` gates
    every later hit).  NOT ``searchsorted``, NOT ``argmax`` over a mask,
    NOT a vectorised interval test: those decide ties by a different
    rule, and a one-ULP move at an interface then selects a different
    source layer AND a different formula, which is a finite discrepancy,
    not a rounding one.

    ``lo_t``/``hi_t`` are ``(nidx, im)`` -- already transposed for the
    scan.  Returns ``(found, idx)`` with ``idx = 0`` where no interval
    matched.  Both arms of the internal select are integer/boolean, so
    the totality rule is trivially satisfied here.
    """
    idxs = jnp.arange(1, nidx + 1, dtype=_IDX)

    def _step(carry, xs):
        found, sel = carry
        idx, lo_i, hi_i = xs
        hit = (~found) & (idx >= start) & (x >= lo_i) & (x <= hi_i)
        return (found | hit, jnp.where(hit, idx, sel)), None

    init = (jnp.zeros(x.shape, bool), jnp.zeros(x.shape, _IDX))
    (found, sel), _ = lax.scan(_step, init, (idxs, lo_t, hi_t))
    return found, sel


# ---------------------------------------------------------------------------
# cs_limiters -- fv_mapz.F90:2693-2768
# ---------------------------------------------------------------------------

def cs_limiters(a4, extm, iv: int):
    """PPM limiter. ``a4`` is ``(5, ...)``; slots 1..4 used, 0 is the dummy.

    Functional twin of ``fv3_native_mapz.cs_limiters``: that one mutates
    a ``(5, im)`` slice in place, this one RETURNS a new array with slots
    2, 3 and 4 rewritten (slots 0 and 1 carried through untouched).

    Every operation is elementwise over the trailing axes, so ONE
    implementation serves both the oracle's single-layer call
    (``a4`` ``(5, im)``, ``extm`` ``(im,)``) and the vectorised interior
    block (``a4`` ``(5, im, nk)``, ``extm`` ``(im, nk)``) -- no second
    copy of the limiter numerics.

    ``iv`` is STATIC (a Python branch).  The ``iv == 1`` arm ignores
    ``extm`` entirely and RECOMPUTES the extremum test from
    ``(qbar-AL)*(qbar-AR) >= 0``; the ``else`` arm (iv=2, and anything
    not 0/1) uses the PRECOMPUTED ``extm``.  That asymmetry is the
    oracle's (``:2728`` vs ``:2747``) and is load-bearing, which is why
    the ``else`` is NOT hardened into a raise: it is a REACHED branch,
    not an unknown-scheme fall-through.

    Non-smooth: the flat / lo / hi selections are C^0 at their switches.
    After a clamp fires, ``a6*(AR-AL) == -(AR-AL)^2`` EXACTLY, so a
    margin on that identity is a decisive off-switch test.
    """
    a4 = jnp.asarray(a4)
    _require_f64_jax("cs_limiters", {"a4": a4})
    a1 = a4[1]
    a2 = a4[2]
    a3 = a4[3]
    a6 = a4[4]

    if iv == 0:
        # fv_mapz.F90:2702-2727 -- positive-definite constraint.
        #
        # TOTALITY (correction 2): the division by a6 is the ONLY partial
        # operation in this module's selects.  It is sanitised, and the
        # sanitisation is provably VALUE-NEUTRAL: `neg` also requires
        # `abs(a3-a2) < -a6`, which at a6 == 0 reads `x < 0` for a
        # non-negative x and is therefore False, so `parabola_min` cannot
        # change `neg` there.  Sanitising removes the inf/NaN that would
        # otherwise be evaluated inside a select.  (In the NumPy lane the
        # same expression runs under `np.errstate(divide=ignore)` and its
        # inf is likewise inert.)
        a6_safe = jnp.where(a6 == 0.0, jnp.ones_like(a6), a6)
        parabola_min = a1 + 0.25 * (a3 - a2) ** 2 / a6_safe + a6 * R12
        neg = (a1 > 0.0) & (jnp.abs(a3 - a2) < -a6) & (parabola_min < 0.0)
        flat = (a1 <= 0.0) | (neg & (a1 < a3) & (a1 < a2))
        up = neg & (~flat) & (a3 > a2)
        dn = neg & (~flat) & (~(a3 > a2))
        new2 = jnp.where(flat, a1, jnp.where(dn, a3 - 3.0 * (a3 - a1), a2))
        new3 = jnp.where(flat, a1, jnp.where(up, a2 - 3.0 * (a2 - a1), a3))
        new4 = jnp.where(flat, jnp.zeros_like(a6),
                         jnp.where(up, 3.0 * (a2 - a1),
                                   jnp.where(dn, 3.0 * (a3 - a1), a6)))
        return a4.at[2].set(new2).at[3].set(new3).at[4].set(new4)

    if iv == 1:
        flat = (a1 - a2) * (a1 - a3) >= 0.0      # :2730
    else:
        flat = jnp.asarray(extm, dtype=bool)     # :2750

    da1 = a3 - a2
    da2 = da1 ** 2
    a6da = a6 * da1
    lo = (~flat) & (a6da < -da2)
    hi = (~flat) & (a6da > da2)
    new2 = jnp.where(flat, a1, jnp.where(hi, a3 - 3.0 * (a3 - a1), a2))
    new3 = jnp.where(flat, a1, jnp.where(lo, a2 - 3.0 * (a2 - a1), a3))
    new4 = jnp.where(flat, jnp.zeros_like(a6),
                     jnp.where(lo, 3.0 * (a2 - a1),
                               jnp.where(hi, 3.0 * (a3 - a1), a6)))
    return a4.at[2].set(new2).at[3].set(new3).at[4].set(new4)


# ---------------------------------------------------------------------------
# the tridiagonal edge reconstructions
# ---------------------------------------------------------------------------

def _edge_solve_iv_m2(a4, delp, km: int, qs):
    """iv == -2 (fv_mapz.F90:1877-1900, identical at :2287-2310).

    Vertical-velocity reconstruction with a bottom BC ``qs``.  Two
    genuine k-RECURRENCES (a Thomas forward sweep in ``(q, gam)`` and its
    back-substitution) -> two ``lax.scan``s, never associative_scan.

    ``2.0 + grat + grat`` is kept literally: it is
    ``(2.0 + grat) + grat``, which is NOT ``2.0 + 2.0*grat`` in floating
    point.

    Returns the 1-based ``(im, km+2)`` interface array ``q``.
    """
    a1 = a4[1]

    q1 = 1.5 * a1[:, 1]                                     # :1881
    gam2 = jnp.full(a1.shape[:1], 0.5, a1.dtype)            # :1880

    def _fwd(carry, xs):
        q_prev, gam_k = carry
        dlo, dhi, a_km1, a_k = xs
        grat = dlo / dhi
        bet = 2.0 + grat + grat - gam_k
        q_k = (3.0 * (a_km1 + a_k) - q_prev) / bet
        return (q_k, grat / bet), (q_k, grat / bet)

    # k = 2 .. km-1  (the oracle's `do k=2,km-1`)
    xs = (delp[:, 1:km - 1].T, delp[:, 2:km].T,
          a1[:, 1:km - 1].T, a1[:, 2:km].T)
    (q_last, gam_km), (q_mid, gam_mid) = lax.scan(_fwd, (q1, gam2), xs)

    grat = delp[:, km - 1] / delp[:, km]                    # :1893
    q_km = (3.0 * (a1[:, km - 1] + a1[:, km]) - grat * qs - q_last) \
        / (2.0 + grat + grat - gam_km)
    q_kmp1 = qs                                             # :1896

    # gam[k] for k = 2..km : gam(2) is the preset 0.5, the scan emitted
    # gam(3..km).
    gam = jnp.concatenate([gam2[:, None], gam_mid.T], axis=1)   # (im, km-1)
    q_raw = jnp.concatenate([q1[:, None], q_mid.T], axis=1)     # (im, km-1)

    # :1898-1900  back-substitution k = km-1 .. 1
    def _bwd(q_next, xs):
        q_k_raw, gam_kp1 = xs
        q_k = q_k_raw - gam_kp1 * q_next
        return q_k, q_k

    _, q_head = lax.scan(_bwd, q_km, (q_raw.T, gam.T), reverse=True)
    return jnp.concatenate(
        [_zeros_like_col(a1), q_head.T, q_km[:, None], q_kmp1[:, None]],
        axis=1)


def _edge_solve_default(a4, delp, km: int):
    """The ``else`` branch (fv_mapz.F90:1901-1928, and :2341-2362).

    The bottom closure at ``:1918-1922`` reuses ``d4`` LEFT OVER from the
    final iteration of ``do k=2,km`` -- here that is simply the last
    emitted ``d4`` of the forward scan, which is the same value.

    Returns the 1-based ``(im, km+2)`` interface array ``q``.
    """
    a1 = a4[1]

    grat = delp[:, 2] / delp[:, 1]                          # :1904
    bet = grat * (grat + 0.5)
    q1 = ((grat + grat) * (grat + 1.0) * a1[:, 1] + a1[:, 2]) / bet
    gam1 = (1.0 + grat * (grat + 1.5)) / bet

    def _fwd(carry, xs):
        q_prev, gam_prev = carry
        dlo, dhi, a_km1, a_k = xs
        d4 = dlo / dhi
        bet_k = 2.0 + d4 + d4 - gam_prev
        q_k = (3.0 * (a_km1 + d4 * a_k) - q_prev) / bet_k
        gam_k = d4 / bet_k
        return (q_k, gam_k), (q_k, gam_k, d4)

    # k = 2 .. km
    xs = (delp[:, 1:km].T, delp[:, 2:km + 1].T,
          a1[:, 1:km].T, a1[:, 2:km + 1].T)
    (q_km, gam_km), (q_tail, gam_tail, d4s) = lax.scan(_fwd, (q1, gam1), xs)

    d4 = d4s[-1]                                            # the k=km value
    a_bot = 1.0 + d4 * (d4 + 1.5)                           # :1918
    q_kmp1 = (2.0 * d4 * (d4 + 1.0) * a1[:, km] + a1[:, km - 1]
              - a_bot * q_km) / (d4 * (d4 + 0.5) - a_bot * gam_km)

    q_raw = jnp.concatenate([q1[:, None], q_tail.T], axis=1)      # k=1..km
    gam = jnp.concatenate([gam1[:, None], gam_tail.T], axis=1)    # k=1..km

    # :1926-1928  back-substitution k = km .. 1
    def _bwd(q_next, xs):
        q_k_raw, gam_k = xs
        q_k = q_k_raw - gam_k * q_next
        return q_k, q_k

    _, q_head = lax.scan(_bwd, q_kmp1, (q_raw.T, gam.T), reverse=True)
    return jnp.concatenate(
        [_zeros_like_col(a1), q_head.T, q_kmp1[:, None]], axis=1)


def _refuse_iv_m3(km: int) -> None:
    """cs_profile's iv == -3 LBC branch reads UNINITIALISED memory.

    Re-states the NumPy lane's ``_refuse_iv_m3`` (private there, so it
    cannot be imported) on the SAME condition, with the same exception
    type: ``fv_mapz.F90:2320-2339``'s ``do k=2,km-1`` assigns
    ``gam(i,2..km-1)`` only, and the back-substitution at ``:2358-2360``
    reads ``gam(i,km)``, which was never written.  ``gam`` is a local
    automatic array, so that read is undefined -- and a port that
    zero-fills it turns undefined oracle behaviour into a specific,
    confidently-wrong number.

    Unreachable on the pinned lane: it needs ``kord_wz < 0`` AND
    ``hydrostatic = .false.``; the pinned NH deck runs ``kord_wz = 9``.
    """
    raise NotImplementedError(
        f"cs_profile(iv=-3, km={km}): fv_mapz.F90:2320-2339 leaves "
        f"gam(i,km) UNASSIGNED and :2358 reads it, so the oracle's answer "
        f"here depends on uninitialised stack. Refusing to invent one. "
        f"The branch needs kord_wz<0; the pinned NH deck runs kord_wz=9 "
        f"(iv=-2), so this arm stays dead there.")


def _large_scale_constraints(q, a4, km: int, iv: int):
    """fv_mapz.F90:1948-1984 (identical at :2382-2418). Returns ``(q, dq)``.

    VECTORISATION PROOF (correction 3) for the ``do k=3,km-1`` loop: it
    reads ``q(k)``, ``gam(k-1)``, ``gam(k+1)``, ``a4(1,k-1)``,
    ``a4(1,k)`` and writes ``q(k)`` ONLY.  ``gam`` and ``a4(1,.)`` are
    never written inside the loop, and the ``q`` write is index-local, so
    no iteration reads a location an earlier iteration wrote.  It is
    therefore an independent per-k loop, not a recurrence, and is
    vectorised over the identical k window.

    ``q(i,1)`` and ``q(i,km+1)`` are deliberately never clamped.
    Non-smooth: every min/max here is C^0 at its switch.
    """
    a1 = a4[1]
    # :1949-1952
    q2 = jnp.minimum(q[:, 2], jnp.maximum(a1[:, 1], a1[:, 2]))
    q2 = jnp.maximum(q2, jnp.minimum(a1[:, 1], a1[:, 2]))
    q = q.at[:, 2].set(q2)

    # :1954-1958 -- gam := dq  (gam(k) for k = 2..km)
    gam = jnp.concatenate(
        [_zeros_like_col(a1, 2), a1[:, 2:km + 1] - a1[:, 1:km],
         _zeros_like_col(a1)], axis=1)                       # (im, km+2)

    if km > 3:                                               # `do k=3,km-1`
        lo, hi = 3, km                                       # k = 3..km-1
        hi_v = jnp.maximum(a1[:, lo - 1:hi - 1], a1[:, lo:hi])
        lo_v = jnp.minimum(a1[:, lo - 1:hi - 1], a1[:, lo:hi])
        smooth = gam[:, lo - 1:hi - 1] * gam[:, lo + 1:hi + 1] > 0.0
        local_max = gam[:, lo - 1:hi - 1] > 0.0
        qk = q[:, lo:hi]
        at_min = jnp.minimum(qk, hi_v)
        if iv == 0:                                          # :1974
            at_min = jnp.maximum(jnp.zeros_like(at_min), at_min)
        qk = jnp.where(smooth,
                       jnp.minimum(jnp.maximum(qk, lo_v), hi_v),
                       jnp.where(local_max, jnp.maximum(qk, lo_v), at_min))
        q = q.at[:, lo:hi].set(qk)

    # :1980-1983
    hi_v = jnp.maximum(a1[:, km - 1], a1[:, km])
    lo_v = jnp.minimum(a1[:, km - 1], a1[:, km])
    q = q.at[:, km].set(jnp.minimum(jnp.maximum(q[:, km], lo_v), hi_v))
    return q, gam


def _edges_and_extrema(a4, q, gam, km: int, kord: int):
    """fv_mapz.F90:1986-2012 (identical at :2420-2446).

    VECTORISATION PROOF: the ``do k=1,km`` loop writes ``a4(2:4,k)``,
    ``extm(k)``, ``ext5(k)``, ``ext6(k)`` and reads ``q(k)``, ``q(k+1)``,
    ``gam(k)``, ``gam(k+1)`` plus the ``a4(2,k)``/``a4(3,k)`` it wrote in
    the SAME iteration.  No cross-iteration read-after-write, so the loop
    is vectorised with the intra-iteration order preserved (edges first,
    then the extremum tests that read them).

    ``ext5``/``ext6`` are built ONLY when ``abs(kord) > 9``.  At
    abs(kord)==9 the Fortran leaves them undefined AND does not pre-fill
    ``a4(4,:,:)``; this returns ``None`` for them so a wrong read is a
    ``TypeError`` rather than a plausible number.

    Returns ``(a4, extm, ext5, ext6)``.
    """
    a1 = a4[1]
    z1 = _zeros_like_col(a1)
    zb = jnp.zeros((a1.shape[0], 1), bool)

    a2 = jnp.concatenate([z1, q[:, 1:km + 1]], axis=1)       # :1987-1990
    a3 = jnp.concatenate([z1, q[:, 2:km + 2]], axis=1)
    a4 = a4.at[2].set(a2).at[3].set(a3)

    # :1994-2000 -- k=1 and k=km use the edge test, the interior uses gam.
    e_top = ((a2[:, 1] - a1[:, 1]) * (a3[:, 1] - a1[:, 1])) > 0.0
    e_bot = ((a2[:, km] - a1[:, km]) * (a3[:, km] - a1[:, km])) > 0.0
    e_int = gam[:, 2:km] * gam[:, 3:km + 1] < 0.0            # k = 2..km-1
    extm = jnp.concatenate(
        [zb, e_top[:, None], e_int, e_bot[:, None], zb], axis=1)

    if abs(kord) <= 9:                                       # :2003
        return a4, extm, None, None

    x0 = 2.0 * a1[:, 1:km + 1] - (a2[:, 1:km + 1] + a3[:, 1:km + 1])
    x1 = jnp.abs(a2[:, 1:km + 1] - a3[:, 1:km + 1])
    a6 = 3.0 * x0
    a4 = a4.at[4, :, 1:km + 1].set(a6)
    ext5 = jnp.concatenate([zb, jnp.abs(x0) > x1, zb], axis=1)
    ext6 = jnp.concatenate([zb, jnp.abs(a6) > x1, zb], axis=1)
    return a4, extm, ext5, ext6


def _huynh_edges(q1, a2_in, a3_in, gam_km1, gam_k, gam_kp1, gam_kp2):
    """Huynh's 2nd constraint on the two edges (fv_mapz.F90:2056-2064).

    The array reads are hoisted into arguments so the SAME body serves
    the single-layer and the vectorised-k call.  Reaches
    ``gam(k-1) .. gam(k+2)``, which is why every loop that calls it runs
    only ``k = 3 .. km-2``.  Non-smooth: nested min/max.
    """
    pmp_1 = q1 - 2.0 * gam_kp1
    lac_1 = pmp_1 + 1.5 * gam_kp2
    a2 = jnp.minimum(
        jnp.maximum(a2_in, jnp.minimum(jnp.minimum(q1, pmp_1), lac_1)),
        jnp.maximum(jnp.maximum(q1, pmp_1), lac_1))
    pmp_2 = q1 + 2.0 * gam_k
    lac_2 = pmp_2 - 1.5 * gam_km1
    a3 = jnp.minimum(
        jnp.maximum(a3_in, jnp.minimum(jnp.minimum(q1, pmp_2), lac_2)),
        jnp.maximum(jnp.maximum(q1, pmp_2), lac_2))
    return a2, a3


def _require_ext56(ext5, kord: int) -> None:
    if ext5 is None:
        raise AssertionError(
            f"kord={kord} needs ext5/ext6, which fv_mapz.F90:2003 builds "
            f"only when abs(kord) > 9. Unreachable by construction; if you "
            f"see this the dispatch above it is wrong.")


def _interior_block(a4, gam, extm, ext5, ext6, km: int, kord: int,
                    qmin):
    """The ``do k=3,km-2`` interior loop, VECTORISED over its k window.

    VECTORISATION PROOF (correction 3): iteration k reads ``a4(1,k)``,
    ``a4(2,k)``, ``a4(3,k)``, ``gam(k-1..k+2)``, ``extm(k-1..k+1)``,
    ``ext5(k-1..k+1)``, ``ext6(k-1..k+1)`` and writes ``a4(2:4,k)``.
    ``gam``, ``extm``, ``ext5``, ``ext6`` are NEVER written inside this
    loop -- they were all built before it -- and the ``a4`` writes are
    index-local.  So although the loop READS k+-1, it reads them from
    arrays it does not write: no iteration reads a location an earlier
    iteration wrote, and vectorising over the identical window is
    FP-identical to the sequential loop.

    Shared by ``scalar_profile`` and ``cs_profile``.  ``qmin is None``
    selects the ``cs_profile`` text, which lacks every qmin clause and
    uses the ``6.*qbar - 3.*(AL+AR)`` A6 grouping at abs(kord)==9; a
    float ``qmin`` selects the ``scalar_profile`` text.  Those are the
    only differences between the two Fortran bodies in this loop.

    ``kord`` is STATIC.  Every reachable value gets an explicit arm and
    the final ``else`` RAISES: the NumPy lane's trailing ``else`` is
    reached only by abs(kord)==11 (abs>16 returned early at :1932/:2365
    and abs<9 was handled above), so naming 11 and refusing anything else
    is behaviourally identical on every input the lane admits while
    turning an unreachable silent default into a loud error
    (dispatch-hardening).

    Returns the ``(a2, a3, a6)`` block for k = 3 .. km-2.
    """
    scalar = qmin is not None
    ak = abs(int(kord))
    lo, hi = 3, km - 1                       # python slice -> k = 3..km-2
    q1 = a4[1][:, lo:hi]
    a2i = a4[2][:, lo:hi]
    a3i = a4[3][:, lo:hi]
    g_km1 = gam[:, lo - 1:hi - 1]
    g_k = gam[:, lo:hi]
    g_kp1 = gam[:, lo + 1:hi + 1]
    g_kp2 = gam[:, lo + 2:hi + 2]

    def _huynh():
        return _huynh_edges(q1, a2i, a3i, g_km1, g_k, g_kp1, g_kp2)

    if ak < 9:                                            # :2053 / :2487
        a2h, a3h = _huynh()
        return a2h, a3h, _a6_3x2(q1, a2h, a3h)

    e_k = extm[:, lo:hi]
    e_km1 = extm[:, lo - 1:hi - 1]
    e_kp1 = extm[:, lo + 1:hi + 1]

    if ak == 9:                                           # :2069 / :2503
        a6f = _a6_3x2 if scalar else _a6_6m3
        flat = (e_k & e_km1) | (e_k & e_kp1)
        if scalar:                                        # :2081 -- scalar
            flat = flat | (e_k & (q1 < qmin))
        a6 = a6f(q1, a2i, a3i)
        fix = (~flat) & (jnp.abs(a6) > jnp.abs(a2i - a3i))
        a2h, a3h = _huynh()
        a2 = jnp.where(fix, a2h, a2i)
        a3 = jnp.where(fix, a3h, a3i)
        a6 = jnp.where(fix, a6f(q1, a2, a3), a6)
        return (jnp.where(flat, q1, a2), jnp.where(flat, q1, a3),
                jnp.where(flat, jnp.zeros_like(a6), a6))

    _require_ext56(ext5, kord)
    x5_k = ext5[:, lo:hi]
    x6_k = ext6[:, lo:hi]
    near5 = ext5[:, lo - 1:hi - 1] | ext5[:, lo + 1:hi + 1]
    near6 = ext6[:, lo - 1:hi - 1] | ext6[:, lo + 1:hi + 1]

    if ak == 10:                                          # :2102 / :2531
        a2h, a3h = _huynh()
        flat = x5_k & near5
        clamp = ((x5_k & (~near5) & near6)
                 | ((~x5_k) & x6_k & near5))
        a2 = jnp.where(flat, q1, jnp.where(clamp, a2h, a2i))
        a3 = jnp.where(flat, q1, jnp.where(clamp, a3h, a3i))
        return a2, a3, _a6_3x2(q1, a2, a3)

    if ak == 12:                                          # :2134 / :2563
        a6 = _a6_6m3(q1, a2i, a3i)
        fix = (~e_k) & (jnp.abs(a6) > jnp.abs(a2i - a3i))
        a2h, a3h = _huynh()
        a2 = jnp.where(fix, a2h, a2i)
        a3 = jnp.where(fix, a3h, a3i)
        a6 = jnp.where(fix, _a6_6m3(q1, a2, a3), a6)
        return (jnp.where(e_k, q1, a2), jnp.where(e_k, q1, a3),
                jnp.where(e_k, jnp.zeros_like(a6), a6))

    if ak == 13:                                          # :2156 / :2586
        flat = x6_k & ext6[:, lo - 1:hi - 1] & ext6[:, lo + 1:hi + 1]
        a2 = jnp.where(flat, q1, a2i)
        a3 = jnp.where(flat, q1, a3i)
        return a2, a3, _a6_3x2(q1, a2, a3)

    if ak == 14:                                          # :2169 / :2599
        return a2i, a3i, _a6_3x2(q1, a2i, a3i)

    if ak == 15:                                          # :2175 / :2605
        # THE TWO ROUTINES NEST THIS DIFFERENTLY -- not just a qmin delta.
        # scalar_profile (:2176-2195) is a FLAT elseif chain, so `elseif
        # (ext6)` is reachable when ext5(k) is set but no neighbour is.
        # cs_profile (:2607-2623) wraps the ext5 tests inside `if
        # (ext5(k))`, so ext5(k) SWALLOWS the ext6 arm and that point is
        # left untouched.
        flat = x5_k & near5
        if scalar:                                        # :2183 -- scalar
            flat = flat | (x5_k & (q1 < qmin))
            clamp = (~flat) & x6_k
        else:
            clamp = (~x5_k) & x6_k
        a2h, a3h = _huynh()
        a2 = jnp.where(flat, q1, jnp.where(clamp, a2h, a2i))
        a3 = jnp.where(flat, q1, jnp.where(clamp, a3h, a3i))
        return a2, a3, _a6_3x2(q1, a2, a3)

    if ak == 16:                                          # :2200 / :2628
        flat = x5_k & near5
        clamp = x5_k & (~near5) & near6
        a2h, a3h = _huynh()
        a2 = jnp.where(flat, q1, jnp.where(clamp, a2h, a2i))
        a3 = jnp.where(flat, q1, jnp.where(clamp, a3h, a3i))
        return a2, a3, _a6_3x2(q1, a2, a3)

    if ak == 11:                                          # :2223 / :2651
        noisy = x5_k & near5
        if scalar:                                        # :2225 -- scalar
            noisy = x5_k & (near5 | (q1 < qmin))
        a2 = jnp.where(noisy, q1, a2i)
        a3 = jnp.where(noisy, q1, a3i)
        return (a2, a3,
                jnp.where(noisy, jnp.zeros_like(q1), _a6_3x2(q1, a2, a3)))

    raise ValueError(                                     # dispatch-hardening
        f"_interior_block: abs(kord)={ak} reaches no arm. The lane admits "
        f"8 (Huynh), 9, 10, 11, 12, 13, 14, 15, 16 and >16 (handled by the "
        f"caller's :1932/:2365 early return); kord<=7 is refused upstream "
        f"by ppm_profile_is_unported.")


def _top_two_layers(a4, extm, iv: int):
    """fv_mapz.F90:2020-2047 (identical at :2454-2481). ``iv`` STATIC.

    Non-smooth: the iv=0 ``max(0, AL)`` floor and the iv=-1 sign rule are
    C^0 at their switches.
    """
    a1 = a4[1]
    if iv == 0:                                           # :2021
        a4 = a4.at[2, :, 1].set(
            jnp.maximum(jnp.zeros_like(a1[:, 1]), a4[2, :, 1]))
    elif iv == -1:                                        # :2024
        a4 = a4.at[2, :, 1].set(
            jnp.where(a4[2, :, 1] * a1[:, 1] <= 0.0,
                      jnp.zeros_like(a1[:, 1]), a4[2, :, 1]))
    elif iv == 2:                                         # :2027
        a4 = a4.at[2, :, 1].set(a1[:, 1])
        a4 = a4.at[3, :, 1].set(a1[:, 1])
        a4 = a4.at[4, :, 1].set(jnp.zeros_like(a1[:, 1]))

    if iv != 2:                                           # :2032
        a4 = a4.at[4, :, 1].set(_a6_3x2(a1[:, 1], a4[2, :, 1], a4[3, :, 1]))
        a4 = a4.at[:, :, 1].set(cs_limiters(a4[:, :, 1], extm[:, 1], 1))

    # :2043-2046 -- always, iv-independent.
    a4 = a4.at[4, :, 2].set(_a6_3x2(a1[:, 2], a4[2, :, 2], a4[3, :, 2]))
    return a4.at[:, :, 2].set(cs_limiters(a4[:, :, 2], extm[:, 2], 2))


def _bottom_two_layers(a4, extm, km: int, iv: int):
    """fv_mapz.F90:2244-2260 (identical at :2672-2688). ``iv`` STATIC."""
    a1 = a4[1]
    if iv == 0:                                           # :2245
        a4 = a4.at[3, :, km].set(
            jnp.maximum(jnp.zeros_like(a1[:, km]), a4[3, :, km]))
    elif iv == -1:                                        # :2248
        a4 = a4.at[3, :, km].set(
            jnp.where(a4[3, :, km] * a1[:, km] <= 0.0,
                      jnp.zeros_like(a1[:, km]), a4[3, :, km]))
    for k in (km - 1, km):                                # :2253-2259
        a4 = a4.at[4, :, k].set(
            _a6_3x2(a1[:, k], a4[2, :, k], a4[3, :, k]))
        a4 = a4.at[:, :, k].set(
            cs_limiters(a4[:, :, k], extm[:, k], 2 if k == km - 1 else 1))
    return a4


def _profile(a4, delp, km: int, iv: int, kord: int, qmin, qs):
    """Shared body of ``scalar_profile`` (qmin set) and ``cs_profile``.

    Functional: RETURNS the new ``a4``.  ``km``/``iv``/``kord``/``qmin``
    are STATIC (Python branch selectors).
    """
    ppm_profile_is_unported(kord)
    a4 = jnp.asarray(a4)
    delp = jnp.asarray(delp)
    _require_f64_jax("profile", {"a4": a4, "delp": delp, "qs": qs})
    if a4.shape[0] != 5 or a4.shape[2] != km + 1:
        raise ValueError(
            f"a4 must be 1-based (5, im, km+1) for km={km}, got {a4.shape}")
    if delp.shape != (a4.shape[1], km + 1):
        raise ValueError(
            f"delp must be 1-based (im, km+1) = ({a4.shape[1]}, {km + 1}), "
            f"got {delp.shape}")
    if km < 4:
        raise ValueError(
            f"km={km}: the profile routines index a4(1,i,km-1) and run "
            f"cs_limiters on k=1,2,km-1,km, which overlap below km=4. The "
            f"oracle's own coordinate table (fv_eta.F90) starts at km=5.")

    if iv == -2:
        if qs is None:
            raise ValueError("iv=-2 needs the bottom BC qs")
        q = _edge_solve_iv_m2(a4, delp, km, jnp.asarray(qs))
    elif iv == -3 and qmin is None:
        # cs_profile has a dedicated -3 arm; scalar_profile does NOT
        # (fv_mapz.F90:1877 tests only iv == -2), so for scalar_profile
        # iv=-3 falls through to the ordinary else below -- which is why
        # this condition checks qmin rather than iv alone.
        _refuse_iv_m3(km)
    else:
        q = _edge_solve_default(a4, delp, km)

    if abs(int(kord)) > 16:                               # :1932 / :2365
        a2 = jnp.concatenate([_zeros_like_col(q), q[:, 1:km + 1]], axis=1)
        a3 = jnp.concatenate([_zeros_like_col(q), q[:, 2:km + 2]], axis=1)
        a4 = a4.at[2].set(a2).at[3].set(a3)
        return a4.at[4].set(_a6_3x2(a4[1], a2, a3))

    q, gam = _large_scale_constraints(q, a4, km, iv)
    a4, extm, ext5, ext6 = _edges_and_extrema(a4, q, gam, km, kord)
    a4 = _top_two_layers(a4, extm, iv)

    if km > 4:                                            # `do k=3,km-2`
        lo, hi = 3, km - 1
        a2, a3, a6 = _interior_block(a4, gam, extm, ext5, ext6, km, kord,
                                     qmin)
        a4 = a4.at[2, :, lo:hi].set(a2)
        a4 = a4.at[3, :, lo:hi].set(a3)
        a4 = a4.at[4, :, lo:hi].set(a6)
        if iv == 0:                                       # :2237 / :2665
            blk = a4[:, :, lo:hi]
            a4 = a4.at[:, :, lo:hi].set(
                cs_limiters(blk, extm[:, lo:hi], 0))

    return _bottom_two_layers(a4, extm, km, iv)


def scalar_profile(a4, delp, km: int, iv: int, kord: int, qmin: float,
                   qs=None):
    """fv_mapz.F90:1855-2262. Functional twin: RETURNS the new ``a4``.

    ``a4`` is 1-based ``(5, im, km+1)``, ``delp`` 1-based ``(im, km+1)``.
    ``km``/``iv``/``kord``/``qmin`` STATIC.
    """
    if qmin is None:
        raise ValueError("scalar_profile requires qmin (fv_mapz.F90:1867)")
    return _profile(a4, delp, km, iv, kord, float(qmin), qs)


def cs_profile(a4, delp, km: int, iv: int, kord: int, qs=None):
    """fv_mapz.F90:2265-2690. Functional twin: RETURNS the new ``a4``."""
    return _profile(a4, delp, km, iv, kord, None, qs)


# ---------------------------------------------------------------------------
# the rezone -- shared body of map_scalar / map1_ppm / map1_q2
# ---------------------------------------------------------------------------

def _build_q4(q1, pe1, km: int):
    """fv_mapz.F90:1398-1403 -- the a4/dp1 setup shared by all three.

    The whole column is copied into ``q4(1,:,:)`` BEFORE anything writes
    the output, which is what makes the oracle's ``q1``/``q2`` aliasing
    (``pt`` is passed as both) benign; a functional return keeps that for
    free.  Slot ``k=0`` of ``q4(1,.)`` stays ZERO even if the caller's
    ``q1`` carries something there -- the oracle's loop runs k=1..km.
    """
    z = _zeros_like_col(q1)
    dp1 = jnp.concatenate([z, pe1[:, 2:km + 2] - pe1[:, 1:km + 1]], axis=1)
    q4_1 = jnp.concatenate([z, q1[:, 1:km + 1]], axis=1)
    q4 = jnp.stack([jnp.zeros_like(q4_1), q4_1, jnp.zeros_like(q4_1),
                    jnp.zeros_like(q4_1), jnp.zeros_like(q4_1)], axis=0)
    return q4, dp1


def _rezone(q4, dp1, pe1, pe2, km: int, kn: int, dp2=None):
    """fv_mapz.F90:1412-1451, and :1715-1754 with ``dp2`` supplied.

    THE hardest node of the port.  Structure, stated explicitly because
    every part of it is load-bearing:

    * **The target-layer loop is a SEQUENTIAL ``lax.scan``, not a
      vectorised one.**  ``k0`` (``:1413``, updated at ``:1424`` and
      ``:1441``) is a monotone search hint threaded FROM one target layer
      TO the next.  Vectorising the target loop would change both the
      traversal and the tie behaviour, because every layer would restart
      the search at 1.
    * **State.** ``k0`` is the ONLY component that crosses a target
      layer; ``found`` (did any source interval bracket the top edge),
      ``qsum`` (the sequential partial integral) and ``out`` (the layer's
      answer) are per-layer and live inside the step.  All four are
      explicit below.
    * **Both inner loops are STATIC-LENGTH ``lax.scan``s over
      ``1 .. km``** with a ``>= k0`` / ``> l`` mask, never a
      ``lax.while_loop`` (a data-dependent while_loop is not
      reverse-mode differentiable, and this lane exists for the adjoint)
      and never a vectorised interval mask or ``searchsorted`` (either
      would change which layer wins a tie).  The masked steps leave the
      accumulator EXACTLY unchanged (``where(active, qsum+inc, qsum)``,
      not ``qsum + where(active, inc, 0)``, which differs on a -0.0
      accumulator).
    * **``qsum`` accumulates in ascending ``m``**, one addition per step,
      so the association order is the Fortran's.  No ``cumsum``, no
      ``associative_scan``.
    * **The two formulas are written differently ON PURPOSE**: a target
      cell wholly inside one source cell gets the closed-form parabola
      mean with NO division by the target thickness, while a spanning
      cell accumulates and divides once (``:1449``).  Reproducing that
      split is what makes bit-exactness reachable.
    * **TOTALITY of the selects (correction 2).**  Every division here
      divides by a SOURCE LAYER THICKNESS ``dp1[m]`` -- strictly
      positive on every input the NumPy lane also accepts -- or by the
      target thickness ``denom``, which is the same quantity the NumPy
      lane divides by.  The gathers at the selected layer are index-
      CLAMPED so they are always in range.  Consequently no arm of any
      ``where`` can manufacture a NaN/Inf that would contaminate the
      reverse-mode gradient of the selected arm.  ``denom`` is
      deliberately NOT sanitised: a zero target thickness is degenerate
      input in both lanes, and substituting a finite divisor there would
      manufacture exactly the plausible-looking number this port refuses
      to invent.
    * **THE NO-BRACKET PATH.**  The NumPy lane RAISES
      ``FloatingPointError`` when ``pe2(i,k)`` is bracketed by no source
      interval (the Fortran falls through to label 123 with ``qsum``
      UNDEFINED).  A data-dependent raise cannot survive ``jit``, so this
      twin does two things instead, and NEITHER is a plausible number:
      it writes ``NaN`` into that target layer, and it RETURNS an
      explicit ``ok`` flag (all layers of all columns bracketed).  The
      public map routines surface that flag on request
      (``return_ok=True``) and always write the NaN.

    ``dp2`` is the ONLY difference between ``map_scalar``/``map1_ppm``
    (which divide by ``pe2(k+1)-pe2(k)``) and ``map1_q2`` (which divides
    by the driver's precomputed ``dp2``).  Those are equal in exact
    arithmetic and not in floating point, so the caller must pass the one
    its oracle counterpart uses.

    Returns ``(q2, ok)`` -- ``q2`` 1-based ``(im, kn+1)``, ``ok`` a
    scalar bool.
    """
    im = q4.shape[1]
    q4_1, q4_2, q4_3, q4_4 = q4[1], q4[2], q4[3], q4[4]
    # Transposed once, reused by both inner scans (k-major for lax.scan).
    pe1_l_t = pe1[:, 1:km + 1].T                 # pe1(l)   , l = 1..km
    pe1_r_t = pe1[:, 2:km + 2].T                 # pe1(l+1) , l = 1..km
    dp1_t = dp1[:, 1:km + 1].T
    q1_t, q2_t, q3_t, q4_t = (q4_1[:, 1:km + 1].T, q4_2[:, 1:km + 1].T,
                              q4_3[:, 1:km + 1].T, q4_4[:, 1:km + 1].T)
    ms = jnp.arange(1, km + 1, dtype=_IDX)

    top_t = pe2[:, 1:kn + 1].T                   # pe2(k)  , k = 1..kn
    bot_t = pe2[:, 2:kn + 2].T                   # pe2(k+1)
    # :1449 vs :1754 -- the divisor is the caller's choice, and the two
    # expressions are NOT interchangeable in floating point.
    den_t = (bot_t - top_t) if dp2 is None else dp2[:, 1:kn + 1].T

    def _target_step(k0, xs):
        pe2k, pe2kp1, denom = xs

        # --- locate the top edge: first l >= k0 with pe1(l) <= pe2(k)
        #     <= pe1(l+1)  (:1415-1417), sequential, first match wins.
        found, ell = _first_bracket_scan(pe2k, pe1_l_t, pe1_r_t, k0, km)

        pe1_l = _gather_k(pe1[:, 1:km + 1], ell - 1)      # pe1(l)
        pe1_r = _gather_k(pe1[:, 2:km + 2], ell - 1)      # pe1(l+1)
        dpl = _gather_k(dp1[:, 1:km + 1], ell - 1)
        a2l = _gather_k(q4_2[:, 1:km + 1], ell - 1)
        a3l = _gather_k(q4_3[:, 1:km + 1], ell - 1)
        a6l = _gather_k(q4_4[:, 1:km + 1], ell - 1)

        pl = (pe2k - pe1_l) / dpl                        # :1418
        inside = pe2kp1 <= pe1_r                         # :1419

        # :1421-1423 -- entire target cell inside source cell l.
        pr = (pe2kp1 - pe1_l) / dpl
        q_in = (a2l + 0.5 * (a6l + a3l - a2l) * (pr + pl)
                - a6l * R3 * (pr * (pr + pl) + pl ** 2))

        # :1428-1430 -- the fractional head of a spanning cell.
        qsum0 = ((pe1_r - pe2k)
                 * (a2l + 0.5 * (a6l + a3l - a2l) * (1.0 + pl)
                    - a6l * (R3 * (1.0 + pl * (1.0 + pl)))))

        # :1431-1444 -- walk m = l+1 .. km, whole layers then the last
        # partial one.  Sequential accumulation, ascending m.
        def _m_step(carry, mx):
            qsum, mdone, k0m = carry
            m, pe1_m, pe1_mp1, dp1_m, a1m, a2m, a3m, a6m = mx
            active = (~mdone) & (m > ell)
            whole = pe2kp1 > pe1_mp1                     # :1433
            dp = pe2kp1 - pe1_m                          # :1437
            esl = dp / dp1_m
            inc = jnp.where(
                whole,
                dp1_m * a1m,                             # :1435
                dp * (a2m + 0.5 * esl                    # :1439-1440
                      * (a3m - a2m + a6m * (1.0 - R23 * esl))))
            qsum = jnp.where(active, qsum + inc, qsum)
            hit = active & (~whole)                      # :1441 k0 = m
            return (qsum, mdone | hit, jnp.where(hit, m, k0m)), None

        (qsum, _, k0m), _ = lax.scan(
            _m_step, (qsum0, jnp.zeros((im,), bool), k0),
            (ms, pe1_l_t, pe1_r_t, dp1_t, q1_t, q2_t, q3_t, q4_t))

        q_span = qsum / denom                            # :1449
        out = jnp.where(found, jnp.where(inside, q_in, q_span), jnp.nan)
        # k0 advances to l on the contained branch (:1424) and to the
        # terminating m on the spanning branch (:1441); if the m walk
        # never terminates the Fortran reaches 123 with k0 UNCHANGED,
        # which is what k0m carries.
        k0_new = jnp.where(found, jnp.where(inside, ell, k0m), k0)
        return k0_new, (out, found)

    k0_0 = jnp.ones((im,), _IDX)                         # :1413
    _, (out_t, found_t) = lax.scan(
        _target_step, k0_0, (top_t, bot_t, den_t))
    q2 = jnp.concatenate([_zeros_like_col(pe2), out_t.T], axis=1)
    return q2, jnp.all(found_t)


def _map_common(pe1, q1, pe2, km: int, kn: int, name: str):
    """Shared entry gate + shape checks for the three map routines."""
    pe1 = jnp.asarray(pe1)
    q1 = jnp.asarray(q1)
    pe2 = jnp.asarray(pe2)
    _require_f64_jax(name, {"pe1": pe1, "q1": q1, "pe2": pe2})
    im = q1.shape[0]
    if pe1.shape != (im, km + 2):
        raise ValueError(f"{name}: pe1 must be 1-based (im, km+2) = "
                         f"({im}, {km + 2}), got {pe1.shape}")
    if q1.shape != (im, km + 1):
        raise ValueError(f"{name}: q1 must be 1-based (im, km+1) = "
                         f"({im}, {km + 1}), got {q1.shape}")
    if pe2.shape != (im, kn + 2):
        raise ValueError(f"{name}: pe2 must be 1-based (im, kn+2) = "
                         f"({im}, {kn + 2}), got {pe2.shape}")
    return pe1, q1, pe2


def map_scalar(pe1, q1, pe2, km: int, kn: int, iv: int, kord: int,
               q_min: float, qs=None, *, return_ok: bool = False):
    """fv_mapz.F90:1361-1453 -- selects :func:`scalar_profile`.

    All arrays 1-based: ``pe1`` (im, km+2), ``q1`` (im, km+1), ``pe2``
    (im, kn+2).  On the pinned lane the coordinate is ln(p), so the
    conserved integral is INT(T dlnp) -- NOT INT(theta dp).

    Returns the 1-based ``(im, kn+1)`` remapped column, or
    ``(q2, ok)`` when ``return_ok`` (STATIC) is set -- see
    :func:`_rezone` on the no-bracket path.  ``km``/``kn``/``iv``/
    ``kord``/``q_min`` are STATIC.
    """
    pe1, q1, pe2 = _map_common(pe1, q1, pe2, km, kn, "map_scalar")
    q4, dp1 = _build_q4(q1, pe1, km)
    q4 = scalar_profile(q4, dp1, km, iv, kord, q_min, qs=qs)
    q2, ok = _rezone(q4, dp1, pe1, pe2, km, kn)
    return (q2, ok) if return_ok else q2


def map1_ppm(pe1, q1, pe2, km: int, kn: int, iv: int, kord: int, qs=None,
             *, return_ok: bool = False):
    """fv_mapz.F90:1456-1547 -- selects :func:`cs_profile`, coordinate p."""
    pe1, q1, pe2 = _map_common(pe1, q1, pe2, km, kn, "map1_ppm")
    q4, dp1 = _build_q4(q1, pe1, km)
    q4 = cs_profile(q4, dp1, km, iv, kord, qs=qs)
    q2, ok = _rezone(q4, dp1, pe1, pe2, km, kn)
    return (q2, ok) if return_ok else q2


def map1_q2(pe1, q1, pe2, dp2, km: int, kn: int, iv: int, kord: int,
            q_min: float, qs=None, *, return_ok: bool = False):
    """fv_mapz.F90:1666-1756 -- the tracer remap.

    Same builder as :func:`map_scalar`, but the spanning-cell branch
    divides by the driver's ``dp2`` (1-based (im, kn+1)) rather than by
    ``pe2(k+1)-pe2(k)``.
    """
    pe1, q1, pe2 = _map_common(pe1, q1, pe2, km, kn, "map1_q2")
    dp2 = jnp.asarray(dp2)
    _require_f64_jax("map1_q2", {"dp2": dp2})
    if dp2.shape != (q1.shape[0], kn + 1):
        raise ValueError(f"map1_q2: dp2 must be 1-based (im, kn+1) = "
                         f"({q1.shape[0]}, {kn + 1}), got {dp2.shape}")
    q4, dp1 = _build_q4(q1, pe1, km)
    q4 = scalar_profile(q4, dp1, km, iv, kord, q_min, qs=qs)
    q2, ok = _rezone(q4, dp1, pe1, pe2, km, kn, dp2=dp2)
    return (q2, ok) if return_ok else q2


# ---------------------------------------------------------------------------
# Lagrangian_to_Eulerian -- fv_mapz.F90:62-1080
# ---------------------------------------------------------------------------

class LagrangianToEulerianOut(NamedTuple):
    """Every array the NumPy lane writes in place, in ITS argument order.

    A NamedTuple rather than a bare tuple: thirteen outputs is exactly
    the arity at which a silent mis-unpack stops being hypothetical, and
    a NamedTuple is still an ordinary pytree for jit/grad.

    ``q`` is a tuple of tracer arrays (possibly empty).  ``w`` and
    ``delz`` are ``None`` on the hydrostatic lane, exactly as they are
    unused there.
    """
    pe: jnp.ndarray
    peln: jnp.ndarray
    pk: jnp.ndarray
    pkz: jnp.ndarray
    delp: jnp.ndarray
    pt: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    ps: jnp.ndarray
    q: tuple
    omga: jnp.ndarray
    w: jnp.ndarray
    delz: jnp.ndarray


def _ikj_cols(a):
    """``(ni, nk, nj)`` -> ``(nj*ni, nk)`` columns, batch order (j, i)."""
    return jnp.transpose(a, (2, 0, 1)).reshape(-1, a.shape[1])


def _cols_ikj(x, nj: int, ni: int, nk: int):
    return jnp.transpose(x.reshape(nj, ni, nk), (1, 2, 0))


def _ijk_cols(a):
    """``(ni, nj, nk)`` -> ``(nj*ni, nk)`` columns, batch order (j, i)."""
    return jnp.transpose(a, (1, 0, 2)).reshape(-1, a.shape[2])


def _cols_ijk(x, nj: int, ni: int, nk: int):
    return jnp.transpose(x.reshape(nj, ni, nk), (1, 0, 2))


def _ij_cols(a):
    """``(ni, nj)`` -> ``(nj*ni,)`` columns, batch order (j, i)."""
    return a.T.reshape(-1)


def _cols_ij(x, nj: int, ni: int):
    return x.reshape(nj, ni).T


def _w_limiter_column(w2, d2, km: int):
    """fv_mapz.F90:368-418 -- the momentum-conserving w clamp.

    TWO ORDERED ``lax.scan``s, one per pass direction, and NOT a bulk
    ``.at[k+1].add`` or a precomputed vector expression (correction 3):
    the down pass at ``:373-390`` clips level k and then adds the spill
    into level k+1, which the NEXT iteration reads; the up pass
    ``:391-407`` does the mirror.  Each level therefore reads what the
    previous iteration just wrote, so this is a genuine recurrence in
    both directions.

    TOTALITY (correction 2): the three arms of the spill select are
    total polynomials -- the only division, ``spill / d2(k+-1)``, sits
    OUTSIDE the select and divides by an Eulerian layer thickness that
    is strictly positive on every admitted input.

    Bounds are the MODULE PARAMETERS w_max=90 / w_min=-60
    (``fv_mapz.F90:51-52``, imported from the NumPy twin); the deck's
    namelist ``W_MAX=75`` is ``flagstruct%w_max``, a DIFFERENT variable
    this routine never reads.

    ``w2``/``d2`` are 0-based ``(im, km)``.  Returns the clamped ``w``.
    """

    def _spill(wk, dk):
        return jnp.where(
            wk > W_MAX_MAPZ, (wk - W_MAX_MAPZ) * dk,
            jnp.where(wk < W_MIN_MAPZ, (wk - W_MIN_MAPZ) * dk,
                      jnp.zeros_like(wk)))

    def _clip(wk):
        return jnp.minimum(jnp.maximum(wk, W_MIN_MAPZ), W_MAX_MAPZ)

    def _down(w_cur, xs):                                  # :373-390
        w_next_in, d_k, d_kp1 = xs
        sp = _spill(w_cur, d_k)
        return w_next_in + sp / d_kp1, _clip(w_cur)

    w_bot, w_head = lax.scan(
        _down, w2[:, 0],
        (w2[:, 1:km].T, d2[:, 0:km - 1].T, d2[:, 1:km].T))
    w_down = jnp.concatenate([w_head.T, w_bot[:, None]], axis=1)

    def _up(w_cur, xs):                                    # :391-407
        w_prev_in, d_k, d_km1 = xs
        sp = _spill(w_cur, d_k)
        return w_prev_in + sp / d_km1, _clip(w_cur)

    w_top, w_tail = lax.scan(
        _up, w_down[:, km - 1],
        (w_down[:, 0:km - 1].T, d2[:, 1:km].T, d2[:, 0:km - 1].T),
        reverse=True)
    w_new = jnp.concatenate([w_top[:, None], w_tail.T], axis=1)

    # :408-416 -- top escape valve at 2x the bounds; the momentum spilled
    # past it is DISCARDED (documented oracle intent, not a bug).
    return w_new.at[:, 0].set(
        jnp.minimum(jnp.maximum(w_new[:, 0], 2.0 * W_MIN_MAPZ),
                    2.0 * W_MAX_MAPZ))


def _omega_interp(mid, pe0_old, pe3_om, om_in, km: int):
    """fv_mapz.F90:504-523 -- omega onto the remapped cell centres.

    Sequential ``lax.scan`` over the target centres, carrying ``k_next``
    exactly as the Fortran does (``:1078-1088`` in the NumPy lane): the
    hint crosses target levels, so this loop is NOT vectorised.  The
    interval search is the shared :func:`_first_bracket_scan`.

    Where no source interval brackets a centre the Fortran's assignment
    simply never executes, so the INCOMING omega is kept -- that is a
    real oracle behaviour, not an error path, and is reproduced with a
    ``where`` (both arms total: the gathers are index-clamped and the
    divisor is a log-pressure layer thickness).
    """
    lo_t = pe0_old[:, 1:km + 1].T
    hi_t = pe0_old[:, 2:km + 2].T

    def _step(k_next, xs):
        x, om_old = xs
        found, kk = _first_bracket_scan(x, lo_t, hi_t, k_next, km)
        p_k = _gather_k(pe0_old[:, 1:km + 1], kk - 1)
        p_kp1 = _gather_k(pe0_old[:, 2:km + 2], kk - 1)
        e_k = _gather_k(pe3_om[:, 1:km + 1], kk - 1)
        e_kp1 = _gather_k(pe3_om[:, 2:km + 2], kk - 1)
        val = e_k + (e_kp1 - e_k) * (x - p_k) / (p_kp1 - p_k)
        return (jnp.where(found, kk, k_next),
                jnp.where(found, val, om_old))

    _, om_t = lax.scan(_step, jnp.ones(mid.shape[:1], _IDX),
                       (mid.T, om_in.T))
    return om_t.T


def _refuse_unported_lane(*, consv: float, fill: bool, kord_tm: int,
                          do_sat_adj: bool, do_inline_mp: bool,
                          do_adiabatic_init: bool, nq: int,
                          last_step: bool) -> None:
    """Reject every configuration whose ``fv_mapz`` branch is not ported.

    Re-states the NumPy lane's private ``_refuse_unported_lane`` on the
    SAME conditions, in the same order, with the same exception type.
    Each of these is a REAL oracle branch that neither lane carries;
    silently taking the ported branch under a different flag would run
    different physics and still return plausible numbers.
    """
    # The energy fixer is inside `if (last_step .and. ...)` at :628, so a
    # non-last_step call never reaches it whatever consv says.
    if last_step and (consv > CONSV_MIN or consv < -CONSV_MIN):
        raise NotImplementedError(
            f"consv={consv} at last_step: the total-energy fixer "
            f"(fv_mapz.F90:628-747) is NOT ported. The reference deck pins "
            f"consv_te=0.0, which leaves dtmp exactly 0. |consv| must be "
            f"<= {CONSV_MIN}.")
    # fillz is called at :336, INSIDE the `elseif (nq > 0)` tracer arm
    # opened at :330 -- with no tracers it is unreachable.
    if fill and nq > 0:
        raise NotImplementedError(
            "fill=True with tracers: fillz (fv_mapz.F90:336) is NOT ported; "
            "the reference deck pins fill=.F.")
    if int(kord_tm) >= 0:
        raise NotImplementedError(
            f"kord_tm={kord_tm} >= 0: the positive-kord_tm lane is a "
            f"DIFFERENT operator, not a limiter variant -- it remaps "
            f"Theta_v in linear p via map1_ppm/cs_profile (fv_mapz.F90:"
            f":319-321), needs pkez (:240) and the te array, and converts "
            f"pt at :495-501 instead of :209-217. Port pkez and that arm "
            f"before using it.")
    if do_sat_adj or do_inline_mp or do_adiabatic_init:
        raise NotImplementedError(
            "do_sat_adj / do_inline_mp / do_adiabatic_init: the fast "
            "saturation-adjustment and inline-MP blocks (fv_mapz.F90:"
            "584-625, 748-820, 1010-1078) are NOT ported.")
    if nq > 5:
        raise NotImplementedError(
            f"nq={nq} > 5 selects mapn_tracer (fv_mapz.F90:327), which is "
            f"NOT ported. The reference deck has nr=2 (ncnst=3, dnats=1).")


def lagrangian_to_eulerian(*, pe, peln, pk, pkz, delp, pt, u, v, ps,
                           ak, bk, ptop, akap, cp, r_vir,
                           km, n, ng,
                           kord_mt, kord_tm, kord_tr,
                           q, omga=None, sphum_index=None,
                           last_step=True,
                           hydrostatic=True, adiabatic=True, consv=0.0,
                           w=None, delz=None, ws=None, kord_wz=9,
                           w_limiter=False, rdgas=None, grav=None,
                           fill=False, do_sat_adj=False, do_inline_mp=False,
                           do_adiabatic_init=False):
    """``Lagrangian_to_Eulerian`` for ONE face (fv_mapz.F90:62-1080).

    FUNCTIONAL twin of ``fv3_native_mapz.lagrangian_to_eulerian``: that
    one mutates its operands, this one RETURNS
    :class:`LagrangianToEulerianOut` -- fields in the NumPy lane's own
    argument order ``(pe, peln, pk, pkz, delp, pt, u, v, ps, q, omga, w,
    delz)``.  Halo cells outside each written window are carried through
    unchanged from the operands.

    Array layouts are the NumPy lane's exactly (``fv3_native_state_3d``
    contract, 0-based numpy with the Fortran origins implied)::

        delp, pt, ps  (m_a, m_a, km) / (m_a, m_a)   [isd, jsd]
        u             (m_a, m_b, km)                [isd, jsd]
        v             (m_b, m_a, km)                [isd, jsd]
        pk            (m_a, m_a, km+1)              [isd, jsd]
        pe            (n+2, km+1, n+2)              [is-1, 1, js-1] (i,k,j)
        peln          (n,   km+1, n)                [is,   1, js]   (i,k,j)
        pkz           (n, n, km)                    [is, js, 1]

    with ``m_a = n + 2*ng``, ``m_b = m_a + 1``, ``is = js = 1``,
    ``ie = je = n``.

    VECTORISATION PROOF for the ``do 1000 j`` loop (correction 3): row j
    reads ``pe`` (which is NOT written until after the loop -- the new
    interfaces are stashed in the separate ``pe4`` buffer at ``:572-576``
    and installed at ``:618-625``), and writes ``peln``/``pkz``/``delz``/
    ``ws`` only at its own row and ``pt``/``delp``/``pk``/``ps``/``q``/
    ``omga``/``w``/``u``/``v`` only at its own ``jd``.  No row reads a
    location another row writes -- including the u remap's deliberate
    read of the OLD ``pe(...,j-1)`` -- so the rows are batched into the
    column axis of the map routines.  That is FP-identical because every
    operation is elementwise per column.

    ``q`` is REQUIRED and has no default: it is the list of tracer arrays
    (each shaped like ``delp``), and ``[]`` means "this configuration has
    none".  There is deliberately no ``qmin`` knob for the tracer remap:
    the oracle passes the literal ``0.`` at ``:335``.

    PRECONDITION THAT IS NOT CHECKABLE HERE: ``pe``'s one-cell halo ring
    must be current -- the u/v remap reads ``pe(i,k,j-1)`` (:536) and
    ``pe(i-1,k,j)`` (:562).  A caller that skips the final ``geopk`` will
    remap the winds against a stale ring and the error will look like a
    boundary-row physics defect.

    ``pt`` enters as virtual POTENTIAL temperature and leaves as ``T``
    (``last_step``) or back as ``theta_v`` (``:994-1002``).
    """
    if q is None or not isinstance(q, (list, tuple)):
        raise TypeError(
            "q must be a list/tuple of tracer arrays ([] for none), not "
            f"{type(q).__name__}. It has no default so that skipping "
            "the tracer remap is always a visible choice: the pinned "
            "deck has nr=2 and the oracle makes two passes through "
            "fv_mapz.F90:330-342.")
    nq = len(q)
    _refuse_unported_lane(consv=consv, fill=fill, kord_tm=kord_tm,
                          do_sat_adj=do_sat_adj, do_inline_mp=do_inline_mp,
                          do_adiabatic_init=do_adiabatic_init, nq=nq,
                          last_step=last_step)
    ppm_profile_is_unported(kord_mt)
    ppm_profile_is_unported(abs(int(kord_tm)))
    kords_tr = ([int(x) for x in kord_tr]
                if isinstance(kord_tr, (list, tuple))
                else [int(kord_tr)] * nq)
    if len(kords_tr) != nq:
        raise ValueError(f"kord_tr has {len(kords_tr)} entries for {nq} "
                         f"tracers")
    for kt in kords_tr:
        ppm_profile_is_unported(kt)
    if r_vir != 0.0 and last_step:
        # :975 divides by (1 + r_vir*q(...,sphum)) using the EXPLICIT
        # `sphum` DUMMY ARGUMENT declared at :80 -- guessing tracer 0
        # would silently divide by the wrong species.
        if nq == 0:
            raise ValueError(
                "r_vir != 0 needs the tracers for the closing T_v -> T "
                "conversion (fv_mapz.F90:975). The reference deck is "
                "adiabatic (driver/solo/atmosphere.F90:156-158, zvir = 0).")
        if sphum_index is None or not (0 <= int(sphum_index) < nq):
            raise ValueError(
                f"r_vir != 0 needs sphum_index in [0, {nq}) -- "
                f"fv_mapz.F90:975 uses the explicit sphum argument, and "
                f"assuming tracer 0 divides by the wrong species. Got "
                f"{sphum_index!r}.")
    if not hydrostatic:
        missing = [nm for nm, a in (("w", w), ("delz", delz), ("ws", ws),
                                    ("rdgas", rdgas), ("grav", grav))
                   if a is None]
        if missing:
            raise ValueError(
                f"hydrostatic=False needs {missing}: the w/delz remap "
                f"(fv_mapz.F90:345-365) reads w, delz and ws, and the NH "
                f"pkz (:480) needs rrg = -rdgas/grav (:167)")
        if int(kord_wz) < 0:
            # kord_wz < 0 selects iv=-3 (:347-351), whose cs_profile LBC
            # branch reads uninitialised memory.
            _refuse_iv_m3(km)
    if last_step and omga is None:
        raise ValueError(
            "last_step=True requires omga: fv_mapz.F90:504-523 interpolates "
            "omega onto the remapped cell centres on every last_step, so "
            "omitting it is a silent divergence. Pass the field (it is "
            "returned updated); pass last_step=False if this is not the "
            "final outer split.")

    pe, peln, pk, pkz = (jnp.asarray(pe), jnp.asarray(peln),
                         jnp.asarray(pk), jnp.asarray(pkz))
    delp, pt, u, v, ps = (jnp.asarray(delp), jnp.asarray(pt),
                          jnp.asarray(u), jnp.asarray(v), jnp.asarray(ps))
    ak, bk = jnp.asarray(ak), jnp.asarray(bk)
    q = [jnp.asarray(x) for x in q]
    _require_f64_jax("lagrangian_to_eulerian", {
        "pe": pe, "peln": peln, "pk": pk, "pkz": pkz, "delp": delp,
        "pt": pt, "u": u, "v": v, "ps": ps, "ak": ak, "bk": bk,
        "omga": omga, "w": w, "delz": delz, "ws": ws,
        **{f"q[{i}]": x for i, x in enumerate(q)}})
    if ak.shape != (km + 1,) or bk.shape != (km + 1,):
        raise ValueError(f"ak/bk must be (km+1,)={km + 1}, got {ak.shape} "
                         f"and {bk.shape}")

    ak1 = pad1(ak[None, :])[0]                      # 1-based k
    bk1 = pad1(bk[None, :])[0]
    ia = ng
    ipe = 1
    abs_kord_tm = abs(int(kord_tm))
    nb = n * n
    z1 = jnp.zeros((nb, 1), pe.dtype)

    # ---------------- the `j /= je+1` block, rows j = 1..n --------------
    pe_row = _ikj_cols(pe[ipe:ipe + n, :, 1:1 + n])          # :197-201
    pe1 = pad1(pe_row)
    psf = pe_row[:, km]
    pe2 = jnp.concatenate([                                  # :203-206
        z1, jnp.full((nb, 1), ptop, pe.dtype),
        ak1[2:km + 1][None, :] + bk1[2:km + 1][None, :] * psf[:, None],
        psf[:, None]], axis=1)                               # :267-273
    dp2 = jnp.concatenate(                                   # :274-276
        [z1, pe2[:, 2:km + 2] - pe2[:, 1:km + 1]], axis=1)

    ptw = _ijk_cols(pt[ia:ia + n, ia:ia + n, :])
    dpw = _ijk_cols(delp[ia:ia + n, ia:ia + n, :])
    pkw = _ijk_cols(pk[ia:ia + n, ia:ia + n, :])
    plnw = _ikj_cols(peln)
    dzw = None
    if hydrostatic:
        # :211-217 -- Theta_v -> T_v with the LAGRANGIAN hydrostatic pkz.
        ptw = ptw * ((pkw[:, 1:] - pkw[:, :-1])
                     / (akap * (plnw[:, 1:] - plnw[:, :-1])))
    else:
        # :218-237 (non-moist arm, :231-232) with the PRE-conversion
        # delz/delp -- NOT the hydrostatic Dpk/(akap Dpeln) form.
        k1k = rdgas / (cp - rdgas)                           # :164
        dzw = _ijk_cols(jnp.asarray(delz))
        ptw = ptw * jnp.exp(k1k * jnp.log(-rdgas / grav * dpw / dzw * ptw))
        dzw = -dzw / dpw                                     # :252-258

    ps_w = pe1[:, km + 1]                                    # :261-263
    delp_w = unpad1(dp2)                                     # :281-285

    # :290-308 -- p**kappa on both coordinates; the ENDPOINTS ARE COPIED
    # (:297-300), which is what makes source and target coincide to the
    # last bit.
    pk1 = pad1(pkw)
    pn2 = jnp.concatenate(
        [z1, plnw[:, 0:1], jnp.log(pe2[:, 2:km + 1]),
         plnw[:, km:km + 1]], axis=1)
    pk2 = jnp.concatenate(
        [z1, pk1[:, 1:2], jnp.exp(akap * pn2[:, 2:km + 1]),
         pk1[:, km + 1:km + 2]], axis=1)

    # :310-316 -- map T in ln(p) with scalar_profile.
    peln1 = pad1(plnw)
    ptw = unpad1(map_scalar(peln1, pad1(ptw), pn2, km, km, 1,
                            abs_kord_tm, T_MIN))

    # :330-343 -- one tracer at a time (nq <= 5 on this lane).
    q_out = []
    for iq in range(nq):
        qw = _ijk_cols(q[iq][ia:ia + n, ia:ia + n, :])
        q_out.append(unpad1(map1_q2(pe1, pad1(qw), pe2, dp2, km, km, 0,
                                    kords_tr[iq], 0.0)))     # :335 literal 0.

    ww = None
    if not hydrostatic:
        # :347-355 -- w in LINEAR p; kord_wz=9 selects iv=-2, whose bottom
        # BC is the D-stage surface velocity ws.
        ww = _ijk_cols(jnp.asarray(w)[ia:ia + n, ia:ia + n, :])
        ww = unpad1(map1_ppm(pe1, pad1(ww), pe2, km, km, -2,
                             abs(int(kord_wz)), qs=_ij_cols(jnp.asarray(ws))))
        # :357-360 -- delz (specific volume) with iv=1, kord_tm.
        dzw = unpad1(map1_ppm(pe1, pad1(dzw), pe2, km, km, 1, abs_kord_tm))
        dzw = -dzw * delp_w                                  # :361-365
        if w_limiter:                                        # :368-418
            ww = _w_limiter_column(ww, delp_w, km)

    pk_w = unpad1(pk2)                                       # :424-428
    omw = None
    if last_step:                                            # :431-441
        omw = _ijk_cols(jnp.asarray(omga)[ia:ia + n, ia:ia + n, :])
        pe3_om = jnp.concatenate([z1, z1, omw], axis=1)

    pe0_old = pad1(plnw)                                     # :444-446
    peln_w = unpad1(pn2)                                     # :447-449

    if hydrostatic:                                          # :454-459
        pkz_w = ((pk2[:, 2:km + 2] - pk2[:, 1:km + 1])
                 / (akap * (peln_w[:, 1:] - peln_w[:, :-1])))
    else:                                                    # :479-481
        pkz_w = jnp.exp(akap * jnp.log(
            -rdgas / grav * delp_w / dzw * ptw))

    if last_step:                                            # :504-523
        mid = 0.5 * (peln_w[:, :-1] + peln_w[:, 1:])         # :507
        omw = _omega_interp(mid, pe0_old, pe3_om, omw, km)

    # ---------------- :528-549 -- map u, rows j = 1..n+1 ----------------
    nbu = (n + 1) * n
    pe_u_self = _ikj_cols(pe[ipe:ipe + n, :, 1:n + 2])
    pe_u_prev = _ikj_cols(pe[ipe:ipe + n, :, 0:n + 1])
    zu = jnp.zeros((nbu, 1), pe.dtype)
    pe0_u = jnp.concatenate(                                 # :528-538
        [zu, pe_u_self[:, 0:1],
         0.5 * (pe_u_prev[:, 1:km + 1] + pe_u_self[:, 1:km + 1])], axis=1)
    bkh = 0.5 * bk1                                          # :540-545
    pe3_u = jnp.concatenate(
        [zu, ak1[1:km + 2][None, :]
         + bkh[1:km + 2][None, :]
         * (pe_u_prev[:, km] + pe_u_self[:, km])[:, None]], axis=1)
    uw = _ijk_cols(u[ia:ia + n, ia:ia + n + 1, :])
    uw = unpad1(map1_ppm(pe0_u, pad1(uw), pe3_u, km, km, -1, int(kord_mt)))

    # ---------------- :551-569 -- map v, rows j = 1..n ------------------
    npv = n + 1
    nbv = n * npv
    pe_v_left = _ikj_cols(pe[ipe - 1:ipe - 1 + npv, :, 1:1 + n])
    pe_v_self = _ikj_cols(pe[ipe:ipe + npv, :, 1:1 + n])
    zv = jnp.zeros((nbv, 1), pe.dtype)
    pe0_v = jnp.concatenate(                                 # :559-561
        [zv, pe_v_self[:, 0:1],
         0.5 * (pe_v_left[:, 1:km + 1] + pe_v_self[:, 1:km + 1])], axis=1)
    pe3_v = jnp.concatenate(                                 # :555-565
        [zv, jnp.full((nbv, 1), ak1[1], pe.dtype),
         ak1[2:km + 2][None, :]
         + bkh[2:km + 2][None, :]
         * (pe_v_left[:, km] + pe_v_self[:, km])[:, None]], axis=1)
    vw = _ijk_cols(v[ia:ia + npv, ia:ia + n, :])
    vw = unpad1(map1_ppm(pe0_v, pad1(vw), pe3_v, km, km, -1, int(kord_mt)))

    # ---------------- :964-1004 -- close out pt -------------------------
    if last_step:
        if r_vir != 0.0:                                     # :975
            ptw = ptw / (1.0 + r_vir * q_out[int(sphum_index)])
    else:                                                    # :996-1001
        ptw = ptw / pkz_w

    # ---------------- write-back ---------------------------------------
    win = slice(ia, ia + n)
    pt_o = pt.at[win, win, :].set(_cols_ijk(ptw, n, n, km))
    delp_o = delp.at[win, win, :].set(_cols_ijk(delp_w, n, n, km))
    pk_o = pk.at[win, win, :].set(_cols_ijk(pk_w, n, n, km + 1))
    ps_o = ps.at[win, win].set(_cols_ij(ps_w, n, n))
    pkz_o = pkz.at[:, :, :].set(_cols_ijk(pkz_w, n, n, km))
    peln_o = peln.at[:, :, :].set(_cols_ikj(peln_w, n, n, km + 1))
    u_o = u.at[win, ia:ia + n + 1, :].set(_cols_ijk(uw, n + 1, n, km))
    v_o = v.at[ia:ia + npv, win, :].set(_cols_ijk(vw, n, npv, km))
    q_o = tuple(qq.at[win, win, :].set(_cols_ijk(qo, n, n, km))
                for qq, qo in zip(q, q_out))
    omga_o = (jnp.asarray(omga).at[win, win, :].set(
        _cols_ijk(omw, n, n, km)) if last_step else omga)
    if hydrostatic:
        w_o, delz_o = w, delz
    else:
        w_o = jnp.asarray(w).at[win, win, :].set(_cols_ijk(ww, n, n, km))
        delz_o = jnp.asarray(delz).at[:, :, :].set(
            _cols_ijk(dzw, n, n, km))

    # :572-576 + :618-625 -- pe(i,k,j) = pe4(i,j,k-1) for k = 2..km.
    pe2r = pe2.reshape(n, n, km + 2)                         # (j, i, k)
    pe_o = pe.at[ipe:ipe + n, 1:km, 1:1 + n].set(
        jnp.transpose(pe2r[:, :, 2:km + 1], (1, 2, 0)))

    return LagrangianToEulerianOut(
        pe=pe_o, peln=peln_o, pk=pk_o, pkz=pkz_o, delp=delp_o, pt=pt_o,
        u=u_o, v=v_o, ps=ps_o, q=q_o, omga=omga_o, w=w_o, delz=delz_o)


# ---------------------------------------------------------------------------
# jit factories -- ONE policy per routine, built HERE so the production
# entry point and any instrumented test wrapper share it (NH-lane
# doctrine).  No donated buffers anywhere in this lane (grad-path).
# ---------------------------------------------------------------------------

def make_pad1_jit(fn=pad1):
    """No static arguments: pad1 is shape-polymorphic in its operand."""
    return jax.jit(fn)


pad1_jit = make_pad1_jit()


def make_unpad1_jit(fn=unpad1):
    return jax.jit(fn)


unpad1_jit = make_unpad1_jit()


def make_cs_limiters_jit(fn=cs_limiters):
    """``iv`` static: it selects a Python branch and must never be traced."""
    return jax.jit(fn, static_argnums=(2,))


cs_limiters_jit = make_cs_limiters_jit()


def make_cs_profile_jit(fn=cs_profile):
    """``km``/``iv``/``kord`` static (window size + branch selectors)."""
    return jax.jit(fn, static_argnums=(2, 3, 4))


cs_profile_jit = make_cs_profile_jit()


def make_scalar_profile_jit(fn=scalar_profile):
    """``km``/``iv``/``kord``/``qmin`` static -- ``qmin`` gates the whole
    scalar-vs-cs text, so tracing it would be a different routine."""
    return jax.jit(fn, static_argnums=(2, 3, 4, 5))


scalar_profile_jit = make_scalar_profile_jit()


def make_map_scalar_jit(fn=map_scalar):
    """``km``/``kn``/``iv``/``kord``/``q_min`` + ``return_ok`` static."""
    return jax.jit(fn, static_argnums=(3, 4, 5, 6, 7),
                   static_argnames=("return_ok",))


map_scalar_jit = make_map_scalar_jit()


def make_map1_ppm_jit(fn=map1_ppm):
    return jax.jit(fn, static_argnums=(3, 4, 5, 6),
                   static_argnames=("return_ok",))


map1_ppm_jit = make_map1_ppm_jit()


def make_map1_q2_jit(fn=map1_q2):
    return jax.jit(fn, static_argnums=(4, 5, 6, 7, 8),
                   static_argnames=("return_ok",))


map1_q2_jit = make_map1_q2_jit()


_L2E_STATIC = (
    "ptop", "akap", "cp", "r_vir", "km", "n", "ng", "kord_mt", "kord_tm",
    "kord_tr", "sphum_index", "last_step", "hydrostatic", "adiabatic",
    "consv", "kord_wz", "w_limiter", "rdgas", "grav", "fill",
    "do_sat_adj", "do_inline_mp", "do_adiabatic_init")


def make_lagrangian_to_eulerian_jit(fn=lagrangian_to_eulerian):
    """Every deck constant and flag is STATIC; only the fields are traced.

    ``kord_tr`` must be an int or a TUPLE (a list is unhashable and would
    be rejected by jit's static-argument cache).  ``ak``/``bk`` stay
    dynamic -- they are arrays, and making them static would embed a
    whole coordinate table in the cache key.
    """
    return jax.jit(fn, static_argnames=_L2E_STATIC)


lagrangian_to_eulerian_jit = make_lagrangian_to_eulerian_jit()
