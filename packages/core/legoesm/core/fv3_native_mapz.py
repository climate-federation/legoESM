"""``fv_mapz`` column kernels — a loop-faithful port of the pinned oracle.

ORACLE
------
``/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-
symmetryclean/model/fv_mapz.F90`` (Zenodo 8327578, symmetryclean).  Every
routine here is a transcription of the Fortran routine whose name it
carries, and each block cites the oracle line it came from.

WHY THIS FILE EXISTS
--------------------
``Lagrangian_to_Eulerian`` is the last stage between this port and a
full-timestep comparison against the oracle's 1-step restart: that
restart is written AFTER the remap (``fv_dynamics.F90:568`` gates the
call on ``npz > 4``), and ``fv_eta.F90``'s coordinate table starts at
npz=5, so no runnable oracle column can dodge it.

WHAT IS PORTED, AND WHAT IS NOT
-------------------------------
The profile builders and the rezone are ported in FULL — every ``kord``
branch, every ``iv`` branch — not only the pinned lane.  A partial
transcription would silently run a different reconstruction the moment
somebody changed ``kord_tm`` in a namelist.

Not ported, each because it is UNREACHABLE rather than merely unused:

``ppm_profile``/``ppm_limiters``/``steepz``
    reachable only at ``kord <= 7``; all three map routines dispatch with
    ``if (kord > 7)`` (``fv_mapz.F90:1406, 1500, 1708``) and every kord on
    this lane is 9.  :func:`ppm_profile_is_unported` turns that into a
    raise rather than a silent fall-through to the wrong builder.
``mapn_tracer``
    (``nq > 5``, ``fv_mapz.F90:327``) ported 2026-09-30 for the nine-slot
    ice decks -- without its ``fill`` arm, which the lane refuses.
``remap_z``/``rst_remap``/``mappm``
    IC / restart / hybrid-z routines, off the timestep path.
``compute_total_energy``/``pkez``
    gated on ``consv_te > 0.`` and on the ``kord_tm > 0`` arm
    (``fv_mapz.F90:239-240``) respectively.

``scalar_profile`` AND ``cs_profile`` ARE NOT THE SAME FUNCTION
---------------------------------------------------------------
They are one routine with three parameterised deltas, verified by
diffing the two Fortran bodies:

1. every ``qmin`` flattening clause (at abs(kord) 9, 15, and the trailing
   11 branch) exists ONLY in ``scalar_profile``;
2. at abs(kord)==9 ``scalar_profile`` writes A6 as
   ``3.*(2.*qbar - (AL+AR))`` and ``cs_profile`` writes
   ``6.*qbar - 3.*(AL+AR)`` — algebraically equal, NOT equal in floating
   point;
3. ``cs_profile`` additionally supports the ``iv == -3`` vertical-velocity
   LBC.

They are expressed here as one shared body with those three switches, so
the deltas are visible instead of buried in two near-identical copies.

THE 1-BASED CONVENTION IS DELIBERATE
------------------------------------
Every vertical index in this module is FORTRAN's: arrays carry a dummy
slot 0 and are addressed ``k = 1 .. km`` (or ``km+1``), and ``a4``'s
first axis is addressed 1..4.  ``fv_mapz`` is dense with ``gam(i,k+2)``,
``extm(i,k-1)`` and ``do k=3,km-2``; rebasing all of that by hand is the
single most likely way to ship a wrong number here, and the cost is one
unused slot per axis.  :func:`pad1`/:func:`unpad1` convert at the
boundary with the rest of the port, which is 0-based.

PRECISION
---------
fp64 throughout, matching the oracle's ``-fdefault-real-8``.  Unlike the
grid stages, bit-exactness IS the right target here: ``fv_mapz`` is
column-local and touches none of the quad-precision geometry helpers of
``fv_grid_utils.F90:43-49``.
"""
from __future__ import annotations

import operator as _operator

import numpy as np

# fv_mapz.F90:44-45 -- module parameters.
T_MIN = 184.0        # below which scalar_profile applies a stricter constraint
R3 = 1.0 / 3.0
R23 = 2.0 / 3.0
R12 = 1.0 / 12.0


def ppm_profile_is_unported(kord: int) -> None:
    """Refuse a ``kord`` that would have selected ``ppm_profile``.

    ``fv_mapz.F90:1406`` dispatches ``if (kord > 7)`` to the cs/scalar
    profile and otherwise to ``ppm_profile``, which is not ported.  A
    silent fall-through would run a DIFFERENT reconstruction under the
    caller's kord -- the "unknown scheme quietly does something else"
    failure the repo's dispatch-hardening rule exists to stop.
    """
    if int(kord) <= 7:
        raise ValueError(
            f"kord={kord} <= 7 selects ppm_profile (fv_mapz.F90:1406/1500/"
            f"1708), which is NOT ported -- the pinned duo lane uses kord 9 "
            f"everywhere. Port ppm_profile/ppm_limiters/steepz before using "
            f"a low-order remap; do not fall through to cs_profile.")


def pad1(a: np.ndarray) -> np.ndarray:
    """0-based ``(im, nk)`` -> 1-based ``(im, nk+1)`` with a dummy k=0."""
    a = np.asarray(a, dtype=np.float64)
    out = np.zeros((a.shape[0], a.shape[1] + 1), dtype=np.float64)
    out[:, 1:] = a
    return out


def unpad1(a: np.ndarray) -> np.ndarray:
    """1-based ``(im, nk+1)`` -> 0-based ``(im, nk)``."""
    return np.ascontiguousarray(a[:, 1:])


def _a6_3x2(q1, a2, a3):
    """``3.*(2.*qbar - (AL+AR))`` -- the grouping used almost everywhere."""
    return 3.0 * (2.0 * q1 - (a2 + a3))


def _a6_6m3(q1, a2, a3):
    """``6.*qbar - 3.*(AL+AR)`` -- cs_profile kord 9, and kord 12 in both.

    Algebraically identical to :func:`_a6_3x2`, and NOT identical in
    floating point.  Which one a branch uses is copied from the oracle
    line by line; do not "simplify" one into the other.
    """
    return 6.0 * q1 - 3.0 * (a2 + a3)


# ---------------------------------------------------------------------------
# cs_limiters -- fv_mapz.F90:2693-2768
# ---------------------------------------------------------------------------

def cs_limiters(a4: np.ndarray, extm: np.ndarray, iv: int) -> None:
    """In-place PPM limiter on ONE layer. ``a4`` is (5, im); 1..4 used.

    Transcribes ``cs_limiters(im, extm, a4, iv)``, which the Fortran
    calls by sequence association on a single k-slice -- hence the slice
    argument rather than an array plus a k.

    The iv=1 arm ignores ``extm`` entirely and RECOMPUTES the extremum
    test from ``(qbar-AL)*(qbar-AR) >= 0``; the ``else`` arm (iv=2, and
    anything not 0/1) uses the PRECOMPUTED ``extm``.  That asymmetry is
    the oracle's and is load-bearing.
    """
    # Snapshot: the Fortran branches are disjoint per point, but writing
    # a4[2] before reading it for a4[3] would alias through the view.
    a1 = np.array(a4[1], copy=True)
    a2 = np.array(a4[2], copy=True)
    a3 = np.array(a4[3], copy=True)
    a6 = np.array(a4[4], copy=True)

    if iv == 0:
        # fv_mapz.F90:2702-2727 -- positive-definite constraint.
        with np.errstate(divide="ignore", invalid="ignore"):
            parabola_min = a1 + 0.25 * (a3 - a2) ** 2 / a6 + a6 * R12
            neg = (a1 > 0.0) & (np.abs(a3 - a2) < -a6) & (parabola_min < 0.0)
        flat = (a1 <= 0.0) | (neg & (a1 < a3) & (a1 < a2))
        up = neg & ~flat & (a3 > a2)
        dn = neg & ~flat & ~(a3 > a2)
        a4[2] = np.where(flat, a1, np.where(dn, a3 - 3.0 * (a3 - a1), a2))
        a4[3] = np.where(flat, a1, np.where(up, a2 - 3.0 * (a2 - a1), a3))
        a4[4] = np.where(flat, 0.0,
                         np.where(up, 3.0 * (a2 - a1),
                                  np.where(dn, 3.0 * (a3 - a1), a6)))
        return

    if iv == 1:
        flat = (a1 - a2) * (a1 - a3) >= 0.0      # :2730
    else:
        flat = np.asarray(extm, dtype=bool)      # :2750

    da1 = a3 - a2
    da2 = da1 ** 2
    a6da = a6 * da1
    lo = (~flat) & (a6da < -da2)
    hi = (~flat) & (a6da > da2)
    a4[2] = np.where(flat, a1, np.where(hi, a3 - 3.0 * (a3 - a1), a2))
    a4[3] = np.where(flat, a1, np.where(lo, a2 - 3.0 * (a2 - a1), a3))
    a4[4] = np.where(flat, 0.0,
                     np.where(lo, 3.0 * (a2 - a1),
                              np.where(hi, 3.0 * (a3 - a1), a6)))


# ---------------------------------------------------------------------------
# the tridiagonal edge reconstructions
# ---------------------------------------------------------------------------

def _edge_solve_iv_m2(a4: np.ndarray, delp: np.ndarray, km: int,
                      qs: np.ndarray) -> np.ndarray:
    """iv == -2 (fv_mapz.F90:1877-1900, identical at :2287-2310).

    Vertical-velocity reconstruction with a bottom BC ``qs``.
    """
    im = a4.shape[1]
    q = np.zeros((im, km + 2), dtype=np.float64)
    gam = np.zeros((im, km + 2), dtype=np.float64)
    gam[:, 2] = 0.5
    q[:, 1] = 1.5 * a4[1, :, 1]
    for k in range(2, km):                       # do k=2,km-1
        grat = delp[:, k - 1] / delp[:, k]
        bet = 2.0 + grat + grat - gam[:, k]
        q[:, k] = (3.0 * (a4[1, :, k - 1] + a4[1, :, k]) - q[:, k - 1]) / bet
        gam[:, k + 1] = grat / bet
    grat = delp[:, km - 1] / delp[:, km]
    q[:, km] = (3.0 * (a4[1, :, km - 1] + a4[1, :, km]) - grat * qs
                - q[:, km - 1]) / (2.0 + grat + grat - gam[:, km])
    q[:, km + 1] = qs
    for k in range(km - 1, 0, -1):               # do k=km-1,1,-1
        q[:, k] = q[:, k] - gam[:, k + 1] * q[:, k + 1]
    return q


def _edge_solve_default(a4: np.ndarray, delp: np.ndarray, km: int,
                        ) -> np.ndarray:
    """The ``else`` branch (fv_mapz.F90:1901-1928, and :2341-2362).

    The bottom closure at ``:1918-1922`` reuses ``d4`` LEFT OVER from the
    final iteration of ``do k=2,km``.  That is the Fortran's own
    behaviour, not a transcription slip.

    ``gam`` is local: the caller immediately rebuilds it as ``dq`` in
    :func:`_large_scale_constraints`, and ``gam(i,1)`` (the only slot the
    rebuild leaves stale) is never read again on any lane that gets here.
    """
    im = a4.shape[1]
    q = np.zeros((im, km + 2), dtype=np.float64)
    gam = np.zeros((im, km + 2), dtype=np.float64)

    grat = delp[:, 2] / delp[:, 1]
    bet = grat * (grat + 0.5)
    q[:, 1] = ((grat + grat) * (grat + 1.0) * a4[1, :, 1] + a4[1, :, 2]) / bet
    gam[:, 1] = (1.0 + grat * (grat + 1.5)) / bet

    d4 = np.zeros(im, dtype=np.float64)
    for k in range(2, km + 1):                   # do k=2,km
        d4 = delp[:, k - 1] / delp[:, k]
        bet = 2.0 + d4 + d4 - gam[:, k - 1]
        q[:, k] = (3.0 * (a4[1, :, k - 1] + d4 * a4[1, :, k])
                   - q[:, k - 1]) / bet
        gam[:, k] = d4 / bet

    a_bot = 1.0 + d4 * (d4 + 1.5)                # d4 holds the k=km value
    q[:, km + 1] = (2.0 * d4 * (d4 + 1.0) * a4[1, :, km] + a4[1, :, km - 1]
                    - a_bot * q[:, km]) / (d4 * (d4 + 0.5)
                                           - a_bot * gam[:, km])

    for k in range(km, 0, -1):                   # do k=km,1,-1
        q[:, k] = q[:, k] - gam[:, k] * q[:, k + 1]
    return q


def _refuse_iv_m3(km: int) -> None:
    """cs_profile's iv == -3 LBC branch reads UNINITIALISED memory.

    RETRACTION: an earlier version of this module implemented ``iv=-3``
    and claimed to be "reproducing the oracle's bug" at
    ``fv_mapz.F90:2337``, on the reading that ``k`` there is the last
    EXECUTED value of ``do k=2,km-1``, i.e. km-1. That is wrong. After a
    Fortran DO loop terminates normally the variable holds
    ``initial + n*step``, i.e. **km**, so ``:2337`` is the perfectly
    ordinary ``q(km) = (3*(a4(1,km-1) + d4*a4(1,km)) - grat*qs -
    q(km-1))/bet``. There was no bug to reproduce, and the port was
    computing the wrong column while documenting the opposite.

    The branch is refused instead of fixed because of a SECOND problem
    that is genuinely unreproducible: the ``do k=2,km-1`` loop assigns
    ``gam(i,2..km-1)`` only, and the back-substitution at ``:2358-2360``
    then reads ``gam(i,km)``, which was never written. ``gam`` is a local
    automatic array, so that read is undefined -- and a port that
    zero-fills it turns undefined oracle behaviour into a specific,
    confidently-wrong number.

    Unreachable on the pinned lane: it needs ``kord_wz < 0`` AND
    ``hydrostatic = .false.`` (``fv_mapz.F90:345-350``), and this lane is
    hydrostatic. Capture the oracle's actual bytes before implementing.
    """
    raise NotImplementedError(
        f"cs_profile(iv=-3, km={km}): fv_mapz.F90:2320-2339 leaves "
        f"gam(i,km) UNASSIGNED and :2358 reads it, so the oracle's answer "
        f"here depends on uninitialised stack. Refusing to invent one. "
        f"The branch needs kord_wz<0; the pinned NH deck runs kord_wz=9 "
        f"(iv=-2), so this arm stays dead there.")


def _large_scale_constraints(q: np.ndarray, a4: np.ndarray, km: int,
                             iv: int) -> np.ndarray:
    """fv_mapz.F90:1948-1984 (identical at :2382-2418). Returns dq.

    ``q(i,1)`` and ``q(i,km+1)`` are deliberately never clamped.
    """
    im = a4.shape[1]
    q[:, 2] = np.minimum(q[:, 2], np.maximum(a4[1, :, 1], a4[1, :, 2]))
    q[:, 2] = np.maximum(q[:, 2], np.minimum(a4[1, :, 1], a4[1, :, 2]))

    gam = np.zeros((im, km + 2), dtype=np.float64)
    for k in range(2, km + 1):                   # :1954-1958 -- gam := dq
        gam[:, k] = a4[1, :, k] - a4[1, :, k - 1]

    for k in range(3, km):                       # :1961 do k=3,km-1
        hi = np.maximum(a4[1, :, k - 1], a4[1, :, k])
        lo = np.minimum(a4[1, :, k - 1], a4[1, :, k])
        smooth = gam[:, k - 1] * gam[:, k + 1] > 0.0
        local_max = gam[:, k - 1] > 0.0
        at_min = np.minimum(q[:, k], hi)
        if iv == 0:                              # :1974
            at_min = np.maximum(0.0, at_min)
        q[:, k] = np.where(smooth,
                           np.minimum(np.maximum(q[:, k], lo), hi),
                           np.where(local_max, np.maximum(q[:, k], lo),
                                    at_min))

    hi = np.maximum(a4[1, :, km - 1], a4[1, :, km])
    lo = np.minimum(a4[1, :, km - 1], a4[1, :, km])
    q[:, km] = np.minimum(np.maximum(q[:, km], lo), hi)
    return gam


def _edges_and_extrema(a4: np.ndarray, q: np.ndarray, gam: np.ndarray,
                       km: int, kord: int):
    """fv_mapz.F90:1986-2012 (identical at :2420-2446).

    ``ext5``/``ext6`` are built ONLY when ``abs(kord) > 9``.  At
    abs(kord)==9 the Fortran leaves them undefined AND does not pre-fill
    ``a4(4,:,:)``; this returns ``None`` for them so a wrong read is a
    ``TypeError`` rather than a plausible number.
    """
    im = a4.shape[1]
    for k in range(1, km + 1):
        a4[2, :, k] = q[:, k]
        a4[3, :, k] = q[:, k + 1]

    extm = np.zeros((im, km + 2), dtype=bool)
    have_ext56 = abs(kord) > 9
    ext5 = np.zeros((im, km + 2), dtype=bool) if have_ext56 else None
    ext6 = np.zeros((im, km + 2), dtype=bool) if have_ext56 else None
    for k in range(1, km + 1):
        if k == 1 or k == km:
            extm[:, k] = ((a4[2, :, k] - a4[1, :, k])
                          * (a4[3, :, k] - a4[1, :, k])) > 0.0
        else:
            extm[:, k] = gam[:, k] * gam[:, k + 1] < 0.0
        if have_ext56:
            x0 = 2.0 * a4[1, :, k] - (a4[2, :, k] + a4[3, :, k])
            x1 = np.abs(a4[2, :, k] - a4[3, :, k])
            a4[4, :, k] = 3.0 * x0
            ext5[:, k] = np.abs(x0) > x1
            ext6[:, k] = np.abs(a4[4, :, k]) > x1
    return extm, ext5, ext6


def _huynh_edges(a4: np.ndarray, gam: np.ndarray, k: int):
    """Huynh's 2nd constraint on the two edges (fv_mapz.F90:2056-2064).

    Reaches ``gam(k-1) .. gam(k+2)``, which is why every loop that calls
    it runs only ``k = 3 .. km-2``.
    """
    q1 = a4[1, :, k]
    pmp_1 = q1 - 2.0 * gam[:, k + 1]
    lac_1 = pmp_1 + 1.5 * gam[:, k + 2]
    a2 = np.minimum(np.maximum(a4[2, :, k],
                               np.minimum(np.minimum(q1, pmp_1), lac_1)),
                    np.maximum(np.maximum(q1, pmp_1), lac_1))
    pmp_2 = q1 + 2.0 * gam[:, k]
    lac_2 = pmp_2 - 1.5 * gam[:, k - 1]
    a3 = np.minimum(np.maximum(a4[3, :, k],
                               np.minimum(np.minimum(q1, pmp_2), lac_2)),
                    np.maximum(np.maximum(q1, pmp_2), lac_2))
    return a2, a3


def _top_two_layers(a4: np.ndarray, extm: np.ndarray, iv: int) -> None:
    """fv_mapz.F90:2020-2047 (identical at :2454-2481)."""
    if iv == 0:
        a4[2, :, 1] = np.maximum(0.0, a4[2, :, 1])
    elif iv == -1:
        a4[2, :, 1] = np.where(a4[2, :, 1] * a4[1, :, 1] <= 0.0,
                               0.0, a4[2, :, 1])
    elif iv == 2:
        a4[2, :, 1] = a4[1, :, 1]
        a4[3, :, 1] = a4[1, :, 1]
        a4[4, :, 1] = 0.0

    if iv != 2:
        a4[4, :, 1] = _a6_3x2(a4[1, :, 1], a4[2, :, 1], a4[3, :, 1])
        cs_limiters(a4[:, :, 1], extm[:, 1], 1)

    a4[4, :, 2] = _a6_3x2(a4[1, :, 2], a4[2, :, 2], a4[3, :, 2])
    cs_limiters(a4[:, :, 2], extm[:, 2], 2)


def _bottom_two_layers(a4: np.ndarray, extm: np.ndarray, km: int,
                       iv: int) -> None:
    """fv_mapz.F90:2244-2260 (identical at :2672-2688)."""
    if iv == 0:
        a4[3, :, km] = np.maximum(0.0, a4[3, :, km])
    elif iv == -1:
        a4[3, :, km] = np.where(a4[3, :, km] * a4[1, :, km] <= 0.0,
                                0.0, a4[3, :, km])
    for k in (km - 1, km):
        a4[4, :, k] = _a6_3x2(a4[1, :, k], a4[2, :, k], a4[3, :, k])
        cs_limiters(a4[:, :, k], extm[:, k], 2 if k == km - 1 else 1)


def _require_ext56(ext5, kord: int):
    if ext5 is None:
        raise AssertionError(
            f"kord={kord} needs ext5/ext6, which fv_mapz.F90:2003 builds "
            f"only when abs(kord) > 9. Unreachable by construction; if you "
            f"see this the dispatch above it is wrong.")


def _interior_layer(a4: np.ndarray, gam: np.ndarray, extm: np.ndarray,
                    ext5, ext6, k: int, kord: int,
                    qmin: float | None) -> None:
    """One ``k`` of the ``do k=3,km-2`` interior loop, in place.

    Shared by ``scalar_profile`` and ``cs_profile``.  ``qmin is None``
    selects the ``cs_profile`` text, which lacks every qmin clause and
    uses the ``6.*qbar - 3.*(AL+AR)`` A6 grouping at abs(kord)==9; a float
    ``qmin`` selects the ``scalar_profile`` text.  Those are the only
    differences between the two Fortran bodies in this loop.
    """
    scalar = qmin is not None
    q1 = a4[1, :, k]
    ak = abs(kord)

    if ak < 9:                                   # :2053 / :2487
        a2, a3 = _huynh_edges(a4, gam, k)
        a4[2, :, k], a4[3, :, k] = a2, a3
        a4[4, :, k] = _a6_3x2(q1, a2, a3)
        return

    if ak == 9:                                  # :2069 / :2503
        a6f = _a6_3x2 if scalar else _a6_6m3
        flat = ((extm[:, k] & extm[:, k - 1])
                | (extm[:, k] & extm[:, k + 1]))
        if scalar:                               # :2081 -- scalar only
            flat = flat | (extm[:, k] & (q1 < qmin))
        a6 = a6f(q1, a4[2, :, k], a4[3, :, k])
        fix = (~flat) & (np.abs(a6) > np.abs(a4[2, :, k] - a4[3, :, k]))
        a2h, a3h = _huynh_edges(a4, gam, k)
        a2 = np.where(fix, a2h, a4[2, :, k])
        a3 = np.where(fix, a3h, a4[3, :, k])
        a6 = np.where(fix, a6f(q1, a2, a3), a6)
        a4[2, :, k] = np.where(flat, q1, a2)
        a4[3, :, k] = np.where(flat, q1, a3)
        a4[4, :, k] = np.where(flat, 0.0, a6)
        return

    _require_ext56(ext5, kord)
    near5 = ext5[:, k - 1] | ext5[:, k + 1]
    near6 = ext6[:, k - 1] | ext6[:, k + 1]

    if ak == 10:                                 # :2102 / :2531
        a2h, a3h = _huynh_edges(a4, gam, k)
        flat = ext5[:, k] & near5
        clamp = ((ext5[:, k] & (~near5) & near6)
                 | ((~ext5[:, k]) & ext6[:, k] & near5))
        a4[2, :, k] = np.where(flat, q1, np.where(clamp, a2h, a4[2, :, k]))
        a4[3, :, k] = np.where(flat, q1, np.where(clamp, a3h, a4[3, :, k]))
        a4[4, :, k] = _a6_3x2(q1, a4[2, :, k], a4[3, :, k])
    elif ak == 12:                               # :2134 / :2563
        a6 = _a6_6m3(q1, a4[2, :, k], a4[3, :, k])
        fix = (~extm[:, k]) & (np.abs(a6)
                               > np.abs(a4[2, :, k] - a4[3, :, k]))
        a2h, a3h = _huynh_edges(a4, gam, k)
        a2 = np.where(fix, a2h, a4[2, :, k])
        a3 = np.where(fix, a3h, a4[3, :, k])
        a6 = np.where(fix, _a6_6m3(q1, a2, a3), a6)
        a4[2, :, k] = np.where(extm[:, k], q1, a2)
        a4[3, :, k] = np.where(extm[:, k], q1, a3)
        a4[4, :, k] = np.where(extm[:, k], 0.0, a6)
    elif ak == 13:                               # :2156 / :2586
        flat = ext6[:, k] & ext6[:, k - 1] & ext6[:, k + 1]
        a4[2, :, k] = np.where(flat, q1, a4[2, :, k])
        a4[3, :, k] = np.where(flat, q1, a4[3, :, k])
        a4[4, :, k] = _a6_3x2(q1, a4[2, :, k], a4[3, :, k])
    elif ak == 14:                               # :2169 / :2599
        a4[4, :, k] = _a6_3x2(q1, a4[2, :, k], a4[3, :, k])
    elif ak == 15:                               # :2175 / :2605
        # THE TWO ROUTINES NEST THIS DIFFERENTLY -- not just a qmin delta.
        # scalar_profile (:2176-2195) is a FLAT elseif chain, so `elseif
        # (ext6)` is reachable when ext5(k) is set but no neighbour is.
        # cs_profile (:2607-2623) wraps the ext5 tests inside `if (ext5(k))`,
        # so ext5(k) SWALLOWS the ext6 arm and that point is left untouched.
        flat = ext5[:, k] & near5
        if scalar:                               # :2183 -- scalar only
            flat = flat | (ext5[:, k] & (q1 < qmin))
            clamp = (~flat) & ext6[:, k]
        else:
            clamp = (~ext5[:, k]) & ext6[:, k]
        a2h, a3h = _huynh_edges(a4, gam, k)
        a4[2, :, k] = np.where(flat, q1, np.where(clamp, a2h, a4[2, :, k]))
        a4[3, :, k] = np.where(flat, q1, np.where(clamp, a3h, a4[3, :, k]))
        a4[4, :, k] = _a6_3x2(q1, a4[2, :, k], a4[3, :, k])
    elif ak == 16:                               # :2200 / :2628
        flat = ext5[:, k] & near5
        clamp = ext5[:, k] & (~near5) & near6
        a2h, a3h = _huynh_edges(a4, gam, k)
        a4[2, :, k] = np.where(flat, q1, np.where(clamp, a2h, a4[2, :, k]))
        a4[3, :, k] = np.where(flat, q1, np.where(clamp, a3h, a4[3, :, k]))
        a4[4, :, k] = _a6_3x2(q1, a4[2, :, k], a4[3, :, k])
    else:                                        # :2223 / :2651 -- kord 11
        noisy = ext5[:, k] & near5
        if scalar:                               # :2225 -- scalar only
            noisy = ext5[:, k] & (near5 | (q1 < qmin))
        a4[2, :, k] = np.where(noisy, q1, a4[2, :, k])
        a4[3, :, k] = np.where(noisy, q1, a4[3, :, k])
        a4[4, :, k] = np.where(noisy, 0.0,
                               _a6_3x2(q1, a4[2, :, k], a4[3, :, k]))


def _profile(a4: np.ndarray, delp: np.ndarray, km: int, iv: int, kord: int,
             qmin: float | None, qs: np.ndarray | None) -> None:
    """Shared body of ``scalar_profile`` (qmin set) and ``cs_profile``."""
    ppm_profile_is_unported(kord)
    if a4.shape[0] != 5 or a4.shape[2] != km + 1:
        raise ValueError(
            f"a4 must be 1-based (5, im, km+1) for km={km}, got {a4.shape}")
    if km < 4:
        raise ValueError(
            f"km={km}: the profile routines index a4(1,i,km-1) and run "
            f"cs_limiters on k=1,2,km-1,km, which overlap below km=4. The "
            f"oracle's own coordinate table (fv_eta.F90) starts at km=5.")

    if iv == -2:
        if qs is None:
            raise ValueError("iv=-2 needs the bottom BC qs")
        q = _edge_solve_iv_m2(a4, delp, km, np.asarray(qs, dtype=np.float64))
    elif iv == -3 and qmin is None:
        # cs_profile has a dedicated -3 arm; scalar_profile does NOT
        # (fv_mapz.F90:1877 tests only iv == -2), so for scalar_profile
        # iv=-3 falls through to the ordinary else below -- which is why
        # this condition checks qmin rather than iv alone.
        _refuse_iv_m3(km)
    else:
        q = _edge_solve_default(a4, delp, km)

    if abs(kord) > 16:                           # :1932 / :2365
        for k in range(1, km + 1):
            a4[2, :, k] = q[:, k]
            a4[3, :, k] = q[:, k + 1]
            a4[4, :, k] = _a6_3x2(a4[1, :, k], a4[2, :, k], a4[3, :, k])
        return

    gam = _large_scale_constraints(q, a4, km, iv)
    extm, ext5, ext6 = _edges_and_extrema(a4, q, gam, km, kord)
    _top_two_layers(a4, extm, iv)

    for k in range(3, km - 1):                   # do k=3,km-2
        _interior_layer(a4, gam, extm, ext5, ext6, k, kord, qmin)
        if iv == 0:                              # :2237 / :2665
            cs_limiters(a4[:, :, k], extm[:, k], 0)

    _bottom_two_layers(a4, extm, km, iv)


def scalar_profile(a4: np.ndarray, delp: np.ndarray, km: int, iv: int,
                   kord: int, qmin: float,
                   qs: np.ndarray | None = None) -> None:
    """fv_mapz.F90:1855-2262. In-place; ``a4`` is 1-based (5, im, km+1)."""
    if qmin is None:
        raise ValueError("scalar_profile requires qmin (fv_mapz.F90:1867)")
    _profile(a4, delp, km, iv, kord, float(qmin), qs)


def cs_profile(a4: np.ndarray, delp: np.ndarray, km: int, iv: int,
               kord: int, qs: np.ndarray | None = None) -> None:
    """fv_mapz.F90:2265-2690. In-place; ``a4`` is 1-based (5, im, km+1)."""
    _profile(a4, delp, km, iv, kord, None, qs)


# ---------------------------------------------------------------------------
# the rezone -- shared body of map_scalar / map1_ppm / map1_q2
# ---------------------------------------------------------------------------

def _build_q4(q1: np.ndarray, pe1: np.ndarray, km: int):
    """fv_mapz.F90:1398-1403 -- the a4/dp1 setup shared by all three.

    The whole column is copied into ``q4(1,:,:)`` BEFORE anything writes
    the output, which is what makes the oracle's ``q1``/``q2`` aliasing
    (``pt`` is passed as both) benign.  Returning a fresh array keeps that.
    """
    im = q1.shape[0]
    q4 = np.zeros((5, im, km + 1), dtype=np.float64)
    dp1 = np.zeros((im, km + 1), dtype=np.float64)
    for k in range(1, km + 1):
        dp1[:, k] = pe1[:, k + 1] - pe1[:, k]
        q4[1, :, k] = q1[:, k]
    return q4, dp1


def _rezone(q4: np.ndarray, dp1: np.ndarray, pe1: np.ndarray,
            pe2: np.ndarray, km: int, kn: int,
            dp2: np.ndarray | None = None, *,
            mapn: bool = False) -> np.ndarray:
    """fv_mapz.F90:1412-1451, and :1715-1754 with ``dp2`` supplied;
    ``mapn=True`` is :1795-1836 (``mapn_tracer``), the SAME integration
    with the products associated as that routine writes them --
    ``fac1 = 0.5*(pr+pl)`` FIRST, then ``(a4+a3-a2)*fac1`` -- which is a
    rounding-level difference from ``map1_q2``'s ``0.5*(...)*(pr+pl)``.

    ``k0`` is a monotone search hint carried ACROSS the target k-loop; it
    is NOT reset per k.  The two cases are written differently on purpose:
    a target cell wholly inside one source cell gets the closed-form
    parabola mean with NO division, while a spanning cell accumulates
    ``qsum`` and divides once.  Reproducing that split is what makes
    bit-exactness reachable.

    ``dp2`` is the ONLY difference between ``map_scalar``/``map1_ppm``
    (which divide by ``pe2(k+1)-pe2(k)``) and ``map1_q2`` (which divides
    by the driver's precomputed ``dp2``).  Those are equal in exact
    arithmetic and not in floating point, so the caller must pass the one
    its oracle counterpart uses.
    """
    im = q4.shape[1]
    q2 = np.zeros((im, kn + 1), dtype=np.float64)
    for i in range(im):
        k0 = 1
        for k in range(1, kn + 1):
            qsum = None
            inside = False
            for ell in range(k0, km + 1):
                if not (pe2[i, k] >= pe1[i, ell]
                        and pe2[i, k] <= pe1[i, ell + 1]):
                    continue
                pl = (pe2[i, k] - pe1[i, ell]) / dp1[i, ell]
                if pe2[i, k + 1] <= pe1[i, ell + 1]:
                    # entire target cell inside source cell ell
                    pr = (pe2[i, k + 1] - pe1[i, ell]) / dp1[i, ell]
                    if mapn:                                 # :1808-1813
                        fac1 = pr + pl
                        fac2 = R3 * (pr * fac1 + pl * pl)
                        fac1 = 0.5 * fac1
                        q2[i, k] = (q4[2, i, ell]
                                    + (q4[4, i, ell] + q4[3, i, ell]
                                       - q4[2, i, ell]) * fac1
                                    - q4[4, i, ell] * fac2)
                    else:
                        q2[i, k] = (q4[2, i, ell]
                                    + 0.5 * (q4[4, i, ell] + q4[3, i, ell]
                                             - q4[2, i, ell]) * (pr + pl)
                                    - q4[4, i, ell] * R3
                                    * (pr * (pr + pl) + pl ** 2))
                    k0 = ell
                    inside = True
                    break
                if mapn:                                     # :1818-1824
                    dp = pe1[i, ell + 1] - pe2[i, k]
                    fac1 = 1.0 + pl
                    fac2 = R3 * (1.0 + pl * fac1)
                    fac1 = 0.5 * fac1
                    qsum = dp * (q4[2, i, ell]
                                 + (q4[4, i, ell] + q4[3, i, ell]
                                    - q4[2, i, ell]) * fac1
                                 - q4[4, i, ell] * fac2)
                else:
                    qsum = ((pe1[i, ell + 1] - pe2[i, k])
                            * (q4[2, i, ell]
                               + 0.5 * (q4[4, i, ell] + q4[3, i, ell]
                                        - q4[2, i, ell]) * (1.0 + pl)
                               - q4[4, i, ell] * (R3 * (1.0 + pl * (1.0 + pl)))))
                for m in range(ell + 1, km + 1):
                    if pe2[i, k + 1] > pe1[i, m + 1]:
                        qsum = qsum + dp1[i, m] * q4[1, i, m]
                    else:
                        dp = pe2[i, k + 1] - pe1[i, m]
                        esl = dp / dp1[i, m]
                        # mapn_tracer :1832-1836 writes fac1 = 0.5*esl,
                        # fac2 = 1 - r23*esl, dp*(a2 + fac1*(a3-a2+a4*fac2))
                        # -- the same association as this expression
                        qsum = qsum + dp * (
                            q4[2, i, m] + 0.5 * esl
                            * (q4[3, i, m] - q4[2, i, m]
                               + q4[4, i, m] * (1.0 - R23 * esl)))
                        k0 = m
                        break
                break
            if inside:
                continue
            if qsum is None:
                # The Fortran falls through to label 123 with qsum
                # UNDEFINED -- a stale value from the previous k. That can
                # only happen when pe2(i,k) lies outside every source
                # interval, which the pn2 construction forbids (:297-298
                # COPIES the endpoints rather than recomputing them). Make
                # it loud instead of inheriting garbage.
                raise FloatingPointError(
                    f"rezone: target edge pe2[{i},{k}]={pe2[i, k]!r} is "
                    f"bracketed by no source interval within "
                    f"[{pe1[i, k0]!r}, {pe1[i, km + 1]!r}]. The oracle would "
                    f"read an uninitialised qsum here; refusing to guess. "
                    f"Check that pe2's endpoints were COPIED from pe1.")
            denom = (pe2[i, k + 1] - pe2[i, k]) if dp2 is None else dp2[i, k]
            q2[i, k] = qsum / denom
    return q2


def map_scalar(pe1: np.ndarray, q1: np.ndarray, pe2: np.ndarray, km: int,
               kn: int, iv: int, kord: int, q_min: float,
               qs: np.ndarray | None = None) -> np.ndarray:
    """fv_mapz.F90:1361-1453 -- selects :func:`scalar_profile`.

    All arrays 1-based: ``pe1`` (im, km+2), ``q1`` (im, km+1), ``pe2``
    (im, kn+2).  On the pinned lane the coordinate is ln(p), so the
    conserved integral is INT(T dlnp) -- NOT INT(theta dp).
    """
    q4, dp1 = _build_q4(q1, pe1, km)
    scalar_profile(q4, dp1, km, iv, kord, q_min, qs=qs)
    return _rezone(q4, dp1, pe1, pe2, km, kn)


def map1_ppm(pe1: np.ndarray, q1: np.ndarray, pe2: np.ndarray, km: int,
             kn: int, iv: int, kord: int,
             qs: np.ndarray | None = None) -> np.ndarray:
    """fv_mapz.F90:1456-1547 -- selects :func:`cs_profile`, coordinate p."""
    q4, dp1 = _build_q4(q1, pe1, km)
    cs_profile(q4, dp1, km, iv, kord, qs=qs)
    return _rezone(q4, dp1, pe1, pe2, km, kn)


def map1_q2(pe1: np.ndarray, q1: np.ndarray, pe2: np.ndarray,
            dp2: np.ndarray, km: int, kn: int, iv: int, kord: int,
            q_min: float, qs: np.ndarray | None = None) -> np.ndarray:
    """fv_mapz.F90:1666-1756 -- the tracer remap.

    Same builder as :func:`map_scalar`, but the spanning-cell branch
    divides by the driver's ``dp2`` (1-based (im, kn+1)) rather than by
    ``pe2(k+1)-pe2(k)``.
    """
    q4, dp1 = _build_q4(q1, pe1, km)
    scalar_profile(q4, dp1, km, iv, kord, q_min, qs=qs)
    return _rezone(q4, dp1, pe1, pe2, km, kn, dp2=dp2)


def mapn_tracer(pe1: np.ndarray, q1: list, pe2: np.ndarray,
                dp2: np.ndarray, km: int, kords: list,
                q_min: float) -> list:
    """fv_mapz.F90:1758-1848 -- the nq > 5 tracer remap (:327).

    Per tracer: ``scalar_profile`` (ALWAYS -- unlike ``map1_q2``, which
    takes ``ppm_profile`` for ``kord <= 7``, :1698-1702) with ``iv=0``
    and its own ``kord(iq)``, then the integration of ``_rezone`` with
    ``mapn=True`` (that routine's product association).  The tracers
    share nothing but ``pe1``/``pe2``/``dp2`` (``k0`` is per column and
    identical across tracers because the edges are), so this is the
    per-tracer routine applied in turn.  ``fill`` (fillz on all nq,
    :1840) is NOT ported: the lane refuses it (``_refuse_unported_lane``).
    """
    out = []
    if len(q1) != len(kords):
        raise ValueError(f"mapn_tracer: {len(q1)} tracers but {len(kords)} "
                         "kord entries (zip would silently drop tracers)")
    for qt, kord in zip(q1, kords):
        q4, dp1 = _build_q4(qt, pe1, km)
        scalar_profile(q4, dp1, km, 0, kord, q_min)
        out.append(_rezone(q4, dp1, pe1, pe2, km, km, dp2=dp2, mapn=True))
    return out


# ---------------------------------------------------------------------------
# Lagrangian_to_Eulerian -- fv_mapz.F90:62-1080
# ---------------------------------------------------------------------------

CONSV_MIN = 0.001                                # fv_mapz.F90:43
# fv_mapz.F90:51-52 -- MODULE PARAMETERS of the w_limiter, compile-time
# fixed.  The deck's namelist W_MAX=75 is flagstruct%w_max, a DIFFERENT
# variable this routine never reads.
W_MAX_MAPZ = 90.0
W_MIN_MAPZ = -60.0


def _refuse_unported_lane(*, hydrostatic: bool, adiabatic: bool, consv: float,
                          fill: bool, kord_tm: int, do_sat_adj: bool,
                          do_inline_mp: bool, do_adiabatic_init: bool,
                          nq: int, last_step: bool,
                          defer_close: bool = False) -> None:
    """Reject every configuration whose ``fv_mapz`` branch is not ported.

    Each of these is a REAL oracle branch that this port does not carry.
    Refusing loudly is the point: silently taking the ported branch under
    a different flag would run different physics and still return
    plausible numbers.
    """
    # hydrostatic=False is ported: fv_mapz.F90:252-258 (delz -> specific
    # volume on the OLD delp), :345-419 (w/delz remap + w_limiter) and
    # :460-493 (the NH pkz, non-moist kord_tm<0 arm).  Its argument
    # requirements are enforced in lagrangian_to_eulerian itself.
    # The energy fixer is inside `if (last_step .and. ...)` at :628, so a
    # non-last_step call never reaches it whatever consv says. Refusing it
    # there would be stricter than the oracle, not safer.
    if (last_step and (consv > CONSV_MIN or consv < -CONSV_MIN)
            and not defer_close):
        # THE FIXER IS PORTED, BUT NOT AS A ONE-CALL OPERATION. Its
        # dtmp is a global reduction over all six faces and this
        # function sees one face, so it can only be run through the
        # two-phase protocol: defer_close=True here, then the caller
        # reduces and calls close_out_pt. A caller that passes consv
        # WITHOUT deferring would get the un-fixed answer and no error,
        # which is the failure this guard now exists to prevent -- it no
        # longer means "not ported", it means "not like that".
        raise NotImplementedError(
            f"consv={consv} at last_step without defer_close: the "
            f"total-energy fixer needs a GLOBAL sum over all six faces "
            f"(fv_mapz.F90:708, g_sum) and this call sees one. Use "
            f"defer_close=True, reduce with "
            f"fv3_native_dynamics.energy_fixer_dtmp, then apply "
            f"close_out_pt -- which is what fv_dynamics_step does. "
            f"|consv| <= {CONSV_MIN} keeps the fixer off entirely.")
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


def close_out_pt(pt, pkz, q, *, sphum_index, r_vir, dtmp, cp,
                 n: int, ng: int) -> None:
    """``fv_mapz.F90:975``, in place -- the deferred half of the remap.

    ``pt = (pt + dtmp/cp*pkz) / (1. + r_vir*q(sphum))``, EXACTLY as
    written, on the HYDROSTATIC arm where :975 is ungated by
    ``adiabatic``.  Split out of :func:`lagrangian_to_eulerian` because
    ``dtmp`` is a GLOBAL reduction over all six faces and that function
    runs one face at a time; see the ``defer_close`` comment there.

    ``dtmp = 0`` and ``r_vir = 0`` make this the identity, and the
    zero-``dtmp`` branches are kept separate so the certified
    ``consv_te = 0`` lane keeps its exact expression rather than
    acquiring an add of a zero.
    """
    ia = ng
    win = (slice(ia, ia + n), slice(ia, ia + n), slice(None))
    if dtmp == 0.0:
        if r_vir != 0.0:
            pt[win] /= (1.0 + r_vir * q[int(sphum_index)][win])
        return
    add = pt[win] + dtmp / cp * pkz
    if r_vir != 0.0:
        pt[win] = add / (1.0 + r_vir * q[int(sphum_index)][win])
    else:
        pt[win] = add


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
                           do_adiabatic_init=False, defer_close=False):
    """``Lagrangian_to_Eulerian`` for ONE face, in place (fv_mapz.F90:62).

    Arrays follow ``fv3_native_state_3d``'s layout contract, 0-based
    numpy with the Fortran origins implied:

        delp, pt, ps  (m_a, m_a, km) / (m_a, m_a)   [isd, jsd]
        u             (m_a, m_b, km)                [isd, jsd]
        v             (m_b, m_a, km)                [isd, jsd]
        pk            (m_a, m_a, km+1)              [isd, jsd]
        pe            (n+2, km+1, n+2)              [is-1, 1, js-1]  (i,k,j)
        peln          (n,   km+1, n)                [is,   1, js]    (i,k,j)
        pkz           (n, n, km)                    [is, js, 1]

    with ``m_a = n + 2*ng``, ``m_b = m_a + 1``, ``is = js = 1``,
    ``ie = je = n``.

    ``q`` is REQUIRED and has no default: it is the list of tracer arrays
    (each shaped like ``delp``), and ``[]`` means "this configuration has
    none". The pinned deck has nr = 2 (ncnst=3, dnats=1), so the oracle
    makes two passes through ``fv_mapz.F90:330-342``; a defaulted
    ``q=None`` would let a caller skip both and still get a plausible
    ``pt``/``delp``. Make the choice visible at the call site.

    There is deliberately no ``qmin`` knob for the tracer remap: the
    oracle passes the literal ``0.`` at ``:335``, so exposing it would be
    an answer-changing option with no upstream counterpart.

    PRECONDITION THAT IS NOT CHECKABLE HERE: ``pe``'s one-cell halo ring
    must be current.  The u/v remap reads ``pe(i,k,j-1)``
    (fv_mapz.F90:536) and ``pe(i-1,k,j)`` (:562).  Upstream that ring is
    written by the D-grid ``geopk`` at ``dyn_core.F90:1401`` on EVERY
    acoustic substep (``dyn_core.F90:2735-2756`` writes i=is-1..ie+1,
    j=js-1..je+1), fed by the ``delp``/``pt`` duo exchange at
    ``dyn_core.F90:1333-1338``.  A caller that skips the final geopk will
    remap the winds against a stale ring and the error will look like a
    boundary-row physics defect.

    ``pt`` enters as virtual POTENTIAL temperature and leaves as ``T``
    (``last_step``) or back as ``theta_v`` (``fv_mapz.F90:994-1002``).
    """
    if q is None or isinstance(q, np.ndarray):
        raise TypeError(
            "q must be a list of tracer arrays ([] for none), not "
            f"{type(q).__name__}. It has no default so that skipping "
            "the tracer remap is always a visible choice: the pinned "
            "deck has nr=2 and the oracle makes two passes through "
            "fv_mapz.F90:330-342.")
    nq = len(q)
    if defer_close and not last_step:
        # SILENTLY IGNORED BEFORE (GLM MINOR, job 9446300): the
        # non-last_step arm does `pt /= pkz` (theta_v back-conversion)
        # regardless, so a caller believing pt was deferred T_v would
        # then hand theta_v to close_out_pt and get
        # (theta_v + dtmp/cp*pkz)/... -- garbage, with no error.
        raise ValueError(
            "defer_close=True with last_step=False: the fixer lives "
            "inside `if (last_step)` (fv_mapz.F90:628), so there is "
            "nothing to defer here, and the non-last_step arm converts "
            "pt back to theta_v -- which close_out_pt must never see.")
    _refuse_unported_lane(hydrostatic=hydrostatic, adiabatic=adiabatic,
                          consv=consv, fill=fill, kord_tm=kord_tm,
                          do_sat_adj=do_sat_adj, do_inline_mp=do_inline_mp,
                          do_adiabatic_init=do_adiabatic_init, nq=nq,
                          last_step=last_step, defer_close=defer_close)
    ppm_profile_is_unported(kord_mt)
    ppm_profile_is_unported(abs(int(kord_tm)))
    kords_tr = ([int(kord_tr)] * nq if np.isscalar(kord_tr)
                else [int(x) for x in kord_tr])
    if len(kords_tr) != nq:
        raise ValueError(f"kord_tr has {len(kords_tr)} entries for {nq} "
                         f"tracers")
    for kt in kords_tr:
        ppm_profile_is_unported(kt)
    if r_vir != 0.0 and last_step:
        # fv_mapz.F90:975 divides by (1 + r_vir*q(...,sphum)) using the
        # EXPLICIT `sphum` DUMMY ARGUMENT declared at :80 -- NOT a
        # get_tracer_index lookup (:169-176 fetch liq_wat/ice_wat/rainwat/
        # snowwat/graupel/cld_amt/ccn/cin, never sphum). Guessing tracer 0
        # would silently divide by the wrong species. The conversion lives
        # inside the `if (last_step)` block at :964, so a non-last_step
        # call never reads sphum and must not be refused for lacking it.
        if q is None:
            raise ValueError(
                "r_vir != 0 needs the tracers for the closing T_v -> T "
                "conversion (fv_mapz.F90:975). The reference deck is "
                "adiabatic (driver/solo/atmosphere.F90:156-158, zvir = 0).")
        # int() would COERCE: True -> 1, 1.5 -> 1, "1" -> 1, so the
        # "never guess the species" guard would admit a bool, a float or
        # a string and divide by tracer 1 (GLM MINOR, job 9442423).
        # __index__ accepts int and numpy integers and rejects the rest;
        # bool has one, so it is excluded by name.
        # operator.index NORMALISES; the bounds test must then run on
        # the normalised value, not the original object (codex MINOR,
        # job 9442482). bool has __index__ too and is excluded by name.
        _si_in = sphum_index          # keep for the message
        if isinstance(sphum_index, bool):
            sphum_index = None
        else:
            try:
                sphum_index = _operator.index(sphum_index)
            except TypeError:
                sphum_index = None
        if sphum_index is None or not 0 <= sphum_index < len(q):
            raise ValueError(
                f"r_vir != 0 needs sphum_index an int in [0, {len(q)}) -- "
                f"fv_mapz.F90:975 uses the explicit sphum argument, and "
                f"assuming tracer 0 divides by the wrong species. Got "
                f"{_si_in!r}.")
        if not hydrostatic and adiabatic:
            # fv_mapz.F90:985 -- on the NON-hydrostatic arm the closing
            # T_v -> T conversion sits inside `if (.not. adiabatic)`, so
            # with adiabatic=.true. the oracle does NOT convert at all
            # and pt stays virtual.  This lane always divides, which is
            # the `.not. adiabatic` branch (:987).  dtmp is initialised
            # to 0. at :627 and assigned ONLY inside `consv > consv_min`
            # (:629/:708) and `consv < -consv_min` (:738-741), both of
            # which this lane refuses, so dtmp is identically 0 and
            # `(pt + dtmp/cv_air*pkz)/(1+r_vir*q)` reduces EXACTLY to
            # what is computed here -- cv_air vs cp cannot matter on a
            # term that is zero.  Refuse the
            # combination rather than silently running a different branch
            # of the oracle than the flag names.  (The HYDROSTATIC :975
            # divide is ungated by adiabatic, which is why this is
            # NH-only.)
            raise NotImplementedError(
                "r_vir != 0 with hydrostatic=False and adiabatic=True: "
                "the oracle SKIPS the closing T_v -> T conversion there "
                "(fv_mapz.F90:985) while this lane always divides. Pass "
                "adiabatic=False -- with consv = 0 the oracle's :987 "
                "expression is exactly this lane's.")
    if not hydrostatic:
        # fv_mapz.F90:345-419 needs the NH prognostics and the D-stage
        # surface velocity; :460-493 needs rrg = -rdgas/grav (:167).
        missing = [nm for nm, a in (("w", w), ("delz", delz), ("ws", ws),
                                    ("rdgas", rdgas), ("grav", grav))
                   if a is None]
        if missing:
            raise ValueError(
                f"hydrostatic=False needs {missing}: the w/delz remap "
                f"(fv_mapz.F90:345-365) reads w, delz and ws, and the NH "
                f"pkz (:480) needs rrg = -rdgas/grav (:167)")
        if int(kord_wz) < 0:
            # kord_wz < 0 selects iv=-3 (fv_mapz.F90:347-351), whose
            # cs_profile LBC branch reads uninitialised memory -- refused
            # exactly as _refuse_iv_m3 documents.  The pinned deck has
            # kord_wz = 9.
            _refuse_iv_m3(km)
    if last_step and omga is None:
        # fv_mapz.F90:431-441 and :504-523 both run under `last_step`, and
        # the oracle ALWAYS has an omga to update. Skipping it silently
        # would be a divergence with no upstream counterpart; a caller that
        # genuinely does not carry omega must say so by passing an array
        # it is willing to have overwritten.
        raise ValueError(
            "last_step=True requires omga: fv_mapz.F90:504-523 interpolates "
            "omega onto the remapped cell centres on every last_step, so "
            "omitting it is a silent divergence. Pass the field (it is "
            "overwritten in place); pass last_step=False if this is not the "
            "final outer split.")

    ak = np.asarray(ak, dtype=np.float64)
    bk = np.asarray(bk, dtype=np.float64)
    if ak.shape != (km + 1,) or bk.shape != (km + 1,):
        raise ValueError(f"ak/bk must be (km+1,)={km + 1}, got {ak.shape} "
                         f"and {bk.shape}")
    ak1, bk1 = pad1(ak[None, :])[0], pad1(bk[None, :])[0]   # 1-based k

    # Row 0 of an [isd..] axis IS Fortran isd = 1-ng (fv3_native_gridstruct
    # module docstring), so Fortran index i sits at i - 1 + ng, and i=is=1
    # sits at ng. pe's origin is is-1 = 0, so its i=is sits at 1.
    ia = ng
    ipe = 1
    abs_kord_tm = abs(int(kord_tm))

    # pe4 (fv_mapz.F90:153, 572-576). Only j=js..je is ever read back at
    # :619-625; the j=je+1 pass writes a pe2 that is stale by construction
    # (its k=2..km fill sits inside the `j /= je+1` guard), so it is not
    # reproduced here.
    pe4 = np.zeros((n, n, km), dtype=np.float64)

    for j in range(1, n + 2):                    # do 1000 j=js,je+1
        jd = j + ng - 1                          # [jsd..] index of row j
        jpe = j                                  # pe's [js-1..] index (js=1)

        # :197-201
        pe1 = pad1(pe[ipe:ipe + n, :, jpe])
        # :203-206
        pe2 = np.zeros((n, km + 2), dtype=np.float64)
        pe2[:, 1] = ptop
        pe2[:, km + 1] = pe[ipe:ipe + n, km, jpe]

        if j != n + 1:                           # :208 if ( j /= (je+1) )
            jl = j - 1                           # [is..]/[js..] index

            if hydrostatic:
                # :211-217 -- Theta_v -> T_v with the LAGRANGIAN
                # hydrostatic pkz. peln is not overwritten with pn2
                # until :447, i.e. after the remap.
                pkc = pk[ia:ia + n, jd, :]
                pln = peln[:, :, jl]
                pt[ia:ia + n, jd, :] *= (
                    (pkc[:, 1:] - pkc[:, :-1])
                    / (akap * (pln[:, 1:] - pln[:, :-1])))
            else:
                # :218-237 (non-moist arm, :231-232) -- "density pt" to
                # "density temp": pt *= exp(k1k*log(rrg*delp/delz*pt))
                # with k1k = rdgas/cv_air (:164) and the PRE-conversion
                # delz/delp (this precedes both :252-258 and :281).
                # codex NH r3 #1: taking the hydrostatic Dpk/(akap
                # Dpeln) form here instead put a uniform ~2.8e-4 theta
                # error on every column -- the 0.408 m delz parity
                # residual.
                k1k = rdgas / (cp - rdgas)
                rrg = -rdgas / grav
                ptw = pt[ia:ia + n, jd, :]
                pt[ia:ia + n, jd, :] = ptw * np.exp(k1k * np.log(
                    rrg * delp[ia:ia + n, jd, :] / delz[:, jl, :] * ptw))

            # :252-258 -- NH: delz -> "specific volume"/grav on the OLD
            # delp, BEFORE :281 overwrites delp with the target dp2.
            if not hydrostatic:
                delz[:, jl, :] = (-delz[:, jl, :]
                                  / delp[ia:ia + n, jd, :])

            # :261-263
            ps[ia:ia + n, jd] = pe1[:, km + 1]

            # :267-276 -- hybrid sigma-p target, then its thicknesses
            for k in range(2, km + 1):
                pe2[:, k] = ak1[k] + bk1[k] * pe[ipe:ipe + n, km, jpe]
            dp2 = np.zeros((n, km + 1), dtype=np.float64)
            for k in range(1, km + 1):
                dp2[:, k] = pe2[:, k + 1] - pe2[:, k]

            # :281-285 -- update delp
            delp[ia:ia + n, jd, :] = unpad1(dp2)

            # :290-308 -- p**kappa on both coordinates. The endpoints are
            # COPIED, not recomputed: :297-300. That is what makes the
            # source and target intervals coincide to the last bit.
            pk1 = pad1(pk[ia:ia + n, jd, :])
            pn2 = np.zeros((n, km + 2), dtype=np.float64)
            pk2 = np.zeros((n, km + 2), dtype=np.float64)
            pn2[:, 1] = peln[:, 0, jl]
            pn2[:, km + 1] = peln[:, km, jl]
            pk2[:, 1] = pk1[:, 1]
            pk2[:, km + 1] = pk1[:, km + 1]
            for k in range(2, km + 1):
                pn2[:, k] = np.log(pe2[:, k])
                pk2[:, k] = np.exp(akap * pn2[:, k])

            # :310-316 -- map T in ln(p) with scalar_profile.
            peln1 = pad1(peln[:, :, jl])
            pt[ia:ia + n, jd, :] = unpad1(map_scalar(
                peln1, pad1(pt[ia:ia + n, jd, :]), pn2, km, km,
                1, abs_kord_tm, T_MIN))

            # :327-343 -- nq > 5 through mapn_tracer (:328), else one
            # tracer at a time (:332); both with the literal 0. q_min
            if nq > 5:
                qn = mapn_tracer(
                    pe1, [pad1(q[iq][ia:ia + n, jd, :]) for iq in range(nq)],
                    pe2, dp2, km, [kords_tr[iq] for iq in range(nq)], 0.0)
                for iq in range(nq):
                    q[iq][ia:ia + n, jd, :] = unpad1(qn[iq])
            else:
                for iq in range(nq):
                    q[iq][ia:ia + n, jd, :] = unpad1(map1_q2(
                        pe1, pad1(q[iq][ia:ia + n, jd, :]), pe2, dp2, km,
                        km, 0, kords_tr[iq], 0.0))   # :335 -- literal 0.

            # :345-419 -- NH: remap w and delz, then the w_limiter.
            if not hydrostatic:
                # :347-355 -- w in LINEAR p; kord_wz=9 selects iv=-2,
                # whose bottom BC is the D-stage surface velocity ws
                # (NH-spec trap #2; the kord_wz<0/iv=-3 arm is refused
                # at entry).
                w[ia:ia + n, jd, :] = unpad1(map1_ppm(
                    pe1, pad1(w[ia:ia + n, jd, :]), pe2, km, km,
                    -2, abs(int(kord_wz)), qs=ws[:, jl]))
                # :357-360 -- delz (specific volume) with iv=1, kord_tm;
                # the oracle passes gz as a dummy qs the iv=1 branch
                # never reads.
                delz[:, jl, :] = unpad1(map1_ppm(
                    pe1, pad1(delz[:, jl, :]), pe2, km, km,
                    1, abs_kord_tm))
                # :361-365 -- back to a (negative) thickness on dp2.
                delz[:, jl, :] = -delz[:, jl, :] * unpad1(dp2)

                if w_limiter:                    # :368-418
                    # Momentum-conserving w clamp.  Bounds are the
                    # MODULE PARAMETERS w_max=90/w_min=-60
                    # (fv_mapz.F90:51-52); the deck's namelist W_MAX=75
                    # is flagstruct's, a different variable.
                    d2 = unpad1(dp2)
                    w2 = np.array(w[ia:ia + n, jd, :], copy=True)
                    for k in range(km - 1):      # :373-390 down pass
                        spill = np.where(
                            w2[:, k] > W_MAX_MAPZ,
                            (w2[:, k] - W_MAX_MAPZ) * d2[:, k],
                            np.where(w2[:, k] < W_MIN_MAPZ,
                                     (w2[:, k] - W_MIN_MAPZ) * d2[:, k],
                                     0.0))
                        w2[:, k] = np.clip(w2[:, k], W_MIN_MAPZ,
                                           W_MAX_MAPZ)
                        w2[:, k + 1] = w2[:, k + 1] + spill / d2[:, k + 1]
                    for k in range(km - 1, 0, -1):   # :391-407 up pass
                        spill = np.where(
                            w2[:, k] > W_MAX_MAPZ,
                            (w2[:, k] - W_MAX_MAPZ) * d2[:, k],
                            np.where(w2[:, k] < W_MIN_MAPZ,
                                     (w2[:, k] - W_MIN_MAPZ) * d2[:, k],
                                     0.0))
                        w2[:, k] = np.clip(w2[:, k], W_MIN_MAPZ,
                                           W_MAX_MAPZ)
                        w2[:, k - 1] = w2[:, k - 1] + spill / d2[:, k - 1]
                    # :408-416 -- top escape valve at 2x the bounds.
                    w2[:, 0] = np.clip(w2[:, 0], 2.0 * W_MIN_MAPZ,
                                       2.0 * W_MAX_MAPZ)
                    w[ia:ia + n, jd, :] = w2

            # :424-428 -- update pk
            pk[ia:ia + n, jd, :] = unpad1(pk2)

            if last_step:                        # :431-441
                pe3_om = np.zeros((n, km + 2), dtype=np.float64)
                for k in range(2, km + 2):       # pe3(1) stays 0 (:434-436)
                    pe3_om[:, k] = omga[ia:ia + n, jd, k - 2]

            # :444-449 -- keep the OLD peln, then install the new one
            pe0_old = pad1(peln[:, :, jl])
            peln[:, :, jl] = unpad1(pn2)

            if hydrostatic:
                # :454-459 -- pkz from the EULERIAN pk2/peln
                pln = peln[:, :, jl]
                pkz[:, jl, :] = ((pk2[:, 2:km + 2] - pk2[:, 1:km + 1])
                                 / (akap * (pln[:, 1:] - pln[:, :-1])))
            else:
                # :479-481 -- NH pkz from the ideal gas law on the
                # remapped delp/delz/T_v (non-moist, kord_tm<0 arm);
                # rrg = -rdgas/grav (:167), and delz<0 keeps the log
                # argument positive.
                rrg = -rdgas / grav
                pkz[:, jl, :] = np.exp(akap * np.log(
                    rrg * delp[ia:ia + n, jd, :] / delz[:, jl, :]
                    * pt[ia:ia + n, jd, :]))
                pln = peln[:, :, jl]      # the omega block below reads it

            # :504-523 -- omega interpolated to the remapped cell centres
            if last_step:
                mid = 0.5 * (pln[:, :-1] + pln[:, 1:])       # :507
                for i in range(n):
                    k_next = 1
                    for nn in range(1, km + 1):
                        for k in range(k_next, km + 1):
                            if (mid[i, nn - 1] <= pe0_old[i, k + 1]
                                    and mid[i, nn - 1] >= pe0_old[i, k]):
                                omga[ia + i, jd, nn - 1] = (
                                    pe3_om[i, k]
                                    + (pe3_om[i, k + 1] - pe3_om[i, k])
                                    * (mid[i, nn - 1] - pe0_old[i, k])
                                    / (pe0_old[i, k + 1] - pe0_old[i, k]))
                                k_next = k
                                break

        # ------------------------------------------------------------
        # :528-549 -- map u. Runs for EVERY j including je+1, which is
        # why the j-loop extends one past je.
        # ------------------------------------------------------------
        pe0_u = np.zeros((n, km + 2), dtype=np.float64)
        pe3_u = np.zeros((n, km + 2), dtype=np.float64)
        pe0_u[:, 1] = pe[ipe:ipe + n, 0, jpe]                # :528-530
        for k in range(2, km + 2):                           # :534-538
            pe0_u[:, k] = 0.5 * (pe[ipe:ipe + n, k - 1, jpe - 1]
                                 + pe1[:, k])
        for k in range(1, km + 2):                           # :540-545
            bkh = 0.5 * bk1[k]
            pe3_u[:, k] = ak1[k] + bkh * (pe[ipe:ipe + n, km, jpe - 1]
                                          + pe1[:, km + 1])
        u[ia:ia + n, jd, :] = unpad1(map1_ppm(
            pe0_u, pad1(u[ia:ia + n, jd, :]), pe3_u, km, km, -1,
            int(kord_mt)))

        if j < n + 1:                            # :551 if (j < je+1)
            # :555-569 -- map v over i = is..ie+1
            npv = n + 1
            pe0_v = np.zeros((npv, km + 2), dtype=np.float64)
            pe3_v = np.zeros((npv, km + 2), dtype=np.float64)
            pe0_v[:, 1] = pe[ipe:ipe + npv, 0, jpe]          # :528-530
            pe3_v[:, 1] = ak1[1]                             # :555-557
            for k in range(2, km + 2):                       # :559-565
                bkh = 0.5 * bk1[k]
                pe0_v[:, k] = 0.5 * (pe[ipe - 1:ipe - 1 + npv, k - 1, jpe]
                                     + pe[ipe:ipe + npv, k - 1, jpe])
                pe3_v[:, k] = ak1[k] + bkh * (
                    pe[ipe - 1:ipe - 1 + npv, km, jpe]
                    + pe[ipe:ipe + npv, km, jpe])
            v[ia:ia + npv, jd, :] = unpad1(map1_ppm(
                pe0_v, pad1(v[ia:ia + npv, jd, :]), pe3_v, km, km, -1,
                int(kord_mt)))

            # :572-576 -- stash the new interfaces for the pe update
            pe4[:, j - 1, :] = pe2[:, 2:km + 2]

    # :618-625 -- pe(i,k,j) = pe4(i,j,k-1) for k=2..km
    for k in range(2, km + 1):
        pe[ipe:ipe + n, k - 1, 1:n + 1] = pe4[:, :, k - 2]

    # :964-1004 -- close out pt. dtmp is exactly 0 (consv <= consv_min,
    # fv_mapz.F90:627/630) and r_vir is 0 on the adiabatic deck, so the
    # last_step arm is an identity there -- implemented as written so a
    # moist lane is not silently wrong.
    if last_step and defer_close:
        # THE ENERGY FIXER NEEDS A GLOBAL SUM AND THIS LANE IS PER FACE.
        # fv_mapz.F90 computes te_2d, reduces it across the domain and
        # applies dtmp at :975 all inside one call, because there a
        # "domain" is every tile at once. Here each face is a separate
        # call, so dtmp cannot be known yet. Leave pt as T_v and let the
        # caller reduce over the six faces and call close_out_pt.
        pass
    elif last_step:
        if r_vir != 0.0:                          # :975, with dtmp == 0
            sphum = q[int(sphum_index)]
            pt[ia:ia + n, ia:ia + n, :] /= (
                1.0 + r_vir * sphum[ia:ia + n, ia:ia + n, :])
    else:                                         # :996-1001
        pt[ia:ia + n, ia:ia + n, :] /= pkz
