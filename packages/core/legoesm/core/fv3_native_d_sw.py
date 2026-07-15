"""FV3-native D-grid full step (d_sw) — loop-faithful port, phase 4b.

Index-exact python transcription of the verbatim d_sw call-tree
extraction ``scripts/validate/fv3_native/fv3_dswcore_extract.F90``
(GFDL_atmos_cubed_sphere @ 6f658bd0, production branch — no SW_DYNAMICS,
no WAVE_FORM, no GFS_PHYS/DCMIP, no USE_SG, no ONE_SIDE, no ROT3) for
the single-tile cubed-sphere case (grid_type=0, bounded_domain=False,
all four corner flags true):

- tp_core_mod: ``fv_tp_2d``, ``copy_corners``, ``xppm``, ``yppm``,
  ``pert_ppm``, ``deln_flux`` (all iord/jord branches ported),
- a2b_edge_mod: ``a2b_ord4`` (+ ``extrap_corner``/``great_circle_dist``)
  — ON the oracle path (dddmp=0.2, nord=1, grid_type=0 reaches the
  ``a2b_ord4`` call in the divergence-damping Smagorinsky-form block),
- dsw_extract_mod: ``d_sw``, ``del6_vt_flux``, ``xtp_u``, ``ytp_v``,
  ``fill_corners_2d`` / ``fill_corners_dgrid`` / ``fill_corners_cgrid``
  / ``fill_corners_agrid`` (pure index-shuffle corner fills),
- ``smag_corner`` is a NotImplementedError stub: its call sites are
  grid_type>=3-only arms, never taken here.

Style follows the certified phase-4a port ``fv3_native_sw_core.py``:
plain-python nested loops mirroring the Fortran statement order, ``fort``
Fortran-index views, ``_fl``-style locals allocated with explicit
Fortran bounds (NaN fill).  ``bounded_domain`` / grid_type>=3 arms of
``d_sw`` itself raise NotImplementedError with the branch structure kept
verbatim; the leaf transport routines port every branch.

Fortran pitfalls handled per the phase-4a bug list: ``do a,b`` ==
``range(a, b+1)``; ``.or.`` binds looser than ``.and.``; ``sign(a,b)``
== ``math.copysign``; tp_core ``near_zero=1e-25`` vs sw_core
``near_zero=1e-9`` are DIFFERENT constants; a2b_ord4 has FUNCTION-LOCAL
``c1=2/3, c2=-1/6`` shadowing the module ``c1/c2/c3``; slice actual
arguments (``crx(is,js)``, ``q1(is1)``, column runs for ``pert_ppm``)
are passed as views with matching origins.
"""

from __future__ import annotations

import math
import os

import numpy as np

from legoesm.core.fv3_native_sw_core import Bounds
from legoesm.grids.fv3_native_gridstruct import fort

# --------------------------------------------------------------------
# tp_core.F90 module constants (verbatim extraction header, lines 41-84;
# production branch: b1..b5 are the non-WAVE_FORM "scheme 2.1" values)
# --------------------------------------------------------------------
PPM_FAC = 1.5
R3 = 1.0 / 3.0
NEAR_ZERO_TP = 1.0e-25          # tp_core near_zero
PPM_LIMITER = 2.0
R12 = 1.0 / 12.0
B1 = 1.0 / 30.0
B2 = -13.0 / 60.0
B3 = -13.0 / 60.0
B4 = 0.45
B5 = -0.05
TP_T11 = 27.0 / 28.0
TP_T12 = -13.0 / 28.0
TP_T13 = 3.0 / 7.0
TP_S11 = 11.0 / 14.0
TP_S14 = 4.0 / 7.0
TP_S15 = 3.0 / 14.0
TP_C1 = -2.0 / 14.0
TP_C2 = 11.0 / 14.0
TP_C3 = 5.0 / 14.0
TP_P1 = 7.0 / 12.0
TP_P2 = -1.0 / 12.0

# --------------------------------------------------------------------
# sw_core.F90 module constants (dsw_extract_mod header, lines 1934-1946)
# big_number is the production non-OVERLOAD_R4 sw_core value (1.E30);
# the extraction's shim-level big_number_sw=1.E8 is UNUSED in this tree.
# --------------------------------------------------------------------
BIG_NUMBER = 1.0e30
SW_R3 = 1.0 / 3.0
SW_T11 = 27.0 / 28.0
SW_T12 = -13.0 / 28.0
SW_T13 = 3.0 / 7.0
SW_T14 = 6.0 / 7.0
SW_T15 = 3.0 / 28.0
SW_S11 = 11.0 / 14.0
SW_S13 = -13.0 / 14.0
SW_S14 = 4.0 / 7.0
SW_S15 = 3.0 / 14.0
NEAR_ZERO_SW = 1.0e-9           # sw_core near_zero (xtp_u/ytp_v iord=10)
SW_P1 = 7.0 / 12.0
SW_P2 = -1.0 / 12.0
SW_A1 = 0.5625
SW_A2 = -0.0625
SW_C1 = -2.0 / 14.0
SW_C2 = 11.0 / 14.0
SW_C3 = 5.0 / 14.0

# --------------------------------------------------------------------
# a2b_edge.F90 module constants (extraction lines 1487-1497)
# --------------------------------------------------------------------
A2B_R3 = 1.0 / 3.0
A2B_A1 = 0.5625                 # 9/16, 4-pt Lagrange
A2B_A2 = -0.0625                # -1/16
A2B_B1 = 7.0 / 12.0             # PPM volume mean form
A2B_B2 = -1.0 / 12.0

# fv_mp_mod fill directions (swcore_shim_mod)
XDIR = 1
YDIR = 2


class fort1:
    """Fortran-indexed 1-D view (lo bound given)."""

    def __init__(self, a: np.ndarray, lo: int):
        self.a = a
        self.lo = lo

    def __getitem__(self, i):
        return self.a[i - self.lo]

    def __setitem__(self, i, v):
        self.a[i - self.lo] = v

    def view(self, lo: int, n: int) -> np.ndarray:
        """Contiguous slice starting at Fortran index ``lo`` (length n)."""
        return self.a[lo - self.lo: lo - self.lo + n]


def _fl(ilo: int, ihi: int, jlo: int, jhi: int,
        fill: float = np.nan) -> fort:
    """New Fortran-indexed local 2-D array with explicit bounds."""
    return fort(np.full((ihi - ilo + 1, jhi - jlo + 1), fill), ilo, jlo)


def _flb(ilo: int, ihi: int, jlo: int, jhi: int) -> fort:
    """New Fortran-indexed local 2-D LOGICAL array (False fill)."""
    return fort(np.full((ihi - ilo + 1, jhi - jlo + 1), False), ilo, jlo)


def _fl1(lo: int, hi: int, fill: float = np.nan) -> fort1:
    """New Fortran-indexed local 1-D array with explicit bounds."""
    return fort1(np.full(hi - lo + 1, fill), lo)


def _fl1b(lo: int, hi: int) -> fort1:
    """New Fortran-indexed local 1-D LOGICAL array (False fill)."""
    return fort1(np.full(hi - lo + 1, False), lo)


def _col(F: fort, ifirst: int, n: int, j: int) -> np.ndarray:
    """Writable view of the Fortran-contiguous run F(ifirst..ifirst+n-1, j).

    Mirrors passing ``F(ifirst, j)`` to a 1-D dummy of length n (column
    slice; the run never crosses a column boundary at any call site).
    """
    return F.a[ifirst - F.ilo: ifirst - F.ilo + n, j - F.jlo]


# =====================================================================
# tp_core_mod
# =====================================================================

def pert_ppm(im: int, a0, al, ar, iv: int) -> None:
    """tp_core.F90 pert_ppm (verbatim; a0/al/ar are 0-based views)."""
    # -----------------------------------
    # Optimized PPM in perturbation form:
    # -----------------------------------
    if iv == 0:
        # Positive definite constraint
        for i in range(1, im + 1):
            if a0[i - 1] <= 0.0:
                al[i - 1] = 0.0
                ar[i - 1] = 0.0
            else:
                a4 = -3.0 * (ar[i - 1] + al[i - 1])
                da1 = ar[i - 1] - al[i - 1]
                if abs(da1) < -a4:
                    fmin = a0[i - 1] + 0.25 / a4 * da1 ** 2 + a4 * R12
                    if fmin < 0.0:
                        if ar[i - 1] > 0.0 and al[i - 1] > 0.0:
                            ar[i - 1] = 0.0
                            al[i - 1] = 0.0
                        elif da1 > 0.0:
                            ar[i - 1] = -2.0 * al[i - 1]
                        else:
                            al[i - 1] = -2.0 * ar[i - 1]
    else:
        # Standard PPM constraint
        for i in range(1, im + 1):
            if al[i - 1] * ar[i - 1] < 0.0:
                da1 = al[i - 1] - ar[i - 1]
                da2 = da1 ** 2
                a6da = 3.0 * (al[i - 1] + ar[i - 1]) * da1
                # abs(a6da) > da2 --> 3.*abs(al+ar) > abs(al-ar)
                if a6da < -da2:
                    ar[i - 1] = -2.0 * al[i - 1]
                elif a6da > da2:
                    al[i - 1] = -2.0 * ar[i - 1]
            else:
                # effect of dm=0 included here
                al[i - 1] = 0.0
                ar[i - 1] = 0.0


def copy_corners(q: fort, npx: int, npy: int, dir_: int,
                 bounded_domain: bool, bd: Bounds,
                 sw_corner: bool, se_corner: bool,
                 nw_corner: bool, ne_corner: bool) -> None:
    """tp_core.F90 copy_corners (verbatim; rotates data through corners)."""
    ng = bd.ng

    if bounded_domain:
        return

    if dir_ == 1:
        # XDir:
        if sw_corner:
            for j in range(1 - ng, 0 + 1):
                for i in range(1 - ng, 0 + 1):
                    q[i, j] = q[j, 1 - i]
        if se_corner:
            for j in range(1 - ng, 0 + 1):
                for i in range(npx, npx + ng - 1 + 1):
                    q[i, j] = q[npy - j, i - npx + 1]
        if ne_corner:
            for j in range(npy, npy + ng - 1 + 1):
                for i in range(npx, npx + ng - 1 + 1):
                    q[i, j] = q[j, 2 * npx - 1 - i]
        if nw_corner:
            for j in range(npy, npy + ng - 1 + 1):
                for i in range(1 - ng, 0 + 1):
                    q[i, j] = q[npy - j, i - 1 + npx]

    elif dir_ == 2:
        # YDir:
        if sw_corner:
            for j in range(1 - ng, 0 + 1):
                for i in range(1 - ng, 0 + 1):
                    q[i, j] = q[1 - j, i]
        if se_corner:
            for j in range(1 - ng, 0 + 1):
                for i in range(npx, npx + ng - 1 + 1):
                    q[i, j] = q[npy + j - 1, npx - i]
        if ne_corner:
            for j in range(npy, npy + ng - 1 + 1):
                for i in range(npx, npx + ng - 1 + 1):
                    q[i, j] = q[2 * npy - 1 - j, i]
        if nw_corner:
            for j in range(npy, npy + ng - 1 + 1):
                for i in range(1 - ng, 0 + 1):
                    q[i, j] = q[j + 1 - npx, npy - i]


def xppm(flux: fort, q: fort, c: fort, iord: int,
         is_: int, ie: int, isd: int, ied: int,
         jfirst: int, jlast: int, jsd: int, jed: int,
         npx: int, npy: int, dxa: fort,
         bounded_domain: bool, grid_type: int, lim_fac: float) -> None:
    """tp_core.F90 xppm (verbatim, all iord branches).

    flux(is:ie+1, jfirst:jlast) OUT; q(isd:ied, jfirst:jlast) IN;
    c(is:ie+1, jfirst:jlast) Courant.  All fort views carry origins that
    line up with the Fortran sequence association at every call site.
    """
    if (not bounded_domain) and grid_type < 3:
        is1 = max(3, is_ - 1)
        ie3 = min(npx - 2, ie + 2)
        ie1 = min(npx - 3, ie + 1)
    else:
        is1 = is_ - 1
        ie3 = ie + 2
        ie1 = ie + 1

    mord = abs(iord)

    for j in range(jfirst, jlast + 1):        # do 666 j=jfirst,jlast

        q1 = _fl1(isd, ied)
        for i in range(isd, ied + 1):
            q1[i] = q[i, j]

        if iord < 7:
            # ord = 2: perfectly linear ppm scheme
            # Diffusivity: ord2 < ord5 < ord3 < ord4 < ord6

            al = _fl1(is_ - 1, ie + 2)
            for i in range(is1, ie3 + 1):
                al[i] = TP_P1 * (q1[i - 1] + q1[i]) \
                    + TP_P2 * (q1[i - 2] + q1[i + 1])

            if (not bounded_domain) and grid_type < 3:
                if is_ == 1:
                    al[0] = TP_C1 * q1[-2] + TP_C2 * q1[-1] + TP_C3 * q1[0]
                    al[1] = 0.5 * (
                        ((2.0 * dxa[0, j] + dxa[-1, j]) * q1[0]
                         - dxa[0, j] * q1[-1]) / (dxa[-1, j] + dxa[0, j])
                        + ((2.0 * dxa[1, j] + dxa[2, j]) * q1[1]
                           - dxa[1, j] * q1[2]) / (dxa[1, j] + dxa[2, j]))
                    al[2] = TP_C3 * q1[1] + TP_C2 * q1[2] + TP_C1 * q1[3]
                if (ie + 1) == npx:
                    al[npx - 1] = TP_C1 * q1[npx - 3] \
                        + TP_C2 * q1[npx - 2] + TP_C3 * q1[npx - 1]
                    al[npx] = 0.5 * (
                        ((2.0 * dxa[npx - 1, j] + dxa[npx - 2, j])
                         * q1[npx - 1] - dxa[npx - 1, j] * q1[npx - 2])
                        / (dxa[npx - 2, j] + dxa[npx - 1, j])
                        + ((2.0 * dxa[npx, j] + dxa[npx + 1, j]) * q1[npx]
                           - dxa[npx, j] * q1[npx + 1])
                        / (dxa[npx, j] + dxa[npx + 1, j]))
                    al[npx + 1] = TP_C3 * q1[npx] \
                        + TP_C2 * q1[npx + 1] + TP_C1 * q1[npx + 2]

            if iord < 0:
                for i in range(is_ - 1, ie + 2 + 1):
                    al[i] = max(0.0, al[i])

            if mord == 1:      # perfectly linear scheme
                bl = _fl1(is_ - 1, ie + 1)
                br = _fl1(is_ - 1, ie + 1)
                b0 = _fl1(is_ - 1, ie + 1)
                smt5 = _fl1b(is_ - 1, ie + 1)
                fx1 = _fl1(is_, ie + 1)
                for i in range(is_ - 1, ie + 1 + 1):
                    bl[i] = al[i] - q1[i]
                    br[i] = al[i + 1] - q1[i]
                    b0[i] = bl[i] + br[i]
                    smt5[i] = abs(lim_fac * b0[i]) < abs(bl[i] - br[i])
                for i in range(is_, ie + 1 + 1):
                    if c[i, j] > 0.0:
                        fx1[i] = (1.0 - c[i, j]) * (br[i - 1]
                                                    - c[i, j] * b0[i - 1])
                        flux[i, j] = q1[i - 1]
                    else:
                        fx1[i] = (1.0 + c[i, j]) * (bl[i] + c[i, j] * b0[i])
                        flux[i, j] = q1[i]
                    if smt5[i - 1] or smt5[i]:
                        flux[i, j] = flux[i, j] + fx1[i]

            elif mord == 2:    # perfectly linear scheme
                for i in range(is_, ie + 1 + 1):
                    xt = c[i, j]
                    if xt > 0.0:
                        qtmp = q1[i - 1]
                        flux[i, j] = qtmp + (1.0 - xt) * (
                            al[i] - qtmp
                            - xt * (al[i - 1] + al[i] - (qtmp + qtmp)))
                    else:
                        qtmp = q1[i]
                        flux[i, j] = qtmp + (1.0 + xt) * (
                            al[i] - qtmp
                            + xt * (al[i] + al[i + 1] - (qtmp + qtmp)))

            elif mord == 3:
                bl = _fl1(is_ - 1, ie + 1)
                br = _fl1(is_ - 1, ie + 1)
                b0 = _fl1(is_ - 1, ie + 1)
                smt5 = _fl1b(is_ - 1, ie + 1)
                smt6 = _fl1b(is_ - 1, ie + 1)
                xt1 = _fl1(is_, ie + 1)
                for i in range(is_ - 1, ie + 1 + 1):
                    bl[i] = al[i] - q1[i]
                    br[i] = al[i + 1] - q1[i]
                    b0[i] = bl[i] + br[i]
                    x0 = abs(b0[i])
                    xt = abs(bl[i] - br[i])
                    smt5[i] = x0 < xt
                    smt6[i] = 3.0 * x0 < xt
                for i in range(is_, ie + 1 + 1):
                    xt1[i] = c[i, j]
                    if xt1[i] > 0.0:
                        if smt5[i - 1] or smt6[i]:
                            flux[i, j] = q1[i - 1] + (1.0 - xt1[i]) * (
                                br[i - 1] - xt1[i] * b0[i - 1])
                        else:
                            flux[i, j] = q1[i - 1]
                    else:
                        if smt6[i - 1] or smt5[i]:
                            flux[i, j] = q1[i] + (1.0 + xt1[i]) * (
                                bl[i] + xt1[i] * b0[i])
                        else:
                            flux[i, j] = q1[i]

            elif mord == 4:
                bl = _fl1(is_ - 1, ie + 1)
                br = _fl1(is_ - 1, ie + 1)
                b0 = _fl1(is_ - 1, ie + 1)
                smt5 = _fl1b(is_ - 1, ie + 1)
                smt6 = _fl1b(is_ - 1, ie + 1)
                xt1 = _fl1(is_, ie + 1)
                fx1 = _fl1(is_, ie + 1)
                hi5 = _fl1b(is_, ie + 1)
                hi6 = _fl1b(is_, ie + 1)
                for i in range(is_ - 1, ie + 1 + 1):
                    bl[i] = al[i] - q1[i]
                    br[i] = al[i + 1] - q1[i]
                    b0[i] = bl[i] + br[i]
                    x0 = abs(b0[i])
                    xt = abs(bl[i] - br[i])
                    smt5[i] = x0 < xt
                    smt6[i] = 3.0 * x0 < xt
                for i in range(is_, ie + 1 + 1):
                    xt1[i] = c[i, j]
                    hi5[i] = smt5[i - 1] and smt5[i]   # more diffusive
                    hi6[i] = smt6[i - 1] or smt6[i]
                    hi5[i] = hi5[i] or hi6[i]
                for i in range(is_, ie + 1 + 1):
                    if xt1[i] > 0.0:
                        fx1[i] = (1.0 - xt1[i]) * (br[i - 1]
                                                   - xt1[i] * b0[i - 1])
                        flux[i, j] = q1[i - 1]
                    else:
                        fx1[i] = (1.0 + xt1[i]) * (bl[i] + xt1[i] * b0[i])
                        flux[i, j] = q1[i]
                    if hi5[i]:
                        flux[i, j] = flux[i, j] + fx1[i]

            else:              # mord 5, 6
                bl = _fl1(is_ - 1, ie + 1)
                br = _fl1(is_ - 1, ie + 1)
                b0 = _fl1(is_ - 1, ie + 1)
                smt5 = _fl1b(is_ - 1, ie + 1)
                fx1 = _fl1(is_, ie + 1)
                if iord == 5:
                    for i in range(is_ - 1, ie + 1 + 1):
                        bl[i] = al[i] - q1[i]
                        br[i] = al[i + 1] - q1[i]
                        b0[i] = bl[i] + br[i]
                        smt5[i] = bl[i] * br[i] < 0.0
                else:
                    if iord == -5:
                        da1 = _fl1(is_ - 1, ie + 1)
                        a4 = _fl1(is_ - 1, ie + 1)
                        for i in range(is_ - 1, ie + 1 + 1):
                            bl[i] = al[i] - q1[i]
                            br[i] = al[i + 1] - q1[i]
                            b0[i] = bl[i] + br[i]
                            smt5[i] = bl[i] * br[i] < 0.0
                            da1[i] = br[i] - bl[i]
                            a4[i] = -3.0 * b0[i]
                        for i in range(is_ - 1, ie + 1 + 1):
                            if abs(da1[i]) < -a4[i]:
                                if q1[i] + 0.25 / a4[i] * da1[i] ** 2 \
                                        + a4[i] * R12 < 0.0:
                                    if not smt5[i]:
                                        br[i] = 0.0
                                        bl[i] = 0.0
                                        b0[i] = 0.0
                                    elif da1[i] > 0.0:
                                        br[i] = -2.0 * bl[i]
                                        b0[i] = -bl[i]
                                    else:
                                        bl[i] = -2.0 * br[i]
                                        b0[i] = -br[i]
                    else:
                        for i in range(is_ - 1, ie + 1 + 1):
                            bl[i] = al[i] - q1[i]
                            br[i] = al[i + 1] - q1[i]
                            b0[i] = bl[i] + br[i]
                            smt5[i] = 3.0 * abs(b0[i]) < abs(bl[i] - br[i])

                    # WMP: fix edge issues
                    if (not bounded_domain) and grid_type < 3:
                        if is_ == 1:
                            smt5[0] = bl[0] * br[0] < 0.0
                            smt5[1] = bl[1] * br[1] < 0.0
                        if (ie + 1) == npx:
                            smt5[npx - 1] = bl[npx - 1] * br[npx - 1] < 0.0
                            smt5[npx] = bl[npx] * br[npx] < 0.0

                for i in range(is_, ie + 1 + 1):
                    if c[i, j] > 0.0:
                        fx1[i] = (1.0 - c[i, j]) * (br[i - 1]
                                                    - c[i, j] * b0[i - 1])
                        flux[i, j] = q1[i - 1]
                    else:
                        fx1[i] = (1.0 + c[i, j]) * (bl[i] + c[i, j] * b0[i])
                        flux[i, j] = q1[i]
                    if smt5[i - 1] or smt5[i]:
                        flux[i, j] = flux[i, j] + fx1[i]

            continue          # goto 666

        else:
            # Monotonic constraints:
            # ord = 8: PPM with Lin's PPM fast monotone constraint
            # ord = 10: PPM with Lin's modification of Huynh 2nd constraint
            # ord = 13: positive definite constraint

            bl = _fl1(is_ - 1, ie + 1)
            br = _fl1(is_ - 1, ie + 1)
            b0 = _fl1(is_ - 1, ie + 1)
            al = _fl1(is_ - 1, ie + 2)
            dm = _fl1(is_ - 2, ie + 2)
            dq = _fl1(is_ - 3, ie + 2)

            for i in range(is_ - 2, ie + 2 + 1):
                xt = 0.25 * (q1[i + 1] - q1[i - 1])
                dm[i] = math.copysign(
                    min(abs(xt),
                        max(q1[i - 1], q1[i], q1[i + 1]) - q1[i],
                        q1[i] - min(q1[i - 1], q1[i], q1[i + 1])), xt)
            for i in range(is1, ie1 + 1 + 1):
                al[i] = 0.5 * (q1[i - 1] + q1[i]) + R3 * (dm[i - 1] - dm[i])

            if iord == 8:
                for i in range(is1, ie1 + 1):
                    xt = 2.0 * dm[i]
                    bl[i] = -math.copysign(
                        min(abs(xt), abs(al[i] - q1[i])), xt)
                    br[i] = math.copysign(
                        min(abs(xt), abs(al[i + 1] - q1[i])), xt)
            elif iord == 10:
                for i in range(is1 - 2, ie1 + 1 + 1):
                    dq[i] = 2.0 * (q1[i + 1] - q1[i])
                for i in range(is1, ie1 + 1):
                    bl[i] = al[i] - q1[i]
                    br[i] = al[i + 1] - q1[i]
                    if abs(dm[i - 1]) + abs(dm[i]) + abs(dm[i + 1]) \
                            < NEAR_ZERO_TP:
                        bl[i] = 0.0
                        br[i] = 0.0
                    elif abs(3.0 * (bl[i] + br[i])) > abs(bl[i] - br[i]):
                        pmp_2 = dq[i - 1]
                        lac_2 = pmp_2 - 0.75 * dq[i - 2]
                        br[i] = min(max(0.0, pmp_2, lac_2),
                                    max(br[i], min(0.0, pmp_2, lac_2)))
                        pmp_1 = -dq[i]
                        lac_1 = pmp_1 + 0.75 * dq[i + 1]
                        bl[i] = min(max(0.0, pmp_1, lac_1),
                                    max(bl[i], min(0.0, pmp_1, lac_1)))
            elif iord == 11:
                # This is emulation of 2nd van Leer scheme using PPM codes
                for i in range(is1, ie1 + 1):
                    xt = PPM_FAC * dm[i]
                    bl[i] = -math.copysign(
                        min(abs(xt), abs(al[i] - q1[i])), xt)
                    br[i] = math.copysign(
                        min(abs(xt), abs(al[i + 1] - q1[i])), xt)
            elif iord == 7 or iord == 12:
                # positive definite (Lin & Rood 1996)
                da1 = _fl1(is_ - 1, ie + 1)
                a4 = _fl1(is_ - 1, ie + 1)
                ext5 = _fl1b(is_ - 1, ie + 1)
                ext6 = _fl1b(is_ - 1, ie + 1)
                for i in range(is1, ie1 + 1):
                    bl[i] = al[i] - q1[i]
                    br[i] = al[i + 1] - q1[i]
                    a4[i] = -3.0 * (bl[i] + br[i])
                    da1[i] = br[i] - bl[i]
                    ext5[i] = br[i] * bl[i] > 0.0
                    ext6[i] = abs(da1[i]) < -a4[i]
                for i in range(is1, ie1 + 1):
                    if ext6[i]:
                        if q1[i] + 0.25 / a4[i] * da1[i] ** 2 \
                                + a4[i] * R12 < 0.0:
                            if ext5[i]:
                                br[i] = 0.0
                                bl[i] = 0.0
                            elif da1[i] > 0.0:
                                br[i] = -2.0 * bl[i]
                            else:
                                bl[i] = -2.0 * br[i]
            else:
                for i in range(is1, ie1 + 1):
                    bl[i] = al[i] - q1[i]
                    br[i] = al[i + 1] - q1[i]

            # Positive definite constraint:
            if iord == 9 or iord == 13:
                pert_ppm(ie1 - is1 + 1, q1.view(is1, ie1 - is1 + 1),
                         bl.view(is1, ie1 - is1 + 1),
                         br.view(is1, ie1 - is1 + 1), 0)

            if (not bounded_domain) and grid_type < 3:
                if is_ == 1:
                    bl[0] = TP_S14 * dm[-1] + TP_S11 * (q1[-1] - q1[0])

                    xt = 0.5 * (
                        ((2.0 * dxa[0, j] + dxa[-1, j]) * q1[0]
                         - dxa[0, j] * q1[-1]) / (dxa[-1, j] + dxa[0, j])
                        + ((2.0 * dxa[1, j] + dxa[2, j]) * q1[1]
                           - dxa[1, j] * q1[2]) / (dxa[1, j] + dxa[2, j]))
                    xt = max(xt, min(q1[-1], q1[0], q1[1], q1[2]))
                    xt = min(xt, max(q1[-1], q1[0], q1[1], q1[2]))
                    br[0] = xt - q1[0]
                    bl[1] = xt - q1[1]
                    xt = TP_S15 * q1[1] + TP_S11 * q1[2] - TP_S14 * dm[2]
                    br[1] = xt - q1[1]
                    bl[2] = xt - q1[2]

                    br[2] = al[3] - q1[2]
                    pert_ppm(3, q1.view(0, 3), bl.view(0, 3),
                             br.view(0, 3), 1)
                if (ie + 1) == npx:
                    bl[npx - 2] = al[npx - 2] - q1[npx - 2]

                    xt = TP_S15 * q1[npx - 1] + TP_S11 * q1[npx - 2] \
                        + TP_S14 * dm[npx - 2]
                    br[npx - 2] = xt - q1[npx - 2]
                    bl[npx - 1] = xt - q1[npx - 1]

                    xt = 0.5 * (
                        ((2.0 * dxa[npx - 1, j] + dxa[npx - 2, j])
                         * q1[npx - 1] - dxa[npx - 1, j] * q1[npx - 2])
                        / (dxa[npx - 2, j] + dxa[npx - 1, j])
                        + ((2.0 * dxa[npx, j] + dxa[npx + 1, j]) * q1[npx]
                           - dxa[npx, j] * q1[npx + 1])
                        / (dxa[npx, j] + dxa[npx + 1, j]))
                    xt = max(xt, min(q1[npx - 2], q1[npx - 1],
                                     q1[npx], q1[npx + 1]))
                    xt = min(xt, max(q1[npx - 2], q1[npx - 1],
                                     q1[npx], q1[npx + 1]))
                    br[npx - 1] = xt - q1[npx - 1]
                    bl[npx] = xt - q1[npx]

                    br[npx] = TP_S11 * (q1[npx + 1] - q1[npx]) \
                        - TP_S14 * dm[npx + 1]
                    pert_ppm(3, q1.view(npx - 2, 3), bl.view(npx - 2, 3),
                             br.view(npx - 2, 3), 1)

        # (only reached for iord >= 7; iord < 7 `continue`s above)
        if iord == 7:
            b0 = _fl1(is_ - 1, ie + 1)
            smt5 = _fl1b(is_ - 1, ie + 1)
            fx1 = _fl1(is_, ie + 1)
            for i in range(is_ - 1, ie + 1 + 1):
                b0[i] = bl[i] + br[i]
                smt5[i] = bl[i] * br[i] < 0.0
            for i in range(is_, ie + 1 + 1):
                if c[i, j] > 0.0:
                    fx1[i] = (1.0 - c[i, j]) * (br[i - 1]
                                                - c[i, j] * b0[i - 1])
                    flux[i, j] = q1[i - 1]
                else:
                    fx1[i] = (1.0 + c[i, j]) * (bl[i] + c[i, j] * b0[i])
                    flux[i, j] = q1[i]
                if smt5[i - 1] or smt5[i]:
                    flux[i, j] = flux[i, j] + fx1[i]
        else:
            for i in range(is_, ie + 1 + 1):
                if c[i, j] > 0.0:
                    flux[i, j] = q1[i - 1] + (1.0 - c[i, j]) * (
                        br[i - 1] - c[i, j] * (bl[i - 1] + br[i - 1]))
                else:
                    flux[i, j] = q1[i] + (1.0 + c[i, j]) * (
                        bl[i] + c[i, j] * (bl[i] + br[i]))


def yppm(flux: fort, q: fort, c: fort, jord: int,
         ifirst: int, ilast: int, isd: int, ied: int,
         js: int, je: int, jsd: int, jed: int,
         npx: int, npy: int, dya: fort,
         bounded_domain: bool, grid_type: int, lim_fac: float) -> None:
    """tp_core.F90 yppm (verbatim, all jord branches).

    flux(ifirst:ilast, js:je+1) OUT; q(ifirst:ilast, jsd:jed) IN;
    c(isd:ied, js:je+1) Courant.
    """
    if (not bounded_domain) and grid_type < 3:
        # Cubed-sphere:
        js1 = max(3, js - 1)
        je3 = min(npy - 2, je + 2)
        je1 = min(npy - 3, je + 1)
    else:
        # Bounded_domain grid OR Doubly periodic domain:
        js1 = js - 1
        je3 = je + 2
        je1 = je + 1

    mord = abs(jord)

    if jord < 7:

        al = _fl(ifirst, ilast, js - 1, je + 2)
        bl = _fl(ifirst, ilast, js - 1, je + 1)
        br = _fl(ifirst, ilast, js - 1, je + 1)
        b0 = _fl(ifirst, ilast, js - 1, je + 1)
        smt5 = _flb(ifirst, ilast, js - 1, je + 1)
        smt6 = _flb(ifirst, ilast, js - 1, je + 1)
        fx1 = _fl1(ifirst, ilast)
        xt1 = _fl1(ifirst, ilast)
        a4 = _fl1(ifirst, ilast)
        hi5 = _fl1b(ifirst, ilast)
        hi6 = _fl1b(ifirst, ilast)

        for j in range(js1, je3 + 1):
            for i in range(ifirst, ilast + 1):
                al[i, j] = TP_P1 * (q[i, j - 1] + q[i, j]) \
                    + TP_P2 * (q[i, j - 2] + q[i, j + 1])

        if (not bounded_domain) and grid_type < 3:
            if js == 1:
                for i in range(ifirst, ilast + 1):
                    al[i, 0] = TP_C1 * q[i, -2] + TP_C2 * q[i, -1] \
                        + TP_C3 * q[i, 0]
                    al[i, 1] = 0.5 * (
                        ((2.0 * dya[i, 0] + dya[i, -1]) * q[i, 0]
                         - dya[i, 0] * q[i, -1]) / (dya[i, -1] + dya[i, 0])
                        + ((2.0 * dya[i, 1] + dya[i, 2]) * q[i, 1]
                           - dya[i, 1] * q[i, 2]) / (dya[i, 1] + dya[i, 2]))
                    al[i, 2] = TP_C3 * q[i, 1] + TP_C2 * q[i, 2] \
                        + TP_C1 * q[i, 3]
            if (je + 1) == npy:
                for i in range(ifirst, ilast + 1):
                    al[i, npy - 1] = TP_C1 * q[i, npy - 3] \
                        + TP_C2 * q[i, npy - 2] + TP_C3 * q[i, npy - 1]
                    al[i, npy] = 0.5 * (
                        ((2.0 * dya[i, npy - 1] + dya[i, npy - 2])
                         * q[i, npy - 1]
                         - dya[i, npy - 1] * q[i, npy - 2])
                        / (dya[i, npy - 2] + dya[i, npy - 1])
                        + ((2.0 * dya[i, npy] + dya[i, npy + 1]) * q[i, npy]
                           - dya[i, npy] * q[i, npy + 1])
                        / (dya[i, npy] + dya[i, npy + 1]))
                    al[i, npy + 1] = TP_C3 * q[i, npy] \
                        + TP_C2 * q[i, npy + 1] + TP_C1 * q[i, npy + 2]

        if jord < 0:
            for j in range(js - 1, je + 2 + 1):
                for i in range(ifirst, ilast + 1):
                    al[i, j] = max(0.0, al[i, j])

        if mord == 1:
            for j in range(js - 1, je + 1 + 1):
                for i in range(ifirst, ilast + 1):
                    bl[i, j] = al[i, j] - q[i, j]
                    br[i, j] = al[i, j + 1] - q[i, j]
                    b0[i, j] = bl[i, j] + br[i, j]
                    smt5[i, j] = abs(lim_fac * b0[i, j]) \
                        < abs(bl[i, j] - br[i, j])
            for j in range(js, je + 1 + 1):
                for i in range(ifirst, ilast + 1):
                    if c[i, j] > 0.0:
                        fx1[i] = (1.0 - c[i, j]) * (br[i, j - 1]
                                                    - c[i, j] * b0[i, j - 1])
                        flux[i, j] = q[i, j - 1]
                    else:
                        fx1[i] = (1.0 + c[i, j]) * (bl[i, j]
                                                    + c[i, j] * b0[i, j])
                        flux[i, j] = q[i, j]
                    if smt5[i, j - 1] or smt5[i, j]:
                        flux[i, j] = flux[i, j] + fx1[i]

        elif mord == 2:   # Perfectly linear scheme
            # Diffusivity: ord2 < ord5 < ord3 < ord4 < ord6 < ord7
            for j in range(js, je + 1 + 1):
                for i in range(ifirst, ilast + 1):
                    xt = c[i, j]
                    if xt > 0.0:
                        qtmp = q[i, j - 1]
                        flux[i, j] = qtmp + (1.0 - xt) * (
                            al[i, j] - qtmp
                            - xt * (al[i, j - 1] + al[i, j]
                                    - (qtmp + qtmp)))
                    else:
                        qtmp = q[i, j]
                        flux[i, j] = qtmp + (1.0 + xt) * (
                            al[i, j] - qtmp
                            + xt * (al[i, j] + al[i, j + 1]
                                    - (qtmp + qtmp)))

        elif mord == 3:
            for j in range(js - 1, je + 1 + 1):
                for i in range(ifirst, ilast + 1):
                    bl[i, j] = al[i, j] - q[i, j]
                    br[i, j] = al[i, j + 1] - q[i, j]
                    b0[i, j] = bl[i, j] + br[i, j]
                    x0 = abs(b0[i, j])
                    xt = abs(bl[i, j] - br[i, j])
                    smt5[i, j] = x0 < xt
                    smt6[i, j] = 3.0 * x0 < xt
            for j in range(js, je + 1 + 1):
                for i in range(ifirst, ilast + 1):
                    xt1[i] = c[i, j]
                for i in range(ifirst, ilast + 1):
                    if xt1[i] > 0.0:
                        if smt5[i, j - 1] or smt6[i, j]:
                            flux[i, j] = q[i, j - 1] + (1.0 - xt1[i]) * (
                                br[i, j - 1] - xt1[i] * b0[i, j - 1])
                        else:
                            flux[i, j] = q[i, j - 1]
                    else:
                        if smt6[i, j - 1] or smt5[i, j]:
                            flux[i, j] = q[i, j] + (1.0 + xt1[i]) * (
                                bl[i, j] + xt1[i] * b0[i, j])
                        else:
                            flux[i, j] = q[i, j]

        elif mord == 4:
            for j in range(js - 1, je + 1 + 1):
                for i in range(ifirst, ilast + 1):
                    bl[i, j] = al[i, j] - q[i, j]
                    br[i, j] = al[i, j + 1] - q[i, j]
                    b0[i, j] = bl[i, j] + br[i, j]
                    x0 = abs(b0[i, j])
                    xt = abs(bl[i, j] - br[i, j])
                    smt5[i, j] = x0 < xt
                    smt6[i, j] = 3.0 * x0 < xt
            for j in range(js, je + 1 + 1):
                for i in range(ifirst, ilast + 1):
                    xt1[i] = c[i, j]
                    hi5[i] = smt5[i, j - 1] and smt5[i, j]
                    hi6[i] = smt6[i, j - 1] or smt6[i, j]
                    hi5[i] = hi5[i] or hi6[i]
                for i in range(ifirst, ilast + 1):
                    if xt1[i] > 0.0:
                        fx1[i] = (1.0 - xt1[i]) * (br[i, j - 1]
                                                   - xt1[i] * b0[i, j - 1])
                        flux[i, j] = q[i, j - 1]
                    else:
                        fx1[i] = (1.0 + xt1[i]) * (bl[i, j]
                                                   + xt1[i] * b0[i, j])
                        flux[i, j] = q[i, j]
                    if hi5[i]:
                        flux[i, j] = flux[i, j] + fx1[i]

        else:  # mord=5,6
            if jord == 5:
                for j in range(js - 1, je + 1 + 1):
                    for i in range(ifirst, ilast + 1):
                        bl[i, j] = al[i, j] - q[i, j]
                        br[i, j] = al[i, j + 1] - q[i, j]
                        b0[i, j] = bl[i, j] + br[i, j]
                        smt5[i, j] = bl[i, j] * br[i, j] < 0.0
            else:
                if jord == -5:
                    for j in range(js - 1, je + 1 + 1):
                        for i in range(ifirst, ilast + 1):
                            bl[i, j] = al[i, j] - q[i, j]
                            br[i, j] = al[i, j + 1] - q[i, j]
                            b0[i, j] = bl[i, j] + br[i, j]
                            xt1[i] = br[i, j] - bl[i, j]
                            a4[i] = -3.0 * b0[i, j]
                            smt5[i, j] = bl[i, j] * br[i, j] < 0.0
                        for i in range(ifirst, ilast + 1):
                            if abs(xt1[i]) < -a4[i]:
                                if q[i, j] + 0.25 / a4[i] * xt1[i] ** 2 \
                                        + a4[i] * R12 < 0.0:
                                    if not smt5[i, j]:
                                        br[i, j] = 0.0
                                        bl[i, j] = 0.0
                                        b0[i, j] = 0.0
                                    elif xt1[i] > 0.0:
                                        br[i, j] = -2.0 * bl[i, j]
                                        b0[i, j] = -bl[i, j]
                                    else:
                                        bl[i, j] = -2.0 * br[i, j]
                                        b0[i, j] = -br[i, j]
                else:
                    for j in range(js - 1, je + 1 + 1):
                        for i in range(ifirst, ilast + 1):
                            bl[i, j] = al[i, j] - q[i, j]
                            br[i, j] = al[i, j + 1] - q[i, j]
                            b0[i, j] = bl[i, j] + br[i, j]
                            smt5[i, j] = 3.0 * abs(b0[i, j]) \
                                < abs(bl[i, j] - br[i, j])

                # WMP: fix edge issues
                if (not bounded_domain) and grid_type < 3:
                    if js == 1:
                        for i in range(ifirst, ilast + 1):
                            smt5[i, 0] = bl[i, 0] * br[i, 0] < 0.0
                            smt5[i, 1] = bl[i, 1] * br[i, 1] < 0.0
                    if (je + 1) == npy:
                        for i in range(ifirst, ilast + 1):
                            smt5[i, npy - 1] = bl[i, npy - 1] \
                                * br[i, npy - 1] < 0.0
                            smt5[i, npy] = bl[i, npy] * br[i, npy] < 0.0

            for j in range(js, je + 1 + 1):
                for i in range(ifirst, ilast + 1):
                    if c[i, j] > 0.0:
                        fx1[i] = (1.0 - c[i, j]) * (br[i, j - 1]
                                                    - c[i, j] * b0[i, j - 1])
                        flux[i, j] = q[i, j - 1]
                    else:
                        fx1[i] = (1.0 + c[i, j]) * (bl[i, j]
                                                    + c[i, j] * b0[i, j])
                        flux[i, j] = q[i, j]
                    if smt5[i, j - 1] or smt5[i, j]:
                        flux[i, j] = flux[i, j] + fx1[i]

        return

    else:
        # Monotonic constraints:
        # ord = 8: PPM with Lin's PPM fast monotone constraint
        # ord > 8: PPM with Lin's modification of Huynh 2nd constraint

        dm = _fl(ifirst, ilast, js - 2, je + 2)
        al = _fl(ifirst, ilast, js - 1, je + 2)
        bl = _fl(ifirst, ilast, js - 1, je + 1)
        br = _fl(ifirst, ilast, js - 1, je + 1)
        b0 = _fl(ifirst, ilast, js - 1, je + 1)
        dq = _fl(ifirst, ilast, js - 3, je + 2)
        smt5 = _flb(ifirst, ilast, js - 1, je + 1)
        fx1 = _fl1(ifirst, ilast)
        xt1 = _fl1(ifirst, ilast)
        a4 = _fl1(ifirst, ilast)
        hi5 = _fl1b(ifirst, ilast)
        hi6 = _fl1b(ifirst, ilast)

        for j in range(js - 2, je + 2 + 1):
            for i in range(ifirst, ilast + 1):
                xt = 0.25 * (q[i, j + 1] - q[i, j - 1])
                dm[i, j] = math.copysign(
                    min(abs(xt),
                        max(q[i, j - 1], q[i, j], q[i, j + 1]) - q[i, j],
                        q[i, j] - min(q[i, j - 1], q[i, j], q[i, j + 1])),
                    xt)
        for j in range(js1, je1 + 1 + 1):
            for i in range(ifirst, ilast + 1):
                al[i, j] = 0.5 * (q[i, j - 1] + q[i, j]) \
                    + R3 * (dm[i, j - 1] - dm[i, j])

        if jord == 8:
            for j in range(js1, je1 + 1):
                for i in range(ifirst, ilast + 1):
                    xt = 2.0 * dm[i, j]
                    bl[i, j] = -math.copysign(
                        min(abs(xt), abs(al[i, j] - q[i, j])), xt)
                    br[i, j] = math.copysign(
                        min(abs(xt), abs(al[i, j + 1] - q[i, j])), xt)
        elif jord == 10:
            for j in range(js1 - 2, je1 + 1 + 1):
                for i in range(ifirst, ilast + 1):
                    dq[i, j] = 2.0 * (q[i, j + 1] - q[i, j])
            for j in range(js1, je1 + 1):
                for i in range(ifirst, ilast + 1):
                    bl[i, j] = al[i, j] - q[i, j]
                    br[i, j] = al[i, j + 1] - q[i, j]
                    if abs(dm[i, j - 1]) + abs(dm[i, j]) \
                            + abs(dm[i, j + 1]) < NEAR_ZERO_TP:
                        bl[i, j] = 0.0
                        br[i, j] = 0.0
                    elif abs(3.0 * (bl[i, j] + br[i, j])) \
                            > abs(bl[i, j] - br[i, j]):
                        pmp_2 = dq[i, j - 1]
                        lac_2 = pmp_2 - 0.75 * dq[i, j - 2]
                        br[i, j] = min(max(0.0, pmp_2, lac_2),
                                       max(br[i, j],
                                           min(0.0, pmp_2, lac_2)))
                        pmp_1 = -dq[i, j]
                        lac_1 = pmp_1 + 0.75 * dq[i, j + 1]
                        bl[i, j] = min(max(0.0, pmp_1, lac_1),
                                       max(bl[i, j],
                                           min(0.0, pmp_1, lac_1)))
        elif jord == 11:
            for j in range(js1, je1 + 1):
                for i in range(ifirst, ilast + 1):
                    xt = PPM_FAC * dm[i, j]
                    bl[i, j] = -math.copysign(
                        min(abs(xt), abs(al[i, j] - q[i, j])), xt)
                    br[i, j] = math.copysign(
                        min(abs(xt), abs(al[i, j + 1] - q[i, j])), xt)
        elif jord == 7 or jord == 12:
            for j in range(js1, je1 + 1):
                for i in range(ifirst, ilast + 1):
                    bl[i, j] = al[i, j] - q[i, j]
                    br[i, j] = al[i, j + 1] - q[i, j]
                    xt1[i] = br[i, j] - bl[i, j]
                    a4[i] = -3.0 * (br[i, j] + bl[i, j])
                    hi5[i] = bl[i, j] * br[i, j] > 0.0
                    hi6[i] = abs(xt1[i]) < -a4[i]
                for i in range(ifirst, ilast + 1):
                    if hi6[i]:
                        if q[i, j] + 0.25 / a4[i] * xt1[i] ** 2 \
                                + a4[i] * R12 < 0.0:
                            if hi5[i]:
                                br[i, j] = 0.0
                                bl[i, j] = 0.0
                            elif xt1[i] > 0.0:
                                br[i, j] = -2.0 * bl[i, j]
                            else:
                                bl[i, j] = -2.0 * br[i, j]
        else:
            for j in range(js1, je1 + 1):
                for i in range(ifirst, ilast + 1):
                    bl[i, j] = al[i, j] - q[i, j]
                    br[i, j] = al[i, j + 1] - q[i, j]

        if jord == 9 or jord == 13:
            # Positive definite constraint:
            for j in range(js1, je1 + 1):
                pert_ppm(ilast - ifirst + 1,
                         _col(q, ifirst, ilast - ifirst + 1, j),
                         _col(bl, ifirst, ilast - ifirst + 1, j),
                         _col(br, ifirst, ilast - ifirst + 1, j), 0)

        if (not bounded_domain) and grid_type < 3:
            if js == 1:
                for i in range(ifirst, ilast + 1):
                    bl[i, 0] = TP_S14 * dm[i, -1] \
                        + TP_S11 * (q[i, -1] - q[i, 0])

                    xt = 0.5 * (
                        ((2.0 * dya[i, 0] + dya[i, -1]) * q[i, 0]
                         - dya[i, 0] * q[i, -1]) / (dya[i, -1] + dya[i, 0])
                        + ((2.0 * dya[i, 1] + dya[i, 2]) * q[i, 1]
                           - dya[i, 1] * q[i, 2]) / (dya[i, 1] + dya[i, 2]))
                    xt = max(xt, min(q[i, -1], q[i, 0], q[i, 1], q[i, 2]))
                    xt = min(xt, max(q[i, -1], q[i, 0], q[i, 1], q[i, 2]))
                    br[i, 0] = xt - q[i, 0]
                    bl[i, 1] = xt - q[i, 1]

                    xt = TP_S15 * q[i, 1] + TP_S11 * q[i, 2] \
                        - TP_S14 * dm[i, 2]
                    br[i, 1] = xt - q[i, 1]
                    bl[i, 2] = xt - q[i, 2]

                    br[i, 2] = al[i, 3] - q[i, 2]
                # Fortran: pert_ppm(3*(ilast-ifirst+1), q(ifirst,0),
                # bl(ifirst,0), br(ifirst,0), 1) — one F-contiguous run
                # over columns j=0,1,2; pert_ppm is element-wise, so
                # per-column calls are exactly equivalent.
                for jj in range(0, 2 + 1):
                    pert_ppm(ilast - ifirst + 1,
                             _col(q, ifirst, ilast - ifirst + 1, jj),
                             _col(bl, ifirst, ilast - ifirst + 1, jj),
                             _col(br, ifirst, ilast - ifirst + 1, jj), 1)
            if (je + 1) == npy:
                for i in range(ifirst, ilast + 1):
                    bl[i, npy - 2] = al[i, npy - 2] - q[i, npy - 2]

                    xt = TP_S15 * q[i, npy - 1] + TP_S11 * q[i, npy - 2] \
                        + TP_S14 * dm[i, npy - 2]
                    br[i, npy - 2] = xt - q[i, npy - 2]
                    bl[i, npy - 1] = xt - q[i, npy - 1]

                    xt = 0.5 * (
                        ((2.0 * dya[i, npy - 1] + dya[i, npy - 2])
                         * q[i, npy - 1]
                         - dya[i, npy - 1] * q[i, npy - 2])
                        / (dya[i, npy - 2] + dya[i, npy - 1])
                        + ((2.0 * dya[i, npy] + dya[i, npy + 1]) * q[i, npy]
                           - dya[i, npy] * q[i, npy + 1])
                        / (dya[i, npy] + dya[i, npy + 1]))
                    xt = max(xt, min(q[i, npy - 2], q[i, npy - 1],
                                     q[i, npy], q[i, npy + 1]))
                    xt = min(xt, max(q[i, npy - 2], q[i, npy - 1],
                                     q[i, npy], q[i, npy + 1]))
                    br[i, npy - 1] = xt - q[i, npy - 1]
                    bl[i, npy] = xt - q[i, npy]

                    br[i, npy] = TP_S11 * (q[i, npy + 1] - q[i, npy]) \
                        - TP_S14 * dm[i, npy + 1]
                # 3-column F-contiguous pert_ppm run — see js==1 note.
                for jj in range(npy - 2, npy + 1):
                    pert_ppm(ilast - ifirst + 1,
                             _col(q, ifirst, ilast - ifirst + 1, jj),
                             _col(bl, ifirst, ilast - ifirst + 1, jj),
                             _col(br, ifirst, ilast - ifirst + 1, jj), 1)

    if jord == 7:
        for j in range(js - 1, je + 1 + 1):
            for i in range(ifirst, ilast + 1):
                b0[i, j] = bl[i, j] + br[i, j]
                smt5[i, j] = bl[i, j] * br[i, j] < 0.0
        for j in range(js, je + 1 + 1):
            for i in range(ifirst, ilast + 1):
                if c[i, j] > 0.0:
                    fx1[i] = (1.0 - c[i, j]) * (br[i, j - 1]
                                                - c[i, j] * b0[i, j - 1])
                    flux[i, j] = q[i, j - 1]
                else:
                    fx1[i] = (1.0 + c[i, j]) * (bl[i, j]
                                                + c[i, j] * b0[i, j])
                    flux[i, j] = q[i, j]
                if smt5[i, j - 1] or smt5[i, j]:
                    flux[i, j] = flux[i, j] + fx1[i]
    else:
        for j in range(js, je + 1 + 1):
            for i in range(ifirst, ilast + 1):
                if c[i, j] > 0.0:
                    flux[i, j] = q[i, j - 1] + (1.0 - c[i, j]) * (
                        br[i, j - 1]
                        - c[i, j] * (bl[i, j - 1] + br[i, j - 1]))
                else:
                    flux[i, j] = q[i, j] + (1.0 + c[i, j]) * (
                        bl[i, j] + c[i, j] * (bl[i, j] + br[i, j]))


def deln_flux(nord: int, is_: int, ie: int, js: int, je: int,
              npx: int, npy: int, damp: float, q: fort,
              fx: fort, fy: fort, gridstruct: dict, bd: Bounds,
              mass: fort | None = None,
              damp_km: fort | None = None) -> None:
    """tp_core.F90 deln_flux (verbatim; non-USE_SG branch).

    Del-n damping for the cell-mean values (A grid):
    nord=0 del-2, nord=1 del-4, nord=2 del-6.
    q ghosted on input; fx/fy updated in place.
    """
    isd, ied = bd.isd, bd.ied
    jsd, jed = bd.jsd, bd.jed

    fx2 = _fl(isd, ied + 1, jsd, jed)
    fy2 = _fl(isd, ied, jsd, jed + 1)
    d2 = _fl(isd, ied, jsd, jed)

    DEL6_V = gridstruct["del6_v"]
    DEL6_U = gridstruct["del6_u"]
    RAREA = gridstruct["rarea"]

    i1 = is_ - 1 - nord
    i2 = ie + 1 + nord
    j1 = js - 1 - nord
    j2 = je + 1 + nord

    if (mass is None) and (damp_km is None):
        for j in range(j1, j2 + 1):
            for i in range(i1, i2 + 1):
                d2[i, j] = damp * q[i, j]
    else:
        for j in range(j1, j2 + 1):
            for i in range(i1, i2 + 1):
                d2[i, j] = q[i, j]

    if nord > 0:
        copy_corners(d2, npx, npy, 1, gridstruct["bounded_domain"], bd,
                     gridstruct["sw_corner"], gridstruct["se_corner"],
                     gridstruct["nw_corner"], gridstruct["ne_corner"])

    for j in range(js - nord, je + nord + 1):
        for i in range(is_ - nord, ie + nord + 1 + 1):
            fx2[i, j] = DEL6_V[i, j] * (d2[i - 1, j] - d2[i, j])

    if nord > 0:
        copy_corners(d2, npx, npy, 2, gridstruct["bounded_domain"], bd,
                     gridstruct["sw_corner"], gridstruct["se_corner"],
                     gridstruct["nw_corner"], gridstruct["ne_corner"])
    for j in range(js - nord, je + nord + 1 + 1):
        for i in range(is_ - nord, ie + nord + 1):
            fy2[i, j] = DEL6_U[i, j] * (d2[i, j - 1] - d2[i, j])

    if nord > 0:

        # ----------
        # high-order
        # ----------

        for n in range(1, nord + 1):

            nt = nord - n

            for j in range(js - nt - 1, je + nt + 1 + 1):
                for i in range(is_ - nt - 1, ie + nt + 1 + 1):
                    d2[i, j] = (fx2[i, j] - fx2[i + 1, j]
                                + fy2[i, j] - fy2[i, j + 1]) * RAREA[i, j]

            copy_corners(d2, npx, npy, 1, gridstruct["bounded_domain"], bd,
                         gridstruct["sw_corner"], gridstruct["se_corner"],
                         gridstruct["nw_corner"], gridstruct["ne_corner"])
            for j in range(js - nt, je + nt + 1):
                for i in range(is_ - nt, ie + nt + 1 + 1):
                    fx2[i, j] = DEL6_V[i, j] * (d2[i, j] - d2[i - 1, j])

            copy_corners(d2, npx, npy, 2, gridstruct["bounded_domain"], bd,
                         gridstruct["sw_corner"], gridstruct["se_corner"],
                         gridstruct["nw_corner"], gridstruct["ne_corner"])
            for j in range(js - nt, je + nt + 1 + 1):
                for i in range(is_ - nt, ie + nt + 1):
                    fy2[i, j] = DEL6_U[i, j] * (d2[i, j] - d2[i, j - 1])

    # ---------------------------------------------
    # Add the diffusive fluxes to the flux arrays:
    # ---------------------------------------------

    if mass is not None:
        # Apply mass weighting to diffusive fluxes:
        if damp_km is not None:
            for j in range(js, je + 1):
                for i in range(is_, ie + 1 + 1):
                    damp2 = 0.25 * damp * (damp_km[i - 1, j]
                                           + damp_km[i, j])
                    fx[i, j] = fx[i, j] + damp2 * (mass[i - 1, j]
                                                   + mass[i, j]) * fx2[i, j]
            for j in range(js, je + 1 + 1):
                for i in range(is_, ie + 1):
                    damp2 = 0.25 * damp * (damp_km[i, j - 1]
                                           + damp_km[i, j])
                    fy[i, j] = fy[i, j] + damp2 * (mass[i, j - 1]
                                                   + mass[i, j]) * fy2[i, j]
        else:
            damp2 = 0.5 * damp
            for j in range(js, je + 1):
                for i in range(is_, ie + 1 + 1):
                    fx[i, j] = fx[i, j] + damp2 * (mass[i - 1, j]
                                                   + mass[i, j]) * fx2[i, j]
            for j in range(js, je + 1 + 1):
                for i in range(is_, ie + 1):
                    fy[i, j] = fy[i, j] + damp2 * (mass[i, j - 1]
                                                   + mass[i, j]) * fy2[i, j]
    else:
        if damp_km is not None:
            for j in range(js, je + 1):
                for i in range(is_, ie + 1 + 1):
                    damp2 = 0.25 * damp * (damp_km[i - 1, j]
                                           + damp_km[i, j])
                    fx[i, j] = fx[i, j] + damp2 * fx2[i, j]
            for j in range(js, je + 1 + 1):
                for i in range(is_, ie + 1):
                    damp2 = 0.25 * damp * (damp_km[i, j - 1]
                                           + damp_km[i, j])
                    fy[i, j] = fy[i, j] + damp2 * fy2[i, j]
        else:
            for j in range(js, je + 1):
                for i in range(is_, ie + 1 + 1):
                    fx[i, j] = fx[i, j] + fx2[i, j]
            for j in range(js, je + 1 + 1):
                for i in range(is_, ie + 1):
                    fy[i, j] = fy[i, j] + fy2[i, j]


def fv_tp_2d(q: fort, crx: fort, cry: fort, npx: int, npy: int, hord: int,
             fx: fort, fy: fort, xfx: fort, yfx: fort,
             gridstruct: dict, bd: Bounds, ra_x: fort, ra_y: fort,
             lim_fac: float,
             mfx: fort | None = None, mfy: fort | None = None,
             mass: fort | None = None, nord: int | None = None,
             damp_c: float | None = None,
             damp_smag: float | None = None,
             damp_km: fort | None = None) -> None:
    """tp_core.F90 fv_tp_2d (verbatim).

    q(isd:ied, jsd:jed) INOUT (corner ghosts mutated via copy_corners!);
    crx/xfx(is:ie+1, jsd:jed); cry/yfx(isd:ied, js:je+1);
    fx(is:ie+1, js:je) / fy(is:ie, js:je+1) OUT.
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed

    DXA = gridstruct["dxa"]
    DYA = gridstruct["dya"]
    AREA = gridstruct["area"]
    bounded_domain = gridstruct["bounded_domain"]
    grid_type = gridstruct["grid_type"]

    q_i = _fl(isd, ied, js, je)
    q_j = _fl(is_, ie, jsd, jed)
    fx2 = _fl(is_, ie + 1, jsd, jed)
    fy2 = _fl(isd, ied, js, je + 1)
    fyy = _fl(isd, ied, js, je + 1)
    fx1 = _fl1(is_, ie + 1)

    if hord == 10:
        ord_in = 8
    else:
        ord_in = hord
    ord_ou = hord

    if not bounded_domain:
        copy_corners(q, npx, npy, 2, bounded_domain, bd,
                     gridstruct["sw_corner"], gridstruct["se_corner"],
                     gridstruct["nw_corner"], gridstruct["ne_corner"])

    yppm(fy2, q, cry, ord_in, isd, ied, isd, ied, js, je, jsd, jed,
         npx, npy, DYA, bounded_domain, grid_type, lim_fac)

    for j in range(js, je + 1 + 1):
        for i in range(isd, ied + 1):
            fyy[i, j] = yfx[i, j] * fy2[i, j]
    for j in range(js, je + 1):
        for i in range(isd, ied + 1):
            q_i[i, j] = (q[i, j] * AREA[i, j] + fyy[i, j]
                         - fyy[i, j + 1]) / ra_y[i, j]

    # Fortran passes crx(is,js): leading extents match, so the callee's
    # c(i,j) is crx(i,j) — the view passes through unchanged.
    xppm(fx, q_i, crx, ord_ou, is_, ie, isd, ied, js, je, jsd, jed,
         npx, npy, DXA, bounded_domain, grid_type, lim_fac)

    if not bounded_domain:
        copy_corners(q, npx, npy, 1, bounded_domain, bd,
                     gridstruct["sw_corner"], gridstruct["se_corner"],
                     gridstruct["nw_corner"], gridstruct["ne_corner"])

    xppm(fx2, q, crx, ord_in, is_, ie, isd, ied, jsd, jed, jsd, jed,
         npx, npy, DXA, bounded_domain, grid_type, lim_fac)

    for j in range(jsd, jed + 1):
        for i in range(is_, ie + 1 + 1):
            fx1[i] = xfx[i, j] * fx2[i, j]
        for i in range(is_, ie + 1):
            q_j[i, j] = (q[i, j] * AREA[i, j] + fx1[i]
                         - fx1[i + 1]) / ra_x[i, j]

    yppm(fy, q_j, cry, ord_ou, is_, ie, isd, ied, js, je, jsd, jed,
         npx, npy, DYA, bounded_domain, grid_type, lim_fac)

    # ----------------
    # Flux averaging:
    # ----------------

    if (mfx is not None) and (mfy is not None):
        # ---------------------------------
        # For transport of pt and tracers
        # ---------------------------------
        for j in range(js, je + 1):
            for i in range(is_, ie + 1 + 1):
                fx[i, j] = 0.5 * (fx[i, j] + fx2[i, j]) * mfx[i, j]
        for j in range(js, je + 1 + 1):
            for i in range(is_, ie + 1):
                fy[i, j] = 0.5 * (fy[i, j] + fy2[i, j]) * mfy[i, j]
        if (nord is not None) and (damp_c is not None) \
                and (mass is not None):
            if damp_c > 1.0e-4:
                damp = (damp_c * gridstruct["da_min"]) ** (nord + 1)
                deln_flux(nord, is_, ie, js, je, npx, npy, damp, q,
                          fx, fy, gridstruct, bd, mass=mass)
        if (damp_smag is not None) and (damp_km is not None) \
                and (mass is not None):
            if damp_smag > 1.0e-3:
                damp = damp_smag * gridstruct["da_min"]   # 2nd order
                deln_flux(0, is_, ie, js, je, npx, npy, damp, q,
                          fx, fy, gridstruct, bd, mass=mass,
                          damp_km=damp_km)
    else:
        # ---------------------------------
        # For transport of delp, vorticity
        # ---------------------------------
        for j in range(js, je + 1):
            for i in range(is_, ie + 1 + 1):
                fx[i, j] = 0.5 * (fx[i, j] + fx2[i, j]) * xfx[i, j]
        for j in range(js, je + 1 + 1):
            for i in range(is_, ie + 1):
                fy[i, j] = 0.5 * (fy[i, j] + fy2[i, j]) * yfx[i, j]
        if (nord is not None) and (damp_c is not None):
            if damp_c > 1.0e-4:
                damp = (damp_c * gridstruct["da_min"]) ** (nord + 1)
                deln_flux(nord, is_, ie, js, je, npx, npy, damp, q,
                          fx, fy, gridstruct, bd)
        if (damp_smag is not None) and (damp_km is not None):
            if damp_smag > 1.0e-3:
                damp = damp_smag * gridstruct["da_min"]   # 2nd order
                deln_flux(0, is_, ie, js, je, npx, npy, damp, q,
                          fx, fy, gridstruct, bd, damp_km=damp_km)


# =====================================================================
# a2b_edge_mod (ON the oracle path: dddmp=0.2 & nord>0 & grid_type<3
# reaches the a2b_ord4 call in d_sw's divergence-damping block)
# =====================================================================

def great_circle_dist(q1, q2, radius: float | None = None) -> float:
    """fv_grid_utils.F90 great_circle_dist (shim verbatim; f_p==double).

    q1/q2 are (lon, lat) pairs.
    """
    p1 = (q1[0], q1[1])
    p2 = (q2[0], q2[1])

    beta = math.asin(math.sqrt(
        math.sin((p1[1] - p2[1]) / 2.0) ** 2
        + math.cos(p1[1]) * math.cos(p2[1])
        * math.sin((p1[0] - p2[0]) / 2.0) ** 2)) * 2.0

    if radius is not None:
        return radius * beta
    else:
        return beta   # Returns the angle


def extrap_corner(p0, p1, p2, q1: float, q2: float) -> float:
    """a2b_edge.F90 extrap_corner (verbatim)."""
    x1 = great_circle_dist(p1, p0)
    x2 = great_circle_dist(p2, p0)

    return q1 + x1 / (x2 - x1) * (q1 - q2)


def a2b_ord4(qin: fort, qout: fort, gridstruct: dict, npx: int, npy: int,
             is_: int, ie: int, js: int, je: int, ng: int,
             replace: bool | None = None) -> None:
    """a2b_edge.F90 a2b_ord4 (verbatim, all branches).

    qin(is-ng:ie+ng, js-ng:je+ng) A-grid INOUT;
    qout same bounds, B-grid OUT (only edge/interior B nodes written).
    """
    # local: compact 4-pt cubic (FUNCTION-LOCAL c1/c2 — these SHADOW the
    # sw_core module c1/c2/c3; a classic transcription trap)
    c1 = 2.0 / 3.0
    c2 = -1.0 / 6.0

    GRID_LON = gridstruct["grid_lon"]
    GRID_LAT = gridstruct["grid_lat"]
    AGRID_LON = gridstruct["agrid_lon"]
    AGRID_LAT = gridstruct["agrid_lat"]
    DXA = gridstruct["dxa"]
    DYA = gridstruct["dya"]
    EDGE_W = gridstruct["edge_w"]
    EDGE_E = gridstruct["edge_e"]
    EDGE_S = gridstruct["edge_s"]
    EDGE_N = gridstruct["edge_n"]

    def _grid(i, j):
        return (GRID_LON[i, j], GRID_LAT[i, j])

    def _agrid(i, j):
        return (AGRID_LON[i, j], AGRID_LAT[i, j])

    qx = _fl(is_, ie + 1, js - ng, je + ng)
    qy = _fl(is_ - ng, ie + ng, js, je + 1)
    qxx = _fl(is_ - ng, ie + ng, js - ng, je + ng)
    qyy = _fl(is_ - ng, ie + ng, js - ng, je + ng)
    q1 = _fl1(is_ - 1, ie + 1)
    q2 = _fl1(js - 1, je + 1)

    if gridstruct["grid_type"] < 3:

        is1 = max(1, is_ - 1)
        js1 = max(1, js - 1)
        is2 = max(2, is_)
        js2 = max(2, js)

        ie1 = min(npx - 1, ie + 1)
        je1 = min(npy - 1, je + 1)

        # Corners:
        # 3-way extrapolation
        if gridstruct["bounded_domain"]:

            for j in range(js - 2, je + 2 + 1):
                for i in range(is_, ie + 1 + 1):
                    qx[i, j] = A2B_B2 * (qin[i - 2, j] + qin[i + 1, j]) \
                        + A2B_B1 * (qin[i - 1, j] + qin[i, j])

        else:

            if gridstruct["sw_corner"]:
                p0 = _grid(1, 1)
                qout[1, 1] = (
                    extrap_corner(p0, _agrid(1, 1), _agrid(2, 2),
                                  qin[1, 1], qin[2, 2])
                    + extrap_corner(p0, _agrid(0, 1), _agrid(-1, 2),
                                    qin[0, 1], qin[-1, 2])
                    + extrap_corner(p0, _agrid(1, 0), _agrid(2, -1),
                                    qin[1, 0], qin[2, -1])) * A2B_R3
            if gridstruct["se_corner"]:
                p0 = _grid(npx, 1)
                qout[npx, 1] = (
                    extrap_corner(p0, _agrid(npx - 1, 1),
                                  _agrid(npx - 2, 2),
                                  qin[npx - 1, 1], qin[npx - 2, 2])
                    + extrap_corner(p0, _agrid(npx - 1, 0),
                                    _agrid(npx - 2, -1),
                                    qin[npx - 1, 0], qin[npx - 2, -1])
                    + extrap_corner(p0, _agrid(npx, 1),
                                    _agrid(npx + 1, 2),
                                    qin[npx, 1], qin[npx + 1, 2])) * A2B_R3
            if gridstruct["ne_corner"]:
                p0 = _grid(npx, npy)
                qout[npx, npy] = (
                    extrap_corner(p0, _agrid(npx - 1, npy - 1),
                                  _agrid(npx - 2, npy - 2),
                                  qin[npx - 1, npy - 1],
                                  qin[npx - 2, npy - 2])
                    + extrap_corner(p0, _agrid(npx, npy - 1),
                                    _agrid(npx + 1, npy - 2),
                                    qin[npx, npy - 1],
                                    qin[npx + 1, npy - 2])
                    + extrap_corner(p0, _agrid(npx - 1, npy),
                                    _agrid(npx - 2, npy + 1),
                                    qin[npx - 1, npy],
                                    qin[npx - 2, npy + 1])) * A2B_R3
            if gridstruct["nw_corner"]:
                p0 = _grid(1, npy)
                qout[1, npy] = (
                    extrap_corner(p0, _agrid(1, npy - 1),
                                  _agrid(2, npy - 2),
                                  qin[1, npy - 1], qin[2, npy - 2])
                    + extrap_corner(p0, _agrid(0, npy - 1),
                                    _agrid(-1, npy - 2),
                                    qin[0, npy - 1], qin[-1, npy - 2])
                    + extrap_corner(p0, _agrid(1, npy),
                                    _agrid(2, npy + 1),
                                    qin[1, npy], qin[2, npy + 1])) * A2B_R3

            # ------------
            # X-Interior:
            # ------------
            for j in range(max(1, js - 2), min(npy - 1, je + 2) + 1):
                for i in range(max(3, is_), min(npx - 2, ie + 1) + 1):
                    qx[i, j] = A2B_B2 * (qin[i - 2, j] + qin[i + 1, j]) \
                        + A2B_B1 * (qin[i - 1, j] + qin[i, j])

            # *** West Edges:
            if is_ == 1:
                for j in range(js1, je1 + 1):
                    q2[j] = (qin[0, j] * DXA[1, j]
                             + qin[1, j] * DXA[0, j]) \
                        / (DXA[0, j] + DXA[1, j])
                for j in range(js2, je1 + 1):
                    qout[1, j] = EDGE_W[j] * q2[j - 1] \
                        + (1.0 - EDGE_W[j]) * q2[j]
                for j in range(max(1, js - 2), min(npy - 1, je + 2) + 1):
                    g_in = DXA[2, j] / DXA[1, j]
                    g_ou = DXA[-1, j] / DXA[0, j]
                    qx[1, j] = 0.5 * (
                        ((2.0 + g_in) * qin[1, j] - qin[2, j])
                        / (1.0 + g_in)
                        + ((2.0 + g_ou) * qin[0, j] - qin[-1, j])
                        / (1.0 + g_ou))
                    qx[2, j] = (3.0 * (g_in * qin[1, j] + qin[2, j])
                                - (g_in * qx[1, j] + qx[3, j])) \
                        / (2.0 + 2.0 * g_in)

            # East Edges:
            if (ie + 1) == npx:
                for j in range(js1, je1 + 1):
                    q2[j] = (qin[npx - 1, j] * DXA[npx, j]
                             + qin[npx, j] * DXA[npx - 1, j]) \
                        / (DXA[npx - 1, j] + DXA[npx, j])
                for j in range(js2, je1 + 1):
                    qout[npx, j] = EDGE_E[j] * q2[j - 1] \
                        + (1.0 - EDGE_E[j]) * q2[j]
                for j in range(max(1, js - 2), min(npy - 1, je + 2) + 1):
                    g_in = DXA[npx - 2, j] / DXA[npx - 1, j]
                    g_ou = DXA[npx + 1, j] / DXA[npx, j]
                    qx[npx, j] = 0.5 * (
                        ((2.0 + g_in) * qin[npx - 1, j] - qin[npx - 2, j])
                        / (1.0 + g_in)
                        + ((2.0 + g_ou) * qin[npx, j] - qin[npx + 1, j])
                        / (1.0 + g_ou))
                    qx[npx - 1, j] = (3.0 * (qin[npx - 2, j]
                                             + g_in * qin[npx - 1, j])
                                      - (g_in * qx[npx, j]
                                         + qx[npx - 2, j])) \
                        / (2.0 + 2.0 * g_in)

        # ------------
        # Y-Interior:
        # ------------

        if gridstruct["bounded_domain"]:

            for j in range(js, je + 1 + 1):
                for i in range(is_ - 2, ie + 2 + 1):
                    qy[i, j] = A2B_B2 * (qin[i, j - 2] + qin[i, j + 1]) \
                        + A2B_B1 * (qin[i, j - 1] + qin[i, j])

        else:

            for j in range(max(3, js), min(npy - 2, je + 1) + 1):
                for i in range(max(1, is_ - 2), min(npx - 1, ie + 2) + 1):
                    qy[i, j] = A2B_B2 * (qin[i, j - 2] + qin[i, j + 1]) \
                        + A2B_B1 * (qin[i, j - 1] + qin[i, j])

            # South Edges:
            if js == 1:
                for i in range(is1, ie1 + 1):
                    q1[i] = (qin[i, 0] * DYA[i, 1]
                             + qin[i, 1] * DYA[i, 0]) \
                        / (DYA[i, 0] + DYA[i, 1])
                for i in range(is2, ie1 + 1):
                    qout[i, 1] = EDGE_S[i] * q1[i - 1] \
                        + (1.0 - EDGE_S[i]) * q1[i]
                for i in range(max(1, is_ - 2), min(npx - 1, ie + 2) + 1):
                    g_in = DYA[i, 2] / DYA[i, 1]
                    g_ou = DYA[i, -1] / DYA[i, 0]
                    qy[i, 1] = 0.5 * (
                        ((2.0 + g_in) * qin[i, 1] - qin[i, 2])
                        / (1.0 + g_in)
                        + ((2.0 + g_ou) * qin[i, 0] - qin[i, -1])
                        / (1.0 + g_ou))
                    qy[i, 2] = (3.0 * (g_in * qin[i, 1] + qin[i, 2])
                                - (g_in * qy[i, 1] + qy[i, 3])) \
                        / (2.0 + 2.0 * g_in)

            # North Edges:
            if (je + 1) == npy:
                for i in range(is1, ie1 + 1):
                    q1[i] = (qin[i, npy - 1] * DYA[i, npy]
                             + qin[i, npy] * DYA[i, npy - 1]) \
                        / (DYA[i, npy - 1] + DYA[i, npy])
                for i in range(is2, ie1 + 1):
                    qout[i, npy] = EDGE_N[i] * q1[i - 1] \
                        + (1.0 - EDGE_N[i]) * q1[i]
                for i in range(max(1, is_ - 2), min(npx - 1, ie + 2) + 1):
                    g_in = DYA[i, npy - 2] / DYA[i, npy - 1]
                    g_ou = DYA[i, npy + 1] / DYA[i, npy]
                    qy[i, npy] = 0.5 * (
                        ((2.0 + g_in) * qin[i, npy - 1] - qin[i, npy - 2])
                        / (1.0 + g_in)
                        + ((2.0 + g_ou) * qin[i, npy] - qin[i, npy + 1])
                        / (1.0 + g_ou))
                    qy[i, npy - 1] = (3.0 * (qin[i, npy - 2]
                                             + g_in * qin[i, npy - 1])
                                      - (g_in * qy[i, npy]
                                         + qy[i, npy - 2])) \
                        / (2.0 + 2.0 * g_in)

        # --------------------------------------

        if gridstruct["bounded_domain"]:

            for j in range(js, je + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    qxx[i, j] = A2B_A2 * (qx[i, j - 2] + qx[i, j + 1]) \
                        + A2B_A1 * (qx[i, j - 1] + qx[i, j])

            for j in range(js, je + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    qyy[i, j] = A2B_A2 * (qy[i - 2, j] + qy[i + 1, j]) \
                        + A2B_A1 * (qy[i - 1, j] + qy[i, j])

                for i in range(is_, ie + 1 + 1):
                    qout[i, j] = 0.5 * (qxx[i, j] + qyy[i, j])  # averaging

        else:

            for j in range(max(3, js), min(npy - 2, je + 1) + 1):
                for i in range(max(2, is_), min(npx - 1, ie + 1) + 1):
                    qxx[i, j] = A2B_A2 * (qx[i, j - 2] + qx[i, j + 1]) \
                        + A2B_A1 * (qx[i, j - 1] + qx[i, j])

            if js == 1:
                for i in range(max(2, is_), min(npx - 1, ie + 1) + 1):
                    qxx[i, 2] = c1 * (qx[i, 1] + qx[i, 2]) \
                        + c2 * (qout[i, 1] + qxx[i, 3])
            if (je + 1) == npy:
                for i in range(max(2, is_), min(npx - 1, ie + 1) + 1):
                    qxx[i, npy - 1] = c1 * (qx[i, npy - 2]
                                            + qx[i, npy - 1]) \
                        + c2 * (qout[i, npy] + qxx[i, npy - 2])

            for j in range(max(2, js), min(npy - 1, je + 1) + 1):
                for i in range(max(3, is_), min(npx - 2, ie + 1) + 1):
                    qyy[i, j] = A2B_A2 * (qy[i - 2, j] + qy[i + 1, j]) \
                        + A2B_A1 * (qy[i - 1, j] + qy[i, j])
                if is_ == 1:
                    qyy[2, j] = c1 * (qy[1, j] + qy[2, j]) \
                        + c2 * (qout[1, j] + qyy[3, j])
                if (ie + 1) == npx:
                    qyy[npx - 1, j] = c1 * (qy[npx - 2, j]
                                            + qy[npx - 1, j]) \
                        + c2 * (qout[npx, j] + qyy[npx - 2, j])

                for i in range(max(2, is_), min(npx - 1, ie + 1) + 1):
                    qout[i, j] = 0.5 * (qxx[i, j] + qyy[i, j])  # averaging

    else:  # grid_type>=3
        # ------------------------
        # Doubly periodic domain:
        # ------------------------
        # X-sweep: PPM
        for j in range(js - 2, je + 2 + 1):
            for i in range(is_, ie + 1 + 1):
                qx[i, j] = A2B_B1 * (qin[i - 1, j] + qin[i, j]) \
                    + A2B_B2 * (qin[i - 2, j] + qin[i + 1, j])
        # Y-sweep: PPM
        for j in range(js, je + 1 + 1):
            for i in range(is_ - 2, ie + 2 + 1):
                qy[i, j] = A2B_B1 * (qin[i, j - 1] + qin[i, j]) \
                    + A2B_B2 * (qin[i, j - 2] + qin[i, j + 1])

        for j in range(js, je + 1 + 1):
            for i in range(is_, ie + 1 + 1):
                qout[i, j] = 0.5 * (
                    A2B_A1 * (qx[i, j - 1] + qx[i, j]
                              + qy[i - 1, j] + qy[i, j])
                    + A2B_A2 * (qx[i, j - 2] + qx[i, j + 1]
                                + qy[i - 2, j] + qy[i + 1, j]))

    if replace is not None:
        if replace:
            for j in range(js, je + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    qin[i, j] = qout[i, j]


# =====================================================================
# dsw_extract_mod
# =====================================================================

def smag_corner(dt: float, u: fort, v: fort, ua: fort, va: fort,
                smag_c: fort, bd: Bounds, npx: int, npy: int,
                gridstruct: dict, ng: int) -> None:
    """sw_core.F90 smag_corner — NOT PORTED.

    Its only call site is the grid_type>=3 arm of d_sw's divergence
    damping (doubly periodic domains); the oracle is grid_type=0.
    """
    raise NotImplementedError(
        "smag_corner not ported: grid_type>=3 only "
        "(oracle runs grid_type=0, d_con=0, do_diss_est=False)")


def del6_vt_flux(nord: int, npx: int, npy: int, damp: float,
                 q: fort, d2: fort, fx2: fort, fy2: fort,
                 gridstruct: dict, bd: Bounds,
                 damp_km: fort | None = None) -> None:
    """sw_core.F90 del6_vt_flux (verbatim; non-USE_SG branch).

    Del-nord damping for the relative vorticity (nord <= 2); like
    tp_core deln_flux but does NOT add the fluxes into regular fluxes.
    q(isd:ied, jsd:jed) INOUT (ghosted); d2/fx2/fy2 are caller work
    arrays (d_sw passes vort/ut/vt!) mutated in place.
    """
    bounded_domain = gridstruct["bounded_domain"]

    DEL6_V = gridstruct["del6_v"]
    DEL6_U = gridstruct["del6_u"]
    RAREA = gridstruct["rarea"]

    is_, ie = bd.is_, bd.ie
    js, je = bd.js, bd.je

    i1 = is_ - 1 - nord
    i2 = ie + 1 + nord
    j1 = js - 1 - nord
    j2 = je + 1 + nord

    for j in range(j1, j2 + 1):
        for i in range(i1, i2 + 1):
            d2[i, j] = damp * q[i, j]

    if nord > 0 and (not bounded_domain):
        copy_corners(d2, npx, npy, 1, bounded_domain, bd,
                     gridstruct["sw_corner"], gridstruct["se_corner"],
                     gridstruct["nw_corner"], gridstruct["ne_corner"])
    for j in range(js - nord, je + nord + 1):
        for i in range(is_ - nord, ie + nord + 1 + 1):
            fx2[i, j] = DEL6_V[i, j] * (d2[i - 1, j] - d2[i, j])

    if nord > 0 and (not bounded_domain):
        copy_corners(d2, npx, npy, 2, bounded_domain, bd,
                     gridstruct["sw_corner"], gridstruct["se_corner"],
                     gridstruct["nw_corner"], gridstruct["ne_corner"])
    for j in range(js - nord, je + nord + 1 + 1):
        for i in range(is_ - nord, ie + nord + 1):
            fy2[i, j] = DEL6_U[i, j] * (d2[i, j - 1] - d2[i, j])

    if nord > 0:
        for n in range(1, nord + 1):
            nt = nord - n
            for j in range(js - nt - 1, je + nt + 1 + 1):
                for i in range(is_ - nt - 1, ie + nt + 1 + 1):
                    d2[i, j] = (fx2[i, j] - fx2[i + 1, j]
                                + fy2[i, j] - fy2[i, j + 1]) * RAREA[i, j]

            if not bounded_domain:
                copy_corners(d2, npx, npy, 1, bounded_domain, bd,
                             gridstruct["sw_corner"],
                             gridstruct["se_corner"],
                             gridstruct["nw_corner"],
                             gridstruct["ne_corner"])

            for j in range(js - nt, je + nt + 1):
                for i in range(is_ - nt, ie + nt + 1 + 1):
                    fx2[i, j] = DEL6_V[i, j] * (d2[i, j] - d2[i - 1, j])

            if not bounded_domain:
                copy_corners(d2, npx, npy, 2, bounded_domain, bd,
                             gridstruct["sw_corner"],
                             gridstruct["se_corner"],
                             gridstruct["nw_corner"],
                             gridstruct["ne_corner"])

            for j in range(js - nt, je + nt + 1 + 1):
                for i in range(is_ - nt, ie + nt + 1):
                    fy2[i, j] = DEL6_U[i, j] * (d2[i, j] - d2[i, j - 1])

    if damp_km is not None:   # Coefficient multiplied in earlier
        for j in range(js, je + 1):
            for i in range(is_, ie + 1 + 1):
                fx2[i, j] = fx2[i, j] * 0.5 * damp_km[i, j]
        for j in range(js, je + 1 + 1):
            for i in range(is_, ie + 1):
                fy2[i, j] = fy2[i, j] * 0.5 * damp_km[i, j]


# ---------------------------------------------------------------------
# fill_corners bodies (tools/fv_mp_mod.F90 r8 variants, verbatim; the
# Fortran reads module-scope is/ie/js/je/ng set from the same bd)
# ---------------------------------------------------------------------

def fill_corners_2d(q: fort, npx: int, npy: int, fill: int, bd: Bounds,
                    agrid: bool | None = None,
                    bgrid: bool | None = None) -> None:
    """fill_corners_2d_r8 (verbatim)."""
    is_, ie, js, je, ng = bd.is_, bd.ie, bd.js, bd.je, bd.ng

    if bgrid is not None:
        if bgrid:
            if fill == XDIR:
                for j in range(1, ng + 1):
                    for i in range(1, ng + 1):
                        if (is_ == 1) and (js == 1):          # SW
                            q[1 - i, 1 - j] = q[1 - j, i + 1]
                        if (is_ == 1) and (je == npy - 1):    # NW
                            q[1 - i, npy + j] = q[1 - j, npy - i]
                        if (ie == npx - 1) and (js == 1):     # SE
                            q[npx + i, 1 - j] = q[npx + j, i + 1]
                        if (ie == npx - 1) and (je == npy - 1):   # NE
                            q[npx + i, npy + j] = q[npx + j, npy - i]
            elif fill == YDIR:
                for j in range(1, ng + 1):
                    for i in range(1, ng + 1):
                        if (is_ == 1) and (js == 1):          # SW
                            q[1 - j, 1 - i] = q[i + 1, 1 - j]
                        if (is_ == 1) and (je == npy - 1):    # NW
                            q[1 - j, npy + i] = q[i + 1, npy + j]
                        if (ie == npx - 1) and (js == 1):     # SE
                            q[npx + j, 1 - i] = q[npx - i, 1 - j]
                        if (ie == npx - 1) and (je == npy - 1):   # NE
                            q[npx + j, npy + i] = q[npx - i, npy + j]
            else:   # case default (same body as XDir)
                for j in range(1, ng + 1):
                    for i in range(1, ng + 1):
                        if (is_ == 1) and (js == 1):          # SW
                            q[1 - i, 1 - j] = q[1 - j, i + 1]
                        if (is_ == 1) and (je == npy - 1):    # NW
                            q[1 - i, npy + j] = q[1 - j, npy - i]
                        if (ie == npx - 1) and (js == 1):     # SE
                            q[npx + i, 1 - j] = q[npx + j, i + 1]
                        if (ie == npx - 1) and (je == npy - 1):   # NE
                            q[npx + i, npy + j] = q[npx + j, npy - i]
    elif agrid is not None:
        if agrid:
            if fill == XDIR:
                for j in range(1, ng + 1):
                    for i in range(1, ng + 1):
                        if (is_ == 1) and (js == 1):          # SW
                            q[1 - i, 1 - j] = q[1 - j, i]
                        if (is_ == 1) and (je == npy - 1):    # NW
                            q[1 - i, npy - 1 + j] = q[1 - j, npy - 1 - i + 1]
                        if (ie == npx - 1) and (js == 1):     # SE
                            q[npx - 1 + i, 1 - j] = q[npx - 1 + j, i]
                        if (ie == npx - 1) and (je == npy - 1):   # NE
                            q[npx - 1 + i, npy - 1 + j] = \
                                q[npx - 1 + j, npy - 1 - i + 1]
            elif fill == YDIR:
                for j in range(1, ng + 1):
                    for i in range(1, ng + 1):
                        if (is_ == 1) and (js == 1):          # SW
                            q[1 - j, 1 - i] = q[i, 1 - j]
                        if (is_ == 1) and (je == npy - 1):    # NW
                            q[1 - j, npy - 1 + i] = q[i, npy - 1 + j]
                        if (ie == npx - 1) and (js == 1):     # SE
                            q[npx - 1 + j, 1 - i] = q[npx - 1 - i + 1, 1 - j]
                        if (ie == npx - 1) and (je == npy - 1):   # NE
                            q[npx - 1 + j, npy - 1 + i] = \
                                q[npx - 1 - i + 1, npy - 1 + j]
            else:   # case default (same body as YDir)
                for j in range(1, ng + 1):
                    for i in range(1, ng + 1):
                        if (is_ == 1) and (js == 1):          # SW
                            q[1 - j, 1 - i] = q[i, 1 - j]
                        if (is_ == 1) and (je == npy - 1):    # NW
                            q[1 - j, npy - 1 + i] = q[i, npy - 1 + j]
                        if (ie == npx - 1) and (js == 1):     # SE
                            q[npx - 1 + j, 1 - i] = q[npx - 1 - i + 1, 1 - j]
                        if (ie == npx - 1) and (je == npy - 1):   # NE
                            q[npx - 1 + j, npy - 1 + i] = \
                                q[npx - 1 - i + 1, npy - 1 + j]


def fill_corners_dgrid(x: fort, y: fort, npx: int, npy: int,
                       mysign: float, bd: Bounds) -> None:
    """fill_corners_dgrid_r8 (verbatim, active non-commented lines)."""
    is_, ie, js, je, ng = bd.is_, bd.ie, bd.js, bd.je, bd.ng

    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            if (is_ == 1) and (js == 1):        # SW
                x[1 - i, 1 - j] = mysign * y[1 - j, i]
            if (is_ == 1) and (je + 1 == npy):  # NW
                x[1 - i, npy + j] = y[1 - j, npy - i]
            if (ie + 1 == npx) and (js == 1):   # SE
                x[npx - 1 + i, 1 - j] = y[npx + j, i]
            if (ie + 1 == npx) and (je + 1 == npy):   # NE
                x[npx - 1 + i, npy + j] = mysign * y[npx + j, npy - i]
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            if (is_ == 1) and (js == 1):        # SW
                y[1 - i, 1 - j] = mysign * x[j, 1 - i]
            if (is_ == 1) and (je + 1 == npy):  # NW
                y[1 - i, npy - 1 + j] = x[j, npy + i]
            if (ie + 1 == npx) and (js == 1):   # SE
                y[npx + i, 1 - j] = x[npx - j, 1 - i]
            if (ie + 1 == npx) and (je + 1 == npy):   # NE
                y[npx + i, npy - 1 + j] = mysign * x[npx - j, npy + i]


def fill_corners_cgrid(x: fort, y: fort, npx: int, npy: int,
                       mysign: float, bd: Bounds) -> None:
    """fill_corners_cgrid_r8 (verbatim)."""
    is_, ie, js, je, ng = bd.is_, bd.ie, bd.js, bd.je, bd.ng

    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            if (is_ == 1) and (js == 1):        # SW
                x[1 - i, 1 - j] = y[j, 1 - i]
            if (is_ == 1) and (je + 1 == npy):  # NW
                x[1 - i, npy - 1 + j] = mysign * y[j, npy + i]
            if (ie + 1 == npx) and (js == 1):   # SE
                x[npx + i, 1 - j] = mysign * y[npx - j, 1 - i]
            if (ie + 1 == npx) and (je + 1 == npy):   # NE
                x[npx + i, npy - 1 + j] = y[npx - j, npy + i]
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            if (is_ == 1) and (js == 1):        # SW
                y[1 - i, 1 - j] = x[1 - j, i]
            if (is_ == 1) and (je + 1 == npy):  # NW
                y[1 - i, npy + j] = mysign * x[1 - j, npy - i]
            if (ie + 1 == npx) and (js == 1):   # SE
                y[npx - 1 + i, 1 - j] = mysign * x[npx + j, i]
            if (ie + 1 == npx) and (je + 1 == npy):   # NE
                y[npx - 1 + i, npy + j] = x[npx + j, npy - i]


def fill_corners_agrid(x: fort, y: fort, npx: int, npy: int,
                       mysign: float, bd: Bounds) -> None:
    """fill_corners_agrid_r8 (verbatim)."""
    is_, ie, js, je, ng = bd.is_, bd.ie, bd.js, bd.je, bd.ng

    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            if (is_ == 1) and (js == 1):          # SW
                x[1 - i, 1 - j] = mysign * y[1 - j, i]
            if (is_ == 1) and (je == npy - 1):    # NW
                x[1 - i, npy - 1 + j] = y[1 - j, npy - 1 - i + 1]
            if (ie == npx - 1) and (js == 1):     # SE
                x[npx - 1 + i, 1 - j] = y[npx - 1 + j, i]
            if (ie == npx - 1) and (je == npy - 1):   # NE
                x[npx - 1 + i, npy - 1 + j] = \
                    mysign * y[npx - 1 + j, npy - 1 - i + 1]
    for j in range(1, ng + 1):
        for i in range(1, ng + 1):
            if (is_ == 1) and (js == 1):          # SW
                y[1 - j, 1 - i] = mysign * x[i, 1 - j]
            if (is_ == 1) and (je == npy - 1):    # NW
                y[1 - j, npy - 1 + i] = x[i, npy - 1 + j]
            if (ie == npx - 1) and (js == 1):     # SE
                y[npx - 1 + j, 1 - i] = x[npx - 1 - i + 1, 1 - j]
            if (ie == npx - 1) and (je == npy - 1):   # NE
                y[npx - 1 + j, npy - 1 + i] = \
                    mysign * x[npx - 1 - i + 1, npy - 1 + j]


def fill_corners_xy_2d(x: fort, y: fort, npx: int, npy: int, bd: Bounds,
                       dgrid: bool | None = None,
                       agrid: bool | None = None,
                       cgrid: bool | None = None,
                       vector: bool | None = None) -> None:
    """fill_corners_xy_2d_r8 (verbatim dispatch)."""
    mysign = 1.0
    if vector is not None:
        if vector:
            mysign = -1.0

    if dgrid is not None:
        fill_corners_dgrid(x, y, npx, npy, mysign, bd)
    elif cgrid is not None:
        fill_corners_cgrid(x, y, npx, npy, mysign, bd)
    elif agrid is not None:
        fill_corners_agrid(x, y, npx, npy, mysign, bd)
    else:
        fill_corners_agrid(x, y, npx, npy, mysign, bd)



# =====================================================================
# sw_core_mod: xtp_u / ytp_v  (D-grid momentum transport, PPM)
# =====================================================================

def xtp_u(is_, ie, js, je, isd, ied, jsd, jed, c, u, v, flux, iord,
          dx, rdx, npx, npy, grid_type, bounded_domain, lim_fac):
    """sw_core.F90 xtp_u (verbatim, all iord branches).

    c/flux fort (is:ie+1, js:je+1); u fort (isd:ied, jsd:jed+1);
    v fort (isd:ied+1, jsd:jed); dx/rdx fort (isd:ied, jsd:jed+1).
    Mutates flux.
    """
    if bounded_domain or grid_type > 3:
        is3 = is_ - 1
        ie3 = ie + 1
    else:
        is3 = max(3, is_ - 1)
        ie3 = min(npx - 3, ie + 1)

    if iord < 8:
        for j in range(js, je + 1 + 1):
            al = _fl1(is_ - 1, ie + 2)
            for i in range(is3, ie3 + 1 + 1):
                al[i] = SW_P1 * (u[i - 1, j] + u[i, j]) \
                    + SW_P2 * (u[i - 2, j] + u[i + 1, j])
            bl = _fl1(is_ - 1, ie + 1)
            br = _fl1(is_ - 1, ie + 1)
            for i in range(is3, ie3 + 1):
                bl[i] = al[i] - u[i, j]
                br[i] = al[i + 1] - u[i, j]

            if (not bounded_domain) and grid_type < 3:
                if is_ == 1:
                    xt = SW_C3 * u[1, j] + SW_C2 * u[2, j] + SW_C1 * u[3, j]
                    br[1] = xt - u[1, j]
                    bl[2] = xt - u[2, j]
                    br[2] = al[3] - u[2, j]
                    if j == 1 or j == npy:
                        bl[0] = 0.0
                        br[0] = 0.0
                        bl[1] = 0.0
                        br[1] = 0.0
                    else:
                        bl[0] = SW_C1 * u[-2, j] + SW_C2 * u[-1, j] \
                            + SW_C3 * u[0, j] - u[0, j]
                        xt = 0.5 * (
                            ((2.0 * dx[0, j] + dx[-1, j]) * (u[0, j])
                             - dx[0, j] * u[-1, j]) / (dx[0, j] + dx[-1, j])
                            + ((2.0 * dx[1, j] + dx[2, j]) * (u[1, j])
                               - dx[1, j] * u[2, j]) / (dx[1, j] + dx[2, j]))
                        br[0] = xt - u[0, j]
                        bl[1] = xt - u[1, j]
                if (ie + 1) == npx:
                    bl[npx - 2] = al[npx - 2] - u[npx - 2, j]
                    xt = SW_C1 * u[npx - 3, j] + SW_C2 * u[npx - 2, j] \
                        + SW_C3 * u[npx - 1, j]
                    br[npx - 2] = xt - u[npx - 2, j]
                    bl[npx - 1] = xt - u[npx - 1, j]
                    if j == 1 or j == npy:
                        bl[npx - 1] = 0.0
                        br[npx - 1] = 0.0
                        bl[npx] = 0.0
                        br[npx] = 0.0
                    else:
                        xt = 0.5 * (
                            ((2.0 * dx[npx - 1, j] + dx[npx - 2, j])
                             * u[npx - 1, j] - dx[npx - 1, j] * u[npx - 2, j])
                            / (dx[npx - 1, j] + dx[npx - 2, j])
                            + ((2.0 * dx[npx, j] + dx[npx + 1, j])
                               * u[npx, j] - dx[npx, j] * u[npx + 1, j])
                            / (dx[npx, j] + dx[npx + 1, j]))
                        br[npx - 1] = xt - u[npx - 1, j]
                        bl[npx] = xt - u[npx, j]
                        br[npx] = SW_C3 * u[npx, j] + SW_C2 * u[npx + 1, j] \
                            + SW_C1 * u[npx + 2, j] - u[npx, j]

            b0 = _fl1(is_ - 1, ie + 1)
            for i in range(is_ - 1, ie + 1 + 1):
                b0[i] = bl[i] + br[i]

            fx0 = _fl1(is_, ie + 1)
            if iord == 1:
                smt5 = _fl1b(is_ - 1, ie + 1)
                for i in range(is_ - 1, ie + 1 + 1):
                    smt5[i] = abs(lim_fac * b0[i]) < abs(bl[i] - br[i])
                for i in range(is_, ie + 1 + 1):
                    if c[i, j] > 0.0:
                        cfl = c[i, j] * rdx[i - 1, j]
                        fx0[i] = (1.0 - cfl) * (br[i - 1] - cfl * b0[i - 1])
                        flux[i, j] = u[i - 1, j]
                    else:
                        cfl = c[i, j] * rdx[i, j]
                        fx0[i] = (1.0 + cfl) * (bl[i] + cfl * b0[i])
                        flux[i, j] = u[i, j]
                    if smt5[i - 1] or smt5[i]:
                        flux[i, j] = flux[i, j] + fx0[i]

            elif iord == 2:
                for i in range(is_, ie + 1 + 1):
                    if c[i, j] > 0.0:
                        cfl = c[i, j] * rdx[i - 1, j]
                        flux[i, j] = u[i - 1, j] \
                            + (1.0 - cfl) * (br[i - 1] - cfl * b0[i - 1])
                    else:
                        cfl = c[i, j] * rdx[i, j]
                        flux[i, j] = u[i, j] \
                            + (1.0 + cfl) * (bl[i] + cfl * b0[i])

            elif iord == 3:
                smt5 = _fl1b(is_ - 1, ie + 1)
                smt6 = _fl1b(is_ - 1, ie + 1)
                for i in range(is_ - 1, ie + 1 + 1):
                    x0 = abs(b0[i])
                    x1 = abs(bl[i] - br[i])
                    smt5[i] = x0 < x1
                    smt6[i] = 3.0 * x0 < x1
                hi5 = _fl1b(is_, ie + 1)
                hi6 = _fl1b(is_, ie + 1)
                for i in range(is_, ie + 1 + 1):
                    fx0[i] = 0.0
                    hi5[i] = smt5[i - 1] and smt5[i]
                    hi6[i] = smt6[i - 1] or smt6[i]
                for i in range(is_, ie + 1 + 1):
                    if c[i, j] > 0.0:
                        cfl = c[i, j] * rdx[i - 1, j]
                        if hi6[i]:
                            fx0[i] = br[i - 1] - cfl * b0[i - 1]
                        elif hi5[i]:
                            fx0[i] = math.copysign(
                                min(abs(bl[i - 1]), abs(br[i - 1])), br[i - 1])
                        flux[i, j] = u[i - 1, j] + (1.0 - cfl) * fx0[i]
                    else:
                        cfl = c[i, j] * rdx[i, j]
                        if hi6[i]:
                            fx0[i] = bl[i] + cfl * b0[i]
                        elif hi5[i]:
                            fx0[i] = math.copysign(
                                min(abs(bl[i]), abs(br[i])), bl[i])
                        flux[i, j] = u[i, j] + (1.0 + cfl) * fx0[i]

            elif iord == 4:
                smt5 = _fl1b(is_ - 1, ie + 1)
                smt6 = _fl1b(is_ - 1, ie + 1)
                for i in range(is_ - 1, ie + 1 + 1):
                    x0 = abs(b0[i])
                    x1 = abs(bl[i] - br[i])
                    smt5[i] = x0 < x1
                    smt6[i] = 3.0 * x0 < x1
                hi5 = _fl1b(is_, ie + 1)
                hi6 = _fl1b(is_, ie + 1)
                for i in range(is_, ie + 1 + 1):
                    hi5[i] = smt5[i - 1] and smt5[i]
                    hi6[i] = smt6[i - 1] or smt6[i]
                    hi5[i] = hi5[i] or hi6[i]
                for i in range(is_, ie + 1 + 1):
                    if c[i, j] > 0.0:
                        cfl = c[i, j] * rdx[i - 1, j]
                        fx0[i] = (1.0 - cfl) * (br[i - 1] - cfl * b0[i - 1])
                        flux[i, j] = u[i - 1, j]
                    else:
                        cfl = c[i, j] * rdx[i, j]
                        fx0[i] = (1.0 + cfl) * (bl[i] + cfl * b0[i])
                        flux[i, j] = u[i, j]
                    if hi5[i]:
                        flux[i, j] = flux[i, j] + fx0[i]

            else:      # iord = 5, 6, 7
                smt5 = _fl1b(is_ - 1, ie + 1)
                if iord == 5:
                    for i in range(is_ - 1, ie + 1 + 1):
                        smt5[i] = bl[i] * br[i] < 0.0
                else:
                    for i in range(is_ - 1, ie + 1 + 1):
                        smt5[i] = 3.0 * abs(b0[i]) < abs(bl[i] - br[i])
                    if (not bounded_domain) and grid_type < 3:
                        if is_ == 1:
                            smt5[0] = bl[0] * br[0] < 0.0
                            smt5[1] = bl[1] * br[1] < 0.0
                        if (ie + 1) == npx:
                            smt5[npx - 1] = bl[npx - 1] * br[npx - 1] < 0.0
                            smt5[npx] = bl[npx] * br[npx] < 0.0
                for i in range(is_, ie + 1 + 1):
                    if c[i, j] > 0.0:
                        cfl = c[i, j] * rdx[i - 1, j]
                        fx0[i] = (1.0 - cfl) * (br[i - 1] - cfl * b0[i - 1])
                        flux[i, j] = u[i - 1, j]
                    else:
                        cfl = c[i, j] * rdx[i, j]
                        fx0[i] = (1.0 + cfl) * (bl[i] + cfl * b0[i])
                        flux[i, j] = u[i, j]
                    if smt5[i - 1] or smt5[i]:
                        flux[i, j] = flux[i, j] + fx0[i]

    else:      # iord = 8, 9, 10, 11
        for j in range(js, je + 1 + 1):
            dm = _fl1(is_ - 2, ie + 2)
            for i in range(is_ - 2, ie + 2 + 1):
                xt = 0.25 * (u[i + 1, j] - u[i - 1, j])
                dm[i] = math.copysign(
                    min(abs(xt),
                        max(u[i - 1, j], u[i, j], u[i + 1, j]) - u[i, j],
                        u[i, j] - min(u[i - 1, j], u[i, j], u[i + 1, j])), xt)
            dq = _fl1(is_ - 3, ie + 2)
            for i in range(is_ - 3, ie + 2 + 1):
                dq[i] = u[i + 1, j] - u[i, j]

            al = _fl1(is_ - 1, ie + 2)
            bl = _fl1(is_ - 1, ie + 1)
            br = _fl1(is_ - 1, ie + 1)
            if grid_type < 3:
                for i in range(is3, ie3 + 1 + 1):
                    al[i] = 0.5 * (u[i - 1, j] + u[i, j]) \
                        + R3 * (dm[i - 1] - dm[i])
                if iord == 8:
                    for i in range(is3, ie3 + 1):
                        xt = 2.0 * dm[i]
                        bl[i] = -math.copysign(
                            min(abs(xt), abs(al[i] - u[i, j])), xt)
                        br[i] = math.copysign(
                            min(abs(xt), abs(al[i + 1] - u[i, j])), xt)
                elif iord == 9:
                    for i in range(is3, ie3 + 1):
                        pmp_1 = -2.0 * dq[i]
                        lac_1 = pmp_1 + 1.5 * dq[i + 1]
                        bl[i] = min(max(0.0, pmp_1, lac_1),
                                    max(al[i] - u[i, j],
                                        min(0.0, pmp_1, lac_1)))
                        pmp_2 = 2.0 * dq[i - 1]
                        lac_2 = pmp_2 - 1.5 * dq[i - 2]
                        br[i] = min(max(0.0, pmp_2, lac_2),
                                    max(al[i + 1] - u[i, j],
                                        min(0.0, pmp_2, lac_2)))
                elif iord == 10:
                    for i in range(is3, ie3 + 1):
                        bl[i] = al[i] - u[i, j]
                        br[i] = al[i + 1] - u[i, j]
                        if abs(dm[i]) < NEAR_ZERO_SW:
                            if abs(dm[i - 1]) + abs(dm[i + 1]) < NEAR_ZERO_SW:
                                bl[i] = 0.0
                                br[i] = 0.0
                        elif abs(3.0 * (bl[i] + br[i])) > abs(bl[i] - br[i]):
                            pmp_1 = -2.0 * dq[i]
                            lac_1 = pmp_1 + 1.5 * dq[i + 1]
                            bl[i] = min(max(0.0, pmp_1, lac_1),
                                        max(bl[i], min(0.0, pmp_1, lac_1)))
                            pmp_2 = 2.0 * dq[i - 1]
                            lac_2 = pmp_2 - 1.5 * dq[i - 2]
                            br[i] = min(max(0.0, pmp_2, lac_2),
                                        max(br[i], min(0.0, pmp_2, lac_2)))
                else:      # 11, unlimited
                    for i in range(is3, ie3 + 1):
                        bl[i] = al[i] - u[i, j]
                        br[i] = al[i + 1] - u[i, j]

                if is_ == 1 and not bounded_domain:
                    br[2] = al[3] - u[2, j]
                    xt = SW_S15 * u[1, j] + SW_S11 * u[2, j] - SW_S14 * dm[2]
                    bl[2] = xt - u[2, j]
                    br[1] = xt - u[1, j]
                    if j == 1 or j == npy:
                        bl[0] = 0.0
                        br[0] = 0.0
                        bl[1] = 0.0
                        br[1] = 0.0
                    else:
                        bl[0] = SW_S14 * dm[-1] - SW_S11 * dq[-1]
                        x0L = 0.5 * ((2.0 * dx[0, j] + dx[-1, j]) * (u[0, j])
                                     - dx[0, j] * (u[-1, j])) \
                            / (dx[0, j] + dx[-1, j])
                        x0R = 0.5 * ((2.0 * dx[1, j] + dx[2, j]) * (u[1, j])
                                     - dx[1, j] * (u[2, j])) \
                            / (dx[1, j] + dx[2, j])
                        xt = x0L + x0R
                        br[0] = xt - u[0, j]
                        bl[1] = xt - u[1, j]
                    _pert_ppm_run(u, bl, br, 2, j, is_col=2)

                if (ie + 1) == npx and not bounded_domain:
                    bl[npx - 2] = al[npx - 2] - u[npx - 2, j]
                    xt = SW_S15 * u[npx - 1, j] + SW_S11 * u[npx - 2, j] \
                        + SW_S14 * dm[npx - 2]
                    br[npx - 2] = xt - u[npx - 2, j]
                    bl[npx - 1] = xt - u[npx - 1, j]
                    if j == 1 or j == npy:
                        bl[npx - 1] = 0.0
                        br[npx - 1] = 0.0
                        bl[npx] = 0.0
                        br[npx] = 0.0
                    else:
                        br[npx] = SW_S11 * dq[npx] - SW_S14 * dm[npx + 1]
                        x0L = 0.5 * ((2.0 * dx[npx - 1, j] + dx[npx - 2, j])
                                     * (u[npx - 1, j])
                                     - dx[npx - 1, j] * (u[npx - 2, j])) \
                            / (dx[npx - 1, j] + dx[npx - 2, j])
                        x0R = 0.5 * ((2.0 * dx[npx, j] + dx[npx + 1, j])
                                     * (u[npx, j])
                                     - dx[npx, j] * (u[npx + 1, j])) \
                            / (dx[npx, j] + dx[npx + 1, j])
                        xt = x0L + x0R
                        br[npx - 1] = xt - u[npx - 1, j]
                        bl[npx] = xt - u[npx, j]
                    _pert_ppm_run(u, bl, br, npx - 2, j, is_col=2)
            else:
                al = _fl1(is_ - 1, ie + 2)
                for i in range(is_ - 1, ie + 2 + 1):
                    al[i] = 0.5 * (u[i - 1, j] + u[i, j]) \
                        + R3 * (dm[i - 1] - dm[i])
                for i in range(is_ - 1, ie + 1 + 1):
                    pmp = -2.0 * dq[i]
                    lac = pmp + 1.5 * dq[i + 1]
                    bl[i] = min(max(0.0, pmp, lac),
                                max(al[i] - u[i, j], min(0.0, pmp, lac)))
                    pmp = 2.0 * dq[i - 1]
                    lac = pmp - 1.5 * dq[i - 2]
                    br[i] = min(max(0.0, pmp, lac),
                                max(al[i + 1] - u[i, j], min(0.0, pmp, lac)))

            for i in range(is_, ie + 1 + 1):
                if c[i, j] > 0.0:
                    cfl = c[i, j] * rdx[i - 1, j]
                    flux[i, j] = u[i - 1, j] + (1.0 - cfl) \
                        * (br[i - 1] - cfl * (bl[i - 1] + br[i - 1]))
                else:
                    cfl = c[i, j] * rdx[i, j]
                    flux[i, j] = u[i, j] + (1.0 + cfl) \
                        * (bl[i] + cfl * (bl[i] + br[i]))


def _pert_ppm_run(field, bl, br, i0, j, is_col):
    """pert_ppm(1, u(i0,j), bl(i0), br(i0), -1) — xtp_u single-cell call.

    xtp_u invokes pert_ppm with im=1 on one interior cell (i0).  Extract
    the point into 0-based length-1 arrays, run the verbatim ported
    pert_ppm, write bl/br back.
    """
    one_a = np.array([float(field[i0, j])])
    one_l = np.array([float(bl[i0])])
    one_r = np.array([float(br[i0])])
    pert_ppm(1, one_a, one_l, one_r, -1)
    bl[i0] = one_l[0]
    br[i0] = one_r[0]


def ytp_v(is_, ie, js, je, isd, ied, jsd, jed, c, u, v, flux, jord,
          dy, rdy, npx, npy, grid_type, bounded_domain, lim_fac):
    """sw_core.F90 ytp_v (verbatim, all jord branches).

    c/flux fort (is:ie+1, js:je+1); u fort (isd:ied, jsd:jed+1);
    v fort (isd:ied+1, jsd:jed); dy/rdy fort (isd:ied+1, jsd:jed).
    Mutates flux.
    """
    if bounded_domain or grid_type > 3:
        js3 = js - 1
        je3 = je + 1
    else:
        js3 = max(3, js - 1)
        je3 = min(npy - 3, je + 1)

    al = _fl(is_, ie + 1, js - 1, je + 2)
    bl = _fl(is_, ie + 1, js - 1, je + 1)
    br = _fl(is_, ie + 1, js - 1, je + 1)
    b0 = _fl(is_, ie + 1, js - 1, je + 1)

    if jord < 8:
        for j in range(js3, je3 + 1 + 1):
            for i in range(is_, ie + 1 + 1):
                al[i, j] = SW_P1 * (v[i, j - 1] + v[i, j]) \
                    + SW_P2 * (v[i, j - 2] + v[i, j + 1])
        for j in range(js3, je3 + 1):
            for i in range(is_, ie + 1 + 1):
                bl[i, j] = al[i, j] - v[i, j]
                br[i, j] = al[i, j + 1] - v[i, j]

        if (not bounded_domain) and grid_type < 3:
            if js == 1:
                for i in range(is_, ie + 1 + 1):
                    bl[i, 0] = SW_C1 * v[i, -2] + SW_C2 * v[i, -1] \
                        + SW_C3 * v[i, 0] - v[i, 0]
                    xt = 0.5 * (
                        ((2.0 * dy[i, 0] + dy[i, -1]) * v[i, 0]
                         - dy[i, 0] * v[i, -1]) / (dy[i, 0] + dy[i, -1])
                        + ((2.0 * dy[i, 1] + dy[i, 2]) * v[i, 1]
                           - dy[i, 1] * v[i, 2]) / (dy[i, 1] + dy[i, 2]))
                    br[i, 0] = xt - v[i, 0]
                    bl[i, 1] = xt - v[i, 1]
                    xt = SW_C3 * v[i, 1] + SW_C2 * v[i, 2] + SW_C1 * v[i, 3]
                    br[i, 1] = xt - v[i, 1]
                    bl[i, 2] = xt - v[i, 2]
                    br[i, 2] = al[i, 3] - v[i, 2]
                if is_ == 1:
                    bl[1, 0] = 0.0
                    br[1, 0] = 0.0
                    bl[1, 1] = 0.0
                    br[1, 1] = 0.0
                if (ie + 1) == npx:
                    bl[npx, 0] = 0.0
                    br[npx, 0] = 0.0
                    bl[npx, 1] = 0.0
                    br[npx, 1] = 0.0
            if (je + 1) == npy:
                for i in range(is_, ie + 1 + 1):
                    bl[i, npy - 2] = al[i, npy - 2] - v[i, npy - 2]
                    xt = SW_C1 * v[i, npy - 3] + SW_C2 * v[i, npy - 2] \
                        + SW_C3 * v[i, npy - 1]
                    br[i, npy - 2] = xt - v[i, npy - 2]
                    bl[i, npy - 1] = xt - v[i, npy - 1]
                    xt = 0.5 * (
                        ((2.0 * dy[i, npy - 1] + dy[i, npy - 2])
                         * v[i, npy - 1] - dy[i, npy - 1] * v[i, npy - 2])
                        / (dy[i, npy - 1] + dy[i, npy - 2])
                        + ((2.0 * dy[i, npy] + dy[i, npy + 1]) * v[i, npy]
                           - dy[i, npy] * v[i, npy + 1])
                        / (dy[i, npy] + dy[i, npy + 1]))
                    br[i, npy - 1] = xt - v[i, npy - 1]
                    bl[i, npy] = xt - v[i, npy]
                    br[i, npy] = SW_C3 * v[i, npy] + SW_C2 * v[i, npy + 1] \
                        + SW_C1 * v[i, npy + 2] - v[i, npy]
                if is_ == 1:
                    bl[1, npy - 1] = 0.0
                    br[1, npy - 1] = 0.0
                    bl[1, npy] = 0.0
                    br[1, npy] = 0.0
                if (ie + 1) == npx:
                    bl[npx, npy - 1] = 0.0
                    br[npx, npy - 1] = 0.0
                    bl[npx, npy] = 0.0
                    br[npx, npy] = 0.0

        for j in range(js - 1, je + 1 + 1):
            for i in range(is_, ie + 1 + 1):
                b0[i, j] = bl[i, j] + br[i, j]

        fx0 = _fl1(is_, ie + 1)
        if jord == 1:
            smt5 = _flb(is_, ie + 1, js - 1, je + 1)
            for j in range(js - 1, je + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    smt5[i, j] = abs(lim_fac * b0[i, j]) \
                        < abs(bl[i, j] - br[i, j])
            for j in range(js, je + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    if c[i, j] > 0.0:
                        cfl = c[i, j] * rdy[i, j - 1]
                        fx0[i] = (1.0 - cfl) * (br[i, j - 1]
                                                - cfl * b0[i, j - 1])
                        flux[i, j] = v[i, j - 1]
                    else:
                        cfl = c[i, j] * rdy[i, j]
                        fx0[i] = (1.0 + cfl) * (bl[i, j] + cfl * b0[i, j])
                        flux[i, j] = v[i, j]
                    if smt5[i, j - 1] or smt5[i, j]:
                        flux[i, j] = flux[i, j] + fx0[i]

        elif jord == 2:
            for j in range(js, je + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    if c[i, j] > 0.0:
                        cfl = c[i, j] * rdy[i, j - 1]
                        flux[i, j] = v[i, j - 1] + (1.0 - cfl) \
                            * (br[i, j - 1] - cfl * b0[i, j - 1])
                    else:
                        cfl = c[i, j] * rdy[i, j]
                        flux[i, j] = v[i, j] + (1.0 + cfl) \
                            * (bl[i, j] + cfl * b0[i, j])

        elif jord == 3:
            smt5 = _flb(is_, ie + 1, js - 1, je + 1)
            smt6 = _flb(is_, ie + 1, js - 1, je + 1)
            for j in range(js - 1, je + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    x0 = abs(b0[i, j])
                    x1 = abs(bl[i, j] - br[i, j])
                    smt5[i, j] = x0 < x1
                    smt6[i, j] = 3.0 * x0 < x1
            hi5 = _fl1b(is_, ie + 1)
            hi6 = _fl1b(is_, ie + 1)
            for j in range(js, je + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    fx0[i] = 0.0
                    hi5[i] = smt5[i, j - 1] and smt5[i, j]
                    hi6[i] = smt6[i, j - 1] or smt6[i, j]
                for i in range(is_, ie + 1 + 1):
                    if c[i, j] > 0.0:
                        cfl = c[i, j] * rdy[i, j - 1]
                        if hi6[i]:
                            fx0[i] = br[i, j - 1] - cfl * b0[i, j - 1]
                        elif hi5[i]:
                            fx0[i] = math.copysign(
                                min(abs(bl[i, j - 1]), abs(br[i, j - 1])),
                                br[i, j - 1])
                        flux[i, j] = v[i, j - 1] + (1.0 - cfl) * fx0[i]
                    else:
                        cfl = c[i, j] * rdy[i, j]
                        if hi6[i]:
                            fx0[i] = bl[i, j] + cfl * b0[i, j]
                        elif hi5[i]:
                            fx0[i] = math.copysign(
                                min(abs(bl[i, j]), abs(br[i, j])), bl[i, j])
                        flux[i, j] = v[i, j] + (1.0 + cfl) * fx0[i]

        elif jord == 4:
            smt5 = _flb(is_, ie + 1, js - 1, je + 1)
            smt6 = _flb(is_, ie + 1, js - 1, je + 1)
            for j in range(js - 1, je + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    x0 = abs(b0[i, j])
                    x1 = abs(bl[i, j] - br[i, j])
                    smt5[i, j] = x0 < x1
                    smt6[i, j] = 3.0 * x0 < x1
            hi5 = _fl1b(is_, ie + 1)
            hi6 = _fl1b(is_, ie + 1)
            for j in range(js, je + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    fx0[i] = 0.0
                    hi5[i] = smt5[i, j - 1] and smt5[i, j]
                    hi6[i] = smt6[i, j - 1] or smt6[i, j]
                    hi5[i] = hi5[i] or hi6[i]
                for i in range(is_, ie + 1 + 1):
                    if c[i, j] > 0.0:
                        cfl = c[i, j] * rdy[i, j - 1]
                        fx0[i] = (1.0 - cfl) * (br[i, j - 1]
                                                - cfl * b0[i, j - 1])
                        flux[i, j] = v[i, j - 1]
                    else:
                        cfl = c[i, j] * rdy[i, j]
                        fx0[i] = (1.0 + cfl) * (bl[i, j] + cfl * b0[i, j])
                        flux[i, j] = v[i, j]
                    if hi5[i]:
                        flux[i, j] = flux[i, j] + fx0[i]

        else:      # jord = 5, 6, 7
            if jord == 5:
                smt5 = _flb(is_, ie + 1, js - 1, je + 1)
                for j in range(js - 1, je + 1 + 1):
                    for i in range(is_, ie + 1 + 1):
                        smt5[i, j] = bl[i, j] * br[i, j] < 0.0
                for j in range(js, je + 1 + 1):
                    for i in range(is_, ie + 1 + 1):
                        if c[i, j] > 0.0:
                            cfl = c[i, j] * rdy[i, j - 1]
                            fx0[i] = (1.0 - cfl) * (br[i, j - 1]
                                                    - cfl * b0[i, j - 1])
                            flux[i, j] = v[i, j - 1]
                        else:
                            cfl = c[i, j] * rdy[i, j]
                            fx0[i] = (1.0 + cfl) * (bl[i, j]
                                                    + cfl * b0[i, j])
                            flux[i, j] = v[i, j]
                        if smt5[i, j - 1] or smt5[i, j]:
                            flux[i, j] = flux[i, j] + fx0[i]
            else:
                smt6 = _flb(is_, ie + 1, js - 1, je + 1)
                for j in range(js - 1, je + 1 + 1):
                    for i in range(is_, ie + 1 + 1):
                        smt6[i, j] = 3.0 * abs(b0[i, j]) \
                            < abs(bl[i, j] - br[i, j])
                if (not bounded_domain) and grid_type < 3:
                    if js == 1:
                        for i in range(is_, ie + 1 + 1):
                            smt6[i, 0] = bl[i, 0] * br[i, 0] < 0.0
                            smt6[i, 1] = bl[i, 1] * br[i, 1] < 0.0
                    if (je + 1) == npy:
                        for i in range(is_, ie + 1 + 1):
                            smt6[i, npy - 1] = \
                                bl[i, npy - 1] * br[i, npy - 1] < 0.0
                            smt6[i, npy] = bl[i, npy] * br[i, npy] < 0.0
                for j in range(js, je + 1 + 1):
                    for i in range(is_, ie + 1 + 1):
                        if c[i, j] > 0.0:
                            cfl = c[i, j] * rdy[i, j - 1]
                            fx0[i] = (1.0 - cfl) * (br[i, j - 1]
                                                    - cfl * b0[i, j - 1])
                            flux[i, j] = v[i, j - 1]
                        else:
                            cfl = c[i, j] * rdy[i, j]
                            fx0[i] = (1.0 + cfl) * (bl[i, j]
                                                    + cfl * b0[i, j])
                            flux[i, j] = v[i, j]
                        if smt6[i, j - 1] or smt6[i, j]:
                            flux[i, j] = flux[i, j] + fx0[i]

    else:      # jord = 8, 9, 10, 11
        dm = _fl(is_, ie + 1, js - 2, je + 2)
        for j in range(js - 2, je + 2 + 1):
            for i in range(is_, ie + 1 + 1):
                xt = 0.25 * (v[i, j + 1] - v[i, j - 1])
                dm[i, j] = math.copysign(
                    min(abs(xt),
                        max(v[i, j - 1], v[i, j], v[i, j + 1]) - v[i, j],
                        v[i, j] - min(v[i, j - 1], v[i, j], v[i, j + 1])), xt)
        dq = _fl(is_, ie + 1, js - 3, je + 2)
        for j in range(js - 3, je + 2 + 1):
            for i in range(is_, ie + 1 + 1):
                dq[i, j] = v[i, j + 1] - v[i, j]

        if grid_type < 3:
            for j in range(js3, je3 + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    al[i, j] = 0.5 * (v[i, j - 1] + v[i, j]) \
                        + R3 * (dm[i, j - 1] - dm[i, j])
            if jord == 8:
                for j in range(js3, je3 + 1):
                    for i in range(is_, ie + 1 + 1):
                        xt = 2.0 * dm[i, j]
                        bl[i, j] = -math.copysign(
                            min(abs(xt), abs(al[i, j] - v[i, j])), xt)
                        br[i, j] = math.copysign(
                            min(abs(xt), abs(al[i, j + 1] - v[i, j])), xt)
            elif jord == 9:
                for j in range(js3, je3 + 1):
                    for i in range(is_, ie + 1 + 1):
                        pmp_1 = -2.0 * dq[i, j]
                        lac_1 = pmp_1 + 1.5 * dq[i, j + 1]
                        bl[i, j] = min(max(0.0, pmp_1, lac_1),
                                       max(al[i, j] - v[i, j],
                                           min(0.0, pmp_1, lac_1)))
                        pmp_2 = 2.0 * dq[i, j - 1]
                        lac_2 = pmp_2 - 1.5 * dq[i, j - 2]
                        br[i, j] = min(max(0.0, pmp_2, lac_2),
                                       max(al[i, j + 1] - v[i, j],
                                           min(0.0, pmp_2, lac_2)))
            elif jord == 10:
                for j in range(js3, je3 + 1):
                    for i in range(is_, ie + 1 + 1):
                        bl[i, j] = al[i, j] - v[i, j]
                        br[i, j] = al[i, j + 1] - v[i, j]
                        if abs(dm[i, j]) < NEAR_ZERO_SW:
                            if abs(dm[i, j - 1]) + abs(dm[i, j + 1]) \
                                    < NEAR_ZERO_SW:
                                bl[i, j] = 0.0
                                br[i, j] = 0.0
                        elif abs(3.0 * (bl[i, j] + br[i, j])) \
                                > abs(bl[i, j] - br[i, j]):
                            pmp_1 = -2.0 * dq[i, j]
                            lac_1 = pmp_1 + 1.5 * dq[i, j + 1]
                            bl[i, j] = min(max(0.0, pmp_1, lac_1),
                                           max(bl[i, j],
                                               min(0.0, pmp_1, lac_1)))
                            pmp_2 = 2.0 * dq[i, j - 1]
                            lac_2 = pmp_2 - 1.5 * dq[i, j - 2]
                            br[i, j] = min(max(0.0, pmp_2, lac_2),
                                           max(br[i, j],
                                               min(0.0, pmp_2, lac_2)))
            else:      # 11, unlimited
                for j in range(js3, je3 + 1):
                    for i in range(is_, ie + 1 + 1):
                        bl[i, j] = al[i, j] - v[i, j]
                        br[i, j] = al[i, j + 1] - v[i, j]

            if js == 1 and not bounded_domain:
                for i in range(is_, ie + 1 + 1):
                    br[i, 2] = al[i, 3] - v[i, 2]
                    xt = SW_S15 * v[i, 1] + SW_S11 * v[i, 2] - SW_S14 * dm[i, 2]
                    br[i, 1] = xt - v[i, 1]
                    bl[i, 2] = xt - v[i, 2]
                    bl[i, 0] = SW_S14 * dm[i, -1] - SW_S11 * dq[i, -1]
                    x0L = 0.5 * ((2.0 * dy[i, 0] + dy[i, -1]) * (v[i, 0])
                                 - dy[i, 0] * (v[i, -1])) \
                        / (dy[i, 0] + dy[i, -1])
                    x0R = 0.5 * ((2.0 * dy[i, 1] + dy[i, 2]) * (v[i, 1])
                                 - dy[i, 1] * (v[i, 2])) \
                        / (dy[i, 1] + dy[i, 2])
                    xt = x0L + x0R
                    bl[i, 1] = xt - v[i, 1]
                    br[i, 0] = xt - v[i, 0]
                if is_ == 1:
                    bl[1, 0] = 0.0
                    br[1, 0] = 0.0
                    bl[1, 1] = 0.0
                    br[1, 1] = 0.0
                if (ie + 1) == npx:
                    bl[npx, 0] = 0.0
                    br[npx, 0] = 0.0
                    bl[npx, 1] = 0.0
                    br[npx, 1] = 0.0
                _pert_ppm_row(v, bl, br, is_, ie, 2)
            if (je + 1) == npy and not bounded_domain:
                for i in range(is_, ie + 1 + 1):
                    bl[i, npy - 2] = al[i, npy - 2] - v[i, npy - 2]
                    xt = SW_S15 * v[i, npy - 1] + SW_S11 * v[i, npy - 2] \
                        + SW_S14 * dm[i, npy - 2]
                    br[i, npy - 2] = xt - v[i, npy - 2]
                    bl[i, npy - 1] = xt - v[i, npy - 1]
                    br[i, npy] = SW_S11 * dq[i, npy] - SW_S14 * dm[i, npy + 1]
                    x0L = 0.5 * ((2.0 * dy[i, npy - 1] + dy[i, npy - 2])
                                 * (v[i, npy - 1])
                                 - dy[i, npy - 1] * (v[i, npy - 2])) \
                        / (dy[i, npy - 1] + dy[i, npy - 2])
                    x0R = 0.5 * ((2.0 * dy[i, npy] + dy[i, npy + 1])
                                 * (v[i, npy])
                                 - dy[i, npy] * (v[i, npy + 1])) \
                        / (dy[i, npy] + dy[i, npy + 1])
                    xt = x0L + x0R
                    br[i, npy - 1] = xt - v[i, npy - 1]
                    bl[i, npy] = xt - v[i, npy]
                if is_ == 1:
                    bl[1, npy - 1] = 0.0
                    br[1, npy - 1] = 0.0
                    bl[1, npy] = 0.0
                    br[1, npy] = 0.0
                if (ie + 1) == npx:
                    bl[npx, npy - 1] = 0.0
                    br[npx, npy - 1] = 0.0
                    bl[npx, npy] = 0.0
                    br[npx, npy] = 0.0
                _pert_ppm_row(v, bl, br, is_, ie, npy - 2)
        else:
            for j in range(js - 1, je + 2 + 1):
                for i in range(is_, ie + 1 + 1):
                    al[i, j] = 0.5 * (v[i, j - 1] + v[i, j]) \
                        + R3 * (dm[i, j - 1] - dm[i, j])
            for j in range(js - 1, je + 1 + 1):
                for i in range(is_, ie + 1 + 1):
                    pmp = 2.0 * dq[i, j - 1]
                    lac = pmp - 1.5 * dq[i, j - 2]
                    br[i, j] = min(max(0.0, pmp, lac),
                                   max(al[i, j + 1] - v[i, j],
                                       min(0.0, pmp, lac)))
                    pmp = -2.0 * dq[i, j]
                    lac = pmp + 1.5 * dq[i, j + 1]
                    bl[i, j] = min(max(0.0, pmp, lac),
                                   max(al[i, j] - v[i, j],
                                       min(0.0, pmp, lac)))

        for j in range(js, je + 1 + 1):
            for i in range(is_, ie + 1 + 1):
                if c[i, j] > 0.0:
                    cfl = c[i, j] * rdy[i, j - 1]
                    flux[i, j] = v[i, j - 1] + (1.0 - cfl) \
                        * (br[i, j - 1] - cfl * (bl[i, j - 1] + br[i, j - 1]))
                else:
                    cfl = c[i, j] * rdy[i, j]
                    flux[i, j] = v[i, j] + (1.0 + cfl) \
                        * (bl[i, j] + cfl * (bl[i, j] + br[i, j]))


def _pert_ppm_row(field, bl, br, is_, ie, j):
    """pert_ppm(ie-is+2, v(is,j), bl(is,j), br(is,j), -1) — row call.

    ytp_v invokes pert_ppm over the i-row at fixed j; extract the row into
    length-(ie-is+2) 1-based arrays, run the ported pert_ppm, write back.
    """
    im = ie - is_ + 2
    a0 = np.array([float(field[is_ + k, j]) for k in range(im)])
    a_l = np.array([float(bl[is_ + k, j]) for k in range(im)])
    a_r = np.array([float(br[is_ + k, j]) for k in range(im)])
    pert_ppm(im, a0, a_l, a_r, -1)
    for k in range(im):
        bl[is_ + k, j] = a_l[k]
        br[is_ + k, j] = a_r[k]
