"""Vertical coordinate (``set_eta``), km-general entry for the ported cases.

STAGE 1 support module for the 3-D duo port: the acoustic units carry
``delp`` as a prognostic, and ``delp`` at initialisation comes from
``ak``/``bk``, so the port cannot build an oracle-comparable initial state
without reproducing ``set_eta`` exactly.

ORACLE: ``tools/fv_eta.F90``. There are TWO ``set_eta`` subroutines in that
file, selected by the ``USE_VAR_ETA`` CPP macro (``:37`` ``#ifdef``, ``:263``
``#else``, ``:785`` ``#endif``). The oracle build does NOT define
``USE_VAR_ETA``, so the live copy is the second one; its ``km == 5`` branch is
``:334-344``. (The VAR_ETA copy handles km=5 at ``:241-248`` with the same
three lines, so the numbers agree either way -- but cite the branch that
actually compiles.)

    fv_eta.F90:334-344
        select case (km)
        case (5,10) ! does this work????
           ! Equivalent Shallow Water: for modon test
           ptop = 500.e2
           ks = 0
           do k=1,km+1
              bk(k) = real(k-1) / real (km)
              ak(k) = ptop*(1.-bk(k))
           enddo

Only km in {5, 10} is implemented here. That is not a simplification: the
3-D duo reference run is npz=5, and `fv_eta` has NO table below km=5 at all,
so npz=3 produces NaN ak/bk and aborts upstream. Any other km must be added
from its own oracle branch rather than extrapolated from this one -- the
other cases are hand-tabulated, not analytic.

The oracle prints "Warning: the chosen setting in set_eta can cause
instability" for this branch; that warning is upstream's own and is expected
in the reference run, not a sign of a bad port.
"""
from __future__ import annotations

import numpy as np

# fv_eta.F90:339. Pa. Only the km in {5,10} analytic branch uses this value;
# the tabulated branches carry their own ptop.
_PTOP_SHALLOW_WATER_PA = 500.0e2


def set_eta_analytic(km: int) -> tuple[np.ndarray, np.ndarray, float, int]:
    """``ak``, ``bk``, ``ptop``, ``ks`` for the km in {5, 10} branch.

    Returns 1-D arrays of length ``km + 1`` indexed 0-based, i.e. element
    ``k`` here is Fortran ``ak(k+1)``.

    ``ks`` is the number of pure-pressure levels; this branch sets it to 0,
    so there is no pure-pressure region and every interface is a hybrid.
    """
    if km not in (5, 10):
        raise ValueError(
            f"set_eta_analytic: km={km} is not the analytic 'Equivalent "
            f"Shallow Water' branch (fv_eta.F90:336 is `case (5,10)`). The "
            f"other km are HAND-TABULATED in fv_eta.F90, not analytic, so "
            f"they must be transcribed from their own case, never "
            f"interpolated from this one. km < 5 has no table at all and "
            f"aborts upstream.")

    k = np.arange(km + 1, dtype=np.float64)          # Fortran k-1
    bk = k / float(km)                               # fv_eta.F90:342
    ak = _PTOP_SHALLOW_WATER_PA * (1.0 - bk)         # fv_eta.F90:343
    return ak, bk, _PTOP_SHALLOW_WATER_PA, 0


def interface_pressure(ak: np.ndarray, bk: np.ndarray,
                       ps: np.ndarray | float) -> np.ndarray:
    """``pe = ak + bk*ps`` -- the hybrid-sigma interface pressures [Pa].

    ``ps`` may be a scalar or an array; the level axis is appended last, so a
    ``(ni, nj)`` surface pressure gives ``(ni, nj, km+1)``.
    """
    ak = np.asarray(ak, dtype=np.float64)
    bk = np.asarray(bk, dtype=np.float64)
    if ak.shape != bk.shape:
        raise ValueError(f"ak {ak.shape} and bk {bk.shape} must match")
    ps = np.asarray(ps, dtype=np.float64)
    return ak + bk * ps[..., None] if ps.ndim else ak + bk * float(ps)


def layer_thickness(ak: np.ndarray, bk: np.ndarray,
                    ps: np.ndarray | float) -> np.ndarray:
    """``delp(k) = pe(k+1) - pe(k)`` [Pa], the prognostic layer mass."""
    pe = interface_pressure(ak, bk, ps)
    return np.diff(pe, axis=-1)
