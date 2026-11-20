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

The analytic function implements km in {5, 10}; FV3_L40_AK/BK below
also provide the source-generated default km=40 coordinate. That is not a simplification: the
3-D duo reference run is npz=5, and `fv_eta` has NO table below km=5 at all,
so npz=3 produces NaN ak/bk and aborts upstream. Any other km must be added
from its own oracle branch rather than extrapolated from this one -- other cases have their own tables or generators.

The oracle prints "Warning: the chosen setting in set_eta can cause
instability" for this branch; that warning is upstream's own and is expected
in the reference run, not a sign of a bad port.

``FV3_L40_AK/BK`` are exact data from ``tools/fv_eta.F90:413-417,759-760``:
``var_hi(40, ak, bk, ptop=5000, ks, pint=30000, s_rate=1.03)``. The
generator is :1136-1311, smoothing :2249-2282. USE_VAR_ETA, USE_GFSL63,
HIWPP and NO_UKMO_HB are undefined; npz_type is empty (:fv_arrays.F90:470).
Source: Zenodo 8327578, archive md5 25b2f54c0a807275e1c4edf4aa41a197.
The coefficients preserve the fp64 source execution, including tiny residuals.
The paper's C768 nonhydrostatic duo run used L32; this is its source's L40
default cold-start coordinate, not that run's coordinate.
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
            f"other km have separate tables/generators in fv_eta.F90, so "
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


# Exact source-generated data: no interpolation and no runtime generator.
# Zenodo 8327578, outer ZIP md5 25b2f54c0a807275e1c4edf4aa41a197.
# tools/fv_eta.F90:37,263-268: USE_VAR_ETA undefined; model/fv_arrays.F90:
# 467-471: USE_GFSL63 undefined -> npz_type=''. fv_eta.F90:413-417,759-760:
# var_hi(40, ak, bk, ptop=5000, ks, pint=30000, s_rate=1.03).
# var_hi :1136-1311, sm1_edge :2249-2282; HIWPP/NO_UKMO_HB undefined
# selects standard stretching + UKMO hybrid. Cold start only.
# GNU Fortran 8.5, -O0 -fdefault-real-8 -fdefault-double-8 -ffp-contract=off;
# linked constants are the existing pinned FV3/FMS GFS constants. This is
# exact for that source execution, not a replication of the paper's L32 run.
# Reproducer: scripts/validate/fv3_native/gen_eta_l40_oracle.py; provenance
# and all independently read coefficients: tests/grids/fixtures/fv3_l40_oracle.json.
FV3_L40_AK = (
    5000.0,
    8132.429001731174,
    10967.99034462965,
    13843.399428329398,
    16351.364894329268,
    18565.178485832956,
    20603.6622765726,
    22552.09501918063,
    24472.875428225412,
    26423.541656153375,
    28447.66023297034,
    30561.060530114148,
    32454.78632237877,
    33853.86431268948,
    34804.06939488379,
    35345.84008809005,
    35515.05362505546,
    35343.67963525374,
    34860.33231010202,
    34090.73748663547,
    33058.12827277716,
    31783.58053723025,
    30286.297700881205,
    28583.852715390523,
    26685.182367985777,
    24611.910263030622,
    22425.828597149168,
    20185.310541818046,
    17925.12645545684,
    15678.663615145488,
    13477.831479425258,
    11352.934270333337,
    9332.518853370026,
    7443.20557605754,
    5709.509387353184,
    4153.658180242928,
    2795.414862627077,
    1651.9091565467686,
    737.4845472478825,
    2.220446049250313e-11,
    0.0,
)
FV3_L40_BK = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0030822618551797133,
    0.011986498786926968,
    0.02624229795322972,
    0.04543005687674537,
    0.0691733374446437,
    0.09713244654286617,
    0.12899904288074496,
    0.1644916040533939,
    0.20335161602464227,
    0.245340370237151,
    0.2902362724578946,
    0.33783258303467084,
    0.38812224423543124,
    0.44073420454972795,
    0.49434010240488,
    0.5477925015175837,
    0.6005308647143642,
    0.6520076453144532,
    0.7016917093963918,
    0.7490721565793999,
    0.7936624174571718,
    0.8350045068955868,
    0.8726733146436753,
    0.9062808183824311,
    0.9354801096395753,
    0.9599691300397929,
    0.979494024168811,
    0.9952049491821027,
    1.0,
)
FV3_L40_KS = 11
