"""DCMIP-2016 baroclinic-wave analytic profiles (``test_case = -13``).

The 3-D duo reference run cold-starts from this IC (``external_ic=.F.``,
``warm_start=.F.``, empty ``INPUT/``), so the port cannot produce an
oracle-comparable initial state without it.

ORACLE: ``tools/test_cases.F90``, subroutine ``DCMIP16_BC`` (``:6458-6855``)
and its five contained functions (``:6774-6853``). ``test_cases.F90:2188``
dispatches BOTH ``test_case==-12`` and ``==-13`` here, passing
``test_case == -13`` as the perturbation flag -- so -13 is -12 plus
``DCMIP16_BC_uwind_pert`` added to the zonal wind at the u- AND v-points, and
nothing else differs.

THIS MODULE IS THE ANALYTIC CORE ONLY: the pointwise profiles and the Newton
height solve. The grid-dependent assembly (D-grid edge midpoints, unit
tangents, the projection onto e_lon, the halo fills) belongs with the
gridstruct and is NOT here.

EXPRESSION FORMS ARE COPIED LITERALLY, NOT SIMPLIFIED. The oracle writes the
same mathematical quantity three different ways in three places, and each
rounds differently in the last bits:

  * ``IT`` uses ``exp(KK*log(cos(lat)))``, not ``cos(lat)**3``  (:6780, :6797)
  * ``Tr`` uses ``zsc**2.`` -- a REAL exponent, i.e. ``pow``     (:6782)
  * ``Tir`` writes the same square as an EXPLICIT PRODUCT        (:6798, :6813)
  * ``UU`` uses INTEGER exponents ``cos(lat)**(int(KK)-1)``      (:6816)

Rewriting these into one consistent form would change the answer in the last
few ulp for no benefit, and parity here is measured in ulp.

PRECISION CEILING. ``great_circle_dist`` (used only by the perturbation) runs
in QUAD internally in the oracle (``fv_grid_utils.F90:43-49``: absent
``NO_QUAD_PRECISION``, ``f_p = selected_real_kind(20)``), so a float64 port
cannot be bit-exact there. Everything else in this module is f64 on both
sides (``-fdefault-real-8``).

WHY THE CONSTANTS ARE ARGUMENTS AND NOT ``legoesm.constants``.
``test_cases.F90:102`` aliases ``radius = cnst_radius`` from FMS
``constants_mod``, and FMS ships TWO constant sets selected at COMPILE time
(``fmsconstants.F90:67-68``: with none of GFDL/GFS/GEOS defined it defaults to
GFDL). The two differ far above any parity tolerance -- 200 m in the Earth's
radius, 6.8e-4 relative in gravity -- and WHICH ONE APPLIES IS A PROPERTY OF
THE ORACLE BINARY, not of this formula. Substituting the model's own
``legoesm.constants`` here would silently compare the port against a planet
the reference run did not use; that exact confound produced a ~1e-4 relative
discrepancy in a reference run earlier in this work. Always confirm against
the run's own log: ``grep FMSConstants logfile.000000.out`` and
``grep "Radius is" run_out.txt``.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

# --- test_cases.F90:6484-6505, verbatim ------------------------------------
P0_PA = 1.0e5              # :6484
B_EXP = 2.0                # :6486
KK = 3.0                   # :6487
TE_K = 310.0               # :6488
TP_K = 240.0               # :6489
T0_K = 0.5 * (TE_K + TP_K)  # :6490 = 275.0 ("WRONG in document" -- oracle's comment)
UP_MS = 1.0                # :6491  perturbation amplitude
ZP_M = 1.5e4               # :6492  perturbation depth scale
LAPSE_K_PER_M = 5.0e-3     # :6497
PW_PA = 34000.0            # :6500
Q0 = 0.018                 # :6501
QT = 1.0e-12               # :6502
PTROP_PA = 1.0e4           # :6503
ZCONV_M = 1.0e-6           # :6505  Newton convergence, metres
_N_NEWTON = 30             # :6584 etc -- `do iter=1,30`

# u0 = 35. (:6485) and dT = 4.8e5 (:6498) are DECLARED AND NEVER USED in the
# oracle. Recorded so a reader diffing against the Fortran does not hunt for
# where they went.


class DCMIP16Constants(NamedTuple):
    """The FMS ``constants_mod`` values the ORACLE BINARY was linked against.

    Not the model's own constants -- see the module docstring. Passing the
    wrong set is the single most likely way to manufacture a fake parity
    failure.
    """
    radius: float
    grav: float
    rdgas: float
    rvgas: float
    omega: float
    pi: float = 3.1415926535897931   # PI_8, both constant sets

    @property
    def zvir(self) -> float:
        """test_cases.F90:6525 -- ``zvir = rvgas/rdgas - 1.``"""
        return self.rvgas / self.rdgas - 1.0

    @property
    def rp(self) -> float:
        """test_cases.F90:6496 -- ``Rp = radius/10.``"""
        return self.radius / 10.0

    @property
    def ppcenter(self) -> tuple:
        """:6493-6495 -- ``lamp = pi/9``, ``phip = 2*lamp``, RADIANS."""
        lamp = self.pi / 9.0
        return (lamp, 2.0 * lamp)

    @property
    def phiw(self) -> float:
        """:6499 -- ``phiW = 2.*pi/9.``  Same value as ``phip`` but a
        separate parameter in the oracle; kept separate here too."""
        return 2.0 * self.pi / 9.0


# The two shipped FMS constant sets, transcribed from
# fms-src/constants/{gfs,gfdl}_constants.h. These are ORACLE-BUILD
# properties, deliberately NOT legoesm.constants:
# const-ok: FMS GFS constant set, a property of the oracle binary (see module
# docstring); substituting the model's own constants is the documented
# confound this argument exists to prevent.
GFS_CONSTANTS = DCMIP16Constants(
    radius=6.3712e6, grav=9.80665, rdgas=287.05, rvgas=461.50,  # const-ok: as above
    omega=7.2921e-5)                                            # const-ok: as above
# const-ok: FMS GFDL constant set, likewise an oracle-binary property.
GFDL_CONSTANTS = DCMIP16Constants(
    radius=6371000.0, grav=9.80, rdgas=287.04, rvgas=461.50,     # const-ok: as above
    omega=7.292e-5)                                             # const-ok: as above


def _it(lat, c: DCMIP16Constants):
    """``IT`` -- test_cases.F90:6780 and :6797 (identical in both functions).

    ``exp(KK*log(cos(lat)))`` NOT ``cos(lat)**3``: at lat = +/-pi/2 the log
    diverges and the oracle inherits that, so the port must too rather than
    quietly returning 0 from a power.
    """
    coslat = np.cos(lat)
    return (np.exp(KK * np.log(coslat))
            - KK / (KK + 2.0) * np.exp((KK + 2.0) * np.log(coslat)))


def temperature(z, lat, c: DCMIP16Constants):
    """Analytic VIRTUAL temperature [K] -- test_cases.F90:6774-6789.

    Virtual, not real: the oracle divides by ``(1 + zvir*q)`` only when
    ``.not. adiabatic`` (:6760-6769), and the reference run has
    ``adiabatic=.true.``, so the restart's ``pt`` IS this quantity.
    """
    zsc = z * c.grav / (B_EXP * c.rdgas * T0_K)              # :6781
    tr = (1.0 - 2.0 * zsc ** 2.0) * np.exp(-zsc ** 2.0)      # :6782 REAL exponent
    t1 = (1.0 / T0_K) * np.exp(LAPSE_K_PER_M * z / T0_K) \
        + (T0_K - TP_K) / (T0_K * TP_K) * tr                 # :6784
    t2 = 0.5 * (KK + 2.0) * (TE_K - TP_K) / (TE_K * TP_K) * tr   # :6785
    return 1.0 / (t1 - t2 * _it(lat, c))                     # :6787


def pressure(z, lat, c: DCMIP16Constants):
    """Analytic pressure [Pa] -- test_cases.F90:6791-6805. ``p(0, lat) == p0``."""
    zs = z * c.grav / (B_EXP * c.rdgas * T0_K)
    tir = z * np.exp(-zs * zs)                               # :6798 explicit product
    ti1 = 1.0 / LAPSE_K_PER_M * (np.exp(LAPSE_K_PER_M * z / T0_K) - 1.0) \
        + tir * (T0_K - TP_K) / (T0_K * TP_K)                # :6800
    ti2 = 0.5 * (KK + 2.0) * (TE_K - TP_K) / (TE_K * TP_K) * tir  # :6801
    return P0_PA * np.exp(-c.grav / c.rdgas * (ti1 - ti2 * _it(lat, c)))  # :6803


def uwind(z, t, lat, c: DCMIP16Constants):
    """Analytic zonal wind [m/s] -- test_cases.F90:6807-6821.

    ``t`` is the LAYER-MEAN VIRTUAL temperature from hydrostatic balance, NOT
    the analytic temperature at ``z``. Passing the latter is a silent error.
    """
    zs = z * c.grav / (B_EXP * c.rdgas * T0_K)
    tir = z * np.exp(-zs * zs)                               # :6813
    ti2 = 0.5 * (KK + 2.0) * (TE_K - TP_K) / (TE_K * TP_K) * tir  # :6814
    coslat = np.cos(lat)
    # :6816 -- INTEGER exponents here, unlike IT's exp/log form
    uu = (c.grav * KK / c.radius) * ti2 * (coslat ** (int(KK) - 1)
                                           - coslat ** (int(KK) + 1)) * t
    orc = c.omega * c.radius * coslat
    return -orc + np.sqrt(orc ** 2 + c.radius * coslat * uu)  # :6817


def uwind_pert(z, lat, lon, c: DCMIP16Constants, great_circle_dist):
    """Perturbation, added ONLY for test_case=-13 -- :6823-6838.

    ``great_circle_dist(p1, p2, radius)`` is INJECTED: the oracle's version
    computes in quad, so whoever supplies it owns the precision floor of the
    result rather than this module hiding it.
    """
    zrat = z / ZP_M                                          # :6830
    zz = np.maximum(1.0 - 3.0 * zrat * zrat + 2.0 * zrat * zrat * zrat, 0.0)
    dst = great_circle_dist((lon, lat), c.ppcenter, c.radius)   # :6834
    return np.maximum(0.0, UP_MS * zz * np.exp(-(dst / c.rp) ** 2))  # :6836


def sphum(p, ps, lat, c: DCMIP16Constants):
    """Specific humidity [kg/kg] -- test_cases.F90:6840-6853.

    The oracle takes ``lon`` and never uses it, so it is omitted. The
    stratospheric branch is a HARD switch at ``p > ptrop``, not a blend.
    """
    p = np.asarray(p, dtype=np.float64)
    eta = p / np.asarray(ps, dtype=np.float64)
    moist = (Q0 * np.exp(-(lat / c.phiw) ** 4)
             * np.exp(-((eta - 1.0) * P0_PA / PW_PA) ** 2))   # :6850
    return np.where(p > PTROP_PA, moist, QT)                  # :6848-6851


def newton_height(p_target, lat, z_guess, c: DCMIP16Constants):
    """Interface height [m] for a target pressure -- test_cases.F90:6584-6597.

    The oracle's iteration verbatim: at most 30 passes, each

        z <- z + (p_analytic(z) - p_target)*(rdgas/grav)*T(z)/p_analytic(z)

    with the convergence test applied AFTER the update (``|dz| < 1e-6 m``).
    The initial guess is the interface BELOW, so the march runs k = npz down
    to 1 and each level is seeded by its neighbour.

    CONTROL-FLOW CAVEAT, stated rather than buried: Fortran EXITS per point;
    this vectorised form keeps updating already-converged points until EVERY
    point has converged. Past convergence each extra update moves z by less
    than 1e-6 m, but that is not nothing at ulp level, so results can differ
    in the last bits from a true early exit. Drive this per point if bit
    parity on z is ever required.
    """
    z = np.array(z_guess, dtype=np.float64, copy=True)
    p_target = np.asarray(p_target, dtype=np.float64)
    for _ in range(_N_NEWTON):
        piter = pressure(z, lat, c)
        titer = temperature(z, lat, c)
        dz = (piter - p_target) * (c.rdgas / c.grav) * titer / piter
        z = z + dz
        if np.all(np.abs(dz) < ZCONV_M):
            break
    return z
