#!/usr/bin/env python
"""#1455 SG: WHICH TORQUE owns the halved spin-up of the closed southern gyre?

CONTEXT (SD/SF, commit 47b520630): the 13 T-rows south of DINO's re-entrant
channel are WALLED AT BOTH ZONAL ENDS -- a closed basin -- and legoESM captures
only 52-58% of NEMO's +3.10 Sv/90 d spin-up of that closed circulation.  Local
density is EXONERATED as the controlling variable; the deficit is region-wide
and zonally coherent.  The next question named by that commit is the BAND'S
VORTICITY / CIRCULATION BUDGET.

WHAT THIS PROBE COMPUTES -- the exact bookkeeping, stated once
--------------------------------------------------------------
Because every southern row is blocked at both ends and T-row 0 is entirely dry
(measured below, fatal if not), the circulation of the closed sub-basin lying
SOUTH of u-row j is, by the discrete Stokes identity on the C grid,

    Gamma(j) = -R(j),      R(j) = sum_i e1u(i,j) * U(i,j),
    U(i,j)   = sum_k e3u_0 * u * umask                      [m2/s]

(the meridional legs of the circuit vanish because the east/west boundary
columns are land, and the southern leg vanishes because u-row 0 is dry).  So
the vorticity budget of the nested closed sub-basins IS the zonally-integrated
zonal-momentum budget of each u-row, and

    dR(j)/dt = Phi(j) = sum_i e1u(i,j) * [ integral of the zonal force / rho0 ]

with every term in [m3/s2].  Over the 90-day window 1 m3/s2 of torque builds
7.776e6 m3/s of R -- the same units R is reported in (1e6 m3/s), so a torque of
1 m3/s2 == 7.78 "Sv" of circulation per 90 days.

TERMS (each computed by ONE piece of code applied to BOTH models' states):
  WIND  : sum_i e1u * tau_u(lat)/rho0                      (surface deposit)
  DRAG  : sum_i e1u * (-r_u * u_bot)                       (bottom cell; NEMO
          zdfdrg ln_non_lin r = Cd0*sqrt(ubar^2+vbar^2+ke0) via the SHARED
          helper nemo_effective_bottom_drag_r, card values printed)
  CORI  : sum_i e1u * sum_k e3u_0 * f_u * vbar^u           (f*v, 4-pt average)
  PRES  : -(1/rho0) * sum_i [ dP - pb_face * dH ]          (Leibniz form: the
          bottom-pressure torque PLUS the side-wall pressure reaction, which in
          a walled band are the same single term -- see the derivation note in
          _pres_leibniz)
  RESID = dR/dt - (WIND + DRAG + CORI + PRES)
        = momentum advection + lateral/vertical viscosity + time-filter +
          every discretization difference between this offline diagnostic and
          either model's own operators.  It is a RESIDUAL BUCKET and can never
          by itself be an attribution (CLAUDE.md Rule 5).

SIGN CONVENTIONS (declared here, checked in the planted controls):
  * z is POSITIVE UP; column depth H > 0; e3* > 0.
  * u > 0 eastward; tau_u > 0 is the stress ON THE OCEAN, eastward.
  * R(j) > 0 means the row's depth-integrated flow is net EASTWARD.
    The closed sub-basin south of that row then has circulation Gamma = -R < 0
    in the counterclockwise-positive sense.  A POSITIVE Phi spins R up eastward.
  * DRAG opposes u (r >= 0, term carries the minus sign).
  * PRES as written is the force -grad(p)/rho0, so a positive PRES accelerates
    the row eastward.

CONTROLS -- all fatal, all printed before any number is interpreted
  C1 dtype    : every geometry/state array float64 (D.dtype_control + ours).
  C2 NaN      : non-finite on the wet mask is fatal, both sides, all days.
  C3 day-0    : every TERM must agree between lego and NEMO at the fp32
                snapshot quantum.  NOTE what this does and does not prove: it
                proves the arms START from NEMO's day-0 state (and that the v
                row offset, u column slice and eta alignment are right).  It
                does NOT validate the operators -- the same code runs on both
                sides, so an operator error is invisible to it.  The operators
                are validated in T0 against NEMO's own dumped trends.
  C4 RECONCILE (Rule 1e): the committed group series must be reproduced from
                THIS probe's machinery -- arm1 day-90 south-group deficit
                -1.320 Sv and NEMO's own +3.10 Sv 90-day spin-up, via
                D.group_transport (the recorded reduction, imported not copied).
  C5 geometry : u-row 0 dry, every southern row blocked at both ends, and the
                boundary v-columns dry -- the three facts the Stokes identity
                rests on.
  P1 planted  : +1 mm/s uniform on u must move DRAG by a hand-computed amount
                and leave WIND and CORI EXACTLY unchanged.
  P2 planted  : a rigid +0.1 m eta shift must leave PRES EXACTLY unchanged
                (analytically invariant -- so this control tests the
                IMPLEMENTATION, which is what a control is for).
  P3 REST     : the instrument's own noise floor for PRES, by ablation
                (Rule 3): replace T and S by their per-level horizontal mean
                and set eta = 0.  A resting, level-isopycnal ocean has zero
                pressure torque analytically; whatever this probe returns is
                its SPURIOUS pressure torque over the DINO topography, and no
                PRES difference smaller than it may be believed.

NEMO's own utrd_* / vtrd_* momentum-trend dumps DO exist in every 10-day
restart of RUN_90D_TWIN.  They are printed for information and are NOT used in
any attribution: measured here, their depth-integral does NOT close against the
state tendency (off by ~3 orders of magnitude, dominated by utrd_zdf, which is
not a telescoping vertical-flux divergence in these dumps).  Rule 5.

Diagnosis only: reads recorded artifacts, writes only its own npz.

Run (fp64):
  JAX_ENABLE_X64=1 .venv/bin/python .../southern_circulation_budget.py ARM.npz [...] \
      [--out-npz /path/budget.npz]
"""
import argparse
import glob
import os
import sys

import netCDF4 as nc
import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import acc_driver_decomp as D  # noqa: E402  (recorded reductions)
import acc_thermal_wind as A  # noqa: E402  (recorded harness: mesh, masks, EOS, J0)
import acceptance_gate_90d as G  # noqa: E402  (recorded loaders)
from legoesm.ocean.dynamics.ocean_tendency_common import (  # noqa: E402
    nemo_effective_bottom_drag_r,
)
from legoesm.ocean.experiments.dino import (  # noqa: E402
    dino_config_for_recipe,
    dino_wind_stress,
)
from rebuild_nemo_restart import rebuild  # noqa: E402

from legoesm import constants  # noqa: E402

DAYS = (0, 30, 60, 90)
NEMO_DAYS = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)
SEC_PER_DAY = 86400.0

# committed (47b520630 / b2ec54382) south-of-band numbers this probe must reproduce
COMMITTED_GSOUTH_D90 = {"arm1_pre": -1.320, "arm2_fixes": -1.297, "arm3_bn2": -1.483}
COMMITTED_NEMO_SPINUP = 3.10        # Sv, NEMO's own d90-d0 south-group change

# ------------------------------------------------------------------ geometry --
_mm = nc.Dataset(f"{A.DINO}/RUN_TRAJ/mesh_mask.nc")
_sq = lambda v: np.asarray(_mm[v][0], dtype=np.float64).squeeze()   # noqa: E731
e1u = _sq("e1u")                      # zonal spacing at u-points [m]
gphiu = _sq("gphiu")
e3u0 = A.llz(_mm["e3u_0"][0]).astype(np.float64)
e3v0 = A.llz(_mm["e3v_0"][0]).astype(np.float64)
e3t0 = np.asarray(A.e3t0, dtype=np.float64)
vmask = A.llz(_mm["vmask"][0]) > 0.5
umask, tmask = A.umask, A.tmask
NY, NX, _NZ = tmask.shape
J0 = A.J0                                   # southern band = T-rows 0..J0-1
ROWS = range(1, J0)                         # row 0 is dry (asserted in C5)

RHO0 = A.RHO0
# NEMO uses grav = 9.80665 (phycst.F90); legoesm.constants.g = 9.80616.  The
# 5e-5 relative difference moves PRES by ~0.015 m3/s2 and cancels to first order
# in the lego-NEMO difference, so the repo constant is kept (CLAUDE.md: physical
# constants come from legoesm.constants, in probes too).
G_ACC = constants.g
f_u = 2.0 * constants.Omega * np.sin(np.deg2rad(gphiu))     # (y,x) [1/s]

H_t = np.sum(np.where(tmask, e3t0, 0.0), axis=2)            # column depth [m]
KBOT_T = np.where(tmask.any(axis=2), tmask.sum(axis=2) - 1, 0)
KBOT_U = np.where(umask.any(axis=2), umask.sum(axis=2) - 1, 0)

# card values -- instantiated and printed, never assumed (Rule 10)
CARD = dino_config_for_recipe("nemo_dino_kamm_mlf")
DRAG_KW = dict(scheme=CARD.bottom_drag_scheme, cd0=float(CARD.C_d_bottom),
               cd_max=1.0e-1, z0=3.0e-3, ke0=2.5e-3,
               von_karman=constants.kappa_von_karman)

# analytic wind stress on the ocean [N/m2] at u-points (lat-only; verified
# against NEMO's own dumped utau_b in the instrument section)
TAU_U = np.asarray(dino_wind_stress(gphiu), dtype=np.float64)


# ------------------------------------------------------------------- reducers --
def row_circulation(u):
    """R(j) = sum_i e1u * sum_k e3u_0 u   [m3/s], per u-row.  POSITIVE = eastward."""
    U = np.sum(np.where(umask, np.asarray(u, np.float64) * e3u0, 0.0), axis=2)
    return np.sum(U * e1u, axis=1)


def phi_wind():
    """WIND torque [m3/s2] per row.  tau_u is lat-only and IDENTICAL on both
    sides by construction (the twin driver's analytic DINO forcing), so this
    term's lego-NEMO difference is ZERO by construction, not by measurement."""
    surf = umask[:, :, 0]
    return np.sum(np.where(surf, TAU_U / RHO0 * e1u, 0.0), axis=1)


def _drag_r_faces(u, v):
    """NEMO zdfdrg r = Cd0*sqrt(ubar^2+vbar^2+ke0) at T-points, 2-pt averaged to
    u-faces (dynzdf zCdu).  u(i) is the EAST face of T(i); v(j) the NORTH face."""
    us, vs = np.asarray(u, np.float64), np.asarray(v, np.float64)
    uw = np.where(umask, us, 0.0)
    vw = np.where(vmask, vs, 0.0)
    u_c = 0.5 * (np.concatenate([np.zeros_like(uw[:, :1]), uw[:, :-1]], axis=1) + uw)
    v_c = 0.5 * (np.concatenate([np.zeros_like(vw[:1]), vw[:-1]], axis=0) + vw)
    k = KBOT_T[:, :, None]
    u_bot_t = np.take_along_axis(u_c, k, axis=2)[:, :, 0]
    v_bot_t = np.take_along_axis(v_c, k, axis=2)[:, :, 0]
    h_bot_t = np.take_along_axis(e3t0, k, axis=2)[:, :, 0]
    r_t = np.asarray(nemo_effective_bottom_drag_r(u_bot_t, v_bot_t, h_bot_t, **DRAG_KW))
    r_t = np.where(tmask[:, :, 0], r_t, 0.0)
    r_u = np.zeros_like(r_t)
    r_u[:, :-1] = 0.5 * (r_t[:, :-1] + r_t[:, 1:])
    return r_u


def phi_drag(u, v):
    """DRAG torque [m3/s2] per row: sum_i e1u * (-r_u * u_bottom).  The bottom
    cell thickness cancels in the depth integral of -r*u/h_bot."""
    r_u = _drag_r_faces(u, v)
    us = np.where(umask, np.asarray(u, np.float64), 0.0)
    u_bot = np.take_along_axis(us, KBOT_U[:, :, None], axis=2)[:, :, 0]
    col = umask.any(axis=2)
    return np.sum(np.where(col, -r_u * u_bot * e1u, 0.0), axis=1)


def phi_cori(v):
    """CORI torque [m3/s2] per row: sum_i e1u * sum_k e3u_0 * f_u * vbar^u,
    vbar^u = 1/4 (v(i,j-1)+v(i,j)+v(i+1,j-1)+v(i+1,j))."""
    vw = np.where(vmask, np.asarray(v, np.float64), 0.0)
    v_jm = np.concatenate([np.zeros_like(vw[:1]), vw[:-1]], axis=0)      # v(j-1)
    v_ns = 0.5 * (vw + v_jm)                                            # at T-point
    v_u = np.zeros_like(v_ns)
    v_u[:, :-1] = 0.5 * (v_ns[:, :-1] + v_ns[:, 1:])                    # at u-point
    integ = np.sum(np.where(umask, e3u0 * v_u, 0.0), axis=2)
    return np.sum(integ * f_u * e1u, axis=1)


def _rho_wet(T, S):
    """In-situ density on wet cells [kg/m3], via the recorded A.rho_of
    (nemo_seos_eos, the card's S-EOS).  A NaN on a WET cell is FATAL -- this
    term is a cancelling difference of ~1e10 Pa m column integrals, so a
    plausible-looking 0.0 substituted for a NaN would be invisible downstream."""
    r = np.asarray(A.rho_of({"T": T, "S": S}, tmask), np.float64)
    bad = int(np.sum(~np.isfinite(r[tmask])))
    if bad:
        raise SystemExit(f"FATAL: the EOS returned {bad} non-finite values on WET "
                         "cells -- refusing to substitute a finite sentinel")
    return np.where(tmask, r, 0.0)


def _column_pressure(T, S, eta, zstar=False):
    """(P, p_b, H_col) per column: P = integral of p dz [Pa m], p_b = sea-floor
    pressure [Pa], H_col = the column thickness the Leibniz term must use.

    zstar=False: fixed top at z = 0, thicknesses e3t_0, surface load rho0*g*eta
    carried inside p, H_col = sum(e3t_0).  The free-surface slab is dropped.
    zstar=True: thicknesses stretched by (1 + eta/H) and H_col = H + eta, i.e.
    the real z-star column with the slab included.
    """
    rho = _rho_wet(T, S)
    dz = np.where(tmask, e3t0, 0.0)
    H = np.sum(dz, axis=2)
    et = np.asarray(eta, np.float64)
    # H_col is the LOWER limit of the Leibniz identity and is ALWAYS the depth of
    # the sea floor: int_{-H}^{eta} dx p dz = dx P - p(eta) dx eta - p_b dx H, and
    # p(eta) = 0.  (An earlier version used H + eta here; that flips the sign of
    # the surface-pressure-gradient contribution and shifted the NEMO band mean by
    # ~650 m3/s2.  Caught by this probe's own variant comparison.)
    H_col = H
    if zstar:
        stretch = 1.0 + np.where(H > 0, et / np.where(H > 0, H, 1.0), 0.0)
        dz = dz * stretch[:, :, None]
        p_top = np.zeros_like(et)                    # p = 0 at the free surface
    else:
        p_top = RHO0 * G_ACC * et                    # rigid top at z = 0
    inc = G_ACC * rho * dz                                      # weight of each cell
    cum_above = np.cumsum(inc, axis=2) - inc                    # weight above cell top
    p_mid = p_top[:, :, None] + cum_above + 0.5 * inc           # p at cell centre
    P = np.sum(p_mid * dz, axis=2)
    p_b = p_top + np.sum(inc, axis=2)
    return P, p_b, H_col


def phi_pres(T, S, eta, zstar=False):
    """PRES torque [m3/s2] per row, Leibniz (bottom-pressure) form.

    ``zstar=True`` integrates the REAL column: layer thicknesses stretched by
    (1 + eta/H) and the total thickness H + eta in the topographic term, i.e.
    the free-surface slab is included instead of dropped.  The difference
    between the two is measured in the PRES uncertainty budget, not asserted.

    The depth-integrated zonal pressure force per unit mass is
        -(1/rho0) * int dp/dx dz = -(1/rho0) * ( dP/dx - p_b * dH/dx )
    (Leibniz, with p = 0 at the free surface absorbed into P).  Multiplying by
    e1u and summing along the row therefore needs NO grid spacing at all:
        Phi = -(1/rho0) * sum_over_wet_u_faces [ (P(i+1)-P(i)) - pb_face*(H(i+1)-H(i)) ]
    In a walled band the first difference telescopes to the pressure difference
    across the basin (the SIDE-WALL reaction) and the second is the topographic
    form stress: the two are inseparable here and are reported as ONE term --
    which is exactly what the area-integrated Jacobian J(p_b, H) is.

    Accuracy: for a resting, level-isopycnal ocean the mid-point p_b makes the
    two pieces cancel to third order in the topographic step; the P3 REST
    ablation measures what is actually left over the DINO bathymetry.

    APPROXIMATION, stated because it is not zero: the column is integrated from
    a FIXED top at z = 0 with the surface load rho0*g*eta carried in p, so the
    Leibniz identity above is exact for that column and the free-surface slab
    between 0 and eta is dropped.  That slab contributes g*d(eta^2)/2 across the
    basin -- of order 1 m3/s2 here, i.e. small against this probe's own closure
    error but NOT against the deficit.  It is the same structure on both sides
    and therefore largely cancels in a lego-NEMO difference.  z-star layer
    stretching is likewise not applied to e3t_0 (an O(eta/H) ~ 1e-4 effect).
    """
    P, p_b, H_col = _column_pressure(T, S, eta, zstar=zstar)
    face = umask.any(axis=2)                                    # wet u-face columns
    dP = np.zeros_like(P)
    dH = np.zeros_like(P)
    pb_f = np.zeros_like(P)
    dP[:, :-1] = P[:, 1:] - P[:, :-1]
    dH[:, :-1] = H_col[:, 1:] - H_col[:, :-1]
    pb_f[:, :-1] = 0.5 * (p_b[:, 1:] + p_b[:, :-1])
    contrib = np.where(face, -(dP - pb_f * dH) / RHO0, 0.0)
    return np.sum(contrib, axis=1)


def phi_pres_levelform(T, S, eta):
    """Same term, z-LEVEL discretization: -(1/rho0) sum_k e3u_0 (p(i+1,k)-p(i,k)).
    Ignores the partial-cell depth mismatch NEMO's zps correction handles, so it
    is the CRUDE alternative -- reported only as a spread against phi_pres."""
    rho = _rho_wet(T, S)
    dz = np.where(tmask, e3t0, 0.0)
    inc = G_ACC * rho * dz
    p_mid = (RHO0 * G_ACC * np.asarray(eta, np.float64)[:, :, None]
             + (np.cumsum(inc, axis=2) - inc) + 0.5 * inc)
    dp = np.zeros_like(p_mid)
    dp[:, :-1] = p_mid[:, 1:] - p_mid[:, :-1]
    return np.sum(np.where(umask, -e3u0 * dp / RHO0, 0.0), axis=(1, 2))


def terms(st):
    """All four torques [m3/s2] per u-row for one state."""
    return {"WIND": phi_wind(),
            "DRAG": phi_drag(st["u"], st["v"]),
            "CORI": phi_cori(st["v"]),
            "PRES": phi_pres(st["T"], st["S"], st["eta"])}


# -------------------------------------------------------------------- loaders --
def load_nemo(day):
    kt = G.KT_RESTART + day * G.STEPS_PER_DAY
    pat = f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart*.nc"
    if not glob.glob(pat):
        raise SystemExit(f"FATAL: no NEMO restart for day {day} ({pat})")
    raw = rebuild(pat, ["tn", "sn", "un", "vn", "sshn"])
    yxz = lambda a: np.moveaxis(np.asarray(a, np.float64), 0, -1)   # noqa: E731
    return {"T": yxz(raw["tn"]), "S": yxz(raw["sn"]), "u": yxz(raw["un"]),
            "v": yxz(raw["vn"]), "eta": np.asarray(raw["sshn"], np.float64).squeeze()}


def load_lego(npz_path, npz, day, voff):
    st = G.load_candidate(npz_path, day)
    st["v"] = np.asarray(npz[f"v3d_day{day}"], np.float64)[voff:voff + NY]
    st["eta"] = np.asarray(npz[f"eta3d_day{day}"], np.float64)
    return st


def pick_v_offset(npz, nemo0):
    """Control V (verbatim from southern_depth_mean_map.py): the lego v array
    holds 200 v-faces; which 199 map to NEMO's vn rows is MEASURED at day 0."""
    ref = np.where(vmask, nemo0["v"], 0.0)
    sc = {o: float(np.max(np.abs(np.where(vmask, np.asarray(npz["v3d_day0"], np.float64)
                                          [o:o + NY], 0.0) - ref))) for o in (0, 1)}
    best = min(sc, key=sc.get)
    print("[control V] lego v row offset: "
          + ", ".join(f"off {o}: {s:.3e} m/s" for o, s in sorted(sc.items()))
          + f"  -> using offset {best}")
    if not (sc[best] < 1e-5 and sc[1 - best] > 100 * max(sc[best], 1e-12)):
        raise SystemExit("FATAL control V: the v row offset is not decided by the "
                         "day-0 identity")
    return best


# ------------------------------------------------------------------- controls --
def c5_geometry():
    print("\n[C5 geometry -- the three facts the Stokes identity rests on]")
    n_u0 = int(umask[0].sum())
    print(f"    u-row 0 wet cells: {n_u0} (want 0 -- the southern wall closes the circuit)")
    if n_u0:
        raise SystemExit("FATAL C5: u-row 0 is not dry; R(0) != 0 and the nested "
                         "sub-basin circulation is not -R(j)")
    blocked = [int(NX - tmask[j, :, 0].sum()) for j in ROWS]
    print(f"    dry surface columns per southern row: {blocked} (0 would mean re-entrant)")
    if min(blocked) == 0:
        raise SystemExit("FATAL C5: a southern row is zonally UNBLOCKED")
    nv = int(vmask[list(ROWS), 0].sum()) + int(vmask[list(ROWS), NX - 1].sum())
    print(f"    wet v cells on the east/west boundary columns of the band: {nv} "
          "(want 0 -- the meridional legs of the circuit)")
    if nv:
        raise SystemExit("FATAL C5: the boundary v-columns are wet; the circuit's "
                         "meridional legs do not vanish")


def instrument_wind():
    """The wind field is the one term that must be identical by construction --
    verify NEMO's OWN dumped stress against the analytic profile this probe uses."""
    raw = rebuild(f"{G.RUN_90D_TWIN}/DINO_{G.KT_RESTART:08d}_restart*.nc",
                  ["utau_b", "vtau_b"])
    ut = np.asarray(raw["utau_b"], np.float64).squeeze()
    vt = np.asarray(raw["vtau_b"], np.float64).squeeze()
    m = umask[:, :, 0]
    d = float(np.max(np.abs((ut - TAU_U)[m])))
    print(f"\n[instrument WIND] max|NEMO utau_b - analytic tau_u| on wet u-faces "
          f"= {d:.3e} Pa (peak |tau| {float(np.abs(ut[m]).max()):.4f} Pa); "
          f"max|NEMO vtau_b| = {float(np.abs(vt[m]).max()):.3e} Pa")
    if d > 1e-4:
        raise SystemExit("FATAL: the analytic wind this probe integrates is not "
                         "the wind NEMO applied")


def planted(st):
    """P1 / P2 -- planted controls with ANALYTIC, non-circular predictions.

    Both reviewers of this probe found the first version's checks vacuous:
    ``phi_wind() - phi_wind()`` is x - x, ``phi_cori`` never reads u so a u
    perturbation could not move it, and the drag prediction was built from the
    same helper ``phi_drag`` calls (a swapped stencil still passed at 8e-17).
    The predictions below are written out from the namelist formula and the
    grid, touching no helper the tested function uses.
    """
    print("\n[P1a planted single-face u, analytic drag prediction]")
    # one wet u-face, everything else at rest: the whole chain (T-point average,
    # namelist drag law, face average, bottom-level pick, e1u weight) reduces to
    # three lines that share no code with phi_drag.
    j0 = int(ROWS[len(ROWS) // 2])
    i0 = int(np.where(umask[j0, :, 0])[0][10])
    k0 = int(KBOT_U[j0, i0])
    u1 = np.zeros_like(np.asarray(st["u"], np.float64))
    v1 = np.zeros_like(np.asarray(st["v"], np.float64))
    u1[j0, i0, k0] = 0.30
    got = phi_drag(u1, v1)[j0]
    # namdrg_bot literals read from NEMO's own namelist_ref (rn_Cd0, rn_ke0), NOT
    # from DRAG_KW: sharing the parameters with the code under test made this
    # control blind to a dropped ke0 (verified -- it was NOT CAUGHT before).
    cd0, ke0 = 1.0e-3, 2.5e-3
    if (abs(DRAG_KW["cd0"] - cd0) > 0 or abs(DRAG_KW["ke0"] - ke0) > 0
            or DRAG_KW["scheme"] != "nemo_quadratic"):
        raise SystemExit("FATAL P1a: the legoESM card's bottom drag "
                         f"({DRAG_KW['scheme']}, cd0={DRAG_KW['cd0']}, "
                         f"ke0={DRAG_KW['ke0']}) does not match NEMO namdrg_bot "
                         "(ln_non_lin, rn_Cd0=1e-3, rn_ke0=2.5e-3)")
    # the face's u reaches T-points i0 and i0+1 with weight 1/2 each, but only
    # where that T-column's own bottom level is k0 (zdf_drg uses mbkt)
    r_t = [cd0 * np.sqrt((0.5 * 0.30) ** 2 + ke0) if KBOT_T[j0, i] == k0
           else cd0 * np.sqrt(ke0) for i in (i0, i0 + 1)]
    want = -0.5 * (r_t[0] + r_t[1]) * 0.30 * e1u[j0, i0]
    print(f"    row {j0}, face i={i0}, k={k0}, u=0.30 m/s -> DRAG {got:.6e}, "
          f"hand-computed {want:.6e} m3/s2  (rel {abs(got - want) / abs(want):.2e})")
    if abs(got - want) > 1e-10 * abs(want):
        raise SystemExit("FATAL P1a: the drag stencil, mask, face average, bottom "
                         "level or namelist coefficient does not match the hand "
                         "calculation")

    print("\n[P1b planted single-face v, analytic Coriolis prediction]")
    # one wet v-face reaches two T-points at weight 1/2, then four u-points at
    # weight 1/4 -- an exactly predictable f*e3u*e1u sum.
    jv = int(ROWS[len(ROWS) // 2])
    iv = int(np.where(vmask[jv, :, 0])[0][10])
    kv = 3
    if not vmask[jv, iv, kv]:
        raise SystemExit("FATAL P1b: the planted v cell is dry -- control vacuous")
    v2 = np.zeros_like(np.asarray(st["v"], np.float64))
    v2[jv, iv, kv] = 1.0
    got = phi_cori(v2)
    want = np.zeros(NY)
    for jt in (jv, jv + 1):                       # v(j) touches T-rows j and j+1
        for iu in (iv - 1, iv):                   # T(i) feeds u-faces i-1 and i
            if 0 <= iu < NX and 0 <= jt < NY and umask[jt, iu, kv]:
                want[jt] += 0.25 * e3u0[jt, iu, kv] * f_u[jt, iu] * e1u[jt, iu]
    err = float(np.max(np.abs((got - want)[list(ROWS)])))
    sc = float(np.max(np.abs(want[list(ROWS)])))
    if sc <= 0:
        raise SystemExit("FATAL P1b: the planted v cell moved nothing -- vacuous")
    print(f"    row {jv}, face i={iv}, k={kv}, v=1 m/s -> CORI matches the "
          f"1/4-weight f*e3u*e1u sum: max|err| {err:.3e} on {sc:.3e} m3/s2")
    if err > 1e-10 * sc:
        raise SystemExit("FATAL P1b: the Coriolis stencil or mask does not match "
                         "the hand calculation")
    # Rule 2 -- what this control CANNOT see, measured rather than asserted: the
    # diagnostic is a ZONAL SUM, so shifting which u-face the v->u average feeds
    # is nearly invisible to it.  Quantify that blind spot here.
    vw = np.where(vmask, np.asarray(st["v"], np.float64), 0.0)
    v_ns = 0.5 * (vw + np.concatenate([np.zeros_like(vw[:1]), vw[:-1]], axis=0))
    shifted = np.zeros_like(v_ns)
    shifted[:, 1:] = 0.5 * (v_ns[:, :-1] + v_ns[:, 1:])          # the wrong-side average
    alt = np.sum(np.sum(np.where(umask, e3u0 * shifted, 0.0), axis=2) * f_u * e1u, axis=1)
    ref = phi_cori(st["v"])
    rel = float(np.max(np.abs((alt - ref)[list(ROWS)]))
                / max(float(np.max(np.abs(ref[list(ROWS)]))), 1e-30))
    print(f"    SENSITIVITY + BLIND SPOT (both measured): on the REAL v field, "
          f"shifting the v->u\n    average one face zonally changes the row integral "
          f"by {100 * rel:.0f}% of the term -- CORI is\n    NOT robust to that choice. "
          "Yet the single-cell control above cannot see it (err\n    stays 0.0 under "
          "the shifted stencil, verified), because e1u/e3u/f are smooth so\n    one "
          "planted cell moves to a near-identical face.  CORI's ONLY real evidence is\n"
          "    therefore the comparison with NEMO's own utrd_pvo in T0 -- and NEMO uses "
          "EEN\n    (namelist_cfg:331 ln_dynvor_een=.true.), not a 4-point average, so "
          "that agreement\n    is empirical, holds on NEMO's states only, and has NO "
          "counterpart on the lego side.")

    print("\n[P1c bulk +1 mm/s on u]  (a consistency check, not a stencil test)")
    du = 1.0e-3
    pert = dict(st, u=st["u"] + du)
    d_drag = phi_drag(pert["u"], pert["v"]) - phi_drag(st["u"], st["v"])
    print(f"    DRAG moves by {float(np.mean(d_drag[list(ROWS)])):+.4f} m3/s2 (band "
          "mean).  WIND and CORI cannot move: phi_wind takes no state at all and "
          "phi_cori\n    takes only v, so their invariance here is STRUCTURAL and "
          "is not asserted.")
    if float(np.max(np.abs(d_drag[list(ROWS)]))) <= 0.0:
        raise SystemExit("FATAL P1c: a uniform u shift did not move the drag at all")

    print("\n[P2 planted rigid +0.1 m eta shift]")
    p0 = phi_pres(st["T"], st["S"], st["eta"])
    p1 = phi_pres(st["T"], st["S"], st["eta"] + 0.1)
    d = float(np.max(np.abs((p1 - p0)[list(ROWS)])))
    sc = float(np.max(np.abs(p0[list(ROWS)])))
    print(f"    PRES shift {d:.3e} m3/s2 on a term of size {sc:.3e}.  A rigid "
          "surface load cancels\n    exactly between dP and pb*dH; this control "
          "DOES fail on a sign flip between the\n    two pieces or an "
          "H_t / p_b inconsistency (both verified by planting them).")
    if d > 1e-6 * max(sc, 1.0):
        raise SystemExit("FATAL P2: a rigid eta shift moved the pressure torque")


def rest_identity(st):
    """P3 -- NOT a noise floor.  RETRACTED and relabelled after review.

    The original version replaced T,S by their per-level horizontal mean, set
    eta = 0, and called the result "this diagnostic's spurious pressure torque
    over the DINO bathymetry".  That was wrong on two counts, both measured:

      * This DINO run is FULL-STEP, not partial-step (namelist_cfg:71
        ln_zps_nam = .false.; measured: the horizontal spread of e3t_0 within
        every level is exactly 0.0).  There is no partial cell for a
        discretization error to live in, so the "3rd order in the topographic
        step" claim was moot as well.
      * With rho = rho(k) and full cells, dP and pb*dH cancel ALGEBRAICALLY for
        any bathymetry.  The 1e-9 that came back was float64 roundoff, and the
        z-level arm returned exactly 0.0 -- the tell.  The ablation deletes the
        horizontal density gradient that PRES exists to measure, so it can
        never bound PRES's error.

    What it still IS: a real gross-error check (a sign flip between the two
    Leibniz pieces, or a broken mask, moves it by ~1e7).  Kept as that, named
    as that.  The honest PRES uncertainty is measured in pres_uncertainty().
    """
    Tm = np.where(tmask, st["T"], np.nan)
    Sm = np.where(tmask, st["S"], np.nan)
    have = tmask.any(axis=(0, 1))
    if not have.any():
        raise SystemExit("FATAL P3: no wet level at all")
    with np.errstate(invalid="ignore"):
        Tb = np.where(have, np.nanmean(np.where(have, Tm, 0.0), axis=(0, 1)), np.nan)
        Sb = np.where(have, np.nanmean(np.where(have, Sm, 0.0), axis=(0, 1)), np.nan)
    if not (np.all(np.isfinite(Tb[have])) and np.all(np.isfinite(Sb[have]))):
        raise SystemExit("FATAL P3: a populated level has a non-finite mean")
    Tb = np.where(have, Tb, Tb[have][-1])
    Sb = np.where(have, Sb, Sb[have][-1])
    Tr = np.broadcast_to(Tb, tmask.shape).copy()
    Sr = np.broadcast_to(Sb, tmask.shape).copy()
    z = np.zeros((NY, NX))
    return phi_pres(Tr, Sr, z), phi_pres_levelform(Tr, Sr, z)


def pres_uncertainty(nemo, lego, day=90):
    """THE honest uncertainty on PRES, by disagreement between defensible
    discretizations of the SAME term applied to the SAME states (Rule 3: the
    instrument's floor is measured, not asserted).

    Three variants, each a legitimate way to write the depth-integrated zonal
    pressure force, all applied identically to both models:
      L   Leibniz form, fixed top at z = 0 (what this probe reports)
      Z   the same, but on the real z-star column (layers stretched by
          1 + eta/H, topographic term using H + eta -- i.e. the free-surface
          slab included rather than dropped)
      K   the crude z-level form, sum_k e3u_0 * (p(i+1,k) - p(i,k))
    What matters for the verdict is not their absolute spread but the spread of
    the LEGO-NEMO DIFFERENCE, because a systematic error common to both states
    cancels there.  Both are printed.
    """
    rows = list(ROWS)
    print("\n" + "=" * 112)
    print(f"P4 PRES UNCERTAINTY BUDGET at day {day} [m3/s2] -- the real floor on the "
          "pressure term")
    print("=" * 112)
    variants = {
        "L  Leibniz, fixed top (reported)": lambda st: phi_pres(st["T"], st["S"], st["eta"]),
        "Z  Leibniz, real z-star column  ": lambda st: phi_pres(st["T"], st["S"], st["eta"],
                                                                zstar=True),
        "K  crude z-level form           ": lambda st: phi_pres_levelform(st["T"], st["S"],
                                                                          st["eta"]),
    }
    vn = {k: f(nemo[day]) for k, f in variants.items()}
    out = {}
    for n in lego:
        vl = {k: f(lego[n][day]) for k, f in variants.items()}
        dif = {k: (vl[k] - vn[k])[rows] for k in variants}
        base = dif["L  Leibniz, fixed top (reported)"]
        print(f"  arm {n}")
        print(f"    {'variant':34s}{'NEMO band mean':>16s}{'lego-NEMO mean':>16s}"
              f"{'vs L, max|d|':>14s}")
        for k in variants:
            spread = float(np.max(np.abs(dif[k] - base)))
            print(f"    {k:34s}{float(np.mean(vn[k][rows])):16.2f}"
                  f"{float(np.mean(dif[k])):16.3f}{spread:14.3f}")
        worst = max(float(np.max(np.abs(dif[k] - base))) for k in variants)
        out[n] = worst
        print(f"    -> the CHOICE of discretization moves the lego-NEMO pressure "
              f"difference by up to\n       {worst:.2f} m3/s2.  Compare that with the "
              "per-row deficit this probe must explain.")
    return out


# ------------------------------------------- NEMO's own dumped momentum trends --
# RUN_90D_TWIN's 10-day restarts carry the full trddyn momentum decomposition,
# written by the lane's own DINO MY_SRC/trddump.F90.  COVERAGE (Rule 1): the
# restart defines 12 momentum trends.  Nine are loaded below.  The three not
# loaded are WAIVED with a reason, not overlooked:
#   utrd_tau  -- identically 0.0 in every dump (measured): with ln_drgimp = .true.
#                the surface stress is applied inside dynzdf, not as a separate
#                explicit trend, so NEMO never fills this slot.
#   utrd_bfr  -- identically 0.0, same reason (explicit bottom friction is off).
#   utrd_bfri -- identically 0.0, same reason (implicit drag is folded into the
#                dynzdf matrix, not diagnosed separately).
# The waiver is ENFORCED below: if any of the three is ever non-zero the probe
# raises rather than silently ignoring a term.
NEMO_TRENDS = ("hpg", "spg", "keg", "rvo", "pvo", "zad", "ldf", "zdf", "atf")
NEMO_TRENDS_WAIVED = ("tau", "bfr", "bfri")


def _row_int_trend(tr):
    """sum_i e1u * sum_k e3u_0 * trend  [m3/s2] per u-row."""
    return np.sum(np.sum(np.where(umask, np.asarray(tr, np.float64) * e3u0, 0.0),
                         axis=2) * e1u, axis=1)


def _zdf_is_barotropic_subtraction(zdf, un, day):
    """Measure -- not merely read -- WHY utrd_zdf's depth-integral is unusable AS
    DUMPED, and recover the true term.

    DINO MY_SRC/dynzdf.F90 saves the pre-solve RHS at :124 and diagnoses the
    trend at :597 as (puu(Kaa)-puu(Kbb))*r1_Dt minus that RHS -- but at :168 it
    has already SUBTRACTED the barotropic mode from puu(Kaa):
        puu(ji,jj,jk,Kaa) = ( puu(ji,jj,jk,Kaa) - uu_b(ji,jj,Kaa) ) * umask
    and never adds it back before the trend is taken.  PREDICTION: the depth-MEAN
    of utrd_zdf equals -u_barotropic/(2*rdt) to a few parts in 1e3.  Measured
    here; if it fails, the explanation in the text is wrong.

    CONSEQUENCE, and the reason this matters: the true vertical-diffusion trend
    is RECOVERABLE as row_int(utrd_zdf) + R/(2*rdt).  Depth-integrated, that is
    NEMO's OWN (surface stress - bottom drag)/rho0 -- an independent oracle check
    on two of this probe's four terms.  An earlier version of this probe called
    utrd_zdf simply "not a telescoping flux divergence" and threw it away; that
    was wrong and is retracted here.
    """
    H = np.sum(np.where(umask, e3u0, 0.0), axis=2)
    ok = H > 0
    Hs = np.where(ok, H, 1.0)
    ub = np.sum(np.where(umask, np.asarray(un, np.float64) * e3u0, 0.0), axis=2) / Hs
    zb = np.sum(np.where(umask, np.asarray(zdf, np.float64) * e3u0, 0.0), axis=2) / Hs
    band = ok.copy()
    band[J0:] = False
    band[0] = False
    if int(band.sum()) == 0:
        raise SystemExit("FATAL: the utrd_zdf test selected no columns")
    _f = sorted(glob.glob(f"{G.RUN_90D_TWIN}/DINO_{G.KT_RESTART:08d}_restart*.nc"))[0]
    dt2 = 2.0 * float(np.asarray(nc.Dataset(_f)["rdt"][...]))
    pred = -ub / dt2
    rel = float(np.median(np.abs(zb - pred)[band] / np.maximum(np.abs(pred)[band], 1e-30)))
    slope = float(np.polyfit(pred[band], zb[band], 1)[0])
    print(f"    [utrd_zdf test, day {day}] depth-mean(utrd_zdf) vs "
          f"-u_barotropic/(2*rdt={dt2:.0f}s) over the band: "
          f"corr {np.corrcoef(zb[band], pred[band])[0, 1]:+.5f}, slope {slope:.4f}, "
          f"median rel err {rel:.4f}")
    if not (abs(slope - 1.0) < 0.02 and rel < 0.02):
        raise SystemExit("FATAL: utrd_zdf is NOT the barotropic-subtraction artifact "
                         "this probe's text claims it is -- fix the text before "
                         "quoting any of these numbers")
    return dt2


def nemo_trend_calibration(Rn, TN):
    yxz = lambda a: np.moveaxis(np.asarray(a, np.float64), 0, -1)   # noqa: E731
    print("\n" + "=" * 112)
    print("T0  INSTRUMENT CALIBRATION against NEMO's OWN dumped momentum trends\n"
          "    (utrd_* in every RUN_90D_TWIN 10-day restart, row-integrated the same\n"
          "    way as this probe's terms).")
    print("=" * 112)
    for day in (0, 90):
        kt = G.KT_RESTART + day * G.STEPS_PER_DAY
        raw = rebuild(f"{G.RUN_90D_TWIN}/DINO_{kt:08d}_restart*.nc",
                      [f"utrd_{t}" for t in NEMO_TRENDS + NEMO_TRENDS_WAIVED] + ["un"])
        for t in NEMO_TRENDS_WAIVED:                 # the waiver, enforced
            mx = float(np.max(np.abs(np.asarray(raw[f"utrd_{t}"], np.float64))))
            if mx != 0.0:
                raise SystemExit(f"FATAL coverage: utrd_{t} is non-zero ({mx:.3e}) "
                                 f"at day {day} -- it was WAIVED as identically zero "
                                 "and is not in the budget")
        tr = {t: _row_int_trend(yxz(raw[f"utrd_{t}"])) for t in NEMO_TRENDS}
        Rnow = row_circulation(yxz(raw["un"]))
        dt2 = _zdf_is_barotropic_subtraction(yxz(raw["utrd_zdf"]), yxz(raw["un"]), day)
        zdf_true = tr["zdf"] + Rnow / dt2            # the recovered true zdf term
        dumped_pres = tr["hpg"] + tr["spg"]
        dumped_all = sum(tr.values())
        dumped_nozdf = dumped_all - tr["zdf"]
        inst = ((Rn[10] - Rn[0]) if day == 0 else (Rn[90] - Rn[80])) / (10 * SEC_PER_DAY)
        mine = (TN[day]["WIND"] + TN[day]["DRAG"] + TN[day]["CORI"] + TN[day]["PRES"])
        rows = list(ROWS)

        print(f"\n  day {day} [m3/s2] -- operator-by-operator")
        print(f"{'row':>4s}{'this WIND+DRAG':>16s}{'NEMO zdf recovered':>20s}{'d':>9s}"
              f"{'this CORI':>11s}{'NEMO pvo':>11s}{'d':>8s}"
              f"{'this PRES':>11s}{'NEMO hpg+spg':>14s}{'d':>9s}"
              f"{'NEMO ldf':>10s}{'keg+rvo+zad':>13s}")
        wd = TN[day]["WIND"] + TN[day]["DRAG"]
        for j in ROWS:
            print(f"{j:4d}{wd[j]:16.2f}{zdf_true[j]:20.2f}{wd[j] - zdf_true[j]:9.2f}"
                  f"{TN[day]['CORI'][j]:11.2f}{tr['pvo'][j]:11.2f}"
                  f"{TN[day]['CORI'][j] - tr['pvo'][j]:8.2f}"
                  f"{TN[day]['PRES'][j]:11.2f}{dumped_pres[j]:14.2f}"
                  f"{TN[day]['PRES'][j] - dumped_pres[j]:9.2f}"
                  f"{tr['ldf'][j]:10.2f}{tr['keg'][j] + tr['rvo'][j] + tr['zad'][j]:13.2f}")
        big = np.abs(wd[rows]) > 5.0
        if big.any():
            e = np.abs((wd - zdf_true)[rows])[big] / np.abs(wd[rows])[big]
            print(f"    WIND+DRAG vs NEMO's own recovered vertical-diffusion term, "
                  f"rows where |WIND+DRAG| > 5: max rel {float(e.max()):.3f} "
                  f"({int(big.sum())} rows).  This is an ORACLE check on two of the "
                  "four terms.")
        print("    NOTE the 'this PRES vs NEMO hpg+spg' column is NOT a clean operator "
              "error: NEMO's\n    split-explicit barotropic solve routes the "
              "depth-integrated surface stress through\n    utrd_spg, and this probe's "
              "PRES carries only the pressure gradient (its own eta\n    terms combine "
              "to exactly -g*H_face*d(eta), the surface-pressure-gradient force).")

        print("\n    CLOSURE, both instruments, against the SAME measured tendency "
              "(Rule 5):")
        handed = mine + tr["ldf"] + tr["keg"] + tr["rvo"] + tr["zad"]
        print(f"      {'row':>4s}{'dR/dt(10d)':>12s}{'NEMO all-but-zdf':>18s}{'err':>8s}"
              f"{'this probe':>12s}{'err':>8s}{'+NEMO ldf/adv':>14s}{'err':>8s}")
        for j in ROWS:
            print(f"      {j:4d}{inst[j]:12.2f}{dumped_nozdf[j]:18.2f}"
                  f"{dumped_nozdf[j] - inst[j]:8.2f}{mine[j]:12.2f}{mine[j] - inst[j]:8.2f}"
                  f"{handed[j]:14.2f}{handed[j] - inst[j]:8.2f}")
        e_n = np.abs(dumped_nozdf - inst)[rows]
        e_m = np.abs(mine - inst)[rows]
        e_h = np.abs(handed - inst)[rows]
        print(f"      band max closure error:  NEMO's own dumped operators "
              f"{float(e_n.max()):.2f} m3/s2   |   THIS OFFLINE PROBE "
              f"{float(e_m.max()):.2f} m3/s2   |   the same probe with NEMO's own\n"
              f"      lateral-viscosity and advection trends added "
              f"{float(e_h.max()):.2f} m3/s2")
        print(f"      sum of ALL dumped trends INCLUDING utrd_zdf AS DUMPED, band max "
              f"|.| = {float(np.max(np.abs(dumped_all[rows]))):.3e} m3/s2 -- that is "
              "the\n      barotropic-subtraction artifact above, not physics.  "
              "Dropping utrd_zdf (whose\n      depth-integrated content is already "
              "carried by utrd_spg through the barotropic\n      solve) leaves a set "
              "that closes to the numbers above.  Adding this probe's\n      "
              "WIND+DRAG on top double-counts, which is why it makes the closure "
              "worse.")


# ----------------------------------------------------------------------- main --
def _trap(vals, days):
    """Time integral [m3/s] of a torque sampled at `days` (trapezoid)."""
    v = np.asarray(vals, np.float64)
    t = np.asarray(days, np.float64) * SEC_PER_DAY
    return np.trapezoid(v, t, axis=0) if hasattr(np, "trapezoid") else np.trapz(v, t, axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--out-npz", default=None)
    args = ap.parse_args(argv)

    # ---- C1 dtype ----------------------------------------------------------
    D.dtype_control()
    for nm, a in (("e1u", e1u), ("e3u_0", e3u0), ("e3v_0", e3v0), ("e3t_0", e3t0),
                  ("gphiu", gphiu), ("H_t", H_t), ("TAU_U", TAU_U)):
        print(f"    {nm:9s} {np.asarray(a).dtype}")
        if np.asarray(a).dtype != np.float64:
            raise SystemExit(f"FATAL C1: {nm} is not float64")
    print(f"[drag] scheme + cd0 from the legoESM card "
          f"(dino_config_for_recipe('nemo_dino_kamm_mlf')): {DRAG_KW['scheme']}, "
          f"cd0={DRAG_KW['cd0']}\n       ke0={DRAG_KW['ke0']}, cd_max={DRAG_KW['cd_max']}, "
          f"z0={DRAG_KW['z0']} are this probe's literals, checked against NEMO "
          "namdrg_bot in P1a\n       (cd_max/z0 are inert for nemo_quadratic) | "
          f"rho0={RHO0} | g={G_ACC}")
    print(f"[protocol] southern band = T-rows 1..{J0 - 1} "
          f"({A.gphit[1, 25]:.1f}..{A.gphit[J0 - 1, 25]:.1f} lat); "
          f"1 m3/s2 of torque == {90 * SEC_PER_DAY / 1e6:.2f} x1e6 m3/s of R per 90 d")

    c5_geometry()
    instrument_wind()

    nemo = {d: load_nemo(d) for d in NEMO_DAYS}
    npz = {os.path.basename(p).replace(".npz", ""): np.load(p) for p in args.arms}
    paths = {os.path.basename(p).replace(".npz", ""): p for p in args.arms}
    voff = pick_v_offset(next(iter(npz.values())), nemo[0])
    for n, z in npz.items():
        if pick_v_offset(z, nemo[0]) != voff:
            raise SystemExit(f"FATAL: arm {n} needs a different v offset")
    lego = {n: {d: load_lego(paths[n], npz[n], d, voff) for d in DAYS} for n in npz}

    # ---- C2 NaN ------------------------------------------------------------
    for d in NEMO_DAYS:
        for tag, st in [("NEMO", nemo[d])] + (
                [(n, lego[n][d]) for n in lego] if d in DAYS else []):
            for k, m in (("u", umask), ("v", vmask), ("T", tmask), ("S", tmask),
                         ("eta", tmask[:, :, 0])):
                bad = int(np.sum(~np.isfinite(np.asarray(st[k], np.float64)[m])))
                if bad:
                    raise SystemExit(f"FATAL C2: {tag} d{d} {k}: {bad} non-finite wet cells")
    print("[C2 NaN] no non-finite wet u/v/T/S/eta cells on either side, all days")

    # ---- C4 reconcile against the committed group series -------------------
    print("\n[C4 RECONCILE -- Rule 1e: the COMMITTED south-group transport series]")
    SOUTH = slice(0, J0)
    gt_n = {d: D._avg(D.group_transport(nemo[d]["u"], umask, SOUTH)) for d in DAYS}
    print("    NEMO south-group transport [Sv]: "
          + " ".join(f"d{d}={gt_n[d]:+.3f}" for d in DAYS)
          + f"  -> 90-d spin-up {gt_n[90] - gt_n[0]:+.3f} "
          f"(committed {COMMITTED_NEMO_SPINUP:+.2f})")
    if abs((gt_n[90] - gt_n[0]) - COMMITTED_NEMO_SPINUP) > 0.02:
        raise SystemExit("FATAL C4: NEMO's committed +3.10 Sv spin-up is not reproduced")
    for n in lego:
        dfc = D._avg(D.group_transport(lego[n][90]["u"], umask, SOUTH)) - gt_n[90]
        cap = 100 * (D._avg(D.group_transport(lego[n][90]["u"], umask, SOUTH))
                     - D._avg(D.group_transport(lego[n][0]["u"], umask, SOUTH))) \
            / (gt_n[90] - gt_n[0])
        print(f"    {n:12s} day-90 deficit {dfc:+.3f} Sv "
              f"(committed {COMMITTED_GSOUTH_D90.get(n, float('nan')):+.3f}) | "
              f"captures {cap:.0f}% of NEMO's spin-up")
        if n in COMMITTED_GSOUTH_D90 and abs(dfc - COMMITTED_GSOUTH_D90[n]) > 2e-3:
            raise SystemExit(f"FATAL C4: {n} day-90 deficit {dfc:.4f} != committed")

    # ---- the circulation series R(j) ---------------------------------------
    R = {"NEMO": {d: row_circulation(nemo[d]["u"]) for d in NEMO_DAYS}}
    for n in lego:
        R[n] = {d: row_circulation(lego[n][d]["u"]) for d in DAYS}

    # ---- C3 day-0 identity of every TERM -----------------------------------
    # The arms start from NEMO's day-0 restart but are stored as float32, so the
    # bar is not "zero": it is the torque change NEMO's OWN state produces when
    # round-tripped through float32.  That reference is MEASURED here, not
    # assumed -- PRES is a cancelling difference of ~1e10 Pa m column integrals,
    # so the fp32 state quantum is amplified by ~5 orders in that term alone and
    # a flat relative bar would either pass vacuously or fail spuriously.
    print("\n[C3 day-0 identity -- same state must give the same diagnosed torque]")
    rows = list(ROWS)
    tn0 = terms(nemo[0])
    n32 = {k: (np.asarray(v, np.float32).astype(np.float64) if k != "eta"
               else np.asarray(v, np.float32).astype(np.float64))
           for k, v in nemo[0].items()}
    t32 = terms(n32)
    ref = {k: max(float(np.max(np.abs((t32[k] - tn0[k])[rows]))), 1e-30)
           for k in ("WIND", "DRAG", "CORI", "PRES")}
    ref_R = max(float(np.max(np.abs((row_circulation(n32["u"]) - R["NEMO"][0])[rows]))),
                1e-30)
    print("    fp32 round-trip of NEMO's OWN day-0 state moves the terms by "
          + "  ".join(f"{k} {ref[k]:.2e}" for k in ref) + f"  R {ref_R:.2e}")
    for n in lego:
        tl0 = terms(lego[n][0])
        line = f"    {n:12s} "
        bad = []
        for k in ("WIND", "DRAG", "CORI", "PRES"):
            dd = float(np.max(np.abs((tl0[k] - tn0[k])[rows])))
            line += f"{k} {dd:.2e} ({dd / ref[k]:.1f}x fp32)  "
            if dd > 3.0 * ref[k]:
                bad.append(k)
        dR = float(np.max(np.abs((R[n][0] - R["NEMO"][0])[rows])))
        line += f"R {dR:.2e} ({dR / ref_R:.1f}x fp32)"
        print(line)
        if bad or dR > 3.0 * ref_R:
            raise SystemExit(f"FATAL C3: arm {n} day-0 torques {bad} exceed 3x the "
                             "fp32 snapshot quantum -- the arms do not start from "
                             "NEMO's state, or the diagnostic is state-sensitive "
                             "beyond rounding")

    planted(nemo[90])

    # ---- P3 rest-state floor ------------------------------------------------
    fl, fl_lv = rest_identity(nemo[0])
    print("\n[P3 REST ablation -- a GROSS-ERROR check, NOT a noise floor "
          "(retracted; see rest_identity)]")
    for _tag, _v in (("Leibniz", fl), ("z-level", fl_lv)):
        print(f"    {_tag:8s} form : max|PRES| over the band = "
              f"{float(np.max(np.abs(_v[rows]))):.3e} m3/s2   "
              f"(mean {float(np.mean(_v[rows])):+.3e})")
    print("    RETRACTION: an earlier version of this probe called these numbers the "
          "PRES\n    noise floor.  They are not.  This run is FULL-STEP "
          "(namelist_cfg:71 ln_zps_nam\n    = .false.; the horizontal spread of e3t_0 "
          "within every level measures exactly\n    0.0), and with rho = rho(k) the two "
          "Leibniz pieces cancel ALGEBRAICALLY for any\n    bathymetry -- so ~1e-9 is "
          "float64 roundoff and the z-level arm's exact 0.0 is\n    the tell.  The "
          "ablation removes the horizontal density gradient PRES exists to\n    "
          "measure, so it cannot bound PRES's error.  It survives only as a gross-error\n"
          "    check (a sign flip between the two pieces moves it by ~1e7).  The real\n"
          "    uncertainty is measured in P4 below.")

    pres_unc = pres_uncertainty(nemo, lego, day=90)

    # ---- the term table -----------------------------------------------------
    TN = {d: terms(nemo[d]) for d in DAYS}
    TL = {n: {d: terms(lego[n][d]) for d in DAYS} for n in lego}
    TN10 = {d: terms(nemo[d]) for d in NEMO_DAYS}

    def resid(t, dRdt):
        return dRdt - (t["WIND"] + t["DRAG"] + t["CORI"] + t["PRES"])

    # instantaneous torque, 90-day time-mean via trapezoid on the MATCHED 0/30/60/90
    # sampling (Rule 7: both sides get the identical protocol)
    span = 90 * SEC_PER_DAY

    def mean_terms(T, days=DAYS):
        return {k: _trap([T[d][k] for d in days], days) / ((days[-1] - days[0]) * SEC_PER_DAY)
                for k in ("WIND", "DRAG", "CORI", "PRES")}

    mt_n = mean_terms(TN)
    mt_n10 = mean_terms(TN10, NEMO_DAYS)
    mt_l = {n: mean_terms(TL[n]) for n in lego}
    dR_n = (R["NEMO"][90] - R["NEMO"][0]) / span
    dR_l = {n: (R[n][90] - R[n][0]) / span for n in lego}

    # ---- instrument calibration against NEMO's OWN dumped operators --------
    nemo_trend_calibration(R["NEMO"], TN)

    print("\n" + "=" * 112)
    print("T1  90-DAY MEAN TORQUE BUDGET of the closed southern band, per u-row "
          "[m3/s2]\n    (trapezoid over days 0/30/60/90 -- the SAME sampling on both "
          "sides; NEMO's\n    10-day version is printed after, as the trapezoid-error "
          "control)")
    print("=" * 112)
    hdr = (f"{'row':>4s}{'lat':>7s}{'side':>10s}{'dR/dt':>10s}{'WIND':>10s}"
           f"{'DRAG':>10s}{'CORI':>10s}{'PRES':>11s}{'CORI+PRES':>11s}{'RESID':>11s}")
    print(hdr)
    for j in ROWS:
        for side, mt, dd in ([("NEMO", mt_n, dR_n)]
                             + [(n, mt_l[n], dR_l[n]) for n in lego]):
            rs = resid({k: mt[k] for k in mt}, dd)
            print(f"{j:4d}{A.gphit[j, 25]:7.1f}{side:>10s}{dd[j]:10.3f}"
                  f"{mt['WIND'][j]:10.3f}{mt['DRAG'][j]:10.3f}{mt['CORI'][j]:10.3f}"
                  f"{mt['PRES'][j]:11.2f}{mt['CORI'][j] + mt['PRES'][j]:11.3f}"
                  f"{rs[j]:11.2f}")
        print()

    print("T1b NEMO's own budget at its NATIVE 10-day sampling (trapezoid-error "
          "control):")
    print(f"{'row':>4s}" + "".join(f"{k:>11s}" for k in
                                   ("dR/dt", "WIND", "DRAG", "CORI", "PRES", "RESID"))
          + "   |  4-sample - 10-sample difference, per term")
    rs10 = resid(mt_n10, dR_n)
    rs4 = resid(mt_n, dR_n)
    for j in ROWS:
        d4 = [mt_n[k][j] - mt_n10[k][j] for k in ("WIND", "DRAG", "CORI", "PRES")]
        print(f"{j:4d}{dR_n[j]:11.3f}{mt_n10['WIND'][j]:11.3f}{mt_n10['DRAG'][j]:11.3f}"
              f"{mt_n10['CORI'][j]:11.3f}{mt_n10['PRES'][j]:11.2f}{rs10[j]:11.2f}"
              f"   | " + " ".join(f"{v:+8.3f}" for v in d4)
              + f"  RESID {rs4[j] - rs10[j]:+8.2f}")

    print("\n" + "=" * 112)
    print("T2  THE DIFFERENCE (lego - NEMO) per term, 90-day mean torque [m3/s2],\n"
          "    and the circulation it builds over 90 days [1e6 m3/s].  The row's\n"
          "    MEASURED deficit is d(dR/dt); the terms must account for it.")
    print("=" * 112)
    for n in lego:
        print(f"  arm {n}")
        print(f"{'row':>4s}{'lat':>7s}{'d(dR/dt)':>11s}{'dWIND':>9s}{'dDRAG':>9s}"
              f"{'dCORI':>9s}{'dPRES':>10s}{'d(CORI+PRES)':>14s}{'dRESID':>11s}"
              f"{'| dR90 meas':>13s}")
        for j in ROWS:
            dd = dR_l[n][j] - dR_n[j]
            dt_ = {k: mt_l[n][k][j] - mt_n[k][j] for k in mt_n}
            dres = dd - sum(dt_.values())
            dR90 = (R[n][90] - R[n][0] - (R["NEMO"][90] - R["NEMO"][0]))[j] / 1e6
            print(f"{j:4d}{A.gphit[j, 25]:7.1f}{dd:11.3f}{dt_['WIND']:9.3f}"
                  f"{dt_['DRAG']:9.3f}{dt_['CORI']:9.3f}{dt_['PRES']:10.2f}"
                  f"{dt_['CORI'] + dt_['PRES']:14.3f}{dres:11.2f}{dR90:13.2f}")
        print()

    # ---- R series for reference --------------------------------------------
    print("=" * 112)
    print("T3  ROW CIRCULATION R(j) [1e6 m3/s] -- the LHS the torques integrate to.\n"
          "    Gamma(closed sub-basin south of row j) = -R(j).")
    print("=" * 112)
    print(f"{'row':>4s}{'lat':>7s}" + "".join(f"{'NEMO d' + str(d):>11s}" for d in DAYS)
          + "".join(f"{n[:8] + ' d90':>13s}" for n in lego) + f"{'cap%':>8s}")
    for j in ROWS:
        line = f"{j:4d}{A.gphit[j, 25]:7.1f}" + "".join(f"{R['NEMO'][d][j] / 1e6:11.2f}"
                                                        for d in DAYS)
        for n in lego:
            line += f"{R[n][90][j] / 1e6:13.2f}"
        n0 = next(iter(lego))
        den = (R["NEMO"][90] - R["NEMO"][0])[j]
        line += f"{100 * (R[n0][90] - R[n0][0])[j] / den if abs(den) > 1e-9 else np.nan:7.0f}%"
        print(line)
    tot_n = float(np.sum((R["NEMO"][90] - R["NEMO"][0])[rows]))
    for n in lego:
        tot_l = float(np.sum((R[n][90] - R[n][0])[rows]))
        cap_committed = 100 * (D._avg(D.group_transport(lego[n][90]["u"], umask, SOUTH))
                               - D._avg(D.group_transport(lego[n][0]["u"], umask, SOUTH))
                               ) / (gt_n[90] - gt_n[0])
        print(f"    band-summed 90-d change in R: NEMO {tot_n / 1e6:+.2f}, "
              f"{n} {tot_l / 1e6:+.2f} x1e6 m3/s  -> {100 * tot_l / tot_n:.0f}% captured "
              f"(the committed depth-mean-transport metric says {cap_committed:.0f}%)")

    # ---- the PRES discretization spread ------------------------------------
    print("\n[PRES discretization spread] Leibniz vs z-level form, day-90 states "
          "[m3/s2]:")
    print(f"{'row':>4s}{'NEMO Leib':>12s}{'NEMO zlev':>12s}"
          + "".join(f"{n[:8] + ' Leib':>14s}{n[:8] + ' zlev':>14s}" for n in lego))
    pn_l = phi_pres(nemo[90]["T"], nemo[90]["S"], nemo[90]["eta"])
    pn_z = phi_pres_levelform(nemo[90]["T"], nemo[90]["S"], nemo[90]["eta"])
    pl = {n: (phi_pres(lego[n][90]["T"], lego[n][90]["S"], lego[n][90]["eta"]),
              phi_pres_levelform(lego[n][90]["T"], lego[n][90]["S"], lego[n][90]["eta"]))
          for n in lego}
    for j in ROWS:
        print(f"{j:4d}{pn_l[j]:12.2f}{pn_z[j]:12.2f}"
              + "".join(f"{pl[n][0][j]:14.2f}{pl[n][1][j]:14.2f}" for n in lego))
    for n in lego:
        dL = (pl[n][0] - pn_l)[rows]
        dZ = (pl[n][1] - pn_z)[rows]
        print(f"    arm {n}: d(PRES) Leibniz {np.mean(dL):+.3f} +- {np.std(dL):.3f}, "
              f"z-level {np.mean(dZ):+.3f} +- {np.std(dZ):.3f} m3/s2 (band mean +- std)")

    # ---- V0: the one result that needs no torque model at all ---------------
    print("\n" + "=" * 112)
    print("V0 THE SHAPE OF THE MEASURED DEFICIT -- state differences only, no torque\n"
          "   model, no discretization choice, nothing this probe could get wrong\n"
          "   beyond the row circulation itself (which is a plain weighted sum of u).")
    print("=" * 112)
    for n in lego:
        d = (dR_l[n] - dR_n)[rows]
        nn = dR_n[rows]
        print(f"  arm {n}")
        print(f"    {'row':>4s}{'lat':>7s}{'NEMO dR/dt':>12s}{'lego dR/dt':>12s}"
              f"{'deficit':>10s}{'as % of NEMO':>14s}")
        for i, j in enumerate(rows):
            pc = (100 * d[i] / nn[i]) if abs(nn[i]) > 1e-3 else np.nan
            print(f"    {j:4d}{A.gphit[j, 25]:7.1f}{nn[i]:12.3f}{dR_l[n][j]:12.3f}"
                  f"{d[i]:10.3f}{pc:13.0f}%")
        print(f"    deficit per row : mean {d.mean():+.3f}, std {d.std():.3f}, "
              f"range {d.min():+.3f}..{d.max():+.3f}  (spread {d.min() / d.max():.1f}x)")
        print(f"    NEMO's own rate : range {nn.min():+.3f}..{nn.max():+.3f}  "
              f"(spread {nn.max() / max(abs(nn.min()), 1e-9):.0f}x)")
        print(f"    wall rows 1-5   : NEMO {nn[:5].sum():+.3f}, deficit {d[:5].sum():+.3f}"
              f"   |   main rows 6-13: NEMO {nn[5:].sum():+.3f}, deficit {d[5:].sum():+.3f}")
        # proportionality tests, and the temporal scatter that bounds them
        w = np.array([mt_n["WIND"][j] for j in rows])
        print(f"    corr(deficit, NEMO rate) = {np.corrcoef(d, nn)[0, 1]:+.2f} | "
              f"corr(deficit, WIND) = {np.corrcoef(d, w)[0, 1]:+.2f}")
        per = np.array([[((R[n][b][j] - R[n][a][j]) - (R["NEMO"][b][j] - R["NEMO"][a][j]))
                         / (30 * SEC_PER_DAY) for a, b in ((0, 30), (30, 60), (60, 90))]
                        for j in rows])
        print(f"    interval-to-interval scatter of the SAME quantity (30-day windows): "
              f"std {per.std(axis=1).mean():.3f} m3/s2 per row")
        print(f"    -> the row-to-row spread ({d.std():.3f}) is at or below that scatter, "
              "so 'row-independent'\n       is CONSISTENT WITH the data, not proven by "
              "it.  What IS refuted is\n       proportionality: deficit / NEMO-rate "
              f"spans {np.nanmin(100 * d / np.where(np.abs(nn) > 1e-3, nn, np.nan)):.0f}%"
              f" .. {np.nanmax(100 * d / np.where(np.abs(nn) > 1e-3, nn, np.nan)):.0f}%.")
    print("\n    The missing torque is the SAME ORDER in every row of the band while NEMO's\n"
          "    own spin-up torque varies ~30x and the wind torque ~100x across those same\n"
          "    rows.  legoESM is therefore NOT simply under-responding in proportion to\n"
          "    the forcing -- both proportionalities are refuted.  Whether the deficit is\n"
          "    strictly row-independent is NOT established: its row-to-row spread sits at\n"
          "    the level of its own 30-day temporal scatter.")

    # ---- VERDICT -----------------------------------------------------------
    print("\n" + "=" * 112)
    print("V  VERDICT -- can this offline instrument attribute the deficit?")
    print("=" * 112)
    for n in lego:
        dd = (dR_l[n] - dR_n)[rows]
        dres = np.array([(dR_l[n] - dR_n)[j]
                         - sum(mt_l[n][k][j] - mt_n[k][j] for k in mt_n) for j in rows])
        ddrag = np.array([mt_l[n]["DRAG"][j] - mt_n["DRAG"][j] for j in rows])
        dpair = np.array([(mt_l[n]["CORI"][j] + mt_l[n]["PRES"][j])
                          - (mt_n["CORI"][j] + mt_n["PRES"][j]) for j in rows])
        print(f"  arm {n}")
        print(f"    the quantity to explain   |d(dR/dt)| per row : "
              f"{np.abs(dd).min():.3f} .. {np.abs(dd).max():.3f} m3/s2")
        print("    WIND  difference                             : EXACTLY 0 -- BY "
              "CONSTRUCTION, not by measurement.\n           phi_wind takes no state, "
              "so it returns one array for both sides.  NEMO's dumped\n           "
              "utau_b matches the analytic profile to 6e-17 Pa, but NOTHING here "
              "checks the\n           stress the ARMS actually applied -- it is not in "
              "the saved snapshots.  WAIVED:\n           the twin driver negates the "
              "DINO analytic stress once for the atmospheric\n           convention "
              "and the PE external-tau block negates it back (dino.py:3602-3609).")
        print(f"    DRAG  difference                             : "
              f"{ddrag.min():+.3f} .. {ddrag.max():+.3f} m3/s2   "
              f"(sign is {'OPPOSITE to' if np.mean(ddrag) * np.mean(dd) < 0 else 'the same as'} "
              "the deficit)\n           ORACLE-VALIDATED: this probe's WIND+DRAG "
              "reproduces NEMO's own recovered\n           vertical-diffusion trend to "
              "<=1e-4 relative on every row (T0), so the drag\n           exoneration "
              "rests on NEMO's operator, not only on using one code twice.")
        print(f"    CORI+PRES (geostrophic pair) difference       : "
              f"{dpair.min():+.3f} .. {dpair.max():+.3f} m3/s2")
        print(f"    PRES discretization floor (P4)                : "
              f"{pres_unc[n]:.2f} m3/s2 on the lego-NEMO difference")
        print(f"    THIS PROBE'S OWN CLOSURE ERROR (|RESID|)      : "
              f"{np.abs(dres).min():.2f} .. {np.abs(dres).max():.2f} m3/s2")
        rat = np.abs(dres) / np.maximum(np.abs(dd), 1e-30)
        print(f"    per-row ratio |dRESID| / |d(dR/dt)|           : "
              f"median {np.median(rat):.1f}x, max {rat.max():.0f}x  "
              "(want << 1 to attribute anything)")
    print("\n    Coriolis and pressure are geostrophically locked: each is 100-300 m3/s2\n"
          "    and they cancel to ~1.  Their lego-NEMO differences are 5-40 m3/s2 each and\n"
          "    also cancel.  The systematic part of this probe's error DOES largely cancel\n"
          "    in the difference (absolute closure error ~60 m3/s2 -> difference-of-\n"
          "    residual 0.2-12), but what survives is still up to ~20x the per-row deficit\n"
          "    and carries no sign structure.  The PRESSURE term is the binding limit: the\n"
          "    choice of discretization alone moves the lego-NEMO pressure difference by\n"
          "    ~6 m3/s2 (P4), an order above the deficit.\n"
          "\n    WHAT IS SETTLED: WIND cannot differ (by construction, waiver above), and\n"
          "    DRAG's difference is small, of the WRONG SIGN, and oracle-validated.  Those\n"
          "    two are exonerated.  Everything else sums to the deficit and this instrument\n"
          "    cannot split that sum.  NO OTHER TERM IS ATTRIBUTED.\n"
          "\n    Momentum advection and lateral viscosity are NOT included here.  That is a\n"
          "    SCOPE CHOICE, not an impossibility -- both are functions of u, v, the mesh\n"
          "    and a namelist coefficient, exactly like CORI and PRES, and NEMO's utrd_ldf\n"
          "    reaches 61 m3/s2 at row 1.  The reason not to build them is measured, not\n"
          "    assumed: T0 hands this probe NEMO's OWN ldf and advection trends and the\n"
          "    closure error only falls from ~60 to ~38 m3/s2.  The residue is in PRES, so\n"
          "    implementing the two missing terms would not make the deficit attributable.")

    if args.out_npz:
        out = {"lat": A.gphit[:J0, 25], "rows": np.array(list(ROWS)),
               "rest_floor_leibniz": fl, "rest_floor_zlevel": fl_lv}
        for d in NEMO_DAYS:
            out[f"R_NEMO_d{d}"] = R["NEMO"][d]
        for d in DAYS:
            for k in ("WIND", "DRAG", "CORI", "PRES"):
                out[f"T_NEMO_{k}_d{d}"] = TN[d][k]
            for n in lego:
                out[f"R_{n}_d{d}"] = R[n][d]
                for k in ("WIND", "DRAG", "CORI", "PRES"):
                    out[f"T_{n}_{k}_d{d}"] = TL[n][d][k]
        np.savez_compressed(args.out_npz, **out)
        print(f"\n[artifact] -> {args.out_npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
