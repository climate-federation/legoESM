#!/usr/bin/env python
"""One FULL FV3 time step, port vs the Fortran oracle, C48/npz=5.

This is the terminal gate of the 3-D duo port: everything below it
(gridstruct, c_sw, the pressure chain, d_sw1-6, the exchanges, fv_mapz)
has a per-stage certificate, and this is the only test that runs them in
the oracle's own order against the oracle's own answer.

REFERENCE
---------
``fv3_oracle_pinned/run_hydro_1step_gfs`` -- test_case = -13
(DCMIP16 J&W baroclinic wave, perturbed), C48, npz=5, hydrostatic,
adiabatic, duogrid, k_split=1, n_split=8, nord=2, vtdm4=0.12,
hord_*=6, kord_mt=9, kord_tm=-9, dt_atmos=1920 s, run for
``minutes = 32`` = EXACTLY one dt_atmos, so ``RESTART/`` is the state
after one ``fv_dynamics`` call.

``run_hydro_zerostep`` is the SAME deck at zero steps, i.e. the oracle's
own initial condition as a field.  It is used here as the instrument
control, not as decoration -- see below.

THE FACE MAP IS SEARCHED, NOT ASSUMED
-------------------------------------
The port's cube is FV3's cube RELABELLED: a permutation of the six faces
composed with a dihedral transform, a sign, and -- on four of the six --
a TRANSPOSE that sends u <-> v.  ``ic_face_map_parity.py`` established
that map at ~1e-14 on the initial condition.

This script re-derives it here from the INITIAL CONDITION on every run
and then applies the derived map to the one-step fields.  Two reasons,
both learned the hard way on this port:

1. A hard-coded map cannot fail, so it cannot certify anything.  Deriving
   it from the IC and REQUIRING a bijection means a grid change that
   silently relabels a face turns this script red instead of quietly
   comparing the wrong pair.
2. The IC agreement is the instrument control.  If the derived IC
   residual is not at the ~1e-14 quad-geometry floor, the harness is
   broken and NO number it prints about the one-step state means
   anything.  The script refuses to report the step comparison in that
   case rather than printing both and letting a reader pick.

WHY ~1e-14 IS THE FLOOR AND BIT-EXACTNESS IS NOT AVAILABLE
----------------------------------------------------------
``fv_grid_utils.F90:43-49`` sets ``f_p = selected_real_kind(20)`` (QUAD)
unless ``NO_QUAD_PRECISION`` is defined, and neither oracle build defines
it.  ``mid_pt3_cart``, ``normalize_vect``, ``inner_prod`` and
``latlon2xyz`` therefore run in quad and round to f64 on return.  A
float64 port cannot reproduce that exactly.  Everything else is f64
(``-fdefault-real-8`` + ``libfms_r8.a``), so f64 is the right precision
for the port; the quad is confined to the geometry helpers.

WHAT A "MATCH" MEANS AFTER ONE STEP, AND WHAT IT DOES NOT
---------------------------------------------------------
One step is 8 acoustic sub-steps and one remap.  Each stage amplifies the
IC's ~1e-14 seed, so the one-step residual is EXPECTED to be larger than
the IC's -- growth of one or two orders is the arithmetic, not a defect.
What would be a defect is a residual near the SIGNAL: the script prints
``|oracle_1step - oracle_IC|`` beside every number so the reader can see
how much the step actually moved the field.  A residual that is a
noticeable fraction of that tendency means the port and the oracle
disagree about the physics, not about the last bits.

CONSTANTS ARE A COMPILE-TIME PROPERTY OF THE ORACLE BINARY.  This deck is
linked against the FMS **GFS** set, whose kappa is ``FV3_RDGAS /
FV3_CP_AIR`` and is NOT the idealised 2/7.  The script asserts the
flavour from the run's OWN log rather than trusting the build
directory's name.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ORACLE_ROOT = "/burg-archive/glab/users/pg2328/fv3_oracle_pinned"
N, NG, KM = 48, 3, 5
DT_ATMOS = 1920.0
K_SPLIT, N_SPLIT = 1, 8
KORD_MT, KORD_TM, KORD_TR = 9, -9, 9
NR_TRACERS = 2          # ncnst=3, dnats=1 -> nr = 2 (run_out.txt:97)

# The IC control must land at the quad-geometry floor. 1e-12 is two
# orders of slack on the 2.2e-14 the IC parity measured, which is enough
# for a different BLAS or a different numpy without being enough to hide
# a real geometry change.
IC_CONTROL_MAX_REL = 1.0e-12


# ----------------------------------------------------------------------
# oracle side
# ----------------------------------------------------------------------

def require_gfs_constants(run_dir: str) -> None:
    """Refuse a GFDL-linked reference.

    FMS defaults to GFDL when neither flavour is defined
    (``fmsconstants.F90:67-68``), and GFDL's radius differs from GFS's by
    3.1e-5 RELATIVE -- ten orders above the parity floor.  Comparing
    against the wrong link reads as a dynamics defect.  Read the run's own
    log, never the build directory's name.
    """
    from legoesm.grids.fv3_native_gridstruct import FV3_RADIUS_M
    log = os.path.join(run_dir, "logfile.000000.out")
    txt = open(log, errors="replace").read() if os.path.exists(log) else ""
    if "FMSConstants: GFS" not in txt:
        raise SystemExit(
            f"{run_dir}: logfile does not say 'FMSConstants: GFS'. The port "
            f"pins the GFS set; a GFDL-linked oracle differs by 3.1e-5 "
            f"relative in radius alone and would look like a physics defect.")
    out = os.path.join(run_dir, "run_out.txt")
    if os.path.exists(out):
        s = open(out, errors="replace").read()
        if f"{FV3_RADIUS_M:.1f}" not in s:
            raise SystemExit(
                f"{run_dir}: run_out.txt does not report the pinned radius "
                f"{FV3_RADIUS_M:.1f}")


def load_oracle(run_dir: str, nh: bool = False) -> list:
    """Six per-tile dicts, arrays AS STORED (j, i) with the time axis gone."""
    import netCDF4 as nc
    require_gfs_constants(run_dir)
    tiles = []
    for t in range(1, 7):
        p = os.path.join(run_dir, "RESTART", f"fv_core.res.tile{t}.nc")
        if not os.path.exists(p):
            raise SystemExit(f"missing oracle restart {p}")
        d = nc.Dataset(p)
        rec = {"u": np.array(d["u"][:])[0],        # (km, n+1, n)
               "v": np.array(d["v"][:])[0],        # (km, n,   n+1)
               "pt": np.array(d["T"][:])[0],       # (km, n,   n)
               "delp": np.array(d["delp"][:])[0],  # (km, n,   n)
               "phis": np.array(d["phis"][:])[0]}  # (n, n)
        if nh:
            for k, v in (("w", "W"), ("delz", "DZ")):
                if v not in d.variables:
                    raise SystemExit(
                        f"{p}: no {v} variable -- is this an NH restart? "
                        f"The hydrostatic decks do not write it.")
                rec[k] = np.array(d[v][:])[0]      # (km, n, n)
        for k, v in rec.items():
            if not np.all(np.isfinite(v)):
                raise SystemExit(f"{p}: {k} has non-finite values")
        tiles.append(rec)
        d.close()
    return tiles


def require_flat_orography(tiles: list) -> None:
    """``mountain = .F.`` and the J&W BC wave has no orography.

    The port's context carries ``hs = 0``; if the oracle's ``phis`` were
    non-zero, ``geopk``'s ``gz(:,:,km+1) = hs`` would differ and the whole
    pressure chain with it.  Check rather than assume -- this is one
    namelist edit away from being false.
    """
    worst = max(float(np.abs(t["phis"]).max()) for t in tiles)
    if worst != 0.0:
        raise SystemExit(
            f"oracle phis is not identically zero (max |phis| = {worst:g}); "
            f"the port runs hs = 0 and would disagree through geopk's "
            f"gz(:,:,km+1) = hs seed")


# ----------------------------------------------------------------------
# port side
# ----------------------------------------------------------------------

def build_port_ic(ctx, ak, bk, nh: bool = False):
    """The test_case = -13 IC on all six faces, in state_3d layout.

    ``nh=True`` adds ``make_nh``'s initial state (``init_hydro.F90:
    147-158``): ``w = 0`` and ``delz = -(rdgas/grav) * T * dpeln`` from
    the IC's own hydrostatic column -- the exact arithmetic the oracle's
    ``delz computed from hydrostatic state`` message announces (zvir = 0
    on the adiabatic deck).
    """
    from legoesm.core.fv3_native_dcmip16_bc import GFS_CONSTANTS
    from legoesm.core.fv3_native_dcmip16_ic import dcmip16_bc_face
    from legoesm.core.fv3_native_state_3d import build_state_3d
    from legoesm.grids.fv3_native_gridstruct import FV3_GRAV, FV3_RDGAS
    from legoesm.grids.fv3_native_metrics import great_circle_dist as _gcd

    def gcdr(p1, p2, r):
        return _gcd(np.asarray(p1, float), np.asarray(p2, float)) * r

    n, ng = ctx["n"], ctx["ng"]
    st = build_state_3d(n, ng, KM, remap_follows=True,
                        hydrostatic=not nh)
    cs, cc = slice(ng, ng + n), slice(ng, ng + n + 1)
    for t in range(6):
        gs = ctx["gs6"][t]
        co = np.stack([np.asarray(gs["grid_lon"])[cc, cc],
                       np.asarray(gs["grid_lat"])[cc, cc]], -1)
        ce = np.stack([np.asarray(gs["agrid_lon"])[cs, cs],
                       np.asarray(gs["agrid_lat"])[cs, cs]], -1)
        o = dcmip16_bc_face(co, ce, ak, bk, KM, do_pert=True,
                            constants=GFS_CONSTANTS, great_circle_dist=gcdr)
        st[t]["delp"][cs, cs, :] = o["delp"]
        st[t]["pt"][cs, cs, :] = o["pt"]
        st[t]["u"][cs, cc, :] = o["u"]
        st[t]["v"][cc, cs, :] = o["v"]
        if nh:
            pe = np.full((n, n), float(ak[0]))
            for k in range(KM):
                dp = st[t]["delp"][cs, cs, k]
                dpeln = np.log(pe + dp) - np.log(pe)
                st[t]["delz"][:, :, k] = (-(FV3_RDGAS / FV3_GRAV)
                                          * st[t]["pt"][cs, cs, k] * dpeln)
                pe = pe + dp
    return st


def port_window(state, ctx) -> list:
    """Compute-window copies, port orientation (i, j, k)."""
    n, ng = ctx["n"], ctx["ng"]
    cs, cc = slice(ng, ng + n), slice(ng, ng + n + 1)
    out = []
    for f in state:
        rec = {"u": np.array(f["u"][cs, cc, :]),
               "v": np.array(f["v"][cc, cs, :]),
               "pt": np.array(f["pt"][cs, cs, :]),
               "delp": np.array(f["delp"][cs, cs, :])}
        if "delz" in f:
            rec["w"] = np.array(f["w"][cs, cs, :])
            rec["delz"] = np.array(f["delz"])
        out.append(rec)
    return out


# ----------------------------------------------------------------------
# the face map
# ----------------------------------------------------------------------

DIHEDRAL = {"id": lambda a: a,
            "fi": lambda a: a[::-1, ...],
            "fj": lambda a: a[:, ::-1, ...],
            "r180": lambda a: a[::-1, ::-1, ...]}

# THE WIND SIGNS ARE DERIVED, NOT SEARCHED. `u` is the D-grid component
# along i and `v` the one along j, so a reflection negates the component
# ALONG the reversed axis and leaves the other alone. That is a property
# of the dihedral, not a free parameter:
DIHEDRAL_SIGNS = {"id": (1.0, 1.0), "fi": (-1.0, 1.0),
                  "fj": (1.0, -1.0), "r180": (-1.0, -1.0)}
# Searching the two signs independently -- which this script did until the
# residuals below forced the question -- lets the search pick a signed
# permutation that is not a rigid relabelling at all (an identity map that
# flips only u), and it WILL pick one whenever a component is numerically
# zero on that face and its sign is therefore unconstrained. Measured: on
# the J&W IC, port faces 2, 4 and 5 each have one wind component at ~1e-13,
# the free search chose arbitrarily on all three, and those are EXACTLY the
# three faces whose one-step residual came out at 2-4e-3 while the other
# three sat at 1e-6..1e-8. Face 2's v residual was 0.0851 against an oracle
# tendency of 0.0385 -- twice the signal, the signature of a flipped sign.
# The derived rule reproduces the search's answer on every face where the
# signs ARE constrained (face 1 transpose fj -> (+,-), face 3 direct r180
# -> (-,-), face 6 direct id -> (+,+)), which is the evidence that the rule
# is right rather than merely tidier.


def oracle_ij(arr, transposed: bool):
    """Oracle array (k, j, i) -> port orientation (i, j, k).

    NOT transposed: the port's i is the oracle's i, so swap the stored
    (j, i) into (i, j).  Transposed: the port's i IS the oracle's j, so
    the stored order is already the port's.
    """
    a = np.moveaxis(arr, 0, -1)              # (j, i, k)
    return a.transpose(1, 0, 2) if not transposed else a


def rel(a, b, scale: float | None = None) -> float:
    """Max abs difference over a scale that is a PHYSICAL magnitude.

    ``scale=None`` falls back to the larger of the two peaks, which is
    right for a field that carries signal.  It is WRONG for a component
    that is identically zero on this face, and that is not hypothetical:
    on the J&W IC, port face 1 has ``v`` bounded by 8.3e-14 while oracle
    tile 4 has ``u`` bounded by 5.3e-13 -- both are numerical zero, but
    ``|a-b| / max(peaks)`` is then 6e-13/5.3e-13 ~ 1.0 and the correct
    face pairing scores as a total mismatch.  Measured: that single
    artifact made the whole IC control fail at 9.7e-01 while the two
    faces whose winds happen to carry signal in both components matched
    at 2.3e-14.

    So the caller passes the scale of the QUANTITY CLASS -- for the D
    winds, the face's largest wind component over both components and
    both sides.  A zero component is then judged against the wind speed
    that is actually present, which is what "these agree" has to mean.
    """
    # A NaN anywhere must be INFINITE, not NaN: `max(0.0, nan)` is 0.0 in
    # Python, so a single NaN in an otherwise exact step would sail through
    # a `worst > threshold` gate. Return inf so it can only ever fail.
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return float("inf")
    if scale is None:
        scale = max(float(np.abs(a).max()), float(np.abs(b).max()))
    if scale == 0.0:
        return 0.0
    return float(np.abs(a - b).max() / scale)


def wind_scale(port, orc) -> float:
    """Largest wind magnitude on this face/tile pair, both components.

    One number for u and v together: they are two components of the same
    vector field, so normalising them separately is what created the
    zero-component artifact above.
    """
    return max(float(np.abs(port["u"]).max()), float(np.abs(port["v"]).max()),
               float(np.abs(orc["u"]).max()), float(np.abs(orc["v"]).max()))


def score_pair(port, orc) -> tuple:
    """Best (rel, transposed, dihedral, s_u, s_v) mapping one port face
    onto one oracle tile, scored on ALL FOUR prognostic fields jointly.

    TWO INDEPENDENT SIGNS, NOT ONE.  ``u`` is the D-grid wind along x and
    ``v`` the one along y, so a reflection negates the component ALONG
    the reversed axis and leaves the other alone: ``fi`` sends
    (u, v) -> (-u, +v), ``fj`` sends it to (+u, -v), and ``r180`` negates
    both.  A single shared sign therefore cannot express four of the six
    face maps, and forcing one is not a conservative simplification -- it
    makes the correct map score ~1.0 and hands the bijection to a wrong
    pairing.  Measured: with one shared sign this search put port face 1
    on tile 1 at rel 9.7e-01; with independent signs it recovers the
    established 1e-14 map.

    Scalars carry no sign: ``pt`` and ``delp`` must match the SAME
    dihedral with a factor of exactly +1.  That is the constraint that
    makes this a real test.  ``ic_face_map_parity.py`` took the MINIMUM
    over the four component pairings, i.e. it accepted a tile if ANY ONE
    of u/v matched -- which cannot detect a transform that is right for
    one component and wrong for the other.  Requiring u, v, pt and delp
    to agree under ONE transform is strictly stronger.
    """
    best = (np.inf, None, None, None, None, None)
    ws = wind_scale(port, orc)
    for transposed in (False, True):
        ou = oracle_ij(orc["u"], transposed)
        ov = oracle_ij(orc["v"], transposed)
        opt = oracle_ij(orc["pt"], transposed)
        odp = oracle_ij(orc["delp"], transposed)
        # A transposed face sends u <-> v; an untransposed one does not.
        pu_o, pv_o = (ov, ou) if transposed else (ou, ov)
        if port["u"].shape != pu_o.shape or port["v"].shape != pv_o.shape:
            continue
        for nm, f in DIHEDRAL.items():
            fu, fv = f(port["u"]), f(port["v"])
            # The scalar constraint does not depend on the signs, so
            # evaluate it once and let it floor the score.
            r_pt = rel(f(port["pt"]), opt)
            r_dp = rel(f(port["delp"]), odp)
            su, sv = DIHEDRAL_SIGNS[nm]
            r_u = rel(su * fu, pu_o, ws)
            r_v = rel(sv * fv, pv_o, ws)
            r = max(r_u, r_v, r_pt, r_dp)
            if r < best[0]:
                best = (r, transposed, nm, su, sv,
                        {"u": r_u, "v": r_v, "pt": r_pt, "delp": r_dp})
    return best


def score_pair_winds_only(port, orc) -> tuple:
    """The WEAKER criterion ``ic_face_map_parity.py`` used, kept as a
    diagnostic ONLY.

    It requires the winds to agree and ignores pt and delp.  It is not a
    fallback: if the joint search fails while this one succeeds, the two
    numbers together name the field that disagrees, which is the thing a
    reader needs.  Never gate on this.
    """
    best = (np.inf, None, None, None, None)
    ws = wind_scale(port, orc)
    for transposed in (False, True):
        ou = oracle_ij(orc["u"], transposed)
        ov = oracle_ij(orc["v"], transposed)
        pu_o, pv_o = (ov, ou) if transposed else (ou, ov)
        if port["u"].shape != pu_o.shape or port["v"].shape != pv_o.shape:
            continue
        for nm, f in DIHEDRAL.items():
            fu, fv = f(port["u"]), f(port["v"])
            su, sv = DIHEDRAL_SIGNS[nm]
            r = max(rel(su * fu, pu_o, ws), rel(sv * fv, pv_o, ws))
            if r < best[0]:
                best = (r, transposed, nm, su, sv)
    return best


def derive_face_map(port, orc) -> tuple:
    """Full 6x6 cost matrix, then the best BIJECTION over it.

    Greedy per-face argmin is what produced a non-injective map last
    time.  The base state is zonally symmetric, so several tiles carry
    identical fields and a greedy pick can hand two faces the same tile.
    6! = 720 permutations, so the exact assignment is free.
    """
    import itertools
    cost = np.full((6, 6), np.inf)
    meta = [[None] * 6 for _ in range(6)]
    per_field = [[None] * 6 for _ in range(6)]
    wind_only = np.full((6, 6), np.inf)
    for pf in range(6):
        for ot in range(6):
            r, tr, nm, su, sv, pf_r = score_pair(port[pf], orc[ot])
            cost[pf, ot] = r
            meta[pf][ot] = (tr, nm, su, sv)
            per_field[pf][ot] = pf_r
            wind_only[pf, ot] = score_pair_winds_only(port[pf], orc[ot])[0]
    best_perm, best_worst = None, np.inf
    for perm in itertools.permutations(range(6)):
        w = max(cost[pf, perm[pf]] for pf in range(6))
        if w < best_worst:
            best_worst, best_perm = w, perm
    return cost, meta, best_perm, best_worst, per_field, wind_only


def apply_map(port_face, orc_tile, meta_entry) -> dict:
    """Port face and oracle tile, both in the port's orientation.

    Scalars beyond pt/delp (NH: ``w``, ``delz``) ride the same dihedral
    with factor +1 -- a transposed face swaps axes but a cell-centred
    scalar carries no component to exchange.
    """
    transposed, nm, su, sv = meta_entry
    f = DIHEDRAL[nm]
    ou = oracle_ij(orc_tile["u"], transposed)
    ov = oracle_ij(orc_tile["v"], transposed)
    pu, pv = f(port_face["u"]) * su, f(port_face["v"]) * sv
    if transposed:
        pairs = {"u": (pu, ov), "v": (pv, ou)}
    else:
        pairs = {"u": (pu, ou), "v": (pv, ov)}
    pairs["pt"] = (f(port_face["pt"]), oracle_ij(orc_tile["pt"], transposed))
    pairs["delp"] = (f(port_face["delp"]),
                     oracle_ij(orc_tile["delp"], transposed))
    for extra in ("w", "delz"):
        if extra in port_face and extra in orc_tile:
            pairs[extra] = (f(port_face[extra]),
                            oracle_ij(orc_tile[extra], transposed))
    return pairs, wind_scale(port_face, orc_tile)


# ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ic-run", default=f"{ORACLE_ROOT}/run_hydro_zerostep")
    ap.add_argument("--step-run", default=f"{ORACLE_ROOT}/run_hydro_1step_gfs")
    ap.add_argument("--n-split", type=int, default=N_SPLIT)
    ap.add_argument("--k-split", type=int, default=K_SPLIT)
    ap.add_argument("--dt", type=float, default=DT_ATMOS)
    ap.add_argument("--json", default=None)
    ap.add_argument("--trace-substeps", action="store_true",
                    help="DIAGNOSTIC: run the acoustic loop one sub-step at "
                         "a time and print max|field - IC| after each, then "
                         "stop. Localises a wrong tendency in time.")
    ap.add_argument("--max-rel", type=float, default=None,
                    help="gate: exit 1 if any field's one-step rel exceeds "
                         "this")
    ap.add_argument("--n-steps", type=int, default=1,
                    help="outer fv_dynamics calls to integrate before "
                         "comparing (the step reference must be an oracle "
                         "run of n_steps*dt_atmos -- pass --step-run "
                         "accordingly). A LINEAR-vs-EXPONENTIAL residual "
                         "growth read across an N sweep (1, 3, 10) is the "
                         "point: a compounding coefficient bug grows "
                         "linearly-or-worse in N while chaos amplifies the "
                         "1e-14 geometry seed exponentially but from far "
                         "below the certified 1-step floor (GLM review "
                         "2026-08-11: judge growth against the N=1 floor, "
                         "never against state tendency).")
    ap.add_argument("--nh", action="store_true",
                    help="non-hydrostatic gate: defaults the runs to "
                         "run_nh_{zerostep,1step}_gfs, adds delz/w to the "
                         "state (make_nh IC), drives fv_dynamics with "
                         "hydrostatic=False (a_imp=1, p_fac=0.05, "
                         "kord_wz=9, w_limiter per the deck), and scores "
                         "W and DZ on their OWN scales (W is 0 at t=0 and "
                         "~2e-4 m/s after one step -- judging it against "
                         "the 20 m/s winds would be the delp-agreement "
                         "trap again)")
    args = ap.parse_args(argv)
    if args.nh:
        if args.ic_run == f"{ORACLE_ROOT}/run_hydro_zerostep":
            args.ic_run = f"{ORACLE_ROOT}/run_nh_zerostep_gfs"
        if args.step_run == f"{ORACLE_ROOT}/run_hydro_1step_gfs":
            args.step_run = f"{ORACLE_ROOT}/run_nh_1step_gfs"

    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.core.fv3_native_dynamics import (
        fv_dynamics_step, p_var_hydrostatic,
    )
    from legoesm.core.fv3_native_mapz import lagrangian_to_eulerian
    from legoesm.core.fv3_native_eta import set_eta_analytic
    from legoesm.core.fv3_native_state_3d import field_shape
    from legoesm.grids.fv3_native_gridstruct import FV3_CP_AIR, FV3_KAPPA

    print(f"port constants: kappa = {FV3_KAPPA!r}  cp_air = {FV3_CP_AIR!r}")
    print(f"                (2/7 = {2/7!r}; rel diff "
          f"{abs(FV3_KAPPA - 2/7)/(2/7):.3e})")

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    n, ng = ctx["n"], ctx["ng"]
    ak, bk, ptop, ks = set_eta_analytic(KM)
    ptop = float(ptop)
    print(f"grid n={n} ng={ng} km={KM} ptop={ptop} ks={ks}")

    orc_ic = load_oracle(args.ic_run, nh=args.nh)
    orc_1 = load_oracle(args.step_run, nh=args.nh)
    require_flat_orography(orc_ic)
    require_flat_orography(orc_1)

    fields = ("u", "v", "pt", "delp") + (("w", "delz") if args.nh else ())

    # How much did ONE step actually move the oracle? Without this the
    # residuals below have no scale and "small" means nothing.
    print("\noracle tendency over one step (max |1step - IC|, per tile):")
    tend = {}
    for f in fields:
        per = [float(np.abs(orc_1[t][f] - orc_ic[t][f]).max())
               for t in range(6)]
        tend[f] = per
        print(f"  {f:5s} " + "  ".join(f"{x:10.4g}" for x in per))
    if max(max(v) for v in tend.values()) == 0.0:
        raise SystemExit(
            "the oracle's one-step state is IDENTICAL to its IC -- the "
            "reference run did not integrate, so any 'match' below is "
            "vacuous. Check that minutes=32 == dt_atmos in the deck.")

    # ---------------- instrument control: the IC ----------------
    state = build_port_ic(ctx, ak, bk, nh=args.nh)
    p_ic = port_window(state, ctx)
    (cost, meta, perm, worst,
     per_field, wind_only) = derive_face_map(p_ic, orc_ic)

    print("\nIC cost matrix rel(port face -> oracle tile):")
    for pf in range(6):
        print(f"  face {pf+1}: " +
              "  ".join(f"{cost[pf, ot]:9.2e}" for ot in range(6)))
    print(f"\nbest bijection (worst-face rel = {worst:.3e}):")
    for pf in range(6):
        tr, nm, su, sv = meta[pf][perm[pf]]
        print(f"  port face {pf+1} -> oracle tile {perm[pf]+1}  "
              f"rel={cost[pf, perm[pf]]:.4e}  "
              f"[{'transpose' if tr else 'direct':9s} {nm:4s} "
              f"s_u{su:+.0f} s_v{sv:+.0f}]")

    if worst > IC_CONTROL_MAX_REL:
        # Localise the disagreement before giving up: print the WIND-ONLY
        # cost matrix (the weaker criterion the earlier map was derived
        # under) and the per-field breakdown of the joint best. If the
        # wind-only matrix has a 1e-14 entry where the joint one does not,
        # the blocking field is named right here instead of guessed at.
        print("\nWIND-ONLY cost matrix (diagnostic; NEVER a gate):")
        for pf in range(6):
            print(f"  face {pf+1}: " +
                  "  ".join(f"{wind_only[pf, ot]:9.2e}" for ot in range(6)))
        print("\nper-field rel at each pair's joint-best transform:")
        for pf in range(6):
            for ot in range(6):
                d = per_field[pf][ot]
                if min(cost[pf, ot], wind_only[pf, ot]) > 1e-6:
                    continue
                print(f"  face {pf+1} -> tile {ot+1}: " +
                      "  ".join(f"{k}={d[k]:9.2e}" for k in
                               ("u", "v", "pt", "delp")))
        print("\nport IC field ranges (compute window):")
        for f in ("u", "v", "pt", "delp"):
            print(f"  {f:5s} " + "  ".join(
                f"[{p_ic[t][f].min():.6g},{p_ic[t][f].max():.6g}]"
                for t in range(6)))
        print("oracle IC field ranges:")
        for f in ("u", "v", "pt", "delp"):
            print(f"  {f:5s} " + "  ".join(
                f"[{orc_ic[t][f].min():.6g},{orc_ic[t][f].max():.6g}]"
                for t in range(6)))
        raise SystemExit(
            f"\nINSTRUMENT CONTROL FAILED: the IC face map's worst residual "
            f"is {worst:.3e}, above the {IC_CONTROL_MAX_REL:.0e} floor. The "
            f"port's initial condition no longer reproduces the oracle's, so "
            f"nothing this script could say about the one-step state would "
            f"mean anything. Refusing to print it.")
    # FROZEN-MAP ASSERT (GLM-5.2 strategy review, 2026-08-10): the
    # bijection requirement alone cannot catch a SELF-CONSISTENT
    # relabelling drift -- a grid change that permutes two faces and an
    # IC builder that permutes them back would still derive a valid map.
    # The established map (ic_face_map_parity.py, 2026-08-07) is
    # therefore frozen here and every run's derived map must match it.
    # Faces 4/5 <-> tiles 1/2 stay a SET: those two oracle tiles are
    # unperturbed and zonally symmetric, so they carry identical fields
    # and either assignment is intrinsically valid (documented in
    # STATE.md; asserting one of them would be inventing information).
    _FROZEN_MAP = {0: {3}, 1: {4}, 2: {2}, 3: {0, 1}, 4: {0, 1}, 5: {5}}
    drift = [(pf + 1, perm[pf] + 1) for pf in range(6)
             if perm[pf] not in _FROZEN_MAP[pf]]
    if drift:
        raise SystemExit(
            f"FACE-MAP DRIFT: derived assignment(s) {drift} (port face, "
            f"oracle tile) differ from the frozen 2026-08-07 map. Either "
            f"the grid/IC labelling changed -- find out which -- or the "
            f"frozen map is stale; do not proceed on a silently different "
            f"relabelling.")

    print(f"\nINSTRUMENT CONTROL PASSED (worst IC rel {worst:.3e} <= "
          f"{IC_CONTROL_MAX_REL:.0e}; map matches the frozen 2026-08-07 "
          f"bijection). The map below is the one applied to the step.")

    if args.nh:
        # SECOND instrument control, NH fields under the SAME map: the
        # port's make_nh delz against the oracle's DZ (a real field,
        # ~1e3 m), and W == 0 EXACTLY on both sides at t=0.
        worst_dz = 0.0
        for pf in range(6):
            ot = perm[pf]
            pairs, _ = apply_map(p_ic[pf], orc_ic[ot], meta[pf][ot])
            worst_dz = max(worst_dz, rel(*pairs["delz"]))
            w_p, w_o = pairs["w"]
            if float(np.abs(w_o).max()) != 0.0:
                raise SystemExit(
                    f"oracle IC W is not identically zero on tile {ot+1} "
                    f"(max {np.abs(w_o).max():g}) -- not a t=0 restart?")
            if float(np.abs(w_p).max()) != 0.0:
                raise SystemExit(f"port IC w is not zero on face {pf+1}")
        print(f"NH IC control: worst delz rel {worst_dz:.3e}; W == 0 "
              f"exactly on both sides.")
        if worst_dz > IC_CONTROL_MAX_REL:
            raise SystemExit(
                f"NH INSTRUMENT CONTROL FAILED: delz IC rel {worst_dz:.3e} "
                f"exceeds {IC_CONTROL_MAX_REL:.0e}; the make_nh replication "
                f"does not reproduce the oracle's delz, so the step "
                f"comparison would start from a different state.")

    # ---------------- the step ----------------
    # p_var is the ONLY producer of the pkz that fv_dynamics.F90:402
    # divides by, so it must exist before either lane below.  The LANE
    # MATTERS: init_hydro.F90's NH branch (:178-184) computes pkz from
    # the ideal gas law on delp/pt/delz, NOT the hydrostatic kappa-mean.
    # Feeding the hydro pkz into the NH theta conversion put a uniform
    # 0.408 m delz error on every column in the first NH parity run.
    if args.nh:
        from legoesm.core.fv3_native_dynamics import p_var_nonhydrostatic
        press = [p_var_nonhydrostatic(f["delp"], f["delz"], f["pt"],
                                      ptop=ptop, akap=FV3_KAPPA,
                                      n=n, ng=ng, km=KM) for f in state]
    else:
        press = [p_var_hydrostatic(f["delp"], ptop=ptop, akap=FV3_KAPPA,
                                   n=n, ng=ng, km=KM) for f in state]
    q = [[np.zeros(field_shape("delp", n, ng, KM), dtype=np.float64)
          for _ in range(NR_TRACERS)] for _ in range(6)]

    if args.trace_substeps and args.nh:
        raise SystemExit(
            "--trace-substeps is wired for the hydrostatic lane only "
            "(its remap block reads the hydro pressure bundle); an NH "
            "trace needs the NH carry threaded through -- extend it "
            "rather than letting it KeyError mid-run.")
    if args.trace_substeps:
        # LOCALISE a wrong tendency in TIME before hunting it in space.
        # dyn_core.F90:337 is `do it=1,n_split`; running the loop one
        # sub-step at a time and printing max|field - IC| after each says
        # whether a spurious tendency appears at sub-step 1 (a stage is
        # wrong) or accumulates (a coefficient is wrong), and whether the
        # remap adds to it. Without this the only number available is the
        # sum of eight sub-steps and one remap.
        #
        # THE CONVERSION IS NOT OPTIONAL HERE. An earlier version of this
        # block drove the acoustic loop with pt still in KELVIN, i.e. the
        # theta_v the oracle passes to dyn_core times pkz ~ 30. geopk's
        # gz = gz + cp*pt*(pk(k+1)-pk(k)) then carried a 30x geopotential,
        # the pressure gradient blew up, and the trace reported 88 m/s at
        # sub-step 1 and a negative delpc by sub-step 3 -- an artefact of
        # the instrument, not a property of the port. Mirror
        # fv_dynamics_step exactly.
        from legoesm.core.fv3_native_acoustic_3d import acoustic_substep_3d
        from legoesm.core.fv3_native_dynamics import pt_to_theta_v
        for t in range(6):
            pt_to_theta_v(state[t]["pt"], press[t]["pkz"], n=n, ng=ng)
        base = port_window(state, ctx)
        dt_sub = args.dt / float(args.k_split) / float(args.n_split)
        print(f"\nSUB-STEP TRACE (dt_sub = {dt_sub} s), "
              f"max |field - post-conversion IC| over faces. pt is theta_v "
              f"throughout, so its numbers are theta_v, not K.")
        press_out: list = []
        for it in range(1, args.n_split + 1):
            acoustic_substep_3d(ctx, state, dt_sub, KM,
                                first_substep=(it == 1), ptop=ptop,
                                akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
                                remap_step=(it == args.n_split),
                                remap_follows=True,
                                press_out=(press_out
                                           if it == args.n_split else None))
            cur = port_window(state, ctx)
            line = "  ".join(
                f"{f}={max(float(np.abs(cur[t][f] - base[t][f]).max()) for t in range(6)):9.4g}"
                for f in ("u", "v", "pt", "delp"))
            print(f"  after it={it}/{args.n_split}: {line}", flush=True)
            if it == 1:
                # WHERE, not just how big. A localised max at a panel edge
                # or corner is a halo/corner defect; a spread-out one is a
                # discrete-balance error in the interior. Those need
                # different fixes, and "3.34 m/s" alone cannot tell them
                # apart. Report the argmax and how concentrated the excess
                # is (cells above 10% of the peak, split by distance to the
                # panel boundary).
                print("  LOCATION of the sub-step-1 increment:")
                for f in ("u", "v", "delp"):
                    for t in range(6):
                        d = np.abs(cur[t][f] - base[t][f])
                        pk_ = float(d.max())
                        if pk_ == 0.0:
                            continue
                        idx = np.unravel_index(int(np.argmax(d)), d.shape)
                        big = d > 0.1 * pk_
                        ni, nj = d.shape[0], d.shape[1]
                        ii, jj = np.nonzero(big.any(axis=2))
                        edge = int(np.count_nonzero(
                            (ii < 3) | (ii >= ni - 3) |
                            (jj < 3) | (jj >= nj - 3)))
                        print(f"    {f:5s} face {t+1}: peak={pk_:10.4g} at "
                              f"(i,j,k)={idx}  cells>10%: {len(ii)} of "
                              f"{ni*nj} columns, {edge} within 3 of a panel "
                              f"boundary")

        # And now the remap ALONE, so the acoustic and remap contributions
        # are separable rather than summed into one number.
        pre = port_window(state, ctx)
        for t in range(6):
            g = press_out[t]
            press[t]["pe"][:] = g["pe"]
            press[t]["peln"][:] = g["peln"]
            press[t]["pkz"][:] = g["pkz"]
            press[t]["pk"][ng:ng + n, ng:ng + n, :] = \
                g["pk_remap"][ng:ng + n, ng:ng + n, :]
            lagrangian_to_eulerian(
                pe=press[t]["pe"], peln=press[t]["peln"], pk=press[t]["pk"],
                pkz=press[t]["pkz"], delp=state[t]["delp"], pt=state[t]["pt"],
                u=state[t]["u"], v=state[t]["v"], ps=press[t]["ps"],
                ak=ak, bk=bk, ptop=ptop, akap=FV3_KAPPA, cp=FV3_CP_AIR,
                r_vir=0.0, km=KM, n=n, ng=ng, kord_mt=KORD_MT,
                kord_tm=KORD_TM, kord_tr=KORD_TR, q=q[t],
                omga=np.zeros(field_shape("delp", n, ng, KM),
                              dtype=np.float64),
                last_step=True, hydrostatic=True, adiabatic=True, consv=0.0,
                fill=False, do_sat_adj=False, do_inline_mp=False,
                do_adiabatic_init=False)
        post = port_window(state, ctx)
        line = "  ".join(
            f"{f}={max(float(np.abs(post[t][f] - pre[t][f]).max()) for t in range(6)):9.4g}"
            for f in ("u", "v", "delp"))
        print(f"  REMAP alone changed: {line}")
        print("  (pt is excluded: the remap converts it theta_v -> K, so a "
              "raw difference there is the conversion, not a tendency)")
        return 0

    print(f"\nintegrating {args.n_steps} step(s): bdt={args.dt} "
          f"k_split={args.k_split} n_split={args.n_split} nh={args.nh} ...",
          flush=True)
    if args.nh:
        # phis == 0 (asserted above); the NH carry derives zs from it.
        m_a = n + 2 * ng
        ctx["hs6"] = [np.zeros((m_a, m_a), dtype=np.float64)
                      for _ in range(6)]
    # Each outer call owns one bdt exactly as the Fortran main loop calls
    # fv_dynamics once per dt_atmos: state and press carry between calls
    # (pt round-trips K -> theta_v -> K inside each call).
    for _step in range(args.n_steps):
        out = fv_dynamics_step(ctx, state, press, bdt=args.dt, km=KM,
                               k_split=args.k_split, n_split=args.n_split,
                               ptop=ptop, ak=ak, bk=bk, akap=FV3_KAPPA,
                               cp_air=FV3_CP_AIR, kord_mt=KORD_MT,
                               kord_tm=KORD_TM, kord_tr=KORD_TR, q=q,
                               hydrostatic=not args.nh,
                               # deck: a_imp=1., p_fac=0.05, kord_wz=9,
                               # use_logp=F, w_limiter=T (resolved namelist)
                               w_limiter=args.nh)
        if out["pt_units"] != "K":
            raise SystemExit(f"driver left pt in {out['pt_units']}, not K")
    p_1 = port_window(state, ctx)

    # THE DISCRIMINATOR. A large residual vs oracle_1step has two very
    # different causes and one number separates them: if the PORT's own
    # tendency is comparable to the oracle's, the step is roughly right
    # and the disagreement is in detail; if the port's tendency is orders
    # larger, the port's step itself is wrong and no amount of staring at
    # the face map will help. Print it before the residuals so it is read
    # first.
    print("\nPORT's own one-step tendency (max |port_1step - port_IC|), "
          "against the ORACLE's on the mapped tile:")
    for f in fields:
        row_p, row_o = [], []
        for pf in range(6):
            row_p.append(float(np.abs(p_1[pf][f] - p_ic[pf][f]).max()))
            row_o.append(tend[f][perm[pf]])
        print(f"  {f:5s} port   " + "  ".join(f"{x:10.4g}" for x in row_p))
        print(f"  {'':5s} oracle " + "  ".join(f"{x:10.4g}" for x in row_o))

    print("\none-step residual, port vs oracle (rel = max|d| / max peak), "
          "with the oracle's own one-step tendency for scale:")
    res = {}
    worst_step = 0.0
    for pf in range(6):
        ot = perm[pf]
        pairs, ws = apply_map(p_1[pf], orc_1[ot], meta[pf][ot])
        row = {}
        for f, (a, b) in pairs.items():
            r = rel(a, b, ws if f in ("u", "v") else None)
            absd = float(np.abs(a - b).max())
            row[f] = {"rel": r, "max_abs_diff": absd,
                      "oracle_tendency": tend[f][ot],
                      "frac_of_tendency": (absd / tend[f][ot]
                                           if tend[f][ot] else float("nan"))}
            worst_step = max(worst_step, r)
        # WHERE the residual lives, on the same terms as the sub-step
        # trace: a panel-boundary-concentrated remainder is a halo/edge
        # defect, a spread one is a discrete-balance difference. Reporting
        # only the peak cannot tell them apart.
        for f in fields:
            a_, b_ = pairs[f]
            d = np.abs(a_ - b_)
            pk_ = float(d.max())
            if pk_ == 0.0:
                continue
            big = d > 0.1 * pk_
            ni, nj = d.shape[0], d.shape[1]
            ii, jj = np.nonzero(big.any(axis=2))
            nedge = int(np.count_nonzero((ii < 3) | (ii >= ni - 3) |
                                         (jj < 3) | (jj >= nj - 3)))
            row[f]["cells_over_10pct"] = int(len(ii))
            row[f]["of_which_within_3_of_boundary"] = nedge
            row[f]["argmax_ijk"] = [int(x) for x in
                                    np.unravel_index(int(np.argmax(d)),
                                                     d.shape)]
        res[f"face{pf+1}->tile{ot+1}"] = row
        print(f"  face {pf+1} -> tile {ot+1}:")
        for f in fields:
            d = row[f]
            print(f"      {f:5s} rel={d['rel']:9.3e}  "
                  f"|d|max={d['max_abs_diff']:11.5g}  "
                  f"tendency={d['oracle_tendency']:11.5g}  "
                  f"|d|/tend={d['frac_of_tendency']:9.3e}  "
                  f"at={d.get('argmax_ijk')}  "
                  f"cells>10%={d.get('cells_over_10pct')} "
                  f"({d.get('of_which_within_3_of_boundary')} at a "
                  f"boundary)")

    print(f"\nWORST one-step rel over all faces and fields: {worst_step:.4e}")
    print(f"IC control (same harness, same map): {worst:.4e}")
    print(f"amplification over one step: "
          f"{worst_step / max(worst, 1e-300):.3g}x")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"ic_worst_rel": worst, "step_worst_rel": worst_step,
                       "face_map": [{"port_face": pf + 1,
                                     "oracle_tile": perm[pf] + 1,
                                     "transposed": bool(meta[pf][perm[pf]][0]),
                                     "dihedral": meta[pf][perm[pf]][1],
                                     "sign_u": meta[pf][perm[pf]][2],
                                     "sign_v": meta[pf][perm[pf]][3]}
                                    for pf in range(6)],
                       "residuals": res}, fh, indent=2)
        print(f"wrote {args.json}")

    if args.max_rel is not None:
        if not np.isfinite(args.max_rel):
            raise SystemExit("--max-rel must be finite; a NaN threshold makes "
                             "`r > thr` always False and the gate vacuous")
        if worst_step > args.max_rel:
            print(f"GATE FAILED: {worst_step:.4e} > {args.max_rel:.4e}")
            return 1
        print(f"GATE PASSED: {worst_step:.4e} <= {args.max_rel:.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
