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
import re
import sys

import numpy as np

ORACLE_ROOT = "/burg-archive/glab/users/pg2328/fv3_oracle_pinned"
N, NG, KM = 48, 3, 5
DT_ATMOS = 1920.0
K_SPLIT, N_SPLIT = 1, 8
KORD_MT, KORD_TM, KORD_TR = 9, -9, 9
NR_TRACERS = 2          # ncnst=3, dnats=1 -> nr = 2 (run_out.txt:97)
# fv_tracer.res variable order == field_table order; the first nr are
# advected+remapped, the dnats tail (rainwat) is INERT -- fv_dynamics.F90
# :191 `nq = nq_tot - flagstruct%dnats`.
ADVECTED_TRACERS = ("sphum", "liq_wat")
# The moist RESPONSE (moist deck minus dry deck) is a first-order
# quantity, not a residual, so port and oracle should agree on it as a
# FIELD. The bound is loose because each side carries its own parity
# noise (1e-9 hydro, 6.6e-4 NH) riding on a difference that is itself
# ~1e-2 of the state, and TOL-PENDING until measured on both arms; it
# is still four orders tighter than "did anything move", and a dropped,
# halved or mis-signed coupling cannot pass it.
MOIST_RESPONSE_MAX_REL = 0.2
# A response only carries information where it clears the parity floor.
# At one step that is pt alone: u/v/delp move at rounding scale on both
# sides, and comparing two noise fields returns ~2.0 by construction.
MOIST_RESPONSE_SNR = 10.0
INERT_TRACERS = ("rainwat",)

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


def load_tracer_oracle(run_dir: str) -> list:
    """Six per-tile dicts of fv_tracer.res fields, (km, j, i) as stored."""
    import netCDF4 as nc
    names = ADVECTED_TRACERS + INERT_TRACERS
    tiles = []
    for t in range(1, 7):
        p = os.path.join(run_dir, "RESTART", f"fv_tracer.res.tile{t}.nc")
        if not os.path.exists(p):
            raise SystemExit(f"missing oracle tracer restart {p}")
        d = nc.Dataset(p)
        rec = {}
        for nm in names:
            if nm not in d.variables:
                raise SystemExit(
                    f"{p}: no {nm!r} variable; found "
                    f"{sorted(d.variables)} -- the field table changed?")
            a = np.array(d[nm][:])[0]              # (km, n, n)
            if not np.all(np.isfinite(a)):
                raise SystemExit(f"{p}: {nm} has non-finite values")
            rec[nm] = a
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

def build_port_ic(ctx, ak, bk, nh: bool = False, zvir: float = 0.0):
    """The test_case = -13 IC on all six faces, in state_3d layout.

    DELEGATES to the committed home,
    ``fv3_native_dcmip16_ic.dcmip16_bc_six_face_state`` -- this function
    is where that assembly was originally established, and the two were
    line-for-line copies. They must not stay copies now that the moist
    arm couples ``pt`` to ``q``: ``test_cases.F90:6760`` divides pt by
    ``(1 + zvir*q(sphum))``, so pt and the tracer IC have to come out of
    ONE construction or a second implementation decides half of it.

    ``nh=True`` adds ``make_nh``'s initial state (``init_hydro.F90:
    147-158``); ``zvir != 0`` selects the moist (non-adiabatic) IC.
    Returns ``(state, sphum)`` -- sphum is the SAME array the pt
    division used.
    """
    from legoesm.core.fv3_native_dcmip16_ic import (
        dcmip16_bc_six_face_state,
    )
    return dcmip16_bc_six_face_state(
        ctx, ak, bk, KM, hydrostatic=not nh, do_pert=True, zvir=zvir)


# --------------------------------------------------------------------
# The moist deck's preconditions, ASSERTED from its own input.nml
# --------------------------------------------------------------------
# Reading the Fortran established that fv_phys is a no-op for this deck;
# these two functions make that a CHECKED precondition of every moist
# run instead of a claim in a docstring. The chain is:
#   atmosphere.F90:474  calls fv_phys whenever npz /= 1 .and. .not.
#                       adiabatic -- and the moist deck IS .not.
#                       adiabatic, so fv_phys DOES get called;
#   fv_phys.F90:590     applies nothing unless `no_tendency` was cleared;
#   no_tendency is cleared ONLY by fv_sg_adj > 0 (:303), do_LS_cond
#   (:306), K_sedi_transport under do_K_warm_rain (:412),
#   do_GFDL_sim_phys (:456), do_reed_sim_phys (:530), and the
#   do_Held_Suarez / do_surf_drag pair (:533/:541, which also writes pt
#   in place at :559-563).
# do_Held_Suarez lives in &fv_core_nml and defaults .false.
# (fv_arrays.F90:511); the rest live in the sim_phys namelists and all
# default .false. (fv_phys.F90:88-142) EXCEPT do_strat_HS_forcing, which
# is .true. by default but is only an ARGUMENT to Held_Suarez_Tend and
# is therefore unreachable while do_Held_Suarez is false.
_PHYSICS_SWITCHES = (
    "do_held_suarez", "do_surf_drag", "do_ls_cond", "do_k_warm_rain",
    "do_gfdl_sim_phys", "do_reed_sim_phys", "do_terminator",
)


def _nml_text(run_dir: str) -> str:
    path = os.path.join(run_dir, "input.nml")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: cannot verify the deck")
    with open(path) as fh:
        return fh.read()


def _nml_value(text: str, key: str, pattern: str):
    """The LAST assignment wins, comments stripped first.

    ONE parser for logicals, integers and reals. The first version had
    ``_nml_logical`` doing this correctly and then let ``fv_sg_adj`` and
    ``consv_te`` bypass it with a raw ``re.search`` -- which is
    FIRST-match-wins and comment-blind, so ``fv_sg_adj = -1`` followed
    by ``fv_sg_adj = 1``, or a commented-out ``consv_te = 0.`` above a
    live non-zero one, would have been read as the safe value and the
    harness would have scored dynamics against dynamics-plus-physics
    (codex MAJOR, job 9442717). These decks routinely carry a commented
    alternative and then the live value -- ``duogrid``, ``do_schmidt``
    and ``dnats`` all appear twice in the pinned ones -- so this is the
    shape the input actually has, not a hypothetical.

    KNOWN LIMIT, stated rather than fixed: the search is whole-file and
    GROUP-BLIND, so a key set in a different ``&group`` with a different
    value would be read here where Fortran would not apply it.  The keys
    checked (adiabatic, fv_sg_adj, consv_te, the physics switches) are
    unique across groups in every pinned deck; a deck that reused one
    would need a group-aware parser.  Likewise a ``!`` inside a quoted
    string truncates the line.

    Returns the last match's captured group, or None.
    """
    val = None
    for raw in text.splitlines():
        line = raw.split("!", 1)[0]          # `!` starts a comment
        m = re.search(rf"\b{key}\s*=\s*({pattern})", line, re.IGNORECASE)
        if m:
            val = m.group(1)
    return val


def _nml_logical(text: str, key: str):
    tok = _nml_value(text, key, r"\.?[A-Za-z]+\.?")
    if tok is None:
        return None
    return tok.strip().lower().strip(".") in ("t", "true")


def _nml_int(text: str, key: str):
    tok = _nml_value(text, key, r"[-+]?\d+")
    return None if tok is None else int(tok)


def _nml_real(text: str, key: str):
    tok = _nml_value(text, key, r"[-+]?[0-9.]+(?:[eEdD][-+]?[0-9]+)?")
    if tok is None:
        return None
    return float(tok.replace("d", "e").replace("D", "e"))


def check_deck_matches_the_arm(run_dir: str, *, nh: bool,
                               moist: bool, consv: float = 0.0,
                               physics: str = "none") -> None:
    """The deck must BE the arm the flags say it is.

    The redirects fire only on exact default-path equality, so explicit
    ``--ic-run``/``--step-run`` bypassed every content check -- and
    ``--moist --ic-run <nh deck>`` without ``--nh`` would run the
    HYDROSTATIC port against NH files, ignore W/DZ, and print a
    meaningless score without raising (codex MAJOR / GLM M2, jobs
    9444413 and 9444414). Worse, if the DRY step deck were ever
    regenerated moist, the dry arm would score a zvir=0 port against a
    moist reference at ~1e-6 rel -- comfortably under its own gate.

    So every arm, not just ``--moist``, asserts the deck's own resolved
    namelist against the flags: ``adiabatic`` is ``.false.`` IFF moist,
    ``hydrostatic``/``phys_hydrostatic`` track ``nh``, ``consv_te`` is
    0, and the physics cannot have touched the state.
    """
    if physics == "held_suarez":
        # The physics arm is the mirror of the inert check: the deck MUST run
        # exactly one physics path (Held-Suarez), dry (nwat=0, the cp-factor=1
        # premise), hydrostatic, no energy fixer, nothing else on.
        text = _nml_text(run_dir)
        if _nml_logical(text, "do_held_suarez") is not True:
            raise SystemExit(
                f"{run_dir}: --physics held_suarez needs do_held_suarez=.true.")
        if _nml_logical(text, "adiabatic") is not False:
            raise SystemExit(
                f"{run_dir}: the HS arm needs adiabatic=.false. so fv_phys runs "
                f"(atmosphere.F90:474).")
        if _nml_int(text, "nwat") != 0:
            raise SystemExit(
                f"{run_dir}: the HS apply port assumes nwat=0 (moist_cp default "
                f"-> cp_air, factor 1.0); deck has nwat={_nml_int(text,'nwat')}.")
        others = [k for k in _PHYSICS_SWITCHES
                  if k != "do_held_suarez" and _nml_logical(text, k) is True]
        if others:
            raise SystemExit(
                f"{run_dir}: physics switches {others} beyond do_held_suarez "
                f"are ON; not ported.")
        sg = _nml_int(text, "fv_sg_adj")
        if sg is None or sg > 0:
            raise SystemExit(
                f"{run_dir}: fv_sg_adj must be pinned <= 0 (got {sg}); > 0 runs "
                f"fv_subgrid_z (fv_phys.F90:305).")
        for key in ("hydrostatic", "phys_hydrostatic"):
            if _nml_logical(text, key) is not True:
                raise SystemExit(
                    f"{run_dir}: the HS arm is hydrostatic; {key} must be .true.")
        got = _nml_real(text, "consv_te")
        if got not in (None, 0.0):
            raise SystemExit(
                f"{run_dir}: the HS arm expects consv_te 0, got {got}.")
        return
    check_physics_is_inert(run_dir)
    text = _nml_text(run_dir)
    want_adiab = not moist
    got_adiab = _nml_logical(text, "adiabatic")
    if got_adiab is not want_adiab:
        raise SystemExit(
            f"{run_dir}: resolves adiabatic = {got_adiab!r} but the "
            f"flags say moist = {moist}. In the solo driver that one "
            f"flag IS the moisture switch (atmosphere.F90:156-161), so "
            f"this pairing scores the port against the wrong physics.")
    for key in ("hydrostatic", "phys_hydrostatic"):
        got = _nml_logical(text, key)
        if got is not (not nh):
            raise SystemExit(
                f"{run_dir}: resolves {key} = {got!r} but the flags say "
                f"nh = {nh}. A hydrostatic port loading NH files simply "
                f"ignores W/DZ and prints a number.")
    got = _nml_real(text, "consv_te")
    if got is None or got != consv:
        raise SystemExit(
            f"{run_dir}: resolves consv_te = "
            f"{got if got is not None else 'nothing'}, but this arm "
            f"expects {consv}. The fixer is last_step-only and moves pt "
            f"by ~4e-6 K, so a mismatched deck reads as a defect rather "
            f"than a configuration error.")


def check_physics_is_inert(run_dir: str) -> None:
    """Refuse a deck whose fv_phys could touch the state."""
    text = _nml_text(run_dir)
    on = [k for k in _PHYSICS_SWITCHES if _nml_logical(text, k) is True]
    if on:
        raise SystemExit(
            f"{run_dir}: physics switches {on} are ON in input.nml. This "
            f"deck's RESTART is one dynamics step PLUS a forcing "
            f"tendency, and the port has no physics -- the residual "
            f"would be unattributable. Use a deck with them off.")
    sg = _nml_int(text, "fv_sg_adj")
    if sg is None:
        raise SystemExit(f"{run_dir}: input.nml does not set fv_sg_adj; "
                         f"fv_phys.F90:303 clears no_tendency when it is "
                         f"> 0, so it must be pinned, not defaulted.")
    if sg > 0:
        raise SystemExit(
            f"{run_dir}: fv_sg_adj = {sg} > 0 runs fv_subgrid_z "
            f"(fv_phys.F90:305) and clears no_tendency. Not ported.")


def check_moist_deck(run_dir: str) -> None:
    # SUPERSEDED by check_deck_matches_the_arm, which asks the same
    # questions with a DIRECTION (adiabatic .false. IFF moist) and adds
    # the hydrostatic pair. Kept because it is the narrower, standalone
    # statement of "this deck is moist" and its tests pin the namelist
    # reader; the parity harness itself no longer calls it.
    """The moist coupling is a DERIVED flag; assert what derives it.

    ``zvir`` is nowhere in the namelist. atmosphere.F90:156-161 sets it
    to ``rvgas/rdgas - 1`` exactly when ``adiabatic = .false.``, and
    sets ``moist_phys = .true.`` in the same branch. So the deck must
    say ``adiabatic = .false.`` or the run this is scored against was
    DRY and the comparison is a category error, not a tolerance
    question. ``consv_te`` must also be 0: the energy fixer is refused.
    """
    text = _nml_text(run_dir)
    adiab = _nml_logical(text, "adiabatic")
    if adiab is not False:
        raise SystemExit(
            f"{run_dir}: input.nml resolves adiabatic = {adiab!r}, not "
            f".false. -- atmosphere.F90:157-161 then leaves zvir = 0 and "
            f"the oracle ran DRY. Scoring the moist port against it "
            f"would measure the coupling itself as the error.")
    consv = _nml_real(text, "consv_te")
    if consv is None or consv != 0.0:
        raise SystemExit(
            f"{run_dir}: consv_te must be pinned to 0 (found "
            f"{consv if consv is not None else 'nothing'}); the "
            f"total-energy fixer (fv_mapz.F90:628-747) is not ported.")


def build_port_tracer_ic(sphum6) -> list:
    """[face][iq] padded tracer arrays for the resolved deck.

    ``sphum`` comes from :func:`build_port_ic`, NOT from a second call:
    on the moist arm the IC's ``pt`` was divided by ``(1 + zvir*q)``
    using that exact array (``test_cases.F90:6760``), so rebuilding q
    here would make the two halves of one IC come from two
    constructions. ``liq_wat`` is identically zero (:6728-6735 zeroes
    all tracers and only sphum is filled -- CONFIRMED against the
    zerostep fv_tracer.res: liq_wat/rainwat are 0.0 everywhere). Halos
    stay zero: the init-time ``mpp_update_domains(q)`` is inside the
    terminator-tracer branch (cl/cl2), absent on this deck; tracer_2d
    fills its own halos via ext_scalar.
    """
    if sphum6 is None:
        raise ValueError(
            "build_port_tracer_ic got sphum6=None -- build_port_ic was "
            "called with with_sphum=False, so there is no humidity to "
            "advect and none to have divided pt on the moist arm.")
    return [[q, np.zeros_like(q)] for q in sphum6]

def tracer_window(q6, ctx) -> list:
    """Compute-window copies of the advected tracers, (i, j, k)."""
    n, ng = ctx["n"], ctx["ng"]
    cs = slice(ng, ng + n)
    return [{nm: np.array(q6[t][iq][cs, cs, :])
             for iq, nm in enumerate(ADVECTED_TRACERS)}
            for t in range(6)]


def map_scalar_pair(port_arr, orc_arr, meta_entry) -> tuple:
    """One cell-centred scalar under the derived face map (factor +1)."""
    transposed, nm, _su, _sv = meta_entry
    return DIHEDRAL[nm](port_arr), oracle_ij(orc_arr, transposed)


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
# Coherent boundary-metric perturbation (codex retro-review findings
# 2+3).  The old in-place perturbation (a) skipped directly consumed
# derived families (rdx/rdy/rdxa/rdya/rdxc/rdyc feed p_grad_c and the
# d_sw KE ranges, rarea_c the CD-vorticity, B-grid cosa/sina/rsina the
# duo contravariant solves), (b) perturbed area AND rarea by the SAME
# factor so area*rarea became fac^2 (invariant violated -- the response
# then mixes the intended geometry mode with an unphysical one), and
# (c) never refreshed the prebuilt ectx metric snapshots (dx6/dy6),
# so the c2l path consumed UNPERTURBED lengths.  Here: perturb the
# PRIMITIVES only, then recompute every derived reciprocal/composite
# where the builder invariant held (sentinel/override slots preserved
# bit-exactly), and refresh the ectx snapshots with the same factors.

# primitive -> multiplicative class; angles get a coherent ROTATION
# theta -> theta + eps*pat instead (multiplying cosa by (1+eps*pat)
# perturbs a near-zero cosa by ~nothing -- the perturb-a-zero trap --
# while a rotation moves cos and sin by O(eps) everywhere and keeps
# cos^2+sin^2 = 1 exactly).
_PERT_LENGTHS = ("dx", "dy", "dxa", "dya", "dxc", "dyc",
                 "area", "area_c")
_PERT_ANGLE_PAIRS = (("cosa_u", "sina_u"), ("cosa_v", "sina_v"),
                     ("cosa", "sina"))
_SENT_GUARD = 1.0e29     # never touch big_number convention slots
_TRIG_GUARD = 1.0e6      # trig sentinel class (poisoned vertices)


def _pert_pattern(shape2, ng, eps):
    """eps*cos(3i+7j) on boundary cells (halo rings + outermost
    compute ring), exactly 0 on the strict interior."""
    mi, mj = shape2
    pat = np.cos(3.0 * np.arange(mi)[:, None]
                 + 7.0 * np.arange(mj)[None, :])
    di = np.minimum(np.arange(mi), mi - 1 - np.arange(mi))
    dj = np.minimum(np.arange(mj), mj - 1 - np.arange(mj))
    strict = (di[:, None] > ng) & (dj[None, :] > ng)
    return np.where(strict, 0.0, eps * pat)


def _replace_where_held(gs, key, cand_old, cand_new, stats):
    """Refresh derived field gs[key] ONLY where the builder invariant
    held pre-perturbation (rtol 1e-12) AND the recomputed value
    actually moved (cand_new != cand_old).  The second condition keeps
    every untouched cell BIT-EXACT: without it, cells whose formula
    output did not change were still overwritten by the recomputation,
    re-rounding sentinel-derived values (1e30*1e30/1e30 != 1e30
    bitwise) and any cell the builder computed through a different
    expression path -- a fake "moved" count and an unintended
    interior-noise perturbation (caught by
    test_zero_cell_perturbation_is_refused)."""
    old = np.asarray(gs[key], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        held = np.isclose(old, cand_old, rtol=1.0e-12, atol=0.0)
    changed = held & (cand_new != cand_old)
    gs[key] = np.where(changed, cand_new, old)
    stats[key] = stats.get(key, 0) + int(changed.sum())


def perturb_boundary_metrics_coherent(ctx, eps: float, n: int, ng: int):
    """Perturb primitive metrics at boundary cells, recompute derived
    families, refresh ectx snapshots. Prints a per-class receipt."""
    from legoesm.grids.fv3_native_gridstruct import (
        TINY_NUMBER,
        rsin_border_override,
    )

    stats = {}
    for t in range(6):
        gs = ctx["gs6"][t]
        old = {}     # pre-perturbation primitives (fp64 copies)
        for k in _PERT_LENGTHS + tuple(
                x for pr in _PERT_ANGLE_PAIRS for x in pr) + (
                "sin_sg", "cos_sg"):
            old[k] = np.asarray(gs[k], dtype=np.float64).copy()
        # snapshot derived-formula inputs BEFORE any write
        for k in ("rdx", "rdy", "rdxa", "rdya", "rdxc", "rdyc",
                  "rarea", "rarea_c", "rsina", "rsin_u", "rsin_v",
                  "rsin2", "divg_u", "divg_v", "del6_u", "del6_v"):
            old[k] = np.asarray(gs[k], dtype=np.float64).copy()

        # 1) lengths/areas: multiplicative on non-sentinel boundary cells
        for k in _PERT_LENGTHS:
            a = old[k]
            fac = 1.0 + _pert_pattern(a.shape, ng, eps)
            gs[k] = np.where(np.abs(a) < _SENT_GUARD, a * fac, a)
            stats[k] = stats.get(k, 0) + int(
                ((np.abs(a) < _SENT_GUARD) & (fac != 1.0)).sum())
        # 2) angle pairs: coherent ROTATION-MATRIX perturbation on
        # real-trig boundary cells: (c,s) -> (c cos(dth) - s sin(dth),
        # s cos(dth) + c sin(dth)).  This moves both members by O(eps)
        # everywhere (no perturb-a-zero hole at cosa ~ 0) and scales
        # the pair's norm by exactly (1 + dth^2) ~ 1 + 1e-24 -- it
        # PRESERVES whatever c^2+s^2 the builder produced.  The earlier
        # cos/sin(arctan2(s,c)+dth) form silently NORMALIZED the pair,
        # which at panel-edge B nodes -- where upstream's edge
        # averaging leaves c^2+s^2-1 ~ 1e-8 -- injected a 1e-8 kick,
        # four orders above eps (caught by the invariants test).
        # The write is gated on dth != 0 so untouched cells stay
        # bit-exact (codex instr r1 H1).
        for ck, sk in _PERT_ANGLE_PAIRS:
            c, s = old[ck], old[sk]
            real = (np.abs(c) < _TRIG_GUARD) & (np.abs(s) < _TRIG_GUARD)
            dth = _pert_pattern(c.shape, ng, eps)
            rot = real & (dth != 0.0)
            gs[ck] = np.where(rot, c * np.cos(dth) - s * np.sin(dth), c)
            gs[sk] = np.where(rot, s * np.cos(dth) + c * np.sin(dth), s)
            stats[ck] = stats.get(ck, 0) + int(rot.sum())
        # 3) sg families: the same slot-wise pair rotation, gated to
        # genuine unit-norm angle slots only -- ghost/transport-patch
        # convention slots (tiny 1e-8 floors, corner patches) violate
        # the unit norm and stay bit-exact
        ssg, csg = old["sin_sg"], old["cos_sg"]
        real = np.abs(ssg * ssg + csg * csg - 1.0) < 1.0e-12
        dth = _pert_pattern(ssg.shape[:2], ng, eps)[:, :, None]
        rot = real & (dth != 0.0)
        gs["sin_sg"] = np.where(rot, ssg * np.cos(dth) + csg * np.sin(dth),
                                ssg)
        gs["cos_sg"] = np.where(rot, csg * np.cos(dth) - ssg * np.sin(dth),
                                csg)
        stats["sin_sg"] = stats.get("sin_sg", 0) + int(rot.sum())

        # 4) recompute EVERY derived family where its builder formula
        # held (invariants restored: area*rarea == 1 again, etc.)
        recips = (("rdx", "dx"), ("rdy", "dy"), ("rdxa", "dxa"),
                  ("rdya", "dya"), ("rdxc", "dxc"), ("rdyc", "dyc"),
                  ("rarea", "area"), ("rarea_c", "area_c"))
        for dk, pk in recips:
            _replace_where_held(gs, dk, 1.0 / old[pk],
                                1.0 / np.asarray(gs[pk]), stats)
        for dk, pk in (("rsina", "sina"), ("rsin_u", "sina_u"),
                       ("rsin_v", "sina_v")):
            so, sn = old[pk], np.asarray(gs[pk])
            pristine = old[dk]
            if pristine.shape != so.shape:
                # rsina is COMPUTE-B in some lanes; skip on mismatch
                continue
            # two builder flavours: the plain 1/max(tiny, s^2) and the
            # panel-border override 1/SIGN(max(tiny,|s|), s); each cell
            # is refreshed by the flavour whose invariant it held.
            c1o = 1.0 / np.maximum(TINY_NUMBER, so * so)
            c1n = 1.0 / np.maximum(TINY_NUMBER, sn * sn)
            c2o = rsin_border_override(so)
            c2n = rsin_border_override(sn)
            h1 = (np.isclose(pristine, c1o, rtol=1.0e-12, atol=0.0)
                  & (c1n != c1o))
            h2 = (np.isclose(pristine, c2o, rtol=1.0e-12, atol=0.0)
                  & (c2n != c2o) & ~h1)
            gs[dk] = np.where(h1, c1n, np.where(h2, c2n, pristine))
            stats[dk] = stats.get(dk, 0) + int((h1 | h2).sum())
        _replace_where_held(
            gs, "rsin2",
            1.0 / np.maximum(TINY_NUMBER, old["sin_sg"][..., 4] ** 2),
            1.0 / np.maximum(TINY_NUMBER,
                             np.asarray(gs["sin_sg"])[..., 4] ** 2),
            stats)
        _replace_where_held(gs, "cosa_s", old["cos_sg"][..., 4],
                            np.asarray(gs["cos_sg"])[..., 4], stats)
        for dk, f in (
                ("divg_u", lambda o: o[0] * o[1] / o[2]),
                ("del6_u", lambda o: o[0] * o[2] / o[1])):
            _replace_where_held(
                gs, dk,
                f((old["sina_v"], old["dyc"], old["dx"])),
                f((np.asarray(gs["sina_v"]), np.asarray(gs["dyc"]),
                   np.asarray(gs["dx"]))), stats)
        for dk, f in (
                ("divg_v", lambda o: o[0] * o[1] / o[2]),
                ("del6_v", lambda o: o[0] * o[2] / o[1])):
            _replace_where_held(
                gs, dk,
                f((old["sina_u"], old["dxc"], old["dy"])),
                f((np.asarray(gs["sina_u"]), np.asarray(gs["dxc"]),
                   np.asarray(gs["dy"]))), stats)

        # 3b) f0 (codex instr r2 H1): a consumed coordinate-derived
        # static (d_sw5 vort = wk + f0, full domain incl the corner
        # wedges that carried the pre-#1585 defect).  ADDITIVE
        # eps*2*Omega*pat at boundary cells -- multiplicative would be
        # a no-op exactly on the equator line where f0 = 0 (the
        # perturb-a-zero trap).  f0 has no derived fields and no ectx
        # snapshot.
        from legoesm.grids.fv3_native_gridstruct import FV3_OMEGA
        f0a = np.asarray(gs["f0"], dtype=np.float64).copy()
        d_f0 = _pert_pattern(f0a.shape, ng, eps) * (2.0 * FV3_OMEGA)
        okf = (np.abs(f0a) < _TRIG_GUARD) & (d_f0 != 0.0)
        gs["f0"] = np.where(okf, f0a + d_f0, f0a)
        stats["f0"] = stats.get("f0", 0) + int(okf.sum())

        # 4b) consumed metric SUMMARIES (codex instr r1 H2): d_sw and
        # the duo sw core read the scalars da_min/da_min_c, computed by
        # the builder as min/max over the compute range (bounded lane:
        # area[ng:ng+n, ng:ng+n], and area_c over the SAME range -- the
        # upstream global_mx_c call range, NOT ie+1).  The perturbation
        # covers the outermost compute ring, so the extremum can move
        # while the stored scalar goes stale.  Where-held analog for
        # scalars: refresh only if the stored value matches the formula
        # on the pre-perturbation arrays.
        sl = slice(ng, ng + n)
        for sk2, pk2, red in (("da_min", "area", np.min),
                              ("da_max", "area", np.max),
                              ("da_min_c", "area_c", np.min),
                              ("da_max_c", "area_c", np.max)):
            if sk2 not in gs:
                continue
            cand_old = float(red(old[pk2][sl, sl]))
            if np.isclose(float(gs[sk2]), cand_old,
                          rtol=1.0e-12, atol=0.0):
                cand_new = float(red(np.asarray(gs[pk2])[sl, sl]))
                if cand_new != cand_old:
                    gs[sk2] = cand_new
                    stats[sk2] = stats.get(sk2, 0) + 1

        # 5) refresh the prebuilt ectx metric snapshots: c2l reads
        # ectx["dx6"]/["dy6"], captured BEFORE this perturbation --
        # apply the SAME factors so the consumed values actually move.
        # (amat6 is built from grid_lon/grid_lat coordinates only --
        # unaffected by a metric perturbation, no rebuild needed.)
        if ctx.get("ectx") is not None:
            for ek, pk in (("dx6", "dx"), ("dy6", "dy")):
                a = np.asarray(ctx["ectx"][ek][t], dtype=np.float64)
                fac = 1.0 + _pert_pattern(a.shape, ng, eps)
                ctx["ectx"][ek][t] = np.where(
                    np.abs(a) < _SENT_GUARD, a * fac, a)

    total = sum(stats.values())
    if total == 0:
        raise SystemExit(
            "PERTURBATION CONTROL FAILED: zero cells moved -- the "
            "probe would be a no-op control (perturb-a-zero class)")
    print(f"BOUNDARY METRICS PERTURBED COHERENTLY eps={eps:g}: "
          f"primitives multiplicative (lengths/areas) + angle "
          f"rotations; ALL derived reciprocals/composites recomputed; "
          f"ectx dx6/dy6 snapshots refreshed")
    print("  perturbed/refreshed cells per family: "
          + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))


def _make_jax_step(ctx):
    """A drop-in for ``fv_dynamics_step`` that steps with the JAX lane.

    The scoring below is ~400 lines that read six per-face NumPy dicts
    and mutate ``state``/``press``/``q`` in place, exactly as the Fortran
    dummies are.  The JAX lane is face-STACKED and FUNCTIONAL (C1/C4), so
    this adapter stacks on the way in and writes back on the way out --
    and NOTHING ELSE about the run changes.  That is deliberate: the IC,
    the oracle files, the derived face map, the tendency scales and every
    tolerance stay byte-identical to the NumPy invocation, so the two
    backends' scores differ only by the lane under test.  A separate
    JAX-flavoured harness would have made any difference unattributable,
    which is the confound this campaign has paid for more than once.

    The write-back is the only subtle part: the caller keeps references
    to the per-face arrays, so the values are copied INTO the existing
    arrays rather than rebound, or the mutation the scoring relies on
    would be invisible.
    """
    import jax.numpy as _jnp
    import numpy as _np

    from legoesm.core import fv3_dynamics as _jdyn
    from legoesm.core.fv3_cgrid_phase_3d import state_3d_to_jax
    from legoesm.core.fv3_duo_stepper import build_jax_duo_stepper_context

    jctx = build_jax_duo_stepper_context(ctx)

    def _stack(per_face, key=None):
        # jnp, not np: the module indexes these with `.at[...]`, which a
        # numpy array does not have. Stacking with numpy and handing the
        # result straight over got as far as the remap's pk update
        # before failing.
        return _jnp.asarray(_np.stack([_np.asarray(f[key] if key else f)
                                       for f in per_face]))

    def _step(_ctx, state, press, **kw):
        jstate = state_3d_to_jax(state)
        jpress = {nm: _stack(press, nm)
                  for nm in ("ps", "pe", "peln", "pk", "pkz")}
        q_in = kw.pop("q")
        # TRACER-MAJOR: fv_dynamics_step takes nq face-stacked arrays,
        # not one (6, nq, ...) stack. The spec's q is [face][iq], so the
        # transpose is here, in the adapter, where every other layout
        # difference between the lanes already lives.
        _nq = len(q_in[0])
        jq = [_jnp.asarray(_np.stack([_np.asarray(q_in[t][iq])
                                      for t in range(6)]))
              for iq in range(_nq)]
        out = _jdyn.fv_dynamics_step(jctx, jstate, jpress, q=jq, **kw)

        # Write back IN PLACE -- see the docstring.
        st = out["state"] if "state" in out else out
        for t in range(6):
            for nm, arr in state[t].items():
                if nm in st:
                    arr[...] = _np.asarray(st[nm])[t]
            for nm in ("ps", "pe", "peln", "pk", "pkz"):
                if nm in out["press"]:
                    press[t][nm][...] = _np.asarray(out["press"][nm])[t]
            for iq, arr in enumerate(q_in[t]):
                arr[...] = _np.asarray(out["q"][iq])[t]
        return out

    return _step


def apply_held_suarez_step(ctx, state, press, *, dt, n, ng, km, strat=True):
    """In place: advance the six-face POST-DYNAMICS duo state by one
    Held-Suarez physics step -- the port of fv_phys + fv_update_phys for
    do_Held_Suarez=.true., dry, hydrostatic, nwat=0 (driver/solo/fv_phys.F90:
    533-591).  fv_dynamics has already run; this mutates state[t]["u"/"v"/"pt"].

    Three passes because the u_dt/v_dt one-cell halo exchange is a cross-face
    barrier (fv_update_phys.F90:645/698, dwind_2d=.false.): (1) per face, D->A
    winds (d2a2c_vect_duo), the Held-Suarez tendencies on the compute domain,
    scattered back into full-domain arrays with zero halos; (2) exchange the
    u_dt/v_dt halos; (3) per face, apply fv_update_phys_dry_duo.  The pe/peln
    axis fix (i,k,j)->(i,j,k) and the pe 1-ring window match p_var_hydrostatic
    (verified against fv3_native_dynamics.py:198-234); pkz is already cell-domain
    k-last.  GLM-authored; codex + Claude reviewed.
    """
    from legoesm.core.fv3_native_duo_sw_core import d2a2c_vect_duo
    from legoesm.grids.fv3_native_metrics import compute_fv3_native_wind_vectors
    from legoesm.core.fv3_native_physics_coupling import (
        held_suarez_tend, fv_update_phys_dry_duo)
    from legoesm.grids.fv3_native_gridstruct import exchange_agrid_scalar_halos

    m = n + 2 * ng
    ci = slice(ng, ng + n)
    assert state[0]["pt"].shape == (m, m, km)
    assert press[0]["pkz"].shape == (n, n, km)

    ua6 = [None] * 6
    va6 = [None] * 6
    t_dt6 = [None] * 6
    u_dt6 = [None] * 6
    v_dt6 = [None] * 6

    # PASS 1: per-face tendencies on the compute domain
    for t in range(6):
        gs = ctx["gs6"][t]
        d = d2a2c_vect_duo(state[t]["u"], state[t]["v"], gs, ctx["bd"],
                           n + 1, n + 1)
        ua_f, va_f = d["ua"], d["va"]                        # (m, m, km)
        ua6[t], va6[t] = ua_f, va_f

        pt_c = state[t]["pt"][ci, ci]                        # (n, n, km)
        delp_c = state[t]["delp"][ci, ci]
        ua_c, va_c = ua_f[ci, ci], va_f[ci, ci]
        pkz_c = press[t]["pkz"]                              # already (n, n, km)
        peln_c = np.transpose(press[t]["peln"], (0, 2, 1))   # (i,k,j)->(i,j,k)
        pe_c = np.transpose(press[t]["pe"][1:n + 1, :, 1:n + 1], (0, 2, 1))
        lat_c = gs["agrid_lat"][ci, ci]                      # (n, n) radians

        t_dt_c, u_dt_c, v_dt_c = held_suarez_tend(
            pt_c, ua_c, va_c, delp_c, peln_c, pkz_c, pe_c, lat_c, dt,
            strat=strat)

        t_dt_f = np.zeros((m, m, km), dtype=np.float64)
        u_dt_f = np.zeros((m, m, km), dtype=np.float64)
        v_dt_f = np.zeros((m, m, km), dtype=np.float64)
        t_dt_f[ci, ci] = t_dt_c
        u_dt_f[ci, ci] = u_dt_c
        v_dt_f[ci, ci] = v_dt_c
        t_dt6[t], u_dt6[t], v_dt6[t] = t_dt_f, u_dt_f, v_dt_f

    # PASS 2: cross-face one-cell halo exchange of the vector tendencies
    for tile in range(1, 7):
        exchange_agrid_scalar_halos(u_dt6, tile, n, ng)
    for tile in range(1, 7):
        exchange_agrid_scalar_halos(v_dt6, tile, n, ng)

    # PASS 3: apply
    for t in range(6):
        gs = ctx["gs6"][t]
        wv = compute_fv3_native_wind_vectors(
            gs["grid_lon"], gs["grid_lat"], gs["agrid_lon"], gs["agrid_lat"])
        u2, v2, pt2, _, _ = fv_update_phys_dry_duo(
            state[t]["u"], state[t]["v"], state[t]["pt"], ua6[t], va6[t],
            u_dt6[t], v_dt6[t], t_dt6[t], dt,
            wv["vlon"], wv["vlat"], wv["es1"], wv["ew2"], ng)
        state[t]["u"], state[t]["v"], state[t]["pt"] = u2, v2, pt2
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ic-run", default=f"{ORACLE_ROOT}/run_hydro_zerostep")
    ap.add_argument("--step-run", default=f"{ORACLE_ROOT}/run_hydro_1step_gfs")
    ap.add_argument("--n-split", type=int, default=N_SPLIT)
    ap.add_argument("--k-split", type=int, default=K_SPLIT)
    ap.add_argument("--dt", type=float, default=DT_ATMOS)
    ap.add_argument("--json", default=None)
    ap.add_argument("--save-fields", default=None,
                    help="write an .npz of the MAPPED one-step port "
                         "and oracle planes (the same arrays the "
                         "residual table scores, after the face map "
                         "and dihedral) so the maps a human looks at "
                         "are the arrays the gate scored, not a "
                         "second rendering of the state.")
    ap.add_argument("--trace-substeps", action="store_true",
                    help="DIAGNOSTIC: run the acoustic loop one sub-step at "
                         "a time and print max|field - IC| after each, then "
                         "stop. Localises a wrong tendency in time.")
    ap.add_argument("--max-rel", type=float, default=None,
                    help="gate: exit 1 if any field's one-step rel exceeds "
                         "this")
    ap.add_argument("--tracers", action="store_true",
                    help="score tracer advection (fv_tracer2d port): "
                         "initialise sphum/liq_wat from the analytic "
                         "test_case=-13 IC, advect them through "
                         "tracer_2d_1L + the remap, and score against "
                         "the deck's own fv_tracer.res restarts on "
                         "each tracer's OWN scale. liq_wat/rainwat are "
                         "identically 0 in the oracle IC (checked at "
                         "runtime), so their agreement is VACUOUS as a "
                         "transport test and is labelled so -- sphum "
                         "carries the signal.")
    ap.add_argument("--ext-metrics", action="store_true",
                    help="build the six-face context with "
                         "use_ext_metrics=True: halo/corner-wedge cells "
                         "carry EXTENDED-lattice metrics instead of the "
                         "kinked builder's. fv_grid_tools.F90:749-835 "
                         "shows the duo oracle builds its model grid "
                         "FROM dg%b_pt with every mpp/fill_corners/"
                         "get_symmetry step skipped -- so the extended "
                         "lattice is the FAITHFUL halo geometry and the "
                         "kinked one is the port's residual suspect. "
                         "This flag is the mechanism-scaling probe.")
    ap.add_argument("--perturb-boundary-metrics", type=float, default=0.0,
                    metavar="EPS",
                    help="seed discriminator, COHERENT protocol (codex "
                         "retro 2026-08-11): perturb PRIMITIVE metrics "
                         "at BOUNDARY cells (halo rings + outermost "
                         "compute ring) -- lengths/areas by "
                         "(1 + EPS*cos(3i+7j)), intersection angles by "
                         "a rotation theta+EPS*pat -- then RECOMPUTE "
                         "every derived reciprocal/composite "
                         "(rdx=1/dx, rarea=1/area, rsina, divg/del6) "
                         "so builder invariants hold, perturb f0 "
                         "(additive eps*2*Omega), and refresh the "
                         "prebuilt ectx dx6/dy6 snapshots. The "
                         "production metrics match the oracle at "
                         "~1e-14; if the one-step residual scales with "
                         "EPS (pre-registered: 1e-12 and 1e-10), the "
                         "9.8e-06 floor is amplified geometry seed; if "
                         "it stays pinned, the floor is a composition/"
                         "formulation difference.")
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
    ap.add_argument("--consv", type=float, default=0.0,
                    help="ENERGY-FIXER gate: score against "
                         "run_hydro_{zerostep,1step}_consv_gfs, the "
                         "certified hydrostatic deck with consv_te set to "
                         "this value (build_consv_te_oracle.sbatch). The "
                         "fixer is last_step-only and touches pt alone, "
                         "so its signal is ~4.2e-06 K -- only ~12x this "
                         "arm's 1.1866e-09 floor, which is thin. The "
                         "RESPONSE gate is what certifies it, exactly as "
                         "on the moist arms.")
    ap.add_argument("--moist", action="store_true",
                    help="MOIST gate: defaults the runs to "
                         "run_hydro_{zerostep,1step}_moist_gfs, which are "
                         "the same test_case=-13 deck with "
                         "adiabatic=.false. -- and in the solo driver "
                         "(atmosphere.F90:156-161) that ONE flag is what "
                         "sets zvir = rvgas/rdgas - 1 and moist_phys=T. "
                         "Requires --tracers, because dp1 = zvir*q(sphum) "
                         "has nothing to read otherwise. The deck's "
                         "physics switches are ASSERTED inert from its own "
                         "input.nml (see check_physics_is_inert); a deck "
                         "that ran Held-Suarez or Kessler would score the "
                         "port against dynamics PLUS forcing and the "
                         "residual would be unattributable.")
    ap.add_argument("--backend", choices=("numpy", "jax"), default="numpy",
                    help="which lane takes the step. 'numpy' is the "
                         "SPECIFICATION and the established score; 'jax' "
                         "runs the ported lane through the SAME scoring "
                         "code, IC, oracle files, face map and tolerances, "
                         "so the two numbers are comparable by "
                         "construction. Only the stepping call differs -- "
                         "everything upstream and downstream of it is "
                         "byte-identical between the two backends, which "
                         "is the whole point (a JAX-specific harness would "
                         "make any difference unattributable).")
    ap.add_argument("--physics", choices=("none", "held_suarez"),
                    default="none",
                    help="run a physics-coupled step after dynamics and score "
                         "against a deck that ran it. 'held_suarez' targets "
                         "run_hs_1step_gfs (do_Held_Suarez=.true., dry, "
                         "hydrostatic, nwat=0) and applies "
                         "apply_held_suarez_step; the dynamics-only arms keep "
                         "refusing any deck whose physics is ON.")
    args = ap.parse_args(argv)
    if args.n_steps < 1:
        raise SystemExit(f"--n-steps must be >= 1, got {args.n_steps}")
    if args.n_steps != 1 and args.step_run in (
            f"{ORACLE_ROOT}/run_hydro_1step_gfs", None):
        # codex nstep review #3: comparing N port steps against the
        # 1-step reference is a silent protocol mismatch that returns
        # normally without --max-rel.  The N-step reference must be
        # chosen EXPLICITLY.
        raise SystemExit(
            f"--n-steps {args.n_steps} requires an explicit --step-run "
            f"pointing at an oracle run of exactly "
            f"{args.n_steps} * dt_atmos (e.g. run_hydro_"
            f"{args.n_steps * 32}min_gfs); the default is the 1-step "
            f"reference.")
    if args.nh:
        if args.ic_run == f"{ORACLE_ROOT}/run_hydro_zerostep":
            args.ic_run = f"{ORACLE_ROOT}/run_nh_zerostep_gfs"
        if args.step_run == f"{ORACLE_ROOT}/run_hydro_1step_gfs":
            args.step_run = f"{ORACLE_ROOT}/run_nh_1step_gfs"
    args.dry_twin_run = None
    if args.physics == "held_suarez":
        if args.nh or args.moist or args.consv:
            raise SystemExit(
                "--physics held_suarez is dry + hydrostatic; not with "
                "--nh/--moist/--consv.")
        if args.step_run == f"{ORACLE_ROOT}/run_hydro_1step_gfs":
            args.step_run = f"{ORACLE_ROOT}/run_hs_1step_gfs"
        if not os.path.isdir(args.step_run):
            raise SystemExit(
                f"missing HS oracle {args.step_run}; build it with "
                f"scripts/cluster/fv3_native/build_hs_oracle.sbatch")
    if args.consv:
        if args.moist or args.nh:
            raise SystemExit(
                "--consv is hydrostatic-and-dry only: the generated deck "
                "is hydrostatic, and both energy integrals take their "
                "NON-hydrostatic branches under --nh, which are not "
                "ported.")
        if args.ic_run == f"{ORACLE_ROOT}/run_hydro_zerostep":
            args.ic_run = f"{ORACLE_ROOT}/run_hydro_zerostep_consv_gfs"
        if args.step_run == f"{ORACLE_ROOT}/run_hydro_1step_gfs":
            args.step_run = f"{ORACLE_ROOT}/run_hydro_1step_consv_gfs"
        args.dry_twin_run = f"{ORACLE_ROOT}/run_hydro_1step_gfs"
        for _r in (args.ic_run, args.step_run):
            if not os.path.isdir(_r):
                raise SystemExit(
                    f"missing consv oracle run {_r}; build it with "
                    f"scripts/cluster/fv3_native/build_consv_te_oracle.sbatch")
    if args.moist:
        if not args.tracers:
            raise SystemExit(
                "--moist requires --tracers: dp1 = zvir*q(sphum) "
                "(fv_dynamics.F90:291) has no specific humidity to read "
                "without the tracer IC, and a zvir with q=None is refused "
                "by the lane anyway.")
        # --nh has already redirected these to the NH decks above, so
        # the moist redirect keys off whichever pair is in play. Both
        # arms then go through the same two deck checkers.
        _arm = "nh" if args.nh else "hydro"
        _dry_ic = (f"{ORACLE_ROOT}/run_nh_zerostep_gfs" if args.nh
                   else f"{ORACLE_ROOT}/run_hydro_zerostep")
        _dry_step = (f"{ORACLE_ROOT}/run_nh_1step_gfs" if args.nh
                     else f"{ORACLE_ROOT}/run_hydro_1step_gfs")
        args.dry_twin_run = _dry_step
        if args.ic_run == _dry_ic:
            args.ic_run = f"{ORACLE_ROOT}/run_{_arm}_zerostep_moist_gfs"
        if args.step_run == _dry_step:
            args.step_run = f"{ORACLE_ROOT}/run_{_arm}_1step_moist_gfs"
        for _r in (args.ic_run, args.step_run):
            if not os.path.isdir(_r):
                _how = ("scripts/cluster/fv3_native/build_nh_moist_oracle.sbatch"
                        if args.nh else "the shipped hydrostatic moist pair")
                raise SystemExit(
                    f"missing moist oracle run {_r}. The {_arm} pair "
                    f"comes from {_how}.")

    # EVERY ARM, not just --moist: the redirects fire only on exact
    # default-path equality, so an explicit --ic-run/--step-run used to
    # bypass all content checking.
    # check_deck_matches_the_arm subsumes check_physics_is_inert and the
    # direction-blind check_moist_deck, and load_oracle already asserts
    # FMSConstants: GFS per deck via require_gfs_constants.
    for _r in (args.ic_run, args.step_run):
        # the physics gate applies ONLY to the STEP deck; the IC (zerostep) is
        # dynamics-only (adiabatic=.true., physics off) and is checked inert.
        check_deck_matches_the_arm(
            _r, nh=args.nh, moist=args.moist,
            physics=(args.physics if _r == args.step_run else "none"),
            consv=args.consv)

    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.core.fv3_native_dynamics import (
        fv_dynamics_step, p_var_hydrostatic,
    )
    from legoesm.core.fv3_native_mapz import lagrangian_to_eulerian
    from legoesm.core.fv3_native_eta import set_eta_analytic
    from legoesm.core.fv3_native_state_3d import field_shape
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_CP_AIR, FV3_KAPPA, FV3_RDGAS, FV3_RVGAS,
    )

    print(f"port constants: kappa = {FV3_KAPPA!r}  cp_air = {FV3_CP_AIR!r}")
    print(f"                (2/7 = {2/7!r}; rel diff "
          f"{abs(FV3_KAPPA - 2/7)/(2/7):.3e})")

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     use_ext_metrics=args.ext_metrics,
                                     oracle_conventions=True)
    if args.perturb_boundary_metrics:
        perturb_boundary_metrics_coherent(
            ctx, float(args.perturb_boundary_metrics), N, NG)
    n, ng = ctx["n"], ctx["ng"]
    ak, bk, ptop, ks = set_eta_analytic(KM)
    ptop = float(ptop)
    print(f"grid n={n} ng={ng} km={KM} ptop={ptop} ks={ks}")

    orc_ic = load_oracle(args.ic_run, nh=args.nh)
    orc_1 = load_oracle(args.step_run, nh=args.nh)
    require_flat_orography(orc_ic)
    require_flat_orography(orc_1)
    orc_tr_ic = orc_tr_1 = None
    if args.tracers:
        orc_tr_ic = load_tracer_oracle(args.ic_run)
        orc_tr_1 = load_tracer_oracle(args.step_run)
        # THE DELP-AGREEMENT TRAP, CHECKED, NOT ASSUMED: a spatially
        # constant tracer is reproduced by ANY mass-consistent
        # transport, so its match certifies nothing about advection.
        # Print each tracer's IC range and label the constant ones.
        print("\noracle tracer IC ranges (per tile min..max; a constant "
              "tracer's score is VACUOUS as a transport test):")
        for nm in ADVECTED_TRACERS + INERT_TRACERS:
            lo = min(float(orc_tr_ic[t][nm].min()) for t in range(6))
            hi = max(float(orc_tr_ic[t][nm].max()) for t in range(6))
            tag = "VACUOUS (constant)" if lo == hi else "carries signal"
            print(f"  {nm:8s} [{lo:.6g}, {hi:.6g}]  -> {tag}")
        # dnats=1: rainwat must be UNTOUCHED by the step. If this fails
        # the dnats reading is wrong and the advected set is wrong too.
        for t in range(6):
            if not np.array_equal(orc_tr_ic[t][INERT_TRACERS[0]],
                                  orc_tr_1[t][INERT_TRACERS[0]]):
                raise SystemExit(
                    f"tile {t + 1}: rainwat changed across the step, but "
                    f"dnats=1 says it is inert -- the advected tracer "
                    f"set is misread; refusing to score.")
        print("  rainwat: bit-identical across the step on all 6 tiles "
              "(dnats=1 inertness CONFIRMED; not carried by the port)")

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

    tr_tend = {}
    if args.tracers:
        print("oracle TRACER tendency over one step (max |1step - IC|, "
              "per tile):")
        for nm in ADVECTED_TRACERS:
            per = [float(np.abs(orc_tr_1[t][nm] - orc_tr_ic[t][nm]).max())
                   for t in range(6)]
            tr_tend[nm] = per
            print(f"  {nm:8s} " + "  ".join(f"{x:10.4g}" for x in per))
        if max(tr_tend["sphum"]) == 0.0:
            raise SystemExit(
                "the oracle's sphum did not move over the step -- the "
                "tracer comparison would be vacuous everywhere. Wrong "
                "runs?")

    # ---------------- instrument control: the IC ----------------
    state, sphum6 = build_port_ic(
        ctx, ak, bk, nh=args.nh,
        zvir=(FV3_RVGAS / FV3_RDGAS - 1.0) if args.moist else 0.0)
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

    q = None
    p_tr_ic = None
    if args.tracers:
        # TRACER instrument control, under the SAME derived map: the
        # port's analytic sphum against the zerostep fv_tracer.res.
        # sphum is analytic in (lat, ak, bk) with no quad step beyond
        # the agrid latitudes, so the quad-geometry floor applies.
        q = build_port_tracer_ic(sphum6)
        p_tr_ic = tracer_window(q, ctx)
        worst_tr_ic = 0.0
        for pf in range(6):
            ot = perm[pf]
            for nm in ADVECTED_TRACERS:
                a, b = map_scalar_pair(p_tr_ic[pf][nm], orc_tr_ic[ot][nm],
                                       meta[pf][ot])
                worst_tr_ic = max(worst_tr_ic, rel(a, b))
        print(f"TRACER IC control: worst rel {worst_tr_ic:.3e} over "
              f"{ADVECTED_TRACERS} (sphum analytic vs zerostep restart; "
              f"liq_wat 0 == 0, vacuous).")
        if worst_tr_ic > IC_CONTROL_MAX_REL:
            raise SystemExit(
                f"TRACER INSTRUMENT CONTROL FAILED: IC rel "
                f"{worst_tr_ic:.3e} exceeds {IC_CONTROL_MAX_REL:.0e}; the "
                f"port's tracer IC does not reproduce the oracle's, so "
                f"the one-step tracer comparison would start from a "
                f"different field. Refusing to print it.")

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
    if q is None:
        # No --tracers: keep the historical all-zero passengers, which
        # exercise the same nr remap passes (and now the same nr
        # tracer_2d passes -- 0 is preserved exactly) without changing
        # this script's default output.
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
                # NOT hardcoded dry: under --moist this localisation
                # tool used to run a DRY remap while the deck and the
                # scored step were moist, i.e. it was wrong in exactly
                # the regime it exists for (GLM MINOR, job 9444414).
                r_vir=(FV3_RVGAS / FV3_RDGAS - 1.0) if args.moist else 0.0,
                sphum_index=(ADVECTED_TRACERS.index("sphum")
                             if args.moist else None),
                km=KM, n=n, ng=ng, kord_mt=KORD_MT,
                kord_tm=KORD_TM, kord_tr=KORD_TR, q=q[t],
                omga=np.zeros(field_shape("delp", n, ng, KM),
                              dtype=np.float64),
                # NOT hardcoded dry any more: under --moist this
                # localisation tool used to run a DRY remap while the
                # deck and the scored step were moist, i.e. it was wrong
                # in exactly the regime it exists for (GLM MINOR, job
                # 9444414).
                last_step=True, hydrostatic=not args.nh,
                adiabatic=(not args.moist) or (not args.nh), consv=0.0,
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
    step_fn = fv_dynamics_step
    if args.backend == "jax":
        step_fn = _make_jax_step(ctx)
    for _step in range(args.n_steps):
        out = step_fn(ctx, state, press, bdt=args.dt, km=KM,
                      k_split=args.k_split, n_split=args.n_split,
                      ptop=ptop, ak=ak, bk=bk, akap=FV3_KAPPA,
                      cp_air=FV3_CP_AIR, kord_mt=KORD_MT,
                      kord_tm=KORD_TM, kord_tr=KORD_TR, q=q,
                      hydrostatic=not args.nh,
                      # deck: a_imp=1., p_fac=0.05, kord_wz=9,
                      # use_logp=F, w_limiter=T (resolved namelist)
                      w_limiter=args.nh,
                      # zvir is NOT a namelist entry: atmosphere.F90:
                      # 156-161 derives it from `adiabatic`, which
                      # check_moist_deck asserts. sphum is index 0 of
                      # ADVECTED_TRACERS, matching build_port_tracer_ic.
                      **({"zvir": FV3_RVGAS / FV3_RDGAS - 1.0,
                          "sphum_index": ADVECTED_TRACERS.index("sphum")}
                         if args.moist else {}),
                      **({"consv_te": args.consv} if args.consv else {}))
        if out["pt_units"] != "K":
            raise SystemExit(f"driver left pt in {out['pt_units']}, not K")
    if args.physics == "held_suarez":
        # fv_phys + fv_update_phys, in place on the post-dynamics state
        apply_held_suarez_step(ctx, state, press, dt=args.dt, n=n, ng=ng, km=KM)
    p_1 = port_window(state, ctx)

    if args.moist or args.consv:
        # ---- THE MOIST-SIGNAL GATE (GLM M1, job 9444414) ------------
        # The headline residual is NOT evidence that the moist coupling
        # is live, and on the NH arm it is structurally blind to the
        # question. Arithmetic: the two decks' one-step pt tendencies
        # differ by ~3.3e-4 K, about 1e-6 of the pt peak, while the NH
        # gate floor is 6.6e-4 -- roughly 660x larger. A port whose
        # moist coupling is dead in an NH-only path (an r_vir dropped at
        # the remap, say) scores IDENTICALLY whether it runs moist or
        # dry. The hydrostatic arm is the opposite case: its 1.19e-9
        # floor sits ~1000x BELOW that signal, so there the headline
        # number does certify the moist path.
        #
        # So measure the moist RESPONSE and compare it to the oracle's
        # own, on a matched experiment: port(moist deck) - port(dry
        # deck) against oracle(moist deck) - oracle(dry deck). Both
        # sides are the same pair of decks, differing in the one flag.
        print("\n=== MOIST-SIGNAL GATE: the response, not the residual ===")
        if args.dry_twin_run is None or not os.path.isdir(args.dry_twin_run):
            raise SystemExit(
                "--moist needs the DRY twin of the step deck to measure "
                "the moist response, and an explicit --step-run gives no "
                "way to identify it. Re-run with the default decks, or "
                "extend this to take the twin explicitly.")
        from legoesm.core.fv3_native_dynamics import (
            p_var_nonhydrostatic as _p_var_nh,
        )
        orc_dry_1 = load_oracle(args.dry_twin_run, nh=args.nh)
        st_d = build_port_ic(ctx, ak, bk, nh=args.nh, zvir=0.0)[0]
        if args.nh:
            press_d = [_p_var_nh(f["delp"], f["delz"], f["pt"],
                                 ptop=ptop, akap=FV3_KAPPA,
                                 n=n, ng=ng, km=KM) for f in st_d]
        else:
            press_d = [p_var_hydrostatic(f["delp"], ptop=ptop,
                                         akap=FV3_KAPPA, n=n, ng=ng,
                                         km=KM) for f in st_d]
        q_d = [[np.zeros_like(a) for a in face] for face in q]
        step_fn(ctx, st_d, press_d, bdt=args.dt, km=KM,
                k_split=args.k_split, n_split=args.n_split, ptop=ptop,
                ak=ak, bk=bk, akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
                kord_mt=KORD_MT, kord_tm=KORD_TM, kord_tr=KORD_TR,
                q=q_d, hydrostatic=not args.nh, w_limiter=args.nh)
        p_dry_1 = port_window(st_d, ctx)

        # FIELD-LEVEL, not max-vs-max. Two different fields share a
        # maximum routinely, and the maxima need not even sit at the
        # same cell -- the same weakness codex flagged in an earlier
        # anti-vacuity gate. So build the response as a STATE and push
        # it through the SAME derived face map the residuals use, then
        # score it the same way.
        resp_p = [{f: p_1[pf][f] - p_dry_1[pf][f] for f in fields}
                  for pf in range(6)]
        resp_o = [{f: orc_1[t][f] - orc_dry_1[t][f] for f in fields}
                  for t in range(6)]
        # apply_map, NOT map_scalar_pair: u and v are STAGGERED
        # ((48,49) vs (49,48)) and a transposed face exchanges them.
        # map_scalar_pair is cell-centred only and raised a broadcast
        # error on the first run -- the staggering trap, again.
        # apply_map returns (pairs, wind_scale); the residual path uses
        # that scale for u/v, so this scores on identical terms.
        mapped = [apply_map(resp_p[pf], resp_o[perm[pf]], meta[pf][perm[pf]])
                  for pf in range(6)]
        # A RESPONSE BELOW THE PARITY FLOOR IS NOT A RESPONSE. At one
        # step the moist signal lives in pt (~3.2 K); u, v and delp move
        # only at rounding scale (~3e-13, ~3e-11) on BOTH sides, and
        # rel() on two uncorrelated noise fields of equal magnitude
        # returns exactly 2.0 -- which is what the first version of this
        # gate reported as a failure. That was comparing two zeros. So a
        # field/face is SCORED only where the oracle's own response
        # clears its own port-vs-oracle residual by a decade; everything
        # else is reported as carrying no resolvable signal.
        state_map = [apply_map(p_1[pf], orc_1[perm[pf]], meta[pf][perm[pf]])
                     for pf in range(6)]
        worst_rel, worst_f, any_signal, n_scored = 0.0, None, False, 0
        for f in fields:
            row_p, row_o, marks = [], [], []
            for pf in range(6):
                pairs_r, ws_r = mapped[pf]
                a, b = pairs_r[f]
                sa, sb = state_map[pf][0][f]
                resid = float(np.abs(sa - sb).max())
                pk_p, pk_o = float(np.abs(a).max()), float(np.abs(b).max())
                row_p.append(pk_p)
                row_o.append(pk_o)
                if pk_o <= MOIST_RESPONSE_SNR * resid:
                    marks.append(".")     # below the floor: not scored
                    continue
                marks.append("*")
                any_signal = any_signal or pk_p > 0.0
                n_scored += 1
                r = rel(a, b, ws_r if f in ("u", "v") else None)
                if r > worst_rel:
                    worst_rel, worst_f = r, f"{f} face{pf + 1}"
            print(f"  {f:5s} port response "
                  + "  ".join(f"{x:9.4g}" for x in row_p)
                  + "   scored: " + "".join(marks))
            print(f"        oracle       "
                  + "  ".join(f"{x:9.4g}" for x in row_o))
        if n_scored == 0:
            raise SystemExit(
                "MOIST-SIGNAL GATE FAILED: NO field/face has a moist "
                "response that clears its own parity residual, so this "
                "run cannot say whether the coupling is live at all.")
        if not any_signal:
            raise SystemExit(
                "MOIST-SIGNAL GATE FAILED: the port's moist and dry runs "
                "are IDENTICAL, so zvir reaches nothing in the step. The "
                "headline residual cannot see this on the NH arm.")
        if worst_rel > MOIST_RESPONSE_MAX_REL:
            raise SystemExit(
                f"MOIST-SIGNAL GATE FAILED: the port's moist response "
                f"differs from the oracle's by {worst_rel:.3e} at "
                f"{worst_f} (limit {MOIST_RESPONSE_MAX_REL:.0e}). The "
                f"port moves under zvir, but not the way the oracle "
                f"does.")
        print(f"MOIST-SIGNAL GATE PASSED: worst port-vs-oracle response "
              f"rel {worst_rel:.3e} at {worst_f} over {n_scored} scored "
              f"field/face pairs (limit {MOIST_RESPONSE_MAX_REL:.0e}; "
              f"pairs whose response sits under {MOIST_RESPONSE_SNR}x "
              f"their own residual are marked '.' above and not scored).")

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
    saved = {}
    for pf in range(6):
        ot = perm[pf]
        pairs, ws = apply_map(p_1[pf], orc_1[ot], meta[pf][ot])
        if args.save_fields:
            for f_, (a_, b_) in pairs.items():
                saved[f"port_f{pf+1}_{f_}"] = np.asarray(a_)
                saved[f"oracle_f{pf+1}_{f_}"] = np.asarray(b_)
            saved[f"perm_f{pf+1}"] = np.asarray(ot)
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

    if args.tracers:
        p_tr_1 = tracer_window(q, ctx)
        print("\nPORT's own one-step TRACER tendency vs the ORACLE's on "
              "the mapped tile:")
        for nm in ADVECTED_TRACERS:
            row_p = [float(np.abs(p_tr_1[pf][nm] - p_tr_ic[pf][nm]).max())
                     for pf in range(6)]
            row_o = [tr_tend[nm][perm[pf]] for pf in range(6)]
            print(f"  {nm:8s} port   "
                  + "  ".join(f"{x:10.4g}" for x in row_p))
            print(f"  {'':8s} oracle "
                  + "  ".join(f"{x:10.4g}" for x in row_o))

        print("\none-step TRACER residual, port vs oracle, each tracer "
              "on its OWN scale:")
        for pf in range(6):
            ot = perm[pf]
            row = res[f"face{pf+1}->tile{ot+1}"]
            for nm in ADVECTED_TRACERS:
                a, b = map_scalar_pair(p_tr_1[pf][nm], orc_tr_1[ot][nm],
                                       meta[pf][ot])
                r = rel(a, b)
                absd = float(np.abs(a - b).max())
                tnd = tr_tend[nm][ot]
                vac = (min(float(orc_tr_ic[t][nm].min()) for t in range(6))
                       == max(float(orc_tr_ic[t][nm].max())
                              for t in range(6)))
                row[nm] = {"rel": r, "max_abs_diff": absd,
                           "oracle_tendency": tnd,
                           "frac_of_tendency": (absd / tnd if tnd
                                                else float("nan")),
                           "vacuous_constant_ic": bool(vac)}
                worst_step = max(worst_step, r)
                print(f"  face {pf+1} -> tile {ot+1}  {nm:8s} "
                      f"rel={r:9.3e}  |d|max={absd:11.5g}  "
                      f"tendency={tnd:11.5g}  "
                      f"|d|/tend={row[nm]['frac_of_tendency']:9.3e}"
                      + ("  [VACUOUS: constant IC]" if vac else ""))

    if args.save_fields:
        saved["fields"] = np.asarray(fields)
        saved["nh"] = np.asarray(bool(args.nh))
        np.savez_compressed(args.save_fields, **saved)
        print(f"\nsaved mapped port/oracle planes -> {args.save_fields}")
    print(f"\nWORST one-step rel over all faces and fields: {worst_step:.4e}")
    print(f"IC control (same harness, same map): {worst:.4e}")
    print(f"amplification over one step: "
          f"{worst_step / max(worst, 1e-300):.3g}x")
    if args.perturb_boundary_metrics:
        # pre-registered decision rule (printed with the number so the
        # log is self-contained): a coherent boundary seed that
        # AMPLIFIES to the floor predicts the residual MOVES with eps
        # (>=2x at eps=1e-10 vs 1e-12); a residual pinned at the
        # unperturbed floor across both eps refutes seed amplification
        # FOR THE PERTURBED INPUTS.
        print(f"PERTURB DECISION RULE: eps={args.perturb_boundary_metrics:g} "
              f"WORST={worst_step:.4e}; PINNED if within ~10% of the "
              f"unperturbed floor at BOTH eps=1e-12 and 1e-10, MOVED "
              f"otherwise")
        # SCOPE (codex instr r2 H1 -- do not overreach): PINNED here
        # refutes amplification of seeds in the perturbed set only:
        # every gridstruct metric family (primitives + recomputed
        # deriveds + da_min scalars), f0, and the ectx dx6/dy6
        # snapshots.  NOT perturbed: ectx amat6 and the ext-vector
        # bases/corner operators (vlon4/vlat4/ew4/es4) -- those are
        # COORDINATE-derived; their equality to the oracle is
        # certified DIRECTLY (face-map coordinate control ~1e-16 here;
        # extchain oracle battery certificates), which bounds the seed
        # but not a hypothetical amplification of it.  A verdict of
        # 'composition/formulation difference' additionally rests on
        # those direct certificates.
        print("PERTURB SCOPE: metrics+deriveds+da_min+f0+ectx(dx6,dy6) "
              "perturbed; amat6/ext-vector bases coordinate-derived, "
              "certified directly, NOT perturbed")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"ic_worst_rel": worst, "step_worst_rel": worst_step,
                       "n_steps": args.n_steps,
                       "step_run": args.step_run,
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
