#!/usr/bin/env python
"""WHICH OPERATOR OWNS THE kt=1 FROZEN FORCING, and which of its statements.

Round 42 left the kt=1 break at the barotropic loop's INPUT: given NEMO's own
operands the loop reproduces NEMO's arithmetic, and what differs is the frozen
forcing it is handed -- ``zv_frc`` by max 8.414e-11 against NEMO's own rms
1.414e-06, almost entirely meridional.  This gate names the operator, then the
statement.

WHY kt=1 FROM REST IS THE CLEAN CASE.  NEMO builds the frozen forcing from the
depth mean of ``puu/pvv(:,:,:,Krhs)`` at ``dyn_spg`` entry
(``DINO_KT1_RANKDUMP dynspg_ts.f90:275-276``), weighted by the REST thickness ``e3v_3d`` and
divided by ``hv_0`` through ``r1_hv_0`` (``domain.f90:215``).  ``stpmlf.f90``
has run ``dyn_adv -> dyn_vor -> dyn_ldf -> dyn_hpg`` by then, and every one of
those but ``dyn_hpg`` is proportional to the velocity, which is identically
zero on a first step from rest.  So at kt=1 the meridional frozen forcing IS
the depth mean of ``hpg_sco`` and nothing else -- and section 1 MEASURES that
rather than asserting it.

SECTION 1, NEMO FROM NEMO (legoESM does not appear in it at all).  A literal
transcription of ``dynhpg.f90:350-454`` -- the surface statement, the
``jk = 2, jpkm1`` recurrence, the ``zuap`` slope term -- is fed NEMO's own
mesh (``gdept_0``, ``e3w_0``, ``e2v``, ``e3v_0``, the masks), NEMO's own
initial T/S, and a transcription of NEMO's S-EOS (``eosbn2.f90:357-369``,
``ln_seos = .true.``) at the depth ``gdept_3d*(1+r3t)``; its ``:273`` depth
mean must reproduce ``spg_dump_zv_frc`` from the rank-tagged record.  That one
row licenses everything below it: it proves the operator SET, the depth-mean
formula, ``r1_hv_0``, and the EOS at once.  If it fails, the preregistered
owner is wrong and the gate says so instead of scoring legoESM.

WHAT SECTION 1 CANNOT SEE (Rule 2): a compensating pair of errors inside the
transcription.  Section 3 is the mitigation -- it scores the SAME transcription
LEVEL BY LEVEL against legoESM, where a compensating pair would have to survive
35 independent rows.

THE RACED 3-D DUMPS PROVE ALMOST NOTHING, and an earlier version of this
header said they proved the operator set.  RETRACTED, by an independent claim
review and then by the record: ``stp_dump_03/04/05`` and the per-operator
``keg_/vor_/ldf_/zad_`` dumps are written by all 16 ranks to ONE untagged
filename with ``STATUS='REPLACE'``, so the surviving bytes are the last
writer's -- a rank that wrote nonzeros is overwritten by a later rank writing
zeros.  The tiles are not even the same size (14 ranks at ``jpi*jpj = 870``,
ranks 14 and 15 at 840) and the file is exactly ``870*jpkm1`` doubles, so it
cannot hold more than one rank's tile: their zeros cover at most one tile --
8.4% of the domain, measured -- and are not attributable to a cell.  They are printed as CORROBORATION
with that coverage stated.  The operator set is established by C1 instead, on
the rank-tagged global stitch.

PROVENANCE.  The record was produced by the config copy ``cfgs/DINO_KT1_RANKDUMP``,
so the line numbers cited here are ITS build: ``dynhpg.f90`` and ``eosbn2.f90``
are byte-identical to ``cfgs/DINO``'s, and ``dynspg_ts.f90`` differs only by
the rank-tagged filenames and three declarations, which shifts its line numbers
by three against the deck's.  ``ln_bt_fw = .FALSE.`` (the run's own
``output.namelist.dyn``), so the wind takes the AVERAGED branch ``:375``
``zztmp*(vtau_b + vtauV)``, not the forward branch ``:369``.

Usage (GPU only -- this card's XLA CPU compile crashes on this machine)::

    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
    python scripts/validate/ocean_fidelity/dino_1226/\\
        spg_kt1_frozen_forcing_hpg.py [--plant] [--plant-lego]

``--plant`` moves one wet cell of the NEMO-from-NEMO calibration by one ulp:
the calibration MUST then read DEBT and the gate MUST refuse to score anything
below it.  ``--plant-lego`` moves one wet cell of legoESM's own array, so every
legoESM row that is AT BAR must flip.  A plant that lands on a row already in
DEBT proves nothing, so neither plant is allowed to.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

RUN_KT1 = ("/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt1_rankdump")
RN_DT = 2700.0

#: NEMO phycst.f90:49 and the DINO namelist_cfg &nameos block, transcribed.
#: Every one of these is READ FROM THE RECORD's own namelist at run time by
#: ``_namelist_check`` below -- the literals here are what the transcription
#: uses and the check is what proves they are this record's.
GRAV = 9.80665
RHO0 = 1026.0
NAMEOS = dict(rn_T0=10.0, rn_S0=35.0, rn_a0=0.165, rn_b0=7.6554e-1,
              rn_lambda1=0.06, rn_lambda2=0.0, rn_mu1=1.4970e-4,
              rn_mu2=0.0, rn_nu=0.0)

#: The transcription's own resolution must come in UNDER these.  Measured
#: 4.1399e-20 absolute and 2.9273e-14 relative on 2026-09-11, so both carry
#: about 24x and 34x of margin.  They are a CEILING on the instrument, never a
#: tolerance granted to the model: every model row is judged against the
#: MEASURED floor, which is smaller.
CALIB_MAX = 1.0e-18
CALIB_REL = 1.0e-12

#: The plants' size.  Above the instrument's floor (2.9e-14 relative) and far
#: below the residual this round removed (5.9e-05), so a plant can neither
#: hide under the floor nor be mistaken for the defect.
PLANT_REL = 1.0e-9


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reader():
    return _load("_rankdump_hpg",
                 os.path.join(_HERE, "nemo_dino_kt1_rankdump",
                              "read_rankdump.py"))


def _headers(run_dir, rr):
    """Every rank's self-describing tile header, from the substep stream."""
    hdrs = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "substep_r*_s001.bin"))):
        rank = int(os.path.basename(p).split("_r")[1][:4])
        hdrs[rank] = rr.read_tile(p)["header"]
    if not hdrs:
        raise SystemExit(
            f"no substep_r*_s001.bin under {run_dir}; this gate needs the "
            "rank-tagged kt=1 record (nemo_dino_kt1_rankdump/run.sh)")
    return hdrs


def stitch_raw(run_dir, rr, hdrs, name, inner=True):
    """One of the oracle's own un-headered 2-D streams, per rank, stitched.

    Geometry comes from the SUBSTEP files' headers for the same ranks -- the
    record's own decomposition -- and a length that disagrees with it is fatal.
    ``zu_frc``/``zv_frc`` and the two increments are written over the INNER
    region (``ji = Nis0..Nie0``); the Coriolis trend over the full local tile.
    """
    shape = rr.haloless_shape(hdrs[0])
    field = np.full(shape, np.nan)
    for rank, h in sorted(hdrs.items()):
        p = os.path.join(run_dir, f"{name}_r{rank:04d}.bin")
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing; the record is incomplete")
        a = np.fromfile(p, dtype="<f8")
        gj0, gj1, gi0, gi1, j0, j1, i0, i1 = rr.tile_slices(h, shape)
        if inner:
            want = (j1 - j0) * (i1 - i0)
            if a.size != want:
                raise SystemExit(
                    f"{p}: {a.size} doubles but this rank's own substep "
                    f"header gives an inner region of {want}")
            field[gj0:gj1, gi0:gi1] = a.reshape(j1 - j0, i1 - i0)
        else:
            if a.size != h["jpi"] * h["jpj"]:
                raise SystemExit(
                    f"{p}: {a.size} doubles but its rank's header says "
                    f"jpi*jpj = {h['jpi'] * h['jpj']}")
            field[gj0:gj1, gi0:gi1] = a.reshape(
                h["jpj"], h["jpi"])[j0:j1, i0:i1]
    if not np.isfinite(field).all():
        raise SystemExit(f"{name}: the stitched field has holes")
    return field


def _namelist_check(run_dir):
    """Every constant this transcription hard-codes, read from the RECORD.

    Rule 10: a namelist value quoted from a comment is a claim.  These are the
    run's own ``namelist_cfg``/``namelist_ref``, and a disagreement is fatal --
    the transcription would silently be a different equation of state.
    """
    text = ""
    for nm in ("namelist_cfg", "namelist_ref"):
        p = os.path.join(run_dir, nm)
        if os.path.exists(p):
            text += "\n" + open(p).read()
    if not text:
        raise SystemExit(f"no namelist under {run_dir}")
    import re
    bad = []
    for key, want in NAMEOS.items():
        hits = [float(m.group(1).replace("d", "e").replace("D", "E"))
                for m in re.finditer(
                    rf"^\s*{key}\s*=\s*([-+0-9.eEdD]+)", text, re.M)]
        if not hits:
            bad.append(f"{key}: not in the record's namelists")
        elif want not in hits:
            bad.append(f"{key}: record has {hits}, transcription uses {want}")
    seos = re.findall(r"^\s*ln_seos\s*=\s*\.(\w+)\.", text, re.M)
    if "true" not in seos:
        bad.append(f"ln_seos: record has {seos}, transcription assumes true")
    if bad:
        raise SystemExit("REFUSED, the transcription is not this record's "
                         "equation of state:\n  " + "\n  ".join(bad))
    print(f"  namelist check: ln_seos=T and all {len(NAMEOS)} S-EOS "
          "coefficients read from the record's own namelists")


def nemo_seos_rhd(t_in, s_in, gdept_3d, r3t, tmask):
    """``eosbn2.f90:357-369``, ``CASE( np_seos )``, transcribed statement for
    statement.  Arrays are ``(jpk, jpj, jpi)``; the loop is ``jk = 1, jpkm1``,
    so the deepest level is left at its initialised zero."""
    c = NAMEOS
    zt = t_in - c["rn_T0"]
    zs = s_in - c["rn_S0"]
    zh = gdept_3d * (1.0 + r3t)
    zn = (-c["rn_a0"] * (1.0 + 0.5 * c["rn_lambda1"] * zt + c["rn_mu1"] * zh)
          * zt
          + c["rn_b0"] * (1.0 - 0.5 * c["rn_lambda2"] * zs - c["rn_mu2"] * zh)
          * zs
          - c["rn_nu"] * zt * zs)
    rhd = zn * (1.0 / RHO0) * tmask
    rhd[-1] = 0.0
    return rhd


def nemo_hpg_sco(rhd, e3w_3d, gdept_3d, r3t, ssh, r1_e1u, r1_e2v):
    """``dynhpg.f90:350-454`` transcribed.  Returns ``(du, dv)``, the pure
    ``hpg_sco`` increment to ``puu/pvv(:,:,:,Krhs)``, ``(jpk, jpj, jpi)``.

    The associations are NEMO's, not algebra's: the two-level sum
    ``(rhd(jk) + rhd(jk-1))`` is formed FIRST, multiplied by the stretched
    ``e3w``, and the EAST-minus-WEST (NORTH-minus-SOUTH) difference is
    accumulated level by level -- not a per-column integral differenced once.
    """
    nk, ny, nx = rhd.shape
    zcoef0 = -GRAV * 0.5
    stretch = 1.0 + r3t
    gdept_z0 = gdept_3d * stretch - ssh

    def di(a):                       # a(ji+1) - a(ji), periodic in i
        return np.roll(a, -1, axis=-1) - a

    def dj(a):                       # a(jj+1) - a(jj); last row has no j+1
        out = np.full_like(a, np.nan)
        out[:-1] = a[1:] - a[:-1]
        return out

    def si(a):                       # a(ji+1) + a(ji)
        return np.roll(a, -1, axis=-1) + a

    def sj(a):
        out = np.full_like(a, np.nan)
        out[:-1] = a[1:] + a[:-1]
        return out

    du = np.zeros((nk, ny, nx))
    dv = np.zeros((nk, ny, nx))
    e3w_s = e3w_3d * stretch
    zhpi = zcoef0 * r1_e1u * di(e3w_s[0] * rhd[0])
    zhpj = zcoef0 * r1_e2v * dj(e3w_s[0] * rhd[0])
    du[0] = zhpi - zcoef0 * si(rhd[0]) * di(gdept_z0[0]) * r1_e1u
    dv[0] = zhpj - zcoef0 * sj(rhd[0]) * dj(gdept_z0[0]) * r1_e2v
    for k in range(1, nk - 1):                       # jk = 2, jpkm1
        pair = rhd[k] + rhd[k - 1]
        zhpi = zhpi + zcoef0 * r1_e1u * di(e3w_s[k] * pair)
        zhpj = zhpj + zcoef0 * r1_e2v * dj(e3w_s[k] * pair)
        du[k] = zhpi - zcoef0 * si(rhd[k]) * di(gdept_z0[k]) * r1_e1u
        dv[k] = zhpj - zcoef0 * sj(rhd[k]) * dj(gdept_z0[k]) * r1_e2v
    return du, dv


def nemo_depth_mean(d3, e3_0, mask3, r1_h0):
    """The REST-thickness depth mean, over ALL ``jpk`` levels, divided by
    ``h_0`` through the precomputed reciprocal (``dynspg_ts.f90:276`` in the
    record's own build; ``r1_hv_0`` from ``domain.f90:215``)."""
    return (e3_0 * d3 * mask3).sum(axis=0) * r1_h0


def _mesh(run_dir):
    """NEMO's own mesh, stitched from the per-rank ``mesh_mask`` tiles."""
    from rebuild_nemo_restart import rebuild
    want = ["e3w_0", "e3t_0", "gdept_0", "e1u", "e2v", "tmask", "umask",
            "vmask", "e3u_0", "e3v_0", "umaskutil", "vmaskutil"]
    m = rebuild(os.path.join(run_dir, "mesh_mask_*.nc"), want)
    missing = [k for k in want if k not in m]
    if missing:
        raise SystemExit(f"mesh_mask is missing {missing}")
    for k, v in m.items():
        if not np.isfinite(v).all():
            raise SystemExit(f"mesh field {k} has holes after stitching")
    return m


def nemo_side(run_dir, tab, plant=False):
    """Section 1 + 2 -- NEMO from NEMO.  Returns the transcription and the
    oracle's own frozen forcing, or raises if the calibration fails."""
    rr = _reader()
    hdrs = _headers(run_dir, rr)
    _namelist_check(run_dir)
    m = _mesh(run_dir)
    tmask, vmask, umask = m["tmask"], m["vmask"], m["umask"]
    nk, ny, nx = tmask.shape

    print()
    print("2. THE OPERATOR SET at kt=1")
    # THE RACED 3-D DUMPS PROVE MUCH LESS THAN THEY LOOK LIKE THEY DO, and an
    # earlier version of this section said they proved the operator set.
    # RETRACTED.  stpmlf.F90's stp_dump_krhs opens ONE untagged filename with
    # STATUS='REPLACE' on EVERY rank, so the surviving bytes are whichever
    # rank finished last -- a rank that wrote nonzeros is silently overwritten
    # by a later rank writing zeros.  Worse, the tiles are not even the same
    # size here (14 ranks at jpi*jpj = 870, ranks 14 and 15 at 840), and the
    # file is exactly 870*jpkm1 doubles, so it cannot hold more than one rank's
    # tile.  Zeros in it therefore cover ONE tile's worth of cells at best and
    # are not attributable to a cell at all.
    #
    # What actually establishes the operator set is C1 below, on the
    # rank-tagged GLOBAL stitch: the transcription of hpg_sco ALONE reproduces
    # NEMO's own zv_frc to 2.9e-14 of its rms over every wet v-cell, which no
    # nonzero advection, vorticity or lateral-mixing contribution could
    # survive.  These rows are printed as CORROBORATION with their coverage
    # stated, never as the measurement.
    shape = rr.haloless_shape(hdrs[0])
    tile = max(h["jpi"] * h["jpj"] for h in hdrs.values())
    cover = tile / float(shape[0] * shape[1])
    nonzero_seen = []
    for stem in ("stp_dump_03_dynadv_kt00000001",
                 "stp_dump_04_dynvor_kt00000001",
                 "stp_dump_05_dynldf_kt00000001", "keg_dump", "vor_dump",
                 "ldf_dump", "zad_dump"):
        for c in ("du", "dv"):
            p_dump = os.path.join(run_dir, f"{stem}_{c}.bin")
            if not os.path.exists(p_dump):
                nonzero_seen.append(f"{stem}_{c}: ABSENT")
                continue
            a = np.fromfile(p_dump, dtype="<f8")
            nz = int((a != 0).sum())
            if nz:
                nonzero_seen.append(f"{stem}_{c}: {nz} of {a.size} nonzero")
    print(f"  the adv / vor / ldf / zad 3-D dumps are RACED (one untagged "
          f"filename, 16 writers, tiles of {sorted({h['jpi'] * h['jpj'] for h in hdrs.values()})} "
          f"points): they can speak for at most {cover:.1%} of the domain")
    if nonzero_seen:
        print("  CORROBORATION WITHHELD -- not all of them are zero: "
              + "; ".join(nonzero_seen))
    else:
        print("  corroboration only: every one of them is bitwise zero, which "
              "is consistent with the operator set but does not establish it")

    fro = {n: stitch_raw(run_dir, rr, hdrs, f"spg_dump_{n}")
           for n in ("zu_frc", "zv_frc")}
    inc = {}
    for nm, stem, innr in (("drag u", "drg_dump_zu_frc_inc", True),
                           ("drag v", "drg_dump_zv_frc_inc", True),
                           ("wind u", "wnd_dump_zu_frc_inc", True),
                           ("wind v", "wnd_dump_zv_frc_inc", True),
                           ("cor2d u", "cor2d_dump_zu_trd_substep1", False),
                           ("cor2d v", "cor2d_dump_zv_trd_substep1", False)):
        a = stitch_raw(run_dir, rr, hdrs, stem, inner=innr)
        inc[nm] = a
        print(f"  {nm:8s} increment to the frozen forcing: max|.| "
              f"{np.abs(a).max():.4e}  nonzero cells {int((a != 0).sum())}")

    print()
    print("1. CALIBRATION -- NEMO's zv_frc rebuilt from NEMO's own operands")
    from legoesm.ocean.fidelity.nemo_dino_mesh import nemo_dino_istate, nemo_dino_mesh
    g = nemo_dino_mesh()
    t_ic, s_ic = nemo_dino_istate(g)
    t_3d = np.moveaxis(np.asarray(t_ic, dtype=np.float64), -1, 0)
    s_3d = np.moveaxis(np.asarray(s_ic, dtype=np.float64), -1, 0)

    # r3t and ssh at kt=1.  The r3t dump is raced; zero is the only claim.
    r3t_raw = np.fromfile(
        os.path.join(run_dir, "r3c_dump_r3t_kt00000001.bin"), dtype="<f8")
    if int((r3t_raw != 0).sum()):
        raise SystemExit("r3t is not zero at kt=1; the qco stretch is live "
                         "and this from-rest simplification does not hold")
    print(f"  r3t at kt=1: bitwise zero over {r3t_raw.size} values, so the "
          "qco stretch is 1 and gdept_z0 is the fixed ladder")
    r3t = np.zeros((ny, nx))
    ssh = np.zeros((ny, nx))

    rhd = nemo_seos_rhd(t_3d, s_3d, m["gdept_0"], r3t, tmask)
    du, dv = nemo_hpg_sco(rhd, m["e3w_0"], m["gdept_0"], r3t, ssh,
                          1.0 / m["e1u"], 1.0 / m["e2v"])
    hv_0 = (m["e3v_0"] * vmask).sum(axis=0)
    hu_0 = (m["e3u_0"] * umask).sum(axis=0)
    r1_hv_0 = m["vmaskutil"] / (hv_0 + 1.0 - m["vmaskutil"])
    r1_hu_0 = m["umaskutil"] / (hu_0 + 1.0 - m["umaskutil"])
    zv_built = nemo_depth_mean(dv, m["e3v_0"], vmask, r1_hv_0)
    zu_built = nemo_depth_mean(du, m["e3u_0"], umask, r1_hu_0) + inc["wind u"]

    # The mask is GEOMETRY, never "where the numbers are finite": dropping a
    # non-finite cell here would hide it from Table._prep, whose job is to call
    # such a row UNMEASURED.  Only the last row is excluded, and for a reason
    # that is not about values -- it has no j+1 neighbour on the haloless
    # global domain.
    vwet = (m["vmaskutil"] > 0.5)
    vwet[-1] = False
    uwet = (m["umaskutil"] > 0.5)
    rms_v = float(np.sqrt(np.mean(fro["zv_frc"][vwet] ** 2)))
    if plant:
        q = np.argwhere(vwet & (np.abs(zv_built) > 0.1 * rms_v))
        if not q.size:
            raise SystemExit("nothing large enough to plant into")
        zv_built = zv_built.copy()
        # A ONE-ULP PLANT HERE IS DEAD, and that is worth stating rather than
        # discovering: ``nextafter`` on a value near 1e-06 moves it by about
        # 2e-22, while the instrument's own residual is 4e-20 -- so the plant
        # would not change max|d| at all and the calibration would stay green
        # while "proving" it can fail.  PLANT_REL is chosen to sit ABOVE the
        # floor and FAR BELOW the residual this round measured (5.9e-05), so a
        # plant that passes could not be mistaken for the defect either.
        zv_built[q[0][0], q[0][1]] *= (1.0 + PLANT_REL)
        print(f"  PLANT: one wet cell of the rebuilt zv_frc scaled by "
              f"1 + {PLANT_REL:.0e} (a 1-ulp plant is BELOW this "
              "instrument's own floor and would prove nothing)")
    row = tab.calibrate("C1 zv_frc rebuilt from hpg_sco", zv_built,
                        fro["zv_frc"], vwet)
    # C1b is WEAK and is labelled so rather than counted as a second
    # independent confirmation: DINO's initial T/S is zonally uniform, so the
    # transcription's zonal HPG is identically 0.0 by construction and this
    # row is close to "NEMO's own wind increment equals itself".  What it does
    # establish is that the zonal depth mean really is exactly zero and that
    # the wind is the only other term in zu_frc.
    zu_row = tab.score("C1b zu_frc rebuilt (hpg + wind)", zu_built,
                       fro["zu_frc"], uwet)
    if zu_row is not None:
        zu_row["name"] += "  [weak]"
    return dict(rr=rr, hdrs=hdrs, mesh=m, fro=fro, inc=inc, rhd=rhd,
                du=du, dv=dv, vwet=vwet, uwet=uwet, rms_v=rms_v,
                c1=row, zv_built=zv_built)


def _legacy_divisor_grid(grid):
    """The SAME grid with ``dy_v`` replaced by what the operator used to
    rebuild: ``0.5*(dy_T[j] + dy_T[j-1])``, the T-point-to-T-point spacing.

    This is the one-variable arm for section 5.  Nothing else about the card
    changes and the production operator is the one that runs -- so the arm
    measures the DIVISOR and only the divisor, which is what a before/after
    quoted from two different commits could never claim.
    """
    import jax.numpy as jnp
    dy_h = jnp.asarray(grid.dy) * 0.5
    dyp = jnp.pad(dy_h, (1, 1), mode="edge")
    dy_v_1d = 0.5 * (dyp[1:] + dyp[:-1])
    return grid._replace(
        dy_v=jnp.broadcast_to(dy_v_1d[:, None], grid.dy_v.shape))


def lego_side(nemo, tab, plant_lego=False, legacy_divisor=False):
    """Sections 3 and 4 -- legoESM's own production arrays.

    The pressure gradient comes from the production stage functions the model
    itself calls (``_bc_geometry_and_density`` then
    ``_bc_ke_and_pressure_gradients``) with the card's own grid, coordinate and
    config and the card's own initial state -- no numerical selector is
    introduced and nothing is re-derived.  ``u = v = 0`` there is the card's
    own rest state, asserted below, which is what makes the fused KE+PGF
    diagnostic the PURE pressure-gradient term.
    """
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.experiments import dino as dm

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    st = dm.dino_lat_lon_state(grid, z, cfg)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    print()
    print("3. legoESM's pressure gradient, from the card's own production "
          "stages")
    print(f"  card: pgf_scheme={getattr(mc, 'pgf_scheme', None)!r} "
          f"quadrature={getattr(mc, 'pgf_quadrature', None)!r} "
          f"eos={getattr(mc, 'eos', None)!r} "
          f"eos_depth={getattr(mc, 'eos_depth', None)!r} "
          f"g={mc.g!r} rho_0={mc.rho_0!r}")
    for nm, a in (("eta", st.eta.data), ("u", st.u.data), ("v", st.v.data)):
        mx = float(np.abs(np.asarray(a)).max())
        if mx != 0.0:
            raise SystemExit(
                f"the card's initial {nm} is not identically zero (max {mx}); "
                "this gate's whole premise is a REST start, and with a nonzero "
                "velocity the KE gradient contaminates the PGF row")
    if legacy_divisor:
        grid = _legacy_divisor_grid(grid)
        print("  ARM: grid.dy_v replaced by the T-point-to-T-point spacing "
              "the operator used to rebuild -- one variable, production "
              "operator unchanged")
    model = LatLonCGridOceanModel(grid, z, mc)
    mask = getattr(model, "land_mask", None)
    if mask is None:
        mask = jnp.asarray(np.asarray(z.is_active).any(axis=-1))
    h_col = jnp.sum(jnp.asarray(z.h_partial), axis=-1)
    with jax.disable_jit():
        _jac, _h, rho_prime, p_prime_filled = pe._bc_geometry_and_density(
            st.eta.data, h_col, z, mc, st.T.data, st.S.data, mask, grid,
            mc.rho_0, mc.g)
        _kx, dp_dx, _ky, dp_dy = pe._bc_ke_and_pressure_gradients(
            jnp.zeros_like(st.u.data), jnp.zeros_like(st.v.data),
            p_prime_filled, rho_prime, grid, mc, z, st.eta.data, h_col,
            mc.g, mask)
    for nm, a in (("dp_dy", dp_dy), ("rho_prime", rho_prime)):
        if np.asarray(a).dtype != np.float64:
            raise SystemExit(f"{nm} is {np.asarray(a).dtype}, not float64; "
                             "Rule 1c -- an oracle comparison runs fp64")
    # legoESM v-face f sits between cells f-1 and f, so it is NEMO's v-point
    # f-1: drop face 0.  The same slice the barotropic ladder's R1 row uses.
    dv_lego = np.moveaxis(-np.asarray(dp_dy) / mc.rho_0, -1, 0)[:, 1:, :]
    dv_plain, rp_plain = None, None
    if plant_lego:
        # A STATUS FLIP IS NOT AVAILABLE HERE and pretending otherwise is what
        # the first version did: EVERY legoESM row is already DEBT (R1 sits at
        # 402x the floor), so "the row must go DEBT" is unfalsifiable, and the
        # plant touched only dv_lego while R-rho is built from rho_prime and
        # could not move at all.  An independent review found both.
        #
        # So the control is MOVEMENT, measured in ONE run: keep the unplanted
        # arrays, perturb copies of BOTH, and require every scored legoESM row
        # to move.  A row that does not is PLANT-DEAD and the gate fails.
        q = np.argwhere(nemo["vwet"])
        j, i = int(q[0][0]), int(q[0][1])
        dv_plain, rp_plain = dv_lego, np.asarray(rho_prime)
        dv_lego = dv_lego.copy()
        dv_lego[:, j, i] *= (1.0 + PLANT_REL)
        rho_prime = np.asarray(rho_prime).copy()
        rho_prime[j, i, :] *= (1.0 + PLANT_REL)
        print(f"  PLANT: legoESM's PGF column ({j},{i}) AND its density there "
              f"scaled by 1 + {PLANT_REL:.0e}; one ulp would be 2e-22, three "
              "orders below this instrument's floor")

    rp = np.moveaxis(np.asarray(rho_prime), -1, 0) / mc.rho_0
    wet3 = nemo["mesh"]["tmask"] > 0.5
    # ``[floor xfer]``: the floor was measured on a 36-term Fortran SUM and
    # these two rows are POINTWISE, so it is roughly 100x loose for them.  It
    # cannot manufacture a false pass here -- both read DEBT with it and would
    # read DEBT without it -- but a row just above it is not a finding, and the
    # label is what stops the next reader treating it as one.
    tab.score("R-rho  rhd (lego rho'/rho0 vs NEMO) [floor xfer]", rp,
              nemo["rhd"], wet3)

    m = nemo["mesh"]
    # THE DISCRIMINATOR FOR WHAT IS LEFT.  After the divisor the R1 row still
    # sits well above the instrument floor, so a second term exists.  Feed the
    # SAME transcription legoESM's OWN density instead of NEMO's and compare to
    # legoESM's pressure gradient: if that closes, every remaining statement of
    # hpg_sco agrees and the residual is the DENSITY, not the operator.
    dv_from_lego_rho = nemo_hpg_sco(
        rp, m["e3w_0"], m["gdept_0"], np.zeros(rp.shape[1:]),
        np.zeros(rp.shape[1:]), 1.0 / m["e1u"], 1.0 / m["e2v"])[1]
    vw_all = (m["vmask"] > 0.5)
    vw_all[:, -1, :] = False
    hpg_row = tab.score("R-hpg  lego PGF vs hpg_sco(lego rho) [floor xfer]",
                        dv_lego, dv_from_lego_rho, vw_all)
    hv_0 = (m["e3v_0"] * m["vmask"]).sum(axis=0)
    r1_hv_0 = m["vmaskutil"] / (hv_0 + 1.0 - m["vmaskutil"])
    zv_lego = nemo_depth_mean(dv_lego, m["e3v_0"], m["vmask"], r1_hv_0)
    r1_row = tab.score("R1 zv_frc  legoESM vs NEMO", zv_lego,
                       nemo["fro"]["zv_frc"], nemo["vwet"])
    if hpg_row and r1_row and hpg_row["rel"] < 0.1 * r1_row["rel"]:
        tab.note(
            f"R-hpg ({hpg_row['rel']:.2e}) is {r1_row['rel'] / hpg_row['rel']:.0f}x "
            f"BELOW R1 ({r1_row['rel']:.2e}): given the same density every "
            "remaining statement of hpg_sco agrees, so what R1 still carries "
            "is the DENSITY (R-rho), amplified by the pressure gradient's own "
            "cancellation")

    # THE PLANT'S OWN CONTROL, in this one run: every scored legoESM row must
    # MOVE.  Status cannot be the control here -- all three rows are already
    # DEBT -- so a row whose number does not change is reported PLANT-DEAD and
    # the gate fails.
    if dv_plain is not None:
        rp0 = np.moveaxis(rp_plain, -1, 0) / mc.rho_0
        dv0 = nemo_hpg_sco(rp0, m["e3w_0"], m["gdept_0"],
                           np.zeros(rp0.shape[1:]), np.zeros(rp0.shape[1:]),
                           1.0 / m["e1u"], 1.0 / m["e2v"])[1]
        zv0 = nemo_depth_mean(dv_plain, m["e3v_0"], m["vmask"], r1_hv_0)
        print()
        print("PLANT CONTROL -- every scored legoESM row must MOVE")
        dead = []
        for name, a0, b0, a1, b1, msk in (
                ("R-rho", rp0, nemo["rhd"], rp, nemo["rhd"], wet3),
                ("R-hpg", dv_plain, dv0, dv_lego, dv_from_lego_rho, vw_all),
                ("R1", zv0, nemo["fro"]["zv_frc"], zv_lego,
                 nemo["fro"]["zv_frc"], nemo["vwet"])):
            before = float(np.abs((a0 - b0)[msk]).max())
            after = float(np.abs((a1 - b1)[msk]).max())
            moved = after != before
            print(f"  {name:6s} max|d| {before:.4e} -> {after:.4e}  "
                  f"{'MOVED' if moved else 'PLANT-DEAD'}")
            if not moved:
                dead.append(name)
        if dead:
            tab.note(f"PLANT-DEAD rows {dead}: the plant cannot reach them, "
                     "so their AT BAR/DEBT verdict is not underwritten")
            tab.fail = True

    print()
    print("4. THE STATEMENT WALK -- legoESM's PGF against the transcription, "
          "level by level")
    print(f"  {'k':>3s}{'cells !=':>10s}{'max|d|':>13s}"
          f"{'rms(NEMO)':>13s}{'max|d|/rms':>13s}")
    vw = m["vmask"] > 0.5
    worst = None
    for k in range(dv_lego.shape[0]):
        sel = vw[k].copy()
        sel[-1] = False
        if not sel.any():
            continue
        a, b = dv_lego[k][sel], nemo["dv"][k][sel]
        ne = int((a != b).sum())
        rms = float(np.sqrt(np.mean(b ** 2)))
        rel = float(np.abs(a - b).max()) / rms if rms else float("nan")
        if k % 5 == 0 or k == dv_lego.shape[0] - 2:
            print(f"  {k:3d}{ne:>10d}{np.abs(a - b).max():13.4e}"
                  f"{rms:13.4e}{rel:13.4e}")
        if worst is None or rel > worst[1]:
            worst = (k, rel)
    print(f"  worst level k={worst[0]} at max|d|/rms {worst[1]:.4e}")
    return dict(dv_lego=dv_lego, zv_lego=zv_lego, grid=grid, worst=worst,
                legacy_divisor=legacy_divisor)


def metric_attribution(nemo, lego, tab):
    """Section 5 -- the divisor, which is the statement the walk lands on.

    ``gradient_y_cgrid``'s non-tripolar branch does not divide by the grid's
    own v-point metric.  It rebuilds one from ``grid.dy`` (= twice the T-cell
    height) as ``0.5*(dy_T[j] + dy_T[j-1])`` -- the finite-difference spacing
    between two cell centres.  NEMO's ``e2v`` on this card is the ISOTROPIC
    scale factor ``ra*rad*COS(rad*gphiv)*rn_e1_deg`` (``usrdef_hgr.F90:118``)
    evaluated AT the v-face latitude, and the card already carries it in
    ``grid.dy_v``.  The two differ, and the row below is what says the whole
    PGF residual is that difference and nothing else.
    """
    grid = lego["grid"]
    dy_h = np.asarray(grid.dy) * 0.5
    dyp = np.pad(dy_h, (1, 1), mode="edge")
    dy_v_operator = 0.5 * (dyp[1:] + dyp[:-1])[1:]        # NEMO v-point f-1
    e2v = nemo["mesh"]["e2v"]
    dy_v_grid = np.asarray(grid.dy_v)[1:, :]
    print()
    print("5. THE DIVISOR")
    print(f"  grid.dy_v            vs NEMO e2v: max|rel| "
          f"{np.abs(dy_v_grid[nemo['vwet']] / e2v[nemo['vwet']] - 1).max():.4e}")
    print(f"  gradient_y's divisor vs NEMO e2v: max|rel| "
          f"{np.abs(dy_v_operator[:, None] / e2v - 1)[nemo['vwet']].max():.4e}")
    # The attribution: at the SURFACE level ``hpg_sco`` is ONE statement, so
    # if the per-cell ratio legoESM/NEMO equals ``e2v/(the old divisor)`` then
    # the divisor is the whole residual and nothing else in the operator
    # contributed.  Printed only in the arm that actually takes the old
    # divisor -- after the fix the ratio is 1 and the check would be
    # meaningless.
    sel = ((nemo["mesh"]["vmask"][0] > 0.5) & nemo["vwet"]
           & (nemo["dv"][0] != 0.0))
    got = lego["dv_lego"][0][sel] / nemo["dv"][0][sel]
    want = np.broadcast_to((e2v / dy_v_operator[:, None]), e2v.shape)[sel]
    if lego["legacy_divisor"]:
        d = float(np.abs(got - want).max())
        print(f"  ARM, k=0 PGF ratio legoESM/NEMO against e2v/(old divisor): "
              f"max|d| {d:.4e} over {int(sel.sum())} cells")
        tab.note(f"in the legacy-divisor arm the k=0 ratio IS that divisor to "
                 f"{d:.1e}, so the residual is the metric and not the "
                 "quadrature, the EOS or the stencil")
    else:
        print(f"  k=0 PGF ratio legoESM/NEMO: max|ratio - 1| "
              f"{float(np.abs(got - 1.0).max()):.4e} "
              f"(the attribution check runs in --legacy-divisor)")


class Table:
    """AT BAR / DEBT / UNMEASURED against a MEASURED instrument resolution.

    Rule 1b puts the bar in the gate rather than in my judgment, and Rule 3
    says a residual means nothing until the instrument's own noise floor is
    known.  Both apply here, and they pull in different directions, so the
    structure is explicit:

    * ``calibrate`` scores the numpy transcription of ``hpg_sco`` against
      NEMO's own ``spg_dump_zv_frc``.  That comparison CANNOT reach zero -- the
      oracle's ``:273`` depth mean is a gfortran ``SUM`` intrinsic compiled at
      ``-O3 -funroll-all-loops``, whose summation order numpy does not
      reproduce -- so its residual is the instrument's RESOLUTION and it is
      gated against a PINNED bound (``CALIB_MAX``), not against zero.  What
      would reach zero is a rank-tagged 3-D ``dynhpg`` dump, which this record
      does not have (its ``hpg_dump_*`` files are written by all 16 ranks to
      one filename); with it there would be no transcription in the middle.
    * every other row is then AT BAR only if it is at or below that measured
      floor, and DEBT otherwise, with ``max|d|/floor`` printed so a row sitting
      just above the floor cannot be read as a finding.

    The floor is never widened to make a row pass: ``CALIB_MAX`` is a constant
    in this file and the calibration must come in UNDER it.
    """

    def __init__(self):
        self.rows = []
        self.notes = []
        self.fail = False
        self.floor = None

    def _prep(self, name, built, oracle, mask):
        b = np.asarray(built, dtype=np.float64)[mask]
        o = np.asarray(oracle, dtype=np.float64)[mask]
        if b.size == 0:
            self.fail = True
            self.rows.append(dict(name=name, n=0, status="UNMEASURED",
                                  reason="the mask selected no cells"))
            return None
        finite = np.isfinite(b) & np.isfinite(o)
        if not finite.all():
            # A non-finite cell counts UNEQUAL and is never filtered away: a
            # row that was half NaN once printed AT BAR on this campaign.
            self.fail = True
            self.rows.append(dict(
                name=name, n=int(b.size), status="UNMEASURED",
                reason=f"{int((~finite).sum())} of {b.size} compared cells "
                       "are not finite"))
            return None
        return b, o

    def calibrate(self, name, built, oracle, mask):
        """The instrument's own resolution, as a RELATIVE number.

        Absolute floors do not transfer between rows: a density anomaly of rms
        8e-04 and a forcing of rms 1e-06 cannot share one.  So the floor the
        rows below are judged against is ``max|d| / rms(oracle)``, measured
        here on the one comparison where both sides are NEMO.
        """
        got = self._prep(name, built, oracle, mask)
        if got is None:
            return None
        b, o = got
        d = float(np.abs(b - o).max())
        rms = float(np.sqrt(np.mean(o ** 2)))
        rel = d / rms if rms else float("inf")
        ok = (d <= CALIB_MAX) and (rel <= CALIB_REL)
        self.floor = rel
        row = dict(name=name, n=int(b.size), cells=int((b != o).sum()),
                   maxabs=d, rms=rms, rel=rel,
                   status="CALIBRATED" if ok else "REFUSED")
        if not ok:
            self.fail = True
        self.rows.append(row)
        return row

    def score(self, name, built, oracle, mask):
        got = self._prep(name, built, oracle, mask)
        if got is None:
            return None
        b, o = got
        d = float(np.abs(b - o).max())
        if self.floor is None:
            raise SystemExit("score() before calibrate(): the floor a row is "
                             "judged against would be undefined")
        rms = float(np.sqrt(np.mean(o ** 2)))
        rel = d / rms if rms else float("inf")
        at_bar = rel <= self.floor
        row = dict(name=name, n=int(b.size), cells=int((b != o).sum()),
                   maxabs=d, rms=rms, rel=rel,
                   status="AT BAR" if at_bar else "DEBT")
        if not at_bar:
            self.fail = True
        self.rows.append(row)
        return row

    def note(self, text):
        self.notes.append(text)

    def render(self):
        print()
        print(f"{'row':38s}{'cells !=':>10s}{'max|d|':>13s}"
              f"{'rms(NEMO)':>13s}{'max|d|/rms':>13s}  status")
        if self.floor is not None:
            print(f"{'':38s}{'':>10s}{'':>13s}{'':>13s}"
                  f"{'floor ' + format(self.floor, '.2e'):>13s}")
        for r in self.rows:
            if r["status"] == "UNMEASURED":
                print(f"{r['name']:38s}{'-':>10s}{'-':>13s}{'-':>13s}"
                      f"{'-':>13s}  UNMEASURED")
                print(f"{'':38s}  needs: {r['reason']}")
                continue
            print(f"{r['name']:38s}{r['cells']:>10d}{r['maxabs']:13.4e}"
                  f"{r['rms']:13.4e}{r['rel']:13.4e}  {r['status']}")
        for n in self.notes:
            print(f"  note: {n}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=RUN_KT1)
    ap.add_argument("--plant", action="store_true",
                    help="move one wet cell of the NEMO-from-NEMO calibration "
                         "by 1 ulp; it MUST then read DEBT")
    ap.add_argument("--legacy-divisor", action="store_true",
                    help="one-variable arm: hand the production operator the "
                         "T-point-to-T-point spacing instead of the grid's "
                         "own v-point metric")
    ap.add_argument("--plant-lego", action="store_true",
                    help="move one wet column of legoESM's PGF by 1 ulp; "
                         "every legoESM row MUST then read DEBT")
    a = ap.parse_args()
    print("kt=1 FROZEN FORCING -- which operator, and which statement")
    print(f"record: {a.run_dir}")
    for k in ("JAX_ENABLE_X64", "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES"):
        print(f"  {k} = {os.environ.get(k)!r}")
    tab = Table()
    nemo = nemo_side(a.run_dir, tab, plant=a.plant)
    if nemo["c1"] is None or nemo["c1"]["status"] != "CALIBRATED":
        tab.note("the NEMO-from-NEMO calibration is not AT BAR, so nothing "
                 "below it would be interpretable and legoESM is NOT scored")
        tab.render()
        print("\nRESULT: REFUSED (calibration)")
        return 1
    lego = lego_side(nemo, tab, plant_lego=a.plant_lego,
                     legacy_divisor=a.legacy_divisor)
    metric_attribution(nemo, lego, tab)
    tab.render()
    scored = [r for r in tab.rows
              if r["status"] in ("AT BAR", "DEBT")]
    bar = sum(1 for r in scored if r["status"] == "AT BAR")
    print(f"\nRESULT: {bar} of {len(scored)} scored rows AT BAR -- "
          f"{'PASS' if not tab.fail else 'FAIL'}")
    return 1 if tab.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
