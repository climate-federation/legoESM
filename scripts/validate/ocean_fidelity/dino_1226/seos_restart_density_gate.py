#!/usr/bin/env python
"""WHICH STATEMENT OF THE S-EOS DEPARTS FROM NEMO'S, per cell.

Round 43 left the kt=1 frozen forcing at ``zv_frc`` rel 1.1755e-11 and showed
that, GIVEN THE SAME DENSITY, every remaining statement of ``hpg_sco`` agrees
to 2.2734e-13 -- 52x below it.  So the next statement is the equation of
state, and this gate scores it directly, with no pressure gradient, no depth
mean and no ``r1_hv_0`` in the middle.

THE RECORD.  ``cfgs/DINO/MY_SRC/restart.F90:203-204`` (compiled at
``BLD/ppsrc/nemo/restart.f90:190-191``) writes ``rhd`` from
``CALL eos( ts, Kmm, rhd )`` -- the three-argument member of ``INTERFACE eos``
(``eosbn2.f90:66-68``), i.e. ``eos_insitu_New`` and its ``np_seos`` branch at
``eosbn2.f90:357-369``.  ``stpmlf.f90:590`` calls ``rst_write(kstp, Nbb, Nnn)``
AFTER the time-level swap at ``:577-579``, so this ``Kmm`` is the AFTER state
-- the SAME level as the file's own ``tn``/``sn``/``sshn``, written in the
adjacent statements of the same routine.  ``rhd``, ``tn``, ``sn`` and ``sshn``
are therefore one self-consistent tuple and NO time level has to be inferred
(Rule 1d is satisfied by construction rather than by memory).

WHY THIS RECORD AND NOT THE INITIAL STATE.  ``sshn`` is NOT zero here (range
-0.0449 .. +0.1169 m), so the ``key_qco`` stretch ``1 + r3t`` is exercised.
On the initial state r3t is identically zero and the depth operand's stretch
would be untested.

THE ORACLE'S STATEMENT, transcribed (``eosbn2.f90:360-369``)::

    zt  = pts(ji,jj,jk,jp_tem,Knn) - rn_T0
    zs  = pts(ji,jj,jk,jp_sal,Knn) - rn_S0
    zh  = ((gdept_3d(ji,jj,jk) ) *(1._wp+r3t(ji,jj,Knn)))
    zn  = - rn_a0 * ( 1 + 0.5*rn_lambda1*zt + rn_mu1*zh ) * zt
          + rn_b0 * ( 1 - 0.5*rn_lambda2*zs - rn_mu2*zh ) * zs
          - rn_nu * zt * zs
    prd = zn * r1_rho0 * tmask

with ``r3t = ssh * r1_ht_0`` (``domqco.f90:206``) and
``ht_0 = SUM(e3t_0*tmask)``.  Coefficients as the record's OWN namelist
resolves them (``RUN_FROMREST_KT1/ocean.output:218-226``, ``:233`` for rho0).

RETRACTED BEFORE IT WAS RECORDED, and it was the preregistered claim
(``PREREG_seos_depth_and_association.md``): "legoESM reads a 1-D ``gdept_1d``
ladder where NEMO reads the 3-D ``gdept_3d``".  FALSE.  NEMO's ``gdept_0`` IS
3-D under ``key_vco_3d`` and it DOES differ from ``gdept_1d`` by up to 104.97 m
-- but it is HORIZONTALLY UNIFORM (spread 0.0 at every level, row D0b), it is
the ``e3_to_depth`` round trip of the analytic ladder rather than the analytic
ladder itself, and the card's ``t_depth_ref`` already reproduces it BIT FOR BIT
(row D0).  The 1-D ladder legoESM reads is the RIGHT ladder; ``gdept_1d`` is a
field NEMO's S-EOS never touches.  What killed the claim is row D0, and it is
kept in the table so the refutation stays visible.

Usage (GPU: this card's XLA CPU compile crashes on this machine)::

    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
        python scripts/validate/ocean_fidelity/dino_1226/\\
            seos_restart_density_gate.py [--plant] [--plant-lego]

``--plant`` moves one wet cell of the NEMO-from-NEMO calibration by one ulp;
C1 must then be REFUSED and nothing below it may be scored.  ``--plant-lego``
moves one wet cell of legoESM's own T field; every row must then MOVE (a
status flip is not demanded, because a row already in DEBT cannot flip and
demanding it would make the control unfalsifiable -- round 43's blocker).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

RUN_KT1 = "/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt1_rankdump"

#: ``RUN_FROMREST_KT1/ocean.output:218-226`` and ``:233``.  ``_namelist_check``
#: below re-reads every one of these from the RECORD's own namelist, so these
#: literals are what the transcription uses and the check is what proves they
#: are this record's.
RHO0 = 1026.0
NAMEOS = dict(rn_T0=10.0, rn_S0=35.0, rn_a0=0.165, rn_b0=7.6554e-1,
              rn_lambda1=0.06, rn_lambda2=0.0, rn_mu1=1.4970e-4,
              rn_mu2=0.0, rn_nu=0.0)

#: The calibration must come in UNDER these.  It is a NEMO-from-NEMO
#: comparison of a pointwise polynomial -- no reduction, no intrinsic whose
#: order numpy cannot reproduce -- so it is expected to be EXACTLY zero and
#: these are a ceiling on the instrument, never a tolerance granted to the
#: model.  Measured 0.0 / 0.0 on 2026-09-11.
CALIB_MAX = 1.0e-18
CALIB_REL = 1.0e-15

#: A plant this size sits far above the floor (0.0) and far below the residual
#: under investigation, so it can neither hide nor be mistaken for the defect.
PLANT_REL = 1.0e-9


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table_cls():
    """Reuse the AT BAR / DEBT / UNMEASURED table, never a second copy."""
    return _load("_spg_gate",
                 os.path.join(_HERE, "spg_kt1_frozen_forcing_hpg.py")).Table


def _namelist_check(run_dir):
    """Every coefficient this gate uses, re-read from the RECORD's namelist."""
    out = os.path.join(run_dir, "ocean.output")
    if not os.path.exists(out):
        raise SystemExit(f"no ocean.output under {run_dir}: the coefficients "
                         "this gate transcribes cannot be proven to be the "
                         "record's, and an unproven coefficient is a guess")
    txt = open(out, errors="replace").read()
    seen = {}
    # ``name = value [units]`` with ``name`` as a WHOLE token: ocean.output
    # also prints ``r1_rho0`` and ``rho0_rcp`` on neighbouring lines, and a
    # substring match would silently read one of those instead.
    pat = re.compile(
        r"(?<![A-Za-z0-9_])(" + "|".join(list(NAMEOS) + ["rho0"])
        + r")\s*=\s*([-+0-9.]+(?:[EeDd][-+]?[0-9]+)?)")
    for line in txt.splitlines():
        for key, val in pat.findall(line):
            try:
                seen.setdefault(key, float(val.replace("D", "E")
                                              .replace("d", "e")))
            except ValueError:
                pass
    bad = []
    for key, want in list(NAMEOS.items()) + [("rho0", RHO0)]:
        got = seen.get(key)
        if got is None:
            bad.append(f"{key}: not found in ocean.output")
        elif got != want:
            bad.append(f"{key}: record has {got!r}, gate uses {want!r}")
    if bad:
        raise SystemExit("the record's namelist disagrees with this gate:\n  "
                         + "\n  ".join(bad))
    print(f"  namelist check: {len(seen)} coefficients re-read from "
          f"{os.path.basename(out)}, all identical to the transcription's")


def nemo_zn_literal(zt_T, zs_S, gdept3, r3t):
    """``eosbn2.f90:360-367``, statement for statement, in numpy fp64.

    Returns ``zn`` [kg/m3] -- the anomaly BEFORE ``:369`` scales it by
    ``r1_rho0``.  Kept separate from :func:`nemo_prd_literal` so this gate's
    OWN reference never performs the round trip it exists to measure: an
    earlier version built the kg/m3 reference as ``prd * rho0`` and read back
    2.2204e-16 on 125136 cells, which was the gate's arithmetic, not the
    model's.
    """
    c = NAMEOS
    zt = zt_T - c["rn_T0"]
    zs = zs_S - c["rn_S0"]
    zh = gdept3 * (1.0 + r3t)
    return (-c["rn_a0"] * (1.0 + 0.5 * c["rn_lambda1"] * zt
                           + c["rn_mu1"] * zh) * zt
            + c["rn_b0"] * (1.0 - 0.5 * c["rn_lambda2"] * zs
                            - c["rn_mu2"] * zh) * zs
            - c["rn_nu"] * zt * zs)


def nemo_prd_literal(zt_T, zs_S, gdept3, r3t, tmask):
    """``eosbn2.f90:369`` -- ``prd = zn * r1_rho0 * tmask``."""
    return nemo_zn_literal(zt_T, zs_S, gdept3, r3t) * (1.0 / RHO0) * tmask


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=RUN_KT1)
    ap.add_argument("--plant", action="store_true")
    ap.add_argument("--plant-lego", action="store_true")
    args = ap.parse_args()

    from rebuild_nemo_restart import rebuild
    import netCDF4 as nc

    run = args.run_dir
    print(f"RECORD: {run}")
    _namelist_check(run)

    mesh = rebuild(os.path.join(run, "mesh_mask_*.nc"),
                   ["gdept_0", "e3t_0", "tmask"])
    rst = rebuild(os.path.join(run, "DINO_00000001_restart_*.nc"),
                  ["tn", "sn", "sshn", "rhd"])
    for name, arr in list(mesh.items()) + list(rst.items()):
        if not np.isfinite(arr).all():
            raise SystemExit(
                f"{name} has {int((~np.isfinite(arr)).sum())} non-finite "
                "values after the rank stitch -- a gap in the stitch would "
                "silently shrink every mask below")
    g3 = mesh["gdept_0"]                       # (z, y, x), NEMO order
    tmask = mesh["tmask"]
    nz, ny, nx = g3.shape
    ht0 = (mesh["e3t_0"] * tmask).sum(axis=0)
    r3t = np.where(ht0 > 0.0, rst["sshn"] / np.where(ht0 > 0.0, ht0, 1.0), 0.0)
    print(f"  domain {nz} x {ny} x {nx};  ssh range "
          f"{rst['sshn'].min():+.6f} .. {rst['sshn'].max():+.6f} m, so the "
          "1+r3t stretch is exercised")

    T_n = rst["tn"].copy()
    S_n = rst["sn"].copy()
    ref = rst["rhd"]
    #: NEMO's loop is ``jk = 1, jpkm1``; level jpk is never written.
    sel = (tmask > 0.5)
    sel[nz - 1:, :, :] = False
    if not sel.any():
        raise SystemExit("the wet mask selected no cell")

    Table = _table_cls()
    tab = Table()
    # C1 is a transcription LOCAL TO THIS FILE (``nemo_prd_literal`` above),
    # NOT ``legoesm.ocean.eos.nemo_seos_prd_literal``.  If it were the same
    # function, R1 - C1 would be zero by construction and R1 would be a
    # vacuous row; an independent claim review raised exactly that.

    if args.plant:
        j, i, k = np.argwhere(sel.transpose(1, 2, 0))[len(
            np.argwhere(sel.transpose(1, 2, 0))) // 2]
        ref = ref.copy()
        ref[k, j, i] *= (1.0 + PLANT_REL)
        print(f"  PLANT: NEMO's own rhd at ({k},{j},{i}) scaled by "
              f"1+{PLANT_REL:g}; C1 must be REFUSED")

    print()
    print("1. CALIBRATION -- NEMO's rhd rebuilt from NEMO's own operands")
    built = nemo_prd_literal(T_n, S_n, g3, r3t[None, :, :], tmask)
    tab.calibrate("C1 literal eosbn2.f90:360-369", built, ref, sel)

    # ---- the card's own geometry and its own production statements --------
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.eos import (
        NemoSEOSConfig, make_eos_fn, nemo_bn2_live_ladders,
        nemo_seos_prd_literal,
    )
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    print()
    print("2. the card, instantiated (Rule 10 -- printed, never declared)")
    print(f"  eos={mc.eos!r} eos_depth={getattr(mc, 'eos_depth', None)!r} "
          f"rho_0={mc.rho_0!r} g={mc.g!r} "
          f"pgf_quadrature={getattr(mc, 'pgf_quadrature', None)!r}")
    # An independent claim review named LEGOESM_NEMO_E3T as a HIDDEN choice
    # that decides which ladder ``t_depth_ref`` carries.  It is read only by
    # ``nemo_state_bridge`` (the topo bridge), and this gate builds the card
    # through ``dino_lat_lon_vertical`` instead -- but an unprinted
    # environment variable is exactly the kind of thing that is claimed to be
    # out of the path and is not, so it is PRINTED and row D0 is what proves
    # the ladder independently of it.
    print(f"  env LEGOESM_NEMO_E3T="
          f"{os.environ.get('LEGOESM_NEMO_E3T', '<unset>')!r} "
          "(read by nemo_state_bridge only; this card is built by "
          "dino_lat_lon_vertical)")
    for nm, a in (("t_depth_ref", z.t_depth_ref),
                  ("nemo_gdept_0", z.nemo_gdept_0)):
        aa = np.asarray(a)
        if aa.dtype != np.float64:
            raise SystemExit(f"{nm} is {aa.dtype}, not float64 (Rule 1c)")
        print(f"  z_coord.{nm:14s} {aa.shape} {aa.dtype}")

    #: legoESM carries (y, x, z); NEMO carries (z, y, x).
    def L(a):
        return np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)

    T_l = np.moveaxis(T_n, 0, -1).copy()
    S_l = np.moveaxis(S_n, 0, -1).copy()
    eta_l = np.asarray(rst["sshn"], dtype=np.float64).copy()
    if args.plant_lego:
        jj, ii, kk = ny // 2, nx // 2, 3
        if not sel[kk, jj, ii]:
            raise SystemExit("the plant cell is not wet")
        T_l[jj, ii, kk] *= (1.0 + PLANT_REL)
        print(f"  PLANT-LEGO: legoESM's T at ({jj},{ii},{kk}) scaled by "
              f"1+{PLANT_REL:g}; every row below must MOVE")

    H_col = np.asarray(jnp.sum(jnp.asarray(z.h_partial), axis=-1))
    tmask_l = np.moveaxis(tmask, 0, -1)

    # D0 -- the RETRACTION row.  The card's static ladder against NEMO's own
    # 3-D gdept_0, and NEMO's gdept_0 against a horizontally uniform column.
    card_g3 = np.asarray(z.nemo_gdept_0, dtype=np.float64)
    tab.score("D0 card nemo_gdept_0 vs NEMO", L(card_g3), g3, sel)
    t_ref_b = np.broadcast_to(
        np.asarray(z.t_depth_ref, dtype=np.float64)[:, None, None], g3.shape)
    tab.score("D0b card t_depth_ref vs NEMO", t_ref_b, g3, sel)
    spread = float(np.max(g3.max(axis=(1, 2)) - g3.min(axis=(1, 2))))
    tab.note(f"NEMO's gdept_0 varies horizontally by {spread:.3e} m -- it is "
             "3-D under key_vco_3d but UNIFORM, which is what refuted the "
             "preregistered depth-operand claim")
    g1 = np.squeeze(nc.Dataset(sorted(glob.glob(
        os.path.join(run, "mesh_mask_*.nc")))[0]
    ).variables["gdept_1d"][:]).astype(np.float64)
    tab.note(f"and it differs from the ANALYTIC gdept_1d by up to "
             f"{np.abs(g3 - g1[:, None, None])[sel].max():.4e} m on "
             f"{int((np.abs(g3 - g1[:, None, None])[sel] > 0).sum())} cells, "
             "a field the S-EOS never reads")

    # D1 -- the LIVE depth the production lanes build.
    live = np.asarray(nemo_bn2_live_ladders(
        z, jnp.asarray(eta_l), jnp.asarray(H_col),
        r3t_evaluation="nemo_reciprocal")[0], dtype=np.float64)
    live = np.broadcast_to(live, (ny, nx, nz)) if live.ndim == 1 else live
    tab.score("D1 live gdept vs NEMO gdept(Kmm)",
              L(live), g3 * (1.0 + r3t[None, :, :]), sel)

    zh_l = jnp.asarray(card_g3) * jnp.asarray(1.0 + r3t)[..., None]
    seos = NemoSEOSConfig(rho0=mc.rho_0)

    # R1 -- legoESM's ORDERED transcription (what GM/Redi's nemo_literal runs)
    r1 = np.asarray(nemo_seos_prd_literal(
        jnp.asarray(T_l), jnp.asarray(S_l), zh_l, seos), dtype=np.float64)
    tab.score("R1 nemo_seos_prd_literal", L(r1) * tmask, ref, sel)

    # R2 -- the PRODUCTION association: rho0 + zn, then subtract rho0
    eos_fn = make_eos_fn(mc.eos, getattr(mc, "eos_linear", None),
                         rho0=mc.rho_0)
    from legoesm import constants
    p_eos = (mc.rho_0 * constants.g) * zh_l
    rho = eos_fn(jnp.asarray(T_l), jnp.asarray(S_l), p_eos)
    r2 = np.asarray((rho - mc.rho_0) / mc.rho_0, dtype=np.float64)
    diag = []
    diag.append(("R2 the statement this round REPLACED: (rho0+zn) - rho0",
                 L(r2) * tmask, ref))

    # R3 -- the production STAGE, end to end, on NEMO's own state
    land = jnp.asarray(np.asarray(z.is_active).any(axis=-1))
    with jax.disable_jit():
        _j, _h, rho_prime, _p = pe._bc_geometry_and_density(
            jnp.asarray(eta_l), jnp.asarray(H_col), z, mc,
            jnp.asarray(T_l), jnp.asarray(S_l), land, grid, mc.rho_0, mc.g)
    if np.asarray(rho_prime).dtype != np.float64:
        raise SystemExit("rho_prime is not float64 (Rule 1c)")
    r3 = np.asarray(rho_prime, dtype=np.float64) / mc.rho_0
    diag.append(("R3 the GATE's own rho'/rho_0 vs NEMO's zn*r1_rho0",
                 L(r3) * tmask, ref))

    # R3b IS THE MODEL'S ROW.  rho' in kg/m3 against NEMO's own ``zn``, with
    # no ``r1_rho0`` and no ``rho0`` on either side.  R3 above divides
    # legoESM's rho' by rho_0 where NEMO multiplies by r1_rho0; those differ
    # by up to one ulp, so once the round trip is gone R3 is measuring THIS
    # GATE's division.  legoESM carries the anomaly in kg/m3 and divides by
    # rho_0 only at the end of the pressure gradient, so kg/m3 is the
    # quantity that exists on both sides without a conversion.
    zn_nemo = np.where(tmask > 0.5,
                       nemo_zn_literal(T_n, S_n, g3, r3t[None, :, :]), 0.0)
    tab.score("R3b production rho' vs NEMO zn",
              L(np.asarray(rho_prime, dtype=np.float64)) * tmask,
              zn_nemo, sel)

    # These two are DIAGNOSTICS, not model rows, and they are printed OUTSIDE
    # the table so the gate's pass/fail cannot be decided by them.  R2 is the
    # statement that was replaced -- it is DEBT by construction and always
    # will be; scoring it would mean this gate can never go green even when
    # the model is exact.  R3 is a unit conversion this gate performs and the
    # model does not (legoESM carries the anomaly in kg/m3 to the end of the
    # pressure gradient; NEMO carries prd).  R3b is the row that decides.
    print()
    print("DIAGNOSTICS (not scored -- neither is a statement the model runs)")
    for nm, b, o in diag:
        bb = np.asarray(b, dtype=np.float64)[sel]
        oo = np.asarray(o, dtype=np.float64)[sel]
        dd = float(np.abs(bb - oo).max())
        rr = float(np.sqrt(np.mean(oo ** 2)))
        print(f"  {nm:56s} max|d|={dd:.4e}  rel={dd / rr:.4e}  "
              f"cells!={int((bb != oo).sum())}")

    tab.note("the reduction is max|d| over cells, normalised by rms(NEMO) -- "
             "NOT a per-cell relative error, which is unbounded wherever the "
             "anomaly passes through zero (T=10 C, S=35)")
    tab.note("--plant-lego moves every row by the same amount because they "
             "share one input; it proves the rows are LIVE, and cannot "
             "discriminate between them")
    tab.render()
    if tab.floor is not None and tab.rows[0]["status"] != "CALIBRATED":
        print("\nthe NEMO-from-NEMO calibration is not CALIBRATED, so no row "
              "below it is underwritten and none may be quoted")
        return 1
    scored = [r for r in tab.rows if r["status"] in ("AT BAR", "DEBT")]
    first = next((r for r in scored if r["status"] == "DEBT"), None)
    print(f"\nRESULT: {sum(1 for r in scored if r['status'] == 'AT BAR')} of "
          f"{len(scored)} scored rows AT BAR against a floor of "
          f"{tab.floor:.4e}")
    if first is not None:
        print(f"FIRST NON-BIT STATEMENT: {first['name']} at "
              f"{first['rel']:.4e} relative on {first['cells']} of "
              f"{first['n']} cells")
    return 1 if tab.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
