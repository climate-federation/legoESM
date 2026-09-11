#!/usr/bin/env python
"""NEMO's kt=1 TRACER ladder on the standalone DINO card, and the closure
fields that feed it.

Preregistered in ``PREREG_kt1_tracer_ladder.md`` (read it first).  Step 1
stands at T 4.996e-6 K rms after the geometry round moved eta/u/v by 300-700x
and left T and S untouched, so the tracer path owns the largest remaining
step-1 fraction.

WHAT THIS MEASURES, in three stages:

  STAGE 0  NEMO-side instrument validation.  NEMO's own per-operator tracer
           trends are carried in the RESTART (``ttrd_*``/``strd_*``, written
           by ``trddump.F90:336-362`` from the store ``trdtra.F90:348`` fills
           when ``ln_tra_trd = T``).  Rule 5: a trend diagnostic is
           integrator bookkeeping until closure is proved, so this stage
           REFUSES to print a ranking unless

               ttrd_tot == ttrd_nsr + ttrd_qsr + ttrd_totad
                           + ttrd_ldf + ttrd_zdf

           to roundoff.  It also prints the SAME check for the
           ``xad``/``yad``/``zad`` triple, which does NOT close -- that triple
           is the ADVECTIVE-form decomposition built by ``trd_tra_adv``
           (``trdtra.F90:215``), not the additive flux-form member, and
           quoting it would have produced exactly the false finding Rule 5
           warns about.

  STAGE 1  The closure fields, the preregistered discriminator: NEMO's
           ``avt_k``/``avm_k``/``en``/``dissl`` at the end of kt=1 vs
           legoESM's ``state.tke_avt``/``tke_avm``/``tke``/``tke_dissl``
           after ONE step of the shipped card.  AT BAR (0 cells unequal) =>
           the TKE closure is exonerated and the vertical solve or the
           surface flux owns the temperature residual; DEBT => the closure
           does.

  STAGE 2  The total tracer tendency through the model's own path, scored
           against ``ttrd_tot``/``strd_tot`` in NEMO's own definition
           (``traatf_qco.f90:133-139``):

               trd_tot = ( T(Kaa)*(1+r3t(Kaa))/(1+r3t(Kmm)) - T(Kmm) ) / rn_Dt

           reconstructed on BOTH sides from the same formula, so the
           comparison is of the models and not of two conventions.  The
           NEMO side is reconstructed from ``tn``/``tb``/``sshn``/``sshb``
           and checked against the stored ``ttrd_tot`` first -- an
           independent second validation of the instrument.

WHAT THIS DELIBERATELY DOES NOT USE.  ``RUN_FROMREST_KT1`` is a **16-rank**
run (``layout.dat``: ``jpnij 16``, ``jpi 30``, ``jpj 29``, ``nn_hls 2``) and
every ``stp_dump_*.bin`` / ``tke_dump_*.bin`` / ``sbc_dump_*.bin`` in it
carries NO rank suffix, so all 16 ranks open the same filename and the
survivor is a race -- possibly a mixture.  Stage 0 PRINTS that evidence from
the file sizes rather than leaving it as a claim.  The previous round
recorded this for the barotropic streams alone; it is every un-suffixed dump
in the record.

Usage (GPU: the card's XLA CPU compile crashes on this machine)::

    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
        python scripts/validate/ocean_fidelity/dino_1226/kt1_tracer_ladder.py

NON-VACUITY.  Every row here reports DEBT on a first run (the bar is EXACT
and nothing is at it yet), so a plant that only makes a row fail would prove
nothing -- it would be a no-op on a gate that is already red.  The plants are
therefore on the INSTRUMENT, which is what a passing row would rest on:

``--plant-closure``  scales one additive trend by 1 + 1e-9.  Stage 0's
    budget check MUST then refuse and the script MUST exit non-zero before
    any row is printed.
``--plant-shift``    shifts stage 1's W-level alignment by one level.  The
    closure residuals MUST then grow by orders of magnitude; if they do not,
    the ~1e-17 agreement is insensitive to the alignment and proves nothing.
    The script exits non-zero unless every field's residual grows by >=1e6x.

``--nemo-only`` runs stage 0 alone (no JAX, no GPU).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from rebuild_nemo_restart import rebuild  # noqa: E402

DEFAULT_RESTART = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
                   "RUN_FROMREST_KT1/DINO_00000001_restart_*.nc")
DT = 2700.0                      # NEMO rn_Dt; rDt == rn_Dt on the Euler start

#: The additive members of NEMO's tracer budget at kt=1 on THIS card.  Every
#: other ``jptra_*`` is either identically zero here (bbc/dmp/npc: the
#: namelist switch is off; atf: ``traatf_qco.f90:152-159`` writes an explicit
#: zero on the Euler step) or a non-additive re-diagnostic (``zdfp``/``evd``,
#: recomputed inside ``trdtra.F90``'s ``jptra_zdfp`` case from ``avt`` and
#: ``ts(Krhs)`` -- Rule 5).
ADDITIVE_T = ("nsr", "qsr", "totad", "ldf", "zdf")
ADDITIVE_S = ("nsr", "totad", "ldf", "zdf")
#: Advective-form decomposition; present, NOT additive.  Kept so the failure
#: is printed rather than silently avoided.
ADVECTIVE_TRIPLE = ("xad", "yad", "zad")
#: Non-additive re-diagnoses, reported for their size only.
REDIAGNOSED = ("zdfp", "evd")
#: Terms whose namelist switch is off on this card; they must be EXACTLY zero
#: and the gate says so if they are not.
MUST_BE_ZERO = ("bbc", "dmp", "npc", "atf")

CLOSURE_RTOL = 1e-12             # the budget is exact algebra, not physics


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> dict:
    d = (a - b)[mask]
    ref = b[mask]
    scale = float(np.sqrt(np.mean(ref ** 2)))
    return {
        "n": int(mask.sum()),
        "ne": int(np.count_nonzero(d)),
        "max": float(np.max(np.abs(d))) if d.size else 0.0,
        "rms": float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0,
        "scale": scale,
        "frac": (float(np.sqrt(np.mean(d ** 2))) / scale) if scale else float("nan"),
    }


def _row(name: str, s: dict) -> str:
    verdict = "AT BAR" if s["ne"] == 0 else "DEBT"
    return (f"  {name:22s} cells!={s['ne']:8d}/{s['n']:<8d} "
            f"max={s['max']:11.4e} rms={s['rms']:11.4e} "
            f"NEMO_rms={s['scale']:11.4e} frac={s['frac']:9.3e}  {verdict}")


# ----------------------------------------------------------------- stage 0
def stage0(restart_glob: str, plant: bool = False) -> tuple[dict, int]:
    """NEMO-side only.  Returns (rebuilt arrays, exit status contribution)."""
    tiles = sorted(glob.glob(restart_glob))
    if not tiles:
        raise SystemExit(f"no restart tiles match {restart_glob}")
    run_dir = os.path.dirname(tiles[0])
    print(f"STAGE 0 -- NEMO's own kt=1 tracer trends ({len(tiles)} tiles)")
    print(f"  record: {run_dir}")

    # --- the rank-race evidence, printed rather than asserted -------------
    layout = os.path.join(run_dir, "layout.dat")
    jpnij = jpi = jpj = None
    if os.path.exists(layout):
        with open(layout) as fh:
            for ln in fh:
                p = ln.split()
                if len(p) >= 6 and p[0].isdigit():
                    jpnij, jpi, jpj = int(p[0]), int(p[1]), int(p[2])
                    break
    probe = os.path.join(run_dir, "stp_dump_14_trasbc_kt00000001_tem.bin")
    if jpnij and jpnij > 1 and os.path.exists(probe):
        sz = os.path.getsize(probe)
        print(f"  UNSUFFIXED DUMPS UNUSABLE ON THIS RECORD: jpnij={jpnij}, "
              f"jpi={jpi}, jpj={jpj}; "
              f"stp_dump_14_trasbc_kt00000001_tem.bin is {sz} B "
              f"= {jpi}x{jpj}x{sz // (8 * jpi * jpj)}x8, ONE rank's tile, "
              f"and all {jpnij} ranks open that same filename. Every "
              "un-suffixed *_dump_*.bin in this record is a filename race.")

    want = []
    for pre in ("ttrd", "strd"):
        for s in (ADDITIVE_T + ADVECTIVE_TRIPLE + REDIAGNOSED
                  + MUST_BE_ZERO + ("tot",)):
            want.append(f"{pre}_{s}")
    want += ["tn", "sn", "tb", "sb", "sshn", "sshb", "un", "vn", "ub", "vb",
             "en", "avt_k", "avm_k", "dissl"]
    R = rebuild(restart_glob, want)
    if "ttrd_tot" not in R:
        raise SystemExit(
            "the restart carries no ttrd_tot: ln_tra_trd was not T for this "
            "run, so there is no per-operator ladder in it at all")

    if plant:
        # Scale ONE additive member by 1 part in 1e9.  A 1-ulp move was tried
        # first and is NOT detectable: the budget's tolerance is 1e-12 of
        # max|ttrd_tot| = 5.9e-17 absolute, while one ulp of ttrd_ldf's
        # largest cell is 2.2e-22 -- five orders below.  Planting it would
        # have printed CLOSED and certified nothing, so the plant states the
        # size it actually tests.
        R["ttrd_ldf"] = R["ttrd_ldf"] * (1.0 + 1.0e-9)
        R["strd_ldf"] = R["strd_ldf"] * (1.0 + 1.0e-9)
        print("  CLOSURE PLANT ACTIVE: ttrd_ldf AND strd_ldf scaled by 1 + 1e-9 (a 1 ppb "
              "error in one operator); the budget check below MUST refuse")

    bad = 0
    for pre, additive in (("ttrd", ADDITIVE_T), ("strd", ADDITIVE_S)):
        tot = np.nan_to_num(R[f"{pre}_tot"])
        scale = float(np.max(np.abs(tot)))
        print(f"\n  {pre}: per-operator max|trend| at kt=1 "
              f"[{'K/s' if pre == 'ttrd' else 'psu/s'}]")
        for s in additive + ADVECTIVE_TRIPLE + REDIAGNOSED + MUST_BE_ZERO:
            k = f"{pre}_{s}"
            if k not in R:
                continue
            a = np.nan_to_num(R[k])
            kind = ("additive" if s in additive else
                    "ADVECTIVE-FORM (not additive)" if s in ADVECTIVE_TRIPLE
                    else "re-diagnosis (not additive)" if s in REDIAGNOSED
                    else "switch off -> must be 0")
            print(f"    {k:14s} max={np.max(np.abs(a)):11.4e} "
                  f"nonzero={int(np.count_nonzero(a)):8d}  {kind}")
            if s in MUST_BE_ZERO and np.any(a != 0.0):
                print(f"    ^^ FAIL: {k} is not identically zero but its "
                      "namelist switch is off")
                bad += 1

        res = tot - sum(np.nan_to_num(R[f"{pre}_{s}"]) for s in additive)
        rel = float(np.max(np.abs(res))) / scale if scale else float("inf")
        ok = rel <= CLOSURE_RTOL
        print(f"    CLOSURE  tot == {' + '.join(additive)}: "
              f"max|res|={np.max(np.abs(res)):.4e} rel={rel:.4e}  "
              f"{'CLOSED' if ok else 'NOT CLOSED'}")
        if not ok:
            bad += 1
        trip = tot - sum(np.nan_to_num(R[f"{pre}_{s}"])
                         for s in ADDITIVE_T[:2] + ADVECTIVE_TRIPLE
                         + ADDITIVE_T[3:] if f"{pre}_{s}" in R)
        print(f"    (the xad/yad/zad triple instead of totad: rel="
              f"{float(np.max(np.abs(trip))) / scale:.4e} -- NOT the additive "
              "decomposition, never quote it)")

    # --- second, independent validation: reconstruct ttrd_tot -------------
    ht0 = None
    mesh = os.path.join(run_dir, "mesh_mask_0000.nc")
    print("\n  ttrd_tot reconstructed from tn/tb/sshn/sshb "
          "(traatf_qco.f90:133-139):")
    print("    needs ht_0; taken from the card's own mesh in stage 2.")
    del ht0, mesh
    return R, bad


# ----------------------------------------------------------------- stages 1/2
def stages12(R: dict, plant_shift: bool) -> int:
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64())                      # Rule 1c
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    state0 = dm.dino_lat_lon_state(grid, z, cfg)
    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    sf_step = (dm.dino_step_surface_forcing(forcing)
               if getattr(cfg, "wind_through_step", False) else None)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    print(f"\nSTAGE 1/2 -- card: outer={mc.flat_get('outer_integrator')!r} "
          f"vmix={getattr(mc, 'vmix_scheme', '?')!r} "
          f"precision={get_policy().storage.__name__} "
          f"c_p={cfg.c_p!r}")

    model = LatLonCGridOceanModel(grid, z, mc)
    st, rate = dm.apply_dino_lat_lon_surface_forcing(
        state0, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
    after = model.step(st, dt=DT, surface_forcing=sf_step,
                       external_tracer_rate=rate)

    g = ndm.nemo_dino_mesh()
    wet3 = g.tmask > 0.5

    def O3(k):
        return np.moveaxis(R[k], 0, -1)

    # ---- STAGE 1: the closure fields ---------------------------------
    print("\nSTAGE 1 -- the closure at the end of kt=1 "
          "(NEMO restart vs legoESM state)")
    pairs = (("avt_k", "tke_avt"), ("avm_k", "tke_avm"),
             ("en", "tke"), ("dissl", "tke_dissl"))
    bad = 0
    carried = []
    for nemo_name, lego_name in pairs:
        obj = getattr(after, lego_name, None)
        if obj is None:
            print(f"  {nemo_name:22s} UNMEASURED: the card carries no "
                  f"state.{lego_name} after one step "
                  "(the closure's output is not retained on state)")
            bad += 1
            continue
        a = np.asarray(obj.data if hasattr(obj, "data") else obj, dtype=float)
        b = O3(nemo_name)
        # LEVEL ALIGNMENT -- READ, then PRINTED, never assumed (Rule 10).
        # NEMO's avt/avm/en/dissl live on jpk = 36 W levels with avt(k) the
        # coefficient at the TOP face of cell k; legoESM's carry is the
        # INTERIOR W grid and its own construction states the mapping:
        # "Interior legoESM W index k is NEMO W level k+2"
        # (ocean_model_latlon_cgrid.py:7219-7220, _seed_tke_preclosure_carry),
        # i.e. Fortran k+2 == python index k+1.  legoESM's array carries
        # nlev-1 = 34 interior interfaces plus a zero tail; NEMO's index 35
        # (Fortran 36) is the permanently dry dummy.  The diagnostic column
        # below prints both sides at the same physical interfaces so a slip
        # is visible instead of silent.
        nint = b.shape[-1] - 2                    # 34 interior W interfaces
        tail = a[..., nint:]
        a_i, b_i = a[..., :nint], b[..., 1:1 + nint]
        # NEMO's wmask(k) = tmask(k)*tmask(k-1): the interface is wet only if
        # both neighbouring cells are.  In legoESM index k that is NEMO
        # python k+1 and k.
        wmask = wet3[..., 1:1 + nint] & wet3[..., :nint]
        col = int(np.argmax(np.sum(wet3, axis=-1)))
        iy, ix = np.unravel_index(col, wet3.shape[:2])
        print(f"    [{nemo_name}] deepest column ({iy},{ix}) interior W "
              f"k=0,1,2: lego {a_i[iy, ix, 0]:.9e} {a_i[iy, ix, 1]:.9e} "
              f"{a_i[iy, ix, 2]:.9e} | NEMO {b_i[iy, ix, 0]:.9e} "
              f"{b_i[iy, ix, 1]:.9e} {b_i[iy, ix, 2]:.9e}; "
              f"lego tail max={float(np.max(np.abs(tail))):.3e}, "
              f"NEMO dummy max="
              f"{float(np.max(np.abs(np.nan_to_num(b[..., -1])))):.3e}")
        a, b, msk = a_i, b_i, wmask
        s = _stats(a, b, msk)
        if plant_shift:
            # Shift the alignment by ONE level and require the residual to
            # explode.  If it does not, the ~1e-17 agreement above is not
            # evidence about the closure -- it is evidence about nothing.
            b2 = O3(nemo_name)[..., 2:2 + nint]
            s2 = _stats(a, b2, msk)
            grew = (s2["rms"] / s["rms"]) if s["rms"] else float("inf")
            print(f"    SHIFT PLANT: one-level shift gives rms "
                  f"{s2['rms']:.4e} vs {s['rms']:.4e} ({grew:.3e}x)")
            if not grew >= 1e6:
                print("    ^^ the alignment is INSENSITIVE: this row does "
                      "not test what it claims to")
                bad += 1
        carried.append((nemo_name, s))
        print(_row(nemo_name, s))
        if s["ne"]:
            bad += 1

    # ---- STAGE 2: the total tracer tendency --------------------------
    # NEMO's own definition, traatf_qco.f90:133-139.  Both sides built by the
    # SAME formula so the comparison is of models, not conventions.
    # ht_0 = SUM( e3t_0 * tmask ) -- NEMO's own definition, domain.F90:144.
    ht0 = np.sum(np.asarray(g.e3t_0, dtype=float)
                 * np.asarray(g.tmask, dtype=float), axis=-1)
    print("\nSTAGE 2 -- total tracer tendency, NEMO's own definition")
    print(f"  ht_0 = sum(e3t_0*tmask) [domain.F90:144]: "
          f"max={ht0.max():.4f} m over {int((ht0 > 0).sum())} wet columns")
    with np.errstate(invalid="ignore", divide="ignore"):
        r3t_n = np.where(ht0 > 0, R["sshn"] / np.where(ht0 > 0, ht0, 1.0), 0.0)
        r3t_b = np.where(ht0 > 0, R["sshb"] / np.where(ht0 > 0, ht0, 1.0), 0.0)
    tm = wet3.astype(float)

    def nemo_trd(aa, mm, r3a, r3m):
        return ((aa * (1.0 + r3a[..., None] * tm)
                 / (1.0 + r3m[..., None] * tm)) - mm) / DT

    for lbl, a_nemo, m_nemo, stored, lego_a, lego_m in (
            ("T", O3("tn"), O3("tb"), np.nan_to_num(O3("ttrd_tot")),
             np.asarray(after.T.data), np.asarray(st.T.data)),
            ("S", O3("sn"), O3("sb"), np.nan_to_num(O3("strd_tot")),
             np.asarray(after.S.data), np.asarray(st.S.data))):
        recon = nemo_trd(a_nemo, m_nemo, r3t_n, r3t_b)
        d = (recon - stored)[wet3]
        sc = float(np.sqrt(np.mean(stored[wet3] ** 2)))
        print(f"  {lbl} instrument check: reconstructed ttrd_tot vs stored: "
              f"max={np.max(np.abs(d)):.4e} rel={np.max(np.abs(d))/sc:.4e}")
        eta_a = np.asarray(after.eta.data)
        eta_m = np.asarray(st.eta.data)
        with np.errstate(invalid="ignore", divide="ignore"):
            l3a = np.where(ht0 > 0, eta_a / np.where(ht0 > 0, ht0, 1.0), 0.0)
            l3m = np.where(ht0 > 0, eta_m / np.where(ht0 > 0, ht0, 1.0), 0.0)
        lego = nemo_trd(lego_a, lego_m, l3a, l3m)
        print(_row(f"{lbl} trd_tot", _stats(lego, stored, wet3)))

        # ---- STAGE 3: WHERE the residual lives ------------------------
        # The closure fields are at roundoff, so the owner is downstream of
        # the TKE closure.  This localises the residual against NEMO's OWN
        # additive operators -- which is a CORRELATION, not a mechanism, and
        # is labelled PLAUSIBLE wherever it is quoted.  Operators overlap in
        # space, so the pairwise correlation of each pair is printed too: a
        # residual that correlates with two mutually-correlated operators
        # attributes to neither.
        pre = "ttrd" if lbl == "T" else "strd"
        ops = ADDITIVE_T if lbl == "T" else ADDITIVE_S
        res = (lego - stored)
        rw = res[wet3]
        print(f"  {lbl} residual localisation (PLAUSIBLE -- correlation, "
              f"not mechanism)")
        print(f"    {'operator':12s}{'op rms':>12s}{'corr(res,op)':>14s}"
              f"{'lstsq c':>12s}{'var expl':>10s}")
        cols = {}
        for o in ops:
            a = np.nan_to_num(O3(f"{pre}_{o}"))[wet3]
            cols[o] = a
            den = float(a @ a)
            c = float(rw @ a) / den if den else 0.0
            corr = (float(rw @ a) / np.sqrt(float(rw @ rw) * den)
                    if den and float(rw @ rw) else 0.0)
            ve = (1.0 - float((rw - c * a) @ (rw - c * a)) / float(rw @ rw)
                  if float(rw @ rw) else 0.0)
            print(f"    {o:12s}{np.sqrt(np.mean(a**2)):12.4e}{corr:14.4f}"
                  f"{c:12.4e}{ve:10.4f}")
        print(f"    {'op pair':24s}{'corr':>10s}")
        keys = list(cols)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                x, y = cols[keys[i]], cols[keys[j]]
                dd = np.sqrt(float(x @ x) * float(y @ y))
                print(f"    {keys[i] + ' vs ' + keys[j]:24s}"
                      f"{(float(x @ y) / dd if dd else 0.0):10.4f}")
        # vertical structure, and the EVD-active cells (NEMO fired rn_evd on
        # 12700 of them; a residual concentrated there indicts the convective
        # branch rather than the background solve)
        evd = np.nan_to_num(O3(f"{pre}_evd")) != 0.0
        for nm, m in (("EVD-active", wet3 & evd), ("EVD-quiet", wet3 & ~evd)):
            if m.sum():
                print(f"    {nm:12s} n={int(m.sum()):8d} "
                      f"res_rms={np.sqrt(np.mean(res[m]**2)):11.4e} "
                      f"NEMO_rms={np.sqrt(np.mean(stored[m]**2)):11.4e} "
                      f"frac={np.sqrt(np.mean(res[m]**2))/np.sqrt(np.mean(stored[m]**2)):9.3e}")
        nz = wet3.shape[-1]
        worst = sorted(range(nz),
                       key=lambda k: -(np.sqrt(np.mean(res[..., k][wet3[..., k]]**2))
                                       if wet3[..., k].any() else 0.0))[:5]
        print("    worst 5 levels by residual rms (k, res_rms, NEMO_rms, frac):")
        for k in worst:
            m = wet3[..., k]
            if not m.any():
                continue
            r = np.sqrt(np.mean(res[..., k][m] ** 2))
            n_ = np.sqrt(np.mean(stored[..., k][m] ** 2))
            print(f"      k={k:3d}  {r:11.4e}  {n_:11.4e}  "
                  f"{(r / n_ if n_ else float('nan')):9.3e}")

    # ---- STAGE 4: tra_ldf, scored operator-to-operator ----------------
    # The localisation above is a CORRELATION.  This turns it into a real
    # per-operator number by capturing legoESM's OWN isoneutral tendency
    # during the SAME step (Rule 10: through the model's own path, not a
    # reconstruction) and scoring it against NEMO's ``ttrd_ldf``/``strd_ldf``
    # from the restart -- full domain, no rank race.
    #
    # The capture is the module-level spy ``traldf_iso_lap_probe`` already
    # uses; it is NOT re-implemented here, only re-pointed.  Identification
    # is simpler on this record than on that probe's: from REST the Nbb and
    # Nnn states are identical, so all four calls see the same input, and
    # the probe ASSERTS that the two T calls produced identical output
    # rather than assuming which is which.
    print("\nSTAGE 4 -- tra_ldf (isoneutral lateral diffusion), "
          "legoESM's own tendency vs NEMO's ttrd_ldf/strd_ldf")
    import jax
    import legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid as gmmod
    real = gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def spy(q, *a, **kw):
        r = real(q, *a, **kw)
        out = r[0] if (isinstance(r, tuple) and kw.get("return_bolus")) else r
        calls.append((np.asarray(q), np.asarray(out)))
        return r

    gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = spy
    try:
        with jax.disable_jit():
            model.step(st, dt=DT, surface_forcing=sf_step,
                       external_tracer_rate=rate)
    finally:
        gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = real
    if len(calls) != 4:
        print(f"  UNMEASURED: the isoneutral tendency was called "
              f"{len(calls)} times, expected 4 (T,S x Nnn,Nbb pass)")
        return bad + 1
    # IDENTIFY the Nbb dissipative pass by the input array, never by
    # position.  legoESM calls the isoneutral tendency 4x per step: the Nnn
    # advective pass (T,S) then the Nbb dissipative pass (T,S), and only the
    # latter is what NEMO's tra_ldf(Kbb) computes.  On THIS record the Nnn
    # pass sees an intermediate T that is NOT state.T, so the wet-masked
    # exact match against state.T/state.S selects the Nbb pass uniquely --
    # and the probe REFUSES if it is not unique rather than taking a
    # position.
    Tin, Sin = np.asarray(st.T.data), np.asarray(st.S.data)
    wT = [i for i, (q, _) in enumerate(calls)
          if q.shape == Tin.shape and np.array_equal(q[wet3], Tin[wet3])]
    wS = [i for i, (q, _) in enumerate(calls)
          if q.shape == Sin.shape and np.array_equal(q[wet3], Sin[wet3])]
    print(f"  Nbb-pass identification by wet-masked exact input match: "
          f"T->{wT}  S->{wS}")
    if len(wT) != 1 or len(wS) != 1:
        print("  UNMEASURED: the Nbb pass is not uniquely identifiable by "
              "its input, so this row would rest on a call position")
        return bad + 1
    iT, iS = wT[0], wS[0]
    for lbl, idx, oracle_key in (("T", iT, "ttrd_ldf"),
                                 ("S", iS, "strd_ldf")):
        lego = calls[idx][1]
        nemo = np.nan_to_num(O3(oracle_key))
        if lego.shape != nemo.shape:
            n = min(lego.shape[-1], nemo.shape[-1])
            lego, nemo = lego[..., :n], nemo[..., :n]
            m = wet3[..., :n]
        else:
            m = wet3
        st_ = _stats(lego, nemo, m)
        print(_row(f"{lbl} tra_ldf", st_))
        den = float(nemo[m] @ nemo[m])
        if den:
            print(f"      best-fit lego/NEMO ratio = "
                  f"{float(lego[m] @ nemo[m]) / den:.9f}")
        if st_["ne"]:
            bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart-glob", default=DEFAULT_RESTART)
    ap.add_argument("--nemo-only", action="store_true")
    ap.add_argument("--plant-closure", action="store_true",
                    help="move one additive trend by 1 ulp; stage 0 MUST "
                         "refuse (non-vacuity for the budget gate)")
    ap.add_argument("--plant-shift", action="store_true",
                    help="shift stage 1's W-level alignment by one; the "
                         "closure residuals MUST explode (non-vacuity for "
                         "the alignment the closure rows rest on)")
    args = ap.parse_args()

    R, bad = stage0(args.restart_glob, plant=args.plant_closure)
    if bad:
        print(f"\nGATE FAIL: the instrument did not validate ({bad} "
              "problems). No row above may be quoted.")
        return 1
    if args.nemo_only:
        print("\n--nemo-only: stages 1/2 skipped.")
        return 0
    bad += stages12(R, args.plant_shift)
    print(f"\n{'GATE PASS' if bad == 0 else f'GATE FAIL ({bad} rows)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
