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
             "en", "avt_k", "avm_k", "dissl",
             # NEMO's own tra_ldf inputs (stage 5) and the solar-term
             # operands (stage 6).
             "uslp_stg", "vslp_stg", "wslpi_stg", "wslpj_stg",
             "qsr_hc_b", "fraqsr_1lev"]
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



# ----------------------------------------------------------------- stage 5
def _replay(real, args, kwargs, **over):
    """Re-run the captured ``tra_ldf`` call with operands substituted.

    Returns ``(tendency, diagnostics)``.  Nothing is re-implemented: this is
    the model's own operator, called with the model's own arguments, with the
    named kwargs replaced (Rule 10).
    """
    kw = dict(kwargs)
    kw.update(over)
    kw["return_diagnostics"] = True
    kw["return_operand_diagnostics"] = True
    r = real(*args, **kw)
    if kw.get("return_bolus"):
        tend, _bolus, diags = r
    else:
        tend, diags = r
    return np.asarray(tend), diags


def _stage_split(diags, args, kwargs, z):
    """Split the operator's own tendency into its three flux stages.

    Horizontal (``zfu``/``zfv``, N3-N6), vertical skew (``zA31``/``zA32``,
    N7-N10) and the MSC A33 diagonal (N11/N13), each taken through the SAME
    divergence + metric divisor the operator applies.  The three MUST sum to
    the returned tendency; that closure is the control (Rule 5 -- a stage
    split that does not close is bookkeeping, not attribution).
    """
    import numpy as _np
    zfu = _np.asarray(diags["zfu"])
    zfv = _np.asarray(diags["zfv"])
    ops = diags["zfw_operands"]
    skew = _np.asarray(ops["skew_current"])
    a33 = _np.asarray(ops["a33_current"])
    act_below = _np.asarray(ops["act_below"])
    e1e2t = _np.asarray(ops["e1e2t"])
    jac = _np.asarray(args[7])
    act = _np.asarray(kwargs.get("active_3d", args[10] if len(args) > 10
                                 else None))
    e3t = _np.asarray(z.dz_ref)[None, None, :] * jac[:, :, None]
    scale = (1.0 / e1e2t)[:, :, None] / e3t * act

    hdiv = ((zfu - _np.roll(zfu, 1, 1)) + (zfv - _np.roll(zfv, 1, 0)))

    def _vdiv(part):
        f = part * act_below
        top = _np.roll(f, 1, 2)
        top[..., 0] = 0.0
        return top - f

    return {
        "horizontal": hdiv * scale,
        "skew": _vdiv(skew) * scale,
        "a33": _vdiv(a33) * scale,
    }


def stage5(calls, iT, iS, R, O3, wet3, z, st, real, plant_e3w: bool,
           plant_slopes: bool) -> int:
    """The preregistered substitution ladder (PREREG_kt1_ldf_stage_walk.md).

    Arms change ONE operand each and are scored against the record's own
    ``ttrd_ldf``/``strd_ldf``.  Two of them are REGRESSION WITNESSES: they
    restore the operand this round replaced, and they must reproduce the
    pre-fix numbers -- so the gate cannot pass by accident, and the fix is
    attributable to the statement it changed rather than to the round.
    """
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing import nemo_e3w0_reference

    print("\nSTAGE 5 -- tra_ldf, one statement at a time "
          "(PREREG_kt1_ldf_stage_walk.md)")
    bad = 0
    args_T, kw_T = calls[iT][2], calls[iT][3]
    jac = np.asarray(args_T[7])

    # -- the level-axis wrap the operator's jnp.roll introduces and NEMO does
    #    not have (wmask(jpk)=0).  Benign ONLY if the deepest level is
    #    permanently dry; asserted, not assumed.
    act = np.asarray(kw_T.get("active_3d", args_T[10]))
    if act[..., -1].any():
        print(f"  LEVEL WRAP UNSAFE: the deepest level carries "
              f"{int(act[..., -1].sum())} active cells, so the operator's "
              "periodic level roll wraps the surface onto the sea floor")
        bad += 1
    else:
        print("  level-axis wrap: deepest level is permanently dry "
              "(0 active cells), so the periodic roll is inert -- checked")

    # -- the rDt operand (a33's stability threshold).  NEMO: domain.f90:310
    #    rDt = 2*rn_Dt, reduced to rn_Dt on the Euler first step
    #    (stpmlf.f90:132) and restored at :618.  This record is kt=1, so the
    #    two agree HERE and the row is a forward-looking finding, not a
    #    residual this record can score.
    print(f"  rDt operand: the card passes dt={kw_T.get('dt')!r} to "
          f"traldf_iso_a33; NEMO's rDt at kt=1 is rn_Dt={DT!r} "
          f"(stpmlf.f90:132) and 2*rn_Dt={2 * DT!r} from kt=2 "
          "(stpmlf.f90:618). UNMEASURED on this record by construction.")

    # -- NEMO's OWN slopes.  TIME LEVEL, read not assumed (Rule 1d):
    #    stpmlf.f90:216 CALL ldf_slp(kstp, rhd, rn2b, Nbb, Nnn) sets the
    #    module SAVE arrays once per step and stpmlf.f90:504 CALL tra_ldf
    #    consumes them unchanged; MY_SRC/trddump.F90:307 copies those arrays.
    nat = None
    try:
        cand = tuple(np.nan_to_num(O3(k)) for k in
                     ("uslp_stg", "vslp_stg", "wslpi_stg", "wslpj_stg"))
    except KeyError:
        cand = None
    if cand is None or max(float(np.abs(a).max()) for a in cand) == 0.0:
        print("  SLOPE ARM UNMEASURED: the record's uslp_stg/vslp_stg/"
              "wslpi_stg/wslpj_stg are identically ZERO on all 16 tiles, "
              "while rhd_stg/tn_stg/rn2_stg from the SAME snapshot carry "
              "data. NEMO's tra_ldf cannot have run on zero slopes (the A33 "
              "stage below is 1.07x the tendency), so the dump -- not the "
              "run -- is what is empty. A frame that copies uslp/vslp/wslpi/"
              "wslpj at tra_ldf's own call site is needed before the slope "
              "routine can be separated from the operator.")
    else:
        nat = cand
        print(f"  NEMO slopes from the record: max|uslp|="
              f"{np.abs(nat[0]).max():.4e} max|wslpi|="
              f"{np.abs(nat[2]).max():.4e} (rn_slpmax=0.01)")

    # -- the ahtu/ahtv masking row (ldftra.f90:433-434).  This round's second
    #    source-literal fix, measured in ISOLATION rather than folded into the
    #    e3w arm: the flux the PRE-FIX operator emitted on faces where NEMO's
    #    ahtu is identically zero, reconstructed from the operator's own
    #    operands, and the tendency that flux produced.
    _t0, _d0 = _replay(real, args_T, kw_T)
    _fo = _d0["zfu_operands"]
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        nemo_iso_face_masks)
    _um, _vm, _ = nemo_iso_face_masks(
        np.asarray(args_T[4]), np.asarray(args_T[5]), act)
    _um = np.asarray(_um)
    _kap = np.asarray(args_T[9])
    if _kap.ndim == 2:
        _kap = _kap[:, :, None]
    _kap = np.broadcast_to(_kap, act.shape)
    # traldf_iso.f90:242  zfu = ahtu*(zA11*zdit + zA13*avg4); zdit already
    # carries umask, so on a closed face only the zA13 term survives.
    _zA13 = -np.asarray(_fo["e2u"])[:, :, None] * np.asarray(_fo["uslp"]) \
        * np.asarray(_fo["zmsku"])
    _removed = np.where(_um == 0.0, _kap * _zA13 * np.asarray(_fo["avg4_u"]),
                        0.0)
    _ncl = int(((_um == 0.0) & (act > 0.0)).sum())
    _nnz = int((_removed != 0.0).sum())
    print(f"  ahtu masking (ldftra.f90:433-434): {_ncl} closed u-faces on wet "
          f"cells; the pre-fix operator emitted a nonzero zA13 flux on "
          f"{_nnz} of them, max |zfu| there = {np.abs(_removed).max():.4e} "
          "(NEMO's ahtu = 0 there, so NEMO emits none)")
    _tend_from = (np.abs(_removed - np.roll(_removed, 1, 1)).max()
                  / max(float(np.abs(np.asarray(_fo['e3t'])).min()), 1e-300))
    print(f"    the flux appears in the divergence of the ADJACENT wet cell; "
          f"max |d(removed zfu)| = "
          f"{np.abs(_removed - np.roll(_removed, 1, 1)).max():.4e} m^3 K/s")
    if np.abs(_removed).max() == 0.0:
        print("    MEASURED INERT on this card at kt=1: every closed u-face "
              "carries zero slope or zero 4-point vertical-gradient average, "
              "so the fix changes no number HERE. It is landed as a "
              "transcription fix, not as a residual this record scores.")
    del _t0, _tend_from

    # -- the e3w operand, and the arm that restores the pre-fix one ---------
    raw_e3w0 = nemo_e3w0_reference(z)
    if raw_e3w0 is None:
        print("  UNMEASURED: this coordinate carries no NEMO e3w_0")
        return bad + 1
    e3w_nemo = np.asarray(raw_e3w0)[..., :z.n_levels] * jac[:, :, None]
    e3t = np.asarray(z.dz_ref)[None, None, :] * jac[:, :, None]
    e3w_mid = 0.5 * (np.roll(e3t, 1, 2) + e3t)
    e3w_mid[..., 0] = e3t[..., 0]
    _rel = np.abs(e3w_mid[..., 1:] - e3w_nemo[..., 1:]) / e3w_nemo[..., 1:]
    print(f"  e3w operand: the pre-fix midpoint vs NEMO e3w_0*(1+r3t) -- "
          f"max rel {_rel.max():.4e}, rms rel "
          f"{np.sqrt((_rel ** 2).mean()):.4e}")
    if plant_e3w:
        print("  E3W PLANT ACTIVE: the witness arm is fed NEMO's OWN e3w "
              "while labelled the pre-fix midpoint; it MUST then NOT "
              "reproduce the pre-fix ratio")
        e3w_witness = jnp.asarray(e3w_nemo)
    else:
        e3w_witness = jnp.asarray(e3w_mid)

    # The stretch's TIME LEVEL.  NEMO reads r3t(Kmm) = Nnn, the step-entry
    # SSH (traldf_iso.f90:285, :831-833; stpmlf.f90:504 calls tra_ldf with
    # Kmm=Nnn).  legoESM threads the jacobian built from state_new.eta, which
    # is Naa -- the SSH AFTER the barotropic step.  The card already carries
    # the Nnn SSH for the flux face thicknesses (redi_flux_eta), so this is a
    # time-level mismatch on one operand, not a missing quantity.  Arm T1
    # measures it; it was found by the diff review, not predicted here.
    from legoesm.ocean.eos import nemo_r3t_stretch
    _stretch_nnn = np.asarray(nemo_r3t_stretch(
        z, jnp.asarray(np.asarray(st.eta.data)),
        jnp.asarray(np.asarray(st.H_bathy.data))))
    print(f"  stretch time level: the operator's jacobian (Naa) spans "
          f"[{jac.min():.9f}, {jac.max():.9f}]; NEMO's (1+r3t(Kmm=Nnn)) "
          f"spans [{_stretch_nnn.min():.9f}, {_stretch_nnn.max():.9f}]")
    e3w_nnn = np.asarray(raw_e3w0)[..., :z.n_levels] * _stretch_nnn[:, :, None]

    arms = [("A0 as the model runs it", {}),
            ("W1 pre-fix midpoint e3w", {"msc_e3w_override": e3w_witness}),
            ("T1 e3w at Kmm=Nnn", {"msc_e3w_override": jnp.asarray(e3w_nnn)})]
    if nat is not None:
        arms.append(("A1 + NEMO's own slopes",
                     {"native_slopes": tuple(jnp.asarray(a) for a in nat)}))
    if plant_slopes and nat is not None:
        print("  SLOPE PLANT ACTIVE: arm A1 is fed the CARD's own slopes "
              "while labelled NEMO's; A1 MUST equal A0 bit-for-bit")
        arms[-1] = ("A1 + NEMO's own slopes",
                    {"native_slopes": kw_T.get("native_slopes")})

    print(f"    {'arm':26s}{'T ratio':>12s}{'T res rms':>12s}"
          f"{'S ratio':>12s}{'S res rms':>12s}")
    results = {}
    for lbl, over in arms:
        row = [lbl]
        for tag, idx, key in (("T", iT, "ttrd_ldf"), ("S", iS, "strd_ldf")):
            args, kw = calls[idx][2], calls[idx][3]
            tend, diags = _replay(real, args, kw, **over)
            nemo = np.nan_to_num(O3(key))
            m = wet3
            den = float(nemo[m] @ nemo[m])
            ratio = float(tend[m] @ nemo[m]) / den if den else float("nan")
            res = float(np.sqrt(np.mean((tend[m] - nemo[m]) ** 2)))
            row += [ratio, res]
            results[(lbl, tag)] = (ratio, res, tend, diags)
        print(f"    {row[0]:26s}{row[1]:12.6f}{row[2]:12.4e}"
              f"{row[3]:12.6f}{row[4]:12.4e}")

    # CONTROL: A0 must be the call stage 4 scored, bit-for-bit.
    a0T = results[(arms[0][0], "T")][2]
    d0 = float(np.max(np.abs(a0T - calls[iT][1])))
    print(f"  control: A0 replay vs the captured stage-4 tendency, "
          f"max|diff| = {d0:.3e} K/s (bar 0.0)")
    if d0 != 0.0:
        print("  ^^ the replay is not the scored call; no arm above is "
              "about the same operator")
        bad += 1

    # WITNESS: restoring the interface-midpoint e3w must put the operator
    # back on the pre-fix ratio (0.996571 T / 0.996696 S, PR #1728).  A gate
    # that passes without this cannot tell the fix from the weather.
    wT_ratio = results[("W1 pre-fix midpoint e3w", "T")][0]
    wS_ratio = results[("W1 pre-fix midpoint e3w", "S")][0]
    print(f"  e3w witness: midpoint arm gives T {wT_ratio:.6f} / "
          f"S {wS_ratio:.6f} against the recorded pre-fix "
          "0.996571 / 0.996696")
    _hit = (abs(wT_ratio - 0.996571) < 2e-5 and abs(wS_ratio - 0.996696) < 2e-5)
    if plant_e3w:
        if _hit:
            print("  ^^ the planted (correct) e3w reproduced the PRE-FIX "
                  "ratio; the witness is not reading the operand it names")
            bad += 1
        else:
            print("  the planted e3w did not reproduce the pre-fix ratio, "
                  "as required")
    elif not _hit:
        print("  ^^ the pre-fix operand no longer reproduces the pre-fix "
              "ratio; something OTHER than e3w moved between the two")
        bad += 1

    # ---- per-stage table, and the akz branch census -------------------
    for tag, idx, key in (("T", iT, "ttrd_ldf"), ("S", iS, "strd_ldf")):
        args, kw = calls[idx][2], calls[idx][3]
        tend, diags = results[(arms[0][0], tag)][2:4]
        parts = _stage_split(diags, args, kw, z)
        tot = parts["horizontal"] + parts["skew"] + parts["a33"]
        clo = float(np.max(np.abs(tot - tend)[wet3]))
        sc = float(np.sqrt(np.mean(tend[wet3] ** 2)))
        print(f"  {tag} stage split closure: max|Sigma stages - tendency| = "
              f"{clo:.3e} against tendency rms {sc:.3e}")
        if clo > 1e-8 * max(sc, 1e-300):
            print("  ^^ the stage split does not close; it is bookkeeping, "
                  "not attribution (Rule 5)")
            bad += 1
        # Closure alone is near-tautological: swapping the skew and a33
        # LABELS leaves it exact and inverts the attribution (diff review
        # finding 7).  So each stage is also rebuilt from the operands that
        # DEFINE it -- the A33 flux from traldf_iso.f90:284-287's own
        # factors -- and required to match the piece it claims to be.
        _ops = diags["zfw_operands"]
        _a33_rebuilt = (np.asarray(_ops["e1e2t"])[:, :, None]
                        / np.asarray(_ops["e3w_kp1"])
                        * (np.asarray(_ops["ah_wslp2"])
                           - np.asarray(_ops["akz"]))
                        * (np.asarray(_ops["qdiff_kp1"])
                           * np.asarray(_ops["wmask"])[..., :]))
        _a33_named = np.asarray(_ops["a33_current"])
        # qdiff_kp1 is the RAW difference; the operator multiplies the
        # wmask-carrying zdkt_kp1, so compare on the wmask-alive cells only.
        _wm1 = np.roll(np.asarray(_ops["wmask"]), -1, 2) > 0.0
        _sel = _wm1 & wet3
        _lab = float(np.max(np.abs(_a33_rebuilt[_sel] - _a33_named[_sel])))
        _scale = float(np.max(np.abs(_a33_named[_sel]))) or 1.0
        print(f"  {tag} stage LABEL check: the piece called 'a33' rebuilt "
              f"from e1e2t/e3w*(ah_wslp2-akz)*(T(k)-T(k+1)) differs by "
              f"{_lab:.3e} against its own max {_scale:.3e}")
        if _lab > 1e-10 * _scale:
            print("  ^^ the stage labelled a33 is not the A33 flux; the "
                  "attribution table above is mislabelled")
            bad += 1
        nemo = np.nan_to_num(O3(key))
        print(f"    {'stage':14s}{'rms':>12s}{'share of |tend|':>18s}"
              f"{'corr with NEMO resid':>22s}")
        resid = (tend - nemo)[wet3]
        for nm in ("horizontal", "skew", "a33"):
            aa = parts[nm][wet3]
            rr = float(np.sqrt(np.mean(aa ** 2)))
            dd = np.sqrt(float(aa @ aa) * float(resid @ resid))
            cc = float(aa @ resid) / dd if dd else float("nan")
            print(f"    {nm:14s}{rr:12.4e}{rr / sc:18.4f}{cc:22.4f}")
        if tag == "T":
            ops = diags["zfw_operands"]
            ah = np.asarray(ops["ah_wslp2"])
            ak = np.asarray(ops["akz"])
            on = (ak > 0.0) & wet3
            print(f"    akz branch census: akz>0 (STABILISED, A33 flux "
                  f"PROPORTIONAL to e3w) on {int(on.sum())}/"
                  f"{int(wet3.sum())} wet cells; akz==0 (A33 flux INVERSE "
                  f"in e3w) on {int((~on & wet3).sum())}. max ah_wslp2 = "
                  f"{np.abs(ah[wet3]).max():.4e}")
            if not on.any():
                # Diff review finding 1, and it is a REAL hole: with akz
                # identically zero, traldf_iso_a33's OWN e3w (the ze3w_2 of
                # traldf_iso.f90:831-833) is multiplied by nothing this
                # record can see, and so is the implicit K33 half of the
                # split.  Only the EXPLICIT divisor (traldf_iso.f90:285) is
                # scored above.  Stated as an UNMEASURED row rather than
                # left to be inferred from the census.
                print("    UNMEASURED: akz is identically zero on this "
                      "record, so traldf_iso_a33's own e3w (ze3w_2, "
                      "traldf_iso.f90:831-833) and the implicit K33 half of "
                      "the split are NOT scored by any row above. A record "
                      "that fires the stabiliser is needed; the reviewer "
                      "planted e3w2*4 at both a33 call sites and every row "
                      "of this gate was byte-identical.")
                bad += 1

    if plant_slopes and nat is not None:
        dT = float(np.max(np.abs(results[arms[-1][0], "T"][2] - a0T)))
        print(f"  SLOPE PLANT: max|A1 - A0| = {dT:.3e} (bar 0.0)")
        if dT != 0.0:
            print("  ^^ feeding the card's own slopes changed the arm, so "
                  "the slope substitution is not what it claims")
            bad += 1
    return bad


# ----------------------------------------------------------------- stages 1/2
def stages12(R: dict, plant_shift: bool,
             plant_ldf_call: bool = False,
             plant_e3w: bool = False,
             plant_slopes: bool = False) -> int:
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
        # jpk-2 = 34 interior interfaces plus a zero tail; NEMO's index 35
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
        # args/kwargs retained so STAGE 5 can replay this EXACT call with one
        # operand substituted -- the model's own path, one variable at a time.
        calls.append((np.asarray(q), np.asarray(out), (q,) + tuple(a),
                      dict(kw)))
        return r

    gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = spy
    try:
        with jax.disable_jit():
            after2 = model.step(st, dt=DT, surface_forcing=sf_step,
                                external_tracer_rate=rate)
    finally:
        gmmod.nemo_iso_lap_tracer_tendency_latlon_cgrid = real
    # CONTROL (review finding): this is a SECOND, un-jitted step.  The
    # tendency captured from it can only explain the residual measured on the
    # FIRST step if the two steps agree.  Checked, not assumed.
    _dT = float(np.max(np.abs(np.asarray(after2.T.data)
                              - np.asarray(after.T.data))))
    _dS = float(np.max(np.abs(np.asarray(after2.S.data)
                              - np.asarray(after.S.data))))
    # The bar is stated, not "zero": turning jit off changes the operation
    # order, so the two steps agree to fp64 roundoff rather than bit-for-bit.
    # What matters is whether that gap can reach the effect being explained,
    # so it is required to sit at least 100x BELOW the residual this stage is
    # about (T rms 1.837e-09 K/s x 2700 s = 4.96e-06 K).
    _margin = 4.96e-06 / 100.0
    print(f"  control: the un-jitted capture step vs the scored step -- "
          f"max|dT| = {_dT:.3e} K, max|dS| = {_dS:.3e} psu, against a "
          f"{_margin:.2e} K bar (1/100 of the residual being explained)")
    if _dT > _margin:
        print("  ^^ the capture step is far enough from the scored step to "
              "matter; the row below would be about a different run")
        bad += 1
    if len(calls) != 4:
        print(f"  UNMEASURED: the isoneutral tendency was called "
              f"{len(calls)} times, expected 4 (T,S x Nnn,Nbb pass)")
        return bad + 1
    # WHICH CALL IS NEMO's tra_ldf?  A review of the first version of this
    # probe found its two comments contradicted each other about the call
    # structure and that nothing PROVED which call was scored.  So the
    # structure is now printed per call and BOTH candidates are scored: if
    # they give the same ratio, the identification cannot change the answer,
    # and if they do not, the disagreement is on the page instead of hidden
    # behind a position.
    Tin, Sin = np.asarray(st.T.data), np.asarray(st.S.data)
    Tbef = np.asarray(st.T_before.data) if st.T_before is not None else None
    Sbef = np.asarray(st.S_before.data) if st.S_before is not None else None
    print("  call structure (max|q - x| on wet cells, per captured call):")
    for i, (q, _, _a, _k) in enumerate(calls):
        bits = []
        for nm, ref in (("T", Tin), ("T_before", Tbef),
                        ("S", Sin), ("S_before", Sbef)):
            if ref is not None and q.shape == ref.shape:
                bits.append(f"{nm}={float(np.max(np.abs((q - ref)[wet3]))):.3e}")
        print(f"    call {i}: " + "  ".join(bits))
    wT = [i for i, (q, _, _a, _k) in enumerate(calls)
          if q.shape == Tin.shape and np.array_equal(q[wet3], Tin[wet3])]
    wS = [i for i, (q, _, _a, _k) in enumerate(calls)
          if q.shape == Sin.shape and np.array_equal(q[wet3], Sin[wet3])]
    print(f"  exact wet-masked match against state.T / state.S: "
          f"T->{wT}  S->{wS}")
    if not wT or not wS:
        print("  UNMEASURED: no captured call saw state.T (or state.S), so "
              "NEMO's Kbb-evaluated tra_ldf is not among them")
        return bad + 1
    iT, iS = wT[-1], wS[-1]
    # The plant's bar is the ratio the CORRECT call gives on THIS commit,
    # measured here, never a constant copied from a previous round -- a
    # hardcoded 0.9966 silently went stale the moment the e3w fix landed.
    _true_ratio = {}
    for tag, idx, key in (("T", iT, "ttrd_ldf"), ("S", iS, "strd_ldf")):
        _n = np.nan_to_num(O3(key))
        _l = calls[idx][1]
        _d = float(_n[wet3] @ _n[wet3])
        _true_ratio[tag] = float(_l[wet3] @ _n[wet3]) / _d if _d else float("nan")
    if plant_ldf_call:
        # The Nnn advective pass, which is NOT what NEMO's tra_ldf computes.
        wT = [i for i in range(len(calls)) if i not in wT and i not in wS][:1]
        wS = wT
        print(f"  LDF-CALL PLANT ACTIVE: scoring call {wT} instead, which "
              "is the Nnn advective pass; the ratio MUST move materially")

    _ratios = []
    for lbl, cands, oracle_key in (("T", wT, "ttrd_ldf"),
                                   ("S", wS, "strd_ldf")):
      for idx in cands:
        lego = calls[idx][1]
        nemo = np.nan_to_num(O3(oracle_key))
        if lego.shape != nemo.shape:
            n = min(lego.shape[-1], nemo.shape[-1])
            lego, nemo = lego[..., :n], nemo[..., :n]
            m = wet3[..., :n]
        else:
            m = wet3
        st_ = _stats(lego, nemo, m)
        print(_row(f"{lbl} tra_ldf [call {idx}]", st_))
        den = float(nemo[m] @ nemo[m])
        if den:
            _r = float(lego[m] @ nemo[m]) / den
            _ratios.append(_r)
            print(f"      best-fit lego/NEMO ratio = {_r:.9f}")
        if st_["ne"] and idx == cands[-1]:
            bad += 1
    if plant_ldf_call:
        # ANY tracer moving is enough, and the first run of this plant showed
        # why "all" would have been the wrong bar: scoring the Nnn advective
        # pass moves SALINITY from 0.9967 to 5.53 but leaves TEMPERATURE at
        # 0.9965, because the two passes' T inputs differ by only 1.5e-2 K out
        # of a ~20 K field while their S inputs differ enough to matter.  So
        # the identification is load-bearing (S proves it) AND the temperature
        # row happens not to depend on it -- both facts are printed rather
        # than collapsed into one verdict.
        moved = [abs(r - t) > 0.01 for r, t in
                 zip(_ratios, (_true_ratio["T"], _true_ratio["S"]))]
        print(f"  LDF-CALL PLANT: ratios {['%.6f' % r for r in _ratios]} "
              f"against the correct call's "
              f"{['%.6f' % _true_ratio[t] for t in ('T', 'S')]}; "
              f"moved: {moved}")
        if not any(moved):
            print("  ^^ scoring the wrong call gives the same answer on EVERY "
                  "tracer, so the identification certifies nothing")
            bad += 1
        else:
            print("  the identification is load-bearing on at least one "
                  "tracer, and it is made by an EXACT wet-masked input match "
                  "(max|q - state.T| = 0.000e+00), not by a call position")
    if plant_ldf_call:
        return bad
    bad += stage5(calls, iT, iS, R, O3, wet3, z, st, real, plant_e3w,
                  plant_slopes)

    # ---- STAGE 6: tra_qsr, scored operator-to-operator ----------------
    # Temperature's remaining explainer was PLAUSIBLE (a regression on NEMO's
    # near-orthogonal operators).  This turns it into a number the same way
    # stage 4 did for tra_ldf: capture legoESM's OWN penetrative-solar
    # tendency during the SAME forcing call and score it against NEMO's
    # ``ttrd_qsr``.  NEMO: stpmlf.f90:505 CALL tra_qsr(kstp, Nnn, ts, Nrhs).
    print("\nSTAGE 6 -- tra_qsr (penetrative solar), legoESM's own "
          "tendency vs NEMO's ttrd_qsr")
    # The DINO card imports the shared Jerlov kernel into its own namespace
    # (experiments/dino.py:4712), so THAT is the name the step resolves.
    ffmod = dm
    _real_sw = ffmod.shortwave_penetration_tendency
    sw_calls = []

    def _sw_spy(*a, **kw):
        r = _real_sw(*a, **kw)
        sw_calls.append(np.asarray(r))
        return r

    ffmod.shortwave_penetration_tendency = _sw_spy
    try:
        dm.apply_dino_lat_lon_surface_forcing(
            state0, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
    finally:
        ffmod.shortwave_penetration_tendency = _real_sw
    if len(sw_calls) != 1:
        print(f"  UNMEASURED: the solar kernel was called {len(sw_calls)} "
              "times, expected 1")
        bad += 1
    else:
        lego_qsr = sw_calls[0]
        nemo_qsr = np.nan_to_num(O3("ttrd_qsr"))
        if lego_qsr.shape != nemo_qsr.shape:
            print(f"  UNMEASURED: shapes differ {lego_qsr.shape} vs "
                  f"{nemo_qsr.shape}")
            bad += 1
        else:
            st_ = _stats(lego_qsr, nemo_qsr, wet3)
            print(_row("tra_qsr", st_))
            den = float(nemo_qsr[wet3] @ nemo_qsr[wet3])
            if den:
                print("      best-fit lego/NEMO ratio = "
                      f"{float(lego_qsr[wet3] @ nemo_qsr[wet3]) / den:.9f}")
            # The step-1 temperature residual this term has to explain.
            print(f"      x rDt={DT}: rms {st_['rms'] * DT:.4e} K against "
                  "the step-1 T residual of 4.791e-06 K")
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
    ap.add_argument("--plant-ldf-call", action="store_true",
                    help="score stage 4 against the WRONG captured call; "
                         "the ratio MUST move materially")
    ap.add_argument("--plant-e3w", action="store_true",
                    help="feed stage 5's e3w arm the MIDPOINT thickness "
                         "while labelling it NEMO's; the arm MUST NOT "
                         "improve")
    ap.add_argument("--plant-slopes", action="store_true",
                    help="feed stage 5's slope arm the CARD's own slopes "
                         "while labelling them NEMO's; the arm MUST equal "
                         "A0 bit-for-bit")
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
    bad += stages12(R, args.plant_shift, args.plant_ldf_call,
                    args.plant_e3w, args.plant_slopes)
    print(f"\n{'GATE PASS' if bad == 0 else f'GATE FAIL ({bad} rows)'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
