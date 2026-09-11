#!/usr/bin/env python
"""The SURFACE row of the kt=1 step-1 residual, operator to operator.

WHY.  The step-1 T residual (4.7912e-06 K rms) is essentially all SURFACE:
level 0 carries 1.041e-08 K/s against 4.67e-10 below, and pooling over the
levels reproduces the measured state residual.  The isoneutral operator is
excluded (2.7 ulp), so the next owner is the surface heat application.

WHAT NEMO DOES, as COMPILED (``cfgs/DINO/BLD/ppsrc/nemo/trasbc.f90``):

  * ``sbc_tsc(:,:,jp_tem) = r1_rho0_rcp * qns``  -- NON-SOLAR heat only;
    ``sbc_tsc(:,:,jp_sal) = r1_rho0 * sfx``      -- salt from freeze/melt.
  * the concentration/dilution ``emp`` term is inside ``IF( lk_linssh )``.
    DINO runs ``key_qco`` (non-linear free surface), so **emp never enters
    tra_sbc on this card** -- and ``sfx`` is zero with no ice.
  * the RHS update is
    ``pts(:,:,1,jn,Krhs) += zfact*(sbc_tsc_b + sbc_tsc)
                            / ( e3t_3d(ji,jj,1) * (1 + r3t(ji,jj,Kmm)*tmask) )``
    -- the REFERENCE first thickness times the LIVE ``Kmm`` stretch, i.e. the
    same step-entry height ``tra_ldf`` indexes.
  * ``zfact = 1`` and ``sbc_tsc_b = 0`` on the no-restart Euler start, and
    ``0.5`` with the swapped ``sbc_tsc_b`` afterwards.  So kt=1 applies the
    WHOLE now-flux and every later step applies the two-step average --
    another statement the from-rest record can see only half of.
  * the SOLAR part is not here at all: ``tra_qsr`` applies it with the
    ``fraqsr_1lev`` split, and its trend is ``ttrd_qsr``.

SO THE FREE CROSS-CHECK (it is why this gate scores S first): salinity has no
solar member, so ``strd_nsr`` is the WHOLE surface salt trend.  A residual
there names the SHARED surface path; a residual only on T names the heat/solar
side.

RULE 10.  The tendency scored is the one the card's own applicator returns to
the driver -- ``apply_dino_lat_lon_surface_forcing(..., return_rate=True)``,
the exact object ``run_dino.py`` threads into ``model.step`` as
``external_tracer_rate`` -- not a reconstruction.

WHAT THIS GATE CANNOT SEE (Rule 2, and a reviewer demonstrated it).  It scores
the APPLICATOR'S RETURNED RATE and never calls ``model.step``.  A change to
what the model DOES with that rate -- a factor planted where
``external_tracer_rate`` is consumed, the backward-Euler vertical solve's
damping of the level-0 increment -- leaves every row below byte-identical.
Two consequences, both stated rather than hidden: the ACCOUNTING line at the
end is a rate integrated as ``rate * dt``, which is NOT how the step applies
it; and the gap between that number and the step-1 gate's is the size of
everything this gate is blind to.

RECONCILED WITH THE BAR GATE (Rule 1e).  ``fidelity_bar_gate.py`` records
``tra_sbc`` at (1.00000000, 1.00000000), residual 1.936e-16.  That is NOT a
disagreement with the rows below: ``coverage_rows_measure.py::measure_tra_sbc``
builds its comparison with ``RestoringConfig(implicit=False)`` and says so in
its own comment -- "an existing, deliberate divergence from production's
implicit=True".  So the bar-gate row measures NEMO's formula reconstructed
WITHOUT the implicit denominator, and is green precisely because it turns off
the statement this gate finds.  Neither number is revised here; the two
measure different objects, and the bar gate's blind spot is now written down.

Usage
-----
    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
      python scripts/validate/ocean_fidelity/dino_1226/kt1_surface_gate.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from rebuild_nemo_restart import rebuild                        # noqa: E402

DT = 2700.0
DEFAULT_RESTART = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
                   "RUN_FROMREST_KT1/DINO_00000001_restart_*.nc")
TRENDS = ("ttrd_nsr", "strd_nsr", "ttrd_qsr", "ttrd_dmp", "strd_dmp")
OPERANDS = ("sbc_hc_b", "sbc_sc_b", "qsr_hc_b", "fraqsr_1lev", "emp_b",
            "sfx_b", "qns_b")


def _row(name, lego, nemo, wet):
    d = np.abs(np.asarray(lego) - np.asarray(nemo))[wet]
    n = int((d != 0.0).sum())
    a, b = np.asarray(lego)[wet], np.asarray(nemo)[wet]
    den = float(b @ b)
    ratio = float(a @ b) / den if den else float("nan")
    print(f"  {name:26s}{int(wet.sum()):>9d}{n:>9d}{d.max():13.4e}"
          f"{float(np.sqrt(np.mean(d ** 2))):13.4e}"
          f"{float(np.sqrt(np.mean(b ** 2))):13.4e}{ratio:13.9f}  "
          f"{'AT BAR' if n == 0 else 'DEBT'}")
    return 0 if n == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--restart-glob", default=DEFAULT_RESTART)
    ap.add_argument("--kt2-dir", default=None,
                    help="the kt=2 trend record; adds the SIZE of the two "
                         "registered surface statements (the flux's time "
                         "level and NEMO's two-step average), read off "
                         "NEMO's own sbc_hc_b/sbc_sc_b at kt=1 and kt=2")
    ap.add_argument("--plant", action="store_true",
                    help="move one wet level-0 rate cell by 1 ulp; on the "
                         "self-test arm the gate MUST then fail")
    ap.add_argument("--oracle-self-test", action="store_true",
                    help="REPLACE legoESM's rate with NEMO's own trend. "
                         "Every row must read AT BAR.  Without this arm "
                         "--plant proves nothing, because the real card is "
                         "not at the bar and the gate fails either way.")
    a = ap.parse_args()

    R = rebuild(a.restart_glob, list(TRENDS) + list(OPERANDS))
    missing = [k for k in TRENDS if k not in R]
    if missing:
        raise SystemExit(f"the restart carries no {missing}: ln_tra_trd off?")

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                          # Rule 1c
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm
    from legoesm import constants

    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    state0 = dm.dino_lat_lon_state(grid, z, cfg)
    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    _, rate = dm.apply_dino_lat_lon_surface_forcing(
        state0, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
    dT = np.asarray(rate[0], dtype=np.float64)
    dS = np.asarray(rate[1], dtype=np.float64)

    def _O3pre(k):
        return np.nan_to_num(np.moveaxis(R[k], 0, -1))

    if a.oracle_self_test:
        print("SELF-TEST: the legoESM side is REPLACED by NEMO's own trends. "
              "This measures the GATE, not the model, and must never be "
              "reported as a fidelity result.")
        dT = (_O3pre("ttrd_nsr") + _O3pre("ttrd_qsr")).copy()
        dS = _O3pre("strd_nsr").copy()
        # The sbc_hc_b row compares against the NON-SOLAR half alone, so on
        # the self-test arm its legoESM side must be the non-solar half too --
        # otherwise the arm reports a failure that is the ARM's construction,
        # not the gate's, and the plant below has nothing clean to break.
        dT_nsr_self = _O3pre("ttrd_nsr").copy()
    if a.plant:
        dT = dT.copy()
        dT[100, 25, 0] = np.nextafter(dT[100, 25, 0], np.inf)
        print("PLANT ACTIVE: one wet level-0 T rate cell moved 1 ulp; the "
              "gate MUST fail")

    g = ndm.nemo_dino_mesh()
    wet3 = g.tmask > 0.5

    def O3(k):
        return np.nan_to_num(np.moveaxis(R[k], 0, -1))

    # What NEMO's own surface buckets contain, BEFORE anything is compared to
    # them: a bucket that is identically zero is not evidence of agreement.
    print("NEMO's surface trend buckets at kt=1 (wet rms, nonzero cells)")
    for k in TRENDS:
        v = O3(k)
        print(f"  {k:12s} rms {float(np.sqrt(np.mean(v[wet3] ** 2))):11.4e}  "
              f"nonzero {int((v[wet3] != 0).sum()):>8d}  "
              f"level-0 rms "
              f"{float(np.sqrt(np.mean(v[..., 0][wet3[..., 0]] ** 2))):11.4e}")
    for k in OPERANDS:
        if k in R:
            v = np.nan_to_num(np.asarray(R[k]))
            print(f"  {k:12s} |max| {np.abs(v).max():11.4e}  "
                  f"nonzero {int((v != 0).sum()):>8d}")

    print("\nlegoESM's surface tracer rate (the object run_dino threads into "
          "model.step) vs NEMO's own trends")
    print(f"  {'row':26s}{'cells':>9s}{'!=':>9s}{'max|d|':>13s}{'rms':>13s}"
          f"{'NEMO rms':>13s}{'ratio':>13s}")
    bad = 0
    # SALINITY FIRST: no solar member, so strd_nsr is the whole surface salt
    # trend and this row is about the SHARED path only.
    bad += _row("S: rate vs strd_nsr", dS, O3("strd_nsr"), wet3)
    bad += _row("S: level 0 only", dS[..., :1], O3("strd_nsr")[..., :1],
                wet3[..., :1])
    tot = O3("ttrd_nsr") + O3("ttrd_qsr")
    bad += _row("T: rate vs nsr+qsr", dT, tot, wet3)
    bad += _row("T: level 0 only", dT[..., :1], tot[..., :1], wet3[..., :1])
    bad += _row("T: sub-surface only", dT[..., 1:], tot[..., 1:], wet3[..., 1:])
    # and the two halves separately, so a split that is right in total but
    # wrong per bucket is visible
    _row("T: rate vs ttrd_nsr alone", dT, O3("ttrd_nsr"), wet3)
    _row("T: rate vs ttrd_qsr alone", dT, O3("ttrd_qsr"), wet3)

    # The OPERAND row: NEMO stores sbc_tsc itself.  trasbc.f90 sets
    # sbc_tsc(jp_tem) = r1_rho0_rcp*qns, and at kt=1 (zfact=1, sbc_tsc_b=0)
    # the applied level-0 trend is exactly sbc_tsc / (e3t_3d(:,:,1)*(1+r3t)).
    # From rest eta = 0, so r3t = 0 and the divisor is the REFERENCE dz -- the
    # one place legoESM's dz_ref[0] and NEMO's live height cannot disagree.
    if "sbc_hc_b" in R:
        e3t0 = float(np.asarray(z.dz_ref)[0])
        implied = np.nan_to_num(np.asarray(R["sbc_hc_b"])) / e3t0
        print(f"\n  NEMO's own stored sbc_hc_b / e3t(1)={e3t0:.9f} m, which "
              "at kt=1 IS the level-0 non-solar trend (zfact=1, sbc_tsc_b=0):")
        _lhs0 = (dT_nsr_self[..., 0] if a.oracle_self_test else dT[..., 0])
        bad += _row("level 0: vs sbc_hc_b/e3t", _lhs0,
                    implied, wet3[..., 0])
        print(f"    implied qns = rho0*cp*sbc_hc_b: |max| "
              f"{float(np.abs(np.nan_to_num(np.asarray(R['sbc_hc_b'])) * constants.rho_ocean_ref * 3991.86795711963).max()):.4e} W/m2"
              if hasattr(constants, "rho_ocean_ref") else "")

    # ---- THE STATEMENT, named and then CHECKED by arithmetic ------------
    # legoESM builds the surface restoring with RestoringConfig(implicit=True)
    # (dino.py:4674), whose denominator is (tau + dt).  NEMO's tra_sbc has no
    # such damping: sbc_tsc = r1_rho0_rcp*qns with qns at the now level,
    # divided once by the live top thickness (trasbc.f90, quoted above).  So
    # legoESM's level-0 tendency must be SMALLER by exactly tau/(tau+dt).
    #
    # Salinity is the clean test: no solar member, so the whole row is the
    # restoring.  If the measured S ratio equals tau_S/(tau_S+dt) the
    # statement is identified; if it does not, this explanation is dead.
    from legoesm.ocean.physics.surface_forcing.config import (
        tau_from_flux_coefficient)
    dz0 = float(np.asarray(z.dz_ref)[0])
    tau_T = float(tau_from_flux_coefficient(cfg.A_theta, cfg.rho_0, cfg.c_p,
                                            dz0))
    tau_S = float(tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0, dz0))
    b_ = O3("strd_nsr")[..., 0][wet3[..., 0]]
    a_ = dS[..., 0][wet3[..., 0]]
    meas = float(a_ @ b_) / float(b_ @ b_)
    pred = tau_S / (tau_S + DT)
    print("\n  THE STATEMENT, AND WHETHER IT IS STILL THERE: an implicit-Euler"
          " restoring denominator (tau + dt).  NEMO has none -- its flux is "
          "evaluated on the before tracer (usrdef_sbc.f90:223, :272-273) and "
          "applied with a single live-thickness division (trasbc.f90:169-170) "
          "-- and the DINO applicator moved to that explicit form in PR #1728 "
          "(user decision 30).  This block now discriminates BOTH ways: a "
          "measured ratio of 1 means the denominator is gone, a measured ratio "
          "equal to tau/(tau+dt) means it is back.")
    print(f"    tau_S = rho_0*dz_0/A_S = {tau_S:.6e} s "
          f"({tau_S / 86400.0:.4f} d),  dt = {DT} s")
    print(f"    predicted S ratio tau_S/(tau_S+dt) = {pred:.9f}")
    print(f"    MEASURED  S ratio                  = {meas:.9f}")
    orth = float(np.sqrt(np.mean((a_ - pred * b_) ** 2)))
    print(f"    residual AFTER removing the predicted factor, "
          f"rms(lego - pred*nemo) = {orth:.4e} K/s "
          f"({orth / float(np.sqrt(np.mean(b_ ** 2))):.3e} of NEMO's own rms)"
          " -- the ratio alone cannot see structure; this can.")
    if abs(meas - 1.0) < 1e-12:
        _verdict = ("REMOVED: the measured ratio is 1 to 1e-12, so NEMO's "
                    "explicit form is in place.  This is the EXPECTED reading "
                    "on the current model.")
    elif abs(pred - meas) < 1e-6:
        _verdict = ("BACK: the level-0 surface deficit IS the implicit "
                    "denominator again -- a regression against "
                    "trasbc.f90:169-170.")
    else:
        _verdict = ("NEITHER: the ratio is neither 1 nor tau/(tau+dt), so the "
                    "level-0 salt deficit is some OTHER statement and this "
                    "block no longer names it.")
    print(f"    |meas - 1| = {abs(meas - 1.0):.3e}, |meas - pred| = "
          f"{abs(pred - meas):.3e}  -> " + _verdict)
    print(f"    tau_T = rho_0*c_p*dz_0/A_theta = {tau_T:.6e} s "
          f"({tau_T / 86400.0:.4f} d); tau_T/(tau_T+dt) = "
          f"{tau_T / (tau_T + DT):.9f} -- the T row also carries the "
          "undamped Q_sr penetration, so its ratio is a BLEND and only the "
          "S row is a clean test of the statement.  Concretely, while the "
          "implicit form was in place the T level-0 ratio measured 0.996917 "
          "against this 0.997406: the Q_sr subtraction never carried the "
          "denominator, so the covariance weights a damped member against an "
          "undamped one.  Anyone reading the T ratio as the factor is reading "
          "a blend.")

    print("\n  per-level residual of the T row (where the surface residual "
          "lives)")
    res = dT - tot
    for k in range(min(6, res.shape[-1])):
        m = wet3[..., k]
        print(f"    k={k:2d}  res rms "
              f"{float(np.sqrt(np.mean(res[..., k][m] ** 2))):11.4e}  "
              f"NEMO rms "
              f"{float(np.sqrt(np.mean(tot[..., k][m] ** 2))):11.4e}")

    # ---- does this statement ACCOUNT for the step-1 state residual? -----
    # The step-1 gate measures 4.7912e-06 K rms over all wet cells.  If this
    # rate residual is the whole story, integrating it over one step and
    # pooling level 0 into the 3-D rms must reproduce that number.  Predicted
    # BEFORE being compared, and it is a prediction that can fail.
    # MOVED with the model, and the old value is kept next to it so the line
    # cannot be read as unchanged: 4.7912e-06 K was the frozen figure while the
    # restoring carried the implicit denominator, and it was re-measured on a
    # deliberate revert arm (same commit, one variable) to confirm this gate's
    # control before the new number was recorded.
    STEP1_T_RMS_K = 1.1760e-07     # PR #1728, nemo_dino_step1_gate.py, NEMO's
                                   # explicit restoring (was 4.7912e-06 with
                                   # the implicit denominator)
    pooled = float(np.sqrt(np.sum((res[wet3]) ** 2) / int(wet3.sum()))) * DT
    print(f"\n  ACCOUNTING: this rate residual integrated as rate*dt and "
          f"pooled over all {int(wet3.sum())} wet cells is {pooled:.4e} K; "
          f"the step-1 gate measures {STEP1_T_RMS_K:.4e} K "
          f"(ratio {pooled / STEP1_T_RMS_K:.4f}).")
    print("    That literal is FROZEN (PR #1728, nemo_dino_step1_gate.py), so "
          "this line compares a fresh number against a recorded one; re-run "
          "the step-1 gate if the model has moved.  And rate*dt is NOT how "
          "the step applies it -- the level-0 increment also passes through "
          "the backward-Euler vertical solve, which damps it.  With the "
          "restoring explicit the surface RATE residual is at rounding while "
          "the step-1 STATE residual is not, so a ratio far below 1 here now "
          "means the remaining step-1 residual is owned by something OTHER "
          "than the surface rate -- which is the finding, not a defect in "
          "this line.")

    # ---- SIZE of the two registered surface statements (PR #1728) --------
    # Both are LAGS, and both are visible in NEMO's own restart without any
    # model run.  trasbc.f90:145-150 swaps sbc_tsc into sbc_tsc_b at every
    # kt > nit000 and sets zfact = 0.5, so from the SECOND step NEMO applies
    #     0.5*( sbc_tsc_b + sbc_tsc ) / e3t(1)      (trasbc.f90:169-170)
    # while legoESM applies its own single evaluation at weight 1.  If the two
    # models' instantaneous fluxes agreed exactly, the whole difference would
    # be   0.5*( sbc_tsc(kt) - sbc_tsc(kt-1) ) / e3t(1),
    # i.e. HALF ONE STEP's change in the surface flux.  The restart's
    # sbc_hc_b/sbc_sc_b hold sbc_tsc as it was at the end of each step
    # (trasbc.f90:175-176), so kt=1's and kt=2's give exactly that difference.
    if a.kt2_dir:
        import glob as _glob
        import os as _os
        _k1 = _os.path.join(_os.path.dirname(a.restart_glob.rstrip("/")),
                            "DINO_00000001_restart_*.nc")
        if not _glob.glob(_k1):
            _k1 = a.restart_glob
        _k2 = _os.path.join(a.kt2_dir, "DINO_00000002_restart_*.nc")
        if not _glob.glob(_k2):
            print(f"\n  SURFACE LAG SIZE: UNMEASURED -- no tiles match {_k2}")
            bad += 1
        else:
            R1s = rebuild(_k1, ["sbc_hc_b", "sbc_sc_b"])
            R2s = rebuild(_k2, ["sbc_hc_b", "sbc_sc_b"])
            e3t0 = float(np.asarray(z.dz_ref)[0])
            RDT = 2.0 * DT          # NEMO's leap-frog rDt at kt=2
            print("\n  SIZE of the two registered surface statements, read "
                  "off NEMO's own sbc_hc_b/sbc_sc_b at kt=1 and kt=2")
            print(f"    {'tracer':8s}{'0.5*d(sbc)/e3t rms':>22s}"
                  f"{'x rDt [K or psu]':>20s}{'pooled 3-D':>14s}")
            _n3 = int(wet3.sum())
            _sizes = {}
            for tag, key in (("T", "sbc_hc_b"), ("S", "sbc_sc_b")):
                d1 = np.nan_to_num(np.asarray(R1s[key], dtype=np.float64))
                d2 = np.nan_to_num(np.asarray(R2s[key], dtype=np.float64))
                lag = 0.5 * (d2 - d1) / e3t0          # [K/s] at level 0
                m2 = wet3[..., 0]
                rms = float(np.sqrt(np.mean(lag[m2] ** 2)))
                per_step = rms * RDT
                pooled = float(np.sqrt(
                    np.sum((lag[m2] * RDT) ** 2) / _n3))
                _sizes[tag] = pooled
                print(f"    {tag:8s}{rms:22.4e}{per_step:20.4e}{pooled:14.4e}")
            # The kt=2 gate's STATE rows are the thing this has to explain.
            KT2_T_RMS_K = 6.1224e-06   # kt2_leapfrog_gate.py, PR #1728
            KT2_S_RMS = 1.1380e-05
            print(f"    the kt=2 gate measures T {KT2_T_RMS_K:.4e} K and "
                  f"S {KT2_S_RMS:.4e} psu of STATE residual after one step")
            print(f"    ratio lag/state:  T {_sizes['T'] / KT2_T_RMS_K:.3f}"
                  f"   S {_sizes['S'] / KT2_S_RMS:.3f}")
            print("    A ratio near 1 means these two statements ACCOUNT for "
                  "the kt=2 state residual and are the next owner; a ratio "
                  "far below 1 means something else owns it.  Both literals "
                  "are FROZEN from that gate, so this compares a fresh "
                  "number against a recorded one.")
            print("    NOT CLOSED HERE: NEMO's average needs sbc_tsc_b, a "
                  "carried field legoESM does not have, and the flux's time "
                  "level needs the Kbb tracer at the forcing call.  Both are "
                  "the user's decision; no kt>=2 surface row may be called "
                  "AT BAR while they stand.")

    print(f"\n{'GATE PASS' if bad == 0 else f'GATE FAIL ({bad} rows)'}")
    if a.oracle_self_test and not a.plant and bad:
        print("  ^^ the SELF-TEST arm must be AT BAR on every row: the gate "
              "cannot reproduce the oracle's own answer, so no number it "
              "prints about the model means anything")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
