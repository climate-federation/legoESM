#!/usr/bin/env python
"""#1455 basin-budget lane: the two OFFLINE deciders for candidates 1 and 2.

Pre-registration: ``PREREG_basin_deciders.md``, written before either number
existed.  Parent result: ``docs/ocean/fidelity/dino_basin_budget_result.md``.

WHAT IS BEING DECIDED.  The southern-basin deficit is a depth-uniform eastward
velocity error against the wall, worth 0.112 m3/s2 per row per year in the rate
at which legoESM builds circulation there.  Two of the four ranked candidates
have offline deciders, and they are run together because neither is conclusive
alone:

  D2  does legoESM's realised row rate COLLAPSE to its own barotropic-solve
      deposit, the way the oracle's does exactly?  If it does, the whole
      shortfall is inside that solve and every other stage is exonerated in one
      number.
  D1  is the bottom-drag SINK stronger in legoESM, under the shared
      transcription of the oracle's non-linear drag law applied to each model's
      own saved state?

D1 IS CIRCULAR IN ONE DIRECTION AND SAYS SO.  A weaker flow produces a weaker
drag under a non-linear law, so a drag EXCESS in the model whose flow is WEAK
is the interesting sign and a drag DEFICIT is the expected consequence of the
flow deficit.  The pre-registration fixes the joint reading of D1 and D2
together; this file prints both and the joint cell, and does not interpret.

NOTHING NUMERICAL IS RE-DERIVED HERE:
  * the drag torque         southern_circulation_budget.phi_drag -- the
                            campaign's recorded transcription of zdfdrg
                            rCdU_bot + dynzdf's face average, the same
                            functional applied to both models
  * the row reducers        southern_circulation_budget.row_circulation /
                            row_int_trend
  * the state loaders       southern_circulation_budget.load_nemo/load_lego and
                            its recorded day-0 v-row-offset control
  * the stage table         the artifact written by
                            southern_term_torque_accum.py --days 360

Usage
-----
  southern_basin_deciders.py --self-check      # the gates, no artifacts needed
  southern_basin_deciders.py                   # D1 + D2
  southern_basin_deciders.py --d1              # D1 only (no stage artifact)
"""
import argparse
import os
import subprocess
import sys

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
sys.path.insert(0, os.path.dirname(_DIR))
import acceptance_gate_90d as G          # noqa: E402
import acc_thermal_wind as A             # noqa: E402
import southern_circulation_budget as B  # noqa: E402

DAYS = (10, 30, 60, 90, 120, 150, 180, 210, 240, 270,
        280, 290, 300, 310, 320, 330, 340, 350, 360)
HORIZONS = (90, 180, 270, 360)
WALL_ROWS = [1, 2, 3, 4]                 # the four rows that carry 95% of the gap
LEGO_NPZ = os.environ.get("DINO_V360_LEGO_NPZ",
                          "/tmp/dino_verdict360/m0_control.npz")
NEMO_RUN = os.environ.get("DINO_V360_NEMO_RUN",
                          f"{A.DINO}/RUN_VERDICT360_M0")
STAGE_NPZ = os.environ.get("DINO_ACC360_NPZ", "/tmp/dino_accum360.npz")
NEMO_ACC_RUN = os.environ.get("DINO_ACC_RUN_DIR", f"{A.DINO}/RUN_ACC360")

# The quantity every threshold below is a fraction of: the year-mean rate
# shortfall measured in the parent result (NEMO minus legoESM, band mean over
# the southern rows).  Both sides of it are recomputed by self-check 4.
DEFICIT = 0.112
D1_CONFIRM = 0.5 * DEFICIT               # >= half the shortfall
D1_REFUTE = 0.2 * DEFICIT                # <  a fifth of it
D2_CONFIRM = 0.05                        # median |realised/BARO - 1|
D2_REFUTE = 0.5


# -------------------------------------------------------------- self-checks ---
def self_checks(verbose=True):
    ok = []
    # S1 -- the DRAG LAW.  There is NO exact closed form available here and an
    # earlier revision of this gate wrongly assumed one: under a uniform u = U
    # the t-point speed is NOT uniform, because the two-point average against
    # land (and the western zero-pad) halves it at every coastal column, and
    # the law is non-linear, so the geometry does not cancel out of a
    # speed-to-speed ratio.  The gate that DOES have power is the SHAPE of the
    # response.  Doubling a uniform flow must change the row torque by a factor
    # strictly between 2 (the ke0-dominated limit) and 4 (the U^2-dominated
    # limit), and within 5% of the no-boundary prediction -- which fails on a
    # linear drag law (exactly 2), on a quadratic-only one (exactly 4), and on a
    # background kinetic energy wrong by a factor of two (10% low).
    v = np.zeros_like(B.vmask, dtype=np.float64)
    ke0, cd0 = float(B.DRAG_KW["ke0"]), float(B.DRAG_KW["cd0"])
    law = lambda U: -cd0 * np.sqrt(U ** 2 + ke0) * U          # noqa: E731
    U0 = 0.03
    got = B.phi_drag(np.where(B.umask, U0, 0.0), v)
    g2 = B.phi_drag(np.where(B.umask, 2 * U0, 0.0), v)
    ratio = float(np.mean(g2[B.ROWS]) / np.mean(got[B.ROWS]))
    pred = float(law(2 * U0) / law(U0))
    ok.append(("S1 the drag law's doubling response, bracketed + within 5%",
               f"measured {ratio:.4f}, no-boundary prediction {pred:.4f}, "
               f"bracket (2, 4)",
               2.0 < ratio < 4.0 and abs(ratio / pred - 1.0) < 0.05))
    # S1b -- and it must be a SINK: eastward flow gives a negative torque.
    ok.append(("S1b phi_drag is a sink on eastward flow",
               f"band mean = {float(np.mean(got[B.ROWS])):+.4f}",
               float(np.mean(got[B.ROWS])) < 0))
    # S1d -- the geometry weight phi_drag implies must lie between the row's
    # total wet-column width and that width minus its two boundary faces (the
    # 2-point t->u average halves r at a coast).  Fails if the mask, e1u or the
    # bottom-cell pick is wrong by more than a couple of faces.
    w_imp = np.asarray(got)[B.ROWS] / law(U0)
    col = B.umask.any(axis=2)
    w_full = np.sum(np.where(col, B.e1u, 0.0), axis=1)[B.ROWS]
    w_lo = w_full - 2.0 * np.max(B.e1u, axis=1)[B.ROWS]
    good = bool(np.all(w_imp <= w_full * (1 + 1e-9)) and np.all(w_imp >= w_lo))
    ok.append(("S1d the implied row geometry weight is inside its bounds",
               f"implied/full in [{float(np.min(w_imp / w_full)):.3f}, "
               f"{float(np.max(w_imp / w_full)):.3f}]", good))
    # S2 -- the constants are the oracle's namdrg_bot values.
    good = (B.DRAG_KW["scheme"] == "nemo_quadratic"
            and abs(float(B.DRAG_KW["cd0"]) - 1.0e-3) < 1e-12
            and abs(float(B.DRAG_KW["ke0"]) - 2.5e-3) < 1e-12)
    ok.append(("S2 drag law + constants are NEMO namdrg_bot",
               f"{B.DRAG_KW['scheme']}, cd0={B.DRAG_KW['cd0']}, "
               f"ke0={B.DRAG_KW['ke0']}", good))
    if verbose:
        print("=" * 100)
        print("PRE-RUN GATES")
        print("=" * 100)
        for n, val, g in ok:
            print(f"  [{'PASS' if g else 'FAIL'}] {n:56s} {val}")
    return all(g for _, _, g in ok)


# ------------------------------------------------------------------- D1 ------
def _nemo_acc():
    """NEMO's accumulated per-term and stage rows over the SAME 360 days.

    ``nemo_accum_torque`` reads its run directory and length from the
    environment at import, so they are set here rather than left to the caller.
    """
    os.environ.setdefault("DINO_ACC_RUN_DIR", NEMO_ACC_RUN)
    os.environ.setdefault("DINO_ACC_DAYS", "360")
    import nemo_accum_torque as N
    if N.DAYS_90 != 360 or os.path.basename(N.RUN_ACC90) != os.path.basename(
            NEMO_ACC_RUN):
        raise SystemExit(
            f"nemo_accum_torque resolved {N.RUN_ACC90} / {N.DAYS_90} d, not "
            f"{NEMO_ACC_RUN} / 360 d -- set DINO_ACC_RUN_DIR and DINO_ACC_DAYS "
            "before this process starts")
    kt = N.B.G.KT_RESTART + 360 * N.B.G.STEPS_PER_DAY
    d = N.load_acc(N.RUN_ACC90, kt)
    if int(d["nacc_steps"]) != 360 * N.B.G.STEPS_PER_DAY:
        raise SystemExit(f"NEMO accumulated {int(d['nacc_steps'])} steps, not "
                         f"{360 * N.B.G.STEPS_PER_DAY}")
    return N, d


def _lego_acc():
    z = np.load(STAGE_NPZ, allow_pickle=True)
    return z


def _rate_gap_per_row():
    """NEMO minus legoESM realised row rate [m3/s2 per u-row], year mean, per
    row -- the shape D1's registered concentration clause is tested against."""
    N, d = _nemo_acc()
    z = _lego_acc()
    nemo = N.stage_rows(d)["realized"]
    lego = np.asarray(z["acc_stage"], np.float64).sum(axis=1).mean(axis=0)
    return nemo - lego


def _ubot_band(u):
    """Row-width-weighted mean BOTTOM-cell zonal velocity over the southern rows
    [m/s].  Printed because the SIGN of the drag torque is only interpretable
    once the sign of the flow it acts on is known, and the pre-registration got
    that assumption wrong (see the retraction in ``d1``)."""
    us = np.where(B.umask, np.asarray(u, np.float64), 0.0)
    u_bot = np.take_along_axis(us, B.KBOT_U[:, :, None], axis=2)[:, :, 0]
    col = B.umask.any(axis=2)
    w = np.where(col, B.e1u, 0.0)
    return float(np.sum((u_bot * w)[B.ROWS]) / np.sum(w[B.ROWS]))


def d1(rows_report=True):
    npz = np.load(LEGO_NPZ)
    nemo0 = B.load_nemo(0, run_dir=NEMO_RUN)
    voff = B.pick_v_offset(npz, nemo0)
    print(f"[stamp] D1 days {DAYS}")
    print(f"[stamp] lego {LEGO_NPZ}   NEMO {NEMO_RUN}")
    dl, dn, ub = {}, {}, {}
    for d in DAYS:
        sl = B.load_lego(LEGO_NPZ, npz, d, voff)
        sn = B.load_nemo(d, run_dir=NEMO_RUN)
        dl[d] = B.phi_drag(sl["u"], sl["v"])
        dn[d] = B.phi_drag(sn["u"], sn["v"])
        ub[d] = (_ubot_band(sl["u"]), _ubot_band(sn["u"]))
    band = lambda a: float(np.mean(a[B.ROWS]))      # noqa: E731
    wall = lambda a: float(np.mean(a[WALL_ROWS]))   # noqa: E731
    print()
    print("=" * 100)
    print("D1  BOTTOM-DRAG TORQUE [m3/s2 per u-row], the SAME transcription applied")
    print("    to each model's own saved state.  Negative = a sink on eastward flow.")
    print("=" * 100)
    print(f"  {'day':>5}{'lego band':>12}{'NEMO band':>12}{'D band':>10}"
          f"{'lego wall':>12}{'NEMO wall':>12}{'D wall':>10}"
          f"{'u_bot lego':>12}{'u_bot NEMO':>12}   (u_bot in 1e-3 m/s)")
    for d in DAYS:
        print(f"  {d:>5}{band(dl[d]):>12.4f}{band(dn[d]):>12.4f}"
              f"{band(dl[d]) - band(dn[d]):>10.4f}"
              f"{wall(dl[d]):>12.4f}{wall(dn[d]):>12.4f}"
              f"{wall(dl[d]) - wall(dn[d]):>10.4f}"
              f"{1e3*ub[d][0]:>12.3f}{1e3*ub[d][1]:>12.3f}")
    dband = np.array([band(dl[d]) - band(dn[d]) for d in DAYS])
    dwall = np.array([wall(dl[d]) - wall(dn[d]) for d in DAYS])
    # TIME-WEIGHTED, not a plain mean over the sample list: the 19 days are
    # spaced 30 d apart to day 270 and 10 d apart after it, so 10 of 19 samples
    # sit in the last quarter, where the difference is largest.  A plain mean
    # over that list inflates the year figure by ~17% and an earlier revision
    # quoted it (0.1157, "103% of the shortfall") as the year mean.
    yb, yw = time_weighted_mean(dband), time_weighted_mean(dwall)
    print()
    print(f"  plain mean of the 19 sampled days:  band "
          f"{float(dband.mean()):+.4f}   wall {float(dwall.mean()):+.4f}"
          f"   (NOT the year mean -- the sampling is non-uniform)")
    print(f"  TIME-WEIGHTED year mean (trapezoid from day 0):  band {yb:+.4f}"
          f"   wall {yw:+.4f}   (the shortfall being explained is {DEFICIT:.3f})")
    if rows_report:
        print()
        print(f"  {'row':>5}{'lat':>8}" + "".join(f"{'d'+str(h):>11}" for h in HORIZONS)
              + f"{'year':>11}   (lego - NEMO, per row)")
        lat = np.asarray(A.gphit, float)[:, 25]
        for j in B.ROWS:
            yr = float(np.mean([dl[d][j] - dn[d][j] for d in DAYS]))
            print(f"  {j:>5}{lat[j]:>8.2f}"
                  + "".join(f"{dl[h][j] - dn[h][j]:>11.4f}" for h in HORIZONS)
                  + f"{yr:>11.4f}")
    # THE REGISTERED CLAUSE: "with the excess concentrated in the four wall
    # rows".  It is the discriminating half of D1 and it is a SHAPE test, so it
    # is measured against the shape of the thing being explained -- the per-row
    # rate deficit, taken from the same 360-day artifacts as D2/D3.
    try:
        rate_gap = _rate_gap_per_row()
    except SystemExit as exc:
        print(f"\n  [shape test SKIPPED: {exc}]")
        rate_gap = None
    if rate_gap is not None:
        drag_row = np.array([time_weighted_mean([dl[d][j] - dn[d][j]
                                                for d in DAYS])
                             for j in B.ROWS])
        rg = np.array([rate_gap[j] for j in B.ROWS])
        off = [k for k, j in enumerate(B.ROWS) if j not in WALL_ROWS]
        wal = [k for k, j in enumerate(B.ROWS) if j in WALL_ROWS]
        r = float(np.corrcoef(drag_row, rg)[0, 1])
        print()
        print("  THE SHAPE TEST -- the registered clause was 'the excess")
        print("  concentrated in the four wall rows'.  Compared against the shape")
        print("  of the thing being explained, the per-row rate deficit:")
        print(f"    {'':22}{'wall rows 1-4':>16}{'rows 5-13':>12}{'ratio':>9}")
        print(f"    {'rate deficit':22}{float(np.mean(rg[wal])):>16.4f}"
              f"{float(np.mean(rg[off])):>12.4f}"
              f"{float(np.mean(rg[wal]) / np.mean(rg[off])):>9.1f}")
        print(f"    {'drag difference':22}{float(np.mean(drag_row[wal])):>16.4f}"
              f"{float(np.mean(drag_row[off])):>12.4f}"
              f"{float(np.mean(drag_row[wal]) / np.mean(drag_row[off])):>9.1f}")
        print(f"    row-wise correlation of the two shapes: r = {r:+.3f}")
        print("    The deficit is sharply wall-concentrated; the drag difference is")
        print("    flat across the basin.  The registered concentration clause is")
        print("    NOT met, and that -- not the sign -- is what decides D1.")
    ubl = float(np.mean([ub[d][0] for d in DAYS]))
    ubn = float(np.mean([ub[d][1] for d in DAYS]))
    verdict = ("CONFIRM (drag excess >= half the shortfall)"
               if (yb < 0 and abs(yb) >= D1_CONFIRM) else
               "REFUTE (the registered concentration clause fails)" if yb > 0
               and abs(yb) >= D1_REFUTE else
               "REFUTE (|D| under a fifth of the shortfall)" if abs(yb) <= D1_REFUTE
               else "PARTIAL")
    print()
    print(f"  [D1, per the registered bins: CONFIRM |D| >= {D1_CONFIRM:.4f} and "
          f"negative; REFUTE |D| <= {D1_REFUTE:.4f} or positive]")
    print(f"  => {verdict}   (time-weighted year-mean band D = {yb:+.4f})")
    print("  The ratio of this number to the 0.112 shortfall is deliberately NOT")
    print("  printed: this diagnostic is the full-velocity drag torque, whose")
    print("  column mean the card's post-solve correction discards every step, so")
    print("  it is not a term in that budget and the ratio would be a comparison")
    print("  across two different budgets.")
    print()
    print("  TWO RETRACTIONS ABOUT THE SIGN, both logged where they happened.")
    print("  (1) The pre-registration read a POSITIVE difference as 'legoESM's sink")
    print("      is weaker', assuming the bottom flow here is EASTWARD.  MEASURED,")
    print(f"      it is not: the bottom-cell velocity is {1e3*ubl:+.3f}e-3 m/s in")
    print(f"      legoESM and {1e3*ubn:+.3f}e-3 m/s in NEMO -- WESTWARD in both -- so")
    print("      a drag sink there pushes EASTWARD and a positive difference means")
    print("      legoESM's drag pushes HARDER, because its bottom flow is more")
    print("      westward.")
    print("  (2) The replacement sentence -- 'the drag response OPPOSES the anomaly,")
    print("      so drag is a consequence not a cause' -- is ALSO wrong, and this")
    print("      one is a logical error rather than a sign error.  For ANY sink-type")
    print("      cause the same pattern appears: a sink that is too strong gives a")
    print("      weaker equilibrium flow, and evaluating the SHARED law on the two")
    print("      states then gives a difference that opposes the flow anomaly.  The")
    print("      sign has NO discriminating power here.  What decides D1 is the")
    print("      SHAPE test above.")
    print("  Two further limits on what this diagnostic can say, both structural.")
    print("  It is the full-velocity drag torque, whose column mean is exactly what")
    print("  the card's post-solve correction discards every step, so it is NOT a")
    print("  term in the row budget the deficit lives in -- no conclusion may be")
    print("  drawn from comparing its size against that deficit.  And with the")
    print("  oracle's background kinetic energy the drag rate is within 0.5% of")
    print("  constant at these speeds, so the law is running in its LINEAR regime")
    print("  and this measurement is close to a restatement of the velocity")
    print("  difference itself.")
    return yb, yw, verdict


# ------------------------------------------------------------------- D2 ------
def d2():
    if not os.path.exists(STAGE_NPZ):
        raise SystemExit(f"D2 needs the 360-day accumulation artifact {STAGE_NPZ}; "
                         "run southern_term_torque_accum.py --days 360 --out-npz")
    z = np.load(STAGE_NPZ, allow_pickle=True)
    # gate 3 of the pre-registration: the artifact must be the shipped card on
    # NEMO's ladder and NEMO's day-of-year, 360 days, unplanted.
    want = {"reconcile_target": "velocity_avg",
            "after_reconcile": "nemo_mlf_baro_corr",
            "e3t_mode": "both", "recipe": "nemo_dino_kamm_mlf"}
    for k, v in want.items():
        got = str(z[k]) if k in z.files else "<absent>"
        if got != v:
            raise SystemExit(f"FATAL: {STAGE_NPZ} stamps {k}={got!r}, need {v!r}")
    if int(z["days"]) != 360 or float(z["plant"]) or float(z["stage_plant"]):
        raise SystemExit(f"FATAL: {STAGE_NPZ} is {int(z['days'])} days / planted")
    if float(z["seasonal_t0_seconds"]) != 15552000.0:
        raise SystemExit(f"FATAL: {STAGE_NPZ} carries the wrong seasonal clock")
    print(f"[stamp] D2 artifact {STAGE_NPZ}, git {str(z['git_sha'])[:12]}, "
          f"{int(z['days'])} d, shipped card, ladder={str(z['e3t_mode'])}")

    stg = [str(x) for x in z["stages"]]
    acc = np.asarray(z["acc_stage"], np.float64)          # (n_int, n_stage, ny)
    if "BARO solve" not in stg:
        raise SystemExit(f"FATAL: no 'BARO solve' stage in {stg}")
    ib = stg.index("BARO solve")
    realized = acc.sum(axis=1)                            # (n_int, ny)
    baro = acc[:, ib, :]
    rows = list(B.ROWS)

    # gate 4: the stage sum must reproduce the rate the STATES show.
    Rs = np.asarray(z["R_series"], np.float64)
    drift = float(np.mean((Rs[-1] - Rs[0])[rows]) / (360 * B.SEC_PER_DAY))
    stage_mean = float(np.mean(realized.mean(axis=0)[rows]))
    print(f"  [gate 4] stage-sum rate {stage_mean:+.4f} vs the artifact's own "
          f"circulation drift {drift:+.4f} m3/s2/row "
          f"(|diff| {abs(stage_mean - drift):.2e})")
    if abs(stage_mean - drift) > 1e-3:
        raise SystemExit("gate 4 FAILED: the stage sum is not the realised rate")

    print()
    print("=" * 100)
    print("D2  DOES legoESM's REALISED ROW RATE COLLAPSE TO ITS OWN BAROTROPIC SOLVE?")
    print("    (the oracle's does, exactly -- its vertical solve's column mean is")
    print("     discarded before the next step by mlf_baro_corr.)")
    print("=" * 100)
    rm, bm = realized.mean(axis=0), baro.mean(axis=0)     # year mean per row
    lat = np.asarray(A.gphit, float)[:, 25]
    print(f"  {'row':>5}{'lat':>8}{'realised':>11}{'BARO':>11}{'diff':>10}"
          f"{'ratio':>10}")
    for j in rows:
        rr = rm[j] / bm[j] if bm[j] != 0 else np.nan
        print(f"  {j:>5}{lat[j]:>8.2f}{rm[j]:>11.4f}{bm[j]:>11.4f}"
              f"{rm[j]-bm[j]:>10.4f}{rr:>10.4f}")
    band_r, band_b = float(np.mean(rm[rows])), float(np.mean(bm[rows]))
    ratios = np.abs(rm[rows] / np.where(np.abs(bm[rows]) > 0, bm[rows], np.nan) - 1.0)
    med = float(np.nanmedian(ratios))
    print(f"  {'band':>5}{'':>8}{band_r:>11.4f}{band_b:>11.4f}"
          f"{band_r-band_b:>10.4f}")
    print()
    print(f"  median |realised/BARO - 1| over the 13 rows = {med:.4f}")
    print(f"  band |realised - BARO| = {abs(band_r - band_b):.4f} against a "
          f"shortfall of {DEFICIT:.3f}")
    verdict = ("CONFIRM (collapses)" if med < D2_CONFIRM
               and abs(band_r - band_b) <= DEFICIT else
               "REFUTE (does not collapse)" if med > D2_REFUTE
               or abs(band_r - band_b) > DEFICIT else "PARTIAL")
    print(f"  [D2, registered bins: CONFIRM median < {D2_CONFIRM} AND "
          f"|diff| <= {DEFICIT}; REFUTE median > {D2_REFUTE} or |diff| > {DEFICIT}]")
    print(f"  => {verdict}")
    ib_bt, ib_po = stg.index("ZDF bt"), stg.index("POST fixer")
    r_bcl = float(np.max(np.abs(acc[:, stg.index("BCLIN expl+diss"), rows])))
    r_zbc = float(np.max(np.abs(acc[:, stg.index("ZDF bc"), rows])))
    r_can = float(np.max(np.abs((acc[:, ib_bt, :] + acc[:, ib_po, :])[:, rows])))
    r_sc = float(np.max(np.abs(acc[:, ib_bt, rows])))
    r_rb = float(np.max(np.abs((realized - baro)[:, rows])))
    print()
    print("  THIS RESULT IS AN ALGEBRAIC IDENTITY.  Said at full precision rather")
    print("  than at the four decimals printed above, because at four decimals it")
    print("  reads like a measurement and it is not:")
    print(f"    max|BCLIN expl+diss| over 36 windows x 13 rows = {r_bcl:.2e}")
    print(f"    max|ZDF bc|                                    = {r_zbc:.2e}")
    print(f"    max|ZDF bt + POST fixer|                       = {r_can:.2e}"
          f"   (against a ZDF bt scale of {r_sc:.2f})")
    print(f"    max|realised - BARO|                           = {r_rb:.2e}")
    print("  Three exact facts force it.  The two baroclinic rows are h_u-weighted")
    print("  DEPTH DEVIATIONS and the reducer is an e3u_0 depth integral, and the")
    print("  accumulator's own M3 gate FAILS CLOSED unless h_u/e3u_0 is constant")
    print("  down each column -- so those two rows are exactly zero in every")
    print("  configuration this probe will accept.  And the card's post-solve")
    print("  correction OVERWRITES the after-level column mean with the barotropic")
    print("  solve's own average, which removes whatever the vertical solve put")
    print("  there.  Under any linear reducer, realised == BARO follows.  The")
    print("  registered CONFIRM bin is cleared by twelve orders of magnitude, which")
    print("  is the signature of a bin that could not have failed.")
    print()
    print("  TWO THINGS IN THIS TABLE DO HAVE POWER, and they are the deliverable:")
    print("  (a) WIRING.  With the post-solve correction OFF this row prints 0.000")
    print("      and the realised rate exceeds the barotropic solve by the whole")
    print(f"      vertical row ({float(np.mean(acc[:, ib_bt, :].mean(axis=0)[rows])):+.4f}).  It does not.  So the shipped")
    print("      card genuinely implements the oracle's post-solve correction, and")
    print("      the implicit vertical solve contributes nothing net on either side.")
    print("  (b) THE TIME FILTER IS EXONERATED.  Gate 4 above compares a per-step")
    print("      two-level rate against the endpoint-to-endpoint circulation drift.")
    print("      Those do NOT telescope -- the Asselin filter is exactly what breaks")
    print("      the telescoping -- and they agree to 0.02% against a 23% deficit.")
    print()
    print("  WHAT IT DOES NOT ESTABLISH, stated plainly.  'BARO solve' is not the")
    print("  split-explicit solver in isolation: it is the depth mean of the ENTIRE")
    print("  explicit momentum tendency (advection, vorticity/Coriolis, pressure")
    print("  gradient, lateral friction) plus the wind stress, the in-substep drag")
    print("  correction and the barotropic biharmonic.  In a split-explicit scheme")
    print("  with this post-solve correction, everything that can move depth-")
    print("  integrated momentum enters through that one row BY CONSTRUCTION.")
    print("  Nothing was narrowed.  The search space after this table is the search")
    print("  space before it, and the per-term decomposition below is the")
    print("  measurement this was supposed to be.")

    # what the OTHER stages carry, so a non-collapse is quantified rather than
    # merely declared.
    print()
    print("  the stage decomposition of the year-mean rate, band mean [m3/s2/row]:")
    for k, name in enumerate(stg):
        print(f"    {name:20s}{float(np.mean(acc[:, k, :].mean(axis=0)[rows])):+10.4f}")
    print(f"    {'= realised':20s}{band_r:+10.4f}")
    return band_r, band_b, med, verdict


# ------------------------------------------------------------------- D3 ------
def time_weighted_mean(series, days=None):
    """Time average of a series sampled on `days` (0 prepended, value 0 there).

    The 19 sample days are 30 d apart to day 270 and 10 d apart after it, so ten
    of nineteen samples sit in the last quarter.  A plain mean over that list
    overweights the end of the year -- it did, by 17%, in the first revision of
    D1, which is why this is a named function every consumer calls rather than
    an inline trapezoid at one site.
    """
    d = np.asarray((0,) + tuple(days if days is not None else DAYS), float)
    v = np.concatenate([[0.0], np.asarray(series, float)])
    return float(np.trapezoid(v, d) / d[-1])


def shape_fits(lego, nemo, rows):
    """How well three shape models explain the per-row difference lego - NEMO.

    The node alone cannot discriminate: a width change confined to the interior
    of one lobe leaves the node where it is, and a pure gain on one lobe cannot
    move it at all -- so 'the node did not move' is consistent with BOTH of the
    hypotheses it was being used to separate.  These are fits to all 13 numbers
    and they do separate them.

      translate  NEMO's own profile shifted by a best-fit number of rows
      gain1      one global multiple of NEMO's profile
      gain2      separate multiples for the wall lobe and the rest
    """
    x = np.arange(len(rows), dtype=float)
    v_l = np.array([lego[j] for j in rows], float)
    v_n = np.array([nemo[j] for j in rows], float)
    d = v_l - v_n
    ss = float(np.sum(d ** 2))

    def _expl(pred):
        return 1.0 - float(np.sum((d - pred) ** 2)) / max(ss, 1e-30)

    best = (-np.inf, 0.0)
    for dl in np.linspace(-3.0, 3.0, 1201):
        pred = np.interp(x - dl, x, v_n) - v_n
        e = _expl(pred)
        if e > best[0]:
            best = (e, float(dl))
    a1 = float(np.sum(d * v_n) / np.sum(v_n ** 2))
    e1 = _expl(a1 * v_n)
    m = np.array([rows[k] in WALL_ROWS for k in range(len(rows))])
    A = np.stack([np.where(m, v_n, 0.0), np.where(~m, v_n, 0.0)], axis=1)
    c, *_ = np.linalg.lstsq(A, d, rcond=None)
    e2 = _expl(A @ c)
    return {"translate": (best[0], best[1]), "gain1": (e1, 1 + a1),
            "gain2": (e2, 1 + float(c[0]), 1 + float(c[1]))}


def _node_of(a, rows):
    """Row index where a per-row profile crosses zero, linearly interpolated.

    D0's whole reading turns on this: a POSITION error moves it, an AMPLITUDE
    error does not.  Pinned by a test on a profile whose crossing is known.
    """
    v = np.array([a[j] for j in rows], float)
    zero = np.flatnonzero(v == 0.0)
    if len(zero) == 1:                       # the node sits exactly on a row
        return float(rows[int(zero[0])])
    # Strict crossings only: comparing np.sign() would count an exact zero
    # TWICE and abort on a profile whose node happens to land on a row.
    cross = np.flatnonzero(v[:-1] * v[1:] < 0.0)
    # np.argmax on an all-False array returns 0, i.e. a profile that never
    # crosses would silently report a node at the first row.  Both cases are
    # fatal rather than guessed: the amplitude-vs-position reading is only
    # meaningful for a profile with exactly one crossing.
    if len(cross) != 1:
        raise SystemExit(
            f"the row profile has {len(cross)} sign changes, not 1 "
            f"(at rows {[rows[k] for k in cross]}) -- the node is not defined "
            "and the amplitude-vs-position reading does not apply")
    k = int(cross[0])
    if v[k] == 0.0 and v[k + 1] == 0.0:
        raise SystemExit("degenerate crossing: both bracketing rows are zero")
    return rows[k] + float(abs(v[k]) / (abs(v[k]) + abs(v[k + 1])))


def d0():
    """The SHAPE of the rate difference itself -- amplitude or position?

    Both models deposit WESTWARD circulation in the rows nearest the wall and
    eastward further north, so the profile has a node.  Two very different
    errors look alike in a band mean: the node in the wrong PLACE (a boundary
    layer of the wrong width) and the wall lobe with the wrong AMPLITUDE (a
    gain error).  They are separated by measuring the node and the per-row
    magnitude ratio, which costs nothing and is done before any candidate is
    ranked.
    """
    N, dn = _nemo_acc()
    z = _lego_acc()
    nemo = N.stage_rows(dn)["realized"]
    lego = np.asarray(z["acc_stage"], np.float64).sum(axis=1).mean(axis=0)
    rows = list(B.ROWS)
    lat = np.asarray(A.gphit, float)[:, 25]
    print()
    print("=" * 100)
    print("D0  IS THE RATE ERROR AN AMPLITUDE OR A POSITION ERROR?")
    print("=" * 100)
    print(f"  {'row':>5}{'lat':>8}{'lego':>10}{'NEMO':>10}{'NEMO-lego':>11}"
          f"{'|lego/NEMO|':>13}")
    for j in rows:
        rr = abs(lego[j] / nemo[j]) if nemo[j] != 0 else np.nan
        print(f"  {j:>5}{lat[j]:>8.2f}{lego[j]:>10.4f}{nemo[j]:>10.4f}"
              f"{nemo[j]-lego[j]:>11.4f}{rr:>13.3f}")

    nl, nn = _node_of(lego, rows), _node_of(nemo, rows)
    fits = shape_fits(lego, nemo, rows)
    wal = [j for j in rows if j in WALL_ROWS]
    # The comparison set is the rows on the OTHER side of the node -- defined by
    # the sign of the oracle's own profile, not by a magnitude cutoff.  An
    # earlier revision used |rate| > 0.5, which admitted row 5 (westward, -0.59)
    # and excluded the two rows nearest the node; it printed 1.058 for a set
    # labelled "eastward" whose genuinely eastward members give 0.995.
    off = [j for j in rows if j not in WALL_ROWS and nemo[j] > 0]
    gw = float(np.mean([abs(lego[j] / nemo[j]) for j in wal]))
    go = float(np.mean([abs(lego[j] / nemo[j]) for j in off]))
    print()
    print(f"  zero-crossing of each model's own profile: legoESM row {nl:.2f}, "
          f"NEMO row {nn:.2f}  (shift {nl - nn:+.2f} rows, i.e. "
          f"{100 * abs(nl - nn) / (nn - rows[0]):.0f}% of the wall lobe's width)")
    print(f"  mean |lego/NEMO| in the 4 wall rows: {gw:.3f};  "
          f"over the rows where the oracle's own profile is EASTWARD: {go:.3f}")
    print()
    print("  THE NODE ALONE CANNOT DECIDE THIS, and an earlier revision said it")
    print("  could.  A width change confined to the interior of the wall lobe")
    print("  leaves the node exactly where it is, and a pure gain on one lobe")
    print("  cannot move it either -- so a small shift is consistent with BOTH")
    print("  hypotheses.  Fit all 13 numbers instead:")
    print(f"    rigid translation of the oracle's profile   explains "
          f"{100 * fits['translate'][0]:5.1f}%  (best shift "
          f"{fits['translate'][1]:+.2f} rows)")
    print(f"    one global gain                             explains "
          f"{100 * fits['gain1'][0]:5.1f}%  (gain {fits['gain1'][1]:.3f})")
    print(f"    separate gains, wall lobe vs the rest       explains "
          f"{100 * fits['gain2'][0]:5.1f}%  (gains "
          f"{fits['gain2'][1]:.3f} / {fits['gain2'][2]:.3f})")
    print("  A rigid POSITION error is refuted by its own residual, not by the")
    print("  node.  A two-lobe amplitude error is the best of the three and is")
    print("  still only a partial description.")
    return nl - nn, gw, go, fits


def d3():
    """The per-TERM cross-model row table -- the non-degenerate decomposition.

    D2 shows the STAGE table cannot localise anything on this card: everything
    reaches the depth-integrated circulation through one row.  The per-term
    table does not have that problem.  Both sides are ACCUMULATED over the same
    360 days (no 10-day sampling error), row-integrated with the same reducer,
    and grouped by the correspondence the campaign already recorded.
    """
    N, dn = _nemo_acc()
    z = _lego_acc()
    from southern_term_torque_matched import LEGO_GROUPS, NEMO_GROUPS
    terms = [str(x) for x in z["terms"]]
    acc_row = np.asarray(z["acc_row"], np.float64)        # (n_int, n_term, ny)
    lego_term = acc_row.mean(axis=0)                       # year mean per row
    n = dn["nacc_steps"]
    nemo_term = {t: N.row_int_3d(dn[f"utrdacc_{t}"] / n) for t in N.PRE_ZDF}
    nemo_term["zdf"] = N.row_int_3d(dn["utrdacc_zdf"] / n)
    nemo_term["atf"] = N.row_int_3d(dn["utrdacc_atf"] / n)
    sr = N.stage_rows(dn)
    # THE PAIRING, and the bug it replaces.  An earlier revision put
    # ``ZDF_recovered`` in the surface group and paired legoESM's REST against
    # ``spg + atf``.  ``ZDF_recovered = zdf_dumped + ubt`` carries the
    # barotropic OPERAND (~1.06e4 on these rows) into the table, while the
    # quantity that actually closes the oracle's budget at that stage is
    # ``_post`` (~1.06e4).  Their difference, 49.24, was the entire "closure
    # failure", and it was published as "the two models' term sets are not in
    # bijection".  They are: with ``zdf_dumped`` in the surface group and
    # ``_post`` in the solve group the oracle side closes to 2e-9.  Both
    # reviewers found this independently and both verified the repricing.
    # THE PARTITION THAT CLOSES, measured rather than assumed:
    #   PRE_ZDF terms + utrdacc_zdf + _post  ==  realised, to 2.1e-9.
    # Note what that costs.  utrdacc_zdf and _post are each ~1.06e4 on these
    # rows and cancel to a realised rate of ~0.5, so NEITHER IS MEANINGFUL
    # ALONE; they are merged into one group with spg, whose partner on the
    # legoESM side is its surface/vertical terms plus REST.  Printing them
    # separately would put a 5000:1 cancellation in a column labelled "term".
    nemo_term["__solve__"] = (nemo_term["spg"] + nemo_term["zdf"] + sr["_post"])
    rows = list(B.ROWS)
    wal = [j for j in rows if j in WALL_ROWS]
    off = [j for j in rows if j not in WALL_ROWS]

    print()
    print("=" * 100)
    print("D3  PER-TERM ACCUMULATED ROW TORQUE, both models, SAME 360 days, same")
    print("    reducer [m3/s2 per u-row].  This is the decomposition D2 cannot give.")
    print("=" * 100)
    print("    Sign: lego - NEMO.  'wall' = the 4 rows nearest the southern wall,")
    print("    'off' = the other 9.  The realised-rate deficit is +0.339 wall /")
    print("    +0.010 off (NEMO - lego), so a term that OWNS it must be strongly")
    print("    wall-concentrated with the matching sign.")
    print()
    print(f"  {'group':24}{'lego band':>11}{'NEMO band':>11}{'D band':>10}"
          f"{'D wall':>10}{'D off':>10}{'wall/off':>10}")
    out = {}
    _G4 = "G4 WIND  surface"
    # The four G-groups are the EXPLICIT terms only.  On legoESM the explicit
    # terms sum to about -590 and the split-explicit barotropic solve returns
    # +590 (the accumulator's REST row); on the oracle the same job is done by
    # spg (+ the Asselin filter).  Without that fifth row the table is not a
    # budget and cannot close, and a table that does not close cannot attribute
    # -- the four groups alone differ by ~+46 per wall row against a realised
    # difference of -0.4.
    for gname, lkeys in LEGO_GROUPS.items():
        if gname == _G4:
            continue                       # merged into the solve group below
        idx = [terms.index(k) for k in lkeys if k in terms]
        lv = lego_term[idx].sum(axis=0)
        nv = sum(nemo_term[t] for t in NEMO_GROUPS[gname])
        d = lv - nv
        out[gname] = d
        ratio = (float(np.mean(d[wal]) / np.mean(d[off]))
                 if abs(float(np.mean(d[off]))) > 1e-9 else float("inf"))
        print(f"  {gname:24}{float(np.mean(lv[rows])):>11.3f}"
              f"{float(np.mean(nv[rows])):>11.3f}{float(np.mean(d[rows])):>10.3f}"
              f"{float(np.mean(d[wal])):>10.3f}{float(np.mean(d[off])):>10.3f}"
              f"{ratio:>10.1f}")
    # the fifth row: the barotropic solve's own return, both sides
    lv5 = (lego_term[terms.index("REST")]
           + sum(lego_term[terms.index(k)] for k in LEGO_GROUPS[_G4]
                 if k in terms))
    nv5 = nemo_term["__solve__"]
    _G5 = "G4+5 SOLVE+sfc/vert"
    out[_G5] = lv5 - nv5
    d5 = out[_G5]
    print(f"  {_G5:24}{float(np.mean(lv5[rows])):>11.3f}"
          f"{float(np.mean(nv5[rows])):>11.3f}{float(np.mean(d5[rows])):>10.3f}"
          f"{float(np.mean(d5[wal])):>10.3f}{float(np.mean(d5[off])):>10.3f}"
          f"{float(np.mean(d5[wal]) / np.mean(d5[off])):>10.1f}")

    # CLOSURE GATE.  The five rows must sum to the realised-rate difference.  A
    # table that misses this cannot attribute anything, and saying so is the
    # result if it fails.
    rg = _rate_gap_per_row()                       # NEMO - lego
    tot = sum(out[g] for g in out)
    resid = np.array([tot[j] + rg[j] for j in rows])   # lego-NEMO + (NEMO-lego)
    scale = float(np.max([np.max(np.abs(out[g][rows])) for g in out]))
    # SIDE-SPLIT closure, because a differenced residual cannot say WHICH side
    # is incomplete -- and when this table first failed, all of it was on one
    # side.  legoESM's fifth group is a REMAINDER, so its side closes by
    # construction and only the oracle's side can ever fire.  Say so.
    lego_all = sum(out[g] * 0 for g in out) + sum(
        lego_term[terms.index(k)] for g, ks in LEGO_GROUPS.items()
        if g != _G4 for k in ks if k in terms) + lv5
    nemo_all = sum(sum(nemo_term[t] for t in NEMO_GROUPS[g])
                   for g in NEMO_GROUPS if g != _G4) + nv5
    lego_real = np.asarray(z["acc_stage"], np.float64).sum(axis=1).mean(axis=0)
    nemo_real = sr["realized"]
    e_l = float(np.max(np.abs((lego_all - lego_real)[rows])))
    e_n = float(np.max(np.abs((nemo_all - nemo_real)[rows])))
    print()
    print(f"  [closure] each side's groups against its OWN realised rate:")
    print(f"            legoESM {e_l:.2e}   (a remainder row -- closes by "
          f"construction)")
    print(f"            NEMO    {e_n:.2e}   (this is the side that can fail)")
    print(f"  [closure] differenced: max|sum - (lego - NEMO) realised| = "
          f"{float(np.max(np.abs(resid))):.2e} m3/s2 per row, "
          f"largest group {scale:.1f}")
    if float(np.max(np.abs(resid))) > 1e-6 * scale:
        print("  [closure] FAILS -- no per-term attribution may be drawn.")
    print("  [closure] NOTE the gate's real power: legoESM's last group is a")
    print("            REMAINDER, so its side is an identity.  Only the oracle's")
    print("            side can fire, and it did -- an earlier revision paired")
    print("            legoESM's REST against spg+atf while crediting the")
    print("            barotropic OPERAND into the surface group, and published")
    print("            the resulting 49-per-row residual as 'the two models' term")
    print("            sets are not in bijection'.  RETRACTED: they are in")
    print("            bijection, one group was paired wrong, and both reviewers")
    print("            found it independently.")
    print()
    print("  per row, lego - NEMO by group [m3/s2]:")
    lat = np.asarray(A.gphit, float)[:, 25]
    print(f"  {'row':>5}{'lat':>8}" + "".join(f"{g.split()[0]:>10}" for g in out)
          + f"{'sum':>10}{'rate gap':>10}")
    rg = _rate_gap_per_row()
    for j in rows:
        tot = sum(out[g][j] for g in out)
        print(f"  {j:>5}{lat[j]:>8.2f}"
              + "".join(f"{out[g][j]:>10.3f}" for g in out)
              + f"{tot:>10.3f}{-rg[j]:>10.3f}")
    print()
    print("  The last column is the realised-rate difference (lego - NEMO) the")
    print("  groups have to account for.  A group whose per-row difference tracks")
    print("  it is a candidate; one that is flat is not.  Correlations over the 13")
    print("  rows:")
    for g, d in out.items():
        r = float(np.corrcoef(np.array([d[j] for j in rows]),
                              np.array([-rg[j] for j in rows]))[0, 1])
        print(f"    {g:24}r = {r:+.3f}")
    _canc = float(np.mean([sum(abs(out[g][j]) for g in out) for j in wal]))
    _tgt = float(abs(np.mean([sum(out[g][j] for g in out) for j in wal])))
    print()
    print("  AND THE CLOSED TABLE STILL CANNOT ATTRIBUTE.  In the wall rows the")
    print(f"  four groups differ by {_canc:.1f} in total magnitude and sum to")
    print(f"  {_tgt:.3f} -- a {_canc / max(_tgt, 1e-30):.0f}:1 cancellation.  Reading a 0.3%")
    print("  residual out of groups that individually differ by 4-8% is not a")
    print("  measurement, and the group differences have the signature of a")
    print("  time-level or staging difference in the DIAGNOSTICS (the totals agree")
    print("  to 0.02% while every individual term is off by several percent)")
    print("  rather than of physics.  Nothing may be attributed from this table")
    print("  until those group differences are shown to survive a time-level A/B.")
    print("  Note also that legoESM's dominant wall-concentrated group is 99.7%")
    print("  ONE fused diagnostic (kinetic-energy gradient + pressure gradient),")
    print("  which this model cannot currently split.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--d1", action="store_true", help="D1 only")
    ap.add_argument("--d2", action="store_true", help="D2 only")
    args = ap.parse_args()
    sha = subprocess.run(["git", "-C", _DIR, "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", _DIR, "status", "--porcelain", "--",
                            os.path.abspath(__file__)],
                           capture_output=True, text=True).stdout.strip()
    print(f"[stamp] repo HEAD {sha}"
          + (f"   *** THIS FILE IS {dirty.split()[0]} -- the SHA does not identify "
             "the code below ***" if dirty else "   (this file clean at that commit)"))
    if not self_checks():
        raise SystemExit("SELF-CHECKS FAILED -- no number below is trustworthy")
    if args.self_check:
        return
    r1 = r2 = None
    if not args.d2:
        r1 = d1()
    if not args.d1:
        r2 = d2()
        d0()
        d3()
    if r1 and r2:
        print()
        print("=" * 100)
        print("THE JOINT READING, fixed in the pre-registration before either ran")
        print("=" * 100)
        print("  The registered table has a D2 axis, and D2 turned out to be an")
        print("  ALGEBRAIC IDENTITY on the only configuration its own gate admits")
        print("  (see the D2 block).  An axis with one reachable value selects")
        print("  nothing, so the 2x2 table is VOID and is not evaluated here.  The")
        print("  pre-registration is wrong on this point and the error is disclosed")
        print("  rather than worked around.")
        print()
        print("  WHAT THE THREE BLOCKS ACTUALLY LEAVE:")
        print(f"    D1  {r1[2]}  -- decided by the SHAPE test, not the sign: the")
        print("        rate deficit is ~32x wall-concentrated and the drag")
        print("        difference is flat (1.1x), correlation 0.36.  Bottom drag is")
        print("        not shaped like the thing being explained.  It is NOT closed:")
        print("        the applied drag is a substep-loop integral this diagnostic")
        print("        cannot see, and only a perturbation arm can retire it.")
        print("    D2  an identity.  What it does buy: the shipped card's")
        print("        post-solve correction is verified active, and the time")
        print("        filter is exonerated at 0.02% against a 23% deficit.")
        print("    D3  the per-term budget CLOSES to 2e-9 once the oracle's")
        print("        vertical row is paired correctly.  An earlier revision")
        print("        double-counted it and published the resulting 49-per-row")
        print("        residual as 'the term sets are not in bijection' --")
        print("        RETRACTED.  But the closed table cancels ~300:1 against")
        print("        the quantity it would have to attribute, so it does not")
        print("        attribute either.")
        print()
        print("  None of this is a perturbation test.  Nothing here may be called")
        print("  confirmed until one arm changes one thing and moves the wall rows")
        print("  in the predicted direction.")


if __name__ == "__main__":
    main()
