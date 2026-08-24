#!/usr/bin/env python
"""#1354/#1515 — close the atmospheric energy budget and isolate any leak.

The EnergyBudgetTracker records, per diag step, the global area-weighted:
  toa_net = sw_down_toa - sw_up_toa - lw_up_toa      [positive down]
  sw_net_sfc, lw_net_sfc                             [positive INTO surface]
  dE_dt   = d/dt (column moist static energy)
  residual = toa_net - dE_dt
and the driver records hfss (SH), hfls (LH)          [positive UP into atm].

The code's own surface_energy_flux gives the net flux FROM atmosphere TO
surface:  F_sfc_net = sw_net_sfc + lw_net_sfc - hfss - hfls,  and the closed
atmospheric budget is  dE_dt = toa_net - F_sfc_net.  So the numerical LEAK is

    LEAK = dE_dt - (toa_net - F_sfc_net)
         = F_sfc_net - (toa_net - dE_dt)
         = (sw_net_sfc + lw_net_sfc - hfss - hfls) - residual        [W/m^2]

VERDICT (GLM #1354): a PERSISTENT LEAK ~ +20 W/m^2 while dE_dt > 0 (warming) =
energy created internally = a physics BOOKKEEPING BUG.  LEAK ~ 0 = the budget
closes, so the warming is driven by a genuine flux imbalance (find which flux).

CAVEAT (instrument, not physics): the tracker's E is moist static energy with a
fixed c_pd / L_v.  If the model conserves a slightly different energy, dE_dt
carries a definitional offset and a SMALL constant LEAK is inconclusive.  A
LARGE or GROWING leak (toward the day-165 detonation) is the unambiguous signal.
"""
import argparse
import glob
import os

import numpy as np


def load_energy_chunks(run_dir):
    incr = os.path.join(run_dir, "timeseries_incremental")
    efiles = sorted(glob.glob(os.path.join(incr, "energy_chunk_*.npz")))
    if not efiles:
        raise SystemExit(f"no energy_chunk_*.npz under {incr} — did the diag path run?")
    keys = ("energy_toa_net", "energy_dE_dt", "energy_residual",
            "sw_net_sfc", "lw_net_sfc")
    out = {k: [] for k in keys}
    for f in efiles:
        z = np.load(f)
        for k in keys:
            if k in z.files:
                out[k].append(np.asarray(z[k], dtype=np.float64))
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}


def load_sh_lh(run_dir):
    # hfss/hfls ride the main incremental timeseries (chunk_*.npz), same cadence
    incr = os.path.join(run_dir, "timeseries_incremental")
    cfiles = sorted(f for f in glob.glob(os.path.join(incr, "chunk_*.npz"))
                    if "energy_chunk" not in f and "moisture_chunk" not in f)
    hfss, hfls = [], []
    for f in cfiles:
        z = np.load(f)
        if "hfss" in z.files:
            hfss.append(np.asarray(z["hfss"], dtype=np.float64))
        if "hfls" in z.files:
            hfls.append(np.asarray(z["hfls"], dtype=np.float64))
    return (np.concatenate(hfss) if hfss else np.array([]),
            np.concatenate(hfls) if hfls else np.array([]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    a = ap.parse_args(argv)

    e = load_energy_chunks(a.run_dir)
    hfss, hfls = load_sh_lh(a.run_dir)
    toa = e["energy_toa_net"]; dEdt = e["energy_dE_dt"]; resid = e["energy_residual"]
    sws = e["sw_net_sfc"]; lws = e["lw_net_sfc"]
    n = min(len(toa), len(dEdt), len(resid), len(sws), len(lws), len(hfss), len(hfls))
    if n == 0:
        raise SystemExit("no overlapping samples across energy + hfss/hfls arrays")
    toa, dEdt, resid = toa[:n], dEdt[:n], resid[:n]
    sws, lws, hfss, hfls = sws[:n], lws[:n], hfss[:n], hfls[:n]

    F_sfc_net = sws + lws - hfss - hfls          # atm -> surface [W/m^2]
    leak = F_sfc_net - (toa - dEdt)              # == F_sfc_net - residual
    # sanity: residual as recorded must equal toa - dEdt
    resid_check = np.max(np.abs(resid - (toa - dEdt)))

    print(f"# samples={n}  residual_selfcheck_max={resid_check:.2e} W/m^2")
    print(f"# {'i':>3} {'toa_net':>9} {'dE_dt':>9} {'F_sfc_net':>10} {'LEAK':>9}  [W/m^2]")
    for i in range(n):
        print(f"  {i:>3} {toa[i]:>9.3f} {dEdt[i]:>9.3f} {F_sfc_net[i]:>10.3f} {leak[i]:>9.3f}")
    # skip the first sample (dE_dt=0 seed)
    body = slice(1, n)
    lm = float(np.mean(leak[body])); ls = float(np.std(leak[body]))
    wm = float(np.mean(dEdt[body]))
    print(f"\nLEAK  mean={lm:+.3f}  std={ls:.3f}  W/m^2   (dE_dt mean={wm:+.3f})")

    # GLM offset-separation (the "large & growing => real" discriminator is
    # UNSOUND: a MSE-vs-model-energy offset scales with d/dt of column water
    # and with P, both of which grow during the runaway).  Regress LEAK against
    # the thermodynamic-bookkeeping predictors we can build from the saved
    # fields: d(CWV)/dt [latent term, kg/m^2/s] and hfls [~ evaporation/precip].
    # High R^2 => the leak IS an instrument offset, not internal creation; the
    # UNEXPLAINED residual is the only part that could be a real leak.
    cwv = _load_series(a.run_dir, "CWV", n)
    hfls_s = hfls[:n]
    verdict_extra = ""
    if cwv is not None and len(cwv) >= n:
        dt_s = _median_dt_seconds(a.run_dir, n)
        dcwv_dt = np.gradient(cwv[:n]) / dt_s              # kg/m^2/s
        X = np.column_stack([np.ones(n), dcwv_dt, hfls_s])[body]
        y = leak[body]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1.0
        r2 = 1.0 - ss_res / ss_tot
        resid_leak = y - pred
        rl_mean = float(np.mean(resid_leak))
        print(f"\noffset-regression  LEAK ~ c0 + c1*d(CWV)/dt + c2*hfls")
        print(f"  R^2={r2:.3f}  c0={coef[0]:+.3f}  c1={coef[1]:+.3e}  c2={coef[2]:+.4f}")
        print(f"  UNEXPLAINED residual leak  mean={rl_mean:+.3f}  "
              f"std={float(np.std(resid_leak)):.3f} W/m^2")
        if r2 >= 0.7:
            verdict_extra = (f"the leak is largely EXPLAINED by thermodynamic "
                             f"bookkeeping (R^2={r2:.2f}) => INSTRUMENT OFFSET, "
                             f"not internal creation; residual leak "
                             f"{rl_mean:+.2f} W/m^2 is the only candidate real term.")
        else:
            verdict_extra = (f"the leak is NOT explained by d(CWV)/dt or hfls "
                             f"(R^2={r2:.2f}) => not a simple MSE offset; a real "
                             f"internal source is in play.")
    print()
    if abs(lm) <= 5.0:
        print("VERDICT: |LEAK| <= 5 W/m^2 — the atmospheric energy budget CLOSES; "
              "the warming (if any) is a real flux imbalance, not internal "
              "energy creation.")
    else:
        print(f"VERDICT: |LEAK| = {abs(lm):.1f} W/m^2 is large. " + (verdict_extra or
              "Run the offset-regression (needs CWV) before convicting the model: "
              "a MSE-vs-model-energy offset can masquerade as a leak (GLM)."))
    # DO NOT use 'large & growing' as the discriminator — GLM: an instrument
    # offset grows with the signal too. The regression residual is the verdict.


def _load_series(run_dir, key, n):
    incr = os.path.join(run_dir, "timeseries_incremental")
    cfiles = sorted(f for f in glob.glob(os.path.join(incr, "chunk_*.npz"))
                    if "energy_chunk" not in f and "moisture_chunk" not in f)
    vals = []
    for f in cfiles:
        z = np.load(f)
        if key in z.files:
            vals.append(np.asarray(z[key], dtype=np.float64))
    return np.concatenate(vals) if vals else None


def _median_dt_seconds(run_dir, n):
    incr = os.path.join(run_dir, "timeseries_incremental")
    for f in sorted(glob.glob(os.path.join(incr, "chunk_*.npz"))):
        if "energy_chunk" in f or "moisture_chunk" in f:
            continue
        z = np.load(f)
        for tk in ("time_seconds", "elapsed_seconds", "times_seconds", "time_days"):
            if tk in z.files:
                t = np.asarray(z[tk], dtype=np.float64)
                if tk == "time_days":
                    t = t * 86400.0
                if len(t) >= 2:
                    return float(np.median(np.diff(t)))
    return 86400.0        # fall back to daily (--diag-days 1)


if __name__ == "__main__":
    main()
