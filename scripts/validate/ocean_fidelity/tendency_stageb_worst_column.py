"""Dump the worst Stage-B EVD-salt mismatch column, profile by profile.

The Stage-B operator test (compare_tendencies_nemo.py) shows an unexplained
asymmetry in EVD columns: our backward-Euler solve with NEMO's own avs gives
salt trends ~46x NEMO's strd_zdf, while the SAME solve is ~5x SMALLER than
NEMO's ttrd_zdf for temperature.  Candidates: comparator fill fabricating
gradients, K interface staggering on the S path, record alignment, or a
genuine semantic mismatch of what strd_zdf contains (the run3 trend file
carries separate strd_evd/strd_zdfp, so strd_zdf's composition must be read
from data, not the docstring).

This probe picks the worst |ours - strd_zdf| EVD column (and a calm control
column) and prints every input and output profile so the mechanism is read
off directly instead of ranked by plausibility.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from compare_tendencies_nemo import load_pair, run_stage_b  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tfile", required=True)
    p.add_argument("--ufile", required=True)
    p.add_argument("--vfile", required=True)
    p.add_argument("--rec", type=int, default=1)
    a = p.parse_args()

    d = load_pair(a.tfile, a.ufile, a.vfile, a.rec)
    dT, dS, ttrd, strd, wet_c, (z, ny, nx), dT2, dS2 = run_stage_b(d)

    import netCDF4 as nc
    ds = nc.Dataset(a.tfile)

    def rec_cols(name, r):
        arr = ds.variables[name][r]
        arr = arr.filled(np.nan) if np.ma.isMaskedArray(arr) else np.asarray(arr)
        return np.transpose(arr.reshape(z, ny * nx), (1, 0)).astype(np.float64)

    avs = rec_cols("avs", a.rec)
    avt = rec_cols("avt", a.rec)
    strd_evd = rec_cols("strd_evd", a.rec)
    strd_zdfp = rec_cols("strd_zdfp", a.rec)
    ttrd_evd = rec_cols("ttrd_evd", a.rec)
    S_prev = rec_cols("vosaline", a.rec - 1)
    S_curr = rec_cols("vosaline", a.rec)
    T_prev = rec_cols("votemper", a.rec - 1)
    e3t = rec_cols("e3t", a.rec - 1)
    ds.close()

    evd_col = (np.nan_to_num(avt, nan=0.0) > 50.0).any(axis=1)
    err = np.nan_to_num(np.abs(dS - strd), nan=0.0)
    err_col = np.where(wet_c, err, 0.0).sum(axis=1)
    err_col = np.where(evd_col, err_col, 0.0)
    worst = int(np.argmax(err_col))
    calm_err = np.where(~evd_col, np.where(wet_c, err, 0.0).sum(axis=1), 0.0)
    calm = int(np.argmax(calm_err))

    for label, c in (("WORST EVD", worst), ("WORST CALM (control)", calm)):
        j, i = divmod(c, nx)
        nwet = int(wet_c[c].sum())
        print(f"\n########## {label} column {c} (j={j}, i={i}), wet levels={nwet}")
        hdr = (f"{'k':>3} {'e3t':>7} {'S(r-1)':>9} {'S(r)':>9} {'T(r-1)':>8} "
               f"{'avs(r)':>9} {'ourdS':>11} {'strd_zdf':>11} {'strd_zdfp':>11} "
               f"{'strd_evd':>11} {'ttrd_evd':>11}")
        print(hdr)
        for k in range(min(nwet + 2, z)):
            print(f"{k:3d} {e3t[c, k]:7.2f} {S_prev[c, k]:9.4f} "
                  f"{S_curr[c, k]:9.4f} {T_prev[c, k]:8.3f} {avs[c, k]:9.3f} "
                  f"{dS[c, k]:11.3e} {strd[c, k]:11.3e} {strd_zdfp[c, k]:11.3e} "
                  f"{strd_evd[c, k]:11.3e} {ttrd_evd[c, k]:11.3e}")
        # Column-integral checks: what does each trend sum to (conservation),
        # and does our solve's total match the actual state change S(r)-S(r-1)?
        w = np.nan_to_num(e3t[c] * wet_c[c], nan=0.0)
        def col(x):
            return float(np.nansum(np.where(wet_c[c], x, 0.0) * w))
        print(f"  col-int (dz-weighted): ourdS {col(dS[c]):+.3e}  "
              f"strd_zdf {col(strd[c]):+.3e}  strd_zdfp {col(strd_zdfp[c]):+.3e}  "
              f"strd_evd {col(strd_evd[c]):+.3e}")
        dS_state = (S_curr[c] - S_prev[c]) / 3600.0
        print(f"  state change (S(r)-S(r-1))/dt: col-int {col(dS_state):+.3e}, "
              f"rms {float(np.sqrt(np.nanmean(np.where(wet_c[c], dS_state, np.nan)**2))):.3e}")
        print(f"  rms ourdS {float(np.sqrt(np.nanmean(np.where(wet_c[c], dS[c], np.nan)**2))):.3e}  "
              f"rms strd_zdf {float(np.sqrt(np.nanmean(np.where(wet_c[c], strd[c], np.nan)**2))):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
