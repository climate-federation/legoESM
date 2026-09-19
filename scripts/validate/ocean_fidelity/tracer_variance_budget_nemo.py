"""Which operator destroys NEMO's tracer variance? The budget, not a guess.

A theta-S census showed a difference in the tracer DISTRIBUTION but is blind
to which operator produced it -- tails-down/centre-up is generic variance
decay, produced alike by advection's implicit diffusion, lateral diffusion,
vertical mixing, or the boundary layer. The discriminator is the
variance-decay budget: for each operator, the volume integral

    chi_op = 2 * integral( theta' * (dtheta/dt)_op ) dV

where theta' is the departure from the volume-weighted mean. chi_op is the
rate at which that operator changes the tracer variance; the most negative
term NAMES the operator instead of ranking guesses. This is the standard
Burchard/Ilicak spurious-mixing diagnostic.

WHAT IS MEASURED HERE: NEMO's OWN operators, from its archived trends. This
is the oracle's side of the budget and the reference our operators must later
be compared against on a matched state.

THE APPROXIMATION, AND WHY IT IS MEASURED RATHER THAN ASSUMED. The DAILY
trend file carries the full operator set but NOT the tracer state, so theta
for the daily budget has to come from the hourly file's records. A product of
daily means is not the daily mean of products -- the sub-daily covariance
between theta' and each tendency is dropped. That error is normally invisible
and would silently set the size of every term.

It is NOT left invisible here. The HOURLY file carries both votemper and
ttrd_zdf, so for the vertical-diffusion operator chi can be computed BOTH
ways: exactly, hour by hour on instantaneous fields, and approximately, from
daily means. Their ratio is a measured error bar on the approximation, and it
is reported next to every daily-mean term. If that ratio is far from one, the
daily-mean terms are not quotable and the budget says so.

CONVENTIONS. Cell volume is e1t*e2t*e3t with e3t taken from the TREND FILE's
own record, i.e. the model's actual z-star geometry at that time, not the
mesh reference thickness. The trend files use votemper/vosaline (the
grid_T files use to/so); all are Conservative Temperature / Absolute Salinity
under ln_teos10.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
from global_tracer_content import load_mesh_metrics  # noqa: E402

# Operators whose trends are non-zero in the daily file (measured, job
# 9880273). ttrd_tot/evd/iso/dmp/zdfp are written but identically zero:
# `tot` means there is no archived total to close against, `iso` means the
# isoneutral contribution rides inside `ldf` and must not be counted twice.
T_OPS = ("ttrd_totad", "ttrd_ldf", "ttrd_zdf", "ttrd_bbl", "ttrd_qsr")
S_OPS = ("strd_totad", "strd_ldf", "strd_zdf", "strd_bbl")


def _fill(x):
    x = x.filled(np.nan) if np.ma.isMaskedArray(x) else np.asarray(x)
    return np.asarray(x, dtype=np.float64)


def chi(tr, tend, dV, wet):
    """2 * integral(tracer' * tendency) dV, with tracer' about the
    VOLUME-WEIGHTED mean over the same wet cells the integral uses."""
    w = np.where(wet, dV, 0.0)
    vol = w.sum()
    if vol <= 0:
        return float("nan")
    mean = float((np.where(wet, tr, 0.0) * w).sum() / vol)
    anom = np.where(wet, tr - mean, 0.0)
    return float(2.0 * (anom * np.where(wet, tend, 0.0) * w).sum())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--daily-trd", required=True)
    p.add_argument("--hourly-trd", required=True,
                   help="same run at hourly resolution; supplies the tracer "
                        "state the daily file lacks AND the exact-vs-"
                        "daily-mean error bar on the zdf operator")
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import netCDF4 as nc
    e1t, e2t, _e3t_ref, tmask = load_mesh_metrics(a.mesh_mask)

    dh = nc.Dataset(a.hourly_trd)
    nt = len(dh.dimensions["time_counter"])
    print(f"[hourly] {nt} records")

    # EXACT zdf variance decay: hour by hour on instantaneous fields.
    chi_exact_T = 0.0
    chi_exact_S = 0.0
    n_valid = 0
    Tsum = None
    Ssum = None
    e3sum = None
    for r in range(nt):
        T = _fill(dh.variables["votemper"][r])
        S = _fill(dh.variables["vosaline"][r])
        e3 = _fill(dh.variables["e3t"][r])
        ttz = _fill(dh.variables["ttrd_zdf"][r])
        stz = _fill(dh.variables["strd_zdf"][r])
        dV = e1t[None] * e2t[None] * e3
        # dV must be finite wherever the integral samples, or one bad cell
        # poisons the whole sum. A single NaN record silently NaN-ed the
        # entire exact budget on the first run, which is why every record is
        # now reported rather than only the total.
        wet = (np.isfinite(T) & np.isfinite(ttz) & np.isfinite(S)
               & np.isfinite(stz) & np.isfinite(dV) & (tmask > 0.5))
        nw = int(wet.sum())
        if nw == 0:
            # Name WHICH input ran out rather than reporting a bare NaN. The
            # daily budget still produced finite numbers while these records
            # were empty, which already says the STATE persists and only the
            # TRENDS stop -- a partial diagnostic window, not a corrupt file.
            print(f"  [rec {r:2d}] EMPTY: votemper={int(np.isfinite(T).sum())} "
                  f"ttrd_zdf={int(np.isfinite(ttz).sum())} "
                  f"strd_zdf={int(np.isfinite(stz).sum())} "
                  f"e3t={int(np.isfinite(e3).sum())}")
            continue
        n_valid += 1
        cT = chi(T, ttz, dV, wet)
        cS = chi(S, stz, dV, wet)
        if r < 3:
            print(f"  [rec {r:2d}] wet={nw:9d} chi_T={cT:+.6e} chi_S={cS:+.6e}")
        chi_exact_T += cT
        chi_exact_S += cS
        # The state average MUST come from exactly the records the exact chi
        # used, or the error bar compares two different sample windows.
        Tsum = T if Tsum is None else Tsum + T
        Ssum = S if Ssum is None else Ssum + S
        e3sum = e3 if e3sum is None else e3sum + e3
    if n_valid == 0:
        raise SystemExit("FATAL: no hourly record carries both state and trends")
    chi_exact_T /= n_valid
    chi_exact_S /= n_valid
    Tbar, Sbar, e3bar = Tsum / n_valid, Ssum / n_valid, e3sum / n_valid
    print(f"[exact] averaged over {n_valid} of {nt} hourly records")
    print(f"[exact]  chi_zdf theta {chi_exact_T:+.6e}   salt {chi_exact_S:+.6e}")

    dd = nc.Dataset(a.daily_trd)
    e3d = _fill(dd.variables["e3t"][0])
    dV = e1t[None] * e2t[None] * e3d
    wetT = np.isfinite(Tbar) & (tmask > 0.5)

    # The SAME operator from daily means -> the measured cost of the
    # product-of-means approximation.
    ttz_d = _fill(dd.variables["ttrd_zdf"][0])
    stz_d = _fill(dd.variables["strd_zdf"][0])
    chi_approx_T = chi(Tbar, ttz_d, dV, wetT & np.isfinite(ttz_d))
    chi_approx_S = chi(Sbar, stz_d, dV, wetT & np.isfinite(stz_d))
    rT = chi_approx_T / chi_exact_T if chi_exact_T else float("nan")
    rS = chi_approx_S / chi_exact_S if chi_exact_S else float("nan")
    print(f"[approx] chi_zdf theta {chi_approx_T:+.6e}   salt {chi_approx_S:+.6e}")
    print(f"[ERROR BAR] daily-mean / exact = {rT:.4f} (theta), {rS:.4f} (salt). "
          "Far from 1.0 means the daily-mean terms below are NOT quotable.")

    res = {"chi_zdf_exact": {"theta": chi_exact_T, "salt": chi_exact_S},
           "chi_zdf_daily_mean": {"theta": chi_approx_T, "salt": chi_approx_S},
           "daily_mean_over_exact": {"theta": rT, "salt": rS},
           "n_hourly_records": nt, "operators": {}}

    for tag, ops, tr in (("theta", T_OPS, Tbar), ("salt", S_OPS, Sbar)):
        print(f"\n=== {tag}: variance decay by operator (negative = destroys "
              "variance) ===")
        terms = {}
        for op in ops:
            if op not in dd.variables:
                print(f"  {op:14s} ABSENT")
                continue
            f = _fill(dd.variables[op][0])
            if f.ndim == 2:      # 2-D surface term: acts on the top level only
                full = np.zeros_like(tr)
                full[0] = f
                f = full
            nz = int(np.count_nonzero(np.nan_to_num(f)))
            if nz == 0:
                print(f"  {op:14s} all-zero, skipped")
                continue
            c = chi(tr, f, dV, wetT & np.isfinite(f))
            terms[op] = c
        tot = sum(terms.values())
        # Share is taken against the sum of ABSOLUTE terms, never against the
        # signed sum: these terms have mixed signs and nearly cancel, so
        # "% of the sum" produced 213% and -197% on the first run -- a
        # meaningless denominator presented as a share.
        scale = sum(abs(v) for v in terms.values()) or 1.0
        for op, c in sorted(terms.items(), key=lambda kv: kv[1]):
            print(f"  {op:14s} chi = {c:+.6e}   "
                  f"({100.0 * abs(c) / scale:6.2f}% of total activity)")
        print(f"  {'NET':14s} chi = {tot:+.6e}   "
              f"(gross activity {scale:.6e}; NET is not a closure -- no "
              "archived total exists)")
        res["operators"][tag] = {"terms": terms, "sum": tot}

    (out / "variance_budget.json").write_text(json.dumps(res, indent=2))
    print(f"\n[report] {out / 'variance_budget.json'}")
    print("NOTE: no archived total trend exists (ttrd_tot is all-zero), so "
          "this SUM is over available operators only and is NOT a closure "
          "test. Missing: surface non-solar for salt, and anything NEMO does "
          "not diagnose separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
