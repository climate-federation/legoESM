"""Score the P-model kphio x beta_cost screen against the pre-registered rules.

Protocol: ``docs/land/pmodel_kphio_beta_calibration_prereg.md`` (fixed BEFORE
the sweep ran).  Reads one output directory per grid point
(``diagnostics/ec_site_calib/k<kphio>_b<beta>/``), computes the pre-registered
objective, and applies the stopping rules mechanically:

* objective = mean over TRAINING sites of ``0.5*nRMSE(GPP) + 0.5*nRMSE(LE)``
  with ``nRMSE = RMSE / sd(obs)`` (latent heat is IN the objective, not a guard)
* a grid point is scored only if EVERY training site produced an output
* an optimum on a grid BOUNDARY is reported UNRESOLVED, not calibrated
* per-site numbers for the winner are printed, including regressions

Skill math is imported from ``plot_ec_site_xsite`` (single home), the same code
that produced the published ladder.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util as _ilu
import os

import numpy as np
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location(
    "_xsite", os.path.join(_HERE, "..", "plot", "plot_ec_site_xsite.py"))
_xsite = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_xsite)

TRAIN = ["US-MMS", "FI-Hyy", "DE-Gri", "US-Ne1", "US-Ton", "US-Whs", "FR-Pue"]
# Grid as the LITERAL strings the sweep wrapper uses in its directory tags
# (run_pmodel_calib_sweep.sbatch), so tag construction cannot drift from the
# directories on disk.
KPHIO_GRID = ["0.040", "0.055", "0.070", "0.081785", "0.100"]
BETA_GRID = ["60", "92", "146", "224", "344"]
DEFAULT = ("0.081785", "146")


def _nrmse(ds, key):
    """RMSE normalised by the observed standard deviation on the SAME mask."""
    s = _xsite._skill(ds, key)
    base = (ds["score_valid"].values if "score_valid" in ds
            else ds["valid"].values).astype(bool)
    m, o = _xsite._pair(ds, key)
    mask = base & np.isfinite(m) & np.isfinite(o)
    if not np.isfinite(s["rmse"]) or mask.sum() < 10:
        return np.nan, s
    sd = float(np.std(o[mask]))
    return (s["rmse"] / sd if sd > 0 else np.nan), s


def score_point(base_dir, tag, sites):
    """(objective, {site: {flux: (nrmse, skill)}}) or (nan, partial) if any
    site is missing — a grid point may not win by dropping a hard site."""
    per = {}
    for site in sites:
        hits = glob.glob(os.path.join(base_dir, tag, f"{site}_*_ec_*.nc"))
        if not hits:
            return np.nan, per
        ds = xr.open_dataset(hits[0])
        per[site] = {f: _nrmse(ds, k) for k, f in (("gpp", "GPP"),
                                                   ("le", "LE"))}
        ds.close()
    vals = [0.5 * per[s]["GPP"][0] + 0.5 * per[s]["LE"][0] for s in sites]
    if not np.all(np.isfinite(vals)):
        return np.nan, per
    return float(np.mean(vals)), per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default="diagnostics/ec_site_calib")
    ap.add_argument("--sites", nargs="+", default=TRAIN)
    args = ap.parse_args()

    rows = []
    for b in BETA_GRID:
        for k in KPHIO_GRID:
            tag = f"k{k}_b{b}"
            obj, per = score_point(args.base_dir, tag, args.sites)
            rows.append((k, b, tag, obj, per))

    print(f"objective = mean over {len(args.sites)} training sites of "
          "0.5*nRMSE(GPP) + 0.5*nRMSE(LE)   (lower better)\n")
    print("kphio     beta   objective")
    for k, b, _, obj, _ in rows:
        print(f"{k:<9s} {b:<6s} " + ("   n/a (incomplete)"
                                       if not np.isfinite(obj)
                                       else f"{obj:.4f}"))

    done = [r for r in rows if np.isfinite(r[3])]
    if not done:
        raise SystemExit("no complete grid point - nothing to score")
    best = min(done, key=lambda r: r[3])
    ctrl = [r for r in done if (r[0], r[1]) == DEFAULT]
    print(f"\nbest: kphio={best[0]} beta={best[1]} objective={best[3]:.4f}")
    if ctrl:
        print(f"default control (kphio={DEFAULT[0]}, beta={DEFAULT[1]}): "
              f"objective={ctrl[0][3]:.4f}  -> improvement "
              f"{100 * (ctrl[0][3] - best[3]) / ctrl[0][3]:+.1f}%")
    else:
        print("default control point MISSING - improvement not quantifiable")

    edge = (best[0] in (KPHIO_GRID[0], KPHIO_GRID[-1])
            or best[1] in (BETA_GRID[0], BETA_GRID[-1]))
    print("\nVERDICT: " + ("UNRESOLVED - the optimum sits on a grid boundary, "
                          "so the true optimum is outside the screened range; "
                          "no tuned value is recommended (pre-registered rule 1)"
                          if edge else
                          "interior optimum - a tuned pair is recommendable "
                          "PENDING the held-out half (pre-registered rule 2)"))

    print("\nper-site at the best point (nRMSE / bias):")
    for site in args.sites:
        g, l = best[4][site]["GPP"], best[4][site]["LE"]
        cg = ctrl[0][4][site]["GPP"] if ctrl else (np.nan, {"bias": np.nan})
        flag = ("  REGRESSION vs default"
                if ctrl and g[0] > cg[0] else "")
        print(f"  {site:8s} GPP {g[0]:.3f} (bias {g[1]['bias']:+.2f})  "
              f"LE {l[0]:.3f} (bias {l[1]['bias']:+.1f}){flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
