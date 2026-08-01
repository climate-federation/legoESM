#!/usr/bin/env python
"""Numerical AMIP skill-score gate against reference climatology.

Complements the ClimateEval visual/HTML pipeline with a scriptable pass/fail
verdict: area-weighted global-mean bias of the key AMIP fields against standard
observational references (CERES-EBAF for TOA/surface radiation, GPCP for precip,
ERA5/obs climatology for tas / CWV / cloud).  This is a MAGNITUDE / global-mean
gate — it certifies that the simulated climate means are within tolerance of
observations; it does NOT replace the spatial-pattern RMSE and seasonal-cycle
inspection that ClimateEval provides (run both).

Usage::

    python -m scripts.validate.amip_skill_score <run_dir>            # CMOR Amon
    python -m scripts.validate.amip_skill_score <run_dir> --json out.json

Exit code 0 if every gated field is within tolerance, 1 otherwise.

Reference global-annual means (with a tolerance band each):
  tas   287.5 K   (ERA5 2m air temperature)
  pr      2.8 mm/day (GPCP v2.3)
  rlut  240.0 W/m^2 (CERES-EBAF OLR)
  rsut   99.0 W/m^2 (CERES-EBAF reflected SW)
  rsdt  340.0 W/m^2 (CERES-EBAF incident SW)
  hfls   88.0 W/m^2 (surface latent heat, obs ~80-90)
  hfss   20.0 W/m^2 (surface sensible heat)
  prw    24.5 kg/m^2 (ERA5 column water vapour)
  clt    67.0 %      (ISCCP total cloud fraction)
  netTOA  0.9 W/m^2 (CERES-EBAF net = rsdt - rsut - rlut, near balance)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# --- reference climatology: field -> (reference_value, tolerance, units, source)
REFERENCE = {
    "tas":    (287.5, 2.0, "K",      "ERA5 2m air T"),
    "pr":     (2.8,   0.8, "mm/day", "GPCP v2.3"),
    "rlut":   (240.0, 8.0, "W/m^2",  "CERES-EBAF OLR"),
    "rsut":   (99.0,  8.0, "W/m^2",  "CERES-EBAF reflected SW"),
    "hfls":   (88.0, 15.0, "W/m^2",  "surface latent heat"),
    "hfss":   (20.0,  8.0, "W/m^2",  "surface sensible heat"),
    "prw":    (24.5,  4.0, "kg/m^2", "ERA5 column water vapour"),
    "clt":    (67.0, 10.0, "%",      "ISCCP total cloud"),
    "netTOA": (0.9,  10.0, "W/m^2",  "CERES-EBAF net TOA (rsdt-rsut-rlut)"),
}

# CMIP variable -> scale factor to the reference unit (CMOR SI -> display unit)
_SCALE = {"pr": 86400.0}  # kg/m^2/s -> mm/day; others already in ref units


def _load_var(cmor_dir: Path, var: str):
    """Load a CMOR variable's full array (time, lat, lon) or grid field."""
    try:
        import netCDF4 as nc
    except Exception:  # pragma: no cover - env fallback
        import xarray as xr
        ds = xr.open_dataset(_amon_path(cmor_dir, var))
        return np.asarray(ds[var].values)
    ds = nc.Dataset(_amon_path(cmor_dir, var))
    a = np.array(ds.variables[var][:])
    ds.close()
    return a


def _amon_path(cmor_dir: Path, var: str) -> Path:
    # Amon fields live in Amon/, grid fields (areacella, orog) in fx/ — search
    # any CMOR table subdir for ``<var>_<table>_*.nc``.
    hits = sorted(cmor_dir.glob(f"*/{var}_*.nc"))
    if not hits:
        raise FileNotFoundError(f"no CMOR file for {var!r} under {cmor_dir}")
    return hits[0]


def area_weighted_global_mean(field: np.ndarray, area: np.ndarray) -> float:
    """Area-weighted global mean over the horizontal axes.

    ``field`` may carry a leading time axis, which is averaged (time MEAN over
    every month the run wrote) — the reference values are annual/climatological
    means, so scoring a single month against them conflates the seasonal cycle
    with model bias (caught 2026-08-01: last-DECEMBER netTOA read +12.9 W/m2
    on a run whose annual mean was -2.9).  ``area`` is the grid-cell area
    (areacella).  NaNs and zero-area cells are excluded; a cell with ANY NaN
    month is dropped whole (plain mean, not nanmean) so partial coverage can
    never silently pass as a full-window mean.
    """
    f = np.asarray(field, dtype=float)
    if f.ndim == area.ndim + 1:
        f = f.mean(axis=0)
    w = np.asarray(area, dtype=float)
    ok = np.isfinite(f) & (w > 0)
    if not np.any(ok):
        return float("nan")
    return float(np.sum(f[ok] * w[ok]) / np.sum(w[ok]))


def score_amip_run(cmor_dir: str | Path) -> dict:
    """Return a per-field skill dict for a CMOR AMIP run directory.

    Each entry: {value, reference, bias, tol, pass, units, source}.  ``netTOA``
    is derived from rsdt - rsut - rlut.  Missing fields are skipped (not failed).
    """
    cmor_dir = Path(cmor_dir)
    area = _load_var(cmor_dir, "areacella")
    means: dict[str, float] = {}
    for var in ("tas", "pr", "rlut", "rsut", "rsdt", "hfls", "hfss", "prw", "clt"):
        try:
            raw = _load_var(cmor_dir, var) * _SCALE.get(var, 1.0)
            means[var] = area_weighted_global_mean(raw, area)
        except FileNotFoundError:
            continue
    if all(k in means for k in ("rsdt", "rsut", "rlut")):
        means["netTOA"] = means["rsdt"] - means["rsut"] - means["rlut"]

    results: dict[str, dict] = {}
    for field, (ref, tol, units, source) in REFERENCE.items():
        if field not in means:
            continue
        val = means[field]
        bias = val - ref
        results[field] = {
            "value": val, "reference": ref, "bias": bias, "tol": tol,
            "pass": bool(np.isfinite(val) and abs(bias) <= tol),
            "units": units, "source": source,
        }
    return results


def format_report(results: dict) -> tuple[str, bool]:
    """Human-readable table + overall pass boolean."""
    lines = [
        f"{'field':7s} {'sim':>9s} {'ref':>9s} {'bias':>9s} {'tol':>7s}  verdict  source",
        "-" * 78,
    ]
    all_pass = True
    for field, r in results.items():
        ok = r["pass"]
        all_pass &= ok
        lines.append(
            f"{field:7s} {r['value']:9.2f} {r['reference']:9.2f} "
            f"{r['bias']:+9.2f} {r['tol']:7.1f}  {'PASS' if ok else 'FAIL':4s}  "
            f"{r['source']} [{r['units']}]"
        )
    lines.append("-" * 78)
    n_pass = sum(r["pass"] for r in results.values())
    lines.append(f"OVERALL: {n_pass}/{len(results)} fields within tolerance "
                 f"-> {'PASS' if all_pass else 'FAIL'}")
    return "\n".join(lines), all_pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", help="AMIP run dir containing cmor/ (or the cmor/ dir)")
    ap.add_argument("--json", type=str, default=None, help="write results as JSON")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    cmor = run / "cmor" if (run / "cmor").is_dir() else run
    results = score_amip_run(cmor)
    if not results:
        print(f"No gated fields found under {cmor} (missing Amon output?)", file=sys.stderr)
        return 2
    report, ok = format_report(results)
    print(report)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
