#!/usr/bin/env python
"""Validate legoESM test matrix outputs and emit a report.

Reads results/<domain>/summary.json for atmosphere, ocean, sea_ice and:
  1. Checks every case's snapshots_latlon.npz / snapshots_native.npz for
     NaN/Inf and physical-range violations.
  2. Reads conservation_timeseries.csv (mass/energy/volume/heat/salt) and
     flags drift beyond thresholds.
  3. Reads mean_timeseries.csv for runaway growth.
  4. Groups wall-time per (grid_type, case) to highlight GPU perf bottlenecks.
  5. Compares against *_previous.json snapshots (if present) to catch
     status / perf / conservation regressions.
  6. Emits a Markdown report to results/validation_report.md (and stdout).

Only stdlib + numpy + pandas (already in the project deps).
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RESULTS_ROOT = Path("results")
DOMAINS = ("atmosphere", "ocean", "sea_ice")

# --- Physical thresholds (from .claude/agents/validate-matrix.md) --------
# Keys are exact field names in snapshots_*.npz. Unknown names are ignored.
ATMO_RANGES = {
    "T": (150.0, 350.0),
    "T_3d": (150.0, 350.0),
    "temperature": (150.0, 350.0),
    "p_s": (7.5e4, 1.1e5),
    "surface_pressure": (7.5e4, 1.1e5),
    "u": (-200.0, 200.0),
    "v": (-200.0, 200.0),
    "wind_speed": (0.0, 250.0),
    "q_v": (0.0, 0.04),
}
OCEAN_RANGES = {
    "T": (-5.0, 45.0),
    "T_3d": (-5.0, 45.0),
    "SST": (-5.0, 45.0),
    "temperature": (-5.0, 45.0),
    "S": (0.0, 50.0),
    "salinity": (0.0, 50.0),
    "eta": (-100.0, 100.0),
    "ssh": (-100.0, 100.0),
    "u": (-10.0, 10.0),
    "v": (-10.0, 10.0),
    "speed": (0.0, 10.0),
}
SEA_ICE_RANGES = {
    "h": (0.0, 30.0),
    "h_ice": (0.0, 30.0),
    "T_ice": (180.0, 271.4),
    "T_srf": (180.0, 273.2),
    "a_ice": (0.0, 1.0),
    "concentration": (0.0, 1.0),
    "u_ice": (-5.0, 5.0),
    "v_ice": (-5.0, 5.0),
}

# Always skip these — they are coordinate axes, timestamps, step counts,
# grid metrics, or otherwise not physical state variables.
SKIP_FIELDS = {
    "times_days", "time", "t", "steps", "step", "day", "days",
    "lat", "lon", "latitude", "longitude", "latCell", "lonCell",
    "level", "levels", "sigma", "height", "bins", "bin_centers",
    "area", "dx", "dy", "dz", "cos_angle", "sin_angle", "radius",
    "land_mask", "ocean_mask", "u_mask", "v_mask",
    "lat_plot", "lon_plot",
}

CONS_THRESH = {
    "atmosphere": {"mass_rel_drift": 0.001, "energy_rel_drift": 0.05},
    "ocean": {"volume_rel_drift": 0.0001, "heat_rel_drift": 0.01,
              "salt_rel_drift": 0.01},
    "sea_ice": {"volume_rel_drift": 0.05, "mass_rel_drift": 0.01},
}

# The "grew Nx" runaway heuristic divides final/initial, so it is meaningless
# for a drift/anomaly column that is ~0 by design (e.g. a machine-precision
# volume anomaly, ~1e-16): the ratio of two near-zero numbers explodes to a
# spurious 500x / -4096x.  Only evaluate it when the series carries a
# physically meaningful magnitude.
_RUNAWAY_MIN_MAGNITUDE = 1e-6


@dataclass
class Finding:
    severity: str        # INFO, WARN, FAIL, REGRESSION
    domain: str
    case: str
    grid: str
    message: str


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _find_case_dir(domain: str, rec: dict) -> Path | None:
    """Locate the per-case output directory from a summary record."""
    base = RESULTS_ROOT / domain
    # Atmosphere: <equation_set>/<case>/<grid_type>/<resolution>/[<vert>]
    candidates: list[Path] = []
    eq = rec.get("equation_set", "")
    case = rec.get("test") or rec.get("case") or ""
    grid = rec.get("grid", "")
    resolution = rec.get("resolution", "")
    vert = rec.get("vertical_coord", "") or ""

    if domain == "atmosphere":
        pat = base / eq / case / grid / resolution
        if vert and vert != "none":
            pat = pat / vert
        candidates.append(pat)
    elif domain == "ocean":
        candidates.append(base / case / grid / resolution)
        # rest_state_* variants are grouped one level down under rest_state/
        # (results/ocean/rest_state/<case>/<grid>/<res>) by the ocean runner.
        # Require the trailing underscore so an unrelated "rest_stateful_*" case
        # cannot match the grouped path.  (Candidate is only used if is_dir.)
        if case.startswith("rest_state_"):
            candidates.append(base / "rest_state" / case / grid / resolution)
    else:  # sea_ice
        category = rec.get("category", "")
        if category:
            candidates.append(base / category / case / grid / resolution)
        candidates.append(base / case / grid / resolution)

    for c in candidates:
        if c.is_dir():
            return c
    # fall back: first directory matching <case> and <grid> anywhere in tree
    for p in base.rglob(case):
        if p.is_dir() and grid in p.parts:
            return p
    return None


def _load_npz_safely(path: Path) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    except Exception as e:  # pragma: no cover
        return None


def _fraction_out_of_range(arr: np.ndarray, lo: float, hi: float) -> float:
    finite = np.isfinite(arr)
    if not finite.any():
        return math.nan
    mask = (arr < lo) | (arr > hi)
    return float(mask[finite].mean())


def _range_check(findings: list[Finding], domain: str, case: str, grid: str,
                 arrays: dict[str, np.ndarray], ranges: dict[str, tuple],
                 eq_set: str = ""):
    # Shallow-water cases have no true tropospheric surface pressure: the runner
    # stores a derived `p_s` (rho*g*h from the SW layer thickness) only for
    # plotting, and a large-amplitude Rossby-Haurwitz layer swings it well past
    # realistic-atmosphere bounds.  Do not range-check SW `p_s` against the
    # hydrostatic surface-pressure limits (hydrostatic p_s is still checked).
    skip_fields = (
        {"p_s", "surface_pressure"} if eq_set == "shallow_water" else set()
    )
    # Reference land_mask: regional ocean/atmosphere cases regrid to a global
    # 181x360 mesh and leave out-of-domain cells as NaN by design, so we only
    # count NaN/Inf in cells the model actually integrates.
    land_mask = arrays.get("land_mask")
    active = None
    if land_mask is not None and land_mask.ndim >= 2:
        # Ocean cells: non-NaN and ≠ 0 (land_mask==1 for ocean in legoESM).
        # Take the first time-slice as the static mask.  Only use it as an
        # (nlat, nlon) spatial mask when it really is 2-D — an unstructured
        # (time, ncells) or an unexpected-rank mask is left unapplied rather
        # than crashing the unpack below.
        lm0 = land_mask[0] if land_mask.ndim >= 3 else land_mask
        if lm0.ndim == 2:
            active = np.isfinite(lm0) & (lm0 > 0.5)

    def _select_active(arr: np.ndarray):
        """Return a 1-D view of the cells the model actually integrates, using
        the (nlat, nlon) `active` mask.  Locate the consecutive spatial axes in
        arr explicitly so it works for (lat,lon), (time,lat,lon),
        (time,lat,lon,nlev) and (lat,lon,nlev).  Falls back to the full array
        only when no (nlat,nlon) axis pair is found.

        NOTE: the previous shape-guessing loop broadcast the mask to
        (nlat,nlon,1,1) for a 4-D (time,lat,lon,nlev) field, which does NOT
        align with (time,lat,lon,nlev); the broadcast raised and the except
        clause returned the FULL array, so out-of-domain NaN fill in regional /
        channel cases (regridded onto the global 181x360 mesh) was counted as
        active-cell NaN — every ``*_3d`` field in those cases false-FAILed."""
        if active is None:
            return arr
        nlat, nlon = active.shape
        # Scan right-to-left so the (lat, lon) pair is matched ahead of any
        # leading axis of a coincidentally-equal size (e.g. a square native
        # grid where time or nlev also equals nlat): lat/lon sit to the right
        # of a leading time axis and to the left of a trailing nlev axis.
        for ax in range(arr.ndim - 2, -1, -1):
            if arr.shape[ax] == nlat and arr.shape[ax + 1] == nlon:
                shp = [1] * arr.ndim
                shp[ax], shp[ax + 1] = nlat, nlon
                mask = active.reshape(shp)
                return arr[np.broadcast_to(mask, arr.shape)]
        return arr

    for name, arr in arrays.items():
        if name in SKIP_FIELDS or name in skip_fields:
            continue
        if arr.dtype.kind not in "fc":
            continue
        if arr.size == 0:
            continue
        checked = _select_active(arr) if arr.ndim >= 2 else arr
        if checked.size == 0:
            continue
        nan_count = int(np.isnan(checked).sum())
        inf_count = int(np.isinf(checked).sum())
        if nan_count or inf_count:
            frac = (nan_count + inf_count) / checked.size
            # Only flag if a *non-trivial* fraction of active cells is bad;
            # a few isolated NaN near a coastline is often just regridding.
            sev = "FAIL" if frac > 0.01 else "WARN"
            findings.append(Finding(
                sev, domain, case, grid,
                f"{name}: {nan_count} NaN, {inf_count} Inf "
                f"({frac*100:.2f}% of active cells)"))
            continue
        if name in ranges:
            lo, hi = ranges[name]
            amin, amax = float(np.nanmin(checked)), float(np.nanmax(checked))
            if amin < lo or amax > hi:
                frac = _fraction_out_of_range(checked, lo, hi)
                sev = "FAIL" if frac > 1e-4 else "WARN"
                findings.append(Finding(
                    sev, domain, case, grid,
                    f"{name} range [{amin:.3g}, {amax:.3g}] violates "
                    f"[{lo}, {hi}] ({frac*100:.2f}% of cells)"))
            continue
        amax_abs = float(np.max(np.abs(checked)))
        if amax_abs > 1e12:
            findings.append(Finding(
                "WARN", domain, case, grid,
                f"{name} |max|={amax_abs:.3g} is suspiciously large"))


def _conservation_check(findings: list[Finding], domain: str, case: str,
                        grid: str, case_dir: Path):
    csv_path = case_dir / "conservation_timeseries.csv"
    if not csv_path.is_file():
        return
    try:
        rows = list(csv.DictReader(csv_path.open()))
    except Exception:
        return
    if not rows:
        return
    thresholds = CONS_THRESH.get(domain, {})
    # Collect time series per numeric column
    cols = {k: [] for k in rows[0] if k not in ("time", "step", "day", "t")}
    for r in rows:
        for k in cols:
            try:
                cols[k].append(float(r[k]))
            except (ValueError, KeyError):
                pass
    # Direct relative-drift columns
    for k, vals in cols.items():
        if not vals:
            continue
        v0, vN = vals[0], vals[-1]
        thr = thresholds.get(k)
        if k.endswith("_rel_drift") and thr is not None:
            peak = max((abs(v) for v in vals), default=0.0)
            if peak > thr:
                findings.append(Finding(
                    "FAIL", domain, case, grid,
                    f"{k} peak={peak:.3e} exceeds {thr:.3e}"))
        # Runaway growth.  Only skip series whose whole magnitude is machine-
        # negligible (a drift/anomaly column that is ~0 by design: the ratio of
        # two ~1e-16 numbers explodes to a spurious 500x).  A genuinely small-
        # but-real baseline that blows up (e.g. 1e-12 -> 1e-4) still trips the
        # ratio, since series_max clears the floor and v0 != 0.
        series_max = max((abs(v) for v in vals), default=0.0)
        if series_max > _RUNAWAY_MIN_MAGNITUDE and v0 != 0 and abs(vN / v0) > 10:
            findings.append(Finding(
                "WARN", domain, case, grid,
                f"{k} grew {vN/v0:.2f}x from t=0 to t=end"))


def _mean_sanity(findings: list[Finding], domain: str, case: str, grid: str,
                 case_dir: Path):
    csv_path = case_dir / "mean_timeseries.csv"
    if not csv_path.is_file():
        return
    try:
        rows = list(csv.DictReader(csv_path.open()))
    except Exception:
        return
    if not rows:
        return
    cols = {k: [] for k in rows[0] if k not in ("time", "step", "day", "t")}
    for r in rows:
        for k in cols:
            try:
                cols[k].append(float(r[k]))
            except (ValueError, KeyError):
                pass
    for k, vals in cols.items():
        if not vals:
            continue
        if any(not math.isfinite(v) for v in vals):
            findings.append(Finding(
                "FAIL", domain, case, grid,
                f"mean_timeseries {k} has non-finite values"))
        if k.lower().startswith("t_") or k.lower().startswith("temp"):
            drift = vals[-1] - vals[0]
            if abs(drift) > 5:
                findings.append(Finding(
                    "WARN", domain, case, grid,
                    f"mean {k} drifted {drift:+.2f} over the run"))


# -------------------------------------------------------------------------
# Per-case validation
# -------------------------------------------------------------------------

def validate_domain(domain: str) -> tuple[list[Finding], list[dict]]:
    findings: list[Finding] = []
    summary_path = RESULTS_ROOT / domain / "summary.json"
    if not summary_path.is_file():
        findings.append(Finding("WARN", domain, "-", "-",
                                f"summary.json missing: {summary_path}"))
        return findings, []
    summary = json.loads(summary_path.read_text())
    records = summary.get("results", [])

    ranges = {"atmosphere": ATMO_RANGES, "ocean": OCEAN_RANGES,
              "sea_ice": SEA_ICE_RANGES}[domain]

    for rec in records:
        status = rec.get("status", "?")
        case = rec.get("test") or rec.get("case") or "?"
        grid = rec.get("grid", "?")
        if status == "ERROR":
            findings.append(Finding(
                "FAIL", domain, case, grid,
                f"ERROR: {rec.get('notes', '')}"))
            continue
        if status == "SKIP":
            findings.append(Finding(
                "INFO", domain, case, grid,
                f"SKIPPED: {rec.get('notes', '')}"))
            continue
        case_dir = _find_case_dir(domain, rec)
        if case_dir is None:
            findings.append(Finding(
                "WARN", domain, case, grid,
                "No output directory found — cannot validate artifacts"))
            continue
        npz = case_dir / "snapshots_latlon.npz"
        if not npz.is_file():
            npz = case_dir / "snapshots_native.npz"
        if npz.is_file():
            arrays = _load_npz_safely(npz)
            if arrays is not None:
                _range_check(findings, domain, case, grid, arrays, ranges,
                             rec.get("equation_set", ""))
        _conservation_check(findings, domain, case, grid, case_dir)
        _mean_sanity(findings, domain, case, grid, case_dir)

        if status == "FAIL":
            findings.append(Finding(
                "FAIL", domain, case, grid,
                f"Matrix FAIL: {rec.get('notes', '')}"))

    return findings, records


# -------------------------------------------------------------------------
# Regression detection
# -------------------------------------------------------------------------

def regressions(domain: str, records: list[dict]) -> list[Finding]:
    prev_path = RESULTS_ROOT / domain / "summary_previous.json"
    if not prev_path.is_file():
        return []
    prev = json.loads(prev_path.read_text())
    prev_idx = {
        (r.get("equation_set", ""), r.get("test") or r.get("case"),
         r.get("grid"), r.get("resolution")): r
        for r in prev.get("results", [])
    }
    out: list[Finding] = []
    for rec in records:
        key = (rec.get("equation_set", ""),
               rec.get("test") or rec.get("case"),
               rec.get("grid"), rec.get("resolution"))
        prior = prev_idx.get(key)
        if not prior:
            continue
        if prior["status"] == "PASS" and rec["status"] != "PASS":
            out.append(Finding(
                "REGRESSION", domain, key[1], key[2],
                f"Status regression: PASS → {rec['status']}"))
        cur_w = float(rec.get("wall_time") or 0)
        prv_w = float(prior.get("wall_time") or 0)
        if prv_w and cur_w > 1.5 * prv_w and cur_w > 5.0:
            out.append(Finding(
                "WARN", domain, key[1], key[2],
                f"Wall time +{(cur_w/prv_w - 1)*100:.0f}% "
                f"(prev {prv_w:.1f}s, now {cur_w:.1f}s)"))
    return out


# -------------------------------------------------------------------------
# Per-grid performance summary
# -------------------------------------------------------------------------

def grid_performance(domain: str, records: list[dict]) -> dict[str, dict]:
    g: dict[str, list[float]] = defaultdict(list)
    g_pass: dict[str, int] = defaultdict(int)
    g_fail: dict[str, int] = defaultdict(int)
    g_error: dict[str, int] = defaultdict(int)
    for r in records:
        grid = r.get("grid", "?")
        w = float(r.get("wall_time") or 0)
        if w > 0:
            g[grid].append(w)
        s = r.get("status", "?")
        if s == "PASS":
            g_pass[grid] += 1
        elif s == "FAIL":
            g_fail[grid] += 1
        elif s == "ERROR":
            g_error[grid] += 1
    out = {}
    for grid, times in g.items():
        arr = np.asarray(times)
        out[grid] = {
            "n": int(arr.size),
            "pass": g_pass[grid],
            "fail": g_fail[grid],
            "error": g_error[grid],
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "max": float(arr.max()),
            "sum": float(arr.sum()),
        }
    return out


# -------------------------------------------------------------------------
# Report writer
# -------------------------------------------------------------------------

def render_report(all_findings: dict[str, list[Finding]],
                  all_records: dict[str, list[dict]],
                  all_perf: dict[str, dict[str, dict]]) -> str:
    lines = ["# legoESM Test Matrix — GPU Validation Report", ""]
    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    for dom in DOMAINS:
        recs = all_records.get(dom, [])
        if not recs:
            lines.append(f"- **{dom}**: no summary found")
            continue
        npass = sum(1 for r in recs if r["status"] == "PASS")
        nfail = sum(1 for r in recs if r["status"] == "FAIL")
        nerr = sum(1 for r in recs if r["status"] == "ERROR")
        nskip = sum(1 for r in recs if r["status"] == "SKIP")
        total = len(recs)
        lines.append(f"- **{dom}**: {npass}/{total} PASS | "
                     f"{nfail} FAIL | {nerr} ERROR | {nskip} SKIP")
    lines.append("")

    # Per-grid performance
    lines.append("## Per-grid GPU Performance")
    for dom in DOMAINS:
        perf = all_perf.get(dom, {})
        if not perf:
            continue
        lines.append(f"### {dom}")
        lines.append("")
        lines.append("| Grid | #cases | PASS | FAIL | ERR | mean (s) | "
                     "median (s) | max (s) | total (s) |")
        lines.append("|------|-------:|-----:|-----:|----:|---------:|"
                     "-----------:|--------:|----------:|")
        for grid in sorted(perf, key=lambda g: -perf[g]["sum"]):
            p = perf[grid]
            lines.append(
                f"| {grid} | {p['n']} | {p['pass']} | {p['fail']} | "
                f"{p['error']} | {p['mean']:.1f} | {p['median']:.1f} | "
                f"{p['max']:.1f} | {p['sum']:.1f} |")
        lines.append("")

    # Findings
    lines.append("## Findings (validation + regression)")
    lines.append("")
    for dom, findings in all_findings.items():
        if not findings:
            lines.append(f"- **{dom}**: no issues")
            continue
        lines.append(f"### {dom}")
        for sev in ("REGRESSION", "FAIL", "WARN", "INFO"):
            rows = [f for f in findings if f.severity == sev]
            if not rows:
                continue
            lines.append(f"**{sev} ({len(rows)})**")
            lines.append("")
            for f in rows:
                lines.append(f"- `{f.case}` / `{f.grid}`: {f.message}")
            lines.append("")

    # Fix prompts — top issues
    lines.append("## Suggested Fix Prompts (highest priority first)")
    lines.append("")
    priorities = []
    for dom, findings in all_findings.items():
        for f in findings:
            score = {"REGRESSION": 0, "FAIL": 1, "WARN": 2, "INFO": 3}[f.severity]
            priorities.append((score, dom, f))
    priorities.sort(key=lambda x: x[0])
    for _, dom, f in priorities[:10]:
        lines.append(
            f"- **[{f.severity}]** `{dom}/{f.case}/{f.grid}` — {f.message}")
        lines.append(
            f"  Reproduce: "
            f"`JAX_ENABLE_X64=1 .venv/bin/python scripts/run_{dom}_test_matrix.py "
            f"--only {f.case} --grid {f.grid} --quick`")
    lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main() -> int:
    all_findings: dict[str, list[Finding]] = {}
    all_records: dict[str, list[dict]] = {}
    all_perf: dict[str, dict[str, dict]] = {}
    for dom in DOMAINS:
        f, recs = validate_domain(dom)
        f += regressions(dom, recs)
        all_findings[dom] = f
        all_records[dom] = recs
        all_perf[dom] = grid_performance(dom, recs)
    report = render_report(all_findings, all_records, all_perf)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "validation_report.md").write_text(report)
    print(report)
    # Exit non-zero if any FAIL / REGRESSION
    bad = sum(1 for dom in all_findings
              for f in all_findings[dom]
              if f.severity in ("FAIL", "REGRESSION"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
