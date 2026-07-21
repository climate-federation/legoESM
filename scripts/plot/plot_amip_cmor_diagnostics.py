#!/usr/bin/env python
"""Systematic AMIP climate diagnostics from a run's CMOR ``Amon`` output.

Produces one publication-style figure per run: a global map for each standard
field (near-surface air temperature, precipitation, TOA outgoing SW/LW, total
cloud fraction, column water vapour, surface latent/sensible heat flux), the
per-field zonal means, the top-of-atmosphere energy budget, and a table of
area-weighted global means annotated against Earth observational reference
values.  Radiation columns are area-weighted by the model's own sin-latitude
``areacella`` (reproduced here so the plotter needs only the ``Amon`` NetCDF,
not the separately-written ``fx`` table).

Pure ``numpy`` / ``xarray`` / ``matplotlib`` — imports no ``legoesm`` and runs
on the CMOR output of any AMIP/coupled run.

Usage::

    python scripts/plot/plot_amip_cmor_diagnostics.py <run_dir> [--label L] [--out fig.png]

``<run_dir>`` is the run directory that contains ``cmor/Amon/*.nc``.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import xarray as xr

# --- Observational targets + realism acceptance bands: the CERES/GPCP/ERA5
#     global-mean references and mechanical pass/fail tolerances live in the
#     dependency-light sibling ``_amip_obs_targets`` so BOTH AMIP scorecards
#     (this CMOR plotter and the timeseries ``plot_convection_scorecard``) grade
#     against ONE source without the timeseries scorecard having to import
#     xarray.  Imported here to preserve the module-level names this file uses. ---
import importlib.util as _ilu  # noqa: E402
_targets_spec = _ilu.spec_from_file_location(
    "_amip_obs_targets", str(Path(__file__).resolve().parent / "_amip_obs_targets.py"))
_targets = _ilu.module_from_spec(_targets_spec)
_targets_spec.loader.exec_module(_targets)
FIELD_TABLE = _targets.FIELD_TABLE
_ALBEDO_REF = _targets._ALBEDO_REF
_REALISM_ABS_TOL = _targets._REALISM_ABS_TOL
_ALBEDO_ABS_TOL = _targets._ALBEDO_ABS_TOL
_R_TOA_ABS_TOL = _targets._R_TOA_ABS_TOL
_REQUIRED_FIELDS = _targets._REQUIRED_FIELDS

# --- Structural-realism thresholds: first-order checks that the run has the
#     right large-scale STRUCTURE, not just the right global-mean scalars (a
#     globally-correct but structurally-dead run — e.g. an isothermal / uniform
#     initial-condition blow-up — passes the global-mean gate yet has no
#     equator-pole gradient and no ITCZ).  These are deliberately LOOSE physical
#     floors any realistic climate clears, chosen to catch a degenerate run, not
#     to grade fidelity — OBSERVATIONAL criteria, not model physics constants. ---
_TAS_TROPICS_MINUS_POLE_MIN_K = 20.0  # eq(|lat|<30)-minus-pole(|lat|>60) tas floor [K]
_PR_ITCZ_ENHANCEMENT_MIN = 1.05       # tropical(|lat|<15)/global mean-precip ratio floor
_TROPICS_LAT_DEG = 30.0               # |lat| bound for the "tropics" band [deg]
_POLE_LAT_DEG = 60.0                  # |lat| bound for the "pole" band [deg]
_ITCZ_LAT_DEG = 15.0                  # |lat| bound for the ITCZ/deep-tropics band [deg]


def _sinlat_area_weights(nlat: int, nlon: int) -> np.ndarray:
    """Reproduce the model's ``areacella`` weights (exact sin-latitude bands,
    matching ``driver/diagnostics.py::_write_cmip_fixed_files``), normalised to
    sum to 1 — the ``R_earth**2 * dlon`` prefactor cancels in a weighted mean."""
    sin_edges = np.sin(np.radians(np.linspace(-90.0, 90.0, nlat + 1)))
    band = sin_edges[1:] - sin_edges[:-1]
    w = np.broadcast_to(band[:, None], (nlat, nlon)).astype(np.float64)
    return w / w.sum()


def _band_weighted_mean(field: np.ndarray, lat: np.ndarray, w: np.ndarray,
                        lat_lo: float, lat_hi: float) -> float:
    """Area-weighted mean of ``field`` over rows with ``lat_lo <= |lat| < lat_hi``.
    Returns ``NaN`` if no row falls in the band (so a degenerate grid fails a
    structural check rather than silently passing)."""
    mask = (np.abs(lat) >= lat_lo) & (np.abs(lat) < lat_hi)
    if not bool(mask.any()):
        return float("nan")
    wm = w[mask, :]
    return float(np.sum(field[mask, :] * wm) / np.sum(wm))


def _load_clim(cmor_amon: str, var: str, spinup_frac: float = 0.0):
    """Time-mean climatology + lat/lon for a CMOR Amon variable, or Nones.

    ``spinup_frac`` (0..1) discards that leading fraction of the monthly record
    as spin-up BEFORE the time-mean, so a short (few-month) run is not graded on
    its initial-condition transient.  At least one time slice is always kept
    (``spinup_frac`` clamped so ``n0 <= ntime-1``); ``0.0`` (the default) means
    the full-record mean and is bit-for-bit the previous behaviour."""
    if not (0.0 <= spinup_frac < 1.0):
        raise ValueError(f"spinup_frac must be in [0, 1), got {spinup_frac!r}")
    fs = sorted(glob.glob(os.path.join(cmor_amon, f"{var}_Amon_*.nc")))
    if not fs:
        return None, None, None
    ds = xr.open_dataset(fs[0])
    da = ds[var]
    tdim = da.dims[0]
    latn = "lat" if "lat" in ds else ("latitude" if "latitude" in ds else None)
    lonn = "lon" if "lon" in ds else ("longitude" if "longitude" in ds else None)
    lat = ds[latn].values if latn else np.arange(da.shape[-2])
    lon = ds[lonn].values if lonn else np.arange(da.shape[-1])
    ntime = da.sizes[tdim]
    n0 = min(int(spinup_frac * ntime), ntime - 1)   # keep >= 1 slice
    if n0 > 0:
        da = da.isel({tdim: slice(n0, None)})
    clim = da.mean(dim=tdim).values
    return clim, lat, lon


def compute_amip_diagnostics(run_dir: str | Path,
                             spinup_frac: float = 0.0) -> dict:
    """Load a run's CMOR Amon climatology and reduce to diagnostics.

    Returns a dict with per-field ``maps`` (2-D climatology × unit-scale),
    ``zonal`` (zonal means), ``lat``/``lon``, area-weighted ``global_means``
    (var -> (value, earth_ref, unit)), and the TOA ``budget``
    (rsdt/rsut/rlut/R_TOA/albedo) when the SW/LW TOA fields are present.
    ``spinup_frac`` (0..1) drops that leading fraction of each variable's
    monthly record as spin-up before the time-mean (see ``_load_clim``); ``0.0``
    (default) is the full-record mean and reproduces the previous behaviour.
    Pure + deterministic — the unit-tested core (no matplotlib)."""
    cmor = os.path.join(str(run_dir), "cmor", "Amon")
    tas0, lat, lon = _load_clim(cmor, "tas", spinup_frac)
    if tas0 is None:
        raise FileNotFoundError(f"no tas_Amon_*.nc under {cmor}")
    nlat, nlon = tas0.shape
    w = _sinlat_area_weights(nlat, nlon)

    maps, zonal, gmeans = {}, {}, {}
    for var, sc, unit, ref, _cmap in FIELD_TABLE:
        clim, _la, _lo = _load_clim(cmor, var, spinup_frac)
        if clim is None:
            continue
        field = clim * sc
        maps[var] = field
        zonal[var] = field.mean(axis=1)
        gmeans[var] = (float(np.sum(field * w)), ref, unit)

    budget = None
    rsdt, _, _ = _load_clim(cmor, "rsdt", spinup_frac)
    rsut, _, _ = _load_clim(cmor, "rsut", spinup_frac)
    rlut, _, _ = _load_clim(cmor, "rlut", spinup_frac)
    if rsdt is not None and rsut is not None and rlut is not None:
        RSDT = float(np.sum(rsdt * w))
        RSUT = float(np.sum(rsut * w))
        RLUT = float(np.sum(rlut * w))
        budget = {
            "rsdt": RSDT, "rsut": RSUT, "rlut": RLUT,
            "R_TOA": RSDT - RSUT - RLUT,
            "albedo": RSUT / max(RSDT, 1e-6),
        }
    return {
        "grid": (nlat, nlon), "lat": lat, "lon": lon,
        "maps": maps, "zonal": zonal, "global_means": gmeans, "budget": budget,
    }


def amip_structural_checks(diag: dict) -> dict:
    """First-order large-scale STRUCTURE checks on a diagnostics dict — catch a
    globally-plausible but structurally-dead run (no equator-pole gradient / no
    ITCZ) that the global-mean gate alone would pass.

    Returns ``{check_name: {value, threshold, comparison, within}}``.  Runs the
    ``tas`` equator-pole gradient whenever a ``tas`` map is present and the
    ``pr`` ITCZ-enhancement whenever a ``pr`` map is present; a check whose band
    is empty (degenerate grid) yields ``value=NaN`` and ``within=False``.  Pure
    + deterministic (no matplotlib, no I/O)."""
    maps = diag.get("maps") or {}
    lat = diag.get("lat")
    grid = diag.get("grid")
    checks: dict = {}
    if lat is None or grid is None:
        return checks
    lat = np.asarray(lat, dtype=np.float64)
    nlat, nlon = grid
    w = _sinlat_area_weights(nlat, nlon)

    if "tas" in maps:
        trop = _band_weighted_mean(maps["tas"], lat, w, 0.0, _TROPICS_LAT_DEG)
        pole = _band_weighted_mean(maps["tas"], lat, w, _POLE_LAT_DEG, 91.0)
        grad = trop - pole
        checks["tas_eqpole_gradient_K"] = {
            "value": float(grad), "threshold": float(_TAS_TROPICS_MINUS_POLE_MIN_K),
            "comparison": ">=", "within": bool(grad >= _TAS_TROPICS_MINUS_POLE_MIN_K)}

    if "pr" in maps:
        trop_pr = _band_weighted_mean(maps["pr"], lat, w, 0.0, _ITCZ_LAT_DEG)
        glob_pr = float(np.sum(maps["pr"] * w))
        ratio = trop_pr / glob_pr if glob_pr > 0.0 else float("nan")
        checks["pr_itcz_enhancement"] = {
            "value": float(ratio), "threshold": float(_PR_ITCZ_ENHANCEMENT_MIN),
            "comparison": ">=", "within": bool(ratio >= _PR_ITCZ_ENHANCEMENT_MIN)}

    return checks


def amip_realism_scorecard(diag: dict, *, tol_scale: float = 1.0,
                           abs_tol: dict | None = None,
                           albedo_tol: float | None = None,
                           r_toa_tol: float | None = None) -> dict:
    """Mechanical publication-realism gate on a ``compute_amip_diagnostics`` dict.

    Each area-weighted global mean is compared to its Earth reference with an
    absolute acceptance band (``_REALISM_ABS_TOL`` scaled by ``tol_scale``), the
    TOA budget's planetary albedo and net imbalance to their own bands, and the
    large-scale STRUCTURE (equator-pole gradient, ITCZ) to physical floors via
    ``amip_structural_checks`` — so a globally-plausible but structurally-dead
    run cannot pass on scalars alone.  Returns::

        {"fields": {var: {value, ref, unit, abs_tol, abs_err, within}},
         "budget": {"albedo": {...}, "r_toa": {...}} | None,
         "structure": {check: {value, threshold, comparison, within}},
         "missing_required": [var, ...],
         "n_pass": int, "n_checks": int, "passed": bool}

    ``passed`` is True iff every required field AND the TOA budget are present
    AND every check (fields + albedo + net-imbalance + structure) lies within
    its band/floor.  A
    missing/blown-up field scores ``within=False`` (``NaN`` errors compare
    False), and an absent/malformed budget is reported in ``missing_required``
    as ``"budget"`` (it needs ``rsdt``/``rsut``/``rlut``) — never a silent pass.
    ``abs_tol`` overrides are MERGED onto the defaults so a partial override can
    never drop a required field's band.  Pure + deterministic — the unit-tested
    realism gate (no matplotlib, no I/O)."""
    if not (tol_scale > 0.0):
        raise ValueError(f"tol_scale must be > 0, got {tol_scale!r}")
    bands = {**_REALISM_ABS_TOL, **(abs_tol or {})}   # merge, never replace
    a_tol = (_ALBEDO_ABS_TOL if albedo_tol is None else albedo_tol) * tol_scale
    rt_tol = (_R_TOA_ABS_TOL if r_toa_tol is None else r_toa_tol) * tol_scale

    gmeans = diag.get("global_means", {})
    fields, n_pass, n_checks = {}, 0, 0
    for var, (value, ref, unit) in gmeans.items():
        if var not in bands:
            continue
        tol = bands[var] * tol_scale
        err = abs(float(value) - float(ref))
        within = bool(err <= tol)   # NaN err -> False -> a blown-up run fails
        fields[var] = {"value": float(value), "ref": float(ref), "unit": unit,
                       "abs_tol": float(tol), "abs_err": float(err), "within": within}
        n_checks += 1
        n_pass += int(within)

    budget = None
    b = diag.get("budget")
    budget_present = isinstance(b, dict) and "albedo" in b and "R_TOA" in b
    if budget_present:
        alb_err = abs(float(b["albedo"]) - _ALBEDO_REF)
        alb_ok = bool(alb_err <= a_tol)
        rt_err = abs(float(b["R_TOA"]))
        rt_ok = bool(rt_err <= rt_tol)
        budget = {
            "albedo": {"value": float(b["albedo"]), "ref": float(_ALBEDO_REF),
                       "abs_tol": float(a_tol), "abs_err": float(alb_err),
                       "within": alb_ok},
            "r_toa": {"value": float(b["R_TOA"]), "ref": 0.0,
                      "abs_tol": float(rt_tol), "abs_err": float(rt_err),
                      "within": rt_ok},
        }
        n_checks += 2
        n_pass += int(alb_ok) + int(rt_ok)

    # Structural checks (equator-pole gradient, ITCZ) — only when the diagnostics
    # carry the maps/lat/grid needed to compute them; each counts as a check.
    structure = amip_structural_checks(diag)
    for chk in structure.values():
        n_checks += 1
        n_pass += int(chk["within"])

    missing_required = [v for v in _REQUIRED_FIELDS if v not in gmeans]
    if not budget_present:                # TOA budget is a required check
        missing_required = missing_required + ["budget"]
    passed = bool(not missing_required and n_checks > 0 and n_pass == n_checks)
    return {"fields": fields, "budget": budget, "structure": structure,
            "missing_required": missing_required,
            "n_pass": n_pass, "n_checks": n_checks, "passed": passed}


def format_scorecard_line(scorecard: dict) -> str:
    """One-line human summary: ``PASS/FAIL realism n/m`` + first offenders."""
    verdict = "PASS" if scorecard["passed"] else "FAIL"
    fails = [v for v, d in scorecard["fields"].items() if not d["within"]]
    if scorecard["budget"] is not None:
        fails += [k for k, d in scorecard["budget"].items() if not d["within"]]
    struct_fails = [k for k, d in scorecard.get("structure", {}).items()
                    if not d["within"]]
    parts = [f"{verdict} realism {scorecard['n_pass']}/{scorecard['n_checks']}"]
    if scorecard["missing_required"]:
        parts.append("missing_required=" + ",".join(scorecard["missing_required"]))
    if fails:
        parts.append("out_of_band=" + ",".join(fails))
    if struct_fails:
        parts.append("structure=" + ",".join(struct_fails))
    return "  ".join(parts)


def _json_safe(obj):
    """Recursively replace non-finite floats (``NaN``/``Inf`` from a blown-up
    run's global means) with ``None`` so the scorecard is always VALID JSON.
    Such fields already score ``within=False``; this only fixes serialization."""
    import math
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def write_scorecard(scorecard: dict, path: str | Path) -> Path:
    """Write a scorecard dict to ``path`` as valid JSON (sorted, indented).
    Non-finite values are normalised to ``null`` and ``allow_nan=False`` guards
    against any stray ``NaN``/``Inf`` slipping through as invalid JSON."""
    import json
    path = Path(path)
    with open(path, "w") as fh:
        json.dump(_json_safe(scorecard), fh, indent=2, sort_keys=True, allow_nan=False)
    return path


def plot_amip_cmor_diagnostics(run_dir: str | Path, label: str, out_path: str | Path) -> Path:
    """Render the systematic-diagnostics figure to ``out_path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diag = compute_amip_diagnostics(run_dir)
    lat, lon = diag["lat"], diag["lon"]
    nlat, nlon = diag["grid"]

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(4, 5, hspace=0.42, wspace=0.32)
    fig.suptitle(f"AMIP systematic diagnostics — {label}  (grid {nlat}x{nlon}, climatology)",
                 fontsize=15, y=0.98)

    for i, (var, _sc, unit, ref, cmap) in enumerate(FIELD_TABLE):
        r, c = divmod(i, 4)
        ax = fig.add_subplot(gs[r, c])
        if var not in diag["maps"]:
            ax.text(0.5, 0.5, f"{var}\n(missing)", ha="center", va="center")
            ax.axis("off")
            continue
        gm = diag["global_means"][var][0]
        im = ax.pcolormesh(lon, lat, diag["maps"][var], cmap=cmap, shading="auto")
        ax.set_title(f"{var}  gmean={gm:.2f} {unit}  (Earth~{ref})", fontsize=10)
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)

    axz = fig.add_subplot(gs[0:2, 4])
    for var, _sc, unit, _ref, _cmap in FIELD_TABLE:
        if var in diag["zonal"]:
            zm = diag["zonal"][var]
            rng = np.ptp(zm) or 1.0
            axz.plot((zm - zm.min()) / rng, lat, label=f"{var} ({unit})", lw=1.6)
    axz.set_title("zonal means (each min-max normalised)", fontsize=10)
    axz.set_ylabel("lat"); axz.set_xlabel("normalised value")
    axz.legend(fontsize=7, loc="lower right"); axz.grid(alpha=0.3)

    axb = fig.add_subplot(gs[2, 4])
    b = diag["budget"]
    if b is not None:
        keys = ["rsdt", "rsut", "rlut", "R_TOA"]
        axb.bar(range(4), [b[k] for k in keys],
                color=["gold", "tab:blue", "tab:red", "tab:green"])
        axb.set_xticks(range(4)); axb.set_xticklabels(keys, fontsize=8)
        axb.set_title(f"TOA budget  albedo={b['albedo']:.3f} (Earth {_ALBEDO_REF})\n"
                      f"R_TOA={b['R_TOA']:.1f} W/m2", fontsize=9)
        for j, k in enumerate(keys):
            axb.text(j, b[k], f"{b[k]:.0f}", ha="center", va="bottom", fontsize=8)
        axb.axhline(0, color="k", lw=0.5)

    axt = fig.add_subplot(gs[3, 4]); axt.axis("off")
    rows = [f"{v:5s} {gm:8.2f} {u:6s} (Earth~{rf})"
            for v, (gm, rf, u) in diag["global_means"].items()]
    axt.text(0.0, 1.0, "AREA-WEIGHTED GLOBAL MEANS\n" + "\n".join(rows),
             va="top", ha="left", family="monospace", fontsize=9)

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="run directory containing cmor/Amon/*.nc")
    p.add_argument("--label", default=None, help="figure title label")
    p.add_argument("--out", default=None, help="output PNG path")
    p.add_argument("--scorecard", action="store_true",
                   help="also compute + write a publication-realism pass/fail scorecard JSON")
    p.add_argument("--no-plot", action="store_true",
                   help="skip the figure; only compute the scorecard (implies --scorecard)")
    p.add_argument("--tol-scale", type=float, default=1.0,
                   help="scale every acceptance band (>1 widen, <1 tighten); default 1.0")
    p.add_argument("--gate", action="store_true",
                   help="exit nonzero if the realism scorecard does not pass")
    args = p.parse_args(argv)
    if args.tol_scale <= 0.0:
        p.error(f"--tol-scale must be > 0, got {args.tol_scale}")
    label = args.label or os.path.basename(str(args.run_dir).rstrip("/"))

    if not args.no_plot:
        out = args.out or os.path.join(str(args.run_dir), f"amip_diagnostics_{label}.png")
        saved = plot_amip_cmor_diagnostics(args.run_dir, label, out)
        print(f"SAVED {saved}")

    if args.scorecard or args.no_plot or args.gate:
        diag = compute_amip_diagnostics(args.run_dir)
        sc = amip_realism_scorecard(diag, tol_scale=args.tol_scale)
        # Sanitize the label for the filename — a label like "C48/L40" would
        # otherwise be read as a path separator (FileNotFoundError).
        _safe_label = re.sub(r"[^0-9A-Za-z._+-]", "_", str(label))
        sc_path = os.path.join(str(args.run_dir),
                               f"amip_scorecard_{_safe_label}.json")
        write_scorecard(sc, sc_path)
        print(f"SCORECARD {sc_path}")
        print(format_scorecard_line(sc))
        if args.gate and not sc["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
