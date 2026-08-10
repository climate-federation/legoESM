#!/usr/bin/env python3
"""Cross-grid baroclinic-instability growth discriminator (#1028 / #1081).

WHAT THIS ANSWERS
-----------------
#1028 measured that the cd-grid cubed sphere equilibrates the Held-Suarez jet
at ~13.8 m/s after the sponge and ``A_h`` sinks were removed and saturated,
against 27 (icosahedral) / 66 (lat-lon) m/s.  The remaining deficit is either

  (a) a GROWTH/CONVERSION deficit — the cube's baroclinic eddies never reach
      the amplitude that supplies the jet, or
  (b) an EQUILIBRATION deficit — eddies grow normally but the equilibrated
      balance is drained by something a net-KE budget could not see.

The discriminator runs the SAME analytic Jablonowski-Williamson perturbed
baroclinic wave on each grid and reports, per grid:

  * the exponential growth rate of eddy kinetic energy in TWO sub-windows,
    early and late.  A single slope across a window that straddles nonlinear
    saturation is not a growth rate, and R^2 cannot see the bend — so the two
    sub-window slopes are reported side by side and their disagreement is the
    curvature measure.  Both adversarial reviewers (codex, GLM-5.2) flagged
    the single-window fit as the probe's most dangerous defect.
  * the eddy-KE RATIO TO A REFERENCE GRID AS A FUNCTION OF TIME.  This is the
    statistic that actually separates (a) from (b) and it costs nothing extra:
    a ratio that is flat in time means the two grids' eddies grow at the same
    rate from different amplitudes (growth operator agrees; look upstream at
    the IC projection), a ratio that is flat and BELOW one from the start of
    growth means (a), and a ratio that STARTS near one and falls means (b).
  * the same rates recomputed from an independent field — the eddy variance
    of temperature at a fixed mid-tropospheric level — so the verdict does not
    rest on the lowest model level's winds alone.

This script prints numbers, never a verdict.  The interpretation belongs in
the issue thread after the controls below have been read.

WHAT IT MEASURES, EXACTLY
-------------------------
From each run's ``snapshots_latlon.npz`` (written by
``scripts/matrix/run_atmosphere_test_matrix.py``), which holds the LOWEST
model level's geographic ``u``/``v`` and the full ``T_3d`` column, regridded to
the SAME 181x360 canvas by the SAME code path for every grid type:

    EKE(t)  = < 1/2 [ (u - u_zonalmean)^2 + (v - v_zonalmean)^2 ] >
    TVAR(t) = < (T_k - T_k_zonalmean)^2 >        at a fixed level index k

with ``< >`` a cos(lat)-weighted mean over a latitude band and the zonal mean
taken per latitude row at each time.  EKE is m^2/s^2 at ONE level with no mass
weighting, so only the RATE and the cross-grid RATIO are meaningful, never the
absolute level.  ``sigma = d ln(.)/dt`` is per DAY; the amplitude e-folding
rate is sigma/2 for EKE and TVAR alike (both are quadratic in the eddy).

KNOWN LIMITS, STATED RATHER THAN HIDDEN
---------------------------------------
* The shared canvas is finer than every native grid, so each arm arrives
  through a different effective interpolation filter.  That biases the LEVEL of
  EKE per grid; it cancels from a growth RATE only to the extent the filter is
  constant in time.  The printed ``eke_t0`` column is the direct check: a grid
  whose regridding injects a time-constant eddy floor (cube panel seams are the
  candidate) shows an anomalously large t=0 value, and any such floor biases
  its fitted slope DOWNWARD.  Read that column before reading the rates.
* Single deterministic run per arm — no ensemble, so no error bar on the ratio.
  The early/late sub-window spread is the only spread reported.
* The thresholds an experiment chooses are NOT baked in here; there is no
  calibration mapping a J-W growth rate to an equilibrated Held-Suarez jet.

Usage
-----
    python scripts/validate/bci_growth_rate.py --self-test
    python scripts/validate/bci_growth_rate.py \
        cube=results/bci/hydrostatic/baroclinic/cubed_sphere/C36/sigma \
        ico=results/bci/hydrostatic/baroclinic/icosahedral/ico5/sigma \
        --reference ico
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# The reduction needs enough latitude rows for the cos(lat) integral to mean
# anything; 10 rows is 9 degrees on the 181-row canvas (GLM-5.2 review S9d).
_MIN_BAND_DEG = 9.0
# A window whose smallest sample is below this fraction of its largest is a
# floor-dominated series, not an exponential: log() of the small end then
# drives the fit (GLM-5.2 review S9a).
_MIN_DYNAMIC_RANGE = 1.0e-3
# ... and a series that barely moves is a constant, whose least-squares slope
# is a rounding artifact rather than a growth rate.
_MIN_VARIATION = 1.0e-6
# Quotability gate, pre-registered in the launcher and enforced HERE rather
# than in prose: an ADDITIVE floor (EKE = floor + exp(sigma t)) satisfies every
# guard above and still returns a slope far below the true rate, with a
# plausible R^2 ~ 0.9 (codex round-2 review).  A rate is only quotable when the
# window is a clean exponential that actually grew.
_QUOTABLE_R2 = 0.95
_QUOTABLE_RATE_PER_DAY = 0.1
_QUOTABLE_GROWTH_FACTOR = 10.0


def _eddy_variance(field: np.ndarray, lat_deg: np.ndarray,
                   band: tuple[float, float]) -> np.ndarray:
    """cos(lat)-weighted variance about the ZONAL MEAN, per time.

    ``field`` is (n_times, n_lat, n_lon).  Shared by the wind and temperature
    statistics so there is exactly one zonal-mean / area-weight implementation.
    """
    f = np.asarray(field, dtype=np.float64)      # never reduce in float32
    if f.ndim != 3:
        raise ValueError(f"expected (n_times, n_lat, n_lon), got {f.shape}")
    if f.shape[1] != lat_deg.size:
        raise ValueError(f"lat axis {f.shape[1]} != len(lat) {lat_deg.size}")
    if not np.all(np.isfinite(f)):
        raise ValueError("non-finite field in the requested window — refusing "
                         "to reduce over it (a nan-mean here would hide the "
                         "very blow-up this probe exists to detect)")
    lo, hi = band
    sel = (lat_deg >= lo) & (lat_deg <= hi)
    # Stated in DEGREES, not rows, so the guard means the same thing on any
    # canvas resolution (GLM-5.2 round-2 #5).
    span = float(np.ptp(lat_deg[sel])) if sel.sum() else 0.0
    if sel.sum() < 2 or span < _MIN_BAND_DEG:
        raise ValueError(
            f"latitude band {band} selects {sel.sum()} rows spanning "
            f"{span:.2f} deg (need >= {_MIN_BAND_DEG} deg)")
    w = np.maximum(np.cos(np.deg2rad(np.asarray(lat_deg, dtype=np.float64))), 0.0)[sel]
    if w.sum() <= 0:
        raise ValueError(f"zero total weight for band {band}")
    anom = f - f.mean(axis=2, keepdims=True)
    return ((anom ** 2)[:, sel, :].mean(axis=2) * w).sum(axis=1) / w.sum()


def eddy_ke(u: np.ndarray, v: np.ndarray, lat_deg: np.ndarray,
            band: tuple[float, float]) -> np.ndarray:
    """cos(lat)-weighted eddy kinetic energy per unit mass, per time."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"u{u.shape} and v{v.shape} disagree")
    return 0.5 * (_eddy_variance(u, lat_deg, band)
                  + _eddy_variance(v, lat_deg, band))


def growth_rate(times_days: np.ndarray, series: np.ndarray,
                day_lo: float, day_hi: float) -> tuple[float, float, int]:
    """Least-squares d(ln series)/dt over [day_lo, day_hi]; (rate, R2, n).

    Fails loudly rather than returning a rate that a log of a near-zero,
    negative, or floor-dominated series would make meaningless.
    """
    t = np.asarray(times_days, dtype=np.float64)
    e = np.asarray(series, dtype=np.float64)
    if t.shape != e.shape:
        raise ValueError(f"times{t.shape} and series{e.shape} disagree")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times_days is not strictly increasing")

    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(e)):
        raise ValueError("non-finite times or series after reduction")
    # A run that stopped early must NOT be fitted over the part of the window
    # it reached and reported under the window that was asked for (codex
    # review, HIGH #5): both endpoints must be inside the sampled span.
    # Tolerance is 2% of the window — 1.9 h on the default 4-day windows.  It
    # absorbs the rounding when dt does not divide the run length exactly and
    # rejects a run that stopped days early; it does NOT certify that the final
    # sample sits exactly on day_hi.
    span_tol = 0.02 * (day_hi - day_lo)
    if t.min() > day_lo + span_tol or t.max() < day_hi - span_tol:
        raise ValueError(
            f"requested window [{day_lo}, {day_hi}] d is not covered by the "
            f"sampled span {t.min():.3f}..{t.max():.3f} d — the run is "
            "truncated relative to the window being claimed")

    sel = (t >= day_lo) & (t <= day_hi)
    n = int(sel.sum())
    if n < 4:
        raise ValueError(
            f"only {n} snapshots inside [{day_lo}, {day_hi}] d "
            f"(have {t.min():.3f}..{t.max():.3f} d, {t.size} samples) — "
            "a 2-3 point exponential fit is not a growth rate")
    y_lin = e[sel]
    if np.any(y_lin <= 0):
        raise ValueError("non-positive value inside the fit window — a "
                         "zonally symmetric state has no growth RATE")
    if y_lin.min() < _MIN_DYNAMIC_RANGE * y_lin.max():
        raise ValueError(
            f"dynamic range {y_lin.min():.3e}..{y_lin.max():.3e} inside the "
            "window spans more than 3 decades — the small end is a numerical "
            "floor and would dominate the log fit")
    # A flat-but-positive series fits slope 0 with R^2 = 0 and would print as
    # "no growth" rather than as the degenerate input it is (codex review #7).
    if y_lin.max() - y_lin.min() < _MIN_VARIATION * y_lin.max():
        raise ValueError(
            f"series varies by {(y_lin.max() - y_lin.min()) / y_lin.max():.2e} "
            "of its maximum across the window — that is a constant, not an "
            "exponential, and its slope is not a growth rate")

    y = np.log(y_lin)
    x = t[sel]
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return float(slope), r2, n


def load_arm(run_dir: Path) -> dict:
    """Read one matrix output dir; fails loudly on a missing/short series."""
    npz = run_dir / "snapshots_latlon.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} — run the matrix case first")
    with np.load(npz) as d:
        need = ("u", "v", "lat", "lon", "times_days")
        missing = [k for k in need if k not in d]
        if missing:
            raise KeyError(f"{npz} lacks {missing} (keys: {sorted(d.files)})")
        arm = {"u": d["u"], "v": d["v"], "lat": d["lat"], "lon": d["lon"],
               "times": d["times_days"],
               "T_3d": d["T_3d"] if "T_3d" in d else None,
               "p_s": d["p_s"] if "p_s" in d else None}
    if arm["u"].ndim != 3:
        raise ValueError(f"{npz}: u has shape {arm['u'].shape}, expected "
                         "(n_times, n_lat, n_lon)")
    if arm["u"].shape[1:] != (arm["lat"].size, arm["lon"].size):
        raise ValueError(
            f"{npz}: u canvas {arm['u'].shape[1:]} does not match its own "
            f"(lat, lon) axes ({arm['lat'].size}, {arm['lon'].size})")
    if arm["T_3d"] is not None and (
            arm["T_3d"].shape[:3] != arm["u"].shape[:3]):
        raise ValueError(
            f"{npz}: T_3d {arm['T_3d'].shape[:3]} is not on the same "
            f"(time, lat, lon) canvas as u {arm['u'].shape[:3]}")
    if not (np.all(np.isfinite(arm["lat"])) and np.all(np.diff(arm["lat"]) > 0)):
        raise ValueError(f"{npz}: lat is not finite and strictly increasing — "
                         "a permuted or corrupted axis would silently "
                         "mis-weight the reduction")
    return arm


def require_matched_arms(arms: dict[str, dict]) -> None:
    """Every arm must share the canvas AND the sample times.

    Comparing rates read off different time stamps or different canvases is the
    classic frame-index confound; this refuses it instead of averaging over it.
    """
    ref_label, ref = next(iter(arms.items()))
    for label, a in arms.items():
        if a["u"].shape[1:] != ref["u"].shape[1:]:
            raise ValueError(
                f"arm {label!r} canvas {a['u'].shape[1:]} != arm "
                f"{ref_label!r} canvas {ref['u'].shape[1:]}")
        if not (np.array_equal(a["lat"], ref["lat"])
                and np.array_equal(a["lon"], ref["lon"])):
            raise ValueError(
                f"arm {label!r} lat/lon axes differ from {ref_label!r}")
        if a["times"].shape != ref["times"].shape or not np.allclose(
                a["times"], ref["times"], rtol=0, atol=1e-9):
            raise ValueError(
                f"arm {label!r} sample times {a['times']} != arm "
                f"{ref_label!r} times {ref['times']} — matched-time "
                "comparison is impossible")


def t_level_series(arm: dict, level: int, lat: np.ndarray,
                   band: tuple[float, float]) -> np.ndarray | None:
    """Eddy temperature variance at one level index, or None if T is absent."""
    if arm["T_3d"] is None:
        return None
    t3 = arm["T_3d"]
    if t3.ndim != 4:
        raise ValueError(f"T_3d has shape {t3.shape}, expected "
                         "(n_times, n_lat, n_lon, nlev)")
    nlev = t3.shape[-1]
    if not (-nlev <= level < nlev):
        raise IndexError(f"--t-level {level} outside 0..{nlev - 1}")
    return _eddy_variance(t3[..., level], lat, band)


def native_ps_perturbation(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(times_days, max|p_s - p_s(0)|) from the matrix's own timeseries CSV.

    This series NEVER PASSES THROUGH THE REGRIDDING: the matrix computes it on
    the native mesh at every diagnostic step.  It is the canonical
    Jablonowski-Williamson growth diagnostic, and it is the one cross-check
    that bounds the shared-canvas concern both reviewers raised — if the cube's
    wave grows more slowly here too, no interpolation filter can be blamed.

    Weakness, stated: it is a MAX over the globe, not an area-weighted mean, so
    it tracks the single strongest column and carries no latitude weighting.
    It is an AMPLITUDE, so its exponential rate is half of an energy rate.
    """
    csv = run_dir / "mean_timeseries.csv"
    if not csv.exists():
        raise FileNotFoundError(f"{csv} — the matrix writes this per case")
    rows = [ln.split(",") for ln in csv.read_text().splitlines()
            if ln and not ln.startswith("#")]
    if not rows or rows[0][0] != "step":
        raise ValueError(f"{csv}: expected a 'step,...' header, got {rows[:1]}")
    head = rows[0]
    for col in ("time_days", "ps_perturbation"):
        if col not in head:
            raise KeyError(f"{csv} lacks column {col!r} (has {head})")
    it, ip = head.index("time_days"), head.index("ps_perturbation")
    t = np.array([float(r[it]) for r in rows[1:]], dtype=np.float64)
    v = np.array([float(r[ip]) for r in rows[1:]], dtype=np.float64)
    if t.size < 4:
        raise ValueError(f"{csv}: only {t.size} diagnostic rows")
    return t, v


def level_pressure_hpa(arm: dict, level: int, band: tuple[float, float],
                       vertical: str = "sigma") -> float | None:
    """Approximate pressure [hPa] of ``level`` from the model's OWN coordinate.

    Uses ``legoesm.grids.vertical.create_sigma_coordinate`` — the same factory
    the matrix calls for the sigma cases — rather than re-deriving a sigma
    ladder, and the arm's own band-mean surface pressure at t=0.  Returned so
    the reader can confirm the fixed level index really is free troposphere and
    really is the same physical level on every arm (GLM-5.2 round-2 #4).
    """
    # A FIXED INDEX is a fixed pressure only on the uniform-sigma ladder; on
    # the hybrid coordinate the same index is a different level (codex round-2
    # #2), so refuse to print a pressure there rather than print a wrong one.
    if vertical != "sigma" or arm["T_3d"] is None or arm.get("p_s") is None:
        return None
    try:
        from legoesm.grids.vertical import create_sigma_coordinate
    except Exception:                                    # pragma: no cover
        return None
    nlev = arm["T_3d"].shape[-1]
    sigma_full = np.asarray(create_sigma_coordinate(nlev).sigma_full,
                            dtype=np.float64)
    lo, hi = band
    sel = (arm["lat"] >= lo) & (arm["lat"] <= hi)
    w = np.maximum(np.cos(np.deg2rad(arm["lat"][sel])), 0.0)
    p_s = np.asarray(arm["p_s"][0], dtype=np.float64)[sel, :].mean(axis=1)
    return float(sigma_full[level] * (p_s * w).sum() / w.sum() / 100.0)


def _git_sha() -> str:
    """HEAD, marked ``-dirty`` when the tree does not match it.

    An UNCOMMITTED probe is not identified by HEAD, so a bare SHA stamped on a
    number would name code that never ran (codex review, MEDIUM #10).
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True,
            text=True, check=True).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True,
            text=True, check=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:                                    # pragma: no cover
        return "unknown"


def self_test() -> None:
    """Recover a KNOWN growth rate, and check the degenerate cases fail."""
    lat = np.linspace(-90.0, 90.0, 181)
    lon = np.linspace(-180.0, 180.0, 360, endpoint=False)
    t = np.arange(0.0, 11.0, 1.0)
    sigma_true = 0.63                                    # 1/day, in ln(KE)

    amp = np.exp(0.5 * sigma_true * t)[:, None, None]
    wave = np.sin(np.deg2rad(6.0 * lon))[None, None, :] * np.ones(
        (1, lat.size, 1))
    u, v = amp * wave, 0.5 * amp * wave
    eke = eddy_ke(u, v, lat, (20.0, 80.0))
    rate, r2, n = growth_rate(t, eke, 4.0, 10.0)
    assert abs(rate - sigma_true) < 1e-9, (rate, sigma_true)
    assert r2 > 1 - 1e-12, r2
    assert n == 7, n

    zonal = np.ones((t.size, lat.size, lon.size)) * 30.0
    assert np.allclose(eddy_ke(zonal, 0.0 * zonal, lat, (20.0, 80.0)), 0.0)
    for bad_call, why in (
            (lambda: growth_rate(t, eddy_ke(zonal, 0.0 * zonal, lat,
                                            (20.0, 80.0)), 4.0, 10.0),
             "zero EKE must not yield a growth rate"),
            (lambda: eddy_ke(np.where(np.arange(u.size).reshape(u.shape) == 0,
                                      np.nan, u), v, lat, (20.0, 80.0)),
             "non-finite input must raise"),
            (lambda: growth_rate(t[:6], eke[:6], 4.0, 10.0),
             "a 2-point window must not yield a rate"),
            (lambda: eddy_ke(u, v, lat, (89.0, 90.0)),
             "a 2-row band must not be reduced"),
    ):
        try:
            bad_call()
        except (ValueError, IndexError):
            pass
        else:                                            # pragma: no cover
            raise AssertionError(why)
    print("self-test OK: recovered sigma = "
          f"{rate:.6f} /day (true {sigma_true}), R2 = {r2:.6f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arms", nargs="*", metavar="LABEL=RUN_DIR")
    p.add_argument("--early", type=float, nargs=2, default=(2.0, 6.0),
                   metavar=("LO", "HI"), help="early fit window [days]")
    p.add_argument("--late", type=float, nargs=2, default=(6.0, 10.0),
                   metavar=("LO", "HI"), help="late fit window [days]")
    p.add_argument("--band", type=float, nargs=2, default=(20.0, 80.0),
                   metavar=("LAT_LO", "LAT_HI"),
                   help="latitude band [deg]; the J-W perturbation is at 40N")
    p.add_argument("--vertical", type=str, default="sigma",
                   choices=["sigma", "hybrid"],
                   help="vertical coordinate of the runs being scored.  A "
                        "fixed level INDEX is a fixed pressure only on the "
                        "uniform-sigma ladder, so the printed level pressure "
                        "is suppressed for 'hybrid'.")
    p.add_argument("--t-level", type=int, default=27,
                   help="level index (from the model top) for the eddy "
                        "temperature-variance cross-check.  All arms use the "
                        "same index and the same uniform-sigma column, so the "
                        "index is the same physical level on every grid.")
    p.add_argument("--reference", type=str, default=None,
                   help="arm label used as the denominator of the ratios")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args(argv)

    if args.self_test:
        self_test()
        if not args.arms:
            return 0
    if not args.arms:
        p.error("no arms given (LABEL=RUN_DIR ...)")

    arms: dict[str, dict] = {}
    for spec in args.arms:
        if "=" not in spec:
            p.error(f"expected LABEL=RUN_DIR, got {spec!r}")
        label, _, path = spec.partition("=")
        if not label or not path:
            p.error(f"empty label or path in {spec!r}")
        if label in arms:
            p.error(f"duplicate arm label {label!r} — the second would "
                    "silently overwrite the first")
        arms[label] = load_arm(Path(path))
        arms[label]["run_dir"] = path
    require_matched_arms(arms)

    ref = args.reference
    if ref is not None and ref not in arms:
        p.error(f"--reference {ref!r} is not among {sorted(arms)}")

    band = (float(args.band[0]), float(args.band[1]))
    out: dict[str, dict] = {}
    for label, a in arms.items():
        eke = eddy_ke(a["u"], a["v"], a["lat"], band)
        tvar = t_level_series(a, args.t_level, a["lat"], band)
        psp_t, psp = native_ps_perturbation(Path(a["run_dir"]))
        rec = {"run_dir": a["run_dir"],
               "times_days": [float(x) for x in a["times"]],
               "eke": [float(x) for x in eke],
               "eke_t_first": float(eke[0]),
               "t_first_days": float(a["times"][0]),
               "eke_growth_factor": (float(eke[-1] / eke[0])
                                     if eke[0] > 0 else float("inf")),
               "level_pressure_hpa": level_pressure_hpa(
                   a, args.t_level, band, args.vertical)}
        for name, series, times in (("eke", eke, a["times"]),
                                    ("tvar", tvar, a["times"]),
                                    ("psp", psp, psp_t)):
            if series is None:
                continue
            for win, (lo, hi) in (("early", args.early), ("late", args.late)):
                r, r2, n = growth_rate(times, series, lo, hi)
                # p_s' is an AMPLITUDE; doubling puts it on the same scale as
                # the quadratic EKE / T-variance rates.
                if name == "psp":
                    r *= 2.0
                w_sel = (times >= lo) & (times <= hi)
                w_vals = np.asarray(series, dtype=np.float64)[w_sel]
                # p_s' is an AMPLITUDE while EKE/TVAR are quadratic, so its
                # in-window factor is SQUARED before the gate compares it with
                # the same threshold — otherwise the doubled rate and the raw
                # factor would be on different bases and a healthy arm could
                # fail its own gate.
                factor = float(w_vals[-1] / w_vals[0]) ** (2 if name == "psp"
                                                           else 1)
                # The gate is CODE, not prose: R^2, a positive rate, and real
                # growth inside the window itself.  An additive regridding
                # floor fails it (R^2 ~ 0.9) instead of printing 0.02 /day as
                # if it were a growth rate.
                rec[f"{name}_{win}"] = {
                    "rate_per_day": r, "r2": r2, "n": n,
                    "window_days": [lo, hi], "growth_factor_in_window": factor,
                    "quotable": bool(r2 >= _QUOTABLE_R2
                                     and r > _QUOTABLE_RATE_PER_DAY
                                     and factor >= _QUOTABLE_GROWTH_FACTOR)}
        if tvar is not None:
            rec["tvar"] = [float(x) for x in tvar]
        out[label] = rec

    print(f"# bci_growth_rate  sha={_git_sha()}  band={band[0]}..{band[1]}N  "
          f"early={args.early[0]}-{args.early[1]}d  "
          f"late={args.late[0]}-{args.late[1]}d  t_level={args.t_level}")
    print("# quantity: lowest-level eddy KE per unit mass [m2/s2] and eddy T "
          "variance [K2] on the shared 181x360 canvas; rates are d ln(.)/dt")
    print("# psp = 2 x d ln(max|p_s - p_s(0)|)/dt from the NATIVE-grid "
          "timeseries CSV (no regridding); factor = eke(t_end)/eke(t_1st)")
    print(f"# quotable = the EARLY EKE window passed R2 >= {_QUOTABLE_R2}, "
          f"rate > {_QUOTABLE_RATE_PER_DAY}/day and in-window growth "
          f">= {_QUOTABLE_GROWTH_FACTOR}x")
    print(f"{'arm':<10} {'eke_early':>10} {'eke_late':>9} {'R2e':>6} "
          f"{'R2l':>6} {'tvar_early':>11} {'tvar_late':>10} "
          f"{'psp_early':>10} {'psp_late':>9} {'eke(t_1st)':>11} "
          f"{'factor':>10} {'p_lev[hPa]':>11} {'quotable':>9} {'t_1st':>6}")
    for label, r in out.items():
        def g(k, f):
            return r[k][f] if k in r else float("nan")
        plev = r["level_pressure_hpa"]
        print(f"{label:<10} {g('eke_early','rate_per_day'):>10.4f} "
              f"{g('eke_late','rate_per_day'):>9.4f} "
              f"{g('eke_early','r2'):>6.3f} {g('eke_late','r2'):>6.3f} "
              f"{g('tvar_early','rate_per_day'):>11.4f} "
              f"{g('tvar_late','rate_per_day'):>10.4f} "
              f"{g('psp_early','rate_per_day'):>10.4f} "
              f"{g('psp_late','rate_per_day'):>9.4f} {r['eke_t_first']:>11.3e} "
              f"{r['eke_growth_factor']:>10.3e} "
              f"{(float('nan') if plev is None else plev):>11.1f} "
              f"{str(g('eke_early','quotable')):>9} "
              f"{r['t_first_days']:>6.2f}")

    if ref:
        # The ratio AS A FUNCTION OF TIME is the statistic that separates a
        # growth deficit (flat, below one) from a drain (starts near one,
        # falls).  Printed raw; this script draws no conclusion from it.
        times = out[ref]["times_days"]
        print(f"\n# eddy-KE ratio to {ref!r} vs time [days], and the SAME "
              "ratio divided by its own t=0 value")
        print("# (the t=0-normalised row is the one to read for growth: any "
              "regridding offset present at t=0 divides out of it)")
        print("arm            " + "".join(f"{t:>9.2f}" for t in times))
        for label, r in out.items():
            if label == ref:
                continue
            rat = [a / b if b > 0 else float("nan")
                   for a, b in zip(r["eke"], out[ref]["eke"])]
            norm = ([x / rat[0] for x in rat] if rat and rat[0] > 0
                    else [float("nan")] * len(rat))
            out[label]["eke_ratio_to_ref"] = rat
            out[label]["eke_ratio_to_ref_norm_t0"] = norm
            print(f"{label:<10} raw " + "".join(f"{x:>9.3f}" for x in rat))
            print(f"{'':<10} /t0 " + "".join(f"{x:>9.3f}" for x in norm))
        out["_reference"] = ref

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"sha": _git_sha(), "band": list(band), "early": list(args.early),
             "late": list(args.late), "t_level": args.t_level, "arms": out},
            indent=2))
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
