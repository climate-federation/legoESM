#!/usr/bin/env python
"""Case-6 (Rossby-Haurwitz wave 4) duo-target gate — score a
``run_duo_stepper_case6.py`` npz against the authoritative FV3 duo
reference (Mouallem 2023, Zenodo 8327578)
``C48.sw.case6.alpha0.duo.hord8/rundir/atmos_daily.nc``.

WHAT THE REFERENCE PROVIDES (all instantaneous — every diag_table entry
is ``.false.``):
* ``ps_ic``/``ua_ic``/``va_ic`` — the INITIAL CONDITION on the 181x360
  T-cell canvas (lon 0.5..359.5).  ``ps`` in this SW file is the depth
  h in metres: measured ``ps == delp/9.80665`` to 1.7e-7 rel (float32
  storage), so gh_ref = ps_ic * FV3_GRAV.
* daily ``delp``/``ucomp``/``vcomp`` at t = 1..100 d; ``delp`` IS gh
  (m^2/s^2) on the SW convention.

SCORES (per field gh/u/v, both printed):
* rel_l2      = ||run - ref||_2 / ||ref||_2, UNWEIGHTED canvas (matches
                the w2 gate convention — both sides live on the same
                storage canvas);
* rel_l2_cosw = the same with cos(lat) area weights (the canvas is a
                lat-lon grid, so the unweighted number over-counts the
                poles; both are stated so neither hides the other);
* max_abs     = max|run - ref| in field units.

For the WIND scores the normalisation is the vector reference norm
sqrt(||u_ref||^2 + ||v_ref||^2) per weighting, so a v-field that passes
through zero cannot inflate its own relative error.

PROTOCOL FLOOR, stated not hidden: the run frames are nearest-cell
samples of the native cube + a c2l_ord2 wind lens; the reference is
fregrid + c2l_ord4.  Field-to-field errors are therefore floored at the
remap-protocol difference even for a perfect solver — this gate
quantifies distance, it attributes nothing (w2-gate honesty note).

``--analytic`` scores the ANALYTIC RH4 fields (shared
williamson_sw_analytic module, GFS constants), evaluated directly at
the reference canvas points, against ``ps_ic``/``ua_ic``/``va_ic``.
That check has NO dependence on the port's stepper, lens or sampling —
it isolates "is the reference initialised with the same formulas?", and
its error is bounded by the reference's own C48 discretisation +
fregrid remap.

ENFORCEMENT: report-only by default (exit 0 with the table).
``--enforce`` gates rel_l2_cosw per field against ``--max-gh``/
``--max-wind`` (exit 1 beyond).  The shipped defaults are PROVISIONAL
PLACEHOLDERS — no measurement has calibrated them yet; until the
calibration note below carries a job id and measured numbers, run
report-only or pass explicit bounds.  (Calibration protocol: ~2x the
worst scored day of the first green C48 deck-config run, so a
regression that doubles the error trips while remap-protocol noise
does not.)
"""
from __future__ import annotations

import argparse
import os

import numpy as np

ZENODO_BASE = ("/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo/"
               "extracted/Code and simulations files")
REF_CASE = "C48.sw.case6.alpha0.duo.hord8"
# PROVISIONAL until the first green run calibrates them (module
# docstring); order-of-magnitude guesses from the remap-protocol floor.
DEFAULT_MAX_GH_REL = 6.0e-3
DEFAULT_MAX_WIND_REL = 6.0e-2


class ContractError(ValueError):
    """The run npz violates the case-6 input contract."""


def _canvas() -> tuple[np.ndarray, np.ndarray]:
    return (np.linspace(-90.0, 90.0, 181),
            0.5 + np.arange(360, dtype=float))


def load_run(npz_path: str) -> dict:
    """Load + contract-check the runner npz."""
    z = np.load(npz_path, allow_pickle=True)
    for key in ("gh", "u", "v", "times_days", "lat", "lon"):
        if key not in z.files:
            raise ContractError(f"npz missing '{key}'")
    t = np.asarray(z["times_days"], dtype=np.float64)
    if t.ndim != 1 or t.size == 0 or not np.isfinite(t).all():
        raise ContractError("times_days must be 1-D, non-empty, finite")
    if abs(float(t[0])) > 1e-9:
        raise ContractError("times_days[0] must be 0.0 (the IC frame)")
    if t.size > 1 and np.any(np.diff(t) <= 0):
        raise ContractError("times_days must be strictly increasing")
    lat_c, lon_c = _canvas()
    lat = np.asarray(z["lat"], dtype=np.float64)
    lon = np.asarray(z["lon"], dtype=np.float64)
    if lat.shape != (181,) or not np.allclose(lat, lat_c, atol=1e-6):
        raise ContractError("lat must be linspace(-90, 90, 181)")
    if lon.shape != (360,) or not np.allclose(lon, lon_c, atol=1e-6):
        raise ContractError(
            "lon must be the reference T-cell centres 0.5..359.5 — an "
            "npz on the historical 0..359 canvas cannot be scored "
            "against atmos_daily.nc")
    fields = {}
    for key in ("gh", "u", "v"):
        f = np.asarray(z[key], dtype=np.float64)
        if f.shape != (t.size, 181, 360):
            raise ContractError(
                f"{key} must be (nt, 181, 360) aligned with times_days; "
                f"got {f.shape} vs nt={t.size}")
        if not np.isfinite(f).all():
            raise ContractError(f"{key} contains non-finite values")
        fields[key] = f
    return {"times_days": t, **fields}


def load_reference(day: float, case: str = REF_CASE) -> dict:
    """Reference (gh, u, v) at ``day`` (0.0 = the *_ic fields), selected
    BY TIME VALUE, never by frame index."""
    import netCDF4

    from legoesm.grids.fv3_native_gridstruct import FV3_GRAV

    path = os.path.join(ZENODO_BASE, case, "rundir", "atmos_daily.nc")
    d = netCDF4.Dataset(path)
    try:
        if day == 0.0:
            gh = np.asarray(d.variables["ps_ic"][:],
                            dtype=np.float64) * FV3_GRAV
            u = np.asarray(d.variables["ua_ic"][0], dtype=np.float64)
            v = np.asarray(d.variables["va_ic"][0], dtype=np.float64)
        else:
            t = np.asarray(d.variables["time"][:], dtype=np.float64)
            idx = int(np.argmin(np.abs(t - day)))
            if abs(float(t[idx]) - day) > 1e-6:
                raise ContractError(f"reference has no time == {day} d")
            gh = np.asarray(d.variables["delp"][idx, 0], dtype=np.float64)
            u = np.asarray(d.variables["ucomp"][idx, 0], dtype=np.float64)
            v = np.asarray(d.variables["vcomp"][idx, 0], dtype=np.float64)
    finally:
        d.close()
    return {"gh": gh, "u": u, "v": v}


def score_day(run: dict, ref: dict, lat_deg: np.ndarray) -> dict:
    """Pure per-day scoring (unit-testable without the Zenodo file).

    run/ref: {"gh","u","v"} 181x360.  Returns per-field rel_l2 (canvas),
    rel_l2_cosw (cos-lat weighted) and max_abs; wind fields are
    normalised by the VECTOR reference norm.
    """
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=np.float64)))[:, None]
    w = np.broadcast_to(w, run["gh"].shape)

    def _norms(diff, base, weights):
        num = float(np.sqrt(np.sum(weights * diff ** 2)))
        den = float(np.sqrt(np.sum(weights * base ** 2)))
        if den == 0.0:
            raise ValueError("reference norm is zero — cannot form a "
                             "relative error")
        return num / den

    ones = np.ones_like(run["gh"])
    out = {}
    dgh = run["gh"] - ref["gh"]
    out["gh"] = {"rel_l2": _norms(dgh, ref["gh"], ones),
                 "rel_l2_cosw": _norms(dgh, ref["gh"], w),
                 "max_abs": float(np.abs(dgh).max())}
    du = run["u"] - ref["u"]
    dv = run["v"] - ref["v"]
    for key, diff in (("u", du), ("v", dv)):
        out[key] = {
            "rel_l2": float(np.sqrt(np.sum(diff ** 2))
                            / np.sqrt(np.sum(ref["u"] ** 2)
                                      + np.sum(ref["v"] ** 2))),
            "rel_l2_cosw": float(
                np.sqrt(np.sum(w * diff ** 2))
                / np.sqrt(np.sum(w * ref["u"] ** 2)
                          + np.sum(w * ref["v"] ** 2))),
            "max_abs": float(np.abs(diff).max()),
        }
    return out


def analytic_ic_fields() -> dict:
    """Analytic RH4 (gh, u, v) at the reference canvas points — GFS
    constants, shared module; no stepper, no lens, no sampling."""
    from legoesm.core.williamson_sw_analytic import (
        RH4_MEAN_DEPTH_M,
        rossby_haurwitz_4_geopotential,
        rossby_haurwitz_4_winds,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_GRAV,
        FV3_OMEGA,
        FV3_RADIUS_M,
    )

    lat_deg, lon_deg = _canvas()
    lon = np.deg2rad(lon_deg)[None, :]
    lat = np.deg2rad(lat_deg)[:, None]
    gh = rossby_haurwitz_4_geopotential(
        lon, lat, radius=FV3_RADIUS_M, omega=FV3_OMEGA,
        gh0=RH4_MEAN_DEPTH_M * FV3_GRAV)
    u, v = rossby_haurwitz_4_winds(lon, lat, radius=FV3_RADIUS_M)
    gh, u, v = (np.broadcast_to(f, (181, 360)).copy() for f in (gh, u, v))
    return {"gh": gh, "u": u, "v": v}


def _print_table(label: str, s: dict) -> None:
    print(f"--- {label} ---")
    print(f"{'field':>5} {'rel_l2':>12} {'rel_l2_cosw':>12} "
          f"{'max_abs':>12}")
    for key in ("gh", "u", "v"):
        r = s[key]
        print(f"{key:>5} {r['rel_l2']:>12.4e} {r['rel_l2_cosw']:>12.4e} "
              f"{r['max_abs']:>12.4e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="run_duo_stepper_case6.py output")
    ap.add_argument("--days", default=None,
                    help="comma list of run days to score (default: every "
                         "whole day in the npz, IC always included)")
    ap.add_argument("--analytic", action="store_true",
                    help="also score the analytic RH4 fields at the "
                         "canvas points against ps_ic/ua_ic/va_ic "
                         "(formula-level check, stepper-independent)")
    ap.add_argument("--enforce", action="store_true",
                    help="exit 1 if any scored day exceeds --max-gh/"
                         "--max-wind on rel_l2_cosw")
    ap.add_argument("--max-gh", type=float, default=DEFAULT_MAX_GH_REL,
                    help="enforced gh rel_l2_cosw bound (calibrated ~2x "
                         "the first green run's worst day)")
    ap.add_argument("--max-wind", type=float, default=DEFAULT_MAX_WIND_REL,
                    help="enforced u/v rel_l2_cosw bound (same "
                         "calibration)")
    args = ap.parse_args()

    run = load_run(args.npz)
    t = run["times_days"]
    if args.days is None:
        days = [float(d) for d in t if abs(d - round(d)) < 1e-9]
    else:
        days = [float(x) for x in args.days.split(",")]
        for d in days:
            if not np.any(np.abs(t - d) < 1e-9):
                raise ContractError(f"run npz has no day {d} frame")
    lat_deg, _ = _canvas()

    print(f"reference: {REF_CASE}/rundir/atmos_daily.nc "
          "(instantaneous daily; ps==delp/FV3_GRAV verified 1.7e-7)")
    print("protocol: run = nearest-cell + c2l_ord2 on the reference "
          "T-cell canvas; reference = fregrid + c2l_ord4 — scores are "
          "floored by the remap-protocol difference (envelope level)")

    ok = True
    if args.analytic:
        s = score_day(analytic_ic_fields(), load_reference(0.0), lat_deg)
        _print_table("ANALYTIC formulas vs reference IC "
                     "(stepper-independent)", s)

    for day in days:
        k = int(np.argmin(np.abs(t - day)))
        frame = {key: run[key][k] for key in ("gh", "u", "v")}
        s = score_day(frame, load_reference(day), lat_deg)
        label = ("IC (day 0) run vs reference *_ic" if day == 0.0
                 else f"day {day:g} run vs reference")
        _print_table(label, s)
        if args.enforce:
            if s["gh"]["rel_l2_cosw"] > args.max_gh:
                ok = False
            if (s["u"]["rel_l2_cosw"] > args.max_wind
                    or s["v"]["rel_l2_cosw"] > args.max_wind):
                ok = False

    if args.enforce:
        print("CASE6_DUO_TARGET_GATE:",
              "PASS" if ok else
              f"FAIL (rel_l2_cosw beyond gh<{args.max_gh:g} / "
              f"wind<{args.max_wind:g})")
        return 0 if ok else 1
    print("CASE6_DUO_TARGET_GATE: REPORT-ONLY (pass --enforce to gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
