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

SCORES (per field gh/u/v, all printed):
* rel_l2      = ||run - ref||_2 / ||ref||_2, UNWEIGHTED canvas (matches
                the w2 gate convention; counts the duplicated pole rows
                at full weight);
* rel_l2_cosw = the same with cos(lat) area weights (the physically
                meaningful global number on a lat-lon canvas);
* max_abs     = max|run - ref| in field units.

Enforcement checks BOTH L2 metrics: cos-weighting alone gives the two
pole rows ~6e-17 weight, so a finite corruption confined there would be
invisible to it (codex c6 r1 #3); the unweighted metric sees it, and a
plausibility band on the loaded fields (below) rejects the known
BIG_NUMBER-class sentinels outright.

For the WIND scores the normalisation is the vector reference norm
sqrt(||u_ref||^2 + ||v_ref||^2) per weighting, so a v-field that passes
through zero cannot inflate (or deflate) its own relative error.

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

ENFORCEMENT (codex c6 r1 #1/#2/#4/#5): report-only by default.
``--enforce`` additionally requires
* EXPLICIT ``--max-gh``/``--max-wind`` bounds (finite, positive) — there
  are deliberately no defaults until a calibration lands here with its
  job id: bounds nobody measured are a gate that cannot mean anything;
* the npz to carry the runner's config record MATCHING the reference
  deck (C48, dt_atmos=1200, n_split=7, d_ext=0, d4_bg=0, oracle
  conventions, no ext exclusions) — a diagnostic variant scores fine in
  report mode but can never PASS as the deck;
* contiguous day coverage 0..requested_days with requested_days >= 1
  (``--allow-ic-only`` relaxes the >= 1, for the pure IC gate) — a
  timeout-truncated incremental npz cannot pass as a complete run;
* every scored metric to be finite and <= its bound (a NaN score or
  bound FAILS — comparisons are written closed, not open).
"""
from __future__ import annotations

import argparse
import os

import numpy as np

ZENODO_BASE = ("/burg-archive/glab/users/pg2328/Code/FV3/duogrid_zenodo/"
               "extracted/Code and simulations files")
REF_CASE = "C48.sw.case6.alpha0.duo.hord8"
REF_MAX_DAY = 100.0            # atmos_daily.nc time axis is 1..100 d

# The reference deck's resolved configuration (logfile.000000.out) — an
# npz must record exactly this to be enforceable as a deck score.
DECK_RECORD = {
    "n": 48,
    "dt_atmos": 1200.0,
    "n_split": 7,
    "d_ext": 0.0,
    "d4_bg": 0.0,
    "k2e_nord": 2,        # the authoritative live default (no nml knob)
    "oracle_conventions": True,
    "ext_exclude": "",
}

# Plausibility bands — SENTINEL DETECTORS, not physics gates: the RH4
# state lives in gh ~ [7.8e4, 1.04e5] and |V| <= ~100 m/s; the known
# failure modes (BIG_NUMBER = 1e8 corner sentinel, a NaN that became a
# huge finite through arithmetic) sit orders of magnitude outside.  A
# genuinely evolving solution never approaches these bounds.
GH_PLAUSIBLE = (1.0e4, 5.0e5)          # m^2/s^2
WIND_PLAUSIBLE_MAX = 500.0             # m/s


class ContractError(ValueError):
    """The run npz violates the case-6 input contract."""


def _canvas() -> tuple[np.ndarray, np.ndarray]:
    return (np.linspace(-90.0, 90.0, 181),
            0.5 + np.arange(360, dtype=float))


def load_run(npz_path: str) -> dict:
    """Load + contract-check the runner npz (arrays, canvas, times,
    finiteness, plausibility band); carry the runner's config record
    through for the enforcement manifest check."""
    z = np.load(npz_path, allow_pickle=True)
    for key in ("gh", "u", "v", "times_days", "lat", "lon"):
        if key not in z.files:
            raise ContractError(f"npz missing '{key}'")
    t = np.asarray(z["times_days"], dtype=np.float64)
    if t.ndim != 1 or t.size == 0 or not np.isfinite(t).all():
        raise ContractError("times_days must be 1-D, non-empty, finite")
    # EXACT zero: the runner writes 0.0 bit-for-bit; a tolerance here
    # opened a window (t[0]=1e-9 passed as IC yet was dropped by the
    # whole-day scorer, so an enforced run never scored its IC —
    # codex c6 r3 #1)
    if float(t[0]) != 0.0:
        raise ContractError("times_days[0] must be exactly 0.0 (the IC "
                            "frame)")
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
    lo, hi = GH_PLAUSIBLE
    if fields["gh"].min() < lo or fields["gh"].max() > hi:
        raise ContractError(
            f"gh outside the plausibility band [{lo:g}, {hi:g}] "
            f"(got [{fields['gh'].min():g}, {fields['gh'].max():g}]) — "
            "sentinel/corruption detector, poles included")
    for key in ("u", "v"):
        m = float(np.abs(fields[key]).max())
        if m > WIND_PLAUSIBLE_MAX:
            raise ContractError(
                f"|{key}| max {m:g} exceeds the plausibility bound "
                f"{WIND_PLAUSIBLE_MAX:g} m/s — sentinel/corruption "
                "detector")
    record = {}
    for key in (*DECK_RECORD, "git_sha"):
        if key in z.files:
            record[key] = np.asarray(z[key]).item()
    if "requested_days" in z.files:
        record["requested_days"] = int(np.asarray(
            z["requested_days"]).item())
    return {"times_days": t, "record": record, **fields}


def check_deck_record(record: dict) -> list[str]:
    """Mismatches between the npz's recorded config and the reference
    deck; a MISSING key is a mismatch (an npz from before the record
    was written cannot be enforced).  Comparison is TYPE-NORMALISED
    (numpy scalars arrive via ``.item()``; ints and floats compare by
    value, strings as str, bools as bool) so a dtype change cannot
    smuggle a mismatch through ``==`` (codex c6 r2 #5).

    LIMITATION, stated: this is the runner's SELF-ATTESTATION — it
    certifies what the npz says it ran, not what a process actually
    executed.  The npz's ``git_sha`` (required present) is the audit
    hook; byte-level provenance is out of scope for this gate.
    """
    problems = []
    for key, want in DECK_RECORD.items():
        if key not in record:
            problems.append(f"{key}: not recorded in npz")
            continue
        got = record[key]
        # TYPE gate first, value second: coercion alone let
        # oracle_conventions="False" pass bool(...) and d_ext=False /
        # d4_bg="0" impersonate numeric zeros (codex c6 r3 #4)
        if isinstance(want, bool):
            same = (isinstance(got, (bool, np.bool_))
                    and bool(got) is want)
        elif isinstance(want, (int, float)):
            same = (isinstance(got, (int, float, np.integer,
                                     np.floating))
                    and not isinstance(got, (bool, np.bool_))
                    and np.isfinite(float(got))
                    and float(got) == float(want))
        else:
            same = isinstance(got, str) and got == want
        if not same:
            problems.append(f"{key}: npz has {got!r}, deck is {want!r}")
    sha = record.get("git_sha")
    # exactly 40 hex chars — the runner emits plain `rev-parse HEAD`
    # (never a -dirty suffix), so a longer string is malformed, not a
    # variant (codex c6 r4 #2)
    if not (isinstance(sha, str) and len(sha) == 40
            and all(ch in "0123456789abcdef" for ch in sha)):
        # well-formedness only — a 40-hex SHA can still be copied into
        # a forged npz; the SELF-ATTESTATION limitation above stands
        problems.append(
            f"git_sha: {sha!r} is not a 40-hex commit id (runner "
            "emits 'unknown' when git fails — such an npz cannot be "
            "enforced)")
    return problems


def check_coverage(times: np.ndarray, record: dict,
                   allow_ic_only: bool = False) -> None:
    """Enforceable runs carry contiguous whole-day frames 0..requested;
    raises ContractError otherwise (truncated/sparse incremental saves
    stay scoreable in report mode only)."""
    if "requested_days" not in record:
        raise ContractError(
            "npz does not record requested_days — cannot certify "
            "completeness (report-only npz)")
    req = record["requested_days"]
    want = np.arange(0.0, float(req) + 0.5)
    # EXACT equality: the runner writes float(day) from Python ints,
    # which is exact.  Any tolerance here re-opens the near-integer
    # window where coverage certifies a frame the whole-day scorer
    # silently drops (codex c6 r2 P0, r3 #1).
    if not np.array_equal(times, want):
        raise ContractError(
            f"times_days {times.tolist()} != contiguous 0..{req} — "
            "truncated or sparse run cannot be enforced")
    if req < 1 and not allow_ic_only:
        raise ContractError(
            "IC-only npz (requested_days=0): pass --allow-ic-only to "
            "enforce the IC score alone")


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
    out = {"gh": gh, "u": u, "v": v}
    for key, f in out.items():
        if not np.isfinite(f).all():
            raise ContractError(
                f"reference {key} at day {day} contains non-finite "
                "values — refusing to score against it")
    return out


def score_day(run: dict, ref: dict, lat_deg: np.ndarray) -> dict:
    """Pure per-day scoring (unit-testable without the Zenodo file).

    run/ref: {"gh","u","v"} 181x360.  Returns per-field rel_l2 (canvas),
    rel_l2_cosw (cos-lat weighted) and max_abs; wind fields are
    normalised by the VECTOR reference norm (zero-norm refs raise).
    """
    w = np.cos(np.deg2rad(np.asarray(lat_deg, dtype=np.float64)))[:, None]
    w = np.broadcast_to(w, run["gh"].shape)

    def _rel(diff, den_sq):
        den = float(np.sqrt(den_sq))
        if den == 0.0:
            raise ValueError("reference norm is zero — cannot form a "
                             "relative error")
        return float(np.sqrt(np.sum(diff)) / den)

    ones = np.ones_like(run["gh"])
    out = {}
    dgh = run["gh"] - ref["gh"]
    out["gh"] = {
        "rel_l2": _rel(ones * dgh ** 2, np.sum(ones * ref["gh"] ** 2)),
        "rel_l2_cosw": _rel(w * dgh ** 2, np.sum(w * ref["gh"] ** 2)),
        "max_abs": float(np.abs(dgh).max()),
    }
    for key in ("u", "v"):
        diff = run[key] - ref[key]
        out[key] = {
            "rel_l2": _rel(ones * diff ** 2,
                           np.sum(ref["u"] ** 2) + np.sum(ref["v"] ** 2)),
            "rel_l2_cosw": _rel(w * diff ** 2,
                                np.sum(w * ref["u"] ** 2)
                                + np.sum(w * ref["v"] ** 2)),
            "max_abs": float(np.abs(diff).max()),
        }
    return out


def verdict(scores_by_day: dict, max_gh: float, max_wind: float) -> bool:
    """CLOSED-form enforcement: every scored day's gh and wind rel-L2
    metrics (BOTH weightings) must be finite and <= their bound.  A NaN
    anywhere is a FAIL, never a pass-through (codex c6 r1 #4); an EMPTY
    score set raises rather than passing vacuously (codex c6 r2 #6)."""
    if not scores_by_day:
        raise ValueError("no scored days — nothing to enforce")
    for bound in (max_gh, max_wind):
        if not (np.isfinite(bound) and bound > 0.0):
            raise ValueError(
                f"enforcement bound {bound!r} must be finite and > 0")
    for s in scores_by_day.values():
        for key, bound in (("gh", max_gh), ("u", max_wind),
                           ("v", max_wind)):
            for metric in ("rel_l2", "rel_l2_cosw"):
                val = s[key][metric]
                if not (np.isfinite(val) and val <= bound):
                    return False
    return True


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
                    help="gate: deck-record + coverage checks, then "
                         "rel_l2 AND rel_l2_cosw <= the explicit bounds")
    ap.add_argument("--max-gh", type=float, default=None,
                    help="gh rel-L2 bound (REQUIRED with --enforce; no "
                         "default until a measured calibration lands "
                         "here with its job id)")
    ap.add_argument("--max-wind", type=float, default=None,
                    help="u/v rel-L2 bound (REQUIRED with --enforce)")
    ap.add_argument("--allow-ic-only", action="store_true",
                    help="permit enforcing an npz with requested_days=0 "
                         "(the pure IC gate)")
    args = ap.parse_args()

    if args.enforce and (args.max_gh is None or args.max_wind is None):
        ap.error("--enforce requires explicit --max-gh and --max-wind "
                 "(there are deliberately no default bounds)")
    if args.enforce and args.days is not None:
        # codex c6 r2 P0: '--enforce --days 0' would score (and PASS)
        # the IC alone while the coverage check certified a full run —
        # enforcement always scores EVERY day the npz carries.
        ap.error("--days cannot restrict an enforced score; enforcement "
                 "covers every whole day in the npz")

    run = load_run(args.npz)
    t = run["times_days"]
    if args.days is None:
        days = [float(d) for d in t if abs(d - round(d)) < 1e-9]
    else:
        days = [float(x) for x in args.days.split(",")]
        for d in days:
            if not np.any(np.abs(t - d) < 1e-9):
                raise ContractError(f"run npz has no day {d} frame")
    for d in days:                     # preflight vs the reference axis
        if d != 0.0 and not (1.0 <= d <= REF_MAX_DAY
                             and abs(d - round(d)) < 1e-9):
            raise ContractError(
                f"day {d} is outside the reference coverage "
                f"(whole days 1..{REF_MAX_DAY:g}, plus 0 = IC)")
    lat_deg, _ = _canvas()

    print(f"reference: {REF_CASE}/rundir/atmos_daily.nc "
          "(instantaneous daily; ps==delp/FV3_GRAV verified 1.7e-7)")
    print("protocol: run = nearest-cell + c2l_ord2 on the reference "
          "T-cell canvas; reference = fregrid + c2l_ord4 — scores are "
          "floored by the remap-protocol difference (envelope level)")
    rec = run["record"]
    print(f"run config record: {rec if rec else 'ABSENT (pre-record npz)'}")

    if args.enforce:
        problems = check_deck_record(rec)
        if problems:
            print("CASE6_DUO_TARGET_GATE: FAIL (config record != deck): "
                  + "; ".join(problems))
            return 1
        try:
            check_coverage(t, rec, allow_ic_only=args.allow_ic_only)
        except ContractError as e:
            print(f"CASE6_DUO_TARGET_GATE: FAIL (coverage): {e}")
            return 1

    if args.analytic:
        s = score_day(analytic_ic_fields(), load_reference(0.0), lat_deg)
        _print_table("ANALYTIC formulas vs reference IC "
                     "(stepper-independent)", s)

    scores_by_day = {}
    for day in days:
        k = int(np.argmin(np.abs(t - day)))
        frame = {key: run[key][k] for key in ("gh", "u", "v")}
        s = score_day(frame, load_reference(day), lat_deg)
        scores_by_day[day] = s
        label = ("IC (day 0) run vs reference *_ic" if day == 0.0
                 else f"day {day:g} run vs reference")
        _print_table(label, s)

    if args.enforce:
        ok = verdict(scores_by_day, args.max_gh, args.max_wind)
        print("CASE6_DUO_TARGET_GATE:",
              "PASS" if ok else
              f"FAIL (a rel-L2 metric is not finite-and-<= gh<"
              f"{args.max_gh:g} / wind<{args.max_wind:g})")
        return 0 if ok else 1
    print("CASE6_DUO_TARGET_GATE: REPORT-ONLY (pass --enforce with "
          "explicit bounds to gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
