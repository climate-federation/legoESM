#!/usr/bin/env python
"""SW duo-target gate — score a ``run_duo_stepper_case6.py`` npz
against an authoritative FV3 duo reference (Mouallem 2023, Zenodo
8327578) ``C48.sw.case<N>.alpha<A>.duo.hord8/rundir/atmos_daily.nc``.

CASES: ``--case 6`` (Rossby-Haurwitz wave 4, alpha0 only — the
original scope of this gate) and ``--case 2`` (solid-body, alpha0 and
alpha45).  ``--ref-alpha`` selects the reference deck's namelist alpha
tag; the npz must RECORD the matching runner ``--alpha`` to be
enforceable.  ALPHA UNITS: the raw ``test_case_nml`` value is RADIANS
(the pinned Fortran feeds it straight to ``sin``/``cos``; the alpha45
decks rotate by 45 rad ~ 58.31 deg — verified against ``ps_ic`` at
rel-L2 1.12e-04 vs 1.11e-01 for the pi/4 reading; see the runner
docstring).

NOT SCOREABLE HERE, on purpose:
* ``C48.sw.case8.alpha45.*`` — that deck's real knob is ``target_lat``
  -90 -> -135 (a 45-deg Schmidt GRID rotation; its ``alpha = 0.75``
  echo is dead in case(8), test_cases.F90:1375-1381).  The port has no
  rotated-target gridstruct, so scoring against it would be a grid
  confound.
* ``C48.sw.case6.alpha45.*`` / ``case111.alpha45.*`` — no such
  references exist in the Zenodo set.
* case-2 hord5/6/10 siblings — references exist but this gate pins the
  hord8 deck the stepper config implements.

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

ADDITIONAL FLOOR AT case-2 alpha45, MEASURED (job 9369321): the rotated
flow has nonzero wind at the geographic poles, where the lat-lon (u, v)
decomposition of the duplicated pole rows is degenerate — analytic
winds vs the fregrid reference IC differ by max 33.4 m/s at rows 0/180
(10.7 at row 1, 6.06 at row 2, 3.3e-3 at the equator), putting the
UNWEIGHTED wind rel-L2 floor at ~8.2e-2 (cos-lat: ~5.0e-3) where the
alpha0 floor is ~4e-3.  Enforcement keeps BOTH weightings (same metric
family as case 6 — protocol held fixed), so alpha45 wind bounds are
pole-artifact-limited, not physics-limited: a bound calibrated at ~2x
the measured worst day constrains the rotated physics mostly through
the cos-lat metric.  Stated here so nobody reads the alpha45 wind
bound as a 10x-looser physics gate.

``--analytic`` scores the ANALYTIC RH4 fields (shared
williamson_sw_analytic module, GFS constants), evaluated directly at
the reference canvas points, against ``ps_ic``/``ua_ic``/``va_ic``.
That check has NO dependence on the port's stepper, lens or sampling —
it isolates "is the reference initialised with the same formulas?", and
its error is bounded by the reference's own C48 discretisation +
fregrid remap.

ENFORCEMENT (codex c6 r1 #1/#2/#4/#5): report-only by default.
``--enforce`` additionally requires
* EXPLICIT ``--max-gh``/``--max-wind`` bounds (finite, positive) — no
  baked-in defaults: bounds nobody measured are a gate that cannot
  mean anything.  MEASURED CALIBRATION (jobs 9356453 + 9356856,
  2026-08-10, C48 deck config, IC + days 1..5; the two runs — code
  ebf475383 and 2033b1f87 — produced BIT-IDENTICAL frames): worst
  rel_l2 over all days was gh 2.304e-3 / wind 2.850e-2, time-flat at
  the remap-protocol floor (IC gh 2.190e-3).  Recommended bounds at
  ~2x the worst measured day: ``--max-gh 5e-3 --max-wind 6e-2`` — a
  regression that doubles the day-5 error trips, remap noise does not.
  CASE-2 CALIBRATION (job 9369321, 2026-08-11, code 5e1a2549d, C48
  deck config, IC + days 1..5, both time-flat):
  * alpha0:  worst gh 5.433e-3 / wind 9.967e-3 (cosw 4.90e-3 /
    6.32e-3) — recommended ``--max-gh 1.1e-2 --max-wind 2e-2``;
  * alpha45: worst gh 5.523e-3 / wind 1.233e-1 (cosw 5.11e-3 /
    1.719e-2) — recommended ``--max-gh 1.1e-2 --max-wind 2.5e-1``.
    The alpha45 wind bound is POLE-ARTIFACT-LIMITED (see the floor
    note above), and its cos-lat metric — enforced at the same bound —
    is what actually constrains the rotated physics (~1.7e-2 measured
    vs the 2.5e-1 ceiling; a physics regression an order below the
    unweighted floor still trips nothing unweighted, by construction);
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
REF_CASE = "C48.sw.case6.alpha0.duo.hord8"     # historical default deck

# References this gate may score against — EVERY entry verified to
# exist in the Zenodo set; anything else is refused (never invent a
# comparison).  Keys: (case, ref_alpha as the namelist tag number).
AVAILABLE_REFS = {(6, 0.0), (2, 0.0), (2, 45.0)}

# Per-case reference time-axis coverage: case-6 atmos_daily.nc is
# daily 1..100 d; the case-2 files are HOURLY (nt=120) ending at 5.0 d
# (whole days 1..5 exist exactly on the axis; selection stays by value).
REF_MAX_DAYS = {6: 100.0, 2: 5.0}

# Shared resolved-deck keys (every C48 duo hord8 SW deck echoes these):
_DECK_COMMON = {
    "n": 48,
    "n_split": 7,
    "d_ext": 0.0,
    "k2e_nord": 2,        # the authoritative live default (no nml knob)
    "oracle_conventions": True,
    "ext_exclude": "",
}


def deck_record(case: int, ref_alpha: float) -> dict:
    """The reference deck's resolved configuration (logfile.000000.out)
    for ``(case, ref_alpha)`` — an npz must record exactly this to be
    enforceable as a deck score.  Refuses unavailable references."""
    if (case, float(ref_alpha)) not in AVAILABLE_REFS:
        raise ValueError(
            f"no Zenodo duo hord8 reference for case {case} "
            f"alpha {ref_alpha:g}; available: {sorted(AVAILABLE_REFS)}")
    per_case = {
        6: {"dt_atmos": 1200.0, "d4_bg": 0.0},     # logfile :184/:403
        2: {"dt_atmos": 3600.0, "d4_bg": 0.12},    # logfile :184/:403
    }[case]
    return {**_DECK_COMMON, **per_case,
            "case": case, "alpha": float(ref_alpha)}


def ref_case_name(case: int, ref_alpha: float) -> str:
    if (case, float(ref_alpha)) not in AVAILABLE_REFS:
        raise ValueError(
            f"no Zenodo duo hord8 reference for case {case} "
            f"alpha {ref_alpha:g}; available: {sorted(AVAILABLE_REFS)}")
    if float(ref_alpha) != int(ref_alpha):
        # int() would silently truncate a future fractional tag into a
        # DIFFERENT deck's name (GLM-carried review, alpha45 r1)
        raise ValueError(
            f"non-integer reference alpha tag {ref_alpha!r} has no "
            "name mapping")
    return f"C48.sw.case{case}.alpha{int(ref_alpha)}.duo.hord8"


# Plausibility bands — SENTINEL DETECTORS, not physics gates: the RH4
# state lives in gh ~ [7.8e4, 1.04e5] and |V| <= ~100 m/s; the known
# failure modes (BIG_NUMBER = 1e8 corner sentinel, a NaN that became a
# huge finite through arithmetic) sit orders of magnitude outside.  A
# genuinely evolving solution never approaches these bounds.  The
# case-2 steady state spans gh ~ [1.07e4, 2.94e4], so its band's lower
# sentinel bound sits below 1e4 (mirrors the runner's CASE_DECKS).
GH_PLAUSIBLE_BY_CASE = {6: (1.0e4, 5.0e5), 2: (5.0e3, 5.0e5)}
WIND_PLAUSIBLE_MAX = 500.0             # m/s


class ContractError(ValueError):
    """The run npz violates the case-6 input contract."""


def _canvas() -> tuple[np.ndarray, np.ndarray]:
    return (np.linspace(-90.0, 90.0, 181),
            0.5 + np.arange(360, dtype=float))


def load_run(npz_path: str, case: int = 6) -> dict:
    """Load + contract-check the runner npz (arrays, canvas, times,
    finiteness, per-case plausibility band); carry the runner's config
    record through for the enforcement manifest check.

    ``case`` selects the plausibility band AND is cross-checked against
    the npz's recorded case in REPORT MODE TOO: scoring a case-2 npz
    against the case-6 reference is meaningless in any mode, so the
    mismatch is a contract error, not merely an enforcement one.  An
    npz with no ``case`` key (pre-matrix runner) is accepted as case 6
    only.
    """
    if case not in GH_PLAUSIBLE_BY_CASE:
        raise ValueError(f"unknown case {case}")
    z = np.load(npz_path, allow_pickle=True)
    if "case" in z.files:
        npz_case = int(np.asarray(z["case"]).item())
        if npz_case != case:
            raise ContractError(
                f"npz records case {npz_case}, gate invoked for case "
                f"{case} — refusing a cross-case score")
    elif case != 6:
        raise ContractError(
            "npz records no case (pre-matrix runner output) — such an "
            "npz is scoreable as case 6 only")
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
    lo, hi = GH_PLAUSIBLE_BY_CASE[case]
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
    for key in (*_DECK_COMMON, "dt_atmos", "d4_bg", "case", "alpha",
                "git_sha"):
        if key in z.files:
            record[key] = np.asarray(z[key]).item()
    if "requested_days" in z.files:
        record["requested_days"] = int(np.asarray(
            z["requested_days"]).item())
    return {"times_days": t, "record": record, **fields}


def check_deck_record(record: dict, deck: dict | None = None) -> list[str]:
    """Mismatches between the npz's recorded config and the reference
    deck (default: the case-6 alpha0 deck); a MISSING key is a mismatch
    (an npz from before the record was written cannot be enforced).
    Comparison is TYPE-NORMALISED (numpy scalars arrive via ``.item()``;
    ints and floats compare by value, strings as str, bools as bool) so
    a dtype change cannot smuggle a mismatch through ``==`` (codex c6
    r2 #5).  The deck dict includes ``case`` and ``alpha``, so an npz
    from the wrong case or rotation can never pass enforcement.

    LIMITATION, stated: this is the runner's SELF-ATTESTATION — it
    certifies what the npz says it ran, not what a process actually
    executed.  The npz's ``git_sha`` (required present) is the audit
    hook; byte-level provenance is out of scope for this gate.
    """
    if deck is None:
        deck = deck_record(6, 0.0)
    problems = []
    for key, want in deck.items():
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


def analytic_ic_fields(case: int = 6, alpha: float = 0.0) -> dict:
    """Analytic (gh, u, v) at the reference canvas points — GFS
    constants, shared :mod:`legoesm.core.williamson_sw_analytic`; no
    stepper, no lens, no sampling.

    case 6: RH4 (alpha must be 0 — the RH4 IC has no alpha term).
    case 2: rotated solid body; ``alpha`` in RAW namelist units
    (RADIANS — pass 45.0 for the alpha45 deck).
    """
    from legoesm.core.williamson_sw_analytic import (
        RH4_MEAN_DEPTH_M,
        W2_GH0,
        rossby_haurwitz_4_geopotential,
        rossby_haurwitz_4_winds,
        solid_body_geopotential,
        solid_body_rotation_speed,
        solid_body_winds,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_GRAV,
        FV3_OMEGA,
        FV3_RADIUS_M,
    )

    lat_deg, lon_deg = _canvas()
    lon = np.deg2rad(lon_deg)[None, :]
    lat = np.deg2rad(lat_deg)[:, None]
    if case == 6:
        if alpha != 0.0:
            raise ValueError("case 6 has no alpha term in its IC")
        gh = rossby_haurwitz_4_geopotential(
            lon, lat, radius=FV3_RADIUS_M, omega=FV3_OMEGA,
            gh0=RH4_MEAN_DEPTH_M * FV3_GRAV)
        u, v = rossby_haurwitz_4_winds(lon, lat, radius=FV3_RADIUS_M)
    elif case == 2:
        u0 = solid_body_rotation_speed(FV3_RADIUS_M)
        gh = solid_body_geopotential(
            lon, lat, radius=FV3_RADIUS_M, omega=FV3_OMEGA, u0=u0,
            gh0=W2_GH0, alpha=alpha)
        u, v = solid_body_winds(lon, lat, u0=u0, alpha=alpha)
    else:
        raise ValueError(f"unknown case {case}")
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
    ap.add_argument("--case", type=int, default=6, choices=(2, 6),
                    help="Zenodo SW deck: 6 = RH4 (default), 2 = "
                         "solid-body")
    ap.add_argument("--ref-alpha", type=float, default=0.0,
                    choices=(0.0, 45.0),
                    help="reference deck alpha tag (raw namelist units "
                         "== RADIANS; 45 only exists for case 2)")
    ap.add_argument("--days", default=None,
                    help="comma list of run days to score (default: every "
                         "whole day in the npz, IC always included)")
    ap.add_argument("--analytic", action="store_true",
                    help="also score the analytic IC fields at the "
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
    try:
        ref_case = ref_case_name(args.case, args.ref_alpha)
        deck = deck_record(args.case, args.ref_alpha)
    except ValueError as e:
        ap.error(str(e))
    gate_tag = f"CASE{args.case}_DUO_TARGET_GATE"
    ref_max_day = REF_MAX_DAYS[args.case]

    run = load_run(args.npz, case=args.case)
    rec_alpha = run["record"].get("alpha")
    if rec_alpha is not None and float(rec_alpha) != float(args.ref_alpha):
        # REPORT MODE TOO (GLM-carried review, alpha45 r1): scoring an
        # alpha0 npz against the alpha45 reference produces garbage
        # numbers with no banner — cross-rotation is as meaningless as
        # cross-case, so it is a contract error in every mode.  (A
        # legacy npz with no alpha key is case-6-only via load_run,
        # and case 6 has only the alpha0 reference.)
        raise ContractError(
            f"npz records alpha {rec_alpha!r}, gate invoked for the "
            f"alpha{args.ref_alpha:g} reference — refusing a "
            "cross-rotation score (report mode included)")
    t = run["times_days"]
    if args.days is None:
        days = [float(d) for d in t if abs(d - round(d)) < 1e-9]
    else:
        days = [float(x) for x in args.days.split(",")]
        for d in days:
            if not np.any(np.abs(t - d) < 1e-9):
                raise ContractError(f"run npz has no day {d} frame")
    for d in days:                     # preflight vs the reference axis
        if d != 0.0 and not (1.0 <= d <= ref_max_day
                             and abs(d - round(d)) < 1e-9):
            raise ContractError(
                f"day {d} is outside the reference coverage "
                f"(whole days 1..{ref_max_day:g}, plus 0 = IC)")
    lat_deg, _ = _canvas()

    print(f"reference: {ref_case}/rundir/atmos_daily.nc "
          "(instantaneous; ps==delp/FV3_GRAV verified 1.7e-7 on case 6)")
    print("protocol: run = nearest-cell + c2l_ord2 on the reference "
          "T-cell canvas; reference = fregrid + c2l_ord4 — scores are "
          "floored by the remap-protocol difference (envelope level)")
    rec = run["record"]
    print(f"run config record: {rec if rec else 'ABSENT (pre-record npz)'}")

    if args.enforce:
        problems = check_deck_record(rec, deck)
        if problems:
            print(f"{gate_tag}: FAIL (config record != deck): "
                  + "; ".join(problems))
            return 1
        try:
            check_coverage(t, rec, allow_ic_only=args.allow_ic_only)
        except ContractError as e:
            print(f"{gate_tag}: FAIL (coverage): {e}")
            return 1

    if args.analytic:
        # analytic alpha = the DECK's alpha (what the reference ran),
        # never the npz's — this arm scores formulas, not the run
        s = score_day(analytic_ic_fields(args.case, args.ref_alpha),
                      load_reference(0.0, case=ref_case), lat_deg)
        _print_table("ANALYTIC formulas vs reference IC "
                     "(stepper-independent)", s)

    scores_by_day = {}
    for day in days:
        k = int(np.argmin(np.abs(t - day)))
        frame = {key: run[key][k] for key in ("gh", "u", "v")}
        s = score_day(frame, load_reference(day, case=ref_case), lat_deg)
        scores_by_day[day] = s
        label = ("IC (day 0) run vs reference *_ic" if day == 0.0
                 else f"day {day:g} run vs reference")
        _print_table(label, s)

    if args.enforce:
        ok = verdict(scores_by_day, args.max_gh, args.max_wind)
        print(f"{gate_tag}:",
              "PASS" if ok else
              f"FAIL (a rel-L2 metric is not finite-and-<= gh<"
              f"{args.max_gh:g} / wind<{args.max_wind:g})")
        return 0 if ok else 1
    print(f"{gate_tag}: REPORT-ONLY (pass --enforce with "
          "explicit bounds to gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
