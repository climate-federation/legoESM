#!/usr/bin/env python
"""Does legoESM's wall 2-step surface flicker OUT-GENERATE NEMO's, out-RING
it, or neither?

The campaign's one live physics thread was that legoESM's free-running sea
surface carries a two-step (2*rn_Dt) mode at 2.9x NEMO's amplitude with 85% of
it on land-adjacent rows against NEMO's 4% (``dino_eta_wave_field_result.md``).
Both of those are EARLY-WINDOW numbers -- the 85% is a first-SAMPLE locus --
and "where does it come from" had two answers needing different fixes:

  GENERATION -- some term's wall stencil injects alternating-sign noise.
  DAMPING    -- the time-filter or barotropic-averaging path removes less of
                the mode at the walls (steady floor = source / damping).

The observable costs nothing to resolve: the artifacts are sampled EVERY STEP
(``capture_every_steps = 1``, ``dt = 2700 s``), so a 2*dt mode sits exactly at
the series' Nyquist frequency -- no rerun, no aliasing.  What DID need care is
which statistic separates the two stories, and the first version of this probe
got that wrong in a way review caught.

WHAT ACTUALLY SEPARATES THEM, in order of how much weight it carries:

1. THE LOCUS OVER TIME (``wall_share``).  A wall MODE keeps its locus; a wall
   TRANSIENT loses it.  This is the statistic the source question turns on and
   it needs no floor subtraction and no fit.
2. THE RESPONSE LANE (``--nemo-free/--lego-free``).  Each model minus its OWN
   free run, so both ring down from an IDENTICAL imposed bump.  This is the
   only lane in which the two decay RATES are comparable at all -- in the free
   lane NEMO has no launch transient, so there is no NEMO rate to be slower
   than, and a free-lane tau ratio divides a real fit by an untouched initial
   guess.  Free-lane fit ratios are therefore WITHHELD unless both sides'
   fits are identifiable (``fit_usable``).
3. THE AMPLITUDE RATIO, last -- reported under three named estimators with a
   block-bootstrap interval, because on this data they disagree by ~30% and a
   single unlabelled number is spurious precision.

Traps this probe is built against.  The first four were earned earlier in the
campaign; the last three were earned by THIS probe failing them in review.
  * land in the denominator -- every statistic is area-weighted over the
    mesh's own ``tmask``;
  * the re-entrant channel's periodic seam mislabelled as wall -- the region
    split is the reviewed ``locus_partition`` with ``periodic_i=True``;
  * a curvature high-pass quoted as a notch -- every amplitude is quoted next
    to a leakage floor, and a level within 2x of its floor names nothing
    (``*_resolved``);
  * leap-frog parity -- the fit runs on the 2-sample mean and the raw odd/even
    split is reported separately (``ewt.alternation_ratio``, reused rather
    than re-derived: an earlier local copy computed the RECIPROCAL under a
    near-identical name).
  * THE FLOOR MUST BE MEASURED ON THE WINDOW IT CORRECTS.  A floor taken over
    the launch transient is inflated by the signal under study; subtracted
    from a late-time amplitude it reversed the sign of a conclusion.  The
    floor now takes an explicit ``window`` and the probe pairs each with its
    own.
  * AN IDENTIFIABILITY GATE MUST TEST SEPARATION FROM ZERO.  The first one
    asked only that the fitted amplitude be positive, which 1e-10 satisfies,
    so it returned "identifiable" for every degenerate fit and "not" for the
    best one.
  * A CONTROL MUST BE ABLE TO FAIL.  The land control was vacuous TWICE: the
    shared poison plants a constant and the Nyquist operator annihilates
    constants exactly (on the real data, where dry-cell eta is exactly 0.0 at
    every step, it moved the statistic by a ratio of 1.000000); and even
    alternating, the production area weights are already zero on land, so the
    region mask was never the thing being tested.  It now plants an
    alternating poison (``plant_dry_violation(alternating=True)``, the flag
    added to the SHARED helper rather than a near-duplicate local copy) AND
    runs against unmasked weights, and a unit test shows a mask admitting
    land turns it red.

An amplitude, a floor and a locus are reported for every region and window;
the interpretation is NOT baked in -- this tool prints numbers, never a
verdict.  The verdict lives in ``docs/ocean/fidelity/dino_wall_flicker_source.md``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eta_wave_twin as ewt  # noqa: E402  (sibling probe, reviewed instrument)


def region_amplitude(alt: np.ndarray, mask: np.ndarray,
                     area: np.ndarray) -> np.ndarray:
    """Area-weighted rms of the alternating field over one region.

    Thin alias for the campaign's single reduction; kept as a name because
    every call here is per-region.
    """
    return ewt.weighted_rms_series(alt, mask, area)


def region_leakage_floor(field: np.ndarray, mask: np.ndarray,
                         area: np.ndarray, window: slice) -> float:
    """``eta_wave_twin.two_dt_leakage_floor`` for one region and one window.

    BOTH arguments matter and both were review blockers.  The wall is where
    the slow field's curvature is largest, so its leakage floor is largest
    too and a global floor cannot bound a wall amplitude.  And the floor must
    be measured over the SAME samples as the amplitude it is quoted against:
    a floor taken over the launch transient inflates by the very signal under
    study, and subtracting it from a late-time amplitude reversed the sign of
    a conclusion (2026-08-24 review).
    """
    return ewt.two_dt_leakage_floor(field, mask, area, mask=mask,
                                    window=window)


def fit_decay(amp: np.ndarray) -> dict:
    """Least-squares ``amp(n) = C + A*exp(-n/tau)``.

    Run on the 2-sample running mean so a leap-frog parity split cannot be
    read as a decay; the raw odd/even ratio is reported separately so the
    reader can see whether that mattered.  ``scipy`` is already a hard
    dependency of this repo, so no hand-rolled optimiser.
    """
    from scipy.optimize import curve_fit

    sm = 0.5 * (amp[:-1] + amp[1:])
    n = np.arange(sm.size, dtype=float)

    def model(x, c, a, tau):
        return c + a * np.exp(-x / tau)

    c0 = float(np.median(sm[sm.size // 2:]))
    a0 = float(max(sm[0] - c0, 1e-12))
    lo_b = (0.0, 0.0, 0.5)
    hi_b = (np.inf, np.inf, 10.0 * sm.size)
    try:
        p, cov = curve_fit(model, n, sm, p0=(c0, a0, 10.0),
                           bounds=(lo_b, hi_b), maxfev=100000)
        err = np.sqrt(np.diag(cov))
    except Exception as exc:                       # noqa: BLE001
        return {"converged": False, "reason": str(exc)}
    resid = sm - model(n, *p)
    ss = 1.0 - float(resid @ resid) / float(((sm - sm.mean()) ** 2).sum())
    # A transient that never decays inside the window makes tau unidentifiable
    # (the fit trades A against C), so say so instead of quoting a number.
    # IDENTIFIABILITY, rewritten after review.  The first version asked only
    # that tau be under half the window and A be positive -- which A = 1e-10
    # satisfies, so it returned True for every DEGENERATE fit (the optimizer
    # never moved off p0, stderr on tau 2800x its value) and False for the
    # best-fitting one.  It was anti-correlated with the thing it named.  The
    # test is now separation from zero on BOTH free parameters plus an
    # exponential that beats a constant.
    usable = bool(p[1] > 3.0 * err[1] and p[2] > 3.0 * err[2]
                  and np.isfinite(err[1]) and np.isfinite(err[2])
                  and ss > 0.0 and p[2] < 0.5 * sm.size)
    # WINDOW SENSITIVITY, not the covariance stderr.  curve_fit's stderr
    # answers "how tightly does THIS window pin tau", which was +/-2.9% here
    # while refitting on defensible sub-windows moved tau by +/-47%.  The
    # honest spread is the one across windows, so it is what gets reported.
    spread = [p[2]]
    for lo, hi in ((2, 20), (2, None), (5, None), (0, sm.size // 2)):
        sub = sm[lo:hi]
        if sub.size < 8:
            continue
        try:
            q, _ = curve_fit(model, np.arange(sub.size, dtype=float), sub,
                             p0=(float(np.median(sub[sub.size // 2:])),
                                 float(max(sub[0] - sub[-1], 1e-12)), 10.0),
                             bounds=(lo_b, hi_b), maxfev=100000)
            spread.append(float(q[2]))
        except Exception:                              # noqa: BLE001, S110
            pass
    return {"converged": True,
            "tau_window_sensitivity_steps": [float(min(spread)),
                                             float(max(spread))],
            "steady_C_m": float(p[0]), "transient_A_m": float(p[1]),
            "tau_steps": float(p[2]),
            "stderr": {"C": float(err[0]), "A": float(err[1]),
                       "tau": float(err[2])},
            "r2": ss,
            "fit_usable": usable,
            "n_fit_samples": int(sm.size)}


# ``eta_wave_twin.alternation_ratio`` is the campaign's odd/even split and it
# is used verbatim here.  The first version of this probe defined its own
# "parity_ratio" that computed the RECIPROCAL under a near-identical name --
# two JSONs in one campaign carrying inverted quantities is exactly the trap
# the no-duplicate-numerics rule exists for.


def decorrelation_lag(x: np.ndarray) -> int:
    """First lag at which the series' autocorrelation drops below 1/e.

    The bootstrap block length is not a free choice: too short and the CI is
    too narrow, too long and it is too wide.  Measure it rather than picking
    10 because 10 looks round (review: the measured lag here is 2, and the
    CI at 10 was wider than the data warrants).
    """
    y = np.asarray(x, dtype=float)
    y = y - y.mean()
    if y.size < 4 or not np.any(y):
        return 1
    ac = np.correlate(y, y, mode="full")[y.size - 1:]
    ac = ac / ac[0]
    below = np.nonzero(ac < 1.0 / np.e)[0]
    return int(below[0]) if below.size else max(1, y.size // 4)


def block_bootstrap_ratio_ci(a: np.ndarray, b: np.ndarray,
                             block: int | None = None, n_boot: int = 2000,
                             seed: int = 0) -> dict:
    """CI on the ratio of two amplitude series' means, resampling BLOCKS.

    A per-sample bootstrap would treat consecutive samples as independent,
    which they are not (the alternating envelope is autocorrelated), and would
    return a CI several times too narrow.  Blocks of ``block`` samples keep
    the short-range correlation.

    Registered as a REQUIREMENT after review: the day-190 decision thresholds
    (1.5 / 2.5) sit inside this estimator's own noise, so a ratio quoted
    without its interval can "confirm" or "refute" with no change in physics.
    """
    rng = np.random.default_rng(seed)
    n = a.size
    if n != b.size:
        raise SystemExit("block_bootstrap_ratio_ci: series lengths differ")
    measured = max(decorrelation_lag(a), decorrelation_lag(b))
    if block is None:
        block = max(1, measured)
    if n < 2 * block:
        return {"ci90": None, "reason": "window shorter than two blocks"}
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))

    def draw() -> np.ndarray:
        s0 = rng.choice(starts, size=k)
        return np.concatenate([np.arange(j, j + block) for j in s0])[:n]

    # The two models are INDEPENDENT trajectories after step 1, so resampling
    # both at the SAME indices imposes a pairing the data does not have and
    # narrows the interval.  Each side is drawn on its own indices.
    out = np.empty(n_boot)
    for i in range(n_boot):
        db = b[draw()].mean()
        out[i] = a[draw()].mean() / db if db > 0 else np.nan
    out = out[np.isfinite(out)]
    if out.size < n_boot // 2:
        return {"ci90": None, "reason": "too many degenerate resamples"}
    lo, hi = (float(v) for v in np.percentile(out, [5.0, 95.0]))
    return {"ci90": [lo, hi], "n_boot": int(out.size), "block": int(block),
            "measured_decorrelation_lag": int(measured),
            "resampling": "independent per side"}


def concentration(alt: np.ndarray, mask: np.ndarray, area: np.ndarray,
                  window: slice) -> dict:
    """How FEW cells carry the region's variance.

    Two fields with the same rms can be a broad band or one hot column, and
    those are different findings.  Review measured NEMO's late wall rms to be
    82% in ten cells while legoESM's is spread over dozens; an rms ratio near
    1 hides that completely.  Returned: the number of cells holding half the
    area-weighted variance, and the share held by the top ten.
    """
    x = alt[window][:, mask]
    w = area[mask]
    v = (w * (x ** 2).mean(axis=0))
    tot = float(v.sum())
    if tot <= 0.0:
        return {"cells_for_half_variance": None, "top10_share": None}
    order = np.argsort(v)[::-1]
    csum = np.cumsum(v[order]) / tot
    return {"cells_for_half_variance": int(np.searchsorted(csum, 0.5) + 1),
            "top10_share": float(csum[min(9, csum.size - 1)]),
            "region_cells": int(mask.sum())}


def ratio_estimators(a_amp: np.ndarray, b_amp: np.ndarray,
                     a_alt: np.ndarray, b_alt: np.ndarray,
                     mask: np.ndarray, area: np.ndarray,
                     window: slice) -> dict:
    """The SAME comparison under three defensible estimators.

    NOTE the third is NOT area-weighted while the first two are: a median over
    cells cannot carry cell weights meaningfully. On a grid spanning 70S-70N
    that makes it a different question, not a worse answer -- "the typical
    cell" rather than "the typical square metre" -- and it is the estimator
    least sensitive to a few hot cells, which is why it is kept.

    On identical data these disagree by ~30%, so quoting one without naming it
    is spurious precision.  None is privileged here; all three are reported.
    """
    mean_of_rms = _ratio(float(a_amp[window].mean()),
                         float(b_amp[window].mean()))
    rms_over_time = _ratio(float(np.sqrt((a_amp[window] ** 2).mean())),
                           float(np.sqrt((b_amp[window] ** 2).mean())))
    def per_cell(alt: np.ndarray) -> np.ndarray:
        return np.sqrt((alt[window][:, mask] ** 2).mean(axis=0))

    median_cell = _ratio(float(np.median(per_cell(a_alt))),
                         float(np.median(per_cell(b_alt))))
    return {"mean_of_per_sample_rms": mean_of_rms,
            "rms_over_time": rms_over_time,
            "median_cell": median_cell}


def _ratio(a: float, b: float) -> float | None:
    """``a/b``, or None when the denominator cannot carry a ratio.

    Returning None (which serialises as JSON ``null``) rather than inf/NaN:
    ``json.dump`` defaults to ``allow_nan=True``, so a NaN ratio would be
    written silently and then read as a number.
    """
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0.0:
        return None
    return float(a / b)


def _registration(nemo: dict, lego: dict, wet: np.ndarray) -> dict:
    """How well the two models' samples sit on the SAME leap-frog time level.

    Equal ``t_seconds`` is the producers' bookkeeping, not evidence.  Both
    artifacts carry the models' own step-1 states, so the honest statement is
    the size of the step-1 difference measured against NEMO's OWN now-minus-
    before gap: if the former is a large fraction of the latter, a one-sample
    registration error is not excluded by this data and every early-window
    number inherits that uncertainty.
    """
    out = {"t_seconds_equal": bool(np.array_equal(nemo["t"], lego["t"]))}
    d1 = float(np.abs(nemo["eta"][0][wet] - lego["eta"][0][wet]).max())
    out["step1_max_abs_diff_m"] = d1
    if "eta_before" in nemo:
        gap = float(np.abs(
            nemo["eta"][0][wet] - nemo["eta_before"][0][wet]).max())
        out["nemo_now_minus_before_max_abs_m"] = gap
        out["step1_diff_as_fraction_of_nemo_time_level_gap"] = _ratio(d1, gap)
    return out


def analyse(nemo_npz: str, lego_npz: str, out: str,
            mesh_mask: str = ewt.MESH_MASK,
            self_check: bool = True,
            nemo_free_npz: str | None = None,
            lego_free_npz: str | None = None,
            single: bool = False) -> dict:
    """Compare the two models' 2-step surface mode, per region, over time.

    With ``nemo_free_npz``/``lego_free_npz`` the observable becomes each
    model's response MINUS ITS OWN FREE RUN.  That lane answers a different
    and sharper question than the free lane: both models are then ringing
    down from the SAME imposed excitation, so the decay rates ARE comparable
    (in the free lane NEMO has no launch transient at all, so there is no
    NEMO decay rate to compare against).  Each side is differenced against
    its own free run, never the other model's, so a background difference
    cannot leak into the response.
    """
    # A CONTROL THAT COMPARES A FILE WITH ITSELF CANNOT FAIL.  Pointing both
    # sides at the same artifact makes every ratio 1.000 BY CONSTRUCTION, and
    # a reader has no way to tell that from a measured agreement.  The
    # single-trajectory use (measuring one field's own locus, e.g. a
    # member-minus-member difference) is legitimate, so it gets an explicit
    # flag -- and in that mode the cross-model ratios are NULLED rather than
    # printed as 1.
    same = (os.path.realpath(nemo_npz) == os.path.realpath(lego_npz)
            and os.path.realpath(str(nemo_free_npz))
            == os.path.realpath(str(lego_free_npz)))
    if same and not single:
        raise SystemExit(
            "--nemo and --lego resolve to the SAME file: every ratio would be "
            "1.000 by construction, not by measurement. Pass --single if you "
            "meant to measure ONE trajectory's own locus; the cross-model "
            "ratios are withheld in that mode.")
    if single and not same:
        raise SystemExit("--single given but the two sides are different "
                         "files; drop the flag or pass one trajectory")
    nemo = ewt.load_side(nemo_npz, "NEMO")
    lego = ewt.load_side(lego_npz, "legoESM")
    if (nemo_free_npz is None) != (lego_free_npz is None):
        raise SystemExit("the response lane needs BOTH free runs or neither "
                         "-- differencing one side only would compare a "
                         "response against a full field")
    lane = "free"
    if nemo_free_npz is not None:
        lane = "minus_own_free"
        nf = ewt.load_side(nemo_free_npz, "NEMO free")
        lf = ewt.load_side(lego_free_npz, "legoESM free")
        for a_, b_ in ((nemo, nf), (lego, lf)):
            if a_["eta"].shape != b_["eta"].shape or not np.array_equal(
                    a_["t"], b_["t"]):
                raise SystemExit("a perturbed run and its free run disagree "
                                 "on shape or time axis")
        nemo = dict(nemo, eta=nemo["eta"] - nf["eta"])
        lego = dict(lego, eta=lego["eta"] - lf["eta"])
    if nemo["eta"].shape != lego["eta"].shape:
        raise SystemExit(f"shape mismatch {nemo['eta'].shape} vs "
                         f"{lego['eta'].shape} -- not a controlled pair")
    if not np.array_equal(nemo["t"], lego["t"]):
        raise SystemExit("the two runs are not on the same time axis")

    import netCDF4
    wet = ewt.wet_mask(mesh_mask)
    with netCDF4.Dataset(mesh_mask) as ds:
        e1t = np.asarray(ds.variables["e1t"][0]).squeeze()
        e2t = np.asarray(ds.variables["e2t"][0]).squeeze()
        lat = np.asarray(ds.variables["gphit"][0]).squeeze()
    area = np.where(wet, e1t * e2t, 0.0)
    regions_only = ewt.locus_partition(wet, lat, periodic_i=True)
    # The wall is TWO different objects and they do not behave alike; an
    # aggregate over both hid a sustained excess on the smaller one.  Both
    # splits are reported: the coarse one for continuity with the campaign's
    # published numbers, the directional one because it is the resolution the
    # source question actually needs.
    walls = ewt.wall_direction_partition(wet, periodic_i=True)
    regions = {"all": wet, **regions_only,
               **{k: v for k, v in walls.items() if v.sum() > 0}}

    sides = {"NEMO": nemo["eta"], "legoESM": lego["eta"]}
    n_alt = nemo["eta"].shape[0] - 2          # samples after the Nyquist op
    # The floor's window must MATCH the amplitude's, and the floor series has
    # already dropped the two padded end samples, so its indices run over
    # n_alt - 2.  first-8 and last-half are expressed in BOTH index spaces.
    win = {"first8": (slice(0, 8), slice(0, 8)),
           "last_half": (slice(n_alt // 2, None),
                         slice(max(0, n_alt // 2 - 1), None))}

    out_d: dict = {"dt_seconds": float(nemo["t"][1] - nemo["t"][0]),
                   "n_samples": int(nemo["eta"].shape[0]),
                   # Outside the free lane one side is a RESPONSE field and
                   # the other a full field, so a step-1 difference between
                   # them is not a registration statistic -- it compared 0.83 m
                   # against 1.8e-7 m and reported the ratio.  Nulled there.
                   "time_level_registration": (
                       _registration(nemo, lego, wet) if lane == "free"
                       else {"withheld": "only meaningful in the free lane"}),
                   "regions": {}}

    for rname, mask in regions.items():
        if mask.sum() == 0:
            continue
        rec: dict = {"cells": int(mask.sum()),
                     "area_fraction": float(area[mask].sum() / area[wet].sum())}
        alt_cache: dict = {}
        for sname, eta in sides.items():
            alt = ewt.two_dt_component(eta)
            amp = region_amplitude(alt, mask, area)
            # Per-sample floor, aligned: floor index i is amp index i+1.
            # A window MEAN above 2x its floor is compatible with one sample
            # carrying it, so the FRACTION of samples that individually clear
            # their own floor is reported next to every fitted decay constant
            # -- that is what says whether a fit was run on signal or on
            # leakage (2026-08-24 review: 1 of 40 response-lane samples
            # cleared the bar while the mean read 2.03x).
            floor_ser = ewt.two_dt_leakage_floor(
                eta, mask, area, mask=mask, return_series=True)
            amp_al = amp[1:-1]
            side: dict = {"amplitude_by_sample_m": [float(v) for v in amp],
                          "alternation_ratio_first8": ewt.alternation_ratio(
                              [float(v) for v in amp]),
                          "fit": fit_decay(amp)}
            for wname, (a_sl, f_sl) in win.items():
                a = float(amp[a_sl].mean())
                f = region_leakage_floor(eta, mask, area, f_sl)
                side[f"{wname}_mean_m"] = a
                side[f"{wname}_floor_m"] = f
                side[f"{wname}_over_floor"] = _ratio(a, f)
                # A "floor" above the amplitude it bounds is not a floor; it
                # means the high-pass leakage dominates and the reading names
                # nothing in either direction.  Say so in the artifact rather
                # than leaving the reader to divide.
                side[f"{wname}_resolved"] = bool(f > 0.0 and a > 2.0 * f)
                sl = slice(max(0, (a_sl.start or 0) - 1),
                           None if a_sl.stop is None else a_sl.stop - 1)
                aa, ff = amp_al[sl], floor_ser[sl]
                side[f"{wname}_resolved_sample_fraction"] = (
                    float(np.mean(aa > 2.0 * ff)) if aa.size else None)
            # Median as well as mean: the late-window ratio is noisy and a
            # mean alone reads more precise than the data supports.
            lh = amp[win["last_half"][0]]
            side["last_half_median_m"] = float(np.median(lh))
            for wname, (a_sl, _f) in win.items():
                side[f"{wname}_concentration"] = concentration(
                    alt, mask, area, a_sl)
            rec[sname] = side
            alt_cache[sname] = alt
        a, b = rec["legoESM"], rec["NEMO"]
        if single:
            rec["ratio_lego_over_nemo"] = {
                "withheld": "single-trajectory mode: both sides are the same "
                            "run, so every ratio would be 1.000 by "
                            "construction"}
            out_d["regions"][rname] = rec
            continue
        rec["ratio_lego_over_nemo"] = {
            "first8": _ratio(a["first8_mean_m"], b["first8_mean_m"]),
            "last_half": _ratio(a["last_half_mean_m"], b["last_half_mean_m"]),
            "last_half_median": _ratio(a["last_half_median_m"],
                                       b["last_half_median_m"]),
            "both_sides_resolved_last_half": bool(
                a["last_half_resolved"] and b["last_half_resolved"]),
        }
        rec["ratio_lego_over_nemo"]["last_half_estimators"] = ratio_estimators(
            np.asarray(a["amplitude_by_sample_m"]),
            np.asarray(b["amplitude_by_sample_m"]),
            alt_cache["legoESM"], alt_cache["NEMO"], mask, area,
            win["last_half"][0])
        rec["ratio_lego_over_nemo"]["last_half_bootstrap"] = (
            block_bootstrap_ratio_ci(
                np.asarray(a["amplitude_by_sample_m"])[win["last_half"][0]],
                np.asarray(b["amplitude_by_sample_m"])[win["last_half"][0]]))
        # Fit-derived ratios are published ONLY when BOTH fits are usable.
        # Publishing them unguarded produced "transient_A 35393" and a tau
        # ratio that divided a real fit by an untouched initial guess.
        if a["fit"].get("fit_usable") and b["fit"].get("fit_usable"):
            rec["ratio_lego_over_nemo"].update({
                "steady_C": _ratio(a["fit"]["steady_C_m"],
                                   b["fit"]["steady_C_m"]),
                "transient_A": _ratio(a["fit"]["transient_A_m"],
                                      b["fit"]["transient_A_m"]),
                "tau": _ratio(a["fit"]["tau_steps"], b["fit"]["tau_steps"]),
            })
        else:
            rec["ratio_lego_over_nemo"]["fit_ratios_withheld"] = (
                "at least one side's decay fit is not identifiable "
                "(A or tau within 3 standard errors of zero, or an "
                "exponential no better than a constant)")
        out_d["regions"][rname] = rec

    # The LOCUS as a function of time.  This, not the amplitude ratio, is the
    # statistic that separates "a wall MODE" from "a wall TRANSIENT": a mode
    # keeps its locus, a transient loses it.  Two readings are given because
    # they answer different questions -- the FIRST SAMPLE (the number the
    # campaign published) and the window's rms field (what the run carries on
    # average over that window).
    # BOTH partitions, because they answer different questions and the coarse
    # one hid the answer: "wall vs not" (continuity with the campaign's
    # published shares) and "which KIND of wall" (where the source question
    # is actually decided).
    out_d["wall_share"] = {}
    out_d["wall_share_by_direction"] = {}
    for sname, eta in sides.items():
        alt = ewt.two_dt_component(eta)
        drec: dict = {}
        sh0d = ewt.locus_shares(alt[0], walls, area)
        drec["first_sample"] = {k: v["share"] for k, v in sh0d.items()}
        drec["first_sample_enrichment"] = {
            k: v["enrichment"] for k, v in sh0d.items()}
        for wname, (a_sl, _f) in win.items():
            rmsd = np.sqrt((alt[a_sl] ** 2).mean(axis=0))
            shd = ewt.locus_shares(rmsd, walls, area)
            drec[wname] = {k: v["share"] for k, v in shd.items()}
            drec[wname + "_enrichment"] = {
                k: v["enrichment"] for k, v in shd.items()}
        out_d["wall_share_by_direction"][sname] = drec

        rec: dict = {}
        sh0 = ewt.locus_shares(alt[0], regions_only, area)
        rec["first_sample"] = {k: v["share"] for k, v in sh0.items()}
        rec["first_sample_enrichment"] = {
            k: v["enrichment"] for k, v in sh0.items()}
        for wname, (a_sl, _f) in win.items():
            rms_field = np.sqrt((alt[a_sl] ** 2).mean(axis=0))
            shw = ewt.locus_shares(rms_field, regions_only, area)
            rec[wname] = {k: v["share"] for k, v in shw.items()}
            rec[wname + "_enrichment"] = {
                k: v["enrichment"] for k, v in shw.items()}
        out_d["wall_share"][sname] = rec

    if self_check:
        out_d["self_check"] = _self_check(nemo["eta"], wet, regions, area)

    out_d = ewt.provenance({
        "probe": "eta_flicker_decay",
        "lane": lane,
        "nemo_npz": nemo_npz, "lego_npz": lego_npz,
        "nemo_free_npz": nemo_free_npz, "lego_free_npz": lego_free_npz,
        "mesh_mask": mesh_mask,
        "nemo_dtype_on_disk": nemo["dtype_on_disk"],
        "lego_dtype_on_disk": lego["dtype_on_disk"],
        **out_d})
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(out_d, fh, indent=2, allow_nan=False)
    return out_d


def _self_check(eta: np.ndarray, wet: np.ndarray, regions: dict,
                area: np.ndarray) -> dict:
    """Two controls that must both pass before any number above is quoted.

    LAND: poison every dry cell with an ALTERNATING huge value -- the only
    poison this probe's statistic can see -- and require every region
    amplitude to be bit-identical.
    PLANT: inject a known Nyquist oscillation of known amplitude into the wall
    band only; the wall amplitude must recover it and the interior must not.
    """
    # LAND.  Two ways this control was vacuous before review, both fixed here.
    #  (1) the shared poison is a CONSTANT, and the Nyquist operator
    #      annihilates constants exactly -- see
    #      ``plant_dry_violation(alternating=True)``;
    #  (2) the production ``area`` is ALREADY zeroed on dry cells, so land
    #      carried zero weight before the region mask was ever consulted and
    #      no change inside this file could turn the control red.  The
    #      control therefore runs against UNMASKED weights (raw cell area
    #      everywhere, land included), leaving the region mask as the only
    #      thing standing between land and the mean -- which is the claim.
    poisoned = ewt.plant_dry_violation(eta, wet, alternating=True)
    area_unmasked = np.where(area > 0.0, area, area[area > 0.0].mean())
    base = region_amplitude(ewt.two_dt_component(eta), regions["all"],
                            area_unmasked)
    pois = region_amplitude(ewt.two_dt_component(poisoned), regions["all"],
                            area_unmasked)
    land_ok = bool(np.array_equal(base, pois))

    # PLANT.  Two amplitudes, because they answer different questions: a large
    # plant verifies the operator's normalisation (that the reported number is
    # A and not 2A), and a plant AT THE MEASUREMENT SCALE, on top of the real
    # field, is the one that can fail informatively -- it asks whether the
    # instrument can still see a wall-band mode of the size actually being
    # reported, against the real background.
    def _plant(amp_planted: float) -> dict:
        sign = ((-1.0) ** np.arange(eta.shape[0]))[:, None, None]
        planted = eta + amp_planted * sign * regions["wall"][None, :, :]
        alt_p = ewt.two_dt_component(planted)
        alt_0 = ewt.two_dt_component(eta)
        wall_before = float(region_amplitude(alt_0, regions["wall"],
                                             area)[:8].mean())
        wall_after = float(region_amplitude(alt_p, regions["wall"],
                                            area)[:8].mean())
        int_before = float(region_amplitude(alt_0, regions["interior"],
                                            area)[:8].mean())
        int_after = float(region_amplitude(alt_p, regions["interior"],
                                           area)[:8].mean())
        # The plant adds in quadrature to whatever is already there, so the
        # recoverable quantity is the quadrature EXCESS, not the raw level.
        excess = float(np.sqrt(max(wall_after ** 2 - wall_before ** 2, 0.0)))
        return {"planted_m": amp_planted,
                "wall_before_m": wall_before, "wall_after_m": wall_after,
                "recovered_excess_m": excess,
                "recovery_ratio": _ratio(excess, amp_planted),
                "interior_before_m": int_before,
                "interior_after_m": int_after,
                # NOT a leakage test.  The wall and interior masks are
                # disjoint by construction, so this can only fail if the
                # partition itself is broken -- which is worth checking, but
                # it must not be labelled as proof that a wall signal does
                # not bleed into the interior statistic.  Named for what it
                # is.
                "interior_untouched_partition_disjoint": bool(
                    abs(int_after - int_before)
                    <= 1e-12 * max(int_before, 1e-30)),
                # The quadrature excess assumes the plant is INCOHERENT with
                # whatever alternating field is already there.  Against a
                # sign-coherent wall mode of the same amplitude -- exactly the
                # signature being hunted -- amplitudes add linearly and the
                # recovery reads ~1.73, which the naive gate would abort on.
                # So it is reported, and only the large incoherent plant is
                # gated.
                "sign_coherent_field_suspected": bool(
                    excess > 1.5 * amp_planted)}

    scale = float(region_amplitude(ewt.two_dt_component(eta),
                                   regions["wall"], area)[:8].mean())
    return {"land_poison_identical": land_ok,
            "plant_large": _plant(1e-3),
            "plant_at_measurement_scale": _plant(max(scale, 1e-12))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nemo", required=True)
    p.add_argument("--lego", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mesh-mask", default=ewt.MESH_MASK)
    p.add_argument("--nemo-free", default=None,
                   help="NEMO's own free run; with --lego-free this switches "
                        "the observable to each model MINUS ITS OWN free run, "
                        "which is the lane in which the two decay RATES are "
                        "comparable (common excitation)")
    p.add_argument("--lego-free", default=None)
    p.add_argument("--single", action="store_true",
                   help="both sides are the SAME trajectory on purpose (e.g. "
                        "a member-minus-member difference): report its locus "
                        "and amplitudes, withhold every cross-model ratio")
    a = p.parse_args()
    d = analyse(a.nemo, a.lego, a.out, a.mesh_mask,
                nemo_free_npz=a.nemo_free, lego_free_npz=a.lego_free,
                single=a.single)
    sc = d["self_check"]
    if not sc["land_poison_identical"]:
        raise SystemExit("SELF-CHECK FAILED: a statistic moved when land was "
                         "poisoned under UNMASKED weights -- it is averaging "
                         "land, not masking it")
    # Only the large plant is GATED: it is 3 orders above the field, so
    # coherence cannot bias it.  The measurement-scale plant is reported --
    # a recovery near 1.73 there is a sign-coherent pre-existing mode, i.e. a
    # FINDING, not an instrument failure.
    for key, lo, hi in (("plant_large", 0.95, 1.05),):
        r = sc[key]["recovery_ratio"]
        if r is None or not (lo <= r <= hi):
            raise SystemExit(
                f"SELF-CHECK FAILED ({key}): a planted wall-band Nyquist mode "
                f"of known amplitude came back at {r} of its true size")
        if not sc[key]["interior_untouched_partition_disjoint"]:
            raise SystemExit(f"SELF-CHECK FAILED ({key}): the wall and "
                             "interior masks are not disjoint")
    print(json.dumps({k: v for k, v in d.items() if k != "regions"},
                     indent=2, allow_nan=False))
    for rn, rec in d["regions"].items():
        r = rec["ratio_lego_over_nemo"]
        print(f"\n--- {rn}  ({rec['cells']} cells, "
              f"{rec['area_fraction']*100:.1f}% of wet area)")
        for sname in ("NEMO", "legoESM"):
            side = rec[sname]
            f = side["fit"]
            for w in ("first8", "last_half"):
                print(f"  {sname:9s} {w:10s} {side[w+'_mean_m']:.4e}  "
                      f"floor {side[w+'_floor_m']:.4e}  "
                      f"x{side[w+'_over_floor']:.2f}  "
                      f"resolved={side[w+'_resolved']}")
            print(f"            fit usable={f.get('fit_usable')}  "
                  f"C {f.get('steady_C_m', float('nan')):.4e}  "
                  f"A {f.get('transient_A_m', float('nan')):.4e}  "
                  f"tau {f.get('tau_steps', float('nan')):.2f}  "
                  f"r2 {f.get('r2', float('nan')):.3f}")
        print("  ratio lego/NEMO: " + "  ".join(
            f"{k}={v}" if not isinstance(v, float) else f"{k}={v:.3f}"
            for k, v in r.items()))
        if "withheld" in r:
            continue


if __name__ == "__main__":
    main()
