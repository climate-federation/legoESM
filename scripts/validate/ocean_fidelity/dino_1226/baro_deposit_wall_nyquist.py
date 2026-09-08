#!/usr/bin/env python
"""#1455 -- the barotropic per-step DEPOSIT, projected PER CELL onto the
two-step mode, resolved at the +/-69.5 deg end-wall rows, BOTH components.

WHAT THIS ANSWERS.  The wall-flicker hunt (513339cba) left the sustained
end-wall two-step excess (2.7x, CI [1.26, 5.10]) source-class but UNOWNED, and
named two live candidates: the barotropic split-explicit machinery, and the
explicit-versus-implicit wind-stress placement at a wind-driven wall row.  The
one-step momentum budget could not reach either -- it compares NEMO's explicit
per-term trends, and the barotropic solve is not among them.  This probe
reaches them from the other side: it takes the campaign's committed barotropic
per-step DEPOSIT instrument, which already compares legoESM's substep loop
against NEMO's own substep dumps at a matched state, and asks which of its
terms deposits ALTERNATING-SIGN signal on the end-wall rows.

BOTH COMPONENTS, and why that is not optional.  The first version of this probe
measured the ZONAL velocity only, and both adversarial reviews returned NO-SHIP
for the same reason: the scored walls are ZONAL walls (land to the north and
south), so the wall-NORMAL component there is the MERIDIONAL one, and it is the
convergence of the wall-normal transport that sets the sea surface against a
closed wall.  u is the TANGENTIAL component at those rows.  A u-only null
cannot exclude the barotropic solve at a zonal wall.  The meridional side is
now built by the same instrument, from inputs that were already on disk, and
validated against NEMO's own final meridional field and legoESM's own running
sum exactly as the zonal side is.  Both components are reported.

THE THREE TERMS, per component, and how they map onto the two candidates:

  TOTAL    lego's boxcar-averaged barotropic velocity minus NEMO's, per cell.
  IN-LOOP  the same with NEMO's own frozen slow forcing substituted into
           lego's loop, so it isolates the SUBSTEP MACHINERY -- candidate (a).
  FORCING  TOTAL minus IN-LOOP: what lego's own slow barotropic forcing does
           that NEMO's ``zu_frc``/``zv_frc`` does not, propagated through the
           loop.  The surface wind stress enters here, so this is where a
           placement difference at a wind-driven wall row lands -- candidate
           (b).

FORCING has no NEMO counterpart by construction: both of its sides are lego
(the same loop driven by two different forcings).  It is scored on its
DIFFERENCE only, its side columns read n/a, and it is a RESIDUAL CATCH-ALL --
every difference that is not the loop machinery lands in it, so it bounds
rather than identifies.

TIME-LEVEL MATCHING.  The zonal-wall momentum budget found exact 2.000
definitional offsets on every term whose time level was not verified matched.
The deposit does not have that failure mode, and the reason is checkable rather
than asserted: both sides ENTER the step from the same bytes, and each of the
four reconstructions is pinned against a field neither this probe nor the
instrument computed -- NEMO's two sides against ``spg_dump_puu_b_final`` and
``spg_dump_pvv_b_final`` at 1e-10, legoESM's against the model's own running
sums at 1e-12, including the substituted run's (that last one was missing until
review 2026-08-26 pointed out it was the one reconstruction pinned by nothing).
All controls re-run inside every child; none is inherited.

WHAT IS EXCLUDED, deliberately.  NEMO's surface-pressure-gradient trend.  It is
downstream of the sea surface, so a two-step surface mode produces a two-step
pressure gradient on BOTH sides by definition; it would reproduce the ratio
already in hand and name nobody.

=====================================================================
THE STATE COUNT IS FIVE, NOT NINE -- a correction to this probe's own brief,
recorded before any number was produced.
=====================================================================
The brief asked for NINE consecutive steps.  The deposit instrument needs
NEMO's INTERNAL substep dumps (``substep_dump.bin``, ``spg_dump_zu_frc.bin``,
``spg_dump_un_adv_final.bin``, ``spg_dump_puu_b_final.bin`` and three more),
which exist once per seqdump lane, i.e. once per NEMO step.  On disk those
lanes cover kt = 5760 (RUN_SEQDUMP_D180_1R) and kt = 5761..5764
(RUN_SEQDUMP_KT<kt>_1R) and stop there.  FIVE consecutive states, verified by
file existence rather than by the directory listing's shape.

Five is the minimum this measurement is defined on: the two-step operator drops
the first and last sample, so five states give THREE Nyquist samples per cell,
and the leakage floor drops two more, giving ONE floor sample.  Extending to
nine is a NEMO-side job costed in the report, not something this probe can do
from disk.

=====================================================================
PRE-REGISTERED CRITERIA
=====================================================================
Version 3, after TWO rounds of dual adversarial review (four reviews, all four
NO-SHIP).  Every change below is a RETRACTION of something in an earlier
version, made because a review refuted it, and each is named rather than
silently swapped.  No bar was ever moved to fit a number: every ratio is
re-reported under three spatial reductions, with both its factors, and with
its operands' own leakage floors, so a reader can see what any choice would
have given.

The short history, because it is the useful part: v1 measured one velocity
component at a wall where the other one is normal, and called it a null.  v2
added the missing component and named a term -- on a post-hoc statistic that
was a null divided by a favourable constant, and on a substitution whose
apparent 31000x effect was the projector annihilating a state-constant field.
v3 restores the comparable the target interval actually belongs to and reports
the state-constancy that makes this whole measurement class blind here.

Statistic: for a per-cell field f(state, cell), the two-step amplitude is
``per_cell_nyquist`` -- the alternating component taken PER CELL across states,
then RMS'd over states, then aggregated with area weights.  The order is
load-bearing: the previous budget took a band mean FIRST and review measured
the resulting suppression of a spatially signed signal at 44x to 80000x.

  S1  FLOOR       amp(diff, wall rows) against its own window-matched leakage
                  floor.  ONE floor sample at five states; reported, never
                  gated.
  S2  ENRICHMENT  E = amp(diff, wall rows) / amp(diff, interior), reported per
                  row and per band WITH an interval and WITH the sub-window
                  spread.  RETRACTED FROM VERSION 1: the bar of 2.0 was
                  TRANSFERRED, not argued -- 2.68/2.85 is a legoESM/NEMO ratio
                  at the wall, whereas E is a wall/interior concentration of a
                  DIFFERENCE field.  Different statistic, different denominator.
                  E is therefore reported WITHOUT a pass/fail.
  S3  GEOMETRY    whether the same reading holds at BOTH j=1 and j=197.  The
                  measured excess is a both-end-walls object.
  S4  SIDE        amp(lego)/amp(nemo) at the wall rows.  THE COMPARABLE, and
                  RESTORED to that role in version 3.  The measured excess
                  (2.7x, CI [1.26, 5.10]) is a legoESM/NEMO amplitude ratio ON
                  THE WALL ROWS with no interior denominator, so this is the
                  only statistic here that can be compared to it without
                  changing units of meaning.  Version 2 retired it using
                  reasons that belong to OTHER terms -- "unfalsifiable" is true
                  only of IN-LOOP, where the two fields agree by construction,
                  and "a magnitude test on a phase disagreement" is a reason
                  the DIFFERENCE field misleads, not a reason two independent
                  amplitudes cannot be compared.  A ratio is only reported as
                  meaningful when BOTH operands clear their own leakage floor
                  by 2x; rows failing that are printed with ok=NO.
  S6  DOUBLE      D = [amp(lego,wall)/amp(lego,interior)] /
      RATIO       [amp(nemo,wall)/amp(nemo,interior)].  VERSION 2 MADE THIS
                  "THE REGISTERED COMPARABLE" AND THAT IS RETRACTED.  D
                  factorises EXACTLY as S4(wall) / S4(interior), and on this
                  data the interior factor is 0.58 in BOTH components -- a
                  basin-wide fact (legoESM alternates ~1.7x less than NEMO
                  everywhere) with nothing to do with walls.  Dividing by it
                  lifts every row by 1.7x, which is what made a null look like
                  a result.  Worse, its numerator IS S4, the statistic v2
                  retired.  D is still reported, with BOTH factors printed
                  beside it and under three spatial reductions, but it is NOT
                  compared to the target interval: it is a different quantity.
  S7  PHASE       ``two_dt_phase_agreement`` per band.  An enrichment of a
                  DIFFERENCE field only reads as "legoESM's excess" when the
                  two sides are in phase; when they are not, the difference
                  measures decorrelation and the enrichment answers a different
                  question.  Reported next to every enrichment.
  S5  MAGNITUDE   reported, NOT gated: the deposit as a per-step sea-surface
                  displacement, evaluated with the LOCAL metrics of the row it
                  names (version 1 used basin medians against a wall-row
                  amplitude, a mixed-location comparison review caught).

POWER.  Version 1 claimed "enrichment is a ratio so the sample count cancels".
THAT IS RETRACTED: it is a bias argument asserted as a variance argument, and
review refuted it from this probe's own data -- the three-state sub-windows
give enrichments spanning a factor of 6 at one row.  What cancels in a ratio is
a common multiplicative normalisation, not noise across two disjoint bands.
Every enrichment therefore carries BOTH a cell bootstrap interval AND the
sub-window spread.  The bootstrap resamples cells independently while wall
cells are spatially correlated, so it is the OPTIMISTIC of the two and the
sub-window spread is the one to believe.

WHAT THIS MEASUREMENT CANNOT SEE.  This is no longer a caveat -- ON THIS DATA
IT IS THE DOMINANT TERM, and reading it as a footnote is what produced version
2's false attribution.

(i) A STATE-CONSTANT residual.  The two-step operator measures state-to-state
variability and annihilates a state-constant field EXACTLY (proved by this
probe's own synthetic control: constant -> 0.0).  Measured here: the end-wall
residual is 99.9% state-constant BY RMS SHARE -- which, because the share is a
ratio of amplitudes, still leaves sqrt(1-share^2) = 4.4-4.7% of the RMS
genuinely varying.  That varying part IS what the projection reports (it
recovers 74-87% of it).  So the correct scoping, and v3 got this wrong by one
word: the projection is blind to the constant ~95%, NOT to everything, and the
null it returns is a null on the varying ~4.5%.  Any decomposition into
"state-independent part" and "the rest" still assigns the whole PROJECTED
signal to "the rest" by construction, which is how v2 read a state-constant
loop bias as an exoneration.  ``state_independence`` is computed for every term
and reports the constant share, the VARYING share and the varying rms side by
side so neither error can recur.

(ii) A source born at one substep and cancelled by another inside the SAME
averaging window.  MEASURED from the saved weights rather than quoted:
attenuation 45x, i.e. suppressed, not annihilated.  A much weaker limit than
(i).

(iii) AMPLIFICATION.  Every state is independently re-bridged, so this
instrument sees INJECTION from a clean state only.

This probe PRINTS NUMBERS AND THE PRE-REGISTERED THRESHOLDS SIDE BY SIDE.  It
does not print a verdict.

USAGE
  # 1. produce the five per-cell deposit maps (~2 min each, GPU 0)
  CUDA_VISIBLE_DEVICES=0 LEGOESM_NEMO_E3T=both JAX_ENABLE_X64=1 \
    .venv/bin/python .../baro_deposit_time_walk.py --consecutive \
      --deposit-map-dir results/dino_1455/maps_consecutive
  # 2. project them
  JAX_ENABLE_X64=1 .venv/bin/python .../baro_deposit_wall_nyquist.py \
      --map-dir results/dino_1455/maps_consecutive --out results/.../x.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import eta_wave_twin as ewt  # noqa: E402
import zonal_wall_momentum_budget as zwb  # noqa: E402

#: The consecutive NEMO steps whose seqdump lanes exist on disk.  A literal
#: list rather than a range, because a missing lane must be a FATAL and not a
#: silently shorter series.
CONSECUTIVE_KTS = (5760, 5761, 5762, 5763, 5764)

#: The baroclinic step, from the DINO namelist (``rn_Dt = 2700.``).  Named
#: rather than inlined: a lane at a different step would otherwise silently
#: produce a wrong sea-surface conversion in S5.
DT_S = 2700.0

#: The measured end-wall excess this work is trying to explain, and its 90%
#: interval.  It is a legoESM/NEMO amplitude ratio ON THE WALL ROWS with no
#: interior denominator, so S4 -- and ONLY S4 -- is comparable to it.  Neither
#: S2 nor S6 is; both carry an interior denominator the target does not have.
MEASURED_EXCESS = 2.7
MEASURED_EXCESS_CI = (1.26, 5.10)

#: The sea-surface two-step amplitude measured at those rows, for S5 only.
MEASURED_FLICKER_M = 1.2e-6

#: A row smaller than this cannot support a per-row ratio; it stays in the
#: aggregate band.  DINO's zonal-wall band is two ~47-cell end walls plus two
#: two-or-three-cell interior shelf edges.
MIN_ROW_CELLS = 10

#: Terms as (label, lego key template, nemo key template, diff key template).
#: FORCING is derived, so its templates are None.
TERM_TEMPLATE = (
    ("TOTAL", "lego_{C}bar_avg", "nemo_{C}bar_avg", "d{C}_avg"),
    ("IN_LOOP", "lego_{C}bar_avg_sub", "nemo_{C}bar_avg", "d{C}_sub"),
    ("FORCING", None, None, None),
)

#: (label, letter, weight key, mask key, what the component IS at a zonal wall)
COMPONENTS = (
    ("u", "U", "acc_w", "wetu", "tangential"),
    ("v", "V", "acc_w_v", "wetv", "WALL-NORMAL"),
)


def load_maps(map_dir: str, kts=CONSECUTIVE_KTS) -> dict:
    """Load the per-cell deposit maps, refusing any hole in the series.

    A dropped state does not merely weaken a two-step projection, it CHANGES
    THE MODE: the operator's alternation is indexed by position in the series,
    so a gap re-phases every sample after it.  Every miss is fatal.
    """
    keys = ("dU_avg", "dU_sub", "acc_w", "wetu", "lego_Ubar_avg",
            "nemo_Ubar_avg", "lego_Ubar_avg_sub",
            "dV_avg", "dV_sub", "acc_w_v", "wetv", "lego_Vbar_avg",
            "nemo_Vbar_avg", "lego_Vbar_avg_sub", "wgt_primary")
    fields, stamps = {}, []
    for kt in kts:
        path = os.path.join(map_dir, f"deposit_map_kt{kt}.npz")
        if not os.path.exists(path):
            raise SystemExit(
                f"FATAL: no deposit map at {path}. Produce the series with "
                "baro_deposit_time_walk.py --consecutive --deposit-map-dir; a "
                "missing state must never become a dropped sample, because "
                "the two-step operator is indexed by POSITION and a gap "
                "re-phases every sample after it.")
        z = np.load(path, allow_pickle=False)
        got = int(z["ic_step"])
        if got != kt:
            raise SystemExit(
                f"FATAL: {path} carries ic_step={got}, expected {kt}. The "
                "series would not be consecutive.")
        for key in keys:
            if key not in z:
                raise SystemExit(
                    f"FATAL: {path} has no {key!r}. It predates the "
                    "two-component deposit map; regenerate the maps rather "
                    "than projecting one velocity component at a zonal wall.")
            fields.setdefault(key, []).append(np.asarray(z[key],
                                                         dtype=np.float64))
        stamps.append({"kt": kt, "path": path,
                       "seqdump": str(z["seqdump"]),
                       "provenance": str(z["provenance"])})
    out = {k: np.stack(v, axis=0) for k, v in fields.items()}
    for key in ("acc_w", "wetu", "acc_w_v", "wetv", "wgt_primary"):
        a = out[key]
        if not np.array_equal(a, np.broadcast_to(a[0], a.shape)):
            raise SystemExit(
                f"FATAL: {key} is not identical across the five states; the "
                "projection would compare different cells at different times.")
    out["_static"] = {"acc_w": out["acc_w"][0], "wetu": out["wetu"][0] > 0.5,
                      "acc_w_v": out["acc_w_v"][0],
                      "wetv": out["wetv"][0] > 0.5,
                      "wgt_primary": out["wgt_primary"][0]}
    out["_stamps"] = stamps
    out["dU_forcing"] = out["dU_avg"] - out["dU_sub"]
    out["dV_forcing"] = out["dV_avg"] - out["dV_sub"]
    return out


def window_attenuation(w: np.ndarray) -> dict:
    """How much this averaging window attenuates a SUBSTEP-alternating source.

    MEASURED from legoESM's own saved primary weights rather than quoted: the
    transfer of a (-1)^n substep sequence through the window is sum(w*(-1)^n).
    This is the number that says what the null does NOT exclude, so it must not
    be a literal carried over from a review comment.
    """
    n = w.shape[0]
    alt = ((-1.0) ** np.arange(n))
    transfer = float(abs(w @ alt))
    nz = np.nonzero(w)[0]
    return {"n_substeps": int(n),
            "n_nonzero": int(nz.size),
            "first_nonzero_substep": int(nz[0] + 1) if nz.size else -1,
            "substep_nyquist_transfer": transfer,
            "attenuation_factor": (1.0 / transfer) if transfer > 0
            else float("inf")}


def build_bands(wet: np.ndarray) -> dict:
    """Zonal-wall / meridional-wall / interior bands on a velocity-face mask.

    ``wall_direction_partition`` classifies by which neighbour is DRY: a cell
    with a dry north or south neighbour and no dry east/west neighbour is a
    ZONAL wall cell -- the first wet row against a closed end wall.  DINO is
    zonally periodic, so the east-west test wraps.  Cells that are both fall
    into ``wall_corner`` and are scored in neither band.

    DEVIATION, declared because the campaign has been burned by conflating
    staggerings: the sibling budget builds its bands on the T mask and maps
    them onto faces with ``to_u_points``.  This probe partitions the FACE mask
    directly, which drops the four corner faces where an end wall meets a
    sidewall (104 -> 100 faces on the u grid).  Those are the most singular
    cells in the domain and they are scored in neither band; the alternative
    would score them in both.  Stated rather than hidden, and the row stamps
    make the difference visible.
    """
    w = ewt.wall_direction_partition(wet, periodic_i=True)
    bands = {"zonal_wall": w["zonal_wall"],
             "meridional_wall": w["meridional_wall"],
             "interior": w["off_wall"]}
    for name, b in bands.items():
        if not b.any():
            raise SystemExit(f"FATAL: band {name!r} is empty")
    names = list(bands)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if (bands[a] & bands[b]).any():
                raise SystemExit(f"FATAL: bands {a} and {b} overlap")
    return bands


def cell_amplitudes(series: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-cell two-step amplitude over a band, BEFORE any spatial reduction.

    This is the inner half of ``per_cell_nyquist``, exposed so a bootstrap can
    resample cells.  ``synthetic_checks`` asserts that aggregating this
    reproduces the committed estimator bit-for-bit, so the probe still has ONE
    estimator rather than two that could drift apart.
    """
    if series.shape[0] < 3:
        raise SystemExit("cell_amplitudes needs at least 3 states")
    alt = ewt.two_dt_component(series[:, mask].astype(np.float64))
    return np.sqrt((alt ** 2).mean(axis=0))


def aggregate(amp: np.ndarray, weights: np.ndarray) -> float:
    """Area-weighted rms of a per-cell amplitude vector."""
    tot = float(weights.sum())
    if tot <= 0.0:
        raise SystemExit("aggregate: band carries no weight")
    return float(np.sqrt(float((weights * amp ** 2).sum()) / tot))


def amp_on(series: np.ndarray, mask: np.ndarray,
           weights: np.ndarray) -> float:
    """Two-step amplitude of a (n_states, ny, nx) series over one 2-D band.

    Delegates to the committed ``per_cell_nyquist`` so this probe and the
    zonal-wall budget are decided by ONE estimator; re-deriving it here is
    exactly how two probes come to disagree about the same field.
    """
    if not mask.any():
        raise SystemExit("amp_on: empty mask")
    return zwb.per_cell_nyquist(series[:, mask], weights[mask])


def ratio_or_refuse(num: float, den: float, what: str) -> float:
    """A ratio whose denominator is structurally zero is an ERROR, not 1e300.

    Version 1 guarded these with ``max(den, 1e-300)``, which turns a dead band
    into a spectacular enrichment instead of a refusal -- a guard that accepts
    the wrong arm.
    """
    if not np.isfinite(den) or den <= 0.0:
        raise SystemExit(
            f"FATAL: {what} has a non-positive denominator ({den!r}); the "
            "band is dead and no ratio on it is meaningful.")
    return float(num) / float(den)


def full_window_floor(field: np.ndarray, wet: np.ndarray,
                      mask: np.ndarray, weights: np.ndarray) -> float:
    """The leakage floor over ALL usable samples, with no window to mis-set.

    Round-3 review mutated the ``window=`` argument at both call sites and the
    suite stayed green.  It was not a test gap: at five states the usable
    series is ONE sample, so every window selects it and the parameter is a
    no-op by construction.  Rather than test a parameter that cannot vary here
    and would silently become live at seven states, the parameter is removed --
    this asks the helper for the per-sample series and averages all of it, so
    "which window" is no longer a choice anyone can get wrong.
    """
    per = np.atleast_1d(ewt.two_dt_leakage_floor(
        field, wet, area=np.where(weights > 0, weights, 0.0), mask=mask,
        return_series=True))
    if per.size == 0:
        raise SystemExit("full_window_floor: no usable floor sample")
    return float(np.mean(per))


def floor_ratio(amp: float, floor: float) -> float:
    """amp/floor, tolerating a floor of exactly zero.

    Unlike an enrichment denominator, a zero LEAKAGE FLOOR is a legitimate
    state and not a dead band: it means the field carries no slow curvature for
    the operator to leak.  With a zero amplitude too the term is simply absent
    (0.0); with a nonzero amplitude it is infinitely above its floor.  Refusing
    here would abort a whole component because one term reproduced the oracle
    exactly, which is the outcome the probe most wants to be able to report.
    """
    if floor > 0.0:
        return float(amp) / float(floor)
    return 0.0 if amp <= 0.0 else float("inf")


def enrichment_interval(series: np.ndarray, row_mask: np.ndarray,
                        int_mask: np.ndarray, weights: np.ndarray,
                        n_boot: int = 2000, seed: int = 0) -> dict:
    """A cell bootstrap AND the sub-window spread for one enrichment.

    Version 1 quoted a bare point estimate against an interval-bearing target,
    on the argument that a ratio cancels the sample count.  That argument is
    retracted (see the module docstring), so both intervals are computed here.
    """
    rng = np.random.default_rng(seed)
    a_row = cell_amplitudes(series, row_mask)
    a_int = cell_amplitudes(series, int_mask)
    w_row, w_int = weights[row_mask], weights[int_mask]
    point = ratio_or_refuse(aggregate(a_row, w_row),
                            aggregate(a_int, w_int), "enrichment")
    draws = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, a_row.size, a_row.size)
        j = rng.integers(0, a_int.size, a_int.size)
        draws[b] = (aggregate(a_row[i], w_row[i])
                    / aggregate(a_int[j], w_int[j]))
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    sub = [ratio_or_refuse(amp_on(series[s:s + 3], row_mask, weights),
                           amp_on(series[s:s + 3], int_mask, weights),
                           "sub-window enrichment")
           for s in range(series.shape[0] - 2)]
    return {"point": point, "boot_lo": lo, "boot_hi": hi,
            "subwindow": sub, "subwindow_lo": float(min(sub)),
            "subwindow_hi": float(max(sub)),
            "subwindow_spread": ratio_or_refuse(max(sub), min(sub),
                                                "sub-window spread")}


def side_ratio_subwindows(lego: np.ndarray, nemo: np.ndarray,
                          row_mask: np.ndarray,
                          weights: np.ndarray) -> list:
    """S4 per three-state sub-window.

    Round-3 review: the only evidence dismissing the one row where the restored
    comparable exceeds 1 was a sub-window sequence taken from S6 -- the
    statistic v3 says is a different quantity.  A caveat on S4 has to be made
    of S4.  Note the windows OVERLAP (they share two states each), so a
    monotone sequence of three is roughly a one-in-three event and is a weak
    signal, not a demonstration.
    """
    return [ratio_or_refuse(amp_on(lego[s:s + 3], row_mask, weights),
                            amp_on(nemo[s:s + 3], row_mask, weights),
                            "sub-window side ratio")
            for s in range(lego.shape[0] - 2)]


def double_ratio_interval(lego: np.ndarray, nemo: np.ndarray,
                          row_mask: np.ndarray, int_mask: np.ndarray,
                          weights: np.ndarray, n_boot: int = 2000,
                          seed: int = 0) -> dict:
    """Interval for S6, the statistic that carries the reading.

    Version 2 moved the comparable from S2 to S6 after review; a bare point
    estimate for S6 would repeat exactly the defect that got S2's bar
    retracted -- comparing a number with no interval against a target that has
    one.  Both the cell bootstrap and the three-state sub-window spread are
    computed, and the sub-window spread is the one to believe.
    """
    rng = np.random.default_rng(seed)
    lr, li = cell_amplitudes(lego, row_mask), cell_amplitudes(lego, int_mask)
    nr, ni = cell_amplitudes(nemo, row_mask), cell_amplitudes(nemo, int_mask)
    w_row, w_int = weights[row_mask], weights[int_mask]

    def _d(lr_, li_, nr_, ni_, wr, wi):
        return ((aggregate(lr_, wr) / aggregate(li_, wi))
                / (aggregate(nr_, wr) / aggregate(ni_, wi)))

    point = _d(lr, li, nr, ni, w_row, w_int)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, lr.size, lr.size)
        j = rng.integers(0, li.size, li.size)
        draws[b] = _d(lr[i], li[j], nr[i], ni[j], w_row[i], w_int[j])
    lo, hi = (float(x) for x in np.percentile(draws, [2.5, 97.5]))
    sub = []
    for s in range(lego.shape[0] - 2):
        a, b2 = lego[s:s + 3], nemo[s:s + 3]
        sub.append((amp_on(a, row_mask, weights)
                    / amp_on(a, int_mask, weights))
                   / (amp_on(b2, row_mask, weights)
                      / amp_on(b2, int_mask, weights)))
    return {"point": point, "boot_lo": lo, "boot_hi": hi,
            "subwindow": [float(x) for x in sub],
            "subwindow_lo": float(min(sub)), "subwindow_hi": float(max(sub))}


def state_independence(series: np.ndarray, mask: np.ndarray,
                       weights: np.ndarray) -> dict:
    """Is this residual the SAME FIELD at every state?

    THE MOST IMPORTANT DIAGNOSTIC IN THIS PROBE, added after round-2 review
    2026-08-26 showed the absence of it had produced a false attribution.

    The two-step operator measures state-to-state VARIABILITY and annihilates a
    state-constant field exactly.  So a residual that is the same field at
    every state projects to ~zero whatever its size, and any decomposition of a
    residual into "state-independent part" and "the rest" hands the entire
    projected signal to "the rest" BY CONSTRUCTION, independent of the physics.
    Reading that as an attribution is a mistake this probe made and this
    function exists to make impossible: the UNPROJECTED amplitude and the
    state-independence are reported next to every projected number.
    """
    x = series[:, mask]
    mean = x.mean(axis=0)
    cors = []
    for a in range(x.shape[0]):
        for b in range(a + 1, x.shape[0]):
            sa, sb = x[a], x[b]
            if sa.std() <= 0.0 or sb.std() <= 0.0:
                continue
            cors.append(float(np.corrcoef(sa, sb)[0, 1]))
    w = weights[mask]
    tot = float(w.sum())
    unproj = float(np.sqrt(float((w * (x ** 2).mean(axis=0)).sum()) / tot))
    const = float(np.sqrt(float((w * mean ** 2).sum()) / tot))
    share = (const / unproj) if unproj > 0 else 0.0
    # The share is a ratio of RMS AMPLITUDES, so a 99.9% constant share still
    # leaves sqrt(1 - share^2) of the RMS genuinely varying -- 4.5%, not 0.1%.
    # Round-3 review caught v3 quoting the share as if it were the invisible
    # fraction; the varying part is what the projector actually reports.
    return {"unprojected_rms": unproj,
            "state_constant_rms": const,
            "state_constant_share": share,
            "state_varying_share_rms": float(
                np.sqrt(max(1.0 - min(share, 1.0) ** 2, 0.0))),
            "state_varying_rms": float(
                unproj * np.sqrt(max(1.0 - min(share, 1.0) ** 2, 0.0))),
            # SIGNED, deliberately: an anti-correlated pair is the two-step
            # mode itself and must never be reported as state-constant.
            "cross_state_corr_min": min(cors) if cors else float("nan"),
            "cross_state_corr_max": max(cors) if cors else float("nan")}


def reduction_sensitivity(lego: np.ndarray, nemo: np.ndarray,
                          row_mask: np.ndarray, int_mask: np.ndarray,
                          weights: np.ndarray) -> dict:
    """The double ratio under THREE spatial reductions, not one.

    Round-1 review found the interior denominator is a hot-cell statistic (the
    top 1% of cells carry ~98% of the alternating power, effective sample ~15
    cells of 9444).  Round 2 showed the headline moves by 2.6x on the choice of
    reduction alone.  A number that is not stable across defensible reductions
    is not a result, so all three are reported and the RANGE is the number.
    """
    lr, li = cell_amplitudes(lego, row_mask), cell_amplitudes(lego, int_mask)
    nr, ni = cell_amplitudes(nemo, row_mask), cell_amplitudes(nemo, int_mask)
    w_row, w_int = weights[row_mask], weights[int_mask]

    def _trim(a):
        lo, hi = np.percentile(a, [10.0, 90.0])
        sel = (a >= lo) & (a <= hi)
        return a[sel] if sel.any() else a

    out = {}
    out["area_weighted_rms"] = ((aggregate(lr, w_row) / aggregate(li, w_int))
                               / (aggregate(nr, w_row) / aggregate(ni, w_int)))
    out["median"] = ((np.median(lr) / np.median(li))
                     / (np.median(nr) / np.median(ni)))
    out["trimmed_10_90"] = (
        (float(np.mean(_trim(lr))) / float(np.mean(_trim(li))))
        / (float(np.mean(_trim(nr))) / float(np.mean(_trim(ni)))))
    vals = [float(v) for v in out.values()]
    out = {k: float(v) for k, v in out.items()}
    out["range_lo"], out["range_hi"] = min(vals), max(vals)
    out["crosses_one"] = bool(min(vals) < 1.0 <= max(vals)
                              or max(vals) < 1.0)
    # The effective number of interior cells actually deciding the
    # denominator, FOR EACH SIDE.  The area-weighted rms is a power mean, so a
    # handful of hot cells can carry it -- on the real data the top 1% of
    # interior cells hold ~98% of the alternating power.  Both sides are
    # reported because a hot cell on either one moves the ratio.
    def _pr(a):
        q = a ** 2
        return float((q.sum() ** 2) / max(float((q ** 2).sum()), 1e-300))

    out["interior_participation_ratio"] = _pr(li)
    out["interior_participation_ratio_nemo"] = _pr(ni)
    out["interior_participation_ratio_min"] = min(_pr(li), _pr(ni))
    out["interior_cells"] = int(li.size)
    return out


def synthetic_checks(mask: np.ndarray, weights: np.ndarray,
                     n_states: int) -> dict:
    """Prove the projector on fields whose answer is known in advance.

    A metric is not trusted because of what it is called.  The alternating
    case pins the normalisation (the operator is written to return A, not 2A),
    and the constant and ramp cases pin that a SLOW field is not reported as a
    mode.
    """
    shape = (n_states,) + mask.shape
    amp_a = 3.7e-4
    sign = ((-1.0) ** np.arange(n_states))[:, None, None]
    alt = np.ascontiguousarray(np.broadcast_to(sign, shape) * amp_a)
    const = np.full(shape, 5.0)
    ramp = (np.arange(n_states, dtype=np.float64)[:, None, None]
            * np.ones(shape))
    got = {"alternating": amp_on(alt, mask, weights),
           "constant": amp_on(const, mask, weights),
           "ramp": amp_on(ramp, mask, weights),
           "alternating_expected": amp_a}
    got["cell_agg_matches_committed"] = bool(
        aggregate(cell_amplitudes(alt, mask), weights[mask])
        == got["alternating"])
    if not got["cell_agg_matches_committed"]:
        raise SystemExit(
            "SELF-CHECK FAILED: the per-cell helper does not reproduce "
            "per_cell_nyquist; the bootstrap and the table would be computed "
            "by two different estimators.")
    if abs(got["alternating"] - amp_a) > 1e-12 * amp_a:
        raise SystemExit(
            "SELF-CHECK FAILED: a pure alternating field of amplitude "
            f"{amp_a:.3e} projected to {got['alternating']:.6e}; the "
            "projector's normalisation is not what the criteria assume.")
    for key in ("constant", "ramp"):
        if got[key] > 1e-12:
            raise SystemExit(
                f"SELF-CHECK FAILED: a {key} field projected to {got[key]:.3e} "
                "instead of zero; the operator is passing the slow field.")
    return got


def land_poison_check(series: np.ndarray, mask: np.ndarray,
                      weights: np.ndarray, wet: np.ndarray) -> dict:
    """ALTERNATING land poison, with an honest account of which arm is live.

    The poison must ALTERNATE over states: a constant poison is annihilated
    exactly by the two-step operator, which made an earlier control on this
    branch vacuous by construction.

    WHICH ARM PROVES WHAT, corrected after review 2026-08-26.  The production
    arm is a STRUCTURAL INVARIANT, not a test: the band comes from
    ``wall_direction_partition``, which intersects with the wet mask, so
    poisoned dry cells can never enter it whatever the weights do.  It is kept
    because it would catch a band built the wrong way, and it is labelled as an
    invariant rather than presented as a live control.  The UNMASKED arm is the
    live one: it opens the band to land and requires the poison to move the
    statistic, which is what makes the masking claim non-vacuous.
    """
    poisoned = ewt.plant_dry_violation(series, wet, alternating=True)
    base = amp_on(series, mask, weights)
    prod = amp_on(poisoned, mask, weights)
    unmasked = np.where(weights > 0, weights, np.median(weights[weights > 0]))
    open_mask = mask | (~wet)
    base_u = amp_on(series, open_mask, unmasked)
    pois_u = amp_on(poisoned, open_mask, unmasked)
    out = {"production_base": base, "production_poisoned": prod,
           "production_identical": bool(prod == base),
           "production_arm_is_structural_invariant": True,
           "unmasked_base": base_u, "unmasked_poisoned": pois_u,
           "unmasked_moves": bool(pois_u > 10.0 * max(base_u, 1e-300))}
    if not out["production_identical"]:
        raise SystemExit(
            "SELF-CHECK FAILED: poisoning land moved a production statistic "
            f"({base:.6e} -> {prod:.6e}); the band is not a subset of the wet "
            "mask, which it is built to be.")
    if not out["unmasked_moves"]:
        raise SystemExit(
            "SELF-CHECK FAILED: the land control is VACUOUS -- the alternating "
            f"poison moved the unmasked statistic {base_u:.3e} -> {pois_u:.3e}, "
            "so passing the masked arm proves nothing.")
    return out


def weights_load_bearing(series: np.ndarray, mask: np.ndarray,
                         weights: np.ndarray) -> dict:
    """The area weights must change the answer, or the label is decoration."""
    uniform = np.ones_like(weights)
    w_amp = amp_on(series, mask, weights)
    u_amp = amp_on(series, mask, uniform)
    rel = abs(w_amp - u_amp) / max(abs(w_amp), 1e-300)
    pos = weights[mask][weights[mask] > 0]
    out = {"weighted": w_amp, "uniform": u_amp, "rel_change": rel,
           "load_bearing": bool(rel > 1e-6),
           "weight_spread": float(weights[mask].max() / max(pos.min(), 1e-300))}
    if not out["load_bearing"]:
        raise SystemExit(
            "SELF-CHECK FAILED: uniform weights reproduce the area-weighted "
            f"answer to {rel:.3e}; the weights are inert and every "
            "'area-weighted' number here would be mislabelled.")
    return out


def zonal_wall_rows(bands: dict) -> list:
    """The rows the zonal-wall band occupies, STAMPED not assumed.

    The prose says j=1 and j=197.  Prose is a pointer; this reads the mask.
    """
    return [{"j": int(j), "cells": int(bands["zonal_wall"][j].sum())}
            for j in np.unique(np.nonzero(bands["zonal_wall"])[0])]


def project_component(maps: dict, letter: str, wkey: str, mkey: str) -> dict:
    """The per-term, per-band, per-row table for ONE velocity component."""
    weights = maps["_static"][wkey]
    wet = maps["_static"][mkey]
    bands = build_bands(wet)

    row_masks = {}
    for r in zonal_wall_rows(bands):
        if r["cells"] < MIN_ROW_CELLS:
            continue
        m = np.zeros_like(bands["zonal_wall"])
        m[r["j"]] = bands["zonal_wall"][r["j"]]
        row_masks[f"j={r['j']}"] = m
    if not row_masks:
        raise SystemExit(
            f"FATAL: no zonal-wall row on the {letter} grid reaches "
            f"{MIN_ROW_CELLS} cells; there is nothing to resolve.")

    n = maps[f"d{letter}_avg"].shape[0]
    out = {"rows": zonal_wall_rows(bands), "scored_rows": sorted(row_masks),
           "band_cells": {k: int(v.sum()) for k, v in bands.items()},
           "terms": {}}

    for label, lego_t, nemo_t, diff_t in TERM_TEMPLATE:
        if label == "FORCING":
            diff, lego_key, nemo_key = maps[f"d{letter}_forcing"], None, None
        else:
            diff = maps[diff_t.format(C=letter)]
            lego_key = lego_t.format(C=letter)
            nemo_key = nemo_t.format(C=letter)
        rec: dict = {"has_nemo_side": nemo_key is not None,
                     "bands": {}, "rows": {}}
        for bname, bmask in bands.items():
            e = {"amp_diff": amp_on(diff, bmask, weights)}
            if nemo_key is not None:
                e["amp_lego"] = amp_on(maps[lego_key], bmask, weights)
                e["amp_nemo"] = amp_on(maps[nemo_key], bmask, weights)
                e["phase_agreement"] = [
                    float(x) for x in ewt.two_dt_phase_agreement(
                        maps[lego_key], maps[nemo_key], bmask,
                        area=weights, n_samples=n - 2)]
            rec["bands"][bname] = e

        int_band = rec["bands"]["interior"]
        for rname, rmask in row_masks.items():
            e = {"amp_diff": amp_on(diff, rmask, weights),
                 "floor": full_window_floor(diff, wet, rmask, weights)}
            e["floor_ratio"] = floor_ratio(e["amp_diff"], e["floor"])
            if int_band["amp_diff"] <= 0.0:
                # the whole term is identically zero -- report it as absent
                # rather than refusing, and skip an interval on 0/0.
                e["enrichment"] = 0.0
                e["enrichment_interval"] = {
                    "point": 0.0, "boot_lo": 0.0, "boot_hi": 0.0,
                    "subwindow": [0.0], "subwindow_lo": 0.0,
                    "subwindow_hi": 0.0, "subwindow_spread": 1.0,
                    "term_identically_zero": True}
            else:
                e["enrichment"] = ratio_or_refuse(
                    e["amp_diff"], int_band["amp_diff"],
                    f"{label} {rname} enrichment")
                e["enrichment_interval"] = enrichment_interval(
                    diff, rmask, bands["interior"], weights)
            if nemo_key is not None:
                e["amp_lego"] = amp_on(maps[lego_key], rmask, weights)
                e["amp_nemo"] = amp_on(maps[nemo_key], rmask, weights)
                e["side_ratio"] = ratio_or_refuse(
                    e["amp_lego"], e["amp_nemo"], f"{label} {rname} side")
                e["side_ratio_subwindows"] = side_ratio_subwindows(
                    maps[lego_key], maps[nemo_key], rmask, weights)
                # S6: how much more wall-concentrated legoESM's alternation
                # is than NEMO's.  NOT the comparable and NOT the same kind of
                # statistic as the 2.7x excess -- that claim is retracted in
                # v3; S6 is S4 divided by the interior ratio, which is a
                # basin-wide fact.  Reported with both factors visible.
                e["lego_enrichment"] = ratio_or_refuse(
                    e["amp_lego"], int_band["amp_lego"], "lego enrichment")
                e["nemo_enrichment"] = ratio_or_refuse(
                    e["amp_nemo"], int_band["amp_nemo"], "nemo enrichment")
                e["double_ratio"] = ratio_or_refuse(
                    e["lego_enrichment"], e["nemo_enrichment"],
                    f"{label} {rname} double ratio")
                e["double_ratio_interval"] = double_ratio_interval(
                    maps[lego_key], maps[nemo_key], rmask, bands["interior"],
                    weights)
                # THE DECOMPOSITION THAT KILLED VERSION 2's HEADLINE, computed
                # so it can never be hidden again: D is exactly the bare wall
                # ratio divided by the bare INTERIOR ratio, and the interior
                # factor is a basin-wide amplitude fact with nothing to do with
                # walls.  Printing both factors makes the inflation visible.
                e["interior_side_ratio"] = ratio_or_refuse(
                    int_band["amp_lego"], int_band["amp_nemo"],
                    "interior side ratio")
                e["reduction_sensitivity"] = reduction_sensitivity(
                    maps[lego_key], maps[nemo_key], rmask, bands["interior"],
                    weights)
                # a ratio is only a mode ratio if BOTH operands clear their own
                # leakage floor; version 2 floored the difference and left the
                # two operands that carry S4/S6 unfloored.
                for who, key in (("lego", lego_key), ("nemo", nemo_key)):
                    fl = full_window_floor(maps[key], wet, rmask, weights)
                    e[f"{who}_floor"] = fl
                    e[f"{who}_floor_ratio"] = floor_ratio(e[f"amp_{who}"], fl)
                e["operands_clear_floor"] = bool(
                    e["lego_floor_ratio"] > 2.0
                    and e["nemo_floor_ratio"] > 2.0)
            rec["rows"][rname] = e

        if int_band["amp_diff"] <= 0.0:
            rec["enrichment_band"] = 0.0
            rec["enrichment_band_interval"] = {
                "point": 0.0, "boot_lo": 0.0, "boot_hi": 0.0,
                "subwindow": [0.0], "subwindow_lo": 0.0, "subwindow_hi": 0.0,
                "subwindow_spread": 1.0, "term_identically_zero": True}
        else:
            rec["enrichment_band"] = ratio_or_refuse(
                rec["bands"]["zonal_wall"]["amp_diff"], int_band["amp_diff"],
                f"{label} band enrichment")
            rec["enrichment_band_interval"] = enrichment_interval(
                diff, bands["zonal_wall"], bands["interior"], weights)
        # Is this residual the SAME FIELD at every state?  If it is, the
        # two-step operator annihilates it exactly and its SIZE says nothing
        # about what the projection reports -- which is how version 2 turned a
        # state-constant loop error into a 31000x "exoneration".
        rec["state_independence"] = {
            b: state_independence(diff, bands[b], weights)
            for b in ("zonal_wall", "interior")}
        if nemo_key is not None:
            rec["double_ratio_band"] = ratio_or_refuse(
                ratio_or_refuse(rec["bands"]["zonal_wall"]["amp_lego"],
                                int_band["amp_lego"], "lego band enrichment"),
                ratio_or_refuse(rec["bands"]["zonal_wall"]["amp_nemo"],
                                int_band["amp_nemo"], "nemo band enrichment"),
                f"{label} band double ratio")
            rec["double_ratio_band_interval"] = double_ratio_interval(
                maps[lego_key], maps[nemo_key], bands["zonal_wall"],
                bands["interior"], weights)
        # S3 GEOMETRY, registered in v1 and never computed until round-2
        # review pointed that out.  The measured excess is a BOTH-end-walls
        # object, so a term that shows an excess at one wall and a deficit at
        # the other does not match it.  Decided on the BARE WALL RATIO (S4),
        # which is the statistic the target interval actually is.
        if nemo_key is not None:
            bare = {k: r["side_ratio"] for k, r in rec["rows"].items()}
            rec["s3_geometry"] = {
                "bare_wall_ratios": bare,
                "rows_above_one": [k for k, v in bare.items() if v > 1.0],
                "rows_at_or_below_one": [k for k, v in bare.items()
                                         if v <= 1.0],
                "holds_at_both_end_walls": bool(
                    len(bare) >= 2 and all(v > 1.0 for v in bare.values())),
            }
        out["terms"][label] = rec
    return out


def magnitude_check(comp: dict, maps: dict, letter: str) -> dict:
    """S5: the deposit as a per-step sea-surface displacement.

    Evaluated with the LOCAL metrics of the row it names.  Version 1 used basin
    medians against a wall-row amplitude, which review correctly called a
    mixed-location comparison inside one formula.  Still ungated: the free
    surface integrates the DIVERGENCE of the transport and this is a
    single-cell convergence scale.
    """
    import netCDF4

    seq = maps["_stamps"][0]["seqdump"]
    d = netCDF4.Dataset(os.path.join(seq, "mesh_mask.nc"))
    # acc_w = e2u*H on the u face and e1v*H on the v face, so the ACROSS-face
    # length that recovers the depth differs by component -- and the
    # convergence LENGTH is the along-flow one, which also differs.  Version 2
    # used the u pairing for both, which was wrong for the meridional branch
    # and harmless only because DINO's two horizontal scale factors are close;
    # the pairing is now correct per component rather than relying on that.
    if letter == "U":
        across = np.asarray(d.variables["e2u"][0]).squeeze()   # acc_w / across
        along = np.asarray(d.variables["e1u"][0]).squeeze()    # convergence dx
    else:
        across = np.asarray(d.variables["e1v"][0]).squeeze()
        along = np.asarray(d.variables["e2v"][0]).squeeze()
    d.close()
    if across.shape != along.shape:
        raise SystemExit("S5: the two metric arrays disagree in shape")
    wet = maps["_static"]["wetu" if letter == "U" else "wetv"]
    weights = maps["_static"]["acc_w" if letter == "U" else "acc_w_v"]
    depth = np.zeros_like(across)
    np.divide(weights, across, out=depth, where=across > 0)

    best = None
    for term in comp["terms"].values():
        for rname, r in term["rows"].items():
            if best is None or r["amp_diff"] > best[1]:
                best = (rname, r["amp_diff"])
    rname, amp = best
    j = int(rname.split("=")[1])
    rowsel = wet[j]
    dx = float(np.median(along[j][rowsel]))
    depth_m = float(np.median(depth[j][rowsel & (depth[j] > 0)]))
    eta = amp * depth_m * DT_S / dx
    return {"row": rname, "amp_ms": amp, "dx_km": dx / 1e3, "H_m": depth_m,
            "eta_m": eta, "ratio": eta / MEASURED_FLICKER_M, "dt_s": DT_S,
            "metrics": "local to the named row"}


def _print_component(cname: str, role: str, comp: dict) -> None:
    print(f"\n{'=' * 96}\n  COMPONENT {cname}  --  at a zonal wall this is the "
          f"{role} component\n{'=' * 96}")
    for r in comp["rows"]:
        print(f"    j={r['j']:3d}  {r['cells']:3d} cells")
    print(f"    scored rows: {comp['scored_rows']}   band cells: "
          f"{comp['band_cells']}")

    print(f"\n  S4, THE COMPARABLE: legoESM's wall amplitude over NEMO's, the "
          f"statistic the {MEASURED_EXCESS}x CI {list(MEASURED_EXCESS_CI)} "
          "actually is.")
    print("  S6 is S4 divided by the INTERIOR side ratio, printed beside it so "
          "the inflation is visible;")
    print("  it is a DIFFERENT quantity from the target and is not compared to "
          "that interval.")
    hdr = (f"  {'term':8s} {'row':7s} {'S4 wall':>8s} {'int.':>6s} "
           f"{'S6=S4/int':>9s} {'S6 by reduction':>22s} "
           f"{'lego/flr':>8s} {'nemo/flr':>8s} {'ok':>4s}")
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, rec in comp["terms"].items():
        for rname, r in rec["rows"].items():
            if "side_ratio" not in r:
                print(f"  {label:8s} {rname:7s} {'n/a (no NEMO side)':>8s}")
                continue
            rs = r["reduction_sensitivity"]
            print(f"  {label:8s} {rname:7s} {r['side_ratio']:8.2f} "
                  f"{r['interior_side_ratio']:6.2f} {r['double_ratio']:9.2f} "
                  f"[{rs['area_weighted_rms']:5.2f} {rs['median']:5.2f} "
                  f"{rs['trimmed_10_90']:5.2f}] "
                  f"{r['lego_floor_ratio']:8.2f} {r['nemo_floor_ratio']:8.2f} "
                  f"{'yes' if r['operands_clear_floor'] else 'NO':>4s}")
        if "s3_geometry" in rec:
            g = rec["s3_geometry"]
            print(f"           S3 GEOMETRY (both end walls must exceed 1): "
                  f"{g['holds_at_both_end_walls']}; above one "
                  f"{g['rows_above_one']}, at-or-below one "
                  f"{g['rows_at_or_below_one']}")
        if rec["has_nemo_side"]:
            for rname, r in rec["rows"].items():
                dv = r["double_ratio_interval"]
                print(f"           sub-windows IN ORDER at {rname}: "
                      f"S4 {[round(x, 2) for x in r['side_ratio_subwindows']]}"
                      f"  S6 {[round(x, 2) for x in dv['subwindow']]}  "
                      "(windows OVERLAP by two states, so three monotone "
                      "values is a ~1-in-3 event: weak, not a demonstration)")
            si = rec["state_independence"]
            for b, v in si.items():
                print(f"           STATE INDEPENDENCE of the {label} residual "
                      f"on {b}: unprojected rms {v['unprojected_rms']:.3e}, "
                      f"state-constant part {v['state_constant_rms']:.3e} "
                      f"({100 * v['state_constant_share']:.1f}%), cross-state "
                      f"corr [{v['cross_state_corr_min']:.5f}, "
                      f"{v['cross_state_corr_max']:.5f}]")
            pa = rec["bands"]["zonal_wall"]["phase_agreement"]
            pai = rec["bands"]["interior"]["phase_agreement"]
            print(f"           S7 phase lego-vs-NEMO: zonal band "
                  f"{[round(x, 3) for x in pa]}  interior "
                  f"{[round(x, 3) for x in pai]}  (a BAND average over rows "
                  "that may disagree in sign)")
        print(f"           S2 enrichment (wall/interior of the DIFFERENCE, "
              f"reported without a bar): band {rec['enrichment_band']:.3f} "
              "-- BELOW ONE MEANS THE WALL ROWS ALTERNATE LESS THAN THE "
              "INTERIOR")
    m5 = comp["s5_magnitude"]
    print(f"\n  S5 (reported, NOT gated): largest end-wall deposit "
          f"{m5['amp_ms']:.3e} m/s at {m5['row']}, LOCAL dx {m5['dx_km']:.1f} "
          f"km / depth {m5['H_m']:.0f} m -> ~{m5['eta_m']:.2e} m of surface "
          f"per step vs the {MEASURED_FLICKER_M:.1e} m measured "
          f"({m5['ratio']:.3g}x).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-dir", required=True,
                    help="directory holding deposit_map_kt<KT>.npz")
    ap.add_argument("--out", required=True, help="JSON artifact path")
    a = ap.parse_args(argv)

    maps = load_maps(a.map_dir)
    static = maps["_static"]
    n = maps["dU_avg"].shape[0]

    print("\n=== #1455: the barotropic per-step DEPOSIT on the two-step mode, "
          "per cell, at the end-wall rows, BOTH components ===")
    print(f"  states: {len(CONSECUTIVE_KTS)} consecutive "
          f"(kt={CONSECUTIVE_KTS[0]}..{CONSECUTIVE_KTS[-1]}) -> "
          f"{n - 2} Nyquist samples, {max(n - 4, 0)} floor sample(s).")
    print("  the brief asked for NINE; five is every consecutive NEMO seqdump "
          "lane that exists on disk (see the module docstring).")

    atten = window_attenuation(static["wgt_primary"])
    print(f"\n  WINDOW ATTENUATION, measured from legoESM's own saved weights: "
          f"{atten['n_nonzero']} nonzero of {atten['n_substeps']} substeps, "
          f"first at #{atten['first_nonzero_substep']};")
    print(f"      a SUBSTEP-alternating source passes at "
          f"{atten['substep_nyquist_transfer']:.4g}, i.e. attenuated "
          f"{atten['attenuation_factor']:.1f}x -- suppressed, NOT annihilated.")

    # CONTROLS RUN PER COMPONENT.  Version 2 ran all three on the zonal grid
    # only, while the meridional grid carried the claim -- round-2 review
    # mutated the v component to score with the u weights (a 68% different
    # array) and every test stayed green.  The docstring already said "none is
    # inherited"; now that is true of the components too.
    checks = {}
    print("\n  CONTROLS, PER COMPONENT (all must pass before any number "
          "below is printed)")
    for cname, letter, wkey, mkey, _role in COMPONENTS:
        w, wet = static[wkey], static[mkey]
        band = build_bands(wet)["zonal_wall"]
        diff = maps[f"d{letter}_avg"]
        c = {"synthetic": synthetic_checks(band, w, n),
             "land_poison": land_poison_check(diff, band, w, wet),
             "weights": weights_load_bearing(diff, band, w)}
        checks[cname] = c
        sc, lp, wl = c["synthetic"], c["land_poison"], c["weights"]
        print(f"    [{cname}] projector on a known alternating field: "
              f"{sc['alternating']:.6e} vs planted "
              f"{sc['alternating_expected']:.6e}; constant "
              f"{sc['constant']:.1e}, ramp {sc['ramp']:.1e}; one estimator: "
              f"{sc['cell_agg_matches_committed']}")
        print(f"    [{cname}] alternating land poison: production "
              f"{lp['production_base']:.6e} -> {lp['production_poisoned']:.6e}"
              f" (STRUCTURAL INVARIANT); unmasked {lp['unmasked_base']:.2e} -> "
              f"{lp['unmasked_poisoned']:.2e} (moves={lp['unmasked_moves']}, "
              "the live arm)")
        print(f"    [{cname}] weights load-bearing: {wl['weighted']:.6e} vs "
              f"uniform {wl['uniform']:.6e} (rel {wl['rel_change']:.2e}), "
              f"spread {wl['weight_spread']:.1f}x")

    res = {"n_states": n, "n_nyquist_samples": n - 2,
           "n_floor_samples": max(n - 4, 0),
           "window_attenuation": atten, "controls": checks, "components": {}}
    for cname, letter, wkey, mkey, role in COMPONENTS:
        comp = project_component(maps, letter, wkey, mkey)
        comp["role_at_a_zonal_wall"] = role
        comp["s5_magnitude"] = magnitude_check(comp, maps, letter)
        res["components"][cname] = comp
        _print_component(cname, role, comp)

    print(f"\n  S4 (bare wall ratio) is THE COMPARABLE, against the measured "
          f"excess {MEASURED_EXCESS}x CI {list(MEASURED_EXCESS_CI)}.")
    print("  S6 (S4 divided by the interior ratio) is a DIFFERENT quantity and "
          "is NOT compared to that interval; version 2 did and that is "
          "retracted.")
    print("  S2 is reported WITHOUT a pass/fail: its version-1 bar was a "
          "transferred statistic and is retracted.")
    print("  Believe the SUBWINDOW sequence over the bootstrap: the bootstrap "
          "resamples cells independently and wall cells are correlated.")

    res["bars"] = {
        "measured_excess": MEASURED_EXCESS,
        "measured_excess_ci": list(MEASURED_EXCESS_CI),
        "comparable": "S4, the bare wall ratio (no interior denominator)",
        "s2_bar": "RETRACTED v2 (transferred statistic)",
        "s4_bar": "RESTORED v3 as THE comparable; v2's retraction of it used "
                  "reasons belonging to other terms and is itself retracted",
        "s6_bar": "RETRACTED v3: S6 = S4 / (interior side ratio), and the "
                  "interior factor is a basin-wide amplitude fact, not a wall "
                  "one. It is a different quantity from the target and is not "
                  "compared to that interval."}
    res["measured_flicker_m"] = MEASURED_FLICKER_M
    res["states"] = maps["_stamps"]
    res["provenance"] = ewt.provenance({"probe": "baro_deposit_wall_nyquist",
                                        "map_dir": os.path.abspath(a.map_dir)})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\n  artifact: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
