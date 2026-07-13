"""Pure-numpy comparison metrics for land-surface evaluation.

This is the single shared metrics library for the land evaluation pipeline
(``docs/land/evaluation_pipeline.md``).  Runners, validators, plotters and
the recipe driver all import from here — no metric is re-implemented in a
script (CLAUDE.md: "No duplicate numerics ... Plotters NOT exempt").

Design follows two established land/ESM benchmarking frameworks:

* **ILAMB** (Collier et al. 2018, *JAMES* 10:2731, doi:10.1029/2018MS001354):
  each variable is summarised by component *scores* on the unit interval,
  obtained by squashing a normalised relative error through an exponential
  ``score = exp(-relative_error)`` (a 50 % relative error maps to ~0.61).
  The bias score normalises by the *reference* variability; the RMSE score
  uses the *centralised* RMSE (bias removed, since bias is scored
  separately); the distribution score is the Taylor-diagram functional
  ``2(1+R) / (sigma_ratio + 1/sigma_ratio)^2``.
* **PLUMBER2 / PALS** (Ukkola et al. 2022, *ESSD* 14:449; Abramowitz et al.
  2024, *BG* 21:5517): normalised mean error (NME), correlation,
  5th/95th-percentile error, and PDF overlap are the canonical flux-tower
  skill metrics.

All functions take 1-D ``ref`` (reference/observation) and ``mod`` (model)
arrays, mask non-finite and sentinel-magnitude entries pairwise, and return
Python floats (``nan`` when there are too few valid pairs).  No JAX — this
is an offline analysis utility.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# Sentinel-magnitude guard: CHATS obs use 1e36 for "missing"; loaders convert
# most sentinels to NaN, but the magnitude guard keeps a stray 1e36 (or a
# -9999 that slipped through) from poisoning a mean.  Kept identical to the
# legacy ``scalar_stats`` guard for byte-compatible validator output.
_MAX_ABS = 1e30

# Numerical floor for "reference has no variability" (avoids /0 in normalised
# scores).  Not a tunable — a regulariser, so a plain module constant.
_STD_FLOOR = 1e-6

# Algorithmic (non-physical) defaults, named so no bare literal sits in a
# function/lambda default (repo inline-coeff gate; these are counts/levels,
# not empirical physics coefficients).
_DIURNAL_N_BINS = 48  # half-hourly composite (24 h / 0.5 h)
_PDF_N_BINS = 50  # histogram bins for the PDF-overlap coefficient
_P_LOW = 5.0  # PLUMBER2 lower-tail percentile level
_P_HIGH = 95.0  # PLUMBER2 upper-tail percentile level


def finite_pair(
    ref: np.ndarray, mod: np.ndarray, max_abs: float = _MAX_ABS
) -> tuple[np.ndarray, np.ndarray]:
    """Return the pairwise-valid ``(ref, mod)`` subset.

    A sample is kept only when BOTH series are finite and below
    ``max_abs`` in magnitude, so every metric below scores exactly the
    same set of timestamps.
    """
    ref = np.asarray(ref, dtype=np.float64).ravel()
    mod = np.asarray(mod, dtype=np.float64).ravel()
    if ref.shape != mod.shape:
        raise ValueError(
            f"ref/mod length mismatch: {ref.shape} vs {mod.shape}"
        )
    mask = (
        np.isfinite(ref)
        & np.isfinite(mod)
        & (np.abs(ref) < max_abs)
        & (np.abs(mod) < max_abs)
    )
    return ref[mask], mod[mask]


# ---------------------------------------------------------------------------
# Elementary error metrics (lower is better)
# ---------------------------------------------------------------------------


def bias(ref: np.ndarray, mod: np.ndarray) -> float:
    """Mean model-minus-reference difference (model units)."""
    r, m = finite_pair(ref, mod)
    if r.size < 1:
        return float("nan")
    return float(np.mean(m - r))


def mae(ref: np.ndarray, mod: np.ndarray) -> float:
    """Mean absolute error."""
    r, m = finite_pair(ref, mod)
    if r.size < 1:
        return float("nan")
    return float(np.mean(np.abs(m - r)))


def rmse(ref: np.ndarray, mod: np.ndarray) -> float:
    """Root-mean-square error (includes the bias contribution)."""
    r, m = finite_pair(ref, mod)
    if r.size < 1:
        return float("nan")
    return float(np.sqrt(np.mean((m - r) ** 2)))


def centered_rmse(ref: np.ndarray, mod: np.ndarray) -> float:
    """Bias-removed (centralised) RMSE.

    ``crmse^2 = rmse^2 - bias^2``.  This is what ILAMB's RMSE score uses,
    so that the same error is not penalised twice (once as bias, once as
    RMSE).
    """
    r, m = finite_pair(ref, mod)
    if r.size < 1:
        return float("nan")
    dm = m - np.mean(m)
    dr = r - np.mean(r)
    return float(np.sqrt(np.mean((dm - dr) ** 2)))


def nrmse(ref: np.ndarray, mod: np.ndarray) -> float:
    """RMSE normalised by the reference standard deviation.

    Matches the legacy ``scalar_stats`` NRMSE (``std`` floored at 1e-6)
    used by ``validate_clm_ml_canopy.py`` so scorecards stay comparable
    with previously reported parity numbers.
    """
    r, m = finite_pair(ref, mod)
    if r.size < 2:
        return float("nan")
    scale = max(float(np.std(r)), _STD_FLOOR)
    return float(rmse(r, m) / scale)


def nme(ref: np.ndarray, mod: np.ndarray) -> float:
    """Normalised mean error (PLUMBER2 / PALS).

    ``NME = sum|mod - ref| / sum|ref - mean(ref)|`` — absolute error
    normalised by the reference's own variability, so 0 is perfect and
    1 is "no better than predicting the reference mean".  Returns ``nan``
    when the reference is constant (denominator 0).
    """
    r, m = finite_pair(ref, mod)
    if r.size < 2:
        return float("nan")
    denom = float(np.sum(np.abs(r - np.mean(r))))
    if denom <= _STD_FLOOR:
        return float("nan")
    return float(np.sum(np.abs(m - r)) / denom)


# ---------------------------------------------------------------------------
# Association / distribution metrics
# ---------------------------------------------------------------------------


def pearson_r(ref: np.ndarray, mod: np.ndarray) -> float:
    """Pearson correlation coefficient.

    Degenerate-variance handling matches the legacy ``scalar_stats``: if
    either series is (numerically) constant, return 1.0 for identical
    series else 0.0, rather than a divide-by-zero ``nan``.
    """
    r, m = finite_pair(ref, mod)
    if r.size < 2:
        return float("nan")
    if np.std(r) > 1e-10 and np.std(m) > 1e-10:
        return float(np.corrcoef(r, m)[0, 1])
    return 1.0 if np.allclose(r, m, atol=1e-8) else 0.0


def std_ratio(ref: np.ndarray, mod: np.ndarray) -> float:
    """Ratio of model to reference standard deviation (Taylor axis)."""
    r, m = finite_pair(ref, mod)
    if r.size < 2:
        return float("nan")
    sr = max(float(np.std(r)), _STD_FLOOR)
    return float(np.std(m) / sr)


def percentile_error(ref: np.ndarray, mod: np.ndarray, q: float) -> float:
    """Model-minus-reference difference of the ``q``-th percentile.

    PLUMBER2 reports the 5th and 95th percentile errors as tail-behaviour
    diagnostics (does the model capture the extremes of the flux
    distribution).  ``q`` is in [0, 100].
    """
    r, m = finite_pair(ref, mod)
    if r.size < 2:
        return float("nan")
    return float(np.percentile(m, q) - np.percentile(r, q))


def pdf_overlap(
    ref: np.ndarray, mod: np.ndarray, n_bins: int = _PDF_N_BINS
) -> float:
    """Overlap coefficient of the two empirical PDFs (PLUMBER2).

    Both series are histogrammed on a shared range into ``n_bins`` bins;
    the overlap is ``sum(min(p_ref, p_mod))`` of the normalised
    densities.  1.0 = identical distributions, 0.0 = disjoint.  Insensitive
    to timing errors (a pure distribution-shape metric), which is exactly
    why PLUMBER2 uses it alongside temporal correlation.
    """
    r, m = finite_pair(ref, mod)
    if r.size < 2:
        return float("nan")
    lo = float(min(r.min(), m.min()))
    hi = float(max(r.max(), m.max()))
    if hi - lo <= _STD_FLOOR:
        return 1.0 if np.allclose(r, m, atol=1e-8) else 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    pr, _ = np.histogram(r, bins=edges, density=True)
    pm, _ = np.histogram(m, bins=edges, density=True)
    width = edges[1] - edges[0]
    # Convert densities to per-bin probability mass before overlapping.
    return float(np.sum(np.minimum(pr, pm)) * width)


# ---------------------------------------------------------------------------
# ILAMB-style unit-interval scores (higher is better, in [0, 1])
# ---------------------------------------------------------------------------


def relative_error_score(relative_error: float) -> float:
    """Squash a non-negative relative error to a [0, 1] score.

    ILAMB's canonical mapping ``score = exp(-relative_error)``: a 0 error
    scores 1, a 50 % relative error scores ``exp(-0.5) ~= 0.61``, a 100 %
    error scores ``exp(-1) ~= 0.37``.  Returns ``nan`` for a ``nan`` input
    so missing variables propagate rather than silently scoring 1.
    """
    if not np.isfinite(relative_error):
        return float("nan")
    return float(np.exp(-abs(float(relative_error))))


def bias_score(ref: np.ndarray, mod: np.ndarray) -> float:
    """ILAMB bias score: ``exp(-|bias| / sigma_ref)``.

    Normalising by the reference standard deviation makes the score
    dimensionless and comparable across variables of different magnitude.
    """
    r, m = finite_pair(ref, mod)
    if r.size < 2:
        return float("nan")
    scale = max(float(np.std(r)), _STD_FLOOR)
    return relative_error_score(abs(bias(r, m)) / scale)


def rmse_score(ref: np.ndarray, mod: np.ndarray) -> float:
    """ILAMB RMSE score: ``exp(-crmse / sigma_ref)`` (centralised RMSE)."""
    r, m = finite_pair(ref, mod)
    if r.size < 2:
        return float("nan")
    scale = max(float(np.std(r)), _STD_FLOOR)
    return relative_error_score(centered_rmse(r, m) / scale)


def taylor_score(ref: np.ndarray, mod: np.ndarray) -> float:
    """Taylor-diagram skill score in [0, 1].

    ``S = 2 (1 + R) / (sigma_ratio + 1/sigma_ratio)^2`` — the ILAMB spatial-
    /distribution-score functional (Taylor 2001).  Rewards both high
    correlation ``R`` and a model standard deviation matching the
    reference (``sigma_ratio -> 1``).  Uses the correlation-clipped form so
    an anti-correlated series cannot produce a negative score.
    """
    r, m = finite_pair(ref, mod)
    if r.size < 2:
        return float("nan")
    R = pearson_r(r, m)
    ratio = std_ratio(r, m)
    if not np.isfinite(ratio) or ratio <= 0.0:
        return float("nan")
    R = max(R, -1.0)
    return float(2.0 * (1.0 + R) / (ratio + 1.0 / ratio) ** 2)


def phase_score(
    ref_cycle: np.ndarray, mod_cycle: np.ndarray
) -> float:
    """Cyclic phase-agreement score in [0, 1] for a composite cycle.

    Given two equal-length composites of ONE period (e.g. a 48-bin
    diurnal composite or a 12-bin seasonal cycle), compares the phase of
    the peak: ``S = 0.5 (1 + cos(2*pi * delta_bin / n_bins))``.  A 0-bin
    shift scores 1; a half-period shift scores 0.  This is the ILAMB
    seasonal-cycle "phase" score generalised to any single cycle.
    """
    a = np.asarray(ref_cycle, dtype=np.float64).ravel()
    b = np.asarray(mod_cycle, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size < 2:
        return float("nan")
    if not (np.any(np.isfinite(a)) and np.any(np.isfinite(b))):
        return float("nan")
    n = a.size
    ia = int(np.nanargmax(a))
    ib = int(np.nanargmax(b))
    delta = ia - ib
    return float(0.5 * (1.0 + np.cos(2.0 * np.pi * delta / n)))


# ---------------------------------------------------------------------------
# Cyclic composites (shared by every diurnal figure — dedup of the copy in
# validate_clm_ml_canopy.py::diurnal_cycle)
# ---------------------------------------------------------------------------


def diurnal_cycle(
    time_days: np.ndarray, values: np.ndarray, n_bins: int = _DIURNAL_N_BINS
) -> np.ndarray:
    """Composite ``values`` into ``n_bins`` bins over the fractional day.

    ``time_days`` is any time coordinate in days (Julian day, calday,
    DOY); only the fractional part is used so bin 0 is 00:00.  Non-finite
    values are dropped per bin; empty bins are ``nan``.  Default 48 bins =
    half-hourly, matching the CHATS7 cadence.
    """
    t = np.asarray(time_days, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if t.shape != v.shape:
        raise ValueError(f"time/values length mismatch: {t.shape} vs {v.shape}")
    frac = t % 1.0
    # Round-and-wrap (bins centred on the sample times), matching the legacy
    # validate_clm_ml_canopy.py::diurnal_cycle so existing figures reproduce.
    # Rounding is robust to the float drift that makes floor((k/n)*n) == k-1.
    idx = np.round(frac * n_bins).astype(int) % n_bins
    out = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = v[idx == b]
        sel = sel[np.isfinite(sel) & (np.abs(sel) < _MAX_ABS)]
        if sel.size:
            out[b] = float(sel.mean())
    return out


# ---------------------------------------------------------------------------
# Registry + backward-compatible aggregate
# ---------------------------------------------------------------------------

# Name -> callable(ref, mod) -> float.  The recipe driver selects metrics by
# name from here; an unknown name is a hard error (dispatch hardening), never
# a silent skip.  Scores (higher-better) and errors (lower-better) coexist;
# the scorecard knows which is which via SCORE_METRICS below.
METRIC_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "bias": bias,
    "mae": mae,
    "rmse": rmse,
    "centered_rmse": centered_rmse,
    "nrmse": nrmse,
    "nme": nme,
    "corr": pearson_r,
    "pearson_r": pearson_r,
    "std_ratio": std_ratio,
    "pdf_overlap": pdf_overlap,
    "bias_score": bias_score,
    "rmse_score": rmse_score,
    "taylor_score": taylor_score,
    "p5_error": lambda ref, mod: percentile_error(ref, mod, _P_LOW),
    "p95_error": lambda ref, mod: percentile_error(ref, mod, _P_HIGH),
}

# Metrics whose value is already a [0, 1] "higher-is-better" score (used by
# the scorecard to build the overall variable score, and to know not to
# invert them).  pdf_overlap is in [0, 1] and higher-better too.
SCORE_METRICS: frozenset[str] = frozenset(
    {"bias_score", "rmse_score", "taylor_score", "pdf_overlap"}
)


def get_metric(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Look up a metric by name, raising on an unknown selection."""
    try:
        return METRIC_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown metric {name!r}; known metrics: "
            f"{sorted(METRIC_REGISTRY)}"
        ) from None


def scalar_stats(ref: np.ndarray, mod: np.ndarray) -> dict:
    """Backward-compatible stats dict (drop-in for the legacy helper).

    Returns the same keys ``validate_clm_ml_canopy.py`` and
    ``diff_chats7_adapter_vs_fortran.py`` already print, so those scripts
    can import this instead of carrying their own copy.
    """
    r, m = finite_pair(ref, mod)
    n = int(r.size)
    if n < 2:
        return dict(
            n=n, rmse=np.nan, mae=np.nan, bias=np.nan,
            r2=np.nan, corr=np.nan, nrmse=np.nan,
        )
    corr = pearson_r(r, m)
    return dict(
        n=n,
        rmse=rmse(r, m),
        mae=mae(r, m),
        bias=bias(r, m),
        r2=corr ** 2,
        corr=corr,
        nrmse=nrmse(r, m),
    )
