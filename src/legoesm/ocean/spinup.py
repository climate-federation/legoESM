"""Multi-decade ocean spin-up workflow + convergence diagnostics.

Provides primitives for centennial OMIP-2 / CMIP-class spin-up runs:

* :func:`compute_amoc_timeseries` — append-only AMOC@26.5°N tracker.
* :class:`SpinupHealth` — per-year health snapshot (AMOC, RPE drift,
  volume / heat / salt drift fractions).
* :func:`evaluate_health` — emit a SpinupHealth from current + initial
  diagnostics.
* :class:`ConvergenceCriteria` — declarative thresholds for
  equilibration detection (AMOC stability window, RPE drift bound,
  volume drift bound).
* :func:`is_converged` — apply criteria to the recent history.
* :func:`find_latest_restart` — locate the most-recent restart file in
  a run directory so the driver auto-resumes.
* :func:`bryan_accelerated_dt` — return the per-phase timestep used by
  the Bryan-Lewis (1984) distorted-physics accelerated spin-up
  protocol (long tracer dt during phase 1, gradual ramp-down to
  physical dt by phase 3).

All helpers are pure-Python / NumPy where possible so the workflow is
JIT-independent and easy to drive from a top-level run script.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np


SV = 1.0e6  # 1 Sverdrup [m³/s]


# ==============================================================================
# AMOC timeseries
# ==============================================================================

def compute_amoc_timeseries(
    amoc_yearly_Sv: list[float] | np.ndarray,
    *,
    window_years: int = 30,
) -> dict:
    """Summarise the AMOC time series for spin-up health.

    Computes:
        * Latest value.
        * Trailing-window mean.
        * Trailing-window standard deviation (variability measure).
        * Trailing-window slope from a linear fit [Sv/yr] (drift rate).
        * Drift fraction over the window: ``|slope · window| / mean``.

    Parameters
    ----------
    amoc_yearly_Sv : sequence of float
        AMOC max at the monitoring latitude (typically 26.5°N) in Sv,
        one value per simulated year (oldest first).
    window_years : int
        Trailing window for stability metrics.  Default 30 years —
        long enough to average out interannual variability, short
        enough to detect persistent drift.
    """
    a = np.asarray(amoc_yearly_Sv, dtype=np.float64)
    if a.size == 0:
        return {
            "latest_Sv": float("nan"),
            "window_mean_Sv": float("nan"),
            "window_std_Sv": float("nan"),
            "window_slope_Sv_per_yr": float("nan"),
            "window_drift_fraction": float("nan"),
            "n_years": 0,
            "window_years": window_years,
        }
    latest = float(a[-1])
    window = a[-window_years:] if a.size >= window_years else a
    mean = float(np.mean(window))
    std = float(np.std(window, ddof=0))
    if window.size >= 2:
        years = np.arange(window.size, dtype=np.float64)
        slope, _ = np.polyfit(years, window, 1)
        slope = float(slope)
    else:
        slope = 0.0
    drift_fraction = (
        abs(slope * window.size) / max(abs(mean), 1e-12)
        if mean != 0.0 else float("inf")
    )
    return {
        "latest_Sv": latest,
        "window_mean_Sv": mean,
        "window_std_Sv": std,
        "window_slope_Sv_per_yr": slope,
        "window_drift_fraction": drift_fraction,
        "n_years": int(a.size),
        "window_years": int(min(window_years, a.size)),
    }


# ==============================================================================
# Spin-up health snapshot
# ==============================================================================

class SpinupHealth(NamedTuple):
    """Per-year spin-up health snapshot.

    ``rpe_drift_W_per_m2`` is the reference-potential-energy drift
    relative to the initial state, area-averaged — a standard
    Griffies (2015) mixing diagnostic.  ``volume_drift_frac`` and
    ``heat_drift_frac`` are the fractional drifts of the global
    volume + heat-content integrals (signed; positive = gain).
    ``salt_drift_frac`` follows the same convention for the salt
    integral.
    """
    year: int
    amoc_Sv: float
    rpe_drift_W_per_m2: float
    volume_drift_frac: float
    heat_drift_frac: float
    salt_drift_frac: float


def evaluate_health(
    *,
    year: int,
    amoc_Sv: float,
    rpe_drift_W_per_m2: float,
    volume_now: float,
    volume_init: float,
    heat_now: float,
    heat_init: float,
    salt_now: float,
    salt_init: float,
) -> SpinupHealth:
    """Build a :class:`SpinupHealth` from raw diagnostics."""
    def _drift(now: float, init: float) -> float:
        if abs(init) < 1e-30:
            return float("nan")
        return float((now - init) / init)

    return SpinupHealth(
        year=int(year),
        amoc_Sv=float(amoc_Sv),
        rpe_drift_W_per_m2=float(rpe_drift_W_per_m2),
        volume_drift_frac=_drift(volume_now, volume_init),
        heat_drift_frac=_drift(heat_now, heat_init),
        salt_drift_frac=_drift(salt_now, salt_init),
    )


# ==============================================================================
# Convergence criteria
# ==============================================================================

class ConvergenceCriteria(NamedTuple):
    """Thresholds for declaring spin-up equilibrium.

    Defaults are conservative OMIP-2 protocol values:
        * AMOC drift fraction < 5 % over a 30-year window.
        * RPE drift < 0.05 W/m² (Griffies 2015 mixing-budget bound).
        * Volume drift fraction < 1e-3.
        * Heat-content drift < 1e-2 (1 % over the spin-up).
        * Salt-mass drift < 1e-3 (sharp because salt is exactly
          conserved by transport).
    """
    window_years: int = 30
    amoc_drift_fraction_max: float = 0.05
    rpe_drift_W_per_m2_max: float = 0.05
    volume_drift_fraction_max: float = 1.0e-3
    heat_drift_fraction_max: float = 1.0e-2
    salt_drift_fraction_max: float = 1.0e-3


def is_converged(
    history: list[SpinupHealth],
    criteria: ConvergenceCriteria = ConvergenceCriteria(),
) -> dict:
    """Apply convergence criteria to the spin-up history.

    Returns a dict with the boolean ``"converged"`` plus per-criterion
    pass/fail flags + the underlying metric values so callers can log
    a detailed status.

    AMOC handling: when every AMOC value in the history is NaN
    (e.g. the driver has not yet wired the MOC streamfunction
    diagnostic), the AMOC criterion is skipped — the returned dict
    sets ``amoc_skipped=True`` and ``amoc_ok=True`` so the
    remaining criteria can still drive a convergence decision.  A
    partial-NaN history is still evaluated against the finite tail.
    """
    if len(history) == 0:
        return {"converged": False, "reason": "empty history"}

    amoc_series = [h.amoc_Sv for h in history]
    # Build the longest contiguous FINITE suffix that ends at the
    # last finite value in the series:
    #   1. Strip TRAILING NaNs (transient diagnostic gaps) — they
    #      don't invalidate the convergence check.
    #   2. From the resulting tail, walk backward over the
    #      contiguous finite values to obtain the evaluation window.
    # Guard: if the most recent ``max_trailing_nan_years`` block is
    # all NaN, treat as a data regression (``amoc_has_recent_nan``)
    # and fail the AMOC criterion — we will not pass convergence on
    # stale finite values that are far in the past.
    max_trailing_nan = max(int(criteria.window_years // 3), 5)

    # Identify the position of the last finite value.
    last_finite_idx = -1
    for i in range(len(amoc_series) - 1, -1, -1):
        if np.isfinite(amoc_series[i]):
            last_finite_idx = i
            break

    all_nan = last_finite_idx < 0
    trailing_nan_count = len(amoc_series) - 1 - last_finite_idx
    has_recent_nan = (not all_nan) and trailing_nan_count > max_trailing_nan

    if last_finite_idx >= 0:
        finite_suffix: list[float] = []
        for i in range(last_finite_idx, -1, -1):
            v = amoc_series[i]
            if np.isfinite(v):
                finite_suffix.append(v)
            else:
                break
        finite_suffix.reverse()
    else:
        finite_suffix = []

    amoc_skipped = all_nan

    if amoc_skipped:
        amoc_ok = True
        summary = {
            "latest_Sv": float("nan"),
            "window_mean_Sv": float("nan"),
            "window_std_Sv": float("nan"),
            "window_slope_Sv_per_yr": float("nan"),
            "window_drift_fraction": float("nan"),
            "n_years": 0,
            "window_years": criteria.window_years,
        }
    elif has_recent_nan:
        # AMOC diagnostic has been missing for more than
        # ``max_trailing_nan`` years — too long to trust the finite
        # past as representative of current state.  Fail until the
        # diagnostic returns.
        amoc_ok = False
        summary = {
            "latest_Sv": float("nan"),
            "window_mean_Sv": float("nan"),
            "window_std_Sv": float("nan"),
            "window_slope_Sv_per_yr": float("nan"),
            "window_drift_fraction": float("nan"),
            "n_years": 0,
            "window_years": criteria.window_years,
        }
    else:
        summary = compute_amoc_timeseries(
            finite_suffix, window_years=criteria.window_years,
        )
        amoc_ok = (
            np.isfinite(summary["window_drift_fraction"])
            and summary["window_drift_fraction"] < criteria.amoc_drift_fraction_max
        )

    latest = history[-1]
    rpe_ok = abs(latest.rpe_drift_W_per_m2) < criteria.rpe_drift_W_per_m2_max
    vol_ok = abs(latest.volume_drift_frac) < criteria.volume_drift_fraction_max
    heat_ok = abs(latest.heat_drift_frac) < criteria.heat_drift_fraction_max
    salt_ok = abs(latest.salt_drift_frac) < criteria.salt_drift_fraction_max

    converged = bool(amoc_ok and rpe_ok and vol_ok and heat_ok and salt_ok)
    return {
        "converged": converged,
        "amoc_ok": amoc_ok,
        "amoc_skipped": amoc_skipped,
        "amoc_has_recent_nan": has_recent_nan,
        "rpe_ok": rpe_ok,
        "volume_ok": vol_ok,
        "heat_ok": heat_ok,
        "salt_ok": salt_ok,
        "amoc_summary": summary,
        "latest": latest._asdict(),
        "criteria": criteria._asdict(),
    }


# ==============================================================================
# Restart-chain helpers
# ==============================================================================

_RESTART_YEAR_RE = re.compile(r"restart_year_(\d{4,})\.npz$")


def find_latest_restart(out_dir: str | Path) -> Path | None:
    """Return the path of the most-recent ``restart_year_YYYY.npz`` file.

    Looks in ``out_dir`` for files matching the
    ``restart_year_<NNNN>.npz`` naming convention used by
    :func:`legoesm.ocean.restart.save_restart`.  Returns the file
    with the maximum year number, or ``None`` if no restart is found.

    Parameters
    ----------
    out_dir : str or Path

    Returns
    -------
    Path or None
    """
    p = Path(out_dir)
    if not p.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for f in p.iterdir():
        m = _RESTART_YEAR_RE.search(f.name)
        if m is None:
            continue
        candidates.append((int(m.group(1)), f))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


# ==============================================================================
# Bryan-Lewis accelerated spin-up timestep
# ==============================================================================

def bryan_accelerated_dt(
    year: int,
    *,
    dt_physical_s: float = 1800.0,
    dt_tracer_ratio: float = 10.0,
    phase1_years: int = 200,
    phase2_years: int = 100,
) -> tuple[float, float]:
    """Return ``(dt_momentum, dt_tracer)`` for the Bryan-Lewis spin-up.

    Bryan & Lewis (1984) introduced a 3-phase distorted-physics
    accelerated spin-up that uses a long tracer timestep to advance
    the slow tracer field while the momentum equation is integrated
    on a physical timestep.  Phase 3 (post-acceleration) uses the
    physical timestep for both equations so the model returns to
    the standard regime before the production phase begins.

    Phase 1 (year < phase1_years): ``dt_tracer = ratio · dt_phys``.
    Phase 2 (phase1 ≤ year < phase1 + phase2): linear ramp from
    ``ratio · dt_phys`` down to ``dt_phys``.
    Phase 3 (year ≥ phase1 + phase2): ``dt_tracer = dt_phys``.

    Parameters
    ----------
    year : int
        0-indexed year inside the spin-up.
    dt_physical_s : float
        Physical timestep [s] for momentum.
    dt_tracer_ratio : float
        Tracer-to-momentum dt ratio during phase 1.
    phase1_years : int
        Length of accelerated phase 1.
    phase2_years : int
        Length of phase-2 ramp-down.

    Returns
    -------
    dt_momentum_s : float
    dt_tracer_s : float
    """
    if year < phase1_years:
        return dt_physical_s, dt_physical_s * dt_tracer_ratio
    if year < phase1_years + phase2_years:
        # Linear ramp from ratio → 1 over phase2.
        progress = (year - phase1_years) / max(phase2_years, 1)
        ratio = dt_tracer_ratio + (1.0 - dt_tracer_ratio) * progress
        return dt_physical_s, dt_physical_s * ratio
    return dt_physical_s, dt_physical_s
