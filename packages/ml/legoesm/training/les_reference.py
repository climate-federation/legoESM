"""Time-mean LES reference profiles for scoring a single-column model.

The LES drivers write one ``profiles/prof_NNN.npz`` per recorded frame
(``scripts/run/les_record.py``): planar-mean profiles plus resolved second
moments, including the ``wth``/``wqv`` turbulent fluxes a boundary-layer
closure is supposed to predict. This module turns those frames into a single
time-mean reference on the SCM's own levels, plus the mass weights the score
uses.

Three rules this module exists to enforce, all of them CLAUDE.md gates that
have been violated before:

* **The averaging window is explicit and travels with the reference.** LES
  cumulus/stratocumulus fields fluctuate frame to frame, so an instantaneous
  final frame is not the reference; a trailing time-mean over a stated window
  is. :attr:`LESReference.window_hours` records it so the SCM can be averaged
  over the SAME window and the window can be printed next to every number.
* **NaN is fatal, never averaged away.** No ``nanmean``: a frame containing a
  non-finite value means the LES went bad, and silently dropping it would
  manufacture a clean reference from a broken run.
* **Interpolation is one-directional and stated.** The LES is finer than the
  SCM, so the LES profile is interpolated ONTO the SCM levels (never the
  reverse), and only over SCM levels inside the LES domain.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "LESReference",
    "REFERENCE_VARIABLES",
    "load_les_reference",
    "mass_weights_from_pressure",
]

# Variables scored against the LES. The state variables are what the closure
# controls; the fluxes are what it directly parameterizes. `qc` is carried for
# plotting/diagnosis and is only scored when the SCM produces condensate.
REFERENCE_VARIABLES: tuple[str, ...] = (
    "theta", "qv", "u", "v", "wth", "wqv",
)

_OPTIONAL_VARIABLES: tuple[str, ...] = ("qc", "cloud_frac", "tke")


@dataclass(frozen=True)
class LESReference:
    """Time-mean LES profiles interpolated onto SCM levels.

    All profile arrays are ``(n_scm_levels,)`` on the SCM's TOP-TO-BOTTOM
    levels, with entries outside the LES domain left as NaN and excluded by
    ``mask``.
    """
    case: str
    source_dir: str
    n_frames: int
    window_hours: tuple[float, float]     # (t_start, t_end) of the mean, hours
    z_scm: np.ndarray                     # (nlev,) [m], top-to-bottom
    mask: np.ndarray                      # (nlev,) bool, inside the LES domain
    weights: np.ndarray                   # (nlev,) mass weights, sum 1 on mask
    profiles: dict[str, np.ndarray]       # name -> (nlev,)
    z_les: np.ndarray                     # (n_les,) [m] native LES levels
    profiles_les: dict[str, np.ndarray]   # name -> (n_les,) native, for plots

    @property
    def window_label(self) -> str:
        """Human-readable averaging window, for printing next to any number."""
        t0, t1 = self.window_hours
        return f"{t0:.2f}-{t1:.2f} h ({self.n_frames} frames)"

    def scored_variables(self) -> tuple[str, ...]:
        return tuple(v for v in REFERENCE_VARIABLES if v in self.profiles)


def mass_weights_from_pressure(p_half: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalized layer-mass weights ``dp/sum(dp)``, zero outside ``mask``.

    Weighting by layer mass rather than by level count keeps the score from
    being dominated by wherever the vertical grid happens to be finest.
    """
    p_half = np.asarray(p_half, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    dp = np.abs(np.diff(p_half))
    if dp.shape != mask.shape:
        raise ValueError(
            f"p_half must have one more entry than mask; got dp {dp.shape} "
            f"vs mask {mask.shape}."
        )
    w = np.where(mask, dp, 0.0)
    total = w.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            "mass weights sum to zero: the LES-domain mask selects no SCM level."
        )
    return w / total


def _frame_time_hours(payload) -> float:
    if "t_hours" not in payload.files:
        raise ValueError("LES profile frame has no 't_hours' entry.")
    return float(payload["t_hours"])


def _read_frames(prof_dir: Path):
    files = sorted(prof_dir.glob("prof_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no LES profile frames in {prof_dir}. Run the LES driver with "
            "--record-frames > 0 first."
        )
    frames = []
    for path in files:
        with np.load(path, allow_pickle=True) as payload:
            frames.append((
                _frame_time_hours(payload),
                path,
                {k: np.asarray(payload[k]) for k in payload.files},
            ))
    frames.sort(key=lambda item: item[0])
    return frames


def _time_mean(frames, name: str) -> np.ndarray:
    """Mean of ``name`` over ``frames``; a non-finite entry is FATAL."""
    stack = []
    for t_hours, path, payload in frames:
        arr = np.asarray(payload[name], dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"LES frame {path.name} (t={t_hours:.3f} h) has non-finite "
                f"'{name}'. Refusing to average it away — the LES run is bad."
            )
        stack.append(arr)
    return np.mean(np.stack(stack, axis=0), axis=0)


def _interp_to_scm(z_les: np.ndarray, values: np.ndarray,
                   z_scm: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Interpolate an LES profile onto the SCM levels inside the LES domain.

    ``z_les`` ascends (spectral LES grid); ``z_scm`` descends (SCM top-to-bottom).
    Levels outside the mask come back NaN so a downstream bug that ignores the
    mask fails loudly instead of scoring against an extrapolation.
    """
    out = np.full(z_scm.shape, np.nan, dtype=np.float64)
    if mask.any():
        out[mask] = np.interp(z_scm[mask], z_les, values)
    return out


def load_les_reference(
    les_dir: str | Path,
    *,
    case: str,
    z_scm: np.ndarray,
    p_half: np.ndarray,
    domain_top_m: float,
    analysis_hours: float,
    min_frames: int = 2,
) -> LESReference:
    """Build a time-mean LES reference on the SCM levels.

    Parameters
    ----------
    les_dir
        LES output directory (the one containing ``profiles/``).
    z_scm
        SCM full-level heights [m], top-to-bottom.
    p_half
        SCM half-level pressures [Pa], ``(nlev + 1,)``, for the mass weights.
    domain_top_m
        Top of the LES domain [m]; SCM levels above it are excluded.
    analysis_hours
        Length of the TRAILING window averaged over. The window actually used
        is recorded on the returned reference.
    min_frames
        Refuse to build a reference from fewer frames than this — a one-frame
        "time mean" is an instantaneous snapshot wearing a mean's name.
    """
    les_dir = Path(les_dir)
    prof_dir = les_dir / "profiles"
    frames = _read_frames(prof_dir)

    t_end = frames[-1][0]
    t_start = t_end - float(analysis_hours)
    window = [f for f in frames if f[0] >= t_start - 1e-9]
    if len(window) < min_frames:
        raise ValueError(
            f"analysis window {t_start:.3f}-{t_end:.3f} h contains "
            f"{len(window)} frame(s), need >= {min_frames}. The LES wrote "
            f"{len(frames)} frames spanning {frames[0][0]:.3f}-{t_end:.3f} h; "
            "lengthen --analysis-hours or rerun the LES with more "
            "--record-frames."
        )

    available = set(window[0][2].keys())
    for _t, path, payload in window:
        if set(payload.keys()) != available:
            raise ValueError(
                f"LES frame {path.name} has a different variable set than the "
                "first frame in the window; refusing to average inconsistent "
                "frames."
            )

    z_les = _time_mean(window, "z")
    if np.any(np.diff(z_les) <= 0.0):
        raise ValueError("LES z must be strictly ascending.")

    z_scm = np.asarray(z_scm, dtype=np.float64)
    if np.any(np.diff(z_scm) >= 0.0):
        raise ValueError("z_scm must be top-to-bottom (strictly descending).")
    # Only levels the LES actually resolves: inside its domain AND not below
    # its lowest cell centre (the SCM's lowest level can sit under it).
    mask = (z_scm <= float(domain_top_m)) & (z_scm >= float(z_les[0]))
    if not mask.any():
        raise ValueError(
            f"no SCM level lies inside the LES domain "
            f"[{z_les[0]:.1f}, {domain_top_m:.1f}] m; SCM spans "
            f"[{z_scm[-1]:.1f}, {z_scm[0]:.1f}] m."
        )
    weights = mass_weights_from_pressure(p_half, mask)

    wanted = [v for v in REFERENCE_VARIABLES + _OPTIONAL_VARIABLES
              if v in available]
    missing_required = [v for v in REFERENCE_VARIABLES if v not in available]
    if "theta" in missing_required or "wth" in missing_required:
        raise ValueError(
            f"LES frames in {prof_dir} lack {missing_required}; they predate "
            "the turbulent-flux recording in les_record.py. Rerun the LES."
        )

    profiles_les, profiles = {}, {}
    for name in wanted:
        native = _time_mean(window, name)
        if native.shape != z_les.shape:
            continue                      # scalar diagnostics, not profiles
        profiles_les[name] = native
        profiles[name] = _interp_to_scm(z_les, native, z_scm, mask)

    return LESReference(
        case=case,
        source_dir=str(les_dir),
        n_frames=len(window),
        window_hours=(float(window[0][0]), float(t_end)),
        z_scm=z_scm,
        mask=mask,
        weights=weights,
        profiles=profiles,
        z_les=z_les,
        profiles_les=profiles_les,
    )
