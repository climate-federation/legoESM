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
    "SCORED_VARIABLES",
    "DIAGNOSTIC_VARIABLES",
    "load_les_reference",
    "mass_weights_from_pressure",
]

# Variables the SCM is SCORED against: the mean state the closure controls.
#
# The turbulent fluxes are deliberately NOT here. The SCM never exposes a
# per-level w'theta': the hydrostatic turbulence driver
# (turbulence/integration.py, the branch that calls each kernel) consumes only
# du_dt/dv_dt/dT_dt/dq_v_dt and cloud_fraction from TurbulenceOutput and drops
# Km, Kh, shflx, lhflx, ustar and h_pbl on the floor, and PhysicsState has no
# slot for them. Exactly one of the nine schemes (prognostic CLUBB) carries a
# genuine w'theta_l' in its packed moment state, so a flux score would be
# available for one scheme and unavailable for eight -- which is not a
# controlled comparison. Under an IDENTICAL forcing and surface BC the mean
# state is set by the flux divergence anyway, so matching the state profile is
# the well-posed uniform target.
SCORED_VARIABLES: tuple[str, ...] = ("theta", "qv", "u", "v")

# Carried on the reference for plotting and diagnosis, never scored.
DIAGNOSTIC_VARIABLES: tuple[str, ...] = (
    "wth", "wqv", "qc", "cloud_frac", "tke", "uw", "vw", "ww",
)


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
        return tuple(v for v in SCORED_VARIABLES if v in self.profiles)


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
    # Normalisation goes through the SHARED helper rather than a local
    # dp/sum(dp): zeroing the masked levels first and normalising the result
    # gives masked-and-renormalised weights in one canonical call.
    from legoesm.training.column_era5_metrics import normalized_mass_weights
    masked = np.where(mask, dp, 0.0)
    if not np.isfinite(masked.sum()) or masked.sum() <= 0.0:
        raise ValueError(
            "mass weights sum to zero: the LES-domain mask selects no SCM level."
        )
    return np.asarray(normalized_mass_weights(masked), dtype=np.float64)


def _frame_time_hours(payload: dict) -> float:
    """``payload`` is the frame's contents as a plain dict (already read off
    disk, so the npz handle is closed before this runs)."""
    if "t_hours" not in payload:
        raise ValueError("LES profile frame has no 't_hours' entry.")
    return float(payload["t_hours"])


def _read_frames(prof_dir: Path, *, expect_case: str | None = None):
    files = sorted(prof_dir.glob("prof_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"no LES profile frames in {prof_dir}. Run the LES driver with "
            "--record-frames > 0 first."
        )
    frames = []
    for path in files:
        with np.load(path, allow_pickle=True) as payload:
            data = {k: np.asarray(payload[k]) for k in payload.files}
        # The frames record which case wrote them. Without this check,
        # `--case rico --les-dir results/les_ref/bomex` loads BOMEX profiles,
        # maps them with RICO geometry and forcing, and labels the result RICO.
        if expect_case is not None:
            # REQUIRED, not optional: a frame with no label, or an empty one,
            # previously sailed through, so a stale frame from another case
            # that happened to lack its label was mapped onto this column and
            # scored as if it belonged here.
            if "case" not in data or not str(data["case"]).strip():
                raise ValueError(
                    f"{path.name} carries no 'case' label, so it cannot be "
                    f"confirmed to belong to {expect_case!r}. Rerun the LES "
                    "(every driver writes the label) rather than scoring "
                    "against an unidentified reference."
                )
            wrote = str(data["case"])
            if not _case_matches(wrote, expect_case):
                raise ValueError(
                    f"{path.name} was written by LES case {wrote!r} but this "
                    f"reference is being built for {expect_case!r}. Pointing "
                    "--les-dir at another case's output would silently score "
                    "against the wrong reference."
                )
        frames.append((_frame_time_hours(data), path, data))
    frames.sort(key=lambda item: item[0])
    # Directories are reused and only matching frame indices get overwritten,
    # so a shorter rerun leaves higher-index files from an EARLIER run behind.
    # Averaging those silently mixes two runs, and duplicate timestamps get
    # double counted, so both are refused rather than warned about.
    times = [t for t, _p, _d in frames]
    for (t_a, path_a), (t_b, path_b) in zip(
        [(t, p) for t, p, _ in frames], [(t, p) for t, p, _ in frames][1:]
    ):
        if abs(t_b - t_a) <= 1.0e-9:
            raise ValueError(
                f"LES frames {path_a.name} and {path_b.name} share t_hours="
                f"{t_a:.6f}. The output directory holds frames from more than "
                "one run; delete it and rerun the LES rather than averaging "
                "two runs together."
            )
    del times
    return frames


# Deck-directory spellings that legitimately identify a case. An explicit
# table, NOT a prefix test: `startswith` accepted "b" and "bomex_experiment"
# for "bomex", which is exactly how a mislabelled or unrelated frame gets in.
_CASE_ALIASES: dict[str, frozenset[str]] = {
    "bomex": frozenset({"bomex"}),
    "rico": frozenset({"rico"}),
    "dycoms": frozenset({"dycoms", "dycoms_rf01", "dycoms_rf02", "dycomsii"}),
    "gabls1": frozenset({"gabls1"}),
    "wangara": frozenset({"wangara"}),
}


def _case_matches(wrote: str, expect: str) -> bool:
    """Exact match against the case name or one of its registered aliases."""
    a, b = wrote.strip().lower(), expect.strip().lower()
    return a == b or a in _CASE_ALIASES.get(b, frozenset())


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
    frames = _read_frames(prof_dir, expect_case=case)

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

    # Every frame in the window must be on the SAME vertical grid. Averaging
    # by index otherwise silently mixes runs with different Lz or nz that
    # happen to share a level COUNT -- the mean would be over two grids.
    z_ref_frame = np.asarray(window[0][2]["z"], dtype=np.float64)
    for _t, path, payload in window[1:]:
        z_i = np.asarray(payload["z"], dtype=np.float64)
        if z_i.shape != z_ref_frame.shape or not np.allclose(
            z_i, z_ref_frame, rtol=0.0, atol=1.0e-6
        ):
            raise ValueError(
                f"LES frame {path.name} is on a different vertical grid than "
                f"{window[0][1].name} (z differs). The directory mixes runs; "
                "delete it and rerun the LES rather than averaging two grids."
            )
    z_les = _time_mean(window, "z")
    if np.any(np.diff(z_les) <= 0.0):
        raise ValueError("LES z must be strictly ascending.")

    z_scm = np.asarray(z_scm, dtype=np.float64)
    if np.any(np.diff(z_scm) >= 0.0):
        raise ValueError("z_scm must be top-to-bottom (strictly descending).")
    # Only levels the LES actually RESOLVES: between its lowest and highest
    # cell centres, and inside the requested domain. The upper bound is
    # z_les[-1], not domain_top_m: np.interp right-CLAMPS, so a level between
    # the top LES cell centre and the lid would otherwise be scored against a
    # copied top-cell value rather than being excluded.
    top = min(float(domain_top_m), float(z_les[-1]))
    mask = (z_scm <= top) & (z_scm >= float(z_les[0]))
    if not mask.any():
        raise ValueError(
            f"no SCM level lies inside the LES-resolved range "
            f"[{z_les[0]:.1f}, {top:.1f}] m; SCM spans "
            f"[{z_scm[-1]:.1f}, {z_scm[0]:.1f}] m."
        )
    weights = mass_weights_from_pressure(p_half, mask)

    wanted = [v for v in SCORED_VARIABLES + DIAGNOSTIC_VARIABLES
              if v in available]
    missing_scored = [v for v in SCORED_VARIABLES if v not in available]
    if missing_scored:
        raise ValueError(
            f"LES frames in {prof_dir} lack the scored variable(s) "
            f"{missing_scored}; available: {sorted(available)}. A dry case has "
            "no 'qv' and cannot score a moist case's variables."
        )
    if "wth" not in available:
        raise ValueError(
            f"LES frames in {prof_dir} have no 'wth'; they predate the "
            "turbulent-flux recording in les_record.py. Rerun the LES so the "
            "reference carries the fluxes for diagnosis."
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
