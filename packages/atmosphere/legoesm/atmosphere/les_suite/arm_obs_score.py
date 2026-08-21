"""Score an ARM-forced SCM run against the ARM observation reference.

The observational-campaign analogue of :mod:`les_suite.score` (which scores an SCM
against LES truth).  Here the reference is REAL observations
(:class:`legoesm.atmosphere.forcing.scm.sccm_arm.ARMObsReference`): the ARM SGP
sounding array (temperature/humidity/wind profiles) plus MWR liquid-water path,
GOES cloud fraction and precipitation.  Two pieces:

* :func:`build_arm_comparables` runs the obs-forced SCM over a time window and
  samples its output at the observation times + pressure levels, returning an
  :class:`ARMComparables` (host-side numpy).  It reuses the shared pressure
  interpolation (`scm_forcing_io.interp_profile_to_pressure`) and the canonical
  column integral (`diagnostics.column_water_vapor`, applied to the cloud-liquid
  tracer for LWP) — no re-derived numerics.
* :func:`score_arm_obs` computes a NaN-aware, per-channel normalized RMSE and an
  RMS-combined score, mirroring :mod:`les_suite.score`'s design: each channel is
  normalized by the observation's own (floored) mass-weighted spread, so the
  score is dimensionless and comparable across channels.  Missing observations
  (NaN) are masked, never zero-filled (which would flatter the fit).

Scope (Phase 2): the runner fills the thermodynamic/dynamic soundings (theta, q,
u, v) and LWP.  Cloud-fraction and precipitation comparables are left ``None``
(the SCM emits neither a fractional cloud diagnostic nor a surface precip flux in
its history yet) — :func:`score_arm_obs` already scores them when supplied, so
they are a drop-in for a later phase.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np
from legoesm.atmosphere.forcing.scm.sccm_arm import ARMObsReference, ARMSCMCase
from legoesm.atmosphere.forcing.scm.scm_forcing_io import interp_profile_to_pressure
from legoesm.atmosphere.physics._shared import exner_function
from legoesm.diagnostics.column_integrals import column_water_vapor

# Normalizing-spread floors (mirror les_suite.score), in each channel's own units,
# so a near-constant observation cannot blow up the normalized RMSE.
_FLOOR_THETA_K = 0.1          # K
_FLOOR_Q = 1.0e-4            # kg/kg (mixing ratio)
_FLOOR_WIND = 0.1            # m/s
_FLOOR_LWP = 1.0e-3         # kg/m^2
_FLOOR_CLOUD = 0.05         # cloud fraction
_FLOOR_PRECIP = 0.5         # mm/day
_TIME_MATCH_TOL_S = 1.0     # obs<->comparable time-match tolerance [s]


@dataclass(frozen=True)
class ARMComparables:
    """SCM output sampled at the observation times + pressure levels (numpy).

    ``theta``/``q``/``u``/``v`` are ``(ntime, nlev)`` on ``pressure_pa`` [Pa];
    ``q`` is a water-vapour mixing ratio (matching the obs ``q``).  The scalar
    series are ``(ntime,)``.  ``None`` marks a channel the runner did not fill.
    """

    time_seconds: np.ndarray
    pressure_pa: np.ndarray
    theta: np.ndarray | None = None
    q: np.ndarray | None = None
    u: np.ndarray | None = None
    v: np.ndarray | None = None
    lwp_kg_m2: np.ndarray | None = None
    total_cloud_fraction: np.ndarray | None = None
    precip_mm_day: np.ndarray | None = None


@dataclass(frozen=True)
class ARMObsScore:
    """Per-channel + combined normalized RMSE of an SCM run vs ARM observations."""

    theta_rmse: float | None
    q_rmse: float | None
    u_rmse: float | None
    v_rmse: float | None
    lwp_rmse: float | None
    cloud_rmse: float | None
    precip_rmse: float | None
    combined: float | None
    channels: tuple[str, ...]


# --- masked normalized metrics (host-side; NaN = missing obs, never zero-filled) ---
def _pressure_thickness_weights(pressure_pa: np.ndarray) -> np.ndarray:
    """Mass (|dp|) layer weights over the obs pressure levels, summing to 1.

    Mass weighting is the natural column weighting (consistent with the
    ``column_water_vapor`` mass integral); it needs no height conversion.
    """
    p = np.asarray(pressure_pa, dtype=np.float64)
    edges = np.empty(p.size + 1)
    edges[1:-1] = 0.5 * (p[:-1] + p[1:])
    edges[0] = p[0] - 0.5 * (p[1] - p[0])
    edges[-1] = p[-1] + 0.5 * (p[-1] - p[-2])
    w = np.abs(np.diff(edges))
    total = w.sum()
    return w / total if total > 0 else np.full_like(w, 1.0 / w.size)


def _profile_scale(obs: np.ndarray, weights: np.ndarray, floor: float) -> float:
    """Floored mass-weighted spread of the observation series (one scale/channel).

    Uses the time-mean obs profile's weighted std over finite levels (mirrors
    ``score._normalized_rmse_series``: a single, stable normalizer per channel).
    """
    if not np.isfinite(obs).any():        # all-missing obs -> no scale, use the floor
        return floor
    with np.errstate(invalid="ignore"):
        mean_prof = np.nanmean(obs, axis=0)
    finite = np.isfinite(mean_prof)
    if finite.sum() < 2:
        return floor
    w = weights[finite]
    w = w / w.sum()
    mu = np.sum(w * mean_prof[finite])
    var = np.sum(w * (mean_prof[finite] - mu) ** 2)
    return max(float(np.sqrt(max(var, 0.0))), floor)


_MODEL_FAILURE = (
    "model produced no finite {kind} values where the observation exists; a "
    "diverged / all-NaN model must not be silently dropped from the score (that "
    "would let producing NaNs improve the combined score)"
)


def _masked_profile_rmse(
    scm: np.ndarray, obs: np.ndarray, pressure_pa: np.ndarray, floor: float
) -> float | None:
    """NaN-aware normalized profile RMSE for a ``(nt, nlev)`` series.

    Returns ``None`` when the OBSERVATION is entirely absent (channel not scored).
    Per time: mass-weighted RMS over the levels where BOTH scm and obs are finite
    (weights renormalized over the valid subset), normalized by the obs-only
    scale; then RMS over the times that had >=1 valid level.  Raises if the obs
    exists but the MODEL is non-finite everywhere the obs is valid (a model
    failure must not be silently omitted).
    """
    scm = np.asarray(scm, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    if not np.isfinite(obs).any():
        return None
    weights = _pressure_thickness_weights(pressure_pa)
    scale = _profile_scale(obs, weights, floor)
    per_time_ms: list[float] = []
    for t in range(obs.shape[0]):
        valid = np.isfinite(scm[t]) & np.isfinite(obs[t])
        if not valid.any():
            continue
        w = weights[valid]
        w = w / w.sum()
        diff = (scm[t, valid] - obs[t, valid]) / scale
        per_time_ms.append(float(np.sum(w * diff**2)))
    if not per_time_ms:
        raise ValueError(_MODEL_FAILURE.format(kind="profile"))
    return float(np.sqrt(np.mean(per_time_ms)))


def _masked_series_rmse(scm: np.ndarray, obs: np.ndarray, floor: float) -> float | None:
    """NaN-aware normalized RMSE for a ``(nt,)`` scalar series.

    ``None`` when the OBSERVATION is absent; raises if the obs exists but the
    MODEL is non-finite everywhere the obs is valid.  The normalizing scale is
    the OBSERVATION's own spread (over its finite samples), independent of where
    the model happens to be finite.
    """
    scm = np.asarray(scm, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    obs_finite = np.isfinite(obs)
    if obs_finite.sum() < 1:
        return None
    valid = obs_finite & np.isfinite(scm)
    if valid.sum() < 1:
        raise ValueError(_MODEL_FAILURE.format(kind="series"))
    obs_v = obs[obs_finite]
    scale = max(float(np.std(obs_v)), floor) if obs_v.size >= 2 else floor
    diff = (scm[valid] - obs[valid]) / scale
    return float(np.sqrt(np.mean(diff**2)))


def _obs_at_times(times_obs: np.ndarray, times_want: np.ndarray) -> np.ndarray:
    """Indices into the obs time axis matching ``times_want`` (exact, within tol)."""
    idx = np.searchsorted(times_obs, times_want)
    idx = np.clip(idx, 0, times_obs.size - 1)
    # snap to the nearer of idx-1/idx, then require an exact-within-tol match.
    for j, tw in enumerate(times_want):
        i = idx[j]
        if i > 0 and abs(times_obs[i - 1] - tw) < abs(times_obs[i] - tw):
            idx[j] = i - 1
    if np.any(np.abs(times_obs[idx] - times_want) > _TIME_MATCH_TOL_S):
        raise ValueError(
            "comparable times do not align with observation times within tolerance; "
            "build_arm_comparables must sample at obs.time_seconds"
        )
    return idx


def score_arm_obs(obs: ARMObsReference, comp: ARMComparables) -> ARMObsScore:
    """Score SCM comparables against ARM observations (per-channel + RMS-combined).

    ``comp.time_seconds`` must be a subset of ``obs.time_seconds`` (the runner
    samples the SCM at the obs times); obs rows are matched by time.
    """
    idx = _obs_at_times(np.asarray(obs.time_seconds, dtype=np.float64),
                        np.asarray(comp.time_seconds, dtype=np.float64))
    p_obs = np.asarray(comp.pressure_pa, dtype=np.float64)
    p_ref = np.asarray(obs.pressure_pa, dtype=np.float64)
    # The obs profiles are on obs.pressure_pa; scoring them with the comparable's
    # pressure grid (Exner + mass weights) is only valid if the grids match.
    if p_obs.shape != p_ref.shape or not np.allclose(p_obs, p_ref, rtol=0.0, atol=1.0):
        raise ValueError(
            "comp.pressure_pa must match obs.pressure_pa (build_arm_comparables "
            "samples the SCM onto the obs pressure levels)")
    exner = np.asarray(exner_function(p_obs), dtype=np.float64)

    scores: dict[str, float] = {}

    def _profile(name, scm_prof, obs_prof, floor):
        if scm_prof is None or obs_prof is None:
            return None
        r = _masked_profile_rmse(scm_prof, np.asarray(obs_prof)[idx], p_obs, floor)
        if r is not None:
            scores[name] = r
        return r

    def _series(name, scm_ser, obs_ser, floor):
        if scm_ser is None or obs_ser is None:
            return None
        r = _masked_series_rmse(np.asarray(scm_ser), np.asarray(obs_ser)[idx], floor)
        if r is not None:
            scores[name] = r
        return r

    # obs T (temperature) -> theta on the obs pressure levels, matching comp.theta.
    obs_theta = None if obs.T is None else np.asarray(obs.T) / exner[None, :]
    theta_rmse = _profile("theta", comp.theta, obs_theta, _FLOOR_THETA_K)
    q_rmse = _profile("q", comp.q, obs.q, _FLOOR_Q)
    u_rmse = _profile("u", comp.u, obs.u, _FLOOR_WIND)
    v_rmse = _profile("v", comp.v, obs.v, _FLOOR_WIND)
    lwp_rmse = _series("lwp", comp.lwp_kg_m2, obs.liquid_water_path_kg_m2, _FLOOR_LWP)
    cloud_rmse = _series(
        "cloud", comp.total_cloud_fraction, obs.total_cloud_fraction, _FLOOR_CLOUD)
    precip_rmse = _series("precip", comp.precip_mm_day, obs.precip_mm_day, _FLOOR_PRECIP)

    combined = (
        float(np.sqrt(np.mean(np.asarray(list(scores.values())) ** 2)))
        if scores else None
    )
    return ARMObsScore(
        theta_rmse=theta_rmse, q_rmse=q_rmse, u_rmse=u_rmse, v_rmse=v_rmse,
        lwp_rmse=lwp_rmse, cloud_rmse=cloud_rmse, precip_rmse=precip_rmse,
        combined=combined, channels=tuple(scores.keys()),
    )


# --- runner: SCM output sampled at obs times + pressure levels --------------------
def _time_interp(times_src: np.ndarray, values: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    """Linear-in-time interpolation of ``values`` (nt_src, ...) onto ``times_dst``."""
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(values.shape[0], -1)
    out = np.empty((times_dst.size, flat.shape[1]), dtype=np.float64)
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(times_dst, times_src, flat[:, j])
    return out.reshape((times_dst.size,) + values.shape[1:])


def build_arm_comparables(
    case: ARMSCMCase,
    physics_config,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    save_every: int = 1,
    dt: float | None = None,
    initialize_from_obs: bool = False,
    **create_overrides,
) -> ARMComparables:
    """Run the ARM-forced SCM over a window and sample it at the obs times/levels.

    The window ``[start_seconds, end_seconds]`` selects the observation times to
    score against (default: the whole IOP).  With ``initialize_from_obs`` the SCM
    is re-initialized from the observed sounding at ``start_seconds`` (the ARM
    "continuous forcing" restart that avoids multi-day free-run drift); otherwise
    it starts from the case IC (only meaningful when ``start_seconds`` is the IOP
    start).  ``dt``/``save_every``/``**create_overrides`` pass through to the SCM.

    Note: the ARM forcing prescribes surface fluxes, so ``physics_config``'s
    turbulence scheme must have its surface heat exchange disabled
    (``surface.Ch_neutral=0``) — otherwise the SCM rejects the config to avoid
    double-counting the surface flux (bulk formula + prescribed channel).
    """
    from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel

    # dt/t0_seconds set nsteps, the calendar start, and the history-time shift; a
    # create-override would desync them from the actual SCM clock.
    clashing = {"dt", "t0_seconds"} & set(create_overrides)
    if clashing:
        raise ValueError(
            f"{sorted(clashing)} cannot be passed via create_overrides; use the "
            "dt= parameter and start_seconds/end_seconds for the window")

    obs = case.obs
    t0 = case.time_seconds[0] if start_seconds is None else float(start_seconds)
    t1 = case.time_seconds[-1] if end_seconds is None else float(end_seconds)
    if t1 <= t0:
        raise ValueError(f"end_seconds ({t1}) must exceed start_seconds ({t0})")
    step = float(dt) if dt is not None else case.dt

    obs_t = np.asarray(obs.time_seconds, dtype=np.float64)
    win = (obs_t >= t0 - _TIME_MATCH_TOL_S) & (obs_t <= t1 + _TIME_MATCH_TOL_S)
    obs_times = obs_t[win]
    if obs_times.size == 0:
        raise ValueError(f"no observation times in window [{t0}, {t1}] seconds")

    kwargs = case.scm_kwargs()
    kwargs["dt"] = step
    kwargs["t0_seconds"] = t0
    if initialize_from_obs:
        kwargs.update(_ic_from_obs(case, t0, int(case.nlev), float(case.sigma_top)))
    kwargs.update(create_overrides)
    scm = SingleColumnModel.create(physics_config=physics_config, **kwargs)

    nsteps = int(np.ceil((t1 - t0) / step))
    doy0 = _day_of_year(case.base_date) + t0 / 86400.0
    sod0 = t0 % 86400.0
    # The column's INITIAL state, taken before it is advanced. The history the
    # run returns starts one timestep in -- its clock is the elapsed time at
    # the END of a step -- while the observation window starts at t0, and
    # linear interpolation clamps rather than refusing. Without this the first
    # observation was scored against the model one timestep later, a penalty a
    # perfect model could not avoid.
    _ic = {
        "T": np.asarray(scm.state.T.data[0, 0, 0], dtype=np.float64),
        "u": np.asarray(scm.state.u.data[0, 0, 0], dtype=np.float64),
        "v": np.asarray(scm.state.v.data[0, 0, 0], dtype=np.float64),
        "p_s": float(np.asarray(scm.state.p_s.data[0, 0, 0])),
    }
    for _name in ("q_v", "q_c"):
        _tr = (scm.state.tracers or {}).get(_name)
        _ic[_name] = (None if _tr is None
                      else np.asarray(_tr.data[0, 0, 0], dtype=np.float64))
    _final, hist = scm.run(
        nsteps, save_every=save_every,
        start_day_of_year=doy0 % 365.0, start_seconds_of_day=sod0,
    )

    # history is top-to-bottom on the SCM sigma grid; forcing/history time is the
    # ELAPSED clock ((k+1)*dt), so shift to absolute IOP seconds before matching.
    hist_times = np.concatenate(
        [[t0], np.asarray(hist.time, dtype=np.float64) + t0])

    def _with_ic(name, arr):
        """Prepend the initial state so the history spans the window's start."""
        if arr is None:
            return None
        head = _ic[name]
        if head is None:
            return None
        return np.concatenate([np.asarray(head)[None, ...],
                               np.asarray(arr, dtype=np.float64)], axis=0)

    p_s_hist = _with_ic("p_s", np.asarray(hist.p_s, dtype=np.float64))
    sigma = scm.sigma_coord

    def _to_obs_levels(field_hist):
        """(nt_hist, nlev_scm) SCM field -> (nt_obs, nlev_obs) on obs pressures."""
        vals_t = _time_interp(hist_times, np.asarray(field_hist), obs_times)
        p_s_t = np.interp(obs_times, hist_times, p_s_hist)
        out = np.empty((obs_times.size, obs.pressure_pa.size), dtype=np.float64)
        for it in range(obs_times.size):
            p_full = np.asarray(sigma.pressure_at_full(p_s_t[it]), dtype=np.float64)
            out[it] = interp_profile_to_pressure(obs.pressure_pa, p_full, vals_t[it])
        return out, np.interp(obs_times, hist_times, p_s_hist)

    T_obsgrid, _ = _to_obs_levels(_with_ic("T", hist.T))
    exner_obs = np.asarray(exner_function(np.asarray(obs.pressure_pa)), dtype=np.float64)
    theta = T_obsgrid / exner_obs[None, :]
    q = (_to_obs_levels(_with_ic("q_v", hist.q_v))[0]
         if hist.q_v is not None else None)
    u = _to_obs_levels(_with_ic("u", hist.u))[0]
    v = _to_obs_levels(_with_ic("v", hist.v))[0]

    lwp = None
    if hist.q_c is not None:
        # canonical mass column integral applied to the cloud-liquid tracer.
        lwp_hist = np.asarray(
            column_water_vapor(_with_ic("q_c", hist.q_c),
                               p_s_hist, sigma.dsigma), dtype=np.float64)
        lwp = np.interp(obs_times, hist_times, lwp_hist)

    return ARMComparables(
        time_seconds=obs_times, pressure_pa=np.asarray(obs.pressure_pa, dtype=np.float64),
        theta=theta, q=q, u=u, v=v, lwp_kg_m2=lwp,
        total_cloud_fraction=None, precip_mm_day=None,
    )


def _ic_from_obs(case: ARMSCMCase, t_start: float, nlev: int, sigma_top: float) -> dict:
    """SCM create-kwargs from the observed sounding at ``t_start`` (obs restart)."""
    from legoesm.grids.vertical import create_sigma_coordinate

    obs = case.obs
    obs_t = np.asarray(obs.time_seconds, dtype=np.float64)
    p_lev = np.asarray(obs.pressure_pa, dtype=np.float64)

    def _at(profile):
        if profile is None:
            return None
        return _time_interp(obs_t, np.asarray(profile), np.asarray([t_start]))[0]

    T_src = _at(obs.T)
    if T_src is None:
        raise ValueError("obs restart requires an observed temperature profile")
    q_src = _at(obs.q)
    u_src = _at(obs.u)
    v_src = _at(obs.v)
    p_s = (
        float(np.interp(t_start, obs_t, obs.surface_pressure_pa))
        if obs.surface_pressure_pa is not None else case.p_s
    )
    sigma = create_sigma_coordinate(nlev, sigma_top=sigma_top)
    p_target = np.asarray(sigma.pressure_at_full(p_s), dtype=np.float64)

    def _to_grid(src, floor0=False):
        if src is None:
            return None
        prof = interp_profile_to_pressure(p_target, p_lev, src)
        return np.maximum(prof, 0.0) if floor0 else prof

    return {
        "T_profile": _to_grid(T_src),
        "q_v_profile": _to_grid(q_src, floor0=True),
        "u": _to_grid(u_src) if u_src is not None else 0.0,
        "v": _to_grid(v_src) if v_src is not None else 0.0,
        "p_s": p_s,
    }


def _day_of_year(base_date_yymmdd: int) -> float:
    """Day-of-year (1-based) for a ``yymmdd`` integer (ARM base date)."""
    if base_date_yymmdd <= 0:
        return 0.0
    yy = (base_date_yymmdd // 10000) % 100
    mm = (base_date_yymmdd // 100) % 100
    dd = base_date_yymmdd % 100
    year = 1900 + yy if yy >= 50 else 2000 + yy
    return float(datetime.date(year, mm, dd).timetuple().tm_yday)
