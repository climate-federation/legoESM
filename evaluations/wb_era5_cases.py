"""Build WB2 forecast-eval cases (ICs + verification + forcing) from ERA5.

The CLI layer (``scripts/validate/run_weatherbench_eval.py``) calls this to turn
a WeatherBench-2 ERA5 Zarr into a list of :class:`~evaluations.wb_orchestrator.ForecastCase`:

* **IC** — built with the trainer's own ``era5_to_spectral_carry`` chain (so the
  eval starts from the SAME spectral state ingestion training used), then lifted
  to a ``SpectralHydrostaticState`` via ``carry_to_spectral_state`` so it matches
  ``ForecastCase.init_state`` and the orchestrator's default spectral rollout.
* **forcing** — the prescribed-SST ``forcing_base`` dict the learned-physics
  rollout consumes, built exactly like ``scale_build._load_era5_samples_spectral``
  (flat ``T_sfc`` regridded to the Gaussian grid, ``sic=0``, 1-based integer
  ``day_of_year`` + ``seconds_of_day`` via the promoted
  ``scale_build.era5_time_to_forcing_calendar``).
* **verification** — the WB2 headline fields read DIRECTLY from the ERA5
  pressure-level / surface variables (via ``resolve_var``), mapped to
  ``HEADLINE_FIELD_KEYS`` and regridded to the WB2 grid with ``regrid_to_wb2``
  (matching the model-side ``diagnose_and_regrid``). Valid masks are all-True;
  the scorer intersects them with the model's below-ground masks.

Network-free: every entry point accepts a pre-opened ``ds=`` dataset, so a
synthetic in-memory dataset drives the tests without touching GCS.

Fail-loud policy (never silently skip / zero-fill): a lead that is not a multiple
of the ERA5 cadence, a lead/init beyond the store's time axis, a missing ERA5
variable, or a missing pressure level all RAISE.
"""
from __future__ import annotations

import numpy as np

from legoesm import constants

from .wb_forecast import HEADLINE_FIELD_KEYS, score_forecast
from .wb_orchestrator import ForecastCase
from .wb_regrid import regrid_to_wb2, wb2_grid

__all__ = [
    "build_forecast_cases",
    "climatology_from_cases",
    "climatology_from_era5",
    "climatology_scorecard",
]

# WB2 headline pressure-level keys -> (ERA5 variable, level [hPa]). z500 is the
# geopotential at 500 hPa divided by g (height, m) to match the model side's
# ``geopotential_height_at``; the rest are the field at the target level.
_PLEVEL_KEYS = {
    "z500": ("geopotential", 500),
    "t850": ("temperature", 850),
    "q700": ("specific_humidity", 700),
    "u850": ("u_component_of_wind", 850), "v850": ("v_component_of_wind", 850),
    "u700": ("u_component_of_wind", 700), "v700": ("v_component_of_wind", 700),
    "u500": ("u_component_of_wind", 500), "v500": ("v_component_of_wind", 500),
    "u250": ("u_component_of_wind", 250), "v250": ("v_component_of_wind", 250),
}
# WB2 headline surface keys -> ERA5 variable (read straight off the store).
_SURFACE_KEYS = {
    "mslp": "mean_sea_level_pressure",
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
}


def _resolve_or_raise(ds, name):
    from legoesm.training.era5_to_state import resolve_var

    resolved = resolve_var(ds, name)
    if resolved is None:
        raise ValueError(
            f"wb_era5_cases: required ERA5 verification variable {name!r} not "
            f"found in the store; available: {sorted(map(str, ds.data_vars))}. "
            "A missing verification field must NOT be silently skipped.")
    return resolved


def _level_dim(ds):
    return "level" if "level" in ds.dims else "pressure_level"


def _read_plevel_native(ds_t, ds, var, level_hpa):
    """(n_lat, n_lon) ERA5 field for ``var`` at ``level_hpa`` on the native grid."""
    resolved = _resolve_or_raise(ds_t, var)
    ldim = _level_dim(ds)
    levels = np.asarray(ds[ldim].values, dtype=np.float64)
    if not np.any(np.isclose(levels, float(level_hpa))):
        raise ValueError(
            f"wb_era5_cases: pressure level {level_hpa} hPa (needed for {var!r}) "
            f"not in the store levels {levels.tolist()}.")
    da = ds_t[resolved].sel({ldim: level_hpa}).transpose("lat", "lon")
    return np.asarray(da.values, dtype=np.float64)


def _read_surface_native(ds_t, var):
    """(n_lat, n_lon) ERA5 surface field on the native grid."""
    resolved = _resolve_or_raise(ds_t, var)
    da = ds_t[resolved].transpose("lat", "lon")
    return np.asarray(da.values, dtype=np.float64)


def _build_verification(ds, time_idx, resolution_deg):
    """WB2-grid headline verification (fields + all-True valid) at ``time_idx``.

    Reads the native ERA5 headline fields, computes ``wind_speed_10m`` as the
    hypot on the NATIVE grid (then regrids), matching the model-side ordering,
    and bilinearly regrids every field to the WB2 grid via ``regrid_to_wb2``.
    """
    ds_t = ds.isel(time=int(time_idx))
    src_lat = np.asarray(ds.lat.values, dtype=np.float64)   # degrees
    src_lon = np.asarray(ds.lon.values, dtype=np.float64)   # degrees, [0, 360)

    native = {}
    for key, (var, level) in _PLEVEL_KEYS.items():
        field = _read_plevel_native(ds_t, ds, var, level)
        if key == "z500":
            field = field / constants.g   # geopotential [m^2/s^2] -> height [m]
        native[key] = field
    for key, var in _SURFACE_KEYS.items():
        native[key] = _read_surface_native(ds_t, var)
    # hypot on the native grid THEN regrid (matches diagnose_headline_fields).
    native["wind_speed_10m"] = np.hypot(native["u10"], native["v10"])

    fields_wb2, valid_wb2 = {}, {}
    for key in HEADLINE_FIELD_KEYS:
        f_wb2, _, _ = regrid_to_wb2(
            native[key], src_lat, src_lon, resolution_deg=resolution_deg)
        fields_wb2[key] = f_wb2
        # All-True: ERA5 verification is valid everywhere; the scorer intersects
        # this with the MODEL's below-ground pressure-level masks.
        valid_wb2[key] = np.ones(f_wb2.shape, dtype=bool)
    return fields_wb2, valid_wb2


def build_forecast_cases(
    era5_cfg, grid, sigma, *, leads_hours, eval_year, n_inits,
    init_stride_hours, resolution_deg, ds=None, smoothing_passes=4,
):
    """Build the WB2 forecast-eval cases for one eval year.

    Parameters
    ----------
    era5_cfg : TrainingERA5Config
        ERA5 store + level + cadence config. ``era5_cfg.dt_hours`` is the store
        cadence; the IC uses ``era5_cfg.levels`` for the vertical interpolation.
    grid, sigma : Gaussian grid + sigma coordinate of the spectral training core.
    leads_hours : iterable of int
        Forecast lead times [h]. Each MUST be a multiple of the ERA5 cadence.
    eval_year : int
        Calendar year whose ``01-01`` anchors the first init.
    n_inits : int
        Number of initializations, spaced ``init_stride_hours`` apart.
    init_stride_hours : int
        Hours between successive inits (must be a multiple of the cadence).
    resolution_deg : float
        WB2 target-grid resolution (identical to the orchestrator's).
    ds : xarray.Dataset, optional
        Pre-opened ERA5 dataset (network-free tests). ``None`` opens
        ``era5_cfg.zarr_store``.

    Returns
    -------
    list[ForecastCase]
        Each with ``init_state`` (SpectralHydrostaticState), ``verif_by_lead``
        (WB2-grid ERA5 verification per lead), and ``forcing`` (the prescribed-SST
        ``forcing_base`` dict).
    """
    import jax.numpy as jnp
    from legoesm.training.era5_to_state import (
        era5_to_spectral_carry,
        load_era5_slice,
        open_era5_zarr,
        regrid_2d_to_gaussian,
    )
    from legoesm.training.neural_gcm_spectral import carry_to_spectral_state
    from legoesm.training.scale_build import era5_time_to_forcing_calendar

    cadence = int(era5_cfg.dt_hours)
    leads = [int(x) for x in leads_hours]
    for lead in leads:
        if lead <= 0 or lead % cadence != 0:
            raise ValueError(
                f"lead {lead} h is not a positive multiple of the ERA5 cadence "
                f"({cadence} h); the verification snapshot would fall between "
                "store times.")
    if init_stride_hours % cadence != 0:
        raise ValueError(
            f"--init-stride-hours {init_stride_hours} must be a multiple of the "
            f"ERA5 cadence ({cadence} h).")

    if ds is None:
        ds = open_era5_zarr(era5_cfg.zarr_store)
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    n_times = len(times)

    base = int(np.searchsorted(times, np.datetime64(f"{eval_year}-01-01")))
    if base >= n_times:
        raise ValueError(
            f"eval_year {eval_year} starts beyond the ERA5 store (last time "
            f"{times[-1]}); pick an eval year the store covers.")
    init_stride = init_stride_hours // cadence

    cases = []
    for k in range(int(n_inits)):
        i_ic = base + k * init_stride
        if i_ic >= n_times:
            raise ValueError(
                f"init {k} (time index {i_ic}) is beyond the ERA5 store "
                f"(n_times={n_times}); reduce --n-inits or --init-stride-hours.")

        ic_slice = load_era5_slice(era5_cfg, i_ic, ds=ds)
        # ``smoothing_passes`` is exposed (default 4 = unchanged) so the
        # orography treatment can be SWEPT. The ERA5 phis is smoothed to be
        # representable at the model truncation and ``p_s`` is hydrostatically
        # reconciled to it, and that reconciliation was measured to account for
        # 76% of the fixed 6 h surface-pressure offset — so varying it is the
        # perturbation test for that attribution.
        init_carry = era5_to_spectral_carry(
            ic_slice, grid, sigma, smoothing_passes=smoothing_passes)
        init_state = carry_to_spectral_state(init_carry, grid)

        # Prescribed-SST forcing on the Gaussian grid (flat ncol), 1-based
        # integer doy + seconds-of-day (scale_build calendar convention).
        sst = jnp.asarray(regrid_2d_to_gaussian(
            ic_slice.sst, ic_slice.lat, ic_slice.lon, grid)).reshape(-1)
        doy_1based, sod = era5_time_to_forcing_calendar(times[i_ic], eval_year)
        forcing = {
            "T_sfc": sst,
            "sic": jnp.zeros_like(sst),
            "day_of_year": jnp.asarray(doy_1based),
            "seconds_of_day": jnp.asarray(sod),
        }

        verif_by_lead = {}
        for lead in leads:
            i_verif = i_ic + lead // cadence
            if i_verif >= n_times:
                raise ValueError(
                    f"lead {lead} h from init {k} (verification time index "
                    f"{i_verif}) is beyond the ERA5 store (n_times={n_times}); "
                    "shorten --leads or pick earlier inits.")
            fields_wb2, valid_wb2 = _build_verification(
                ds, i_verif, resolution_deg)
            verif_by_lead[int(lead)] = {"fields": fields_wb2, "valid": valid_wb2}

        cases.append(ForecastCase(
            init_state=init_state, verif_by_lead=verif_by_lead, forcing=forcing))
    return cases


def climatology_from_era5(era5_cfg, *, eval_year, n_samples=73,
                          resolution_deg, ds=None):
    """Annual-mean climatology sampled EVENLY across ``eval_year``.

    Use this, not :func:`climatology_from_cases`, whenever the number is going
    to be reported. The window-mean alternative is only self-consistent when the
    inits already span a year.

    WHY (measured 2026-07-28, `scripts/tmp/diag_wb2_persistence_selfcheck.py`):
    with the default scorecard sampling of 8 consecutive inits from 01-01, the
    eval-window mean is a local 8-day mean, so it lands *below* the persistence
    floor — z500 24 h climatology 58.7 m vs persistence 63.9 m, which is
    physically impossible for a real climatology at day 1. Sampled evenly across
    the year instead, the same quantity is 107.8 m and window-mean and year-mean
    converge (107.6 vs 107.8). The old floor was ~1.8x too low and made every
    arm look closer to climatology than it is.

    Still NOT the published WB2 reference, which is a 1990-2019 day-of-year
    climatology (that removes the seasonal cycle, so it scores lower — 83.6 m
    for z500 after the /g conversion). This one retains the seasonal cycle
    because it is a single annual mean. Comparable across our own arms under
    identical sampling; quote the published number when comparing to the
    leaderboard.

    Only verification snapshots are built (no spectral initial conditions), so
    the cost is ``n_samples`` ERA5 reads.

    Parameters
    ----------
    era5_cfg : TrainingERA5Config
    eval_year : int
    n_samples : int
        Snapshots spread over the year (73 ~ every 5 days).
    resolution_deg : float
        WB2 target grid, identical to the scorecard's.
    ds : xarray.Dataset, optional
        Pre-opened ERA5 (network-free tests).

    Returns
    -------
    dict[str, array] — same keys/grid as ``climatology_from_cases``.
    """
    from legoesm.training.era5_to_state import open_era5_zarr

    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    if ds is None:
        ds = open_era5_zarr(era5_cfg.zarr_store)
    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    n_times = len(times)
    cadence = int(era5_cfg.dt_hours)

    base = int(np.searchsorted(times, np.datetime64(f"{eval_year}-01-01")))
    if base >= n_times:
        raise ValueError(
            f"eval_year {eval_year} starts beyond the ERA5 store (last time "
            f"{times[-1]}).")
    # Bound the sample set by TIMESTAMP, not by an assumed regular stride.
    # An index stride silently (a) walks into the NEXT year on a truncated or
    # gappy store, mixing years into a "2017 annual" mean, (b) mis-strides a
    # leap year, and (c) returns a partial-year mean with no error at all
    # (codex review 2026-07-30). Selecting on the real time axis makes all
    # three impossible, and a short year is reported rather than hidden.
    year_end = np.datetime64(f"{int(eval_year) + 1:04d}-01-01")
    stop = int(np.searchsorted(times, year_end))
    in_year = stop - base
    if in_year <= 0:
        raise ValueError(
            f"climatology_from_era5: the store holds no snapshots inside "
            f"{eval_year} (base={base}, next-year index={stop}).")
    stride = max(1, in_year // n_samples)

    accum, n = {}, 0
    for k in range(n_samples):
        i = base + k * stride
        if i >= stop:          # never cross into the following year
            break
        fields_wb2, _valid = _build_verification(ds, i, resolution_deg)
        for key, field in fields_wb2.items():
            f = np.asarray(field, dtype=np.float64)
            accum[key] = f if key not in accum else accum[key] + f
        n += 1
    if n == 0:
        raise ValueError(
            "climatology_from_era5: no snapshots inside the ERA5 store for "
            f"eval_year={eval_year}.")
    # Coverage is part of the result: a mean built from a third of the year is
    # a seasonal mean wearing an annual label. Report it rather than let the
    # caller assume full coverage.
    span_days = float(
        (times[min(base + (n - 1) * stride, stop - 1)] - times[base])
        / np.timedelta64(1, "D"))
    if n < n_samples or span_days < 300.0:
        import warnings

        warnings.warn(
            f"climatology_from_era5({eval_year}): {n}/{n_samples} snapshots "
            f"spanning {span_days:.0f} days — this is NOT a full-year mean; "
            "treat it as a seasonal reference.")
    return {key: accum[key] / float(n) for key in accum}


def climatology_from_cases(cases):
    """Eval-window sample-mean climatology over all verification snapshots (v1).

    PREFER :func:`climatology_from_era5` for any reported number — with inits
    that do not span a year this mean is a LOCAL mean and scores below the
    persistence floor (see that function's docstring for the measured numbers).

    ``clim[key] = mean_over_(case, lead) verif["fields"][key]``. Self-consistent
    for ranking checkpoints under identical sampling, but NOT comparable to the
    published WB2 ACC climatology (1990-2019 hourly) — see the CLI ``meta`` note.
    """
    accum, n = {}, 0
    for case in cases:
        for verif in case.verif_by_lead.values():
            for key, field in verif["fields"].items():
                f = np.asarray(field, dtype=np.float64)
                accum[key] = f if key not in accum else accum[key] + f
            n += 1
    if n == 0:
        raise ValueError(
            "climatology_from_cases: no verification snapshots to average.")
    return {key: accum[key] / float(n) for key in accum}


def climatology_scorecard(cases, leads_hours, clim_fields_wb2, *, resolution_deg):
    """Score the (lead-independent) climatology field set against each case/lead.

    A must-beat floor computed with the SAME ``score_forecast`` primitive as the
    model path (RMSE/ACC/bias), averaged over cases. Uses the verification's
    all-True valid mask (climatology has no below-ground concept), so unlike the
    model/persistence scorecards it does not exclude below-ground cells — a v1
    caveat flagged in the CLI ``meta``.
    """
    wb2_lat, _ = wb2_grid(resolution_deg)
    metrics = ("rmse", "acc", "bias")
    accum: dict = {}
    for case in cases:
        for lead in leads_hours:
            verif = case.verif_by_lead[int(lead)]
            scores = score_forecast(
                clim_fields_wb2, verif["fields"], clim_fields_wb2, wb2_lat,
                valid=verif["valid"])
            for key, m in scores.items():
                accum.setdefault((key, int(lead)), []).append(m)
    return {
        kl: {name: float(np.mean([d[name] for d in per_case])) for name in metrics}
        for kl, per_case in accum.items()
    }
