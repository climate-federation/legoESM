#!/usr/bin/env python
"""Generate NMC background error samples and fit GEN_BE covariance.

Uses the NMC method (Parrish & Derber 1992): background errors are estimated
from differences between 48h and 24h forecasts verifying at the same time:

    error_proxy(T_v) = f_48h(T_v - 48h) - f_24h(T_v - 24h)

Both forecasts verify at T_v. Collecting N such differences and calling
``fit_gen_be`` produces the GEN_BE background error covariance parameters,
which are saved to disk for use in 4D-Var via GenBETransform.

Model: spectral primitive equation (Gaussian grid) with full physics via
ModelDriver — gray radiation (Frierson et al. 2006), Rayleigh friction in
the planetary boundary layer, and analytical SST boundary condition.
ERA5 data: a local Zarr cache of ERA5 pressure-level fields. The
``--era5`` flag points at this Zarr store; see ``era5_to_state.py`` for
the lat-lon-to-grid ingestion path that consumes it.

Vertical levels: the ``--nlev`` flag sets the forecast model's vertical
resolution.  20 levels is a reasonable minimum for testing; production runs
should use 37+ to match ERA5 pressure-level spacing.

Usage
-----
# Requires a local ERA5 Zarr cache at the path passed to ``--era5``.

# Generate NMC samples and fit GEN_BE:
JAX_ENABLE_X64=1 python -m legoesm.da.generate_nmc \\
    --era5 /data/era5/era5_20190101_20201231.zarr \\
    --output outputs/gen_be \\
    --start 2020-01-01 --end 2020-12-31 \\
    --interval-days 5 \\
    --n-max 21 --nlev 37

# Quick test with a short period:
JAX_ENABLE_X64=1 python -m legoesm.da.generate_nmc \\
    --era5 /data/era5/era5_test.zarr \\
    --output outputs/gen_be_test \\
    --start 2020-01-01 --end 2020-03-01 \\
    --interval-days 15 --n-max 21 --nlev 10

Output
------
Writes two files:
  <output>.npz   — GenBEParams arrays (vert_eig_vec, reg_coeff, len_scale, ...)
  <output>.json  — static metadata (tracer_names, n_levels)

Load with:
  from legoesm.da.gen_be import load_gen_be_params
  params = load_gen_be_params("outputs/gen_be")
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# JAX x64 must be enabled before any JAX imports
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class NMCConfig(NamedTuple):
    """Configuration for NMC background error generation."""
    era5_zarr_path: str               # Local ERA5 Zarr cache (see era5_to_state.py)
    output_path: str                  # Base path for output (no extension)
    date_start: str                   # Verification range start "YYYY-MM-DD"
    date_end: str                     # Verification range end
    interval_days: int = 5            # Spacing between verification times [days]
    n_max: int = 21                   # Spectral truncation (T21 ≈ 2.8°)
    nlev: int = 20                    # Vertical levels (37+ recommended for production)
    dt: float = 1800.0                # Model time step [s]
    vert_corr_length: float = 2.0     # Vertical correlation length [levels]
    default_len_scale_km: float = 500.0  # Fallback horizontal length scale [km]
    forecast_48h: int = 48            # Long forecast length [h]; standard is 48h
    forecast_24h: int = 24            # Short forecast length [h]; standard is 24h


# ---------------------------------------------------------------------------
# ERA5 ↔ spectral state conversion
# ---------------------------------------------------------------------------

def _era5_to_spectral(era5, grid, sigma, include_tracers: bool = True):
    """Convert an ERA5Slice to SpectralHydrostaticState on a Gaussian grid.

    Pipeline:
    1. Regrid ERA5 lat-lon → Gaussian grid (linear interpolation)
    2. Vertical interpolation: pressure levels → sigma levels
    3. (u, v) → spectral (vorticity, divergence) via SH analysis
    4. T, ln(p_s), phis → spectral
    5. When ``include_tracers=True`` (default), package the regridded
       ERA5 specific humidity as a grid-space ``q_v`` ``Field`` on
       ``state.tracers``.  The dycore RHS will then advect q_v through
       the forecast and any moisture-aware physics (gray radiation,
       Kessler, ...) will consume it.

    Parameters
    ----------
    era5 : ERA5Slice
    grid : GaussianGrid
    sigma : SigmaCoordinate
    include_tracers : bool
        Whether to attach ``q_v`` as a grid-space tracer on the
        returned state.  Default ``True`` enables the moisture-aware
        NMC pipeline; pass ``False`` to recover the pre-tracer dry
        pipeline (matches the legacy behavior).

    Returns
    -------
    SpectralHydrostaticState
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralHydrostaticState
    from legoesm.core.field import Field
    from legoesm.grids.gaussian import (
        sh_analysis,
        sh_analysis_3d,
        sh_analysis_oc2_3d,
        sh_analysis_dmu_3d,
    )
    from legoesm.training.era5_to_state import (
        regrid_latlon_to_gaussian,
        regrid_2d_to_gaussian,
    )
    from legoesm.training.vertical_interp import interp_pressure_to_sigma

    T_ll, u_ll, v_ll, q_ll, p_s_ll = regrid_latlon_to_gaussian(era5, grid)
    phis_ll = regrid_2d_to_gaussian(era5.phis, era5.lat, era5.lon, grid)

    # 2. Vertical interpolation
    sigma_f = jnp.array(sigma.sigma_full, dtype=jnp.float64)
    plev = jnp.array(era5.plev_Pa, dtype=jnp.float64)
    p_s = jnp.array(p_s_ll, dtype=jnp.float64)

    T_model = interp_pressure_to_sigma(
        jnp.array(T_ll, dtype=jnp.float64), plev, p_s, sigma_f
    )
    u_model = interp_pressure_to_sigma(
        jnp.array(u_ll, dtype=jnp.float64), plev, p_s, sigma_f
    )
    v_model = interp_pressure_to_sigma(
        jnp.array(v_ll, dtype=jnp.float64), plev, p_s, sigma_f
    )
    phis_jax = jnp.array(phis_ll, dtype=jnp.float64)

    # 3. SH analysis: u,v → vor/div
    a = float(grid.radius)
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    cos_lat_3d = grid.cos_lat[:, None, None]
    u_cos = u_model * cos_lat_3d
    v_cos = v_model * cos_lat_3d

    vor_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, v_cos)
        + one_over_a * sh_analysis_dmu_3d(grid, u_cos)
    )
    div_hat = (
        im_over_a[:, None] * sh_analysis_oc2_3d(grid, u_cos)
        - one_over_a * sh_analysis_dmu_3d(grid, v_cos)
    )
    T_hat = sh_analysis_3d(grid, T_model)
    lnps_hat = sh_analysis(grid, jnp.log(p_s))
    phis_hat = sh_analysis(grid, phis_jax)

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)
    grid_dims_3d = ("lat", "lon", "level")

    tracers = None
    if include_tracers:
        # Vertical interpolation for q_v (clip to ≥ 0; ERA5 occasionally
        # has tiny negative values from the interpolator).
        q_model = jnp.maximum(
            interp_pressure_to_sigma(
                jnp.array(q_ll, dtype=jnp.float64), plev, p_s, sigma_f,
            ),
            0.0,
        )
        tracers = {
            "q_v": Field(
                data=q_model, name="q_v",
                dims=grid_dims_3d, units="kg/kg",
            ),
        }

    return SpectralHydrostaticState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
        phis_hat=Field(data=phis_hat, name="phis_hat", dims=dims_2d, units="m2/s2"),
        tracers=tracers,
    )


def _spectral_to_hydrostatic(spec_state, grid, sigma, include_tracers=False):
    """Convert SpectralHydrostaticState → HydrostaticState on Gaussian grid.

    When ``include_tracers=True`` and ``spec_state.tracers`` is non-
    empty, the grid-space tracer dict is forwarded to the
    HydrostaticState (matches the dycore-side spectral PE convention
    that tracers stay in grid space).  Default ``False`` preserves the
    pre-tracer dry pipeline.

    Parameters
    ----------
    spec_state : SpectralHydrostaticState
    grid : GaussianGrid
    sigma : SigmaCoordinate
    include_tracers : bool
        If True, copy ``spec_state.tracers`` into the returned
        HydrostaticState's ``tracers`` field.

    Returns
    -------
    HydrostaticState
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_to_grid
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    gp = spectral_pe_to_grid(spec_state, grid, sigma)

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    tracers_out = None
    if include_tracers and spec_state.tracers is not None:
        # Tracers are already grid-space on the spectral PE side, so
        # forward them as-is; this preserves Field-vs-raw container types.
        tracers_out = dict(spec_state.tracers)

    return HydrostaticState(
        u=Field(data=gp["u"],   name="u",   dims=dims_3d, units="m/s"),
        v=Field(data=gp["v"],   name="v",   dims=dims_3d, units="m/s"),
        T=Field(data=gp["T"],   name="T",   dims=dims_3d, units="K"),
        p_s=Field(data=gp["p_s"], name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=gp["phis"], name="phis", dims=dims_2d, units="m2/s2"),
        tracers=tracers_out,
    )


# ---------------------------------------------------------------------------
# Full-physics forecast helpers
# ---------------------------------------------------------------------------

class _DayRef:
    """Mutable container for the current simulation day.

    Passed into the physics closure so the day advances across loop
    iterations without rebuilding the function object.
    """
    __slots__ = ("day",)

    def __init__(self, day: float = 0.0):
        self.day = day


def _build_spectral_physics_fn(driver, gray_config, n_levels: int):
    """Build the gray radiation + Rayleigh friction physics tendency function.

    Mirrors the physics used in ModelDriver._run_spectral: gray radiation
    (Frierson et al. 2006) with analytical SST boundary and Rayleigh
    friction in the planetary boundary layer (σ > σ_b).

    Parameters
    ----------
    driver : ModelDriver
        Fully setup() driver; provides grid, sigma, config, get_sst_sic.
    gray_config : GrayRadiationConfig
    n_levels : int
        Number of vertical sigma levels.

    Returns
    -------
    physics_fn : callable(state, grid, sigma_coord) -> SpectralHydrostaticState
        Spectral tendency function; pass to model.step(physics_fn=...).
    day_ref : _DayRef
        Mutable day reference; set day_ref.day before each model step.
    """
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        spectral_pe_to_grid,
        SpectralHydrostaticState,
    )
    from legoesm.grids.gaussian import (
        sh_analysis_3d,
        sh_analysis_oc2_3d,
        sh_analysis_dmu_3d,
    )
    from legoesm.atmosphere.physics.radiation.gray import gray_radiation
    from legoesm.atmosphere.physics.radiation.solar import daily_mean_insolation
    from legoesm.forcing.surface_utils import blend_surface_temperature

    grid = driver.grid
    sigma = driver.sigma
    cfg = driver.config

    sigma_full = jnp.array(sigma.sigma_full)

    # Rayleigh friction profile — same as ModelDriver._run_spectral
    K_F = 1.0 / 86400.0
    SIGMA_B = cfg.sigma_b
    k_f = K_F * jnp.maximum(0.0, (sigma_full - SIGMA_B) / (1.0 - SIGMA_B))

    shape_2d = (grid.n_lat, grid.n_lon)
    shape_3d = (*shape_2d, n_levels)
    ncol = shape_2d[0] * shape_2d[1]

    a = float(grid.radius)
    _im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    _one_over_a = 1.0 / a
    cos_lat_3d = grid.cos_lat[:, None, None]

    T_ice = cfg.T_ice
    S_0 = cfg.S_0

    grid_lat = grid.grid_lat
    lat_2d = (
        jnp.broadcast_to(grid_lat[:, None], shape_2d)
        if grid_lat.ndim == 1 else grid_lat
    )
    lat_col = lat_2d.reshape(-1)

    day_ref = _DayRef(0.0)

    def physics_fn(state, grid_arg, sigma_coord, forcing_data=None):
        """Spectral tendency: gray radiation + Rayleigh friction.

        Iter-96 fix: when ``forcing_data`` is provided, reads
        ``day``, ``sst``, ``sic``, ``insol`` from the TRACED
        pytree.  This avoids the iter-74 ``_DayRef`` JIT-cache
        stale-day pathology where the closure-captured day was
        baked at first trace.  Falls back to ``day_ref.day``
        for backward compatibility with existing 3-arg callers.
        """
        fields = spectral_pe_to_grid(state, grid_arg, sigma_coord)
        T_g = fields["T"]
        u_g = fields["u"]
        v_g = fields["v"]
        p_s_g = fields["p_s"]

        # SST/SIC boundary condition.  When forcing_data is provided
        # (iter-96 migration to the new API), use the TRACED values
        # — JAX retraces ONCE at first call but subsequent values
        # propagate as dynamic inputs.  Otherwise fall back to the
        # legacy ``day_ref.day`` closure read (which has the iter-74
        # JIT-cache stale-day issue, but preserves backward compat).
        if forcing_data is not None and "sst" in forcing_data:
            sst = forcing_data["sst"]
            sic = forcing_data["sic"]
            current_day = forcing_data["day"]
        else:
            sst, sic = driver.get_sst_sic(day_ref.day)
            current_day = day_ref.day
        if sst.ndim == 1:
            sst = jnp.broadcast_to(sst[:, None], shape_2d)
            sic = jnp.broadcast_to(sic[:, None], shape_2d)
        T_sfc = blend_surface_temperature(sst, sic, T_ice)

        # Pressure arrays
        p_full = p_s_g[..., None] * sigma_full
        p_half = p_s_g[..., None] * sigma_coord.sigma_half
        T_col = T_g.reshape(ncol, n_levels)
        p_full_col = p_full.reshape(ncol, n_levels)
        p_half_col = p_half.reshape(ncol, n_levels + 1)
        # Pull q_v from the spectral state's tracer dict when present
        # (PR1's spectral PE tracers).  Falling back to zero matches
        # the pre-tracer dry pipeline.
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = jnp.maximum(_qv_data.reshape(ncol, n_levels), 0.0)
        else:
            q_v_col = jnp.zeros_like(T_col)
        T_sfc_col = T_sfc.reshape(ncol)

        # Gray radiation — use traced day from forcing_data when supplied
        if forcing_data is not None and "insol" in forcing_data:
            insol = forcing_data["insol"]
        else:
            insol = daily_mean_insolation(lat_col, current_day, S_0)
        rad_out = gray_radiation(
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col,
            q_v=q_v_col, insolation=insol, config=gray_config,
        )
        dT_dt_rad = rad_out.heating_rate.reshape(shape_3d)

        # Rayleigh friction
        du_dt = -k_f * u_g
        dv_dt = -k_f * v_g

        # Spectral analysis of tendencies
        du_cos = du_dt * cos_lat_3d
        dv_cos = dv_dt * cos_lat_3d
        dvor_hat = (
            _im_over_a[:, None] * sh_analysis_oc2_3d(grid_arg, dv_cos)
            + _one_over_a * sh_analysis_dmu_3d(grid_arg, du_cos)
        )
        ddiv_hat = (
            _im_over_a[:, None] * sh_analysis_oc2_3d(grid_arg, du_cos)
            - _one_over_a * sh_analysis_dmu_3d(grid_arg, dv_cos)
        )
        dT_hat = sh_analysis_3d(grid_arg, dT_dt_rad)

        # Mirror the input state's tracer pytree as zero tendencies so
        # the dycore RHS sees a consistent structure (this matches what
        # the spectral PE orchestrator does in
        # ``_make_spectral_pe_combined``).  Gray radiation + Rayleigh
        # friction don't move tracers themselves; the dycore advection
        # does, plus any microphysics in the orchestrator path.
        from legoesm.atmosphere.physics._shared import zero_like_tracers
        return SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=dvor_hat),
            div_hat=state.div_hat.replace(data=ddiv_hat),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=jnp.zeros_like(state.lnps_hat.data)),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            tracers=zero_like_tracers(state.tracers),
        )

    return physics_fn, day_ref


def _run_forecast_with_physics(
    driver,
    physics_fn,
    day_ref: _DayRef,
    ic_spectral,
    n_hours: int,
    start_day: float = 0.0,
):
    """Run the spectral model with full physics for n_hours.

    Uses gray radiation + Rayleigh friction, mirroring ModelDriver._run_spectral.
    ``day_ref.day`` is advanced at each step so the radiation and SST
    interpolation use the correct simulation time.

    Parameters
    ----------
    driver : ModelDriver
        Fully-initialized driver (provides model, grid, sigma, config).
    physics_fn : callable
        Physics tendency function from _build_spectral_physics_fn.
    day_ref : _DayRef
        Mutable day reference; updated at each step.
    ic_spectral : SpectralHydrostaticState
        Initial condition.
    n_hours : int
        Forecast length in hours.
    start_day : float
        Simulation day at the IC (for SST/solar forcing).

    Returns
    -------
    SpectralHydrostaticState at T + n_hours.
    """
    dt = float(driver.config.dycore.dt)
    n_steps = int(n_hours * 3600 / dt)

    logger.debug(
        "Running %dh forecast with physics (%d steps, dt=%.0fs)",
        n_hours, n_steps, dt,
    )

    # Cold start: clear leapfrog history so each forecast is independent
    driver.model._state_prev = None

    state = ic_spectral
    # Iter-96 migration: build TRACED forcing_data each step (day,
    # sst, sic, insol as JAX arrays) and pass through the iter-92/95
    # ``model.step(forcing_data=...)`` API.  JAX traces over the
    # array values dynamically — single compile, dynamic forcing.
    # Closes the iter-74 ``_DayRef`` JIT-cache stale-day pathology
    # for the NMC forecast path.
    grid = driver.grid
    grid_lat = grid.grid_lat
    lat_2d = (
        jnp.broadcast_to(grid_lat[:, None], (grid.n_lat, grid.n_lon))
        if grid_lat.ndim == 1 else grid_lat
    )
    lat_col_local = lat_2d.reshape(-1)
    S_0_local = driver.config.S_0
    # iter-169: import daily_mean_insolation here — F821 NameError
    # on this branch otherwise (the line-310 import inside
    # ``_build_spectral_physics_fn`` does not reach this scope).
    from legoesm.atmosphere.physics.radiation.solar import (
        daily_mean_insolation,
    )
    for step in range(n_steps):
        current_day = start_day + (step + 1) * dt / 86400.0
        day_ref.day = current_day  # backward-compat: keep _DayRef synced
        # Build TRACED forcing_data
        sst, sic = driver.get_sst_sic(current_day)
        insol = daily_mean_insolation(lat_col_local, current_day, S_0_local)
        forcing_data = {
            "day": jnp.asarray(current_day),
            "sst": sst,
            "sic": sic,
            "insol": insol,
        }
        state = driver.model.step(
            state, dt, physics_fn=physics_fn, forcing_data=forcing_data,
        )

    return state


# ---------------------------------------------------------------------------
# NMC difference
# ---------------------------------------------------------------------------

def _hydrostatic_diff(state_a, state_b):
    """Compute state_a - state_b as a HydrostaticState (NMC error proxy).

    phis is taken from state_a (static field, difference ≈ 0).

    When both ``state_a.tracers`` and ``state_b.tracers`` are non-empty
    dicts, per-key differences are returned in the result's ``tracers``
    field.  Tracers present in only one operand are dropped (the NMC
    error proxy requires both 48h and 24h forecasts to carry the same
    tracer set).  When either side has ``tracers=None`` the result also
    has ``tracers=None`` (matches the legacy dry pipeline).

    Parameters
    ----------
    state_a, state_b : HydrostaticState

    Returns
    -------
    HydrostaticState  (difference)
    """
    def _diff(fa, fb):
        return fa.replace(data=fa.data - fb.data)

    def _diff_value(va, vb):
        # Duck-type Field-vs-raw container so the helper handles both.
        if hasattr(va, "data") and hasattr(va, "replace"):
            return va.replace(data=va.data - vb.data)
        return va - vb

    tracers_diff = None
    if state_a.tracers is not None and state_b.tracers is not None:
        common_keys = set(state_a.tracers.keys()) & set(state_b.tracers.keys())
        if common_keys:
            tracers_diff = {
                k: _diff_value(state_a.tracers[k], state_b.tracers[k])
                for k in common_keys
            }

    return state_a._replace(
        u=_diff(state_a.u, state_b.u),
        v=_diff(state_a.v, state_b.v),
        T=_diff(state_a.T, state_b.T),
        p_s=_diff(state_a.p_s, state_b.p_s),
        tracers=tracers_diff,
    )


# ---------------------------------------------------------------------------
# Time index lookup
# ---------------------------------------------------------------------------

def _build_time_index(zarr_path: str):
    """Open ERA5 Zarr and return (datetime_list, time_array) for index lookups."""
    import xarray as xr

    logger.info("Opening ERA5 store to read time coordinate: %s", zarr_path)
    ds = xr.open_zarr(zarr_path, chunks=None)
    times = ds.time.values
    dt_list = [
        datetime.utcfromtimestamp(int(t) / 1e9)
        for t in times.astype("datetime64[ns]").astype("int64")
    ]
    return dt_list, times


def _find_time_idx(dt_list, target_dt: datetime) -> int:
    """Return the ERA5 time index closest to target_dt (within ±3h)."""
    target_ts = target_dt.timestamp()
    best_idx = None
    best_diff = float("inf")
    for i, dt in enumerate(dt_list):
        diff = abs(dt.timestamp() - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    if best_diff > 3 * 3600:
        raise ValueError(
            f"No ERA5 time within 3h of {target_dt}. "
            f"Closest: {dt_list[best_idx]} ({best_diff/3600:.1f}h away). "
            "Ensure your ERA5 store covers this date range."
        )
    return best_idx


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_nmc(config: NMCConfig) -> None:
    """Run NMC background error generation and fit GEN_BE parameters.

    Steps:
    1. Build ModelDriver with spectral PE + full physics (gray radiation +
       Rayleigh friction + analytical SST boundary condition).
    2. For each verification time T_v in [date_start, date_end]:
       a. Load ERA5 at T_v - 48h → run 48h forecast → physical state f48
       b. Load ERA5 at T_v - 24h → run 24h forecast → physical state f24
       c. NMC error = f48 - f24
    3. fit_gen_be on all error samples
    4. save_gen_be_params to output_path
    """
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
    from legoesm.training.era5_to_state import TrainingERA5Config, load_era5_slice
    from legoesm.da.gen_be import fit_gen_be, save_gen_be_params

    # -- 1. Build ModelDriver with full spectral physics --
    tmp_dir = Path(config.output_path).parent / "_nmc_driver_tmp"
    exp_config = ExperimentConfig(
        grid=GridConfig(
            grid_type="gaussian",
            resolution=config.n_max,
            nlev=config.nlev,
            vertical_coord="sigma",
        ),
        dycore=DycoreConfig(
            model_type="spectral_pe",
            discretization="spectral",
            dt=config.dt,
        ),
        output=OutputConfig(
            output_dir=str(tmp_dir),
            diag_days=0,
        ),
        days=2,           # placeholder — NMC uses its own time loop
        radiation="gray",
        convection="none",
        turbulence="none",
        dataset="analytical",
        precision="fp64",
    )

    logger.info(
        "Initializing ModelDriver: T%d, %d levels, dt=%.0fs, "
        "gray radiation + Rayleigh friction",
        config.n_max, config.nlev, config.dt,
    )
    driver = ModelDriver(exp_config, output_dir=str(tmp_dir))
    driver.setup()

    grid = driver.grid
    sigma = driver.sigma

    logger.info(
        "Grid: %d×%d Gaussian, %d spectral modes",
        grid.n_lat, grid.n_lon, grid.n_sh,
    )

    gray_config = GrayRadiationConfig()
    physics_fn, day_ref = _build_spectral_physics_fn(driver, gray_config, config.nlev)

    # -- 2. ERA5 time index --
    era5_cfg = TrainingERA5Config(local_cache_dir=config.era5_zarr_path)
    dt_list, _times = _build_time_index(config.era5_zarr_path)

    t_start = datetime.strptime(config.date_start, "%Y-%m-%d")
    t_end = datetime.strptime(config.date_end, "%Y-%m-%d")
    delta_long = timedelta(hours=config.forecast_48h)
    delta_short = timedelta(hours=config.forecast_24h)

    verification_times = []
    t = t_start
    while t <= t_end:
        verification_times.append(t)
        t += timedelta(days=config.interval_days)

    logger.info(
        "Verification times: %d samples from %s to %s (every %dd)",
        len(verification_times),
        config.date_start, config.date_end,
        config.interval_days,
    )

    # -- 3. NMC generation loop --
    nmc_errors = []
    skipped = 0

    for i, t_v in enumerate(verification_times):
        t_ic48 = t_v - delta_long
        t_ic24 = t_v - delta_short

        logger.info(
            "[%d/%d] T_v: %s | IC48: %s | IC24: %s",
            i + 1, len(verification_times),
            t_v.strftime("%Y-%m-%d %HZ"),
            t_ic48.strftime("%Y-%m-%d %HZ"),
            t_ic24.strftime("%Y-%m-%d %HZ"),
        )

        try:
            idx48 = _find_time_idx(dt_list, t_ic48)
            idx24 = _find_time_idx(dt_list, t_ic24)
        except ValueError as e:
            logger.warning("Skipping %s: %s", t_v, e)
            skipped += 1
            continue

        era5_48 = load_era5_slice(era5_cfg, idx48)
        era5_24 = load_era5_slice(era5_cfg, idx24)

        ic48 = _era5_to_spectral(era5_48, grid, sigma)
        ic24 = _era5_to_spectral(era5_24, grid, sigma)

        logger.info("  Running %dh forecast...", config.forecast_48h)
        f48_spec = _run_forecast_with_physics(
            driver, physics_fn, day_ref, ic48, config.forecast_48h,
        )
        jax.block_until_ready(f48_spec.T_hat.data)

        logger.info("  Running %dh forecast...", config.forecast_24h)
        f24_spec = _run_forecast_with_physics(
            driver, physics_fn, day_ref, ic24, config.forecast_24h,
        )
        jax.block_until_ready(f24_spec.T_hat.data)

        f48_phys = _spectral_to_hydrostatic(
            f48_spec, grid, sigma, include_tracers=True,
        )
        f24_phys = _spectral_to_hydrostatic(
            f24_spec, grid, sigma, include_tracers=True,
        )

        error = _hydrostatic_diff(f48_phys, f24_phys)
        nmc_errors.append(error)

        # Fuse the RMS reductions into one host transfer.  When the
        # error sample carries a q_v tracer (moisture-aware NMC), also
        # report rms(q_v) so the user can monitor moisture-error
        # magnitudes alongside dry diagnostics.
        rms_arrays = [
            jnp.sqrt(jnp.mean(error.T.data ** 2)),
            jnp.sqrt(jnp.mean(error.u.data ** 2)),
            jnp.sqrt(jnp.mean(error.p_s.data ** 2)),
        ]
        has_q_v = (
            error.tracers is not None
            and "q_v" in error.tracers
        )
        if has_q_v:
            _qv_raw = error.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            rms_arrays.append(jnp.sqrt(jnp.mean(_qv_data ** 2)))
        _h = np.asarray(jnp.stack(rms_arrays))
        if has_q_v:
            logger.info(
                "  Error sample %d: rms(T)=%.3f K, rms(u)=%.3f m/s, "
                "rms(p_s)=%.1f Pa, rms(q_v)=%.2e kg/kg",
                len(nmc_errors),
                float(_h[0]),
                float(_h[1]),
                float(_h[2]),
                float(_h[3]),
            )
        else:
            logger.info(
                "  Error sample %d: rms(T)=%.3f K, rms(u)=%.3f m/s, "
                "rms(p_s)=%.1f Pa",
                len(nmc_errors),
                float(_h[0]),
                float(_h[1]),
                float(_h[2]),
            )

    if not nmc_errors:
        raise RuntimeError(
            "No NMC samples generated. "
            f"Check that ERA5 store covers {config.date_start} to {config.date_end} "
            f"plus ±{config.forecast_48h}h."
        )

    logger.info(
        "Generated %d NMC error samples (%d skipped due to missing ERA5 data)",
        len(nmc_errors), skipped,
    )

    # -- 4. Fit GEN_BE --
    logger.info("Fitting GEN_BE background error covariance...")
    params = fit_gen_be(
        nmc_errors,
        grid,
        vert_corr_length=config.vert_corr_length,
        default_len_scale_km=config.default_len_scale_km,
    )

    logger.info(
        "GEN_BE fit: nlev=%d, n_channels=%d, std_ps=%.1f Pa, "
        "len_scale=[%.0f, %.0f] km",
        params.n_levels,
        params.reg_coeff.shape[0],
        float(params.std_ps),
        float(params.len_scale.min()) / 1000,
        float(params.len_scale.max()) / 1000,
    )

    # -- 5. Save --
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_gen_be_params(params, out_path)
    logger.info(
        "Saved GEN_BE params to %s.npz + %s.json", out_path, out_path,
    )

    print("\n" + "=" * 60)
    print("NMC GEN_BE generation complete")
    print(f"  Samples:    {len(nmc_errors)}")
    print(f"  Levels:     {params.n_levels}")
    print(f"  Channels:   {params.reg_coeff.shape[0]}")
    print(f"  std_ps:     {float(params.std_ps):.1f} Pa")
    print(f"  len_scale:  [{float(params.len_scale.min())/1000:.0f}, "
          f"{float(params.len_scale.max())/1000:.0f}] km")
    print(f"  Output:     {out_path}.npz + .json")
    print("=" * 60)
    print("\nTo use in 4D-Var:")
    print("  from legoesm.da.gen_be import load_gen_be_params, GenBETransform")
    print(f"  params = load_gen_be_params('{out_path}')")
    print("  B = GenBETransform(params, spec, grid)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate NMC background error samples and fit GEN_BE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--era5", required=True, metavar="ZARR_PATH",
        help="Local ERA5 Zarr store (see era5_to_state.py for ingestion)",
    )
    p.add_argument(
        "--output", required=True, metavar="BASE_PATH",
        help="Output base path without extension, e.g. outputs/gen_be",
    )
    p.add_argument(
        "--start", required=True, metavar="YYYY-MM-DD",
        help="Verification time range start",
    )
    p.add_argument(
        "--end", required=True, metavar="YYYY-MM-DD",
        help="Verification time range end",
    )
    p.add_argument(
        "--interval-days", type=int, default=5, metavar="N",
        help="Spacing between verification times in days (default: 5)",
    )
    p.add_argument(
        "--n-max", type=int, default=21,
        help="Spectral truncation — T21 ≈ 2.8° resolution (default: 21)",
    )
    p.add_argument(
        "--nlev", type=int, default=20,
        help="Number of vertical sigma levels (default: 20; use 37+ for production)",
    )
    p.add_argument(
        "--dt", type=float, default=1800.0,
        help="Model time step in seconds (default: 1800)",
    )
    p.add_argument(
        "--forecast-48h", type=int, default=48,
        help="Long forecast length in hours (default: 48; some centers use 36)",
    )
    p.add_argument(
        "--forecast-24h", type=int, default=24,
        help="Short forecast length in hours (default: 24; some centers use 12)",
    )
    p.add_argument(
        "--vert-corr-length", type=float, default=2.0,
        help="Vertical exponential correlation length in levels (default: 2.0)",
    )
    p.add_argument(
        "--default-len-scale-km", type=float, default=500.0,
        help="Fallback horizontal length scale [km] for channels with low variance (default: 500)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    config = NMCConfig(
        era5_zarr_path=args.era5,
        output_path=args.output,
        date_start=args.start,
        date_end=args.end,
        interval_days=args.interval_days,
        n_max=args.n_max,
        nlev=args.nlev,
        dt=args.dt,
        forecast_48h=args.forecast_48h,
        forecast_24h=args.forecast_24h,
        vert_corr_length=args.vert_corr_length,
        default_len_scale_km=args.default_len_scale_km,
    )
    generate_nmc(config)


if __name__ == "__main__":
    main()
