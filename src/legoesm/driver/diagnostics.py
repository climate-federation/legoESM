"""Diagnostic collection for the composable model driver.

Extracts the diagnostic accumulation, snapshot capture, and output
logic from run_amip.py into a reusable class.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from legoesm.diagnostics.column_integrals import column_water_vapor
from legoesm.diagnostics.energy_budget import EnergyBudgetTracker
from legoesm.forcing.time_utils import day_to_calendar
from legoesm.io.cmor_output import CMIP6_PLEV19


class _StructuredRegridWeights:
    """Precomputed bilinear interpolation weights for structured grids."""
    __slots__ = ('i_lo', 'j_lo', 'wi', 'wj', 'src_nlat', 'src_nlon')

    def __init__(self, i_lo, j_lo, wi, wj, src_nlat, src_nlon):
        self.i_lo = i_lo
        self.j_lo = j_lo
        self.wi = wi
        self.wj = wj
        self.src_nlat = src_nlat
        self.src_nlon = src_nlon


def _build_structured_regrid_weights(
    src_lat_rad: np.ndarray,
    src_lon_rad: np.ndarray,
    tgt_nlat: int,
    tgt_nlon: int,
) -> _StructuredRegridWeights:
    """Build bilinear interpolation weights from a native structured grid
    to a regular CMIP lat-lon grid.

    Parameters
    ----------
    src_lat_rad : (n_lat_src,) — source latitudes in radians, S→N
    src_lon_rad : (n_lon_src,) — source longitudes in radians, [0, 2π)
    tgt_nlat, tgt_nlon : target CMIP grid dimensions
    """
    src_lat = np.degrees(src_lat_rad)  # S→N
    src_lon = np.degrees(src_lon_rad)  # [0, 360)

    dlat = 180.0 / tgt_nlat
    dlon = 360.0 / tgt_nlon
    tgt_lat = np.linspace(-90.0 + dlat / 2, 90.0 - dlat / 2, tgt_nlat)
    tgt_lon = np.linspace(dlon / 2, 360.0 - dlon / 2, tgt_nlon)

    # For each target lat, find bracketing source lat indices + weight
    i_lo = np.searchsorted(src_lat, tgt_lat) - 1
    i_lo = np.clip(i_lo, 0, len(src_lat) - 2)
    denom_i = src_lat[i_lo + 1] - src_lat[i_lo]
    denom_i = np.where(denom_i == 0, 1.0, denom_i)
    wi = np.clip((tgt_lat - src_lat[i_lo]) / denom_i, 0.0, 1.0)

    # For each target lon, find bracketing source lon indices + weight
    j_lo = np.searchsorted(src_lon, tgt_lon) - 1
    j_lo = np.clip(j_lo, 0, len(src_lon) - 2)
    denom_j = src_lon[j_lo + 1] - src_lon[j_lo]
    denom_j = np.where(denom_j == 0, 1.0, denom_j)
    wj = np.clip((tgt_lon - src_lon[j_lo]) / denom_j, 0.0, 1.0)

    return _StructuredRegridWeights(
        i_lo=i_lo, j_lo=j_lo, wi=wi, wj=wj,
        src_nlat=len(src_lat), src_nlon=len(src_lon),
    )


def _apply_structured_regrid_2d(
    field: np.ndarray,
    w: _StructuredRegridWeights,
) -> np.ndarray:
    """Apply bilinear interpolation to a 2-D field (nlat_src, nlon_src)
    → (nlat_tgt, nlon_tgt)."""
    i0 = w.i_lo
    i1 = np.minimum(i0 + 1, w.src_nlat - 1)
    j0 = w.j_lo
    j1 = np.minimum(j0 + 1, w.src_nlon - 1)
    wi = w.wi
    wj = w.wj
    # Bilinear: f = (1-wi)(1-wj)*f00 + wi*(1-wj)*f10 + (1-wi)*wj*f01 + wi*wj*f11
    f00 = field[np.ix_(i0, j0)]
    f10 = field[np.ix_(i1, j0)]
    f01 = field[np.ix_(i0, j1)]
    f11 = field[np.ix_(i1, j1)]
    return ((1 - wi[:, None]) * (1 - wj[None, :]) * f00
            + wi[:, None] * (1 - wj[None, :]) * f10
            + (1 - wi[:, None]) * wj[None, :] * f01
            + wi[:, None] * wj[None, :] * f11)


def _apply_structured_regrid_3d(
    field: np.ndarray,
    w: _StructuredRegridWeights,
) -> np.ndarray:
    """Apply bilinear interpolation to a 3-D field (nlat_src, nlon_src, nlev)
    → (nlat_tgt, nlon_tgt, nlev)."""
    nlev = field.shape[2]
    result = np.empty((len(w.i_lo), len(w.j_lo), nlev), dtype=field.dtype)
    for k in range(nlev):
        result[:, :, k] = _apply_structured_regrid_2d(field[:, :, k], w)
    return result


class DiagnosticCollector:
    """Accumulates diagnostics during a simulation.

    Manages time-series arrays, vertical profiles, 2D snapshots,
    energy budget tracking, and optional monthly-mean accumulation.

    Parameters
    ----------
    nlev : int
        Number of vertical levels.
    sigma_full : array
        Sigma at full levels.
    dsigma : array
        Layer thickness in sigma.
    monthly_means : bool
        Enable monthly-mean zonal diagnostics.
    cmip_output : bool
        Enable CF/CMOR NetCDF output.
    clear_sky_diag : bool
        Track clear-sky radiation diagnostics.
    n_days : int
        Total simulation days (for snapshot selection).
    output_dir : str or Path
        Output directory for diagnostics and CMOR files.
    """

    def __init__(
        self,
        nlev: int,
        sigma_full,
        dsigma,
        experiment_id: str = "amip",
        monthly_means: bool = False,
        cmip_output: bool = False,
        clear_sky_diag: bool = False,
        n_days: int = 200,
        output_dir: str | Path = "",
        cmip_resolution_deg: float = 5.0,
    ):
        self.nlev = nlev
        self.sigma_full = sigma_full
        self.dsigma = dsigma
        self.clear_sky_diag = clear_sky_diag

        # Time-series storage
        self.times: list[float] = []
        self.sst: list[float] = []
        self.sic: list[float] = []
        self.T_atm: list[float] = []
        self.T_low: list[float] = []
        self.max_wind: list[float] = []
        self.precip: list[float] = []
        self.CWV: list[float] = []
        self.sw_up_toa: list[float] = []
        self.lw_up_toa: list[float] = []
        self.sw_net_sfc: list[float] = []
        self.lw_net_sfc: list[float] = []
        self.dry_mass: list[float] = []
        self.profiles_T: list[np.ndarray] = []
        self.profiles_qv: list[np.ndarray] = []

        # Energy budget tracker
        self.energy_tracker = EnergyBudgetTracker()

        # Moisture budget tracker
        from legoesm.diagnostics.energy_budget import MoistureBudgetTracker
        self.moisture_tracker = MoistureBudgetTracker()

        # Monthly means
        self.monthly_means = monthly_means
        self.monthly_accum = None
        if monthly_means:
            from legoesm.diagnostics.monthly_means import MonthlyAccumulator
            self.monthly_accum = MonthlyAccumulator(nlev=nlev, n_lat_bins=90)

        # CFWriter + spatial monthly accumulator for CMIP output
        self.cf_writer = None
        self._spatial_monthly = None
        self._cs_regrid_weights = None  # cached cubed-sphere → lat-lon weights
        self._cmip_start_year = 1  # updated by set_cmip_start_year()
        if cmip_output:
            from legoesm.io.cmor_output import CFWriter
            cmor_dir = str(Path(output_dir) / "cmor") if output_dir else "cmor"
            self.cf_writer = CFWriter(
                output_dir=cmor_dir,
                experiment_id=experiment_id,
                model_id="legoESM-1-0",
                freq="mon",
                calendar="noleap",
                ref_date="0001-01-01",
            )
            if not monthly_means:
                from legoesm.diagnostics.monthly_means import MonthlyAccumulator
                self.monthly_means = True
                self.monthly_accum = MonthlyAccumulator(nlev=nlev, n_lat_bins=90)
            # Full spatial accumulator for CMIP NetCDF output
            from legoesm.diagnostics.monthly_means import SpatialMonthlyAccumulator
            _cmip_nlon = int(round(360.0 / cmip_resolution_deg))
            _cmip_nlat = int(round(180.0 / cmip_resolution_deg))
            self._spatial_monthly = SpatialMonthlyAccumulator(
                nlat=_cmip_nlat, nlon=_cmip_nlon, nlev=nlev,
            )
            self._cmip_nlat = _cmip_nlat
            self._cmip_nlon = _cmip_nlon

        # Snapshots
        self.snapshot_days: set[int] = set()
        for d in [5, 10, 15, 20, 25, 30, 60, 100, 200, 300]:
            if d <= n_days:
                self.snapshot_days.add(d)
        if n_days not in self.snapshot_days:
            self.snapshot_days.add(n_days)
        self.snapshots: dict[int, dict[str, np.ndarray]] = {}

    def set_cmip_grid_info(self, grid_type: str, grid=None, start_year: int = 1):
        """Configure CMIP output grid and regridding weights.

        Must be called after setup() and before the first collect() when
        ``cmip_output=True``.

        Parameters
        ----------
        grid_type : str
            "cubed_sphere", "gaussian", "latlon", or "voronoi".
        grid : object, optional
            Grid object (needed for regridding weights).
        start_year : int
            Calendar start year for time axis in NetCDF files.
        """
        self._cmip_start_year = start_year
        self._cmip_grid_type = grid_type

        if self._spatial_monthly is None:
            return

        if grid_type == "cubed_sphere" and grid is not None:
            from legoesm.grids.regridding import (
                get_cubedsphere_to_latlon_weights,
            )
            n = grid.n
            self._cs_regrid_weights = get_cubedsphere_to_latlon_weights(
                n, n_lon=self._cmip_nlon, n_lat=self._cmip_nlat,
            )
        elif grid_type in ("latlon", "gaussian") and grid is not None:
            # For structured grids, store native 1-D coordinates (degrees)
            # for bilinear regridding when native shape != CMIP target.
            native_lat = np.asarray(grid.lat)  # radians, 1-D
            native_lon = np.asarray(grid.lon)  # radians, 1-D
            n_lat_native = native_lat.shape[0]
            n_lon_native = native_lon.shape[0]
            if (n_lat_native, n_lon_native) == (self._cmip_nlat, self._cmip_nlon):
                # Native grid matches CMIP target — no regridding needed.
                self._structured_regrid = None
            else:
                # Precompute regridding from native → CMIP lat-lon.
                self._structured_regrid = _build_structured_regrid_weights(
                    src_lat_rad=native_lat,
                    src_lon_rad=native_lon,
                    tgt_nlat=self._cmip_nlat,
                    tgt_nlon=self._cmip_nlon,
                )
        elif grid_type in ("voronoi", "mpas"):
            raise ValueError(
                f"CMIP output is not supported for grid_type={grid_type!r}. "
                f"Voronoi/MPAS grids require unstructured-to-latlon regridding "
                f"which is not yet implemented."
            )

    def _regrid_to_latlon_2d(self, field) -> np.ndarray | None:
        """Regrid a 2-D field to the CMIP lat-lon grid.

        Returns (nlat, nlon) numpy array, or None if no weights.
        """
        if self._cs_regrid_weights is not None:
            from legoesm.grids.regridding import apply_cubedsphere_to_latlon
            return apply_cubedsphere_to_latlon(
                np.asarray(field), self._cs_regrid_weights,
            )
        # Structured grids (lat-lon / Gaussian)
        arr = np.asarray(field)
        if arr.ndim == 2:
            regrid = getattr(self, '_structured_regrid', None)
            if regrid is not None:
                return _apply_structured_regrid_2d(arr, regrid)
            # Resample when source shape differs from CMIP target
            if arr.shape != (self._cmip_nlat, self._cmip_nlon):
                from scipy.ndimage import zoom
                factors = (self._cmip_nlat / arr.shape[0],
                           self._cmip_nlon / arr.shape[1])
                return zoom(arr, factors, order=1)
            return arr
        return None

    def _regrid_to_latlon_3d(self, field) -> np.ndarray | None:
        """Regrid a 3-D field to the CMIP lat-lon grid.

        Returns (nlat, nlon, nlev) numpy array, or None if no weights.
        """
        if self._cs_regrid_weights is not None:
            from legoesm.grids.regridding import apply_cubedsphere_to_latlon_3d
            return apply_cubedsphere_to_latlon_3d(
                np.asarray(field), self._cs_regrid_weights,
            )
        arr = np.asarray(field)
        if arr.ndim == 3:
            regrid = getattr(self, '_structured_regrid', None)
            if regrid is not None:
                return _apply_structured_regrid_3d(arr, regrid)
            # Resample when source shape differs from CMIP target
            if arr.shape[:2] != (self._cmip_nlat, self._cmip_nlon):
                from scipy.ndimage import zoom
                factors = (self._cmip_nlat / arr.shape[0],
                           self._cmip_nlon / arr.shape[1],
                           1.0)
                return zoom(arr, factors, order=1)
            return arr
        return None

    def _interp_to_plev19(self, field_3d, p_s) -> np.ndarray | None:
        """Interpolate a 3-D field from model levels to CMIP6 plev19.

        Parameters
        ----------
        field_3d : array, shape (..., nlev)
            Field on model levels.
        p_s : array, shape (...)
            Surface pressure [Pa].

        Returns (..., 19) numpy array on CMIP6 standard pressure levels,
        or None if sigma_full is not available.
        """
        sigma = np.asarray(self.sigma_full)
        p_s_np = np.asarray(p_s)
        field_np = np.asarray(field_3d)

        # Model pressure at each level: p_k = sigma_k * p_s
        # Shape: (..., nlev)
        p_model = p_s_np[..., None] * sigma

        # Target pressure levels (ascending for interpolation)
        plev_target = np.sort(CMIP6_PLEV19)  # ascending (100 Pa → 100000 Pa)

        # Log-pressure linear interpolation (numpy version)
        log_p_model = np.log(np.maximum(p_model, 1e-10))
        log_plev = np.log(plev_target)

        # For each target level, find bracketing model levels and interpolate
        n_target = len(plev_target)
        out_shape = field_np.shape[:-1] + (n_target,)
        result = np.empty(out_shape, dtype=np.float64)

        for k in range(n_target):
            log_pt = log_plev[k]
            # searchsorted on the last axis of log_p_model
            # p_model is ascending (sigma is ascending: top→bottom)
            idx_hi = np.searchsorted(
                sigma, plev_target[k] / np.maximum(p_s_np, 1e-10),
            )
            idx_hi = np.clip(idx_hi, 1, len(sigma) - 1)
            idx_lo = idx_hi - 1

            # Gather bracket values using advanced indexing
            flat_shape = field_np.shape[:-1]
            f_lo = np.take_along_axis(field_np, idx_lo[..., None], axis=-1)[..., 0]
            f_hi = np.take_along_axis(field_np, idx_hi[..., None], axis=-1)[..., 0]
            lp_lo = np.take_along_axis(log_p_model, idx_lo[..., None], axis=-1)[..., 0]
            lp_hi = np.take_along_axis(log_p_model, idx_hi[..., None], axis=-1)[..., 0]

            denom = lp_hi - lp_lo
            denom = np.where(denom == 0.0, 1.0, denom)
            alpha = np.clip((log_pt - lp_lo) / denom, 0.0, 1.0)
            result[..., k] = f_lo + alpha * (f_hi - f_lo)

        return result

    def collect(
        self,
        elapsed_day: float,
        day: float,
        state,
        q_v,
        q_c,
        q_r,
        sst,
        sic,
        precip_total,
        sw_up_toa,
        lw_up_toa,
        sw_net_sfc,
        lw_net_sfc,
        sw_down_toa,
        T_ice: float,
        lat_deg_grid=None,
        shflx=None,
        lhflx=None,
    ) -> None:
        """Collect diagnostics at a diagnostic interval.

        Parameters
        ----------
        elapsed_day : float
            Days since simulation start.
        day : float
            Absolute simulation day.
        state : HydrostaticState
        q_v, q_c, q_r : jax.Array
        sst, sic : jax.Array
        precip_total : jax.Array
            Total precipitation [kg/m2/s].
        sw_up_toa, lw_up_toa : jax.Array
        sw_net_sfc, lw_net_sfc : jax.Array
        sw_down_toa : jax.Array
        T_ice : float
        lat_deg_grid : array, optional
            Latitude in degrees for monthly means.
        """
        from legoesm.forcing.surface_utils import blend_surface_temperature

        mean_sst = float(jnp.mean(sst))
        mean_sic = float(jnp.mean(sic))
        mean_T = float(jnp.mean(state.T.data))
        mean_T_low = float(jnp.mean(state.T.data[..., -1]))
        if hasattr(state, 'v'):
            max_v = float(jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2)))
        else:
            max_v = float(jnp.max(jnp.abs(state.u.data)))
        mean_precip = float(jnp.mean(precip_total)) * 86400.0
        cwv = column_water_vapor(q_v, state.p_s.data, self.dsigma)
        mean_cwv = float(jnp.mean(cwv))
        mean_sw_toa = float(jnp.mean(sw_up_toa))
        mean_lw_toa = float(jnp.mean(lw_up_toa))
        mean_ps = float(jnp.mean(state.p_s.data))
        mean_sw_sfc = float(jnp.mean(sw_net_sfc))
        mean_lw_sfc = float(jnp.mean(lw_net_sfc))

        self.times.append(elapsed_day)
        self.sst.append(mean_sst)
        self.sic.append(mean_sic)
        self.T_atm.append(mean_T)
        self.T_low.append(mean_T_low)
        self.max_wind.append(max_v)
        self.precip.append(mean_precip)
        self.CWV.append(mean_cwv)
        self.sw_up_toa.append(mean_sw_toa)
        self.lw_up_toa.append(mean_lw_toa)
        self.sw_net_sfc.append(mean_sw_sfc)
        self.lw_net_sfc.append(mean_lw_sfc)
        self.dry_mass.append(mean_ps)

        # Mean over all spatial axes except the last (vertical).
        # Cubed-sphere: (6,n,n,nlev) → mean over (0,1,2) → (nlev,)
        # Lat-lon:      (nlat,nlon,nlev) → mean over (0,1) → (nlev,)
        spatial_axes = tuple(range(state.T.data.ndim - 1))
        self.profiles_T.append(
            np.asarray(jnp.mean(state.T.data, axis=spatial_axes))
        )
        self.profiles_qv.append(
            np.asarray(jnp.mean(q_v, axis=spatial_axes)) * 1000.0
        )

        # Snapshots
        iday = int(round(elapsed_day))
        if iday in self.snapshot_days:
            T_sfc_snap = blend_surface_temperature(sst, sic, T_ice)
            q_c_low = np.asarray(jnp.zeros_like(q_v[..., -1]) if q_c is None else q_c[..., -1]) * 1000.0
            q_r_low = np.asarray(jnp.zeros_like(q_v[..., -1]) if q_r is None else q_r[..., -1]) * 1000.0
            self.snapshots[iday] = {
                'SST': np.asarray(sst),
                'SIC': np.asarray(sic),
                'T_sfc': np.asarray(T_sfc_snap),
                'T_low': np.asarray(state.T.data[..., -1]),
                'q_v_low': np.asarray(q_v[..., -1]) * 1000.0,
                'q_c_low': q_c_low,
                'q_r_low': q_r_low,
                'precip': np.asarray(precip_total) * 86400.0,
                'wind': np.asarray(
                    jnp.sqrt(state.u.data[..., -1] ** 2 + state.v.data[..., -1] ** 2)
                ),
            }

        # Energy budget
        elapsed_s = elapsed_day * 86400.0
        self.energy_tracker.update(
            state.T.data, q_v, state.u.data, state.v.data,
            state.phis.data, state.p_s.data,
            self.dsigma, self.sigma_full,
            sw_down_toa, sw_up_toa, lw_up_toa, sw_net_sfc, lw_net_sfc,
            elapsed_seconds=elapsed_s,
        )

        # Moisture budget
        self.moisture_tracker.update(
            q_v, state.p_s.data, self.dsigma,
            precip_total, elapsed_seconds=elapsed_s,
        )

        # Monthly means
        if self.monthly_means and self.monthly_accum is not None and lat_deg_grid is not None:
            doy, _ = day_to_calendar(day)
            year = int(day // 365.0)
            fields_2d = {
                'T_low': np.asarray(state.T.data[..., -1]),
                'precip': np.asarray(precip_total) * 86400.0,
                'sw_up_toa': np.asarray(sw_up_toa),
                'lw_up_toa': np.asarray(lw_up_toa),
                'sw_net_sfc': np.asarray(sw_net_sfc),
                'lw_net_sfc': np.asarray(lw_net_sfc),
            }
            self.monthly_accum.add_2d(doy, year, fields_2d, lat_deg_grid)
            self.monthly_accum.add_3d(doy, year, {
                'T': np.asarray(state.T.data),
                'u': np.asarray(state.u.data),
                'q_v': np.asarray(q_v) * 1000.0,
            }, lat_deg_grid)
            ebudget = self.energy_tracker
            self.monthly_accum.add_scalar(doy, year, {
                'T_atm': mean_T,
                'T_low': mean_T_low,
                'precip': mean_precip,
                'sw_up_toa': mean_sw_toa,
                'lw_up_toa': mean_lw_toa,
            })

        # Spatial monthly accumulation for CMIP output
        if self._spatial_monthly is not None:
            from legoesm import constants as _c

            doy, _ = day_to_calendar(day)
            year = int(day // 365.0)
            fields_2d = {}
            r = self._regrid_to_latlon_2d(state.T.data[..., -1])
            if r is not None:
                # tas: lowest model level T as proxy for 2 m air temperature.
                # True 2 m diagnostic requires a surface-layer scheme.
                fields_2d['tas'] = r
            r = self._regrid_to_latlon_2d(precip_total)
            if r is not None:
                fields_2d['pr'] = r
            r = self._regrid_to_latlon_2d(sw_up_toa)
            if r is not None:
                fields_2d['rsut'] = r
            r = self._regrid_to_latlon_2d(lw_up_toa)
            if r is not None:
                fields_2d['rlut'] = r
            r = self._regrid_to_latlon_2d(state.p_s.data)
            if r is not None:
                fields_2d['ps'] = r

            # rsdt: TOA incoming shortwave [W/m2] — correctly available
            r = self._regrid_to_latlon_2d(sw_down_toa)
            if r is not None:
                fields_2d['rsdt'] = r

            # NOTE: rsds/rlds (surface downwelling) are NOT computed here.
            # The runtime only provides sw_net_sfc/lw_net_sfc (net fluxes),
            # which are not equal to the downwelling component.  Publishing
            # net fluxes under CMIP downwelling names would be scientifically
            # incorrect.  These variables will be added when the physics
            # pipeline exposes separate downwelling surface fluxes.

            # NOTE: clt (total cloud cover) is NOT computed here.  The
            # available binary column cloud mask (q_c > threshold → 100%)
            # does not represent cloud area fraction as defined by CMIP.
            # A proper cloud overlap / random-maximum scheme is needed.

            # psl: sea-level pressure via hypsometric equation
            # p_sl = p_s * exp(phis / (R_d * T_lowest))
            # This is a standard GCM approximation; a more accurate
            # extrapolation (e.g., WMO method) would improve results
            # over steep topography.
            T_lowest = np.asarray(state.T.data[..., -1])
            phis = np.asarray(state.phis.data)
            p_s_np = np.asarray(state.p_s.data)
            T_lowest_safe = np.maximum(T_lowest, 200.0)  # avoid div-by-zero
            psl = p_s_np * np.exp(phis / (_c.R_d * T_lowest_safe))
            r = self._regrid_to_latlon_2d(psl)
            if r is not None:
                fields_2d['psl'] = r

            # prw: column water vapor [kg/m2]
            cwv_field = np.asarray(
                column_water_vapor(q_v, state.p_s.data, self.dsigma)
            )
            r = self._regrid_to_latlon_2d(cwv_field)
            if r is not None:
                fields_2d['prw'] = r

            # hfss: surface upward sensible heat flux [W/m2]
            if shflx is not None:
                r = self._regrid_to_latlon_2d(shflx)
                if r is not None:
                    fields_2d['hfss'] = r

            # hfls: surface upward latent heat flux [W/m2]
            if lhflx is not None:
                r = self._regrid_to_latlon_2d(lhflx)
                if r is not None:
                    fields_2d['hfls'] = r

            if fields_2d:
                self._spatial_monthly.add_2d(doy, year, fields_2d)

            # 3-D fields: interpolate from model levels to plev19, then regrid
            fields_3d = {}
            ta_plev = self._interp_to_plev19(state.T.data, state.p_s.data)
            ua_plev = self._interp_to_plev19(state.u.data, state.p_s.data)
            hus_plev = self._interp_to_plev19(q_v, state.p_s.data)
            if ta_plev is not None:
                r = self._regrid_to_latlon_3d(ta_plev)
                if r is not None:
                    fields_3d['ta'] = r
            if ua_plev is not None:
                r = self._regrid_to_latlon_3d(ua_plev)
                if r is not None:
                    fields_3d['ua'] = r
            if hus_plev is not None:
                r = self._regrid_to_latlon_3d(hus_plev)
                if r is not None:
                    fields_3d['hus'] = r

            # va: northward wind on pressure levels
            if hasattr(state, 'v'):
                va_plev = self._interp_to_plev19(state.v.data, state.p_s.data)
                if va_plev is not None:
                    r = self._regrid_to_latlon_3d(va_plev)
                    if r is not None:
                        fields_3d['va'] = r

            if fields_3d:
                self._spatial_monthly.add_3d(doy, year, fields_3d)

        return {
            'mean_sst': mean_sst,
            'mean_sic': mean_sic,
            'mean_T': mean_T,
            'mean_T_low': mean_T_low,
            'max_v': max_v,
            'mean_precip': mean_precip,
            'mean_cwv': mean_cwv,
            'mean_sw_toa': mean_sw_toa,
            'mean_lw_toa': mean_lw_toa,
            'mean_sw_sfc': mean_sw_sfc,
            'mean_lw_sfc': mean_lw_sfc,
        }

    def collect_lightweight(
        self,
        elapsed_day: float,
        state,
        q_v,
        sst,
        sic,
        precip_total,
        sw_up_toa,
        lw_up_toa,
        sw_net_sfc,
        lw_net_sfc,
    ) -> dict:
        """Collect only scalar reduction diagnostics (no host materialization).

        This is the performance-mode alternative to :meth:`collect`.
        It computes global means and max-wind via ``jnp.mean``/``jnp.max``
        which work correctly on SPMD-sharded arrays (JAX handles
        cross-device reductions internally).  No ``np.asarray()`` calls,
        no snapshot capture, no profile extraction, no monthly means.

        Use this for scaling benchmarks where diagnostic overhead must
        not dominate wall-clock time.

        Returns the same dict keys as ``collect`` for logging compatibility.
        """
        mean_sst = float(jnp.mean(sst))
        mean_sic = float(jnp.mean(sic))
        mean_T = float(jnp.mean(state.T.data))
        mean_T_low = float(jnp.mean(state.T.data[..., -1]))
        if hasattr(state, 'v'):
            max_v = float(jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2)))
        else:
            max_v = float(jnp.max(jnp.abs(state.u.data)))
        mean_precip = float(jnp.mean(precip_total)) * 86400.0
        cwv = column_water_vapor(q_v, state.p_s.data, self.dsigma)
        mean_cwv = float(jnp.mean(cwv))
        mean_sw_toa = float(jnp.mean(sw_up_toa))
        mean_lw_toa = float(jnp.mean(lw_up_toa))
        mean_ps = float(jnp.mean(state.p_s.data))
        mean_sw_sfc = float(jnp.mean(sw_net_sfc))
        mean_lw_sfc = float(jnp.mean(lw_net_sfc))

        # Append to time-series (same as collect, for continuity).
        self.times.append(elapsed_day)
        self.sst.append(mean_sst)
        self.sic.append(mean_sic)
        self.T_atm.append(mean_T)
        self.T_low.append(mean_T_low)
        self.max_wind.append(max_v)
        self.precip.append(mean_precip)
        self.CWV.append(mean_cwv)
        self.sw_up_toa.append(mean_sw_toa)
        self.lw_up_toa.append(mean_lw_toa)
        self.sw_net_sfc.append(mean_sw_sfc)
        self.lw_net_sfc.append(mean_lw_sfc)
        self.dry_mass.append(mean_ps)

        return {
            'mean_sst': mean_sst,
            'mean_sic': mean_sic,
            'mean_T': mean_T,
            'mean_T_low': mean_T_low,
            'max_v': max_v,
            'mean_precip': mean_precip,
            'mean_cwv': mean_cwv,
            'mean_sw_toa': mean_sw_toa,
            'mean_lw_toa': mean_lw_toa,
            'mean_sw_sfc': mean_sw_sfc,
            'mean_lw_sfc': mean_lw_sfc,
        }

    def flush_to_disk(self, output_dir: str | Path) -> None:
        """Flush accumulated timeseries to disk and clear in-memory lists.

        Appends to ``timeseries_incremental/`` directory as numbered
        chunks.  Called periodically (e.g. annually) during long runs
        to prevent unbounded memory growth.
        """
        output_dir = Path(output_dir)
        incr_dir = output_dir / "timeseries_incremental"
        incr_dir.mkdir(parents=True, exist_ok=True)

        # Find next chunk number
        existing = sorted(incr_dir.glob("chunk_*.npz"))
        chunk_idx = len(existing)

        if not self.times:
            return  # nothing to flush

        sigma = np.asarray(self.sigma_full)

        # Save current batch
        np.savez(
            incr_dir / f"chunk_{chunk_idx:04d}.npz",
            days=np.array(self.times),
            T_atm=np.array(self.T_atm),
            T_low=np.array(self.T_low),
            max_wind=np.array(self.max_wind),
            precip=np.array(self.precip),
            CWV=np.array(self.CWV),
            sw_up_toa=np.array(self.sw_up_toa),
            lw_up_toa=np.array(self.lw_up_toa),
            sw_net_sfc=np.array(self.sw_net_sfc),
            lw_net_sfc=np.array(self.lw_net_sfc),
            dry_mass_ps=np.array(self.dry_mass),
            profiles_T=np.array(self.profiles_T) if self.profiles_T else np.array([]),
            profiles_qv=np.array(self.profiles_qv) if self.profiles_qv else np.array([]),
        )

        # Also flush energy tracker
        energy_data = self.energy_tracker.flush_to_lists()
        if energy_data["times"]:
            np.savez(
                incr_dir / f"energy_chunk_{chunk_idx:04d}.npz",
                **{k: np.array(v) for k, v in energy_data.items()},
            )

        # Clear all timeseries lists (snapshots and monthly accum preserved)
        self.times.clear()
        self.sst.clear()
        self.sic.clear()
        self.T_atm.clear()
        self.T_low.clear()
        self.max_wind.clear()
        self.precip.clear()
        self.CWV.clear()
        self.sw_up_toa.clear()
        self.lw_up_toa.clear()
        self.sw_net_sfc.clear()
        self.lw_net_sfc.clear()
        self.dry_mass.clear()
        self.profiles_T.clear()
        self.profiles_qv.clear()

        # Also flush moisture tracker
        moisture_data = self.moisture_tracker.flush_to_lists()
        if moisture_data["times"]:
            np.savez(
                incr_dir / f"moisture_chunk_{chunk_idx:04d}.npz",
                **{k: np.array(v) for k, v in moisture_data.items()},
            )

    def flush_cmip_monthly(self, current_day: float) -> None:
        """Write completed CMIP months incrementally and free their memory.

        Call this periodically (e.g. at each diagnostic interval) during
        long runs.  Only months strictly before the current month are
        flushed; the in-progress month is kept for further accumulation.
        """
        if self._spatial_monthly is None or self.cf_writer is None:
            return

        from legoesm.forcing.time_utils import day_to_calendar
        doy, _ = day_to_calendar(current_day)
        current_year = int(current_day // 365.0)
        from legoesm.diagnostics.monthly_means import MonthlyAccumulator
        current_month = MonthlyAccumulator.day_to_month(doy)

        data = self._spatial_monthly.pop_completed_months(
            current_year, current_month,
        )
        months = data.get('months', [])
        if not months:
            return

        self._write_cmip_data(data)

    def save(self, output_dir: str | Path) -> None:
        """Save all accumulated diagnostics to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sigma = np.asarray(self.sigma_full)

        np.savez(
            output_dir / "timeseries.npz",
            days=np.array(self.times),
            sst=np.array(self.sst),
            sic=np.array(self.sic),
            T_atm=np.array(self.T_atm),
            T_low=np.array(self.T_low),
            max_wind=np.array(self.max_wind),
            precip=np.array(self.precip),
            CWV=np.array(self.CWV),
            sw_up_toa=np.array(self.sw_up_toa),
            lw_up_toa=np.array(self.lw_up_toa),
            sw_net_sfc=np.array(self.sw_net_sfc),
            lw_net_sfc=np.array(self.lw_net_sfc),
            dry_mass_ps=np.array(self.dry_mass),
            sigma=sigma,
            profiles_T=np.array(self.profiles_T) if self.profiles_T else np.array([]),
            profiles_qv=np.array(self.profiles_qv) if self.profiles_qv else np.array([]),
            energy_toa_net=np.array(self.energy_tracker.toa_net),
            energy_column=np.array(self.energy_tracker.column_energy),
            energy_dE_dt=np.array(self.energy_tracker.dE_dt),
            energy_residual=np.array(self.energy_tracker.residual),
            moisture_column_water=np.array(self.moisture_tracker.column_water),
            moisture_precip_rate=np.array(self.moisture_tracker.precip_rate),
            moisture_residual=np.array(self.moisture_tracker.residual),
        )

        if self.monthly_means and self.monthly_accum is not None:
            self.monthly_accum.save(
                output_dir / "monthly_means.npz",
                sigma=sigma,
            )

        if self.snapshots:
            snap_data = {"snapshot_days": np.array(sorted(self.snapshots.keys()))}
            for day_key, fields in self.snapshots.items():
                for field_name, arr in fields.items():
                    snap_data[f"day{day_key:03d}_{field_name}"] = arr
            np.savez(output_dir / "snapshots.npz", **snap_data)

        if self.cf_writer is not None:
            self._write_cmip_monthly_files()
            self.cf_writer.close()

    def _write_cmip_monthly_files(self) -> None:
        """Write CMIP-compliant NetCDF files from the spatial accumulator.

        Called at end-of-run to flush any remaining data.
        """
        if self._spatial_monthly is None or self.cf_writer is None:
            return
        data = self._spatial_monthly.finalize()
        self._write_cmip_data(data)

    def _write_cmip_data(self, data: dict) -> None:
        """Write a batch of CMIP monthly data to NetCDF files.

        Shared implementation used by both end-of-run finalization
        and incremental monthly flushing.
        """
        if self.cf_writer is None:
            return

        months = data.get('months', [])
        if not months:
            return

        dlat = 180.0 / self._cmip_nlat
        dlon = 360.0 / self._cmip_nlon
        lat = np.linspace(-90.0 + dlat / 2, 90.0 - dlat / 2, self._cmip_nlat)
        lon = np.linspace(dlon / 2, 360.0 - dlon / 2, self._cmip_nlon)

        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        start_year = self._cmip_start_year

        from legoesm.io.cmor_output import CMIP6_PLEV19

        for i, (yr, mo) in enumerate(months):
            year_offset = (start_year - 1 + yr) * 365.0
            day_start = year_offset + sum(month_days[:mo - 1])
            day_end = day_start + month_days[mo - 1]
            time_mid = 0.5 * (day_start + day_end)
            time_bounds = (day_start, day_end)

            for key, arr in data.items():
                if not key.startswith("field_2d_"):
                    continue
                var_name = key[len("field_2d_"):]
                field_slice = arr[i]
                if np.all(np.isnan(field_slice)):
                    continue
                try:
                    self.cf_writer.write_field(
                        var_name=var_name,
                        data=field_slice,
                        time=time_mid,
                        time_bounds=time_bounds,
                        lat=lat,
                        lon=lon,
                    )
                except (KeyError, ValueError):
                    pass

            for key, arr in data.items():
                if not key.startswith("field_3d_"):
                    continue
                var_name = key[len("field_3d_"):]
                field_slice = arr[i]
                if np.all(np.isnan(field_slice)):
                    continue
                nlev = field_slice.shape[2]
                plev = np.sort(CMIP6_PLEV19)[:nlev] if nlev <= len(CMIP6_PLEV19) else None
                field_plev = np.transpose(field_slice, (2, 0, 1))
                try:
                    self.cf_writer.write_field(
                        var_name=var_name,
                        data=field_plev,
                        time=time_mid,
                        time_bounds=time_bounds,
                        lat=lat,
                        lon=lon,
                        plev=plev,
                    )
                except (KeyError, ValueError):
                    pass

    def print_summary(self) -> str:
        """Return end-of-run summary string."""
        parts = [self.energy_tracker.summary()]
        if len(self.moisture_tracker.times) >= 2:
            parts.append(self.moisture_tracker.summary())
        return "\n\n".join(parts)

    def check_stability(self, state, elapsed_day: float) -> str | None:
        """Check for blow-up conditions.

        Returns
        -------
        str or None
            Error message if blow-up detected, None if stable.
        """
        if not jnp.all(jnp.isfinite(state.u.data)):
            return f"BLOWUP at day {elapsed_day:.0f}: non-finite winds"
        if hasattr(state, 'v'):
            max_v = float(jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2)))
        else:
            max_v = float(jnp.max(jnp.abs(state.u.data)))
        if max_v > 500:
            return f"BLOWUP at day {elapsed_day:.0f}: max wind {max_v:.1f} m/s"
        if not jnp.all(jnp.isfinite(state.T.data)):
            return f"BLOWUP at day {elapsed_day:.0f}: non-finite T"

        # Temperature bounds (physical range for Earth atmosphere)
        T_min_val = float(jnp.min(state.T.data))
        T_max_val = float(jnp.max(state.T.data))
        if T_min_val < 100.0 or T_max_val > 400.0:
            return (
                f"BLOWUP at day {elapsed_day:.0f}: temperature out of physical bounds "
                f"(min={T_min_val:.1f}K, max={T_max_val:.1f}K). "
                f"Check dt, hyperdiffusion, and physics configuration."
            )

        # Surface pressure bounds
        if hasattr(state, 'p_s'):
            ps_min = float(jnp.min(state.p_s.data))
            ps_max = float(jnp.max(state.p_s.data))
            if ps_min < 40000.0 or ps_max > 115000.0:
                return (
                    f"BLOWUP at day {elapsed_day:.0f}: surface pressure out of bounds "
                    f"(min={ps_min:.0f}Pa, max={ps_max:.0f}Pa). "
                    f"Check dt and dynamics configuration."
                )

        return None


class EnsembleDiagnosticCollector:
    """Collects ensemble-specific diagnostics: spread, CRPS, rank histograms.

    Wraps a base ``DiagnosticCollector`` (which handles ensemble-mean diagnostics)
    and adds per-member spread tracking, per-member checkpointing, and
    probabilistic verification scores.

    Parameters
    ----------
    base_collector : DiagnosticCollector
        The base collector (operates on ensemble-mean state).
    n_members : int
        Number of ensemble members.
    """

    def __init__(self, base_collector: DiagnosticCollector, n_members: int):
        self.base = base_collector
        self.n_members = n_members

        # Time-series of ensemble spread
        self.spread_T: list[float] = []
        self.spread_u: list[float] = []
        self.spread_ps: list[float] = []
        self.spread_times: list[float] = []

    def collect_ensemble(
        self,
        elapsed_day: float,
        full_carry,
        mean_carry,
    ) -> dict[str, float]:
        """Collect ensemble spread diagnostics from the full ensemble carry.

        Parameters
        ----------
        elapsed_day : float
            Days since simulation start.
        full_carry : SegmentCarry with leading (n_members, ...) dimension
            Full ensemble state.
        mean_carry : SegmentCarry without ensemble dimension
            Ensemble mean (for base collector).

        Returns
        -------
        dict with spread metrics.
        """
        from legoesm.parallel.ensemble import ensemble_spread

        spread = ensemble_spread(full_carry)
        self.spread_times.append(elapsed_day)
        self.spread_T.append(spread.get('T', 0.0))
        self.spread_u.append(spread.get('u', 0.0))
        self.spread_ps.append(spread.get('p_s', 0.0))

        return {
            'spread_T': spread.get('T', 0.0),
            'spread_u': spread.get('u', 0.0),
            'spread_ps': spread.get('p_s', 0.0),
        }

    def save_member_checkpoint(
        self,
        full_carry,
        step: int,
        day: float,
        output_dir,
        config,
    ) -> None:
        """Save per-member checkpoints for the full ensemble.

        Each member is saved as a separate checkpoint file:
        ``ensemble_member_NNN_day_DDDD.npz``.
        """
        from legoesm.io.restart import save_restart
        from legoesm.core.field import Field
        from legoesm.core.state import HydrostaticState

        output_dir = Path(output_dir)
        n = self.n_members

        for m in range(n):
            # Extract single member from batched carry
            member_carry = jax.tree.map(lambda x: x[m], full_carry)

            # Build a minimal state for save_restart
            dims_3d = ("face", "x", "y", "level")
            dims_2d = ("face", "x", "y")
            member_state = HydrostaticState(
                u=Field(member_carry.u, name="u", dims=dims_3d, units="m/s"),
                v=Field(member_carry.v, name="v", dims=dims_3d, units="m/s"),
                T=Field(member_carry.T, name="T", dims=dims_3d, units="K"),
                p_s=Field(member_carry.p_s, name="p_s", dims=dims_2d, units="Pa"),
                phis=Field(member_carry.phis, name="phis", dims=dims_2d, units="m2/s2"),
            )

            elapsed = day - config.start_day
            path = output_dir / f"ensemble_member_{m:03d}_day_{int(elapsed):04d}.npz"
            save_restart(
                path=path,
                state=member_state,
                q_v=member_carry.q_v,
                step=step,
                day=day,
                config=config,
                q_c=member_carry.q_c,
                q_r=member_carry.q_r,
            )

    def save(self, output_dir) -> None:
        """Save ensemble diagnostics alongside base diagnostics."""
        output_dir = Path(output_dir)

        # Save spread timeseries
        if self.spread_times:
            np.savez(
                output_dir / "ensemble_spread.npz",
                days=np.array(self.spread_times),
                spread_T=np.array(self.spread_T),
                spread_u=np.array(self.spread_u),
                spread_ps=np.array(self.spread_ps),
            )

        # Base diagnostics
        self.base.save(output_dir)
