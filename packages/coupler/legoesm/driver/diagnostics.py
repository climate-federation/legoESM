"""Diagnostic collection for the composable model driver.

Extracts the diagnostic accumulation, snapshot capture, and output
logic from run_amip.py into a reusable class.
"""
from __future__ import annotations

import math as _math

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from legoesm.diagnostics.cloud_overlap import maximum_random_overlap
from legoesm.diagnostics.column_integrals import column_water_vapor
from legoesm.diagnostics.energy_budget import (
    EnergyBudgetTracker,
    MoistureBudgetTracker,
    area_weighted_mean,
    area_weighted_profile,
)
from legoesm.diagnostics.monthly_means import MonthlyAccumulator
from legoesm.forcing.surface_utils import blend_surface_temperature
from legoesm.forcing.time_utils import day_to_calendar
from legoesm.io.cmor_output import CMIP6_PLEV19


def _frozen_condensate(state):
    """Summed frozen condensate q_i+q_s+q_g from a state's tracers, or None.

    The phase-completeness input for the energy tracker (#1354/#1515): the
    conserved column energy carries −L_f·q_frozen.  Sums whatever frozen
    species the active microphysics carries (Morrison: q_i/q_s/q_g; warm-rain:
    none → None → vapor-only moist static energy, byte-identical legacy).
    """
    tr = getattr(state, "tracers", None)
    if not tr:
        return None
    total = None
    for k in ("q_i", "q_s", "q_g"):
        if k in tr and tr[k] is not None:
            d = tr[k].data
            total = d if total is None else total + d
    return total


# Sidecar key carrying the CMOR lat sampling stamp (see CFWriter append guard).
_SIDECAR_SAMPLING_KEY = "meta.lat_sampling"


class _StructuredRegridWeights:
    """Precomputed bilinear interpolation weights for structured grids.

    ``i_lo``/``i_hi`` index the native latitude axis and ``j_lo``/``j_hi``
    the native longitude axis in the field's OWN order; ``j_hi`` wraps
    periodically."""
    __slots__ = ('i_lo', 'i_hi', 'j_lo', 'j_hi', 'wi', 'wj')

    def __init__(self, i_lo, i_hi, j_lo, j_hi, wi, wj):
        self.i_lo, self.i_hi = i_lo, i_hi
        self.j_lo, self.j_hi = j_lo, j_hi
        self.wi, self.wj = wi, wj


def _bracket(src: np.ndarray, tgt: np.ndarray, period: float | None):
    """Indices (into ``src`` as given) bracketing each ``tgt`` value, and the
    linear weight of the upper one.  ``period`` wraps the axis (longitude);
    otherwise targets outside the source range take the edge value."""
    if period is not None:
        src = np.mod(src, period)
        tgt = np.mod(tgt, period)
    order = np.argsort(src, kind="stable")
    s = src[order]
    n = s.size
    k = np.searchsorted(s, tgt, side="right") - 1
    if period is None:
        k = np.clip(k, 0, n - 2)
        s_lo, s_hi = s[k], s[k + 1]
        lo, hi = k, k + 1
    else:
        lo, hi = np.mod(k, n), np.mod(k + 1, n)
        s_lo = np.where(k < 0, s[-1] - period, s[lo])
        s_hi = np.where(k + 1 >= n, s[0] + period, s[hi])
    w = np.clip((tgt - s_lo) / np.where(s_hi == s_lo, 1.0, s_hi - s_lo),
                0.0, 1.0)
    return order[lo], order[hi], w


def _build_structured_regrid_weights(
    src_lat_rad: np.ndarray,
    src_lon_rad: np.ndarray,
    tgt_nlat: int,
    tgt_nlon: int,
    tgt_lat_deg: np.ndarray | None = None,
    tgt_lon_deg: np.ndarray | None = None,
) -> _StructuredRegridWeights:
    """Bilinear weights from a native structured grid (either latitude order)
    to the CMIP lat-lon target, periodic in longitude.

    Targets default to the regular cell centres the CMOR files are labelled
    with (``-90+dlat/2 ..``, ``dlon/2 ..``); callers pass the labels so the
    two cannot drift.
    """
    dlat = 180.0 / tgt_nlat
    dlon = 360.0 / tgt_nlon
    tgt_lat = (np.linspace(-90.0 + dlat / 2, 90.0 - dlat / 2, tgt_nlat)
               if tgt_lat_deg is None else np.asarray(tgt_lat_deg, np.float64))
    tgt_lon = (np.linspace(dlon / 2, 360.0 - dlon / 2, tgt_nlon)
               if tgt_lon_deg is None else np.asarray(tgt_lon_deg, np.float64))
    i_lo, i_hi, wi = _bracket(np.degrees(src_lat_rad), tgt_lat, None)
    j_lo, j_hi, wj = _bracket(np.degrees(src_lon_rad), tgt_lon, 360.0)
    return _StructuredRegridWeights(i_lo, i_hi, j_lo, j_hi, wi, wj)


def _apply_structured_regrid_2d(
    field: np.ndarray,
    w: _StructuredRegridWeights,
) -> np.ndarray:
    """Apply bilinear interpolation to a 2-D field (nlat_src, nlon_src)
    → (nlat_tgt, nlon_tgt)."""
    wi = w.wi[:, None]
    wj = w.wj[None, :]
    return ((1 - wi) * (1 - wj) * field[np.ix_(w.i_lo, w.j_lo)]
            + wi * (1 - wj) * field[np.ix_(w.i_hi, w.j_lo)]
            + (1 - wi) * wj * field[np.ix_(w.i_lo, w.j_hi)]
            + wi * wj * field[np.ix_(w.i_hi, w.j_hi)])


def _apply_structured_regrid_3d(
    field: np.ndarray,
    w: _StructuredRegridWeights,
) -> np.ndarray:
    """Apply bilinear interpolation to a 3-D field (nlat_src, nlon_src, nlev)
    → (nlat_tgt, nlon_tgt, nlev), vectorised over levels."""
    wi = w.wi[:, None, None]
    wj = w.wj[None, :, None]
    return ((1 - wi) * (1 - wj) * field[np.ix_(w.i_lo, w.j_lo)]
            + wi * (1 - wj) * field[np.ix_(w.i_hi, w.j_lo)]
            + (1 - wi) * wj * field[np.ix_(w.i_lo, w.j_hi)]
            + wi * wj * field[np.ix_(w.i_hi, w.j_hi)])


# --- Run-time blow-up bounds (physical Earth-atmosphere range).  A state
#     outside these is a blow-up, not a bias.  ONE source of truth shared by the
#     compiled ``check_stability`` and the raw MPAS/spectral daily checks — the
#     #871 MPAS autopsy found an 8e8 K state that ran 1138 steps under a
#     finiteness-only guard because those daily checks lacked bounds. ---
_T_BLOWUP_MIN_K = 100.0
_T_BLOWUP_MAX_K = 400.0
_PS_BLOWUP_MIN_PA = 40000.0
_PS_BLOWUP_MAX_PA = 115000.0
# Tolerance for "global T_min sits AT the dycore floor": the clip is an exact
# jnp.maximum, so a pinned column reports T_min == floor up to fp rounding.
_T_FLOOR_TOL_K = 1e-3


def diagnostic_interval_steps(diag_days, dt_s, fallback_steps):
    """Diagnostic / blow-up-check cadence in STEPS, from a cadence in days.

    ``diag_days <= 0`` is the "no periodic cadence" sentinel and returns
    ``fallback_steps`` (each run loop decides what that means — one sample at
    the end of the link, or the writer disabled).

    A positive cadence never returns 0.  A sub-step request (``diag_days``
    below one timestep) truncates to zero steps, and every guard in every run
    loop reads a zero interval as "diagnostics disabled" — so asking for the
    FINEST possible cadence would silently switch the blow-up check OFF, the
    exact opposite of the request.  One step is the floor.

    This is the single definition of that arithmetic: the cadence decides how
    precisely a blow-up can be located in time (the reported failure is the
    first SAMPLE, not the first bad step), so a lane that quietly computed it
    differently would report a different failure time for the same run.
    """
    if not _math.isfinite(diag_days):
        # NaN would take the "no cadence" branch below and inf would overflow
        # the int(); either way the run would proceed with a cadence nobody
        # asked for, so refuse instead.
        raise ValueError(
            f"diag_days must be a finite number of days; got {diag_days!r}.")
    if diag_days <= 0.0:
        return fallback_steps
    return max(1, int(diag_days * 86400.0 / dt_s))


def physical_state_blowup_reason(elapsed_day, T_min, T_max,
                                 ps_min=None, ps_max=None):
    """BLOWUP reason string if T (and optional p_s) are outside the physical
    Earth-atmosphere range, else ``None``.  Pure/scalar so every run-time
    detector shares the SAME bounds — a runaway T aborts at the first daily
    check instead of running hundreds of steps under a finiteness-only guard."""
    if T_min < _T_BLOWUP_MIN_K or T_max > _T_BLOWUP_MAX_K:
        return (
            f"BLOWUP at day {elapsed_day:.0f}: temperature out of physical "
            f"bounds (min={T_min:.1f}K, max={T_max:.1f}K). "
            "Check dt, hyperdiffusion, and physics configuration."
        )
    if ps_min is not None and ps_max is not None:
        if ps_min < _PS_BLOWUP_MIN_PA or ps_max > _PS_BLOWUP_MAX_PA:
            return (
                f"BLOWUP at day {elapsed_day:.0f}: surface pressure out of "
                f"bounds (min={ps_min:.0f}Pa, max={ps_max:.0f}Pa). "
                "Check dt and dynamics configuration."
            )
    return None


def t_min_floor_blowup_reason(elapsed_day, T_min, T_floor):
    """BLOWUP reason string when the global minimum temperature is pinned at
    the ``T_min`` dycore floor, else ``None``.

    The eager MPAS path silently clips T to ``config.T_min`` each step
    (``primitive_eq_mpas`` step "Floors").  A column pinned at the floor is an
    unbudgeted energy source that MASKS a runaway (#930): the clip holds the
    reported minimum steady even as the instability grows, so a diverging run
    can look "successful".  This LOUD guard labels that specific failure.

    Pure/scalar and gated on ``T_floor > 0`` (floor disabled ⇒ never fires, to
    match the ``config.T_min > 0`` gate on the clip itself), so it *composes
    with* — and is deliberately MORE specific than —
    :func:`physical_state_blowup_reason`: when a floor of, say, 150 K sits
    above the 100 K generic lower bound, this catches a masked runaway the
    generic bounds check would miss entirely.
    """
    if T_floor > 0.0 and T_min <= T_floor + _T_FLOOR_TOL_K:
        return (
            f"BLOWUP at day {elapsed_day:.0f}: T_min floor activated "
            f"(min T={T_min:.2f}K pinned at the {T_floor:.0f}K dycore floor "
            "— masked runaway; see #930)."
        )
    return None


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
        start_year: int = 1979,
        cloud_config=None,
        vcoord=None,
        surface_stability_scheme: str = "dyer1974",
        surface_bulk_scheme: str = "constant",
    ):
        self.nlev = nlev
        self.sigma_full = sigma_full
        self.dsigma = dsigma
        # MOST selectors for the ``tas`` 2 m profile (``_tas_2m``): the SAME
        # ``ExperimentConfig.surface_stability_scheme`` /
        # ``surface_bulk_scheme`` the surface fluxes run with, so a
        # large_yeager or grachev/gryanik experiment's published ``tas`` uses
        # the same similarity functions as its fluxes.  The non-MOST
        # ``"constant"`` bulk scheme (the ExperimentConfig default) has no
        # similarity profile, so it keeps the historical coare3 profile —
        # which also keeps the default byte-identical.
        self.surface_stability_scheme = surface_stability_scheme
        self.tas_profile_scheme = (
            surface_bulk_scheme
            if surface_bulk_scheme in ("most", "coare3", "large_yeager",
                                       "large_yeager_cesm")
            else "coare3"
        )
        # Vertical coordinate object (``SigmaCoordinate`` or
        # ``HybridSigmaPressureCoordinate``).  REQUIRED to get level pressures
        # right on a HYBRID grid, which is the driver DEFAULT
        # (``GridConfig.vertical_coord = "hybrid"``): there
        # ``sigma_full`` is only a COMPATIBILITY VIEW returning ``A_full +
        # B_full``, so ``p = sigma_full * p_s`` is wrong by ``A*(p_s - p_ref)``
        # -- a few hPa near sea level but ~80 hPa over high terrain, which
        # lands CMOR ``ta``/``ua``/``hus`` on the wrong pressure surfaces.
        # ``None`` keeps the pure-sigma formula (exact when A == 0).
        self.vcoord = vcoord
        self.clear_sky_diag = clear_sky_diag
        # Cloud config (``atmosphere.physics.clouds.CloudConfig`` or ``None``)
        # for the total-cloud-cover ``clt`` diagnostic — the SAME scheme
        # selection radiation uses, so ``clt`` reflects the model's actual
        # fractional cloud fraction (issue #689).  ``None`` (cloud scheme
        # 'none') => ``clt`` is not published.
        self._cloud_config = cloud_config
        # Set by ``feed_cmip_accumulators_native`` (MPAS lane) when it derives
        # clwvi/clivi from the RADIATION cloud diagnosis (floor included).
        # The cube lane's :meth:`collect` integrates the PROGNOSTIC condensate
        # instead, so the two need different file-level comments.
        self._cloud_paths_radiative = False

        # Per-cell horizontal area weights for global-mean diagnostics.
        # ``None`` => unweighted ``jnp.mean`` (legacy behaviour); the driver
        # calls ``set_area_weights(grid.grid_area)`` so lat-lon polar rows do
        # not over-weight every <R_TOA>/<SST>/<CWV> global mean (see
        # ``area_weighted_mean``).
        self._area_w = None

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
        self.rsdt: list[float] = []
        self.hfss: list[float] = []
        self.hfls: list[float] = []
        self.profiles_T: list[np.ndarray] = []
        self.profiles_qv: list[np.ndarray] = []

        # Grid rotation angle for cubed-sphere wind rotation to geographic
        # components.  Set via set_wind_rotation_angle(); None for other grids.
        self._wind_rotation_angle: np.ndarray | None = None

        # Energy budget tracker
        self.energy_tracker = EnergyBudgetTracker()

        # Moisture budget tracker
        self.moisture_tracker = MoistureBudgetTracker()

        # Monthly means
        self.monthly_means = monthly_means
        self.monthly_accum = None
        if monthly_means:
            self.monthly_accum = MonthlyAccumulator(nlev=nlev, n_lat_bins=90)

        # CFWriter + spatial monthly accumulator for CMIP output
        self.cf_writer = None
        self._spatial_monthly = None
        self._cs_regrid_weights = None  # cached cubed-sphere → lat-lon weights
        self._voronoi_regrid_weights = None  # cached MPAS cell → lat-lon weights
        # Time axis reference is the experiment start year (CMIP6 AMIP
        # convention: ``days since <start_year>-01-01``), which makes the
        # stored time values start at zero and decode to the correct
        # wall-clock dates without relying on a distant epoch.
        self._cmip_start_year = start_year
        if cmip_output:
            from legoesm.io.cmor_output import CFWriter
            cmor_dir = str(Path(output_dir) / "cmor") if output_dir else "cmor"
            self.cf_writer = CFWriter(
                output_dir=cmor_dir,
                experiment_id=experiment_id,
                model_id="legoESM-1-0",
                freq="mon",
                calendar="noleap",
                ref_date=f"{start_year:04d}-01-01",
            )
            if not monthly_means:
                # iter-169: removed redundant local import that
                # caused F823 "referenced before assignment" — the
                # local import shadows the module-level
                # ``MonthlyAccumulator`` (line 19) for the entire
                # function scope, making the line-206 reference
                # inside the same ``__init__`` block invalid.  Use
                # the module-level import directly.
                self.monthly_means = True
                self.monthly_accum = MonthlyAccumulator(nlev=nlev, n_lat_bins=90)
            # Full spatial accumulator for CMIP NetCDF output
            from legoesm.diagnostics.monthly_means import (
                SpatialMonthlyAccumulator, SpatialDailyAccumulator,
            )
            _cmip_nlon = int(round(360.0 / cmip_resolution_deg))
            _cmip_nlat = int(round(180.0 / cmip_resolution_deg))
            self._spatial_monthly = SpatialMonthlyAccumulator(
                nlat=_cmip_nlat, nlon=_cmip_nlon, nlev=nlev,
            )
            # Daily accumulator (CMIP6 ``day`` table) — tracks running
            # min/max for tas so tasmin/tasmax can be emitted.
            self._spatial_daily = SpatialDailyAccumulator(
                nlat=_cmip_nlat, nlon=_cmip_nlon,
                track_extremes={"tas"},
            )
            self._cmip_nlat = _cmip_nlat
            self._cmip_nlon = _cmip_nlon
            # Time-invariant (``fx`` table) fields — filled by
            # ``set_fixed_fields`` if the driver supplies topography /
            # land mask; written once at end-of-run.
            self._fixed_phis: np.ndarray | None = None
            self._fixed_land_fraction: np.ndarray | None = None
        else:
            self._spatial_daily = None
            self._fixed_phis = None
            self._fixed_land_fraction = None

        # Snapshots
        self.snapshot_days: set[int] = set()
        for d in [5, 10, 15, 20, 25, 30, 60, 100, 200, 300]:
            if d <= n_days:
                self.snapshot_days.add(d)
        if n_days not in self.snapshot_days:
            self.snapshot_days.add(n_days)
        self.snapshots: dict[int, dict[str, np.ndarray]] = {}

    def set_area_weights(self, area) -> None:
        """Register per-cell horizontal areas for area-weighted global means.

        Parameters
        ----------
        area : array or None
            Per-cell area on the native horizontal grid (``grid.grid_area``:
            lat-lon ``(n_lat, n_lon)``, cube ``(6, n, n)``).  Passing ``None``
            (or a grid that lacks ``grid_area``) keeps the legacy unweighted
            ``jnp.mean``.  Without this the polar rows of a lat-lon grid count
            equally with the equatorial rows despite spanning ``~cos(lat)``
            less area, biasing <R_TOA>, <SST>, <CWV> and friends toward the
            cold high latitudes.
        """
        self._area_w = None if area is None else jnp.asarray(area)

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

        # Sample at the SAME row latitudes the files are labelled with
        # (cell centres); the regridders' default is pole-to-pole, which put
        # every row up to half a cell off its label.
        _lat_labels, _lon_labels = self._cmip_target_latlon()
        if grid_type == "cubed_sphere" and grid is not None:
            from legoesm.grids.regridding import (
                get_cubedsphere_to_latlon_weights,
            )
            n = grid.n
            self._cs_regrid_weights = get_cubedsphere_to_latlon_weights(
                n, n_lon=self._cmip_nlon, n_lat=self._cmip_nlat,
                lat_cent=_lat_labels,
            )
            if self.cf_writer is not None:
                self.cf_writer.require_centre_sampling_on_append = True
        elif grid_type in ("latlon", "gaussian") and grid is not None:
            # Structured grids: native 1-D coordinates are radians.
            native_lat = np.asarray(grid.lat)  # radians, 1-D
            native_lon = np.asarray(grid.lon)  # radians, 1-D
            n_lat_native = native_lat.shape[0]
            n_lon_native = native_lon.shape[0]
            if ((n_lat_native, n_lon_native) == (self._cmip_nlat, self._cmip_nlon)
                    and np.allclose(np.degrees(native_lat), _lat_labels,
                                    rtol=0.0, atol=1e-6)
                    and np.allclose(np.degrees(native_lon) % 360.0,
                                    _lon_labels, rtol=0.0, atol=1e-6)):
                # Native grid IS the labelled CMIP target — no regridding.
                self._structured_regrid = None
            else:
                # Interpolate to the labelled cell centres.  A same-SHAPE
                # grid is not enough: the lat-lon grid's lon 0..355 sits half
                # a cell off the 2.5..357.5 labels, a Gaussian grid's rows
                # off the regular ones.
                self._structured_regrid = _build_structured_regrid_weights(
                    src_lat_rad=native_lat,
                    src_lon_rad=native_lon,
                    tgt_nlat=self._cmip_nlat,
                    tgt_nlon=self._cmip_nlon,
                    tgt_lat_deg=_lat_labels,
                    tgt_lon_deg=_lon_labels,
                )
                if self.cf_writer is not None:
                    self.cf_writer.require_centre_sampling_on_append = True
        elif grid_type in ("mpas", "voronoi") and grid is not None:
            # SCVT/Voronoi unstructured cells → regular lat-lon via IDW
            # k-nearest weights (the AMIP forcing path does the inverse,
            # lat-lon→cells, with the same KD-tree idea).  ``latCell`` /
            # ``lonCell`` are in radians.
            from legoesm.grids.regridding import (
                compute_voronoi_to_latlon_weights,
            )
            self._voronoi_regrid_weights = compute_voronoi_to_latlon_weights(
                np.asarray(grid.latCell), np.asarray(grid.lonCell),
                n_lon=self._cmip_nlon, n_lat=self._cmip_nlat,
                lat_cent=_lat_labels,
            )
            if self.cf_writer is not None:
                self.cf_writer.require_centre_sampling_on_append = True

    def set_fixed_fields(
        self,
        phis: np.ndarray | None = None,
        land_fraction: np.ndarray | None = None,
    ) -> None:
        """Register time-invariant source fields for the CMIP6 ``fx`` file.

        Parameters
        ----------
        phis : array, optional
            Surface geopotential [m2/s2] on the native model grid.
            Converted to orography (``orog = phis / g``) and regridded
            to the CMIP target grid at save time.
        land_fraction : array, optional
            Land fraction in [0, 1] on the native model grid.  Emitted
            as ``sftlf`` (percent) on the CMIP target grid.
        """
        if phis is not None:
            self._fixed_phis = np.asarray(phis)
        if land_fraction is not None:
            self._fixed_land_fraction = np.asarray(land_fraction)

    def set_wind_rotation_angle(self, angle) -> None:
        """Register the grid-to-geographic wind rotation angle.

        Parameters
        ----------
        angle : array
            Grid rotation angle [rad] on the native model grid.
            For cubed-sphere, shape is (6, n, n).  Stored as a NumPy
            array so the diagnostics path has no JAX dependency at
            save time.  Used to rotate panel-local (u, v) to geographic
            (east, north) before computing zonal-mean profile_u.
        """
        self._wind_rotation_angle = np.asarray(angle)

    def _roll_to_cmip_lon(self, arr: np.ndarray) -> np.ndarray:
        """Reorder regridder output columns to the CMIP [0, 360) lon labels.

        The cube and Voronoi regridders emit columns on lon
        ``linspace(-180, 180, n_lon, endpoint=False) + 180/n_lon`` (column 0
        at -180+dlon/2), while ``_cmip_target_latlon`` labels the file
        ``dlon/2 .. 360-dlon/2`` (column 0 at 0+dlon/2).  Writing one under
        the other's labels shifts every map by 180 deg (the tuned-year clt
        vs coastlines bug; same quirk the ocean matrix runner rolls for).
        A half-turn is an integer column roll only for even n_lon — fail
        loud rather than write a fractionally-shifted field.
        """
        if self._cmip_nlon % 2 != 0:
            raise ValueError(
                f"CMIP n_lon={self._cmip_nlon} must be even to map the "
                "regridder's [-180,180) columns onto [0,360) labels.")
        return np.roll(arr, self._cmip_nlon // 2, axis=1)

    def _regrid_to_latlon_2d(self, field) -> np.ndarray | None:
        """Regrid a 2-D field to the CMIP lat-lon grid.

        Returns (nlat, nlon) numpy array, or None if no weights.
        """
        if self._cs_regrid_weights is not None:
            from legoesm.grids.regridding import apply_cubedsphere_to_latlon
            return self._roll_to_cmip_lon(apply_cubedsphere_to_latlon(
                np.asarray(field), self._cs_regrid_weights,
            ))
        if self._voronoi_regrid_weights is not None:
            from legoesm.grids.regridding import apply_voronoi_to_latlon
            return self._roll_to_cmip_lon(apply_voronoi_to_latlon(
                np.asarray(field).reshape(-1), self._voronoi_regrid_weights,
            ))
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
            # (nlat, nlon, nlev): lon is axis 1, same roll as the 2-D path.
            return self._roll_to_cmip_lon(apply_cubedsphere_to_latlon_3d(
                np.asarray(field), self._cs_regrid_weights,
            ))
        if self._voronoi_regrid_weights is not None:
            from legoesm.grids.regridding import apply_voronoi_to_latlon_3d
            arr = np.asarray(field)
            # MPAS callers pass an explicit 2-D (nCells, nlev) array;
            # flattened (nCells*nlev,) inputs are not supported here.
            return self._roll_to_cmip_lon(
                apply_voronoi_to_latlon_3d(arr, self._voronoi_regrid_weights))
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

    def _p_full(self, p_s):
        """Full-level pressure [Pa] for this vertical coordinate.

        ``p = A p_ref + B p_s`` for hybrid, ``p = sigma p_s`` for pure sigma.
        Never assume the latter: on a hybrid grid ``sigma_full`` is the
        ``A_full + B_full`` compatibility view, whose own docstring warns it is
        for utilities that do NOT assume pure-sigma pressure dependence.
        """
        if self.vcoord is not None:
            return self.vcoord.pressure_at_full(p_s)
        import jax.numpy as jnp
        return jnp.asarray(p_s)[..., None] * jnp.asarray(self.sigma_full)

    def _dp(self, p_s):
        """Layer pressure thickness [Pa]; ``dA p_ref + dB p_s`` for hybrid."""
        if self.vcoord is not None:
            return self.vcoord.layer_thickness_dp(p_s)
        import jax.numpy as jnp
        return jnp.asarray(p_s)[..., None] * jnp.asarray(self.dsigma)

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
        p_s_np = np.asarray(p_s)
        field_np = np.asarray(field_3d)

        # Model pressure at each level, from the ACTUAL vertical coordinate:
        # p_k = A_k p_ref + B_k p_s (hybrid) or sigma_k p_s (pure sigma).
        # Shape: (..., nlev)
        p_model = np.asarray(self._p_full(p_s_np))

        # Target pressure levels (ascending for interpolation)
        plev_target = np.sort(CMIP6_PLEV19)  # ascending (100 Pa → 100000 Pa)

        # Log-pressure linear interpolation (numpy version)
        log_p_model = np.log(np.maximum(p_model, 1e-10))
        log_plev = np.log(plev_target)

        # Vectorise over the target-pressure axis instead of looping
        # ``n_target`` times.  Each target level only needed two
        # bracketing model levels and a log-linear interp; we can do
        # all target levels in a single ``take_along_axis`` by
        # broadcasting the searchsorted indices to ``(..., n_target)``.
        # Iter 11: 19 full-grid NumPy passes per 3-D field → 1 vectorised
        # pass per field.
        n_target = len(plev_target)

        # Bracketing index per cell, shape (..., n_target).  This counts the
        # model levels below each target pressure, which is exactly
        # ``searchsorted`` on the (ascending) per-column pressure profile.
        #
        # It replaces a 1-D ``searchsorted(sigma, plev/p_s)``: converting a
        # target PRESSURE to a single sigma and searching one shared 1-D
        # column is only valid for a PURE-sigma grid, where p = sigma*p_s makes
        # the level sigmas identical in every column.  On the (default) hybrid
        # grid p = A p_ref + B p_s, so the bracketing levels genuinely differ
        # per column and that shortcut selects the wrong pair over terrain.
        idx_hi = np.sum(p_model[..., None] < plev_target, axis=-2)
        idx_hi = np.clip(idx_hi, 1, p_model.shape[-1] - 1)
        idx_lo = idx_hi - 1

        f_lo = np.take_along_axis(field_np, idx_lo, axis=-1)
        f_hi = np.take_along_axis(field_np, idx_hi, axis=-1)
        lp_lo = np.take_along_axis(log_p_model, idx_lo, axis=-1)
        lp_hi = np.take_along_axis(log_p_model, idx_hi, axis=-1)

        log_pt = log_plev.reshape((1,) * p_s_np.ndim + (n_target,))
        denom = np.where(lp_hi == lp_lo, 1.0, lp_hi - lp_lo)
        alpha = np.clip((log_pt - lp_lo) / denom, 0.0, 1.0)
        return f_lo + alpha * (f_hi - f_lo)

    def _tas_2m(self, state, q_v, sst, sic, T_ice, u_low=None, v_low=None,
                *, T_land=None, q_land=None, land_fraction=None):
        """2 m air temperature for CMIP ``tas`` from the MOST surface-layer
        similarity profile (interpolate the lowest model level down to 2 m).
        Returns the lowest-level T when the surface inputs (SST / sigma) are
        unavailable — e.g. a prescribed-SST run that does not pass SST here.

        ``T_land`` / ``land_fraction`` (keyword-only) make the diagnostic
        LAND-AWARE.  Without them the surface is the blended SST/sea-ice skin
        with a saturated humidity EVERYWHERE, so the published ``tas`` over
        land is the neighbouring ocean's temperature rather than the model's
        land — the reason a February Arctic land bias could not be read from
        this field.  There is no below-canopy 2 m profile in this code, so the
        land branch uses the same MOST profile anchored on the land skin; do
        not read the result as a sub-canopy screen temperature.

        ``u_low`` / ``v_low`` override the lowest-level winds (lets the MPAS
        path pass the reconstructed CELL winds from
        ``reconstruct_cell_velocity`` instead of the edge-normal ``state.u``,
        which is not cell-collocated); default reads them off ``state``."""
        T_low = state.T.data[..., -1]
        if sst is None or self.sigma_full is None:
            return T_low
        import jax.numpy as jnp
        from legoesm import constants
        from legoesm.thermo import saturation_mixing_ratio
        from legoesm.core.bulk_flux import compute_most_fluxes
        u_low = state.u.data[..., -1] if u_low is None else u_low
        v_low = state.v.data[..., -1] if v_low is None else v_low
        q_low = q_v[..., -1] if q_v is not None else jnp.zeros_like(T_low)
        p_s = state.p_s.data
        p_low = jnp.asarray(self._p_full(p_s))[..., -1]
        rho_low = p_low / (constants.R_d * T_low)
        # Similarity profile matched to the experiment's surface fluxes:
        # SAME bulk scheme (constant -> the historical coare3 stand-in, see
        # __init__) and SAME stable-branch selector (codex 2026-08-02: a
        # large_yeager or grachev/gryanik run otherwise published a
        # coare3-native-default tas).  The 2 m value is set by stability, so
        # gustiness is left scheme-native here (pre-existing choice).
        def _profile_to_2m(T_sfc, q_sfc, *, ocean: bool):
            """One surface's 2 m temperature.

            MERGE NOTE (#1773), and it is a physics decision, not a textual
            one. ``main`` added a ``large_yeager_cesm`` arm that returns CESM
            ``shr_flux_atmOcn``'s own ``tref`` instead of the MOST profile;
            this branch split the diagnostic into a LAND surface and a SEA
            surface. ``shr_flux_atmOcn`` is the atmosphere-OCEAN coupler law
            (the module's own docstring: "SAM ocean surface fluxes"), with
            Charnock roughness and a saturated surface, so handing it a land
            skin would re-create the exact artifact this branch exists to
            remove -- land ``tas`` computed as if the surface were ocean.

            So the CESM arm applies to the SEA leg ONLY; the land leg keeps
            the MOST profile on every scheme. Review (GLM) flagged the
            alternative -- dispatching for both legs -- as a third scheme
            neither side tested.
            """
            if ocean and self.tas_profile_scheme == "large_yeager_cesm":
                # CESM shr_flux_atmOcn's own ``tref`` diagnostic (2 m).
                from legoesm.core.bulk_flux import compute_sam_oceflx_fluxes
                *_, T_2m = compute_sam_oceflx_fluxes(
                    u_low, v_low, T_low, q_low, T_sfc, q_sfc, rho_low,
                    z_bot=10.0, variant="cesm", return_2m=True,
                )
                return T_2m
            *_, T_2m = compute_most_fluxes(
                u_low, v_low, T_low, q_low, T_sfc, q_sfc, rho_low,
                scheme=(self.tas_profile_scheme
                        if self.tas_profile_scheme != "large_yeager_cesm"
                        else "large_yeager"),
                return_2m=True,
                stability_scheme=self.surface_stability_scheme,
            )
            return T_2m

        T_sfc_sea = blend_surface_temperature(sst, sic, T_ice)
        q_sfc_sea = saturation_mixing_ratio(T_sfc_sea, p_s)
        if T_land is None or land_fraction is None:
            # Legacy path, bit-identical: ONE profile anchored on the
            # ocean/sea-ice skin with a saturated surface.  Over land that is
            # the neighbouring ocean's temperature and a saturated surface the
            # land does not have, so a caller that owns a land skin should pass
            # it — see the land branch below.
            return _profile_to_2m(T_sfc_sea, q_sfc_sea, ocean=True)

        # Land and sea get their OWN surface, then combine by land fraction.
        # Both branches keep the same bulk scheme and stability selector, so
        # this introduces no second convention — only the missing surface.
        # ``q_land`` is a surface mixing ratio; without one the land surface is
        # treated as saturated, which is what the legacy path assumed
        # everywhere and is an OVERESTIMATE of surface humidity over dry or
        # frozen ground.
        q_sfc_land = (saturation_mixing_ratio(T_land, p_s) if q_land is None
                      else q_land)
        T_2m_land = _profile_to_2m(T_land, q_sfc_land, ocean=False)
        T_2m_sea = _profile_to_2m(T_sfc_sea, q_sfc_sea, ocean=True)
        f_land = jnp.clip(jnp.asarray(land_fraction, dtype=T_2m_land.dtype),
                          0.0, 1.0)
        return f_land * T_2m_land + (1.0 - f_land) * T_2m_sea

    def _cam6_cloud_kwargs(self, cloud_fraction, conv_mass_flux_up,
                           conv_icwmr, lat_deg, p_s, ncol, nlev):
        """Extra ``compute_cloud_properties`` kwargs for cloud_scheme
        'cam6_clubb': the CLUBB cloud fraction + deep-convection carries, lat
        [rad] and interface pressures.  Empty (byte-identical call) for every
        other scheme; a cam6 run without the carry raises here rather than
        silently scoring an RH cloud cover the radiation never used.

        SAMPLING: the carry handed in is the one written by THIS step's
        physics (CLUBB's current PDF fraction), applied to the current state;
        the radiation call read the carry as it stood at its last update
        (one physics step behind, more under radiation subcycling).  clt/clivi
        therefore describe the model's CURRENT cloud field, not a readout of
        what radiation integrated -- the same convention the conv_precip
        convective cover and the RH schemes' clt already follow."""
        if getattr(self._cloud_config, "scheme", None) != "cam6_clubb":
            return {}
        if (cloud_fraction is None or conv_mass_flux_up is None
                or conv_icwmr is None or lat_deg is None):
            raise ValueError(
                "cloud_scheme='cam6_clubb' diagnostics need the lagged "
                "cloud_fraction / conv_mass_flux_up / conv_icwmr carries and "
                "lat_deg; got None.")
        if self.vcoord is None:
            raise ValueError(
                "cloud_scheme='cam6_clubb' diagnostics need interface "
                "pressures (a vertical coordinate on the collector); got none.")
        return dict(
            p_half=jnp.reshape(
                jnp.asarray(self.vcoord.pressure_at_half(
                    jnp.reshape(jnp.asarray(p_s), (ncol,)))), (ncol, nlev + 1)),
            cloud_fraction_override=jnp.reshape(
                jnp.asarray(cloud_fraction), (ncol, nlev)),
            lat=jnp.deg2rad(jnp.reshape(jnp.asarray(lat_deg), (ncol,))),
            conv_mass_flux_up=jnp.reshape(
                jnp.asarray(conv_mass_flux_up), (ncol, nlev + 1)),
            conv_icwmr=jnp.reshape(jnp.asarray(conv_icwmr), (ncol, nlev)),
        )

    def _clt_percent(self, T, p_s, q_v, q_c, q_i=None, *, cloud_fraction=None,
                     conv_mass_flux_up=None, conv_icwmr=None, lat_deg=None):
        """Total cloud cover [%] under MAXIMUM-RANDOM overlap, or ``None``.

        SHARED by the cube/lat-lon :meth:`collect` path and the lean MPAS
        :meth:`feed_cmip_accumulators_native` path.  The two lanes must not
        carry separate copies of this reduction: a drift between them would
        report a different cloud cover for the SAME state depending only on
        which dycore ran.

        Arrays are ``(*horiz, nlev)`` with ``p_s`` ``(*horiz,)``.  The
        horizontal dims are flattened to the ``(ncol, nlev)`` column layout
        ``compute_cloud_properties`` documents and the scalar overlap result is
        reshaped back — C-order round-trips exactly, so cells map back for the
        downstream regridder.  Works for cubed-sphere ``(6,n,n,nlev)``,
        lat-lon ``(nlat,nlon,nlev)`` and native MPAS ``(nCells,nlev)``.

        Returns ``None`` when the run has no cloud scheme (``cloud_scheme=
        "none"``) or carries no liquid/vapour tracer, so a caller can simply
        skip the field rather than publish a zero that reads as "clear".

        CAVEAT -- this is MAXIMUM-RANDOM overlap, the CMIP convention for
        ``clt``, which is NOT necessarily the overlap the radiation applied.
        The subcolumn generator (``clouds/subcolumns.py``) is OPT-IN, so on the
        default path radiation sees no vertical-overlap treatment at all.
        ``clt`` is therefore a statement about the model's CLOUD FIELD, not a
        readout of what the radiation integrated: do not difference it against
        a shortwave bias and call the residual an overlap error.
        """
        if self._cloud_config is None or q_c is None or q_v is None:
            return None
        from legoesm.atmosphere.physics.clouds.cloud_fraction import (
            compute_cloud_properties,
        )
        horiz_shape = T.shape[:-1]
        nlev = T.shape[-1]
        ncol = int(np.prod(horiz_shape)) if horiz_shape else 1
        # Level pressures from the ACTUAL vertical coordinate -- p_s *
        # sigma_full is wrong on the (default) hybrid grid.
        p_s_col = jnp.reshape(p_s, (ncol,))
        p_full = jnp.asarray(self._p_full(p_s_col))
        dp = jnp.asarray(self._dp(p_s_col))
        # Cloud fraction takes CLOUD ice q_i only, matching the radiation
        # call (physics_pipeline ``q_ice=q_i_col``) and now also the clivi ice
        # path.  Including the precipitating q_s/q_g would over-count
        # condensate for the condensate-dependent schemes
        # (xu_randall/resolved).
        q_ice_col = None if q_i is None else jnp.reshape(q_i, (ncol, nlev))
        cloud_props = compute_cloud_properties(
            jnp.reshape(T, (ncol, nlev)),
            p_full,
            jnp.reshape(q_v, (ncol, nlev)),
            dp,
            self._cloud_config,
            q_cloud=jnp.reshape(q_c, (ncol, nlev)),
            q_ice=q_ice_col,
            **self._cam6_cloud_kwargs(
                cloud_fraction, conv_mass_flux_up, conv_icwmr, lat_deg,
                p_s, ncol, nlev),
        )
        return np.asarray(
            jnp.reshape(maximum_random_overlap(cloud_props.cloud_fraction),
                        horiz_shape)
        ) * 100.0  # CMIP units: %

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
        sw_up_toa_clr=None,
        lw_up_toa_clr=None,
        q_i=None,
        q_s=None,
        q_g=None,
        t_low_mean=None,
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
        q_i : jax.Array, optional
            Cloud-ice mixing ratio [kg/kg]; ``None`` for warm-rain-only
            microphysics (e.g. kessler) that carries no ice tracer.
        sst, sic : jax.Array
        precip_total : jax.Array
            Total precipitation [kg/m2/s].
        sw_up_toa, lw_up_toa : jax.Array
        sw_net_sfc, lw_net_sfc : jax.Array
        sw_down_toa : jax.Array
            Radiative fluxes.  The compiled-segment driver passes SEGMENT
            MEANS (time integrals from the carry accumulators / segment
            duration) so the CMOR monthly means, timeseries and energy
            budget are free of the fixed-UTC diurnal snapshot alias; the
            per-step (debug) driver still passes instantaneous values.
        T_ice : float
        lat_deg_grid : array, optional
            Latitude in degrees for monthly means.
        t_low_mean : array, optional
            Segment-mean lowest-level air temperature [K].  When given, the
            CMOR ``tas`` uses it in place of the instantaneous lowest-level
            temperature: ``tas = tas_2m(instant) + (t_low_mean - T_low)``,
            i.e. the instantaneous MOST 2 m stability offset applied to the
            segment-mean temperature (removes the dominant fixed-UTC
            diurnal alias over land; the residual alias of the stability
            offset itself is small).  ``None`` (default) keeps the legacy
            instantaneous ``tas``.
        """
        # Fuse 12 diagnostic reductions into one ``jnp.stack`` +
        # ``np.asarray`` host transfer.  Each ``float(jnp.X(...))``
        # was previously its own device→host sync, serialising the
        # GPU pipeline at every diagnostic interval.  The model step
        # following ``collect()`` cannot launch until all 12 have
        # round-tripped — fusing them collapses the stall to one.
        if hasattr(state, 'v'):
            wind_term = jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2))
        else:
            wind_term = jnp.max(jnp.abs(state.u.data))
        cwv = column_water_vapor(q_v, state.p_s.data, self.dsigma,
                                 dp=self._dp(state.p_s.data))
        _aw = self._area_w
        # Build a zero-padded sentinel for optional fields (shflx, lhflx,
        # sw_down_toa) so they can be fused into the single device→host
        # transfer.  Using jnp.zeros(()) keeps the scalar shape uniform.
        _zero = jnp.zeros((), dtype=state.T.data.dtype)
        _stats = jnp.stack([
            area_weighted_mean(sst, _aw),
            area_weighted_mean(sic, _aw),
            area_weighted_mean(state.T.data, _aw),
            area_weighted_mean(state.T.data[..., -1], _aw),
            wind_term,
            area_weighted_mean(precip_total, _aw),
            area_weighted_mean(cwv, _aw),
            area_weighted_mean(sw_up_toa, _aw),
            area_weighted_mean(lw_up_toa, _aw),
            area_weighted_mean(state.p_s.data, _aw),
            area_weighted_mean(sw_net_sfc, _aw),
            area_weighted_mean(lw_net_sfc, _aw),
            area_weighted_mean(sw_down_toa, _aw) if sw_down_toa is not None else _zero,
            area_weighted_mean(shflx, _aw) if shflx is not None else _zero,
            area_weighted_mean(lhflx, _aw) if lhflx is not None else _zero,
        ])
        _stats_host = np.asarray(_stats)
        mean_sst = float(_stats_host[0])
        mean_sic = float(_stats_host[1])
        mean_T = float(_stats_host[2])
        mean_T_low = float(_stats_host[3])
        max_v = float(_stats_host[4])
        mean_precip = float(_stats_host[5]) * 86400.0
        mean_cwv = float(_stats_host[6])
        mean_sw_toa = float(_stats_host[7])
        mean_lw_toa = float(_stats_host[8])
        mean_ps = float(_stats_host[9])
        mean_sw_sfc = float(_stats_host[10])
        mean_lw_sfc = float(_stats_host[11])
        mean_rsdt = float(_stats_host[12]) if sw_down_toa is not None else float('nan')
        mean_hfss = float(_stats_host[13]) if shflx is not None else float('nan')
        mean_hfls = float(_stats_host[14]) if lhflx is not None else float('nan')

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
        self.rsdt.append(mean_rsdt)
        self.hfss.append(mean_hfss)
        self.hfls.append(mean_hfls)

        # Mean over all spatial axes except the last (vertical).
        # Cubed-sphere: (6,n,n,nlev) → mean over (0,1,2) → (nlev,)
        # Lat-lon:      (nlat,nlon,nlev) → mean over (0,1) → (nlev,)
        # Stacked into one ``np.asarray`` host transfer (same dtype as
        # T) so the two profile means share a single device→host sync.
        _profiles_host = np.asarray(jnp.stack([
            area_weighted_profile(state.T.data, self._area_w),
            area_weighted_profile(q_v, self._area_w).astype(state.T.data.dtype),
        ]))
        self.profiles_T.append(_profiles_host[0])
        self.profiles_qv.append(_profiles_host[1] * 1000.0)

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
                'hfls': (np.asarray(lhflx) if lhflx is not None
                         else np.zeros_like(np.asarray(sst))),
                'hfss': (np.asarray(shflx) if shflx is not None
                         else np.zeros_like(np.asarray(sst))),
            }

        # Energy budget
        elapsed_s = elapsed_day * 86400.0
        self.energy_tracker.update(
            state.T.data, q_v, state.u.data, state.v.data,
            state.phis.data, state.p_s.data,
            self.dsigma, self.sigma_full,
            sw_down_toa, sw_up_toa, lw_up_toa, sw_net_sfc, lw_net_sfc,
            elapsed_seconds=elapsed_s,
            area_weights=self._area_w,
            dp=self._dp(state.p_s.data),
            p_full=self._p_full(state.p_s.data),
            q_frozen=_frozen_condensate(state),
        )

        # Moisture budget.  lhflx is the SAME field reported as CMOR hfls
        # (positive-up evaporation source) so the E − P − dW/dt closure
        # residual shares one flux definition with the output diagnostics;
        # a fluxless config (lhflx None) closes against E = 0.
        self.moisture_tracker.update(
            q_v, state.p_s.data, self.dsigma,
            precip_total,
            lhflx if lhflx is not None else jnp.zeros_like(state.p_s.data),
            elapsed_seconds=elapsed_s,
            area_weights=self._area_w,
            dp=self._dp(state.p_s.data),
        )

        # Monthly means
        if self.monthly_means and self.monthly_accum is not None and lat_deg_grid is not None:
            from legoesm import constants as _c
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
            # rsdt: TOA incoming SW [W/m²] — available unconditionally
            if sw_down_toa is not None:
                fields_2d['rsdt'] = np.asarray(sw_down_toa)
            # hfss / hfls: surface heat fluxes [W/m²]
            if shflx is not None:
                fields_2d['hfss'] = np.asarray(shflx)
            if lhflx is not None:
                fields_2d['hfls'] = np.asarray(lhflx)
            # psl: sea-level pressure via hypsometric equation
            # p_sl = p_s * exp(phis / (R_d * T_lowest))
            # Reuse the T_low array already materialised above.
            _phis_np = np.asarray(state.phis.data)
            _ps_np = np.asarray(state.p_s.data)
            fields_2d['psl'] = _ps_np * np.exp(
                _phis_np / (_c.R_d * np.maximum(fields_2d['T_low'], _c.T_min_atmosphere))
            )
            self.monthly_accum.add_2d(doy, year, fields_2d, lat_deg_grid)
            # profile_u: use geographic eastward wind when rotation angle is
            # available (cubed-sphere).  Panel-local u averaged over latitude
            # bands produces sign cancellations across cube faces.
            # Rotation: u_east = cos(angle)*u_grid - sin(angle)*v_grid
            # (pure NumPy — no JAX dependency in this diagnostic path).
            if (self._wind_rotation_angle is not None
                    and hasattr(state, 'v') and state.v is not None):
                _angle = self._wind_rotation_angle
                # Angle is (6,n,n); winds are (6,n,n,nlev) — broadcast vertically
                _cos_a = np.cos(_angle)[..., np.newaxis]
                _sin_a = np.sin(_angle)[..., np.newaxis]
                _u_np = np.asarray(state.u.data)
                _v_np = np.asarray(state.v.data)
                _u_for_profile = _cos_a * _u_np - _sin_a * _v_np
            else:
                _u_for_profile = np.asarray(state.u.data)
            fields_3d = {
                'T': np.asarray(state.T.data),
                'u': _u_for_profile,
                'q_v': np.asarray(q_v) * 1000.0,
            }
            # Cloud-water profiles (g/kg). Guarded: kessler carries q_c but
            # no q_i (None); morrison carries both. Only emit a field when
            # its tracer is present so warm-rain runs don't fabricate a
            # zero q_i profile.
            if q_c is not None:
                fields_3d['q_c'] = np.asarray(q_c) * 1000.0
            if q_i is not None:
                fields_3d['q_i'] = np.asarray(q_i) * 1000.0
            self.monthly_accum.add_3d(doy, year, fields_3d, lat_deg_grid)
            self.monthly_accum.add_scalar(doy, year, {
                'T_atm': mean_T,
                'T_low': mean_T_low,
                'precip': mean_precip,
                'sw_up_toa': mean_sw_toa,
                'lw_up_toa': mean_lw_toa,
                'rsdt': mean_rsdt,
                'hfss': mean_hfss,
                'hfls': mean_hfls,
            })

        # Spatial monthly accumulation for CMIP output
        if self._spatial_monthly is not None:
            from legoesm import constants as _c

            doy, _ = day_to_calendar(day)
            year = int(day // 365.0)
            fields_2d = {}
            # tas: 2 m air temperature.  The lowest model level (~100 m at
            # nlev=20) reads colder than 2 m over a warm surface, so use the
            # MOST surface-layer similarity profile to interpolate the lowest
            # level down to 2 m (CMIP tas convention).  Falls back to the
            # lowest level if the surface-layer inputs are unavailable.
            tas_field = self._tas_2m(state, q_v, sst, sic, T_ice)
            if t_low_mean is not None:
                # Segment-mean tas: instantaneous 2 m stability offset on the
                # segment-mean lowest-level T (see the ``t_low_mean`` doc).
                tas_field = tas_field + (
                    np.asarray(t_low_mean) - np.asarray(state.T.data[..., -1])
                )
            r = self._regrid_to_latlon_2d(tas_field)
            if r is not None:
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
            # Clear-sky TOA outgoing fluxes (Phase 1 of cloud-micro plan):
            # populated by RRTMGP when do_clear_sky=True; zeros otherwise.
            # SW_CRE = rsut - rsutcs, LW_CRE = rlutcs - rlut.
            if sw_up_toa_clr is not None:
                r = self._regrid_to_latlon_2d(sw_up_toa_clr)
                if r is not None:
                    fields_2d['rsutcs'] = r
            if lw_up_toa_clr is not None:
                r = self._regrid_to_latlon_2d(lw_up_toa_clr)
                if r is not None:
                    fields_2d['rlutcs'] = r
            r = self._regrid_to_latlon_2d(state.p_s.data)
            if r is not None:
                fields_2d['ps'] = r

            # rsdt: TOA incoming shortwave [W/m2] — correctly available
            r = self._regrid_to_latlon_2d(sw_down_toa)
            if r is not None:
                fields_2d['rsdt'] = r

            # tos / siconc: sea-surface temperature [K] (CMOR Omon) and
            # sea-ice area fraction [%] (CMOR SImon).  sst/sic already arrive
            # on the model grid (prescribed for AMIP; from the coupler /
            # get_sst_sic override for a coupled run), so the same regridder
            # used for atmosphere fields applies.  tos is in K (matching the
            # repo CMOR table units), so no conversion; siconc is fraction*100.
            # NOTE: emitted UNMASKED.  The coupled driver is slab-ocean today
            # (no land/sea mask is in collect()'s scope), so these are not yet
            # masked to "where sea" per the CMOR cell_methods.  Mask them when a
            # prognostic ocean + land mask is wired through the coupled driver.
            r = self._regrid_to_latlon_2d(sst)
            if r is not None:
                fields_2d['tos'] = r
            r = self._regrid_to_latlon_2d(np.asarray(sic) * 100.0)
            if r is not None:
                fields_2d['siconc'] = r

            # NOTE: rsds/rlds (surface downwelling) are NOT computed here.
            # The runtime only provides sw_net_sfc/lw_net_sfc (net fluxes),
            # which are not equal to the downwelling component.  Publishing
            # net fluxes under CMIP downwelling names would be scientifically
            # incorrect.  These variables will be added when the physics
            # pipeline exposes separate downwelling surface fluxes.

            # Cloud-ice path (clivi), condensed-water path (clwvi) and total
            # cloud cover (clt).  CMIP convention: clivi = column-integrated
            # frozen condensate that is RADIATIVELY ACTIVE (here cloud ice
            # only — see the q_frozen block below); clwvi = that plus the
            # liquid path.  Earlier code hardcoded clivi=0 (a warm-rain-era
            # placeholder) which threw away all model ice; the fix then
            # over-corrected by summing q_i+q_s+q_g.
            # clt is a random-overlap approximation of a soft layer cloud
            # fraction (sigmoid on total condensate) — useful for spatial
            # diagnosis of cloud-deficit regions, not a max-random overlap scheme.
            if q_c is not None:
                lwp_field = np.asarray(
                    column_water_vapor(q_c, state.p_s.data, self.dsigma,
                                       dp=self._dp(state.p_s.data))
                )
                # Frozen condensate path: CLOUD ICE ONLY.
                #
                # CMIP6 defines ``clivi`` as the column ice mass, "including
                # precipitating frozen hydrometeors ONLY IF the precipitating
                # hydrometeor affects the calculation of radiative transfer in
                # model".  Radiation here takes cloud ice alone --
                # ``radiation/integration.py`` reads ``tracers["q_i"]`` (:411)
                # / tracer slot 3 (:1436) and never receives snow (slot 4) or
                # graupel (slot 5) -- so snow and graupel are radiatively INERT
                # and must NOT be reported.
                #
                # They previously were, and it is not a small correction:
                # snow+graupel were roughly half the published clivi on
                # lat-lon and ~70% on MPAS in this campaign's checkpoints.
                # (Sizes are indicative only -- observational IWP products
                # differ in whether they include precipitating ice, so this
                # does NOT by itself establish the sign of a model bias.)
                #
                # If precipitating ice is ever made radiatively active, CMIP6
                # requires it here -- but NOT by re-summing ``q_s``/``q_g``.
                # That sum is not even dimensionally valid across schemes: P3
                # aliases the slot-4/5 tracers to rime MASS and rime VOLUME
                # (``microphysics/p3.py``), so adding them to a mass mixing
                # ratio is wrong.  Any future treatment must be SCHEME-AWARE
                # and carry its own PSD/effective radius, exactly as the
                # optics would need.
                q_frozen = q_i
                if q_frozen is not None:
                    iwp_field = np.asarray(
                        column_water_vapor(q_frozen, state.p_s.data, self.dsigma,
                                           dp=self._dp(state.p_s.data))
                    )
                else:
                    iwp_field = np.zeros_like(lwp_field)
                r_lwp = self._regrid_to_latlon_2d(lwp_field)
                r_iwp = self._regrid_to_latlon_2d(iwp_field)
                if r_lwp is not None and r_iwp is not None:
                    fields_2d['clivi'] = r_iwp
                    fields_2d['clwvi'] = r_lwp + r_iwp  # liquid + frozen

                # Total cloud cover (clt, CMIP %) from the MODEL's fractional
                # layer cloud fraction — the SAME sundqvist/xu_randall/resolved
                # scheme radiation uses, via the shared ``compute_cloud_properties``
                # — reduced by MAXIMUM-RANDOM vertical overlap.  This replaces the
                # retired near-binary condensate mask (sigmoid sharpness 1e6 on a
                # 1 mg/kg threshold + pure random overlap) that saturated clt to
                # ~100% wherever a column held any trace of condensate (issue #689).
                # Diagnostics-only; no cloud-fraction numerics are re-derived (the
                # shared function is called).  The ``_cloud_config`` is built with
                # ``convective_cloud=False`` (see ModelDriver._create_diagnostics),
                # so this is the model's STRATIFORM cloud cover — the opt-in
                # convective radiative-tuning add-on is intentionally not counted.
                clt_field = self._clt_percent(
                    state.T.data, state.p_s.data, q_v, q_c, q_i)
                if clt_field is not None:
                    r_clt = self._regrid_to_latlon_2d(clt_field)
                    if r_clt is not None:
                        fields_2d['clt'] = r_clt

            # psl: sea-level pressure via hypsometric equation
            # p_sl = p_s * exp(phis / (R_d * T_lowest))
            # This is a standard GCM approximation; a more accurate
            # extrapolation (e.g., WMO method) would improve results
            # over steep topography.
            T_lowest = np.asarray(state.T.data[..., -1])
            phis = np.asarray(state.phis.data)
            p_s_np = np.asarray(state.p_s.data)
            T_lowest_safe = np.maximum(T_lowest, _c.T_min_atmosphere)
            psl = p_s_np * np.exp(phis / (_c.R_d * T_lowest_safe))
            r = self._regrid_to_latlon_2d(psl)
            if r is not None:
                fields_2d['psl'] = r

            # prw: column water vapor [kg/m2]
            cwv_field = np.asarray(
                column_water_vapor(q_v, state.p_s.data, self.dsigma,
                                   dp=self._dp(state.p_s.data))
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

            # Daily accumulation (CMIP6 ``day`` table).  Reuses the 2-D
            # regridded fields computed above, plus 850 hPa winds sliced
            # out of the already-regridded 3-D ua/va arrays.
            if self._spatial_daily is not None:
                daily_2d: dict[str, np.ndarray] = {}
                for _name in ("tas", "pr", "psl"):
                    if _name in fields_2d:
                        daily_2d[_name] = fields_2d[_name]
                # 850 hPa is index 16 in the ascending-sorted PLEV19 axis
                # (same sort order used by ``_interp_to_plev19``).
                _plev_sorted = np.sort(CMIP6_PLEV19)
                _idx850 = int(np.argmin(np.abs(_plev_sorted - 85000.0)))
                for _src, _dst in (("ua", "ua850"), ("va", "va850")):
                    if _src in fields_3d and fields_3d[_src].shape[2] > _idx850:
                        daily_2d[_dst] = fields_3d[_src][:, :, _idx850]
                if daily_2d:
                    self._spatial_daily.add_2d(doy, year, daily_2d)

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
        sw_down_toa=None,
        shflx=None,
        lhflx=None,
    ) -> dict:
        """Collect only scalar reduction diagnostics (no host materialization).

        This is the performance-mode alternative to :meth:`collect`.
        It computes global means via ``area_weighted_mean`` (cell-area
        weighted when ``set_area_weights`` was called, else a plain
        ``jnp.mean``) and max-wind via ``jnp.max``; both work correctly on
        SPMD-sharded arrays (JAX handles cross-device reductions
        internally).  No ``np.asarray()`` calls, no snapshot capture, no
        profile extraction, no monthly means.

        Use this for scaling benchmarks where diagnostic overhead must
        not dominate wall-clock time.

        Returns the same dict keys as ``collect`` for logging compatibility.
        """
        # Fuse the 12 reductions into one ``jnp.stack`` + ``np.asarray``
        # device→host transfer.  The "minimal" docstring promised low
        # overhead, but the previous per-scalar ``float(...)`` chain
        # serialised 12 GPU stalls per diagnostic step — exactly the
        # sin the long ``collect`` path was already corrected for.
        if hasattr(state, 'v'):
            wind_term = jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2))
        else:
            wind_term = jnp.max(jnp.abs(state.u.data))
        cwv = column_water_vapor(q_v, state.p_s.data, self.dsigma,
                                 dp=self._dp(state.p_s.data))
        _aw = self._area_w
        _stats = jnp.stack([
            area_weighted_mean(sst, _aw),
            area_weighted_mean(sic, _aw),
            area_weighted_mean(state.T.data, _aw),
            area_weighted_mean(state.T.data[..., -1], _aw),
            wind_term,
            area_weighted_mean(precip_total, _aw),
            area_weighted_mean(cwv, _aw),
            area_weighted_mean(sw_up_toa, _aw),
            area_weighted_mean(lw_up_toa, _aw),
            area_weighted_mean(state.p_s.data, _aw),
            area_weighted_mean(sw_net_sfc, _aw),
            area_weighted_mean(lw_net_sfc, _aw),
        ])
        _h = np.asarray(_stats)
        mean_sst = float(_h[0])
        mean_sic = float(_h[1])
        mean_T = float(_h[2])
        mean_T_low = float(_h[3])
        max_v = float(_h[4])
        mean_precip = float(_h[5]) * 86400.0
        mean_cwv = float(_h[6])
        mean_sw_toa = float(_h[7])
        mean_lw_toa = float(_h[8])
        mean_ps = float(_h[9])
        mean_sw_sfc = float(_h[10])
        mean_lw_sfc = float(_h[11])

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
        # rsdt/hfss/hfls: recorded when the caller provides them (a real AMIP
        # run does; a bare scaling benchmark does not), else NaN so the
        # timeseries arrays stay aligned across flush chunks.
        _rsdt = (float(area_weighted_mean(sw_down_toa, _aw))
                 if sw_down_toa is not None else float('nan'))
        _hfss = (float(area_weighted_mean(shflx, _aw))
                 if shflx is not None else float('nan'))
        _hfls = (float(area_weighted_mean(lhflx, _aw))
                 if lhflx is not None else float('nan'))
        self.rsdt.append(_rsdt)
        self.hfss.append(_hfss)
        self.hfls.append(_hfls)

        # Energy-budget tracker on the lightweight path too (#1354/#1515): the
        # tracker is grid-agnostic (column integral over the trailing level
        # axis, area weighting over the leading horizontal axes — works on the
        # MPAS ``(nCells, nlev)`` layout exactly as on the cube ``(6,n,n,nlev)``
        # one).  Runs only when the radiation fluxes are present, so a bare
        # scaling benchmark that passes no fluxes keeps its lean path.
        _v_data = getattr(state, 'v', None)
        _v_data = _v_data.data if _v_data is not None else None
        if sw_down_toa is not None and q_v is not None and _v_data is not None:
            # Only the grids that carry a cell-centred v (cube / lat-lon) run
            # the tracker here; the MPAS edge-wind lane runs it in _run_mpas
            # with a Perot reconstruction (a None v here means edge winds we
            # must not feed the column KE term as if they were cell winds).
            self.energy_tracker.update(
                state.T.data, q_v, state.u.data, _v_data,
                state.phis.data, state.p_s.data,
                self.dsigma, self.sigma_full,
                sw_down_toa, sw_up_toa, lw_up_toa, sw_net_sfc, lw_net_sfc,
                elapsed_seconds=elapsed_day * 86400.0,
                area_weights=_aw,
                dp=self._dp(state.p_s.data),
                p_full=self._p_full(state.p_s.data),
                q_frozen=_frozen_condensate(state),
            )

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

    def feed_daily_extremes_native(self, day: float, *, tas) -> None:
        """Regrid an hourly temperature sample without feeding any means."""
        if self._spatial_daily is None:
            return
        field = self._regrid_to_latlon_2d(np.asarray(tas, dtype=np.float64))
        if field is not None:
            doy, _ = day_to_calendar(day)
            self._spatial_daily.add_extremes_2d(
                doy, int(day // 365.0), {"tas": field})

    def feed_cmip_accumulators_native(
        self,
        day: float,
        *,
        T,
        p_s,
        lat_deg=None,
        cloud_fraction=None,
        conv_mass_flux_up=None,
        conv_icwmr=None,
        q_v=None,
        q_c=None,
        q_i=None,
        u_east=None,
        v_north=None,
        precip=None,
        phis=None,
        tas=None,
        ts=None,
        rlut=None,
        rsut=None,
        rsdt=None,
        hfss=None,
        hfls=None,
        rsutcs=None,
        rlutcs=None,
        wap=None,
        flux_interval_days=None,
        include_state=True,
    ) -> bool:
        """Feed the CMIP spatial (``Amon``/``day``) + zonal-mean monthly
        accumulators from NATIVE-grid host arrays, bypassing the heavy
        :meth:`collect` path.

        This is the lean-driver counterpart to :meth:`collect`'s spatial
        block.  The MPAS (Voronoi) execution path materialises only a handful
        of diagnostic arrays per interval and never assembles :meth:`collect`'s
        full kwarg set (the surface/radiation flux block), so it historically
        fed NOTHING into the CMOR accumulators (manifests showed
        ``call_counts: []`` / ``max_count_ever: 0`` even though the sidecar
        SAVE ran).  This feeds the subset the lean path CAN provide through the
        SAME regrid + plev19 interpolation helpers :meth:`collect` uses, so the
        published fields share one pipeline:

        * 2-D (``add_2d``): ``tas`` (2 m air temperature — MOST similarity when
          the caller supplies *tas*, else the lowest-model-level fallback),
          ``ps``, ``pr`` (when *precip* given), ``prw`` (column water vapour,
          when *q_v* given), ``psl`` (hypsometric, when *phis* given).
        * 3-D on plev19 (``add_3d``): ``ta``, ``hus`` (*q_v*), ``ua``
          (*u_east*), ``va`` (*v_north*).
        * Daily (``SpatialDailyAccumulator``): ``tas``/``pr``/``psl`` plus
          ``ua850``/``va850`` sliced from the regridded 3-D winds.
        * Zonal (``MonthlyAccumulator``): ``T_low``/``precip``/``psl`` 2-D
          bands and ``T``/``u``/``q_v`` profiles (needs *lat_deg*).

        Sampling / accuracy caveats (documented, not silently hidden):

        * STATE-derived fields are sampled hourly by the MPAS driver, before
          cloud diagnosis, pressure interpolation and regridding. Their means
          therefore sample the entire diurnal cycle. The independent diagnostic
          feed passes ``include_state=False`` and supplies only per-step flux
          interval means from ``_MPASSfcFluxAccum``; it cannot double-count state.
        * ``ua850``/``va850`` come from :meth:`_interp_to_plev19`, which BOUNDED
          -extrapolates below the lowest model level (inherited shared-helper
          behaviour, identical to :meth:`collect`; not masked to NaN).
        * The zonal ``MonthlyAccumulator`` arithmetic-averages cells per lat
          band (inherited); exact only for equal-area cells — a variable-
          resolution MPAS mesh biases the zonal mean.  The SPATIAL (regridded)
          CMOR path is unaffected (area-neutral IDW to a regular grid).

        Commit is ATOMIC: every regrid / plev interpolation AND the zonal
        lat-length check (the failure-prone work) run in PHASE 1, before any
        ``add_*`` in PHASE 2 — the spatial fields are shape-guaranteed by
        ``_regrid_to_latlon_*``, so a mid-computation error leaves all three
        accumulators untouched rather than half-updated.

        Parameters
        ----------
        day : float
            ABSOLUTE simulated day (``start_day + elapsed``).  The month/year
            bucket keys are ``doy, _ = day_to_calendar(day)`` and
            ``year = int(day // 365)`` — identical to :meth:`collect`, so a
            restart chain keeps monotonic calendar months.
        T : array, shape ``(nCells, nlev)``
            Air temperature [K] on model levels (level ``-1`` = lowest/surface).
        p_s : array, shape ``(nCells,)``
            Surface pressure [Pa].
        lat_deg : array, shape ``(nCells,)``, optional
            Cell-centre latitude [degrees] for the zonal-mean bands.  When
            ``None`` the zonal accumulator is skipped.
        q_v : array, shape ``(nCells, nlev)``, optional
            Water-vapour mixing ratio [kg/kg].
        u_east, v_north : array, shape ``(nCells, nlev)``, optional
            GEOGRAPHIC cell-centre zonal / meridional wind [m/s] (MPAS: from
            ``reconstruct_cell_velocity`` on the edge-normal ``state.u``).
        precip : array, shape ``(nCells,)``, optional
            Surface precipitation rate [kg/m2/s] (CMOR ``pr`` units).
        phis : array, shape ``(nCells,)``, optional
            Surface geopotential [m2/s2] for the sea-level-pressure reduction.
        tas : array, shape ``(nCells,)``, optional
            2 m air temperature [K] (MOST similarity, computed by the caller
            from sst/sic + surface-layer winds).  Falls back to the lowest
            model level when ``None`` so the field is never dropped.
        ts : array, shape ``(nCells,)``, optional
            Whole-cell land/sea/ice blended surface skin temperature [K].
        rsutcs, rlutcs : array, shape ``(nCells,)``, optional
            CLEAR-SKY TOA outgoing SW / LW flux [W/m2, positive up — the same
            CMOR sign as rsut/rlut] from the clouds-off second radiation pass
            (``--clear-sky-diag``, #843).  Interval means like the other flux
            fields; absent (None) keeps the output byte-identical.
        q_c, q_i : array, shape ``(nCells, nlev)``, optional
            CLOUD liquid / CLOUD ice mixing ratio [kg/kg] (the tracers the
            radiation's cloud diagnosis consumes — NOT the precipitating
            q_s/q_g).  When supplied together with *q_v* and a collector
            ``cloud_config``, three cloud CMOR fields are derived:

            * ``clt`` — stratiform total cloud cover [%] via the model's own
              layer cloud fraction reduced by MAXIMUM-RANDOM overlap (the
              shared ``compute_cloud_properties`` + ``maximum_random_overlap``
              pair the cube-lane :meth:`collect` uses).
            * ``clwvi`` / ``clivi`` — the DIAGNOSTIC condensed-water / ice
              water path [kg/m2]: column sums of the per-layer grid-mean
              lwp/iwp that ``compute_cloud_properties`` builds from this
              collector's own ``cloud_config``, i.e. INCLUDING that scheme's
              sub-grid in-cloud condensate floor (the diagnostic-fraction
              schemes sundqvist / xu_randall; ``resolved`` has no floor) and
              EXCLUDING precipitating snow/graupel.  Reported in mass units,
              so the cloud-optics sub-grid inhomogeneity and partial-coverage
              thinning factors (radiative-transfer corrections, not mass) are
              pinned off.

              This is the condensate-floor-inclusive path the albedo/CRE
              investigation needs — the prognostic-only path (the cube-lane
              :meth:`collect` clwvi/clivi, which integrate q_c+q_i+q_s+q_g)
              reads ~0 wherever the coarse grid-mean never saturates.  It is
              NOT a faithful record of what radiation solved with, in exactly
              two respects: this config is stratiform-only
              (``convective_cloud=False``, #689) and carries no CLUBB
              cloud-fraction override.  Every other cloud knob radiation uses
              (rh_crit, q_c_diagnostic, Xu-Randall p/alpha, the
              diagnostic-condensate scheme and its adiabatic rate) IS threaded
              here by ModelDriver._create_diagnostics, so the floor formula
              matches.  Closing the two gaps needs radiation to export its own
              ``CloudProperties``; the NetCDF ``comment`` states the
              limitation.

        Returns
        -------
        bool
            ``True`` if any accumulator was fed, ``False`` (no-op) when the
            collector has neither the spatial CMIP accumulators nor a zonal
            monthly accumulator with *lat_deg* supplied.
        """
        have_spatial = self._spatial_monthly is not None
        have_zonal = (
            self.monthly_means
            and self.monthly_accum is not None
            and lat_deg is not None
        )
        if not (have_spatial or have_zonal):
            return False

        from legoesm import constants as _c

        doy, _ = day_to_calendar(day)
        year = int(day // 365.0)
        # #1353 (codex-1 BLOCKER 1): interval-MEAN flux fields cover
        # [day - flux_interval_days, day], so calendar-bin them at the
        # interval MIDPOINT — endpoint binning would push each month's last
        # daily mean into the next month (Jan-31's [30,31] mean landing in
        # February) and shift daily ``pr`` one interval late.  The feed
        # cadence divides the day evenly in practice (diag_days = 1 or
        # 1/2^k), so intervals never straddle a month boundary and midpoint
        # binning is exact. State samples use the caller-supplied hour bin.
        if flux_interval_days:
            # No clamp at 0: negative-epoch runs are supported and
            # ``day_to_calendar``/``//`` handle negatives consistently
            # (Python floor semantics) — clamping shifted the first
            # interval of such a run into the wrong bucket (codex-3 LOW).
            _flux_day = float(day) - 0.5 * float(flux_interval_days)
            flux_doy, _ = day_to_calendar(_flux_day)
            flux_year = int(_flux_day // 365.0)
        else:
            flux_doy, flux_year = doy, year
        # Flux-field name sets (Amon spatial / daily / zonal) used to split
        # the PHASE-2 commits between the two calendar bins.
        _FLUX_2D = ("pr", "rlut", "rsut", "rsdt", "hfss", "hfls", "evspsbl",
                    "rsutcs", "rlutcs")
        _FLUX_DAILY = ("pr",)
        _FLUX_ZONAL = ("precip",)

        # Coerce to a numeric float array (``dtype=float64``): a non-numeric
        # (object/string) input raises HERE, in PHASE 1, instead of inside an
        # accumulator's ``np.mean``/``np.add.at`` mid-commit.  T is the REQUIRED
        # field and sets ``(nCells, nlev)`` for every shape check below — reject
        # a malformed rank / wrong level count FIRST (against the collector's
        # configured ``self.nlev``, so a mixed-nlev feed cannot commit a
        # different-shape ``T_low`` in add_2d before add_3d raises on the
        # pre-existing profile).
        T_np = np.asarray(T, dtype=np.float64)
        if T_np.ndim != 2 or T_np.shape[1] != self.nlev:
            raise ValueError(
                "feed_cmip_accumulators_native: T must be 2-D "
                f"(nCells, nlev={self.nlev}), got shape {T_np.shape}")
        p_s_np = np.asarray(p_s, dtype=np.float64)
        # Level -1 is the lowest (near-surface) model level (sigma_full is
        # ascending, ~1.0 at the surface — see _interp_to_plev19 / _tas_2m).
        T_low = T_np[..., -1]
        # tas is the 2 m MOST temperature when the caller supplies it (sst/sic
        # available), else the lowest-model-level fallback (applied AFTER the
        # shape check below) so the field is never dropped.
        _f64 = np.float64
        tas_field = None if tas is None else np.asarray(tas, dtype=_f64)
        ts_field = None if ts is None else np.asarray(ts, dtype=_f64)
        q_v_np = None if q_v is None else np.asarray(q_v, dtype=_f64)
        q_c_np = None if q_c is None else np.asarray(q_c, dtype=_f64)
        q_i_np = None if q_i is None else np.asarray(q_i, dtype=_f64)
        u_east_np = None if u_east is None else np.asarray(u_east, dtype=_f64)
        v_north_np = None if v_north is None else np.asarray(v_north, dtype=_f64)
        precip_np = None if precip is None else np.asarray(precip, dtype=_f64)
        phis_np = None if phis is None else np.asarray(phis, dtype=_f64)
        # TOA / surface-flux CMOR fields (already in CMOR sign conventions:
        # rlut/rsut/hfss/hfls positive up, rsdt positive down; see the
        # radiation/turbulence packers).
        rlut_np = None if rlut is None else np.asarray(rlut, dtype=_f64)
        rsut_np = None if rsut is None else np.asarray(rsut, dtype=_f64)
        rsdt_np = None if rsdt is None else np.asarray(rsdt, dtype=_f64)
        hfss_np = None if hfss is None else np.asarray(hfss, dtype=_f64)
        hfls_np = None if hfls is None else np.asarray(hfls, dtype=_f64)
        rsutcs_np = None if rsutcs is None else np.asarray(rsutcs, dtype=_f64)
        rlutcs_np = None if rlutcs is None else np.asarray(rlutcs, dtype=_f64)

        # Shape contract — validated UP FRONT so BOTH the spatial regrid AND the
        # zonal binning are transactional.  A malformed optional input raises
        # HERE (PHASE 1), before any accumulator is mutated: the ZONAL-only
        # config has no regrid phase to catch a wrong-length precip / cell wind
        # / q_v, so without this a bad sibling field would commit ``T_low`` then
        # raise inside ``np.add.at``.  ``ncol``/``nlev`` come from ``T`` (the
        # one required field).
        _ncol, _nlev = int(T_np.shape[0]), int(T_np.shape[1])
        _shape_checks: list[tuple[str, np.ndarray | None, tuple]] = [
            ("p_s", p_s_np, (_ncol,)),
            ("precip", precip_np, (_ncol,)),
            ("phis", phis_np, (_ncol,)),
            ("tas", tas_field, (_ncol,)),
            ("ts", ts_field, (_ncol,)),
            ("rlut", rlut_np, (_ncol,)),
            ("rsut", rsut_np, (_ncol,)),
            ("rsdt", rsdt_np, (_ncol,)),
            ("hfss", hfss_np, (_ncol,)),
            ("hfls", hfls_np, (_ncol,)),
            ("rsutcs", rsutcs_np, (_ncol,)),
            ("rlutcs", rlutcs_np, (_ncol,)),
            ("q_v", q_v_np, (_ncol, _nlev)),
            ("q_c", q_c_np, (_ncol, _nlev)),
            ("q_i", q_i_np, (_ncol, _nlev)),
            ("u_east", u_east_np, (_ncol, _nlev)),
            ("v_north", v_north_np, (_ncol, _nlev)),
        ]
        for _nm, _arr, _shp in _shape_checks:
            if _arr is not None and _arr.shape != _shp:
                raise ValueError(
                    f"feed_cmip_accumulators_native: {_nm} shape {_arr.shape} "
                    f"!= expected {_shp}")

        # tas is the 2 m MOST temperature when supplied, else the lowest level.
        if tas_field is None:
            tas_field = T_low

        # Sea-level pressure (hypsometric) — shared by the spatial ``psl`` and
        # the zonal ``psl`` band; compute once when phis is available (shape
        # validated above, so the broadcast is safe).
        psl = None
        if phis_np is not None:
            T_low_safe = np.maximum(T_low, _c.T_min_atmosphere)
            psl = p_s_np * np.exp(phis_np / (_c.R_d * T_low_safe))

        # ================================================================
        # PHASE 1 — compute everything (regrid / plev interp) up front.
        # No accumulator is mutated here, so a failure commits nothing.
        # ================================================================
        fields_2d: dict[str, np.ndarray] = {}
        fields_3d: dict[str, np.ndarray] = {}
        daily_2d: dict[str, np.ndarray] = {}
        if have_spatial:
            # evspsbl [kg/m2/s] = latent heat flux / L_v. PRE-EXISTING
            # documented approximation for this lane: no sublimation split
            # (L_s over ice-covered cells => ~13% undercount there; see
            # coupling_fields.surface_mass_flux for the phase-aware form),
            # matching the bulk-flux scheme's own L_v-only partition of
            # lhflx.  Wiring the phase-aware flux needs the ice fraction in
            # this feed — tracked as a follow-up, unchanged by #1353.
            evspsbl_np = None if hfls_np is None else hfls_np / _c.L_v
            for _name, _src in (
                ('tas', tas_field),
                ('ts', ts_field),
                ('ps', p_s_np),
                ('pr', precip_np),   # CMOR kg/m2/s — native, no conversion
                ('psl', psl),
                ('rlut', rlut_np),
                ('rsut', rsut_np),
                ('rsdt', rsdt_np),
                ('hfss', hfss_np),
                ('hfls', hfls_np),
                ('evspsbl', evspsbl_np),
                ('rsutcs', rsutcs_np),
                ('rlutcs', rlutcs_np),
            ):
                if _src is None or (not include_state and _name not in _FLUX_2D):
                    continue
                r = self._regrid_to_latlon_2d(_src)
                if r is not None:
                    fields_2d[_name] = r
            if include_state and q_v_np is not None:
                cwv_field = np.asarray(
                    column_water_vapor(q_v_np, p_s_np, self.dsigma,
                                       dp=self._dp(p_s_np)))
                r = self._regrid_to_latlon_2d(cwv_field)
                if r is not None:
                    fields_2d['prw'] = r
            # Total cloud cover: the SAME shared reduction the cube/lat-lon
            # ``collect`` path uses, so the MPAS lane cannot report a
            # different cloud cover for an identical state.  Skipped (not
            # zeroed) on a cloud-free run -- see ``_clt_percent``.
            if include_state:
                clt_field = self._clt_percent(
                    T_np, p_s_np, q_v_np, q_c_np, q_i_np,
                    cloud_fraction=cloud_fraction,
                    conv_mass_flux_up=conv_mass_flux_up,
                    conv_icwmr=conv_icwmr, lat_deg=lat_deg)
                if clt_field is not None:
                    r = self._regrid_to_latlon_2d(clt_field)
                    if r is not None:
                        fields_2d['clt'] = r

            # Cloud water paths (clwvi / clivi) from the model's OWN
            # diagnostic cloud fraction + RADIATIVE condensate — one
            # ``compute_cloud_properties`` call (the same shared function the
            # radiation cloud diagnosis and the cube-lane clt use; the
            # collector's ``_cloud_config`` is the stratiform
            # convective_cloud=False build, see ModelDriver._create_diagnostics).
            # ``clt`` is NOT produced here: it comes from ``_clt_percent``
            # above, the ONE reduction both lanes share.  The cloud FRACTION
            # this call returns is identical (the ``_replace`` below touches
            # only the lwp/iwp optical factors), so publishing it here too
            # would be a second copy of a shared numeric — the thing
            # ``_clt_percent`` exists to prevent.
            # clwvi/clivi are column sums of the per-layer grid-mean lwp/iwp:
            # the prognostic q_c/q_i raised by THIS config's sub-grid in-cloud
            # condensate floor, excluding precipitating snow/graupel.  It is
            # the model's DIAGNOSTIC cloud, not a reconstruction of
            # radiation's cloud state (this config is stratiform-only and
            # carries no CLUBB cf override; every other cloud knob IS
            # threaded) — see the docstring and the NetCDF comment.
            # MASS, not optical path: the sub-grid
            # inhomogeneity (Cahalan/two_region chi) and partial-coverage
            # (two_column) factors ``compute_cloud_properties`` applies to
            # lwp/iwp are RADIATIVE-TRANSFER corrections, so they are pinned
            # OFF here — publishing a chi-thinned path under CMIP6
            # ``atmosphere_mass_content_of_cloud_condensed_water`` would be a
            # silently wrong number in exactly the obs comparison this feed
            # exists for.  (Today's ``_create_diagnostics`` config leaves all
            # three at their no-op defaults; the _replace makes that an
            # invariant instead of an accident.) Requires q_v (the
            # RH-based fraction) and a cloud config; skipped otherwise.
            if (include_state and (q_c_np is not None or q_i_np is not None)
                    and q_v_np is not None
                    and self._cloud_config is not None):
                from legoesm.atmosphere.physics.clouds.cloud_fraction import (
                    compute_cloud_properties,
                )
                _p_full_cl = jnp.asarray(self._p_full(p_s_np))
                _dp_cl = jnp.asarray(self._dp(p_s_np))
                _mass_cfg = self._cloud_config._replace(
                    cloud_optics_inhomogeneity="constant",
                    cloud_inhomogeneity_factor=1.0,
                    cloud_partial_coverage_optics="none",
                )
                cloud_props = compute_cloud_properties(
                    jnp.asarray(T_np),
                    _p_full_cl,
                    jnp.asarray(q_v_np),
                    _dp_cl,
                    _mass_cfg,
                    # EITHER tracer alone is enough (an ice-only state must
                    # still get clivi); both None keeps the fields absent.
                    q_cloud=None if q_c_np is None else jnp.asarray(q_c_np),
                    q_ice=None if q_i_np is None else jnp.asarray(q_i_np),
                    **self._cam6_cloud_kwargs(
                        cloud_fraction, conv_mass_flux_up, conv_icwmr,
                        lat_deg, p_s_np, *np.shape(T_np)[-2:]),
                )
                self._cloud_paths_radiative = True
                # Per-layer grid-mean paths [kg/m2] -> column path.
                iwp_col = np.asarray(jnp.sum(cloud_props.iwp, axis=-1))
                cwp_col = np.asarray(
                    jnp.sum(cloud_props.lwp, axis=-1)) + iwp_col
                for _name, _src in (('clwvi', cwp_col),
                                    ('clivi', iwp_col)):
                    r = self._regrid_to_latlon_2d(_src)
                    if r is not None:
                        fields_2d[_name] = r

            # 3-D fields: model levels → plev19, then regrid.  The plev
            # interpolation is column-wise and works unchanged on native
            # (nCells, nlev) with p_s (nCells,).
            for _name, _src in (
                ('ta', T_np),
                ('hus', q_v_np),
                ('ua', u_east_np),
                ('va', v_north_np),
                # Pressure vertical velocity: same plev19 + regrid path as the
                # rest, so subsidence becomes a published field instead of a
                # continuity guess made downstream from monthly-mean winds.
                ('wap', None if wap is None else np.asarray(wap)),
            ):
                if _src is None or not include_state:
                    continue
                _plev = self._interp_to_plev19(_src, p_s_np)
                if _plev is None:
                    continue
                r = self._regrid_to_latlon_3d(_plev)
                if r is not None:
                    fields_3d[_name] = r

            # Daily: reuse the regridded 2-D fields + 850 hPa winds sliced from
            # the 3-D arrays (same index convention as collect()).
            if self._spatial_daily is not None:
                for _name in ("tas", "pr", "psl"):
                    if _name in fields_2d:
                        daily_2d[_name] = fields_2d[_name]
                _plev_sorted = np.sort(CMIP6_PLEV19)
                _idx850 = int(np.argmin(np.abs(_plev_sorted - 85000.0)))
                for _src_name, _dst in (("ua", "ua850"), ("va", "va850")):
                    if (_src_name in fields_3d
                            and fields_3d[_src_name].shape[2] > _idx850):
                        daily_2d[_dst] = fields_3d[_src_name][:, :, _idx850]

        z2d: dict[str, np.ndarray] = {}
        z3d: dict[str, np.ndarray] = {}
        lat_np = None
        if have_zonal:
            # Validate the zonal lat vector HERE, in PHASE 1, before any commit
            # (the sibling zonal fields were shape-checked up front).  Force a
            # numeric float array so a shape-correct-but-non-numeric (object/
            # string) lat_deg raises here, not inside ``np.digitize`` mid-
            # commit; also require finite values.
            lat_np = np.asarray(lat_deg, dtype=np.float64)
            if lat_np.shape != (_ncol,):
                raise ValueError(
                    "feed_cmip_accumulators_native: lat_deg shape "
                    f"{lat_np.shape} != expected ({_ncol},)")
            if not np.all(np.isfinite(lat_np)):
                raise ValueError(
                    "feed_cmip_accumulators_native: lat_deg has non-finite "
                    "values")
            if include_state:
                z2d['T_low'] = np.asarray(T_low)   # lowest model level, NOT 2 m tas
            if precip_np is not None:
                z2d['precip'] = precip_np * 86400.0     # [mm/day], like collect()
            if include_state and psl is not None:
                z2d['psl'] = np.asarray(psl)
            if include_state:
                z3d['T'] = T_np
                if u_east_np is not None:
                    z3d['u'] = u_east_np
                if q_v_np is not None:
                    z3d['q_v'] = q_v_np * 1000.0             # [g/kg]

        # ================================================================
        # PHASE 2 — commit (cheap, shape-checked add_* only).  Flux fields
        # commit under the interval-midpoint calendar bin, state samples
        # under the caller-supplied hour bin (identical when flux_interval_days unset —
        # the split dicts are then committed in one call each; the
        # accumulators keep per-FIELD (sum, count) pairs, so a split commit
        # changes no mean).
        # ================================================================
        def _split(d: dict, flux_names) -> tuple[dict, dict]:
            flux = {k: v for k, v in d.items() if k in flux_names}
            state = {k: v for k, v in d.items() if k not in flux_names}
            return state, flux

        # Same CALENDAR BUCKET (year, month) — not same doy: away from month
        # boundaries the midpoint and endpoint fall in the same month and
        # the commit stays a single legacy add_* call per accumulator.
        _same_bin = (
            flux_year == year
            and (MonthlyAccumulator.day_to_month(flux_doy)
                 == MonthlyAccumulator.day_to_month(doy)))
        fed = False
        if have_spatial:
            _state_2d, _flux_2d = _split(fields_2d, _FLUX_2D)
            if _same_bin:
                _state_2d, _flux_2d = fields_2d, {}
            if _state_2d:
                self._spatial_monthly.add_2d(doy, year, _state_2d)
                fed = True
            if _flux_2d:
                self._spatial_monthly.add_2d(flux_doy, flux_year, _flux_2d)
                fed = True
            if fields_3d:
                self._spatial_monthly.add_3d(doy, year, fields_3d)
                fed = True
            if daily_2d and self._spatial_daily is not None:
                # DAILY buckets are per (year, int(doy)) — finer than the
                # monthly month bucket, so the collapse test is DAY
                # equality, not month equality (codex-2 finding 7: a
                # month-level collapse binned interior daily pr at the
                # interval ENDPOINT day, re-introducing the one-day lag
                # for every day except month boundaries).
                _same_day = (flux_year == year
                             and int(flux_doy) == int(doy))
                _state_d, _flux_d = _split(daily_2d, _FLUX_DAILY)
                if _same_day:
                    _state_d, _flux_d = daily_2d, {}
                if _state_d:
                    self._spatial_daily.add_2d(doy, year, _state_d)
                    fed = True
                if _flux_d:
                    self._spatial_daily.add_2d(flux_doy, flux_year, _flux_d)
                    fed = True
        if have_zonal:
            _state_z, _flux_z = _split(z2d, _FLUX_ZONAL)
            if _same_bin:
                _state_z, _flux_z = z2d, {}
            if _state_z:
                self.monthly_accum.add_2d(doy, year, _state_z, lat_np)
            if _flux_z:
                self.monthly_accum.add_2d(flux_doy, flux_year, _flux_z, lat_np)
            if z3d:
                self.monthly_accum.add_3d(doy, year, z3d, lat_np)
            fed = True

        return fed

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
            rsdt=np.array(self.rsdt),
            hfss=np.array(self.hfss),
            hfls=np.array(self.hfls),
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
        self.rsdt.clear()
        self.hfss.clear()
        self.hfls.clear()
        self.profiles_T.clear()
        self.profiles_qv.clear()

        # Also flush moisture tracker
        moisture_data = self.moisture_tracker.flush_to_lists()
        if moisture_data["times"]:
            np.savez(
                incr_dir / f"moisture_chunk_{chunk_idx:04d}.npz",
                **{k: np.array(v) for k, v in moisture_data.items()},
            )

    def flush_cmip_monthly(self, current_day: float, *,
                           write: bool = True) -> None:
        """Write completed CMIP months incrementally and free their memory.

        Call this periodically (e.g. at each diagnostic interval) during
        long runs.  Only months strictly before the current month are
        flushed; the in-progress month is kept for further accumulation.

        ``write=False`` pops (frees) the completed months WITHOUT writing —
        for multi-controller SPMD non-root processes, whose accumulators
        fill identically to root's (every process runs the gathered full
        ``collect()``) but must never touch the shared output files; without
        the pop they would retain every completed month for the whole run
        (codex round-10 Medium).
        """
        if self._spatial_monthly is None or self.cf_writer is None:
            return

        doy, _ = day_to_calendar(current_day)
        current_year = int(current_day // 365.0)
        current_month = MonthlyAccumulator.day_to_month(doy)

        data = self._spatial_monthly.pop_completed_months(
            current_year, current_month,
        )
        months = data.get('months', [])
        if not months or not write:
            return

        self._write_cmip_data(data)

    def finalize_cmip_daily(self, current_day: float) -> None:
        """Write the COMPLETED days of the CMIP6 ``day`` table if a CMIP
        writer is active.

        Companion to :meth:`finalize_cmip_fixed` for the graceful wallclock
        exit.  The daily accumulator is otherwise drained only by :meth:`save`
        at the end of a run, so a restart-chain ``sys.exit(0)`` would drop this
        SLURM segment's daily means (day/tas, tasmin, tasmax, …).  Mirrors
        :meth:`flush_cmip_monthly`: only days STRICTLY BEFORE ``current_day``
        are emitted (and freed).  The in-progress day is deliberately withheld
        — writing it here and again from the restart segment (which resumes
        inside the same ``(year, doy)``) would create duplicate ``time``
        coordinates, since ``CFWriter.write_field`` appends blindly.  The one
        boundary day straddling the exit is therefore a bounded imperfection
        (its restart-segment mean omits the pre-exit samples), matching the
        monthly-boundary limitation; a fully lossless chain would require
        checkpointing the accumulator state.  Guarded on ``cf_writer`` and the
        daily accumulator (no-op for non-CMOR / daily-off runs)."""
        if self._spatial_daily is None or self.cf_writer is None:
            return
        doy, _ = day_to_calendar(current_day)
        current_year = int(current_day // 365.0)
        data = self._spatial_daily.pop_completed_days(current_year, int(doy))
        if not data.get('days'):
            return
        lat, lon = self._cmip_target_latlon()
        # Preserve hourly-extreme metadata on restart-chain flushes too.
        self.cf_writer.write_daily(
            data, lat=lat, lon=lon,
            extra_attrs_by_var=self._daily_extreme_attrs())

    def finalize_cmip_fixed(self) -> None:
        """Write the CMOR ``fx`` table (areacella / sftlf / orog) if a CMIP
        writer is active.

        The time-invariant ``fx`` fields are normally written once by
        :meth:`save` at the end of a run.  A wallclock-graceful exit
        (:meth:`ModelDriver._maybe_wallclock_exit`) calls ``sys.exit(0)`` and
        never reaches :meth:`save`, so without this public entry a
        restart-chained run — i.e. EVERY multi-hour AMIP run, whose year does
        not finish in a single SLURM window — writes its incremental monthly
        ``Amon`` files but never ``areacella``/``sftlf``.  The resulting CMOR
        output is non-compliant (each variable's ``external_variables``
        attribute references ``areacella``/``sftlf``) and blocks any
        area-weighted or land/ocean-split diagnostic.  Idempotent: rewriting
        the same static fields on a later flush is harmless."""
        if self.cf_writer is not None:
            self._write_cmip_fixed_files()

    def save_cmor_accumulators(self, path: str | Path) -> None:
        """Persist the CMOR accumulator state to an additive sidecar ``.npz``.

        Serializes the spatial-monthly, spatial-daily, and zonal-monthly
        accumulators — each namespaced with a ``"monthly."`` / ``"daily."`` /
        ``"zonal."`` key prefix — into a single ``np.savez`` file written next
        to the model checkpoint.  This lets a calendar month split across
        restart-chain links (each link is only ~10 days; a month is ~30) be
        completed on resume; without it the monthly accumulator is recreated
        empty every link and the CMOR ``Amon`` means are never written.

        Additive by design: it does NOT touch the checkpoint ``.npz`` schema.
        No-op when CMIP output is inactive (no ``cf_writer`` / accumulators).

        The write is ATOMIC (``<name>.tmp`` then ``Path.replace``): a partial
        write can never leave a corrupt sidecar, and a re-persist that fails
        never clobbers the previous (e.g. drained) sidecar with a half-file.
        """
        if self.cf_writer is None:
            return
        merged: dict[str, np.ndarray] = {}
        for prefix, accum in (
            ("monthly.", self._spatial_monthly),
            ("daily.", self._spatial_daily),
            ("zonal.", self.monthly_accum),
        ):
            if accum is None:
                continue
            for key, arr in accum.get_state().items():
                merged[prefix + key] = arr
        if not merged:
            return
        if getattr(self.cf_writer, "require_centre_sampling_on_append", False):
            from legoesm.io.cmor_output import LAT_SAMPLING_CENTRES
            merged[_SIDECAR_SAMPLING_KEY] = np.array(LAT_SAMPLING_CENTRES)
        path = Path(path)
        tmp_path = path.parent / (path.name + ".tmp")
        try:
            # Write via a file handle so ``np.savez`` does not append a second
            # ``.npz`` to the ``.tmp`` name, then atomically move into place.
            with open(tmp_path, "wb") as fh:
                np.savez(fh, **merged)
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)  # never leave a partial temp
            raise

    def load_cmor_accumulators(self, path: str | Path) -> bool:
        """Restore CMOR accumulator state from a sidecar written by
        :meth:`save_cmor_accumulators`.

        Splits the merged ``.npz`` back into the ``"monthly."`` / ``"daily."`` /
        ``"zonal."`` namespaces and calls ``set_state`` on each active
        accumulator.  Returns ``True`` if any state was restored, ``False`` if
        the sidecar is absent or CMIP output is inactive — an older run with no
        sidecar simply resumes with empty accumulators (backward-compatible).
        """
        path = Path(path)
        if self.cf_writer is None or not path.exists():
            return False
        namespaced = {
            "monthly.": self._spatial_monthly,
            "daily.": self._spatial_daily,
            "zonal.": self.monthly_accum,
        }
        substates: dict[str, dict] = {prefix: {} for prefix in namespaced}
        with np.load(str(path), allow_pickle=False) as npz:
            stamp = (str(npz[_SIDECAR_SAMPLING_KEY])
                     if _SIDECAR_SAMPLING_KEY in npz.files else None)
            from legoesm.io.cmor_output import (
                LAT_SAMPLING_CENTRES, LatSamplingMismatchError)
            if (stamp != LAT_SAMPLING_CENTRES and getattr(
                    self.cf_writer, "require_centre_sampling_on_append", False)):
                raise LatSamplingMismatchError(
                    f"{path}: CMOR accumulators written before lat-lon output "
                    "was sampled at its labelled cell centres (rows were "
                    "pole-to-pole). Resuming would mix both grids in the open "
                    "month. Resume with the code that wrote it, or delete the "
                    "sidecar to drop the partial month.")
            for full_key in npz.files:
                for prefix in namespaced:
                    if full_key.startswith(prefix):
                        substates[prefix][full_key[len(prefix):]] = npz[full_key]
                        break
        restored = False
        for prefix, accum in namespaced.items():
            sub = substates[prefix]
            if accum is not None and sub:
                accum.set_state(sub)
                restored = True
        return restored

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
            rsdt=np.array(self.rsdt),
            hfss=np.array(self.hfss),
            hfls=np.array(self.hfls),
            sigma=sigma,
            profiles_T=np.array(self.profiles_T) if self.profiles_T else np.array([]),
            profiles_qv=np.array(self.profiles_qv) if self.profiles_qv else np.array([]),
            energy_toa_net=np.array(self.energy_tracker.toa_net),
            energy_column=np.array(self.energy_tracker.column_energy),
            energy_dE_dt=np.array(self.energy_tracker.dE_dt),
            energy_residual=np.array(self.energy_tracker.residual),
            moisture_column_water=np.array(self.moisture_tracker.column_water),
            moisture_precip_rate=np.array(self.moisture_tracker.precip_rate),
            moisture_evap_rate=np.array(self.moisture_tracker.evap_rate),
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
            self._write_cmip_daily_files()
            self._write_cmip_fixed_files()
            self.cf_writer.close()

    def _write_cmip_monthly_files(self) -> None:
        """Write CMIP-compliant NetCDF files from the spatial accumulator.

        Called at end-of-run to flush any remaining data.
        """
        if self._spatial_monthly is None or self.cf_writer is None:
            return
        data = self._spatial_monthly.finalize()
        self._write_cmip_data(data)

    def _cmip_target_latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lat, lon) 1-D arrays for the CMIP target grid."""
        dlat = 180.0 / self._cmip_nlat
        dlon = 360.0 / self._cmip_nlon
        lat = np.linspace(
            -90.0 + dlat / 2, 90.0 - dlat / 2, self._cmip_nlat,
        )
        lon = np.linspace(
            dlon / 2, 360.0 - dlon / 2, self._cmip_nlon,
        )
        return lat, lon

    def _daily_extreme_attrs(self) -> dict | None:
        """Describe daily extrema sampled by the independent hourly hook."""
        hourly = {}
        if (self._spatial_daily is not None
                and self._spatial_daily._extreme_max_count_ever > 0):
            hourly = {
                name: {"cell_methods": f"time: {method}",
                       "comment": "Daily extremum of hourly instantaneous samples "
                                  "(first model step at or after each hour)."}
                for name, method in (("tasmin", "minimum"), ("tasmax", "maximum"))
            }
        return hourly or None

    def _write_cmip_daily_files(self) -> None:
        """Flush the daily accumulator to CMIP6 ``day`` NetCDF files."""
        if self._spatial_daily is None or self.cf_writer is None:
            return
        data = self._spatial_daily.finalize()
        if not data.get("days"):
            return
        lat, lon = self._cmip_target_latlon()
        self.cf_writer.write_daily(
            data, lat=lat, lon=lon,
            extra_attrs_by_var=self._daily_extreme_attrs())

    def _write_cmip_fixed_files(self) -> None:
        """Write the CMIP6 ``fx`` file (orog / sftlf / areacella).

        ``areacella`` is always computed from the CMIP target lat-lon
        grid (cosine-latitude weights on Earth's radius).  ``orog`` and
        ``sftlf`` are written only when ``set_fixed_fields`` supplied
        the source data.
        """
        if self.cf_writer is None:
            return
        if getattr(self, "_cmip_nlat", None) is None:
            return

        from legoesm import constants as _c

        lat, lon = self._cmip_target_latlon()

        # areacella: cell area on the target grid [m^2].  Using exact
        # sin-latitude differences (not small-angle approx) keeps total
        # surface area equal to 4 π R^2 to machine precision.
        lat_edges_deg = np.linspace(-90.0, 90.0, self._cmip_nlat + 1)
        sin_edges = np.sin(np.radians(lat_edges_deg))
        band_area_frac = sin_edges[1:] - sin_edges[:-1]  # (nlat,)
        dlon_rad = 2.0 * np.pi / self._cmip_nlon
        R = float(_c.R_earth)
        area_lat = (R ** 2) * band_area_frac * dlon_rad  # (nlat,)
        areacella = np.broadcast_to(
            area_lat[:, None], (self._cmip_nlat, self._cmip_nlon),
        ).astype(np.float64)
        try:
            self.cf_writer.write_fixed(
                var_name="areacella",
                data=areacella, lat=lat, lon=lon,
            )
        except (KeyError, ValueError):
            pass

        # orog: surface altitude = phis / g
        if self._fixed_phis is not None:
            orog_native = np.asarray(self._fixed_phis) / float(_c.g)
            orog = self._regrid_to_latlon_2d(orog_native)
            if orog is not None:
                try:
                    self.cf_writer.write_fixed(
                        var_name="orog",
                        data=orog, lat=lat, lon=lon,
                    )
                except (KeyError, ValueError):
                    pass

        # sftlf: land area fraction in %.  Clipped to [0, 100] to guard
        # against small negative overshoots from bilinear regridding.
        if self._fixed_land_fraction is not None:
            lf_native = np.asarray(self._fixed_land_fraction)
            sftlf = self._regrid_to_latlon_2d(lf_native)
            if sftlf is not None:
                sftlf = np.clip(sftlf * 100.0, 0.0, 100.0)
                try:
                    self.cf_writer.write_fixed(
                        var_name="sftlf",
                        data=sftlf, lat=lat, lon=lon,
                    )
                except (KeyError, ValueError):
                    pass

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

        # ``yr`` here is the 0-based run-relative calendar year (not the
        # absolute calendar year) because the CFWriter's ``ref_date`` is
        # ``<start_year>-01-01``, so time values must start from zero at
        # the first simulation month.
        for i, (yr, mo) in enumerate(months):
            year_offset = yr * 365.0
            day_start = year_offset + sum(month_days[:mo - 1])
            day_end = day_start + month_days[mo - 1]
            time_mid = 0.5 * (day_start + day_end)
            time_bounds = (day_start, day_end)

            # Semantics notes for the MPAS-lane cloud trio.  The collector
            # recomputes the cloud from its OWN ``_cloud_config``, which
            # ModelDriver._create_diagnostics builds from the same
            # ExperimentConfig cloud fields as radiation EXCEPT: it forces
            # convective_cloud=False (#689) and plumbs no CLUBB
            # cloud-fraction override.  (rh_crit, q_c_diagnostic, p_xr,
            # alpha_xr, diagnostic_condensate_scheme and adiabatic_lwc_rate
            # ARE threaded, so the floor formula matches radiation's.)  Those
            # two divergences are what the comment discloses; closing them
            # needs radiation to export its own CloudProperties, a separate
            # change. Keyed on ``_cloud_paths_radiative``, which the native
            # feed sets when it derives the paths from cloud diagnosis.
            # The cube-lane collect() clwvi/clivi integrate the PROGNOSTIC
            # condensate q_c+q_i+q_s+q_g, never set the flag, and keep the
            # bare table attrs.
            _cloud_notes = {
                "clwvi": ("DIAGNOSTIC condensed-water path: column integral "
                          "of the grid-mean liquid+ice condensate from the "
                          "model's own cloud scheme, INCLUDING that scheme's "
                          "sub-grid in-cloud condensate floor where it has "
                          "one (the diagnostic-fraction schemes sundqvist / "
                          "xu_randall; cloud_scheme='resolved' has no floor "
                          "and reports the explicit condensate).  So it is "
                          "NOT the prognostic condensed water mass -- the "
                          "floor is not carried by the water budget.  It is "
                          "also NOT necessarily what the radiation solved "
                          "with: this diagnostic is stratiform-only and gets "
                          "no CLUBB cloud-fraction override, so a run using "
                          "the convective cloud add-on or a CLUBB cloud "
                          "fraction in radiation will see those in "
                          "rsut/rsutcs but not here.  Precipitating "
                          "snow/graupel are excluded.  A condensate loading "
                          "in kg m-2, not an optical path -- the cloud-optics "
                          "sub-grid inhomogeneity and partial-coverage "
                          "thinning factors are excluded."),
                "clivi": ("DIAGNOSTIC ice water path: the cloud-ice share of "
                          "the clwvi column integral above, with the same "
                          "inclusions and exclusions."),
                "clt": ("Stratiform layer cloud fraction (model's own "
                        "diagnostic scheme, convective add-on excluded) "
                        "reduced by maximum-random vertical overlap."),
            }
            _notes_on = getattr(self, "_cloud_paths_radiative", False)

            def _attrs_for(var_name):
                if _notes_on and var_name in _cloud_notes:
                    return {"comment": _cloud_notes[var_name]}
                return None

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
                        extra_attrs=_attrs_for(var_name),
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
                # _make_plev_da sorts plev descending (highest pressure first).
                # Flip the pressure axis so data[0] aligns with plev[0]=max pressure.
                field_plev = np.transpose(field_slice, (2, 0, 1))[::-1]
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
        # Fuse all device→host syncs into one ``jnp.stack`` so the
        # blowup probe (called every diagnostic interval inside the
        # integration loop) costs one GPU stall per call instead of
        # 6-7.  The boolean ``isfinite`` checks on u and T are folded
        # into the same stack as 0/1 floats.
        # Wind finiteness covers BOTH components: a NaN in v makes
        # wind_term = max(sqrt(u^2+v^2)) NaN, and NaN > 500 is False, so a
        # v-only blow-up would otherwise pass the max-wind check silently.
        if hasattr(state, 'v'):
            wind_term = jnp.max(jnp.sqrt(state.u.data ** 2 + state.v.data ** 2))
            wind_finite = jnp.logical_and(
                jnp.all(jnp.isfinite(state.u.data)),
                jnp.all(jnp.isfinite(state.v.data)))
        else:
            wind_term = jnp.max(jnp.abs(state.u.data))
            wind_finite = jnp.all(jnp.isfinite(state.u.data))
        has_p_s = hasattr(state, 'p_s')
        terms = [
            wind_finite.astype(state.T.data.dtype),
            jnp.all(jnp.isfinite(state.T.data)).astype(state.T.data.dtype),
            wind_term.astype(state.T.data.dtype),
            jnp.min(state.T.data).astype(state.T.data.dtype),
            jnp.max(state.T.data).astype(state.T.data.dtype),
        ]
        if has_p_s:
            terms.append(jnp.min(state.p_s.data).astype(state.T.data.dtype))
            terms.append(jnp.max(state.p_s.data).astype(state.T.data.dtype))
            # Finiteness of p_s explicitly: a NaN p_s makes BOTH bound
            # comparisons below False and would otherwise pass the probe (the
            # min/max are NaN, and NaN < lo / NaN > hi are both False), letting
            # a garbage state be checkpointed at a wallclock-graceful exit.
            terms.append(
                jnp.all(jnp.isfinite(state.p_s.data)).astype(state.T.data.dtype))
        host = np.asarray(jnp.stack(terms))
        wind_finite = bool(host[0] > 0.5)
        T_finite = bool(host[1] > 0.5)
        max_v = float(host[2])
        T_min_val = float(host[3])
        T_max_val = float(host[4])

        if not wind_finite:
            return f"BLOWUP at day {elapsed_day:.0f}: non-finite winds"
        if max_v > 500:
            return f"BLOWUP at day {elapsed_day:.0f}: max wind {max_v:.1f} m/s"
        if not T_finite:
            return f"BLOWUP at day {elapsed_day:.0f}: non-finite T"
        if has_p_s and not bool(host[7] > 0.5):
            return f"BLOWUP at day {elapsed_day:.0f}: non-finite surface pressure"

        # Physical-plausibility bounds (shared helper — one source of truth).
        ps_min = float(host[5]) if has_p_s else None
        ps_max = float(host[6]) if has_p_s else None
        return physical_state_blowup_reason(
            elapsed_day, T_min_val, T_max_val, ps_min=ps_min, ps_max=ps_max)


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
