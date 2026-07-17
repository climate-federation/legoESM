"""AMIP forcing: prescribed SST and sea-ice from PCMDI reference datasets.

Loads monthly-mean SST and sea-ice concentration from NetCDF files
(COBE-SST2, HadISST, or custom), regrids to the cubed-sphere grid,
and provides linear time interpolation for use during integration.

Data sources:
- COBE-SST2: ftp://ftp.cgd.ucar.edu/archive/SSTICE/MODEL.SST.COBE-SST2.*.nc
  Variables: SST_cpl [K], ice_cov [%]
- HadISST: Met Office Hadley Centre (1x1 deg monthly)
  Variables: sst [C], sic [fraction]

References
----------
- Taylor, K. E., Williamson, D., & Zwiers, F. (2000). The sea surface
  temperature and sea-ice concentration boundary conditions for AMIP II
  simulations. PCMDI Report No. 60.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm import constants


class AMIPForcingConfig(NamedTuple):
    """Configuration for AMIP boundary conditions.

    Fields
    ------
    dataset : str
        Dataset preset: "cobe", "hadisst", or "custom".
    path : str
        Path to NetCDF file.
    sst_var : str
        SST variable name in NetCDF.
    sic_var : str
        Sea-ice concentration variable name in NetCDF.
    time_var : str
        Time variable name.
    lat_var : str
        Latitude variable name.
    lon_var : str
        Longitude variable name.
    sst_offset : float
        Additive offset for SST (e.g., +273.15 if data in Celsius) [K].
    sic_scale : float
        Multiplicative scale for SIC (e.g., 0.01 if data in percent).
    sic_path : str
        Separate SIC file path.  When non-empty, SIC is loaded from
        this file instead of the main ``path``.
    T_ice : float
        Sea-ice surface temperature [K].
    albedo_ice : float
        Sea-ice albedo.
    albedo_ocean : float
        Open ocean albedo.
    """
    dataset: str = "cobe"
    path: str = ""
    sst_var: str = "SST_cpl"
    sic_var: str = "ice_cov"
    time_var: str = "time"
    lat_var: str = "lat"
    lon_var: str = "lon"
    sst_offset: float = 0.0
    sic_scale: float = 1.0
    sic_path: str = ""
    # NOTE: ``T_ice`` is the seawater freezing point used as the SST
    # floor / SIC ramp threshold — NOT the sea-ice surface temperature.
    # The legacy name is preserved for AMIP config compatibility, but
    # the value tracks ``constants.T_freeze_ocean`` (CLAUDE.md naming
    # discipline gap; renaming is tracked tech debt).
    T_ice: float = constants.T_freeze_ocean
    albedo_ice: float = 0.65
    albedo_ocean: float = 0.06


class AMIPForcing(NamedTuple):
    """Loaded and regridded AMIP forcing data.

    Fields
    ------
    times : jax.Array
        Time coordinate, shape (ntime,). Absolute days since
        ``start_year-01-01`` when loaded with ``start_year`` (AMIP-II date
        anchoring), else days since the first record.
    sst : jax.Array
        Sea surface temperature [K], shape (ntime, ...) where ``...``
        is ``(6, n, n)`` for cubed-sphere or ``(n_lat, n_lon)`` for Gaussian.
    sic : jax.Array
        Sea-ice concentration [0-1], same shape as sst.
    config : AMIPForcingConfig
        Configuration used to load the data.
    """
    times: jnp.ndarray
    sst: jnp.ndarray
    sic: jnp.ndarray
    config: AMIPForcingConfig


def get_amip_preset(dataset_name: str) -> AMIPForcingConfig:
    """Return an AMIPForcingConfig with preset variable names and offsets.

    Parameters
    ----------
    dataset_name : str
        One of "cobe", "hadisst".

    Returns
    -------
    AMIPForcingConfig
        Config with correct variable names and unit conversions.
    """
    if dataset_name == "cobe":
        return AMIPForcingConfig(
            dataset="cobe",
            sst_var="SST_cpl",
            sic_var="ice_cov",
            sst_offset=0.0,       # already in K
            sic_scale=0.01,       # percent -> fraction
        )
    elif dataset_name == "hadisst":
        return AMIPForcingConfig(
            dataset="hadisst",
            sst_var="sst",
            sic_var="sic",
            sst_offset=constants.T_freeze,    # Celsius -> Kelvin
            sic_scale=1.0,        # already fraction
        )
    else:
        raise ValueError(
            f"Unknown dataset preset: {dataset_name!r}. "
            f"Use 'cobe', 'hadisst', or create a custom AMIPForcingConfig."
        )


def _is_icon_unstructured(ds) -> bool:
    """Detect whether a dataset uses the ICON unstructured grid format.

    ICON boundary condition files have a ``cell`` dimension with
    ``clon``/``clat`` coordinates in radians.
    """
    return "cell" in ds.dims and "clon" in ds.coords and "clat" in ds.coords


def _time_coord_to_days(time_coord, epoch_year: int | None = None) -> np.ndarray:
    """Convert numeric, NumPy datetime, or cftime coordinates to days.

    When ``epoch_year`` is given and the axis is calendar-aware (datetime64 or
    cftime), returns **absolute** days since ``epoch_year-01-01`` so a model day
    (also days since ``start_year-01-01``) indexes the file by real calendar
    date — the AMIP-II requirement that a run use the SST/SIC of its *simulated*
    dates, and preserving the mid-month bcs anchor (Jan value at day ~15.5, not
    day 0).  With ``epoch_year=None`` it falls back to days since the first
    record (legacy relative indexing).  A bare-numeric axis carries no reference
    date, so it is returned as-is when an epoch is requested (the synthetic AMIP
    deck already writes it as days since ``start_year-01-01``).
    """
    time_arr = np.asarray(time_coord)

    if np.issubdtype(time_arr.dtype, np.datetime64):
        if epoch_year is not None:
            epoch = np.datetime64(f"{epoch_year:04d}-01-01")
            return ((time_arr - epoch) / np.timedelta64(1, "D")).astype(np.float64)
        t0 = time_arr[0]
        return ((time_arr - t0) / np.timedelta64(1, "D")).astype(np.float64)

    first = time_arr[0]
    if hasattr(first, "calendar"):
        import cftime

        if epoch_year is not None:
            units = f"days since {epoch_year:04d}-01-01 00:00:00"
        else:
            units = (
                "days since "
                f"{first.year:04d}-{first.month:02d}-{first.day:02d} "
                f"{first.hour:02d}:{first.minute:02d}:{first.second:02d}"
            )
        return np.asarray(
            cftime.date2num(list(time_arr), units=units, calendar=first.calendar),
            dtype=np.float64,
        )

    try:
        values = time_arr.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Unsupported AMIP time coordinate type. Expected numeric, "
            "numpy.datetime64, or cftime datetimes."
        ) from exc

    if epoch_year is not None:
        # A bare-numeric axis carries no reference date, so it cannot be
        # calendar-anchored: its records could reference any year, and trusting
        # them as days-since-start_year would silently serve the wrong era (the
        # exact bug this fix removes). Real input4MIPs / the synthetic deck ship
        # CF ``units`` and decode to datetime64/cftime, so this only rejects a
        # metadata-less transient file — fail loud instead of guessing.
        raise ValueError(
            "Cannot calendar-anchor a bare-numeric AMIP time axis (no CF "
            "reference date). Add `units`/`calendar` metadata (e.g. 'days "
            "since 1979-01-01') so the axis decodes to a datetime/cftime type."
        )
    return values - values[0]


def _load_icon_unstructured(
    config: AMIPForcingConfig, grid, start_year: int | None = None
) -> AMIPForcing:
    """Load AMIP forcing from ICON unstructured NetCDF files.

    Handles separate SST/SIC files (``config.sic_path``), unit
    conversions, and KD-tree nearest-neighbour regridding from the
    ICON cell centroids to the target grid.  ``start_year`` anchors a
    multi-year (transient) file to the run calendar exactly like the
    lat-lon path (AMIP-II date anchoring + coverage guard); a short
    climatology stays first-record-relative so its seasonal phase is
    independent of ``start_year``.
    """
    import xarray as xr
    from scipy.spatial import cKDTree

    # --- Open SST file ---
    ds_sst = xr.open_dataset(config.path)
    try:
        clon = ds_sst["clon"].values.astype(np.float64)   # radians
        clat = ds_sst["clat"].values.astype(np.float64)   # radians
        sst_data = ds_sst[config.sst_var].values.astype(np.float64)
        time_coord = ds_sst[config.time_var].values
    finally:
        ds_sst.close()

    # --- Open SIC file (separate or same) ---
    sic_path = config.sic_path or config.path
    ds_sic = xr.open_dataset(sic_path)
    try:
        sic_data = ds_sic[config.sic_var].values.astype(np.float64)
    finally:
        ds_sic.close()

    # Ensure 2D: (ntime, ncells)
    if sst_data.ndim == 1:
        sst_data = sst_data[None, :]
        sic_data = sic_data[None, :]

    # --- Unit conversions ---
    sst_data = sst_data + config.sst_offset
    sic_data = sic_data * config.sic_scale
    sic_data = np.clip(sic_data, 0.0, 1.0)
    # Clamp SST to the seawater freezing point (271.35 K).  Was 200 K
    # — 71 K below physical freezing — which silently allowed
    # unphysical sub-freezing SST values to leak into surface-flux
    # bulk formulas.
    sst_data = np.maximum(sst_data, constants.T_freeze_ocean)

    # --- Build KD-tree from ICON cell centroids (3D Cartesian) ---
    x_src = np.cos(clat) * np.cos(clon)
    y_src = np.cos(clat) * np.sin(clon)
    z_src = np.sin(clat)
    tree = cKDTree(np.stack([x_src, y_src, z_src], axis=-1))

    # --- Target grid points (3D Cartesian) ---
    grid_lat = np.asarray(grid.grid_lat)  # radians
    grid_lon = np.asarray(grid.grid_lon)  # radians
    target_shape = grid_lat.shape
    x_tgt = np.cos(grid_lat.ravel()) * np.cos(grid_lon.ravel())
    y_tgt = np.cos(grid_lat.ravel()) * np.sin(grid_lon.ravel())
    z_tgt = np.sin(grid_lat.ravel())
    _, idx = tree.query(np.stack([x_tgt, y_tgt, z_tgt], axis=-1))

    # --- Vectorised regridding (all timesteps at once) ---
    ntime = sst_data.shape[0]
    sst_regridded = sst_data[:, idx].reshape(ntime, *target_shape)
    sic_regridded = sic_data[:, idx].reshape(ntime, *target_shape)

    # --- Time axis ---
    # Mirror the lat-lon path: build RELATIVE first (span is invariant to
    # anchoring) to classify the file, then anchor ONLY a multi-year
    # (transient) file to the run calendar so a model day indexes it by real
    # date.  A <=12-record single-year climatology stays first-record-relative
    # (anchoring would randomise its seasonal phase against start_year).
    times_days = _time_coord_to_days(time_coord)
    ntime_axis = len(times_days)
    span_days = float(times_days[-1] - times_days[0]) if ntime_axis > 1 else 0.0
    is_transient = ntime_axis > 12 or (ntime_axis > 1 and span_days >= 366.0)
    if start_year is not None and is_transient:
        times_days = _time_coord_to_days(time_coord, epoch_year=start_year)
        # Coverage guard: model day 0 (== start_year-01-01) must fall inside
        # the record window (allow the first record up to ~1 month in, the
        # mid-month bcs anchor), else interpolation would silently clamp to
        # the wrong-era endpoint.  Start-of-run check only; a run extending
        # past the last record holds it (no wrap for transient files).
        if times_days[0] > 31.0 or times_days[-1] < 0.0:
            raise ValueError(
                f"AMIP forcing file does not cover start_year={start_year}: "
                f"records span days [{times_days[0]:.0f}, {times_days[-1]:.0f}]"
                f" relative to {start_year}-01-01. Stage a file that includes"
                " the run period, or set start_year to a year in the file."
            )

    sic_regridded = np.clip(sic_regridded, 0.0, 1.0)

    from legoesm.core.precision import get_policy
    _dtype = get_policy().storage
    # times pinned to float64: an anchored transient axis carries large
    # absolute day counts (~1e4-1e5) whose sub-day resolution (mid-month .5)
    # would be lost at float32 (ULP ~5e-3 day).
    return AMIPForcing(
        times=jnp.asarray(times_days, dtype=jnp.float64),
        sst=jnp.array(sst_regridded, dtype=_dtype),
        sic=jnp.array(sic_regridded, dtype=_dtype),
        config=config,
    )


def load_amip_forcing(
    config: AMIPForcingConfig, grid, start_year: int | None = None
) -> AMIPForcing:
    """Load AMIP forcing from NetCDF and regrid to the target grid.

    Supports regular lat-lon grids (COBE-SST2, HadISST, custom) and
    ICON unstructured grids (detected via ``cell`` dimension with
    ``clon``/``clat`` coordinates).

    Parameters
    ----------
    config : AMIPForcingConfig
        Forcing configuration with file path and variable names.
    grid : CubedSphereGrid or GaussianGrid
        Target grid.  Detected via ``hasattr(grid, 'n_lat')``.
    start_year : int, optional
        Simulation start year.  When given, the time axis is anchored to
        ``start_year-01-01`` so a model day indexes the file by real calendar
        date (AMIP-II protocol) — e.g. a 1979 run reads the 1979 records of a
        1870-2022 file instead of the 1870 records.  A multi-year file is then
        checked to cover the run start.  ``None`` keeps the legacy
        first-record-relative indexing (single-year / unit-test use).

    Returns
    -------
    AMIPForcing
        Regridded forcing data as JAX arrays.
        Shape is (ntime, 6, n, n) for cubed-sphere or
        (ntime, n_lat, n_lon) for Gaussian.
    """
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator

    path = config.path
    if not path:
        raise ValueError("AMIPForcingConfig.path is empty — provide a NetCDF file path")

    try:
        ds_sst = xr.open_dataset(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"AMIP forcing file not found: {path}"
        )
    except Exception as exc:
        raise OSError(
            f"Failed to open AMIP forcing file: {path}\n  {exc}"
        ) from exc

    sic_path = config.sic_path or path
    ds_sic = ds_sst
    if sic_path != path:
        try:
            ds_sic = xr.open_dataset(sic_path)
        except FileNotFoundError:
            ds_sst.close()
            raise FileNotFoundError(
                f"AMIP sea-ice forcing file not found: {sic_path}"
            )
        except Exception as exc:
            ds_sst.close()
            raise OSError(
                f"Failed to open AMIP sea-ice forcing file: {sic_path}\n  {exc}"
            ) from exc

    # --- Detect ICON unstructured grid ---
    if _is_icon_unstructured(ds_sst):
        if ds_sic is not ds_sst:
            ds_sic.close()
        ds_sst.close()
        return _load_icon_unstructured(config, grid, start_year=start_year)

    try:
        # --- Validate required variables ---
        missing = []
        for vname, label in [
            (config.sst_var, "SST"),
            (config.lat_var, "latitude"),
            (config.lon_var, "longitude"),
            (config.time_var, "time"),
        ]:
            if vname not in ds_sst:
                missing.append(f"  {label}: expected variable '{vname}' in {path}")
        if config.sic_var not in ds_sic:
            missing.append(
                f"  SIC: expected variable '{config.sic_var}' in {sic_path}"
            )
        if missing:
            available = ", ".join(
                sorted(
                    (set(ds_sst.data_vars) | set(ds_sst.coords))
                    | (set(ds_sic.data_vars) | set(ds_sic.coords))
                )
            )
            raise KeyError(
                "Missing variables in AMIP forcing:\n"
                + "\n".join(missing)
                + f"\nAvailable: {available}"
            )

        # Extract source grid from the SST file
        lat_src = ds_sst[config.lat_var].values.astype(np.float64)
        lon_src = ds_sst[config.lon_var].values.astype(np.float64)
        time_coord = ds_sst[config.time_var].values

        # Ensure longitude is in [0, 360) for wrapping
        lon_src = lon_src % 360.0

        # Sort by longitude if needed
        lon_order = np.argsort(lon_src)
        lon_src = lon_src[lon_order]

        # Extract SST and SIC
        sst_data = ds_sst[config.sst_var].values
        sic_data = ds_sic[config.sic_var].values

        # Units-attribute consistency guard (audit 2026-06-11) — the
        # same defense-in-depth the ozone (_ozone_unit_factor) and GHG
        # (_GHG_UNIT_TO_MOLE_FRACTION) loaders carry.  ``sst_offset`` /
        # ``sic_scale`` are CONFIG-driven, so a Kelvin SST file run with
        # the default +273.15 offset (or a percent SIC file with
        # scale=1.0) would silently corrupt the boundary condition
        # (SST→~575 K, masked only on the low side by the freeze floor;
        # SIC→100×, masked by the [0,1] clip).  Cross-check the file's
        # ``units`` attribute against the configured conversion and
        # raise a precise error instead of silently mis-forcing.
        def _norm(s):
            return s.strip().lower().replace("_", " ").replace("-", " ")
        _sst_units = _norm(str(ds_sst[config.sst_var].attrs.get("units", "")))
        _sic_units = _norm(str(ds_sic[config.sic_var].attrs.get("units", "")))
        _sst_is_celsius = _sst_units in (
            "degc", "deg c", "celsius", "c", "degrees celsius",
            "degree celsius", "degreesc",
        )
        _sst_is_kelvin = _sst_units in (
            "k", "kelvin", "degk", "deg k", "degrees kelvin",
            "degree kelvin",
        )
        # A Celsius file needs the +273.15 K offset (tight tolerance,
        # correct sign); anything else (wrong sign, 100, 150) is a
        # misconfiguration, not a C->K conversion (codex review).
        if _sst_is_celsius and abs(config.sst_offset - constants.T_freeze) > 1.0:
            raise ValueError(
                f"SST file {config.sst_var!r} has units={_sst_units!r} "
                f"(Celsius) but sst_offset={config.sst_offset} is not the "
                f"+{constants.T_freeze} K Celsius->Kelvin offset. Pass "
                "--sst-offset 273.15 (constants.T_freeze)."
            )
        if _sst_is_kelvin and abs(config.sst_offset) > 1.0:
            raise ValueError(
                f"SST file {config.sst_var!r} has units={_sst_units!r} "
                f"(Kelvin) but sst_offset={config.sst_offset} would add a "
                "spurious +273.15. Pass --sst-offset 0 for a Kelvin file."
            )
        # Units-INDEPENDENT physical sanity on the CONVERTED SST: even a
        # file with no/unknown units attribute is caught here.  A Kelvin
        # file run with the default +273.15 offset lands at ~575 K; a
        # Celsius file with offset=0 lands at ~0-30 K.  Both are
        # unphysical for an ocean surface (the downstream T_freeze_ocean
        # floor only catches the cold side, masking the hot Kelvin+offset
        # failure).  Bounds are generous (ocean SST spans ~271-310 K).
        _sst_max = float(np.nanmax(sst_data.astype(np.float64) + config.sst_offset))
        _sst_min = float(np.nanmin(sst_data.astype(np.float64) + config.sst_offset))
        if _sst_max > 340.0 or _sst_min < 240.0:
            raise ValueError(
                f"SST file {config.sst_var!r} converts to "
                f"[{_sst_min:.1f}, {_sst_max:.1f}] K after "
                f"sst_offset={config.sst_offset} (units={_sst_units!r}). "
                "That is outside the physical ocean range (~240-340 K) — "
                "likely a Kelvin file with a spurious +273.15 offset or a "
                "Celsius file with offset=0. Set --sst-offset correctly."
            )
        # SIC: '%'/'percent' need scale 0.01; '1'/'fraction' need 1.0.
        # Empty/unknown units are AMBIGUOUS (codex review): do not assume
        # fraction — the converted-value sanity below catches a bad scale.
        _sic_is_percent = _sic_units in ("%", "percent")
        # "(0 1)" is the NCAR-RDA ERA5 sea-ice (ci) units string "(0-1)" after _norm's
        # "-"->" " — an unambiguous [0,1] fraction, so positively validate it (else a wrong
        # sic_scale=0.01 on a real ERA5 file would slip past, all-zeros*0.01 masking it in
        # the value-sanity floor) (codex-review iter 420).
        _sic_is_fraction = _sic_units in ("1", "fraction", "dimensionless", "(0 1)")
        # sic_scale == 0 is an explicit no-sea-ice sensitivity run (zeroes SIC);
        # exempt it from the units/scale checks — it is unambiguously intentional
        # (nobody misconfigures a scale of exactly 0) and yields a valid [0,1] field.
        _no_ice = config.sic_scale == 0.0
        if _sic_is_percent and not _no_ice and abs(config.sic_scale - 0.01) > 1e-6:
            raise ValueError(
                f"SIC file {config.sic_var!r} has units={_sic_units!r} "
                f"(percent) but sic_scale={config.sic_scale}. Pass "
                "--sic-scale 0.01 for a percent SIC file."
            )
        if _sic_is_fraction and not _no_ice and abs(config.sic_scale - 1.0) > 1e-6:
            raise ValueError(
                f"SIC file {config.sic_var!r} has units={_sic_units!r} "
                f"(fraction) but sic_scale={config.sic_scale}. Pass "
                "--sic-scale 1.0 for a fraction SIC file."
            )
        # Units-independent SIC sanity on the CONVERTED value: a percent
        # file (0-100) scaled by 1.0 lands at ~50-100 — a wrong-scale error.
        # The threshold allows the PCMDI mid-month "bcs" convention (Taylor
        # et al. 2000), whose reconstructed anchors legitimately overshoot
        # [0,1] to ~+/-25 after correct 0.01 scaling (raw ~+/-2500%); those
        # overshoots are clipped only AFTER time interpolation, per protocol,
        # so they must not be rejected here. A genuine 100x scale error still
        # trips it.
        _sic_max = float(np.nanmax(np.abs(sic_data.astype(np.float64)
                                          * config.sic_scale)))
        if _sic_max > 50.0:
            raise ValueError(
                f"SIC file {config.sic_var!r} reaches {_sic_max:.1f} "
                f"after sic_scale={config.sic_scale} (units={_sic_units!r}) "
                "— far beyond the +/-25 of the mid-month bcs convention. "
                "Likely a percent file needing --sic-scale 0.01."
            )

        if sst_data.ndim == 2:
            sst_data = sst_data[None, ...]
        if sic_data.ndim == 2:
            sic_data = sic_data[None, ...]
        if sst_data.ndim != 3 or sic_data.ndim != 3:
            raise ValueError(
                "AMIP forcing variables must be 2D or 3D with optional time axis. "
                f"Got SST ndim={sst_data.ndim}, SIC ndim={sic_data.ndim}."
            )

        if sic_data.shape[0] != sst_data.shape[0]:
            raise ValueError(
                "SST and SIC forcing must have the same number of time records. "
                f"Got SST={sst_data.shape[0]} and SIC={sic_data.shape[0]}."
            )

        sst_data = sst_data[:, :, lon_order]
        sic_data = sic_data[:, :, lon_order]

        # Apply unit conversions
        sst_data = sst_data.astype(np.float64) + config.sst_offset
        sic_data = sic_data.astype(np.float64) * config.sic_scale

        # Fill NaN (land points) with nearest neighbor
        sst_data = _fill_nan_nearest(sst_data, lat_src, lon_src)
        sic_data = _fill_nan_nearest(sic_data, lat_src, lon_src)

        # NOTE: SIC clamp to [0,1] and the SST freezing floor are applied only
        # AFTER time interpolation (get_forcing_at_time), NOT to these mid-month
        # anchors. The PCMDI "bcs" values (Taylor et al. 2000) deliberately
        # overshoot the physical range so that clip(linear-interp(anchors))
        # reproduces the observed monthly means; clamping the anchors here would
        # damp the SIC/SST seasonal cycle near the pack ice and cold tongue.

        # Wrap longitude for interpolation continuity
        # Pad one column at each end
        lon_wrapped = np.concatenate([lon_src[-1:] - 360.0, lon_src, lon_src[:1] + 360.0])
        sst_wrapped = np.concatenate([sst_data[:, :, -1:], sst_data, sst_data[:, :, :1]], axis=2)
        sic_wrapped = np.concatenate([sic_data[:, :, -1:], sic_data, sic_data[:, :, :1]], axis=2)

        # Use protocol: grid_lat gives 2D (or 3D for CS) lat in radians
        grid_lat = np.asarray(grid.grid_lat)
        grid_lon = np.asarray(grid.grid_lon)
        is_gaussian = grid_lat.ndim == 2 and not hasattr(grid, 'n')

        if is_gaussian:
            target_lat_1d = np.asarray(grid.lat) * 180.0 / np.pi
            target_lon_1d = np.asarray(grid.lon) * 180.0 / np.pi
            target_lon_1d = target_lon_1d % 360.0
            target_lon_2d, target_lat_2d = np.meshgrid(target_lon_1d, target_lat_1d)
            target_shape = grid_lat.shape
        else:
            target_lat_2d = grid_lat * 180.0 / np.pi
            target_lon_2d = grid_lon * 180.0 / np.pi
            target_lon_2d = target_lon_2d % 360.0
            target_shape = grid_lat.shape

        ntime = sst_data.shape[0]
        sst_regridded = np.zeros((ntime, *target_shape), dtype=np.float64)
        sic_regridded = np.zeros((ntime, *target_shape), dtype=np.float64)

        target_points = np.stack([target_lat_2d.ravel(), target_lon_2d.ravel()], axis=-1)

        for t in range(ntime):
            interp_sst = RegularGridInterpolator(
                (lat_src, lon_wrapped), sst_wrapped[t],
                method="linear", bounds_error=False, fill_value=None,
            )
            interp_sic = RegularGridInterpolator(
                (lat_src, lon_wrapped), sic_wrapped[t],
                method="linear", bounds_error=False, fill_value=None,
            )
            sst_regridded[t] = interp_sst(target_points).reshape(target_shape)
            sic_regridded[t] = interp_sic(target_points).reshape(target_shape)

        # Time axis. Build RELATIVE first (the span is invariant to anchoring)
        # to classify the file, then anchor ONLY a multi-year (transient) file
        # to the run calendar so a model day indexes it by real date. A short
        # climatology stays first-record-relative: anchoring a 12-month file
        # dated some nominal year to a different start_year, then wrapping by its
        # ~334-day span, would randomise its seasonal phase against that
        # arbitrary year. Left relative, its annual wrap keeps the phase.
        times_days = _time_coord_to_days(time_coord)
        span_days = float(times_days[-1] - times_days[0]) if ntime > 1 else 0.0
        # Transient (multi-year) vs a <=12-record single-year climatology. Key
        # on BOTH: a 13-month noleap series spans exactly 365 days (< 366) yet
        # touches two calendar years, so >12 records also counts as transient.
        is_transient = ntime > 12 or (ntime > 1 and span_days >= 366.0)
        if start_year is not None and is_transient:
            times_days = _time_coord_to_days(time_coord, epoch_year=start_year)
            # Coverage guard: model day 0 (== start_year-01-01) must fall inside
            # the record window (allow the first record up to ~1 month in, the
            # mid-month bcs anchor), else the interpolation would silently clamp
            # to the wrong-era endpoint. Note this checks the run START only; a
            # run extending past the last record holds it (no wrap for transient
            # files — see get_forcing_at_time), not a jump back to the file era.
            if times_days[0] > 31.0 or times_days[-1] < 0.0:
                raise ValueError(
                    f"AMIP forcing file does not cover start_year={start_year}: "
                    f"records span days [{times_days[0]:.0f}, {times_days[-1]:.0f}]"
                    f" relative to {start_year}-01-01. Stage a file that includes"
                    " the run period, or set start_year to a year in the file."
                )
    finally:
        if ds_sic is not ds_sst:
            ds_sic.close()
        ds_sst.close()

    # Anchor values are kept un-clamped (mid-month bcs convention); the physical
    # SIC [0,1] clip and SST freezing floor are applied after time interpolation
    # in get_forcing_at_time. ``times`` is pinned to float64 because an anchored
    # transient axis carries large absolute day counts (~1e4-1e5) whose sub-day
    # resolution (mid-month .5) would be lost at float32 (ULP ~5e-3 day).
    from legoesm.core.precision import get_policy
    _dtype = get_policy().storage
    return AMIPForcing(
        times=jnp.asarray(times_days, dtype=jnp.float64),
        sst=jnp.array(sst_regridded, dtype=_dtype),
        sic=jnp.array(sic_regridded, dtype=_dtype),
        config=config,
    )


def get_forcing_at_time(
    forcing: AMIPForcing,
    day: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Linearly interpolate SST and SIC to a given day.

    Parameters
    ----------
    forcing : AMIPForcing
        Loaded forcing data.
    day : float
        Day on the same axis as ``forcing.times`` — absolute days since
        ``start_year-01-01`` when the forcing was loaded with ``start_year``
        (AMIP-II protocol), else days since the first record.

    Returns
    -------
    sst : jax.Array
        Interpolated SST [K], same spatial shape as forcing.sst[0].
    sic : jax.Array
        Interpolated SIC [0-1], same spatial shape as forcing.sic[0].
    """
    times = forcing.times
    ntime = times.shape[0]
    # Physical bounds are applied to the instantaneous INTERPOLATED value only,
    # not to the mid-month bcs anchors (Taylor et al. 2000; see load_amip_forcing).
    t_freeze = forcing.config.T_ice

    # Single-record forcing (e.g. a climatological mean, or a 2D file promoted
    # to shape (1, ...) by load_amip_forcing): no interpolation is possible, so
    # return the single field. ``ntime`` is a static shape, so this Python
    # branch is JIT-safe. Without it, the cyclic-wrap below leaves period==0 and
    # ``clip(idx, 0, ntime-2) = clip(idx, 0, -1)`` returns idx=-1 with dt=0,
    # blowing up the weight (~5e10) and producing ~0 K SST via FP cancellation
    # for any requested day != times[0].
    if ntime == 1:
        return (jnp.maximum(forcing.sst[0], t_freeze),
                jnp.clip(forcing.sic[0], 0.0, 1.0))

    # Repeat the annual cycle ONLY for a monthly climatology: <=12 records that
    # don't already span a full year. Wrap on a 365-day noleap (model-clock)
    # period, NOT the ~334-day record span (which would drift the season
    # ~31 d/yr). A >12-record OR >=1-yr file is TRANSIENT (mirror
    # load_amip_forcing's ``is_transient = ntime > 12 or span >= 366``): period 0
    # => never wrap, so a run past the file holds its last record instead of
    # jumping back to the first era. ``ntime`` is a static shape => JIT-safe
    # Python branch; ``span``/``period`` are traced.
    span = times[-1] - times[0]
    if ntime > 12:
        period = jnp.zeros((), dtype=times.dtype)
    else:
        period = jnp.where(span < 366.0, 365.0, 0.0)
    wrap = period > 0
    # Safe divisor: the transient path (period == 0) still TRACES the modulo in
    # the non-selected jnp.where branch, and ``x % 0`` emits a NaN that trips
    # JAX_DEBUG_NANS and poisons any reverse-mode VJP (grad flows through both
    # where-branches). Divide by 1.0 there — the result is discarded by
    # ``where(wrap, ..., day)``.
    period_safe = jnp.where(wrap, period, 1.0)
    day = jnp.where(wrap, times[0] + (day - times[0]) % period_safe, day)

    # Bracketing indices. The climatology path closes the Dec->Jan seam: the
    # anchor after the last mid-month record (idx == ntime-1) is field[0] one
    # period ahead, so days past mid-December interpolate December->January
    # (Taylor et al. 2000 cyclic bcs) rather than holding December then jumping
    # at the year boundary. The transient path clamps to the interior and lets
    # the [0,1] weight clamp HOLD the endpoints (no extrapolation off a clamped
    # bracket, which is physically unbounded: SST would run to hundreds of K).
    # Index arithmetic only (no per-step array growth — ntime is ~1836 for a
    # transient input4MIPs file).
    idx = jnp.searchsorted(times, day, side="right") - 1
    idx = jnp.where(wrap, jnp.clip(idx, 0, ntime - 1),
                    jnp.clip(idx, 0, ntime - 2))
    idx_next = jnp.where(wrap, jnp.mod(idx + 1, ntime), idx + 1)
    t_i = times[idx]
    t_next = times[idx_next] + jnp.where(wrap & (idx == ntime - 1), period, 0.0)
    dt = jnp.maximum(t_next - t_i, 1e-10)  # avoid division by zero
    weight = jnp.clip((day - t_i) / dt, 0.0, 1.0)

    # Linear interpolation
    sst = (1.0 - weight) * forcing.sst[idx] + weight * forcing.sst[idx_next]
    sic = (1.0 - weight) * forcing.sic[idx] + weight * forcing.sic[idx_next]

    # Clamp to physical bounds (SIC in [0,1], SST at/above the seawater freezing
    # point) — on the interpolated value only. Cast back to the STORED forcing
    # dtype: the float64 time axis (required for multi-decade days-since-epoch
    # arithmetic) otherwise promotes a float32-policy SST/SIC field to float64
    # through the interpolation weight.
    sst = jnp.maximum(sst, t_freeze).astype(forcing.sst.dtype)
    sic = jnp.clip(sic, 0.0, 1.0).astype(forcing.sic.dtype)

    return sst, sic


def _fill_nan_nearest(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    """Fill NaN values with nearest valid neighbor on lat-lon grid.

    Parameters
    ----------
    data : np.ndarray
        Data array, shape (ntime, nlat, nlon). NaN marks missing values.
    lat : np.ndarray
        Latitude array [degrees], shape (nlat,).
    lon : np.ndarray
        Longitude array [degrees], shape (nlon,).

    Returns
    -------
    np.ndarray
        Data with NaNs filled.
    """
    if not np.any(np.isnan(data)):
        return data

    from scipy.interpolate import NearestNDInterpolator

    # Build coordinate meshgrid
    lon_2d, lat_2d = np.meshgrid(lon, lat)

    # Convert to Cartesian for distance (handles wrap-around)
    lat_rad = np.deg2rad(lat_2d)
    lon_rad = np.deg2rad(lon_2d)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    coords = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)

    filled = data.copy()
    for t in range(data.shape[0]):
        frame = data[t]
        mask_valid = ~np.isnan(frame.ravel())
        if mask_valid.all() or not mask_valid.any():
            continue

        interp = NearestNDInterpolator(
            coords[mask_valid], frame.ravel()[mask_valid]
        )
        frame_filled = interp(coords).reshape(frame.shape)
        filled[t] = frame_filled

    return filled
