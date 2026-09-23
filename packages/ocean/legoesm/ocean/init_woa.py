"""WOA18 (World Ocean Atlas 2018) ocean initialization.

Provides realistic temperature and salinity initial conditions by
interpolating WOA18 climatology (annual or monthly) to the model grid
and vertical coordinate.  Supports cubed-sphere, lat-lon, MPAS, and
spectral grids.

WOA18 data are expected as Zarr stores or NetCDF files on a 1-degree
lat-lon grid with standard depth levels.  If files are not available, a
latitude-dependent analytical fallback is provided.

OMIP-2 protocol notes
---------------------
The OMIP-2 (CMIP6-OMIP) experimental design (Griffies et al. 2016, GMD)
specifies that ocean simulations start from observed temperature and
salinity climatology to suppress spin-up artefacts.  This module
implements that recipe:

    * Annual or monthly WOA18 climatology (``month=None`` or
      ``month in 1..12``).
    * Bilinear horizontal interpolation (OMIP-2 default).  Nearest-
      neighbour is retained for column-grid debugging.
    * Bathymetry-aware vertical mask so model cells below the local
      water column receive the deep-ocean fill rather than spurious
      surface-extended values.

References
----------
Locarnini, R. A., et al. (2019): World Ocean Atlas 2018, Volume 1:
    Temperature. NOAA Atlas NESDIS 81, 52 pp.
Zweng, M. M., et al. (2019): World Ocean Atlas 2018, Volume 2:
    Salinity. NOAA Atlas NESDIS 82, 50 pp.
Griffies, S. M., et al. (2016): OMIP contribution to CMIP6: Experimental
    and diagnostic protocol for the physical component of the Ocean
    Model Intercomparison Project. Geosci. Model Dev., 9, 3231-3296.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import warnings

import numpy as np

from legoesm import constants
from legoesm.ocean.vertical import OceanZStarCoordinate


# ==============================================================================
# WOA18 standard depth levels [m] (57 levels, surface to 5500 m)
# ==============================================================================
WOA_DEPTHS = np.array([
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
    80, 85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300,
    325, 350, 375, 400, 425, 450, 475, 500, 550, 600, 650, 700, 750,
    800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300,
    1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800,
    1850, 1900, 1950, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
    2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600,
    3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600,
    4700, 4800, 4900, 5000, 5100, 5200, 5300, 5400, 5500,
], dtype=np.float64)


def interp_column_to_depths(
    col_src: np.ndarray,
    src_depths: np.ndarray,
    dst_depths: np.ndarray,
) -> np.ndarray:
    """Interpolate one T/S column from ``src_depths`` onto ``dst_depths``.

    NaN-safe: non-finite source entries (land, below-seafloor fill) are
    dropped before interpolating, and a column with fewer than two valid
    entries returns all-NaN so the caller's fill logic — not a fabricated
    value — decides what goes there.

    BELOW the deepest valid source level the result is ``NaN``, not the
    deepest valid value.  Holding that value (``np.interp``'s edge behaviour)
    is how a shelf column's warm bottom water reached the abyss: re-levelling
    the PHC3 winter climatology onto the 102 WOA depths wrote 583 source
    columns carrying above 15 degC below 1000 m, the worst at 29.4 degC where
    the real deep ocean is about 2 degC, and 580 of those had an exactly
    constant sub-1000 m tail — the signature of the held value.  Those columns
    then fed every model column standing over them.  ``NaN`` instead lets the
    per-level nearest-valid fill in :func:`init_ocean_from_woa` take the value
    from the nearest source column that actually has an observation at that
    depth.  ABOVE the shallowest valid level the edge value is still held: the
    shallowest observation is the surface, and holding it is what an
    observational analysis does there.

    Shared helper for the IC-preparation scripts that re-level an external
    climatology onto :data:`WOA_DEPTHS`.  NOTE this is no longer *required*:
    :func:`init_ocean_from_woa` now reads the source depth axis from the FILE
    and only falls back to :data:`WOA_DEPTHS` when the file has none.
    Re-levelling first remains useful when a caller wants one common axis for
    several sources.

    Parameters
    ----------
    col_src : 1-D array of values at ``src_depths``.
    src_depths : 1-D array, positive downward, ascending.
    dst_depths : 1-D array, positive downward, ascending.
    """
    # Both sides must be finite: a valid value paired with a NaN DEPTH would
    # otherwise enter np.interp's xp (undefined) and make the deep cutoff's
    # ``.max()`` NaN, silently disabling it (codex adversarial review).
    good = np.isfinite(col_src) & np.isfinite(np.asarray(src_depths, dtype=float))
    if good.sum() < 2:
        return np.full(np.shape(dst_depths), np.nan)
    src_depths = np.asarray(src_depths)
    out = np.interp(dst_depths, src_depths[good], np.asarray(col_src)[good])
    return np.where(np.asarray(dst_depths) > src_depths[good].max(),
                    np.nan, out)


def _interp_profile_to_z_coord(
    profile: np.ndarray,
    woa_depths: np.ndarray,
    z_coord: OceanZStarCoordinate,
    model_depths: np.ndarray | None = None,
) -> np.ndarray:
    """Interpolate a 1-D WOA profile onto model z-star full levels.

    Parameters
    ----------
    profile : array, shape (n_woa,)
        WOA profile values (T or S).
    woa_depths : array, shape (n_woa,)
        WOA depth levels [m] (positive downward).
    z_coord : OceanZStarCoordinate

    Returns
    -------
    array, shape (nlev,)
        Profile on model full levels.
    """
    # Model level depths (positive down).  ``model_depths`` is THIS column's
    # true cell-centre depths when the caller has partial cells; without it the
    # reference full-cell centres are used, which is what a pure z* column has.
    if model_depths is None:
        model_depths = np.abs(np.asarray(z_coord.z_full_ref))
    else:
        model_depths = np.abs(np.asarray(model_depths, dtype=np.float64))
        if not np.all(np.isfinite(model_depths)):
            raise ValueError(
                "model_depths must be finite; a non-finite cell-centre depth "
                "would place the profile at an undefined level")
        if not np.all(np.diff(model_depths) >= 0.0):
            raise ValueError(
                "model_depths must not decrease downward; an inverted column "
                "means an off-by-one level or a wrong sign. (A FLAT tail is "
                "legitimate: cells below the seafloor have zero thickness and "
                "inherit the bottom cell's depth, and they are masked out.)")

    # Remove NaN entries from WOA profile
    valid = ~np.isnan(profile)
    if valid.sum() < 2:
        # Not enough valid data.  ONE valid entry is propagated down the
        # column; NONE returns NaN so the caller's fill (T_fill_C / S_fill_psu)
        # applies.
        #
        # Returning 0.0 here instead of NaN was a real defect: 0.0 is not NaN,
        # so ``np.where(np.isnan(T_out), T_fill, T_out)`` below never replaced
        # it, and a wet cell whose source column had no data entered the model
        # as T = 0 degC, S = 0 PSU.  Fresh water at 0 degC is rho = 999.8
        # against ~1027 for real sea water, so such a cell sat next to normal
        # ocean with a density jump up to 30 kg/m^3 -- larger than the entire
        # ocean's density range.  On the FESOM2-matched CORE2 config that gave
        # a hydrostatic pressure-gradient acceleration of 2e-2 m/s^2 and ~42 m/s
        # in a single 2400 s step, i.e. an immediate blowup (1569 wet columns
        # affected; see docs/ocean/fidelity/fesom2_gap_analysis.md 3.7).
        return np.full(
            z_coord.n_levels,
            profile[valid][0] if valid.any() else np.nan,
        )

    return np.interp(model_depths, woa_depths[valid], profile[valid])


def _analytical_woa_profiles(
    lat_deg: np.ndarray,
    z_coord: OceanZStarCoordinate,
) -> tuple[np.ndarray, np.ndarray]:
    """Latitude-dependent analytical T/S profiles mimicking WOA18.

    Provides a reasonable approximation when WOA18 NetCDF files are not
    available.  Temperature uses a latitude-dependent surface value with
    exponential decay.  Salinity uses a latitude-dependent profile with
    a subsurface maximum (subtropical halocline).

    Parameters
    ----------
    lat_deg : array, shape (...)
        Latitude in degrees.
    z_coord : OceanZStarCoordinate

    Returns
    -------
    T : array, shape (..., nlev) — potential temperature [degC]
    S : array, shape (..., nlev) — salinity [PSU]
    """
    z_full = np.asarray(z_coord.z_full_ref)  # negative, (nlev,)
    lat = np.asarray(lat_deg)

    # --- Temperature ---
    # SST: ~28 degC at equator, ~0 degC at poles
    T_water_init_C = 28.0 * np.cos(np.radians(lat)) ** 2
    T_deep = 1.5  # Bottom water ~1.5 degC globally
    scale = 500.0  # e-folding depth [m]

    # z_full is negative; exp(z/scale) = exp(-depth/scale)
    T = T_deep + (T_water_init_C[..., np.newaxis] - T_deep) * np.exp(z_full / scale)

    # --- Salinity ---
    # Surface: fresher near equator (ITCZ precip) and poles; saltier subtropics
    S_surface = 34.5 + 1.0 * np.cos(np.radians(2 * lat)) ** 2
    S_deep = 34.7  # Deep water ~34.7 PSU

    # Subsurface salinity maximum at ~100m in subtropics
    S_scale = 1000.0
    S = S_deep + (S_surface[..., np.newaxis] - S_deep) * np.exp(z_full / S_scale)

    return T, S


def _open_woa_dataset(path: str | Path):
    """Open a WOA18 file as xarray Dataset (Zarr or NetCDF).

    Uses ``decode_times=False`` because the NCEI WOA18 distribution
    encodes the time axis as ``"months since 1955-01-01"``, which
    xarray cannot decode without ``cftime`` and which is irrelevant
    for our use (we only read the annual-mean climatology, which is
    a single time slice we treat as static).
    """
    import os
    import xarray as xr
    path_str = str(path)
    if os.path.isdir(path_str) or path_str.endswith(".zarr"):
        return xr.open_zarr(path_str, decode_times=False)
    return xr.open_dataset(path_str, decode_times=False)


#: Coordinate names a climatology may use for its vertical axis.  Matched
#: case-insensitively; the variable's OWN dimension order is consulted first,
#: so an unusual name is still found as long as it is the field's 2nd axis.
_DEPTH_AXIS_NAMES: tuple[str, ...] = (
    "depth", "z", "lev", "level", "deptht", "depth_bnds_mid", "olevel",
)


def _depth_axis_of(ds, var_name: str):
    """Positive depths [m] for ``ds[var_name]``'s vertical axis, or ``None``.

    Resolution order, so an unusual coordinate name is not silently missed:
    (1) the variable's OWN second dimension (time, DEPTH, lat, lon) when that
    dimension has coordinate values; (2) any recognised name from
    :data:`_DEPTH_AXIS_NAMES`, case-insensitively.
    """
    dims = getattr(ds[var_name], "dims", ())
    if len(dims) >= 2:
        cand = dims[1]
        if cand in ds.coords or cand in ds.variables:
            return np.abs(np.asarray(ds[cand].values, dtype=np.float64))
    lowered = {str(k).lower(): k for k in
               list(ds.coords) + list(ds.variables)}
    for name in _DEPTH_AXIS_NAMES:
        if name in lowered:
            return np.abs(
                np.asarray(ds[lowered[name]].values, dtype=np.float64))
    return None


def load_woa18(
    T_path: str | Path | None = None,
    S_path: str | Path | None = None,
    month: int | None = None,
    T_var: str | None = None,
    S_var: str | None = None,
    monthly_layout: str = "concatenated",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray | None]:
    """Load WOA18 climatology (annual or monthly) from Zarr/NetCDF.

    Parameters
    ----------
    T_path : path or None
        Path to WOA18 temperature file/store.
    S_path : path or None
        Path to WOA18 salinity file/store.
    month : int or None
        Month index 1..12 to extract from the WOA dataset's time axis.
        ``None`` returns the annual mean (time index 0, which is the
        WOA18 convention for ``*_an`` annual fields).  Monthly fields
        are stored along the ``time`` dimension with length 12 in the
        WOA18 monthly distribution (Vols. 1/2 NetCDF format
        ``woa18_decav_t<MM>_01.nc`` concatenated, or the single-file
        ``woa18_decav_t01_01.nc`` monthly products).
    T_var, S_var : str or None
        Override the dataset variable names.  Defaults: ``"t_an"`` and
        ``"s_an"`` for annual mean, ``"t_an"`` / ``"s_an"`` again for
        the monthly products (they reuse the name with a length-12
        time axis).  Set to the actual variable name (e.g. ``"t_mn"``)
        for non-standard files.
    monthly_layout : {"concatenated", "single_file_per_month"}
        File layout convention:
            * ``"concatenated"`` (default): one file with a length-12
              time axis.  ``month=N`` reads index ``N - 1``.
            * ``"single_file_per_month"``: caller has already selected
              the file matching the desired month.  ``month`` must be
              ``None`` (the loader reads the single time slice);
              passing ``month is not None`` raises ``ValueError`` to
              prevent silent month mismatch where ``month=2`` is
              requested but the file is a January climatology.

    Returns
    -------
    T_woa : array, shape (n_lat, n_lon, n_depth)
    S_woa : array, shape (n_lat, n_lon, n_depth)
    lat_woa : array, shape (n_lat,)  — degrees
    lon_woa : array, shape (n_lon,)  — degrees [0, 360)
    depth_woa : array, shape (n_depth,) — POSITIVE depths [m] read from the
        FILE, or ``None`` if it carries no recognisable depth coordinate
        (``depth``/``z``/``lev``/``level``/``deptht``).  Callers must prefer
        this over :data:`WOA_DEPTHS`: a non-WOA climatology on its own levels
        (PHC3's 33, WOCE's 75) read against the hardcoded WOA axis puts deep
        water in the thermocline without raising anything.
    """
    if monthly_layout not in ("concatenated", "single_file_per_month"):
        raise ValueError(
            f"monthly_layout must be 'concatenated' or "
            f"'single_file_per_month'; got {monthly_layout!r}."
        )
    if monthly_layout == "single_file_per_month" and month is not None:
        raise ValueError(
            "monthly_layout='single_file_per_month' expects the caller "
            "to have pre-selected the file matching the desired month; "
            "pass month=None and the loader will read the single time "
            "slice.  Got month="
            f"{month}."
        )
    if month is not None and not (1 <= month <= 12):
        raise ValueError(
            f"month must be 1..12 or None (annual); got {month}."
        )
    requested_idx = 0 if month is None else (month - 1)
    t_name = T_var if T_var is not None else "t_an"
    s_name = S_var if S_var is not None else "s_an"

    def _read(ds, var_name, label, path):
        if var_name not in ds.data_vars:
            avail = list(ds.data_vars)
            ds.close()
            raise ValueError(
                f"WOA {label} file {path!r} has no variable "
                f"{var_name!r}; available data vars: {avail}.  Pass "
                f"the correct {label}_var kwarg."
            )
        n_time = ds[var_name].shape[0]
        # Convention dispatch:
        #   * ``month is None`` → read index 0 (annual mean, or the
        #     single time slice in per-month files).
        #   * ``monthly_layout='concatenated'`` + ``month=N`` →
        #     index N-1; raise if the time axis is too short.  No
        #     silent fallback to index 0, since that would return
        #     the wrong month with no error.
        if month is None:
            idx = 0
        else:
            # Per-month-file callers go through ``monthly_layout=
            # 'single_file_per_month'`` which forces ``month is None``
            # before reaching this point (validated above).
            idx = requested_idx
        if idx >= n_time:
            ds.close()
            raise IndexError(
                f"WOA {label} file {path!r} variable {var_name!r} has "
                f"time length {n_time}; cannot extract month={month} "
                f"(index {idx}).  Pass a concatenated length-12 "
                f"monthly file, or use monthly_layout='single_file_"
                f"per_month' with the matching per-month file and "
                f"month=None."
            )
        return np.array(ds[var_name].values[idx, :, :, :])

    ds_T = _open_woa_dataset(T_path)
    T_woa = _read(ds_T, t_name, "T", T_path)
    lat_woa = np.array(ds_T["lat"].values)
    lon_woa = np.array(ds_T["lon"].values)
    # The FILE's own depth axis, when it has one.  Reading it here rather than
    # assuming ``WOA_DEPTHS`` is what lets a non-WOA climatology (PHC3, WOCE,
    # EN4) be used without its deep values being read as if they sat in the
    # top few hundred metres -- a silently wrong ocean, not an error.  A file
    # with no recognisable depth coordinate returns ``None`` and the caller
    # falls back to ``WOA_DEPTHS``.
    depth_woa = _depth_axis_of(ds_T, t_name)
    ds_T.close()

    ds_S = _open_woa_dataset(S_path)
    S_woa = _read(ds_S, s_name, "S", S_path)
    depth_S = _depth_axis_of(ds_S, s_name)
    ds_S.close()

    # T and S are interpolated onto the model levels with ONE axis, so they
    # must agree.  Equal level COUNTS with different depths would otherwise
    # place salinity at the temperature file's depths without a word.
    if depth_woa is not None and depth_S is not None:
        # Tolerance is PRACTICAL, not exact: a legitimate pair can differ by
        # float32-vs-float64 storage of the same nominal levels (5500.0 m is
        # 5500.0005 in float32), and rejecting that would be a false alarm.
        # 1 mm absolute + 1e-6 relative is far below any real level spacing
        # and far above storage roundoff.
        if depth_woa.shape != depth_S.shape or not np.allclose(
                depth_woa, depth_S, rtol=1e-6, atol=1e-3):
            raise ValueError(
                f"T file {T_path!r} and S file {S_path!r} have DIFFERENT "
                f"depth axes ({depth_woa[:3]}... vs {depth_S[:3]}...). "
                "They are interpolated onto the model levels with a single "
                "axis, so salinity would be placed at the temperature file's "
                "depths. Re-level them onto a common axis first."
            )
    elif depth_S is not None and depth_woa is None:
        # Only S carries an axis: use it rather than falling back to WOA's.
        depth_woa = depth_S

    # Transpose to (lat, lon, depth) for easier interpolation
    T_woa = np.transpose(T_woa, (1, 2, 0))
    S_woa = np.transpose(S_woa, (1, 2, 0))

    for _axis, _field, _name, _path, _label in (
        (depth_woa, T_woa, t_name, T_path, "T"),
        (depth_S, S_woa, s_name, S_path, "S"),
    ):
        if _axis is not None and _axis.size != _field.shape[-1]:
            raise ValueError(
                f"WOA {_label} file {_path!r}: depth coordinate has "
                f"{_axis.size} entries but {_name!r} has "
                f"{_field.shape[-1]} levels."
            )
    if T_woa.shape[-1] != S_woa.shape[-1]:
        raise ValueError(
            f"T file {T_path!r} has {T_woa.shape[-1]} levels but S file "
            f"{S_path!r} has {S_woa.shape[-1]}; they are interpolated onto "
            "the model levels together."
        )

    return T_woa, S_woa, lat_woa, lon_woa, depth_woa


def _fill_source_levels_nearest_valid(
    fields: "list[np.ndarray] | tuple[np.ndarray, ...]",
    src_lat_deg: np.ndarray,
    src_lon_deg: np.ndarray,
    *,
    void_fill: bool = False,
) -> tuple[list[np.ndarray], int, int]:
    """Give every source cell an observation at every depth it is missing one.

    ``void_fill=True`` additionally treats a missing cell whose nearest donor
    is NOT one of its eight grid neighbours as lying in a DATA VOID (a basin or
    channel the climatology never observed: the Marmara, the Black Sea, the
    Canadian Archipelago channels).  Nearest-donor filling stitches such a
    region from whichever observed cells happen to be closest, so adjacent
    void cells take water from different seas and the seam is a density wall
    (3.7 kg/m3 across 6 km in the Marmara on ORCA12; 10 kg/m3 at 73N 107W).
    Void cells are instead filled with the harmonic (Laplace) field bounded by
    the surrounding non-void cells, per level and per field: horizontally
    smooth, and by the maximum principle never outside the range of the
    surrounding data.  A cell within one grid step of an observation keeps its
    nearest donor as before.

    For each depth level independently, a source cell with no observation takes
    the value of the NEAREST source cell that does have one AT THAT SAME DEPTH.
    This is what the observational analyses themselves do, and it is the only
    fill that keeps a water mass both local and vertically ordered.

    Why it is needed here: an observed column stops at the local seafloor, so a
    model column standing over a shallow source cell has no observation below
    that cell's floor.  The column fallback in
    :func:`_interp_profile_to_z_coord` then propagated the shallowest valid
    value down the whole column -- surface water written into the abyss.  On
    the 28 km icosahedral mesh with ETOPO bathymetry that put 1228 wet columns
    above 15 degC below a kilometre, the worst at 29.3 degC, against ~2 degC
    for the real deep ocean; the resulting density contrast drove a first-step
    pressure-gradient acceleration of 4.5e-3 m/s^2.

    A level with NO observation anywhere on the source grid is left untouched:
    there is no donor for it.  Such a level is then bridged in the VERTICAL by
    :func:`_interp_profile_to_z_coord`, which interpolates across it from the
    observed levels above and below — it does NOT reach the caller's constant
    deep fill (codex adversarial review; the earlier wording here claimed it
    did and was wrong).

    All the fields are filled TOGETHER, from one donor column per cell and
    depth: a cell counts as missing where ANY field is missing there, and the
    donor supplies every field.  Temperature and salinity therefore always
    describe the same water.  Choosing a donor per field would let a cell take
    its temperature from one ocean and its salinity from another, which is a
    density anomaly built out of two real observations.

    Parameters
    ----------
    fields : sequence of arrays, each shape (n_lat, n_lon, n_depth)
        Source fields with ``NaN`` marking a missing observation.
    src_lat_deg, src_lon_deg : array, shape (n_lat,) / (n_lon,)
        Source grid coordinates in degrees.

    Returns
    -------
    filled : list of arrays, same shapes as ``fields``
    n_filled : int
        Number of (cell, level) entries per field that were given a donor.
    n_void : int
        Number of those entries that were harmonic-filled as data void
        (always 0 unless ``void_fill``).
    """
    from legoesm.grids.regridding import fill_missing_nearest_valid

    arrs = [np.asarray(f, dtype=np.float64) for f in fields]
    if not arrs:
        raise ValueError("fields must not be empty.")
    arr = arrs[0]
    for a in arrs:
        if a.ndim != 3:
            raise ValueError(
                f"source field must be (n_lat, n_lon, n_depth); got {a.shape}.")
        if a.shape != arr.shape:
            raise ValueError(
                f"all source fields must share a shape; got {a.shape} and "
                f"{arr.shape}.")

    # Unit-sphere Cartesian coordinates: the nearest source cell is then exact
    # across the dateline and over the poles, which differencing degrees is not.
    lon2d, lat2d = np.meshgrid(np.asarray(src_lon_deg, dtype=np.float64),
                               np.asarray(src_lat_deg, dtype=np.float64))
    lat_rad = np.radians(lat2d).ravel()
    lon_rad = np.radians(lon2d).ravel()
    cos_lat = np.cos(lat_rad)
    coords = np.stack([cos_lat * np.cos(lon_rad),
                       cos_lat * np.sin(lon_rad),
                       np.sin(lat_rad)], axis=-1)

    n_depth = arr.shape[-1]
    n_points = arr.shape[0] * arr.shape[1]
    missing = np.zeros((n_depth, n_points), dtype=bool)
    for a in arrs:
        missing |= ~np.isfinite(a.reshape(-1, n_depth).T)
    has_donor = ~np.all(missing, axis=1)
    # A level with no donor anywhere: every field is set to NaN wherever ANY
    # field is missing, so T and S still describe the same water when the
    # vertical bridge later interpolates across that level.
    for a in arrs:
        lv = a.reshape(-1, n_depth).T
        lv[~has_donor, :] = np.nan   # every point on such a level misses SOME field
    n_filled = int(missing[has_donor].sum())
    if n_filled == 0:
        return arrs, 0, 0

    # Find the donor ONCE, as a point index, and gather every field through it.
    # A point index is exact in float64 well past any plausible source grid.
    donor = np.tile(np.arange(n_points, dtype=np.float64), (n_depth, 1))
    donor[missing] = np.nan
    donor = fill_missing_nearest_valid(donor[has_donor], coords)
    donor = np.rint(donor).astype(np.intp)             # (n_donor_levels, n_pts)

    n_lat, n_lon = arr.shape[0], arr.shape[1]
    void = np.zeros_like(donor, dtype=bool)
    # Source OCEAN cells = observed at some depth in every field (each field
    # reduced over depth on its own, then intersected).  The
    # harmonic fill works on this set only: land cells are not part of the
    # stencil, so a void cannot be bridged across an isthmus (Suez, Panama,
    # the archipelago land) the way a plain lat-lon Laplacian would.  A basin
    # the climatology never sampled at ANY depth is indistinguishable from
    # land here and is therefore not a void (it stays nearest-stitched, or is
    # masked with --tripole-closed-seas); a bathymetry-derived source mask
    # would be the next step if such basins must be filled.
    ocean = np.ones(n_points, dtype=bool)
    for a in arrs:
        ocean &= np.isfinite(a.reshape(-1, n_depth)).any(axis=1)
    if void_fill:
        p = np.arange(n_points)
        dj = np.abs(donor // n_lon - (p // n_lon)[None, :])
        di = np.abs(donor % n_lon - (p % n_lon)[None, :])
        di = np.minimum(di, n_lon - di)                 # periodic longitude
        void = missing[has_donor] & ocean[None, :] & (np.maximum(dj, di) > 1)
    n_void = int(void.sum())

    out = []
    for a in arrs:
        levels = a.reshape(-1, n_depth).T.copy()       # (n_depth, n_points)
        levels[has_donor] = np.take_along_axis(levels[has_donor], donor, axis=1)
        if n_void:
            lv = levels[has_donor]
            for k in range(lv.shape[0]):
                if void[k].any():
                    lv[k] = _harmonic_fill_2d(
                        lv[k].reshape(n_lat, n_lon),
                        void[k].reshape(n_lat, n_lon),
                        domain=ocean.reshape(n_lat, n_lon)).ravel()
            levels[has_donor] = lv
        out.append(levels.T.reshape(a.shape))
    return out, n_filled, n_void


# A filled level lighter than the level above it by more than this (in-situ
# density, both evaluated at the lower level's pressure) is a cross-basin
# donor, not water that could sit there.  Measured on PHC3 winter (1 deg, 102
# levels, 3.25e6 filled entries): the deficit has two populations -- up to
# ~0.1 kg/m3 from ordinary column-to-column contrast between a cell and its
# donor (68k entries at 0.01, 16k at 0.1), and >= 0.5 kg/m3 (10k entries,
# 7.7k columns: Baltic 19, Caspian 15, Aegean 8, Arctic shelves) from a
# donor in another basin.  0.1 sits between them.
_FILL_STABILITY_TOL_KG_M3 = 0.1


def _reject_unstable_donors(
    T: np.ndarray, S: np.ndarray, filled: np.ndarray, depth_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Replace a filled (donor) level that is lighter than the level above by
    that level, top to bottom, so the fill never manufactures a static
    inversion.  Observed levels are never changed.

    Why: the per-level nearest donor is the nearest source cell with data AT
    THAT DEPTH, wherever it is.  Where a whole basin has no observation below
    some depth, that cell lies in another basin -- for the Aegean below
    ~1000 m it is the Black Sea (8.9 degC / 22 psu), 150 km away across a
    1-deg land row.  On ORCA12 that put 27 psu water under 39 psu water in
    the North Aegean trench column (an 8 kg/m3 inversion at 1047 m, measured
    through this module's own bilinear + vertical sampling), and the
    velocity there grew from the first step (jobs 27459479 / 27464705).
    Extending the last credible level downward is what the observational
    analyses do for such a basin.  A denser wrong-basin donor still passes:
    this guards the column, not the geography.

    The comparison is against the last FINITE level above, so a source depth
    with no donor anywhere (left NaN for the vertical bridge) does not hide an
    inversion across it.  Returns the corrected fields and the number of
    (cell, level) entries replaced.
    """
    from legoesm.ocean.eos import wright_eos

    T = np.array(T, dtype=np.float64, copy=True)
    S = np.array(S, dtype=np.float64, copy=True)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    n_replaced = 0
    # last finite (T, S) above each column, carried down the walk
    fin0 = np.isfinite(T[..., 0]) & np.isfinite(S[..., 0])
    T_ref = np.where(fin0, T[..., 0], np.nan)
    S_ref = np.where(fin0, S[..., 0], np.nan)
    for k in range(1, T.shape[-1]):
        p_pa = jnp.asarray(constants.rho_ocean * constants.g * depth_m[k])
        Tk = T[..., k]                         # views: masked writes reach T/S
        Sk = S[..., k]
        cand = (filled[..., k] & np.isfinite(Tk) & np.isfinite(Sk)
                & np.isfinite(T_ref) & np.isfinite(S_ref))
        if cand.any():
            rho_here = np.asarray(wright_eos(jnp.asarray(Tk[cand]),
                                             jnp.asarray(Sk[cand]), p_pa))
            rho_above = np.asarray(wright_eos(jnp.asarray(T_ref[cand]),
                                              jnp.asarray(S_ref[cand]), p_pa))
            lighter = np.zeros(cand.shape, dtype=bool)
            lighter[cand] = rho_here < rho_above - _FILL_STABILITY_TOL_KG_M3
            if lighter.any():
                Tk[lighter] = T_ref[lighter]
                Sk[lighter] = S_ref[lighter]
                n_replaced += int(lighter.sum())
        fin = np.isfinite(Tk) & np.isfinite(Sk)
        T_ref[fin] = Tk[fin]
        S_ref[fin] = Sk[fin]
    return T, S, n_replaced


def _harmonic_fill_2d(field: np.ndarray, void: np.ndarray,
                      domain: "np.ndarray | None" = None) -> np.ndarray:
    """Replace ``field`` inside ``void`` by the discrete harmonic function with
    the non-void ``domain`` cells as boundary values: 4-point Laplacian,
    periodic in longitude (axis 1), clamped at the latitude edges; a neighbour
    outside ``domain`` (land) is simply not a neighbour, so the fill never
    crosses it.  Plain grid Laplacian, no spherical metric: the fill is a
    smooth blend, not a physical diffusion.  One sparse direct solve per call.
    A connected void region with no non-void domain neighbour at all (a basin
    missing at this level in every cell) has no boundary to be harmonic with
    respect to; it is set to the mean of the values it arrived with (the
    nearest donors), which removes the wall inside it without inventing a
    gradient."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csg
    import scipy.sparse.linalg as spla

    f = np.array(field, dtype=np.float64)
    n_lat, n_lon = f.shape
    dom = np.ones(f.shape, dtype=bool) if domain is None else np.asarray(domain, dtype=bool)
    void = np.asarray(void, dtype=bool) & dom
    idx = -np.ones(f.shape, dtype=np.int64)
    vj, vi = np.nonzero(void)
    n_void = vj.size
    if n_void == 0:
        return f
    idx[vj, vi] = np.arange(n_void)
    rows, cols, vals = [], [], []
    rhs = np.zeros(n_void)
    deg = np.zeros(n_void)
    n_fixed = np.zeros(n_void)
    for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nj, ni = vj + dj, (vi + di) % n_lon
        ok = (nj >= 0) & (nj < n_lat)
        ok[ok] &= dom[nj[ok], ni[ok]]
        deg[ok] += 1
        nb = idx[nj[ok], ni[ok]]
        r = np.arange(n_void)[ok]
        is_void = nb >= 0
        rows.append(r[is_void]); cols.append(nb[is_void]); vals.append(-np.ones(int(is_void.sum())))
        np.add.at(rhs, r[~is_void], f[nj[ok][~is_void], ni[ok][~is_void]])
        np.add.at(n_fixed, r[~is_void], 1.0)
    adj = sp.csr_matrix((np.ones(sum(len(c) for c in cols)), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(n_void, n_void))
    _, comp = csg.connected_components(adj, directed=False)
    bounded = np.bincount(comp, weights=n_fixed) > 0
    solve = bounded[comp]
    for c in np.nonzero(~bounded)[0]:
        m = comp == c
        f[vj[m], vi[m]] = f[vj[m], vi[m]].mean()
    if solve.any():
        sub = np.cumsum(solve) - 1                      # void index -> solved index
        keep_r = solve[np.concatenate(rows)] & solve[np.concatenate(cols)]
        rr = sub[np.concatenate(rows)[keep_r]]; cc = sub[np.concatenate(cols)[keep_r]]
        vv = np.concatenate(vals)[keep_r]
        n_s = int(solve.sum())
        A = sp.csr_matrix((np.concatenate([vv, deg[solve]]),
                           (np.concatenate([rr, np.arange(n_s)]), np.concatenate([cc, np.arange(n_s)]))),
                          shape=(n_s, n_s))
        x = spla.spsolve(A.tocsc(), rhs[solve])
        if not np.all(np.isfinite(x)):
            raise ValueError("harmonic void fill did not converge to finite values")
        f[vj[solve], vi[solve]] = x
    return f


def _nearest_neighbor_2d(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor interpolation from regular lat-lon to target points.

    Parameters
    ----------
    target_lat, target_lon : array, shape (...)
        Target coordinates in degrees.
    src_lat : array, shape (n_lat,)
    src_lon : array, shape (n_lon,)
    field : array, shape (n_lat, n_lon, ...)
        Source field.

    Returns
    -------
    array, shape (..., remaining_dims)
    """
    # Find nearest lat/lon indices
    lat_idx = np.argmin(np.abs(src_lat[:, np.newaxis] - target_lat.ravel()[np.newaxis, :]),
                        axis=0)
    # Normalize longitude to source range
    tlon = target_lon.ravel() % 360.0
    slon = src_lon % 360.0
    lon_idx = np.argmin(np.abs(slon[:, np.newaxis] - tlon[np.newaxis, :]),
                        axis=0)

    result = field[lat_idx, lon_idx]  # (n_points, ...)
    out_shape = target_lat.shape + field.shape[2:]
    return result.reshape(out_shape)


def _bilinear_2d(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    field: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation from regular lat-lon to target points.

    NaN-aware: if any of the four surrounding source cells is NaN, the
    weighted average is renormalised over the valid cells.  When all
    four are NaN the output is NaN (the caller fills with the deep-
    ocean climatology).  This is the OMIP-2 default horizontal
    interpolation and is required for smooth surface forcing — the
    nearest-neighbour path leaves grid-scale steps that show up as
    spurious gradients in coarse models.

    Parameters
    ----------
    target_lat, target_lon : array, shape (...)
        Target coordinates in degrees.  Longitude is wrapped to
        ``[0, 360)``.
    src_lat : array, shape (n_lat,)
        Source latitudes in degrees, **monotonically increasing**.
    src_lon : array, shape (n_lon,)
        Source longitudes in degrees, **monotonically increasing**,
        assumed periodic with period 360°.
    field : array, shape (n_lat, n_lon, ...)
        Source field.  Trailing dims are passed through.

    Returns
    -------
    array, shape (target_lat.shape + field.shape[2:])
    """
    src_lat = np.asarray(src_lat, dtype=np.float64)
    if not np.all(np.diff(src_lat) > 0):
        raise ValueError("src_lat must be monotonically increasing.")

    # Longitude bookkeeping: accept any source convention
    # (``[0, 360)``, ``[-180, 180)``, ``[-540, -180)``, etc.).  Wrap
    # into ``[0, 360)`` and sort the field along the lon axis so the
    # subsequent searchsorted / bracket logic sees a strictly
    # increasing axis.  The post-mod values are checked to ensure no
    # duplicate longitudes survived (which would indicate aliasing).
    src_lon_raw = np.asarray(src_lon, dtype=np.float64)
    src_lon_mod = src_lon_raw % 360.0
    sort_idx = np.argsort(src_lon_mod, kind="stable")
    src_lon = src_lon_mod[sort_idx]
    field = field[:, sort_idx]
    if not np.all(np.diff(src_lon) > 0):
        # Two source longitudes mapped to the same modulo bucket.
        raise ValueError(
            "src_lon contains duplicate longitudes modulo 360 — "
            "cannot build a strictly increasing axis."
        )

    tlat = np.asarray(target_lat, dtype=np.float64).ravel()
    tlon = np.asarray(target_lon, dtype=np.float64).ravel() % 360.0

    # Clip latitude to source bounds (no extrapolation past poles).
    tlat = np.clip(tlat, src_lat[0], src_lat[-1])

    # Latitude bracketing
    j_hi = np.searchsorted(src_lat, tlat, side="left")
    j_hi = np.clip(j_hi, 1, src_lat.size - 1)
    j_lo = j_hi - 1
    dlat = src_lat[j_hi] - src_lat[j_lo]
    wj = np.where(dlat > 0, (tlat - src_lat[j_lo]) / dlat, 0.0)  # 0..1

    # Longitude bracketing — periodic.  ``i_hi`` may wrap from
    # n_lon - 1 to 0.
    nlon = src_lon.size
    # Insert a virtual point src_lon[-1] - 360 ... src_lon[0] handled
    # by treating any tlon < src_lon[0] as bracketed by (n_lon-1, 0)
    # with the upper point shifted by +360.
    i_lo = np.searchsorted(src_lon, tlon, side="right") - 1
    i_lo = i_lo % nlon
    i_hi = (i_lo + 1) % nlon
    # Wrapped neighbour location: when i_hi wraps to 0, treat src_lon[0]
    # as src_lon[0] + 360 so the bilinear interval stays monotonic.
    src_lon_hi = np.where(
        i_hi == 0, src_lon[0] + 360.0, src_lon[i_hi],
    )
    src_lon_lo = src_lon[i_lo]
    dlon = src_lon_hi - src_lon_lo
    # tlon may be on the wrap interval (tlon < src_lon[0]).  Shift up
    # so the formula uses the same continuous interval.
    tlon_shifted = np.where(tlon < src_lon_lo, tlon + 360.0, tlon)
    wi = np.where(dlon > 0, (tlon_shifted - src_lon_lo) / dlon, 0.0)
    wi = np.clip(wi, 0.0, 1.0)

    # Gather the four corners
    f00 = field[j_lo, i_lo]
    f01 = field[j_lo, i_hi]
    f10 = field[j_hi, i_lo]
    f11 = field[j_hi, i_hi]

    # Broadcast scalar weights to trailing dims
    wj_b = wj.reshape(wj.shape + (1,) * (f00.ndim - 1))
    wi_b = wi.reshape(wi.shape + (1,) * (f00.ndim - 1))
    w00 = (1.0 - wj_b) * (1.0 - wi_b)
    w01 = (1.0 - wj_b) * wi_b
    w10 = wj_b * (1.0 - wi_b)
    w11 = wj_b * wi_b

    # NaN-aware reweighting
    m00 = (~np.isnan(f00)).astype(np.float64)
    m01 = (~np.isnan(f01)).astype(np.float64)
    m10 = (~np.isnan(f10)).astype(np.float64)
    m11 = (~np.isnan(f11)).astype(np.float64)
    w00 = w00 * m00
    w01 = w01 * m01
    w10 = w10 * m10
    w11 = w11 * m11
    wsum = w00 + w01 + w10 + w11
    # Replace NaN inputs with zero for the weighted sum, then divide
    # by the active weight sum.  Where all four are NaN, wsum == 0
    # and we restore NaN so the caller knows to apply the fill.
    f00 = np.where(np.isnan(f00), 0.0, f00)
    f01 = np.where(np.isnan(f01), 0.0, f01)
    f10 = np.where(np.isnan(f10), 0.0, f10)
    f11 = np.where(np.isnan(f11), 0.0, f11)
    num = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
    result = np.where(wsum > 0, num / np.where(wsum > 0, wsum, 1.0), np.nan)

    out_shape = np.asarray(target_lat).shape + field.shape[2:]
    return result.reshape(out_shape)


def _grid_lat_lon_deg(grid):
    """Target-cell (lat_deg, lon_deg) for any ocean grid, shape = spatial layout."""
    # Extract latitude in degrees from the grid.
    # Dispatch order matters: GaussianGrid has both 'lat' and 'lat2d',
    # while CubedSphereGrid has 'lat' (3D) but no 'lat2d'.
    if hasattr(grid, 'lat2d'):
        # Grids exposing a 2-D lat2d (GaussianGrid, LatLonGrid, and the
        # orthogonal-curvilinear tripole LatLonCGridGeometry). Use the grid's
        # TRUE 2-D longitude (lon2d) -- on the tripole grid longitude varies
        # down every i-column, and grid.lon is only the 1-D SOUTHERNMOST row
        # (lon_T[0, :]); pairing it with the 2-D lat_T mis-sampled WOA by a
        # median ~9 deg (up to ~155 deg in the bipolar cap) at every cell. For
        # regular lat-lon / Gaussian grids lon2d is the meshgrid of the 1-D lon,
        # so this is identical there (no regression); fall back to the broadcast
        # 1-D lon only for a grid that lacks lon2d.
        lat_deg = np.asarray(grid.lat2d) * (180.0 / np.pi)
        lon2d = getattr(grid, 'lon2d', None)
        if lon2d is not None:
            lon_deg = np.asarray(lon2d) * (180.0 / np.pi)
        else:
            lon_1d = np.asarray(grid.lon) * (180.0 / np.pi)
            lon_deg = np.broadcast_to(lon_1d[np.newaxis, :], lat_deg.shape)
    elif hasattr(grid, 'latCell'):
        # VoronoiMesh: latCell in radians, shape (nCells,)
        lat_deg = np.asarray(grid.latCell) * (180.0 / np.pi)
        lon_deg = np.asarray(grid.lonCell) * (180.0 / np.pi)
    elif hasattr(grid, 'lat') and hasattr(grid, 'lon'):
        # CubedSphereGrid or LatLonGrid: lat/lon in radians
        lat_deg = np.asarray(grid.lat) * (180.0 / np.pi)
        lon_deg = np.asarray(grid.lon) * (180.0 / np.pi)
    else:
        raise TypeError(f"Unsupported grid type: {type(grid)}")
    return lat_deg, lon_deg


def init_ocean_from_woa(
    grid,
    z_coord: OceanZStarCoordinate,
    T_path: str | Path | None = None,
    S_path: str | Path | None = None,
    *,
    month: int | None = None,
    interp: str = "bilinear",
    bathymetry_depth: np.ndarray | None = None,
    cell_center_depths: np.ndarray | None = None,
    T_fill_C: float | None = None,
    S_fill_psu: float | None = None,
    T_var: str | None = None,
    S_var: str | None = None,
    monthly_layout: str = "concatenated",
    void_fill: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Initialize ocean T and S from WOA18 climatology.

    ``void_fill`` harmonic-fills unobserved basins/channels on the source grid
    instead of stitching them from the nearest observed cells (see
    :func:`_fill_source_levels_nearest_valid`).

    If ``T_path`` and ``S_path`` are provided, loads WOA18 NetCDF files
    and interpolates to the model grid.  Otherwise, uses analytical
    latitude-dependent profiles mimicking WOA18.

    Parameters
    ----------
    grid : CubedSphereGrid, LatLonGrid, VoronoiMesh, or GaussianGrid
    z_coord : OceanZStarCoordinate
    T_path, S_path : path or None
        Paths to WOA18 NetCDF annual or monthly climatology files.
    month : int or None
        Month 1..12 for monthly climatology; ``None`` (default) uses
        the annual mean.  Requires the input files to be monthly
        products (time axis of length 12).
    interp : {"bilinear", "nearest"}
        Horizontal interpolation method.  ``"bilinear"`` is the
        OMIP-2 default and is required for smooth surface forcing on
        coarse grids; ``"nearest"`` preserves legacy behaviour and is
        useful for debugging column-by-column.
    bathymetry_depth : array or None
        Positive water-column depth [m] at each horizontal target
        point, broadcastable to the grid's spatial shape.  When
        supplied, every model level below the local bathymetry is
        replaced with the fill value (``T_fill_C`` / ``S_fill_psu``)
        instead of carrying surface-extended WOA data into the
        seafloor — this avoids spurious density inversions when the
        model's bottom cell sits below the WOA-resolved water column.
    T_fill_C : float or None
        Replacement value [°C] for ``NaN`` profile entries and for
        levels below ``bathymetry_depth``.  Defaults to
        ``constants.T_deep_ocean_ref_C`` (1.5°C — WOA18 abyssal mean).
    S_fill_psu : float or None
        Replacement value [PSU] for ``NaN`` / below-bathymetry levels.
        Defaults to ``constants.S_deep_ocean_ref_psu`` (34.7 PSU).
    T_var, S_var : str or None
        Override the WOA dataset variable names.  Pass-through to
        :func:`load_woa18`.  Defaults: ``"t_an"`` / ``"s_an"``.  Use
        when the input dataset uses a non-standard name (e.g.
        ``"t_mn"`` for the monthly-mean object in newer WOA bundles).
    monthly_layout : {"concatenated", "single_file_per_month"}
        Pass-through to :func:`load_woa18`.  Defaults to
        ``"concatenated"`` (one 12-month file).  Use
        ``"single_file_per_month"`` when the caller has pre-selected
        a length-1 monthly file for the desired ``month`` — the
        loader then reads index 0 with ``month=None`` internally,
        avoiding silent month mismatch.

    Returns
    -------
    T : jax.Array, shape (..., nlev) — potential temperature [degC]
    S : jax.Array, shape (..., nlev) — salinity [PSU]
    """
    if interp not in ("bilinear", "nearest"):
        raise ValueError(
            f"interp must be 'bilinear' or 'nearest'; got {interp!r}."
        )
    if monthly_layout not in ("concatenated", "single_file_per_month"):
        raise ValueError(
            f"monthly_layout must be 'concatenated' or "
            f"'single_file_per_month'; got {monthly_layout!r}."
        )
    if monthly_layout == "single_file_per_month" and month is not None:
        # Same contract as load_woa18: the caller must pre-select the
        # file matching the desired month and pass month=None.
        # Silently dropping ``month`` here would mask file/month
        # mismatch and return the wrong climatology.
        raise ValueError(
            "init_ocean_from_woa(monthly_layout='single_file_per_month') "
            "requires month=None — pass the per-month file already "
            "selected for the target month.  Got month=" f"{month}."
        )
    T_fill = (
        float(constants.T_deep_ocean_ref_C)
        if T_fill_C is None else float(T_fill_C)
    )
    S_fill = (
        float(constants.S_deep_ocean_ref_psu)
        if S_fill_psu is None else float(S_fill_psu)
    )
    if not np.isfinite(T_fill):
        raise ValueError(
            f"T_fill_C must be finite; got {T_fill}."
        )
    if not np.isfinite(S_fill):
        raise ValueError(
            f"S_fill_psu must be finite; got {S_fill}."
        )

    lat_deg, lon_deg = _grid_lat_lon_deg(grid)

    if T_path is not None and S_path is not None:
        # Load from WOA18 NetCDF files.  The guard above ensures that
        # ``monthly_layout='single_file_per_month'`` implies
        # ``month is None``, so this single call covers both layouts.
        T_woa, S_woa, lat_woa, lon_woa, depth_woa = load_woa18(
            T_path, S_path, month=month,
            T_var=T_var, S_var=S_var,
            monthly_layout=monthly_layout,
        )

        # Per-LEVEL nearest-valid fill on the SOURCE grid, before any
        # horizontal interpolation: a source cell missing an observation at a
        # given depth takes it from the nearest source cell that has one at
        # that same depth.  Without this a model column over a shallow source
        # cell inherited that cell's shallowest value all the way down.
        _filled = ~(np.isfinite(T_woa) & np.isfinite(S_woa))
        (T_woa, S_woa), n_fill, n_void = _fill_source_levels_nearest_valid(
            (T_woa, S_woa), lat_woa, lon_woa, void_fill=void_fill)
        _src_depths = (np.asarray(depth_woa, dtype=np.float64) if depth_woa is not None
                       else WOA_DEPTHS[:T_woa.shape[-1]])
        T_woa, S_woa, n_unstable = _reject_unstable_donors(
            T_woa, S_woa, _filled & np.isfinite(T_woa), _src_depths)
        del _filled
        if n_unstable:
            print(f"[setup] observed T/S: {n_unstable} filled source cell-levels "
                  "were lighter than the level above (a cross-basin donor) and "
                  "were replaced by the level above")
        if n_fill:
            n_src = T_woa.shape[0] * T_woa.shape[1] * T_woa.shape[2]
            print(f"[setup] observed T/S: filled {n_fill} of {n_src} source "
                  "cell-levels from the nearest source column holding an "
                  "observation at the same depth (one donor for T and S); "
                  f"{n_void} of them harmonic-filled as data void "
                  f"(void_fill={void_fill})")

        if interp == "bilinear":
            T_woa_horiz = _bilinear_2d(
                lat_deg, lon_deg, lat_woa, lon_woa, T_woa,
            )
            S_woa_horiz = _bilinear_2d(
                lat_deg, lon_deg, lat_woa, lon_woa, S_woa,
            )
        else:
            T_woa_horiz = _nearest_neighbor_2d(
                lat_deg, lon_deg, lat_woa, lon_woa, T_woa,
            )
            S_woa_horiz = _nearest_neighbor_2d(
                lat_deg, lon_deg, lat_woa, lon_woa, S_woa,
            )

        # Vertical interpolation to model levels (vectorized over columns)
        flat_shape = (-1, T_woa_horiz.shape[-1])
        T_flat = T_woa_horiz.reshape(flat_shape)
        S_flat = S_woa_horiz.reshape(flat_shape)

        # Source depth axis: the FILE's own when it has one, otherwise the
        # WOA18 standard levels.  Reading it from the file is what lets a
        # non-WOA climatology (PHC3's 33 levels, WOCE's 75) be used directly;
        # the ``WOA_DEPTHS[:n]`` fallback silently mis-levels such a file, so
        # it is only reached when the file genuinely has no depth coordinate.
        if depth_woa is not None:
            woa_depths = np.asarray(depth_woa, dtype=np.float64)
            if not np.all(np.diff(woa_depths) > 0):
                raise ValueError(
                    "WOA/climatology depth coordinate is not strictly "
                    f"ascending after abs(); got {woa_depths[:5]}..."
                    f"{woa_depths[-3:]}. np.interp requires monotonic source "
                    "depths."
                )
        else:
            woa_depths = WOA_DEPTHS[:T_woa.shape[-1]]

        # Per-column sampling depths.  A partial bottom cell's centre sits
        # ABOVE the reference full-cell centre -- by a measured median 22 m and
        # up to 112 m at ico8/ETOPO -- so sampling the observation at the
        # reference depth pulls water from the wrong level exactly where cells
        # are cut, i.e. along the shelf break.
        _cd = None
        from legoesm.ocean.vertical import OceanPartialCellCoordinate
        if cell_center_depths is None and isinstance(
                z_coord, OceanPartialCellCoordinate):
            warnings.warn(
                "init_ocean_from_woa: partial-cell coordinate WITHOUT "
                "cell_center_depths -- the observed profiles are being sampled "
                "at the REFERENCE full-cell centres, which puts the wrong water "
                "in every cut bottom cell (measured: up to 112 m of offset, a "
                "first-step acceleration of 4.5e-3 m/s^2 at ico8/ETOPO). Pass "
                "compute_centroid_depth(eta, H_bathy, z_coord).",
                RuntimeWarning, stacklevel=2)
        if cell_center_depths is not None:
            _cd = np.abs(np.asarray(cell_center_depths, dtype=np.float64))
            if _cd.shape[-1] != z_coord.n_levels:
                raise ValueError(
                    "cell_center_depths must have the model's level count "
                    f"({z_coord.n_levels}); got shape {_cd.shape}")
            _cd = _cd.reshape(-1, z_coord.n_levels)
            if _cd.shape[0] != T_flat.shape[0]:
                raise ValueError(
                    f"cell_center_depths has {_cd.shape[0]} columns but the "
                    f"grid has {T_flat.shape[0]}")

        T_out = np.stack([
            _interp_profile_to_z_coord(T_flat[i], woa_depths, z_coord,
                                       None if _cd is None else _cd[i])
            for i in range(T_flat.shape[0])
        ]).reshape(lat_deg.shape + (z_coord.n_levels,))

        S_out = np.stack([
            _interp_profile_to_z_coord(S_flat[i], woa_depths, z_coord,
                                       None if _cd is None else _cd[i])
            for i in range(S_flat.shape[0])
        ]).reshape(lat_deg.shape + (z_coord.n_levels,))

        # Bathymetry-aware mask: blank out levels below the local
        # water-column depth so the fill (rather than the
        # surface-extended profile) drives the deep cells.  Model
        # full-level depths are ``|z_full_ref|`` in metres.
        if bathymetry_depth is not None:
            bath_arr = np.asarray(bathymetry_depth, dtype=np.float64)
            if not np.all(np.isfinite(bath_arr)):
                raise ValueError(
                    "bathymetry_depth contains non-finite values; "
                    "expected positive water-column depth in metres."
                )
            if np.any(bath_arr <= 0.0):
                raise ValueError(
                    "bathymetry_depth must be strictly positive "
                    "(land/dry cells should be excluded by the caller "
                    "or set to a small wet-cell depth); got values "
                    f"with min={float(bath_arr.min())}."
                )
            # Partial cells: mask on the TRUE centre depths the profile was
            # sampled at, not the reference centres, or a cut bottom cell whose
            # reference centre lies below the floor loses its valid sample.
            model_depths = (np.abs(np.asarray(z_coord.z_full_ref)) if _cd is None
                            else _cd.reshape(T_out.shape))
            nlev = T_out.shape[-1]
            # Only 1-D z_full_ref of length nlev is supported here.
            # Shaped model depths (e.g. spatially varying interfaces
            # under z-tilde) need a separate code path and an
            # explicit shape match.
            if model_depths.shape == (nlev,):
                depth_grid = np.broadcast_to(
                    model_depths.reshape((1,) * (T_out.ndim - 1) + (-1,)),
                    T_out.shape,
                )
            elif model_depths.shape == T_out.shape:
                depth_grid = model_depths
            else:
                raise ValueError(
                    f"z_coord.z_full_ref has shape {model_depths.shape}; "
                    f"expected ({nlev},) for a column-uniform z* "
                    f"coordinate, or {T_out.shape} for a spatially-"
                    f"varying coordinate."
                )
            bath = np.broadcast_to(bath_arr[..., None], T_out.shape)
            below_floor = depth_grid > bath
            T_out = np.where(below_floor, np.nan, T_out)
            S_out = np.where(below_floor, np.nan, S_out)

        # Fill remaining NaNs (whole-column failures + below-floor mask)
        T_out = np.where(np.isnan(T_out), T_fill, T_out)
        S_out = np.where(np.isnan(S_out), S_fill, S_out)

    else:
        # Analytical fallback
        T_out, S_out = _analytical_woa_profiles(lat_deg, z_coord)

    return jnp.array(T_out), jnp.array(S_out)


def woa_ocean_mask(
    grid,
    T_path: str | Path,
    *,
    T_var: str | None = None,
    monthly_layout: str = "concatenated",
    ocean_fraction_threshold: float = 0.5,
) -> np.ndarray:
    """Derive a realistic ``(1=ocean, 0=land)`` wet mask from WOA18 surface T.

    A grid cell is OCEAN where the area-weighted bilinear regrid of the WOA18
    surface-temperature *valid-data fraction* (1 over ocean, 0 over WOA's land
    fill) reaches ``ocean_fraction_threshold``, and LAND otherwise.  Using the
    regridded fraction (rather than the NaN-aware T itself) gives a crisp,
    threshold-controlled coastline on coarse target grids instead of flooding
    every coastal cell to ocean.

    This is the single source of truth for the realistic coupled run's wet
    domain: the same mask initialises the ocean ``land_mask`` AND the
    atmosphere ``f_land`` (``= 1 - ocean_mask``) on the shared lat-lon grid, so
    the surface/ocean wet masks agree EXACTLY — no atmosphere flux leaks onto
    an ocean-masked cell (codex Phase-1 HIGH).  Only the lat-lon grid is
    supported because the prognostic 3D ocean is lat-lon-only.

    Parameters
    ----------
    grid : LatLonGrid
        Must expose ``lat2d`` / ``lon2d`` (radians).
    T_path : path
        WOA18 temperature NetCDF/Zarr (the ``t_an`` annual field, or a
        monthly product — only the surface level, time index 0, is read).
    T_var : str or None
        Override the temperature variable name (default ``"t_an"``).
    monthly_layout : str
        Accepted for signature parity with :func:`load_woa18`; the surface
        mask only ever reads time index 0, so the layout does not change the
        result.
    ocean_fraction_threshold : float, default 0.5
        Regridded ocean-fraction cutoff for classifying a cell as ocean.

    Returns
    -------
    ocean_mask : np.ndarray, shape (n_lat, n_lon), float64
        ``1.0`` over ocean, ``0.0`` over land.  Caller casts to the active
        precision policy (e.g. via ``rest_state_latlon_cgrid_ocean``).

    Notes
    -----
    Marginal/enclosed seas present in WOA (Caspian, Black, etc.) are classified
    as ocean here; for a coupled run that has previously destabilised the
    barotropic solver (see the OMIP ``--mask-marginal-seas`` note), so a
    follow-up may want to prune them.  Left in for a first realistic run.
    """
    if monthly_layout not in ("concatenated", "single_file_per_month"):
        raise ValueError(
            f"monthly_layout must be 'concatenated' or "
            f"'single_file_per_month'; got {monthly_layout!r}."
        )
    if not (0.0 < ocean_fraction_threshold <= 1.0):
        raise ValueError(
            "ocean_fraction_threshold must be in (0, 1]; got "
            f"{ocean_fraction_threshold}."
        )
    lat2d = getattr(grid, "lat2d", None)
    lon2d = getattr(grid, "lon2d", None)
    if lat2d is None or lon2d is None:
        raise TypeError(
            "woa_ocean_mask requires a lat-lon grid exposing lat2d/lon2d "
            f"(radians); got {type(grid).__name__}."
        )
    lat_deg = np.asarray(lat2d) * (180.0 / np.pi)
    lon_deg = np.asarray(lon2d) * (180.0 / np.pi)

    t_name = T_var if T_var is not None else "t_an"
    ds = _open_woa_dataset(T_path)
    if t_name not in ds.data_vars:
        avail = list(ds.data_vars)
        ds.close()
        raise ValueError(
            f"WOA T file {str(T_path)!r} has no variable {t_name!r}; "
            f"available data vars: {avail}.  Pass the correct T_var."
        )
    # Select the surface level of the first time slice by DIMENSION NAME
    # (robust to transposed / preprocessed files): time->0, the depth axis
    # (the dim that is not lat/lon/time)->0, then order as (lat, lon).
    da = ds[t_name]
    isel = {}
    if "time" in da.dims:
        isel["time"] = 0
    depth_dim = next(
        (d for d in da.dims if d not in ("lat", "lon", "time")), None)
    if depth_dim is not None:
        isel[depth_dim] = 0
    da_surf = da.isel(**isel)
    if {"lat", "lon"}.issubset(da_surf.dims):
        da_surf = da_surf.transpose("lat", "lon")
    T_surf = np.asarray(da_surf.values)  # (n_lat_woa, n_lon_woa)
    lat_woa = np.array(ds["lat"].values)
    lon_woa = np.array(ds["lon"].values)
    ds.close()
    if T_surf.ndim != 2:
        raise ValueError(
            f"WOA surface T from {str(T_path)!r} is {T_surf.ndim}-D after "
            f"selecting time/depth (var dims {da.dims}); expected 2-D "
            "(lat, lon).")

    # Ocean-fraction source field: 1 where WOA has valid surface T (ocean),
    # 0 where NaN (land fill).  Regrid with the (NaN-free) bilinear operator
    # and threshold.  Trailing singleton dim so _bilinear_2d's trailing-dim
    # broadcasting applies.
    ocean_src = (~np.isnan(T_surf)).astype(np.float64)[..., None]
    frac = _bilinear_2d(lat_deg, lon_deg, lat_woa, lon_woa, ocean_src)[..., 0]
    # Cells with no valid bracketing source (frac is NaN) are land.
    frac = np.where(np.isnan(frac), 0.0, frac)
    ocean_mask = (frac >= ocean_fraction_threshold).astype(np.float64)
    return ocean_mask


def woa_ocean_bathymetry(
    grid,
    T_path: str | Path,
    *,
    H_max: float = 5500.0,
    min_depth_m: float = 200.0,
    T_var: str | None = None,
    ocean_fraction_threshold: float = 0.5,
) -> np.ndarray:
    """Derive a realistic per-column bathymetry [m] from WOA18 temperature.

    For each model column the depth is the DEEPEST WOA standard level that is
    still ocean (valid T) after the NaN-aware bilinear regrid of the per-level
    valid-data fraction, clamped to ``[min_depth_m, H_max]``.  Land columns
    (surface fraction below threshold) get ``0``.

    A realistic (varying) bathymetry — instead of a flat ``H_max`` everywhere —
    is REQUIRED for a stable WOA cold start: the flat bottom extends the WOA
    abyssal-fill density into a spurious deep column and (with the plain z-star
    coord) gates off the smc03 partial-cell pressure-gradient correction.  Feed
    the result through :func:`make_partial_cell_latlon` (smoothing + thin-cell
    snap) before building the ocean coordinate.

    Parameters
    ----------
    grid : LatLonGrid (lat2d/lon2d in radians).
    T_path : WOA18 temperature file.
    H_max : maximum depth clamp [m].
    min_depth_m : minimum wet-column depth [m] (shallow shelves are floored
        here; ``make_partial_cell_latlon(min_levels=...)`` later masks columns
        with too few active levels).
    T_var : override temperature variable name (default ``"t_an"``).
    ocean_fraction_threshold : ocean cutoff for the per-level regrid.

    Returns
    -------
    H_bathy : np.ndarray (n_lat, n_lon), float64 — depth [m], 0 over land.
    """
    if not (0.0 < ocean_fraction_threshold <= 1.0):
        raise ValueError(
            f"ocean_fraction_threshold must be in (0, 1]; got "
            f"{ocean_fraction_threshold}.")
    if not (0.0 < min_depth_m < H_max):
        raise ValueError(
            f"min_depth_m must be in (0, H_max={H_max}); got {min_depth_m}.")
    lat2d = getattr(grid, "lat2d", None)
    lon2d = getattr(grid, "lon2d", None)
    if lat2d is None or lon2d is None:
        raise TypeError(
            "woa_ocean_bathymetry requires a lat-lon grid exposing lat2d/lon2d.")
    lat_deg = np.asarray(lat2d) * (180.0 / np.pi)
    lon_deg = np.asarray(lon2d) * (180.0 / np.pi)

    t_name = T_var if T_var is not None else "t_an"
    ds = _open_woa_dataset(T_path)
    if t_name not in ds.data_vars:
        avail = list(ds.data_vars)
        ds.close()
        raise ValueError(
            f"WOA T file {str(T_path)!r} has no variable {t_name!r}; "
            f"available: {avail}.")
    da = ds[t_name]
    isel = {}
    if "time" in da.dims:
        isel["time"] = 0
    depth_dim = next(
        (d for d in da.dims if d not in ("lat", "lon", "time")), None)
    if depth_dim is None:
        ds.close()
        raise ValueError(
            f"WOA T variable has no depth dim (dims {da.dims}).")
    da3 = da.isel(**isel).transpose(depth_dim, "lat", "lon")
    T3 = np.asarray(da3.values)                       # (n_depth, lat, lon)
    lat_woa = np.array(ds["lat"].values)
    lon_woa = np.array(ds["lon"].values)
    ds.close()

    n_depth = T3.shape[0]
    depths = WOA_DEPTHS[:n_depth]
    # Ocean fraction per WOA level, regridded to the model grid.
    valid = (~np.isnan(T3)).astype(np.float64)        # (depth, lat, lon)
    valid_lld = np.moveaxis(valid, 0, -1)             # (lat, lon, depth)
    fr = _bilinear_2d(lat_deg, lon_deg, lat_woa, lon_woa, valid_lld)
    fr = np.where(np.isnan(fr), 0.0, fr)
    ocean_at_depth = fr >= ocean_fraction_threshold   # (n_lat, n_lon, n_depth)
    # Deepest ocean depth per column (0 if no ocean at any level).
    H = np.where(ocean_at_depth, depths[None, None, :], 0.0).max(axis=-1)
    wet = H > 0.0
    H = np.where(wet, np.clip(H, min_depth_m, H_max), 0.0)
    return H.astype(np.float64)


def _fesom_profiles_per_layer(xyz_nodes, nlevels, T_ic, S_ic, xyz_targets, k=4, chunk=200_000):
    """T_prof, S_prof (ntarget, nz) on the FESOM layer mid-depths; for layer kk the donors are the nodes wet at kk (nlevels-2 >= kk)."""
    from scipy.spatial import cKDTree

    xyz_nodes = np.ascontiguousarray(xyz_nodes, dtype=np.float64)
    xyz_targets = np.ascontiguousarray(xyz_targets, dtype=np.float64)
    nlevels = np.asarray(nlevels)
    T_ic = np.asarray(T_ic, dtype=np.float64)
    S_ic = np.asarray(S_ic, dtype=np.float64)

    nz = T_ic.shape[1] - 1
    nt = xyz_targets.shape[0]

    T_prof = np.empty((nt, nz), dtype=np.float64)
    S_prof = np.empty((nt, nz), dtype=np.float64)
    d0 = np.empty(nt, dtype=np.float64)
    i0 = np.empty(nt, dtype=np.int64)

    # Build one KD-tree per layer, up front; identical donor sets share a single tree.
    # Donor sets are nested (threshold kk grows), so (size, first, last) uniquely identifies a set.
    nlim = nlevels - 2
    donors_l = [None] * nz
    trees = [None] * nz
    memo = {}
    for kk in range(nz):
        donors = np.flatnonzero(nlim >= kk)
        if donors.size == 0:
            # no node is wet at this layer: the profile holds the layer above
            # (a shallower donor set must never be reused here -- its padded
            # 0.0 values would enter the average)
            donors_l[kk] = None
            trees[kk] = None
            continue
        key = (int(donors.size), int(donors[0]), int(donors[-1]))
        hit = memo.get(key)
        if hit is None:
            hit = (donors, cKDTree(xyz_nodes[donors]))
            memo[key] = hit
        donors_l[kk], trees[kk] = hit

    for s in range(0, nt, chunk):
        e = min(s + chunk, nt)
        pts = xyz_targets[s:e]
        m = e - s
        for kk in range(nz):
            donors = donors_l[kk]
            tree = trees[kk]
            if donors is None:
                T_prof[s:e, kk] = T_prof[s:e, kk - 1]
                S_prof[s:e, kk] = S_prof[s:e, kk - 1]
                continue
            d, j = tree.query(pts, k=min(k, donors.size))
            d = d.reshape(m, -1)
            j = j.reshape(m, -1)
            if kk == 0:
                d0[s:e] = d[:, 0]
                i0[s:e] = donors[j[:, 0]]
            zero = d == 0.0
            w = 1.0 / np.maximum(d, 1e-12) ** 2
            zr = zero.any(axis=1)
            if zr.any():
                w[zr] = zero[zr]
            w /= w.sum(axis=1, keepdims=True)
            idx = donors[j]
            T_prof[s:e, kk] = np.einsum('ij,ij->i', w, T_ic[idx, kk])
            S_prof[s:e, kk] = np.einsum('ij,ij->i', w, S_ic[idx, kk])

    return T_prof, S_prof, d0, i0, donors_l, trees


_FESOM_IC_CACHE_VERSION = 1  # bump when the remap algorithm changes values
_FESOM_IC_CACHE_POLL_S = 5.0  # poll interval (s) for non-zero ranks waiting on rank 0's cache file


def _load_fesom_ic_cache(path, spatial, nlev, log):
    """Load a FESOM IC cache entry published by rank 0.

    Returns ``(T, S)`` as float64 jax arrays, or ``None`` if the file is
    corrupt/truncated, missing arrays, or shaped for a different grid.  Never
    raises: the reason is logged so rank 0 can rebuild and other ranks can
    fail with a clear message.
    """
    expected = tuple(spatial) + (nlev,)
    try:
        with np.load(path) as blob:
            T = np.asarray(blob["T"], dtype=np.float64)
            S = np.asarray(blob["S"], dtype=np.float64)
    except Exception as exc:  # missing arrays, bad zip, truncated file, ...
        log(f"init_ocean_from_fesom_mesh: unusable cache {path}: {exc!r}")
        return None
    if T.shape != expected or S.shape != expected:
        log(f"init_ocean_from_fesom_mesh: unusable cache {path}: "
            f"T.shape={T.shape}, S.shape={S.shape}, expected {expected}")
        return None
    log(f"init_ocean_from_fesom_mesh: cache hit {path}")
    return (jnp.asarray(T, dtype=jnp.float64), jnp.asarray(S, dtype=jnp.float64))


def init_ocean_from_fesom_mesh(grid, z_coord, mesh_dir, *, cell_center_depths=None,
                               k=4, isolated_factor=3.0, wet_mask=None, log=print,
                               cache_dir=None, cache_wait_timeout_s=6 * 3600):
    """Build initial potential temperature and salinity on the model grid from
    a FESOM2-JAX mesh's cached initial field.

    Horizontal: for each FESOM LAYER, inverse-distance-squared weighting of the
    ``k`` nearest nodes that are WET at that layer (chord distances on the unit
    sphere; a coincident node gets weight 1).  A target deeper than its
    neighbouring nodes' bottoms therefore takes the nearest nodes that ARE that
    deep -- real water of that depth -- never a shallow profile extended
    downward (that extension wrote 15 degC water to 2800 m in the Bismarck Sea
    next to cells fed by deep nodes, a 2.4 kg/m3 wall).  The log line reports
    the median and 99th-percentile donor distance at the deepest layer any wet
    target uses, so distant-donor filling is visible.
    Vertical: linear interpolation in depth from the 69 layer mid-depths |Z| to
    each target level's depth (``cell_center_depths`` when given, else the
    reference |z_full_ref|); the top value is held above |Z[0]| and the deepest
    layer value below |Z[-1]|.

    Units: T_ic/S_ic potential temperature [degC] / salinity [psu] (nod2D, 70),
    padded with 0.0 below each node's bottom (layers 0..nlevels_nod2D-2 are
    wet); Z [m] negative; target depths [m] positive-down; distances on the
    unit sphere, converted with ``constants.R_earth`` for the diagnostics.

    Closed seas: every target cell is filled; masking of closed seas (and land)
    is the caller's job.  ``wet_mask`` (spatial, True = ocean) only restricts
    the isolated-cell report (nearest surface donor farther than
    ``isolated_factor`` x that donor's mesh resolution) to ocean cells.

    ``cache_dir`` (default ``$LEGOESM_MESH_CACHE_DIR``; must be an absolute
    path on storage shared by every process): rank 0 builds once and publishes
    an ``.npz`` keyed on the mesh files' identity and the target geometry; the
    other ranks wait for it and load the same bytes.  A cache hit returns the
    stored fields without rebuilding, so the isolated-cell and donor-distance
    diagnostics are skipped; their log lines appear only on a build.

    Returns ``(T, S)`` as float64 jax arrays of shape ``spatial + (nlev,)``.
    """
    import hashlib
    import os
    import time

    import jax

    if cache_dir is None:
        # same env-var convention as legoesm/grids/voronoi.py
        cache_dir = os.environ.get("LEGOESM_MESH_CACHE_DIR") or None
    nlev = int(z_coord.n_levels)
    lat_deg, lon_deg = _grid_lat_lon_deg(grid)
    lat_flat = np.asarray(lat_deg, dtype=np.float64).ravel()
    lon_flat = np.asarray(lon_deg, dtype=np.float64).ravel()
    spatial = np.shape(lat_deg)
    ncell = lat_flat.size
    if cell_center_depths is not None:
        ccd = np.asarray(cell_center_depths, dtype=np.float64)
        if ccd.shape[-1] != nlev or ccd.shape[:-1] != spatial:
            raise ValueError(f"cell_center_depths shape {ccd.shape} != {spatial + (nlev,)}")
        depths = ccd.reshape(ncell, nlev)
    else:
        depths = np.broadcast_to(
            np.abs(np.asarray(z_coord.z_full_ref, dtype=np.float64)), (ncell, nlev))

    cache_path = None
    if cache_dir is not None:
        # cache_dir is written by rank 0 and polled by every other rank, so it
        # MUST be an absolute path on storage shared by all jax processes.
        if not os.path.isabs(cache_dir):
            raise ValueError(
                "init_ocean_from_fesom_mesh: cache_dir must be an absolute path on a "
                f"filesystem shared by all jax processes, got {cache_dir!r}")
        key = hashlib.sha256()
        # Key the mesh files by identity (basename, size, mtime_ns) instead of
        # content -- the staleness criterion make/rsync use -- so no rank hashes
        # ~16 GB of mesh at every launch.  An in-place edit that preserves size
        # and mtime_ns is NOT detected; bump _FESOM_IC_CACHE_VERSION for that.
        for name in ("T_ic.npy", "S_ic.npy", "geo_coord_nod2D.npy", "Z.npy", "nlevels_nod2D.npy"):
            st = os.stat(os.path.join(mesh_dir, name))
            key.update(f"{name}|{st.st_size}|{st.st_mtime_ns}\n".encode())
        key.update(repr(tuple(spatial)).encode())
        key.update(str(nlev).encode())
        key.update(hashlib.sha256(np.ascontiguousarray(lat_flat)).hexdigest().encode())
        key.update(hashlib.sha256(np.ascontiguousarray(lon_flat)).hexdigest().encode())
        if cell_center_depths is None:
            # depths is a broadcast of one nlev vector; hash just that vector
            # (materialising the view is ~8 GB at ORCA12).
            key.update(b"ref")
            key.update(hashlib.sha256(
                np.abs(np.asarray(z_coord.z_full_ref, dtype=np.float64)).tobytes()).hexdigest().encode())
        else:
            key.update(b"ccd")
            # buffer protocol, no .tobytes() copy of the (ncell, nlev) array (8 GB at ORCA12)
            key.update(hashlib.sha256(np.ascontiguousarray(ccd)).hexdigest().encode())
        key.update(f"{k!r}|{isolated_factor!r}|{_FESOM_IC_CACHE_VERSION}".encode())
        cache_path = os.path.join(
            cache_dir, f"fesom_ic_v{_FESOM_IC_CACHE_VERSION}_{key.hexdigest()[:24]}.npz")
    if cache_path is not None and os.path.exists(cache_path):
        cached = _load_fesom_ic_cache(cache_path, spatial, nlev, log)
        if cached is not None:
            return cached
        if jax.process_index() != 0:
            raise RuntimeError(
                f"init_ocean_from_fesom_mesh: cache file {cache_path} is corrupt or "
                f"incompatible (expected T/S of shape {tuple(spatial) + (nlev,)})")
        log(f"init_ocean_from_fesom_mesh: unusable cache, rebuilding {cache_path}")
        # Remove the unusable file now, so peers arriving during the (long)
        # rebuild see no file and enter the wait loop instead of raising.
        try:
            os.unlink(cache_path)
        except FileNotFoundError:
            pass
    if cache_path is not None and jax.process_index() != 0:
        # Never build on a non-zero rank: a per-rank build is 16x the cost, and
        # every rank must load the very same bytes (byte-mismatch class defect
        # seen before).  Wait for process 0 to publish the file, then load it.
        waited = 0.0
        last_log = 0.0
        while not os.path.exists(cache_path):
            if waited >= cache_wait_timeout_s:
                raise RuntimeError(
                    f"init_ocean_from_fesom_mesh: timed out after {waited:.0f}s waiting for "
                    f"{cache_path} from process 0")
            time.sleep(_FESOM_IC_CACHE_POLL_S)
            waited += _FESOM_IC_CACHE_POLL_S
            if waited - last_log >= 600.0:
                log(f"init_ocean_from_fesom_mesh: still waiting for {cache_path} ({waited:.0f}s)")
                last_log = waited
        cached = _load_fesom_ic_cache(cache_path, spatial, nlev, log)
        if cached is None:
            raise RuntimeError(
                f"init_ocean_from_fesom_mesh: cache file {cache_path} is corrupt or "
                f"incompatible (expected T/S of shape {tuple(spatial) + (nlev,)})")
        return cached

    T_ic = np.asarray(np.load(os.path.join(mesh_dir, "T_ic.npy")), dtype=np.float64)
    S_ic = np.asarray(np.load(os.path.join(mesh_dir, "S_ic.npy")), dtype=np.float64)
    geo = np.asarray(np.load(os.path.join(mesh_dir, "geo_coord_nod2D.npy")), dtype=np.float64)
    Z = np.asarray(np.load(os.path.join(mesh_dir, "Z.npy")), dtype=np.float64)
    nlevels = np.asarray(np.load(os.path.join(mesh_dir, "nlevels_nod2D.npy")), dtype=np.int64)
    res_path = os.path.join(mesh_dir, "mesh_resolution.npy")
    mesh_res = (np.asarray(np.load(res_path), dtype=np.float64)
                if os.path.exists(res_path) else None)

    if T_ic.ndim != 2 or T_ic.shape != S_ic.shape:
        raise ValueError(f"T_ic/S_ic shape mismatch: {T_ic.shape} vs {S_ic.shape}")
    nod2, nz_full = T_ic.shape
    if geo.shape != (nod2, 2):
        raise ValueError(f"geo_coord_nod2D shape {geo.shape} != {(nod2, 2)}")
    if nlevels.shape != (nod2,):
        raise ValueError(f"nlevels_nod2D shape {nlevels.shape} != {(nod2,)}")
    if Z.ndim != 1 or Z.shape[0] != nz_full - 1:
        raise ValueError(f"Z shape {Z.shape} incompatible with {nz_full} profile layers")
    if np.any(nlevels < 2) or np.any(nlevels > nz_full):
        raise ValueError("nlevels_nod2D out of range [2, n_layers]")
    if mesh_res is not None and mesh_res.shape != (nod2,):
        raise ValueError(f"mesh_resolution shape {mesh_res.shape} != {(nod2,)}")
    z_abs = np.abs(Z)
    if np.any(np.diff(z_abs) <= 0.0):
        raise ValueError("|Z| must be strictly increasing with depth")
    nz = z_abs.size

    def _xyz(lat_r, lon_r):
        return np.stack((np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r),
                         np.sin(lat_r)), axis=1)
    xyz_n = _xyz(geo[:, 1], geo[:, 0])
    xyz_t = _xyz(np.deg2rad(lat_flat), np.deg2rad(lon_flat))

    T_prof, S_prof, d0, i0, donors_l, trees = _fesom_profiles_per_layer(
        xyz_n, nlevels, T_ic, S_ic, xyz_t, k=k)

    # vertical: |Z| -> target depths, top held above z_abs[0], bottom held below z_abs[-1]
    T_out = np.empty((ncell, nlev), dtype=np.float64)
    S_out = np.empty((ncell, nlev), dtype=np.float64)
    chunk = 200_000
    for s in range(0, ncell, chunk):
        e = min(s + chunk, ncell)
        D = depths[s:e]                                             # (m, nlev)
        i0v = np.searchsorted(z_abs, D, side="right") - 1
        i_lo = np.clip(i0v, 0, nz - 1)
        i_hi = np.minimum(i_lo + 1, nz - 1)
        lo_z = z_abs[i_lo]
        span = z_abs[i_hi] - lo_z
        frac = np.clip((D - lo_z) / np.where(span > 0.0, span, 1.0), 0.0, 1.0)
        rows = np.arange(s, e)[:, None]
        T_out[s:e] = (1.0 - frac) * T_prof[rows, i_lo] + frac * T_prof[rows, i_hi]
        S_out[s:e] = (1.0 - frac) * S_prof[rows, i_lo] + frac * S_prof[rows, i_hi]
    del T_prof, S_prof

    # diagnostics: isolated cells (surface donor), donor distance at the deepest layer used
    wm = (np.asarray(wet_mask, dtype=bool).reshape(ncell) if wet_mask is not None
          else np.ones(ncell, dtype=bool))
    if mesh_res is None:
        thresh = isolated_factor * np.median(d0[wm]) * constants.R_earth
    else:
        thresh = isolated_factor * mesh_res[i0]
    isolated = (d0 * constants.R_earth > thresh) & wm
    n_iso = int(isolated.sum())
    far = np.argsort(np.where(wm, d0, -1.0))[-5:][::-1]
    far_str = ", ".join(f"({lat_flat[i]:.4f}, {lon_flat[i]:.4f})" for i in far)
    log(f"init_ocean_from_fesom_mesh: {n_iso} isolated wet target cells "
        f"(nearest surface donor > {isolated_factor:g} x mesh resolution); "
        f"5 farthest (lat, lon) deg: {far_str}")
    if wm.any():
        # deepest layer actually read by any wet target: the UPPER bracket of
        # the deepest target depth (the lower bracket omits the deeper donor)
        d_max = float(np.max(depths[wm]))
        kk_max = int(min(nz - 1, max(0, np.searchsorted(z_abs, d_max, side="right"))))
        while trees[kk_max] is None:
            kk_max -= 1
        dd, _ = trees[kk_max].query(xyz_t[wm], k=1)
        dd_km = np.asarray(dd) * constants.R_earth / 1e3
        log(f"init_ocean_from_fesom_mesh: deepest layer used {kk_max} (|Z| {z_abs[kk_max]:.0f} m) "
            f"has {donors_l[kk_max].size} wet donor nodes; nearest-donor distance over wet targets "
            f"median {np.median(dd_km):.1f} km, p99 {np.percentile(dd_km, 99):.1f} km, max {dd_km.max():.1f} km")

    T_out = T_out.reshape(spatial + (nlev,))
    S_out = S_out.reshape(spatial + (nlev,))
    if cache_path is not None:
        import tempfile
        os.makedirs(cache_dir, exist_ok=True)
        # mkstemp keeps the tmp file inside cache_dir (same filesystem, so the
        # os.replace below is atomic) and unique across hosts/jobs, unlike a
        # pid-based name which can collide between rank 0s on different nodes.
        # Saved arrays are the pre-jnp float64 arrays: a hit is bit-identical.
        fd, tmp = tempfile.mkstemp(
            dir=cache_dir, prefix=os.path.basename(cache_path) + ".tmp.")
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, T=T_out, S=S_out)
            umask = os.umask(0)   # mkstemp creates 0600; a shared cache dir needs 0644
            os.umask(umask)
            os.chmod(tmp, 0o644 & ~umask)
            os.replace(tmp, cache_path)
            log(f"init_ocean_from_fesom_mesh: cache written {cache_path}")
        except Exception as exc:
            log(f"init_ocean_from_fesom_mesh: WARNING: cache write failed: {exc!r}")
            try:
                os.unlink(tmp)
            except OSError:
                pass
            # Peers are polling for this publication; swallowing the failure
            # would hang them until cache_wait_timeout_s.
            if jax.process_count() > 1:
                raise
    return jnp.asarray(T_out, dtype=jnp.float64), jnp.asarray(S_out, dtype=jnp.float64)
