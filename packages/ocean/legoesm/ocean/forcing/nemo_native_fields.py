"""Loaders for NEMO-native eORCA1 forcing/IC fields (exact-recipe inputs).

The reference NEMO run reads several fields the faithful legoESM runs
should consume VERBATIM instead of re-deriving from other products:

- ``sss_climatology_for_restoring.nc`` (``presalt``, 12 × y × x): the
  ``sn_sss`` target of the sbcssr salinity damping.  Restoring toward
  anything else (e.g. the annual IC surface) holds the Arctic shelves
  several PSU too salty — the day-90 northern SSS residual.
- ``woce_{temp,salt}_monthly_init_4p2.nc`` (``contemp``/``presalt``,
  12 × 75 × y × x): the ``sn_tem``/``sn_sal`` initial state.  A run
  starting 1 January should initialise from month 1, not the annual
  mean.

File fields are the eORCA1 INTERIOR (y=331, x=360).  The tripole model
grid carries the ORCA halos (y+1 north row, 2 cyclic overlap columns,
i.e. 332 × 362): interior cells embed at ``[:-1, 1:-1]``, the cyclic
columns copy per the 2-point overlap, and the (land-masked) north halo
row copies the row below.  Other grids go through the shared
:class:`~legoesm.ocean.forcing.curvilinear_regrid.NearestWetRegridder`.

``strip_north_rows`` (default 0) targets a tripole built with
``create_tripole_grid(strip_north_rows=N)``: the result is exactly the
unstripped embed with its last ``N`` rows dropped (the stripped fold-halo
row is a copy of the row below, so nothing independent is lost).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from legoesm import constants
from legoesm.ocean.forcing.curvilinear_regrid import (
    NearestWetRegridder,
    coords_match,
)


def _read_var(path: str, var: str) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray]:
    """Return (field, nav_lat, nav_lon) with masked values as NaN."""
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        if var not in ds.variables:
            raise ValueError(
                f"{path}: variable {var!r} not found; has "
                f"{sorted(ds.variables)[:12]}")
        arr = np.ma.filled(ds.variables[var][:], np.nan).astype(np.float64)
        lat = np.asarray(ds.variables["nav_lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["nav_lon"][:], dtype=np.float64)
    return arr, lat, lon


def read_nemo_tmask_interior(mesh_path: str) -> np.ndarray:
    """NEMO's own 3-D wet mask on the eORCA1 INTERIOR, ``(nlev, y, x)`` bool.

    The NEMO input files (``woce_*_monthly_init``, ``sss_climatology_for_
    restoring``) are FLOOD-FILLED: every land / below-seafloor cell holds a
    finite number NEMO never reads because ``tmask`` is 0 there (40914 of
    106208 finite surface cells are land; along the Ligurian coast the
    filled values at 200 m are S~31 next to 38.4 wet water).  A regridder
    that takes "finite" as "wet" samples them.  The mesh_mask carries the
    halos ``(1, nlev, y+1, x+2)``; the interior is ``[:-1, 1:-1]``.
    """
    import netCDF4

    with netCDF4.Dataset(mesh_path) as ds:
        if "tmask" not in ds.variables:
            raise ValueError(f"{mesh_path}: no 'tmask' variable")
        tm = np.asarray(ds.variables["tmask"][:])
    tm = tm.reshape(tm.shape[-3:])
    return tm[:, :-1, 1:-1] > 0.5


def nemo_src_tmask_for(mesh_path: str, field_path: str,
                       coord_tol_deg: float = 1e-3) -> np.ndarray:
    """``read_nemo_tmask_interior(mesh_path)`` PROVEN to sit on the grid of
    ``field_path``: the mesh_mask's ``gphit``/``glamt`` interior must match
    the file's ``nav_lat``/``nav_lon`` to ``coord_tol_deg`` (measured
    7.6e-6 / 5.3e-5 deg for eORCA1.2 vs the 4p2 IC files; any other halo
    offset differs by whole degrees).  A shape check alone cannot see an
    off-by-one row/column, and a mismatched mesh must RAISE — falling back
    to "finite == wet" re-arms the flood-fill bug this guards against."""
    import netCDF4

    with netCDF4.Dataset(field_path) as ds:
        f_lat = np.asarray(ds.variables["nav_lat"][:], dtype=np.float64)
        f_lon = np.asarray(ds.variables["nav_lon"][:], dtype=np.float64)
    with netCDF4.Dataset(mesh_path) as ds:
        m_lat = np.asarray(ds.variables["gphit"][:], dtype=np.float64)
        m_lon = np.asarray(ds.variables["glamt"][:], dtype=np.float64)
    m_lat = m_lat.reshape(m_lat.shape[-2:])[:-1, 1:-1]
    m_lon = m_lon.reshape(m_lon.shape[-2:])[:-1, 1:-1]
    if m_lat.shape != f_lat.shape:
        raise ValueError(
            f"mesh_mask {mesh_path} interior {m_lat.shape} != {field_path} "
            f"grid {f_lat.shape}: pass the mesh_mask of the grid the NEMO "
            "field lives on (its land mask is needed; the file is "
            "flood-filled over land).")
    dlat = float(np.abs(m_lat - f_lat).max())
    dlon = float(np.abs(((m_lon - f_lon + 180.0) % 360.0) - 180.0).max())
    if dlat > coord_tol_deg or dlon > coord_tol_deg:
        raise ValueError(
            f"mesh_mask {mesh_path} interior coordinates do not match "
            f"{field_path} (max |dlat| {dlat:.3e}, |dlon| {dlon:.3e} deg > "
            f"{coord_tol_deg}): halo offset or different grid.")
    return read_nemo_tmask_interior(mesh_path)


def embed_orca_interior(src: np.ndarray, n_lat: int,
                        n_lon: int, strip_north_rows: int = 0) -> np.ndarray:
    """Embed an eORCA-interior field (..., y, x) into the model's
    (..., n_lat, n_lon) halo-carrying grid.

    Requires ``n_lat == y+1`` and ``n_lon == x+2`` (the ORCA north halo
    row + 2-point cyclic overlap columns).  The wrap columns follow the
    2-point overlap (``col0 <- col[nx-2]``, ``col[nx-1] <- col1``) and
    the north halo row copies the row below (it is land-masked in the
    model; the copy just keeps the array finite).

    ``strip_north_rows = N > 0``: the model grid had its last ``N`` rows
    removed, so ``n_lat == y + 1 - N``; the result is the unstripped embed
    minus its last ``N`` rows.
    """
    strip = int(strip_north_rows)
    if strip < 0:
        raise ValueError(f"strip_north_rows must be >= 0, got {strip}")
    if strip:
        return embed_orca_interior(src, n_lat + strip, n_lon)[..., :n_lat, :]
    src = np.asarray(src)
    ny, nx = src.shape[-2:]
    if (ny, nx) != (n_lat - 1, n_lon - 2):
        raise ValueError(
            f"embed_orca_interior: source interior {(ny, nx)} does not "
            f"match model grid ({n_lat}, {n_lon}) minus ORCA halos "
            f"({n_lat - 1}, {n_lon - 2})")
    out = np.full(src.shape[:-2] + (n_lat, n_lon), np.nan,
                  dtype=np.float64)
    out[..., :-1, 1:-1] = src
    out[..., :-1, 0] = src[..., :, -1]     # col0 <- interior col[nx-1]
    out[..., :-1, -1] = src[..., :, 0]     # col[-1] <- interior col0
    out[..., -1, :] = out[..., -2, :]      # north halo row (masked)
    return out


def _to_model_grid_2d(field_i, src_lat, src_lon, lat_T_deg, lon_T_deg,
                      wet_mask, regridder=None, src_wet=None,
                      strip_north_rows: int = 0):
    """One 2-D interior field -> model grid (embed or nearest-wet).

    Structured targets (regular lat-lon / tripole) pass a 2-D ``lat_T_deg``:
    the tripole native mesh embeds directly (ORCA halo fill), other structured
    grids regrid nearest-wet.  Unstructured targets (MPAS Voronoi) pass 1-D
    PAIRED cell centres ``(nCells,)`` -> always nearest-wet, no embed (the
    ORCA-halo embed is meaningless off a structured mesh).
    """
    lat_arr = np.asarray(lat_T_deg)
    if lat_arr.ndim == 1:                      # unstructured (MPAS) paired pts
        if regridder is None:
            regridder = NearestWetRegridder(src_lon, src_lat,
                                            _source_wet(field_i, src_wet),
                                            lon_T_deg, lat_T_deg,
                                            structured=False)
        return regridder(field_i), regridder
    n_lat, n_lon = lat_arr.shape
    strip = int(strip_north_rows)
    # Rows the interior shares with the model: all of them on a stripped grid
    # (n_lat = y + 1 - strip), all but the north halo row otherwise.
    n_cmp = min(n_lat, n_lat - 1 + strip)
    if (field_i.shape == (n_lat - 1 + strip, n_lon - 2)
            and coords_match(src_lat[:n_cmp], src_lon[:n_cmp],
                             lat_arr[:n_cmp, 1:-1],
                             np.asarray(lon_T_deg)[:n_cmp, 1:-1])):
        return (embed_orca_interior(field_i, n_lat, n_lon, strip),
                regridder)
    if regridder is None:
        regridder = NearestWetRegridder(src_lon, src_lat,
                                        _source_wet(field_i, src_wet),
                                        lon_T_deg, lat_T_deg)
    return regridder(field_i), regridder


def _source_wet(field_i, src_wet):
    """Source cells the nearest-wet regridder may sample: finite AND (when
    NEMO's tmask is supplied) wet in NEMO — the input files are flood-filled
    over land / below the seafloor (see read_nemo_tmask_interior)."""
    wet = np.isfinite(field_i)
    if src_wet is None:
        return wet
    src_wet = np.asarray(src_wet, dtype=bool)
    if src_wet.shape != wet.shape:
        raise ValueError(
            f"src_wet shape {src_wet.shape} != source field {wet.shape}")
    return wet & src_wet


def load_nemo_sss_restoring_climatology(
    path: str,
    lat_T_deg: np.ndarray,
    lon_T_deg: np.ndarray,
    wet_mask: np.ndarray,
    var: str = "presalt",
    src_tmask: np.ndarray | None = None,
    strip_north_rows: int = 0,
) -> np.ndarray:
    """(12, n_lat, n_lon) monthly SSS-restoring target [PSU].

    ``src_tmask`` (``(nlev, y, x)`` or ``(y, x)`` bool, see
    ``read_nemo_tmask_interior``): NEMO's wet mask on the source grid — the
    file is flood-filled over land, so regridding grids must pass it or the
    nearest-"wet" search samples land values.

    Native-grid tripole runs embed directly (halo fill); other grids
    regrid nearest-wet.  Land cells are filled with the nearest wet
    value (the restoring apply is masked anyway) so the array is
    NaN-free.
    """
    arr, src_lat, src_lon = _read_var(path, var)
    if arr.ndim == 4:                # (12, 1, y, x) depth-degenerate
        arr = arr[:, 0]
    if arr.ndim != 3 or arr.shape[0] != 12:
        raise ValueError(
            f"{path}: expected (12, y, x) monthly SSS, got {arr.shape}")
    out = np.empty((12,) + np.asarray(lat_T_deg).shape, dtype=np.float64)
    regridder = None
    _src_wet0 = (None if src_tmask is None
                 else np.asarray(src_tmask, dtype=bool).reshape(
                     (-1,) + arr.shape[-2:])[0])
    for m in range(12):
        out[m], regridder = _to_model_grid_2d(
            arr[m], src_lat, src_lon, lat_T_deg, lon_T_deg, wet_mask,
            regridder, src_wet=_src_wet0, strip_north_rows=strip_north_rows)
    # NaN-free: fill any remaining gaps (land / unmapped) with the
    # monthly wet mean so masked applies never touch NaNs.
    for m in range(12):
        bad = ~np.isfinite(out[m])
        if bad.any():
            out[m][bad] = np.nanmean(out[m])
    return out


def _read_depth_axis(path: str, var: str = "deptht") -> np.ndarray:
    """1-D positive-down depth ladder [m] of a NEMO init file.

    NEMO files carry the ladder as ``deptht`` (grid_T outputs) or
    ``nav_lev`` (the woce_*_monthly_init IC layout) — accept both, first
    match wins.  Raises (never guesses a ladder) when neither is present
    or the axis is not strictly increasing — a mis-levelled vertical
    interpolation is a silently wrong IC, not an error.
    """
    import netCDF4

    candidates = (var, "nav_lev") if var == "deptht" else (var,)
    with netCDF4.Dataset(path) as ds:
        found = next((c for c in candidates if c in ds.variables), None)
        if found is None:
            raise ValueError(
                f"{path}: depth axis {candidates!r} not found (needed for "
                f"vertical interpolation onto a non-NEMO ladder); has "
                f"{sorted(ds.variables)[:12]}")
        var = found
        depths = np.asarray(ds.variables[var][:], dtype=np.float64).ravel()
    if depths.size < 2 or not np.all(np.diff(depths) > 0.0) \
            or not np.all(depths >= 0.0):
        raise ValueError(
            f"{path}: {var!r} must be a strictly increasing positive-down "
            f"ladder; got {depths[:4]}...{depths[-2:]}")
    return depths


def _interp_columns_to_depths(field: np.ndarray, src_depths: np.ndarray,
                              target_depths: np.ndarray) -> np.ndarray:
    """np.interp every column of ``field (..., nlev_src)`` from the
    ``src_depths`` ladder onto ``target_depths`` (both 1-D, positive-down,
    strictly increasing).  Vectorised over columns via shared bracketing
    indices (the ladder is column-independent); out-of-range targets clamp
    to the end values — exactly ``np.interp``'s semantics.
    """
    src = np.asarray(src_depths, dtype=np.float64)
    tgt = np.asarray(target_depths, dtype=np.float64)
    j = np.clip(np.searchsorted(src, tgt), 1, src.size - 1)
    w = (tgt - src[j - 1]) / (src[j] - src[j - 1])
    w = np.clip(w, 0.0, 1.0)          # end-clamp (np.interp behaviour)
    return field[..., j - 1] * (1.0 - w) + field[..., j] * w


def load_nemo_monthly_init_ts(
    temp_path: str,
    salt_path: str,
    lat_T_deg: np.ndarray,
    lon_T_deg: np.ndarray,
    n_levels: int,
    month: int = 1,
    temp_var: str = "contemp",
    salt_var: str = "presalt",
    target_depths: np.ndarray | None = None,
    src_tmask: np.ndarray | None = None,
    nemo_tint: bool = False,
    strip_north_rows: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """(T, S) 3-D initial state for the given month (1-based).

    ``nemo_tint=True`` reproduces what NEMO actually starts from: with
    ``sn_tem/sn_sal ... ln_tint = .true.`` (ORCA1 namelist_cfg:81-82)
    ``fldread`` places each monthly record at the MIDDLE of its month and
    interpolates linearly to the model time (fldread.F90:913-917, 225-227),
    so a run starting at 00:00 on 1 January takes ``0.5 * December +
    0.5 * January`` (31/62, whatever the year). Only ``month=1`` is
    supported with this option: other start dates need the run calendar
    (2000 is a leap year) to place the record centres.
    Measured 2026-09-14 (hudson2_9771697): NEMO's day-5 Hudson Bay surface
    salinity is 28.90 against 25.10 for January alone and 28.88 for the
    blend (rms over the box 2.98 vs 0.22); globally the blend fits NEMO's
    day-5 S (0.34 vs 0.45) and SST (0.32 vs 0.61) better as well.
    ``False`` (default) keeps the historical single-month field.

    ``src_tmask`` (``(nlev, y, x)`` bool from ``read_nemo_tmask_interior``):
    NEMO's per-level wet mask on the source grid.  The IC files are
    flood-filled over land and below the seafloor; regridding grids MUST
    pass it or the per-level nearest-"wet" search samples those fills (the
    FESOM Ligurian column got S~31 at 200 m from a land cell next to 38.4
    water and blew up within a day).  The native tripole embed ignores it.

    The files carry NEMO's 75 reference levels.  Default
    (``target_depths=None``): the model must run the SAME ladder (the
    ``nemolev`` configurations) — anything else raises rather than
    silently interpolating.  Note the temperature variable is
    Conservative Temperature (``contemp``): the reference NEMO run
    reads it with the same label into its TEOS-10 state; the faithful
    runs consume it as-is (documented convention difference for the
    EOS-80-style EOS paths).

    ``target_depths`` (1-D, POSITIVE-DOWN cell-centre depths [m], length
    ``n_levels``, strictly increasing — e.g. ``-mesh.Z`` for a FESOM
    mesh whose ``Z`` is negative-down): enables a model ladder different
    from the file's.  The horizontal regrid still runs per SOURCE level
    (the wet mask shrinks with depth), the NaN fill runs FIRST on the
    source ladder, then each column is linearly interpolated from the
    file's ``deptht`` ladder onto ``target_depths`` (end-clamped,
    ``np.interp`` semantics).
    """
    if not (1 <= int(month) <= 12):
        raise ValueError(f"month must be 1..12, got {month!r}")
    if nemo_tint:
        # Codex: the record centres follow NEMO's calendar (2000 is a leap
        # year), so the weight is start-date specific; only the 1 January
        # start (31/62 = 0.5 whatever the year) is defined here. Each month
        # goes through the SAME wet-regrid + column-fill pipeline on its own
        # and the finished fields are blended, so a hole in one month never
        # discards the other month's value.
        if int(month) != 1:
            raise ValueError(
                "nemo_tint is defined for a 1 January start only "
                "(0.5 * December + 0.5 * January); other start months need "
                "the run calendar to place NEMO's record centres.")
        kw = dict(n_levels=n_levels, temp_var=temp_var, salt_var=salt_var,
                  target_depths=target_depths, src_tmask=src_tmask,
                  nemo_tint=False, strip_north_rows=strip_north_rows)
        T_jan, S_jan = load_nemo_monthly_init_ts(
            temp_path, salt_path, lat_T_deg, lon_T_deg, month=1, **kw)
        T_dec, S_dec = load_nemo_monthly_init_ts(
            temp_path, salt_path, lat_T_deg, lon_T_deg, month=12, **kw)
        print("[setup] NEMO monthly init: fldread ln_tint blend for 1 January "
              "= 0.5 * December + 0.5 * January")
        return 0.5 * (T_jan + T_dec), 0.5 * (S_jan + S_dec)
    T_arr, src_lat, src_lon = _read_var(temp_path, temp_var)
    S_arr, _, _ = _read_var(salt_path, salt_var)
    for name, arr in (("temp", T_arr), ("salt", S_arr)):
        if arr.ndim != 4 or arr.shape[0] != 12:
            raise ValueError(
                f"{name} init: expected (12, nlev, y, x), got {arr.shape}")
    nlev_src = T_arr.shape[1]
    if src_tmask is not None:
        src_tmask = np.asarray(src_tmask, dtype=bool)
        if src_tmask.shape != T_arr.shape[1:]:
            raise ValueError(
                f"src_tmask shape {src_tmask.shape} != IC field "
                f"{T_arr.shape[1:]} (nlev, y, x)")
    src_depths = None
    if target_depths is None:
        if nlev_src != int(n_levels):
            raise ValueError(
                f"NEMO monthly init has {nlev_src} levels; the model runs "
                f"{n_levels}. The loader is exact-ladder only (nemolev "
                "runs) unless target_depths is given "
                "— no implicit vertical interpolation.")
    else:
        target_depths = np.asarray(target_depths, dtype=np.float64).ravel()
        if target_depths.size != int(n_levels):
            raise ValueError(
                f"target_depths has {target_depths.size} levels; the model "
                f"runs {n_levels}.")
        if not np.all(np.diff(target_depths) > 0.0) \
                or not np.all(target_depths >= 0.0):
            raise ValueError(
                "target_depths must be strictly increasing POSITIVE-DOWN "
                f"depths [m]; got {target_depths[:4]}... (a negative-down "
                "ladder like mesh.Z must be negated by the caller).")
        src_depths = _read_depth_axis(temp_path)
        if src_depths.size != nlev_src:
            raise ValueError(
                f"{temp_path}: deptht has {src_depths.size} entries but the "
                f"field carries {nlev_src} levels.")
    m = int(month) - 1
    shape = np.asarray(lat_T_deg).shape
    T_out = np.empty(shape + (nlev_src,), dtype=np.float64)
    S_out = np.empty_like(T_out)
    for k in range(nlev_src):
        # PER-LEVEL regridder: the wet mask shrinks with depth, and a
        # level-0 nearest-wet map would sample below-bathy NaN at depth
        # on the regrid path (codex r11 MED#1). T and S share the level's
        # bathymetry, so one regridder serves both. Native-embed targets
        # never build one (coords_match short-circuits).
        _sw_k = None if src_tmask is None else np.asarray(src_tmask)[k]
        if _sw_k is not None and not np.any(np.isfinite(T_arr[m, k]) & _sw_k):
            # NEMO's deepest level(s) are all land: nothing to sample, leave
            # NaN for the below-seafloor fill below (was: regridder raise).
            T_out[..., k] = np.nan
            S_out[..., k] = np.nan
            continue
        T_out[..., k], regridder_k = _to_model_grid_2d(
            T_arr[m, k], src_lat, src_lon, lat_T_deg, lon_T_deg, None,
            None, src_wet=_sw_k, strip_north_rows=strip_north_rows)
        S_out[..., k], _ = _to_model_grid_2d(
            S_arr[m, k], src_lat, src_lon, lat_T_deg, lon_T_deg, None,
            regridder_k, src_wet=_sw_k, strip_north_rows=strip_north_rows)
    # Finite everywhere: below-seafloor / land cells inherit the deepest
    # finite value of their column (masked in the model, but the state
    # arrays must be NaN-free), then any all-NaN column takes the level
    # mean.
    for out in (T_out, S_out):
        for k in range(1, nlev_src):
            bad = ~np.isfinite(out[..., k])
            out[..., k][bad] = out[..., k - 1][bad]
        for k in range(nlev_src):
            bad = ~np.isfinite(out[..., k])
            if bad.any():
                out[..., k][bad] = np.nanmean(out[..., k])
    if target_depths is not None:
        # NaN fill ran FIRST (above) on the source ladder, so every column
        # is finite before the vertical interpolation — a NaN neighbour
        # would otherwise poison both bracketing levels.
        T_out = _interp_columns_to_depths(T_out, src_depths, target_depths)
        S_out = _interp_columns_to_depths(S_out, src_depths, target_depths)
    return T_out, S_out


# ---------------------------------------------------------------------------
# SI3 ice initial state (Ice_initialization.nc)
# ---------------------------------------------------------------------------

# SI3 ice-IC variable names (NEMO namelist_ice &namini sn_ati/sn_hti/
# sn_hts/sn_smi/sn_tsu).  at_i + ht_i define the ice cover and are
# REQUIRED; the rest are optional extras (returned as None when absent).
_ICE_INIT_REQUIRED = (("concentration", "at_i"), ("h_ice", "ht_i"))
_ICE_INIT_OPTIONAL = (("h_snow", "ht_s"), ("S_ice", "sm_i"), ("T_su", "tmsu"))
# Physical floor for a believable ice-surface temperature [K]; file values
# at/below this (SI3 writes 0 on ice-free cells) are treated as missing.
_ICE_TSU_MIN_K = 150.0


class NemoIceInit(NamedTuple):
    """SI3 ice initial state regridded onto the model T-grid.

    ``concentration``/``h_ice`` are always present, mutually coherent
    (``concentration > 0  <=>  h_ice > 0``), clamped (conc in [0, 1],
    thicknesses >= 0) and zero on target land / ice-free cells.
    ``h_snow``/``S_ice`` follow the same convention; ``T_su`` [K] is
    capped at the melting point and carries ``np.nan`` wherever there is
    no ice or no valid reading — consumers keep their own default there
    (apply via ``np.where(np.isfinite(T_su), T_su, current)``).
    Optional fields are ``None`` when the file does not carry them.
    """
    concentration: np.ndarray
    h_ice: np.ndarray
    h_snow: Optional[np.ndarray]
    S_ice: Optional[np.ndarray]
    T_su: Optional[np.ndarray]


def _read_ice_file(path: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Read the SI3 ice-IC variables + source coords in ONE file pass.

    Returns ``(fields, src_lat, src_lon)`` with ``fields[name]`` a 2-D
    ``(y, x)`` float64 array (time axis squeezed, masked values as NaN)
    or ``None`` for absent optional variables.  Coordinates come from
    ``nav_lat``/``nav_lon`` when present, else the SI3 ``y``/``x`` 2-D
    variables (same values under different names).
    """
    import netCDF4

    fields: dict = {}
    with netCDF4.Dataset(path) as ds:
        for name, var in _ICE_INIT_REQUIRED:
            if var not in ds.variables:
                raise ValueError(
                    f"{path}: required ice-IC variable {var!r} not found; "
                    f"has {sorted(ds.variables)[:12]}")
        for name, var in _ICE_INIT_REQUIRED + _ICE_INIT_OPTIONAL:
            if var not in ds.variables:
                fields[name] = None
                continue
            arr = np.ma.filled(ds.variables[var][:], np.nan).astype(
                np.float64)
            if arr.ndim == 3:            # (time, y, x) -> first record
                arr = arr[0]
            if arr.ndim != 2:
                raise ValueError(
                    f"{path}: {var!r} has shape {arr.shape}; expected "
                    "(y, x) or (time, y, x)")
            fields[name] = arr
        if "nav_lat" in ds.variables and "nav_lon" in ds.variables:
            lat = np.asarray(ds.variables["nav_lat"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["nav_lon"][:], dtype=np.float64)
        elif "y" in ds.variables and "x" in ds.variables \
                and ds.variables["y"].ndim == 2:
            lat = np.asarray(ds.variables["y"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["x"][:], dtype=np.float64)
        else:
            raise ValueError(
                f"{path}: no 2-D coordinates found (need nav_lat/nav_lon "
                "or the SI3 y/x pair)")
    ref = fields["concentration"].shape
    for name, arr in fields.items():
        if arr is not None and arr.shape != ref:
            raise ValueError(
                f"{path}: {name} shape {arr.shape} != at_i shape {ref}")
    if lat.shape != ref or lon.shape != ref:
        raise ValueError(
            f"{path}: coord shapes lat {lat.shape} / lon {lon.shape} != "
            f"field shape {ref}")
    return fields, lat, lon


def load_nemo_ice_init(
    path: str,
    lat_T_deg: np.ndarray,
    lon_T_deg: np.ndarray,
    wet_mask: np.ndarray,
    strip_north_rows: int = 0,
) -> NemoIceInit:
    """SI3 ``Ice_initialization.nc`` -> ice IC on the model T-grid.

    Same grid convention as :func:`load_nemo_monthly_init_ts`: a
    native-eORCA1 tripole target embeds the interior exactly
    (:func:`embed_orca_interior`); any other grid goes nearest-wet
    through the shared :class:`NearestWetRegridder`.  The SI3 file
    writes (0, 0) lat/lon on land cells (real eORCA1 T-points sit at
    half-degree longitude offsets, so no ocean point is at exactly
    (0, 0)), so source "wetness" = coordinate validity — nearest-wet
    lookups can never sample a land cell, and the same-mesh check runs
    on the valid cells only.

    ``lat_T_deg``/``lon_T_deg`` may be 2-D (structured grids) or 1-D
    POINT LISTS (MPAS cell centres — NOT separable axes; handled here
    because ``NearestWetRegridder`` meshgrids 1-D inputs).
    ``wet_mask`` (1=ocean, 0=land, target shape) zeroes the ice on
    model land.  Clamping/coherence per :class:`NemoIceInit`.
    ``strip_north_rows``: see :func:`embed_orca_interior`.
    """
    fields, src_lat, src_lon = _read_ice_file(path)
    tgt_lat = np.asarray(lat_T_deg, dtype=np.float64)
    tgt_lon = np.asarray(lon_T_deg, dtype=np.float64)
    if tgt_lat.shape != tgt_lon.shape:
        raise ValueError(
            f"load_nemo_ice_init: target lat {tgt_lat.shape} and lon "
            f"{tgt_lon.shape} differ")
    wet = np.asarray(wet_mask, dtype=np.float64)
    if wet.shape != tgt_lat.shape:
        raise ValueError(
            f"load_nemo_ice_init: wet_mask shape {wet.shape} != target "
            f"grid shape {tgt_lat.shape}")

    # Source coordinate validity (SI3 zeroes coords on land).
    coord_valid = ~((src_lat == 0.0) & (src_lon == 0.0))
    if not coord_valid.any():
        raise ValueError(f"{path}: no coordinate-valid source cells")

    ny, nx = fields["concentration"].shape
    n_lat, n_lon = (tgt_lat.shape if tgt_lat.ndim == 2 else (0, 0))
    strip = int(strip_north_rows)
    n_cmp = min(n_lat, n_lat - 1 + strip)
    native = (
        tgt_lat.ndim == 2
        and (ny, nx) == (n_lat - 1 + strip, n_lon - 2)
        and coords_match(src_lat[:n_cmp], src_lon[:n_cmp],
                         tgt_lat[:n_cmp, 1:-1], tgt_lon[:n_cmp, 1:-1],
                         valid=coord_valid[:n_cmp])
    )
    out: dict = {}
    if native:
        for name, arr in fields.items():
            out[name] = (None if arr is None
                         else embed_orca_interior(arr, n_lat, n_lon, strip))
    else:
        # 1-D targets are POINT LISTS (MPAS): lift to (n, 1) so the
        # regridder does not meshgrid them into an (n, n) product grid.
        point_list = tgt_lat.ndim == 1
        rg_lat = tgt_lat[:, None] if point_list else tgt_lat
        rg_lon = tgt_lon[:, None] if point_list else tgt_lon
        regridder = NearestWetRegridder(src_lon, src_lat, coord_valid,
                                        rg_lon, rg_lat)
        for name, arr in fields.items():
            if arr is None:
                out[name] = None
                continue
            mapped = regridder(arr)
            out[name] = mapped[:, 0] if point_list else mapped

    # --- Clamp + coherence on the MODEL grid --------------------------
    conc = np.clip(np.nan_to_num(out["concentration"]), 0.0, 1.0)
    h_ice = np.maximum(np.nan_to_num(out["h_ice"]), 0.0)
    present = (wet > 0.5) & (conc > 0.0) & (h_ice > 0.0)
    conc = np.where(present, conc, 0.0)
    h_ice = np.where(present, h_ice, 0.0)
    h_snow = out["h_snow"]
    if h_snow is not None:
        h_snow = np.where(present,
                          np.maximum(np.nan_to_num(h_snow), 0.0), 0.0)
    S_ice = out["S_ice"]
    if S_ice is not None:
        S_ice = np.where(present,
                         np.maximum(np.nan_to_num(S_ice), 0.0), 0.0)
    T_su = out["T_su"]
    if T_su is not None:
        # Valid reading = on ice AND above the physical floor (SI3 writes
        # 0 K on ice-free cells); cap at the melting point.  NaN elsewhere
        # per the NemoIceInit contract (consumer keeps its default).
        T_raw = np.nan_to_num(T_su)
        good = present & (T_raw > _ICE_TSU_MIN_K)
        T_su = np.where(good, np.minimum(T_raw, constants.T_freeze),
                        np.nan)
    return NemoIceInit(concentration=conc, h_ice=h_ice, h_snow=h_snow,
                       S_ice=S_ice, T_su=T_su)
