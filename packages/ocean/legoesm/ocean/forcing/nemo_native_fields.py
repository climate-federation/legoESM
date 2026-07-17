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


def embed_orca_interior(src: np.ndarray, n_lat: int,
                        n_lon: int) -> np.ndarray:
    """Embed an eORCA-interior field (..., y, x) into the model's
    (..., n_lat, n_lon) halo-carrying grid.

    Requires ``n_lat == y+1`` and ``n_lon == x+2`` (the ORCA north halo
    row + 2-point cyclic overlap columns).  The wrap columns follow the
    2-point overlap (``col0 <- col[nx-2]``, ``col[nx-1] <- col1``) and
    the north halo row copies the row below (it is land-masked in the
    model; the copy just keeps the array finite).
    """
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
                      wet_mask, regridder=None):
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
            src_wet = np.isfinite(field_i)
            regridder = NearestWetRegridder(src_lon, src_lat, src_wet,
                                            lon_T_deg, lat_T_deg,
                                            structured=False)
        return regridder(field_i), regridder
    n_lat, n_lon = lat_arr.shape
    if (field_i.shape == (n_lat - 1, n_lon - 2)
            and coords_match(src_lat, src_lon,
                             lat_arr[:-1, 1:-1],
                             np.asarray(lon_T_deg)[:-1, 1:-1])):
        return embed_orca_interior(field_i, n_lat, n_lon), regridder
    if regridder is None:
        src_wet = np.isfinite(field_i)
        regridder = NearestWetRegridder(src_lon, src_lat, src_wet,
                                        lon_T_deg, lat_T_deg)
    return regridder(field_i), regridder


def load_nemo_sss_restoring_climatology(
    path: str,
    lat_T_deg: np.ndarray,
    lon_T_deg: np.ndarray,
    wet_mask: np.ndarray,
    var: str = "presalt",
) -> np.ndarray:
    """(12, n_lat, n_lon) monthly SSS-restoring target [PSU].

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
    for m in range(12):
        out[m], regridder = _to_model_grid_2d(
            arr[m], src_lat, src_lon, lat_T_deg, lon_T_deg, wet_mask,
            regridder)
    # NaN-free: fill any remaining gaps (land / unmapped) with the
    # monthly wet mean so masked applies never touch NaNs.
    for m in range(12):
        bad = ~np.isfinite(out[m])
        if bad.any():
            out[m][bad] = np.nanmean(out[m])
    return out


def load_nemo_monthly_init_ts(
    temp_path: str,
    salt_path: str,
    lat_T_deg: np.ndarray,
    lon_T_deg: np.ndarray,
    n_levels: int,
    month: int = 1,
    temp_var: str = "contemp",
    salt_var: str = "presalt",
) -> tuple[np.ndarray, np.ndarray]:
    """(T, S) 3-D initial state for the given month (1-based).

    The files carry NEMO's 75 reference levels; the model must run the
    SAME ladder (the ``nemolev`` configurations) — anything else raises
    rather than silently interpolating.  Note the temperature variable
    is Conservative Temperature (``contemp``): the reference NEMO run
    reads it with the same label into its TEOS-10 state; the faithful
    runs consume it as-is (documented convention difference for the
    EOS-80-style EOS paths).
    """
    if not (1 <= int(month) <= 12):
        raise ValueError(f"month must be 1..12, got {month!r}")
    T_arr, src_lat, src_lon = _read_var(temp_path, temp_var)
    S_arr, _, _ = _read_var(salt_path, salt_var)
    for name, arr in (("temp", T_arr), ("salt", S_arr)):
        if arr.ndim != 4 or arr.shape[0] != 12:
            raise ValueError(
                f"{name} init: expected (12, nlev, y, x), got {arr.shape}")
    nlev_src = T_arr.shape[1]
    if nlev_src != int(n_levels):
        raise ValueError(
            f"NEMO monthly init has {nlev_src} levels; the model runs "
            f"{n_levels}. The loader is exact-ladder only (nemolev runs) "
            "— no vertical interpolation.")
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
        T_out[..., k], regridder_k = _to_model_grid_2d(
            T_arr[m, k], src_lat, src_lon, lat_T_deg, lon_T_deg, None,
            None)
        S_out[..., k], _ = _to_model_grid_2d(
            S_arr[m, k], src_lat, src_lon, lat_T_deg, lon_T_deg, None,
            regridder_k)
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
    native = (
        tgt_lat.ndim == 2
        and (ny, nx) == (n_lat - 1, n_lon - 2)
        and coords_match(src_lat, src_lon,
                         tgt_lat[:-1, 1:-1], tgt_lon[:-1, 1:-1],
                         valid=coord_valid)
    )
    out: dict = {}
    if native:
        for name, arr in fields.items():
            out[name] = (None if arr is None
                         else embed_orca_interior(arr, n_lat, n_lon))
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
