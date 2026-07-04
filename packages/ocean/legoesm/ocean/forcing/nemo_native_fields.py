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

import numpy as np

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
    """One 2-D interior field -> model grid (embed or nearest-wet)."""
    n_lat, n_lon = np.asarray(lat_T_deg).shape
    if (field_i.shape == (n_lat - 1, n_lon - 2)
            and coords_match(src_lat, src_lon,
                             np.asarray(lat_T_deg)[:-1, 1:-1],
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
