"""Loader for the NEMO zdfiwm internal-wave mixing forcing maps.

Reads the de Lavergne et al. (2020) power / decay-scale product as
shipped with the NEMO ORCA1 reference configuration
(``zdfiwm_forcing_TRA.nc``: curvilinear eORCA1 ``nav_lon``/``nav_lat``,
variables ``power_bot``/``power_cri``/``power_nsq``/``power_sho``
[W/m²] and ``scale_bot``/``scale_cri`` [m]) and produces an
:class:`~legoesm.ocean.physics.vertical_mixing.internal_wave_mixing.IWMForcing`
on the model grid.

Regridding: when the target grid IS the source eORCA grid (tripole runs
on the same mesh) the fields pass through untouched.  Otherwise a
nearest-neighbour lookup on the unit sphere maps each target cell to the
closest WET source cell, and each POWER map is then rescaled by one
global factor so its area-integrated total (the TW budget NEMO prints at
``zdf_iwm_init``) is preserved — nearest-neighbour alone does not
conserve the integral, and the column powers are the physically
constrained quantity.  Decay scales are plain nearest-neighbour (they
are local length scales, not densities).
"""

from __future__ import annotations

import numpy as np

from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
    IWMForcing,
)

_POWER_VARS = ("power_bot", "power_cri", "power_nsq", "power_sho")
_SCALE_VARS = ("scale_bot", "scale_cri")
# NEMO zdfiwm_init hard-coded pre-read decay-scale default [m]
_SCALE_DEFAULT_M = 100.0


def _unit_sphere(lon_deg, lat_deg):
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon),
                     np.cos(lat) * np.sin(lon),
                     np.sin(lat)], axis=-1)


def read_iwm_file(path: str) -> dict:
    """Read the six zdfiwm variables + coordinates from ``path`` (numpy)."""
    import netCDF4 as nc

    out = {}
    with nc.Dataset(path) as ds:
        for v in _POWER_VARS + _SCALE_VARS:
            if v not in ds.variables:
                raise ValueError(
                    f"iwm forcing file {path!r} is missing variable {v!r} "
                    f"(expected the NEMO zdfiwm_forcing layout)")
            arr = np.asarray(ds.variables[v][:], dtype=np.float64)
            arr = np.squeeze(arr)          # drop the length-1 time axis
            out[v] = np.nan_to_num(arr, nan=0.0)
        out["nav_lon"] = np.asarray(ds.variables["nav_lon"][:], dtype=np.float64)
        out["nav_lat"] = np.asarray(ds.variables["nav_lat"][:], dtype=np.float64)
    return out


def load_iwm_forcing(
    path: str,
    lat_T,
    lon_T,
    *,
    target_area=None,
    source_area=None,
    land_mask=None,
    coord_match_tol_deg: float = 1.0e-3,
) -> IWMForcing:
    """Load + regrid the zdfiwm forcing onto the model tracer grid.

    Parameters
    ----------
    path : str
        NetCDF file (NEMO ``zdfiwm_forcing_TRA.nc`` layout).
    lat_T, lon_T : array (n_lat, n_lon) or broadcastable
        Target tracer-point coordinates [deg].  1-D lat/lon are
        broadcast to 2-D.
    target_area, source_area : array, optional
        Cell areas [m²] for the power-total renormalisation.  When either
        is missing the renormalisation uses spherical cos(lat) weights —
        adequate for the global-total scaling factor.
    land_mask : array, optional
        Target ocean mask (1 = ocean).  Applied to the POWER maps only
        (NEMO masks with ``smask0``); decay scales stay unmasked.
    coord_match_tol_deg : float
        If the target coordinates equal the source's within this
        tolerance (same mesh), skip regridding entirely.

    Returns
    -------
    IWMForcing  (jnp arrays; note ``hcri_inv`` = 1/scale_cri, matching
    zdfiwm_init).
    """
    import jax.numpy as jnp

    data = read_iwm_file(path)
    src_lon, src_lat = data["nav_lon"], data["nav_lat"]

    lat_T = np.asarray(lat_T, dtype=np.float64)
    lon_T = np.asarray(lon_T, dtype=np.float64)
    if lat_T.ndim == 1 and lon_T.ndim == 1:
        lat_T, lon_T = np.meshgrid(lat_T, lon_T, indexing="ij")
    if lat_T.shape != lon_T.shape:
        raise ValueError(
            f"lat_T {lat_T.shape} and lon_T {lon_T.shape} must match")

    same_mesh = (
        lat_T.shape == src_lat.shape
        and float(np.max(np.abs(lat_T - src_lat))) < coord_match_tol_deg
        and float(np.max(np.abs(
            (lon_T - src_lon + 180.0) % 360.0 - 180.0))) < coord_match_tol_deg
    )

    if same_mesh:
        fields = {v: data[v] for v in _POWER_VARS + _SCALE_VARS}
    else:
        from scipy.spatial import cKDTree

        # Wet source cells only (zero-power cells on land would bleed
        # zeros into coastal target cells; NEMO's own file is masked).
        src_wet = np.zeros(src_lat.shape, dtype=bool)
        for v in _POWER_VARS:
            src_wet |= data[v] > 0.0
        if not np.any(src_wet):
            raise ValueError(f"iwm forcing file {path!r} has no wet cells")
        tree = cKDTree(_unit_sphere(src_lon[src_wet], src_lat[src_wet]))
        _, idx = tree.query(
            _unit_sphere(lon_T, lat_T).reshape(-1, 3), k=1)
        fields = {}
        for v in _POWER_VARS + _SCALE_VARS:
            fields[v] = data[v][src_wet][idx].reshape(lat_T.shape)
        # Preserve each power map's global area integral (the TW totals).
        if source_area is None:
            src_w = np.cos(np.deg2rad(src_lat))
        else:
            src_w = np.asarray(source_area, dtype=np.float64)
        if target_area is None:
            tgt_w = np.cos(np.deg2rad(lat_T))
        else:
            tgt_w = np.asarray(target_area, dtype=np.float64)
        tgt_mask = (np.asarray(land_mask, dtype=np.float64)
                    if land_mask is not None else np.ones_like(lat_T))
        for v in _POWER_VARS:
            src_total = float((data[v] * src_w).sum())
            tgt_total = float((fields[v] * tgt_w * tgt_mask).sum())
            if tgt_total > 0.0 and src_total > 0.0:
                fields[v] = fields[v] * (src_total / tgt_total)

    # Guard the decay scales (hcri is INVERTED downstream).
    for v in _SCALE_VARS:
        fields[v] = np.where(fields[v] > 0.0, fields[v], _SCALE_DEFAULT_M)
    if land_mask is not None:
        m = np.asarray(land_mask, dtype=np.float64)
        for v in _POWER_VARS:
            fields[v] = fields[v] * m

    return IWMForcing(
        ebot=jnp.asarray(fields["power_bot"]),
        ecri=jnp.asarray(fields["power_cri"]),
        ensq=jnp.asarray(fields["power_nsq"]),
        esho=jnp.asarray(fields["power_sho"]),
        hbot=jnp.asarray(fields["scale_bot"]),
        hcri_inv=jnp.asarray(1.0 / fields["scale_cri"]),
    )
