"""NEMO restart + mesh_mask reader for the differentiable-NEMO fidelity harness.

The NEMO counterpart of :mod:`mitgcm_io`. NEMO writes plain NetCDF (via the
native ``iom_nf90`` path — no XIOS needed), so this is a thin xarray reader; the
only conventions it reconciles are:

* **Halo strip.** NEMO global arrays carry an ``nn_hls``-cell halo on every side
  (``nn_hls=1`` for GYRE: a 30x20 physical domain is stored 32x22). The physical
  interior is ``[nn_hls:-nn_hls, nn_hls:-nn_hls]``.
* **Axis order.** NEMO 3-D fields are ``(z, y, x)`` on disk; legoESM wants the
  vertical LAST — ``(y=lat, x=lon, z=lev)`` — a single ``moveaxis(0, -1)``.
* **Vertical order.** NEMO ``k=1`` is the surface, ``k`` increasing downward —
  the SAME top-down order legoESM uses, so there is NO z-reversal.

This module only READS + reshapes into plain NumPy; the staggering / grid
construction (NEMO east-face ``U`` -> legoESM ``u_face``, the beta-plane
geometry) lives in :mod:`nemo_state_bridge`, per the harness/model split in
``docs/ocean/fidelity/oracle_recipe_strategy.md``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import xarray as xr


class NemoGrid(NamedTuple):
    """Halo-stripped NEMO horizontal+vertical grid from a ``mesh_mask.nc``.

    Horizontal fields are ``(n_lat, n_lon)``; 3-D masks are ``(n_lat, n_lon,
    nlev)``; 1-D vertical fields are ``(nlev,)``. All in NumPy float64.
    """
    glamt: np.ndarray        # T-point longitude [deg] (n_lat, n_lon)
    gphit: np.ndarray        # T-point latitude  [deg]
    e1t: np.ndarray          # zonal T-cell width  [m]
    e2t: np.ndarray          # merid T-cell height [m]
    e1u: np.ndarray          # zonal U-cell width  [m]
    e2v: np.ndarray          # merid V-cell height [m]
    ff_t: np.ndarray         # Coriolis at T-points [1/s]
    ff_f: np.ndarray         # Coriolis at F-points (NE corner) [1/s]
    e3t_1d: np.ndarray       # reference T-cell thickness [m] (nlev,)
    gdept_1d: np.ndarray     # reference T-point depth [m] (nlev,)
    gdepw_1d: np.ndarray     # reference W-point (interface) depth [m] (nlev,)
    tmask: np.ndarray        # T-point wet mask (n_lat, n_lon, nlev)
    umask: np.ndarray        # U-point wet mask
    vmask: np.ndarray        # V-point wet mask


class NemoState(NamedTuple):
    """Halo-stripped NEMO prognostic state from a restart file.

    Tracers/density are ``(n_lat, n_lon, nlev)`` at T-points; velocities are at
    their NEMO faces (still T-shaped here — the face-index convention is applied
    in the bridge); ``ssh`` is ``(n_lat, n_lon)``.
    """
    T: np.ndarray            # potential temperature [degC]
    S: np.ndarray            # practical salinity [PSU]
    u: np.ndarray            # zonal velocity at NEMO U-points (east face)
    v: np.ndarray            # meridional velocity at NEMO V-points (north face)
    ssh: np.ndarray          # sea-surface height [m]
    rhd: np.ndarray | None   # in-situ density anomaly (rho-rho0)/rho0, if dumped


def _check_hls(nn_hls: int) -> None:
    # a[h:-h] silently returns an empty slice for h=0, so a no-halo file would
    # have its whole interior chopped. Require an explicit halo.
    if nn_hls < 1:
        raise ValueError(
            f"nn_hls must be >= 1 (NEMO always writes a >=1-cell halo); got "
            f"{nn_hls}. Pass the file's actual halo width."
        )


def _strip_halo_2d(a: np.ndarray, nn_hls: int) -> np.ndarray:
    _check_hls(nn_hls)
    h = nn_hls
    a = np.asarray(a, dtype=np.float64)
    out = a[h:-h, h:-h]
    assert out.shape == (a.shape[0] - 2 * h, a.shape[1] - 2 * h)
    return out


def _to_latlon_lev(a: np.ndarray, nn_hls: int) -> np.ndarray:
    """``(z, y, x)`` with halo -> ``(y-h, x-h, z)`` interior, vertical last."""
    _check_hls(nn_hls)
    a = np.asarray(a, dtype=np.float64)
    h = nn_hls
    return np.moveaxis(a[:, h:-h, h:-h], 0, -1)


def read_nemo_mesh_mask(path: str, *, nn_hls: int = 1) -> NemoGrid:
    """Read + halo-strip a NEMO ``mesh_mask.nc`` into a :class:`NemoGrid`."""
    m = xr.open_dataset(path, decode_times=False)

    def h2(name: str) -> np.ndarray:
        return _strip_halo_2d(np.asarray(m[name].values).squeeze(), nn_hls)

    def m3(name: str) -> np.ndarray:
        return _to_latlon_lev(np.asarray(m[name].values).squeeze(), nn_hls)

    def v1(name: str) -> np.ndarray:
        return np.asarray(m[name].values).ravel().astype(np.float64)

    return NemoGrid(
        glamt=h2("glamt"), gphit=h2("gphit"),
        e1t=h2("e1t"), e2t=h2("e2t"), e1u=h2("e1u"), e2v=h2("e2v"),
        ff_t=h2("ff_t"), ff_f=h2("ff_f"),
        e3t_1d=v1("e3t_1d"), gdept_1d=v1("gdept_1d"), gdepw_1d=v1("gdepw_1d"),
        tmask=m3("tmask"), umask=m3("umask"), vmask=m3("vmask"),
    )


def read_nemo_restart(path: str, *, nn_hls: int = 1) -> NemoState:
    """Read + halo-strip a NEMO restart. Uses the ``tn/sn/un/vn`` (Kbb) fields.

    ``rhd`` is included when the restart carries it (the MY_SRC EOS dump);
    otherwise ``None``.
    """
    r = xr.open_dataset(path, decode_times=False)

    def m3(name: str) -> np.ndarray:
        return _to_latlon_lev(np.asarray(r[name].values).squeeze(), nn_hls)

    return NemoState(
        T=m3("tn"), S=m3("sn"), u=m3("un"), v=m3("vn"),
        ssh=_strip_halo_2d(np.asarray(r["sshn"].values).squeeze(), nn_hls),
        rhd=(m3("rhd") if "rhd" in r else None),
    )


__all__ = ("NemoGrid", "NemoState", "read_nemo_mesh_mask", "read_nemo_restart")
