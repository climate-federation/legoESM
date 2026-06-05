"""Climate-scale ocean diagnostics for multi-decade evaluation.

Thin wrappers around the existing streamfunction module + tracer-budget
helpers that map directly onto the OMIP-2 / Bryan-1987 acceptance bars:

* :func:`amoc_at_latitude` -- maximum Atlantic-Meridional-Overturning
  streamfunction at a target latitude (default 26.5 deg N), reduced
  from the full MOC streamfunction returned by
  ``legoesm.ocean.diagnostics_streamfunction.moc_streamfunction``.
* :func:`acc_transport` -- total zonal volume transport through a
  meridional section spanning the Drake passage band (default 65-45
  deg S), reduced from ``barotropic_streamfunction``.
* :func:`sst_climatology_bias` -- area-weighted RMSE + bias of model
  SST relative to a reference climatology (WOA / HadISST). Reference
  field is passed in as a (n_lat, n_lon) array so we don't pull a
  big dataset into legoESM proper; the bulletproof reports load WOA
  separately.

The acceptance bars used by the Phase F long-run validation:

* AMOC @ 26.5 deg N -- 15 +/- 3 Sv (Tsujino 2020 ensemble; Cunningham
  et al. 2007 RAPID observation).
* ACC transport @ Drake -- 130 +/- 15 Sv (Donohue 2016).
* SST bias -- < 1.5 deg C globally vs WOA climatology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


_SV: float = 1.0e6   # 1 Sverdrup = 1e6 m^3/s


@dataclass(frozen=True)
class AmocResult:
    target_lat_deg: float
    streamfunction_Sv: float
    depth_of_max_m: float


@dataclass(frozen=True)
class AccResult:
    drake_lat_south_deg: float
    drake_lat_north_deg: float
    transport_Sv: float


@dataclass(frozen=True)
class SstBiasResult:
    bias_K: float
    rmse_K: float
    n_cells: int


def amoc_at_latitude(moc_streamfunction_m3s: np.ndarray,
                     lat_deg: np.ndarray,
                     depth_m: np.ndarray,
                     *,
                     target_lat: float = 26.5,
                     basin_mask: Optional[np.ndarray] = None,
                     ) -> AmocResult:
    """Max of the meridional overturning streamfunction at ``target_lat``.

    Parameters
    ----------
    moc_streamfunction_m3s : ndarray (n_lat, n_depth)
        Pre-computed MOC streamfunction (e.g. from
        ``diagnostics_streamfunction.moc_streamfunction`` summed over
        the requested basin).
    lat_deg : ndarray (n_lat,)
        Latitudes in degrees, monotonically increasing.
    depth_m : ndarray (n_depth,)
        Depth axis (positive downward) in meters.
    target_lat : float
        Latitude at which to evaluate the maximum (default 26.5 N).
    basin_mask : ndarray (n_lat, n_depth) or None
        Optional weighting to restrict to the Atlantic basin; passed
        through unchanged.
    """
    if basin_mask is not None:
        psi = np.asarray(moc_streamfunction_m3s) * np.asarray(basin_mask)
    else:
        psi = np.asarray(moc_streamfunction_m3s)
    lat_arr = np.asarray(lat_deg)
    j = int(np.argmin(np.abs(lat_arr - target_lat)))
    profile = psi[j, :]
    k_max = int(np.argmax(profile))
    return AmocResult(
        target_lat_deg=float(lat_arr[j]),
        streamfunction_Sv=float(profile[k_max]) / _SV,
        depth_of_max_m=float(depth_m[k_max]),
    )


def acc_transport(barotropic_streamfunction_m3s: np.ndarray,
                  lat_deg: np.ndarray,
                  *,
                  drake_lat_south: float = -65.0,
                  drake_lat_north: float = -45.0,
                  ) -> AccResult:
    """Drake-passage zonal volume transport.

    The barotropic streamfunction ``psi`` is constant along streamlines;
    the difference ``psi(lat_north) - psi(lat_south)`` at a meridional
    section gives the total zonal volume flux through that section.
    Reduces the (n_lat, n_lon) barotropic streamfunction to a scalar
    by taking the max-minus-min within the Drake band.
    """
    lat_arr = np.asarray(lat_deg)
    psi = np.asarray(barotropic_streamfunction_m3s)
    band_mask = (lat_arr >= drake_lat_south) & (lat_arr <= drake_lat_north)
    if not band_mask.any():
        return AccResult(
            drake_lat_south_deg=drake_lat_south,
            drake_lat_north_deg=drake_lat_north,
            transport_Sv=float("nan"),
        )
    band_psi = psi[band_mask, :]
    transport = float(band_psi.max() - band_psi.min())
    return AccResult(
        drake_lat_south_deg=drake_lat_south,
        drake_lat_north_deg=drake_lat_north,
        transport_Sv=transport / _SV,
    )


def sst_climatology_bias(sst_model_K: np.ndarray,
                          sst_ref_K: np.ndarray,
                          area_m2: np.ndarray,
                          mask: Optional[np.ndarray] = None,
                          ) -> SstBiasResult:
    """Area-weighted bias + RMSE of model SST vs a reference climatology.

    ``sst_model_K``, ``sst_ref_K``, ``area_m2`` and ``mask`` (optional)
    must all share the same ``(n_lat, n_lon)`` shape.
    """
    model = np.asarray(sst_model_K, dtype=np.float64)
    ref = np.asarray(sst_ref_K, dtype=np.float64)
    A = np.asarray(area_m2, dtype=np.float64)
    if mask is None:
        m = np.ones_like(model)
    else:
        m = np.asarray(mask, dtype=np.float64)
    diff = (model - ref) * m
    total = float((A * m).sum())
    if total <= 0.0:
        return SstBiasResult(bias_K=float("nan"), rmse_K=float("nan"),
                              n_cells=0)
    bias = float((diff * A).sum() / total)
    rmse = float(np.sqrt((diff ** 2 * A).sum() / total))
    return SstBiasResult(
        bias_K=bias, rmse_K=rmse, n_cells=int(m.sum()),
    )


__all__ = [
    "AmocResult", "AccResult", "SstBiasResult",
    "amoc_at_latitude", "acc_transport", "sst_climatology_bias",
]
