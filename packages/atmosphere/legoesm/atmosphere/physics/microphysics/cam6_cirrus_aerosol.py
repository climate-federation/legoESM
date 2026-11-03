"""Prescribed modal aerosol for CAM6 cirrus nucleation: CAM's prescribed_aero MAM3 climatology.

Source: a CAM prescribed modal aerosol file as named by namelist_defaults_cam.xml:354-357 at
cam_cesm2_1_rel_60 (``mam3_1.9x2.5_L30_2000clim_c130319.nc``, a single-year cyclical climatology).  CAM's
prescribed_aero reads each interstitial field's ``_logm`` and ``_logv``, maps them with tracer_data
(horizontal lininterp, cyclic time interpolation, vert_interp) and sets the field once a day to
``min(exp(logm + randn std), 5 exp(logm), exp(logm + 3 std))`` with one normal deviate per day from
gfortran's generator seeded by the step number (prescribed_aero.F90:455-520).  KNOWN DIFFERENCE: legoESM
uses ``randn = 0``, the median ``exp(logm)`` (the generator cannot be reproduced).

The file is mapped once onto the run's cells by :func:`map_to_cells` (host, numpy, CAM tracer_data
horizontal weights from :mod:`cam6_aerosol_climatology`) and saved as an npz; at run time the two records
bracketing the step are held, interpolated in time (CAM cyclic fact1), rebuilt on the file's hybrid
pressures and vert_interp'd to the model levels, in log space, then exponentiated.

Fields needed by nucleate_ice_cam_calc for MAM3 (accum a1, aitken a2, coarse a3 holding dust, sea salt and
sulfate): num_a1, num_a2, num_a3 [1/kg]; dst_a3, ncl_a3, so4_a3 [kg/kg].
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.microphysics.cam6_aerosol_climatology import (
    bracket,
    horizontal_weights,
    interpolate_to_model,
    noleap_days_since_jan1,
    time_fact1,
    NOLEAP_ONE_YR,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.cam_lininterp import lininterp_2d

__physics_contract__ = {
    "summary": "CAM prescribed_aero MAM3 climatology for cirrus nucleation: mapping, time and vertical interpolation.",
    "inputs": {"lat": "rad", "lon": "rad", "pmid": "Pa"},
    "outputs": {"number": "1/kg", "mass": "kg/kg"},
    "sign_convention": "none (mixing ratios)",
    "conserves": ["none"],
    "differentiable": False,
    "reference": "CAM cam_cesm2_1_rel_60 prescribed_aero.F90, tracer_data.F90",
    "idealized_test": "tests/atmosphere/microphysics/unit/test_cam6_cirrus_aerosol.py",
}

CIRRUS_FIELDS = ("num_a1", "num_a2", "num_a3", "dst_a3", "ncl_a3", "so4_a3")


class CirrusAerosolSetup(NamedTuple):
    """Start-up constants: file hybrid coefficients, record times [noleap days since Jan 1], CAM top_lev."""
    hyam: np.ndarray
    hybm: np.ndarray
    p0: float
    record_times: tuple
    top_lev: int


def map_to_cells(nc_path, lat_rad, lon_rad):
    """Host: every ``<field>_logm`` and PS of the CAM file on the cells, ``{name: (12, [lev,] ncol)}``, plus
    hyam, hybm, P0 and the record dates."""
    from netCDF4 import Dataset
    with Dataset(nc_path) as d:
        d.set_auto_mask(False)
        lon_w, lat_w = horizontal_weights(np.asarray(d["lat"][:], np.float64), np.asarray(d["lon"][:], np.float64),
                                          np.asarray(lat_rad, np.float64), np.asarray(lon_rad, np.float64))
        out = {f"{v}_logm": lininterp_2d(np.asarray(d[f"{v}_logm"][:], np.float64), lon_w, lat_w)
               for v in CIRRUS_FIELDS}
        out["PS"] = lininterp_2d(np.asarray(d["PS"][:], np.float64), lon_w, lat_w)
        out["hyam"] = np.asarray(d["hyam"][:], np.float64)
        out["hybm"] = np.asarray(d["hybm"][:], np.float64)
        out["P0"] = np.asarray(float(d["P0"][...]))
        out["date"] = np.asarray(d["date"][:], np.int64)
    return out


def load_setup(npz_path, top_lev):
    z = np.load(npz_path)
    times = tuple(noleap_days_since_jan1(int(dd)) for dd in z["date"])
    return CirrusAerosolSetup(z["hyam"], z["hybm"], float(z["P0"]), times, int(top_lev))


def cirrus_aerosol_forcing(npz_path, setup, t, held=None, cells=None):
    """Host side, as cam6_aerosol_forcing: the two records bracketing model time ``t`` [noleap days since
    Jan 1], reloaded when the bracket moves, and this step's fact1.

    ``cells``: global indices of this process's cells (the Voronoi MPI partition's ``local_cells``, owned
    then halo); the file holds the global mesh, so each record is cut to those cells along its last axis.
    None (serial) keeps every cell, so serial runs are unchanged."""
    months = bracket(setup.record_times, t)
    if held is None or held[0] != months:
        z = np.load(npz_path)
        cut = (lambda a: a) if cells is None else (lambda a: a[..., np.asarray(cells)])
        rec = {k: jnp.asarray(cut(z[f"{k}_logm"][list(months)])) for k in CIRRUS_FIELDS}
        held = (months, {"rec": rec, "PS": jnp.asarray(cut(z["PS"][list(months)]))})
    n, np1 = months
    fact1 = time_fact1(setup.record_times[n], setup.record_times[np1], t, NOLEAP_ONE_YR)
    return held, {**held[1], "fact1": jnp.asarray(fact1)}


def cirrus_aerosol_at_model_levels(aer, setup, pmid):
    """Traced: ``{field: (ncol, nlev)}`` = exp of the time- and vertically interpolated logm."""
    def one(v):
        r = aer["rec"][v]
        return jnp.exp(interpolate_to_model(r[0].T, r[1].T, aer["PS"][0], aer["PS"][1], aer["fact1"],
                                            setup.hyam, setup.hybm, setup.p0, pmid))
    return {v: one(v) for v in CIRRUS_FIELDS}


__all__ = ["CIRRUS_FIELDS", "CirrusAerosolSetup", "map_to_cells", "load_setup", "cirrus_aerosol_forcing",
           "cirrus_aerosol_at_model_levels"]
