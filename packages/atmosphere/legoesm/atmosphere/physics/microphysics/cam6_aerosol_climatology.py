"""Prescribed CAM6 MAM4 aerosol climatology: PPE file loader and horizontal mapping.

Input (user decision 2026-10-16): CESM2.2-CAM6 PPE member 000 present-day monthly
timeseries (GDEX d651076, DOI 10.26024/BZNE-YF09, CC-BY-4.0), interstitial MAM4
species/number mixing ratios and PS, 36 monthly records averaged into one
12-month cycling year; hybrid coefficients from the member's first raw history
file.  All files are sha256-pinned and live outside the repo.

Horizontal mapping ports CAM6 ``tracer_data`` as used by ``prescribed_aero``
(``weight_by_lat`` false): file lon/lat in degrees times ``d2r``
(tracer_data.F90:306, :318), ``lininterp_init`` cycle on [0, 2pi) for longitude
and bndry for latitude (:1539-1540), ``lininterp3d2d``/``lininterp2d1d``.  No
``polar_average`` (CAM applies it only for the FV dycore, :1560).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.microphysics.cam6_modal_aerosol import (
    MAM4_SPECIES,
    Cam6ModeProps,
    read_pinned_netcdf,
)
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.cam_lininterp import lininterp_2d, lininterp_init

__physics_contract__ = {
    "summary": "CAM6 PPE prescribed-aerosol monthly climatology loader + CAM tracer_data horizontal mapping.",
    "inputs": {"lat": "rad", "lon": "rad"},
    "outputs": {"species": "kg/kg", "number": "1/kg", "PS": "Pa"},
    "sign_convention": "none (mixing ratios as on file)",
    "conserves": ["none"],
    "differentiable": False,
    "reference": "CAM6 tracer_data.F90 / interpolate_data.F90; PPE member 000 (Eidhammer et al. 2024)",
    "idealized_test": "gfortran build of verbatim lininterp_init/lininterp3d2d vs this port; "
                      "month-by-month read == all-at-once read",
}

# --- PPE member 000 files (manifest ppe000_sha256.txt, 2026-10-16) ----------
PPE_TIMESERIES = "cc_PPE_250_ensemble_PD.000.h0.{var}.nc"
PPE_RAW_FILE = "PPE_250_ensemble_PD.000.cam.h0.0001-01.nc"
PPE_NUMBER_FIELDS = tuple(f"num_a{m + 1}" for m in range(len(MAM4_SPECIES)))
PPE_SPECIES_FIELDS = tuple(f"{s}_a{m + 1}" for m, sp in enumerate(MAM4_SPECIES) for s in sp)
PPE_SHA256 = {
    "cc_PPE_250_ensemble_PD.000.h0.bc_a1.nc": "685dc8afab6fee22d14fa84a87d3fdced9475396ee36d508e0d70b8e2b8d28c6",
    "cc_PPE_250_ensemble_PD.000.h0.bc_a4.nc": "87ddf9666620eb6c6e7106773d6cfd1aca2da25243c06b1a8026c4d12d8587bc",
    "cc_PPE_250_ensemble_PD.000.h0.dst_a1.nc": "8fd9e9d22bdd28017fb80172055243a8b9529f9c67165a001ce19e5937f888bf",
    "cc_PPE_250_ensemble_PD.000.h0.dst_a2.nc": "98bc82b137ca4623d24905c028cc3c1cd21229015d007ef22d5c6030d38a01d7",
    "cc_PPE_250_ensemble_PD.000.h0.dst_a3.nc": "68e6442cc9b2ca627d9a04da182c35c489bfeba3538fc71cc424cc932dcc7591",
    "cc_PPE_250_ensemble_PD.000.h0.ncl_a1.nc": "58cffcc535e7ef64f03860b1f2851507646a75505d79b5c787592ad1fd513ace",
    "cc_PPE_250_ensemble_PD.000.h0.ncl_a2.nc": "2fa08efe5492f8c418d696efe724009f4538f44d56599a75d1d6bd37fa5409e0",
    "cc_PPE_250_ensemble_PD.000.h0.ncl_a3.nc": "8074380cc69cffdf8dba41590f33e9e35a30431f31d41619bc28ce760dd77808",
    "cc_PPE_250_ensemble_PD.000.h0.num_a1.nc": "57c730b6f18330df3bc5a13764a88e725794989050e0fb10f667b6aac6d0a91b",
    "cc_PPE_250_ensemble_PD.000.h0.num_a2.nc": "f47ceb2e2a2a159b4707adaf7cee0a752e79d4165497e7ead65937f51e905425",
    "cc_PPE_250_ensemble_PD.000.h0.num_a3.nc": "712237a12f1c1a92ab583a64f5326c6d2a387659937a6b74f71d5147a38b9054",
    "cc_PPE_250_ensemble_PD.000.h0.num_a4.nc": "eaa20066ba866a314dfcd221e40524f2ff766e34a14b669f2c5ba99560b08575",
    "cc_PPE_250_ensemble_PD.000.h0.pom_a1.nc": "0d9ddb381aac7267fbbb8ff44ea6c794faf3cb353a7ebe92a10f5eefe28fb815",
    "cc_PPE_250_ensemble_PD.000.h0.pom_a4.nc": "136a95e03aa13d0240985b2392bb9355afbeef045da5f31c95383c23d31054f1",
    "cc_PPE_250_ensemble_PD.000.h0.PS.nc": "aaf974baa6810457d1e75bbf8ddfd79a171bedcdab5f3b2ffd486806ce37c11a",
    "cc_PPE_250_ensemble_PD.000.h0.so4_a1.nc": "a8f741d4072d3ce07b1012815fba9b2489ff5f6420f95fe9a5c1fe55f6478050",
    "cc_PPE_250_ensemble_PD.000.h0.so4_a2.nc": "f6f84226a58157b0243c7ed131c70e8c5b79d661487f9c63c3bbd15744c6977b",
    "cc_PPE_250_ensemble_PD.000.h0.so4_a3.nc": "81b7055f773d49cd0812bb7c5671216cd414026c3c31ab7ca136b8798ec8d735",
    "cc_PPE_250_ensemble_PD.000.h0.soa_a1.nc": "d2b4581b2516eb960e4a2120d792fea73942b7772ae8bcb64fd8e3e4f1176fad",
    "cc_PPE_250_ensemble_PD.000.h0.soa_a2.nc": "f8235eb34efc402fb4548dd6b75eded65889251a5a5ea8c4a13ce2c1168959c8",
    PPE_RAW_FILE: "0662883d71037df480c86b6a91c28bffdab058cd77ffcdd5d919383ad35994f2",
}
# --- record dates: CAM clim-file convention (user decision 1b, 2026-10-16) ---
# ``date`` of mam3_1.9x2.5_L30_2000clim_c130319.nc (no ``datesec`` -> 0), the
# m-th month of the PPE mean is stamped with the m-th date; as days since Jan 1
# 00:00 on the noleap calendar (set_time_float_from_date differences).
CAM_CLIM_DATES = (20000116, 20000214, 20000316, 20000415, 20000516, 20000615,
                  20000716, 20000816, 20000915, 20001016, 20001115, 20001216)
_NOLEAP_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
# one_yr as CAM computes it (tracer_data.F90:265-275: Jan 1 of year 2 minus Jan 1 of
# year 1) on the model's only calendar, noleap (legoesm.forcing.time_utils, a layer
# above atmosphere, so the month table is restated here; a test pins them equal).
NOLEAP_ONE_YR = float(sum(_NOLEAP_MONTH_DAYS))


def noleap_days_since_jan1(date: int, sec: int = 0) -> float:
    """Days since Jan 1 00:00 of the date's year (noleap) for YYYYMMDD ``date``."""
    month, day = (date % 10000) // 100, date % 100  # coeff-ok: YYYYMMDD digit split (open_trc_datafile)
    return float(sum(_NOLEAP_MONTH_DAYS[:month - 1]) + day - 1) + sec / 86400.0  # coeff-ok: seconds per day


CLIM_RECORD_TIMES = tuple(noleap_days_since_jan1(d) for d in CAM_CLIM_DATES)
# --- record layout: 36 monthly means = 3 years x 12 months (user decision) ---
_N_MONTHS = 12
_N_YEARS = 3


def read_monthly_climatology(data_dir, var: str, months=range(12)):
    """Mean over the 3 years of one PPE field, read month by month.

    Returns ``(field, lat_deg, lon_deg, lev_hPa)``; ``field`` float64 of shape
    (len(months), [lev,] lat, lon), 0-based ``months`` (CAM holds only the two
    bracketing records, user decision 2b: pass ``months=(n, np1)``).  Month ``m`` is the mean of records m, m+12, m+24,
    each cast float32 -> float64 (as CAM's pio read into r8), summed in that
    order and divided by 3.  Raw values are read (no netCDF masking, as pio).
    """
    path = Path(data_dir) / PPE_TIMESERIES.format(var=var)
    with read_pinned_netcdf(path, PPE_SHA256) as d:
        d.set_auto_mask(False)
        v = d[var]
        if v.shape[0] != _N_YEARS * _N_MONTHS:
            raise ValueError(f"{path}: {v.shape[0]} records, expected {_N_YEARS * _N_MONTHS}")
        months = tuple(months)
        if any(m not in range(_N_MONTHS) for m in months):
            raise ValueError(f"months must be 0-based month indices 0..11, got {months}")
        out = np.empty((len(months),) + v.shape[1:], dtype=np.float64)
        for i, m in enumerate(months):
            r = [np.asarray(v[m + _N_MONTHS * y], dtype=np.float64) for y in range(_N_YEARS)]
            out[i] = (r[0] + r[1] + r[2]) / 3.0
        coords = tuple(np.asarray(d[c][:], dtype=np.float64) for c in ("lat", "lon"))
        lev = np.asarray(d["lev"][:], dtype=np.float64) if "lev" in d.variables else None
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{path}: non-finite values in {var}")
    return out, coords[0], coords[1], lev


def read_hybrid_coefficients(data_dir):
    """``(hyam, hybm, P0 [Pa])`` float64 from the pinned PPE raw history file."""
    with read_pinned_netcdf(Path(data_dir) / PPE_RAW_FILE, PPE_SHA256) as d:
        d.set_auto_mask(False)
        return (np.asarray(d["hyam"][:], dtype=np.float64), np.asarray(d["hybm"][:], dtype=np.float64),
                float(d["P0"][...]))


def horizontal_weights(file_lat_deg, file_lon_deg, lat_rad, lon_rad):
    """CAM tracer_data lon (cycle) and lat (bndry) ``lininterp_init`` weights
    for model columns at ``lat_rad``, ``lon_rad`` (any range; reduced to
    [0, 2pi) as CAM's ``rlon``)."""
    d2r = np.pi / 180.0  # coeff-ok: mo_constants d2r
    twopi = 2.0 * np.pi
    lon = np.mod(np.asarray(lon_rad, dtype=np.float64), twopi)
    lon_w = lininterp_init(np.asarray(file_lon_deg, dtype=np.float64) * d2r, lon, 2, 0.0, twopi)
    lat_w = lininterp_init(np.asarray(file_lat_deg, dtype=np.float64) * d2r, lat_rad, 1)
    return lon_w, lat_w



def load_bracket_records(data_dir, months, lon_w, lat_w, fields=None):
    """The two held records (user decision 2b) of every PPE field mapped to the
    model columns: ``{var: (2, lev, ncol)}`` and ``{"PS": (2, ncol)}``, for
    0-based ``months = (n, np1)`` from :func:`bracket`.  ``fields`` defaults to
    PS + all species + all mode numbers."""
    if fields is None:
        fields = ("PS",) + PPE_SPECIES_FIELDS + PPE_NUMBER_FIELDS
    return {var: lininterp_2d(read_monthly_climatology(data_dir, var, months)[0], lon_w, lat_w)
            for var in fields}


def bracket(times, t: float):
    """CAM cyclic ``find_times`` (tracer_data.F90:1070-1077, findplb): 0-based
    record indices (n, np1) bracketing model time ``t`` [days since Jan 1]."""
    nx = len(times)
    if t < times[0] or t >= times[nx - 1]:
        n = nx - 1
    else:
        n = next(i - 1 for i in range(1, nx) if t < times[i])
    return n, (0 if n == nx - 1 else n + 1)


def time_fact1(datatimem, datatimep, t, one_yr):
    """CAM ``interpolate_trcdata`` fact1 for a cyclical file (:1636-1645)."""
    deltat = datatimep - datatimem
    wrap = deltat < 0.0
    deltat_w = deltat + one_yr
    f_wrap = jnp.where(datatimep >= t, (datatimep - t) / deltat_w, (datatimep + one_yr - t) / deltat_w)
    return jnp.where(wrap, f_wrap, (datatimep - t) / deltat)


def _time_interp(fact1, a, b):
    """``fact1*a + fact2*b`` with CAM's ``fact2 == 0`` branch (:1666-1670, :1724-1728)."""
    fact2 = 1.0 - fact1
    return jnp.where(fact2 == 0, fact1 * a, fact1 * a + fact2 * b)


def vert_interp(pin, pmid, datain):
    """CAM ``vert_interp`` (tracer_data.F90:2291-2361) on the trailing level axis:
    ``pin`` (..., levsiz) increasing, ``pmid`` (..., pver), ``datain`` (..., levsiz)."""
    levsiz = pin.shape[-1]
    j = jnp.clip(jnp.sum(pin[..., None, :] < pmid[..., :, None], axis=-1) - 1, 0, levsiz - 2)
    p_lo = jnp.take_along_axis(pin, j, axis=-1)
    p_hi = jnp.take_along_axis(pin, j + 1, axis=-1)
    d_lo = jnp.take_along_axis(datain, j, axis=-1)
    d_hi = jnp.take_along_axis(datain, j + 1, axis=-1)
    dpu = pmid - p_lo
    dpl = p_hi - pmid
    interior = (d_lo * dpl + d_hi * dpu) / (dpl + dpu)
    above = datain[..., :1] * pmid / pin[..., :1]
    return jnp.where(pmid < pin[..., :1], above, jnp.where(pmid > pin[..., -1:], datain[..., -1:], interior))


def interpolate_to_model(rec_m, rec_p, ps_m, ps_p, fact1, hyam, hybm, p0, pmid):
    """Time-interpolate the two held records and PS, build the file pressures
    ``pin = P0*hyam + ps*hybm`` and ``vert_interp`` to ``pmid`` (:1660-1745).
    ``rec_*`` (ncol, levsiz), ``ps_*`` (ncol,), ``pmid`` (ncol, pver)."""
    ps = _time_interp(fact1, ps_m, ps_p)
    datain = _time_interp(fact1, rec_m, rec_p)
    pin = p0 * hyam + ps[..., None] * hybm
    return vert_interp(pin, pmid, datain)


class Cam6AerosolActivation(NamedTuple):
    """Start-up constants of the prescribed-aerosol droplet activation: mode
    properties, the PPE file's hybrid coefficients, CAM's ``trop_cloud_top_lev``
    (0-based) and the static mixing-substep bound ``n_max``."""
    props: Cam6ModeProps
    hyam: np.ndarray
    hybm: np.ndarray
    p0: float
    top_lev: int
    n_max: int


def trop_cloud_top_lev(pref_mid, trop_cloud_top_press: float) -> int:
    """0-based ``ref_pres`` ``press_lim_idx(trop_cloud_top_press, top=.true.)``
    (ref_pres.F90:126, :150-160): first level with ``pref_mid > p``, else ``pver``."""
    above = np.nonzero(np.asarray(pref_mid) > trop_cloud_top_press)[0]
    return int(above[0]) if above.size else len(pref_mid)


def aerosol_at_model_levels(aer: dict, setup: Cam6AerosolActivation, pmid):
    """Per-step ``forcing["cam6_aerosol"]`` (``{"rec": {var: (2, lev, ncol)},
    "PS": (2, ncol), "fact1": ()}``) time- and vertically interpolated to
    ``pmid`` (ncol, nlev): ``(aer_num (nmode, ncol, nlev) [1/kg], aer_mass)``,
    ``aer_mass`` a per-mode tuple of (nspec, ncol, nlev) [kg/kg] in MAM4_SPECIES order."""
    def one(var):
        r = aer["rec"][var]
        return interpolate_to_model(r[0].T, r[1].T, aer["PS"][0], aer["PS"][1], aer["fact1"],
                                    setup.hyam, setup.hybm, setup.p0, pmid)
    num = jnp.stack([one(v) for v in PPE_NUMBER_FIELDS])
    mass = tuple(jnp.stack([one(f"{s}_a{m + 1}") for s in sp]) for m, sp in enumerate(MAM4_SPECIES))
    return num, mass


def cam6_aerosol_forcing(data_dir, t: float, lon_w, lat_w, held=None):
    """Host side of CAM ``prescribed_aero_adv``: the two records bracketing model
    time ``t`` [noleap days since Jan 1], reloaded only when the bracket moves
    (user decision 2026-10-24, 2a), plus this step's ``fact1``.  Returns
    ``(held, forcing_entry)``; pass ``held`` back on the next call."""
    months = bracket(CLIM_RECORD_TIMES, t)
    if held is None or held[0] != months:
        recs = load_bracket_records(data_dir, months, lon_w, lat_w)
        held = (months, {"rec": {k: jnp.asarray(v) for k, v in recs.items() if k != "PS"},
                         "PS": jnp.asarray(recs["PS"])})
    n, np1 = months
    fact1 = time_fact1(CLIM_RECORD_TIMES[n], CLIM_RECORD_TIMES[np1], t, NOLEAP_ONE_YR)
    return held, {**held[1], "fact1": jnp.asarray(fact1)}
