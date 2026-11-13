"""IFS aerosol inputs to ecRad, OpenIFS 49r1 (8f6f722) NAERMACC=1 arm (JAX port).

The chain the IFS runs before RADIATION_SCHEME, for the 49r1 code defaults
except the climatology dimensionality (declared departure, vote A2a: the 48r1
3D file lacks the "lev"/"pressure" variables the 49r1 3D reader needs, so
LAER3D=F and the 2D column-mass file aerosol_cams_climatology_43R3a.nc):

* :func:`month_and_year`: updecaec.F90:140-192 time weight (PMONTH, Jan 15 = 1).
* :func:`load_cams_2d`, :func:`update_2d`: yoeaerc.F90 SU_CLIMATOLOGY2D and
  UPDATE (2D arm), with the suecrad.F90:2656-2667 scale heights.
* :func:`interp_points`, :func:`calc_profiles_2d`: yoeaerc.F90
  GET_INTERP_POINTS and CALC_PROFILES (2D arm): column mass bilinear in
  (lon, sin lat), distributed with the eta^(8434/H) cumulative fraction.
* :func:`background_aerosol`: radact.F90:657-787, the two Tegen-array slots
  the NAERMACC=1 arm uses (1: tropospheric background, 6: stratospheric
  background; LHVOLCA=F, NVOLCVERT=0), floored at REPAER.
* :func:`radintg_aerosol`: radintg.F90:1355-1430 assembly of PAEROSOL (11
  climatological species, 0 for the 2 parametric ones) and PAEROSOL_OLD.
* :func:`macc_mixing_ratio`: radiation_scheme.F90:545-575, max(PAEROSOL, 0)
  plus the backgrounds over (layer mass x 550-nm dry mass extinction).

Horizontal and time interpolation act on static column positions and dates
(numpy, set-up time).  Traced: pressure, temperature, mixing ratios.
Smoothing (votes A10a, B4a, C1-C3): one static ``sharpness`` (``math.inf`` =
the oracle's hard form exactly), helpers of :mod:`.two_stream`.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.radiation.ecrad_ifs.two_stream import (
    blend, lower_clamp, step, upper_clamp)

__physics_contract__ = {
    "summary": (
        "OpenIFS 49r1 NAERMACC=1 aerosol inputs to ecRad: CAMS 2D climatology "
        "(time and bilinear interpolation, scale-height vertical distribution), "
        "radact tropospheric/stratospheric background, radiation_scheme "
        "mixing-ratio assembly."
    ),
    "inputs": {
        "pressure_hl": "Pa (ncol, nlev+1) top-down",
        "temperature_hl": "K (ncol, nlev+1)",
        "eta_hl": "1 (nlev+1) reference-atmosphere p/p_surface (GPHPRE, VP00)",
        "sinlat, lon": "1, rad (ncol,) static column positions",
        "nindat, minute": "YYYYMMDD, minutes since 00 UTC of nindat (static)",
    },
    "outputs": {
        "paerosol": "kg/kg (ncol, nlev, 13) climatological species",
        "paer1, paer6": "1 (ncol, nlev) background 550-nm layer optical depth (Tegen slots 1, 6)",
        "mixing_ratio": "kg/kg (ncol, nlev, 13) ecRad aerosol mixing ratio",
    },
    "sign_convention": "mixing ratios and optical depths >= 0 (floored).",
    "conserves": ["tracer"],   # 2D arm: sum over layers of mmr*dp/g = interpolated column mass
    "differentiable": True,
    "reference": (
        "OpenIFS 49r1 (8f6f722) arpifs/module/yoeaerc.F90, phys_ec/updecaec.F90, "
        "phys_radi/radact.F90, phys_radi/radintg.F90, phys_radi/radiation_scheme.F90, "
        "phys_radi/suecrad.F90, phys_ec/suaerv.F90; Bozzo et al. (2020) GMD 13."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_aerosol_oracle.py: matches the 49r1 "
        "harness dumps of PAERO, PAER and the ecRad mixing ratio to 1e-12."
    ),
}

# --- species seen by radiation (su_radiation_scheme.F90:365-383) ---
CLIM_SPECIES = (
    ("Sea_Salt_bin1", "SS"), ("Sea_Salt_bin2", "SS"), ("Sea_Salt_bin3", "SS"),
    ("Mineral_Dust_bin1", "DD"), ("Mineral_Dust_bin2", "DD"), ("Mineral_Dust_bin3", "DD"),
    ("Organic_Matter_hydrophilic", "OM"), ("Organic_Matter_hydrophobic", "OM"),
    ("Black_Carbon_hydrophilic", "BC"), ("Black_Carbon_hydrophobic", "BC"),
    ("Sulfates", "SU"),
)
N_TYPES = len(CLIM_SPECIES) + 2          # + parametric trop/strat backgrounds

# --- scale heights (m): yoeaerc.F90:274 default, suecrad.F90:815-819, 2656-2667 ---
_SCALE_HEIGHT_DEFAULT_M = 2000.0
_SCALE_HEIGHT_M = {"SS": 1000.0, "OM": 2000.0, "BC": 1000.0, "SU": 4000.0}
_DD_SCALE_HEIGHT_MON_M = (1000.0, 1000.0, 2000.0, 2000.0, 2000.0, 3000.0,
                          3000.0, 2000.0, 2000.0, 2000.0, 1000.0, 1000.0)
_RPRESSURE_SCALE_HEIGHT_M = 8434.0       # yoeaerc.F90:145

# --- updecaec.F90:87-93 (minutes from Jan 1 00 UTC to the 15th of each month) ---
_ZYTIME = (22320.0, 64980.0, 107640.0, 151560.0, 195480.0, 239400.0,
           283320.0, 327960.0, 371880.0, 415800.0, 459720.0, 503640.0)
_ZMDAY = (31.0, 59.25, 90.25, 120.25, 151.25, 181.25,
          212.25, 243.25, 273.25, 304.25, 334.25, 365.25)
_MIN_PER_DAY = 1440.0
_FEB_MIN_PER_DAY = 1452.8571             # updecaec.F90:145
_JAN_MIN = 44640.0                       # 31 days, updecaec.F90:145,157
_YEAR_MIN = 525960.0                     # updecaec.F90:149
_FEB_IDY_CAP = 28                        # updecaec.F90:144 (29 Feb treated as 28 Feb)
_YYYYMMDD_YEAR = 10000                   # fcttim.func.h NCCAA = KGRDAT/10000
_DEC15_TO_JAN15_END_MIN = 548280.0       # updecaec.F90:161

# --- background aerosol (suecrad.F90:822-823, suaerv.F90:152-158, radact.F90) ---
_TRBKG = 0.0534                          # suecrad.F90:822
_STBKG = 0.0045                          # suecrad.F90:823
_RCTRBGA = _TRBKG / (101325.0 - 19330.0)  # suaerv.F90:152  [1/Pa]
_RCSTBGA = _STBKG / 19330.0              # suaerv.F90:154  [1/Pa]
_RCTRPT = 30.0                           # suaerv.F90:158
_P_STRAT_TOP_PA = 999.0                  # radact.F90:728
_REPAER = 1.0e-12                        # yoeaerc_tegen.F90:389
_RPI = 2.0 * math.asin(1.0)              # YOMCST RPI
# min(1, T(k)/T(k+1)) smoothing scale (vote D1a): 1/RCTRPT, because the clamped
# ratio is raised to the power RCTRPT and compounded down the column (scale 1
# measured +178 % on the stratospheric background column at sharpness 100).
T_RATIO_CLAMP_SCALE = 1.0 / _RCTRPT
MMR_CLAMP_SCALE = 1.0e-9                 # kg/kg, max(PAEROSOL, 0) smoothing scale (vote B4a, interim)


class MaccAerosol(NamedTuple):
    """RADIATION_SCHEME aerosol inputs of the NAERMACC=1 arm: PAEROSOL (ncol, nlev, 13)
    and PAEROSOL_OLD slots 1 and 6 (ncol, nlev)."""
    paerosol: jnp.ndarray
    paer1: jnp.ndarray
    paer6: jnp.ndarray


class Cams2D(NamedTuple):
    """SU_CLIMATOLOGY2D state: column mass (nlon, nlat, 12, naer) kg/m2, grid in rad."""
    column_mass_mon: np.ndarray
    scale_height_mon: np.ndarray          # (12, naer) m
    lat1: float
    dlat: float
    lon1: float
    dlon: float


def load_cams_2d(path: str) -> Cams2D:
    """yoeaerc.F90 SU_CLIMATOLOGY2D + SET_SCALE_HEIGHT/SET_VARIABLE_SCALE_HEIGHT
    (suecrad.F90:2656-2667).  The file must hold every species of :data:`CLIM_SPECIES`."""
    import netCDF4

    with netCDF4.Dataset(path) as f:
        f.set_auto_mask(False)
        lat = np.asarray(f["lat"][:], np.float64)
        lon = np.asarray(f["lon"][:], np.float64)
        cm = [np.transpose(np.asarray(f[name][:], np.float64), (2, 1, 0))   # (lon, lat, month)
              for name, _ in CLIM_SPECIES]
    sh = np.full((12, len(CLIM_SPECIES)), _SCALE_HEIGHT_DEFAULT_M)
    for ja, (_, kind) in enumerate(CLIM_SPECIES):
        if kind in _SCALE_HEIGHT_M:
            sh[:, ja] = _SCALE_HEIGHT_M[kind]
        if kind == "DD":
            sh[:, ja] = _DD_SCALE_HEIGHT_MON_M
    return Cams2D(np.stack(cm, axis=-1), sh,
                  lat[0] * _RPI / 180.0, (lat[1] - lat[0]) * _RPI / 180.0,
                  lon[0] * _RPI / 180.0, (lon[1] - lon[0]) * _RPI / 180.0)


def month_and_year(nindat: int, minute: int):
    """updecaec.F90:140-174 (NCMIPFIXYR = 0): returns (IYR, PMONTH = IM1+1-ZTIMI)."""
    idy = nindat % 100 - 1                 # NDD - 1 (fcttim.func.h)
    imn = (nindat // 100) % 100            # NMM
    if imn == 1:
        xtime = idy * _MIN_PER_DAY + minute
    elif imn == 2:
        if idy == _FEB_IDY_CAP:
            idy -= 1
        xtime = _JAN_MIN + idy * _FEB_MIN_PER_DAY + minute
    else:
        xtime = (_ZMDAY[imn - 2] + idy) * _MIN_PER_DAY + minute
    xtime = math.fmod(xtime, _YEAR_MIN)
    if xtime <= _ZYTIME[0]:
        im1, timi = 12, (_ZYTIME[0] - xtime) / _JAN_MIN
    elif xtime > _ZYTIME[11]:
        im1, timi = 12, (_DEC15_TO_JAN15_END_MIN - xtime) / _JAN_MIN
    else:
        im1 = next(m for m in range(1, 12) if _ZYTIME[m - 1] < xtime <= _ZYTIME[m])
        timi = (xtime - _ZYTIME[im1]) / (_ZYTIME[im1 - 1] - _ZYTIME[im1])
    return nindat // _YYYYMMDD_YEAR, im1 + 1.0 - timi


def update_2d(clim: Cams2D, pmonth: float, eta_hl):
    """yoeaerc.F90 UPDATE, 2D arm: (column mass (nlon, nlat, naer), cum fraction (nlev+1, naer))."""
    w = pmonth - int(pmonth)
    im1 = (int(pmonth) - 1) % 12           # zero-based MOD(INT(PMONTH)-1,12)
    im2 = int(pmonth) % 12
    cm = (1.0 - w) * clim.column_mass_mon[:, :, im1, :] + w * clim.column_mass_mon[:, :, im2, :]
    sh = (1.0 - w) * clim.scale_height_mon[im1, :] + w * clim.scale_height_mon[im2, :]
    eta = np.asarray(eta_hl, np.float64)
    cum = np.stack([eta ** (_RPRESSURE_SCALE_HEIGHT_M / sh[ja]) for ja in range(sh.shape[0])], -1)
    return cm, cum


def interp_points(clim: Cams2D, column_mass, sinlat, lon):
    """GET_INTERP_POINTS + the CALC_PROFILES bilinear step (2D arm):
    column mass at each column (ncol, naer).  Static numpy."""
    sinlat = np.asarray(sinlat, np.float64)
    lon = np.asarray(lon, np.float64)
    nlon, nlat = column_mass.shape[:2]
    lonw = 1.0 + (lon - clim.lon1) / clim.dlon
    ilon1 = np.maximum(1, np.minimum(np.trunc(lonw).astype(np.int64), nlon))
    ilon2 = ilon1 % nlon + 1
    lonw = lonw - ilon1
    latw = 1.0 + (np.arcsin(sinlat) - clim.lat1) / clim.dlat
    ilat1 = np.maximum(0, np.minimum(np.trunc(latw).astype(np.int64), nlat - 1))
    if np.any(ilat1 < 1):
        # KLAT1 = 0 reads outside the 49r1 array (vote A14: unreachable for a file
        # spanning both poles; refuse rather than read out of bounds)
        raise ValueError("column poleward of the climatology's first latitude")
    ilat2 = ilat1 + 1
    s1 = np.sin(clim.lat1 + (ilat1 - 1) * clim.dlat)
    latw = (sinlat - s1) / (np.sin(clim.lat1 + (ilat2 - 1) * clim.dlat) - s1)
    c = lambda i, j: column_mass[i - 1, j - 1, :]                       # noqa: E731
    a = c(ilon1, ilat1) + latw[:, None] * (c(ilon1, ilat2) - c(ilon1, ilat1))
    b = c(ilon2, ilat1) + latw[:, None] * (c(ilon2, ilat2) - c(ilon2, ilat1))
    return a + lonw[:, None] * (b - a)


def calc_profiles_2d(column_mass_pts, cum_fraction, pressure_hl):
    """CALC_PROFILES 2D arm (yoeaerc.F90:852-869): (ncol, nlev, naer) kg/kg."""
    factor = constants.g_ifs / (pressure_hl[..., 1:] - pressure_hl[..., :-1])  # RG/dp
    dcum = jnp.asarray(cum_fraction[1:] - cum_fraction[:-1])                  # (nlev, naer)
    return jnp.asarray(column_mass_pts)[:, None, :] * factor[..., None] * dcum[None]


def _sqrt_guarded(x, sharpness):
    """sqrt(x), with x <= threshold -> 0 by where-before-sqrt (vote E1a): threshold 0
    in the hard form (bitwise), sqrt(finfo.tiny) in the smooth form, so an
    underflowed tropopause weight cannot give an inf derivative."""
    thr = 0.0 if sharpness == math.inf else float(np.sqrt(np.finfo(np.dtype(x.dtype)).tiny))
    ok = x > thr
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, x, 1.0)), 0.0)


def background_aerosol(pressure_hl, temperature_hl, sharpness=100.0):
    """radact.F90:657-787 slots 1 and 6 for NAERMACC=1, LHVOLCA=F (ZAERVO = RCSTBGA),
    NVOLCVERT=0; (ncol, nlev) each, after MAX(PAER, REPAER)."""
    p_top = pressure_hl[..., 0]

    def body(carry, xs):
        dpo, aetro = carry
        p_k, p_k1, t_k, t_k1 = xs
        grth = t_k / t_k1
        strat = step(0.5 * (p_k + p_k1), _P_STRAT_TOP_PA, _P_STRAT_TOP_PA, sharpness, False)
        # 0.5*(p(k)+p(k+1)) < 999 -> ZAETRN = 1 (weight 1 - strat)
        aetrn = blend(strat, aetro * upper_clamp(grth, 1.0, T_RATIO_CLAMP_SCALE, sharpness) ** _RCTRPT, 1.0)
        aetr = _sqrt_guarded(aetrn * aetro, sharpness)
        dpnmo = p_k1 - dpo
        paer1 = (1.0 - aetr) * (_RCTRBGA * dpnmo)
        paer6 = aetr * _RCSTBGA * dpnmo
        return (p_k1, aetrn), (paer1, paer6)

    xs = tuple(jnp.moveaxis(a, -1, 0) for a in (
        pressure_hl[..., :-1], pressure_hl[..., 1:], temperature_hl[..., :-1], temperature_hl[..., 1:]))
    _, (paer1, paer6) = jax.lax.scan(body, (p_top, jnp.ones_like(p_top)), xs)
    floor = lambda x: lower_clamp(jnp.moveaxis(x, 0, -1), _REPAER, _REPAER, sharpness)  # noqa: E731
    return floor(paer1), floor(paer6)


def radintg_aerosol(paero_clim):
    """radintg.F90:1415-1436: PAEROSOL (ncol, nlev, 13) = the 11 climatological
    species in YAER_RAD_DESC order, 0 for the 2 parametric types (SCALING = 1).
    Of PAEROSOL_OLD only slots 1 and 6 (:func:`background_aerosol`) are read
    downstream (radiation_scheme.F90:559-574), so they are passed on directly."""
    zeros = jnp.zeros(paero_clim.shape[:-1] + (N_TYPES - paero_clim.shape[-1],), paero_clim.dtype)
    return jnp.concatenate([paero_clim, zeros], axis=-1)


def macc_mixing_ratio(paerosol, paer1, paer6, layer_mass, trop_bg_mass_ext, strat_bg_mass_ext,
                      itype_trop_bg, itype_strat_bg, sharpness=100.0):
    """radiation_scheme.F90:545-575 (non-Tegen arm): max(PAEROSOL, 0) plus the
    trop/strat backgrounds PAEROSOL_OLD(:,1|6,:)/(layer mass * mass ext);
    paer1/paer6 (ncol, nlev) are PAEROSOL_OLD(:,1,:)/(:,6,:)."""
    mr = lower_clamp(paerosol, 0.0, MMR_CLAMP_SCALE, sharpness)
    cols = [mr[..., j] for j in range(mr.shape[-1])]
    if trop_bg_mass_ext > 0.0:
        cols[itype_trop_bg - 1] = cols[itype_trop_bg - 1] + paer1 / (
            layer_mass * trop_bg_mass_ext)
    if strat_bg_mass_ext > 0.0:
        cols[itype_strat_bg - 1] = cols[itype_strat_bg - 1] + paer6 / (
            layer_mass * strat_bg_mass_ext)
    return jnp.stack(cols, axis=-1)


def climatology_at_columns(clim: Cams2D, nindat: int, minute: int, eta_hl, sinlat, lon):
    """Set-up-time half of the chain (static numpy): UPDECAEC -> TEAERC%UPDATE ->
    the CALC_PROFILES horizontal step.  Returns (column mass (ncol, 11) kg/m2,
    cumulative fraction (nlev+1, 11)).  Lane adapter hook."""
    _, pmonth = month_and_year(nindat, minute)
    cm, cum = update_2d(clim, pmonth, eta_hl)
    return interp_points(clim, cm, sinlat, lon), cum


def macc_inputs(column_mass_pts, cum_fraction, pressure_hl, temperature_hl, sharpness=100.0):
    """Traced half (RADACT + radintg assembly): the :class:`MaccAerosol` that
    :func:`.ifs_interface.prepare_inputs` takes as ``aerosol``.  Lane adapter hook."""
    paer1, paer6 = background_aerosol(pressure_hl, temperature_hl, sharpness)
    paero = calc_profiles_2d(column_mass_pts, cum_fraction, pressure_hl)
    return MaccAerosol(radintg_aerosol(paero), paer1, paer6)
