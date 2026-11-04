"""OpenIFS 49r1 IFS-side radiation interface (JAX port).

Line-by-line port of the 49r1 (8f6f722) ``arpifs/phys_radi`` routines that sit
around the ecRad core (:mod:`.ecrad`), for the 49r1 default configuration:

* ``radiation_scheme.F90`` input preparation (lines 300-618: thermodynamics,
  SATUR, single-level fields, McICA seed, cloud fields, overlap parameter,
  fractional std, Tegen aerosol arm, ozone and gas units) and output handling
  (lines 666-751);
* ``liquid_effective_radius.F90`` (NRADLP=2, Martin et al. CCN land/sea,
  LCCNL = LCCNO = T), ``ice_effective_radius.F90`` (NRADIP=3, Sun 2001,
  NMINICE=1), ``cloud_overlap_decorr_len.F90`` (NDECOLAT=2);
* the RADINTG model-grid glue (radintg.F90:547-556 curvature mu0 and
  1870-2058, as compiled in scripts/validate/oracle/ecrad_ifs/
  ifs49_radintg_post.F90);
* ``radheatn.F90`` (approximate SW/LW updates every step, user decision A10:
  LAPPROXLWUPDATE, LMANNERSSWUPDATE; NLWOUT=1; SPP off; LAPE=F).

Other options of these routines raise (vote IF6).  Arrays are column first,
levels top-down (index 0 = model top), any leading batch shape.

Between-call products (user decision A19) are :class:`RadiationCarry`:
everything RADHEATN reads from the last full radiation call.  It is NOT wired
into any lane or config.

Smoothing (user rule; votes IF1/IF2): one static ``sharpness`` (100 in
production, ``math.inf`` = the oracle's hard form exactly).  Formula switches
are centred cubic steps of half-width ``0.5*scale/sharpness`` around the
threshold (two_stream.step) blended with both branches evaluated; zero-threshold
tests (``x > 0``) are one-sided cubic steps, exactly 0 for ``x <= 0`` and 1 for
``x >= scale/sharpness``; clamps are inactive-exact (two_stream.lower_clamp /
upper_clamp).  Scales: ``|thr|`` for nonzero thresholds, 1 for fractions
(transmissivity, emissivity, albedo, land fraction), the bound for radius
clamps, the TOA flux ``RII0*AMU0`` for the RADINTG surface-flux tests, and
1e-9 kg/kg (ecRad's cloud mixing-ratio threshold) for condensate presence.
Oracle guards that bind only where their branch weight is exactly zero stay
hard, with benign inputs substituted there (vote IF5).  SATUR saturation
(aerosol hydration only) reuses the convection helper (vote IF4), with its
gradient kept now that it feeds the RH bin of the aerosol optics (vote A9a).
Aerosols: ``aerosol=None`` is the Tegen arm with ecRad aerosols off (vote IV1);
an :class:`.ifs_aerosol.MaccAerosol` selects the NAERMACC=1 arm with ecRad
aerosols on (:mod:`.ifs_aerosol`, :mod:`.aerosol_optics`).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics._ifs_thermo import ifs_thermo, satur_qsat
from legoesm.atmosphere.physics.radiation.ecrad_ifs import aerosol_optics, ecrad, ifs_aerosol
from legoesm.atmosphere.physics.radiation.ecrad_ifs import mcica_solvers as ms
from legoesm.atmosphere.physics.radiation.ecrad_ifs import rrtmg_prepare as prep
from legoesm.atmosphere.physics.radiation.ecrad_ifs.two_stream import (
    blend, lower_clamp, step, upper_clamp)
from legoesm.core.smooth import smoothstep_cubic

__physics_contract__ = {
    "summary": (
        "OpenIFS 49r1 IFS radiation interface: ecRad input preparation "
        "(effective radii, overlap, seeds, gases), flux outputs, RADINTG "
        "glue and RADHEATN approximate SW/LW updates and heating rates."
    ),
    "inputs": {
        "pressure_hl": "Pa (ncol, nlev+1), top-down", "temperature_hl": "K",
        "pressure_fl, temperature_fl": "Pa, K (ncol, nlev)",
        "q, q_liquid, q_ice, q_rain, q_snow, gas mmr": "kg/kg",
        "o3_dp": "Pa kg/kg", "cloud_fraction": "1", "mu0": "1",
        "gelam": "rad", "gemu": "1 (sine of latitude)", "ccn": "cm-3",
        "solar_irradiance": "W/m^2",
    },
    "outputs": {
        "re_liquid_um, re_ice_um": "um", "decorr_len_km": "km",
        "net fluxes, surface fluxes": "W/m^2 (positive down)",
        "transmissivities": "1", "heating rates": "K/s",
    },
    "sign_convention": (
        "Net fluxes are down minus up (positive downward), on half levels "
        "top-down; heating = -g/(cp dp) * d(net flux) across a layer, so "
        "flux convergence warms."
    ),
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "OpenIFS 49r1 (8f6f722) arpifs/phys_radi radiation_scheme.F90, "
        "liquid_effective_radius.F90, ice_effective_radius.F90, "
        "cloud_overlap_decorr_len.F90, radintg.F90, radheatn.F90; Martin et "
        "al. (1994), Wood (2000), Sun (2001), Manners et al. (2009)."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_interface_oracle.py: 49r1 oracle "
        "match at 1e-12 (hard form), column energy closure, gradients."
    ),
}

# --- 49r1 configuration defaults (suecrad.F90, su0phy.F90) ---
_NRADLP, _NRADIP, _NDECOLAT, _NMINICE, _NLWOUT = 2, 3, 2, 1, 1
_RRE2DE = 0.64952                 # suecrad.F90:696
_RMINICE = 60.0                   # suecrad.F90:709, um
_RCADECOR = 0.75                  # suecrad.F90:713, km
_RCBDECOR = 2.149                 # suecrad.F90:714, km
_RCLOUD_FRAC_STD = 1.0            # suecrad.F90:745
_REPSCW = 1.0e-12                 # suecrad.F90:996 (YOERDU)
_REPLOG = 1.0e-12                 # suecrad.F90:997
_RRAE = 0.1277e-2                 # surdi.F90 (49r1) Earth-curvature factor
_RSUNDUR = 120.0                  # surdi.F90 sunshine threshold, W/m^2
_RPLDARE = 1.0                    # sudyncore.F90:90 (no small planet)
_RVTMP2 = 0.0                     # suphec.F90:228

# --- liquid_effective_radius.F90 (Martin et al. 1994; Wood 2000) ---
_RE_LIQ_MIN_UM, _RE_LIQ_MAX_UM = 4.0, 30.0          # :71-72
_CF_RE_THRESHOLD = 0.001                            # :187 (liquid >=), ice :148 (>)
_DISPERSION_SEA, _DISPERSION_LAND = 0.77, 0.69      # :156, :164
_NTOT_SEA = (-1.15e-3, 0.963, 5.30)                 # :157 a*c*c + b*c + d
_NTOT_LAND = (-2.10e-4, 0.568, -27.9)               # :165
_WOOD_DISP_REF = 0.222                              # :174 (0.222/disp)**0.333
_THIRD_FORTRAN = 0.333                              # :174, :206 literal
_WOOD_RAIN_EXP = 0.666                              # :196
_WOOD_RAIN_COEFF = 0.2                              # :197
_G_PER_KG = 1000.0                                  # air density in g m-3
_UM_PER_CM = 100.0                                  # :205 re in um from cm-3 units
_LAND_THRESHOLD = 0.5                               # :152 PLAND_FRAC < 0.5 = sea

# --- ice_effective_radius.F90 (NRADIP=3, Sun 2001) ---
_RE_ICE_DEFAULT_D_UM = 80.0                         # :133
_D_MIN_EQUATOR_OFFSET_UM = 20.0                     # :139
_ICE_A = (45.8966, 0.2214)                          # :153
_ICE_B = (0.7957, 0.2535)                           # :154
_ICE_T_COEFF = (1.2351, 0.0105)                     # :155
_ICE_T_OFFSET_K = 83.15                             # :156
_D_MAX_UM = 155.0                                   # :164

# --- radiation_scheme.F90 / ecRad gas constants ---
_O2_VMR = 0.20944                                   # radiation_scheme.F90:613
_O2_MOLAR_MASS = 31.9988                            # radiation_gas_constants.F90:43 (O2)
_AIR_MOLAR_MASS = 28.970                            # radiation_gas_constants.F90:42
_N_GASES = 12
_N_TEGEN = 6
_BB_NET_THRESHOLD = float(np.float32(1.0e-5))       # :707 default-real literal
_EMIS_MIN, _EMIS_MAX = 0.8, 0.99                    # :710
_UV_BOUNDS_M = (0.2e-6, 0.4415e-6)                  # su_radiation_scheme.F90:717
_PAR_BOUNDS_M = (0.4e-6, 0.7e-6)                    # su_radiation_scheme.F90:720
_SEED_LON_FACTOR = 108000000.0                      # radiation_scheme.F90:368
_SEED_LAT_FACTOR = 6000.0                           # :369
_SECONDS_PER_MINUTE = 60.0                          # :391
_KM_TO_M = 1000.0                                   # :464
_UM_TO_M = 1.0e-6                                   # :443, :448

# --- radintg.F90 ---
_MU0_CURV_THRESHOLD = 1.0e-10                       # :548
_ALBEDO_MAX = 0.9999                                # :1917
_RADINTG_EPS = 1000.0 * float(np.finfo(np.float64).tiny)   # :484

# --- radheatn.F90 ---
_EPS = float(np.finfo(np.float64).eps)
_RADHEATN_EPS = 100.0 * _EPS                        # :282
_RADHEATN_EPS1 = 10000.0 * _EPS                     # :283
_LW_DN_DIFF_RATIO = float(np.float32(0.2))          # :203 default-real literal
_MANNERS_HALF = 0.5                                 # :425 half to surface, half to space

# --- condensate presence scale: ecRad cloud_mixing_ratio_threshold (vote IF2) ---
_Q_PRESENCE_SCALE = 1.0e-9                          # radiation_config.F90:187

# IFS constants (49r1 yomcst via legoesm.constants, vote IF8)
_RD = constants.R_d_ifs
_RG = constants.g_ifs
_RCPD = constants.c_pd_ifs
_RTT = constants.T_triple_ifs
_RSIGMA = constants.sigma_sb_ifs
_RPI = float(jnp.pi)
_R_OVER_G_ECRAD = constants.R_d_ecrad / constants.g_ecrad   # radiation_cloud.F90:206
_INV_G_ECRAD = 1.0 / constants.g_ecrad                      # radiation_thermodynamics.F90:181

# SATUR thermodynamics: 49r1 suphec.F90:228-256 (YOETHF), as the oracle driver sets them
_R2ES_E0 = 611.21  # satcurve-ok: YOETHF R2ES prefactor handed to the shared SATUR helper
_R3LES = 17.502
_R4LES = 32.19


def _hard(sharpness):
    return sharpness == math.inf


def _pos(x, scale, sharpness):
    """Weight of ``x > 0``: exactly 0 for x <= 0, 1 for x >= scale/sharpness."""
    if _hard(sharpness):
        x = jnp.asarray(x)
        return (x > 0.0).astype(x.dtype)
    return smoothstep_cubic(x * (sharpness / scale))


def _clamp(x, lo, hi, scale_lo, scale_hi, sharpness):
    """MAX(lo, MIN(x, hi)), inactive-exact."""
    return lower_clamp(upper_clamp(x, hi, scale_hi, sharpness), lo, scale_lo, sharpness)


def _clamp_lo_first(x, lo, hi, scale_lo, scale_hi, sharpness):
    """MIN(MAX(x, lo), hi), inactive-exact."""
    return upper_clamp(lower_clamp(x, lo, scale_lo, sharpness), hi, scale_hi, sharpness)


def _pow4(x):
    """x**4 as gfortran expands it (powi): (x*x)*(x*x)."""
    x2 = x * x
    return x2 * x2


def _nint(x):
    """Fortran NINT (nearest, halves away from zero) to int32."""
    t = jnp.trunc(x)
    r = jnp.where(jnp.abs(x - t) == 0.5, t + jnp.sign(x), jnp.round(x))
    return r.astype(jnp.int32)


def _satur_thermo():
    rd, rv = constants.R_d_ifs, constants.R_v_ifs
    return ifs_thermo(
        RG=constants.g_ifs, RD=rd, RV=rv, RCPD=constants.c_pd_ifs,
        RLVTT=constants.L_v_ifs, RLMLT=constants.L_s_ifs - constants.L_v_ifs,
        RTT=constants.T_triple_ifs, RA=constants.R_earth,
        R2ES=_R2ES_E0 * rd / rv, R3LES=_R3LES, R4LES=_R4LES)


# ---------------------------------------------------------------------------
# Effective radii and decorrelation length
# ---------------------------------------------------------------------------

def _liquid_re_branch(rho, cloud_fraction, q_liq, q_rain, w_cloud, ntot, dispersion, sharpness):
    """liquid_effective_radius.F90:174-212 for one CCN branch (NRADLP=2)."""
    zratio = (_WOOD_DISP_REF / dispersion) ** _THIRD_FORTRAN
    cf = jnp.where(w_cloud > 0.0, cloud_fraction, 1.0)
    lwc = rho * q_liq / cf
    rwc = rho * q_rain / cf
    w_lwc = step(lwc, _REPSCW, _REPSCW, sharpness, strict=True)
    lwc_safe = jnp.where(w_lwc > 0.0, lwc, 1.0)
    rain_ratio = rwc / lwc_safe
    wood = ((1.0 + rain_ratio) ** _WOOD_RAIN_EXP) / (1.0 + _WOOD_RAIN_COEFF * zratio * rain_ratio)
    wood = blend(w_lwc, wood, 1.0)
    re_cubed = (3.0 * (lwc + rwc)) / (4.0 * _RPI * ntot * dispersion)
    w_re = step(re_cubed, _REPLOG, _REPLOG, sharpness, strict=True)
    re_cubed_safe = jnp.where(w_re > 0.0, re_cubed, 1.0)
    re = wood * _UM_PER_CM * jnp.exp(_THIRD_FORTRAN * jnp.log(re_cubed_safe))
    re = _clamp(re, _RE_LIQ_MIN_UM, _RE_LIQ_MAX_UM, _RE_LIQ_MIN_UM, _RE_LIQ_MAX_UM, sharpness)
    re = blend(w_re, re, _RE_LIQ_MIN_UM)
    return blend(w_cloud, re, _RE_LIQ_MIN_UM)


def liquid_effective_radius(pressure_fl, temperature_fl, cloud_fraction, q_liq, q_rain,
                            land_frac, ccn_land, ccn_sea, sharpness=100.0):
    """Liquid effective radius in microns (ncol, nlev); land_frac, ccn (ncol,)."""
    rho = _G_PER_KG * pressure_fl / (_RD * temperature_fl)
    w_cloud = (step(cloud_fraction, _CF_RE_THRESHOLD, _CF_RE_THRESHOLD, sharpness, strict=False)
               * _pos(q_liq + q_rain, _Q_PRESENCE_SCALE, sharpness))

    def ntot(c, coeffs):
        a, b, d = coeffs
        return a * c * c + b * c + d

    re_sea = _liquid_re_branch(rho, cloud_fraction, q_liq, q_rain, w_cloud,
                               ntot(ccn_sea, _NTOT_SEA)[..., None], _DISPERSION_SEA, sharpness)
    re_land = _liquid_re_branch(rho, cloud_fraction, q_liq, q_rain, w_cloud,
                                ntot(ccn_land, _NTOT_LAND)[..., None], _DISPERSION_LAND, sharpness)
    # IF (PLAND_FRAC < 0.5) sea ELSE land (vote IF3: smooth, class scale 1)
    w_land = step(land_frac, _LAND_THRESHOLD, 1.0, sharpness, strict=False)[..., None]
    return blend(w_land, re_land, re_sea)


def ice_effective_radius(pressure_fl, temperature_fl, cloud_fraction, q_ice, q_snow, gemu,
                         sharpness=100.0):
    """Ice effective radius in microns (ncol, nlev); gemu (ncol,) sine of latitude."""
    default_re = _RE_ICE_DEFAULT_D_UM * _RRE2DE
    d_min = (_D_MIN_EQUATOR_OFFSET_UM
             + (_RMINICE - _D_MIN_EQUATOR_OFFSET_UM) * jnp.cos(jnp.arcsin(gemu)))[..., None]
    q_tot = q_ice + q_snow
    w = (step(cloud_fraction, _CF_RE_THRESHOLD, _CF_RE_THRESHOLD, sharpness, strict=True)
         * _pos(q_tot, _Q_PRESENCE_SCALE, sharpness))
    cf = jnp.where(w > 0.0, cloud_fraction, 1.0)
    q_tot = jnp.where(w > 0.0, q_tot, 1.0)
    rho = _G_PER_KG * pressure_fl / (_RD * temperature_fl)
    iwc = rho * q_tot / cf
    t_c = temperature_fl - _RTT
    aiwc = _ICE_A[0] * iwc ** _ICE_A[1]
    biwc = _ICE_B[0] * iwc ** _ICE_B[1]
    diameter = (_ICE_T_COEFF[0] + _ICE_T_COEFF[1] * t_c) * (aiwc + biwc * (temperature_fl - _ICE_T_OFFSET_K))
    diameter = _clamp_lo_first(diameter, d_min, _D_MAX_UM, d_min, _D_MAX_UM, sharpness)  # :184
    return blend(w, diameter * _RRE2DE, default_re)


def cloud_overlap_decorr_len(gemu):
    """NDECOLAT=2: (decorrelation length for cloud boundaries in km (ncol,), ratio 0.5)."""
    # cos(asin(gemu))**2 spelled 1 - gemu**2 (vote IF12: same value, finite gradient at the poles)
    cos2_lat = 1.0 - gemu * gemu
    return _RCADECOR + (_RCBDECOR * cos2_lat), 0.5


def overlap_param(pressure_hl, temperature_hl, decorr_len_m):
    """radiation_cloud.F90 set_overlap_param, pressure increasing with index:
    (ncol, nlev-1).  decorr_len_m (ncol,)."""
    L = decorr_len_m[..., None]
    first = jnp.exp(-(_R_OVER_G_ECRAD / L[..., 0]) * temperature_hl[..., 1]
                    * jnp.log(pressure_hl[..., 2] / pressure_hl[..., 1]))
    rest = jnp.exp(-(0.5 * _R_OVER_G_ECRAD / L) * temperature_hl[..., 2:-1]
                   * jnp.log(pressure_hl[..., 3:] / pressure_hl[..., 1:-2]))
    return jnp.concatenate([first[..., None], rest], axis=-1)


def mcica_seed(nindat, nstep, tstep, gelam, gemu):
    """radiation_scheme.F90:390-405 McICA seed (ncol,) int32.  nindat (YYYYMMDD),
    nstep static ints, tstep (s); gelam (rad) and gemu as passed to RADIATION_SCHEME."""
    iday = int(nindat) % 100                                   # NDD(NINDAT)
    x = int(nstep) * float(tstep) / _SECONDS_PER_MINUTE        # static: NINT on the host
    t = math.trunc(x)
    itim = t + int(math.copysign(1, x)) if abs(x - t) == 0.5 else round(x)
    return itim + iday + _nint(gelam * _SEED_LON_FACTOR + jnp.arcsin(gemu) * _SEED_LAT_FACTOR)


def mcica_seed_traced(iday, elapsed_s, gelam, gemu):
    """:func:`mcica_seed` with a TRACED elapsed time (vote S3a): ``iday`` the
    day of month of NINDAT (the run start date), ``elapsed_s`` =
    NSTEP*TSTEP [s] since that date.  Same NINT (halves away from zero) as
    the host form, so both agree wherever both apply."""
    itim = _nint(jnp.asarray(elapsed_s) / _SECONDS_PER_MINUTE)
    return itim + jnp.asarray(iday, jnp.int32) + _nint(gelam * _SEED_LON_FACTOR + jnp.arcsin(gemu) * _SEED_LAT_FACTOR)


def _curvature(mu0, threshold, strict, sharpness):
    """RRAE/(SQRT(mu0**2+CRAE)-mu0) above the threshold, RRAE/SQRT(CRAE) below."""
    crae = _RRAE * (_RRAE + 2.0)
    w = step(mu0, threshold, threshold, sharpness, strict=strict)
    mu0_s = jnp.where(w > 0.0, mu0, 1.0)
    return blend(w, _RRAE / (jnp.sqrt(mu0_s * mu0_s + crae) - mu0_s), _RRAE / math.sqrt(crae))


def curvature_mu0(mu0, sharpness=100.0):
    """radintg.F90:547-556 (IF mu0 > 1e-10): Earth-curvature-corrected cos(solar zenith)."""
    return _curvature(mu0, _MU0_CURV_THRESHOLD, True, sharpness)


# ---------------------------------------------------------------------------
# RADIATION_SCHEME input preparation
# ---------------------------------------------------------------------------

def prepare_inputs(*, solar_irradiance, mu0, skin_temperature, albedo_dif, albedo_dir,
                   spectral_emiss, ccn_land, ccn_sea, gelam, gemu, land_sea_mask,
                   pressure_fl, temperature_fl, pressure_hl, temperature_hl, q,
                   co2, ch4, n2o, cfc11, cfc12, hcfc22, ccl4, o3_dp,
                   cloud_fraction, q_liquid, q_ice, q_rain, q_snow, aerosol_old,
                   nindat, nstep, tstep, sharpness=100.0, aerosol=None):
    """RADIATION_SCHEME lines 300-618.  ``mu0`` is the curvature-corrected value
    radintg passes (:func:`curvature_mu0`); ``aerosol_old`` (ncol, 6, nlev) the
    Tegen layer optical depths, read only when ``aerosol`` is None (aerosols off
    in ecRad, vote IF6); ``aerosol`` an :class:`.ifs_aerosol.MaccAerosol`.

    Returns ``(ecrad_kwargs, extra)``: the keyword arguments of
    :func:`.ecrad.radiation` except the McICA uniforms, and the other ecRad
    object fields and sub-routine outputs (``iseed``, ``h2o_sat_liq``,
    ``aerosol_mixing_ratio``, ``gas_iunits``, ``re_liquid_um``,
    ``re_ice_um``, ``decorr_len_km``, ``decorr_len_ratio``,
    ``cloud_inhom_decorr_scaling``, ``spectral_solar_cycle_multiplier``)."""
    # thermodynamics: surface half-level temperature (:346-348)
    t_sfc = temperature_fl[..., -1] + 0.5 * (temperature_hl[..., -1] - temperature_hl[..., -2])
    t_hl = jnp.concatenate([temperature_hl[..., :-1], t_sfc[..., None]], axis=-1)
    # SATUR(..., KFLAG=2) = FOEEWM branch (vote IF4)
    h2o_sat = satur_qsat(_satur_thermo(), temperature_fl, pressure_fl)   # vote A9a: no stop_gradient

    iseed = mcica_seed(nindat, nstep, tstep, gelam, gemu)

    q_ice_rad = q_ice + q_snow
    re_liq_um = liquid_effective_radius(pressure_fl, temperature_fl, cloud_fraction, q_liquid,
                                        q_rain, land_sea_mask, ccn_land, ccn_sea, sharpness)
    re_ice_um = ice_effective_radius(pressure_fl, temperature_fl, cloud_fraction, q_ice, q_snow,
                                     gemu, sharpness)
    decorr_km, decorr_ratio = cloud_overlap_decorr_len(gemu)
    alpha = overlap_param(pressure_hl, t_hl, decorr_km * _KM_TO_M)
    fractional_std = jnp.full_like(cloud_fraction, _RCLOUD_FRAC_STD)

    layer_mass = (pressure_hl[..., 1:] - pressure_hl[..., :-1]) * _INV_G_ECRAD
    if aerosol is None:
        # Tegen arm (:584-592): optical depth per unit dry-air mass
        aerosol_mr = jnp.swapaxes(aerosol_old, -1, -2) / layer_mass[..., None]
    else:
        # NAERMACC=1 arm (:545-575), aerosols on in ecRad (ifs_aerosol, aerosol_optics)
        aerosol_mr = ifs_aerosol.macc_mixing_ratio(
            aerosol.paerosol, aerosol.paer1, aerosol.paer6, layer_mass,
            **aerosol_optics.background_mass_ext(), sharpness=sharpness)

    # ozone Pa*kg/kg -> kg/kg (:597-602) and gases in ecRad order (:604-617);
    # SET_GAS_UNITS(IMassMixingRatio) converts only the well-mixed O2 VMR
    o3 = o3_dp / (pressure_hl[..., 1:] - pressure_hl[..., :-1])
    gas = [jnp.zeros_like(q)] * _N_GASES
    gas[prep.IH2O], gas[prep.ICO2], gas[prep.IO3], gas[prep.IN2O] = q, co2, o3, n2o
    gas[prep.ICH4], gas[prep.ICFC11], gas[prep.ICFC12] = ch4, cfc11, cfc12
    gas[prep.IHCFC22], gas[prep.ICCL4] = hcfc22, ccl4
    gas[prep.IO2] = jnp.full_like(q, _O2_VMR * (1.0 * _O2_MOLAR_MASS / _AIR_MOLAR_MASS * 1.0))
    gas_mmr = jnp.stack(gas, axis=-1)

    ecrad_kwargs = dict(
        pressure_hl=pressure_hl, temperature_hl=t_hl, gas_mmr=gas_mmr,
        skin_temperature=skin_temperature, cos_sza=mu0, solar_irradiance=solar_irradiance,
        sw_albedo=albedo_dif, sw_albedo_direct=albedo_dir, lw_emissivity=spectral_emiss,
        cloud_fraction=cloud_fraction, q_liquid=q_liquid, q_ice=q_ice_rad,
        re_liquid=re_liq_um * _UM_TO_M, re_ice=re_ice_um * _UM_TO_M,
        overlap_param=alpha, fractional_std=fractional_std)
    if aerosol is not None:
        ecrad_kwargs.update(aerosol_mixing_ratio=aerosol_mr, h2o_sat_liq=h2o_sat)
    extra = dict(
        iseed=iseed, h2o_sat_liq=h2o_sat, aerosol_mixing_ratio=aerosol_mr,
        gas_iunits=np.zeros(_N_GASES, dtype=np.int64),     # all IMassMixingRatio (= 0)
        re_liquid_um=re_liq_um, re_ice_um=re_ice_um, decorr_len_km=decorr_km,
        decorr_len_ratio=decorr_ratio, cloud_inhom_decorr_scaling=decorr_ratio,
        spectral_solar_cycle_multiplier=0.0)
    return ecrad_kwargs, extra


# ---------------------------------------------------------------------------
# RADIATION_SCHEME output handling
# ---------------------------------------------------------------------------

def _band_weights(bounds):
    """RAD_CONFIG%GET_SW_WEIGHTS (radiation_config.F90:1578-1653), use_bands
    branch with Planck = 1: the fraction of each SW band inside [w1, w2].
    Returns (zero-based band indices, weights), static numpy."""
    mapping = ms.sw_albedo_weights(bounds, (1, 2, 3))[1]
    iband = np.nonzero(mapping > 0.0)[0]
    return iband, mapping[iband]


UV_WEIGHTS = _band_weights(_UV_BOUNDS_M)
PAR_WEIGHTS = _band_weights(_PAR_BOUNDS_M)


def _weighted_band_sum(band_flux, weights):
    iband, w = weights
    acc = jnp.zeros_like(band_flux[..., 0])
    for j in range(len(iband)):                       # Fortran loop order
        acc = acc + w[j] * band_flux[..., iband[j]]
    return acc


def scheme_outputs(flux, mu0, skin_temperature, spectral_emiss, sharpness=100.0):
    """RADIATION_SCHEME lines 666-751 from the :func:`.ecrad.radiation` dict.
    ``mu0`` is the cos_sza ecRad saw.  Returns the RADIATION_SCHEME outputs
    keyed by their Fortran names (lower case, without the P prefix)."""
    sfc = lambda a: a[..., -1]  # noqa: E731
    out = {
        "flux_sw": flux["flux_dn_sw"] - flux["flux_up_sw"],
        "flux_lw": flux["flux_dn_lw"] - flux["flux_up_lw"],
        "flux_sw_clear": flux["flux_dn_sw_clear"] - flux["flux_up_sw_clear"],
        "flux_lw_clear": flux["flux_dn_lw_clear"] - flux["flux_up_lw_clear"],
        "flux_sw_dn": sfc(flux["flux_dn_sw"]),
        "flux_lw_dn": sfc(flux["flux_dn_lw"])[..., None],          # NLWOUT = 1
        "flux_sw_dn_clear": sfc(flux["flux_dn_sw_clear"]),
        "flux_lw_dn_clear": sfc(flux["flux_dn_lw_clear"]),
        "flux_dir": sfc(flux["flux_dn_direct_sw"]),
        "flux_dir_clear": sfc(flux["flux_dn_direct_sw_clear"]),
        "flux_sw_dn_toa": flux["flux_dn_sw"][..., 0],
    }
    # WHERE (PMU0 > EPSILON(1.0)) dir/mu0 ELSE 0 (:686-689)
    w = step(mu0, _EPS, _EPS, sharpness, strict=True)
    out["flux_dir_into_sun"] = blend(w, out["flux_dir"] / jnp.where(w > 0.0, mu0, 1.0), 0.0)
    band = flux["spectral_flux_dn_sw_surf"]
    out["flux_uv"] = _weighted_band_sum(band, UV_WEIGHTS)
    out["flux_par"] = _weighted_band_sum(band, PAR_WEIGHTS)
    out["flux_par_clear"] = _weighted_band_sum(flux["spectral_flux_dn_sw_surf_clear"], PAR_WEIGHTS)
    # broadband emissivity, NLWOUT = 1 (:704-711)
    bb_net = out["flux_lw_dn"][..., 0] - _RSIGMA * _pow4(skin_temperature)
    w = step(jnp.abs(bb_net), _BB_NET_THRESHOLD, _BB_NET_THRESHOLD, sharpness, strict=True)
    ratio = sfc(out["flux_lw"]) / jnp.where(w > 0.0, bb_net, 1.0)
    emis = _clamp(ratio, _EMIS_MIN, _EMIS_MAX, 1.0, 1.0, sharpness)
    out["emis_out"] = blend(w, emis, spectral_emiss[..., 0])
    out["lwderivative"] = flux["lw_derivative"]                    # LAPPROXLWUPDATE
    out["swdiffuseband"] = flux["canopy_flux_dn_diffuse_sw_surf"]  # LAPPROXSWUPDATE
    out["swdirectband"] = flux["canopy_flux_dn_direct_sw_surf"]
    return out


# ---------------------------------------------------------------------------
# RADINTG model-grid glue and between-call carry
# ---------------------------------------------------------------------------

class RadiationCarry(NamedTuple):
    """Between-call products of the last full radiation call (user decision
    A19): the RADINTG outputs RADHEATN reads, plus the cos(solar zenith) the
    radiation call was made with (PMU0M)."""
    emtd: jax.Array       # (ncol, nlev+1) net LW flux, W/m^2
    trso: jax.Array       # (ncol, nlev+1) net SW transmissivity
    trsc: jax.Array       # (ncol, nlev+1) clear-sky net SW transmissivity
    emtc: jax.Array       # (ncol, nlev+1) clear-sky net LW flux
    frted: jax.Array      # (ncol, NLWOUT) surface LW down
    trsod: jax.Array      # (ncol,) surface SW down transmissivity
    trsodc: jax.Array     # (ncol,)
    emtdc: jax.Array      # (ncol,) clear-sky surface LW down
    emit: jax.Array       # (ncol,) broadband surface emissivity
    sudu: jax.Array       # (ncol,) direct flux into the sun's plane
    uvdf: jax.Array
    parf: jax.Array
    parcf: jax.Array
    tincf: jax.Array
    fdir: jax.Array
    cdir: jax.Array
    lwderivative: jax.Array   # (ncol, nlev+1)
    mu0_rad: jax.Array        # (ncol,) PMU0M: cos(sza) of the radiation call (uncorrected)


def radintg_post(rs, solar_irradiance, amu0, albedo_dif, albedo_dir, mu0_rad, sharpness=100.0):
    """radintg.F90:1870-2058 (radiation grid = model grid) from the
    :func:`scheme_outputs` dict.  ``amu0`` curvature-corrected mu0 of the call,
    ``mu0_rad`` the uncorrected one (stored for RADHEATN).  Returns RadiationCarry."""
    toa = solar_irradiance * amu0
    frso = rs["flux_sw"]
    frso_sfc = frso[..., -1]
    frsod = lower_clamp(rs["flux_sw_dn"], frso_sfc, toa, sharpness)                  # :1891
    frsodc = lower_clamp(rs["flux_sw_dn_clear"], rs["flux_sw_clear"][..., -1], toa, sharpness)

    # LAPPROXSWUPDATE: surface net flux for the model-grid albedo (:1902-1988)
    w = (_pos(frsod, toa, sharpness) * _pos(frso_sfc, toa, sharpness)
         * _pos(frsod - frso_sfc, toa, sharpness))
    live = w > 0.0
    d_s = jnp.where(live, frsod, toa)
    n_s = jnp.where(live, frso_sfc, 0.5 * toa)
    n_t = jnp.where(live, frso[..., 0], 0.5 * toa)
    dirb, difb = rs["swdirectband"], rs["swdiffuseband"]
    s_dir = jnp.zeros_like(d_s)
    s_dif = jnp.zeros_like(d_s)
    for j in range(dirb.shape[-1]):                      # Fortran SUM order
        s_dir = s_dir + dirb[..., j] * albedo_dir[..., j]
        s_dif = s_dif + difb[..., j] * albedo_dif[..., j]
    den = jnp.sign(d_s) * jnp.maximum(jnp.abs(d_s), _RADINTG_EPS)                   # IF5: hard
    albedo = _clamp((s_dir + s_dif) / den, 0.0, _ALBEDO_MAX, 1.0, 1.0, sharpness)
    dn_top = solar_irradiance * amu0
    up_top = dn_top - n_t
    up_sfc = d_s - n_s
    det = dn_top * dn_top - up_sfc * up_sfc
    trans = (dn_top * d_s - up_sfc * up_top) / det
    refl = (dn_top * up_top - up_sfc * d_s) / det
    new_net = dn_top * trans * (1.0 - albedo) / (1.0 - albedo * refl)
    net_change = blend(w, new_net - n_s, 0.0)
    frsod = blend(w, new_net / (1.0 - albedo), frsod)
    frso = frso + net_change[..., None]

    inv = lambda x: x / (solar_irradiance * amu0)  # noqa: E731
    inv2 = lambda x: x / (solar_irradiance * amu0[..., None])  # noqa: E731
    c01 = lambda x: _clamp(x, 0.0, 1.0, 1.0, 1.0, sharpness)  # noqa: E731
    return RadiationCarry(
        emtd=rs["flux_lw"], trso=c01(inv2(frso)), trsc=c01(inv2(rs["flux_sw_clear"])),
        emtc=rs["flux_lw_clear"], frted=rs["flux_lw_dn"],
        trsod=c01(inv(frsod)), trsodc=c01(inv(frsodc)), emtdc=rs["flux_lw_dn_clear"],
        emit=c01(rs["emis_out"]), sudu=rs["flux_dir_into_sun"],
        uvdf=c01(inv(rs["flux_uv"])), parf=c01(inv(rs["flux_par"])),
        parcf=c01(inv(rs["flux_par_clear"])), tincf=c01(inv(rs["flux_sw_dn_toa"])),
        fdir=c01(inv(rs["flux_dir"])), cdir=c01(inv(rs["flux_dir_clear"])),
        lwderivative=c01(rs["lwderivative"]), mu0_rad=mu0_rad)


def radiation_scheme(*, mu0, uniforms_lw, uniforms_sw, sharpness=100.0, **scheme_kwargs):
    """One full radiation call on the model grid: radintg.F90:547-556 (curvature
    mu0) -> RADIATION_SCHEME (:func:`prepare_inputs`, :func:`.ecrad.radiation`,
    :func:`scheme_outputs`) -> radintg.F90:1870-2058 (:func:`radintg_post`).

    ``mu0`` is the RAW cos(solar zenith) (vote C3); ``uniforms_lw/sw`` the McICA
    uniforms, required (vote C2): in production
    ``ecrad.draw_uniforms(mcica_seed(...), nlev)``.  ``scheme_kwargs`` are the
    other :func:`prepare_inputs` keywords.  Returns
    ``(scheme_outputs dict, RadiationCarry)`` (vote C4)."""
    amu0 = curvature_mu0(mu0, sharpness)
    kw, _ = prepare_inputs(mu0=amu0, sharpness=sharpness, **scheme_kwargs)
    flux = ecrad.radiation(**kw, uniforms_lw=uniforms_lw, uniforms_sw=uniforms_sw,
                           sharpness=sharpness)
    rs = scheme_outputs(flux, amu0, kw["skin_temperature"], kw["lw_emissivity"], sharpness)
    carry = radintg_post(rs, kw["solar_irradiance"], amu0, kw["sw_albedo"],
                         kw["sw_albedo_direct"], mu0, sharpness)
    return rs, carry


# ---------------------------------------------------------------------------
# RADHEATN
# ---------------------------------------------------------------------------

def _mu0_curv_radheatn(mu0, sharpness):
    """radheatn.F90:401-410 (IF mu0 >= ZEPSILON1), same formula as radintg."""
    return _curvature(mu0, _RADHEATN_EPS1, False, sharpness)


def _manners(zi0, mu0, w_mu0, ratio, fdiri, trsod, trso_sfc, sharpness):
    """radheatn.F90:418-447 (one sky): (net diff, direct scaling, surface scaling)."""
    w = (w_mu0 * step(fdiri, _RADHEATN_EPS1, _RADHEATN_EPS1, sharpness, strict=False)
         * step(trsod, _RADHEATN_EPS1, _RADHEATN_EPS1, sharpness, strict=False))
    live = w > 0.0
    f = jnp.where(live, fdiri, 0.5)
    t = jnp.where(live, trsod, 1.0)
    trans = f ** jnp.where(live, ratio, 1.0)
    net_diff = _MANNERS_HALF * zi0 * (trans - f) * trso_sfc / t
    dir_scal = trans / f
    sfc_scal = 1.0 + _MANNERS_HALF * (trans - f) / t
    return blend(w, net_diff, 0.0), blend(w, dir_scal, 1.0), blend(w, sfc_scal, 1.0)


def radheatn(carry, pressure_hl, q, skin_temperature, mu0, solar_irradiance, te=None,
             sharpness=100.0):
    """RADHEATN (radheatn.F90) for one physics step, NLWOUT = 1, LAPE = F, SPP off.

    ``carry`` from the last full radiation call; ``mu0`` the instantaneous
    cos(solar zenith) (PMU0), ``skin_temperature`` the current skin T
    (PTSM1M), ``te`` the incoming temperature tendency (zero if None).
    Returns a dict keyed by the RADHEATN output names (lower case, no P)."""
    if te is None:
        te = jnp.zeros_like(q)
    # 2. incident solar radiation (:293-307)
    w0 = step(mu0, _RADHEATN_EPS, _RADHEATN_EPS, sharpness, strict=False)
    zi0 = blend(w0, solar_irradiance * mu0, 0.0)
    dsrp = w0

    # 4.0.1 approximate LW update (:318-380)
    emtd, lwd = carry.emtd, carry.lwderivative
    lw_up_diff = emtd[..., -1] + carry.emit * (_RSIGMA * _pow4(skin_temperature) - carry.frted[..., 0])
    lw_dn_diff = _LW_DN_DIFF_RATIO * lw_up_diff
    # 0/0 if lwd(TOA) == 1 (fully transparent column), as in the oracle (vote IF11)
    lw_dn_deriv = (lwd - lwd[..., :1]) / (1.0 - lwd[..., :1])
    frth = emtd + lw_dn_diff[..., None] * lw_dn_deriv - lw_up_diff[..., None] * lwd
    frthd = carry.frted[..., 0] + lw_dn_diff

    # 4.0.2 Manners et al. (2009) SW path-length update (:392-447)
    mu0_curv = _mu0_curv_radheatn(mu0, sharpness)
    mu0m_curv = _mu0_curv_radheatn(carry.mu0_rad, sharpness)
    ratio = mu0m_curv / mu0_curv
    w_mu0 = step(mu0, _RADHEATN_EPS1, _RADHEATN_EPS1, sharpness, strict=False)
    net_diff, dir_scal, sfc_scal = _manners(zi0, mu0, w_mu0, ratio, carry.fdir, carry.trsod,
                                            carry.trso[..., -1], sharpness)
    net_diff_c, dir_scal_c, sfc_scal_c = _manners(zi0, mu0, w_mu0, ratio, carry.cdir,
                                                  carry.trsodc, carry.trsc[..., -1], sharpness)

    # scale transmissivities (:462-475) and surface floor (:480-484)
    frso = zi0[..., None] * carry.trso + net_diff[..., None]
    frso = frso.at[..., -1].set(lower_clamp(frso[..., -1], _RADHEATN_EPS, _RADHEATN_EPS, sharpness))
    frsoc = jnp.stack([zi0 * carry.trsc[..., 0] + net_diff_c,
                       zi0 * carry.trsc[..., -1] + net_diff_c], axis=-1)
    frsod = zi0 * carry.trsod * sfc_scal
    frsodc = zi0 * carry.trsodc * sfc_scal_c

    # 4.1-4.4 heating rates (:488-560)
    cons3 = (_RG * _RPLDARE) / _RCPD
    flux_to_hr = -cons3 / ((pressure_hl[..., 1:] - pressure_hl[..., :-1]) * (1.0 + _RVTMP2 * q))
    fl = frso + frth
    sw_clear = zi0[..., None] * carry.trsc
    out = {
        "te": te + flux_to_hr * (fl[..., 1:] - fl[..., :-1]),
        "hrsw": flux_to_hr * (frso[..., 1:] - frso[..., :-1]),
        "hrlw": flux_to_hr * (frth[..., 1:] - frth[..., :-1]),
        "hrsc": flux_to_hr * (sw_clear[..., 1:] - sw_clear[..., :-1]),
        "hrlc": flux_to_hr * (carry.emtc[..., 1:] - carry.emtc[..., :-1]),
        "frso": frso, "frth": frth, "frsod": frsod, "frsodc": frsodc,
        "frthd": frthd, "frthdc": carry.emtdc, "frsoc": frsoc,
        "frthc": jnp.stack([carry.emtc[..., 0], carry.emtc[..., -1]], axis=-1),
        "incsr": zi0,
    }
    # 4.5 surface fluxes (:567-583)
    out["uvdf"] = zi0 * carry.uvdf * sfc_scal
    out["parf"] = zi0 * carry.parf * sfc_scal
    out["parcf"] = zi0 * carry.parcf * sfc_scal_c
    out["tincf"] = zi0 * carry.tincf
    out["fdir"] = zi0 * carry.fdir * dir_scal
    out["cdir"] = zi0 * carry.cdir * dir_scal_c
    dsrp = dsrp * solar_irradiance * carry.fdir * dir_scal
    out["dsrp"] = dsrp
    out["sdur"] = step(dsrp, _RSUNDUR, _RSUNDUR, sharpness, strict=False)
    return out


def check_configuration(*, nradlp=_NRADLP, nradip=_NRADIP, ndecolat=_NDECOLAT,
                        nminice=_NMINICE, nlwout=_NLWOUT, lspp=False, lape=False):
    """The only configuration ported (vote IF6); anything else raises."""
    want = dict(nradlp=_NRADLP, nradip=_NRADIP, ndecolat=_NDECOLAT, nminice=_NMINICE,
                nlwout=_NLWOUT, lspp=False, lape=False)
    got = dict(nradlp=nradlp, nradip=nradip, ndecolat=ndecolat, nminice=nminice,
               nlwout=nlwout, lspp=lspp, lape=lape)
    bad = {k: v for k, v in got.items() if v != want[k]}
    if bad:
        raise ValueError(f"ecrad_ifs.ifs_interface ports only the 49r1 default path {want}; got {bad}")
