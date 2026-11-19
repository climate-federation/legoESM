"""Interactive radiation of a CloudBench column, configured as the LES runs were.

The CloudBench LES (Chammas et al. 2026, Sci. Adv. eaec8488, Methods; run config
``radiative_transfer``) solved RRTMGP two-stream every 4 simulated minutes on its own
state, with the column extended above the 6 km domain on the driving GCM's T and q
to about 1 hPa; CO2 400 ppm, other gases from the GCM; diurnal-mean insolation at
an insolation-weighted zenith angle (``parameters.json``: ``irrad`` the beam
irradiance, ``zenith`` [rad]); surface albedo 0.07 and emissivity 0.98 over the SST;
condensate split into liquid and ice by temperature (swirl-lm
``water.liquid_fraction``: all ice at or below t_icenuc = 233 K, linear to all liquid
at T_freeze), each phase's radius by the Liu-Hallett 1/3 power law at N_d = 1e8 m^-3
with that phase's density (``microphysics_one_moment.cloud_particle_effective_radius``),
ice through the tables' medium-roughness ice optics; optics
tables rrtmgp-gas-lw-g128 / sw-g112 / cloudysky (the files this repo ships are
byte-identical).

:func:`column_radiation` runs the repo's RRTMGP on an SCM column (top-first arrays)
plus the sounding levels above the column top, and returns the column's heating
from the net fluxes at its interfaces, so the column energy
``sum(dp/g c_pd dT/dt) = F_net_down(top) - F_net_down(surface)`` holds exactly.

KNOWN DIFFERENCES (measured by ``scripts/validate/les_suite/cloudbench_rad_validate.py``):
the column sees the horizontal-mean cloud water as a homogeneous layer, not the
LES's resolved clouds; ozone is the SAM mid-latitude-summer profile and the other
trace gases the swirl-lm ``vmr_global_means.json`` values, not the per-case GCM
profiles (not public); the extension above the column stops at the GCM sounding's
top level (~2.5 hPa) and is held at the sounding's T and q, its q all vapour (no
cloud above the column); radiation is called
every column step (600 s), not every 240 s.

KNOWN DIFFERENCE (RULE 5, differentiability kept): swirl-lm's liquid fraction is
``clip((T - 233) / (T_freeze - 233), 0, 1)``, kinked at both ends. The default here
is that clip smoothed by softplus over ``CaseRadiation.ice_ramp_width_k`` =
``CB_ICE_RAMP_WIDTH_K`` = 0.5 K (votes 2026-11-18: author 0.5, codex 0.5, GLM 1),
off the clip by at most 0.5 ln2 / 40.15 = 0.0086 in liquid fraction, at the kinks;
``ice_ramp_width_k = 0`` gives swirl-lm's exact ramp. The flux gap the smoothing
leaves is recorded per fixture column in ``tests/atmosphere/les_suite/
test_cloudbench_radiation.py`` (``ICE_RAMP_GAPS``).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.ozone_mls import mls_ozone_vmr
from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
    preload_rrtmgp_optics,
    rrtmgp_radiation,
)

# --- the runs' radiative_transfer block and the paper's Methods ---
CB_SFC_ALBEDO = 0.07
CB_SFC_EMISSIVITY = 0.98
CB_CO2_VMR = 4.0e-4
# swirl-lm physics/radiation/optics/test_data/vmr_global_means.json (the file name the
# runs' config points to; its content is not published with the runs: UNVERIFIED);
# CO2 is replaced by the paper's 400 ppm
CB_GAS_VMR = {"ccl4": 8.306993e-11, "cf4": 8.1092494e-11, "cfc11": 2.3307986e-10,
              "cfc12": 5.2058097e-10, "cfc22": 2.2954207e-10, "ch4": 1.8314709e-06,
              "co": 1.2e-07, "co2": CB_CO2_VMR, "hfc125": 1.5355009e-11,
              "hfc134a": 8.051573e-11, "hfc143a": 1.5252782e-11, "hfc23": 2.6890436e-11,
              "hfc32": 8.336969e-12, "n2": 0.781, "n2o": 3.26988e-07, "o2": 0.209}
# --- Liu & Hallett (1997) eq. 8 as swirl-lm microphysics_one_moment.py ports it
# (constants from microphysics_one_moment_constants.py) ---
LH_DROPLET_N_M3 = 1.0e8
LH_ASYMMETRY = 0.8
# swirl-lm thermodynamics.proto ``t_icenuc`` default (the runs' water block leaves it
# unset) [K]
LES_T_ICENUC_K = 233.0
#: default width [K] of the smoothed liquid/ice ramp (0 = swirl-lm's exact clip)
CB_ICE_RAMP_WIDTH_K = 0.5
#: cover below which a cloudy layer's water is treated as spread over the layer
CF_MIN = 1.0e-3


class RadiationRecord(NamedTuple):
    """One radiation call on a column (top-first): ``dT_dt`` [K/s] per column layer,
    ``net_down_top`` / ``net_down_sfc`` [W/m2] the net (LW + SW) downward flux at the
    column's top interface and at the surface."""

    dT_dt: jnp.ndarray
    net_down_top: jnp.ndarray
    net_down_sfc: jnp.ndarray


class CaseRadiation(NamedTuple):
    """A case's radiation setting: SST [K], cos of the zenith angle, the beam
    irradiance [W/m2], and the sounding above the column top (top-first): full-level
    pressure [Pa], T [K], q_v [kg/kg], and the interface pressures from the
    extension's top down to the column's top interface (n_ext + 1);
    ``ice_ramp_width_k`` [K] the width of the smoothed liquid/ice ramp (static;
    0 = swirl-lm's exact ramp)."""

    sst: float
    cos_zenith: float
    irrad: float
    ext_p_full: np.ndarray
    ext_T: np.ndarray
    ext_q: np.ndarray
    ext_p_half: np.ndarray
    ice_ramp_width_k: float = CB_ICE_RAMP_WIDTH_K


def rrtmgp_config(irrad: float) -> RRTMGPConfig:
    """The RRTMGP configuration of a CloudBench case (clouds ON; ``S_0`` is the
    case's beam irradiance, which the solver scales by cos(zenith))."""
    return RRTMGPConfig(include_clouds=True, co2_ppmv=CB_CO2_VMR * 1.0e6,
                        ch4_ppbv=CB_GAS_VMR["ch4"] * 1.0e9,
                        n2o_ppbv=CB_GAS_VMR["n2o"] * 1.0e9,
                        sfc_emissivity=CB_SFC_EMISSIVITY, sfc_albedo=CB_SFC_ALBEDO,
                        S_0=float(irrad), gpoint_checkpoint=False)


def case_radiation(params: dict, sounding: dict, p_top_column: float,
                   ice_ramp_width_k: float = CB_ICE_RAMP_WIDTH_K) -> CaseRadiation:
    """The case's setting from its ``parameters.json`` and GCM sounding. The extension
    takes the sounding levels above the column's top interface ``p_top_column``; its
    interfaces lie halfway between levels in ln p, the column top below and, above
    the highest level, half its ln p spacing to the level beneath it."""
    p = np.asarray(sounding["p"], dtype=np.float64)
    above = p < p_top_column
    if np.count_nonzero(above) < 2:
        raise ValueError("the sounding needs at least two levels above the column top")
    p_ext = p[above][::-1]                        # top-first
    T_ext = np.asarray(sounding["T"], dtype=np.float64)[above][::-1]
    q_ext = np.asarray(sounding["q_t"], dtype=np.float64)[above][::-1]
    # interfaces halfway in ln p (hydrostatic: halfway in height at uniform T); the
    # top one half a spacing above the highest level, so it stays above p = 0
    mid = np.sqrt(p_ext[1:] * p_ext[:-1])
    top = p_ext[0] * np.sqrt(p_ext[0] / p_ext[1])
    p_half = np.concatenate([[top], mid, [p_top_column]])
    if not np.all(np.diff(p_half) > 0.0):
        raise ValueError("extension interfaces must increase downward")
    # load the optics tables now: a first load inside a jit (the column's scan) fails
    preload_rrtmgp_optics(rrtmgp_config(float(params["irrad"])))
    return CaseRadiation(sst=float(params["sst"]), cos_zenith=float(np.cos(params["zenith"])),
                         irrad=float(params["irrad"]), ext_p_full=p_ext, ext_T=T_ext,
                         ext_q=q_ext, ext_p_half=p_half,
                         ice_ramp_width_k=float(ice_ramp_width_k))


def liu_hallett_r_eff(rho, q_c, rho_condensate=constants.rho_water):
    """Effective radius [m] of cloud particles of density ``rho_condensate`` (liquid
    by default; ``constants.rho_ice`` for ice) at air density ``rho`` and that phase's
    specific humidity ``q_c`` (swirl-lm's port of Liu & Hallett 1997 eq. 8)."""
    alpha = (4.0 / 3.0 * np.pi * rho_condensate * LH_ASYMMETRY) ** (-1.0 / 3.0)
    q = jnp.maximum(q_c, 0.0)
    # the cube root's derivative is infinite at q = 0: evaluate it on a safe value
    # there so a gradient through the radius floor is 0, not 0 * inf = nan
    wet = q > 0.0
    q_safe = jnp.where(wet, q, 1.0)
    return jnp.where(wet, alpha * (rho * q_safe / LH_DROPLET_N_M3) ** (1.0 / 3.0), 0.0)


def liquid_fraction(T, width_k: float = CB_ICE_RAMP_WIDTH_K):
    """Liquid share of the condensate at temperature ``T`` [K]: swirl-lm's
    ``clip((T - 233) / (T_freeze - 233), 0, 1)`` with its two kinks smoothed by
    softplus over ``width_k`` [K] (static); ``width_k = 0`` is the exact clip."""
    if not isinstance(width_k, (int, float)):
        raise TypeError("width_k must be a static Python number")
    span = constants.T_freeze - LES_T_ICENUC_K
    if width_k == 0.0:
        return jnp.clip((T - LES_T_ICENUC_K) / span, 0.0, 1.0)
    sp = lambda x: width_k * jax.nn.softplus(x / width_k)   # noqa: E731
    return (sp(T - LES_T_ICENUC_K) - sp(T - constants.T_freeze)) / span


def column_fluxes(case: CaseRadiation, T, q_v, q_c, p_full, p_half, rho,
                  *, ozone=mls_ozone_vmr, clouds: bool = True, cloud_fraction=None):
    """RRTMGP fluxes of the column with its extension above (``out``, top-first on all
    ``n_ext + nlev + 1`` interfaces) and ``n_ext``; inputs as :func:`column_radiation`."""
    dtype = jnp.asarray(T).dtype
    n_ext = case.ext_p_full.shape[0]
    # checked when concrete; a traced column (inside the SCM's jit) was built on the
    # grid whose top the extension was cut at (cloudbench_scm.build_cloudbench_scm)
    if not isinstance(p_half, jax.core.Tracer) and abs(
            float(case.ext_p_half[-1]) - float(p_half[0])) > 1e-9 * float(p_half[0]):
        raise ValueError("the extension does not end at the column's top interface")
    ext = lambda a: jnp.asarray(a, dtype)   # noqa: E731
    T_all = jnp.concatenate([ext(case.ext_T), T])
    q_all = jnp.concatenate([ext(case.ext_q), q_v])
    pf_all = jnp.concatenate([ext(case.ext_p_full), p_full])
    ph_all = jnp.concatenate([ext(case.ext_p_half[:-1]), p_half])
    zeros_ext = jnp.zeros((n_ext,), dtype)
    q_cl = jnp.maximum(q_c, 0.0)
    cf = None
    if cloud_fraction is not None:
        cf = jnp.clip(jnp.asarray(cloud_fraction, dtype), 0.0, 1.0)
        # in-cloud water; a layer with water but (numerically) no cover keeps the mean
        q_cl = jnp.where(cf > CF_MIN, q_cl / jnp.maximum(cf, CF_MIN), q_cl)
        cf = jnp.where((cf <= CF_MIN) & (q_cl > 0.0), 1.0, cf)
        cf = jnp.concatenate([zeros_ext, cf])
    f_liq = liquid_fraction(T, case.ice_ramp_width_k)
    q_liq, q_ice = q_cl * f_liq, q_cl * (1.0 - f_liq)
    path = jnp.concatenate([zeros_ext, q_liq * jnp.diff(p_half) / constants.g])
    path_ice = jnp.concatenate([zeros_ext, q_ice * jnp.diff(p_half) / constants.g])
    r_floor = jnp.full((n_ext,), 1.0e-6, dtype)   # coeff-ok: unused radius in cloud-free layers [m]
    r_eff = jnp.concatenate([r_floor, jnp.maximum(liu_hallett_r_eff(rho, q_liq), 1.0e-6)])  # coeff-ok: RRTMGP's radius floor [m]
    r_eff_ice = jnp.concatenate([r_floor, jnp.maximum(  # coeff-ok: RRTMGP's radius floor [m]
        liu_hallett_r_eff(rho, q_ice, constants.rho_ice), 1.0e-6)])
    out = rrtmgp_radiation(
        T_all[None], pf_all[None], ph_all[None], jnp.asarray([case.sst], dtype),
        q_all[None], jnp.asarray([case.cos_zenith], dtype), rrtmgp_config(case.irrad),
        sfc_albedo_override=CB_SFC_ALBEDO, sfc_emissivity_override=CB_SFC_EMISSIVITY,
        o3_vmr=ozone(pf_all)[None],
        cloud_path_liq=(path[None] if clouds else None),
        cloud_r_eff_liq=(r_eff[None] if clouds else None),
        cloud_path_ice=(path_ice[None] if clouds else None),
        cloud_r_eff_ice=(r_eff_ice[None] if clouds else None),
        ghg_vmr_override=CB_GAS_VMR,
        mcica_cloud_fraction=(cf[None] if (clouds and cf is not None) else None))
    return out, n_ext


def column_radiation(case: CaseRadiation, T, q_v, q_c, p_full, p_half, rho,
                     *, ozone=mls_ozone_vmr, clouds: bool = True,
                     cloud_fraction=None) -> RadiationRecord:
    """Radiation of the column (top-first ``T``, ``q_v``, ``q_c``, ``p_full``, ``rho``
    on its ``nlev`` layers, ``p_half`` its ``nlev + 1`` interfaces) under ``case``.
    The heating of each column layer is ``g/c_pd`` times the net downward flux
    converging into it over its pressure thickness (the extension's heating is not
    used). ``cloud_fraction`` (top-first, optional): the cloud water is then in
    clouds covering that fraction of each layer (in-cloud water ``q_c / cf`` for the
    path and the droplet radius), sampled per g-point with maximum-random overlap
    (McICA); without it the layer's mean cloud water is spread over the whole layer."""
    dtype = jnp.asarray(T).dtype
    out, n_ext = column_fluxes(case, T, q_v, q_c, p_full, p_half, rho, ozone=ozone,
                               clouds=clouds, cloud_fraction=cloud_fraction)
    net = ((out.lw_flux_down + out.sw_flux_down - out.lw_flux_up - out.sw_flux_up)[0]
           [n_ext:].astype(dtype))                                # column interfaces, top-first
    dT_dt = constants.g / constants.c_pd * (net[:-1] - net[1:]) / jnp.diff(p_half)
    return RadiationRecord(dT_dt=dT_dt, net_down_top=net[0], net_down_sfc=net[-1])
