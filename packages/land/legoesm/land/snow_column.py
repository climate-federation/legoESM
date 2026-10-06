"""Multi-layer snowpack column (energy- and mass-conserving).

Used by the multilayer land when ``MultiLayerLandConfig.snow_scheme ==
"layered"``: the land step accumulates snowfall and rain into the pack
(``snow_add_mass``), remaps and compacts it (``snow_remap_compact``), solves the
pack and the soil as ONE implicit heat-conduction column
(``soil_thermal.solve_snow_soil_thermal``, fed by ``snow_thermal_props``), then
re-equilibrates phase and percolates liquid (``snow_phase_and_percolate``).
``step_snow_column`` is the stand-alone composition with a PRESCRIBED base flux
(kept for the module's own conservation tests; explicit snow-soil coupling is
unstable for a thin pack at a 30-min land step, which is why the land model
does not use it).

Physics (fixed ``n_layers`` equal-SWE-mass layers; the total pack SWE is
remapped to ``n_layers`` equal-mass layers each step, a conservative 1-D
cumulative-mass remap so total ice, liquid, and enthalpy are preserved to
machine precision while shapes stay static and differentiable):

* accumulation — snowfall adds ice at ``min(T_air, T_freeze)`` with CLM5's
  fresh-snow density (``new_snow_bulk_density``: temperature and wind) (added as
  enthalpy so the top layer stays conservative even if it holds liquid);
* compaction — CLM5 (``clm5_compaction_rate``: destructive metamorphism,
  Vionnet 2012 overburden, wind drift), setting layer thickness
  ``dz = swe_water_equiv / density`` and the snow conductivity
  ``k = k_snow_ref * (rho / rho_snow_ref)^2`` (Sturm 1997 style);
* thermal — implicit (backward-Euler) heat diffusion through the layers with a
  prescribed surface flux ``Q_top`` and a base flux ``G_bottom`` (to the soil),
  using the full sensible heat capacity ``ice*c_ice + liq*c_liq``;
* phase change (ENTHALPY method) — each layer's water mass ``w`` and enthalpy
  ``H`` (relative to all-ice-at-``T_freeze``) determine the equilibrium
  ice/liquid/T: ``H<=0`` all ice below freezing, ``H>=w*L_f`` all liquid above
  freezing, else a two-phase mix at ``T_freeze`` with ``liq=H/L_f``.  This
  captures melt AND refreeze in one enthalpy-conserving step;
* percolation — liquid beyond the irreducible holding capacity moves down (as
  water at ``T_freeze``, carrying its latent enthalpy ``L_f``), re-equilibrating
  each receiving layer; the liquid leaving the base is ``drainage``.

Conserved (see ``tests/land/test_snow_column.py``):
* WATER: ``sum(swe_ice + swe_liq)`` changes only by ``snowfall*dt - drainage``.
* ENERGY: column enthalpy ``H = sum[(ice*c_ice + liq*c_liq)*(T-T_freeze) +
  liq*L_f]`` changes only by ``(Q_top - G_bottom)*dt`` (boundary heat) plus the
  fresh-snow enthalpy in, minus the drainage enthalpy out (``drainage_heat``,
  the water's full enthalpy ``drainage*(c_liq*(T-T_freeze) + L_f)``).
"""

from __future__ import annotations

import numbers
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.timestepping.tridiagonal import thomas_solve

# --- reference snow material properties (CICE / Sturm 1997) ---
_RHO_SNOW_REF = constants.rho_snow      # kg/m^3, conductivity reference density
_K_SNOW_REF = constants.k_snow          # W/m/K at the reference density
_C_ICE = constants.c_pi                 # J/kg/K (snow grains ~ ice)
_C_LIQ = constants.c_pw                 # J/kg/K liquid water
_TF = constants.T_freeze                # freezing point [K]
_LF = constants.L_f  # latent-ok: enthalpy referenced to ice at T_freeze, where L_f(T_freeze) == L_f
_EPS = 1e-12                            # generic small floor
_DZ_HALF_MIN = 1e-4                     # [m] min interface distance (empty-pack conductance bound)
_COEFF_MIN = 1e-6                       # [W/m^2/K] floor on every inter-layer conductance (no layer decouples)


# --- CLM5 snow compaction (CTSM clm5.0 SnowHydrologyMod.SnowCompaction, SNTHERM;
#     CLM5 defaults: Vionnet 2012 overburden, clm50_params.c250311) ---
_CLM_C3_PER_S = 2.777e-6          # destructive-metamorphism rate at T_freeze [1/s]
_CLM_C4_PER_K = 0.04              # its temperature sensitivity [1/K]
_CLM_C5_LIQ = 2.0                 # liquid-water enhancement of metamorphism [-]
# CTSM: "if (h2osoi_liq(c,j) > 0.01_r8*dz(c,j)*frac_sno(c)) ddz1=ddz1*c5" compares
# kg/m2 with 0.01 x metres, i.e. liquid per unit thickness > 0.01 (kg/m3 numerically).
_CLM_LIQ_FLAG_KG_M3 = 0.01
_CLM_UPPLIM_DM_KG_M3 = 175.0      # upplim_destruct_metamorph (clm50_params) [kg/m3]
_CLM_DM_DECAY_M3_KG = 46.0e-3     # metamorphism decay above that limit [m3/kg]
_VIONNET_ETA0 = 7.62237e6         # eta0_vionnet (clm50_params) [kg s/m2]
_VIONNET_C_ETA_KG_M3 = 450.0      # ceta [kg/m3]
_VIONNET_A_ETA_PER_K = 0.1        # aeta [1/K]
_VIONNET_B_ETA_M3_KG = 0.023      # beta [m3/kg]
_VIONNET_F2 = 4.0                 # f2, fixed at its maximum in CLM5 [-]
_VIONNET_F1_LIQ = 60.0            # f1 liquid-water factor [-]
_CLM_VOID_MIN = 0.001             # compaction only for a non-saturated layer [-]
_CLM_ICE_MIN_KG_M2 = 0.1          # ... holding more than 0.1 kg/m2 of ice
# Wind-drift compaction (CTSM WindDriftCompaction, CLM5 default ON; Vionnet 2012):
_DRIFT_RHO_MIN_KG_M3 = 50.0       # rho_min
_DRIFT_RHO_MAX_KG_M3 = 350.0      # rho_max (clm50_params)
_DRIFT_GS = 0.35e-3               # drift_gs (clm50_params), used as in CTSM
_DRIFT_SPH = 1.0                  # drift_sph
_DRIFT_TAU_REF_S = 48.0 * 3600.0  # tau_ref [s]
_DRIFT_SI_MAX = 3.25              # driftability index cap
_DRIFT_Z_DECAY_M = 0.1            # pseudo-depth e-folding [m]
_DRIFT_FRHO_0 = 1.25              # mobility density factor: Frho = 1.25 - 0.0042 (rho - rho_min)
_DRIFT_FRHO_SLOPE = 0.0042
_DRIFT_MO_GRAIN = 0.34            # MO = 0.34 (-0.583 gs - 0.833 sph + 0.833) + 0.66 Frho
_DRIFT_MO_GS = 0.583
_DRIFT_MO_SPH = 0.833
_DRIFT_MO_FRHO = 0.66
_DRIFT_SI_AMP = 2.868             # SI = -2.868 exp(-0.085 wind) + 1 + MO
_DRIFT_SI_WIND = 0.085
# Fresh-snow bulk density (CTSM NewSnowBulkDensity, CLM5: lotmp Slater2017 and
# wind_dependent_snow_density = .true.):
_BIFALL_BASE_KG_M3 = 50.0
_BIFALL_COEF = 1.7
_BIFALL_T_WARM_K = 2.0            # above Tf + 2: capped at the Tf + 2 value
_BIFALL_T_COLD_K = 15.0           # Tf - 15 .. Tf + 2: 50 + 1.7 (T - Tf + 15)^1.5
_SLATER_T_FLOOR_C = -57.55        # Slater (2017) turnover limit [deg C]
_SLATER_LIN = 50.0 / 15.0 + 0.0333 * 15.0
_SLATER_QUAD = 0.0333
_BIFALL_WIND_MIN_M_S = 0.1
_BIFALL_WIND_AMP_KG_M3 = 266.861  # Liston et al. (2007) / Slater (2016) wind offset
_BIFALL_WIND_SCALE_M_S = 5.0
_BIFALL_WIND_EXP = 8.8
# Density given to a layer with no mass (never enters any flux; mixing is mass-weighted).
_RHO_EMPTY_LAYER = 100.0


class SnowColumnConfig(NamedTuple):
    """Multi-layer snowpack configuration."""
    n_layers: int = 5                    # fixed number of snow layers
    irreducible_liq_frac: float = 0.05   # liquid held per unit ice mass [-]
    k_conductivity_exponent: float = 2.0  # k ~ (rho/rho_ref)^exp (Sturm 1997)
    # --- coupling to the land surface (used by multilayer_land, snow_scheme="layered") ---
    # Broadband thermal-IR emissivity of snow: observed ~0.97-0.99 (Warren 1982;
    # Hori et al. 2006, fine grains near 0.99).  Default CLM5's fixed 0.97 (user
    # decision 2026-09-27); tunable within the observed bounds.
    emissivity_snow: float = 0.97
    # Density of a pack seeded from a bulk SWE (cold start / land IC) [kg/m^3].
    seed_density: float = 250.0


__param_spec__ = {
    "SnowColumnConfig": {
        "scheme_key": "land.snow_column",
        "excluded": {
            "k_conductivity_exponent": "material: Sturm (1997) conductivity exponent",
            "seed_density": "initial condition: density of a pack seeded from bulk SWE",
        },
        "params": {
            "irreducible_liq_frac": {
                "units": "1", "bounds": (0.0, 0.15), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "irreducible liquid water holding capacity (CLM5)", "shape": None,
            },
            "emissivity_snow": {
                "units": "1", "bounds": (0.96, 0.995), "tunable_tier": 2,
                "transform": "sigmoid", "category": "radiative",
                "reference": "snow thermal-IR emissivity (Warren 1982 Rev. Geophys. 20:67; "
                             "Hori et al. 2006 Remote Sens. Environ. 100:486)",
                "shape": None,
            },
        },
    },
}


class SnowColumnState(NamedTuple):
    """Prognostic multi-layer snow state, all shape ``(..., n_layers)``.

    ``swe_ice`` / ``swe_liq`` are the frozen / liquid water masses per layer
    [kg/m^2] (SWE, i.e. water-equivalent); ``T`` is layer temperature [K];
    ``density`` is the snow (ice-matrix) density [kg/m^3] (for thickness +
    conductivity).  Layer 0 is the TOP (surface).
    """
    swe_ice: jnp.ndarray
    swe_liq: jnp.ndarray
    T: jnp.ndarray
    density: jnp.ndarray


def initial_snow_state(shape, config: SnowColumnConfig = SnowColumnConfig()):
    """An empty snowpack (all masses zero) of the given leading ``shape``."""
    n = config.n_layers
    z = jnp.zeros(shape + (n,))
    return SnowColumnState(
        swe_ice=z, swe_liq=z,
        T=jnp.full(shape + (n,), _TF),
        density=jnp.full(shape + (n,), _RHO_EMPTY_LAYER),
    )


def _sensible_hc(swe_ice, swe_liq):
    """Sensible volumetric heat capacity per layer [J/m^2/K]."""
    return swe_ice * _C_ICE + swe_liq * _C_LIQ


def _enthalpy(swe_ice, swe_liq, T):
    """Per-layer enthalpy relative to all-ice-at-T_freeze [J/m^2]:
    ``H = (ice*c_ice + liq*c_liq)*(T - T_freeze) + liq*L_f``."""
    return _sensible_hc(swe_ice, swe_liq) * (T - _TF) + swe_liq * _LF


def _phase_from_w_H(w, H):
    """Equilibrium (ice, liq, T) from total water ``w`` [kg/m^2] and enthalpy
    ``H`` [J/m^2] (relative to all-ice-at-T_freeze).  Exactly conserves w and H.

    * ``H <= 0``   -> all ice, ``T = T_freeze + H/(w*c_ice) <= T_freeze``.
    * ``H >= w*Lf`` -> all liquid, ``T = T_freeze + (H-w*Lf)/(w*c_liq) >= T_freeze``.
    * else -> two-phase at ``T_freeze`` with ``liq = H/Lf``.
    """
    all_liquid = H >= w * _LF
    all_ice = H <= 0.0
    liq_mix = jnp.clip(H / _LF, 0.0, w)
    liq = jnp.where(all_liquid, w, jnp.where(all_ice, 0.0, liq_mix))
    ice = w - liq
    C = jnp.maximum(_sensible_hc(ice, liq), _EPS)
    T = _TF + (H - liq * _LF) / C
    # Empty layers (w ~ 0): no sensible mass -> pin to T_freeze.
    T = jnp.where(w > _EPS, T, _TF)
    return ice, liq, T


def _remap_equal_mass(swe_ice, swe_liq, T, density):
    """Conservative 1-D remap of the pack onto ``n`` EQUAL-SWE-mass layers
    (cumulative-mass coordinate).  Preserves total ice, total liquid, total
    enthalpy ``sum(H)`` and total thickness ``sum(mass/density)`` exactly.  All arrays
    ``(..., n)``.
    """
    n = swe_ice.shape[-1]
    mass = swe_ice + swe_liq
    total = jnp.sum(mass, axis=-1, keepdims=True)
    enth = _enthalpy(swe_ice, swe_liq, T)                    # extensive, conserved
    # Thickness is conserved (as CLM's layer combination conserves dz), so a
    # merged layer's density is the mass-weighted HARMONIC mean of its parts.
    thick = jnp.where(mass > _EPS, mass / jnp.maximum(density, _EPS), 0.0)

    cum = jnp.cumsum(mass, axis=-1)
    lo_in = cum - mass
    target = total / n
    idx = jnp.arange(n)
    out_lo = idx * target
    out_hi = (idx + 1) * target
    lo = jnp.maximum(out_lo[..., :, None], lo_in[..., None, :])
    hi = jnp.minimum(out_hi[..., :, None], cum[..., None, :])
    overlap = jnp.clip(hi - lo, 0.0, None)                   # (..., n_out, n_in)
    # True safe divide: contribute nothing from a zero-mass input layer.
    frac = jnp.where(mass[..., None, :] > _EPS,
                     overlap / jnp.where(mass[..., None, :] > _EPS,
                                         mass[..., None, :], 1.0),
                     0.0)

    def _g(x):
        return jnp.sum(frac * x[..., None, :], axis=-1)

    new_ice, new_liq = _g(swe_ice), _g(swe_liq)
    new_enth, new_thick = _g(enth), _g(thick)
    new_mass = new_ice + new_liq
    empty = new_mass <= _EPS
    # Re-derive the equilibrium phase from the conserved (water, enthalpy).
    # Total WATER and total ENTHALPY are conserved by the remap; the ice/liquid
    # split re-equilibrates (physical: mixing a cold-ice layer with a
    # warm-liquid layer partially refreezes), which _phase_from_w_H enforces.
    out_ice, out_liq, out_T = _phase_from_w_H(new_mass, new_enth)
    new_density = jnp.where(empty, _RHO_EMPTY_LAYER,
                            new_mass / jnp.maximum(new_thick, _EPS))
    return out_ice, out_liq, out_T, new_density


def _thickness_and_conductivity(swe_ice, swe_liq, density, config):
    """Layer thickness [m] and Sturm-1997 conductivity [W/m/K]."""
    mass = swe_ice + swe_liq
    dz = mass / jnp.maximum(density, _EPS)
    k = _K_SNOW_REF * (density / _RHO_SNOW_REF) ** config.k_conductivity_exponent
    return dz, k


def seed_snow_state(swe, T_top, config: SnowColumnConfig = SnowColumnConfig()):
    """A pack holding ``swe`` [kg/m^2] as ICE in equal-mass layers at
    ``min(T_top, T_freeze)`` and ``seed_density`` (cold start / land IC)."""
    n = config.n_layers
    swe = jnp.maximum(jnp.asarray(swe), 0.0)
    m = jnp.broadcast_to((swe / n)[..., None], swe.shape + (n,))
    T = jnp.broadcast_to(jnp.minimum(jnp.asarray(T_top), _TF)[..., None], m.shape)
    return SnowColumnState(swe_ice=m, swe_liq=jnp.zeros_like(m), T=T,
                           density=jnp.full_like(m, config.seed_density))


def new_snow_bulk_density(T_air, wind):
    """Bulk density of newly fallen dry snow [kg/m^3], CLM5 (CTSM
    ``SnowHydrologyMod.NewSnowBulkDensity`` with ``lotmp_snowdensity_method =
    'Slater2017'`` and ``wind_dependent_snow_density = .true.``)::

        T > Tf + 2:        50 + 1.7 * 17^1.5
        Tf - 15 < T:       50 + 1.7 * (T - Tf + 15)^1.5
        colder:            -(50/15 + 0.0333*15) Tc - 0.0333 Tc^2,  Tc >= -57.55 C
        wind > 0.1 m/s:    + 266.861 * ((1 + tanh(wind/5)) / 2)^8.8

    ``T_air`` [K] the air temperature, ``wind`` [m/s] the wind speed.
    """
    x = T_air - _TF
    warm = _BIFALL_BASE_KG_M3 + _BIFALL_COEF * (_BIFALL_T_COLD_K + _BIFALL_T_WARM_K) ** 1.5
    mid = _BIFALL_BASE_KG_M3 + _BIFALL_COEF * jnp.maximum(x + _BIFALL_T_COLD_K, 0.0) ** 1.5
    tc = jnp.maximum(x, _SLATER_T_FLOOR_C)
    cold = -_SLATER_LIN * tc - _SLATER_QUAD * tc ** 2
    rho = jnp.where(x > _BIFALL_T_WARM_K, warm,
                    jnp.where(x > -_BIFALL_T_COLD_K, mid, cold))
    gust = _BIFALL_WIND_AMP_KG_M3 * (
        (1.0 + jnp.tanh(wind / _BIFALL_WIND_SCALE_M_S)) / 2.0) ** _BIFALL_WIND_EXP
    return rho + jnp.where(wind > _BIFALL_WIND_MIN_M_S, gust, 0.0)


def snow_add_mass(state: SnowColumnState, snowfall, T_snow, rho_fresh=None,
                  rain=None, T_rain=None):
    """Add snowfall (ice at ``min(T_snow, T_freeze)``, density ``rho_fresh``) and
    rain (liquid at ``max(T_rain, T_freeze)``) [kg/m^2] to the TOP layer, as mass
    + enthalpy, and re-equilibrate it (rain refreezes into a cold pack, releasing
    ``L_f``).  Snowfall adds its own thickness ``snowfall / rho_fresh`` (CLM5
    ``dz_snowf``); rain adds no thickness of its own here (the layer keeps its
    density).  ``rho_fresh=None`` is allowed only for a rain-only call
    (``snowfall`` the Python number 0).

    Enthalpy added (relative to ice at ``T_freeze``):
    ``snowfall*c_ice*(T_s - Tf) + rain*(c_liq*(T_r - Tf) + L_f)``.
    """
    if rho_fresh is None and not (isinstance(snowfall, numbers.Real) and snowfall == 0):
        raise ValueError("snow_add_mass: snowfall needs its fresh-snow density rho_fresh")
    swe_ice, swe_liq, T, density = state
    H_add = snowfall * _C_ICE * (jnp.minimum(T_snow, _TF) - _TF)
    w_add = snowfall
    if rain is not None:
        H_add = H_add + rain * (_C_LIQ * (jnp.maximum(T_rain, _TF) - _TF) + _LF)
        w_add = w_add + rain
    old_top = swe_ice[..., 0] + swe_liq[..., 0]
    top_w = old_top + w_add
    top_H = _enthalpy(swe_ice[..., 0], swe_liq[..., 0], T[..., 0]) + H_add
    ti, tl, tt = _phase_from_w_H(top_w, top_H)
    rho_top = density[..., 0]
    if rho_fresh is not None:
        # Thickness adds: dz = old_top/rho_old + snowfall/rho_fresh.
        thick = (jnp.where(old_top > _EPS, old_top / jnp.maximum(rho_top, _EPS), 0.0)
                 + snowfall / rho_fresh)
        rho_top = jnp.where(snowfall > _EPS,
                            (old_top + snowfall) / jnp.maximum(thick, _EPS), rho_top)
    return SnowColumnState(swe_ice=swe_ice.at[..., 0].set(ti),
                           swe_liq=swe_liq.at[..., 0].set(tl),
                           T=T.at[..., 0].set(tt),
                           density=density.at[..., 0].set(rho_top))


def _wind_drift_rate(bi, dz, active, wind):
    """CTSM ``WindDriftCompaction`` for every layer at once.

    As in CTSM, the result is a DENSITY rate (kg/m3/s) that CTSM adds as is to
    the fractional thickness rates [1/s]; it is ported unchanged (do not "fix"
    the units without the user: it is what CLM5 runs).  At a 1800 s step a thin,
    light, windy top layer therefore reaches the saturated-thickness floor in
    one step, exactly as in CLM5.  With this pack's 5 equal-mass layers the
    0.1 m pseudo-depth decay confines drift to the top layer of a deep pack.

    Top-down in CTSM: a layer drifts while every layer above it was
    compactable and driftable (``mobile``); the pseudo-depth ``zpseudo``
    accumulates ``dz (3.25 - SI)`` over those layers, half of it for the layer
    itself.  ``Frho = 1.25 - 0.0042 (max(rho_min, bi) - rho_min)``,
    ``MO = 0.34 (-0.583 gs - 0.833 sph + 0.833) + 0.66 Frho``,
    ``SI = -2.868 exp(-0.085 wind) + 1 + MO`` (capped at 3.25),
    rate ``= -max(0, rho_max - bi) SI exp(-zpseudo / 0.1) / tau_ref``.
    """
    Frho = _DRIFT_FRHO_0 - _DRIFT_FRHO_SLOPE * (
        jnp.maximum(_DRIFT_RHO_MIN_KG_M3, bi) - _DRIFT_RHO_MIN_KG_M3)
    MO = (_DRIFT_MO_GRAIN * (-_DRIFT_MO_GS * _DRIFT_GS - _DRIFT_MO_SPH * _DRIFT_SPH
                             + _DRIFT_MO_SPH) + _DRIFT_MO_FRHO * Frho)
    SI_raw = -_DRIFT_SI_AMP * jnp.exp(-_DRIFT_SI_WIND * wind[..., None]) + 1.0 + MO
    drifts = active & (SI_raw > 0.0)
    SI = jnp.minimum(SI_raw, _DRIFT_SI_MAX)
    # mobile at layer j: every layer above drifts (exclusive cumulative AND).
    above = jnp.cumprod(drifts.astype(dz.dtype), axis=-1)
    mobile = jnp.concatenate([jnp.ones_like(above[..., :1]), above[..., :-1]], axis=-1) > 0.5
    step = jnp.where(drifts, dz * (_DRIFT_SI_MAX - SI), 0.0)
    zpseudo = jnp.cumsum(step, axis=-1) - 0.5 * step
    rate = (-jnp.maximum(0.0, _DRIFT_RHO_MAX_KG_M3 - bi)
            * SI * jnp.exp(-zpseudo / _DRIFT_Z_DECAY_M) / _DRIFT_TAU_REF_S)
    return jnp.where(mobile & drifts, rate, 0.0)


def clm5_compaction_rate(state: SnowColumnState, wind):
    """CLM5 fractional thickness change rate per layer [1/s]; see
    :func:`_clm5_rate_and_active`."""
    return _clm5_rate_and_active(state, wind)[0]


def _clm5_rate_and_active(state: SnowColumnState, wind):
    """CLM5 fractional thickness change rate per layer [1/s] (negative =
    compaction): destructive metamorphism + Vionnet (2012) overburden + wind
    drift, as CTSM clm5.0 ``SnowHydrologyMod.SnowCompaction``::

        ddz1 = -c3 exp(-c4 (Tf - T)) [x exp(-0.046 (bi - 175)) if bi > 175]
                                     [x c5 if liquid/dz > 0.01]
        eta  = f1 f2 (bi / ceta) exp(aeta (Tf - T) + beta bi) eta0,
        f1 = 1 / (1 + 60 liq / (rho_w dz)),   ddz2 = -(burden + w/2) / eta
        ddz4 = wind drift (:func:`_wind_drift_rate`)

    with ``bi`` the layer's ice per unit thickness, ``burden`` the water above
    the layer and ``wind`` [m/s] the wind speed, shape ``state.T.shape[:-1]``.
    Zero for a (near-)saturated layer or one with <= 0.1 kg/m2 of ice.
    Thickness is the pack's own (``density`` is the in-pack density), i.e. CLM's
    form at ``frac_sno = 1``.  CLM5's melt term (``ddz3``) is NOT applied: its
    first part shrinks a layer in proportion to the mass it loses, which this
    pack already does by keeping density as mass melts; its second part is tied
    to CLM's snow-cover-fraction depletion curve, which this pack does not carry.
    """
    ice, liq, T, density = state
    mass = ice + liq
    dz = mass / jnp.maximum(density, _EPS)
    dz_safe = jnp.maximum(dz, _EPS)
    burden = jnp.cumsum(mass, axis=-1) - mass            # water above the layer
    void = 1.0 - (ice / constants.rho_ice + liq / constants.rho_water) / dz_safe
    active = (void > _CLM_VOID_MIN) & (ice > _CLM_ICE_MIN_KG_M2)
    bi = ice / dz_safe                                   # partial ice density [kg/m3]
    td = _TF - T
    ddz1 = -_CLM_C3_PER_S * jnp.exp(-_CLM_C4_PER_K * td)
    ddz1 = jnp.where(bi > _CLM_UPPLIM_DM_KG_M3,
                     ddz1 * jnp.exp(-_CLM_DM_DECAY_M3_KG
                                    * jnp.maximum(bi - _CLM_UPPLIM_DM_KG_M3, 0.0)),
                     ddz1)
    ddz1 = jnp.where(liq / dz_safe > _CLM_LIQ_FLAG_KG_M3, ddz1 * _CLM_C5_LIQ, ddz1)
    f1 = 1.0 / (1.0 + _VIONNET_F1_LIQ * liq / (constants.rho_water * dz_safe))
    bi_a = jnp.where(active, bi, _VIONNET_C_ETA_KG_M3)   # keep inactive layers finite
    eta = (f1 * _VIONNET_F2 * (bi_a / _VIONNET_C_ETA_KG_M3)
           * jnp.exp(_VIONNET_A_ETA_PER_K * td + _VIONNET_B_ETA_M3_KG * bi_a)
           * _VIONNET_ETA0)
    ddz2 = -(burden + 0.5 * mass) / eta
    ddz4 = _wind_drift_rate(bi, dz, active, jnp.asarray(wind))
    return jnp.where(active, ddz1 + ddz2 + ddz4, 0.0), active


def snow_compact(state: SnowColumnState, dt, wind):
    """CLM5 compaction of every layer at fixed mass:
    ``dz <- max(dz (1 + rate dt), dz_saturated)``, ``density = mass / dz``
    (CTSM: "Limit compaction to be no greater than fully saturated layer
    thickness")."""
    ice, liq, _, density = state
    mass = ice + liq
    dz = mass / jnp.maximum(density, _EPS)
    dz_sat = ice / constants.rho_ice + liq / constants.rho_water
    rate, active = _clm5_rate_and_active(state, wind)
    # CTSM changes only compactable layers; a (near-)saturated or ice-poor
    # layer keeps its thickness, even if denser than the floor.
    dz_new = jnp.where(active, jnp.maximum(dz * (1.0 + rate * dt), dz_sat), dz)
    return state._replace(
        density=jnp.where(mass > _EPS, mass / jnp.maximum(dz_new, _EPS), density))


def snow_remap_compact(state: SnowColumnState, dt, wind):
    """Equal-mass remap (conserves water, enthalpy and thickness), then CLM5
    compaction (:func:`snow_compact`) with the wind speed ``wind`` [m/s]."""
    return snow_compact(SnowColumnState(*_remap_equal_mass(*state)), dt, wind)


def snow_thermal_props(state: SnowColumnState, config: SnowColumnConfig,
                       f_snow, empty_cover_swe):
    """Pack thermal properties for an implicit conduction solve, CELL-MEAN.

    The pack lies on the snow-covered fraction ``f_snow`` of the cell only (CLM5
    ``frac_sno``): its layer thickness there is ``dz / f`` (``dz = mass /
    density`` is the cell-mean thickness) and conduction happens over that area,
    so a cell-mean conductance is ``f * k / dz_covered`` (CTSM SoilTemperatureMod
    solves the snow rows per covered area and hands the soil ``frac_sno * fn``).
    Returns ``(C, coeff, r_base)``: per-layer sensible heat capacity per unit
    cell area [J/m^2/K] (floored so an empty layer stays non-singular), the
    cell-mean conductance between adjacent layers [W/m^2/K] (harmonic-mean k
    over the covered node spacing, floored at ``_COEFF_MIN`` so no layer ever
    decouples), and the resistance from the base node to the pack bottom over
    the covered area ``dz_cov_last / (2 k_last)`` [m^2 K/W]; the caller adds the
    soil half-layer and weights the series conductance by ``f``
    (``soil_thermal.solve_snow_soil_thermal``).  ``f_snow = 1`` is a pack
    spread over the whole cell.

    ``empty_cover_swe`` [kg/m^2] is the limit of ``SWE / f_snow`` as the pack
    empties (``snow_depth_crit`` for the tanh cover, 0 for ``f_snow = 1``): the
    covered thickness ``dz_l / f = (m_l / M) (M / f) / rho_l`` then keeps its
    finite limit (and zero slope) at an empty pack instead of collapsing to 0.
    """
    dz, k = _thickness_and_conductivity(state.swe_ice, state.swe_liq, state.density, config)
    f = jnp.asarray(f_snow)[..., None]
    mass = state.swe_ice + state.swe_liq
    M = jnp.sum(mass, axis=-1, keepdims=True)
    has = M > 0.0
    share = jnp.where(has, mass / jnp.where(has, M, 1.0), 1.0 / mass.shape[-1])
    # M / f, keyed on f itself so the trace branch has no floored-f jump
    covered = f > _EPS
    swe_cov = jnp.where(covered, M / jnp.where(covered, f, 1.0), empty_cover_swe)
    dz_cov = share * swe_cov / jnp.maximum(state.density, _EPS)
    C = jnp.maximum(_sensible_hc(state.swe_ice, state.swe_liq), _EPS)
    coeff = jnp.maximum(f * _interface_coeff(dz_cov, k), _COEFF_MIN)
    r_base = dz_cov[..., -1] / (2.0 * k[..., -1])
    return C, coeff, r_base


def _interface_coeff(dz, k):
    dz_half = jnp.maximum(0.5 * (dz[..., :-1] + dz[..., 1:]), _DZ_HALF_MIN)
    k_half = 2.0 * k[..., :-1] * k[..., 1:] / (k[..., :-1] + k[..., 1:] + _EPS)
    return k_half / dz_half


def snow_phase_and_percolate(state: SnowColumnState,
                             config: SnowColumnConfig = SnowColumnConfig()):
    """Enthalpy re-equilibration (melt + refreeze) of every layer at its solved
    ``T``, then percolation of liquid beyond the irreducible holding capacity.

    Returns ``(state, drainage [kg/m^2], drainage_heat [J/m^2])`` — the liquid
    leaving the base and its enthalpy relative to ice at ``T_freeze``
    (``drainage*(c_liq*(T-Tf) + L_f)``).
    """
    swe_ice, swe_liq, T, density = state
    n = swe_ice.shape[-1]
    w = swe_ice + swe_liq
    H = _enthalpy(swe_ice, swe_liq, T)
    swe_ice, swe_liq, T = _phase_from_w_H(w, H)

    # Scan top->bottom: excess liquid beyond the irreducible holding capacity
    # flows down and each receiving layer re-equilibrates.  Carry BOTH the
    # downward liquid MASS and its ENTHALPY, so above-freezing liquid transports
    # its sensible heat too (not just latent L_f) and the budget closes.
    def _perc(carry, i):
        in_m, in_H = carry
        ice_i = jnp.take(swe_ice, i, axis=-1)
        liq_i = jnp.take(swe_liq, i, axis=-1)
        T_i = jnp.take(T, i, axis=-1)
        w_i = ice_i + liq_i + in_m
        H_i = _enthalpy(ice_i, liq_i, T_i) + in_H
        ice_i, liq_i, T_i = _phase_from_w_H(w_i, H_i)
        # Liquid beyond the holding capacity leaves at the layer temperature T_i,
        # carrying c_liq*(T_i-Tf)+Lf per unit mass.
        hold = config.irreducible_liq_frac * ice_i
        drain = jnp.maximum(liq_i - hold, 0.0)
        drain_H = drain * (_C_LIQ * (T_i - _TF) + _LF)
        liq_i = liq_i - drain
        return (drain, drain_H), (ice_i, liq_i, T_i)

    z = jnp.zeros(swe_ice.shape[:-1])
    (drainage, drainage_heat), (ice_s, liq_s, T_s) = jax.lax.scan(
        _perc, (z, z), jnp.arange(n))
    new_state = SnowColumnState(swe_ice=jnp.moveaxis(ice_s, 0, -1),
                                swe_liq=jnp.moveaxis(liq_s, 0, -1),
                                T=jnp.moveaxis(T_s, 0, -1), density=density)
    return new_state, drainage, drainage_heat


def step_snow_column(
    state: SnowColumnState,
    precip_snow: jnp.ndarray,
    T_air: jnp.ndarray,
    Q_top: jnp.ndarray,
    G_bottom: jnp.ndarray,
    dt: float,
    config: SnowColumnConfig = SnowColumnConfig(),
    *,
    wind: jnp.ndarray,
):
    """Advance a STAND-ALONE pack one step with a prescribed base flux.

    ``precip_snow`` snowfall rate [kg/m^2/s]; ``T_air`` fresh-snow temperature
    [K]; ``wind`` wind speed [m/s] (fresh-snow density, wind drift); ``Q_top``
    net flux INTO the pack top and ``G_bottom`` conductive flux
    from the base INTO the soil [W/m^2].  Returns ``(state, drainage [kg/m^2],
    drainage_heat [J/m^2])``.  The land model couples the pack to the soil
    implicitly instead (``soil_thermal.solve_snow_soil_thermal``).
    """
    state = snow_add_mass(state, precip_snow * dt, T_air,
                          rho_fresh=new_snow_bulk_density(T_air, wind))
    state = snow_remap_compact(state, dt, wind)
    C, coeff, _ = snow_thermal_props(state, config, 1.0, 0.0)   # stand-alone: whole cell
    diag = C / dt
    diag = diag.at[..., 1:].add(coeff)
    diag = diag.at[..., :-1].add(coeff)
    rhs = C / dt * state.T
    rhs = rhs.at[..., 0].add(Q_top)                          # surface flux INTO top
    rhs = rhs.at[..., -1].add(-G_bottom)                     # base loses G_bottom to soil
    a = jnp.pad(-coeff, [(0, 0)] * (coeff.ndim - 1) + [(1, 0)])
    c = jnp.pad(-coeff, [(0, 0)] * (coeff.ndim - 1) + [(0, 1)])
    T = thomas_solve(a, diag, c, rhs)
    return snow_phase_and_percolate(state._replace(T=T), config)


def total_water(state: SnowColumnState) -> jnp.ndarray:
    """Total pack water (ice + liquid) [kg/m^2], summed over layers."""
    return jnp.sum(state.swe_ice + state.swe_liq, axis=-1)


def column_enthalpy(state: SnowColumnState) -> jnp.ndarray:
    """Column enthalpy relative to T_freeze [J/m^2] (sensible + fusion of the
    liquid fraction), summed over layers — for energy-conservation checks."""
    return jnp.sum(_enthalpy(state.swe_ice, state.swe_liq, state.T), axis=-1)
