"""Reduce a monthly climatology to per-cell climate features.

Companion to ``legoesm.land.climate_forcing`` (Task 1): that module goes
features -> single-column forcing; this one goes the other direction --
a 12-month climatology per grid cell -> the handful of scalar climate
features (``mat_k``, ``map_yr``, ``t_seasonal_amp_k``, ``aridity``,
``sw_mean_w``) used to key a global carbon initial-condition archetype
lookup. Pure numpy: this runs once, offline, over a full global grid
(``ncell`` up to O(1e5-1e6)) to build a lookup table, not inside any
JAX-traced model step, so there is no autodiff/JIT requirement here.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from legoesm import constants

_SECONDS_PER_YEAR = 365.0 * 86400.0
_PT_ALPHA = 1.26            # Priestley-Taylor coefficient [-]
_PT_GAMMA_OVER_S = 0.5      # (gamma/(s+gamma)) ~ 0.5 mid-latitude proxy [-]
_PET_MIN = 1e-9             # coeff-ok: divide-by-zero floor on PET


class ClimateFeatures(NamedTuple):
    mat_k: np.ndarray
    map_yr: np.ndarray
    t_seasonal_amp_k: np.ndarray
    aridity: np.ndarray
    sw_mean_w: np.ndarray


def reduce_climatology_to_features(
    monthly_t_k, monthly_precip_rate, monthly_sw_w, monthly_netrad_w
) -> ClimateFeatures:
    """Reduce ``(ncell, 12)`` monthly climatologies to per-cell features.

    Parameters
    ----------
    monthly_t_k : array (ncell, 12)
        Monthly-mean near-surface air temperature [K].
    monthly_precip_rate : array (ncell, 12)
        Monthly-mean precipitation rate [kg/m2/s].
    monthly_sw_w : array (ncell, 12)
        Monthly-mean downward shortwave [W/m2].
    monthly_netrad_w : array (ncell, 12)
        Monthly-mean surface net radiation [W/m2].

    Returns
    -------
    ClimateFeatures
        Five ``(ncell,)`` arrays: annual-mean T, mean annual precipitation
        [kg/m2/yr], half-range seasonal T amplitude [K], an aridity index
        (mean annual precipitation over Priestley-Taylor PET), and the
        annual-mean downward shortwave [W/m2].
    """
    monthly_t_k = np.asarray(monthly_t_k)
    monthly_precip_rate = np.asarray(monthly_precip_rate)
    monthly_sw_w = np.asarray(monthly_sw_w)
    monthly_netrad_w = np.asarray(monthly_netrad_w)

    mat = monthly_t_k.mean(axis=1)
    map_yr = monthly_precip_rate.mean(axis=1) * _SECONDS_PER_YEAR
    t_seasonal = 0.5 * (monthly_t_k.max(axis=1) - monthly_t_k.min(axis=1))
    # PET [kg/m2/yr] from net radiation via Priestley-Taylor (energy/L_v -> mass).
    pet = (
        _PT_ALPHA
        * _PT_GAMMA_OVER_S
        * np.maximum(monthly_netrad_w.mean(axis=1), 0.0)
        / constants.L_v
        * _SECONDS_PER_YEAR
    )
    aridity = map_yr / np.maximum(pet, _PET_MIN)
    sw_mean = monthly_sw_w.mean(axis=1)
    return ClimateFeatures(mat, map_yr, t_seasonal, aridity, sw_mean)


def priestley_taylor_pet_rate(netrad_w, t_k, p_pa):
    """Priestley & Taylor (1972) potential evapotranspiration [kg m-2 s-1].

    ``PET = alpha * s / (s + gamma) * max(Rn, 0) / L_v`` with alpha = 1.26,
    ``s`` the slope of the saturation vapour-pressure curve and ``gamma`` the
    psychrometric constant, both from the canopy's own meteorology helper
    (``canopy_met_variables``; no re-derived saturation curve).  Unlike the
    fixed 0.5 proxy of :func:`reduce_climatology_to_features`, ``s/(s+gamma)``
    here follows temperature (~0.5 at 7 degC, ~0.77 at 30 degC), which the
    Sandoval et al. (2026) aridity index needs: its peak quantum yield falls
    as AI^4.09.  Element-wise; intended for daily-mean inputs.
    """
    import jax.numpy as jnp
    from legoesm.land.canopy.energy_balance import canopy_met_variables

    t = jnp.asarray(t_k, dtype=jnp.float64)
    p = jnp.asarray(p_pa, dtype=jnp.float64)
    # Saturated air: only the slope s and gamma are used, not humidity.
    _, _, _, _, s, _, gamma = canopy_met_variables(p, t, jnp.zeros_like(t))
    rn = jnp.maximum(jnp.asarray(netrad_w, dtype=jnp.float64), 0.0)
    return np.asarray(_PT_ALPHA * s / (s + gamma) * rn / constants.L_v)
