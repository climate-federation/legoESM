"""TOPMODEL (SIMTOP) runoff parameterization — CLM4.5 (Niu et al. 2005).

An opt-in alternative to the Green-Ampt bucket runoff (``bucket_hydrology``).
Where the bucket generates saturation-excess (Dunne) runoff only once the WHOLE
column overfills, TOPMODEL represents SUB-GRID topographic wetness: a fraction
``f_sat`` of the grid cell is saturated (and runs off any rain immediately),
and the column drains a topographically-controlled BASEFLOW ``q_drai`` even
when unsaturated.  Both decay exponentially with the water-table depth ``z_wt``:

    f_sat  = f_max   * exp(-f_over * z_wt)          [saturated area fraction]
    q_drai = q_drai_max * exp(-f_drai * z_wt)       [baseflow / subsurface runoff]

For the slab bucket the water-table depth is diagnosed from the storage deficit
``z_wt = z_wt_max * (1 - W/W_max)`` (deeper when drier).  Column water is
conserved exactly (with ``limit_evaporation=True``):
``P_input == dW/dt + evap_actual + runoff`` where
``runoff = surface_runoff + baseflow``.

References
----------
- Niu, G.-Y., Yang, Z.-L., Dickinson, R. E. & Gulden, L. E. (2005). A simple
  TOPMODEL-based runoff parameterization (SIMTOP) for use in GCMs.
  J. Geophys. Res., 110, D21106.
- Oleson, K. W. et al. (2013). CLM4.5 Technical Note, NCAR/TN-503+STR, §7.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants


__param_spec__ = {
    "TopmodelConfig": {
        "scheme_key": "land.topmodel",
        "excluded": {},
        "params": {
            "f_max": {"units": "1", "bounds": (0.05, 0.6), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Niu 2005 / CLM4.5 (Oleson 2013) max saturated fraction", "shape": None},
            "f_over": {"units": "1/m", "bounds": (0.1, 5.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Niu 2005 / CLM4.5 saturated-fraction decay", "shape": None},
            "q_drai_max": {"units": "kg/m^2/s", "bounds": (1e-5, 1e-2), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Niu 2005 / CLM4.5 max baseflow", "shape": None},
            "f_drai": {"units": "1/m", "bounds": (0.5, 6.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "Niu 2005 / CLM4.5 baseflow decay", "shape": None},
            "z_wt_max": {"units": "m", "bounds": (1.0, 20.0), "tunable_tier": 2, "transform": "sigmoid", "category": "closure", "reference": "diagnostic water-table depth scale (slab deficit map)", "shape": None},
        },
    },
}


class TopmodelConfig(NamedTuple):
    """SIMTOP TOPMODEL runoff parameters (CLM4.5 defaults)."""
    f_max: float = 0.38          # Max saturated area fraction [-]
    f_over: float = 0.5          # Saturated-fraction decay with z_wt [1/m]
    q_drai_max: float = 1.0e-3   # Max baseflow (subsurface runoff) [kg/m^2/s]
    f_drai: float = 2.5          # Baseflow decay with z_wt [1/m]
    z_wt_max: float = 5.0        # Water-table depth at zero storage [m]


def water_table_depth(W, W_max, config: TopmodelConfig):
    """Diagnostic water-table depth ``z_wt = z_wt_max*(1 - W/W_max)`` [m].

    Deeper (larger) when the bucket is drier; 0 at saturation.  Clamped to
    ``[0, z_wt_max]`` so the exponentials stay in range."""
    frac = jnp.clip(W / jnp.maximum(W_max, 1e-6), 0.0, 1.0)
    return config.z_wt_max * (1.0 - frac)


def saturated_area_fraction(z_wt, config: TopmodelConfig):
    """Sub-grid saturated area fraction ``f_sat = f_max*exp(-f_over*z_wt)``."""
    return config.f_max * jnp.exp(-config.f_over * jnp.maximum(z_wt, 0.0))


def baseflow(z_wt, config: TopmodelConfig):
    """Topographic baseflow ``q_drai = q_drai_max*exp(-f_drai*z_wt)`` [kg/m^2/s]."""
    return config.q_drai_max * jnp.exp(-config.f_drai * jnp.maximum(z_wt, 0.0))


def partition_topmodel_runoff(W, P_input, evap_demand, dt, W_max,
                              K_infiltration, suction_boost,
                              config: TopmodelConfig,
                              infiltration_excess: bool = True,
                              limit_evaporation: bool = True):
    """SIMTOP TOPMODEL step — drop-in replacement for ``partition_bucket_runoff``.

    Surface runoff = saturation-excess over the saturated fraction ``f_sat``
    PLUS Green-Ampt infiltration-excess (Hortonian) over the unsaturated
    ``(1 - f_sat)``; the column additionally loses topographic ``baseflow``.

    Returns ``(W_new, evap_actual, runoff, runoff_surface, runoff_base)``
    [kg/m^2, kg/m^2/s x4], matching ``partition_bucket_runoff``'s signature so
    the caller is agnostic to the runoff scheme.  Conserves column water
    exactly (``limit_evaporation=True``): ``P_input == dW/dt + evap + runoff``.
    """
    rho_w = constants.rho_water
    z_wt = water_table_depth(W, W_max, config)
    f_sat = saturated_area_fraction(z_wt, config)

    # Surface runoff: on the saturated fraction ALL input runs off (Dunne);
    # on the unsaturated fraction the Green-Ampt infiltration cap sheds the
    # Hortonian excess.
    deficit = jnp.clip(1.0 - W / jnp.maximum(W_max, 1e-6), 0.0, 1.0)
    infil_cap = K_infiltration * rho_w * (1.0 + suction_boost * deficit)
    if infiltration_excess:
        q_horton = jnp.maximum(P_input - infil_cap, 0.0)
    else:
        q_horton = jnp.zeros_like(jnp.asarray(P_input) + jnp.asarray(infil_cap))
    runoff_surface = f_sat * P_input + (1.0 - f_sat) * q_horton
    infiltration = P_input - runoff_surface

    # Evaporation throttle (same convention as the bucket).
    if limit_evaporation:
        max_evap = jnp.maximum(W / dt + infiltration, 0.0)
        evap_actual = jnp.minimum(evap_demand, max_evap)
    else:
        evap_actual = evap_demand

    # Baseflow, capped by the water available after infiltration + evaporation
    # so it can never drive storage negative.
    q_base_demand = baseflow(z_wt, config)
    avail = jnp.maximum(W / dt + infiltration - evap_actual, 0.0)
    runoff_base = jnp.minimum(q_base_demand, avail)

    W_unclamped = W + dt * (infiltration - evap_actual - runoff_base)
    runoff_overflow = jnp.maximum(W_unclamped - W_max, 0.0) / dt   # residual Dunne
    W_new = jnp.clip(W_unclamped, 0.0, W_max)

    runoff_surface = runoff_surface + runoff_overflow
    runoff = runoff_surface + runoff_base
    return W_new, evap_actual, runoff, runoff_surface, runoff_base
