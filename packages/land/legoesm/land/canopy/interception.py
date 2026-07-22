"""Canonical canopy-water interception — shared by the two-leaf and CLM-ML canopies.

Precipitation falling on a vegetated surface splits into three fates:

1. **Throughfall** — the part that misses the canopy (direct) or drips off once
   the leaves are saturated (canopy drip).  It reaches the soil.
2. **Interception storage** — water held on leaf/stem surfaces, a prognostic
   store ``W_canopy`` [kg m-2] carried between steps.
3. **Wet-leaf evaporation** — intercepted water that evaporates directly off the
   wet leaf surface, at the potential rate scaled by the wetted fraction.  This
   ET component is separate from transpiration (which draws root-zone water
   through stomata); transpiration acts on the *dry* leaf fraction.

The formulation is the CLM-ML / CLM5 one (``MLCanopyWaterMod``), aggregated to a
single big-leaf layer so the two-leaf canopy and the multi-layer canopy share
the same math.  The plant-area used is ``LAI + SAI``; a two-leaf canopy with no
stem-area term passes ``sai = 0``.

Sign convention: all fluxes are rates [kg m-2 s-1] positive in the direction of
the name (``throughfall`` down onto soil, ``wet_evap`` up off the leaf; a
negative ``wet_evap`` is dew deposition adding to the store).  Water conserved:

    precip = throughfall + d(W_canopy)/dt + wet_evap

verified to machine precision by ``water_balance_residual`` and its test.

References
----------
Bonan et al. (2018) CLM-ML; Lawrence et al. (2019) CLM5 tech note, canopy
hydrology.  Constants (``dewmx``, ``fwet_exponent``, ``interception_fraction``,
``maximum_leaf_wetted_fraction``) match ``clm-ml-jax`` ``MLclm_varcon``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

__physics_contract__ = {
    "units": {
        "W_canopy": "kg m-2",
        "precip": "kg m-2 s-1",
        "throughfall": "kg m-2 s-1",
        "wet_evap": "kg m-2 s-1",
        "fwet": "1",
    },
    "signs": "throughfall positive down onto soil; wet_evap positive up "
             "(evaporation), negative down (dew); W_canopy >= 0",
    "conserves": "water: precip = throughfall + dW_canopy/dt + wet_evap",
    "differentiable": True,
    "reference": "Bonan et al. 2018 CLM-ML MLCanopyWaterMod; CLM5 tech note",
    "idealized_test": "tests/land/unit/test_interception.py "
                      "(closure to machine precision; drip caps storage; "
                      "wetted fraction bounded)",
}

# Floors that keep gradients finite where a physical quantity may hit zero
# (empty canopy, dry store).  Not tunable — pure numerics.
_EPS_PAI = 1.0e-12       # m2 m-2, guards divide by plant area
_EPS_POW = 1.0e-30       # guards 0**exponent -> inf gradient

__param_spec__ = {
    "InterceptionConfig": {
        "scheme_key": "land.canopy.interception",
        "params": {
            # CLM5 canopy hydrology closures — all per-site tunable (tier 2),
            # bounds interior to the physical range so they round-trip through
            # the trainable collector's transform inverse.
            "dewmx": {"units": "kg m-2", "bounds": (0.05, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "canopy_hydrology", "reference": "CLM5 max leaf water per unit plant area (0.1)", "shape": None},
            "interception_fraction": {"units": "1", "bounds": (0.1, 1.05), "tunable_tier": 2, "transform": "sigmoid", "category": "canopy_hydrology", "reference": "CLM5 interception efficiency (1.0)", "shape": None},
            "fwet_exponent": {"units": "1", "bounds": (0.3, 1.0), "tunable_tier": 2, "transform": "sigmoid", "category": "canopy_hydrology", "reference": "CLM5 wetted-fraction exponent (0.667)", "shape": None},
            "maximum_leaf_wetted_fraction": {"units": "1", "bounds": (0.01, 0.5), "tunable_tier": 2, "transform": "sigmoid", "category": "canopy_hydrology", "reference": "CLM5 cap on the wet leaf fraction (0.05)", "shape": None},
        },
    }
}


class InterceptionConfig(NamedTuple):
    """Canopy-water interception parameters (CLM-ML / CLM5 defaults).

    Shared by both canopy schemes so a tuned value applies consistently.  All
    four are per-site tunable closures (``__param_spec__`` tier 2).
    """
    # --- canopy hydrology (CLM5 / Bonan MLCanopyWaterMod) ---
    dewmx: float = 0.1                          # max leaf water [kg m-2 / plant-area]
    interception_fraction: float = 1.0          # fraction of precip intercepted at high LAI
    fwet_exponent: float = 0.667                # wetted-fraction power
    maximum_leaf_wetted_fraction: float = 0.05  # cap on wet leaf fraction


def max_canopy_water(pai: jnp.ndarray, cfg: InterceptionConfig) -> jnp.ndarray:
    """Maximum intercepted-water storage ``h2ocanmx = dewmx * PAI`` [kg m-2]."""
    return cfg.dewmx * jnp.maximum(pai, 0.0)


def wetted_fraction(W_canopy: jnp.ndarray, pai: jnp.ndarray,
                    cfg: InterceptionConfig) -> jnp.ndarray:
    """Wet fraction of the canopy ``fwet in [0, max_leaf_wetted]``.

    ``fwet = min((W / h2ocanmx)**fwet_exponent, maximum_leaf_wetted_fraction)``.
    Transpiration later acts on ``1 - fwet``; wet-leaf evaporation on ``fwet``.
    Empty canopy (``pai == 0``) is fully dry.
    """
    h2ocanmx = max_canopy_water(pai, cfg)
    has_pai = pai > _EPS_PAI
    # 0**p has an infinite gradient at 0; floor the base so grad stays finite.
    base = jnp.where(has_pai,
                     jnp.maximum(W_canopy / jnp.maximum(h2ocanmx, _EPS_POW),
                                 _EPS_POW),
                     _EPS_POW)
    fwet = jnp.minimum(base ** cfg.fwet_exponent,
                       cfg.maximum_leaf_wetted_fraction)
    return jnp.where(has_pai, fwet, 0.0)


def update_canopy_water(W_canopy: jnp.ndarray, precip: jnp.ndarray,
                        le_pot_wet: jnp.ndarray, pai: jnp.ndarray, dt: float,
                        cfg: InterceptionConfig, L_v: float):
    """Advance the canopy-water store one step; return the water partition.

    Ordering mirrors CLM-ML: intercept precip (cap storage, spill to drip),
    then evaporate the wet leaf at the potential rate scaled by ``fwet``
    (or deposit dew when the potential flux is downward).

    Parameters
    ----------
    W_canopy : (ncol,) intercepted-water store at the start of the step [kg m-2].
    precip   : (ncol,) rate of liquid precip onto the canopy [kg m-2 s-1].
    le_pot_wet : (ncol,) POTENTIAL latent-heat flux for a fully wet leaf
        [W m-2], positive up.  The wet-leaf evaporation demand is
        ``fwet * le_pot_wet``; a negative value is dew.
    pai      : (ncol,) plant area index ``LAI + SAI`` [m2 m-2].
    dt       : timestep [s].
    cfg      : :class:`InterceptionConfig`.
    L_v      : latent heat of vaporization [J kg-1] (``constants.L_v``); converts
        the wet-leaf latent-heat demand to a mass flux.

    Returns
    -------
    W_new : (ncol,) updated store [kg m-2], in ``[0, h2ocanmx]``.
    throughfall : (ncol,) water reaching the soil [kg m-2 s-1] = direct
        throughfall + canopy drip.
    wet_evap : (ncol,) actual wet-leaf evaporation [kg m-2 s-1], positive up;
        negative = dew retained.  Capped so the store cannot go negative.
    fwet : (ncol,) wetted fraction used [-].
    """
    pai = jnp.maximum(pai, 0.0)
    h2ocanmx = max_canopy_water(pai, cfg)

    # --- interception + throughfall (CLM5) ---
    fpi = cfg.interception_fraction * jnp.tanh(pai)   # intercepted fraction
    intercepted = precip * fpi                         # onto leaves [kg m-2 s-1]
    through_direct = precip * (1.0 - fpi)              # misses the canopy

    W_int = W_canopy + intercepted * dt                # provisional store
    drip = jnp.maximum(W_int - h2ocanmx, 0.0) / dt     # spill once saturated
    W_int = jnp.minimum(W_int, h2ocanmx)
    throughfall = through_direct + drip

    # --- wet-leaf evaporation / dew ---
    fwet = wetted_fraction(W_int, pai, cfg)
    evap_demand = fwet * le_pot_wet / L_v              # [kg m-2 s-1], +up
    # Cannot evaporate more water than is stored; dew (negative) adds freely.
    max_evap = W_int / dt                              # empties the store at most
    wet_evap = jnp.minimum(evap_demand, max_evap)
    W_new = W_int - wet_evap * dt                      # dew (wet_evap<0) grows W
    # Guard tiny negatives from round-off; conservation uses the guarded W_new.
    W_new = jnp.maximum(W_new, 0.0)
    return W_new, throughfall, wet_evap, fwet


def water_balance_residual(W_old: jnp.ndarray, W_new: jnp.ndarray,
                           precip: jnp.ndarray, throughfall: jnp.ndarray,
                           wet_evap: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Closure residual [kg m-2 s-1]: ``precip - throughfall - dW/dt - wet_evap``.

    Zero to machine precision when the store is not clamped at its floor (a
    clamp at ``W_new = 0`` legitimately destroys the sub-floor round-off; the
    test exercises the unclamped regime).
    """
    return precip - throughfall - (W_new - W_old) / dt - wet_evap
