"""CLUBB saturation mixing-ratio adapter (CAM default: Flatau formula).

Thin adapter over the canonical Flatau saturation-vapor-pressure curves in
``legoesm.thermo`` (``saturation_vapor_pressure_flatau`` /
``saturation_vapor_pressure_ice_flatau``). The saturation *curve* is NOT
re-derived here (CLAUDE.md: saturation only from ``thermo``); this module only
assembles the saturation mixing ratio with CLUBB's exact denominator guard.

Faithful to ``saturation.F90`` / ``CLUBB-JAX/.../saturation.py``
(``sat_mixrat_liq`` / ``sat_mixrat_ice``) with ``I_sat_sphum = .false.``
(mixing ratio, not specific humidity):

    rsat = ep * esat / (p - esat)    where  p - esat >= 1 Pa
         = ep                         otherwise (esat ~ p, near pure vapor)

``ep`` is ``constants.epsilon`` (R_d / R_v). The denominator is clamped BEFORE
the division (``where(safe, p-esat, 1)``) so the unconditional quotient is
finite even where it is masked out — otherwise a near-zero/negative ``p-esat``
gives an inf/nan whose reverse-mode VJP poisons the gradient (``nan*0 = nan``)
even though the forward ``where`` discards it. Forward-identical to a plain
``where``, and AD-safe.

The CAM-default CLUBB tree uses ``saturation_formula = flatau`` and
``l_rcm_supersat_adj = .false.`` (so the bisection ``rcm_sat_adj`` is NOT in the
tree — CLUBB diagnoses cloud water from the assumed PDF instead, handled later
in the PDF-closure module). Hence this adapter exposes only the liquid/ice
saturation mixing ratios.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.thermo import (
    saturation_vapor_pressure_flatau,
    saturation_vapor_pressure_ice_flatau,
)

from legoesm import constants

_SAT_DENOM_MIN_PA = 1.0  # p - esat floor [Pa] (saturation.F90 convention)


def _mixrat_from_esat(p: jax.Array, esat: jax.Array) -> jax.Array:
    """Assemble rsat = ep*esat/(p-esat) with CLUBB's AD-safe denominator guard."""
    safe = (p - esat) >= _SAT_DENOM_MIN_PA
    denom_safe = jnp.where(safe, p - esat, 1.0)
    return jnp.where(safe, constants.epsilon * esat / denom_safe, constants.epsilon)


def sat_mixrat_liq(p: jax.Array, T: jax.Array) -> jax.Array:
    """Saturation mixing ratio over liquid water (Flatau), [kg/kg].

    Parameters
    ----------
    p : jax.Array
        Pressure [Pa].
    T : jax.Array
        Temperature [K] (same shape as ``p``).

    Returns
    -------
    jax.Array
        Saturation mixing ratio over liquid [kg/kg].
    """
    return _mixrat_from_esat(p, saturation_vapor_pressure_flatau(T))


def sat_mixrat_ice(p: jax.Array, T: jax.Array) -> jax.Array:
    """Saturation mixing ratio over ice (Flatau), [kg/kg].

    Parameters
    ----------
    p : jax.Array
        Pressure [Pa].
    T : jax.Array
        Temperature [K] (same shape as ``p``).

    Returns
    -------
    jax.Array
        Saturation mixing ratio over ice [kg/kg].
    """
    return _mixrat_from_esat(p, saturation_vapor_pressure_ice_flatau(T))


__all__ = ["sat_mixrat_liq", "sat_mixrat_ice"]
