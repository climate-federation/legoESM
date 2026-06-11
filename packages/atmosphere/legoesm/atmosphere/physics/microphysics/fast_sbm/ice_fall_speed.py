"""Per-bin ice terminal velocities by category (computed; oracle VR2..VR5).

The oracle reads bin fall-speed tables (``VR1..VR5``) from data files not in
the WRF repo, so — consistent with the kernel strategy (see
``docs/specs/bin_microphysics.md``) — ice fall speeds are computed from
published power laws instead. Each ice category falls at

    V(m) = a · D^b · (ρ₀/ρ_air)^½          D = (6 m / (π ρ_cat))^{1/3}

with ``a, b`` the Locatelli & Hobbs (1974) coefficients for that habit and
``ρ_cat`` its bulk density (pristine crystals / snow aggregates are low
density and fall slowly; graupel/hail are dense and fall several times
faster — the physical reason to carry them as separate categories). The
``(ρ₀/ρ_air)^½`` Foote-du Toit density correction speeds fall in thin air.

Coefficients (SI, ``D`` in m, ``V`` in m/s) live in ``FastSBMConfig`` so
they stay tunable/faithful without module-level magic numbers; the defaults
are the Locatelli-Hobbs unrimed-aggregate and lump-graupel fits.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig

__physics_contract__ = {
    "summary": (
        "Per-bin ice terminal velocity by category — V = a·D^b·(ρ0/ρ)^½ "
        "with Locatelli-Hobbs power laws and category bulk density "
        "(computed replacement for oracle VR2..VR5 tables)."
    ),
    "inputs": {
        "masses": "kg (bin centres)",
        "rho_air": "kg m^-3",
        "category": "str (crystal/snow vs graupel/hail coefficients)",
    },
    "outputs": {"v_term": "m s^-1 (positive downward, per bin)"},
    "sign_convention": (
        "v_term > 0 (downward); monotonically increasing in particle mass; "
        "graupel (dense) faster than crystals/snow (low density) at equal "
        "mass; faster in thinner (lower-ρ_air) air."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Locatelli & Hobbs (1974) JGR 79:2185 (unrimed aggregates, lump "
        "graupel); Foote & du Toit (1969) density correction; oracle "
        "module_mp_fast_sbm.F VR2..VR5"
    ),
    "idealized_test": (
        "V ∝ D^b monotone in mass; graupel > snow at equal mass; "
        "density correction raises V as ρ_air drops; ~0.5 m/s snow and "
        "~1-3 m/s graupel at mm sizes (Locatelli-Hobbs range); "
        "differentiable in mass and ρ_air."
    ),
}

# Reference air density for the (ρ₀/ρ)^½ correction [kg/m³].
_RHO0 = 1.0
_SIX_OVER_PI = 6.0 / math.pi


def ice_fall_speed(
    masses: jax.Array,
    rho_air: jax.Array,
    category: str = "snow",
    config: FastSBMConfig = FastSBMConfig(),
) -> jax.Array:
    """Per-bin ice terminal velocity [m/s] for ``category`` ∈
    {``"snow"`` (crystals/aggregates), ``"graupel"`` (graupel/hail)}.

    ``rho_air`` may be a scalar or broadcast against the leading axes; the
    returned velocity has shape ``rho_air.shape + masses.shape`` when
    ``rho_air`` is an array (via the trailing density factor), or
    ``masses.shape`` for scalar ``rho_air``.
    """
    if category == "snow":
        a, b, rho_cat = (config.fall_a_snow, config.fall_b_snow,
                         config.rho_snow)
    elif category == "graupel":
        a, b, rho_cat = (config.fall_a_graupel, config.fall_b_graupel,
                         config.rho_graupel)
    else:
        raise ValueError(
            f"Unknown ice fall-speed category: {category!r} "
            "(expected 'snow' or 'graupel')")
    diameter = (_SIX_OVER_PI * masses / rho_cat) ** (1.0 / 3.0)
    v_base = a * diameter ** b                       # (n_bins,)
    rho_corr = jnp.sqrt(_RHO0 / rho_air)             # scalar or (...,)
    return v_base * jnp.expand_dims(rho_corr, -1) if jnp.ndim(rho_air) > 0 \
        else v_base * rho_corr
