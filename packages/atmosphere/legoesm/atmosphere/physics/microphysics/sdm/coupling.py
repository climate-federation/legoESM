"""Particle <-> grid coupling for the Super-Droplet Method.

Closes the two-way exchange between the Lagrangian super-droplets and the
Eulerian thermodynamic state of the air they are mixed in, following ERF
``SuperDropletsMoist`` (``computeQcQrWater`` deposition + ``phaseChange``
q_t closure and latent heating):

* **deposition** — sum the represented liquid mass ``Σ ξ_i m_i`` in a cell of
  volume ``V_cell`` to a liquid-water *density* [kg/m³], split into cloud
  (``R < r_rain``) and rain (``R >= r_rain``), and divide by the air density to
  obtain mixing ratios ``q_c, q_r`` [kg/kg];
* **condensation back-reaction** — when the droplets gain liquid mass ``Δq_l``
  the vapor mixing ratio drops by the same amount (total water ``q_t = q_v+q_l``
  conserved) and the temperature rises by the released latent heat
  ``ΔT = (L_v/c_p)·Δq_l``.

Pure, differentiable, ``jit``/``vmap``-friendly. Physical constants
(``rho_water``, ``L_v``, ``c_pd``) come from :mod:`legoesm.constants`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.sdm.particles import (
    SuperDropletState,
    water_mass_per_droplet,
)

__physics_contract__ = {
    "summary": (
        "Super-droplet <-> grid coupling: deposit represented liquid mass to "
        "cloud/rain mixing ratios and apply the condensation back-reaction "
        "(total-water closure + latent heating)."
    ),
    "inputs": {
        "multiplicity": "1",
        "radius": "m",
        "V_cell": "m^3",
        "rho_air": "kg/m^3",
        "q_v": "kg/kg",
        "T": "K",
        "dq_liquid": "kg/kg",
    },
    "outputs": {"q_c": "kg/kg", "q_r": "kg/kg", "q_v": "kg/kg", "T": "K"},
    "sign_convention": (
        "Deposited mixing ratios are non-negative. Condensation (Δq_l > 0) "
        "removes vapor (Δq_v = -Δq_l) and warms the air (ΔT = +L_v/c_p·Δq_l); "
        "evaporation reverses both signs. Total water q_v+q_l is conserved."
    ),
    "conserves": ["moisture", "energy"],
    "differentiable": True,
    "reference": "Shima et al. (2009) QJRMS 135:1307; ERF SuperDropletsMoist (phaseChange / computeQcQrWater)",
    "idealized_test": (
        "q_t = q_v + q_c + q_r conserved under the condensation exchange; "
        "condensing Δq raises T by exactly L_v/c_p·Δq; deposited q_l = "
        "Σξ(4/3 π ρ_w R³)/(ρ_air V_cell)."
    ),
}


def liquid_water_content(
    state: SuperDropletState,
    V_cell: float | jax.Array,
) -> jax.Array:
    """Represented liquid-water *density* ``Σ_i ξ_i m_i / V_cell`` [kg/m³]."""
    m = water_mass_per_droplet(state)
    return jnp.sum(state.active * state.multiplicity * m) / V_cell


def cloud_rain_mixing_ratios(
    state: SuperDropletState,
    V_cell: float | jax.Array,
    rho_air: float | jax.Array,
    r_rain: float,
) -> tuple[jax.Array, jax.Array]:
    """Deposit super-droplet liquid to ``(q_c, q_r)`` mixing ratios [kg/kg].

    Cloud water is the liquid carried by droplets with ``R < r_rain``; rain is
    the rest (``R >= r_rain``). Each is the represented mass density of that
    subset divided by the air density.
    """
    m = water_mass_per_droplet(state)
    represented = state.active * state.multiplicity * m  # [kg] per super-droplet
    is_cloud = state.radius < r_rain
    inv = 1.0 / (V_cell * rho_air)
    q_c = jnp.sum(jnp.where(is_cloud, represented, 0.0)) * inv
    q_r = jnp.sum(jnp.where(is_cloud, 0.0, represented)) * inv
    return q_c, q_r


def condensation_exchange(
    q_l_before: jax.Array,
    q_l_after: jax.Array,
    q_v: jax.Array,
    T: jax.Array,
    c_p: float = constants.c_pd,
    L_v: float = constants.L_v,
) -> tuple[jax.Array, jax.Array]:
    """Apply the condensation back-reaction to vapor and temperature.

    Given the droplet liquid mixing ratio before/after a growth step, the
    condensed amount ``Δq_l = q_l_after - q_l_before`` is removed from vapor and
    released as latent heat::

        q_v_new = q_v - Δq_l           (total water q_v + q_l conserved)
        T_new   = T   + (L_v/c_p)·Δq_l (latent heating; evaporation cools)

    Returns ``(q_v_new, T_new)``.
    """
    dq = q_l_after - q_l_before
    return q_v - dq, T + (L_v / c_p) * dq
