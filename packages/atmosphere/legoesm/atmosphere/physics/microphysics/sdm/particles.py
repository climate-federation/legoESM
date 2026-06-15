"""Super-droplet particle representation (structure-of-arrays pytree).

A super-droplet is a computational particle standing for ``multiplicity`` (ξ)
identical real droplets (Shima et al. 2009). The ensemble is stored
structure-of-arrays: each attribute is a length-``n_sd`` array, so the whole
:class:`SuperDropletState` is a JAX pytree that ``vmap``/``lax.scan`` can carry.

Mirrors ERF ``ERF_SuperDropletPCDefinitions.H`` (radius, multiplicity, species
masses). This revision carries the attributes needed by diffusional growth;
Lagrangian drivers keep positions in their own state wrapper so the legacy
particle pytree arity stays unchanged.

Conventions
-----------
* ``radius`` is the *wet effective* radius R [m]: the radius of an equivalent
  sphere of pure water of the same total condensed-water volume.
* per-droplet condensed-water mass ``m_w = (4/3) π ρ_w R³``;
* the *represented* water mass (what the grid sees) is ``ξ · m_w``.
* ``solute_mass`` [kg] is the dissolved (soluble) aerosol mass carried inside
  the droplet; 0 for a pure-water droplet.
* ``active`` is a {0,1} float flag; inactive (recycled/empty) slots are skipped.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm import constants

# (4/3) π — volume prefactor for a sphere; pure geometry, not a tunable.
_FOUR_THIRDS_PI = 4.0 / 3.0 * jnp.pi


class SuperDropletState(NamedTuple):
    """Structure-of-arrays ensemble of super-droplets. All fields ``(n_sd,)``."""

    multiplicity: jax.Array   # ξ [-]   number of real droplets represented
    radius: jax.Array         # R [m]   wet effective radius
    solute_mass: jax.Array    # [kg]    dissolved solute mass per droplet (0 = pure)
    active: jax.Array         # {0,1}   active flag


def water_mass_per_droplet(
    state: SuperDropletState,
    rho_w: float = constants.rho_water,
) -> jax.Array:
    """Condensed-water mass of one real droplet ``m_w = (4/3) π ρ_w R³`` [kg]."""
    return _FOUR_THIRDS_PI * rho_w * state.radius**3


def represented_water_mass(
    state: SuperDropletState,
    rho_w: float = constants.rho_water,
) -> jax.Array:
    """Water mass each super-droplet contributes to the grid ``ξ · m_w`` [kg]."""
    return state.active * state.multiplicity * water_mass_per_droplet(state, rho_w)


def make_monodisperse(
    n_sd: int,
    radius: float,
    multiplicity: float,
    solute_mass: float = 0.0,
    dtype=None,
) -> SuperDropletState:
    """Create ``n_sd`` identical active super-droplets.

    Parameters
    ----------
    n_sd : int
        Number of super-droplets.
    radius : float
        Initial wet radius R [m] of every droplet.
    multiplicity : float
        Initial multiplicity ξ [-] of every droplet.
    solute_mass : float, default 0.0
        Dissolved solute mass [kg] in every droplet (0 = pure water).
    dtype : optional
        Float dtype; defaults to the JAX default (x64-aware).
    """
    ones = jnp.ones((n_sd,), dtype=dtype)
    return SuperDropletState(
        multiplicity=ones * multiplicity,
        radius=ones * radius,
        solute_mass=ones * solute_mass,
        active=ones,
    )
