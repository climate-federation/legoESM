"""Ocean-scoped ``ConstantsConfig`` (Phase G, G-C1).

A first-class config block for the handful of physical constants the ocean
tendency path consumes, so a recipe can pin them to a reference model's values
(e.g. Veros) through the public config API instead of the ``override_constants``
monkey-patch (``ocean/fidelity/recipe_constants.py``).

Defaults reference ``legoesm.constants`` (canonical Earth), so introducing this
field is a **zero-behaviour change** until call sites are migrated to read from
it (G-C2 onward).

Scope (ocean): exactly the 5 base constants the recipe pins. EOS coefficients
are NOT here — they already live in the EOS configs (``VerosNonlin2Config`` etc.).
Derived constants (``kappa``, ``epsilon``) are atmosphere/thermo quantities and
stay in ``legoesm.constants``.

Static vs traced: ``R_earth`` and ``Omega`` are consumed at grid/mesh
construction (outside JIT) and stay concrete; ``g``, ``rho_0``, ``c_sw`` are
consumed inside the jitted tendency, so as pytree leaves they become
differentiable when a JAX scalar is supplied (a capability Fortran models lack).
"""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants


class ConstantsConfig(NamedTuple):
    """Base physical constants for the ocean tendency path. Defaults are the
    canonical ``legoesm.constants`` Earth values."""

    g: float = constants.g              # gravitational acceleration [m/s^2]
    rho_0: float = constants.rho_ocean  # Boussinesq reference density [kg/m^3]
    c_sw: float = constants.c_sw        # seawater specific heat [J/(kg K)]
    Omega: float = constants.Omega      # Earth rotation rate [rad/s] (grid-init)
    R_earth: float = constants.R_earth  # Earth radius [m] (grid-init)


# Veros canonical values, for recipes that pin constants to Veros (replaces the
# VEROS_CONSTANTS dict fed to the override_constants monkey-patch). Kept here so
# the recipe can build ConstantsConfig(**VEROS_CONSTANTS) directly (G-C4).
VEROS_CONSTANTS_CONFIG = ConstantsConfig(
    g=9.81, rho_0=1024.0, c_sw=3994.0, Omega=7.292115e-5, R_earth=6.370e6,
)


__all__ = ("ConstantsConfig", "VEROS_CONSTANTS_CONFIG")
