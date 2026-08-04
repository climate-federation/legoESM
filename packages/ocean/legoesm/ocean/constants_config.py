"""Ocean-scoped ``ConstantsConfig`` (Phase G, G-C1).

A first-class config block for the handful of physical constants the ocean
tendency path consumes, so a recipe can pin them to a reference model's values
(e.g. Veros) through the public config API. This replaced an earlier
``override_constants`` monkey-patch of the ``legoesm.constants`` module, which
was deleted in G-C4 once the recipe pinned every constant via config.

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


# Veros canonical values, for recipes that pin constants to Veros. This is the
# single source of truth for "what Veros uses" — the ACC recipe references it
# directly (G-C4) so no separate constants dict / monkey-patch is needed.
VEROS_CONSTANTS_CONFIG = ConstantsConfig(
    g=9.81, rho_0=1024.0, c_sw=3994.0, Omega=7.292115e-5, R_earth=6.370e6,
)


# NEMO 5.0.2 canonical values, for recipes that pin constants to NEMO. Read out
# of NEMO itself, not from a reference table:
#   grav  = 9.80665            src/OCE/DOM/phycst.F90:38
#   omega = 7.292116e-05       src/OCE/DOM/phycst.F90:89
#   ra    = 6371229.           src/OCE/DOM/phycst.F90:37
#   rho0  = 1026.              src/OCE/TRA/eosbn2.F90:1898
#   rcp   = 3991.86795711963   src/OCE/TRA/eosbn2.F90:1899
#
# ``g`` is the one that matters, and the one legoESM had wrong for the NEMO
# oracle: ``legoesm.constants.g`` is the canonical Earth value, whereas NEMO
# uses STANDARD gravity -- a 5.0e-5 relative difference that enters EVERY
# buoyancy term (N^2, isopycnal slopes, the pressure gradient).  Measured on the
# DINO y5 twin against NEMO's own dumped ``rn2b`` (#1226): adopting NEMO's g cut
# the N^2 median relative error from 4.95e-05 to 6.96e-06 -- that single
# constant WAS the whole remaining bn2 residual, once the live-e3w z-star
# stretch was accounted for.
NEMO_CONSTANTS_CONFIG = ConstantsConfig(
    g=9.80665, rho_0=1026.0, c_sw=3991.86795711963,
    # Deliberately NOT legoesm.constants: reproducing NEMO's own numbers IS the
    # point of this preset (provenance in the block comment above).
    Omega=7.292116e-05, R_earth=6371229.0,  # const-ok: NEMO phycst.F90 omega/ra
)


__all__ = ("ConstantsConfig", "VEROS_CONSTANTS_CONFIG", "NEMO_CONSTANTS_CONFIG")
