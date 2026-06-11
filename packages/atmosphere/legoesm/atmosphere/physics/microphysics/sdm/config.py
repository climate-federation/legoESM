"""Configuration for the Super-Droplet Method (SDM) microphysics.

All *tunable* scheme parameters live here (audit rule: no magic numbers in JAX
hot loops). Physical *constants* (vapor diffusivity, air thermal conductivity,
water surface tension, gas constants, latent heat, water density) are pulled
from :mod:`legoesm.constants` inside the physics modules, never duplicated here.

The field set grows by iteration as processes are added (collision kernels,
terminal velocity, initialization spectra). Only fields consumed by the current
code are declared, so no dead config fields accumulate.
"""

from __future__ import annotations

from typing import NamedTuple


class SDMConfig(NamedTuple):
    """Super-Droplet Method configuration.

    Fields
    ------
    n_substeps_condensation : int
        Number of equal sub-steps used to integrate the diffusional-growth ODE
        across one physics step. Small droplets near activation are stiff;
        increase this for accuracy. Must be >= 1.
    condensation_integrator : str
        ODE integrator for droplet growth: ``"rk4"`` (4th-order Runge-Kutta,
        default) or ``"euler"`` (forward Euler). Unknown values raise.
    include_curvature : bool
        Include the Kelvin curvature term (raises equilibrium vapor pressure
        over a curved surface). True is the physically complete Köhler growth.
    include_solute : bool
        Include the Raoult solute term (dissolved aerosol lowers equilibrium
        vapor pressure). True is the physically complete Köhler growth.
    solute_ionization : float
        Van't Hoff ionization factor ``i`` of the dissolved aerosol
        (NaCl -> 2). Used to convert solute mass to effective solute moles.
    solute_molar_mass : float
        Molar mass of the dissolved aerosol [kg/mol] (NaCl -> 0.05844).
    """

    n_substeps_condensation: int = 1
    condensation_integrator: str = "rk4"
    include_curvature: bool = True
    include_solute: bool = True
    solute_ionization: float = 2.0        # van't Hoff i for NaCl
    solute_molar_mass: float = 0.05844    # [kg/mol] NaCl
