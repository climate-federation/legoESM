"""Super-Droplet Method (SDM) microphysics — particle-based, probabilistic.

Faithful JAX port of the Super-Droplet Method (Shima et al. 2009,
QJRMS 135:1307-1320), using ERF ``Source/Microphysics/SuperDropletsMoist`` +
``Source/Particles/ERF_SuperDropletPC*`` as the algorithmic oracle. See
``docs/specs/superdroplet_sdm.md`` for the full oracle digest and port plan.

SDM represents the droplet population by a small number of *super-droplets*,
each a computational particle carrying a multiplicity ``ξ`` (the number of
identical real droplets it stands for). The processes are:

* **condensation/evaporation** — diffusional growth of each droplet
  (``condensation.py``); differentiable.
* **collision-coalescence** — Shima's Monte-Carlo algorithm
  (``coalescence.py``, later iteration); stochastic, not differentiable.
* **sedimentation/advection** — terminal-velocity fall (later iteration).

This sub-package is being built incrementally; this revision provides the
particle representation, configuration, and the diffusional-growth core.
"""

from __future__ import annotations

from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.particles import (
    SuperDropletState,
    make_monodisperse,
    represented_water_mass,
    water_mass_per_droplet,
)
from legoesm.atmosphere.physics.microphysics.sdm.condensation import (
    drsq_dt,
    integrate_radius,
)
from legoesm.atmosphere.physics.microphysics.sdm.kernels import (
    collision_kernel,
    golovin_kernel,
    long_kernel,
    sedimentation_kernel,
    terminal_velocity,
    terminal_velocity_atlas_ulbrich,
    terminal_velocity_cloud_rain_shima,
    terminal_velocity_rogers_yau,
)

__all__ = [
    "SDMConfig",
    "SuperDropletState",
    "make_monodisperse",
    "represented_water_mass",
    "water_mass_per_droplet",
    "drsq_dt",
    "integrate_radius",
    "collision_kernel",
    "golovin_kernel",
    "sedimentation_kernel",
    "long_kernel",
    "terminal_velocity",
    "terminal_velocity_rogers_yau",
    "terminal_velocity_atlas_ulbrich",
    "terminal_velocity_cloud_rain_shima",
]
