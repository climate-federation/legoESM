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
  (``coalescence.py``); stochastic, pure function of an explicit PRNG key, not
  differentiable.
* **collision kernels + terminal velocities** (``kernels.py``).
* **particle <-> grid coupling** — deposition, latent heating, total-water
  closure (``coupling.py``).
* **sedimentation + surface rain accumulation** (``sedimentation.py``):
  terminal-velocity fall with exact airborne-water + precipitation
  conservation; ``column_rainout`` quiescent-column driver.
* **drivers** (``box_model.py``): a composed persistent box
  (``box_step`` / ``run_box``, condensation + coalescence) and an adiabatic
  parcel (``parcel_step`` / ``run_parcel``).
* **switchable column scheme** ``scheme="sdm"`` (``column.py``,
  ``sdm_microphysics``) — a stateless diffusional-condensation adapter for the
  Eulerian column-physics interface.

Flow advection of the particles (resolved-wind transport) is not implemented;
the drivers are quiescent (0-D box / 1-D still column).
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
    hall_kernel,
    long_kernel,
    sedimentation_kernel,
    terminal_velocity,
    terminal_velocity_atlas_ulbrich,
    terminal_velocity_cloud_rain_shima,
    terminal_velocity_rogers_yau,
)
from legoesm.atmosphere.physics.microphysics.sdm.coalescence import (
    coalescence_step,
    represented_number,
)
from legoesm.atmosphere.physics.microphysics.sdm.coupling import (
    cloud_rain_mixing_ratios,
    condensation_exchange,
    liquid_water_content,
)
from legoesm.atmosphere.physics.microphysics.sdm.box_model import (
    BoxState,
    ParcelState,
    box_step,
    box_water,
    liquid_mixing_ratio,
    parcel_step,
    run_box,
    run_parcel,
    saturation_ratio,
)
from legoesm.atmosphere.physics.microphysics.sdm.column import sdm_microphysics
from legoesm.atmosphere.physics.microphysics.sdm.sedimentation import (
    column_rainout,
    sediment_step,
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
    "hall_kernel",
    "terminal_velocity",
    "terminal_velocity_rogers_yau",
    "terminal_velocity_atlas_ulbrich",
    "terminal_velocity_cloud_rain_shima",
    "coalescence_step",
    "represented_number",
    "liquid_water_content",
    "cloud_rain_mixing_ratios",
    "condensation_exchange",
    "ParcelState",
    "parcel_step",
    "run_parcel",
    "BoxState",
    "box_step",
    "run_box",
    "box_water",
    "liquid_mixing_ratio",
    "saturation_ratio",
    "sdm_microphysics",
    "sediment_step",
    "column_rainout",
]
