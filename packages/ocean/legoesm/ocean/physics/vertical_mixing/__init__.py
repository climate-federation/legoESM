"""Vertical mixing parameterizations for the ocean."""

from legoesm.ocean.physics.vertical_mixing.config import (
    VerticalMixingConfig,
    ConstantVerticalMixingConfig,
    RichardsonVerticalMixingConfig,
    KPPConfig,
)
from legoesm.ocean.physics.vertical_mixing.output import VerticalMixingOutput
from legoesm.ocean.physics.vertical_mixing.integration import (
    make_vertical_mixing_physics,
)
from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
    implicit_vertical_diffusion_ocean,
    implicit_vertical_diffusion_ocean_batched,
    implicit_vertical_diffusion_ocean_pair,
    implicit_vertical_diffusion_nemo_momentum,
    implicit_vertical_diffusion_nemo_tracer_pair,
    implicit_vertical_diffusion_ocean_momentum_dispatch,
    implicit_vertical_diffusion_ocean_tracer_pair_dispatch,
    build_dz_half,
    nemo_e3w_kmm,
    nemo_e3w0_reference,
)
from legoesm.ocean.physics.vertical_mixing.k_profiles import (
    compute_vertical_K_profiles,
)
from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
    IWMConfig,
    IWMForcing,
    compute_iwm_diffusivity,
    uniform_iwm_forcing,
)

__all__ = [
    "IWMConfig",
    "IWMForcing",
    "compute_iwm_diffusivity",
    "uniform_iwm_forcing",
    "VerticalMixingConfig",
    "ConstantVerticalMixingConfig",
    "RichardsonVerticalMixingConfig",
    "KPPConfig",
    "VerticalMixingOutput",
    "make_vertical_mixing_physics",
    "implicit_vertical_diffusion_ocean",
    "implicit_vertical_diffusion_ocean_batched",
    "implicit_vertical_diffusion_ocean_pair",
    "implicit_vertical_diffusion_nemo_momentum",
    "implicit_vertical_diffusion_nemo_tracer_pair",
    "implicit_vertical_diffusion_ocean_momentum_dispatch",
    "implicit_vertical_diffusion_ocean_tracer_pair_dispatch",
    "build_dz_half",
    "nemo_e3w_kmm",
    "nemo_e3w0_reference",
    "compute_vertical_K_profiles",
]
