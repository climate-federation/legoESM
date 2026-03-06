"""Ocean component for legoESM.

Boussinesq hydrostatic primitive equations with free surface,
Wright (1997) EOS, z-star vertical coordinate, and split-explicit
barotropic/baroclinic time stepping.

Discretizations:
- Centered finite differences on cubed-sphere: OceanModel(discretization="centered")
- FC-Gram spectral on cubed-sphere: OceanModel(discretization="fc_gram")
- FC-Gram + div damping on cubed-sphere: OceanModel(discretization="fc_gram_cgrid")
- Spectral on Gaussian grid: SpectralOceanModel
"""

from legoesm.ocean.dynamics.ocean_model import OceanModel, OCEAN_DISCRETIZATIONS
from legoesm.ocean.dynamics.spectral_ocean_pe import (
    SpectralOceanModel,
    rest_state_spectral_ocean,
)
from legoesm.ocean.dynamics.sfno_ocean import (
    SFNOOceanModel,
    SFNOOceanConfig,
)
from legoesm.ocean.state import (
    OceanState,
    OceanTendencies,
    OceanConfig,
    SpectralOceanState,
    SpectralOceanConfig,
)
from legoesm.ocean.eos import wright_eos, density_perturbation
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    create_ocean_z_star,
    compute_layer_thickness,
    compute_ocean_jacobian,
)
from legoesm.ocean.init import (
    rest_state_ocean,
    idealized_bathymetry,
    wind_driven_gyre_init,
)
from legoesm.ocean.simple_ocean import (
    SimpleOceanConfig,
    SlabOceanState,
    make_ocean,
    init_slab_state,
)

__all__ = [
    "OceanModel",
    "OCEAN_DISCRETIZATIONS",
    "SpectralOceanModel",
    "SFNOOceanModel",
    "SFNOOceanConfig",
    "OceanState",
    "OceanTendencies",
    "OceanConfig",
    "SpectralOceanState",
    "SpectralOceanConfig",
    "wright_eos",
    "density_perturbation",
    "OceanZStarCoordinate",
    "create_ocean_z_star",
    "compute_layer_thickness",
    "compute_ocean_jacobian",
    "rest_state_ocean",
    "rest_state_spectral_ocean",
    "idealized_bathymetry",
    "wind_driven_gyre_init",
    "SimpleOceanConfig",
    "SlabOceanState",
    "make_ocean",
    "init_slab_state",
]
