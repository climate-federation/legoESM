"""Ocean component for legoESM.

Boussinesq hydrostatic primitive equations with free surface,
Wright (1997) EOS, z-star vertical coordinate, and split-explicit
barotropic/baroclinic time stepping.

Two discretizations:
- Finite-volume on cubed-sphere: OceanModel
- Spectral on Gaussian grid: SpectralOceanModel
"""

from legoesm.ocean.dynamics.ocean_model import OceanModel
from legoesm.ocean.dynamics.spectral_ocean_pe import (
    SpectralOceanModel,
    rest_state_spectral_ocean,
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

__all__ = [
    "OceanModel",
    "SpectralOceanModel",
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
]
