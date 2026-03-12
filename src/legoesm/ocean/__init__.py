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
from legoesm.ocean.mpas_config import MPASOceanConfig, MPASSimpleOceanConfig
from legoesm.ocean.simple_ocean_mpas import (
    MPASSlabOceanState,
    init_mpas_slab_state,
    make_mpas_ocean,
)
from legoesm.ocean.init_mpas import (
    rest_state_mpas_ocean,
    idealized_bathymetry_mpas,
    reconstruct_cell_velocity,
)
from legoesm.ocean.conservation_mpas import mpas_ocean_conservation_fixer
from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    zero_freshwater,
    net_freshwater_flux,
    freshwater_eta_tendency,
    virtual_salt_flux,
    freshwater_from_coupler,
)
from legoesm.ocean.bathymetry import (
    BathymetryConfig,
    CRITICAL_STRAITS,
    init_ocean_bathymetry,
    load_bathymetry_cubed_sphere,
    load_bathymetry_mpas,
    load_bathymetry_gaussian,
    rest_state_ocean_realistic,
    enforce_straits,
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
    # MPAS Voronoi ocean
    "MPASOceanModel",
    "MPASOceanConfig",
    "MPASSimpleOceanConfig",
    "MPASSlabOceanState",
    "init_mpas_slab_state",
    "make_mpas_ocean",
    "rest_state_mpas_ocean",
    "idealized_bathymetry_mpas",
    "reconstruct_cell_velocity",
    "mpas_ocean_conservation_fixer",
    # Bathymetry
    "BathymetryConfig",
    "CRITICAL_STRAITS",
    "init_ocean_bathymetry",
    "load_bathymetry_cubed_sphere",
    "load_bathymetry_mpas",
    "load_bathymetry_gaussian",
    "rest_state_ocean_realistic",
    "enforce_straits",
    # Freshwater
    "FreshwaterForcing",
    "zero_freshwater",
    "net_freshwater_flux",
    "freshwater_eta_tendency",
    "virtual_salt_flux",
    "freshwater_from_coupler",
]
