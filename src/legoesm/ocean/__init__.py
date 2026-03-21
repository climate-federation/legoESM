"""Ocean component for legoESM.

Boussinesq hydrostatic primitive equations with free surface,
Wright (1997) EOS, z-star vertical coordinate, and split-explicit
barotropic/baroclinic time stepping.

Supported implementations:

=========================  ============================
Grid                       Class
=========================  ============================
Cubed-sphere (C-D grid)    ``OceanModel``
Spectral (Gaussian)        ``SpectralOceanModel``
Lat-lon (FV)               ``LatLonOceanModel``
MPAS Voronoi               ``MPASOceanModel``
SFNO (data-driven)         ``SFNOOceanModel``
=========================  ============================

Deprecated aliases for ``OceanModel(discretization=...)``:
``"centered"``, ``"finite_volume"``, ``"fv"`` all map to ``"cdgrid"``
and emit ``DeprecationWarning``.
"""

import importlib as _importlib

# -----------------------------------------------------------------------
# Core ocean imports (always needed for basic use)
# -----------------------------------------------------------------------
from legoesm.ocean.dynamics.ocean_model import OceanModel, OCEAN_DISCRETIZATIONS
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

# -----------------------------------------------------------------------
# Lazy imports for heavier/less-common subsystems (spectral, MPAS,
# lat-lon, biogeochemistry, bathymetry, freshwater) to avoid pulling in
# their transitive dependencies at package-import time.
# -----------------------------------------------------------------------
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Spectral ocean
    "SpectralOceanModel": ("legoesm.ocean.dynamics.spectral_ocean_pe", "SpectralOceanModel"),
    "rest_state_spectral_ocean": ("legoesm.ocean.dynamics.spectral_ocean_pe", "rest_state_spectral_ocean"),
    # MPAS Voronoi ocean
    "MPASOceanConfig": ("legoesm.ocean.mpas_config", "MPASOceanConfig"),
    "MPASSimpleOceanConfig": ("legoesm.ocean.mpas_config", "MPASSimpleOceanConfig"),
    "MPASSlabOceanState": ("legoesm.ocean.simple_ocean_mpas", "MPASSlabOceanState"),
    "init_mpas_slab_state": ("legoesm.ocean.simple_ocean_mpas", "init_mpas_slab_state"),
    "make_mpas_ocean": ("legoesm.ocean.simple_ocean_mpas", "make_mpas_ocean"),
    "rest_state_mpas_ocean": ("legoesm.ocean.init_mpas", "rest_state_mpas_ocean"),
    "idealized_bathymetry_mpas": ("legoesm.ocean.init_mpas", "idealized_bathymetry_mpas"),
    "reconstruct_cell_velocity": ("legoesm.ocean.init_mpas", "reconstruct_cell_velocity"),
    "mpas_ocean_conservation_fixer": ("legoesm.ocean.conservation_mpas", "mpas_ocean_conservation_fixer"),
    "MPASOceanModel": ("legoesm.ocean.dynamics.ocean_model_mpas", "MPASOceanModel"),
    # Freshwater
    "FreshwaterForcing": ("legoesm.ocean.freshwater", "FreshwaterForcing"),
    "zero_freshwater": ("legoesm.ocean.freshwater", "zero_freshwater"),
    "net_freshwater_flux": ("legoesm.ocean.freshwater", "net_freshwater_flux"),
    "freshwater_eta_tendency": ("legoesm.ocean.freshwater", "freshwater_eta_tendency"),
    "virtual_salt_flux": ("legoesm.ocean.freshwater", "virtual_salt_flux"),
    "freshwater_from_coupler": ("legoesm.ocean.freshwater", "freshwater_from_coupler"),
    # Biogeochemistry
    "BiogeoConfig": ("legoesm.ocean.biogeochemistry", "BiogeoConfig"),
    "OceanBiogeoState": ("legoesm.ocean.biogeochemistry", "OceanBiogeoState"),
    "BiogeoTendencies": ("legoesm.ocean.biogeochemistry", "BiogeoTendencies"),
    "AirSeaCO2Diagnostics": ("legoesm.ocean.biogeochemistry", "AirSeaCO2Diagnostics"),
    "init_biogeo_state": ("legoesm.ocean.biogeochemistry", "init_biogeo_state"),
    "step_ocean_biogeochemistry": ("legoesm.ocean.biogeochemistry", "step_ocean_biogeochemistry"),
    "compute_biogeo_tendencies": ("legoesm.ocean.biogeochemistry", "compute_biogeo_tendencies"),
    "air_sea_co2_flux": ("legoesm.ocean.biogeochemistry", "air_sea_co2_flux"),
    "solve_carbonate_system": ("legoesm.ocean.biogeochemistry", "solve_carbonate_system"),
    # Lat-lon FV ocean
    "LatLonOceanModel": ("legoesm.ocean.dynamics.ocean_model_latlon", "LatLonOceanModel"),
    "LatLonOceanState": ("legoesm.ocean.state", "LatLonOceanState"),
    "LatLonOceanTendencies": ("legoesm.ocean.state", "LatLonOceanTendencies"),
    "LatLonOceanConfig": ("legoesm.ocean.state", "LatLonOceanConfig"),
    "rest_state_latlon_ocean": ("legoesm.ocean.init_latlon", "rest_state_latlon_ocean"),
    "idealized_bathymetry_latlon": ("legoesm.ocean.init_latlon", "idealized_bathymetry_latlon"),
    "wind_driven_gyre_latlon": ("legoesm.ocean.init_latlon", "wind_driven_gyre_latlon"),
    "latlon_ocean_conservation_fixer": ("legoesm.ocean.conservation_latlon", "latlon_ocean_conservation_fixer"),
    # Bathymetry
    "BathymetryConfig": ("legoesm.ocean.bathymetry", "BathymetryConfig"),
    "CRITICAL_STRAITS": ("legoesm.ocean.bathymetry", "CRITICAL_STRAITS"),
    "init_ocean_bathymetry": ("legoesm.ocean.bathymetry", "init_ocean_bathymetry"),
    "load_bathymetry_cubed_sphere": ("legoesm.ocean.bathymetry", "load_bathymetry_cubed_sphere"),
    "load_bathymetry_mpas": ("legoesm.ocean.bathymetry", "load_bathymetry_mpas"),
    "load_bathymetry_gaussian": ("legoesm.ocean.bathymetry", "load_bathymetry_gaussian"),
    "rest_state_ocean_realistic": ("legoesm.ocean.bathymetry", "rest_state_ocean_realistic"),
    "enforce_straits": ("legoesm.ocean.bathymetry", "enforce_straits"),
}

_LAZY_CACHE: dict[str, object] = {}

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
    # Biogeochemistry
    "BiogeoConfig",
    "OceanBiogeoState",
    "BiogeoTendencies",
    "AirSeaCO2Diagnostics",
    "init_biogeo_state",
    "step_ocean_biogeochemistry",
    "compute_biogeo_tendencies",
    "air_sea_co2_flux",
    "solve_carbonate_system",
    # Bathymetry
    "BathymetryConfig",
    "CRITICAL_STRAITS",
    "init_ocean_bathymetry",
    "load_bathymetry_cubed_sphere",
    "load_bathymetry_mpas",
    "load_bathymetry_gaussian",
    "rest_state_ocean_realistic",
    "enforce_straits",
    # Lat-lon FV ocean
    "LatLonOceanModel",
    "LatLonOceanState",
    "LatLonOceanTendencies",
    "LatLonOceanConfig",
    "rest_state_latlon_ocean",
    "idealized_bathymetry_latlon",
    "wind_driven_gyre_latlon",
    "latlon_ocean_conservation_fixer",
    # Freshwater
    "FreshwaterForcing",
    "zero_freshwater",
    "net_freshwater_flux",
    "freshwater_eta_tendency",
    "virtual_salt_flux",
    "freshwater_from_coupler",
]


def __getattr__(name):
    """Lazy imports for ocean subsystems and ML-dependent components."""
    # SFNO (ML-dependent, separate lazy path)
    _sfno_names = {"SFNOOceanModel", "SFNOOceanConfig"}
    if name in _sfno_names:
        from legoesm.ocean.dynamics.sfno_ocean import SFNOOceanModel, SFNOOceanConfig
        _map = {"SFNOOceanModel": SFNOOceanModel, "SFNOOceanConfig": SFNOOceanConfig}
        return _map[name]
    # General lazy imports (spectral, MPAS, lat-lon, biogeo, bathymetry, freshwater)
    if name in _LAZY_IMPORTS:
        if name in _LAZY_CACHE:
            return _LAZY_CACHE[name]
        mod_path, attr = _LAZY_IMPORTS[name]
        mod = _importlib.import_module(mod_path)
        obj = getattr(mod, attr)
        _LAZY_CACHE[name] = obj
        return obj
    raise AttributeError(f"module 'legoesm.ocean' has no attribute {name!r}")
