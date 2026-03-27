"""Component factory for the legoESM driver.

Resolves configured atmosphere, ocean, land, ice, and coupler
components into concrete model instances.  The driver delegates all
component construction here, so that ``ModelDriver`` never hardcodes
a particular dynamical core or surface component class.

Design
------
* The atmosphere factory uses the two-axis resolver already in
  ``atmosphere.dynamics`` (``resolve_solver_name`` / ``create_model``),
  but wraps it with driver-specific logic:
  - Computes physical diffusion coefficients from grid properties.
  - Validates that the grid type is compatible with the chosen solver.
  - Returns the model instance ready for time-stepping.
* ``create_ocean_component``, ``create_land_component``,
  ``create_ice_component``, and ``create_coupler`` delegate to the
  canonical component constructors in ``legoesm.ocean``,
  ``legoesm.land``, ``legoesm.ice``, and ``legoesm.coupler``.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.driver.config import ExperimentConfig, DycoreConfig

logger = logging.getLogger("legoesm.driver.factory")


# =========================================================================
# Supported-combination matrix
# =========================================================================

# (model_type, discretization, grid_type) -> flat solver name
#
# Only combinations listed here are supported through the driver.  Any
# other combination fails fast with a precise diagnostic message.

_DRIVER_SUPPORTED: dict[tuple[str, str, str], str] = {
    # --- Cubed-sphere C-D grid ---
    ("shallow_water", "centered",       "cubed_sphere"): "cdgrid_shallow_water",
    ("shallow_water", "cdgrid",         "cubed_sphere"): "cdgrid_shallow_water",
    ("hydrostatic",   "centered",       "cubed_sphere"): "cdgrid_primitive_equations",
    ("hydrostatic",   "cdgrid",         "cubed_sphere"): "cdgrid_primitive_equations",
    ("nonhydrostatic","centered",       "cubed_sphere"): "cdgrid_compressible_euler",
    ("nonhydrostatic","cdgrid",         "cubed_sphere"): "cdgrid_compressible_euler",

    # --- Cubed-sphere finite-volume (alias) ---
    ("shallow_water", "finite_volume",  "cubed_sphere"): "cdgrid_shallow_water",
    ("hydrostatic",   "finite_volume",  "cubed_sphere"): "cdgrid_primitive_equations",
    ("nonhydrostatic","finite_volume",  "cubed_sphere"): "cdgrid_compressible_euler",

    # --- Spectral (Gaussian grid) ---
    ("shallow_water", "spectral",       "gaussian"):     "spectral_shallow_water",
    ("hydrostatic",   "spectral",       "gaussian"):     "spectral_primitive_equations",
    ("nonhydrostatic","spectral",       "gaussian"):     "spectral_compressible_euler",

    # --- Lat-lon finite-volume ---
    ("shallow_water", "finite_volume",  "latlon"):       "fv_shallow_water_latlon",
    ("hydrostatic",   "finite_volume",  "latlon"):       "fv_primitive_equations_latlon",
    ("nonhydrostatic","finite_volume",  "latlon"):       "fv_compressible_euler_latlon",
    ("shallow_water", "centered",       "latlon"):       "fv_shallow_water_latlon",
    ("hydrostatic",   "centered",       "latlon"):       "fv_primitive_equations_latlon",
    ("nonhydrostatic","centered",       "latlon"):       "fv_compressible_euler_latlon",

    # --- MPAS icosahedral ---
    ("hydrostatic",   "mpas",           "voronoi"):      "mpas_primitive_equations",
    ("nonhydrostatic","mpas",           "voronoi"):      "mpas_compressible_euler",

    # --- SFNO data-driven ---
    ("shallow_water", "sfno",           "cubed_sphere"): "sfno_shallow_water",
    ("hydrostatic",   "sfno",           "cubed_sphere"): "sfno_primitive_equations",
}


def supported_matrix() -> list[dict[str, str]]:
    """Return the driver-supported (model_type, discretization, grid) matrix."""
    return [
        {"model_type": k[0], "discretization": k[1], "grid_type": k[2],
         "solver": v}
        for k, v in sorted(_DRIVER_SUPPORTED.items())
    ]


# =========================================================================
# Diffusion coefficient helpers
# =========================================================================

class DiffusionCoeffs(NamedTuple):
    """Physical diffusion coefficients computed from grid properties."""
    A_h: float         # Laplacian viscosity [m^2/s]
    hyperdiff: float   # Biharmonic hyperdiffusion [m^4/s]
    div_damp: float    # Divergence damping [m^2/s]


def compute_diffusion(grid, dc: DycoreConfig) -> DiffusionCoeffs:
    """Compute physical diffusion coefficients from grid and config.

    Parameters
    ----------
    grid : Grid object
        Must expose a ``dx`` attribute (minimum grid spacing in metres).
    dc : DycoreConfig
        Dycore configuration with ``dt``, ``hyperdiff_scale``,
        ``div_damp_scale``.

    Returns
    -------
    DiffusionCoeffs
    """
    dx_min = float(jnp.min(jnp.asarray(grid.dx))) / 2.0 if hasattr(grid, 'dx') else 1e5
    DT = dc.dt

    # Laplacian viscosity: CFL-safe Smagorinsky-like default
    A_h = 0.05 * dx_min ** 2 / DT

    # Biharmonic: e-folding time for grid-scale noise
    tau_efold = 24.0 * 3600.0
    hyperdiff = dc.hyperdiff_scale * dx_min ** 4 / tau_efold

    # Divergence damping: damps external gravity wave mode
    c_grav = 300.0
    div_damp = dc.div_damp_scale * c_grav * dx_min / (2.0 * 3.14159)

    return DiffusionCoeffs(A_h=A_h, hyperdiff=hyperdiff, div_damp=div_damp)


# =========================================================================
# Atmosphere factory
# =========================================================================

def create_atmosphere_dycore(
    config: ExperimentConfig,
    grid,
    sigma,
) -> Any:
    """Resolve and instantiate the configured atmosphere dynamical core.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment configuration.
    grid
        Horizontal grid (cubed-sphere, Gaussian, lat-lon, etc.).
    sigma
        Vertical coordinate.

    Returns
    -------
    model
        The instantiated dynamical core, ready for ``step_with_physics``.

    Raises
    ------
    ValueError
        If the (model_type, discretization, grid_type) combination is
        not supported.
    """
    dc = config.dycore
    gc = config.grid

    model_type = dc.model_type
    discretization = dc.discretization
    grid_type = gc.grid_type

    key = (model_type, discretization, grid_type)

    if key not in _DRIVER_SUPPORTED:
        _fail_unsupported(model_type, discretization, grid_type)

    solver_name = _DRIVER_SUPPORTED[key]
    diff = compute_diffusion(grid, dc)

    logger.info(
        "Atmosphere: model_type=%s, discretization=%s, grid=%s -> %s",
        model_type, discretization, grid_type, solver_name,
    )

    # ----- Cubed-sphere C-D grid solvers -----
    if solver_name == "cdgrid_shallow_water":
        from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
            CDGridShallowWaterModel, CDGridShallowWaterConfig,
        )
        cfg = CDGridShallowWaterConfig(
            A_h=diff.A_h,
            hyperdiff_coeff=diff.hyperdiff,
            use_conservation_fixer=dc.conservation_fixer,
            fix_mass=dc.fix_mass,
        )
        return CDGridShallowWaterModel(grid, cfg)

    if solver_name == "cdgrid_primitive_equations":
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
        )
        cfg = CDGridPrimitiveEquationConfig(
            A_h=diff.A_h,
            hyperdiff_coeff=diff.hyperdiff, hyperdiff_ps_coeff=diff.hyperdiff,
            div_damp_coeff=diff.div_damp,
            use_conservation_fixer=dc.conservation_fixer,
            fix_mass=dc.fix_mass,
        )
        return CDGridPrimitiveEquationModel(grid, sigma, cfg)

    if solver_name == "cdgrid_compressible_euler":
        from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import (
            CDGridCompressibleEulerModel, CDGridCompressibleEulerConfig,
        )
        cfg = CDGridCompressibleEulerConfig(
            A_h=diff.A_h,
            hyperdiff_coeff=diff.hyperdiff,
        )
        # Non-hydrostatic requires height coordinate and terrain metric.
        # Expect grid to provide these or construct defaults.
        height_coord = getattr(grid, "height_coord", None)
        terrain_metric = getattr(grid, "terrain_metric", None)
        if height_coord is None or terrain_metric is None:
            from legoesm.grids.vertical import (
                create_height_coordinate, compute_terrain_metric,
            )
            if height_coord is None:
                # Build flat (zero topography) height coordinate.
                nlev = sigma.sigma_full.shape[0] if hasattr(sigma, "sigma_full") else 40
                H_top = 30_000.0  # metres
                height_coord = create_height_coordinate(nlev, H_top)
            if terrain_metric is None:
                z_s = jnp.zeros_like(grid.lat)
                terrain_metric = compute_terrain_metric(z_s, height_coord)
        return CDGridCompressibleEulerModel(grid, height_coord, terrain_metric, cfg)

    # ----- Spectral solvers (Gaussian grid) -----
    if solver_name == "spectral_shallow_water":
        from legoesm.atmosphere.dynamics.spectral_sw import SpectralShallowWaterModel
        return SpectralShallowWaterModel(grid=grid)

    if solver_name == "spectral_primitive_equations":
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralPrimitiveEquationModel, SpectralPEConfig,
        )
        # Compute hyperdiffusion from truncation: 0.5-hour e-folding at max wavenumber
        n_max = grid.n_max
        a = grid.radius
        eig_max = n_max * (n_max + 1) / (a * a)
        hyperdiff = 1.0 / (0.5 * 3600.0 * eig_max ** 2)
        pe_config = SpectralPEConfig(
            hyperdiff_coeff=hyperdiff,
            hyperdiff_order=2,
            time_integrator="ssp_rk54",
            p_floor=200.0,
            dealiasing_fraction=0.667,
        )
        return SpectralPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma, config=pe_config,
        )

    if solver_name == "spectral_compressible_euler":
        from legoesm.atmosphere.dynamics.spectral_nh import SpectralCompressibleEulerModel
        return SpectralCompressibleEulerModel(grid=grid, sigma_coord=sigma)

    # ----- Lat-lon FV solvers -----
    if solver_name == "fv_shallow_water_latlon":
        from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
            FVShallowWaterLatLonModel, FVShallowWaterLatLonConfig,
        )
        cfg = FVShallowWaterLatLonConfig(
            hyperdiff_coeff=diff.hyperdiff,
            use_conservation_fixer=dc.conservation_fixer,
            fix_mass=dc.fix_mass,
        )
        return FVShallowWaterLatLonModel(grid=grid, config=cfg)

    if solver_name == "fv_primitive_equations_latlon":
        from legoesm.atmosphere.dynamics.primitive_eq_fv_latlon import (
            FVLatLonPrimitiveEquationModel, FVLatLonPrimitiveEquationConfig,
        )
        cfg = FVLatLonPrimitiveEquationConfig(
            A_h=diff.A_h,
            hyperdiff_coeff=diff.hyperdiff,
            hyperdiff_ps_coeff=diff.hyperdiff,
            use_conservation_fixer=dc.conservation_fixer,
            fix_mass=dc.fix_mass,
        )
        return FVLatLonPrimitiveEquationModel(grid=grid, sigma_coord=sigma, config=cfg)

    if solver_name == "fv_compressible_euler_latlon":
        from legoesm.atmosphere.dynamics.compressible_euler_fv_latlon import (
            FVCompressibleEulerLatLonModel, FVCompressibleEulerLatLonConfig,
        )
        cfg = FVCompressibleEulerLatLonConfig(
            hyperdiff_coeff=diff.hyperdiff,
        )
        return FVCompressibleEulerLatLonModel(grid=grid, sigma_coord=sigma, config=cfg)

    # ----- MPAS icosahedral -----
    if solver_name == "mpas_primitive_equations":
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
        )
        cfg = MPASPrimitiveEquationConfig(
            nu_del4=diff.hyperdiff,
            nu_del4_ps=diff.hyperdiff,
            fix_mass=dc.fix_mass,
        )
        return MPASPrimitiveEquationModel(mesh=grid, sigma_coord=sigma, config=cfg)

    if solver_name == "mpas_compressible_euler":
        from legoesm.atmosphere.dynamics.compressible_euler_mpas import MPASCompressibleEulerModel
        return MPASCompressibleEulerModel(mesh=grid, sigma_coord=sigma)

    # ----- SFNO data-driven -----
    if solver_name == "sfno_shallow_water":
        from legoesm.atmosphere.dynamics.sfno_sw import SFNOShallowWaterModel
        return SFNOShallowWaterModel(grid=grid)

    if solver_name == "sfno_primitive_equations":
        from legoesm.atmosphere.dynamics.sfno_pe import SFNOPrimitiveEquationModel
        return SFNOPrimitiveEquationModel(grid=grid, sigma_coord=sigma)

    # Should be unreachable — the key check above guarantees this.
    raise RuntimeError(f"Internal error: unhandled solver {solver_name!r}")


def _fail_unsupported(model_type: str, discretization: str, grid_type: str):
    """Raise a precise diagnostic for an unsupported combination."""
    # Gather what IS available for each partial match.
    available_for_grid = sorted({
        (k[0], k[1]) for k in _DRIVER_SUPPORTED if k[2] == grid_type
    })
    available_for_model = sorted({
        (k[1], k[2]) for k in _DRIVER_SUPPORTED if k[0] == model_type
    })

    msg_parts = [
        f"Unsupported atmosphere configuration: "
        f"model_type={model_type!r}, discretization={discretization!r}, "
        f"grid_type={grid_type!r}.",
    ]
    if available_for_grid:
        combos = ", ".join(f"({m}, {d})" for m, d in available_for_grid)
        msg_parts.append(
            f"  On grid_type={grid_type!r}, supported (model_type, discretization) "
            f"pairs are: {combos}."
        )
    else:
        all_grids = sorted({k[2] for k in _DRIVER_SUPPORTED})
        msg_parts.append(
            f"  grid_type={grid_type!r} has no driver-supported solvers. "
            f"Supported grid types: {all_grids}."
        )
    if available_for_model:
        combos = ", ".join(f"({d}, {g})" for d, g in available_for_model)
        msg_parts.append(
            f"  For model_type={model_type!r}, supported (discretization, grid_type) "
            f"pairs are: {combos}."
        )
    raise ValueError("\n".join(msg_parts))


# =========================================================================
# Ocean, land, ice, and coupler component factories
# =========================================================================

def create_ocean_component(
    config: ExperimentConfig,
    grid,
    vertical_coord=None,
    *,
    ocean_config=None,
    sst_map=None,
):
    """Create the configured ocean component.

    This factory supports two modes:

    1. **Simple ocean** (default): returns a step-function created by
       ``legoesm.ocean.simple_ocean.make_ocean``.  Pass an optional
       ``SimpleOceanConfig`` via *ocean_config*; if omitted, the
       default fixed-SST configuration is used.
    2. **Full ocean model**: if *ocean_config* is an ``OceanConfig``,
       instantiates a full ``OceanModel`` from ``legoesm.ocean``.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment-level configuration (used for logging).
    grid
        Horizontal grid object.
    vertical_coord
        Vertical coordinate (needed for the full ocean model; ignored
        for the simple ocean).
    ocean_config
        A ``SimpleOceanConfig`` or ``OceanConfig`` instance.  If *None*,
        defaults to ``SimpleOceanConfig()`` (fixed SST at 300 K).
    sst_map : array-like, optional
        Spatial SST map for fixed/slab modes.

    Returns
    -------
    step_fn or OceanModel
        A callable step function (simple ocean) or an ``OceanModel``
        instance (full ocean).
    """
    from legoesm.ocean.simple_ocean import SimpleOceanConfig, make_ocean
    from legoesm.ocean.state import OceanConfig

    if ocean_config is None:
        ocean_config = SimpleOceanConfig()

    if isinstance(ocean_config, SimpleOceanConfig):
        logger.info(
            "Ocean: simple mode=%s (h_mix=%.0f m)",
            ocean_config.mode, ocean_config.h_mix,
        )
        return make_ocean(ocean_config, sst_map=sst_map)

    if isinstance(ocean_config, OceanConfig):
        from legoesm.ocean import OceanModel
        logger.info("Ocean: full OceanModel")
        return OceanModel(grid=grid, vertical_coord=vertical_coord, config=ocean_config)

    raise TypeError(
        f"ocean_config must be SimpleOceanConfig or OceanConfig, "
        f"got {type(ocean_config).__name__!r}"
    )


def create_land_component(config: ExperimentConfig, grid, *, land_config=None):
    """Create the configured land surface model.

    Returns a ``step_land`` callable (slab land) or a
    ``step_multilayer_land`` callable, depending on the config type.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment-level configuration (used for logging).
    grid
        Horizontal grid object (used for determining spatial shape).
    land_config
        A ``LandConfig`` or ``MultiLayerLandConfig``.  If *None*,
        defaults to ``LandConfig()``.

    Returns
    -------
    step_fn : callable
        The land surface step function.
    """
    from legoesm.land import (
        LandConfig, MultiLayerLandConfig,
        step_land, step_multilayer_land,
    )

    if land_config is None:
        land_config = LandConfig()

    if isinstance(land_config, MultiLayerLandConfig):
        logger.info("Land: multilayer model (n_soil=%d)", land_config.n_soil)
        return step_multilayer_land

    if isinstance(land_config, LandConfig):
        logger.info("Land: slab model")
        return step_land

    raise TypeError(
        f"land_config must be LandConfig or MultiLayerLandConfig, "
        f"got {type(land_config).__name__!r}"
    )


def create_ice_component(config: ExperimentConfig, grid, *, ice_config=None):
    """Create the configured sea-ice model.

    Returns the ``step_sea_ice`` function from ``legoesm.ice``.
    The caller should also construct a ``SeaIceState`` for initialization.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment-level configuration (used for logging).
    grid
        Horizontal grid object (needed when ice dynamics are enabled).
    ice_config
        A ``SeaIceConfig`` instance.  If *None*, defaults to
        ``SeaIceConfig()``.

    Returns
    -------
    step_fn : callable
        ``step_sea_ice(state, forcing, config, dt) -> SeaIceState``.
    """
    from legoesm.ice import SeaIceConfig, step_sea_ice

    if ice_config is None:
        ice_config = SeaIceConfig()

    logger.info(
        "Ice: dynamics=%s, n_categories=%d",
        ice_config.dynamics, ice_config.n_categories,
    )
    return step_sea_ice


def create_coupler(
    config: ExperimentConfig,
    components: dict,
    *,
    coupler_config=None,
    land_config=None,
    ice_config=None,
    lake_config=None,
    lat=None,
    grid=None,
):
    """Create the surface coupler.

    Delegates to ``legoesm.coupler.coupler.make_coupler``, which
    returns a ``step_surface`` callable.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment-level configuration (used for logging).
    components : dict
        Not currently used; reserved for future multi-component
        coupling patterns.
    coupler_config
        A ``CouplerConfig`` instance.  If *None*, defaults to
        ``CouplerConfig()``.
    land_config
        A ``LandConfig`` for the land tile.  If *None*, defaults to
        ``LandConfig()``.
    ice_config
        A ``SeaIceConfig`` for the ice tile.  If *None*, defaults to
        ``SeaIceConfig()``.
    lake_config
        A ``LakeConfig`` for the lake tile.  If *None*, defaults to
        ``LakeConfig()``.
    lat : array-like, optional
        Latitude array [radians] (needed for carbon cycle).
    grid : optional
        Grid object (needed when ice dynamics or transport are enabled).

    Returns
    -------
    step_surface : callable
        ``(SurfaceState, AtmToSurface, TileConfig, ocean_sst, ocean_u,
        ocean_v, dt, doy) -> (SurfaceState, SurfaceToAtm)``.
    """
    from legoesm.coupler.coupler import make_coupler
    from legoesm.coupler.config import CouplerConfig
    from legoesm.coupler.lake import LakeConfig
    from legoesm.land import LandConfig
    from legoesm.ice import SeaIceConfig

    if coupler_config is None:
        coupler_config = CouplerConfig()
    if land_config is None:
        land_config = LandConfig()
    if ice_config is None:
        ice_config = SeaIceConfig()
    if lake_config is None:
        lake_config = LakeConfig()

    logger.info(
        "Coupler: coupling_dt=%.0f s, bulk_scheme=%s",
        coupler_config.coupling_dt, coupler_config.bulk_scheme,
    )
    return make_coupler(
        coupler_config=coupler_config,
        land_config=land_config,
        ice_config=ice_config,
        lake_config=lake_config,
        lat=lat,
        grid=grid,
    )
