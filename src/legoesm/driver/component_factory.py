"""Component factory for the legoESM driver.

Resolves configured atmosphere (and future ocean/land/ice) solvers into
concrete model instances.  The driver delegates all solver construction
here, so that ``ModelDriver`` never hardcodes a particular dynamical
core class.

Design
------
* The atmosphere factory uses the two-axis resolver already in
  ``atmosphere.dynamics`` (``resolve_solver_name`` / ``create_model``),
  but wraps it with driver-specific logic:
  - Computes physical diffusion coefficients from grid properties.
  - Validates that the grid type is compatible with the chosen solver.
  - Returns the model instance ready for time-stepping.
* The factory is designed to be extended: ``create_ocean_component``,
  ``create_land_component``, ``create_ice_component``, and
  ``create_coupler`` stubs are provided for future patches.
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
            hyperdiff_coeff=diff.hyperdiff, hyperdiff_ps_coeff=0.0,
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
        )
        return SpectralPrimitiveEquationModel(
            grid=grid, sigma_coord=sigma, config=pe_config,
        )

    if solver_name == "spectral_compressible_euler":
        from legoesm.atmosphere.dynamics.spectral_nh import SpectralCompressibleEulerModel
        return SpectralCompressibleEulerModel(grid=grid, sigma_coord=sigma)

    # ----- Lat-lon FV solvers -----
    if solver_name == "fv_shallow_water_latlon":
        from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import FVShallowWaterLatLonModel
        return FVShallowWaterLatLonModel(grid=grid)

    if solver_name == "fv_primitive_equations_latlon":
        from legoesm.atmosphere.dynamics.primitive_eq_fv_latlon import FVLatLonPrimitiveEquationModel
        return FVLatLonPrimitiveEquationModel(grid=grid, sigma_coord=sigma)

    if solver_name == "fv_compressible_euler_latlon":
        from legoesm.atmosphere.dynamics.compressible_euler_fv_latlon import FVCompressibleEulerLatLonModel
        return FVCompressibleEulerLatLonModel(grid=grid, sigma_coord=sigma)

    # ----- MPAS icosahedral -----
    if solver_name == "mpas_primitive_equations":
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import MPASPrimitiveEquationModel
        return MPASPrimitiveEquationModel(mesh=grid, sigma_coord=sigma)

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
# Future component stubs — ocean, land, ice, coupler
# =========================================================================

def create_ocean_component(config: ExperimentConfig, grid, vertical_coord):
    """Create the configured ocean dynamical core.

    Not yet implemented — placeholder for the next patch.
    """
    raise NotImplementedError(
        "Ocean component factory is not yet wired. "
        "Configure the ocean solver directly for now."
    )


def create_land_component(config: ExperimentConfig, grid):
    """Create the configured land surface model.

    Not yet implemented — placeholder for the next patch.
    """
    raise NotImplementedError(
        "Land component factory is not yet wired."
    )


def create_ice_component(config: ExperimentConfig, grid):
    """Create the configured sea-ice model.

    Not yet implemented — placeholder for the next patch.
    """
    raise NotImplementedError(
        "Ice component factory is not yet wired."
    )


def create_coupler(config: ExperimentConfig, components: dict):
    """Create the component coupler.

    Not yet implemented — placeholder for the next patch.
    """
    raise NotImplementedError(
        "Coupler factory is not yet wired."
    )
