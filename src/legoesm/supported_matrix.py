"""Supported solver / discretization matrix for legoESM.

This module is the single source of truth for which dynamical core +
discretization + grid combinations are genuinely distinct implementations.
Aliases and legacy names that map to one of these implementations are
documented separately so that users cannot accidentally believe two names
represent different code paths.

The matrix is intentionally flat: each entry is a unique ``(component,
dynamics, grid)`` triple backed by its own solver class.

Deprecation schedule
--------------------
* **v0.next** (this release): Legacy names emit ``DeprecationWarning``.
* **v0.next+1**: Legacy names removed; only canonical names accepted.
"""
from __future__ import annotations

from typing import NamedTuple


class SolverEntry(NamedTuple):
    """One genuinely distinct solver implementation."""
    component: str        # "atmosphere" | "ocean"
    dynamics: str         # "shallow_water" | "hydrostatic" | "nonhydrostatic"
    grid: str             # "cubed_sphere_cdgrid" | "spectral" | "latlon_fv" | ...
    canonical_name: str   # Canonical flat name for create_model / config
    class_name: str       # Canonical Python class name
    module: str           # Defining module path


# =====================================================================
# Atmosphere
# =====================================================================

ATMOSPHERE_MATRIX: tuple[SolverEntry, ...] = (
    # -- Cubed-sphere C-D grid (FV3-style) --
    SolverEntry(
        "atmosphere", "shallow_water", "cubed_sphere_cdgrid",
        "cdgrid_shallow_water", "CDGridShallowWaterModel",
        "legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid",
    ),
    SolverEntry(
        "atmosphere", "hydrostatic", "cubed_sphere_cdgrid",
        "cdgrid_primitive_equations", "CDGridPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.primitive_eq_cdgrid",
    ),
    SolverEntry(
        "atmosphere", "nonhydrostatic", "cubed_sphere_cdgrid",
        "cdgrid_compressible_euler", "CDGridCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.compressible_euler_cdgrid",
    ),

    # -- Spectral (Gaussian grid) --
    SolverEntry(
        "atmosphere", "shallow_water", "spectral_gaussian",
        "spectral_shallow_water", "SpectralShallowWaterModel",
        "legoesm.atmosphere.dynamics.spectral_sw",
    ),
    SolverEntry(
        "atmosphere", "hydrostatic", "spectral_gaussian",
        "spectral_primitive_equations", "SpectralPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.spectral_pe",
    ),
    SolverEntry(
        "atmosphere", "nonhydrostatic", "spectral_gaussian",
        "spectral_compressible_euler", "SpectralCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.spectral_nh",
    ),

    # -- SFNO (data-driven) --
    SolverEntry(
        "atmosphere", "shallow_water", "sfno",
        "sfno_shallow_water", "SFNOShallowWaterModel",
        "legoesm.atmosphere.dynamics.sfno_sw",
    ),
    SolverEntry(
        "atmosphere", "hydrostatic", "sfno",
        "sfno_primitive_equations", "SFNOPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.sfno_pe",
    ),

    # -- MPAS icosahedral --
    SolverEntry(
        "atmosphere", "hydrostatic", "mpas_voronoi",
        "mpas_primitive_equations", "MPASPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.primitive_eq_mpas",
    ),
    SolverEntry(
        "atmosphere", "nonhydrostatic", "mpas_voronoi",
        "mpas_compressible_euler", "MPASCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.compressible_euler_mpas",
    ),

    # -- Lat-lon C-grid (FV) --
    SolverEntry(
        "atmosphere", "shallow_water", "latlon_cgrid",
        "latlon_cgrid_shallow_water", "CGridLatLonShallowWaterModel",
        "legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid",
    ),
    SolverEntry(
        "atmosphere", "hydrostatic", "latlon_cgrid",
        "latlon_cgrid_primitive_equations", "CGridLatLonPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid",
    ),

    # -- Doubly-periodic plane (CRM rollout, PR2c) --
    SolverEntry(
        "atmosphere", "nonhydrostatic", "plane",
        "plane_compressible_euler", "PlaneCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.compressible_euler_plane",
    ),

    # -- Tracer transport --
    SolverEntry(
        "atmosphere", "tracer_transport", "cubed_sphere_cdgrid",
        "tracer_transport", "TracerTransportModel",
        "legoesm.atmosphere.dynamics.tracer_transport",
    ),
    SolverEntry(
        "atmosphere", "tracer_transport", "voronoi",
        "tracer_transport_mpas", "TracerTransportMPASModel",
        "legoesm.atmosphere.dynamics.tracer_transport_mpas",
    ),
    SolverEntry(
        "atmosphere", "tracer_transport", "latlon_cgrid",
        "tracer_transport_latlon", "TracerTransportLatLonModel",
        "legoesm.atmosphere.dynamics.tracer_transport_latlon",
    ),
)


# =====================================================================
# Ocean
# =====================================================================

OCEAN_MATRIX: tuple[SolverEntry, ...] = (
    SolverEntry(
        "ocean", "hydrostatic", "cubed_sphere_cdgrid",
        "cdgrid", "OceanModel",
        "legoesm.ocean.dynamics.ocean_model",
    ),
    # NOTE: spectral ocean is unsupported — land boundary handling in
    # spectral space causes Gibbs ringing and unreliable masking.
    # Kept for reference; not included in the ocean test matrix.
    # See https://github.com/gentine/legoESM/issues/99
    SolverEntry(
        "ocean", "hydrostatic", "spectral_gaussian",
        "spectral", "SpectralOceanModel",
        "legoesm.ocean.dynamics.spectral_ocean_pe",
    ),
    SolverEntry(
        "ocean", "hydrostatic", "latlon_cgrid",
        "latlon_cgrid", "LatLonCGridOceanModel",
        "legoesm.ocean.dynamics.ocean_model_latlon_cgrid",
    ),
    SolverEntry(
        "ocean", "hydrostatic", "mpas_voronoi",
        "mpas", "MPASOceanModel",
        "legoesm.ocean.dynamics.ocean_model_mpas",
    ),
    SolverEntry(
        "ocean", "hydrostatic", "sfno",
        "sfno", "SFNOOceanModel",
        "legoesm.ocean.dynamics.sfno_ocean",
    ),
)


SUPPORTED_MATRIX = ATMOSPHERE_MATRIX + OCEAN_MATRIX


# =====================================================================
# Alias → canonical mapping (atmosphere)
# =====================================================================

#: Names that silently resolve to a canonical solver.  Each key is a
#: (location, alias_name) and the value is (canonical_name, reason).
ATMOSPHERE_DEPRECATED_ALIASES: dict[str, tuple[str, str]] = {
    # --- Class names ---
    "ShallowWaterModel": (
        "CDGridShallowWaterModel",
        "Use CDGridShallowWaterModel explicitly.",
    ),
    "PrimitiveEquationModel": (
        "CDGridPrimitiveEquationModel",
        "Use CDGridPrimitiveEquationModel explicitly.",
    ),
    "PrimitiveEquationConfig": (
        "CDGridPrimitiveEquationConfig",
        "Use CDGridPrimitiveEquationConfig explicitly.",
    ),
    "CompressibleEulerModel": (
        "CDGridCompressibleEulerModel",
        "Use CDGridCompressibleEulerModel explicitly.",
    ),
    "FVShallowWaterModel": (
        "CDGridShallowWaterModel",
        "The FV cubed-sphere solver is CDGridShallowWaterModel.",
    ),
    "FVPrimitiveEquationModel": (
        "CDGridPrimitiveEquationModel",
        "The FV cubed-sphere solver is CDGridPrimitiveEquationModel.",
    ),
    "FVCompressibleEulerModel": (
        "CDGridCompressibleEulerModel",
        "The FV cubed-sphere solver is CDGridCompressibleEulerModel.",
    ),
    "FVPrimitiveEquationConfig": (
        "CDGridPrimitiveEquationConfig",
        "The FV cubed-sphere config is CDGridPrimitiveEquationConfig.",
    ),
    "FVCompressibleEulerConfig": (
        "CDGridCompressibleEulerConfig",
        "The FV cubed-sphere config is CDGridCompressibleEulerConfig.",
    ),
    "CGShallowWaterCubedModel": (
        "CDGridShallowWaterModel",
        "The C-grid cubed-sphere solver is CDGridShallowWaterModel.",
    ),
    "CGPrimitiveEquationModel": (
        "CDGridPrimitiveEquationModel",
        "The C-grid cubed-sphere solver is CDGridPrimitiveEquationModel.",
    ),
    "CGCompressibleEulerModel": (
        "CDGridCompressibleEulerModel",
        "The C-grid cubed-sphere solver is CDGridCompressibleEulerModel.",
    ),
    # --- Tendency functions ---
    "shallow_water_tendencies": (
        "cdgrid_shallow_water_tendencies",
        "Use cdgrid_shallow_water_tendencies explicitly.",
    ),
    "hydrostatic_tendencies": (
        "cdgrid_hydrostatic_tendencies",
        "Use cdgrid_hydrostatic_tendencies explicitly.",
    ),
    "compressible_euler_slow_tendencies": (
        "cdgrid_compressible_euler_slow_tendencies",
        "Use cdgrid_compressible_euler_slow_tendencies explicitly.",
    ),
    "fv_shallow_water_tendencies": (
        "cdgrid_shallow_water_tendencies",
        "The FV cubed-sphere tendency function is cdgrid_shallow_water_tendencies.",
    ),
    "fv_hydrostatic_tendencies": (
        "cdgrid_hydrostatic_tendencies",
        "The FV cubed-sphere tendency function is cdgrid_hydrostatic_tendencies.",
    ),
    "fv_compressible_euler_slow_tendencies": (
        "cdgrid_compressible_euler_slow_tendencies",
        "The FV cubed-sphere tendency function is cdgrid_compressible_euler_slow_tendencies.",
    ),
}


#: Discretization names that map to the same implementation.
#: "centered" and "finite_volume" are grid-dependent: they resolve
#: to "cdgrid" on cubed-sphere grids and "latlon_cgrid" on lat-lon
#: grids.  The default (without grid context) is "cdgrid".
#: "cgrid" is a true deprecated alias that always means "cdgrid".
ATMOSPHERE_DEPRECATED_DISCRETIZATIONS: dict[str, str] = {
    "cgrid": "cdgrid",
}

#: Flat solver names that are aliases.
ATMOSPHERE_DEPRECATED_SOLVER_NAMES: dict[str, str] = {
    "shallow_water": "cdgrid_shallow_water",
    "primitive_equations": "cdgrid_primitive_equations",
    "compressible_euler": "cdgrid_compressible_euler",
    "fv_shallow_water": "cdgrid_shallow_water",
    "fv_primitive_equations": "cdgrid_primitive_equations",
    "fv_compressible_euler": "cdgrid_compressible_euler",
    "cgrid_shallow_water": "cdgrid_shallow_water",
    "cgrid_primitive_equations": "cdgrid_primitive_equations",
    "cgrid_compressible_euler": "cdgrid_compressible_euler",
}


# =====================================================================
# Alias → canonical mapping (ocean)
# =====================================================================

OCEAN_DEPRECATED_DISCRETIZATIONS: dict[str, str] = {
    "centered": "cdgrid",
    "finite_volume": "cdgrid",
    "fv": "cdgrid",
}


# =====================================================================
# Public helpers
# =====================================================================

def canonical_solver_names(component: str = "atmosphere") -> list[str]:
    """Return the list of canonical solver names for a component."""
    matrix = ATMOSPHERE_MATRIX if component == "atmosphere" else OCEAN_MATRIX
    return [e.canonical_name for e in matrix]


def print_matrix(component: str | None = None) -> None:
    """Pretty-print the supported implementation matrix."""
    entries = SUPPORTED_MATRIX
    if component:
        entries = tuple(e for e in entries if e.component == component)

    header = f"{'Component':<12} {'Dynamics':<18} {'Grid':<24} {'Canonical Name':<35} {'Class'}"
    print(header)
    print("-" * len(header))
    for e in entries:
        print(f"{e.component:<12} {e.dynamics:<18} {e.grid:<24} {e.canonical_name:<35} {e.class_name}")
