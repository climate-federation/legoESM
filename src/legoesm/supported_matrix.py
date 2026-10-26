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
    """One genuinely distinct solver implementation.

    ``(canonical_name, dynamics)`` pairs are UNIQUE across the matrix
    (enforced by tests/unit/test_deprecation_warnings.py); a solver
    serving several dynamics axes gets ONE ROW PER AXIS so consumers
    filtering on ``dynamics`` see real coverage — free-text ``note`` is
    NOT machine-readable (codex 2026-08-18) and never carries support
    claims.
    """
    component: str        # "atmosphere" | "ocean"
    dynamics: str         # "shallow_water" | "hydrostatic" | "nonhydrostatic"
    grid: str             # "cubed_sphere_cdgrid" | "spectral" | "latlon_fv" | ...
    canonical_name: str   # Canonical flat name for create_model / config
    class_name: str       # Canonical Python class name
    module: str           # Defining module path
    note: str = ""        # Extra machine-readable coverage (e.g. more dynamics axes)


# =====================================================================
# Atmosphere
# =====================================================================

ATMOSPHERE_MATRIX: tuple[SolverEntry, ...] = (
    # -- Cubed-sphere C-D grid (FV3-style) --
    SolverEntry(
        "atmosphere", "shallow_water", "cubed_sphere_cdgrid",
        "cdgrid_shallow_water", "CDGridShallowWaterModel",
        "legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid",
    ),
    SolverEntry(
        "atmosphere", "hydrostatic", "cubed_sphere_cdgrid",
        "cdgrid_primitive_equations", "CDGridPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid",
    ),
    SolverEntry(
        "atmosphere", "nonhydrostatic", "cubed_sphere_cdgrid",
        "cdgrid_compressible_euler", "CDGridCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid",
    ),

    # -- Spectral (Gaussian grid) --
    SolverEntry(
        "atmosphere", "shallow_water", "spectral_gaussian",
        "spectral_shallow_water", "SpectralShallowWaterModel",
        "legoesm.atmosphere.dynamics.gcm.spectral_sw",
    ),
    SolverEntry(
        "atmosphere", "hydrostatic", "spectral_gaussian",
        "spectral_primitive_equations", "SpectralPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.gcm.spectral_pe",
    ),
    SolverEntry(
        "atmosphere", "nonhydrostatic", "spectral_gaussian",
        "spectral_compressible_euler", "SpectralCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.gcm.spectral_nh",
    ),

    # -- SFNO (data-driven) --
    SolverEntry(
        "atmosphere", "shallow_water", "sfno",
        "sfno_shallow_water", "SFNOShallowWaterModel",
        "legoesm.atmosphere.dynamics.neural.sfno_sw",
    ),
    SolverEntry(
        "atmosphere", "hydrostatic", "sfno",
        "sfno_primitive_equations", "SFNOPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.neural.sfno_pe",
    ),

    # -- MPAS / SCVT Voronoi mesh + TRiSK discretization --
    SolverEntry(
        "atmosphere", "hydrostatic", "mpas",
        "mpas_primitive_equations", "MPASPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas",
    ),
    SolverEntry(
        "atmosphere", "nonhydrostatic", "mpas",
        "mpas_compressible_euler", "MPASCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas",
    ),

    # -- FV3 six-face duo cube (certified fv_dynamics JAX lane) --
    # ONE solver serves the hydrostatic AND nonhydrostatic arms (a static
    # ``hydrostatic`` switch in the certified core), so a single entry;
    # listed under its canonical "hydrostatic" axes.
    SolverEntry(
        "atmosphere", "hydrostatic", "fv3_duo_cube",
        "fv3_duo_primitive_equations", "FV3DuoDynamicsModel",
        "legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics",
    ),
    SolverEntry(
        "atmosphere", "nonhydrostatic", "fv3_duo_cube",
        "fv3_duo_primitive_equations", "FV3DuoDynamicsModel",
        "legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics",
        note="same class as the hydrostatic row: ONE certified core with "
             "a static hydrostatic switch.",
    ),

    # -- U-cast (unstructured-cast hydrostatic primitive equations) --
    SolverEntry(
        "atmosphere", "hydrostatic", "u_cast",
        "ucast_primitive_equations", "UCastPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.neural.ucast_pe",
    ),

    # -- Lat-lon C-grid (FV) --
    SolverEntry(
        "atmosphere", "shallow_water", "latlon_cgrid",
        "latlon_cgrid_shallow_water", "CGridLatLonShallowWaterModel",
        "legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid",
    ),
    SolverEntry(
        "atmosphere", "hydrostatic", "latlon_cgrid",
        "latlon_cgrid_primitive_equations", "CGridLatLonPrimitiveEquationModel",
        "legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid",
    ),

    # -- Doubly-periodic plane (CRM rollout, PR2c) --
    SolverEntry(
        "atmosphere", "nonhydrostatic", "plane",
        "plane_compressible_euler", "PlaneCompressibleEulerModel",
        "legoesm.atmosphere.dynamics.les.compressible_euler_plane",
    ),

    # -- Tracer transport --
    SolverEntry(
        "atmosphere", "tracer_transport", "cubed_sphere_cdgrid",
        "tracer_transport", "TracerTransportModel",
        "legoesm.atmosphere.dynamics.shared.tracer_transport",
    ),
    SolverEntry(
        "atmosphere", "tracer_transport", "mpas",
        "tracer_transport_mpas", "TracerTransportMPASModel",
        "legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas",
    ),
    SolverEntry(
        "atmosphere", "tracer_transport", "latlon_cgrid",
        "tracer_transport_latlon", "TracerTransportLatLonModel",
        "legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon",
    ),
    SolverEntry(
        "atmosphere", "tracer_transport", "spectral_gaussian",
        "tracer_transport_spectral", "SpectralTracerTransportModel",
        "legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral",
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
    # See https://github.com/climate-federation/legoESM/issues/99
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
        "ocean", "hydrostatic", "mpas",
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
