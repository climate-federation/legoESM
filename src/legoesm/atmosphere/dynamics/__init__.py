"""Dynamical cores for the legoESM atmosphere.

Architecture
------------
The cubed-sphere implementation is unified around a single FV3-style C-D grid
discretisation (Lin 2004, Putman & Lin 2007):

- **D-grid** winds (cell corners) are prognostic for momentum.
- **C-grid** velocities (cell edges) are diagnosed for mass/scalar transport.
- **Vorticity** from circulation (exact on D-grid, avoids the
  Hollingsworth-Kallberg instability that plagues A-grid solvers).
- The same ``operators_cdgrid`` module is shared by the atmosphere
  (shallow water, hydrostatic PE, non-hydrostatic CE) and the ocean.

Solver selection
----------------
Use two-axis selection (recommended)::

    dynamics:       "shallow_water" | "hydrostatic" | "nonhydrostatic"
    discretization: "cdgrid" | "spectral" | "sfno" | "latlon_fv" | "mpas"

Supported implementation matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
===========================  ==============  =================================
dynamics                     discretization  solver
===========================  ==============  =================================
``shallow_water``            cdgrid          CDGridShallowWaterModel
``shallow_water``            spectral        SpectralShallowWaterModel
``shallow_water``            sfno            SFNOShallowWaterModel
``shallow_water``            latlon_fv       FVShallowWaterLatLonModel
``shallow_water``            latlon_cgrid    CGShallowWaterLatLonModel
``hydrostatic``              cdgrid          CDGridPrimitiveEquationModel
``hydrostatic``              spectral        SpectralPrimitiveEquationModel
``hydrostatic``              sfno            SFNOPrimitiveEquationModel
``hydrostatic``              latlon_fv       FVLatLonPrimitiveEquationModel
``hydrostatic``              mpas            MPASPrimitiveEquationModel
``nonhydrostatic``           cdgrid          CDGridCompressibleEulerModel
``nonhydrostatic``           spectral        SpectralCompressibleEulerModel
``nonhydrostatic``           latlon_fv       FVCompressibleEulerLatLonModel
``nonhydrostatic``           mpas            MPASCompressibleEulerModel
===========================  ==============  =================================

Deprecated aliases
~~~~~~~~~~~~~~~~~~
The following names silently resolve to the C-D grid implementation and
emit ``DeprecationWarning``.  They will be removed in a future release:

- ``ShallowWaterModel``, ``FVShallowWaterModel``, ``CGShallowWaterCubedModel``
  → use ``CDGridShallowWaterModel``
- ``PrimitiveEquationModel``, ``FVPrimitiveEquationModel``,
  ``CGPrimitiveEquationModel`` → use ``CDGridPrimitiveEquationModel``
- ``CompressibleEulerModel``, ``FVCompressibleEulerModel``,
  ``CGCompressibleEulerModel`` → use ``CDGridCompressibleEulerModel``
- discretization names ``"centered"``, ``"finite_volume"``, ``"cgrid"``
  → use ``"cdgrid"``

See :mod:`legoesm.supported_matrix` for the full implementation matrix.
"""

# --- C-D grid cubed-sphere cores (FV3-style) ---
from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterModel,
    CDGridShallowWaterConfig,
    CDGridShallowWaterState,
    cdgrid_shallow_water_tendencies,
)
from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel,
    CDGridPrimitiveEquationConfig,
    cdgrid_hydrostatic_tendencies,
)
from legoesm.atmosphere.dynamics.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    CDGridCompressibleEulerConfig,
    cdgrid_compressible_euler_slow_tendencies,
)

# --- Spectral cores ---
from legoesm.atmosphere.dynamics.spectral_sw import (
    SpectralShallowWaterModel,
    spectral_sw_tendencies,
)
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralPrimitiveEquationModel,
    spectral_pe_tendencies,
)
from legoesm.atmosphere.dynamics.spectral_nh import (
    SpectralCompressibleEulerModel,
    spectral_nh_slow_tendencies,
)

# --- SFNO (data-driven) ---
from legoesm.atmosphere.dynamics.sfno_sw import (
    SFNOShallowWaterModel,
    SFNOShallowWaterConfig,
)
from legoesm.atmosphere.dynamics.sfno_pe import (
    SFNOPrimitiveEquationModel,
    SFNOPrimitiveEquationConfig,
)

# --- Lat-lon cores ---
from legoesm.atmosphere.dynamics.shallow_water_fv_latlon import (
    FVShallowWaterLatLonModel,
    fv_shallow_water_tendencies_latlon,
)
from legoesm.atmosphere.dynamics.shallow_water_cgrid_latlon import (
    CGShallowWaterLatLonModel,
    CGShallowWaterConfig,
    cgrid_shallow_water_tendencies as cgrid_shallow_water_tendencies_latlon,
    a_to_cgrid,
    cgrid_to_a,
)
from legoesm.atmosphere.dynamics.primitive_eq_fv_latlon import (
    FVLatLonPrimitiveEquationModel,
    fv_latlon_hydrostatic_tendencies,
)
from legoesm.atmosphere.dynamics.compressible_euler_fv_latlon import (
    FVCompressibleEulerLatLonModel,
    fv_compressible_euler_latlon_slow_tendencies,
)

# --- MPAS icosahedral ---
from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
    MPASPrimitiveEquationModel,
    MPASPrimitiveEquationConfig,
    mpas_hydrostatic_tendencies,
)
from legoesm.atmosphere.dynamics.compressible_euler_mpas import (
    MPASCompressibleEulerModel,
    MPASCompressibleEulerConfig,
    mpas_compressible_euler_slow_tendencies,
)

# --- Tracer transport ---
from legoesm.atmosphere.dynamics.tracer_transport import (
    TracerTransportModel,
    tracer_tendencies,
)

# --- Shared utilities (acoustic substeps, sponge, Exner) ---
from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerConfig,
    compute_exner_perturbation,
    _sponge_profile,
    acoustic_substeps,
    acoustic_substeps_semi_implicit,
)


# -----------------------------------------------------------------------
# Deprecated aliases — emit DeprecationWarning on access
# -----------------------------------------------------------------------

import warnings as _warnings

_DEPRECATED_ALIASES = {
    "ShallowWaterModel": ("CDGridShallowWaterModel", CDGridShallowWaterModel),
    "PrimitiveEquationModel": ("CDGridPrimitiveEquationModel", CDGridPrimitiveEquationModel),
    "PrimitiveEquationConfig": ("CDGridPrimitiveEquationConfig", CDGridPrimitiveEquationConfig),
    "CompressibleEulerModel": ("CDGridCompressibleEulerModel", CDGridCompressibleEulerModel),
    "FVShallowWaterModel": ("CDGridShallowWaterModel", CDGridShallowWaterModel),
    "FVPrimitiveEquationModel": ("CDGridPrimitiveEquationModel", CDGridPrimitiveEquationModel),
    "FVCompressibleEulerModel": ("CDGridCompressibleEulerModel", CDGridCompressibleEulerModel),
    "FVPrimitiveEquationConfig": ("CDGridPrimitiveEquationConfig", CDGridPrimitiveEquationConfig),
    "FVCompressibleEulerConfig": ("CDGridCompressibleEulerConfig", CDGridCompressibleEulerConfig),
    "CGShallowWaterCubedModel": ("CDGridShallowWaterModel", CDGridShallowWaterModel),
    "CGPrimitiveEquationModel": ("CDGridPrimitiveEquationModel", CDGridPrimitiveEquationModel),
    "CGCompressibleEulerModel": ("CDGridCompressibleEulerModel", CDGridCompressibleEulerModel),
    "shallow_water_tendencies": ("cdgrid_shallow_water_tendencies", cdgrid_shallow_water_tendencies),
    "hydrostatic_tendencies": ("cdgrid_hydrostatic_tendencies", cdgrid_hydrostatic_tendencies),
    "compressible_euler_slow_tendencies": ("cdgrid_compressible_euler_slow_tendencies", cdgrid_compressible_euler_slow_tendencies),
    "fv_shallow_water_tendencies": ("cdgrid_shallow_water_tendencies", cdgrid_shallow_water_tendencies),
    "fv_hydrostatic_tendencies": ("cdgrid_hydrostatic_tendencies", cdgrid_hydrostatic_tendencies),
    "fv_compressible_euler_slow_tendencies": ("cdgrid_compressible_euler_slow_tendencies", cdgrid_compressible_euler_slow_tendencies),
}


def __getattr__(name):
    if name in _DEPRECATED_ALIASES:
        canonical, obj = _DEPRECATED_ALIASES[name]
        _warnings.warn(
            f"legoesm.atmosphere.dynamics.{name} is deprecated; "
            f"use {canonical} instead. "
            f"This alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return obj
    raise AttributeError(f"module 'legoesm.atmosphere.dynamics' has no attribute {name!r}")


# Canonical solver names (genuinely distinct implementations only)
AVAILABLE_SOLVERS = [
    "cdgrid_shallow_water",
    "cdgrid_primitive_equations",
    "cdgrid_compressible_euler",
    "spectral_shallow_water",
    "spectral_primitive_equations",
    "spectral_compressible_euler",
    "sfno_shallow_water",
    "sfno_primitive_equations",
    "fv_shallow_water_latlon",
    "fv_primitive_equations_latlon",
    "fv_compressible_euler_latlon",
    "cgrid_shallow_water_latlon",
    "mpas_primitive_equations",
    "mpas_compressible_euler",
    "tracer_transport",
]

# Deprecated flat names that alias a canonical solver
_DEPRECATED_SOLVER_NAMES = {
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

# Legacy names still accepted by create_model for backward compatibility
_ALL_SOLVER_NAMES = AVAILABLE_SOLVERS + list(_DEPRECATED_SOLVER_NAMES)

# Valid values for the two-axis config keys
DYNAMICS_OPTIONS = ["shallow_water", "hydrostatic", "nonhydrostatic"]
DISCRETIZATION_OPTIONS = ["cdgrid", "spectral", "sfno", "latlon_fv", "latlon_cgrid", "mpas"]

# Deprecated discretization names
_DEPRECATED_DISCRETIZATIONS = {
    "centered": "cdgrid",
    "finite_volume": "cdgrid",
    "cgrid": "cdgrid",
    "fv": "cdgrid",
}

# (dynamics, discretization) -> canonical flat solver name
_AXIS_TO_SOLVER = {
    ("shallow_water", "cdgrid"): "cdgrid_shallow_water",
    ("shallow_water", "spectral"): "spectral_shallow_water",
    ("shallow_water", "sfno"): "sfno_shallow_water",
    ("shallow_water", "latlon_fv"): "fv_shallow_water_latlon",
    ("shallow_water", "latlon_cgrid"): "cgrid_shallow_water_latlon",
    ("hydrostatic", "cdgrid"): "cdgrid_primitive_equations",
    ("hydrostatic", "spectral"): "spectral_primitive_equations",
    ("hydrostatic", "sfno"): "sfno_primitive_equations",
    ("hydrostatic", "latlon_fv"): "fv_primitive_equations_latlon",
    ("hydrostatic", "mpas"): "mpas_primitive_equations",
    ("nonhydrostatic", "cdgrid"): "cdgrid_compressible_euler",
    ("nonhydrostatic", "spectral"): "spectral_compressible_euler",
    ("nonhydrostatic", "latlon_fv"): "fv_compressible_euler_latlon",
    ("nonhydrostatic", "mpas"): "mpas_compressible_euler",
}

# flat solver name -> (dynamics, discretization)
_SOLVER_TO_AXIS = {v: k for k, v in _AXIS_TO_SOLVER.items()}


def resolve_solver_name(
    *,
    dynamics: str | None = None,
    discretization: str | None = None,
    equations: str | None = None,
) -> str:
    """Resolve a flat solver name from either axis-based or legacy config.

    Parameters
    ----------
    dynamics : str, optional
    discretization : str, optional
    equations : str, optional
        Legacy flat solver name.

    Returns
    -------
    str
        A canonical solver name from ``AVAILABLE_SOLVERS``.
    """
    # --- Legacy flat name path ---
    if equations is not None:
        if equations in _DEPRECATED_SOLVER_NAMES:
            canonical = _DEPRECATED_SOLVER_NAMES[equations]
            _warnings.warn(
                f"Solver name {equations!r} is deprecated; "
                f"use {canonical!r} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return canonical
        if equations in AVAILABLE_SOLVERS:
            return equations

    # --- Axis-based resolution ---
    dyn = dynamics or "shallow_water"
    disc = discretization or "cdgrid"

    if dyn not in DYNAMICS_OPTIONS:
        raise ValueError(
            f"Unknown dynamics={dyn!r}. Choose from {DYNAMICS_OPTIONS}"
        )

    # Resolve deprecated discretization names
    if disc in _DEPRECATED_DISCRETIZATIONS:
        canonical_disc = _DEPRECATED_DISCRETIZATIONS[disc]
        _warnings.warn(
            f"Discretization {disc!r} is deprecated; "
            f"use {canonical_disc!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        disc = canonical_disc

    if disc not in DISCRETIZATION_OPTIONS:
        raise ValueError(
            f"Unknown discretization={disc!r}. "
            f"Choose from {DISCRETIZATION_OPTIONS}"
        )

    key = (dyn, disc)
    if key not in _AXIS_TO_SOLVER:
        raise ValueError(
            f"The combination dynamics={dyn!r} + "
            f"discretization={disc!r} is not yet implemented. "
            f"Available: {list(_AXIS_TO_SOLVER.keys())}"
        )

    return _AXIS_TO_SOLVER[key]


def solver_axes(name: str) -> tuple[str, str]:
    """Return the (dynamics, discretization) pair for a flat solver name."""
    return _SOLVER_TO_AXIS[name]


def create_model(name: str = None, legoesm_config=None, **kwargs):
    """Create a dynamical core model by name or from config."""
    # --- Apply runtime hardware config when available ---
    if legoesm_config is not None:
        from legoesm.runtime.config import bootstrap_from_yaml_config
        bootstrap_from_yaml_config(legoesm_config)

    # --- Resolve name from config if not given directly ---
    if name is None:
        if legoesm_config is None:
            raise ValueError(
                "Either 'name' or 'legoesm_config' must be provided."
            )
        name = resolve_solver_name(
            dynamics=legoesm_config.get("atmosphere.dynamics"),
            discretization=legoesm_config.get("atmosphere.discretization"),
            equations=legoesm_config.get("atmosphere.equations"),
        )

    # Map deprecated solver names with warning
    if name in _DEPRECATED_SOLVER_NAMES:
        canonical = _DEPRECATED_SOLVER_NAMES[name]
        _warnings.warn(
            f"Solver name {name!r} is deprecated; "
            f"use {canonical!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        name = canonical

    # --- Instantiate ---
    if name == "cdgrid_shallow_water":
        return CDGridShallowWaterModel(**kwargs)
    elif name == "cdgrid_primitive_equations":
        return CDGridPrimitiveEquationModel(**kwargs)
    elif name == "cdgrid_compressible_euler":
        return CDGridCompressibleEulerModel(**kwargs)
    elif name == "spectral_shallow_water":
        if legoesm_config is not None:
            kwargs.setdefault("legoesm_config", legoesm_config)
        return SpectralShallowWaterModel(**kwargs)
    elif name == "spectral_primitive_equations":
        if legoesm_config is not None:
            kwargs.setdefault("legoesm_config", legoesm_config)
        return SpectralPrimitiveEquationModel(**kwargs)
    elif name == "spectral_compressible_euler":
        if legoesm_config is not None:
            kwargs.setdefault("legoesm_config", legoesm_config)
        return SpectralCompressibleEulerModel(**kwargs)
    elif name == "sfno_shallow_water":
        return SFNOShallowWaterModel(**kwargs)
    elif name == "sfno_primitive_equations":
        return SFNOPrimitiveEquationModel(**kwargs)
    elif name == "fv_shallow_water_latlon":
        return FVShallowWaterLatLonModel(**kwargs)
    elif name == "fv_primitive_equations_latlon":
        return FVLatLonPrimitiveEquationModel(**kwargs)
    elif name == "fv_compressible_euler_latlon":
        return FVCompressibleEulerLatLonModel(**kwargs)
    elif name == "cgrid_shallow_water_latlon":
        return CGShallowWaterLatLonModel(**kwargs)
    elif name == "tracer_transport":
        return TracerTransportModel(**kwargs)
    elif name == "mpas_primitive_equations":
        return MPASPrimitiveEquationModel(**kwargs)
    elif name == "mpas_compressible_euler":
        return MPASCompressibleEulerModel(**kwargs)
    else:
        raise ValueError(
            f"Unknown solver: {name!r}. "
            f"Available: {AVAILABLE_SOLVERS}"
        )
