"""Dynamical cores for the legoESM atmosphere.

Solver selection
----------------
There are two ways to choose a dynamical core:

1. **Two-axis selection** (recommended)::

       dynamics:       "shallow_water" | "hydrostatic" | "nonhydrostatic"
       discretization: "cdgrid" | "spectral" | ...

   These combine as:

   ========================  ==============  ===============================
   dynamics                  discretization  solver
   ========================  ==============  ===============================
   ``shallow_water``         cdgrid          CDGridShallowWaterModel
   ``shallow_water``         spectral        SpectralShallowWaterModel
   ``hydrostatic``           cdgrid          CDGridPrimitiveEquationModel
   ``hydrostatic``           spectral        SpectralPrimitiveEquationModel
   ``nonhydrostatic``        cdgrid          CDGridCompressibleEulerModel
   ``nonhydrostatic``        spectral        SpectralCompressibleEulerModel
   ========================  ==============  ===============================

2. **Legacy flat name** (still supported)::

       "cdgrid_shallow_water", "spectral_shallow_water",
       "cdgrid_primitive_equations", "spectral_primitive_equations",
       "cdgrid_compressible_euler", "spectral_compressible_euler",
       "tracer_transport"

Use :func:`resolve_solver_name` to convert between the two, and
:func:`create_model` to instantiate.
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


# Backward compatibility aliases
ShallowWaterModel = CDGridShallowWaterModel
PrimitiveEquationModel = CDGridPrimitiveEquationModel
CompressibleEulerModel = CDGridCompressibleEulerModel
FVShallowWaterModel = CDGridShallowWaterModel
FVPrimitiveEquationModel = CDGridPrimitiveEquationModel
FVCompressibleEulerModel = CDGridCompressibleEulerModel
CGShallowWaterCubedModel = CDGridShallowWaterModel
CGPrimitiveEquationModel = CDGridPrimitiveEquationModel
CGCompressibleEulerModel = CDGridCompressibleEulerModel
FVPrimitiveEquationConfig = CDGridPrimitiveEquationConfig
FVCompressibleEulerConfig = CDGridCompressibleEulerConfig
shallow_water_tendencies = cdgrid_shallow_water_tendencies
hydrostatic_tendencies = cdgrid_hydrostatic_tendencies
compressible_euler_slow_tendencies = cdgrid_compressible_euler_slow_tendencies
fv_shallow_water_tendencies = cdgrid_shallow_water_tendencies
fv_hydrostatic_tendencies = cdgrid_hydrostatic_tendencies
fv_compressible_euler_slow_tendencies = cdgrid_compressible_euler_slow_tendencies


# Available solver names for the factory (flat namespace)
AVAILABLE_SOLVERS = [
    "shallow_water",
    "cdgrid_shallow_water",
    "spectral_shallow_water",
    "primitive_equations",
    "cdgrid_primitive_equations",
    "spectral_primitive_equations",
    "tracer_transport",
    "compressible_euler",
    "cdgrid_compressible_euler",
    "spectral_compressible_euler",
    "sfno_shallow_water",
    "sfno_primitive_equations",
    "fv_shallow_water",
    "fv_primitive_equations",
    "fv_compressible_euler",
    "fv_shallow_water_latlon",
    "fv_primitive_equations_latlon",
    "fv_compressible_euler_latlon",
    "cgrid_shallow_water_latlon",
    "cgrid_shallow_water",
    "cgrid_primitive_equations",
    "cgrid_compressible_euler",
    "mpas_primitive_equations",
    "mpas_compressible_euler",
]

# Valid values for the two-axis config keys
DYNAMICS_OPTIONS = ["shallow_water", "hydrostatic", "nonhydrostatic"]
DISCRETIZATION_OPTIONS = ["centered", "spectral", "sfno", "finite_volume", "cgrid", "cdgrid", "mpas"]

# (dynamics, discretization) -> flat solver name
_AXIS_TO_SOLVER = {
    ("shallow_water", "centered"): "cdgrid_shallow_water",
    ("shallow_water", "spectral"): "spectral_shallow_water",
    ("shallow_water", "sfno"): "sfno_shallow_water",
    ("hydrostatic", "centered"): "cdgrid_primitive_equations",
    ("hydrostatic", "spectral"): "spectral_primitive_equations",
    ("hydrostatic", "sfno"): "sfno_primitive_equations",
    ("nonhydrostatic", "centered"): "cdgrid_compressible_euler",
    ("nonhydrostatic", "spectral"): "spectral_compressible_euler",
    ("shallow_water", "finite_volume"): "cdgrid_shallow_water",
    ("hydrostatic", "finite_volume"): "cdgrid_primitive_equations",
    ("nonhydrostatic", "finite_volume"): "cdgrid_compressible_euler",
    ("shallow_water", "cgrid"): "cdgrid_shallow_water",
    ("hydrostatic", "cgrid"): "cdgrid_primitive_equations",
    ("nonhydrostatic", "cgrid"): "cdgrid_compressible_euler",
    ("shallow_water", "cdgrid"): "cdgrid_shallow_water",
    ("hydrostatic", "cdgrid"): "cdgrid_primitive_equations",
    ("nonhydrostatic", "cdgrid"): "cdgrid_compressible_euler",
    ("hydrostatic", "mpas"): "mpas_primitive_equations",
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
    """
    # --- Legacy path ---
    if equations is not None and equations in AVAILABLE_SOLVERS:
        defaults_match = (
            (dynamics is None or dynamics == "shallow_water")
            and (discretization is None or discretization == "centered")
        )
        if defaults_match:
            return equations

    # --- Axis-based resolution ---
    dyn = dynamics or "shallow_water"
    disc = discretization or "centered"

    if dyn not in DYNAMICS_OPTIONS:
        raise ValueError(
            f"Unknown dynamics={dyn!r}. Choose from {DYNAMICS_OPTIONS}"
        )
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
        from legoesm.core.hardware import apply_hardware_config
        apply_hardware_config(legoesm_config)

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

    # Map legacy names to cdgrid
    _legacy_map = {
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
    name = _legacy_map.get(name, name)

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
