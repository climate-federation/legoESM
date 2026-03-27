"""Dynamical cores for the legoESM atmosphere.

Architecture
------------
The cubed-sphere implementation is unified around a single FV3-style C-D grid
discretisation (Lin 2004, Putman & Lin 2007):

- **D-grid** winds (cell corners) are prognostic for momentum.
- **C-grid** velocities (cell edges) are diagnosed for mass/scalar transport.
- **Vorticity** from circulation (exact on D-grid, avoids the
  Hollingsworth-Kallberg instability that plagues collocated solvers).
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

# -----------------------------------------------------------------------
# Lazy imports — the full solver zoo is loaded on first access to avoid
# pulling every solver module (and its transitive dependencies such as
# JAX/XLA compilation) at package-import time.
# -----------------------------------------------------------------------

import importlib as _importlib
import warnings as _warnings

# Mapping from public name -> (module path, attribute name in that module).
# When a name appears here it is resolved lazily via __getattr__ below.
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # --- C-D grid cubed-sphere cores (FV3-style) ---
    "CDGridShallowWaterModel": ("legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid", "CDGridShallowWaterModel"),
    "CDGridShallowWaterConfig": ("legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid", "CDGridShallowWaterConfig"),
    "CDGridShallowWaterState": ("legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid", "CDGridShallowWaterState"),
    "cdgrid_shallow_water_tendencies": ("legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid", "cdgrid_shallow_water_tendencies"),
    "CDGridPrimitiveEquationModel": ("legoesm.atmosphere.dynamics.primitive_eq_cdgrid", "CDGridPrimitiveEquationModel"),
    "CDGridPrimitiveEquationConfig": ("legoesm.atmosphere.dynamics.primitive_eq_cdgrid", "CDGridPrimitiveEquationConfig"),
    "cdgrid_hydrostatic_tendencies": ("legoesm.atmosphere.dynamics.primitive_eq_cdgrid", "cdgrid_hydrostatic_tendencies"),
    "fv3_hydrostatic_tendencies": ("legoesm.atmosphere.dynamics.primitive_eq_cdgrid", "fv3_hydrostatic_tendencies"),
    "hydrostatic_to_fv3": ("legoesm.atmosphere.dynamics.primitive_eq_cdgrid", "hydrostatic_to_fv3"),
    "fv3_to_hydrostatic": ("legoesm.atmosphere.dynamics.primitive_eq_cdgrid", "fv3_to_hydrostatic"),
    "CDGridCompressibleEulerModel": ("legoesm.atmosphere.dynamics.compressible_euler_cdgrid", "CDGridCompressibleEulerModel"),
    "CDGridCompressibleEulerConfig": ("legoesm.atmosphere.dynamics.compressible_euler_cdgrid", "CDGridCompressibleEulerConfig"),
    "cdgrid_compressible_euler_slow_tendencies": ("legoesm.atmosphere.dynamics.compressible_euler_cdgrid", "cdgrid_compressible_euler_slow_tendencies"),
    # --- Spectral cores ---
    "SpectralShallowWaterModel": ("legoesm.atmosphere.dynamics.spectral_sw", "SpectralShallowWaterModel"),
    "spectral_sw_tendencies": ("legoesm.atmosphere.dynamics.spectral_sw", "spectral_sw_tendencies"),
    "SpectralPrimitiveEquationModel": ("legoesm.atmosphere.dynamics.spectral_pe", "SpectralPrimitiveEquationModel"),
    "spectral_pe_tendencies": ("legoesm.atmosphere.dynamics.spectral_pe", "spectral_pe_tendencies"),
    "SpectralCompressibleEulerModel": ("legoesm.atmosphere.dynamics.spectral_nh", "SpectralCompressibleEulerModel"),
    "spectral_nh_slow_tendencies": ("legoesm.atmosphere.dynamics.spectral_nh", "spectral_nh_slow_tendencies"),
    # --- SFNO (data-driven) ---
    "SFNOShallowWaterModel": ("legoesm.atmosphere.dynamics.sfno_sw", "SFNOShallowWaterModel"),
    "SFNOShallowWaterConfig": ("legoesm.atmosphere.dynamics.sfno_sw", "SFNOShallowWaterConfig"),
    "SFNOPrimitiveEquationModel": ("legoesm.atmosphere.dynamics.sfno_pe", "SFNOPrimitiveEquationModel"),
    "SFNOPrimitiveEquationConfig": ("legoesm.atmosphere.dynamics.sfno_pe", "SFNOPrimitiveEquationConfig"),
    # --- Lat-lon cores ---
    "FVShallowWaterLatLonModel": ("legoesm.atmosphere.dynamics.shallow_water_fv_latlon", "FVShallowWaterLatLonModel"),
    "fv_shallow_water_tendencies_latlon": ("legoesm.atmosphere.dynamics.shallow_water_fv_latlon", "fv_shallow_water_tendencies_latlon"),
    "FVLatLonPrimitiveEquationModel": ("legoesm.atmosphere.dynamics.primitive_eq_fv_latlon", "FVLatLonPrimitiveEquationModel"),
    "fv_latlon_hydrostatic_tendencies": ("legoesm.atmosphere.dynamics.primitive_eq_fv_latlon", "fv_latlon_hydrostatic_tendencies"),
    "FVCompressibleEulerLatLonModel": ("legoesm.atmosphere.dynamics.compressible_euler_fv_latlon", "FVCompressibleEulerLatLonModel"),
    "fv_compressible_euler_latlon_slow_tendencies": ("legoesm.atmosphere.dynamics.compressible_euler_fv_latlon", "fv_compressible_euler_latlon_slow_tendencies"),
    # --- MPAS icosahedral ---
    "MPASPrimitiveEquationModel": ("legoesm.atmosphere.dynamics.primitive_eq_mpas", "MPASPrimitiveEquationModel"),
    "MPASPrimitiveEquationConfig": ("legoesm.atmosphere.dynamics.primitive_eq_mpas", "MPASPrimitiveEquationConfig"),
    "mpas_hydrostatic_tendencies": ("legoesm.atmosphere.dynamics.primitive_eq_mpas", "mpas_hydrostatic_tendencies"),
    "MPASCompressibleEulerModel": ("legoesm.atmosphere.dynamics.compressible_euler_mpas", "MPASCompressibleEulerModel"),
    "MPASCompressibleEulerConfig": ("legoesm.atmosphere.dynamics.compressible_euler_mpas", "MPASCompressibleEulerConfig"),
    "mpas_compressible_euler_slow_tendencies": ("legoesm.atmosphere.dynamics.compressible_euler_mpas", "mpas_compressible_euler_slow_tendencies"),
    # --- Tracer transport ---
    "TracerTransportModel": ("legoesm.atmosphere.dynamics.tracer_transport", "TracerTransportModel"),
    "tracer_tendencies": ("legoesm.atmosphere.dynamics.tracer_transport", "tracer_tendencies"),
    # --- Shared utilities (acoustic substeps, sponge, Exner) ---
    "CompressibleEulerConfig": ("legoesm.atmosphere.dynamics.compressible_euler", "CompressibleEulerConfig"),
    "compute_exner_perturbation": ("legoesm.atmosphere.dynamics.compressible_euler", "compute_exner_perturbation"),
    "_sponge_profile": ("legoesm.atmosphere.dynamics.compressible_euler", "_sponge_profile"),
    "acoustic_substeps": ("legoesm.atmosphere.dynamics.compressible_euler", "acoustic_substeps"),
    "acoustic_substeps_semi_implicit": ("legoesm.atmosphere.dynamics.compressible_euler", "acoustic_substeps_semi_implicit"),
}

# Cache for already-resolved lazy imports (avoids repeated importlib calls).
_LAZY_CACHE: dict[str, object] = {}


# -----------------------------------------------------------------------
# Deprecated aliases — emit DeprecationWarning on access
# -----------------------------------------------------------------------

# NOTE: _DEPRECATED_ALIASES references the lazy names above; the values
# are resolved through __getattr__ as well, so no eager import is needed.

# Deprecated alias name -> canonical lazy-import name it maps to.
_DEPRECATED_ALIASES: dict[str, str] = {
    "ShallowWaterModel": "CDGridShallowWaterModel",
    "PrimitiveEquationModel": "CDGridPrimitiveEquationModel",
    "PrimitiveEquationConfig": "CDGridPrimitiveEquationConfig",
    "CompressibleEulerModel": "CDGridCompressibleEulerModel",
    "FVShallowWaterModel": "CDGridShallowWaterModel",
    "FVPrimitiveEquationModel": "CDGridPrimitiveEquationModel",
    "FVCompressibleEulerModel": "CDGridCompressibleEulerModel",
    "FVPrimitiveEquationConfig": "CDGridPrimitiveEquationConfig",
    "FVCompressibleEulerConfig": "CDGridCompressibleEulerConfig",
    "CGShallowWaterCubedModel": "CDGridShallowWaterModel",
    "CGPrimitiveEquationModel": "CDGridPrimitiveEquationModel",
    "CGCompressibleEulerModel": "CDGridCompressibleEulerModel",
    "shallow_water_tendencies": "cdgrid_shallow_water_tendencies",
    "hydrostatic_tendencies": "cdgrid_hydrostatic_tendencies",
    "compressible_euler_slow_tendencies": "cdgrid_compressible_euler_slow_tendencies",
    "fv_shallow_water_tendencies": "cdgrid_shallow_water_tendencies",
    "fv_hydrostatic_tendencies": "cdgrid_hydrostatic_tendencies",
    "fv_compressible_euler_slow_tendencies": "cdgrid_compressible_euler_slow_tendencies",
}


def _resolve_lazy(name: str) -> object:
    """Resolve a lazy import by name, caching the result."""
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]
    mod_path, attr = _LAZY_IMPORTS[name]
    mod = _importlib.import_module(mod_path)
    obj = getattr(mod, attr)
    _LAZY_CACHE[name] = obj
    return obj


def __getattr__(name):
    # Deprecated aliases — warn and redirect to the canonical lazy name.
    if name in _DEPRECATED_ALIASES:
        canonical = _DEPRECATED_ALIASES[name]
        _warnings.warn(
            f"legoesm.atmosphere.dynamics.{name} is deprecated; "
            f"use {canonical} instead. "
            f"This alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _resolve_lazy(canonical)
    # Lazy imports for canonical names.
    if name in _LAZY_IMPORTS:
        return _resolve_lazy(name)
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
DISCRETIZATION_OPTIONS = ["cdgrid", "spectral", "sfno", "latlon_fv", "mpas"]

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
    # --- Axis-based keys take priority when explicitly provided ---
    # The legacy "equations" key is only used as a fallback when neither
    # dynamics nor discretization is set.  This prevents a stale default
    # "equations" value from overriding an explicit axis-based selection.
    if dynamics is None and discretization is None and equations is not None:
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

    # --- Instantiate (resolves lazy imports on demand) ---
    _SOLVER_TO_CLASS = {
        "cdgrid_shallow_water": "CDGridShallowWaterModel",
        "cdgrid_primitive_equations": "CDGridPrimitiveEquationModel",
        "cdgrid_compressible_euler": "CDGridCompressibleEulerModel",
        "spectral_shallow_water": "SpectralShallowWaterModel",
        "spectral_primitive_equations": "SpectralPrimitiveEquationModel",
        "spectral_compressible_euler": "SpectralCompressibleEulerModel",
        "sfno_shallow_water": "SFNOShallowWaterModel",
        "sfno_primitive_equations": "SFNOPrimitiveEquationModel",
        "fv_shallow_water_latlon": "FVShallowWaterLatLonModel",
        "fv_primitive_equations_latlon": "FVLatLonPrimitiveEquationModel",
        "fv_compressible_euler_latlon": "FVCompressibleEulerLatLonModel",
        "tracer_transport": "TracerTransportModel",
        "mpas_primitive_equations": "MPASPrimitiveEquationModel",
        "mpas_compressible_euler": "MPASCompressibleEulerModel",
    }

    # Spectral solvers accept legoesm_config as a kwarg.
    _SPECTRAL_SOLVERS = {
        "spectral_shallow_water",
        "spectral_primitive_equations",
        "spectral_compressible_euler",
    }

    class_name = _SOLVER_TO_CLASS.get(name)
    if class_name is None:
        raise ValueError(
            f"Unknown solver: {name!r}. "
            f"Available: {AVAILABLE_SOLVERS}"
        )

    if name in _SPECTRAL_SOLVERS and legoesm_config is not None:
        kwargs.setdefault("legoesm_config", legoesm_config)

    cls = _resolve_lazy(class_name)
    return cls(**kwargs)
