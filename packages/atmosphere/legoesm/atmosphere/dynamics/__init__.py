"""Atmosphere dynamical cores. FV3 duo is the only cubed-sphere atmosphere core.

Shared C/D-grid ocean kernels remain available in the substrate.
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
    "iter1009_dual_target_config": ("legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid", "iter1009_dual_target_config"),
    # --- Spectral cores ---
    "SpectralShallowWaterModel": ("legoesm.atmosphere.dynamics.gcm.spectral_sw", "SpectralShallowWaterModel"),
    "spectral_sw_tendencies": ("legoesm.atmosphere.dynamics.gcm.spectral_sw", "spectral_sw_tendencies"),
    "SpectralPrimitiveEquationModel": ("legoesm.atmosphere.dynamics.gcm.spectral_pe", "SpectralPrimitiveEquationModel"),
    "spectral_pe_tendencies": ("legoesm.atmosphere.dynamics.gcm.spectral_pe", "spectral_pe_tendencies"),
    "SpectralCompressibleEulerModel": ("legoesm.atmosphere.dynamics.gcm.spectral_nh", "SpectralCompressibleEulerModel"),
    "spectral_nh_slow_tendencies": ("legoesm.atmosphere.dynamics.gcm.spectral_nh", "spectral_nh_slow_tendencies"),
    # --- SFNO (data-driven) ---
    "SFNOShallowWaterModel": ("legoesm.atmosphere.dynamics.neural.sfno_sw", "SFNOShallowWaterModel"),
    "SFNOShallowWaterConfig": ("legoesm.atmosphere.dynamics.neural.sfno_sw", "SFNOShallowWaterConfig"),
    "SFNOPrimitiveEquationModel": ("legoesm.atmosphere.dynamics.neural.sfno_pe", "SFNOPrimitiveEquationModel"),
    "SFNOPrimitiveEquationConfig": ("legoesm.atmosphere.dynamics.neural.sfno_pe", "SFNOPrimitiveEquationConfig"),
    # --- U-Cast (convolutional U-Net, data-driven) ---
    "UCastPrimitiveEquationModel": ("legoesm.atmosphere.dynamics.neural.ucast_pe", "UCastPrimitiveEquationModel"),
    "UCastPrimitiveEquationConfig": ("legoesm.atmosphere.dynamics.neural.ucast_pe", "UCastPrimitiveEquationConfig"),
    # --- Lat-lon C-grid cores ---
    "CGridLatLonShallowWaterModel": ("legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid", "CGridLatLonShallowWaterModel"),
    "CGridLatLonShallowWaterConfig": ("legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid", "CGridLatLonShallowWaterConfig"),
    "CGridLatLonShallowWaterState": ("legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid", "CGridLatLonShallowWaterState"),
    "cgrid_latlon_sw_tendencies": ("legoesm.atmosphere.dynamics.gcm.shallow_water_generic", "cgrid_latlon_sw_tendencies"),
    "CGridLatLonPrimitiveEquationModel": ("legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid", "CGridLatLonPrimitiveEquationModel"),
    "CGridLatLonPrimitiveEquationConfig": ("legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid", "CGridLatLonPrimitiveEquationConfig"),
    "CGridLatLonHydrostaticState": ("legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid", "CGridLatLonHydrostaticState"),
    "cgrid_latlon_pe_tendencies": ("legoesm.atmosphere.dynamics.gcm.primitive_eq_generic", "cgrid_latlon_pe_tendencies"),
    # --- FV3 six-face duo cube (certified fv_dynamics JAX lane) ---
    "FV3DuoDynamicsModel": ("legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics", "FV3DuoDynamicsModel"),
    "FV3DuoConfig": ("legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics", "FV3DuoConfig"),
    # --- MPAS icosahedral ---
    "MPASPrimitiveEquationModel": ("legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas", "MPASPrimitiveEquationModel"),
    "MPASPrimitiveEquationConfig": ("legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas", "MPASPrimitiveEquationConfig"),
    "mpas_pe_tendencies": ("legoesm.atmosphere.dynamics.gcm.primitive_eq_generic", "mpas_pe_tendencies"),
    "MPASCompressibleEulerModel": ("legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas", "MPASCompressibleEulerModel"),
    "MPASCompressibleEulerConfig": ("legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas", "MPASCompressibleEulerConfig"),
    "mpas_compressible_euler_slow_tendencies": ("legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas", "mpas_compressible_euler_slow_tendencies"),
    # --- Doubly-periodic plane (CRM rollout, PR2c) ---
    "PlaneCompressibleEulerModel": ("legoesm.atmosphere.dynamics.les.compressible_euler_plane", "PlaneCompressibleEulerModel"),
    "plane_compressible_euler_slow_tendencies": ("legoesm.atmosphere.dynamics.les.compressible_euler_plane", "plane_compressible_euler_slow_tendencies"),
    "plane_acoustic_substeps": ("legoesm.atmosphere.dynamics.les.compressible_euler_plane", "plane_acoustic_substeps"),
    # --- Tracer transport ---
    "TracerTransportModel": ("legoesm.atmosphere.dynamics.shared.tracer_transport", "TracerTransportModel"),
    "tracer_tendencies": ("legoesm.atmosphere.dynamics.shared.tracer_transport", "tracer_tendencies"),
    "TracerTransportMPASModel": ("legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas", "TracerTransportMPASModel"),
    "tracer_tendencies_mpas": ("legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas", "tracer_tendencies_mpas"),
    "TracerTransportLatLonModel": ("legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon", "TracerTransportLatLonModel"),
    "tracer_tendencies_latlon": ("legoesm.atmosphere.dynamics.gcm.tracer_transport_latlon", "tracer_tendencies_latlon"),
    "SpectralTracerTransportModel": (
        "legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral",
        "SpectralTracerTransportModel",
    ),
    "tracer_tendencies_spectral": (
        "legoesm.atmosphere.dynamics.gcm.tracer_transport_spectral",
        "tracer_tendencies_spectral",
    ),
    # --- Shared utilities (acoustic substeps, sponge, Exner) ---
    "CompressibleEulerConfig": ("legoesm.atmosphere.dynamics.gcm.compressible_euler", "CompressibleEulerConfig"),
    "compute_exner_perturbation": ("legoesm.atmosphere.dynamics.gcm.compressible_euler", "compute_exner_perturbation"),
    "sponge_profile": ("legoesm.atmosphere.dynamics.gcm.compressible_euler", "sponge_profile"),
    "acoustic_column_kernel": ("legoesm.atmosphere.dynamics.gcm.compressible_euler", "acoustic_column_kernel"),
    "semi_implicit_acoustic_column_kernel": ("legoesm.atmosphere.dynamics.gcm.compressible_euler", "semi_implicit_acoustic_column_kernel"),
    "acoustic_substeps": ("legoesm.atmosphere.dynamics.gcm.compressible_euler", "acoustic_substeps"),
    "acoustic_substeps_semi_implicit": ("legoesm.atmosphere.dynamics.gcm.compressible_euler", "acoustic_substeps_semi_implicit"),
}

# Cache for already-resolved lazy imports (avoids repeated importlib calls).
_LAZY_CACHE: dict[str, object] = {}


# -----------------------------------------------------------------------
# Deprecated aliases — emit DeprecationWarning on access
# -----------------------------------------------------------------------

# NOTE: _DEPRECATED_ALIASES references the lazy names above; the values
# are resolved through __getattr__ as well, so no eager import is needed.

# Deprecated alias name -> canonical lazy-import name it maps to.
_DEPRECATED_ALIASES: dict[str, str] = {}


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
    "spectral_shallow_water",
    "spectral_primitive_equations",
    "spectral_compressible_euler",
    "sfno_shallow_water",
    "sfno_primitive_equations",
    "ucast_primitive_equations",
    "latlon_cgrid_shallow_water",
    "latlon_cgrid_primitive_equations",
    "mpas_primitive_equations",
    "mpas_compressible_euler",
    "fv3_duo_primitive_equations",
    "plane_compressible_euler",
    "tracer_transport",
    "tracer_transport_mpas",
    "tracer_transport_latlon",
    "tracer_transport_spectral",
]

# Deprecated flat names that alias a canonical solver
_DEPRECATED_SOLVER_NAMES = {
}

# Legacy names still accepted by create_model for backward compatibility
_ALL_SOLVER_NAMES = AVAILABLE_SOLVERS + list(_DEPRECATED_SOLVER_NAMES)

# Valid values for the two-axis config keys
DYNAMICS_OPTIONS = ["shallow_water", "hydrostatic", "nonhydrostatic"]
DISCRETIZATION_OPTIONS = [
    "spectral", "sfno", "u_cast", "mpas", "latlon_cgrid", "plane",
    "fv3_duo",
    # Grid-aware lat-lon aliases; rejected without that grid context.
    "finite_volume", "centered",
]

# No deprecated atmosphere discretizations remain supported.
_DEPRECATED_DISCRETIZATIONS = {}

# (dynamics, discretization) -> canonical flat solver name
_AXIS_TO_SOLVER = {
    ("shallow_water", "spectral"): "spectral_shallow_water",
    ("shallow_water", "sfno"): "sfno_shallow_water",
    ("hydrostatic", "spectral"): "spectral_primitive_equations",
    ("hydrostatic", "sfno"): "sfno_primitive_equations",
    ("hydrostatic", "u_cast"): "ucast_primitive_equations",
    ("hydrostatic", "mpas"): "mpas_primitive_equations",
    ("nonhydrostatic", "spectral"): "spectral_compressible_euler",
    ("nonhydrostatic", "mpas"): "mpas_compressible_euler",
    ("nonhydrostatic", "plane"): "plane_compressible_euler",
    # ONE solver serves both fv3_duo arms: the certified fv_dynamics lane
    # is a single program with a static ``hydrostatic`` switch (unlike
    # mpas/cdgrid, whose NH is a separate solver class).  NH is listed
    # FIRST so the reverse map below (last key wins on a value collision)
    # reports the hydrostatic pair as the canonical axes.
    ("nonhydrostatic", "fv3_duo"): "fv3_duo_primitive_equations",
    ("hydrostatic", "fv3_duo"): "fv3_duo_primitive_equations",
    ("shallow_water", "latlon_cgrid"): "latlon_cgrid_shallow_water",
    ("hydrostatic", "latlon_cgrid"): "latlon_cgrid_primitive_equations",
    # "finite_volume" and "centered" default to cdgrid when used without
    # grid context (backward compat).  The driver overrides this using
    # the grid_type-aware _DRIVER_SUPPORTED table.
}

# flat solver name -> (dynamics, discretization)
# Only use canonical discretization names (not the finite_volume/centered
# aliases) so the reverse mapping is unique.
_SOLVER_TO_AXIS = {
    v: k for k, v in _AXIS_TO_SOLVER.items()
    if k[1] not in ("finite_volume", "centered")
}


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
    if discretization in ("cdgrid", "cgrid", "fv", "centered", "finite_volume"):
        raise ValueError(f"Retired atmosphere discretization {discretization!r}; use fv3_duo for cubed_sphere (latlon_cgrid for latlon).")
    if dynamics is None and discretization is None and equations is not None and (equations.startswith(("cdgrid_", "cgrid_", "fv_"))
                                  or equations in ("shallow_water", "primitive_equations", "compressible_euler")):
        raise ValueError(f"Retired cube atmosphere solver {equations!r}; use fv3_duo.")
    # --- Axis-based keys take priority when explicitly provided ---
    # The legacy "equations" key is only used when neither dynamics nor
    # discretization is set.  This prevents a stale default "equations"
    # value from overriding an explicit axis-based selection.
    if dynamics is None and discretization is None:
        if equations is None:
            raise ValueError(
                "resolve_solver_name: no solver selected; pass dynamics= and "
                "discretization=, or a flat equations= solver name."
            )
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
        raise ValueError(
            f"Unknown equations={equations!r}. Choose from {AVAILABLE_SOLVERS} "
            f"or deprecated names {sorted(_DEPRECATED_SOLVER_NAMES)}."
        )

    # --- Axis-based resolution: both axes are required ---
    if dynamics is None or discretization is None:
        raise ValueError(
            "resolve_solver_name needs both dynamics= and discretization=; "
            f"got dynamics={dynamics!r}, discretization={discretization!r}."
        )
    dyn = dynamics
    disc = discretization

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


# Canonical flat solver name -> model class name (lazy-imported via _resolve_lazy).
# Module-level so get_solver_class() can resolve a built-in dycore CLASS by name
# (the registry uses this); create_model() instantiates using the same table.
_SOLVER_TO_CLASS = {
    "spectral_shallow_water": "SpectralShallowWaterModel",
    "spectral_primitive_equations": "SpectralPrimitiveEquationModel",
    "spectral_compressible_euler": "SpectralCompressibleEulerModel",
    "sfno_shallow_water": "SFNOShallowWaterModel",
    "sfno_primitive_equations": "SFNOPrimitiveEquationModel",
    "ucast_primitive_equations": "UCastPrimitiveEquationModel",
    "tracer_transport": "TracerTransportModel",
    "tracer_transport_mpas": "TracerTransportMPASModel",
    "tracer_transport_latlon": "TracerTransportLatLonModel",
    "tracer_transport_spectral": "SpectralTracerTransportModel",
    "mpas_primitive_equations": "MPASPrimitiveEquationModel",
    "mpas_compressible_euler": "MPASCompressibleEulerModel",
    "fv3_duo_primitive_equations": "FV3DuoDynamicsModel",
    "plane_compressible_euler": "PlaneCompressibleEulerModel",
    "latlon_cgrid_shallow_water": "CGridLatLonShallowWaterModel",
    "latlon_cgrid_primitive_equations": "CGridLatLonPrimitiveEquationModel",
}

# Spectral solvers accept legoesm_config as a kwarg.
_SPECTRAL_SOLVERS = {
    "spectral_shallow_water",
    "spectral_primitive_equations",
    "spectral_compressible_euler",
}


def get_solver_class(name: str) -> type:
    """Return the dycore CLASS registered under the canonical solver *name*.

    Public companion to :func:`create_model` (which *instantiates*): lets the
    component/dycore registry resolve a built-in dycore class by name without
    reaching for a private symbol.  Raises ``ValueError`` on an unknown name.
    """
    if name.startswith(("cdgrid_", "cgrid_", "fv_")) or name in ("shallow_water", "primitive_equations", "compressible_euler"):
        raise ValueError(f"Retired cube atmosphere solver {name!r}; use fv3_duo.")
    class_name = _SOLVER_TO_CLASS.get(name)
    if class_name is None:
        raise ValueError(
            f"Unknown solver: {name!r}. Available: {AVAILABLE_SOLVERS}"
        )
    return _resolve_lazy(class_name)


def create_model(name: str = None, legoesm_config=None, **kwargs):
    """Create a dynamical core model by name or from config."""
    # Track whether the resolved name came from an ambiguous discretization
    # (finite_volume/centered) that should be rerouted grid-aware, vs an
    # explicit request (name= directly, equations=, or discretization=
    # "cdgrid"/"latlon_cgrid"/etc.) that must be honored as-is.
    _reroute_latlon = False

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
        _disc = legoesm_config.get("atmosphere.discretization")
        _eqs = legoesm_config.get("atmosphere.equations")
        _dyn = legoesm_config.get("atmosphere.dynamics")
        if _disc in ("centered", "finite_volume"):
            from legoesm.grids.latlon import LatLonGrid
            if isinstance(kwargs.get("grid"), LatLonGrid):
                _disc = "latlon_cgrid"
                _grid = kwargs["grid"]
                # Apply basic pole-cell CFL safety.  The driver factory
                # does a more thorough job (diffusion coefficients, etc.),
                # but create_model() callers get at least the dt guard.
                from legoesm.core.cfl import pole_cell_dx, cfl_max_dt
                dx_pole = pole_cell_dx(_grid)
                _dt = kwargs.get("dt", 600.0)
                _dt_clamped = min(_dt, cfl_max_dt(dx_pole, 300.0,
                                                   cfl_number=0.8, ndim=1))
                if _dt_clamped < _dt:
                    _warnings.warn(
                        f"create_model: dt={_dt:.0f}s exceeds pole-cell "
                        f"CFL limit ({_dt_clamped:.0f}s); clamping.",
                        stacklevel=2,
                    )
                    kwargs["dt"] = _dt_clamped
        name = resolve_solver_name(
            dynamics=_dyn, discretization=_disc, equations=_eqs,
        )

    # --- Instantiate (resolves lazy imports on demand) ---
    if kwargs.get("grid") is not None and not name.startswith("tracer_transport"):
        from legoesm.grids.cubed_sphere import CubedSphereGrid
        if isinstance(kwargs["grid"], CubedSphereGrid) and name != "fv3_duo_primitive_equations":
            raise ValueError("Cubed-sphere atmosphere supports only fv3_duo.")
    if name.startswith(("cdgrid_", "cgrid_", "fv_")) or name in ("shallow_water", "primitive_equations", "compressible_euler"):
        raise ValueError(f"Retired cube atmosphere solver {name!r}; use fv3_duo.")
    class_name = _SOLVER_TO_CLASS.get(name)
    if class_name is None:
        raise ValueError(
            f"Unknown solver: {name!r}. "
            f"Available: {AVAILABLE_SOLVERS}"
        )

    if name in _SPECTRAL_SOLVERS and legoesm_config is not None:
        kwargs.setdefault("legoesm_config", legoesm_config)

    cls = _resolve_lazy(class_name)
    model = cls(**kwargs)

    # Expose clamped timestep so callers know what dt to use for step().
    if not hasattr(model, 'effective_dt'):
        _clamped = kwargs.get("dt")
        if _clamped is not None:
            model.effective_dt = _clamped

    return model
