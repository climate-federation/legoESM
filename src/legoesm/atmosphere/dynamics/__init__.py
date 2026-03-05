"""Dynamical cores for the legoESM atmosphere.

Solver selection
----------------
There are two ways to choose a dynamical core:

1. **Two-axis selection** (recommended)::

       dynamics:       "shallow_water" | "hydrostatic" | "nonhydrostatic"
       discretization: "centered" | "spectral"

   These combine as:

   ========================  ==============  ===============================
   dynamics                  discretization  solver
   ========================  ==============  ===============================
   ``shallow_water``         centered        ShallowWaterModel
   ``shallow_water``         spectral        SpectralShallowWaterModel
   ``hydrostatic``           centered        PrimitiveEquationModel
   ``hydrostatic``           spectral        SpectralPrimitiveEquationModel
   ``nonhydrostatic``        centered        CompressibleEulerModel
   ``nonhydrostatic``        spectral        SpectralCompressibleEulerModel
   ========================  ==============  ===============================

2. **Legacy flat name** (still supported)::

       "shallow_water", "spectral_shallow_water",
       "primitive_equations", "spectral_primitive_equations",
       "compressible_euler", "spectral_compressible_euler",
       "tracer_transport"

Use :func:`resolve_solver_name` to convert between the two, and
:func:`create_model` to instantiate.
"""

from legoesm.atmosphere.dynamics.shallow_water import (
    ShallowWaterModel,
    shallow_water_tendencies,
)
from legoesm.atmosphere.dynamics.spectral_sw import (
    SpectralShallowWaterModel,
    spectral_sw_tendencies,
)
from legoesm.atmosphere.dynamics.primitive_eq import (
    PrimitiveEquationModel,
    hydrostatic_tendencies,
)
from legoesm.atmosphere.dynamics.tracer_transport import (
    TracerTransportModel,
    tracer_tendencies,
)
from legoesm.atmosphere.dynamics.compressible_euler import (
    CompressibleEulerModel,
    compressible_euler_slow_tendencies,
)
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralPrimitiveEquationModel,
    spectral_pe_tendencies,
)
from legoesm.atmosphere.dynamics.spectral_nh import (
    SpectralCompressibleEulerModel,
    spectral_nh_slow_tendencies,
)
from legoesm.atmosphere.dynamics.sfno_sw import (
    SFNOShallowWaterModel,
    SFNOShallowWaterConfig,
)
from legoesm.atmosphere.dynamics.sfno_pe import (
    SFNOPrimitiveEquationModel,
    SFNOPrimitiveEquationConfig,
)
# Available solver names for the factory (flat namespace)
AVAILABLE_SOLVERS = [
    "shallow_water",
    "spectral_shallow_water",
    "primitive_equations",
    "spectral_primitive_equations",
    "tracer_transport",
    "compressible_euler",
    "spectral_compressible_euler",
    "sfno_shallow_water",
    "sfno_primitive_equations",
]

# Valid values for the two-axis config keys
DYNAMICS_OPTIONS = ["shallow_water", "hydrostatic", "nonhydrostatic"]
DISCRETIZATION_OPTIONS = ["centered", "spectral", "sfno"]

# (dynamics, discretization) -> flat solver name
_AXIS_TO_SOLVER = {
    ("shallow_water", "centered"): "shallow_water",
    ("shallow_water", "spectral"): "spectral_shallow_water",
    ("shallow_water", "sfno"): "sfno_shallow_water",
    ("hydrostatic", "centered"): "primitive_equations",
    ("hydrostatic", "spectral"): "spectral_primitive_equations",
    ("hydrostatic", "sfno"): "sfno_primitive_equations",
    ("nonhydrostatic", "centered"): "compressible_euler",
    ("nonhydrostatic", "spectral"): "spectral_compressible_euler",
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
        Vertical physics: ``"shallow_water"``, ``"hydrostatic"``, or
        ``"nonhydrostatic"``.
    discretization : str, optional
        Horizontal method: ``"finite_volume"`` or ``"spectral"``.
    equations : str, optional
        Legacy flat solver name (e.g. ``"primitive_equations"``).
        If this matches a known solver and *dynamics* / *discretization*
        are at their defaults, it takes precedence for backward
        compatibility.

    Returns
    -------
    str
        Flat solver name suitable for :func:`create_model`.

    Raises
    ------
    ValueError
        If the combination is invalid or not yet implemented.

    Examples
    --------
    >>> resolve_solver_name(dynamics="hydrostatic", discretization="finite_volume")
    'primitive_equations'
    >>> resolve_solver_name(dynamics="nonhydrostatic")
    'compressible_euler'
    >>> resolve_solver_name(equations="spectral_shallow_water")
    'spectral_shallow_water'
    """
    # --- Legacy path: explicit equations name ---
    if equations is not None and equations in AVAILABLE_SOLVERS:
        # If the user also set dynamics/discretization to something
        # other than the defaults, prefer the axis-based resolution
        # (the legacy key is kept for backward compatibility only).
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
    """Return the (dynamics, discretization) pair for a flat solver name.

    Parameters
    ----------
    name : str
        Flat solver name (e.g. ``"primitive_equations"``).

    Returns
    -------
    (dynamics, discretization) : tuple[str, str]

    Raises
    ------
    KeyError
        If the name has no axis mapping (e.g. ``"tracer_transport"``).
    """
    return _SOLVER_TO_AXIS[name]


def create_model(name: str = None, legoesm_config=None, **kwargs):
    """Create a dynamical core model by name or from config.

    Parameters
    ----------
    name : str, optional
        Flat solver name (e.g. ``"shallow_water"``).  If *None*,
        the solver is resolved from *legoesm_config* using the
        ``atmosphere.dynamics`` and ``atmosphere.discretization``
        keys (falling back to ``atmosphere.equations``).
    legoesm_config : Config, optional
        legoESM global configuration.  Used to resolve the solver
        name when *name* is None, and forwarded to spectral models
        for the ``atmosphere.spectral.allow_unsupported`` guard.
    **kwargs
        Keyword arguments passed to the model constructor.

    Returns
    -------
    Model instance.

    Raises
    ------
    ValueError
        If the solver name is not recognized or the axis combination
        is invalid.

    Examples
    --------
    Create by flat name (legacy)::

        model = create_model("shallow_water", grid=grid)

    Create by config (recommended)::

        cfg = Config.from_dict({
            "atmosphere": {
                "dynamics": "nonhydrostatic",
                "discretization": "finite_volume",
            }
        })
        model = create_model(legoesm_config=cfg, grid=grid,
                             height_coord=hc, terrain_metric=tm)

    Create by axis keywords passed through name=None::

        model = create_model(
            legoesm_config=Config.from_dict({
                "atmosphere": {"dynamics": "hydrostatic"}
            }),
            grid=grid, sigma_coord=sc,
        )
    """
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

    # --- Instantiate ---
    if name == "shallow_water":
        return ShallowWaterModel(**kwargs)
    elif name == "spectral_shallow_water":
        if legoesm_config is not None:
            kwargs.setdefault("legoesm_config", legoesm_config)
        return SpectralShallowWaterModel(**kwargs)
    elif name == "primitive_equations":
        return PrimitiveEquationModel(**kwargs)
    elif name == "tracer_transport":
        return TracerTransportModel(**kwargs)
    elif name == "compressible_euler":
        return CompressibleEulerModel(**kwargs)
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
    else:
        raise ValueError(
            f"Unknown solver: {name!r}. "
            f"Available: {AVAILABLE_SOLVERS}"
        )
