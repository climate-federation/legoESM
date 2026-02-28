"""Dynamical cores for the legoESM atmosphere."""

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

# Available solver names for the factory
AVAILABLE_SOLVERS = [
    "shallow_water",
    "spectral_shallow_water",
    "primitive_equations",
    "tracer_transport",
]


def create_model(name: str, legoesm_config=None, **kwargs):
    """Create a dynamical core model by name.

    Parameters
    ----------
    name : str
        Solver name. One of: "shallow_water", "spectral_shallow_water",
        "primitive_equations", "tracer_transport".
    legoesm_config : Config, optional
        legoESM global configuration. For spectral models, the
        ``atmosphere.spectral.allow_unsupported`` value is forwarded
        to the backend compatibility guard.
    **kwargs
        Keyword arguments passed to the model constructor.

    Returns
    -------
    Model instance.

    Raises
    ------
    ValueError
        If the name is not recognized.

    Examples
    --------
    >>> model = create_model("shallow_water", grid=grid, config=config)
    >>> model = create_model("tracer_transport", grid=grid,
    ...                      sigma_coord=sigma, wind_fn=wind)
    """
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
    else:
        raise ValueError(
            f"Unknown solver: {name!r}. "
            f"Available: {AVAILABLE_SOLVERS}"
        )
