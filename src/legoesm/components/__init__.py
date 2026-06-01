"""Component contracts for the legoESM 'Earth System Legos' architecture.

This package holds the *seams* the Stage-B architectural refactor builds on —
the Protocols and abstract bases that let dycores, physics schemes, and surface
components be discovered by the registry and composed by the coupler while each
remains an interchangeable, individually-differentiable brick.

See :mod:`legoesm.components.protocol`.
"""

from legoesm.components.complexity import (  # noqa: F401
    LandComplexity,
    OceanComplexity,
    ocean_simple_mode,
)
from legoesm.components.interface import (  # noqa: F401
    CouplingBrick,
    FormBrick,
    Interface,
    RegridBrick,
    TransferBrick,
    validate_interface,
)
from legoesm.components.protocol import (  # noqa: F401
    AbstractComponent,
    DycoreProtocol,
    PhysicsModuleProtocol,
    SurfaceComponentProtocol,
    validate_dycore,
)

__all__ = [
    "AbstractComponent",
    "DycoreProtocol",
    "PhysicsModuleProtocol",
    "SurfaceComponentProtocol",
    "validate_dycore",
    "Interface",
    "FormBrick",
    "TransferBrick",
    "RegridBrick",
    "CouplingBrick",
    "validate_interface",
    "OceanComplexity",
    "LandComplexity",
    "ocean_simple_mode",
]
