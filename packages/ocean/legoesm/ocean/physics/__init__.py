"""Ocean physics parameterizations.

Submodules
----------
vertical_mixing : Constant, Richardson, KPP
lateral_mixing  : Harmonic, biharmonic, GM-Redi
surface_forcing : Prescribed, restoring, bulk formulas
bottom_drag     : Linear, quadratic
convection      : Enhanced diffusion, plume
"""

from legoesm.ocean.physics.combined import OceanPhysicsConfig, make_ocean_physics

from legoesm.ocean.physics.vertical_mixing import (
    VerticalMixingConfig,
    ConstantVerticalMixingConfig,
    RichardsonVerticalMixingConfig,
    KPPConfig,
)
from legoesm.ocean.physics.lateral_mixing import (
    LateralMixingConfig,
    HarmonicConfig,
    BiharmonicConfig,
    GMRediConfig,
    VisbeckConfig,
    BackscatterConfig,
)
from legoesm.ocean.physics.surface_forcing import (
    SurfaceForcingConfig,
    PrescribedForcingConfig,
    RestoringConfig,
    BulkFormulaConfig,
)
from legoesm.ocean.physics.bottom_drag import (
    BottomDragConfig,
)
from legoesm.ocean.physics.convection import (
    OceanConvectionConfig,
    EnhancedDiffusionConfig,
    PlumeConfig,
)

__all__ = [
    "OceanPhysicsConfig",
    "make_ocean_physics",
    "VerticalMixingConfig",
    "ConstantVerticalMixingConfig",
    "RichardsonVerticalMixingConfig",
    "KPPConfig",
    "LateralMixingConfig",
    "HarmonicConfig",
    "BiharmonicConfig",
    "GMRediConfig",
    "VisbeckConfig",
    "BackscatterConfig",
    "SurfaceForcingConfig",
    "PrescribedForcingConfig",
    "RestoringConfig",
    "BulkFormulaConfig",
    "BottomDragConfig",
    "OceanConvectionConfig",
    "EnhancedDiffusionConfig",
    "PlumeConfig",
]
