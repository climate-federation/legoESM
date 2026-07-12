"""Lateral mixing parameterizations for the ocean."""

from legoesm.ocean.physics.lateral_mixing.config import (
    LateralMixingConfig,
    HarmonicConfig,
    BiharmonicConfig,
    GMRediConfig,
    VisbeckConfig,
)
from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput
from legoesm.ocean.physics.lateral_mixing.integration import (
    make_lateral_mixing_physics,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi import (
    compute_visbeck_kappa_gm,
)
from legoesm.ocean.physics.lateral_mixing.backscatter import (
    BackscatterConfig,
    backscatter_tendency_cgrid,
    backscatter_tendency_mpas,
    backscatter_power_density_cgrid,
    update_eddy_energy,
    diagnostic_eddy_energy,
    cfl_cap_eddy_energy,
    diagnostic_backscatter_cgrid,
)

__all__ = [
    "LateralMixingConfig",
    "HarmonicConfig",
    "BiharmonicConfig",
    "GMRediConfig",
    "VisbeckConfig",
    "LateralMixingOutput",
    "make_lateral_mixing_physics",
    "compute_visbeck_kappa_gm",
    "BackscatterConfig",
    "backscatter_tendency_cgrid",
    "backscatter_tendency_mpas",
    "backscatter_power_density_cgrid",
    "update_eddy_energy",
    "diagnostic_eddy_energy",
    "cfl_cap_eddy_energy",
    "diagnostic_backscatter_cgrid",
]
