"""Machine learning integration for legoESM.

Provides the Spherical Fourier Neural Operator (SFNO) and supporting
infrastructure for learned dynamical cores and hybrid ML-physics models.
"""

from legoesm.ml.normalization import (
    NormalizationStats,
    normalize,
    denormalize,
    compute_normalization_stats,
)
from legoesm.ml.spectral_conv import SpectralConv
from legoesm.ml.sfno_block import SFNOBlock
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.channel_packing import (
    SWChannelSpec,
    PE3DChannelSpec,
    OceanChannelSpec,
    WB2_PRESSURE_LEVELS,
    pack_sw_state,
    unpack_sw_output,
    pack_pe_state,
    unpack_pe_output,
    pack_ocean_state,
    unpack_ocean_output,
)
from legoesm.ml.conservation import (
    correct_dry_air_mass,
    correct_moisture,
    clip_humidity,
    correct_ocean_volume,
    correct_ocean_heat,
    correct_ocean_salt,
)
