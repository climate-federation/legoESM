"""Machine learning integration for legoESM.

Provides the Spherical Fourier Neural Operator (SFNO) and supporting
infrastructure for learned dynamical cores and hybrid ML-physics models.

Integration status
------------------
The ``ml/`` subtree is staged-not-integrated experimental scaffolding.

Audited 2026-05-13: 36 modules in this subtree are not imported by any
non-``ml/`` source file and many lack direct unit tests
(``ml/training.py``, ``ml/s2s/paths.py``, ``ml/s2s/plotting.py``,
``ml/physics/{plotting,model,train,evaluate,data}.py``,
``ml/s2s/neuralgcm_slab/{metrics,neuralgcm_backend,cli}.py``,
``ml/s2s/sfno_slab/{config,evaluation,postprocess,cli,rollout,preparation,data,regrid}.py``).

The leaf training/eval/plotting modules are reachable only from CLI
entry points and notebooks — they are research scaffolding, not yet
wired into the core dynamical-core ↔ physics pipeline that the rest of
``src/legoesm/`` services.  New ML work should either (a) commit to
adding direct unit tests when a module is touched, or (b) move the
module to a ``_future/`` sibling per the CLAUDE.md staged-not-integrated
rule.  Do not extend the untested-leaf footprint of this subtree.

The publicly re-exported names below (``SFNO``, ``SFNOBlock``,
``SpectralConv``, channel-packing helpers, conservation correctors) ARE
covered by direct tests in ``tests/`` and are safe to import from
production callers.
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
from legoesm.ml.ucast import UCast, UCastConfig
# NOTE: ocean channel packing (OceanChannelSpec / pack_ocean_state /
# unpack_ocean_output) lives in legoesm.ocean.dynamics.channel_packing — NOT
# re-exported here, since legoesm.ml must not import the ocean component
# (component independence; see import-linter contracts).
from legoesm.ml.channel_packing import (
    PE3DChannelSpec,
    WB2_PRESSURE_LEVELS,
    pack_sw_state,
    unpack_sw_output,
    pack_pe_state,
    unpack_pe_output,
)
from legoesm.ml.conservation import (
    correct_dry_air_mass,
    correct_moisture,
    clip_humidity,
    correct_ocean_volume,
    correct_ocean_heat,
    correct_ocean_salt,
)


def __getattr__(name):
    """Lazy imports for S2S/ChaosBench functionality that requires imageio."""
    s2s_names = {
        "CHAOSBENCH_ATMOS_VARS",
        "CHAOSBENCH_DATA_DIR",
        "CHAOSBENCH_PRESSURE_LEVELS",
        "ChaosBenchS2SConfig",
        "S2SSlabCouplingConfig",
        "S2STrainingConfig",
        "TargetGridSpec",
        "atmospheric_param_labels",
        "available_s2s_dates",
        "build_target_grid",
        "build_sfno_from_checkpoint",
        "count_trainable_parameters",
        "create_s2s_training_iterator",
        "compute_daily_metrics",
        "coupled_rollout_to_dataset",
        "denormalize_atmospheric_channels",
        "denormalize_forcing_channels",
        "estimate_sfno_parameter_count",
        "forcing_channel_labels",
        "load_normalization_bundle",
        "load_training_metadata",
        "load_s2s_sample",
        "load_surface_sequence",
        "metadata_path_for_checkpoint",
        "normalize_atmospheric_channels",
        "normalize_forcing_channels",
        "parse_window_specs",
        "regrid_channels_to_gaussian",
        "resolve_s2s_sample_dates",
        "rollout_with_forcing",
        "rollout_to_dataset",
        "sample_index_from_date",
        "save_training_metadata",
        "select_available_fields",
        "summarize_window_metrics",
        "train_s2s_step",
        "train_sfno_s2s",
        "validate_s2s_step",
        "write_metric_rows",
    }
    if name in s2s_names:
        from legoesm.ml.s2s import sfno_slab
        return getattr(sfno_slab, name)
    raise AttributeError(f"module 'legoesm.ml' has no attribute {name!r}")
