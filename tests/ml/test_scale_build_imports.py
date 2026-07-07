"""Import-resolution check for scale_build's installed-API bindings.

Does NOT run the pipeline (needs GPU + ERA5) — it confirms every package symbol
scale_build binds to actually exists in this checkout (catches the run_aimip_latlon
API divergences the scout flagged). Run on a compute node (imports jax/legoesm).
"""
import importlib

import pytest


def test_scale_build_module_imports():
    m = importlib.import_module("legoesm.training.scale_build")
    for fn in ("build_mode_components", "load_era5_samples", "evaluate_wb2",
               "build_latlon_config", "make_loss_config"):
        assert hasattr(m, fn), fn


def test_physics_mode_symbols_resolve():
    from legoesm.driver.model_driver import ModelDriver          # noqa: F401
    from legoesm.driver.physics_pipeline import build_physics_pipeline  # noqa: F401
    from legoesm.training.training_driver import build_training_segment  # noqa: F401
    from legoesm.training.trainable_params import TrainablePhysicsParams  # noqa: F401
    from legoesm.core.cfl import cfl_max_dt                      # noqa: F401
    from legoesm.training.dycore_rollout import single_day_rollout  # noqa: F401
    from legoesm.training.losses import combined_loss, LossConfig  # noqa: F401


def test_neural_gcm_mode_symbols_resolve():
    from legoesm.atmosphere.physics.neural_physics import (      # noqa: F401
        NeuralPhysics, make_neural_step_unified,
    )
    from legoesm.core.grid_adapters import make_adapter          # noqa: F401


def test_era5_symbols_resolve():
    from legoesm.training.era5_to_state import (                 # noqa: F401
        TrainingERA5Config, open_era5_zarr, load_era5_slice,
        era5_to_latlon_carry, regrid_2d_to_gaussian,
    )
    from legoesm.driver.compiled_segments import pack_forcing    # noqa: F401


def test_sfno_mode_symbols_resolve():
    # HARD assertion, no skip: the old skip-on-missing masked the sfno
    # factory referencing a nonexistent API for weeks (the symbol was never
    # on main; the "stale worktree" comment was wrong) — the sfno smoke then
    # died on ImportError (#797). Every symbol the factory uses must import.
    from legoesm.training.sfno_dycore_coupling import (      # noqa: F401
        SFNOPhysics, make_sfno_step_unified_latlon,
    )
    from legoesm.grids.gaussian import create_gaussian_grid  # noqa: F401
    from legoesm.grids.regridding import (                   # noqa: F401
        compute_latlon_to_voronoi_weights, regrid_scalar,
    )
    from legoesm.ml.sfno import SFNO, SFNOConfig             # noqa: F401
    # the wired signature: (sfno_physics, w_ll2g, w_g2ll)
    import inspect
    params = list(inspect.signature(make_sfno_step_unified_latlon).parameters)
    assert params == ["sfno_physics", "w_latlon_to_gauss", "w_gauss_to_latlon"]


def test_run_amip_parser_available():
    # build_latlon_config reuses run_amip's parser, which needs current-main
    # config symbols (e.g. EvaluationConfig). Skip on a stale checkout.
    m = importlib.import_module("legoesm.training.scale_build")
    try:
        ra = m._load_run_amip()
    except ImportError as exc:
        pytest.skip(f"run_amip parser needs current-main config API ({exc})")
    assert hasattr(ra, "build_arg_parser") and hasattr(ra, "build_config_from_args")
