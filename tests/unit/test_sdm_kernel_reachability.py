"""The SDM collision kernel is selectable, and defaults to a PHYSICAL kernel.

Two defects, one root cause. ``MicrophysicsConfig`` was built as
``MicrophysicsConfig(scheme=cfg.microphysics)`` at every driver build site, so
the whole ``SDMConfig`` sub-config was unreachable from ``ExperimentConfig``.
And ``SDMConfig.collision_kernel`` defaulted to ``"golovin"`` -- which its own
docstring calls the "analytic test kernel": K = b(x+y) has a closed-form
solution and exists to verify the coalescence solver against PySDM. So every
production ``--microphysics sdm`` run silently used a *verification* kernel for
collision-coalescence, and no config file could change it.

The sibling bin scheme settles the intended default: ``FastSBMConfig`` already
defaults to ``"hall"`` (Hall 1980, physical).

``str`` fields are not ``__param_spec__``-eligible by design (only ``:float``
is), so ``--params`` could never have reached this -- a CLI/config field is the
only possible route.
"""
from __future__ import annotations

import pytest
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.microphysics.fast_sbm.config import FastSBMConfig
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.driver.config import VALID_SDM_COLLISION_KERNELS, ExperimentConfig
from legoesm.driver.physics_pipeline import microphysics_config_for


def test_sdm_defaults_to_a_physical_kernel():
    """golovin is a verification kernel; it must not be the production default."""
    assert SDMConfig().collision_kernel == "hall", (
        "SDM defaults to the analytic TEST kernel -- a production run would use "
        "unphysical collision-coalescence"
    )


def test_sdm_default_matches_its_sibling_bin_scheme():
    """Two bin schemes in one model should not disagree on the physical kernel."""
    assert SDMConfig().collision_kernel == FastSBMConfig().collision_kernel


def test_experiment_config_defaults_to_physical_kernel():
    assert ExperimentConfig().sdm_collision_kernel == "hall"


@pytest.mark.parametrize("kernel", VALID_SDM_COLLISION_KERNELS)
def test_kernel_reaches_the_nested_sdm_config(kernel):
    """The selector must reach SDMConfig through the SHARED builder every driver
    backend uses -- not merely land on ExperimentConfig.

    This is the load-bearing assertion: pinning only the intermediate
    ExperimentConfig would pass even if no backend consumed the value.
    """
    cfg = ExperimentConfig(microphysics="sdm", sdm_collision_kernel=kernel)
    assert microphysics_config_for(cfg).sdm.collision_kernel == kernel


def test_golovin_is_still_selectable_for_verification():
    """The PySDM validation scripts set golovin explicitly; demoting it from the
    default must not remove it."""
    assert "golovin" in VALID_SDM_COLLISION_KERNELS
    cfg = ExperimentConfig(microphysics="sdm", sdm_collision_kernel="golovin")
    assert microphysics_config_for(cfg).sdm.collision_kernel == "golovin"


def test_unknown_kernel_rejected_by_validate_strict():
    """Dispatch-hardening: a typo must fail at config validation, not at JIT."""
    cfg = ExperimentConfig(microphysics="sdm", sdm_collision_kernel="haII")
    with pytest.raises(ValueError, match="sdm_collision_kernel"):
        cfg.validate_strict()


def test_non_sdm_schemes_are_untouched():
    """The builder must only _replace the sub-config of the ACTIVE scheme, so
    every other scheme stays byte-identical."""
    for scheme in ("morrison", "thompson", "kessler", "none"):
        cfg = ExperimentConfig(microphysics=scheme, sdm_collision_kernel="long")
        assert microphysics_config_for(cfg) == MicrophysicsConfig(scheme=scheme)


def test_cli_choices_derive_from_the_canonical_tuples():
    """run_amip's allowlists must BE the canonical sets, not copies of them.

    A hardcoded copy is exactly how ``--microphysics ml_emulator`` came to be
    rejected by this CLI while validate_strict accepted it.
    """
    import importlib.util

    from legoesm.driver.config import (
        VALID_CONVECTION_SCHEMES,
        VALID_MICROPHYSICS,
    )

    spec = importlib.util.spec_from_file_location(
        "_run_amip_for_test", "scripts/run/run_amip.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actions = {a.dest: a for a in mod.build_arg_parser()._actions}

    assert set(actions["microphysics"].choices) == set(VALID_MICROPHYSICS)
    assert set(actions["convection"].choices) == set(VALID_CONVECTION_SCHEMES)
    assert set(actions["sdm_collision_kernel"].choices) == set(
        VALID_SDM_COLLISION_KERNELS
    )
