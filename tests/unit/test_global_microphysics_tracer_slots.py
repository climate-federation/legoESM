"""Global driver microphysics tracer-slot sizing."""

from __future__ import annotations

import pytest

from legoesm.atmosphere.physics.microphysics.integration import min_tracer_slots
from legoesm.core.tracers import (
    init_tracers,
    make_full_moisture_registry,
    make_moisture_registry,
)
from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
from legoesm.driver.model_driver import ModelDriver


def _cfg(microphysics: str) -> ExperimentConfig:
    return ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"),
        microphysics=microphysics,
    )


@pytest.mark.parametrize(
    "scheme",
    (
        "none",
        "kessler",
        "sundqvist",
        "sdm",
        "morrison",
        "thompson",
        "p3",
        "seifert_beheng",
        "fast_sbm",
    ),
)
def test_global_model_driver_allocates_min_microphysics_tracer_slots(
    scheme,
    tmp_path,
):
    driver = ModelDriver(_cfg(scheme), output_dir=tmp_path / scheme)
    need_slots = min_tracer_slots(scheme)
    warm_slots = make_moisture_registry().n_tracers
    full_slots = make_full_moisture_registry().n_tracers
    expected_slots = warm_slots if need_slots <= warm_slots else full_slots

    assert driver.tracer_registry.n_tracers == expected_slots
    assert driver.tracer_registry.n_tracers >= need_slots
    driver.tracers = init_tracers(driver.tracer_registry, shape_3d=(1, 1, 1))
    assert driver._validate_microphysics_tracer_state() == need_slots


def test_global_model_driver_rejects_undersized_microphysics_tracer_state(
    tmp_path,
):
    driver = ModelDriver(_cfg("fast_sbm"), output_dir=tmp_path / "fast_sbm")
    driver.tracers = {
        "q_v": object(),
        "q_c": object(),
        "q_r": object(),
    }

    with pytest.raises(ValueError, match="fast_sbm.*have=3, need=9"):
        driver._validate_microphysics_tracer_state(
            context="synthetic undersized tracer state",
        )
