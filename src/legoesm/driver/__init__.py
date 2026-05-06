"""Composable model driver for legoESM.

Extracts the shared grid-creation, physics-setup, time-stepping,
diagnostics, and checkpoint logic from the monolithic AMIP scripts
into reusable components.

Usage::

    from legoesm.driver import (
        GridConfig, DycoreConfig, OutputConfig, ExperimentConfig,
        PhysicsPipeline, DiagnosticCollector, ModelDriver,
    )

    config = ExperimentConfig(...)
    driver = ModelDriver(config)
    driver.setup()
    driver.run()
"""

from legoesm.driver.config import (
    GridConfig,
    DycoreConfig,
    OutputConfig,
    ExperimentConfig,
    experiment_config_to_dict,
    experiment_config_from_dict,
    save_experiment_config,
    load_experiment_config,
)
from legoesm.driver.physics_pipeline import PhysicsPipeline, PhysicsOutput, HeldRadiation, build_physics_pipeline
from legoesm.driver.grid_adapters import ColumnAdapter, SingleColumnGrid, make_adapter
from legoesm.driver.kernel_registry import (
    RADIATION_REGISTRY, CONVECTION_REGISTRY, MICROPHYSICS_REGISTRY,
    resolve_kernel, available_schemes,
)
from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.driver.model_driver import ModelDriver
from legoesm.driver.run_status import status_to_exit_code
from legoesm.driver.earth_system_driver import EarthSystemDriver
from legoesm.driver.coupled_esm_driver import CoupledESMDriver
from legoesm.driver.coupled_config import CoupledConfig, PRESETS as COUPLED_PRESETS
