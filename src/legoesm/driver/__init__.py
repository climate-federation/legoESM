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
)
from legoesm.driver.physics_pipeline import PhysicsPipeline, PhysicsOutput, HeldRadiation, build_physics_pipeline
from legoesm.driver.diagnostics import DiagnosticCollector
from legoesm.driver.model_driver import ModelDriver
