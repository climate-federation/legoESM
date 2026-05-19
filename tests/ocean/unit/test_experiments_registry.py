"""Smoke tests for the experiments package registry.

Catches drift between the source modules under ``src/legoesm/ocean/experiments/``
and the ``AVAILABLE_EXPERIMENTS`` dict that other consumers
(scripts, tests, the ocean fidelity layer) rely on.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

from legoesm.ocean import experiments


EXPECTED_EXPERIMENTS = {
    "rest_state",
    "barotropic_wave",
    "regional_gyre",
    "baroclinic",
    "baroclinic_gyre",
    "phillips_two_layer",
    "inertia_gravity_wave",
    "lock_exchange",
    "overflow",
    "stommel_gyre_tracer",
    "geostrophic_adjustment",
    "global_barotropic_wind",
    "eady_instability",
    "eady_uniform",
    "acc_channel",
    "global_overturning",
    "dino",
}


def test_all_expected_experiments_registered():
    missing = EXPECTED_EXPERIMENTS - set(experiments.AVAILABLE_EXPERIMENTS)
    assert not missing, f"AVAILABLE_EXPERIMENTS is missing: {sorted(missing)}"


def test_every_registered_experiment_has_config_class():
    for name, cfg in experiments.AVAILABLE_EXPERIMENTS.items():
        assert isinstance(cfg, dict), f"{name} EXPERIMENT_CONFIG must be dict"
        assert "name" in cfg, f"{name} EXPERIMENT_CONFIG missing 'name'"
        assert cfg["name"] == name, (
            f"EXPERIMENT_CONFIG['name']={cfg['name']!r} disagrees with "
            f"registered key {name!r}"
        )


def test_no_orphan_modules_in_package():
    """Every .py module under experiments/ must be registered.

    A module that lands in ``src/legoesm/ocean/experiments/`` but is not added
    to ``AVAILABLE_EXPERIMENTS`` is the exact bit-rot mode that bit
    ``eady_uniform`` for months; this test catches a recurrence.
    """
    package_modules = {
        info.name
        for info in pkgutil.iter_modules(experiments.__path__)
        if not info.name.startswith("_")
    }
    registered = set(experiments.AVAILABLE_EXPERIMENTS)
    orphans = package_modules - registered
    assert not orphans, (
        f"modules under experiments/ are not in AVAILABLE_EXPERIMENTS: "
        f"{sorted(orphans)}. Add them to __init__.py."
    )


def test_eady_uniform_specifically_present():
    """Regression guard for the 2026-05 audit finding."""
    assert "eady_uniform" in experiments.AVAILABLE_EXPERIMENTS
    cfg = experiments.AVAILABLE_EXPERIMENTS["eady_uniform"]
    assert cfg["name"] == "eady_uniform"
    assert callable(cfg["create_initial_conditions"])
    assert callable(cfg["create_forcings"])
    assert callable(cfg["validate"])
    assert cfg["grid_support"]["latlon_channel"] is True
