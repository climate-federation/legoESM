"""Strict launch-config loading refuses to silently drop keys.

Motivated by a real failure (2026-08-02): a synthesised config used
``grid.type`` where this repo's field is ``grid_type``. The permissive loader
dropped it without a word, so the config would have loaded cleanly and run a
DIFFERENT experiment than the file described. Checkpoint reload keeps the
permissive behaviour on purpose; launch paths opt into strict.
"""
from __future__ import annotations

import pytest

from legoesm.driver.config import (
    ExperimentConfig,
    experiment_config_from_dict,
    experiment_config_to_dict,
)


def _round_trip_dict():
    return experiment_config_to_dict(ExperimentConfig())


def test_permissive_default_still_drops_unknown_keys():
    """Checkpoint forward-compat must not regress."""
    d = _round_trip_dict()
    d["a_field_removed_in_a_later_version"] = 7
    cfg = experiment_config_from_dict(d)          # no strict= -> permissive
    assert isinstance(cfg, ExperimentConfig)


def test_strict_rejects_unknown_top_level_key():
    d = _round_trip_dict()
    d["definitely_not_a_field"] = 1
    with pytest.raises(ValueError, match="definitely_not_a_field"):
        experiment_config_from_dict(d, strict=True)


def test_strict_rejects_the_real_grid_type_typo():
    """The exact typo that motivated this: `type` instead of `grid_type`."""
    d = _round_trip_dict()
    d["grid"]["type"] = "cubed_sphere"
    with pytest.raises(ValueError, match=r"grid\.type"):
        experiment_config_from_dict(d, strict=True)


def test_strict_accepts_a_clean_round_trip():
    """Non-vacuity: strict must PASS on a config this repo itself produced,
    otherwise the test above proves nothing about typos specifically."""
    cfg = experiment_config_from_dict(_round_trip_dict(), strict=True)
    assert isinstance(cfg, ExperimentConfig)
