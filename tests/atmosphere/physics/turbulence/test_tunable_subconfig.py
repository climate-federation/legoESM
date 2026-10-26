"""Tests for the shared turbulence tunable-leaf descend/re-wrap helpers."""
from __future__ import annotations

from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig, CLUBBParams
from legoesm.atmosphere.physics.turbulence.config import MYNN25Config
from legoesm.atmosphere.physics.turbulence.tunable_subconfig import (
    rewrap_tunable_subconfig,
    tunable_subconfig,
)


def test_flat_config_is_its_own_tunable_leaf():
    cfg = MYNN25Config()
    assert tunable_subconfig(cfg) is cfg  # flat scheme: no descent


def test_flat_config_rewrap_returns_tuned_leaf():
    cfg = MYNN25Config()
    tuned = MYNN25Config(A1=0.9)
    assert rewrap_tunable_subconfig(cfg, tuned) is tuned  # flat: identity re-wrap


def test_clubb_descends_to_nested_params():
    cfg = CLUBBConfig()
    leaf = tunable_subconfig(cfg)
    assert isinstance(leaf, CLUBBParams)
    assert leaf is cfg.params  # the nested tuple owns the tunable coefficients


def test_clubb_rewrap_folds_params_back_and_preserves_wrapper():
    cfg = CLUBBConfig(clubb_dt=123.0)  # a non-default wrapper field to check preserved
    tuned_params = cfg.params._replace(C1=cfg.params.C1 * 1.1)
    out = rewrap_tunable_subconfig(cfg, tuned_params)
    assert isinstance(out, CLUBBConfig)
    assert out.params is tuned_params           # nested params replaced
    assert out.clubb_dt == 123.0                # wrapper fields preserved


def test_none_passes_through():
    assert tunable_subconfig(None) is None


def test_roundtrip_identity_for_clubb():
    cfg = CLUBBConfig()
    # descend then re-wrap the SAME leaf reproduces an equal config
    assert rewrap_tunable_subconfig(cfg, tunable_subconfig(cfg)) == cfg
