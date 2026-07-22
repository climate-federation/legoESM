"""CLM-ML stomatal-conductance model selection + per-site Vcmax override.

``CLMMLCanopyConfig.stomatal_model`` selects the backend ``gs_type``
(medlyn=0, ball_berry=1, wue=2 default).  The applier must patch BOTH module
namespaces (``MLLeafPhotosynthesisMod`` imports ``gs_type`` by value, the same
trap as the layering counts), and ``vcmax25_override`` — the per-site value that
goes beyond the global PFT table — requires the Medlyn path.  These need no
``clm-ml-jax``: the backend modules are faked in ``sys.modules``.
"""
from __future__ import annotations

import sys
import types

import pytest

from legoesm.land.canopy.clm_ml_interface import _apply_stomatal_model
from legoesm.land.canopy.config import (
    CLM_ML_STOMATAL_GS_TYPE,
    VALID_CLM_ML_STOMATAL_MODELS,
    CLMMLCanopyConfig,
)


@pytest.fixture
def fake_backend(monkeypatch):
    ctl = types.ModuleType("multilayer_canopy.MLclm_varctl")
    ctl.gs_type = 2
    photo = types.ModuleType("multilayer_canopy.MLLeafPhotosynthesisMod")
    photo.gs_type = 2                      # by-value copy the kernel branches on
    pkg = sys.modules.get("multilayer_canopy") or types.ModuleType("multilayer_canopy")
    monkeypatch.setitem(sys.modules, "multilayer_canopy", pkg)
    monkeypatch.setitem(sys.modules, "multilayer_canopy.MLclm_varctl", ctl)
    monkeypatch.setitem(sys.modules,
                        "multilayer_canopy.MLLeafPhotosynthesisMod", photo)
    return ctl, photo


def test_default_is_wue():
    assert CLMMLCanopyConfig().stomatal_model == "wue"


def test_gs_type_map_is_the_three_backend_codes():
    assert CLM_ML_STOMATAL_GS_TYPE == {"medlyn": 0, "ball_berry": 1, "wue": 2}
    assert VALID_CLM_ML_STOMATAL_MODELS == ("medlyn", "ball_berry", "wue")


@pytest.mark.parametrize("scheme,gs", list(CLM_ML_STOMATAL_GS_TYPE.items()))
def test_apply_patches_both_namespaces(fake_backend, scheme, gs):
    """The by-value consumer namespace is the one the kernel actually reads."""
    ctl, photo = fake_backend
    _apply_stomatal_model(CLMMLCanopyConfig(stomatal_model=scheme))
    assert ctl.gs_type == gs
    assert photo.gs_type == gs          # would stay stale if only ctl patched


def test_apply_rejects_unknown_before_touching_backend():
    with pytest.raises(ValueError, match="unknown CLM-ML stomatal_model"):
        _apply_stomatal_model(CLMMLCanopyConfig(stomatal_model="bogus"))


def test_validate_rejects_unknown_stomatal_model():
    with pytest.raises(ValueError, match="unknown CLM-ML stomatal_model"):
        CLMMLCanopyConfig(stomatal_model="monteith").validate()


def test_validate_accepts_every_valid_model():
    for scheme in VALID_CLM_ML_STOMATAL_MODELS:
        cfg = CLMMLCanopyConfig(stomatal_model=scheme)
        assert cfg.validate() is cfg


def test_switching_model_clears_kernel_caches(fake_backend, monkeypatch):
    """The backend leaf-kernel lru_caches omit gs_type from their key, so a
    stomatal-model SWITCH in one process must clear them or a stale kernel is
    reused with the new model's params (codex).  A no-switch re-apply must NOT
    pay the clear."""
    import legoesm.land.canopy.clm_ml_interface as iface
    _, photo = fake_backend
    cleared = {"n": 0}

    class _Cached:              # stand-in for an lru_cache'd factory
        def cache_clear(self):
            cleared["n"] += 1
    photo.some_kernel_factory = _Cached()
    monkeypatch.setattr(iface, "_APPLIED_GS_TYPE", None, raising=False)

    iface._apply_stomatal_model(CLMMLCanopyConfig(stomatal_model="medlyn"))
    assert cleared["n"] == 0                      # first apply: nothing to clear
    iface._apply_stomatal_model(CLMMLCanopyConfig(stomatal_model="medlyn"))
    assert cleared["n"] == 0                      # same model: no clear
    iface._apply_stomatal_model(CLMMLCanopyConfig(stomatal_model="ball_berry"))
    assert cleared["n"] == 1                      # switch: caches cleared


def test_vcmax_override_builds_the_right_pft_entry():
    """vcmax25_override edits ONLY this PFT's vcmaxpft entry (the injection the
    backend N-profile reads), leaving the rest of the table intact."""
    import jax.numpy as jnp
    base = jnp.arange(20.0, 40.0)                 # a fake vcmaxpft table
    pft = 7
    out = base.at[pft].set(55.0)
    assert float(out[pft]) == 55.0
    assert jnp.array_equal(out.at[pft].set(base[pft]), base)  # only pft changed
