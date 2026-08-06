"""CLM-ML canopy layer counts — config contract + the two-namespace install.

``CLMMLCanopyConfig.nlevmlcan`` used to be inert: nothing wrote it to the
backend, so CLM-ML always built its vertical structure in height-increment mode
(``dz_tall = 0.5 m``).  For a real forest canopy (htop ~ 27 m) that is ~54
layers whose thin beta-distribution tails fall below ``dpai_min``, and the run
aborts in ``initVerticalStructure`` with "canopy layer has zero plant area
index".  ``_apply_canopy_layering`` installs the explicit counts instead.

The install must reach TWO module namespaces: ``MLinitVerticalMod`` does
``from MLclm_varctl import nlayer_within, nlayer_above`` — a by-value import, so
writing only ``MLclm_varctl`` is a silent no-op.  These tests need no
``clm-ml-jax``: the two backend modules are faked in ``sys.modules``.
"""

from __future__ import annotations

import sys
import types

import pytest

from legoesm.land.canopy.clm_ml_interface import _apply_canopy_layering
from legoesm.land.canopy.config import CLMMLCanopyConfig


@pytest.fixture
def fake_backend(monkeypatch):
    """Stand-in ``MLclm_varctl`` / ``MLinitVerticalMod`` carrying the counts."""
    ctl = types.ModuleType("multilayer_canopy.MLclm_varctl")
    ctl.nlayer_within = 0        # backend default: height-increment mode
    ctl.nlayer_above = 0
    vert = types.ModuleType("multilayer_canopy.MLinitVerticalMod")
    vert.nlayer_within = 0       # by-value copies of the same defaults
    vert.nlayer_above = 0
    pkg = sys.modules.get("multilayer_canopy") or types.ModuleType("multilayer_canopy")
    monkeypatch.setitem(sys.modules, "multilayer_canopy", pkg)
    monkeypatch.setitem(sys.modules, "multilayer_canopy.MLclm_varctl", ctl)
    monkeypatch.setitem(sys.modules, "multilayer_canopy.MLinitVerticalMod", vert)
    monkeypatch.setattr(pkg, "MLclm_varctl", ctl, raising=False)
    monkeypatch.setattr(pkg, "MLinitVerticalMod", vert, raising=False)
    return ctl, vert


def test_default_layering_is_nine_layers():
    cfg = CLMMLCanopyConfig()
    assert (cfg.nlevmlcan, cfg.nlayer_above) == (9, 1)


def test_apply_writes_both_namespaces(fake_backend):
    """The by-value consumer namespace is the one the backend actually reads."""
    ctl, vert = fake_backend
    _apply_canopy_layering(CLMMLCanopyConfig())
    for mod in (ctl, vert):
        assert (mod.nlayer_within, mod.nlayer_above) == (8, 1)


def test_within_plus_above_equals_nlevmlcan(fake_backend):
    ctl, vert = fake_backend
    cfg = CLMMLCanopyConfig(nlevmlcan=13, nlayer_above=3)
    _apply_canopy_layering(cfg)
    assert vert.nlayer_within + vert.nlayer_above == cfg.nlevmlcan
    assert (vert.nlayer_within, vert.nlayer_above) == (10, 3)


def test_explicit_mode_is_selected_not_height_increment(fake_backend):
    """Both counts strictly positive => the backend takes the explicit branch.

    ``MLinitVerticalMod`` selects height-increment mode when EITHER count is
    zero, so a partial install would silently restore the aborting default.
    """
    ctl, vert = fake_backend
    _apply_canopy_layering(CLMMLCanopyConfig())
    assert vert.nlayer_within > 0 and vert.nlayer_above > 0


@pytest.mark.parametrize("nlevmlcan,nlayer_above", [(9, 0), (1, 1), (2, 3), (9, -1)])
def test_validate_rejects_degenerate_layering(nlevmlcan, nlayer_above):
    """A layering that would fall back to (or under-run) the backend default."""
    cfg = CLMMLCanopyConfig(nlevmlcan=nlevmlcan, nlayer_above=nlayer_above)
    with pytest.raises(ValueError, match="canopy layering"):
        cfg.validate()


def test_validate_accepts_the_default():
    cfg = CLMMLCanopyConfig()
    assert cfg.validate() is cfg


@pytest.mark.parametrize("bad", [9.5, 9.0, True])
def test_validate_rejects_non_integer_counts(bad):
    """Float/bool counts silently truncate under the backend int() cast."""
    cfg = CLMMLCanopyConfig(nlevmlcan=bad)
    with pytest.raises(ValueError, match="must be a plain int"):
        cfg.validate()
