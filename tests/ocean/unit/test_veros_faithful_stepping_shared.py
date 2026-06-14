"""Ratchet: the Veros free-run recipes share ONE faithful-stepping fragment (#433).

The five Veros free-run recipes (acc, acc_basic, global_4deg, global_flexible,
global_1deg) must all select the SAME time-stepping composition via the shared
``veros_stepping_common.veros_faithful_stepping`` fragment.  Before #433 this was
five copy-pasted blocks and PR #432 caught a silent divergence (acc_basic shipped
the leaky matsuno_split/"total" composition).  This ratchet makes any future
copy-paste — a recipe that hardcodes a divergent value or omits a new composition
field — go red.

The self-test ``test_ratchet_is_non_vacuous`` proves the check actually catches a
violation (it is a tripwire, not a no-op).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
from legoesm.ocean.fidelity.veros_stepping_common import (
    VEROS_FAITHFUL_STEPPING_SIGNATURE,
    veros_faithful_stepping,
)
from legoesm.ocean.state import LatLonCGridOceanConfig


def _freerun_configs():
    """The five Veros free-run model configs (acc/acc_basic on the free-run
    path; the three globals are native free-run)."""
    from legoesm.ocean.fidelity.veros_acc_basic_recipe import build_acc_basic_recipe
    from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_recipe
    from legoesm.ocean.fidelity.veros_global_1deg_recipe import (
        build_global_1deg_model_config,
    )
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
        build_global_4deg_model_config,
    )
    from legoesm.ocean.fidelity.veros_global_flexible_recipe import (
        build_global_flexible_model_config,
    )
    return {
        "acc": build_acc_recipe(with_surface_forcing=True).model_config,
        "acc_basic": build_acc_basic_recipe(
            with_surface_forcing=True).model_config,
        "global_4deg": build_global_4deg_model_config(),
        "global_flexible": build_global_flexible_model_config(),
        "global_1deg": build_global_1deg_model_config(),
    }


def _carries_signature(cfg) -> bool:
    return all(getattr(cfg, f) == v
               for f, v in VEROS_FAITHFUL_STEPPING_SIGNATURE.items())


# --------------------------------------------------------------- the fragment

def test_fragment_returns_signature_plus_ratio():
    out = veros_faithful_stepping(dt_mom_ratio=7.0)
    assert out == {**VEROS_FAITHFUL_STEPPING_SIGNATURE, "dt_mom_ratio": 7.0}
    assert isinstance(out["dt_mom_ratio"], float)


def test_signature_fields_are_real_config_fields():
    fields = set(LatLonCGridOceanConfig._fields)
    for f in VEROS_FAITHFUL_STEPPING_SIGNATURE:
        assert f in fields, f"{f} is not a LatLonCGridOceanConfig field"
    assert "dt_mom_ratio" in fields


def test_fragment_constructs_a_valid_config():
    """The splat must build a LatLonCGridOceanConfig without unknown/duplicate
    kwargs (and pass its own validation via the model)."""
    cfg = LatLonCGridOceanConfig(**veros_faithful_stepping(dt_mom_ratio=9.0))
    assert _carries_signature(cfg)
    assert cfg.dt_mom_ratio == 9.0


# --------------------------------------------------------------- the ratchet

@pytest.mark.parametrize("name", [
    "acc", "acc_basic", "global_4deg", "global_flexible", "global_1deg"])
def test_freerun_recipe_carries_shared_fragment(name):
    cfg = _freerun_configs()[name]
    for field, val in VEROS_FAITHFUL_STEPPING_SIGNATURE.items():
        assert getattr(cfg, field) == val, (
            f"{name}.{field} = {getattr(cfg, field)!r} diverged from the shared "
            f"Veros-faithful stepping fragment ({field}={val!r}); use "
            f"veros_faithful_stepping(...) instead of copy-pasting")


def test_per_setup_dt_mom_ratios():
    from legoesm.ocean.fidelity.veros_global_1deg_recipe import DT_MOM_RATIO as R1
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import DT_MOM_RATIO as R4
    from legoesm.ocean.fidelity.veros_global_flexible_recipe import DT_MOM_RATIO as RF
    cfgs = _freerun_configs()
    assert cfgs["acc"].dt_mom_ratio == 9.0
    assert cfgs["global_4deg"].dt_mom_ratio == R4
    assert cfgs["global_flexible"].dt_mom_ratio == RF
    assert cfgs["global_1deg"].dt_mom_ratio == R1


# ------------------------------------------------- gating: frozen path is OFF

@pytest.mark.parametrize("builder_name", ["acc", "acc_basic"])
def test_frozen_path_keeps_config_defaults(builder_name):
    """The acc/acc_basic frozen-state probe path (with_surface_forcing=False)
    must NOT carry the faithful composition — every signature field equals the
    LatLonCGridOceanConfig default (so omitting the splat is bit-identical)."""
    from legoesm.ocean.fidelity.veros_acc_basic_recipe import build_acc_basic_recipe
    from legoesm.ocean.fidelity.veros_acc_recipe import build_acc_recipe
    build = {"acc": build_acc_recipe, "acc_basic": build_acc_basic_recipe}[
        builder_name]
    cfg = build(with_surface_forcing=False).model_config
    defaults = LatLonCGridOceanConfig._field_defaults
    for field in VEROS_FAITHFUL_STEPPING_SIGNATURE:
        assert getattr(cfg, field) == defaults[field], (
            f"{builder_name} frozen path {field} != default")
    assert cfg.dt_mom_ratio == defaults["dt_mom_ratio"]


# ------------------------------------------------------------- non-vacuous

def test_ratchet_is_non_vacuous():
    """A config that diverges from the fragment must FAIL _carries_signature —
    proving the ratchet is a real tripwire."""
    good = LatLonCGridOceanConfig(**veros_faithful_stepping(dt_mom_ratio=9.0))
    assert _carries_signature(good)
    bad = good._replace(coriolis_scheme="matsuno_split")  # the #432 divergence
    assert not _carries_signature(bad)
