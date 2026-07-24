"""Nested-CLUBB tunability in the shipped SCM parameter-tuning drivers.

Full CLUBB nests its tunable coefficients in ``CLUBBConfig.params`` (a
``CLUBBParams``), so the ``__param_spec__``/scheme_key live one level down. The
SCM-RCE campaign and the AD trainer must descend into ``.params`` to discover and
apply CLUBB overrides — otherwise ``_scheme_key_for_subconfig`` returns ``None``
and CLUBB tuning silently no-ops (Codex review 2026-07-11, finding 2). These are
fast unit checks of the discovery + apply + re-wrap plumbing (no CRM reference
data / SCM integration required).
"""
from __future__ import annotations

from legoesm.atmosphere.physics.combined import PhysicsConfig
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig, CLUBBParams
from legoesm.atmosphere.physics.turbulence.config import SmagorinskyConfig
from legoesm.atmosphere.physics.turbulence.tunable_subconfig import (
    rewrap_tunable_subconfig,
    tunable_subconfig,
)
from legoesm.training.param_collector import apply_param_overrides

import scripts.run.run_scm_rce_campaign as campaign


def test_scheme_key_descends_into_nested_clubb_params():
    assert campaign._scheme_key_for_subconfig(CLUBBConfig()) == "atm.turb.CLUBBParams"
    # a flat scheme still resolves to its own class key (no regression).
    assert (campaign._scheme_key_for_subconfig(SmagorinskyConfig())
            == "atm.turb.SmagorinskyConfig")


def test_tunable_and_rewrap_clubb_vs_flat():
    # CLUBB: tunable target is the nested CLUBBParams; rewrap restores the wrapper
    # and preserves its other fields (surface, clubb_dt, ...).
    cfg = CLUBBConfig()
    tunable = tunable_subconfig(cfg)
    assert isinstance(tunable, CLUBBParams)
    tuned = apply_param_overrides(tunable, {"C1": 2.5})
    wrapped = rewrap_tunable_subconfig(cfg, tuned)
    assert isinstance(wrapped, CLUBBConfig)
    assert wrapped.params.C1 == 2.5
    assert wrapped.surface == cfg.surface and wrapped.clubb_dt == cfg.clubb_dt
    # flat scheme: both helpers are identity.
    smag = SmagorinskyConfig()
    assert tunable_subconfig(smag) is smag
    assert rewrap_tunable_subconfig(smag, smag) is smag


def _clubb_physics_config() -> PhysicsConfig:
    base = PhysicsConfig()
    turb = base.turbulence._replace(scheme="clubb", clubb=CLUBBConfig())
    return base._replace(turbulence=turb)


def test_campaign_active_subconfig_roundtrip_sets_nested_param():
    base = _clubb_physics_config()
    _component, scheme, subcfg = campaign._active_subconfig(base, "turbulence")
    assert scheme == "clubb"
    scheme_key = campaign._scheme_key_for_subconfig(subcfg)
    assert scheme_key == "atm.turb.CLUBBParams"
    tuned_tunable = apply_param_overrides(
        tunable_subconfig(subcfg), {"gamma_coef": 0.42})
    tuned_sub = rewrap_tunable_subconfig(subcfg, tuned_tunable)
    new_cfg = campaign._set_active_subconfig(base, "turbulence", tuned_sub)
    assert new_cfg.turbulence.clubb.params.gamma_coef == 0.42
    # untouched coefficient keeps its default.
    assert new_cfg.turbulence.clubb.params.C1 == CLUBBParams().C1


def test_train_static_apply_reaches_nested_clubb_param():
    from scripts.run.train_scm_rce_params import _apply_static_record_values
    base = _clubb_physics_config()
    records = [{"scheme_key": "atm.turb.CLUBBParams",
               "parameter": "beta", "tuned": 3.1}]
    out = _apply_static_record_values(base, records)
    assert out.turbulence.clubb.params.beta == 3.1
