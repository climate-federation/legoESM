"""Direct tests for the full-CLUBB ``__param_spec__`` (CLUBBParams + CLUBBConfig).

The module-level structural gate lives in ``tests/test_param_specs.py`` (AST-only,
covers every physics config). This file exercises the *runtime* consequences that
make CLUBB tunable on the same footing as the other closures (the Q2-fairness
requirement of the LES suite): the collector materialises the coefficients, the
tier continuum selects the canonical primary knobs, and trained overrides splice
back into the NESTED ``CLUBBConfig.params`` tuple.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest
from legoesm.atmosphere.physics.turbulence import clubb as clubb_mod
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig, CLUBBParams
from legoesm.core.param_overrides import apply_param_overrides
from legoesm.training.param_collector import (
    build_registry,
    build_trainable_params,
)

_PARAMS_KEY = "atm.turb.CLUBBParams"
_CONFIG_KEY = "atm.turb.CLUBBConfig"
# The canonical primary CLUBB tuning knobs (what tier-1 "core" must select).
_EXPECTED_CORE = {"C1", "C8", "C11", "C14", "beta", "c_K", "gamma_coef", "mu"}


def test_spec_present_and_registered():
    spec = clubb_mod.__param_spec__
    assert set(spec) == {"CLUBBParams", "CLUBBConfig"}
    assert spec["CLUBBParams"]["scheme_key"] == _PARAMS_KEY
    assert spec["CLUBBConfig"]["scheme_key"] == _CONFIG_KEY
    # CLUBBConfig floats are all tolerances/timestep/reference -> nothing tunable.
    assert spec["CLUBBConfig"]["params"] == {}
    from legoesm.training.param_collector import SPEC_MODULES
    assert "legoesm.atmosphere.physics.turbulence.clubb" in SPEC_MODULES


def test_every_param_default_seeds_and_round_trips():
    """Each param's default must lie strictly inside its bounds AND clear the
    sigmoid clamp margin, so the collector seed round-trips to the default.

    range_to_sigmoid_array clamps the normalised value to [1e-3, 1-1e-3]; a
    default at/near a bound (e.g. an off-in-CAM 0.0 term) would seed to a
    DIFFERENT value, silently perturbing 'tuning from default'. Those terms are
    excluded, so every remaining param must have margin. (Codex review finding 1.)
    """
    margin = 1e-3
    defaults = CLUBBParams()._asdict()
    for field, p in clubb_mod.__param_spec__["CLUBBParams"]["params"].items():
        lo, hi = p["bounds"]
        d = defaults[field]
        assert lo < hi, f"{field}: bounds not ordered"
        assert lo <= d <= hi, f"{field}: default {d} outside {p['bounds']}"
        if p["transform"] == "sigmoid":
            frac = (d - lo) / (hi - lo)
            assert margin < frac < 1.0 - margin, (
                f"{field}: default {d} is within the sigmoid clamp margin of a "
                f"bound {p['bounds']} (frac={frac:.4f}) — it will not round-trip; "
                f"exclude it or widen the bound")


def test_registry_and_tier_continuum():
    reg = {m.qualified_name: m for m in build_registry() if m.scheme_key == _PARAMS_KEY}
    # 78 tunable = 8 core + 40 extended + 30 aggressive (9 off-in-CAM zero-default
    # terms are excluded — a sigmoid seed cannot represent a lower-bound default).
    assert len(reg) == 78
    core = build_trainable_params(active_scheme_keys={_PARAMS_KEY}, tier="core")
    ext = build_trainable_params(active_scheme_keys={_PARAMS_KEY}, tier="extended")
    agg = build_trainable_params(active_scheme_keys={_PARAMS_KEY}, tier="aggressive")
    core_fields = {n.split(".")[-1] for n in core.raw_values}
    assert core_fields == _EXPECTED_CORE
    # continuum is monotone and covers every registered param at "aggressive".
    assert len(core.raw_values) < len(ext.raw_values) < len(agg.raw_values)
    assert len(agg.raw_values) == 78


def test_config_key_has_no_tunable_params():
    # The CLUBBConfig entry exists only to classify its float fields (all
    # excluded: tolerances / sub-step / reference T0). It contributes no
    # trainable params, so it never becomes a registered scheme_key.
    scheme_keys = {m.scheme_key for m in build_registry()}
    assert _PARAMS_KEY in scheme_keys
    assert _CONFIG_KEY not in scheme_keys


def test_override_round_trips_into_nested_params():
    base = CLUBBConfig()
    tp = build_trainable_params(active_scheme_keys={_PARAMS_KEY}, tier="core")
    overrides = {k: float(v) for k, v in tp.to_overrides()[_PARAMS_KEY].items()}
    new_params = apply_param_overrides(base.params, overrides)
    cfg = base._replace(params=new_params)
    assert isinstance(cfg.params, CLUBBParams)
    # seeded-from-default: the core knobs recover their defaults through the
    # sigmoid round-trip (float32 seed tolerance).
    for field in _EXPECTED_CORE:
        assert getattr(cfg.params, field) == pytest.approx(
            getattr(base.params, field), rel=1e-3
        )
    # a non-selected coefficient is untouched.
    assert cfg.params.C10 == base.params.C10


def test_apply_rejects_unknown_field():
    with pytest.raises(ValueError):
        apply_param_overrides(CLUBBParams(), {"not_a_clubb_param": jnp.asarray(1.0)})
