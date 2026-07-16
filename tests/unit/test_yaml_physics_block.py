"""The nested-YAML ``physics:`` block reaches ExperimentConfig.

Regression guard for a silent-drop defect: ``Config.to_experiment_config``
built its canonical dict with keys for grid / dycore / output / radiation and
**no keys at all** for the column-physics axes, so every ``physics:`` key was
parsed and then discarded.  ``experiment_config_from_dict`` drops unknown keys
silently, so the run proceeded on ExperimentConfig defaults and nothing warned.

The concrete casualty was the shipped, ``maturity: run_tested`` Held-Suarez
template: ``physics.forcing: held_suarez`` was dropped, so the *dry*
dynamical-core benchmark ran with ``held_suarez_forcing=False`` plus moist
``sbm`` convection and gray radiation — i.e. not Held-Suarez at all.

Held-Suarez needs BOTH halves and this file pins both, because either alone is
still wrong: HS forcing is *additive* to the physics pipeline (see
``model_driver``: "HS is ADDITIVE ... would apply BOTH the Louis surface stress
AND this Rayleigh drag"), so enabling the forcing while leaving the moist
defaults active merely trades one wrong run for another.  ``run_amip``'s
``_require_full_physics_for_amip`` states the same rule from the other side: HS
is exempt from full-physics only when the ENTIRE stack is dry.
"""
from __future__ import annotations

import pytest

from legoesm.config import Config

# The axes run_amip calls _AMIP_REQUIRED_PHYSICS — exactly the set the YAML
# boundary used to drop.  ``clouds`` is the YAML spelling of ``cloud_scheme``.
_PHYSICS_AXES = {
    "convection": "convection",
    "microphysics": "microphysics",
    "turbulence": "turbulence",
    "clouds": "cloud_scheme",
    "gravity_wave_drag": "gravity_wave_drag",
}

_HS_TEMPLATE = "config/templates/3d_idealized/held_suarez.yaml"


def _ec(d: dict):
    return Config.from_dict(d).to_experiment_config()


def test_held_suarez_template_actually_enables_held_suarez():
    """The shipped HS template must produce a real Held-Suarez configuration."""
    ec = Config.from_yaml(_HS_TEMPLATE).to_experiment_config()
    assert ec.held_suarez_forcing is True, (
        "the Held-Suarez template's physics.forcing was dropped at the YAML "
        "boundary — it would run WITHOUT Newtonian relaxation"
    )


def test_held_suarez_template_is_fully_dry():
    """HS forcing is ADDITIVE, so the moist stack must be off (else the dry
    benchmark runs HS *on top of* convection/radiation)."""
    ec = Config.from_yaml(_HS_TEMPLATE).to_experiment_config()
    active = {
        name: getattr(ec, field)
        for name, field in _PHYSICS_AXES.items()
        if getattr(ec, field) != "none"
    }
    assert not active, (
        f"Held-Suarez is a DRY benchmark but these axes are active: {active}. "
        "HS forcing is additive to the physics pipeline, so this would run "
        "Newtonian relaxation on top of moist physics."
    )
    assert ec.radiation == "none", (
        "HS Newtonian relaxation replaces radiative heating; radiation="
        f"{ec.radiation!r} would double-count the thermal forcing"
    )
    ec.validate_strict()


@pytest.mark.parametrize(
    "yaml_key,ec_field,value",
    [
        ("convection", "convection", "bechtold"),
        ("microphysics", "microphysics", "thompson"),
        ("turbulence", "turbulence", "mynn25"),
        ("clouds", "cloud_scheme", "xu_randall"),
        ("gravity_wave_drag", "gravity_wave_drag", "hines"),
        ("radiation", "radiation", "rrtmgp"),
    ],
)
def test_physics_axis_reaches_experiment_config(yaml_key, ec_field, value):
    """Each physics axis set in YAML must arrive on ExperimentConfig.

    Non-vacuity: every ``value`` differs from the ExperimentConfig default, so
    the assert cannot pass by accident on the default.
    """
    ec = _ec({"physics": {yaml_key: value}})
    assert getattr(ec, ec_field) == value, (
        f"YAML physics.{yaml_key}={value!r} was dropped; ExperimentConfig."
        f"{ec_field} = {getattr(ec, ec_field)!r}"
    )


def test_absent_physics_block_preserves_defaults():
    """A config with no ``physics:`` block must resolve exactly as before the
    mapping existed — the defaults here mirror the ExperimentConfig fields."""
    ec = _ec({})
    assert ec.held_suarez_forcing is False
    assert (ec.convection, ec.microphysics, ec.turbulence) == ("sbm", "none", "none")
    assert (ec.cloud_scheme, ec.gravity_wave_drag, ec.radiation) == (
        "none", "none", "gray",
    )


def test_held_suarez_template_starts_dry():
    """Disabling the moist schemes is not enough: the driver initializes q_v
    UNCONDITIONALLY from rh_init (model_driver: q_v = rh_init * q_sat * sigma^2,
    default 0.7), and that moisture still perturbs the dynamics via virtual
    temperature -> density even with every parameterization off."""
    ec = Config.from_yaml(_HS_TEMPLATE).to_experiment_config()
    assert ec.rh_init == 0.0, (
        f"Held-Suarez template initializes moisture (rh_init={ec.rh_init}); a "
        "dry dynamical-core benchmark must start with zero q_v"
    )


def test_unknown_physics_forcing_raises():
    """An unrecognized forcing must fail LOUDLY, not run something else — the
    dispatch-hardening rule that this whole module regressed against."""
    with pytest.raises(ValueError, match="physics.forcing"):
        _ec({"physics": {"forcing": "newtonian_typo"}})


@pytest.mark.parametrize("bad_key", ["convectoin", "cloud_scheme", "sbm"])
def test_unknown_physics_key_raises(bad_key):
    """A typo'd or wrong-spelling physics key must be rejected.

    Mapping only the recognized keys would preserve the exact defect this
    module fixes: ``convectoin: bechtold`` silently running ``sbm``.
    ``cloud_scheme`` is the ExperimentConfig field name and a plausible wrong
    guess for the YAML key (which is ``clouds``).
    """
    with pytest.raises(ValueError, match="unknown physics key"):
        _ec({"physics": {bad_key: "bechtold"}})


def test_null_physics_block_raises_clearly():
    """``physics:`` with no children parses to None; that must produce a clear
    error, not an AttributeError from the first ``.get``."""
    with pytest.raises(ValueError, match="physics: must be a mapping"):
        _ec({"physics": None})


def test_physics_forcing_none_leaves_held_suarez_off():
    assert _ec({"physics": {"forcing": "none"}}).held_suarez_forcing is False


def test_time_integrator_reaches_dycore_config():
    """``atmosphere.time_integrator`` is declared in DEFAULT_CONFIG and written
    by the experiment wizard (wizard_core: "atmosphere.time_integrator"), but
    was never mapped — so the wizard asked the user to pick an integrator and
    silently discarded the answer."""
    ec = _ec({"atmosphere": {"time_integrator": "ssp_rk54"}})
    assert ec.dycore.time_integrator == "ssp_rk54"


def test_absent_time_integrator_stays_auto():
    """Unset must remain the grid-aware "auto" resolution — the nested default
    was a stale "ssp_rk3" that, once mapped, would have pinned every config."""
    assert _ec({}).dycore.time_integrator == "auto"
