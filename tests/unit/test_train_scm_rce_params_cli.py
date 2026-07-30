"""CLI + arm-selection tests for the SCM-RCE **gradient** training arm.

``scripts/run/train_scm_rce_params.py`` is the gradient arm of the convection
intercomparison (literature defaults / derivative-free / gradient).  It used to
be able to train only ONE convection scheme -- whichever the campaign's
``recommended_defaults.json`` named as the winner.  ``--convection`` selects the
scheme explicitly so each of the ten schemes gets its own job, and
``--subsidence-solve`` threads the shared matched-kernel selector.

These are pure-CPU argument/config tests: no CRM reference data, no SCM
integration, no optimizer step.  Importing the trainer does pull JAX (it imports
``equinox``/``optax`` and the physics package at module scope), so the platform
is pinned to CPU *before* the import -- the trainer itself does
``os.environ.setdefault("JAX_PLATFORMS", "cuda")`` at import time, which would
otherwise poison a CPU-only test session for every later test.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest

import scripts.run.run_scm_rce_campaign as campaign
import scripts.run.train_scm_rce_params as trainer

ALL_CONVECTION_SCHEMES = (
    "sbm", "dca", "kuo", "mass_flux", "edmf", "zhang_mcfarlane",
    "kain_fritsch", "emanuel", "tiedtke", "bechtold",
)

# A minimal recommended_defaults.json payload: no `recommended_physics_config`,
# just the winners block, so the tests exercise the winner-resolution path
# without touching disk or the campaign results tree.
RECOMMENDED_WINNERS = {
    "radiation": "gray",
    "turbulence": "louis",
    "microphysics": "kessler",
    "convection": "dca",
    "gravity_wave_drag": "none",
}


def _recommended(**overrides):
    winners = dict(RECOMMENDED_WINNERS)
    winners.update(overrides)
    return {"winners": winners}


# --------------------------------------------------------------------------- #
# (a) the flag itself
# --------------------------------------------------------------------------- #
def test_scheme_sweep_list_is_the_ten_campaign_schemes():
    """The flag's choices come from the campaign module, never a local copy."""
    assert tuple(campaign.SCHEME_SWEEPS["convection"]) == ALL_CONVECTION_SCHEMES


def test_convection_defaults_to_none_and_leaves_everything_else_alone():
    """Default = today's behaviour, so existing invocations are unchanged."""
    args = trainer.parse_args([])
    assert args.convection is None
    assert args.subsidence_solve == "as_shipped"
    assert args.bechtold_policy == "auto"


@pytest.mark.parametrize("scheme", ALL_CONVECTION_SCHEMES)
def test_every_convection_scheme_is_accepted(scheme):
    assert trainer.parse_args(["--convection", scheme]).convection == scheme


def test_unknown_convection_scheme_exits():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--convection", "not_a_scheme"])


# --------------------------------------------------------------------------- #
# (c) the matched-kernel flag
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", campaign.SUBSIDENCE_SOLVE_MODES)
def test_every_subsidence_solve_mode_is_accepted(mode):
    assert trainer.parse_args(["--subsidence-solve", mode]).subsidence_solve == mode


def test_unknown_subsidence_solve_mode_exits():
    with pytest.raises(SystemExit):
        trainer.parse_args(["--subsidence-solve", "implicitflux"])


# --------------------------------------------------------------------------- #
# (b) explicit selection must not be silently overwritten
# --------------------------------------------------------------------------- #
def test_mass_flux_bechtold_policy_conflicts_loudly_with_explicit_convection():
    """`--bechtold-policy mass_flux` rewrites the convection winner.  Silently
    doing that to an explicitly requested scheme would mislabel the whole arm."""
    with pytest.raises(SystemExit, match="conflicts with"):
        trainer.parse_args(
            ["--convection", "tiedtke", "--bechtold-policy", "mass_flux"]
        )


def test_mass_flux_policy_is_allowed_when_it_agrees_with_the_explicit_choice():
    args = trainer.parse_args(
        ["--convection", "mass_flux", "--bechtold-policy", "mass_flux"]
    )
    assert args.convection == "mass_flux"


def test_mass_flux_policy_still_overrides_when_no_explicit_choice():
    """Unchanged legacy behaviour for the historical invocation."""
    winners, note = trainer._apply_bechtold_policy_to_winners(
        {**RECOMMENDED_WINNERS}, "mass_flux", convection_explicit=False,
    )
    assert winners["convection"] == "mass_flux"
    assert "campaign recommended convection winner" in note


def test_note_does_not_claim_campaign_winner_when_scheme_is_explicit():
    """The log must not say the trainer 'follows the campaign winner' when the
    user picked the scheme."""
    _w, explicit_note = trainer._apply_bechtold_policy_to_winners(
        {**RECOMMENDED_WINNERS, "convection": "tiedtke"},
        "auto",
        convection_explicit=True,
    )
    assert "explicitly" in explicit_note
    assert "campaign recommended winner" not in explicit_note

    _w2, campaign_note = trainer._apply_bechtold_policy_to_winners(
        {**RECOMMENDED_WINNERS, "convection": "tiedtke"},
        "auto",
        convection_explicit=False,
    )
    assert "campaign recommended winner" in campaign_note


def test_explicit_bechtold_still_gets_the_bechtold_ad_note():
    _w, note = trainer._apply_bechtold_policy_to_winners(
        {**RECOMMENDED_WINNERS, "convection": "bechtold"},
        "train_deterministic",
        convection_explicit=True,
    )
    assert "stochastic" in note


# --------------------------------------------------------------------------- #
# winner resolution
# --------------------------------------------------------------------------- #
def test_override_wins_over_the_recommended_winner_for_convection_only():
    winners = trainer._recommended_scheme_winners(
        _recommended(), None, overrides={"convection": "emanuel"},
    )
    assert winners["convection"] == "emanuel"
    for field in ("radiation", "turbulence", "microphysics", "gravity_wave_drag"):
        assert winners[field] == RECOMMENDED_WINNERS[field]


def test_override_works_even_when_no_convection_winner_is_recorded():
    """A per-scheme gradient job must not require the campaign to have named a
    convection winner."""
    payload = {"winners": {k: v for k, v in RECOMMENDED_WINNERS.items()
                           if k != "convection"}}
    winners = trainer._recommended_scheme_winners(
        payload, None, overrides={"convection": "kain_fritsch"},
    )
    assert winners["convection"] == "kain_fritsch"


def test_missing_convection_winner_without_override_still_raises():
    payload = {"winners": {k: v for k, v in RECOMMENDED_WINNERS.items()
                           if k != "convection"}}
    with pytest.raises(ValueError, match="do not specify a winner"):
        trainer._recommended_scheme_winners(payload, None)


def test_unknown_override_field_raises():
    with pytest.raises(ValueError, match="not campaign scheme"):
        trainer._recommended_scheme_winners(
            _recommended(), None, overrides={"convecton": "dca"},
        )


# --------------------------------------------------------------------------- #
# (b)+(d) round-trip: the chosen scheme reaches the built PhysicsConfig
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme", ALL_CONVECTION_SCHEMES)
def test_chosen_scheme_reaches_the_built_physics_config(scheme):
    cfg, winners, _note, status = trainer._build_recommended_base_config(
        _recommended(),
        bechtold_policy="auto",
        convection=scheme,
    )
    assert cfg.convection.scheme == scheme
    assert winners["convection"] == scheme
    assert status == "as_shipped"
    # the other components are resolved exactly as before
    assert cfg.turbulence.scheme == RECOMMENDED_WINNERS["turbulence"]
    assert cfg.microphysics.scheme == RECOMMENDED_WINNERS["microphysics"]
    assert cfg.gravity_wave_drag.scheme == RECOMMENDED_WINNERS["gravity_wave_drag"]
    assert cfg.radiation.scheme == RECOMMENDED_WINNERS["radiation"]


def test_default_none_still_follows_the_campaign_recommended_winner():
    cfg, winners, _note, status = trainer._build_recommended_base_config(
        _recommended(convection="zhang_mcfarlane"),
        bechtold_policy="auto",
        convection=None,
    )
    assert cfg.convection.scheme == "zhang_mcfarlane"
    assert winners["convection"] == "zhang_mcfarlane"
    assert status == "as_shipped"


def test_subsidence_override_is_applied_to_the_built_config():
    cfg, _winners, _note, status = trainer._build_recommended_base_config(
        _recommended(),
        bechtold_policy="auto",
        convection="tiedtke",
        subsidence_solve="implicit_flux",
    )
    assert status == "forced:tiedtke=implicit_flux"
    assert cfg.convection.tiedtke.subsidence_solve == "implicit_flux"


def test_subsidence_override_reports_not_applicable_instead_of_silently_skipping():
    cfg, _winners, _note, status = trainer._build_recommended_base_config(
        _recommended(),
        bechtold_policy="auto",
        convection="dca",
        subsidence_solve="implicit_flux",
    )
    assert status == "not_applicable:dca"
    assert cfg.convection.scheme == "dca"


def test_unknown_subsidence_mode_raises_from_the_shared_selector():
    """Dispatch-hardening below the argparse layer (a programmatic caller must
    not silently get the as-shipped arm reported as the matched one)."""
    with pytest.raises(ValueError, match="unknown mode"):
        trainer._build_recommended_base_config(
            _recommended(),
            bechtold_policy="auto",
            convection="tiedtke",
            subsidence_solve="implicitflux",
        )


# --------------------------------------------------------------------------- #
# (e) the trainable collector follows the CHOSEN scheme
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "scheme,expected_key",
    [
        ("sbm", "atm.conv.SBMConfig"),
        ("tiedtke", "atm.conv.TiedtkeConfig"),
        ("bechtold", "atm.conv.BechtoldConfig"),
        ("emanuel", "atm.conv.EmanuelConfig"),
    ],
)
def test_active_scheme_key_follows_the_chosen_convection_scheme(scheme, expected_key):
    """``_active_subconfig_by_scheme_key`` keys off ``component.scheme``, which
    is what ``build_trainable_params(active_scheme_keys=...)`` is fed -- so the
    collected parameter set tracks ``--convection`` automatically."""
    cfg, _winners, _note, _status = trainer._build_recommended_base_config(
        _recommended(), bechtold_policy="auto", convection=scheme,
    )
    by_key = trainer._active_subconfig_by_scheme_key(cfg)
    assert expected_key in by_key
    component_name, _component, _subcfg = by_key[expected_key]
    assert component_name == "convection"
    convection_keys = [
        key for key, (comp, _c, _s) in by_key.items() if comp == "convection"
    ]
    assert convection_keys == [expected_key]
