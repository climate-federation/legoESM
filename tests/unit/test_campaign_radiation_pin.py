"""WeatherBench and AIMIP always run RRTMGP radiation.

User directive 2026-08-12: *"gray radiation should not be used as one of the
parameterizations in weatherbench or AIMIP — always use RRTMGP. But do not
delete gray radiation from legoESM."* Gray stays available to idealized cases,
the SCM, DA and the land drivers; it is simply not a campaign option, and it
carries no trainable knob (see test_classical_scheme_families).

The pin used to cover the CLASSICAL variant only, and carried an
``allow_non_rrtmgp`` escape that two shipped configs used — so column_nn,
sfno_physics and sfno_full could each select gray, and classical could opt out.
Both holes are closed here.
"""
from __future__ import annotations

import ast
import pathlib

import pytest
import yaml

from legoesm.training.campaign_driver import (
    CLASSICAL_RADIATION,
    validate_campaign_radiation,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]

# Every campaign entry point that selects a radiation backend. The AST test
# below asserts each one calls the pin; the completeness test asserts nobody
# adds a sibling driver without landing here.
_ENTRY_POINTS = (
    "scripts/run/run_aimip.py",
    "scripts/run/run_aimip_latlon.py",
    "scripts/run/run_aimip_ablation.py",
    "scripts/run/run_aimip_amip_finetune.py",
)


# --------------------------------------------------------------------------
# The validator itself
# --------------------------------------------------------------------------

def test_rrtmgp_is_the_pin():
    assert CLASSICAL_RADIATION == "rrtmgp"
    assert validate_campaign_radiation("rrtmgp", campaign="aimip") == "rrtmgp"
    assert validate_campaign_radiation("rrtmgp", campaign="wb") == "rrtmgp"


@pytest.mark.parametrize("campaign", ["aimip", "wb"])
def test_gray_is_rejected_for_a_science_run(campaign):
    with pytest.raises(ValueError, match="rrtmgp"):
        validate_campaign_radiation("gray", campaign=campaign)


@pytest.mark.parametrize("campaign", ["aimip", "wb"])
def test_an_unknown_backend_is_rejected(campaign):
    with pytest.raises(ValueError):
        validate_campaign_radiation("neon", campaign=campaign)
    # ... including under smoke: cheap is not the same as anything goes.
    with pytest.raises(ValueError):
        validate_campaign_radiation("neon", campaign=campaign, smoke=True)


def test_smoke_may_still_use_gray():
    """A smoke run produces no scientific output — it checks the code is wired,
    and RRTMGP costs ~10x the compile."""
    assert validate_campaign_radiation("gray", smoke=True) == "gray"


def test_the_debug_escape_is_gone():
    """``allow_non_rrtmgp`` let two shipped configs run gray. "Always" leaves
    no room for it, so passing it is now an error rather than a bypass."""
    with pytest.raises(TypeError):
        validate_campaign_radiation("gray", allow_non_rrtmgp=True)


# --------------------------------------------------------------------------
# Every entry point is wired, and the list stays complete
# --------------------------------------------------------------------------

def _pin_calls(tree):
    return [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "validate_campaign_radiation"
    ]


@pytest.mark.parametrize("rel", _ENTRY_POINTS)
def test_entry_point_calls_the_pin(rel):
    tree = ast.parse((_REPO / rel).read_text())
    assert _pin_calls(tree), (
        f"{rel} selects a radiation backend without the campaign pin")


@pytest.mark.parametrize("rel", _ENTRY_POINTS)
def test_the_pin_is_not_nested_under_a_variant_check(rel):
    """The exact hole this had twice: the call sat inside
    ``if variant == "classical"``, so column_nn / sfno could still pick gray.
    Assert at least one call is at the function's top level, not inside an
    ``if`` (a token-presence check passes right through that bug — codex)."""
    tree = ast.parse((_REPO / rel).read_text())
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for inner in ast.walk(node):
                guarded.add(id(inner))
    unguarded = [c for c in _pin_calls(tree) if id(c) not in guarded]
    assert unguarded, (
        f"{rel} only pins radiation inside a conditional — some variant path "
        "can still select gray")


def test_no_campaign_driver_escapes_the_list():
    """A new run_aimip*/run_weatherbench* driver must be added here — and then
    the test above forces it to call the pin."""
    found = {
        p.relative_to(_REPO).as_posix()
        for pattern in ("run_aimip*.py", "run_weatherbench*.py")
        for p in (_REPO / "scripts" / "run").glob(pattern)
    }
    # Drivers that do not select radiation at all (inference replays a trained
    # checkpoint; the sweep planners only emit configs).
    exempt = {
        "scripts/run/run_aimip_amip_inference.py",
        "scripts/run/run_aimip_arch_sweep.py",
        "scripts/run/run_aimip_classical_sweep_stage1.py",
        "scripts/run/run_aimip_classical_sweep_stage2.py",
        "scripts/run/run_aimip_headtohead_t106.py",
        "scripts/run/run_aimip_latlon_sweep.py",
        "scripts/run/run_weatherbench_campaign.py",
    }
    unlisted = found - set(_ENTRY_POINTS) - exempt
    assert not unlisted, (
        f"new campaign driver(s) {sorted(unlisted)}: add to _ENTRY_POINTS (and "
        "call validate_campaign_radiation), or to `exempt` with a reason")


# --------------------------------------------------------------------------
# Configs cannot re-select gray
# --------------------------------------------------------------------------

def _campaign_yamls():
    for root in (_REPO / "config" / "aimip", _REPO / "config" / "wb"):
        for path in root.rglob("*.yaml"):
            try:
                doc = yaml.safe_load(path.read_text())
            except yaml.YAMLError:  # pragma: no cover - malformed config
                continue
            if isinstance(doc, dict):
                yield path, doc


def test_no_campaign_config_selects_gray():
    offenders = [
        f"{path.relative_to(_REPO)}:{key}"
        for path, doc in _campaign_yamls()
        for key, value in doc.items()
        if key.endswith("radiation") and value == "gray"
    ]
    assert not offenders, f"gray radiation back in a campaign config: {offenders}"


def test_no_campaign_config_carries_the_removed_escape():
    offenders = [
        str(path.relative_to(_REPO))
        for path, doc in _campaign_yamls()
        if "allow_non_rrtmgp" in doc
    ]
    assert not offenders, f"allow_non_rrtmgp is gone; still set in: {offenders}"


def test_the_config_sweep_is_not_vacuous():
    """It must actually be reading configs — otherwise both tests above pass by
    finding nothing anywhere."""
    seen = list(_campaign_yamls())
    assert len(seen) > 20
    assert any(k.endswith("radiation") for _, doc in seen for k in doc)


# --------------------------------------------------------------------------
# One classical model, both campaigns
# --------------------------------------------------------------------------

def test_wb_and_aimip_classical_are_the_same_model():
    """The owner's matrix is {WB, AIMIP} x {classical, column_nn, sfno}: the
    variant names a MODEL, so both campaigns must build the same one.

    WB's classical used to be ``TrainablePhysicsParams`` — 6 knobs, gray
    radiation, and only convection + radiation — while AIMIP's was the
    50-knob six-family ``AIMIPClassicalParams``.
    """
    from legoesm.training.aimip_params import AIMIPClassicalParams
    from legoesm.training.model_registry import build_variant

    params = build_variant("classical", nlev=8)
    assert isinstance(params, AIMIPClassicalParams)
    assert len(params.raw_values) == 50


def test_the_wb_campaign_config_names_every_family():
    """Defaulted families are how two runs end up differing in more than the
    variable under test."""
    import yaml as _yaml

    import inspect

    from legoesm.training.aimip_params import (
        make_aimip_classical_spectral_physics,
    )

    defaults = {
        name: param.default
        for name, param in inspect.signature(
            make_aimip_classical_spectral_physics).parameters.items()
    }
    for path in ("spectral_t63.yaml", "spectral_t106.yaml", "spectral_smoke.yaml"):
        cfg = _yaml.safe_load(
            (_REPO / "config" / "wb" / "campaign" / path).read_text())
        classical = cfg["classical"]
        for family, arg in (("convection", "convection_scheme"),
                            ("turbulence", "turbulence_scheme"),
                            ("cloud", "cloud_scheme"),
                            ("microphysics", "microphysics_scheme"),
                            ("gwd", "gwd_scheme")):
            assert classical[family] not in (None, "", "none"), (path, family)
            # ... and it must be the scheme the AIMIP factory would pick, or
            # "the same model" is only true of the container class. A test that
            # merely asserted non-empty would pass with WB on edmf and AIMIP on
            # tiedtke (Claude review).
            assert classical[family] == defaults[arg], (path, family)
        assert cfg.get("radiation", "rrtmgp") == "rrtmgp"
