"""A classical model trains real parameterizations, and fills every family.

Two directives from 2026-08-11, both mechanically enforced here:

1. Gray radiation is NEVER trained. It stays available as the cheap fixed
   backend, but carries no trainable knob — neither through the hand-written
   ``AIMIPClassicalParams`` set nor through the spec-driven collector. The
   nine knobs removed are the seven ``gray_*`` plus ``tau_equator`` /
   ``tau_pole``, which are gray optical depths despite their generic names.
2. A classical model has one parameterization of EVERY family (convection,
   turbulence, cloud, microphysics, radiation, gravity-wave drag). An empty
   family is a different model, not a smaller one, and it silently invalidates
   scheme-swap comparisons.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")

from legoesm.training.aimip_params import (  # noqa: E402
    AIMIP_CLASSICAL_CONSTRAINTS,
    AIMIPClassicalParams,
    CLASSICAL_SCHEME_FAMILIES,
    validate_classical_scheme_set,
)

# The nine knobs that were trainable before 2026-08-11.
_GRAY_KNOBS = frozenset({
    "gray_linear_frac", "gray_tau_moist_coeff", "gray_lw_diff_factor",
    "gray_sfc_emissivity", "gray_sw_tau_0", "gray_sw_exponent",
    "gray_sfc_albedo", "tau_equator", "tau_pole",
})


def _families(**over):
    base = dict(convection="tiedtke", turbulence="louis", cloud="xu_randall",
                microphysics="sundqvist", radiation="rrtmgp",
                gwd="mcfarlane")
    base.update(over)
    return base


# --------------------------------------------------------------------------
# 1. Gray radiation is not trainable, by either route.
# --------------------------------------------------------------------------

def test_no_gray_knob_is_in_the_constraint_table():
    names = {c.name for c in AIMIP_CLASSICAL_CONSTRAINTS}
    assert not (names & _GRAY_KNOBS)


def test_no_gray_knob_reaches_a_built_params_object():
    params = AIMIPClassicalParams.from_defaults()
    assert not (set(params.raw_values) & _GRAY_KNOBS)
    # ... and the remaining set is not empty (a vacuous pass would satisfy the
    # assertion above just as well).
    assert len(params.raw_values) > 40


def test_the_gray_config_builder_is_gone():
    """Its existence is load-bearing: ``aimip_legacy_owned_fields`` derives the
    spec collector's exclusion set by CALLING every ``to_*_config`` method, so
    a leftover gray builder would keep claiming gray fields as legacy-owned."""
    assert not hasattr(AIMIPClassicalParams, "to_gray_radiation_config")


def test_the_spec_collector_cannot_re_expose_gray():
    """Deleting the builder above removes gray from the reflection-derived
    exclusion set, so the freeze has to live in the spec itself."""
    from legoesm.atmosphere.physics.radiation import config as rad_cfg

    spec = rad_cfg.__param_spec__["GrayRadiationConfig"]
    # tier 0 = never selected: build_trainable_params takes 1 <= tier <= level.
    # The bounds stay, because the LES feedback loop promotes gray_tau_* to
    # per-column fields and clamps that diagnosis to exactly these ranges — so
    # "not trainable" must not be spelled as "no bounds".
    for name, meta in spec["params"].items():
        assert meta["tunable_tier"] == 0, f"{name} is still trainable"
        assert meta["bounds"], f"{name} lost its bounds"
    assert "sfc_emissivity" in spec["excluded"]


def test_the_trainable_collector_selects_no_gray_parameter():
    """The end-to-end check on the tier: ask the collector for the broadest
    tier and confirm gray contributes nothing."""
    from legoesm.training.param_collector import build_trainable_params

    params = build_trainable_params(
        active_scheme_keys={"atm.rad.GrayRadiationConfig"}, tier="aggressive")
    assert not params.raw_values, (
        f"gray must contribute no trainable leaf: {sorted(params.raw_values)}")

    # Control: the same call for a scheme that IS trainable must be non-empty,
    # or this test would pass on a broken collector that returns nothing.
    # (Louis, not RRTMGP: RRTMGPConfig's spec declares no trainable params —
    # its two trainable surface knobs come from the hand-written
    # AIMIPClassicalParams set and reach the solver as per-call overrides.)
    live = build_trainable_params(
        active_scheme_keys={"atm.turb.LouisConfig"}, tier="aggressive")
    assert live.raw_values


def test_rrtmgp_still_trains_exactly_its_two_surface_knobs():
    """The directive keeps radiation trainable — through RRTMGP, and only the
    surface albedo/emissivity there (plus the surface roughness knobs, which
    belong to the surface layer, not to radiation)."""
    params = AIMIPClassicalParams.from_defaults()
    rrtmgp = {k for k in params.raw_values if k.startswith("rrtmgp_")}
    assert rrtmgp == {"rrtmgp_sfc_albedo", "rrtmgp_sfc_emissivity"}
    assert "surface_z0" in params.raw_values


def test_gray_radiation_still_runs_at_its_published_defaults():
    """Not trained is not the same as not available: the gray branch must
    still build, now from the canonical config."""
    from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig

    cfg = GrayRadiationConfig()
    assert cfg.tau_equator == pytest.approx(7.2)
    assert cfg.tau_pole == pytest.approx(1.8)


# --------------------------------------------------------------------------
# 2. Every family filled.
# --------------------------------------------------------------------------

def test_a_full_scheme_set_validates_and_is_returned():
    assert validate_classical_scheme_set(**_families()) == _families()


@pytest.mark.parametrize("family", CLASSICAL_SCHEME_FAMILIES)
@pytest.mark.parametrize("unfilled", ["none", "None", "", "  ", "off"])
def test_an_unfilled_family_is_rejected(family, unfilled):
    with pytest.raises(ValueError, match="EVERY family"):
        validate_classical_scheme_set(**_families(**{family: unfilled}))


def test_the_error_names_every_unfilled_family_at_once():
    """One error per submitted job otherwise."""
    with pytest.raises(ValueError) as excinfo:
        validate_classical_scheme_set(
            **_families(cloud="none", microphysics="none", gwd=""))
    msg = str(excinfo.value)
    for family in ("cloud", "microphysics", "gwd"):
        assert family in msg


def test_the_six_families_are_the_documented_ones():
    assert set(CLASSICAL_SCHEME_FAMILIES) == {
        "convection", "turbulence", "cloud", "microphysics", "radiation",
        "gwd"}


# --------------------------------------------------------------------------
# 3. The ablation waiver reaches the place that actually raises.
# --------------------------------------------------------------------------

def test_the_waiver_reaches_the_factory_not_just_the_runner():
    """The runner validated with the waiver and then built the physics WITHOUT
    it, so the factory's own gate re-raised for exactly the ablation suites the
    runner had just cleared (codex round 3). Assert the parameter exists and
    both settings behave."""
    import inspect

    from legoesm.training.aimip_params import (
        make_aimip_classical_spectral_physics,
    )

    sig = inspect.signature(make_aimip_classical_spectral_physics)
    assert "allow_unfilled_families" in sig.parameters
    assert sig.parameters["allow_unfilled_families"].default is False

    # And the runner passes it through: the call site must name it, or the
    # ablation suites break again the next time someone edits that call.
    import pathlib as _pl
    runner = _pl.Path(
        __file__).resolve().parents[2] / "scripts" / "run" / "run_aimip.py"
    src = runner.read_text()
    assert src.count("allow_unfilled_families=_allow_unfilled") >= 1, (
        "run_aimip builds the classical physics without forwarding the waiver")


def test_the_waiver_flag_is_not_fooled_by_a_quoted_false():
    """``bool("false")`` is True: a quoted YAML flag would have waived the gate
    while reading as if it did not (codex round 3)."""
    import importlib.util
    import pathlib as _pl

    runner = _pl.Path(
        __file__).resolve().parents[2] / "scripts" / "run" / "run_aimip.py"
    spec = importlib.util.spec_from_file_location("_run_aimip_flag", runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._as_bool("false") is False
    assert mod._as_bool("False") is False
    assert mod._as_bool("") is False
    assert mod._as_bool(False) is False
    assert mod._as_bool("true") is True
    assert mod._as_bool(True) is True
