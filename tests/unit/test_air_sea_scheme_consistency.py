"""The atmosphere surface layer and the coupler OCEAN tile must agree.

``ExperimentConfig.surface_bulk_scheme`` drives the ATMOSPHERE surface layer;
``CouplerConfig.bulk_scheme`` drives the coupler OCEAN tile (the 3D-ocean
air-sea flux). They parameterize the SAME interface.

``run_coupled`` built ``CouplerConfig`` without ``bulk_scheme``, so it stayed at
its ``"constant"`` default while the atmosphere ran the requested MOST scheme:

    --surface-bulk-scheme coare3 --gustiness-zi 300  ->
       atmosphere surface layer : 'coare3'
       coupler OCEAN tile       : 'constant'     <-- split interface
       coupler gustiness_w_zi   : 300.0          (a COARE3 term, inert here)

Nothing failed; the interface just quietly stopped conserving -- the ocean's
latent-heat loss was not the atmosphere's moisture gain. Both the flag's help
("for the atmosphere surface layer + the coupler OCEAN tile") and the driver's
log line already claimed the tile used the requested scheme, and the block sets
``gustiness_w_zi`` *specifically* to make the interface energy-consistent --
which cannot hold when the tile's scheme has no gustiness term at all.

config/cmip/cmip_tuned_physics.yaml pins ``surface_bulk_scheme: coare3`` under a
"TUNED air-sea fluxes: COARE 3.0 MOST with convective gustiness" header, so that
tuning was calibrated against the split. Fixing it changes coupled answers; the
tuned config is owed a re-tune (tracked separately -- NOT silently absorbed
here).
"""
from __future__ import annotations

import importlib.util
import sys

import pytest
from legoesm.coupler.config import CouplerConfig
from legoesm.driver.air_sea_consistency import (
    resolve_effective_atm_surface,
    validate_air_sea_consistency,
)
from legoesm.driver.config import ExperimentConfig

_MOST_SCHEMES = ("coare3", "large_yeager")


def _coupled_parser():
    sys.argv = ["_rc_consistency_test"]
    spec = importlib.util.spec_from_file_location(
        "_rc_consistency_test", "scripts/run/run_coupled.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("scheme", _MOST_SCHEMES)
def test_cli_threads_the_scheme_into_the_coupler_ocean_tile(scheme):
    """The load-bearing pin: assert the config run_coupled ACTUALLY builds.

    Calls ``build_coupler_config`` -- the production function main() uses. An
    earlier draft of this test re-derived the CouplerConfig itself and stayed
    GREEN with the fix reverted, because it was asserting its own mirror.
    Parser-only coverage is what let the omission survive in the first place;
    a mirror is the same mistake one layer up.
    """
    mod = _coupled_parser()
    args = mod.build_parser().parse_args(
        ["--surface-bulk-scheme", scheme, "--gustiness-zi", "300"]
    )
    cpl = mod.build_coupler_config(args)
    assert cpl is not None
    assert cpl.bulk_scheme == scheme, (
        "the coupler ocean tile did not get the requested air-sea scheme -- "
        "the atmosphere would run MOST against a 'constant' tile"
    )
    assert cpl.gustiness_w_zi == 300.0
    # and the pair the driver guard compares must be consistent
    atm = ExperimentConfig(surface_bulk_scheme=args.surface_bulk_scheme,
                           turbulence="louis")
    validate_air_sea_consistency(atm, cpl)


def test_default_run_still_takes_the_driver_defaults():
    """``None`` keeps a default run byte-identical (the driver builds the same
    CouplerConfig), so this fix cannot perturb an untouched run."""
    mod = _coupled_parser()
    assert mod.build_coupler_config(mod.build_parser().parse_args([])) is None


@pytest.mark.parametrize("scheme", _MOST_SCHEMES + ("constant",))
def test_matched_schemes_pass_the_guard(scheme):
    atm = ExperimentConfig(surface_bulk_scheme=scheme, turbulence="louis")
    validate_air_sea_consistency(atm, CouplerConfig(bulk_scheme=scheme))


def test_split_interface_is_rejected():
    """The exact historical bug: atmosphere coare3, ocean tile constant."""
    atm = ExperimentConfig(surface_bulk_scheme="coare3", turbulence="louis")
    with pytest.raises(ValueError, match="air-sea bulk-flux scheme mismatch"):
        validate_air_sea_consistency(atm, CouplerConfig())


@pytest.mark.parametrize(
    "atm_scheme,ocean_scheme",
    [("coare3", "constant"), ("constant", "coare3"),
     ("coare3", "large_yeager"), ("large_yeager", "most")],
)
def test_any_mismatch_is_rejected(atm_scheme, ocean_scheme):
    atm = ExperimentConfig(surface_bulk_scheme=atm_scheme, turbulence="louis")
    with pytest.raises(ValueError, match="air-sea bulk-flux scheme mismatch"):
        validate_air_sea_consistency(
            atm, CouplerConfig(bulk_scheme=ocean_scheme)
        )


def test_none_coupler_config_is_validated_against_the_driver_default():
    """``coupler_config=None`` must NOT be waved through.

    An earlier version of this file asserted the opposite, on the claim that the
    driver's defaults are "self-consistent by construction". They are not:
    ``setup()`` materializes a bare ``CouplerConfig()`` whose bulk_scheme is
    "constant" regardless of the atmosphere, so this exact call produced the
    split the guard exists to catch -- and the guard skipped it.
    """
    atm = ExperimentConfig(surface_bulk_scheme="coare3", turbulence="louis")
    with pytest.raises(ValueError, match="air-sea bulk-flux scheme mismatch"):
        validate_air_sea_consistency(atm, None)


def test_none_coupler_config_ok_when_atmosphere_matches_the_default():
    """The default atmosphere ('constant') matches the driver's default tile, so
    an untouched run is unaffected."""
    validate_air_sea_consistency(ExperimentConfig(), None)


# --- the guard must be REACHED, not merely correct -------------------------
#
# Every test above calls the helper directly, so deleting the call from a
# driver's __init__ left them all green (codex). These pin the wiring itself:
# they construct the PUBLIC driver and fail if the guard is not invoked.

@pytest.mark.parametrize("driver_name", ["CoupledESMDriver", "EarthSystemDriver"])
def test_public_driver_ctor_rejects_the_split(driver_name):
    """Both drivers do `self._coupler_config or CouplerConfig()` in setup() and
    feed it to make_coupler's ocean tile, so BOTH need the guard.
    EarthSystemDriver did not have it (codex)."""
    import importlib
    mod = importlib.import_module(
        "legoesm.driver.coupled_esm_driver" if driver_name == "CoupledESMDriver"
        else "legoesm.driver.earth_system_driver"
    )
    driver_cls = getattr(mod, driver_name)
    atm = ExperimentConfig(surface_bulk_scheme="coare3", turbulence="louis")
    with pytest.raises(ValueError, match="air-sea bulk-flux scheme mismatch"):
        driver_cls(atm)


# --- effective config, not declared config ---------------------------------

def test_override_that_declares_constant_but_runs_coare3_is_rejected():
    """The declared field is NOT what the atmosphere runs.

    ``apply_surface_flux_config`` returns a ``turbulence_override`` UNCHANGED
    when every experiment-level surface field is default, and validate_strict
    only requires the override to share the active turbulence *scheme* -- never
    its nested surface settings. So this config declares 'constant', runs
    COARE3, and a guard reading the declared field waves the split through
    (codex). Verified: validate_strict accepts it.
    """
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    base = TurbulenceConfig(scheme="louis")
    ovr = base._replace(
        louis=base.louis._replace(
            surface=base.louis.surface._replace(bulk_scheme="coare3")))
    cfg = ExperimentConfig(turbulence="louis", surface_bulk_scheme="constant",
                           turbulence_override=ovr)
    cfg.validate_strict()  # the override is legal ...
    assert resolve_effective_atm_surface(cfg).bulk_scheme == "coare3"  # ... and runs COARE3
    with pytest.raises(ValueError, match="air-sea bulk-flux scheme mismatch"):
        validate_air_sea_consistency(cfg, CouplerConfig())


def test_effective_resolver_matches_the_declared_field_when_no_override():
    """Without an override the effective scheme IS the declared one, so the
    resolver cannot have loosened the common case."""
    for scheme in _MOST_SCHEMES + ("constant",):
        cfg = ExperimentConfig(surface_bulk_scheme=scheme, turbulence="louis")
        assert resolve_effective_atm_surface(cfg).bulk_scheme == scheme


def test_no_surface_sub_config_is_skipped():
    """turbulence='none' carries no surface layer -> nothing to compare.
    (coare3 + turbulence='none' is already rejected by validate_strict.)"""
    assert resolve_effective_atm_surface(ExperimentConfig(turbulence="none")) is None
    validate_air_sea_consistency(ExperimentConfig(turbulence="none"),
                                 CouplerConfig(bulk_scheme="coare3"))


# --- clubb: None sub-config means "the default one", not "no config" --------

def test_surface_bulk_scheme_reaches_clubb():
    """THE REACHABILITY BUG: --surface-bulk-scheme was silently ignored here.

    ``TurbulenceConfig.clubb`` defaults to None and dispatch substitutes a fresh
    ``CLUBBConfig()``, so ``apply_surface_flux_config``'s bail-out on the None
    sub-config dropped the injection entirely: the run used CLUBB's own default
    'constant' surface layer while the user asked for coare3, and nothing said
    so (codex). This is the same class of bug as the rest of this PR -- a
    parameterization that cannot be selected from config.
    """
    cfg = ExperimentConfig(turbulence="clubb", surface_bulk_scheme="coare3")
    cfg.validate_strict()
    surf = resolve_effective_atm_surface(cfg)
    assert surf is not None, "clubb's surface config must be materialized"
    assert surf.bulk_scheme == "coare3", (
        "--surface-bulk-scheme did not reach the CLUBB surface layer"
    )


def test_clubb_dispatch_and_injection_agree():
    """The production dispatch and the injection must materialize the SAME
    sub-config, or the guard reasons about a config the model never runs."""
    from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn

    cfg = ExperimentConfig(turbulence="clubb", surface_bulk_scheme="coare3")
    from legoesm.driver.physics_pipeline import turbulence_config_for
    _, _, sub = get_turbulence_fn(turbulence_config_for(cfg))
    assert sub.surface.bulk_scheme == "coare3"


def test_clubb_split_against_the_ocean_tile_is_caught():
    """Default clubb runs its own 'constant' surface; a COARE3 tile splits the
    interface. The guard previously skipped it (resolver returned None)."""
    cfg = ExperimentConfig(turbulence="clubb")     # default => constant surface
    assert resolve_effective_atm_surface(cfg).bulk_scheme == "constant"
    with pytest.raises(ValueError, match="air-sea bulk-flux scheme mismatch"):
        validate_air_sea_consistency(cfg, CouplerConfig(bulk_scheme="coare3"))


def test_default_clubb_config_is_untouched():
    """The all-defaults early return still returns tc UNCHANGED (same object),
    so materializing cannot perturb a default run."""
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.driver.physics_pipeline import apply_surface_flux_config

    tc = TurbulenceConfig(scheme="clubb")
    assert apply_surface_flux_config(tc, ExperimentConfig(turbulence="clubb")) is tc
    assert tc.clubb is None


# --- T/q measurement heights are not same-named across the interface -------

@pytest.mark.parametrize("axis", ["z_t_atm", "z_q_atm"])
def test_coupler_tq_height_must_match_the_atmosphere_z_ref(axis):
    """The atmosphere's MOST call OMITS z_t/z_q, so compute_most_fluxes defaults
    them to its z_ref; the coupler passes z_t_atm/z_q_atm explicitly. There is no
    same-named atm field, so a name-matching loop misses this split (codex)."""
    atm = ExperimentConfig(surface_bulk_scheme="coare3", turbulence="louis")
    with pytest.raises(ValueError, match=axis):
        validate_air_sea_consistency(
            atm, CouplerConfig(bulk_scheme="coare3", **{axis: 2.0}))


def test_tq_heights_not_compared_under_the_constant_closure():
    """The constant closure runs no MOST solve, so the heights are inert."""
    validate_air_sea_consistency(
        ExperimentConfig(turbulence="louis"),
        CouplerConfig(z_t_atm=2.0, z_q_atm=2.0))


# --- the interface splits on more than the scheme name ---------------------

@pytest.mark.parametrize("axis,atm_kw,cpl_kw", [
    ("thermo_convention", {"surface_thermo_convention": "aerobulk"},
     {"thermo_convention": "legoesm"}),
    ("stability_scheme", {"surface_stability_scheme": "sheba"},
     {"stability_scheme": "dyer1974"}),
])
def test_non_scheme_axes_also_split_the_interface(axis, atm_kw, cpl_kw):
    """bulk_scheme was one of SEVERAL same-named fields on both sides; a split
    in any of them means the two sides run different physics on one interface."""
    atm = ExperimentConfig(surface_bulk_scheme="coare3", turbulence="louis",
                           **atm_kw)
    with pytest.raises(ValueError, match=axis):
        validate_air_sea_consistency(
            atm, CouplerConfig(bulk_scheme="coare3", **cpl_kw))


@pytest.mark.parametrize("atm_kw", [
    {"surface_thermo_convention": "aerobulk"},
    {"surface_stability_scheme": "sheba"},
])
def test_most_only_axes_are_not_compared_under_the_constant_closure(atm_kw):
    """MUST NOT reject: under 'constant' BOTH sides ignore these fields.

    surface_layer.py passes thermo_convention/stability_scheme/z_ref/
    bulk_n_iter exclusively inside its ("coare3","large_yeager") branch (the
    constant branch reads only Cd_neutral/Ch_neutral), and coupler.py gates them
    behind `_is_most` while ocean_surface_q_sat deliberately keeps the constant
    closure on Tetens. So the two sides run IDENTICALLY here.

    An earlier draft of the guard compared these unconditionally and rejected
    `run_coupled --bulk-thermo-convention aerobulk` -- a real config, since
    build_coupler_config returns None under the default constant scheme. That
    would have been a regression introduced BY the guard.
    """
    cfg = ExperimentConfig(turbulence="louis", **atm_kw)   # default: constant
    validate_air_sea_consistency(cfg, None)


def test_matching_most_is_still_a_split_because_the_atmosphere_lacks_it():
    """Equal strings are not equal physics.

    The atmosphere dispatches MOST on ("coare3","large_yeager") ONLY, so
    bulk_scheme='most' silently degrades it to the constant branch -- while the
    coupler tile's `_is_most` DOES include 'most' and runs the MOST solver. The
    names match; the physics does not.
    """
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    base = TurbulenceConfig(scheme="louis")
    ovr = base._replace(louis=base.louis._replace(
        surface=base.louis.surface._replace(bulk_scheme="most")))
    cfg = ExperimentConfig(turbulence="louis", turbulence_override=ovr)
    with pytest.raises(ValueError, match="does NOT implement"):
        validate_air_sea_consistency(cfg, CouplerConfig(bulk_scheme="most"))


def test_most_geometry_split_is_rejected():
    """z_ref/bulk_n_iter feed the same MOST solver on both sides."""
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    base = TurbulenceConfig(scheme="louis")
    ovr = base._replace(louis=base.louis._replace(
        surface=base.louis.surface._replace(bulk_scheme="coare3", z_ref=2.0)))
    cfg = ExperimentConfig(turbulence="louis", turbulence_override=ovr)
    with pytest.raises(ValueError, match="z_ref"):
        validate_air_sea_consistency(cfg, CouplerConfig(bulk_scheme="coare3"))


def test_most_geometry_is_not_compared_under_the_constant_closure():
    """The constant closure reads neither z_ref nor bulk_n_iter, so a
    difference there is not a split and must NOT be rejected."""
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    base = TurbulenceConfig(scheme="louis")
    ovr = base._replace(louis=base.louis._replace(
        surface=base.louis.surface._replace(z_ref=2.0)))
    cfg = ExperimentConfig(turbulence="louis", turbulence_override=ovr)
    validate_air_sea_consistency(cfg, CouplerConfig())  # must not raise


@pytest.mark.xfail(
    reason="SEPARATE, VERIFIED pre-existing bug: CouplerConfig.gustiness_w_zi is "
           "`float = 0.0` and so CANNOT express 'scheme-native', while the "
           "atmosphere's SurfaceLayerConfig.gustiness_w_zi is `float | None = "
           "None` (-> 600 m for coare3, bulk_flux._COARE_GUSTINESS_ZI). So "
           "`--surface-bulk-scheme coare3` WITHOUT --gustiness-zi splits the "
           "interface again: atmosphere 600 m vs ocean tile 0.0 (off). Fixing it "
           "means making the coupler field nullable -- a further physics change "
           "on top of this PR's, so it is pinned here rather than silently "
           "absorbed.",
    strict=True,
)
def test_gustiness_is_consistent_when_left_scheme_native():
    mod = _coupled_parser()
    args = mod.build_parser().parse_args(["--surface-bulk-scheme", "coare3"])
    cpl = mod.build_coupler_config(args)
    # atmosphere: surface_gustiness_zi=None -> SurfaceLayerConfig keeps its own
    # None -> coare3 resolves scheme-native 600 m.
    assert args.surface_gustiness_zi is None
    # the tile should mean the same thing; today it is coerced to 0.0 == OFF.
    assert cpl.gustiness_w_zi is None or cpl.gustiness_w_zi == 600.0
