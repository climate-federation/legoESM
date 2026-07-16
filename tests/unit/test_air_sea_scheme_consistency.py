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
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.coupled_esm_driver import _validate_air_sea_scheme_consistency

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
    _validate_air_sea_scheme_consistency(atm, cpl)


def test_default_run_still_takes_the_driver_defaults():
    """``None`` keeps a default run byte-identical (the driver builds the same
    CouplerConfig), so this fix cannot perturb an untouched run."""
    mod = _coupled_parser()
    assert mod.build_coupler_config(mod.build_parser().parse_args([])) is None


@pytest.mark.parametrize("scheme", _MOST_SCHEMES + ("constant",))
def test_matched_schemes_pass_the_guard(scheme):
    atm = ExperimentConfig(surface_bulk_scheme=scheme, turbulence="louis")
    _validate_air_sea_scheme_consistency(atm, CouplerConfig(bulk_scheme=scheme))


def test_split_interface_is_rejected():
    """The exact historical bug: atmosphere coare3, ocean tile constant."""
    atm = ExperimentConfig(surface_bulk_scheme="coare3", turbulence="louis")
    with pytest.raises(ValueError, match="air-sea bulk-flux scheme mismatch"):
        _validate_air_sea_scheme_consistency(atm, CouplerConfig())


@pytest.mark.parametrize(
    "atm_scheme,ocean_scheme",
    [("coare3", "constant"), ("constant", "coare3"),
     ("coare3", "large_yeager"), ("large_yeager", "most")],
)
def test_any_mismatch_is_rejected(atm_scheme, ocean_scheme):
    atm = ExperimentConfig(surface_bulk_scheme=atm_scheme, turbulence="louis")
    with pytest.raises(ValueError, match="air-sea bulk-flux scheme mismatch"):
        _validate_air_sea_scheme_consistency(
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
        _validate_air_sea_scheme_consistency(atm, None)


def test_none_coupler_config_ok_when_atmosphere_matches_the_default():
    """The default atmosphere ('constant') matches the driver's default tile, so
    an untouched run is unaffected."""
    _validate_air_sea_scheme_consistency(ExperimentConfig(), None)


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
