"""The coupled runner takes the SAME external ozone/solar forcing flags as
run_amip.py, so one forcing set serves every lane (AMIP on any grid and the
coupled CMIP lane).

Before 2026-10-06 run_coupled.py had none of these flags: the coupled lane
silently ran ``ExperimentConfig``'s defaults (built-in ozone profile, constant
sun) whatever the AMIP deck prescribed.  The drift guard pins defaults and
choices of the two parsers to each other so the lanes cannot diverge again.
"""
import pytest

from scripts.run.run_amip import build_arg_parser as amip_parser
from scripts.run.run_coupled import build_parser as coupled_parser

_FORCING_DESTS = (
    "ozone_source", "ozone_forcing", "ozone_file",
    "solar_source", "solar_file", "solar_tsi_var", "solar_spectral_var",
    "solar_spectral_band_order",
)


def _actions(parser):
    return {a.dest: a for a in parser._actions}


def test_forcing_flags_match_run_amip_defaults_and_choices():
    amip, coupled = _actions(amip_parser()), _actions(coupled_parser())
    # symmetric: a ninth ozone/solar flag added to either runner must be
    # added to the other (solar_s0 is run_amip's gray-radiation constant)
    amip_set = {d for d in amip if d.startswith(("ozone_", "solar_"))} - {"solar_s0"}
    coupled_set = {d for d in coupled if d.startswith(("ozone_", "solar_"))}
    assert amip_set == coupled_set == set(_FORCING_DESTS)
    for dest in _FORCING_DESTS:
        assert dest in coupled, dest
        assert coupled[dest].default == amip[dest].default, dest
        assert coupled[dest].choices == amip[dest].choices, dest


def test_forcing_flags_round_trip():
    args = coupled_parser().parse_args([
        "--ozone-forcing", "external", "--ozone-file", "/o3.nc",
        "--solar-source", "spectral_file", "--solar-file", "/sol.nc",
        "--solar-tsi-var", "TSI", "--solar-spectral-var", "SSI_frac",
        "--solar-spectral-band-order", "rrtmg_sw",
    ])
    assert (args.ozone_forcing, args.ozone_file) == ("external", "/o3.nc")
    assert (args.solar_source, args.solar_file) == ("spectral_file", "/sol.nc")
    assert (args.solar_tsi_var, args.solar_spectral_var,
            args.solar_spectral_band_order) == ("TSI", "SSI_frac", "rrtmg_sw")


def test_forcing_args_reach_the_atm_config():
    """``main`` builds the atmosphere ``ExperimentConfig`` inline; the only
    place the parsed forcing can be lost is that keyword list, so pin it on
    the source of the function that runs (``run_coupled.main``)."""
    import inspect

    from scripts.run import run_coupled
    src = inspect.getsource(run_coupled.main)
    for dest in _FORCING_DESTS:
        assert f"{dest}=args.{dest}," in src, dest


def test_external_channel_without_a_file_is_refused():
    """The ozone loader substitutes its reference profile when the path is
    empty: an external channel with no file must fail loudly, as run_amip's
    _postprocess_args does (tests/unit/test_run_amip_cli.py)."""
    from scripts.run import run_coupled
    for argv in (["--ozone-forcing", "external"],
                 ["--solar-source", "spectral_file"]):
        with pytest.raises(SystemExit):
            run_coupled.require_forcing_files(coupled_parser().parse_args(argv))
