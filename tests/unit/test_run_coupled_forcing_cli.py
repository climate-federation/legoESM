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
    "ghg_forcing", "ghg_file",
    "aerosol_forcing", "aerosol_file", "aerosol_ccn_file", "aerosol_reference_aod",
    "volcanic_aerosol_file", "volcanic_aerosol_scale", "volcanic_aerosol_lw",
)
# run_amip-only: solar_s0 (gray-radiation constant), aerosol_ccn (a physics
# switch, not a forcing input).
_AMIP_ONLY = {"solar_s0", "aerosol_ccn"}
_PREFIXES = ("ozone_", "solar_", "ghg_", "aerosol_", "volcanic_")


def _actions(parser):
    return {a.dest: a for a in parser._actions}


def test_forcing_flags_match_run_amip_defaults_and_choices():
    amip, coupled = _actions(amip_parser()), _actions(coupled_parser())
    # symmetric: a ninth ozone/solar flag added to either runner must be
    # added to the other (solar_s0 is run_amip's gray-radiation constant)
    amip_set = {d for d in amip if d.startswith(_PREFIXES)} - _AMIP_ONLY
    coupled_set = {d for d in coupled if d.startswith(_PREFIXES)}
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
                 ["--solar-source", "spectral_file"],
                 ["--ghg-forcing", "external"],
                 ["--aerosol-forcing", "external"]):
        with pytest.raises(SystemExit):
            run_coupled.require_forcing_files(coupled_parser().parse_args(argv))


# Terrain / land-sea mask / subgrid orography (F36, review 2026-10-10): the
# coupled atmosphere was always flat and the slab deck's land analytic because
# run_coupled had none of these.  Same dests/defaults as run_amip, and each
# reaches the ExperimentConfig built in run_coupled.main.
_TERRAIN = {   # run_coupled/run_amip dest -> ExperimentConfig field
    "topography": "topography",
    "topo_smoothing": "topo_smoothing",
    "land_mask_file": "land_mask_path",
    "subgrid_orography_file": "subgrid_orography_path",
}


def test_terrain_flags_match_run_amip_and_reach_the_atm_config():
    import inspect

    from scripts.run import run_coupled
    amip, coupled = _actions(amip_parser()), _actions(coupled_parser())
    for dest in _TERRAIN:
        assert dest in coupled, dest
        assert coupled[dest].default == amip[dest].default, dest
        assert coupled[dest].type == amip[dest].type, dest
    src = inspect.getsource(run_coupled.main)
    for dest, field in _TERRAIN.items():
        assert f"{field}=args.{dest}," in src, dest
    # the vertical coordinate that must hold the terrain (#1029)
    for dest in ("vertical_coord", "transition_exponent"):
        assert coupled[dest].default == amip[dest].default, dest
        assert coupled[dest].choices == amip[dest].choices, dest
    assert "vertical_coord=args.vertical_coord," in src
    assert "transition_exponent=(args.transition_exponent" in src


def test_cmip_decks_select_the_forcing_schemes():
    """F40: the canonical coupled decks carry the forcing SCHEME keys of the
    production AMIP deck; only file paths come from the launcher."""
    from pathlib import Path

    from legoesm.driver.run_config_yaml import read_yaml_with_includes
    root = Path(__file__).resolve().parents[2]
    amip = read_yaml_with_includes(root / "config/amip/amip_production.yaml")
    keys = ("ozone_forcing", "solar_source", "solar_tsi_var",
            "solar_spectral_var", "solar_spectral_band_order", "ghg_forcing",
            "aerosol_forcing", "diurnal_cycle", "orbital_insolation",
            "rrtmgp_overhead_layer")
    for deck in ("cmip_ocean_slab.yaml", "cmip_ocean_3D.yaml"):
        cfg = read_yaml_with_includes(root / "config/cmip" / deck)
        for k in keys:
            assert cfg.get(k) == amip.get(k), (deck, k)
