"""Direct unit tests for scripts/validate/run_amip_climateeval.py.

Only the pure (non-iris/climateeval) logic is exercised here: the
legoESM/JAX test environment does not have ClimateEval installed (it
lives in a separate, dedicated environment — see
``legoesm.driver.climateeval_hook``), so ``main()``'s deferred
iris/climateeval imports are out of scope for this suite.
"""

from __future__ import annotations

import copy
from pathlib import Path

from scripts.validate.run_amip_climateeval import (
    build_arg_parser,
    cmor_nc_paths,
    obs_only_suite_def,
    suite_db_path,
)

# Mirrors the real ClimateEval suite schema: a list of diagnostic blocks,
# each with a `variables` list of per-variable settings dicts (see
# climateeval/suites/Tier2_atmosphere_monthly.yml).
_SAMPLE_SUITE_DEF = [
    {
        "name": "map",
        "diagnostic": "climateeval.diags.simple.Map",
        "variables": [
            {
                "id": "pr",
                "reference_data": "climateeval.data.GPCP",
                "other_data": ["climateeval.data.CMIP6HistoricalR1I1P1F1"],
            },
            {
                "id": "tas",
                "reference_data": "climateeval.data.HadCRUT5",
            },
        ],
    },
    "not_a_dict_entry",
]


def test_obs_only_suite_def_keeps_per_variable_reference_by_default():
    """The default must NOT rewrite references.

    Forcing one reference silently drops every variable that dataset does not
    carry — the local ERA5 tree has no rsut/rlut/rsutcs, so the whole TOA
    radiation budget went unscored.
    """
    suite_def = copy.deepcopy(_SAMPLE_SUITE_DEF)
    out = obs_only_suite_def(suite_def)
    variables = {v["id"]: v for v in out[0]["variables"]}
    assert variables["pr"]["reference_data"] == "climateeval.data.GPCP"
    assert variables["tas"]["reference_data"] == "climateeval.data.HadCRUT5"


def test_obs_only_suite_def_force_reference_overrides_every_variable():
    suite_def = copy.deepcopy(_SAMPLE_SUITE_DEF)
    out = obs_only_suite_def(
        suite_def, force_reference="climateeval.data.ERA5Monthly"
    )
    variables = {v["id"]: v for v in out[0]["variables"]}
    assert variables["pr"]["reference_data"] == "climateeval.data.ERA5Monthly"
    assert variables["tas"]["reference_data"] == "climateeval.data.ERA5Monthly"


def test_obs_only_suite_def_strips_other_data():
    suite_def = copy.deepcopy(_SAMPLE_SUITE_DEF)
    out = obs_only_suite_def(suite_def)
    variables = {v["id"]: v for v in out[0]["variables"]}
    assert "other_data" not in variables["pr"]


def test_obs_only_suite_def_tolerates_non_dict_entries():
    suite_def = copy.deepcopy(_SAMPLE_SUITE_DEF)
    # Must not raise on the trailing plain-string list entry.
    out = obs_only_suite_def(suite_def)
    assert out[1] == "not_a_dict_entry"


def test_obs_only_suite_def_no_reference_data_key_untouched():
    suite_def = [{"name": "map", "variables": [{"id": "clt", "other_data": ["x"]}]}]
    out = obs_only_suite_def(suite_def)
    clt_settings = out[0]["variables"][0]
    assert "reference_data" not in clt_settings
    assert "other_data" not in clt_settings


def test_obs_only_suite_def_shared_alias_list_is_mutated_once():
    """YAML anchors/aliases resolve to the SAME list object across
    diagnostics; mutating it once must be visible from every diagnostic
    that references it (this is what makes map/zonal_line/etc. all pick
    up the ERA5-only override from a single &2d_variables anchor)."""
    shared_variables = [{"id": "tas", "reference_data": "climateeval.data.HadCRUT5",
                         "other_data": ["climateeval.data.CMIP6HistoricalR1I1P1F1"]}]
    suite_def = [
        {"name": "map", "variables": shared_variables},
        {"name": "zonal_line", "variables": shared_variables},
    ]
    out = obs_only_suite_def(
        suite_def, force_reference="climateeval.data.ERA5Monthly"
    )
    assert out[0]["variables"] is out[1]["variables"]
    assert out[1]["variables"][0]["reference_data"] == "climateeval.data.ERA5Monthly"
    assert "other_data" not in out[1]["variables"][0]


def test_build_arg_parser_defaults():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--cmor-dir", "/tmp/cmor",
        "--data-root-dir", "/tmp/climateeval_data",
        "--output-dir", "/tmp/run",
    ])
    assert args.cmor_dir == "/tmp/cmor"
    # default (unset) = empty -> main() discovers + runs ALL bundled suites
    assert args.suites == []
    assert args.model_id == "legoESM-1-0"
    assert args.report_name == "climateeval_report.html"
    assert args.fail_on_missing_data is False
    assert args.download_missing_data is False


def test_cmor_nc_paths(tmp_path):
    cmor = tmp_path / "cmor"
    # monthly tables + fx populated; day/ present but must be EXCLUDED
    for table, fname in [("Amon", "tas.nc"), ("Omon", "tos.nc"),
                         ("SImon", "siconc.nc"), ("fx", "areacella.nc"),
                         ("day", "tas_day.nc")]:
        d = cmor / table
        d.mkdir(parents=True)
        (d / fname).write_text("")
    got = {p.parent.name for p in cmor_nc_paths(cmor)}
    assert got == {"Amon", "Omon", "SImon", "fx"}          # day excluded
    # pointed directly at a table dir -> loads that table's files (back-compat)
    assert [p.name for p in cmor_nc_paths(cmor / "Amon")] == ["tas.nc"]


def test_build_arg_parser_multiple_suites_flow_through():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--cmor-dir", "/tmp/cmor/Amon",
        "--suite", "Tier1_sanity_checks", "Tier2_atmosphere_monthly",
        "--model-id", "legoESM-1-0-era5",
        "--experiment-id", "amip",
        "--variant-id", "r1i1p1f1",
        "--data-root-dir", "/tmp/climateeval_data",
        "--timerange", "19790101/19791231",
        "--fail-on-missing-data",
        "--download-missing-data",
        "--output-dir", "/tmp/run",
    ])
    assert args.suites == ["Tier1_sanity_checks", "Tier2_atmosphere_monthly"]
    assert args.timerange == "19790101/19791231"
    assert args.fail_on_missing_data is True
    assert args.download_missing_data is True


def test_suite_db_path():
    out = suite_db_path(Path("/tmp/run"), "Tier2_atmosphere_monthly")
    assert out == Path("/tmp/run/climateeval_Tier2_atmosphere_monthly.ddb")
