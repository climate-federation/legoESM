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
    restrict_suite_def,
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


def _two_block_suite():
    return [
        {"name": "annual_cycle", "diagnostic": "x.AnnualCycle",
         "variables": [{"id": "tas", "var_name": "tas"}]},
        {"name": "map", "diagnostic": "x.Map",
         "variables": [{"id": "tas", "var_name": "tas"},
                       {"id": "pr", "var_name": "pr",
                        "additional_preprocessors": {"a.b": {"c": 1}}}]},
    ]


def test_restrict_suite_def_keeps_only_named_blocks():
    out = restrict_suite_def(_two_block_suite(), "s", ["map"], [])
    assert [b["name"] for b in out] == ["map"]


def test_restrict_suite_def_unknown_diagnostic_is_a_hard_error():
    import pytest
    with pytest.raises(SystemExit, match="not in suite s"):
        restrict_suite_def(_two_block_suite(), "s", ["map", "nope"], [])


def test_restrict_suite_def_months_reach_every_variable_and_keep_existing():
    out = restrict_suite_def(_two_block_suite(), "s", [], [1, 2])
    for block in out:
        for var in block["variables"]:
            pre = var["additional_preprocessors"]
            assert pre["run_amip_climateeval.extract_months_keep_time"] == {"months": [1, 2]}
    assert out[1]["variables"][1]["additional_preprocessors"]["a.b"] == {"c": 1}


def test_restrict_suite_def_noop_without_flags():
    assert restrict_suite_def(_two_block_suite(), "s", [], []) == _two_block_suite()


def test_restrict_suite_def_does_not_mutate_input():
    src = _two_block_suite()
    restrict_suite_def(src, "s", ["map"], [1])
    assert src == _two_block_suite()


def test_main_rejects_month_outside_1_12_before_any_work():
    import pytest
    from scripts.validate import run_amip_climateeval as bridge
    with pytest.raises(SystemExit) as exc:
        bridge.main(["--cmor-dir", "/nonexistent", "--data-root-dir", "/nonexistent",
                     "--output-dir", "/nonexistent", "--month", "13"])
    assert exc.value.code == 2  # argparse usage error, not a late ClimateEval failure


def test_main_routes_the_suite_through_restrict_suite_def(monkeypatch, tmp_path):
    """Regression guard for the wiring: deleting the restrict_suite_def call in
    main() must fail this test (the fake Suite records the YAML it was given)."""
    import sys
    import types
    from scripts.validate import run_amip_climateeval as bridge

    seen = {}

    class _FakeSuite:
        def __init__(self, yml, **kw):
            import yaml
            with open(yml) as f:
                seen["def"] = yaml.safe_load(f)

        def get_database(self, *a, **kw):
            raise RuntimeError("stop after suite load")

    fake_ce = types.ModuleType("climateeval")
    fake_suites = types.ModuleType("climateeval.suites"); fake_suites.Suite = _FakeSuite
    fake_data = types.ModuleType("climateeval.data")
    fake_data.DataSourceInformation = lambda **kw: None
    fake_report = types.ModuleType("climateeval.report"); fake_report.serve = None
    fake_iris = types.ModuleType("iris"); fake_iris.load = lambda paths: []
    suite_dir = tmp_path / "suites"; suite_dir.mkdir()
    (suite_dir / "toy.yml").write_text(
        "- name: annual_cycle\n  diagnostic: x\n  variables: [{id: tas, var_name: tas}]\n"
        "- name: map\n  diagnostic: y\n  variables: [{id: tas, var_name: tas}]\n")
    for name, mod in (("climateeval", fake_ce), ("climateeval.suites", fake_suites),
                      ("climateeval.data", fake_data), ("climateeval.report", fake_report),
                      ("iris", fake_iris)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(bridge, "cmor_nc_paths", lambda d: [tmp_path / "x.nc"])
    import importlib.resources as resources
    monkeypatch.setattr(resources, "files", lambda pkg: suite_dir)
    try:
        bridge.main(["--cmor-dir", str(tmp_path), "--data-root-dir", str(tmp_path),
                     "--output-dir", str(tmp_path), "--suite", "toy",
                     "--diagnostic", "map", "--month", "1"])
    except Exception:
        pass
    assert "def" in seen, "Suite was never constructed"
    assert [b["name"] for b in seen["def"]] == ["map"]
    assert "run_amip_climateeval.extract_months_keep_time" in \
        seen["def"][0]["variables"][0]["additional_preprocessors"]
