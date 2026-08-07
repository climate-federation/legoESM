"""Direct unit tests for scripts/validate/run_amip_climateeval.py.

Only the pure (non-iris/climateeval) logic is exercised here: the
legoESM/JAX test environment does not have ClimateEval installed (it
lives in a separate, dedicated environment — see
``legoesm.driver.climateeval_hook``), so ``main()``'s deferred
iris/climateeval imports are out of scope for this suite.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

from scripts.validate.run_amip_climateeval import (
    ERA5_MONTHLY,
    ERA5_SUBMONTHLY,
    CheckOutcome,
    apply_reference_fallback,
    block_frequencies,
    build_arg_parser,
    cmor_nc_paths,
    cmor_tables_for,
    complex_diagnostics,
    era5_fallback_for,
    group_blocks_by_frequency,
    is_group_unscoreable,
    iter_variable_settings,
    suite_db_path,
    suite_skip_reason,
    write_suite_yaml,
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


# The staged reference data this deployment actually has (verified on disk):
# ERA5 has NO rsut/rlut/hfls/hfss/clwvi; CERES-EBAF has the TOA fluxes and
# MERRA2 the surface fluxes. This is exactly the situation in which the old
# blanket ERA5 rewrite destroyed metrics.
_STAGED = {
    ("climateeval.data.ERA5Monthly", "mon"): {"pr", "tas", "clt", "clivi", "prw"},
    ("climateeval.data.ERA5Hourly", "1hr"): {"pr"},
    ("climateeval.data.CERESEBAF", "mon"): {"rsut", "rlut"},
    ("climateeval.data.MERRA2", "mon"): {"hfls", "hfss"},
    ("climateeval.data.GPCP", "mon"): {"pr"},
    ("climateeval.data.HadCRUT5", "mon"): {"tas"},
    # ESACCICloud is NOT staged -> every lookup misses.
}


def _fake_is_available(source: str, var_name: str, frequency: str) -> bool:
    return var_name in _STAGED.get((source, frequency), set())


def _by_id(suite_def):
    return {
        v["id"]: v for block in suite_def
        if isinstance(block, dict) for v in block.get("variables", [])
    }


_TIER2_LIKE_SUITE_DEF = [
    {
        "name": "map",
        "diagnostic": "climateeval.diags.simple.Map",
        "variables": [
            {"id": "rsut", "var_name": "rsut", "frequency": "mon",
             "reference_data": "climateeval.data.CERESEBAF",
             "other_data": ["climateeval.data.CMIP6HistoricalR1I1P1F1"]},
            {"id": "hfls", "var_name": "hfls", "frequency": "mon",
             "reference_data": "climateeval.data.MERRA2"},
            {"id": "clt", "var_name": "clt", "frequency": "mon",
             "reference_data": "climateeval.data.ESACCICloud"},
            {"id": "clwvi", "var_name": "clwvi", "frequency": "mon",
             "reference_data": "climateeval.data.ESACCICloud"},
            {"id": "pr", "var_name": "pr", "frequency": "mon",
             "reference_data": "climateeval.data.GPCP"},
        ],
    },
]


def test_reference_fallback_keeps_available_designated_references():
    """THE core regression gate for the deleted TOA/surface-flux metrics.

    rsut must stay on CERES-EBAF and hfls on MERRA2 -- the old blanket
    rewrite sent both to ERA5, which has neither variable, so all four
    diagnostics scored nothing for them.
    """
    suite_def, _ = apply_reference_fallback(
        copy.deepcopy(_TIER2_LIKE_SUITE_DEF), _fake_is_available,
    )
    variables = _by_id(suite_def)
    assert variables["rsut"]["reference_data"] == "climateeval.data.CERESEBAF"
    assert variables["hfls"]["reference_data"] == "climateeval.data.MERRA2"
    assert variables["pr"]["reference_data"] == "climateeval.data.GPCP"


def test_reference_fallback_uses_era5_only_when_designated_is_missing():
    """ESACCI-CLOUD is not staged; clt exists in ERA5 -> fall back."""
    suite_def, notes = apply_reference_fallback(
        copy.deepcopy(_TIER2_LIKE_SUITE_DEF), _fake_is_available,
    )
    assert _by_id(suite_def)["clt"]["reference_data"] == ERA5_MONTHLY
    assert any("clt" in n and "fallback" in n for n in notes)


def test_reference_fallback_leaves_designated_when_nothing_has_it():
    """clwvi is in neither ESACCI-CLOUD (unstaged) nor ERA5.

    The designated reference is kept so ClimateEval reports honest missing
    data for that ONE variable, rather than scoring it against a reference
    that does not contain the quantity.
    """
    suite_def, notes = apply_reference_fallback(
        copy.deepcopy(_TIER2_LIKE_SUITE_DEF), _fake_is_available,
    )
    assert _by_id(suite_def)["clwvi"]["reference_data"] == "climateeval.data.ESACCICloud"
    assert any("clwvi" in n and "NO reference available" in n for n in notes)


def test_reference_fallback_era5_only_mode_reproduces_old_behaviour():
    suite_def, _ = apply_reference_fallback(
        copy.deepcopy(_TIER2_LIKE_SUITE_DEF), _fake_is_available, era5_only=True,
    )
    variables = _by_id(suite_def)
    assert variables["pr"]["reference_data"] == ERA5_MONTHLY
    assert variables["clt"]["reference_data"] == ERA5_MONTHLY
    # ...and the variables ERA5 lacks are left designated, i.e. unscored --
    # which is precisely the metric loss the default policy now avoids.
    assert variables["rsut"]["reference_data"] == "climateeval.data.CERESEBAF"


def test_reference_fallback_strips_other_data_by_default():
    suite_def, _ = apply_reference_fallback(
        copy.deepcopy(_TIER2_LIKE_SUITE_DEF), _fake_is_available,
    )
    assert "other_data" not in _by_id(suite_def)["rsut"]


def test_reference_fallback_can_keep_other_data():
    suite_def, _ = apply_reference_fallback(
        copy.deepcopy(_TIER2_LIKE_SUITE_DEF), _fake_is_available,
        strip_other_data=False,
    )
    assert _by_id(suite_def)["rsut"]["other_data"] == [
        "climateeval.data.CMIP6HistoricalR1I1P1F1",
    ]


def test_reference_fallback_tolerates_non_dict_entries():
    suite_def = copy.deepcopy(_SAMPLE_SUITE_DEF)
    out, _ = apply_reference_fallback(suite_def, _fake_is_available)
    assert out[1] == "not_a_dict_entry"


def test_reference_fallback_no_reference_data_key_untouched():
    suite_def = [{"name": "map", "variables": [{"id": "clt", "other_data": ["x"]}]}]
    out, notes = apply_reference_fallback(suite_def, _fake_is_available)
    clt_settings = out[0]["variables"][0]
    assert "reference_data" not in clt_settings
    assert "other_data" not in clt_settings
    assert notes == []


def test_reference_fallback_sub_monthly_uses_hourly_era5():
    suite_def = [
        {"name": "diurnal_cycle", "variables": [
            {"id": "pr", "var_name": "pr", "frequency": "1hr",
             "reference_data": "climateeval.data.ESACCICloud"},
        ]},
    ]
    out, _ = apply_reference_fallback(suite_def, _fake_is_available)
    assert out[0]["variables"][0]["reference_data"] == ERA5_SUBMONTHLY


def test_era5_fallback_for_frequency():
    assert era5_fallback_for("mon") == ERA5_MONTHLY
    for freq in ("day", "1hr", "3hr"):
        assert era5_fallback_for(freq) == ERA5_SUBMONTHLY


def test_iter_variable_settings_yields_shared_alias_dict_once():
    """YAML anchors/aliases resolve to the SAME objects across diagnostics.

    Mutating a shared settings dict once must be visible from every
    diagnostic referencing it (that is what makes map/zonal_line/etc. all
    pick up the resolution from a single &2d_variables anchor), and the
    iterator must not yield it twice or the decision log double-counts.
    """
    shared_variables = [{"id": "tas", "var_name": "tas", "frequency": "mon",
                         "reference_data": "climateeval.data.ESACCICloud"}]
    suite_def = [
        {"name": "map", "variables": shared_variables},
        {"name": "zonal_line", "variables": shared_variables},
    ]
    assert len(list(iter_variable_settings(suite_def))) == 1

    out, notes = apply_reference_fallback(suite_def, _fake_is_available)
    assert out[0]["variables"] is out[1]["variables"]
    assert out[1]["variables"][0]["reference_data"] == ERA5_MONTHLY
    assert len(notes) == 1


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
    assert args.fail_on_metric_error is False
    assert args.download_missing_data is False


def _make_cmor_tree(tmp_path):
    cmor = tmp_path / "cmor"
    for table, fname in [("Amon", "tas.nc"), ("Omon", "tos.nc"),
                         ("SImon", "siconc.nc"), ("fx", "areacella.nc"),
                         ("day", "tas_day.nc")]:
        d = cmor / table
        d.mkdir(parents=True)
        (d / fname).write_text("")
    return cmor


def test_cmor_nc_paths_monthly_excludes_day(tmp_path):
    """A monthly suite must not also get the daily cube.

    ClimateEval selects model cubes by var_name alone, so a `day` tas
    alongside the `Amon` tas makes extract_cube raise.
    """
    cmor = _make_cmor_tree(tmp_path)
    got = {p.parent.name for p in cmor_nc_paths(cmor, ["mon"])}
    assert got == {"Amon", "Omon", "SImon", "fx"}          # day excluded
    # pointed directly at a table dir -> loads that table's files (back-compat)
    assert [p.name for p in cmor_nc_paths(cmor / "Amon")] == ["tas.nc"]


def test_cmor_nc_paths_daily_loads_day_table(tmp_path):
    """THE regression gate for the frequency-blind loader.

    A `day` suite (Tier3_dynamics' Hovmoller) previously got the MONTHLY
    tables because only monthly tables were ever loaded, so it silently
    scored monthly means as if they were daily.
    """
    cmor = _make_cmor_tree(tmp_path)
    got = {p.parent.name for p in cmor_nc_paths(cmor, ["day"])}
    assert got == {"day", "fx"}
    assert "Amon" not in got


def test_cmor_nc_paths_absent_frequency_yields_nothing(tmp_path):
    """No 1hr table written -> no paths, so the caller can skip honestly.

    This is what stops a sub-daily diagnostic being served monthly data
    and producing a plausible-but-wrong number.
    """
    cmor = _make_cmor_tree(tmp_path)
    assert cmor_nc_paths(cmor, ["1hr"]) == []


def test_cmor_tables_for_always_includes_fixed_fields():
    for frequency in ("mon", "day", "1hr"):
        assert "fx" in cmor_tables_for([frequency])


def test_block_frequencies_reads_variable_frequency():
    block = {"name": "diurnal_cycle", "variables": [
        {"id": "pr", "frequency": "1hr"},
    ]}
    assert block_frequencies(block) == frozenset({"1hr"})
    # missing `frequency` defaults to monthly, matching ClimateEval
    assert block_frequencies({"name": "x", "variables": [{"id": "pr"}]}) == frozenset(
        {"mon"},
    )
    # a complex diagnostic declares no variables at all
    assert block_frequencies({"name": "ecs"}) == frozenset({"mon"})


def test_group_blocks_by_frequency_splits_mixed_suite():
    """Tier3_dynamics mixes a DAILY Hovmoller with a MONTHLY QBO.

    They must run as two groups: one flat CubeList cannot hold both a
    daily and a monthly `pr` without extract_cube raising.
    """
    suite_def = [
        {"name": "Hovmoller", "variables": [
            {"id": "pr", "frequency": "day"},
            {"id": "rlut", "frequency": "day"},
        ]},
        {"name": "QBO", "variables": [{"id": "ua", "frequency": "mon"}]},
    ]
    groups = group_blocks_by_frequency(suite_def)
    assert len(groups) == 2
    by_freq = {freqs: [b["name"] for b in blocks] for freqs, blocks in groups}
    assert by_freq[frozenset({"day"})] == ["Hovmoller"]
    assert by_freq[frozenset({"mon"})] == ["QBO"]


def test_group_blocks_by_frequency_keeps_single_frequency_suite_whole():
    suite_def = [
        {"name": "map", "variables": [{"id": "pr", "frequency": "mon"}]},
        {"name": "zonal_line", "variables": [{"id": "tas", "frequency": "mon"}]},
    ]
    groups = group_blocks_by_frequency(suite_def)
    assert len(groups) == 1
    assert [b["name"] for b in groups[0][1]] == ["map", "zonal_line"]


# --- Defect 4: complex diagnostics ----------------------------------------
_ECS_SUITE_DEF = [
    {"name": "ecs", "diagnostic": "climateeval.diags.complex.ECS",
     "additional_diagnostic_kwargs": {}},
]


def test_complex_diagnostics_detected_from_class_path():
    assert complex_diagnostics(_ECS_SUITE_DEF) == ["ecs"]
    assert complex_diagnostics(_TIER2_LIKE_SUITE_DEF) == []


def test_suite_skip_reason_skips_ecs_for_amip_with_clear_reason():
    reason = suite_skip_reason("Tier3_ecs", "amip", _ECS_SUITE_DEF)
    assert reason is not None
    assert "ecs" in reason
    assert "abrupt-4xCO2" in reason and "piControl" in reason
    # experiment ids are suffixed in practice (e.g. amip_cc_on)
    assert suite_skip_reason("Tier3_ecs", "amip_cc_on", _ECS_SUITE_DEF) is not None


def test_suite_skip_reason_non_amip_names_the_wrapper_limitation():
    reason = suite_skip_reason("Tier3_ecs", "abrupt-4xCO2", _ECS_SUITE_DEF)
    assert reason is not None
    assert "data dictionary" in reason


def test_suite_skip_reason_none_for_ordinary_suites():
    assert suite_skip_reason("Tier2_atmosphere_monthly", "amip",
                             _TIER2_LIKE_SUITE_DEF) is None


# --- Skipping frequency groups that cannot produce a single metric ---------
# Measured on the real tree: Tier2_ocean_monthly scores 0/6 required model
# variables on an atmosphere-only AMIP run, yet still spent ~4 h of a 4 h 25 m
# evaluation loading ORAS5 references for variables the model never wrote.


def test_group_unscoreable_when_every_model_variable_fails():
    """The measured Tier2_ocean_monthly case: 0/6 usable -> no metric possible."""
    statuses = {
        "amoc": "fail", "chl": "fail", "mlotst": "fail",
        "phcint": "fail", "so": "fail", "tos": "fail",
    }
    assert is_group_unscoreable(statuses) is True


def test_group_scoreable_when_any_model_variable_is_usable():
    """The measured Tier3_dynamics daily case: 2/3 usable -> must still run.

    `pr` and `rlut` are usable even though `tos` is missing, so a metric
    remains conceivable and the group must not be skipped.
    """
    assert is_group_unscoreable({"pr": "pass", "rlut": "pass", "tos": "fail"}) is False
    # a `warn` is still usable
    assert is_group_unscoreable({"pr": "warn", "tos": "fail"}) is False


def test_group_never_skipped_on_ignorance():
    """An EMPTY status map means "no information", never "nothing usable".

    This is the fail-open path taken when `climateeval check` itself raises
    (check_model_input returns CheckOutcome(passed=True, statuses={})), and
    when a suite requires no model input at all. Skipping there would
    silently discard scoreable work — the exact class of defect this whole
    wrapper audit was about.
    """
    assert is_group_unscoreable({}) is False


def test_unscoreable_decision_ignores_reference_availability():
    """A reference the availability probe cannot see must NEVER cause a skip.

    ORAS5 and RAPID override `get_cube` and read from outside
    --data-root-dir, so `make_availability_checker` reports them
    unavailable even though ClimateEval can load them. The skip decision is
    therefore taken from MODEL variable statuses ONLY: `is_group_unscoreable`
    accepts no reference argument at all, and a group whose model data is
    usable stays scheduled no matter what the reference probe believes.
    """
    # Every reference is invisible to the probe...
    assert all(
        not _fake_is_available(src, "mlotst", "mon")
        for src in ("climateeval.data.ORAS5", "climateeval.data.RAPID")
    )
    # ...yet a group with usable MODEL data is still run.
    assert is_group_unscoreable({"mlotst": "pass"}) is False
    # And the decision function has no way to consult the probe: its only
    # input is the model-variable status mapping.
    assert list(
        inspect.signature(is_group_unscoreable).parameters,
    ) == ["variable_statuses"]


def test_check_outcome_carries_statuses_for_the_skip_decision():
    outcome = CheckOutcome(passed=False, statuses={"tos": "fail"})
    assert outcome.passed is False
    assert is_group_unscoreable(outcome.statuses) is True


def test_run_unscoreable_groups_flag_defaults_to_skipping():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--cmor-dir", "/tmp/cmor",
        "--data-root-dir", "/tmp/climateeval_data",
        "--output-dir", "/tmp/run",
    ])
    assert args.run_unscoreable_groups is False
    args = parser.parse_args([
        "--cmor-dir", "/tmp/cmor",
        "--data-root-dir", "/tmp/climateeval_data",
        "--output-dir", "/tmp/run",
        "--run-unscoreable-groups",
    ])
    assert args.run_unscoreable_groups is True


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
        "--fail-on-metric-error",
        "--download-missing-data",
        "--output-dir", "/tmp/run",
    ])
    assert args.suites == ["Tier1_sanity_checks", "Tier2_atmosphere_monthly"]
    assert args.timerange == "19790101/19791231"
    assert args.fail_on_missing_data is True
    assert args.fail_on_metric_error is True
    assert args.download_missing_data is True


def test_suite_db_path():
    out = suite_db_path(Path("/tmp/run"), "Tier2_atmosphere_monthly")
    assert out == Path("/tmp/run/climateeval_Tier2_atmosphere_monthly.ddb")


# --- YAML round-trip must preserve preprocessor ORDER -----------------------
# ClimateEval executes `additional_preprocessors` in mapping insertion order
# (climateeval/_utils.py::_run_additional_preprocessors), so key order IS the
# pipeline order. PyYAML's yaml.dump default (sort_keys=True) alphabetises it.
# Verbatim from climateeval/suites/Tier1_consistency_checks.yml:
_TIER1_PREPROCESSOR_CHAIN = [
    "esmvalcore.preprocessor.regrid",
    "esmvalcore.preprocessor.area_statistics",
    "esmvalcore.preprocessor.annual_statistics",
    "esmvalcore.preprocessor.regrid_time",
    "esmvalcore.preprocessor.anomalies",
]
_CONSERVATION_SUITE_DEF = [
    {
        "name": "timeseries",
        "diagnostic": "climateeval.diags.simple.TimeSeriesNoPreprocessing",
        "variables": [
            {
                "id": "air_mass_anomaly",
                "var_name": "ps",
                "frequency": "mon",
                "additional_preprocessors": {
                    "esmvalcore.preprocessor.regrid": {
                        "target_grid": "2x2",
                        "scheme": "area_weighted",
                    },
                    "esmvalcore.preprocessor.area_statistics": {"operator": "sum"},
                    "esmvalcore.preprocessor.annual_statistics": {"operator": "sum"},
                    "esmvalcore.preprocessor.regrid_time": {"frequency": "yr"},
                    "esmvalcore.preprocessor.anomalies": {
                        "period": "full",
                        "relative": True,
                    },
                },
                "reference_data": "climateeval.data.ERA5Monthly",
            },
        ],
    },
]


def test_write_suite_yaml_preserves_preprocessor_order(tmp_path):
    """The Tier-1 conservation chain must survive the YAML round-trip.

    This is the regression gate for the defect that zeroed out the entire
    Tier-1 conservation suite: alphabetising moves `area_statistics`
    (which collapses lat/lon) ahead of `regrid`, and the area-weighted
    regrid is then rejected on the now-unstructured cube.

    Non-vacuity: reverting `write_suite_yaml` to PyYAML's default
    (`sort_keys=True`) makes this assertion fail with the alphabetised
    order `annual_statistics, anomalies, area_statistics, regrid,
    regrid_time` -- verified by running it against a sort_keys=True dump
    in `test_write_suite_yaml_order_gate_is_not_vacuous` below.
    """
    yaml = pytest.importorskip("yaml")
    out = tmp_path / "suite.yml"
    write_suite_yaml(copy.deepcopy(_CONSERVATION_SUITE_DEF), out)

    round_tripped = yaml.safe_load(out.read_text())
    chain = list(
        round_tripped[0]["variables"][0]["additional_preprocessors"],
    )
    assert chain == _TIER1_PREPROCESSOR_CHAIN
    # regrid must come BEFORE the lat/lon-collapsing area_statistics
    assert chain.index("esmvalcore.preprocessor.regrid") < chain.index(
        "esmvalcore.preprocessor.area_statistics",
    )
    # anomalies is the LAST step: the relative anomaly OF the global sum,
    # not the global mean of local relative anomalies.
    assert chain[-1] == "esmvalcore.preprocessor.anomalies"


def test_write_suite_yaml_order_gate_is_not_vacuous(tmp_path):
    """Prove the order assertion above can actually fail.

    Dumps the SAME definition the way the defective code did
    (PyYAML's default sort_keys=True) and asserts the chain is corrupted
    into exactly the alphabetical order observed in production. If PyYAML
    ever stopped sorting by default, this test goes red and tells us the
    gate above has become vacuous.
    """
    yaml = pytest.importorskip("yaml")
    out = tmp_path / "suite_sorted.yml"
    with out.open("w") as f:
        yaml.dump(copy.deepcopy(_CONSERVATION_SUITE_DEF), f)  # sort_keys=True

    chain = list(
        yaml.safe_load(out.read_text())[0]["variables"][0][
            "additional_preprocessors"
        ],
    )
    assert chain != _TIER1_PREPROCESSOR_CHAIN
    assert chain == [
        "esmvalcore.preprocessor.annual_statistics",
        "esmvalcore.preprocessor.anomalies",
        "esmvalcore.preprocessor.area_statistics",
        "esmvalcore.preprocessor.regrid",
        "esmvalcore.preprocessor.regrid_time",
    ]
