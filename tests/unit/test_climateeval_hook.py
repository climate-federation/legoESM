"""Direct unit tests for legoesm.driver.climateeval_hook.

Exercises the subprocess-bridging logic without invoking a real
ClimateEval environment (mocked ``subprocess.run``).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from legoesm.driver.climateeval_hook import maybe_run_climateeval
from legoesm.driver.config import EvaluationConfig, ExperimentConfig, OutputConfig


def _config(evaluation: EvaluationConfig) -> ExperimentConfig:
    return ExperimentConfig(output=OutputConfig(cmip_output=True, evaluation=evaluation))


def test_disabled_is_a_noop():
    config = _config(EvaluationConfig(enabled=False))
    with patch("legoesm.driver.climateeval_hook.subprocess.run") as run:
        result = maybe_run_climateeval(config, "/tmp/some_run")
    run.assert_not_called()
    assert result is None


def test_enabled_invokes_expected_subprocess_command():
    evaluation = EvaluationConfig(
        enabled=True,
        suites=("Tier1_sanity_checks", "Tier2_atmosphere_monthly"),
        model_id="legoESM-1-0",
        experiment_id="amip",
        variant_id="r1i1p1f1",
        data_root_dir="/work/bd1179/b309141/climateeval_input",
        timerange="19790101/19791231",
        fail_on_missing_data=True,
        download_missing_data=True,
        climateeval_python="/opt/climateeval_env/bin/python",
    )
    config = _config(evaluation)
    mock_result = MagicMock(returncode=0)
    with patch("legoesm.driver.climateeval_hook.subprocess.run", return_value=mock_result) as run:
        result = maybe_run_climateeval(config, "/scratch/b/b309178/amip_run")

    run.assert_called_once()
    cmd = run.call_args.args[0]
    assert cmd[0] == "/opt/climateeval_env/bin/python"
    assert cmd[1].endswith("scripts/validate/run_amip_climateeval.py")
    assert "--cmor-dir" in cmd
    # cmor ROOT (so ocean/sea-ice tables load), not just Amon
    assert cmd[cmd.index("--cmor-dir") + 1] == "/scratch/b/b309178/amip_run/cmor"
    # explicit suites passed after --suite (nargs="+"); one report via --output-dir
    suite_i = cmd.index("--suite")
    assert cmd[suite_i + 1:suite_i + 3] == ["Tier1_sanity_checks", "Tier2_atmosphere_monthly"]
    assert cmd[cmd.index("--output-dir") + 1] == "/scratch/b/b309178/amip_run"
    assert cmd[cmd.index("--timerange") + 1] == "19790101/19791231"
    assert "--fail-on-missing-data" in cmd
    assert "--download-missing-data" in cmd
    assert result == 0


def test_empty_suites_omits_suite_flag():
    # default (empty) suites -> no --suite -> runner discovers + runs ALL tiers
    evaluation = EvaluationConfig(enabled=True, climateeval_python="/x/py")
    config = _config(evaluation)
    mock_result = MagicMock(returncode=0)
    with patch("legoesm.driver.climateeval_hook.subprocess.run", return_value=mock_result) as run:
        maybe_run_climateeval(config, "/tmp/run")
    cmd = run.call_args.args[0]
    assert "--suite" not in cmd
    assert cmd[cmd.index("--cmor-dir") + 1] == "/tmp/run/cmor"


def test_empty_timerange_omits_flag():
    evaluation = EvaluationConfig(enabled=True, timerange="")
    config = _config(evaluation)
    mock_result = MagicMock(returncode=0)
    with patch("legoesm.driver.climateeval_hook.subprocess.run", return_value=mock_result) as run:
        maybe_run_climateeval(config, "/tmp/run")
    cmd = run.call_args.args[0]
    assert "--timerange" not in cmd


def test_nonzero_exit_is_logged_not_raised(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="legoesm.driver.climateeval_hook")
    evaluation = EvaluationConfig(enabled=True)
    config = _config(evaluation)
    mock_result = MagicMock(returncode=1)
    with patch("legoesm.driver.climateeval_hook.subprocess.run", return_value=mock_result):
        result = maybe_run_climateeval(config, "/tmp/run")
    assert result == 1
    assert any("ClimateEval evaluation failed" in r.message for r in caplog.records)
