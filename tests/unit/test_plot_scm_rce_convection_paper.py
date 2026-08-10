"""Gates for ``scripts/plot/plot_scm_rce_convection_paper.py``.

The figure script consumes artifacts written by TWO drivers that name the
untuned arm differently (``a_priori`` vs ``prior``). A silent miss there does
not raise — it drops the dashed a-priori curve from every panel, which looks
like a deliberate plotting choice rather than a bug, so it is gated here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load():
    spec = importlib.util.spec_from_file_location(
        "_rce_paper_plots",
        REPO_ROOT / "scripts" / "plot" / "plot_scm_rce_convection_paper.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def plots():
    return _load()


def test_reads_the_tuning_driver_key(plots):
    rec = {"a_priori": {"T_profile": [1.0]}, "tuned": {"T_profile": [2.0]}}
    assert plots._arm(rec, tuned=False)["T_profile"] == [1.0]
    assert plots._arm(rec, tuned=True)["T_profile"] == [2.0]


def test_reads_the_intercomparison_driver_key(plots):
    rec = {"prior": {"T_profile": [1.0]}, "tuned": {"T_profile": [2.0]}}
    assert plots._arm(rec, tuned=False)["T_profile"] == [1.0]


def test_prefers_a_priori_when_both_are_present(plots):
    """Deterministic, so a checkpoint carrying both cannot plot differently
    depending on dict ordering."""
    rec = {"prior": {"T_profile": [9.0]}, "a_priori": {"T_profile": [1.0]},
           "tuned": {"T_profile": [2.0]}}
    assert plots._arm(rec, tuned=False)["T_profile"] == [1.0]


def test_a_checkpoint_with_no_a_priori_arm_raises(plots):
    """NON-VACUITY: the failure mode is a silently missing curve, so the miss
    must raise rather than return an empty dict."""
    with pytest.raises(KeyError, match="a-priori"):
        plots._arm({"tuned": {"T_profile": [2.0]}}, tuned=False)


def test_rmse_summary_reads_the_physical_unit_columns(plots):
    """The bar chart is in K and g/kg; those column names are the contract
    with run_scm_rce_convection_intercomparison.CSV_FIELDS."""
    src = (REPO_ROOT / "scripts" / "plot"
           / "plot_scm_rce_convection_paper.py").read_text()
    body = src.split("def _rmse_summary")[1].split("\ndef ")[0]
    for name in ("apriori_T_rmse_K", "tuned_T_rmse_K",
                 "apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg"):
        assert name in body, name


def test_the_csv_the_figures_read_is_the_one_the_campaign_writes(plots):
    """Both sides of the contract, in one place: the figure script's file name
    and the intercomparison driver's emitted names must agree."""
    plot_src = (REPO_ROOT / "scripts" / "plot"
                / "plot_scm_rce_convection_paper.py").read_text()
    driver_src = (REPO_ROOT / "scripts" / "run"
                  / "run_scm_rce_convection_intercomparison.py").read_text()
    assert '"summary_table.csv"' in plot_src
    assert '"summary_table.csv"' in driver_src
