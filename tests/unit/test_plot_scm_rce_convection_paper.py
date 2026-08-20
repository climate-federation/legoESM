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


def test_rmse_summary_plots_the_physical_unit_columns(plots, tmp_path,
                                                      monkeypatch):
    """EXECUTED, not grepped: a source search passes while the code reads the
    normalised columns and the names survive only in a comment.  Sentinel
    values go in; the bar geometry must come out."""
    import matplotlib
    matplotlib.use("Agg")
    captured = {}
    monkeypatch.setattr(plots, "_save", lambda fig, base: captured.setdefault(
        "widths", [[float(p.get_width()) for p in ax.patches]
                   for ax in fig.axes]))
    rows = [
        {"scheme": "alpha", "apriori_T_rmse_K": "3.0", "tuned_T_rmse_K": "1.0",
         "apriori_qv_rmse_g_kg": "0.7", "tuned_qv_rmse_g_kg": "0.2"},
        {"scheme": "beta", "apriori_T_rmse_K": "5.0", "tuned_T_rmse_K": "2.0",
         "apriori_qv_rmse_g_kg": "0.9", "tuned_qv_rmse_g_kg": "0.4"},
    ]
    plots._apply_paper_style()
    plots._rmse_summary(tmp_path / "fig", rows)
    widths = captured["widths"]
    assert len(widths) == 2, "expected a temperature panel and a humidity panel"
    # Panel (a) carries the K values, panel (b) the g/kg values — and the
    # a-priori bars must be the LONGER ones here, which is what makes the
    # sentinels distinguishable from the normalised columns (absent entirely).
    assert sorted(widths[0]) == pytest.approx([1.0, 2.0, 3.0, 5.0])
    assert sorted(widths[1]) == pytest.approx([0.2, 0.4, 0.7, 0.9])


def test_the_csv_the_figures_read_is_the_one_the_campaign_writes(plots,
                                                                 tmp_path):
    """Both sides of the contract, EXERCISED: write the file name the driver
    emits, and require the figure script's loader to find and parse it."""
    import csv as _csv
    fields = ["scheme", "apriori_T_rmse_K", "tuned_T_rmse_K",
              "apriori_qv_rmse_g_kg", "tuned_qv_rmse_g_kg"]
    with (tmp_path / "summary_table.csv").open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerow(dict(zip(fields, ["alpha", "3.0", "1.0", "0.7", "0.2"])))
    rows = plots._load_rows(tmp_path)
    assert [r["scheme"] for r in rows] == ["alpha"]
    # and the driver really writes that name
    driver_src = (REPO_ROOT / "scripts" / "run"
                  / "run_scm_rce_convection_intercomparison.py").read_text()
    assert 'summary_table.csv' in driver_src
