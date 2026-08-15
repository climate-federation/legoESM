"""Controls for the before/after T and humidity ranking figure.

A ranking figure is an instrument: it decides what a reader believes about
which scheme is best.  These pin the two ways it could lie — ordering by the
wrong number, and drawing a scheme that produced no number as if it had.
"""

from __future__ import annotations

import csv
import math

import pytest

from scripts.plot import plot_scm_rce_tq_ranking as P


def _row(scheme, T_prior, T_tuned, rh_prior, rh_tuned, n_params=5):
    return {
        "scheme": scheme,
        "objective": "thermo",
        "n_tuned_params": str(n_params),
        "apriori_T_rmse_K": str(T_prior),
        "tuned_T_rmse_K": str(T_tuned),
        "apriori_trop_rh_rmse": str(rh_prior),
        "tuned_trop_rh_rmse": str(rh_tuned),
        "apriori_qv_rmse_g_kg": str(rh_prior),
        "tuned_qv_rmse_g_kg": str(rh_tuned),
        "apriori_trop_qv_rmse_g_kg": str(rh_prior),
        "tuned_trop_qv_rmse_g_kg": str(rh_tuned),
    }


def _write(tmp_path, rows):
    path = tmp_path / "merged.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_ff_maps_blank_and_garbage_to_nan():
    assert math.isnan(P._ff(""))
    assert math.isnan(P._ff(None))
    assert math.isnan(P._ff("n/a"))
    assert P._ff("1.5") == 1.5


def test_structural_note_only_when_actually_flat():
    flat = {"scheme": "kuo", "n_tuned_params": "2"}
    assert P._structural_note(flat, 1.0, 1.0) is not None
    assert P._structural_note(flat, 1.0, 0.5) is None
    # A zero-parameter scheme is flat by construction whatever the numbers are.
    assert P._structural_note(
        {"scheme": "dca", "n_tuned_params": "0"}, 1.0, 0.5) is not None
    # A scheme not on the list is never annotated.
    assert P._structural_note(
        {"scheme": "bechtold", "n_tuned_params": "0"}, 1.0, 1.0) is None


def test_panel_ranks_by_the_tuned_value_worst_first(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [_row("a", 2.0, 0.4, 0.2, 0.05),
            _row("b", 0.5, 1.9, 0.1, 0.30),
            _row("c", 1.0, 1.0, 0.3, 0.10)]
    fig, ax = plt.subplots()
    P._panel(ax, rows, P.T_COLUMNS, title="T", log_x=False)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    plt.close(fig)
    # tuned T: a 0.4, b 1.9, c 1.0 -> worst first is b, c, a
    assert labels == ["b", "c", "a"]


def test_a_row_with_no_numbers_is_dropped_not_plotted_at_zero(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [_row("a", 2.0, 0.4, 0.2, 0.05), _row("dead", "", "", "", "")]
    fig, ax = plt.subplots()
    P._panel(ax, rows, P.T_COLUMNS, title="T", log_x=False)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    plt.close(fig)
    assert labels == ["a"]


def test_end_to_end_writes_a_file(tmp_path):
    rows = [_row("a", 2.0, 0.4, 0.2, 0.05), _row("kuo", 1.0, 1.0, 0.3, 0.3, 0)]
    csv_path = _write(tmp_path, rows)
    out = tmp_path / "fig.png"
    assert P.main([str(csv_path), "--out", str(out)]) == 0
    assert out.exists() and out.stat().st_size > 0


def test_unknown_humidity_choice_raises(tmp_path):
    rows = [_row("a", 2.0, 0.4, 0.2, 0.05)]
    with pytest.raises(ValueError, match="unknown humidity"):
        P.make_figure(rows, tmp_path / "x.png", humidity="specific",
                      log_x=False, suptitle=None)


def test_empty_csv_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("scheme\n")
    with pytest.raises(ValueError, match="no data rows"):
        P.read_rows(path)
