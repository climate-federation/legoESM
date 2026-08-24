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
        "apriori_thermo_T_term": str(T_prior),
        "tuned_thermo_T_term": str(T_tuned),
        # Full-column values deliberately DIFFERENT, so a test that reads the
        # wrong pair fails instead of silently agreeing.
        "apriori_T_rmse_K": str(T_prior * 3.0),
        "tuned_T_rmse_K": str(T_tuned * 3.0),
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
    """A NAMED structural reason applies only while the row is genuinely flat."""
    named = {"scheme": "kuo", "n_tuned_params": "2"}
    assert P._structural_note(named, 1.0, 1.0) is not None
    assert P._structural_note(named, 1.0, 0.5) is None
    # A scheme with no named reason and a nonzero parameter count is never
    # annotated, however flat it happens to be.
    assert P._structural_note(
        {"scheme": "bechtold", "n_tuned_params": "3"}, 1.0, 1.0) is None


def test_zero_parameter_note_comes_from_the_row_not_a_hardcoded_scheme_list():
    """`dca` had no tunable parameter under the extended tier and HAS one under
    the physical set, so a hardcoded per-scheme note would become a false label
    the moment the parameter set changes. The note must follow the count."""
    assert P._structural_note(
        {"scheme": "dca", "n_tuned_params": "0"}, 1.0, 0.5) == P._NO_PARAMS_NOTE
    assert P._structural_note(
        {"scheme": "dca", "n_tuned_params": "1"}, 1.0, 0.5) is None
    # ...and it is not specific to dca: ANY zero-parameter row gets it.
    assert P._structural_note(
        {"scheme": "bechtold", "n_tuned_params": "0"}, 1.0, 1.0) == (
            P._NO_PARAMS_NOTE)
    assert "dca" not in P.STRUCTURALLY_FLAT


def test_panel_ranks_by_the_tuned_value_worst_first(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [_row("a", 2.0, 0.4, 0.2, 0.05),
            _row("b", 0.5, 1.9, 0.1, 0.30),
            _row("c", 1.0, 1.0, 0.3, 0.10)]
    fig, ax = plt.subplots()
    P._panel(ax, rows, P.T_COLUMN_CHOICES["trop"], title="T", log_x=False)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    plt.close(fig)
    # tuned T: a 0.4, b 1.9, c 1.0 -> worst first is b, c, a
    assert labels == ["b", "c", "a"]


def test_a_row_with_no_numbers_is_dropped_not_plotted_at_zero(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [_row("a", 2.0, 0.4, 0.2, 0.05)]
    dead = _row("dead", 1.0, 1.0, 1.0, 1.0)
    for k in list(dead):
        if k not in ("scheme", "objective", "n_tuned_params"):
            dead[k] = ""
    rows.append(dead)
    fig, ax = plt.subplots()
    P._panel(ax, rows, P.T_COLUMN_CHOICES["trop"], title="T", log_x=False)
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


def test_temperature_panel_defaults_to_the_scored_domain():
    """The two panels must share a vertical domain. Plotting a FULL-COLUMN
    temperature beside a TROPOSPHERIC humidity puts two different domains side
    by side, and on real data it understated the tuned improvement (edmf:
    8.35->7.22 K over the column against 6.91->3.65 K over the mask)."""
    assert P.T_COLUMN_CHOICES["trop"][0] == "apriori_thermo_T_term"
    assert P.T_COLUMN_CHOICES["trop"][1] == "tuned_thermo_T_term"
    assert "tropospheric" in P.T_COLUMN_CHOICES["trop"][2]
    assert "full-column" in P.T_COLUMN_CHOICES["column"][2]


def test_temperature_choice_changes_what_is_plotted(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [_row("a", 2.0, 0.4, 0.2, 0.05)]
    seen = {}
    for choice in ("trop", "column"):
        fig, ax = plt.subplots()
        P._panel(ax, rows, P.T_COLUMN_CHOICES[choice], title="T", log_x=False)
        seen[choice] = [c.get_offsets().tolist() for c in ax.collections]
        plt.close(fig)
    assert seen["trop"] != seen["column"], (
        "the two temperature choices must plot different numbers, else the "
        "option is decoration")


def test_unknown_temperature_choice_raises(tmp_path):
    rows = [_row("a", 2.0, 0.4, 0.2, 0.05)]
    with pytest.raises(ValueError, match="unknown temperature"):
        P.make_figure(rows, tmp_path / "x.png", humidity="logq", log_x=False,
                      suptitle=None, temperature="stratosphere")
