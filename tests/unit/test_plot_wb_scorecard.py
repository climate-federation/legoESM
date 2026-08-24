"""Login-node-safe unit tests for scripts/plot/plot_wb_scorecard.py.

No JAX, no network. Matplotlib Agg backend, synthetic scorecard + SOTA dicts
matching the real JSON schema written by run_weatherbench_eval.main:
  {"meta", "model", "persistence", "climatology"} where each of the last three
  is {field_key: {str(lead_hours): {"rmse"/"acc"/"bias": value}}}.
The SOTA dict matches evaluations.baselines.load_sota_headline:
  {model: {(variable, level, lead_hours): rmse}}.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest

# Import the plot script by path (scripts/ is not an importable package).
_REPO = Path(__file__).resolve().parents[2]
_MOD_PATH = _REPO / "scripts" / "plot" / "plot_wb_scorecard.py"
_spec = importlib.util.spec_from_file_location("plot_wb_scorecard", _MOD_PATH)
plot_wb_scorecard = importlib.util.module_from_spec(_spec)
sys.modules["plot_wb_scorecard"] = plot_wb_scorecard
_spec.loader.exec_module(plot_wb_scorecard)


def _synthetic_scorecard(z500_rmse, t850_rmse, *, with_acc=True):
    """A scorecard dict for two headline fields at leads 24/72 h."""
    def leadmap(base, acc0):
        out = {}
        for lead, mult, acc in ((24, 1.0, acc0), (72, 1.8, acc0 - 0.15)):
            m = {"rmse": base * mult, "bias": 0.1 * base}
            if with_acc:
                m["acc"] = acc
            out[str(lead)] = m
        return out
    section = {
        "z500": leadmap(z500_rmse, 0.95),
        "t850": leadmap(t850_rmse, 0.90),
    }
    # persistence/climatology floors are worse (higher RMSE)
    persist = {k: {ld: {**v, "rmse": v["rmse"] * 3.0}
                   for ld, v in fld.items()} for k, fld in section.items()}
    clim = {k: {ld: {**v, "rmse": v["rmse"] * 4.0}
                for ld, v in fld.items()} for k, fld in section.items()}
    return {"meta": {"mode": "test"}, "model": section,
            "persistence": persist, "climatology": clim}


def _synthetic_sota():
    """{model: {(variable, level, lead_hours): rmse}} — z500 + t850 only."""
    return {
        "graphcast": {
            ("geopotential", 500, 24): 45.0,
            ("geopotential", 500, 72): 120.0,
            ("temperature", 850, 24): 0.6,
            ("temperature", 850, 72): 1.1,
        },
        "ifs_hres": {
            ("geopotential", 500, 24): 55.0,
            ("geopotential", 500, 72): 150.0,
        },
    }


def test_build_figure_axes_count_and_lines():
    scorecards = {
        "physics": _synthetic_scorecard(50.0, 0.7),
        "neural_gcm": _synthetic_scorecard(40.0, 0.6),
    }
    sota = _synthetic_sota()
    fig = plot_wb_scorecard.build_scorecard_figure(
        scorecards, sota, metric="rmse", ncols=2)
    # 2 fields -> 2 populated axes on a 1x2 grid
    axes = fig.get_axes()
    populated = [ax for ax in axes if ax.get_title()]
    assert len(populated) == 2

    # z500 panel must carry both legoESM family lines + both SOTA models
    z_ax = next(ax for ax in populated if ax.get_title().startswith("Z500"))
    labels = {t.get_text() for t in z_ax.get_legend().get_texts()}
    assert "legoESM:physics" in labels
    assert "legoESM:neural_gcm" in labels
    assert "SOTA:graphcast" in labels
    assert "SOTA:ifs_hres" in labels
    assert "persistence" in labels
    assert "climatology" in labels

    import matplotlib.pyplot as plt
    plt.close(fig)


def test_missing_field_and_metric_are_absent_not_crash():
    # one family missing t850 entirely; ACC absent from the other
    sc_a = _synthetic_scorecard(50.0, 0.7, with_acc=False)
    sc_b = _synthetic_scorecard(40.0, 0.6, with_acc=True)
    del sc_b["model"]["t850"]              # drop a whole field from one family
    fig = plot_wb_scorecard.build_scorecard_figure(
        {"a": sc_a, "b": sc_b}, None, metric="acc")
    # acc requested but family 'a' has none -> only 'b' contributes; z500 present
    z_ax = next(ax for ax in fig.get_axes() if ax.get_title().startswith("Z500"))
    labels = {t.get_text() for t in z_ax.get_legend().get_texts()}
    assert "legoESM:b" in labels
    assert "legoESM:a" not in labels      # no acc -> no line, no crash
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_field_subset_and_unknown_field_warn():
    scorecards = {"p": _synthetic_scorecard(50.0, 0.7)}
    with pytest.warns(UserWarning):
        fig = plot_wb_scorecard.build_scorecard_figure(
            scorecards, None, fields=["z500", "nonexistent_field"])
    populated = [ax for ax in fig.get_axes() if ax.get_title()]
    assert len(populated) == 1            # only z500 survives
    assert populated[0].get_title().startswith("Z500")
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_variable_level_mapping_helper():
    assert plot_wb_scorecard.sota_key_for_field("z500") == ("geopotential", 500)
    assert plot_wb_scorecard.sota_key_for_field("t850") == ("temperature", 850)
    assert plot_wb_scorecard.sota_key_for_field("mslp") == \
        ("mean_sea_level_pressure", 0)
    assert plot_wb_scorecard.sota_key_for_field("u250") == \
        ("u_component_of_wind", 250)
    assert plot_wb_scorecard.sota_key_for_field("not_a_field") is None


def test_unmapped_field_warns_for_sota_overlay():
    # a field with no SOTA (variable, level) mapping still plots legoESM curves
    # but warns once when a SOTA dict is supplied. 'tcwv' is a plausible extra
    # field_key that is NOT in FIELD_KEY_TO_SOTA.
    assert plot_wb_scorecard.sota_key_for_field("tcwv") is None   # precondition
    sc = {"meta": {}, "model": {"tcwv": {"24": {"rmse": 2.0}}},
          "persistence": {}, "climatology": {}}
    sota = {"graphcast": {("geopotential", 500, 24): 45.0}}
    with pytest.warns(UserWarning):
        fig = plot_wb_scorecard.build_scorecard_figure({"p": sc}, sota)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_parse_args_round_trip():
    args = plot_wb_scorecard.parse_args([
        "--scorecard", "physics=/a/physics/scorecard.json",
        "--scorecard", "neural_gcm=/b/ngcm/scorecard.json",
        "--sota-csv", "/c/sota.csv",
        "--out", "/d/out.png",
        "--metric", "acc",
        "--fields", "z500,t850",
    ])
    assert args.scorecard == [
        "physics=/a/physics/scorecard.json",
        "neural_gcm=/b/ngcm/scorecard.json",
    ]
    assert args.sota_csv == "/c/sota.csv"
    assert args.out == "/d/out.png"
    assert args.metric == "acc"
    assert args.fields == "z500,t850"


def test_load_scorecards_bad_spec_and_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        plot_wb_scorecard._load_scorecards(["no_equals_sign"])
    with pytest.raises(SystemExit):
        plot_wb_scorecard._load_scorecards(["x=/definitely/missing.json"])


def test_main_writes_png(tmp_path):
    import json
    sc_path = tmp_path / "physics" / "scorecard.json"
    sc_path.parent.mkdir(parents=True)
    sc_path.write_text(json.dumps(_synthetic_scorecard(50.0, 0.7)))
    out = tmp_path / "sub" / "wb_scorecard.png"
    ret = plot_wb_scorecard.main([
        "--scorecard", f"physics={sc_path}",
        "--out", str(out),
    ])
    assert out.exists() and out.stat().st_size > 0
    assert ret == str(out)


def test_invalid_metric_raises():
    with pytest.raises(ValueError):
        plot_wb_scorecard.build_scorecard_figure(
            {"p": _synthetic_scorecard(50.0, 0.7)}, None, metric="bogus")


def test_a_confounded_family_is_named_on_the_figure():
    """A status recorded in a file nobody reads is decoration.

    The figure is what gets read, so a family whose learned column ran without
    the classical arm's surface friction has to be named on it (GLM-5.2)."""
    import importlib.util
    import pathlib

    import matplotlib
    matplotlib.use("Agg")

    root = pathlib.Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "plot_wb_scorecard", root / "scripts" / "plot" / "plot_wb_scorecard.py")
    plot = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["plot_wb_scorecard"] = plot
    spec.loader.exec_module(plot)

    def _card(note):
        return {"meta": {"surface_drag_confound": note},
                "model": {"z500": {"rmse": {"24": 50.0, "48": 70.0}}}}

    clean = {"physics": _card("arms carry the same surface stress")}
    fig = plot.build_scorecard_figure(clean)
    assert "NOT AN EQUALISED" not in fig._suptitle.get_text()

    dirty = dict(clean)
    dirty["neural_gcm"] = _card(
        "learned arm has NO surface stress: the classical arm runs 'clubb'")
    fig = plot.build_scorecard_figure(dirty)
    title = fig._suptitle.get_text()
    assert "NOT AN EQUALISED" in title and "neural_gcm" in title, title
