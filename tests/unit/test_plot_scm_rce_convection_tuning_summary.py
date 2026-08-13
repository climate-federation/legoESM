"""Gates for the SCM-RCE convection tuning-summary plotter.

The figure's job is to keep three things from being misread, and each one is
gated behaviourally rather than by inspecting the source:

* a ``None`` diagnostic must become NaN and be dropped, never 0.0 -- a zero
  plots as a real measurement at the bottom of the axis;
* a pair that is flat BY CONSTRUCTION (a scheme with no tunable parameters,
  or one that is inactive in a single column) must be drawn and labelled
  differently from a pair that a search failed to move; and
* a checkpoint missing an arm must stop the figure rather than produce a
  half-populated one.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module():
    path = REPO_ROOT / "scripts" / "plot" / \
        "plot_scm_rce_convection_tuning_summary.py"
    spec = importlib.util.spec_from_file_location("scm_rce_tuning_summary",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


def _rec(scheme, prior_score, tuned_score, n_par, **extra):
    """A minimal checkpoint. Component keys default to the score so the
    per-component panels have finite data."""
    def arm(score):
        return {
            "score": score, "T_rmse": score, "qv_rmse": score,
            "cloud_rmse": score, "precip_rmse": score,
            "precip_mm_day": 1.0, "evap_mm_day": 1.0,
            "precip_ref_mm_day": 2.395,
            "T_profile": [300.0, 250.0], "qv_profile": [0.01, 0.001],
            "qcond_profile": [1e-5, 1e-6],
            **extra,
        }
    return {
        "scheme": scheme,
        "prior": arm(prior_score),
        "tuned": arm(tuned_score),
        "records": [{"parameter": f"p{i}"} for i in range(n_par)],
    }


def _write(tmp_path, recs, meta=None):
    for r in recs:
        (tmp_path / f"scheme_{r['scheme']}.json").write_text(json.dumps(r))
    (tmp_path / "run_meta.json").write_text(json.dumps(meta or {
        "days": 100.0, "dt": 600.0, "analysis_days": 5.0,
        "reference_dir": "/nowhere/rcemip_ref_sam300",
    }))
    return tmp_path


# --------------------------------------------------------------------------- #
# _ff: a missing diagnostic is NaN, never 0.0
# --------------------------------------------------------------------------- #

def test_none_becomes_nan_not_zero(mod):
    """0.0 is a plottable value and would appear as a perfect score."""
    assert np.isnan(mod._ff(None))
    assert np.isnan(mod._ff("not a number"))
    assert mod._ff("2.5") == pytest.approx(2.5)
    assert mod._ff(0.0) == 0.0          # a real zero still survives


# --------------------------------------------------------------------------- #
# _structural_note: flat-by-construction vs failed search
# --------------------------------------------------------------------------- #

NO_CONV = 5.736693549527354


def test_zero_parameter_scheme_is_annotated(mod):
    assert mod._structural_note(_rec("dca", 9.3, 9.3, 0), NO_CONV) \
        == "no tunable parameters"


def test_a_scheme_scoring_the_baseline_is_annotated(mod):
    """kuo exposes two tunable parameters but contributes nothing in a single
    column, which shows up as its score EQUALLING the measured no-convection
    baseline. That measurement is the evidence, not the scheme's name."""
    assert mod._structural_note(_rec("kuo", NO_CONV, NO_CONV, 2), NO_CONV) \
        == "scores the no-convection baseline"


def test_the_label_is_not_keyed_on_the_scheme_name(mod):
    """NON-VACUITY for the codex finding that a name table cannot express
    causality: ANY scheme sitting at the baseline earns the label, and `kuo`
    away from the baseline does NOT -- so wiring kuo up for single columns
    removes the label with no edit here."""
    assert mod._structural_note(_rec("emanuel", NO_CONV, NO_CONV, 7),
                                NO_CONV) == "scores the no-convection baseline"
    assert mod._structural_note(_rec("kuo", 4.10, 4.10, 2), NO_CONV) is None


def test_a_scheme_that_moved_is_never_annotated(mod):
    assert mod._structural_note(_rec("kuo", NO_CONV, 4.10, 2), NO_CONV) is None
    assert mod._structural_note(_rec("dca", 9.3, 8.0, 0), NO_CONV) is None


def test_an_ordinary_scheme_whose_search_failed_is_not_annotated(mod):
    """A flat pair with real parameters, away from the baseline, is a failed
    SEARCH and must read as one -- the case the grey styling must not
    swallow."""
    assert mod._structural_note(_rec("tiedtke", 3.21, 3.21, 16),
                                NO_CONV) is None


def test_a_nonfinite_pair_is_not_called_structural(mod):
    """`NaN != NaN` would read as 'moved'; an inf pair would read as 'flat'.
    Neither is a structural constant -- both are failures to be shown."""
    assert mod._structural_note(_rec("x", float("nan"), float("nan"), 0),
                                NO_CONV) is None
    assert mod._structural_note(_rec("x", float("inf"), float("inf"), 0),
                                NO_CONV) is None


def test_baseline_label_requires_the_baseline_to_be_supplied(mod):
    """Without a measured baseline the label must not be guessed."""
    assert mod._structural_note(_rec("kuo", NO_CONV, NO_CONV, 2), None) is None


# --------------------------------------------------------------------------- #
# _load: refuse a malformed or partial checkpoint set
# --------------------------------------------------------------------------- #

def test_missing_arm_is_refused(mod, tmp_path):
    bad = _rec("sbm", 11.9, 7.6, 2)
    del bad["prior"]
    (tmp_path / "scheme_sbm.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="prior"):
        mod._load(tmp_path, None)


def test_empty_directory_is_refused(mod, tmp_path):
    with pytest.raises(SystemExit, match="no scheme_"):
        mod._load(tmp_path, None)


def test_results_are_sorted_by_tuned_score(mod, tmp_path):
    _write(tmp_path, [_rec("dca", 9.3, 9.3, 0),
                      _rec("emanuel", 0.94, 0.79, 11),
                      _rec("edmf", 0.98, 0.95, 5)])
    got = [r["scheme"] for r in mod._load(tmp_path, None)]
    assert got == ["emanuel", "edmf", "dca"]


def test_scheme_filter_selects_a_subset(mod, tmp_path):
    _write(tmp_path, [_rec("dca", 9.3, 9.3, 0),
                      _rec("emanuel", 0.94, 0.79, 11)])
    got = [r["scheme"] for r in mod._load(tmp_path, ["emanuel"])]
    assert got == ["emanuel"]


def test_a_filter_matching_nothing_is_a_hard_error(mod, tmp_path):
    """Dispatch hardening: an exact selector matching nothing must raise, not
    silently draw an empty figure."""
    _write(tmp_path, [_rec("dca", 9.3, 9.3, 0)])
    with pytest.raises(SystemExit, match="no scheme matched"):
        mod._load(tmp_path, ["nosuchscheme"])


# --------------------------------------------------------------------------- #
# The figure itself
# --------------------------------------------------------------------------- #

def test_before_after_figure_writes_and_reports_every_component(mod, tmp_path):
    _write(tmp_path, [_rec("emanuel", 0.94, 0.79, 11),
                      _rec("kuo", 5.7367, 5.7367, 2),
                      _rec("dca", 9.3, 9.3, 0),
                      _rec("sbm", 11.9, 7.6, 2)])
    recs = mod._load(tmp_path, None)
    out = tmp_path / "fig.png"
    medians = mod.figure_before_after(recs, {}, out, 5.737)
    assert out.exists() and out.stat().st_size > 0
    assert set(medians) == {k for k, _ in mod.COMPONENTS}
    # median of the four tuned scores 0.79, 5.7367, 7.6, 9.3
    mb, ma = medians["score"]
    assert ma == pytest.approx(np.median([0.79, 5.7367, 7.6, 9.3]))
    assert mb == pytest.approx(np.median([0.94, 5.7367, 9.3, 11.9]))


def test_a_non_finite_component_does_not_kill_the_panel(mod, tmp_path):
    """A scheme that blew up scores +inf. It must drop out of the box rather
    than take the axis limits to infinity with it."""
    blown = _rec("broken", float("inf"), float("inf"), 3)
    _write(tmp_path, [_rec("emanuel", 0.94, 0.79, 11), blown,
                      _rec("edmf", 0.98, 0.95, 5)])
    recs = mod._load(tmp_path, None)
    out = tmp_path / "fig.png"
    medians = mod.figure_before_after(recs, {}, out, None)
    assert out.exists()
    # only the two finite pairs enter the median
    assert medians["score"][1] == pytest.approx(np.median([0.79, 0.95]))


def test_every_component_panel_is_actually_drawn(mod, tmp_path, monkeypatch):
    """The combined score is dominated by the condensate term, so dropping a
    per-component panel would hide the temperature and humidity rankings the
    campaign is about.  Asserted by SPYING on the panel calls, not by
    comparing the COMPONENTS constant to a literal -- the latter passes even
    if the figure draws nothing (codex finding 15).
    """
    seen = []
    real = mod._panel

    def spy(ax, names, before, after, notes, title, ylabel, baseline=None):
        seen.append((title, tuple(before), tuple(after)))
        return real(ax, names, before, after, notes, title, ylabel,
                    baseline=baseline)

    monkeypatch.setattr(mod, "_panel", spy)
    _write(tmp_path, [_rec("emanuel", 0.94, 0.79, 11),
                      _rec("dca", 9.3, 9.3, 0)])
    recs = mod._load(tmp_path, None)
    mod.figure_before_after(recs, {}, tmp_path / "f.png", NO_CONV)

    assert [t for t, _, _ in seen] == [t for _, t in mod.COMPONENTS]
    # and each panel received THAT component's numbers, not the score's
    by_title = {t: (b, a) for t, b, a in seen}
    assert by_title["temperature"][1] == (0.79, 9.3)     # T_rmse mirrors score
    assert len(seen) == 5


def _segment_colours(mod, names, before, after, notes):
    """Draw one panel on a throwaway axis and return each scheme's SEGMENT
    colour, so styling claims are asserted on the artist rather than on the
    fact that a file was written (codex round 2, finding 13)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    mod._panel(ax, names, before, after, notes, "t", None)
    # the segments are the 2-point lines; endpoint markers are 1-point plots
    segs = [ln for ln in ax.lines if len(ln.get_xdata()) == 2]
    out = {n: ln.get_color() for n, ln in zip(names, segs)}
    plt.close(fig)
    return out


def test_a_component_that_moved_is_coloured_by_its_own_direction(mod):
    """codex round 1 finding 2: structural status is decided on the COMBINED
    score, so a scheme can be flat overall while a component moved. That
    component must take its own direction colour, not the grey that the
    legend labels 'flat by construction'."""
    names = ["kuo", "edmf", "sbm"]
    # kuo is structurally flat overall, but in THIS component it improved
    notes = ["scores the no-convection baseline", None, None]
    got = _segment_colours(mod, names,
                           before=[0.30, 0.98, 0.40],
                           after=[0.20, 0.95, 0.55], notes=notes)
    assert got["kuo"] == mod.AFTER, (
        f"a component that moved was drawn {got['kuo']}, not the improved "
        f"colour — the grey legend entry would hide a real change")
    assert got["edmf"] == mod.AFTER
    assert got["sbm"] == mod.WORSE


def test_a_structurally_flat_component_is_greyed_and_dashed(mod):
    """The other half: when the component really did not move AND the scheme
    is structurally flat, it must be grey and dashed."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    mod._panel(ax, ["kuo", "edmf"], [5.7367, 0.98], [5.7367, 0.95],
               ["scores the no-convection baseline", None], "t", None)
    segs = [ln for ln in ax.lines if len(ln.get_xdata()) == 2]
    assert segs[0].get_color() == mod.INERT
    assert segs[0].get_linestyle() != "-"
    assert segs[1].get_color() == mod.AFTER
    plt.close(fig)


def test_an_unchanged_component_is_not_coloured_as_degraded(mod):
    """codex round 1 finding 7: equality is not degradation."""
    got = _segment_colours(mod, ["a", "b"], [1.0, 1.0], [1.0, 1.5],
                           [None, None])
    assert got["a"] == mod.INERT
    assert got["b"] == mod.WORSE


def test_a_missing_pair_is_disclosed_on_the_panel(mod):
    """codex round 2 finding 3: complete-case filtering shrinks the cohort
    silently, so the median can describe fewer schemes than the caption
    claims."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    mod._panel(ax, ["good", "gone"], [1.0, float("nan")],
               [0.5, float("nan")], [None, None], "t", None)
    texts = " ".join(t.get_text() for t in ax.texts)
    plt.close(fig)
    assert "1/2" in texts and "gone" in texts, texts


def test_an_all_missing_score_refuses_to_publish_a_nan_title(mod, tmp_path):
    """codex round 2 finding 4: a figure captioned 'median nan -> nan (nan%)'
    looks like a result."""
    a = _rec("a", float("nan"), float("nan"), 3)
    b = _rec("b", float("nan"), float("nan"), 3)
    _write(tmp_path, [a, b])
    recs = mod._load(tmp_path, None)
    with pytest.raises(SystemExit, match="not finite"):
        mod.figure_before_after(recs, {}, tmp_path / "f.png", None)


def test_missing_water_budget_is_refused(mod, tmp_path):
    """codex round 2 finding 5: matplotlib skips a NaN bar, and a blank bar
    reads as a measured zero."""
    class _Ref:
        T_ref = np.array([300.0, 250.0])
        qv_ref = np.array([0.018, 0.001])
        qcond_ref = np.array([2e-5, 1e-6])

    a = _rec("emanuel", 0.94, 0.79, 11)
    a["tuned"]["precip_mm_day"] = None
    _write(tmp_path, [a, _rec("edmf", 0.98, 0.95, 5)])
    recs = mod._load(tmp_path, None)
    with pytest.raises(SystemExit, match="missing precip/evap"):
        mod.figure_profiles(recs, {}, tmp_path / "p.png", _Ref(),
                            np.array([1000.0, 200.0]))


def test_a_single_reference_copy_does_not_satisfy_agreement(mod, tmp_path):
    """codex round 2 finding 6: one finite value beside NaNs passes an
    agreement test vacuously."""
    class _Ref:
        T_ref = np.array([300.0, 250.0])
        qv_ref = np.array([0.018, 0.001])
        qcond_ref = np.array([2e-5, 1e-6])

    b = _rec("edmf", 0.98, 0.95, 5)
    b["tuned"]["precip_ref_mm_day"] = None
    _write(tmp_path, [_rec("emanuel", 0.94, 0.79, 11), b])
    recs = mod._load(tmp_path, None)
    with pytest.raises(SystemExit, match="no finite precip_ref_mm_day"):
        mod.figure_profiles(recs, {}, tmp_path / "p.png", _Ref(),
                            np.array([1000.0, 200.0]))


def test_nonfinite_scores_sort_last_deterministically(mod, tmp_path):
    """codex finding 3: NaN comparisons are unordered, so sorting on the raw
    float leaves a missing-score scheme wherever the filename order put it
    while the legend advertises best-first."""
    _write(tmp_path, [_rec("zzz_nan", 1.0, float("nan"), 3),
                      _rec("aaa_good", 5.0, 4.0, 3),
                      _rec("mmm_best", 2.0, 0.5, 3)])
    got = [r["scheme"] for r in mod._load(tmp_path, None)]
    assert got == ["mmm_best", "aaa_good", "zzz_nan"], got


def test_disagreeing_reference_precipitation_is_refused(mod, tmp_path):
    """codex finding 5: two checkpoints scored against different references
    must not share a figure -- silently drawing recs[0]'s value makes the CRM
    line depend on sort order."""
    class _Ref:
        T_ref = np.array([300.0, 250.0])
        qv_ref = np.array([0.018, 0.001])
        qcond_ref = np.array([2e-5, 1e-6])

    a = _rec("emanuel", 0.94, 0.79, 11)
    b = _rec("edmf", 0.98, 0.95, 5)
    b["tuned"]["precip_ref_mm_day"] = 3.100      # a different reference
    _write(tmp_path, [a, b])
    recs = mod._load(tmp_path, None)
    with pytest.raises(SystemExit, match="precip_ref_mm_day"):
        mod.figure_profiles(recs, {}, tmp_path / "p.png", _Ref(),
                            np.array([1000.0, 200.0]))


def test_profiles_figure_writes(mod, tmp_path):
    class _Ref:
        T_ref = np.array([300.0, 250.0])
        qv_ref = np.array([0.018, 0.001])
        qcond_ref = np.array([2e-5, 1e-6])

    _write(tmp_path, [_rec("emanuel", 0.94, 0.79, 11),
                      _rec("dca", 9.3, 9.3, 0)])
    recs = mod._load(tmp_path, None)
    out = tmp_path / "prof.png"
    mod.figure_profiles(recs, {"days": 100.0, "analysis_days": 5.0},
                        out, _Ref(), np.array([1000.0, 200.0]))
    assert out.exists() and out.stat().st_size > 0
