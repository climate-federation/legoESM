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

def test_zero_parameter_scheme_is_annotated(mod):
    assert mod._structural_note(_rec("dca", 9.3, 9.3, 0)) \
        == "no tunable parameters"


def test_inactive_scheme_is_annotated_even_though_it_has_parameters(mod):
    """kuo exposes two tunable parameters but is deliberately off without a
    large-scale moisture-convergence operator, so its flat pair is structural
    and its row is the no-convection baseline."""
    assert mod._structural_note(_rec("kuo", 5.7367, 5.7367, 2)) \
        == "inactive in a single column"


def test_a_scheme_that_moved_is_never_annotated(mod):
    """NON-VACUITY. The annotation must key off the measurement, not the
    name: if kuo is ever wired up for single columns and starts moving, the
    'inactive' label must disappear by itself."""
    assert mod._structural_note(_rec("kuo", 5.7367, 4.10, 2)) is None
    assert mod._structural_note(_rec("dca", 9.3, 8.0, 0)) is None


def test_an_ordinary_scheme_whose_search_failed_is_not_annotated(mod):
    """A flat pair with real parameters and no structural reason is a failed
    SEARCH and must read as one -- this is the case the grey styling must not
    swallow."""
    assert mod._structural_note(_rec("tiedtke", 3.21, 3.21, 16)) is None


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


def test_every_component_panel_is_drawn(mod):
    """The combined score is ~cloud_rmse/2 because the condensate term
    dominates the quadrature, so dropping the per-component panels would hide
    the temperature and humidity rankings the campaign is actually about."""
    keys = [k for k, _ in mod.COMPONENTS]
    assert keys == ["score", "T_rmse", "qv_rmse", "cloud_rmse", "precip_rmse"]


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
