"""Gates for the SCM-RCE derivative-free tuner's sampling and acceptance.

Two properties decide what a "tuned" parameter value MEANS:

* how candidates are drawn across a parameter's bounds — a linear draw over a
  range spanning decades never visits the bottom of it, so the tuned value is
  biased high by construction rather than by evidence; and
* which candidate is accepted — a comparison against a non-finite incumbent is
  always False, so a NaN would make the tuner silently report the defaults.

Both are cheap to get wrong and invisible in the output.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_campaign():
    path = REPO_ROOT / "scripts" / "run" / "run_scm_rce_campaign.py"
    spec = importlib.util.spec_from_file_location("scm_rce_campaign_tuner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def camp():
    return _load_campaign()


def _c(name, lo, hi):
    return SimpleNamespace(name=name, field=name, min_val=lo, max_val=hi,
                           transform="sigmoid")


# --------------------------------------------------------------------------- #
# scale selection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("lo,hi,want", [
    (1.0e-6, 1.0e-2, "log"),     # 4 decades — the case this exists for
    (1.0e-3, 1.0e-1, "log"),     # exactly 2 decades — at the threshold
    (0.1, 0.9, "linear"),        # sub-decade
    (1.0, 50.0, "linear"),       # 1.7 decades
    (-1.0, 1.0, "linear"),       # crosses zero: log is undefined
    (0.0, 1.0e3, "linear"),      # lo == 0: log is undefined
])
def test_sample_scale(camp, lo, hi, want):
    assert camp._sample_scale(_c("x", lo, hi)) == want


def test_interp_is_geometric_on_a_log_range(camp):
    c = _c("x", 1.0e-6, 1.0e-2)
    assert camp._interp(c, 0.0) == pytest.approx(1.0e-6)
    assert camp._interp(c, 1.0) == pytest.approx(1.0e-2)
    # midpoint is the GEOMETRIC mean, not the arithmetic one
    assert camp._interp(c, 0.5) == pytest.approx(1.0e-4, rel=1e-12)
    assert camp._interp(c, 0.5) != pytest.approx(0.5 * (1e-6 + 1e-2))


def test_interp_is_linear_on_a_narrow_range(camp):
    c = _c("x", 0.1, 0.9)
    assert camp._interp(c, 0.25) == pytest.approx(0.1 + 0.25 * 0.8)


# --------------------------------------------------------------------------- #
# candidate stream
# --------------------------------------------------------------------------- #

def test_one_at_a_time_sweep_uses_the_log_scale(camp):
    """NON-VACUITY: the sweep values must differ from the linear ones, and must
    land inside the decades a linear sweep would skip."""
    cs = [_c("k", 1.0e-6, 1.0e-2)]
    defaults = {"k": 1.0e-4}
    cands = list(camp._candidate_values(defaults, cs, n_eval=3, seed=0))
    assert cands[0] == defaults                      # first candidate = defaults
    swept = [c["k"] for c in cands[1:]]
    assert len(swept) == 2
    lin = [1e-6 + f * (1e-2 - 1e-6) for f in (0.25, 0.75)]
    for got, linear in zip(swept, lin):
        assert got != pytest.approx(linear)
        assert got < linear                          # log draws sit lower
    # 0.25 of four decades is 1e-5: a decade a linear sweep never reaches.
    assert swept[0] == pytest.approx(1.0e-6 * (1.0e4) ** 0.25, rel=1e-12)


def test_random_draws_stay_inside_bounds_on_both_scales(camp):
    cs = [_c("log_p", 1.0e-8, 1.0e-2), _c("lin_p", 0.2, 0.8)]
    defaults = {"log_p": 1.0e-5, "lin_p": 0.5}
    for cand in camp._candidate_values(defaults, cs, n_eval=40, seed=7):
        assert 1.0e-8 <= cand["log_p"] <= 1.0e-2
        assert 0.2 <= cand["lin_p"] <= 0.8


def test_a_longer_budget_is_a_superset_of_a_shorter_one(camp):
    """The campaign reports one budget but compares against another; that is
    only legitimate if the shorter stream is a PREFIX of the longer one."""
    cs = [_c("a", 1.0e-6, 1.0e-2), _c("b", 0.1, 0.9), _c("c", 1.0, 4.0)]
    defaults = {"a": 1.0e-4, "b": 0.5, "c": 2.0}
    short = list(camp._candidate_values(defaults, cs, n_eval=12, seed=20260810))
    long = list(camp._candidate_values(defaults, cs, n_eval=60, seed=20260810))
    assert len(short) == 12 and len(long) == 60
    for i, (s, l) in enumerate(zip(short, long)):
        assert s == pytest.approx(l), i


def test_log_sampling_covers_the_bottom_decade(camp):
    """The defect in one line: with linear draws, a 4-decade parameter almost
    never sees its lowest decade."""
    cs = [_c("k", 1.0e-6, 1.0e-2)]
    defaults = {"k": 1.0e-4}
    draws = [c["k"] for c in camp._candidate_values(defaults, cs, 200, seed=3)]
    bottom_decade = [v for v in draws if v < 1.0e-5]
    assert len(bottom_decade) > 20, (
        f"only {len(bottom_decade)}/200 draws below 1e-5 — the sampler is not "
        "on a log scale")


# --------------------------------------------------------------------------- #
# acceptance
# --------------------------------------------------------------------------- #

def _diag(camp, score, status="ok"):
    return camp.RunDiagnostics(
        label="x", config={}, status=status, reason="",
        T_rmse=score, qv_rmse=score, cloud_rmse=score, precip_rmse=score,
        score=score, drift_T_rmse_K=0.0, drift_qv_rmse=0.0,
        drift_qcond_rmse=0.0, T_profile=[300.0], qv_profile=[0.01],
        qcond_profile=[0.0],
    )


def test_a_nan_default_score_does_not_block_every_trial(camp, monkeypatch):
    """SYNTHETIC VIOLATION of the acceptance rule: with a NaN incumbent, every
    ``trial.score < best.score`` is False, so the tuner would report the
    DEFAULTS as tuned while silently discarding a better trial."""
    calls = {"n": 0}

    def fake_run_cached(cache, cfg, ref, *, label, **kwargs):
        calls["n"] += 1
        # first call is the default run
        return _diag(camp, float("nan") if calls["n"] == 1 else 0.5)

    monkeypatch.setattr(camp, "run_cached", fake_run_cached)
    base = camp.make_physics_config(convection="bechtold")
    _cfg, records, tuned = camp.tune_category_winner(
        "convection", base, ref=SimpleNamespace(), cache={},
        days=1.0, dt=600.0, analysis_days=1.0,
        require_equilibrium=False, require_realism=False,
        tune_evals=3, seed=1,
        equil_T_tol_K=1.0, equil_qv_tol=1.0, equil_qcond_tol=1.0,
    )
    assert math.isfinite(tuned.score), "tuner kept the NaN default"
    assert tuned.score == pytest.approx(0.5)
    assert records, "no parameter records written for a successful tune"


def test_a_non_finite_trial_is_never_accepted(camp, monkeypatch):
    calls = {"n": 0}

    def fake_run_cached(cache, cfg, ref, *, label, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _diag(camp, 2.0)               # finite default
        return _diag(camp, float("-inf"))          # a "better" but broken trial

    monkeypatch.setattr(camp, "run_cached", fake_run_cached)
    base = camp.make_physics_config(convection="bechtold")
    _cfg, _records, tuned = camp.tune_category_winner(
        "convection", base, ref=SimpleNamespace(), cache={},
        days=1.0, dt=600.0, analysis_days=1.0,
        require_equilibrium=False, require_realism=False,
        tune_evals=3, seed=1,
        equil_T_tol_K=1.0, equil_qv_tol=1.0, equil_qcond_tol=1.0,
    )
    assert tuned.score == pytest.approx(2.0), (
        "a -inf score was accepted as the best run")
