"""E2 scoring of CloudBench cases in the co-scientist experimenter (cosci_eval)."""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
from legoesm.atmosphere.les_suite import cloudbench as cb
from legoesm.atmosphere.les_suite.bridge import save_artifact
from legoesm.atmosphere.les_suite.candidate_eval import e2_nsteps, score_weights

from tests.atmosphere.les_suite.test_cloudbench_scm import case  # noqa: F401  (fixture)

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts" / "run"))
sys.path.insert(0, str(_REPO / "scripts" / "validate" / "les_suite"))
import cosci_eval as ce  # noqa: E402

_NAME = "cloudbench_s000_m07_amip"
_DT = 600.0
_NLEV = 12


def _stats(path, cover):
    import xarray as xr
    ds = xr.Dataset({"cloud_cover": (("site", "month", "experiment"),
                                     np.full((1, 1, 1), cover))},
                    coords={"site": [0], "month": [7], "experiment": ["amip"]})
    ds.to_netcdf(path)
    return path


def _held(path, sites=(0,), seed=cb.HELD_OUT_SEED):
    path.write_text(json.dumps({"held_out_sites": list(sites),
                                "fraction": cb.HELD_OUT_FRACTION, "seed": seed}))
    return path


@pytest.fixture()
def inputs(case, tmp_path):  # noqa: F811
    art, forcing = case
    adir = tmp_path / "art"
    adir.mkdir()
    save_artifact(art, adir / f"{_NAME}.npz")
    _held(adir / cb.HELD_OUT_FILE, sites=(1,))
    return adir / f"{_NAME}.npz", forcing, _stats(tmp_path / "stats.nc", 0.7)


def test_weights_cut_at_the_recorded_nudging_base(case):  # noqa: F811
    art, _ = case
    z = np.asarray(art.heights_m)
    w = np.asarray(score_weights(art, z))
    base = art.les_setup["forcing_recipe"]["nudge_z_i_m"]
    assert np.all(w[z > base] == 0.0) and np.all(w[z <= base] > 0.0)
    assert abs(w.sum() - 1.0) < 1e-14
    low = dataclasses.replace(art, les_setup={
        **art.les_setup, "forcing_recipe": {**art.les_setup["forcing_recipe"],
                                            "nudge_z_i_m": float(z[2])}})
    assert np.count_nonzero(np.asarray(score_weights(low, z))) == 3


@pytest.mark.parametrize("cover,regime", [(0.5, "stratocumulus"),
                                          (0.4999, "shallow_cumulus")])
def test_regime_from_the_statistics_cover(tmp_path, cover, regime):
    split = cb.case_split([_NAME], _stats(tmp_path / "s.nc", cover),
                          _held(tmp_path / "h.json", sites=(0,)))
    assert split == {_NAME: (regime, True)}
    split = cb.case_split([_NAME], tmp_path / "s.nc", _held(tmp_path / "h2.json", (3,)))
    assert split[_NAME][1] is False


def test_split_refuses_a_missing_case_and_another_draw(tmp_path):
    stats = _stats(tmp_path / "s.nc", 0.2)
    with pytest.raises(ValueError, match="not in the statistics"):
        cb.case_split(["cloudbench_s001_m07_amip"], stats, _held(tmp_path / "h.json"))
    with pytest.raises(ValueError, match="fraction/seed"):
        cb.case_split([_NAME], stats, _held(tmp_path / "h2.json", seed=1))


def test_an_unsplit_cloudbench_case_is_never_training_by_default():
    ce._CLOUDBENCH.update(forcing_dir=None, split={})
    with pytest.raises(SystemExit, match="without a regime"):
        ce._held_out(_NAME)
    with pytest.raises(SystemExit, match="without a regime"):
        ce._regime(_NAME)


def _e2(inputs, tmp_path, *extra):
    path, forcing, stats = inputs
    out = tmp_path / "e2.json"
    ce.main(["--stage", "E2", "--baseline-only", "smagorinsky", "--no-tune",
             "--dt", str(_DT), "--nlev", str(_NLEV), "--artifacts", str(path),
             "--cloudbench-forcing", str(forcing), "--cloudbench-stats", str(stats),
             "--out", str(out), *extra])
    return json.loads(out.read_text())


def test_e2_scores_a_cloudbench_case_through_the_gate(inputs, tmp_path):
    rec = _e2(inputs, tmp_path)
    res = rec["result"]
    assert res["split"][f"{_NAME}.npz"] == {"held_out": False, "regime": "stratocumulus"}
    s = res["arms"]["baseline:smagorinsky"]["default"][f"{_NAME}.npz"]
    assert s["conservation_passed"] and "failed" not in s, s
    assert set(s["per_var"]) == {"theta", "qt", "u", "v", "wtheta", "wqt"}
    assert np.isfinite(s["combined"])
    summ = res["summary"]["baseline:smagorinsky"]["default"]["train"]["stratocumulus"]
    assert summ == {"n": 1, "failed": 0, "failure_fraction": 0.0,
                    "mean_over_survivors": s["combined"]}
    assert rec["provenance"]["cloudbench"]["stats"]["sha256"]


def test_e2_needs_the_cloudbench_inputs(inputs, tmp_path):
    path, _, _ = inputs
    with pytest.raises(SystemExit, match="--cloudbench-forcing"):
        ce.main(["--stage", "E2", "--baseline-only", "smagorinsky", "--no-tune",
                 "--artifacts", str(path), "--out", str(tmp_path / "x.json")])


def test_a_gate_failure_is_recorded_and_left_out_of_the_mean(inputs, tmp_path,
                                                             monkeypatch):
    real = ce.check_conservation

    def failing(rec, dt):
        return real(rec, dt)._replace(passed=False, reason="planted")
    monkeypatch.setattr(ce, "check_conservation", failing)
    res = _e2(inputs, tmp_path)["result"]
    s = res["arms"]["baseline:smagorinsky"]["default"][f"{_NAME}.npz"]
    assert s["failed"] == "conservation: planted"
    summ = res["summary"]["baseline:smagorinsky"]["default"]["train"]["stratocumulus"]
    assert summ["failure_fraction"] == 1.0 and summ["mean_over_survivors"] is None


def test_cloudbench_runs_only_at_the_pi_step(inputs, tmp_path):
    with pytest.raises(SystemExit, match="--dt 600"):
        _e2(inputs, tmp_path, "--dt", "10")


def test_a_run_that_raises_is_recorded_failed(inputs, tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise FloatingPointError("planted blow-up")
    monkeypatch.setattr(ce.Arm, "run_prepared", boom)
    res = _e2(inputs, tmp_path)["result"]
    s = res["arms"]["baseline:smagorinsky"]["default"][f"{_NAME}.npz"]
    assert s == {"failed": "exception: FloatingPointError: planted blow-up"}
    summ = res["summary"]["baseline:smagorinsky"]["default"]["train"]["stratocumulus"]
    assert summ["failure_fraction"] == 1.0


def test_tuning_rollout_matches_the_scored_rollout(inputs):
    path, forcing, stats = inputs
    art = ce.load_artifact(path)
    ce.configure_cloudbench([(str(path), art)], forcing, stats)
    base, _ = ce._roster_base("smagorinsky")
    arm = ce.Arm(base)
    prep = arm.prepare(art, nlev=_NLEV, dt=_DT)
    n = e2_nsteps(art, _DT)
    scored, _ = arm.run_prepared(prep, n)
    tuned, _ = arm.run_with_turb(prep, art, n, base)
    assert ce.check_conservation(tuned, _DT).passed
    for k in ("theta_l", "q_t", "u", "v"):
        np.testing.assert_array_equal(getattr(tuned.inputs, k), getattr(scored.inputs, k))


def test_a_candidate_run_that_raises_is_recorded_failed(inputs, tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise FloatingPointError("planted blow-up")
    monkeypatch.setattr(ce.Arm, "run_prepared", boom)
    path, forcing, stats = inputs
    out = tmp_path / "c.json"
    ce.main(["--stage", "E2", "--candidate",
             str(_REPO / "scripts/validate/les_suite/cosci_candidates/hb_equivalent.py"),
             "--baseline", "holtslag_boville", "--heat-form", "theta", "--no-tune",
             "--dt", str(_DT), "--nlev", str(_NLEV), "--artifacts", str(path),
             "--cloudbench-forcing", str(forcing), "--cloudbench-stats", str(stats),
             "--out", str(out)])
    entry = next(iter(json.loads(out.read_text())["result"]["arms"].values()))
    want = {"failed": "exception: FloatingPointError: planted blow-up"}
    assert entry["default"][f"{_NAME}.npz"] == want
    assert entry["max_abs_diff_vs_baseline_rollout"][f"{_NAME}.npz"] == want
