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


# --- derivative-free tuning rounds ----------------------------------------------------

def _metas(name="ysu"):
    return ce._scheme_tunable_metas(ce._roster_base(name)[1], (1,))


def test_round_candidates_screen_then_draw_active_then_shrink():
    metas = _metas()
    r0 = ce.df_round_candidates(metas, 0, None)
    assert r0[0] == {} and len(r0) == 1 + 2 * len(metas)
    frozen = {metas[0].field: "x"}
    r1 = ce.df_round_candidates(metas, 1, {"frozen": frozen})
    assert len(r1) == ce.DF_RANDOM and all(metas[0].field not in c for c in r1)
    for m in metas[1:]:
        assert all(m.bounds[0] <= c[m.field] <= m.bounds[1] for c in r1)
    assert r1 == ce.df_round_candidates(metas, 1, {"frozen": frozen})
    inc = {m.field: float(m.bounds[1]) for m in metas}   # at the upper bound: clipped
    r2 = ce.df_round_candidates(metas, 2, {"frozen": frozen, "incumbent": inc})
    for m in metas[1:]:
        lo = m.bounds[1] - ce.DF_BOX_FRAC * (m.bounds[1] - m.bounds[0])
        assert all(lo <= c[m.field] <= m.bounds[1] for c in r2)
    with pytest.raises(SystemExit, match="no incumbent"):
        ce.df_round_candidates(metas, 2, {"frozen": {}, "incumbent": None})


def _shard(rows_by_case, cands, k=0, n=1, rnd=0, fields=("a", "b"), held=(),
           n_cases=None):
    return {"scheme": "ysu", "round": rnd, "shard": [k, n], "fields": list(fields),
            "settings": {"dt": 600.0, "nlev": 12, "git_sha": "x", "manifest": "m",
                         "n_cases": len(rows_by_case) if n_cases is None else n_cases},
            "case_sha256": {c: "h" + c for c in rows_by_case},
            "candidates": cands, "cases": rows_by_case,
            "split": {c: {"held_out": c in held, "regime": "shallow_cumulus"}
                      for c in rows_by_case}}


def _row(x, failed=False):
    return {"combined": x, **({"failed": "planted"} if failed else {})}


def test_aggregate_freezes_only_parameters_whose_probes_change_nothing(monkeypatch):
    monkeypatch.setattr(ce, "_scheme_tunable_metas", lambda *_: [])
    cands = [{}, {"a": 1}, {"a": 2}, {"b": 1}, {"b": 2}]
    rows = {"c1": [_row(1.0), _row(1.0), _row(1.0), _row(0.9), _row(1.1)],
            "c2": [_row(2.0), _row(2.0), _row(2.0), _row(2.0), _row(2.0)],
            "h1": [_row(5.0), _row(4.0), _row(4.0), _row(5.0), _row(5.0)]}
    st = ce.df_aggregate([_shard(rows, cands, held=("h1",))], None, None)
    assert set(st["frozen"]) == {"a"}            # held-out differences do not count
    assert st["incumbent_at"] == [0, 3]          # train mean 1.45 beats 1.5
    rows["c2"][1] = _row(2.0, failed=True)       # a failed probe is no evidence
    st = ce.df_aggregate([_shard(rows, cands, held=("h1",))], None, "survivor_mean")
    assert set(st["frozen"]) == set()


def test_aggregate_needs_the_pi_rule_once_a_training_run_fails(monkeypatch):
    monkeypatch.setattr(ce, "_scheme_tunable_metas", lambda *_: [])
    cands = [{}, {"a": 1}, {"a": 2}, {"b": 1}, {"b": 2}]
    rows = {"c1": [_row(3.0), _row(1.0, failed=True), _row(2.0), _row(2.5), _row(3.0)],
            "c2": [_row(3.0), _row(1.0), _row(2.0), _row(2.5, failed=True), _row(3.0)]}
    with pytest.raises(SystemExit, match="PI"):
        ce.df_aggregate([_shard(rows, cands)], None, None)
    pick = {s: ce.df_aggregate([_shard(rows, cands)], None, s)["incumbent_at"]
            for s in ce.DF_SELECT}
    assert pick == {"reject_failed": [0, 2], "fewest_failures": [0, 2],
                    "survivor_mean": [0, 1]}


def test_aggregate_refuses_a_missing_or_mixed_shard(monkeypatch):
    monkeypatch.setattr(ce, "_scheme_tunable_metas", lambda *_: [])
    cands = [{}, {"a": 1}, {"a": 2}, {"b": 1}, {"b": 2}]
    rows = {"c1": [_row(1.0)] * 5}
    with pytest.raises(SystemExit, match="every shard"):
        ce.df_aggregate([_shard(rows, cands, k=0, n=2, n_cases=2)], None, None)
    with pytest.raises(SystemExit, match="every shard"):
        ce.df_aggregate([_shard(rows, cands, k=0, n=2, n_cases=2),
                         _shard({"c2": [_row(1.0)] * 5}, cands[:4] + [{"b": 3}], k=1,
                                n=2, n_cases=2)], None, None)
    with pytest.raises(SystemExit, match="two shards"):
        ce.df_aggregate([_shard(rows, cands, k=0, n=2, n_cases=2),
                         _shard(rows, cands, k=1, n=2, n_cases=2)], None, None)
    with pytest.raises(SystemExit, match="the list has"):
        ce.df_aggregate([_shard(rows, cands, n_cases=2)], None, None)


def test_aggregate_chains_rounds_of_one_scheme_and_rule(monkeypatch):
    monkeypatch.setattr(ce, "_scheme_tunable_metas", lambda *_: [])
    cands = [{}, {"a": 1}, {"a": 2}, {"b": 1}, {"b": 2}]
    rows = {"c1": [_row(1.0), _row(1.0), _row(1.0), _row(0.5), _row(2.0)]}
    s0 = ce.df_aggregate([_shard(rows, cands)], None, None)
    r1 = {"c1": [_row(0.4), _row(0.7)]}
    with pytest.raises(SystemExit, match="needs the state"):
        ce.df_aggregate([_shard(r1, [{"b": 3}, {"b": 4}], rnd=1)], None, None)
    with pytest.raises(SystemExit, match="needs the state"):
        ce.df_aggregate([_shard(r1, [{"b": 3}, {"b": 4}], rnd=2)], s0, None)
    with pytest.raises(SystemExit, match="needs the state"):
        ce.df_aggregate([_shard(r1, [{"b": 3}, {"b": 4}], rnd=1)], s0, "survivor_mean")
    with pytest.raises(SystemExit, match="needs the state"):
        ce.df_aggregate([_shard(rows, cands)], s0, None)        # round 0 twice
    s1 = ce.df_aggregate([_shard(r1, [{"b": 3}, {"b": 4}], rnd=1)], s0, None)
    assert set(s1["rounds"]) == {"0", "1"} and s1["incumbent_at"] == [1, 0]
    other = _shard(r1, [{"b": 3}, {"b": 4}], rnd=1)
    other["case_sha256"] = {"c1": "changed"}
    with pytest.raises(SystemExit, match="differ from the earlier"):
        ce.df_aggregate([other], s0, None)


def test_round_zero_runs_each_candidate_through_the_scored_path(inputs, tmp_path):
    path, forcing, stats = inputs
    out = tmp_path / "r0.json"
    common = ["--dt", str(_DT), "--nlev", str(_NLEV), "--cloudbench-forcing", str(forcing),
              "--cloudbench-stats", str(stats)]
    ce.main(["--stage", "E2", "--baseline-only", "smagorinsky", "--df-round", "0",
             "--artifacts", str(path), "--out", str(out), *common])
    res = json.loads(out.read_text())["result"]
    rows = res["cases"][f"{_NAME}.npz"]
    assert len(rows) == len(res["candidates"]) == 5
    ref = _e2(inputs, tmp_path)["result"]["arms"]["baseline:smagorinsky"]["default"]
    want = ref[f"{_NAME}.npz"]                                   # default == scored path
    assert rows[0]["conservation_passed"] and want["conservation_passed"]
    assert rows[0]["combined"] == pytest.approx(want["combined"], rel=1e-10)
    assert rows[1]["combined"] != rows[0]["combined"]           # overrides reach the run
    art = ce.load_artifact(path)
    base, _ = ce._roster_base("smagorinsky")
    turb = ce.apply_overrides_to_base(base, res["candidates"][1])
    scored = ce._case_scores(ce.Arm(base), [(str(path), art)], _NLEV, _DT, turb=turb)
    assert rows[1]["combined"] == pytest.approx(scored[f"{_NAME}.npz"]["combined"],
                                                rel=1e-10)
    st = tmp_path / "s0.json"
    ce.main(["--stage", "DFAGG", "--df-shards", str(out), "--out", str(st)])
    state = json.loads(st.read_text())
    assert state["round"] == 0 and state["incumbent"] is not None


def test_shards_partition_the_cases():
    paths = [f"/a/c{i:02d}.npz" for i in (5, 1, 3, 0, 4, 2, 6)]
    parts = [ce.df_shard_paths(paths, (k, 3)) for k in range(3)]
    assert sorted(p for part in parts for p in part) == sorted(paths)
    assert parts[0] == ["/a/c00.npz", "/a/c03.npz", "/a/c06.npz"]
    with pytest.raises(SystemExit):
        ce.df_shard_paths(paths, (3, 3))
    assert ce.df_shard_paths(["/a/z.npz", "/b/a.npz"], (0, 2)) == ["/b/a.npz"]
    with pytest.raises(SystemExit, match="share a file name"):
        ce.df_shard_paths(["/a/x.npz", "/b/x.npz"], (0, 1))
    with pytest.raises(SystemExit, match="k/n"):
        ce.parse_shard("0")


def test_aggregate_refuses_shards_that_are_not_every_nth_case(monkeypatch):
    monkeypatch.setattr(ce, "_scheme_tunable_metas", lambda *_: [])
    cands = [{}, {"a": 1}, {"a": 2}, {"b": 1}, {"b": 2}]
    swapped = [_shard({"c2": [_row(1.0)] * 5}, cands, k=0, n=2, n_cases=2),
               _shard({"c1": [_row(1.0)] * 5}, cands, k=1, n=2, n_cases=2)]
    with pytest.raises(SystemExit, match="every n-th"):
        ce.df_aggregate(swapped, None, None)


def test_a_round_refuses_a_state_from_another_setup_before_running():
    state = {"scheme": "ysu", "round": 0,
             "settings": {"dt": 10.0, "nlev": 32, "manifest": "m", "git_sha": "x"}}
    with pytest.raises(SystemExit, match="another dt"):
        ce.stage_df_round("ysu", 1, state, [], (0, 1), 32, 600.0, ("m", 0))


def test_a_non_finite_training_objective_ranks_last(monkeypatch):
    monkeypatch.setattr(ce, "_scheme_tunable_metas", lambda *_: [])
    cands = [{}, {"a": 1}, {"a": 2}, {"b": 1}, {"b": 2}]
    rows = {"c1": [_row(float("nan")), _row(2.0), _row(3.0), _row(4.0), _row(5.0)]}
    assert ce.df_aggregate([_shard(rows, cands)], None, None)["incumbent_at"] == [0, 1]
