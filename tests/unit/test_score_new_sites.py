"""Checks for the out-of-sample scoring of towers added after benchmark v5."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import jax.numpy as jnp  # noqa: E402

import candidate_formulations as cf  # noqa: E402
import round2 as r2  # noqa: E402
import score_new_sites as sn  # noqa: E402


def test_frozen_j3_spec_matches_the_benchmark_j3():
    p = {"a0": 4.03, "a_inf": 0.47, "zc": 1.22, "pr0": 0.9}
    zeta = jnp.asarray(np.concatenate([-np.logspace(1, -4, 40), np.logspace(-4, 1, 40)]))
    ref = cf.make("J3", [p[k] for k in ("a0", "a_inf", "zc", "pr0")])
    for i, k in enumerate("mh"):
        spec = sn.j3_specs(p)[k]
        f, _, _ = r2.build(spec)
        got = np.asarray(f(zeta, jnp.asarray([v[0] for v in spec["params"].values()])))
        want = np.asarray([ref[i](z) for z in zeta])
        assert got == pytest.approx(want, rel=1e-12)


def _bench(v):
    return {"standard": ["dyer1974"], "sites": {"s": {arm: {"m_stable": {"dyer1974": {"gradient": {"log_rmse": v}}}}
                                                     for arm in ("offset_corrected", "uncorrected")}}}


def test_unchanged_flags_any_change_and_a_score_on_one_side_only():
    assert sn.unchanged(_bench(0.3), _bench(0.3)) == (0.0, 2)
    assert sn.unchanged(_bench(0.3), _bench(0.33))[0] == pytest.approx(0.1)
    assert sn.unchanged(_bench(0.3), _bench(np.nan))[0] == np.inf


def test_unchanged_counts_a_panel_present_only_in_v6():
    v6 = _bench(0.3)
    v6["sites"]["s"]["offset_corrected"]["h_stable"] = {"dyer1974": {"gradient": {"log_rmse": 0.2}}}
    assert sn.unchanged(_bench(0.3), v6)[0] == np.inf


def test_rank_row_ranks_against_finite_standards_only():
    r = sn.rank_row(0.2, np.nan, {"dyer1974": 0.3, "b": 0.1, "c": np.nan})
    assert r["rank_J3_v5"] == 2 and r["rank_J3_v6"] is None and r["n_standard"] == 2
    assert r["best_standard"] == "b" and r["J3_v5_minus_dyer"] == pytest.approx(-0.1)


def test_missing_from_cache_names_a_benchmark_score_the_cache_lacks():
    rng = np.random.default_rng(1)
    z = np.concatenate([rng.uniform(0.05, 1, 40), -rng.uniform(0.05, 1, 40)])
    data = {"s": {k: {0: (z, np.ones_like(z)), 1: (z, np.ones_like(z))} for k in "mh"}}
    bench = {"sites": {"s": {"offset_corrected": {p: {"dyer1974": {"gradient": {"log_rmse": 0.1}}} for p in sn.PANELS}},
                       "t": {"offset_corrected": {"m_stable": {"dyer1974": {"gradient": {"log_rmse": 0.1}}}}}}}
    assert sn.missing_from_cache(data, bench, {"s"}, 30) == []
    assert sn.missing_from_cache(data, bench, {"s", "t"}, 30) == [("m_stable", "t")]


def test_unchanged_reference_must_cover_every_v5_site():
    v5 = {"sites": {"a": {}, "b": {}}}
    assert sn.same_sites({"sites": {"a": {}, "b": {}}}, v5)
    with pytest.raises(SystemExit):
        sn.same_sites({"sites": {"a": {}}}, v5)
