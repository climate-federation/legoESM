"""Round-2 MOST harness: the screen rejects planted defects, Elo bookkeeping, coherent permutation, fit recovery."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import round2 as r2  # noqa: E402
import round2_elo as elo  # noqa: E402
import round2_selfcorr as sc  # noqa: E402

DYER_M = {"stable": "1 + 5*z", "unstable": "(1 + 16*x)**-0.25", "params": {}}


def _screen(stable, unstable, kind="m"):
    return r2.screen({"stable": stable, "unstable": unstable, "params": {}}, [], kind)


def test_screen_passes_dyer_and_slope_jump_at_zero():
    s = r2.screen(DYER_M, [], "m")
    assert s["pass"], s  # slope 5 vs 4: a jump at zeta = 0 is allowed
    assert _screen("1.2 + 5*z", "1.2*(1 + 16*x)**-0.5", "h")["pass"]


@pytest.mark.parametrize("stable,unstable,failed", [
    ("1 - 0.5*z", "(1 + 16*x)**-0.25", "stable_finite_positive"),           # negative phi
    ("1 + 3*z*jnp.exp(-z)", "(1 + 16*x)**-0.25", "stable_monotone"),         # turnover
    ("1 + 5*z + 2*jnp.sqrt((z - 1)**2 + 1e-12)", "(1 + 16*x)**-0.25", "stable_c1"),  # interior kink at z = 1
    ("1 + 5*jnp.sqrt(z)", "(1 + 16*x)**-0.25", "stable_finite_slope_at_0"),  # singular neutral slope
    ("1.1 + 5*z", "(1 + 16*x)**-0.25", "continuous_at_0"),                  # jump in phi at 0
])
def test_screen_rejects_planted_defects(stable, unstable, failed):
    s = _screen(stable, unstable)
    assert not s[failed] and not s["pass"], s


def test_screen_rejects_too_many_params():
    spec = {"stable": "1 + a*z + b*z/(1 + z/c) + d*z/(1 + z)", "unstable": "(1 + 16*x)**-0.25",
            "params": {k: [1, 0, 2] for k in "abcd"}}
    assert not r2.screen(spec, [1, 1, 1, 1], "m")["pass"]


def test_fit_recovers_known_parameters():
    rng = np.random.default_rng(0)
    spec = {"stable": "1 + a*z/(1 + z/zc)", "unstable": "(1 + 16*x)**-0.25", "params": {"a": [3, 0, 20], "zc": [1, 0.01, 20]}}
    data = {}
    for s in range(3):
        z = np.r_[rng.uniform(0.01, 5, 400), -rng.uniform(0.01, 5, 400)]
        ph = np.where(z > 0, 1 + 4.0 * z / (1 + z / 0.7), (1 - 16 * np.minimum(z, 0)) ** -0.25)  # exact: a 4.0, zc 0.7
        data[f"s{s}"] = {"m": {0: (z, ph), 1: (z, ph)}, "h": {0: (np.empty(0),) * 2, 1: (np.empty(0),) * 2}}
    p, diag = r2.fit(spec, "m", data, 30)
    assert diag["success"] and abs(p[0] - 4.0) < 1e-5 and abs(p[1] - 0.7) < 1e-5, (p, diag)


def test_elo_update_and_missing_verdict():
    state = {"ratings": {"m_stable": {"A": 1200.0, "B": 1200.0}}, "games": []}
    elo.update(state, [["m_stable", "A", "B"]], [{"G0": "A"}, {"G0": "A"}])
    assert state["ratings"]["m_stable"]["A"] == pytest.approx(1216.0)
    elo.update(state, [["m_stable", "A", "B"]], [{"G1": "A"}, {"G1": "B"}])  # split = draw
    assert state["games"][-1][-1] == 0.5
    with pytest.raises(ValueError):
        elo.update(state, [["m_stable", "A", "B"]], [{"G2": "A"}, {}])
    with pytest.raises(ValueError):
        elo.update(state, [["m_stable", "A", "B"]], [])


def test_permutation_is_coherent_and_rescales_u_star():
    t = np.array([0.0, 0.0, 1800.0, 1800.0])  # two times, two level pairs each
    base = {"zeta": np.array([0.5, 1.0, 0.5, 1.0]), "phi": np.array([2.0, 3.0, 2.0, 3.0]), "ustar": np.array([0.2, 0.2, 0.4, 0.4]),
            "h": np.array([-20.0, -20.0, -20.0, -20.0]), "rho": np.ones(4), "theta": np.full(4, 280.0),
            "inv_l": np.array([0.1, 0.1, 0.1, 0.1]), "z2": np.ones(4), "time": t}
    rawd = {("s", "m", 0): base}
    for seed in range(10):
        out = sc.permute(rawd, "ustar", np.random.default_rng(seed))["s"]["m"][0]
        z, ph = out
        swapped = not np.allclose(z, base["zeta"])
        if swapped:  # time 0 took u* 0.4: zeta / 8, phi_m / 2 for both of its level pairs
            assert np.allclose(z, [0.0625, 0.125, 4.0, 8.0]) and np.allclose(ph, [1.0, 1.5, 4.0, 6.0])
            break
    else:
        pytest.fail("permutation never swapped the two times")


def test_neutral_point_evaluates_to_phi_at_zero():
    f, _, _ = r2.build(DYER_M)
    assert float(f(0.0, [])) == pytest.approx(1.0, abs=1e-12)  # was the unstable form at x = 1 (0.4925)


def test_cache_check_fails_closed():
    bench = {"global_fit": {"J3": {"a0": 4.0, "a_inf": 0.5, "zc": 1.0, "pr0": 1.0}}, "sites": {"s0": {"offset_corrected": {}}}}
    assert r2.check({}, bench, 30) == np.inf  # nothing compared
    rng = np.random.default_rng(0)
    z = np.r_[rng.uniform(0.01, 5, 400), -rng.uniform(0.01, 5, 400)]
    ph = np.where(z > 0, 1 + 5 * z, (1 - 16 * np.minimum(z, 0)) ** -0.25) * 1.1
    data = {"s0": {"m": {0: (z, ph), 1: (z, ph)}, "h": {0: (np.empty(0),) * 2, 1: (np.empty(0),) * 2}}}
    assert r2.check(data, bench, 30) == np.inf  # scored here, absent from the reference


def test_restrict_unstable_ignores_stable_samples():
    x = np.linspace(0.1, 8, 200)
    z = np.r_[-x, np.linspace(0.1, 5, 200)]
    ph = np.where(z < 0, (1 + 14 * np.abs(np.minimum(z, 0))) ** -0.4, 50.0)  # exact g 14, p 0.4
    spec = {"stable": "1 + g*z", "unstable": "(1 + g*x)**(-p)", "params": {"g": [16, 1, 60], "p": [0.45, 0.2, 0.7]}}  # g shared, so stable samples would pull it
    data = {"s": {"h": {0: (z, ph), 1: (z, ph)}, "m": {0: (np.empty(0),) * 2, 1: (np.empty(0),) * 2}}}
    p, diag = r2.fit(spec, "h", data, 30, restrict="unstable")
    assert diag["success"] and np.allclose(p, [14, 0.4], rtol=1e-5), p
    p_all, _ = r2.fit(spec, "h", data, 30)
    assert not np.allclose(p_all, [14, 0.4], rtol=1e-3)


def test_main_passes_candidate_restrict_to_fit(tmp_path, monkeypatch):
    import json
    seen = []

    def fake_fit(spec, kind, data, min_count, restrict=None):
        seen.append(restrict)
        raise RuntimeError("stop")
    monkeypatch.setattr(r2, "fit", fake_fit)
    np.savez(tmp_path / "s.npz", **{"a|h|0|zeta": np.ones(1), "a|h|0|phi": np.ones(1)})
    (tmp_path / "b.json").write_text("{}")
    cands = [{"name": "u", "kind": "h", "restrict": "unstable", "stable": "1", "unstable": "1", "params": {}},
             {"name": "n", "kind": "h", "stable": "1", "unstable": "1", "params": {}}]
    (tmp_path / "c.json").write_text(json.dumps(cands))
    monkeypatch.setattr(sys, "argv", ["round2", "--samples", str(tmp_path / "s.npz"), "--bench", str(tmp_path / "b.json"),
                                      "--candidates", str(tmp_path / "c.json"), "--out", str(tmp_path / "o.json")])
    r2.main()
    assert seen == ["unstable", None]
