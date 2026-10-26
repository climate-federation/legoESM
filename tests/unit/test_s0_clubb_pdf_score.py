"""s0_clubb_pdf_score.py: the cold-layer mean and the moment alarms are non-vacuous."""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

_TOOL = (pathlib.Path(__file__).resolve().parents[2]
         / "scripts" / "validate" / "amip_bias" / "s0_clubb_pdf_score.py")
_spec = importlib.util.spec_from_file_location("s0_score", _TOOL)
s0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s0)


def _grid():
    lat = np.array([80.0, 80.0, 40.0])          # two polar cells, one mid-latitude
    area = np.array([1.0, 3.0, 1.0])
    T = np.full((3, 4), 260.0)
    T[:, 2:] = 240.0                             # lowest two layers cold
    return lat, area, T


def test_cold_cap_mean_weights_by_area_and_masks_warm_layers():
    lat, area, T = _grid()
    f = np.zeros((3, 4))
    f[0, 2:] = 1.0                               # cell 0 cold layers cloudy
    f[1, :2] = 1.0                               # cell 1 only WARM layers cloudy -> masked
    f[2, :] = 1.0                                # mid-latitude -> masked
    mean, n = s0.cold_cap_mean(f, T, lat, area, 75.0, 253.0)
    assert n == 4
    assert mean == pytest.approx(1.0 * 2 / (2 * 1.0 + 2 * 3.0))


def test_cold_cap_mean_refuses_empty_mask():
    lat, area, T = _grid()
    with pytest.raises(SystemExit):
        s0.cold_cap_mean(np.zeros_like(T), T, lat, area, 75.0, 100.0)


def _packed(ncol, nlev, **fields):
    from legoesm.atmosphere.physics.turbulence.clubb import (
        CLUBBMomentState, pack_clubb_moments)
    import jax.numpy as jnp
    kw = {}
    for name in CLUBBMomentState._fields:
        nz = nlev if name in ("rtm", "thlm", "um", "vm", "wp3") else nlev + 1
        quiet = 1e-6 if name == "rtp2" else 0.1        # every field below its alarm
        kw[name] = jnp.asarray(fields.get(name, np.full((ncol, nz), quiet)))
    return np.asarray(pack_clubb_moments(CLUBBMomentState(**kw)))


def test_moment_alarm_fires_and_is_silent_when_quiet():
    lat, area, T = _grid()
    quiet = _packed(3, 4)
    res, alarms = s0.moment_stats(quiet, T, lat, area, 75.0, 253.0)
    assert alarms == []
    assert res["wp2"][0] == pytest.approx(0.1)
    hot = np.full((3, 5), 0.1); hot[1, 0] = 250.0   # zm index 0 = surface
    loud = _packed(3, 4, thlp2=hot)
    _, alarms = s0.moment_stats(loud, T, lat, area, 75.0, 253.0)
    assert alarms and alarms[0].startswith("thlp2 |max| 250")


def test_non_finite_moment_is_an_alarm():
    lat, area, T = _grid()
    bad = np.full((3, 4), 0.1); bad[0, 1] = np.nan
    _, alarms = s0.moment_stats(_packed(3, 4, wp3=bad), T, lat, area, 75.0, 253.0)
    assert any("non-finite" in a for a in alarms)


def test_surface_layer_lands_top_down():
    # zm surface value must be averaged into the LOWEST top-down layer (index -1).
    lat, area, T = _grid()
    T[:] = 260.0; T[:, -1] = 240.0                # only the lowest layer is cold
    w = np.full((3, 5), 0.0); w[:, 0] = 2.0        # ascending zm index 0 = surface
    res, _ = s0.moment_stats(_packed(3, 4, wp2=w), T, lat, area, 75.0, 253.0)
    assert res["wp2"][0] == pytest.approx(1.0)     # 0.5*(2+0) on the surface layer


def test_provenance_refuses_rh_cover(monkeypatch):
    d = {"cf": np.full((3, 4), 0.3), "trc_q_c": np.zeros((3, 4)), "trc_q_i": np.zeros((3, 4))}
    monkeypatch.setattr(s0, "rh_cover", lambda d, exp: np.full((3, 4), 0.3))
    with pytest.raises(SystemExit, match="grid-scale cover"):
        s0.check_provenance("ctl", d, {})
    monkeypatch.setattr(s0, "rh_cover", lambda d, exp: np.full((3, 4), 0.1))
    assert s0.check_provenance("ctl", d, {})["global"] == 0.0
    # clear points do not count: identical zeros with ONE differing cloudy point passes
    d["cf"][:] = 0.0; d["cf"][0, 0] = 0.5
    monkeypatch.setattr(s0, "rh_cover", lambda d, exp: np.zeros((3, 4)))
    assert s0.check_provenance("ctl", d, {})["global"] == 0.0
    # saturated points do not count either: an overcast PDF arm equal to a
    # saturated grid-scale cover must NOT be called grid-scale cover
    d["cf"][:] = 1.0; d["cf"][1, 1] = 0.7
    sat = np.ones((3, 4)); sat[1, 1] = 0.5
    monkeypatch.setattr(s0, "rh_cover", lambda d, exp: sat)
    assert s0.check_provenance("ctl", d, {})["global"] == 0.0
    mask = np.zeros((3, 4), bool); mask[0, :] = True         # mask fully saturated
    assert np.isnan(s0.check_provenance("ctl", d, {}, mask=mask)["scoring mask"])
    monkeypatch.setattr(s0, "rh_cover", lambda d, exp: np.ones((3, 4)))
    with pytest.raises(SystemExit, match="anywhere"):
        s0.check_provenance("ctl", d, {})
    # equality only OUTSIDE the scoring mask passes globally but aborts inside it
    d["cf"][:] = 0.3; d["cf"][2, :] = 0.6
    monkeypatch.setattr(s0, "rh_cover", lambda d, exp: np.full((3, 4), 0.3))
    mask = np.zeros((3, 4), bool); mask[:2, :] = True
    with pytest.raises(SystemExit, match="scoring mask"):
        s0.check_provenance("ctl", d, {}, mask=mask)
    monkeypatch.setattr(s0, "rh_cover", lambda d, exp: np.full((3, 4), np.nan))
    with pytest.raises(SystemExit, match="not a valid field"):
        s0.check_provenance("ctl", d, {})


def test_rh_cover_hands_condensate_to_the_scheme(monkeypatch):
    import legoesm.atmosphere.physics.clouds.cloud_fraction as cfmod
    seen = {}

    class _P:
        cloud_fraction = np.zeros((3, 4))

    def fake(T, p_full, q_v, dp, cfg, **kw):
        seen.update(kw); seen["p_full"] = p_full; return _P()
    monkeypatch.setattr(cfmod, "compute_cloud_properties", fake)

    class _Cfg:
        convective_cloud = False
    cfg = _Cfg()
    monkeypatch.setattr(s0.cl, "resolved_cloud_config", lambda exp: cfg)
    d = {"T": np.full((3, 4), 250.0), "q_v": np.zeros((3, 4)), "p_s": np.full(3, 1e5),
         "vgrid": np.stack([np.linspace(0, 0.01, 5), np.linspace(0, 0.99, 5)]),
         "trc_q_c": np.full((3, 4), 1e-4), "trc_q_i": np.full((3, 4), 2e-5),
         "trc_N_c": None, "trc_N_i": None, "conv_precip": np.full(3, 1e-8)}
    s0.rh_cover(d, {})
    assert seen["q_cloud"] is d["trc_q_c"] and seen["q_ice"] is d["trc_q_i"]
    assert seen["conv_precip"] is None                        # scheme does not want it
    assert np.all(np.diff(seen["p_full"], axis=1) > 0)        # top-down pressure
    cfg.convective_cloud = True
    s0.rh_cover(d, {})
    assert seen["conv_precip"] is d["conv_precip"]
    d["conv_precip"] = None
    with pytest.raises(SystemExit, match="conv_precip"):
        s0.rh_cover(d, {})


def test_arm_without_moments_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(s0.rb, "ROOT", str(tmp_path))
    (tmp_path / "arm").mkdir()
    np.savez(tmp_path / "arm" / "checkpoint_day_0045.npz",
             T=np.full((3, 4), 250.0), physstate_cloud_fraction=np.zeros((3, 4)),
             trc_q_v=np.zeros((3, 4)), p_s=np.full(3, 1e5),
             meta_vgrid=np.stack([np.linspace(0, 0.01, 5), np.linspace(0, 0.99, 5)]),
             physstate_col_index=np.arange(3), physstate_clubb_moments=np.zeros((3, 1, 1)),
             day=np.array(45.0))
    with pytest.raises(SystemExit, match="clubb_moments"):
        s0.load_day("arm", 45, 3, require_moments=True)
    assert s0.load_day("arm", 45, 3)["moments"] is None      # (ncol,1,1) placeholder = absent
    bad = dict(np.load(tmp_path / "arm" / "checkpoint_day_0045.npz"))
    bad["physstate_clubb_moments"] = np.zeros((3, 15, 4))     # present but malformed
    np.savez(tmp_path / "arm" / "checkpoint_day_0045.npz", **bad)
    with pytest.raises(SystemExit, match="malformed"):
        s0.load_day("arm", 45, 3)


def test_all_fifteen_fields_are_screened_for_non_finite():
    lat, area, T = _grid()
    packed = _packed(3, 4).copy()
    packed[2, 7, 3] = np.inf                        # a field the stats never report
    _, alarms = s0.moment_stats(packed, T, lat, area, 75.0, 253.0)
    assert any("packed moments non-finite" in a for a in alarms)


def _pair_on_disk(tmp_path, ncol=3, nlev=4, arm_cf_bump=0.0, plant_nan=False, days=(45, 46)):
    for run, prog in (("ctl", False), ("prog", True)):
        d = tmp_path / run
        d.mkdir()
        exp = {"clubb_prognostic": prog, "turbulence": "clubb", "grid": {"grid_type": "mpas"},
               "output": {"output_dir": str(d)}, "clubb_trop_cloud_top_press": 15000.0}
        (d / "experiment_config.json").write_text(json.dumps(exp))
        seed = tmp_path / "s0_seed" / "checkpoint_day_0040.npz"
        seed.parent.mkdir(exist_ok=True); seed.write_bytes(b"")
        (d / "run_manifest.json").write_text(json.dumps(
            {"run": {"command_line": f"run_amip.py --restart-from {seed}"}}))
        for day in days:
            T = np.full((ncol, nlev), 240.0)
            if prog:
                T[:, :2] = 260.0                      # arm warms its upper layers: its OWN
            cf = np.full((ncol, nlev), 0.2)           # mask would drop them; control mask keeps them
            if prog:
                cf[:, 2:] += arm_cf_bump              # excess only in the layers cold on BOTH
                cf[:, :2] += 3 * arm_cf_bump          # decoy excess where the arm is warm
            mom = np.full((ncol, 15, nlev + 1), 1e-7)
            if plant_nan and prog:
                mom[0, 3, 1] = np.nan
            np.savez(d / f"checkpoint_day_{day:04d}.npz", T=T, physstate_cloud_fraction=cf,
                     trc_q_v=np.zeros((ncol, nlev)), p_s=np.full(ncol, 1e5),
                     trc_q_c=np.zeros((ncol, nlev)), trc_q_i=np.zeros((ncol, nlev)),
                     meta_vgrid=np.stack([np.linspace(0, 0.01, nlev + 1),
                                          np.linspace(0, 0.99, nlev + 1)]),
                     physstate_col_index=np.arange(ncol), physstate_clubb_moments=mom,
                     day=np.array(float(day)))


def _fake_mesh(monkeypatch, tmp_path):
    monkeypatch.setattr(s0.rb, "ROOT", str(tmp_path))
    monkeypatch.setattr(s0.cl, "mesh_coords",
                        lambda exp: (np.array([80.0, 80.0, 40.0]), np.zeros(3), np.ones(3)))
    monkeypatch.setattr(s0, "rh_cover", lambda d, exp: np.zeros_like(d["cf"]))


def test_end_to_end_confirms_a_planted_cloud_excess(tmp_path, monkeypatch, capsys):
    _pair_on_disk(tmp_path, arm_cf_bump=0.1)
    _fake_mesh(monkeypatch, tmp_path)
    s0.main(["--control", "ctl", "--arm", "prog", "--days", "45", "46"])
    out = capsys.readouterr().out
    assert "EXPLORATORY VERDICT: CONFIRM" in out      # days are not the registered 45-50
    assert "+0.2000" in out                           # control mask: the +0.2 planted in the
    assert "+0.4000" not in out                       # cold-on-both layers, not the arm-mask decoy


def test_end_to_end_planted_nan_fails(tmp_path, monkeypatch, capsys):
    _pair_on_disk(tmp_path, arm_cf_bump=0.1, plant_nan=True)
    _fake_mesh(monkeypatch, tmp_path)
    assert s0.main(["--control", "ctl", "--arm", "prog", "--days", "45", "46"]) == 2
    assert "VERDICT: FAIL" in capsys.readouterr().out


def test_unregistered_pair_is_refused(tmp_path, monkeypatch):
    _pair_on_disk(tmp_path)
    _fake_mesh(monkeypatch, tmp_path)
    exp = json.loads((tmp_path / "prog" / "experiment_config.json").read_text())
    exp["cloud_rh_crit"] = 0.9                        # a second differing field
    (tmp_path / "prog" / "experiment_config.json").write_text(json.dumps(exp))
    with pytest.raises(SystemExit, match="not the registered pair"):
        s0.main(["--control", "ctl", "--arm", "prog", "--days", "45"])


def test_day_stamp_is_required_and_must_match(tmp_path, monkeypatch):
    monkeypatch.setattr(s0.rb, "ROOT", str(tmp_path))
    (tmp_path / "arm").mkdir()
    base = dict(T=np.full((3, 4), 250.0), physstate_cloud_fraction=np.zeros((3, 4)),
                trc_q_v=np.zeros((3, 4)), p_s=np.full(3, 1e5),
                meta_vgrid=np.stack([np.linspace(0, 0.01, 5), np.linspace(0, 0.99, 5)]),
                physstate_col_index=np.arange(3))
    np.savez(tmp_path / "arm" / "checkpoint_day_0045.npz", **base)
    with pytest.raises(SystemExit, match="day stamp"):
        s0.load_day("arm", 45, 3)
    np.savez(tmp_path / "arm" / "checkpoint_day_0045.npz", day=np.array(44.0), **base)
    with pytest.raises(SystemExit, match="stamped day"):
        s0.load_day("arm", 45, 3)


def test_pressure_must_increase_downward_in_every_column(tmp_path, monkeypatch):
    monkeypatch.setattr(s0.rb, "ROOT", str(tmp_path))
    (tmp_path / "arm").mkdir()
    # coefficients monotone at p_s = 1e5 but inverted for a low-pressure column
    a = np.array([0.0, 0.50, 0.30, 0.10, 0.0]); b = np.array([0.0, 0.00, 0.30, 0.70, 1.0])
    assert np.all(np.diff(a * 1e5 + b * 1e5) > 0) and not np.all(np.diff(a * 1e5 + b * 5e4) > 0)
    np.savez(tmp_path / "arm" / "checkpoint_day_0045.npz", day=np.array(45.0),
             T=np.full((3, 4), 250.0), physstate_cloud_fraction=np.zeros((3, 4)),
             trc_q_v=np.zeros((3, 4)), p_s=np.array([1e5, 1e5, 5e4]),
             meta_vgrid=np.stack([a, b]), physstate_col_index=np.arange(3))
    with pytest.raises(SystemExit, match="every column"):
        s0.load_day("arm", 45, 3)
