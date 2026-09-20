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
    d = {"cf": np.full((3, 4), 0.3)}
    monkeypatch.setattr(s0, "rh_cover", lambda d, exp: np.full((3, 4), 0.3))
    with pytest.raises(SystemExit):
        s0.check_provenance("ctl", d, {})
    monkeypatch.setattr(s0, "rh_cover", lambda d, exp: np.full((3, 4), 0.1))
    assert s0.check_provenance("ctl", d, {}) == 0.0


def test_arm_without_moments_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(s0.rb, "ROOT", str(tmp_path))
    (tmp_path / "arm").mkdir()
    np.savez(tmp_path / "arm" / "checkpoint_day_0045.npz",
             T=np.full((3, 4), 250.0), physstate_cloud_fraction=np.zeros((3, 4)),
             trc_q_v=np.zeros((3, 4)), p_s=np.full(3, 1e5), meta_vgrid=np.zeros((2, 5)),
             physstate_col_index=np.arange(3), physstate_clubb_moments=np.zeros((3, 1, 1)))
    with pytest.raises(SystemExit, match="clubb_moments"):
        s0.load_day("arm", 45, 3, require_moments=True)
    assert s0.load_day("arm", 45, 3)["moments"] is None


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
        (d / "run_manifest.json").write_text(json.dumps(
            {"command_line": "run_amip.py --restart-from /seed/checkpoint_day_0040.npz"}))
        for day in days:
            T = np.full((ncol, nlev), 240.0)
            cf = np.full((ncol, nlev), 0.2 + (arm_cf_bump if prog else 0.0))
            mom = np.full((ncol, 15, nlev + 1), 1e-7)
            if plant_nan and prog:
                mom[0, 3, 1] = np.nan
            np.savez(d / f"checkpoint_day_{day:04d}.npz", T=T, physstate_cloud_fraction=cf,
                     trc_q_v=np.zeros((ncol, nlev)), p_s=np.full(ncol, 1e5),
                     meta_vgrid=np.zeros((2, nlev + 1)), physstate_col_index=np.arange(ncol),
                     physstate_clubb_moments=mom)


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
    assert "+0.1000" in out


def test_end_to_end_planted_nan_fails(tmp_path, monkeypatch, capsys):
    _pair_on_disk(tmp_path, arm_cf_bump=0.1, plant_nan=True)
    _fake_mesh(monkeypatch, tmp_path)
    s0.main(["--control", "ctl", "--arm", "prog", "--days", "45", "46"])
    assert "VERDICT: FAIL" in capsys.readouterr().out


def test_unregistered_pair_is_refused(tmp_path, monkeypatch):
    _pair_on_disk(tmp_path)
    _fake_mesh(monkeypatch, tmp_path)
    exp = json.loads((tmp_path / "prog" / "experiment_config.json").read_text())
    exp["cloud_rh_crit"] = 0.9                        # a second differing field
    (tmp_path / "prog" / "experiment_config.json").write_text(json.dumps(exp))
    with pytest.raises(SystemExit, match="not the registered pair"):
        s0.main(["--control", "ctl", "--arm", "prog", "--days", "45"])
