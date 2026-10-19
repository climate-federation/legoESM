"""snow_node_column_replay: capture round trip and an offline on/off replay."""
import argparse
import importlib.util
import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "snow_node_column_replay",
    ROOT / "scripts/validate/amip_bias/snow_node_column_replay.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)

pytestmark = pytest.mark.skipif(
    not jax.config.jax_enable_x64, reason="needs JAX_ENABLE_X64=1")


def _setup(ncol=3):
    from legoesm import constants
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.multilayer_land import init_multilayer_land_state
    from legoesm.land.soil_grid import SoilGridConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    cfg = MultiLayerLandConfig(soil_grid=SoilGridConfig(n_layers=10, total_depth=3.0),
                               surface_scheme=TwoLeafCanopyConfig())
    cfg = cfg._replace(thermal=cfg.thermal._replace(enable_freeze_thaw=True))
    st = init_multilayer_land_state(ncol, cfg, T_init=constants.T_freeze - 2.0,
                                    theta_init=0.25)
    st = st._replace(snow_depth=jnp.array([20.0, 0.0, 20.0]))
    o = jnp.ones(ncol)
    p = 1.0e5 * o
    f = AtmToSurface(sw_down=20.0 * o, lw_down=220.0 * o, precip_total=0.0 * o,
                     precip_snow=0.0 * o, T_lowest=250.0 * o, q_lowest=0.0005 * o,
                     u_lowest=3.0 * o, v_lowest=0.0 * o, p_lowest=0.99 * p,
                     p_surface=p, rho_lowest=p / (constants.R_d * 250.0),
                     cos_zenith=0.1 * o, co2_ppmv=412.0 * o, has_radiation=o,
                     has_precipitation=o)
    return cfg, st, f


def test_split_join_round_trip_survives_none_leaves_and_pickle():
    cfg, st, f = _setup()
    tree = (st, f, (cfg, 1.0, 1800.0), {"lat": jnp.full(3, 1.0), "doy": None})
    seen = {}

    @jax.jit
    def g(s):
        td, stat, dyn, pos = rp.split_static(
            (s, f, (cfg, 1.0, 1800.0), {"lat": jnp.full(3, 1.0), "doy": None}),
            lambda v: isinstance(v, jax.core.Tracer))
        jax.debug.callback(lambda *v: seen.update(
            out=pickle.loads(pickle.dumps(
                (td, stat, [np.asarray(x) for x in v], pos)))),
            *dyn)
        return s.T_soil

    g(st)
    jax.effects_barrier()
    td, stat, vals, pos = seen["out"]
    back = rp.join_static(td, stat, vals, pos)
    assert back[3]["doy"] is None and back[2][0] == cfg
    np.testing.assert_array_equal(back[0].T_soil, np.asarray(tree[0].T_soil))
    assert back[0].T_snow is None


def test_replay_on_off_from_a_capture(tmp_path, capsys):
    cfg, st, f = _setup()
    args = (st, f, (cfg, 1.0, 1800.0),
            {"lat": jnp.deg2rad(jnp.array([60.0, 60.0, 20.0]))})
    td, stat, dyn, pos = rp.split_static(args, lambda v: isinstance(v, jax.Array))
    cap = {"run": "synthetic", "day": 330, "argv": [],
           "args": (td, stat, [np.asarray(v) for v in dyn], pos),
           "forcing": [jax.tree_util.tree_map(np.asarray, f)] * 48,  # one day
           "f_land_packed": np.ones(3)}
    p = tmp_path / "cap.pkl"
    p.write_bytes(pickle.dumps(cap))
    rp.replay(argparse.Namespace(capture=str(p), days=1, lat_band=(45.0, 70.0)))
    out = capsys.readouterr().out
    assert "selected 1 snow-covered land columns" in out
    row = [ln for ln in out.splitlines() if ln.startswith("  1 |")][0]
    gain_off, gain_on = (float(x) for x in row.split("|")[1].split()[:2])
    # Cold air over a frozen column: the insulated soil loses less heat.
    assert gain_on > gain_off
    assert "columns with on-arm skin > T_freeze under snow: 0" in out


def test_replay_refuses_a_partial_day(tmp_path):
    cfg, st, f = _setup()
    td, stat, dyn, pos = rp.split_static((st, f, (cfg, 1.0, 1800.0), {"lat": jnp.zeros(3)}),
                                         lambda v: isinstance(v, jax.Array))
    cap = {"run": "synthetic", "day": 330, "argv": [],
           "args": (td, stat, [np.asarray(v) for v in dyn], pos),
           "forcing": [jax.tree_util.tree_map(np.asarray, f)] * 2,
           "f_land_packed": np.ones(3)}
    p = tmp_path / "cap.pkl"
    p.write_bytes(pickle.dumps(cap))
    with pytest.raises(SystemExit, match="not one day"):
        rp.replay(argparse.Namespace(capture=str(p), days=1, lat_band=(45.0, 70.0)))


def _capture_file(tmp_path, cfg, st, f, n=48):
    td, stat, dyn, pos = rp.split_static(
        (st, f, (cfg, 1.0, 1800.0), {"lat": jnp.deg2rad(jnp.array([60.0, 60.0, 20.0]))}),
        lambda v: isinstance(v, jax.Array))
    p = tmp_path / "cap.pkl"
    p.write_bytes(pickle.dumps({
        "run": "synthetic", "day": 330, "argv": [],
        "args": (td, stat, [np.asarray(v) for v in dyn], pos),
        "forcing": [jax.tree_util.tree_map(np.asarray, f)] * n,
        "f_land_packed": np.ones(3)}))
    return p


def _budget_lines(out):
    inst = [ln for ln in out.splitlines() if ln.startswith("instrument:")][0]
    res = [ln for ln in out.splitlines() if ln.startswith("all land")][0]
    lin = float(inst.split("flux|")[1].split()[0])
    sub = float(inst.split("solve|")[1].split()[0])
    r = float(res.split("residual")[1].split()[0])
    return lin, sub, r


def test_soil_budget_control_closes_and_freeze_thaw_does_not(tmp_path, capsys):
    """Known answer: freeze/thaw off and no snow, the heat capacity is
    constant, so the budget closes to rounding; with freeze/thaw on near
    freezing it does not (the curtain linearisation), and the instrument's own
    checks (discrete flux identity, hand sub-stepping) still hold."""
    from legoesm import constants
    cfg, st, f = _setup()
    st = st._replace(T_soil=st.T_soil.at[:, :3].set(constants.T_freeze - 0.3))
    p = _capture_file(tmp_path, cfg, st, f)
    rp.soil_budget(argparse.Namespace(capture=str(p), lat_band=(45.0, 70.0),
                                      nq=200, control=True, substeps=0))
    lin, sub, r = _budget_lines(capsys.readouterr().out)
    assert lin < 1e-6 and sub < 1e-9 and abs(r) < 1e-6, (lin, sub, r)
    rp.soil_budget(argparse.Namespace(capture=str(p), lat_band=(45.0, 70.0),
                                      nq=200, control=False, substeps=0))
    lin, sub, r = _budget_lines(capsys.readouterr().out)
    assert lin < 1e-6 and sub < 1e-9 and abs(r) > 1e-3, (lin, sub, r)


def test_trace_summary_classes_and_runaway_count():
    tf = 273.15  # const-ok: synthetic known-answer value, passed in explicitly
    swe = np.array([1e-9, 5.0, 0.0, 1e-9, 5.0])
    land = np.array([True, True, True, True, False])
    T_snow = np.array([tf + 50, tf + 6, tf + 90, tf - 1, tf + 80])
    out = rp.trace_summary(T_snow, T_snow, T_snow, swe, land, tf)
    assert out["trace"]["n"] == 2 and out["trace"]["runaway"] == 1
    assert out["trace"]["T_snow_max"] == tf + 50
    assert out["healthy"]["n"] == 1 and out["healthy"]["runaway"] == 1
    # snow-free land (T_snow mirrors soil) is not a runaway; sea column ignored
    assert out["land"]["n"] == 4 and out["land"]["runaway"] == 2
    assert out["land"]["T_snow_max"] == tf + 50       # snow-free 90 K column excluded
    from legoesm import constants
    hot = rp.trace_summary(np.full(5, tf + constants.L_f / constants.c_pi + 1), T_snow,
                           T_snow, swe, land, tf, cohort=np.array([1, 0, 0, 0, 1], bool))
    assert hot["trace"]["past_meltout"] == 2 and hot["cohort"]["n"] == 1
    later = rp.trace_summary(T_snow - 60, T_snow, T_snow, swe, land, tf)
    day = rp.worst_of([out, later])
    assert day["trace"]["runaway"] == 1 and day["trace"]["T_snow_max"] == tf + 50


def test_plain_round_trip_and_field_drift_log():
    cfg, st, f = _setup()
    tree = {"state": st, "cfg": cfg, "lat": jnp.ones(3), "doy": None}
    plain = rp.to_plain(tree)
    log = {}
    back = rp.from_plain(plain, log)
    assert back["cfg"] == cfg and back["doy"] is None
    np.testing.assert_array_equal(back["state"].T_soil, np.asarray(st.T_soil))
    assert all(d == ([], []) for d in log.values())
    # an exporting checkout with an extra field, and one lacking a field
    plain["state"]["field_from_elsewhere"] = np.zeros(3)
    del plain["state"]["T_snow"]                      # has a default (None)
    log = {}
    back = rp.from_plain(plain, log)
    assert log[type(st).__qualname__] == (["field_from_elsewhere"], ["T_snow"])
    assert back["state"].T_snow is None
    del plain["state"]["snow_age"]                    # no default: must refuse
    with pytest.raises(TypeError):
        rp.from_plain(plain, {})


def test_export_then_trace_runs_both_arms(tmp_path, capsys):
    cfg, st, f = _setup()
    st = st._replace(snow_depth=jnp.array([1e-9, 0.0, 20.0]))
    p = _capture_file(tmp_path, cfg, st, f)
    q = tmp_path / "plain.pkl"
    rp.export(argparse.Namespace(capture=str(p), out=str(q)))
    ns = dict(plain=str(q), days=1, trace_swe_to=None)
    res = rp.trace(argparse.Namespace(**ns))
    assert set(res) == {"off", "on"} and len(res["on"]) == 1
    assert res["on"][0]["land"]["n"] == 3
    assert "  1 on  | land" in capsys.readouterr().out
    assert np.isnan(res["off"][0]["land"]["T_snow_max"])     # no node when OFF
    assert res["off"][0]["land"]["runaway"] == 0
    res = rp.trace(argparse.Namespace(**{**ns, "trace_swe_to": 10.0}))
    assert res["on"][0]["trace"]["n"] == 0
    assert res["on"][0]["cohort"]["n"] >= 1      # the perturbed column is still followed


def test_node_budget_sign():
    # stored 3600 J/m2 over 1800 s = 2 W/m2 against 1.5 W/m2 supplied: created
    assert rp.node_budget(1800.0, 1800.0, 1.0, 0.25, 0.25, 1800.0) == pytest.approx(0.5)
    assert rp.node_budget(0.0, 0.0, 1.0, 0.0, 0.0, 1800.0) == pytest.approx(-1.0)


def test_export_then_mechanism_records_every_final_solve(tmp_path, capsys):
    cfg, st, f = _setup()
    # freeze/thaw off: constant heat capacity, so both budgets must close
    cfg = cfg._replace(thermal=cfg.thermal._replace(enable_freeze_thaw=False))
    st = st._replace(snow_depth=jnp.array([1e-9, 0.0, 20.0]))
    p = _capture_file(tmp_path, cfg, st, f)
    q = tmp_path / "plain.pkl"
    rp.export(argparse.Namespace(capture=str(p), out=str(q)))
    ns = dict(plain=str(q), days=1, trace_swe_to=None, uncap=False, pin_melt=False, cover=False, no_freeze_thaw=False, nq=200)
    out = rp.mechanism(argparse.Namespace(**ns))
    assert out["on"]["G"].shape == out["off"]["G"].shape == (48, 3)
    for arm in ("off", "on"):       # instrument check: the solves conserve energy
        assert np.abs(out[arm]["res"]).max() < 1e-6
    assert "trace" in capsys.readouterr().out
    unc = rp.mechanism(argparse.Namespace(**{**ns, "uncap": True}))
    np.testing.assert_array_equal(unc["off"]["G"], out["off"]["G"])   # OFF untouched
    pin = rp.mechanism(argparse.Namespace(**{**ns, "pin_melt": True}))
    assert np.abs(pin["on"]["res"]).max() < 1e-6     # melt enthalpy closes the budget
    cov = rp.mechanism(argparse.Namespace(**{**ns, "cover": True, "pin_melt": True}))
    assert np.abs(cov["on"]["res"]).max() < 1e-6     # cover + melt enthalpy closes too
