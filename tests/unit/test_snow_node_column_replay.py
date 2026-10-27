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
