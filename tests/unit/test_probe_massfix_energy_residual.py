"""Direct test for ``scripts/validate/probe_massfix_energy_residual.py``
(review 2026-10-14 F15 V8).

The probe's numbers went into the vote, so its one piece of data handling is
pinned: re-keying a fu/composable-p3 lat-lon column checkpoint into main's
``latloncol_ckpt_v1`` schema keeps every value and reshapes the flattened
column fields row-major onto ``(n_lat, n_lon)`` (the ColumnMesh order).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "probe_massfix_energy_residual.py"


def _load():
    spec = importlib.util.spec_from_file_location("_massfix_probe", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_massfix_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_p3_checkpoint_is_rekeyed_without_changing_values(tmp_path):
    nlat, nlon, nlev = 3, 4, 2
    rng = np.random.default_rng(0)
    T = rng.random((nlat * nlon, nlev))
    p_s, phis = rng.random(nlat * nlon), rng.random(nlat * nlon)
    q_v, q_c = rng.random((nlat * nlon, nlev)), rng.random((nlat * nlon, nlev))
    u_f, v_f = rng.random((nlat, nlon + 1, nlev)), rng.random((nlat + 1, nlon, nlev))
    src = tmp_path / "ck.npz"
    np.savez(src, T=T, p_s=p_s, phis=phis, trc_q_v=q_v, trc_q_c=q_c,
             tracer_names=np.asarray(["q_v", "q_c"]), llc_u_face=u_f,
             llc_v_face=v_f)
    cfg = tmp_path / "experiment_config.json"
    cfg.write_text('{"dycore": {"mpas_div_damp4_scale": 1.0}}')
    out = np.load(_load()._convert_p3_latlon(str(src), str(tmp_path), str(cfg)))
    assert str(out["latloncol_schema"]) == "latloncol_ckpt_v1"
    # the div-4 stamp is the writing run's own scale (driver restore checks it)
    assert float(out["latloncol_div4_scale"]) == 1.0
    np.testing.assert_array_equal(out["latloncol_u"], u_f)
    np.testing.assert_array_equal(out["latloncol_v"], v_f)
    np.testing.assert_array_equal(out["latloncol_T"], T.reshape(nlat, nlon, nlev))
    np.testing.assert_array_equal(out["latloncol_p_s"], p_s.reshape(nlat, nlon))
    np.testing.assert_array_equal(out["latloncol_phis"], phis.reshape(nlat, nlon))
    # tracers sorted by name, slot i = i-th name
    assert list(out["latloncol_tracer_names"]) == ["q_c", "q_v"]
    np.testing.assert_array_equal(out["latloncol_q_0"], q_c.reshape(nlat, nlon, nlev))
    np.testing.assert_array_equal(out["latloncol_q_1"], q_v.reshape(nlat, nlon, nlev))
    # row-major: column k = (k // nlon, k % nlon)
    assert out["latloncol_p_s"][1, 2] == p_s[1 * nlon + 2]


def test_summarize_rates_tags_and_raw_net():
    """W/m^2 = J / (A dt); the restore rows (efix_mf) never enter the dycore
    efix mean; raw/net follow each lane's measurement point; missing rows
    are refused."""
    import pytest
    summarize = _load().summarize
    A, dt, steps = 2.0, 10.0, 2
    rows = [("efix", 40.0, 1e-5), ("massfix", -4.0, -2e-4), ("efix_mf", 4.0, 1e-6),
            ("efix", 60.0, 1e-5), ("efix_mf", 4.0, 1e-6), ("massfix", -4.0, -4e-4)]
    m = summarize(rows, A, dt, "mpas", steps)   # massfix measured BEFORE restore
    assert m["efix_mean_Wm2"] == pytest.approx(2.5)        # (40+60)/2/(2*10)
    assert m["massfix_raw_Wm2"] == pytest.approx(-0.2)
    assert m["massfix_net_Wm2"] == pytest.approx(0.0)
    assert m["dps_absmean_Pa"] == pytest.approx(3e-4)
    assert m["ratio_raw_over_efix"] == pytest.approx(-0.08)
    net_rows = [r if r[0] != "massfix" else ("massfix", 0.0, r[2]) for r in rows]
    ll = summarize(net_rows, A, dt, "latlon", steps)      # massfix AFTER restore
    assert ll["massfix_net_Wm2"] == pytest.approx(0.0)
    assert ll["massfix_raw_Wm2"] == pytest.approx(-0.2)
    with pytest.raises(RuntimeError, match="efix_mf: 1 rows"):
        summarize(rows[:-2] + rows[-1:], A, dt, "mpas", steps)


def test_p3_conversion_stamps_the_run_scale_and_refuses_a_missing_one(tmp_path):
    """The stamp follows the writing run's config (0.0 stays 0.0, not a
    default); a config without dycore.mpas_div_damp4_scale is refused."""
    import json
    import pytest
    nlat, nlon, nlev = 2, 3, 1
    src = tmp_path / "ck.npz"
    np.savez(src, T=np.zeros((nlat * nlon, nlev)), p_s=np.zeros(nlat * nlon),
             phis=np.zeros(nlat * nlon), trc_q_v=np.zeros((nlat * nlon, nlev)),
             tracer_names=np.asarray(["q_v"]),
             llc_u_face=np.zeros((nlat, nlon + 1, nlev)),
             llc_v_face=np.zeros((nlat + 1, nlon, nlev)))
    probe = _load()
    for scale in (0.0, 0.5):
        d = tmp_path / f"s{scale}"
        d.mkdir()
        cfg = d / "experiment_config.json"
        cfg.write_text(json.dumps({"dycore": {"mpas_div_damp4_scale": scale}}))
        out = np.load(probe._convert_p3_latlon(str(src), str(d), str(cfg)))
        assert float(out["latloncol_div4_scale"]) == scale
    for body in ({"dycore": {"dt": 450.0}}, {"grid": {}}):
        cfg = tmp_path / "missing.json"
        cfg.write_text(json.dumps(body))
        with pytest.raises(SystemExit, match="mpas_div_damp4_scale"):
            probe._convert_p3_latlon(str(src), str(tmp_path), str(cfg))
