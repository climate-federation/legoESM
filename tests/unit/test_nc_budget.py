"""Known-answer controls for the pure helpers of the #2216 droplet-number budget
probe (scripts/validate/amip_bias/nc_budget.py) and its relaunch-line parser."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_D = Path(__file__).resolve().parents[2] / "scripts/validate/amip_bias"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _D / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nb = _load("nc_budget")


def test_aggregate_known_answer():
    # two 1-kg cells at rho 1 (2 m^3): 1 and 3 per mg -> 2 cm^-3; a sink of
    # 1e6 /kg/day in both -> -1 cm^-3/day, sink time 2 days
    mass, rho, mk = np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1), bool)
    n0 = np.array([[1e6], [3e6]])
    rows = {"sink": np.full((2, 1), -1e6 / 86400.0), "src": np.full((2, 1), 2e6 / 86400.0)}
    inv, r, tau_all, tau = nb.aggregate(rows, n0, mass, rho, mk)
    assert inv == pytest.approx(2.0)
    assert r["sink"] == pytest.approx(-1.0) and r["src"] == pytest.approx(2.0)
    assert tau_all == pytest.approx(48.0) and tau["sink"] == pytest.approx(48.0)
    assert tau["src"] == np.inf
    # the mask excludes a cell
    inv1, r1, _, _ = nb.aggregate(rows, n0, mass, rho, np.array([[True], [False]]))
    assert inv1 == pytest.approx(1.0) and r1["sink"] == pytest.approx(-1.0)


def test_incloud_known_answer():
    # 100 cm^-3 grid-mean at half cloud cover -> 200 in-cloud; the clear cell
    # (q_c below the threshold) is excluded
    n = np.full((3, 1), 1e8)
    cf = np.array([[0.5], [0.5], [0.0]])
    qc = np.array([[1e-5], [1e-5], [0.0]])
    ic = nb.incloud(n, cf, qc, np.ones((3, 1)), np.ones((3, 1), bool))
    assert ic["n_cells"] == 2 and ic["n_cf_zero"] == 0
    assert ic["grid_mean"] == pytest.approx(100.0) and ic["in_cloud"] == pytest.approx(200.0)
    assert ic["median_in_cloud"] == pytest.approx(200.0) and ic["cloud_fraction"] == pytest.approx(0.5)


def test_incloud_median_skips_zero_cloud_fraction():
    ic = nb.incloud(np.full((2, 1), 1e8), np.array([[0.5], [0.0]]), np.full((2, 1), 1e-5),
                    np.ones((2, 1)), np.ones((2, 1), bool))
    assert ic["n_cf_zero"] == 1 and ic["median_in_cloud"] == pytest.approx(200.0)


def test_incloud_refuses_nonfinite():
    with pytest.raises(SystemExit, match="non-finite N_c"):
        nb.incloud(np.array([[np.nan]]), np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)),
                   np.ones((1, 1), bool))


def test_masks_regions_and_layers():
    m = nb.masks(np.array([-50.0, 0.0]), np.array([[800.0, 500.0], [800.0, 500.0]]))
    np.testing.assert_array_equal(m[("SO 40-65S", "column")], [[True, True], [False, False]])
    np.testing.assert_array_equal(m[("SO 40-65S", "950-700 hPa")], [[True, False], [False, False]])
    np.testing.assert_array_equal(m[("global", "950-700 hPa")], [[True, False], [True, False]])


def test_closes_catches_a_missing_term_and_mean_calls_weights():
    a, b = np.array([1.0, -2.0]), np.array([0.5, 0.25])
    assert nb.closes([a, b], a + b)[0]
    assert not nb.closes([a], a + b)[0]
    # a complete miss in a tiny cell next to a large one is still caught
    big, tiny = np.array([1e6, 1e-9]), np.array([0.0, 1e-9])
    assert not nb.closes([big - tiny], big)[0]
    np.testing.assert_allclose(nb.mean_calls([a, b, b]), (a + 2 * b) / 3)
    with pytest.raises(SystemExit, match="GATE FAILED"):
        nb.gate(False, "x")


def test_figure_writes(tmp_path):
    res = {(rn, ln): (1.0, {"a": -1.0, "b": 2.0}, 24.0, {}) for rn in nb.REGIONS for ln in nb.LAYERS}
    nb.figure(res, tmp_path / "f.png", "t")
    assert (tmp_path / "f.png").stat().st_size > 0


def test_launch_argv_reads_monthly_baseline_lines(tmp_path, monkeypatch):
    H = _load("mpas_onestep_param_grad")
    (tmp_path / "arm").mkdir()
    (tmp_path / "arm" / "checkpoint_day_0030.npz").write_bytes(b"")
    (tmp_path / "arm" / "slurm-1.out").write_text(
        "[jan] ARM ngpu=4\n[jan] python -u scripts/run/run_amip.py --config x.yaml "
        "--output o --distributed --days 30\n")
    monkeypatch.setattr(H, "ROOT", tmp_path)
    argv = H.launch_argv("arm", 30, tmp_path / "out")
    assert argv[:2] == ["--config", "x.yaml"] and "--distributed" not in argv
    assert argv[argv.index("--days") + 1] == "1"


def test_terrain_override_reports_only_small_magnitude_drift(monkeypatch, capsys):
    from types import SimpleNamespace
    from legoesm.driver.model_driver import ModelDriver
    msg = {"text": ""}

    def refuse(self):
        raise RuntimeError(msg["text"])
    monkeypatch.setattr(ModelDriver, "_check_restart_terrain", refuse)
    nb.report_terrain_drift()                         # wraps the refusal patched above
    check = ModelDriver._check_restart_terrain

    def fake(drift, n_state=4):
        b = np.full(4, 5.0e4)                         # this run's terrain product
        return SimpleNamespace(_phis_data=b, state=SimpleNamespace(   # the checkpoint's
            phis=SimpleNamespace(data=np.full(n_state, 5.0e4) + drift)))
    msg["text"] = "restart: the checkpoint's terrain differs from this run's terrain product (max |d| x)"
    check(fake(5.0e4 * 1.4e-6))                       # measured size: reported
    assert "TERRAIN DRIFT 1.40e-06 of max phis" in capsys.readouterr().out
    with pytest.raises(RuntimeError):                 # above the cap: refused
        check(fake(5.0e4 * 2.0e-5))
    with pytest.raises(RuntimeError, match="max"):    # magnitude text, unequal shapes: refused
        check(fake(0.0, n_state=3))
    msg["text"] = "restart: the checkpoint's terrain has shape (3,), this run's terrain product (4,)"
    with pytest.raises(RuntimeError, match="has shape"):   # any other refusal: refused
        check(fake(0.0, n_state=3))


def test_spy_switches_off_graupel_riming_only(monkeypatch):
    from typing import NamedTuple
    import legoesm.atmosphere.physics.microphysics.integration as mi

    class Cfg(NamedTuple):
        publish_qc_budget: bool = False
        do_graupel_riming: bool = True
        do_snow_riming: bool = True

    class Out(NamedTuple):
        dN_c_dt: object = None
        nc_budget: object = None
        qc_budget: object = None
    seen = []

    def fn(*a, **kw):
        seen.append(a[8])
        return Out()
    for flag, want in ((True, True), (False, False)):
        monkeypatch.setattr(mi, "_get_microphysics_fn", lambda c: ("morrison", fn, Cfg()))
        nb.install_micro_spy([], graupel_riming=flag)
        _, spy, cfg = mi._get_microphysics_fn(None)
        spy(*range(8), cfg)
        assert seen[-1] == Cfg(publish_qc_budget=True, do_graupel_riming=want)
    # a scheme without the switch is refused, not silently run with riming on
    class NoSwitch(NamedTuple):
        publish_qc_budget: bool = False
    monkeypatch.setattr(mi, "_get_microphysics_fn", lambda c: ("x", fn, NoSwitch()))
    nb.install_micro_spy([], graupel_riming=False)
    with pytest.raises(SystemExit, match="no do_graupel_riming"):
        mi._get_microphysics_fn(None)


def test_compare_known_answer(tmp_path):
    # inventory 6 cm-3, sink time 3 h, activation +48 /day -> N* = 48 * 3 / 24 = 6
    names = np.array(["micro: activation", "micro: riming_graupel"])

    def save(path, rim):
        d = {}
        for rn in nb.REGIONS:
            for ln in nb.LAYERS:
                k = f"{rn}|{ln}"
                d[k] = np.array([6.0, 3.0, 48.0, rim])
                d[k + "|incloud"] = np.array([1, 10.0, 20.0, 20.0, 0.5])
                d[k + "|incloud_last"] = np.array([1, 11.0, 22.0, 22.0, 0.5])
        np.savez(path, row_names=names, dt=112.5, **d)
    save(tmp_path / "on.npz", -12.0)
    save(tmp_path / "off.npz", 0.0)
    out = nb.compare(tmp_path / "on.npz", tmp_path / "off.npz", tmp_path / "c.png")
    a = out["SO 40-65S|950-700 hPa"]
    assert a["on"]["n_star"] == pytest.approx(6.0) and a["on"]["net"] == pytest.approx(36.0)
    assert a["off"]["rows"]["micro: riming_graupel"] == 0.0
    assert a["on"]["incloud_first"] == 20.0 and a["on"]["incloud_last"] == 22.0
    assert (tmp_path / "c.png").stat().st_size > 0


def test_spy_reset_width_ab_runs_both_arms_on_the_same_inputs(monkeypatch):
    from typing import NamedTuple
    import legoesm.atmosphere.physics.microphysics.integration as mi

    class Cfg(NamedTuple):
        publish_qc_budget: bool = False
        mg2_drop_early_reset_log_width: float = 0.01

    class Out(NamedTuple):
        dq_c_dt: object = None
        dq_r_dt: object = None
        dq_s_dt: object = None
        dq_v_dt: object = None
        dN_r_dt: object = None
        nc_budget: object = None
        qc_budget: object = None
    seen = []

    def fn(*a, **kw):
        seen.append((a[:8], a[8]))
        w = a[8].mg2_drop_early_reset_log_width
        one = np.ones((1, 1))
        return Out(*(one * (1.0 + w),) * 4, dN_r_dt=one * 2.0,
                   qc_budget={"autoconversion": one * (0.5 + w), "riming_snow": one * 0.5})
    monkeypatch.setattr(mi, "_get_microphysics_fn", lambda c: ("morrison", fn, Cfg()))
    calls = []
    nb.install_micro_spy(calls, reset_width_ab=True)
    _, spy, cfg = mi._get_microphysics_fn(None)
    ret = spy(*range(5), np.full((1, 1), 2.0), 6, 7, cfg)
    assert [s[0] for s in seen] == [seen[0][0]] * 2           # same inputs
    assert [s[1] for s in seen] == [Cfg(True, 0.01), Cfg(True, 0.0)]
    assert ret.qc_budget is None and float(ret.dq_c_dt[0, 0]) == pytest.approx(1.01)
    assert calls[0]["width"] == 0.01 and calls[0]["hard"]["dN_r_dt"][0, 0] == 1.0
    rows = nb.reset_ab_rows(calls)
    t = nb.reset_ab_table(rows, np.ones((1, 1)), np.ones((1, 1), bool))
    assert t["q_c: autoconversion"] == pytest.approx((0.5 * 86400, 0.51 * 86400, 0.01 * 86400))
    assert t["dq_c_dt"][2] == pytest.approx(0.01 * 86400)
    # a budget that does not close is refused
    calls[0]["hard"]["qc"]["riming_snow"] = np.zeros((1, 1))
    with pytest.raises(SystemExit, match="GATE FAILED: hard: cloud-water budget"):
        nb.reset_ab_rows(calls)
    # a configured width of 0 makes the comparison vacuous
    calls[0]["width"] = 0.0
    with pytest.raises(SystemExit, match="vacuous"):
        nb.reset_ab_rows(calls)
    class NoField(NamedTuple):
        publish_qc_budget: bool = False
    monkeypatch.setattr(mi, "_get_microphysics_fn", lambda c: ("x", fn, NoField()))
    nb.install_micro_spy([], reset_width_ab=True)
    with pytest.raises(SystemExit, match="no mg2_drop_early_reset_log_width"):
        mi._get_microphysics_fn(None)


def test_spy_reset_width_ab_sedimentation_field(monkeypatch):
    """#2216: the sedimentation fall-speed reset width is switched instead, and
    the droplet-number rows (whole tendency, sedimentation) are recorded."""
    from typing import NamedTuple
    import legoesm.atmosphere.physics.microphysics.integration as mi

    class Cfg(NamedTuple):
        publish_qc_budget: bool = False
        mg2_drop_early_reset_log_width: float = 0.01
        mg2_drop_sed_reset_log_width: float = 0.01

    class Out(NamedTuple):
        dq_c_dt: object = None
        dq_r_dt: object = None
        dq_s_dt: object = None
        dq_v_dt: object = None
        dN_r_dt: object = None
        dN_c_dt: object = None
        nc_budget: object = None
        qc_budget: object = None
    seen = []

    def fn(*a, **kw):
        seen.append(a[8])
        w = a[8].mg2_drop_sed_reset_log_width
        one = np.ones((1, 1))
        return Out(*(one * (1.0 + w),) * 4, dN_r_dt=one * 2.0, dN_c_dt=one * (4.0 + w),
                   nc_budget={"sedimentation": one * (2.0 + w)},
                   qc_budget={"sedimentation": one * (1.0 + w)})
    monkeypatch.setattr(mi, "_get_microphysics_fn", lambda c: ("morrison", fn, Cfg()))
    calls = []
    nb.install_micro_spy(calls, reset_width_ab=True, reset_field="mg2_drop_sed_reset_log_width")
    _, spy, cfg = mi._get_microphysics_fn(None)
    spy(*range(5), np.full((1, 1), 2.0), 6, 7, cfg)
    assert seen == [Cfg(True, 0.01, 0.01), Cfg(True, 0.01, 0.0)]   # only the sed width moves
    assert calls[0]["field"] == "mg2_drop_sed_reset_log_width"
    t = nb.reset_ab_table(nb.reset_ab_rows(calls), np.ones((1, 1)), np.ones((1, 1), bool))
    assert t["N_c: sedimentation"] == pytest.approx((86400.0, 1.005 * 86400, 0.005 * 86400))
    assert t["dN_c_dt"][2] == pytest.approx(0.005 * 86400)
    assert t["q_c: sedimentation"][2] == pytest.approx(0.01 * 86400)
    # no droplet sedimentation in the calls (specified droplet number): vacuous
    for arm in ("hard", "default"):
        calls[0][arm]["qc"] = {"other": calls[0][arm]["qc"]["sedimentation"]}
    with pytest.raises(SystemExit, match="droplet sedimentation ran"):
        nb.reset_ab_rows(calls)
    nb.install_micro_spy([], reset_width_ab=True, reset_field="mg2_drop_readjust_log_width")
    with pytest.raises(SystemExit, match="unknown reset field"):
        mi._get_microphysics_fn(None)


def test_reset_ab_path_keeps_the_two_fields_apart(tmp_path):
    early = nb.reset_ab_path(tmp_path, "mg2_drop_early_reset_log_width", "r_d0010")
    sed = nb.reset_ab_path(tmp_path, "mg2_drop_sed_reset_log_width", "r_d0010")
    assert early.name == "reset_ab_r_d0010.npz" and sed.name == "reset_ab_sed_r_d0010.npz"
    with pytest.raises(SystemExit, match="unknown reset field"):
        nb.reset_ab_path(tmp_path, "mg2_drop_readjust_log_width", "r_d0010")
