"""Every production driver writes run_manifest.json, and a failed write aborts."""
from __future__ import annotations

import json

import pytest

from legoesm.driver import restart


class _LoopReached(Exception):
    pass


def _manifest(path):
    m = restart.read_run_manifest(path)
    restart.validate_run_manifest(m)
    return m


def test_run_omip_writes_manifest_before_the_loop(tmp_path, monkeypatch):
    from scripts.run import run_omip

    def _stop(*a, **kw):
        raise _LoopReached()

    monkeypatch.setattr(run_omip, "_run_omip_loop", _stop)
    args = run_omip.parse_args([
        "--grid", "latlon", "--resolution", "4x8", "--nlev", "4",
        "--dt", "10800", "--days", "1", "--output", str(tmp_path)])
    with pytest.raises(_LoopReached):
        run_omip.run_omip_single("latlon", args)
    m = _manifest(tmp_path / "latlon" / "4x8")
    assert m["config"]["config_kind"] == "ocean"
    rc = m["config"]["resolved_config"]
    assert rc["runner"] == "run_omip" and rc["n_levels"] == 4
    assert rc["runtime_config"]["__type__"].startswith("legoesm.ocean")


def test_manifest_write_failure_raises(tmp_path, monkeypatch):
    def _boom(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(restart, "write_run_manifest", _boom)
    with pytest.raises(RuntimeError, match="run manifest not written"):
        restart.write_run_manifest_on_root(tmp_path, {"runner": "x"},
                                           config_kind="land")


def test_unknown_config_kind_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown run-manifest config_kind"):
        restart.write_run_manifest(tmp_path, {"runner": "x"}, config_kind="lnd")


def test_run_lmip_writes_manifest_before_state_init(tmp_path, monkeypatch):
    from scripts.run import run_lmip

    def _stop(*a, **kw):
        raise _LoopReached()

    monkeypatch.setattr(run_lmip, "init_multilayer_land_state", _stop)
    monkeypatch.setattr("sys.argv", [
        "run_lmip.py", "--lat", "45.0", "--days", "1",
        "--output", str(tmp_path)])
    with pytest.raises(_LoopReached):
        run_lmip.main()
    m = _manifest(tmp_path)
    assert m["config"]["config_kind"] == "land"
    rc = m["config"]["resolved_config"]
    assert rc["runner"] == "run_lmip"
    assert rc["run_config"]["land"]["__type__"].endswith(":MultiLayerLandConfig")
    assert rc["cli_args"]["lat"] == 45.0


_COUPLED_DECKS = sorted(
    p for p in (__import__("pathlib").Path(__file__).resolve().parents[2]
                / "config" / "cmip").glob("cmip_ocean_*.yaml"))


@pytest.mark.parametrize("deck", [None] + _COUPLED_DECKS,
                         ids=lambda d: "default" if d is None else d.name)
def test_run_coupled_writes_manifest_before_run(tmp_path, monkeypatch, deck):
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from scripts.run import run_coupled

    def _stop(self, *a, **kw):
        raise _LoopReached()

    # setup is stubbed (no forcing files here); the manifest follows it
    monkeypatch.setattr(CoupledESMDriver, "setup", lambda self: None)
    monkeypatch.setattr(CoupledESMDriver, "run", _stop)
    argv = ["--output", str(tmp_path), "--days", "1"]
    if deck is not None:
        argv = ["--config", str(deck)] + argv + [
            "--ozone-file", "ozone.nc", "--solar-file", "solar.nc",
            "--ghg-file", "ghg.nc", "--aerosol-file", "aerosol.nc",
            "--woa-t-path", "woa_t.nc", "--woa-s-path", "woa_s.nc"]
    with pytest.raises(_LoopReached):
        run_coupled.main(argv)
    m = _manifest(tmp_path / "coupled")
    assert m["config"]["config_kind"] == "coupled"
    rc = m["config"]["resolved_config"]
    assert rc["runner"] == "run_coupled"
    assert rc["atmosphere"]["__type__"].endswith(":ExperimentConfig")
    assert rc["coupled"]["__type__"].endswith(":CoupledConfig")


def test_run_dino_writes_manifest_before_first_snapshot(tmp_path, monkeypatch):
    from scripts.run import run_dino

    def _stop(*a, **kw):
        raise _LoopReached()

    monkeypatch.setattr(run_dino, "_save_snapshot", _stop)
    monkeypatch.setattr("sys.argv", [
        "run_dino.py", "--n-lon", "10", "--days", "1",
        "--output-dir", str(tmp_path)])
    with pytest.raises(_LoopReached):
        run_dino.main()
    m = _manifest(tmp_path)
    assert m["config"]["config_kind"] == "ocean"
    rc = m["config"]["resolved_config"]
    assert rc["runner"] == "run_dino"
    assert rc["runtime_config"]["__type__"].startswith("legoesm.ocean")
    assert "vmix_scheme" in rc["dino_config"]


def test_record_final_state_digest_fills_result(tmp_path):
    import numpy as np

    path = restart.write_run_manifest(tmp_path, {"runner": "x"},
                                      config_kind="land")
    digest = restart.record_final_state_digest(path, {"T": np.ones(3)})
    assert digest and restart.recorded_state_digest(
        restart.read_run_manifest(path)) == digest


def test_rerun_into_same_dir_archives_previous_manifest(tmp_path):
    restart.write_run_manifest_on_root(tmp_path, {"runner": "a"},
                                       config_kind="land", rng_seeds={"m": 1})
    restart.write_run_manifest_on_root(tmp_path, {"runner": "a"},
                                       config_kind="land", rng_seeds={"m": 2})
    prev = list(tmp_path.glob("run_manifest.prev-*.json"))
    assert len(prev) == 1
    assert json.loads(prev[0].read_text())["result"]["rng_seeds"] == {"m": 1}
    assert _manifest(tmp_path)["result"]["rng_seeds"] == {"m": 2}


@pytest.mark.parametrize("root_failed", [True, False])
def test_non_root_process_raises_when_root_write_failed(tmp_path, monkeypatch,
                                                        root_failed):
    import jax
    from jax.experimental import multihost_utils

    monkeypatch.setattr(jax, "process_index", lambda: 1)
    monkeypatch.setattr(jax, "process_count", lambda: 2)
    monkeypatch.setattr(multihost_utils, "broadcast_one_to_all",
                        lambda x, is_source=None: int(root_failed))
    if root_failed:
        with pytest.raises(RuntimeError, match="failed on process 0"):
            restart.write_run_manifest_on_root(tmp_path, {"runner": "x"},
                                               config_kind="land")
    else:
        assert restart.write_run_manifest_on_root(
            tmp_path, {"runner": "x"}, config_kind="land") is None
    assert not (tmp_path / "run_manifest.json").exists()


def _fesom_args(**kw):
    from types import SimpleNamespace
    base = dict(
        emp_freshwater=True, dm2dc=False, sw_rgb_chl=True,
        prognostic_sea_ice=False, sss_restore=True,
        sss_restore_channel="water_flux", runoff=False,
        nemo_monthly_init=["t_init.nc", "s_init.nc"], woa_init=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _fesom_model(vmix_config=None):
    from types import SimpleNamespace

    from legoesm.ocean.dynamics.ocean_model_fesom import FesomOceanConfig
    return SimpleNamespace(
        config=FesomOceanConfig(dt=1800.0, vertical_coordinate="zstar"),
        vmix_config=vmix_config,
        _params=SimpleNamespace(k_ver=1.0e-5, a_ver=1.0e-4))


@pytest.mark.parametrize("lane", ["fesom_b1_unforced", "fesom_b4_core2_forced"])
def test_fesom_lanes_write_a_valid_manifest(tmp_path, lane):
    from legoesm.ocean.config import FesomRunRecord, ocean_config_from_dict
    from scripts.run import run_omip_core2

    vmix = run_omip_core2.build_tripole_vmix_config("tke")
    model = _fesom_model(vmix)
    path = run_omip_core2._write_fesom_manifest(
        tmp_path, _fesom_args(), model, lane, n_steps=48, dt=1800.0,
        total_days=1.0)
    assert path == tmp_path / "run_manifest.json"
    m = _manifest(path)
    assert m["config"]["config_kind"] == "ocean"
    assert m["reproducibility"]["runner_tag"] == "run_omip_core2"
    rec = ocean_config_from_dict(m["config"]["resolved_config"])
    assert isinstance(rec, FesomRunRecord)
    assert rec.lane == lane and rec.n_steps == 48 and rec.total_days == 1.0
    assert rec.runtime_config == model.config
    forcing = ("emp_freshwater", "dm2dc", "sw_rgb_chl", "prognostic_sea_ice",
               "sss_restore", "sss_restore_channel", "runoff")
    if lane == "fesom_b1_unforced":
        # User decision 2026-10-26: B1 applies none of these forcings.
        assert all(getattr(rec, f) == "not_applicable" for f in forcing)
    else:
        assert [getattr(rec, f) for f in forcing] == [
            True, False, True, False, True, "water_flux", False]
    # #2102 bundle: the two NEMO init paths, the resolved fesom_jax
    # background mixing, and the TKE card the model ran.
    assert rec.nemo_monthly_init == ["t_init.nc", "s_init.nc"]
    assert (rec.k_ver, rec.a_ver) == (1.0e-5, 1.0e-4)
    assert rec.vmix_config == vmix


def test_fesom_manifest_hash_distinguishes_tke_settings(tmp_path):
    from scripts.run import run_omip_core2

    hashes = []
    for i, eice in enumerate((0, 1)):
        d = tmp_path / str(i)
        d.mkdir()
        vmix = run_omip_core2.build_tripole_vmix_config("tke", tke_eice=eice)
        run_omip_core2._write_fesom_manifest(
            d, _fesom_args(), _fesom_model(vmix), "fesom_b4_core2_forced",
            n_steps=48, dt=1800.0, total_days=1.0)
        hashes.append(_manifest(d / "run_manifest.json")
                      ["config"]["config_hash"])
    assert hashes[0] != hashes[1]


def test_fesom_manifest_records_unset_nemo_init_as_none(tmp_path):
    from legoesm.ocean.config import ocean_config_from_dict
    from scripts.run import run_omip_core2

    run_omip_core2._write_fesom_manifest(
        tmp_path, _fesom_args(nemo_monthly_init=None), _fesom_model(),
        "fesom_b1_unforced", n_steps=48, dt=1800.0, total_days=1.0)
    rec = ocean_config_from_dict(
        _manifest(tmp_path / "run_manifest.json")["config"]["resolved_config"])
    assert rec.nemo_monthly_init is None and rec.vmix_config is None


def _fesom_lane_args(tmp_path):
    return _fesom_args(
        output=str(tmp_path), dt=1800.0, smoke=False, years=1.0 / 365.0,
        snapshot_every_days=0, forcing_path=None, forcing_ramp_days=0.0,
        sss_restore=False, sw_rgb_chl=False, nemo_monthly_init=None)


class _SteppingModel:
    def __init__(self):
        m = _fesom_model()
        self.config, self.vmix_config, self._params = (
            m.config, m.vmix_config, m._params)

    def step(self, *a, **kw):
        raise _LoopReached()


def test_fesom_b1_writes_manifest_before_the_loop(tmp_path):
    from scripts.run import run_omip_core2

    with pytest.raises(_LoopReached):
        run_omip_core2.run_fesom_b1_smoke(
            _fesom_lane_args(tmp_path), None, None, _SteppingModel(), None)
    assert _manifest(tmp_path / "run_manifest.json")


def test_fesom_forced_writes_manifest_before_the_loop(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    import legoesm.ocean.forcing as forcing_mod
    from scripts.run import run_omip_core2

    monkeypatch.setattr(forcing_mod, "load_core2_nyf", lambda **kw:
                        SimpleNamespace(u10=np.zeros((4, 3))))

    def _stop(*a, **kw):
        raise _LoopReached()

    monkeypatch.setattr(run_omip_core2, "_idx_t", _stop)
    grid = SimpleNamespace(lat=np.zeros(3), lon=np.zeros(3))
    with pytest.raises(_LoopReached):
        run_omip_core2.run_fesom_forced_loop(
            _fesom_lane_args(tmp_path), grid, None, _SteppingModel(), None)
    assert _manifest(tmp_path / "run_manifest.json")


def test_fesom_b1_records_the_final_state_digest(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    import scripts.run.run_fesom_core2 as fesom_core2
    from scripts.run import run_omip_core2

    monkeypatch.setattr(fesom_core2, "write_snapshot", lambda *a, **kw: None)
    model = _SteppingModel()
    # Each step adds 1, so the digested state must be the post-loop one.
    model.step = lambda state, dt: SimpleNamespace(
        inner=SimpleNamespace(T=state.inner.T + 1.0))
    model.mesh = None
    initial = SimpleNamespace(inner=SimpleNamespace(T=np.ones(3)))
    digested = []
    monkeypatch.setattr(run_omip_core2, "_record_final_state_digest",
                        lambda path, state: digested.append((path, state)))
    run_omip_core2.run_fesom_b1_smoke(
        _fesom_lane_args(tmp_path), None, None, model, initial)
    [(path, state)] = digested
    assert path == tmp_path / "run_manifest.json"
    # 1 day at dt=1800 s is 48 steps.
    np.testing.assert_array_equal(state.inner.T, np.full(3, 49.0))


def test_core2_ice_digest_is_separate_from_the_ocean_digest(tmp_path):
    """#2176: the prognostic sea ice gets its own result.ice_state_digest;
    the ocean state_digest stays the ocean-only digest (old manifests still
    compare), and a run without ice writes no ice field."""
    import numpy as np
    from scripts.run import run_omip_core2

    ocean = {"T": np.ones(3)}
    path = restart.write_run_manifest(tmp_path, {"runner": "x"},
                                      config_kind="land")
    run_omip_core2._record_final_state_digest(path, ocean)
    res = restart.read_run_manifest(path)["result"]
    assert "ice_state_digest" not in res
    ocean_only = res["state_digest"]

    for h in (1.0, 2.0):
        ice = {"h_ice": np.full(3, h)}
        run_omip_core2._record_final_state_digest(path, ocean, ice)
        res = restart.read_run_manifest(path)["result"]
        assert res["state_digest"] == ocean_only
        assert res["ice_state_digest"] == restart.pytree_state_digest(ice)
    assert res["ice_state_digest"] != restart.pytree_state_digest(
        {"h_ice": np.full(3, 1.0)})


@pytest.mark.parametrize("fn", ["run_fesom_forced_loop", "main"])
def test_core2_ice_lanes_pass_the_ice_to_the_digest(fn):
    """The two core2 lanes that carry a prognostic ice_state hand it to the
    run-end digest (source of the functions that run)."""
    import inspect
    from scripts.run import run_omip_core2

    src = inspect.getsource(getattr(run_omip_core2, fn))
    assert "_record_final_state_digest(manifest_path, state, ice_state)" in src


def test_core2_ice_digest_failure_keeps_the_ocean_digest(tmp_path, monkeypatch):
    """An ice digest that raises is reported; the ocean digest is still
    recorded and no ice field is written."""
    import numpy as np
    from scripts.run import run_omip_core2

    real = restart.pytree_state_digest

    def flaky(*trees):
        if "h_ice" in trees[0]:
            raise RuntimeError("not addressable")
        return real(*trees)

    monkeypatch.setattr(restart, "pytree_state_digest", flaky)
    path = restart.write_run_manifest(tmp_path, {"runner": "x"},
                                      config_kind="land")
    run_omip_core2._record_final_state_digest(
        path, {"T": np.ones(3)}, {"h_ice": np.ones(3)})
    res = restart.read_run_manifest(path)["result"]
    assert res["state_digest"] == real({"T": np.ones(3)})
    assert "ice_state_digest" not in res
