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
def test_run_coupled_writes_manifest_before_setup(tmp_path, monkeypatch, deck):
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from scripts.run import run_coupled

    def _stop(self, *a, **kw):
        raise _LoopReached()

    monkeypatch.setattr(CoupledESMDriver, "setup", _stop)
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
                                       config_kind="land")
    restart.write_run_manifest_on_root(tmp_path, {"runner": "b"},
                                       config_kind="land")
    prev = list(tmp_path.glob("run_manifest.prev-*.json"))
    assert len(prev) == 1
    assert json.loads(prev[0].read_text())["config"]["resolved_config"] == {
        "runner": "a"}
    assert _manifest(tmp_path)["config"]["resolved_config"] == {"runner": "b"}


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
