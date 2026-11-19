"""IFS Cariolle ozone tracer on the MPAS dycore lane (smallest mesh, CAM L32):
the driver carries o3_ifs, cold-starts it from k1, the dynamics conserves its
global mass, the run stays finite and positive, and a checkpoint without the
tracer is refused (votes C4a, C5a, O4a, O5a)."""
from __future__ import annotations

import glob
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.radiation import ozone_cariolle as oc  # noqa: E402

DT = 300.0


def _driver(tmp, n_steps, ckpt_days=0):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    from legoesm.driver.model_driver import ModelDriver
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=32,
                        vertical_coord="cam_l32"),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="mpas", dt=DT),
        output=OutputConfig(output_dir=str(tmp), diag_days=0,
                            checkpoint_days=ckpt_days),
        days=n_steps * DT / 86400.0 + (1.0 / 86400.0 if ckpt_days else 0.0),
        dataset="analytical", distributed=False, n_devices=1,
        turbulence="louis", radiation="none", convection="none",
        microphysics="kessler", gravity_wave_drag="none", cloud_scheme="none",
        ifs_ozone_chem=True, precision="fp64")
    d = ModelDriver(cfg, output_dir=str(tmp))
    d.setup()
    return d


def _k1(d, day):
    lat = np.rad2deg(np.asarray(d._owned_p_s_and_lat()[1])).reshape(-1)
    p = d.sigma.pressure_at_full(jnp.asarray([constants.p_atm_std], jnp.float64))
    m = oc.load_cariolle_coefficients(p, lat)
    return oc.cariolle_daily_coefficients(m, d._calendar_for_radiation(day)[0])[..., 0]


def _ozone_mass(d, state):
    dp = np.asarray(d.sigma.layer_thickness_dp(state.p_s.data))
    area = np.asarray(d.model.mesh.areaCell)
    o3 = np.asarray(state.tracers[oc.CARIOLLE_TRACER].data)
    return float(((o3 * dp).sum(1) * area).sum() / constants.g)


@pytest.mark.slow
def test_tracer_cold_start_transport_and_run(tmp_path):
    d = _driver(tmp_path / "a", 2)
    assert oc.CARIOLLE_TRACER in d.state.tracers
    k1 = _k1(d, float(d.config.start_day))
    # dynamics alone (chemistry off) conserves the global ozone mass
    s = d.state._replace(tracers={**d.state.tracers, oc.CARIOLLE_TRACER:
                                  d.state.tracers[oc.CARIOLLE_TRACER].replace(
                                      data=jnp.asarray(k1))})
    m0 = _ozone_mass(d, s)
    for _ in range(3):
        s = d.model.step(s, DT)
    assert _ozone_mass(d, s) == pytest.approx(m0, rel=1e-12)
    # the driver run: cold start from k1, finite and positive after 2 steps;
    # no segment callback, radiation off, no external forcing: the daily
    # coefficient refresh must still run (codex review 4, P1)
    assert d.run() == "COMPLETED"
    o3 = np.asarray(d.state.tracers[oc.CARIOLLE_TRACER].data)
    assert np.isfinite(o3).all() and o3.min() > 0.0
    assert np.abs(o3 - k1).max() < 0.05 * k1.max()      # 600 s from k1
    assert np.abs(o3 - k1).max() > 0.0


@pytest.mark.slow
def test_checkpoint_without_ozone_is_refused(tmp_path):
    dA = _driver(tmp_path / "a", 2, ckpt_days=1)
    assert dA.run(segment_callback=lambda drv, day, dt: None) == "COMPLETED"
    ckpt = sorted(glob.glob(str(tmp_path / "a" / "checkpoint_day_*.npz")))[-1]
    with np.load(ckpt) as f:
        payload = {k: f[k] for k in f.files}
    assert "trc_o3_ifs" in payload
    del payload["trc_o3_ifs"]
    payload["tracer_names"] = np.asarray(
        [n for n in payload["tracer_names"] if str(n) != "o3_ifs"])
    bad = str(tmp_path / "old.npz")
    np.savez(bad, **payload)
    import shutil
    shutil.copy(str(tmp_path / "a" / "run_manifest.json"), str(tmp_path))
    with pytest.raises(ValueError, match="has no o3_ifs"):
        _driver(tmp_path / "b", 2).load_checkpoint(bad)
