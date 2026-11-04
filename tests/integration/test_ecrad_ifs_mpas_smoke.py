"""radiation='ecrad_ifs' on the MPAS column lane (smallest test mesh), with
IFS CLOUDSC + CUMASTRN as in the IFS deck.

The lane refuses to build until the user answers votes L12-L14 (TSI, gases,
ozone), so the three inputs are INJECTED here by wrapping the factory the
combined physics calls.  The injected values are test inputs only.

Pins: without the sources the lane refuses (vote S2a); with them, at
rad_update_steps=2, a 1-step run uses only the full-call variant, a 3-step run
(full, RADHEATN-only, full) also traces the RADHEATN-only variant and ends on
a REFRESHED carry; both stay finite and change the state."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

DT = 300.0
NLEV = 8

# test-only inputs (not a model choice): present-day-like well-mixed mass
# mixing ratios and a uniform ozone, TSI x distance factor = 1361 W/m^2
_SRC_ARGS = dict(
    solar_irradiance=1361.0,
    gas_mmr=dict(co2=6.2e-4, ch4=9.0e-7, n2o=4.8e-7, cfc11=1.1e-9, cfc12=2.6e-9,
                 hcfc22=7.0e-10, ccl4=4.5e-10),
    o3_mmr=1.0e-7)


def _inject(monkeypatch, ncol):
    import jax.numpy as jnp

    from legoesm.atmosphere.physics import combined
    from legoesm.atmosphere.physics.radiation.ecrad_ifs.config import EcradIFSSources
    src = EcradIFSSources(
        solar_irradiance=_SRC_ARGS["solar_irradiance"], gas_mmr=_SRC_ARGS["gas_mmr"],
        o3_mmr=jnp.full((ncol, NLEV), _SRC_ARGS["o3_mmr"]))
    orig = combined.make_radiation_physics

    def with_sources(*a, **k):        # combined passes ecrad_ifs_sources=None
        return orig(*a, **{**k, "ecrad_ifs_sources": src})

    monkeypatch.setattr(combined, "make_radiation_physics", with_sources)


def _driver(tmp, n_steps):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    from legoesm.driver.model_driver import ModelDriver
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=NLEV,
                        vertical_coord="hybrid"),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="mpas", dt=DT),
        output=OutputConfig(output_dir=str(tmp), diag_days=0, checkpoint_days=0),
        days=n_steps * DT / 86400.0, dataset="analytical", fix_moisture=True,
        precision="fp64",                # the IFS deck row (port is fp64)
        distributed=False, n_devices=1, turbulence="louis", radiation="ecrad_ifs",
        rad_update_steps=2, diurnal_cycle=True, convection="bechtold",
        bechtold_use_ifs_ascent=True, microphysics="ifs_cloudsc",
        gravity_wave_drag="none", cloud_scheme="none", cld_macmic_num_steps=1)
    d = ModelDriver(cfg, output_dir=str(tmp))
    d.setup()
    return d


def test_refuses_without_sources(tmp_path):
    with pytest.raises(ValueError, match="A15"):
        _driver(tmp_path, 1).run(compiled=True, segment_callback=lambda drv, day, dt: None)


def _run(tmp, n_steps, monkeypatch):
    from legoesm.atmosphere.physics.radiation.ecrad_ifs import lane
    d = _driver(tmp, n_steps)
    _inject(monkeypatch, int(np.asarray(d.state.T.data).shape[0]))
    calls = []
    orig = lane.step

    def spy(*a, **k):                 # records each traced variant
        calls.append(k["full_call"])
        return orig(*a, **k)

    monkeypatch.setattr(lane, "step", spy)
    T0 = np.asarray(d.state.T.data).copy()
    assert d.run(compiled=True, segment_callback=lambda drv, day, dt: None) == "COMPLETED"
    s = d.state
    assert np.isfinite(np.asarray(s.T.data)).all()
    for name, f in s.tracers.items():
        assert np.isfinite(np.asarray(f.data)).all(), name
    assert np.max(np.abs(np.asarray(s.T.data) - T0)) > 0.0
    ps = getattr(d, "_mpas_phys_state", None)
    assert ps is not None and ps.ecrad_carry is not None
    c = np.asarray(ps.ecrad_carry)
    assert np.isfinite(c).all() and np.max(np.abs(c)) > 0.0
    return c, set(calls)


@pytest.mark.slow
def test_ecrad_lane_runs_full_and_held_steps(tmp_path, monkeypatch):
    c1, v1 = _run(tmp_path / "one", 1, monkeypatch)
    c3, v3 = _run(tmp_path / "three", 3, monkeypatch)
    assert v1 == {True}                  # step 0: the full call only
    assert v3 == {True, False}           # step 1: the RADHEATN-only variant
    # step 2 is a new full call: the carry is refreshed, not held
    assert np.max(np.abs(c3 - c1)) > 0.0
