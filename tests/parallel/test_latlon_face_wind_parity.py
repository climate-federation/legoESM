"""Lat-lon C-grid runs carry their face winds between steps, on every lane.

The single-device compiled driver used to rebuild the face winds from
cell-centre winds every step; the 2-device lat-band lanes kept them.  After
108 steps of dry Held-Suarez (no surface fluxes) the two disagreed by 0.23 m/s
and 143 Pa, and the per-step rebuild removed most of the eddy kinetic energy
within a day.  With the face winds carried, one device and two devices agree
to rounding, and the single-device driver tracks a bare face-wind loop.

Needs ``XLA_FLAGS=--xla_force_host_platform_device_count=2`` for the 2-device
cases (skipped otherwise).
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    LatLonFaceWindState, facewind_to_hydrostatic, hydrostatic_to_facewind)
from legoesm.atmosphere.forcing.idealized.held_suarez import (
    held_suarez_forcing_latlon)
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver

_DT = 100.0
_N = 108


def _cfg(tmp_path, tag, *, spmd=False, turbulence="none", n=_N):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=4),
        dycore=DycoreConfig(model_type="hydrostatic",
                            discretization="latlon_cgrid", dt=_DT,
                            fix_mass=True),
        output=OutputConfig(output_dir=str(tmp_path / tag), diag_days=0,
                            checkpoint_days=0),
        days=n * _DT / 86400.0, dataset="analytical", held_suarez_forcing=True,
        turbulence=turbulence, radiation="none", convection="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        fix_moisture=True, enable_latlon_spmd=spmd, distributed=False,
        n_devices=2 if spmd else 1,
        # no surface fluxes for dry HS; a turbulence scheme needs its own
        # surface exchange, applied identically on both lanes
        surface_bulk_scheme="none" if turbulence == "none" else "constant")


def _run(cfg):
    d = ModelDriver(cfg, output_dir=cfg.output.output_dir)
    d.setup()
    assert d.run(compiled=True) == "COMPLETED"
    return d


def _fields(d):
    st = d.cell_centre_state()
    return {k: np.asarray(getattr(st, k).data, np.float64)
            for k in ("T", "u", "v", "p_s")}


def test_serial_driver_carries_face_winds_like_a_bare_loop(tmp_path):
    """The compiled driver ends on the face-wind state and matches a bare
    ``model.step`` loop fed its own face winds back each step."""
    d = _run(_cfg(tmp_path, "drv"))
    ref = ModelDriver(_cfg(tmp_path, "ref"), output_dir=str(tmp_path / "ref"))
    ref.setup()
    st = hydrostatic_to_facewind(ref.state, ref.grid)
    for _ in range(_N):
        st = ref.model.step(st, _DT, physics_fn=held_suarez_forcing_latlon)
    a = _fields(d)
    assert np.abs(a["u"]).max() > 0.05      # winds must have developed
    cc = facewind_to_hydrostatic(st, ref.grid)
    b = {k: np.asarray(getattr(cc, k).data, np.float64)
         for k in ("T", "u", "v", "p_s")}
    # The driver applies HS through its own Euler tail and Rayleigh decay, the
    # bare loop through the dycore's physics hook: same physics, different
    # operator order, so not bitwise.  The per-step rebuild this removes gave
    # 0.23 m/s / 143 Pa here.
    assert np.abs(a["u"] - b["u"]).max() < 0.03
    assert np.abs(a["p_s"] - b["p_s"]).max() < 20.0
    assert isinstance(d.state, LatLonFaceWindState)


@pytest.mark.parametrize("turbulence", ["none", "louis"])
def test_one_vs_two_devices_wind_path(tmp_path, turbulence):
    """turbulence='none' -> the 2-device STATELESS lane; 'louis' -> the
    2-device OPERATOR-SPLIT lane.  Both must match one device."""
    if len(jax.devices()) < 2:
        pytest.skip("needs --xla_force_host_platform_device_count=2")
    one = _fields(_run(_cfg(tmp_path, "one", turbulence=turbulence)))
    two = _fields(_run(_cfg(tmp_path, "two", spmd=True,
                            turbulence=turbulence)))
    umax = np.abs(one["u"]).max()
    assert umax > 0.05, umax          # winds must have developed
    # The per-step rebuild gave 0.23 m/s / 143 Pa here; rounding-level lane
    # differences are far below these absolute bounds.
    assert np.abs(one["u"] - two["u"]).max() < 0.01
    assert np.abs(one["v"] - two["v"]).max() < 0.01
    assert np.abs(one["p_s"] - two["p_s"]).max() < 5.0
