"""The compiled driver lane applies the physics schemes' wind tendencies.

``PhysicsOutput.du_dt``/``dv_dt`` carry the turbulence surface stress, the
gravity-wave drag and the prescribed surface-stress kick.  The per-step lane
adds them (``ModelDriver._add_cc_wind_increment``); the compiled segment lane
used to drop them, so on a lat-lon or cube compiled run no scheme ever moved
the winds (an injected 1e-3 m/s^2 for 10 steps changed nothing, bit for bit).

Each test runs the SAME compiled run twice, once with the physics' own
momentum tendency and once with a large constant added to it.  If the lane
consumed nothing the two runs are identical; consuming it moves the winds by
about ``n_steps * dt * extra``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver import physics_pipeline as pp
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver

_DT = 100.0
_N_STEPS = 3
_EXTRA = 1e-3   # m/s^2 -> 0.3 m/s over the run if applied


def _run(tmp_path, monkeypatch, tag, grid, dycore, extra):
    orig = pp.PhysicsPipeline.build_step_unified

    def _wrap(self, *a, **k):
        f = orig(self, *a, **k)

        def g(*aa, **kk):
            r = f(*aa, **kk)
            po = r[0]
            po = po._replace(du_dt=po.du_dt + extra, dv_dt=po.dv_dt - extra)
            return (po,) + tuple(r[1:])
        return g

    monkeypatch.setattr(pp.PhysicsPipeline, "build_step_unified", _wrap)
    out = tmp_path / tag
    cfg = ExperimentConfig(
        grid=grid, dycore=dycore,
        output=OutputConfig(output_dir=str(out), diag_days=0, checkpoint_days=0),
        days=_N_STEPS * _DT / 86400.0, dataset="analytical",
        held_suarez_forcing=False, turbulence="louis", radiation="none",
        convection="none", microphysics="none", gravity_wave_drag="none",
        cloud_scheme="none", fix_moisture=True, distributed=False, n_devices=1)
    d = ModelDriver(cfg, output_dir=str(out))
    d.setup()
    assert d.run(compiled=True) == "COMPLETED"
    st = d.cell_centre_state()
    return (np.asarray(st.u.data, dtype=np.float64),
            np.asarray(st.v.data, dtype=np.float64))


_LANES = {
    "latlon": (GridConfig(grid_type="latlon", resolution=8, nlev=4),
               DycoreConfig(model_type="hydrostatic",
                            discretization="latlon_cgrid", dt=_DT)),
    "cube": (GridConfig(grid_type="cubed_sphere", resolution=6, nlev=4),
             DycoreConfig(model_type="hydrostatic", discretization="cdgrid",
                          dt=_DT)),
    "cube_persistent_dgrid": (
        GridConfig(grid_type="cubed_sphere", resolution=6, nlev=4),
        DycoreConfig(model_type="hydrostatic", discretization="cdgrid",
                     dt=_DT, persistent_dgrid=True)),
}


@pytest.mark.parametrize("lane", sorted(_LANES))
def test_compiled_lane_consumes_physics_wind_tendency(tmp_path, monkeypatch,
                                                       lane):
    grid, dycore = _LANES[lane]
    u0, v0 = _run(tmp_path, monkeypatch, "base", grid, dycore, 0.0)
    u1, v1 = _run(tmp_path, monkeypatch, "kick", grid, dycore, _EXTRA)
    expected = _N_STEPS * _DT * _EXTRA
    du = float(np.mean(u1 - u0))
    dv = float(np.mean(v1 - v0))
    # Dynamics and the turbulence response redistribute part of the kick, so
    # only the order of magnitude and the sign are pinned; a lane that drops
    # the tendency gives exactly 0.
    assert du > 0.3 * expected, (lane, du, expected)
    assert dv < -0.3 * expected, (lane, dv, expected)
