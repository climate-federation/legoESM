"""IFS 49r1 CLOUDSC on the MPAS column lane (smallest MPAS test mesh): the run
completes and stays finite, and the cloud_fraction_ifs tracer is ADVECTED.

Advection is shown without trusting any diagnostic: two one-step runs differ
only in a cloud-fraction blob planted in a few columns (the same cloud water
in both), on a solid-body zonal wind.  CLOUDSC is column-local, so a column
outside the blob that the cloud fraction did NOT reach sees identical inputs
in both runs and gives identical output; a difference there after one step
is the transported cloud fraction."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

DT = 300.0
NLEV = 8
_BLOB = np.arange(12)            # cell indices of the planted cloud
_K = NLEV // 2


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
        distributed=False, n_devices=1, turbulence="louis", radiation="none",
        convection="bechtold", bechtold_use_ifs_ascent=True,
        microphysics="ifs_cloudsc", gravity_wave_drag="none",
        cloud_scheme="none", cld_macmic_num_steps=1)
    d = ModelDriver(cfg, output_dir=str(tmp))
    d.setup()
    return d


_U0 = 20.0                       # solid-body zonal wind [m/s]


def _wind(d):
    """u_normal = U0 cos(lat) cos(angleEdge) on every level (identical in
    both arms), so the blob has something to be carried by."""
    m = d.model.mesh
    un = _U0 * np.cos(np.asarray(m.latEdge)) * np.cos(np.asarray(m.angleEdge))
    u = np.broadcast_to(un[:, None], np.asarray(d.state.u.data).shape)
    d.state = d.state._replace(u=d.state.u.replace(
        data=jnp.asarray(u, d.state.u.data.dtype)))


def _plant(d, blob):
    """Cloud water in the blob columns in BOTH arms; the cloud fraction only
    in the ``blob`` arm.  The lane steps the dynamics before the physics, so
    after one step a neighbour column holds the same transported condensate
    in both arms and differs only in the transported cloud fraction."""
    tr = dict(d.state.tracers)
    for name, val in (("q_c", 1.0e-4), ("cloud_fraction_ifs", 0.5 if blob else 0.0)):
        a = np.asarray(tr[name].data).copy()
        a[_BLOB, _K] = val
        tr[name] = tr[name].replace(data=jnp.asarray(a, tr[name].data.dtype))
    d.state = d.state._replace(tracers=tr)


def _run(tmp, n_steps, blob):
    d = _driver(tmp, n_steps)
    assert "cloud_fraction_ifs" in d.state.tracers
    assert float(jnp.max(jnp.abs(d.state.tracers["cloud_fraction_ifs"].data))) == 0.0
    _wind(d)
    _plant(d, blob)
    # a no-op segment callback switches on the lane's per-step forcing
    # build, which hands the convection the surface heat fluxes CUMASTRN's
    # trigger reads (the analytical case has no SST forcing to do it)
    assert d.run(compiled=True,
                 segment_callback=lambda drv, day, dt: None) == "COMPLETED"
    return d


@pytest.mark.slow
def test_cloud_fraction_is_advected_and_the_run_stays_finite(tmp_path):
    with_blob = _run(tmp_path / "blob", 1, True)
    without = _run(tmp_path / "none", 1, False)
    a1 = np.asarray(with_blob.state.tracers["cloud_fraction_ifs"].data)
    a0 = np.asarray(without.state.tracers["cloud_fraction_ifs"].data)
    outside = np.ones(a1.shape[0], bool)
    outside[_BLOB] = False
    diff = np.abs(a1 - a0)[outside]
    # one step of 300 s at <= 20 m/s on a ~1000 km mesh moves a few percent
    assert diff.max() > 1.0e-4, diff.max()            # transported out of the blob
    assert np.isfinite(a1).all() and np.isfinite(a0).all()

    longer = _run(tmp_path / "long", 4, True)
    s = longer.state
    for name, f in s.tracers.items():
        assert np.isfinite(np.asarray(f.data)).all(), name
    assert np.isfinite(np.asarray(s.T.data)).all()
    assert np.isfinite(np.asarray(s.p_s.data)).all()
    a = np.asarray(s.tracers["cloud_fraction_ifs"].data)
    assert a.min() >= 0.0
