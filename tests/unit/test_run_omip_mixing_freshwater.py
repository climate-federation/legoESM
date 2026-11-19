"""Issue #2234: the OMIP JRA55-do driver hands vertical mixing the run's
surface freshwater.

With sea ice off, every forcing path of ``run_omip`` used to build
``OceanSurfaceForcing(freshwater=None)``, so KPP dropped the freshwater term
from its surface buoyancy and ePBL refused to run.  The driver now fills that
channel with the same form the sea-ice blend uses when ice is on:
``P - E + R + ice_fw`` with the SSS-restoring term excluded.  The mass and salt
still enter once, through ``model.step(freshwater=fw)``.

A spy on the model step records what each path passes; the expected value is
recomputed from the ``fw`` the same call receives (so the interpolated forcing
on the model grid, not the raw cache).  The synthetic cache carries non-zero
runoff, and the single-step path is given a non-zero restoring term, so a
wrong term set would show.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("zarr")

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_jra55_dispatch_mixing_fw", _HERE / "test_run_omip_jra55_dispatch.py")
_d = importlib.util.module_from_spec(_spec)
sys.modules["_jra55_dispatch_mixing_fw"] = _d
_spec.loader.exec_module(_d)
run_omip = _d.run_omip

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.freshwater import net_freshwater_flux  # noqa: E402

N_LAT, N_LON = 8, 16


def _cache_with_runoff(tmp_path):
    raw = _d._make_synthetic_cache(tmp_path, n_lat=N_LAT, n_lon=N_LON,
                                   n_records=16)
    ds = xr.open_zarr(str(raw)).load()
    rng = np.random.default_rng(1)
    ds["friver"] = (("time", "lat", "lon"),
                    rng.uniform(0.0, 1e-5, ds["prra"].shape))
    out = tmp_path / "with_runoff.zarr"
    ds.to_zarr(str(out), mode="w", consolidated=True)
    return out


def _setup(tmp_path):
    cache = _cache_with_runoff(tmp_path)
    grid, z_coord, _, model, _ = _d._make_tiny_latlon_setup(
        n_lat=N_LAT, n_lon=N_LON)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _d._argparse_namespace(jra55_cache=str(cache))
    js = run_omip._setup_jra55_forcing_state(args, grid, "latlon")
    assert not js.get("enable_sea_ice", False)
    js["_ocean_mask_2d"] = state.land_mask.data > 0.5
    return model, state, grid, z_coord, js


def _spy(model, attr, rec):
    orig = getattr(model, attr)

    def spy(st, d, **kw):
        sf, fw = kw["surface_forcing"], kw["freshwater"]
        rec["none"].append(sf.freshwater is None)
        if sf.freshwater is not None:
            jax.debug.callback(
                lambda a, b, ab, r: rec["vals"].append(
                    tuple(np.asarray(x) for x in (a, b, ab, r))),
                sf.freshwater, net_freshwater_flux(fw._replace(restoring=None)),
                net_freshwater_flux(fw), fw.runoff)
        return orig(st, d, **kw)

    setattr(model, attr, spy)


@pytest.mark.parametrize("path", ["single_step", "block", "gpu_interp"])
def test_mixing_sees_physical_net_freshwater(tmp_path, monkeypatch, path):
    model, state, grid, z_coord, js = _setup(tmp_path)
    if path == "single_step":
        js["_use_single_step"] = True
        attr = "step"
        # Give the restoring term a non-zero value so its exclusion is tested.
        import legoesm.forcing.jra55_do as jd
        orig_fw = jd.jra55_to_freshwater

        def with_restoring(*a, **k):
            fw = orig_fw(*a, **k)
            return fw._replace(restoring=jnp.full_like(fw.precip, 3e-6))
        monkeypatch.setattr(jd, "jra55_to_freshwater", with_restoring)
    else:
        attr = "_step_impl"
        js["_gpu_interp"] = path == "gpu_interp"
    rec = {"none": [], "vals": []}
    _spy(model, attr, rec)
    _, _, _, ok, _ = run_omip._run_omip_loop(
        model, state, "latlon", grid, z_coord,
        dt=300.0, n_steps=2, diag_every=2, jra55_state=js,
        checkpoint_days=None, checkpoint_dir=None)
    assert ok
    assert rec["none"] and not any(rec["none"]), rec["none"]
    assert rec["vals"]
    for got, want, with_r, runoff in rec["vals"]:
        np.testing.assert_array_equal(got, want)
        assert np.max(np.abs(got)) > 0.0
        assert np.max(np.abs(runoff)) > 0.0
        if path == "single_step":
            assert not np.allclose(got, with_r, rtol=0.0, atol=1e-7)


def test_zero_surface_fluxes_also_zeroes_the_mixing_channel(tmp_path):
    model, state, grid, z_coord, js = _setup(tmp_path)
    js["zero_surface_fluxes"] = True
    rec = {"none": [], "vals": []}
    _spy(model, "_step_impl", rec)
    run_omip._run_omip_loop(
        model, state, "latlon", grid, z_coord,
        dt=300.0, n_steps=2, diag_every=2, jra55_state=js,
        checkpoint_days=None, checkpoint_dir=None)
    assert rec["vals"] and not any(rec["none"])
    for got, *_ in rec["vals"]:
        assert np.all(got == 0.0)
