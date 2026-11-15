"""The single-step JRA55 lane routes runoff like the block lane (static replacement + the
dry-to-wet routing plan).  Before the fix it passed the raw cache runoff straight to the ocean."""
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_run_omip_jra55_dispatch as D  # noqa: E402

R = D.run_omip


class _Recorder:
    """Stands in for the ocean model: records the freshwater forcing it is stepped with."""

    def __init__(self):
        self.fw = None

    def step(self, state, dt, freshwater=None, surface_forcing=None, sponge=None):
        self.fw = freshwater
        return state


@pytest.mark.parametrize("with_static", [True, False])
def test_single_step_runoff_equals_block_runoff(tmp_path, with_static):
    from legoesm.ocean.forcing.runoff_mapper import build_runoff_map
    n_lat, n_lon = 8, 16
    cache = D._make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon, n_records=16)
    rng = np.random.default_rng(0)
    # nonzero cache runoff everywhere, so dry cells carry runoff the plan must move
    zarr.open(str(cache), mode="r+")["friver"][:] = rng.uniform(1e-6, 1e-5, (16, n_lat, n_lon))
    grid, z_coord, *_ = D._make_tiny_latlon_setup(n_lat=n_lat, n_lon=n_lon)
    state = R._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    js = R._setup_jra55_forcing_state(D._argparse_namespace(jra55_cache=str(cache)),
                                      grid, "latlon")
    for k in ("enable_sponge", "enable_sss_restoring", "enable_freeze_cap"):
        js[k] = False
    mask = np.ones((n_lat, n_lon), bool)
    mask[3:5, 4:7] = False                                   # dry cells that must donate
    area = np.asarray(grid.area)
    js["runoff_map"] = build_runoff_map(mask, area, np.asarray(grid.lat2d),
                                        np.asarray(grid.lon2d), radius_m=3.0e6)
    assert js["runoff_map"].unrouted_fraction == 0.0
    if with_static:
        js["runoff_static"] = jnp.asarray(rng.uniform(0, 1e-5, (n_lat, n_lon)))
        raw = np.asarray(js["runoff_static"])
    else:
        _, raw = R._preload_jra55_forcing_block(
            0, 1, 3.0 * 3600.0, {**js, "runoff_map": None})
        raw = np.asarray(raw[0])                             # unrouted, on model cells
    dt = 3.0 * 3600.0
    rec = _Recorder()
    R._jra55_step(state, 0, dt, rec, js)
    out = np.asarray(rec.fw.runoff)
    _, runoff_block = R._preload_jra55_forcing_block(0, 1, dt, js)
    np.testing.assert_allclose(out, np.asarray(runoff_block[0]), rtol=1e-12, atol=1e-20)
    assert float(np.abs(out[~mask]).max()) == 0.0            # dry cells emptied
    assert float(np.abs(raw[~mask]).min()) > 0.0             # ...and they had runoff to move
    np.testing.assert_allclose((out * area).sum(), (raw * area).sum(), rtol=1e-12)
