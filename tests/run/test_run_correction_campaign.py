"""Smoke test for the LES-informed correction CAMPAIGN driver composition.

Exercises the genuinely-new orchestration (``make_clubb_build_driver`` +
``make_les_diagnose_fn`` + ``build_correction_campaign``) WITHOUT a real model
run: a mock driver supplies the column state and a mock LES supplies the plane
state, so the REAL process_column forcing-extract + diagnose path runs (the real
model run + real LES are covered by iter 35/37 and iter 20 respectively).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.column_les import ColumnLESConfig  # noqa: E402
from legoesm.atmosphere.dynamics.les_regime import (  # noqa: E402
    LESRegimeConfig,
    LESResolutionConfig,
)
from legoesm.training.compare_reanalysis import ColumnState  # noqa: E402

from scripts.run.run_correction_campaign import (  # noqa: E402
    build_correction_campaign,
    make_clubb_build_driver,
    make_les_diagnose_fn,
)

_SMALL_RES = LESResolutionConfig(
    dx_m=50.0, nx=8, ny=8, nlev=8, domain_top_m=2000.0, dz_sfc_m=50.0)
_SMALL_REGIME = LESRegimeConfig(shallow=_SMALL_RES, deep=_SMALL_RES)


def _full_grid_state(nlat=8, nlon=16, nlev=5):
    shp, sfc = (nlat, nlon, nlev), (nlat, nlon)
    return ColumnState(
        T=jnp.full(shp, 280.0), q_v=jnp.full(shp, 5e-3),
        u=jnp.full(shp, 5.0), v=jnp.zeros(shp),
        p_s=jnp.full(sfc, 1.0e5), sst_K=jnp.full(sfc, 290.0))


class _FakeDriver:
    def __init__(self, state):
        self.state = state

    def run(self, segment_callback, **kwargs):  # noqa: ARG002
        segment_callback(self, 0.0, 1.0)        # one diagnostic segment
        return "OK"


def _mock_run_les(setup):
    """Mock plane-LES result: a synthetic state shaped to the setup's grid/hc."""
    from legoesm.atmosphere.dynamics.compressible_euler_plane import make_rest_state

    grid, hc = setup.grid, setup.height_coord
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    ny, nx, nlev = grid.ny, grid.nx, hc.n_levels
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    s = jnp.asarray(np.where((ii + jj) % 2 == 0, 1.0, -1.0))
    w = 2.0 * s[:, :, None] * jnp.ones((ny, nx, nlev + 1))
    thp = 0.5 * s[:, :, None] * jnp.ones((ny, nx, nlev))
    tr = jnp.zeros((ny, nx, nlev, 3)).at[..., 0].set(0.01)
    return state._replace(
        w=state.w.replace(data=w),
        theta_prime=state.theta_prime.replace(data=thp),
        tracers=state.tracers.replace(data=tr))


def _base_config():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite")


def test_make_clubb_build_driver_injects_override():
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

    captured = {}

    def build_base(cfg):
        captured["cfg"] = cfg
        return _FakeDriver(_full_grid_state())

    build_driver = make_clubb_build_driver(_base_config(), build_base)
    build_driver(CLUBBLiteConfig(C_K=0.9))
    cfg = captured["cfg"]
    assert cfg.turbulence == "clubb_lite"
    assert cfg.turbulence_override is not None
    assert float(cfg.turbulence_override.clubb_lite.C_K) == 0.9


def test_make_les_diagnose_fn_runs_process_column():
    """diagnose_fn(record, model_ctx) drives the REAL process_column + a mock LES."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les)

    class _Env:
        cape_J_kg = 200.0  # noqa: N815  (mirrors ColumnEnvironment field)

    class _Rec:
        grid_index = (4, 8)
        lat_deg = 20.0
        environment = _Env()

    out = diagnose_fn(_Rec(), _full_grid_state())
    assert out.K.shape == (_SMALL_RES.nlev - 1,)   # eddy-K profile
    assert bool(jnp.all(jnp.isfinite(out.K)))


@pytest.mark.slow
def test_build_correction_campaign_wiring_one_round():
    """WIRING test: the whole campaign composes + runs one round with a mock
    driver + mock LES — a per-column clubb C_K is produced and the bias is a
    finite measurement.  Does NOT validate that the C_K injection changes the
    model output (the mock driver returns a fixed state) — that mechanism is
    covered by iter 35/37."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    # reference = model with a localized +6 K bias → one deterministic worst column.
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    def build_base(cfg):           # mock: fixed state regardless of C_K
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_base_config(), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1)

    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.n_diagnosed == 1
    # The campaign produced a per-column clubb C_K (the loop closed).
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 1
    assert result.final_config.C_K.shape == (8 * 16,)
    assert np.isfinite(float(it.bias.baseline_bias))
    assert np.isfinite(float(it.bias.updated_bias))
