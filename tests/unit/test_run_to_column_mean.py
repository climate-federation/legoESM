"""Unit tests for :mod:`legoesm.training.run_to_column_mean`.

The run→time-mean→compare wiring: the generic accumulation logic is tested with
a mock driver (deterministic, cheap), and the real CMIP path with a tiny coupled
run (slow).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace  # noqa: E402

from legoesm.training.compare_reanalysis import ColumnState  # noqa: E402
from legoesm.training.run_to_column_mean import (  # noqa: E402
    amip_column_state,
    cmip_column_state,
    make_run_fn,
    model_phis_from_driver,
    phis_or_none_if_flat,
    run_to_column_mean,
)


def _state(scale, *, nlat=2, nlon=3, nlev=4):
    shp = (nlat, nlon, nlev)
    sfc = (nlat, nlon)
    return ColumnState(
        T=jnp.full(shp, 280.0 + scale),
        q_v=jnp.full(shp, 5e-3),
        u=jnp.full(shp, scale),
        v=jnp.zeros(shp),
        p_s=jnp.full(sfc, 1.0e5),
        sst_K=jnp.full(sfc, 290.0 + scale),
    )


class _FakeDriver:
    """Mimics a driver: ``run(segment_callback=...)`` fires the callback once per
    segment, advancing an internal index so the extractor sees a new state."""

    def __init__(self, states):
        self.states = states
        self.i = -1
        self.run_kwargs = None

    def run(self, segment_callback, **kwargs):
        self.run_kwargs = kwargs
        for k in range(len(self.states)):
            self.i = k
            segment_callback(self, float(k), 1.0)
        return "OK"


def _extract(driver, day, dt):  # noqa: ARG001
    return driver.states[driver.i]


def test_run_to_column_mean_is_time_mean():
    states = [_state(0.0), _state(3.0), _state(6.0)]  # mean scale = 3
    driver = _FakeDriver(states)
    mean = run_to_column_mean(driver, _extract)
    np.testing.assert_allclose(np.asarray(mean.T), 283.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(mean.u), 3.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(mean.sst_K), 293.0, rtol=1e-12)


def test_run_to_column_mean_samples_the_outer_driver_not_the_callback_arg():
    """The extractor MUST read the OUTER (closed-over) driver, NOT the object the driver
    passes as the callback's first arg.  This is load-bearing for CMIP: the outer COUPLED
    driver carries ``ocean_state`` (the prognostic SST), but a wrapper driver could pass an
    atm SUB-driver to the callback — which lacks the ocean.  Lock it non-vacuously: a driver
    that passes a WRONG sentinel as the callback's first arg must STILL be sampled via the
    outer reference (so ``cmip_column_state`` reaches the real ocean_state)."""
    sentinel = object()

    class _PassesWrongArg(_FakeDriver):
        def run(self, segment_callback, **kwargs):
            self.run_kwargs = kwargs
            for k in range(len(self.states)):
                self.i = k
                segment_callback(sentinel, float(k), 1.0)   # WRONG first arg (not self)
            return "OK"

    driver = _PassesWrongArg([_state(0.0), _state(6.0)])    # outer-driver mean scale = 3
    seen = []

    def extract(d, day, dt):  # noqa: ARG001
        seen.append(d)
        return d.states[d.i]                                # sentinel has no .states → would raise

    mean = run_to_column_mean(driver, extract)
    assert seen and all(d is driver for d in seen)          # always the OUTER driver
    np.testing.assert_allclose(np.asarray(mean.T), 283.0, rtol=1e-12)  # mean(280, 286)


def test_run_to_column_mean_forwards_run_kwargs():
    driver = _FakeDriver([_state(1.0)])
    run_to_column_mean(driver, _extract, run_kwargs={"start_day": 7.0})
    assert driver.run_kwargs == {"start_day": 7.0}


def test_run_to_column_mean_no_segment_raises():
    """A run that fires no segment boundary must raise, not return a zero mean."""
    driver = _FakeDriver([])  # run() fires the callback zero times
    with pytest.raises(ValueError, match="no segment boundary fired"):
        run_to_column_mean(driver, _extract)


def test_run_to_column_mean_single_segment():
    driver = _FakeDriver([_state(4.0)])
    mean = run_to_column_mean(driver, _extract)
    np.testing.assert_allclose(np.asarray(mean.T), 284.0, rtol=1e-12)


def test_run_to_column_mean_threads_segment_day():
    """The segment `day` is threaded into the extractor (NOT day-0 every time) —
    AMIP SST is time-dependent, so a multi-day mean must see each segment's day."""
    seen_days = []

    def extract(driver, day, dt):
        seen_days.append(day)
        return driver.states[driver.i]

    driver = _FakeDriver([_state(0.0), _state(1.0), _state(2.0)])
    run_to_column_mean(driver, extract)
    assert seen_days == [0.0, 1.0, 2.0]


class _FakeAmipDriver:
    """Minimal AMIP driver: fixed atm state, TIME-DEPENDENT prescribed SST."""

    def __init__(self, n_seg, nlat=2, nlon=3, nlev=4):
        shp, sfc = (nlat, nlon, nlev), (nlat, nlon)
        self.state = SimpleNamespace(
            T=jnp.full(shp, 280.0), u=jnp.zeros(shp),
            v=jnp.zeros(shp), p_s=jnp.full(sfc, 1.0e5),
        )
        self.q_v = jnp.full(shp, 5e-3)
        self._n_seg = n_seg
        self._sfc = sfc

    def get_sst_sic(self, day):
        # SST rises with the calendar day → mean must reflect all sampled days.
        return jnp.full(self._sfc, 290.0 + day), None

    def run(self, segment_callback, **kwargs):  # noqa: ARG002
        for k in range(self._n_seg):
            segment_callback(self, float(k), 1.0)
        return "OK"


def test_amip_prescribed_sst_threaded_through_segments():
    """The AMIP day-threading bug guard: with day-varying prescribed SST, the
    time-mean SST is the mean over the SAMPLED days, not day-0 repeated."""
    driver = _FakeAmipDriver(n_seg=3)  # days 0,1,2 → SST 290,291,292
    mean = run_to_column_mean(driver, amip_column_state)
    np.testing.assert_allclose(np.asarray(mean.sst_K), 291.0, rtol=1e-12)  # mean(290,291,292)
    # (A day-0-only bug would give 290.0.)


def test_cmip_column_state_mpas_reconstructs_winds_and_carries_u_edge():
    """The CMIP (coupled) × MPAS path: cmip_column_state reconstructs the cell
    wind from the native edge velocity (driver.grid is the VoronoiMesh, via the
    iter-74 CoupledESMDriver.grid property), carries u_edge for the LES extractor,
    and uses the COUPLED-ocean SST — the only mode×grid combo not yet covered."""
    from legoesm.grids.voronoi import create_voronoi_mesh, reconstruct_cell_velocity

    mesh = create_voronoi_mesh(2)
    nlev = 4
    u_edge = 6.0 * jnp.cos(jnp.asarray(mesh.angleEdge))[:, None] * jnp.ones((1, nlev))
    # MPAS prognostic state: edge velocity in u, NO cell v (the MPAS marker).
    state = SimpleNamespace(
        T=jnp.full((mesh.nCells, nlev), 285.0), u=u_edge, v=None,
        p_s=jnp.full((mesh.nCells,), 1.0e5))
    coupled = SimpleNamespace(
        state=state, q_v=jnp.full((mesh.nCells, nlev), 6e-3),
        ocean_state=SimpleNamespace(T_sfc=jnp.full((mesh.nCells,), 301.0)),
        grid=mesh)                                   # the CoupledESMDriver.grid property

    cs = cmip_column_state(coupled)
    assert cs.u.shape == (mesh.nCells, nlev) and cs.v.shape == (mesh.nCells, nlev)
    assert cs.u_edge is not None and cs.u_edge.shape == (mesh.nEdges, nlev)
    u_ref, v_ref = reconstruct_cell_velocity(
        jnp.asarray(u_edge, dtype=cs.T.dtype), mesh)
    np.testing.assert_allclose(np.asarray(cs.u), np.asarray(u_ref), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(cs.v), np.asarray(v_ref), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(cs.sst_K), 301.0)   # the COUPLED ocean SST


def test_make_run_fn_builds_driver_from_config_and_means():
    """make_run_fn(config) builds a fresh driver from the config each call and
    returns the time mean — the run_amip_fn/run_cmip_fn make_compare_fn expects."""
    seen = {}

    def build_driver(config):
        seen["config"] = config
        return _FakeDriver([_state(0.0), _state(2.0)])  # mean scale = 1

    run_fn = make_run_fn(build_driver, _extract)
    mean = run_fn("my-config")
    assert seen["config"] == "my-config"
    np.testing.assert_allclose(np.asarray(mean.T), 281.0, rtol=1e-12)


def test_make_run_fn_output_depends_on_config():
    """config flows through build_driver into the run output (the loop's premise:
    a different config ⇒ a different scored state), not silently ignored."""
    def build_driver(config):
        return _FakeDriver([_state(float(config))])  # state scales with config

    run_fn = make_run_fn(build_driver, _extract)
    mean_a = run_fn(2.0)
    mean_b = run_fn(5.0)
    assert not bool(jnp.allclose(mean_a.T, mean_b.T))
    np.testing.assert_allclose(np.asarray(mean_a.T), 282.0, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(mean_b.T), 285.0, rtol=1e-12)


def _tiny_coupled(days, diag_days, n_lat=8, nlev=5):
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.grids.latlon import create_latlon_grid

    atm_config = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=n_lat, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=diag_days),
        radiation="gray", days=days,
    )
    ocean_grid = create_latlon_grid(n_lat=n_lat, n_lon=2 * n_lat)
    driver = CoupledESMDriver(atm_config, PRESETS["aquaplanet"](), ocean_grid=ocean_grid)
    driver.setup()
    return driver


@pytest.mark.slow
def test_run_to_column_mean_real_cmip():
    """A real tiny coupled (CMIP) run produces a finite time-mean ColumnState
    with the coupled SST; multiple diagnostic segments are sampled + averaged."""
    driver = _tiny_coupled(days=1.0, diag_days=0.5)  # ~2 segments
    n_samples = {"n": 0}

    def counting_extract(d, day, dt):
        n_samples["n"] += 1
        return cmip_column_state(d, day, dt)

    mean = run_to_column_mean(driver, counting_extract)
    assert n_samples["n"] > 1  # multiple diagnostic segments sampled + averaged
    assert isinstance(mean, ColumnState)
    n_lat, n_lon, nlev = mean.T.shape
    assert (n_lat, n_lon, nlev) == (8, 16, 5)
    for name in ("T", "q_v", "u", "v", "p_s"):
        assert bool(jnp.all(jnp.isfinite(getattr(mean, name))))
    # CMIP: the coupled-ocean SST is carried through.
    assert mean.sst_K is not None
    assert mean.sst_K.shape == (8, 16)
    assert bool(jnp.all(jnp.isfinite(mean.sst_K)))


@pytest.mark.slow
def test_segment_callback_compose_is_optional():
    """CoupledESMDriver.run with no segment_callback still runs (byte-identical
    coupling path); the new optional hook does not break the plain run."""
    driver = _tiny_coupled(days=0.5, diag_days=0.5)
    status = driver.run()  # no segment_callback → plain coupled run
    assert isinstance(status, str)


@pytest.mark.slow
def test_make_compare_fn_real_cmip_run_to_mean():
    """Full real compare path through the loop adapters: make_run_fn(real CMIP
    build) → make_compare_fn → compare_fn(config) runs the coupled model,
    time-means it, and scores it against a synthetic ERA5 reference, emitting the
    worst-column manifest — the run→time-mean→compare→score chain end-to-end."""
    from legoesm.training.correction_loop import make_compare_fn

    rad2deg = 180.0 / np.pi
    probe = _tiny_coupled(days=1.0, diag_days=0.5)  # for grid/sigma only (not run)
    sigma = probe._atm.sigma
    grid = probe._atm.grid
    lat_deg = jnp.asarray(np.asarray(grid.grid_lat) * rad2deg)
    lon_deg = jnp.asarray(np.asarray(grid.grid_lon) * rad2deg)

    def build_driver(config):  # config unused here (the run is config-fixed)
        return _tiny_coupled(days=1.0, diag_days=0.5)

    run_fn = make_run_fn(build_driver, cmip_column_state)

    # Reference = one real time-mean run with a localized +6 K cold bias in one
    # column (deterministic run ⇒ that column is the worst).
    model0 = run_fn(None)
    n_lat, n_lon, _ = model0.T.shape
    bias = np.zeros((n_lat, n_lon))
    bias[n_lat // 2, n_lon // 2] = 6.0
    reference = model0._replace(T=model0.T - jnp.asarray(bias)[:, :, None])

    compare_fn = make_compare_fn(
        reference=reference,
        sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=lat_deg, lon_deg=lon_deg,
        area_weights=jnp.ones((n_lat, n_lon)),
        n_worst=1, run_amip_fn=run_fn,
    )
    result = compare_fn(None)
    assert result.combined_score.shape == (n_lat, n_lon)
    assert bool(jnp.all(jnp.isfinite(result.combined_score)))
    assert len(result.manifest) == 1
    # The +6 K biased column is the worst — and by a clear DOMINANCE MARGIN over
    # the runner-up (the run is deterministic, so non-biased columns score ~0),
    # so the assertion is robust to any minor run-to-run difference, not flaky.
    assert result.manifest[0].grid_index == (n_lat // 2, n_lon // 2)
    score = np.asarray(result.combined_score)
    biased = float(score[n_lat // 2, n_lon // 2])
    others = score.copy()
    others[n_lat // 2, n_lon // 2] = -np.inf
    runner_up = float(others.max())
    assert biased > 0.0
    assert runner_up < 0.5 * biased  # biased column ≥ 2× any other column


def test_model_phis_from_driver_extracts_topography():
    """The model's STATIC topography (g·z_s) is extracted from the built driver's
    state and UNWRAPPED to a raw array — the consistent-topography source for the
    orographic geostrophic term (iter 117/122), avoiding a mismatched-phis footgun."""
    from legoesm.core.field import Field

    phis_arr = jnp.full((8, 16), 5000.0)            # an arbitrary g·z_s [m²/s²]
    # A non-spectral grid state (grid_winds_from_spectral passes it through unchanged).
    driver = SimpleNamespace(
        state=SimpleNamespace(phis=Field(phis_arr)), grid=None, sigma=None)
    out = model_phis_from_driver(driver)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(phis_arr))   # Field unwrapped


def test_model_phis_from_driver_none_for_flat_model():
    """A flat/aquaplanet model (state carries no phis) ⇒ None, so the orographic
    term correctly stays OFF (the geostrophic forcing falls back to above-surface)."""
    driver = SimpleNamespace(state=SimpleNamespace(), grid=None, sigma=None)
    assert model_phis_from_driver(driver) is None


def test_model_phis_from_driver_none_for_zero_topography():
    """A real flat ModelDriver initialises phis = zeros(...) (NOT absent), so an
    identically-zero field must ALSO map to None — the honest "None for flat"
    contract. Otherwise the orographic term would activate with zero topography
    (breaking exact flat-path parity) and a 'terrain required but flat' fail-loud
    that checks `is None` would silently pass. Both Field-wrapped and raw zeros."""
    from legoesm.core.field import Field

    zeros = jnp.zeros((8, 16))
    driver_field = SimpleNamespace(
        state=SimpleNamespace(phis=Field(zeros)), grid=None, sigma=None)
    assert model_phis_from_driver(driver_field) is None
    driver_raw = SimpleNamespace(
        state=SimpleNamespace(phis=np.zeros((8, 16))), grid=None, sigma=None)
    assert model_phis_from_driver(driver_raw) is None


def test_phis_or_none_if_flat():
    """The shared flat-detection predicate (reused by model_phis_from_driver AND the
    CLI's resolve_orographic_phis): None→None, identically-zero→None (effectively
    flat), ANY non-zero→passthrough unchanged."""
    assert phis_or_none_if_flat(None) is None
    assert phis_or_none_if_flat(jnp.zeros((4, 8))) is None
    assert phis_or_none_if_flat(np.zeros((4, 8))) is None        # raw numpy zeros too
    arr = jnp.zeros((4, 8)).at[2, 1].set(1500.0)                 # one non-zero cell
    out = phis_or_none_if_flat(arr)
    assert out is not None
    np.testing.assert_array_equal(np.asarray(out), np.asarray(arr))


def test_model_phis_from_driver_keeps_partially_zero_topography():
    """A field with ANY non-zero element (e.g. land beside ocean at sea level) is
    REAL terrain — returned unchanged, NOT None. Only an IDENTICALLY-zero field is
    'flat'; a coastline (∇phis ≠ 0) must keep the orographic term ON."""
    from legoesm.core.field import Field

    arr = jnp.zeros((4, 8)).at[1, 2].set(3000.0)   # one mountain cell, rest sea level
    driver = SimpleNamespace(state=SimpleNamespace(phis=Field(arr)), grid=None, sigma=None)
    out = model_phis_from_driver(driver)
    assert out is not None
    np.testing.assert_array_equal(np.asarray(out), np.asarray(arr))


def test_model_phis_from_driver_none_when_no_state():
    """A driver with no `state` attribute ⇒ None (no crash)."""
    assert model_phis_from_driver(SimpleNamespace()) is None


def test_model_phis_from_driver_routes_spectral_state(monkeypatch):
    """A SPECTRAL driver's state is routed through grid_winds_from_spectral (which
    synthesizes the grid `phis`) WITH the driver's grid + sigma — so the spectral
    path is reached; the synthesis itself is covered by grid_winds_from_spectral's
    own tests. Confirms the helper delegates correctly + unwraps the result."""
    import legoesm.training.run_to_column_mean as rtcm
    from legoesm.core.field import Field

    seen = {}

    def spy_grid_winds(state, grid, sigma):
        seen["args"] = (state, grid, sigma)
        return SimpleNamespace(phis=Field(jnp.full((4, 8), 3000.0)))  # synthesized phis

    monkeypatch.setattr(rtcm, "grid_winds_from_spectral", spy_grid_winds)
    driver = SimpleNamespace(state="SPECTRAL_STATE", grid="GAUSS_GRID", sigma="SIGMA")
    out = model_phis_from_driver(driver)
    assert seen["args"] == ("SPECTRAL_STATE", "GAUSS_GRID", "SIGMA")   # grid+sigma passed
    np.testing.assert_array_equal(np.asarray(out), 3000.0)            # synthesized phis unwrapped


def test_model_phis_from_driver_raw_array_passthrough():
    """A state whose `phis` is already a RAW array (not a Field) passes through
    UNCHANGED — the isinstance(Field) guard must NOT unwrap a numpy array's `.data`
    memoryview buffer (the hasattr-'data' footgun Codex flagged)."""
    raw = np.full((4, 8), 4200.0)                 # numpy array HAS a .data memoryview
    driver = SimpleNamespace(state=SimpleNamespace(phis=raw), grid=None, sigma=None)
    out = model_phis_from_driver(driver)
    assert isinstance(out, np.ndarray)            # NOT a memoryview
    np.testing.assert_array_equal(out, raw)
