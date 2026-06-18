"""Smoke test for the LES-informed correction CAMPAIGN driver composition.

Exercises the genuinely-new orchestration (``make_clubb_build_driver`` +
``make_les_diagnose_fn`` + ``build_correction_campaign``) WITHOUT a real model
run: a mock driver supplies the column state and a mock LES supplies the plane
state, so the REAL process_column forcing-extract + diagnose path runs (the real
model run + real LES are covered by iter 35/37 and iter 20 respectively).
"""

from __future__ import annotations

from types import SimpleNamespace

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
    _distributed_campaign_kwargs,
    build_correction_campaign,
    build_distributed_multi_correction_campaign,
    build_multi_correction_campaign,
    make_base_driver_builder,
    make_clubb_build_driver,
    make_les_diagnose_fn,
    refuse_unsupported_multirank,
)


def test_refuse_unsupported_multirank_guards_cli():
    """The single-process CLI refuses an mpirun -np >1 launch LOUDLY (it wires none
    of the distributed hooks); a single rank (or no MPI) is allowed (iter 88)."""
    refuse_unsupported_multirank(comm_size=1)          # single rank: allowed
    with pytest.raises(SystemExit, match="SINGLE-PROCESS CLI"):
        refuse_unsupported_multirank(comm_size=2)      # multi-rank: refused

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


def test_make_les_diagnose_fn_mpas_routes_edge_velocity():
    """An MPAS worst column spins off its LES end-to-end (iter 76): the model_ctx
    carries the native u_edge, make_les_diagnose_fn routes it to the Voronoi
    forcing extractor (grid=mesh, v=None), and the mock LES yields a diagnosis."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    diagnose_fn = make_les_diagnose_fn(
        mesh, sigma, les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les)

    # An MPAS ColumnState: cell T/q_v/p_s + the NATIVE edge velocity in u_edge.
    u_edge = 6.0 * jnp.cos(jnp.asarray(mesh.angleEdge))[:, None] * jnp.ones((1, nlev))
    mpas_ctx = ColumnState(
        T=jnp.full((mesh.nCells, nlev), 285.0),
        q_v=jnp.full((mesh.nCells, nlev), 6e-3),
        u=jnp.zeros((mesh.nCells, nlev)),     # cell winds present but unused (routed)
        v=jnp.zeros((mesh.nCells, nlev)),
        p_s=jnp.full((mesh.nCells,), 1.0e5),
        u_edge=u_edge)

    cell = 40

    class _Env:
        cape_J_kg = 200.0  # noqa: N815

    class _Rec:
        grid_index = (cell,)                  # arity-1 cell index for MPAS
        lat_deg = float(np.rad2deg(np.asarray(mesh.latCell)[cell]))
        environment = _Env()

    out = diagnose_fn(_Rec(), mpas_ctx)
    assert out.K.shape == (_SMALL_RES.nlev - 1,)
    assert bool(jnp.all(jnp.isfinite(out.K)))


def test_make_les_diagnose_fn_mpas_u_edge_on_wrong_grid_raises():
    """u_edge set but a non-Voronoi grid → loud grid/state mismatch (Codex)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    diagnose_fn = make_les_diagnose_fn(
        grid, create_sigma_coordinate(5),
        les_config=ColumnLESConfig(regime=_SMALL_REGIME), run_les_fn=_mock_run_les)
    ctx = _full_grid_state()._replace(u_edge=jnp.zeros((10, 5)))

    class _Rec:
        grid_index = (4, 8)
        lat_deg = 20.0

    with pytest.raises(ValueError, match="not a VoronoiMesh"):
        diagnose_fn(_Rec(), ctx)


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
        run_les_fn=_mock_run_les, n_worst=1,
        # Test the config→model wiring (a per-column C_K is applied); the mock
        # compare has no real improvement signal, so disable the monotonic gate
        # that would otherwise reject this round (gate tested in test_correction_loop).
        accept_only_if_improved=False)

    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.n_diagnosed == 1
    # The campaign produced a per-column clubb C_K (the loop closed).
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 1
    assert result.final_config.C_K.shape == (8 * 16,)


def _mpas_base_config(nlev=5):
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=2, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="mpas"),
        radiation="gray", turbulence="clubb_lite")


def _mpas_full_state(mesh, nlev=5, *, bias_cell=None, bias_dt=0.0):
    """An MPAS comparison ColumnState: cell T/q_v/p_s + the native u_edge."""
    temp = np.full((mesh.nCells, nlev), 285.0)
    if bias_cell is not None:
        temp[bias_cell] += bias_dt
    u_edge = 6.0 * np.cos(np.asarray(mesh.angleEdge))[:, None] * np.ones((1, nlev))
    return ColumnState(
        T=jnp.asarray(temp), q_v=jnp.full((mesh.nCells, nlev), 6e-3),
        u=jnp.zeros((mesh.nCells, nlev)), v=jnp.zeros((mesh.nCells, nlev)),
        p_s=jnp.full((mesh.nCells,), 1.0e5),
        sst_K=jnp.full((mesh.nCells,), 290.0),
        u_edge=jnp.asarray(u_edge))


@pytest.mark.parametrize("feedback_strategy", ["static", "environment"])
def test_build_correction_campaign_mpas_one_round(feedback_strategy):
    """CAPSTONE (iters 73-76): the FULL MPAS pipeline composes through the REAL
    build_correction_campaign — cell compare/rank → Voronoi forcing extract (via
    the native u_edge) → mock LES → per-CELL clubb C_K feedback → re-run → finite
    bias. NON-VACUOUS: a SHEARED mock LES gives a VALID clubb_coefficient
    diagnosis, so the diagnosed C_K (clamped to bounds) actually REPLACES the
    background at the worst cell (static) / spreads via the env regression +
    produces a deploy kernel (environment) — not a no-op. Mock driver + mock LES;
    the C_K-changes-MODEL-output mechanism is iter 35/37."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    bias_cell = 37
    bg = float(CLUBBLiteConfig().C_K)              # production default (0.4)
    model_state = _mpas_full_state(mesh, nlev)
    # reference = model with one cold-biased cell → a deterministic worst cell.
    reference = _mpas_full_state(mesh, nlev, bias_cell=bias_cell, bias_dt=-6.0)

    def build_base(cfg):           # mock: fixed MPAS state regardless of C_K
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        # clubb_coefficient + a SHEARED mock → a VALID, non-background C_K diagnosis.
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        feedback_strategy=feedback_strategy,
        accept_only_if_improved=False)

    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.n_diagnosed == 1                 # the worst CELL spun off its LES
    assert bool(np.isfinite(float(it.bias.updated_bias)))
    # A per-CELL clubb C_K reached the config (the loop closed on the 1-D cell axis).
    ck = np.asarray(result.final_config.C_K)
    assert ck.ndim == 1 and ck.shape == (mesh.nCells,)
    assert np.all(np.isfinite(ck))
    # NON-VACUOUS: the correction actually changed C_K away from the background.
    assert not np.allclose(ck, bg), "the diagnosed C_K never reached the cells"
    if feedback_strategy == "static":
        # The diagnosed value landed on the worst CELL; the rest stay background.
        assert not np.isclose(float(ck[bias_cell]), bg)
        np.testing.assert_allclose(np.delete(ck, bias_cell), bg)
    else:
        # The env-strategy round produced a transferable deploy kernel (iter 70).
        assert it.env_kernel is not None and it.env_kernel.field == "C_K"


def test_build_correction_campaign_owned_cell_mask_excludes_halo():
    """WIRING (iter 86): the ``valid_mask`` param threads through
    build_correction_campaign → compose_compare_fn → make_compare_fn, so a
    DISTRIBUTED-MPAS owned-cell mask keeps a HALO cell out of the ranking. A halo
    cell carrying the globally-LARGEST bias is masked out; the campaign instead
    spins off + corrects the worst OWNED cell — proving the mask both threads
    through AND changes the outcome (non-vacuous)."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.training.compare_reanalysis import owned_cell_valid_mask

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    n_owned = 120                          # owned = [0, 120); halo = [120, 162)
    halo_worst, owned_worst = 150, 50
    bg = float(CLUBBLiteConfig().C_K)
    model_state = _mpas_full_state(mesh, nlev)
    # reference: TWO cold-biased cells — the halo one colder (worst), the owned one
    # second.  Without the mask the halo cell is worst; with it, the owned cell is.
    ref = _mpas_full_state(mesh, nlev)
    rt = np.asarray(ref.T).copy()
    rt[halo_worst] -= 12.0
    rt[owned_worst] -= 8.0
    reference = ref._replace(T=jnp.asarray(rt))
    owned_mask = owned_cell_valid_mask(jnp.arange(mesh.nCells) < n_owned)

    def build_base(cfg):
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False, valid_mask=owned_mask)

    it = result.iterations[0]
    assert it.n_diagnosed == 1
    ck = np.asarray(result.final_config.C_K)
    # The masked HALO cell was NEVER diagnosed (stays background); the worst OWNED
    # cell got the correction.
    assert np.isclose(float(ck[halo_worst]), bg), "a halo cell was wrongly corrected"
    assert not np.isclose(float(ck[owned_worst]), bg), "the owned worst cell was skipped"


def test_build_correction_campaign_mpas_les_budget_clusters_cells():
    """The LES-cost reduction (env clustering) works on the MPAS cell layout: 4
    worst cells in 2 distinct-SST environments + les_budget=2 → only 2 LES run
    (the cluster representatives), but all 4 cells are corrected. Exercises
    cluster_columns_by_environment over the 1-D cell manifest."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    # 4 worst cells split into 2 SST environments (2 cells each) → 2 clusters.
    cold_cells, warm_cells = [10, 20], [120, 130]
    worst = cold_cells + warm_cells

    def _state(*, biased):
        temp = np.full((mesh.nCells, nlev), 285.0)
        if biased:
            for c in worst:
                temp[c] -= 6.0
        sst = np.full((mesh.nCells,), 285.0)
        sst[warm_cells] = 300.0                    # the env tag that splits clusters
        u_edge = 6.0 * np.cos(np.asarray(mesh.angleEdge))[:, None] * np.ones((1, nlev))
        return ColumnState(
            T=jnp.asarray(temp), q_v=jnp.full((mesh.nCells, nlev), 6e-3),
            u=jnp.zeros((mesh.nCells, nlev)), v=jnp.zeros((mesh.nCells, nlev)),
            p_s=jnp.full((mesh.nCells,), 1.0e5), sst_K=jnp.asarray(sst),
            u_edge=jnp.asarray(u_edge))

    model_state = _state(biased=False)
    reference = _state(biased=True)               # the 4 biased cells are the worst

    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=4, les_budget=2,
        accept_only_if_improved=False)

    it = result.iterations[0]
    assert it.n_corrected == 4                    # all 4 worst cells corrected
    assert it.n_diagnosed == 2                    # but only 2 LES (cluster reps)
    assert np.isfinite(float(it.bias.baseline_bias))
    assert np.isfinite(float(it.bias.updated_bias))


def _gaussian_full_state(grid, nlev=5, *, bias_col=None, bias_dt=0.0):
    """A Gaussian (spectral) comparison ColumnState on the (n_lat, n_lon) grid.

    The zonal wind is DIVERGENT (``u = U cos λ``, varying with longitude) so the SH
    continuity chain in the iter-83 Gaussian extractor produces a genuinely NONZERO
    ω — a solid-body (``u = U cos φ``) field is non-divergent (∇·v = 0) and would
    leave ω ≈ 0, making the spectral path vacuously exercised.  T/q_v/p_s are
    uniform with one optionally cold-biased column for a deterministic worst column;
    the winds are identical in model and reference, so the worst-column ranking is
    driven purely by the T bias.  ALL arrays are float64 (the Gaussian extractor
    hard-raises on a float32 ``T``)."""
    n_lat, n_lon = grid.n_lat, grid.n_lon
    temp = np.full((n_lat, n_lon, nlev), 285.0)
    if bias_col is not None:
        temp[bias_col] += bias_dt
    lon = np.asarray(grid.grid_lon)[:, :, None]    # (n_lat, n_lon, 1)
    u = 15.0 * np.cos(lon) * np.ones((n_lat, n_lon, nlev))   # divergent → ω ≠ 0
    return ColumnState(
        T=jnp.asarray(temp, dtype=jnp.float64),
        q_v=jnp.full((n_lat, n_lon, nlev), 6e-3, dtype=jnp.float64),
        u=jnp.asarray(u, dtype=jnp.float64),
        v=jnp.zeros((n_lat, n_lon, nlev), dtype=jnp.float64),
        p_s=jnp.full((n_lat, n_lon), 1.0e5, dtype=jnp.float64),
        sst_K=jnp.full((n_lat, n_lon), 290.0, dtype=jnp.float64))


def test_build_correction_campaign_gaussian_one_round(monkeypatch):
    """CAPSTONE (iters 83-84): the FULL Gaussian/spectral pipeline composes through
    the REAL build_correction_campaign — 2-D (n_lat, n_lon) compare/rank → the
    iter-83 SH-divergence forcing extract (``extract_column_forcing_gaussian`` via
    the GaussianGrid dispatch branch, col_index=(i_lat, i_lon)) → sheared mock LES →
    per-COLUMN clubb C_K feedback → re-run → finite bias.

    NON-VACUOUS, each property MACHINE-CHECKED rather than inferred:
      1. A SPY wraps ``extract_column_forcing_gaussian`` and asserts the spectral
         branch fired EXACTLY ONCE, for the BIASED worst column, producing a FINITE,
         NONZERO ω (the SH continuity chain genuinely ran — the divergent wind makes
         ω ≠ 0).  The lat-lon FD extractor is monkeypatched to RAISE, so a silent
         FD fallback is impossible (Codex iter-85 issues 1+2).
      2. A SHEARED mock LES gives a VALID clubb_coefficient diagnosis, so the
         diagnosed C_K actually REPLACES the background at the worst column and the
         OTHER columns stay at the background (Codex iter-85 issue 4).
      3. The float64 ``T`` survives the campaign into the extractor's float64 guard
         (a float32 ``T`` would raise; the other fields are widened to T's dtype).
    Mock driver + mock LES; the C_K-changes-model-output mechanism is iter 35/37."""
    from legoesm.atmosphere.dynamics import column_large_scale_extract as clse
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_gaussian_grid(n_max=21, dealiasing="linear")
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    # mid-grid, off the poles AND off the u=15cos(λ) divergence nodes (λ=0,π at
    # i_lon 0,22) so the SH continuity chain produces a genuinely nonzero ω there.
    bias_col = (11, 10)
    bg = float(CLUBBLiteConfig().C_K)              # production default (0.4)
    model_state = _gaussian_full_state(grid, nlev)
    # reference = model with one cold-biased column → a deterministic worst column.
    reference = _gaussian_full_state(grid, nlev, bias_col=bias_col, bias_dt=-6.0)

    # ROUTING + FORCING probe (Codex iter-85): the dispatcher resolves both
    # extractors as module globals, so patch them on the module the dispatcher reads.
    real_gaussian = clse.extract_column_forcing_gaussian
    calls: dict = {"n": 0}

    def _spy_gaussian(*, col_index, **kw):
        out = real_gaussian(col_index=col_index, **kw)
        omega = np.asarray(out.omega)
        assert np.all(np.isfinite(omega)), "spectral extractor produced non-finite ω"
        assert float(np.max(np.abs(omega))) > 1e-4, "ω is vacuously ~0 (no divergence)"
        calls["n"] += 1
        calls["col_index"] = tuple(int(c) for c in col_index)
        return out

    def _no_latlon(**kw):  # noqa: ARG001
        raise AssertionError("lat-lon FD extractor called for a GaussianGrid")

    monkeypatch.setattr(clse, "extract_column_forcing_gaussian", _spy_gaussian)
    monkeypatch.setattr(clse, "extract_column_forcing_latlon", _no_latlon)

    def build_base(cfg):           # mock: fixed Gaussian state regardless of C_K
        return _FakeDriver(model_state)

    def extract(driver, day, dt):  # noqa: ARG001
        return driver.state

    result = build_correction_campaign(
        base_atm_config=_base_config(), build_base_driver=build_base,
        extract_column_state=extract, reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.asarray(grid.grid_area), n_iterations=1,
        # clubb_coefficient + a SHEARED mock → a VALID, non-background C_K diagnosis.
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)

    assert len(result.iterations) == 1
    it = result.iterations[0]
    assert it.n_diagnosed == 1                 # the worst COLUMN spun off its LES
    assert bool(np.isfinite(float(it.bias.updated_bias)))
    # ROUTING: the SPECTRAL branch fired exactly once, for the biased worst column
    # (the lat-lon FD extractor would have raised — proven not taken).
    assert calls["n"] == 1, "the Gaussian/spectral extractor was not the path taken"
    assert calls["col_index"] == bias_col, "ranked the wrong column as worst"
    # A per-column clubb C_K reached the config, flattened on the (n_lat*n_lon) axis.
    ck = np.asarray(result.final_config.C_K)
    assert ck.ndim == 1 and ck.shape == (grid.n_lat * grid.n_lon,)
    assert np.all(np.isfinite(ck))
    # NON-VACUOUS: the diagnosed C_K landed on the worst column, others stay at bg.
    flat_worst = bias_col[0] * grid.n_lon + bias_col[1]
    assert not np.isclose(ck[flat_worst], bg), "diagnosed C_K never reached the column"
    np.testing.assert_allclose(np.delete(ck, flat_worst), bg)  # rest untouched


def _mock_run_les_sheared(setup):
    """Mock plane-LES with a mean-wind shear so the clubb_coefficient diagnosis
    yields a VALID dimensionless C_K (the rest-state mock has no shear)."""
    state = _mock_run_les(setup)
    z = jnp.asarray(setup.height_coord.z_full)
    ny, nx = setup.grid.ny, setup.grid.nx
    u = (0.01 * z)[None, None, :] * jnp.ones((ny, nx, z.shape[0]))   # constant shear
    return state._replace(u=state.u.replace(data=u))


def test_build_correction_campaign_clubb_coefficient_method():
    """The dimensionless clubb_coefficient diagnosis is wired end-to-end: the
    campaign auto-populates l_mix_max from the GCM config, the LES diagnoses a
    DIMENSIONLESS C_K, the loop reduces it (same method), and a per-column,
    in-bounds C_K is produced — proving the units-correct path runs."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)            # wiring test (no improvement signal)

    it = result.iterations[0]
    assert it.n_diagnosed == 1
    ck = np.asarray(result.final_config.C_K).reshape(-1)
    assert ck.shape == (8 * 16,)
    # clip_to_bounds default ON ⇒ the dimensionless C_K stays in (0.1, 1.2).
    assert float(ck.min()) >= 0.1 and float(ck.max()) <= 1.2


def test_build_correction_campaign_rejects_multi_methods():
    """build_correction_campaign is single-coefficient: a diagnosis_methods config
    (which makes process_column return a dict) is rejected up front, not crashed
    downstream (the simultaneous multi-coefficient campaign is not wired here)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    with pytest.raises(ValueError, match="diagnosis_methods"):
        build_correction_campaign(
            base_atm_config=_base_config(),
            build_base_driver=lambda cfg: _FakeDriver(model_state),
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            reference=model_state, sigma=sigma, grid=grid,
            area_weights=jnp.ones((8, 16)), n_iterations=1,
            les_config=ColumnLESConfig(
                regime=_SMALL_REGIME,
                diagnosis_methods=("clubb_coefficient", "prandtl_number"),
                clubb_l_mix_max=100.0),
            run_les_fn=_mock_run_les_sheared, n_worst=1)


def test_build_multi_correction_campaign_mpas_three_coefficients():
    """The SIMULTANEOUS multi-coefficient campaign composes on the MPAS cell
    layout: C_K + Pr_t + C_eps co-corrected from ONE Voronoi LES spin-off per
    worst CELL (via the native u_edge through the SAME make_les_diagnose_fn) →
    three in-bounds per-CELL (nCells,) fields. Exercises run_multi_correction_*
    + the {method: diagnosis} dict path for MPAS."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    model_state = _mpas_full_state(mesh, nlev)
    reference = _mpas_full_state(mesh, nlev, bias_cell=37, bias_dt=-6.0)

    result = build_multi_correction_campaign(
        base_atm_config=_mpas_base_config(nlev),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_K", "Pr_t", "C_eps"), accept_only_if_improved=False)

    assert set(result.final_fields) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    ck = np.asarray(result.final_config.C_K).reshape(-1)
    prt = np.asarray(result.final_config.Pr_t).reshape(-1)
    ceps = np.asarray(result.final_config.C_eps).reshape(-1)
    assert ck.shape == (mesh.nCells,) and prt.shape == (mesh.nCells,)
    assert ceps.shape == (mesh.nCells,)
    assert float(ck.min()) >= 0.1 and float(ck.max()) <= 1.2       # C_K bounds
    assert float(prt.min()) >= 0.3 and float(prt.max()) <= 1.5     # Pr_t bounds
    assert float(ceps.min()) >= 0.06 and float(ceps.max()) <= 0.6  # C_eps bounds


def test_build_correction_campaign_prandtl_number_method():
    """The prandtl_number diagnosis is wired end-to-end: the campaign selects the
    clubb_lite_Pr_t promotion + Pr_t background by method, the LES diagnoses a
    DIMENSIONLESS Pr_t, and an in-bounds per-column Pr_t is produced."""
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="prandtl_number"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)

    # The corrected coefficient is Pr_t (not C_K); C_K stays the scalar default.
    prt = np.asarray(result.final_config.Pr_t).reshape(-1)
    assert prt.shape == (8 * 16,)
    assert float(prt.min()) >= 0.3 and float(prt.max()) <= 1.5   # Pr_t bounds
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 0   # C_K untouched
    assert result.final_config.C_K == CLUBBLiteConfig().C_K


def test_build_correction_campaign_bias_tol_early_stops():
    """build_correction_campaign forwards bias_tol/patience: a mock driver that
    never improves the bias is rejected every round, so the campaign stops early
    ('converged') instead of running all n_iterations."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),  # fixed → never improves
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=10,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        bias_tol=1e-9, patience=2)            # default gate ON
    assert result.stop_reason == "converged"
    assert len(result.iterations) == 2        # 2 rejected rounds → early stop


def test_build_correction_campaign_default_gate_rejects_non_improving():
    """build_correction_campaign defaults the monotonic gate ON: the mock driver
    returns a fixed state regardless of C_K, so the round does not lower the bias
    and is REJECTED — the config stays the uncorrected scalar default."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1)            # default gate ON

    assert result.accepted == (False,)                  # non-improving → rejected
    assert not bool(result.iterations[0].bias.improved)
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 0  # unchanged default


def test_make_base_driver_builder_dispatch():
    """The run-mode dispatch returns the right SST extractor + raises on a bad
    mode / a CMIP call missing the coupled pieces (dispatch hardening, iter 40)."""
    from legoesm.training.run_to_column_mean import (
        amip_column_state,
        cmip_column_state,
    )

    _build_amip, extract_amip = make_base_driver_builder("amip")
    assert extract_amip is amip_column_state
    assert callable(_build_amip)

    # cmip needs ONLY coupled_preset; ocean_grid is optional (None ⇒ the coupled
    # driver uses its own atm grid, same-grid coupling).
    _build_cmip, extract_cmip = make_base_driver_builder(
        "cmip", coupled_preset=object())
    assert extract_cmip is cmip_column_state
    assert callable(_build_cmip)

    with pytest.raises(ValueError, match="unknown mode"):
        make_base_driver_builder("xyz")
    with pytest.raises(ValueError, match="requires coupled_preset"):
        make_base_driver_builder("cmip")  # missing coupled_preset


@pytest.mark.slow
def test_make_base_driver_builder_amip_builds_real_driver():
    """The AMIP builder constructs + sets up a real ModelDriver (the campaign's
    build_base_driver path that main() uses)."""
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.model_driver import ModelDriver

    build_amip, _ = make_base_driver_builder("amip")
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite")
    driver = build_amip(cfg)
    assert isinstance(driver, ModelDriver)
    assert driver.grid is not None and driver.sigma is not None


@pytest.mark.slow
def test_make_base_driver_builder_cmip_builds_real_driver():
    """The CMIP builder constructs + sets up a real CoupledESMDriver with
    ocean_grid=None (same-grid coupling: ocean on the atm grid), exposing the
    state/q_v/ocean_state that cmip_column_state reads."""
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    build_cmip, _ = make_base_driver_builder(
        "cmip", coupled_preset=PRESETS["aquaplanet"](), ocean_grid=None)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite")
    driver = build_cmip(cfg)
    assert isinstance(driver, CoupledESMDriver)
    # cmip_column_state reads these; same-grid ⇒ SST on the atm column shape.
    assert driver.state is not None and driver.q_v is not None
    assert tuple(driver.ocean_state.T_sfc.data.shape) == (8, 16)


@pytest.mark.slow
def test_build_correction_campaign_checkpoint_passthrough():
    """build_correction_campaign forwards checkpoint_callback + start_round to
    run_correction_campaign (the restart wiring main() uses)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    calls = []

    def ckpt(round_idx, res, field):
        calls.append((round_idx, np.asarray(field).copy(), res))

    # initial_field (resume base): every non-worst column keeps this value, so the
    # checkpointed accumulated field proves initial_field was forwarded + used.
    # Gate OFF: this is the forwarding/wiring test (the gate is tested separately),
    # so the round is accepted and the worst column gets a correction on top of base.
    init_field = jnp.full((8, 16), 0.55)
    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1,
        initial_field=init_field, start_round=7, checkpoint_callback=ckpt,
        accept_only_if_improved=False)

    assert len(calls) == 1
    round_idx, field, res = calls[0]
    assert round_idx == 7            # start_round forwarded; callback fired once
    flat = field.reshape(-1)
    worst = 4 * 16 + 8               # the +6 K worst column (row-major)
    others = np.delete(flat, worst)
    # Non-worst columns keep initial_field (0.55), NOT the default background 0.4
    # — so initial_field was genuinely forwarded + used as the round's base.
    np.testing.assert_allclose(others, 0.55)
    # Resume contract (the CLI checkpoint fix relies on this): the persisted
    # accumulated field IS the accepted state and its flattened form equals the
    # accepted config's per-column C_K — so rebuilding the config FROM the field on
    # resume cannot desync. (Accepted round ⇒ res.updated_config.C_K matches too.)
    np.testing.assert_allclose(flat, np.asarray(res.updated_config.C_K))
    np.testing.assert_allclose(
        flat, np.asarray(result.final_config.C_K).reshape(-1))


@pytest.mark.slow
def test_build_correction_campaign_mpas_resume_accumulates_on_cells():
    """The restartable campaign (§1) works on the MPAS cell layout: an
    initial_field (nCells,) is the round-0 base, so every non-worst CELL keeps it
    and the worst CELL is corrected ON TOP — the (ncol,)-field accumulation +
    grid.grid_shape_2d=(nCells,) resume path is grid-agnostic."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2)
    nlev = 5
    sigma = create_sigma_coordinate(nlev)
    bias_cell = 37
    model_state = _mpas_full_state(mesh, nlev)
    reference = _mpas_full_state(mesh, nlev, bias_cell=bias_cell, bias_dt=-6.0)

    calls = []
    init_field = jnp.full((mesh.nCells,), 0.55)       # the resume base
    result = build_correction_campaign(
        base_atm_config=_mpas_base_config(nlev),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=mesh,
        area_weights=jnp.asarray(mesh.grid_area), n_iterations=1,
        les_config=ColumnLESConfig(
            regime=_SMALL_REGIME, diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        initial_field=init_field, start_round=7,
        checkpoint_callback=lambda r, res, f: calls.append((r, np.asarray(f).copy())),
        accept_only_if_improved=False)

    assert len(calls) == 1 and calls[0][0] == 7       # start_round forwarded
    ck = np.asarray(result.final_config.C_K)
    assert ck.shape == (mesh.nCells,)
    # Non-worst cells keep the resume base; the worst cell is corrected on top.
    np.testing.assert_allclose(np.delete(ck, bias_cell), 0.55)
    assert not np.isclose(float(ck[bias_cell]), 0.55)


@pytest.mark.slow
def test_build_correction_campaign_environment_strategy():
    """feedback_strategy='environment' runs end-to-end through the campaign
    (column_environment_grid computes the full-grid env each round) → a
    per-column C_K."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les, n_worst=1, feedback_strategy="environment",
        accept_only_if_improved=False)  # wiring test; gate tested separately

    assert len(result.iterations) == 1
    ck = np.asarray(result.final_config.C_K)
    assert ck.shape == (8 * 16,)             # per-column (env-generalized) field
    assert np.all(np.isfinite(ck))


def test_build_multi_correction_campaign_corrects_both_coefficients():
    """build_multi_correction_campaign co-corrects C_K AND Pr_t from ONE LES run
    per column: it sets diagnosis_methods, auto-populates l_mix_max, and the multi
    campaign produces in-bounds per-column fields for BOTH coefficients."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_multi_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_K", "Pr_t"), accept_only_if_improved=False)

    assert set(result.final_fields) == {"clubb_lite_C_K", "clubb_lite_Pr_t"}
    ck = np.asarray(result.final_config.C_K).reshape(-1)
    prt = np.asarray(result.final_config.Pr_t).reshape(-1)
    assert ck.shape == (8 * 16,) and prt.shape == (8 * 16,)
    assert float(ck.min()) >= 0.1 and float(ck.max()) <= 1.2     # C_K bounds
    assert float(prt.min()) >= 0.3 and float(prt.max()) <= 1.5   # Pr_t bounds


def test_build_correction_campaign_c_eps_single_method():
    """Single-coefficient c_eps is wired: build_correction_campaign selects the
    clubb_lite_C_eps promotion + C_eps background + auto-populates l_mix_max."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME, diagnosis_method="c_eps"),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        accept_only_if_improved=False)
    ceps = np.asarray(result.final_config.C_eps).reshape(-1)
    assert ceps.shape == (8 * 16,)
    assert float(ceps.min()) >= 0.06 and float(ceps.max()) <= 0.6
    assert jnp.ndim(jnp.asarray(result.final_config.C_K)) == 0   # C_K untouched


def test_build_multi_correction_campaign_three_coefficients_with_c_eps():
    """C_K + Pr_t + C_eps co-corrected from one LES run; C_eps closes the wp2-
    identification gap. All three end up in-bounds per-column fields."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_multi_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_K", "Pr_t", "C_eps"), accept_only_if_improved=False)

    assert set(result.final_fields) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    ceps = np.asarray(result.final_config.C_eps).reshape(-1)
    assert ceps.shape == (8 * 16,)
    assert float(ceps.min()) >= 0.06 and float(ceps.max()) <= 0.6   # C_eps bounds


def test_build_multi_correction_campaign_sequential_staged():
    """sequential=True routes the staged (block-coordinate-descent) mode through
    build_multi_correction_campaign; per-coefficient fractions are reported."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    bias = np.zeros((8, 16))
    bias[4, 8] = 6.0
    reference = model_state._replace(T=model_state.T - jnp.asarray(bias)[:, :, None])

    result = build_multi_correction_campaign(
        base_atm_config=_base_config(),
        build_base_driver=lambda cfg: _FakeDriver(model_state),
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        reference=reference, sigma=sigma, grid=grid,
        area_weights=jnp.ones((8, 16)), n_iterations=1,
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1,
        coefficients=("C_eps", "C_K", "Pr_t"), sequential=True,
        accept_only_if_improved=False)
    assert set(result.final_fields) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    assert result.iterations[0].step_fractions_by_key is not None


def test_build_multi_correction_campaign_rejects_unknown_coefficient():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    model_state = _full_grid_state()
    with pytest.raises(ValueError, match="unknown coefficient"):
        build_multi_correction_campaign(
            base_atm_config=_base_config(),
            build_base_driver=lambda cfg: _FakeDriver(model_state),
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            reference=model_state, sigma=sigma, grid=grid,
            area_weights=jnp.ones((8, 16)), n_iterations=1,
            les_config=ColumnLESConfig(regime=_SMALL_REGIME),
            run_les_fn=_mock_run_les_sheared, n_worst=1,
            coefficients=("C_K", "bogus"))


# --- Distributed multi-coefficient campaign wrapper (iter 95) ----------------
def _mock_layout(owned, local_cells, n_global=None):
    lc = np.asarray(local_cells)
    if n_global is None:                              # default: tightest mesh spanning lc
        n_global = int(lc.max()) + 1 if lc.size else 0
    return SimpleNamespace(
        owned_mask_cells=jnp.asarray(owned),
        partition=SimpleNamespace(local_cells=lc, nCells_global=n_global))


def _global_cell_state(ncells, nlev=4):
    rng = np.arange(ncells * nlev, dtype=float).reshape(ncells, nlev)
    return ColumnState(
        T=jnp.asarray(rng), q_v=jnp.zeros((ncells, nlev)), u=jnp.zeros((ncells, nlev)),
        v=jnp.zeros((ncells, nlev)), p_s=jnp.arange(float(ncells)),
        sst_K=jnp.arange(float(ncells)) + 290.0)


def test_distributed_campaign_kwargs_composes_hooks_and_slices():
    """``_distributed_campaign_kwargs`` (the shared composition for BOTH the single +
    multi distributed wrappers) builds the three hooks from the layout AND slices the
    GLOBAL reference + area_weights to the rank's local cells — order-preserving."""
    from functools import partial

    from legoesm.parallel.reductions import global_sum_mpi
    from legoesm.training.distributed_manifest import gather_global_worst_columns

    # 4 global cells; this rank's local_cells = [2, 3, 1] (owned 2,3; halo 1).
    layout = _mock_layout([True, True, False], [2, 3, 1])
    ref = _global_cell_state(4)
    area = jnp.arange(4.0) + 1.0
    kw = _distributed_campaign_kwargs(layout, ref, area, n_worst=1, base_valid_mask=None)
    np.testing.assert_array_equal(np.asarray(kw["valid_mask"]), [True, True, False])
    assert kw["global_reduce"] is global_sum_mpi
    assert isinstance(kw["manifest_reducer"], partial)
    assert kw["manifest_reducer"].func is gather_global_worst_columns
    assert kw["n_worst"] == 1
    # reference + area sliced to local_cells [2,3,1] (order preserved).
    np.testing.assert_array_equal(
        np.asarray(kw["reference"].T), np.asarray(ref.T)[[2, 3, 1]])
    np.testing.assert_array_equal(np.asarray(kw["area_weights"]), [3.0, 4.0, 2.0])


def test_distributed_campaign_kwargs_rejects_mesh_mismatch():
    """A GLOBAL area_weights / reference whose cell-count ≠ the partitioned mesh is
    REJECTED LOUDLY — the JAX gather would otherwise silently clamp out-of-range cell
    ids (too-short) or mis-align (wrong mesh), corrupting the rank's compare/weights
    on a multi-day run (iter 97; EXACT-N vs nCells_global also catches too-LONG)."""
    layout = _mock_layout([True, True, False], [2, 3, 4], n_global=5)
    ref5 = _global_cell_state(5)
    # area_weights shorter than the global mesh (5).
    with pytest.raises(ValueError, match="area_weights has 4 cells.*global mesh has 5"):
        _distributed_campaign_kwargs(
            layout, ref5, jnp.ones(4), n_worst=1, base_valid_mask=None)
    # area_weights LONGER than the global mesh — exact check (a bounds check misses this).
    with pytest.raises(ValueError, match="area_weights has 6 cells.*global mesh has 5"):
        _distributed_campaign_kwargs(
            layout, ref5, jnp.ones(6), n_worst=1, base_valid_mask=None)
    # a non-1-D area_weights is rejected (must be per-cell).
    with pytest.raises(ValueError, match="must be 1-D"):
        _distributed_campaign_kwargs(
            layout, ref5, jnp.ones((5, 2)), n_worst=1, base_valid_mask=None)
    # reference cell-count ≠ the mesh (here a reference for 6 cells, mesh has 5).
    with pytest.raises(ValueError, match="reference has 6 cells.*global mesh has 5"):
        _distributed_campaign_kwargs(
            layout, _global_cell_state(6), jnp.ones(5), n_worst=1, base_valid_mask=None)


def test_build_distributed_multi_forwards_composed_kwargs(monkeypatch):
    """``build_distributed_multi_correction_campaign`` forwards the composed
    distributed kwargs (the three hooks + the rank-local reference/area slice) AND the
    multi-coefficient ``campaign_kwargs`` (e.g. ``coefficients``) to
    ``build_multi_correction_campaign`` — the multi sibling of the iter-89 single
    wrapper, sharing the SAME composition (no MPI: only the wiring is exercised)."""
    captured = {}

    def _fake_multi(**kwargs):
        captured.update(kwargs)
        return "MULTI_RESULT"

    monkeypatch.setattr(
        "scripts.run.run_correction_campaign.build_multi_correction_campaign",
        _fake_multi)
    layout = _mock_layout([True, True, False], [2, 3, 1])
    ref = _global_cell_state(4)
    out = build_distributed_multi_correction_campaign(
        layout=layout, reference=ref, area_weights=jnp.arange(4.0) + 1.0, n_worst=1,
        base_atm_config="CFG", coefficients=("C_K", "Pr_t", "C_eps"))
    assert out == "MULTI_RESULT"
    # the distributed hooks + rank-local slice reached build_multi...
    np.testing.assert_array_equal(np.asarray(captured["valid_mask"]), [True, True, False])
    assert captured["manifest_reducer"] is not None and captured["global_reduce"] is not None
    np.testing.assert_array_equal(
        np.asarray(captured["reference"].T), np.asarray(ref.T)[[2, 3, 1]])
    # ...and the multi-only kwargs pass through.
    assert captured["coefficients"] == ("C_K", "Pr_t", "C_eps")
    assert captured["base_atm_config"] == "CFG"


def test_build_distributed_multi_rejects_duplicate_hook_kwarg():
    """A caller cannot set the distributed hooks inconsistently — passing one in
    campaign_kwargs collides with the supplied one (TypeError)."""
    layout = _mock_layout([True, True], [0, 1])
    ref = _global_cell_state(2)
    with pytest.raises(TypeError):
        build_distributed_multi_correction_campaign(
            layout=layout, reference=ref, area_weights=jnp.ones(2), n_worst=1,
            valid_mask=jnp.ones(2, dtype=bool))   # duplicate of the supplied hook


@pytest.mark.parametrize(
    "multi, target",
    [(False, "build_distributed_correction_campaign"),
     (True, "build_distributed_multi_correction_campaign")])
def test_build_distributed_mpas_campaign_wires_local_mesh(monkeypatch, multi, target):
    """``build_distributed_mpas_campaign`` (iter 96, the one-call RUNNABLE entry point)
    partitions the GLOBAL mesh and forwards the rank-LOCAL mesh as BOTH the campaign
    ``grid`` and the bound ``build_base_driver`` — and dispatches to the single vs
    multi distributed wrapper on ``multi`` (no MPI: the layout build + both wrappers
    are monkeypatched, only the wiring is exercised)."""
    import scripts.run.run_correction_campaign as rcc

    layout = SimpleNamespace(local_mesh="LOCAL_MESH")
    seen_global = {}

    def _fake_make_layout(global_mesh, rank, n_ranks):
        seen_global.update(global_mesh=global_mesh, rank=rank, n_ranks=n_ranks)
        return layout

    monkeypatch.setattr(
        "legoesm.parallel.voronoi_mpi.make_voronoi_partition_layout", _fake_make_layout)

    captured = {}

    def _fake_wrapper(**kwargs):
        captured.update(kwargs)
        return "WRAPPED"

    # patch BOTH wrappers; only ``target`` should actually be called.
    for name in ("build_distributed_correction_campaign",
                 "build_distributed_multi_correction_campaign"):
        monkeypatch.setattr(rcc, name,
                            _fake_wrapper if name == target else _boom_wrapper)

    driver_calls = []

    def _build_local_driver(cfg, local_mesh):
        driver_calls.append((cfg, local_mesh))
        return "DRIVER"

    out = rcc.build_distributed_mpas_campaign(
        global_mesh="GMESH", rank=0, n_ranks=1, reference="REF", area_weights="AREA",
        n_worst=2, build_local_driver=_build_local_driver, multi=multi,
        validate_partition=False,                    # mock layout + no MPI collective
        base_atm_config="CFG")

    assert out == "WRAPPED"
    assert seen_global == dict(global_mesh="GMESH", rank=0, n_ranks=1)
    # the rank-LOCAL mesh (not the global one) is wired as the grid + into the driver.
    assert captured["grid"] == "LOCAL_MESH"
    assert captured["layout"] is layout
    assert captured["reference"] == "REF" and captured["n_worst"] == 2
    assert captured["base_atm_config"] == "CFG"        # campaign_kwargs pass through
    # the bound build_base_driver binds local_mesh and forwards the user's builder.
    assert captured["build_base_driver"]("the_cfg") == "DRIVER"
    assert driver_calls == [("the_cfg", "LOCAL_MESH")]


def _boom_wrapper(**kwargs):                            # the wrapper that must NOT run
    raise AssertionError("wrong distributed wrapper dispatched")
