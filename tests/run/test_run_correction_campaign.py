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
    build_multi_correction_campaign,
    make_base_driver_builder,
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
    assert np.isfinite(float(it.bias.baseline_bias))
    assert np.isfinite(float(it.bias.updated_bias))


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
