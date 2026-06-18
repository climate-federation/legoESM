"""Smoke test for the perfect-model OSSE CLI wiring (scripts/validate).

``build_perfect_model_osse`` is exercised end-to-end with a mock C_K-sensitive
driver + a mock plane-LES: it must generate the pseudo-truth from ``true_clubb``,
run the real run->time-mean->compare->real-process_column->campaign chain from
``biased_clubb``, and return an :class:`OSSEResult` whose bias obeys the monotonic
gate.  The RECOVERY logic itself is unit-tested with controllable diagnoses in
``tests/unit/test_perfect_model_osse.py``; here the LES diagnosis is the real
(mock-fed) path, so only the wiring + gate invariant are asserted.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.column_les import ColumnLESConfig  # noqa: E402
from legoesm.atmosphere.dynamics.les_regime import (  # noqa: E402
    LESRegimeConfig,
    LESResolutionConfig,
)
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig  # noqa: E402
from legoesm.training.compare_reanalysis import ColumnState  # noqa: E402
from legoesm.training.perfect_model_osse import osse_verdict  # noqa: E402

from scripts.validate.run_perfect_model_osse import (  # noqa: E402
    build_multi_perfect_model_osse,
    build_perfect_model_osse,
)

_SMALL_RES = LESResolutionConfig(
    dx_m=50.0, nx=8, ny=8, nlev=8, domain_top_m=2000.0, dz_sfc_m=50.0)
_SMALL_REGIME = LESRegimeConfig(shallow=_SMALL_RES, deep=_SMALL_RES)


class _FakeDriver:
    def __init__(self, state):
        self.state = state

    def run(self, segment_callback, **kwargs):  # noqa: ARG002
        segment_callback(self, 0.0, 1.0)
        return "OK"


def _full_grid_state(temp, nlat=8, nlon=16, nlev=5):
    shp, sfc = (nlat, nlon, nlev), (nlat, nlon)
    return ColumnState(
        T=temp, q_v=jnp.full(shp, 5e-3), u=jnp.full(shp, 5.0), v=jnp.zeros(shp),
        p_s=jnp.full(sfc, 1.0e5), sst_K=jnp.full(sfc, 290.0))


def _ck_sensitive_base(cfg):
    """A driver whose column temperature depends on the injected C_K, so the true
    and biased configs produce DIFFERENT states (a real bias to correct)."""
    ck = jnp.asarray(cfg.turbulence_override.clubb_lite.C_K)
    nlat, nlon, nlev = 8, 16, 5
    ck2d = (jnp.full((nlat, nlon), float(ck)) if ck.ndim == 0
            else ck.reshape(nlat, nlon))
    temp = 280.0 + 10.0 * ck2d[:, :, None] * jnp.ones((nlat, nlon, nlev))
    return _FakeDriver(_full_grid_state(temp))


def _mock_run_les_sheared(setup):
    from legoesm.atmosphere.dynamics.compressible_euler_plane import make_rest_state

    grid, hc = setup.grid, setup.height_coord
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    ny, nx, nlev = grid.ny, grid.nx, hc.n_levels
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    s = jnp.asarray(np.where((ii + jj) % 2 == 0, 1.0, -1.0))
    w = 2.0 * s[:, :, None] * jnp.ones((ny, nx, nlev + 1))
    thp = 0.5 * s[:, :, None] * jnp.ones((ny, nx, nlev))
    tr = jnp.zeros((ny, nx, nlev, 3)).at[..., 0].set(0.01)
    z = jnp.asarray(hc.z_full)
    u = (0.01 * z)[None, None, :] * jnp.ones((ny, nx, z.shape[0]))
    return state._replace(
        w=state.w.replace(data=w), theta_prime=state.theta_prime.replace(data=thp),
        tracers=state.tracers.replace(data=tr), u=state.u.replace(data=u))


def _base_config():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=5),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray", turbulence="clubb_lite")


def test_build_perfect_model_osse_wiring():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)

    res = build_perfect_model_osse(
        base_atm_config=_base_config(),
        build_base_driver=_ck_sensitive_base,
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        sigma=sigma, grid=grid, area_weights=jnp.ones((8, 16)),
        true_clubb=CLUBBLiteConfig(C_K=0.9),
        biased_clubb=CLUBBLiteConfig(C_K=0.4),
        les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                   diagnosis_method="clubb_coefficient"),
        run_les_fn=_mock_run_les_sheared, n_worst=1, n_iterations=1)

    # Pseudo-truth generated from true_clubb; biased start recorded.
    assert res.true_value == 0.9
    assert res.initial_value == 0.4
    # A real bias existed (true and biased states differ by construction).
    assert res.initial_bias > 0.0
    # Monotonic gate invariant: the OSSE never reports a worsened bias.
    assert res.final_bias <= res.initial_bias + 1e-9
    assert res.n_rounds == 1
    # The full chain produced finite recovery metrics (real numbers, not NaN).
    assert np.isfinite(res.recovered_value)
    assert np.isfinite(res.final_param_error)
    # The verdict is one of the gated outcomes (the recovered/no_change outcomes
    # are asserted non-vacuously with controllable diagnoses in the unit test).
    assert osse_verdict(res).status in {"recovered", "bias_only", "no_change"}


def test_build_multi_perfect_model_osse_wiring():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)

    res = build_multi_perfect_model_osse(
        base_atm_config=_base_config(),
        build_base_driver=_ck_sensitive_base,
        extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
        sigma=sigma, grid=grid, area_weights=jnp.ones((8, 16)),
        true_clubb=CLUBBLiteConfig(C_K=0.9, Pr_t=0.5, C_eps=0.3),
        biased_clubb=CLUBBLiteConfig(C_K=0.4, Pr_t=0.33, C_eps=0.1),
        les_config=ColumnLESConfig(regime=_SMALL_REGIME),
        run_les_fn=_mock_run_les_sheared, n_worst=1, n_iterations=1,
        coefficients=("C_K", "Pr_t", "C_eps"))

    # All three coefficients are tracked; the monotonic gate invariant holds.
    assert set(res.per_coefficient) == {
        "clubb_lite_C_K", "clubb_lite_Pr_t", "clubb_lite_C_eps"}
    assert res.initial_bias > 0.0
    assert res.final_bias <= res.initial_bias + 1e-9
    assert all(np.isfinite(c.final_param_error)
               for c in res.per_coefficient.values())


def test_build_multi_perfect_model_osse_rejects_unknown_coefficient():
    import pytest
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    with pytest.raises(ValueError, match="unknown coefficient"):
        build_multi_perfect_model_osse(
            base_atm_config=_base_config(),
            build_base_driver=_ck_sensitive_base,
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            sigma=sigma, grid=grid, area_weights=jnp.ones((8, 16)),
            true_clubb=CLUBBLiteConfig(C_K=0.9),
            biased_clubb=CLUBBLiteConfig(C_K=0.4),
            les_config=ColumnLESConfig(regime=_SMALL_REGIME),
            run_les_fn=_mock_run_les_sheared, n_worst=1, n_iterations=1,
            coefficients=("C_K", "bogus"))


def test_build_perfect_model_osse_rejects_unknown_method():
    import pytest
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_latlon_grid(8, 16, dtype=jnp.float64)
    sigma = create_sigma_coordinate(5)
    with pytest.raises(ValueError, match="unknown diagnosis_method"):
        build_perfect_model_osse(
            base_atm_config=_base_config(),
            build_base_driver=_ck_sensitive_base,
            extract_column_state=lambda d, day, dt: d.state,  # noqa: ARG005
            sigma=sigma, grid=grid, area_weights=jnp.ones((8, 16)),
            true_clubb=CLUBBLiteConfig(C_K=0.9),
            biased_clubb=CLUBBLiteConfig(C_K=0.4),
            les_config=ColumnLESConfig(regime=_SMALL_REGIME,
                                       diagnosis_method="bogus_method"),
            run_les_fn=_mock_run_les_sheared, n_worst=1, n_iterations=1)
