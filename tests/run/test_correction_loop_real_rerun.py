"""Capstone: the full correction loop closing with a REAL model re-run.

Every prior real-component test stopped at ``apply_feedback_to_scheme`` (iter 26)
— before the config-injection (iter 35) existed, so the corrected coefficient was
never fed back into a running model.  This composes the WHOLE loop with real
components AND the re-run:

  baseline CLUBBLiteConfig
    → make_run_fn(build_driver) runs a REAL coupled (CMIP) model + time-means it
    → make_compare_fn scores it vs a synthetic ERA5 reference → worst column
    → (mock LES diagnose — the real LES is covered by iter 20/26)
    → assemble_feedback_field → apply_feedback_to_scheme → per-column clubb C_K
    → run_correction_iteration's SECOND compare_fn RE-RUNS the real model with
      that per-column C_K injected via turbulence_override → re-scores → bias.

The new thing proven here: the loop's re-compare genuinely re-runs the real model
with the LES-informed per-column C_K (the iter-35 injection + iter-36 ordering),
so the offline correction loop is closed end-to-end with real model runs.  The
SIGN of the bias change is NOT asserted — that empirical claim needs real ERA5 at
HPC scale — only that the loop closes and the re-run used the corrected config.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.config import (  # noqa: E402
    CLUBBLiteConfig,
    TurbulenceConfig,
)
from legoesm.training.correction_loop import (  # noqa: E402
    make_compare_fn,
    run_correction_iteration,
)
from legoesm.training.run_to_column_mean import (  # noqa: E402
    cmip_column_state,
    make_run_fn,
)

_NLAT, _NLON, _NLEV = 8, 16, 5


class _Eddy:
    """Mock eddy-diffusivity diagnosis (the real LES is tested in iter 20/26)."""

    def __init__(self, k):
        self.K = jnp.array([k, k])
        self.valid = jnp.array([True, True])


def _build_driver(clubb_cfg):
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.grids.latlon import create_latlon_grid

    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=_NLAT, nlev=_NLEV),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=0.5), radiation="gray",
        turbulence="clubb_lite", days=0.5,
        turbulence_override=TurbulenceConfig(
            scheme="clubb_lite", clubb_lite=clubb_cfg),
    )
    driver = CoupledESMDriver(
        atm, PRESETS["aquaplanet"](),
        ocean_grid=create_latlon_grid(n_lat=_NLAT, n_lon=2 * _NLAT))
    driver.setup()
    return driver


@pytest.mark.slow
def test_full_correction_loop_real_rerun_with_injected_ck():
    from legoesm.training.compare_reanalysis import compare_state_to_reference

    # Capture every driver the loop builds so we can prove the RE-RUN's driver
    # actually resolved the injected per-column C_K (not the scalar default).
    built: list = []

    def build_driver(clubb_cfg):
        driver = _build_driver(clubb_cfg)
        built.append((clubb_cfg, driver))
        return driver

    run_fn = make_run_fn(build_driver, cmip_column_state)

    # Grid + sigma for the comparison (probe driver: setup only, no run; built
    # directly so it is NOT recorded in `built`).
    probe = _build_driver(CLUBBLiteConfig())
    sigma = probe._atm.sigma
    grid = probe._atm.grid
    rad2deg = 180.0 / np.pi
    sigma_full = jnp.asarray(sigma.sigma_full)
    sigma_half = jnp.asarray(sigma.sigma_half)
    lat_deg = jnp.asarray(np.asarray(grid.grid_lat) * rad2deg)
    lon_deg = jnp.asarray(np.asarray(grid.grid_lon) * rad2deg)

    # Baseline run → deterministic reference with a localized +6 K bias so the
    # worst column is pinned.
    model0 = run_fn(CLUBBLiteConfig())
    assert model0.T.shape == (_NLAT, _NLON, _NLEV)
    flat = _NLAT // 2 * _NLON + _NLON // 2     # the biased worst column (row-major)
    bias = np.zeros((_NLAT, _NLON))
    bias[_NLAT // 2, _NLON // 2] = 6.0
    reference = model0._replace(T=model0.T - jnp.asarray(bias)[:, :, None])

    # The +6 K bias makes that column UNAMBIGUOUSLY the worst (the loop's baseline
    # run is deterministic ⇒ reproduces model0, so its manifest matches this).
    comparison = compare_state_to_reference(
        model=model0, reference=reference, sigma_full=sigma_full,
        sigma_half=sigma_half, lat_deg=lat_deg, lon_deg=lon_deg,
        time_index=0, n_worst=1)
    assert comparison.manifest[0].flat_index == flat
    score = np.asarray(comparison.error_fields.combined_score).reshape(-1)
    assert score[flat] > 2.0 * float(np.delete(score, flat).max())  # dominance

    compare_fn = make_compare_fn(
        reference=reference, sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=lat_deg, lon_deg=lon_deg, area_weights=jnp.ones((_NLAT, _NLON)),
        n_worst=1, run_amip_fn=run_fn,
    )

    def diagnose_fn(record, model_ctx):
        return _Eddy(0.8)  # a valid diagnosed eddy diffusivity

    background = float(CLUBBLiteConfig().C_K)
    result = run_correction_iteration(
        CLUBBLiteConfig(),
        compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        promotion_key="clubb_lite_C_K", grid_shape=(_NLAT, _NLON),
        background=background,
    )

    # The loop closed: 1 worst column diagnosed, the updated config carries a
    # PER-COLUMN C_K (the feedback reached the scheme config).
    assert result.n_diagnosed == 1
    assert result.n_corrected == 1
    ck = np.asarray(result.updated_config.C_K)
    assert ck.shape == (_NLAT * _NLON,)
    assert ck[flat] == pytest.approx(0.8)       # got the diagnosed coefficient
    assert np.allclose(np.delete(ck, flat), background)  # others keep background

    # NON-VACUITY: the loop's SECOND compare_fn(updated_config) built a driver
    # whose resolved turbulence kernel config carries EXACTLY the per-column C_K
    # array — i.e. the re-run genuinely consumed the LES-informed injection (had
    # turbulence_override been ignored, this would be the scalar default 0.4).
    updated_cfg, updated_driver = built[-1]
    assert np.ndim(np.asarray(updated_cfg.C_K)) == 1
    np.testing.assert_array_equal(
        np.asarray(updated_driver._atm.physics.turbulence_config.C_K),
        np.asarray(result.updated_config.C_K))

    # The re-run produced a finite, real bias measurement.  Sign NOT asserted
    # (that empirical claim needs real ERA5 at HPC scale).
    assert np.isfinite(float(result.bias.baseline_bias))
    assert np.isfinite(float(result.bias.updated_bias))
    assert np.isfinite(float(result.worst_column_change))
