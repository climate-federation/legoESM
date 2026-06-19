"""Full pipeline end-to-end with REAL model code (no mocks on either heavy end).

Combines the two heavy real paths into one end-to-end smoke:
  real lat-lon coupled run (CMIP mode)  →  real compare + worst-column manifest
    →  REAL plane-LES spin-off for the worst column  →  closure-coefficient
    →  feedback field  →  applied to the real CLUBB-lite C_K config.

This validates that the ENTIRE compare-reanalysis pipeline composes with real
model physics (a real coupled ESM + a real compressible-Euler plane LES), not
just mocks.  It does NOT assert a bias reduction — that needs an HPC-scale
multi-day run plus the second corrected run — but it proves the loop runs.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.les_regime import (  # noqa: E402
    LESRegimeConfig,
    LESResolutionConfig,
)
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig  # noqa: E402
from legoesm.training.compare_reanalysis import (  # noqa: E402
    column_state_from_hydrostatic,
    compare_state_to_reference,
)
from legoesm.training.feedback_assembly import assemble_feedback_field  # noqa: E402
from legoesm.training.promotable_params import apply_feedback_to_scheme  # noqa: E402

# Tiny LES box so the spin-off is cheap: 50 m * 8 = 400 m < 2 km top.
_SMALL_RES = LESResolutionConfig(
    dx_m=50.0, nx=8, ny=8, nlev=8, domain_top_m=2000.0, dz_sfc_m=50.0
)
_SMALL_REGIME = LESRegimeConfig(shallow=_SMALL_RES, deep=_SMALL_RES)


def _run_tiny_latlon_coupled(days=1, n_lat=8, nlev=5):
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
        output=OutputConfig(diag_days=max(days, 1)),
        radiation="gray", days=days,
    )
    # Ocean grid == atm grid so the coupled SST aligns with the atm column grid
    # (resolution=n_lat → atm is n_lat × 2·n_lat).
    ocean_grid = create_latlon_grid(n_lat=n_lat, n_lon=2 * n_lat)
    driver = CoupledESMDriver(
        atm_config, PRESETS["aquaplanet"](), ocean_grid=ocean_grid,
    )
    driver.setup()
    driver.run()
    return driver


@pytest.mark.slow
def test_full_pipeline_real_model_and_les():
    from legoesm.atmosphere.dynamics.column_les import (
        ColumnLESConfig,
        process_column,
        run_forced_les,
    )

    driver = _run_tiny_latlon_coupled(days=1)
    atm = driver._atm
    assert bool(jnp.all(jnp.isfinite(atm.state.T.data)))

    # --- real model state -> compare to a synthetic ERA5 reference ---
    sst_K = driver._ocean_state.T_sfc.data
    model = column_state_from_hydrostatic(atm.state, atm.q_v, sst_K=sst_K)
    n_lat, n_lon, nlev = model.T.shape
    # +6 K cold bias in a single column so the worst column is deterministic.
    bias = np.zeros((n_lat, n_lon))
    bias[n_lat // 2, n_lon // 2] = 6.0
    reference = model._replace(T=model.T - jnp.asarray(bias)[:, :, None])

    sigma = atm.sigma
    rad2deg = 180.0 / jnp.pi
    comparison = compare_state_to_reference(
        model=model, reference=reference,
        sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=jnp.asarray(atm.grid.grid_lat) * rad2deg,
        lon_deg=jnp.asarray(atm.grid.grid_lon) * rad2deg,
        time_index=0, n_worst=1,
    )
    assert len(comparison.manifest) == 1
    rec = comparison.manifest[0]
    assert rec.grid_index == (n_lat // 2, n_lon // 2)  # the biased column

    # --- REAL plane-LES spin-off for the worst column -> diagnose K ---
    # gate_les_realism=False: this gate verifies the run→compare→LES→DIAGNOSE→feedback
    # CHAIN, so it runs a deliberately SHORT (2-step, for speed) LES that resolves a valid
    # down-gradient K but is correctly rejected by the realism gate as not-yet-turbulent.
    # The realism gate is exercised separately (its own unit tests + the realistic-LES
    # OSSE/capstone); disabling it here lets the diagnosis MATH be asserted (else the gate
    # masks the otherwise-valid 3/7 diagnosis to all-invalid). The diagnosis below is the
    # raw diagnose_eddy_diffusivity result.
    les_cfg = ColumnLESConfig(regime=_SMALL_REGIME, gate_les_realism=False)
    run_fn = partial(run_forced_les, dt_s=0.5, n_steps=2)
    diagnosis = process_column(
        rec, T=atm.state.T.data, q_v=atm.q_v, u=atm.state.u.data,
        v=atm.state.v.data, p_s=atm.state.p_s.data,
        grid=atm.grid, sigma=sigma, config=les_cfg, run_les_fn=run_fn,
    )
    assert diagnosis.K.shape == (_SMALL_RES.nlev - 1,)
    assert bool(jnp.all(jnp.isfinite(diagnosis.K)))
    # Non-vacuity: the real LES must actually resolve a down-gradient flux at
    # ≥1 interior interface (a strictly positive, VALID K) — otherwise the whole
    # diagnosis→feedback chain below would be an all-zero no-op and this "end-to-
    # end" gate would pass without the LES doing any physics.  Deterministic: the
    # θ' seed is a fixed PRNGKey(0), so the diagnosis is reproducible — empirically
    # 3/7 interior interfaces are valid with Kmax≈0.48 m²/s at this (8³, 2-step)
    # config (not flaky).
    assert bool(jnp.any(diagnosis.valid))
    assert float(jnp.max(jnp.where(diagnosis.valid, diagnosis.K, 0.0))) > 0.0

    # --- feedback field -> applied to the real CLUBB-lite C_K config ---
    background = float(CLUBBLiteConfig().C_K)
    field = assemble_feedback_field(
        [rec], [diagnosis], (n_lat, n_lon),
        method="eddy_diffusivity", background=background,
    )
    assert field.shape == (n_lat, n_lon)
    # expected_ncol makes the splice-length guard load-bearing in this test.
    corrected = apply_feedback_to_scheme(
        CLUBBLiteConfig(), "clubb_lite_C_K", field, expected_ncol=n_lat * n_lon)
    # A genuine per-column C_K reached a real GCM turbulence config — and the
    # diagnosed worst column actually moved off the uniform background.
    assert corrected.C_K.shape == (n_lat * n_lon,)
    assert bool(jnp.all(jnp.isfinite(corrected.C_K)))
    assert bool(jnp.any(corrected.C_K != background))
    assert float(corrected.C_K[int(rec.flat_index)]) != background
