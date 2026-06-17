"""Real coupled-run (CMIP-mode) → compare-to-reanalysis integration smoke.

Mock-free validation of the AMIP/CMIP → compare path that the unit tests stub:
runs a tiny REAL coupled ESM (slab-ocean aquaplanet = interactive SST = CMIP
mode) for a day, then feeds its actual atmosphere state through the run-mode-
agnostic bridge (`column_state_from_hydrostatic`) into the real
`compare_state_to_reference` (scoring + worst-column manifest).  This is the
companion to the real-dycore column-LES integration (the LES half) — together
they exercise both heavy ends of the pipeline against real model physics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Requires the workspace editable install (`pip install -e ".[dev]"`).
jax.config.update("jax_enable_x64", True)

from legoesm.training.compare_reanalysis import (
    column_state_from_hydrostatic,
    compare_state_to_reference,
)


def _run_tiny_coupled(days=1, resolution=8, nlev=5, dt=600.0):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver

    atm_config = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=dt, model_type="hydrostatic"),
        output=OutputConfig(diag_days=max(days, 1)),
        radiation="gray", days=days,
    )
    driver = CoupledESMDriver(atm_config, PRESETS["aquaplanet"]())
    driver.setup()
    driver.run()
    return driver


@pytest.mark.slow
def test_real_coupled_run_feeds_compare():
    driver = _run_tiny_coupled(days=1)

    # The coupled run stayed finite (sanity), and exposes a real atmosphere
    # state + moisture + an INTERACTIVE (CMIP) SST from the slab ocean.
    atm = driver._atm  # the wrapped ModelDriver (atmosphere)
    assert bool(jnp.all(jnp.isfinite(atm.state.T.data)))
    q_v = atm.q_v
    assert q_v is not None
    sst_K = driver._ocean_state.T_sfc.data  # coupled-ocean SST (CMIP source)

    # Run-mode-agnostic bridge: driver HydrostaticState (+ separate q_v, coupled
    # SST) -> ColumnState (unwraps the Field leaves).
    model = column_state_from_hydrostatic(atm.state, q_v, sst_K=sst_K)
    assert model.T.shape == atm.state.T.data.shape

    # Synthetic ERA5 reference = the model with a localized cold bias so the
    # worst-column ranking has something to flag.
    bias = np.zeros(model.T.shape)
    bias_flat = bias.reshape(-1)
    bias_flat[: max(1, bias_flat.size // 50)] = 8.0  # +8 K in ~2% of cells
    reference = model._replace(T=model.T - jnp.asarray(bias))

    sigma = atm.sigma
    rad2deg = 180.0 / jnp.pi
    lat_deg = jnp.asarray(atm.grid.grid_lat) * rad2deg
    lon_deg = jnp.asarray(atm.grid.grid_lon) * rad2deg

    comparison = compare_state_to_reference(
        model=model, reference=reference,
        sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=lat_deg, lon_deg=lon_deg, time_index=0, n_worst=5,
    )

    # The real comparison produced finite per-column scores and a worst-column
    # manifest (the LES would spin off from these).
    assert bool(jnp.all(jnp.isfinite(comparison.error_fields.combined_score)))
    assert float(jnp.max(comparison.error_fields.combined_score)) > 0.0
    assert 1 <= len(comparison.manifest) <= 5
    rec = comparison.manifest[0]
    # The worst column carries finite environment tags (SST/CAPE/shear) for the
    # LES regime selection.
    assert np.isfinite(rec.environment.sst_K)
    assert np.isfinite(rec.environment.cape_J_kg)
    # The worst column's T RMSE is the imposed bias (the biased cells rank top).
    assert rec.T_rmse_K == pytest.approx(8.0, abs=1e-6)
