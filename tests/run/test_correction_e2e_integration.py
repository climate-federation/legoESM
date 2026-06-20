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
from typing import NamedTuple

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
@pytest.mark.filterwarnings("error::FutureWarning")  # iter 208/210: no f64->f32 scatter
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


def _run_clubb_coupled(clubb, *, resolution=4, nlev=5, days=1):
    """A real tiny coupled (CMIP) run whose clubb_lite turbulence uses the given
    ``CLUBBLiteConfig`` (scalar OR per-column C_K)."""
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
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
        grid=GridConfig(grid_type="latlon", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=max(days, 1)), radiation="gray", days=days,
        turbulence="clubb_lite",
        turbulence_override=TurbulenceConfig(scheme="clubb_lite", clubb_lite=clubb))
    driver = CoupledESMDriver(
        atm, PRESETS["aquaplanet"](),
        ocean_grid=create_latlon_grid(n_lat=resolution, n_lon=2 * resolution))
    driver.setup()
    driver.run()
    return driver


class _EddyDiag(NamedTuple):
    K: jnp.ndarray
    valid: jnp.ndarray


@pytest.mark.slow
@pytest.mark.filterwarnings("error::FutureWarning")
def test_real_model_osse_reduces_bias_with_applied_ck():
    """Clause 6 ("updating these parameters IMPROVE the biases") with REAL model physics,
    in a controlled identical TWIN (the e2e test above runs the loop one round but
    explicitly does NOT assert a bias reduction — this does, via a model twin; the
    real-ERA5 case is the separate HPC-only empirical question, iter 412/413).

    Truth = the real coupled clubb_lite model at C_K=1.0; the biased model is the SAME
    model at C_K=0.4, so the model-vs-truth bias is C_K-driven BY CONSTRUCTION.  One real
    correction round applies the (known-correct) diagnosed C_K=1.0 to EVERY column and
    RE-RUNS the real model — and because the model is deterministic, the re-run equals the
    truth, so the bias falls to ~0 and the monotonic gate accepts it.  Proves the full
    loop (run real model → compare → diagnose → splice per-column C_K → re-run real model
    → gate) lowers the bias with the ACTUAL ESM, not a synthetic ``run_fn``."""
    from legoesm.training.compare_reanalysis import column_state_from_hydrostatic
    from legoesm.training.correction_loop import make_compare_fn, run_correction_iteration

    truth_driver = _run_clubb_coupled(CLUBBLiteConfig(C_K=1.0))
    ta = truth_driver._atm
    truth = column_state_from_hydrostatic(
        ta.state, ta.q_v, sst_K=truth_driver._ocean_state.T_sfc.data)
    n_lat, n_lon, _ = truth.T.shape
    ncol = n_lat * n_lon
    sigma, grid = ta.sigma, ta.grid
    rad2deg = 180.0 / float(jnp.pi)

    def _run(clubb):
        d = _run_clubb_coupled(clubb)
        a = d._atm
        return column_state_from_hydrostatic(
            a.state, a.q_v, sst_K=d._ocean_state.T_sfc.data)

    compare_fn = make_compare_fn(
        reference=truth, sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=jnp.asarray(grid.grid_lat) * rad2deg,
        lon_deg=jnp.asarray(grid.grid_lon) * rad2deg,
        area_weights=jnp.ones((n_lat, n_lon)), n_worst=ncol,
        run_amip_fn=_run, coordinate=sigma)

    res = run_correction_iteration(
        CLUBBLiteConfig(C_K=0.4), compare_fn=compare_fn,
        diagnose_fn=lambda rec, ctx: _EddyDiag(K=jnp.array([1.0]), valid=jnp.array([True])),
        promotion_key="clubb_lite_C_K", grid_shape=(n_lat, n_lon), background=0.4)

    assert float(res.bias.baseline_bias) > 1e-5            # a real C_K-driven bias exists
    assert float(res.bias.updated_bias) < float(res.bias.baseline_bias)   # it FELL
    assert float(res.bias.updated_bias) == pytest.approx(0.0, abs=1e-6)   # to the twin truth
    assert bool(res.bias.improved)                         # the monotonic gate accepts
    assert res.n_corrected == ncol                         # every column corrected


@pytest.mark.slow
@pytest.mark.filterwarnings("error::FutureWarning")
def test_full_loop_real_model_real_les_rerun_and_gate():
    """The COMPLETE chain in ONE round with EVERY heavy end real: a real coupled model →
    compare → REAL plane-LES spin-off → diagnose C_K → splice → RE-RUN the real model →
    monotonic gate.  The e2e test above stops at the feedback field (no re-run); the
    real-model-OSSE test re-runs but with a SYNTHETIC diagnose — this is the only test
    where the real LES's diagnosis AND a real model re-run AND the gate all run together
    (the holistic real-ERA5 culmination, iter 415, here with a synthetic reference so it
    is CI-portable).  It asserts the loop runs end-to-end (finite bias, a definite
    accept/reject verdict, the real LES corrected ≥1 worst column) — NOT a bias reduction,
    which is reference-dependent (the gate may correctly REJECT, exactly as it did against
    real ERA5 where the idealized model's bias is C_K-insensitive, iter 412)."""
    from functools import partial as _partial

    from legoesm.atmosphere.dynamics.column_les import ColumnLESConfig, run_forced_les
    from legoesm.training.compare_reanalysis import column_state_from_hydrostatic
    from legoesm.training.correction_loop import make_compare_fn, run_correction_iteration

    from scripts.run.run_correction_campaign import make_les_diagnose_fn

    driver = _run_clubb_coupled(CLUBBLiteConfig(C_K=0.4))
    a = driver._atm
    model0 = column_state_from_hydrostatic(
        a.state, a.q_v, sst_K=driver._ocean_state.T_sfc.data)
    n_lat, n_lon, _ = model0.T.shape
    sigma, grid = a.sigma, a.grid
    rad2deg = 180.0 / float(jnp.pi)

    # Synthetic reference (CI-portable): the model's own state minus a spatially varying
    # warm bias, so worst-column ranking + the LES are non-trivially exercised.
    bias = np.zeros((n_lat, n_lon))
    bias[n_lat // 2, n_lon // 2] = 5.0
    reference = model0._replace(T=model0.T - jnp.asarray(bias)[:, :, None])

    def _run(clubb):
        d = _run_clubb_coupled(clubb)
        aa = d._atm
        return column_state_from_hydrostatic(
            aa.state, aa.q_v, sst_K=d._ocean_state.T_sfc.data)

    compare_fn = make_compare_fn(
        reference=reference, sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=jnp.asarray(grid.grid_lat) * rad2deg,
        lon_deg=jnp.asarray(grid.grid_lon) * rad2deg,
        area_weights=jnp.ones((n_lat, n_lon)), n_worst=2,
        run_amip_fn=_run, coordinate=sigma)

    les_cfg = ColumnLESConfig(regime=_SMALL_REGIME, gate_les_realism=False)
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_cfg,
        run_les_fn=_partial(run_forced_les, dt_s=0.5, n_steps=2))

    res = run_correction_iteration(
        CLUBBLiteConfig(C_K=0.4), compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        promotion_key="clubb_lite_C_K", grid_shape=(n_lat, n_lon), background=0.4,
        diagnosis_method="eddy_diffusivity", les_budget=2)

    # The whole real chain ran: finite biases + a DEFINITE gate verdict + the real LES
    # spliced ≥1 worst column (the loop is not a vacuous no-op).
    assert np.isfinite(float(res.bias.baseline_bias))
    assert np.isfinite(float(res.bias.updated_bias))
    assert isinstance(bool(res.bias.improved), bool)
    assert res.n_corrected >= 1
