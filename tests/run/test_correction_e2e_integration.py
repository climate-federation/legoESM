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

from legoesm.atmosphere.dynamics.les.les_regime import (  # noqa: E402
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

from tests._offline_era5_rda import write_forcing_archive  # noqa: E402

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
    from legoesm.atmosphere.dynamics.les.column_les import (
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


def _run_clubb_coupled(clubb, *, resolution=4, nlev=5, days=1, grid_type="latlon"):
    """A real tiny coupled (CMIP) run whose clubb_lite turbulence uses the given
    ``CLUBBLiteConfig`` (scalar OR per-column C_K).  ``grid_type`` (default latlon) also
    accepts ``cubed_sphere`` — then ``ocean_grid=None`` so the coupled driver uses its OWN
    atm grid (same-grid identity coupling, the iter-493 pattern)."""
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
        grid=GridConfig(grid_type=grid_type, resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=max(days, 1)), radiation="gray", days=days,
        turbulence="clubb_lite",
        turbulence_override=TurbulenceConfig(scheme="clubb_lite", clubb_lite=clubb))
    ocean_grid = (create_latlon_grid(n_lat=resolution, n_lon=2 * resolution)
                  if grid_type == "latlon" else None)
    driver = CoupledESMDriver(atm, PRESETS["aquaplanet"](), ocean_grid=ocean_grid)
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
@pytest.mark.parametrize("surface_flux", [False, True])
def test_full_loop_real_model_real_les_rerun_and_gate(surface_flux):
    """The COMPLETE chain in ONE round with EVERY heavy end real: a real coupled model →
    compare → REAL plane-LES spin-off → diagnose C_K → splice → RE-RUN the real model →
    monotonic gate.  The e2e test above stops at the feedback field (no re-run); the
    real-model-OSSE test re-runs but with a SYNTHETIC diagnose — this is the only test
    where the real LES's diagnosis AND a real model re-run AND the gate all run together
    (the holistic real-ERA5 culmination, iter 415, here with a synthetic reference so it
    is CI-portable).  It asserts the loop runs end-to-end (finite bias, a definite
    accept/reject verdict, the real LES corrected ≥1 worst column) — NOT a bias reduction,
    which is reference-dependent (the gate may correctly REJECT, exactly as it did against
    real ERA5 where the idealized model's bias is C_K-insensitive, iter 412).

    Parametrized over ``surface_flux`` (iter 473): the True case is the FIRST end-to-end run
    of the REALISTIC diagnosis path — the real model's coupled SST → the GCM bulk surface
    flux → the prescribed-flux surface BC on the real LES → C_K — the exact diagnosis the
    recommended realistic empirical run uses (--surface-flux). The model state carries sst_K
    (from the coupled ocean), so the shared make_les_diagnose_fn threads it."""
    from functools import partial as _partial

    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig, run_forced_les
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

    les_cfg = ColumnLESConfig(regime=_SMALL_REGIME, gate_les_realism=False,
                              surface_flux=surface_flux)
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


@pytest.mark.slow
@pytest.mark.filterwarnings("error::FutureWarning")
def test_full_loop_ocean_only_excludes_masked_columns():
    """REGRESSION (iter 487): the FULL EXECUTION loop with a grid-shaped ocean valid_mask —
    the realistic ``--ocean-only`` path through run → compare → RANK → LES → diagnose →
    re-run → GATE. The iter-484 bug was a mask-SHAPE crash at HARNESS CONSTRUCTION; the
    smoke (484/485/486) exercises the dry-run preamble — this is the first test of the mask
    through EXECUTION (it flows to BOTH the ranking AND the gate, which reads
    ``baseline.valid_mask``, iter 451). Non-vacuous: the LARGEST bias is in a MASKED ('land')
    column, so ocean-only MUST exclude it from the worst-column manifest (without the mask it
    would be the #1 worst)."""
    from functools import partial as _partial

    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig, run_forced_les
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

    # The LARGEST bias is at the MASKED ('land') peak; a smaller bias at an ocean column.
    r0, c0 = n_lat // 2, n_lon // 2          # masked land peak (T bias 5 K)
    r1, c1 = 0, 0                            # ocean column (T bias 3 K)
    bias = np.zeros((n_lat, n_lon))
    bias[r0, c0] = 5.0
    bias[r1, c1] = 3.0
    reference = model0._replace(T=model0.T - jnp.asarray(bias)[:, :, None])
    # Grid-shaped ocean mask (the iter-484 fix shape), excluding the land peak.
    ocean_mask = jnp.ones((n_lat, n_lon), bool).at[r0, c0].set(False)

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
        valid_mask=ocean_mask, run_amip_fn=_run, coordinate=sigma)

    # RANKING: the masked land peak (LARGEST bias) is EXCLUDED; the ocean bias IS ranked.
    cr = compare_fn(CLUBBLiteConfig(C_K=0.4))
    worst_flat = {int(rec.flat_index) for rec in cr.manifest}
    assert (r0 * n_lon + c0) not in worst_flat              # ocean-only excluded the land peak
    assert (r1 * n_lon + c1) in worst_flat                  # the ocean column IS a worst column

    les_cfg = ColumnLESConfig(regime=_SMALL_REGIME, gate_les_realism=False)
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_cfg,
        run_les_fn=_partial(run_forced_les, dt_s=0.5, n_steps=2))

    # FULL LOOP: the grid-shaped mask flows through compare → rank → LES → re-run → gate
    # (the gate reads baseline.valid_mask) with NO shape crash — the iter-484 EXECUTION analog.
    res = run_correction_iteration(
        CLUBBLiteConfig(C_K=0.4), compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        promotion_key="clubb_lite_C_K", grid_shape=(n_lat, n_lon), background=0.4,
        diagnosis_method="eddy_diffusivity", les_budget=2)
    assert np.isfinite(float(res.bias.baseline_bias))
    assert np.isfinite(float(res.bias.updated_bias))
    assert isinstance(bool(res.bias.improved), bool)
    assert res.n_corrected >= 1
    # The masked land peak is NEVER corrected — it keeps the background coefficient.
    assert float(np.asarray(res.feedback_field).reshape(n_lat, n_lon)[r0, c0]) == \
        pytest.approx(0.4)


@pytest.mark.slow
@pytest.mark.filterwarnings("error::FutureWarning")
def test_full_loop_ocean_only_on_cubed_sphere():
    """REGRESSION (iter 496): the ocean-only EXECUTION on a CUBED-SPHERE grid — the LAST
    untested realistic combination. iter 487 tested ocean-only execution on LATLON; iter 495
    locked the cubed-sphere model RUN alone; this runs the FULL loop (run → compare → RANK →
    LES → re-run → gate) with a 3-D (6,n,n) grid-shaped ocean mask. The iter-484 mask-shape fix
    reshapes to the grid shape, which for cubed-sphere is (6,n,n) — a DIFFERENT shape than
    lat-lon's (nlat,nlon); the ranking flattens it (row-major) + the gate broadcasts it. The LES
    spin-off uses the cubed-sphere extractor. Non-vacuous: the LARGEST bias is in a MASKED
    column, so ocean-only must exclude it from the worst columns."""
    from functools import partial as _partial

    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig, run_forced_les
    from legoesm.training.compare_reanalysis import column_state_from_hydrostatic
    from legoesm.training.correction_loop import make_compare_fn, run_correction_iteration

    from scripts.run.run_correction_campaign import make_les_diagnose_fn

    driver = _run_clubb_coupled(CLUBBLiteConfig(C_K=0.4), grid_type="cubed_sphere")
    a = driver._atm
    model0 = column_state_from_hydrostatic(
        a.state, a.q_v, sst_K=driver._ocean_state.T_sfc.data)
    horiz = tuple(int(d) for d in model0.T.shape[:-1])    # (6, n, n)
    assert len(horiz) == 3 and horiz[0] == 6              # genuinely a cubed-sphere column grid
    sigma, grid = a.sigma, a.grid
    rad2deg = 180.0 / float(jnp.pi)

    # The LARGEST bias at a MASKED ('land') column; a smaller bias at an ocean column.
    masked_idx, ocean_idx = (0, 0, 0), (1, 0, 0)
    bias = np.zeros(horiz)
    bias[masked_idx] = 5.0
    bias[ocean_idx] = 3.0
    reference = model0._replace(T=model0.T - jnp.asarray(bias)[..., None])
    ocean_mask = jnp.ones(horiz, bool).at[masked_idx].set(False)   # 3-D (6,n,n) grid-shaped

    def _run(clubb):
        d = _run_clubb_coupled(clubb, grid_type="cubed_sphere")
        aa = d._atm
        return column_state_from_hydrostatic(
            aa.state, aa.q_v, sst_K=d._ocean_state.T_sfc.data)

    compare_fn = make_compare_fn(
        reference=reference, sigma_full=jnp.asarray(sigma.sigma_full),
        sigma_half=jnp.asarray(sigma.sigma_half),
        lat_deg=jnp.asarray(grid.grid_lat) * rad2deg,
        lon_deg=jnp.asarray(grid.grid_lon) * rad2deg,
        area_weights=jnp.ones(horiz), n_worst=2,
        valid_mask=ocean_mask, run_amip_fn=_run, coordinate=sigma)

    # RANKING on (6,n,n): the masked land peak (LARGEST bias) is EXCLUDED; the ocean bias ranked.
    cr = compare_fn(CLUBBLiteConfig(C_K=0.4))
    worst_flat = {int(rec.flat_index) for rec in cr.manifest}
    assert int(np.ravel_multi_index(masked_idx, horiz)) not in worst_flat
    assert int(np.ravel_multi_index(ocean_idx, horiz)) in worst_flat

    les_cfg = ColumnLESConfig(regime=_SMALL_REGIME, gate_les_realism=False)
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_cfg,
        run_les_fn=_partial(run_forced_les, dt_s=0.5, n_steps=2))

    # FULL LOOP: the 3-D mask flows through compare → rank → LES → re-run → gate with no crash.
    res = run_correction_iteration(
        CLUBBLiteConfig(C_K=0.4), compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        promotion_key="clubb_lite_C_K", grid_shape=horiz, background=0.4,
        diagnosis_method="eddy_diffusivity", les_budget=2)
    assert np.isfinite(float(res.bias.baseline_bias))
    assert np.isfinite(float(res.bias.updated_bias))
    assert res.n_corrected >= 1
    # The masked land peak keeps the background coefficient (never corrected).
    assert float(np.asarray(res.feedback_field).reshape(horiz)[masked_idx]) == \
        pytest.approx(0.4)


@pytest.mark.slow
@pytest.mark.filterwarnings("error::FutureWarning")
def test_full_loop_environment_strategy_builds_a_deployable_kernel():
    """REGRESSION (iter 488): the env-kernel feedback strategy (--feedback-strategy
    environment, the cross-grid deploy's UPSTREAM) through a REAL model loop. The kernel BUILD
    is unit-tested from synthetic records (test_feedback_assembly) and the cross-resolution
    OSSE uses SYNTHETIC env functions — but the campaign building the kernel from a REAL run's
    worst-column env tags (with the REAL prescribed/coupled SST, the kernel's DOMINANT
    predictor, threaded by amip/cmip_column_state) was untested end-to-end. Uses the SAME
    env_grid_fn the campaign builds (maybe_env_grid_fn → column_environment_grid with the
    coordinate's hybrid-correct default pressures, consistent with the deploy). Asserts the
    loop produces a NON-None EnvKernel that round-trips (the on-disk env_kernel.json contract)
    AND DEPLOYS — apply_env_kernel_override yields a finite per-column override on the model's
    own environment (the cross-grid deploy's core operation)."""
    from functools import partial as _partial

    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig, run_forced_les
    from legoesm.training.compare_reanalysis import column_state_from_hydrostatic
    from legoesm.training.correction_loop import make_compare_fn, run_correction_iteration
    from legoesm.training.deploy_correction import (
        apply_env_kernel_override,
        env_kernel_from_dict,
        env_kernel_to_dict,
    )
    from legoesm.training.feedback_assembly import column_environment_grid

    from scripts.run.run_correction_campaign import make_les_diagnose_fn, maybe_env_grid_fn

    driver = _run_clubb_coupled(CLUBBLiteConfig(C_K=0.4))
    a = driver._atm
    model0 = column_state_from_hydrostatic(
        a.state, a.q_v, sst_K=driver._ocean_state.T_sfc.data)
    n_lat, n_lon, _ = model0.T.shape
    sigma, grid = a.sigma, a.grid
    rad2deg = 180.0 / float(jnp.pi)

    bias = np.zeros((n_lat, n_lon))
    bias[n_lat // 2, n_lon // 2] = 5.0
    bias[0, 0] = 3.0
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
        area_weights=jnp.ones((n_lat, n_lon)), n_worst=3,
        run_amip_fn=_run, coordinate=sigma)

    les_cfg = ColumnLESConfig(regime=_SMALL_REGIME, gate_les_realism=False)
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_cfg,
        run_les_fn=_partial(run_forced_les, dt_s=0.5, n_steps=2))

    # The SAME env_grid_fn the campaign builds for --feedback-strategy environment.
    env_grid_fn = maybe_env_grid_fn("environment", sigma)
    res = run_correction_iteration(
        CLUBBLiteConfig(C_K=0.4), compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        promotion_key="clubb_lite_C_K", grid_shape=(n_lat, n_lon), background=0.4,
        diagnosis_method="eddy_diffusivity", les_budget=3,
        feedback_strategy="environment", env_grid_fn=env_grid_fn)

    # The env-kernel was BUILT from the REAL run's worst-column env tags.
    assert res.env_kernel is not None
    # It round-trips through the on-disk <out>.env_kernel.json contract ...
    kernel = env_kernel_from_dict(env_kernel_to_dict(res.env_kernel))
    assert kernel.field == "C_K"
    # ... and DEPLOYS: evaluating it on the model's OWN environment yields a finite per-column
    # override (the cross-grid deploy's core operation; same default hybrid pressures).
    grid_env, _ = column_environment_grid(model0, sigma)
    override, coverage = apply_env_kernel_override(kernel, grid_env)
    ck = np.asarray(override.clubb_lite.C_K)
    assert ck.shape == (n_lat * n_lon,) and bool(np.all(np.isfinite(ck)))
    assert 0.0 <= float(coverage["fraction_covered"]) <= 1.0


@pytest.mark.slow
@pytest.mark.filterwarnings("error::FutureWarning")
def test_full_loop_multi_coefficient_corrects_all_from_one_les():
    """REGRESSION (iter 489): the SIMULTANEOUS multi-coefficient correction (one LES run →
    C_K + Pr_t + C_eps diagnoses → ATOMIC gate) through a REAL model loop. The multi path is
    unit-tested only with MOCK compares (test_correction_loop); the 3 diagnosis functions are
    each unit-tested in isolation — but their SIMULTANEOUS application from ONE real LES
    output + the atomic all-or-nothing gate was untested end-to-end (the iter-484 integration
    lesson: pieces tested separately can still break composed). Builds the specs + the
    multi-method diagnose_fn the SAME way ``build_multi_correction_campaign`` does."""
    from functools import partial as _partial

    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig, run_forced_les
    from legoesm.training.compare_reanalysis import column_state_from_hydrostatic
    from legoesm.training.correction_loop import (
        CorrectionSpec,
        make_compare_fn,
        run_multi_correction_iteration,
    )

    from scripts.run.run_correction_campaign import (
        COEFFICIENT_SPEC_MAP,
        make_les_diagnose_fn,
    )

    driver = _run_clubb_coupled(CLUBBLiteConfig(C_K=0.4))
    a = driver._atm
    model0 = column_state_from_hydrostatic(
        a.state, a.q_v, sst_K=driver._ocean_state.T_sfc.data)
    n_lat, n_lon, _ = model0.T.shape
    sigma, grid = a.sigma, a.grid
    rad2deg = 180.0 / float(jnp.pi)

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

    # Correct ALL THREE CLUBB coefficients simultaneously — specs + the multi-method
    # diagnose_fn built EXACTLY as build_multi_correction_campaign does (one LES → 3 methods).
    coeffs = ("C_K", "Pr_t", "C_eps")
    specs = [CorrectionSpec(*COEFFICIENT_SPEC_MAP[n],
                            float(getattr(CLUBBLiteConfig(), n))) for n in coeffs]
    methods = tuple(dict.fromkeys(s.diagnosis_method for s in specs))
    les_cfg = ColumnLESConfig(
        regime=_SMALL_REGIME, gate_les_realism=False, diagnosis_methods=methods,
        clubb_l_mix_max=float(CLUBBLiteConfig().l_mix_max))
    diagnose_fn = make_les_diagnose_fn(
        grid, sigma, les_config=les_cfg,
        run_les_fn=_partial(run_forced_les, dt_s=0.5, n_steps=2))

    res = run_multi_correction_iteration(
        CLUBBLiteConfig(C_K=0.4), specs, compare_fn=compare_fn, diagnose_fn=diagnose_fn,
        grid_shape=(n_lat, n_lon), les_budget=2)

    # The whole multi chain ran from ONE LES: finite biases + a definite atomic gate verdict.
    assert np.isfinite(float(res.bias.baseline_bias))
    assert np.isfinite(float(res.bias.updated_bias))
    assert isinstance(bool(res.bias.improved), bool)
    assert res.n_corrected >= 1
    # ALL THREE coefficients got a per-column feedback field (the simultaneous correction).
    for name in coeffs:
        key = COEFFICIENT_SPEC_MAP[name][0]
        assert key in res.feedback_fields, (key, sorted(res.feedback_fields))
        fld = np.asarray(res.feedback_fields[key]).reshape(-1)
        assert fld.shape == (n_lat * n_lon,) and bool(np.all(np.isfinite(fld)))


def _amip_base_with_forcing(tmp_path, *, resolution=4, nlev=5):
    """A base AMIP ``ExperimentConfig`` with a forcing BUILT from the synthetic local ERA5
    archive and injected via the shared ``apply_amip_forcing_to_config`` (the turnkey path)."""
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )

    from scripts.data.load_local_era5 import (
        apply_amip_forcing_to_config,
        build_era5_amip_forcing,
    )

    write_forcing_archive(tmp_path)             # the shared 8x16 forcing archive
    fcfg = build_era5_amip_forcing(
        str(tmp_path), "20200101", str(tmp_path / "amip_forcing.nc"), hour_stride=24)
    base = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=1), radiation="gray", days=1,
        turbulence="clubb_lite")
    return apply_amip_forcing_to_config(base, fcfg)


def _run_clubb_amip(clubb, base_cfg):
    """A real AMIP ``ModelDriver`` run that splices the given ``CLUBBLiteConfig`` into the
    forcing-injected ``base_cfg`` (the per-round C_K correction); returns the driver."""
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.driver.model_driver import ModelDriver

    cfg = base_cfg._replace(
        turbulence_override=TurbulenceConfig(scheme="clubb_lite", clubb_lite=clubb))
    driver = ModelDriver(cfg)
    driver.setup()
    driver.run()
    return driver


@pytest.mark.slow
@pytest.mark.filterwarnings("error::FutureWarning")
def test_full_offline_amip_loop_forcing_drives_correction_and_persists(tmp_path):
    """The COMPLETE offline-AMIP turnkey path in ONE round: build the SST forcing from a
    synthetic local ERA5 archive → a real ModelDriver AMIP run → compare → REAL plane-LES →
    diagnose C_K → RE-RUN the AMIP model → monotonic gate.  The iter-415 full-loop test is
    COUPLED (CMIP); this is the ONLY place the offline ERA5 forcing flows through the WHOLE
    correction loop in AMIP mode — and it pins that the forcing PERSISTS through the
    per-round C_K ``_replace`` (every re-run uses it, the iter-421 claim).  CI-portable
    (synthetic archive + reference); asserts the loop runs end-to-end (finite biases, a
    definite verdict, ≥1 worst column flagged + corrected), NOT a reference-dependent bias
    reduction (the gate may correctly REJECT, as it did against real ERA5)."""
    from functools import partial as _partial

    from legoesm.atmosphere.dynamics.les.column_les import ColumnLESConfig, run_forced_les
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.training.correction_loop import make_compare_fn, run_correction_iteration
    from legoesm.training.run_to_column_mean import amip_column_state

    from scripts.run.run_correction_campaign import make_les_diagnose_fn

    base_cfg = _amip_base_with_forcing(tmp_path)
    # The forcing PERSISTS through a per-round C_K correction (the _replace of the turbulence
    # override does NOT touch the forcing fields) — the iter-421 "every corrected round
    # carries the forcing" claim, verified directly.
    corrected_cfg = base_cfg._replace(
        turbulence_override=TurbulenceConfig(
            scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=0.7)))
    assert base_cfg.dataset == "custom" and corrected_cfg.dataset == "custom"
    assert corrected_cfg.forcing_path == base_cfg.forcing_path != ""

    driver0 = _run_clubb_amip(CLUBBLiteConfig(C_K=0.4), base_cfg)
    model0 = amip_column_state(driver0, day=0.0)
    n_lat, n_lon, _ = model0.T.shape
    sigma, grid = driver0.sigma, driver0.grid
    rad2deg = 180.0 / float(jnp.pi)
    # The prescribed SST in the column state carries the ERA5 forcing's ~21 K equator-pole
    # gradient (NOT a degenerate constant) — with the dataset="custom" persistence check
    # above, this confirms the real built forcing (not an analytical fallback) loaded.
    sst0 = jnp.asarray(model0.sst_K)
    assert float(jnp.max(sst0) - jnp.min(sst0)) > 10.0

    # Synthetic reference (CI-portable): the model minus a localized warm bias so the
    # worst-column ranking + the LES are non-trivially exercised.
    bias = np.zeros((n_lat, n_lon))
    bias[n_lat // 2, n_lon // 2] = 5.0
    reference = model0._replace(T=model0.T - jnp.asarray(bias)[:, :, None])

    def _run(clubb):
        return amip_column_state(_run_clubb_amip(clubb, base_cfg), day=0.0)

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

    # The whole offline-AMIP chain ran: finite biases + a DEFINITE gate verdict + ≥1 worst
    # column flagged + corrected (the loop is not a vacuous no-op).  n_corrected counts the
    # flagged worst columns; the tiny 2-step LES need not yield a VALID diagnosis every run
    # (so n_diagnoses_valid is not asserted, matching the iter-415 coupled full-loop test).
    assert np.isfinite(float(res.bias.baseline_bias))
    assert np.isfinite(float(res.bias.updated_bias))
    assert isinstance(bool(res.bias.improved), bool)
    assert res.n_corrected >= 1


@pytest.mark.slow
@pytest.mark.filterwarnings("error::FutureWarning")
def test_amip_offline_forcing_osse_reduces_bias_with_applied_ck(tmp_path):
    """Clause 6 ("updating these parameters IMPROVE the biases") — the bias REDUCTION — in
    AMIP mode with the OFFLINE ERA5 forcing, the closest in-repo proxy to the HPC empirical
    demo (real prescribed SST + AMIP physics).  iter-413 demonstrated this REDUCTION for
    CMIP (coupled); iter-422 runs the AMIP loop but asserts only that it RUNS (synthetic
    reference, gate may reject).  The AMIP twin: truth = the offline-forced AMIP model
    @C_K=1.0, biased = the SAME model @C_K=0.4 (the bias is C_K-driven BY CONSTRUCTION, the
    prescribed SST forcing identical across runs), apply the diagnosed C_K=1.0 to EVERY
    column and RE-RUN → the deterministic re-run equals the truth, the bias falls to ~0, and
    the monotonic gate ACCEPTS — proving the full AMIP loop with the offline forcing LOWERS
    the bias with the real ESM (the real-ERA5 case stays the HPC-only empirical question)."""
    from legoesm.training.correction_loop import make_compare_fn, run_correction_iteration
    from legoesm.training.run_to_column_mean import amip_column_state

    base_cfg = _amip_base_with_forcing(tmp_path)
    truth_driver = _run_clubb_amip(CLUBBLiteConfig(C_K=1.0), base_cfg)
    truth = amip_column_state(truth_driver, day=0.0)
    n_lat, n_lon, _ = truth.T.shape
    ncol = n_lat * n_lon
    sigma, grid = truth_driver.sigma, truth_driver.grid
    rad2deg = 180.0 / float(jnp.pi)

    def _run(clubb):
        return amip_column_state(_run_clubb_amip(clubb, base_cfg), day=0.0)

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

    assert float(res.bias.baseline_bias) > 1e-5            # a real C_K-driven AMIP bias exists
    assert float(res.bias.updated_bias) < float(res.bias.baseline_bias)   # it FELL
    assert float(res.bias.updated_bias) == pytest.approx(0.0, abs=1e-6)   # to the twin truth
    assert bool(res.bias.improved)                         # the monotonic gate accepts
    assert res.n_corrected == ncol                         # every column corrected
