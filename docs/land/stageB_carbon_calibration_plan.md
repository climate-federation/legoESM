# Stage B — differentiable carbon-parameter calibration (v1: SOC)

Warm-start from the Stage-A IC-map defaults and gradient-tune the DifferLand SOM
parameters against observed soil organic carbon, so the freeze-floor / turnover /
transfer values are calibrated instead of hand-set (the `som_freeze_floor`
"Stage-B calibration target" the realism inspection flagged). Modular observation
registry so LAI / biomass / SIF / δ¹³C plug in later.

## What already exists (reuse, do not re-derive)
- **Trainable collection:** `legoesm.land.carbon.config` is in
  `param_collector.SPEC_MODULES`; `build_trainable_params(CarbonConfig, tier=…)`
  collects the tier-1/2 SOM params (`tor_som_active/slow/passive`,
  `f_active_to_slow`, `f_slow_to_passive`, `som_freeze_floor`, `Q10_het_exp`,
  `cwd_humification_eff`, …) with sigmoid/softplus transforms + bounds from
  `__param_spec__`. `apply_param_overrides(cfg, field_values)` injects TRACED
  leaves (`NamedTuple._replace`) — the SegmentForcing/SCM-RCE doctrine.
- **Forward map (differentiable):** `build_archetypes` (k-means) is STATIC numpy —
  the archetype table + cell membership do NOT depend on the trainable physics, so
  they are built ONCE. `equilibrate_archetypes` → `iter_archetype_batches` →
  `run_semi_analytic_spinup` is JAX (`lax.scan` + analytic solve) → differentiable
  w.r.t. the CarbonConfig SOM leaves.
- **Observation (no external fetch):** the surfdata carries per-layer organic
  carbon (`global_surface_data.organic_var="ORGANIC"` [kg/m³]; HWSD `ORG_CARBON`);
  observed column SOC [kgC/m²] = ∫ organic·dz over the soil column, per cell.
- **Optimizer + loop + output:** mirror `scripts/run/run_scm_rce_campaign.py` —
  `ml.training.create_optimizer` (MUON, warmup+cosine+clip), `filter_value_and_grad`,
  `to_overrides()` → `apply_param_overrides` INSIDE the loss, write a RECOMMENDED
  `tuned_parameters.json` under `results/` (NEVER mutate production `*Config`
  defaults; production keeps static Python-float leaves).
- **Losses:** `ml/loss.py::area_weighted_mse`.

## Forward map for training
1. Build the archetype table + `cell_archetype_id/weight` ONCE (real CLM5 cover +
   ERA5 climate, or load a saved `archetypes.npz`).
2. **Per-archetype observed SOC target** = cover-weighted mean of column-integrated
   `ORGANIC` over the cells assigned to that archetype (via `cell_archetype_id/weight`).
   (Optionally biomass target from the CLM5 monthly biomass/LAI — modular, v2.)
3. `equilibrate_archetypes_traced(table, carbon_overrides) → per-archetype
   CarbonState`: identical to `equilibrate_archetypes` but `apply_param_overrides`
   injects the TRACED `carbon_overrides` into each group's `CarbonConfig` before the
   spin-up, so `∂(C_som_total_eq)/∂params` flows via `jax.grad`. Use
   `jax.checkpoint` on the per-year spin-up scan if reverse-mode memory is the bound;
   a shorter training spin-up is acceptable (fast pools + the analytic slow reset do
   the heavy lifting).
4. **Modular loss registry:** `{obs_name: (target, model_extractor, weight)}` —
   v1 = `som_total` vs observed SOC (area/cover-weighted MSE); registry so
   biomass/LAI/SIF/δ¹³C add without touching the loop.

## Phased
- **B1 (crux — retire the differentiability risk):** the observed-SOC loader +
  `equilibrate_archetypes_traced` + a test proving `jax.grad(loss)(params)` is
  FINITE and non-zero on a tiny 2–3-archetype table (i.e. the SOM params actually
  move the equilibrium SOC and the gradient flows). No optimizer yet.
- **B2 (training loop):** `scripts/run/train_carbon_params.py` — warm-start from
  IC-map defaults, `build_trainable_params(tier="extended")`, the modular SOC(+biomass)
  loss, `create_optimizer` MUON, N steps, write `results/carbon_calibration/
  tuned_carbon_parameters.json` + a before/after SOC scorecard. A short training run
  on a compute node; report the tuned `som_freeze_floor` etc. vs the hand-set default.

## Guardrails
- Every new `.py` gets a direct test. Overrides applied INSIDE the loss (traced);
  production defaults untouched. No new hardcoded constants. Codex adversarial review
  on the differentiable forward + the loss. Compute via `srun`/`sbatch` only.
