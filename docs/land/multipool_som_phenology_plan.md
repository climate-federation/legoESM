# Multi-pool SOM + per-PFT phenology — design & phased plan

Stage-A.1 follow-ups (post realism-inspection): the zonal-demo IC build showed
boreal PFTs collapse to 0 and grasses/high-latitude soils under-accumulate SOC.
The boreal collapse is chiefly a **climate** artifact (fixed separately by the
ERA5 2-D climatology, `build_land_forcing_climatology.py`). The grass/high-lat
**SOC** underestimate is a **model** limitation of the single soil-carbon pool,
documented in `carbon_equilibrium_audit.md` §Residual. This plan fixes it.

## Design decisions

### Multi-pool SOM (CENTURY / CLM4.5-style, 3 pools)
- `CarbonState.C_som` → `C_som_active`, `C_som_slow`, `C_som_passive` (6→8 pools).
  A module-level `som_total(state)` helper returns their sum for the many
  total-context call sites (diagnostics, conservation, mapping).
- **Topology = forward cascade** (active→slow→passive) with a heterotrophic
  respiration fraction leaving each pool; litter decomposition + CWD humification
  feed the active pool. No back-transfer → transfer matrix `K` is
  lower-triangular, so the analytic equilibrium is exact **forward substitution**
  (`C_active_eq = I_a/k_a`; `C_slow_eq = (I_s + f_as·loss_a)/k_s`; …), preserving
  the current per-pool `jnp.where(loss>eps, …, unchanged)` degenerate-column
  guard — no singular 3×3 inverse. Base turnover times (reference T, moist):
  active ~1–5 yr, slow ~20–50 yr, passive ~500–1000 yr (CLM4.5 / Koven 2013).
- **Moisture + freeze control** multiplies each pool's decomposition rate:
  `k = k_base · f_temp(Q10_het) · f_moist(precip) · f_freeze(T)`. `f_freeze` is a
  smooth suppression as `T → below constants.T_freeze` (reusing the freeze-curve
  concept in `soil_thermal.liquid_water_content`, not a re-derived curve) →
  cold/frozen soils retain carbon = the high-latitude SOC fix. Extend the shared
  `_temperate_modifier`, do not copy it.
- Init partitions `C_som_init` across the three pools by CENTURY equilibrium
  fractions (active small, passive large).

### Per-PFT phenology
- Add `evergreen: bool` (categorical, `tunable_tier` excluded) to `CarbonConfig`.
  `compute_phenology`: evergreen → near-continuous leaf turnover
  (`leaf_lifespan`-based) instead of the temperate Gaussian `Fday` pulse.
- Classify each archetype's PFT via an `_is_evergreen` substring matcher on
  `CLM5_PFT_NAMES` (mirroring `_is_woody`); add `phenology_type` as a 3rd group
  key in `global_init.iter_archetype_batches` so tropical-evergreen archetypes
  equilibrate with continuous phenology. **Offline IC path only** — production
  coupled runs keep the default (no new per-column plumbing through
  `step_carbon`/`multilayer_land`; that is a later, larger change).

## Phased plan (each phase: conservation-gated + codex adversarial review)

- **A1 — pytree 6→8, behavior-preserving.** Rename `C_som`→`C_som_active`, add
  inert `C_som_slow`/`C_som_passive` (zero flux), add `som_total()`. Update ALL
  20 `CarbonState(` construction sites + ~28 `.C_som` accesses + `init_carbon_state`
  + `run_lmip` restart read/write + the two hardcoded `_POOL_FIELDS` copies +
  `_total_carbon` + `SlowPoolFluxes`/`analytic_slow_pool_equilibrium` (active
  only) + `CarbonConfig.C_som_init` partition. Slow/passive carry zero flux →
  **behaviour identical**; every existing conservation + carbon test stays green.
- **A2 — cascade dynamics.** Real active→slow→passive transfers + per-pool
  respiration + moisture/freeze modifier in `step_carbon_differland`; new
  `CarbonConfig` turnover/transfer fields + `__param_spec__`. Conservation
  (`ΣΔC == −NEE·dt`) machine-precision gate. Realism: high-lat/grass SOC rises.
- **A3 — semi-analytic solve.** Generalize `analytic_slow_pool_equilibrium` to the
  forward-substitution cascade for the 3 SOM pools; the `run_semi_analytic_spinup`
  verify-drift stays →0.
- **B — per-PFT phenology.** `evergreen` flag + `compute_phenology` branch +
  `_is_evergreen` archetype grouping.

## Risk register (from the blast-radius map)
1. `C_som` replace touches 28 bare `.C_som` accesses + `C_som_init`/`tor_som` in 5
   files — A1 must catch all; a `_POOL_FIELDS == CarbonState._fields` assertion
   test is added to prevent the 4 hardcoded copies drifting.
2. Positional `CarbonState(a(),…)` at `test_global_init.py:125` — new fields must
   have **no default** so a missed site fails loudly (TypeError), never silently.
3. `run_lmip` restart read uses a hardcoded field list — update in the same PR.
4. Analytic solve is a coupled system — solved as forward substitution (§design),
   not a copy-pasted per-pool guard.
