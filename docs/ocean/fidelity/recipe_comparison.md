# Recipe wiring comparison

**GENERATED — do not edit by hand.** Source: the recipe catalog `legoesm.ocean.recipes` + the oracle-recipe factories (`oceananigans_recipe`, `mitgcm_recipe`); regenerate with `scripts/validate/ocean_fidelity/build_recipe_comparison.py`. Freshness enforced by `tests/ocean/fidelity/test_recipe_comparison.py`.

Each recipe is a bundle of dycore-identity numerics choices. This table puts them side by side: the **✏ rows differ** between recipes (what makes each numerically distinct); _(shared)_ rows are common. Compare two columns to see exactly what changes in the numerics. `*` marks the oracle-recipe factories (Oceananigans, MITgcm) — runnable-oracle dycores, not `list_recipes()` catalog entries.

## latlon recipes (effective dycore values)

9 recipes × 14 dycore-identity fields. **13 fields differ** (✏, first). `*` = oracle-recipe factory (not a `list_recipes()` catalog entry). Values are EFFECTIVE (defaults resolved), so two columns compared = the real numerical difference.

| field | `default_wright_v1` | `eady_weno5_v1` | `legoesm_linear_v1` | `nemo_dino_v1` | `nemo_v1` | `omip_nemo_match_tripole_v1` | `veros_faithful_v1` | `oceananigans*` | `mitgcm*` |
|---|---|---|---|---|---|---|---|---|---|
| ✏ **eos** | `wright` | `linear` | `linear` | `wright` | `veros_gsw` | `wright` | `veros_nonlin2` | `linear` | `linear` |
| ✏ **momentum_advection** | `vector_invariant` | `weno5` | `vector_invariant` | `vector_invariant` | `vector_invariant` | `vector_invariant` | `flux_form` | `vector_invariant` | `flux_form` |
| ✏ **tracer_advection** | `tvd` | `weno5` | `tvd` | `tvd` | `ppm_fct` | `tvd` | `centered` | `weno7` | `centered` |
| ✏ **coriolis_scheme** | `matsuno_split` | `matsuno_split` | `matsuno_split` | `matsuno_split` | `matsuno_split` | `matsuno_split` | `explicit_ab2` | `explicit_ab2` | `explicit_ab2` |
| ✏ **barotropic_solver** | `explicit_substep` | `implicit_cn` | `explicit_substep` | `implicit_cn` | `explicit_substep` | `implicit_cn` | `rigid_lid` | `implicit_cn` | `implicit_unsplit` |
| ✏ **pgf_scheme** | `adcroft` | `smc03` | `adcroft` | `adcroft` | `smc03` | `adcroft` | `adcroft` | `adcroft` | `adcroft` |
| ✏ **ke_gradient_scheme** | `centered` | `centered` | `centered` | `hollingsworth` | `hollingsworth` | `centered` | `centered` | `centered` | `centered` |
| ✏ **lateral_viscosity_operator** | `vector_laplacian` | `vector_laplacian` | `vector_laplacian` | `vector_laplacian` | `vector_laplacian` | `vector_laplacian` | `flux_divergence` | `flux_divergence` | `flux_divergence` |
| ✏ **outer_integrator** | `forward_euler` | `ab2` | `forward_euler` | `forward_euler` | `forward_euler` | `forward_euler` | `ab2` | `ab2` | `ab2` |
| ✏ **tracer_time_integrator** | `euler` | `rk3` | `euler` | `euler` | `euler` | `euler` | `euler` | `euler` | `euler` |
| ✏ **vertical_momentum_scheme** | `upwind_perturbation` | `upwind_perturbation` | `upwind_perturbation` | `upwind_perturbation` | `upwind_perturbation` | `upwind_perturbation` | `centered_full` | `upwind_perturbation` | `upwind_perturbation` |
| ✏ **ab2_scope** | `total` | `total` | `total` | `total` | `total` | `total` | `advective` | `total` | `total` |
| ✏ **momentum_flux_scheme** | `upwind` | `upwind` | `upwind` | `upwind` | `upwind` | `upwind` | `centered` | `upwind` | `centered` |
| barotropic_time_filter _(shared)_ | `cosine` | `cosine` | `cosine` | `cosine` | `cosine` | `cosine` | `cosine` | `cosine` | `cosine` |

## MPAS recipes (declared scheme bundle)

3 recipes × 6 fields. **4 differ** (✏, first). Declared overrides (`—` = inherits the MPAS model default); no oracle-recipe factories exist for MPAS.

| field | `default_wright_mpas_v1` | `legoesm_linear_mpas_v1` | `omip_nemo_match_mpas_v1` |
|---|---|---|---|
| ✏ **barotropic_solver** | `explicit_substep` | `explicit_substep` | `implicit_cn` |
| ✏ **eos** | `wright` | `linear` | `wright` |
| ✏ **pgf_scheme** | `centered` | `centered` | `adcroft` |
| ✏ **tracer_advection** | `upwind` | `upwind` | `tvd` |
| implicit_vertical_mixing _(shared)_ | `True` | `True` | `True` |
| pv_scheme _(shared)_ | `enstrophy` | `enstrophy` | `enstrophy` |
