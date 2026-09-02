# Recipe wiring comparison

**GENERATED — do not edit by hand.** Source: the recipe catalog `legoesm.ocean.recipes` + the oracle-recipe factories (`oceananigans_recipe`, `mitgcm_recipe`); regenerate with `scripts/validate/ocean_fidelity/build_recipe_comparison.py`. Freshness enforced by `tests/ocean/fidelity/test_recipe_comparison.py`.

Each recipe is a bundle of dycore-identity numerics choices. This table puts them side by side: the **✏ rows differ** between recipes (what makes each numerically distinct); _(shared)_ rows are common. Compare two columns to see exactly what changes in the numerics. The oracle dycores `oceananigans_v1` / `mitgcm_v1` are ordinary catalog recipes here — their `*_canonical_ocean_config` factories now source their defaults from the catalog (single source).

## latlon recipes (effective dycore values)

9 recipes × 14 dycore-identity fields. **13 fields differ** (✏, first). Values are EFFECTIVE (defaults resolved), so two columns compared = the real numerical difference. The oracle dycores `oceananigans_v1` / `mitgcm_v1` are now ordinary catalog recipes (their factories source defaults from them).

| field | `default_wright_v1` | `eady_weno5_v1` | `legoesm_linear_v1` | `legoesm_nemo_like_v1` | `mitgcm_v1` | `nemo_dino_v1` | `oceananigans_v1` | `omip_nemo_match_tripole_v1` | `veros_faithful_v1` |
|---|---|---|---|---|---|---|---|---|---|
| ✏ **eos** | `wright` | `linear` | `linear` | `veros_gsw` | `linear` | `wright` | `linear` | `wright` | `veros_nonlin2` |
| ✏ **momentum_advection** | `vector_invariant` | `weno5` | `vector_invariant` | `vector_invariant` | `flux_form` | `vector_invariant` | `vector_invariant` | `vector_invariant` | `flux_form` |
| ✏ **tracer_advection** | `tvd` | `weno5` | `tvd` | `ppm_fct` | `tvd` | `tvd` | `weno7` | `tvd` | `centered` |
| ✏ **coriolis_scheme** | `matsuno_split` | `matsuno_split` | `matsuno_split` | `matsuno_split` | `explicit_ab2` | `matsuno_split` | `explicit_ab2` | `matsuno_split` | `explicit_ab2` |
| ✏ **barotropic_solver** | `explicit_substep` | `implicit_cn` | `explicit_substep` | `explicit_substep` | `explicit_substep` | `implicit_cn` | `implicit_cn` | `implicit_cn` | `rigid_lid` |
| ✏ **pgf_scheme** | `adcroft` | `smc03` | `adcroft` | `smc03` | `adcroft` | `adcroft` | `adcroft` | `adcroft` | `adcroft` |
| ✏ **ke_gradient_scheme** | `centered` | `centered` | `centered` | `hollingsworth` | `centered` | `hollingsworth` | `centered` | `centered` | `centered` |
| ✏ **lateral_viscosity_operator** | `vector_laplacian` | `vector_laplacian` | `vector_laplacian` | `vector_laplacian` | `flux_divergence` | `vector_laplacian` | `flux_divergence` | `vector_laplacian` | `flux_divergence` |
| ✏ **outer_integrator** | `forward_euler` | `ab2` | `forward_euler` | `forward_euler` | `ab2` | `forward_euler` | `ab2` | `forward_euler` | `ab2` |
| ✏ **tracer_time_integrator** | `euler` | `rk3` | `euler` | `euler` | `euler` | `euler` | `euler` | `euler` | `euler` |
| ✏ **vertical_momentum_scheme** | `upwind_perturbation` | `upwind_perturbation` | `upwind_perturbation` | `upwind_perturbation` | `upwind_perturbation` | `upwind_perturbation` | `upwind_perturbation` | `upwind_perturbation` | `centered_full` |
| ✏ **ab2_scope** | `total` | `total` | `total` | `total` | `total` | `total` | `total` | `total` | `advective` |
| ✏ **momentum_flux_scheme** | `upwind` | `upwind` | `upwind` | `upwind` | `centered` | `upwind` | `upwind` | `upwind` | `centered` |
| barotropic_time_filter _(shared)_ | `cosine` | `cosine` | `cosine` | `cosine` | `cosine` | `cosine` | `cosine` | `cosine` | `cosine` |

## MPAS recipes (declared scheme bundle)

3 recipes × 7 fields. **5 differ** (✏, first). Declared overrides (`—` = inherits the MPAS model default); no oracle-recipe factories exist for MPAS.

| field | `default_wright_mpas_v1` | `legoesm_linear_mpas_v1` | `omip_nemo_match_mpas_v1` |
|---|---|---|---|
| ✏ **barotropic_solver** | `explicit_substep` | `explicit_substep` | `implicit_cn` |
| ✏ **eos** | `wright` | `linear` | `wright` |
| ✏ **normalize_freshwater** | `—` | `—` | `True` |
| ✏ **pgf_scheme** | `centered` | `centered` | `adcroft` |
| ✏ **tracer_advection** | `upwind` | `upwind` | `tvd` |
| implicit_vertical_mixing _(shared)_ | `True` | `True` | `True` |
| pv_scheme _(shared)_ | `enstrophy` | `enstrophy` | `enstrophy` |
