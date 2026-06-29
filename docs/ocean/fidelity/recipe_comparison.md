# Recipe wiring comparison

**GENERATED — do not edit by hand.** Source: the recipe catalog `legoesm.ocean.recipes` (`get_recipe`); regenerate with `scripts/validate/ocean_fidelity/build_recipe_comparison.py`. Freshness enforced by `tests/ocean/fidelity/test_recipe_comparison.py`.

Each named recipe is a bundle of dycore-identity numerics choices (the Veros / NEMO / Oceananigans / MITgcm / legoESM-default dycores). This table puts them side by side: the **✏ rows differ** between recipes (what makes each numerically distinct); the _(shared)_ rows are common to all. Compare two columns to see exactly what changes in the numerics.

`—` = the recipe does not set this field (uses the model default). A field where some recipes set a value and others show `—` still **differs** (one pins it, another inherits the default).

## latlon recipes

7 recipes × 21 dycore-identity fields. **21 fields differ** (✏, listed first) — those are what distinguishes these recipes; the rest are shared.

| field | `default_wright_v1` | `eady_weno5_v1` | `legoesm_linear_v1` | `nemo_dino_v1` | `nemo_v1` | `omip_nemo_match_tripole_v1` | `veros_faithful_v1` |
|---|---|---|---|---|---|---|---|
| ✏ **A_h_lat_scaling** | `—` | `—` | `—` | `True` | `—` | `—` | `—` |
| ✏ **ab2_scope** | `—` | `—` | `—` | `—` | `—` | `—` | `advective` |
| ✏ **adaptive_implicit_vertadv** | `—` | `—` | `—` | `—` | `True` | `—` | `—` |
| ✏ **barotropic_solver** | `explicit_substep` | `implicit_cn` | `explicit_substep` | `implicit_cn` | `explicit_substep` | `implicit_cn` | `rigid_lid` |
| ✏ **barotropic_time_filter** | `—` | `—` | `—` | `—` | `cosine` | `—` | `—` |
| ✏ **coriolis_scheme** | `matsuno_split` | `—` | `matsuno_split` | `matsuno_split` | `—` | `matsuno_split` | `explicit_ab2` |
| ✏ **eos** | `wright` | `linear` | `linear` | `wright` | `veros_gsw` | `wright` | `veros_nonlin2` |
| ✏ **implicit_vertical_mixing** | `True` | `—` | `True` | `True` | `True` | `True` | `True` |
| ✏ **implicit_vmix_dzw_slot** | `—` | `—` | `—` | `—` | `—` | `—` | `True` |
| ✏ **ke_gradient_scheme** | `centered` | `centered` | `centered` | `hollingsworth` | `hollingsworth` | `centered` | `centered` |
| ✏ **lateral_viscosity_operator** | `—` | `—` | `—` | `—` | `—` | `—` | `flux_divergence` |
| ✏ **momentum_advection** | `vector_invariant` | `weno5` | `vector_invariant` | `vector_invariant` | `vector_invariant` | `vector_invariant` | `flux_form` |
| ✏ **momentum_flux_scheme** | `—` | `—` | `—` | `—` | `—` | `—` | `centered` |
| ✏ **momentum_friction_additive** | `—` | `—` | `—` | `—` | `—` | `—` | `True` |
| ✏ **momentum_time_integrator** | `—` | `—` | `—` | `—` | `rk3` | `—` | `—` |
| ✏ **n_barotropic_substeps** | `—` | `—` | `30` | `—` | `30` | `30` | `—` |
| ✏ **outer_integrator** | `forward_euler` | `ab2` | `forward_euler` | `forward_euler` | `—` | `forward_euler` | `ab2` |
| ✏ **pgf_scheme** | `adcroft` | `smc03` | `adcroft` | `adcroft` | `smc03` | `adcroft` | `adcroft` |
| ✏ **tracer_advection** | `tvd` | `weno5` | `tvd` | `tvd` | `ppm_fct` | `tvd` | `centered` |
| ✏ **tracer_time_integrator** | `euler` | `rk3` | `euler` | `euler` | `—` | `euler` | `euler` |
| ✏ **vertical_momentum_scheme** | `—` | `—` | `—` | `—` | `—` | `—` | `centered_full` |

## mpas recipes

3 recipes × 6 dycore-identity fields. **4 fields differ** (✏, listed first) — those are what distinguishes these recipes; the rest are shared.

| field | `default_wright_mpas_v1` | `legoesm_linear_mpas_v1` | `omip_nemo_match_mpas_v1` |
|---|---|---|---|
| ✏ **barotropic_solver** | `explicit_substep` | `explicit_substep` | `implicit_cn` |
| ✏ **eos** | `wright` | `linear` | `wright` |
| ✏ **pgf_scheme** | `centered` | `centered` | `adcroft` |
| ✏ **tracer_advection** | `upwind` | `upwind` | `tvd` |
| implicit_vertical_mixing _(shared)_ | `True` | `True` | `True` |
| pv_scheme _(shared)_ | `enstrophy` | `enstrophy` | `enstrophy` |
