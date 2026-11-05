# SI3 PR duplicate implementations

This inventory is the grep-based review requested after the main catch-up at
`041c21c2c`.  A row means that the branch carries two executable translations
of the same NEMO routine or source block: the restored main implementation and
the SI3-ladder implementation.  The count is **22 routine/source-block rows**.
Closely coupled x/y sweeps and forward/inverse conversion pairs remain separate
where they have separate executable bodies.  Pure dataclasses, pack/unpack
helpers, state schemas, and wrappers with only one numerical body are excluded.

| NEMO routine or source block | Main implementation | SI3-lane implementation | Direct callers | Shared-fixture agreement |
|---|---|---|---|---|
| `icestp` / sea-ice step dispatch | `sea_ice.py:218` `step_sea_ice` | `sea_ice.py:584` `step_sea_ice_si3_lane` | Main: coupled model, SCM, main tests. Lane: `test_si3_thd_da` and the lane C1D composition. | Not compared as one output: they deliberately call different `ice_thd` rows below. |
| `ice_thd` column chain | `bitz_lipscomb.py:1529` `si3_column_step_arrays` | `bitz_lipscomb.py:3463` `si3_lane_column_step_arrays` | Main: `_si3_step_with_trace`, t4col tests, phase-2/year gates. Lane: `_si3_lane_step_with_trace`, O9/O11/O12 tests. | C9 fixture: 22/72 common leaves equal; maximum absolute difference `9.313225746154785e10`. |
| `ice_var_salprof`, option 2 | `bitz_lipscomb.py:262` `option2_salinity_profile` | `bitz_lipscomb.py:1613` `_si3_502_option2_salinity_profile` | Main column/card builders; lane `ice_thd` chain. | Not compared: main carries its shallow-ice validation, lane carries 5.0.2 statement order. |
| `ice_var_glo2eqv`, ice enthalpy | `bitz_lipscomb.py:294` `ice_enthalpy_from_temperature` | `bitz_lipscomb.py:1663` `_si3_502_ice_enthalpy_from_temperature` | Main column/card/test setup; lane `ice_thd` chain. | Not compared: source-order and rounding contracts differ. |
| `ice_var_glo2eqv`, ice temperature | `bitz_lipscomb.py:329` `ice_temperature_from_enthalpy` | `bitz_lipscomb.py:1697` `_si3_502_ice_temperature_from_enthalpy` | Main column/card/tests; lane ZDF/DH/temperature stages. | Not compared: main keeps bounded/smoothed inverse controls; lane is the 5.0.2 source order. |
| `ice_var_glo2eqv`, snow enthalpy | `bitz_lipscomb.py:441` `snow_enthalpy_from_temperature` | `bitz_lipscomb.py:1761` `_si3_502_snow_enthalpy_from_temperature` | Main column/card setup; lane `ice_thd` chain. | Not compared: source-rounding contracts differ. |
| `ice_var_glo2eqv`, snow temperature | `bitz_lipscomb.py:449` `snow_temperature_from_enthalpy` | `bitz_lipscomb.py:1775` `_si3_502_snow_temperature_from_enthalpy` | Main column/tests; lane ZDF/DH stages. | Not compared: main and lane carry different bound/order contracts. |
| `ice_thd_zdf_bl99`, P07 conductivity | `bitz_lipscomb.py:471` `p07_conductivity` | `bitz_lipscomb.py:1819` `_si3_502_p07_conductivity` | Main `_si3_zdf_bl99_step`; lane `_si3_zdf_bl99_core`. | Not compared: lane includes the 5.0.2 floor and statement rounding. |
| `ice_thd_zdf_bl99`, shortwave radiation | `bitz_lipscomb.py:483` `_radiation` | `bitz_lipscomb.py:1893` `_si3_502_radiation` | Main `_si3_zdf_bl99_step`; lane `_si3_zdf_bl99_core`. | Not compared: the two routines expose different SSL/source-order contracts. |
| `ice_thd_zdf_bl99`, iterative solve | `bitz_lipscomb.py:513` `_si3_zdf_bl99_step` | `bitz_lipscomb.py:1961` `_si3_zdf_bl99_core` (via `si3_ice_thd_zdf_bl99`) | Main column chain; lane category stages. | Main digest restoration: all 180 previously moved ZDF observations returned to equality with `ee0de42471`. |
| `snw_ent` / `ice_var_vremap` cumulative remap | `bitz_lipscomb.py:813` `_nemo_snow_enthalpy_remap` and main DH body | `bitz_lipscomb.py:2388` `_cumulative_remap` in `_si3_ice_thd_dh_core` | Main column chain; lane `si3_ice_thd_dh`. | Not compared: main includes its reverse-safe/floor fixes; lane preserves 5.0.2 order. |
| `ice_thd_do` | `bitz_lipscomb.py:1435` `_do_step` | `bitz_lipscomb.py:3231` `si3_ice_thd_do` | Main column chain; lane column and ORCA2 stages. | Not compared: main is the resolved jpl=1/5.0.1 arm; lane is the general 5.0.2 arm. |
| `icedyn_rhg_evp`, final deformation | `dynamics.py:1249` `si3_cgrid_deformation` | `dynamics.py:1326` `si3_lane_cgrid_deformation` | Main ICE_RHEO/rung-3.3/trajectory paths; lane ICE_RHEO recipe and its six probes. | Periodic fixture: 2/2 arrays exactly equal; fold programs differ by contract. |
| `icedyn_rhg_evp`, aEVP solver | `dynamics.py:1374` `si3_cgrid_aevp_solver` | `dynamics.py:2007` `_si3_orca2_cgrid_aevp_core` via `si3_orca2_aevp_solver` | Main ORCA1, ICE_RHEO, landfast, rung-3.3; lane ORCA2 O5 wrappers. | Two-subcycle periodic fixture: 5/5 arrays exactly equal. |
| `ice_var_strength` H79 arm | `dynamics.py:1427` inline H79 strength in `si3_cgrid_aevp_solver` | `dynamics.py:2764` `si3_orca2_ice_strength` | Main ORCA1 aEVP; lane ORCA2 O5 dynamics. | Not compared: grep finds no current `_si3_ice_strength` definition; the restored main body is inline and the lane body adds smoothing/fold handling. |
| `ice_dyn_rdgrft` | `ridging.py:590` `apply_si3_jpl1_ridging` | `ridging.py:1300` `si3_ice_dyn_rdgrft` | Main JPL1 tests and main ICE_RHEO recipe; lane `ridge_ice_rheo` and ORCA2 O6. | Shared ICE_RHEO card: post-ridging maximum absolute difference `2.0208201778615376e-10`. |
| `icedyn_adv_pra`, x/y sweeps | `transport.py:2091` `advect_si3_prather_2d` | `transport.py:2349` `si3_adv_x` plus `si3_adv_y` | Main ADV2D/ICE_RHEO/rung-3.3; lane ORCA2 O4 state step. | Main wrappers share most kernels, but lane selects strict loop domains; no single equivalent end-to-end fixture. |
| `icedyn_adv_pra`, x/y limiter tie order | `transport.py:615,648` `_si3_prather_limit_x/_y` | `transport.py:681,715` `si3_lane_prather_limit_x/_y` | Main 1-D/2-D Prather substeps; lane `si3_adv_x/si3_adv_y` through strict loop domains. | At the exact binary tie `1.25 - 1.0 = 0.25`, main x/y forward bits and VJPs are bit-identical to `ee0de42471`: value `0.25`, primary VJP `0.5`, second-moment VJP `0.5`. The lane retains its gfortran second-operand tie form. |
| `Hbig_pra` / `Hsnow_pra` / `ice_var_zapneg` correction chain | `nemo_adv2d_testcase_recipe.py:414` `apply_si3_prather_source_corrections` | same file `:746` `apply_si3_prather_source_corrections_lane`, using `si3_hbig_pra`, `si3_hsnow_pra`, `si3_var_zapneg` | Main ADV2D and residual tests; lane ICE_RHEO recipe. | Uniform hard-NEMO fixture: exactly equal; contracts differ for ramps and returned residual ledgers. |
| `icemax2d_pra` nine-point maximum | `nemo_adv2d_testcase_recipe.py:390` `_max9` | same file `:403` `_si3_lane_max9` (and transport `si3_icemax2d_pra`) | Main correction chain; lane correction chain. | Not separately compared; covered by the equal uniform correction fixture. |
| `ice_cor(kn=1)` | `nemo_rheo_testcase_recipe.py:538` `_ice_cor` | `nemo_rheo_testcase_recipe_lane.py:640` `_ice_cor` | Respective main and lane ICE_RHEO compositions. | Shared synthetic card: contents and surface temperature exactly equal. |
| `icedyn` ICE_RHEO `rhg -> adv -> rdgrft -> cor` composition | `nemo_rheo_testcase_recipe.py:615` `step_ice_rheo_card` | `nemo_rheo_testcase_recipe_lane.py:697` `step_ice_rheo_card` | Main/API callers use the original recipe; six phase-2 probes plus the rung-3.4 card test import the lane recipe. | `ice_cor` agrees exactly; post-ridging difference is the `2.0208201778615376e-10` above. |

## Joint decision

The earlier five decision items—column chain, C-grid deformation, C-grid aEVP,
Prather correction, and ICE_RHEO composition—are rows in this table rather
than the full inventory.  Consolidating any row requires choosing which source
version, halo program, category contract, rounding order, and derivative tie
semantics become shared.  This repair pass makes no such choice: main callers
retain `ee0de42471` behavior and lane callers retain their lane-named paths.
