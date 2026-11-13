# SI3 PR duplicate implementations

This inventory is the grep-based review requested after the main catch-up at
`3b1d47ce6`.  A row means that the branch carries two or more executable translations
of the same NEMO routine or source block: the restored main implementation and
one or more SI3-ladder implementations.  The count is **33 routine/source-block rows**.
Closely coupled x/y sweeps and forward/inverse conversion pairs remain separate
where they have separate executable bodies.  Pure dataclasses, pack/unpack
helpers, state schemas, and wrappers with only one numerical body are excluded.

| NEMO routine or source block | Main implementation | SI3-lane implementation | Direct callers | Shared-fixture agreement |
|---|---|---|---|---|
| `icestp` / sea-ice step dispatch | `sea_ice.py:264` `step_sea_ice` | `sea_ice.py:630` `step_sea_ice_si3_lane` | Main: coupled model, SCM, main tests. Lane: `test_si3_thd_da` and the lane C1D composition. | Not compared as one output: they deliberately call different `ice_thd` rows below. |
| `ice_thd` column chain | `bitz_lipscomb.py:1538` `si3_column_step_arrays` | `bitz_lipscomb.py:3472` `si3_lane_column_step_arrays` | Main: `_si3_step_with_trace` and t4col tests. Lane: `_si3_lane_step_with_trace`, phase-2/year gates, `test_bitz_lipscomb.py` lane tests, and O9/O11/O12 tests. | C9 fixture: 22/72 common leaves equal; maximum absolute difference `9.313225746154785e10`. |
| `ice_var_salprof`, option 2 | `bitz_lipscomb.py:266` `option2_salinity_profile` | `bitz_lipscomb.py:1622` `_si3_502_option2_salinity_profile`; `si3_orca2_state.py:788` `si3_var_salprof` | Main column/card builders. Lane C1D `ice_thd` chain and ORCA2 O1/O2/O4 state paths. | Not compared: main carries its shallow-ice validation; the two lane bodies carry 5.0.2 statement order at different array boundaries. |
| `ice_var_glo2eqv`, ice enthalpy | `bitz_lipscomb.py:298` `ice_enthalpy_from_temperature` | `bitz_lipscomb.py:1672` `_si3_502_ice_enthalpy_from_temperature`; `si3_orca2_state.py:890` `si3_var_glo2eqv` | Main column/card/test setup. Lane C1D `ice_thd` chain and ORCA2 state/restart paths. | Not compared: source-order, rounding, and scalar-versus-global contracts differ. |
| `ice_var_glo2eqv`, ice temperature | `bitz_lipscomb.py:333` `ice_temperature_from_enthalpy` | `bitz_lipscomb.py:1706` `_si3_502_ice_temperature_from_enthalpy`; `si3_orca2_state.py:890` `si3_var_glo2eqv` | Main column/card/tests. Lane C1D ZDF/DH/temperature stages and ORCA2 state/restart paths. | Not compared: main keeps bounded/smoothed inverse controls; lane bodies preserve 5.0.2 source order. |
| `ice_var_glo2eqv`, snow enthalpy | `bitz_lipscomb.py:445` `snow_enthalpy_from_temperature` | `bitz_lipscomb.py:1770` `_si3_502_snow_enthalpy_from_temperature`; `si3_orca2_state.py:890` `si3_var_glo2eqv` | Main column/card setup. Lane C1D chain and ORCA2 state/restart paths. | Not compared: source-rounding and array-boundary contracts differ. |
| `ice_var_glo2eqv`, snow temperature | `bitz_lipscomb.py:453` `snow_temperature_from_enthalpy` | `bitz_lipscomb.py:1784` `_si3_502_snow_temperature_from_enthalpy`; `si3_orca2_state.py:890` `si3_var_glo2eqv` | Main column/tests. Lane C1D ZDF/DH stages and ORCA2 state/restart paths. | Not compared: main and lane carry different bound/order contracts. |
| `ice_thd_zdf_bl99`, P07 conductivity | `bitz_lipscomb.py:475` `p07_conductivity` | `bitz_lipscomb.py:1828` `_si3_502_p07_conductivity` | Main `_si3_zdf_bl99_step`; lane `_si3_zdf_bl99_core`. | Not compared: lane includes the 5.0.2 floor and statement rounding. |
| `ice_thd_zdf_bl99`, shortwave radiation | `bitz_lipscomb.py:487` `_radiation` | `bitz_lipscomb.py:1902` `_si3_502_radiation` | Main `_si3_zdf_bl99_step`; lane `_si3_zdf_bl99_core`. | Not compared: the two routines expose different SSL/source-order contracts. |
| `ice_thd_zdf_bl99`, iterative solve | `bitz_lipscomb.py:517` `_si3_zdf_bl99_step` | `bitz_lipscomb.py:1970` `_si3_zdf_bl99_core` (via `si3_ice_thd_zdf_bl99`) | Main column chain; lane category stages. | Main digest restoration: all 180 previously moved ZDF observations returned to equality with `ee0de42471`. |
| `snw_ent` / `ice_var_vremap` cumulative remap | `bitz_lipscomb.py:817` `_nemo_snow_enthalpy_remap` and main DH body | `bitz_lipscomb.py:2397` `_cumulative_remap` in `_si3_ice_thd_dh_core` | Main column chain; lane `si3_ice_thd_dh`. | Not compared: main includes its reverse-safe/floor fixes; lane preserves 5.0.2 order. |
| `ice_thd_dh` | `bitz_lipscomb.py:877` `_dh_step_fluxes` | `bitz_lipscomb.py:2444` `_si3_ice_thd_dh_core` via `si3_ice_thd_dh` | Main column chain and t4col DH tests. Lane column chain, phase-2 ablation gate, and O9 tests. | Not compared end to end: the row above isolates only their remap sub-block. |
| `ice_thd_sal` | `bitz_lipscomb.py:1399` `_sal_step` | `bitz_lipscomb.py:3122` `si3_ice_thd_sal` / `_si3_ice_thd_sal4_core` | Main column chain and t4col salinity tests. Lane column chain, phase-2 ablation gate, and O10 tests. | Not compared: main is the ORCA1 option-2/smoothed-gate arm; lane is the ORCA2 salinity-4 drainage/flushing arm. |
| `ice_thd_do` | `bitz_lipscomb.py:1444` `_do_step` | `bitz_lipscomb.py:3240` `si3_ice_thd_do` | Main column chain; lane column and ORCA2 stages. | Not compared: main is the resolved jpl=1/5.0.1 arm; lane is the general 5.0.2 arm. |
| `ice_alb` | `sea_ice.py:882` `_nemo_si3_ice_albedo` | `si3_surface.py:300` `si3_ice_alb` | Main ORCA1/C1D bulk-flux path and its oracle tests. Lane O7 surface step, O16 finalization, and `nemo_si3_o7_lane_bulk_flux_gate.py`. | Not compared on one fixture: main is the scalar no-pond identity; lane includes category, pond, and lid arms. |
| `sbc_ice` Goff saturation over ice | `thermo.py:689` `nemo_si3_saturation_over_ice` | `thermo.py:656` `nemo_si3_e_sat_ice`, `thermo.py:669` `nemo_si3_q_sat_ice`, and `thermo.py:676` `nemo_si3_dq_sat_dt_ice` | Main: `core.bulk_flux.nemo_si3_constant_fluxes`. Lane: `core.bulk_flux.nemo_si3_ice_turbulent` and `si3_surface.si3_blk_ice_1`. | Not compared as interchangeable APIs: main preserves its configurable smooth switch and current constants; the lane preserves the source-rounded hard NEMO statement order. |
| `blk_ice_1` constant-coefficient turbulent fluxes | `core/bulk_flux.py:2414` `nemo_si3_constant_fluxes` | `core/bulk_flux.py:2362` `nemo_si3_blk_ice_1_cst` and `core/bulk_flux.py:2394` `nemo_si3_ice_turbulent` | Main ORCA1/C1D surface code calls the smooth-capable public helper. Lane O7 and `nemo_si3_o7_lane_bulk_flux_gate.py` call the source-literal helpers through `si3_surface`. | Not compared as one output contract: main exposes a scalar configurable switch; the lane exposes the ORCA2 array/category statement sequence. |
| `blk_ice_2` | `sea_ice.py:948` `_nemo_si3_blk_ice_2` | `si3_surface.py:384` `si3_blk_ice_2` | Main ORCA1/C1D bulk-flux path and slot/namelist/mask tests. Lane O7 surface step and `nemo_si3_o7_lane_bulk_flux_gate.py`. | Not compared on one fixture: the array and selector contracts differ. |
| `ice_flx_other` | `sea_ice.py:1128` `_nemo_si3_ice_flx_other` | `si3_surface.py:467` `si3_ice_flx_other` | Main C1D surface path and qlead/mask tests. Lane O7 surface step and `nemo_si3_o7_lane_bulk_flux_gate.py`. | Not compared: main is the active C1D arm; lane carries the ORCA2 neighbour and category program. |
| `icedyn_rhg_evp`, final deformation | `dynamics.py:1249` `si3_cgrid_deformation` | `dynamics.py:1326` `si3_lane_cgrid_deformation` | Main ICE_RHEO/rung-3.3/trajectory paths; lane ICE_RHEO recipe and its six probes. | Periodic fixture: 2/2 arrays exactly equal; fold programs differ by contract. |
| `icedyn_rhg_evp`, aEVP solver | `dynamics.py:1374` `si3_cgrid_aevp_solver` | `dynamics.py:2007` `_si3_orca2_cgrid_aevp_core` via `si3_orca2_aevp_solver` | Main ORCA1, ICE_RHEO, landfast, rung-3.3; lane ORCA2 O5 wrappers. | Two-subcycle periodic fixture: 5/5 arrays exactly equal. |
| `ice_var_strength` H79 arm | `dynamics.py:1427` inline H79 strength in `si3_cgrid_aevp_solver` | `dynamics.py:2764` `si3_orca2_ice_strength` | Main ORCA1 aEVP; lane ORCA2 O5 dynamics. | Not compared: grep finds no current `_si3_ice_strength` definition; the restored main body is inline and the lane body adds smoothing/fold handling. |
| `ice_dyn_rdgrft` | `ridging.py:590` `apply_si3_jpl1_ridging` | `ridging.py:1300` `si3_ice_dyn_rdgrft` | Main JPL1 tests and main ICE_RHEO recipe; lane `ridge_ice_rheo` and ORCA2 O6. | Shared ICE_RHEO card: post-ridging maximum absolute difference `2.0208201778615376e-10`. |
| `ice_itd_rem` | `itd.py:641` `lipscomb_2001_remap` | `itd.py:823` `si3_ice_itd_rem` | Main multicategory thermodynamic remap and main ITD tests. Lane `step_sea_ice_si3_lane` exact O13 remap and O13 tests. | Not compared: main exposes a differentiable generic Lipscomb remap; lane carries NEMO 5.0.2 global arrays, ponds/lids, salinity and flux ledgers. |
| `ice_itd_reb` | No `rebin_categories` symbol exists at `3b1d47ce6`, `ee0de42471`, or in reachable history. | `itd.py:1213-1291` `_si3_itd_shiftice` and `si3_ice_itd_reb`; `si3_finalize.py:265-438` roundoff and transfer helpers and `:512-599` the untraced body of `_ice_itd_reb` | First lane body: post-dynamics `si3_ice_cor_dyn`, and post-thermodynamics `si3_ice_cor` when traced (`_ice_itd_reb` dispatches to it at `si3_finalize.py:455-511`). Second lane body: post-thermodynamics `si3_ice_cor` when not traced. | Not compared. The review's proposed main symbol was checked rather than inferred; the duplicate is between the two lane bodies. |
| `ice_cor` salinity bounds | Inline in `si3_ice_cor.py:196-216` `apply_si3_jpl1_ice_cor` | `itd.py:1352-1380` `si3_ice_cor_salinity_bounds`; `si3_finalize.py:671-759` traced dispatch and `:760-786` untraced body in `si3_ice_cor` | Main JPL1 `ice_cor`. Lane post-dynamics `si3_ice_cor_dyn` calls the `itd.py` body; lane O15 finalization selects the same static-trip body when traced and carries the statement-ordered loop otherwise. | Not compared as one contract: main admits bulk or layered salt and smoothing; the two lane paths preserve the layered 5.0.2 bounds and statement order. |
| `ice_var_zapsmall` | Inline in `si3_ice_cor.py:218` `apply_si3_jpl1_ice_cor` | `itd.py:1294-1349` `si3_var_zapsmall`; `transport.py:3334` `si3_var_zapsmall` | Main JPL1 `ice_cor`. Lane post-dynamics `si3_ice_cor_dyn` uses the `itd.py` body; lane O4 callers use the transport body; lane O15 finalization (`si3_ice_cor`) uses the `itd.py` body when traced and the transport body otherwise. | Not compared: three separately executable bodies implement different admitted category/salinity/pond contracts. |
| `icedyn_adv_pra`, x/y sweeps | `transport.py:2091` `advect_si3_prather_2d` | `transport.py:2349` `si3_adv_x` plus `si3_adv_y` | Main ADV2D/ICE_RHEO/rung-3.3; lane ORCA2 O4 state step. | Main wrappers share most kernels, but lane selects strict loop domains; no single equivalent end-to-end fixture. |
| `icedyn_adv_pra`, x/y limiter tie order | `transport.py:615,648` `_si3_prather_limit_x/_y` | `transport.py:681,715` `si3_lane_prather_limit_x/_y` | Main 1-D/2-D Prather substeps; lane `si3_adv_x/si3_adv_y` through strict loop domains. | At the exact binary tie `1.25 - 1.0 = 0.25`, main x/y forward bits and VJPs are bit-identical to `ee0de42471`: value `0.25`, primary VJP `0.5`, second-moment VJP `0.5`. The lane retains its gfortran second-operand tie form. |
| `Hbig_pra` / `Hsnow_pra` / `ice_var_zapneg` correction chain | `nemo_adv2d_testcase_recipe.py:414` `apply_si3_prather_source_corrections` | same file `:746` `apply_si3_prather_source_corrections_lane`, using `si3_hbig_pra`, `si3_hsnow_pra`, `si3_var_zapneg` | Main ADV2D and residual tests; lane ICE_RHEO recipe. | Uniform hard-NEMO fixture: exactly equal; contracts differ for ramps and returned residual ledgers. |
| `icemax2d_pra` nine-point maximum | `nemo_adv2d_testcase_recipe.py:390` `_max9` | same file `:403` `_si3_lane_max9` (and transport `si3_icemax2d_pra`) | Main correction chain; lane correction chain. | Not separately compared; covered by the equal uniform correction fixture. |
| `ice_cor(kn=1)` | `si3_ice_cor.py:144` `apply_si3_jpl1_ice_cor`, via `nemo_rheo_testcase_recipe.py:539` `_ice_cor` | `itd.py:1311` `si3_ice_cor_dyn`, via `nemo_rheo_testcase_recipe_lane.py:641` `_ice_cor` | Respective main and lane ICE_RHEO compositions; the lane body also serves ORCA2 O6. | Shared synthetic card: contents and surface temperature exactly equal. |
| `icedyn` ICE_RHEO `rhg -> adv -> rdgrft -> cor` composition | `nemo_rheo_testcase_recipe.py:616` `step_ice_rheo_card` | `nemo_rheo_testcase_recipe_lane.py:698` `step_ice_rheo_card` | Main/API callers use the original recipe; six phase-2 probes plus the rung-3.4 card test import the lane recipe. | `ice_cor` agrees exactly; post-ridging difference is the `2.0208201778615376e-10` above. |

### Systematic grep closure

The audit enumerated every `si3_*`/`_si3_*` definition under
`packages/ice/legoesm/ice`, extracted the NEMO routine named by its docstring,
and searched the main-side files for that routine and for equivalent callers.
The rows above include every hit with a second executable numerical body.

Excluded after inspection, rather than by name alone:

- `si3_blk_ice_1` and `si3_eos_fzp` are wrappers around the one shared
  `nemo_si3_blk_ice_1_cst` and `nemo_eos_fzp` numerical bodies.
- `ice_thd_da`, `ice_thd_pnd`/`pnd_LEV`/`ice_perm_eff`, `ice_update_flx`,
  `ice_update_tau`, `ice_var_agg`, and `ice_var_snwfra_3d` have no second
  translation of the same NEMO routine on this branch.  Generic main pond,
  aggregation, snow-cover, response, and stress code is not labelled as the
  same source routine merely because it serves related physics.
- State schemas, halo/restart bridges, pack/unpack helpers, configuration
  builders, and rung-composition wrappers remain excluded because they do not
  contain a second numerical body.

## Joint decision

The earlier five decision items—column chain, C-grid deformation, C-grid aEVP,
Prather correction, and ICE_RHEO composition—are rows in this table rather
than the full inventory.  Consolidating any row requires choosing which source
version, halo program, category contract, rounding order, and derivative tie
semantics become shared.  This repair pass makes no such choice: main callers
retain `ee0de42471` behavior and lane callers retain their lane-named paths.
