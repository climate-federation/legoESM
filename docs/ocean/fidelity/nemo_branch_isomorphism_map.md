# NEMO -> legoESM branch-isomorphism map (one ocean time step)

READ-ONLY audit. Checkout `/tmp/l1-overflow-ro` @ `28d166428`
(tip of `fidelity/nemo-testcases-l1-codex`). GYRE card read from
`origin/fidelity/nemo-testcases-l2-gyre-codex`. ORCA1 card read from
`scripts/run/run_omip_core2.py` + `scripts/cluster/omip_nemo/run_standard_faithful_1deg.sbatch`
on this checkout. NEMO source `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE`.

No tracked file was modified. `/tmp/codex-testcases` (r3u mean-vs-min) NOT touched;
that question is IN FLIGHT and is row R-QCO-1 below.

## 0. Resolved selectors per card (measured, not read off comments)

Resolved by constructing each card and printing the config
(`dino_config_for_recipe('nemo_dino_kamm_mlf')` -> `dino_lat_lon_model_config`;
`build_nemo_testcase_card(...)`). GYRE read statically from the l2 branch
source (`_model_config(whole_step_identity="gyre_vector_ene_c2")` ->
`nemo_lat_lon_model_config(NEMOModelRecipeConfig(...))`). ORCA1 = CLI flags on
the sbatch + `LatLonCGridOceanConfig` defaults for everything not passed.

| selector | DINO kamm_mlf | LOCK_EXCHANGE-zco | OVERFLOW-zps | GYRE | ORCA1 OMIP |
|---|---|---|---|---|---|
| outer_integrator | leapfrog | forward_euler | forward_euler | forward_euler | forward_euler |
| momentum_time_integrator | euler | rk3_ws | rk3_ws | rk3_ws | **rk3** (SSP) |
| tracer_time_integrator | euler | rk3_ws | rk3_ws | rk3_ws | **euler** |
| eos | nemo_seos | nemo_teos10 | nemo_teos10 | nemo_teos10 | **wright** (default) |
| eos_depth | geometric | geometric | geometric | geometric | **insitu** (default) |
| tracer_advection | fct2 | fct2 | fct2 | fct2 | **superbee** |
| momentum_advection | vector_invariant | flux_form | flux_form | vector_invariant | vector_invariant |
| momentum_flux_scheme | upwind (inert) | upwind3 | upwind3 | n/a | upwind (inert) |
| vertical_momentum_scheme | nemo_advective | nemo_up3 | nemo_up3 | nemo_advective | **upwind_perturbation** |
| vorticity_scheme | een_total | **al81** | **al81** | ene_total | **al81** |
| coriolis_scheme | explicit_ab2 | **matsuno_split** | **matsuno_split** | explicit_ab2 | **matsuno_split** |
| pgf_scheme | nemo_sco | nemo_sco | nemo_sco | nemo_sco | **smc03** |
| pgf_quadrature | nemo_trapezoid | nemo_trapezoid | nemo_trapezoid | nemo_trapezoid | **cell_integral** |
| lateral_viscosity_operator | nemo_div_curl | vector_laplacian (A_h=0) | vector_laplacian (A_h=0) | nemo_div_curl | **vector_laplacian** |
| lateral_viscosity_e3_weighting | **off** | off (inert) | off (inert) | nemo_e3 | off |
| adaptive_implicit_vertadv | False | True | True | False | True |
| zdf_implicit_solver_evaluation | nemo_literal | nemo_literal | nemo_literal | nemo_literal | **shared_thomas** |
| implicit_vmix_e3t_now_divisor | **False** | True | True | True | **False** |
| zad_qco_evaluation | nemo_literal | **generic** | **generic** | nemo_literal | **generic** |
| wzv_call2_evaluation | nemo_literal | **generic** | **generic** | nemo_literal | **generic** |
| barotropic_solver | explicit_substep | explicit_substep | explicit_substep | explicit_substep | explicit_substep |
| barotropic_time_filter | nemo_boxcar_ab3 | nemo_ab3am4 | nemo_boxcar1_ab3 | nemo_ab3am4 | **cosine** |
| barotropic_coriolis | een_metric | **avg** | **avg** | ene_metric | **avg** |
| barotropic_coriolis_split | live | **frozen** | **frozen** | live | **frozen** |
| barotropic_een_seed | nemo_kmm | **window_start** | **window_start** | nemo_kmm | window_start |
| barotropic_after_reconcile | nemo_mlf_baro_corr | off | off | off | off |
| nemo_stage_mean_imposition | False (n/a MLF) | **False** | **False** | True | False |
| bbl_adv_option | 0 | 0 | 2 | 0 | 2 (driver-side) |
| tracer_wall_neumann_fill | True | True | True | True | True |
| fix_eta_drift | **True** | False | False | False | True (default) |

Bold = the card is NOT on the NEMO-transcribed arm for that row.

## 1. Routine -> implementation map

Cards: `D`=DINO kamm_mlf (MLF), `L`=LOCK_EXCHANGE-zco, `O`=OVERFLOW-zps,
`G`=GYRE, `A`=ORCA1 OMIP. All legoESM paths are under
`packages/ocean/legoesm/ocean/`; `omlc` = `dynamics/ocean_model_latlon_cgrid.py`,
`opl` = `dynamics/ocean_pe_latlon_cgrid.py`, `blc` = `dynamics/barotropic_latlon_cgrid.py`.

| # | NEMO routine (file:line) | NEMO switch | cards | legoESM impl(s) (file:line) | selector | disposition |
|---|---|---|---|---|---|---|
| S-01 | `sbc` stpmlf.F90:170 / stprk3.F90:138 | `ln_usr` / bulk | D G A | `ocean/forcing/*`, `physics/surface_forcing/*`; DINO `experiments/dino.py:dino_step_surface_forcing`; GYRE `fidelity/nemo_recipe.py:restoring_surface_forcing`; ORCA1 `bulk_flux_omip.py` | per-card forcing builder | NEMO_SWITCH (analytic usr vs bulk) — L/O have no SBC |
| S-02 | `eos_rab` eosbn2.F90:490+ | `ln_teos10`/`ln_seos` | D G A(-) | `eos.py:1134 nemo_roquet_alpha_beta`; `eos.py:540 nemo_seos_alpha_beta`; `eos.py:235 eos_density_derivatives` | `eos` | NEMO_SWITCH (teos10/seos); ORCA1 `wright` = **ARTIFICIAL** (no NEMO arm) |
| S-03 | `bn2` eosbn2.F90:253-288 | same as eos | D G A | `eos.py:624 compute_buoyancy_frequency_nemo_bn2`; `eos.py:2627 compute_buoyancy_frequency`; `eos.py:2667 ..._adiabatic` | `tke.n2_mode` (`nemo_bn2`/`adiabatic`), `n2_eos_form` | ARTIFICIAL_BRANCH — 3 impls, one NEMO routine; DINO/GYRE on `nemo_bn2`, ORCA1 arm UNVERIFIED. Not reached by L/O (`ln_zdfcst`, no TKE/EVD) |
| S-04 | `bn2` live-e3w geometry eosbn2.F90:253-258 | none | D G | `eos.py:754 nemo_bn2_live_geometry`, `:775 nemo_e3w_from_live_gdept`, `:892 nemo_bn2_live_ladders`, `:590 _nemo_bn2_zrw` | `zrw_evaluation`, `tke_n2_evaluation_stage` | ARTIFICIAL_BRANCH (generic vs literal ladder; NEMO has one) |
| S-05 | `zdf_phy`->`zdf_cst` zdfphy.F90 | `ln_zdfcst` | L O | constant `A_v`/`K_v` in `physics/vertical_mixing/constant.py` + config `A_v=1e-4,K_v=0` | `physics=None` + `A_v`/`K_v` | SHARED |
| S-06 | `zdf_phy`->`zdf_tke` zdftke.F90 | `ln_zdftke` | D G A | `physics/vertical_mixing/tke.py` (one module) | `vertical_mixing.tke` | SHARED operator, but see S-07 |
| S-07 | `zdf_tke` internals (matrix/solver/mxl/etau/htau/Langmuir/shear/N2 stage) | **none** (one NEMO program) | D G A | `tke.py` (one module, ~10 two-arm switches) | `tke_matrix_evaluation`, `tke_solver_evaluation`, `tke_mxl_raw_evaluation`, `tke_etau_exponential_evaluation`, `tke_htau_evaluation`, `tke_langmuir_evaluation`, `tke_shear_evaluation_stage`, `tke_shear_metric_source`, `tke_n2_evaluation_stage`, `tke_preclosure_coeff_source` (`config.py:357,586-624`) | ARTIFICIAL_BRANCH x10 — defaults are the non-NEMO arm; ORCA1 on defaults |
| S-08 | `zdf_evd` zdfevd.F90:93-120 | `ln_zdfevd`,`nn_evdm` | D G | `physics/convection/enhanced_diffusion.py` | `convection.scheme="enhanced_diffusion"` + `two_level_trigger`,`convection_evd_n2_time_level` | NEMO_SWITCH for `nn_evdm`; `convection_evd_n2_time_level` = ARTIFICIAL (no NEMO arm) |
| S-09 | `ldf_slp` ldfslp.F90 | `ln_traldf_iso` | D G | `physics/lateral_mixing/gm_redi_latlon_cgrid.py` (one module, ~12 `*_evaluation` switches) | `gm_redi_slope_*`, `gm_redi_a33_evaluation`, `gm_redi_horizontal_evaluation`, … | ARTIFICIAL_BRANCH x12 — one NEMO routine, two arms each |
| S-10 | `ldf_tra`/`ldf_eiv` ldftra.F90:290,332 | `nn_aht_ijk_t`,`nn_aei_ijk_t` | D G A | `physics/lateral_mixing/config.py` Treguier/Visbeck + static kappa (`omlc:1388 static_kappa_redi_override`) | `gm_kappa_scheme` | NEMO_SWITCH (nn_aei_ijk_t=21 <-> treguier); Visbeck = ARTIFICIAL (no NEMO arm) |
| S-11 | `ldf_dyn` ldfdyn.F90 | `nn_ahm_ijk_t=0` | D L O G A | scalar `A_h` in config (constant) | `A_h`,`A_h_lat_scaling` | SHARED |
| S-12 | `stp_2D` stp2d.F90:49-288 (RK3 pre-step: eos+hpg+ldf+vor+wzv+adv @Kbb, `dyn_drg_init`, `dyn_spg_ts`) | key_RK3 | L O G | `omlc:3950 self.tendencies(...)`, its depth mean in `F_slow_u/F_slow_v`, then the BEFORE velocity `omlc:4448 u_star = u0` into the barotropic solve | implicit in `_step_impl` ordering | ~~ARTIFICIAL_BRANCH~~ **SHARED — COLLAPSED 2026-09-02**, see the final section. The pre-solve WS ladder is deleted; the solver is now seeded exactly as `stp2d.F90:177-186,280-281` seeds `dyn_spg_ts` (single Kbb RHS depth mean + `uu(Kbb)`) |
| S-13 | `dyn_spg_ts` external loop dynspg_ts.F90:~560-800 | `ln_dynspg_ts` | D L O G A | `blc:1873 barotropic_substeps_latlon_cgrid` (+`blc:1135 _run_substep_loop`); alternates `barotropic_implicit_latlon_cgrid`, `rigid_lid_latlon_cgrid` | `barotropic_solver` | SHARED for all 5 (all `explicit_substep`); the other two solvers are non-NEMO oracles (Veros/MITgcm) |
| S-14 | `dyn_spg_ts` external velocity update dynspg_ts.F90:719-763 | `ln_dynadv_vec .OR. lk_linssh` (vector vs flux form) | L O (flux) / D G A (vector) | `blc:1105 _nemo_flux_form_external_velocity_update` (flux) ; `blc:1464-1512` inline vector update | gate `blc:1856 nemo_flux_form_update_active` = `rk3_ws AND flux_form` | NEMO_SWITCH on the vector/flux axis, but the gate carries an EXTRA `momentum_time_integrator=="rk3_ws"` condition NEMO does not have -> **ARTIFICIAL** conjunct (latent: an MLF+flux-form card would silently get the vector update) |
| S-15 | `dyn_spg_ts` backward face depth `zhu_bck` dynspg_ts.F90:738-747 | `key_qcoTest_FluxForm` (simple avg) vs default (e1e2t-weighted avg) | D L O G A | `blc:451 nemo_ssh_avg_face_depth` (+`_nemo_ssh_avg_prep/_apply`); `blc:581 _min_rule_face_depths` | `barotropic_face_depth` (`nemo_ssh_avg`/`min_rule`) | NEMO_SWITCH; on a lat-lon C-grid `e1e2t(i,j)==e1e2t(i+1,j)` so the two NEMO arms coincide (CONFIRMED algebraically, not measured). `min_rule` = ARTIFICIAL (no NEMO arm), ORCA1 on it (UNVERIFIED which arm ORCA1 resolves — `barotropic_face_depth` default is `min_rule`) |
| S-16 | `dyn_spg_ts` continuity / transport accumulation / spg dynspg_ts.F90:~640-700,~840 | none | D L O G | `blc:494 nemo_literal_metric_transports`, `:512 nemo_literal_accumulate_transport`, `:548 nemo_literal_continuity_divergence`, `:224 _nemo_literal_barotropic_pressure_gradient`, `:147 _nemo_literal_seed_from_reference_mesh` vs the generic inline arms in `_run_substep_loop` | `barotropic_continuity_evaluation`, `barotropic_transport_accumulation_evaluation`, `barotropic_pgf_evaluation`, `barotropic_seed_evaluation`, `barotropic_seed_face_depth` | ARTIFICIAL_BRANCH x5 — NEMO has one program; defaults are the generic arm; ORCA1 on defaults |
| S-17 | `dyn_cor_2D` (in-substep barotropic Coriolis) dynspg_ts.F90:359,689 | `ln_dynvor_ene/ens/een` | D G | `blc:944 een_barotropic_coriolis`, `:1058 barotropic_coriolis_een_pre_step`, `:731 _nemo_literal_een_coefficients`, `:840 _build_een_barotropic_inputs`; generic 4-pt `f_u*V_at_u` at `blc:1441,1494` | `barotropic_coriolis` (`avg`/`een`/`een_metric`/`ene_metric`), `barotropic_coriolis_split` (`frozen`/`live`), `barotropic_een_seed`, `barotropic_een_coefficient_evaluation` | NEMO_SWITCH on ene/een; `avg` + `frozen` = ARTIFICIAL (no NEMO arm) and is what L/O/A resolve to |
| S-18 | `dom_qco_r3c` / `dom_qco_r3c_RK3` domqco.F90:140-186, :189-240 (r3u = **surface-weighted MEAN** of ssh) | `key_qco`; `key_qcoTest_FluxForm` sub-arm | D L O G A | ~~(a) `dynamics/latlon_cgrid_operators.py:277 min_cell_to_uface`/`:312 min_cell_to_vface` — MIN of live thickness, used by `omlc:1063 _nemo_ws_stage_transport`~~ **CORRECTED 2026-09-02 (post-1d6a7448d) — see the "Triage against HEAD 648e5cd69" section below.** `_nemo_ws_stage_transport`'s stage-transport builder (`_nemo_ws_qco_stage_faces`) does NOT call `min_cell_to_uface` for this quantity (that citation conflated this row with S-19's `zad_qco_evaluation` gate, which lives in a different function, `nemo_qco_wzv_operands`); it calls the shared kernel via `vertical.py:218 nemo_qco_live_face_geometry_cgrid` (`:140 nemo_qco_live_face_geometry_from_operands` underneath), same as the MLF tracer transport and GM/Redi. `min_cell_to_uface`'s citation at `omlc:1379-1384` is a genuinely different routine (Matsuno-split Coriolis depth-average, S-27) | `zad_qco_evaluation`/`wzv_call2_evaluation`/`gm_redi_*_face_thickness_evaluation` gate the SEPARATE S-19/S-43 consumers of the shared kernel, not this row | **OTHER_RECIPE** (was `ARTIFICIAL_BRANCH (rank 1)`) — one NEMO kernel (`nemo_qco_live_face_geometry_cgrid`/`_from_operands`) reached by every card, beside `min_cell_to_uface`'s real, separately-cited MOM6/MITgcm hFacW=min role (PE-lane depth-average/slow-forcing, `opl:1364`/`omlc:4125`), reached by DINO/LOCK/OVERFLOW/ORCA1 **and** by `veros_faithful_v1`/`mitgcm_v1`/`oceananigans_v1` (which structurally cannot reach the NEMO arm — no raw NEMO mesh operands) |
| S-19 | `wzv` sshwzv.F90 (np_velocity / np_transport) | none (arg-level) | D L O G A | `vertical.py diagnose_w_from_flux_div` (generic); `opl nemo_qco_wzv_operands` (literal, incl. `nemo_qco_kmm_velocity_cycle`) | `wzv_call2_evaluation` (`generic`/`nemo_literal`) | ARTIFICIAL_BRANCH — one NEMO routine, two impls; D/G literal, L/O/A generic. **MEASURED 2026-09-02** — the branch is arithmetic association only: stage `w` differs 3.0e-18 m/s (OVERFLOW) / 2.4e-21 m/s (LOCK), every kt=1..10 row BIT-IDENTICAL, and it does NOT own OVERFLOW's 2.599e-7 kt=2 velocity debt. Operand wall REMOVED (`nemo_qco_resolved_mesh_operands`); the remaining wall is NEMO's own call-site split. See "S-19 arm result" below |
| S-20 | `wAimp` sshwzv.F90 (adaptive-implicit w split) | `ln_zad_Aimp` | L O A | `vertical.py:52 nemo_wicker_aimp_partition_transport` (called at `omlc:1085` per stage) | `adaptive_implicit_vertadv` | SHARED (one impl). ORCA1 consumes it at a different site (`omlc:5300+` post-program) — composition differs |
| S-21 | stage transport `zFu/zFv` + `zub` correction stprk3_stg.F90:257-277 | none (RK3 identity) | L O G | `omlc:1046 _nemo_ws_stage_transport` | none (private, `_NEMOWSRK3TestHooks` only) | SHARED across the three RK3 cards |
| S-22 | `dyn_adv` dispatch dynadv.F90:87-89 | `ln_dynadv_vec` / `ln_dynadv_cen2` / `ln_dynadv_up3` | D G A (vec) / L O (up3) | `opl` flux-form UP3 path (`momentum_flux_scheme="upwind3"`) vs vector-invariant path (`opl:2203+`) | `momentum_advection`, `momentum_flux_scheme` | NEMO_SWITCH |
| S-23 | `dyn_adv_up3` vertical flux dynadv_up3.F90:239-365 | inside `ln_dynadv_up3` | L O | `vertical.py:1556 nemo_up3_vertical_momentum_advection`, called per stage at `omlc:4781 _stage_vertical_up3` AND once post-program at `omlc:5310+` (gated `momentum_time_integrator != "rk3_ws"`) | `vertical_momentum_scheme="nemo_up3"` | SHARED arithmetic, two call sites gated mutually exclusive; composition differs from NEMO only on the non-rk3_ws path |
| S-24 | `dyn_zad` dynzad.F90:86-119 | `ln_dynadv_vec` | D G | `vertical.py:1632 nemo_advective_vertical_momentum_advection` | `vertical_momentum_scheme="nemo_advective"`, `zad_bottom_face_mask`, `zad_qco_evaluation` | SHARED impl; `zad_bottom_face_mask` (`min_rule`/`nemo_faithful`) = ARTIFICIAL (no NEMO arm) |
| S-25 | vertical momentum advection, non-NEMO arms | — | A | `vertical.py:1403 flux_form_vertical_momentum_advection` (upwind_perturbation), `:1476 ..._centered` | `vertical_momentum_scheme` = `upwind_perturbation`/`centered_full` | ARTIFICIAL_BRANCH — ORCA1 runs `upwind_perturbation`, which corresponds to no NEMO arm |
| S-26 | `dyn_vor` dynvor.F90 (`vor_ene`/`vor_ens`/`vor_een`) | `ln_dynvor_ene`/`_ens`/`_een` | D(een) G(ene) / L O(**ens**) A | `opl:2222-2520` — `al81`, `ene`, `ene_total`, `een_total` | `vorticity_scheme` (+`een_e3f_scheme`,`een_q_boundary`,`een_metric_weighting`) | **`vor_ens` is ABSENT.** L/O namelists set `ln_dynvor_ens=.true.`; the cards resolve `al81` and `nemo_testcase_recipe.py:validate_nemo_testcase_card` requires `f==0` and one wet row so the operator is structurally inert. `al81` reached by ORCA1 = ARTIFICIAL |
| S-27 | Coriolis time placement (NEMO: always inside `dyn_vor` RHS) | none | D G (rhs) / L O A (**split**) | `omlc:1246 _forward_backward_coriolis_3d` (Matsuno rotation sub-step) vs in-RHS `f x u` | `coriolis_scheme` (`matsuno_split`/`explicit_ab2`) | ARTIFICIAL_BRANCH — `matsuno_split` has no NEMO arm; inert on L/O (f=0) but LIVE on ORCA1 |
| S-28 | `dyn_hpg`->`hpg_sco` dynhpg.F90:117-123 dispatch, :340-390 body | `ln_hpg_sco` (vs `ln_hpg_zco`/`_zps`/`_djc`/`_isf`) | D L O G | `opl:1846-1936` `pgf_scheme="nemo_sco"` branch | `pgf_scheme`, `pgf_quadrature` | NEMO_SWITCH for `nemo_sco`; `hpg_zps`/`hpg_zco`/`hpg_djc` = **ABSENT** |
| S-29 | `dyn_hpg` on ORCA1 (NEMO ORCA1 uses `ln_hpg_zps`) | `ln_hpg_zps` | A | `opl:1828 pgf_scheme=="smc03"` (`dynamics/pgf_smc03.py`), `adcroft` branch, `dynamics/pgf_ahh08.py` | `pgf_scheme` (CLI `choices=[adcroft,smc03]` — `nemo_sco` UNREACHABLE from `run_omip_core2.py:4570`) | **ARTIFICIAL_BRANCH (rank 2)** — 3 non-NEMO PGF impls; the ORCA1 driver cannot even select the NEMO one |
| S-30 | stage momentum time-stepping (qco thickness-weighted) stprk3_stg.F90:344-374 | `key_qco` | L O G | `omlc:4817-5025` — ONE ladder, after the barotropic solve, corrected, overwrites `state_new.u/v` | none — it runs for every `momentum_time_integrator="rk3_ws"` step; `stage_barotropic_correction` (default True) now ablates only the per-stage external-mode replacement inside it | ~~ARTIFICIAL_BRANCH (rank 3)~~ **SHARED — COLLAPSED 2026-09-02**, see the final section. The second copy of the recurrence is deleted |
| S-31 | stage 2/3 `eos(Kmm)+dyn_hpg(Kmm)` operands stprk3_stg.F90:317-320 | none | L O G | `omlc:4808 _stage_hpg_operands` -> `_mom_pert_ws(stage_tracers_eta=...)` -> `compute_frozen_geom_density` | private hooks `freeze_stage_hpg_operands`/`_tracers`/`_eta` only | SHARED (today's fix #2) |
| S-32 | `dyn_ldf`->`ldf_lap` dynldf_lap_blp.F90 | `ln_dynldf_lap` + `ln_dynldf_lev`/`_hor` | D G / L O (A_h=0) / A | `opl` `vector_laplacian`, `flux_divergence`, `nemo_div_curl` (+ `nemo_ldf_lap_viscosity_e3_cgrid`) | `lateral_viscosity_operator`, `lateral_viscosity_e3_weighting` | ARTIFICIAL_BRANCH — 3 operator forms for one NEMO arm; `flux_divergence` is Veros's; ORCA1 on `vector_laplacian`. `lateral_viscosity_e3_weighting="off"` (DINO) is a non-NEMO simplification of the SAME operator GYRE runs with `nemo_e3` |
| S-33 | `dyn_zdf` dynzdf.F90 (implicit vertical friction + time integration) | `ln_zdfimp` (always) | D L O G A | `omlc:7537 _apply_implicit_vertical_mixing` (one function) | `zdf_implicit_solver_evaluation` (`shared_thomas`/`nemo_literal`) | ARTIFICIAL_BRANCH — one NEMO program, two solver evaluations; ORCA1 on `shared_thomas` |
| S-34 | `trazdf`/`dynzdf` gradient divisor `e3w(Kmm)` trazdf.F90:219-220 | none | L O G (true) / D A (**false**) | `omlc:7928` `_dzw_slot` / `implicit_vmix_e3t_now_divisor` branch inside `_apply_implicit_vertical_mixing` | `implicit_vmix_e3t_now_divisor`, `implicit_vmix_dzw_slot` (Veros) | **ARTIFICIAL_BRANCH (rank 4)** — the NEMO divisor is opt-in and OFF by default; the certified MLF card (DINO) is on the non-NEMO midpoint divisor (`state.py:2596` even says "NOT wired into any recipe/kamm card") |
| S-35 | stage-3 barotropic correction `zub` stprk3_stg.F90:433-446 (AFTER `dyn_zdf` at :430) | none | L O G | (a) `omlc:4707 _replace_stage_mean` — applied per stage BEFORE the implicit solve; (b) `omlc:6234 _impose_mean` + `_fixed_depth_means` — re-imposed AFTER the implicit solve | (b) gated by `barotropic.nemo_stage_mean_imposition` | ARTIFICIAL_BRANCH — two sites for one NEMO step; GYRE has both, L/O only (a), so L/O apply the correction on the wrong side of `dyn_zdf` |
| S-36 | `tra_adv_trp` stprk3_stg.F90:463,494 (reuse the momentum zF triplet) | none | L O G | `omlc:5226` reuses `_nemo_ws_live_stage_geometry`; legacy rebuild at `omlc:5240` | private hook `kmm_tracer_transports` | SHARED |
| S-37 | `tra_adv` dispatch traadv.F90:355-364; FCT at stage 3 only (`ll_dofct`, :279-283) | `ln_traadv_fct`, `nn_fct_h/v=2` | D L O G / A | `omlc:1093 _nemo_ws_rk3_tracer_pair_step` (stages 0-1 -> `"centered"`, stage 2 -> `fct2`); `omlc:709 compute_advection_flux_div_pair`; `advection.py:828 fct_tracer_advection` | `tracer_advection`, stage keying is hard-wired (no selector) | SHARED for D/L/O/G. ORCA1 runs `superbee` = **ARTIFICIAL** (`traadv_mus`/`ubs`/`qck` are ABSENT; superbee is Veros's) |
| S-38 | `tra_adv_fct` implicit-w treatment traadv_fct.F90:143,439-453 (`nn_fct_imp=1` -> `ll_zAimp1`) | `nn_fct_imp` | L O G | `advection.py:925-1010` `low_order_predictor` (`one_step`/`nemo_rk3_two_step`) + `fct_implicit_w` | `fct_low_order_predictor` (private hook `two_step_fct_predictor`) | UNVERIFIED — did not trace `ll_zAimp1` line-by-line against `nemo_rk3_two_step` |
| S-39 | tracer stage time-stepping (qco `(1+r3t)` weighting) stprk3_stg.F90:540-560 | `key_qco` | L O G | `omlc:1214 _stage` inside `_nemo_ws_rk3_tracer_pair_step` (uses `h_one_third`/`h_one_half`) | none | SHARED |
| S-40 | `tra_sbc` / `tra_sbc_RK3` trasbc.F90; stprk3_stg.F90:521 | `nn_fsbc` | D G A | `physics/surface_forcing/*`; DINO `surface_tendency_placement` (`applied_now`/`leapfrog_rhs`) | `surface_tendency_placement` (DINOConfig only) | ARTIFICIAL_BRANCH — a DINO-scoped selector for a NEMO site every card has; RK3 cards have no equivalent |
| S-41 | `tra_qsr` traqsr.F90 | `ln_traqsr`, `ln_qsr_2bd`/`ln_qsr_rgb` | G A | `physics/shortwave_penetration.py:443 apply_shortwave_penetration` | `shortwave_penetration.scheme` (`jerlov_2band`/`rgb_chl`) | NEMO_SWITCH |
| S-42 | `bbl` + `tra_bbl` trabbl.F90:129-136,243-284; stprk3_stg.F90:468,498,588 | `ln_trabbl`, `nn_bbl_adv=2` | O A | `physics/bbl_adv.py:176 bbl_transports` + `:232 apply_bbl_adv_tendency` — called in-stage at `omlc:1188` (OVERFLOW) **and** via `physics/bbl_adv.py:312 apply_bbl_adv_step` from `run_omip_core2.py:8420` (ORCA1, host post-step Euler + an extra transport cap NEMO has none of) | `bbl_adv_option` vs driver `--bbl-adv` | ARTIFICIAL_BRANCH — shared arithmetic, two compositions, and the ORCA1 path adds a clamp |
| S-43 | `tra_ldf` traldf_iso.F90 | `ln_traldf_lap`+`ln_traldf_iso` | D G A | `physics/lateral_mixing/gm_redi_latlon_cgrid.py` | `lateral_tracer_mixing`, `gm_redi.*` | see S-09 |
| S-44 | `tra_zdf` trazdf.F90 | always | D L O G A | `omlc:7537 _apply_implicit_vertical_mixing` (same fn as `dyn_zdf`) | `zdf_implicit_solver_evaluation` | see S-33/S-34 |
| S-45 | `tra_npc` tranpc.F90 | `ln_zdfnpc` | — | none | — | ABSENT (no card selects it) |
| **MLF-only rows below** | | | | | | |
| M-01 | `stp_MLF` whole-step composition stpmlf.F90:108-473 | no `key_RK3` | D | `omlc:9698 _leapfrog_step` (two `_step_impl` passes) **and** `omlc:10235 _nemo_mlf_step` (one pass, `_ldf_state=`) | `outer_integrator` (`leapfrog`/`nemo_mlf`) | **ARTIFICIAL_BRANCH (rank 5)** — two impls of `stp_MLF`; `nemo_mlf` is selected by NO card (only `scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round47.py:204` calls it directly) = dead branch |
| M-02 | `ssh_nxt` (+`div_hor`) sshwzv.F90; stpmlf.F90:214 | none | D | inside `blc` barotropic solve (eta from the substep window) | `barotropic_*` | SHARED |
| M-03 | `ssh_atf` sshwzv.F90; stpmlf.F90:316 | `rn_atfp` | D | `omlc:1537 _thickness_weighted_asselin` / `_leapfrog_step` RA filter | `asselin_gamma`, `tracer_combine` | SHARED (one impl) |
| M-04 | `tra_atf_qco` / `dyn_atf_qco` traatfqco.F90/dynatfqco.F90; stpmlf.F90:394-395 | `key_qco` | D | `omlc:1472 thickness_weighted_tracer_combine`, `:1523 thickness_weighted_tracer_content` | `tracer_combine` (`concentration`/`thickness_weighted`) | NEMO_SWITCH-shaped, but `concentration` has no NEMO arm -> ARTIFICIAL (inert for D; L/O/G/A are non-MLF) |
| M-05 | `mlf_baro_corr` stpmlf.F90:392 | `ln_dynspg_ts` | D | `omlc:9534 _apply_after_level_reconcile` | `barotropic_after_reconcile` (`off`/`nemo_mlf_baro_corr`) | NEMO_SWITCH-shaped; `off` has no NEMO arm -> ARTIFICIAL for any MLF card that leaves it off |
| M-06 | `dyn_ldf`/`tra_ldf` at Kbb inside one pass stpmlf.F90:250,368 | none | D | `_leapfrog_step` achieves it by running the whole tendency pipeline TWICE (`_ab2_scope_override="advective"`); `_nemo_mlf_step` by `_ldf_state=` | `outer_integrator` | see M-01 |

## 2. Ranked artificial branches, collapse plan, risk

Ranked by (cards affected) x (how much arithmetic diverges) x (whether the default is the non-NEMO arm).

1. ~~**`dom_qco_r3c` r3u/r3v face thickness — MIN vs MEAN (S-18).**~~
   **RESOLVED 2026-09-02 (post-1d6a7448d) — not a duplicate.** The premise of
   this item was wrong: `_nemo_ws_stage_transport` never actually reached the
   MIN rule for this quantity (the "LOCK/OVERFLOW reach MIN everywhere"
   framing conflated this row's own gate, a private test hook nothing sets,
   with S-19's `zad_qco_evaluation`). Both lanes already delegated to one
   shared kernel; see the "Triage against HEAD 648e5cd69" section below and
   the registry's S-18 row (now `OTHER_RECIPE`). No collapse needed here.

2. **PGF: `nemo_sco` unreachable from the ORCA1 driver (S-29).**
   `run_omip_core2.py:4570` restricts `--pgf-scheme` to `{adcroft, smc03}`;
   `nemo_sco` and any `hpg_zps` transcription do not exist on that lane, and
   `hpg_zps` does not exist at all.
   *Collapse:* widen the driver's choices to the model's
   `{adcroft, smc03, nemo_sco}` and add `nemo_zps` as the NEMO arm ORCA1
   actually resolves; then delete `adcroft` from NEMO-card reach.
   *Risk:* MEDIUM. Adding a choice is additive; the default move
   (`smc03` -> `nemo_*`) is a Rule-3 ASK, not a silent flip. `hpg_zps` is new
   code with no oracle yet.

3. **The WS momentum ladder is written twice (S-30).**
   `omlc:4253-4335` and `omlc:4830-4886` both encode
   `u_k = u0 + dt_k * RHS(u_{k-1})`. The first exists only to produce
   `state_mid.u` for the barotropic seed; NEMO seeds `dyn_spg_ts` from a single
   Kbb RHS (`stp2d.F90:127-196`).
   *Collapse:* seed the barotropic solve from the stage-1 RHS depth mean
   (`du_dt_pert`) as `stp_2D` does, delete the first ladder.
   *Risk:* MEDIUM. Removes 2 `tendencies()` calls/step (cheaper), but changes
   the barotropic seed on all three RK3 cards -> re-run the phase-3 gates.
   A flaw fixed in one ladder today is NOT fixed in the other (this is exactly
   how today's `_stage_vertical_up3` / live-HPG fix landed in ladder #2 only).

4. **`implicit_vmix_e3t_now_divisor` default keeps the non-NEMO divisor (S-34).**
   `trazdf.F90:219-220` has one divisor, `e3w(Kmm)`. legoESM's default is a
   midpoint of the AFTER thickness; the certified DINO MLF card runs the
   default. Rule 3 shape: the fix exists behind a knob whose default preserves
   the bug.
   *Collapse:* delete the midpoint arm; make `e3w(Kmm)` unconditional for every
   qco/NEMO card, keep `implicit_vmix_dzw_slot` for the Veros oracle only.
   *Risk:* MEDIUM. Moves the DINO twin's numbers; needs the Y5 certification
   re-run. Must be ASKED before the default moves.

5. **Two `stp_MLF` implementations, one dead (M-01).**
   `_leapfrog_step` (two `_step_impl` passes) and `_nemo_mlf_step` (one pass).
   Only the first is selectable by any card; the second is reachable only from
   a probe script.
   *Collapse:* pick one. `_nemo_mlf_step` is the structurally faithful one
   (NEMO calls `dyn_ldf(Kbb,Kmm)` in the single pass); prove bit-agreement,
   repoint `outer_integrator="leapfrog"` at it, delete the other.
   *Risk:* LOW-MEDIUM. Halves the per-step cost of the DINO card. If they do
   not agree bit-for-bit, the disagreement IS the finding.

6. **`nemo_stage_mean_imposition` — stage-3 `zub` on the wrong side of `dyn_zdf` (S-35).**
   NEMO does `dyn_zdf` (:430) then the `zub` correction (:433-446). L/O do the
   correction inside the stage ladder, i.e. BEFORE the implicit solve, and
   leave `nemo_stage_mean_imposition=False`; GYRE does both.
   *Collapse:* always re-impose after the implicit solve for `rk3_ws` cards;
   delete the selector.
   *Risk:* LOW-MEDIUM. One-variable change; affects L/O gates only when the
   implicit solve actually shifts the depth mean (A_v=1e-4, so small but nonzero).

7. **`nemo_flux_form_update_active` carries a non-NEMO `rk3_ws` conjunct (S-14).**
   NEMO branches on `ln_dynadv_vec .OR. lk_linssh` alone
   (`dynspg_ts.F90:719,733`).
   *Collapse:* drop the `momentum_time_integrator=="rk3_ws"` term; gate on
   `momentum_advection=="flux_form" or linear_free_surface`.
   *Risk:* LOW. No current card changes arm (D/G/A are vector-form, L/O are
   rk3_ws+flux-form). Purely removes a latent mis-route.

8. **`vor_ens` is ABSENT; L/O hide it behind `f==0` (S-26).**
   Both L1 namelists set `ln_dynvor_ens=.true.`; the cards resolve `al81` and
   the card validator refuses to run unless `f==0` and there is one wet row.
   *Collapse:* transcribe `dynvor.F90 vor_ens` as `vorticity_scheme="ens"` and
   select it on both L1 cards; then drop the f==0 guard to a plain assertion of
   what the namelist says.
   *Risk:* LOW for L/O (the operator is provably inert at f=0, so the gates
   should not move at all — which is also the non-vacuity check). It is the
   prerequisite for any L2 case with f != 0.

9. **`coriolis_scheme="matsuno_split"` on ORCA1 (S-27).**
   NEMO always carries Coriolis inside `dyn_vor`'s RHS. The Matsuno rotation
   sub-step is a legoESM invention, inert on L/O (f=0) but live on ORCA1.
   *Collapse:* move ORCA1 to `explicit_ab2` + `ene_total`/`een_total`.
   *Risk:* HIGH for ORCA1 stability (the GYRE note at
   `fidelity/nemo_recipe.py:940-946` records that the split blew up GYRE at
   f*dt~1 and coupling fixed it — so this may *improve* ORCA1). Needs a
   one-variable A/B against the certified 180-day card, and a Rule-3 ASK.

10. **~48 distinct `*_evaluation` two-arm selectors (S-04, S-07, S-09, S-16, S-19, S-33).**
    Each is "legoESM-generic vs `nemo_literal`" for a routine NEMO writes once;
    the default is the generic arm on every one. Full name list obtained by
    `grep -o "^    [a-z0-9_]*evaluation[a-z0-9_]*: str"` over
    `packages/ocean/legoesm/ocean/` (48 unique names, 69 declaration sites).
    *Collapse:* per family, promote the `nemo_literal` arm to the only arm and
    delete the selector, one PR per family, each landing with the gate that
    proved it.
    *Risk:* MEDIUM overall, LOW per family. Each is a bit-for-bit change on the
    cards that were on the generic arm; the point is that a defect fixed in the
    literal arm today is still live in the generic arm every non-DINO card runs.

11. **ORCA1's operator set is off the NEMO lane entirely (S-02, S-25, S-29, S-32, S-33, S-37, and the `momentum_time_integrator="rk3"` SSP ladder).**
    `--momentum-rk3` resolves to `momentum_time_integrator="rk3"` — the
    Shu-Osher SSP ladder at `omlc:4217-4253` — not `rk3_ws`. Together with
    `wright` EOS, `insitu` `eos_depth`, `superbee` tracers, `smc03` PGF,
    `upwind_perturbation` vertical momentum, `vector_laplacian` viscosity,
    `cosine` barotropic filter and `shared_thomas` ZDF, the ORCA1 card shares
    almost no operator with the certified NEMO cards.
    *Collapse:* build ONE `nemo_orca1` card through
    `fidelity/nemo_recipe.py:nemo_lat_lon_model_config` the way GYRE does,
    instead of assembling ORCA1 from `run_omip_core2.py` CLI defaults.
    *Risk:* HIGH. This is a new production configuration, not an edit; it must
    be A/B'd one variable at a time against the certified 180-day run and every
    default move ASKED. But until it exists, "NEMO-faithful 1-degree" names a
    card that runs a different model from the twins.

12. **Two card-assembly paths for the same RK3 identity (S-12, and §0 table).**
    LOCK/OVERFLOW build their config with a bare
    `LatLonCGridOceanConfig.from_flat(...)` inside
    `fidelity/nemo_testcase_recipe.py:_model_config`; GYRE builds via
    `nemo_lat_lon_model_config(NEMOModelRecipeConfig(...))`. The consequence is
    visible in §0: L/O silently keep `vorticity_scheme="al81"`,
    `coriolis_scheme="matsuno_split"`, `barotropic_coriolis="avg"`,
    `barotropic_een_seed="window_start"`, `zad_qco_evaluation="generic"` and
    `wzv_call2_evaluation="generic"` — six defaults the shared NEMO assembler
    would have set to the NEMO arm.
    *Collapse:* route all three RK3 cards (and DINO) through
    `nemo_lat_lon_model_config`; the `whole_step_identity` selector added on the
    l2 branch (`gyre_testcase_recipe.py:69-80`) is a per-case branch NEMO does
    not have and disappears with it.
    *Risk:* MEDIUM. Flipping those six on L/O is six one-variable A/Bs against
    the phase-3 gates; four of them should be provably inert at f=0.

13. **BBL applied at two different composition points (S-42).**
    Shared `bbl_transports`/`apply_bbl_adv_tendency`, but OVERFLOW folds it into
    the stage-3 tracer RHS (NEMO's site) while ORCA1 calls
    `apply_bbl_adv_step` from the driver as a host post-step forward Euler with
    an extra `0.25*V/dt` transport cap.
    *Collapse:* delete `apply_bbl_adv_step`; give ORCA1 the in-step tendency.
    *Risk:* LOW. The cap is documented as inactive at ORCA1 scales, so the
    change should be near-inert — which is the non-vacuity check.

## 3. UNVERIFIED

- Whether the LOCK_EXCHANGE / OVERFLOW oracle builds define
  `key_qcoTest_FluxForm`. `barotropic_face_depth="nemo_ssh_avg"` transcribes
  that arm (`dynspg_ts.F90:738-741`), and the DINO handoff doc asserts DINO's
  `cpp_DINO.fcm` has ZERO occurrences of it — I did not find the testcase
  `cpp_*.fcm` in this checkout. On a lat-lon C-grid the two NEMO arms coincide
  algebraically (`e1e2t(i,j)==e1e2t(i+1,j)`), so this is a provenance gap, not
  a numeric one.
- `nn_fct_imp=1` -> `ll_zAimp1` (`traadv_fct.F90:143,439-453`) vs legoESM's
  `fct_low_order_predictor="nemo_rk3_two_step"` + `fct_implicit_w`: not traced
  line by line (S-38).
- ORCA1's resolved `barotropic_face_depth`, `eos`, and `vorticity_scheme`: read
  off `LatLonCGridOceanConfig` defaults plus the sbatch flags, NOT by
  constructing the config (the tripole builder needs the mesh/forcing files,
  which are not on this machine's path). Everything marked ORCA1 in §0 is
  "default + CLI", not "measured".
- The GYRE row values are read from
  `git show origin/fidelity/nemo-testcases-l2-gyre-codex:.../nemo_testcase_recipe.py`
  source, not from a constructed card.
- NEMO's ORCA1 reference namelist was not read; the claim that ORCA1 resolves
  `ln_hpg_zps` is inference from "partial cells + z-coordinate", labelled
  PLAUSIBLE.
- The `dyn_drg_init` / `pu_RHSi` baroclinic-residual chain
  (`stp2d.F90:196`, `dynspg_ts.F90:1627-1642`) was not enumerated; bottom drag
  is inert on L/O (`bottom_drag_r=0`) but live on GYRE (`nemo_quadratic`) and
  ORCA1.
- `ldf_slp` / GM-Redi rows (S-09, S-43) were enumerated by selector name only;
  the ~12 `gm_redi_*_evaluation` arms were not individually traced to their
  NEMO lines.
- No claim here is backed by a numeric run. Everything is source-traced.

## 6. 2026-09-02 reclassification (mistake vs. legitimate multi-recipe fork)

Independent follow-up pass, reproduced verbatim from a separate agent's
reclassification table (source checkout `/tmp/l1-overflow-ro` @ `28d166428`,
branch `fidelity/nemo-testcases-l1-codex`; GYRE card read at
`b2f7c298`/`fidelity/nemo-testcases-l2-gyre-codex` via `git show`, not merged
to that checkout; no tracked file modified in either checkout). Applied to
`tests/ocean/unit/_nemo_branch_isomorphism_baseline.py` in the same PR that
added this section — see that file's module docstring for the mechanical
consequences (registry `disposition` changes, baseline-dict removals).

**Count correction**: this doc literally tags **19** rows `ARTIFICIAL_BRANCH`
(S-03, S-04, S-07, S-09, S-12, S-16, S-18, S-19, S-25, S-27, S-29, S-30,
S-32, S-33, S-34, S-35, S-40, S-42, M-01 — grep-counted), not 27. All 19 are
classified below.

### 6.1 Full table

| # | NEMO routine | implementations | selected by (recipes/cards) | reference named | classification | confidence |
|---|---|---|---|---|---|---|
| S-03 | `bn2` eosbn2.F90:253-288 | `eos.py:2627` insitu; `eos.py:2667` adiabatic; `eos.py:624` nemo_bn2 | insitu: KPP module + OMIP climate-match recipes (`omip_nemo_match_tripole_v1`/`_mpas_v1`) + any untouched card; adiabatic: Veros recipes (`veros_global_4deg_recipe.py:217`, `veros_acc_*recipe.py`); nemo_bn2: DINO, GYRE, and ORCA1 (`run_omip_core2.py:770`, confirmed, upgrades doc's "UNVERIFIED" to CONFIRMED) | nemo_bn2 cites eosbn2.F90 directly; adiabatic = Veros's own scheme (config.py:310-316); insitu = unreferenced legacy/KPP default | **OTHER_RECIPE** | CONFIRMED |
| S-04 | `bn2` live-e3w geometry eosbn2.F90:253-258 | `eos.py:754/775/892/590` (nemo_literal geometry) vs `preassembled_live` (default) | nemo arm: DINO + GYRE only (both `n2_mode="nemo_bn2"` AND `tke_n2_evaluation_stage="step_entry"`); no Veros/MITgcm/Oceananigans recipe touches `tke_n2_evaluation_stage` at all | nemo arm cites eosbn2.F90 (config.py:352-356); default names no reference | **OTHER_RECIPE** | CONFIRMED (gating); PLAUSIBLE (no non-NEMO recipe will ever need the alt arm — absence of evidence) |
| S-07 | `zdf_tke` internals (10 selectors) | one module `tke.py`, 10 `tke_*` fields, each documented "historical/generic" vs "DINO NEMO cards opt in" | default: 5 Veros fidelity recipes (`veros_acc_recipe.py`, `veros_acc_basic_recipe.py`, `veros_global_*_recipe.py`) construct `TKEConfig` untouched; nemo arm: DINO (`dino.py:1247-1256`) + GYRE (l2 branch) | each field's own comment names the arm's home explicitly (quoted) | **OTHER_RECIPE** | CONFIRMED |
| S-09 | `ldf_slp` ldfslp.F90 (~9 selectors) | one module `gm_redi_latlon_cgrid.py`, `GMRediConfig` | default: Veros recipes (`veros_acc_recipe.py:218`, `veros_global_4deg_recipe.py:227`) construct `GMRediConfig` untouched; nemo arm (`slope_scheme="nemo_iso_lap"`): DINO + GYRE via `nemo_recipe.py:392` | each field documents "historical" vs "DINO NEMO cards opt in" | **OTHER_RECIPE** | CONFIRMED (top-level `slope_scheme`); PLAUSIBLE (the other ~8 sub-fields not individually re-traced) |
| S-12 | `stp_2D` pre-step stp2d.F90:49-288 | no legoESM equivalent as a unit — a full 3-stage WS ladder run only to seed the barotropic solve | only cards with `momentum_time_integrator="rk3_ws"`: LOCK, OVERFLOW, GYRE (l2). No catalog recipe uses `rk3_ws` | none (structural, not a config selector) | **NEMO_DUPLICATE** (same defect as S-30, different altitude) | CONFIRMED |
| S-16 | `dyn_spg_ts` continuity/transport/spg dynspg_ts.F90:~640-840 (5 selectors) | generic FV form (SM2005-style, unreferenced) vs `nemo_literal`/`nemo_ssh_avg` | nemo arm: DINO, LOCK, OVERFLOW (`_model_config`, this checkout), GYRE (l2); generic: ORCA1 (`run_omip_core2.py` never sets these fields) and one Veros frozen-state tendency probe (deliberate bit-identical default, not a physics claim) | generic names no NEMO/other-oracle reference; nemo arm cites dynspg_ts.F90 | **NEMO_DUPLICATE** | CONFIRMED |
| S-18 | `dom_qco_r3c` r3u/r3v MEAN vs MIN domqco.F90:140-240 | `min_cell_to_uface`/`_vface` (Adcroft/Hill/Marshall 1997 MOM6/MITgcm convention, CONFIRMED cited) vs `nemo_qco_live_face_thicknesses` (NEMO MEAN) | MEAN: DINO + GYRE (l2); MIN: LOCK, OVERFLOW **on this checkout** (verified `_model_config` never sets `zad_qco_evaluation`/`wzv_call2_evaluation` — this is a live gap on the certified L1 cards too, not ORCA1-only as the source doc's framing implied) + ORCA1 | MIN cites Adcroft et al. 1997 (a real reference, but for a *different* model's convention, not NEMO); MEAN cites domqco.F90 | ~~**NEMO_DUPLICATE**~~ **SUPERSEDED 2026-09-02, post-1d6a7448d: OTHER_RECIPE** — this row's own "verified" gating claim was the error: `zad_qco_evaluation`/`wzv_call2_evaluation` gate `nemo_qco_wzv_operands` (S-19's routine, in `ocean_pe_latlon_cgrid.py`), not `_nemo_ws_stage_transport` (S-18's routine, in `ocean_model_latlon_cgrid.py`), which was never gated by them. See "Triage against HEAD 648e5cd69" below | CONFIRMED then, **RETRACTED** now |
| S-19 | `wzv` sshwzv.F90 | `diagnose_w_from_flux_div` (generic) vs `nemo_qco_wzv_operands` (literal), same selector family as S-18 | same as S-18: DINO/GYRE on literal; LOCK/OVERFLOW/ORCA1 on generic | same as S-18 | **NEMO_DUPLICATE** | CONFIRMED |
| S-25 | vertical momentum advection, non-NEMO arms | `upwind_perturbation` (default) / `centered_full` / `nemo_advective` | upwind_perturbation: unclaimed default, run by most catalog recipes + ORCA1 (no override); centered_full: `veros_faithful_v1`; nemo_advective: DINO + GYRE | centered_full cites Veros `core/momentum.py` (quoted); nemo_advective cites dynzad.F90; default names nothing | **OTHER_RECIPE** (ORCA1 landing on the unclaimed default instead of `nemo_advective` is a driver reachability gap, not a duplicate) | CONFIRMED |
| S-27 | Coriolis time placement | `matsuno_split` (default, unreferenced legacy) vs `explicit_ab2` (in-RHS) | matsuno_split: unclaimed default, incl. ORCA1 (unset); explicit_ab2: `veros_faithful_v1`, `oceananigans_v1`, `mitgcm_v1` (recipes.py) **and** DINO + GYRE (this arm is simultaneously Veros's own scheme AND NEMO's real Coriolis placement) | explicit_ab2 docstring: "VEROS-FAITHFUL" + DINOConfig comment "(MITgcm/Oceananigans/Veros)" + cites dynspg_ts.F90 for GYRE's use; matsuno_split names nothing | **OTHER_RECIPE** (ORCA1 defaulting to matsuno_split while live is the real gap the source doc's own item 9 names — a missing override, not a second NEMO implementation) | CONFIRMED |
| S-29 | `dyn_hpg` on ORCA1 (`ln_hpg_zps`) | `adcroft` (default) / `smc03` / `nemo_sco` (unreachable from ORCA1) | `run_omip_core2.py:4570` `--pgf-scheme choices=[adcroft,smc03]` (nemo_sco absent, CONFIRMED); adcroft: default recipes; smc03: `eady_weno5_v1`; nemo_sco: DINO/LOCK/OVERFLOW/GYRE | adcroft cites Adcroft/Hallberg/Hill 2008 in `pgf_ahh08.py` (state.py's own comment says "Adcroft & Campin 2004" for the same string — citation-year mismatch, flagged, unresolved); smc03 cites Shchepetkin & McWilliams 2003 | **OTHER_RECIPE** (both ORCA1-reachable arms are real, separately-cited non-NEMO papers with genuine recipe consumers; the defect is the driver CLI cannot reach `nemo_sco` at all — a reachability gap, not duplicated NEMO work) | CONFIRMED |
| S-30 | WS momentum ladder written twice stprk3_stg.F90:344-374 | ladder #1 (seeds barotropic solve only) vs ladder #2 (barotropic-corrected, kept) — same recurrence, both always run | only cards with `momentum_time_integrator="rk3_ws"`: LOCK, OVERFLOW, GYRE. No catalog recipe, no ORCA1 (uses `"rk3"`, a third distinct branch) | none — no non-NEMO recipe ever reaches `rk3_ws` | **NEMO_DUPLICATE** | CONFIRMED |
| S-32 | `dyn_ldf`->`ldf_lap` dynldf_lap_blp.F90 | `nemo_div_curl` / `vector_laplacian` (default) / `flux_divergence` | nemo_div_curl: DINO + GYRE; vector_laplacian: unclaimed default incl. ORCA1; flux_divergence: `veros_faithful_v1`, `oceananigans_v1`, `mitgcm_v1` | nemo_div_curl cites "NEMO dyn_ldf_lev_lap" (dino.py:914); flux_divergence cites Veros `core/friction.py harmonic_friction`; vector_laplacian names nothing | **OTHER_RECIPE** (clean 3-way split; side-finding: DINO's `off` vs GYRE's `nemo_e3` e3-weighting sub-flag is a MEASURED deliberate choice — an A/B found `nemo_e3` worsens DINO's gate rows — not an oversight) | CONFIRMED |
| S-33 | `dyn_zdf`/`tra_zdf` implicit solver | `shared_thomas` (default, generic normalized-row solver) vs `nemo_literal` (unnormalized NEMO recurrence) | shared_thomas: unclaimed default incl. all catalog recipes + ORCA1; nemo_literal: DINO, LOCK, OVERFLOW, GYRE | shared_thomas names nobody's oracle fidelity (ordinary numerics); nemo_literal implicitly contrasts against dynzdf/trazdf (F90 line not independently re-derived) | **OTHER_RECIPE** (ORCA1 on shared_thomas instead of nemo_literal is the same reachability-gap pattern as S-25/27/29/32, not duplicated NEMO work) | CONFIRMED (selection facts); PLAUSIBLE (exact F90 citation) |
| S-34 | `trazdf`/`dynzdf` divisor e3w(Kmm) | legacy midpoint (default, both flags False) / `implicit_vmix_e3t_now_divisor=True` (NEMO) / `implicit_vmix_dzw_slot=True` (Veros) | legacy: DINO (certified kamm_mlf card — does NOT set the NEMO flag, stays on legacy!) + unclaimed default recipes + ORCA1; NEMO arm: LOCK, OVERFLOW, GYRE; Veros arm: `veros_faithful_v1` | NEMO arm cites trazdf.F90:219-220 (quoted verbatim, still accurate); Veros arm cites Veros `thermodynamics.py:267` (quoted); legacy names nothing | **OTHER_RECIPE**, with a flagged separate Rule-3 defect: DINO is a certified NEMO card silently running the unreferenced legacy divisor instead of its own available NEMO arm — a hidden-default bug, not a duplicate-implementation bug | CONFIRMED |
| S-35 | stage-3 barotropic correction `zub` stprk3_stg.F90:433-446 | site (a) `_replace_stage_mean` (before the implicit solve) vs site (b) `_impose_mean` (after the implicit solve) | site (a): every rk3_ws card unconditionally (LOCK, OVERFLOW, GYRE); site (b): gated by `nemo_stage_mean_imposition` (default False) — only GYRE (l2) sets it True; LOCK/OVERFLOW never do | **both comments cite the identical NEMO line range** stprk3_stg.F90:433-446/:440 | **NEMO_DUPLICATE** — GYRE runs both (redundant); LOCK/OVERFLOW run only the wrong-side-of-`dyn_zdf` one | CONFIRMED |
| S-40 | `tra_sbc`/`tra_sbc_RK3` trasbc.F90 | `applied_now` (pre-step mutation, legacy) vs `leapfrog_rhs` (folds into MLF Nnn RHS) | applied_now: DINO's non-MLF oracle sub-recipes (`veros`/`mitgcm`/`oceananigans` DINO variants); leapfrog_rhs: only the certified MLF card (`nemo_dino_kamm_mlf`) — and `run_dino.py:625-634` hard-forbids the buggy pairing (`leapfrog` + `applied_now`) via `SystemExit` | leapfrog_rhs cites tra_sbc.F90 Nnn-RHS placement; applied_now names nothing | **OTHER_RECIPE** — RK3 cards (GYRE/L/O/ORCA1) have no equivalent selector because their own site is hard-wired, not selectable; the one two-arm case (DINO) already has a driver-level guard closing the Rule-3 failure mode | CONFIRMED |
| S-42 | `bbl`/`tra_bbl` trabbl.F90 | `apply_bbl_adv_tendency` (in-stage) vs `apply_bbl_adv_step` (driver post-step Euler, extra `0.25*V/dt` cap NEMO lacks) | in-stage: OVERFLOW (`bbl_adv_option=2`); driver-only: ORCA1's `--bbl-adv` flag — which **never sets** `bbl_adv_option` (grep confirmed zero hits), so ORCA1's in-model dispatch stays permanently off at its 0 default while physics runs through the separate driver path instead | **both docstrings cite NEMO trabbl / Campin & Goosse BBL exchange for the same routine** | **NEMO_DUPLICATE** — two independent NEMO-trabbl transcriptions with zero common selector between them | CONFIRMED |
| M-01 | `stp_MLF` whole-step composition stpmlf.F90:108-473 | `_leapfrog_step` (two `_step_impl` passes) vs `_nemo_mlf_step` (one pass, single-pass transcription) | dispatch exists NOW at `outer_integrator` ("leapfrog"->a, "nemo_mlf"->b — contradicts (b)'s own stale docstring claiming it's unwired); DINO's certified card selects `"leapfrog"` (a). No recipe/card selects `"nemo_mlf"`. (b) DOES have real committed test coverage (`tests/ocean/unit/test_nemo_mlf_step_transcription.py`) — correction to source doc, which called it probe-only/dead | both cite stpmlf.F90 `stp_MLF` directly; (b) additionally cites a spec doc | **NEMO_DUPLICATE** (not ORPHAN: (b) names a reference and has real test coverage — it is validated-but-unpromoted, not dead) | CONFIRMED |

### 6.2 Counts

| classification | rows | which |
|---|---|---|
| NEMO_DUPLICATE | 7 (was 8) | S-12, S-16, S-19, S-30, S-35, S-42, M-01 |
| OTHER_RECIPE | 12 (was 11) | S-03, S-04, S-07, S-09, S-18 (moved here 2026-09-02, see below), S-25, S-27, S-29, S-32, S-33, S-34, S-40 |
| ORPHAN | 0 | — |

**S-18 moved NEMO_DUPLICATE -> OTHER_RECIPE, 2026-09-02 (post-1d6a7448d)**: the
row above still records this pass's original (CONFIRMED, at the time) verdict
as history. It was retracted the same day — see "Triage against HEAD
648e5cd69" below for the correction and the registry's S-18 row for the
mechanical fix.

**Headline correction**: several rows this doc flagged among its top-priority
"mistakes to collapse" — S-27 (Coriolis split), S-29 (PGF), S-33/S-34
(ZDF solver/divisor) — are legitimate Veros/MITgcm/Oceananigans/paper-cited
forks, each carrying a *different* real defect: ORCA1 (and, for S-34, DINO)
simply never selects the NEMO arm it already has, a reachability/hidden-
default bug, not duplicated NEMO implementation work. S-16/S-19 (barotropic
transport, wzv) are confirmed genuine duplicates reaching **every** certified
NEMO card, including LOCK/OVERFLOW on this checkout for S-19 (not just ORCA1,
as this doc's own §0 framing suggested). **S-18 (qco face-thickness) is NOT**:
the "reaches MIN on LOCK/OVERFLOW" claim below conflated S-18's own gate (a
private test hook nothing sets) with S-19's `zad_qco_evaluation` — see the
"Triage against HEAD 648e5cd69" section for the correction; S-18 moved to
OTHER_RECIPE the same day.

S-12 and S-30 are both genuine NEMO_DUPLICATE (S-30 is this doc's own rank-3
collapse item, §2 above — the one row with an already-realized bug: a fix to
`_stage_vertical_up3`/live-HPG landed in "ladder #2" only). Neither carries a
baseline-dict entry: both duplicate sites are inline code inside one function
(`_step_impl`), not AST-distinct symbols, so the ratchet cannot mechanically
enforce them — same carve-out already used for S-33/S-34's divisor branch.
The classification (genuine defect, not a false alarm) still stands; only the
mechanical enforcement is unavailable until a refactor promotes the two sites
to named symbols.

### 6.3 Disagreements

**None.** The nemo_duplicate/other_recipe/orphan axis is new with this pass —
the audit doc (§0-§5 above) never itself assigned a "kind" to compare
against, so no row's OTHER_RECIPE/NEMO_DUPLICATE verdict here contradicts a
prior claim in this doc. Two adjacent, non-blocking items are worth a mention
so they aren't mistaken for a disagreement: (a) S-29/S-28's PGF citations
have an unresolved citation-year mismatch (`state.py`'s field comment says
"Adcroft & Campin 2004" where `pgf_ahh08.py`'s docstring says "Adcroft,
Hallberg & Hill 2008" for what is presumably the same code) — flagged as a
separate, still-open finding, explicitly said not to change the row's
verdict; (b) S-27's docstring claim that `coriolis_scheme="explicit_ab2"`
hard-requires `barotropic_solver="rigid_lid"` is contradicted by GYRE's card
(which pairs it with `barotropic_solver="explicit_substep"` successfully) —
a code-vs-docstring inconsistency, not a reclassification dispute, and also
does not change S-27's verdict.

### 6.4 ORPHAN candidates

Zero of the 19 rows classify ORPHAN. `_nemo_mlf_step` (M-01) was the one
candidate this doc's §2 rank-5 item suggested might be dead code; it has a
real committed unit test (`tests/ocean/unit/test_nemo_mlf_step_transcription.py`)
and names a NEMO reference, so it fails the ORPHAN bar ("selected by no
recipe **and** naming no reference") on the reference prong — it stays
NEMO_DUPLICATE (validated-but-unpromoted).

## Triage against HEAD 648e5cd69

Worktree `/tmp/wt-branch-iso`, branch `fidelity/nemo-branch-isomorphism-audit`,
HEAD `648e5cd69` (clean). Read-only re-verification of six rows against the
CURRENT tree — every claim below is a fresh grep/read on this checkout, not a
citation of the §6 table above (which is now stale for S-18, see its own
struck-through rows). NEMO source
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE`.

### S-18 (dom_qco_r3c) — see the registry fix, summary only

Reclassified `ARTIFICIAL_BRANCH` → `OTHER_RECIPE` in this same commit (see the
struck-through §1 item 1, §6.1 row, and §6.2 count above). The "LOCK/OVERFLOW
reach MIN" finding this doc and the registry both carried was a misreading:
`_nemo_ws_stage_transport`'s face-thickness builder (`_nemo_ws_qco_stage_faces`
→ `vertical.py:218 nemo_qco_live_face_geometry_cgrid` →
`vertical.py:140 nemo_qco_live_face_geometry_from_operands`) was never gated by
`zad_qco_evaluation`/`wzv_call2_evaluation` — those gate a *different*
function, `ocean_pe_latlon_cgrid.py:1472 nemo_qco_wzv_operands` (S-19's own
routine). `min_cell_to_uface`'s only competing role for THIS quantity is
`_NEMOWSRK3TestHooks.legacy_stage_min_face_thickness`
(`ocean_model_latlon_cgrid.py:1049`), set `True` only inside
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_phase3_stage_sweep_gate.py`,
never by a production card. `min_cell_to_uface`'s real role (MOM6/MITgcm
hFacW=min, `latlon_cgrid_operators.py:277-284`) is the PE-lane
depth-average/slow-forcing (`ocean_pe_latlon_cgrid.py:1364`,
`ocean_model_latlon_cgrid.py:4125`), reached by DINO/LOCK/OVERFLOW/ORCA1 *and*
by `veros_faithful_v1`/`mitgcm_v1`/`oceananigans_v1` (confirmed real catalog
names via `legoesm.ocean.recipes.list_recipes()`), none of which can ever
reach the NEMO arm (it needs `z_coord.nemo_hu_0`/`nemo_e1e2*`, only present on
NEMO-mesh-carrying coordinates). Classification: **ALREADY_COLLAPSED /
OTHER_RECIPE** (both apply: the genuine WS-RK3-vs-MLF structural duplication
was already resolved by commit `1d6a7448d`, a pure refactor per its own
message; the row's remaining second impl was never a duplicate of this
quantity at all).

### S-19 (`wzv`) — generic vs `nemo_literal`

1. **Duplicate present at HEAD?** Yes. `vertical.py:1374
   diagnose_w_from_flux_div` (generic) vs `ocean_pe_latlon_cgrid.py:1472
   nemo_qco_wzv_operands` (literal), both reachable — confirmed by direct
   `grep`, both symbols AST-resolve and both have live callers
   (`ocean_model_latlon_cgrid.py:456,1154,5340` for the generic diagnostic;
   `ocean_model_latlon_cgrid.py:5333` for the literal path). Gated by
   `state.py:2221 zad_qco_evaluation` / `:2225 wzv_call2_evaluation`, both
   default `"generic"`.
2. **What do the certified cards run?** DINO's certified card
   (`dino.py:1418-1419`, inside `dino_r1_exact_config`) explicitly sets both to
   `"nemo_literal"`. `nemo_testcase_recipe.py` (LOCK/OVERFLOW's builder) never
   sets either field (`grep` — zero hits) → both stay at the config default
   `"generic"`, i.e. LOCK/OVERFLOW do **not** reach the NEMO arm for `wzv`.
3. **ORCA1?** `scripts/run/run_omip_core2.py` never sets either field (`grep`
   — zero hits) → resolves `"generic"`, same as LOCK/OVERFLOW.
4. **Any non-NEMO recipe/card/matrix reaches the legacy arm deliberately?** No.
   `packages/ocean/legoesm/ocean/recipes.py` never sets either field for any
   catalog recipe (`veros_faithful_v1`/`mitgcm_v1`/`oceananigans_v1` all fall
   to the same unclaimed `"generic"` default as ORCA1) — this is the "unclaimed
   default", not a deliberate OTHER_RECIPE selection.
5. **Config-only re-point?** No, for LOCK/OVERFLOW *and* ORCA1.
   `nemo_qco_wzv_operands` hard-`raise`s unless `z_coord` carries
   `nemo_e3t_0`/`nemo_hu_0`/`nemo_hv_0`/`nemo_e1e2t`/`nemo_e1e2u`/`nemo_e1e2v`
   (`ocean_pe_latlon_cgrid.py:1494-1503`) — unlike S-18's
   `_nemo_ws_qco_stage_faces`, this function does **not** fall back to building
   these operands generically from the card's own grid; it requires the raw
   NEMO mesh fields to already be attached to `z_coord`. ORCA1's driver reads
   `data/grids/eORCA1.2_mesh_mask.nc` via `read_mesh_mask_bathy` but that
   helper (per its call sites, `run_omip_core2.py:1378` etc.) returns only
   `land_mask`/`H_bathy`, not the `nemo_hu_0`/`nemo_e1e2*` fields the function
   needs — same missing-operand wall as S-18's ORCA1 case. LOCK/OVERFLOW carry
   the same structural gap (their z-coordinate is not built from a real NEMO
   mesh_mask at all).
6. **Classification: NEEDS_ORCA1_OPERANDS** (extends to LOCK/OVERFLOW too —
   `nemo_qco_wzv_operands` would need the generic-operand-construction fallback
   `_nemo_ws_qco_stage_faces` already has before any of the three cards could
   be re-pointed by config alone).

*Recommendation*: give `nemo_qco_wzv_operands` the same generic-operand
fallback `_nemo_ws_qco_stage_faces` uses (build `hu_0`/`e1e2*` from the card's
own grid when `z_coord.nemo_*` is absent) before attempting the collapse.

### S-16 (`dyn_spg_ts` continuity/transport/spg)

1. **Duplicate present at HEAD?** Yes. `barotropic_latlon_cgrid.py:548
   nemo_literal_continuity_divergence` (+ 4 sibling `nemo_literal_*`/
   `nemo_ssh_avg_*` helpers) vs `:1135 _run_substep_loop`'s generic inline arms,
   gated by 5 selectors (`barotropic_continuity_evaluation`,
   `_transport_accumulation_evaluation`, `_pgf_evaluation`, `_seed_evaluation`,
   `_seed_face_depth`), all defaulting to `"generic"`/`"min_rule"`
   (`state.py:1130`).
2. **Certified cards?** `nemo_testcase_recipe.py:80-84` (LOCK/OVERFLOW) sets
   ALL FIVE to their NEMO-literal values. DINO (`dino.py:1516` + siblings, same
   `dino_r1_exact_config` override block as S-19) also sets the NEMO arm. GYRE
   (l2 branch, per the existing §6.1 row, unchanged) too. So — unlike S-18/
   S-19 — **LOCK/OVERFLOW genuinely reach the NEMO arm on this checkout.**
3. **ORCA1?** `scripts/run/run_omip_core2.py` never sets any of the 5 fields
   (`grep` — zero hits) → resolves the generic default.
4. **Non-NEMO recipe reach?** No — `recipes.py` never sets these fields either
   (zero hits); same unclaimed-default pattern as ORCA1, not a deliberate
   OTHER_RECIPE fork.
5. **Config-only re-point for ORCA1?** Yes. `nemo_literal_continuity_divergence`
   (`barotropic_latlon_cgrid.py:548-577`) takes only `H_u, H_v, U, V, u_mask,
   v_mask, grid` — ordinary model-generic operands, no `z_coord.nemo_*`
   fields. The `"nemo_ssh_avg"` face-depth rule (`_nemo_ssh_avg_prep`/`_apply`,
   called near `barotropic_latlon_cgrid.py:1230`) likewise only needs
   `H_bathy, mask, grid, dtype` — everything ORCA1 already has. No missing
   operand blocks this arm, unlike S-18/S-19.
6. **Classification: COLLAPSIBLE_NOW** — ORCA1 is a five-field config flip
   away from the arm its own DINO/LOCK/OVERFLOW/GYRE siblings already run; no
   new code or operand needed. (Per Rule 3/the map's own §2 ranked item 12,
   the actual default flip is a one-line-per-field ASK, not a silent move.)

**2026-09-02 RETRACTION of items 3, 5 and 6 above (MEASURED on this checkout,
fp64).** They claimed ORCA1 "resolves the generic default" and that the
collapse is a five-field ORCA1 config flip. Both are wrong, in exactly the way
S-18's retracted claim was wrong: a config field can RESOLVE to `"generic"` on
a card that never EXECUTES the code that field gates.

* **ORCA1 is not on this code path at all.** Instantiating the ORCA1/OMIP
  standard card's own builder on this checkout —
  `fidelity/nemo_match_recipe.py:394 nemo_match_tripole_model_config()`, the
  config `run_omip_core2.py --grid tripole` builds — prints
  `barotropic.barotropic_solver = "implicit_cn"` (measured, not read off a
  comment). `scripts/cluster/omip_nemo/run_standard_faithful_1deg.sbatch`
  passes no `--barotropic-solver`, so it takes that; the five other committed
  ORCA/eORCA cards (`_e025_fullcard_smoke_{4gpu,4gpu_fp32,8gpu,8gpu_nobbl}`,
  `run_eorca025_4gpu_multinode`) pass `--barotropic-solver implicit_cn`
  explicitly. `implicit_cn` dispatches to
  `dynamics/barotropic_implicit_latlon_cgrid.py`, which carries its OWN
  private `_depth_average_to_faces` (`:96`, called at `:1370`) and reads NONE
  of the five selectors — `_run_substep_loop` and the split-explicit
  `_depth_average_to_faces` never run on ORCA1. The five fields resolving to
  `"generic"` there is an INERT VALUE, not a running arm. Consequence: the
  ORCA1 resolved-config diff for this row is EMPTY, and the ORCA1 re-point
  mandate cannot authorise this collapse because ORCA1 is not what it changes.
* **The generic arm's real consumers are non-NEMO recipes and test-matrix
  cases.** `barotropic_solver` defaults to `"explicit_substep"`
  (`state.py:1403`) and three catalog recipes select it: `default_wright_v1`,
  `legoesm_linear_v1`, `nemo_v1`. MEASURED (fp64, call counters on the
  module's own symbols, 3 substeps on an 8x16 basin): all three call
  `divergence_cgrid` and never `nemo_literal_continuity_divergence` /
  `nemo_literal_accumulate_transport` /
  `_nemo_literal_barotropic_pressure_gradient` /
  `_nemo_literal_seed_from_reference_mesh`. Via
  `experiments/recipe_map.py` — whose tags are drift-tested against each
  experiment's REAL assembled config by `tests/ocean/unit/test_recipe_map.py`
  — those two default recipes are the dycore of 19 ocean test-matrix
  experiments: `acc_channel`, `baroclinic`, `baroclinic_gyre`,
  `barotropic_wave`, `geostrophic_adjustment`, `global_barotropic_wind`,
  `inertia_gravity_wave`, `isomip_plus`, `lock_exchange`, `munk_gyre`,
  `neverworld2_lite`, `overflow`, `phillips_two_layer`, `regional_gyre`,
  `rest_state`, `stommel_gyre_tracer` (`default_wright_v1`) plus
  `global_overturning`, `eady_instability`, `held_larichev`
  (`legoesm_linear_v1`).
* **The two arms are NOT numerically equivalent**, so this is not a free
  deletion: the committed test `tests/ocean/unit/
  test_barotropic_continuity_and_drag.py::
  test_association_selector_holds_face_depth_and_drag_fixed` asserts the two
  arms produce bit-DISTINCT `eta`.

**Revised triage: NEMO_DUPLICATE (classification UNCHANGED) but BLOCKED for
collapse, not `COLLAPSIBLE_NOW`.** Collapsing S-16 moves the numbers of 19
test-matrix experiments plus the `nemo_v1` catalog dycore. It does NOT move
ORCA1. Unblocking it needs a separate, explicitly-asked decision about those
19 idealized cases (either re-baseline them on the NEMO arm, or give the
generic FV arm a real reference and keep it as an OTHER_RECIPE fork) — it is
not an ORCA1 re-point.

The NEMO oracle lines are re-confirmed on this checkout, so the row's
duplicate-ness is not in doubt: `dynspg_ts.F90:604-609` (`zhU`/`zhV` metric
transports), `:627` (`zhdiv`), `:641-642` (`un_adv` accumulation), `:682-684`
(`zu_spg`) — ONE NEMO program, two legoESM implementations. The row keeps its
baseline entry.

### S-35 (stage-3 `zub` barotropic correction)

1. **Duplicate present at HEAD?** Yes. Site (a) `ocean_model_latlon_cgrid.py:
   4800 _replace_stage_mean`, called unconditionally per WS-RK3 stage (call
   sites `:4963,4979,4997`). Site (b) `:6555 _fixed_depth_means`, called at
   `:6382,6465` but gated: `_impose_mean = getattr(_cfg_b.barotropic,
   "nemo_stage_mean_imposition", False) and _apply_implicit_vmix` (`:6378-6380`).
2. **Certified cards?** `nemo_stage_mean_imposition` defaults `False`
   (`state.py:1265`). Only `fidelity/nemo_recipe.py:978` (GYRE) sets it `True`.
   `nemo_testcase_recipe.py` (LOCK/OVERFLOW) never sets it (`grep` — zero
   hits) → both stay on site (a) only, i.e. the WRONG side of `dyn_zdf` per
   NEMO (`stprk3_stg.F90:430,433-446`) — a live gap on the certified L1 cards.
3. **ORCA1?** `run_omip_core2.py` never sets it either → same site-(a)-only
   gap.
4. **Non-NEMO recipe reach?** No — `recipes.py` never sets this field (zero
   hits); unclaimed default shared with ORCA1/LOCK/OVERFLOW.
5. **Config-only re-point?** Yes, structurally: `_fixed_depth_means(self, st,
   z_coord=None, config=None, grid=None)` takes only model-generic arguments,
   no NEMO mesh operand. Flipping `nemo_stage_mean_imposition=True` for LOCK/
   OVERFLOW/ORCA1 needs no new code.
6. **Classification: COLLAPSIBLE_NOW** — same config-flip shape as S-16, no
   operand gap. (The map's own §2 ranked item 6 already flags this as a
   pending Rule-3 ASK, "Risk: LOW-MEDIUM... needs re-running the phase-3
   gates" — a real, live TODO, not resolved.)

### S-42 (BBL: in-stage vs driver post-step)

1. **Duplicate present at HEAD?** Yes. `physics/bbl_adv.py:232
   apply_bbl_adv_tendency` (in-stage), called from
   `ocean_model_latlon_cgrid.py:1278` — gated by `bbl_adv_option` (default `0`,
   `state.py:2082`). `physics/bbl_adv.py:312 apply_bbl_adv_step` (driver
   post-step Euler with an extra transport cap), called ONLY from
   `scripts/run/run_omip_core2.py:8421` — its sole caller in the whole tree.
2. **Certified cards?** `nemo_testcase_recipe.py:215` (LOCK) sets
   `bbl_adv_option=0` (BBL off — flat bottom, no bathymetric step). `:271`
   (OVERFLOW) sets `bbl_adv_option=2` (BBL on, in-stage) — the NEMO arm.
3. **ORCA1?** Never sets `bbl_adv_option` in `run_omip_core2.py` (`grep` —
   zero hits) → stays at the config default `0` (in-model BBL OFF). Instead it
   has its OWN CLI flag, `--bbl-adv` (`run_omip_core2.py:4803`), which drives
   `apply_bbl_adv_step` entirely outside the `bbl_adv_option` dispatch — two
   disconnected selectors for the same NEMO physics (`trabbl.F90`).
4. **Non-NEMO recipe reach?** No — `recipes.py` never sets `bbl_adv_option`;
   not checked, no catalog recipe runs a bathymetric-step case that would need
   BBL at all.
5. **Config-only re-point for ORCA1?** Plausibly yes, not fully proven here.
   `apply_bbl_adv_tendency` needs `h_k, area, geom: BBLGeometry` — the same
   `BBLGeometry` object `run_omip_core2.py:6345` already builds
   (`bbl_static_geometry`) to feed its OWN `--bbl-adv` driver path. No NEMO-
   mesh-only operand is visible in the signature, so setting
   `bbl_adv_option=2` in ORCA1's config (instead of `--bbl-adv`) looks
   structurally feasible, but the two paths' extra transport cap difference
   (noted in the existing §6.1 row) means this needs an A/B, not just a flip.
6. **Classification: COLLAPSIBLE_NOW** (operands already exist; needs an A/B
   + Rule-3 ASK before the default moves, not new plumbing).

### M-01 (`stp_MLF`: `_leapfrog_step` vs `_nemo_mlf_step`)

1. **Duplicate present at HEAD?** Yes. `ocean_model_latlon_cgrid.py:9842
   _leapfrog_step` vs `:10379 _nemo_mlf_step`, dispatched at `:8938`/`:8949` on
   `outer_integrator in ("leapfrog", "nemo_mlf")` — both are real, wired
   dispatch values (multiple `raise` guards elsewhere reference both, e.g.
   `:2998,3062,3473,4731`), not a stub.
2. **Certified cards?** DINO's certified card uses `outer_integrator=
   "leapfrog"` (repeatedly documented in `dino.py`, e.g. `:302,323,627,638,
   998,1484,1720,1904` — "Requires outer_integrator='leapfrog'"). No
   production card or catalog recipe sets `"nemo_mlf"` (`grep` across
   `dino.py`/`recipes.py` — zero hits); its only caller outside the dispatch
   itself is the committed test `tests/ocean/unit/
   test_nemo_mlf_step_transcription.py` (22 test functions).
3. **ORCA1?** N/A — M-01 is MLF-only (DINO is the sole MLF-lane card in the
   catalog); ORCA1 runs RK3, not MLF.
4. **Non-NEMO recipe reach?** No — `"nemo_mlf"` names a structural
   transcription choice, not a scheme any catalog recipe selects.
5. **Config-only re-point?** Yes, in principle — `outer_integrator="nemo_mlf"`
   is already a first-class dispatch value; no missing operand blocks it. But
   the map's own §2 ranked item 5 says bit-agreement between `_leapfrog_step`
   and `_nemo_mlf_step` still needs to be PROVEN before re-pointing DINO at
   it — that measurement was not found to have been run on this checkout
   (only the transcription test's own internal checks, not a DINO-card A/B).
6. **Classification: COLLAPSIBLE_NOW, gated on an unmeasured bit-agreement
   check** — not an operand gap (there is none here); the blocker is a
   not-yet-run A/B, per the map's own existing plan.

### Summary table

| row | duplicate still live? | certified-card arm | ORCA1 arm | config-only re-point? | classification |
|---|---|---|---|---|---|
| S-18 | resolved (was a misreading) | n/a | n/a | n/a | ALREADY_COLLAPSED / OTHER_RECIPE |
| S-19 | yes | DINO=literal; LOCK/OVERFLOW=generic | generic | ~~no~~ **YES since 2026-09-02** — `nemo_qco_resolved_mesh_operands` rebuilds `hu_0`/`e1e2*`/`e2u`/`e1v` from any card's own grid + reference ladder | ~~NEEDS_ORCA1_OPERANDS~~ **MEASURED_INERT** (see "S-19 arm result") |
| S-16 | yes | DINO/LOCK/OVERFLOW/GYRE=nemo_literal | **not on this path** — ORCA1 resolves `barotropic_solver=implicit_cn` (measured), which never runs the gated code | n/a for ORCA1; the generic arm's real consumers are 19 test-matrix experiments + `nemo_v1` | **BLOCKED** (was COLLAPSIBLE_NOW — retracted 2026-09-02, see the S-16 retraction above) |
| S-35 | yes | GYRE=both sites; LOCK/OVERFLOW=site (a) only (wrong side of dyn_zdf) | site (a) only | yes — `_fixed_depth_means` is model-generic | COLLAPSIBLE_NOW |
| S-42 | yes | OVERFLOW=in-stage (2); LOCK=off (0) | separate driver-side `--bbl-adv` path, `bbl_adv_option` stays 0 | plausibly — same `BBLGeometry` the driver path already builds | COLLAPSIBLE_NOW |
| M-01 | yes | DINO=`leapfrog`; `nemo_mlf` selected by no card (test-only) | n/a (MLF-only row) | yes, structurally — blocked on an unmeasured bit-agreement check, not an operand gap | COLLAPSIBLE_NOW (pending A/B) |

## 2026-09-02 ORCA1 executed-arm corrections (appended; history above not rewritten)

Full authoritative table: `docs/ocean/fidelity/orca1_card_executed_arms.md`
(same worktree/HEAD as this section). Built by instantiating the actual
production assembly path (`run_standard_faithful_1deg.sbatch` &rarr;
`run_omip_core2.build_tripole` &rarr; `run_omip._create_setup("tripole")` &rarr;
`nemo_match_tripole_model_config` &rarr; `config.replace_flat(**_ovr)`) in fp64
on CPU, and cross-checked against NEMO's real ORCA1
`namelist_cfg` (`/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_cfg`) read
directly for the first time in this audit chain.

**This resolves this doc's own §3 UNVERIFIED item** ("ORCA1's resolved
`barotropic_face_depth`, `eos`, and `vorticity_scheme`: read off
`LatLonCGridOceanConfig` defaults plus the sbatch flags ... NOT by
constructing the config") — the config IS now constructed (mesh/grid/state
mocked out; nothing about a scheme selector needed the mesh file). Resolved:
`eos=wright`, `barotropic_solver=implicit_cn`, `vorticity_scheme=al81`,
`momentum_time_integrator=rk3` (Shu-Osher SSP, a THIRD lane distinct from both
`rk3_ws` and `leapfrog`), `outer_integrator=forward_euler`,
`physics.vertical_mixing.scheme=tke` (NOT the "kpp (run-dependent)" the
recipe's own block-mapping table implies for an untouched card — production
overrides it via `--tripole-vmix tke`).

**Per-row corrections/confirmations** (full evidence and the ranked work list
in `orca1_ card_executed_arms.md`):

- **S-16, S-18, S-19**: re-verified independently — CONFIRMED still correct.
  S-18's "NEEDS_ORCA1_OPERANDS" framing is refined: the PRIMARY blocker for
  ORCA1 is that `momentum_time_integrator="rk3"` (not `rk3_ws`) and
  `outer_integrator="forward_euler"` (not `leapfrog`) mean ORCA1 calls NEITHER
  known caller of this row's routine at all — the missing-mesh-operand gap
  would only matter if a caller in ORCA1's own lane existed. What ORCA1
  actually uses for r3u/r3v-equivalent face thickness in its own lane is
  UNVERIFIED.
- **S-29 corrected** (not merely confirmed): this doc's §3 called "ORCA1
  resolves `ln_hpg_zps`" an inference, labelled PLAUSIBLE. Reading
  `namelist_cfg` directly shows NEMO ORCA1 sets `ln_hpg_sco=.true.` (the only
  `ln_hpg_*` line present), NOT `ln_hpg_zps`. The already-implemented
  `nemo_sco` arm (`_bc_ke_and_pressure_gradients`, used by D/L/O/G) is exactly
  the arm ORCA1 should run; it is blocked only by `run_omip_core2.py:4570`'s
  `--pgf-scheme choices=[adcroft,smc03]`.
- **S-35, S-42, M-01**: re-verified — CONFIRMED, no correction. S-35 and M-01
  are rk3_ws-only / leapfrog-only and structurally NOT_EXECUTED for ORCA1 at
  all (a stronger, ORCA1-specific statement this doc's own text for those rows
  did not make). S-42's existing ORCA1 text is CONFIRMED and additionally now
  cross-checked against the real namelist (`ln_trabbl=T`, `nn_bbl_adv=2`).
- **Two findings new to this audit, not previously in this doc or the
  registry, NOT added to either** (out of this task's scope): (1)
  `barotropic_implicit_latlon_cgrid.py` — the module ORCA1 actually
  dispatches to — runs its own uncited forward-backward "avg"-style
  barotropic Coriolis unconditionally, a third, unregistered implementation
  of S-17's quantity. (2) ORCA1's production TKE closure resolves all 10 of
  S-07's structural sub-selectors to the generic/Veros default, not DINO's
  `nemo_literal` set, despite `orca1_zdftke_config` being built as a literal
  `&namzdf_tke` namelist-value transcription.
- **19 real NEMO namelist switches read directly for the first time**
  (`ln_teos10`, `ln_dynadv_vec`, `ln_dynvor_een`, `ln_hpg_sco`,
  `ln_dynspg_ts`, `ln_zad_Aimp`, `ln_zdftke`, `ln_zdfevd`, `ln_traadv_fct`
  +`nn_fct_h/v=2`, `ln_traldf_lap/_iso/_msc`, `nn_aht_ijk_t=21`, `ln_ldfeiv`
  +`nn_aei_ijk_t=21`, `ln_trabbl`+`nn_bbl_adv=2`, `ln_qsr_rgb`,
  `ln_dynldf_lap/_lev`) upgrade several of this doc's PLAUSIBLE claims to
  CONFIRMED and correct one (S-29). Full citations in
  `orca1_card_executed_arms.md`.

## S-19 arm result (2026-09-02, branch `fidelity/nemo-wzv-generic-operands`)

Predictions were registered first in
`docs/ocean/fidelity/testcases/nemo_testcases_l1_wzv_arm_preregister.md`
(commit `25f738583`), which also carries the full operand table and the NEMO
call-site inventory. Outcome, all fp64:

**1. The operand wall is gone.** `nemo_qco_wzv_operands` and
`nemo_qco_kmm_velocity_cycle` now resolve NEMO's `hu_0`/`hv_0`/`e1e2t`/
`e1e2u`/`e1e2v`/`e2u`/`e1v` through one `nemo_qco_resolved_mesh_operands`:
NEMO's own arrays verbatim when a card carries `mesh_mask.nc`, otherwise
rebuilt from the card's grid and reference ladder by the SAME
`nemo_qco_card_mesh_operands` that `_nemo_ws_qco_stage_faces` uses (no second
copy). LOCK_EXCHANGE, OVERFLOW and ORCA1 can now select the NEMO arm.

**2. A second wall was found, and it is NEMO's own.**
`zad_qco_evaluation='nemo_literal'` requires
`vertical_momentum_scheme='nemo_advective'`, which is itself refused with
`adaptive_implicit_vertadv=True`. Those selectors gate NEMO's MLF/`dynzad`
call sites (`stpmlf.F90:227,270`). The L1 cards run NEMO's RK3 flux-form lane
(`dynadv_up3` + `ln_zad_Aimp`, where `dynzad` is dead), whose `wzv` call is
`stprk3_stg.F90:297`. So the row's two impls were never selectable against
each other by those selectors on these cards; the arm below was run at the
RK3 stage call site through the private
`_NEMOWSRK3TestHooks.literal_stage_wzv` control.

**3. The branch is arithmetic association, nothing else.**

| | LOCK_EXCHANGE-zco | OVERFLOW-zps |
|---|---|---|
| predicted bound on max abs dw (kt=1 state) | <= 1.2e-20 m/s | <= 5.2e-18 m/s |
| measured max abs dw, stage 1/2/3 | 2.4e-21 / 9.5e-21 | 3.0e-18 / 5.9e-18 / 3.0e-18 |
| relative to max abs stage w | ~1e-15 | ~1e-15 |
| kt=2 `T` / `u` / `ssh`, generic -> literal | 1.6277e-13 / 2.1388e-10 / 4.78e-28 -> unchanged | 1.1191e-14 / 2.5988e-07 / 1.0492e-14 -> unchanged |
| kt=10 `T` / `u` / `ssh` | 3.00e-12 / 2.07e-10 / 3.58e-13 -> unchanged | 7.71e-08 / 2.64e-05 / 9.24e-05 -> unchanged |
| every kt=1..10 trajectory row | BIT-IDENTICAL | BIT-IDENTICAL |

Non-vacuity, run in the same probe: the literal branch was entered 3 times per
step on both cards, its stage `w` is NOT bit-identical to the generic one, and
a planted +1e-6 scaling inside it moves kt=2 `T` by 5.1e-9 (OVERFLOW) /
3.2e-11 (LOCK). So the null is a measurement, not a dead switch.

**4. P6 — DINO is untouched.** 5-day `nemo_dino_kamm_mlf` twin, 160 leapfrog
steps, `LEGOESM_NEMO_E3T=both`, CPU fp64, byte-identical invocation, one
variable (the commit): base `646415f02` vs HEAD `267c7b673`.

```
33 keys, worst numeric array difference = 0.000000e+00
non-identical keys: ['producer_git_sha']    (646415f02... -> 267c7b673...)
sha256 3cbfa0fe50911608a4fb04cfe089b8a5884989bb6662933f64f753bc8e5742ab  BASE
sha256 0ed4963776f53b3c4a492289ec8e94aef5317b5e64bb127777589efe4ad5086f  HEAD
```

Same blind spot as `dino_reach_check.md`: the archive stores surface slices in
float32, so this resolves a base-vs-HEAD difference only to ~1e-7 relative and
only at the surface. What carries the claim is that pairing plus the raw
branch's operand-for-operand identity, not the surface fields alone.

**5. Stage sweep, and the profile of the debt this row does not own.** The
phase-3 stage sweep re-run at HEAD reproduces the certified faithful row
exactly (`kt1.stage3.faithful.instantaneous_u = 2.598797930308122e-07`, the
same 16 digits as before the refactor). It also shows where the debt is born:

| OVERFLOW-zps kt=1, faithful arm | stage 1 | stage 2 | stage 3 |
|---|---|---|---|
| instantaneous u vs NEMO Kaa | 6.502e-15 | 9.433e-11 | 2.599e-07 |

so the kt=2 velocity residual is a STAGE-3 event, four orders above stage 2.
`wzv` runs identically at all three stages, which is a second, independent
reason it is not the owner.

P1-P6 all CONFIRMED. **The S-19 branch is exonerated as an owner of OVERFLOW's
kt=2 velocity debt** (Rule 4: the arm moves the metric by exactly zero, eight
orders below the debt).

**Correction to the preregistration's stated bound.** Section 4 of the
preregister derives its bound for the FREE-SURFACE term only; it does not cover
the second difference between the arms, which is that the literal path executes
NEMO's divide-by-`e3t`-then-multiply-back literally while the generic path
cancels it algebraically. That channel is closed empirically instead, by
diffing the full 3-D stage `w` arrays: measured 3.0e-18..5.9e-18 m/s, the same
order as the free-surface bound, so the conclusion is unaffected. Raised by the
mechanism reviewer; recorded here rather than silently folded in.

**Disposition.** The arm was NOT promoted to unbranched behaviour, because the
condition for that was "confirmed as an improvement", and it is not an
improvement — it is indistinguishable. Collapsing the row is still the right
end state under "one NEMO routine, one legoESM implementation", but it changes
executed arithmetic on certified cards at the 1e-18 level and therefore is a
one-line ASK, not a silent default move. OPEN.

### Review

Two independent adversarial reviewers, both on Claude-authored code (the codex
CLI is unavailable on this account, so the standing codex+GLM pair was served
by two fresh subagents with disjoint briefs -- stated rather than implied).

*Diff reviewer.* One Critical: the operand resolver had no guard for a
z-coordinate carrying SOME but not all of the eight raw NEMO fields, so a
half-attached bridge would have stopped raising and silently rebuilt every
operand from the card. Fixed in `03f0a1a07` (found independently before the
review returned; the reviewer confirmed the fix). One Important: the new
operand-source test asserted only on array SHAPES and could not fail, and its
fixture's monotone floor made the min-rule face coincide with `e3t_0` anyway.
Fixed in `267c7b673` (ridge fixture, real assertions). Verified clean: raw-path
bit-identity, the `_nemo_ws_qco_stage_faces` refactor, the arm's Kbb/Kaa time
levels, differentiability, no dangling references.

*Mechanism reviewer.* CONFIRMED all four claims -- the same-weighting algebra
(`r3t = ssh*r1_ht_0`, `domqco.F90:160,209`, giving the identical
`H_below(k)/ht_0` fraction), the scaling bound (with the correction recorded
above), the exoneration, and the call-site structural finding. No REFUTED or
UNDECIDED items.

## 2026-09-02 S-35 arm result — the stage-3 `zub` reordering is REFUTED as an owner

Preregistered first, in
`docs/ocean/fidelity/testcases/nemo_testcases_l1_stage3_zub_preregister.md`
(commit `e57b810fd`, before any arm ran).  fp64
(`PrecisionPolicy.fp64()`, `JAX_ENABLE_X64=1`, dtypes printed and asserted
`float64`), CPU, legoESM `e57b810fd`, clean tree.

### Three corrections to this doc's own NEMO description of S-35

Read off `nemo_5.0.2/src/OCE/stprk3_stg.F90` directly, not inferred:

1. The correction runs at **ALL THREE stages**, not stage 3 only — the `:433`
   banner reads "All stages: correct the barotropic component ... at Kaa =
   N+1/3, N+1/2 or N+1" and the `DO_2D`/`DO_3D` block at `:439-446` sits
   OUTSIDE any `kstg` guard.  legoESM's per-stage `_replace_stage_mean` at
   stages 1 and 2 therefore HAS a NEMO counterpart; §2 item 6's phrase "L/O do
   the correction inside the stage ladder" implied it did not.
2. `dyn_zdf` is called at **stage 3 only** (`:430`, `IF( kstg == 3 )`), so the
   before/after ordering question is vacuous at stages 1 and 2 and exists at
   stage 3 alone.
3. The weights are the **reference** thicknesses `e3u_0` / `r1_hu_0` (`:440`),
   not `e3u(Kaa)`.  The update at `:444` is an ADDITIVE column-uniform shift,
   so the correction can only move the `e3u_0`-weighted depth mean and cannot
   touch the baroclinic anomaly at all.

### Scaling, computed before the arm (from oracle dumps + the faithful arm)

Because the two orders differ by a column-uniform shift, the reorder's
leverage is bounded by the DEPTH-MEAN part of the residual:

| card | stage-3 u residual | depth-mean part | baroclinic part | mean / full |
|---|---|---|---|---|
| OVERFLOW-zps | `2.598797930308122e-07` | `4.753142e-15` | `2.598797978e-07` | `1.83e-08` |
| LOCK_EXCHANGE-zco | `2.138804128921056e-10` | `1.929880e-17` | `2.138803936e-10` | `9.02e-08` |

### Arm: one variable, `nemo_stage_mean_imposition=True`

Asserted at run time to be the ONLY differing `BarotropicConfig` field; the
public cards, `recipes.py` and every selector are untouched.

| card | u movement | T movement | SSH movement | stage-3 / kt=2 u residual before -> after |
|---|---|---|---|---|
| OVERFLOW-zps | `1.387779e-17 m/s` | `0.0 K` | `0.0 m` | `2.598797930308e-07` -> `2.598797930169e-07` |
| LOCK_EXCHANGE-zco | `1.292470e-26 m/s` | `0.0 K` | `0.0 m` | `2.138804128921e-10` -> `2.138804128921e-10` (bit-identical) |

kt=2 `T` residual `2.238210e-13 K` (OVERFLOW) / `4.883205e-12 K` (LOCK) and
kt=2 SSH residual `1.049161e-14 m` / `4.782138e-28 m` are unchanged in every
digit.  All eight frozen predictions Z1-Z8 **MET**; the refute threshold
(`2.6e-08 m/s`) is missed by nine orders of magnitude.  **REFUTED.**

### Proof the arm's path executed, and that it did not perturb a zero

A roundoff-sized movement proves nothing unless the branch ran.  Wrapping
`_fixed_depth_means` (the site-(b) helper) and counting calls per step:

| card | faithful | arm |
|---|---|---|
| OVERFLOW-zps | 0 calls | **2 calls** |
| LOCK_EXCHANGE-zco | 0 calls | **2 calls** |

and the quantity the block perturbs is not a zero — the pre-solve depth mean
is `4.502970e-02 m/s` (OVERFLOW) and `1.135367e-03 m/s` (LOCK).  The
correction the block actually applies, `_du = mean_baro - mean_now`, i.e.
exactly the depth-mean shift the implicit vertical solve introduces, is
`8.673617e-19 m/s` (OVERFLOW) and `3.101927e-26 m/s` (LOCK) — a relative shift
of `1.9e-17`, about `0.09 x` fp64 eps.

**Mechanism, now measured rather than argued:** with `A_v = 1e-4`, zero bottom
drag, no implicit surface stress and no-flux boundary conditions, the implicit
vertical momentum solve conserves the thickness-weighted column integral of
`u` to machine precision.  A conserved depth mean commutes with a
column-uniform shift, so applying `zub` before or after `dyn_zdf` is a no-op on
these two cards.

### Consequences

- The OVERFLOW kt=2 velocity debt `2.598797930308122e-07 m/s` is **100%
  baroclinic** and remains **UNOWNED**.  S-35 is now explicitly EXONERATED for
  it; do not re-chase this site.
- §2 item 6's risk note "affects L/O gates only when the implicit solve
  actually shifts the depth mean (A_v=1e-4, so small but nonzero)" is
  **quantified and superseded**: the shift is `8.7e-19 m/s`, i.e. inert.
- S-35 stays **ARTIFICIAL_BRANCH** with both sites and its registry baseline
  entry, because the collapse was gated on the arm confirming.  Its
  COLLAPSIBLE_NOW classification in the summary table is unchanged and is now
  known to be **numerically free** on L/O (movement `<= 1.4e-17 m/s`) — but
  collapsing is a behaviour change on every `rk3_ws` card and was not
  authorised by this task, so it is left as a pending ASK, not done.

## 2026-09-02 S-30 / S-12 collapse — the WS momentum ladder is now written once

§2's rank-3 item is DONE, and S-12 with it (they are the same duplication seen
from two altitudes).  Both rows move `ARTIFICIAL_BRANCH -> SHARED` in §1 above
and in `tests/ocean/unit/_nemo_branch_isomorphism_baseline.py`; neither carried
a baseline entry (both sites were inline code inside `_step_impl`, not
AST-distinct symbols), so the shrink-only allow-list is unchanged.

**What was deleted.**  The pre-barotropic copy of the Wicker-Skamarock stage
recurrence, at what was `omlc:4253-4335`.  Its only consumer was the barotropic
solve's velocity seed.  `u_star, v_star = u0, v0` replaces it (`omlc:4448`).

**Why NEMO has nothing there.**  `stp_2D` evaluates the Kbb RHS ONCE
(`stp2d.F90:126-171`), depth-means it into `Ue_rhs`/`Ve_rhs` (`:177-186`), and
calls `dyn_spg_ts( kt, Kbb, Kbb, Krhs, uu, vv, ... )` at `:280-281` — `Kmm ==
Kbb`, the BEFORE velocity, no stage in between.  legoESM already carried that
depth mean separately in `F_slow_u`/`F_slow_v`.

**What survives.**  One ladder, `omlc:4817-5025`, after the barotropic solve,
where it can use the actual external-mode velocity — NEMO's own placement
(`stprk3_stg.F90:344-374`, with the `zub` replacement at `:433-446`).  The
private `stage_barotropic_correction` hook moved INSIDE `_replace_stage_mean`:
it now ablates the per-stage external-mode replacement instead of selecting a
second recurrence.

**Cost removed.**  Two `tendencies()` evaluations per step on every `rk3_ws`
card, and the failure mode that put the stage vertical-UP3 and live-HPG
operand fixes into one copy only.

**Measured** (fp64, CPU, collapse commit `28df515a8` vs `9070cf276`; full
tables in `testcases/nemo_testcases_l1_phase3_receipt.md`):

| | LOCK_EXCHANGE-zco | OVERFLOW-zps |
|---|---|---|
| certified rows compared, kt=1..10 | 50 | 50 |
| rows bit-identical | 37 | 37 |
| largest move | 0.006 ulp | 0.500 ulp |
| every T and S row, kt=1..10 | bit-identical | bit-identical |
| `first_over_bar` | `kt=2 {T,u}`, unchanged | `kt=2 {T,u,ssh}`, unchanged |
| stage sweep, `faithful` arm | PASS, max 0.001 ulp | PASS, max 0.094 ulp |
| DINO 5-day twin | worst array difference `0.000000e+00` | — |

The bar is a machine check, not a judgement: `MAX_ULP_MOVE = 2` in
`legoesm.ocean.fidelity.ulp_move_gate`, reached as `--compare-to` on both
phase-3 gates, with a `--compare-plant-ulps 3` control that must turn it red
(measured: it does, exit 1).

**One arm is deliberately redefined.**  `omit_stage_primary_velocity_correction`
moves 5.1e12 / 2.0e14 ulps, because the hook it drives no longer means "skip
the post-solve ladder" (there is only one ladder) but "keep each stage's own
depth mean".  Every OTHER sweep arm — `faithful`, the three `freeze_stage_hpg`
arms, `omit_stage_vertical_up3`, `legacy_velocity_primary_average`,
`legacy_stage_min_face_thickness`, `omit_stage_qco_factor`,
`omit_momentum_transport_reconcile` — is inside the bar at <= 0.125 ulp, so
the `.faithful.` filter used for the sweep comparison hides nothing.

**Not claimed.**  The OVERFLOW kt=2 velocity debt is unchanged at
`2.5987979300999553e-07 m/s` and stays UNOWNED — this collapse is exonerated
for it, as S-35 already was.  The 6120-step statistical scorer was not re-run.
GYRE, the third `rk3_ws` card, has no gate on this branch and was not run.

**Left standing, named:** with one ladder, the tracer program's legacy
stage-transport rebuild (its `_nemo_ws_live_stage_geometry is None` branch)
is unreachable for every `rk3_ws` card, because the config validator couples
the momentum and tracer integrators.  It is dead code, not a second NEMO
implementation, and its removal is a separate one-line change with its own
gate.

### Same-day corrections to the section above, after dual adversarial review

Full detail and numbers in `testcases/nemo_testcases_l1_phase3_receipt.md`
("RETRACTION and correction"). Three things in the table above are narrower
than they read:

1. **"every T and S row bit-identical" is a claim about the gate's REDUCTIONS,
   not about the tracer FIELDS.** Measured per cell with the new committed
   probe `nemo_testcase_state_ulp_probe.py`: OVERFLOW's T field is
   bit-identical for six steps and then departs at five cells, reaching
   `1.07e-14 K` at twelve cells by ten steps, while its T row never moves. LOCK
   T/S/v are bit-identical throughout; OVERFLOW S/v are too. This is
   unavoidable in principle — once velocity moves, advected tracers follow —
   and it is the gate's own declared blind spot firing in practice.
2. **The "pure re-association" claim is certified for f = 0 only.** Both L1
   cards are built with `f0=0.0, beta=0.0`, so the Matsuno rotation sitting
   between the deleted ladder and the barotropic seed is the exact identity and
   the seed's 3-D structure has no channel into the solve. On a ROTATING
   `rk3_ws` card the four-point Coriolis average of a zero-`h_v`-mean field does
   not have zero `h_u`-weighted mean where column weights vary, so the seed's
   depth mean would change at `O(dt^2 f)` — first order. GYRE is rotating and
   ungated here: on GYRE this is an UNMEASURED behaviour change, not a
   re-association.
3. **The 6120-step statistics are UNMEASURED, not "expected to hold".** The
   growth fits cannot separate exponential from polynomial on a four-point
   tail.

The bar itself has two defensible readings that disagree on the verdict: 2 ulps
at unit scale (what the gate implements) gives a worst move of 0.500 and
ADMITS; 2 ulps of the row's own field scale gives 3.0 on one OVERFLOW stage-1
row and REFUSES by one ulp. That choice is left to the user; the gate constant
was not touched, and the looseness is now written into the module's blind-spot
list.
