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
| momentum_flux_scheme | upwind (inert) | nemo_up3 | nemo_up3 | n/a | upwind (inert) |
| vertical_momentum_scheme | nemo_advective | nemo_up3 | nemo_up3 | nemo_advective | **upwind_perturbation** |
| vorticity_scheme | een_total | **al81** | **al81** | ene_total | **al81** |
| coriolis_scheme | explicit_ab2 | **matsuno_split** | **matsuno_split** | explicit_ab2 | **matsuno_split** |
| pgf_scheme | nemo_sco | nemo_sco | nemo_sco | nemo_sco | **smc03** |
| pgf_quadrature | nemo_trapezoid | nemo_trapezoid | nemo_trapezoid | nemo_trapezoid | **cell_integral** |
| lateral_viscosity_operator | nemo_div_curl | vector_laplacian (A_h=0) | vector_laplacian (A_h=0) | nemo_div_curl | **vector_laplacian** |
| lateral_viscosity_e3_weighting | **off** | off (inert) | off (inert) | nemo_e3 | off |
| adaptive_implicit_vertadv | False | True | True | False | True |
| zdf_implicit_solver_evaluation | nemo_literal | nemo_literal | nemo_literal | nemo_literal | **shared_thomas** |
| ~~implicit_vmix_e3t_now_divisor~~ (DELETED 2026-09-02; the NEMO divisor comes with `zdf_implicit_solver_evaluation="nemo_literal"` above) | n/a | n/a | n/a | n/a | n/a |
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
| S-19 | `wzv` sshwzv.F90 (np_velocity / np_transport) | none (arg-level) | D L O G A | `vertical.py diagnose_w_from_flux_div` (generic); `opl nemo_qco_wzv_operands` (literal, incl. `nemo_qco_kmm_velocity_cycle`) | `wzv_call2_evaluation` (`generic`/`nemo_literal`) | ARTIFICIAL_BRANCH — one NEMO routine, two impls; D/G literal, L/O/A generic. **MEASURED 2026-09-02** — the branch is arithmetic association only: stage `w` differs 3.0e-18 m/s (OVERFLOW) / 2.4e-21 m/s (LOCK), every kt=1..10 row BIT-IDENTICAL, and it does NOT own OVERFLOW's 2.599e-7 kt=2 velocity debt. **GYRE 2026-09-03:** the literal transport call now consumes the already materialized stage `pFu/pFv`, exactly as `traadv.F90:220-226`; the private legacy-rebuild hook is diagnostic only. Operand wall REMOVED (`nemo_qco_resolved_mesh_operands`); the remaining wall is NEMO's own call-site split. See "S-19 arm result" below |
| S-20 | `wAimp` sshwzv.F90 (adaptive-implicit w split) | `ln_zad_Aimp` | L O A | `vertical.py:52 nemo_wicker_aimp_partition_transport` (called by `_nemo_ws_stage_transport` per stage) | `adaptive_implicit_vertadv` | SHARED (one impl). OVERFLOW now carries the source 20 m `e3w_0` operand (`usrdef_zgr.F90:157-168`); the former midpoint is private-test-only. ORCA1 consumes the kernel at a different post-program site — composition differs |
| S-21 | stage transport `zFu/zFv/zFw` + `zub` correction stprk3_stg.F90:257-304; stage clocks at :123-124,177-178,221-222 feed `sshwzv.F90:334-335` | none (RK3 identity) | L O G | `omlc:_nemo_ws_stage_transport` + `_nemo_stage_corrected_velocity` + `_nemo_metric_stage_transport`; coupled QCO reciprocal from `vertical:nemo_qco_live_face_geometry_from_operands` | no public selector; private legacy 3-D re-reduction ablation and Round-21 `source_stage_wzv_clock_arm` measurement only | SHARED for the tracer stages and all three momentum stages — momentum stage 1 advects with the same Kmm transport (`zub`) as stages 2-3; `zub/zvb` consume NEMO's separately stored `uu_b/vv_b(Kmm)` and the stored `r1_hu_0/(1+r3u)` / `r1_hv_0/(1+r3v)` from `domqco.F90:175-181,219-222`, rather than re-deriving `1/SUM(e3_face(Kmm))`.  The metric-bearing products remain materialized as `(e2u*e3u)*(uu+zub*umask)` / `(e1v*e3v)*(vv+zvb*vmask)` instead of re-reduction or divide/remultiply regrouping.  ORCA2's non-uniform partial-cell discriminator closes zFu/zFv from `102431/108307` unequal to `0/0` (Round 22). The stored-mean change remains **HOLD** beyond kt=1 pending prognostic `uu_b/vv_b` design; the stage-W clock claim was refuted by the admitted Round-21 records (`tests/ocean/unit/test_nemo_ws_stage1_transport.py`, `test_nemo_ws_stage_transport_preserves_fortran_product_association`, `test_nemo_ws_stage_corrected_velocity_matches_oracle_bits`; GYRE Round-20/21/22 receipt) |
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
| S-34 | `trazdf`/`dynzdf` gradient divisor `e3w(Kmm)` trazdf.F90:219-221, dynzdf.F90:182-195 | none | D L O G (all, unbranched) | ONE symbol: `implicit_solver.py::nemo_e3w_kmm`, called by the tracer solve AND the momentum solve; OVERFLOW's U/V divisor uses its raw 20 m `e3uw_0/e3vw_0` with `r3u/r3v` (`domzgr_substitute.h90:131-133`) | none — the divisor comes with the NEMO identity `zdf_implicit_solver_evaluation="nemo_literal"`; midpoint restores exist only as private causal hooks | **SHARED (operand-complete 2026-09-03)** — OVERFLOW no longer silently reconstructs `e3w_0` from a partial T-cell midpoint; every NEMO card reaches the same implementation and its reference operand |
| S-35 | stage-3 barotropic correction `zub` stprk3_stg.F90:433-446 (AFTER `dyn_zdf` at :430) | none | L O G | (a) `omlc:4707 _replace_stage_mean` — applied per stage BEFORE the implicit solve; (b) `omlc:6234 _impose_mean` + `_fixed_depth_means` — re-imposed AFTER the implicit solve | (b) gated by `barotropic.nemo_stage_mean_imposition` | ARTIFICIAL_BRANCH — two sites for one NEMO step; GYRE has both, L/O only (a), so L/O apply the correction on the wrong side of `dyn_zdf` |
| S-36 | `tra_adv_trp` stprk3_stg.F90:463,494 (reuse the momentum zF triplet) | none | L O G | `omlc:5226` reuses `_nemo_ws_live_stage_geometry`; legacy rebuild at `omlc:5240` | private hook `kmm_tracer_transports` | SHARED |
| S-37 | `tra_adv` dispatch traadv.F90:355-364; FCT at stage 3 only (`ll_dofct`, :279-283) | `ln_traadv_fct`, `nn_fct_h/v=2` | D L O G / A | `omlc:1093 _nemo_ws_rk3_tracer_pair_step` (stages 0-1 -> `"centered"`, stage 2 -> `fct2`); `omlc:709 compute_advection_flux_div_pair`; `advection.py:828 fct_tracer_advection` | `tracer_advection`, stage keying is hard-wired (no selector) | SHARED for D/L/O/G. ORCA1 runs `superbee` = **ARTIFICIAL** (`traadv_mus`/`ubs`/`qck` are ABSENT; superbee is Veros's) |
| S-38 | `tra_adv_fct` implicit-w treatment traadv_fct.F90:143,439-453 (`nn_fct_imp=1` -> `ll_zAimp1`) | `nn_fct_imp` | L O G | `advection.py:925-1010` `low_order_predictor` (`one_step`/`nemo_rk3_two_step`) + `fct_implicit_w` | `fct_low_order_predictor` (private hook `two_step_fct_predictor`) | UNVERIFIED — did not trace `ll_zAimp1` line-by-line against `nemo_rk3_two_step` |
| S-39 | tracer stage time-stepping (qco `(1+r3t)` weighting) stprk3_stg.F90:540-560 | `key_qco` | L O G | `omlc:1214 _stage` inside `_nemo_ws_rk3_tracer_pair_step` (uses `h_one_third`/`h_one_half`) | none | SHARED |
| S-40 | `tra_sbc` / `tra_sbc_RK3` trasbc.F90; stprk3_stg.F90:521 | `nn_fsbc` | D G A | `physics/surface_forcing/*`; DINO `surface_tendency_placement` (`applied_now`/`leapfrog_rhs`) | `surface_tendency_placement` (DINOConfig only) | ARTIFICIAL_BRANCH — a DINO-scoped selector for a NEMO site every card has; RK3 cards have no equivalent |
| S-41 | `tra_qsr` traqsr.F90 | `ln_traqsr`, `ln_qsr_2bd`/`ln_qsr_rgb` | G A | `physics/shortwave_penetration.py:443 apply_shortwave_penetration` | `shortwave_penetration.scheme` (`jerlov_2band`/`rgb_chl`) | NEMO_SWITCH |
| S-42 | `bbl` + `tra_bbl` trabbl.F90:129-136,243-284; stprk3_stg.F90:468,498,588 | `ln_trabbl`, `nn_bbl_adv=2` | O A | `physics/bbl_adv.py:176 bbl_transports` + `:232 apply_bbl_adv_tendency` — called in-stage at `omlc:1188` (OVERFLOW) **and** via `physics/bbl_adv.py:312 apply_bbl_adv_step` from `run_omip_core2.py:8420` (ORCA1, host post-step Euler + an extra transport cap NEMO has none of) | `bbl_adv_option` vs driver `--bbl-adv` | ARTIFICIAL_BRANCH — ONE transcription, two composition SITES. **2026-09-02: the clamp is DELETED** and `apply_bbl_adv_step` now adds no arithmetic of its own; the remaining branch is PLACEMENT and is NOT config-flippable (see the S-42 addendum) |
| S-43 | `tra_ldf` traldf_iso.F90 | `ln_traldf_lap`+`ln_traldf_iso` | D G A | `physics/lateral_mixing/gm_redi_latlon_cgrid.py` | `lateral_tracer_mixing`, `gm_redi.*` | see S-09 |
| S-44 | `tra_zdf` trazdf.F90 | always | D L O G A | `omlc:7537 _apply_implicit_vertical_mixing` (same fn as `dyn_zdf`) | `zdf_implicit_solver_evaluation` | see S-33/S-34 |
| S-45 | `tra_npc` tranpc.F90 | `ln_zdfnpc` | — | none | — | ABSENT (no card selects it) |
| S-47 | `dyn_adv_up3` face THICKNESS: `zFu = e2u*e3u(Kmm)*uu` (dynadv_up3.F90:160), divisor `e3u(Kmm)` (:205-207), `e3u(Kmm) = e3u_0*(1+r3u(Kmm))` (domzgr_substitute.h90:127, domqco.F90:219-220); under WS-RK3 the same thickness in the stage transport (stprk3_stg.F90:273) | none (one macro) | L O | `opl:_bc_horizontal_momentum_advection_flux_form` consumes `momentum_flux_face_thickness` = the stage pair from `omlc:_nemo_ws_qco_stage_faces` (-> `vertical.nemo_qco_live_face_geometry_cgrid`, the S-18/S-21 kernel) at all four WS-RK3 `tendencies()` sites; every other caller keeps `tendencies()`' own `min_cell_to_uface(h_k)` | none public; private `_NEMOWSRK3TestHooks.legacy_hadv_min_face_thickness` (gate ablation arm only) | SHARED on the WS-RK3 cards (**LANDED 2026-09-02**, `tests/ocean/unit/test_nemo_ws_hadv_face_thickness.py`); the min-of-STRETCHED-T rule was a THIRD construction of `e3u(Kmm)` (after S-18's seed and S-21's transport), first order wrong in the ssh difference across the face; owner of the OVERFLOW kt=2 stage-3 `u` remainder, the stage-2 residual and the `slow_u` debt (receipt `nemo_testcases_l1_stage3_remainder_receipt.md`) |

| S-46 | `dyn_adv_up3` face THICKNESS: `zFu = e2u*e3u(Kmm)*uu` (dynadv_up3.F90:160), divisor `e3u(Kmm)` (:205-207), `e3u(Kmm) = e3u_0*(1+r3u(Kmm))` (domzgr_substitute.h90:127, domqco.F90:219-220); under WS-RK3 the same thickness in the stage transport (stprk3_stg.F90:273) | none (one macro) | L O | `opl:_bc_horizontal_momentum_advection_flux_form` consumes `momentum_flux_face_thickness` = the stage pair from `omlc:_nemo_ws_qco_stage_faces` (-> `vertical.nemo_qco_live_face_geometry_cgrid`, the S-18/S-21 kernel) at all four WS-RK3 `tendencies()` sites; every other caller keeps `tendencies()`' own `min_cell_to_uface(h_k)` | none public; private `_NEMOWSRK3TestHooks.legacy_hadv_min_face_thickness` (gate ablation arm only) | SHARED on the WS-RK3 cards (**LANDED 2026-09-02**, `tests/ocean/unit/test_nemo_ws_hadv_face_thickness.py`); the min-of-STRETCHED-T rule was a THIRD construction of `e3u(Kmm)` (after S-18's seed and S-21's transport), first order wrong in the ssh difference across the face; owner of the OVERFLOW kt=2 stage-3 `u` remainder, the stage-2 residual and the `slow_u` debt (receipt `nemo_testcases_l1_stage3_remainder_receipt.md`) |
| S-48 | `stprk3_stg` stage velocity MASK RANK: `uu(Kaa) = (...)*umask(ji,jj,jk)` (stprk3_stg.F90:367 for `ln_dynadv_vec .OR. lk_linssh`, :375 for the compiled `key_qco` branch, :382 for the `#else`), the barotropic correction `uu(Kaa) = uu(Kaa) + zub*umask(ji,jj,jk)` (:444), and the SAME correction inside the advective transport `zFu = e2u*e3u(Kmm)*( uu(Kmm) + zub*umask(ji,jj,jk) )` (:273-274) -- so `uu` is EXACTLY zero below the seabed and `dyn_adv_up3`'s k-slab stencil reads that zero rather than skipping a dry neighbour (dynadv_up3.F90:142-143, :160, :166-176) | none (one array; RK3 identity) | L O | `omlc:_replace_stage_mean`, its sibling `_transport_stage`, and `omlc:_mom_pert_ws`'s transport construction all mask with the 3-D live face mask from the shared `latlon_cgrid_operators.compute_face_masks_3d` -- the same pair the S-46 stage face-thickness kernel consumes | none public; private `_NEMOWSRK3TestHooks.legacy_2d_stage_face_mask` (gate ablation arm only) | SHARED on the WS-RK3 cards (**LANDED 2026-09-03**, `tests/ocean/unit/test_nemo_ws_stage_face_mask_rank.py`); the 2-D `state.u_mask` broadcast over levels left the barotropic depth-mean increment standing at EVERY level below a staircase face's own seabed (OVERFLOW-zps: 0.228 m/s by kt=6, 78% of the card's own wet maximum, exactly constant down the dry column), which the deeper neighbour's wet bottom level then read as an UP3 stencil neighbour (receipt `nemo_testcases_l1_phantom_velocity_receipt.md`) |
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

4. **~~`implicit_vmix_e3t_now_divisor` default keeps the non-NEMO divisor
   (S-34).~~ DONE 2026-09-02 — collapsed exactly as prescribed below.**
   `trazdf.F90:219-221` has one divisor, `e3w(Kmm)`. legoESM's default was a
   midpoint of the AFTER thickness and the certified DINO MLF card ran it.
   *Collapse, as landed:* the midpoint arm is unreachable from any NEMO card;
   `e3w(Kmm)` is unbranched inside the NEMO identity
   `zdf_implicit_solver_evaluation="nemo_literal"` and produced by one shared
   symbol for the tracer AND momentum solves; `implicit_vmix_dzw_slot` stays
   for the Veros oracle only; `implicit_vmix_e3t_now_divisor` is deleted.
   *Realised risk:* the DINO twin's numbers move (5-day arm disclosed in
   `dino_zdf_divisor_arm_receipt.md`); the Y5 certification re-run and the
   20-year member are NOT run (GPU, user's decision). LOCK_EXCHANGE and
   OVERFLOW are bit-identical (their ladders are uniform). ASKED and approved
   before the default moved.

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
| S-16 | `dyn_spg_ts` continuity/transport/spg dynspg_ts.F90:~640-840 (5 selectors) | generic FV form (SM2005-style, unreferenced) vs `nemo_literal`/`nemo_ssh_avg` | nemo arm: DINO, LOCK, OVERFLOW (`_model_config`, this checkout), GYRE (l2); generic: ORCA1 (`run_omip_core2.py` never sets these fields) and one Veros frozen-state tendency probe (deliberate bit-identical default, not a physics claim) | generic names no NEMO/other-oracle reference; nemo arm cites dynspg_ts.F90 | ~~**NEMO_DUPLICATE**~~ **SUPERSEDED 2026-09-02 (USER DECISION): OTHER_RECIPE** — the generic arm is KEPT (it is the dycore of 19 idealized test-matrix experiments via `default_wright_v1`/`legoesm_linear_v1`, plus `legoesm_nemo_like_v1`, renamed from `nemo_v1`) and given a real reference: legoESM's own in-house forward-backward split-explicit design (commit `adbb49f83`, 2026-04-08, #87), whose time-averaging and BEBT closure were explicitly modeled on the MOM6/ROMS family (Hallberg 1997; Shchepetkin & McWilliams 2005) per their own introducing commits. See the dated addendum below the S-16 narrative section and the registry's S-16 row for the mechanical fix | CONFIRMED then, reclassified now |
| S-18 | `dom_qco_r3c` r3u/r3v MEAN vs MIN domqco.F90:140-240 | `min_cell_to_uface`/`_vface` (Adcroft/Hill/Marshall 1997 MOM6/MITgcm convention, CONFIRMED cited) vs `nemo_qco_live_face_thicknesses` (NEMO MEAN) | MEAN: DINO + GYRE (l2); MIN: LOCK, OVERFLOW **on this checkout** (verified `_model_config` never sets `zad_qco_evaluation`/`wzv_call2_evaluation` — this is a live gap on the certified L1 cards too, not ORCA1-only as the source doc's framing implied) + ORCA1 | MIN cites Adcroft et al. 1997 (a real reference, but for a *different* model's convention, not NEMO); MEAN cites domqco.F90 | ~~**NEMO_DUPLICATE**~~ **SUPERSEDED 2026-09-02, post-1d6a7448d: OTHER_RECIPE** — this row's own "verified" gating claim was the error: `zad_qco_evaluation`/`wzv_call2_evaluation` gate `nemo_qco_wzv_operands` (S-19's routine, in `ocean_pe_latlon_cgrid.py`), not `_nemo_ws_stage_transport` (S-18's routine, in `ocean_model_latlon_cgrid.py`), which was never gated by them. See "Triage against HEAD 648e5cd69" below | CONFIRMED then, **RETRACTED** now |
| S-19 | `wzv` sshwzv.F90 | `diagnose_w_from_flux_div` (generic) vs `nemo_qco_wzv_operands` (literal), same selector family as S-18 | same as S-18: DINO/GYRE on literal; LOCK/OVERFLOW/ORCA1 on generic | same as S-18 | **NEMO_DUPLICATE** | CONFIRMED |
| S-25 | vertical momentum advection, non-NEMO arms | `upwind_perturbation` (default) / `centered_full` / `nemo_advective` | upwind_perturbation: unclaimed default, run by most catalog recipes + ORCA1 (no override); centered_full: `veros_faithful_v1`; nemo_advective: DINO + GYRE | centered_full cites Veros `core/momentum.py` (quoted); nemo_advective cites dynzad.F90; default names nothing | **OTHER_RECIPE** (ORCA1 landing on the unclaimed default instead of `nemo_advective` is a driver reachability gap, not a duplicate) | CONFIRMED |
| S-27 | Coriolis time placement | `matsuno_split` (default, unreferenced legacy) vs `explicit_ab2` (in-RHS) | matsuno_split: unclaimed default, incl. ORCA1 (unset); explicit_ab2: `veros_faithful_v1`, `oceananigans_v1`, `mitgcm_v1` (recipes.py) **and** DINO + GYRE (this arm is simultaneously Veros's own scheme AND NEMO's real Coriolis placement) | explicit_ab2 docstring: "VEROS-FAITHFUL" + DINOConfig comment "(MITgcm/Oceananigans/Veros)" + cites dynspg_ts.F90 for GYRE's use; matsuno_split names nothing | **OTHER_RECIPE** (ORCA1 defaulting to matsuno_split while live is the real gap the source doc's own item 9 names — a missing override, not a second NEMO implementation) | CONFIRMED |
| S-29 | `dyn_hpg` on ORCA1 (`ln_hpg_zps`) | `adcroft` (default) / `smc03` / `nemo_sco` (unreachable from ORCA1) | `run_omip_core2.py:4570` `--pgf-scheme choices=[adcroft,smc03]` (nemo_sco absent, CONFIRMED); adcroft: default recipes; smc03: `eady_weno5_v1`; nemo_sco: DINO/LOCK/OVERFLOW/GYRE | adcroft cites Adcroft/Hallberg/Hill 2008 in `pgf_ahh08.py` (state.py's own comment says "Adcroft & Campin 2004" for the same string — citation-year mismatch, flagged, unresolved); smc03 cites Shchepetkin & McWilliams 2003 | **OTHER_RECIPE** (both ORCA1-reachable arms are real, separately-cited non-NEMO papers with genuine recipe consumers; the defect is the driver CLI cannot reach `nemo_sco` at all — a reachability gap, not duplicated NEMO work) | CONFIRMED |
| S-30 | WS momentum ladder written twice stprk3_stg.F90:344-374 | ladder #1 (seeds barotropic solve only) vs ladder #2 (barotropic-corrected, kept) — same recurrence, both always run | only cards with `momentum_time_integrator="rk3_ws"`: LOCK, OVERFLOW, GYRE. No catalog recipe, no ORCA1 (uses `"rk3"`, a third distinct branch) | none — no non-NEMO recipe ever reaches `rk3_ws` | **NEMO_DUPLICATE** | CONFIRMED |
| S-32 | `dyn_ldf`->`ldf_lap` dynldf_lap_blp.F90 | `nemo_div_curl` / `vector_laplacian` (default) / `flux_divergence` | nemo_div_curl: DINO + GYRE; vector_laplacian: unclaimed default incl. ORCA1; flux_divergence: `veros_faithful_v1`, `oceananigans_v1`, `mitgcm_v1` | nemo_div_curl cites "NEMO dyn_ldf_lev_lap" (dino.py:914); flux_divergence cites Veros `core/friction.py harmonic_friction`; vector_laplacian names nothing | **OTHER_RECIPE** (clean 3-way split; side-finding: DINO's `off` vs GYRE's `nemo_e3` e3-weighting sub-flag is a MEASURED deliberate choice — an A/B found `nemo_e3` worsens DINO's gate rows — not an oversight) | CONFIRMED |
| S-33 | `dyn_zdf`/`tra_zdf` implicit solver | `shared_thomas` (default, generic normalized-row solver) vs `nemo_literal` (unnormalized NEMO recurrence) | shared_thomas: unclaimed default incl. all catalog recipes + ORCA1; nemo_literal: DINO, LOCK, OVERFLOW, GYRE | shared_thomas names nobody's oracle fidelity (ordinary numerics); nemo_literal implicitly contrasts against dynzdf/trazdf (F90 line not independently re-derived) | **OTHER_RECIPE** (ORCA1 on shared_thomas instead of nemo_literal is the same reachability-gap pattern as S-25/27/29/32, not duplicated NEMO work) | CONFIRMED (selection facts); PLAUSIBLE (exact F90 citation) |
| S-34 | `trazdf`/`dynzdf` divisor e3w(Kmm) | ONE NEMO implementation (`nemo_e3w_kmm`, tracer + momentum) reached by the NEMO identity; legacy midpoint and Veros `implicit_vmix_dzw_slot` reachable only OFF that identity | NEMO arm: DINO (certified kamm_mlf), LOCK, OVERFLOW, GYRE — all four, unbranched; OVERFLOW supplies the raw 20 m mesh ladder instead of the previously silent partial-cell midpoint; legacy midpoint: unclaimed default recipes + ORCA1; Veros arm: `veros_faithful_v1` | NEMO arm cites trazdf.F90:219-221 + dynzdf.F90:182-195 + usrdef_zgr.F90:157-168 + domzgr_substitute.h90:131-133; Veros arm cites Veros `thermodynamics.py:267`; legacy names nothing | **SHARED (operand-complete 2026-09-03)** — implementation and reference mesh operand now agree on OVERFLOW; private hooks alone reproduce the causal legacy arms | CONFIRMED |
| S-35 | stage-3 barotropic correction `zub` stprk3_stg.F90:433-446 | site (a) `_replace_stage_mean` (before the implicit solve) vs site (b) `_impose_mean` (after the implicit solve) | site (a): every rk3_ws card unconditionally (LOCK, OVERFLOW, GYRE); site (b): gated by `nemo_stage_mean_imposition` (default False) — only GYRE (l2) sets it True; LOCK/OVERFLOW never do | **both comments cite the identical NEMO line range** stprk3_stg.F90:433-446/:440 | **NEMO_DUPLICATE** — GYRE runs both (redundant); LOCK/OVERFLOW run only the wrong-side-of-`dyn_zdf` one | CONFIRMED |
| S-40 | `tra_sbc`/`tra_sbc_RK3` trasbc.F90 | `applied_now` (pre-step mutation, legacy) vs `leapfrog_rhs` (folds into MLF Nnn RHS) | applied_now: DINO's non-MLF oracle sub-recipes (`veros`/`mitgcm`/`oceananigans` DINO variants); leapfrog_rhs: only the certified MLF card (`nemo_dino_kamm_mlf`) — and `run_dino.py:625-634` hard-forbids the buggy pairing (`leapfrog` + `applied_now`) via `SystemExit` | leapfrog_rhs cites tra_sbc.F90 Nnn-RHS placement; applied_now names nothing | **OTHER_RECIPE** — RK3 cards (GYRE/L/O/ORCA1) have no equivalent selector because their own site is hard-wired, not selectable; the one two-arm case (DINO) already has a driver-level guard closing the Rule-3 failure mode | CONFIRMED |
| S-42 | `bbl`/`tra_bbl` trabbl.F90 | `apply_bbl_adv_tendency` (in-stage) vs `apply_bbl_adv_step` (driver post-step Euler; its extra `0.25*V/dt` cap is **DELETED 2026-09-02**) | in-stage: OVERFLOW (`bbl_adv_option=2`); driver-only: ORCA1's `--bbl-adv` flag — which **never sets** `bbl_adv_option` (grep confirmed zero hits), so ORCA1's in-model dispatch stays permanently off at its 0 default while physics runs through the separate driver path instead | **both docstrings cite NEMO trabbl / Campin & Goosse BBL exchange for the same routine** | ~~**NEMO_DUPLICATE** — two independent NEMO-trabbl transcriptions~~ **RETRACTED 2026-09-02**: `apply_bbl_adv_step` CALLS the same two operators, so there is ONE transcription and a PLACEMENT branch; see the S-42 addendum | RETRACTED |
| M-01 | `stp_MLF` whole-step composition stpmlf.F90:108-473 | `_leapfrog_step` (two `_step_impl` passes) vs `_nemo_mlf_step` (one pass, single-pass transcription) | dispatch exists NOW at `outer_integrator` ("leapfrog"->a, "nemo_mlf"->b — contradicts (b)'s own stale docstring claiming it's unwired); DINO's certified card selects `"leapfrog"` (a). No recipe/card selects `"nemo_mlf"`. (b) DOES have real committed test coverage (`tests/ocean/unit/test_nemo_mlf_step_transcription.py`) — correction to source doc, which called it probe-only/dead | both cite stpmlf.F90 `stp_MLF` directly; (b) additionally cites a spec doc | **NEMO_DUPLICATE** (not ORPHAN: (b) names a reference and has real test coverage — it is validated-but-unpromoted, not dead) | CONFIRMED |

### 6.2 Counts

| classification | rows | which |
|---|---|---|
| NEMO_DUPLICATE | 6 (was 7, was 8) | S-12, S-19, S-30, S-35, S-42, M-01 |
| OTHER_RECIPE | 13 (was 12, was 11) | S-03, S-04, S-07, S-09, S-16 (moved here 2026-09-02, see below), S-18 (moved here 2026-09-02, see below), S-25, S-27, S-29, S-32, S-33, S-34, S-40 |
| ORPHAN | 0 | — |

**S-18 moved NEMO_DUPLICATE -> OTHER_RECIPE, 2026-09-02 (post-1d6a7448d)**: the
row above still records this pass's original (CONFIRMED, at the time) verdict
as history. It was retracted the same day — see "Triage against HEAD
648e5cd69" below for the correction and the registry's S-18 row for the
mechanical fix.

**S-16 moved NEMO_DUPLICATE -> OTHER_RECIPE, 2026-09-02 (USER DECISION, this
PR)**: unlike S-18, this is not a retraction of a measurement error — the
duplicate-ness finding stands (two legoESM implementations of one NEMO
program). What changed is the disposition of the generic side: it is kept,
deliberately, as a legitimate non-NEMO fork now carrying its own real
reference (legoESM's own forward-backward split-explicit design, MOM6/ROMS-
informed closure choices) rather than sitting unreferenced. See the dated
addendum in the "S-16 (`dyn_spg_ts` continuity/transport/spg)" narrative
section below and the registry's S-16 row for the mechanical fix.

**Headline correction**: several rows this doc flagged among its top-priority
"mistakes to collapse" — S-27 (Coriolis split), S-29 (PGF), S-33/S-34
(ZDF solver/divisor) — are legitimate Veros/MITgcm/Oceananigans/paper-cited
forks, each carrying a *different* real defect: ORCA1 (and, for S-34, DINO)
simply never selects the NEMO arm it already has, a reachability/hidden-
default bug, not duplicated NEMO implementation work. S-19 (wzv) is a
confirmed genuine duplicate reaching **every** certified NEMO card, including
LOCK/OVERFLOW on this checkout (not just ORCA1, as this doc's own §0 framing
suggested); S-16 (barotropic transport) was in the same shape until the
2026-09-02 user decision above gave its generic side a real reference and
moved it to OTHER_RECIPE — its NEMO side still reaches every certified NEMO
card, that part of the finding is unchanged. **S-18 (qco face-thickness) is NOT**:
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
(`ocean_model_latlon_cgrid.py:1050`), set `True` only inside
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_phase3_stage_sweep_gate.py`,
never by a production card. `min_cell_to_uface`'s real role (MOM6/MITgcm
hFacW=min, `latlon_cgrid_operators.py:277-284`) is the PE-lane
depth-average/slow-forcing (`ocean_pe_latlon_cgrid.py:1364`,
`ocean_model_latlon_cgrid.py:4293`), reached by DINO/LOCK/OVERFLOW/ORCA1 *and*
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
   (`ocean_model_latlon_cgrid.py:457,1154,5340` for the generic diagnostic;
   `ocean_model_latlon_cgrid.py:5534` for the literal path). Gated by
   `state.py:2236 zad_qco_evaluation` / `:2225 wzv_call2_evaluation`, both
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
   (`state.py:1145`).
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
  (`state.py:1418`) and three catalog recipes select it: `default_wright_v1`,
  `legoesm_linear_v1`, `legoesm_nemo_like_v1` (renamed from `nemo_v1`
  2026-09-02). MEASURED (fp64, call counters on the
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

**Revised triage (superseded by the 2026-09-02 USER DECISION below):
NEMO_DUPLICATE (classification UNCHANGED) but BLOCKED for
collapse, not `COLLAPSIBLE_NOW`.** Collapsing S-16 moves the numbers of 19
test-matrix experiments plus the `legoesm_nemo_like_v1` (then `nemo_v1`)
catalog dycore. It does NOT move
ORCA1. Unblocking it needs a separate, explicitly-asked decision about those
19 idealized cases (either re-baseline them on the NEMO arm, or give the
generic FV arm a real reference and keep it as an OTHER_RECIPE fork) — it is
not an ORCA1 re-point.

The NEMO oracle lines are re-confirmed on this checkout, so the row's
duplicate-ness is not in doubt: `dynspg_ts.F90:604-609` (`zhU`/`zhV` metric
transports), `:627` (`zhdiv`), `:641-642` (`un_adv` accumulation), `:682-684`
(`zu_spg`) — ONE NEMO program, two legoESM implementations. The row keeps its
baseline entry.

**2026-09-02 USER DECISION: the "either/or" above is resolved — give the
generic arm a real reference and keep it as an OTHER_RECIPE fork.** No
re-baseline of the 19 idealized test-matrix cases.

*Reference found* (read off the code's own git history, not invented):
the generic arm — `_run_substep_loop`'s continuity divergence
(`divergence_cgrid`), transport accumulation (`Hu_sum += w*flux_u`), PGF
(`-g*gradient_{x,y}_cgrid(eta)`), and seed (the shared stacked reduction) —
is **legoESM's own forward-backward split-explicit C-grid solver**,
introduced with no external citation by commit `adbb49f83` (2026-04-08,
"Add C-grid lat-lon ocean model to fix checkerboard instability", #87). It
is not a transcription of NEMO's `dyn_spg_ts`, nor a literal port of any one
paper's equations. Its later closure choices were explicitly modeled on the
MOM6/ROMS split-explicit family, per those commits' own messages: the
substep time-averaging "follow[s] the standard approach in MOM6, MPAS-Ocean,
and ROMS (Hallberg 1997, Shchepetkin & McWilliams 2005)" (commit `06f4de939`,
2026-04-11, "Add barotropic time-averaging for split-explicit stability")
and the BEBT semi-implicit pressure-gradient blend "match[es] MOM6 default"
`bebt=0.2` (commit `3d0170d04`, 2026-04-20, #205, "Add BEBT, cosine time
filter, slow-forcing coupling to barotropic solver"). This doc's own §6.1
row already called it "SM2005-style, unreferenced" — that framing is now
made literal in the module docstring and the registry.

*Mechanical fix* (same PR): `barotropic_latlon_cgrid.py`'s module docstring
gained a paragraph naming this reference. The registry's S-16 row
(`tests/ocean/unit/_nemo_branch_isomorphism_baseline.py`) changed disposition
`ARTIFICIAL_BRANCH` -> `OTHER_RECIPE`; the `_run_substep_loop` `Impl`'s
`Reference` is now `model="legoesm_legacy"`, citing the commits above, with
`selected_by=("default_wright_v1", "legoesm_linear_v1", "legoesm_nemo_like_v1")`
— the three lat-lon catalog recipes MEASURED (call-counter check, unchanged
from the measurement above) to actually execute this arm. The row's
`ARTIFICIAL_BRANCH_BASELINE` entry is removed (no longer needed —
`_row_needs_baseline_entry` is now False for this row).

*Companion decision, same PR*: the recipe named `nemo_v1` is renamed — see
`docs/ocean/fidelity/orca1_card_nemo_arm_repoint.md` for the rename record —
because it resolves this exact generic (non-NEMO) barotropic arm, not
NEMO's `dyn_spg_ts`, so "nemo_v1" overclaimed fidelity precisely on the
routine this row is about.

### S-35 (stage-3 `zub` barotropic correction)

1. **Duplicate present at HEAD?** Yes. Site (a) `ocean_model_latlon_cgrid.py:
   4800 _replace_stage_mean`, called unconditionally per WS-RK3 stage (call
   sites `:4963,4979,4997`). Site (b) `:6555 _fixed_depth_means`, called at
   `:6382,6465` but gated: `_impose_mean = getattr(_cfg_b.barotropic,
   "nemo_stage_mean_imposition", False) and _apply_implicit_vmix` (`:6378-6380`).
2. **Certified cards?** `nemo_stage_mean_imposition` defaults `False`
   (`state.py:1280`). Only `fidelity/nemo_recipe.py:978` (GYRE) sets it `True`.
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
   `ocean_model_latlon_cgrid.py:1279` — gated by `bbl_adv_option` (default `0`,
   `state.py:2097`). `physics/bbl_adv.py:312 apply_bbl_adv_step` (driver
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

1. **Duplicate present at HEAD?** Yes. `ocean_model_latlon_cgrid.py:10120
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
| S-16 | yes (both arms remain; kept deliberately, not a defect) | DINO/LOCK/OVERFLOW/GYRE=nemo_literal | **not on this path** — ORCA1 resolves `barotropic_solver=implicit_cn` (measured), which never runs the gated code | n/a for ORCA1; the generic arm's real consumers are 19 test-matrix experiments + `legoesm_nemo_like_v1` (renamed from `nemo_v1`) | **OTHER_RECIPE** (was BLOCKED/COLLAPSIBLE_NOW — 2026-09-02 USER DECISION: give the generic arm a real reference and keep both arms, no collapse; see the S-16 addendum above) |
| S-35 | yes | GYRE=both sites; LOCK/OVERFLOW=site (a) only (wrong side of dyn_zdf) | site (a) only | yes — `_fixed_depth_means` is model-generic | COLLAPSIBLE_NOW |
| S-42 | yes | OVERFLOW=in-stage (2); LOCK=off (0) | separate driver-side `--bbl-adv` path, `bbl_adv_option` stays 0 | ~~plausibly~~ **NO — RETRACTED 2026-09-02**: the in-model hook is rk3_ws-only and ORCA1 runs the euler tracer lane, so the flip would run NO BBL | ~~COLLAPSIBLE_NOW~~ **BLOCKED** (see the S-42 addendum) |
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


## 2026-09-02 S-42 (BBL) — the clamp is gone; the remaining branch is PLACEMENT, and it is blocked

**What the row actually contains at HEAD.** There are not two transcriptions of
`tra_bbl_adv`. `physics/bbl_adv.py:312 apply_bbl_adv_step` CALLS
`bbl_transports` (`:176`) and `apply_bbl_adv_tendency` (`:232`) — the same two
functions the in-stage site calls from
`ocean_model_latlon_cgrid.py:1287-1332`. The arithmetic of
`trabbl.F90:243-284` is written ONCE. This corrects the row's earlier
"two independent NEMO-trabbl transcriptions" framing (its own baseline reason
string already said "shared transport/tendency arithmetic").

**Landed: the transport cap is deleted (Rule 9).** `apply_bbl_adv_step` used to
clamp `|utr|`/`|vtr|` at `0.25*V_min/dt`. NEMO's `tra_bbl_adv`
(`trabbl.F90:243-284`) clamps neither `utr_bbl` nor `vtr_bbl`, and
`recipes.py` never mentions BBL, so no non-NEMO recipe selected the capped arm.
Measured in `tests/ocean/unit/test_bbl_adv.py`:

| | old capped path | survivor |
|---|---|---|
| ORCA1-like face (area 1e9 m^2, dt 600 s) | cap never binds | T/S **bit-identical** to the capped path |
| pathological face (area 1e6 m^2) | cap binds, transport reduced | uncapped NEMO transport, states differ |

so the deletion is inert for every production configuration and non-vacuous as
a test. Restoring the cap turns
`test_transport_cap_is_gone_and_moved_nothing_at_ocean_scales` red (verified).

**REFUSED as a config flip: routing ORCA1 through the in-stage site.** The
in-model BBL hook is built only inside the WS-RK3 tracer lane
(`ocean_model_latlon_cgrid.py:6211`, under `elif _tti == "rk3_ws":`) and is
passed only to `_nemo_ws_rk3_tracer_pair_step`. ORCA1 resolves
`tracer_time_integrator="euler"` (`run_omip_core2.py:7463`), whose branch has
no BBL hook at all. So setting `bbl_adv_option=2` on ORCA1 — the "config-only
re-point" the earlier triage called *plausible* — would silently run **no BBL**
on a production lane that currently runs it (`--bbl-adv` is passed by
`scripts/cluster/omip_nemo/run_standard_faithful_1deg.sbatch:67` and ~20 A/B
decks). Deleting `apply_bbl_adv_step` without first building a euler-lane site
would do the same. That earlier triage line is RETRACTED.

**And the site to collapse ONTO does not exist as a NEMO arm.** NEMO applies
`tra_bbl` inside the tracer step's RHS, at `stprk3_stg.F90:468,498,588` (RK3)
or `stpmlf.F90` (MLF). ORCA1's legoESM lane is a forward-Euler tracer step,
which transcribes neither. Wiring BBL into it would invent a THIRD composition,
not collapse onto NEMO's.

**Disposition: ARTIFICIAL_BRANCH stays, with the branch narrowed to placement.**
The open item is a Rule-3 ASK, one line: does ORCA1's BBL keep the host
post-step split (today), or does the OMIP lane move to a NEMO tracer lane
(RK3-stage or MLF) that has NEMO's own `tra_bbl` site? That changes ORCA1's
answers and is not a default to move silently.

**2026-09-02 follow-up (this pass): the silent-drop hazard is now a hard
error.** `bbl_adv_option=2` paired with any `tracer_time_integrator` other
than `rk3_ws` now raises `ValueError` at `LatLonCGridOceanModel._validate_config`
(`ocean_model_latlon_cgrid.py:3510-3529`), naming the mismatched lane and
pointing at the driver-side `apply_bbl_adv_step` (`run_omip_core2.py`'s
`--bbl-adv`) as the alternative. Verified inert on ORCA1 (never sets
`bbl_adv_option`, stays at the default 0). This closes the "SILENTLY run NO
BBL" hazard; the placement-collapse ASK above is unchanged and still open.

**Gates.** LOCK_EXCHANGE-zco and OVERFLOW-zps never call `apply_bbl_adv_step`
(their BBL, where on at all, is the in-stage site), so both cards' stage-sweep
and kt=1..10 trajectory gates are bit-identical across this change — measured,
not assumed; see the commit message.



## 2026-09-02 M-01 (`stp_MLF` written twice) — MEASURED DIFFERENCE, no collapse

The row's collapse question is whether DINO's certified
`outer_integrator="leapfrog"` (`_leapfrog_step`, two `_step_impl` passes) can be
repointed at `_nemo_mlf_step`, the structurally faithful single pass in which
only `dyn_ldf`/`tra_ldf`/Redi read the before level as their own call argument
(`stpmlf.F90:275`, `:437`). Answer: no. Measured, not argued.

**Harness.** `scripts/validate/ocean_fidelity/dino_1226/mlf_step_mechanism_ab.py`,
fp64 (policy set explicitly — `JAX_ENABLE_X64` alone leaves the bridge at f32),
CPU, card `nemo_dino_kamm_mlf` bridged from NEMO's day-180 restart with
`LEGOESM_NEMO_E3T=both`. Both arms are applied to the SAME state, under the SAME
`jax.jit` wrapper, with the same surface forcing and external tracer rate; the
trajectory then advances on the certified (leapfrog) arm, so every row is a
one-step A/B rather than a compounding one. `--bridge-tke` is on because the card
selects `tke_shear_evaluation_stage="step_entry"` +
`tke_preclosure_coeff_source="carried_previous_step"`, which require a carried
`avm_k` that only the restart bridge supplies; both arms see the same seed.

**Identity control**: `_leapfrog_step` against itself is `0.0` on every field, so a
non-zero reading below is the METHOD, not the harness.

| max abs difference | T [degC] | S [PSU] | u [m/s] | v [m/s] | eta [m] |
|---|---|---|---|---|---|
| step 1 | 4.963e-4 | 4.192e-5 | 5.196e-6 | 9.568e-6 | 5.725e-6 |
| step 2 | 4.968e-4 | 4.191e-5 | 1.477e-6 | 1.573e-6 | 1.062e-6 |
| step 1, GM/Redi OFF | 1.709e-7 | 1.783e-8 | 5.196e-6 | 9.568e-6 | 5.725e-6 |

Worst row is T at step 1: **8.6e10 ulp** of its own field scale. The bar for a
collapse is 2 ulp. **M-01 is a measured DIFFERENCE, not a re-association**, and
the certified DINO card is NOT repointed.

**Two things the GM/Redi ablation settles.** Turning GM/Redi off drops the
tracer difference by a factor ~2900 on this state (T 4.96e-4 -> 1.71e-7) — a ratio, not a characterisation: the swapped operand enters through the neutral-slope denominator, so it is not expected to be stable across stratifications. What it does establish is GM/Redi's tracer source as the mechanism the committed transcription test
already predicted (`tests/ocean/unit/test_nemo_mlf_step_transcription.py`: the
two-pass method feeds it `T_mid ~ Nbb`, the one-pass method the raw `Nbb` of
`stpmlf.F90:437`). But the MOMENTUM difference is **bit-unchanged by that
ablation** — `u`, `v` and `eta` at step 1 are the same to all printed digits
with GM/Redi on and off. So the momentum gap is NOT the GM/Redi channel. The
transcription test calls its momentum residual "XLA JIT-fusion floating-point
noise" seeded at ~1e-9 and amplified through the barotropic substeps; on the
DINO card that residual is 5.2e-6 m/s in `u` and 9.6e-6 m/s in `v`. The mechanism
reviewer names a third candidate and it is the more likely one: the two-pass arm
runs its whole second `_step_impl` on a full before-level state, so `dyn_ldf` sees
before-level `eta` and therefore before-level `h_k`/`h_u`/`h_v` and viscosity,
while `_ldf_state` swaps only the tracer/velocity operand and leaves the
thicknesses at the now level. A 5000x amplification of 1e-9 in ONE step through a
stable barotropic solve is not credible, so "fusion noise" is probably the wrong
owner. Left UNVERIFIED — neither explanation is measured here.

**Also blocked structurally, independent of the numbers.** `outer_integrator=
"nemo_mlf"` hard-requires `implicit_vmix_e3t_now_divisor=True`
(`ocean_model_latlon_cgrid.py:3187`), which DINO's certified card sets False
(S-34). So the collapse could not be a one-variable config move even if the
states agreed: it would flip S-34's divisor at the same time. The A/B above
sidesteps that by calling the two methods directly at DINO's own config, which
is why it isolates the mechanism.

**Disposition: ARTIFICIAL_BRANCH stays.** Two implementations of one NEMO
routine remain, and the faithful one is still selected by no card. The open item
is now a physics question with a price tag, not a refactor: adopting
`stpmlf.F90:437`'s literal `pts(:,:,:,:,Kbb)` read moves the certified DINO
tracer field by ~5e-4 degC per step, so it needs the Rule-8 treatment (a
faithful change that moves a certified number is a finding, not a revert), and
a decision from the user before any card moves.


## 2026-09-02 S-19 (`wzv`) round 2 — the collapse is REFUSED by the committed bar

The ask was to make the NEMO `wzv` arm the unbranched behaviour of the NEMO
WS-RK3 identity and of the NEMO MLF identity, and to delete
`diagnose_w_from_flux_div` unless a non-NEMO recipe selects it. Neither half
survives contact with the measurement.

**1. RETRACTION.** The "S-19 arm result" section above records "every kt=1..10
trajectory row BIT-IDENTICAL" on both cards. That was measured on
`fidelity/nemo-wzv-generic-operands` at `267c7b673`. It is NO LONGER TRUE at
this HEAD (after the S-30/S-12 momentum-ladder collapse). Re-measured, fp64
CPU, `--arm-literal-stage-wzv --compare-to` against the same committed
references:

| | LOCK_EXCHANGE-zco | OVERFLOW-zps |
|---|---|---|
| gate JSON vs the baseline run | byte-identical except the arm's own selector | **three rows move** |
| moved rows | none | `kt8.before.T` 2.665e-16 (1.200 ulp), `kt9.before.T` 3.553e-16 (1.600 ulp), `kt10.before.T` 3.553e-16 (1.600 ulp) |
| `largest_move_ulps` | 0.006 (unchanged) | 1.600 |
| `first_over_bar` | kt=2 {T,u} — unchanged | kt=2 {T,u,ssh} — unchanged |
| `--compare-to` verdict | fails only on the selector | **FAIL** |

The harness is deterministic here: the same OVERFLOW trajectory run twice
across the S-42 commit was byte-identical, so the three moves are the arm.

And the SHAPE of those three moves argues FOR algebraic equivalence, not
against it: kt=1..7 are bit-identical, then 2.665e-16, 3.553e-16, 3.553e-16 —
flat, not doubling. Chaotic amplification is exponential; this is a sub-ulp
per-step increment crossing the rounding threshold at step 8 and saturating.
So "almost free but not free" is the right wording and must not be upgraded to
"the two arms disagree".

**2. Why that is a refusal and not a rounding argument.** The committed gate
(`legoesm/ocean/fidelity/ulp_move_gate.py`) holds TRACER rows to BIT-IDENTITY,
not to `MAX_ULP_MOVE`. All three moved rows are `T`. 1.2-1.6 ulp is inside the
2-ulp bar for a velocity row and outside the bar for a tracer row, and
relaxing that distinction would be weakening the gate. So the collapse is
REFUSED and nothing was landed for this row.

**3. And `diagnose_w_from_flux_div` cannot be deleted regardless.** It is the
generic vertical-velocity diagnostic for three other lanes —
`ocean_pe_mpas.py:312` and `ocean_model_mpas.py:980` (MPAS),
`ocean_pe_cdgrid.py:361` (C-D grid), `ocean_pe_latlon_cgrid.py:1369` and
`ocean_model_latlon_cgrid.py:458` (the lat-lon PE lane), plus
`fidelity/box_heat_budget.py:310`. So the row cannot become OTHER_RECIPE
either: LOCK/OVERFLOW/ORCA1 are NEMO cards and they resolve
`wzv_call2_evaluation="generic"`, so a NEMO card still reaches the generic arm
at the MLF call-2 site (`ocean_model_latlon_cgrid.py:5568`).

**4. The second site is NEMO's own split, restated.** `wzv_call2_evaluation=
"nemo_literal"` requires `zad_qco_evaluation="nemo_literal"`, which requires
`vertical_momentum_scheme="nemo_advective"`, which is refused together with
`adaptive_implicit_vertadv=True`. Those selectors gate NEMO's MLF/`dynzad`
call sites; the L1 cards run NEMO's RK3 flux-form lane where `dynzad` is dead.
The two sites are not two implementations competing for one call — they are
NEMO's two different `wzv` calls, and each card reaches the one its lane has.

**Disposition: ARTIFICIAL_BRANCH stays, and the row is now BLOCKED rather than
COLLAPSIBLE.** The open question is a one-line ASK: OVERFLOW's kt=8..10
temperature rows move 1.2-1.6 ulp under the faithful arm — is that admissible
for a tracer row (the gate says no), or does the row stay branched?

**A structural consequence worth naming with it.** A bar that demands
BIT-IDENTITY on tracer rows makes ANY re-association permanently
un-collapsible, however faithful, because re-association is exactly what
changes the last bits. That is a property of the bar, not a physics verdict on
this row, and it will recur on every future collapse that touches a tracer.

## 2026-09-02 S-46 — `dyn_adv_up3` T-point upwind SELECTOR (new row; owner of the OVERFLOW stage-3 `u` debt)

| id | NEMO routine | NEMO switch | cards | legoESM implementation | selector | classification |
|---|---|---|---|---|---|---|
| S-46 | `dyn_adv_up3` T-point upwind branch (dynadv_up3.F90:166,169-170 `zui = uu_i + uu_{i+1}`, `zvj` :172; magnitude by the transport pair :176; F-point cross fluxes :179-187 and vertical flux :294-295 by the transport pair) | none (one routine) | L O; plus any card citing NEMO for its momentum advection | `opl:4114 _up3_reconstruct`, reached from `_bc_horizontal_momentum_advection_flux_form` — ONE implementation, TWO reference arms (sub-table below) | `momentum_flux_scheme` (`nemo_up3` / `oceananigans_up3`); no bare `upwind3` | SHARED — the T-point selector follows the REFERENCE the caller names, not the time integrator (scoped 2026-09-02) |

The two reference arms of that one implementation:

| arm | reference | T-point upwind selector | `momentum_flux_scheme` | selected by |
|---|---|---|---|---|
| NEMO | NEMO 5.0.2 `dynadv_up3.F90:166,169-170` (`zui = puu(ji)+puu(ji+1)`) | the ADVECTED-VELOCITY pair | `nemo_up3` | LOCK, OVERFLOW cards (`nemo_testcase_recipe.py`); `nemo_recipe.py` `momentum_core="flux_form_upwind3"` |
| Oceananigans | `UpwindBiased(order=3)`, `upwind_biased_advective_fluxes.jl:18-24` (`ũ = symmetric_interpolate(Ax_qᶠᶜᶜ, U)`, the transport) | the TRANSPORT pair | `oceananigans_up3` | `oceananigans_v1` decks — Silvestri UP3 jet (`silvestri_schemes.py`), `internal_tide` comparator |

Measured: OVERFLOW-zps kt=1 stage-3 baroclinic `u` `2.598798e-07 -> 4.551736e-10 m/s`
(replay corr `0.999994`, slope `1.00015`); LOCK_EXCHANGE-zco stage 2/3 and kt=2
`u`, `T` all at the `1e-15` bar (`2.9e-17`, `2.3e-17`, `T = 0.0`).  Receipt:
`nemo_testcases_l1_phase3_receipt.md`, "UP3 upwind-selector round"; preregistration
`nemo_testcases_l1_stage3_baroclinic_preregister.md`.

**2026-09-02 SCOPE FIX — the selector is keyed by the REFERENCE, not by the
time integrator.**  The row above originally read "the WS-RK3 stage program
passes `up3_upwind_selector="velocity"`; every other caller keeps the
transport-sign selector", i.e. a NEMO-referenced card that was not WS-RK3 ran
Oceananigans' rule.  That was a hidden branch point: ONE public scheme value
(`momentum_flux_scheme="upwind3"`) served two references whose T-point
selectors genuinely differ, and which one you got depended on
`momentum_time_integrator`.  Split into the two named arms above; the bare
`"upwind3"` is removed from `VALID_MOMENTUM_FLUX_SCHEME`, so a caller that does
not name its reference now fails validation instead of inheriting one.  The
integrator gate in `omlc` is gone; the private
`_NEMOWSRK3TestHooks.legacy_up3_transport_sign_selector` survives unchanged as
the stage-sweep gate's one-variable ablation arm.

The one caller whose arm this MOVES is `nemo_recipe.py`'s
`momentum_core="flux_form_upwind3"` option on a non-WS-RK3 integrator (no card
selects it; the `rest`/`eady` setups can construct it, and `gyre` cannot —
it forces `rk3_ws`, which requires the `nemo_up3` vertical arm the UP3
`_momentum_options` block does not set).  Its numbers are in the commit
message and the phase-3 receipt.

**2026-09-03 — that non-WS-RK3 construction is now REFUSED.** S-47 below wires
NEMO's own `e3u(Kmm)` face thickness only inside the `rk3_ws` stage program;
off that lane, `nemo_up3` falls back to the legacy min-of-stretched-T-thickness
rule S-47 measured as first-order wrong — a pairing NEMO itself never runs.
`_validate_config` now raises on `momentum_flux_scheme="nemo_up3"` paired with
any `momentum_time_integrator` other than `rk3_ws`
(`tests/ocean/unit/test_config_footguns.py`), so the `rest`/`eady`
`flux_form_upwind3` arm above is constructible only with
`momentum_time_integrator="rk3_ws"` too; a non-NEMO UP3 arm on those setups
selects `momentum_flux_scheme="oceananigans_up3"` instead.

**Open follow-up (NOT silently taken here).** NEMO evaluates `dyn_adv_up3`'s
curvature at `Kbb`; `stprk3_stg.F90:316,326-331` passes `Kmm` as BOTH velocity
levels, which is why the WS-RK3 lane is faithful with one live velocity.  A
NEMO UP3 caller on an MLF/leapfrog integrator would additionally need the
`Kbb` curvature, which legoESM does not supply — such a caller does not exist
today (no MLF card selects `flux_form_upwind3`), and the `rest`/`eady` NEMO
recipe option is a single-level RK3 lane, not MLF.  Do not hand a future MLF
UP3 card `Kmm` and call it faithful.

Corrections to existing rows:

- **S-21** ("stage transport `zFu/zFv` + `zub` correction … SHARED across the three RK3
  cards"): true for the tracer stages and for momentum stages 2-3.  The step-entry
  `tendencies()` call that seeds momentum stage 1 (`omlc:4070`) is handed NO separate
  transport, so stage-1 momentum advection runs on `Q = h u(Kbb)` while NEMO's stage-1
  `dyn_adv` (`stprk3_stg.F90:315`) consumes `zFu = e2u e3u (uu(Kbb) + zub)` built
  unconditionally at `:259-275`.  Inert at kt=1 (the cards start from rest, so the
  transport multiplies a zero advected velocity), live from kt=2.  MEASURED in the
  S-21 round (preregistration `nemo_testcases_l1_stage1_transport_preregister.md`,
  receipt `nemo_testcases_l1_phase3_receipt.md` "S-21 stage-1 transport round"):
  stage-1 RHS difference at the OVERFLOW kt=2 entry `3.136e-07 m/s^2`, free-run
  effect `+0.29%` on kt=3 `u`, `<1e-7` relative at kt=10 and kt=60, and EXACTLY
  zero on `ssh` at every kt on both cards.  So it is NOT the owner of the kt>=3
  SSH walk — that candidate is RETRACTED.  **LANDED 2026-09-02** as the
  measured-inert, NEMO-faithful stage-1 transport (the difference of the stage
  helper with and without the `zub` transport, added to the stage-1 RHS at
  `omlc` stage 1); before/after rows reproduce the S-21 arm exactly (OVERFLOW
  kt=3 `u` `9.079022e-09 -> 9.105049e-09`, kt=10/60 unchanged to 7 digits, kt=2
  bit-identical on both cards).  Its 6120-step statistics ride with the
  SSH-walk seed round (`nemo_testcases_l1_ssh_walk_preregister.md`).
- **S-18 (seed corollary, 2026-09-02, SSH-walk round)** — a THIRD construction of
  NEMO's `e3u(Kmm)` existed on the barotropic LOOP-ENTRY seed of every card that
  does not carry `mesh_mask.nc` (LOCK/OVERFLOW): `barotropic_latlon_cgrid.py`
  `_depth_average_to_faces` (`seed_evaluation="nemo_literal"`, card path) took the
  per-level MIN of the two STRETCHED T-cell thicknesses (`min_cell_to_uface(h_k)`)
  rescaled by `H_u_nemo / min(H_west, H_east)`, summed against the literal inverse
  `1/H_u_nemo`.  Where the two columns' reference depths differ (the OVERFLOW-zps
  shelf break) the level-min and the column-min follow different columns, so the
  weights sum to `hu_0 (1+r3t_e)(1+r3u)/(1+r3t_w)` and a uniform velocity is returned
  scaled by `(1+r3t_e)/(1+r3t_w)` — measured `1.05e-9 -> 3.1e-7 -> 4.3e-6 m/s`
  against NEMO's carried `uu_b(Kbb)` at kt=2..4 with NEMO's EXACT entry state, the
  re-injected owner of the kt>=3 OVERFLOW SSH walk (kt=10 SSH `9.237e-5 -> 1.474e-6`).
  **COLLAPSED** onto the one shared kernel (`vertical.py`
  `nemo_qco_card_mesh_operands` + `nemo_qco_live_face_geometry_from_operands`, the same
  pair `_nemo_ws_qco_stage_faces` and the MLF tracer transport use) in
  `_nemo_literal_seed_from_card_mesh`; the carried-mesh (DINO) seed is untouched;
  `_NEMOWSRK3TestHooks.legacy_seed_min_rule_faces` is the harness-only control.  The
  PE-lane depth-average / slow-forcing use of `min_cell_to_uface` (`opl:1364`,
  `omlc h_u_pre`) is normalised by its own sum and therefore unbiased for a uniform
  velocity; it remains the MOM6/MITgcm hFacW citation.  It is NOT the owner of the
  residual `slow_u` operand: the independent review measured its normalised weight
  error against NEMO's `e3u_0/hu_0` rule at the OVERFLOW kt=2 entry as `4.65e-9` at
  ONE face (u-column 23, a 26|26 partial-cell mismatch) and `<= 1.7e-16` everywhere
  else, far too small to supply `slow_u = 1.30e-10` (see the SSH-walk receipt, open
  item 2, for the discriminating measurement).  **2026-09-02, S-47**: the
  `slow_u` operand WAS a third `min_cell_to_uface` site -- not the depth-mean
  WEIGHTS (correctly exonerated above) but the advection OPERATOR's face
  thickness inside `tendencies()` (`opl:1364`), whose min-of-stretched-T rule
  enters the neighbouring faces' T-point fluxes and does not cancel there.
  Collapsed onto the same kernel; see the S-47 section below.
- **S-33/S-34** (dynzdf operands), bottom-localised, measured inert for the front columns
  (no partial cell at columns 20/21) and bounded by NEMO's whole stage-3 increment
  `1.09e-08` elsewhere: legoESM `e3uw(Kmm)` = midpoint of `e3t_now` (12.25 m at a
  partial-cell interface) vs NEMO `e3uw_0 = e3w_1d = 20 m` (`usrdef_zgr.F90:167`);
  legoESM `e3u(Kaa)` = masked cell->face average vs NEMO `e3u_0 = min` at a staircase
  face.  Neither owns the remaining `4.55e-10`.

## 2026-09-02 S-47 — `dyn_adv_up3` face THICKNESS (new row; owner of the OVERFLOW stage-3 remainder AND the `slow_u` debt)

Preregistration `nemo_testcases_l1_stage3_remainder_preregister.md`
(commit `f8247f9a3`, frozen before the arm); receipt
`nemo_testcases_l1_stage3_remainder_receipt.md`.

NEMO consumes ONE face thickness in `dyn_adv_up3`: `zFu = e2u*e3u(Kmm)*uu`
(`dynadv_up3.F90:160`, or the stage transport `stprk3_stg.F90:273`) and the
divisor `e3u(ji,jj,jk,Kmm)` of the flux divergence (`:205-207`), with
`e3u(Kmm) = e3u_0*(1+r3u(Kmm)*umask)` (`domzgr_substitute.h90:127`) and
`r3u` the `e1e2t`-weighted ssh mean over `hu_0` (`domqco.F90:219-220`).
legoESM's `latlon_cgrid_ocean_baroclinic_tendencies` built its own
`h_u = min_cell_to_uface(compute_layer_thickness(eta))` (`opl:1221,1364`)
and handed it to the flux-form momentum advection (`opl:4504`): the MIN of
the two STRETCHED T thicknesses, `e3u_0*(1 + min(r3t_W, r3t_E))` on equal
columns, i.e. `-0.5*|ssh_W - ssh_E|/hu_0` off NEMO's rule -- the same
first-order defect S-18 (seed) and S-21 (transport) had, at a third site.
It cancels in the T-point flux divergence AT the front face (both fluxes
around it carry the same `c_20 F_20`) and survives at the flanking faces
19/21, which is exactly where the `4.55e-10` remainder and the `slow_u` debt
sat.

Replay on NEMO's stage-2 Kaa operands (legoESM's own operator, `h_u` the
only variable): stage-3 remainder corr `0.99979`, slope `0.989`, residual
`7.1e-12` of `4.55e-10`; stage-2 residual corr `0.9998`, slope `0.989`;
`slow_u` at kt=2/3/4 corr `1.0000 / 0.9999 / 0.974`, slope `1.000 / 0.9997
/ 0.999`.  Every other stage-3 operator (HPG, vertical UP3, qco ratios,
transport triplet, implicit ZDF, stage-mean weights) `<= 1e-15 m/s` on the
same operands.  Landed (`60d0c5420`): the WS-RK3 stage program hands the
stage's `(e3u, e3v)(Kmm)` pair from `_nemo_ws_qco_stage_faces` to all four
of its `tendencies()` calls through the new `momentum_flux_face_thickness`
argument; every other caller is bit-identical.  Measured at kt=1 (stage
sweep): OVERFLOW stage-3 / kt=2 `u` `4.551736e-10 -> 7.064252e-12`, stage 2
`9.433404e-11 -> 1.566344e-12`, stage 1 bit-identical, LOCK bit-identical;
the legacy arm reproduces the pre-fix rows bit for bit.

Corrections to existing rows: **S-18** -- the `slow_u` operand was this site,
not the depth-mean weights (see the S-18 bullet above); **S-21** -- the
"SHARED across the three RK3 cards" statement now also holds for the
momentum advection's thickness, which had silently diverged from the
transport's.

## 2026-09-03 S-48 follow-up — the stage depth-mean WEIGHTS, and the census row

Receipt: `testcases/nemo_testcases_l1_census_receipt.md`; instrument
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_census_map_probe.py`
(`faces` command).  No model code changed.

**The rule.**  NEMO removes a REFERENCE-weighted depth mean from every WS-RK3
stage velocity:

```fortran
zub(ji,jj) = uu_b(ji,jj,Kaa) - SUM( e3u_0(ji,jj,:)*uu(ji,jj,:,Kaa) ) * r1_hu_0(ji,jj)  ! stprk3_stg.F90:440
hu_0(:,:)  = hu_0(:,:) + e3u_0(:,:,jk) * umask(:,:,jk)                                 ! domain.F90:145
```

legoESM removes a LIVE-weighted one: `_replace_stage_mean` uses
`sum(u*h_u_pre)/H_u_pre` with `h_u_pre = min_cell_to_uface(h_k_pre)`
(`ocean_model_latlon_cgrid.py:4432`), `h_k_pre` being the ssh-stretched
thickness.  This was the phantom round's open item 3, recorded UNMEASURED.

**RETRACTED before it was believed.**  A first pass claimed the two weightings
are algebraically identical, because under z*/qco every level of a column
carries one `1 + r3u`.  That is true of NEMO's `e3u_0`-based mean and FALSE of
legoESM's, which takes a MIN over two columns whose Jacobians differ, so the
per-level argmin can switch sides.  The first pass had measured
reference-against-reference.  Caught by the diff reviewer, whose independent
probe reproduced the corrected values to the last digit.

**Measured** (free run from the card's initial state, fp64, gate u frame):

| kt | max `\|live - ref\|` per-level weight | wet levels differing `> 1e-12` | at gate faces 20/21/22 |
|---:|---:|---:|---:|
| 1 | `0.0` | 0 | `0.0` |
| 2 | `2.323655e-09` | 26 | `6.94e-18` |
| 5 | `3.761720e-06` | 105 | `6.94e-18` |
| 9 | `8.609751e-06` | 167 | `6.94e-18` |
| 10 | `7.570658e-06` | 173 | `6.94e-18` |

Reference-against-reference is still EXACT (`e3u_0` vs the model's reference
face thickness, `hu_0` vs its column depth, and the two reference weight
ratios: all `0.0` over every wet face), so the GEOMETRY is certified; the
branch is in the WEIGHT the stage mean uses, and only where a face's two
columns differ in depth.

**Disposition.**  CLOSED at the `k=24` injection faces (gate 20/21/22 have
equal bathymetry on both sides, so the min picks one uniform Jacobian and the
difference is one ULP at every step) -- which is what the `k=24` ownership
question needed.  **RESOLVED the same day, on the staircase faces too: the arm ran and NEMO's
reference weights are now the default** (`9a23ea2c3`, receipt
`testcases/nemo_testcases_l1_stage_mean_weights_receipt.md`), with the live
weighting behind a private `legacy_live_stage_mean_weights` hook that
reproduces the pre-fix OVERFLOW kt=1..10 rows on 50 of 50 rows bit for bit.

MEASURED INERT: every OVERFLOW trajectory row through kt=60 is unchanged beyond
the fifth significant digit, and all six 6120-step statistics move by EXACTLY
zero.  The preregistration's scaling estimate -- which predicted a `1.5x`
improvement at kt=10 from `shear x dw` -- is RETRACTED: it used a mature slope
shear the first ten steps do not have.  The change stays landed on Rule 0
alone.  Third consumer, named and UNMEASURED: `build_nemo_gyre_recipe`
(`nemo_recipe.py:970`) also selects `rk3_ws`, and its four tests fail at the
BASE tree on an unrelated `pgf_quadrature` validation error.

**Not a finding about the census.**  The same round instrumented the one
OVERFLOW statistic still outside the NEMO scheme spread
(`final_water_mass_census`) and could NOT attribute it: the row is not monotone
in bulk mixing (NEMO's own FCT4 run moves it the same direction as legoESM
while moving the dilution measure the opposite way), so no operator attribution
to that row is supported yet.  See the receipt's retraction list.

## 2026-09-03 S-49 — nonlinear implicit split-explicit bottom drag

| id | NEMO routine | NEMO switch | cards | legoESM implementation | selector | classification |
|---|---|---|---|---|---|---|
| S-49 | `zdf_drg` rate + `dyn_drg_init` external-mode residual/rate + `dyn_zdf` implicit bottom cell | `ln_non_lin` + `ln_drgimp` + `ln_dynspg_ts` | D G | `nemo_bottom_drag_rate_faces`, consumed by the existing external-substep and implicit-ZDF sites | `bottom_drag_scheme="nemo_quadratic"` plus the inseparable `zdf_drag_in_matrix` + `zdf_baroclinic_only` + `barotropic_drag_substep` identity | SHARED |

NEMO has one stored Kmm `rCdU_bot`: `zdfdrg.F90:138-190` constructs it,
`dynspg_ts.F90:1584-1643` freezes its face rate and baroclinic residual for the
external-mode solve, `:699-705` applies the entry-velocity drag inside every
substep, and `dynzdf.F90:148-160,293-305` composes the same rate into the
baroclinic-only implicit solve. DINO already selected this canonical program;
GYRE now selects the same three-part identity from its resolved
`ln_non_lin=T`, `ln_drgimp=T`, `ln_dynspg_ts=T` namelist. The private
`omit_barotropic_substep_drag` hook is a one-variable measurement ablation of
the in-substep boundary only, cites this reference, and is not a constructible
scheme arm.

## 2026-09-03 S-39/S-40/S-41 — GYRE stage-3 tracer completion

The GYRE walk closes an artificial ordering error without adding a routine:
the existing `_nemo_ws_rk3_tracer_pair_step` now carries non-advective physics
in its stage-3 Krhs and returns that same content RHS to the one existing
NEMO-literal `tra_zdf` solver.  This matches `stprk3_stg.F90:565-600` and
`trazdf.F90:271-286`; stages 1–2 still receive only nonlinear EMP transport.

Two-band QSR also has one implementation.  The shared physics kernel owns the
deposit, while the external-forcing consumer supplies only qns on the NEMO
selector (`trasbc.F90:299-315`; `traqsr.F90:665-712`).  The prior second
deposit was an artificial inline consumer, not a NEMO switch.  GYRE additionally
selects NEMO's full-qsr convention (rather than legoESM's legacy 94% split) and
the card constants.  Its TEOS surface EMP operand uses the literal shared
`nemo_potential_temperature_from_conservative` recurrence from
`eosbn2.F90:1493-1542`.  These are components of the existing collapsed
`gyre_vector_ene_c2` identity; no independently constructible hybrid was added.

## 2026-09-03 S-50 — GYRE literal `hpg_sco` operands and recurrence

| id | NEMO routine | NEMO switch | cards | legoESM implementation | selector | classification |
|---|---|---|---|---|---|---|
| S-50 | `hpg_sco` density polynomial plus bottom-up pressure-gradient recurrence (`dynhpg.F90:340-390`; `eosbn2.F90:265-288`) | `ln_hpg_sco` | D L O G | `nemo_roquet_density_anomaly_ratio` + `nemo_hpg_sco_literal_cgrid`, selected by the existing `pgf_scheme="nemo_sco"` identity | none beyond the NEMO `hpg_sco` switch | SHARED |

The GYRE stage-2 WRITE-only operand record made `rhd`, `e3w(Kmm)`, and
`gdept(Kmm)` independently observable.  The shared implementation preserves
NEMO's multiply-before-subtract density anomaly and the exact bottom-up
trapezoid/face-gradient statement order.  Private freeze/legacy hooks only
form preregistered causal arms; they cannot select a card and do not create a
second `hpg_sco` implementation.  It also includes NEMO's final `* tmask`
(`eosbn2.F90:288`), which round 9 restored after the density anomaly had been
left unmasked.

The 2026-09-03 round-9 audit disclosed a shared reassociation as well: the one
Roquet implementation now uses NEMO's literal `zn3/zn2/zn1/zn0` nesting for
both EOS-80 and TEOS-10.  Direct old-versus-literal probes moved density by at
most `4.55e-13 kg m-3` and `prd` by `4.43e-16`; this is a source-association
correction within S-50, not a GYRE-only identity arm.  OVERFLOW and
LOCK_EXCHANGE were assigned phase-3 2-ulp and tracer-bit-identity guards.
Round 10 proved the exact OVERFLOW EOS mask equals oracle `tmask` (zero
differing cells and zero NEMO-wet cells zeroed); the earlier `3.051e12`-ulp
result was primarily missing raw W-grid geometry.  With every card supplying
oracle `e3w_0`, LOCK_EXCHANGE's stage sweep passes at `0.017648` ulp, but
OVERFLOW's stage sweep still fails at `28.062` ulp and the kt=1…10 trajectory
guards fail at `513504.875` ulp (OVERFLOW) and `2.743` ulp plus lost tracer bit
identity (LOCK_EXCHANGE).  Direct-`prd` and prior-wet-evaluator one-variable
diagnostics did not remove the trajectory amplification.  The shared
EOS/HPG-association boundary therefore remains open and fail-closed; it is not
a passed isomorphism exception and no card guard exists.

Round 12 resolves the remaining source-operation association inside this same
S-50 implementation.  A stage-2 config-local WRITE-only record of the
executed four-argument `eos_insitu_pot_New_t` overload follows
`eosbn2.F90:260-288` through `zh/zt/zs/ztm`, `zn3/zn2/zn1/zn0`, `zn`, and
`prd`; all 52 coefficients and the normalizers/reference density at
`:1898,1926-1982,2331-2334` are also recorded.  QCO's depth operand is the
literal `r3t=ssh*r1_ht_0` then `gdept_0*(1+r3t)` path
(`domqco.F90:159-161`; `domzgr_substitute.h90:50,56,75,139`).  That input and
every normalized coordinate are bit-identical; the first departure was the
compiled `zn0` evaluation (4 ulp), while a pure NumPy transcription of the
Fortran statements was bit-identical through `prd`.  The shared JAX evaluator
now preserves a rounding boundary after each source operation, with no new
selector, card guard, callback, or duplicate routine.  On identical oracle
inputs every intermediate and `prd` is bit-identical under production JIT.

This does not close GYRE stage 2.  Propagating the shared change produces nine
one-ulp T/S cells at the live EOS entry and two differing `prd` cells
(`2.220446049250313e-16`); HPG improves about 32–37×, while corrected Kaa
improves about 23–26× to `2.0033670902752654e-14` u and
`2.0003665607629117e-14` v, still DEBT.  The next upstream tracer-state owner
is UNMEASURED, so the stage-3 transport walk remains gated.  Cross-card oracle
scoring is mixed: LOCK_EXCHANGE stage rows improve and remain AT-BAR;
OVERFLOW stage-3 u moves away by 3.281–4.0625 ulp, while its stage-1 u
improves.

The 2026-09-04 user decision replaces the prior-output compare-to criterion
with the shared, oracle-relative cellwise criterion.  Every scored boundary
now persists its compressed per-cell ``abs(legoESM - NEMO)`` field.  A shared
change fails when any cell worsens by more than two ulp of that cell's oracle
magnitude, when an AT-BAR row becomes DEBT, or when ``first_over_bar`` moves
earlier.  Movement versus the preceding legoESM output remains disclosed but
is not an acceptance criterion.  This policy is Rule 12 of the oracle-fidelity
skill; it is not a card-specific S-50 exception.

### PLAUSIBLE debt: barrier-only shared recurrences

Round 12's optimized-HLO census showed that ``optimization_barrier`` alone is
not a materialized rounding boundary.  The following shared NEMO-literal
recurrences still rely on that barrier without the EOS path's surviving IEEE
``isfinite``/``select`` identity.  They are therefore **PLAUSIBLE debt**, not
measured defects and not implementation exceptions; this round changes none
of them.

| shared recurrence | implementation | reason for registration |
|---|---|---|
| QCO live e3f recurrence | `vertical.py` (`nemo_qco_e3f_from_faces`) | source-operation barriers may disappear before contraction/reassociation; the live face-thickness/reciprocal recurrence was hardened and closed by ORCA2 in Round 22 |
| external-mode literal helpers | `dynamics/barotropic_common.py` | source-associated transport/update arithmetic uses barrier-only boundaries |
| RK3 external-mode and transport accumulation | `dynamics/barotropic_latlon_cgrid.py` | substep, weighted-mean, and final-update recurrences use barrier-only boundaries |
| PE HPG/QCO literal path | `dynamics/ocean_pe_latlon_cgrid.py` | HPG operands and the depth/transport recurrence use barrier-only boundaries |
| Thomas recurrence | `physics/vertical_mixing/implicit_solver.py` | forward elimination and back-substitution are association-sensitive recurrences |
| GM/Redi recurrence | `physics/lateral_mixing/gm_redi_latlon_cgrid.py` | slope/tensor accumulation carries barrier-only source order |

Each boundary remains on the ordered fidelity register and requires an
operand-level oracle discriminator before any code change.

## 2026-09-04 Round 19 — S-16/S-17 external-mode source recurrences

Round 19 keeps the existing RoutineRows S-16 and S-17; these are refinements
of their one registered implementations, not exceptions or new rows.  The
S-16 `nemo_literal` arm now follows `dynspg_ts.F90:549-562,603-609,627-629,
653-685,719-731,771-778` in written binary64 order for the AB3 midpoint,
continuity update, surface-weighted face SSH and inverse depth, backward SSH
blend, pressure-gradient handoff, and vector velocity update.  The private
`legacy_barotropic_continuity_association` hook is a non-constructible
one-variable test ablation of `dynspg_ts.F90:629`; it is not a selector.

S-17's existing `ene_metric` arm now preserves the live ENE coefficient
program at `dynspg_ts.F90:1383-1410`: it accumulates
`e3u*e3v*mask/e3f_vor` vertically before applying `ff_f` in the post-loop
coefficient statement.  The former algebraic folding of `ff_f/e3f_vor` into
each level was not a NEMO branch.  Both S-16 and S-17 use the one shared
`legoesm.core.source_rounding.nemo_source_round` helper; no GYRE guard or
second NEMO routine implementation was introduced.

## 2026-09-04 S-51 — coastal surface-stress face factors

| RoutineRow | NEMO routine / source | NEMO selector | cards | one legoESM implementation | selector / private seam | disposition |
|---|---|---|---|---|---|---|
| S-51 | U/V stress interpolation and coastal factors (`sbcmod.F90:539-546`) | none | D G A | `ocean_pe_latlon_cgrid.surface_stress_faces`, consumed by both the external-mode slow forcing and implicit momentum-ZDF surface boundary | no public selector; private `_NEMOWSRK3TestHooks.legacy_coastal_surface_stress_factors` is a gate-only ablation | SHARED |

The NEMO identity applies the written `(2-umask)*MAX(tmask,tmask_east)`
factor (and its meridional analogue) after the face average.  The generic
non-NEMO caller may omit masks; every NEMO WS-RK3 consumer supplies the same
cell and face masks.  GYRE's 580 active U and 570 active V faces are unchanged
because their factor is one, while the source-written dry coastal inventory
moves on 124 U and 102 V faces and becomes bit-identical to oracle V2.  The
private legacy hook does not construct a model card and introduces no second
physics implementation.
