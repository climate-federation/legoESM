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
| S-12 | `stp_2D` stp2d.F90:49-288 (RK3 pre-step: eos+hpg+ldf+vor+wzv+adv @Kbb, `dyn_drg_init`, `dyn_spg_ts`) | key_RK3 | L O G | **none as a unit.** legoESM instead runs `omlc:3950 self.tendencies(...)` then a FIRST full WS 3-stage momentum ladder `omlc:4253-4335`, then the barotropic solve `omlc:4686` | implicit in `_step_impl` ordering | ARTIFICIAL_BRANCH — NEMO evaluates ONE Kbb RHS to seed `dyn_spg_ts`; legoESM evaluates a 3-stage ladder whose only consumer is the barotropic seed, then throws it away and re-runs the ladder at `omlc:4830-4886` |
| S-13 | `dyn_spg_ts` external loop dynspg_ts.F90:~560-800 | `ln_dynspg_ts` | D L O G A | `blc:1873 barotropic_substeps_latlon_cgrid` (+`blc:1135 _run_substep_loop`); alternates `barotropic_implicit_latlon_cgrid`, `rigid_lid_latlon_cgrid` | `barotropic_solver` | SHARED for all 5 (all `explicit_substep`); the other two solvers are non-NEMO oracles (Veros/MITgcm) |
| S-14 | `dyn_spg_ts` external velocity update dynspg_ts.F90:719-763 | `ln_dynadv_vec .OR. lk_linssh` (vector vs flux form) | L O (flux) / D G A (vector) | `blc:1105 _nemo_flux_form_external_velocity_update` (flux) ; `blc:1464-1512` inline vector update | gate `blc:1856 nemo_flux_form_update_active` = `rk3_ws AND flux_form` | NEMO_SWITCH on the vector/flux axis, but the gate carries an EXTRA `momentum_time_integrator=="rk3_ws"` condition NEMO does not have -> **ARTIFICIAL** conjunct (latent: an MLF+flux-form card would silently get the vector update) |
| S-15 | `dyn_spg_ts` backward face depth `zhu_bck` dynspg_ts.F90:738-747 | `key_qcoTest_FluxForm` (simple avg) vs default (e1e2t-weighted avg) | D L O G A | `blc:451 nemo_ssh_avg_face_depth` (+`_nemo_ssh_avg_prep/_apply`); `blc:581 _min_rule_face_depths` | `barotropic_face_depth` (`nemo_ssh_avg`/`min_rule`) | NEMO_SWITCH; on a lat-lon C-grid `e1e2t(i,j)==e1e2t(i+1,j)` so the two NEMO arms coincide (CONFIRMED algebraically, not measured). `min_rule` = ARTIFICIAL (no NEMO arm), ORCA1 on it (UNVERIFIED which arm ORCA1 resolves — `barotropic_face_depth` default is `min_rule`) |
| S-16 | `dyn_spg_ts` continuity / transport accumulation / spg dynspg_ts.F90:~640-700,~840 | none | D L O G | `blc:494 nemo_literal_metric_transports`, `:512 nemo_literal_accumulate_transport`, `:548 nemo_literal_continuity_divergence`, `:224 _nemo_literal_barotropic_pressure_gradient`, `:147 _nemo_literal_seed_from_reference_mesh` vs the generic inline arms in `_run_substep_loop` | `barotropic_continuity_evaluation`, `barotropic_transport_accumulation_evaluation`, `barotropic_pgf_evaluation`, `barotropic_seed_evaluation`, `barotropic_seed_face_depth` | ARTIFICIAL_BRANCH x5 — NEMO has one program; defaults are the generic arm; ORCA1 on defaults |
| S-17 | `dyn_cor_2D` (in-substep barotropic Coriolis) dynspg_ts.F90:359,689 | `ln_dynvor_ene/ens/een` | D G | `blc:944 een_barotropic_coriolis`, `:1058 barotropic_coriolis_een_pre_step`, `:731 _nemo_literal_een_coefficients`, `:840 _build_een_barotropic_inputs`; generic 4-pt `f_u*V_at_u` at `blc:1441,1494` | `barotropic_coriolis` (`avg`/`een`/`een_metric`/`ene_metric`), `barotropic_coriolis_split` (`frozen`/`live`), `barotropic_een_seed`, `barotropic_een_coefficient_evaluation` | NEMO_SWITCH on ene/een; `avg` + `frozen` = ARTIFICIAL (no NEMO arm) and is what L/O/A resolve to |
| S-18 | `dom_qco_r3c` / `dom_qco_r3c_RK3` domqco.F90:140-186, :189-240 (r3u = **surface-weighted MEAN** of ssh) | `key_qco`; `key_qcoTest_FluxForm` sub-arm | D L O G A | (a) `dynamics/latlon_cgrid_operators.py:277 min_cell_to_uface` / `:312 min_cell_to_vface` — **MIN of live thickness**, used by `omlc:1063 _nemo_ws_stage_transport`, `omlc:4031`, `blc:334,896,1850`; (b) `vertical.py:197 nemo_qco_live_face_thicknesses` (+`nemo_qco_live_face_geometry_from_operands`) — r3u MEAN, used at `omlc:5128`, `opl:1508`, `gm_redi_latlon_cgrid.py:954,4252` | (b) gated by `zad_qco_evaluation="nemo_literal"` / `wzv_call2_evaluation="nemo_literal"` / `gm_redi_*_face_thickness_evaluation="nemo_qco_live"` | **ARTIFICIAL_BRANCH (rank 1)** — two impls of one NEMO quantity; D/G reach the MEAN rule at ldfslp/zad, L/O/A reach the MIN rule at every site. IN FLIGHT: a separate agent is measuring mean-vs-min in `/tmp/codex-testcases` |
| S-19 | `wzv` sshwzv.F90 (np_velocity / np_transport) | none (arg-level) | D L O G A | `vertical.py:1293 diagnose_w_from_flux_div` (generic); `opl:~1450 nemo_qco_wzv_operands` (literal, incl. `nemo_qco_kmm_velocity_cycle`) | `wzv_call2_evaluation` (`generic`/`nemo_literal`) | ARTIFICIAL_BRANCH — one NEMO routine, two impls; D/G literal, L/O/A generic |
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
| S-30 | stage momentum time-stepping (qco thickness-weighted) stprk3_stg.F90:344-374 | `key_qco` | L O G | `omlc:4253-4335` (ladder #1, no barotropic correction) **and** `omlc:4830-4886` (ladder #2, corrected, overwrites `state_new.u/v`) | none — both always run when `momentum_time_integrator="rk3_ws"` and `stage_barotropic_correction` (default True) | **ARTIFICIAL_BRANCH (rank 3)** — the WS stage recurrence is written twice; ladder #1 costs 2 extra `tendencies()` evaluations and its only consumer is the barotropic seed |
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

1. **`dom_qco_r3c` r3u/r3v face thickness — MIN vs MEAN (S-18).**
   `min_cell_to_uface`/`min_cell_to_vface` (live-thickness MIN) and
   `nemo_qco_live_face_thicknesses` (NEMO's surface-weighted MEAN of ssh over
   `e3u_0`) both implement `domqco.F90:165-170`. DINO/GYRE reach the MEAN at
   `ldf_slp` and `dyn_zad`; LOCK/OVERFLOW/ORCA1 reach the MIN everywhere,
   including inside `_nemo_ws_stage_transport` (the operand of BOTH `dyn_adv`
   and `tra_adv`).
   *Collapse:* make `nemo_qco_live_face_thicknesses` the single `e3u(Kmm)`
   producer for every qco card and delete `zad_qco_evaluation` /
   `wzv_call2_evaluation` / `gm_redi_*_face_thickness_evaluation`.
   *Risk:* HIGH-VALUE, HIGH-CHURN. Changes every RK3 card's transport
   bit-for-bit, so the certified DINO twin and both L1 gates must be re-run.
   The mean-vs-min measurement is IN FLIGHT in `/tmp/codex-testcases`; do not
   land ahead of it.

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
| S-18 | `dom_qco_r3c` r3u/r3v MEAN vs MIN domqco.F90:140-240 | `min_cell_to_uface`/`_vface` (Adcroft/Hill/Marshall 1997 MOM6/MITgcm convention, CONFIRMED cited) vs `nemo_qco_live_face_thicknesses` (NEMO MEAN) | MEAN: DINO + GYRE (l2); MIN: LOCK, OVERFLOW **on this checkout** (verified `_model_config` never sets `zad_qco_evaluation`/`wzv_call2_evaluation` — this is a live gap on the certified L1 cards too, not ORCA1-only as the source doc's framing implied) + ORCA1 | MIN cites Adcroft et al. 1997 (a real reference, but for a *different* model's convention, not NEMO); MEAN cites domqco.F90 | **NEMO_DUPLICATE** | CONFIRMED |
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
| NEMO_DUPLICATE | 8 | S-12, S-16, S-18, S-19, S-30, S-35, S-42, M-01 |
| OTHER_RECIPE | 11 | S-03, S-04, S-07, S-09, S-25, S-27, S-29, S-32, S-33, S-34, S-40 |
| ORPHAN | 0 | — |

**Headline correction**: several rows this doc flagged among its top-priority
"mistakes to collapse" — S-27 (Coriolis split), S-29 (PGF), S-33/S-34
(ZDF solver/divisor) — are legitimate Veros/MITgcm/Oceananigans/paper-cited
forks, each carrying a *different* real defect: ORCA1 (and, for S-34, DINO)
simply never selects the NEMO arm it already has, a reachability/hidden-
default bug, not duplicated NEMO implementation work. Conversely S-16/S-18/
S-19 (barotropic transport, qco face-thickness, wzv) are confirmed genuine
duplicates reaching **every** certified NEMO card, including LOCK/OVERFLOW on
this checkout for S-18/S-19 (not just ORCA1, as this doc's own §0 framing
suggested).

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
