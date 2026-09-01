# Three-grid unification design (2026-09-01)

Goal (user directive): only the grid may differ across tripole / MPAS / FESOM2
OMIP runs; all three faithful to NEMO ORCA1 GATEWAY (whose namelist confirms
ln_dm2dc=T, ln_qsr_rgb=T+chl, ln_zdfiwm=T, ln_isf=T (param. melt),
ln_trabbl=T with nn_bbl_adv=2). Reviews: GLM + codex on THIS design before code.

## Part A — MPAS unification (close the 3 remaining gaps)

A1. --isf (remap-only). ice_shelf_apply.apply_isf_prescribed_melt_step is
   grid-agnostic (..., nlev); apply site is the shared host loop (same as
   --geothermal, already unguarded on MPAS). Edits: drop grid gate
   (run_omip_core2.py:6294), latCell/lonCell branch (:6317), areaCell (:6327),
   and give isf_spe.py:80-87 the paired-cell structured=False regrid branch
   copied from nemo_native_fields._to_model_grid_2d. Renorm weights: pass
   areaCell (not cos-lat) for the melt-total renormalisation.

A2. --iwm (remap + plumbing; no new numerics). Kernel
   (internal_wave_mixing.compute_iwm_diffusivity) and geometry assembler
   (k_profiles.iwm_K_profile) already shape-generic. Edits: (a) iwm_forcing.py
   :101-118 paired-cell branch + target_area=areaCell for the TW renorm;
   (b) MPASOceanModel gains iwm_forcing= ctor kwarg mirroring
   ocean_model_latlon_cgrid.py:1495-1512 (both-or-neither check);
   (c) additive splice K_v_cells += K_iwm, A_v_cells += K_iwm at
   mpas_integration.py:909-917 BEFORE m_half/bottom_level masks (NEMO zdfphy
   order: closure first, zdf_iwm added on top — same as latlon :6982-7001);
   (d) remove 4 rejects (run_omip_core2.py:2073, mpas_integration.py:385/595/
   775) and un-hard-code iwm=None at :6055 so orca1_zdftke_config(iwm_enabled)
   flows rn_emin=1e-10 / rmxl_min=1e-3 (molecular backgrounds already forced
   on the MPAS card).

A3. --bbl-adv (real port). bbl_adv.py is structurally 2-D (i/j face arrays).
   Port Campin–Goosse advective BBL (nn_bbl_adv=2) to MPAS edges:
   geometry from cellsOnEdge/dvEdge (down-slope edge selection by bottom-depth
   difference), transports per edge, tendency scatter cell-wise. New module
   ocean/physics/bbl_adv_mpas.py reusing the same physics constants/config;
   shared host-loop apply site. Tests: analytic two-cell slope case matching
   the structured implementation on an equivalent geometry + conservation
   (column-sum T/S invariance). NOTE: largest-risk item in Part A; lands as
   its own PR after A1/A2.

## Part B — FESOM harmonization (staged; adapter route, NOT run_fesom_core2)

Principle: legoESM computes ALL surface physics once (shared CORE-II bulk,
dm2dc factor, RGB-chl penetration, legoESM prognostic ice, SSS restore,
runoff) exactly as on tripole/MPAS; the FESOM adapter INJECTS the resulting
fluxes at fesom_jax's SurfaceFluxes level (stress_surf, bc_T, bc_S, sw_3d,
water_flux), BYPASSING fesom's own L&Y bulk, 2-band Sweeney SW, PHC-restore
and internal ice. Rationale: cross-grid runs then measure the GRID, not two
bulk formulas / two SW schemes / two ice models. fesom_jax interior dynamics
(momentum, ALE zstar, tracer adv/diff, GM) stay untouched.

B1. Dispatch: --grid fesom in run_omip_core2; build_fesom_ocean() returns
   (grid, z_coord, model, state, H_bathy) from fesom_jax.mesh.load_mesh
   (real CORE2 bathymetry mesh, NOT build_flat_bottom_mesh); z_coord shim on
   mesh.Z (reject --nlev/--nemo-dz mismatches loudly); explicit accept/reject
   arm in every args.grid guard (hard-error on unsupported flags — no silent
   drops). step() gains surface_forcing=/freshwater=/t_seconds= kwargs,
   raising at stage B1 (accept-and-ignore forbidden).

B2. Momentum forcing: per-step stress channel replacing the frozen zeros
   (_stress_surf). legoESM node tau -> element tau (mean over elem_nodes),
   positional stress_surf with step_forcing=None. stress_node_surf synthesized
   from the same tau for the mixing closure.

B3. Tracer forcing (the crux): OceanSurfaceForcing -> SurfaceFluxes
   translator. bc_T from q_net (W/m2 -> K·m via VCPW, sign per fesom
   convention), sw_3d computed by the SHARED rgb_chl kernel on fesom columns
   (with the dm2dc host factor on the sw input — same code path as MPAS);
   zstar => use_virt_salt=False, real water_flux (kg/m2/s -> m/s) into SSH
   RHS + the -dt·T_adv·water_flux term; geothermal via the shared host-loop
   applicator on (N, nl) state. Double-count guards: sf.freshwater hard-reject
   in _validate_kpp_freshwater_contract fesom entry; fesom internal bulk /
   Sweeney / surf_relax_S paths must be provably OFF (step_forcing=None).

B4. IC/ice/SSS/runoff parity: WOA/NEMO-monthly loader on the fesom mesh
   (wrap phc_ic._load_one_variable with its own bracket setup; SKIP insitu2pot
   — WOA/NEMO monthly are potential temp; keep _extrap_nod3D land fill);
   legoESM prognostic sea ice + free_drift via the standard AtmToSurface lane
   (fesom internal ice OFF — one ice model across grids); SSS restore +
   runoff through the shared legoESM channels feeding water_flux/bc_S.

B5. Vertical mixing: PHASE 1 keep fesom's CVMix-TKE port (a validated TKE
   family; configured nemo_dirichlet-like surface BC + mxl anchor as today)
   driven by the injected stress_node_surf/heat_flux; measure. PHASE 2 (only
   if cross-grid MLD/SST gap is attributed to the closure): inject legoESM
   orca1_zdftke K/A profiles column-wise (needs a fesom_jax Kv-profile hook).
   Rationale: staged one-variable discipline; closure swap is a big lever and
   must be measured, not assumed.

## Execution order and gates
1. A1+A2 (one PR, small): unit tests (remap paired-cell test, iwm additive
   splice test vs latlon reference column), then 30d MPAS arm vs the d30
   fullharm baseline (one variable = isf+iwm).
2. A3 bbl port (own PR): analytic + conservation tests, then 30d arm.
3. B1..B4 sequential PRs, each with unit gates; B done => 30d FESOM arm
   scored vs GATEWAY rec 5 like the others.
4. Final: matched 180d x 3 grids at one SHA, scored d30/60/90 vs GATEWAY
   recs 5/11/17 + plateau trajectories; cross-grid band tables + maps.
All PRs: codex + GLM adversarial review before merge; probes committed.

## Known open risks (flagged, not hidden)
- bbl edge-port numerics (A3) and the B3 sign/unit translator are the two
  highest-defect-risk pieces; both get analytic tests before any GPU run.
- fesom EOS is JM95/EOS-80 vs tripole TEOS-10-ish NEMO polynomial — an
  irreducible lane difference for now; documented, measured, not silently
  "fixed".
- fesom TKE != legoESM TKE in phase 1 (B5) — deliberate staged choice.
