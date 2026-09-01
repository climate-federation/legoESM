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

## Review dispositions (GLM + codex, 2026-09-01 — design reviewed BEFORE code)
Adopted (design amended accordingly):
- Staging: A1 isf-only 30d arm, THEN A2 iwm arm (both reviewers; isf meltwater
  stabilizes columns and damps K_iwm via 1/N2 — coupled, so separate arms).
- iwm remap target = the static forcing ATLAS only; K_iwm recomputed from each
  grid's own N2 (GLM CRITICAL; design already intended this, now explicit).
- iwm ctor kwarg must survive the MPAS runoff-depth/freshwater-config model
  REBUILDS at run_omip_core2.py:6586-6607/6653-6665 (codex MAJOR) — add a
  rebuild-preservation test.
- iwm TKE energy feed (rn_efr) ports WITH the K splice (GLM MINOR).
- B translator sign/unit table (codex CRITICALs, each becomes a unit test):
  bc_T uses NON-SOLAR heat only (q_net minus SW — SW enters via sw_3d);
  bc_T = +dt*q_nonsolar/VCPW (fesom RHS is -dt*heat_flux/VCPW, opposite sign
  convention); stress negated (legoESM applies -tau, fesom stress_surf is
  ocean-convention); water_flux = -F_fw/rho_w (fesom positive-UP); sw_3d needs
  layer-heating [K/s] -> interface-flux [K*m/s] reconstruction from the shared
  RGB kernel; bc_S conversion to fesom's time-integrated PSU*m increment;
  zstar must be EXPLICITLY selected in B1 (adapter defaults linfs => virtual
  salt) before use_virt_salt=False is legal; either/or guard so the host-loop
  SSS-restore applicator (run_omip_core2.py:8191-8205/8300-8337) cannot fire
  on top of the injected water_flux channel.
- B1 gains a closed surface heat/freshwater BUDGET test (GLM CRITICAL): sum of
  injected fluxes == ocean column heat/FW change over a step, on the real mesh.
- Bulk runs per-step against FESOM's OWN SST/ice state (GLM CRITICAL) — same
  contract as the MPAS lane; never computed on another grid and remapped.
- Ice model runs NATIVELY on the fesom mesh via the standard lane (GLM MAJOR).
- B5 order inverted per GLM: run the single-column TKE twin (fesom CVMix-TKE
  vs legoESM orca1-TKE, identical column+forcing) FIRST to bound the closure
  residual and set 3D tolerances, before any 3D FESOM arm is interpreted.
- NEW B6 (GLM CRITICAL gap): FESOM iwm + isf parity via the same grid-agnostic
  kernels (iwm K on (N,nl) columns + additive splice into fesom impl_vert_diff
  needs one Kv-profile hook in fesom_jax; isf via the shared host applicator).
  FESOM bbl explicitly DEFERRED with rationale: third unstructured port,
  pending A3's edge formulation proving out on MPAS first.
- Acceptance metrics defined: 30d arms validate CODE parity only (regression
  vs baseline within band-table tolerances); climate-level equivalence is the
  matched 180d x3 with d30/60/90 GATEWAY scores + plateau trajectories; the
  residuals ledger (EOS, TKE closure phase 1, vertical coordinate, GM/visc
  settings, bathymetry/strait geometry, runoff placement, forcing cadence)
  is carried in this doc and each item is measured, not assumed away.
Rejected/deferred (with reason):
- "Confirm GATEWAY EOS basis" — carried as a check in B1 (grep nameos), cheap.
- GLM's dcEdge/dvEdge + ssh-including bottom depth + dye/conservation tests
  for A3: adopted as A3's test list verbatim.

## Known open risks (flagged, not hidden)
- bbl edge-port numerics (A3) and the B3 sign/unit translator are the two
  highest-defect-risk pieces; both get analytic tests before any GPU run.
- fesom EOS is JM95/EOS-80 vs tripole TEOS-10-ish NEMO polynomial — an
  irreducible lane difference for now; documented, measured, not silently
  "fixed".
- fesom TKE != legoESM TKE in phase 1 (B5) — deliberate staged choice.
