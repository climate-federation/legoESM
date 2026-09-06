# Three-grid harmonization plan (user directive 2026-09-05)

Directive: same resolution for FESOM; one lateral-viscosity strategy across the three grids
(pick the best of: Laplacian 1e4 + Smagorinsky [tripole], Laplacian 1e4-1e5 + Smagorinsky
[MPAS], flow-aware biharmonic [FESOM]); one vertical closure (our TKE vs CVMix-TKE); FESOM on
the 75 NEMO levels; no regression vs the 2026-09-04 virtual-salt unified arms; delete old
test runs; codex + GLM review.

## Reference for "no regression"
The 2026-09-04 unified virtual-salt arms (trp_unified180, mpas_unified_180d) and fesom_b5_d90,
scored at GATEWAY days 30/60/90 (results/omip_nemo/threegrid_unified_table_2026-09-04.md and
baseline_trajectory_2026-09-04.md). Every change below runs as ONE-VARIABLE arms against them at
matched days. The real-freshwater closure stays OFF (its dilution defect is reviewed and open).

## P1 Lateral viscosity — decide by measurement, then unify
NEMO ORCA1 itself runs a LAPLACIAN with a prescribed file (eddy_viscosity_3D.nc: 20000 m2/s
mid-latitude -> 1000 at the equator), i.e. none of the three current strategies is the oracle's.
Ladder on the tripole unified card, 30 days, one variable each, scored vs GATEWAY rec 5 (SST,
SSS, nino3, EUC from the new u_east snapshots):
  (a) ctl: Laplacian 1e4 + Smagorinsky schedule (= yesterday);
  (b) NEMO file profile (--A-h-profile-file, A_h 2e4 mid-lat, Smagorinsky off);
  (c) biharmonic only (B_h scaled ~dx^3 to FESOM's value).
"Best" = lowest SST/SSS rmse with the undercurrent closest to NEMO; (b) is the faithful one by
construction. The winner is then ported: MPAS needs the latitudinal profile plumbing (only an
equatorial boost exists), FESOM needs a Laplacian option in fesom_jax momentum (only opt_visc=7
biharmonic is ported). Both ports get the same one-variable arm before adoption.

## P2 Vertical closure — decide by the single-column twin
Our TKE is the NEMO zdftke port (tendency-matched to ORCA1); CVMix-TKE is FESOM's own Gaspar
implementation. Reuse scripts/validate/ocean_fidelity/frozen_column_tke_twin.py: same column
state + forcing into both closures, compare K_v and MLD against NEMO's avt at the same column.
If ours is closer, FESOM gets our TKE through the existing closure bridge (the iwm splice
pattern); otherwise the tripole/MPAS closure question is reopened.

## P3 FESOM on 75 NEMO levels
fesom_jax loads per-node/per-element level counts (nlvls/elvls) that FESOM's Fortran mesh setup
derived from node depth + the 47-level zbar. No FESOM build here, so: infer the level rule from
the shipped nlvls.out/elvls.out (self-check must reproduce all 126,858 node counts on the 47-level
ladder before it is trusted), regenerate aux3d/nlvls/elvls with NEMO's 76 interfaces, run
prepare_mesh.py, then a FESOM b5-style 30-day arm on the new ladder (dt may need to drop from
1800 s for the 1 m top cell — measured, not assumed). One variable vs fesom_b5_d90 at d30.

## P4 FESOM resolution
Only the CORE2 mesh exists locally (nominal 1 deg, 25 km equatorial refinement, ~126 k nodes)
vs eORCA1 (1 deg, 1/3 deg equatorial). Comparable nominal resolution; an identical-resolution
FESOM mesh needs an external mesh generator (not available here) — flagged, not attempted.

## P5 Space
Deleted (this session's scratch, 22 GB): l8bisect_*, mpas_fwnemo_l8_diag1d, mpas_fwnemo_l8_180d.
Candidates for user approval: 48 GB of >14-day arms (nemolev_ico7_*, nemolev_mpas*, legoesm_e025_*)
and 43 GB of late-August daily-snapshot probes (nemolev_trp_{kmean,bilin_dailysnap,
std1deg_dailysnap,windsonly,modfrc,iceab20}_d30). Kept: everything referenced by a table/memory.

## GPU budget
Three 48 GB glab1 GPUs hold the real-freshwater pair + twin (verdict in hand: closure defective
for SSS, twin divergence 1e-3 C). The level-8 chain (K_zeta_bih=0) is queued behind them.
P1 needs 3 short arms; P3 one; P2 is CPU. Proposal: cancel the real-FW trio to free the GPUs.

## Review dispositions (2026-09-05 02:00)
GLM: day-30 cannot rank viscosity (EUC/WBC need 100+ d) -> ladder runs to 90 d; (b) as written
changes 3 things (profile, base 1e4->2e4, Smag off) -> hold the ramp/Smag/dt, vary the background
only; P2 metric = MLD evolution through a forced sequence, run P2 AFTER P3 on the 76-interface
column; P3: a 1 m top cell at dt 1800 is a vertical-advective-CFL risk (w~1e-3 -> 1.8 m/step) ->
CFL census from baseline w before any arm; reproducing node counts validates the counter, not the
depth rule -> also check reconstructed bottom depths.
codex: --A-h-profile-file is a surface zonal-median, depth-invariant, latitude-only proxy applied
as a multiplier after the operator (run_omip_core2.py:942, ocean_pe_latlon_cgrid.py:3247) -> NOT
NEMO's 3-D coefficient in flux form; faithful option needs a prescribed 2-D/3-D field inside the
Laplacian flux. Ladder: profile-only atop the unchanged schedule -> Smag off -> 2e4 endpoint;
drop the biharmonic arm. P2 harness frozen_column_tke_twin.py has only control|nemo modes and no
MLD -> needs a CVMix adapter + matched prognostic column trajectories. P3: verify elvls/nlvls/
aux3d linkage and the minimum-level policy, not just counts.

## Status 2026-09-05 04:10
- P3: FESOM2 find_levels ported from fvom_init.F90 and verified EXACT (0/244659 elements,
  0/126858 nodes on the shipped 47-level files); NEMO 75-level bundle at data/fesom2_core2/
  mesh_nemo75 (nl=76). CFL census on the tripole baseline: |w_1| max 2.5e-5 m/s => top-cell CFL
  0.044 at dt 1800 (GLM's 1.8 m/step worry refuted) -> dt kept. Arm fesom_nemo75_d30 running
  (one variable vs fesom_b5_d90). OPEN: min-column policy (FESOM's 5-level floor = 4.5 m on this
  ladder vs 30 m on its own; 30 m = NEMO level 14) -> user.
- P2 CORRECTION: the FESOM lane already runs OUR TKE (--fesom-vmix legoesm_tke, Dirichlet surface
  BC, mxl 3, prognostic) -- the earlier "CVMix-TKE" row was wrong. Vertical closure is unified;
  a CVMix comparison is optional, not a harmonization gap.
- P1: waits for the GPU decision (real-FW trio) and the ladder go.
- 05:00 P3 RESULT (fesom_nemo75_d30 vs fesom_b5, day 30, GATEWAY rec 5): SST rmse 0.54 -> 0.65,
  Nino-3 +0.41 -> +1.39, Nino-3.4 +0.53 -> +1.13, SH-mid/Antarctic SST +0.1 warmer; SSS unchanged.
  FESOM on NEMO's ladder reproduces the tripole/MPAS early cold-tongue warm transient => the
  "FESOM Pacific advantage" was the coarse 47-level ladder, not viscosity (P1 premise weakened:
  the viscosity ladder is now a faithfulness question, not a cross-grid difference).
- 05:30 P1 MEASURED PREMISE (job 9648737, day 30, unified tripole vs GATEWAY rec 5): Z20 gap
  -5 m (tilt 102% of NEMO) = thermocline depth matched; undercurrent core HALF of NEMO's
  (200E 0.289 vs 0.496; 220E 0.183 vs 0.546; 240E 0.201 vs 0.260 m/s); 10 m surface flow
  matched (240E -0.80 vs -0.79). Real-freshwater arm identical => not a freshwater effect.
  Unified card runs a flat 1e4 background (10x NEMO's equatorial 1000). RUNG 1 launched
  (job 9648801, _trp_visc_r1_d30.sbatch, pin 152927318): ORCA1 file SHAPE at unchanged 1e4
  mid-latitude background (~500 at 0N), Smag ramp/dt untouched, 30 d, pre-registered in the
  card header (EUC 200E >= 0.40 confirm / < 0.32 refute; nino3 <= +1.05 vs ctl +1.30).
- 06:50 THREE-GRID UNDERCURRENT (threegrid_unified_table_2026-09-04.md, EUC section): FESOM-75 = NEMO's
  undercurrent (1.14/1.00/1.18 of NEMO at 200/220/240E) yet nino3 +1.39 (tripole 0.58/0.34/0.77, nino3
  +1.30) => the shared day-30 equatorial warm transient is NOT undercurrent-limited; rung 1's nino3
  pre-registration (<= +1.05) is now EXPECTED TO FAIL and the arm is judged on the EUC alone (a
  circulation-faithfulness lever for the tripole, not a cold-tongue lever). Candidate for the shared
  transient: the common column (TKE Prandtl at the 10 ceiling vs NEMO 1.2; ORCA1 nn_mxl=2 vs our 3) --
  see omip_gateway_reference_2026-08-22.md "STILL OPEN" 2-3; unmeasured on the unified card.
  INSTRUMENT FIX: equatorial_thermocline.py U block now REQUIRES --nemo-w-recs (it silently averaged all
  18 records when the flag was dropped with the w block: FESOM read as 0.78 of NEMO at 220E, matched = 1.00).
- 07:40 SHARED TRANSIENT LOCALISED (slab tendency d15->d30, all three grids, table in
  threegrid_unified_table_2026-09-04.md): at 220-240E NEMO cools 0-150 m and lifts Z20 9 m; ours warm
  50-150 m (+0.6..+1.0) with Z20 flat/deeper => missing equatorial ASCENT at 220-240E, common to all
  three grids (FESOM included, despite its NEMO-strength undercurrent). Consistent with the 08-18/08-23
  finding that the ORCA1 viscosity SHAPE (off-equator ramp) restored the 240E upwelling and Ekman v(lat).
  RUNG 1 PRE-REGISTRATION EXTENDED (before its d30 lands; resolved config verified: eq 500, midlat 1e4,
  Smag ramp unchanged): CONFIRM if 220-240E dT(50-150 m) d15->d30 <= 0 (ctl +0.94, NEMO -0.51) AND
  dSST <= +0.2 (ctl +0.68); REFUTE if dT(50-150) >= +0.6. The EUC threshold (>=0.40 at 200E) stands.
  If CONFIRMED the shape is the lever on all three grids: MPAS needs the latitudinal profile plumbing
  (only an equatorial boost exists) and FESOM a Laplacian-with-profile option -> ports are the next
  arms; if REFUTED the ascent defect is not lateral friction -> instrument the 220-240E w budget.
- 08:00 GLM CLAIM REVIEW (task kjnu33zqi): (1) CRITICAL the slab table does NOT discriminate ascent from
  horizontal convergence or enhanced downward diffusion -> offline discriminators named; the cheapest on
  existing snapshots = T on sigma0 surfaces (heave keeps it, diapycnal mixing changes it) -> added
  (--isopycnal, job 9649107); (2) MPAS's near-NEMO 0-50 m cooling with the weakest EUC weakens "one
  shared mechanism"; (3) downward mixing NOT ruled out (the closure twin used NEMO's shear, not ours);
  (4) non-friction setters: wind-stress curl 2-5N, vertical viscosity below the ML, Yoshida spin-up;
  "lateral friction = Ekman carrier" is dynamically loose. (c) CRITICAL the rung-1 gate was unreachable
  for a half-strength shape (the 2x arm reached only +0.38). PRE-REGISTRATION REVISED (still before the
  d30 lands, arm at ~day 22): LEVER REAL if 220-240E dSST(d15->d30) drops by >= 0.2 from ctl (+0.68 ->
  <= +0.48) AND dT(50-150 m) drops by >= 0.3 (+0.94 -> <= +0.64); NULL if both move < 0.1; CLOSES only
  if dT(50-150) <= 0. The absolute "<= +0.2" gate is withdrawn as a false-refutation risk.
- 08:30 HEAVE CONFIRMED (isopycnal depth d15->d30, job 9649116, table in threegrid_unified_table):
  T(sigma0) changes alike on all sides (mixing ruled out); NEMO lifts sigma0 23.5-26 by 6-12 m at
  220-240E, ours sink the sigma0 >= 25 surfaces 5-17 m (tripole/FESOM; MPAS 4-10). Missing ascent
  ~1.5e-5 m/s in the lower thermocline, confined to 220-240E, all three grids. codex CRITICAL-1 (slab
  cannot separate ascent from convergence/diffusion) answered for diffusion; ascent-vs-lateral-
  convergence is the same thing for a heave (continuity) but the SOURCE of the convergence (which
  velocity field) is open: next = 50-150 m box budget from the STORED mass_flux_u/v/w (tripole) and
  NEMO's uocetr/vocetr — which wall carries the anomalous convergence at 220-240E.
- 08:45 WALL ATTRIBUTION (box budget, job 9649505, table in threegrid_unified_table): same Ekman divergence as
  NEMO (33 vs 37 Sv, 0-50 m, day 15) but our compensating inflow is MERIDIONAL in 50-150 m (-12.9 Sv; NEMO +4.6)
  while NEMO's sits below 300 m (w 37 Sv uniform to 286 m). Zonal convergence matches per layer. => shallow
  tropical cell = the heave. RUNG-1 DAY-15 PRE-REGISTRATION (mechanism test, hours before d30): friction shape
  CONFIRMED as the operator if the day-15 50-150 m meridional term moves from -12.9 to >= -6 Sv AND w(155 m)
  from +11 to >= +18 Sv; REFUTED if both within 3 Sv of control. If refuted the next lever is the interior
  vertical viscosity below the mixed layer (avm 65-300 m vs NEMO's avm in grid_W recs 2:3) — needs the K dump
  (--kprofile-snapshots) on the unified card: a config choice -> ASK.
- 15:20 REAL-FW d90 CONCLUSION, dual-reviewed (codex job 9654158, GLM): the d90 result (SST equal, SSS
  worse under real_freshwater) CANNOT rank the closures: the real arm is an incomplete port (no surface
  dilution). Both: keep virtual salt PROVISIONALLY; the completed real-volume closure is the NEMO-faithful
  target by construction. Corrections to my claim: (i) codex CRITICAL: NEMO trasbc has no emp*sss term,
  sfx = ice salt only, emp enters volume; virtual salt is first-order equivalent ONLY with LOCAL S and live
  h — our production card runs freshwater_salinity=s_ref (fixed 35, "legacy") => leading-order mismatch
  at rivers/ice, not only at S->0. 'local' is rejected with the freshwater normalization (salt covariance)
  => the real closure is the only way to have both normalization and NEMO dilution. (ii) GLM CRITICAL:
  ice — NEMO freezing tendency ~ (S - S_ice) F_ice; the real arm supplies only the S_ice channel (~10%),
  explaining the Arctic flip quantitatively; the fix must KEEP the ice salt flux (not "zero surface tracer
  flux"); the virtual arm may double-count (S + S_ice) if the brine channel also fires — disambiguate.
  (iii) GLM MAJOR: the global SSS degradation is plausibly polar-driven, so the fix should recover most of
  it (my "local gain only" withdrawn). (iv) 0.004 PSU stretch term: present in BOTH arms, not a closure
  difference; second-order term (F dt/(rho h1))^2 reaches % in thin cells (MINOR). (v) one-column test:
  exact S_new = S_old h/(h + F dt/rho), both F signs, NEMO time-centering. FALSIFIER (GLM, cheap): regress
  NEMO's dS_top over 30 d against -S_bar sum(F) dt/(rho h1) on ice-free non-plume points; slope ~1 confirms
  top-cell dilution; then include ice points: slope (S-S_ice)/S vs (S+S_ice)/S exposes the double count.
  RULE-3 note: freshwater_salinity default 's_ref' is documented "legacy" => a defect-report knob.
- 16:20 RUNG 1 DAY 15 = REFUTED as the ascent operator (table in threegrid_unified_table): 50-150 m meridional
  -13.8 Sv (ctl -12.9), w(155 m) +11.9 (ctl +11.3), NEMO +4.6 / +36.7. Lateral friction shape is not what
  keeps the equatorial cell shallow. Day 30 (leg 2) still decides the EUC/nino3 lever question. Remaining
  candidates for the shallow return flow: interior vertical viscosity below the mixed layer (avm; the
  08-23 K instrument said ~10x NEMO at 65-105 m, caveated), the vertical momentum advection / adaptive-
  implicit scheme, the pressure-gradient scheme (smc03) — all need the K dump or a term budget; ASK before
  any arm.
- 16:20 DILUTION FIX (defcbe43f + reviews): GLM MAJOR + codex CRITICAL = evaporation/ice growth needs the
  UPWIND (cell-below) donor -> fixed (donor by flux sign; surface S=0/T=T_1 both signs); codex MAJOR =
  runoff must enter over NEMO's h_rnf (sbc_rnf_div) -> per-level entry profile (runoff_entry_profile,
  shared weights runoff_spread_layer_fractions); GLM (6) stratified partial-column both-sign test added
  (catches the H_below off-by-one and the donor). Falsifier (dilreg 9654288): virtual-minus-real vs
  NEMO-minus-real slope 0.61-0.79 open ocean, 1.2-1.3 at 30-60 lat, 0.3-0.6 plumes, ice 0.98 (30 m);
  r 0.3-0.55 => linearisation not falsified, virtual over-dilutes plumes as predicted. Open (codex
  PLAUSIBLE, needs a source trace): ice channel (S - S_ice) vs double count; emp*sst heat-content term
  in our bulk; AB2 / leapfrog-before / restart interaction of the post-advection correction.
- 16:40 DILUTION FIX round 1 applied + pushed (cad1c1e92): upwind donor by flux sign; runoff over h_rnf
  (per-level entry profile, shared level weights); per-channel entry temperature (rain/evap/restoring 0 degC
  since q_net carries their heat content = NEMO blk_oce_2/sbcssr; runoff at SST over h_rnf = rnf_tsc; ice
  melt + normalisation residual at local T, PLAUSIBLE). SOURCE TRACES: ice brine channel = TRUE ice salt
  content (salt_old - salt_stored)/dt, negative on growth, positive on melt (brine.py) => real closure gives
  (S - S_ice) x growth like NEMO sfx, NO double count (CONFIRMED); our q_net has -evap cp T_s + rain cp
  theta_air + snow terms (omip2_applicator.py 1090-1120) => the VIRTUAL closure (production) applies that
  heat content WITHOUT the compensating tracer dilution (NEMO ln_linssh adds emp*sst in trasbc) => spurious
  q ~ -(E-P) cp T ~ 2 W/m2 cooling where E>P (PLAUSIBLE, sent to round-2 reviewers). TRAP: numpy allclose
  default rtol=1e-5 on S=35 hid a 1.7e-6 PSU signal -> the step test passed on pre-fix code; rtol=0 now.
  Round-2 reviews: codex job 9654366, GLM task k5aa9zga2.
- 17:00 DILUTION FIX round 2 (codex 9654366, GLM): implementation PASS on both (upwind donor, entry-profile
  transport, MPAS shapes); codex MINOR stale comment fixed; interior sign change (river below an
  evaporating surface) is conserving and monotone -> test added. Open PLAUSIBLE: ice-melt heat parity with
  SI3 icesbc (GLM: SI3 subtracts no cp*T*fw, local-T entry is right). CONFIRMED by both as a PRODUCTION
  defect: the virtual closure applies the rain/evap/restoring heat-content terms in q_net with no
  compensating temperature term (NEMO ln_linssh: sbc_tsc(jp_tem) += emp*sst/rho0; our GYRE recipe
  nemo_recipe.py:1079 has it, OMIP block 8 does not) => spurious surface flux -(E-P) cp SST, ~2-6 W/m2
  cooling in the evaporative subtropics, warming under the rain belts (GLM: up to 10-15 W/m2 in trade
  regions). USER DECISIONS ASKED: (A) add the temperature twin to the virtual closure + re-baseline;
  (B) launch the fixed-real-closure 30-day arm.
- 22:10 RUNG 1 DAY 30 (table in threegrid_unified_table): surface lever only (EUC 0.29 -> 0.35, nino3
  +1.30 -> +1.10, SST rmse 0.51 -> 0.50, 240-260E surface warming halved); heave criterion FAILED
  (50-150 m warming +0.94 -> +1.05, sigma0 >= 25 sinking slightly larger). P1 CONCLUSION: the lateral
  viscosity strategy is not what separates our equatorial cell from NEMO's; rungs 2 (Smag off, ran away
  before) and 3 (2e4 endpoint) would answer a different question and are NOT recommended. Whether to
  ADOPT rung 1's shape (NEMO-faithful background, small gain) is a production config choice -> user.
  NEXT OPERATOR SEARCH for the shallow return flow: vertical viscosity/diffusivity below the mixed layer
  (K dump = --kprofile-snapshots on the unified card, one arm) and the vertical momentum advection /
  pressure-gradient scheme (term budget on the day-15 state) -> both need a run or a config flag -> ASK.
  TRAP: a 15-day leg writes its final state only as snapshot_final.npz; a symlink is overwritten by the
  next leg -> copy it (day 15 of rung 1 rebuilt from restart_leg1.npz, no mass fluxes).

## 2026-09-06 — all 11 decisions approved by the user; dual review of the batch
- (5) real-FW trio cancelled; (6) 82 GB / 68 dirs deleted (4.1 TB free).
- (3) temperature twin for the virtual closure (4eb1ac429): GLM CRITICAL "double count" REFUTED by the
  NEMO 5 source: trasbc.F90:141 adds emp*T1/rho0 under lk_linssh with no compensating removal from qns
  (none in sbcmod); sbcssr.F90:138 and sbcfwb.F90:294 put the restoring/correction water's heat in qns =
  our sss_restoring heat_flux; our normalisation water enters eta with no heat = NEMO's net. codex: signs,
  masks, MPAS single application, real-branch exclusion correct; restoring twin active only with the
  water-flux channel (fw.restoring None otherwise) = consistent with where the heat goes.
- (9)(10) vorticity filter (e74d8b7cd + follow-up): codex MAJOR the gate was only in step_checked ->
  step() is now a host wrapper (gate once, traced dt skipped) over the jitted _step_jit; bound 8/dv_min^2
  is heuristic (power iteration = future); ~20 MPAS cards now run filter-off by the recipe default
  (decision 9). (8) fesom rule: no findings. (11) src-first: no module overlap (comm). Cards: fwfix = 3
  closure flags only; base2 is a NEW BASELINE (4 changes, header says so); codex MAJOR fesom card had no
  pipefail -> fixed.
