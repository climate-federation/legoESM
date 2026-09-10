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
- 05:20 LAUNCHED from frozen _base2_wt @2c3258cc8 (short partition): trp_base2 9657917->9657918 (two
  legs, --kprofile-snapshots), trp_fwfix 9657919->9657920 (fixed real closure, one variable vs base2),
  mpas_base2 9657921 (one leg, K_zeta 0 + twin), fesom_base2 9657922 (mesh_nemo75_30m). Pre-registrations
  in the card headers. Outputs results/omip_nemo/{trp_base2,trp_fwfix,mpas_base2,fesom_base2}_d30.
- 12:25 fesom_base2 done: SST 0.62 (was 0.65), SSS 0.88, nino3 +1.39 -> pre-reg met. Tripole legs 1 done
  (day 15, snapshot_day0015 stamped 15.0; fwfix SSS 34.223 vs base2 34.214). Mixing profile vs NEMO at
  day 15 (job 9663450, card _eq_diffusivity_base2): surface avm x0.54 (cold tongue) / x0.39 (warm pool),
  entrainment avt x0.32 / x0.33, entrainment Pr 11.5 vs 2.6; fwfix == base2. Closure code is faithful
  on NEMO's state (08-27) and the now2 shear arm was refuted for the cell (08-18), so this is a state
  difference (higher Ri), not the shallow-cell operator. Full table in
  results/omip_nemo/threegrid_unified_table_2026-09-04.md. Open faithfulness ASK: card
  `--tke-mxl-choice 3` vs ORCA1 nn_mxl=2 (= choice 4).
- 15:30 mpas_base2 done (9657921): SST 0.49/-0.01, SSS 0.52/+0.02, nino3 +1.00, MLD 23.8/+5.1. The
  previous MPAS baseline rescored with the same command (card _score_mpas_unified_d30, job 9664889):
  0.50 / 0.54 / +1.08 -> pre-reg met, small improvement on all three, no regression. (The table's
  earlier 0.34 SSS for MPAS d30 was a different scoring.) Maps sent.
- 17:20 trp_base2 done (9657918; maps sent): SST 0.500/+0.06, SSS 0.427, nino3 +1.12, nino34 +1.00,
  Antarctic SST +0.22, Arctic -0.07, NH-mid -0.33; EUC 200/220/240E 0.347/0.204/0.226 (= rung 1).
  Control trp_unified180 rescored like-for-like (card _score_prev_d30 ARM=..., job 9665611): SST
  0.511, SSS 0.426, nino3 +1.30, polar SST identical -> pre-reg met except nino3 by 0.02 (ceiling
  +1.10). Box budget on base2 (9665616): day-15 50-150 m meridional -14.0 Sv (control -12.9, NEMO
  +4.6) -> the shallow cell is unchanged by the re-baseline, as expected. max|v| 2.57 m/s near
  (-0.7, 260E) at d30 (control 2.03, rung 1 2.59): east-Pacific equatorial spike grows with the
  equatorial A_h 500; PLAUSIBLE grid-scale, not opened.
- 17:30 NEW READING of the slab table (job 9665615): at 220-240E our SST is 25.9 vs NEMO 23.2 while
  our 0-50 m mean is 22.0 vs NEMO 22.3 (colder). Direct profile: ours falls 25.9 -> 22.2 over the
  top 19 m, NEMO 23.2 -> 22.5. The cold-tongue warm bias is a thin surface LENS over water that is
  already as cold as NEMO's; a top-50 m mixed like NEMO's would give ~22.9, i.e. slightly COLD.
  Both cards run --dm2dc and --sw-rgb-chl (RGB branch confirmed live in ocean_pe_latlon_cgrid).
  Probe extended (--surface-profile, card _surface_lens.sbatch, job 9666071: three grids, d15/d30,
  four boxes) to see whether the lens is shared and when it forms.
- 18:00 trp_fwfix done (9657920; maps sent): SST 0.497, SSS 0.385 (base2 0.427; tropics 0.155 vs 0.377), Arctic
  SSS -0.016, Antarctic +0.085, nino3 +1.10, no clamp -> every pre-registration met; the real-freshwater closure
  with the dilution fix now beats virtual salt. Falsifier rerun (9666730): open-ocean virtual-real dilution
  difference 0.0005 PSU (closures agree; regression degenerate); NEMO freshens river plumes 0.21 more than both.
- 18:05 LENS confirmed on all three grids (9666071; table). d30 5-65 m avm in the cold tongue x0.003 of NEMO
  (9666727) -> the lens has strangled the TKE closure. Mechanism: ORCA1 ln_zdfevd (100 m2/s tracers, rn2<=-1e-12,
  MIN(now,before), jk=1..jpkm1) fires nightly; our production convection.scheme='none' on all grids. Claim sent to
  codex (9666664) and GLM BEFORE code (docs/ocean/fidelity/lens_claim_2026-09-06.txt): both accept EVD as the primary
  analogue and the cells-1-2 interface; both demand the exact trigger wiring (threshold, two-level, now-geometry), a
  runtime firing-fraction/K-at-top-interface diagnostic, an imposed-inversion firing test (GLM: the 2026-08-01 no-op
  is unexplained by the compressibility offset alone), and a sampling-safe primary endpoint. Offline firing test =
  card _evd_lens_trigger (job 9666892). Decisions for the user queued (EVD arm config, driver flags, endpoint, mxl
  choice, fwfix adoption).
- 18:30 Offline firing test (9666892): in the 220-240E and 240-260E boxes the day-30 lens is stable as written
  (NEMO bn2 N2 +2.6e-4 at the first interface, firing fraction 0.000); after 4 h of 100 W/m2 cooling on the
  top cell alone, fraction 1.000 at the first interface, 0 below (N2 -6.6e-4). The trigger WOULD fire nightly
  on our own state. The 2026-08-01 EVD arm (nemolev_trp_evd100, run dir deleted) moved the tropics SST bias
  only +0.12 -> +0.09; the compressibility offset (+4e-5) cannot explain a no-op at N2 -6.6e-4, so the runtime
  firing-fraction diagnostic both reviewers asked for is the discriminator, not a rerun of that arm. Fourway
  figure (9666909) sent. Decisions queued for the user.
- 19:10 Item 2 (pure addition) done and dual-reviewed: run_omip_core2 flags --convection-n2-mode/-n2-eos/
  -trigger/-n2-threshold (build_enhanced_diffusion_config; two-level REFUSED: no leap-frog outer integrator
  here, now-only as NEMO key_RK3) and --evd-occupancy-every-hours (hourly K read through
  diagnose_vertical_K = the solve's own additive uncapped K; columns evd_top_occ_eq/glob + evd_top3_occ_eq,
  allowed on a control at 50 m2/s). Fixed from review: csv column order with ice (codex CRITICAL), silent
  no-op on a missing K key, 0.0==False flag trap, resolved-manifest gate in the arm card (occupancy alone
  cannot tell the hard bn2 trigger from the legacy smooth one). Tests 21 passed. Arm card _trp_evd_d30
  drafted with pre-registration; NOT launched (user decision 1 pending).
- 2026-09-07 12:45 EVD ARM LAUNCHED (jobs 9676756 -> 9676757, frozen tree _evd_wt @500483876, outputs
  results/omip_nemo/trp_evd_d30, ~5 h/leg). One variable vs trp_base2: ORCA1's ln_zdfevd (rn_evd 100 on
  tracers, nn_evdm 0) with NEMO's own bn2/TEOS-10 hard trigger at -1e-12.
  THREE-WAY REVIEW BEFORE ANY CODE: GLM approve-after-registry-edits, and it settled the coefficient
  question -- NEMO's published mean avt is rn_evd x duty cycle, so K_conv=100 is the namelist value, not
  a fit; codex all MINOR (backward Euler stable at 100 m2/s, K_bg=0 removes only the EVD stable branch,
  nu_conv=0 leaves momentum alone, csv append and manifest path fine); Claude CRITICAL that
  diagnose_vertical_K carries no jit of its own, so the hourly sampler was an eager eORCA1 step.
  THE PREFLIGHT THEN CAUGHT THREE MORE, each in minutes rather than hours into a 14 GPU-hour arm:
  (1) even compiled the K sampler cost 42 s/call (~17 steps) -> instrument swapped to the TRIGGER
  (NEMO bn2 at the top 3 interfaces from a 4-level T/S slice; under a hard trigger that IS the firing
  fraction, and a control can be measured too, at rn_evd/2);
  (2) that sampler read z_center_ref, which OceanPartialCellCoordinate does not carry -> ladders now
  come from the shared nemo_bn2_depth_ladders, the same helper the trigger itself uses, with a
  partial-cell regression test;
  (3) the run's reported rate is CUMULATIVE and so was swamped by the compile (0.066 steps/s on 63
  steps) -> the preflight now derives the incremental rate between diagnostic rows and refuses to pass
  a leg that would not fit its wall clock. Measured steady rate WITH the sampler: 0.471 steps/s
  (control 0.42-0.44), i.e. the instrument is free; and 13 h exceeded the short partition's 12 h
  ceiling, which parks a job in PartitionTimeLimit forever -> 11:55.
  Preflight also confirmed the resolved config is exactly the NEMO trigger and that the instrument
  discriminates: over a full diurnal cycle on the spin-up state, cold-tongue firing 0.61 mean / 0.77
  max at the first interface, 0.81 over the top three, global 0.31.
- 2026-09-08 02:50 CONVECTION ARM CLOSED, NEGATIVE, AND THE FOLLOW-UP CLAIM CLOSED TOO. (1) The 30-day
  arm met its pre-registered REFUTE condition: lens 3.87 vs control 3.80 vs NEMO 0.76, every regional
  band identical to 2 dp, while firing (0.60 nightly) and delivery (24.0% of wet columns at >=50 m2/s,
  max 101.8, control 0.000%) were both confirmed. The inversions it acts on are ~1e-6 K (control
  1.4e-3, max 0.01 anywhere), so the firing fraction flattered it. (2) The diurnal pair removes the
  last caveat: scheme on and off give identical cycles at every hour (min 2.15 both), our amplitude
  0.43-0.52 matches NEMO's 0.45, and the gap is a 1.75 K offset around the clock. (3) The reframing
  that suggested ("top-20 m mixing deficit") is refuted by GLM's own discriminator: 0-20 m tracer
  diffusivity ratio 1.04 (daily) / 1.62 (matched hour) -- our mixing is equal or stronger.
  INSTRUMENT WARNING: that probe's heat-content line does not align layer thicknesses; do not quote it.
  CODEX CRITICAL, open: the card's mixing-length choice differs from ORCA1's namelist, weakening
  ours-vs-oracle attribution for mixing questions (decision item 4, recommended to the user).
  NEXT: the remaining candidate is vertical advection, reconnecting to the earlier box budget (return
  flow -13/-14 Sv at 50-150 m vs NEMO +4.6; NEMO's ascent 37 Sv uniform to 286 m; sigma0>=25 surfaces
  sinking 5-17 m where NEMO lifts 6-12 m). Put that to codex+GLM before any code.

## 2026-09-08 — momentum mixing is NOT matched (codex CRITICAL, then measured)

The 0-20 m mixing comparison had only ever been run on HEAT. Codex's review of
the advection claim pointed out that a matched tracer diffusivity says nothing
about the momentum one, and the momentum one is what sets the undercurrent.
Extending the same probe (same box, same file, same hours) gives:

| 0-20 m, cold tongue 220-240E | ours | NEMO daily mean | NEMO matched hour |
|---|---|---|---|
| tracer diffusivity [m2/s] | 2.55e-03 | 2.46e-03 | 1.58e-03 |
| momentum diffusivity [m2/s] | 5.75e-03 | 1.01e-02 | 7.06e-03 |
| momentum / tracer | 2.25 | 4.09 | — |

So we mix heat correctly and momentum at roughly half strength, and the split
between the two is off by a factor 1.8. This is the same closure the
mixing-length mismatch lives in (our card runs choice 3 while ORCA1's namelist
sets nn_mxl=2 = our choice 4), and it is a direct candidate for the weak
undercurrent (0.20 vs 0.55 m/s at 220E). It does NOT reinstate enhanced
diffusion: that refutation was arm vs control on identical cards.

Also this day: the probe's 0-20 m heat-content block was DELETED rather than
repaired. The two sides integrated to different actual depths and both were
divided by a nominal 20 m, so its +5.16 K was the misalignment, not the ocean.

## 2026-09-08 — the offline heat budget is NOT a usable discriminator

Codex and GLM both proposed the same next measurement: split the layer's heat
budget into horizontal advection, vertical advection and vertical diffusion on
each side and see which term differs. It was built (six manufactured-solution
tests pin the terms and the land mask), and it FAILS ITS OWN CONTROLS for
reasons that live in the archived data, not in the arithmetic:

- OUR side stores INSTANTANEOUS fluxes. The three terms are therefore checked
  against a 15-day mean tendency, and the residual at 15-25 m (+3.3 K/month)
  is larger than every term including the vertical one under test. By the
  probe's own pre-registered rule that voids the comparison.
- THE ORACLE publishes 5-day MEANS, and a mean diffusivity times a mean
  gradient is not a mean flux. Where mixing is intermittent the two are
  strongly anticorrelated: avt at 50 m in the cold tongue has median
  4.1e-04 m2/s and max 3.8e+01. The reconstruction gives +852 K/month of
  diffusive heating at 50-150 m -- an ocean that would boil.

Two real instrument defects were found on the way and fixed (a sea-floor fill
temperature read as a gradient, worth a factor 16 in a synthetic case; and no
visibility of single-cell outliers). Neither was the cause of the 852: that is
the product-of-means limitation above, and it cannot be fixed offline.

WHAT WOULD MAKE IT WORK: time-averaged fluxes on our side, i.e. a 15-day rerun
of the production tripole card writing DAILY snapshots so the fluxes can be
averaged over the oracle's own 5-day window. That is a compute request and is
NOT made unilaterally.

## ★★★2026-09-08 — the mixing deficit is REAL, it is just BELOW 20 m

Every mixing comparison in this campaign had been run on the top 20 m, and on
heat. Widening it by depth and to momentum changes the verdict completely
(cold tongue 220-240E, |lat|<=2, ours day 30 against the oracle's daily mean):

| band | tracer ours/NEMO | momentum ours/NEMO | Prandtl ours | Prandtl NEMO |
|---|---|---|---|---|
| 0-20 m | 1.02 | 0.58 | 2.25 | 3.93 |
| **20-60 m** | **0.009** | **0.052** | **6.91** | **1.15** |
| 60-160 m | 0.16 | 1.33 | 17.8 | 2.10 |

At 20-60 m we mix heat at ONE PERCENT of the oracle's rate and momentum at five
percent. That is the largest single discrepancy measured in this campaign, and
it sits exactly where the cold-tongue profile goes wrong: heat trapped above
20 m (we are 0.5-1 K too warm at 0.5 m) over water that never mixes upward (we
are 1.3 K too cold at 20 m). The oracle keeps a nearly neutral Prandtl number
there (1.15); ours is 6.9, i.e. we damp heat exchange far harder than momentum.

RETRACTION, stated plainly: "our mixing is equal or stronger, so the deficit is
refuted" was true ONLY for 0-20 m. The refutation does not extend below, and I
should have measured the band before generalising. The EVD refutation is
untouched -- that was arm against control on identical cards.

This also gives the mixing-length mismatch a measured motive: nn_mxl=2 bounds
the vertical derivative of the length scale, which is precisely what sets how
far turbulence reaches BELOW the surface layer, and it is the band where we
collapse.

### 2026-09-08, later — the band deficit survives GLM's objection, and a second mismatch turns up

GLM's strongest objection to the 20-60 m measurement was averaging order: ours
is one instantaneous field, the oracle's a daily mean, and for an intermittent
lognormal field that alone could manufacture a 100x gap. MEASURED and REFUTED
(job 9685498) -- the oracle's own hour-to-hour spread of the box-median
diffusivity is tiny:

| band | hourly min | hourly max | max/min |
|---|---|---|---|
| 0-20 m | 1.47e-03 | 3.41e-03 | 2.32 |
| 20-60 m | 9.52e-04 | 1.87e-03 | 1.96 |
| 60-160 m | 9.89e-07 | 1.09e-06 | 1.11 |

Every hour of the oracle's day sits within a factor 2 at 20-60 m, three orders
above our 1.15e-05. Averaging order cannot explain the gap; the deficit is real.

GLM's other candidate, the sub-mixed-layer TKE penetration (nn_etau), is ALREADY
ON in our runs -- our driver defaults to the ORCA1 card (etau_mode="below_ml",
rn_efr 0.08, Langmuir on) and the production card does not override it.

BUT checking it exposed a genuine mismatch. ORCA1 inherits nn_htau=1 from
namelist_ref, i.e. a latitude-dependent penetration depth
htau = max(0.5, min(30, 45|sin(lat)|)); our config default is "constant10m"
(nn_htau=0). At |lat| <= 2 the oracle's htau is about 1.6 m against our 10 m.
NOTE THE SIGN: this makes OUR sub-mixed-layer TKE injection reach DEEPER than
the oracle's, so correcting it would make our 20-60 m deficit worse, not better.
It is a faithfulness item, not a lever. NOT changed -- it is an ASK.

MECHANISM now visible in the numbers: our Prandtl number at 20-60 m is 6.91
against the oracle's 1.15. Ours is clamp(4.5*Ri, 1, 10), so 6.91 implies a
Richardson number near 1.5 where the oracle's implies about 0.26 -- our shear is
far too weak there, which is the weak undercurrent again. The tracer limiter
then suppresses heat mixing hardest exactly where the shear is missing. GLM's
consequent warning is worth carrying into the arm's reading: the mixing-length
flip could pass its 0.2 gate through the back door (deeper mixing lowers N2,
which lowers Ri, which unclamps the limiter), so the arm should ALSO be read on
whether the 20-60 m Prandtl number closes toward 1-2.

### ★2026-09-08 — CORRECTION: the band numbers were the EVD ARM at day 15, not the baseline at day 30

Codex found it: the band card hard-coded the EVD arm's day-15 snapshot while I
read its output as the production baseline at day 30, and compared it against a
NEMO hourly file from a DIFFERENT YEAR (RUN_TRD2 is 2001; our run window and the
GATEWAY files are 2000). Both halves of the comparison were mislabelled. The
card now takes the snapshot, the oracle file and the record selection as inputs
and prints all three.

Redone on the production baseline at day 30 against the MATCHED 5-day window
(GATEWAY grid_W record 5):

| band | tracer ours | tracer NEMO | momentum ours | momentum NEMO | Prandtl ours / NEMO |
|---|---|---|---|---|---|
| 0-20 m | 3.38e-05 | 1.54e+01 | 1.38e-04 | 2.36e-02 | 4.09 / ~0 |
| 20-60 m | 2.60e-07 | 1.93e-03 | 2.95e-06 | 2.37e-03 | 11.38 / 1.23 |
| 60-160 m | 1.68e-07 | 6.21e-07 | 2.95e-06 | 2.02e-06 | 17.49 / 3.25 |

WHAT CHANGES, stated plainly:
- "0-20 m mixing is matched (1.02)" is WITHDRAWN. That was the EVD arm, whose
  near-surface diffusivity is 75x the baseline's (2.55e-03 against 3.38e-05)
  precisely because enhanced diffusion was on. Our BASELINE mixes far less than
  the oracle at every depth.
- The oracle's 5-day MEAN diffusivity at 0-20 m is 15 m2/s. That is not a
  turbulence strength, it is the average of a knob that fires at 100 m2/s on
  most nights, so a ratio against it measures nothing. Means of intermittent
  diffusivities are not diffusivities.
- WHAT SURVIVES BOTH COMPARISONS, and is the defensible claim: at 20-60 m our
  mixing is orders of magnitude weaker than the oracle's, and our Prandtl number
  is far too high (11.4 or 6.9 against its 1.2). The DIRECTION is robust; the
  "one percent" figure is NOT, and should not be repeated.

Status of the claim is downgraded from CONFIRMED to PLAUSIBLE pending a
comparison in which both sides resolve the intermittency.

## ★2026-09-10 — THE SHEAR ARM IS A NULL. THE SHEAR HYPOTHESIS IS RETIRED.

Job 9685519 (leg 1) + 9689614 (leg 2), one variable against the production
baseline: TKE shear production moved from our centred-square form to NEMO's own
zdf_sh2 (face-native differences, product at NOW squared, coastal doubling), the
variant fixed by the oracle's key_RK3 build. Both legs rc=0, all fields finite,
and the leg-1 state digest differed from the baseline's, so the discretisation
was genuinely active — this is a real null, not a no-op.

Scored at day 30 against GATEWAY record 5, against the pre-registration written
before the run:

| gate | pre-registered | measured | verdict |
|---|---|---|---|
| PRIMARY 20-60 m tracer K | rise from 2.60e-07 by >=10x; refuted below 2x | 3.19e-07, a factor **1.22** | **REFUTED** |
| co-gate 20-60 m Prandtl | falls from 11.4 toward 1.2 | 9.36 | barely moved |
| secondary EUC core at 220E | 0.20 -> toward 0.55 m/s | 0.199 | unchanged |
| guardrail SST rmse | within 0.05 of the baseline's 0.51 | 0.499 | passed |
| guardrail nino3 bias | <= +1.10 | +1.15 | marginally breached |

Momentum diffusivity at 20-60 m also did not move (2.98e-06 against the
baseline's 2.95e-06) — still pinned at the background.

VERDICT, stated plainly: the shear-production discretisation is NOT what keeps
our turbulence dormant below 18 m. Taken with the earlier face-native arm, which
was null on the vertical velocity, the meridional transport and the Pacific
index, **the shear hypothesis is retired.** Two arms, two different measured
quantities, both null.

WHAT THIS LEAVES. The mixing collapse below 18 m is real and unexplained: our
diffusivity sits at the background from 18 m to 155 m while the oracle carries
1.9e-03 at 20-60 m, and neither the enhanced-diffusion knob, the mixing length,
nor the shear discretisation moves it. The remaining candidates from the ledger
are the eddy coefficient (staged, awaiting the user) and the tracer advection
scheme; but neither is a turbulence term, so the honest position is that the
production defect has not yet been localised to a named term.

INSTRUMENT NOTE: the band probe's turbulent-layer-depth block raised
StopIteration on this run — the 5-day grid_W file carries no temperature
variable, so the mixed-layer part cannot run against it. The band ratios above
are unaffected. Fix the guard before quoting that block again.

## 2026-09-10 — the heat budget CLOSES at 0-25 m, and advection is not deficient

With the daily snapshots averaged over days 11-15 against the oracle's matching
record, the 0-25 m budget closes: residual -5.3 K/month against terms of order
60. Ratios ours/NEMO: horizontal advection 1.29, vertical advection 1.25,
vertical diffusion 1.05.

So in the top 25 m our advection is slightly STRONGER than the oracle's, not
weaker, and the diffusion matches. The cold water at 20 m is therefore not an
advective shortfall in this layer — which, with the shear arm retired, removes
another candidate rather than adding one.

The 15-25 m and 50-150 m layers remain unusable: the oracle's mean diffusivity
there is contaminated by intermittent mixing and yields ~1000 K/month, the same
product-of-means artefact recorded on 2026-09-08. Only the 0-25 m row is quotable.

## 2026-09-10 — the collapse is NOT a clamp: energy and length are both free and both small

Decomposing our own diffusivity at day 30 in the cold tongue, using
K_M = c_k * l * sqrt(2e) with our c_k = 0.1:

| band | turbulence energy | implied length | K_M |
|---|---|---|---|
| 0-20 m | 1.80e-05 | 0.23 m | 1.38e-04 |
| 20-60 m | 2.83e-08 | 0.12 m | 2.95e-06 |
| 60-160 m | 3.86e-09 | 0.34 m | 2.95e-06 |

At 20-60 m the energy sits 283x above its own floor (1e-10) and the length 124x
above the mixing-length floor (1e-3 m). NEITHER IS CLAMPED. The closure is
running freely and simply producing very little, so every remaining hypothesis
has to explain small PRODUCTION, not a limiter, a floor, or a cap.

That also rules out the last reading of the floors question: we had established
the floors are faithful to the oracle; this shows they are not even active here.

## ★★★2026-09-10 — PRODUCTION IS NOT REACHING THE TKE EQUATION

GLM's discriminator, run offline on the baseline day-30 snapshot in the cold
tongue: per level, shear production P = K_M * M^2 against dissipation
eps = c_eps * e^1.5 / l (c_eps = 0.7, l inferred from our own K_M and e).

Above 18 m, P/eps runs 0.07-0.6 — the surface layer is dissipation-dominated,
as expected where wave breaking supplies the energy. BELOW 18 m it inverts and
keeps climbing: 1.49 at 17.9 m, 5.68 at 24.6 m, 8.89 at 44.1 m, 16.16 at 73.2 m.

A closure in balance has P ~ eps. Ours has production exceeding dissipation by
up to a factor 16 while the energy sits flat at 2.83e-08 and the diffusivity is
constant to four significant figures over 140 m. Energy that large and that
persistent cannot be being added and then removed; it is NOT ENTERING THE
EQUATION THAT SETS e.

This reframes the whole campaign: the five eliminated candidates were all
physics hypotheses, and this says the defect is upstream of the physics — a live
source with no leverage on the prognostic variable. GLM's reading, which
predicted exactly this signature: a production term dropped inside the implicit
/ tridiagonal TKE solve; index offsets, wrong time levels and stale reads shift
or lag structure but cannot erase vertical correlation the way a missing
interior source does.

Status: the P/eps numbers are CONFIRMED from the snapshot; the attribution to a
dropped source is PLAUSIBLE and is the next thing to read in the code, not to
run. The shear arm's 1.22x is consistent — changing how production is
DISCRETISED cannot matter if production is not reaching e at all.

### ★CORRECTION, same day: the P/eps imbalance is REAL but SMALL in its implications

I wrote that production "is not entering the equation". That overstates what the
ratio supports, and I am withdrawing it. Dissipation goes as e^1.5, so a
production-to-dissipation ratio of 16 implies an equilibrium energy only
16^(2/3) ~ 6x above ours, not the orders of magnitude the diffusivity is short.
Working it directly: eps = P at 20-60 m gives e_eq ~ 7.5e-08 against our
2.83e-08 — a factor 2.6.

Also checked in the code rather than assumed: the shear production is an
EXPLICIT source in the right-hand side (rhs = e_old + dt*(P_s + buoy_source)),
so it is not dropped from the solve. GLM's "missing interior source" reading is
not supported by the source.

WHAT THE ARITHMETIC ACTUALLY SAYS. With our energy, K = 0.1*l*sqrt(2e) needs
l ~ 80 m to reach the oracle's 1.9e-03; with an oracle-like energy of ~1e-06 it
still needs l ~ 14 m. Ours is 0.12 m. The LENGTH SCALE is the dominant
shortfall, and in this closure the length is itself set by the energy through
the buoyancy length sqrt(2e)/N: at our e and a thermocline N, that is ~0.02 m.
So the two collapse together, which is why no single physics knob has moved it.

This is consistent with codex's CRITICAL, which names the same feedback from the
other end: if the coefficient consumed by the closure is the freshly-collapsed
one rather than the carried previous-step value, production is evaluated with an
already-dead K and the loop sustains itself.

STATUS: the P/eps ratios stand as measured. The "production never arrives"
attribution is RETRACTED. The live question is now narrow and checkable in code:
which diffusivity does our shear production consume, the carried one or the
recomputed one.
