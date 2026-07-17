# NEMO GYRE_BARE Differentiable-Oracle Fidelity — Plan & Resume

Goal: make legoESM `build_nemo_gyre_recipe` reproduce NEMO 5.0.2 `GYRE_BARE`'s
discrete numerics nut-for-nut, differentiability being the only intended
difference. Method (validated this campaign): build a complete **wiring diagram**
of both codebases, then match every scheme and parameter one at a time and verify
against a native NEMO GYRE trajectory. This plan is the resume point.

Branch: `feat/nemo-eos80-differentiable`. Companion: the full parameter table in
[`nemo_gyre_wiring_comparison.md`](./nemo_gyre_wiring_comparison.md).

## 1. Success metric
5-yr matched-forcing trajectory vs a native NEMO GYRE run (same dt=14400 s, IC,
forcing, calendar). Compare: wet-masked domain RMS `|U|`, surface WBC max|u|, and
the **velocity depth structure** (surf/deep rms(u) — the decisive diagnostic).
NEMO target: RMS ratio 1.0, surf/deep rms(u) ≈ 8.2 (surface-intensified).

## 2. Where it stands (2026-07-15)
- GYRE runs **stably** at NEMO's dt (was NaN by day 12 before the Coriolis fix).
- Thermodynamics + most single-step operators match; **7 code-derived fixes
  committed** (§4).
- Residual: **RMS ratio ~0.66**, and legoESM is **bottom-intensified**
  (surf/deep 0.4) where NEMO is surface-intensified (8.2).
- Root cause fully diagnosed (§5): a coupled **abyssal density drift → deep
  convection → downward momentum spreading**. Not a config knob.

## 3. The wiring comparison — status per component
See the companion doc for the full parameter table. Summary:

MATCHED (✓): horizontal grid + f0/beta, IC T(z)/S(z), 30-level thicknesses, flat
bottom, EOS-80 (certified 4.5e-13), lateral viscosity A_h=1e5, iso-neutral tracer
diffusion A_ht=1000 + kappa_GM=0, KE-gradient C2, ENE vorticity, bottom drag
(nemo_quadratic Cd0=1e-3/ke0=2.5e-3), TKE core params + background floors, EVD
convective-adjustment operator (hard N²<0), Haney+solar+E-P thermal forcing,
360-day calendar, free-slip lateral BC. **WIND STRESS now added (§4).**

MISMATCHES REMAINING (the worklist, §6): z-star vs `key_linssh`, tracer stepping
(forward-Euler vs RK3), PGF/coordinate at depth, barotropic filter (cosine+spatial
vs Demange nn_bt_flt=3), RK3 variant (Shu-Osher vs Wicker-Skamarock), plus minors
(rho_0 1025 vs 1026; TKE N² adiabatic vs in-situ; nn_mxl 2 vs 3 — deconfounded).

## 4. Committed fixes (branch feat/nemo-eos80-differentiable)
| commit | fix | effect |
|---|---|---|
| `70ab903c9` | couple Coriolis+PGF in RK3 (coriolis_scheme=explicit_ab2) | cured the dt=14400 blow-up (NaN → stable) |
| `ffc98144d` | faithful namelist knobs: const A_h=1e5 (no lat-scale/Smag), kappa_Redi=1000, fct2, nemo_quadratic drag, E-P freshwater | fixed the over-concentrated WBC |
| `bfc3126a4` | A_v=0/K_v=0 — remove vertical-mixing double-count (TKE floors supply avmb/avtb) | SST 17.7 → 19.5 (best NEMO match) |
| `456922278` | faithful ln_zdfevd (hard N²<0 threshold, smooth_transition=False, K=nu=100) | reproduces NEMO's warm well-mixed ~83 m layer |
| `a6f97134a` | the wiring-comparison document | — |
| `8be372059` | the missing double-gyre WIND STRESS (nemo_gyre_wind, exact usrdef_sbc, seasonal, top-layer momentum) | primary forcing now present |

## 5. Diagnosis of the residual gap (evidence)
1. **Driving is not the problem**: legoESM's zonal-mean SST gradient is 12%
   *stronger* than NEMO's, and its depth-integrated density gradient 2.36× — yet
   the flow is 0.58×. Stronger driver, weaker flow ⇒ dynamics/structure, not
   forcing. (`_diag_driving_vs_dissipation.py`, `_diag_density_gradient.py`)
2. **Velocity is bottom-intensified**: rms(u) grows with depth (surf 6.6e-3 →
   4150 m 3.4e-2 = 24× NEMO's bottom), surf/deep 0.19–0.40 vs NEMO 8.2.
   (`_diag_velocity_depth.py`)
3. **Density gradient is uniform with depth** (barotropic-like) vs NEMO's
   surface-intensified (baroclinic) ⇒ weak thermal-wind shear ⇒ weak surface WBC.
4. **Wind momentum spreads down**: adding wind didn't raise the surface flow;
   removing the momentum-EVD (nu_conv=0) doubled it (0.58→0.66), i.e. the
   deep-convection momentum-EVD spreads the surface stress down the column.
5. **Mechanism**: an abyssal density drift (deep horizontal gradient absent in
   NEMO) drives a deep geostrophic flow AND seeds deep static instability →
   deep EVD fires → homogenizes the column (barotropic density) + spreads
   momentum to the bottom. Self-reinforcing.

## 6. Remaining worklist (ranked; pick up here)
Each item: what NEMO does, what legoESM does, the fix, expected effect, effort.

**A. z-star → linear free surface (`key_linssh`) — DONE 2026-07-16
(128feaffb mechanism + recipe flip). THE ABYSSAL-DRIFT ROOT CAUSE, confirmed by
construction: the z-star sigma redistribution of deta/dt manufactured the deep
circulation. Acceptance: surf/deep rms(u) 0.20 → 5.91 (NEMO 8.2), bottom rms(u)
4.98e-2 → 2.7e-3 (18×, NEMO 1.4e-3), abyssal density contrast collapsed to
NEMO's. Scalar RMS now reads 0.502 — HONEST (the old 0.805 was inflated by the
spurious deep flow); the residual is a ~0.4-0.5 fairly-uniform amplitude factor
with the structure now NEMO-like. Remaining amplitude candidates: barotropic
Demange filter (item D), PGF 3.4% (C), minors.** Original description (was the
prime candidate): (2026-07-16: Redi ruled out — with Redi OFF the abyssal contrast grows
5-50× WORSE, i.e. Redi DAMPS the deep gradient; tracer integrator ruled out;
the drift is ADVECTIVE, pointing at the deep w / thickness-breathing difference).
NEMO holds layer thicknesses FIXED (ssh evolves but doesn't stretch the column);
legoESM uses z-star (thicknesses stretch). Most fundamental deep-ocean structural
difference; directly affects the deep pressure/coordinate. legoESM's
`OceanZStarCoordinate` has **no** `linear_free_surface` field — needs a real
implementation (a fixed-thickness mode or a frozen Jacobian). Effort: medium-high
(load-bearing coordinate change; design + review + conservation checks required).

**B. Tracer time-stepping: forward-Euler → RK3. TESTED 2026-07-16 — RULED OUT
as the drift lever.** The `_ssp_rk3_tracer_pair_step` path already exists (config
flip, `tracer_time_integrator="rk3"`); a 5-yr run is IDENTICAL to Euler (ratio
0.615, surf/deep 0.17, bottom rms unchanged). Still the NEMO-faithful choice and
empirically neutral — optional card adoption, but not the abyssal-drift seed.

**B'. NEMO Dirichlet surface-TKE BC (NEW, from the 2026-07-16 audit).**
The wind now reaches TKE (571530df2) but via the Veros flux BC
`surface_flux=(taum/ρ0)^1.5` — surface TKE ~7.7e-5, ~60× below NEMO's Dirichlet
`en(1)=max(rn_emin0, rn_ebb·|τ|/ρ0)` ≈ 4.9e-3. The formula exists in tke.py:1117
but only inside the inactive (nn_etau=0) sub-ML penetration path. Implement a
proper Dirichlet surface-TKE option (NEMO nn_bc_surf=1) in tke.py as a selectable
config; expect a deeper wind-mixed layer + stronger Ekman response. Effort:
low-medium (bounded, one module + tests + review).
**DONE 2026-07-16 (76b239532)**: TKEConfig.surface_bc="nemo_dirichlet", hard
Dirichlet in both solves, review SOUND (NEMO's en(1) confirmed hard-Dirichlet in
the Fortran). RMS ratio 0.61 → **0.805**; WBC peak **104%** of NEMO; mid-depth
rms(u) ratio 1.02. Remaining deficit = the deep abyssal-drift flow (item A).

**C. PGF / hydrostatic pressure at depth — CLOSED 2026-07-16 (comparison
artifact, machine-precision certificate).** The "~3% hpg deficit" was a
TIME-LEVEL artifact, not a PGF defect. NEMO RK3 recomputes `eos(ts, Kmm)` at
stages 2&3 (stprk3_stg.F90:322) and `tra_sbc_RK3` runs at EVERY stage, so the
dumped stage-3 `utrd_hpg` uses tracers advanced dt/2 from the step-start state;
our per-term probe compared at the unadvanced state. Certificate (single-column
probe `gap_audit/_probe_hpg_column.py` + the `rhd_stg`/`tn_stg`/`sn_stg`
stage-snapshot already captured by MY_SRC/trddump.F90):
(1) hand `zhpi(rhd_stg)` vs dumped `utrd_hpg`, full wet field: max diff
**5.3e-23** (field rms 5.8e-8) — the e3w ladder is bit-exact;
(2) legoESM `nemo_roquet_eos(tn_stg, sn_stg)` vs NEMO `rhd_stg`: 5.8e-9 on
rhd~2e-3 = **3e-6 relative**;
(3) legoESM `pgf_quadrature="nemo_trapezoid"` == hand-zhpi to ~1e-4 at the
initial state (a ~0.7% residual at abyssal levels where |hpg|~3e-9, ~0.2% of
field rms — negligible; likely bottom e3w detail).
Cross-check: zero-forcing rerun (MY_SRC/usrdef_sbc.F90 env switch
`NEMO_ZERO_FORCING=1`, EXP_NOFORCE) gives utrd_hpg ≡ 0 from the horizontally
uniform IC — all step-1 hpg structure is stage-generated, confirming the
artifact mechanism. Fair per-term hpg comparisons MUST use the stage snapshot
(`rhd_stg`), never the step-start state.

**D. Barotropic filter — DONE 2026-07-16 (483b6477a).** nemo_ab3am4
implemented (AB3 extrapolation + AM4 backward ssh interpolation alpha=0.07 +
final-value output + uniform transports; review: numerics EXACT, direct tests
added). The cosine window-averaging was retarding the gyre by ~dt/2 per step.
RMS ratio 0.50 → **0.71** (yr-1 0.93), surface rms +55%, surf/deep 11.6
(brackets NEMO 8.2). **Cross-window histories (sweep #5) DONE 2026-07-16**:
`state.bt_hist` carries the AB3/AM4 substep histories across windows in
DEVIATION form (X_final−X_b, X_final−X_bb) reconstructed against the new
window's now-values; NEMO's ll_init ramp is now cold-start-only (dynspg_ts
:200-226 exactly). Deviation form, not raw values: NEMO re-imposes the stp2d
barotropic mean on the 3D velocity after every stage (stprk3_stg.F90:440
zub correction) so its raw histories never see a window-boundary jump —
legoESM's post-solve implicit vmix can shift the depth mean, and a raw
carry would feed that jump into the ×1.781 AB3 extrapolation each window;
deviation form is jump-transparent and identical to NEMO's raw carry when
the mean is preserved (NEMO's invariant). Empirically BOTH forms are
trajectory-NEUTRAL on the 5-yr GYRE (yr-1 0.962 / yr-5 0.758 / identical
depth structure, matching the WS-RK3 baseline — the vmix mean shift is
small here); deviation form is kept on structural-safety grounds. Gate:
two carried windows == one continuous 2n-substep window at 1e-13 (f64).
NB an intermediate scare (surf/deep "collapse") was a STALE-DIAG artifact —
the gap_audit diag scripts hardcoded old npz paths; they now honor argv. NEW STRUCTURAL ITEM surfaced: NEMO's
per-stage barotropic-mean IMPOSITION (:440) — legoESM lets implicit vmix +
bottom drag modify the depth mean after the barotropic solve, NEMO discards
that increment from the state (it re-enters via the next step's slow
forcing). Candidate item F if the residual amplitude gap needs it.
Remaining minors: substeps 120 vs NEMO 50 (card already 50). Original
description:
NEMO uses temporal dissipation (AB3-AM4, rn_bt_alpha=0.07), NO spatial diffusion;
legoESM uses cosine time-averaging + `barotropic_diffusion_alpha=0.01` (spatial,
×120 substeps) which damps the WBC η-gradient. Also substeps 120 vs NEMO's 50.
Effort: medium (implement Demange in the barotropic solver).

**E. RK3 variant — DONE 2026-07-16 (e2b4b03a0, hardened 05027902e).**
momentum_time_integrator="rk3_ws" = NEMO stprk3_stg exactly (stages from u0,
dt/3-dt/2-dt, LDF stages 1&3 via skip_lateral_viscosity, ZDF stage-3-equiv).
THE FIRST NON-NEUTRAL sweep item: 0.722 → 0.758 (yr-1 0.962). Review SHIP; the
consolidated sweep review's findings (WENO+ene_total latent Coriolis-drop
guard, direct tests, constant promotion) are in 05027902e.

**F. SST cold bias / winter ML over-deepening — ROOT CAUSE FOUND + FIXED
2026-07-16: the card ran `tke_mxl_choice=2` (Veros buoyancy length), not
NEMO's `nn_mxl=3`.** Diagnosis chain: yr-5 column heat +0.54 C (heat BURIED,
not missing) + winter ML uniform to 209 m vs NEMO 75 m + failed spring
restratification + damped/lagged SST seasonal cycle; 15-day daily comparison
(new EXP_15D, nn_stock=6) shows the over-deepening from DAY 1; forcing
generators (t_star/qsr) verified machine-exact vs the Fortran; column
closure probe (`gap_audit/_probe_tke_column.py` — NB NEMO restarts dump
instantaneous `en`/`avt_k`/`avm_k`; the 122-day-mean `votkeavt` is
EVD-saturated and useless for closure comparison) seeded with NEMO's OWN
en/T/S: choice-2 K_H(10 m) = 0.21 vs NEMO avt 0.0297 (7x — the uncapped
buoyancy length gives l≈73 m in the near-neutral ML) while **choice-3
reproduces NEMO's avt to 3-4 significant figures at every level** (the
ln_mxl0 wind anchor seeds the |dl/dz|<=e3t lup ladder, zdftke.F90:645-712).
The old "nn_mxl=3 is NOT the lever" deconfounding (corr 0.9998 for levels
>=2) held only in the stratified interior. RESULTS: 15-day SST deficit
0.31 → 0.11, MLD 136 → 111 m (NEMO 91); 5-yr SST max 17.33 → **18.36**/18.16
(NEMO 18.98, same-file) for the l_k variant; the COMMITTED config (with the
l_eps dissipation-length review fix) reads SST 18.16, surf/deep 2.69 →
**3.87** (NEMO 8.23), mid-depth (1551 m) rms(u) 1.89x → **1.25x** NEMO,
ratio 0.710 (yr-1 0.935). The scalar wet-RMS ratio drop from 0.758 is an
HONEST change: the baseline number was inflated by
spurious mid-depth flow (same lesson as the 0.805 pre-linssh episode);
surface rms changed only 0.617 → 0.576. The depth-resolved profile is the
metric of record.

**G. Residual surface-circulation amplitude (surf rms 0.63x NEMO) — ROOT
DIAGNOSED 2026-07-16 as SUBTROPICAL THERMOCLINE EROSION, a vertical-mixing
lever (NOT a resolution floor, NOT a WBC/boundary-stencil issue).**
CORRECTION of two earlier mis-diagnoses in this section: (1) the "coarse-grid
Munk floor" framing was WRONG — both models run the IDENTICAL 32x22x30 grid
(masks bit-identical, 600 wet cells each), so any difference is a findable
numerics difference, not a resolution floor; (2) the "wall-trapped WBC spike /
one-cell mask offset" was a STAGGERING/indexing ARTIFACT in the transect probe
(compared NEMO U-face vs legoESM T-centre; NEMO umask(i)=T(i)·T(i+1) vs
legoESM's face convention — same physical faces). Clean staggering-correct
T-centre comparison (`gap_audit/_plot_two_way_final.py`): the deficit is
~uniform west(0.65x)/interior(0.62x) — NOT wall-trapping — and by LATITUDE
the SUBTROPICAL gyre MATCHES NEMO (rows j7/10/13 ratio 1.04/1.11/1.05) while
the SUBPOLAR north is ~half (j16 0.57, j19 0.54). Root: the yr-5 SUBTROPICAL
column (j6-10) is VERTICALLY HOMOGENISED — uniform T=15.24 from surface to
428 m, surf-to-200 m contrast **0.00** vs NEMO's **0.76** (NEMO keeps a proper
thermocline 15.1->13.3 over the top 428 m); the missing surface heat is buried
(+1.9 C too warm at 428 m). No thermocline -> no baroclinic thermal-wind shear
-> weak surface gyre (worst in the north where stratification is weakest).
ISOLATION EXPERIMENTS (2026-07-16, 5-yr runs + single-column avt certificate):
- **TIME EVOLUTION reframes it**: NEMO does NOT preserve the IC thermocline
  either — at yr1 NEMO's subtropical contrast is only 0.11 (IC was ~5.1), then
  it BUILDS UP: 0.11→0.44→0.82→1.10→1.34 over yrs 1-5 (Ekman-pumping spin-up of
  the subtropical thermocline). legoESM collapses to 0.00 by day 182 and stays
  flat. So the failure is INABILITY TO ACCUMULATE stratification over years,
  not a one-time over-mix.
- **Iso diffusion is NOT the eroder**: kappa_Redi=0 (GYRE_KREDI=0) 5-yr run
  leaves the thermocline just as eroded (contrast 0.00). [The S_max=0.005 vs
  NEMO 0.01 mismatch found here IS real and is now FIXED (redi_S_max=0.01) as a
  matched-parameter fix, but it is thermocline-neutral.]
- **EVD contributes only partially**: GYRE_CONV=none 5-yr → contrast 0.079
  (vs baseline 0.00, NEMO 1.34). So EVD over-fires vs NEMO (a trigger
  difference) but is not the dominant eroder.
- **Interior background avt MATCHES NEMO EXACTLY**: single-column certificate
  (NEMO EXP_15D day-5 en/T/S seeded into legoESM's closure) gives subtropical
  interior K_H = 1.20e-5 at every level below the ML == NEMO avt_k. And
  background 1.2e-5 mixes only ~14 m in 6 months — so the 428 m homogenisation
  is NOT slow interior diffusion.
- **MECHANISM CAUGHT (2026-07-16, harness now dumps `state.tke`; avt
  reconstructed per-snapshot via the validated `tke_set_diffusivities`,
  `gap_audit/_diag_avt_seasonal.py`): the subtropical column SELF-LOCKS in a
  mixed state.** Summer subtropical avt NEVER drops to background — it stays
  0.03-0.09 (3000-6000x the 1.2e-5 floor) down to ~380 m ALL YEAR. Depth
  profile at day152 (summer): N²≈+4e-10 (essentially NEUTRAL, T uniform
  17.36), TKE energy e=1.5-2.9e-3 (≈1000x the ~1e-6 floor) SUSTAINED to
  234 m, mixing length l_k grows to ~200 m (buoyancy length √(2e)/N blows up
  as N→0, capped only by distance-to-surface). Feedback: N²≈0 → l~200 m →
  avt large → column stays neutral → l~200 m. The first winter (d30) is still
  STRATIFIED (N²=+6.3e-5, e=0 at depth, thermocline intact) — the lock closes
  in spring (d30 contrast 0.22 → d61 0.00). Each summer surface heating tries
  to restratify the top but the sustained mixing homogenises it as fast as it
  is applied → mixing wins → thermocline never rebuilds.
- **=> It is a BISTABILITY / REBUILD-FAILURE, not a single eroder term.**
  NEMO SUMMER dump obtained (EXP_LONG has 30-day restarts with instantaneous
  en/avt_k; day1260/1620 are spun-up summers): NEMO's subtropical TKE decays
  to en=1.0e-6 (the rn_emin FLOOR) below ~55 m, avt→background 1.2e-5, thermo-
  cline contrast 3.17. legoESM's summer en stays 8e-4..1.9e-3 (~1000x NEMO) to
  380 m. First-year timeline (both homogenise by winter day90, contrast 0):
  at spring day150 NEMO's deep en drops 220x to the floor and the thermocline
  REBUILDS (0→0.40→1.07 by day180); legoESM's en stays 1.4e-3 and it never
  rebuilds. **DECISIVE IC-SWAP TEST** (legoESM from NEMO's spun-up stratified
  summer T,S + matched summer forcing phase, GYRE_IC_NEMO+GYRE_TOFFSET_DAY):
  legoESM HOLDS the thermocline through summer (3.2→1.9 at day90, NOT
  catastrophic), winter homogenises it (like NEMO), then the NEXT summer FAILS
  to rebuild (0.04→0.02). So legoESM's mixing is NOT catastrophically too
  strong — the specific failure is the MIXED→STRATIFIED transition (rebuild).
- **RULED OUT as the lock lever (all 5-yr / 2-yr runs, subtropical contrast
  stays ~0)**: iso diffusion (GYRE_KREDI=0), Langmuir (GYRE_LC=0), TKE
  dissipation ×3 (GYRE_CEPS=2.0), shear-production mode (GYRE_SHEAR=pre_solve,
  the NEMO avm·S² form); EVD (GYRE_CONV=none) only partial (0.079). Interior
  background avt MATCHES NEMO (1.2e-5). TKE floors MATCH NEMO
  (tke_background=1e-6=rn_emin, tke_surface_min=1e-4=rn_emin0). c_k/c_eps match
  rn_ediff/rn_ediss. So NO TKE-scheme knob enables the rebuild.
- **MECHANISM COMPLETE — the SHALLOW SUMMER ML fails to form.** Summer rebuild
  from a mixed state, top-60 m T profile: NEMO concentrates the summer heat in
  the top ~48 m (61 m LAGS — day180 surface 0.42 C warmer than 61 m) → a
  shallow ML forms → seeds restratification → thermocline rebuilds. legoESM
  warms the whole top 60 m UNIFORMLY (5 m only 0.007 C warmer than 61 m) → no
  shallow ML → no seed → stays mixed. Root: the residual mixed-state TKE
  (~1e-3) + the surface Dirichlet TKE source (en(1)=rn_ebb·|τ|/ρ0≈5e-3)
  diffuse down through the NEUTRAL column and mix the top-~23 m solar heat down
  over ~60 m before it can stratify. In NEMO's stratified column that downward
  TKE diffusion is blocked by the small thermocline mixing length; legoESM
  never gets stratified enough to block it → self-locking race legoESM loses.
- **THE UNMATCHED NUMERIC FOUND + FIXED (alpha_tke 30->1, commit on branch):**
  candidate (a) was it. NEMO diffuses TKE with avm x1 (zdftke.F90:130
  "d(avm d(en)/dz)/dz"; tridiag coeff = face-avg avm, zfact1=-0.5*rn_Dt). The
  ported Veros/CATKE scheme defaulted alpha_tke=30 -> TKE diffused 30x too
  fast, flooding the surface Dirichlet TKE source down the near-neutral column,
  sustaining deep e ~1000x rn_emin and blocking the shallow summer ML. This is
  the "identical numerics must land on the same branch" difference the user
  flagged: the numerics were NOT identical (30x vs 1x), and the bistability
  amplified it into opposite branches. Fix: alpha_tke=1 in _nemo_tke_config.
  EFFECT: deep summer e drops 1-2 orders (54m 2.5e-3->4.8e-5) toward NEMO's
  floor; the seasonal thermocline REBUILD unlocks (phased-from-NEMO: winter
  0.19 not 0.04, next summer rebuilds 0.5-1.0 vs stuck 0.02); baroclinicity
  surf/deep 3.87->4.93 (NEMO 8.23).
- **SECOND UNMATCHED NUMERIC FOUND + FIXED — EVD triggered on IN-SITU N²,
  NEMO uses ADIABATIC (rn2).** alpha_tke=1 alone did NOT fix the cold start
  (all summers ~0); a second factor: the card's enhanced_diffusion (EVD) used
  n2_mode="insitu", but NEMO's ln_zdfevd triggers on rn2 = the ADIABATIC
  Brunt-Vaisala frequency. In-situ N² carries the compressibility term and goes
  spuriously NEGATIVE in a statically-STABLE column, so legoESM's EVD fired in
  the subtropical SPRING where NEMO's (rn2>0) does not — re-mixing the shoaling
  ML every step and blocking the rebuild. (Isolation: MLD stays 263-334m all
  year in legoESM vs NEMO shoaling 263->61m in spring; EVD-off unlocks the
  rebuild; the -1e-12 threshold does NOT (marginal N² is more negative), but
  n2_mode="adiabatic" DOES — identical to EVD-off but NEMO-faithful, EVD stays
  on.) Fix: EnhancedDiffusionConfig.n2_mode="adiabatic" on the card (+ a new
  n2_threshold field, default 0.0, for NEMO's -1e-12; inert here, kept for
  fidelity).
- **NEXT-TARGET TRACE (2026-07-16 cont.): the residual is the SPRING
  VERTICAL-ADVECTION (Ekman pumping) chain, not any column-physics numeric.**
  Evidence chain: (1) day-91 states MATCH everywhere (max 0.09 C) — the first
  winter is now faithful; (2) the divergence develops days 90-180 in the
  lat 30-36 band: surface stays ~0.4-0.7 cold with heat parked at 111-209 m;
  (3) the closure is certified EXACT in this regime too (NEMO day-150
  fossil-layer column: our K_H == avt_k at every level incl. the floor);
  (4) twin-from-NEMO-day-90 (GYRE_IC_NEMO, matched phase): NEMO cools 187 m
  by 0.4 C in 30 days = w*dT/dz with w ~ 1 m/day (spring Ekman upwelling
  north of the wind max sharpening the seasonal thermocline from below);
  the rest-started twin does not — the term lives in w, not in mixing.
  => the loop is DYNAMICAL: 0.71x baroclinic gyre -> 0.71x w pattern ->
  weaker isotherm doming/sharpening -> 0.81x density structure -> 0.71x gyre.
  All column physics is certified; the remaining comparison target is the
  W FIELD (legoESM state.w vs NEMO vovecrtz 2920h means) and the wind-curl
  -> w_Ek realization. Harness: add "w" to the snapshot dict to enable.
- **W-FIELD COMPARISON (the decisive dynamical diagnostic, 2026-07-16 cont.):
  RESOLVED-SCALE w MATCHES NEMO EXACTLY; the residual is GRID-SCALE w NOISE.**
  Time-mean w vs NEMO vovecrtz (2920h means; lego snapshots time-averaged over
  the same windows): raw rms ratio 1.3-4.5x with corr 0.57-0.85 — but after a
  2x2 box smooth the rms ratio is **1.00** (9.64e-7 vs 9.64e-7 at 101 m,
  rec1) with corr 0.83. => the Ekman-pumping/wind-curl chain is EXACT; lego
  carries EXTRA 2Δx w noise (grid-scale variance fraction 0.32-0.35 vs NEMO
  0.20). The noise floor is DEPTH-UNIFORM (~2.7e-6 at 54/101/187 m), so the
  spurious divergence is concentrated in the TOP (Ekman) cells — noisy
  surface-layer velocities, not barotropic-mode divergence (which would grow
  with depth). MECHANISM: grid-scale w stirs the seasonal thermocline through
  the FCT limiter = spurious diapycnal mixing = the summer build-rate deficit
  (0.5 vs 1.07) -> weaker doming (0.81x) -> weaker gyre (0.71x).
  ATTRIBUTION (instantaneous high-pass structure, day 180, 101 m): the noise
  is BROADBAND grid-scale — NOT a pure Coriolis checkerboard (projections on
  (-1)^i / (-1)^j / (-1)^(i+j) only 0.12/0.14/0.03); geographically strongest
  in the NORTH (subpolar: rms 3.0e-6 vs 1.7e-6 mid/south); grows through the
  upper column (15 m 5.9e-7 -> 101 m 2.2e-6) => noisy DIVERGENCE distributed
  over the upper ~100 m baroclinic velocities, subpolar-concentrated. Revised
  suspects: (a) upper-ocean inertia-gravity/adjustment noise excited by the
  winter convection episodes in the north (insufficiently damped vs NEMO —
  NEMO's leapfrog+Asselin FILTERS what WS-RK3 does not: NEMO GYRE is key_RK3
  though — check what damps NEMO's upper-ocean noise, e.g. ln_dynvor_ene
  enstrophy properties or the FCT2 tracer-w coupling); (b) the wind-stress
  top-cell body force vs NEMO's implicit dynzdf surface BC (deposition
  roughness); (c) spatially noisy avm -> noisy Ekman spiral. NEXT: snapshot
  MAPS of hi-pass w lego-vs-NEMO (needs a NEMO w snapshot — add w to the
  trend-dump or use a 1-day mean), + NOWIND ablation to split (b)/(c) from
  (a). Harness now dumps state.w (snaps['w']).
- **THE RETENTION INVARIANT: ROOT CAUSE FOUND — nemo_iso_lap SLOPE SIGN FLIP
  (fixed, certified; a wall hot spot remains the open blocker).** Winter
  ttrd budget (new EXP_WINTER, 12-step restart from NEMO's Jan-yr5 state with
  trends): NEMO's ttrd_ldf WARMS the 200-430 m band at +5.2e-8 K/s all winter
  (x90 d = +0.4 K = exactly NEMO's retention) — the ISO DIFFUSION is the
  permanent-thermocline builder. legoESM's iso on the SAME state: -1.0e-7
  (OPPOSITE SIGN). Slope-vs-slope: corr(S_lego, wslpi_stg) = -0.995..-0.997,
  same magnitude — a pure convention flip (producer S = +dx(rho)/|drho_dz|;
  NEMO ldfslp slp = zau/(zbu<0) = -dx(rho)/|drho_dz|); the operator was
  certified consuming NEMO's convention, so the subduction ran BACKWARD in
  every nemo_iso_lap run of the campaign. FIX: negate at the dispatch
  (committed, sign-gate test); verified on the winter state (+6.8e-8, mode-b
  amplitude).
- **5-yr WITH the sign fix: the invariant BREAKS — winter retention appears
  (1.4-5.9 vs the 10-yr-constant 0.00) and baroclinic u' UNPINS (0.71 ->
  1.52) — but the run develops a SOUTHERN-WALL HOT SPOT** (j=1-3, i=17-23,
  column-mixed patch heating to 39-45 C, quasi-steady against the Haney
  restoring; T-max climbs from year 1). A local upgradient pump at the wall:
  the negated slopes are certified in the interior (corr -0.995 => ~0.5%
  mismatch concentrated at boundaries/ML edges), and the wall traps the
  wrong-signed residual flux with no advective escape. OPEN BLOCKER for the
  card. NEXT (code-first): trace NEMO ldfslp's WALL-ROW handling vs ours —
  which rows get slopes at all (NEMO computes DO_2D(1,1,1,1) and Shapiro
  writes rows 2..jpjm1 only; boundary uslp may remain 0 where lego's ring
  rows carry active slopes), the coastal zcofw factor detail, and
  traldf_iso's wall-face flux masking. The retention physics is now PROVEN
  reachable — the remaining work is the boundary detail.
- **HOT-SPOT BISECT + FIRST CLEAN MEASUREMENT (2026-07-17): the permanent
  thermocline BUILDS.** Bisect: nemo_cap+signfix -> hot (39-45 C);
  dm95+signfix -> HEALTHY (21 C); ML ramp innocent => the mode-b operator's
  ~1.35x amplitude exceeds the stability margin NEMO's e3/7e3 bound encodes
  for full-kappa flux at capped slopes. Card -> dm95 (nemo_cap stays wired +
  tested as the option; becomes default when the native four-position slopes
  land). Wall verified CLEAN at the matched state (slopes corr +0.99/+1.00
  post-fix; tendencies == ttrd_ldf at j=1-3); NEMO wall delta noted:
  wet-face-COUNT gradient normalization (zci=MAX(sum umask,eps)) vs lego's
  always-/2. CLEAN 5-yr (signfix+dm95): winter retention
  -0.07/0.01/0.31/0.44/**0.69** over 5 winters — the FIRST healthy build ever
  (NEMO 0.11..1.34; ~half rate = the dm95 under-transport cost, still
  accelerating at yr 5); barotropic 1.03, mid-depth 1.06, SST max 21.3,
  baroclinic u' 0.67 (lags the building retention; the contaminated cap run
  reached u'=1.52 => the circulation responds once subduction runs at full
  strength). TOP QUEUED FEATURE: native four-position slopes (uslp/vslp at
  T-levels + wslpi/wslpj at w-levels) — one feature fixing amplitude
  (1.35->1.0), cap stability, and plausibly the remaining retention factor 2.
- **IMPLICIT WIND-STRESS DEPOSITION SHIPPED (surface_stress_implicit; review
  SHIP)** — the wind-path w-noise source fixed: stage-10b's explicit ~0.1 m/s
  per-step top-cell kick replaced by NEMO's dynzdf arrangement (stress in the
  implicit solve's top-cell RHS + tau/(rho0 H) in F_slow = stp2d + the
  stprk3_stg:440 mean imposition, now LOAD-BEARING). Winter w noise HALVED
  (rms 2.71e-6 -> 1.03e-6, NEMO 6.0e-7; hi-pass 8.78e-7 -> 4.36e-7, NEMO
  1.12e-7 — residual == the NOWIND buoyancy floor). Momentum input identical
  to the explicit path (2e-7; no double count — algebra review-verified).
- **10-YR VERDICT: noise fixed, CORE INVARIANT SURVIVES.** Seasonal
  thermocline now OVERSHOOTS and keeps climbing (summers 3.98 -> 4.86 by
  yr 9.5; NEMO equilibrates at 3.17); mid-depth rms 1.19x, barotropic 0.97x;
  BUT winter retention still 0.00-0.02 every winter for 10 years (NEMO 0.40)
  and baroclinic u' still exactly 0.71 at yr 10. THE PARADOX localizes the
  next target: lego builds MORE summer stratification than NEMO yet retains
  NONE through winter => its summer heat is trapped TOO SHALLOW (above the
  ~250 m winter mixing reach); NEMO moves summer heat into the 200-400 m
  band (permanent thermocline) where winter cannot reach. Resolved-scale w
  is EXACT (ratio 1.00), so the question is the VERTICAL DISTRIBUTION of
  the seasonal heat gain: compare lego-vs-NEMO summer heat-content profiles
  in the 100-400 m band (where does the summer heat END UP), then trace the
  responsible term (vertical advection realization of the doming vs the
  100-200 m mixing that in NEMO deepens the seasonal thermocline base).
- **zdftke CODE-TRACE COMPLETE (same session): dissipation_discretization=
  "nemo_1p5_split" implemented (NEMO zfact2/zfact3, on card, tested);
  TKE-first order / positivity=floor / pre_solve shear / -1e-12 threshold /
  tracer RK3 ALL verified trajectory-inert on the fixed config.** Remaining
  traced minors: bottom TKE Dirichlet (0.001875*rCdU*|u_bot|), EVD
  min(rn2,rn2b) two-level.
- **RESOLVED (both fixes on the card): the subtropical thermocline now BUILDS
  to NEMO's level.** 5-yr, card native (alpha_tke=1 + EVD adiabatic): summer
  contrast 0.54 -> 1.79 -> 2.41 -> 2.96 -> **3.46 (NEMO summer ~3.17)**; winter
  matches NEMO (~0 to 0.4). Baroclinicity surf/deep **3.87 -> 6.21** (NEMO
  8.23). Self-lock broken; legoESM is on NEMO's stratified branch. The two
  unmatched numerics were: (1) TKE diffused 30x too fast (alpha_tke 30->1);
  (2) EVD on in-situ vs adiabatic N². Both NEMO-faithful. Remaining gap to full
  NEMO (surf/deep 6.21 vs 8.23) is now a smaller residual, no longer the
  thermocline lock. NOT a resolution floor — the user's "identical numerics ->
  same branch" logic located two real unmatched terms.

**Minors (low priority):** TKE buoyancy N² adiabatic → in-situ; residual
winter-ML depth (111 vs 91 m at day 15) — candidates: EVD reach interplay,
tra_sbc per-stage placement; NEMO per-stage barotropic-mean imposition
(item under D).

## 7. Ruled out — do NOT re-test
- Vertical momentum advection (upwind_perturbation vs centred): no effect on the gap.
- Redi S_max (0.005 vs 0.01): no effect.
- A_v background as a *flow* lever (it was a real SST bug, not the flow).
- TKE `bg_diff_*` depth profile: inactive (`enable_kappaH_profile=False`).
- Lateral viscosity / bottom drag / advection / EOS / thermal forcing: matched.

## 8. How to resume (reproducibility)
- **NEMO oracle (5-yr)**: `~/oracle-builds/nemo5/nemo_5.0.2/cfgs/GYRE_BARE/EXP_LONG`
  (nn_itend=10950, nn_stock=180). Rebuild: env `nemo-build`, `mpirun -np 1 ./nemo`.
  KEEP `EXP00` (the certified 12-step trend-dump oracle) untouched.
- **legoESM runs**: `~/oracle-builds/nemo5/gap_audit/run_native_gyre.py N out.npz`
  with `CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1`. Env toggles (throwaway test
  harness, NOT the recipe): `GYRE_WIND` (now redundant — recipe has wind),
  `GYRE_CONV/GYRE_KCONV/GYRE_NUCONV`, `GYRE_VADV`, `GYRE_SMAX`, `GYRE_AV/GYRE_KV`,
  `GYRE_BDIFF`, `GYRE_LINSSH` (broken — z-star has no linssh field), `GYRE_NSNAP`.
- **Comparison scripts** (`gap_audit/`): `_diag_equilibration_compare.py` (RMS
  trajectory), `_diag_velocity_depth.py` (surf/deep — the key metric),
  `_diag_density_gradient.py`, `_diag_driving_vs_dissipation.py`.
- **Rebuild the wiring diagrams**: two general-purpose agents (NEMO WORK/*.F90 +
  namelists; legoESM build_nemo_gyre_recipe + model.step) — see this session.

## 9. Open questions / risks
- Which of A/B/C actually seeds the abyssal drift is not yet isolated — the fixes
  are coupled, so validate each with `_diag_velocity_depth.py` (surf/deep) and
  `_diag_density_gradient.py`, not just the scalar RMS ratio.
- The z-star→linssh change touches a load-bearing coordinate — design carefully,
  add conservation/coordinate unit tests, run the mandatory adversarial review.
- Fidelity ceiling: a JAX re-implementation won't be bit-identical; the target is
  a close laminar trajectory (surf/deep and RMS both near NEMO), not roundoff.
</content>
