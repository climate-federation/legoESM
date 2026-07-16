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

**Minors (low priority):** rho_0 1025 → 1026 (done); TKE buoyancy N²
adiabatic → in-situ; residual surface rms 0.58x + residual winter-ML depth
(111 vs 91 m at day 15) — next candidates: EVD reach interplay, tra_sbc
per-stage placement.

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
