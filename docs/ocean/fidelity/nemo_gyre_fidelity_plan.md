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

**C. PGF / hydrostatic pressure at depth.**
NEMO `ln_hpg_zco` (z-coord, e3w-weighted integral of the 2-level-averaged density
anomaly); legoESM `adcroft` (≈0.966 at t=0, 3.4% amplitude deficit). A small
persistent deep-PGF error seeds a deep flow. Verify the deep hpg vs NEMO;
consider a zco-faithful option. Effort: medium.

**D. Barotropic filter — DONE 2026-07-16 (483b6477a).** nemo_ab3am4
implemented (AB3 extrapolation + AM4 backward ssh interpolation alpha=0.07 +
final-value output + uniform transports; review: numerics EXACT, direct tests
added). The cosine window-averaging was retarding the gyre by ~dt/2 per step.
RMS ratio 0.50 → **0.71** (yr-1 0.93), surface rms +55%, surf/deep 11.6
(brackets NEMO 8.2). Remaining minors: per-window ramp difference (documented),
substeps 120 vs NEMO 50. Original description:
NEMO uses temporal dissipation (AB3-AM4, rn_bt_alpha=0.07), NO spatial diffusion;
legoESM uses cosine time-averaging + `barotropic_diffusion_alpha=0.01` (spatial,
×120 substeps) which damps the WBC η-gradient. Also substeps 120 vs NEMO's 50.
Effort: medium (implement Demange in the barotropic solver).

**E. RK3 variant: Shu-Osher SSP → Wicker-Skamarock** (+ asymmetric per-stage RHS:
NEMO applies LDF at stages 1&3 only, ZDF at stage 3 only). Effort: medium.

**Minors (low priority):** rho_0 1025 → 1026; TKE buoyancy N² adiabatic → in-situ;
nn_mxl 2 → 3 (previously deconfounded — likely inert).

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
