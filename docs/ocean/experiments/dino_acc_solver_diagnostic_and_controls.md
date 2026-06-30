# DINO ACC: barotropic solver/diagnostic validation + the ACC controls

Follow-up to `dino_rigid_lid_acc_analysis.md`.  User question (2026-06-29): the
barotropic streamfunction "looks weird" — is it the **solver** or the
**diagnostic**?  Plan: (1) a geostrophic-adjustment test (explicit vs production
barotropic solver), then (2) the two idealized-channel ACC controls — **bottom
friction** and **bathymetry** — against a Kamm-et-al. oracle.

## P1 — solver + diagnostic are BOTH correct

| test | result |
| --- | --- |
| **Diagnostic** unit test: known transport ψ → u=−∂ψ/∂y/H → `barotropic_streamfunction` → recover | corr **0.998**, 1.7 % RMS (discretization) — CORRECT. The `dy = grid.dy*0.5` is right (`grid.dy` is the 2-cell span, `latlon.py:74`). |
| **Solver**, steady: Munk wind-driven gyre (analytic Stommel-Munk WBC ~20 Sv) | `implicit_cn` ≡ `explicit_substep` **bit-identical** (corr 1.0, RMS 0.0, WBC 16.7 Sv). |
| Solver, transient: constant-density DINO + SSH bump (geostrophic adjustment) | `implicit_cn` vs `explicit` diverge (corr −0.29) — **but this is the gravity-wave transient**, not a steady-balance error (implicit-CN θ=0.55 damps the sloshing waves the radiating bump excites; the balanced residual was weak). |

**Conclusion:** neither the solver nor the diagnostic is the "weird streamfunction".
On a *steady* barotropic balance — which the ACC is — the solvers agree exactly.
Use a steady test (not a wave-radiating transient) to compare barotropic solvers.

## Oracle anchoring

Kamm Zenodo 15016824 provides EXP_R1 (1°, ACC 206 Sv) and EXP_R16 (1/16°) — **no
1/4°**.  NEMO reaches 206 Sv at **1°**, so 206 is achievable at our resolution
(not an eddy-resolution limit); the gap is a model/configuration issue.  The R1
restart grid (199×52×36) ≈ ours (198×50×36, top levels identical).  Longitude
convention differs: NEMO 0–50 °E, ours −50–0 (shift NEMO lon −50).  The regridded
NEMO R1 velocity reads **ACC = 209 Sv** in our diagnostic (≈ 206) — a third
independent confirmation that the diagnostic is correct.

A **warm-start** from the R1 state (full u,v,η; or T,S at rest; smoothed; dt down
to 300 s) NaNs within days every time — the regridded NEMO state is not in our
model's discrete hydrostatic/geostrophic balance (sharp fronts + level-mismatch
inversions).  Model-from-model initialization needs a dedicated balancing step;
shelved.  The clean equilibrium comparison is therefore still open; the cold-start
control sweep below answers the *controls* question regardless.

## P2 — the ACC controls: bathymetry, not friction

Cold-start DINO, validated `implicit_cn`, ACC at day 120 (baseline ~41 Sv; the
absolute value is thermocline-spin-up-confounded — NEMO's 206 is a 3000-yr
equilibrium — but the *sensitivities* are the physics):

| control | ACC (Sv) |
| --- | --- |
| bottom drag C_d = 2.5e-4 / 5e-4 / 1e-3 / 2e-3 | 41 / 40 / 41 / 41 — **inert** |
| sill H_sill = 2500 / 3000 / 3500 / 4000 m | 41 / 46 / 176 / 758 (runaway) |

**Bottom friction is irrelevant** (40–41 Sv across an 8× drag range).  **The sill
bathymetry is the dominant control** — there is a sharp threshold (~3000→3500 m:
46→176 Sv), and removing the sill entirely (4000 m) lets the ACC run away (758 Sv
and climbing).  So the sill provides the **topographic form stress** that *limits*
the ACC — exactly the expected idealized-channel physics, and the same
runaway-without-a-limiter behaviour seen for the rigid lid.

The 41-vs-206 gap at our nominal sill is **thermocline spin-up modulated by the
sill depth**: the geostrophic shear over the sill is immature at 120 d, and the
ACC is extremely sensitive to how much of the water column the sill blocks.  Our
discretized sill crest is ~2805 m (vs nominal 2500 — a Gaussian-ring sampling
offset).  A deeper sill (3500 m) reaches ~176 Sv (near 206) much faster.

## Where this leaves the ACC

- Solver + diagnostic: validated, not the problem.
- The ACC is **sill-form-stress-controlled**; bottom friction is not a lever.
- Matching NEMO's 206 Sv at 1° requires the right sill form stress + a mature
  thermocline; the open piece is a balanced warm-start (or a long spin-up) to
  compare equilibria cleanly, plus checking the sill discretization vs the paper.

## Levers L1–L3 — RESOLVED: the gap was thermocline spin-up

Following the controls result, three levers were worked in order:

**L2 (sill geometry) — correct, not the cause.** Our discretized bathymetry equals
the analytic eq A5 at every point; the channel-band sill crest is 2309 m (≈ the
nominal 2500). The sill is faithfully represented, so the low ACC is not a sill
discretization error.

**L1 (balanced warm-start, clean equilibrium comparison) — blocked.** A warm-start
of our model from the NEMO R1 equilibrium NaNs every way tried (full u,v,η; T,S at
rest; smoothed; dt down to 300 s; `apply_balanced_init` geostrophic/thermal-wind
balance; balanced + smoothed). The thermal-wind ACC our balancer reconstructs from
NEMO's density is clip-artifact-dominated (480 → 909 Sv, |u| clipping at 2.5 m/s):
the ACC's steep isopycnals over the deep column + the 1500 m level-of-no-motion +
the regridded fronts produce unphysically large velocities. Model-from-model
initialisation of a deep-reaching ACC is a genuinely hard balancing problem;
shelved. (It did confirm the ACC is **thermal-wind / density-controlled**: the
immature cold-start density gives a low ACC, a mature density a high one.)

**L3 (thermocline spin-up) — the ACC OVERSHOOTS, it does not settle at 206.** A
6-year cold-start (nominal sill, `implicit_cn`):

| year | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 | 5.5 | 6.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACC [Sv] | 46 | 41 | 69 | 81 | 126 | 243 | 412 | 576 | 713 | 753 | 752 | 691 |

The ACC is flat ~45 Sv through year 1 (immature thermocline — the "too low 44 Sv"
WAS spin-up), then accelerates through 206 (~yr 2.7) and **keeps climbing to ~753 Sv
by year 5** (3.7× NEMO) before slowly declining. So **our DINO equilibrium ACC is
far too HIGH** (~700 Sv), not too low.

**Cause — the thermocline over-deepens (not eddy saturation).** A GM sweep
(`visbeck_kappa_max` 2000 → 6000 → 15000, α 0.015 → 0.06) has **no effect** — all
overshoot to ~750 Sv. So the overshoot is GM-insensitive (the same GM-insensitivity
as the rigid-lid barotropic runaway). With too-weak eddy flattening unable to be
the lever, the thermocline keeps deepening and the thermal-wind ACC grows without
saturation. NEMO's 206 Sv is set by a shallower equilibrium thermocline — a
**vertical-mixing / buoyancy-forcing / EOS** difference vs NEMO, the real
model-vs-NEMO discrepancy.

**Warm-start (clean equilibrium comparison) is blocked** (6+ variants NaN,
including convective-adjustment of the 6.3 % regrid density inversions — the
in-place adjustment itself was unstable). Model-from-model ACC initialisation is a
genuinely hard balancing problem.

**Conclusion (corrected):** the barotropic solver and diagnostic are correct;
bottom friction is irrelevant; the sill geometry is correct. The "50 Sv too low"
was a 1-year spin-up artifact — but the model's *equilibrium* ACC is too HIGH
(~700 Sv, thermocline over-deepening, GM-insensitive). Matching NEMO's 206 Sv is a
**vertical-mixing / buoyancy-forcing / EOS** tuning problem (the thermocline depth),
not a solver/friction/sill/GM/eddy issue.

## NEMO side-by-side (the oracle, running)

Built NEMO 5.0.1 + the DINO config (`vopikamm/DINO`, branch `wip/DINO_5.0.1`) and
ran the actual oracle cold-start beside ours.  Build notes: register `DINO OCE` in
`tests/demo_cfgs.txt` (else OCE isn't linked); `makenemo -n DINO_R1 -a DINO -m
ORCA1_GCC` inside the morays Singularity container.  Run notes: singularity module
only on `short`; MPI needs `--oversubscribe`; **attached XIOS** (`using_server=false`)
— the detached XIOS server deadlocks (0.13 → 20 steps/s).  ACC extracted from `uoce`
with the same barotropic-streamfunction diagnostic.

**Cold-start ACC, year by year:**

| year | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| **NEMO** | 62 | 66 | 74 | 90 | 101 | 110 |
| **ours** | 41 | 81 | 243 | 576 | 753 | 691 |

- **Laminar years 1–2 AGREE** (~50–80 Sv) — the dynamics match early.
- **NEMO marches slowly + stably** toward its 206 Sv equilibrium (a 50-yr spin-up).
- **Ours over-spins the thermocline ~10× too fast** → overshoots to ~750 Sv.

**Ablations (N3) — clears GM:**
- Our `vmix`: kpp overshoots; **tke and constant both NaN** (unstable in our 1°
  DINO), so the alternatives can't be run our side.
- Our GM is **active** (wired + applied) but **insensitive** (κ_GM ×7.5 no effect).
- **NEMO with GM OFF** (`ln_ldfeiv=.false.`, verified in `ocean.output`) gives an
  **identical** early spin-up (62,66,74,90) to GM-on.  So **GM has no effect on the
  laminar spin-up in either model** — it only saturates the ACC near equilibrium
  (isopycnal slopes are small early).

**So the discrepancy is the diabatic thermocline spin-up RATE** — set by vertical
mixing (our KPP vs NEMO TKE), the EOS (our Wright vs NEMO's cabbeling/thermobaric
S-EOS), convection, and the surface buoyancy forcing — **not** the barotropic
solver, diagnostic, sill, friction, or GM (all cleared).  The next lever is the
thermocline physics (a KPP/TKE-mixing or EOS ablation; our TKE NaNs and we lack
NEMO's exact S-EOS, so both need work first).

NEMO build/run: `scripts/cluster/omip_nemo/{_build_dino,_run_dino_nemo}.sbatch`,
extract `scripts/tmp/_diag_nemo_acc_extract.py`.
Diagnostics: `scripts/tmp/_diag_{psi_unittest,geoadj_balanced,munk_solver,
r1_warmstart_check,r1_warmstart_balanced,dino_sill_geom,dino_acc_controls,
dino_spinup_multiyear,dino_vmix_ablation}.py`.
