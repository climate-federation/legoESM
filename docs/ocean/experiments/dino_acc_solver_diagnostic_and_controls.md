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

Diagnostics: `scripts/tmp/_diag_{psi_unittest,geoadj_balanced,munk_solver,
r1_warmstart_check,dino_acc_controls}.py`.
