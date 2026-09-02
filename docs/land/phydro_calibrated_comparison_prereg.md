# Pre-registration — calibrated-vs-calibrated: two-leaf +/- P-hydro at EC sites

Written BEFORE the tuner exists (research branch, PR #1698). This is the test
the stage-2 ladder could not make: the empirical soil-moisture stress
(beta_theta) is an implicitly tuned baseline, so comparing it against DEFAULT
P-hydro confounds scheme fidelity with parameterisation. Here each arm is
tuned to its own best, with 5-seed multi-start (the turbulence/convection
protocol), and the comparison happens on held-out years only.

## Arms (both carry the P-model bundle: capacity+g1 = p_model, medlyn)

| | arm A (beta_theta) | arm B (phydro) |
|---|---|---|
| stress | empirical `(theta-wp)/(fc-wp)` root-zone beta | Joshi-2022 hydraulic profit optimum |
| tuned, shared mechanism | `root_depth`, `theta_wp`, `theta_fc`, `soil_evap_litter_resistance_s_m` | same four |
| tuned, scheme-specific | — | `gplant_mol`, `minlwp_mpa`, `gamma_cost`, `root_biomass_gm2` |
| total tuned | 4 | 8 |

**Capacity asymmetry is deliberate and stated**: the four extra parameters ARE
P-hydro's real degrees of freedom; hiding them would test a crippled scheme.
Both arms tune the same shared soil/root set (GLM: arm A must not reshape a
soil that arm B inherits untuned — root_depth and the soil-evap exponent are
shared mechanism too, since the P-hydro supply consumes the same root profile
and the soil-evap beta stays soil-controlled in both arms). Overfitting from
B's extra capacity is arbitrated by the held-out years, and the verdict rule
carries a margin.

## Design-review amendments folded in (codex, before implementation)

* **The originally chosen `soil_evap_resistance_exp` is INERT on this path**
  (VERIFIED: `soil_evap_series_resistance=True` by default, so the exponent
  branch is dead code at `multilayer_land.py:682`) - the live soil-evaporation
  knob on the series path is `soil_evap_litter_resistance_s_m` (tier 2,
  0-400 s/m), which replaces it. The no-inert gate would have aborted; codex
  caught it pre-build.
* **The prognostic NaN-revert is not AD-safe** (`jnp.where` can leak NaN
  cotangents from the rejected branch): the tuner forward runs WITHOUT the
  revert; any non-finite loss or gradient marks that SEED FAILED (recorded in
  the results), never silently skipped and never masked into the objective.
* **`theta_fc` is parameterised as `theta_wp + softplus gap`** so independent
  sigmoids cannot produce the unphysical `fc < wp` inversion.
* **Framing**: this is a calibrated PREDICTIVE comparison of the two stress
  architectures (theta thresholds do not alter the Richards retention curve
  but do alter shared soil-plant availability), not an isolated
  stress-scheme-only test.
* **Year eligibility is frozen from observation/forcing masks alone** (joint
  GPP+LE validity, never model output, never per arm/seed), in one committed
  table before either arm runs; model NaNs on eval are penalised, not masked.

## Data protocol (per site; 5 dry sites: US-SRM, US-Whs, US-Ton, US-Var, FR-Pue)

* **Train**: the TWO calendar years with the highest `score_valid` coverage
  (ranked once, arm-independent, from the driver alone — recorded in the
  results file; ties broken by earlier year).
* **Test**: ALL remaining calendar years with >= 20% valid coverage
  (evaluation is forward-only, so extra years are nearly free — GLM's
  highest-leverage fix). No test-year quantity is ever read during tuning or
  seed selection.
* Gap-free drivers, prognostic mode, per-site texture (`--soil auto`
  equivalents), same initial-condition rule as the harness.

## Tuning

* Differentiable loss through the prognostic scan (`jax.checkpoint` on the
  step): joint over the 5 sites, mean of `0.5*nMSE(GPP) + 0.5*nMSE(LE)`,
  nMSE = MSE/var(obs) on the `score_valid` mask of the TRAIN years.
* Params sigmoid-transformed to their `__param_spec__` bounds; Adam;
  5 seeds = 5 random inits uniform in transformed space, PLUS the defaults as
  a 6th deterministic init (so "tuning can only help on train" is checkable).
* Gates before any result is quoted (repo AD rules): finite-difference check
  on >= 1 parameter per arm (ratio ~ 1); zero-gradient leaf aborts the run
  (no-inert-parameters); NaN-revert fraction reported per run.
* Inner-loop caveat (GLM Q5): tuned-optimum runs report the fraction of
  timesteps with the drawdown sigmoid cap binding, and the inner profit
  gradient residual at the returned optimum; a large binding fraction at the
  tuned point is reported next to the verdict.

## Verdict rule (fixed now; GLM-sharpened)

Seed SELECTION on train years only (best train loss). Per arm, the reported
central value is the **seed-median test score**; the seed spread is reported
as a multimodality check (collapse expected on a smooth 4-8D problem — a wide
spread is itself a finding of a degenerate landscape).

**P-hydro "wins" iff** the median paired per-site-per-test-year difference of
joint loss (A_tuned - B_tuned) exceeds a **5% relative margin** AND the sign
is consistent at >= 4 of 5 sites. Anything less = "no demonstrated advantage
of the hydraulic optimum over the tuned empirical stress at these sites".
The default-vs-default and default-vs-tuned cells are reported alongside
(4-cell matrix x seeds), but the headline claim is tuned-vs-tuned only.

## Stated limits

* 5 sites, one climate class (semiarid/Mediterranean) — a dry-site verdict,
  not a global one.
* Single global parameter vector per arm (no per-PFT traits).
* Leaf-water-potential observations remain unused; they stay the recommended
  discriminating instrument beyond flux skill.


## DEVIATION (measured, 2026-09-02): gradient tuning refuted by its own gate

The pre-registered FD gate FAILED for both arms before any tuning ran:
beta arm analytic/FD ratio 0.082; phydro arm analytic gradient ~1.9e11 vs
FD -44 — the backward pass explodes through the year-long prognostic scan
containing the 40-step unrolled hydraulic inner loop (the exact channel GLM's
design review flagged as Q5). Per the repo rule that an AD path failing its
verification must not produce results, the tuner now optimises the IDENTICAL
loss with per-seed Nelder-Mead (forward-only; non-finite candidates get a
recorded 1e3 penalty; a seed whose best point is penalised is FAILED).
Everything else in this pre-registration is unchanged: loss, year split,
bounds, seeds, verdict rule.

## DEVIATION 2 (2026-09-02, dual review of the Nelder-Mead tuner; user-approved)

Measured: one evaluation = 12 min with the 5 sites run in sequence (smoke,
16 CPUs). Both reviewers judged 80 evaluations a local polish in 8-D.
Changes, each asked and approved before any production run:

* **Non-finite forward -> +inf**, not 1e3 (codex P0: a finite normalised-MSE
  loss can exceed any fixed penalty, so a NaN point could be selected; GLM: a
  flat plateau gives the simplex no direction). Per-seed non-finite count and
  scipy convergence status are recorded; a seed is FAILED only if NO finite
  evaluation exists.
* **Budget 200 evals (arm A, 4 params) / 400 evals (arm B, 8 params)** —
  ~50 per parameter, equal per-parameter search (GLM: a fixed budget starves
  the 8-param arm). The 5 site forwards run in parallel processes so an
  evaluation costs one site-forward of wall-clock.
* **Initial simplex width 1.0 in raw (logit) space** (~25% of a parameter's
  range at mid-range); scipy's default 5%-of-x0 steps cannot move materially.
* **Headline pairing = the SELECTED seed per arm** (best train loss), paired
  per-site-per-test-year with the 5% margin and 4/5 site sign rule as written;
  a site-level bootstrap CI of the median is printed next to it. The
  matched-seed population (same init id in both arms) is reported as
  SECONDARY. The earlier sentence "reported central value is the seed-median"
  mixed two estimators (GLM) and is superseded for the verdict; the seed
  spread is still reported as the multimodality check.
* **init 0 is the GLOBAL-MEAN texture init** (wp/fc averaged over the 5 sites;
  one vector tunes all sites), not the per-site default. The untuned
  "default" cell of the 4-cell matrix uses the PER-SITE texture defaults —
  exactly what the harness runs (codex P1).
* **Test-year coverage floor 20% -> 10%** (codex P0, user-approved): US-Whs
  never exceeds 15% joint GPP+LE coverage in any year, so the 20% floor left
  it with NO test year and the 4-of-5 site rule would have run on four sites.
  At 10% US-Whs has 9 test years and US-SRM gains two; train years (top-2
  coverage) are unchanged. The split table was re-frozen before any tuning.
  The verdict now refuses to run unless all five sites are paired.
* Bug fix (not a choice): the initial soil state is now taken from the first
  timestep of the SELECTED years, not the whole driver (codex P0: a held-out
  year could initialise the training forward).

Stated limits added by the round-2 review (GLM): held-out YEARS at the SAME
sites test temporal generalisation only — arm B's extra parameters can absorb
time-invariant site structure that transfers across years but not across
sites (leave-one-site-out refit is the follow-up, not part of this verdict);
with 5 sites the >=4/5 sign rule is weak (exact one-sided p=0.19; 5/5 gives
0.03) and the site bootstrap is descriptive only; the 200/400 budgets
confound "more parameters" with "more search" — B's best loss within its
first 200 evaluations is recorded in the eval history and is the
budget-matched control.
