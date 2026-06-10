# Veros ACC oracle — final scoreboard (Phase G close-out, 2026-06-09)

**The question.** Can legoESM, assembled from its canonical blocks by pure
config (the recipe), replicate a production model — Veros's ACC channel — at
the per-tendency level (tier 2), the realized-increment level (rung 6), and
the climate level (multi-year statistics)?

**The answer.** The **dycore-numerics-matching program is complete**: every
Veros dycore option is imported as a config-selectable, default-off,
bit-identical-when-off option, and each either matches Veros at or near
machine precision or was *measured* not to be the climate lever. The remaining
climate residual is the **eddy-mean equilibration** (parameterized-eddy APE
drainage under the cold-started prognostic EKE / vertical momentum partition)
— a physics-regime difference, definitively not the discretization. Details,
evidence and commits: `oracle_recipe_strategy.md` §8 ledgers.

## Per-tendency / per-block scoreboard (tier 2 + rung 6, frozen bridged states)

| Block | Match vs Veros | Note |
|---|---|---|
| Setup / constants / grid | bit-identical | `VEROS_CONSTANTS_CONFIG`, no monkey-patching (G-C4) |
| Density / EOS / pressure | corr 1.0000, L2 0.037 kg/m³ | shared discretization; veros_nonlin2 verified vs `get_rho.py` |
| Coriolis | corr 0.96 / 0.98 | validates grid alignment + bridge |
| Lateral friction operator | machine-0 term-for-term | flux-divergence `harmonic_friction` + matched K_diss_h |
| Momentum advection | corr 0.654 / 0.715 (flux-form) | tier-2 row is a decomposition artifact; total matches (below) |
| Implicit vertical friction (du_mix) | corr 0.99 / 0.96 | operator evaluated on bridged u^n |
| Explicit momentum tendency (total) | corr 0.95 / 0.91 | after fixing the Veros stale-time-level capture |
| Realized momentum increment | corr 0.11 → **0.28**, L2 4.8 → **2.3** (u) | additive friction placement; ceiling 0.44 is bridge-seeding-limited; v noisy (documented) |
| EKE closure (K_gm given E, L) | machine-exact (corr 1.0, rel err 0) | E9; ACC params verified vs `acc.py` |
| eke_len (Rhines/Rossby) | machine-exact (5.8e-16 / 0.0) | L4 |
| K_iso = K_gm (prognostic Redi) | machine-exact (transitive) | R3 |
| Convection / static-stability N² | machine-exact (2461 = 2461 unstable cells; abyssal K_H 3.78e-5 = 3.78e-5) | adiabatic-displacement N² — same class as the slope fix |
| GM/Redi T_iso tendency | 0.17 → **0.37** (Veros-K_iso-fed) | neutral-density slopes; K_33 ~2× (was 20–100× small); kr-sum refinement inert on uniform ACC (0.3665→0.3666), matters for variable metrics |
| Tracer realized increment | L2 3.85 → **1.69** (floor 1.23) | implicit surface-forcing placement (rung 6 discovery) |
| Outer integrator (AB2 + dt_mom≠dt_tracer + rigid lid) | bit-faithful (rung 6: adds no error) | #44; async ratio 9 conserves at machine precision under rigid lid |

## Climate scoreboard (10-yr vs 10-yr, same setup + metrics — REFRESHED 2026-06-10)

The 2026-05-31 numbers (KE 0.94×, EKE 2.42×) predated five fidelity options and
used different metrics; the refreshed apples-to-apples comparison
(`.physics-validator/veros_ablation/`, full faithful stack incl. prognostic
TKE `2bfa841a`) is the authoritative one:

| Metric | Veros | legoESM (diag TKE) | legoESM (prog TKE) | Status |
|---|---|---|---|---|
| ACC transport | 248 Sv | 1.52× | **1.35×** | OPEN, improved |
| Total KE | 5.9e16 J | 5.41× | **4.51×** | OPEN, improved |
| Mean EKE | 1.7e-3 | 19.8× | **17.4×** | OPEN, improved |
| Abyssal T | 0.27 °C | +2.8 °C | +3.15 °C | OPEN |
| u_bot/u_surf | 0.123 | 2.42× | **2.31×** | OPEN, improved |
| Surface T | 12.03 °C | 0.99× | 0.99× | restoring pins it ✓ |

**The 2026-06-09/10 Layer-1 campaign** (user challenge: "chaos explains
weather, not climate"): the closure energy cycle was audited and CLOSED —
prognostic TKE built (the one structural hole; moves the residual ~15% the
right way), all recycling/budget terms (H1/H2/H3/H6) measured **climate-inert
in Veros itself** via runtime ablations, and the forcing fields exonerated at
machine precision (realized wind acceleration ratio 1.000000). The residual
is therefore NOT the closures' form, the forcing, the integrator, or the
placements.

The three OPEN rows are one structural difference: the too-barotropic vertical
momentum partition under the cold-started parameterized-eddy field (legoESM
holds ~5× the APE; the drainage — GM isopycnal flattening — is the weak link,
not the generation). Every dycore candidate for this residual was imported and
measured NOT to move it (~7 measure-first negatives: integrator scheme,
friction form *and* coefficient, momentum advection, convection, A_v,
EKE/GM depth structure, bottom-drag mapping).

## Method lessons (what made this defensible)

1. **Measure first, never tune to the metric.** Five separate hypotheses this
   arc were killed by their own isolation runs (momentum advection, convection
   as abyssal lever, EKE/GM depth unification, A_v, "missing barotropic
   dissipation").
2. **Truth tiers outrank oracle-matching.** Every option shipped with
   bit-identity-off, conservation, cancellation and AD gates before its oracle
   number counted.
3. **Map oracle parameters by scheme/units, not value** (the r_bot rate-vs-
   stress 276× lesson), and **verify the oracle's own time level** (stale
   du[tau] capture; the 3-slot rotation aliasing).
4. **Comparison artifacts are findings too**: du_adv (decomposition), du_mix
   (implicit vs explicit), constant-kappa T_iso (kappa confound) all looked
   like model gaps until the harness was made fair.

## Open levers (ranked, post-campaign 2026-06-10)

Never-isolated comparisons — the only places the residual can still hide
before the eddy-mean-equilibration attribution stands on closed ground:

1. **Rigid-lid ψ-solve / barotropic-coupling OUTPUTS** on an identical RHS
   (never snapshot-compared): Veros's dpsi extrapolation as CG guess, the
   AB2-of-dpsi structure, and the vertical-mean removal/re-add sequence
   (`solve_stream.py:164-199`) vs legoESM's un-AB2'd barotropic mode.
2. **Spherical metric/curvature terms** in the momentum budget (Veros
   `momentum.py:43-73` tantr terms) — bounded by coriolis corr 0.96/0.98 but
   never isolated; systematic O(u·v·tanφ/R).
3. **Isolated vertical momentum advection** + **realized bottom-drag
   tendency** (formula-matched, never tier-2'd).
4. **T_iso 0.37 pattern** (the APE-drainage operator's spatial structure).
5. **Equilibration experiments** (warm-started / equilibrated EKE+TKE,
   longer runs): the climate lever if 1–4 clear. 10-yr runs cost ~25 min on
   a V100S — an attribution matrix is cheap now.


## Overnight falsification campaign (2026-06-10, addendum)

Run-length asymptote (10/30/100-yr matched pairs): KE ratio 4.51 → 3.46 →
**3.24** (flattening ≈3×), EKE → 13.7×, abyss ratio dissolves to 1.86 (Veros's
own abyss drifts to 2.64 °C). Verdict: a genuine ≈3× attractor difference plus
a transient overshoot.

Falsifications (each a 10-yr GPU run via the override/monkeypatch harness,
`.physics-validator/veros_ablation/legoesm_10yr_variant.py`):
- background `A_v=0`: NO effect (0.997–1.018) — killed;
- realized TKE shear production (1/(1+2δ) attenuation of the measured 6.1×
  over-forcing): NO effect (KE +6%) — the TKE pathway does not control this
  ACC's energetics;
- `kappa_gm_max=5e4`: marginal (KE −3%, EKE +14% — the cap partially CONTAINS
  the overshoot).

Expert audits (ocean-model + dycore personas; reports under
`.physics-validator/{ocean_expert_audit,dycore_expert_audit}/`): priority
dycore suspects verified clean (dt_mom semantics faithful; AB2 barotropic
composition algebraically equivalent on flat bottom; tracer-velocity offset a
non-compounding 0.2% phase lead; the balanced mode exactly neutral in both
assembled steps). THREE concrete code-level survivors, never matched:
1. **D3 — vertical momentum advection**: legoESM 1st-order upwind on the
   baroclinic perturbation u′ vs Veros 2nd-order centered on full u (the
   recipe's `momentum_flux_scheme="centered"` reaches only the horizontal
   block). Missing ∂z(w·U_bar) + upwind viscosity ≈1e-3 m²/s damping
   baroclinic shear → too-barotropic (matches 0.30 vs 0.12); ~15% of the jet
   spin-up rate. Best symptom match.
2. **D1 — Coriolis composition**: sequential Matsuno sub-step inside the outer
   AB2 destroys 14–23 %/step of near-inertial energy (+ ≈13–15° discrete-Ekman
   rotation) vs Veros's explicit-AB2 Coriolis that routes ageostrophic energy
   through friction → TKE.
3. **D2 — AB2 scope**: legoESM extrapolates the TOTAL explicit increment;
   Veros AB2s only {Coriolis, advection, wind, PGF} with dissipative terms at
   weight 1.0.
Ocean-side smaller gaps (TKE buoyancy time level, surface-buoyancy TKE source,
negative-TKE reservoir, Ri definition, z-grid T-point placement
`u_centered_grid` vs midpoints) are catalogued in the audit report.
