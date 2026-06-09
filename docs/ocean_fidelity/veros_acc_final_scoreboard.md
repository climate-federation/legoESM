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

## Climate scoreboard (multi-year free runs, faithful stack)

| Metric | legoESM / Veros | Status |
|---|---|---|
| ACC transport | O(100 Sv) vs 110.6 Sv (chaotic at 1 yr) | rigid lid restored Veros magnitude (free surface gave 18 Sv) |
| Total KE | **0.94×** (was 1.48×) | closed by the EKE-source match cascade |
| Front sharpness | **0.97×** | closed |
| EKE | **2.42× overshoot** | OPEN — faithful overshoot (source matches on same state; legoESM flow more energetic) |
| Abyssal T | **+5 °C warm** | OPEN — driven by the ~6× too-strong resolved Eulerian overturning |
| u_bot/u_surf | 0.49 vs 0.21 (too barotropic) | OPEN — vertical momentum partition / eddy form stress |

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

## Open levers (ranked, for whoever picks this up)

1. **Eddy-mean equilibration** (the climate residual): research-level — eddy
   form stress / APE drainage / warm-started or equilibrated EKE experiments.
2. **Momentum realized-increment ceiling**: bridge Veros's psi/dpsin + du
   history at higher fidelity if the 0.28→0.44 gap matters.
3. **T_iso beyond 0.37**: the remaining confound is the 3-D kappa structure
   fed to the comparison + per-corner metric factors (tripolar); the slope
   form itself is now Veros-faithful including the kr pairing.
