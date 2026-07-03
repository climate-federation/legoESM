# Veros ACC oracle — final scoreboard (Phase G close-out, 2026-06-09)

**The question.** Can legoESM, assembled from its canonical blocks by pure
config (the recipe), replicate a production model — Veros's ACC channel — at
the per-tendency level (tier 2), the realized-increment level (rung 6), and
the climate level (multi-year statistics)?

**The answer (FINAL, 2026-06-10): YES — and the residual was numerics after
all.** Every Veros option is a config-selectable, default-off block, and the
multi-year climate now matches: 10-yr cold-start KE **0.95×** Veros, and
legoESM **holds Veros's equilibrium** when initialized in it (drop-hold KE
1.07× over 5 yr, where the pre-fix stack departed ×8). The earlier conclusion
that the residual was "the eddy-mean equilibration, definitively not the
dycore numerics" was **overturned** by the 2026-06-09/10 Layer-1 campaign:
the dominant drivers were (1) the EKE mixing-length computed from the in-situ
N² (`abcbd255`), (2) the positive-definite GM-skew EKE source form
(`5bdce06a`), (3) upwind-on-perturbation vertical momentum advection
(`678e0cbc`, fixed the transport overshoot), and above all (4) **the Matsuno
Coriolis time-composition** (`3bb0bedb`) — destroying near-inertial energy at
|G|≈0.83/step where Veros's explicit-AB2 Coriolis is ~0.997, invisible to
every per-operator comparison. The method lesson: per-tendency matching at
corr 0.95+ does NOT bound the assembled step's energy pathways; drop-hold
tests from the oracle's own equilibrium + term-by-term budgets on that state
do. Details: `oracle_recipe_strategy.md` §8 ledgers.

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


## ROOT CAUSE FOUND AND FIXED (2026-06-10 morning, `abcbd255`)

The attractor-drop test (legoESM departs a verified 1%-faithful copy of
Veros's 30-yr equilibrium: KE ×6.5 in 5 yr, **EKE runs away first ×26**, Veros
control flat ±1.1%) plus a term-by-term EKE budget on that same state
localized the runaway: **net dE/dt flips sign** (Veros −9.2e9 W relaxing,
legoESM +3.14e10 W growing). 71% of the bias: `_eady_growth_and_length` used
the **in-situ N²** (compressibility-biased; int_N_dz 2.67× too large → L_def
2.1× → eke_len 23% long → dissipation ∝1/L too weak). The third instance of
the in-situ-vs-locally-referenced bug class; it survived oracle gate L4
because L4 fed Veros's own N² — formula exact, input wrong.

Fix: `n2_mode="adiabatic"` on the EKE/Visbeck Eady–Rossby chain (`abcbd255`,
default-off byte-identical, validator SHIP). **10-yr falsification: ACC
transport 1.35× → 1.03× (matched); EKE 17.4× → 8.9× (halved); KE 4.51× →
4.18×.** Necessary but not sufficient — remaining E-budget levers: the
positive-definite realized-skew form (1.22× vs Veros's signed −P_diss_skew,
32% of the bias), the missing −P_diss_iso sink (15%), and the dycore-audit
items D1 (Coriolis composition), D2 (AB2 scope), D3 (vertical momentum
advection: upwind-on-perturbation vs centered-on-full-u — the best match for
the residual barotropic/KE bias).


## FINAL RESOLUTION (2026-06-10 afternoon)

| Metric (10-yr cold start vs Veros) | before campaign | after `3bb0bedb` |
|---|---|---|
| total KE | 5.4× | **0.95×** |
| u_surf | 2.06× | **0.99×** |
| u_bot/u_surf | 2.42× | **1.29×** (0.154 vs 0.119) |
| mean EKE | 19.8× | **2.5×** |
| ACC transport | 1.52× | 0.87× (221 Sv — modest undershoot, open) |
| abyssal T | +2.8 °C | +1.7 °C (open, halved) |
| **drop-hold from Veros equilibrium** | departs ×8 | **HOLDS (1.07×)** |

Open residuals (small, named): the ACC-transport undershoot (interaction of
D1×D3×additive-friction worth one attribution pass), the remaining abyssal
warm drift (both models drift; legoESM faster), EKE 2.5×, and the
documented-deferred items (P_diss_adv/nonlin TKE terms, the w→v-face metric
averaging, D2 AB2-scope, free-surface Coriolis scoping). The oracle-recipe
method is validated end-to-end: recipe + bridge + frozen-state tiers +
**drop-hold** + budget-on-state probes + Veros-side runtime ablations.


## Phase H: open-issues sweep + the first transfer test (2026-06-10 evening)

- **Transport attribution**: full stack 221 Sv / without-D3 219 / without-
  additive-friction 279, vs Veros 248 — and Veros's own transport wanders
  218–255 Sv across its 10/30/100-yr runs. The additive-friction placement is
  the dominant composition knob (±26%); both brackets sit at the edges of the
  oracle's natural variability. Closed as composition sensitivity, not a bug.
- **D2 (AB2 scope, `8db3a75b`)**: dissipative terms at weight 1.0 (Veros's
  exact scope). The first build's routing bug (diss depth-mean withheld from
  the barotropic forcing — Veros's uloc includes du_mix) was caught by the
  independent review AND the 3865 Sv falsification blow-up, fixed per the
  reviewer's prescription, re-verified (ψ-ratio 0.9999, was 40×).
  Climate-neutral on the ACC (218 vs 221 Sv) — a faithfulness/stability
  option. All four dycore-audit items are now built and falsified.
- **TRANSFER TEST #1 — ACC_Basic (`88e768e3`): the recipe blocks GENERALIZE.**
  Veros's analytic TKE-only twin differs in exactly two physics settings;
  both mapped onto existing config options, zero new numerics. 10-yr verdict:
  KE 0.978×, transport 0.894× (oracle-wander class) — the same closeness
  class as the matched ACC. The oracle-recipe method's generalization claim
  has its first proof point.

Remaining (small): the shared abyssal warm drift (both models drift; legoESM
faster — equilibration-rate difference, characterized not closed), EKE 2.5×,
and the documented-deferred minor items. NEXT: the first data-backed
transfer target, Veros `global_4deg` (monthly forcing, real bathymetry, ice
mask — harness glue only, no new numerics expected).

## Phase H (cont.): transfer test #2 — global_4deg, the first data-backed oracle (2026-06-10/11)

**Pre-validation — the ACC two-flip mini-transfer** (`.physics-validator/acc_two_flip/`):
global_4deg's two new options (gsw EOS `9d11dd09`, TKE W-grid superbee advection
`8fdd20db`) flipped on BOTH models in the already-matched ACC config. Cross-model
two-flip: KE 0.967, transport 0.931 — same class as the matched baseline (0.914 /
0.892), slightly tighter. Delta axis: Veros climate-neutral (1.002); legoESM yearly
traces ≤1.2% through yr 7 (the +6% endpoint is eddy-spindown phase). Attribution:
the EOS carries the endpoint sensitivity; TKE advection verified-active but
climate-inert in both models. A fresh stock run at HEAD reproduced the banked
baseline exactly — all session commits confirmed default-inert.

**Transfer test #2 — global_4deg (`f4d93945`).** "No new numerics expected" was
wrong by exactly two (gsw EOS, TKE superbee advection — both now canonical
config options) plus one forcing form (`flux_feedback`: prescribed flux +
qnec·(SST−T) feedback + ice mask, `2ca8d21f`). The recipe itself is pure config
with ONE physics flip vs ACC (`isopycnal_diffusion=False` = the Veros default
for the absent setting). `eke_diss_surfbot` proven inert (idemix-gated).

**The composition surface was the real discovery: NINE variable-bathymetry
bugs, all flat-bottom no-ops** (ACC could never see them), each independently
review-proven (flat-bottom bit-identity by 22-leaf sha256; manufactured
stepped-bathymetry conservation probes). Headline: GM/Redi triad fluxes
crossed closed faces and the seafloor → −6%/yr volume-salt leak (now −3e-17),
which alone drove the downstream instability cascade. Also: AB2 closed-face
recurrence |r|=1.13/step, face-A_v seafloor momentum leak (w 100× oracle),
sub-seafloor velocity consumption in TKE/EKE advection, rock-cell GM/convection
artifacts, gsw grad NaN at dry cells, island 4→8 connectivity (4-conn produced
11 islands with 5 contradictory ψ constraints; Veros island.py:18 verbatim).

**10-yr verdict vs the banked Veros oracle** (26-min CPU oracle; legoESM 45 s
on a V100S):
- EKE-off (Veros's constant-K branch): **ψ_range 0.99, vol-S 1.002 — PASS**;
  transients converge from opposite sides (Veros spins down 240→212 Sv from
  data IC, legoESM spins up from rest 187→211 Sv). vol-T +0.16 K and KE 1.62×
  = the known equilibration class.
- Fully-faithful (prognostic 3-D EKE): yr-1 transport ratio 1.00, then the
  EKE equilibration RUNS AWAY on the global domain (×18 by yr 8, NaN at 8.9).
  The ACC's 2.4× EKE overshoot is hereby confirmed as THE single open
  residual of the oracle program — and it now has a 45-s/10-yr GPU testbed.
