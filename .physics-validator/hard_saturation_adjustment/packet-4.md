You are an independent ADVERSARIAL physics reviewer for LegoESM (JAX-native,
differentiable ESM). ROUND 4. Be CONCISE and decisive: verdict per concern
(confirmed-bug / false-positive / ambiguous) with line-cited reasoning; end with
`OVERALL VERDICT: <no substantive findings | findings remain: ...>`.

# THE DELTA SINCE ROUND 3 (two things added)

## (1) Per-step RATE LIMIT (new field-evidence requirement)
Field observation: the naive post-step cap DETONATED a run at day 65 by draining
a 67 g/kg pool in ONE step (~165 K of latent heat that step). Requirement: the
hard adjustment must bound the per-step condensation so large pools drain over
MANY steps.

Implementation (in `saturation_adjustment`, AFTER the smooth->hard blend):
```python
    condensation = condensation + activation * (cond_hard - condensation)
    dqv_cap = hard_max_heating_K * constants.c_pd / constants.L_v   # per-step Δq_v [kg/kg]
    cond_cap = jnp.minimum(dqv_cap, jnp.maximum(q_v, 0.0)) / jnp.maximum(dt, 1.0e-10)
    condensation = jnp.where(condensation > 0.0,
                             jnp.minimum(condensation, cond_cap), condensation)
```
- New config field `hard_sat_max_heating_K: float = 5.0` [K] on all 5 warm-rain
  configs (+ __param_spec__ tier-2, units K, bounds (0.5, 50.0)); threaded to the
  5 callers; baselined in the reachability audit like the threshold.
- Bounds the per-step LATENT HEATING (ΔT = L_v*cond*dt/c_pd) to
  hard_max_heating_K (default 5 K = ~2 g/kg per step, dt-independent increment).
- Applied to the BLENDED positive rate (not just cond_hard) so the smooth
  branch's own over-condensation cannot leak past the cap via (1-activation)
  (a real bug caught in test: at RH=1.3/400 K the smooth rate alone heats 8.7 K).
- The `min(dqv_cap, q_v)` also enforces MASS-POSITIVITY (never condense more than
  the column holds) — see (2).
- Mild super-saturation (full drain below the cap) is untouched → still lands ON
  the curve in one step.  The uncapped solver (`hard_max_heating_K` huge) still
  lands on the curve (round-1/2 Bug-1 regressions retained).

## (2) Round-3 fixes
- Finding A (mass-positivity): `saturation_mixing_ratio` returns a tiny NEGATIVE
  value (~ -2e-11 kg/kg) at very cold T (150 K) from the smooth upper cap, so the
  smooth branch could drive q_v < 0 at a dry/sub-saturated cold cell.  The
  `min(dqv_cap, q_v)/dt` cap above now bounds the positive condensation by the
  available vapour, so q_v_new >= 0 always.  New test
  `test_mass_positivity_cold_dry_column` (q_v in {0, 1e-11, 1e-8, 1e-5} at
  150-200 K, dt in {30,100,1200}) asserts q_v stays >= 0.  (The tiny-q_sat cold
  cell's sub-saturation MIS-judgment persists — the true q_sat ~ 8e-11 is lost to
  the -2e-11 cap — but the vapour there is ~1e-11 kg/kg (negligible) and mass is
  now non-negative; fixing the negative q_sat is a thermo-module change out of
  scope and would break bit-identity of the smooth path.)
- Finding B (wording): the "exactly on-curve wherever above threshold" claim is
  corrected to a tolerance-qualified statement (a finite sigmoid never reaches
  exactly 1; RH_new ~ 1.001 at RH=1.25, < 1e-3 by RH ~ 1.5).

# THE MULTI-STEP DRAIN (coordinator's required behaviour), verified
`test_large_pool_drains_to_saturation_over_many_steps`: a 67 g/kg column at 230 K
is iterated; each call has ΔT <= 5 K, q_v is monotone-decreasing, RH stays >= 1
(approaches saturation FROM ABOVE — no overshoot), and it reaches RH~1 after
MANY steps (not one), T rising from the released heat.  Mechanism: each step
condenses min(full_drain_to_curve, cap); while capped, only Δq_v_cap condenses so
q_v_new > q_eq >= q_sat(T_new) (RH stays > 1); the final steps (full_drain < cap)
land exactly on the curve.

# UNCHANGED / still-holding (rounds 1-3)
Bisection + implicit-gradient solver (worst |RH-1| 8.9e-14 across p>=100 Pa,
q_v<=1 kg/kg; FD-grad match rel<=5e-7); conservation of c_pd*T+L_v*q_v (structural,
rtol 1e-12); flag-off byte identity; liquid q_sat + L_v signs; donor-clamp
composition; jit/vmap; unsupported-scheme raise; tier-2 param specs.

# TEST RESULTS (JAX_ENABLE_X64=1, CPU) — all green
19/19 in test_hard_saturation_adjustment.py (conservation; UNCAPPED lands on
curve for Tibet/low-p/extreme-q_v; smooth-overshoot contrast; per-step heating
rate-limited across dt in {30,100,300,1200}; 67 g/kg multi-step drain monotone +
no overshoot + many steps; mild lands on curve in one call; flag-off identity;
grad finite + uncapped FD-match d/dq_v,d/dT,d/dp; jit/vmap; cap region;
hot-subsaturated not condensed; no-NaN; mass-positivity cold/dry; config
threading + raise; kessler scheme heating-bounded + off byte-identical + both
conserve).  No regressions: 4034 passed (param_specs, no_inline_coeffs,
no_saturation_reimpl, reachability, warm_rain, physics_microphysics,
run_amip_cli).  fp32 path verified earlier (RH_new ~ 1.000003, flag-off
byte-identical).

# YOUR TASK (ROUND 4)
Confirm the rate limit is correct and complete: (a) is the per-step heating truly
bounded for ALL dt and any q_v (incl. the smooth-leak and mass-positivity paths);
(b) does the multi-step drain really converge monotonically from above without
overshoot for cold AND warm large pools; (c) is the `min(dqv_cap, q_v)` mass
bound sufficient (any remaining path to q_v<0 when hard_adjust=True, e.g. jointly
with the q_c evaporation clamp or the caller donor clamps); (d) any
differentiability regression from the added `where`/`minimum` (dead gradient
where a cap binds is expected/correct — flag only if a LIVE-sensitivity region
lost its gradient); (e) anything else. If no substantive findings remain, say so
and justify each candidate concern. End with the OVERALL VERDICT line.
