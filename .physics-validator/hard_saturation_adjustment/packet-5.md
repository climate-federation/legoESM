You are an independent ADVERSARIAL physics reviewer for LegoESM (JAX-native,
differentiable ESM). ROUND 5 (final confirmation). Be CONCISE. Verdict per
concern; end with `OVERALL VERDICT: <no substantive findings | findings remain: ...>`.

# THE DELTA SINCE ROUND 4 (the round-4 confirmed findings, fixed)

## Round-4 finding (b) — warm-pool OVERSHOOT — FIXED
You reproduced: T=320 K, p=65400 Pa, q_v=0.2, dt=100 s -> after the rate-capped
first step the next blended step reached RH=0.999151 (past saturation) and then
re-evaporated 3e-6 kg/kg.  Cause: when the heating cap is inactive, the finite
smooth term in the blend `smooth + activation*(cond_hard - smooth)` can EXCEED
`cond_hard` (the on-curve drain), so the blend overshoots the curve.

FIX: cap the blended condensation at `cond_hard` (the bracketed on-curve drain,
which by construction never overshoots), BEFORE the heating/mass rate limit:
```python
    blended = condensation + activation * (cond_hard - condensation)
    condensation = jnp.minimum(blended, cond_hard)          # <-- NEW: no overshoot
    dqv_cap = hard_max_heating_K * constants.c_pd / constants.L_v
    cond_cap = jnp.minimum(dqv_cap, jnp.maximum(q_v, 0.0)) / jnp.maximum(dt, 1.0e-10)
    condensation = jnp.where(condensation > 0.0,
                             jnp.minimum(condensation, cond_cap), condensation)
```
Why this is correct + preserves intent:
- `cond_hard >= 0` lands EXACTLY on the curve, so `min(blended, cond_hard)` can
  never condense past saturation -> the multi-step drain is MONOTONE FROM ABOVE
  (RH stays >= 1), for cold AND warm pools.
- The UNDER-condensing smooth branch (smooth < cond_hard: the below-threshold /
  mild case) is untouched by the min -> below-threshold behaviour preserved.
  The min only bites where the smooth OVER-condenses (cold-strong, or the warm
  linearised-psychrometric case) -> exactly the overshoot it must prevent.
- Evaporation (blended < 0 <= cond_hard) passes unchanged.
- Differentiable: where the min binds, condensation = cond_hard carries the
  implicit-function gradient (finite); elsewhere it is the blend.

## Round-4 finding (e) — test gap — FIXED
`test_large_pool_drains_to_saturation_over_many_steps` is now parametrised over a
COLD (230 K, 67 g/kg), a MODERATE (260 K, 100 g/kg) and the WARM counterexample
(320 K, 200 g/kg) pool, asserting: per-step ΔT <= 5 K; q_v monotone-decreasing
(no re-evaporation); RH >= 1 - 1e-4 EVERY step (strict, no undersaturation
tolerance); reaches RH~1; and a case-appropriate step count + temperature rise.
(The warm pool has q_sat ~ 0.12 there, so only ~3 g/kg drains in a few steps --
the point there is the NO-OVERSHOOT, not many steps.)

## Round-4 finding (c) — sub-ULP mass breach — ACKNOWLEDGED, non-substantive
The `min(dqv_cap, q_v)/dt` mass cap is mathematically sufficient; the ~1e-24
breach is `q_v/dt*dt` float rounding (sub-ULP).  `test_mass_positivity_cold_dry_column`
asserts `q_v_new >= -1e-20` (accommodates the rounding); the comment says the
column-vapour bound is exact in exact arithmetic.

## Round-4 (a) and (d) were false-positives (heating bound holds incl. the
smooth-leak path; no live-gradient loss) — unchanged.

# TEST RESULTS (JAX_ENABLE_X64=1, CPU) — all green
21/21 in test_hard_saturation_adjustment.py (incl. the 3 parametrised multi-step
pools with strict RH>=1-1e-4 and the warm 320 K overshoot counterexample).  No
regressions: 424 in no_inline_coeffs / warm_rain / reachability / param_specs;
earlier 4034 across the full ratchet + scheme + CLI set.

# UNCHANGED / still-holding (rounds 1-4)
Bracketed-bisection + implicit-gradient on-curve solver (worst |RH-1| 8.9e-14
across p>=100 Pa, q_v<=1 kg/kg; FD-grad match rel<=5e-7); per-step latent-heating
rate limit (<= hard_max_heating_K, default 5 K, dt-independent); mass-positivity;
c_pd*T+L_v*q_v conserved (structural, rtol 1e-12); flag-off byte identity; liquid
q_sat + L_v signs; jit/vmap; 5-scheme config threading + tier-2 specs +
unsupported-scheme raise.

# YOUR TASK (ROUND 5)
Confirm the overshoot fix is correct and complete: (a) can `min(blended,
cond_hard)` still overshoot saturation for ANY (T, p, q_v, dt) -- cold, warm, or
at the q_sat cap; (b) does it wrongly SUPPRESS legitimate below-threshold smooth
condensation (i.e. is `smooth < cond_hard` really the mild/under-condensing
regime, so the min is inactive there); (c) interaction with the subsequent
heating/mass cap and the q_c evaporation clamp; (d) any differentiability or
conservation regression; (e) anything else.  If no substantive findings remain,
say so explicitly and justify each candidate concern.  End with the OVERALL
VERDICT line.
