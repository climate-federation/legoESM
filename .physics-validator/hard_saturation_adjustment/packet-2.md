You are an independent ADVERSARIAL physics reviewer for LegoESM (JAX-native,
differentiable ESM). This is ROUND 2. Be CONCISE and decisive: give a short
verdict per concern (confirmed-bug / false-positive / ambiguous) with line-cited
reasoning, and end with an overall verdict. Do not re-derive the whole repo;
focus on the diff described here. You may read files under the working root
(read-only) but keep exploration tight.

# WHAT CHANGED

An opt-in HARD (Newton-iterated) saturation adjustment for the warm-rain
microphysics path. Where q_v > threshold*q_sat, it lands q_v ON the liquid
saturation curve, conserving c_pd*T + L_v*q_v exactly (the caller maps the
returned condensation rate to dT_dt=+L_v/c_pd*rate and dq_v_dt=-rate, so the
enthalpy invariant is structural). Opt-in `hard_saturation_adjustment: bool` +
`hard_sat_adjust_threshold: float=1.1`; byte-identical when off.

Files (read the diffed regions):
- packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py
  lines ~20-46 (constants), ~159-211 (`_hard_saturation_condensation`),
  ~214-336 (`saturation_adjustment` with the `if hard_adjust:` block ~300-323).
- config.py: 2 fields + __param_spec__ entry on Kessler/SeifertBeheng/Morrison/
  Thompson/P3; `apply_microphysics_experiment_flags` threads the bool (raises on
  a scheme lacking the field).
- morrison/kessler/seifert_beheng/thompson/p3.py: pass the two config values to
  `saturation_adjustment`.
- driver/config.py (ExperimentConfig bool), model_driver._run_mpas +
  physics_pipeline._resolve_microphysics (threading), run_amip.py (CLI flag).
- tests/unit/test_hard_saturation_adjustment.py (15 tests, all pass, x64 CPU).

# ROUND-1 FINDING (from your numerical probing) — ALREADY FIXED

You found that at LOW pressure (p ~ 100-1000 Pa, model-top levels) the
Clausius-Clapeyron quasi-Newton Jacobian `dq_sat/dT = L_v*q_sat/(R_v*T^2)`
UNDER-estimates the (very steep) true slope of q_sat = epsilon*e_sat/(p-e_sat)
— the true derivative carries an extra p/(p-e_sat) factor. So the iteration is
damped-OSCILLATORY (not monotone) there and 8 steps landed at RH ~ 0.35
(over-condensed / sub-saturated) at 200 K / 100 Pa.

FIX APPLIED: `_HARD_SAT_NEWTON_ITERS` raised 8 -> 16 (a fixed loop count, not a
config knob). Verified numerically: 16 steps converge |RH-1| to <= 1e-7 across
the full p >= 100 Pa, T in [150,400] K, q_v=66 g/kg battery (model p_top is 200
Pa, so 100 Pa is beyond range — a safety margin). The exact-derivative (jvp)
alternative was tested and REJECTED: it oscillates/overshoots inside the q_sat
cap (T->~400 K, e_sat>p, exact slope ~0 => psychro~1 => no damping). The CC
Jacobian stays large in the cap (uses the capped q_sat) and damps there; the
extra iterations handle the low-p steepness. A new regression test
(`test_hard_adjustment_lands_on_curve_at_low_pressure`) pins RH~1 for
p in {100,200,500,1000} Pa. The clip(q_eq,[0,q_v]) each iteration keeps the
solve bounded and finite; conservation was never violated (structural), only the
LANDING point was off before the fix.

# THE SOLVER (current)

```python
def _hard_saturation_condensation(T, q_v, p_full, dt, q_sat):
    l_over_cp = constants.L_v / constants.c_pd
    q_v_pos = jnp.maximum(q_v, 0.0)
    q_eq = jnp.clip(q_sat, 0.0, q_v_pos)                 # init below root
    for _ in range(16):                                 # _HARD_SAT_NEWTON_ITERS
        t_new = T + l_over_cp * (q_v_pos - q_eq)
        q_sat_new = saturation_mixing_ratio(t_new, p_full)   # LIQUID, capped
        dqsdt = constants.L_v * q_sat_new / (constants.R_v * t_new ** 2)
        psychrometric = 1.0 + dqsdt * l_over_cp
        q_eq = q_eq - (q_eq - q_sat_new) / psychrometric
        q_eq = jnp.clip(q_eq, 0.0, q_v_pos)
    return (q_v_pos - q_eq) / dt
```
Blend (in `saturation_adjustment`, after the smooth `condensation` is formed;
q_sat is the capped liquid saturation already computed, fp64-then-cast under x64):
```python
    if hard_adjust:
        cond_hard = _hard_saturation_condensation(... fp64 when x64 ...)
        rel_humidity = q_v / jnp.maximum(q_sat, 1e-12)   # capped q_sat => cap region drains
        activation = jax.nn.sigmoid(20.0 * (rel_humidity - hard_threshold))
        condensation = condensation + activation * (cond_hard - condensation)
```

# TEST RESULTS (JAX_ENABLE_X64=1, CPU) — all green

15/15 in test_hard_saturation_adjustment.py: enthalpy conservation (rtol 1e-12);
lands-on-curve for the Tibet battery (116-290 K) AND the new low-p battery
(100-1000 Pa); cold-strong-supersat smooth OVERSHOOTS while hard lands on curve
(hard condenses less); warm-mild-supersat smooth UNDER-condenses while hard
lands on curve (hard condenses more); flag-off byte-identity; grad finite+nonzero
+ matches centered FD; jit & vmap match eager; cap region q_v=1.3 kg/kg@400 K
drains to ~cap; hot sub-saturated cell not condensed; no NaN on extreme+dry;
experiment-flag threading + raise for sundqvist; kessler scheme drains + off is
bit-identical + both conserve. Also green with no regression: test_warm_rain,
test_param_specs, test_no_inline_physics_coeffs, test_no_saturation_reimpl (6735
passed), test_dispatch_hardening, test_validate_strict_coverage,
test_physics_contracts, test_params_reachability_audit, physics_microphysics,
kk2000, seifert_beheng/kessler/thompson faithful, run_amip_cli.

# YOUR TASK (ROUND 2)

1. Confirm the round-1 low-pressure fix is SOUND and complete (is 16 enough for
   the model range; can the damped oscillation still land off-curve or NaN for
   any physical (T,p,q_v); is rejecting the exact-jvp Jacobian justified).
2. Find any REMAINING bug: sign/units (liquid q_sat + L_v vs ice/L_s),
   differentiability (dead-gradient/NaN paths through the clips, div by t_new /
   psychrometric / the 1e-12 floor), conservation once the rate flows through
   morrison's qv/qc donor clamps and kessler's extra q_v/dt clamp, blend
   continuity (F3) and cap handling (F2), byte-identity when off, repo-doctrine
   (hardcoded constants/sharpness, saturation re-impl, dispatch raises,
   param-spec tier).
3. If you believe there are NO substantive remaining findings, say so explicitly
   and justify why each candidate concern is not a bug.

End with: OVERALL VERDICT: <no substantive findings | findings remain: ...>.
