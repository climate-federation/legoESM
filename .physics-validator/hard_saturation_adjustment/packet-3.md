You are an independent ADVERSARIAL physics reviewer for LegoESM (JAX-native,
differentiable ESM). This is ROUND 3. Be CONCISE and decisive. Verify the two
ROUND-2 confirmed-bugs are FIXED and find any remaining substantive issue. End
with `OVERALL VERDICT: <no substantive findings | findings remain: ...>`.

# ROUND-2 confirmed-bugs and the fixes applied

## Bug 1 (convergence for strong cold super-saturation) — FIXED by switching the
solver from a Clausius-Clapeyron Newton iteration to a BRACKETED BISECTION with
an implicit-function-theorem gradient.

Rationale: the residual f(q_eq) = q_eq - q_sat_liq(T + (L_v/c_pd)*(q_v - q_eq), p)
is strictly increasing in q_eq with f(q_sat(T)) <= 0 <= f(q_v), so the root is
bracketed in [q_sat(T), q_v].  Bisection is unconditionally convergent for ANY
(T,p,q_v) — including the q_sat cap edge and low pressure where a CC-Newton step
overshoots into a sub-saturated (over-condensed) state.  Verified: worst |RH-1| =
8.9e-14 across p>=100 Pa, T in [150,400] K, q_v in [0.066, 1.0] kg/kg (your 0.3
and 1.0 cases now land on the curve).

The bracketing `where` reductions give a finite-but-inaccurate raw gradient, so
the bisection root is `stop_gradient`-detached and the EXACT gradient re-attached
by one Newton correction at the converged root (residual ~0 so VALUE unchanged;
derivative = implicit gradient dq_eq/dtheta = (dq_sat/dtheta)/psychrometric, with
psychrometric = exact 1+(L_v/c_pd)*dq_sat/dT via a jvp, held constant).  Verified
vs centred finite differences: rel <= 5e-7 in d/dq_v, d/dT, d/dp at moderate AND
extreme (q_v=1.0, p=200 Pa) states; jit/vmap consistent; grad finite on
extreme+dry inputs.

Current solver (packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py):
```python
def _hard_saturation_condensation(T, q_v, p_full, dt, q_sat):
    l_over_cp = constants.L_v / constants.c_pd
    q_v_pos = jnp.maximum(q_v, 0.0)
    lo = jnp.clip(q_sat, 0.0, q_v_pos)     # f(lo) <= 0
    hi = q_v_pos                           # f(hi) >= 0
    for _ in range(30):                    # _HARD_SAT_BISECT_ITERS
        mid = 0.5 * (lo + hi)
        t_new = T + l_over_cp * (q_v_pos - mid)
        f_mid = mid - saturation_mixing_ratio(t_new, p_full)
        hi = jnp.where(f_mid > 0.0, mid, hi)
        lo = jnp.where(f_mid > 0.0, lo, mid)
    q_eq = jax.lax.stop_gradient(0.5 * (lo + hi))
    t_new = T + l_over_cp * (q_v_pos - q_eq)
    q_sat_new, dqsat_dt = jax.jvp(
        lambda t: saturation_mixing_ratio(t, p_full),
        (t_new,), (jnp.ones_like(t_new),))
    psychrometric = jax.lax.stop_gradient(1.0 + dqsat_dt * l_over_cp)
    q_eq = q_eq - (q_eq - q_sat_new) / psychrometric
    return (q_v_pos - q_eq) / dt
```

## Bug 2 (blend/F3: activation 0.5 at threshold, misleading comment) — FIXED.
The blend is UNCHANGED in form (still a smooth sigmoid ramp — a sharp step would
reintroduce the F3 discontinuity), but: (a) the false "activation ~ 0 at
threshold" comment is corrected to state activation = 0.5 at the threshold,
ramping 0 -> 1 across it; (b) the ramp is sharpened (steepness 20 -> 30 in RH) so
a cell reaches full hard adjustment within ~0.15 RH above the threshold; (c) the
guarantee is stated precisely: cells in the narrow transition band are a smooth
handoff (NOT exactly on-curve, by construction), while strong super-saturation
(RH >> threshold — the guard's target) always gets the full on-curve adjustment.
A new test (`test_near_threshold_is_a_safe_smooth_handoff`) pins the contract:
monotone in RH, NEVER over-condensed in the band (RH_new >= 1), on-curve for
RH0 >= 1.25.  The cap test and a new extreme-q_v test
(`test_hard_adjustment_lands_on_curve_at_extreme_q_v`, q_v up to 1 kg/kg) pass.

Blend (in `saturation_adjustment`, q_sat = capped liquid saturation):
```python
    rel_humidity = q_v / jnp.maximum(q_sat, 1e-12)
    activation = jax.nn.sigmoid(30.0 * (rel_humidity - hard_threshold))
    condensation = condensation + activation * (cond_hard - condensation)
```

# UNCHANGED (round-2 false-positives, still hold)
Units/signs (liquid q_sat + L_v, caller maps rate to +L_v/c_pd*rate and -rate =>
c_pd*T+L_v*q_v conserved structurally), donor-clamp composition (morrison qv/qc
clamps + kessler q_v/dt clamp scale the rate consistently), flag-off byte
identity, doctrine (threshold tier-2 spec; sharpness/iter counts are module
constants; saturation reused not reimplemented; unsupported schemes raise).

# TEST RESULTS (JAX_ENABLE_X64=1, CPU) — all green
17/17 in test_hard_saturation_adjustment.py (enthalpy conservation rtol 1e-12;
lands-on-curve for Tibet + low-p (100-1000 Pa) + extreme-q_v (up to 1 kg/kg)
batteries; smooth over/under-condense contrast; near-threshold safe handoff;
flag-off byte-identity; grad finite+FD-match; jit/vmap; cap region; hot cell not
condensed; no-NaN; experiment-flag threading + raise; kessler scheme). fp32 path
verified (RH_new ~ 1.000003, flag-off byte-identical). No regressions: 3908 in
warm_rain/param_specs/no_inline_coeffs/no_saturation_reimpl/physics_contracts;
292 in physics_microphysics/kk2000/kessler+thompson faithful/reachability/
run_amip_cli.

# YOUR TASK (ROUND 3)
Confirm Bug 1 and Bug 2 are resolved. Scrutinise the implicit-gradient trick (is
the stop_gradient + single Newton correction the exact implicit gradient; any
NaN path in the jvp/psychrometric at the cap or low p; is `psychrometric` ever
<= 0), the bisection bracket validity (is f truly monotone and bracketed for all
physical inputs incl. sub-saturated and dry q_v=0), and anything else. If no
substantive findings remain, say so and justify. End with the OVERALL VERDICT
line.
