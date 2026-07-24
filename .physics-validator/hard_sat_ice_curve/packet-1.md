You are an independent ADVERSARIAL physics reviewer for LegoESM (JAX-native,
fully differentiable ESM). ROUND 1. Find EVERY bug, sign error, unit
inconsistency, broken-gradient pattern, conservation violation, and test-case
discrepancy in the change below. Cite line numbers / function names. If you
believe a candidate concern is NOT a bug, say so and explain why. Be CONCISE;
end with `OVERALL VERDICT: <no substantive findings | findings remain: ...>`.

# SCOPE: mixed-phase (ice-curve) hard-saturation drain — TTL dehydration fix

## Motivation
First ClimateEval scorecard bias chain: ta @100 hPa tropics **+17.8 K** warm;
TTL vapour **20x ERA5** (50.4 vs 2.5 mg/kg @100 hPa). ROOT CAUSE: the existing
opt-in hard-saturation DRAIN (MPAS driver post-step hook) gates AND lands on
the LIQUID saturation curve; at 195 K the liquid curve sits ~60% above the ice
curve, so the TTL legally holds permanent ice-supersaturation (self-reinforcing
wet -> warm). The change adds an opt-in `ice_curve=True` mode that gates + lands
on the w(T)-blended liquid/ice curve with a blended latent heat and routes cold
condensate to cloud ice.

## Thermo primitives (unchanged, for reference)
- `saturation_mixing_ratio(T,p)`  = Tetens (liquid), softplus-floored/capped.
- `saturation_mixing_ratio_ice(T,p)` = Clausius-Clapeyron over ice (L_s/R_v),
  same softplus floor/cap.
- constants: `L_v=2.501e6`, `L_s=2.834e6`, `c_pd=1004.64`, `T_freeze=273.15`,
  NEW `T_hom_freeze=233.15` (homogeneous freezing, ~-40 C; Pruppacher & Klett).

## The three new public helpers (`_warm_rain.py`)
```python
def mixed_phase_liquid_fraction(T):
    # linear w over [T_hom_freeze, T_freeze]; all-ice below -40C, all-liquid >0C
    return jnp.clip((T - constants.T_hom_freeze)
                    / (constants.T_freeze - constants.T_hom_freeze), 0.0, 1.0)

def mixed_phase_saturation_mixing_ratio(T, p_full):
    w = mixed_phase_liquid_fraction(T)
    return (w * saturation_mixing_ratio(T, p_full)
            + (1.0 - w) * saturation_mixing_ratio_ice(T, p_full))

def mixed_phase_l_over_cp(T):
    w = mixed_phase_liquid_fraction(T)
    return (w * constants.L_v + (1.0 - w) * constants.L_s) / constants.c_pd
```

## The solve (`_hard_saturation_condensation`) — key edit: injectable curve + L
Solves `q_eq = sat_fn(T + l_over_cp*(q_v - q_eq), p)` by 30-iter bracketed
bisection on `[q_sat(T), q_v]`, then re-attaches the EXACT implicit-function
gradient via one detached-root Newton correction (psychrometric
`1 + l_over_cp*dsat/dT`, held constant). Default `sat_fn=saturation_mixing_ratio`,
`l_over_cp=None -> L_v/c_pd` (byte-identical liquid path). Ice-curve caller
injects `sat_fn=mixed_phase_saturation_mixing_ratio` and a per-cell FROZEN-at-T0
`l_over_cp = mixed_phase_l_over_cp(T0)` array. Inside the solve, the blend weight
w is re-evaluated at `t_new` each iteration (through sat_fn) while `l_over_cp` is
frozen at T0.

## The rate limit (`_hard_saturation_rate_limit`) — FIXED THIS ROUND (my Finding G)
```python
def _hard_saturation_rate_limit(condensation, q_v, dt, hard_max_heating_K,
                                l_over_cp=None):
    if l_over_cp is None:
        dqv_cap = hard_max_heating_K * constants.c_pd / constants.L_v  # exact prior
    else:
        dqv_cap = hard_max_heating_K / l_over_cp        # per-cell L_eff(T0)/c_pd
    cond_cap = jnp.minimum(dqv_cap, jnp.maximum(q_v, 0.0)) / jnp.maximum(dt, 1e-10)
    return jnp.where(condensation > 0.0, jnp.minimum(condensation, cond_cap),
                     condensation)
```
Before the fix the cap used `L_v` unconditionally, so the ice branch (deposits
with L_s > L_v) heated **up to 13% past** the stated `hard_max_heating_K` cap
(measured 5.65 K vs 5.0 K at 234 K). Now the cap uses the SAME per-cell L the
caller heats with; default path is byte-identical (kept the exact prior
expression).

## The driver post-step (`_mpas_hard_saturation_poststep`)
Return arity 4->5 (`q_i_new` inserted). Under `ice_curve`: heats with the SAME
frozen `mixed_phase_l_over_cp(T0)` the solve used (so `c_pd*T + L_eff(T0)*q_v`
conserved exactly); splits condensate `w(T0)->q_c`, `(1-w)->q_i` when q_i tracer
exists (else all to q_c). `q_c is None -> true no-op`. Post-step hook is EAGER
(outside jit, no autodiff). Morrison carries `q_i` on the MPAS lane (verified).

## FULL DIFFS
See attached (in this repo):
`.physics-validator/hard_sat_ice_curve/warm_rain.diff`
`.physics-validator/hard_sat_ice_curve/model_driver.diff`
Config: `ExperimentConfig.hard_sat_ice_curve: bool=False`; validate_strict
refuses `True` without `hard_saturation_adjustment`. CLI `--hard-sat-ice-curve`.

# MY STATIC + SIGN ANALYSIS
- **Convention**: condensation/deposition dq>0 => q_v-=dq, q_c/q_i+=dq, T+=L/cp*dq
  (heating). Drain is pure (q_eq clipped to [0,q_v] => rate>=0), no spurious
  evaporation of sub-saturated cells.
- **Bracket monotonicity** with the blend: d/dT[w q_L + (1-w) q_I] = w q_L' +
  (1-w) q_I' + w'(q_L - q_I) > 0 (all three terms >0 below freezing since
  q_L>q_I and w'>0). Residual df/dq_eq = 1 + l_over_cp*sat'(t_new) > 1 > 0 with
  frozen l_over_cp>0. => bracket valid.
- **Enthalpy**: solve + heating both use frozen L_eff(T0); exact discrete
  conservation of c_pd*T + L_eff(T0)*q_v. Landing on blend(t_new) (w at NEW T).
- **AD**: q_eq detached; implicit correction re-attaches (dsat/dtheta)/psychro;
  l_over_cp(T) dependence flows through t_new in q_sat_new, so d/dT includes the
  l_over_cp'(T) term (verified exact vs FD, below).

# NUMERICAL EVIDENCE (JAX_ENABLE_X64=1, node l40048)
All PASS after the rate-limit fix:
- A blend d(sat)/dT>0 over 190-295K (min 1.15e-7); monotone by FD.
- B residual strictly increasing in q_eq (frozen lcp, T-varying w); f(lo)<=0<=f(hi).
- C ice_curve=False == explicit liquid path BIT-IDENTICAL (max|diff|=0); warm-cell
  ON==OFF bit-identical.
- D fp32 vs fp64 ice_curve parity rel 1.5e-7.
- E grad vs centered-FD, ice_curve=True: TTL 195K & mixed 250K, d/dqv d/dT d/dp
  all rel <= 1.3e-6; grads finite.
- F poststep total-water conserved (max|dW|=0); enthalpy c_pd*T+L_eff(T0)*qv
  conserved (max|dH|=2.8e-14); all-cold routes to q_i, q_c untouched.
- G heating cap: NOW exactly 5.000 K at 250K & 234K (was 5.39/5.65 K).
- H 250K condensate split q_c=w*dq, q_i=(1-w)*dq (w=0.421).
- I lands ON/above warmed blended curve (resid 8e-18, pure drain).
- J RHi<=1.1 -> 0 drain; RHi=1.20 -> drains.

Tests: `tests/unit/test_hard_sat_ice_curve.py` (16) +
`tests/unit/test_hard_saturation_adjustment.py` (24) = **40 passed**.
(`test_run_amip_cli.py` has 2 FAILS — `mcfarlane` vs `mcfarlane+hines` gwd
default — CONFIRMED pre-existing at clean HEAD 621781aa0, independent of this
change.)

# POINTS I WANT YOU TO SCRUTINIZE (candidate concerns I judged NOT bugs)
1. `thermo.saturation_mixing_ratio_blend(T,p, T_blend_top=T_freeze,
   T_blend_width=20)` already exists (FV3 LES, 20K ramp [253,273]). The new
   `mixed_phase_saturation_mixing_ratio` re-derives the blend with a WIDER 40K
   ramp [233,273]. I kept it separate because (a) the physical ramp differs
   (thermodynamic vs FV3 parity), (b) I need a single `mixed_phase_liquid_fraction`
   feeding sat+L+condensate-routing consistently, which the existing helper
   doesn't expose. Duplication? Over-engineering? Or acceptable?
2. Threshold 1.1 ON the ice curve at TTL = RHi 110% trigger. Krämer 2009: cirrus
   RHi persists to 120-150%. BUT the drain LANDS on the ice curve (RHi ~100%,
   the standard GCM saturation-adjust-to-ice target), and only FIRES above
   RHi 110%; it's opt-in + rate-limited. Over-drying the TTL, or fine? (I lean:
   fine; threshold is a config knob; erring drier is the intended direction to
   fix a 20x-too-wet bias.)
3. In-scheme smooth path (`saturation_adjustment`/`_hard_saturation_blend`)
   stays liquid-only; only the post-step drain gets the ice curve. Consistency
   issue, or acceptable (post-step runs every step on the final state and
   dominates on the MPAS lane)?
4. Frozen-at-T0 l_over_cp vs t_new-varying w inside the solve: is the residual
   still strictly monotone (I argue yes, only needs l_over_cp>0)? Is the enthalpy
   landing contract still exact (drain>=0)?
5. q_i missing-key fallback to q_c (non-Morrison schemes): acceptable documented
   approximation, or a silent-conservation trap?

Attack anything else: dtype promotion in the fixed rate-limit path, the
`.astype(q_v.dtype)` on l_over_cp, stop_gradient placement, the clip kinks in
w(T) at 233/273 K, sub-freezing above-ramp condensate routing, the log-line
formatting, validate_strict corner cases.
