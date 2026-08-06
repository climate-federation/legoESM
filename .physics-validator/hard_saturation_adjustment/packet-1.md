You are an independent ADVERSARIAL physics reviewer for LegoESM, a JAX-native,
fully-differentiable Earth System Model. Find EVERY bug, sign error, unit
inconsistency, broken-gradient pattern, conservation violation, and test-case
discrepancy in the implementation below. Cite line numbers. If you believe there
are no bugs, say so and explain why each candidate concern is not one.

Be concrete and skeptical. Treat every `clip`, `where`, `maximum`, and mask as
suspect for differentiability. Conservation and sign conventions are first-class.

The full repo is available at the working root (read-only). Key files:
- packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py
- packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py
- packages/atmosphere/legoesm/atmosphere/physics/microphysics/{morrison,kessler,seifert_beheng,thompson,p3}.py
- packages/core/legoesm/thermo.py  (saturation_mixing_ratio: Tetens e_sat, softplus denom floor, smooth cap at 1 kg/kg)
- packages/core/legoesm/constants.py (L_v=2.501e6, c_pd=1004.64, R_v=461.51, epsilon=0.622)
- tests/unit/test_hard_saturation_adjustment.py

# GOAL / CONTEXT

Moist MPAS AMIP runs die from a local vapour-accumulation runaway: 65-70 g/kg
single-cell pools that condense ~150 K of latent heat and detonate a vertical
2Δσ checkerboard. The existing smooth *sigmoid* `saturation_adjustment` cannot
control these pools: (a) at hot cells (T~400 K) q_sat smooth-caps at 1 kg/kg so
q_v-q_sat < 0 and it does not fire; (b) at cold cells it evaluates the
psychrometric factor at the ORIGINAL cold T (where dq_sat/dT~0) and removes the
whole excess in one step, OVERSHOOTING far past saturation (releasing full
latent heat). We add an opt-in HARD (Newton-iterated) saturation adjustment that
lands q_v exactly ON the liquid saturation curve, conserving c_pd*T + L_v*q_v.

Requirements: implemented INSIDE the microphysics tendency; opt-in
`hard_saturation_adjustment: bool = False` + `hard_sat_adjust_threshold: float =
1.1` (RH trigger, with __param_spec__); byte-identical when off; smooth/AD-safe;
pure pytree (jit/vmap safe); uses the TRUE p_full; condensed water goes to the
scheme cloud-water tendency; must drain the capped-q_sat region (vapour above
capped q_sat*threshold).

# HOW CONSERVATION WORKS (the caller contract)

Every warm-rain caller maps the returned `condensation` rate as:
  dq_v_dt += -condensation ; dq_c_dt += +condensation ; dT_dt += +L_v*condensation/c_pd
So c_pd*T + L_v*q_v is conserved by construction for ANY condensation value.
The hard adjustment only changes the MAGNITUDE of `condensation`.

# CORE IMPLEMENTATION (_warm_rain.py)

Module constants:
```python
_DEFAULT_HARD_SAT_THRESHOLD = 1.1     # RH trigger: fire where q_v > thr*q_sat [-]
_HARD_SAT_NEWTON_ITERS = 8            # fixed Newton-step count (loop count)
_HARD_SAT_MASK_SHARPNESS = 20.0       # smooth activation ramp width [1/RH]
_HARD_SAT_QSAT_FLOOR = 1.0e-12        # q_sat floor in the RH trigger ratio only
```

Newton helper:
```python
def _hard_saturation_condensation(T, q_v, p_full, dt, q_sat):
    l_over_cp = constants.L_v / constants.c_pd
    q_v_pos = jnp.maximum(q_v, 0.0)
    q_eq = jnp.clip(q_sat, 0.0, q_v_pos)             # init below the root
    for _ in range(_HARD_SAT_NEWTON_ITERS):
        t_new = T + l_over_cp * (q_v_pos - q_eq)
        q_sat_new = saturation_mixing_ratio(t_new, p_full)   # LIQUID, capped
        dqsdt = constants.L_v * q_sat_new / (constants.R_v * t_new ** 2)
        psychrometric = 1.0 + dqsdt * l_over_cp
        q_eq = q_eq - (q_eq - q_sat_new) / psychrometric
        q_eq = jnp.clip(q_eq, 0.0, q_v_pos)
    return (q_v_pos - q_eq) / dt
```
Design claims to verify/refute:
- Root of f(q_eq)=q_eq - q_sat(T+(L/cp)(q_v-q_eq)) is the enthalpy-conserving
  moist-adjustment equilibrium. f is concave (q_sat convex in T) and Newton with
  the CC-approx (always-positive) psychrometric Jacobian is a MONOTONE climb from
  q_eq0=q_sat(T) up to the root — no overshoot into sub-saturation.
- clip(q_eq,0,q_v) every iter: (i) bounds t_new to [T, T+(L/cp)q_v] so the solve
  is finite even where q_sat caps (T->~400K, e_sat>p); (ii) makes the rate a pure
  DRAIN >=0 (sub-saturated column pins q_eq=q_v -> rate 0, evaporation left to the
  smooth branch).
- Exact derivative of the CAPPED curve would give psychro~1 inside the cap and
  Newton would then overshoot/oscillate (clip+cap 2-cycle) — that is WHY the
  CC-approx Jacobian (large in the cap => small damped step) is used deliberately.
- 8 iters: worst case is a moderate-cold cell (T~230-290 K) whose trial-T transits
  the cap region; verified residual <= 6e-7 kg/kg for the battery.

Blend inside saturation_adjustment (after the smooth `condensation` is formed):
```python
    if hard_adjust:
        if jax.config.jax_enable_x64:
            cond_hard = _hard_saturation_condensation(
                T.astype(f64), q_v.astype(f64), p_full.astype(f64), dt,
                q_sat.astype(f64)).astype(q_v.dtype)
        else:
            cond_hard = _hard_saturation_condensation(T, q_v, p_full, dt, q_sat)
        rel_humidity = q_v / jnp.maximum(q_sat, _HARD_SAT_QSAT_FLOOR)
        activation = jax.nn.sigmoid(
            _HARD_SAT_MASK_SHARPNESS * (rel_humidity - hard_threshold))
        condensation = condensation + activation * (cond_hard - condensation)
    if q_c is not None:                       # existing evaporation donor clamp
        q_c_avail = jnp.clip(q_c, 0.0, None)
        condensation = jnp.maximum(condensation, -q_c_avail / jnp.maximum(dt, 1e-10))
    return condensation, q_sat
```
`q_sat` here is the (capped) LIQUID saturation the function already computed
(fp64-then-cast when x64 is on). `hard_adjust=False` skips the whole block, so
the result is byte-identical to before.

# SIGNATURE + CALLERS

`saturation_adjustment(T, q_v, p_full, dt, sharpness=50.0, q_c=None,
 hard_adjust=False, hard_threshold=1.1)`. New kwargs appended (back-compatible).
All five warm-rain schemes (morrison/kessler/seifert_beheng/thompson/p3) now pass
`hard_adjust=config.hard_saturation_adjustment,
 hard_threshold=config.hard_sat_adjust_threshold`. Kessler additionally clamps the
returned condensation by q_v/dt afterwards (`minimum(condensation, q_v/dt)`).

# CONFIG / PLUMBING

- Added `hard_saturation_adjustment: bool = False` + `hard_sat_adjust_threshold:
  float = 1.1` to KesslerConfig, SeifertBehengConfig, MorrisonConfig,
  ThompsonConfig, P3Config (appended at the END of each NamedTuple).
- Added `hard_sat_adjust_threshold` __param_spec__ entry per config: units "1",
  bounds (1.0, 2.0), tier 2, transform sigmoid, category condensation.
- `apply_microphysics_experiment_flags(..., hard_saturation_adjustment=False)`
  threads the bool onto the scheme config, RAISING for a scheme lacking the field
  (sundqvist/ml/sdm/fast_sbm) — dispatch hardening.
- ExperimentConfig gained `hard_saturation_adjustment: bool = False`; threaded in
  model_driver._run_mpas (MPAS path) and physics_pipeline._resolve_microphysics
  (coupled path).
- run_amip: `--hard-saturation-adjustment` (BooleanOptionalAction, default off),
  wired into build_config_from_args; threshold reachable via `--params`
  (per-scheme spec float; baselined in _params_reachability_baseline.py like all
  sibling atm-micro params).

# STATIC ANALYSIS SUMMARY

- Units: threshold dimensionless (RH); rate kg/kg/s; T K; p Pa; q kg/kg. q_sat is
  the LIQUID capped curve; L_v (not L_s) used — consistent with the caller's L_v
  mapping and the warm-rain condensation path (ice handled separately by
  deposition).
- Signs: condensation >= 0 in the active region (drain); dq_v = -cond (loss),
  dT = +L_v*cond/c_pd (warming), dq_c = +cond (gain). Sub-saturated -> rate 0
  (no evaporation via the hard branch). Hot capped cell (q_v < thr*q_sat_cap) ->
  activation ~ 0 and cond_hard clipped to 0 -> no spurious condensation.
- JAX purity: pure function; unrolled static-count Python loop (jit-safe); no
  in-place mutation, no host sync, no python control flow on traced values;
  `jax.config.jax_enable_x64` is a STATIC compile-time flag (matches the existing
  x64 branch). vmap/jit verified against eager.
- Conservation: structural (see caller contract). Verified to rtol 1e-12.
- Limiters: two clips — `clip(q_eq,0,q_v)` (dead-gradient only in the pathological
  cap region where the feature is inert, w~0) and `maximum(q_sat,1e-12)` (floor in
  the RH ratio only; Newton uses the true q_sat). t_new>0 and q_sat_floor>0 so no
  div-by-zero; T-clip inside saturation_vapor_pressure (>=150 K) bounds e_sat.

# TEST RESULTS (JAX_ENABLE_X64=1, CPU)

tests/unit/test_hard_saturation_adjustment.py — 14 passed:
- conserves moist enthalpy c_pd*T+L_v*q_v (rtol 1e-12) across the battery;
- lands on the saturation curve (RH_new ~ 1, rtol 3e-3) for the Tibet cold cells
  (116-290 K, q_v=66 g/kg); coldest drains ~93%, 200 K drains ~58%;
- cold strong super-sat: smooth OVERSHOOTS (RH<0.5), hard lands on curve, hard
  condenses LESS; warm mild super-sat: smooth UNDER-condenses (RH>1), hard lands
  on curve, hard condenses MORE;
- flag-off byte-identity; grad finite+nonzero; grad matches centered FD (rtol
  1e-4); jit & vmap match eager; cap region (q_v=1.3 kg/kg @400 K -> drains to
  ~cap 1.0); hot sub-saturated cell NOT condensed; no NaN on extreme+dry inputs;
- experiment-flag threading + raise for sundqvist; kessler scheme drains + off is
  bit-identical + both paths conserve enthalpy.

Also green (no regressions, flag off = byte-identical): test_warm_rain.py,
test_warm_rain_fp32_saturation.py, test_param_specs.py (467 items),
test_no_inline_physics_coeffs.py, test_dispatch_hardening.py,
test_validate_strict_coverage.py, test_physics_contracts.py,
test_params_reachability_audit.py, test_physics_microphysics.py,
test_kk2000_warm_rain.py, test_seifert_beheng_faithful.py, kessler/thompson
faithful, morrison_p3_comparison, aerosol_ccn_combined, advertised_buildability,
run_amip_cli.

# YOUR TASK

Adversarially review the physics and code. In particular scrutinize:
1. Does the Newton iteration truly converge (monotone, no overshoot) for ALL
   physical (T,p,q_v) — incl. the cap-region trial temperatures — or can it
   land subsaturated / oscillate / NaN? Is 8 iters enough? Is initialising at
   clip(q_sat,0,q_v) always below the root?
2. Sign/units: is using LIQUID q_sat + L_v (not ice/L_s) correct here given the
   caller mapping and the warm-rain split? Any hidden sign inversion?
3. Differentiability: any dead-gradient region that SHOULD carry sensitivity?
   Any NaN-gradient path (div by t_new, psychro, q_sat_floor; the clips)?
4. Conservation: is c_pd*T+L_v*q_v really conserved once the hard rate flows
   through the downstream donor clamps (morrison qv/qc clamps) and kessler's
   extra q_v/dt clamp? Any double-count or leak?
5. Composition: is the smooth->hard blend continuous and free of the naive-cap
   discontinuity (F3)? Does the mask correctly fire against the CAPPED q_sat so
   the cap region is handled (F2)? Any regime where blend is wrong?
6. Byte-identity when off; any way the flag=False path differs from before?
7. Any repo-doctrine violation (hardcoded constants/sharpness, saturation
   re-impl, dispatch that should raise, param-spec classification).

List findings as confirmed-bug / false-positive / ambiguous with line-cited
reasoning. If none, justify why each candidate concern is not a bug.
