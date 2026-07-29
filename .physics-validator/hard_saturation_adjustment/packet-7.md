You are an independent ADVERSARIAL physics reviewer for LegoESM. ROUND 7 (final
confirmation of the round-6 fixes for the MPAS driver post-step hook). Be
CONCISE; end with `OVERALL VERDICT: <no substantive findings | findings remain: ...>`.

# ROUND-6 findings and the fixes

## Finding 1 (q_c=None loses total water) -- FIXED
`_mpas_hard_saturation_poststep` now NO-OPS when ``q_c is None`` (no reservoir
for the condensate -> draining vapour would lose water), returning inputs
unchanged with dq=0; and the `_run_mpas` hook now gates on BOTH ``"q_v" in _trc
and "q_c" in _trc`` (the moist warm-rain path always carries both).  So total
water is conserved on every path.

## Finding 2 (smooth activation leaks below the 1.1 threshold) -- FIXED
The POST-STEP `hard_saturation_drain` now uses a HARD RH gate
``jnp.where(q_v > hard_threshold*q_sat, cond_hard, 0.0)`` instead of the
in-scheme smooth sigmoid.  This matches the VALIDATED post-step cap exactly (no
sub-threshold leakage: RH=1.05 -> exactly 0), leaving mild sub-threshold
super-saturation for the next microphysics call.  A hard gate is appropriate
because the post-step hook is EAGER (outside jit / no autodiff), so the smooth
ramp the in-scheme AD path needs is unnecessary; ``cond_hard`` is the on-curve
drain so the gated rate never overshoots.  The IN-SCHEME path
(`saturation_adjustment`/`_hard_saturation_blend`) still uses the smooth sigmoid
(unchanged, all 21 in-scheme tests pass).

Factoring: the reviewed rate limit is now a shared `_hard_saturation_rate_limit`
used by BOTH `_hard_saturation_blend` (in-scheme) and `hard_saturation_drain`
(post-step).  The post-step reuses the reviewed solver
`_hard_saturation_condensation` + this rate limit EXACTLY; only the GATE differs
(hard vs smooth).

```python
def hard_saturation_drain(T, q_v, p_full, dt, hard_threshold=1.1, hard_max_heating_K=5.0):
    # q_sat, cond_hard solved (fp64 when available)
    drain = jnp.where(q_v > hard_threshold * jnp.maximum(q_sat, 1e-12), cond_hard, 0.0)
    return _hard_saturation_rate_limit(drain, q_v, dt, hard_max_heating_K)
```

## Finding 3 (false --params tuning claim) -- FIXED
The CLI help + ExperimentConfig doc no longer claim `--params` tuning.  They now
state the threshold + cap use the scheme-config defaults (1.1, 5 K -- MATCHING
the validated configuration), and that like all atmosphere microphysics params
they are calibratable via the SCM-RCE training path, NOT the run_amip/run_coupled
`--params` loader (atm micro configs are not routable onto ExperimentConfig --
the reachability audit baselines the whole family).  The MPAS post-step reads
the (default) threshold/cap from the scheme config.

# TESTS (JAX_ENABLE_X64=1, CPU) -- green
23/23 in test_hard_saturation_adjustment.py, including the updated:
- test_hard_saturation_drain_is_pure_gated_drain: adds a mild RH~1.05 cell and
  asserts dq == 0 there (HARD gate, no leak); sub/hot-subsat untouched;
  super-saturated drains, rate-limited; conserves c_pd*T + L_v*q_v.
- test_mpas_poststep_hook_conserves_and_rate_limits: conservation of
  c_pd*T + L_v*q_v AND total water; per-step ΔT <= 5 K; pure drain; q_c gains
  exactly what q_v loses; sub-saturated column untouched; and the None-q_c path
  is a conserving NO-OP (q_v, T unchanged, dq=0).
The 21 in-scheme tests still pass (smooth path unchanged).

# YOUR TASK (ROUND 7)
Confirm the three fixes are complete and correct: (a) is total water now
conserved on ALL paths incl. q_c=None (no-op) and q_c-present (donor
q_v->q_c)?  (b) does the HARD gate eliminate ALL sub-threshold drain while still
draining the transport spike, and is it safe/eager-correct (no autodiff/jit
through the post-step)?  (c) is the reuse of the reviewed solver + shared
rate-limit exact (no new math), and does the in-scheme smooth path remain
byte-unchanged?  (d) any remaining conservation/sign/units/eager-safety issue, or
anything else?  If no substantive findings, say so and justify.  End with the
OVERALL VERDICT line.
