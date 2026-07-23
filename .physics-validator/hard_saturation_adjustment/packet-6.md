You are an independent ADVERSARIAL physics reviewer for LegoESM (JAX-native ESM).
ROUND 6 -- a NEW delta: a DRIVER-LEVEL POST-STEP application of the (already
reviewed, rounds 1-5 clean) hard saturation adjustment on the MPAS path. Be
CONCISE; verdict per concern; end with `OVERALL VERDICT: <no substantive
findings | findings remain: ...>`.

# WHY (integration-trial evidence)
The in-scheme hard adjustment (reviewed, rounds 1-5) BLEW UP a moist MPAS AMIP
run at day 24 -- the exact grave of the unfixed config. The empirical POST-STEP
cap at the same threshold + per-step rate has 40/40 runs + 2 multi-month pilots.
PLACEMENT is load-bearing: the working intervention fires post-step on the FINAL
state (after the dycore's per-step vertical vapour transport), while the
in-scheme version leaves that step's transport spike uncorrected until the next
physics call, and at dt=100 s the transport+latent feedback detonates within the
window.

# WHAT CHANGED (delta only)
1. Factored the reviewed hard-adjustment core into ONE shared helper
   `_warm_rain._hard_saturation_blend(condensation, T, q_v, p_full, dt, q_sat,
   hard_threshold, hard_max_heating_K)` (bracketed-bisection on-curve solve +
   smooth->hard activation + on-curve overshoot cap + per-step heating & vapour
   rate limit).  `saturation_adjustment` now calls it (behaviour byte-unchanged;
   all 21 prior tests still pass).
2. New PUBLIC entry `_warm_rain.hard_saturation_drain(T, q_v, p_full, dt,
   hard_threshold, hard_max_heating_K)` = `_hard_saturation_blend` with a ZERO
   smooth baseline -> a PURE activation-gated DRAIN (>= 0, no smooth
   condensation/evaporation baseline).  Same reviewed core, no new math.
3. `model_driver._mpas_hard_saturation_poststep(T, q_v, q_c, p_s, sigma_full,
   dt, hard_threshold, hard_max_heating_K)` (module-level, testable): builds
   pure-sigma `p_full = p_s * sigma_full`, calls `hard_saturation_drain`, and
   applies the conserving bookkeeping `dq=rate*dt; q_v-=dq; q_c+=dq;
   T+=(L_v/c_pd)*dq`.  Returns (T_new, q_v_new, q_c_new, dq).
4. `model_driver._run_mpas`: the hard flag is NO LONGER threaded in-scheme on the
   MPAS path (in-scheme SKIPPED -> no double application); a fail-loud guard
   raises for a non-warm-rain scheme with the flag; the post-step hook runs
   immediately after the existing sponge multiply (same site as the QCAPv2
   reference), gated on the flag, with a loud counter at low cadence
   (`step % 432`, a module const).  The coupled/spectral path (physics_pipeline)
   STILL applies it in-scheme (unchanged).

```python
# model_driver.py, module level:
def _mpas_hard_saturation_poststep(T, q_v, q_c, p_s, sigma_full, dt,
                                   hard_threshold, hard_max_heating_K):
    from legoesm.atmosphere.physics.microphysics._warm_rain import hard_saturation_drain
    p_full = p_s[:, None] * jnp.asarray(sigma_full)[None, :]
    rate = hard_saturation_drain(T, q_v, p_full, dt, hard_threshold, hard_max_heating_K)
    dq = rate * dt
    T_new = T + (constants.L_v / constants.c_pd) * dq
    q_v_new = q_v - dq
    q_c_new = None if q_c is None else q_c + dq
    return T_new, q_v_new, q_c_new, dq

# _run_mpas step loop, right after the sponge multiply:
_trc = self.state.tracers
if _hard_sat_on and _trc is not None and "q_v" in _trc:
    _qc_fld = _trc.get("q_c")
    _T_hs, _qv_hs, _qc_hs, _dq_hs = _mpas_hard_saturation_poststep(
        self.state.T.data, _trc["q_v"].data,
        None if _qc_fld is None else _qc_fld.data,
        self.state.p_s.data, self.sigma.sigma_full, DT,
        _hard_sat_threshold, _hard_sat_max_heating)
    if (step % _HARD_SAT_LOG_CADENCE_STEPS) == 0:
        _n_hs = int(jnp.sum(_dq_hs > _HARD_SAT_LOG_QV_EPS))
        if _n_hs > 0: logger.warning(... _n_hs, max dq_v, cap, step ...)
    _new_trc = dict(_trc); _new_trc["q_v"] = _trc["q_v"].replace(data=_qv_hs)
    if _qc_fld is not None: _new_trc["q_c"] = _qc_fld.replace(data=_qc_hs)
    self.state = self.state._replace(T=self.state.T.replace(data=_T_hs), tracers=_new_trc)
```

# DESIGN CHOICES (justify or refute)
- PURE DRAIN (smooth baseline = 0): the post-step must NOT evaporate
  sub-saturated cells nor re-condense mild super-saturation the in-step
  microphysics already handled -- only drain the transport spike (RH > threshold)
  like the validated cap.  Verified: dq >= 0 everywhere; dq = 0 at sub-saturated
  and hot-capped cells.
- IN-SCHEME SKIPPED ON MPAS (not threaded), post-step applied -> no double
  application.  Coupled path keeps in-scheme (its dycore has no MPAS transport
  spike).  The single flag thus means "in-scheme on coupled, post-step on MPAS";
  documented on the ExperimentConfig field + the CLI help.
- PURE-SIGMA p_full = p_s * sigma_full: the SAME convention the driver's own
  diagnostics use; this MPAS path is pure sigma (noted in code).
- Threshold + per-step heating cap read from the (unmodified) scheme config
  (--params tunable); fail-loud for a non-warm-rain scheme.
- Eager (outside jit): the `int(jnp.sum(...))` host sync is at low cadence only.

# TESTS (JAX_ENABLE_X64=1, CPU) -- green
23/23 in test_hard_saturation_adjustment.py, incl.:
- test_hard_saturation_drain_is_pure_gated_drain: dq >= 0; sub-saturated/hot
  cells untouched; super-saturated drain rate-limited to 5 K -> ~2 g/kg;
  conserves c_pd*T + L_v*q_v.
- test_mpas_poststep_hook_conserves_and_rate_limits: fabricated 2x5 column (a
  cold pool + a sub-saturated column) -> conservation of c_pd*T + L_v*q_v AND
  total water; per-step ΔT <= 5 K; pure drain; q_c gains exactly what q_v loses;
  sub-saturated column untouched; None-q_c path handled.
The 21 pre-existing in-scheme tests still pass (factoring byte-unchanged).  No
regressions: 3815 passed across no_inline_coeffs / no_saturation_reimpl /
dispatch_hardening / param_specs / reachability / warm_rain.

# YOUR TASK (ROUND 6)
(a) Is the post-step placement + PURE-drain reuse correct -- does it conserve
c_pd*T + L_v*q_v and total water exactly, never evaporate, and never overshoot?
(b) Is skipping in-scheme on MPAS while keeping post-step the right way to avoid
double application; any path where BOTH fire on MPAS?  (c) Is reading the
threshold/cap from the scheme config + the fail-loud non-warm-rain guard correct?
(d) Eager-loop safety (host syncs, retrace, Field bookkeeping via
`.replace(data=...)`, the None-q_c branch)?  (e) Any conservation/units/sign
issue with pure-sigma p_full, or anything else?  If no substantive findings,
say so and justify.  End with the OVERALL VERDICT line.
