# The hard saturation-adjustment default: evidence, not a decision

`hard_saturation_adjustment` defaults to **False** on all seven guarded
microphysics schemes. This note records what was MEASURED about that default so
the question can be decided on evidence rather than re-litigated from memory.
**Nothing here changes the default** — that is a physics decision affecting
every existing run and every tuned parameter set.

## Protocol

Single column, 4 levels, dt = 300 s, T0 = 300 K, p = 90000 Pa,
q_c = q_r = q_i = q_s = q_g = 0, q_v = 1.4 · q_sat, `JAX_ENABLE_X64=1`, CPU.
RH is recomputed each step from `legoesm.thermo.saturation_mixing_ratio` at the
current (T, q_v). The two arms differ ONLY in the flag.
Source: `scripts/tmp/_probe_supersat_guard.py`, SLURM job 9331634.

| scheme | RH@1 off | RH@2 off | RH@5 off | RH@20 off | RH@1 on |
|---|---|---|---|---|---|
| kessler / seifert_beheng / morrison / thompson / p3 | 1.0589 | 1.0250 | 1.0026 | 1.0000 | 1.0000 |
| sundqvist | 0.2255 | 0.2256 | 0.2256 | 0.2257 | 1.0000 |
| ml_emulator (untrained) | 1.2459 | 1.1108 | 0.7949 | 0.1604 | 1.0000 |

## What the numbers actually say

**CONFIRMED.** In a CLOSED box the five bulk schemes' default smooth branch does
*not* leave the column permanently super-saturated: it relaxes geometrically and
is within 0.3 % of saturation by step 5. The defect it has is that it is SLOW
(≈85 % of the excess removed in the first step, then a halving per step as the
`sigmoid(sharpness · excess)` factor collapses toward 0.5 for a small excess).
With a continuing moisture source — dynamical convergence, i.e. the actual model
— a slow relaxation supports a *standing* super-saturation. The guard lands the
column on the curve in one step.

**CONFIRMED.** Sundqvist's default is a different, worse failure: its sigmoid
gates on RH (an O(1) argument that saturates), so it removes the entire excess
toward `q_sat(T_old)` in one step while ignoring the latent heating that raises
`q_sat`. From RH 1.4 it OVERSHOOTS to RH 0.23 with a ~20 K one-step heating
spike, and by step 40 it has rained out 28 % of the column water versus 5 % with
the guard on. For this scheme the guard is a clear physical improvement.

**RETRACTION / correction of the motivating premise.** The task framing was that
the default "under-drains by design and supersaturated vapour never condenses
out", explaining an 80-day persistent CWV drift. The box measurement does not
support the strong form of that claim at warm low-level conditions — the default
does drain, just slowly. That the default *alone* caused the observed persistent
drift is **PLAUSIBLE, not CONFIRMED**, and should be instrumented in the failing
run (log the per-step condensation and the supersaturation source separately)
before it is used to justify a global default flip.

## For flipping the default to True

* One-step convergence onto the saturation curve instead of a multi-step
  relaxation; removes the standing super-saturation a continuing source
  sustains.
* `hard_sat_max_heating_K = 5.0` already bounds the risk: a large pool drains
  over many steps (≈2 g/kg per step) rather than detonating. The failure mode a
  naive post-step cap caused is already designed out.
* The adjustment is enthalpy-conserving by construction, verified numerically
  (c_pd·T + L_v·q_v residual < 1e-12 relative, Kessler, step 1).
* For Sundqvist specifically the default is arguably a defect, not a choice.

## Against flipping the default to True

* It changes answers on **every** existing run. Byte-identity of the OFF path is
  precisely what protects every tuned parameter set, scorecard, and controlled
  comparison in the repo; a silent flip invalidates comparisons against every
  prior number.
* The in-scheme guard lands on the **liquid** saturation curve
  (`ice_curve=False`). At TTL temperatures the liquid curve sits well above the
  ice curve, so a global default-on would leave permanent ice-supersaturation in
  the upper troposphere — the known 20×-ERA5 TTL vapour bias. The mitigation
  (`hard_sat_ice_curve`) is currently gated to `grid_type='mpas'` **and**
  `microphysics='morrison'`, so it is unavailable on most lanes.
* For Sundqvist, default-on would silently change what "sundqvist" means
  relative to the published SBK89 formulation.
* The causal attribution above is not yet CONFIRMED.

## Recommendation

**Do not flip the global default.** Instead:

1. Turn it on where the failure mode is real, via run configs/presets — the
   RCE/CRM lanes that can start from a super-saturated sounding — not via the
   code default.
2. Before any global flip, run a controlled pair (identical forcing, grid,
   window, metric; the flag as the ONLY variable) and check the TTL vapour bias
   specifically, since the liquid-only landing point is the concrete regression
   risk.
3. If a global flip is later wanted, land it together with a
   temperature-appropriate saturation curve (ice/mixed-phase) so the cold branch
   does not get worse while the warm branch gets better.
