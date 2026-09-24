# CLAIM REVIEW (before any code): Zhang-McFarlane on the spectral training lane

Repository: legoESM (differentiable ESM in JAX). Worktree: this directory, branch wb/cam6-baseline (a0038f279).
You are reviewing a CLAIM that will justify a code change. No code has been written. Be adversarial: attack the
premise, the mechanism, the conservation argument, the scope, and name anything that must be MEASURED first.

## Situation
The owner wants the WeatherBench parameter-training lane (spectral T63/L32 sigma, `model_type="spectral_pe"`,
`packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py`, spectral bridge `physics_fn` at ~L1444)
to run the production AMIP CAM6 physics UNCHANGED (ZM deep convection, CLUBB, cam6_clubb macrophysics, Morrison,
RRTMGP). Absolute owner constraint: no scheme substitution.

First compile of the CAM6 WB deck died at:
```
ValueError: convection scheme 'zhang_mcfarlane' emits a signed NET rain-flux divergence (dq_r_conv_dt) that must be
column-integrated to surface precipitation; the spectral bridge has no surface-precip sink and booking it per layer
into a tracer would create condensate sinks in layers holding none. Use the hydrostatic bridge or the unified PhysicsPipeline.
```
(integration.py ~L1785-1794; trait `rain_is_net_flux=scheme_name in ("zhang_mcfarlane",)` at L214.)

## What the working paths do (READ OFF THE CODE, please re-verify)
- ZM contract (`zhang_mcfarlane.py` L29-33, L94-96): `dq_r_conv_dt` is CAM `ntprprd` = prdprec - evpprec, signed
  per layer; "column-integrates to the surface convective precipitation (kg/m^2/s = -sum dp (dq_v_dt + dq_c_conv_dt)/g)".
- Hydrostatic bridge (integration.py ~L805-870): `_to_sfc = rain_to_surface or _tr.rain_is_net_flux`; when `_to_sfc`
  the rain is NOT booked into q_r/q_c; the column sum (clipped at 0 for the signed field) is emitted as
  `precip=Field(precip_conv)` on the tendency and as the `conv_precip` lag-carry diagnostic.
- Unified PhysicsPipeline (`packages/coupler/legoesm/driver/physics_pipeline.py` ~L1791-1797): column-integrates
  `dq_r_conv_dt*dp/g`, clips at 0, adds to surface precip. This is what the production AMIP run does.
- Spectral bridge (~L1805-1870): books dq_v -> q_v, dq_c -> q_c, and dq_r -> q_r (or folds into q_c). Its tendency
  type `SpectralHydrostaticState` (dynamics/gcm/spectral_pe.py L119) has NO precip slot; the spectral lane emits
  no surface precipitation anywhere (the combined accumulator's `getattr(t, "precip", None)` is always None there).
  Comment at L783-785: "the implied precip simply leaves the prescribed surface".
- Nothing on the spectral lane reads the `PhysicsState.conv_precip` carry (grep of radiation/integration.py: none).

## THE CLAIM
C1 (mechanism): On the spectral bridge, the correct treatment of a `rain_is_net_flux` scheme is the hydrostatic
bridge's surface route: do NOT book `dq_r_conv_dt` into any tracer; book dq_v and dq_c exactly as today. The
rain leaves the column, which is what every other precipitating species already does on this lane.
C2 (conservation): Column water is then closed by the scheme's own contract (sum dp*dq_r = -sum dp*(dq_v+dq_c)), i.e.
the water removed from vapour+cloud equals the surface rain, identical to the pipeline's bookkeeping; energy is
neutral because the latent heat is already in dT_dt_conv. No new clip, no new sign, no re-evaporation invented.
C3 (scope): the refusal at L1785-1794 is replaced by that branch; the `state.tracers is None` guard stays; the
bechtold/tiedtke q_r/q_c routes stay byte-identical (dq_r is a >=0 per-layer source for them, the trait is False).
Not publishing `conv_precip` on spectral (nothing reads it there) — flag if you think that is wrong.
C4 (test): a spectral-bridge test with a spied conv fn returning a signed dq_r profile asserts (a) no raise,
(b) q_v/q_c tendencies equal the spied dq_v/dq_c exactly, (c) no q_r key / no dq_r contribution to q_c, and
(d) shown to FAIL (raise) with the change reverted.

## Questions for you
1. Is C1 the right route, or does the spectral lane need a real surface-precip channel first (and why)?
2. Any hidden choice here (a default, a fallback, a clip) the owner should be asked about explicitly?
3. What must be MEASURED rather than argued? (e.g. column-water closure of the ZM kernel itself on this lane)
4. Anything in the WB spectral lane that consumes rain mass downstream and would now see less of it?
Answer with a verdict line `SHIP` / `HOLD` first, then numbered findings, each with file:line evidence.
