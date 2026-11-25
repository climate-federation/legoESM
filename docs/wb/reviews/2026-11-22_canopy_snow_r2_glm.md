## VOTES

V1 [A] — refusing is the conservative default; silently dropping real water onto the ground changes an unrelated run's snowpack on restart.
V2 [A] — the non-negativity proof is exactly max ≤ 3·min, i.e. width ≤ 1; [B] would forbid legal configurations for no safety gain.
V3 [A] — 1e-2 of snocanmx makes fcansno ≈ (1e-2)^0.15 ≈ 0.5 at one-hundredth of capacity, a large albedo departure from CLM5 near empty stores; 1e-3 keeps the gap ~2.5e-3-scale while fixing the infinite slope. [A].
V4 [A] — empty store is the no-mass case; refusing would block branch-continuation restarts for zero benefit (a carrying checkpoint still refuses).
V5 [A] — reserving before the cap is the only ordering that guarantees the snow-funded share is never blocked by a frozen/dry root zone; loosening the cap by the potential supply (B) lets the liquid share leak past the cap too.

## FINDINGS

1. MINOR — `interception.py` `split_wet_evap` (~:415-425): the chained smooth mins feed *derived* operands back in: `liq_avail - l_cold` and `demand_warm - l_warm`. If `min_rel` can overshoot its smaller operand (not just undershoot, which the round-1 proof bounds), `l_cold > liq_avail` makes the next operand negative, and `min_rel(nonneg, negative)` then plausibly returns a **negative** draw → `W_canopy_new = _W_int - _from_liq` grows the liquid store with no input (mass creation, O(relative_width · store)). The width validation proves non-negativity of each min in isolation, not of the chained residuals. Fix: `jnp.maximum(liq_avail - l_cold, 0.0)` on both residual operands (CLM's order is preserved; the clamp is active only inside the smoothing band). Please verify `min_rel`'s overshoot direction; if it is the `0.5(a+b−smooth|a−b|)` form with smooth-abs ≥ |·| it cannot overshoot and this downgrades to INFO.

2. MINOR — `multilayer_land.py` ~:948: `_fwet_pre` uses `state.W_canopy` (pre-rain-interception liquid) while the phase-2 draw uses `_W_int` (post-interception). When liquid interception is on with rain, the wetted fraction and the store the demand draws from are different snapshots of the same liquid store. Small fidelity wobble, no conservation break (budget is on `_W_int`). Use `_W_int` in `_fwet_pre` too (or `_liq_i` in the draw).

3. INFO — `canopy_snow_step`: `frac = min_rel(k·dt, 1, w)` can exceed `k·dt` if `min_rel` overshoots, i.e. unload marginally faster than CLM's rate; mass-conserving and capped by the store, fidelity-only.

4. INFO — no land-step test that an over-draw (demand above both stores, the CLM-negative case) empties both stores *in the column* (it is asserted only at unit/oracle level). The `evap_transp` cap `_dem = min_rel(_dem_wet, evap_transp·dt)` path with a tiny transpiration stream is also untested; a regression there would silently under-sublimate.

5. CLEAN — wiring: CLI → ExperimentConfig (with the inert-width and range refusals, incl. the 1e-30 floor) → `build_multilayer_land_tile` → `MultiLayerLandConfig.canopy_snow`; state allocation/restart/restore/merge refusal paths all tested; held-column diagnostic zeroing correct; the SH/LE fusion correction conserves `Rn − G`; the bit-identity off-test now covers every response field and state field; delta-form albedo pass-through is exactly the two-stream's fcansno term.

## VERDICT

**NOT CLEAN** — finding 1 (potential negative liquid draw → store growth via chained smooth mins) needs either a clamp or a one-line proof that `min_rel` never exceeds its smaller operand; finding 2 is a trivial consistency fix. Both are small, localized, and do not disturb the round-1 design decisions, which I endorse in full (votes above).
