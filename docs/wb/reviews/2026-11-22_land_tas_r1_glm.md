**Findings (ranked)**

1. `two_leaf_canopy.py:733` — `screen_level_temperature(Ta, Tc_cvg, z_ref - displa, fluxes_per_col["zeta"], z0m)`: zldis and z0m(=z0h) match CLM5 (`zldis = zrefw - displa`, CLM uses z0m for heat), and dth/thm frame matches `t_ref2m = thm + temp1*dth*(1/temp12m − 1/temp1)`. Good. **But** `obu = zldis / zeta` reconstructs the Obukhov length from the *capped/smoothed* zeta (module caps stable zeta at `_ZETA_MAX_STABLE` with a smooth cap), so in strongly stable columns the diagnostic uses a different obu than the flux solve. Prefer passing obu directly from the solve. ~sub-0.1 K effect; diagnostic only.
2. `model_driver.py:14719-14752` — the non-`_land_none` return and the packed-scatter tuple both append `_t2m` after `_emis`, and the consumer at `model_driver.py:16318` unpacks it in the same slot. Consistent. **Not visible in the diff**: the `_land_none` / `_land_step_none` producer and the multi-rank halo block — if any of those still produce/slice 10 fields, the 11-slot unpack breaks at runtime. The `_held_f > 0.5 → NaN` and `_land_T2m_last` cadence (set only when a land step ran) are coherent with the V2 fallback.
3. `_LAND_STEP_N_CELL = 11` — if the checkpoint land-window payload is sized from this constant, old restarts have a layout mismatch; needs a guard or a documented version bump (ties to V3).
4. Tests: oracle is a genuinely independent scalar transcription (Paulson/BH91), covers unstable/stable/neutral, monotonicity between canopy air and reference, exact replacement-and-fallback in `_tas_2m` (fails pre-fix), and gradients are finite/non-zero. Non-vacuous. Not covered: held-column NaN export and halo/restart fallbacks — worth one test.
5. No state/flux/coupling mutation anywhere; `T_2m` is export-only in `SurfaceFluxOutput`. Differentiability preserved (jnp.where on isfinite keeps gradients through finite entries).

**Verdict: CLEAN** (subject to items 2/3 being confirmed in the unshown `_land_none`/halo/checkpoint paths; nothing in the diff itself is wrong).

**Votes**
- **V1: A** — CLM's `thm` already carries the lapse correction (thm = th + g/cp·z_forc), so verbatim `t_ref2m` is already a temperature; an extra −0.02 K would double-correct.
- **V2: A** — keeps continuity with the pre-fix behaviour and the sea legs; a jump to lowest-level T is a larger, discontinuous change for held/halo columns.
- **V3: A** — it is a diagnostic; first-interval/halo fallback is bounded by the profile error already accepted. Add the payload field only if item 3 forces a format bump anyway.
