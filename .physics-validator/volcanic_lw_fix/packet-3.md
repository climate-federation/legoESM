# Adversarial review request — Volcanic LW aerosol fix, ROUND 3 (LegoESM)

Independent adversarial physics reviewer, round 3. In round 2 you confirmed the
runtime physics has no remaining sign/unit/conservation/monotonicity/AD defect,
and left 3 test/doc items. Each is now addressed (below). Please confirm they
are resolved and state whether any substantive finding remains. cd = repo root;
diff at `.physics-validator/volcanic_lw_fix/fix.diff` (pasted at end).

## Round-2 residuals and resolutions

1. **B2 log-pressure choice not regression-tested (FIXED).** Added
   `TestStratosphericPlacement.test_log_pressure_within_layer_split`: one source
   layer `[1000, 2000]` Pa, unit AOD, a model interface at the log-p midpoint
   `sqrt(1000*2000)` → asserts the split is `[0.5, 0.5]` (atol 1e-6). A reverted
   linear-pressure CDF gives `[0.4142, 0.5858]` and fails. This is exactly your
   suggested discriminator.

2. **Source→OLR regression only compositional (FIXED).** Added
   `TestRealFileEndToEnd.test_loader_to_solver_lowers_olr`: the full chain
   file → `get_aerosol_lw_at_time` → `place_stratospheric_aod_profile_to_layers`
   → RRTMGP `solve_columns`, asserting TOA OLR drops (correct warming sign) by a
   small but resolved amount for the real 1979 profile. (A shared lru_cached
   solver + tropical column keeps the added table-load cost to one.)

3. **B3 overclaim in `_planck_band_weights` + stale "<1%" comment (FIXED).**
   `_planck_band_weights` docstring now calls it a physically-motivated HEURISTIC
   (better than a flat mean, NOT the exact gray equivalent). The
   `_USSA1976_LAYERS` comment now states the pressure effect is ~2% at 30 km /
   ~3% at 40 km and that the geopotential conversion IS applied.

## Test status (CPU, x64): all 18 pass
`tests/unit/test_volcanic_lw_aerosol.py` — 18 passed (82 s), including the new
log-p discriminator and the loader→remap→solver end-to-end OLR regression.
Collateral unchanged (convection 6/6, bechtold 3/3).

## Ask
Confirm the three round-2 items are resolved and that no substantive physics,
unit, sign, conservation, gradient, or test-adequacy finding remains in the
LW-aerosol diff. If you still see one, cite file:line. (SW volcanic vertical
mixing and lat-dependent placement remain documented residuals per rounds 1-2;
the Bechtold diff is a separate landed fix outside this change.)

## Full unified diff
See `.physics-validator/volcanic_lw_fix/fix.diff`.
