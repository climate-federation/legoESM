1. **BLOCKER — `packages/land/legoesm/land/restart.py:552`: restart can silently destroy canopy snow.** When the loaded restart carries `W_canopy_snow` but the destination disables it, the merge discards the store. Reproduced: 5 kg m⁻² disappears; ground snow remains zero. **Fix:** reject this transition or explicitly transfer the mass to ground snow.

2. **MAJOR — `packages/coupler/legoesm/driver/model_driver.py:1793`: coupled checkpoint compatibility is missing.** `_LAND_ML_INIT_ON_RESTORE` contains only `T_snow`. Enabling canopy snow with an older coupled checkpoint therefore fails the field-set check at line 3305, contrary to the agreed empty-store behavior. **Fix:** permit missing `W_canopy_snow`, initialize it empty, and test the coupled checkpoint path.

3. **MAJOR — `packages/land/legoesm/land/multilayer_land.py:1539`: snow supply enters a shared evaporation cap.** The subsequent partition distributes that supply between canopy and bare-soil evaporation before the snow-store draw. For equal canopy and soil demands, zero soil supply, and sufficient canopy snow, half the allowance goes toward bare-soil demand; canopy sublimation receives only half. **Fix:** reserve store-supported wet evaporation first, limit the remaining soil demands separately, then recombine. Test dry soil with simultaneous wet-canopy and ground demand.

4. **MAJOR — `packages/coupler/legoesm/driver/config.py:3072`; `packages/land/legoesm/land/canopy/interception.py:348`: accepted widths produce negative unloading.** Width `3` passes validation, but `min_rel(0, 1, 3)` is negative. Reproduced with no precipitation, cold calm air, and a 5 kg m⁻² store: storage grows to 5.208333 kg m⁻² while ground snowfall becomes negative. **Fix:** enforce the nonnegative operands’ valid relative-width range at both configuration entry points, or use a positivity-preserving smooth cap.

5. **MAJOR — `packages/land/legoesm/land/canopy/interception.py:361`: the default snow-cover fraction is not differentiable at an empty store.** The `where` reports zero derivative exactly at zero, but the unchanged `s**0.15` has an unbounded right derivative. The guard prevents NaNs at zero; it does not smooth the function. **Fix:** regularize the low-store power under the smooth configuration, retain the exact hard form at width zero, and test zero/near-zero gradients.

6. **MINOR — `packages/land/legoesm/land/canopy/interception.py:342`; `scripts/validate/oracle/clm5_canopy_snow/make_fixture.py:36`: precipitation-free falloff escapes oracle coverage.** CLM gates capacity shedding on precipitation; this helper sheds regardless, including at widths zero. Fixtures always prescribe positive snowfall. The extension conserves water and is defensible, but the helper’s `KNOWN DIFFERENCES` omits it. **Fix:** document the hard-limit exception and add dry, overcapacity cases, or reproduce CLM’s gate.

7. **MINOR — `packages/land/legoesm/land/multilayer_land.py:2255`: held columns retain a false sublimation diagnostic.** The store and exchanged flux are reverted, then `canopy_snow_sublimation` receives the unreverted value. **Fix:** zero this diagnostic under `_held_mask`; test a planted held column.

8. **MINOR — `tests/land/unit/test_canopy_snow.py:254`: the bit-identity test makes no comparison.** It only checks that the store is `None`; arbitrary changes to off-path states and fluxes would pass. **Fix:** compare states and responses against the baseline behavior.

The `fSun`-weighted leaf temperature is a sane bulk surrogate for CLM’s single vegetation temperature, with an approximation when leaf temperatures straddle freezing.

All 15 canopy-snow tests passed after bypassing the excluded import error in memory. No files edited.

**Verdict: NOT CLEAN.**
