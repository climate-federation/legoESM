## Findings

1. **MAJOR — multilayer_land.py ~1534 (`max_soil_evap` loosening)**. The evaporation cap is loosened by `min(LE_wet_canopy, snocan_mid/dt)` — the *potential* snow supply — but the actual snow draw is decided later by `split_wet_evap` and depends on the leaf-phase split. With warm leaves (`_cold ≈ 0`) and a full liquid store, `_from_snow = 0` yet the cap was still loosened by up to the whole snow store: `soil_evap` can then exceed `extractable + infil + melt`, creating water from nothing (or driving soil moisture negative) — exactly what the cap exists to prevent. The loosening must be tied to the realised `_from_snow` (e.g., recompute/iterate, or move the cap after the split). The water-closure test never exercises a warm-leaf + liquid-present + snow-present case, so it cannot catch this.

2. **MAJOR — tests/land/unit/test_canopy_snow.py `test_canopy_snow_off_is_bit_identical_to_no_store`**. The name promises bit-identity but the body only asserts `W_canopy_snow is None` after running with `on=False`. There is no comparison against a config that never heard of canopy snow; the test cannot fail in any meaningful way. Fix: run both `canopy_snow=None` and a masker-only config and compare states/fluxes bitwise.

3. **MAJOR — oracle `_valid` filter + `ok.sum() > 0.8 * ok.size`**. Cases where the demand exceeds the stores (the over-draw regime, where CLM goes negative and the port refuses) are dropped by a *post hoc* mask, and the only guard is that ≥80% survive — a soft assertion that can pass while quietly excluding the hardest cases (large `fe=0.9`, small stores). Either clamp the fixture demand to the stores at generation, or assert the *specific* excluded set and count in the recorded metadata.

4. **MINOR — no oracle for the albedo leg (SurfaceAlbedoMod 1319-1333)**. The fcansno whitening is validated only by the smoke test `a_on > a_off + 0.02`; the two-stream snow-optics call (`omega_snow`, `fcansno` in `clm5_two_stream_albedo`) is not diffed against CLM at any tolerance. At minimum a unit check that `_ts(elai, esai, ground, fcansno=1)` equals the CLM snow-branch value.

5. **MINOR — audit remaining `precip_snow_eff` reads**. The diff changes `fresh_snow_mass` and `update_snow` to `snow_to_ground`, but any *other* read of `precip_snow_eff` in `multilayer_land.py` (balance checks, snow-age reset, diagnostics — not visible in the diff) would still see forcing snowfall and silently break the "ground snow uses post-interception snowfall" vote. Needs an explicit grep/test; the water-closure test only covers the total column, not per-path reads.

6. **MINOR — verify `surface_mass_flux` includes the canopy-snow sublimation.** The diff attaches `canopy_snow_sublimation` to `surface_out` and adjusts lhflx/shflx, but the response `r.surface_mass_flux` picking it up is not shown. The vote ("store sublimation in surface_mass_flux") must hold at the export end as well as the coupler's. Add an assertion `r.surface_mass_flux` vs soil+canopy parts in the sublimation test (currently it infers it from `lhflx - E·L_v`).

7. **MINOR — fwet source asymmetry.** With canopy snow on and interception on, `_fwet_pre` uses the *pre-step* `W_canopy` (plus `_snocan_mid`), while the liquid-only branch's fwet path is replaced wholesale (`elif`). If the liquid branch previously included this step's rain interception, the two paths disagree; the fixture only covers rain=0. Record it alongside the other known differences or align.

8. **MINOR — `base ** 0.15` gradient.** The double-`where` avoids NaN but the derivative still scales as `s^(-0.85)`; finite yet unboundedly large near zero store. `check_grads` with `eps=1e-6` at `s=15` won't probe it. Acceptable (documented), but note it as a tuning hazard rather than "finite gradient".

9. **T_leaf choice (asked)**: the fSun-weighted `Tf_Sun/Tf_Sh` mean is a sane differentiable stand-in for CLM's single `t_veg`; it biases the phase split toward whichever leaf dominates and can misclassify when sunlit/shaded straddle freezing. Acceptable *if recorded* as a known difference alongside the smooth width — I don't see it in the fixture metadata or a KNOWN DIFFERENCES list; add it.

10. **Falloff-without-precip (asked)**: acceptable. Mass is conserved either way (excess goes to ground snow), the difference is only *when*, and it is documented in the docstring. Recorded adequately.

11. **MINOR — `test_canopy_sublimation...` last assertion** compares `r.shflx + r.lhflx` to `so.shflx + so.lhflx`; if both come from the same post-adjustment source this is trivially true and tests nothing about energy conservation. Assert `Rn − G = SH + LE` from independent components instead.

## Verdict

**NOT CLEAN** — findings 1–3 must be addressed (a conditional conservation leak and two tests that cannot fail), plus confirm 5–6.
