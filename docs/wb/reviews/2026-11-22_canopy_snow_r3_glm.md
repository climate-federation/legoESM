**Findings**

**1. MEDIUM — `interception.py:~365` (`canopy_snow_step`, falloff):** `positive_part_rel(s1 - cap, cap, cfg.relative_width)` uses `cap` itself as the smoothing scale. When the canopy is buried (`vai = 0` → `cap = 0`) while holding snow — the exact transition the port deliberately supports ("a buried canopy still unloads") — the falloff term sees scale 0 with a positive argument (`s1 = snocan > 0`, cold weight = 1). Unless `positive_part_rel` hardens at zero scale, this is a 0/0 (NaN) / gradient blow-up at the burial transition. No test exercises a snow-loaded canopy being buried (the land tests use 60 mm SWE against hbot = 8.5 m). Same scale-zero question for `canopy_snow_fraction`'s `min_rel(power, 1.0, rw)` is safe, but the capacity-relative caps are not verified at `cap → 0⁺`.

**2. LOW — `multilayer_land.py:~1588-1590`:** `_snow_reserved` is added to **both** `evap_transp` and `soil_evap`, and phase 2 later subtracts `_wet_evap` (snow **and** liquid) only from `evap_transp`. If `soil_evap` downstream feeds the reported total ET stream *and* `evap_transp` is its root-drawn share, this is consistent; but the liquid interception branch (`_do_intercept`) shows no matching `soil_evap +=` in the visible diff, so the accounting paths diverge between the two stores. The water-closure test closes the *column* total and would not catch a within-LE misallocation between the reported bare/transpiration components.

**3. LOW — `multilayer_land.py:~1780` vs `~2265`:** the L_s−L_v fusion correction to `lhflx_actual`/`shflx` and the `canopy_snow_sublimation` diagnostic are computed before `_apply_hold_or_fallback`. The diagnostic is re-zeroed only on `_held_mask`; for a **fallback** column (`_fb_mask`) the state is reverted/re-run by the fallback path, yet `_E_canopy_snow` — and the fusion shift already folded into shflx/lhflx from the failed primary solve — is not explicitly reverted, and no test covers a fallback column under canopy snow (only `test_held_column_...`, NaN forcing → held).

**4. LOW — VAI inconsistency (`multilayer_land.py:~795` vs `~530`):** interception/unloading/sublimation use `_canopy_exposed_vai` built from prognostic `compute_prognostic_lai`, while the albedo's `fcansno` (inside `_canopy_masked_band`) uses the `lai`/`LAI_override` of the masking path. If those ever differ, the store whitens a canopy area different from the one that intercepted the snow. Today they coincide in tests; nothing pins it.

**5. INFO — `test_canopy_snow.py _forcing`:** `rho_lowest` uses 1e5 Pa while `p_lowest = 95000` — inconsistent but physically inert for these tests.

Finding 1 is the only one I would block on; 2-4 are wiring/robustness debt.

**Verdict: NOT CLEAN** (pending verification that `positive_part_rel` is safe at scale 0 — if confirmed hardened, findings 2-4 alone are minor and I would downgrade to CLEAN-with-notes).
