# Adversarial review — Volcanic LW aerosol fix, ROUND 4 (LegoESM)

Independent adversarial reviewer, round 4. Round 3 you found "no substantive
runtime physics, unit, sign, conservation, monotonicity, or AD defect remains"
and left two items. Both now addressed:

1. **B3 loader Planck use not regression-protected (FIXED).** Added
   `TestLoaderAbsorptionScaling.test_loader_uses_planck_not_flat_band_mean`: a
   band-VARYING synthetic file (only far-IR band 0 carries OD) whose loaded gray
   column must equal the Planck-weighted collapse `W[0]*OD/sum(W)*thickness` and
   is asserted to differ from the flat band mean by >20%. A Planck→flat swap in
   the loader now fails. The `_make_synth_volc_lw` fixture accepts a per-band
   extinction array.
2. **"strictly better" wording (FIXED).** Both occurrences softened to
   "better-motivated than a flat band mean."

All 19 tests pass (CPU, x64, 83 s). Collateral unchanged (convection 6/6,
bechtold 3/3).

Please confirm no substantive finding remains in the LW-aerosol diff, or cite
file:line. Documented residuals (SW volcanic vertical mixing; lat-independent
climatological placement) and the separate Bechtold diff are out of scope.
Diff: `.physics-validator/volcanic_lw_fix/fix.diff` (pasted at end).
